"""学习调度与审阅同源服务；独立有界 executor，授权配置只保存在进程内。"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..agent import collab_rules
from ..agent.agent_exec import _endpoint_fp_from_config
from ..agent.learning_loop import run_learning_loop
from ..llm.llm_client import should_use_llm
from .conversation_runs import RunError
from .learning_store import LearningStore

_SERVICES = {}
_SERVICES_LOCK = threading.Lock()


def service_for(conversations, workspace):
    key = (str(conversations.path), str(conversations.project_root))
    with _SERVICES_LOCK:
        if key not in _SERVICES:
            _SERVICES[key] = LearningService(conversations, workspace)
        return _SERVICES[key]


class _KnowledgeBinding:
    def __init__(self, service, owner, endpoint_fp, project_id='', *, request_namespace=''):
        self.service = service
        self.owner, self.endpoint_fp, self.project_id = owner, endpoint_fp, project_id
        self.request_namespace = request_namespace

    def entries(self, **kwargs):
        return self.service.entries(self.owner, self.endpoint_fp, self.project_id, **kwargs)

    def snapshot(self):
        return self.service.snapshot(self.owner, self.endpoint_fp, self.project_id)

    def category_definitions(self):
        return self.service.status(self.owner, self.endpoint_fp, self.project_id)['category_definitions']

    def propose(self, operations, *, source_refs=(), request_id='', actor='learning', _check=None, category_definitions=()):
        if _check is not None:
            _check()
        refs = sorted(source_refs, key=lambda r: r['id'])
        fingerprint = [self.request_namespace, operations, refs]
        if category_definitions:
            fingerprint.append(category_definitions)
        rid = request_id or 'lp_' + collab_rules.content_fingerprint(fingerprint)[:24]
        return self.service.propose(self.owner, self.endpoint_fp, self.project_id, request_id=rid,
            operations=operations, source_refs=list(refs), actor=actor, expected_identity=self.owner, check=_check,
            category_definitions=category_definitions)

    def assess(self, change_id, **kwargs):
        return self.service.assess(self.owner, self.endpoint_fp, self.project_id, change_id=change_id, **kwargs)

    def change(self, change_id):
        return self.service.change(self.owner, self.endpoint_fp, change_id)


class LearningService:
    def __init__(self, conversations, workspace, *, loop=run_learning_loop, workers=1, capacity=8,
                 knowledge=None, available_tools=None):
        self.store = LearningStore(conversations)
        self.workspace, self.root, self.loop = workspace, conversations.project_root, loop
        self.workers, self.capacity = workers, capacity
        self._knowledge = knowledge
        from ..agent.knowledge import registered_tools
        self.available_tools = tuple(registered_tools() if available_tools is None else available_tools)
        self._lock = threading.RLock()
        self._configs, self._tasks, self._pending = {}, {}, {}
        self._manual_configs = set()
        self._pool = None
        self._closing = False

    @property
    def knowledge(self):
        if self._knowledge is None:
            from .knowledge_service import KnowledgeService
            self._knowledge = KnowledgeService(self.store.conversations, self.workspace)
        return self._knowledge

    def _knowledge_binding(self, owner, endpoint_fp, project_id='', *, request_namespace=''):
        return _KnowledgeBinding(self.knowledge, owner, endpoint_fp, project_id, request_namespace=request_namespace)

    def checked_project(self, owner, project_id, *, writable=False):
        from ..agent.knowledge import identifier
        project_id = identifier(project_id or '', empty=True)
        if project_id:
            project = next((p for p in self.workspace.list_projects(owner, include_archived=True) if p['id'] == project_id), None)
            if project is None:
                raise RunError('project_not_found', '没有这个学习项目。')
            if writable and project['archived']:
                raise RunError('learning_project_archived', '归档项目不能开启或写入学习。')
        return project_id

    @staticmethod
    def check_identity(owner, expected_identity):
        from ..agent.learning_scope import canonical_principal
        if expected_identity is not None and canonical_principal(expected_identity) != canonical_principal(owner):
            raise RunError('learning_identity_conflict', '账户已切换，请在当前账户重新操作。')

    def authorize(self, owner, cfg, quota, *, project_id='', schedule=False, manual=False,
                  expected_config_epoch=None, expected_scope_epoch=None):
        endpoint = _endpoint_fp_from_config(cfg)
        key = (owner, endpoint, project_id)
        scope, config_epoch = self.store.authorization(*key, expected_config_epoch=expected_config_epoch,
            expected_scope_epoch=expected_scope_epoch)
        if str(cfg.provider).strip().lower() == 'trial' or not should_use_llm(cfg)[0]:
            self.release(owner, endpoint, expected_config_epoch=config_epoch)
            return endpoint
        with self._lock, self.store.conversations.database(write=True) as conn:
            current, _ = self.store.authorization(*key, expected_config_epoch=config_epoch,
                expected_scope_epoch=scope['epoch'], conn=conn)
            if current['enabled'] or manual:
                self._configs[key] = (replace(cfg), quota, time.monotonic() + 900, current['epoch'], config_epoch)
                if current['enabled']:
                    self._manual_configs.discard(key)
                else:
                    self._manual_configs.add(key)
        if schedule:
            self.notify(owner, endpoint, project_id)
        return endpoint

    def _config(self, key):
        with self._lock:
            record = self._configs.get(key)
        valid = False
        if record and record[2] > time.monotonic():
            scope, epoch = self.store.authorization(*key)
            if scope['epoch'] == record[3] and epoch == record[4]:
                valid = True
                if scope['enabled']:
                    return record
        if not valid:
            with self._lock:
                if self._configs.get(key) is record:
                    self._configs.pop(key, None)
                    self._manual_configs.discard(key)
        return None

    def release(self, owner, endpoint_fp, *, expected_identity=None, expected_config_epoch=None):
        self.check_identity(owner, expected_identity)
        with self._lock:
            epoch = self.store.release_configs(owner, endpoint_fp, expected_config_epoch=expected_config_epoch)
            for key in list(self._configs):
                if key[:2] == (owner, endpoint_fp):
                    self._configs.pop(key, None)
        return {'ok': True, 'released': True, 'configured': False, 'endpoint_fp': endpoint_fp,
                'config_epoch': epoch, 'detail': '旧AI配置授权已释放；长期学习开关、规则和已有候选保留。'}

    def control(self, owner, cfg, *, project_id='', enabled, threshold=3, quota=None,
                expected_identity=None, expected_config_epoch=None, expected_scope_epoch=None):
        self.check_identity(owner, expected_identity)
        endpoint = _endpoint_fp_from_config(cfg)
        scope, config_epoch = self.store.authorization(owner, endpoint, project_id,
            expected_config_epoch=expected_config_epoch, expected_scope_epoch=expected_scope_epoch)
        project_id = self.checked_project(owner, project_id)
        key = (owner, endpoint, project_id)
        with self._lock:
            current = self.store.control(*key, enabled=enabled, threshold=threshold,
                expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'])
            self._configs.pop(key, None)
        if enabled:
            self.authorize(owner, cfg, quota, project_id=project_id, expected_config_epoch=config_epoch,
                expected_scope_epoch=current['epoch'])
            self.notify(owner, endpoint, project_id)
        return {'ok': True, **self.status(owner, endpoint, project_id)}

    def _check_authorization(self, job, config, *, conn=None):
        key = (job['owner'], job['endpoint_fp'], job['project_id'])
        if self.store.config_epoch(key[0], key[1], conn=conn) != config[4]:
            raise RunError('learning_config_released', '旧AI配置授权已释放，迟到结果不会保存。')
        if config[3] != job['epoch']:
            raise RunError('learning_paused', '学习范围授权已变化，旧配置不能执行新任务。')
        self.store.check(job, conn=conn)
        with self._lock:
            current = self._configs.get(key)
            if not current or current[3:] != config[3:] or min(config[2], current[2]) <= time.monotonic():
                raise RunError('learning_config_expired', '授权配置已过期，请在同一账户和端点恢复学习。')

    def notify(self, owner, endpoint_fp, project_id='', *, force=False):
        key = (owner, endpoint_fp, project_id)
        config = self._config(key)
        self.store.status(*key)
        try:
            job = self.store.schedule(*key, force=force, configured=False,
                expected_config_epoch=config[4] if config else None, expected_scope_epoch=config[3] if config else None)
        except RunError as exc:
            if exc.code == 'learning_config_conflict':
                return None
            raise
        if job and config:
            self._enqueue(job, config)
        return job

    def _enqueue(self, job, config):
        from ..corpus.fs_utils import acquire_file_lock, release_file_lock, FileLockBusy
        key = (job['owner'], job['endpoint_fp'], job['project_id'])
        with self._lock:
            if self._closing:
                return
            try:
                self._check_authorization(job, config)
            except RunError as exc:
                self.store.set_status(job, 'paused', {'ok': False, 'error_code': exc.code})
                if self._pending.get(key) == (job, config):
                    self._pending.pop(key, None)
                return
            task_key = (job['owner'], job['id'])
            if task_key in self._tasks:
                self._pending[key] = (job, config)
                return
            self._pending[key] = (job, config)
            if len(self._tasks) >= self.capacity or any(k[0] == job['owner'] for k in self._tasks):
                return
            try:
                handle = acquire_file_lock(self.store.lock_path(job), wait=False)
            except FileLockBusy:
                return
            try:
                self.store.set_status(job, 'queued')
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='memory-learning')
                self._tasks[task_key] = self._pool.submit(self._execute, job, config, handle)
                self._pending.pop(key, None)
            except Exception:
                release_file_lock(handle)
                self._tasks.pop(task_key, None)
                self.store.set_status(job, 'waiting_config', {'error_code': 'learning_enqueue_failed'})

    def _resume_pending(self):
        with self._lock:
            for job, config in list(self._pending.values()):
                try:
                    self._enqueue(job, config)
                except Exception:
                    continue

    def _execute(self, job, config, handle):
        from ..corpus.fs_utils import release_file_lock, FileLockBusy
        from ..agent.learning_scope import bank_transaction
        key = (job['owner'], job['endpoint_fp'], job['project_id'])
        try:
            self._check_authorization(job, config)
            self.store.set_status(job, 'running')
            cfg, quota = config[:2]
            started = time.monotonic()
            def check(*, conn=None):
                self._check_authorization(job, config, conn=conn)
                if time.monotonic() - started > 240:
                    raise RunError('learning_time_limit', '本次学习达到独立时间上限。')
            def charge(effective):
                check()
                self.store.charge(job['owner'], quota=lambda: quota(effective) if quota else None)
            def quota_only(effective):
                check()
                admitted = quota(effective) if quota else None
                check()
                return admitted
            charge.quota_only = quota_only
            source_ids = set(json.loads(job['sources_json']))
            fetched = {}
            def read(**kwargs):
                offset, limit = kwargs.get('offset', 0), kwargs.get('limit', 12)
                if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12:
                    raise RunError('invalid_learning_page', '来源读取分页无效。')
                rows = self.store.sources_by_ids(key[0], key[1], source_ids)
                page = [dict(r, principal=r['owner']) for r in rows[offset:offset + limit]]
                from ..agent.agent_exec import _example_args_zh
                with self.store.conversations.database() as conn:
                    for row in page:
                        row.update(self.store.source_snapshot(conn, row))
                        run = conn.execute('SELECT status,result_json FROM conversation_runs WHERE owner=? AND id=?',
                            (row['owner'], row['origin_run_id'])).fetchone() if row['origin_run_id'] else None
                        if run and run['status'] == 'done' and run['result_json']:
                            actual = json.loads(run['result_json'])
                            plan = actual.get('plan') if type(actual.get('plan')) is dict else {}
                            steps = actual.get('steps') or plan.get('steps')
                            if type(steps) is list and steps and not actual.get('cancelled') and not actual.get('result_truncated') and all(type(s) is dict and s.get('ok') is True for s in steps):
                                row['verified_steps'] = [{'verb': s['verb'], 'args': _example_args_zh(s['verb'], s.get('slots') or {})} for s in steps]
                fetched.update({r['id']: r for r in page})
                return page
            binding = self._knowledge_binding(*key, request_namespace=job['id'])
            def memories():
                result = binding.snapshot()
                return [dict(r, principal=key[0], endpoint_fp=key[1], project_id=str(r.get('project_id') or '')) for r in result['items']]
            def save(item, **kwargs):
                check()
                sources = kwargs.get('sources') or []
                if not {s['id'] for s in sources} <= set(fetched):
                    raise RunError('learning_source_invalidated', '候选出处不是本任务实读事件。')
                value = {name: item[name] for name in ('kind', 'title', 'text', 'trigger', 'tier', 'permanence', 'payload', 'links') if name in item}
                value.setdefault('kind', 'memory')
                value.setdefault('permanence', kwargs.get('permanence', 'unknown'))
                receipt = binding.propose([{'op': 'create', 'after': value}],
                    source_refs=[{'type': 'learning', 'id': s['id']} for s in sources], _check=check)
                if not receipt['ok']:
                    raise RunError(receipt['error_code'], receipt['detail'])
                check()
                return {'ok': True, 'candidate': {'id': receipt['change_id'], 'status': receipt['stage'], **receipt}}
            def verify(cid):
                return binding.change(cid)
            result = self.loop(cfg=cfg, read_evidence=read, query_memory=memories,
                save_candidate=save, verify_candidate=verify, before_llm=charge, check=check,
                on_trace=lambda entry: self.store.trace(job, entry), knowledge=binding, available_tools=self.available_tools)
            check()
            status = 'done' if result.get('ok') is True else 'waiting_config' if result.get('error_code') == 'learning_config_expired' else 'paused' if result.get('state') == 'paused' else 'failed'
            self.store.set_status(job, status, result)
        except FileLockBusy:
            pass
        except RunError as exc:
            self.store.set_status(job, 'waiting_config' if exc.code == 'learning_config_expired' else 'paused' if exc.code in {'learning_paused', 'learning_source_invalidated', 'learning_config_released'} else 'failed', {'ok': False, 'error_code': exc.code})
        except Exception:
            self.store.set_status(job, 'failed', {'ok': False, 'error_code': 'learning_internal_error'})
        finally:
            if handle is not None:
                release_file_lock(handle)
            with self._lock:
                pending = self._pending.get(key)
                same_attempt = pending is None or (pending[0]['id'] == job['id'] and pending[0].get('attempt', 0) == job.get('attempt', 0))
                if key in self._manual_configs and same_attempt:
                    self._configs.pop(key, None)
                    self._manual_configs.discard(key)
                self._tasks.pop((job['owner'], job['id']), None)
                self._resume_pending()

    def status(self, owner, endpoint_fp, project_id='', **kwargs):
        self.checked_project(owner, project_id)
        result = self.store.status(owner, endpoint_fp, project_id, **kwargs)
        result['configured'] = self._config((owner, endpoint_fp, project_id)) is not None
        result['config_epoch'] = self.store.config_epoch(owner, endpoint_fp)
        result['web_path'] = '/?panel=memory'
        return result

    def sources(self, owner, endpoint_fp, project_id=None, **kwargs):
        if project_id is not None:
            project_id = self.checked_project(owner, project_id)
        return {'ok': True, **self.store.sources(owner, endpoint_fp, project_id, **kwargs)}

    def trigger(self, owner, cfg, *, project_id='', request_id='', feedback='', conversation_id='', quota=None,
                expected_identity=None, expected_config_epoch=None, expected_scope_epoch=None,
                selection=None, source_ids=None, local_materials=None, manual=False,
                allow_cross_project=False, expected_source_scope_epochs=None):
        from ..agent.knowledge import identifier
        self.check_identity(owner, expected_identity)
        endpoint = _endpoint_fp_from_config(cfg)
        scope, config_epoch = self.store.authorization(owner, endpoint, project_id,
            expected_config_epoch=expected_config_epoch, expected_scope_epoch=expected_scope_epoch)
        if self._closing:
            raise RunError('learning_service_closed', '学习执行器已关闭，未接受新选择。')
        project_id = self.checked_project(owner, project_id, writable=True)
        if type(manual) is not bool or type(allow_cross_project) is not bool or (feedback and manual):
            raise RunError('invalid_learning_trigger', '一次性授权与跨项目标记必须是严格布尔值，不能夹带自动反馈。')
        for values, bound in ((selection, 64), (source_ids, 200), (local_materials, 12)):
            if values is not None and (type(values) is not list or len(values) > bound):
                raise RunError('invalid_learning_trigger', '来源选择清单类型或长度无效。')
        explicit = bool(selection or source_ids or local_materials)
        if allow_cross_project and (not manual or project_id or feedback or not explicit):
            raise RunError('invalid_learning_trigger', '跨项目用户层汇总只接受明确选来源的一次性手动授权。')
        if expected_source_scope_epochs is None:
            captured = self.store.authorization_snapshot(owner)
            expected_source_scope_epochs = {project: value for (fp, project), value in captured['scope_epochs'].items() if fp == endpoint}
        if not scope['enabled'] and not manual:
            raise RunError('learning_consent_required', '请开启当前范围自动学习，或明确一次性授权。')
        if manual and not explicit:
            return {'ok': True, 'scheduled': False, 'detail': '没有选定来源，本次未调度。', **self.status(owner, endpoint, project_id)}
        if cfg.provider == 'trial' or not should_use_llm(cfg)[0]:
            raise RunError('learning_model_unavailable', '整理需要开启非试用 AI；未提交的资料与草稿保持不变。')
        if not explicit:
            self.authorize(owner, cfg, quota, project_id=project_id, expected_config_epoch=config_epoch,
                expected_scope_epoch=scope['epoch'])
            if feedback:
                if conversation_id and self.workspace.assignments(owner).get(conversation_id, '') != project_id:
                    raise RunError('learning_scope_conflict', '反馈对话不属于当前学习范围。')
                self.store.feedback(owner, endpoint, project_id, request_id=request_id, text=feedback,
                    conversation_id=conversation_id, expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'])
            job = self.notify(owner, endpoint, project_id, force=True)
            return {'ok': True, 'scheduled': bool(job), 'detail': '' if job else '没有可处理的新来源。',
                **self.status(owner, endpoint, project_id)}
        assignments = self.workspace.assignments(owner)
        resolved = []
        for item in selection or []:
            if type(item) is not dict or set(item) != {'project_id', 'conversation_ids'} or type(item['conversation_ids']) is not list or len(item['conversation_ids']) > 64:
                raise RunError('invalid_learning_trigger', '对话选择需要明确项目与有界编号清单。')
            item_project = identifier(item['project_id'], empty=True)
            for cid in item['conversation_ids']:
                cid = identifier(cid)
                real = assignments.get(cid, '')
                if item_project != real:
                    raise RunError('learning_scope_conflict', '所选项目与对话实际归属不一致。')
                resolved.append((cid, real))
        picked = []
        with self.store.conversations.database(write=True) as conn:
            self.store.authorization(owner, endpoint, project_id, expected_config_epoch=config_epoch,
                expected_scope_epoch=scope['epoch'], conn=conn)
            projects = {project for _, project in resolved}
            rows = []
            if source_ids:
                source_ids = [identifier(sid) for sid in source_ids]
                rows = self.store.sources_by_ids(owner, endpoint, source_ids, conn=conn)
                if set(source_ids) != {r['id'] for r in rows}:
                    raise RunError('learning_source_invalidated', '所选来源不存在、不属于本人端点或已失效。')
                for row in rows:
                    projects.update(self.store.source_snapshot(conn, row)['scope_epochs'])
                picked.extend(r['id'] for r in rows)
            for material in local_materials or []:
                if type(material) is not dict:
                    raise RunError('invalid_learning_material', '本机材料须是明确对象。')
                projects.add(identifier(material.get('project_id', ''), empty=True))
            cross_project = bool(any(projects))
            if not project_id and cross_project and not allow_cross_project:
                raise RunError('learning_scope_conflict', '用户层汇总项目来源须明确manual和allow_cross_project授权。')
            if allow_cross_project and not cross_project:
                raise RunError('invalid_learning_trigger', '所选来源没有项目内容，无需跨项目授权。')
            for project in projects:
                self.checked_project(owner, project, writable=True)
                self.store.authorization(owner, endpoint, project, expected_config_epoch=config_epoch,
                    expected_scope_epoch=expected_source_scope_epochs.get(project, 0), conn=conn)
            if resolved:
                rows = self.store.snapshot_conversations(owner, endpoint, resolved, target_project=project_id,
                    expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'], conn=conn)
                picked.extend(r['id'] for r in rows)
            if local_materials:
                rows = self.store.materialize(owner, endpoint, project_id, local_materials,
                    expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'], conn=conn,
                    allow_cross_project=allow_cross_project)
                picked.extend(r['id'] for r in rows)
        picked = list(dict.fromkeys(picked))
        self.authorize(owner, cfg, quota, project_id=project_id, manual=manual,
            expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'])
        key = (owner, endpoint, project_id)
        with self._lock:
            config = self._configs.get(key)
        job = self.store.schedule(*key, source_ids=picked, manual=manual, request_id=request_id,
            configured=False, expected_config_epoch=config_epoch, expected_scope_epoch=scope['epoch'])
        if job and config and job['status'] in {'queued', 'waiting_config', 'running'}:
            self._enqueue(job, config)
        current = self.store.job_receipt(owner, job['id']) if job else None
        scheduled = current is not None and current['status'] in {'queued', 'waiting_config', 'running'}
        return {'ok': True, **self.status(owner, endpoint, project_id),
            **({key: current[key] for key in ('request_id', 'attempt', 'selected_total', 'selected_count', 'batch_count',
                'read_count', 'handled_count', 'consumed_count', 'remaining_selected_count', 'remaining_source_ids', 'selected_source_ids', 'partial')} if current else {}),
            'job_id': current['id'] if current else None, 'scheduled': scheduled,
            'complete': current is not None and current['status'] == 'done',
            'detail': '这份固定选择已处理完成，本次没有重新调用。' if current and current['status'] == 'done'
                else '' if scheduled else '所选来源未调度，请查看真实任务状态或重新选择。'}

    def candidates(self, owner, endpoint_fp, project_id='', **kwargs):
        self.checked_project(owner, project_id)
        return collab_rules.list_learning_records(self.root, principal=owner, endpoint_fp=endpoint_fp, project_id=project_id, **kwargs)

    def review(self, owner, endpoint_fp, project_id='', *, candidate_id, decision,
               expected_fingerprint, expected_rules_fingerprint='', feedback='', human_confirmed=False):
        self.checked_project(owner, project_id)
        if decision != 'reject' and not human_confirmed:
            return {'ok': False, 'error_code': 'human_confirmation_required', 'web_path': '/?panel=memory',
                    'detail': '请在网页记忆面板逐条确认，MCP 不能代替真人批准或替代规则。'}
        if decision == 'revoke':
            return collab_rules.revoke_learning_rule(self.root, candidate_id, principal=owner,
                endpoint_fp=endpoint_fp, project_id=project_id, expected_fingerprint=expected_fingerprint)
        result = collab_rules.review_learning_candidate(self.root, candidate_id, principal=owner,
            endpoint_fp=endpoint_fp, project_id=project_id, decision=decision,
            expected_fingerprint=expected_fingerprint, expected_rules_fingerprint=expected_rules_fingerprint,
            feedback=feedback, human_confirmed=human_confirmed)
        if result.get('ok') and str(feedback or '').strip():
            self.store.feedback(owner, endpoint_fp, project_id,
                request_id='review:' + collab_rules.content_fingerprint([candidate_id, expected_fingerprint])[:32], text=feedback)
        if feedback:
            self.notify(owner, endpoint_fp, project_id)
        return result

    def close(self):
        with self._lock:
            self._closing = True
            self._configs.clear()
            self._manual_configs.clear()
            self._pending.clear()
            pool, self._pool = self._pool, None
        if pool:
            pool.shutdown(wait=True)
