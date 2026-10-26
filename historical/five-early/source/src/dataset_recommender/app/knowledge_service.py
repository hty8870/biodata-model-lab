"""知识提案唯一道闸；模型在锁外判关系，真实评估与身份/CAS 在短事务内绑定。"""
from __future__ import annotations

import inspect
import json
import uuid
from functools import wraps

from ..agent.knowledge import (KINDS, MAX_ASSESS_CALLS, digest, enforce_snapshot_budget,
    identifier, normalize_body, registered_tools, revision, system_digest, system_entries, text_value)
from ..agent.learning_scope import canonical_principal, principal_variants
from ..agent.memory_relations import _config_ready, assess_relations
from ..llm.llm_client import call_llm
from .conversation_runs import RunError
from .durable_jobs import JobStoreError
from .knowledge_store import KnowledgeStore, encode
from .learning_store import LearningStore
from .workspace_store import now_iso, owner_id


def _mutation(function):
    @wraps(function)
    def invoke(self, owner, endpoint_fp, project_id='', **kwargs):
        try:
            self._scope(owner, endpoint_fp, project_id, writable=True)
            return function(self, owner, endpoint_fp, project_id, **kwargs)
        except JobStoreError as exc:
            result = {'ok': False, 'stage': 'draft', 'change_id': kwargs.get('change_id'),
                'change_revision': kwargs.get('expected_change_revision'), 'change_fingerprint': None,
                'catalog_generation': None, 'error_code': exc.code, 'detail': exc.message}
            if result['change_id']:
                try:
                    trusted, endpoint, _ = self._scope(owner, endpoint_fp)
                    with self.store.database() as conn:
                        current = self.store.change(conn, trusted, endpoint, identifier(result['change_id']))
                        result.update(self._receipt(conn, self.store, trusted, endpoint, current, ok=False))
                except JobStoreError:
                    pass
            return result
    return invoke


def _check(callback, conn=None):
    """授权/暂停回调的统一口径：抛异常=拒绝（原样传播）；返回 False=拒绝
    （旧布尔约定）；返回 None/True=无异议放行；其它返回值=坏回调，fail-closed 拒绝。
    回调漏写 return 得到的 None 是「无异议」，不会误判为拒绝；但回调绝不能靠
    返回truthy杂物表达放行。"""
    if callback is None:
        return
    result = callback(conn=conn) if conn is not None and 'conn' in inspect.signature(callback).parameters else callback()
    if result is False:
        raise RunError('knowledge_check_failed', '操作授权或来源已失效，草稿未生效。')
    if result is not None and result is not True:
        raise RunError('knowledge_invalid_check_result', '授权回调须以异常或 False 拒绝，以 None/True 放行。')


class KnowledgeService:
    def __init__(self, conversations, workspace=None):
        self.store = KnowledgeStore(conversations)
        self.root = self.store.root
        self.conversations = conversations
        self.workspace = workspace or conversations.workspace
        self.learning = LearningStore(conversations)

    def _scope(self, owner, endpoint_fp, project_id='', *, expected_identity=None, writable=False):
        owner = owner_id(canonical_principal(identifier(owner)))
        endpoint_fp, project_id = identifier(endpoint_fp, empty=True), identifier(project_id, empty=True)
        if expected_identity is not None and canonical_principal(identifier(expected_identity)) != owner:
            raise RunError('knowledge_identity_conflict', '账户已切换，请刷新后重新操作。')
        if project_id:
            self._project(owner, project_id, writable=writable)
        return owner, endpoint_fp, project_id

    def _project(self, owner, project_id, *, writable=False):
        for alias in principal_variants(owner):
            for project in self.workspace.list_projects(alias, include_archived=True):
                if project['id'] == project_id:
                    if writable and project['archived']:
                        raise RunError('knowledge_project_archived', '归档项目只读，不能修改知识。')
                    return project
        raise RunError('knowledge_project_not_found', '没有这个知识项目。')

    def _permissions(self, conn, owner, endpoint_fp):
        schema = self.workspace.membership_schema(conn)
        projects, epochs, scopes = [], [], []
        for alias in principal_variants(owner):
            projects.extend(dict(row) for row in conn.execute(
                f'SELECT id,archived FROM {schema}.conversation_projects WHERE owner=? ORDER BY id', (alias,)))
            epochs.append((alias, self.learning.config_epoch(alias, endpoint_fp, conn=conn)))
            scopes.extend(dict(row) for row in conn.execute(
                'SELECT owner,project_id,epoch,enabled FROM learning_scopes WHERE owner=? AND endpoint_fp=? ORDER BY project_id',
                (alias, endpoint_fp)))
        return digest({'owner': owner, 'endpoint_fp': endpoint_fp, 'projects': projects, 'epochs': epochs, 'scopes': scopes})

    def status(self, owner, endpoint_fp, project_id=''):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        with self.store.database() as conn:
            return {'ok': True, 'identity': owner, 'catalog_generation': self.store.generation(conn, owner, endpoint_fp),
                'system_digest': system_digest(), 'migration': self.store.migration(conn=conn),
                'config_epoch': max(self.learning.config_epoch(alias, endpoint_fp, conn=conn) for alias in principal_variants(owner)),
                'capabilities': {'kinds': list(KINDS), 'scopes': ['system', 'user', 'project'],
                    'system_read_only': True, 'human_confirmation_required': True,
                    'registered_tools': sorted(registered_tools())}}

    @staticmethod
    def _page(rows, offset, limit):
        revision(offset)
        if type(limit) is not int or not 1 <= limit <= 200:
            raise RunError('knowledge_invalid_page', '分页容量须为 1–200。')
        return {'items': rows[offset:offset + limit], 'next_offset': offset + limit if len(rows) > offset + limit else None}

    def entries(self, owner, endpoint_fp, project_id='', *, scope_level='user', kind='', status='active',
                include_inherited=False, offset=0, limit=100):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        if (any(type(v) is not str for v in (scope_level, kind, status)) or scope_level not in {'user', 'project', 'system'}
            or kind not in {'', *KINDS} or status not in {'', 'active', 'retracted'} or type(include_inherited) is not bool):
            raise RunError('knowledge_invalid_filter', '知识筛选参数无效。')
        self._page([], offset, limit)
        with self.store.database() as conn:
            if scope_level == 'system':
                rows = [r for r in system_entries() if (not kind or r['kind'] == kind) and (not status or r['status'] == status)]
                page = self._page(rows, offset, limit)
            else:
                target = project_id if scope_level == 'project' else ''
                scopes = [target, ''] if include_inherited and target else [target]
                marks = ','.join('?' for scope in scopes)
                query = f'SELECT data_json FROM knowledge_entries WHERE owner=? AND endpoint_fp=? AND project_id IN ({marks})'
                values = [owner, endpoint_fp, *scopes]
                for column, value in (('kind', kind), ('status', status)):
                    if value:
                        query += f' AND {column}=?'
                        values.append(value)
                rows = [json.loads(r['data_json']) for r in conn.execute(query + ' ORDER BY id LIMIT ? OFFSET ?',
                    (*values, limit + 1, offset))]
                page = {'items': rows[:limit], 'next_offset': offset + limit if len(rows) > limit else None}
            return {'ok': True, **page, 'catalog_generation': self.store.generation(conn, owner, endpoint_fp),
                    'system_digest': system_digest()}

    def entry(self, owner, endpoint_fp, entry_id):
        owner, endpoint_fp, _ = self._scope(owner, endpoint_fp)
        identifier(entry_id)
        if entry_id.startswith('system:'):
            row = next((r for r in system_entries() if r['id'] == entry_id), None)
            if row is None:
                raise RunError('knowledge_not_found', '没有这条内置说明。')
            return row
        with self.store.database() as conn:
            return self.store.entry(conn, owner, endpoint_fp, entry_id)

    def _public_change(self, conn, owner, endpoint, change):
        change = dict(change)
        row = conn.execute('SELECT data_json FROM knowledge_assessments WHERE owner=? AND endpoint_fp=? AND id=?',
            (owner, endpoint, change.get('assessment_id'))).fetchone()
        change['assessment'] = json.loads(row['data_json']) if row else None
        if change['assessment'] and change['status'] in {'pending', 'draft'}:
            assessment = change['assessment']
            try:
                stale = (assessment['catalog_generation'] != self.store.generation(conn, owner, endpoint)
                    or assessment['system_digest'] != system_digest()
                    or assessment['permission_hash'] != self._permissions(conn, owner, endpoint)
                    or assessment['source_hash'] != digest(self._sources(conn, owner, endpoint, change['source_refs'])))
            except JobStoreError:
                stale = True
            change['assessment_stale'] = stale
            if stale:
                change['status'] = 'stale'
                assessment['committable'] = False
        return change

    def changes(self, owner, endpoint_fp, project_id='', *, status='', offset=0, limit=100):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        if type(status) is not str or status not in {'', 'draft', 'pending', 'stale', 'committed', 'rejected'}:
            raise RunError('knowledge_invalid_filter', '提案状态无效。')
        self._page([], offset, limit)
        with self.store.database() as conn:
            selected, rows = 0, []
            cursor = conn.execute('SELECT data_json FROM knowledge_changes WHERE owner=? AND endpoint_fp=? AND project_id=? ORDER BY rowid DESC',
                (owner, endpoint_fp, project_id))
            for raw in cursor:
                row = self._public_change(conn, owner, endpoint_fp, json.loads(raw['data_json']))
                if status and row['status'] != status:
                    continue
                if selected >= offset:
                    rows.append(row)
                    if len(rows) > limit:
                        break
                selected += 1
            return {'ok': True, 'items': rows[:limit], 'next_offset': offset + limit if len(rows) > limit else None,
                    'catalog_generation': self.store.generation(conn, owner, endpoint_fp)}

    def change(self, owner, endpoint_fp, change_id):
        owner, endpoint_fp, _ = self._scope(owner, endpoint_fp)
        with self.store.database() as conn:
            return self._public_change(conn, owner, endpoint_fp,
                self.store.change(conn, owner, endpoint_fp, identifier(change_id)))

    def snapshot(self, owner, endpoint_fp, project_id=''):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        with self.store.database() as conn:
            items = [r for r in self.store.entries(conn, owner, endpoint_fp, active=True)
                     if not r['project_id'] or r['project_id'] == project_id] + system_entries()
            enforce_snapshot_budget(items)
            return {'ok': True, 'items': items, 'catalog_generation': self.store.generation(conn, owner, endpoint_fp),
                    'system_digest': system_digest()}

    @staticmethod
    def _receipt(conn, store, owner, endpoint, change, *, ok=True, stage=None, **extra):
        return {'ok': ok, 'stage': stage or change['status'], 'change_id': change['id'],
                'change_revision': change['revision'], 'change_fingerprint': change['fingerprint'],
                'catalog_generation': store.generation(conn, owner, endpoint),
                'assessment_id': change.get('assessment_id'), **extra}

    def _sources(self, conn, owner, endpoint, refs):
        if type(refs) not in {list, tuple} or len(refs) > 8:
            raise RunError('knowledge_invalid_sources', '提案出处须为不超过八项的清单。')
        sources = []
        for ref in refs:
            if type(ref) is not dict or type(ref.get('type')) is not str:
                raise RunError('knowledge_invalid_sources', '出处形状无效。')
            origin_hash = ''
            if ref['type'] == 'learning':
                if set(ref) != {'type', 'id'}:
                    raise RunError('knowledge_invalid_sources', '学习出处只接受真实事件编号。')
                sid = identifier(ref['id'])
                found = []
                for alias in principal_variants(owner):
                    found.extend(dict(r) for r in conn.execute(
                        'SELECT * FROM learning_sources WHERE owner=? AND endpoint_fp=? AND id=? AND valid=1', (alias, endpoint, sid)))
                if len(found) != 1:
                    raise RunError('knowledge_source_invalid', '出处不存在、不是当前身份端点或已经失效。')
                raw = found[0]
                self._check_conversation(conn, raw['owner'], raw['conversation_id'])
                source = {'id': sid, 'type': 'learning', 'owner': raw['owner'], 'project_id': raw['project_id'],
                    'conversation_id': raw['conversation_id'], 'origin_run_id': raw['origin_run_id'],
                    'text': text_value(raw['text'], 4000), 'outcome': raw['outcome']}
                run = conn.execute('SELECT status,fingerprint,result_json FROM conversation_runs WHERE owner=? AND id=?',
                    (raw['owner'], raw['origin_run_id'])).fetchone() if raw['origin_run_id'] else None
                origin_hash = digest({'source': raw, 'run': dict(run) if run else None})
            elif ref['type'] == 'user':
                if set(ref) - {'type', 'id', 'text', 'project_id', 'conversation_id'}:
                    raise RunError('knowledge_invalid_sources', '用户材料有未知字段。')
                text = text_value(ref.get('text'), 4000)
                project = identifier(ref.get('project_id', ''), empty=True)
                if project:
                    self._project(owner, project)
                conversation = identifier(ref.get('conversation_id', ''), empty=True)
                if conversation:
                    self._check_conversation(conn, owner, conversation, aliases=True)
                sid = identifier(ref['id']) if 'id' in ref else 'user:' + digest([text, project, conversation])[:32]
                source = {'id': sid, 'type': 'user', 'project_id': project, 'conversation_id': conversation,
                          'text': text, 'outcome': 'user_provided'}
            else:
                raise RunError('knowledge_invalid_sources', '不支持这个出处类型。')
            source['source_hash'] = digest({'source': source, 'origin_hash': origin_hash})
            sources.append(source)
        if len({r['id'] for r in sources}) != len(sources):
            raise RunError('knowledge_invalid_sources', '出处编号不能重复。')
        return sources

    @staticmethod
    def _check_conversation(conn, owner, conversation_id, *, aliases=False):
        if not conversation_id:
            return
        names = principal_variants(owner) if aliases else (owner,)
        for name in names:
            row = conn.execute('SELECT deleted FROM shared_conversations WHERE owner=? AND id=?', (name, conversation_id)).fetchone()
            if row and not row['deleted']:
                return
        raise RunError('knowledge_source_invalid', '出处对话不属于当前身份或已经删除。')

    def _operations(self, conn, owner, endpoint, project, request_id, operations):
        if type(operations) is not list or not 1 <= len(operations) <= 32:
            raise RunError('knowledge_invalid_operations', '提案须含 1–32 个明确操作。')
        normalized, seen = [], set()
        for index, op in enumerate(operations):
            if type(op) is not dict or set(op) - {'op', 'entry_id', 'expected_entry_revision', 'after', 'destination'}:
                raise RunError('knowledge_invalid_operations', '知识操作有未知字段。')
            action = op.get('op')
            if type(action) is not str or action not in {'create', 'update', 'retract', 'move'}:
                raise RunError('knowledge_invalid_operations', '不支持这个知识操作。')
            clean = {'op': action}
            if action == 'create':
                if set(op) != {'op', 'after'}:
                    raise RunError('knowledge_invalid_operations', '新增只接受正文，不接受客户端编号或版本。')
                eid = 'ke_' + digest([owner, endpoint, project, request_id, index])[:32]
            else:
                eid = identifier(op.get('entry_id'))
                if eid.startswith('system:'):
                    raise RunError('knowledge_system_read_only', '系统内置说明只读。')
                old = self.store.entry(conn, owner, endpoint, eid)
                if old['project_id'] != project:
                    raise RunError('knowledge_scope_conflict', '只能在条目原作用域提案，移动须明确两端。')
                expected = revision(op.get('expected_entry_revision'))
                if old['revision'] != expected or old['status'] != 'active':
                    raise RunError('knowledge_entry_conflict', '知识已变化，请刷新后重新提案。')
                clean['expected_entry_revision'] = expected
            if eid in seen:
                raise RunError('knowledge_invalid_operations', '一个提案不能重复修改同一条目。')
            seen.add(eid)
            clean['entry_id'] = eid
            if action in {'create', 'update'}:
                if 'destination' in op or (action == 'update' and set(op) != {'op', 'entry_id', 'expected_entry_revision', 'after'}):
                    raise RunError('knowledge_invalid_operations', '正文变更不能隐式移动作用域。')
                clean['after'] = normalize_body(op.get('after'))
            elif action == 'move':
                if set(op) != {'op', 'entry_id', 'expected_entry_revision', 'destination'}:
                    raise RunError('knowledge_invalid_operations', '移动只接受明确目标作用域。')
                destination = op.get('destination')
                if type(destination) is not dict or set(destination) != {'scope_level', 'project_id'}:
                    raise RunError('knowledge_invalid_scope', '移动须明确层级及项目。')
                target = identifier(destination['project_id'], empty=True)
                level = destination['scope_level']
                if type(level) is not str or level not in {'user', 'project'} or (level == 'project') != bool(target):
                    raise RunError('knowledge_invalid_scope', '系统只读，用户与项目层级须匹配。')
                if target:
                    self._project(owner, target, writable=True)
                if target == project:
                    raise RunError('knowledge_invalid_scope', '移动的两端不能相同。')
                clean['destination'] = dict(destination)
            elif set(op) != {'op', 'entry_id', 'expected_entry_revision'}:
                raise RunError('knowledge_invalid_operations', '撤回不能夹带正文或目标作用域。')
            normalized.append(clean)
        return normalized

    @_mutation
    def propose(self, owner, endpoint_fp, project_id='', *, request_id, operations, source_refs=(),
                actor='human', expected_identity=None, check=None):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id, expected_identity=expected_identity)
        _check(check)
        request_id = identifier(request_id)
        if type(actor) is not str or actor not in {'human', 'model', 'learning'}:
            raise RunError('knowledge_invalid_actor', '提案者标签无效，标签不授予批准权。')
        material = {'project_id': project_id, 'operations': operations, 'source_refs': source_refs, 'actor': actor}
        try:
            serialized = encode(material)
        except (ValueError, TypeError, RecursionError):
            raise RunError('knowledge_invalid_input', '提案必须是有界 JSON 数据，不能包含对象或非有限数字。') from None
        if len(serialized) > 128000:
            raise RunError('knowledge_input_limit', '提案超过完整保存预算，尚未创建草稿。')
        request_fp = digest(material)
        with self.store.database(write=True) as conn:
            _check(check, conn)
            previous = conn.execute('SELECT data_json,request_fingerprint FROM knowledge_changes WHERE owner=? AND endpoint_fp=? AND request_id=?',
                (owner, endpoint_fp, request_id)).fetchone()
            if previous:
                if previous['request_fingerprint'] != request_fp:
                    raise RunError('knowledge_request_conflict', '同一请求编号不能改写内容。')
                return self._receipt(conn, self.store, owner, endpoint_fp, json.loads(previous['data_json']), created=False)
            ops = self._operations(conn, owner, endpoint_fp, project_id, request_id, operations)
            sources = self._sources(conn, owner, endpoint_fp, source_refs)
            if any(op['op'] != 'retract' for op in ops) and not sources:
                raise RunError('knowledge_sources_required', '正文或作用域变更需要明确提交材料或真实学习出处。')
            now = now_iso()
            change = {'id': 'kc_' + uuid.uuid4().hex, 'project_id': project_id, 'request_id': request_id,
                'request_fingerprint': request_fp, 'status': 'draft', 'revision': 1, 'operations': ops,
                'source_refs': [{'type': s['type'], 'id': s['id'], **({key: s[key] for key in
                    ('text', 'project_id', 'conversation_id')} if s['type'] == 'user' else {})} for s in sources],
                'sources': sources, 'actor': actor,
                'assessment_id': None, 'created_at': now, 'updated_at': now}
            change['fingerprint'] = digest(change)
            self.store.put_change(conn, owner, endpoint_fp, change)
            self.store.audit(conn, owner, endpoint_fp, project_id, 'propose', change_id=change['id'], fingerprint=change['fingerprint'])
            return self._receipt(conn, self.store, owner, endpoint_fp, change, created=True)

    def _load_change(self, conn, owner, endpoint, project, change_id, expected):
        change = self.store.change(conn, owner, endpoint, identifier(change_id))
        if change['project_id'] != project:
            raise RunError('knowledge_scope_conflict', '提案不在当前作用域。')
        if revision(expected) != change['revision'] or change['status'] in {'committed', 'rejected'}:
            raise RunError('knowledge_change_conflict', '提案已经变化或结束，请刷新后审阅。')
        return change

    def _plan(self, conn, owner, endpoint, change):
        all_rows = self.store.entries(conn, owner, endpoint)
        active = {r['id']: r for r in all_rows if r['status'] == 'active'}
        final = dict(active)
        changed, differences = [], []
        for op in change['operations']:
            eid = op['entry_id']
            old = active.get(eid)
            if op['op'] != 'create' and (old is None or old['revision'] != op['expected_entry_revision']):
                raise RunError('knowledge_entry_conflict', '提案引用的现役版本已变化。')
            if op['op'] == 'create' and any(r['id'] == eid for r in all_rows):
                raise RunError('knowledge_entry_conflict', '新增编号已经存在。')
            if old and old['project_id'] != change['project_id']:
                raise RunError('knowledge_scope_conflict', '知识作用域已变化。')
            if op['op'] == 'retract':
                final.pop(eid)
                new = None
            else:
                project = op['destination']['project_id'] if op['op'] == 'move' else change['project_id']
                if project:
                    self._project(owner, project, writable=True)
                body = op['after'] if 'after' in op else {key: old[key] for key in
                    ('kind', 'title', 'text', 'trigger', 'tier', 'permanence', 'payload', 'links')}
                new = {**normalize_body(body), 'id': eid, 'project_id': project, 'scope_level': 'project' if project else 'user',
                    'principal': owner, 'endpoint_fp': endpoint, 'revision': (old['revision'] if old else 0) + 1,
                    'status': 'active', 'created_at': old['created_at'] if old else now_iso()}
                final[eid] = new
                changed.append(new)
            differences.append({'op': op['op'], 'entry_id': eid, 'before': old, 'after': new})
        system = system_entries()
        indexed = {r['id']: r for r in system}
        indexed.update(final)
        for row in final.values():
            for link in row['links']:
                target = indexed.get(link)
                if target is None or (target['project_id'] and target['project_id'] != row['project_id']):
                    raise RunError('knowledge_dependency_conflict', '有知识依赖将悬空或跨入不可见项目；须在同一提案更新依赖。')
                if target['id'] == row['id']:
                    raise RunError('knowledge_dependency_conflict', '知识不能依赖自己。')
        def visit(eid, pending, completed):
            if eid in pending:
                raise RunError('knowledge_dependency_cycle', '知识依赖不能成环。')
            if eid in completed:
                return
            for link in indexed[eid]['links']:
                visit(link, pending | {eid}, completed)
            completed.add(eid)
        completed = set()
        for eid in indexed:
            visit(eid, set(), completed)
        enforce_snapshot_budget(list(final.values()) + system)
        return changed, final, system, differences

    def _success(self, conn, owner, row, sources):
        if row['kind'] != 'example' or not row['payload']['verified_success']:
            return
        from ..llm.dream import _norm
        for source in sources:
            if source['type'] != 'learning' or source['outcome'] != 'done' or not source['origin_run_id']:
                continue
            run = conn.execute('SELECT status,result_json FROM conversation_runs WHERE owner=? AND id=?',
                (source['owner'], source['origin_run_id'])).fetchone()
            if not run or run['status'] != 'done' or not run['result_json']:
                continue
            result = json.loads(run['result_json'])
            if type(result) is not dict or result.get('cancelled') or result.get('result_truncated'):
                continue
            plan = result.get('plan') if type(result.get('plan')) is dict else {}
            steps = result.get('steps') or plan.get('steps')
            if type(steps) is list and steps and all(type(s) is dict and s.get('ok') is True for s in steps):
                wanted = row['payload']['steps']
                from ..agent.agent_exec import _example_args_zh
                args_match = len(steps) == len(wanted) and all(not item['args'] or item['args'] ==
                    _example_args_zh(actual.get('verb', ''), actual.get('slots') if type(actual.get('slots')) is dict else {})
                    for actual, item in zip(steps, wanted))
                if args_match and [s.get('verb') for s in steps] == [s['verb'] for s in wanted] and _norm(source['text']) == _norm(row['payload']['utterance']):
                    return
        raise RunError('knowledge_success_unverified', '没有匹配这份示例的真实成功执行记录，不能标为成功。')

    @_mutation
    def assess(self, owner, endpoint_fp, project_id='', *, change_id, expected_change_revision,
               cfg, before_llm=None, check=None, llm_call=None, predictor=None):
        """评估提案并绑定身份/CAS/来源/权限快照。

        check：授权/暂停回调，每次真实动作前后复查——抛异常或返回 False 拒绝，
        返回 None/True 放行，其它返回值 fail-closed 拒绝（见 _check）。
        before_llm：每次独立评估调用前的额度准入回调，同一口径（False/异常拒绝，
        None/True 放行，杂物拒绝）；拒绝的评估只会让提案留在 draft，绝不存成
        可提交评估，也不会核销旧评估凭据以外的任何状态。
        """
        from dataclasses import replace
        from ..llm.llm_client import LLMConfig
        if isinstance(cfg, LLMConfig):
            cfg = replace(cfg)
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        _check(check)
        with self.store.database(write=True) as conn:
            change = self._load_change(conn, owner, endpoint_fp, project_id, change_id, expected_change_revision)
            generation = self.store.generation(conn, owner, endpoint_fp)
            permissions = self._permissions(conn, owner, endpoint_fp)
            sources = self._sources(conn, owner, endpoint_fp, change['source_refs'])
            changed, final, system, differences = self._plan(conn, owner, endpoint_fp, change)
            for row in changed:
                self._success(conn, owner, row, sources)
            source_hash = digest(sources)
            if change.get('assessment_id'):
                change.update(revision=change['revision'] + 1, status='draft', assessment_id=None, updated_at=now_iso())
                change.pop('fingerprint')
                change['fingerprint'] = digest(change)
                self.store.put_change(conn, owner, endpoint_fp, change)
                self.store.audit(conn, owner, endpoint_fp, project_id, 'invalidate_assessment', change_id=change_id)
        sys_digest = system_digest()
        reports = []
        code = _config_ready(cfg, local=predictor is not None) if changed else None
        if (predictor is not None and not callable(predictor)) or (llm_call is not None and not callable(llm_call)):
            code = 'invalid_predictor'
        if predictor is not None and llm_call is not None:
            code = 'ambiguous_predictor'
        if len(changed) > MAX_ASSESS_CALLS:
            code = 'knowledge_assessment_limit'
        evidence = [{'id': s['id'], 'text': s['text']} for s in sources]
        quotes = [s['text'][:1000] for s in sources]
        if code is None:
            for row in changed:
                existing = [r for r in final.values() if r['id'] != row['id'] and
                            (not row['project_id'] or not r['project_id'] or r['project_id'] == row['project_id'])]
                existing += [dict(r, principal=owner, endpoint_fp=endpoint_fp) for r in system]
                candidate = dict(row, evidence=quotes)
                _check(check)
                denied = []
                def admission():
                    try:
                        _check(check)
                        def quota():
                            if before_llm is None:
                                return None
                            admitted = before_llm(cfg)
                            if admitted is False:
                                raise RunError('knowledge_budget_exceeded', '独立评估额度未准入。')
                            if admitted is not None and admitted is not True:
                                raise RunError('knowledge_invalid_admission_result',
                                    '额度回调须以异常或 False 拒绝，以 None/True 放行。')
                            return None
                        self.learning.charge(owner, quota=quota)
                        _check(check)
                    except JobStoreError as exc:
                        denied.append(exc)
                        raise
                def api(prompt, actual):
                    admission()
                    result = (llm_call or call_llm)(prompt, actual)
                    _check(check)
                    return result
                def local(request):
                    admission()
                    result = predictor(request)
                    _check(check)
                    return result
                relation = assess_relations(candidate, existing, evidence, cfg, knowledge=True,
                    **({'predictor': local} if predictor is not None else {'llm_call': api}))
                if denied:
                    raise denied[0]
                _check(check)
                reports.append({'entry_id': row['id'], 'relation': relation, 'context_ids': [r['id'] for r in existing],
                    'checked_ids': [r['id'] for r in existing] if not relation.get('error_code') else []})
                if relation.get('error_code'):
                    code = relation['error_code']
                elif relation['relation'] in {'conflict', 'supersede', 'temporary', 'uncertain'}:
                    code = 'knowledge_relation_blocked'
                elif relation['relation'] == 'contextual' and not row['trigger']:
                    code = 'knowledge_condition_required'
                elif row['permanence'] != 'persistent':
                    code = 'knowledge_not_permanent'
                if code:
                    break
        _check(check)
        with self.store.database(write=True) as conn:
            _check(check, conn)
            current = self._load_change(conn, owner, endpoint_fp, project_id, change_id, change['revision'])
            if current['fingerprint'] != change['fingerprint'] or self.store.generation(conn, owner, endpoint_fp) != generation or system_digest() != sys_digest:
                raise RunError('knowledge_assessment_stale', '评估期间知识或提案已变化，结果不能保存为可提交评估。')
            if self._permissions(conn, owner, endpoint_fp) != permissions or digest(self._sources(conn, owner, endpoint_fp, current['source_refs'])) != source_hash:
                raise RunError('knowledge_assessment_stale', '评估期间来源或权限已变化。')
            current.update(revision=current['revision'] + 1, status='draft' if code else 'pending', updated_at=now_iso(),
                           assessment_id='kas_' + uuid.uuid4().hex)
            current.pop('fingerprint')
            current['fingerprint'] = digest(current)
            assessment = {'id': current['assessment_id'], 'change_id': change_id, 'change_revision': current['revision'],
                'change_fingerprint': current['fingerprint'], 'catalog_generation': generation, 'system_digest': sys_digest,
                'source_hash': source_hash, 'permission_hash': permissions, 'reports': reports, 'differences': differences,
                'committable': code is None, 'complete': code is None, 'assessed_count': len(reports),
                'required_count': len(changed), 'error_code': code, 'created_at': now_iso()}
            conn.execute('INSERT INTO knowledge_assessments VALUES(?,?,?,?,?)',
                (owner, endpoint_fp, assessment['id'], change_id, encode(assessment)))
            self.store.put_change(conn, owner, endpoint_fp, current)
            self.store.audit(conn, owner, endpoint_fp, project_id, 'assess', change_id=change_id, assessment_id=assessment['id'], committable=not code)
            return self._receipt(conn, self.store, owner, endpoint_fp, current, ok=code is None,
                assessment=assessment, **({'error_code': code, 'detail': '评估未通过；提案仍是未生效草稿。'} if code else {}))

    @_mutation
    def commit(self, owner, endpoint_fp, project_id='', *, change_id, expected_change_revision,
               assessment_id, expected_catalog_generation, human_confirmed=False, expected_identity=None):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id, expected_identity=expected_identity)
        if human_confirmed is not True:
            raise RunError('human_confirmation_required', '须在可信真人审核入口逐条确认；模型不能代批。')
        revision(expected_catalog_generation)
        with self.store.database(write=True) as conn:
            change = self._load_change(conn, owner, endpoint_fp, project_id, change_id, expected_change_revision)
            row = conn.execute('SELECT data_json FROM knowledge_assessments WHERE owner=? AND endpoint_fp=? AND id=? AND change_id=?',
                (owner, endpoint_fp, identifier(assessment_id), change_id)).fetchone()
            if row is None or change['assessment_id'] != assessment_id:
                raise RunError('knowledge_assessment_required', '必须使用服务端保存的本版真实评估。')
            assessment = json.loads(row['data_json'])
            if not assessment['committable'] or change['status'] != 'pending':
                raise RunError('knowledge_not_committable', '这份评估未通过，草稿不能生效。')
            sources = self._sources(conn, owner, endpoint_fp, change['source_refs'])
            if (assessment['change_revision'] != change['revision'] or assessment['change_fingerprint'] != change['fingerprint']
                or expected_catalog_generation != assessment['catalog_generation']
                or expected_catalog_generation != self.store.generation(conn, owner, endpoint_fp)
                or assessment['system_digest'] != system_digest() or assessment['source_hash'] != digest(sources)
                or assessment['permission_hash'] != self._permissions(conn, owner, endpoint_fp)):
                raise RunError('knowledge_assessment_stale', '评估绑定的内容、知识代际、来源或权限已变化；须重新核验。')
            changed, final, system, differences = self._plan(conn, owner, endpoint_fp, change)
            for item in changed:
                self._success(conn, owner, item, sources)
            now = now_iso()
            for difference in differences:
                before, after = difference['before'], difference['after']
                entry = after if after is not None else dict(before, status='retracted', revision=before['revision'] + 1, retracted_at=now)
                entry.update(updated_at=now, confirmed_at=now,
                    provenance={'type': 'knowledge_change', 'change_id': change_id, 'assessment_id': assessment_id,
                                'sources': sources, 'quotes': [s['text'][:1000] for s in sources]})
                self.store.put_entry(conn, owner, endpoint_fp, entry)
                self.store.audit(conn, owner, endpoint_fp, entry['project_id'], difference['op'], change_id=change_id,
                    entry_id=entry['id'], before=before, after=entry)
            self.store.advance(conn, owner, endpoint_fp)
            change.update(status='committed', revision=change['revision'] + 1, updated_at=now)
            change.pop('fingerprint')
            change['fingerprint'] = digest(change)
            self.store.put_change(conn, owner, endpoint_fp, change)
            return self._receipt(conn, self.store, owner, endpoint_fp, change, applied=len(differences))

    @_mutation
    def reject(self, owner, endpoint_fp, project_id='', *, change_id, expected_change_revision, feedback=''):
        owner, endpoint_fp, project_id = self._scope(owner, endpoint_fp, project_id)
        feedback = text_value(feedback, 2000, empty=True)
        with self.store.database(write=True) as conn:
            change = self._load_change(conn, owner, endpoint_fp, project_id, change_id, expected_change_revision)
            change.update(status='rejected', revision=change['revision'] + 1, updated_at=now_iso(), feedback=feedback)
            change.pop('fingerprint')
            change['fingerprint'] = digest(change)
            self.store.put_change(conn, owner, endpoint_fp, change)
            self.store.audit(conn, owner, endpoint_fp, project_id, 'reject', change_id=change_id, feedback=feedback)
            return self._receipt(conn, self.store, owner, endpoint_fp, change)
