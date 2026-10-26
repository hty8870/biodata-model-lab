"""共享回合执行服务；HTTP/MCP 只适配身份和配置，不复制 Agent 或推荐算法。"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from ..agent import action_plan, agent_exec, turn
from ..agent.learning_scope import bind_project_scope
from ..llm.config import get_settings
from ..llm.llm_client import should_use_llm
from .conversation_runs import ConversationRuntime, RunError, safe_value
from .conversation_store import ConversationStore
from .llm_admission import Admission, gate_quota, materialize
from .recommend_rows import recommend_payload
from .request_validation import validate_date_window, validate_iso_date, validate_sources
from .runtime_paths import get_app_paths
from .workflow import DatasetRecommendationWorkflow, RecommendParams, sanitize_facet_filters, sanitize_lenient_dims, sanitize_suppressed
from .workspace_store import WorkspaceStore


STATE_FIELDS = frozenset({'query', 'sources', 'search_params'})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def search_parameters(raw=None):
    raw = dict(raw or {})
    permitted = {'top_k', 'rerank', 'rerank_top_n', 'recall', 'strategy', 'facet_filters', 'suppressed_constraints', 'lenient_dims', 'date_from', 'date_to', 'polish', 'auto_parse_sources', 'auto_allow_llm'}
    if set(raw) - permitted:
        raise RunError('invalid_search_parameter', '不支持这个检索参数。')
    top = raw.get('top_k')
    if top is not None and (not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 50):
        raise RunError('bad_param', 'top_k 须为 1–50。')
    rerank_top = raw.get('rerank_top_n')
    if rerank_top is not None and (not isinstance(rerank_top, int) or isinstance(rerank_top, bool) or not 1 <= rerank_top <= 50):
        raise RunError('bad_param', 'rerank_top_n 须为 1–50。')
    for flag in ('auto_parse_sources', 'auto_allow_llm', 'polish'):
        if flag in raw and not isinstance(raw[flag], bool):
            raise RunError('bad_param', flag + ' 须为布尔值。')
    start, end = validate_iso_date(raw.get('date_from'), name='date_from'), validate_iso_date(raw.get('date_to'), name='date_to')
    validate_date_window(start, end)
    return {'top_k': top, 'rerank_top_n': rerank_top,
            'auto_parse_sources': bool(raw.get('auto_parse_sources', False)),
            'auto_allow_llm': bool(raw.get('auto_allow_llm', False)),
            'rerank': 'llm' if raw.get('rerank') == 'llm' else 'off',
            'recall': raw.get('recall') if raw.get('recall') in {'dense', 'cross_encoder', 'vector'} else 'off',
            'strategy': 'auto' if raw.get('strategy') == 'auto' else 'fixed',
            'facet_filters': sanitize_facet_filters(raw.get('facet_filters')),
            'suppressed_constraints': sanitize_suppressed(raw.get('suppressed_constraints')),
            'lenient_dims': sorted(sanitize_lenient_dims(raw.get('lenient_dims'))),
            'date_from': start, 'date_to': end, 'polish': bool(raw.get('polish', True))}


def route_with_project(project_id, text, **kwargs):
    with bind_project_scope(project_id):
        return turn.route_turn(text, **kwargs)


def project_turn(result):
    body = {key: result[key] for key in ('route', 'query', 'plan', 'echo_zh', 'retrieval', 'via')}
    body.update(ok=True, needs_agent=bool(result.get('needs_agent')), suggestions=list(result.get('suggestions') or []),
                agent={'available': agent_exec.agent_available(), 'used': result['via'] == 'agent'},
                result_payload=result.get('result_payload'), preliminary_final=bool(result.get('preliminary_final')))
    for key in ('retrieval_note', 'result_batches', 'active_batch', 'trace_turn_id', 'policy_id', 'policy_id_str', 'experiment', 'recipe_note', 'ask_relax'):
        if key in result and (result[key] is not None) and (key == 'ask_relax' or result[key]):
            body[key] = result[key]
    return body


def _owned_turn(operation):
    from functools import wraps
    @wraps(operation)
    def execute(*args, **kwargs):
        from ..corpus.patch_package import bind_patch_scope
        principal = str(kwargs.get('principal') or 'anonymous')
        with bind_patch_scope(None if principal == 'anonymous' else principal), bind_project_scope(kwargs.get('project_id') or ''):
            return operation(*args, **kwargs)
    return execute


@_owned_turn
def execute_turn(text, *, cfg, principal, project_id='', state=None, use_agent=True,
                 history=None, ask_answer=None, on_event=None, artifact_context='', suggested_recipe='',
                 complete_search=False, before_llm=None, route=None):
    state = dict(state or {})
    params = search_parameters(state.get('search_params')) if complete_search else state.get('search_params')
    sources = state.get('sources')
    if complete_search and params['auto_parse_sources']:
        from ..retrieval.search_request import resolve_search_request
        from ..corpus.corpus import known_source_values
        settings = get_settings()
        sources = resolve_search_request(text, sources,
            known_source_values(settings.data_dir, settings.project_root), auto_parse_sources=True).sources
    prefetched = []
    kwargs = dict(has_results=bool(state.get('has_results')), result_total=int(state.get('result_total') or 0),
                  current_query=state.get('query') or '', current_filters=state.get('current_filters'), sources=sources,
                  config=cfg, keyword_mapping=get_settings().keyword_mapping, use_agent=use_agent, principal=principal,
                  search_params=params, artifact_context=artifact_context, suggested_recipe=suggested_recipe, ask_answer=ask_answer)
    if on_event is not None:
        kwargs['on_event'] = on_event
    if complete_search:
        kwargs.update(conversation_history=history or [], prefetched_out=prefetched)
    if before_llm and use_agent:
        before_llm(cfg)
    result = (route or route_with_project)(project_id, text, **kwargs)
    if complete_search and result.get('route') == 'search':
        query = result.get('query') or text
        batches = result.get('result_batches') or []
        verified_batch = bool(batches and batches[-1].get('query_effective') == query and batches[-1].get('kind') != 'preliminary')
        armed = should_use_llm(cfg)[0]
        paid_search = bool(armed and (params['polish'] or params['rerank'] == 'llm' or
            (params['strategy'] == 'auto' and params['auto_allow_llm'])))
        payload = result.get('result_payload') if (result.get('preliminary_final') and not paid_search) or verified_batch else None
        if payload is None and not paid_search:
            payload = next((p['payload'] for p in prefetched if p['query'] == query), None)
        if payload is None:
            preferred, available = 'cross_encoder', None
            if params['strategy'] == 'auto':
                from ..retrieval.vector_recall import resolve_auto_recall_preference
                preferred, available = resolve_auto_recall_preference()
            meta = DatasetRecommendationWorkflow().run_with_meta(RecommendParams(
                query=query, top_k=params['top_k'], sources=sources, provider=cfg.provider,
                auto_parse_sources=params['auto_parse_sources'],
                use_llm=bool(cfg.enable_llm and params['polish']), mock_llm=cfg.mock_llm,
                rerank_backend=params['rerank'] if cfg.enable_llm else 'off', rerank_top_n=params['rerank_top_n'],
                recall_backend=params['recall'], strategy=params['strategy'],
                facet_filters=params['facet_filters'], suppressed_constraints=params['suppressed_constraints'],
                lenient_dims=params['lenient_dims'], date_from=params['date_from'], date_to=params['date_to'],
                base_llm_config=cfg, llm_available=bool(params['auto_allow_llm'] and should_use_llm(cfg)[0]),
                preferred_recall=preferred, recall_available=available),
                before_llm=before_llm)
            from .conversation_results import complete_payload
            payload = complete_payload(meta, query=query, params=params, sources=sources, provider=cfg.provider)
        from .conversation_results import policy_fields, settle_search_result
        payload = dict(payload)
        policy = policy_fields(params, payload.get('applied_sources') or sources, cfg.model)
        payload.update(policy)
        settle_search_result(result, payload, query=query, text=text,
            sources=payload.get('applied_sources') or sources, params=params,
            allow_preliminary_final=bool(result.get('preliminary_final') and not paid_search))
        result.update(policy)
    body = project_turn(result)
    if complete_search:
        payload = body.get('result_payload')
        if isinstance(payload, dict):
            batches = body.get('result_batches') or []
            active_query = (batches[-1].get('query_effective') if batches else None) or body.get('query') or state.get('query', '')
            from .conversation_results import state_from_payload
            state = state_from_payload(dict(state, search_params=params), payload, query=active_query)
            params = state['search_params']
        state['search_params'] = search_parameters(params)
        state['ask_relax'] = body.get('ask_relax')
        body['_current_state'] = state
        plan = body.get('plan') or {}
        from .conversation_results import client_projection
        body = client_projection(body)
        body['agent']['degraded'] = body['via'] != 'agent'
        if body['agent']['degraded']:
            body['agent']['reason'] = (plan.get('llm_status') or body['via'])
    return body


class ConversationService:
    def __init__(self, project_root, *, runtime=None, workspace=None):
        self.project_root = Path(project_root)
        self.workspace = workspace or WorkspaceStore(project_root)
        self.store = runtime.store if runtime is not None else ConversationStore(project_root)
        self.store.workspace = self.workspace
        self.runtime = runtime or ConversationRuntime(self.store)

    def _project_conversations(self, owner, conversations):
        assignments = self.workspace.assignments(owner)
        return [dict(conversation, project_id=assignments.get(conversation['id'], '')) for conversation in conversations]

    def list_conversations(self, owner, *, offset=0, limit=100, include_archived=False):
        result = self.store.list_conversations(owner, offset=offset, limit=limit, include_archived=include_archived)
        result['items'] = self._project_conversations(owner, result['items'])
        return result

    def get_conversation(self, owner, conv_id, *, after_message=0, after_frame=0, limit=100):
        conversation = self.store.get(owner, conv_id, after_message=after_message, after_frame=after_frame, limit=limit)
        return self._project_conversations(owner, [conversation])[0]

    def chat(self, owner, text, *, request_id, conv_id=None, expected_revision=None, project_id=None,
             state_patch=None, ask_choice=None, use_agent=True, configure=None, quota=None, config_fingerprint=''):
        text = action_plan.normalize_utterance(text)
        if not request_id:
            raise RunError('request_id_required', '共享回合需要稳定 request_id，重试时请保持原编号。')
        raw = {'text': text, 'conv_id': conv_id, 'expected_revision': expected_revision, 'project_id': project_id,
               'state_patch': state_patch, 'ask_choice': ask_choice, 'use_agent': use_agent, 'config_fingerprint': config_fingerprint}
        fp = fingerprint(raw)
        try:
            previous = self.store.find_request(owner, request_id)
        except RunError as exc:
            if exc.code != 'run_not_found':
                raise
        else:
            return self.runtime.submit(owner, request_id, fp, prepare=lambda: None)
        with self.store.database(write=True) as conn:
            if conv_id:
                conversation = self.store._public_conversation(self.store._row(conn, owner, conv_id))
            else:
                conv_id = 'c_' + fingerprint({'owner': owner, 'request_id': request_id})[:32]
                conversation = self.store.create(owner, conv_id=conv_id, title=safe_value(text)[:180], conn=conn)
            revision = conversation['revision'] if expected_revision is None else expected_revision
            project = self.workspace.resolve_conversation(owner, conv_id, project_id, title=safe_value(text)[:180], conn=conn)
        patch = dict(state_patch or {})
        if set(patch) - STATE_FIELDS:
            raise RunError('invalid_state_patch', '只允许修改查询、来源或检索参数；执行事实不可由调用方写入。')
        if 'search_params' in patch:
            merged = dict(conversation['current_state'].get('search_params') or {})
            merged.update(patch['search_params'] or {})
            patch['search_params'] = search_parameters(merged)
        if 'sources' in patch:
            from ..corpus.corpus import known_source_values
            from ..corpus.patch_package import bind_patch_scope
            with bind_patch_scope(None if owner == 'anonymous' else owner):
                validate_sources(patch['sources'], known=known_source_values(get_settings().data_dir, self.project_root))
        def prepare():
            from .learning_service import service_for
            learning, snapshot = None, None
            authorization = {'configured': False, 'error_code': ''}
            try:
                learning = service_for(self.store, self.workspace)
                snapshot = learning.store.authorization_snapshot(owner)
            except Exception as exc:
                authorization['error_code'] = str(getattr(exc, 'code', 'learning_snapshot_unavailable'))[:80]
            cfg = configure() if configure else materialize(config_root=get_app_paths().config_root)
            learning_quota = quota or (lambda c: gate_quota(self.project_root, username='', cfg=c))
            admit = Admission(learning_quota)
            endpoint_fp = agent_exec._endpoint_fp_from_config(cfg)
            if snapshot is not None:
                try:
                    learning.authorize(owner, cfg, learning_quota, project_id=project,
                        expected_config_epoch=snapshot['config_epochs'].get(endpoint_fp, 0),
                        expected_scope_epoch=snapshot['scope_epochs'].get((endpoint_fp, project), 0))
                    authorization['configured'] = learning.status(owner, endpoint_fp, project)['configured']
                except Exception as exc:
                    authorization['error_code'] = str(getattr(exc, 'code', 'learning_authorization_unavailable'))[:80]
            if use_agent:
                admit(cfg)
            admitted = self.store.find_request(owner, request_id)
            self.store.bind_learning_endpoint(owner, admitted['id'], endpoint_fp, authorization=authorization)
            state = admitted['request']['current_state']
            answer = None
            if ask_choice:
                pending = state.get('ask_relax') or {}
                choices = pending.get('choices') or pending.get('options') or []
                choice = next((c for c in choices if str(c.get('key') or c.get('choice_key')) == str(ask_choice)), None)
                if choice is None:
                    raise RunError('ask_choice_not_found', '该选择不属于当前待回答问题，请刷新对话。')
                answer = dict(choice, kind='ask.relax', choice_key=ask_choice)
                params = dict(state.get('search_params') or {})
                params['suppressed_constraints'] = choice.get('suppress') or []
                state = dict(state, search_params=params)
            history = self.store.get(owner, conv_id, after_message=max(0, conversation.get('message_count', 0) - 24))['messages']
            def execute(emit):
                effective = state.get('query') or text if answer else text
                body = execute_turn(effective, cfg=cfg, principal=owner, project_id=project, state=state,
                    use_agent=use_agent, history=history, ask_answer=answer, on_event=emit,
                    complete_search=True, before_llm=admit)
                from . import replay_service
                with replay_service.owned_access(self.project_root, owner, conv_id):
                    replay_service.record_turn(self.project_root, owner, conv_id, '', 'chat',
                        {'user': text, 'route': body['route'], 'reply': (body.get('plan') or {}).get('report_zh') or body.get('echo_zh', ''),
                         'steps': (body.get('plan') or {}).get('steps') or [], 'run_id': admitted['id']}, cfg, before_llm=admit)
                return body
            if not authorization['error_code']:
                execute.after_commit = lambda: learning.notify(owner, endpoint_fp, project)
            return execute
        return self.runtime.submit(owner, request_id, fp, prepare=prepare, conversation_id=conv_id,
            project_id=project, request={'shared': True, 'text': text, 'expected_revision': revision,
                'project_id': project, 'state_patch': patch})

    def conversation_action(self, owner, conv_id, action, *, expected_revision=None, frame_id='', title='', project_id='', import_id='', data=None):
        if action == 'delete':
            return {'ok': True, 'conversation': self.store.action(owner, conv_id, action, expected_revision=expected_revision)}
        from contextlib import nullcontext
        from .replay_store import transaction
        lock = transaction(self.project_root, owner, conv_id) if conv_id else nullcontext()
        with lock, self.store.database(write=True) as conn:
            if action == 'create':
                conversation = self.store.create(owner, title=title, conn=conn)
            elif action == 'import':
                if not data or not import_id:
                    raise RunError('import_confirmation_required', '请显式提供预览后确认的历史内容及 import_id。')
                conversation = self.store.import_conversation(owner, import_id, fingerprint(data), data, conn=conn)
            elif action == 'project':
                row = self.store._row(conn, owner, conv_id)
                self.store._mutable(row, expected_revision)
                assignment = self.workspace.assign_conversation(owner, conv_id, project_id, conn=conn)
                conn.execute('UPDATE shared_conversations SET revision=revision+1 WHERE owner=? AND id=?', (owner, conv_id))
                conversation = self.store._public_conversation(self.store._row(conn, owner, conv_id))
                return {'ok': True, 'conversation': dict(conversation, project_id=assignment['project_id'])}
            else:
                conversation = self.store.action(owner, conv_id, action, expected_revision=expected_revision,
                    frame_id=frame_id, title=title, conn=conn)
            project = self.workspace.resolve_conversation(owner, conversation['id'],
                project_id if action in {'create', 'branch', 'import'} else None, title=conversation.get('title', ''), conn=conn)
            return {'ok': True, 'conversation': dict(conversation, project_id=project)}

    def replay(self, owner, conv_id, action='points', data=None, *, configure=None, quota=None):
        from . import replay_service as replay
        data = dict(data or {})
        allowed = {'pid', 'text', 'snap_id', 'observed_version', 'observed_pids'}
        if set(data) - allowed:
            raise RunError('invalid_replay_parameter', '不支持这个复盘参数。')
        with replay.owned_access(self.project_root, owner, conv_id):
            args = (self.project_root, owner, conv_id, '')
            if action == 'status':
                result = replay.status(*args)
            elif action == 'points':
                result = replay.points(*args)
            elif action == 'annotate':
                result = {'ok': bool(replay.annotate(*args, data.get('pid', ''), str(data.get('text', ''))))}
            elif action == 'snapshot':
                result = replay.take_snapshot(*args)
            elif action == 'annotate_snapshot':
                result = replay.annotate_snapshot(*args, data.get('snap_id', ''), data.get('pid', ''), str(data.get('text', '')))
            elif action == 'seen':
                result = replay.mark_seen(*args, observed_version=data.get('observed_version'), observed_pids=data.get('observed_pids'))
            elif action in {'chat', 'digest'}:
                if action == 'chat':
                    text = str(data.get('text', '')).strip()
                    if not text or len(text) > 2000:
                        raise RunError('invalid_replay_text', '复盘问题须为 1–2000 个字符。')
                if not replay.authorized_owned(self.project_root, owner, conv_id):
                    raise RunError('replay_not_found', '复盘内容不可访问。')
                cfg = configure() if configure else materialize(config_root=get_app_paths().config_root)
                if quota:
                    quota(cfg)
                if action == 'chat':
                    result = replay.chat(*args, text, cfg)
                else:
                    result = replay.manual_digest(*args, cfg)
            else:
                raise RunError('invalid_replay_action', '不支持这个复盘动作。')
        if result is None or result is False:
            raise RunError('replay_not_found', '复盘内容或标注目标不存在。')
        return safe_value(result)
