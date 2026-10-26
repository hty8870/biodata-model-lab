"""完整结果与客户端动作的共享投影；不执行检索、不把服务器文件路径当下载完成。"""
from __future__ import annotations

from urllib.parse import quote

from . import recommend_rows
from .recommend_rows import rows_from_retrieved
from .conversation_runs import safe_value


POLICY_ID_SCHEMA = 'biodata-policy-id/1'
SHARED_POLICY_VERSION = 'shared-turn/v1'
ROUTE_POLICY_VERSION = 'turn-route/v1'


def policy_from_snapshot(snapshot, *, sources, strategy, rerank, recall, model, app_version, base_source):
    return {'schema': POLICY_ID_SCHEMA,
        'corpus': {'snapshot_id': snapshot.get('snapshot_id', ''), 'n_records': snapshot.get('n_records', 0)},
        'sources': [str(source) for source in sources] if sources else [base_source],
        'ranking': {'strategy': str(strategy or ''), 'rerank': str(rerank or ''), 'recall': str(recall or '')},
        'model': str(model or ''), 'app_version': app_version, 'router_version': ROUTE_POLICY_VERSION}


def policy_token(value, limit=32):
    import re
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip()).strip('_')
    return (token or 'unknown')[:limit]


def policy_id_string(policy):
    import hashlib
    import json
    if not isinstance(policy, dict):
        return None
    digest = hashlib.sha256(json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    corpus = policy.get('corpus') if isinstance(policy.get('corpus'), dict) else {}
    ranking = policy.get('ranking') if isinstance(policy.get('ranking'), dict) else {}
    parts = (('snap', policy_token(corpus.get('snapshot_id'), 16)),
        ('strategy', policy_token(ranking.get('strategy'))), ('rerank', policy_token(ranking.get('rerank'))),
        ('recall', policy_token(ranking.get('recall'))), ('model', policy_token(policy.get('model'))),
        ('app', policy_token(policy.get('app_version'))), ('router', policy_token(policy.get('router_version'))))
    return 'bpol1:' + ';'.join(f'{key}={value}' for key, value in parts) + ';h=' + digest


def policy_fields(params, sources=None, model=None):
    try:
        from ..llm.config import get_settings
        from ..corpus.corpus import load_normalized_corpus, corpus_snapshot, _base_source
        settings = get_settings()
        snapshot = corpus_snapshot(load_normalized_corpus(settings.data_dir, settings.project_root, list(sources) if sources else None))
        policy = policy_from_snapshot(snapshot, sources=sources, strategy=params.get('strategy'),
            rerank=params.get('rerank', params.get('rerank_backend', 'off')),
            recall=params.get('recall', params.get('recall_backend', 'off')), model=model,
            app_version=SHARED_POLICY_VERSION, base_source=_base_source())
    except Exception:
        policy = None
    return {'policy_id': policy, 'policy_id_str': policy_id_string(policy)}


def complete_payload(meta, *, query='', params=None, sources=None, provider='', policy=None, experiment=None):
    params = dict(params or {})
    out = recommend_rows.recommend_payload(meta)
    if not out.get('provider'):
        out['provider'] = provider
    suppressed = list(params.get('suppressed_constraints') or [])
    out.update(prompt_name=getattr(meta, 'prompt_name', None),
        applied_facets=params.get('facet_filters') or [], applied_suppressed=suppressed,
        applied_lenient=sorted(params.get('lenient_dims') or []),
        action_markers=getattr(meta, 'action_markers', []) or [], strategy=getattr(meta, 'strategy', None),
        experiment=experiment, policy_id=None, policy_id_str=None)
    out.update(policy if policy is not None else policy_fields(params, sources, getattr(meta, 'model', None)))
    out['relaxation_options'] = [{
        'key': opt.get('key', ''), 'label': opt.get('label', ''), 'kind': opt.get('kind', 'drop'),
        'count': opt.get('count', 0), 'suppress': list(dict.fromkeys(suppressed + list(opt.get('suppress') or []))),
        'results': rows_from_retrieved(opt.get('retrieved_data') or [])}
        for opt in getattr(meta, 'relaxation_options', []) or []]
    degraded = getattr(meta, 'degraded_search', None)
    out['degraded_search'] = ({'ignored_terms': list(degraded.get('ignored_terms') or []),
        'query': degraded.get('query', ''), 'count': degraded.get('count', 0),
        'results': rows_from_retrieved(degraded.get('results') or []),
        'active_filters': degraded.get('active_filters') or []} if degraded else None)
    vector = getattr(meta, 'vector_fallback', None)
    out['vector_fallback'] = ({'reason': vector.get('reason', ''), 'note': vector.get('note', ''),
                             'ignored_terms': list(vector.get('ignored_terms') or [])} if vector else None)
    if not query:
        interpretation = getattr(meta, 'interpretation', {}) or {}
        query = str(interpretation.get('original_query') or interpretation.get('query') or '')
    from ..content import identifiers
    from ..corpus.corpus import load_full_corpus
    from ..llm.config import get_settings
    settings = get_settings()
    out['identifier_lookup'] = identifiers.lookup(query, lambda: load_full_corpus(settings.data_dir, settings.project_root)) if query else None
    out['applied_search_params'] = {key: params[key] for key in (
        'top_k', 'strategy', 'facet_filters', 'suppressed_constraints', 'date_from', 'date_to', 'polish',
        'auto_parse_sources', 'auto_allow_llm', 'rerank_top_n') if key in params}
    out['applied_sources'] = list((getattr(meta, 'interpretation', {}) or {}).get('effective_sources') or sources or [])
    if 'lenient_dims' in params:
        out['applied_search_params']['lenient_dims'] = sorted(params['lenient_dims'] or [])
    for wire, backend in (('recall', 'recall_backend'), ('rerank', 'rerank_backend')):
        if wire in params or backend in params:
            out['applied_search_params'][wire] = params.get(wire, params.get(backend))
    return out


def settle_search_result(result, payload, *, query, text, sources, params, allow_preliminary_final):
    import copy
    import uuid
    from datetime import datetime, timezone
    from ..agent.turn import _batch_scope_fingerprint
    batches = copy.deepcopy(result.get('result_batches') or [])
    identical = bool(batches and batches[-1].get('query_effective') == query and batches[-1].get('payload') == payload)
    preliminary_final = bool(allow_preliminary_final and identical and batches[-1].get('kind') == 'preliminary')
    if not identical:
        sequence = max((int(batch.get('seq') or 0) for batch in batches), default=0) + 1
        batches.append({'batch_id': 'b' + str(sequence), 'seq': sequence, 'kind': 'final',
            'label': query[:20], 'query_raw': text, 'query_effective': query, 'payload': payload,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'turn_id': batches[-1].get('turn_id') if batches else uuid.uuid4().hex,
            'scope_fingerprint': _batch_scope_fingerprint(query, sources, params)})
    result.update(result_payload=payload, result_batches=batches, active_batch=batches[-1]['batch_id'],
                  preliminary_final=preliminary_final)


def state_from_payload(state, payload, *, query=None):
    state = dict(state)
    params = dict(state.get('search_params') or {})
    params.update(payload.get('applied_search_params') or {})
    for field, applied in (('facet_filters', 'applied_facets'), ('suppressed_constraints', 'applied_suppressed'), ('lenient_dims', 'applied_lenient')):
        if applied in payload:
            params[field] = payload[applied]
    state.update(search_params=params, query=query if query is not None else state.get('query', ''),
        has_results=bool(payload.get('results')), result_total=int(payload.get('result_total') or 0),
        current_filters=payload.get('query_constraints') or [])
    if payload.get('applied_sources'):
        state['sources'] = payload['applied_sources']
    return state


def client_projection(body):
    from ..agent.action_plan import EXEC, VERB_SPECS
    from ..agent.agent_exec import LOOP_TOOLS
    import copy
    body = copy.deepcopy(body)
    plan = body.get('plan') or {}
    pending = list(plan.get('pending_frontend') or [])
    if plan.get('kind') == EXEC and not plan.get('steps'):
        if plan.get('verb') not in LOOP_TOOLS:
            pending.append(plan)
        else:
            body['execution'] = {'status': 'not_executed', 'reason': 'agent_unavailable'}
    specs = {s.verb: s for s in VERB_SPECS}
    body['client_actions'] = [{'verb': p.get('verb'), 'verb_zh': getattr(specs.get(p.get('verb')), 'label_zh', ''),
        'slots': safe_value(p.get('slots') or {}), 'quoted': p.get('quoted', ''),
        'status': 'requires_client', 'completed': False}
        for p in pending if not p.get('cancelled') and p.get('verb') in specs]
    artifacts = []
    for step in plan.get('steps') or []:
        if step.get('ok') and step.get('verb') == 'cite.export':
            for item in (step.get('result') or {}).get('files') or []:
                filename = str(item.get('filename') or '')
                if filename and all(part not in filename for part in ('/', '\\', '..')):
                    artifacts.append(dict(item, download_path='/api/citations/download?f=' + quote(filename)))
    body['artifacts'] = artifacts
    def strip_paths(value):
        if isinstance(value, dict):
            return {k: strip_paths(v) for k, v in value.items() if k not in {'out_dir', 'dest_dir', 'dir', 'path', 'local_path'}}
        if isinstance(value, list):
            return [strip_paths(v) for v in value]
        return value
    return safe_value(strip_paths(body))
