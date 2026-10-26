                       
"""协作规则兼容面：只读统一 SQLite 投影，旧修改入口只提交知识提案。

原 JSONL 是永久保留的迁移原档，不再写入或参与活跃读取。正文、撤回、候选审阅均
经过 KnowledgeService；调用方传来的关系不是服务端评估，actor 标签也不授予批准权。
真人提交须引用实际保存的 assessment 与版本、来源、目录及权限快照；草稿不引用。
旧批准计数只统计真实生效，不把已提案描述成已批准。所有入口保持账户、端点、项目隔离。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..llm import dream as _dream
from . import learning_scope as _learning_scope

                                                                          
                              
_COLLAB_RULES_LEDGER_NAME = "collab_rules.jsonl"

STATUS_ACTIVE = "active"
STATUS_RETRACTED = "retracted"


def ledger_path(root: Path) -> Path:
    """原账本路径：只作迁移留档定位（`.userdata/`，随每日备份/迁移走），不再有活跃读写。"""
    from ..corpus.net_ledger import net_ledger_path

    return net_ledger_path(Path(root)).parent / _COLLAB_RULES_LEDGER_NAME


def _now_iso() -> str:
    from ..corpus.net_ledger import now_iso as _iso

    return _iso()


def _knowledge_service(root):
    from ..app.conversation_store import ConversationStore
    from ..app.knowledge_service import KnowledgeService
    return KnowledgeService(ConversationStore(Path(root)))


def _candidate_projection(row):
    operations = row.get('operations', [])
    body = operations[0].get('after', {}) if operations else {}
    assessment = row.get('assessment') or {}
    reports = assessment.get('reports', [])
    return dict(row, **body, record_type='candidate', version=row['revision'],
        fingerprint=row['fingerprint'], existing_fingerprint=str(assessment.get('catalog_generation', '')),
        change_id=row['id'], change_revision=row['revision'],
        evidence=[s['text'] for s in row.get('sources', [])],
        relation=reports[0]['relation'] if reports else None)


def is_rule(row: dict) -> bool:
    return isinstance(row, dict) and row.get('record_type', 'rule') == 'rule'


def content_fingerprint(value) -> str:
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def rules_fingerprint(rows: list[dict]) -> str:
    normalized = [dict(r, principal=_learning_scope.canonical_principal(r.get('principal')),
        endpoint_fp=str(r.get('endpoint_fp') or ''), project_id=str(r.get('project_id') or ''))
        for r in rows if is_rule(r)]
    return content_fingerprint(sorted(normalized, key=lambda r: str(r.get('id', ''))))


def learning_rules_snapshot(root, *, principal, endpoint_fp, project_id=''):
    from .knowledge import legacy_rules
    from .learning_loop import MAX_RULES
    selected = legacy_rules(_knowledge_service(root).snapshot(principal, endpoint_fp,
        _learning_scope.resolve_project_id(project_id))['items'])
    if len(selected) > MAX_RULES:
        raise _candidate_error('learning_context_limit', '完整知识超过核验预算，不会截断潜在冲突。')
    return selected


def list_collab_rules(root: Path, *, principal: "str | None" = None,
                      endpoint_fp: "str | None" = None, project_id: str | None = None) -> list[dict]:
    """原规则只读统一知识投影，不读回留档文件。"""
    from .knowledge import legacy_rules
    if principal is None:
        return []
    service = _knowledge_service(root)
    project = _learning_scope.resolve_project_id(project_id)
    rows, offset = [], 0
    while True:
        page = service.entries(principal, endpoint_fp or '', project,
            scope_level='project' if project else 'user', kind='memory', status='', offset=offset, limit=200)
        rows.extend(page['items'])
        offset = page['next_offset']
        if offset is None:
            return list(reversed(legacy_rules(rows)))


def approve_collab_rules(root: Path, items: Any, *, principal: str = "anonymous",
                         endpoint_fp: str = "", project_id: str | None = None) -> dict:
    """旧确认入口只产生统一草稿；无保存评估和真人提交不能计 approved。"""
    if type(items) is not list or len(items) > 32:
        raise _candidate_error('knowledge_invalid_operations', '规则提案需要有界清单。')
    project_id = _learning_scope.resolve_project_id(project_id)
    service = _knowledge_service(root)
    current = service.entries(principal, endpoint_fp, project_id, scope_level='project' if project_id else 'user')['items']
    result = {'approved': 0, 'duplicated': 0, 'rejected': 0, 'proposed_count': 0, 'changes': []}
    for item in items:
        clean = _dream.normalize_rule_candidate(item) if type(item) is dict and item.get('record_type', 'rule') == 'rule' else None
        if clean is None:
            result['rejected'] += 1
            continue
        if any(row['status'] == 'active' and row['text'] == clean['text'] and row['trigger'] == clean.get('trigger', '') for row in current):
            result['duplicated'] += 1
            continue
        after = {'kind': 'memory', 'title': '', 'text': clean['text'], 'trigger': clean.get('trigger', ''),
            'tier': clean['tier'], 'permanence': item.get('permanence', 'unknown'), 'payload': {}, 'links': []}
        sources = [{'type': 'user', 'text': '\n'.join(clean['evidence']) or clean['text'], 'project_id': project_id}]
        receipt = service.propose(principal, endpoint_fp, project_id,
            request_id='legacy-rule:' + content_fingerprint(after)[:32],
            operations=[{'op': 'create', 'after': after}], source_refs=sources)
                                                       
                                                
                              
        while receipt['ok'] and service.change(principal, endpoint_fp, receipt['change_id'])['status'] in {'committed', 'rejected'}:
            receipt = service.propose(principal, endpoint_fp, project_id,
                request_id='legacy-rule:' + content_fingerprint([after, 'follow', receipt['change_id']])[:32],
                operations=[{'op': 'create', 'after': after}], source_refs=sources)
        result['changes'].append(receipt)
        result['proposed_count' if receipt['ok'] else 'rejected'] += 1
    result.update(stage='draft', detail='已保存的内容仅为提案；请在知识审查中评估并真人确认。')
    if len(result['changes']) == 1:
        result.update({key: result['changes'][0].get(key) for key in ('change_id', 'change_revision', 'assessment_id')})
    return result


def _resolve_entry(service, principal, endpoint_fp, entry_id, filename='collab_rules.jsonl'):
    from ..app.conversation_runs import RunError
    try:
        return service.entry(principal, endpoint_fp, entry_id)
    except RunError as exc:
        if exc.code != 'knowledge_not_found':
            raise
    new_id = service.store.resolve_legacy_id(principal, endpoint_fp, entry_id, filename=filename)
    return service.entry(principal, endpoint_fp, new_id)


def revoke_collab_rules(root: Path, ids: Any, *, principal: str = "anonymous",
                        endpoint_fp: str = "", project_id: str | None = None) -> dict:
    """撤回仍先过独立依赖检查；旧入口没有提交凭据，只回待真人确认提案。"""
    if type(ids) is not list or not 1 <= len(ids) <= 32:
        raise _candidate_error('knowledge_invalid_operations', '撤回编号清单无效。')
    service = _knowledge_service(root)
    project = _learning_scope.resolve_project_id(project_id)
    operations = []
    for rid in ids:
        row = _resolve_entry(service, principal, endpoint_fp, rid)
        if row['project_id'] != project:
            raise _candidate_error('knowledge_scope_conflict', '知识不属于这个项目。')
        if row['status'] == 'active':
            operations.append({'op': 'retract', 'entry_id': row['id'], 'expected_entry_revision': row['revision']})
    if not operations:
        return {'ok': True, 'revoked': 0, 'stage': 'unchanged'}
    receipt = service.propose(principal, endpoint_fp, project,
        request_id='legacy-retract:' + content_fingerprint(operations)[:32], operations=operations)
    if receipt['ok']:
        receipt = service.assess(principal, endpoint_fp, project, change_id=receipt['change_id'],
            expected_change_revision=receipt['change_revision'], cfg=None)
    return {'revoked': 0, **receipt, 'detail': '撤回未生效；通过依赖检查后仍须在知识审查中真人确认。'}


def _candidate_error(code, hint):
    from ..app.conversation_runs import RunError
    return RunError(code, hint)


def list_learning_records(root, *, principal, endpoint_fp, project_id='', record_type='candidate',
                          status='', offset=0, limit=100):
    service = _knowledge_service(root)
    identity = service.status(principal, endpoint_fp, project_id)['identity']
    if record_type == 'candidate':
        page = service.changes(principal, endpoint_fp, project_id, status=status, offset=offset, limit=limit)
        return {**page, 'items': [dict(_candidate_projection(row), principal=identity, endpoint_fp=endpoint_fp) for row in page['items']]}
    if record_type != 'audit':
        raise _candidate_error('invalid_learning_record', '不支持这个记录类型。')
    service._page([], offset, limit)
    import json
    with service.store.database() as conn:
        rows = conn.execute('SELECT data_json FROM knowledge_audit WHERE owner=? AND endpoint_fp=? AND project_id=? ORDER BY rowid DESC LIMIT ? OFFSET ?',
            (identity, endpoint_fp, project_id, limit + 1, offset)).fetchall()
    return {'ok': True, 'items': [dict(json.loads(row['data_json']), record_type='audit', relation_label=None) for row in rows[:limit]],
        'next_offset': offset + limit if len(rows) > limit else None}


def get_learning_candidate(root, candidate_id, *, principal, endpoint_fp, project_id=''):
    from ..app.conversation_runs import RunError
    service = _knowledge_service(root)
    try:
        row = service.change(principal, endpoint_fp, candidate_id)
    except RunError as exc:
        if exc.code != 'knowledge_change_not_found':
            raise
        target = service.store.resolve_legacy_id(principal, endpoint_fp, candidate_id, filename='collab_rules.jsonl')
        row = service.change(principal, endpoint_fp, target)
    if row['project_id'] != project_id:
        raise _candidate_error('knowledge_scope_conflict', '提案不属于这个项目。')
    return _candidate_projection(row)


def save_learning_candidate(root, item, *, principal, endpoint_fp, project_id, sources,
                            relation, existing, permanence='unknown'):
    """兼容旧候选调用，但客户端传来的关系不是保存评估。"""
    clean = _dream.normalize_rule_candidate(item)
    if clean is None or not clean['evidence'] or not sources:
        raise _candidate_error('candidate_evidence_required', '候选需要真实出处。')
    current = learning_rules_snapshot(root, principal=principal, endpoint_fp=endpoint_fp, project_id=project_id)
    if rules_fingerprint(current) != rules_fingerprint(existing):
        raise _candidate_error('candidate_rules_conflict', '知识已变化，请刷新后提案。')
    service = _knowledge_service(root)
    after = {'kind': 'memory', 'title': '', 'text': clean['text'], 'trigger': clean.get('trigger', ''),
        'tier': clean['tier'], 'permanence': permanence, 'payload': {}, 'links': []}
    refs = [{'type': 'learning', 'id': source['id']} for source in sources]
    receipt = service.propose(principal, endpoint_fp, project_id,
        request_id='legacy-learning:' + content_fingerprint([after, refs])[:32],
        operations=[{'op': 'create', 'after': after}], source_refs=refs, actor='learning')
    if not receipt['ok']:
        raise _candidate_error(receipt['error_code'], receipt['detail'])
    row = _candidate_projection(service.change(principal, endpoint_fp, receipt['change_id']))
    return {**receipt, 'candidate': row}


def review_learning_candidate(root, candidate_id, *, principal, endpoint_fp, project_id='',
                              decision, expected_fingerprint, expected_rules_fingerprint='',
                              feedback='', human_confirmed=False):
    if decision not in {'approve', 'replace', 'reject'}:
        raise _candidate_error('invalid_learning_review', '请选择确认或驳回；替代须在提案里明确全部操作。')
    service = _knowledge_service(root)
    candidate = service.change(principal, endpoint_fp, candidate_id)
    if candidate['project_id'] != project_id or candidate['fingerprint'] != expected_fingerprint:
        raise _candidate_error('candidate_revision_conflict', '提案或范围已变化，请刷新。')
    if decision == 'reject':
        result = service.reject(principal, endpoint_fp, project_id, change_id=candidate_id,
            expected_change_revision=candidate['revision'], feedback=feedback)
    else:
        if human_confirmed is not True:
            raise _candidate_error('human_confirmation_required', '模型不能代替真人确认。')
        assessment = candidate.get('assessment')
        if not assessment:
            raise _candidate_error('knowledge_assessment_required', '须先在统一知识道闸保存本版真实评估。')
        if expected_rules_fingerprint != str(assessment['catalog_generation']):
            raise _candidate_error('knowledge_assessment_stale', '知识目录代际已变化，请重新核验。')
        result = service.commit(principal, endpoint_fp, project_id, change_id=candidate_id,
            expected_change_revision=candidate['revision'], assessment_id=assessment['id'],
            expected_catalog_generation=assessment['catalog_generation'], human_confirmed=True)
    if result['ok']:
        result['candidate'] = _candidate_projection(service.change(principal, endpoint_fp, candidate_id))
    return result


def revoke_learning_rule(root, rule_id, *, principal, endpoint_fp, project_id, expected_fingerprint):
    service = _knowledge_service(root)
    rule = _resolve_entry(service, principal, endpoint_fp, rule_id)
    from .knowledge import legacy_rules
    if content_fingerprint(legacy_rules([rule])[0]) != expected_fingerprint:
        raise _candidate_error('rule_revision_conflict', '知识已变化，请刷新后撤回。')
    return revoke_collab_rules(root, [rule['id']], principal=principal, endpoint_fp=endpoint_fp, project_id=project_id)
