"""知识分类与条件的纯投影；主题不授予权限，也不证明适用域互斥。"""
from __future__ import annotations

import json

from ..app.conversation_runs import RunError

UNCATEGORIZED = 'uncategorized'
BODY_FIELDS = ('kind', 'title', 'text', 'trigger', 'tier', 'permanence', 'payload', 'links', 'disclosure')
PROJECTED_FIELDS = ('id', 'scope_level', 'project_id', 'kind', 'title', 'text', 'trigger',
                    'tier', 'permanence', 'payload', 'links', 'revision', 'disclosure')
MAX_READ_ITEMS = 64
MAX_READ_CHARS = 24000


def normalize_disclosure(value):
    from .knowledge import identifier
    if type(value) is not dict or set(value) - {'categories', 'always'}:
        raise RunError('knowledge_invalid_disclosure', '披露配置只接受分类和常驻标记；适用条件请写在正文或适用场景。')
    categories = value.get('categories', [])
    if type(categories) is not list or len(categories) > 8 or any(type(v) is not str for v in categories):
        raise RunError('knowledge_invalid_disclosure', '一条知识最多关联八个分类编号。')
    categories = [identifier(v) for v in categories]
    if UNCATEGORIZED in categories or len(set(categories)) != len(categories):
        raise RunError('knowledge_invalid_disclosure', '分类编号不能重复或使用未分类保留编号。')
    always = value.get('always', False)
    if type(always) is not bool:
        raise RunError('knowledge_invalid_disclosure', '常驻标记须为布尔值。')
    return {'categories': sorted(categories), 'always': always}


def disclosure_of(row):
    return normalize_disclosure(row.get('disclosure', {}))


def category_definitions(values):
    from .knowledge import identifier, normalize_body, text_value
    if type(values) is not list or len(values) > 32:
        raise RunError('knowledge_invalid_category', '新分类清单最多三十二项。')
    result = []
    for value in values:
        if type(value) is not dict or set(value) != {'id', 'title', 'description'}:
            raise RunError('knowledge_invalid_category', '分类须含编号、名称和用途说明。')
        body = normalize_body({'kind': 'memory', 'title': text_value(value['title'], 100),
                               'text': text_value(value['description'], 500)})
        item = {'id': identifier(value['id']), 'title': body['title'], 'description': body['text']}
        if item['id'] == UNCATEGORIZED or any(v['id'] == item['id'] for v in result):
            raise RunError('knowledge_invalid_category', '分类编号重复或使用保留编号。')
        result.append(item)
    return result


def projected(row):
    return {key: row[key] for key in PROJECTED_FIELDS if key in row}


def eligible(items, available_tools):
    tools = set(available_tools)
    return [row for row in items if row['scope_level'] != 'system'
            and (row['kind'] != 'skill' or set(row['payload']['tools']) <= tools)
            and (row['kind'] != 'example' or {s['verb'] for s in row['payload']['steps']} <= tools)]


def directory(items, definitions):
    groups = {item['id']: {**item, 'count': 0} for item in definitions}
    groups[UNCATEGORIZED] = {'id': UNCATEGORIZED, 'title': '未分类',
                            'description': '尚未归类的生效知识，需要时可继续读取。', 'count': 0}
    for row in items:
        for cid in disclosure_of(row)['categories'] or [UNCATEGORIZED]:
            if cid not in groups:
                raise RunError('knowledge_category_not_found', '知识引用的分类不可见，请刷新目录。')
            groups[cid]['count'] += 1
    return [groups[key] for key in sorted(groups) if groups[key]['count']]


def read_page(items, categories, *, entry_ids=(), offset=0, limit=20, auxiliary=()):
    from .knowledge import revision
    revision(offset)
    if type(limit) is not int or not 1 <= limit <= MAX_READ_ITEMS:
        raise RunError('knowledge_invalid_page', '每次读取须为一至六十四项。')
    if type(categories) is not list or len(categories) > 8 or any(type(v) is not str for v in categories):
        raise RunError('knowledge_invalid_category', '每次最多读取八个分类。')
    rows = [*items, *auxiliary]
    index = {row['id']: row for row in rows}
    if len(entry_ids) > 8 or any(type(eid) is not str or eid not in index for eid in entry_ids):
        raise RunError('knowledge_not_found', '条目不存在或不在当前可见范围。')
    selected = [row for row in rows if row['id'] in entry_ids or
                (row['scope_level'] != 'system' and
                 set(disclosure_of(row)['categories'] or [UNCATEGORIZED]) & set(categories))]
    selected.sort(key=lambda row: row['id'])
    page, used = [], 0
    for row in selected[offset:offset + limit]:
        item = projected(row)
        size = len(json.dumps(item, ensure_ascii=False))
        if used + size > MAX_READ_CHARS:
            if not page:
                raise RunError('knowledge_read_limit', '单条知识超过读取预算，请在知识窗口检查该条内容。')
            break
        page.append(item)
        used += size
    seen = {row['id'] for row in page}
    queue = [eid for row in page for eid in row['links']]
    dependencies, pending, unavailable = [], [], []
    for eid in queue:
        if eid in seen:
            continue
        seen.add(eid)
        if eid not in index:
            unavailable.append(eid)
            continue
        item = projected(index[eid])
        size = len(json.dumps(item, ensure_ascii=False))
        if used + size > MAX_READ_CHARS:
            pending.append(eid)
            continue
        dependencies.append(item)
        used += size
        queue.extend(item['links'])
    next_offset = offset + len(page)
    return {'items': page, 'dependency_items': dependencies, 'pending_dependency_ids': pending,
            'unavailable_dependency_ids': unavailable, 'total': len(selected),
            'next_offset': next_offset if next_offset < len(selected) else None}
