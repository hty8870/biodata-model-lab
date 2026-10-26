"""共享会话库中的统一知识持久层；短事务与原档保留迁移，不另建数据库。"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..agent.knowledge import digest, normalize_body, identifier, revision, text_value
from ..agent.learning_scope import bank_transaction, canonical_principal
from ..agent.memory_relations import _redact
from .conversation_runs import RunError
from .workspace_store import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_entries (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,id TEXT NOT NULL,
 project_id TEXT NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,revision INTEGER NOT NULL,
 data_json TEXT NOT NULL,PRIMARY KEY(owner,endpoint_fp,id)
);
CREATE TABLE IF NOT EXISTS knowledge_changes (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,id TEXT NOT NULL,project_id TEXT NOT NULL,
 request_id TEXT NOT NULL,request_fingerprint TEXT NOT NULL,status TEXT NOT NULL,
 revision INTEGER NOT NULL,fingerprint TEXT NOT NULL,data_json TEXT NOT NULL,
 PRIMARY KEY(owner,endpoint_fp,id),UNIQUE(owner,endpoint_fp,request_id)
);
CREATE TABLE IF NOT EXISTS knowledge_assessments (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,id TEXT NOT NULL,change_id TEXT NOT NULL,
 data_json TEXT NOT NULL,PRIMARY KEY(owner,endpoint_fp,id)
);
CREATE TABLE IF NOT EXISTS knowledge_catalog (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(owner,endpoint_fp)
);
CREATE TABLE IF NOT EXISTS knowledge_audit (
 id TEXT PRIMARY KEY,owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,project_id TEXT NOT NULL,
 data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_migration (
 version INTEGER PRIMARY KEY,data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_legacy_map (
 filename TEXT NOT NULL,line INTEGER NOT NULL,row_hash TEXT NOT NULL,legacy_id TEXT NOT NULL,
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,target_id TEXT NOT NULL,disposition TEXT NOT NULL,
 PRIMARY KEY(filename,line)
);
"""
LEGACY_FILES = ('collab_rules.jsonl', 'curate_examples.jsonl', 'curate_example_candidates.jsonl')
MAX_MIGRATION_BYTES = 8 * 1024 * 1024


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _safe_legacy(value, depth=0):
    if depth > 8:
        return '[unavailable]'
    if type(value) is str:
        return _redact(value, None)
    if type(value) is list:
        return [_safe_legacy(item, depth + 1) for item in value]
    if type(value) is dict:
        from .conversation_runs import safe_value
        return {str(k): _safe_legacy(v, depth + 1) for k, v in safe_value(value).items()}
    return value if value is None or type(value) in {bool, int, float} else '[unavailable]'


class KnowledgeStore:
    def __init__(self, conversations):
        self.conversations = conversations
        self.path = conversations.path
        self.root = Path(conversations.project_root)
        with self.conversations.database(write=True) as conn:
            for statement in SCHEMA.split(';'):
                if statement.strip():
                    conn.execute(statement)
        self.migrate()

    @contextmanager
    def database(self, *, write=False):
        with self.conversations.database(write=write) as conn:
            if not write:
                workspace = self.conversations.workspace
                if workspace.path.resolve() != self.path.resolve():
                    conn.execute('ATTACH DATABASE ? AS workspace', (str(workspace.path),))
                if not conn.in_transaction:
                    conn.execute('BEGIN')
            yield conn

    @staticmethod
    def generation(conn, owner, endpoint_fp):
        row = conn.execute('SELECT generation FROM knowledge_catalog WHERE owner=? AND endpoint_fp=?',
            (owner, endpoint_fp)).fetchone()
        return row['generation'] if row else 0

    @staticmethod
    def advance(conn, owner, endpoint_fp):
        conn.execute('INSERT INTO knowledge_catalog VALUES(?,?,1) ON CONFLICT(owner,endpoint_fp) DO UPDATE SET generation=generation+1',
            (owner, endpoint_fp))
        return KnowledgeStore.generation(conn, owner, endpoint_fp)

    @staticmethod
    def entries(conn, owner, endpoint_fp, *, active=False):
        query = 'SELECT data_json FROM knowledge_entries WHERE owner=? AND endpoint_fp=?'
        if active:
            query += " AND status='active'"
        return [json.loads(r['data_json']) for r in conn.execute(query + ' ORDER BY id', (owner, endpoint_fp))]

    @staticmethod
    def entry(conn, owner, endpoint_fp, entry_id):
        row = conn.execute('SELECT data_json FROM knowledge_entries WHERE owner=? AND endpoint_fp=? AND id=?',
            (owner, endpoint_fp, entry_id)).fetchone()
        if row is None:
            raise RunError('knowledge_not_found', '没有这条知识。')
        return json.loads(row['data_json'])

    @staticmethod
    def put_entry(conn, owner, endpoint_fp, row):
        conn.execute('INSERT INTO knowledge_entries VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(owner,endpoint_fp,id) DO UPDATE SET project_id=excluded.project_id,kind=excluded.kind,status=excluded.status,revision=excluded.revision,data_json=excluded.data_json',
            (owner, endpoint_fp, row['id'], row['project_id'], row['kind'], row['status'], row['revision'], encode(row)))

    @staticmethod
    def change(conn, owner, endpoint_fp, change_id):
        row = conn.execute('SELECT data_json FROM knowledge_changes WHERE owner=? AND endpoint_fp=? AND id=?',
            (owner, endpoint_fp, change_id)).fetchone()
        if row is None:
            raise RunError('knowledge_change_not_found', '没有这份知识提案。')
        return json.loads(row['data_json'])

    @staticmethod
    def put_change(conn, owner, endpoint_fp, change):
        conn.execute('INSERT INTO knowledge_changes VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(owner,endpoint_fp,id) DO UPDATE SET status=excluded.status,revision=excluded.revision,fingerprint=excluded.fingerprint,data_json=excluded.data_json',
            (owner, endpoint_fp, change['id'], change['project_id'], change['request_id'], change['request_fingerprint'],
             change['status'], change['revision'], change['fingerprint'], encode(change)))

    @staticmethod
    def audit(conn, owner, endpoint_fp, project_id, action, **data):
        row = {'id': 'ka_' + uuid.uuid4().hex, 'action': action, 'created_at': now_iso(), **data}
        conn.execute('INSERT INTO knowledge_audit VALUES(?,?,?,?,?)',
            (row['id'], owner, endpoint_fp, project_id, encode(_safe_legacy(row))))
        return row

    def migration(self, *, conn=None):
        if conn is None:
            with self.database() as db:
                return self.migration(conn=db)
        row = conn.execute('SELECT data_json FROM knowledge_migration WHERE version=1').fetchone()
        return json.loads(row['data_json']) if row else {'state': 'pending'}

    def resolve_legacy_id(self, owner, endpoint_fp, legacy_id, *, filename=''):
        """把迁移前旧账本编号解析成统一知识编号（只读，不写）。

        返回 target_id 字符串。目标有两类，由旧行类型决定、互不混合：
        规则/示例行（disposition active/retracted）指向知识条目 id，可传给
        KnowledgeService.entry()；候选行（disposition stale/committed/rejected）
        指向提案 id，可传给 KnowledgeService.change()。调用方按业务面二选一使用，
        拿错表会得到对应的 not_found。disposition=quarantined 的行永不解析。
        同一 legacy_id 命中多条非隔离记录（跨原账本或同文件多行）时抛
        knowledge_legacy_id_ambiguous，可用 filename 收敛到单份原账本；
        无命中抛 knowledge_not_found。
        """
        owner = identifier(canonical_principal(identifier(owner)))
        endpoint_fp = identifier(endpoint_fp, empty=True)
        legacy_id = identifier(legacy_id)
        if filename and filename not in LEGACY_FILES:
            raise RunError('knowledge_invalid_legacy_file', '不支持这个原账本。')
        with self.database() as conn:
            rows = conn.execute('SELECT target_id FROM knowledge_legacy_map WHERE owner=? AND endpoint_fp=? AND legacy_id=? AND (?=\'\' OR filename=?) AND disposition<>\'quarantined\'',
                (owner, endpoint_fp, legacy_id, filename, filename)).fetchall()
        if len(rows) != 1:
            raise RunError('knowledge_legacy_id_ambiguous' if rows else 'knowledge_not_found',
                '原编号未命中唯一迁移记录；请刷新统一知识编号。')
        return rows[0]['target_id']

    def migrate(self):
        if self.migration()['state'] == 'complete':
            return
        from .runtime_paths import instance_data_dir_for
        directory = instance_data_dir_for(self.root, '.userdata')
        with bank_transaction(self.root):
            raw_files = []
            size = 0
            for name in LEGACY_FILES:
                path = directory / name
                raw = path.read_bytes() if path.is_file() else b''
                size += len(raw)
                if size > MAX_MIGRATION_BYTES:
                    raise RunError('knowledge_migration_limit', '原账本超过迁移预算；尚未切换，原档不变。')
                raw_files.append((name, raw, path.is_file()))
            prepared = []
            for name, raw, exists in raw_files:
                try:
                    lines = raw.decode('utf-8-sig').splitlines()
                except UnicodeError:
                    raise RunError('knowledge_migration_invalid', '原账本编码无效；尚未切换，原档不变。') from None
                for number, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    import hashlib
                    row_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                    try:
                        row = json.loads(line)
                        prepared.append(self._legacy_record(name, number, row_hash, row))
                    except (ValueError, TypeError, KeyError, RunError) as exc:
                        reason = exc.code if isinstance(exc, RunError) else (str(exc) or type(exc).__name__)
                        prepared.append({'filename': name, 'line': number, 'row_hash': row_hash,
                            'legacy_id': '', 'owner': 'anonymous', 'endpoint_fp': '', 'target_id': '',
                            'disposition': 'quarantined', 'quarantine_reason': str(reason)[:120]})
            with self.database(write=True) as conn:
                if self.migration(conn=conn)['state'] == 'complete':
                    return
                touched = set()
                for item in prepared:
                    mapping = {key: item[key] for key in ('filename', 'line', 'row_hash', 'legacy_id',
                               'owner', 'endpoint_fp', 'target_id', 'disposition')}
                    conn.execute('INSERT INTO knowledge_legacy_map VALUES(?,?,?,?,?,?,?,?)', tuple(mapping.values()))
                    owner, endpoint = item['owner'], item['endpoint_fp']
                    if 'entry' in item:
                        self.put_entry(conn, owner, endpoint, item['entry'])
                        touched.add((owner, endpoint))
                    if 'change' in item:
                        self.put_change(conn, owner, endpoint, item['change'])
                        touched.add((owner, endpoint))
                    self.audit(conn, owner, endpoint, item.get('project_id', ''), 'migration', mapping=mapping,
                        quarantine_reason=item.get('quarantine_reason'), legacy=item.get('legacy_audit'))
                for owner, endpoint in touched:
                    self.advance(conn, owner, endpoint)
                import hashlib
                files = [{'filename': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'existed': exists}
                         for name, raw, exists in raw_files]
                for name, raw, exists in raw_files:
                    path = directory / name
                    if path.is_file() != exists or (exists and path.read_bytes() != raw):
                        raise RunError('knowledge_migration_conflict', '迁移期间原账本变化；全部回滚，原档保留。')
                record = {'state': 'complete', 'version': 1, 'files': files, 'records': len(prepared),
                          'quarantined': sum(r['disposition'] == 'quarantined' for r in prepared),
                          'quarantine': [{'filename': r['filename'], 'line': r['line'],
                                          'reason': r['quarantine_reason']}
                                         for r in prepared if r['disposition'] == 'quarantined'][:50],
                          'switched_at': now_iso(), 'legacy_write_disabled': True}
                conn.execute('INSERT INTO knowledge_migration VALUES(1,?)', (encode(record),))

    @staticmethod
    def _legacy_record(name, number, row_hash, raw):
        if type(raw) is not dict:
            raise ValueError('invalid_legacy')
        owner = identifier(canonical_principal(raw.get('principal') or 'anonymous'))
        endpoint = identifier(raw.get('endpoint_fp') or '', empty=True)
        project = identifier(raw.get('project_id') or '', empty=True)
        old_id = text_value(str(raw.get('id') or ''), 128, empty=True)
        target = 'kl_' + digest([name, number, row_hash])[:32]
        base = {'filename': name, 'line': number, 'row_hash': row_hash, 'legacy_id': old_id,
                'owner': owner, 'endpoint_fp': endpoint, 'project_id': project, 'target_id': target}
        record_type = raw.get('record_type', 'rule')
        if type(record_type) is not str or record_type not in {'rule', 'candidate', 'audit'}:
            raise ValueError('invalid_legacy_record_type')
        if record_type == 'audit':
            return dict(base, disposition='audit', legacy_audit=_safe_legacy(raw))
        example = name != 'collab_rules.jsonl'
        payload = {'utterance': raw.get('utterance'), 'steps': raw.get('steps'), 'verified_success': False} if example else {}
        body = normalize_body({'kind': 'example' if example else 'memory', 'title': '',
            'text': raw.get('utterance') if example else raw.get('text'), 'trigger': raw.get('trigger', ''),
            'tier': raw.get('tier', 'explicit'), 'permanence': 'persistent', 'payload': payload, 'links': []})
        provenance = {'type': 'legacy', 'filename': name, 'line': number, 'row_hash': row_hash,
                      'legacy_id': old_id, 'quotes': [_redact(str(q), None) for q in raw.get('evidence', [])]
                      if type(raw.get('evidence', [])) is list else []}
        pending = record_type == 'candidate' or name == 'curate_example_candidates.jsonl'
        now = now_iso()
        if pending:
            refs = []
            sources = raw.get('sources', [])
            if type(sources) is list:
                for source in sources:
                    if type(source) is dict and type(source.get('id')) is str:
                        refs.append({'type': 'learning', 'id': identifier(source['id'])})
            old_status = raw.get('status', 'pending')
            statuses = {'pending': 'stale', 'approved': 'committed', 'rejected': 'rejected'}
            if old_status not in statuses:
                raise ValueError('invalid_legacy_status')
            change = {'id': target, 'request_id': 'legacy:' + target, 'project_id': project,
                'request_fingerprint': row_hash, 'status': statuses[old_status], 'revision': 1,
                'operations': [{'op': 'create', 'entry_id': target + '_entry', 'after': body}],
                'source_refs': refs, 'actor': 'legacy', 'created_at': now, 'updated_at': now,
                'legacy_provenance': provenance, 'assessment_id': None}
            change['fingerprint'] = digest(change)
            return dict(base, disposition=statuses[old_status], change=change, legacy_audit=_safe_legacy(raw))
        status = raw.get('status', 'active')
        if status not in {'active', 'retracted'}:
            raise ValueError('invalid_legacy_status')
        entry = {**body, 'id': target, 'principal': owner, 'endpoint_fp': endpoint,
                 'scope_level': 'project' if project else 'user', 'project_id': project,
                 'status': status, 'revision': revision(raw.get('revision', 1)), 'provenance': provenance,
                 'created_at': text_value(str(raw.get('created_at') or raw.get('ts') or now), 80),
                 'updated_at': now, 'confirmed_at': raw.get('confirmed_at'), 'retracted_at': raw.get('retracted_at')}
        return dict(base, disposition=status, entry=entry)
