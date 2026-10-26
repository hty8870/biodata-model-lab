"""学习事件、授权与执行回执；与共享会话共用 SQLite，凭据不入库。"""
from __future__ import annotations

import json
import uuid

from .conversation_runs import RunError, _json
from .workspace_store import now_iso, owner_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_sources (
 owner TEXT NOT NULL,id TEXT NOT NULL,endpoint_fp TEXT NOT NULL,project_id TEXT NOT NULL,
 conversation_id TEXT NOT NULL,origin_run_id TEXT NOT NULL,kind TEXT NOT NULL,
 text TEXT NOT NULL,outcome TEXT NOT NULL,valid INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL,
 PRIMARY KEY(owner,id)
);
CREATE TABLE IF NOT EXISTS learning_scopes (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,project_id TEXT NOT NULL,
 enabled INTEGER NOT NULL DEFAULT 0,epoch INTEGER NOT NULL DEFAULT 0,threshold INTEGER NOT NULL DEFAULT 3,
 PRIMARY KEY(owner,endpoint_fp,project_id)
);
CREATE TABLE IF NOT EXISTS learning_jobs (
 owner TEXT NOT NULL,id TEXT NOT NULL,endpoint_fp TEXT NOT NULL,project_id TEXT NOT NULL,
 epoch INTEGER NOT NULL,status TEXT NOT NULL,sources_json TEXT NOT NULL,
 result_json TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 PRIMARY KEY(owner,id)
);
CREATE TABLE IF NOT EXISTS learning_consumed (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,project_id TEXT NOT NULL,source_id TEXT NOT NULL,
 PRIMARY KEY(owner,endpoint_fp,project_id,source_id)
);
CREATE TABLE IF NOT EXISTS learning_trace (
 owner TEXT NOT NULL,job_id TEXT NOT NULL,seq INTEGER NOT NULL,data_json TEXT NOT NULL,
 PRIMARY KEY(owner,job_id,seq)
);
CREATE TABLE IF NOT EXISTS learning_usage (
 owner TEXT NOT NULL,day TEXT NOT NULL,calls INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(owner,day)
);
CREATE TABLE IF NOT EXISTS learning_config_epochs (
 owner TEXT NOT NULL,endpoint_fp TEXT NOT NULL,epoch INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(owner,endpoint_fp)
);
"""


def initialize(conn):
    for statement in SCHEMA.split(';'):
        if statement.strip():
            conn.execute(statement)
    additions = {'manual': 'INTEGER NOT NULL DEFAULT 0', 'attempt': 'INTEGER NOT NULL DEFAULT 0',
        'request_id': "TEXT NOT NULL DEFAULT ''", 'config_epoch': 'INTEGER NOT NULL DEFAULT 0',
        'selection_json': "TEXT NOT NULL DEFAULT '[]'", 'completed_json': "TEXT NOT NULL DEFAULT '[]'",
        'read_json': "TEXT NOT NULL DEFAULT '[]'", 'consumed_json': "TEXT NOT NULL DEFAULT '[]'"}
    columns = {row['name'] for row in conn.execute('PRAGMA table_info(learning_jobs)')}
    if set(additions) - columns:
        if not conn.in_transaction:
            conn.execute('BEGIN IMMEDIATE')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(learning_jobs)')}
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f'ALTER TABLE learning_jobs ADD COLUMN {name} {declaration}')


def register_turn(conn, row, request, status):
    if not request.get('learning_endpoint_fp'):
        return
    conn.execute('INSERT OR IGNORE INTO learning_sources VALUES(?,?,?,?,?,?,?,?,?,?,?)',
        (row['owner'], 'run:' + row['id'], request['learning_endpoint_fp'], row['project_id'],
         row['conversation_id'], row['id'], 'turn', str(request.get('text') or '')[:4000], status, 1, now_iso()))


class LearningStore:
    def __init__(self, conversation_store):
        self.conversations = conversation_store
        self.path = conversation_store.path

    def scope(self, owner, endpoint_fp, project_id='', *, conn=None):
        def load(db):
            row = db.execute('SELECT * FROM learning_scopes WHERE owner=? AND endpoint_fp=? AND project_id=?',
                (owner_id(owner), endpoint_fp, project_id)).fetchone()
            return dict(row) if row else {'owner': owner_id(owner), 'endpoint_fp': endpoint_fp,
                'project_id': project_id, 'enabled': 0, 'epoch': 0, 'threshold': 3}
        if conn is not None:
            return load(conn)
        with self.conversations.database() as db:
            return load(db)

    def config_epoch(self, owner, endpoint_fp, *, conn=None):
        def load(db):
            row = db.execute('SELECT epoch FROM learning_config_epochs WHERE owner=? AND endpoint_fp=?',
                (owner_id(owner), endpoint_fp)).fetchone()
            return row['epoch'] if row else 0
        if conn is not None:
            return load(conn)
        with self.conversations.database() as db:
            return load(db)

    def config_epochs(self, owner, *, conn=None):
        def load(db):
            return {row['endpoint_fp']: row['epoch'] for row in db.execute(
                'SELECT endpoint_fp,epoch FROM learning_config_epochs WHERE owner=?', (owner_id(owner),))}
        if conn is not None:
            return load(conn)
        with self.conversations.database() as db:
            return load(db)

    def authorization_snapshot(self, owner):
        with self.conversations.database() as conn:
            if not conn.in_transaction:
                conn.execute('BEGIN')
            config_epochs = self.config_epochs(owner, conn=conn)
            scope_epochs = {(row['endpoint_fp'], row['project_id']): row['epoch'] for row in conn.execute(
                'SELECT endpoint_fp,project_id,epoch FROM learning_scopes WHERE owner=?', (owner_id(owner),))}
            return {'config_epochs': config_epochs, 'scope_epochs': scope_epochs}

    def authorization(self, owner, endpoint_fp, project_id='', *, expected_config_epoch=None,
                      expected_scope_epoch=None, conn=None):
        def load(db):
            epoch = self.config_epoch(owner, endpoint_fp, conn=db)
            if expected_config_epoch is not None and (type(expected_config_epoch) is not int or expected_config_epoch < 0 or epoch != expected_config_epoch):
                raise RunError('learning_config_conflict', 'AI配置授权已变化，请刷新状态后重新操作。')
            scope = self.scope(owner, endpoint_fp, project_id, conn=db)
            if expected_scope_epoch is not None and scope['epoch'] != expected_scope_epoch:
                raise RunError('learning_config_conflict', '学习范围授权已变化，请刷新状态后重新操作。')
            return scope, epoch
        if conn is not None:
            return load(conn)
        with self.conversations.database() as db:
            if not db.in_transaction:
                db.execute('BEGIN')
            return load(db)

    def release_configs(self, owner, endpoint_fp, *, expected_config_epoch=None):
        owner = owner_id(owner)
        with self.conversations.database(write=True) as conn:
            _, current = self.authorization(owner, endpoint_fp, expected_config_epoch=expected_config_epoch, conn=conn)
            epoch = current + 1
            conn.execute('INSERT INTO learning_config_epochs VALUES(?,?,?) ON CONFLICT(owner,endpoint_fp) DO UPDATE SET epoch=excluded.epoch',
                (owner, endpoint_fp, epoch))
            conn.execute('UPDATE learning_scopes SET epoch=epoch+1 WHERE owner=? AND endpoint_fp=?', (owner, endpoint_fp))
            result = {'ok': False, 'error_code': 'learning_config_released',
                'detail': '旧AI配置授权已释放，等待当前账户的新授权配置；已有候选和长期学习开关保留。'}
            conn.execute("UPDATE learning_jobs SET status='paused',result_json=?,updated_at=? WHERE owner=? AND endpoint_fp=? AND status IN ('queued','running','waiting_config')",
                (_json(result), now_iso(), owner, endpoint_fp))
        return epoch

    def control(self, owner, endpoint_fp, project_id='', *, enabled, threshold=3,
                expected_config_epoch=None, expected_scope_epoch=None):
        if type(enabled) is not bool or type(threshold) is not int or not 1 <= threshold <= 20:
            raise RunError('invalid_learning_control', '学习开关须为布尔值，间隔须为 1–20 个新回合。')
        with self.conversations.database(write=True) as conn:
            current, _ = self.authorization(owner, endpoint_fp, project_id,
                expected_config_epoch=expected_config_epoch, expected_scope_epoch=expected_scope_epoch, conn=conn)
            epoch = current['epoch'] + 1
            conn.execute('INSERT INTO learning_scopes VALUES(?,?,?,?,?,?) ON CONFLICT(owner,endpoint_fp,project_id) DO UPDATE SET enabled=excluded.enabled,epoch=excluded.epoch,threshold=excluded.threshold',
                (owner_id(owner), endpoint_fp, project_id, int(enabled), epoch, int(threshold)))
            conn.execute("UPDATE learning_jobs SET status='paused',updated_at=? WHERE owner=? AND endpoint_fp=? AND project_id=? AND status IN ('queued','waiting_config','running')",
                (now_iso(), owner, endpoint_fp, project_id))
            return self.scope(owner, endpoint_fp, project_id, conn=conn)

    def evidence(self, owner, endpoint_fp, project_id='', *, fresh=False, limit=12, offset=0):
        query = 'SELECT s.* FROM learning_sources s WHERE s.owner=? AND s.endpoint_fp=? AND s.project_id=? AND s.valid=1'
        if fresh:
            query += ' AND NOT EXISTS (SELECT 1 FROM learning_consumed c WHERE c.owner=s.owner AND c.endpoint_fp=s.endpoint_fp AND c.project_id=s.project_id AND c.source_id=s.id)'
        query += ' ORDER BY s.created_at,s.id LIMIT ? OFFSET ?'
        with self.conversations.database() as conn:
            rows = conn.execute(query, (owner_id(owner), endpoint_fp, project_id, min(12, max(1, int(limit))), max(0, int(offset)))).fetchall()
        return [dict(row) for row in rows]

    def feedback(self, owner, endpoint_fp, project_id, *, request_id, text, conversation_id='',
                 expected_config_epoch=None, expected_scope_epoch=None):
        if not request_id or not str(text or '').strip() or len(text) > 2000:
            raise RunError('invalid_learning_feedback', '反馈需要稳定编号和 1–2000 个字符的原话。')
        from ..agent.collab_rules import content_fingerprint
        sid = 'feedback:' + content_fingerprint([owner, endpoint_fp, project_id, request_id])[:32]
        with self.conversations.database(write=True) as conn:
            self.authorization(owner, endpoint_fp, project_id, expected_config_epoch=expected_config_epoch,
                expected_scope_epoch=expected_scope_epoch, conn=conn)
            if conversation_id:
                self.conversations._row(conn, owner_id(owner), conversation_id)
            prior = conn.execute('SELECT text FROM learning_sources WHERE owner=? AND id=?', (owner, sid)).fetchone()
            if prior and prior['text'] != text:
                raise RunError('learning_feedback_conflict', '同一反馈编号不能改写原话。')
            from .conversation_runs import safe_value
            conn.execute('INSERT OR IGNORE INTO learning_sources VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (owner_id(owner), sid, endpoint_fp, project_id, conversation_id, '', 'feedback', safe_value(text), 'feedback', 1, now_iso()))
        return sid

    def sources(self, owner, endpoint_fp, project_id=None, *, offset=0, limit=100):
        """本人有效来源清单，保留原项目与消费状态。"""
        from ..agent.collab_rules import content_fingerprint
        owner = owner_id(owner)
        if type(limit) is not int or not 1 <= limit <= 200 or type(offset) is not int or offset < 0:
            raise RunError('invalid_learning_page', '来源分页参数无效。')
        query = '''SELECT s.id,s.project_id,s.conversation_id,s.origin_run_id,s.kind,s.text,s.outcome,s.created_at,
            COALESCE(c.title,'') AS title,
            CASE WHEN EXISTS(SELECT 1 FROM learning_consumed x WHERE x.owner=s.owner AND x.endpoint_fp=s.endpoint_fp
                AND x.project_id=s.project_id AND x.source_id=s.id) THEN 0 ELSE 1 END AS fresh
            FROM learning_sources s LEFT JOIN shared_conversations c ON c.owner=s.owner AND c.id=s.conversation_id
            WHERE s.owner=? AND s.endpoint_fp=? AND s.valid=1 AND (s.conversation_id='' OR c.deleted=0)'''
        params = [owner, endpoint_fp]
        if project_id is not None:
            query += ' AND s.project_id=?'
            params.append(project_id)
        query += ' ORDER BY s.created_at,s.id LIMIT ? OFFSET ?'
        params += [limit + 1, offset]
        with self.conversations.database() as conn:
            rows = conn.execute(query, params).fetchall()
        items = [{'id': r['id'], 'project_id': r['project_id'], 'conversation_id': r['conversation_id'],
                  'title': r['title'], 'kind': r['kind'], 'text_preview': str(r['text'] or '')[:200],
                  'fresh': bool(r['fresh']),
                  'source_hash': content_fingerprint([r['id'], r['kind'], r['text'], r['project_id'],
                                                      r['conversation_id'], r['origin_run_id']])[:32]}
                 for r in rows[:limit]]
        return {'items': items, 'next_offset': offset + limit if len(rows) > limit else None}

    def sources_by_ids(self, owner, endpoint_fp, ids, *, conn=None):
        """按编号读取本人有效来源，保留原作用域供核销。"""
        ids = list(ids or [])
        if not ids:
            return []
        def load(db):
            placeholders = ','.join('?' for _ in ids)
            rows = db.execute('SELECT * FROM learning_sources WHERE owner=? AND endpoint_fp=? AND valid=1 AND id IN (' + placeholders + ') ORDER BY created_at,id',
                              (owner_id(owner), endpoint_fp, *ids)).fetchall()
            return [dict(r) for r in rows]
        if conn is not None:
            return load(conn)
        with self.conversations.database() as db:
            return load(db)

    def materialize(self, owner, endpoint_fp, project_id, materials, *, expected_config_epoch=None,
                    expected_scope_epoch=None, conn=None, allow_cross_project=False):
        """显式材料按编号及正文哈希保存，不伪造执行记录。

        默认材料 project_id 必须与目标作用域一致；仅在用户层一次性手动汇总
        （allow_cross_project=True，调用方已核 manual+显式选择）时允许材料保留
        自己的原项目编号——出处保原项目，核销仍在来源原 scope。"""
        from ..agent.collab_rules import content_fingerprint
        from .conversation_runs import safe_value
        if type(allow_cross_project) is not bool:
            raise RunError('invalid_learning_material', '跨项目标记须为严格布尔值。')
        def insert(db):
            self.authorization(owner, endpoint_fp, project_id, expected_config_epoch=expected_config_epoch,
                               expected_scope_epoch=expected_scope_epoch, conn=db)
            out = []
            for material in materials:
                if type(material) is not dict or set(material) - {'client_id', 'text', 'project_id', 'conversation_id'}:
                    raise RunError('invalid_learning_material', '材料字段无效。')
                client_id, text = material.get('client_id'), material.get('text')
                if type(client_id) is not str or not client_id or len(client_id) > 128 or type(text) is not str or not text.strip() or len(text) > 4000:
                    raise RunError('invalid_learning_material', '材料需要有界编号及 1–4000 字原文。')
                material_project = material.get('project_id', '')
                if material_project != project_id and not (allow_cross_project and not project_id):
                    raise RunError('learning_scope_conflict', '材料项目须与目标一致；跨项目只用于用户层一次性汇总。')
                conversation_id = material.get('conversation_id', '')
                if conversation_id:
                    self.conversations._row(db, owner_id(owner), conversation_id)
                sid = 'user:' + content_fingerprint([owner, endpoint_fp, client_id, text, material_project, conversation_id])[:32]
                db.execute('INSERT OR IGNORE INTO learning_sources VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                           (owner_id(owner), sid, endpoint_fp, material_project, conversation_id, '', 'user_provided',
                            safe_value(text), '', 1, now_iso()))
                out.append(dict(db.execute('SELECT * FROM learning_sources WHERE owner=? AND id=?',
                                           (owner_id(owner), sid)).fetchone()))
            return out
        if conn is not None:
            return insert(conn)
        with self.conversations.database(write=True) as db:
            return insert(db)

    def snapshot_conversations(self, owner, endpoint_fp, items, *, target_project='', expected_config_epoch=None,
                               expected_scope_epoch=None, conn=None):
        """本人显式对话固定快照；有界预算不足则拒绝，不截断成伪完整来源。"""
        from ..agent.collab_rules import content_fingerprint
        from .conversation_runs import safe_value
        def insert(db):
            self.authorization(owner, endpoint_fp, target_project, expected_config_epoch=expected_config_epoch,
                               expected_scope_epoch=expected_scope_epoch, conn=db)
            out = []
            for conv_id, project_id in items:
                schema = self.conversations.workspace.membership_schema(db)
                membership = db.execute(f'SELECT project_id FROM {schema}.conversations WHERE owner=? AND id=?',
                    (owner_id(owner), conv_id)).fetchone()
                if (membership['project_id'] if membership else '') != project_id:
                    raise RunError('learning_scope_conflict', '对话项目归属已变化，请刷新后重新选择。')
                messages = self.conversations.learning_messages(owner, conv_id, conn=db)
                text = '\n'.join('%s：%s' % (m['role'], m['text']) for m in messages)
                if not text.strip():
                    continue
                if len(text) > 4000:
                    raise RunError('learning_material_limit', '所选对话超过单份核验预算，请提交明确分段材料。')
                sid = 'user:' + content_fingerprint([owner, endpoint_fp, conv_id, project_id, text])[:32]
                db.execute('INSERT OR IGNORE INTO learning_sources VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                           (owner_id(owner), sid, endpoint_fp, project_id, conv_id, '', 'user_provided',
                            safe_value(text), '', 1, now_iso()))
                out.append(dict(db.execute('SELECT * FROM learning_sources WHERE owner=? AND id=?',
                                           (owner_id(owner), sid)).fetchone()))
            return out
        if conn is not None:
            return insert(conn)
        with self.conversations.database(write=True) as db:
            return insert(db)

    def fresh_stats(self, owner, endpoint_fp, project_id='', *, kinds=None, conn=None):
        kinds = None if kinds is None else tuple(kinds)
        def count(db):
            query = '''SELECT COUNT(*) AS total,COALESCE(SUM(CASE WHEN s.kind='feedback' THEN 1 ELSE 0 END),0) AS feedback
                FROM learning_sources s WHERE s.owner=? AND s.endpoint_fp=? AND s.project_id=? AND s.valid=1
                AND NOT EXISTS (SELECT 1 FROM learning_consumed c WHERE c.owner=s.owner AND c.endpoint_fp=s.endpoint_fp AND c.project_id=s.project_id AND c.source_id=s.id)'''
            params = [owner, endpoint_fp, project_id]
            if kinds:
                query += ' AND s.kind IN (' + ','.join('?' for _ in kinds) + ')'
                params += list(kinds)
            return dict(db.execute(query, params).fetchone())
        if conn is not None:
            return count(conn)
        with self.conversations.database() as db:
            return count(db)

    def source_snapshot(self, conn, row):
        from ..agent.collab_rules import content_fingerprint
        if not row['valid']:
            raise RunError('learning_source_invalidated', '来源已经失效。')
        if len(row['text']) > 4000:
            raise RunError('learning_source_budget', '完整来源超过单份核验预算；请明确分段，不会截断后核销。')
        workspace = self.conversations.workspace
        if workspace.path.resolve() != self.path.resolve() and workspace.membership_schema(conn) != 'workspace':
            conn.execute('ATTACH DATABASE ? AS workspace', (str(workspace.path),))
        schema = workspace.membership_schema(conn)
        current_project = row['project_id']
        if row['conversation_id']:
            conversation = self.conversations._row(conn, row['owner'], row['conversation_id'])
            if conversation['archived']:
                raise RunError('learning_source_archived', '来源对话已归档，请恢复后再整理。')
            member = conn.execute(f'SELECT project_id FROM {schema}.conversations WHERE owner=? AND id=?',
                (row['owner'], row['conversation_id'])).fetchone()
            current_project = member['project_id'] if member else ''
        epochs = {}
        for project in {row['project_id'], current_project}:
            if project:
                project_row = conn.execute(f'SELECT archived FROM {schema}.conversation_projects WHERE owner=? AND id=?',
                    (row['owner'], project)).fetchone()
                if project_row is None:
                    raise RunError('learning_source_invalidated', '来源原项目已经不存在。')
                if project_row['archived']:
                    raise RunError('learning_source_archived', '来源项目已归档，请恢复后再整理。')
            epochs[project] = self.scope(row['owner'], row['endpoint_fp'], project, conn=conn)['epoch']
        run = conn.execute('SELECT status,fingerprint,result_json FROM conversation_runs WHERE owner=? AND id=?',
            (row['owner'], row['origin_run_id'])).fetchone() if row['origin_run_id'] else None
        material = {key: row[key] for key in ('id', 'owner', 'endpoint_fp', 'project_id', 'conversation_id',
            'origin_run_id', 'kind', 'text', 'outcome')}
        return {'id': row['id'], 'project_id': row['project_id'], 'scope_epochs': epochs,
            'source_hash': content_fingerprint([material, dict(run) if run else None, current_project])}

    def schedule(self, owner, endpoint_fp, project_id='', *, force=False, configured=False,
                 expected_config_epoch=None, expected_scope_epoch=None,
                 source_ids=None, manual=False, request_id=''):
        from ..agent.collab_rules import content_fingerprint
        owner = owner_id(owner)
        with self.conversations.database(write=True) as conn:
            scope, config_epoch = self.authorization(owner, endpoint_fp, project_id,
                expected_config_epoch=expected_config_epoch, expected_scope_epoch=expected_scope_epoch, conn=conn)
            if not scope['enabled'] and not manual:
                return None
            explicit = source_ids is not None
            if explicit:
                if len(source_ids) > 200:
                    raise RunError('learning_selection_limit', '一次固定选择最多200份来源。')
                rows = self.sources_by_ids(owner, endpoint_fp, source_ids, conn=conn)
                if {r['id'] for r in rows} != set(source_ids):
                    raise RunError('learning_source_invalidated', '选择含不存在或失效的来源。')
            else:
                rows = [dict(r) for r in conn.execute('''SELECT s.* FROM learning_sources s WHERE s.owner=? AND s.endpoint_fp=? AND s.project_id=? AND s.valid=1
                    AND NOT EXISTS (SELECT 1 FROM learning_consumed c WHERE c.owner=s.owner AND c.endpoint_fp=s.endpoint_fp AND c.project_id=s.project_id AND c.source_id=s.id)
                    ORDER BY s.created_at,s.id LIMIT 200''', (owner, endpoint_fp, project_id))]
                stats = self.fresh_stats(owner, endpoint_fp, project_id, kinds=('turn', 'feedback'), conn=conn)
                if not rows or (not force and stats['total'] < scope['threshold'] and not stats['feedback']):
                    return None
            selection = [self.source_snapshot(conn, row) for row in rows]
            if not selection:
                return None
            selection.sort(key=lambda item: item['id'])
            fingerprint = content_fingerprint([(r['id'], r['source_hash']) for r in selection])
            active = conn.execute("SELECT * FROM learning_jobs WHERE owner=? AND endpoint_fp=? AND project_id=? AND status IN ('queued','waiting_config','running') ORDER BY created_at LIMIT 1",
                (owner, endpoint_fp, project_id)).fetchone()
            if active:
                active = dict(active)
                try:
                    self.check(active, conn=conn)
                except RunError as exc:
                    conn.execute("UPDATE learning_jobs SET status='paused',result_json=?,updated_at=? WHERE owner=? AND id=?", (_json({'error_code': exc.code}), now_iso(), owner, active['id']))
                else:
                    same = content_fingerprint([(r['id'], r['source_hash']) for r in json.loads(active['selection_json'])]) == fingerprint
                    if explicit and not same:
                        raise RunError('learning_job_conflict', '另一份固定选择仍在处理；本次新选择未受理。')
                    return active
            rid = request_id or 'auto:' + uuid.uuid4().hex
            jid = 'lj_' + content_fingerprint([owner, endpoint_fp, project_id, rid])[:32]
            previous = conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND id=?', (owner, jid)).fetchone()
            completed, read, consumed, attempt = [], [], [], 0
            if previous:
                previous = dict(previous)
                old_selection = json.loads(previous['selection_json'])
                if content_fingerprint([(r['id'], r['source_hash']) for r in old_selection]) != fingerprint:
                    raise RunError('learning_request_conflict', '同一请求编号不能更换固定来源或改写材料。')
                completed, read, consumed = (json.loads(previous[key]) for key in ('completed_json', 'read_json', 'consumed_json'))
                if previous['status'] == 'done':
                    return previous
                attempt = previous['attempt'] + 1
            remaining = [r['id'] for r in selection if r['id'] not in completed]
            if not remaining:
                return dict(previous) if previous else None
            now = now_iso()
            state = 'queued' if configured else 'waiting_config'
            conn.execute('''INSERT INTO learning_jobs(owner,id,endpoint_fp,project_id,epoch,status,manual,sources_json,result_json,created_at,updated_at,
                request_id,config_epoch,selection_json,completed_json,read_json,consumed_json,attempt) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(owner,id) DO UPDATE SET epoch=excluded.epoch,status=excluded.status,updated_at=excluded.updated_at,
                config_epoch=excluded.config_epoch,selection_json=excluded.selection_json,sources_json=excluded.sources_json,
                attempt=excluded.attempt,result_json=NULL''',
                (owner, jid, endpoint_fp, project_id, scope['epoch'], state, int(manual), _json(remaining[:12]), None, now, now,
                 rid, config_epoch, _json(selection), _json(completed), _json(read), _json(consumed), attempt))
            return dict(conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND id=?', (owner, jid)).fetchone())

    def lock_path(self, job):
        return self.path.parent / (self.path.name + '.learning') / (job['id'] + '.lock')

    def check(self, job, *, conn=None):
        def verify(db):
            scope = self.scope(job['owner'], job['endpoint_fp'], job['project_id'], conn=db)
            row = db.execute('SELECT * FROM learning_jobs WHERE owner=? AND id=?', (job['owner'], job['id'])).fetchone()
            if row is None or row['attempt'] != job.get('attempt', 0):
                raise RunError('learning_job_superseded', '旧执行批次已结束，不能覆盖新的重试。')
            if self.config_epoch(job['owner'], job['endpoint_fp'], conn=db) != job.get('config_epoch', 0):
                raise RunError('learning_config_released', '旧配置授权已释放，迟到结果不会保存。')
            if (not scope['enabled'] and not bool(job.get('manual'))) or scope['epoch'] != job['epoch'] or row['status'] not in {'queued', 'running', 'waiting_config'}:
                raise RunError('learning_paused', '学习已暂停，迟到结果不会保存。')
            selection = json.loads(job.get('selection_json', '[]'))
            if not selection:
                raise RunError('learning_selection_required', '旧任务缺少完整来源快照，请重新明确选择。')
            for selected in selection:
                source = db.execute('SELECT * FROM learning_sources WHERE owner=? AND endpoint_fp=? AND id=?',
                    (job['owner'], job['endpoint_fp'], selected['id'])).fetchone()
                if source is None or self.source_snapshot(db, dict(source)) != selected:
                    raise RunError('learning_source_invalidated', '来源内容、原项目授权或归属已变化，请重新选择。')
            if job['project_id']:
                schema = self.conversations.workspace.membership_schema(db)
                target = db.execute(f'SELECT archived FROM {schema}.conversation_projects WHERE owner=? AND id=?',
                    (job['owner'], job['project_id'])).fetchone()
                if target is None or target['archived']:
                    raise RunError('learning_paused', '目标项目不存在或已归档。')
        if conn is not None:
            return verify(conn)
        with self.conversations.database() as db:
            return verify(db)

    def set_status(self, job, status, result=None, consumed_sources=None):
        """核销只认本批服务器工具读取及成功结束回执，不认模型提交的编号清单。"""
        with self.conversations.database(write=True) as conn:
            current = conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND id=?', (job['owner'], job['id'])).fetchone()
            if current is None or current['attempt'] != job.get('attempt', 0):
                return {'ok': False, 'error_code': 'learning_job_superseded'}
            if current['status'] in {'done', 'partial'} and status not in {'done', 'partial'}:
                return self.public_job(current)
            if status in {'queued', 'running', 'done', 'partial'}:
                self.check(job, conn=conn)
            if status in {'queued', 'running'}:
                conn.execute('UPDATE learning_jobs SET status=?,updated_at=? WHERE owner=? AND id=?',
                    (status, now_iso(), job['owner'], job['id']))
                return {'ok': True, 'state': status}
            selection = {row['id']: row for row in json.loads(current['selection_json'])}
            batch = set(json.loads(job['sources_json']))
            read = set(json.loads(current['read_json']))
            completed = set(json.loads(current['completed_json']))
            consumed = set(json.loads(current['consumed_json']))
            traces = [json.loads(row['data_json']) for row in conn.execute(
                'SELECT data_json FROM learning_trace WHERE owner=? AND job_id=? ORDER BY seq', (job['owner'], job['id']))]
            traces = [entry for entry in traces if entry.get('attempt', 0) == job.get('attempt', 0)]
            actual_read = set()
            for entry in traces:
                observation = entry.get('observation') or {}
                if entry.get('action', {}).get('tool') == 'read_evidence' and observation.get('ok') is True:
                    for evidence in observation.get('evidence', []):
                        sid = evidence.get('id')
                        if sid in batch and evidence.get('source_hash') == selection[sid]['source_hash']:
                            actual_read.add(sid)
            read.update(actual_read)
            if status in {'done', 'partial'}:
                final = traces[-1] if traces else {}
                proof = final.get('observation') or {}
                handled = set(proof.get('processed_source_ids', []))
                if (final.get('action', {}).get('tool') != 'finish' or proof.get('ok') is not True
                    or proof.get('finished') is not True or not handled or not handled <= actual_read):
                    status = 'failed'
                    result = {**(result or {}), 'ok': False, 'state': 'failed', 'error_code': 'learning_processing_receipt_required'}
                else:
                    for sid in handled:
                        row = conn.execute('SELECT * FROM learning_sources WHERE owner=? AND endpoint_fp=? AND id=?',
                            (job['owner'], job['endpoint_fp'], sid)).fetchone()
                        if row is None or self.source_snapshot(conn, dict(row)) != selection[sid]:
                            raise RunError('learning_source_invalidated', '已处理来源在核销前变化。')
                        inserted = conn.execute('INSERT OR IGNORE INTO learning_consumed VALUES(?,?,?,?)',
                            (job['owner'], job['endpoint_fp'], row['project_id'], sid)).rowcount
                        if inserted:
                            consumed.add(sid)
                    completed.update(handled)
                    status = 'done' if set(selection) <= completed else 'partial'
            remaining = sorted(set(selection) - completed)
            result = {**(result or {}), 'state': status, 'selected_total': len(selection), 'selected_count': len(selection),
                'batch_count': len(batch), 'read_count': len(read), 'handled_count': len(completed),
                'consumed_count': len(consumed), 'remaining_selected_count': len(remaining),
                'remaining_source_ids': remaining, 'partial': bool(remaining), 'attempt': current['attempt'],
                'request_id': current['request_id']}
            conn.execute('UPDATE learning_jobs SET status=?,result_json=?,read_json=?,completed_json=?,consumed_json=?,updated_at=? WHERE owner=? AND id=?',
                (status, _json(result), _json(sorted(read)), _json(sorted(completed)), _json(sorted(consumed)), now_iso(), job['owner'], job['id']))
            return result

    def trace(self, job, entry):
        with self.conversations.database(write=True) as conn:
            self.check(job, conn=conn)
            seq = conn.execute('SELECT COALESCE(MAX(seq),0)+1 FROM learning_trace WHERE owner=? AND job_id=?', (job['owner'], job['id'])).fetchone()[0]
            if seq > 512:
                raise RunError('learning_trace_limit', '本选择的审计预算已满；来源未核销，请显式发起新选择。')
            conn.execute('INSERT INTO learning_trace VALUES(?,?,?,?)',
                (job['owner'], job['id'], seq, _json(dict(entry, attempt=job.get('attempt', 0)), 256000)))

    def charge(self, owner, *, daily_limit=40, quota=None):
        from ..agent.learning_scope import principal_variants
        owner, day = owner_id(owner), now_iso()[:10]
        aliases = principal_variants(owner)
        with self.conversations.database(write=True) as conn:
            row = conn.execute('SELECT COALESCE(SUM(calls),0) AS calls FROM learning_usage WHERE owner IN ('
                + ','.join('?' for _ in aliases) + ') AND day=?', (*aliases, day)).fetchone()
            if row['calls'] >= daily_limit:
                raise RunError('learning_budget_exceeded', '今天的独立学习调用预算已用完。')
            conn.execute('INSERT INTO learning_usage VALUES(?,?,1) ON CONFLICT(owner,day) DO UPDATE SET calls=calls+1', (owner, day))
        try:
            if quota:
                admitted = quota()
                if admitted is False:
                    raise RunError('learning_budget_exceeded', '独立调用额度未准入。')
                if admitted is not None and admitted is not True:
                    raise RunError('learning_invalid_admission_result', '额度回调须以异常拒绝或返回None/True。')
        except BaseException as exc:
            with self.conversations.database(write=True) as conn:
                conn.execute('UPDATE learning_usage SET calls=calls-1 WHERE owner=? AND day=? AND calls>0', (owner, day))
            if getattr(exc, 'status_code', None) in {429, 503}:
                raise RunError('llm_quota_exceeded' if exc.status_code == 429 else 'llm_quota_unavailable', '独立模型调用额度未准入。') from None
            raise

    @staticmethod
    def public_job(row):
        selection = json.loads(row['selection_json'])
        completed = set(json.loads(row['completed_json']))
        ids = [item['id'] for item in selection]
        remaining = [sid for sid in ids if sid not in completed]
        counts = {'selected_total': len(ids), 'selected_count': len(ids),
            'batch_count': len(json.loads(row['sources_json'])),
            'read_count': len(json.loads(row['read_json'])), 'handled_count': len(completed),
            'consumed_count': len(json.loads(row['consumed_json'])),
            'remaining_selected_count': len(remaining), 'remaining_source_ids': remaining,
            'selected_source_ids': ids, 'partial': bool(remaining)}
        result = json.loads(row['result_json']) if row['result_json'] else None
        return {**{key: row[key] for key in ('id', 'status', 'created_at', 'updated_at', 'request_id', 'attempt')},
            'manual': bool(row['manual']), 'selection_available': bool(selection), **counts,
            'result': {**result, **counts, 'state': row['status'], 'ok': row['status'] in {'done', 'partial'}} if result is not None else None}

    def job_receipt(self, owner, job_id):
        with self.conversations.database() as conn:
            row = conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND id=?', (owner_id(owner), job_id)).fetchone()
            if row is None:
                raise RunError('learning_job_not_found', '没有这份本人学习选择。')
            return self.public_job(row)

    def status(self, owner, endpoint_fp, project_id='', *, after=0, job_id=''):
        from ..corpus.fs_utils import acquire_file_lock, release_file_lock, FileLockBusy
        with self.conversations.database(write=True) as conn:
            rows = conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND endpoint_fp=? AND project_id=? ORDER BY created_at DESC LIMIT 20', (owner, endpoint_fp, project_id)).fetchall()
            for row in rows:
                if row['status'] not in {'queued', 'running', 'waiting_config'}:
                    continue
                try:
                    self.check(dict(row), conn=conn)
                except RunError as exc:
                    conn.execute("UPDATE learning_jobs SET status='paused',result_json=?,updated_at=? WHERE owner=? AND id=?", (_json({'error_code': exc.code}), now_iso(), owner, row['id']))
                    continue
                if row['status'] == 'waiting_config':
                    continue
                try:
                    handle = acquire_file_lock(self.lock_path(dict(row)), wait=False)
                except FileLockBusy:
                    continue
                release_file_lock(handle)
                conn.execute("UPDATE learning_jobs SET status='waiting_config',updated_at=? WHERE owner=? AND id=? AND status IN ('queued','running')", (now_iso(), owner, row['id']))
            rows = conn.execute('SELECT * FROM learning_jobs WHERE owner=? AND endpoint_fp=? AND project_id=? ORDER BY created_at DESC LIMIT 20', (owner, endpoint_fp, project_id)).fetchall()
            traces = conn.execute('SELECT seq,data_json FROM learning_trace WHERE owner=? AND job_id=? AND seq>? ORDER BY seq LIMIT 32', (owner, job_id, max(0, int(after)))).fetchall() if job_id and any(r['id'] == job_id for r in rows) else []
        return {'scope': self.scope(owner, endpoint_fp, project_id),
                'jobs': [self.public_job(row) for row in rows],
                'fresh_count': self.fresh_stats(owner, endpoint_fp, project_id)['total'],
                'trace': [{'seq': r['seq'], 'data': json.loads(r['data_json'])} for r in traces]}
