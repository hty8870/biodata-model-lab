"""共享消息、现场与结果引用；与任务终态共用短 SQLite 事务，不持锁调用模型。"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager

from .conversation_contracts import MAX_CONVERSATIONS, MAX_FRAMES, MAX_MESSAGES, MAX_IMPORT_BYTES, MAX_PAGE_SIZE
from .conversation_runs import RunError, RunStore, _json, safe_value
from ..secret_patterns import redact
from .workspace_store import identifier, now_iso, owner_id

LOGGER = logging.getLogger(__name__)
_CLEANUP_BATCH_SIZE = 16
_CLEANUP_BATCH_SECONDS = .5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_conversations (
 owner TEXT NOT NULL, id TEXT NOT NULL, title TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
 archived INTEGER NOT NULL DEFAULT 0, deleted INTEGER NOT NULL DEFAULT 0,
 state_json TEXT NOT NULL DEFAULT '{}', active_run TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY(owner,id)
);
CREATE TABLE IF NOT EXISTS shared_messages (
 owner TEXT NOT NULL, conversation_id TEXT NOT NULL, seq INTEGER NOT NULL,
 role TEXT NOT NULL, text TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL,
 created_at TEXT NOT NULL, PRIMARY KEY(owner,conversation_id,seq)
);
CREATE TABLE IF NOT EXISTS shared_frames (
 owner TEXT NOT NULL, conversation_id TEXT NOT NULL, seq INTEGER NOT NULL,
 run_id TEXT NOT NULL DEFAULT '', batch_index INTEGER NOT NULL DEFAULT -1,
 payload_json TEXT, state_json TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(owner,conversation_id,seq)
);
CREATE TABLE IF NOT EXISTS shared_imports (
 owner TEXT NOT NULL, import_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
 conversation_id TEXT NOT NULL, PRIMARY KEY(owner,import_id)
);
CREATE TABLE IF NOT EXISTS shared_cleanup (
 owner TEXT NOT NULL, conversation_id TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
 last_error TEXT NOT NULL DEFAULT '', PRIMARY KEY(owner,conversation_id)
);
"""


_CONVERSATION_SELECT = '''SELECT c.*,
    (SELECT COUNT(*) FROM shared_messages m WHERE m.owner=c.owner AND m.conversation_id=c.id) AS message_count,
    (SELECT COUNT(*) FROM shared_frames f WHERE f.owner=c.owner AND f.conversation_id=c.id) AS frame_count
    FROM shared_conversations c'''


class ConversationStore(RunStore):
    def __init__(self, project_root, **kwargs):
        super().__init__(project_root, **kwargs)
        self.project_root = project_root
        self._cleanup_lock = threading.Lock()
        self._cleanup_cursor = ('', '')

    def has_pending_cleanup(self):
        import sqlite3
        from contextlib import closing
        if not self.path.is_file():
            return False
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='shared_cleanup'").fetchone()
            return bool(table and conn.execute('SELECT 1 FROM shared_cleanup LIMIT 1').fetchone())

    def heartbeat(self, boot_id, *, expire=False):
        super().heartbeat(boot_id, expire=expire)
        if not expire:
            self._consume_deleted()

    def _consume_deleted(self):
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            with self.database() as conn:
                pending = conn.execute('''SELECT owner,conversation_id FROM shared_cleanup
                    WHERE (owner,conversation_id)>(?,?) ORDER BY owner,conversation_id LIMIT ?''',
                    (*self._cleanup_cursor, _CLEANUP_BATCH_SIZE)).fetchall()
            deadline = time.monotonic() + _CLEANUP_BATCH_SECONDS
            processed = 0
            for row in pending:
                if time.monotonic() >= deadline:
                    break
                self._cleanup_cursor = (row['owner'], row['conversation_id'])
                try:
                    self.cleanup_deleted(*self._cleanup_cursor, lock_timeout=0)
                except Exception as exc:
                    LOGGER.warning('Conversation file cleanup deferred (%s)', type(exc).__name__)
                processed += 1
            if processed == len(pending) and len(pending) < _CLEANUP_BATCH_SIZE:
                self._cleanup_cursor = ('', '')
        finally:
            self._cleanup_lock.release()

    @property
    def workspace(self):
        from .workspace_store import WorkspaceStore
        return getattr(self, '_workspace', None) or WorkspaceStore(self.project_root)

    @workspace.setter
    def workspace(self, store):
        self._workspace = store

    @contextmanager
    def database(self, *, write=False, conn=None):
        if conn is not None:
            yield conn
            return
        if write:
            with self.workspace.joint_database(self) as db:
                yield db
            return
        with super().database() as conn:
            for statement in _SCHEMA.split(';'):
                if statement.strip():
                    conn.execute(statement)
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(shared_frames)')}
            if 'message_boundary' not in columns:
                if not conn.in_transaction:
                    conn.execute('BEGIN IMMEDIATE')
                columns = {row['name'] for row in conn.execute('PRAGMA table_info(shared_frames)')}
                if 'message_boundary' not in columns:
                    conn.execute('ALTER TABLE shared_frames ADD COLUMN message_boundary INTEGER NOT NULL DEFAULT 0')
                    conn.execute('''UPDATE shared_frames SET message_boundary=COALESCE((SELECT MAX(m.seq) FROM shared_messages m
                        WHERE m.owner=shared_frames.owner AND m.conversation_id=shared_frames.conversation_id AND m.run_id=shared_frames.run_id),0)
                        WHERE run_id<>'' ''')
            from .learning_store import initialize
            initialize(conn)
            yield conn

    def bind_learning_endpoint(self, owner, run_id, endpoint_fp, *, authorization=None):
        with self.database(write=True) as conn:
            row = conn.execute('SELECT request_json FROM conversation_runs WHERE owner=? AND id=? AND status=\'queued\'', (owner, run_id)).fetchone()
            if row is not None:
                request = json.loads(row['request_json'])
                request['learning_endpoint_fp'] = endpoint_fp
                if authorization is not None:
                    request['learning_authorization'] = safe_value(authorization)
                conn.execute('UPDATE conversation_runs SET request_json=? WHERE owner=? AND id=?', (_json(request), owner, run_id))

    @staticmethod
    def _row(conn, owner, conv_id, *, include_deleted=False):
        row = conn.execute(_CONVERSATION_SELECT + ' WHERE c.owner=? AND c.id=?', (owner, conv_id)).fetchone()
        if row is None or (row['deleted'] and not include_deleted):
            raise RunError('conversation_not_found', '没有这个共享对话。')
        return row

    @staticmethod
    def _public_conversation(row):
        return {'id': row['id'], 'conv_id': row['id'], 'title': row['title'], 'revision': row['revision'],
                'archived': bool(row['archived']), 'active_run_id': row['active_run'] or None,
                'current_state': json.loads(row['state_json']), 'created_at': row['created_at'],
                'updated_at': row['updated_at'], 'shared': True,
                'frame_count': row['frame_count'], 'message_count': row['message_count']}

    def exists(self, owner, conv_id, *, include_deleted=False):
        if not conv_id:
            return False
        with self.database() as conn:
            try:
                self._row(conn, owner_id(owner), identifier(conv_id), include_deleted=include_deleted)
                return True
            except RunError:
                return False

    def create(self, owner, *, conv_id=None, title='', conn=None):
        owner, conv_id = owner_id(owner), identifier(conv_id or 'c_' + uuid.uuid4().hex)
        title = redact(str(title or ''))[:180]
        def insert(db):
            if db.execute('SELECT 1 FROM shared_conversations WHERE owner=? AND id=?', (owner, conv_id)).fetchone():
                return self._public_conversation(self._row(db, owner, conv_id))
            count = db.execute('SELECT COUNT(*) FROM shared_conversations WHERE owner=? AND deleted=0', (owner,)).fetchone()[0]
            if count >= MAX_CONVERSATIONS:
                raise RunError('conversation_limit', '共享对话容量已达上限，请显式删除不用的对话。')
            now = now_iso()
            db.execute('INSERT INTO shared_conversations(owner,id,title,created_at,updated_at) VALUES(?,?,?,?,?)',
                       (owner, conv_id, str(title or '')[:180], now, now))
            return self._public_conversation(self._row(db, owner, conv_id))
        if conn is not None:
            return insert(conn)
        with self.database(write=True) as db:
            return insert(db)

    def list_conversations(self, owner, *, offset=0, limit=100, include_archived=False):
        owner = owner_id(owner)
        limit, offset = min(MAX_PAGE_SIZE, max(1, int(limit))), max(0, int(offset))
        with self.database() as conn:
            rows = conn.execute(_CONVERSATION_SELECT + ' WHERE c.owner=? AND c.deleted=0 AND (? OR c.archived=0) ORDER BY c.updated_at DESC,c.id LIMIT ? OFFSET ?',
                                (owner, int(include_archived), limit + 1, offset)).fetchall()
        return {'items': [self._public_conversation(r) for r in rows[:limit]],
                'next_offset': offset + limit if len(rows) > limit else None}

    def _frame(self, conn, row):
        payload = json.loads(row['payload_json']) if row['payload_json'] else None
        if payload is None and row['run_id']:
            receipt = conn.execute('SELECT result_json FROM conversation_runs WHERE owner=? AND id=?', (row['owner'], row['run_id'])).fetchone()
            result = json.loads(receipt[0]) if receipt and receipt[0] else {}
            batches = result.get('result_batches') or []
            index = row['batch_index']
            payload = batches[index].get('payload') if 0 <= index < len(batches) else result.get('result_payload')
        return {'id': 'f_' + str(row['seq']), 'seq': row['seq'], 'run_id': row['run_id'],
                'batch_index': row['batch_index'], 'message_boundary': row['message_boundary'], 'payload': payload,
                'state': json.loads(row['state_json']), 'created_at': row['created_at']}

    def get(self, owner, conv_id, *, after_message=0, after_frame=0, limit=100):
        owner, conv_id = owner_id(owner), identifier(conv_id)
        limit = min(MAX_PAGE_SIZE, max(1, int(limit)))
        with self.database() as conn:
            if not conn.in_transaction:
                conn.execute('BEGIN')
            out = self._public_conversation(self._row(conn, owner, conv_id))
            messages = conn.execute('SELECT * FROM shared_messages WHERE owner=? AND conversation_id=? AND seq>? ORDER BY seq LIMIT ?',
                                    (owner, conv_id, max(0, after_message), limit + 1)).fetchall()
            frames = conn.execute('SELECT * FROM shared_frames WHERE owner=? AND conversation_id=? AND seq>? ORDER BY seq LIMIT ?',
                                  (owner, conv_id, max(0, after_frame), limit + 1)).fetchall()
            out['messages'] = [{k: r[k] for k in ('seq', 'role', 'text', 'run_id', 'kind', 'created_at')} for r in messages[:limit]]
            out['frames'] = [self._frame(conn, r) for r in frames[:limit]]
            out['next_message'] = messages[limit - 1]['seq'] if len(messages) > limit else None
            out['next_frame'] = frames[limit - 1]['seq'] if len(frames) > limit else None
        return out

    @staticmethod
    def _mutable(row, revision):
        if revision is None or int(revision) != row['revision']:
            raise RunError('conversation_revision_conflict', '共享对话已更新，请刷新实际记录后继续。')
        if row['active_run']:
            raise RunError('conversation_busy', '对话正在执行，请先等待或取消当前任务。')

    def _claim_conversation(self, conn, owner, run_id, conversation_id, request):
        if not request.get('shared'):
            return request
        row = self._row(conn, owner, identifier(conversation_id))
        self._mutable(row, request.get('expected_revision'))
        if row['archived']:
            raise RunError('conversation_archived', '请先恢复归档对话。')
        count = conn.execute('SELECT COUNT(*) FROM shared_messages WHERE owner=? AND conversation_id=?', (owner, conversation_id)).fetchone()[0]
        frames = conn.execute('SELECT COUNT(*) FROM shared_frames WHERE owner=? AND conversation_id=?', (owner, conversation_id)).fetchone()[0]
        if count + 2 > MAX_MESSAGES or frames + 10 > MAX_FRAMES:
            raise RunError('conversation_capacity', '对话已达容量上限，请创建新对话，或从消息较少的较早结果帧分支继续。')
        self.workspace.resolve_conversation(owner, conversation_id, request.get('project_id'), conn=conn)
        state = json.loads(row['state_json'])
        patch = request.get('state_patch') or {}
        state.update(patch)
        request = dict(request, current_state=state)
        conn.execute('UPDATE shared_conversations SET active_run=? WHERE owner=? AND id=?', (run_id, owner, conversation_id))
        return request

    def _finish_conversation(self, conn, row, status, encoded):
        request = json.loads(row['request_json'])
        if not request.get('shared'):
            return encoded
        owner, conv_id = row['owner'], row['conversation_id']
        conversation = self._row(conn, owner, conv_id)
        if conversation['active_run'] != row['id']:
            raise RunError('conversation_run_conflict', '执行任务与对话版本不一致。')
        result = json.loads(encoded) if encoded else {}
        state = json.loads(conversation['state_json'])
        if status == 'done':
            state = result.pop('_current_state', request.get('current_state') or state)
        batches = result.get('result_batches') or []
        if not batches and isinstance(result.get('result_payload'), dict):
            batches = [{'payload': result['result_payload'], 'query_effective': result.get('query', '')}]
        seq = conn.execute('SELECT COALESCE(MAX(seq),0) FROM shared_frames WHERE owner=? AND conversation_id=?', (owner, conv_id)).fetchone()[0]
        boundary = conn.execute('SELECT COALESCE(MAX(seq),0)+2 FROM shared_messages WHERE owner=? AND conversation_id=?', (owner, conv_id)).fetchone()[0]
        for index, batch in enumerate(batches[:10]):
            seq += 1
            from .conversation_results import state_from_payload
            frame_state = state_from_payload(state, batch.get('payload') or {}, query=batch.get('query_effective') or state.get('query', ''))
            frame_state['active_frame_id'] = 'f_' + str(seq)
            conn.execute('INSERT INTO shared_frames VALUES(?,?,?,?,?,?,?,?,?)',
                         (owner, conv_id, seq, row['id'], index if result.get('result_batches') else -1, None, _json(frame_state), now_iso(), boundary))
        if batches:
            state['active_frame_id'] = 'f_' + str(seq)
        message_seq = conn.execute('SELECT COALESCE(MAX(seq),0) FROM shared_messages WHERE owner=? AND conversation_id=?', (owner, conv_id)).fetchone()[0]
        plan = result.get('plan') or {}
        reply = str(plan.get('report_zh') or result.get('echo_zh') or result.get('detail') or '')
        if not reply and result.get('result_payload'):
            reply = f"本目录匹配 {result['result_payload'].get('result_total', 0)} 条；结果与条件已保存。"
        now, revision = now_iso(), conversation['revision'] + 1
        for increment, role, text in ((1, 'user', request.get('text', '')), (2, 'assistant', reply)):
            conn.execute('INSERT INTO shared_messages VALUES(?,?,?,?,?,?,?,?)',
                         (owner, conv_id, message_seq + increment, role, str(text), row['id'], 'turn', now))
        conn.execute("UPDATE shared_conversations SET state_json=?,revision=?,active_run='',updated_at=? WHERE owner=? AND id=?",
                     (_json(state), revision, now, owner, conv_id))
        result.update(shared=True, conv_id=conv_id, run_id=row['id'], revision=revision, current_state=state)
        from .learning_store import register_turn
        register_turn(conn, row, request, status)
        return _json(result)

    def abort_admission(self, owner, run_id):
        with self.database(write=True) as conn:
            conn.execute("UPDATE shared_conversations SET active_run='' WHERE owner=? AND active_run=?", (owner_id(owner), identifier(run_id)))
            conn.execute("DELETE FROM conversation_runs WHERE owner=? AND id=? AND status='queued'", (owner_id(owner), run_id))

    def _delete(self, owner, conv_id, expected_revision):
        from .replay_store import transaction
        with transaction(self.project_root, owner, conv_id):
            with self.database(write=True) as conn:
                row = self._row(conn, owner, conv_id, include_deleted=True)
                if not row['deleted']:
                    self._mutable(row, expected_revision)
                    conn.execute('UPDATE learning_sources SET valid=0 WHERE owner=? AND conversation_id=?', (owner, conv_id))
                    conn.execute("UPDATE shared_conversations SET deleted=1,state_json='{}',title='',revision=revision+1,updated_at=? WHERE owner=? AND id=?", (now_iso(), owner, conv_id))
                    conn.execute('DELETE FROM shared_messages WHERE owner=? AND conversation_id=?', (owner, conv_id))
                    conn.execute('DELETE FROM shared_frames WHERE owner=? AND conversation_id=?', (owner, conv_id))
                    conn.execute('DELETE FROM conversation_run_events WHERE owner=? AND run_id IN (SELECT id FROM conversation_runs WHERE owner=? AND conversation_id=?)', (owner, owner, conv_id))
                    conn.execute('DELETE FROM conversation_runs WHERE owner=? AND conversation_id=?', (owner, conv_id))
                    self.workspace.forget_conversation(owner, conv_id, conn=conn)
                    conn.execute('INSERT OR IGNORE INTO shared_cleanup(owner,conversation_id) VALUES(?,?)', (owner, conv_id))
            return self.cleanup_deleted(owner, conv_id)

    def cleanup_deleted(self, owner, conv_id, *, lock_timeout=None):
        owner, conv_id = owner_id(owner), identifier(conv_id)
        from . import replay_store, session_log
        from ..corpus.fs_utils import file_critical_section
        replay = replay_store.store_path(self.project_root, owner, conv_id)
        with file_critical_section(replay.with_suffix('.json.lock'), timeout=15 if lock_timeout is None else lock_timeout):
            with self.database() as conn:
                if not self._row(conn, owner, conv_id, include_deleted=True)['deleted']:
                    raise RunError('conversation_not_deleted', '只允许清理已删除对话。')
            errors = []
            log = session_log.log_path(self.project_root, owner, conv_id)
            with file_critical_section(log.with_suffix('.jsonl.lock'), timeout=60 if lock_timeout is None else lock_timeout):
                for path in (replay, log):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as exc:
                        errors.append(type(exc).__name__)
            with self.database(write=True) as conn:
                if errors:
                    conn.execute('''INSERT INTO shared_cleanup(owner,conversation_id,attempts,last_error) VALUES(?,?,1,?)
                        ON CONFLICT(owner,conversation_id) DO UPDATE SET attempts=attempts+1,last_error=excluded.last_error''',
                        (owner, conv_id, ','.join(errors)))
                else:
                    conn.execute('DELETE FROM shared_cleanup WHERE owner=? AND conversation_id=?', (owner, conv_id))
            result = {'deleted': True, 'conv_id': conv_id, 'cleanup_pending': bool(errors)}
            if errors:
                result['detail'] = '对话已不可访问，部分文件清理未完成；服务将在后台重试清理。'
            return result

    def action(self, owner, conv_id, action, *, expected_revision, frame_id='', title='', conn=None):
        owner, conv_id = owner_id(owner), identifier(conv_id)
        title = redact(str(title or ''))
        if action == 'delete':
            if conn is not None:
                raise RunError('invalid_delete_transaction', '删除必须先独立提交持久标记。')
            return self._delete(owner, conv_id, expected_revision)
        from .replay_store import transaction
        with transaction(self.project_root, owner, conv_id), self.database(write=True, conn=conn) as conn:
            row = self._row(conn, owner, conv_id)
            self._mutable(row, expected_revision)
            if action in {'archive', 'restore'}:
                conn.execute('UPDATE shared_conversations SET archived=? WHERE owner=? AND id=?', (int(action == 'archive'), owner, conv_id))
            elif action == 'rename':
                if not str(title).strip() or len(title) > 180:
                    raise RunError('invalid_title', '对话标题须为 1–180 个字符。')
                conn.execute('UPDATE shared_conversations SET title=? WHERE owner=? AND id=?', (title, owner, conv_id))
            elif action in {'select_frame', 'branch'}:
                frames = conn.execute('SELECT * FROM shared_frames WHERE owner=? AND conversation_id=? AND seq=?',
                                      (owner, conv_id, int(str(frame_id).removeprefix('f_')))).fetchone()
                if frames is None:
                    raise RunError('frame_not_found', '没有这个结果帧。')
                state = json.loads(frames['state_json'])
                if action == 'branch':
                    new = self.create(owner, title=title or row['title'], conn=conn)
                    prior_frames = conn.execute('SELECT * FROM shared_frames WHERE owner=? AND conversation_id=? AND seq<=? ORDER BY seq', (owner, conv_id, frames['seq'])).fetchall()
                    for prior in prior_frames:
                        payload = self._frame(conn, prior)['payload']
                        conn.execute('INSERT INTO shared_frames VALUES(?,?,?,?,?,?,?,?,?)', (owner, new['id'], prior['seq'], '', -1, _json(payload), prior['state_json'], prior['created_at'], prior['message_boundary']))
                    boundary = frames['message_boundary']
                    if not boundary and conn.execute('SELECT 1 FROM shared_messages WHERE owner=? AND conversation_id=? LIMIT 1', (owner, conv_id)).fetchone():
                        raise RunError('frame_boundary_unavailable', '该旧帧没有可靠消息边界，请显式导入完整历史后再分支。')
                    conn.execute('INSERT INTO shared_messages SELECT owner,?,seq,role,text,run_id,kind,created_at FROM shared_messages WHERE owner=? AND conversation_id=? AND seq<=?', (new['id'], owner, conv_id, boundary))
                    conn.execute('UPDATE shared_conversations SET state_json=? WHERE owner=? AND id=?', (_json(state), owner, new['id']))
                    return self._public_conversation(self._row(conn, owner, new['id']))
                conn.execute('UPDATE shared_conversations SET state_json=? WHERE owner=? AND id=?', (_json(state), owner, conv_id))
            else:
                raise RunError('invalid_conversation_action', '不支持这个对话动作。')
            conn.execute('UPDATE shared_conversations SET revision=revision+1,updated_at=? WHERE owner=? AND id=?', (now_iso(), owner, conv_id))
            return self._public_conversation(self._row(conn, owner, conv_id))

    def import_conversation(self, owner, import_id, fingerprint, data, *, conn=None):
        owner, import_id = owner_id(owner), identifier(import_id)
        data = safe_value(data)
        _json(data, MAX_IMPORT_BYTES)
        messages, frames = data.get('messages') or [], data.get('frames') or []
        if len(messages) > MAX_MESSAGES or len(frames) > MAX_FRAMES:
            raise RunError('conversation_capacity', '导入超出对话容量，未写入。')
        with self.database(write=True, conn=conn) as conn:
            previous = conn.execute('SELECT * FROM shared_imports WHERE owner=? AND import_id=?', (owner, import_id)).fetchone()
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise RunError('import_conflict', '这个导入编号已对应另一份内容。')
                return self._public_conversation(self._row(conn, owner, previous['conversation_id']))
            out = self.create(owner, title=data.get('title', ''), conn=conn)
            conv_id = out['id']
            for seq, message in enumerate(messages, 1):
                if message.get('role') not in {'user', 'assistant'}:
                    raise RunError('invalid_history_role', '导入只接受用户和助手历史文本。')
                conn.execute('INSERT INTO shared_messages VALUES(?,?,?,?,?,?,?,?)', (owner, conv_id, seq, message['role'], str(message.get('text', '')), '', 'imported', now_iso()))
            for seq, frame in enumerate(frames, 1):
                boundary = int(frame.get('message_boundary', len(messages)))
                if not 0 <= boundary <= len(messages):
                    raise RunError('invalid_frame_boundary', '结果帧的消息边界超出导入历史。')
                conn.execute('INSERT INTO shared_frames VALUES(?,?,?,?,?,?,?,?,?)', (owner, conv_id, seq, '', -1, _json(frame.get('payload')), _json(frame.get('state') or {}), now_iso(), boundary))
            state = data.get('current_state') or {}
            conn.execute('UPDATE shared_conversations SET state_json=? WHERE owner=? AND id=?', (_json(state), owner, conv_id))
            conn.execute('INSERT INTO shared_imports VALUES(?,?,?,?)', (owner, import_id, fingerprint, conv_id))
            return self._public_conversation(self._row(conn, owner, conv_id))

    def learning_messages(self, owner, conv_id, *, limit=400, conn=None):
        """学习快照只读对话的用户/助手文本，不含结果帧载荷。"""
        owner, conv_id = owner_id(owner), identifier(conv_id)
        if type(limit) is not int or not 1 <= limit <= 400:
            raise RunError('learning_source_budget', '完整对话读取预算无效。')
        def load(db):
            self._row(db, owner, conv_id)
            rows = db.execute('SELECT role,text FROM shared_messages WHERE owner=? AND conversation_id=? ORDER BY seq LIMIT ?',
                              (owner, conv_id, limit + 1)).fetchall()
            if len(rows) > limit:
                raise RunError('learning_source_budget', '完整对话超过消息预算；未截断、未核销，请明确分段材料。')
            return [{'role': r['role'], 'text': str(r['text'])} for r in rows]
        if conn is not None:
            return load(conn)
        with self.database() as db:
            return load(db)
