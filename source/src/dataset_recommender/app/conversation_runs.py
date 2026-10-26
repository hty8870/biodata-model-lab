"""Bounded conversation execution with account-owned durable receipts.

Only safe request summaries and results are persisted. Credentials remain in the
in-memory execution closure. Independent executor leases protect live processes;
expired executors are reconciled without replaying potentially effective tools.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from ..agent.run_control import RunCancelled, RunTimedOut, bind_run_check
from ..secret_patterns import redact
from .durable_jobs import JobStoreError, open_database
from .runtime_paths import assert_runtime_path, instance_data_dir_for
from .workspace_store import identifier, now_iso, owner_id

LOGGER = logging.getLogger(__name__)
RUN_INTERNAL_ERROR_DETAIL = "处理这句话时出了内部错误，请重试。"
TERMINAL = frozenset({"done", "failed", "cancelled", "interrupted"})
_CREDENTIAL_KEYS = frozenset({"api_key", "apikey", "password", "authorization", "access_token", "refresh_token", "session_token", "secret", "replay_key", "bearer"})
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_runs (
 owner TEXT NOT NULL, id TEXT NOT NULL, request_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
 conversation_id TEXT NOT NULL DEFAULT '', project_id TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 cancel_requested INTEGER NOT NULL DEFAULT 0, boot_id TEXT NOT NULL,
 PRIMARY KEY(owner,id), UNIQUE(owner,request_key)
);
CREATE TABLE IF NOT EXISTS conversation_run_events (
 owner TEXT NOT NULL, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
 event TEXT NOT NULL, data_json TEXT NOT NULL,
 PRIMARY KEY(owner,run_id,seq)
);
CREATE INDEX IF NOT EXISTS conversation_runs_state ON conversation_runs(status,owner);
"""


class RunError(JobStoreError):
    pass


def safe_value(value):
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items() if str(k).lower() not in _CREDENTIAL_KEYS}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    return value


def _json(value, limit=8 * 1024 * 1024):
    text = json.dumps(safe_value(value), ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > limit:
        raise RunError("run_result_too_large", "任务结果超过可保存的大小，已停止写入。")
    return text


def normalize_request_id(raw: str | None) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if len(text) <= 64:
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:48]


def _request_key(raw):
    return hashlib.sha256((normalize_request_id(raw) or "").encode("utf-8")).hexdigest()


def _compact_result(value):
    budget = 128000
    priority = ("completed_result", "plan", "steps", "result", "trace", "observations")

    def visit(item, depth=0):
        nonlocal budget
        if depth > 10 or budget <= 0:
            return "[omitted by result budget]"
        if isinstance(item, dict):
            keys = [key for key in priority if key in item]
            keys.extend(key for key in item if key not in priority)
            result = {}
            for key in keys[:64]:
                name = str(key)
                if name.lower() in _CREDENTIAL_KEYS:
                    continue
                if budget <= 0:
                    break
                name = redact(name).encode("utf-8", "replace")[:512].decode("utf-8", "ignore")
                budget -= len(name.encode("utf-8"))
                result[name] = visit(item[key], depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            result = []
            for part in item[:64]:
                if budget <= 0:
                    break
                result.append(visit(part, depth + 1))
            return result
        if isinstance(item, str):
            text = redact(item).encode("utf-8", "replace")[:min(2048, budget)].decode("utf-8", "ignore")
            budget -= len(text.encode("utf-8"))
            return text
        if item is None or isinstance(item, (bool, int, float)):
            budget -= 32
            return item
        return {"unavailable_type": type(item).__name__}

    return visit(value)


def _terminal_payload(status, result):
    try:
        return status, _json(result) if result is not None else None
    except Exception as exc:
        code = exc.code if isinstance(exc, RunError) else "run_result_not_serializable"
        LOGGER.warning("Conversation terminal result was summarized (%s)", code)
        status = "cancelled" if status == "cancelled" else "failed"
        try:
            summary = _compact_result(result)
            if not isinstance(summary, dict):
                summary = {"completed_result": summary}
            summary.update(ok=False, result_truncated=True, result_error_code=code, may_have_side_effects=True)
            summary.setdefault("detail", "完整结果未能保存，已保留可用的完成步骤摘要。已完成操作没有撤销，请核对回执。")
            return status, _json(summary)
        except Exception:
            LOGGER.warning("Conversation completion summary was unavailable")
            return status, _json({"ok": False, "result_truncated": True, "result_error_code": code,
                "completion_summary_unavailable": True, "may_have_side_effects": True,
                "detail": "结果与完成摘要均无法保存。任务已停止，已完成操作没有撤销，请核对实际产物。"})


class RunStore:
    def __init__(self, project_root: Path, *, path: Path | None = None):
        default = instance_data_dir_for(project_root, ".userdata") / "conversation-runs.sqlite"
        override = os.environ.get("BIODATA_RUNS_DB", "").strip()
        selected = path if path is not None else Path(override) if override else default
        self.path = assert_runtime_path(selected, error_cls=RunError)

    def database(self, *, write=False):
        return open_database(self.path, schema=_SCHEMA + """
            CREATE TABLE IF NOT EXISTS conversation_executors (
                boot_id TEXT PRIMARY KEY, expires_at REAL NOT NULL
            );
            """, write=write)

    def executor_lock_path(self, boot_id):
        return self.path.parent / (self.path.name + '.executors') / (identifier(boot_id) + '.lock')

    def heartbeat(self, boot_id, *, expire=False):
        from .conversation_contracts import LEASE_SECONDS
        with self.database(write=True) as conn:
            conn.execute("INSERT INTO conversation_executors VALUES(?,?) ON CONFLICT(boot_id) DO UPDATE SET expires_at=excluded.expires_at",
                         (boot_id, 0 if expire else time.time() + LEASE_SECONDS))

    def _claim_conversation(self, conn, owner, run_id, conversation_id, request):
        return request

    def _finish_conversation(self, conn, row, status, encoded):
        return encoded

    @staticmethod
    def _public(row):
        result = {key: row[key] for key in ("id", "conversation_id", "project_id", "status", "created_at", "updated_at")}
        result["cancel_requested"] = bool(row["cancel_requested"])
        result["request"] = json.loads(row["request_json"])
        result["result"] = json.loads(row["result_json"]) if row["result_json"] else None
        return result

    def load(self, owner, run_id):
        owner, run_id = owner_id(owner), identifier(run_id)
        with self.database() as conn:
            row = conn.execute("SELECT * FROM conversation_runs WHERE owner=? AND id=?", (owner, run_id)).fetchone()
        if row is None:
            raise RunError("run_not_found", "没有这个对话任务。")
        return self._public(row)

    def find_request(self, owner, request_id):
        owner = owner_id(owner)
        key = _request_key(request_id)
        with self.database() as conn:
            row = conn.execute("SELECT * FROM conversation_runs WHERE owner=? AND request_key=?", (owner, key)).fetchone()
        if row is None:
            raise RunError("run_not_found", "没有找到这次请求的执行记录，不会自动重发。")
        return self._public(row)

    def claim(self, owner, request_id, fingerprint, *, conversation_id="", project_id="", request=None,
              boot_id, max_inflight=16, max_per_owner=1):
        owner = owner_id(owner)
        request_key = _request_key(request_id)
        with self.database(write=True) as conn:
            row = conn.execute("SELECT * FROM conversation_runs WHERE owner=? AND request_key=?", (owner, request_key)).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise RunError("run_request_conflict", "这个请求编号已对应另一条请求。")
                return self._public(row), False
            active = "status IN ('queued','running','cancelling')"
            total = conn.execute("SELECT COUNT(*) FROM conversation_runs WHERE " + active).fetchone()[0]
            own = conn.execute("SELECT COUNT(*) FROM conversation_runs WHERE owner=? AND " + active, (owner,)).fetchone()[0]
            if total >= max_inflight or own >= max_per_owner:
                raise RunError("run_capacity", "在途任务已达上限，请等待当前任务完成后再试。")
            run_id, now = "r_" + uuid.uuid4().hex, now_iso()
            request = self._claim_conversation(conn, owner, run_id, conversation_id, request or {})
            conn.execute("""INSERT INTO conversation_runs(owner,id,request_key,fingerprint,conversation_id,project_id,
                status,request_json,created_at,updated_at,boot_id) VALUES(?,?,?,?,?,?,'queued',?,?,?,?)""",
                (owner, run_id, request_key, fingerprint, conversation_id, project_id, _json(request, 64000), now, now, boot_id))
            claimed = self._public(conn.execute('SELECT * FROM conversation_runs WHERE owner=? AND id=?', (owner, run_id)).fetchone())
        return claimed, True

    def transition(self, owner, run_id, status, result=None):
        if status in TERMINAL:
            status, encoded = _terminal_payload(status, result)
        else:
            encoded = _json(result) if result is not None else None
        with self.database(write=True) as conn:
            row = conn.execute("SELECT * FROM conversation_runs WHERE owner=? AND id=?",
                               (owner_id(owner), identifier(run_id))).fetchone()
            if row is None or row["status"] in TERMINAL:
                return
            if status in TERMINAL:
                encoded = self._finish_conversation(conn, row, status, encoded)
            conn.execute("UPDATE conversation_runs SET status=?,result_json=?,updated_at=? WHERE owner=? AND id=?",
                         (status, encoded, now_iso(), owner_id(owner), identifier(run_id)))

    def abort_admission(self, owner, run_id):
        """Release a request that failed preparation before any worker was queued."""
        with self.database(write=True) as conn:
            conn.execute("DELETE FROM conversation_runs WHERE owner=? AND id=? AND status='queued'",
                         (owner_id(owner), identifier(run_id)))

    def append_event(self, owner, run_id, event, data):
        try:
            encoded = _json(data, 256000)
        except Exception as exc:
            code = exc.code if isinstance(exc, RunError) else "run_result_not_serializable"
            LOGGER.warning("Conversation event was summarized (%s)", code)
            try:
                encoded = _json({"event_truncated": True, "event_error_code": code,
                                 "completed_data": _compact_result(data)}, 256000)
            except Exception:
                encoded = _json({"event_truncated": True, "event_error_code": code,
                                 "completion_summary_unavailable": True})
        with self.database(write=True) as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM conversation_run_events WHERE owner=? AND run_id=?",
                               (owner, run_id)).fetchone()[0]
            if seq > 512:
                LOGGER.warning("Conversation event replay limit reached; completion remains in the terminal receipt")
                return
            conn.execute("INSERT INTO conversation_run_events VALUES(?,?,?,?,?)", (owner, run_id, seq, str(event), encoded))

    def events(self, owner, run_id, after=0):
        self.load(owner, run_id)
        with self.database() as conn:
            rows = conn.execute("SELECT seq,event,data_json FROM conversation_run_events WHERE owner=? AND run_id=? AND seq>? ORDER BY seq LIMIT 100",
                                (owner, run_id, max(0, int(after)))).fetchall()
        return [{"seq": row["seq"], "event": row["event"], "data": json.loads(row["data_json"])} for row in rows]

    def request_cancel(self, owner, run_id):
        current = self.load(owner, run_id)
        if current["status"] in TERMINAL:
            return current
        with self.database(write=True) as conn:
            conn.execute("UPDATE conversation_runs SET cancel_requested=1,status='cancelling',updated_at=? WHERE owner=? AND id=? AND status IN ('queued','running','cancelling')",
                         (now_iso(), owner, run_id))
        return self.load(owner, run_id)

    def reconcile_interrupted(self, boot_id):
        result = _json({"ok": False, "interrupted": True, "may_have_side_effects": True,
                        "detail": "服务重启中断了这次任务。请核对已执行的步骤，不会自动重复执行。"})
        with self.database(write=True) as conn:
            from ..corpus.fs_utils import acquire_file_lock, release_file_lock, FileLockBusy
            rows = conn.execute("SELECT * FROM conversation_runs WHERE boot_id<>? AND status IN ('queued','running','cancelling')", (boot_id,)).fetchall()
            for row in rows:
                try:
                    handle = acquire_file_lock(self.executor_lock_path(row['boot_id']), wait=False)
                except FileLockBusy:
                    continue
                release_file_lock(handle)
                encoded = self._finish_conversation(conn, row, "interrupted", result)
                conn.execute("UPDATE conversation_runs SET status='interrupted',result_json=?,updated_at=? WHERE owner=? AND id=?",
                             (encoded, now_iso(), row["owner"], row["id"]))


class ConversationRuntime:
    def __init__(self, store: RunStore, *, workers=2, max_inflight=16, max_per_owner=1, max_seconds=900):
        self.store = store
        self.boot_id = uuid.uuid4().hex
        self.max_inflight, self.max_per_owner = max_inflight, max_per_owner
        self.max_seconds = max_seconds
        self._lock = threading.RLock()
        self._tasks = {}
        self._waiters = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="conversation-run")
        self._closed = threading.Event()
        from ..corpus.fs_utils import acquire_file_lock
        self._executor_handle = acquire_file_lock(self.store.executor_lock_path(self.boot_id), wait=False)
        self.store.heartbeat(self.boot_id)
        self.store.reconcile_interrupted(self.boot_id)
        self._heartbeat_thread = threading.Thread(target=self._renew_lease, daemon=True, name="conversation-lease")
        self._heartbeat_thread.start()

    def _renew_lease(self):
        from .conversation_contracts import HEARTBEAT_SECONDS
        while not self._closed.wait(HEARTBEAT_SECONDS):
            try:
                self.store.heartbeat(self.boot_id)
                self.store.reconcile_interrupted(self.boot_id)
            except Exception:
                LOGGER.warning("Conversation executor lease renewal failed")

    def submit(self, owner, request_id, fingerprint, *, prepare, conversation_id="", project_id="", request=None):
        owner = owner_id(owner)
        self.store.reconcile_interrupted(self.boot_id)
        with self._lock:
            run, created = self.store.claim(owner, request_id, fingerprint, conversation_id=conversation_id,
                project_id=project_id, request=request, boot_id=self.boot_id,
                max_inflight=self.max_inflight, max_per_owner=self.max_per_owner)
            if not created:
                return run
            try:
                execute = prepare()                                                          
            except BaseException:
                self.store.abort_admission(owner, run["id"])
                raise
            stop, dispatched, admission_aborted = threading.Event(), threading.Event(), threading.Event()
            try:
                future = self._pool.submit(self._execute, owner, run["id"], stop, execute, time.monotonic(), dispatched, admission_aborted)
                self._tasks[(owner, run["id"])] = (stop, future)
            except BaseException as exc:
                admission_aborted.set()
                stop.set()
                dispatched.set()
                self.store.abort_admission(owner, run["id"])
                raise RunError('run_enqueue_failed', '任务未排入队列，未执行工具；配置准入可能已经计量，请核对后重试。') from exc
            dispatched.set()
            try:
                return self.store.load(owner, run["id"])
            except Exception:
                LOGGER.warning('Queued task status read unavailable; original run identifier remains recoverable')
                return run

    def _execute(self, owner, run_id, stop, execute, submitted_at, dispatched=None, admission_aborted=None):
        if dispatched is not None:
            dispatched.wait()
            if admission_aborted is not None and admission_aborted.is_set():
                return
        from ..agent.run_control import completed_snapshot
        result = completed = None
        def check():
            current = self.store.load(owner, run_id)
            if stop.is_set() or current["cancel_requested"] or current["status"] in TERMINAL:
                raise RunCancelled(completed_snapshot())
            if time.monotonic() - submitted_at > self.max_seconds:
                raise RunTimedOut(completed_snapshot())
        def emit(event, data):
            check()
            self.store.append_event(owner, run_id, event, data)
        try:
            check()
            self.store.transition(owner, run_id, "running")
            with bind_run_check(check):
                try:
                    result = execute(emit)
                    check()
                finally:
                    completed = completed_snapshot()
            status = "done" if result.get("ok", True) else "failed"
            self.store.transition(owner, run_id, status, result)
        except RunCancelled as exc:
            self.store.transition(owner, run_id, "cancelled", {"ok": False, "cancelled": True,
                "may_have_side_effects": True,
                "completed_result": result if result is not None else exc.completed_result,
                "detail": "已在安全边界停止后续执行；已经完成的操作没有撤销。"})
        except RunTimedOut as exc:
            self.store.transition(owner, run_id, "failed", {"ok": False, "code": "run_timeout",
                "may_have_side_effects": True,
                "completed_result": result if result is not None else exc.completed_result,
                "detail": "任务达到时间上限，已在安全边界停止后续执行。请核对已完成步骤。"})
        except Exception:
            LOGGER.exception("Conversation execution failed")
                                                                
                                                                         
                                                            
            failed = {"ok": False, "detail": RUN_INTERNAL_ERROR_DETAIL, "may_have_side_effects": True}
            snapshot = result if result is not None else completed
            if snapshot is not None:
                failed["completed_result"] = snapshot
            self.store.transition(owner, run_id, "failed", failed)
        finally:
            with self._lock:
                self._tasks.pop((owner, run_id), None)
            after_commit = getattr(execute, 'after_commit', None)
            if after_commit:
                try:
                    if self.store.load(owner, run_id)['status'] in TERMINAL:
                        after_commit()
                except Exception:
                    LOGGER.warning('Independent learning notification failed; committed turn remains available')

    def cancel(self, owner, run_id):
        owner = owner_id(owner)
        with self._lock:
            current = self.store.request_cancel(owner, run_id)
            task = self._tasks.get((owner, run_id))
            if task:
                stop, future = task; stop.set()
                if future.cancel():
                    self.store.transition(owner, run_id, "cancelled", {"ok": False, "cancelled": True,
                        "may_have_side_effects": False, "detail": "排队任务已取消，尚未执行。"})
                    self._tasks.pop((owner, run_id), None)
            return self.store.load(owner, run_id)

    def close(self, *, wait=True):
        self._pool.shutdown(wait=wait)
        if wait:
            self._closed.set()
            self._heartbeat_thread.join(timeout=3)
            self.store.heartbeat(self.boot_id, expire=True)
            from ..corpus.fs_utils import release_file_lock
            if self._executor_handle is not None:
                release_file_lock(self._executor_handle)
                self._executor_handle = None

    @contextmanager
    def waiting_transport(self, owner):
        """Legacy synchronous consumers cannot occupy the whole HTTP thread pool."""
        owner = owner_id(owner)
        with self._lock:
            if sum(self._waiters.values()) >= 8 or self._waiters.get(owner, 0) >= 2:
                raise RunError("run_transport_capacity", "等待中的连接过多，请稍后核对任务状态。")
            self._waiters[owner] = self._waiters.get(owner, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._waiters[owner] -= 1
                if not self._waiters[owner]:
                    del self._waiters[owner]

    def wait_result(self, owner, run_id, timeout=180):
        with self.waiting_transport(owner):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                run = self.store.load(owner, run_id)
                if run["status"] in TERMINAL:
                    return run["result"] or {"ok": False, "detail": "任务没有保存可恢复的结果。"}
                time.sleep(.02)
        raise RunError("run_wait_timeout", "任务仍在执行，请稍后使用原请求编号核对结果。")
