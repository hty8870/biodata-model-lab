"""统一 tmp 维护 runner：generation/时间双触发、幂等、有界预算、失败重试、重启恢复。

设计约束（预注册设计 §5.2）：
- 任务类型只是 handler 数据，runner 本身不预设业务；新任务 = 新 handler + 注册一行。
- 没有 commit 通道：runner 只暴露 propose/assess 所需的知识服务，绝不代批。
- 状态落研究侧 tmp SQLite（不是生产库、不建真实 cron/RQ）；fake clock 可注入。
- 预算按窗口记账并持久化；耗尽 → `budget_wait`，下一次窗口再跑，不偷偷超发。
- 失败按 `max_attempts` 退避重试；超过即 `failed`，不把半成品当成功。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
    task_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    interval_seconds REAL NOT NULL,
    generation_binding TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_due REAL NOT NULL,
    last_error TEXT,
    result_json TEXT,
    last_generation TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS budget(
    window_key TEXT PRIMARY KEY,
    calls INTEGER NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS view_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    view_key TEXT NOT NULL,
    generation INTEGER,
    source_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    task_key TEXT,
    kind TEXT,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


class TaskRetry(Exception):
    """可重试失败（LLM 空回包、输出不合 schema、预算临时不足等）。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code, self.detail = code, detail


class TaskFatal(Exception):
    """不可重试失败（永久契约破坏）。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code, self.detail = code, detail


class BudgetWait(Exception):
    """生产侧预算拒绝：不是失败，等下一个窗口。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code, self.detail = code, detail


def _sha(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class RunnerStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.database() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def database(self):
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

                
    def put_task(self, row: dict) -> None:
        with self.database() as conn:
            conn.execute(
                """INSERT INTO tasks(task_key,kind,payload_json,payload_digest,interval_seconds,
                       generation_binding,status,attempts,next_due,last_error,result_json,last_generation,
                       created_at,updated_at)
                   VALUES(:task_key,:kind,:payload_json,:payload_digest,:interval_seconds,
                       :generation_binding,:status,:attempts,:next_due,:last_error,:result_json,
                       :last_generation,:created_at,:updated_at)
                   ON CONFLICT(task_key) DO UPDATE SET
                       kind=excluded.kind, payload_json=excluded.payload_json,
                       payload_digest=excluded.payload_digest, interval_seconds=excluded.interval_seconds,
                       generation_binding=excluded.generation_binding, next_due=excluded.next_due,
                       updated_at=excluded.updated_at""",
                row)

    def task(self, task_key: str) -> dict | None:
        with self.database() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            return dict(row) if row else None

    def tasks(self) -> list[dict]:
        with self.database() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM tasks ORDER BY task_key")]

    def update_task(self, task_key: str, **values) -> None:
        assert values
        columns = ", ".join(f"{key}=?" for key in values)
        with self.database() as conn:
            conn.execute(f"UPDATE tasks SET {columns} WHERE task_key=?",
                         (*values.values(), task_key))

    def recover_running(self, now: float) -> int:
        with self.database() as conn:
            cursor = conn.execute(
                "UPDATE tasks SET status='pending', next_due=?, updated_at=? WHERE status='running'",
                (now, now))
            return cursor.rowcount

                 
    def reserve(self, window_key: str, *, limit: int, calls: int = 1) -> bool:
        with self.database() as conn:
            conn.execute("INSERT OR IGNORE INTO budget(window_key) VALUES(?)", (window_key,))
            row = conn.execute("SELECT calls FROM budget WHERE window_key=?", (window_key,)).fetchone()
            if row["calls"] + calls > limit:
                return False
            conn.execute("UPDATE budget SET calls=calls+? WHERE window_key=?", (calls, window_key))
            return True

    def settle(self, window_key: str, *, reserved: int, used: int, chars: int = 0) -> None:
        with self.database() as conn:
            conn.execute("UPDATE budget SET calls=calls+?, chars=chars+? WHERE window_key=?",
                         (used - reserved, chars, window_key))

    def budget(self, window_key: str) -> dict:
        with self.database() as conn:
            row = conn.execute("SELECT * FROM budget WHERE window_key=?", (window_key,)).fetchone()
            return dict(row) if row else {"window_key": window_key, "calls": 0, "chars": 0}

                
    def put_view(self, *, view_key: str, generation: int | None, source_digest: str,
                 payload: dict, now: float) -> int:
        with self.database() as conn:
            previous = conn.execute(
                "SELECT id FROM view_history WHERE view_key=? AND stale=0 ORDER BY id DESC LIMIT 1",
                (view_key,)).fetchone()
            cursor = conn.execute(
                """INSERT INTO view_history(view_key,generation,source_digest,payload_json,
                       created_at,stale,superseded_by) VALUES(?,?,?,?,?,0,NULL)""",
                (view_key, generation, source_digest, json.dumps(payload, ensure_ascii=False), now))
            new_id = cursor.lastrowid
            if previous:
                conn.execute("UPDATE view_history SET stale=1, superseded_by=? WHERE id=?",
                             (new_id, previous["id"]))
            return new_id

    def mark_stale(self, view_key: str) -> None:
        with self.database() as conn:
            conn.execute("UPDATE view_history SET stale=1 WHERE view_key=? AND stale=0", (view_key,))

    def latest_view(self, view_key: str) -> dict | None:
        with self.database() as conn:
            row = conn.execute(
                "SELECT * FROM view_history WHERE view_key=? ORDER BY id DESC LIMIT 1",
                (view_key,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["payload"] = json.loads(result.pop("payload_json"))
            return result

    def view_history(self, view_key: str) -> list[dict]:
        with self.database() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM view_history WHERE view_key=? ORDER BY id", (view_key,))]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

                 
    def event(self, *, at: float, task_key: str, kind: str, event: str, detail: dict) -> dict:
        record = {"at": at, "task_key": task_key, "kind": kind, "event": event, "detail": detail}
        with self.database() as conn:
            conn.execute("INSERT INTO events(at,task_key,kind,event,detail_json) VALUES(?,?,?,?,?)",
                         (at, task_key, kind, event, json.dumps(detail, ensure_ascii=False)))
        return record

    def events(self) -> list[dict]:
        with self.database() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY id")]
        for row in rows:
            row["detail"] = json.loads(row.pop("detail_json"))
        return rows


@dataclass
class TaskContext:
    task_key: str
    kind: str
    payload: dict
    now: float
    store: RunnerStore
    knowledge: object
    scope: dict
    cfg: object
    predictor: object
    publication_reviewer: object
    _llm: object
    call_limit: int
    llm_calls: int = 0
    llm_chars: int = 0
    prompts: list[str] = field(default_factory=list)
    responses: list[str | None] = field(default_factory=list)

    def llm(self, prompt: str) -> str | None:
        if self.llm_calls >= self.call_limit:
            raise TaskFatal("handler_call_overrun", f"单任务最多 {self.call_limit} 次调用")
        self.llm_calls += 1
        self.llm_chars += len(prompt)
        self.prompts.append(prompt)
        response = self._llm(prompt)
        if response is not None:
            self.llm_chars += len(response)
        self.responses.append(response)
        return response

    def log(self, event: str, **detail) -> None:
        self.store.event(at=self.now, task_key=self.task_key, kind=self.kind,
                         event=event, detail=detail)

    def active_ids(self) -> list[str]:
        snapshot = self.knowledge.snapshot(self.scope["owner"], self.scope["endpoint_fp"],
                                           self.scope["project"])
        return sorted(row["id"] for row in snapshot["items"] if row["scope_level"] != "system")


class MaintenanceRunner:
    def __init__(self, db_path, *, clock, llm, knowledge, owner, endpoint_fp, project_id="",
                 max_calls_per_window=8, window_seconds=86_400, max_attempts=3,
                 retry_delay=60, task_call_limit=1, generation_sources=None,
                 handlers=None, cfg=None, predictor=None, publication_reviewer=None):
        self.store = RunnerStore(db_path)
        self.clock = clock
        self.llm = llm
        self.knowledge = knowledge
        self.scope = {"owner": owner, "endpoint_fp": endpoint_fp, "project": project_id}
        self.max_calls_per_window = max_calls_per_window
        self.window_seconds = window_seconds
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.task_call_limit = task_call_limit
        self.generation_sources = dict(generation_sources or {})
        self.handlers = dict(handlers or {})
        self.cfg = cfg
        self.predictor = predictor
        self.publication_reviewer = publication_reviewer
        self.store.recover_running(self.clock())
        self.events = self.store.events()

             
    def register(self, task_key: str, kind: str, payload: dict, *,
                 interval_seconds: float = 3600, generation_binding: str = "") -> dict:
        if kind not in self.handlers:
            raise ValueError(f"no handler for kind={kind}")
        if generation_binding and generation_binding not in self.generation_sources:
            raise ValueError(f"no generation source for binding={generation_binding}")
        digest = _sha(payload)
        existing = self.store.task(task_key)
        if existing:
            if existing["payload_digest"] != digest:
                raise ValueError(f"task {task_key} exists with a different payload")
            return existing
        now = self.clock()
        row = {"task_key": task_key, "kind": kind,
               "payload_json": json.dumps(payload, ensure_ascii=False), "payload_digest": digest,
               "interval_seconds": float(interval_seconds),
               "generation_binding": generation_binding, "status": "pending", "attempts": 0,
               "next_due": now, "last_error": None, "result_json": None,
               "last_generation": None, "created_at": now, "updated_at": now}
        self.store.put_task(row)
        return self.store.task(task_key)

               
    def _generation(self, binding: str):
        source = self.generation_sources.get(binding)
        if source is None:
            return None
        return source()

    def _due(self, task: dict, now: float, forced: set[str]) -> tuple[bool, str]:
        if task["task_key"] in forced:
            return True, "forced"
        if task["status"] in {"pending", "retry_wait"} and task["next_due"] <= now:
            return True, "time"
        if task["status"] == "done" and now >= task["next_due"]:
            return True, "time"
        if task["status"] == "done" and task["generation_binding"]:
            current = self._generation(task["generation_binding"])
            if current is not None and str(current) != (task["last_generation"] or ""):
                return True, "generation"
        return False, ""

    def tick(self, *, force_task_keys=()) -> list[dict]:
        now = self.clock()
        self.store.recover_running(now)
        emitted = []
        forced = set(force_task_keys)
        for task in self.store.tasks():
            due, reason = self._due(task, now, forced)
            if not due:
                continue
            window = int(now // self.window_seconds)
            window_key = f"{self.window_seconds}:{window}"
            if not self.store.reserve(window_key, limit=self.max_calls_per_window):
                next_window = (window + 1) * self.window_seconds
                self.store.update_task(task["task_key"], next_due=next_window, updated_at=now)
                emitted.append(self.store.event(
                    at=now, task_key=task["task_key"], kind=task["kind"], event="budget_wait",
                    detail={"window": window_key, "budget": self.store.budget(window_key),
                            "next_due": next_window}))
                continue
            self.store.update_task(task["task_key"], status="running", updated_at=now)
            context = TaskContext(task_key=task["task_key"], kind=task["kind"],
                                  payload=json.loads(task["payload_json"]), now=now,
                                  store=self.store, knowledge=self.knowledge, scope=dict(self.scope),
                                  cfg=self.cfg, predictor=self.predictor,
                                  publication_reviewer=self.publication_reviewer,
                                  _llm=self.llm, call_limit=self.task_call_limit)
            emitted.append(self.store.event(
                at=now, task_key=task["task_key"], kind=task["kind"], event="start",
                detail={"trigger": reason, "attempts": task["attempts"]}))
            try:
                result = self.handlers[task["kind"]](context)
            except (TaskRetry, TaskFatal, BudgetWait) as exc:
                self.store.settle(window_key, reserved=1, used=context.llm_calls,
                                  chars=context.llm_chars)
                attempts = task["attempts"] + 1
                if isinstance(exc, BudgetWait):
                    next_window = (window + 1) * self.window_seconds
                    self.store.update_task(task["task_key"], status="pending",
                                           next_due=next_window, updated_at=now,
                                           last_error=exc.code)
                    event = "budget_wait"
                elif isinstance(exc, TaskFatal) or attempts >= self.max_attempts:
                    self.store.update_task(task["task_key"], status="failed", attempts=attempts,
                                           next_due=float("inf"), updated_at=now,
                                           last_error=exc.code)
                    event = "failed"
                else:
                    self.store.update_task(task["task_key"], status="retry_wait", attempts=attempts,
                                           next_due=now + self.retry_delay, updated_at=now,
                                           last_error=exc.code)
                    event = "retry"
                emitted.append(self.store.event(
                    at=now, task_key=task["task_key"], kind=task["kind"], event=event,
                    detail={"code": exc.code, "detail": getattr(exc, "detail", ""),
                            "attempts": attempts, "llm_calls": context.llm_calls,
                            "prompts": context.prompts, "responses": context.responses}))
                continue
            except Exception as exc:                           
                self.store.settle(window_key, reserved=1, used=context.llm_calls,
                                  chars=context.llm_chars)
                attempts = task["attempts"] + 1
                status = "failed" if attempts >= self.max_attempts else "retry_wait"
                self.store.update_task(task["task_key"], status=status, attempts=attempts,
                                       next_due=float("inf") if status == "failed" else now + self.retry_delay,
                                       updated_at=now, last_error=type(exc).__name__)
                emitted.append(self.store.event(
                    at=now, task_key=task["task_key"], kind=task["kind"], event=status,
                    detail={"code": "unexpected", "error": type(exc).__name__, "attempts": attempts}))
                continue
            self.store.settle(window_key, reserved=1, used=context.llm_calls, chars=context.llm_chars)
            generation = self._generation(task["generation_binding"]) if task["generation_binding"] else None
            next_due = now + task["interval_seconds"]
            self.store.update_task(
                task["task_key"], status="done", attempts=0, next_due=next_due, updated_at=now,
                last_error=None, last_generation=None if generation is None else str(generation),
                result_json=json.dumps(result, ensure_ascii=False))
            emitted.append(self.store.event(
                at=now, task_key=task["task_key"], kind=task["kind"], event="done",
                detail={"trigger": reason, "llm_calls": context.llm_calls, "result": result,
                        "prompts": context.prompts, "responses": context.responses}))
        self.events.extend(emitted)
        return emitted

              
    def exposure(self, view_key: str, *, generation: int | None = None) -> dict | None:
        view = self.store.latest_view(view_key)
        if view is None:
            return None
        stale = bool(view["stale"])
        if generation is not None and view["generation"] != generation:
            stale = True
        view["stale"] = stale
        view["serving"] = not stale
        return view


__all__ = ["BudgetWait", "MaintenanceRunner", "RunnerStore", "TaskContext", "TaskFatal", "TaskRetry"]
