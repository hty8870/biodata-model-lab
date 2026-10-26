                       
"""RQ 消息队列后端 —— corpus-sync 长任务的 opt-in 跨进程执行通道。

定位：默认形态是**进程内后台线程单飞 job**（`webapp._corpus_sync_job_*`，一行不动）；
只有 env `BIODATA_JOB_BACKEND=rq` 时才走到本模块。因此：

- **顶层不 import rq / redis**：最小安装（默认 `requirements.txt`）没有 rq 包，顶层引入会让
  整个应用起不来。`start()` 里才 `import rq`，缺包抛 `JobBackendError("job_backend_unavailable")`。
- **连接复用 `redis_store`**：本模块不解析 URL、不建连接池，一切走 `redis_store.get_client()`
  （唯一建连真源）。
- **状态形状与内存快照逐键一致**：对外始终是 `{"status","started_at","finished_at","result","error"}`，
  status 取值集合 `idle/running/done/failed` 与内存路径完全相同——前端 watchSyncJobState /
  sync_button 的契约对状态取值封闭，新增取值会静默走到未知分支。

跨进程缓存失效（本模块存在的核心理由）
--------------------------------------
RQ worker 跑在**独立进程**：worker 里调 `invalidate_external_cache()` /
`recall_api.invalidate_vectors()` 只能清掉 worker 自己的内存，对 web 进程无效。唯一通道是
worker 在状态 hash 里保存稳定 `corpus_generation`；每个 web consumer 记录自己的
last-generation，普通 API 请求即调用 `consume_web_invalidate_flag()` 检查，不依赖状态轮询。
通知不清零；失效复用 webapp 的既有语料/向量缓存入口，成功后才推进本进程游标。
Queue、Worker 与 Job.fetch 共用 JSONSerializer；旧 pickle 作业不提供反序列化兜底。
补偿结果持久保留，业务终态已确认且 RQ 已 finished 后才安全回收。

错误契约
--------
`JobBackendError(code, message)` 与 `AccountError` / `JobStoreError` 同构。与内存路径的
「快照永远读得出来」不同，RQ 路径下状态真源在 Redis：读不到就**如实 503**，不静默伪造
一个 idle（用户会误以为任务从没跑过）。翻译职责在接口层——Web 端点把 `JobBackendError`
翻成 HTTPException 503（响应体用 `message`，`code` 可进 `X-Error-Code`）。

连接双通道（RQ 二进制约束）
--------------------------
RQ 2.x 要求 job 数据以**二进制**写入 Redis hash（`Job.to_dict` 对序列化结果做 zlib 压缩），
decode_responses=True 的连接读它会抛 UnicodeDecodeError（真实 worker 取不到任务、job 停在
QUEUED）。因此：队列/Worker 走 `redis_store.get_rq_client()`（bytes 模式），本模块自己的
状态 hash（纯文本 JSON）仍走 `redis_store.get_client()`（decode 模式）。两个单例共用同一
URL/超时真源（见 redis_store 模块 docstring「两种解码模式」）。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from . import redis_store
from .limits import CORPUS_SYNC_JOB_TIMEOUT_SECONDS

                                       
_STATUS_KEY = "biodata:job:corpus_sync"
                                          
_QUEUE_NAME = "biodata"

                                                            
_BACKEND_ENV = "BIODATA_JOB_BACKEND"
_BACKEND_VALUE = "rq"

                                                          
_ENTRY_PATH = "dataset_recommender.app.rq_jobs.corpus_sync_entry"

_FLAG_FIELD = "needs_web_invalidate"
_GENERATION_FIELD = "corpus_generation"
RQ_SERIALIZER = "rq.serializers.JSONSerializer"
_RESULT_TTL = -1
                                
_JOB_ID_FIELD = "rq_job_id"
_CLAIMED_AT_FIELD = "claimed_at"
_ENQUEUE_GRACE_SECONDS = 30                               

_STORE_UNAVAILABLE_MSG = "任务状态存储暂时不可用，请稍后重试。"


class JobBackendError(Exception):
    """机器码 + 人读提示；接口层翻成 4xx/5xx，不泄漏内部细节。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def backend_active() -> bool:
    """env `BIODATA_JOB_BACKEND` 是否点亮 RQ 后端。

    空（未设 / 全空白）→ False（默认形态）；"rq" → True；其他非空值 → 抛
    `JobBackendError("bad_config")`——拼错开关名时静默回落到内存后端会让「我明明开了
    消息队列」变成一个查不出来的错觉，宁可在启动路径上直接报错。
    """
    value = (os.environ.get(_BACKEND_ENV, "") or "").strip().lower()
    if not value:
        return False
    if value == _BACKEND_VALUE:
        return True
    raise JobBackendError(
        "bad_config",
        f"环境变量 {_BACKEND_ENV} 取值无法识别（当前 {value!r}）；"
        f"留空走默认形态，或设为 {_BACKEND_VALUE!r} 启用 RQ 消息队列后端。",
    )


def _now() -> str:
    """ISO UTC 秒级时间戳，与 `webapp._corpus_sync_now` / `durable_jobs._utc_now` 同形同口径
    （同一批状态值可能由内存快照、SQLite 镜像、Redis 三处呈现，格式必须逐字节一致）。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _client() -> "Any":
    """共享 Redis 客户端（decode 模式，供状态 hash 读写）；缺包/建连失败翻成
    `JobBackendError`（code 原样透传）。"""
    try:
        return redis_store.get_client()
    except redis_store.RedisUnavailable as exc:
        raise JobBackendError(exc.code, exc.message) from exc


def _rq_client() -> "Any":
    """RQ 队列专用 bytes 模式客户端（RQ job 数据是 zlib 二进制，decode 模式读不了）。
    缺包/建连失败翻成 `JobBackendError`（code 原样透传）。"""
    try:
        return redis_store.get_rq_client()
    except redis_store.RedisUnavailable as exc:
        raise JobBackendError(exc.code, exc.message) from exc


def _as_text(value: "Any") -> str:
    """Redis 值 → str。`redis_store` 当前按 `decode_responses=True` 建连，这里仍兼容 bytes，
    避免连接模式变化时状态端点整体崩掉。"""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _read_hash() -> "dict[str, str]":
    try:
        raw = _client().hgetall(_STATUS_KEY)
    except (redis_store.RedisUnavailable, *redis_store.redis_error_types()) as exc:
        raise JobBackendError("job_store_unavailable", _STORE_UNAVAILABLE_MSG) from exc
    return {_as_text(k): _as_text(v) for k, v in (raw or {}).items()}


def _write_hash(mapping: "dict[str, str]") -> None:
    """写状态字段（只覆盖给定字段，不整行替换——终态写不该抹掉 started_at）。失败抛错。"""
    try:
        _client().hset(_STATUS_KEY, mapping=mapping)
    except (redis_store.RedisUnavailable, *redis_store.redis_error_types()) as exc:
        raise JobBackendError("job_store_unavailable", _STORE_UNAVAILABLE_MSG) from exc


def _compare_hash(expected: dict[str, str], mapping: dict[str, str]) -> bool:
    """Redis WATCH 比较并更新同一任务快照；抢占与终态对账共用，无 Lua/额外依赖。"""
    import redis
    try:
        for _ in range(5):
            with _client().pipeline() as pipe:
                try:
                    pipe.watch(_STATUS_KEY)
                    if any(_as_text(pipe.hget(_STATUS_KEY, k)) != v for k, v in expected.items()):
                        return False
                    pipe.multi()
                    pipe.hset(_STATUS_KEY, mapping=mapping)
                    pipe.execute()
                    return True
                except redis.WatchError:
                    continue
    except redis.RedisError as exc:
        raise JobBackendError("job_store_unavailable", _STORE_UNAVAILABLE_MSG) from exc
    raise JobBackendError("job_store_unavailable", _STORE_UNAVAILABLE_MSG)


def _parse_result(raw: str) -> "Any":
    """result 以紧凑 JSON 存 hash（Redis 无嵌套结构）；空串/坏 JSON → None（坏行只该退化成
    「结果读不出来」，不该让状态端点整条 500）。"""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _write_current(job_id: str | None, mapping: dict[str, str], *, quiet: bool = False) -> bool:
    """旧 worker 不能改写新一轮；终态写失败可由 RQ 返回值在读侧补偿。"""
    try:
        if job_id:
            return _compare_hash({_JOB_ID_FIELD: job_id, "status": "running"}, mapping)
        _write_hash(mapping)                           
        return True
    except (JobBackendError, redis_store.RedisUnavailable, *redis_store.redis_error_types()):
        if not quiet:
            raise
        return False


def _reconcile_running(data: dict[str, str]) -> dict[str, str]:
    """只对 running 查真实 RQ 生命周期；不自动重跑可能已发生的写入。"""
    if data.get("status") != "running":
        return data
    job_id = data.get(_JOB_ID_FIELD, "")
    terminal = {"status": "failed", "finished_at": _now(), "result": "",
                "error": "队列任务已中断或缺失，请核对已导入的数据后重试。", _FLAG_FIELD: "1",
                _GENERATION_FIELD: job_id or uuid.uuid4().hex}
    try:
        from rq.job import Job
        from rq.exceptions import NoSuchJobError
        try:
            job = Job.fetch(job_id, connection=_rq_client(), serializer=RQ_SERIALIZER) if job_id else None
        except NoSuchJobError:
            job = None
        if job is None:
            try:
                age = time.time() - float(data.get(_CLAIMED_AT_FIELD, "0"))
            except ValueError:
                age = _ENQUEUE_GRACE_SECONDS
            if job_id and age < _ENQUEUE_GRACE_SECONDS:
                return data
        else:
            status = job.get_status(refresh=True)
            status = getattr(status, "value", status)
            if status == "started":
                heartbeat = job.last_heartbeat or job.started_at
                timeout = job.timeout or 0
                age = ((datetime.datetime.now(datetime.timezone.utc) - heartbeat).total_seconds()
                       if heartbeat is not None else 0)
                                                       
                if timeout <= 0 or age <= timeout + 60:
                    return data
            elif status in {"created", "queued", "deferred", "scheduled"}:
                return data
            if status == "finished":
                outcome = job.return_value(refresh=True)
                if isinstance(outcome, dict) and outcome.get("status") in {"done", "failed"}:
                    terminal.update({k: outcome[k] for k in terminal if k in outcome})
                else:
                    terminal["error"] = "队列任务已结束，但结果记录缺失；请核对已导入的数据后重试。"
        if _compare_hash({"status": "running", _JOB_ID_FIELD: job_id}, terminal):
            return {**data, **terminal}
        return _read_hash()
    except ImportError as exc:
        raise JobBackendError("job_backend_unavailable", "队列依赖不可用，暂时无法核对任务状态。") from exc
    except redis_store.redis_error_types() as exc:
        raise JobBackendError("job_store_unavailable", _STORE_UNAVAILABLE_MSG) from exc
    except JobBackendError:
        raise
    except Exception as exc:
        raise JobBackendError("job_payload_unavailable",
            "队列记录无法按 JSON 读取；请核对旧作业，不会自动迁移或重复执行。") from exc


def _cleanup_acknowledged_job(data: dict[str, str]) -> None:
    if data.get("status") not in {"done", "failed"} or not data.get(_JOB_ID_FIELD):
        return
    try:
        from rq.job import Job
        from rq.exceptions import NoSuchJobError
    except ImportError:
        return
    try:
        job = Job.fetch(data[_JOB_ID_FIELD], connection=_rq_client(), serializer=RQ_SERIALIZER)
        status = job.get_status(refresh=True)
        if getattr(status, "value", status) == "finished":
            job.delete(remove_from_queue=False)
    except NoSuchJobError:
        pass
    except Exception as exc:
        logging.getLogger(__name__).warning("RQ compensation cleanup deferred (%s)", type(exc).__name__)


def snapshot() -> dict:
    """读状态快照，形状与 `webapp._corpus_sync_job_snapshot()` 逐键一致。

    空 hash（从未跑过）→ idle 形状（五个键全在，值为 None，与内存 idle 同形同值）。
    读失败 → `JobBackendError("job_store_unavailable")`：RQ 路径下状态真源就在 Redis，
    读不到只能如实报错，伪造 idle 会让用户以为任务不存在。
    """
    data = _reconcile_running(_read_hash())
    _cleanup_acknowledged_job(data)
    if not data.get("status"):
        return {"status": "idle", "started_at": None, "finished_at": None,
                "result": None, "error": None}
    return {
        "status": data["status"],
        "started_at": data.get("started_at") or None,
        "finished_at": data.get("finished_at") or None,
        "result": _parse_result(data.get("result", "")),
        "error": data.get("error") or None,
    }


def start(sources: "list[str] | None") -> dict:
    """启动或附着 corpus-sync job，返回最新快照（与内存路径单飞语义一致）。

    running 中重复触发 → 不新建、不抛 busy，直接返回现状（调用方看到的永远是同一个 job）。

    入队顺序：**先落 running 再 enqueue**。反过来写会把竞态变成 bug——worker 可能瞬间取走
    任务并写终态，随后 web 的 running 写入会把 done 覆盖成 running，状态永远停在 running。
    """
    observed = _reconcile_running(_read_hash())
    if observed.get("status") == "running":
        return snapshot()

    try:
        import rq                               
    except ImportError as exc:
        raise JobBackendError(
            "job_backend_unavailable",
            "未安装 RQ 消息队列依赖包，无法启用消息队列后端；请先执行 "
            "pip install -r requirements/requirements-mq.txt，"
            f"或去掉 {_BACKEND_ENV} 环境变量以回到默认形态。",
        ) from exc

                               
    client = _rq_client()
    queue = rq.Queue(_QUEUE_NAME, connection=client, serializer=RQ_SERIALIZER)
    _cleanup_acknowledged_job(observed)
    job_id = f"corpus-sync-{uuid.uuid4().hex}"
    claimed = _compare_hash({k: observed.get(k, "") for k in ("status", _JOB_ID_FIELD)}, {
        "status": "running",
        "started_at": _now(),
        "finished_at": "",
        "result": "",
        "error": "",
        _FLAG_FIELD: "0",
        _JOB_ID_FIELD: job_id,
        _CLAIMED_AT_FIELD: str(time.time()),
    })
    if not claimed:
        return snapshot()
    try:
        queue.enqueue(_ENTRY_PATH, sources, job_id=job_id, job_timeout=CORPUS_SYNC_JOB_TIMEOUT_SECONDS,
                      result_ttl=_RESULT_TTL)
    except Exception as exc:
        try:
            _reconcile_running(_read_hash())
        except JobBackendError:
            pass
        raise JobBackendError("job_enqueue_failed",
            "任务入队确认失败，可能已经提交；请核对任务状态，不会自动重复执行。") from exc
    return snapshot()


def corpus_sync_entry(sources: "list[str] | None") -> dict | None:
    """worker 进程侧执行体：sync →（有新增）向量重建 → 写终态与共享代际。

    worker 清不了 Web 的内存缓存；每个 Web 进程按自己的代际游标调用既有失效口。
    遗留旗标保留只读兼容，新任务的代际不在启动或首消费者读取时清零。

    业务异常收口为 failed；返回同一终态给 RQ 持久化，以补偿业务 hash 写失败。
    已过时的排队任务零执行返回 None，不允许覆盖新一轮状态。
    """
    from rq import get_current_job
    job = get_current_job()
    job_id = job.id if job is not None else None
    if not _write_current(job_id, {
        "status": "running",
        "started_at": _now(),
        "finished_at": "",
        "result": "",
        "error": "",
        _FLAG_FIELD: "0",
    }):
        return None
    result: "Any" = None
    error = None
    try:
        from . import webapp                                    

        result = webapp._corpus_sync_execute(sources)
        vec_err = None
        if int(result.get("imported_total") or 0) > 0:
            vec_err = webapp._corpus_sync_rebuild_vectors()
        error = vec_err
    except Exception as exc:                                           
        error = getattr(exc, "hint", "") or str(exc) or type(exc).__name__
    terminal = {
        "status": "failed" if error else "done", "finished_at": _now(),
        "result": "" if result is None else json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        "error": error or "", _FLAG_FIELD: "1",
        _GENERATION_FIELD: job_id or uuid.uuid4().hex,
    }
    acknowledged = _write_current(job_id, terminal, quiet=True)
    if acknowledged and job is not None:
        job.result_ttl = 0
    return terminal


class InvalidationConsumer:
    def __init__(self):
        self.last_generation = None
        self._lock = threading.Lock()

    def consume(self, invalidate=None) -> bool:
        with self._lock:
            try:
                data = _reconcile_running(_read_hash())
            except (JobBackendError, redis_store.RedisUnavailable, *redis_store.redis_error_types()):
                logging.getLogger(__name__).warning("RQ generation unavailable; local caches are not trusted")
                if invalidate is not None:
                    invalidate()
                return False
            generation = data.get(_GENERATION_FIELD)
            if not generation and data.get(_FLAG_FIELD) == "1":
                generation = "legacy:" + data.get(_JOB_ID_FIELD, "") + ":" + data.get("finished_at", "")
            if not generation or generation == self.last_generation:
                return False
            if invalidate is not None:
                invalidate()
            self.last_generation = generation
            return True


_WEB_INVALIDATION_CONSUMER = InvalidationConsumer()


def consume_web_invalidate_flag(invalidate=None, *, consumer=None) -> bool:
    """兼容旧入口名；代际只读，每个 consumer 在成功失效后记录自己的游标。"""
    return (consumer or _WEB_INVALIDATION_CONSUMER).consume(invalidate)


__all__ = [
    "JobBackendError",
    "RQ_SERIALIZER",
    "InvalidationConsumer",
    "backend_active",
    "snapshot",
    "start",
    "corpus_sync_entry",
    "consume_web_invalidate_flag",
]
