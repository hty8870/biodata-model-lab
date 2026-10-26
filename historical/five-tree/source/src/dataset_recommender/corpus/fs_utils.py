                       
"""corpus 包文件级共享基元：旁挂台账缓存装载、跨进程文件锁。唯一锚点，勿复制变体。"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

                                                                                                    
                                                                    
                                                                                                    


def make_sidecar_loader(
    *,
    data_path: Callable[[], str],
    shape_gate: Callable[[Any], Any],
    missing: Any,
) -> Callable[[], Any]:
    """by-uid 旁挂台账装载器骨架（唯一锚点）：返回 `load()`，且带 `load.cache_clear` 失效仪式。

    各台账只喂差异内容：`data_path`（路径解析，通常「现读环境变量、未设置回落 import 期快照」）、
    `shape_gate`（校验并归一文件内容）、`missing`（文件缺失时的稳定空值）。纪律随骨架一并生效：

    - 路径含进缓存键：`data_path` 每次加载现调，长驻进程内改环境变量即时生效，
      换路径自然换缓存条目。
    - 失败不入缓存：文件不存在 → 返回 `missing`（稳定状态，缓存合法）；读了但失败
      （坏 JSON / IO / 形状闸抛出）→ 异常传出不入缓存，由调用方降级——故障消除后同进程
      下次调用自动重试，不会把失败缓存到进程结束。
    """
    @lru_cache(maxsize=4)
    def load_sidecar_cached(path: str):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return missing
        return shape_gate(data)

    def load():
        try:
            return load_sidecar_cached(data_path())
        except Exception:
            return missing

    load.cache_clear = load_sidecar_cached.cache_clear                           
    return load


def by_uid_shape(label_zh: str) -> Callable[[Any], "tuple[dict, dict]"]:
    """形状闸：顶层 dict（损坏时 ValueError 文案以 `label_zh` 开头）→ 值过滤为 dict → 派生 by_url。

    只保留值为 dict 的记录：即便文件被部分损坏（某条记录值退化成标量/列表），
    下游 getter 的 r.get(...) 也不会 AttributeError（守住「永不崩→降级」合同）。"""
    def gate(data: Any) -> "tuple[dict, dict]":
        if not isinstance(data, dict):
            raise ValueError(f"{label_zh}顶层不是 dict（文件损坏）")
        by_uid = {k: v for k, v in data.items() if isinstance(v, dict)}
        by_url = {r.get("url"): r for r in by_uid.values() if r.get("url")}
        return by_uid, by_url
    return gate


                                                                                                    
                               
                                                                                                    


class WritableRootError(ValueError):
    """写目标（解析后）落入冻结基准或随包只读层——调用方翻成各自的领域错误码。"""


def _resolved_under(child: Path, parent: Path) -> bool:
    try:
        Path(child).expanduser().resolve(strict=False).relative_to(
            Path(parent).expanduser().resolve(strict=False))
        return True
    except (ValueError, OSError):
        return False


def assert_writable_dir(target: Path, *, project_root: Path | None = None) -> None:
    """写侧共享守卫（唯一锚点，勿复制变体）：写目标目录（或其链接目标）不得落进
    冻结基准 `database/base` 与随包只读资源层（分离布局的 resource_root）。

    上传/恢复等写入口只拼接 `database/external`，若该目录是指向 `database/base` 的
    符号链接/目录联接，一次正常上传就会在冻结基准新增 JSON；指向随包资源层则写到只读
    发布面。这里对**解析后**路径做前缀判定（Windows junction 由 resolve 跟随）；
    目标尚不存在也安全（strict=False 解析已存在的前缀）。常态路径与两处不相交，零影响。

    `project_root`：调用方的项目根——只有**分离布局实例根**（frozen：资源层与数据层
    分家）才做资源层前缀判定；测试注入的临时根（恰在仓库 = 资源根之内）不做，
    保持单目录语义与历史逐位一致。"""
    from ..app.runtime_paths import get_app_paths, repo_database_dir, uses_split_layout

                                                 
                                               
    frozen_candidates = [repo_database_dir() / "base"]
    if project_root is not None:
        frozen_candidates.append(Path(project_root) / "database" / "base")
    for frozen_base in frozen_candidates:
        if _resolved_under(Path(target), frozen_base):
            raise WritableRootError(
                "写入目标实际落在冻结基准 database/base 内（目录链接/联接映射）——已拒绝写入；"
                "请把 database/external 恢复为真实目录后重试。")
    if project_root is None:
        return
    try:
        if not uses_split_layout(Path(project_root)):
            return
        paths = get_app_paths()
    except Exception:
        return
                                                              
                                                           
                                   
    for shipped in (paths.shipped_base_dir, paths.shipped_external_dir):
        if _resolved_under(Path(target), shipped):
            raise WritableRootError(
                "写入目标实际落在随包只读层内（数据层目录是指向资源层快照的链接）——已拒绝写入；"
                "请修正实例数据根后重试。")


                                                                                                    
                                          
                                                                                                    


class FileLockBusy(RuntimeError):
    """`acquire_file_lock` 非阻塞被占、或等待超时仍未获得锁时抛出；句柄已关闭。
    调用方把它翻译成各自的领域错误（如 UploadError(lock_busy) / CurateError(sync_busy)）。"""


def acquire_file_lock(
    lock_path: "str | Path",
    *,
    wait: bool = True,
    timeout: float = 60.0,
    poll_interval: float = 0.1,
):
    """获取跨进程互斥文件锁并持有至 `release_file_lock`（stdlib only：msvcrt on Windows /
    fcntl on POSIX 的唯一锚点），返回打开的句柄。

    - wait=True：非阻塞尝试 + `poll_interval` 秒退避重试，至 `timeout` 秒仍未得 →
      FileLockBusy。即使 timeout=0 也先尝试一次（「先试再判」的循环形状）。
    - wait=False：单次尝试，被占立即 FileLockBusy，不等待不重试。

    锁文件按需创建、只创建不删除（删除与重建之间有竞态）；进程崩溃遗留的锁文件不永久
    busy——OS 层文件锁随进程消亡自动释放。"""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+b")
    deadline = time.monotonic() + timeout
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if not wait or time.monotonic() >= deadline:
                fh.close()
                raise FileLockBusy(f"文件锁被占用：{lock_path.name}") from None
            time.sleep(poll_interval)


def release_file_lock(fh) -> None:
    """释放 `acquire_file_lock` 拿到的锁并关闭句柄（解锁与加锁锁同一字节位：先 seek(0)）。"""
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


_PATH_LOCKS: "dict[str, threading.RLock]" = {}
_PATH_LOCKS_GUARD = threading.Lock()


def path_lock(path: Path) -> threading.RLock:
    """按路径记忆化的进程内锁（2026-10-01 整理批单源化：原 agent/trace/recorder、
    app/replay_store、app/session_log、corpus/patch_package 四处各自实现——两份 Lock
    两份 RLock，统一为 RLock 超集；跨模块同路径从此共享同一把锁）。"""
    key = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


_FILE_LOCK_STATE = threading.local()


@contextmanager
def file_critical_section(lock_path: Path, *, timeout: float = 60.0):
    """同路径读改写事务：线程 RLock → OS 文件锁，同线程同路径重入不重复锁 OS。
    锁文件只创建不删除；不同路径独立，调用方不应在事务内等待网络或模型。"""
    path = Path(os.path.normcase(str(Path(lock_path).expanduser().resolve())))
    key = str(path)
    held = getattr(_FILE_LOCK_STATE, "held", None)
    if held is None:
        held = _FILE_LOCK_STATE.held = {}
    if key in held:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    deadline = time.monotonic() + max(0.0, timeout)
    lock = path_lock(path)
    if not lock.acquire(timeout=max(0.0, timeout)):
        raise FileLockBusy(f"文件锁被占用：{path.name}")
    try:
        fh = acquire_file_lock(path, timeout=max(0.0, deadline - time.monotonic()))
        try:
            held[key] = 1
            yield
        finally:
            held.pop(key, None)
            release_file_lock(fh)
    finally:
        lock.release()


def reset_path_locks_for_test() -> None:
    """清空 path_lock 注册表（测试隔离用；运行期调用会丢弃在用的锁对象，仅限测试间调用）。"""
    with _PATH_LOCKS_GUARD:
        _PATH_LOCKS.clear()
