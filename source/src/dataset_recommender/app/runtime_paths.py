                       
"""运行时路径**单一真源**（安装器工程 W1）：resource/data 双根解析与全部路径派生。

背景与动机
----------
项目历史上每条路径都各自 `Path(__file__).resolve().parents[3]` 推导（约 20 处），
frozen（PyInstaller 打包）后这些推导全部失效——源码树不在磁盘上、数据库/模型/日志
不可写（`sys._MEIPASS` 是临时解包目录）。本模块收口为唯一解析点，提供三模式：

- ``"source"``   ：从源码树运行（开发 / 评测 / 既有启动器）。resource == data == 项目根，
  所有子路径与历史布局**逐字节一致**（.userdata、database/external、database/trace、
  models、outputs、项目根/.env）。
- ``"portable"`` ：非 frozen，但 `BIODATA_RESOURCE_ROOT` / `BIODATA_DATA_ROOT` 显式指定
  （便携安装 / 数据分离部署）。同根时与 source 逐字节一致；显式指向不同目录时启用
  与 frozen 相同的**双根分离布局**。
- ``"frozen"``    ：PyInstaller 打包。install_root = exe 所在目录；resource_root =
  `sys._MEIPASS`（随包静态资源，只读）；data_root = `%LOCALAPPDATA%\\BioDataAgent`
  （用户实例数据，可写）。

解析优先级：`BIODATA_RESOURCE_ROOT`/`BIODATA_DATA_ROOT` 显式环境变量 > frozen 状态
（`getattr(sys, "frozen", False)` + `sys._MEIPASS`）> 源码项目根
（`Path(__file__).resolve().parents[3]`）。既有更具体覆盖（`BIODATA_LLM_ENV_FILE`、
`BIODATA_ACCOUNTS_FILE`、`BIODATA_SESSIONS_FILE`、`DATA_DIR` 等）在各自消费点仍然优先于
本模块的 AppPaths（本模块只管「根」，不管单文件覆盖）。

分工原则：**读静态资源用 resource_root，写盘用 data_root**。两层都只提供相对项目的
子路径结构——frozen 下官方快照（database/base、database/external/*.json 随包部分）从
resource 层读，用户数据（.userdata、上传、models、trace、日志、导出）从 data 层读写。

对外公共函数
------------
- `get_app_paths() -> AppPaths`       ：缓存单例；进程内只解析一次。
- `reset_app_paths_cache()`           ：清缓存（测试用：改 env / sys.frozen 后重算）。
- `instance_data_dir_for(root, rel)`  ：给定调用方项目根，解析实例**数据**子目录
  （frozen 布局实例根 → data_root/rel；其余 → root/rel）。
- `resource_file_for(root, rel)`      ：给定调用方项目根，解析随包**静态资源**文件
  （frozen 布局实例根 → resource_root/rel；其余 → root/rel）。
- `uses_split_layout(root)`           ：该调用方项目根是否启用双根分离（frozen 布局
  且 root == 实例 data_root）。
- `atomic_write_json(path, payload)`  ：JSON 落盘唯一原子写通道（随机 tmp 后缀 +
  os.replace）；账户/会话/配额/令牌/补丁库共用。
- `repo_database_dir()`               ：仓库 `database/` 目录单一真源（路径守卫共用）。
- `assert_runtime_path(path, error_cls)`：运行时状态文件绝不落进仓库 `database/` 的守卫
  （2026-08-10 架构评审裁决；各存储模块以自己的异常类型薄委托）。
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

RESOURCE_ROOT_ENV = "BIODATA_RESOURCE_ROOT"
DATA_ROOT_ENV = "BIODATA_DATA_ROOT"

                                                                     
_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AppPaths:
    """运行时路径快照。frozen 双根分离时各字段按设计裁决落位；source/portable 单根时
    resource == data == install，所有子路径与历史布局逐字节一致。"""

    install_root: Path                                                     
    resource_root: Path                                                            
    data_root: Path                                                                               
    config_root: Path                                                                               
    shipped_base_dir: Path                                            
    shipped_external_dir: Path                                               
    user_external_dir: Path                                              
    userdata_dir: Path                                                  
    model_root: Path                                      
    log_root: Path                                 
    trace_root: Path                                                      
    export_root: Path                                                                        
    run_root: Path                                             
    runtime_mode: str                                           


def _env_root(name: str) -> Path | None:
    """读显式环境变量根（expanduser；空/纯空白视为未设置）。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def default_data_root_frozen() -> Path:
    """frozen 下 data_root 的缺省：%LOCALAPPDATA%/BioDataAgent。LOCALAPPDATA 缺失的
    防御兜底 = exe 目录旁（Windows 环境变量异常的罕见情形，保证可写目录仍可解析）。

    公开供启动器做「旧根 .env 未随 BIODATA_DATA_ROOT 重定向」的检测（A2-L5）。"""
    local = os.getenv("LOCALAPPDATA") or os.getenv("LOCAL_APPDATA")
    if local:
        return Path(local) / "BioDataAgent"
    return Path(sys.executable).resolve().parent / "BioDataAgent"


def _resolve_roots() -> "tuple[Path, Path, Path, str]":
    """三模式根解析：(install_root, resource_root, data_root, runtime_mode)。

    优先级：显式环境变量 > frozen 状态 > 源码项目根。显式环境变量可各自独立覆盖
    resource/data 根（frozen 下同样生效——测试与数据重定向场景）。"""
    explicit_resource = _env_root(RESOURCE_ROOT_ENV)
    explicit_data = _env_root(DATA_ROOT_ENV)
    if bool(getattr(sys, "frozen", False)):
        install = Path(sys.executable).resolve().parent
        resource = explicit_resource or Path(getattr(sys, "_MEIPASS", install))
        data = explicit_data or default_data_root_frozen()
        return install, resource, data, "frozen"
    if explicit_resource is not None or explicit_data is not None:
                                               
                                         
        return _SOURCE_PROJECT_ROOT, explicit_resource or _SOURCE_PROJECT_ROOT, \
            explicit_data or _SOURCE_PROJECT_ROOT, "portable"
    return _SOURCE_PROJECT_ROOT, _SOURCE_PROJECT_ROOT, _SOURCE_PROJECT_ROOT, "source"


@lru_cache(maxsize=1)
def get_app_paths() -> AppPaths:
    """解析并缓存 AppPaths（进程内单一真源）。测试改 env / sys.frozen 后调用
    `reset_app_paths_cache()` 重算。"""
    install, resource, data, mode = _resolve_roots()
    if resource != data:
                                                            
                                  
        return AppPaths(
            install_root=install,
            resource_root=resource,
            data_root=data,
            config_root=data / "config",
            shipped_base_dir=resource / "database" / "base",
            shipped_external_dir=resource / "database" / "external",
            user_external_dir=data / "database" / "external",
            userdata_dir=data / ".userdata",
            model_root=data / "models",
            log_root=data / "logs",
            trace_root=data / "database" / "trace",
            export_root=data / "exports",
            run_root=data / "run",
            runtime_mode=mode,
        )
                                                   
    return AppPaths(
        install_root=install,
        resource_root=resource,
        data_root=data,
        config_root=resource,
        shipped_base_dir=resource / "database" / "base",
        shipped_external_dir=resource / "database" / "external",
        user_external_dir=resource / "database" / "external",
        userdata_dir=resource / ".userdata",
        model_root=resource / "models",
        log_root=resource / "logs",
        trace_root=resource / "database" / "trace",
        export_root=resource / "outputs",
        run_root=resource / "run",
        runtime_mode=mode,
    )


def reset_app_paths_cache() -> None:
    """清缓存（仅供测试）：修改 env / sys.frozen 模拟后调用，下次 get_app_paths 重算。"""
    get_app_paths.cache_clear()


def _is_instance_data_root(root: Path) -> bool:
    """调用方传入的项目根是否就是当前实例的 data_root（判定「实例自身调用」）。

    source/portable 单根下 data_root == 项目根，任何传项目根的调用都命中——但此时
    resource == data、无分离布局，`uses_split_layout` 的第二个条件会把它挡回根相对。"""
    try:
        return Path(root).resolve() == get_app_paths().data_root.resolve()
    except OSError:
        return False


def uses_split_layout(project_root: Path) -> bool:
    """该调用方项目根是否触发 resource/data 双根分离（frozen 布局且 root == 实例
    data_root）。测试注入的临时根 ≠ data_root → 恒 False，保持历史单目录语义。"""
    paths = get_app_paths()
    return paths.resource_root != paths.data_root and _is_instance_data_root(Path(project_root))


def instance_data_dir_for(project_root: Path, rel: str) -> Path:
    """给定调用方项目根，解析实例**数据**子目录：frozen 布局实例根 → data_root/rel；
    其余（source/portable/测试注入根）→ project_root/rel（与历史逐字节一致）。

    `rel` 为相对路径段（如 ``"database/external"``、``".userdata"``、``"database/trace"``）。
    这是写盘侧的统一入口——写操作永远落 data 层。"""
    if uses_split_layout(Path(project_root)):
        return get_app_paths().data_root / rel
    return Path(project_root) / rel


def account_data_dir_for(project_root: Path, rel: str, principal: str | None = None) -> Path:
    """Resolve an account-owned runtime directory; unbound local callers keep their layout."""
    base = instance_data_dir_for(project_root, rel)
    if principal is None or principal == "":
        return base
    owner = str(principal)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", owner):
        raise ValueError("Invalid runtime account identifier")
    return base / ("u_" + owner)


def resource_file_for(project_root: Path, rel: str) -> Path:
    """给定调用方项目根，解析随包**静态资源**文件：frozen 布局实例根 → resource_root/rel；
    其余 → project_root/rel。读官方快照/冻结基准/提示词等只读资源用这一侧。"""
    if uses_split_layout(Path(project_root)):
        return get_app_paths().resource_root / rel
    return Path(project_root) / rel


def _atomic_write(path: Path, write, *, binary: bool) -> None:
    """所有单文件原子写共用替换与失败清理；调用方决定原始字节或文本格式。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        kwargs = {} if binary else {"encoding": "utf-8"}
        with open(tmp, "wb" if binary else "w", **kwargs) as handle:
            write(handle)
        for attempt in range(4):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.05)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """保留精确字节的原子替换；失败不截断既有文件，供快照及回滚复用。"""
    _atomic_write(path, lambda handle: handle.write(payload), binary=True)


def atomic_write_json(path: Path, payload: Any, *, indent: "int | None" = 2) -> None:
    """JSON 落盘唯一原子写通道：同目录随机 tmp 后缀 + os.replace（rename 原子性），
    防半路崩溃留半截文件。账户/会话/配额/令牌/补丁库共用本函数。

    `indent` 默认 2（账户库/令牌库历史格式）；紧凑格式传 ``indent=None``（会话库/
    配额账本/补丁包历史格式）。文本模式默认换行翻译（Windows 下 CRLF）——与各模块
    历史落盘字节逐字节一致，刻意不 fsync、不固定 ``newline``（desktop_launcher 那款
    带 fsync + LF 的 `_atomic_write_json` 语义不同，不属于本通道）。"""
    _atomic_write(path, lambda handle: json.dump(
        payload, handle, ensure_ascii=False, indent=indent), binary=False)


def repo_database_dir() -> Path:
    """仓库 `database/` 目录单一真源（冻结基准与元数据库所在）；路径守卫共用。"""
    return _SOURCE_PROJECT_ROOT / "database"


def assert_runtime_path(path: Path, error_cls: "type[Exception]") -> Path:
    """运行时状态文件绝不许落进仓库 `database/`——那里是冻结基准与元数据库，环境变量
    误配也不许把运行时写引进来。

    两侧都做同一规范化（resolve）：仓库 `database/` 自身可能是符号链接/目录联接，
    只 resolve 候选会让「指向其真实目录」的路径因前缀不同而漏过。

    `error_cls` 须为 ``(code, message)`` 构造的异常类型（AccountError / McpTokenError
    同款约定）；patch_package 的守卫文案保留补丁包语境，不走本函数（仅共用目录真源）。"""
    resolved = Path(path).expanduser().resolve()
    forbidden = repo_database_dir().resolve()
    try:
        resolved.relative_to(forbidden)
    except ValueError:
        return resolved
    raise error_cls(
        "bad_store_path",
        f"运行时状态文件不许落在仓库 database/ 目录内（收到 {resolved}）；请改用 .userdata/ 或仓库外路径。")


__all__ = [
    "RESOURCE_ROOT_ENV",
    "DATA_ROOT_ENV",
    "AppPaths",
    "get_app_paths",
    "reset_app_paths_cache",
    "instance_data_dir_for",
    "resource_file_for",
    "uses_split_layout",
    "atomic_write_json",
    "repo_database_dir",
    "assert_runtime_path",
]
