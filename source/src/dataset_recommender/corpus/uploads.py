                       
"""数据集上传/摄取的**单一真源**：把一个数据集 JSON 落入外部平台库 `database/external/`。

Web（`/api/upload`）与本地 stdio MCP（`upload_dataset` 工具）**共用**本模块的 `ingest_dataset`，
避免两处各写一份「落盘 + 校验 + 打标 + 缓存失效」逻辑而悄悄走样——尤其是两条安全不变量必须逐位一致：

  1. **只进外部库**：写入路径恒为 `<project_root>/database/external/`（复用 `corpus.EXTERNAL_DIR_NAME`），
     **绝不**碰冻结基准 `database/base/` → 官方 767 评测走 base-only、永不读外部库 → 上传不影响基准。
  2. **保留 upload_ 命名空间**：每个上传文件名加 `upload_<时间戳>_` 前缀，使用户上传**永不**能占用某个
     公开快照的发布白名单文件名（即使该公开文件此刻恰好不在）。

摄取本身**非确定性**（时间戳文件名、同秒防冲突重试），这是写操作的固有属性；检索侧的确定性不受影响。
错误以 `UploadError(code, hint)` 抛出（携带机器码 + 人读提示），由各调用方翻译成自己的错误契约：
Web → `HTTPException(400, hint)`；MCP → `ToolError(f"{code}: {hint}")`（客户端 isError=true）。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..app.runtime_paths import instance_data_dir_for
from . import fs_utils
from .corpus import EXTERNAL_DIR_NAME, invalidate_external_cache, publish_corpus_write
from .data_loader import extract_records

                                                    
DEFAULT_UPLOAD_SOURCE = "用户上传"
                                                    
DEFAULT_UPLOAD_NOTE = "用户上传（网页端 /api/upload）。"

                                                      
                                                                           
                                                             
                                           
MAX_INGEST_RECORDS = 200_000

                                        
                                             
EXTERNAL_TOTAL_MAX_RECORDS = 1_000_000

                                          
                                                                  
                                                   
_INGEST_LOCK = threading.Lock()

                                                
_INGEST_LOCK_FILE_NAME = "upload_ingest.lock"
                                                              
                                                   
_INGEST_LOCK_TIMEOUT_DEFAULT = 60.0
_ingest_lock_state = threading.local()                        


def _ingest_lock_timeout() -> float:
    try:
        return float(os.environ.get("BIODATA_INGEST_LOCK_TIMEOUT", "") or _INGEST_LOCK_TIMEOUT_DEFAULT)
    except ValueError:
        return _INGEST_LOCK_TIMEOUT_DEFAULT


def _acquire_os_ingest_lock(project_root: Path):
    """获取跨进程 OS 文件锁（等待/平台骨架唯一锚点在 `fs_utils.acquire_file_lock`），返回句柄。
    非阻塞尝试 + 100ms 退避，到 `_ingest_lock_timeout()` 仍未得 → UploadError(lock_busy)。"""
    lock_path = instance_data_dir_for(Path(project_root), ".userdata") / _INGEST_LOCK_FILE_NAME
    try:
        return fs_utils.acquire_file_lock(lock_path, timeout=_ingest_lock_timeout())
    except fs_utils.FileLockBusy:
        raise UploadError(
            "lock_busy",
            "另一个进程正在写入外部库（等待摄取锁超时）。本次没有任何写入；请稍后重试。",
        ) from None


def _release_os_ingest_lock(fh) -> None:
    """释放 `_acquire_os_ingest_lock` 拿到的锁并关闭句柄。"""
    fs_utils.release_file_lock(fh)


@contextlib.contextmanager
def ingest_critical_section(project_root: Path):
    """摄取临界区（进程内线程锁 + 跨进程 OS 文件锁；**同线程可重入**）。

    计数 → 落盘 → 强制流水账同一临界区。`ingest_dataset` 自身经它进入；curate 的
    「实体级重检 → 落盘」复合段（apply_search_online / sync_updates）也经它闭合
    TOCTOU——重检与落盘之间不再有任何通路能插队（残余跨进程微竞态的收口）。
    """
    depth = getattr(_ingest_lock_state, "depth", 0)
    if depth > 0:                                       
        _ingest_lock_state.depth = depth + 1
        try:
            yield
        finally:
            _ingest_lock_state.depth = depth
        return
    with _INGEST_LOCK:                                              
        fh = _acquire_os_ingest_lock(project_root)
        try:
            _ingest_lock_state.depth = 1
            yield
        finally:
            _ingest_lock_state.depth = 0
            _release_os_ingest_lock(fh)
                                                  
KNOWN_SPECIES_LOWER = {
    "human", "mouse", "rat", "zebrafish", "drosophila", "macaque", "marmoset",
    "chimpanzee", "chicken", "pig", "dog", "rabbit", "cattle",
}
                                                           
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")
                                                        
_STAMP_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]{6}")


                                                                       
                                                                      
                                                        
                                                 
                               
UploadCode = Literal["bad_file", "bad_encoding", "invalid_json", "no_records",
                     "too_large", "journal_failed", "lock_busy", "uid_conflict",
                                                         
                     "bad_store_path"]


class UploadError(ValueError):
    """带机器码的摄取失败。`code` 供调用方分类映射，`hint` 供人读定位；`str()` 为「code: hint」。

    调用方翻译：Web → HTTPException(400, hint)；MCP → ToolError("code: hint")（isError=true）。
    继承 ValueError，向后兼容任何 `except ValueError` 的旧调用点。"""

    def __init__(self, code: UploadCode, hint: str) -> None:
        self.code = code
        self.hint = hint
        super().__init__(f"{code}: {hint}")


@dataclass
class UploadResult:
    """一次成功摄取的结果（供各接口层组装自己的响应）。"""

    filename: str                                           
    saved_to: str                                                                            
    record_count: int                     
    sources: dict[str, int]                         
    warnings: list[str]                       


def sanitize_upload_name(filename: str) -> str:
    """取文件名叶子 + 净化非法字符；**必须以 .json 结尾**，否则 UploadError(bad_file)。"""
    leaf = Path(filename).name
    cleaned = SAFE_FILENAME_PATTERN.sub("_", leaf)
    if not cleaned.lower().endswith(".json"):
        raise UploadError("bad_file", "Only .json files are supported.")
    return cleaned


def new_upload_name(filename: str, timestamp: str | None = None) -> str:
    """给每个上传文件一个保留前缀，与公开快照文件名区隔（防占用发布白名单名）。"""
    cleaned = sanitize_upload_name(filename)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not _STAMP_PATTERN.fullmatch(stamp):
        raise ValueError("upload timestamp must use YYYYMMDD_HHMMSS_microseconds")
    return f"upload_{stamp}_{cleaned}"


def first_nonempty(record: dict, keys: tuple[str, ...]) -> str:
    """按候选键（大小写不敏感）取第一个非空值的字符串；都没有则空串。用于上传校验提示。"""
    low = {str(k).lower(): v for k, v in record.items()}
    for k in keys:
        v = low.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def decode_json_bytes(raw_bytes: bytes) -> str:
    """把上传字节按 utf-8-sig（容 BOM）→ utf-8 解码；都失败 → UploadError(bad_encoding)。"""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UploadError(
        "bad_encoding",
        f"File must be valid UTF-8 JSON. decode errors: {' | '.join(errors)}",
    )


def _external_record_total(project_root: Path) -> int:
    """external 全库累计条数（写入汇自用口径：逐文件装载计数；坏文件跳过不计——
    与 loader 的宽容装载同哲学）。只在持 `ingest_critical_section` 时调用（计数与落盘同一临界区）。"""
    ext_dir = Path(project_root) / EXTERNAL_DIR_NAME
    if not ext_dir.is_dir():
        return 0
    total = 0
    for path in sorted(ext_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            total += len(extract_records(payload))
        except (OSError, ValueError):
            continue
    return total


def _append_upload_journal(project_root: Path, entry: dict) -> None:
    """强制摄取流水账：每次成功落盘必须有一行
    `.userdata/uploads_journal.jsonl`（action/filename/record_count/sha256/时间/备注/请求来源）。
    写与账不可分离——账写不进去 → 调用方回滚落盘文件并抛 journal_failed。"""
    path = instance_data_dir_for(Path(project_root), ".userdata") / "uploads_journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_orphaned_uploads(external_dir: Path, journal_path: Path) -> list[str]:
    """扫描 `external_dir` 的 `upload_*.json`，返回在 `journal_path` 中**无对应
    `filename`** 的名单（仅告警用，不自动删、不阻塞启动）。

    历史/极端 kill 可能留下「文件在、账不在」的残缺态（波次B 已把正常写入改成
    `.tmp → 流水账 → os.replace` 原子写，不会再新增该态；本函数只做启动期的遗留告警）。
    语义：
    - 目录/账本缺失 → 返回空（或保守全部告警）；
    - 账本行损坏/缺 filename → 该行视为「无账」；
    - 只认 `upload_` 前缀的 `.json`（官方快照 geo.json 等不带此前缀，天然不参与）。"""
    journaled: set[str] = set()
    try:
        if journal_path.is_file():
            with journal_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict):
                        filename = entry.get("filename")
                        if isinstance(filename, str) and filename:
                            journaled.add(filename)
    except OSError:
                                                       
        journaled = set()

    if not external_dir.is_dir():
        return []
    try:
        names = sorted(p.name for p in external_dir.glob("upload_*.json") if p.is_file())
    except OSError:
        return []
    return [name for name in names if name not in journaled]


def _tag_records_for_ingest(
    payload: Any, records: list[dict], form_source: str
) -> "tuple[str, dict[str, int], int, dict[str, int]]":
    """逐条打来源标 + 收集校验素材（2026-08-26 从 `_ingest_write_and_journal` 抽出，
    供 external 落盘与补丁包两条写入路径共用同一口径；行为与原内联实现逐位一致）。

    返回 (fallback_source, source_counts, missing_name, unknown_species)。"""
                                                                   
    wrapper_source = str(payload.get("source") or "").strip() if isinstance(payload, dict) else ""
    fallback_source = (form_source or "").strip() or wrapper_source or DEFAULT_UPLOAD_SOURCE

    source_counts: dict[str, int] = {}
    missing_name = 0
    unknown_species: dict[str, int] = {}
    for r in records:
        rec_src = str(r.get("source") or "").strip() or fallback_source
        r["source"] = rec_src                                  
        source_counts[rec_src] = source_counts.get(rec_src, 0) + 1
        if not first_nonempty(r, ("dataset_name", "name", "title", "dataset_title", "dataset")):
            missing_name += 1
        sp = first_nonempty(r, ("species", "organism"))
        if sp and not any(k in sp.lower() for k in KNOWN_SPECIES_LOWER):
            unknown_species[sp] = unknown_species.get(sp, 0) + 1
    return fallback_source, source_counts, missing_name, unknown_species


def _ingest_warnings(missing_name: int, unknown_species: dict[str, int]) -> list[str]:
    """可读校验提示（两条路径同口径；空列表=无问题）。"""
    warnings: list[str] = []
    if missing_name:
        warnings.append(f"{missing_name} 条缺少 dataset_name（数据集名称），可能不会被展示或检索。")
    if unknown_species:
        shown = "、".join(list(unknown_species)[:3])
        warnings.append(
            f"物种字段用了非通用名（如 {shown}）。物种筛选按英文通用名匹配（Human/Mouse…），"
            "这些记录可能被物种约束漏掉，建议改成英文通用名。"
        )
    return warnings


def _ingest_write_and_journal(
    *,
    payload: Any,
    records: list[dict],
    raw_bytes: bytes,
    safe_name: str,
    project_root: Path,
    form_source: str,
    note: str,
) -> UploadResult:
    """摄取的事务段（持 `_INGEST_LOCK` 调用）：打标 → 写 `.tmp` → 强制流水账 → 原子正名 → 清缓存。
    账写不进去 → 删 `.tmp` 回滚 + UploadError（journal_failed)——「文件在、账不在」正是要灭的
    状态。2026-08-21 起落盘改为 `.tmp` → 记账 → `os.replace` 正名：杀进程只留 `.tmp` 残留
    （loader 只 glob `*.json`，不会被当残缺 JSON 装载），不再出现「upload_*.json 在、账不在」。"""
    fallback_source, source_counts, missing_name, unknown_species = _tag_records_for_ingest(
        payload, records, form_source)

    upload_dir = instance_data_dir_for(Path(project_root), EXTERNAL_DIR_NAME)
                                                      
                                           
    try:
        fs_utils.assert_writable_dir(upload_dir, project_root=Path(project_root))
    except fs_utils.WritableRootError as exc:
        raise UploadError("bad_store_path", str(exc)) from None
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / safe_name
    if target.exists():
        stem, suffix = target.stem, target.suffix
        while True:                                         
            cand = upload_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
            if not cand.exists():
                target = cand
                break
                                                       
    out = {
        "source": fallback_source,
        "note": note,
        "record_count": len(records),
        "records": records,
    }
                                                             
                                                                 
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _append_upload_journal(project_root, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "ingest",
            "filename": target.name,
            "record_count": len(records),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "note": note,
            "form_source": form_source or "",
        })
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        invalidate_external_cache()
        raise UploadError(
            "journal_failed",
            f"摄取流水账写不进去（{exc}）——文件已回滚，没有写入；请检查 .userdata 目录可写性。",
        ) from exc
    try:
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    invalidate_external_cache()                                 
    publish_corpus_write(project_root)                              

    return UploadResult(
        filename=target.name,
        saved_to=target.relative_to(project_root).as_posix(),
        record_count=len(records),
        sources=source_counts,
        warnings=_ingest_warnings(missing_name, unknown_species),
    )


def ingest_dataset(
    *,
    raw_bytes: bytes,
    safe_name: str,
    project_root: Path,
    form_source: str = "",
    note: str = DEFAULT_UPLOAD_NOTE,
) -> UploadResult:
    """摄取一份数据集 JSON 字节流 → 规范化落入 `database/external/`（**绝不入** base）→ 即时可检索。

    调用方须先用 `new_upload_name(filename)` 生成 `safe_name`（含 upload_ 前缀与 .json 校验），
    再把原始字节与该名一起传入本函数。本函数负责：解码 → 解析 → 抽取记录 → **条数闸
    （单文件 + 全库累计，写入汇真源）** → 逐条打来源标签 → 收集可读校验提示 → 防冲突落盘 →
    **强制流水账** → 清外部库缓存。

    参数：
      raw_bytes:    原始上传字节（未解码）。
      safe_name:    `new_upload_name` 生成的目标文件名（upload_ 前缀 + .json）。
      project_root: 仓库根；写入目录恒为 `project_root/database/external/`。
      form_source:  表单/查询显式来源（优先级：每条自带 source > 本参 > 文件包裹层 source > 默认）。
      note:         落盘包裹的溯源备注（默认网页端；MCP 传自己的）。
    失败：UploadError(bad_encoding / invalid_json / no_records / too_large / journal_failed)。
    """
    text = decode_json_bytes(raw_bytes)
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UploadError("invalid_json", f"不是合法 JSON：{exc}") from exc

    records = [r for r in extract_records(payload) if isinstance(r, dict)]
    if not records:
        raise UploadError(
            "no_records",
            '未解析出任何数据集记录。文件应是记录数组 [ {…} ]，或对象 { "records": [ {…} ] }。',
        )
                                            
                                                                     
    if len(records) > MAX_INGEST_RECORDS:
        raise UploadError(
            "too_large",
            f"记录数 {len(records)} 超过单文件上限 {MAX_INGEST_RECORDS} 条。"
            "本产品面向数据集元数据管护，超大规模目录请拆分文件后再导入。",
        )
                                                     
                                                                          
                                                    
    from .patch_package import current_patch_scope                     
    scope = current_patch_scope()
    if scope:
        from .patch_package import ingest_records_to_patch
        return ingest_records_to_patch(
            payload=payload, records=records, project_root=Path(project_root),
            account_id=scope, form_source=form_source, note=note)
    with ingest_critical_section(project_root):
        existing_total = _external_record_total(project_root)
        if existing_total + len(records) > EXTERNAL_TOTAL_MAX_RECORDS:
            raise UploadError(
                "too_large",
                f"外部库现有 {existing_total} 条，再导入 {len(records)} 条会超过全库累计上限 "
                f"{EXTERNAL_TOTAL_MAX_RECORDS} 条——这个上限防的是全库加载拖垮检索"
                "（150 万条实测 /api/datasets 单请求 37.8s）。"
                "请先把不再用的文件移入回收站，或把目录拆分管理后再导入。",
            )
        return _ingest_write_and_journal(
            payload=payload, records=records, raw_bytes=raw_bytes,
            safe_name=safe_name, project_root=Path(project_root),
            form_source=form_source, note=note,
        )
