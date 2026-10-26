                       
"""对话级 session log（2026-09-28 复盘批新增）：按（principal, 对话）分区的
append-only JSONL 痕迹——复盘（`replay_*`）的第一个消费者，也是后续用户模型/评测的数据面。

项目痕迹谱系（本模块是第三类，不替代前两类）：
- `database/trace/<principal>/<turn_id>.jsonl`：agent 单轮内部 trace（调试向，turn 级）；
- `.userdata/mcp_calls.jsonl`：MCP 调用留痕（接口级）；
- 本模块：**对话级事实流**（用户说了什么、系统路由/检索/回复了什么），conversation 级，
  分区键 = 登录 principal（无会话 → "anonymous"）+ 前端生成的对话 id。

纪律（与 trace recorder / OOV 账本同族）：
- **脱敏第一道工序**：整行落盘前过 `secret_patterns.redact`；
- append-only：只追加不改写；读时坏行跳过不计数；跨进程路径锁内重读持久最大 seq；
- fail-soft：写失败打 stderr 即弃，绝不掀翻主路径；
- `BIODATA_SESSION_LOG=off` 关停（缺省开）；运行时状态经 `runtime_paths`
  （source/portable = project_root/.userdata，frozen = data_root/.userdata；绝不落 database/）。

记录信封：`{"seq": int, "ts": epoch秒, "kind": str, "payload": {...}}`。
kind 当前三档：`search`（一次 /api/recommend 检索）/ `chat`（一轮 /api/utterance 对话）/
`backfill`（用户显式补建的本机历史）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..secret_patterns import redact
from .runtime_paths import instance_data_dir_for

_SWITCH_ENV = "BIODATA_SESSION_LOG"


def enabled() -> bool:
    """缺省开；显式 off/0/false/no 关停。"""
    return os.environ.get(_SWITCH_ENV, "").strip().lower() not in ("off", "0", "false", "no")


def _safe(part: str) -> str:
    """路径段净化：只留字母数字与 -_，截 80——principal/conv_id 都是调用方可控值。"""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(part))[:80] or "_"


def log_path(project_root: Path, principal: str, conv_id: str) -> Path:
    root = instance_data_dir_for(Path(project_root), ".userdata") / "sessionlog" / _safe(principal)
    return root / f"{_safe(conv_id)}.jsonl"


                                                                               
                                                       
                                 

def _read_lines(path: Path) -> "tuple[list[str], bool]":
    """按字节切行：返回（可解码的完整行文本, 尾部是否残缺）。
    未换行尾部暂不确认；追加先封行，完整有效 JSON 随后参与 seq 推导，坏尾仍原样保留。"""
    raw = path.read_bytes()
    if not raw:
        return [], False
    truncated = not raw.endswith(b"\n")
    chunks = raw.split(b"\n")[:-1]                                   
    lines: list[str] = []
    for chunk in chunks:
        try:
            lines.append(chunk.decode("utf-8"))
        except UnicodeDecodeError:
            continue                                             
    return lines, truncated


def _records_from_lines(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("payload"), dict):
            out.append(rec)
    return out


def _max_seq(path: Path) -> int:
    """已落数据的最大 seq（真源）；无有效记录 → 0。"""
    if not path.exists():
        return 0
    lines, _ = _read_lines(path)
    seqs = [r["seq"] for r in _records_from_lines(lines)
            if isinstance(r.get("seq"), int) and not isinstance(r["seq"], bool)]
    return max(seqs, default=0)


def _needs_leading_newline(path: Path) -> bool:
    """文件非空且最后一个字节不是换行：下一条记录前先补 \\n，避免被粘进未封行的坏尾。"""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return False
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except FileNotFoundError:
        return False


def append_turn(project_root: Path, principal: str, conv_id: str, kind: str, payload: dict) -> None:
    """追加一条对话记录（fail-soft：任何异常只打 stderr，绝不抛出）。

    跨进程锁内先封尾（完整 JSON 可能只缺换行），再从持久最大 seq 续编；
    不信任进程缓存，未落有效记录不耗号。
    写入只持日志锁，不获取复盘锁，兼容复盘 → 日志的锁顺序。"""
    if not conv_id or not enabled():
        return
    path = log_path(project_root, principal, conv_id)
    try:
        from ..corpus.fs_utils import file_critical_section
        with file_critical_section(path.with_suffix('.jsonl.lock')):
            if _needs_leading_newline(path):
                with path.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
            seq = _max_seq(path) + 1
            line = json.dumps(
                {"seq": seq, "ts": time.time(), "kind": str(kind)[:24], "payload": payload},
                ensure_ascii=False,
            )
                                                
            line = redact(line) or line
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
                                                         
        print(f"session log 写入失败（{type(exc).__name__}: {exc}），本次跳过，不影响主路径。", file=sys.stderr)


def read_records(project_root: Path, principal: str, conv_id: str) -> list[dict]:
    """读出全部**完整已确认**记录（坏行/断尾跳过；文件不存在/读取失败 → 空列表）。
    断尾只影响未写完整的最后一段，既有历史照常可读；坏文件不删。对话规模下单次全读可承受。"""
    path = log_path(project_root, principal, conv_id)
    if not path.exists():
        return []
    try:
        lines, _ = _read_lines(path)
    except Exception:
        return []
    return _records_from_lines(lines)


def _reset_for_tests() -> None:
    """测试专用：清空进程内路径锁（配合 tmp project_root 使用）。"""
    from ..corpus.fs_utils import reset_path_locks_for_test

    reset_path_locks_for_test()
