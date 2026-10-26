                       
"""复盘编排（2026-09-28 复盘批新增）：记 turn → 后台静默整理 → 角标/标注/快照/sidechat。

设计要点：
- **fail-soft 总闸**：`record_turn` 自身捕获一切异常（webapp 挂钩点另有一层兜底）——
  复盘是旁路观测能力，永远不许掀翻主路径（检索/对话）。
- **后台整理只走进程内 daemon 线程**（默认部署不变；RQ 后端留交接）：每（principal, conv）
  单飞——整理在途时新 turn 只置 dirty，收尾后补跑一次（封顶 4 轮防追不上时死循环）。
- **自动整理的 LLM 资格**（`_draw_worthy`）：mock/无 key 不整理（没有真模型，整理出来也是假的）；
  provider=trial 不自动整理（试用日配额不烧在用户没点头的后台调用上）——skip_reason 落库，
  状态端点如实回报，前端在窗口里给一句人话说明。手动整理（用户显式点）不受 trial 限制。
- LLM 调用走 `llm.llm_client.call_llm`（同步一次性），配置用请求落点物化好的不可变 LLMConfig
  （请求级 key/模型冻结在内，后台线程绝不读 env）。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from ..llm.llm_client import LLMConfig, call_llm
from . import replay_digest, replay_store, session_log

logger = logging.getLogger(__name__)


from contextlib import contextmanager


@contextmanager
def owned_access(project_root, principal, conv_id):
    from .conversation_store import ConversationStore
    from .conversation_runs import RunError
    if principal == "anonymous" or not ConversationStore(project_root).exists(principal, conv_id):
        raise RunError("conversation_not_found", "没有这个共享对话。")
    yield


                                                                                
_DRAW_GUARD = threading.Lock()
_DRAW: dict[tuple[str, str], dict] = {}


def _draw_state(principal: str, conv_id: str) -> dict:
    with _DRAW_GUARD:
        return _DRAW.setdefault((principal, conv_id), {"dirty": False, "running": False})


def drawing(principal: str, conv_id: str) -> bool:
    return bool(_draw_state(principal, conv_id)["running"])


def _draw_worthy(cfg: LLMConfig | None) -> tuple[bool, str]:
    """自动整理的资格判定（手动整理不走这里——trial 也可手动）。"""
    if cfg is None:
        return False, "no_config"
    if cfg.mock_llm:
        return False, "mock"
    if not (cfg.enable_llm and cfg.api_key):
        return False, "no_key"
    if str(cfg.provider or "") == "trial":
        return False, "trial"
    return True, ""


def record_turn(project_root: Path, principal: str, conv_id: str, replay_key: str,
                kind: str, payload: dict, llm_config: LLMConfig | None, *,
                before_llm: Callable[[LLMConfig], None] | None = None) -> None:
    """一轮对话落痕 + 按需调度后台整理。fail-soft：任何异常只进日志。

    写序：先在 store 锁内绑定/校验 key 并落库，再写 session log——「有痕迹无 key」的可认领态
    被结构性排除；无 key 但已有内容/密钥损坏 → 静默拒。
    before_llm 与主请求共用准入；拒绝只阻止整理，不回滚已经落下的痕迹。缺省 None 保持旧调用。"""
    try:
        with replay_store.transaction(project_root, principal, conv_id):
            data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
            if data is None:
                return                                        
            session_log.append_turn(project_root, principal, conv_id, kind, payload)
            worthy, reason = _draw_worthy(llm_config)
            if not worthy:
                data["skip_reason"] = reason
            replay_store.save(project_root, principal, conv_id, data)
        if worthy and session_log.read_records(project_root, principal, conv_id):
            if before_llm is not None:
                before_llm(llm_config)
            schedule_draw(project_root, principal, conv_id, llm_config)
    except Exception:
        logger.exception("复盘 record_turn 失败（不影响主路径）：conv=%s", conv_id)


def schedule_draw(project_root: Path, principal: str, conv_id: str, cfg: LLMConfig) -> None:
    """后台整理单飞调度：在途 → 置 dirty（收尾补跑）；否则起 daemon 线程。"""
    st = _draw_state(principal, conv_id)
    with _DRAW_GUARD:
        if st["running"]:
            st["dirty"] = True
            return
        st["running"] = True
    threading.Thread(
        target=_draw_loop, args=(Path(project_root), principal, conv_id, cfg, st),
        daemon=True, name="replay-digest",
    ).start()


def _draw_rounds(project_root: Path, principal: str, conv_id: str, cfg: LLMConfig,
                 st: dict, *, max_rounds: int = 4) -> None:
    """单飞语义的唯一实现：整理 → 若有待补轮次（dirty）再整理，封顶 max_rounds。
    手动与后台共用它——手动整理期间落痕的新轮次同样要补跑。"""
    for _ in range(max_rounds):
        _draw_once_safe(project_root, principal, conv_id, cfg)
        with _DRAW_GUARD:
            if not st["dirty"]:
                return
            st["dirty"] = False


def _draw_loop(project_root: Path, principal: str, conv_id: str, cfg: LLMConfig, st: dict) -> None:
    try:
        _draw_rounds(project_root, principal, conv_id, cfg, st)
    except Exception:
        logger.exception("复盘后台整理异常：conv=%s", conv_id)
    finally:
        with _DRAW_GUARD:
            st["running"] = False
            st["dirty"] = False


def _draw_once_safe(project_root: Path, principal: str, conv_id: str, cfg: LLMConfig) -> None:
    with replay_store.transaction(project_root, principal, conv_id):
        data, state = replay_store.load_checked(project_root, principal, conv_id)
        key = str(data.get('key') or '')
        if state == 'corrupt' or _authorized(project_root, principal, conv_id, key) is None:
            return
        records = session_log.read_records(project_root, principal, conv_id)
    if not records:
        return
    from_seq, to_seq = replay_digest.seq_range(records)
    text1 = _call_text(replay_digest.build_prompt(records), cfg)
    parsed = replay_digest.parse_digest(text1)
    text2 = None
    if parsed is None:
        text2 = _call_text(replay_digest.retry_prompt(records), cfg)
        parsed = replay_digest.parse_digest(text2)
    if parsed is None:
        parsed = {"outputs": [], "knowledge": [], "decisions": [], "edges": [], "motivation": None,
                  "parse_failed": True, "raw": (text2 or text1 or "")[:2000]}
    with replay_store.transaction(project_root, principal, conv_id):
        if _authorized(project_root, principal, conv_id, key) is None:
            return
        replay_store.save_digest(project_root, principal, conv_id, parsed,
                                 from_seq=from_seq, to_seq=to_seq, model=cfg.model)


def _call_text(prompt: str, cfg: LLMConfig) -> str | None:
    try:
        result = call_llm(prompt, cfg)
    except Exception:
        logger.exception("复盘 LLM 调用抛出")
        return None
    return result.text if result and result.succeeded and result.text else None


def manual_digest(project_root: Path, principal: str, conv_id: str, replay_key: str,
                  cfg: LLMConfig | None) -> dict | None:
    """用户显式点的「重新整理」：同步执行（前端 loading 态等待）。None = key 不符/遗留无 key 态。"""
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        replay_store.save(project_root, principal, conv_id, data)
    if cfg is None or cfg.mock_llm:
        return {"ok": False, "skipped": "mock" if cfg and cfg.mock_llm else "no_config"}
    if not (cfg.enable_llm and cfg.api_key):
        return {"ok": False, "skipped": "no_key"}
    if drawing(principal, conv_id):
        import time
        deadline = time.monotonic() + 60
        while drawing(principal, conv_id) and time.monotonic() < deadline:
            threading.Event().wait(.02)
        if drawing(principal, conv_id):
            return {"ok": True, "drawing": True}
    st = _draw_state(principal, conv_id)
    with _DRAW_GUARD:
        st["running"] = True
    try:
        _draw_rounds(project_root, principal, conv_id, cfg, st)                       
    finally:
        with _DRAW_GUARD:
            st["running"] = False
            st["dirty"] = False
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key)
        return {"ok": True, "has_digest": bool(data.get("digest"))} if data is not None else None


                                                                                     

def _authorized(project_root: Path, principal: str, conv_id: str, replay_key: str,
                *, bind: bool = False) -> dict | None:
    """统一的 key 校验与首写绑定（在 store 锁内完成，读写同径）：

    - 已落 key：必须匹配（不匹配 → None，端点一律 404，不区分「不存在」与「key 错」）；
    - 未落 key + 无任何内容（全新会话）：读放行（保留空态读取）；`bind=True`（写路径）时
      在**同一事务**里把非空 supplied key 落库——首写绑定；
    - 未落 key 但已有内容（旧版本遗留）：拒绝——不把「尚无 key」当作已有内容的授权；
    - 密钥 JSON 损坏（文件在、解析失败）：一律拒绝——不把损坏态当新会话；
    - 空的 supplied key 不能用于写入（绑定必须是非空 key）。
    """
    from .conversation_store import ConversationStore
    conversations = ConversationStore(project_root)
    supplied = str(replay_key or "")[:64]
    with replay_store.transaction(project_root, principal, conv_id):
        shared = conversations.exists(principal, conv_id, include_deleted=True)
        if shared and (principal == "anonymous" or not conversations.exists(principal, conv_id)):
            return None
        data, state = replay_store.load_checked(project_root, principal, conv_id)
        if shared:
            return data if state != "corrupt" else None
        stored = str(data.get("key") or "")
        if stored:
            return data if stored == supplied else None
        if state == "corrupt":
            return None
        if replay_store.has_content(data) or session_log.read_records(project_root, principal, conv_id):
            return None                                                      
        if bind:
            if not supplied:
                return None
            data["key"] = supplied
            replay_store.save(project_root, principal, conv_id, data)
        return data


def authorized_owned(project_root, principal, conv_id):
    with owned_access(project_root, principal, conv_id):
        return _authorized(project_root, principal, conv_id, '') is not None


def status(project_root: Path, principal: str, conv_id: str, replay_key: str) -> dict | None:
    data = _authorized(project_root, principal, conv_id, replay_key)
    if data is None:
        return None
    un = replay_store.unseen(data)
    dg = data.get("digest") or {}
    return {
        "ok": True,
        "n_turns": len(session_log.read_records(project_root, principal, conv_id)),
        "has_digest": bool(data.get("digest")),
        "parse_failed": bool(dg.get("parse_failed")),
        "unseen": len(un["worth"]),
        "unseen_knowledge": len(un["knowledge"]),
        "drawing": drawing(principal, conv_id),
        "skip_reason": data.get("skip_reason") or "",
        "lint_count": len(dg.get("lint") or []),
        "digest_version": int(data.get("digest_version") or 0),
    }


def points(project_root: Path, principal: str, conv_id: str, replay_key: str) -> dict | None:
    data = _authorized(project_root, principal, conv_id, replay_key)
    if data is None:
        return None
    un = replay_store.unseen(data)
    return {
        "ok": True,
        "digest": data.get("digest"),
        "digest_version": int(data.get("digest_version") or 0),
        "pids": replay_store.all_pids(data),
        "threads": data.get("threads") or {},
        "snapshots": (data.get("snapshots") or [])[-30:],
        "unseen": un["worth"],
        "unseen_knowledge": un["knowledge"],
        "n_turns": len(session_log.read_records(project_root, principal, conv_id)),
        "drawing": drawing(principal, conv_id),
        "skip_reason": data.get("skip_reason") or "",
    }


                                                                                   

def annotate(project_root: Path, principal: str, conv_id: str, replay_key: str,
             pid: str, text: str) -> bool | None:
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        point, _section = replay_store.find_point(data, pid)
        if point is None:
            return False
        replay_store.append_thread(project_root, principal, conv_id, pid, "user", text)
    return True


def take_snapshot(project_root: Path, principal: str, conv_id: str, replay_key: str) -> dict | None:
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        return replay_store.take_snapshot(project_root, principal, conv_id)


def annotate_snapshot(project_root: Path, principal: str, conv_id: str, replay_key: str,
                      snap_id: str, pid: str, text: str) -> dict | None | False:
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        snap = replay_store.annotate_snapshot(project_root, principal, conv_id, snap_id, pid, text)
    return snap if snap is not None else False


def mark_seen(project_root: Path, principal: str, conv_id: str, replay_key: str, *,
              observed_version: "int | None" = None,
              observed_pids: "list[str] | None" = None) -> dict | None:
    """核销未读；核销集绑定用户实际看到的那一版（语义见 replay_store.mark_seen）。
    None = key 不符/遗留无 key 态。"""
    with replay_store.transaction(project_root, principal, conv_id):
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        return replay_store.mark_seen(project_root, principal, conv_id,
                                      observed_version=observed_version,
                                      observed_pids=observed_pids)


                                                                                 

_CHAT_RULES = """你是复盘助手，帮助用户复盘 TA 与「BioData 数据集检索助手」的一段对话。
依据：【决策因果图】（模型整理，可能遗漏）、【浓缩对话痕迹】（机械记录的事实）、【用户标注】。
规则：只依据所给材料，不得编造；引用回合号（#N）指明出处；回答用简洁中文；拿不准明说拿不准。"""

                                                     
_CHAT_INPUT_BUDGET = 32000
                            
_CHAT_ANNOTATION_MAX = 6000


def _truncate_with_note(text: str, limit: int, what: str) -> str:
    """超预算时保留**最新内容（尾部）** + 头部省略说明；返回长度 ≤ max(limit, 说明长度)。

    不静默丢用户标注：磁盘上的完整标注一字不动，只在注入 prompt 的这一份上截断并说明。"""
    if len(text) <= limit:
        return text
    omitted = len(text)
    for _ in range(3):                                             
        note = f"（{what}超过输入预算，已省略较早的 {omitted} 字；完整内容仍保存在复盘窗口）\n"
        keep = limit - len(note)
        if keep <= 0:
            return note[:max(limit, 0)] or note
        omitted = len(text) - keep
    note = f"（{what}超过输入预算，已省略较早的 {omitted} 字；完整内容仍保存在复盘窗口）\n"
    keep = max(0, limit - len(note))
    return (note + text[-keep:]) if keep else note


def chat(project_root: Path, principal: str, conv_id: str, replay_key: str,
         text: str, cfg: LLMConfig | None) -> dict | None:
    """sidechat 一轮：全上下文（digest + 浓缩痕迹 + 标注 + 复盘史）→ LLM → 落 threads["session"]。

    首写先绑定非空 key；完整 prompt 受输入预算约束：标注按剩余预算截断并
    如实披露，磁盘标注不动。"""
    data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
    if data is None:
        return None
    if cfg is None or cfg.mock_llm or not (cfg.enable_llm and cfg.api_key):
        return {"ok": False, "reply": "（复盘对话需要可用的大模型：请先配置 API Key，或使用网站试用额度。）"}
    records = session_log.read_records(project_root, principal, conv_id)
    digest_json = json.dumps(data.get("digest") or {}, ensure_ascii=False)[:4000]
    condensed = replay_digest.condense_records(records, cap=16000) if records else "（尚无痕迹）"
    annotations_raw = replay_store.compose_annotations(data).get("text") or "（无）"
    history = (data.get("threads") or {}).get("session") or []
    hist_lines = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {str(m.get('content') or '')[:600]}"
        for m in history[-8:]
    )
    head = f"{_CHAT_RULES}\n\n【决策因果图】\n{digest_json}\n\n【浓缩对话痕迹】\n{condensed}"
    tail = f"\n\n【复盘对话】\n{hist_lines or '（暂无）'}\n\n用户: {text}"
    wrapper = "\n\n【用户标注】\n"
    ann_limit = min(_CHAT_ANNOTATION_MAX,
                    max(0, _CHAT_INPUT_BUDGET - len(head) - len(tail) - len(wrapper)))
    annotations = _truncate_with_note(annotations_raw, ann_limit, "用户标注")
    prompt = head + wrapper + annotations + tail
    reply = _call_text(prompt, cfg)
    if reply is None:
        return {"ok": False, "reply": "（这次没有接上大模型，请稍后再试。）"}
    with replay_store.transaction(project_root, principal, conv_id):
        if _authorized(project_root, principal, conv_id, replay_key, bind=True) is None:
            return None
        replay_store.append_thread(project_root, principal, conv_id, "session", "user", text)
        replay_store.append_thread(project_root, principal, conv_id, "session", "assistant", reply)
    return {"ok": True, "reply": reply}


                                                                             

def _planned_turns(turns: list[dict]) -> "list[tuple[dict, str]]":
    """按落痕口径过滤（前 60 条、单条截 2000、去掉空白文本），一次算好用于计数与写入。"""
    out: list[tuple[dict, str]] = []
    for t in turns[:60]:
        if not isinstance(t, dict):
            continue
        text = str(t.get("text") or "")[:2000]
        if text.strip():
            out.append((t, text))
    return out


def backfill(project_root: Path, principal: str, conv_id: str, replay_key: str,
             turns: list[dict], cfg: LLMConfig | None, *,
             quota_gate: "Callable[[], None] | None" = None) -> dict | None:
    """把本机历史（localStorage 对话）补建成 session log 记录——只在服务端尚无记录时生效
    （fail-closed 防重复）；用户显式点按钮才上传（对齐 /api/dream 的主动发送先例）。

    `quota_gate`：确认本次**确实会整理**（worthy 且有轮次）后、任何写入之前调用；配额不足时
    由它抛出（Web 层抛 HTTPException → 429），本次零落痕、零 key 绑定；过闸后在同事务
    持久化 key 再追加日志，末次状态保存失败也不丢失已落痕的访问凭据。"""
    planned = _planned_turns(turns)
    supplied = str(replay_key or "")[:64]
    with replay_store.transaction(project_root, principal, conv_id):
                                                
                                                                         
                                        
        data = _authorized(project_root, principal, conv_id, replay_key, bind=False)
        if data is None:
            return None
        if not supplied:
            return None                                                     
        if session_log.read_records(project_root, principal, conv_id):
            return {"ok": True, "skipped": "has_records"}
        worthy, reason = _draw_worthy(cfg)
        if worthy and planned and quota_gate is not None:
            quota_gate()                                             
        data = _authorized(project_root, principal, conv_id, replay_key, bind=True)
        if data is None:
            return None
        if not worthy:
            data["skip_reason"] = reason
        n = 0
        for t, text in planned:
            role = "user" if t.get("role") == "user" else "assistant"
            session_log.append_turn(project_root, principal, conv_id, "backfill", {"role": role, "text": text})
            n += 1
        replay_store.save(project_root, principal, conv_id, data)
    if worthy and n:
        schedule_draw(project_root, principal, conv_id, cfg)
    return {"ok": True, "n": n}


def _reset_for_tests() -> None:
    with _DRAW_GUARD:
        _DRAW.clear()
