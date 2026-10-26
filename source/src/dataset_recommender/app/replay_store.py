                       
"""复盘产物持久化（2026-09-28 复盘批新增）：`.userdata/replay/<principal>/<conv>.json`。

移植自内部工具「复盘器」`replay/points_store.py`（2026-09-28），适配：
- 存储根走 `runtime_paths`（instance_data_dir_for）；写盘唯一通道 `atomic_write_json`；按路径 RLock；
- 每个（principal, conv）一份文件；首写落 `key`（前端生成的随机 replay_key），之后读写必须匹配——
  anonymous 分区下 conv_id 熵不高，key 是防串号的第二道闸；
- 不做 .bak 归档（session log 本身就是历史，digest 可随时重建）；
- 无 adhoc 痕迹条目（biodata 侧 MVP 不移植该通道）。

结构：{conv, key, digest, threads, snapshots, last_seq, seen_pids, skip_reason, created_at, updated_at}
- digest：一次整理的条目与范围（motivation/outputs/knowledge/decisions/edges，
  条目带 pid，边带 eid/from/to/label；边端点是 pid，"start"=动机节点；lint 体检警告随 digest 落库）
- threads：{pid|eid|"session": [{role, content, ts}]}——图上的用户标注 + sidechat 复盘对话（同一通道）
- snapshots：「添加到对话」打包的标注快照 [{snap_id, created_at, count, items, text}]（截 30 份）
- seen_pids：用户已在窗口里看过的条目 pid——角标 unseen = digest 条目 − seen_pids
- skip_reason：自动整理被跳过的原因（mock/no_key/trial/parse_failed/…），"" = 无；供状态端点如实回报
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from .replay_digest import lint_digest
from .runtime_paths import atomic_write_json, instance_data_dir_for

from ..corpus.fs_utils import file_critical_section


def _lock_for(path):
    return file_critical_section(Path(str(path) + '.lock'), timeout=15)
_SNAP_SEQ = itertools.count()


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(part))[:80] or "_"


def store_path(project_root: Path, principal: str, conv_id: str) -> Path:
    root = instance_data_dir_for(Path(project_root), ".userdata") / "replay" / _safe(principal)
    return root / f"{_safe(conv_id)}.json"


def _empty(conv_id: str) -> dict:
    return {"conv": conv_id, "key": "", "digest": None, "threads": {}, "snapshots": [],
            "last_seq": None, "seen_pids": [], "skip_reason": "", "point_history": {},
            "digest_version": 0, "created_at": time.time(), "updated_at": time.time()}


def has_content(data: dict) -> bool:
    """是否已有任何用户内容（用于「无 key 但有内容 → fail-closed」判定）。
    skip_reason/seen_pids/last_seq 只是元数据，不算内容。"""
    return bool(data.get("digest") or data.get("threads") or data.get("snapshots")
                or data.get("point_history"))


def load(project_root: Path, principal: str, conv_id: str) -> dict:
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        return _load_unlocked(p, conv_id)


def load_checked(project_root: Path, principal: str, conv_id: str) -> "tuple[dict, str]":
    """带状态的读取：state ∈ {"fresh"（无文件）/ "ok" / "corrupt"（文件存在但解析失败）}。
    损坏态必须与「全新空态」区分：前者不得被当作新会话放行。"""
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        return _load_with_state(p, conv_id)


def transaction(project_root: Path, principal: str, conv_id: str):
    """多步读改写复用同一份路径 RLock；调用方应将 LLM/后台调度留在锁外。"""
    return _lock_for(store_path(project_root, principal, conv_id))


def _load_with_state(p: Path, conv_id: str) -> "tuple[dict, str]":
    if not p.exists():
        return _empty(conv_id), "fresh"
    try:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            base = _empty(conv_id)
            base.update({k: data.get(k, v) for k, v in base.items()})
            return base, "ok"
    except Exception:
        pass
    return _empty(conv_id), "corrupt"


def _load_unlocked(p: Path, conv_id: str) -> dict:
    return _load_with_state(p, conv_id)[0]


def save(project_root: Path, principal: str, conv_id: str, data: dict) -> None:
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data["updated_at"] = time.time()
        atomic_write_json(p, data)


def ensure_key(data: dict, replay_key: str) -> bool:
    """首写落 key；之后必须匹配。返回 False = key 不符（调用方静默拒，不泄漏存在性）。"""
    stored = str(data.get("key") or "")
    if not stored:
        data["key"] = str(replay_key or "")[:64]
        return True
    return stored == str(replay_key or "")[:64]


                                                                            

_POINT_PREFIX = {"outputs": "o", "knowledge": "k", "decisions": "d", "motivation": "m", "edge": "e"}


def _point_identity(kind: str, item: dict) -> tuple:
    """模型局部 id/数组位置不参与身份；内容变化视作新条目，旧标注保留原文。"""
    fields = ("from", "to", "label") if kind == "edge" else ("title", "detail")
    return tuple(item.get(k, "") for k in fields) + (tuple(sorted(item.get("turn_refs") or [])),)


def _stable_point_id(history: dict, used: set, kind: str, item: dict) -> str:
    identity = _point_identity(kind, item)
    pid = next((pid for pid, old in history.items()
                if pid not in used and old["kind"] == kind
                and _point_identity(kind, old["point"]) == identity), None)
    if pid is None:
        prefix = _POINT_PREFIX[kind]
        index = 0
        while f"{prefix}-{index}" in history:
            index += 1
        pid = "start" if kind == "motivation" and "start" not in history else f"{prefix}-{index}"
    used.add(pid)
    history[pid] = {"kind": kind, "point": {**item, ("eid" if kind == "edge" else "pid"): pid}}
    return pid


def save_digest(project_root: Path, principal: str, conv_id: str, digest: dict, *,
                from_seq: int, to_seq: int, model: str | None) -> dict:
    """LLM 局部 id → 持久 pid 翻译 + lint 体检后落库（边端点缺失/自环丢弃）。"""
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data = _load_unlocked(p, conv_id)
        current_to = max(int(data.get('last_seq') or 0), int((data.get('digest') or {}).get('to') or 0))
        if int(to_seq) < current_to:
            return data
        history = data["point_history"]
        previous = data.get("digest") or {}
                                       
        for kind, items in [(k, previous.get(k) or []) for k in ("outputs", "knowledge", "decisions")] + [
                ("motivation", [previous["motivation"]] if previous.get("motivation") else []),
                ("edge", previous.get("edges") or [])]:
            for item in items:
                pid = item.get("eid" if kind == "edge" else "pid")
                if pid:
                    history.setdefault(pid, {"kind": kind, "point": dict(item)})
        used: set[str] = set()
        id_map: dict[str, str] = {}
        for i, item in enumerate(digest.get("outputs", [])):
            item["pid"] = _stable_point_id(history, used, "outputs", item)
            if item.get("id"):
                id_map[item["id"]] = item["pid"]
        for i, item in enumerate(digest.get("knowledge", [])):
            item["pid"] = _stable_point_id(history, used, "knowledge", item)
            if item.get("id"):
                id_map[item["id"]] = item["pid"]
        for i, item in enumerate(digest.get("decisions", [])):
            item["pid"] = _stable_point_id(history, used, "decisions", item)
            if item.get("id"):
                id_map[item["id"]] = item["pid"]
        motivation = digest.get("motivation")
        if motivation:
            motivation = dict(motivation)
            motivation["pid"] = _stable_point_id(history, used, "motivation", motivation)
            id_map["start"] = motivation["pid"]
            if motivation.get("id") and motivation["id"] != "start":
                id_map[motivation["id"]] = "start"
        edges: list[dict] = []
        for e in digest.get("edges", []):
            frm, to = id_map.get(e.get("from", "")), id_map.get(e.get("to", ""))
            if not frm or not to or frm == to:
                continue
            edge = {"from": frm, "to": to, "label": e.get("label", ""),
                    "turn_refs": e.get("turn_refs", [])[:12]}
            edge["eid"] = _stable_point_id(history, used, "edge", edge)
            edges.append(edge)
        data["digest"] = {
            "from": from_seq, "to": to_seq,
            "created_at": time.time(),
            "model": model or digest.get("model") or "",
            "motivation": motivation,
            "outputs": digest.get("outputs", []),
            "knowledge": digest.get("knowledge", []),
            "decisions": digest.get("decisions", []),
            "edges": edges,
            "parse_failed": digest.get("parse_failed", False),
            "raw": digest.get("raw", ""),
        }
                                                     
        data["digest"]["lint"] = lint_digest(data["digest"], from_seq, to_seq)
                                                      
                                    
        data["digest_version"] = int(data.get("digest_version") or 0) + 1
        data["last_seq"] = to_seq
        data["skip_reason"] = ""
        data["updated_at"] = time.time()
        atomic_write_json(p, data)
        return data


                                                                              

def find_point(data: dict, pid: str) -> tuple[dict | None, str]:
    dg = data.get("digest") or {}
    mot = dg.get("motivation")
    if mot and pid == mot.get("pid", "start"):
        return mot, "motivation"
    for item in dg.get("outputs", []):
        if item.get("pid") == pid:
            return item, "outputs"
    for item in dg.get("knowledge", []):
        if item.get("pid") == pid:
            return item, "knowledge"
    for item in dg.get("decisions", []):
        if item.get("pid") == pid:
            return item, "decisions"
    for item in dg.get("edges", []):
        if item.get("eid") == pid:
            return item, "edge"
    archived = (data.get("point_history") or {}).get(pid)
    if archived:
        return archived["point"], archived["kind"]
    return None, ""


def resolve_edge(data: dict, eid: str) -> tuple[dict | None, dict | None, dict | None]:
    """e-x → (edge, from_point, to_point)；端点缺失时对应位置为 None。"""
    edge, section = find_point(data, eid)
    if section != "edge":
        return None, None, None
    frm, _ = find_point(data, edge.get("from", ""))
    to, _ = find_point(data, edge.get("to", ""))
    return edge, frm, to


_KIND_LABEL = {"outputs": "成果", "knowledge": "知识点", "decisions": "决策点",
               "motivation": "动机", "edge": "关系"}


def _head_of(data: dict, pid: str) -> str:
    point, section = find_point(data, pid)
    if section == "edge":
        edge, frm, to = resolve_edge(data, pid)
        head = f"关系「{(frm or {}).get('title', '?')} —[{(edge or {}).get('label', '')}]→ {(to or {}).get('title', '?')}」"
    elif point:
        head = f"{_KIND_LABEL.get(section, section)}「{point.get('title', '')}」"
    else:
        head = f"条目 {pid}"
    return head


def _compose_items(items: list[dict]) -> dict:
    text = "\n\n".join(
        f"【{it['head']}】\n" + "\n".join(f"- {n}" for n in it["notes"]) for it in items
    )
    return {"count": sum(len(it["notes"]) for it in items), "items": items, "text": text}


def compose_annotations(data: dict) -> dict:
    """把全部线程里的用户标注与其对应条目拼接成可注入对话的文本（单源，供「添加到对话」）。
    返回 {count, items, text}；items 每项 {pid, head, kind, point, notes[]}——kind/point 让快照
    自包含条目内容（重整理后旧标注对象仍有节点可渲染/可继续编辑）。"""
    items: list[dict] = []
    for pid, msgs in (data.get("threads") or {}).items():
        notes = [m.get("content", "") for m in msgs if m.get("role") == "user" and m.get("content")]
        if not notes:
            continue
        if pid == "session":
            continue                       
        point, kind = find_point(data, pid)
        items.append({"pid": pid, "head": _head_of(data, pid),
                      "kind": kind, "point": point, "notes": notes})
    return _compose_items(items)


def _edges_within(data: dict, pids: set) -> list[dict]:
    """捕获时的关系（两端都在快照内）——快照自带图数据，不依赖当前 digest 布局。"""
    return [dict(e) for e in ((data.get("digest") or {}).get("edges") or [])
            if e.get("from") in pids and e.get("to") in pids]


def _snapshot_edges(data: dict, items: list[dict]) -> "tuple[list[dict], list[dict]]":
    """快照的关系闭包：返回（edges, context_items）。

    - edges ＝ 两端都被标注的关系（既有语义）∪ **被标注的关系本身**：只标一条边的用户，
      找回的快照图里同样要能看到这条关系（旧实现只按端点过滤，edges 恒空 → 前端回退当前图）；
    - context_items ＝ 被标关系的两端 point/kind，notes 空：只补画布需要的节点闭包，
      **不是标注**——不进 count、不改旧 note 条数；用户续标这些节点时按普通条目追加。
    端点定位走 find_point（digest 找不到时回退 point_history），故重整理删图后旧关系仍可画。"""
    annotated = {it["pid"] for it in items}
    edges: list[dict] = _edges_within(data, annotated)
    seen = {(e.get("eid", ""), e.get("from", ""), e.get("to", "")) for e in edges}
    context: list[dict] = []
    for it in items:
        if it.get("kind") != "edge" or not it.get("point"):
            continue
        edge = dict(it["point"])
        key = (edge.get("eid", ""), edge.get("from", ""), edge.get("to", ""))
        if key not in seen:
            seen.add(key)
            edges.append(edge)
        for pid in (edge.get("from"), edge.get("to")):
            if not pid or pid in annotated:
                continue
            point, kind = find_point(data, pid)
            if not point:
                continue
            annotated.add(pid)
            context.append({"pid": pid, "head": _head_of(data, pid), "kind": kind,
                            "point": point, "notes": []})
    return edges, context


                                                                              

def take_snapshot(project_root: Path, principal: str, conv_id: str) -> dict:
    """打包当前全部用户标注为快照并清空图上标注（「添加到对话」的唯一通道）。
    无标注时返回 {"count": 0, ...}，不落盘。快照自带 items[].kind/point 与 edges，
    只标关系时连带两端 point/kind 与关系本身（见 _snapshot_edges）。"""
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data = _load_unlocked(p, conv_id)
        composed = compose_annotations(data)
        if not composed["count"]:
            return {"count": 0, "items": [], "text": ""}
        edges, context = _snapshot_edges(data, composed["items"])
        snap = {"snap_id": f"s-{int(time.time() * 1000) % 10**9}-{next(_SNAP_SEQ)}",
                "created_at": time.time(), "count": composed["count"],
                "items": composed["items"] + context, "text": composed["text"],
                "edges": edges}
        data["snapshots"].append(snap)
        data["snapshots"] = data["snapshots"][-30:]
        for it in composed["items"]:
            data["threads"].pop(it["pid"], None)
        data["updated_at"] = time.time()
        atomic_write_json(p, data)
        return snap


def annotate_snapshot(project_root: Path, principal: str, conv_id: str,
                      snap_id: str, pid: str, text: str) -> dict | None:
    """往快照里追加一条标注（pill 找回后的继续编辑）；快照不存在返回 None。"""
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data = _load_unlocked(p, conv_id)
        for s in data.get("snapshots", []):
            if s.get("snap_id") != snap_id:
                continue
            for it in s["items"]:
                if it["pid"] == pid:
                    it["notes"].append(text)
                    break
            else:
                point, kind = find_point(data, pid)
                s["items"].append({"pid": pid, "head": _head_of(data, pid),
                                   "kind": kind, "point": point, "notes": [text]})
            s.update(_compose_items(s["items"]))
            data["updated_at"] = time.time()
            atomic_write_json(p, data)
            return s
    return None


def append_thread(project_root: Path, principal: str, conv_id: str,
                  pid: str, role: str, content: str) -> None:
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data = _load_unlocked(p, conv_id)
        data["threads"].setdefault(pid, []).append(
            {"role": role, "content": content, "ts": time.time()}
        )
        data["updated_at"] = time.time()
        atomic_write_json(p, data)


                                                                                    

def all_pids(data: dict) -> list[str]:
    dg = data.get("digest") or {}
    out: list[str] = []
    for key in ("outputs", "knowledge", "decisions"):
        out.extend(it.get("pid", "") for it in (dg.get(key) or []) if it.get("pid"))
    return out


def unseen(data: dict) -> dict:
    """未读条目：worth = 成果+决策点（「值得看一眼」），knowledge = 知识点（窗口说明行用）。"""
    seen = set(data.get("seen_pids") or [])
    dg = data.get("digest") or {}
    worth = [it.get("pid") for key in ("outputs", "decisions") for it in (dg.get(key) or [])
             if it.get("pid") and it["pid"] not in seen]
    know = [it.get("pid") for it in (dg.get("knowledge") or [])
            if it.get("pid") and it["pid"] not in seen]
    return {"worth": worth, "knowledge": know}


def mark_seen(project_root: Path, principal: str, conv_id: str, *,
              observed_version: "int | None" = None,
              observed_pids: "list[str] | None" = None) -> dict:
    """核销未读。核销集必须绑定用户**实际看到的那一版**：

    - `observed_pids`：调用方直接给出展示过的 pid 集（精确回显），只核销其中当前仍存在的；
    - `observed_version` 等于当前 `digest_version`：用户看的就是当前版本 → 全量核销；
    - `observed_version` 落后/超前：不猜、不核销（`stale=true`），返回当前版本供刷新；
    - 两者都未给（旧调用）：维持历史语义＝核销当前全部。
    seen 集合只增不减：旧版本调用永远不会「覆盖掉」更新版本产生的未读，也不会吞掉新成果。"""
    p = store_path(project_root, principal, conv_id)
    with _lock_for(p):
        data = _load_unlocked(p, conv_id)
        current = int(data.get("digest_version") or 0)
        current_pids = all_pids(data)
        stale = False
        if observed_pids is not None:
            wanted = [str(x) for x in observed_pids]
            keep = set(current_pids)
            marked = [pid for pid in dict.fromkeys(wanted) if pid in keep]
        elif observed_version is None:
            marked = list(current_pids)
        elif int(observed_version) == current:
            marked = list(current_pids)
        else:
            marked = []
            stale = True
        seen = list(dict.fromkeys(list(data.get("seen_pids") or []) + marked))
        data["seen_pids"] = seen
        data["updated_at"] = time.time()
        atomic_write_json(p, data)
        return {"ok": True, "marked": len(marked), "digest_version": current, "stale": stale}


def _reset_for_tests() -> None:
    from ..corpus.fs_utils import reset_path_locks_for_test

    reset_path_locks_for_test()
