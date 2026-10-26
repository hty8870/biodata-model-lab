                       
"""复盘整理（digest）纯逻辑：session log 记录 → 浓缩文本 → prompt；LLM 输出 → 严格 JSON 解析；
内容体检 lint。移植自内部工具「复盘器」`replay/digest.py` + `replay/condense.py`（2026-09-28），
痕迹源从 Kimi Code wire 换成本项目 `session_log` 记录。

红线（与复盘器同源）：
- 本模块零 IO、零 LLM 调用——prompt 文本与解析解耦，全部纯函数（可测）；
- 图谱契约：LLM 给每条目一个局部 id（o1/k2/d1…），motivation 固定 id="start"；
  edges 用局部 id 引用端点，`replay_store.save_digest` 落库时翻译成持久 pid；
- 每条目必须带 turn_refs（回合号 = 记录 seq）供回查痕迹。
"""
from __future__ import annotations

import json

_PROMPT = """你是复盘助手。以下是一位科研人员与「BioData 数据集检索助手」对话的浓缩痕迹
（已脱敏，✗ 表示失败步骤）。请整理出三类条目，并梳理它们之间的因果关系，画成一张有向图：

1. 动机（motivation，至多 1 条）：本段对话的起点——用户的研究目标/检索需求/问题。id 固定为 "start"。
2. 成果（outputs）：本段对话**自身产出**的东西——检索得到的关键数据集与命中数、对比/检查得出的
   结论、产出的工件（任务包/引文/下载/导出）。成果是「这段对话产出的知识」。
3. 知识点（knowledge）：**从外部带入或需要跟进学习**的概念与事实——数据集/物种/技术/平台术语、
   工具用法、外部资料要点。一个结论若由本段对话的检索/对比/讨论得出，归成果而非知识点。
4. 决策点（decisions）：过程中做出的选择、转向、裁决——换检索词、放宽/收紧条件、采纳或放弃
   某条路线。发生了什么、为何这样做、依据是什么。

每条含 id（短字符串：start/o1/o2/k1/d1…，各条目唯一）、title（≤20字）、detail（≤150字）、
turn_refs（相关回合号的整数数组）。

边（edges）：用条目的 id 表达因果链，每条含 from、to、label（≤8字，如 驱动/促成/依赖/产出/
修正/质疑/源于）、turn_refs（可选）。从动机出发、到成果收尾；中间节点多为知识点与决策点。
只画有把握的关系，宁缺毋滥；边总数 ≤ 条目数 ×1.5。

只依据痕迹内容，不得编造；拿不准的不要列入。条目总数控制在 4~14 条。
严格只输出 JSON：
{"motivation":{...},"outputs":[...],"knowledge":[...],"decisions":[...],"edges":[...]}
不要输出任何其他文字。"""

_RETRY_SUFFIX = "\n\n上次输出无法解析。请严格只输出 JSON。"


                                                                               

def condense_record(rec: dict) -> str:
    """单条 session log 记录 → 浓缩文本块（回合号 = 记录 seq）。"""
    seq = rec.get("seq", "?")
    kind = rec.get("kind", "")
    p = rec.get("payload") or {}
    if kind == "search":
        lines = [f"回合 #{seq} 用户（检索）: {str(p.get('query') or '').strip()[:500]}"]
        total = p.get("total", 0)
        status = str(p.get("status") or "")
        tops = [str(t).strip()[:80] for t in (p.get("top") or [])[:5] if str(t).strip()]
        line = f"  检索: 命中 {total} 条（{status}）" if status else f"  检索: 命中 {total} 条"
        if tops:
            line += "；靠前: " + " ｜ ".join(tops)
        lines.append(line)
        return "\n".join(lines)
    if kind == "chat":
        lines = [f"回合 #{seq} 用户: {str(p.get('user') or '').strip()[:500]}"]
        steps = p.get("steps") or []
        for st in steps[:20]:
            mark = "✗" if not st.get("ok", True) else "·"
            lines.append(f"  步骤{mark} {str(st.get('verb_zh') or st.get('verb') or '').strip()[:100]}")
        if len(steps) > 20:
            lines.append(f"  …（另有 {len(steps) - 20} 个步骤略）")
        reply = str(p.get("reply") or "").strip()
        if reply:
            lines.append(f"  助手: {reply[:800]}")
        return "\n".join(lines)
                              
    role = "用户" if p.get("role") == "user" else "助手"
    return f"回合 #{seq} {role}: {str(p.get('text') or '').strip()[:600]}"


def condense_records(records: list[dict], cap: int = 60000) -> str:
    """按 cap 截断：保留前 5 与后 10 个回合，中间省略标注。"""
    blocks = [condense_record(r) for r in records]
    total = sum(len(b) for b in blocks)
    if total <= cap:
        return "\n\n".join(blocks)
    head = blocks[:5]
    tail = blocks[-10:] if len(blocks) > 15 else blocks[5:]
    omitted = len(blocks) - len(head) - len(tail)
    parts = list(head)
    while tail and sum(len(p) for p in parts) + sum(len(p) for p in tail) > cap - 200:
        tail.pop(0)
        omitted += 1
    parts.append(f"……（中间 {omitted} 个回合因长度省略）……")
    parts.extend(tail)
    return "\n\n".join(parts)


def build_prompt(records: list[dict]) -> str:
    return f"{_PROMPT}\n\n【浓缩痕迹】\n{condense_records(records)}"


def retry_prompt(records: list[dict]) -> str:
    return build_prompt(records) + _RETRY_SUFFIX


def seq_range(records: list[dict]) -> tuple[int, int]:
    seqs = [int(r.get("seq", 0)) for r in records]
    return (min(seqs), max(seqs)) if seqs else (0, 0)


                                                                            

def _norm_refs(raw) -> list[int]:
    if not isinstance(raw, list):
        return []
    return [int(r) for r in raw if isinstance(r, (int, float, str)) and str(r).lstrip("-").isdigit()][:12]


def _norm_item(item: dict) -> dict:
    return {
        "id": str(item.get("id", ""))[:16],
        "title": str(item.get("title", ""))[:60],
        "detail": str(item.get("detail", ""))[:600],
        "turn_refs": _norm_refs(item.get("turn_refs")),
    }


def parse_digest(text: str | None) -> dict | None:
    """从 LLM 输出中提取并规范化 digest JSON（容忍 ```json 围栏与前后散文）；失败返回 None。"""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return None
    out = {"outputs": [], "knowledge": [], "decisions": [], "edges": [], "motivation": None}
    for key in ("outputs", "knowledge", "decisions"):
        raw = data.get(key)
        if not isinstance(raw, list):
            continue
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            it = _norm_item(item)
            if not it["id"]:
                it["id"] = f"{key[0]}{i + 1}"
            out[key].append(it)
    mot = data.get("motivation")
    if isinstance(mot, dict) and (mot.get("title") or mot.get("detail")):
        out["motivation"] = _norm_item(mot)
        out["motivation"]["id"] = "start"
    known = {"start"} | {it["id"] for key in ("outputs", "knowledge", "decisions") for it in out[key]}
    raw_edges = data.get("edges")
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            frm, to = str(e.get("from", ""))[:16], str(e.get("to", ""))[:16]
            if not frm or not to or frm == to or frm not in known or to not in known:
                continue
            out["edges"].append({
                "from": frm, "to": to,
                "label": str(e.get("label", ""))[:16],
                "turn_refs": _norm_refs(e.get("turn_refs")),
            })
    if not out["outputs"] and not out["knowledge"] and not out["decisions"] and out["motivation"] is None:
                                                     
                                                    
        return None
    return out


                                                                                

def lint_digest(digest: dict, from_i: int, to_i: int) -> list[str]:
    """digest 内容体检（观察模式，不拦截）：返回人类可读警告列表。纯函数。

    在 save_digest 完成 id→pid 翻译后调用（边端点与条目 pid 一致）。
    积累 LLM 制图错误率的真实数据：turn_refs 越界、孤岛条目、时间逆向边、
    重复标题、有多个条目却无因果边。
    """
    warns: list[str] = []
    items: list[dict] = []
    mot = digest.get("motivation")
    if mot:
        items.append(mot)
    for key in ("outputs", "knowledge", "decisions"):
        items.extend(digest.get(key) or [])

    def iid(it: dict) -> str:
        return it.get("pid") or it.get("id") or "?"

    for it in items:
        bad = [r for r in (it.get("turn_refs") or []) if isinstance(r, int) and not (from_i <= r <= to_i)]
        if bad:
            warns.append(f"条目 {iid(it)} 的回合引用越界（#{bad[0]} 不在 #{from_i}–#{to_i} 内）")

    titles: dict[str, list[str]] = {}
    for it in items:
        t = (it.get("title") or "").strip()
        if t:
            titles.setdefault(t, []).append(iid(it))
    for t, ids in titles.items():
        if len(ids) > 1:
            warns.append(f"标题重复「{t}」（{len(ids)} 处：{', '.join(ids)}）")

    edges = digest.get("edges") or []
    tmap: dict[str, int | None] = {}
    for it in items:
        refs = [r for r in (it.get("turn_refs") or []) if isinstance(r, int)]
        tmap[iid(it)] = min(refs) if refs else None
    deg: dict[str, int] = {}
    n_backward = 0
    for e in edges:
        frm, to = e.get("from"), e.get("to")
        deg[frm] = deg.get(frm, 0) + 1
        deg[to] = deg.get(to, 0) + 1
        tf, tt = tmap.get(frm), tmap.get(to)
        if tf is not None and tt is not None and tf > tt:
            n_backward += 1
    if n_backward:
        warns.append(f"{n_backward} 条边时间逆向（起点回合晚于终点，已按回看边渲染）")
    root_id = iid(mot) if mot else "start"
    orphans = [iid(it) for it in items if iid(it) != root_id and not deg.get(iid(it))]
    if orphans:
        warns.append(f"{len(orphans)} 个条目无因果边（孤岛：{', '.join(orphans[:6])}{'…' if len(orphans) > 6 else ''}）")
    if len(items) >= 4 and not edges:
        warns.append(f"{len(items)} 个条目但没有任何因果边——LLM 只列点没梳理关系")
    return warns[:20]
