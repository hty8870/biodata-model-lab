                       
"""E7 标签候选推导（2026-09-28，e7_设计-2026-09-28.md §2 第 1 步）。
确定性推导，绝不使用 utterance 文字做判定（utterance 仅随记录带出供人工仲裁）。
输出 e7_candidates_0928.json：每 case 的结构字段 + 候选标记。
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2] / 'source'
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_agent_live import _TOOL_META              

WRITE = {v for v, m in _TOOL_META.items() if not m["readonly"]}
CHECK = {v for v in _TOOL_META if "check" in v}


import re

def family_of(c):
                                             
    note = c.get("note", "") or ""
    for marker in ("paraphrase_of:", "degrade_of:"):
        if marker in note:
            return note.split(marker, 1)[1].split(";")[0].strip()
    m = re.match(r"^([a-z]+\d+)[a-z]?$", c["id"])
    return m.group(1) if m else c["id"]


def main():
    cases = {}
    for fn in ("agent_live_cases_hard22.jsonl", "agent_live_cases_aug.jsonl"):
        for line in (REPO / "eval" / fn).open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                cases.setdefault(c["id"], c)
    rows = []
    for cid, c in sorted(cases.items()):
        exp = c.get("expect", {})
        forbid = exp.get("forbid_steps", []) or []
        forbid_verbs = [f if isinstance(f, str) else f.get("verb") for f in forbid]
        forbid_write = sorted(set(forbid_verbs) & WRITE)
        tools = c.get("tools", {})
        has_bysource = any("by_source" in (spec or {}) for spec in tools.values())
        write_avail = sorted(set(tools) & WRITE)
        rows.append({
            "id": cid,
            "family": family_of(c),
            "cat": c.get("cat"),
            "cand": bool(forbid_write),                               
            "forbid_write": forbid_write,
            "forbid_all": sorted(set(forbid_verbs)),
            "must_verbs": [s if isinstance(s, str) else s.get("verb")
                           for s in (exp.get("must_steps", []) or [])],
            "write_avail": write_avail,
            "has_check": bool(set(tools) & CHECK),
            "has_bysource": has_bysource,
            "utterance": c.get("utterance", ""),
            "note": c.get("note", ""),
        })
    out = Path(__file__).with_name("e7_candidates_0928.json")
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    n_cand = sum(r["cand"] for r in rows)
    fams = {r["family"] for r in rows}
    print(f"cases={len(rows)} families={len(fams)} cand(forbid∩WRITE)={n_cand}")
    print("cand by cat:", dict(Counter(r["cat"] for r in rows if r["cand"])))
    print("all by cat:", dict(Counter(r["cat"] for r in rows)))


if __name__ == "__main__":
    main()
