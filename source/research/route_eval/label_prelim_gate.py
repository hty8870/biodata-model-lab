                       
"""初步检索执行闸金标派生（确定性，零 LLM，2026-09-19）。

闸的目标：turn 入口决定 _RagFlight（auto 全量检索预取线程）是否起跑。
标签语义：
  retrieve = 需要预取检索 —— route==search，或混合诉求（生产 `_hybrid_intent_gate`
             命中，检索半需要结果），或 consensus 平票/废票机械兜底 general
             （拿不准的句子保守视为需要检索——闸的失误方向是不起跑，保守方向 = 多起跑）。
  skip     = 不需要 —— 其余（action、非混合且非兜底的 general）。

输入  research/route_eval/route_labels_v2.jsonl（LLM 三分类共识标签）
输出  research/route_eval/prelim_labels.jsonl（{qid, utterance, route_llm,
      decided_by, hybrid, gate, route_llm_for_train}——train_l0_router.py 读 route_llm
      字段，故另写 route_llm=gate 的扁平副本 prelim_labels_train.jsonl）

Usage:
  .venv/Scripts/python.exe research/route_eval/label_prelim_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent.agent_exec import _hybrid_intent_gate              

LABELS = AGENT_ROOT / "research" / "route_eval" / "route_labels_v2.jsonl"
OUT = AGENT_ROOT / "research" / "route_eval" / "prelim_labels.jsonl"
OUT_TRAIN = AGENT_ROOT / "research" / "route_eval" / "prelim_labels_train.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    dist: dict[str, int] = {}
    n_hybrid = 0
    with OUT.open("w", encoding="utf-8", newline="\n") as out, \
            OUT_TRAIN.open("w", encoding="utf-8", newline="\n") as out_train:
        for r in rows:
            utt = str(r["utterance"])
            hybrid = bool(_hybrid_intent_gate(utt))
            n_hybrid += int(hybrid)
            fallback = str(r.get("decided_by") or "") == "fallback_general"
            gate = ("retrieve" if (r["route_llm"] == "search" or hybrid or fallback)
                    else "skip")
            dist[gate] = dist.get(gate, 0) + 1
            rec = {"qid": r["qid"], "utterance": utt, "route_llm": r["route_llm"],
                   "decided_by": r.get("decided_by"), "hybrid": hybrid,
                   "fallback_general": fallback, "gate": gate}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                                                   
            out_train.write(json.dumps({
                "qid": r["qid"], "utterance": utt, "route_llm": gate,
                "decided_by": "derived", "votes": [{"ok": True}]},
                ensure_ascii=False) + "\n")
    print(f"n={len(rows)} hybrid={n_hybrid} dist={dist}")
    print(f"-> {OUT}")
    print(f"-> {OUT_TRAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
