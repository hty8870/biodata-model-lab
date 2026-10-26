                       
"""选题 1 生死检验：一致性增益 vs 置信度分层（2026-09-22，OOD 108，生产代码直调）。
问题（Codex 选题评审的挑刺）：「一致时全对」会不会只是两个模型共同筛出简单题？
检验：按 decider 单模型置信度分层，看 L0 附和/反对在各层内的准确率差；
并把「双判官一致闸」与「decider 置信度阈值」在等覆盖率下比条件错误率。
若中间置信带的附和增益消失 → 一致性只是「简单题简单」的同义反复；
若仍在 → 一致性携带超出单模型置信度的增量信息。
产物：consistency_vs_confidence_0922.json + 同名 md（研究/02-工具执行一致性 相关；放 route_eval）。
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3] / 'source'
sys.path.insert(0, str(REPO / "src"))

from dataset_recommender.agent import decider, l0_router              

OOD = REPO / "research" / "route_eval" / "ood_route_queries.jsonl"
OUT = REPO / "research" / "route_eval" / "consistency_vs_confidence_0922"

rows = []
for line in OOD.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    d = decider.score("binary", r["utterance"])
    l = l0_router.score(l0_router.ROUTE_BINARY, r["utterance"])
    if d is None or l is None:
        print("SKIP(None)", r["qid"], d is None, l is None)
        continue
    rows.append({"qid": r["qid"], "cat": r.get("cat"), "gold": r["gate"],
                 "d_label": d["label"], "d_conf": d["confidence"],
                 "l_label": l["label"], "l_conf": l["confidence"]})
print(f"rows={len(rows)}")

BINS = [(0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
strata = defaultdict(lambda: {"n": 0, "d_correct": 0, "agree_n": 0, "agree_correct": 0,
                              "disagree_n": 0, "disagree_correct": 0})
for r in rows:
    c = r["d_conf"]
    for lo, hi in BINS:
        if lo <= c < hi:
            s = strata[f"[{lo},{hi})"]
            s["n"] += 1
            d_ok = r["d_label"] == r["gold"]
            s["d_correct"] += d_ok
            if r["l_label"] == r["d_label"]:
                s["agree_n"] += 1
                s["agree_correct"] += d_ok
            else:
                s["disagree_n"] += 1
                s["disagree_correct"] += d_ok
            break

print("\n按 decider 置信度分层（OOD binary，n=108）：")
print(f"{'置信带':<12}{'n':>4}{'decider acc':>13}{'附和n':>7}{'附和 acc':>10}{'分歧n':>7}{'分歧 acc':>10}")
for b in BINS:
    k = f"[{b[0]},{b[1]})"
    s = strata[k]
    if not s["n"]:
        continue
    da = s["d_correct"] / s["n"]
    aa = f"{s['agree_correct']/s['agree_n']:.3f}" if s["agree_n"] else "  n/a"
    da2 = f"{s['disagree_correct']/s['disagree_n']:.3f}" if s["disagree_n"] else "  n/a"
    print(f"{k:<12}{s['n']:>4}{da:>13.3f}{s['agree_n']:>7}{aa:>10}{s['disagree_n']:>7}{da2:>10}")

agree_covered = [r for r in rows if r["l_label"] == r["d_label"]]
cov = len(agree_covered) / len(rows)
gate_err = 1 - sum(r["d_label"] == r["gold"] for r in agree_covered) / max(len(agree_covered), 1)
sorted_by_conf = sorted(rows, key=lambda r: -r["d_conf"])
thr_covered = sorted_by_conf[:len(agree_covered)]
thr_err = 1 - sum(r["d_label"] == r["gold"] for r in thr_covered) / max(len(thr_covered), 1)
print(f"\n等覆盖对照（覆盖率 {cov:.3f}）：")
print(f"  双判官一致闸   覆盖集错误率 = {gate_err:.4f}（n={len(agree_covered)}）")
print(f"  置信度阈值选择 覆盖集错误率 = {thr_err:.4f}（n={len(thr_covered)}）")

sweep = []
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    cov_rows = [r for r in rows if r["d_conf"] >= tau]
    if not cov_rows:
        continue
    err = 1 - sum(r["d_label"] == r["gold"] for r in cov_rows) / len(cov_rows)
    sweep.append({"tau": tau, "coverage": len(cov_rows) / len(rows), "error": err})
    print(f"  τ={tau:<5} 覆盖 {len(cov_rows)/len(rows):.3f} 错误率 {err:.4f}")

dec_acc = sum(r["d_label"] == r["gold"] for r in rows) / len(rows)
l0_acc = sum(r["l_label"] == r["gold"] for r in rows) / len(rows)
print(f"\n锚点：decider 单模型 acc={dec_acc:.3f}，L0 单模型 acc={l0_acc:.3f}")

OUT.with_suffix(".json").write_text(json.dumps({
    "rows": rows,
    "strata": {k: v for k, v in strata.items()},
    "equal_coverage": {"coverage": cov, "agree_gate_error": gate_err, "conf_threshold_error": thr_err},
    "sweep": sweep, "decider_acc": dec_acc, "l0_acc": l0_acc,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n写出 {OUT.with_suffix('.json')}")
