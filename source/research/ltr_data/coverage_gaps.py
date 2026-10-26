                       
"""xdata-reuse WP2: 目录覆盖空洞地图（纯本地统计，零网络零 LLM）。

定义：某 query 的全部候选中，融合金标签（llm_semantic）最高分 <= 1
→ 目录对该需求「没有可给的好答案」。按 cluster_id（train/heldout）与
ood_class（OOD）聚合，回答「该去补哪类数据集」。
输出：_coverage_gaps.json + 打印摘要。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent
OOD = DATA.parent / "ood_eval"


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gap_stats(rows, key_fn):
    per_q = []
    for r in rows:
        fused = [c.get("llm_semantic") for c in r.get("candidates", []) if c.get("llm_semantic") is not None]
        best = max(fused) if fused else None
        per_q.append({
            "query_id": r.get("query_id"), "query": r.get("query"),
            "key": key_fn(r), "best": best, "n_cand": len(r.get("candidates", [])),
            "constraints": r.get("constraints"),
            "abstain": r.get("abstain"), "abstain_reason": r.get("abstain_reason"),
        })
    agg = collections.defaultdict(lambda: {"n": 0, "gap": 0, "weak": 0, "best_sum": 0.0, "best_n": 0})
    for q in per_q:
        a = agg[q["key"]]
        a["n"] += 1
        if q["best"] is None:
            a["gap"] += 1                         
        elif q["best"] <= 1:
            a["gap"] += 1
        elif q["best"] == 2:
            a["weak"] += 1
        if q["best"] is not None:
            a["best_sum"] += q["best"]
            a["best_n"] += 1
    table = []
    for k, a in agg.items():
        table.append({
            "key": k, "n_queries": a["n"], "gap_queries": a["gap"], "weak_queries": a["weak"],
            "gap_rate": round(a["gap"] / a["n"], 4),
            "mean_best": round(a["best_sum"] / a["best_n"], 3) if a["best_n"] else None,
        })
    table.sort(key=lambda x: (-x["gap_rate"], -x["n_queries"]))
    return per_q, table


def main():
    report = {}
    for name in ["train", "heldout"]:
        rows = load(DATA / f"synth_{name}_xscore.jsonl")
        per_q, table = gap_stats(rows, lambda r: r.get("cluster_id") or "<none>")
        gaps = [q for q in per_q if q["best"] is None or q["best"] <= 1]
        report[name] = {
            "n_queries": len(per_q), "n_gap": len(gaps),
            "gap_rate": round(len(gaps) / len(per_q), 4),
            "by_cluster_top": [t for t in table if t["n_queries"] >= 2][:25],
            "gap_query_list": gaps[:300],
        }
        print(f"== {name}: {len(per_q)} queries, gap {len(gaps)} ({report[name]['gap_rate']})")
        for t in report[name]["by_cluster_top"][:10]:
            print(f"   {t['key']}: n={t['n_queries']} gap={t['gap_queries']} rate={t['gap_rate']} mean_best={t['mean_best']}")
    ood_path = OOD / "synth_ood_label_input_xscore.jsonl"
    if ood_path.exists():
        rows = load(ood_path)
        per_q, table = gap_stats(rows, lambda r: r.get("ood_class") or "<none>")
        gaps = [q for q in per_q if q["best"] is None or q["best"] <= 1]
                                                 
        by_expect = collections.defaultdict(lambda: {"n": 0, "gap": 0})
        for q, r in zip(per_q, rows):
            e = str(r.get("expected"))
            by_expect[e]["n"] += 1
            if q["best"] is None or q["best"] <= 1:
                by_expect[e]["gap"] += 1
        report["ood"] = {
            "n_queries": len(per_q), "n_gap": len(gaps),
            "gap_rate": round(len(gaps) / len(per_q), 4),
            "by_ood_class": table,
            "by_expected": {k: {**v, "gap_rate": round(v["gap"] / v["n"], 4)} for k, v in by_expect.items()},
            "gap_query_list": gaps,
        }
        print(f"== ood: {len(per_q)} queries, gap {len(gaps)} ({report['ood']['gap_rate']})")
        for t in table:
            print(f"   {t['key']}: n={t['n_queries']} gap={t['gap_queries']} rate={t['gap_rate']} mean_best={t['mean_best']}")
        print("   by_expected:", json.dumps(report["ood"]["by_expected"], ensure_ascii=False))
    out = DATA / "_coverage_gaps.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
