                       
"""prelim 灰带带沿优化（真·小 Jev 批，2026-09-20）。

读 eval_arms_dual.py 落盘的 prelim_raw_scores（l0_p / jev_p / gold，corpus+OOD），
网格扫描带沿 (lo, hi)：p<lo→skip、p≥hi→start、带内→decider 判（skip iff jev_p<jev_τ）。
约束：OOD recall ≥ 0.98（不得比 L0 现状差）∧ band_rate ≤ 0.30（摊销时延预算）；
目标：最大化 OOD skip_correct（平局取 band_rate 低者，再平取 hi 大者）。

输出 research/route_eval/band_optimize.json + 控制台表格。

Usage:
  .venv/Scripts/python.exe research/route_eval/band_optimize.py --arm qwen3_decider_b
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = AGENT_ROOT / "research" / "route_eval"


def rule_metrics(rows, lo, hi, l0_tau, jev_tau):
    n_pos = n_neg = rec_hit = skip_hit = band = 0
    for r in rows:
        p = r["l0_p"]
        if p < lo:
            start = False
        elif p >= hi:
            start = True
        else:
            band += 1
            start = r["jev_p"] >= jev_tau
        if r["gold"] == "retrieve":
            n_pos += 1
            rec_hit += int(start)
        else:
            n_neg += 1
            skip_hit += int(not start)
    n = max(n_pos + n_neg, 1)
    return {"recall": rec_hit / max(n_pos, 1), "skip_correct": skip_hit / max(n_neg, 1),
            "band_rate": band / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--l0-tau", type=float, default=0.0,
                    help="L0 工件工作点（缺省从 l0_router 读）")
    ap.add_argument("--recall-floor", type=float, default=0.98)
    ap.add_argument("--band-cap", type=float, default=0.30)
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    sys.path.insert(0, str(AGENT_ROOT / "src"))
    from dataset_recommender.agent import l0_router as l0
    l0_tau = args.l0_tau or l0.operating_tau(l0.PRELIM_GATE, "", "tau_retrieve", 0.5)

    rep = json.loads((EVAL_DIR / "arms_dual_eval.json").read_text(encoding="utf-8"))
    arm = rep["arms"][args.arm]
    spec = json.loads((EVAL_DIR / args.arm / "decider_spec.json").read_text(encoding="utf-8"))
    jev_tau = float(spec["operating_points"]["prelim"]["tau_retrieve"])
    corpus = arm["prelim_raw_scores"]["corpus"]
    ood = arm["prelim_raw_scores"]["ood"]
    print(f"rows corpus={len(corpus)} ood={len(ood)} l0_τ={l0_tau:.4f} jev_τ={jev_tau:.4f}",
          flush=True)

    base_c = rule_metrics(corpus, l0_tau, l0_tau, l0_tau, jev_tau)
    base_o = rule_metrics(ood, l0_tau, l0_tau, l0_tau, jev_tau)
    print(f"[L0 现状] corpus {base_c} | ood {base_o}", flush=True)

    grid_lo = np.unique([0.0] + sorted(r["l0_p"] for r in corpus))
    grid_hi = np.unique([1.1] + sorted(r["l0_p"] for r in corpus))
    results = []
    for lo in grid_lo:
        if lo > l0_tau + 1e-9:
            break                                      
        for hi in grid_hi:
            if hi <= max(lo, l0_tau):
                continue
            mc = rule_metrics(corpus, lo, hi, l0_tau, jev_tau)
            if mc["band_rate"] > args.band_cap:
                continue
            mo = rule_metrics(ood, lo, hi, l0_tau, jev_tau)
            results.append({"lo": float(lo), "hi": float(hi),
                            "corpus": mc, "ood": mo})
    ok = [r for r in results if r["ood"]["recall"] >= args.recall_floor]
    ok.sort(key=lambda r: (-r["ood"]["skip_correct"], r["corpus"]["band_rate"], -r["hi"]))
    best = ok[0] if ok else None
    print("\nTop5（OOD recall≥{:.2f} 约束内按 OOD skip_correct 排）:".format(args.recall_floor),
          flush=True)
    for r in ok[:5]:
        print(f"  lo={r['lo']:.4f} hi={r['hi']:.4f} band={r['corpus']['band_rate']:.3f} "
              f"corpus rec={r['corpus']['recall']:.3f}/skip={r['corpus']['skip_correct']:.3f} "
              f"| OOD rec={r['ood']['recall']:.3f}/skip={r['ood']['skip_correct']:.3f}",
              flush=True)
    out = {"arm": args.arm, "l0_tau": l0_tau, "jev_tau": jev_tau,
           "l0_baseline": {"corpus": base_c, "ood": base_o},
           "best": best, "top5": ok[:5]}
    (EVAL_DIR / "band_optimize.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {EVAL_DIR / 'band_optimize.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
