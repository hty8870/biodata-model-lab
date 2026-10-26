                       
"""实验 S 的分析器：CONS−SQ 簇配对对比 + 已知答案校验 + 预注册判据。

复用（import，不复制）W23 分析器的核心件：`paired_cluster_bootstrap` / `significant` /
`violation_index` / `violation_rate` / `read_jsonl`（`exp_20260929/analyze_exp_ab.py`）——
bootstrap 实现全项目只此一份。

已知答案校验（预注册 §6）：
- KA1 staging 奇偶由 stage_data.py 负责（本器复核臂账层数）；
- KA2 **跨实验复现**：S_SQ_<seed> 在 heldout5/oodonly 的逐查询 nDCG@10 必须等于 W23
  `A_f0_d1_<seed>`（同 seed/数据/标签/deterministic；W23 产物缺失时降级为警告）；
- KA3 自比自：S_SQ_<seed> − S_SQ_<seed> 的 CI 恰为 [0,0]；
- KA4 合成 +0.01 平移注入 CONS 臂 → 判据必须 ACCEPT（验证判据链路没断）。

主判据（预注册 §4）：CONS−SQ，heldout5∪oodonly 并集宏平均，5 seed 池化逐查询配对，
簇配对 bootstrap 10,000 次；≥+0.005 且 CI 下界>0 → ACCEPT；≤−0.005 且 CI 上界<0 → 反向；其余 HOLD。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "exp_20260929"))
import analyze_exp_ab as ab              

OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_labelchain"
W23_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260929_ab"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"
OOD_EVALONLY = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_evalonly_xscore.jsonl"

SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)
PRIMARY_SETS = ("heldout5", "oodonly")
ALL_SETS = ("heldout5", "oodonly", "heldsingle")
MODEL = "ranknet_mlp_full"
ACCEPT_MARGIN = 0.005


                                                                       
def load_panel() -> dict:
    index_path = OUT_ROOT / "arms_index.json"
    if not index_path.is_file():
        raise SystemExit(f"! 没有臂账：{index_path}（先跑 run_arms.py）")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    panel: dict[str, dict] = {}
    for arm_id, meta in index.items():
        if meta.get("returncode") != 0:
            continue
        per_query = OUT_ROOT / "arms" / arm_id / "per_query.jsonl"
        if not per_query.is_file():
            continue
        by_set: dict[str, dict] = {}
        from collections import Counter as _Counter
        counters: _Counter = _Counter()
        for row in ab.read_jsonl(per_query):
                                                                        
                                                     
            face = row["eval_set"]
            key = f"{row['query_id']}#{counters[(face, row['query_id'])]}"
            counters[(face, row["query_id"])] += 1
            by_set.setdefault(face, {})[key] = {
                "ndcg": float(row["ndcg10"][MODEL]),
                "hist": float(row["ndcg10"].get("hist_gbdt_full", float("nan"))),
                "lex": float(row["ndcg10"].get("lex", float("nan"))),
                "top10": row["top10_uids"].get(MODEL) or [],
            }
        panel[arm_id] = by_set
    return index, panel


def vectors(by_set: dict, eval_sets: tuple[str, ...]) -> dict[str, float]:
    return {f"{name}/{qid}": entry["ndcg"] for name in eval_sets
            for qid, entry in by_set.get(name, {}).items()}


def cluster_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in ab.read_jsonl(HELDOUT_GOLD):
        key = row.get("cluster_id") or row["query_id"]
        out[f"heldout5/{row['query_id']}#0"] = f"heldout5:{key}"
        out[f"heldsingle/{row['query_id']}#0"] = f"heldsingle:{key}"
    from collections import Counter as _Counter
    seen: _Counter = _Counter()
    for row in ab.read_jsonl(OOD_EVALONLY):
        key = row.get("ood_class") or row["query_id"]
        occ = seen[row["query_id"]]
        seen[row["query_id"]] += 1
        out[f"oodonly/{row['query_id']}#{occ}"] = f"oodonly:{key}"
    return out


def contrast(panel: dict, arm_a: str, arm_b: str, clusters: dict, eval_sets: tuple[str, ...],
             n_boot: int, seed: int) -> dict:
    """跨 seed 池化：逐 (seed, 查询) 取 A−B，按簇自助。A/B 的键集不一致直接炸。"""
    values: dict[str, list[float]] = {}
    per_seed: dict[int, float] = {}
    for s in SEEDS:
        va = vectors(panel[f"S_{arm_a}_{s}"], eval_sets)
        vb = vectors(panel[f"S_{arm_b}_{s}"], eval_sets)
        if set(va) != set(vb):
            raise AssertionError(f"seed {s} 两臂评测查询集不同：{len(set(va) ^ set(vb))} 条")
        deltas = [va[k] - vb[k] for k in va]
        per_seed[s] = float(np.mean(deltas))
        for k, d in zip(va, deltas):
            values.setdefault(clusters.get(k, f"unknown:{k}"), []).append(d)
    result = ab.paired_cluster_bootstrap(values, n_boot=n_boot, seed=seed)
    result["per_seed"] = per_seed
    return result


def arm_mean(by_set: dict, eval_sets: tuple[str, ...]) -> float:
    vals = [e["ndcg"] for name in eval_sets for e in by_set.get(name, {}).values()]
    return float(np.mean(vals))


                                                                         
def ka2_w23_reproduction(panel: dict) -> dict:
    """S_SQ 必须逐位复现 W23 A_f0_d1（同 seed/数据/标签/确定性）。"""
    w23_index = W23_ROOT / "arms_index.json"
    if not w23_index.is_file():
        return {"status": "warn", "detail": f"W23 臂产物缺失（{W23_ROOT}），降级为警告", "mismatch": None}
    w23 = json.loads(w23_index.read_text(encoding="utf-8"))
    mismatches = []
    compared = 0
    for s in SEEDS:
        w23_arm = f"A_f0_d1_{s}"
        if w23.get(w23_arm, {}).get("returncode") != 0:
            return {"status": "warn", "detail": f"W23 臂 {w23_arm} 不可用", "mismatch": None}
        w23_pq = W23_ROOT / "arms" / w23_arm / "per_query.jsonl"
        w23_vec: dict[str, float] = {}
        from collections import Counter as _Counter
        counters: _Counter = _Counter()
        for row in ab.read_jsonl(w23_pq):
            face = row["eval_set"]
            key = f"{face}/{row['query_id']}#{counters[(face, row['query_id'])]}"
            counters[(face, row["query_id"])] += 1
            w23_vec[key] = float(row["ndcg10"][MODEL])
        mine = vectors(panel[f"S_SQ_{s}"], PRIMARY_SETS)
        if set(mine) != set(w23_vec):
            mismatches.append({"seed": s, "key_diff": len(set(mine) ^ set(w23_vec))})
            continue
        for k in mine:
            compared += 1
            if mine[k] != w23_vec[k]:
                mismatches.append({"seed": s, "key": k, "mine": mine[k], "w23": w23_vec[k]})
    ok = not mismatches
    return {"status": "pass" if ok else "fail",
            "detail": f"{compared} 个 (seed,查询,面) 值全部逐位一致" if ok
            else f"{len(mismatches)} 处不一致（首例 {mismatches[0]}）",
            "mismatch": mismatches[:5]}


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

                              
    rc = ab.self_test()
    check("W23 分析器自测", rc == 0, f"returncode={rc}")

                             
    rng = np.random.default_rng(3)
    queries = [f"heldout5/q{i}" for i in range(80)] + [f"oodonly/q{i}" for i in range(40)]
    base = {q: 0.9 + float(rng.normal(0, 0.05)) for q in queries}
    cl = {q: f"{q.split('/')[0]}:c{i // 4}" for i, q in enumerate(queries)}
    panel_syn: dict[str, dict] = {}
    for s in SEEDS:
        noise = {q: float(rng.normal(0, 0.02)) for q in queries}
        mk = lambda shift: {name: {q.split("/", 1)[1]: {"ndcg": base[q] + noise[q] + shift}}
                            for name in ("heldout5", "oodonly") for q in queries
                            if q.startswith(name)}
        panel_syn[f"S_SQ_{s}"] = mk(0.0)
        panel_syn[f"S_CONS_{s}"] = mk(0.01)               
    r0 = contrast(panel_syn, "SQ", "SQ", cl, PRIMARY_SETS, n_boot=2000, seed=1)
    check("KA3 自比自 CI 恰为 [0,0]", r0["mean"] == 0.0 and r0["ci_low"] == 0.0 == r0["ci_high"],
          f"mean={r0['mean']} CI=[{r0['ci_low']},{r0['ci_high']}]")
    r1 = contrast(panel_syn, "CONS", "SQ", cl, PRIMARY_SETS, n_boot=2000, seed=1)
    check("KA4 恒 +0.01 平移 → 判据 ACCEPT",
          abs(r1["mean"] - 0.01) < 1e-12 and ab.significant(r1, ACCEPT_MARGIN),
          f"mean={r1['mean']:.6f} CI=[{r1['ci_low']:.6f},{r1['ci_high']:.6f}]")
    r2 = contrast(panel_syn, "SQ", "CONS", cl, PRIMARY_SETS, n_boot=2000, seed=1)
    check("KA4b 恒 −0.01 → 反向判据触发",
          abs(r2["mean"] + 0.01) < 1e-12 and r2["ci_high"] < 0,
          f"mean={r2['mean']:.6f}")

    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


                                                                      
def _fmt(v, digits: int = 5) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(r: dict) -> str:
    return f"[{_fmt(r['ci_low'])},{_fmt(r['ci_high'])}]"


def verdict(r: dict) -> str:
    if r["mean"] is not None and r["mean"] >= ACCEPT_MARGIN and r["ci_low"] > 0:
        return "ACCEPT（建议换共识链）"
    if r["mean"] is not None and r["mean"] <= -ACCEPT_MARGIN and r["ci_high"] < 0:
        return "反向（建议维持单评委）"
    return "HOLD（无决定性差异）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    index, panel = load_panel()
    missing = [f"S_{a}_{s}" for a in ("SQ", "CONS") for s in SEEDS if f"S_{a}_{s}" not in panel]
    if missing:
        raise SystemExit(f"! 缺臂：{missing}")
    clusters = cluster_map()
    report: dict = {"n_boot": args.n_boot, "boot_seed": args.seed,
                    "accept_margin": ACCEPT_MARGIN, "seeds": list(SEEDS), "model": MODEL}

    print("=== KA2 跨实验复现（S_SQ vs W23 A_f0_d1）===")
    ka2 = ka2_w23_reproduction(panel)
    print(f"  [{ka2['status']}] {ka2['detail']}")
    report["ka2_w23_reproduction"] = ka2
    if ka2["status"] == "fail":
        raise SystemExit("! KA2 失败：现役臂不能复现 W23，停盘查因，不信任何后续数字")

    print("\n=== 逐臂行账（ranknet_mlp_full，eval-label sem）===")
    print(f"{'arm':<18}{'heldout5':>10}{'oodonly':>10}{'heldsingle':>12}{'hist(h5)':>10}")
    for arm_id in sorted(panel):
        by_set = panel[arm_id]
        row = {"heldout5": arm_mean(by_set, ("heldout5",)),
               "oodonly": arm_mean(by_set, ("oodonly",)),
               "heldsingle": arm_mean(by_set, ("heldsingle",)),
               "hist_h5": float(np.mean([e["hist"] for e in by_set.get("heldout5", {}).values()]))}
        report.setdefault("arms", {})[arm_id] = row
        print(f"{arm_id:<18}{_fmt(row['heldout5']):>10}{_fmt(row['oodonly']):>10}"
              f"{_fmt(row['heldsingle']):>12}{_fmt(row['hist_h5']):>10}")

    print("\n=== 主判据：CONS − SQ（heldout5∪oodonly 并集，簇配对）===")
    primary = contrast(panel, "CONS", "SQ", clusters, PRIMARY_SETS, args.n_boot, args.seed)
    report["primary"] = primary
    print(f"  Δ = {primary['mean']:+.5f}  CI={_ci(primary)}  P(>0)={primary['share_positive']:.3f}"
          f"  簇={primary['clusters']} 查询×seed={primary['queries']}")
    print(f"  逐 seed Δ：{ {s: round(v, 5) for s, v in primary['per_seed'].items()} }")
    v = verdict(primary)
    report["verdict"] = v
    print(f"  → {v}")

    print("\n=== 分面拆开 ===")
    per_face = {}
    for face in ALL_SETS:
        r = contrast(panel, "CONS", "SQ", clusters, (face,), args.n_boot, args.seed)
        per_face[face] = r
        print(f"  {face:<11} Δ = {r['mean']:+.5f}  CI={_ci(r)}  → {verdict(r)}")
    report["per_face"] = per_face

    print("\n=== 评测面噪声顶（同一 SQ 模型：heldsingle − heldout5，配对）===")
    ceiling_values: dict[str, list[float]] = {}
    for s in SEEDS:
        v_single = vectors(panel[f"S_SQ_{s}"], ("heldsingle",))
        v_gold = vectors(panel[f"S_SQ_{s}"], ("heldout5",))
        keys_single = {k.split("/", 1)[1]: v for k, v in v_single.items()}
        keys_gold = {k.split("/", 1)[1]: v for k, v in v_gold.items()}
        assert set(keys_single) == set(keys_gold), "heldout 两面的查询集应一致"
        for qid in keys_single:
                                                                    
                                                           
            ceiling_values.setdefault(clusters.get(f"heldout5/{qid}", f"unknown:{qid}"), []).append(
                keys_single[qid] - keys_gold[qid])
    ceiling = ab.paired_cluster_bootstrap(ceiling_values, n_boot=args.n_boot, seed=args.seed)
    report["eval_face_ceiling"] = ceiling
    print(f"  单评委面 − 金标面 = {ceiling['mean']:+.5f}  CI={_ci(ceiling)}"
          f"（{'单评委面更宽松（读数虚高）' if ceiling['mean'] > 0 else '单评委面更苛刻（读数虚低）'}）")

    print("\n=== 违规率（top-5 grade<2，W23 口径）===")
    grades = ab.violation_index()
    viol = {}
    for arm in ("SQ", "CONS"):
        rates = []
        for s in SEEDS:
                                                                 
                             
            by_set = {}
            for face, entries in panel[f"S_{arm}_{s}"].items():
                if face in PRIMARY_SETS:
                    by_set[face] = {k.split("#")[0]: v for k, v in entries.items()}
            rate, n = ab.violation_rate(by_set, grades)
            rates.append(rate)
        viol[arm] = {"mean": float(np.mean(rates)), "per_seed": rates}
        print(f"  {arm}: mean={viol[arm]['mean']:.4f}  per-seed={[round(r, 4) for r in rates]}")
    report["violation_top5"] = viol

                                     
    print("\n=== 探索性：Δ(CONS−SQ) × xscore_spread（heldout5 面）===")
    spread_by_q: dict[str, float] = {}
    for row in ab.read_jsonl(HELDOUT_GOLD):
        sp = [c.get("xscore_spread") for c in row.get("candidates") or []
              if c.get("xscore_spread") is not None]
        if sp:
            spread_by_q[row["query_id"]] = float(np.mean(sp))
    deltas_by_q: dict[str, list[float]] = {}
    for s in SEEDS:
        va = vectors(panel[f"S_CONS_{s}"], ("heldout5",))
        vb = vectors(panel[f"S_SQ_{s}"], ("heldout5",))
        for k in va:
                                                      
            deltas_by_q.setdefault(k.split("/", 1)[1].split("#")[0], []).append(va[k] - vb[k])
    xs = [spread_by_q[q] for q in deltas_by_q if q in spread_by_q]
    ys = [float(np.mean(deltas_by_q[q])) for q in deltas_by_q if q in spread_by_q]
    if len(xs) >= 30:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        hi = [y for x, y in zip(xs, ys) if x >= 1.0]
        lo = [y for x, y in zip(xs, ys) if x == 0.0]
        slice_info = {"n_queries": len(xs), "pearson": corr,
                      "mean_delta_spread>=1": float(np.mean(hi)) if hi else None,
                      "mean_delta_spread=0": float(np.mean(lo)) if lo else None,
                      "n_hi": len(hi), "n_lo": len(lo)}
        report["spread_slice"] = slice_info
        print(f"  n={len(xs)}  pearson(spread, Δ)={corr:+.3f}")
        print(f"  spread≥1 的 {len(hi)} 查询：平均 Δ={slice_info['mean_delta_spread>=1']}；"
              f"spread=0 的 {len(lo)} 查询：平均 Δ={slice_info['mean_delta_spread=0']}")
    else:
        report["spread_slice"] = {"n_queries": len(xs), "note": "样本太少，跳过"}

    out_path = OUT_ROOT / "analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"\n最终裁决（预注册判据）：{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
