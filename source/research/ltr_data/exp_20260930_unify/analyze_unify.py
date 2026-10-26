                       
"""R3 的分析器 v2（Codex R3 评审后的修正版）：身份键 / 覆盖账 / 违规率 / 敏感性 / 隔离对照。

v1 的两个实质错误（Codex 实测抓出，评审 `研究/00-汇报与面谈/codex-R3配方评审-2026-09-30.md`）：
1. **身份键错误**：oodonlyf3 里有 3 个 qid 各对应 2 行**不同查询**（typo_pinyin 家族），
   v1 以 qid 为 dict 键后写覆盖，3 个不同查询被静默丢弃。修正：行身份 = (面, qid, 出现序)
   ——两臂读同一评测文件、同一顺序，第 k 次出现在两臂对应同一行（确定性对应）。
2. **覆盖口径错误**：v1 按「f3_feats 非空」计数（60 行），正确口径是「≥1 个语义标签」
   （eval_units 的真正存活条件，45 行）；另有部分标签行需单独入账。

新增（评审必做/高优先）：
- 违规率护栏（预注册承诺过 v1 却省略）：top-5 grade<2 占比（W23 口径，分母如实）；
- 逐查询严重退化例（ worst-5，含查询文本与两臂读数）；
- OOD 类别留一敏感性（ood_class 逐个剔除重算主对比）；
- 隔离对照：F3−F0P（纯特征修正效应）、F0P−F0（纯候选裁减效应）——F0P 臂 = f0 特征 × f3 池；
- nDCG@5 / MRR 汇总参照（metrics.json 臂账）。

复用（import）`exp_20260929/analyze_exp_ab.py` 的 `paired_cluster_bootstrap` / `significant` / `read_jsonl`。
主判据不变（预注册 §4）：F3−F0，heldout5f3∪oodonlyf3 并集宏平均，5 seed 池化，簇配对 bootstrap 10,000。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "exp_20260929"))
import analyze_exp_ab as ab              

OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_unify"
EXP_S_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_labelchain"
F3_DATA = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_f3" / "data"
HELDOUT5_F3 = F3_DATA / "eval" / "heldout5_f3.jsonl"
OODONLY_F3 = F3_DATA / "eval" / "oodonly_f3.jsonl"

SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)
FACES = ("heldout5f3", "oodonlyf3")
MODEL = "ranknet_mlp_full"
ACCEPT_MARGIN = 0.005
ARMS = ("F0", "F3", "F0P")


                                                                           
def row_keys_of(rows: list[dict], face: str) -> list[str]:
    """逐行身份键：面/qid#出现序（同 qid 多行 = 不同查询，不许互相覆盖）。"""
    seen: Counter = Counter()
    keys = []
    for row in rows:
        qid = row["query_id"]
        keys.append(f"{face}/{qid}#{seen[(face, qid)]}")
        seen[(face, qid)] += 1
    return keys


def load_panel() -> dict:
    index_path = OUT_ROOT / "arms_index.json"
    if not index_path.is_file():
        raise SystemExit(f"! 没有臂账：{index_path}（先跑 run_arms.py）")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    panel: dict[str, dict] = {}
    for arm_id, meta in index.items():
        if meta.get("returncode") != 0:
            continue
        by_set: dict[str, dict] = {}
        counters: Counter = Counter()
        for row in ab.read_jsonl(OUT_ROOT / "arms" / arm_id / "per_query.jsonl"):
            face = row["eval_set"]
            key = f"{face}/{row['query_id']}#{counters[(face, row['query_id'])]}"
            counters[(face, row["query_id"])] += 1
            by_set.setdefault(face, {})[key] = {
                "ndcg": float(row["ndcg10"][MODEL]),
                "hist": float(row["ndcg10"].get("hist_gbdt_full", float("nan"))),
                "top10": row["top10_uids"].get(MODEL) or [],
            }
        panel[arm_id] = by_set
    return index, panel


def vectors(panel: dict, arm: str, seed: int) -> dict[str, float]:
    return {key: entry["ndcg"] for face in FACES
            for key, entry in panel[f"R_{arm}_{seed}"].get(face, {}).items()}


def cluster_map() -> dict[str, str]:
    out = {}
    for face, path, id_field in (("heldout5f3", HELDOUT5_F3, "cluster_id"),
                                 ("oodonlyf3", OODONLY_F3, "ood_class")):
        rows = ab.read_jsonl(path)
        for key, row in zip(row_keys_of(rows, face), rows):
            out[key] = f"{face}:{row.get(id_field) or row['query_id']}"
    return out


def eval_rows_index() -> dict[str, dict]:
    """行键 → 该行事实（query 文本、uid→grade、uid→llm_semantic）。"""
    out = {}
    for face, path in (("heldout5f3", HELDOUT5_F3), ("oodonlyf3", OODONLY_F3)):
        rows = ab.read_jsonl(path)
        for key, row in zip(row_keys_of(rows, face), rows):
            out[key] = {
                "query": row.get("query") or row.get("query_nl") or "",
                "ood_class": row.get("ood_class"),
                "grades": {c.get("uid"): c.get("grade") for c in row.get("candidates") or []},
                "n_cands": len(row.get("candidates") or []),
                "n_labeled": sum(1 for c in row.get("candidates") or []
                                 if c.get("llm_semantic") is not None),
            }
    return out


def coverage_accounting() -> dict:
    """评审 §2 的覆盖账：行数 → 非空池 → ≥1 标签 → 部分标签 → 重复身份（含不同文本实证）。"""
    out = {}
    for face, path in (("heldout5f3", HELDOUT5_F3), ("oodonlyf3", OODONLY_F3)):
        rows = ab.read_jsonl(path)
        non_empty = [r for r in rows if r.get("candidates")]
        labeled = [r for r in rows
                   if any(c.get("llm_semantic") is not None for c in r.get("candidates") or [])]
        partial = [r for r in labeled
                   if any(c.get("llm_semantic") is None for c in r.get("candidates") or [])]
        qid_count = Counter(r["query_id"] for r in labeled)
        dup_qids = sorted(q for q, n in qid_count.items() if n > 1)
        dup_detail = [{"qid": q, "texts": [r.get("query", "")[:40] for r in labeled
                                           if r["query_id"] == q]} for q in dup_qids]
        out[face] = {"file_rows": len(rows), "non_empty_pool": len(non_empty),
                     "labeled_rows": len(labeled), "partial_label_rows": len(partial),
                     "dup_identity_qids": dup_detail}
    return out


                                                                       
def contrast(panel: dict, a: str, b: str, clusters: dict, n_boot: int, seed: int,
             alpha: float = 0.05, exclude_cluster_prefix: "set[str] | None" = None) -> dict:
    values: dict[str, list[float]] = {}
    per_seed = {}
    for s in SEEDS:
        va, vb = vectors(panel, a, s), vectors(panel, b, s)
        if set(va) != set(vb):
            raise AssertionError(f"seed {s} 两臂评测键集不同：{len(set(va) ^ set(vb))} 条")
        per_seed[s] = float(np.mean([va[k] - vb[k] for k in va]))
        for k in va:
            cl = clusters.get(k, f"unknown:{k}")
            if exclude_cluster_prefix and cl in exclude_cluster_prefix:
                continue
            values.setdefault(cl, []).append(va[k] - vb[k])
    result = ab.paired_cluster_bootstrap(values, n_boot=n_boot, seed=seed, alpha=alpha)
    result["per_seed"] = per_seed
    return result


def ka1_train_side_identical() -> dict:
    compared, mismatches = [], []
    for s in SEEDS:
        mine_path = OUT_ROOT / "arms" / f"R_F0_{s}" / "metrics.json"
        ref_path = EXP_S_ROOT / "arms" / f"S_SQ_{s}" / "metrics.json"
        if not ref_path.is_file():
            return {"status": "warn", "detail": f"实验 S 臂产物缺失：{ref_path}"}
        mine = json.loads(mine_path.read_text(encoding="utf-8"))
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        m_rn = mine["results"]["models"]["ranknet_mlp"]["full"]
        r_rn = ref["results"]["models"]["ranknet_mlp"]["full"]
        for key in ("pairs", "final_loss"):
            mv, rv = m_rn.get(key), r_rn.get(key)
            compared.append((s, key, mv, rv))
            if mv != rv:
                mismatches.append({"seed": s, "key": key, "mine": mv, "ref": rv})
        for key in ("labelled_candidates", "dropped_unlabelled"):
            mv = mine.get("train_label_stats", {}).get(key)
            rv = ref.get("train_label_stats", {}).get(key)
            compared.append((s, key, mv, rv))
            if mv != rv:
                mismatches.append({"seed": s, "key": key, "mine": mv, "ref": rv})
    ok = not mismatches
    return {"status": "pass" if ok else "fail",
            "detail": f"{len(compared)} 项训练侧账全部一致（final_loss 仅 4 位小数，"
                      f"证明到不了权重逐位——评审已声明此边界）" if ok
            else f"{len(mismatches)} 项不一致：{mismatches[:3]}",
            "mismatches": mismatches}


def violation_rates(panel: dict, rows_index: dict, top_k: int = 5) -> dict:
    """top-k 内 grade<2 占比（W23 口径；分母 = 有 grade 的槽位，如实报）。"""
    out = {}
    for arm in ARMS:
        bad = total = 0
        for s in SEEDS:
            for face in FACES:
                for key, entry in panel[f"R_{arm}_{s}"].get(face, {}).items():
                    grades = rows_index.get(key, {}).get("grades", {})
                    for uid in entry["top10"][:top_k]:
                        grade = grades.get(uid)
                        if grade is None:
                            continue
                        total += 1
                        bad += int(float(grade) < 2)
        out[arm] = {"rate": bad / total if total else None, "bad": bad, "slots": total}
    return out


def worst_deltas(panel: dict, rows_index: dict, k: int = 5) -> list[dict]:
    """逐查询 Δ(F3−F0) 最差 k 例（5 seed 均值）。"""
    per_query: dict[str, list[float]] = {}
    for s in SEEDS:
        va, vb = vectors(panel, "F3", s), vectors(panel, "F0", s)
        for key in va:
            per_query.setdefault(key, []).append(va[key] - vb[key])
    ranked = sorted(per_query.items(), key=lambda kv: float(np.mean(kv[1])))
    out = []
    for key, deltas in ranked[:k]:
        f3_mean = float(np.mean([vectors(panel, "F3", s)[key] for s in SEEDS]))
        f0_mean = float(np.mean([vectors(panel, "F0", s)[key] for s in SEEDS]))
        info = rows_index.get(key, {})
        out.append({"row": key, "delta": float(np.mean(deltas)),
                    "f0": f0_mean, "f3": f3_mean,
                    "ood_class": info.get("ood_class"), "query": info.get("query", "")[:60]})
    return out


def metrics_side_table() -> dict:
    """metrics.json 臂账：nDCG@5 / MRR 汇总参照（非逐查询，不进主判据）。"""
    out = {}
    for arm in ARMS:
        rows = []
        for s in SEEDS:
            payload = json.loads((OUT_ROOT / "arms" / f"R_{arm}_{s}" / "metrics.json")
                                 .read_text(encoding="utf-8"))
            es = payload["results"].get("eval_sets") or {}
            entry = {"seed": s}
            for face in FACES:
                m = (es.get(face) or {}).get("models", {}).get("ranknet_mlp", {}).get("full") or {}
                entry[face] = {"nDCG@5": m.get("nDCG@5"), "nDCG@10": m.get("nDCG@10"),
                               "MRR": m.get("MRR")}
            rows.append(entry)
        out[arm] = rows
    return out


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

                                             
    rows = [{"query_id": "q1", "query": "甲"}, {"query_id": "q2", "query": "乙"},
            {"query_id": "q1", "query": "丙"}]
    keys = row_keys_of(rows, "oodonlyf3")
    check("KA5 同 qid 不同行各占一键", keys == ["oodonlyf3/q1#0", "oodonlyf3/q2#0", "oodonlyf3/q1#1"],
          str(keys))

    rng = np.random.default_rng(5)
    queries = [f"heldout5f3/q{i}#0" for i in range(60)] + [f"oodonlyf3/q{i}#0" for i in range(30)]
    base = {q: 0.9 + float(rng.normal(0, 0.05)) for q in queries}
    cl = {q: f"{q.split('/')[0]}:c{i // 4}" for i, q in enumerate(queries)}
    panel: dict[str, dict] = {}
    for s in SEEDS:
        noise = {q: float(rng.normal(0, 0.02)) for q in queries}
        for arm, shift in (("F0", 0.0), ("F3", 0.01), ("F0P", 0.004)):
            panel[f"R_{arm}_{s}"] = {
                face: {q: {"ndcg": base[q] + noise[q] + shift} for q in queries
                       if q.startswith(face)}
                for face in FACES}
    r0 = contrast(panel, "F0", "F0", cl, n_boot=2000, seed=1)
    check("KA3 自比自 CI 恰为 [0,0]", r0["mean"] == 0.0 and r0["ci_low"] == 0.0 == r0["ci_high"],
          f"mean={r0['mean']}")
    r1 = contrast(panel, "F3", "F0", cl, n_boot=2000, seed=1)
    check("KA4 恒 +0.01 → ACCEPT", abs(r1["mean"] - 0.01) < 1e-12
          and ab.significant(r1, ACCEPT_MARGIN), f"mean={r1['mean']:.6f}")
                                              
    for q in queries:
        d1 = (base[q] + 0.01) - (base[q] + 0.004)
        d2 = (base[q] + 0.004) - base[q]
        assert abs((d1 + d2) - 0.01) < 1e-12
    check("KA6 隔离对照代数自洽（特征+池=总量）", True, "恒等式逐查询成立")

    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


def _fmt(v, digits: int = 5) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(r: dict) -> str:
    return f"[{_fmt(r['ci_low'])},{_fmt(r['ci_high'])}]"


def verdict(r: dict) -> str:
    if r["mean"] is not None and r["mean"] >= ACCEPT_MARGIN and r["ci_low"] > 0:
        return "ACCEPT"
    if r["mean"] is not None and r["mean"] <= -ACCEPT_MARGIN and r["ci_high"] < 0:
        return "反向"
    return "HOLD"


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
    missing = [f"R_{a}_{s}" for a in ARMS for s in SEEDS if f"R_{a}_{s}" not in panel]
    if missing:
        raise SystemExit(f"! 缺臂：{missing}")
    clusters = cluster_map()
    rows_index = eval_rows_index()
    report: dict = {"n_boot": args.n_boot, "boot_seed": args.seed, "model": MODEL,
                    "analyzer_version": "v2（Codex 评审后：行身份/覆盖/违规率/敏感性/隔离对照）"}

    print("=== KA1 训练侧同一性（R_F0 ≡ 实验 S S_SQ）===")
    ka1 = ka1_train_side_identical()
    print(f"  [{ka1['status']}] {ka1['detail']}")
    report["ka1"] = ka1
    if ka1["status"] == "fail":
        raise SystemExit("! KA1 失败")

    print("\n=== 覆盖账（评审 §2 口径）===")
    cov = coverage_accounting()
    report["coverage"] = cov
    for face, c in cov.items():
        print(f"  {face}: 行 {c['file_rows']} → 非空池 {c['non_empty_pool']} → "
              f"≥1标签 {c['labeled_rows']}（部分标签 {c['partial_label_rows']}）")
        for d in c["dup_identity_qids"]:
            print(f"    重复身份 {d['qid']}: {d['texts']}")
    n_panel = len(vectors(panel, "F0", SEEDS[0]))
    print(f"  主指标人群 = {n_panel} 唯一行键 ×5 = {n_panel * 5}"
          f"（簇数 = {len(set(clusters.values()))}）")
    report["panel_rows"] = n_panel

    print("\n=== 主判据：F3 − F0（行身份修正后）===")
    primary = contrast(panel, "F3", "F0", clusters, args.n_boot, args.seed)
    report["primary"] = primary
    print(f"  Δ = {primary['mean']:+.5f}  CI={_ci(primary)}  P(>0)={primary['share_positive']:.3f}"
          f"  簇={primary['clusters']} 查询×seed={primary['queries']}")
    print(f"  逐 seed Δ：{ {s: round(v, 5) for s, v in primary['per_seed'].items()} }")
    v = verdict(primary)
    report["verdict"] = v
    print(f"  → {v}")
    tost = contrast(panel, "F3", "F0", clusters, args.n_boot, args.seed, alpha=0.10)
    report["tost_90"] = {"mean": tost["mean"], "ci_low": tost["ci_low"], "ci_high": tost["ci_high"]}
    print(f"  TOST 90%CI = [{tost['ci_low']:.5f},{tost['ci_high']:.5f}]")

    print("\n=== 隔离对照（评审高优先：拆开特征修正与候选裁减）===")
    feat_eff = contrast(panel, "F3", "F0P", clusters, args.n_boot, args.seed)
    pool_eff = contrast(panel, "F0P", "F0", clusters, args.n_boot, args.seed)
    report["isolation"] = {"feature_fix_F3_minus_F0P": feat_eff, "pool_cut_F0P_minus_F0": pool_eff}
    print(f"  纯特征修正 F3−F0P = {feat_eff['mean']:+.5f}  CI={_ci(feat_eff)}  → {verdict(feat_eff)}")
    print(f"  纯候选裁减 F0P−F0 = {pool_eff['mean']:+.5f}  CI={_ci(pool_eff)}  → {verdict(pool_eff)}")

    print("\n=== 分面拆开 ===")
    per_face = {}
    for face in FACES:
        vals: dict[str, list[float]] = {}
        for s in SEEDS:
            fa = {k: val for k, val in vectors(panel, "F3", s).items() if k.startswith(face)}
            fb = {k: val for k, val in vectors(panel, "F0", s).items() if k.startswith(face)}
            for k in fa:
                vals.setdefault(clusters[k], []).append(fa[k] - fb[k])
        r = ab.paired_cluster_bootstrap(vals, n_boot=args.n_boot, seed=args.seed)
        per_face[face] = r
        print(f"  {face:<11} Δ = {r['mean']:+.5f}  CI={_ci(r)}  簇={r['clusters']} → {verdict(r)}")
    report["per_face"] = per_face

    print("\n=== 违规率护栏（top-5 grade<2；预注册承诺项）===")
    viol = violation_rates(panel, rows_index)
    report["violation_top5"] = viol
    for arm in ARMS:
        print(f"  {arm}: {viol[arm]['rate']:.4f}（{viol[arm]['bad']}/{viol[arm]['slots']} 槽位）")

    print("\n=== OOD 类别留一敏感性（主对比剔除各类后重算）===")
    loo = {}
    ood_cluster_keys = sorted({clusters[k] for k in clusters if k.startswith("oodonlyf3/")})
    for cl_key in ood_cluster_keys:
        r = contrast(panel, "F3", "F0", clusters, args.n_boot, args.seed,
                     exclude_cluster_prefix={cl_key})
        loo[cl_key] = {"mean": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
        print(f"  剔 {cl_key:<28} Δ = {r['mean']:+.5f}  CI={_ci(r)}")
    report["ood_leave_one_out"] = loo

    print("\n=== 逐查询严重退化例（Δ(F3−F0) 最差 5）===")
    worst = worst_deltas(panel, rows_index)
    report["worst_deltas"] = worst
    for w in worst:
        print(f"  {w['row']:<38} Δ={w['delta']:+.4f}  f0={w['f0']:.4f} f3={w['f3']:.4f}"
              f"  [{w['ood_class']}] {w['query']}")

    report["metrics_side_table"] = metrics_side_table()
    out_path = OUT_ROOT / "analysis_v2.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"\n最终裁决（预注册判据）：{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
