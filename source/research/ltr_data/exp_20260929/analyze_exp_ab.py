                       
"""实验 A/B 的配对分析器：簇配对 bootstrap + 预注册判据 + 逐臂行账。

**先过已知答案自测再信汇总表**（`研究/AGENTS.md` §7）：`--self-test` 跑五条必过/必败样本——

1. 自比自（同一臂减自己）→ 逐查询差恒 0，CI 必须恰为 [0,0]（不是"近似 0"）；
2. 构造 +0.01 恒差 → CI 下界必须恰为 +0.01（排零）；
3. 同分布零均值噪声 → CI 必须**包含** 0（不显著），否则就是假阳性机器；
4. 簇内完全相关 → 簇级 CI 必须**比查询级 CI 宽**（否则"配对簇"是装饰）；
5. 同输入同 seed 跑两次 → 逐位相同（bootstrap 不许自带随机漂移）。

正式判据（预注册原文见报告 §1.4）：主指标 = 固定评测集（heldout5 + oodonly 两集查询并集、
逐查询等权）上的 per-query nDCG@10 宏平均，模型固定 `ranknet_mlp_full`，评测标签固定 `sem`；
接受线 = 交互量 ≥ +0.005 **且** 簇配对 95%CI 下界 > 0。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260929_ab"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"
OOD_EVALONLY = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_evalonly_xscore.jsonl"

N_BOOT = 10_000
BOOT_SEED = 20260929
PRIMARY_MODEL = "ranknet_mlp_full"
PRIMARY_SETS = ("heldout5", "oodonly")
SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)
ACCEPT_MARGIN = 0.005


                                                                       
def cluster_key(eval_set: str, row: dict) -> str:
    """簇键：heldout 用行自带 `cluster_id`；oodonly 无簇字段，用 `ood_class` 作家族代理（更保守）。

    集合名前缀保证两个评测集的同名簇不互相污染。
    """
    if eval_set == "heldout5":
        raw = row.get("cluster_id") or row.get("query_id")
    else:
        raw = row.get("ood_class") or row.get("query_id")
    return f"{eval_set}:{raw}"


def paired_cluster_bootstrap(values: "dict[str, list[float]]", n_boot: int = N_BOOT,
                             seed: int = BOOT_SEED, alpha: float = 0.05) -> dict:
    """簇配对 bootstrap：`values` = 簇键 → 该簇的逐查询配对差（每簇内先平均，再对抽到的簇平均）。

    返回 `mean`（点估计，等于全体查询的均值，与自助抽样无关）、`ci_low` / `ci_high`
    （alpha/2 与 1−alpha/2 分位；默认 alpha=0.05 即 95%CI，alpha=0.10 对应 TOST 的 90%CI）、
    `share_positive`（重采样均值 > 0 的比例）、`clusters` / `queries` 计数。
    """
    clusters = sorted(values)
    if not clusters:
        return {"mean": None, "ci_low": None, "ci_high": None, "share_positive": None,
                "clusters": 0, "queries": 0}
    per_cluster = np.asarray([float(np.mean(values[key])) for key in clusters], dtype=np.float64)
    weights = np.asarray([len(values[key]) for key in clusters], dtype=np.float64)
    point = float(np.average(per_cluster, weights=weights))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        draws[b] = float(np.average(per_cluster[pick], weights=weights[pick]))
    return {
        "mean": point,
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "share_positive": float((draws > 0).mean()),
        "clusters": len(clusters),
        "queries": int(weights.sum()),
    }


def significant(result: dict, margin: float) -> bool:
    """预注册接受线：点估计 ≥ margin 且 CI 下界 > 0。"""
    return bool(result["mean"] is not None and result["mean"] >= margin and result["ci_low"] > 0)


                                                                           
def _clusterise(deltas: "list[float]", clusters: "list[str]") -> "dict[str, list[float]]":
    out: "dict[str, list[float]]" = {}
    for key, value in zip(clusters, deltas):
        out.setdefault(key, []).append(value)
    return out


def _synthetic_cell_of(feat_effect: "dict[str, float]", d2_gain: "dict[str, float]",
                       n_query: int = 120, seed_noise: float = 0.03):
    """合成一个"已知答案"的臂面板：`ndcg = base(q) + feat_effect[tag] + (d1 ? d2_gain[tag] : 0) + 噪声`。

    **噪声只随 (查询, seed) 变**（不随臂/数据变）——现实中同 seed 同数据下臂间差本来就成对，
    这样构造出的交互量 I(tag vs f0) = d2_gain[tag] − d2_gain[f0]、主效应 = feat_effect[tag] −
    feat_effect[f0] 都是解析常数，判据聚合的正确性可以逐位核对，而不是"看起来像"。
    """
    rng = np.random.default_rng(11)
    queries = [f"heldout5:q{i}" for i in range(n_query)]
    base = {q: 0.9 + float(rng.normal(0.0, 0.05)) for q in queries}
    noise = {seed: {q: float(rng.normal(0.0, seed_noise)) for q in queries} for seed in SEEDS}

    def cell_of(tag: str, data: str, seed: int):
        return {q: base[q] + feat_effect[tag] + (d2_gain[tag] if data == "d1" else 0.0)
                + noise[seed][q] for q in queries}

    clusters = {q: f"heldout5:c{i // 4}" for i, q in enumerate(queries)}
    return cell_of, clusters


def self_test() -> int:
    failures: "list[str]" = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(7)
    n = 200
    clusters = [f"c{i // 4}" for i in range(n)]               

                      
    zeros = _clusterise([0.0] * n, clusters)
    r = paired_cluster_bootstrap(zeros, n_boot=2000)
    check("SELF 自比自 CI 恰为 [0,0]", r["ci_low"] == 0.0 == r["ci_high"] and r["mean"] == 0.0,
          f"mean={r['mean']} CI=[{r['ci_low']},{r['ci_high']}]")
    check("SELF 自比自不算显著", not significant(r, ACCEPT_MARGIN), "0 ≥ 0.005 不成立")

                                       
    flat = _clusterise([0.01] * n, clusters)
    r = paired_cluster_bootstrap(flat, n_boot=2000)
    check("KNOWN +0.01 恒差 CI 下界排零且恰为 +0.01",
          abs(r["mean"] - 0.01) < 1e-12 and abs(r["ci_low"] - 0.01) < 1e-12
          and abs(r["ci_high"] - 0.01) < 1e-12,
          f"mean={r['mean']:.6f} CI=[{r['ci_low']:.6f},{r['ci_high']:.6f}]")
    check("KNOWN +0.01 ≥ 0.005 且排零 → 判显著", significant(r, ACCEPT_MARGIN), "接受线成立")

                                         
    noise_raw = rng.normal(0.0, 0.02, size=n)
    noise_centred = list(noise_raw - noise_raw.mean())                   
    r = paired_cluster_bootstrap(_clusterise(noise_centred, clusters), n_boot=4000)
    check("NULL 零效应噪声 CI 包含 0（不显著）",
          abs(r["mean"]) < 1e-12 and r["ci_low"] < 0.0 < r["ci_high"] and not significant(r, ACCEPT_MARGIN),
          f"mean={r['mean']:.2e} CI=[{r['ci_low']:.5f},{r['ci_high']:.5f}]")
    check("NULL 未归零的同分布噪声样本均值落在 ±3SE 内（机器没放大效应）",
          abs(noise_raw.mean()) < 3 * noise_raw.std(ddof=1) / math.sqrt(n),
          f"样本均值={noise_raw.mean():+.5f} SE={noise_raw.std(ddof=1) / math.sqrt(n):.5f}")

                                        
    small = _clusterise([0.004] * n, clusters)
    r = paired_cluster_bootstrap(small, n_boot=1000)
    check("MARGIN +0.004 排零但低于 0.005 → 不判显著",
          r["ci_low"] > 0 and not significant(r, ACCEPT_MARGIN), f"mean={r['mean']:.5f}")

                               
    shared = list(rng.normal(0.0, 0.02, size=len(set(clusters))))
    mapped = {c: v for c, v in zip(sorted(set(clusters)), shared)}
    corr = _clusterise([mapped[c] for c in clusters], clusters)
    wide = paired_cluster_bootstrap(corr, n_boot=4000)
    narrow = paired_cluster_bootstrap(_clusterise([mapped[c] for c in clusters],
                                                  [f"q{i}" for i in range(n)]), n_boot=4000)
    wide_w = wide["ci_high"] - wide["ci_low"]
    narrow_w = narrow["ci_high"] - narrow["ci_low"]
    check("CLUSTER 簇内相关时簇级 CI 明显更宽", wide_w > 1.5 * narrow_w,
          f"簇级宽={wide_w:.5f} vs 查询级宽={narrow_w:.5f}（比 {wide_w / narrow_w:.2f}×）")

                         
    sample = _clusterise(noise_centred, clusters)
    a = paired_cluster_bootstrap(sample, n_boot=1000, seed=BOOT_SEED)
    b = paired_cluster_bootstrap(sample, n_boot=1000, seed=BOOT_SEED)
    check("DETERMINISM 同 seed 两次跑逐位相同", a == b, "bootstrap 可复现")

                                        
    feat = {"f0": 0.0, "f1": 0.02, "f2": 0.03}
    d2 = {"f0": 0.0, "f1": 0.01, "f2": 0.012}
    cell_of, clusters = _synthetic_cell_of(feat, d2)
    for tag, expected_i in (("f1", 0.01), ("f2", 0.012)):
        inter = interaction_over_seeds(cell_of, tag, "f0", clusters, n_boot=3000)
        check(f"E2E I(F{tag[1]}) == 构造常数 {expected_i}",
              abs(inter["mean"] - expected_i) < 1e-9 and inter["ci_low"] > 0,
              f"mean={inter['mean']:.6f} CI={_ci(inter)}")
                                                
    main_cell, main_clusters = _synthetic_cell_of({"f0": 0.0, "f1": 0.02, "f2": 0.03},
                                                  {"f0": 0.0, "f1": 0.0, "f2": 0.0})
    for tag in ("f1", "f2"):
        maineff = main_effect_over_seeds(main_cell, tag, "f0", main_clusters, n_boot=3000)
        check(f"E2E 主效应 F{tag[1]}−F0 == 构造常数 {feat[tag]}",
              abs(maineff["mean"] - feat[tag]) < 1e-9,
              f"mean={maineff['mean']:.6f}")
                                                              
    flat_cell, flat_clusters = _synthetic_cell_of({"f0": 0.0, "f1": 0.02}, {"f0": 0.01, "f1": 0.01})
    inter = interaction_over_seeds(flat_cell, "f1", "f0", flat_clusters, n_boot=2000)
    check("E2E D2 增益相同 → I 恰为 0 且判 HOLD",
          abs(inter["mean"]) < 1e-12 and inter["ci_high"] <= 0.0
          and not significant(inter, ACCEPT_MARGIN),
          f"mean={inter['mean']:.2e} CI={_ci(inter)}")
                                                      
    small_cell, small_clusters = _synthetic_cell_of({"f0": 0.0, "f1": 0.0}, {"f0": 0.0, "f1": 0.004})
    inter = interaction_over_seeds(small_cell, "f1", "f0", small_clusters, n_boot=2000)
    check("E2E I=+0.004 排零但低于 0.005 → HOLD",
          abs(inter["mean"] - 0.004) < 1e-12 and inter["ci_low"] > 0
          and not significant(inter, ACCEPT_MARGIN),
          f"mean={inter['mean']:.6f} CI={_ci(inter)}")

    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


                                                                       
def read_jsonl(path: Path) -> "list[dict]":
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_panel(out_root: Path) -> "tuple[dict, dict]":
    """装载全部臂：`{arm_id: meta}` 与 `{arm_id: {eval_set: {query_id: {"ndcg":..,"uids":[..]}}}}`。"""
    index_path = out_root / "arms_index.json"
    if not index_path.is_file():
        raise SystemExit(f"! 没有臂账：{index_path}（先跑 run_arms.py）")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    panel: "dict[str, dict]" = {}
    for arm_id, meta in index.items():
        if meta.get("returncode") != 0:
            continue
        per_query = out_root / "arms" / arm_id / "per_query.jsonl"
        if not per_query.is_file():
            continue
        by_set: "dict[str, dict]" = {}
        for row in read_jsonl(per_query):
            if PRIMARY_MODEL not in row["ndcg10"]:
                continue
            by_set.setdefault(row["eval_set"], {})[row["query_id"]] = {
                "ndcg": float(row["ndcg10"][PRIMARY_MODEL]),
                "hist": float(row["ndcg10"].get("hist_gbdt_full", float("nan"))),
                "lex": float(row["ndcg10"].get("lex", float("nan"))),
                "top10": row["top10_uids"].get(PRIMARY_MODEL) or [],
                "top10_by_model": {key: value for key, value in row["top10_uids"].items()},
            }
        panel[arm_id] = by_set
    return index, panel


def arms_of(panel: dict, experiment: str) -> "dict[str, dict]":
    """`{arm_id: {key: value}}`——只留跑成功的臂。"""
    prefix = f"{experiment}_"
    out = {}
    for arm_id, by_set in panel.items():
        if arm_id.startswith(prefix):
            out[arm_id] = by_set
    return out


def arm_mean(by_set: dict, eval_sets: "tuple[str, ...]" = PRIMARY_SETS) -> float | None:
    """两集查询并集上的宏平均（逐查询等权）。"""
    values = [entry["ndcg"] for name in eval_sets for entry in by_set.get(name, {}).values()]
    return float(np.mean(values)) if values else None


def query_vectors(by_set: dict, eval_sets: "tuple[str, ...]" = PRIMARY_SETS) -> "dict[str, float]":
    """`{集合:查询}` → nDCG@10（缺该集的臂这里会少键，调用方必须先断言键集一致）。"""
    return {f"{name}/{qid}": entry["ndcg"] for name in eval_sets
            for qid, entry in by_set.get(name, {}).items()}


def cluster_map(experiment_rows: "dict[str, dict]") -> "dict[str, str]":
    """评测集里的查询 → 簇键（从数据文件读，不依赖臂账）。"""
    out: "dict[str, str]" = {}
    for name, path in (("heldout5", HELDOUT_GOLD), ("oodonly", OOD_EVALONLY)):
        for row in read_jsonl(path):
            out[f"{name}/{row['query_id']}"] = cluster_key(name, row)
    return out


def paired_difference(vectors_a: "dict[str, float]", vectors_b: "dict[str, float]",
                      clusters: "dict[str, str]", n_boot: int = N_BOOT) -> dict:
    """A − B 的簇配对 CI（A/B 的键集必须一致——不一致就是臂之间评测集不同，直接炸）。"""
    if set(vectors_a) != set(vectors_b):
        raise AssertionError(
            f"两臂的评测查询集不同：差 {len(set(vectors_a) ^ set(vectors_b))} 条"
            f"（{sorted(set(vectors_a) ^ set(vectors_b))[:5]}）"
        )
    values: "dict[str, list[float]]" = {}
    for key, value in vectors_a.items():
        values.setdefault(clusters.get(key, f"unknown:{key}"), []).append(value - vectors_b[key])
    return paired_cluster_bootstrap(values, n_boot=n_boot)


                                                                       
def interaction_over_seeds(cell_of, new_tag: str, base_tag: str, clusters: "dict[str, str]",
                           seeds=SEEDS, n_boot: int = N_BOOT) -> dict:
    """跨 seed 的交互量：先把同一查询在 5 个 seed 上的交互差平均，再按簇自助。"""
    values: "dict[str, list[float]]" = {}
    for seed in seeds:
        d1_new, d0_new = cell_of(new_tag, "d1", seed), cell_of(new_tag, "d0", seed)
        d1_base, d0_base = cell_of(base_tag, "d1", seed), cell_of(base_tag, "d0", seed)
        if d1_new is None or d0_new is None or d1_base is None or d0_base is None:
            raise SystemExit(f"! 交互量缺臂：new={new_tag} base={base_tag} seed={seed}")
        if not (set(d1_new) == set(d0_new) == set(d1_base) == set(d0_base)):
            raise AssertionError("四格评测查询集不一致")
        for key in d1_new:
            delta = (d1_new[key] - d0_new[key]) - (d1_base[key] - d0_base[key])
            values.setdefault(clusters.get(key, f"unknown:{key}"), []).append(delta)
    return paired_cluster_bootstrap(values, n_boot=n_boot)


def interaction_per_seed(cell_of, new_tag: str, base_tag: str, seeds=SEEDS) -> "dict[int, float]":
    """逐 seed 的交互量（点估计，不自助）：`I_s = mean_q[(new,d1) − (new,d0) − (base,d1) + (base,d0)]`。

    用来展示种子散布——CI 之外还要能看见"这 5 个 seed 是不是同一个方向"。
    """
    out: "dict[int, float]" = {}
    for seed in seeds:
        d1_new, d0_new = cell_of(new_tag, "d1", seed), cell_of(new_tag, "d0", seed)
        d1_base, d0_base = cell_of(base_tag, "d1", seed), cell_of(base_tag, "d0", seed)
        if None in (d1_new, d0_new, d1_base, d0_base):
            raise SystemExit(f"! 逐 seed 交互缺臂：new={new_tag} base={base_tag} seed={seed}")
        out[seed] = float(np.mean([(d1_new[k] - d0_new[k]) - (d1_base[k] - d0_base[k]) for k in d1_new]))
    return out


def main_effect_over_seeds(cell_of, new_tag: str, base_tag: str, clusters: "dict[str, str]",
                           seeds=SEEDS, n_boot: int = N_BOOT) -> dict:
    """主效应 = 两个 data 臂的平均：逐 (seed, data, 查询) 取 `new−base` 后按簇自助。

    注意这里是"先把两个 data 臂的差都收进同一袋再求均值"——不要再除 2（除 2 会把主效应算成一半）。
    """
    values: "dict[str, list[float]]" = {}
    for seed in seeds:
        for data in ("d0", "d1"):
            a, b = cell_of(new_tag, data, seed), cell_of(base_tag, data, seed)
            if a is None or b is None:
                raise SystemExit(f"! 主效应缺臂：new={new_tag} base={base_tag} seed={seed} data={data}")
            for key in a:
                values.setdefault(clusters.get(key, f"unknown:{key}"), []).append(a[key] - b[key])
    return paired_cluster_bootstrap(values, n_boot=n_boot)


                                                                         
def violation_index() -> "dict[str, dict[str, int | None]]":
    """`{集合:查询:uid → grade}`——违规口径用**既有 grade 标签**（grade<2 = 至少违反一项 intended 约束）。

    不另写判官：grade 是采集期按冻结基准外部裁判对 intended 约束（含 excluded）算出来的既有字段；
    `None` = 该查询/候选没有约束口径（无约束类查询），违规率分母只算有 grade 的候选。
    """
    out: "dict[str, dict]" = {}
    for name, path in (("heldout5", HELDOUT_GOLD), ("oodonly", OOD_EVALONLY)):
        per_query = {}
        for row in read_jsonl(path):
            per_query[row["query_id"]] = {c["uid"]: c.get("grade") for c in row["candidates"]}
        out[name] = per_query
    return out


def violation_rate(by_set: dict, grades: dict, top_k: int = 5,
                   model_key: str = PRIMARY_MODEL) -> "tuple[float, int]":
    """模型 top-k 里 grade<2 的比例（分母只算有 grade 的槽位），返回 (rate, n_slots)。"""
    bad = total = 0
    for name in PRIMARY_SETS:
        for qid, entry in by_set.get(name, {}).items():
            table = grades.get(name, {}).get(qid) or {}
            for uid in (entry.get("top10_by_model", {}).get(model_key) or entry["top10"])[:top_k]:
                grade = table.get(uid)
                if grade is None:
                    continue
                total += 1
                bad += int(float(grade) < 2)
    return (bad / total if total else float("nan")), total


                                                                      
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _ci(result: dict, digits: int = 4) -> str:
    return f"[{_fmt(result['ci_low'], digits)},{_fmt(result['ci_high'], digits)}]"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    index, panel = load_panel(args.out_root)
    clusters = cluster_map(panel)
    grades = violation_index()
    report: dict = {"n_boot": args.n_boot, "boot_seed": args.seed, "accept_margin": ACCEPT_MARGIN,
                    "seeds": list(SEEDS), "eval_sets": list(PRIMARY_SETS), "arms": {}, "analysis": {}}

    print("=== 逐臂行账（主模型 ranknet_mlp_full / 评测标签 sem）===")
    print(f"{'arm':<22}{'heldout5':>10}{'oodonly':>10}{'pooled':>10}{'hist(持)':>10}{'lex(持)':>10}"
          f"{'违/持':>8}{'违/ood':>8}{'违/lex':>8}{'pairs':>8}")
    print("-" * 112)
    for arm_id in sorted(panel):
        by_set = panel[arm_id]
        pooled = arm_mean(by_set)
        held = arm_mean(by_set, ("heldout5",))
        ood = arm_mean(by_set, ("oodonly",))
        lex_held = float(np.mean([e["lex"] for e in by_set.get("heldout5", {}).values()])) \
            if by_set.get("heldout5") else float("nan")
        hist_held = float(np.mean([e["hist"] for e in by_set.get("heldout5", {}).values()])) \
            if by_set.get("heldout5") else float("nan")
                                                   
        v_held_only, n_held = violation_rate({k: v for k, v in by_set.items() if k in ("heldout5",)}, grades)
        ood_only = {k: v for k, v in by_set.items() if k in ("oodonly",)}
        v_ood_only, _ = violation_rate(ood_only, grades)
        v_lex, _ = violation_rate(by_set, grades, model_key="lex")
        metrics_path = args.out_root / "arms" / arm_id / "metrics.json"
        pairs = None
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            pairs = payload["results"]["models"]["ranknet_mlp"]["full"].get("pairs")
        report["arms"][arm_id] = {"heldout5": held, "oodonly": ood, "pooled": pooled,
                                  "hist_heldout5": hist_held, "lex_heldout5": lex_held,
                                  "violation_top5": {"heldout5": v_held_only,
                                                     "oodonly": v_ood_only, "lex_baseline": v_lex},
                                  "violation_slots": n_held, "pairs": pairs}
        print(f"{arm_id:<22}{_fmt(held):>10}{_fmt(ood):>10}{_fmt(pooled):>10}{_fmt(hist_held):>10}"
              f"{_fmt(lex_held):>10}{_fmt(v_held_only, 3):>8}{_fmt(v_ood_only, 3):>8}"
              f"{_fmt(v_lex, 3):>8}{str(pairs):>8}")

    a_arms = {arm_id for arm_id in panel if arm_id.startswith("A_")}
    b_arms = {arm_id for arm_id in panel if arm_id.startswith("B_")}

    if a_arms:
        def cell_of(tag: str, data: str, seed: int):
            arm_id = f"A_{tag}_{data}_{seed}"
            return query_vectors(panel[arm_id]) if arm_id in panel else None

                                              
        cells = {}
        for tag in ("f0", "f1", "f2"):
            for data in ("d0", "d1"):
                means = [arm_mean(panel[f"A_{tag}_{data}_{s}"]) for s in SEEDS
                         if f"A_{tag}_{data}_{s}" in panel]
                cells[(tag, data)] = float(np.mean(means)) if means else None
        deltas = {tag: (cells[(tag, "d1")] - cells[(tag, "d0")])
                  for tag in ("f0", "f1", "f2") if cells[(tag, "d0")] is not None}

        print("\n=== 实验 A：格均宏平均（5 seed 平均）与 D1−D0 ===")
        for (tag, data), value in cells.items():
            print(f"  F{tag[1]}×{data.upper()}: {_fmt(value, 5)}")
        for tag, value in deltas.items():
            print(f"  D1−D0 @ F{tag[1]}: {value:+.5f}")

        print(f"\n=== 实验 A：接受线 I（{PRIMARY_SETS[0]}+{PRIMARY_SETS[1]} 并集，"
              f"簇配对 {args.n_boot} 次, seed {args.seed}）===")
        analysis_a = {}
        for tag in ("f1", "f2"):
            if tag not in deltas:
                continue
            inter = interaction_over_seeds(cell_of, tag, "f0", clusters, n_boot=args.n_boot)
            maineff = main_effect_over_seeds(cell_of, tag, "f0", clusters, n_boot=args.n_boot)
            per_seed = interaction_per_seed(cell_of, tag, "f0")
            accepted = significant(inter, ACCEPT_MARGIN)
            analysis_a[f"F{tag[1]}"] = {"interaction": inter, "main_effect": maineff,
                                        "interaction_per_seed": per_seed,
                                        "main_effect_pass": significant(maineff, ACCEPT_MARGIN),
                                        "accepted": accepted}
            print(f"  I(F{tag[1]}, F0) = {inter['mean']:+.5f}  CI={_ci(inter)}  "
                  f"P(>0)={inter['share_positive']:.3f}  簇={inter['clusters']} 查询×seed={inter['queries']}"
                  f"  → {'ACCEPT' if accepted else 'HOLD'}")
            print(f"    逐 seed I：{ {s: round(v, 5) for s, v in per_seed.items()} }")
            print(f"   主效应 F{tag[1]}−F0 = {maineff['mean']:+.5f} CI={_ci(maineff)} "
                  f"→ {'显著' if significant(maineff, ACCEPT_MARGIN) else '不显著'}")
        report["analysis"]["A"] = analysis_a

    if b_arms:
        def bcell(label: str, data: str, seed: int):
            for tag, arm_id in ((label, f"B_{label}_{data}_{seed}"),):
                if arm_id in panel:
                    return query_vectors(panel[arm_id])
            return None

        print("\n=== 实验 B：训练标签臂（特征固定 F1，评测固定 sem）===")
        print(f"{'label×data':<18}{'mean':>10}")
        bmeans = {}
        for label in ("det", "mix", "sem"):
            for data in ("d0", "d1"):
                means = [arm_mean(panel[f"B_{label}_{data}_{s}"]) for s in SEEDS
                         if f"B_{label}_{data}_{s}" in panel]
                bmeans[(label, data)] = float(np.mean(means)) if means else None
                print(f"{label+'×'+data.upper():<18}{_fmt(bmeans[(label, data)], 5):>10}")

        analysis_b = {}
        for label in ("det", "sem"):
            if None in (bmeans[(label, "d0")], bmeans[("mix", "d0")]):
                continue
            delta = float(np.mean([bmeans[(label, d)] - bmeans[("mix", d)] for d in ("d0", "d1")]))
                                                                     
            values: "dict[str, list[float]]" = {}
            for seed in SEEDS:
                for data in ("d0", "d1"):
                    a = bcell(label, data, seed)
                    b = bcell("mix", data, seed)
                    if a is None or b is None:
                        raise SystemExit(f"! 标签臂缺：{label}/{data}/{seed}")
                    for key in a:
                        values.setdefault(clusters.get(key, f"unknown:{key}"), []).append(a[key] - b[key])
            result = paired_cluster_bootstrap(values, n_boot=args.n_boot, seed=args.seed)
            inter = interaction_over_seeds(bcell, label, "mix", clusters, n_boot=args.n_boot)
            per_seed = interaction_per_seed(bcell, label, "mix")
            accepted = significant(result, ACCEPT_MARGIN)
            analysis_b[label] = {"delta_vs_mix": result, "delta_point_mean": delta,
                                 "data_interaction": inter, "data_interaction_per_seed": per_seed,
                                 "accepted": accepted,
                                 "cell_means": {f"{lab}×{d}": bmeans[(lab, d)]
                                                for lab in (label, "mix") for d in ("d0", "d1")}}
            print(f"  Δ({label}−mix) = {result['mean']:+.5f} CI={_ci(result)} "
                  f"P(>0)={result['share_positive']:.3f} → {'ACCEPT' if accepted else 'HOLD'}")
            print(f"   D1−D0 交互（{label}−mix） = {inter['mean']:+.5f} CI={_ci(inter)}")
            print(f"   逐 seed 交互：{ {s: round(v, 5) for s, v in per_seed.items()} }")
        report["analysis"]["B"] = analysis_b

    out_path = args.out_root / "analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
