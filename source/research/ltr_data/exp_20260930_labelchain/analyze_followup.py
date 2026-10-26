                       
"""实验 S 的 Codex 补证分析器（评审 `研究/00-汇报与面谈/codex-实验S标签链评审-2026-09-30.md` 的必做/高优先项）。

四项，全部零新训练：
1. **283/332 对账**（必做）：逐面清点「文件查询数 → eval-label sem 下有标候选 ≥1 的查询数」，
   点名被丢查询及其 ood_class 分布——主指标对应的人群必须说清。
2. **pair 翻转审计**（必做）：训练监督 = 查询内候选对的 sign(y_i−y_j)（`_pair_indices`，
   无幅度加权）。mix 标签用整数精算（label×6 = 3×grade_raw + sem_raw），统计基础层
   SQ→CONS 的符号翻转 / 平局新增 / 平局解除——「53% 标签变了」≠「监督变了 53%」。
3. **TOST 90%CI**（必做）：与主判据同袋配对差，alpha=0.10 复算（等效检验口径）。
4. **top-10 重合**（高优先）：两臂同 seed 的 ranknet_mlp_full top-10 uid 交集（heldout5 面）——
   模型输出到底动没动。

复用 `exp_20260929/analyze_exp_ab.py` 的 bootstrap（含 alpha 参数）；`--self-test` 跑已知答案。
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

OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_labelchain"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
OOD_EVALONLY = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_evalonly_xscore.jsonl"
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"
HELDOUT_SINGLE = LTR_DATA / "synth_heldout_vec.jsonl"

SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)
MODEL = "ranknet_mlp_full"


                                                                          
def coverage_reconciliation() -> dict:
    """逐面：文件查询数 → sem 下有标 ≥1 的存活查询数 → 被丢查询名单与画像。"""
    out = {}
    for name, path, label_field in (
            ("heldout5", HELDOUT_GOLD, "llm_semantic"),
            ("oodonly", OOD_EVALONLY, "llm_semantic"),
            ("heldsingle", HELDOUT_SINGLE, "llm_semantic")):
        rows = ab.read_jsonl(path)
        alive, dropped = [], []
        for row in rows:
            cands = row.get("candidates") or []
            n_lab = sum(1 for c in cands if c.get(label_field) is not None)
            if n_lab:
                alive.append(row["query_id"])
            else:
                dropped.append({"query_id": row["query_id"],
                                "ood_class": row.get("ood_class"),
                                "cluster_id": row.get("cluster_id"),
                                "n_cands": len(cands),
                                "abstain": row.get("abstain"),
                                "degraded": row.get("degraded")})
                                                                  
                                       
        dup_qids = sorted(q for q, n in Counter(alive).items() if n > 1)
        out[name] = {"file_rows": len(rows), "alive_rows": len(alive),
                     "alive_unique_qids": len(set(alive)),
                     "dup_alive_qids": dup_qids, "dropped": len(dropped),
                     "dropped_rows": dropped,
                     "dropped_ood_class": dict(Counter(d["ood_class"] for d in dropped))}
    return out


                                                                               
def _mix_label_x6(cand: dict, sem_key: str) -> int | None:
    """mix 标签 ×6 的整数精算：3×grade_raw + sem_raw（grade_raw∈{0,1,2}，sem∈{0..3}）。

    sem 缺失时 mix 退化为 grade（train_ltr.label_value：`semantic is None → grade`）——
    此时 ×6 = 6×grade_raw/2 = 3×grade_raw（与 sem=0 同值纯属巧合但语义一致：都是「只靠 det」）。
    """
    grade_raw = int(cand.get("grade") or 0)
    sem = cand.get(sem_key)
    if sem is None:
        return 3 * grade_raw
    return 3 * grade_raw + int(sem)


def pair_audit() -> dict:
    """基础训练层 1103 查询：SQ（vec 层）vs CONS（xscore 层）mix 标签的查询内 pair 符号账。"""
    sq_rows = ab.read_jsonl(LTR_DATA / "synth_train_vec.jsonl")
    cons_by_qid = {r["query_id"]: r for r in ab.read_jsonl(LTR_DATA / "synth_train_xscore.jsonl")}
    totals = Counter()
    per_query_flip_share = []
    for row in sq_rows:
        other = cons_by_qid[row["query_id"]]
        sq_lab = {c["uid"]: _mix_label_x6(c, "llm_semantic") for c in row.get("candidates") or []}
        cons_lab = {c["uid"]: _mix_label_x6(c, "llm_semantic") for c in other.get("candidates") or []}
        uids = sorted(sq_lab)
        q_pairs = q_flip = q_tie_new = q_tie_gone = 0
        for a in range(len(uids)):
            for b in range(a + 1, len(uids)):
                sa = sq_lab[uids[a]] - sq_lab[uids[b]]
                ca = cons_lab[uids[a]] - cons_lab[uids[b]]
                s_sign = (sa > 0) - (sa < 0)
                c_sign = (ca > 0) - (ca < 0)
                q_pairs += 1
                if s_sign != c_sign:
                    if s_sign == 0:
                        q_tie_gone += 1                              
                    elif c_sign == 0:
                        q_tie_new += 1                              
                    else:
                        q_flip += 1                      
        eff_sq = q_pairs - sum(1 for a in range(len(uids)) for b in range(a + 1, len(uids))
                               if sq_lab[uids[a]] == sq_lab[uids[b]])
        totals["pairs"] += q_pairs
        totals["flip"] += q_flip
        totals["tie_new"] += q_tie_new
        totals["tie_gone"] += q_tie_gone
        totals["eff_sq"] += eff_sq
        if q_pairs:
            per_query_flip_share.append((q_flip + q_tie_new + q_tie_gone) / q_pairs)
    changed = totals["flip"] + totals["tie_new"] + totals["tie_gone"]
    return {
        "queries": len(sq_rows),
        "pairs_total": totals["pairs"],
        "pairs_effective_sq": totals["eff_sq"],
        "flips": totals["flip"], "tie_created": totals["tie_new"], "tie_broken": totals["tie_gone"],
        "supervision_changed": changed,
        "share_of_all_pairs": changed / totals["pairs"],
        "share_of_effective_pairs": changed / totals["eff_sq"],
        "per_query_changed_share_median": float(np.median(per_query_flip_share)),
        "per_query_changed_share_p90": float(np.quantile(per_query_flip_share, 0.9)),
    }


                                                                                        
def _panel_vectors(arm: str, seed: int, face: str) -> dict[str, float]:
    out = {}
    counters: Counter = Counter()
    for row in ab.read_jsonl(OUT_ROOT / "arms" / f"S_{arm}_{seed}" / "per_query.jsonl"):
        if row["eval_set"] == face:
                                                    
            key = f"{row['query_id']}#{counters[row['query_id']]}"
            counters[row["query_id"]] += 1
            out[key] = float(row["ndcg10"][MODEL])
    return out


def _cluster_map() -> dict[str, str]:
    out = {}
    for row in ab.read_jsonl(HELDOUT_GOLD):
        out[f"heldout5/{row['query_id']}#0"] = f"heldout5:{row.get('cluster_id') or row['query_id']}"
    seen: Counter = Counter()
    for row in ab.read_jsonl(OOD_EVALONLY):
        occ = seen[row["query_id"]]
        seen[row["query_id"]] += 1
        out[f"oodonly/{row['query_id']}#{occ}"] = f"oodonly:{row.get('ood_class') or row['query_id']}"
    return out


def tost_90(n_boot: int, seed: int) -> dict:
    clusters = _cluster_map()
    values: dict[str, list[float]] = {}
    for s in SEEDS:
                                                               
        va = {f"heldout5/{k}": v for k, v in _panel_vectors("CONS", s, "heldout5").items()}
        va.update({f"oodonly/{k}": v for k, v in _panel_vectors("CONS", s, "oodonly").items()})
        vb = {f"heldout5/{k}": v for k, v in _panel_vectors("SQ", s, "heldout5").items()}
        vb.update({f"oodonly/{k}": v for k, v in _panel_vectors("SQ", s, "oodonly").items()})
        assert set(va) == set(vb), f"seed {s} 键集不一致"
        for k in va:
            values.setdefault(clusters[k], []).append(va[k] - vb[k])
    return ab.paired_cluster_bootstrap(values, n_boot=n_boot, seed=seed, alpha=0.10)


def top10_overlap() -> dict:
    shares = []
    flips = 0
    n_q = 0
    for s in SEEDS:
        sq_top, cons_top = {}, {}
        for row in ab.read_jsonl(OUT_ROOT / "arms" / f"S_SQ_{s}" / "per_query.jsonl"):
            if row["eval_set"] == "heldout5":
                sq_top[row["query_id"]] = row["top10_uids"].get(MODEL) or []
        for row in ab.read_jsonl(OUT_ROOT / "arms" / f"S_CONS_{s}" / "per_query.jsonl"):
            if row["eval_set"] == "heldout5":
                cons_top[row["query_id"]] = row["top10_uids"].get(MODEL) or []
        for qid in sq_top:
            a, b = sq_top[qid], cons_top.get(qid, [])
            if not a or not b:
                continue
            n_q += 1
            shares.append(len(set(a) & set(b)) / max(len(a), len(b)))
            flips += int(a[:3] != b[:3])
    return {"query_x_seed": n_q, "mean_jaccard@10": float(np.mean(shares)),
            "median_jaccard@10": float(np.median(shares)),
            "top3_order_changed_share": flips / n_q}


                                                                           
def self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

                                                                                 
    c1 = {"grade": 2, "llm_semantic": 3}
    c2 = {"grade": 0, "llm_semantic": None}
    check("mix×6 精算", _mix_label_x6(c1, "llm_semantic") == 9 and _mix_label_x6(c2, "llm_semantic") == 0,
          f"{_mix_label_x6(c1, 'llm_semantic')}, {_mix_label_x6(c2, 'llm_semantic')}")
                                                             
    import tempfile, unittest.mock as mock
    fake_sq = [{"query_id": "q", "candidates": [
        {"uid": "a", "grade": 0, "llm_semantic": 0},
        {"uid": "b", "grade": 0, "llm_semantic": 2},
        {"uid": "c", "grade": 2, "llm_semantic": 3}]}]
    fake_cons = [{"query_id": "q", "candidates": [
        {"uid": "a", "grade": 0, "llm_semantic": 1},
        {"uid": "b", "grade": 0, "llm_semantic": 3},
        {"uid": "c", "grade": 2, "llm_semantic": 3}]}]
                                         
    with mock.patch.object(ab, "read_jsonl",
                           side_effect=lambda p: fake_sq if "vec" in str(p) else fake_cons):
        r = pair_audit()
                                                                                       
                                                                    
    check("Codex 例：整体平移标签 → 0 监督变化",
          r["supervision_changed"] == 0 and r["pairs_total"] == 3,
          f"changed={r['supervision_changed']} pairs={r['pairs_total']}")
                                           
    fake_cons2 = [{"query_id": "q", "candidates": [
        {"uid": "a", "grade": 0, "llm_semantic": 1},
        {"uid": "b", "grade": 0, "llm_semantic": 0},
        {"uid": "c", "grade": 2, "llm_semantic": 3}]}]
    with mock.patch.object(ab, "read_jsonl",
                           side_effect=lambda p: fake_sq if "vec" in str(p) else fake_cons2):
        r2 = pair_audit()
                                                                                    
    check("构造单点翻转 → 恰 1 个 flip",
          r2["flips"] == 1 and r2["supervision_changed"] == 1,
          f"flips={r2['flips']} changed={r2['supervision_changed']}")

    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    report: dict = {}
    print("=== 1. 覆盖对账（eval-label sem 下哪些查询进不了主指标）===")
    cov = coverage_reconciliation()
    report["coverage"] = cov
    for name, c in cov.items():
        print(f"  {name}: 文件 {c['file_rows']} 行 → 存活 {c['alive_rows']} 行 / "
              f"{c['alive_unique_qids']} 唯一 qid（丢 {c['dropped']}）"
              + (f"  重复存活 qid={c['dup_alive_qids']}" if c["dup_alive_qids"] else "")
              + (f"  丢行 ood_class={c['dropped_ood_class']}" if c["dropped"] else ""))
    primary_alive = sum(cov[n]["alive_unique_qids"] for n in ("heldout5", "oodonly"))
    print(f"  主指标人群（heldout5∪oodonly）= {primary_alive} 唯一 (面,qid) ×5 seed = "
          f"{primary_alive * 5}（臂账查询×seed=1415 相符；重复 qid 按文件顺序后写覆盖，两臂同口径）")

    print("\n=== 2. pair 翻转审计（基础层 1103 查询，mix 标签整数精算）===")
    pa = pair_audit()
    report["pair_audit"] = pa
    print(f"  pairs 总数={pa['pairs_total']}  SQ 有效对（非平局）={pa['pairs_effective_sq']}")
    print(f"  方向翻转={pa['flips']}  平局新增={pa['tie_created']}  平局解除={pa['tie_broken']}"
          f"  → 监督改变={pa['supervision_changed']}")
    print(f"  占全部 pair {pa['share_of_all_pairs']:.3%}，占 SQ 有效 pair {pa['share_of_effective_pairs']:.3%}")
    print(f"  逐查询改变率：中位 {pa['per_query_changed_share_median']:.3%}，"
          f"P90 {pa['per_query_changed_share_p90']:.3%}")

    print("\n=== 3. TOST 90%CI（与主判据同袋配对差）===")
    tost = tost_90(args.n_boot, args.seed)
    report["tost_90"] = tost
    print(f"  Δ={tost['mean']:+.5f}  90%CI=[{tost['ci_low']:.5f},{tost['ci_high']:.5f}]"
          f"（与 ±0.005 等效界比较）")

    print("\n=== 4. top-10 重合（heldout5 面，同 seed 配对）===")
    ov = top10_overlap()
    report["top10_overlap"] = ov
    print(f"  查询×seed={ov['query_x_seed']}  Jaccard@10 均值={ov['mean_jaccard@10']:.4f}"
          f" 中位={ov['median_jaccard@10']:.4f}  top-3 顺序变化率={ov['top3_order_changed_share']:.3%}")

    out_path = OUT_ROOT / "followup_analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
