                       
"""F3 特征构建器（队列② R2）：回放**生产存活池**，用 R1 共享函数产出与生产推理逐值相同的 9 维特征。

为什么（实验 A 报告 §7.1 的下一棒）：离线训练行是 A/B 两轨**并集池**（track A=生产忠实检索、
track B=rescue 强行救回，dedup by uid 后 A 在前），而生产 `_ltr_fused_order` 只看单次检索的
**存活池**。F1 手工对齐了 pos/matched/raw 三处，但 lex/vec 的池级口径仍是并集池：
lex_max 可能被 B 轨（rescue intent 下打的分）抬高、B 轨候选的 vec 用的是 rescue 查询向量。
本构建器把池裁回存活池，并直接调用生产同一个构造函数
`dataset_recommender.retrieval.ltr_features.ltr_feature_rows`（R1 单源）——
训练侧 `--feature-set f3` 读到的每一格都与生产推理逐值相同。

**池口径**（与 harvest 机制逐条同构，禁止写第二份判官）：

- 并集行（带 `n_returned`，A 轨按构造恒为前缀，全量核验 0 违例）：存活池 = `candidates[:n_returned]`；
  B-only 候选出池（生产从未展示给 LTR），**整行候选随之裁减**——f3 的数据形状就是生产池形状。
- 单轨行（`n_returned` 缺席：base_train / heldout5）：整列表即存活池。
- 弃权行（`n_returned=0`）：生产如实无结果、LTR 不运行 → 存活池为空，行保留但候选为空。
- intent 用 F1 构建器的 `effective_intent`（同一函数 import，含 unresolved_term 降级回放）；
  lex 分取自带 `lex_score`（= 生产 `RetrievedCandidate.score` 的 round-4），
  vec 分取自带 `vec_score`（harvest 时同一嵌入器/同一序列化/同一查询向量），
  缺失 → `has_vec=0、vec_norm=0`（共享函数显式语义，与 `train_ltr.candidate_features` 一致）。

**保真度对账**（写进审计、不靠人眼）：用生产 `DatasetRetriever.retrieve` 对每个查询现跑一遍
（解析→硬过滤→规则序→家族封顶，top_k 与当时 harvest 一致），逐行比对 uid 序列与存量池；
不一致 = 语料/解析器自 harvest 后漂移（实测集中在 OOD typo_pinyin 类，见 f3_audit.json）。

不变量：保留候选的标签列（`uid/pos/lex_score/vec_score/grade/llm_semantic`）与源行**逐位相同**；
行数与 query_id 序不变；只裁候选、只加 `f3_feats` / `f3_pool` 字段（每次都断言）。

用法：
    python research/ltr_data/build_f3_features.py             # 全量构建 f3 变体 + 审计
    python research/ltr_data/build_f3_features.py --self-test   # 已知答案自测（不写文件）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))
sys.path.insert(0, str(AGENT_ROOT / "research" / "ltr_data" / "exp_20260929"))

import numpy as np              

import train_ltr as T              
from augment_vec_scores import load_uid_records              
from build_f1_features import (                                            
    HELDOUT_GOLD,
    LTR_DATA,
    OOD_EVALONLY,
    OOD_TRAINABLE,
    WAVE2,
    datasets,
    effective_intent,
    read_jsonl,
    write_jsonl,
)
from evaluate_recommendation import load_pipeline              
from dataset_recommender.retrieval.ltr_features import ltr_feature_rows              
from dataset_recommender.retrieval.retriever import DatasetRetriever, RetrievedCandidate              

                                                                   
OUT_ROOT = LTR_DATA / "_exp_20260930_f3"
DATA_ROOT = OUT_ROOT / "data"


                                                                           
def survival_pool(row: dict) -> "list[dict]":
    """生产回放存活池：并集行取 A 轨前缀 `candidates[:n_returned]`；单轨行取整列表。"""
    candidates = row.get("candidates") or []
    n = row.get("n_returned")
    if n is None:
        return list(candidates)
    return list(candidates[: max(0, int(n))])


def f3_pool_features(pool: "list[dict]", intent, uid_records: dict) -> "list[list[float]]":
    """对一个存活池调 R1 共享函数，返回 9 维特征行（顺序 = `train_ltr.FEATURE_NAMES`）。

    uid 查不到记录 → 抛错（fail-closed：pool 组成变了是数据事故，不许静默记 0 混过去；
    实测全 split missing_uid=0，与 F1 报告一致）。
    """
    items = []
    dense: "list[float | None]" = []
    for cand in pool:
        record = uid_records.get(cand.get("uid"))
        if record is None:
            raise ValueError(f"存活池候选 uid 查不到记录：{cand.get('uid')!r}")
        items.append(RetrievedCandidate(record=record, score=float(cand.get("lex_score") or 0.0)))
        dense.append(cand.get("vec_score"))                                       
    rows = ltr_feature_rows(items, dense, intent, T.FEATURE_NAMES)
    if rows is None:
        raise ValueError("ltr_feature_rows 返回 None——intent 不应为 None（effective_intent 恒有值）")
    return rows


def f3_overlay_row(row: dict, settings, uid_records: dict) -> "tuple[dict, dict]":
    """给一行做 f3 改造：候选裁到存活池 + 每候选写 `f3_feats`；返回 `(新行, 审计)`；不改入参。"""
    _intent, degraded, effective = effective_intent(str(row.get("query") or ""), settings)
    source_candidates = row.get("candidates") or []
    pool = survival_pool(row)
    feats = f3_pool_features(pool, effective, uid_records)
    new_candidates = []
    for cand, row9 in zip(pool, feats):
        enriched = dict(cand)
        enriched["f3_feats"] = [float(v) for v in row9]
        new_candidates.append(enriched)
    out = dict(row)
    out["candidates"] = new_candidates
    out["f3_pool"] = len(pool)
    audit = {
        "query_id": row.get("query_id"),
        "pool": len(pool),
        "dropped": len(source_candidates) - len(pool),
        "abstain": bool(getattr(_intent, "abstain", False)),
        "degraded": degraded is not None,
        "vec_missing": sum(1 for c in pool if c.get("vec_score") is None),
    }
    return out, audit


def assert_labels_untouched(source_rows: "list[dict]", overlay_rows: "list[dict]") -> None:
    """f3 不变量：行数/query_id 序不变；**保留候选**（存活池前缀）的标签列逐位相同。"""
    if len(source_rows) != len(overlay_rows):
        raise AssertionError(f"行数变了：{len(source_rows)} -> {len(overlay_rows)}")
    for src, dst in zip(source_rows, overlay_rows):
        if src.get("query_id") != dst.get("query_id"):
            raise AssertionError(f"行序漂移：{src.get('query_id')} -> {dst.get('query_id')}")
        sc, dc = src.get("candidates") or [], dst.get("candidates") or []
        if len(dc) > len(sc):
            raise AssertionError(f"{src.get('query_id')} 候选变多了：{len(sc)} -> {len(dc)}（只许裁不许加）")
        for a, b in zip(sc, dc):                     
            for key in ("uid", "pos", "lex_score", "vec_score", "grade", "llm_semantic"):
                if a.get(key) != b.get(key):
                    raise AssertionError(f"{src.get('query_id')}/{a.get('uid')} 的 {key} 被改动")


                                                                        
def replay_fidelity(rows: "list[dict]", settings, records, retriever) -> dict:
    """现跑生产检索（同一 retrieve），逐行比对 uid 序列与存量存活池；返回计数与漂移样本。"""
    exact = reorder = drift = 0
    drift_examples = []
    for row in rows:
        _i, _d, effective = effective_intent(str(row.get("query") or ""), settings)
        top_k = max(12, int(row.get("n_returned") or 12))                                    
        results = [] if getattr(effective, "abstain", False) else retriever.retrieve(records, effective, top_k=top_k)
        replay = [str(getattr(c.record, "raw", {}).get("dataset_uid") or "") for c in results]
        stored = [str(c.get("uid") or "") for c in survival_pool(row)]
        if replay == stored:
            exact += 1
        elif set(replay) == set(stored):
            reorder += 1
        else:
            drift += 1
            if len(drift_examples) < 10:
                drift_examples.append({
                    "query_id": row.get("query_id"), "stored": len(stored), "replay": len(replay),
                    "only_stored": sorted(set(stored) - set(replay))[:4],
                    "only_replay": sorted(set(replay) - set(stored))[:4],
                })
    return {"rows": len(rows), "exact": exact, "reorder": reorder, "drift": drift,
            "drift_examples": drift_examples}


                                                                                    
def _matrices(row: dict, feature_set: str) -> np.ndarray:
    return T.candidate_features(row, feature_set)[0].astype(np.float64)


def quantify_delta(source_rows: "list[dict]", f3_rows: "list[dict]", settings, uid_records: dict) -> dict:
    """f3 相对 f0（并集全量）与 f1（同前缀）的输入改动量机器账。

    f0 覆盖并集全部候选；f3 只覆盖存活池（恒为源列表前缀）→ 前缀逐格比对 + 裁减单列账。
    f1 与 f3 在**同一前缀**上比对：理论上只有 lex_norm/vec 池级口径可能不同，
    matched/raw/inv_pos 应逐位相同（本函数用计数钉死这个「应该」）。
    """
    from build_f1_features import overlay_row as f1_overlay_row             

    names = list(T.FEATURE_NAMES)
    total = {"f0_candidates": 0, "f3_candidates": 0, "dropped": 0, "empty_pool_rows": 0}
    f0_diff = np.zeros(len(names), dtype=np.int64)
    f0_changed = 0
    f1_diff = np.zeros(len(names), dtype=np.int64)
    f1_changed = 0
    f1_cells = 0
    max_abs = np.zeros(len(names))
    for src, f3 in zip(source_rows, f3_rows):
        pool_n = len(f3.get("candidates") or [])
        src_n = len(src.get("candidates") or [])
        total["f0_candidates"] += src_n
        total["f3_candidates"] += pool_n
        total["dropped"] += src_n - pool_n
        if pool_n == 0:
            total["empty_pool_rows"] += 1
            continue
        m0 = _matrices(src, "f0")[:pool_n]
        m3 = _matrices(f3, "f3")
        diff = np.abs(m0 - m3) > 1e-9
        f0_diff += diff.sum(axis=0)
        f0_changed += int(diff.any(axis=1).sum())
        max_abs = np.maximum(max_abs, np.abs(m0 - m3).max(axis=0))
                                    
        f1_row, _ = f1_overlay_row(src, settings, uid_records, "f1")
        m1 = _matrices(f1_row, "f1")[:pool_n]
        diff1 = np.abs(m1 - m3) > 1e-9
        f1_diff += diff1.sum(axis=0)
        f1_changed += int(diff1.any(axis=1).sum())
        f1_cells += int(diff1.sum())
    return {
        **total,
        "f0_prefix_changed_candidates": f0_changed,
        "f0_prefix_cells_by_column": {name: int(v) for name, v in zip(names, f0_diff)},
        "f0_prefix_max_abs_delta": {name: round(float(v), 6) for name, v in zip(names, max_abs)},
        "f1_prefix_changed_candidates": f1_changed,
        "f1_prefix_changed_cells": f1_cells,
        "f1_prefix_cells_by_column": {name: int(v) for name, v in zip(names, f1_diff)},
    }


                                                                           
def self_test(settings, uid_records: dict, records) -> int:
    """合成池已知答案（逐值比对）+ 真实数据不变量 + 保真度对账（干净 split 必须 100% exact）。"""
    failures: "list[str]" = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    from types import SimpleNamespace

    from dataset_recommender.retrieval.normalizer import DatasetRecord

    def rec(uid, species="", tissue="", disease="", raw=None):
        return DatasetRecord(dataset_name=uid, species=species, tissue=tissue, disease=disease,
                             chemistry="", count="", unit="", has_raw_data=raw, url="",
                             source_file="x.json", description="d", raw={})

    synth_records = {
        "A": rec("A", "human", "breast", "breast cancer", True),
        "B": rec("B", "mouse", "liver", "", False),
        "C": rec("C", "human", "brain", "", None),
    }
    intent = SimpleNamespace(constraints={"species": ["human"]},
                             preferred_constraints={"tissue": ["breast"]},
                             has_raw_data_required=True)

                                                        
    pool = [{"uid": "A", "lex_score": 4.0, "vec_score": 0.5},
            {"uid": "B", "lex_score": 2.0, "vec_score": None},
            {"uid": "C", "lex_score": 0.0, "vec_score": 0.1}]
    rows = f3_pool_features(pool, intent, synth_records)
    expected = [
                                                                                              
        [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.1, 1.0, 1.0 / 3.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ]
    check("KA-1 合成池 9 维逐值", rows == expected, f"{rows}")

                                          
    rows = f3_pool_features([{"uid": "A", "lex_score": 0.0, "vec_score": 0.2},
                             {"uid": "B", "lex_score": None, "vec_score": None}],
                            intent, synth_records)
    check("KA-2 lex_max=0 → lex_norm 全 0",
          [r[0] for r in rows] == [0.0, 0.0] and [r[2] for r in rows] == [1.0, 0.0],
          f"{[r[:3] for r in rows]}")

                                                        
    free = SimpleNamespace(constraints={}, preferred_constraints={}, has_raw_data_required=None)
    rows = f3_pool_features([{"uid": "A", "lex_score": 3.0, "vec_score": 0.4}], free, synth_records)
    check("KA-3 无约束查询", rows == [[1.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], f"{rows}")

                            
    check("KA-4 空池", f3_pool_features([], intent, synth_records) == [], "空池返回 []")

                                            
    wave2 = read_jsonl(WAVE2)
    union_row = next((r for r in wave2
                      if int(r.get("n_returned") or 0) > 0
                      and len(r.get("candidates") or []) > int(r["n_returned"])), None)
    check("REAL-1a 找得到候选数 > n_returned 的真实并集行（夹具前提）", union_row is not None,
          f"query={union_row['query_id'] if union_row else None}")
    if union_row is not None:
        out, audit = f3_overlay_row(union_row, settings, uid_records)
        n = int(union_row["n_returned"])
        check("REAL-1b f3 候选裁到 A 轨前缀",
              [c["uid"] for c in out["candidates"]] == [c["uid"] for c in union_row["candidates"][:n]]
              and audit["dropped"] == len(union_row["candidates"]) - n,
              f"pool={audit['pool']} dropped={audit['dropped']}")
        first_lex_norm = out["candidates"][0]["f3_feats"][list(T.FEATURE_NAMES).index("lex_norm")]
        check("REAL-1c 存活池首候选 lex_norm 恒 1（规则序首名即池内 max）",
              first_lex_norm == 1.0, f"lex_norm[0]={first_lex_norm}")

                                         
    abstain_row = next((r for r in read_jsonl(OOD_EVALONLY) if int(r.get("n_returned") or 0) == 0), None)
    check("REAL-2a 找得到 n_returned=0 的真实弃权行（夹具前提）", abstain_row is not None,
          f"query={abstain_row['query_id'] if abstain_row else None}")
    if abstain_row is not None:
        out, _audit = f3_overlay_row(abstain_row, settings, uid_records)
        check("REAL-2b 弃权行存活池为空（生产如实无结果，LTR 不运行）",
              out["candidates"] == [] and out["f3_pool"] == 0, f"f3_pool={out['f3_pool']}")

                                                                 
    from build_f1_features import overlay_row as f1_overlay_row
    base_rows = T.load_split(LTR_DATA, "train")[0][:30]
    mismatch = 0
    for row in base_rows:
        f3_row, _ = f3_overlay_row(row, settings, uid_records)
        f1_row, _ = f1_overlay_row(row, settings, uid_records, "f1")
        m3 = T.candidate_features(f3_row, "f3")[0]
        m1 = T.candidate_features(f1_row, "f1")[0]
        if m3.shape != m1.shape or not np.allclose(m3.astype(np.float64), m1.astype(np.float64), atol=0, rtol=0):
            mismatch += 1
    check("REAL-3 单轨行前 30 行 f3 == f1 逐位（float32 后）", mismatch == 0, f"mismatch={mismatch}/30")

                                      
    sample = wave2[:20]
    overlaid = [f3_overlay_row(r, settings, uid_records)[0] for r in sample]
    try:
        assert_labels_untouched(sample, overlaid)
        check("REAL-4 标签不变量断言不炸", True, "wave2 前 20 行")
    except AssertionError as exc:
        check("REAL-4 标签不变量断言不炸", False, str(exc))

                                                                       
    retriever = DatasetRetriever(top_k=12)
    for name, path in (("wave2", WAVE2), ("heldout5", HELDOUT_GOLD)):
        rep = replay_fidelity(read_jsonl(path), settings, records, retriever)
        check(f"REAL-5 {name} 回放保真度 100% exact",
              rep["exact"] == rep["rows"] and rep["reorder"] == 0,
              f"{rep['exact']}/{rep['rows']} drift={rep['drift']}")

    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


                                                                            
def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="只跑已知答案自测，不写任何文件")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--skip-fidelity", action="store_true", help="跳过保真度对账（省一半时间；默认开启）")
    args = parser.parse_args(argv)
    data_root = args.out_root / "data"                                     

    settings, records = load_pipeline()
    uid_records = load_uid_records()
    print(f"语料 {len(uid_records)} 条记录 / keyword_mapping {len(settings.keyword_mapping or {})} 项")

    if args.self_test:
        return self_test(settings, uid_records, records)

    sets = datasets()
    for name, rows in sets.items():
        print(f"  {name:16s} rows={len(rows):5d} cand={sum(len(r.get('candidates') or []) for r in rows):6d}")

    audit: dict = {"built_at_unix": int(time.time()), "splits": {}, "delta": {}, "fidelity": {}}
    retriever = DatasetRetriever(top_k=12)
    for name, rows in sets.items():
        overlaid, row_audits = [], []
        for row in rows:
            out, row_audit = f3_overlay_row(row, settings, uid_records)
            overlaid.append(out)
            row_audits.append(row_audit)
        assert_labels_untouched(rows, overlaid)
        audit["splits"][name] = {
            "rows": len(rows),
            "source_candidates": sum(len(r.get("candidates") or []) for r in rows),
            "pool_candidates": sum(a["pool"] for a in row_audits),
            "dropped": sum(a["dropped"] for a in row_audits),
            "empty_pool_rows": sum(1 for a in row_audits if a["pool"] == 0),
            "abstain": sum(int(a["abstain"]) for a in row_audits),
            "degraded": sum(int(a["degraded"]) for a in row_audits),
            "vec_missing_in_pool": sum(a["vec_missing"] for a in row_audits),
        }
        audit["delta"][name] = quantify_delta(rows, overlaid, settings, uid_records)
        if not args.skip_fidelity:
            audit["fidelity"][name] = replay_fidelity(rows, settings, records, retriever)
        else:
            audit["fidelity"][name] = {"skipped": True}

                                                     
        if name == "heldout_merged":
            continue
        if name in ("d0", "d1"):
            write_jsonl(data_root / f"{name}_f3" / "synth_train.jsonl", overlaid)
        if name == "heldout5":
            for variant in (data_root / "d0_f3", data_root / "d1_f3"):
                write_jsonl(variant / "synth_heldout.jsonl", overlaid)
            write_jsonl(data_root / "eval" / "heldout5_f3.jsonl", overlaid)
        if name == "ood_evalonly":
            write_jsonl(data_root / "eval" / "oodonly_f3.jsonl", overlaid)
        if name == "d0":
            write_jsonl(data_root / "eval" / "d0_f3_train.jsonl", overlaid)
        if name == "d1":
            write_jsonl(data_root / "eval" / "d1_f3_train.jsonl", overlaid)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "f3_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== F3 池形状（存活池 vs 并集） ===")
    for name, block in audit["splits"].items():
        print(f"{name:16s} rows={block['rows']:5d} pool={block['pool_candidates']:6d} "
              f"dropped={block['dropped']:5d} empty={block['empty_pool_rows']:4d} "
              f"vec_missing={block['vec_missing_in_pool']}")
    print("\n=== F3 相对 F0 的输入改动量（同前缀逐格） ===")
    for name, block in audit["delta"].items():
        changed = {k: v for k, v in block["f0_prefix_cells_by_column"].items() if v}
        print(f"{name:16s} pool={block['f3_candidates']:6d} dropped={block['dropped']:5d} "
              f"changed_cand={block['f0_prefix_changed_candidates']:5d} 按列={changed}")
        f1_changed = {k: v for k, v in block["f1_prefix_cells_by_column"].items() if v}
        print(f"{'':16s} vs f1 同前缀：changed_cand={block['f1_prefix_changed_candidates']} 按列={f1_changed}")
    if not args.skip_fidelity:
        print("\n=== 回放保真度（现跑 retrieve vs 存量存活池） ===")
        for name, rep in audit["fidelity"].items():
            print(f"{name:16s} exact={rep['exact']:5d}/{rep['rows']:5d} reorder={rep['reorder']} drift={rep['drift']}")
    print(f"\nwrote {args.out_root / 'f3_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
