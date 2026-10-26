                       
"""实验 A 的 F1 特征构建器：把训练行里的 9 维输入按**生产推理期**的口径重放一遍。

为什么（2026-09-29 Codex 评审「训练特征来源与生产语义不一致」）：离线行的四类输入与
`src/dataset_recommender/retrieval/vector_recall.py::_ltr_fused_order` 现场算出来的不是一回事——

1. **位置**：离线读候选自带的 `pos`，合并池（A/B 两条轨 union）里同一 `pos` 会出现多次
   （wave2 42/187 查询、OOD 可训练层 92/352、只评测层 48/181）；生产读的是存活池里的下标，
   `inv_pos = 1/(i+1)`，天然严格递减。
2. **判官**：离线 `matched` 用的是 `evaluate_recommendation.constraint_satisfied`（冻结基准的
   外部裁判：拿 `judge` 词表对字段做子串匹配），生产用的是
   `retrieval.retriever.constraint_satisfied`（领域描述符 + 规范化取值）。
3. **意图来源**：离线 `matched` 判的是**人工构造的 intended constraints**（harvest 里的
   `judge_map`），生产判的是**解析得到的 intent**（`constraints + preferred_constraints`），
   拼音/否定/降级类查询上两者本就不同。
4. **raw 双槽**：wave2 与 OOD 行**没有顶层 `has_raw_data_required`**，训练侧落到 `False`；
   生产读 `intent.has_raw_data_required`（三态，`None` 与 `False` 不同）。

本脚本把上述四项按生产口径重算，写成**叠加字段**（不改原字段，原行保持可追溯）：

- 行级 `f1_raw_required`（bool）：生效 intent 的 raw 要求是否 `is True`；
- 行级 `f1_intent`：生效 intent 的约束投影 + 是否降级 + 未收录词，供审计；
- 候选级 `f1_matched`：`{species,tissue,disease,has_raw_data} → 0/1`，前三维用生产裁判对生效
  intent 的 `constraints + preferred_constraints` 判定（无该项约束记 0），第四维与生产同构
  （要求 raw **且** 记录 `has_raw_data is True` 才记 1）；
- 行级 `f2_query`（仅 `--feature-set f2`）：8 个 query 侧信号，顺序 = `train_ltr.F2_QUERY_FEATURES`。

**判官只有一份**：本脚本 import 生产函数（`query_parser.parse_query`、`retriever.
constraint_satisfied`、`app.workflow.strip_terms`），不复制任何判定逻辑；降级规则复用
`harvest_ood_candidates.has_condition`——即标签采集当时用的那一条，保证「生效 intent」与
标签所在的那次采集同源。

不变量：标签列（`grade` / `llm_semantic` / `vec_score` / `lex_score`）与原行**逐位相同**，
本脚本只新增字段（每次都断言，不靠人眼）。

用法：
    python research/ltr_data/exp_20260929/build_f1_features.py            # 建 F1/F2 全部变体
    python research/ltr_data/exp_20260929/build_f1_features.py --self-test  # 已知答案自测（不写文件）
    python research/ltr_data/exp_20260929/build_f1_features.py --feature-set f1  # 只建一层
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "scripts"))

import numpy as np              

import train_ltr as T              
from augment_vec_scores import load_uid_records              
from dataset_recommender.app.workflow import strip_terms              
from dataset_recommender.retrieval.query_parser import parse_query              
from dataset_recommender.retrieval.retriever import constraint_satisfied              
from evaluate_recommendation import load_pipeline              
from harvest_ood_candidates import has_condition              

                                        
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
OOD_DIR = AGENT_ROOT / "research" / "ood_eval"
OOD_TRAINABLE = OOD_DIR / "synth_ood_trainable_xscore.jsonl"
WAVE2 = LTR_DATA / "synth_train_wave2_20260929_xscore.jsonl"
OOD_EVALONLY = OOD_DIR / "synth_ood_evalonly_xscore.jsonl"
                                                                        
                                                                               
                                                          
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"

                                                                 
                    
OUT_ROOT = LTR_DATA / "_exp_20260929_ab"
DATA_ROOT = OUT_ROOT / "data"

                                                   
MATCH_DIMS = ("species", "tissue", "disease")


def read_jsonl(path: Path) -> "list[dict]":
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: "list[dict]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")


                                                                         
def effective_intent(query: str, settings) -> "tuple[object, dict | None, object]":
    """`(原始 intent, 降级标注, 生效 intent)`——降级规则与 harvest/生产自动降级逐条同构。

    生产（`app/workflow.py` 的 auto-degrade）与采集（`scripts/harvest_ood_candidates.py`）用的是
    同一条规则：unresolved_term 弃权 → 挖掉未收录词重解析 → 可执行且还剩条件才采用。这里改用
    harvest 暴露的同名判定函数，而不是照抄一遍；`_r_cands` 非空是生产侧的额外条件，离线无从
    检验（不重新检索），差异记在报告局限里。
    """
    intent = parse_query(query, settings.keyword_mapping)
    degraded = None
    effective = intent
    if getattr(intent, "abstain", False) and getattr(intent, "abstain_reason", "") == "unresolved_term":
        remaining = " ".join(
            strip_terms(intent.original_query, [t for t in intent.unresolved_terms if t.strip()]).split()
        )
        if remaining and remaining != intent.original_query.strip():
            reparsed = parse_query(remaining, settings.keyword_mapping)
            if (reparsed.parse_status == "executable" and not reparsed.abstain
                    and has_condition(reparsed)):
                degraded = {"ignored_terms": [t for t in intent.unresolved_terms if t.strip()],
                            "query": remaining}
                effective = reparsed
    return intent, degraded, effective


def dim_terms(intent, dim: str) -> "list[str]":
    """生产 `_ltr_fused_order._matched` 的取项口径：硬约束 + 软偏好（可能都为空）。"""
    hard = getattr(intent, "constraints", None) or {}
    soft = getattr(intent, "preferred_constraints", None) or {}
    return list(hard.get(dim) or []) + list(soft.get(dim) or [])


def f1_matched(record, intent) -> "dict[str, float]":
    """生产同构的四个 `matched` 槽（record 为 None 时全部记 0，并按调用方口径计数上报）。"""
    out: "dict[str, float]" = {}
    for dim in MATCH_DIMS:
        terms = dim_terms(intent, dim)
        out[dim] = 0.0 if not terms else (1.0 if constraint_satisfied(record, dim, terms) else 0.0)
    raw_required = getattr(intent, "has_raw_data_required", None) is True
    out["has_raw_data"] = 1.0 if (raw_required and getattr(record, "has_raw_data", None) is True) else 0.0
    return out


def kendall_disagreement(lex: "list[float]", vec: "list[float]") -> float:
    """lex 序与 vec 序的归一化 Kendall tau 距离（0=同序，1=完全相反）。

    序 = 分数降序、并列按原下标升序（与 `train_ltr.lex_order` 的稳定排序同款）；只统计两侧
    都有向量分的候选；不足 2 个可比候选记 0。
    """
    idx = [i for i, value in enumerate(vec) if value == value]              
    n = len(idx)
    if n < 2:
        return 0.0
    lex_order = sorted(idx, key=lambda i: (-lex[i], i))
    vec_order = sorted(idx, key=lambda i: (-vec[i], i))
    rank_lex = {i: r for r, i in enumerate(lex_order)}
    rank_vec = {i: r for r, i in enumerate(vec_order)}
    discordant = 0
    for a in range(n):
        for b in range(a + 1, n):
            i, j = idx[a], idx[b]
            if (rank_lex[i] < rank_lex[j]) != (rank_vec[i] < rank_vec[j]):
                discordant += 1
    return discordant / (n * (n - 1) / 2.0)


def f2_query_features(row: dict, intent, effective) -> "list[float]":
    """`train_ltr.F2_QUERY_FEATURES` 口径的 8 个 query 侧信号（见报告设计节的预注册定义）。

    约束存在位与 raw 位取**生效** intent（生产真正看到的那一份）；未收录比例取**降级前**的
    原始 intent（降级后该信息已被抹掉，正是要保留的信号）。
    """
    bits = [1.0 if dim_terms(effective, dim) else 0.0 for dim in MATCH_DIMS]
    raw_bit = 1.0 if getattr(effective, "has_raw_data_required", None) is not None else 0.0
    tokens = [t for t in str(row.get("query") or "").split() if t]
    unresolved = [t for t in (getattr(intent, "unresolved_terms", None) or []) if str(t).strip()]
                                                        
                                               
    ratio = len(unresolved) / float(len(unresolved) + len(tokens)) if (unresolved or tokens) else 0.0
    lex = [float(c.get("lex_score") or 0.0) for c in (row.get("candidates") or [])]
    vec = [float(c["vec_score"]) if c.get("vec_score") is not None else float("nan")
           for c in (row.get("candidates") or [])]
    return [
        *bits,
        raw_bit,
        ratio,
        float(len(str(row.get("query") or "").strip())),
        max(lex) if lex else 0.0,
        kendall_disagreement(lex, vec),
    ]


def overlay_row(row: dict, settings, uid_records: dict, feature_set: str) -> "tuple[dict, dict]":
    """给一行加 F1（+F2）字段，返回 `(新行, 审计)`；不修改入参。"""
    intent, degraded, effective = effective_intent(str(row.get("query") or ""), settings)
    candidates = row.get("candidates") or []
    missing_uid = 0
    new_candidates = []
    for cand in candidates:
        record = uid_records.get(cand.get("uid"))
        if record is None:
            missing_uid += 1
            record = None
        enriched = dict(cand)
        enriched["f1_matched"] = f1_matched(record, effective)
        new_candidates.append(enriched)
    out = dict(row)
    out["candidates"] = new_candidates
    out["f1_raw_required"] = getattr(effective, "has_raw_data_required", None) is True
    out["f1_intent"] = {
        "constraints": {dim: dim_terms(effective, dim) for dim in MATCH_DIMS},
        "preferred": {dim: list((getattr(effective, "preferred_constraints", None) or {}).get(dim) or [])
                      for dim in MATCH_DIMS},
        "has_raw_data_required": getattr(effective, "has_raw_data_required", None),
        "abstain": bool(getattr(intent, "abstain", False)),
        "abstain_reason": str(getattr(intent, "abstain_reason", "") or ""),
        "degraded": degraded,
    }
    if feature_set == "f2":
        out["f2_query"] = f2_query_features(row, intent, effective)
    audit = {
        "query_id": row.get("query_id"),
        "abstain": bool(getattr(intent, "abstain", False)),
        "degraded": degraded is not None,
        "missing_uid": missing_uid,
        "raw_required": out["f1_raw_required"],
    }
    return out, audit


def assert_labels_untouched(source_rows: "list[dict]", overlay_rows: "list[dict]") -> None:
    """标签层不变量：grade / llm_semantic / vec_score / lex_score / pos 逐位不变。"""
    if len(source_rows) != len(overlay_rows):
        raise AssertionError(f"行数变了：{len(source_rows)} -> {len(overlay_rows)}")
    for src, dst in zip(source_rows, overlay_rows):
        if src.get("query_id") != dst.get("query_id"):
            raise AssertionError(f"行序漂移：{src.get('query_id')} -> {dst.get('query_id')}")
        sc, dc = src.get("candidates") or [], dst.get("candidates") or []
        if len(sc) != len(dc):
            raise AssertionError(f"{src.get('query_id')} 候选数变了：{len(sc)} -> {len(dc)}")
        for a, b in zip(sc, dc):
            for key in ("uid", "pos", "lex_score", "vec_score", "grade", "llm_semantic"):
                if a.get(key) != b.get(key):
                    raise AssertionError(f"{src.get('query_id')}/{a.get('uid')} 的 {key} 被改动")


def feature_matrices(rows: "list[dict]", feature_set: str) -> np.ndarray:
    """整份数据集的 9/17 维特征矩阵（f0 用自带 pos/matched，f1/f2 用叠加字段）。"""
    blocks = [T.candidate_features(row, feature_set)[0] for row in rows]
    width = len(T.feature_names(feature_set))
    if not blocks:
        return np.zeros((0, width), dtype=np.float32)
    return np.vstack(blocks)


def quantify_delta(rows: "list[dict]", feature_set: str = "f1") -> dict:
    """F1（或 F2）相对 F0 的输入改动量：行数、候选格数、按列的改动格数与最大绝对差。

    f0 与 f1 同为 9 列；f2 = f1 + query 块，故只在**共有的 9 列**上比对改动，
    新增列单列取值分布（它们本来就不存在于 F0，谈不上"改动"）。
    """
    names = list(T.feature_names(feature_set))
    f0 = feature_matrices(rows, "f0")
    fnew = feature_matrices(rows, feature_set)
    shared = f0.shape[1]
    if fnew.shape[0] != f0.shape[0] or fnew.shape[1] < shared:
        raise AssertionError(f"矩阵形状不可比：f0={f0.shape} vs {feature_set}={fnew.shape}")
    a, b = f0, fnew[:, :shared]
    diff = np.abs(a - b) > 1e-9
    per_col = {name: int(diff[:, i].sum()) for i, name in enumerate(names[:shared])}
    changed_candidates = int(diff.any(axis=1).sum())
    extra = {}
    for i in range(shared, fnew.shape[1]):
        column = fnew[:, i].astype(np.float64)
        extra[names[i]] = {
            "min": round(float(column.min()), 6) if column.size else None,
            "max": round(float(column.max()), 6) if column.size else None,
            "mean": round(float(column.mean()), 6) if column.size else None,
            "nonzero": int((column != 0).sum()),
        }
    return {
        "queries": len(rows),
        "candidates": int(f0.shape[0]),
        "candidates_changed": changed_candidates,
        "candidates_changed_share": round(changed_candidates / f0.shape[0], 4) if f0.shape[0] else None,
        "cell_diffs": int(diff.sum()),
        "cells_by_column": per_col,
        "max_abs_delta_by_column": {
            name: round(float(np.abs(a[:, i] - b[:, i]).max()), 6) for i, name in enumerate(names[:shared])
        },
        "added_columns": extra,
    }


                                                                           
def self_test(settings, uid_records: dict) -> int:
    """三条已知答案检查：pos 重复必被修正 / 判官与意图来源分歧 / 缺 raw 标记必被补。

    全部对着**真实数据**跑，断言的是"F1 与 F0 在该情形上必然不同"，不是"看起来差不多"。
    """
    failures: "list[str]" = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    wave2 = read_jsonl(WAVE2)
    dup_rows = [r for r in wave2
                if len([c.get("pos") for c in (r.get("candidates") or [])])
                != len({c.get("pos") for c in (r.get("candidates") or [])})]
    check("POS-1 存在 pos 重复的真实行（夹具前提）", bool(dup_rows),
          f"wave2 里 {len(dup_rows)}/{len(wave2)} 行 pos 有重复")
    if dup_rows:
        row = dup_rows[0]
        f0 = T.candidate_features(row, "f0")[0]
        override = {
            **row,
            "f1_raw_required": False,
            "candidates": [{**c, "f1_matched": {d: 0.0 for d in T.DIM_SLOTS}}
                           for c in row["candidates"]],
        }
        f1 = T.candidate_features(override, "f1")[0]
        n = len(row["candidates"])
        pos_col = T.FEATURE_NAMES.index("inv_pos")
        expected = np.asarray([1.0 / (i + 1) for i in range(n)], dtype=np.float64)
        check("POS-2 F1 的 inv_pos 严格按池内下标",
              bool(np.allclose(f1[:, pos_col].astype(np.float64), expected, atol=1e-6, rtol=0.0)),
              f"query={row['query_id']} n={n}")
        check("POS-3 同一条上 F0 的 inv_pos 确实有重复（修复点真实存在）",
              len(set(f0[:, pos_col].tolist())) < n,
              f"F0 去重后 {len(set(f0[:, pos_col].tolist()))} 个不同值 / {n} 个候选")

                                                             
    divergence = None
    for row in read_jsonl(OOD_EVALONLY) + read_jsonl(WAVE2) + T.load_split(LTR_DATA, "train")[0]:
        intent, _deg, eff = effective_intent(str(row.get("query") or ""), settings)
        cands = row.get("candidates") or []
        if not cands:
            continue
        for cand in cands:
            record = uid_records.get(cand.get("uid"))
            prod = f1_matched(record, eff)
            stored = cand.get("matched") or {}
            for dim in ("species", "tissue", "disease", "has_raw_data"):
                if bool(stored.get(dim)) != bool(prod.get(dim)):
                    divergence = (row["query_id"], cand["uid"], dim, bool(stored.get(dim)), bool(prod.get(dim)))
                    break
            if divergence:
                break
        if divergence:
            break
    check("INTENT-1 找得到 matched 分歧的真实样本（判官/意图来源确有差异）", divergence is not None,
          f"{divergence}")
    if divergence:
        qid, uid, dim, old, new = divergence
        check("INTENT-2 分歧样本进 F1 后该格确实翻面", old != new,
              f"{qid}/{uid}/{dim}: intended外判={old} -> 生产={new}")

                                                                         
    with_raw = 0
    for row in read_jsonl(WAVE2) + read_jsonl(OOD_EVALONLY):
        if "has_raw_data_required" in row:
            continue
        _i, _d, eff = effective_intent(str(row.get("query") or ""), settings)
        if getattr(eff, "has_raw_data_required", None) is not None:
            with_raw += 1
    check("RAW-1 缺顶层 raw 标记的行里，确实有生效 intent 带 raw 判断的", with_raw > 0,
          f"{with_raw} 行（这些行在 F0 下恒被当作 raw_required=0）")

    row = next((r for r in read_jsonl(WAVE2) + read_jsonl(OOD_EVALONLY)
                if "has_raw_data_required" not in r
                and overlay_row(r, settings, uid_records, "f1")[0]["f1_raw_required"]), None)
    check("RAW-2 存在「缺顶层标记但生效 intent 要求 raw」的真实行", row is not None,
          f"query={row['query_id'] if row else None}")
    if row is not None:
        f0 = T.candidate_features(row, "f0")[0]
        new_row, _audit = overlay_row(row, settings, uid_records, "f1")
        f1 = T.candidate_features(new_row, "f1")[0]
        raw_col = T.FEATURE_NAMES.index("raw_required")
        matched_raw_col = T.FEATURE_NAMES.index("matched_has_raw_data")
        check("RAW-3 该行 F0 的 raw 双槽恒 0（缺标记被当 False），F1 已按生效 intent 置位",
              set(f0[:, raw_col].tolist()) == {0.0} and set(f1[:, raw_col].tolist()) == {1.0},
              f"query={row['query_id']} F0 raw={sorted(set(f0[:, raw_col].tolist()))} "
              f"F1 raw={sorted(set(f1[:, raw_col].tolist()))} "
              f"F1 matched_has_raw_data={sorted(set(f1[:, matched_raw_col].tolist()))}")
    print(f"self-test: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


                                                                        
def build_variant(rows: "list[dict]", settings, uid_records: dict, feature_set: str) -> "tuple[list[dict], dict]":
    out = []
    audit = {"rows": 0, "abstain": 0, "degraded": 0, "missing_uid": 0, "raw_required_true": 0}
    for row in rows:
        enriched, row_audit = overlay_row(row, settings, uid_records, feature_set)
        out.append(enriched)
        audit["rows"] += 1
        audit["abstain"] += int(row_audit["abstain"])
        audit["degraded"] += int(row_audit["degraded"])
        audit["missing_uid"] += row_audit["missing_uid"]
        audit["raw_required_true"] += int(row_audit["raw_required"])
    assert_labels_untouched(rows, out)
    return out, audit


def label_provenance(rows: "list[dict]") -> dict:
    """`llm_semantic` 的评委口径（label_model / 评委数 / 取值分布）——监督来源随文件而异，必须记账。"""
    models: "dict[str, int]" = {}
    spread: "dict[str, int]" = {}
    values: "dict[str, int]" = {}
    for row in rows:
        for cand in row.get("candidates") or []:
            models[str(cand.get("label_model"))] = models.get(str(cand.get("label_model")), 0) + 1
            spread[str(cand.get("xscore_n"))] = spread.get(str(cand.get("xscore_n")), 0) + 1
            values[str(cand.get("llm_semantic"))] = values.get(str(cand.get("llm_semantic")), 0) + 1
    return {"label_models": models, "xscore_n": spread, "llm_semantic": values}


def datasets() -> "dict[str, list[dict]]":
    """训练/评测口径的显式装配（与 `train_ltr` 的层合并/混合实现**同一份代码**）。

    - `d0` / `d1`：现役链（`load_split` 取 synth_train 的最富层）+ 可训练 OOD 层，D1 另加 wave2；
    - `heldout5`：**固定评测目标**（5 评委共识）；
    - `heldout_merged`：`--in-dir` 那条分层链看到的 heldout（单评委语义）——只作口径对照，
      不进评测；
    - `ood_evalonly`：OOD 面的只评测层。
    """
    base_train = T.load_split(LTR_DATA, "train")[0]
    d0, stats0 = T.merge_mix_in(list(base_train), [OOD_TRAINABLE])
    d1, stats1 = T.merge_mix_in(list(d0), [WAVE2])
    if stats1[WAVE2.name]["replaced"]:
        raise AssertionError("wave2 不应替换既有 query_id——口径变了，先查再跑")
    return {
        "base_train": base_train,
        "d0": d0,
        "d1": d1,
        "heldout5": read_jsonl(HELDOUT_GOLD),
        "heldout_merged": T.load_split(LTR_DATA, "heldout")[0],
        "ood_evalonly": read_jsonl(OOD_EVALONLY),
    }


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", choices=("f1", "f2", "both"), default="both")
    parser.add_argument("--self-test", action="store_true", help="只跑已知答案自测，不写任何文件")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)

    settings, _records = load_pipeline()
    uid_records = load_uid_records()
    print(f"语料 {len(uid_records)} 条记录 / keyword_mapping {len(settings.keyword_mapping or {})} 项")

    if args.self_test:
        return self_test(settings, uid_records)

    sets = datasets()
    for name, rows in sets.items():
        print(f"  {name:16s} rows={len(rows):5d} cand={sum(len(r.get('candidates') or []) for r in rows):6d}")

    feature_sets = ("f1", "f2") if args.feature_set == "both" else (args.feature_set,)
    audit: dict = {"feature_sets": {}, "shapes": {}, "delta_vs_f0": {}, "label_provenance": {},
                   "built_at_unix": int(time.time())}
    for name, rows in sets.items():
        audit["shapes"][name] = {
            "rows": len(rows),
            "candidates": sum(len(r.get("candidates") or []) for r in rows),
            "queries_with_duplicate_pos": sum(
                1 for r in rows
                if len([c.get("pos") for c in (r.get("candidates") or [])])
                != len({c.get("pos") for c in (r.get("candidates") or [])})
            ),
        }
        audit["label_provenance"][name] = label_provenance(rows)

    for feature_set in feature_sets:
        audit["feature_sets"][feature_set] = {}
        for name, rows in sets.items():
            enriched, row_audit = build_variant(rows, settings, uid_records, feature_set)
            audit["feature_sets"][feature_set][name] = row_audit
            audit["delta_vs_f0"][f"{name}::{feature_set}"] = quantify_delta(enriched, feature_set)
            if name == "heldout_merged":
                continue                                  
            if name in ("d0", "d1"):
                variant = args.out_root / "data" / f"{name}_{feature_set}"
                write_jsonl(variant / "synth_train.jsonl", enriched)
            if name == "heldout5":
                for variant in (args.out_root / "data" / f"d0_{feature_set}",
                                args.out_root / "data" / f"d1_{feature_set}"):
                    write_jsonl(variant / "synth_heldout.jsonl", enriched)
                write_jsonl(args.out_root / "data" / "eval" / f"heldout5_{feature_set}.jsonl", enriched)
            if name == "ood_evalonly":
                write_jsonl(args.out_root / "data" / "eval" / f"oodonly_{feature_set}.jsonl", enriched)
            if name == "d0":
                write_jsonl(args.out_root / "data" / "eval" / f"d0_{feature_set}_train.jsonl", enriched)
            if name == "d1":
                write_jsonl(args.out_root / "data" / "eval" / f"d1_{feature_set}_train.jsonl", enriched)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "f1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== F1/F2 相对 F0 的输入改动量 ===")
    for key, block in audit["delta_vs_f0"].items():
        print(f"{key:26s} cand={block['candidates']:6d} changed={block['candidates_changed']:6d} "
              f"({block['candidates_changed_share']}) cells={block['cell_diffs']}")
        changed = {k: v for k, v in block["cells_by_column"].items() if v}
        print(f"{'':26s} 按列改动：{changed}")
    print("\n=== 语义标签来源（llm_semantic 的评委口径） ===")
    for name, prov in audit["label_provenance"].items():
        print(f"  {name:16s} label_model={prov['label_models']} xscore_n={prov['xscore_n']}")
    print(f"\nwrote {args.out_root / 'f1_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
