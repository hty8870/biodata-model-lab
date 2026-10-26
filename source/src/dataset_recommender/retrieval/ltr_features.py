                       
"""LTR 融合 9 维推理期特征构造的**唯一真源**（生产推理与离线训练共用）。

特征契约名与顺序见 `models/ltr/ltr_spec.json` 的 `feature_order`（训练侧镜像为
`scripts/train_ltr.py` 的 `FEATURE_NAMES`）：lex 按**池内 max** 归一、vec 为原始余弦、
位置 1/(i+1)、逐维约束命中（生产裁判 `retriever.constraint_satisfied` 对 intent 的
硬约束 + 软偏好并集逐维判定）、raw 双槽（未要求 raw 时两槽皆 0）。

为什么单列模块（2026-09-30 队列② R1）：这段构造历史上同时存在于
`vector_recall._ltr_fused_order`（推理侧）与 `train_ltr.candidate_features`（训练侧），
实验 A（研究/10-自我递归与自动训练/LTR实验AB-特征与标签-2026-09-29.md §7.1）证明
手工对齐必然残留口径差（并集池 vs 存活池的 lex/vec 池级差异）。单源化之后：

- 生产：`_ltr_fused_order` 只负责加载打分器、调本函数、畸形分守卫与稳定排序；
- 离线：`research/ltr_data/build_f3_features.py` 回放生产存活池后调同一函数产出 f3 叠加层，
  训练侧（`train_ltr --feature-set f3`）读到的即是生产推理逐值相同的特征。

红线（改动前必读）：

- 输入候选只需 duck-type 出 `.record`（DatasetRecord）与 `.score`（词面分）——生产传
  `RetrievedCandidate`，离线以同型对象回放；`dense_scores` 与候选等长，绝不截断/补齐。
- `dense_scores[i] is None` 表示该候选**无向量分** → `has_vec=0`、`vec_norm=0`，与训练侧
  `candidate_features` 的缺 vec 语义一致；生产 dense/vector 路径每个存活候选都会算出余弦
  （恒非 None → has_vec 恒 1），与改造前内联实现逐位一致。
- intent 为 None → 返回 None：无解析约束时特征处于训练分布外，调用方按不可用回退
  （`_ltr_fused_order` 既有回退语义不变）。
- `constraint_satisfied` 保持函数体内局部导入（沿用 `_ltr_fused_order` 原样风格，
  不把对 retriever 的依赖硬化到模块顶层；惰性环纪律见 MODULES.md 注）。
- 本模块不引入任何重依赖：torch/模型加载/打分都在调用方，这里只做确定性数值构造。
"""
from __future__ import annotations

from typing import Sequence


def ltr_feature_rows(
    candidates: "Sequence[object]",
    dense_scores: "Sequence[float | None]",
    intent: object,
    feature_order: "Sequence[str]",
) -> "list[list[float]] | None":
    """按 `feature_order` 组织 9 维特征行（每候选一行）；intent 为 None 时返回 None。

    - `candidates`：候选序列（生产 = 存活集，离线 = 回放存活池），顺序即位置特征的基准；
    - `dense_scores`：每候选稠密余弦分，`None` 表示无向量分；
    - `feature_order`：打分器声明的特征顺序（生产传 `scorer.ltr_feature_order`，离线传
      `train_ltr.FEATURE_NAMES`）；含未知名 → KeyError——契约破裂必须响亮失败，不静默。
    """
    if intent is None:
        return None                          
    from .retriever import constraint_satisfied                                

    lex_raw = [float(getattr(c, "score", 0.0) or 0.0) for c in candidates]
    lex_max = max(lex_raw) if lex_raw else 0.0
    hard = getattr(intent, "constraints", None) or {}
    soft = getattr(intent, "preferred_constraints", None) or {}
    raw_required = getattr(intent, "has_raw_data_required", None) is True

    def _matched(record, dim: str) -> float:
        terms = list(hard.get(dim) or []) + list(soft.get(dim) or [])
        if not terms:
            return 0.0
        return 1.0 if constraint_satisfied(record, dim, terms) else 0.0

    rows: list[list[float]] = []
    for i, cand in enumerate(candidates):
        record = cand.record
        vec = dense_scores[i]
        feats = {
            "lex_norm": (lex_raw[i] / lex_max) if lex_max > 0 else 0.0,
            "vec_norm": 0.0 if vec is None else float(vec),
            "has_vec": 0.0 if vec is None else 1.0,
            "inv_pos": 1.0 / (i + 1),
            "matched_species": _matched(record, "species"),
            "matched_tissue": _matched(record, "tissue"),
            "matched_disease": _matched(record, "disease"),
            "matched_has_raw_data": 1.0 if (raw_required and record.has_raw_data is True) else 0.0,
            "raw_required": 1.0 if raw_required else 0.0,
        }
        rows.append([feats[name] for name in feature_order])
    return rows
