                       
"""候选 → 打分文本序列化的**单一真源**（自 `vector_recall` 抽离，行为逐位不变）。

序列化格式经消融确认（带标签结构化字段 + 描述截断 ≤400 明显优于裸串 + ≤240），
是基准选型胜出的输入形态；稠密索引（vector_index）、API 批处理（recall_api）与
本地打分（vector_recall）三处必须吃到同一形态，故独立成叶子（仅依赖 retriever 的
候选类型，不构成新环）。
"""
from __future__ import annotations

from .retriever import RetrievedCandidate


def candidate_text(cand: RetrievedCandidate) -> str:
    """候选序列化为打分文本：题名 + **带标签**的结构化字段 + 截断描述（≤400）。"""
    r = cand.record
    desc = (r.description or "").strip().replace("\n", " ")
    if len(desc) > 400:
        desc = desc[:400]
    return (f"{r.dataset_name} | 物种 {r.species} | 组织 {r.tissue} | 疾病 {r.disease} "
            f"| 平台 {r.platform_family} | 实验 {r.assay} | {desc}")
