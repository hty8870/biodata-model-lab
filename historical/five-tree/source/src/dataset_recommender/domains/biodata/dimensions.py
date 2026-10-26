"""biodata 的 6 个检索维度描述符。

逐项复刻骨肉分离前 `retriever._dim_field_text` / `constraint_satisfied` /
`_facet_raw` 写死 if 链的行为（含 assay 维「硬约束用 assay+chemistry 联合文本、
分面只用 assay 字段」的不对称），访问器 duck-typed、不 import 记录类。
维度顺序刻意保持旧 `query_parser.DIMENSIONS` 元组顺序。
"""
from __future__ import annotations

from typing import Any

from ...domain.descriptors import MATCH_EXACT, MATCH_SUBSTRING, DimensionDescriptor


def _technology_text(record: Any) -> str:
                                                            
    return f"{getattr(record, 'assay', '')} {getattr(record, 'chemistry', '')}".lower()


def _field(record: Any, name: str) -> str:
    return getattr(record, name, "") or ""


def _stripped(name: str):
    def _get(record: Any) -> str:
        return (_field(record, name) or "").strip()
    return _get


DIMENSIONS: tuple[DimensionDescriptor, ...] = (
    DimensionDescriptor(
        "species", "物种", lambda r: _field(r, "species"), MATCH_SUBSTRING,
        facet_text=_stripped("species"), casefold=True,
    ),
    DimensionDescriptor(
        "tissue", "组织", lambda r: _field(r, "tissue"), MATCH_SUBSTRING,
        partial_capable=True, facet_text=_stripped("tissue"), casefold=True,
    ),
    DimensionDescriptor(
        "disease", "疾病", lambda r: _field(r, "disease"), MATCH_SUBSTRING,
        partial_capable=True, facet_text=_stripped("disease"), casefold=True,
    ),
    DimensionDescriptor(
        "platform", "平台", lambda r: _field(r, "platform_family"), MATCH_EXACT,
        facet_text=_stripped("platform_family"),
    ),
    DimensionDescriptor(
        "assay", "技术", _technology_text, MATCH_SUBSTRING,
        facet_text=_stripped("assay"),
    ),
    DimensionDescriptor(
        "modality", "模态", lambda r: _field(r, "modality"), MATCH_EXACT,
        facet_text=_stripped("modality"),
    ),
)
