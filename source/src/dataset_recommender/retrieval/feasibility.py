                       
"""可行性概览：聚合本次候选的 Cells 数量与分布，不将不同计量单位混加。

数量区间取下界；缺失数量或非 Cells 单位不计入该合计。它不代表全库规模、
独立细胞总量或统计功效，多个数据集之间也可能重复采样。
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .normalizer import is_missing_value
from ..content.item_view import published_year


def _field(item: dict[str, Any], key: str) -> str:
    v = str(item.get(key) or "").strip()
    return "" if is_missing_value(v) else v


def _has_fastq(item: dict[str, Any]) -> bool:
    """是否**确认**含原始测序数据（FASTQ）。真源 = 结构化三态字段，绝不做字面子串判断。

    `raw_data_status` 是 dict（code/label/warning/note），它的 label/note 里天然带 "FASTQ"
    字样（note 还专门写着「要下 FASTQ 定论…别只看目录级信号」）——把整个 dict 转字符串找
    子串，会把 `no_fastq`（明确没有）与 `unknown`（我们没查）全算成「含 FASTQ」。
    unknown≠不满足：没查过就不许编码成有，也不许编码成没有（同一口径）。
    """
    if item.get("has_raw_data") is True:
        return True
    status = item.get("raw_data_status")
    if isinstance(status, dict):
        return str(status.get("code") or "") == "has_fastq"
    if isinstance(status, str):
                                                              
        return status.strip() == "has_fastq"
    return False


def _count_unit(item: dict[str, Any]) -> tuple[str, str]:
    raw = item.get("sample_size")
    if isinstance(raw, dict):
        c = str(raw.get("count") or "").strip()
        u = str(raw.get("unit") or "").strip()
        if c or u:
            return c, u
    return str(item.get("count") or "").strip(), str(item.get("unit") or "").strip()


def _as_int(text: str) -> int | None:
    """完整解析数量；区间逐端解析后取下界，不将标点两侧的数字拼接。"""
    parts = re.split(r"\s*(?:-|–|—|~|至|到)\s*", str(text or "").strip())
    if not 1 <= len(parts) <= 2:
        return None
    values = []
    for part in parts:
        match = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*([kKmM]?)", part)
        if match is None:
            return None
        try:
            scale = {"": 1, "k": 1000, "m": 1000000}[match.group(3).lower()]
            fraction = match.group(2) or ""
            denominator = 10 ** len(fraction)
            scaled = (int(match.group(1).replace(",", "")) * denominator
                      + int(fraction or "0")) * scale
            value, remainder = divmod(scaled, denominator)
            if remainder:
                return None                      
            values.append(value)
        except (ValueError, OverflowError):
            return None
    return min(values)


def build_report(survivors: list[dict[str, Any]], result_total: int, truncated: bool = False) -> dict[str, Any]:
    """通过硬过滤的候选（序列化 dict）→ 可行性概览。`result_total` 是未截断的命中总数；
    `truncated` 表示 survivors 是否只是命中集的一个前缀（top_k 截断）。"""
    n = len(survivors)
    total_cells = 0
    counted = 0
    no_count_sources: Counter = Counter()
    species_dist: Counter = Counter()
    platform_dist: Counter = Counter()
    year_dist: Counter = Counter()
    source_dist: Counter = Counter()
    downloadable = 0
    fastq = 0

    for s in survivors:
        src = _field(s, "source") or "未知来源"
        source_dist[src] += 1
        c, u = _count_unit(s)
        cell_n = _as_int(c) if u.strip().lower() == "cells" else None
        if cell_n is not None:
            total_cells += cell_n
            counted += 1
        else:
            no_count_sources[src] += 1
        for tok in [t.strip() for t in _field(s, "species").split(",") if t.strip()]:
            species_dist[tok] += 1
        plat = _field(s, "platform") or _field(s, "platform_family")
        if plat:
            platform_dist[plat] += 1
        yr = s.get("published_year")
        if yr is None:
            yr = published_year(_field(s, "published_date"))
        if yr:
            year_dist[str(yr)] += 1
        if _field(s, "download_url") or _field(s, "url"):
            downloadable += 1
        if _has_fastq(s):
            fastq += 1

    no_count = n - counted
    gaps: list[str] = []
    if no_count:
        by = "、".join(f"{src}（{c}）" for src, c in no_count_sources.most_common(5))
        gaps.append(
            f"{no_count}/{n} 个数据集未纳入 Cells 数量合计（{by}）：数量缺失、不可解析或单位不是 Cells。"
            "ArrayExpress 源库不提供研究级细胞数（这是来源本身不提供，不是漏抓）。"
        )
    if truncated:
        gaps.append(f"命中 {result_total} 条、本报告只统计了前 {n} 条；分布与总量是样本、非全量。")
    if n and downloadable < n:
        gaps.append(f"{n - downloadable}/{n} 个数据集没有公开访问链接，投稿前需到来源确认可获取性。")

    return {
        "candidate_count": result_total,
        "aggregated_count": n,
        "truncated": bool(truncated),
        "total_cells_lower_bound": total_cells,                            
        "datasets_with_cell_count": counted,
        "datasets_without_cell_count": no_count,
        "species": [{"value": k, "count": v} for k, v in species_dist.most_common(12)],
        "platforms": [{"value": k, "count": v} for k, v in platform_dist.most_common(12)],
        "years": [{"value": k, "count": v} for k, v in sorted(year_dist.items())],
        "sources": [{"value": k, "count": v} for k, v in source_dist.most_common()],
        "downloadable_count": downloadable,
        "fastq_count": fastq,
        "gaps": gaps,
        "caveat": (
            "本概览只统计**本次检索所选来源范围内**、通过硬过滤的候选。数量只合计 Cells（区间取下界），"
            "Spots、Nuclei 和未知单位不换算、不混加；未去重跨数据集的重复采样，不代表独立细胞总量。"
            "仅凭元数据无法判断统计功效，够不够需结合原始数据与实验设计评估。"
        ),
    }
