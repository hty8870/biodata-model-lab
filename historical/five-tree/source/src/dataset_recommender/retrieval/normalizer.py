from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..corpus.data_loader import RawRecord
from ..corpus.downloads import has_fastq as _fastq_actual
from ..corpus.sample_supplement import count_fill as _supp_count_fill, gene_count as _supp_gene_count
from ..domain.registry import get_domain as _get_domain


def _derive_vocab():
    """领域词表句柄（派生规则随活动领域包；biodata = `domains/biodata/vocabulary.py`）。

    缺对应函数的领域包优雅降级为空串——biodata 包三个函数齐全，行为逐位不变。
    """
    return _get_domain().vocab

                                                                             

                                                
  
                                       
                                                                                    
                                                                    
                                                                            
                                                       
                                          
  
                                                                 
                                                                                  
                            
  
                                                         
                                                                  
                                           
MISSING_VALUE_TOKENS: frozenset[str] = frozenset({
    "unknown",
    "not specified",
    "not provided",
    "n/a",
    "na",
    "none",
    "null",
    "-",
    "未说明",
    "未知",
})


def is_missing_value(text: object) -> bool:
    """该元数据文本是否等同于「未标注」（空、纯空白，或一个缺失哨兵）。

    只做 strip + casefold 后的**整串**比较：不做子串匹配，否则 `"unknown syndrome"`（一个真实病名）
    会被误判成未标注。
    """
    s = str(text or "").strip()
    return not s or s.lower() in MISSING_VALUE_TOKENS


@dataclass(slots=True)
class DatasetRecord:
    dataset_name: str
    species: str
    tissue: str
    disease: str
    chemistry: str
    count: str
    unit: str
    has_raw_data: bool | None
    url: str
    source_file: str
    description: str
    raw: dict[str, Any]
                
    platform_family: str = ""                                                               
    assay: str = ""                                                                       
    modality: str = ""                                                                             
    family_id: str = ""                                      
    gene_count: str = ""                                          
                                                  
                                                                   
    facets: dict[str, str] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return " | ".join(
            [
                self.dataset_name,
                self.species,
                self.tissue,
                self.disease,
                self.chemistry,
                self.description,
            ]
        ).lower()


                                                        
                                                        
_FAMILY_SUFFIX_PATTERNS = [
    re.compile(r"\bblock\s+[a-z0-9]+\s+section\s+[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bsection\s+[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\breplicate\s+[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\brep\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bsample\s+[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bslide\s+[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b(rep|section|sample)[\s_-]*\d+\b", re.IGNORECASE),
]


def derive_family_id(dataset_name: str) -> str:
    name = (dataset_name or "").lower()
    for pattern in _FAMILY_SUFFIX_PATTERNS:
        name = pattern.sub(" ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or (dataset_name or "").strip().lower()


FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "dataset_name": (
        "dataset_name",
        "name",
        "title",
        "dataset_title",
        "dataset",
    ),
    "species": ("species", "organism", "sample_species", "host"),
    "tissue": ("tissue", "organ", "tissue_type", "anatomical_site"),
    "disease": ("disease", "condition", "diagnosis", "phenotype", "disease_state"),
    "chemistry": (
        "chemistry",
        "technology",
        "assay",
        "platform",
        "library_prep",
        "sequencing_technology",
    ),
    "count": (
        "total_records",
        "count",
        "sample_size",
        "n_cells",
        "n_spots",
        "number_of_cells",
        "number_of_spots",
        "cells",
        "spots",
    ),
    "unit": ("unit", "count_unit", "measurement_unit"),
    "has_raw_data": ("has_raw_data", "raw_data", "raw_fastq", "has_fastq", "fastq_available"),
    "url": ("url", "download_url", "link", "dataset_url", "source_url"),
    "description": ("description", "summary", "desc", "abstract", "notes"),
}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _flatten_values(payload: Any, parent_key: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            compound_key = f"{parent_key}.{key}" if parent_key else str(key)
            flattened[compound_key] = value
            flattened.update(_flatten_values(value, compound_key))
    elif isinstance(payload, list):
        if payload and all(not isinstance(item, (list, dict)) for item in payload):
            flattened[parent_key] = ", ".join(str(item) for item in payload)
    return flattened


def _extract_value(record: dict[str, Any], field_name: str) -> Any:
    candidates = FIELD_CANDIDATES[field_name]
    lower_record = {str(k).lower(): v for k, v in record.items()}
    for candidate in candidates:
        if candidate in lower_record:
            return lower_record[candidate]

    flattened = _flatten_values(record)
    normalized_map = {_normalize_key(k): v for k, v in flattened.items()}
    for candidate in candidates:
        normalized = _normalize_key(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        return "" if stripped.lower() in {"none", "null", "nan"} else stripped
    return str(value).strip()


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_unit(value: Any) -> str:
    text = _stringify(value).lower()
    if not text:
        return ""
    if text in {"cells", "cell"}:
        return "Cells"
    if text in {"spots", "spot"}:
        return "Spots"
    if text in {"nuclei", "nucleus"}:
        return "Nuclei"
    return text.title()


def _parse_count_from_name(name: str) -> str:
    """
    Parse shorthand counts from dataset names, such as 7.5k / 10k.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", name)
    if not match:
        return ""
    number = float(match.group(1)) * 1000
    return str(int(number)) if math.isfinite(number) else ""


def _normalize_count(value: Any, dataset_name: str) -> str:
    text = _stringify(value)
    if text:
        compact = text.replace(",", "")
        if compact.isdigit():
            return compact
        return text
    return _parse_count_from_name(dataset_name)


def normalize_dataset_record(raw_record: dict[str, Any], source_file: str) -> DatasetRecord:
    dataset_name = _stringify(_extract_value(raw_record, "dataset_name"))
    species = _stringify(_extract_value(raw_record, "species"))
    tissue = _stringify(_extract_value(raw_record, "tissue"))
    disease = _stringify(_extract_value(raw_record, "disease"))
    chemistry = _stringify(_extract_value(raw_record, "chemistry"))
    _raw_count = _extract_value(raw_record, "count")
    count = _normalize_count(_raw_count, dataset_name)
    unit = _normalize_unit(_extract_value(raw_record, "unit"))
    has_raw_data = _normalize_bool(_extract_value(raw_record, "has_raw_data"))
                                                                   
                                                                     
    _uid = raw_record.get("dataset_uid") if isinstance(raw_record, dict) else None
    _actual = _fastq_actual(_uid)
    if _actual is not None:
        has_raw_data = _actual
    url = _stringify(_extract_value(raw_record, "url"))
                                                              
                                                              
                                                               
                                                                       
    _supp_key = _uid or url
    if _supp_count_fill(_supp_key) and not _stringify(_raw_count):
        count = _supp_count_fill(_supp_key)
    gene_count = _supp_gene_count(_supp_key)
    description = _stringify(_extract_value(raw_record, "description"))

    platform_raw = _stringify(raw_record.get("platform")) if isinstance(raw_record, dict) else ""
    _v = _derive_vocab()
    platform_family = getattr(_v, "derive_platform_family", lambda _x: "")(platform_raw)
    assay = getattr(_v, "derive_assay", lambda _x: "")(chemistry)
    modality = getattr(_v, "derive_modality", lambda _p, _c: "")(platform_family, chemistry)
    family_id = derive_family_id(dataset_name)
                                                              
                                             
    _raw_facets = raw_record.get("facets") if isinstance(raw_record, dict) else None
    facets = {str(k): str(v) for k, v in _raw_facets.items()} if isinstance(_raw_facets, dict) else {}

    return DatasetRecord(
        dataset_name=dataset_name,
        species=species,
        tissue=tissue,
        disease=disease,
        chemistry=chemistry,
        count=count,
        unit=unit,
        has_raw_data=has_raw_data,
        url=url,
        source_file=source_file,
        description=description,
        raw=raw_record,
        platform_family=platform_family,
        assay=assay,
        modality=modality,
        family_id=family_id,
        gene_count=gene_count,
        facets=facets,
    )


def normalize_records(raw_records: Iterable[RawRecord]) -> list[DatasetRecord]:
    normalized: list[DatasetRecord] = []
    for raw in raw_records:
        if not isinstance(raw.record, dict):
            continue
        normalized.append(normalize_dataset_record(raw.record, raw.source_file))
    return normalized


def effective_has_raw_data(record: DatasetRecord) -> "bool | None":
    """记录的有效 raw 三态：**猜测性 False → None**（unknown ≠ 已确认没有）。

    判据真源 = `corpus.corpus.raw_data_false_is_guess`（展示层 `workflow._raw_false_is_guess`
    同一真源、同一档：仅有 ArrayExpress 抓取占位的 False 算猜，逐条核验过的 AE 记录不算）。
    只影响猜测档：有据 False（10x 台账 / CELLxGENE / SCEA / HCA 库级事实）原样保留——
    冻结 base 口径逐位不变。惰性 import 避免 normalizer ↔ corpus.corpus 的模块环。"""
    if record.has_raw_data is not False:
        return record.has_raw_data
    from ..corpus.corpus import raw_data_false_is_guess
    if raw_data_false_is_guess({**record.raw, "has_raw_data": False}):
        return None
    return False
