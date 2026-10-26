                       
"""共享查询入口解释：把数据来源专名从自然语言中安全解析为语料范围。

时间、生物实体与否定仍由 :mod:`query_parser` 负责；本模块只解决一个必须发生在装载语料之前的
问题：来源决定要读哪些库，因此不能等到 QueryIntent 创建后再处理。Web 与 MCP 都调用这里，避免
浏览器维护一套来源别名、MCP 又要求调用者手写精确来源名。

自动识别遵循保守原则：只认高辨识度专名；一旦句中出现排除/否定语义就完全不自动收窄；显式来源
池与查询点名冲突时尊重显式池且不删词，让后续 fail-closed 解析暴露矛盾，而不是静默改写意图。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from . import query_lexicon as V
from ..domain.registry import domain_catalog


                                                                                
                                                                 
                                               
 
                                                                    
                                                             
                                     
                                                   
                                                                                 
                                                         
                                       
SOURCE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CELLxGENE Discover", ("cellxgene discover", "cell x gene discover", "cellxgene", "cxg", "czi cell", "细胞图谱cellxgene")),
    ("Human Cell Atlas", ("human cell atlas", "人类细胞图谱", "人体细胞图谱", "hca")),
    ("ArrayExpress", ("arrayexpress", "array express", "array-express")),
                                                                  
                                                   
                                          
    ("ENCODE", ("encode project", "encodeproject", "encode portal", "encode 数据库", "encode 数据")),
    (
        "EBI Single Cell Expression Atlas",
        ("single cell expression atlas", "single-cell expression atlas", "ebi scea", "scea", "ebi 单细胞表达图谱"),
    ),
                                                         
                                                                       
                                                          
                                                      
    ("HuBMAP", ("hubmap consortium", "hubmap")),
    (
        "Broad Single Cell Portal",
        ("broad single cell portal", "single cell portal", "single-cell portal", "scp"),
    ),
    ("NCBI GEO", ("ncbi geo", "geo 数据库", "geo 数据", "geo")),
                                                              
                                                 
    ("Zenodo", ("zenodo 数据库", "zenodo 数据", "zenodo")),
                                                                            
                                                                
    ("refine.bio", ("refine.bio 数据库", "refine.bio 数据", "refine.bio", "refinebio", "refine bio")),
    ("10x Genomics", ("10x genomics", "tenx genomics", "10x 官方", "10x官方", "10x", "tenx")),
)

                                                                
                                           
                                                            
                                 
_SOURCE_NEG_VOCAB_CN = frozenset({"排除", "剔除", "去掉", "不含", "不包含", "不要", "不是", "不用", "非"})
_SOURCE_NEG_PREFIX_EXTRA_CN = ("除了", "除去", "不限于")
_SOURCE_NEG_VOCAB_EN = frozenset({"other than", "except", "exclude", "excluding", "without", "omit", "skip"})


def _source_negation_prefix_re() -> "re.Pattern[str]":
    cn = [m for m in V.NEG_MORPHEMES_CN if m in _SOURCE_NEG_VOCAB_CN] + list(_SOURCE_NEG_PREFIX_EXTRA_CN)
    en = [r"\s+".join(re.escape(p) for p in m.split())
          for m in V.NEG_MORPHEMES_EN if m in _SOURCE_NEG_VOCAB_EN]
    alts = sorted([re.escape(m) for m in cn] + en, key=len, reverse=True)                                
    return re.compile(r"(?:" + "|".join(alts) + r")\s*$", re.IGNORECASE)


SOURCE_NEGATION_PREFIX_RE = _source_negation_prefix_re()
SOURCE_NEGATION_SUFFIX_RE = re.compile(r"^\s*(?:以外|之外)", re.IGNORECASE)

                                                     
                                           
                                                          
                                                             
                                                         
                                                              
                                                                 
def _prefer_prefix_re() -> "re.Pattern[str]":
    def group(markers: "Sequence[str]", conn: str) -> str:
        alts = "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True))
        return rf"(?:{alts})\s*[{re.escape(conn)}]?\s*$"
    return re.compile(group(V.SOFT_PREFER_PREFIX_CN, V.PREFER_CONNECTOR_CHARS)
                      + "|" + group(V.HEDGE_PREFER_PREFIX_CN, V.HEDGE_CONNECTOR_CHARS))


SOURCE_PREFER_PREFIX_RE = _prefer_prefix_re()


                                                     
                                                               
_ASSAY_GUARD_WORDS: frozenset[str] | None = None
_ASSAY_GUARD_WORDS_CATALOG_ID: int | None = None


def _assay_guard_words() -> frozenset[str]:
    """「10X Multiome / 10X ATAC」这类**试剂盒名**的尾巴词：活动领域 catalog assay 维的
    ASCII 别名/target（单一真源，词表扩了自动跟随）。CJK 别名刻意不收——「10x 的多组学数据」
    里的「10x」仍按来源读（歧义句往「不静默改检索池」的保守方向倒）。"""
    global _ASSAY_GUARD_WORDS, _ASSAY_GUARD_WORDS_CATALOG_ID
    catalog = domain_catalog()
    if _ASSAY_GUARD_WORDS is None or _ASSAY_GUARD_WORDS_CATALOG_ID != id(catalog):
        _ASSAY_GUARD_WORDS = frozenset(
            str(alias).lower().strip()
            for entry in catalog.get("assay", [])
            for alias in list(entry.get("aliases", [])) + list(entry.get("targets", []))
            if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,}", str(alias).lower().strip())
        )
        _ASSAY_GUARD_WORDS_CATALOG_ID = id(catalog)
    return _ASSAY_GUARD_WORDS


def _token_pattern(token: str) -> re.Pattern[str]:
    escaped = re.escape(token)
    if any(ord(ch) > 127 for ch in token):
        return re.compile(escaped, re.IGNORECASE)
    guard = ""
    if token.lower() in ("10x", "tenx"):
                                                                              
                                                                
                                                                          
                                                                 
                                                                      
        assay = "|".join(sorted(_assay_guard_words(), key=len, reverse=True))
        if assay:
            guard = rf"(?!\s*-?\s*(?:{assay})(?![a-z0-9]))"
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])" + guard, re.IGNORECASE)


def _is_source_negated(query: str, pattern: re.Pattern[str]) -> bool:
    """只在否定词紧邻来源专名时触发守卫，避免“非小细胞肺癌”等合法实体误伤来源解析。"""
    match = pattern.search(query)
    if not match:
        return False
    before = query[max(0, match.start() - 24):match.start()]
    after = query[match.end():match.end() + 8]
    return bool(SOURCE_NEGATION_PREFIX_RE.search(before) or SOURCE_NEGATION_SUFFIX_RE.search(after))


def _preferred_source_spans(query: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """找出「<优先标记><来源专名>」的整段位置（含标记词）。空 = 该来源不是被「优先」修饰的。"""
    out: list[tuple[int, int]] = []
    for match in pattern.finditer(query):
        before = query[max(0, match.start() - 12):match.start()]
        m = SOURCE_PREFER_PREFIX_RE.search(before)
        if m:
            out.append((match.start() - (len(before) - m.start()), match.end()))
    return out


def source_alias_spans(query: str, known_sources: "Sequence[str] | None" = None) -> list[dict]:
    """只读：标出查询里每一段数据来源专名的位置。纯函数，不装载语料、不改任何解析行为。

    条件板（board.py）用它建立「来源保护区」：来源专名在 `resolve_search_request` 里是**先于**
    `parse_query` 被整段剥掉的（见本文件下方 `cleaned` 那段），所以任何在原句上做实体定位的模块
    如果不先把专名遮蔽，就会把「人类细胞图谱」里的「人类」当成物种约束——那既是凭空多出的条件，
    也会在改写时把专名打碎、静默换掉检索池。

    与 `resolve_search_request` 共用同一套 `_token_pattern`，保证两边认定的边界一致。
    `known_sources=None`（缺省）表示不按可用来源过滤：比服务端更宽，是 fail-closed 方向
    （宁可多保护一段，也不要漏保护）。

    返回 `[{source, alias, start, end}]`，按 start 升序、互不重叠（长 alias 优先占位，与正向解析的
    「长 alias 先消费」同构）。
    """
    text = str(query or "")
    known = set(known_sources) if known_sources is not None else None
    candidates: list[dict] = []
    for source, aliases in SOURCE_ALIASES:
        if known is not None and source not in known:
            continue
                                                                             
                                                           
                                              
                                                                                
        for alias in aliases:
            matches = list(_token_pattern(alias).finditer(text))
            if not matches:
                continue
            for match in matches:
                candidates.append(
                    {"source": source, "alias": alias, "start": match.start(), "end": match.end()}
                )
            break
    candidates.sort(key=lambda s: (s["end"] - s["start"], -s["start"]), reverse=True)
    occupied = [False] * (len(text) + 1)
    chosen: list[dict] = []
    for span in candidates:
        if any(occupied[span["start"]:span["end"]]):
            continue
        for pos in range(span["start"], span["end"]):
            occupied[pos] = True
        chosen.append(span)
    chosen.sort(key=lambda s: s["start"])
                                                              
                                           
                                              
                                                              
                                        
    prefix_len: dict[int, int] = {}
    for pos, span in enumerate(chosen):
        before = text[max(0, span["start"] - 12):span["start"]]
        m = SOURCE_PREFER_PREFIX_RE.search(before)
        prefix_len[pos] = (len(before) - m.start()) if m else 0
    by_source: dict[str, list[int]] = {}
    for pos, span in enumerate(chosen):
        by_source.setdefault(span["source"], []).append(pos)
    for positions in by_source.values():
        if all(prefix_len[p] > 0 for p in positions):
            for p in positions:
                chosen[p]["start"] -= prefix_len[p]
    return chosen


def mask_source_spans(query: str, spans: "Sequence[dict] | None" = None) -> str:
    """把来源专名整段替换成**等长**空格，索引与原句逐位对齐。

    等长遮蔽（而不是删字符）是有意的：后续实体定位、替换、贴回都按原句索引操作，删字符会让索引漂移，
    还可能让左右残字相邻拼出一个原本不存在的别名。
    """
    text = str(query or "")
    marks = list(spans) if spans is not None else source_alias_spans(text)
    if not marks:
        return text
    chars = list(text)
    for span in marks:
        for pos in range(max(0, int(span["start"])), min(len(chars), int(span["end"]))):
            chars[pos] = " "
    return "".join(chars)


@dataclass(frozen=True)
class SearchRequestResolution:
    original_query: str
    parsed_query: str
    sources: list[str] | None
    source_mode: str
    detected_sources: list[str] = field(default_factory=list)
    removed_terms: list[str] = field(default_factory=list)
    automatic_requested: bool = False
    automatic_skipped_reason: str | None = None
                                   
    preferred_sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "parsed_query": self.parsed_query,
            "sources": list(self.sources) if self.sources is not None else None,
            "source_mode": self.source_mode,
            "detected_sources": list(self.detected_sources),
            "removed_terms": list(self.removed_terms),
            "automatic_requested": self.automatic_requested,
            "automatic_skipped_reason": self.automatic_skipped_reason,
            "preferred_sources": list(self.preferred_sources),
        }


def resolve_search_request(
    query: str,
    sources: Sequence[str] | None,
    known_sources: Sequence[str],
    *,
    auto_parse_sources: bool = False,
) -> SearchRequestResolution:
    """解析来源专名并返回送入 query_parser 的查询。

    ``sources`` 是调用方允许的来源池。Web 自动模式传全部来源，识别到专名后在池内收窄；MCP
    默认传 None，未点名时仍保持基础语料默认，点名时才扩到对应公开来源。手动模式把
    ``auto_parse_sources`` 设为 False，完全尊重显式选择。
    """
    original = str(query or "")
    explicit = [str(x).strip() for x in (sources or []) if str(x).strip()] or None

    known = set(known_sources)
    matches: list[tuple[str, str, re.Pattern[str]]] = []
    for source, aliases in SOURCE_ALIASES:
        if source not in known:
            continue
        for alias in aliases:
            pattern = _token_pattern(alias)
            if pattern.search(original):
                matches.append((source, alias, pattern))
                break

                                     
                                               
    preferred_sources: list[str] = []
    prefer_cuts: list[tuple[int, int]] = []
    remaining: list[tuple[str, str, re.Pattern[str]]] = []
    for source, alias, pattern in matches:
        cuts = _preferred_source_spans(original, pattern)
        if cuts and len(cuts) == len(pattern.findall(original)):
            preferred_sources.append(source)
            prefer_cuts.extend(cuts)
        else:
            remaining.append((source, alias, pattern))
    matches = remaining
                                          
                                        
    base_query = re.sub(r"\s+", " ", mask_source_spans(
        original, [{"start": a, "end": b} for a, b in prefer_cuts])).strip() if prefer_cuts else original

                                              
                                                   
                                               
    if not auto_parse_sources:
        return SearchRequestResolution(
            original, base_query, explicit, "explicit" if explicit else "default",
            preferred_sources=preferred_sources,
        )

    detected = [source for source, _alias, _pattern in matches]
    if not detected:
        return SearchRequestResolution(
            original, base_query, explicit, "explicit" if explicit else "default",
            automatic_requested=True, preferred_sources=preferred_sources,
        )

                                                  
                                    
    if any(_is_source_negated(original, pattern) for _source, _alias, pattern in matches):
        return SearchRequestResolution(
            original,
            base_query,
            explicit,
            "explicit" if explicit else "default",
            detected_sources=detected,
            automatic_requested=True,
            automatic_skipped_reason="source_negation_guard",
            preferred_sources=preferred_sources,
        )

    effective = detected
    if explicit is not None:
        overlap = [source for source in detected if source in explicit]
        if not overlap:
            return SearchRequestResolution(
                original,
                base_query,
                explicit,
                "explicit_conflict",
                detected_sources=detected,
                automatic_requested=True,
                automatic_skipped_reason="detected_source_outside_explicit_pool",
                preferred_sources=preferred_sources,
            )
        effective = overlap

                                                     
                                                                               
                                                                                       
                                                            
                                                                          
                                                        
                                                           
    cleaned = base_query
    removed: list[str] = []
    for source, alias, pattern in matches:
        if source not in effective:
            continue
        cleaned, n = pattern.subn(" ", cleaned)
        if n:
            removed.append(alias)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return SearchRequestResolution(
        original,
        cleaned,
        effective,
        "auto_detected",
        detected_sources=effective,
        removed_terms=removed,
        automatic_requested=True,
        preferred_sources=preferred_sources,
    )
