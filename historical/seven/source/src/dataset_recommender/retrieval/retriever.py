"""
检索：hard_filter（保证 0% 违规）→ 存活集排序 → family 去重（不越界）→ 终检 → 解释字段。
无匹配/弃权 → 结构化无结果诊断（缺失约束 + 放宽建议）。

架构约束（与 Codex 辩论收敛）：
- 硬约束只在结构化字段上判定，先过滤后排序；语义/embedding 若启用也只排存活集，永不引入违规。
- 硬约束是存活集的常量，不参与排序权重，只进解释字段。
- family 上限严格执行（宁可少返，绝不超额）。
- 终检：选出后对每条重新核验全部硬约束（不变量守卫）。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from .normalizer import DatasetRecord, effective_has_raw_data, is_missing_value
from .query_parser import QueryIntent, relax_key_to_suppressed
from .synonyms import expand_term
from ..domain.descriptors import MATCH_EXACT
from ..domain.registry import domain_dimensions, get_domain

                                                                    
                                                                            


@dataclass(slots=True)
class RetrievedCandidate:
    record: DatasetRecord
    score: float
    matched_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class NoResultDiagnosis:
    kind: str                                                                 
    reason: str                           
    detail: str                           
    per_constraint_support: dict = field(default_factory=dict)                     
    relaxations: list[str] = field(default_factory=list)


def normalize_family_key(record: DatasetRecord) -> str:
    """家族标识（题根派生，与查询无关）。保留此函数名以兼容评测/旧调用。"""
    return record.family_id or record.dataset_name.strip().lower()


                                               
def _field_contains(field_text: str, terms: list[str]) -> bool:
    t = (field_text or "").lower()
    return any(term in t for term in terms if term)


def _dim_field_text(record: DatasetRecord, dim: str) -> str:
    """某硬约束维度对应的原始字段文本（供「是否已标注」判定 + 诚实降级用）。

    维度 → 字段文本的映射由活动领域包的 DimensionDescriptor 声明（旧写死 if 链的
    数据化）；未知维度返回 ""（与旧 if 链末尾行为一致）。
    """
    desc = get_domain().descriptor_for(dim)
    if desc is None:
        return ""
    return desc.text_of(record)


def _partial_capable_dims() -> frozenset:
    """声明了「取值集合可能不完整」的维度名（领域包描述符驱动）。"""
    return frozenset(d.name for d in get_domain().dimensions if d.partial_capable)


def _completeness_text(record: DatasetRecord, name: str) -> str:
    """完整性 tie-breaker 的计分文本：维度名走领域包描述符访问器，否则按记录字段名取。"""
    desc = get_domain().descriptor_for(name)
    if desc is not None:
        return desc.text_of(record)
    return getattr(record, name, "") or ""


                                                          
                                                         
                                       
def _dim_value_set_complete(record: DatasetRecord, dim: str) -> bool:
    """该维的取值集合是否**穷尽**。

    背景：`ingest_ebi_scea.py` 的 design 文件逐细胞、最大 453MB，只能按 Range 多点抽样（中位覆盖 2.5%、
    最差 0.1%），故在记录上写 `metadata_provenance.complete=False` 声明「取值可能不全」。

    没有 provenance 的记录（冻结 base/10x 与 cellxgene/hca/arrayexpress 三库）→ 沿用旧语义**视作完整**
    → 判定逐位不变。只有显式 `complete is False` 才算不完整（缺键/None/异常形状一律保守视作完整，
    避免因 provenance 形状漂移把全库降级成「无法核验」）。
    """
    if dim not in _partial_capable_dims():
        return True
    raw = record.raw if isinstance(record.raw, dict) else {}
    prov = raw.get("metadata_provenance")
    if not isinstance(prov, dict):
        return True
    return prov.get("complete") is not False


def _dim_field_present(record: DatasetRecord, dim: str) -> bool:
    """该维是否**已核验**：字段有真实取值 **且** 取值集合穷尽。否则＝无法核验该条件
    （诚实降级层据此把「不匹配」与「不知道」分开）。

    为什么不只看非空（2026-07-16 对抗评审证伪）：SCEA 补上抽样得来的 tissue 后，记录会从「不知道」
    静默升格成「已知」——实测 `tissue=alveolus` 查询下，人类肺图谱 `E-ANND-1` 命中 0、caveat 也不再
    报它、连 lenient 都捞不回；而修复前它至少还被算进「另有 160 条无法核验」。抽样值**命中**是可信
    证据（搜「肺」照常命中，走 `_field_contains`，不经本函数）；抽样值**不命中**不构成否证 → 必须
    仍报 caveat、仍可被 lenient 纳入。诚实层原本只有二态，这里补上第三态「已标注但值集不完整」。

    为什么不只看空字符串（2026-07-17 全盘审计）：判空曾是裸 `.strip()`，于是 `disease="unknown"`
    被当成「已核验的真取值」。冻结 base 767 条里这样的有 **298 条（38.9%，296 个 "unknown" + 2 个
    "n/a"）** → 搜「肿瘤」时 caveat 恒空、lenient 一条都捞不回，而同一界面的卡片对同一条记录显示
    「疾病：未说明」（introduction 认哨兵、这里不认）。**两个模块对同一条记录给出互相矛盾的结论**，
    且诚实降级层在它的旗舰语料 × disease 这一格上 100% 失效——正是它被造出来要消灭的静默判负。
    现改用 `normalizer.is_missing_value`（哨兵判定的单一真源，此前一个概念三套定义）。

    调用面（已 grep 核实，仅两处，均非默认路径）：`constraint_satisfied` 的 `lenient` 分支（opt-in，
    默认 False 时根本不调）与 `coverage_caveats`（官方评测/CLI 不调用）→ 默认检索与冻结 767 不受影响。
    """
    if is_missing_value(_dim_field_text(record, dim)):
        return False
    return _dim_value_set_complete(record, dim)


def constraint_satisfied(record: DatasetRecord, dim: str, terms: list[str], lenient: bool = False) -> bool:
    if not terms:
        return True
                                                  
                                                              
    if lenient and not _dim_field_present(record, dim):
        return True
    desc = get_domain().descriptor_for(dim)
    if desc is None:
                                                  
        return True
    text = desc.text_of(record)
    if desc.match == MATCH_EXACT:
                                                                   
        return text.lower() in terms
    return _field_contains(text, terms)


def passes_hard_filter(record: DatasetRecord, intent: QueryIntent) -> bool:
                                                         
    if getattr(intent, "parse_status", "executable") != "executable":
        return False
    lenient_dims = getattr(intent, "lenient_dims", None) or set()
    for dim in domain_dimensions():
                                                                    
                                                                                        
        inc = intent.constraints.get(dim)
                                                                     
        if inc and not constraint_satisfied(record, dim, inc, lenient=(dim in lenient_dims)):
            return False
        exc = intent.excluded_constraints.get(dim)
        if exc and constraint_satisfied(record, dim, exc):
            return False
                                                                                
                                                       
                             
    if (intent.has_raw_data_required is not None
            and effective_has_raw_data(record) is not intent.has_raw_data_required):
        return False
                                                               
                                                                
    if intent.date_from or intent.date_to:
        pub = str(record.raw.get("published_date", "")).strip()[:10] if isinstance(record.raw, dict) else ""
        if not pub:
            return False
        if intent.date_from and pub < intent.date_from:
            return False
        if intent.date_to and pub > intent.date_to:
            return False
    return True


                                                 
                      
                                                        
                                         
                                                                             
                                                    
                                
                                            
                                                            
def _facet_order() -> list[str]:
    return list(get_domain().facet_order)


def _facet_labels() -> dict:
    return {
        **get_domain().dim_labels_zh,                     
        "source": "数据来源", "has_raw_data": "原始数据", "year": "发表年份",
    }


def _facet_casefold() -> set:
    """按小写归并分面键的维度集合（领域包描述符 `casefold` 驱动；biodata = species/tissue/disease）。"""
    return {d.name for d in get_domain().dimensions if d.casefold}
                                                           
                                                            
                                                                                 
                                              
                                                                       


def _facet_source(record: DatasetRecord) -> str:
    raw = record.raw if isinstance(record.raw, dict) else {}
                                                     
                                                  
                                        
    return (str(raw.get("source") or "").strip()) or "未标注来源"


def _facet_year(record: DatasetRecord) -> str:
    raw = record.raw if isinstance(record.raw, dict) else {}
                                                          
                                                                 
    return str(raw.get("published_date", "")).strip()[:4]


def _facet_raw(dim: str, record: DatasetRecord) -> str:
    """记录在某维度上的**原始**取值（未归一大小写）。空/占位返回 ""。"""
    if dim == "source":
        v = _facet_source(record)
    elif dim == "has_raw_data":
        _raw_yes, _raw_no = get_domain().raw_data_labels
                                                  
        v = {True: _raw_yes, False: _raw_no}.get(effective_has_raw_data(record), "")
    elif dim == "year":
        v = _facet_year(record)
    else:
        desc = get_domain().descriptor_for(dim)
        if desc is None:
            return ""
        v = desc.facet_of(record)
    return "" if is_missing_value(v) else v


def facet_value(dim: str, record: DatasetRecord) -> str:
    """记录在某分面维度上的**归一化键**（用于分组计数 + 精确等值过滤）。
    自由文本维度按小写归并；空/占位返回 ""（不作为分面）。
    计数与过滤都走此键 → 「点了 N 条就正好剩 N 条」的一致性由此保证。"""
    v = _facet_raw(dim, record)
    if v and dim in _facet_casefold():
        return v.lower()
    return v


def record_passes_facets(record: DatasetRecord, facet_filters: list[dict] | None) -> bool:
    """分面过滤谓词：跨维度 AND，同维度多值 OR。空/非法过滤项忽略。"""
    if not facet_filters:
        return True
    by_dim: dict[str, set[str]] = {}
    for f in facet_filters:
        if not isinstance(f, dict):
            continue
        dim, val = f.get("dim"), f.get("value")
        if dim in _facet_labels() and isinstance(val, str) and val.strip():
            by_dim.setdefault(dim, set()).add(val)
    for dim, vals in by_dim.items():
        if facet_value(dim, record) not in vals:
            return False
    return True


                                              
                                                     
                                                                          
                                                            
                                                  
                                               
               
PREFERENCE_BOOST = 2.0

                                           
                                                 
_WORD_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _word_hit(token: str, text: str) -> bool:
    """`token` 是否作为**独立词**出现在 `text` 里（两侧不能紧邻 ASCII 字母数字）。

    判据与 `query_parser._alias_occurrences` 同源：中文不分词、不判边界；这里只用于 ASCII 短词。
    """
    if not token or not text:
        return False
    pat = _WORD_RE_CACHE.get(token)
    if pat is None:
        pat = re.compile(r"(?<![0-9a-z])" + re.escape(token) + r"(?![0-9a-z])")
        _WORD_RE_CACHE[token] = pat
    return bool(pat.search(text))


def _term_hits_text(term: str, text: str) -> bool:
    """单个查询自由词对一段文本的命中判定（短 ASCII 词边界 / 长词子串）。

    拆出供 `_rank_score` 与受控同义扩展复用，判据与 2026-07-22 之前逐字相同。
    """
    if not term or not text:
        return False
    if len(term) <= 3 and term.isascii():
        return _word_hit(term, text)
    return term in text


def _any_synonym_hits(term: str, text: str) -> bool:
    """受控同义扩展命中判定：原词或任一登记同义词命中即真（见 synonyms.py 维护规范）。

    保守性：`expand_term` 对未登记词返回 `(term,)` 单元素组 → 判定退化为原词命中，
    逐字等价于扩展前的行为；只有登记词目才会额外认它的同义写法。词边界/子串的
    判据随 `term` 自身走（同义组各成员按自己的长度/字符集判，不混用）。
    """
    return any(_term_hits_text(v, text) for v in expand_term(term))


class DatasetRetriever:
    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

                                                          
    def _rank_score(self, record: DatasetRecord, intent: QueryIntent) -> float:
        score = 0.0
        name = record.dataset_name.lower()
        desc = record.description.lower()
                                      
                                                                 
                                              
        for dim, targets in (intent.preferred_constraints or {}).items():
            if targets and constraint_satisfied(record, dim, targets):
                score += PREFERENCE_BOOST
        if intent.preferred_raw and record.has_raw_data is True:
            score += PREFERENCE_BOOST
        if intent.preferred_sources:
                                                        
            raw_src = str(record.raw.get("source") or "").strip() if isinstance(record.raw, dict) else ""
            if raw_src and _facet_source(record) in set(intent.preferred_sources):
                score += PREFERENCE_BOOST
        if intent.preferred_date_from or intent.preferred_date_to:
            pub_pref = str(record.raw.get("published_date", "")) if isinstance(record.raw, dict) else ""
            pub_pref = pub_pref.strip()[:10]
            if pub_pref and (not intent.preferred_date_from or pub_pref >= intent.preferred_date_from) \
                    and (not intent.preferred_date_to or pub_pref <= intent.preferred_date_to):
                score += PREFERENCE_BOOST
                                 
         
                                                             
                                                                     
                                                      
                                                                       
                                  
                                                                
                                                                 
                                                            
                                           
                                                        
                                                 
        for t in intent.free_text_terms:
            if not t:
                continue
            hit_name, hit_desc = _any_synonym_hits(t, name), _any_synonym_hits(t, desc)
            if hit_name:
                score += 1.0
            elif hit_desc:
                score += 0.3
                                  
        for dim in domain_dimensions():
            for term in intent.constraints.get(dim, []):
                if term and term in name:
                    score += 0.5
                                                               
                                                                    
                                                                             
                                                          
                                                                                        
                                                 
                                                                          
                                                                        
                                                     
                                                                                            
                                                              
                                                                  
        filled = sum(
            1 for name in get_domain().completeness_fields
            if (lambda v: bool(v) and v.strip().lower() not in ("", "unknown"))(
                _completeness_text(record, name))
            and _dim_value_set_complete(record, name)
        )
        score += 0.15 * filled
                         
        pub = str(record.raw.get("published_date", "")) if isinstance(record.raw, dict) else ""
        score += 0.4 if pub >= "2025" else (0.2 if pub >= "2024" else 0.0)
                                          
        try:
            n = int(str(record.count).replace(",", ""))
            if n > 0:
                                               
                score += 0.5 if n >= 50000 else n / 100000.0
        except (ValueError, TypeError):
            pass
        return score

    def _explain(self, record: DatasetRecord, intent: QueryIntent) -> tuple[list[str], list[str], str]:
        matched: list[str] = []
        missing: list[str] = []
        for dim in domain_dimensions():
            terms = intent.constraints.get(dim)
            if not terms:
                continue
            (matched if constraint_satisfied(record, dim, terms) else missing).append(dim)
        if intent.has_raw_data_required is not None:
            ok = effective_has_raw_data(record) is intent.has_raw_data_required
            (matched if ok else missing).append("has_raw_data")
                                 
        parts: list[str] = []
        disp = intent.display_map
        for dim in get_domain().explain_order:
            if dim in matched and disp.get(dim):
                parts.append("/".join(disp[dim]))
        reason = "匹配 " + "、".join(parts) if parts else "满足基本检索条件"
                                    
        ex_parts: list[str] = []
        for dim in get_domain().explain_order:
            ed = intent.excluded_display.get(dim)
            if ed:
                ex_parts.append("/".join(ed))
        if ex_parts:
            reason += "；排除 " + "、".join(ex_parts)
        if "has_raw_data" in matched:
            reason += "；且含 FASTQ 原始数据" if intent.has_raw_data_required is True else "；且不含 FASTQ 原始数据"
        reason += "。"
        return matched, missing, reason

    def _apply_family_cap(self, scored: list[RetrievedCandidate], limit: int, cap: int) -> list[RetrievedCandidate]:
        selected: list[RetrievedCandidate] = []
        counts: dict[str, int] = {}
        for cand in scored:
            key = normalize_family_key(cand.record)
            if counts.get(key, 0) >= cap:
                continue
            selected.append(cand)
            counts[key] = counts.get(key, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def retrieve(
        self,
        records: list[DatasetRecord],
        intent: QueryIntent,
        top_k: int | None = None,
        rerank_backend: str = "off",
        rerank_top_n: int = 12,
        llm_config: object | None = None,
        recall_backend: str = "off",
        recall_alpha: float = 0.5,
        recall_fusion: "str | None" = None,
        embedder: object | None = None,
        cross_scorer: object | None = None,
        facet_filters: list[dict] | None = None,
        execution_trace: dict | None = None,
    ) -> list[RetrievedCandidate]:
        """rerank_backend / recall_backend 均为**显式参数**（默认 "off"），刻意不在此读环境变量：
        官方评测入口不传这些参数 → 结构性走确定性路径，残留 env 无法泄漏进二元门。

        facet_filters（默认 None）：分面细化的精确等值后置过滤，只在硬过滤后收窄存活集，
        官方评测不传 → 整段 no-op、确定性零影响。

        recall_fusion（缺省 None → 按后端取默认）：dense/vector 召回的融合法（"linear"=min-max+α /
        "rrf"=名次融合 k=60 / "ltr"=学习排序融合）；dense/vector 缺省 "ltr"（工件缺失时
        fail-soft 回退 dense→linear、vector→rrf，与旧默认逐位一致）。
        官方评测不传（recall off）→ 召回块整体跳过、确定性零影响。"""
        limit = max(1, top_k if top_k is not None else self.top_k)
                                   
        if intent.abstain:
            return []
                                            
        survivors = [r for r in records if passes_hard_filter(r, intent)]
        if facet_filters:
            survivors = [r for r in survivors if record_passes_facets(r, facet_filters)]
        if not survivors:
            return []
                                                              
                                                             
                                                      
        scores = {id(r): self._rank_score(r, intent) for r in survivors}
        ranked = sorted(
            survivors,
            key=lambda r: (-scores[id(r)], r.family_id, r.dataset_name),
        )
        candidates: list[RetrievedCandidate] = []
        for r in ranked:
            matched, missing, reason = self._explain(r, intent)
            candidates.append(
                RetrievedCandidate(
                    record=r, score=round(scores[id(r)], 4),
                    matched_fields=matched, missing_fields=missing, reason=reason,
                )
            )
                                                       
                                                        
                                              
                                                                
                                                                      
                                                          
                         
        eff_recall_fusion = str(recall_fusion or "").strip().lower() or ("ltr" if recall_backend in ("dense", "vector") else "linear")
        if recall_backend and recall_backend != "off":
            from .vector_recall import recall_rerank                               

            candidates = recall_rerank(
                query=intent.original_query,
                candidates=candidates,
                backend=recall_backend,
                alpha=recall_alpha,
                fusion=eff_recall_fusion,
                embedder=embedder,
                cross_scorer=cross_scorer,
                intent=intent,
                trace=(execution_trace.setdefault("recall", {}) if execution_trace is not None else None),
            )
                                                              
                                                                      
                                                  
                                                                            
                                                                                   
                                                
                                                                                
        if rerank_backend and rerank_backend != "off":
            from .rerank import rerank_candidates                         

            pool_size = max(1, rerank_top_n, limit)
            pool = self._apply_family_cap(candidates, pool_size, cap=1)               
            pool_ids = {id(c) for c in pool}
            tail = [c for c in candidates if id(c) not in pool_ids]                         
            reranked_pool = rerank_candidates(
                query=intent.original_query,
                candidates=pool,
                backend=rerank_backend,
                top_k=None,
                config=llm_config,
                trace=(execution_trace.setdefault("rerank", {}) if execution_trace is not None else None),
            )
            candidates = reranked_pool + tail
                                                            
                                                    
                                                               
                                                                  
        if execution_trace is not None:
            ordered_uids = [str(c.record.raw.get("dataset_uid") or "") for c in candidates]
            canonical_order = json.dumps(ordered_uids, ensure_ascii=False, separators=(",", ":"))
            execution_trace["ranking_snapshot"] = {
                "schema": "biodata-ranking-snapshot/1",
                "feature_schema": "rule-rank-features/1",
                "parameters": {
                    "top_k": limit, "rerank_top_n": int(rerank_top_n),
                    "rerank_backend": str(rerank_backend or "off"),
                    "recall_backend": str(recall_backend or "off"),
                    "recall_alpha": float(recall_alpha), "recall_fusion": eff_recall_fusion,
                    "family_cap_primary": 1, "family_cap_fallback": 2,
                    "weights": {"preference": PREFERENCE_BOOST, "free_text_title": 1.0,
                                "free_text_description": 0.3, "constraint_title": 0.5,
                                "completeness_each": 0.15, "freshness_2025": 0.4,
                                "freshness_2024": 0.2, "sample_size_cap": 0.5},
                },
                "survivor_count": len(survivors), "candidate_count": len(candidates),
                "ordered_uids": ordered_uids,
                "ordered_uids_sha256": hashlib.sha256(canonical_order.encode("utf-8")).hexdigest(),
                "features": [
                    {"uid": str(c.record.raw.get("dataset_uid") or ""), "base_score": c.score,
                     "family_id": c.record.family_id, "matched_fields": c.matched_fields,
                     "missing_fields": c.missing_fields}
                    for c in candidates[:500]
                ],
                "features_truncated": len(candidates) > 500,
            }
                                                          
        selected = self._apply_family_cap(candidates, limit, cap=1)
        if len(selected) < limit:
            selected = self._apply_family_cap(candidates, limit, cap=2)
                                      
        selected = [c for c in selected if passes_hard_filter(c.record, intent)]
        return selected

                                 
    def explain_empty(self, records: list[DatasetRecord], intent: QueryIntent) -> NoResultDiagnosis:
        if getattr(intent, "parse_status", "executable") == "clarification_required":
            return NoResultDiagnosis(
                kind="clarification", reason=intent.clarification_reason,
                detail=intent.clarification_detail or "查询含义需要确认。",
            )
        if intent.abstain:
            return NoResultDiagnosis(
                kind="abstain", reason=intent.abstain_reason,
                detail=intent.abstain_detail or "查询无法安全解析，已弃权。",
            )
                                 
        support: dict[str, int] = {}
        active: dict[str, list[str]] = {d: intent.constraints[d] for d in domain_dimensions() if intent.constraints.get(d)}
        for dim, terms in active.items():
            support[dim] = sum(1 for r in records if constraint_satisfied(r, dim, terms))
        if intent.has_raw_data_required is not None:
            want = intent.has_raw_data_required
            support["has_raw_data"] = sum(1 for r in records if effective_has_raw_data(r) is want)
        for dim in domain_dimensions():
            exc = intent.excluded_constraints.get(dim)
            if exc:
                support[f"exclude:{dim}"] = sum(1 for r in records if not constraint_satisfied(r, dim, exc))
                                    
        zero = [d for d, n in support.items() if n == 0]
        disp = intent.display_map
        def label(key: str) -> str:
            if key == "has_raw_data":
                return "需要 FASTQ" if intent.has_raw_data_required else "不要 FASTQ"
            if key.startswith("exclude:"):
                d = key.split(":", 1)[1]
                return "排除 " + "/".join(intent.excluded_display.get(d, [d]))
            return "/".join(disp.get(key, [key]))
        relaxations: list[str] = []
        if zero:
            detail = "库中不存在满足这些条件的数据：" + "、".join(label(d) for d in zero) + "。"
            for d in zero:
                relaxations.append(f"去掉「{label(d)}」这个条件")
        else:
            detail = "各单项条件都有数据，但同时满足全部条件的组合在库中不存在。"
                             
            for d, _n in sorted(support.items(), key=lambda x: x[1])[:2]:
                relaxations.append(f"放宽「{label(d)}」")
        return NoResultDiagnosis(
            kind="no_match", reason="empty_intersection",
            detail=detail, per_constraint_support=support, relaxations=relaxations,
        )

                                             
    def relaxation_options(
        self,
        records: list[DatasetRecord],
        intent: QueryIntent,
        top_k: int | None = None,
        max_options: int = 8,
        include_dead: bool = False,
    ) -> list[dict]:
        """空交集（no_match）时试算「若放宽某处能救回多少 + top-k 预览」，回传多种放宽策略供用户自选。

        **两档策略**（每项带 `kind`，前端据它分组）：

        - `kind="drop"` **去掉一个条件**：逐个丢掉**一个**硬约束（各约束维度 + FASTQ 要求 +
          排除项 + 发表时间范围）。保守档——你说的其余条件全都还在。

        include_dead=True 时放松后仍零存活的维度也保留（count=0、candidates=[]）——
        「放松也无救」对环内 rerank 的探测报告是信息量；缺省 False 维持既有 no_match
        行为逐位不变。
        - `kind="only"` **只按一个条件搜**：只保留**一个**维度的正向约束，其余（含 FASTQ /
          时间范围 / 排除项）全放开。激进档，但它是第一档**全军覆没**时唯一还救得回来的一档——
          条件互相叠加得太死时，去掉任何单独一个仍然是 0 条，用户就会看到「无结果且无任何建议」。

        第二档仅在第一档的放宽目标 **≥3 个**时生成：恰好 2 个目标时「只按 A 搜」逐字等于「去掉 B」，
        两条一模一样的建议并排列出只会让人以为它们不同。

        - **弃权（abstain）不在此列**：弃权是"查询无法安全解析"，不是"约束太严"，返回空（前端只提示原因）。
        - 每个放宽项都用确定性 retrieve（rerank/recall 默认关）出预览 → 不引入违规、可复现。
        - 每项带 `suppress`：本项放松量的 filter_id 清单（`relax_key_to_suppressed` 单锚点），
          供放松卡零 LLM 重搜直接下发为 suppressed_constraints（装配侧负责与既有忽略合并）。
        - 只读，不改 intent（用 dataclasses.replace 克隆）；官方评测/CLI 不调用此方法，确定性零影响。
        """
        if intent.abstain:
            return []
        limit = max(1, top_k if top_k is not None else self.top_k)

        targets: list[tuple[str, str, QueryIntent]] = []
        for dim in domain_dimensions():
            if intent.constraints.get(dim):
                label = "/".join(intent.display_map.get(dim, [dim])) or dim
                relaxed = replace(intent, constraints={d: v for d, v in intent.constraints.items() if d != dim})
                targets.append((f"dim:{dim}", label, relaxed))
        if intent.has_raw_data_required is not None:
            lbl = "FASTQ 原始数据" if intent.has_raw_data_required else "『不要 FASTQ』条件"
            targets.append(("raw", lbl, replace(intent, has_raw_data_required=None)))
        for dim in domain_dimensions():
            if intent.excluded_constraints.get(dim):
                lbl = "排除 " + ("/".join(intent.excluded_display.get(dim, [dim])) or dim)
                relaxed = replace(intent, excluded_constraints={d: v for d, v in intent.excluded_constraints.items() if d != dim})
                targets.append((f"exclude:{dim}", lbl, relaxed))
        if intent.date_from or intent.date_to:
            a, b = intent.date_from[:4], intent.date_to[:4]
            rng = (a + " 起") if a and not b else ("截至 " + b) if b and not a else (a + "–" + b)
            targets.append(("date", "发表时间(" + rng + ")", replace(intent, date_from="", date_to="")))

        def measure(key: str, label: str, kind: str, relaxed: QueryIntent) -> dict | None:
            survivors = sum(1 for r in records if passes_hard_filter(r, relaxed))
            if not survivors and not include_dead:
                return None
            cands = self.retrieve(records, relaxed, top_k=limit) if survivors else []
            if survivors and not cands:
                return None
                                                                         
                                                 
            return {"key": key, "label": label, "kind": kind, "count": survivors,
                    "candidates": cands, "suppress": relax_key_to_suppressed(intent, key)}

        drops = [o for o in (measure(k, lb, "drop", rx) for k, lb, rx in targets) if o]
        drops.sort(key=lambda o: -o["count"])

                                                                    
                                                           
        onlys: list[dict] = []
        if len(targets) >= 3:
            for dim in domain_dimensions():
                if not intent.constraints.get(dim):
                    continue
                label = "/".join(intent.display_map.get(dim, [dim])) or dim
                kept = replace(
                    intent,
                    constraints={dim: intent.constraints[dim]},
                    excluded_constraints={},
                    has_raw_data_required=None,
                    date_from="",
                    date_to="",
                )
                opt = measure(f"only:{dim}", label, "only", kept)
                if opt:
                    onlys.append(opt)
            onlys.sort(key=lambda o: -o["count"])

                                                         
                                             
        head = drops[: max(0, max_options - min(len(onlys), 3))]
        return head + onlys[: max_options - len(head)]

                                                         
    def coverage_caveats(
        self, records: list[DatasetRecord], intent: QueryIntent, facet_filters: list[dict] | None = None
    ) -> list[dict]:
        """对每个**正向**硬约束维度 D，统计「满足其它全部约束、但 D 字段为空（无法核验）」的记录，
        按来源分组。语义＝『这些数据集满足你其它条件，但它们没标注 <D>，无法确认是否匹配』——本可能
        相关却被 fail-closed 静默判负的那批。让用户看见覆盖缺口、而非误以为「库里就这么点」。

        - 精确区分「不匹配」与「不知道」：记录 D 字段**有值但不同** → 属真实不匹配，不计入（诚实判负）；
          仅 D 字段**未标注/值集不全** → 计入（无法核验）。
        - **只数「你还没看见的」**：D 上抽样值已经命中、记录本就在结果列表里的，不得再计进「另有 N 条」
          （2026-07-17 修，见下方 `not constraint_satisfied` 守卫）。
        - **与当前存活集同口径**：若有活跃分面 `facet_filters`，unknown 集也套用 `record_passes_facets`
          （与 `matched_survivors` 同源）——保证 caveat 计数 == 点「也纳入」后真正新增数（前端重跑保留分面），
          不越过分面把用户已过滤掉的来源/记录算进「另有 N 条」。
        - 已被用户宽容（dim ∈ lenient_dims）的维度不再报 caveat（已纳入未标注项、缺口已消解）。
        - 只读：用 dataclasses.replace 克隆 intent，绝不改原 intent；官方评测/CLI 不调用此方法 →
          与 relaxation_options / facets 同为「结构性隔离冻结门」，确定性零影响。
        """
        if intent.abstain or getattr(intent, "parse_status", "executable") != "executable":
            return []
        lenient_dims = getattr(intent, "lenient_dims", None) or set()
        caveats: list[dict] = []
        for dim in domain_dimensions():
            terms = intent.constraints.get(dim)
            if not terms or dim in lenient_dims:
                continue
                                                            
            other = replace(intent, constraints={d: v for d, v in intent.constraints.items() if d != dim})
            unknown = [
                r for r in records
                if passes_hard_filter(r, other) and not _dim_field_present(r, dim)
                                                               
                                                                            
                                                                  
                                                                   
                                                                           
                                                                          
                and not constraint_satisfied(r, dim, terms)
                and (not facet_filters or record_passes_facets(r, facet_filters))
            ]
            if not unknown:
                continue
            by_source = Counter(_facet_source(r) for r in unknown)
            caveats.append({
                "dim": dim,
                                                                           
                "label": _facet_labels().get(dim, dim),
                "count": len(unknown),
                "by_source": [{"source": s, "count": c} for s, c in by_source.most_common()],
            })
        caveats.sort(key=lambda c: -c["count"])
        return caveats

                                
    def matched_survivors(
        self,
        records: list[DatasetRecord],
        intent: QueryIntent,
        facet_filters: list[dict] | None = None,
    ) -> list[DatasetRecord]:
        """当前查询（含分面过滤）的**全部**命中记录（未截断 top-k）。弃权 → 空。
        与 retrieve 第 1 步同源，供「库中匹配总数 + 分面分组」统计。"""
        if intent.abstain:
            return []
        out = [r for r in records if passes_hard_filter(r, intent)]
        if facet_filters:
            out = [r for r in out if record_passes_facets(r, facet_filters)]
        return out

    def facets(
        self,
        records: list[DatasetRecord],
        intent: QueryIntent,
        facet_filters: list[dict] | None = None,
        max_values: int = 6,
        max_groups: int = 6,
    ) -> dict:
        """有结果时，按**未被查询固定、也未被分面选中**的维度，统计存活集里各取值的条数，
        回传 {"total": 命中总数, "groups": [{dim,label,values:[{value,display,count}]}]}。

        - 只保留「至少 2 个不同取值」的维度（单值无收窄意义）。
        - facet_filters 已选中的维度整组不再出现（该维度已被收窄，靠前端面包屑撤销）。
        - 弃权/命中<2 → groups 空（无可细化）；官方评测不调用此方法，确定性零影响。
        """
        survivors = self.matched_survivors(records, intent, facet_filters)
        total = len(survivors)
        if total < 2:
            return {"total": total, "groups": []}
        pinned = {d for d in domain_dimensions() if intent.constraints.get(d)}
                                                                          
                                     
        if intent.has_raw_data_required is not None:
            pinned.add("has_raw_data")
        if intent.date_from or intent.date_to:
            pinned.add("year")
        active = {f.get("dim") for f in (facet_filters or []) if isinstance(f, dict)}
        groups: list[dict] = []
        for dim in _facet_order():
            if dim in pinned or dim in active:
                continue
            counter: dict[str, int] = {}
            disp: dict[str, dict[str, int]] = {}                                
            for r in survivors:
                v = facet_value(dim, r)
                if not v:
                    continue
                counter[v] = counter.get(v, 0) + 1
                raw = _facet_raw(dim, r) or v
                d = disp.setdefault(v, {})
                d[raw] = d.get(raw, 0) + 1
            if len(counter) < 2:
                continue
            if dim == "year":
                                                              
                top = sorted(counter.items(), key=lambda kv: kv[0], reverse=True)[:max_values]
            else:
                top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:max_values]
            values = []
            for v, n in top:
                display = max(disp[v].items(), key=lambda kv: (kv[1], kv[0]))[0] if disp.get(v) else v
                values.append({"value": v, "display": display, "count": n})
            groups.append({"dim": dim, "label": _facet_labels()[dim], "values": values})
            if len(groups) >= max_groups:
                break
        return {"total": total, "groups": groups}
