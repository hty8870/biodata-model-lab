from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..llm.config import Settings, get_settings
from ..corpus.corpus import known_source_values, load_normalized_corpus, raw_data_false_is_guess
from ..corpus.downloads import all_real_urls, file_count, primary_url
from ..llm.llm_client import (
    ALLOWED_FINISH_REASONS, LLMConfig, call_llm, is_auth_error, is_complete_finish_reason,
    load_llm_config, should_use_llm,
)
from ..retrieval.normalizer import DatasetRecord
from ..llm.prompts import PROMPT_NAME, build_curator_prompt
from ..corpus import reachability as _reachability
from ..retrieval.query_parser import QueryIntent, active_filters, detect_action_markers, parse_query
from ..retrieval.retriever import (DatasetRetriever, RetrievedCandidate,
                        _facet_casefold, _facet_labels, _facet_source, passes_hard_filter)
from ..retrieval.units import UNIT_EXPLANATIONS, explain_unit, format_sample_size
from ..content.labels import RAW_FASTQ_NO, raw_fastq_status
from ..domain.registry import domain_dimensions


def __getattr__(name: str):
    """兼容 shim（PEP 562）：抑制白名单常量由活动领域维度集派生。biodata 默认逐位不变。"""
    if name == "SUPPRESSIBLE_DIMS":
        return _suppressible_dims()
    if name == "SUPPRESSIBLE_FILTER_IDS":
        return _suppressible_filter_ids()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


RAW_WARNING = (
    "**提示：标有 ❌ 无 FASTQ 的数据集仅包含分析结果或处理后文件，不支持重新从 FASTQ 跑完整流程。**"
)
TABLE_HEADER = "| 数据集名称 | 物种 | 组织 | 疾病 | 技术方案 | 样本量 | 原始数据状态 | 下载链接 |"
TABLE_SEPARATOR = "|---|---|---|---|---|---|---|---|"
LINK_PATTERN = re.compile(r"^\[点击下载\]\((https?://[^)\s]+)\)$")
                                   
                                                               
_LLM_PROMPT_CANDIDATE_CAP = 20
                                                          
_LLM_CURATOR_MAX_ROWS = 10
FALLBACK_PIPELINE = "query -> load_json -> normalize -> retrieve_top_k -> llm_rerank_or_fallback -> format_answer"
LLM_PIPELINE_SUCCESS = (
    "query -> load_json -> normalize -> retrieve_top_k -> build_curator_prompt -> call_llm -> validate_llm_answer -> format_answer"
)
LLM_PIPELINE_FALLBACK = (
    "query -> load_json -> normalize -> retrieve_top_k -> build_curator_prompt -> call_llm -> validate_llm_answer -> fallback_to_rule_based_format"
)


@dataclass(slots=True)
class WorkflowResult:
    answer: str
    pipeline: str
    llm_mode: str
    llm_attempted: bool
    llm_succeeded: bool
    llm_response_used: bool
    llm_provider: str | None = None
    model: str | None = None
    prompt_name: str | None = None
    fallback: str | None = None
    fallback_reason: str | None = None
    llm_called: bool = False
    retrieved_dataset_names: list[str] = field(default_factory=list)
    retrieved_urls: list[str] = field(default_factory=list)
                                                               
                                     
    retrieved_data: list[dict[str, object]] = field(default_factory=list)
                                                                         
                                             
    relaxation_options: list[dict[str, object]] = field(default_factory=list)
                                                              
                                                      
                                                                 
                                                                  
    degraded_search: dict[str, object] | None = None
                                                
                                                                  
                                                           
                                                           
    vector_fallback: dict[str, object] | None = None
                                                                                 
                                                  
    result_total: int = 0
    facets: list[dict[str, object]] = field(default_factory=list)
                                                     
                                                                                 
                                                                   
    coverage_caveats: list[dict[str, object]] = field(default_factory=list)
                                                     
                                                                   
                                                           
    unused_query_terms: list[str] = field(default_factory=list)
                                                                  
                                            
                                                               
    or_handling: dict = field(default_factory=dict)
                                                     
                                                   
    action_markers: list[str] = field(default_factory=list)
                                                     
    active_filters: list[dict[str, object]] = field(default_factory=list)
                                                                             
                                                                       
    resolution_status: str = "results"
                                                                                            
    clarification: dict | None = None
                                                 
                                                                                          
                                      
    strategy: dict | None = None
                                                      
    interpretation: dict = field(default_factory=dict)
    search_trace: dict = field(default_factory=dict)


def intent_projection(intent: QueryIntent) -> dict:
    """把 QueryIntent 投影成稳定、JSON 友好的公共结构，供 Web 与 MCP 共用。"""
    return {
        "constraints": intent.constraints,
        "excluded_constraints": intent.excluded_constraints,
        "display": intent.display_map,
        "excluded_display": intent.excluded_display,
        "active_filters": active_filters(intent),
        "has_raw_data_required": intent.has_raw_data_required,
                                                 
                                                       
        "preferred_constraints": intent.preferred_constraints,
        "preferred_display": intent.preferred_display,
        "preferred_raw": intent.preferred_raw,
        "preferred_sources": intent.preferred_sources,
        "preferred_date_from": intent.preferred_date_from,
        "preferred_date_to": intent.preferred_date_to,
        "free_text_terms": intent.free_text_terms,
        "date_from": intent.date_from,
        "date_to": intent.date_to,
        "parse_status": intent.parse_status,
        "abstain": intent.abstain,
        "abstain_reason": intent.abstain_reason,
        "abstain_detail": intent.abstain_detail,
                                                   
                                                              
        "unresolved_terms": list(intent.unresolved_terms or []),
        "clarification_reason": intent.clarification_reason,
        "clarification_detail": intent.clarification_detail,
        "clarification_options": intent.clarification_options,
                                           
                                        
                                                                   
        "or_handling": intent.or_handling,
    }


def _update_trace_step(
    trace: dict, step_id: str, status: str, detail: str, fallback_note: str | None = None
) -> None:
    for step in trace.get("steps", []):
        if step.get("id") == step_id:
            step.update({"status": status, "detail": detail})
                                                                   
                                               
            if status == "fallback" and fallback_note:
                step["fallback_note"] = fallback_note
            else:
                step.pop("fallback_note", None)
            trace["summary"] = " → ".join(
                str(x.get("label")) for x in trace.get("steps", []) if x.get("status") == "used"
            )
            return


_TRACE_REASON_LABELS = {
    "model_or_dependency_unavailable": "本地模型或运行依赖不可用",
    "runtime_error": "语义排序运行异常",
    "invalid_scores": "本地模型返回了无效分数",
    "invalid_vectors": "本地模型返回了无效向量",
    "llm_not_configured": "服务端还没有配置可用的 AI 接口",
    "llm_call_failed": "AI 接口调用失败或返回为空",
                                             
                                                  
    "llm_auth_failed": "API 密钥无效或没有权限（401/403），请到「设置」里检查并更新密钥",
                                                 
                                                 
    "llm_unavailable_or_empty": "AI 服务不可用或没有返回有效内容",
    "invalid_order": "AI 返回的排序格式无效",
    "invalid_llm_answer": "AI 说明没通过反捏造校验",
}

                                                       
                                            
                                                    
_FALLBACK_MEANS_NOT_ENABLED = frozenset({"model_or_dependency_unavailable", "llm_not_configured"})


def _fallback_note(step: dict) -> str:
    """回退措辞的**单一真源**。

    诚实红线：`status="fallback"` 有两种完全不同的因由，**用户可见的措辞必须分开**——
    「未启用」是一个选择，「没能完成」是一次故障。 抓到前端把两者合成
    「本次未启用」：provider 真返 400 的那几天，摘要句读起来像是「系统自己决定没用这一层」。
    所以这句话由后端出、前端一个字都不自己编（`search_trace.steps[].fallback_note`）。
    """
    reason = str(step.get("reason") or "")
    label = _TRACE_REASON_LABELS.get(reason, "执行未成功")
    prefix = "未启用" if reason in _FALLBACK_MEANS_NOT_ENABLED else "没能完成"
    return f"{prefix}：{label}"


def _trace_duration_suffix(step: dict) -> str:
    """给真实执行/回退步骤追加可读耗时；禁用步骤不制造“0 毫秒”噪声。"""
    try:
        duration_ms = max(0, int(step.get("duration_ms", 0)))
    except (TypeError, ValueError):
        return ""
    if duration_ms <= 0:
        return ""
    if duration_ms < 1000:
        return f"（耗时 {duration_ms} 毫秒）"
    return f"（耗时 {duration_ms / 1000:.1f} 秒）"


def _trace_fallback_detail(step: dict, fallback_target: str) -> str:
    reason = _TRACE_REASON_LABELS.get(str(step.get("reason", "")), "执行未成功")
    return f"已尝试，但因{reason}回退到{fallback_target}{_trace_duration_suffix(step)}。"


def _finalize_search_trace(trace: dict, started_at: float) -> None:
    trace["total_duration_ms"] = max(0, int(round((time.perf_counter() - started_at) * 1000)))


def _build_search_trace(resolution, intent: QueryIntent, execution: dict, decision, result_total: int) -> dict:
    source_status = "used"
    if resolution.automatic_skipped_reason:
        source_status = "fallback"
    if resolution.source_mode == "auto_detected":
        source_detail = "自动识别并限定到：" + "、".join(resolution.detected_sources)
    elif resolution.source_mode == "explicit_conflict":
        source_detail = "查询中的来源与手动范围冲突，保留手动范围且未静默删词。"
    elif resolution.automatic_skipped_reason == "source_negation_guard":
        source_detail = "检测到来源排除语义，安全跳过自动收窄。"
    elif resolution.automatic_requested:
        source_detail = "自动识别已开启；未识别到特定来源，检索当前可用来源。"
    elif resolution.source_mode == "explicit":
        source_detail = "使用手动选择的数据来源。"
    else:
        source_detail = "未点名来源，使用调用方默认范围。"

    filters = active_filters(intent)
    parse_detail = "未识别到硬条件。" if not filters else "识别：" + "；".join(
        f"{x.get('label')}={','.join(str(v) for v in x.get('values', []))}" for x in filters
    )
    if intent.parse_status != "executable":
        parse_detail = intent.abstain_detail or intent.clarification_detail or parse_detail

    recall = execution.get("recall", {})
    rerank = execution.get("rerank", {})
    recall_status = recall.get("status", "skipped")
    rerank_status = rerank.get("status", "skipped")
    fallback_layers = []
    if recall_status == "fallback":
        fallback_layers.append("语义排序")
    if rerank_status == "fallback":
        fallback_layers.append("AI 候选排序")
    actual_layers = []
    if recall_status == "used":
        actual_layers.append("语义排序")
    if rerank_status == "used":
        actual_layers.append("AI 候选排序")
    final_method = " + ".join(actual_layers) if actual_layers else "规则排序"
    if decision is None:
        if fallback_layers:
            strategy_detail = f"使用手动选择的排序设置；{'、'.join(fallback_layers)}发生回退，最终采用{final_method}。"
        else:
            strategy_detail = "使用手动选择的排序设置。"
    else:
        n = int(decision.signals.get("n_survivors", result_total))
        if fallback_layers:
            planned_layers = []
            if decision.recall_backend != "off":
                planned_layers.append("语义排序")
            if decision.rerank_backend == "llm":
                planned_layers.append("AI 候选排序")
            strategy_detail = (
                f"自动策略原计划使用{' + '.join(planned_layers or fallback_layers)}；"
                f"{'、'.join(fallback_layers)}发生回退，最终采用{final_method}。"
            )
        elif decision.tier == "precise":
            strategy_detail = f"筛选后只有 {n} 条候选，规则顺序已经足够，不额外重排。"
        elif recall_status == "used" and rerank_status == "used":
            strategy_detail = f"候选较多且语义偏好丰富，先做语义排序，再用 AI 精排有限候选。"
        elif recall_status == "used":
            strategy_detail = f"筛选后有 {n} 条候选，使用语义排序提高相关性。"
        elif rerank_status == "used":
            strategy_detail = "本地语义模型未采用，实际使用获准的 AI 对候选重新排序。"
        else:
            strategy_detail = f"筛选后有 {n} 条候选，本地语义模型或 AI 授权未就绪，保持规则顺序。"
    steps = [
        {"id": "source_parse", "label": "数据来源", "status": source_status, "detail": source_detail},
        {"id": "constraint_parse", "label": "自动解析条件", "status": "used", "detail": parse_detail},
        {
            "id": "hard_filter", "label": "必选条件筛选",
            "status": "used" if intent.parse_status == "executable" else "skipped",
            "detail": f"筛选后命中 {result_total} 条；后续排序不会加入不满足必选条件的数据。",
        },
        {
            "id": "rule_rank", "label": "规则排序", "status": "used" if result_total else "skipped",
            "detail": "按词面相关性、完整度、新鲜度和样本量生成确定性基础顺序。" if result_total else "没有候选可排序。",
        },
        {
            "id": "local_semantic", "label": "语义排序",
            "status": recall.get("status", "skipped"),
            "detail": (
                f"已对候选完成本地语义重排{_trace_duration_suffix(recall)}。" if recall.get("status") == "used"
                else (_trace_fallback_detail(recall, "规则顺序") if recall.get("status") == "fallback" else "本次未使用。")
            ),
        },
        {
            "id": "llm_rerank", "label": "AI 候选排序",
            "status": rerank.get("status", "skipped"),
            "detail": (
                f"AI 已对有限候选池重新排序{_trace_duration_suffix(rerank)}。" if rerank.get("status") == "used"
                else (_trace_fallback_detail(rerank, "此前顺序") if rerank.get("status") == "fallback" else "本次未使用。")
            ),
        },
        {"id": "llm_polish", "label": "AI 说明润色", "status": "pending", "detail": "等待最终格式化状态。"},
        {
            "id": "final_guard", "label": "去重与终检", "status": "used" if result_total else "skipped",
            "detail": "限制同族重复，并再次核验每条硬条件。" if result_total else "没有结果需要终检。",
        },
    ]
                                                  
                                                
                                                              
    for step, sink in (("local_semantic", recall), ("llm_rerank", rerank)):
        for entry in steps:
            if entry["id"] == step and entry["status"] == "fallback":
                entry["fallback_note"] = _fallback_note(sink)
    used = [x["label"] for x in steps if x["status"] == "used" and x["id"] != "llm_polish"]
    return {
        "version": 1,
        "automatic": bool(resolution.automatic_requested or decision is not None),
        "summary": " → ".join(used),
        "strategy_reason": strategy_detail,
        "steps": steps,
                                                                   
                                                      
        "ranking_snapshot": execution.get("ranking_snapshot"),
    }


                                                      
                                                                          
                                                    
def _suppressible_dims() -> frozenset:
    return frozenset(domain_dimensions()) | {"has_raw_data", "date"}


def _suppressible_filter_ids() -> frozenset:
    return (
        frozenset(f"include:{d}" for d in domain_dimensions())
        | frozenset(f"exclude:{d}" for d in domain_dimensions())
        | {"raw:required", "raw:forbidden", "date:range"}
                                                       
                                                                    
                                             
        | frozenset(f"prefer:{d}" for d in domain_dimensions())
        | {"prefer:raw", "prefer:source", "prefer:date"}
    )


                                                                  
                                                                                  
                                                                       
                                                         
                                               
                                                                          
                                
def sanitize_facet_filters(raw: object) -> list[dict]:
    """把调用方传来的分面过滤项收敛为 [{dim,value}]：只保留合法维度 + 非空字符串值，去重、限量。
    非法/越界一律丢弃（安全默认）；分面维度白名单取自 retriever._facet_labels()（活动领域包）。

    自由文本维度（biodata：物种/组织/疾病，由领域包描述符 casefold 声明）的 value 归一为小写——
    与 retriever.facet_value 的分面键完全对齐：直连调用方即便传 'Homo sapiens'（原始大小写）也能命中。"""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        dim = str(item.get("dim", "")).strip()
        value = str(item.get("value", "")).strip()
        if dim in _facet_casefold():
            value = value.lower()
        if dim in _facet_labels() and value and (dim, value) not in seen:
            seen.add((dim, value))
            out.append({"dim": dim, "value": value})
        if len(out) >= 12:                        
            break
    return out


def sanitize_suppressed(raw: object) -> list[str]:
    """把调用方传来的「被忽略的命中筛选项」收敛为白名单：接受**极性 filter_id**
    （include:<dim>/exclude:<dim>/raw:required/raw:forbidden/date:range）与**裸 dim**（旧调用兼容），
    与 active_filters()[].filter_id 对齐。非法/越界一律丢弃（安全默认）；缺省/空 → [] → 整段 no-op。"""
    allowed = _suppressible_dims() | _suppressible_filter_ids()
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        dim = str(item).strip()
        if dim in allowed and dim not in seen:
            seen.add(dim)
            out.append(dim)
        if len(out) >= 16:                              
            break
    return out


def sanitize_lenient_dims(raw: object) -> set[str]:
    """把调用方传来的「宽容维度」（诚实降级：字段为空视作通过）收敛为 DIMENSIONS 白名单集合。
    非法/未知维度一律丢弃（安全默认）；缺省/空 → 空集 → passes_hard_filter 逐位 no-op（官方评测不传）。"""
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {d for d in (str(x).strip() for x in raw) if d in domain_dimensions()}


def apply_explicit_date_range(intent: QueryIntent, date_from: str, date_to: str) -> None:
    """把调用方显式传入的发表时间范围盖到 intent 上（就地改）。

    显式传入的时间范围（网页上的年份下拉框）优先；**未传时保留** parse_query 从自然语言里解析出的范围
    （否则 MCP / CLI 这类不传 date 的路径会把「2022年以后」解析出的范围清空）。空串=不限 → 官方评测 no-op。

    从 `_prepare_context` 里原样抽出，供 board.py 复用同一份判据——两份手抄必然漂移，这是本仓库
    已经吃过好几次的亏（最近一次是交付集与 gitignore 两份清单从未对账）。
    """
    if (date_from or "").strip():
        intent.date_from = date_from.strip()
    if (date_to or "").strip():
        intent.date_to = date_to.strip()


def apply_suppressed_constraints(intent: QueryIntent, dims: "list[str] | None") -> None:
    """就地从已解析 intent 抹掉被抑制的**硬约束**——实现「命中筛选项可自由忽略」。

    支持**极性 filter_id**：`include:<dim>`（抹正向）/ `exclude:<dim>`（抹负向）/ `raw:required` /
    `raw:forbidden` / `date:range`；**裸 dim**（旧调用）= 该维正负全抑制。这样 include Human 与
    exclude Mouse 同为 species 时，忽略其一不会连坐另一个（否则会误删）。

    抑制在 `parse_query` 之后、检索之前生效；`dims` 缺省 None/空 → 完全 no-op（官方评测/MCP/CLI 不传）。
    """
    if not dims:
        return
    for raw in dims:
        s = str(raw).strip()
        if not s:
            continue
        pol, _, dim = s.partition(":") if ":" in s else ("both", "", s)
                                                                   
                                                  
                                              
        if pol == "prefer":
            if dim == "raw":
                intent.preferred_raw = None
            elif dim == "source":
                intent.preferred_sources = []
            elif dim == "date":
                intent.preferred_date_from = ""
                intent.preferred_date_to = ""
            elif dim in domain_dimensions():
                intent.preferred_constraints.pop(dim, None)
                intent.preferred_display.pop(dim, None)
            continue
                
        if s in ("raw:required", "raw:forbidden", "has_raw_data") or dim == "has_raw_data":
            intent.has_raw_data_required = None
            continue
              
        if s == "date:range" or dim == "date":
            intent.date_from = ""
            intent.date_to = ""
            continue
                    
        if dim in domain_dimensions():
            if pol in ("include", "both"):
                intent.constraints.pop(dim, None)
                intent.display_map.pop(dim, None)
            if pol in ("exclude", "both"):
                intent.excluded_constraints.pop(dim, None)
                intent.excluded_display.pop(dim, None)
                                                      
                                         
            if pol == "both":
                intent.preferred_constraints.pop(dim, None)
                intent.preferred_display.pop(dim, None)


def _safe_cell(text: str) -> str:
    return (text or "").replace("|", r"\|").strip()


def _format_sample_size(record: DatasetRecord) -> str:
                                                                   
    return format_sample_size(record.count, record.unit)


def _raw_false_is_guess(record: DatasetRecord) -> bool:
    """record 的 has_raw_data=False 是否为抓取时的猜测占位——
    判定真源在 corpus.corpus.raw_data_false_is_guess（冻结路径不得 import provenance，
    见 test_provenance.py 闭包隔离钉）；展示时带生效值（normalizer 可能用台账实测覆盖过索引值）。"""
    return raw_data_false_is_guess({**record.raw, "has_raw_data": record.has_raw_data})


def _raw_status(record: DatasetRecord) -> str:
    return raw_fastq_status(record.has_raw_data, guessed_false=_raw_false_is_guess(record))


                                            
_RAW_STATUS_NOTE = "目录级信号；unknown≠不满足；要下 FASTQ 定论请用 dataset_uid 调 get_file_manifest。"


def _raw_data_status(record: DatasetRecord) -> dict[str, object]:
    """把 has_raw_data 的三态语义上移成结构化字段（code/label/warning/authoritative/note）。
    label 逐字等于 _raw_status(record)；warning 仅 no_fastq 时=RAW_WARNING。
    猜测性 False（ArrayExpress 抓取占位）归 unknown——「我们没查」不许编码成「它没有」。"""
    if record.has_raw_data is True:
        code = "has_fastq"
    elif record.has_raw_data is False and not _raw_false_is_guess(record):
        code = "no_fastq"
    else:
        code = "unknown"
    return {
        "code": code,
        "label": _raw_status(record),
        "warning": RAW_WARNING if code == "no_fastq" else None,
        "authoritative": False,
        "note": _RAW_STATUS_NOTE,
    }


def _same_hard_filter(a: QueryIntent, b: QueryIntent) -> bool:
    """Whether two intents produce the same hard-filter survivor set.

    This deterministic comparator belongs to the Agent's ``search.rerun``
    adoption gate. It is not the retired LLM audit channel: free text and
    ranking order deliberately do not participate.
    """
    if bool(a.abstain) != bool(b.abstain):
        return False
    if a.abstain and b.abstain:
        return True

    def _norm(constraints: dict) -> dict:
        return {dim: frozenset(values or ())
                for dim, values in (constraints or {}).items() if values}

    return (
        _norm(a.constraints) == _norm(b.constraints)
        and _norm(a.excluded_constraints) == _norm(b.excluded_constraints)
        and a.has_raw_data_required == b.has_raw_data_required
        and (a.date_from or "") == (b.date_from or "")
        and (a.date_to or "") == (b.date_to or "")
    )


def _build_unit_explanation(candidates: list[RetrievedCandidate]) -> str:
                                                                       
                                                                
    present = {candidate.record.unit.strip().lower() for candidate in candidates if candidate.record.unit}
    explanations = [explain_unit(unit) for unit in UNIT_EXPLANATIONS if unit in present]
    return "\n".join(exp for exp in explanations if exp)


def _format_translated_keywords(intent: QueryIntent) -> str:
    structured_parts: list[str] = []
    for dim in ("species", "tissue", "disease", "modality", "platform", "assay"):
        disp = intent.display_map.get(dim)
        if disp:
            structured_parts.append(f"{dim}={'/'.join(disp)}")
    for dim in ("species", "tissue", "disease", "modality", "platform", "assay"):
        ed = intent.excluded_display.get(dim)
        if ed:
            structured_parts.append(f"exclude {dim}={'/'.join(ed)}")
    if intent.has_raw_data_required is True:
        structured_parts.append("raw_data=FASTQ / raw data")
    elif intent.has_raw_data_required is False:
        structured_parts.append("raw_data=no FASTQ")

    if structured_parts:
        return "; ".join(structured_parts)
    if intent.free_text_terms:
        return " ".join(term.strip() for term in intent.free_text_terms if term.strip())
    return intent.original_query.strip()


def format_candidates_markdown(candidates: list[RetrievedCandidate]) -> str:
    header = [TABLE_HEADER, TABLE_SEPARATOR]
    rows: list[str] = []
    has_false_fastq = False

    for candidate in candidates:
        record = candidate.record
        raw_status = _raw_status(record)
        if raw_status == RAW_FASTQ_NO:
            has_false_fastq = True

                                                            
        real_link = primary_url(record.raw.get("dataset_uid")) or record.url
        link = f"[点击下载]({real_link})" if real_link else "-"
        rows.append(
            "| {dataset_name} | {species} | {tissue} | {disease} | {chemistry} | {sample_size} | {raw_status} | {url} |".format(
                dataset_name=_safe_cell(record.dataset_name) or "-",
                species=_safe_cell(record.species) or "-",
                tissue=_safe_cell(record.tissue) or "-",
                disease=_safe_cell(record.disease) or "-",
                chemistry=_safe_cell(record.chemistry) or "-",
                sample_size=_safe_cell(_format_sample_size(record)),
                raw_status=raw_status,
                url=link,
            )
        )

    blocks = ["\n".join(header + rows)]
    if has_false_fastq:
        blocks.append(RAW_WARNING)

                              
    reason_lines = [
        f"- **{_safe_cell(c.record.dataset_name)}**：{c.reason}"
        for c in candidates
        if getattr(c, "reason", "")
    ]
    if reason_lines:
        blocks.append("**推荐理由：**\n" + "\n".join(reason_lines))

    unit_explanation = _build_unit_explanation(candidates)
    if unit_explanation:
        blocks.append(unit_explanation)
    return "\n\n".join(blocks)


def build_not_found_message(intent: QueryIntent) -> str:
    translated = _format_translated_keywords(intent)
    return f"抱歉，数据库中未检索到符合【{translated}】条件的数据。"


def render_no_result(diagnosis, intent: QueryIntent) -> str:
    """把结构化无结果诊断渲染成可读文案（弃权 / 空交集 + 放宽建议）。"""
    if diagnosis is None:
        return build_not_found_message(intent)
                                        
    if getattr(diagnosis, "kind", "") == "clarification":
        lines = ["需要确认 FASTQ 条件：" + (diagnosis.detail or "")]
        for opt in intent.clarification_options:
            rw = opt.get("rewrite")
            hint = f"（改写为：{rw}）" if rw else "（删除该条件后重试）"
            lines.append(f"- {opt.get('label', '')}{hint}")
        return "\n".join(lines)
    header = "抱歉，" + (diagnosis.detail or "未检索到符合条件的数据。")
    lines = [header]
    if diagnosis.relaxations:
        lines.append("可尝试放宽为：")
        for i, item in enumerate(diagnosis.relaxations, 1):
            lines.append(f"{i}. {item}")
    return "\n".join(lines)


def _record_download_url(record: DatasetRecord) -> str:
    """下载直链优先级：10x 阶段二 by_uid 直链 → 外部平台库自带 download_url → 数据集页面 url。
    10x 记录 raw 无 download_url 键 → 与历史行为一致（primary_url 命中或回退 record.url）。"""
    raw = record.raw if isinstance(record.raw, dict) else {}
    return primary_url(raw.get("dataset_uid")) or str(raw.get("download_url") or "") or record.url


def _record_source(record: DatasetRecord) -> str:
    """数据来源标签（前端徽章）。外部平台库记录自带 source；基础语料默认 10x Genomics。"""
    raw = record.raw if isinstance(record.raw, dict) else {}
    return str(raw.get("source") or "").strip() or "10x Genomics"


def _serialize_retrieved_data(candidates: list[RetrievedCandidate]) -> tuple[str, list[dict[str, object]]]:
    payload = [
        {
            "dataset_name": candidate.record.dataset_name,
            "species": candidate.record.species,
            "tissue": candidate.record.tissue,
            "disease": candidate.record.disease,
            "chemistry": candidate.record.chemistry,
            "platform_family": candidate.record.platform_family,
            "assay": candidate.record.assay,
            "count": candidate.record.count,
            "unit": candidate.record.unit,
                                                          
            "gene_count": candidate.record.gene_count,
            "has_raw_data": candidate.record.has_raw_data,
                                                 
            "raw_data_status": _raw_data_status(candidate.record),
            "sample_size": {
                "count": candidate.record.count,
                "unit": candidate.record.unit,
                "display": format_sample_size(candidate.record.count, candidate.record.unit),
                "unit_explanation": explain_unit(candidate.record.unit),
            },
            "published_date": candidate.record.raw.get("published_date", "") if isinstance(candidate.record.raw, dict) else "",
            "url": candidate.record.url,
            "source": _record_source(candidate.record),
                                                      
            "download_url": _record_download_url(candidate.record),
                                                                                   
            "reachability": _reachability.classify(_record_download_url(candidate.record)),
                                                                   
            "dataset_uid": candidate.record.raw.get("dataset_uid") or "",
            "n_files": file_count(candidate.record.raw.get("dataset_uid")),
                                                                                 
                                                                             
                                                                         
                                                                                 
            "collection_doi": (candidate.record.raw.get("collection_doi") or "") if isinstance(candidate.record.raw, dict) else "",
            "description": candidate.record.description,
            "preservation_method": candidate.record.raw.get("preservation_method", "") if isinstance(candidate.record.raw, dict) else "",
            "analysis_software": candidate.record.raw.get("analysis_software", "") if isinstance(candidate.record.raw, dict) else "",
            "software_version": (candidate.record.raw.get("analysis_software_version", "") or candidate.record.raw.get("software_version", "")) if isinstance(candidate.record.raw, dict) else "",
            "source_file": candidate.record.source_file,
            "score": round(candidate.score, 4),
            "matched_fields": candidate.matched_fields,
            "missing_fields": candidate.missing_fields,
            "reason": candidate.reason,
        }
        for candidate in candidates
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2), payload


def _parse_table_rows(answer: str) -> tuple[list[list[str]], str | None]:
    lines = answer.splitlines()
    header_index = -1
    for idx, line in enumerate(lines):
        if line.strip() == TABLE_HEADER:
            header_index = idx
            break
    if header_index < 0:
        return [], "LLM response missing required Markdown table header"

    separator_index = -1
    for idx in range(header_index + 1, len(lines)):
        if not lines[idx].strip():
            continue
        separator_index = idx
        break
    if separator_index < 0 or lines[separator_index].strip() != TABLE_SEPARATOR:
        return [], "LLM response missing Markdown table separator"

    rows: list[list[str]] = []
    for idx in range(separator_index + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            if rows:
                break
            continue
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        if not stripped.endswith("|"):
            return [], f"incomplete markdown table row at line {idx + 1}"
        cols = [part.strip() for part in stripped.split("|")[1:-1]]
        if len(cols) != 8:
            return [], f"invalid column count ({len(cols)}) in table row at line {idx + 1}"
        rows.append(cols)

    if not rows:
        return [], "LLM response does not contain valid table rows"

    return rows, None


def _extract_markdown_url(cell: str) -> str | None:
    match = LINK_PATTERN.fullmatch(cell.strip())
    if not match:
        return None
    return match.group(1).strip()


def _normalize_url(url: str) -> str:
    normalized = (url or "").strip()
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def _has_unclosed_download_link(text: str) -> bool:
    opened = len(re.findall(r"\[点击下载\]\(", text))
    closed = len(re.findall(r"\[点击下载\]\(https?://[^)\s]+\)", text))
    if opened != closed:
        return True
    return bool(re.search(r"\[点击下载\]\([^\)]*$", text))


def validate_llm_answer(
    answer: str,
    candidates: list[RetrievedCandidate],
    intent: QueryIntent,
    finish_reason: str | None = None,
) -> tuple[bool, str | None]:
    if not answer or not answer.strip():
        return False, "LLM returned empty response"
    if not is_complete_finish_reason(finish_reason):
        return False, f"invalid finish_reason: {finish_reason}"

    if TABLE_HEADER not in answer:
        return False, "LLM response missing required Markdown table header"
    if _has_unclosed_download_link(answer):
        return False, "incomplete markdown link detected"

    rows, parse_error = _parse_table_rows(answer)
    if parse_error:
        return False, parse_error

    candidate_names = {candidate.record.dataset_name.strip().lower() for candidate in candidates}
                                              
                                                          
                                                           
                                                  
    links_by_name: dict[str, set[str]] = {}
    for candidate in candidates:
        name = candidate.record.dataset_name.strip().lower()
        key = candidate.record.raw.get("dataset_uid") or candidate.record.url
        allowed = links_by_name.setdefault(name, set())
        if candidate.record.url:
            allowed.add(_normalize_url(candidate.record.url))
        primary = primary_url(key)
        if primary:
            allowed.add(_normalize_url(primary))
                                                
        ext_dl = candidate.record.raw.get("download_url") if isinstance(candidate.record.raw, dict) else ""
        if ext_dl:
            allowed.add(_normalize_url(str(ext_dl)))
        allowed |= {_normalize_url(u) for u in all_real_urls(key)}

    for row in rows:
        dataset_name = row[0].strip().lower()
        sample_size = row[5].strip()
        url = _extract_markdown_url(row[7])

        if sample_size in {"Cells", "Spots", "Nuclei"}:
            return False, f"invalid sample size format: {sample_size}"

        if dataset_name not in candidate_names:
            return False, f"LLM introduced unknown dataset name: {row[0]}"

        if not url:
            return False, f"invalid markdown download link: {row[7]}"

        if _normalize_url(url) not in links_by_name.get(dataset_name, set()):
            return False, f"LLM linked dataset '{row[0]}' to a URL that is not its own: {url}"

    if intent.has_raw_data_required is True and "❌ 无 FASTQ" in answer:
        return False, "LLM response violates FASTQ-required constraint"
    if intent.has_raw_data_required is False and "✅ 包含 FASTQ" in answer:
        return False, "LLM response violates FASTQ-forbidden constraint"

    return True, None


def strip_terms(query: str, terms: "list[str]") -> str:
    """从原句里挖掉若干词（大小写不敏感，长词先挖）。残差词取自 lower 后的工作串，
    原句大小写可能不同，所以不能用朴素 `str.replace`。"""
    live = [t for t in terms if t and t.strip()]
    if not live:
        return query
    pattern = "|".join(re.escape(t) for t in sorted(live, key=len, reverse=True))
    return re.sub(pattern, " ", query, flags=re.IGNORECASE)


def _collect_candidate_debug(candidates: list[RetrievedCandidate]) -> tuple[list[str], list[str]]:
    names = [candidate.record.dataset_name for candidate in candidates]
    urls = [candidate.record.url for candidate in candidates if candidate.record.url]
    return names, urls


                                                                                   
                                                           
                                                                   
                                                                       


def _llm_call_would_happen(config: "LLMConfig | None") -> bool:
    """「这一次会真实发出 LLM 请求」的判据（与 llm_client 调用闸 `should_use_llm` 同源）：
    mock / 未启用 / 无 key 一律为否——这些情况下调用方内部直接回退、零 provider 请求，
    因此不构成任何计费/准入事件。只读判定、无副作用、不读取环境。"""
    return bool(config is not None and should_use_llm(config)[0])


                                                                         
@dataclass
class RecommendParams:
    """`run_with_meta` 的全部检索杠杆（单一参数真源）。

    背景：原 `run_with_meta` 是 24 个 kwargs 的平铺签名，仓内 13 个调用点各自手拼子集
    已在漂（task-pack 不传 strategy、feasibility 不传 rerank——刻意还是遗漏无从约束），
    每加一个杠杆都要改一串调用点。本对象把字段与默认值收进一处；调用方 `RecommendParams(**kwargs)`
    即可机械迁移，dict 也能直接解包构造（turn.py/agent_exec.py 的既有封装透传）。
    `run_with_meta` 同步提供 `**kwargs` 兼容通道承接存量测试调用（见其 docstring），
    但新调用一律走参数对象。

    纪律：纯参数容器——不做校验（校验在接口层 `app/request_validation.py`）、不含行为；
    默认值与原 kwargs 签名逐位一致（行为零变化，冻结评测钉死）。
    """

    query: str
    top_k: int | None = None
    use_llm: bool | None = None
    mock_llm: bool | None = None
    provider: str | None = None
    rerank_backend: str = "off"
    rerank_top_n: int | None = None
    recall_backend: str = "off"
    recall_alpha: float | None = None
    date_from: str = ""
    date_to: str = ""
    sources: "list[str] | None" = None
    auto_parse_sources: bool = False
    facet_filters: "list[dict] | None" = None
    suppressed_constraints: "list[str] | None" = None
    lenient_dims: "list[str] | None" = None
    strategy: str = "fixed"
    recall_available: "bool | None" = None
    llm_available: "bool | None" = None
    preferred_recall: str = "cross_encoder"
    base_llm_config: LLMConfig | None = None


class DatasetRecommendationWorkflow:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = DatasetRetriever(top_k=self.settings.top_k)

    def _effective_llm_config(
        self,
        use_llm: bool | None = None,
        mock_llm: bool | None = None,
        provider: str | None = None,
        *,
        base: LLMConfig | None = None,
    ) -> LLMConfig:
        """取当前生效的 LLM 连接配置。

        `base`：调用方在 ENV_LOCK 内已物化的请求级配置——提供时**不再**读
        os.environ（锁外读 env 会拿到别人的请求覆盖甚至服务端默认），只在其上叠加
        use_llm/mock_llm/provider 参数；缺省时行为逐位不变（读 env 自解析，供无请求级
        覆盖的调用方：MCP/CLI/评测/测试）。"""
        if base is not None:
            config = LLMConfig(**asdict(base))
        else:
            config = load_llm_config(
                project_root=self.settings.project_root,
                provider_override=provider,
            )
            config = LLMConfig(**asdict(config))
        if use_llm is not None:
            config.enable_llm = use_llm
        if mock_llm is not None:
            config.mock_llm = mock_llm
        if provider is not None:
            config.provider = provider.strip().lower()
        return config

                                                      
                               
    _VECTOR_FALLBACK_EXCLUDED = frozenset({"identifier_direct", "identifier_fragment", "action_only"})

    def _prepare_context(
        self,
        query: str,
        top_k: int | None = None,
        rerank_backend: str = "off",
        rerank_llm_config: LLMConfig | None = None,
        rerank_top_n: int | None = None,
        recall_backend: str = "off",
        recall_alpha: float | None = None,
        date_from: str = "",
        date_to: str = "",
        sources: "list[str] | None" = None,
        auto_parse_sources: bool = False,
        facet_filters: "list[dict] | None" = None,
        suppressed_constraints: "list[str] | None" = None,
        lenient_dims: "list[str] | None" = None,
        strategy: str = "fixed",
        recall_available: bool = False,
        llm_available: bool = False,
        preferred_recall: str = "cross_encoder",
        before_llm: "Callable[[LLMConfig], None] | None" = None,
    ) -> tuple:
                                                                         
                                                         
        from ..retrieval.search_request import resolve_search_request
        resolution = resolve_search_request(
            query,
            sources,
            known_source_values(self.settings.data_dir, self.settings.project_root),
            auto_parse_sources=auto_parse_sources,
        )
                                                
                                                      
                                                       
                                               
                                           
        load_sources = _load_sources_for(resolution, auto_parse_sources=auto_parse_sources)
        normalized = load_normalized_corpus(
            self.settings.data_dir, self.settings.project_root, load_sources
        )
        intent = parse_query(resolution.parsed_query, self.settings.keyword_mapping)
                                                              
                                                     
                                               
        _pool_sources = {_facet_source(r) for r in normalized}
        _live = [s for s in resolution.preferred_sources if s in _pool_sources]
        _dead = [s for s in resolution.preferred_sources if s not in _pool_sources]
        intent.preferred_sources = _live
        if _dead:
            intent.unused_query_terms = list(intent.unused_query_terms) + _dead
        apply_explicit_date_range(intent, date_from, date_to)
                                                            
        apply_suppressed_constraints(intent, suppressed_constraints)
                                                                        
        intent.lenient_dims = sanitize_lenient_dims(lenient_dims)
                                                                         
                                                           
                                                                
        decision = None
        eff_recall_backend = recall_backend
        eff_rerank_backend = rerank_backend
        ranking_records = normalized
        survivor_records: list[DatasetRecord] | None = None
        if str(strategy or "fixed").strip().lower() == "auto":
            from ..retrieval.strategy import classify_strategy
                                                                           
                                                                             
                                                     
                                                                          
                                                                          
                                            
            survivor_records = self.retriever.matched_survivors(normalized, intent, facet_filters)
            ranking_records = survivor_records
            decision = classify_strategy(
                intent, survivor_records,
                recall_available=recall_available,
                llm_available=llm_available,
                preferred_recall=preferred_recall,
                top_k=(top_k if top_k is not None else self.settings.top_k),
                n_survivors=len(survivor_records),
            )
            eff_recall_backend = decision.recall_backend
            eff_rerank_backend = decision.rerank_backend
                                                                                    
        retrieve_kwargs: dict[str, object] = dict(
            top_k=top_k, rerank_backend=eff_rerank_backend, llm_config=rerank_llm_config,
            recall_backend=eff_recall_backend,
        )
        execution: dict[str, object] = {}
        retrieve_kwargs["execution_trace"] = execution
        if rerank_top_n is not None:
            retrieve_kwargs["rerank_top_n"] = rerank_top_n
        if recall_alpha is not None:
            retrieve_kwargs["recall_alpha"] = recall_alpha
                                                                    
        if facet_filters:
            retrieve_kwargs["facet_filters"] = facet_filters
                                                        
                                                         
                                                                          
                                                                   
                              
        if (before_llm is not None
                and str(eff_rerank_backend or "").strip().lower() == "llm"
                and _llm_call_would_happen(rerank_llm_config)):
                                                                 
                                                  
            _rerank_survivors = (survivor_records if survivor_records is not None
                                 else self.retriever.matched_survivors(normalized, intent, facet_filters))
            if _rerank_survivors:
                before_llm(rerank_llm_config)
                                                                           
        candidates = self.retriever.retrieve(ranking_records, intent, **retrieve_kwargs)
        retrieved_dataset_names, retrieved_urls = _collect_candidate_debug(candidates)
        return (
            intent, candidates, normalized, retrieved_dataset_names, retrieved_urls,
            decision, resolution, execution,
        )

    def build_prompt_preview(self, query: str, top_k: int | None = None) -> str:
        intent, candidates, _, _, _, _, _, _ = self._prepare_context(query=query, top_k=top_k)
        retrieved_data_text, _ = _serialize_retrieved_data(candidates[:_LLM_PROMPT_CANDIDATE_CAP])
        translated_keywords = _format_translated_keywords(intent)
        return build_curator_prompt(
            user_query=query,
            retrieved_data=retrieved_data_text,
            translated_keywords=translated_keywords,
            max_rows=min(len(candidates[:_LLM_PROMPT_CANDIDATE_CAP]), _LLM_CURATOR_MAX_ROWS),
            n_shown=len(candidates[:_LLM_PROMPT_CANDIDATE_CAP]),
            n_total=len(candidates),
        )

    def run(
        self,
        query: str,
        top_k: int | None = None,
        use_llm: bool | None = None,
        mock_llm: bool | None = None,
        provider: str | None = None,
        rerank_backend: str = "off",
        rerank_top_n: int | None = None,
        recall_backend: str = "off",
        recall_alpha: float | None = None,
        strategy: str = "fixed",
        recall_available: "bool | None" = None,
        llm_available: "bool | None" = None,
        preferred_recall: str = "cross_encoder",
    ) -> str:
                                                                           
        return self.run_with_meta(
            query=query,
            top_k=top_k,
            use_llm=use_llm,
            mock_llm=mock_llm,
            provider=provider,
            rerank_backend=rerank_backend,
            rerank_top_n=rerank_top_n,
            recall_backend=recall_backend,
            recall_alpha=recall_alpha,
            strategy=strategy,
            recall_available=recall_available,
            llm_available=llm_available,
            preferred_recall=preferred_recall,
        ).answer

    def run_with_meta(
        self,
        p: "RecommendParams | None" = None,
        *,
        before_llm: "Callable[[LLMConfig], None] | None" = None,
        **kwargs,
    ) -> WorkflowResult:
                                                          
                                                                      
                                                             
                                                          
                                                                
         
                                                                 
                                                                      
                                                               
                                                              
                                                             
                    
        if p is None:
            p = RecommendParams(**kwargs)
        elif kwargs:
            raise TypeError("run_with_meta: 已传 RecommendParams 时不要再传 kwargs")
        before_llm_fired = False

        def _before_llm_once(config: LLMConfig) -> None:
            nonlocal before_llm_fired
            if before_llm is None or before_llm_fired:
                return
            before_llm_fired = True
            before_llm(config)

        _before_llm_hook = _before_llm_once if before_llm is not None else None
        query = p.query
        top_k = p.top_k
        use_llm = p.use_llm
        mock_llm = p.mock_llm
        provider = p.provider
        rerank_backend = p.rerank_backend
        rerank_top_n = p.rerank_top_n
        recall_backend = p.recall_backend
        recall_alpha = p.recall_alpha
        date_from = p.date_from
        date_to = p.date_to
        sources = p.sources
        auto_parse_sources = p.auto_parse_sources
        facet_filters = p.facet_filters
        suppressed_constraints = p.suppressed_constraints
        lenient_dims = p.lenient_dims
        strategy = p.strategy
        recall_available = p.recall_available
        llm_available = p.llm_available
        preferred_recall = p.preferred_recall
        base_llm_config = p.base_llm_config
        workflow_started_at = time.perf_counter()
                                                 
                                                                                              
        auto = str(strategy or "fixed").strip().lower() == "auto"
        rerank_llm_config: LLMConfig | None = None
                                                                         
                                                        
                                                                 
                                                         
                                                          
                                                          
                               
        want_rerank_llm = bool(rerank_backend and rerank_backend != "off")
        if want_rerank_llm or (auto and llm_available is not False):
            rerank_llm_config = self._effective_llm_config(
                use_llm=True, mock_llm=False, provider=provider, base=base_llm_config,
            )
                                                        
                                                                                       
                                                                                             
                                                     
        eff_recall_available = recall_available
        eff_llm_available = llm_available
        if auto:
            if eff_llm_available is None:
                eff_llm_available = bool(rerank_llm_config and getattr(rerank_llm_config, "api_key", None))
            if eff_recall_available is None:
                from ..retrieval.vector_recall import recall_backend_ready
                eff_recall_available = recall_backend_ready(preferred_recall)
                                                               
                                                          
        ctx_kwargs: dict[str, object] = dict(
            top_k=top_k,
            rerank_backend=rerank_backend,
            rerank_llm_config=rerank_llm_config,
            rerank_top_n=rerank_top_n,
            recall_backend=recall_backend,
            recall_alpha=recall_alpha,
            date_from=date_from,
            date_to=date_to,
            sources=sources,
            auto_parse_sources=auto_parse_sources,
            facet_filters=facet_filters,
            suppressed_constraints=suppressed_constraints,
            lenient_dims=lenient_dims,
            strategy=strategy,
            recall_available=bool(eff_recall_available),
            llm_available=bool(eff_llm_available),
            preferred_recall=preferred_recall,
        )
        (intent, candidates, normalized, retrieved_dataset_names,
         retrieved_urls, decision, resolution, execution) = self._prepare_context(
            query=query, before_llm=_before_llm_hook, **ctx_kwargs
        )

                                                    
                                                        
                                                          
                                                   
                                                                      
                                                  
                                                  
                                  
        auto_degraded: dict[str, object] | None = None
        if intent.abstain and intent.abstain_reason == "unresolved_term":
            _dq = strip_terms(
                intent.original_query, [t for t in intent.unresolved_terms if t.strip()]
            ).strip()
            _dq = " ".join(_dq.split())
            if _dq and _dq != intent.original_query.strip():
                (_r_intent, _r_cands, _r_norm, _r_names, _r_urls,
                 _r_decision, _r_resolution, _r_execution) = self._prepare_context(
                    query=_dq, before_llm=_before_llm_hook, **ctx_kwargs
                )
                _has_condition = (
                    any(_r_intent.constraints.get(d) for d in domain_dimensions())
                    or any(_r_intent.excluded_constraints.get(d) for d in domain_dimensions())
                    or _r_intent.has_raw_data_required is not None
                    or bool(_r_intent.date_from or _r_intent.date_to)
                )
                if (_r_intent.parse_status == "executable" and not _r_intent.abstain
                        and _r_cands and _has_condition):
                    auto_degraded = {
                        "ignored_terms": [t for t in intent.unresolved_terms if t.strip()],
                        "query": _dq,
                        "active_filters": active_filters(_r_intent),
                    }
                    (intent, candidates, normalized, retrieved_dataset_names,
                     retrieved_urls, decision, resolution, execution) = (
                        _r_intent, _r_cands, _r_norm, _r_names, _r_urls,
                        _r_decision, _r_resolution, _r_execution)

                                                       
                                                     
                                                 
                                                  
                                                  
        vector_fallback: dict[str, object] | None = None
        if (not candidates and intent.abstain
                and intent.abstain_reason not in self._VECTOR_FALLBACK_EXCLUDED):
            from ..retrieval.vector_recall import semantic_fallback_recall
            _fb = semantic_fallback_recall(normalized, query, intent=intent,
                                           top_k=top_k or self.settings.top_k)
            if _fb:
                candidates = _fb
                vector_fallback = {
                    "reason": intent.abstain_reason,
                    "note": "条件未能执行，以下按语义相似度给出，未经条件核验",
                    "ignored_terms": list(intent.unresolved_terms or []),
                }

                                        
        _, retrieved_data_payload = _serialize_retrieved_data(candidates)

                                                                       
        fac = self.retriever.facets(normalized, intent, facet_filters)
                                                             
                                                                                      
        coverage_caveats = self.retriever.coverage_caveats(normalized, intent, facet_filters)
                                              
        if intent.parse_status == "clarification_required":
            res_status = "clarification_required"
            clar: dict | None = {
                "reason": intent.clarification_reason,
                "detail": intent.clarification_detail,
                "options": intent.clarification_options,
            }
        elif vector_fallback:
            res_status, clar = "vector_fallback", None
        elif intent.abstain:
            res_status, clar = "abstained", None
        elif candidates:
                                                         
            res_status, clar = ("degraded" if auto_degraded else "results"), None
        else:
            res_status, clar = "no_match", None
        interpretation = resolution.as_dict()
                                            
                                                     
        interpretation["effective_sources"] = list(
            _load_sources_for(resolution, auto_parse_sources=auto_parse_sources)
            or resolution.sources or ["10x Genomics"])
        interpretation["intent"] = intent_projection(intent)
        search_trace = _build_search_trace(resolution, intent, execution, decision, int(fac["total"]))
                                                          
                          
        rule_action_markers = detect_action_markers(query)
                                                                           
        fac_fields: dict[str, object] = {
            "result_total": fac["total"], "facets": fac["groups"], "active_filters": active_filters(intent),
            "coverage_caveats": coverage_caveats,
                                                               
            "unused_query_terms": intent.unused_query_terms,
                                                                         
            "or_handling": intent.or_handling,
                                                    
            "action_markers": rule_action_markers,
            "resolution_status": res_status, "clarification": clar,
                                                                      
            "vector_fallback": vector_fallback,
                                                                      
            "degraded_search": auto_degraded,
                                                                 
            "strategy": decision.as_dict() if decision is not None else None,
            "interpretation": interpretation,
            "search_trace": search_trace,
        }

        if not candidates:
            _update_trace_step(search_trace, "llm_polish", "skipped", "没有候选结果，不进行说明润色。")
            diagnosis = self.retriever.explain_empty(normalized, intent)
            if intent.parse_status == "clarification_required":
                fallback_reason = "clarification_required: " + intent.clarification_reason
            elif intent.abstain:
                fallback_reason = "abstained: " + intent.abstain_reason
            else:
                fallback_reason = "no matched candidates"
                                                         
                                                         
                                                  
            relaxation_options: list[dict[str, object]] = []
            if not facet_filters:
                relax_raw = self.retriever.relaxation_options(normalized, intent, top_k=top_k)
                relaxation_options = [
                    {
                        "key": opt["key"],
                        "label": opt["label"],
                                                                  
                        "kind": opt.get("kind", "drop"),
                        "count": opt["count"],
                                                                
                                                                  
                        "suppress": list(opt.get("suppress") or []),
                        "retrieved_data": _serialize_retrieved_data(opt["candidates"])[1],
                    }
                    for opt in relax_raw
                ]
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=render_no_result(diagnosis, intent),
                pipeline=FALLBACK_PIPELINE,
                llm_mode="disabled",
                llm_attempted=False,
                llm_succeeded=False,
                llm_response_used=False,
                llm_called=False,
                fallback="rule-based formatting",
                fallback_reason=fallback_reason,
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                relaxation_options=relaxation_options,
                **fac_fields,
            )

                                                    
                                                         
        if vector_fallback:
            _update_trace_step(search_trace, "llm_polish", "skipped", "语义兜底结果不做 AI 润色。")
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=str(vector_fallback["note"]) + "。\n\n" + format_candidates_markdown(candidates),
                pipeline=FALLBACK_PIPELINE,
                llm_mode="disabled",
                llm_attempted=False,
                llm_succeeded=False,
                llm_response_used=False,
                llm_called=False,
                fallback="rule-based formatting",
                fallback_reason="vector semantic fallback (parse abstain)",
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                **fac_fields,
            )

                                                               
                                                       
                                                                
                                                            
        llm_config: LLMConfig | None = None
        if llm_available is not False or use_llm:
            llm_config = self._effective_llm_config(
                use_llm=use_llm,
                mock_llm=mock_llm,
                provider=provider,
                base=base_llm_config,
            )

        if llm_config is None or (not llm_config.enable_llm and not llm_config.mock_llm):
            _update_trace_step(search_trace, "llm_polish", "skipped", "本次未开启 AI 说明润色。")
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=format_candidates_markdown(candidates),
                pipeline=FALLBACK_PIPELINE,
                llm_mode="disabled",
                llm_attempted=False,
                llm_succeeded=False,
                llm_response_used=False,
                llm_called=False,
                fallback="rule-based formatting",
                fallback_reason="LLM disabled",
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                **fac_fields,
            )

                                                            
                                                           
                                                  
        _, retrieved_data_payload = _serialize_retrieved_data(candidates)
        retrieved_data_text = json.dumps(
            retrieved_data_payload[:_LLM_PROMPT_CANDIDATE_CAP], ensure_ascii=False, indent=2)
        translated_keywords = _format_translated_keywords(intent)
        prompt = build_curator_prompt(
            user_query=query,
            retrieved_data=retrieved_data_text,
            translated_keywords=translated_keywords,
            max_rows=min(len(retrieved_data_payload[:_LLM_PROMPT_CANDIDATE_CAP]), _LLM_CURATOR_MAX_ROWS),
            n_shown=len(retrieved_data_payload[:_LLM_PROMPT_CANDIDATE_CAP]),
            n_total=len(retrieved_data_payload),
        )

                                                          
                                                         
        if _before_llm_hook is not None and _llm_call_would_happen(llm_config):
            _before_llm_hook(llm_config)
        llm_result = call_llm(prompt=prompt, config=llm_config, retrieved_records=retrieved_data_payload)

        if not llm_result.attempted:
                                                              
            _update_trace_step(
                search_trace, "llm_polish", "fallback", "AI 未实际调用，使用规则生成的推荐说明。",
                fallback_note=_fallback_note({"reason": "llm_not_configured"}),
            )
            llm_mode = "enabled" if llm_config.mock_llm else "disabled"
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=format_candidates_markdown(candidates),
                pipeline=FALLBACK_PIPELINE,
                llm_mode=llm_mode,
                llm_attempted=False,
                llm_succeeded=False,
                llm_response_used=False,
                llm_called=False,
                llm_provider=llm_result.provider,
                model=llm_result.model,
                prompt_name=PROMPT_NAME if (llm_config.enable_llm or llm_config.mock_llm) else None,
                fallback="rule-based formatting" if llm_result.error != "missing OPENAI_API_KEY" else "missing OPENAI_API_KEY",
                fallback_reason=llm_result.error,
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                **fac_fields,
            )

        if not llm_result.succeeded or not llm_result.text:
                                                       
                                                 
                                   
            _update_trace_step(
                search_trace, "llm_polish", "fallback", "AI 调用失败或返回为空，使用规则说明。",
                fallback_note=_fallback_note({
                    "reason": "llm_auth_failed" if is_auth_error(llm_result.error) else "llm_call_failed"
                }),
            )
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=format_candidates_markdown(candidates),
                pipeline=LLM_PIPELINE_FALLBACK,
                llm_mode="enabled",
                llm_attempted=True,
                llm_succeeded=False,
                llm_response_used=False,
                llm_called=True,
                llm_provider=llm_result.provider,
                model=llm_result.model,
                prompt_name=PROMPT_NAME,
                fallback="rule-based formatting",
                fallback_reason=llm_result.error or "empty LLM response",
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                **fac_fields,
            )

        valid, reason = validate_llm_answer(
            llm_result.text,
            candidates,
            intent,
            finish_reason=llm_result.finish_reason,
        )
        if not valid:
                                               
            _update_trace_step(
                search_trace, "llm_polish", "fallback", "AI 说明未通过反捏造校验，使用规则说明。",
                fallback_note=_fallback_note({"reason": "invalid_llm_answer"}),
            )
            _finalize_search_trace(search_trace, workflow_started_at)
            return WorkflowResult(
                answer=format_candidates_markdown(candidates),
                pipeline=LLM_PIPELINE_FALLBACK,
                llm_mode="enabled",
                llm_attempted=True,
                llm_succeeded=True,
                llm_response_used=False,
                llm_called=True,
                llm_provider=llm_result.provider,
                model=llm_result.model,
                prompt_name=PROMPT_NAME,
                fallback="rule-based formatting",
                fallback_reason=f"invalid LLM answer: {reason}",
                retrieved_dataset_names=retrieved_dataset_names,
                retrieved_urls=retrieved_urls,
                retrieved_data=retrieved_data_payload,
                **fac_fields,
            )

        _update_trace_step(search_trace, "llm_polish", "used", "AI 只润色推荐说明，不改变数据、过滤条件或排序。")
        _finalize_search_trace(search_trace, workflow_started_at)
        return WorkflowResult(
            answer=llm_result.text,
            pipeline=LLM_PIPELINE_SUCCESS,
            llm_mode="enabled",
            llm_attempted=True,
            llm_succeeded=True,
            llm_response_used=True,
            llm_called=True,
            llm_provider=llm_result.provider,
            model=llm_result.model,
            prompt_name=PROMPT_NAME,
            retrieved_dataset_names=retrieved_dataset_names,
            retrieved_urls=retrieved_urls,
            retrieved_data=retrieved_data_payload,
            **fac_fields,
        )


def _load_sources_for(resolution, *, auto_parse_sources: bool):
    """偏好来源并入装载池的**唯一判据**。

    只在 `default` 模式（调用方既没显式限定来源、查询也没被来源专名收窄）并入「优先 <来源>」：
    此时池子只有基础库，不并入偏好来源偏好就是结构性 no-op。用户显式限定（手动勾选）或查询
    已点名的来源范围必须尊重——范围外偏好由调用方走既有「未生效」回显，绝不偷偷扩容检索池
    （此前无条件并池：结果可能来自范围外来源，解释却仍回显收窄后的范围）。
    """
    sources = resolution.sources
    if (auto_parse_sources and resolution.preferred_sources
            and resolution.source_mode == "default"):
        return sorted(set(sources or [])
                      | set(resolution.preferred_sources)
                      | ({"10x Genomics"} if sources is None else set()))
    return sources
