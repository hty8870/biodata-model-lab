                       
"""统一对话管线（turn pipeline）：**一切指令 → 「AI 执行」闸 → 规则检索直达 / LLM 分流**。

## 这是什么

用户在任何输入框里说的一句话，都先问本模块「该交给哪条管线」。**只路由，不执行**：
不产交付物、不落盘、不联网取数据（LLM 分流那一次调用除外）。

## 管线（2026-08-03 定稿：「AI 执行」开关 = LLM 分流器的总闸）

0. **编号快速道**：贴数据集编号/直链（GSE… / E-MTAB… / DOI）→ 直接 search（按原话），
   不付 LLM 延迟。编号必须排在执行词前面（「把 E-MTAB-1234 打包」要先查出那一条，
   才谈得上打包它），句中执行诉求由 `/api/recommend` 的 `action_markers` 如实指路。
   **唯一例外**（2026-08-15 「编号 + 管护操作」（删掉/检查更新类，
   `action_plan.rule_curate_op_marker` 命中）不是检索诉求——快速道抢在 agent_off 气泡
   与 LLM 分流之前会把操作意图静默吃掉，这类句子落入正常分流。
1. **「AI 执行」（维度 C，请求的 agent 标志）关闭**：LLM 分流器**永不启动**——不拼装
   分流提示词、不发调用，一切输入按规则检索处理（route=search 原话直达，真正的规则
   匹配由 `/api/recommend` 侧做，这里不再重复付一次）。**唯一例外**：规则（非 LLM）
   检出操作意图的句子（`action_plan.rule_operation_marker`）不静默当检索处理——
   回降级气泡（route=none, needs_agent=True），口径「这是操作指令，需开启 AI 执行」。
2. **「AI 执行」开启**：**所有消息 100% 过 LLM 分流**，分两喂——
   **规则匹配概览**（命中概览 / 零命中 / 弃权；为检索路径服务，但一切指令都经此处理）
   连同**原始查询**一起进 LLM。**零命中/弃权 ≠ 无效查询**：工具调用句往往零命中
   （「检查10x数据库是否有更新」里没有一个检索词），这是本模块存在的理由。
   LLM 分流优先 `agent_exec.plan_with_agent`（langgraph 编排）；不可用/失败原样回退
   `action_plan.plan_action` 保底——封闭动词表 + 机械护栏，两条路径同一份护栏真源：
   - 路径1 · 检索指令：`search.new` / `refine.conditions` / `lookup.identifier`，
     携带 `effective_query`（LLM 据当前查询+条件改写的完整检索句）——
     真正的检索仍由调用方拿着它走 `/api/recommend`（LLM 重排 → 重写 → 润色都在那边）。
   - 路径2 · 工具调用：EXEC 动词，只出 plan，执行由前端 act 结构完成。
   - `none`：如实回音——LLM 真判的 none 说「没听懂」；LLM 缺席/失败时规则兜底的 none
     说「大模型没有接上」（这两件事绝不能混：用户改说法治不好连接问题）。
   **LLM 缺席/失败**（plan.source=="rule"）的规则兜底：动作词命中 → 规则档 plan（tool）；
   `board.search_shaped`（长检索的脸）→ search（按原话）；其余 → none。
   **绝不**把零命中/带执行标记的句子 fail-open 成检索。
"""
from __future__ import annotations

import threading
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from . import action_plan as _ap
from . import agent_exec as _agent
from . import decider as _decider
from .trace import bind_recorder as _bind_recorder
from .trace import events as _te
from .trace import recorder_for_turn as _recorder_for_turn
from ..app.board import search_shaped
from ..content.identifiers import classify as _classify_identifier
from ..llm.llm_client import LLMConfig, load_llm_config
from ..retrieval.query_parser import detect_action_markers, parse_query
from ..retrieval.search_request import mask_source_spans

                                               
ROUTE_SEARCH = "search"
ROUTE_TOOL = "tool"
ROUTE_NONE = "none"

_SUMMARY_TOP_TITLES = 3

                                                                  
                                                              
                                                                   
                                    
                                                         
                                                        
                                               
                                                                       
_FRONTEND_EXEC_PLANE: frozenset[str] = frozenset(
    s.verb for s in _ap.VERB_SPECS if s.frontend_dispatch)

                                                     
                                        
                                         
_LLM_ABSENT_REASON_ZH = {
    "disabled": "它在设置里被关着",
    "no_key": "还没有配置密钥（API Key）",
    "mock_not_used": "当前是本地演示模式",
    "empty": "它这次没有回话",
    "unparsable": "它这次的回答读不懂",
}


def _partition_intents(intents: list[dict[str, Any]]
                       ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None,
                                  list[dict[str, Any]] | None]:
    """子意图枚举清单 → ``(None, intent_checklist, pending_frontend)``。

    「不少于我」下限合同的分流器（2026-09-01）：
    - 空清单 / 全 cancelled → ``(None, None, None)``：无 EXEC 待办，走 agent 图（现状）。
    - 所有 active 项都进入 checklist 并注入图初始 state；图是唯一规划终态通道。
    - 环内 EXEC 标 ``plane="inloop"`` 待核销；前端执行项标 ``plane="frontend"``，
      同时进入 ``pending_frontend``，图结束后交给同一 act dispatcher。
    - 返回三元组首项恒 None，保留调用形状只为兼容；不再存在 agent bypass。

    checklist 注入图初始
      state（环内项 ``plane="inloop"`` 待核销；直派面项 ``plane="frontend"`` 如实告知
      decide「已交前端」、环内免核销），直派面项图跑完后经 additive
      ``plan["pending_frontend"]`` 挂回响应让前端接力——**绝不进 plan.steps**
      （steps 非空是「后端已执行」所有权令牌）。
    """
    active = [p for p in intents if not p.get("cancelled")]
    if not active:
        return None, None, None
    frontend = [p for p in active if p.get("verb") in _FRONTEND_EXEC_PLANE]
    inloop = [p for p in active if p.get("verb") not in _FRONTEND_EXEC_PLANE]
    checklist = [
        {"verb": p.get("verb") or "", "verb_zh": p.get("verb_zh") or "",
         "quoted": p.get("quoted") or "", "plane": "inloop"}
        for p in inloop
    ] + [
        {"verb": p.get("verb") or "", "verb_zh": p.get("verb_zh") or "",
         "quoted": p.get("quoted") or "", "plane": "frontend"}
        for p in frontend
    ]
    return None, checklist, (frontend or None)


def _norm_scope_list(value: Any) -> list[str]:
    """规范化 scope 列表（供指纹比较）：去空、逐项 str、排序去重——scope 不因项序变化。"""
    if value is None:
        return []
    out: list[str] = []
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        if isinstance(item, dict):
            out.append("|".join(str(item.get(k, "")) for k in ("dim", "value")))
        else:
            out.append(str(item or "").strip())
    return sorted({x for x in out if x})


def _batch_scope_fingerprint(query_effective: str, sources: Any, search_params: dict | None) -> str:
    """批次检索范围指纹。

    规范化 query + sources + facet/suppressed/lenient + date 检索条件后 SHA-256 十六进制。
    前端据它判「是否同一次检索 scope」（同=去重不追加；异=真换词批可作备选）。这是**契约级
    身份键**：加字段/改口径 = contract_change，须生产者/消费者/测试原子落地。"""
    sp = search_params or {}
    descriptor = {
        "query": str(query_effective or "").strip(),
        "sources": _norm_scope_list(sources),
        "facet_filters": _norm_scope_list(sp.get("facet_filters")),
        "suppressed_constraints": _norm_scope_list(sp.get("suppressed_constraints")),
        "lenient_dims": _norm_scope_list(sp.get("lenient_dims")),
        "date_from": str(sp.get("date_from") or "").strip(),
        "date_to": str(sp.get("date_to") or "").strip(),
    }
    canonical = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _llm_absent_zh(llm_status: Any) -> str:
    s = str(llm_status or "")
    if s.startswith("error:"):
        return "调用出了错"
    if s.startswith("config_error:"):
        return "配置没能加载"
    return _LLM_ABSENT_REASON_ZH.get(s, "这次没能连上")


                                           
                                       
                                          
_HYBRID_ABSTAIN_ZH = (
    "这句话里串了好几件要分步做的事（又有库操作、又有检索）——"
    "一次只能办一件的简化通道做不全，我没有只挑一半做，这次什么都没有执行。"
    "「AI 执行」开着、大模型接得上时，我才能把整句按顺序完整做完："
    "到「设置」里确认「AI 执行」已开启、大模型连接正常，再把这句原样说一次就行。"
)

                                                      
                                       
                      
_HYBRID_ABSTAIN_AGENT_DOWN_ZH = (
    "这句话里串了好几件要分步做的事（又有库操作、又有检索），"
    "本来该由「AI 执行」整句按顺序做完——但这次大模型没能接得上"
    "（连接或试用额度问题），我没有只挑一半做，这次什么都没有执行。"
    "你的「AI 执行」是开着的，不用改设置：稍后把这句原样重说一次就行；"
    "若反复失败，到「设置」里看一眼大模型连接与试用额度。"
)


def rule_match_summary(text: str, *, sources: Any = None,
                       search_params: dict | None = None,
                       meta_out: list | None = None) -> dict[str, Any]:
    """管线第一段：规则匹配概览。**为 LLM 分流提供信息，绝不抛异常、绝不判死刑。**

    返回 `{status, total, top_titles, abstain_reason, unresolved_terms}`：
    - status ∈ results / no_match / abstained / clarification_required / error；
    - total 是未截断的命中总数（弃权/零命中为 0）；
    - top_titles 只取前几条做概览（prompt 体积有预算）；
    - 弃权时补一次轻量重解析（mask 来源专名后过 `parse_query`，与检索热路径同径），
      把「哪几个词卡住了」如实带给 LLM。

    `search_params`（2026-08-16 初步结果先行）：给了就改用**真实检索参数**跑
    完整确定性管线（top_k/向量召回/分面/忽略/放宽/日期/策略；rerank 恒 off、
    use_llm 恒 False——pre-loop 不付 LLM 重排与润色）；None = 旧行为（top_k=3 轻量
    概览）逐位不变。`meta_out` 给了就把本次 run_with_meta 的 WorkflowResult append
    进去（route_turn 拿它做 preliminary 载荷与机械闸判定；投影摘要逐键同形，
    top_titles 恒切前 3 条、与 top_k 无关）。
    """
    out: dict[str, Any] = {
        "status": "error", "total": 0, "top_titles": [],
        "abstain_reason": "", "unresolved_terms": [], "note": "",
    }
    try:
        from ..app.workflow import DatasetRecommendationWorkflow, RecommendParams

        run_kwargs: dict[str, Any] = {
            "query": text, "top_k": _SUMMARY_TOP_TITLES,
            "use_llm": False, "mock_llm": False, "provider": "mock",
            "sources": sources,
        }
        if search_params is not None:
            sp = dict(search_params)
            strategy = "auto" if str(sp.get("strategy") or "") == "auto" else "fixed"
            recall_available = None
            preferred_recall = "cross_encoder"
            if strategy == "auto":
                                                                             
                from ..retrieval.vector_recall import resolve_auto_recall_preference
                preferred_recall, recall_available = resolve_auto_recall_preference()
                                                                 
                                                              
            run_kwargs.update({
                "top_k": sp.get("top_k"),
                "auto_parse_sources": bool(sp.get("auto_parse_sources", False)),
                "rerank_top_n": sp.get("rerank_top_n"),
                "rerank_backend": "off",
                "recall_backend": str(sp.get("recall") or "off"),
                "date_from": str(sp.get("date_from") or ""),
                "date_to": str(sp.get("date_to") or ""),
                "facet_filters": sp.get("facet_filters"),
                "suppressed_constraints": sp.get("suppressed_constraints"),
                "lenient_dims": sp.get("lenient_dims"),
                "strategy": strategy,
                "recall_available": recall_available,
                "llm_available": False,
                "preferred_recall": preferred_recall,
            })
        meta = DatasetRecommendationWorkflow().run_with_meta(RecommendParams(**run_kwargs))
    except Exception as exc:                                 
        out["note"] = type(exc).__name__[:40]
        return out
    if meta_out is not None:
        meta_out.append(meta)
    status = str(getattr(meta, "resolution_status", "results") or "results")
    data = list(meta.retrieved_data or [])
    out["status"] = status
    out["total"] = int(getattr(meta, "result_total", 0) or len(data))
    out["top_titles"] = [
        str(r.get("title") or r.get("name") or "") for r in data[:_SUMMARY_TOP_TITLES]
    ]
    if status == "abstained":
        try:
            intent = parse_query(mask_source_spans(text))
            out["abstain_reason"] = str(intent.abstain_reason or "")
            out["unresolved_terms"] = [t for t in (intent.unresolved_terms or []) if str(t).strip()]
        except Exception:
            pass
    return out


                                                        
                                                                    
                                                  
                                                      
def _none_route_suggestions(*, has_results: bool) -> list[dict[str, str]]:
    out = [
        {"label": "清点库里有什么", "utterance": "清点一下数据库里现在有什么"},
        {"label": "检查来源更新", "utterance": "检查数据库来源有没有更新"},
    ]
    if has_results:
        out.append({"label": "打包当前结果", "utterance": "把当前这批结果打包下载"})
    return out


                                                      
                                                     
                                                       
                                                             
                                               
                               
_RAG_MAX_CONCURRENT = 3
_RAG_SEMAPHORE = threading.Semaphore(_RAG_MAX_CONCURRENT)
_RAG_EXECUTOR: "ThreadPoolExecutor | None" = None
_RAG_EXECUTOR_LOCK = threading.Lock()


def _rag_executor() -> ThreadPoolExecutor:
    """惰性单例线程池（模块级、进程常驻；测试可注入替换）。"""
    global _RAG_EXECUTOR
    if _RAG_EXECUTOR is None:
        with _RAG_EXECUTOR_LOCK:
            if _RAG_EXECUTOR is None:
                _RAG_EXECUTOR = ThreadPoolExecutor(
                    max_workers=_RAG_MAX_CONCURRENT,
                    thread_name_prefix="rag-flight",
                )
    return _RAG_EXECUTOR


class _RagFlight:
    """一次 pre-loop 规则检索（rule_match_summary）的并发执行体。

    - start()：准入信号量拿槽 → 线程池提交；拿不到 → 标 deferred 不起线程。
    - join()/done() 幂等；deferred 在 join 点由调用线程同步补跑。
    - abandoned：verdict=action 时置位；未起跑 future 可 cancel（僵尸上限 = 池容量）。
    - emitted/payload：初步结果发射的原子状态（图线程发射，回填 turn 局部真源）。
    - summary/meta：rule_match_summary 的结果与 WorkflowResult（发射闸与载荷数据源）。
    """

    def __init__(self, text: str, *, sources: Any = None,
                 search_params: dict | None = None) -> None:
        self.text = text
        self.sources = sources
        self.search_params = search_params
        self.lock = threading.Lock()
        self.summary: dict | None = None
        self.meta: Any = None
        self._future: "Future | None" = None
        self._deferred = False
        self._started = False
        self._joined = False
        self.abandoned = False
        self.emitted = False
        self.payload: dict | None = None

                       
    def start(self) -> None:
        """拿槽提交线程池；拿不到 → deferred（不起线程）。幂等。"""
        if self._started or self.abandoned:
            return
        self._started = True
        if _RAG_SEMAPHORE.acquire(blocking=False):
            self._future = _rag_executor().submit(copy_context().run, self._run)
        else:
                                              
            self._deferred = True

    def _release_slot(self) -> None:
        if self._started and not self._deferred:
            _RAG_SEMAPHORE.release()

    def _run(self) -> None:
        """线程池线程内执行（只算不说不听）。无论成败必须释放槽位。"""
        try:
            holder: list = []
            self.summary = rule_match_summary(
                self.text, sources=self.sources,
                search_params=self.search_params, meta_out=holder)
            if holder:
                self.meta = holder[0]
        except Exception:
                                                                  
                                                                           
            self.summary = {"status": "error", "total": 0, "top_titles": [],
                            "abstain_reason": "", "unresolved_terms": [], "note": "flight_error"}
            self.meta = None
        finally:
            self._release_slot()

                      
    def done(self) -> bool:
        if self._joined:
            return True
        if self._deferred:
            return False
        fut = self._future
        return fut is not None and fut.done()

    def join(self) -> dict | None:
        """幂等汇合：deferred 就地同步补跑；线程池任务等待完成。返回 summary。"""
        if self._joined:
            return self.summary
        if self.abandoned and self._deferred:
                                              
            self._joined = True
            return None
        if self._deferred:
                                       
            self._run()
        else:
            fut = self._future
            if fut is not None:
                if fut.cancelled():
                                                             
                    self._joined = True
                    return None
                try:
                    fut.result()                       
                except BaseException:
                                                        
                    pass
        self._joined = True
        return self.summary

    def cancel(self) -> None:
        """verdict=action：置 abandoned；未起跑 future 取消（僵尸上限 = 池容量）。"""
        self.abandoned = True
        fut = self._future
        if fut is not None and not fut.done():
            if fut.cancel():
                                                            
                self._release_slot()

    def ensure_payload(self) -> dict | None:
        """recommend_payload 建一次存 flight（原则不重跑）。"""
        if self.payload is None and self.meta is not None:
            from ..app.conversation_results import complete_payload
            self.payload = complete_payload(self.meta, query=self.text, params=self.search_params, sources=self.sources)
        return self.payload

    @property
    def has_hits(self) -> bool:
        """发射闸的命中判定：status==results ∧ total>0（与既有机械闸同口径）。"""
        m = self.meta
        if m is None:
            return False
        try:
            return (str(getattr(m, "resolution_status", "") or "") == "results"
                    and int(getattr(m, "result_total", 0) or 0) > 0)
        except Exception:
            return False


def _warmup_rag_environment(search_params: dict | None) -> None:
    """预热闭合（r3）：主线程幂等 ensure vector/env 初始化——flight 线程从此
    只读、不写不读 os.environ 易变项（`_setup_determinism` 写 CUBLAS_WORKSPACE_CONFIG、
    模型加载写全局缓存，都必须在 flight 起跑前由主线程完成；幂等 + 模块级单飞锁保证
    多线程并发安全）。"""
    try:
        from ..retrieval import vector_recall as _vr
        _vr._setup_determinism()
        sp = search_params or {}
        if str(sp.get("strategy") or "").strip().lower() == "auto":
            backend = str(sp.get("recall") or "off").strip().lower()
                                                                     
                                                    
            if backend in ("cross_encoder", "dense", "vector") and _vr.recall_backend_local_available(backend):
                _vr.warm_recall_backend(backend)
    except Exception:
        pass


def _emit_preliminary(flight: "_RagFlight | None", *, agent_path: bool,
                      on_event: Callable | None, state: dict) -> bool:
    """发射（verdict-gated）：闸**全与**才发，至多一次，盒子持锁原子。

    闸 = agent_path ∧ on_event 在场 ∧ flight 完成 ∧ status==results ∧ total>0 ∧
    ¬abandoned ∧ ¬emitted。发射后置 flight.emitted 并回填 turn 局部真源
    （state["preliminary_sent"]/state["prelim_payload"]——b 档与批次组卷依赖，
    r3 关键核查②）。
    """
    if not (agent_path and on_event is not None and flight is not None
            and flight.done() and flight.has_hits
            and not flight.abandoned and not flight.emitted):
        return False
    with flight.lock:
        if flight.emitted:
            return False
        payload = flight.ensure_payload()
        if payload is None:
            return False
        flight.emitted = True
        state["preliminary_sent"] = True
        state["prelim_payload"] = payload
        on_event("preliminary", payload)
    return True


def _make_route_verdict_hook(holder: dict) -> Callable[[str], None]:
    """route_consensus verdict hook（**只做 abandoned/lazy 标记，不发射**）。

    holder: {"flight", "markers", "text", "sources", "search_params"}——图线程调用，
    跨线程共享的飞行状态盒。
    """
    def hook(route: str) -> None:
        if route == "action":
            flight = holder.get("flight")
            if flight is not None:
                flight.cancel()
            return
        if holder.get("flight") is not None:
            return
        if holder.get("markers"):
                                                                   
                                                          
            _lazy_start_flight(holder)
        elif holder.get("prelim_gated") and route == "search":
                                                                
                                                               
                                                        
                                             
            _lazy_start_flight(holder)
    return hook


def _lazy_start_flight(holder: dict) -> None:
    """verdict hook 的 lazy 补起唯一出口：重新入池（池满则 deferred）。"""
    _warmup_rag_environment(holder.get("search_params"))
    new_flight = _RagFlight(
        holder["text"], sources=holder.get("sources"),
        search_params=holder.get("search_params"))
    new_flight.start()
    holder["flight"] = new_flight


                                                     
                                                        
                                                              
                                                              
                                                      
                                                        
                                                                
                                                   
             
                                                                  
                                                     
                                                    
                                                
                                                                  
                                                                    
                                                    
_PRELIM_GATE_ENV = "BIODATA_PRELIM_GATE"
_PRELIM_GATE_TAU_ENV = "BIODATA_PRELIM_GATE_TAU"
_PRELIM_GATE_BAND_ENV = "BIODATA_PRELIM_GATE_BAND"
_PRELIM_GATE_BAND_LO_ENV = "BIODATA_PRELIM_GATE_BAND_LO"
_PRELIM_GATE_BAND_HI_ENV = "BIODATA_PRELIM_GATE_BAND_HI"


def _prelim_gate_skip(text: str) -> bool:
    """闸判定：True = 不起 flight。任何不确定都回 False（现状行为，多跑不错跑）。
    实现链：L0 一线（µs）→ 灰带内 decider 仲裁（有则用之）→ 全不确定回放行。"""
    if not _agent._env_default_on(_PRELIM_GATE_ENV):
        return False
    try:
        hit = _decider.score_with_fallback("prelim", text, prefer=("l0", "decider"))
        if hit is None:
            return False
        tau = _decider.tau_with_fallback("prelim", _PRELIM_GATE_TAU_ENV, "tau_retrieve",
                                         str(hit.get("impl") or ""), 0.0)
        p_ret = float((hit["probs"] or {}).get("retrieve") or 0.0)
        if str(hit.get("impl") or "") == "l0" and _agent._env_default_on(
                _PRELIM_GATE_BAND_ENV):
                                                               
                                   
            lo = _decider.operating_tau("prelim", _PRELIM_GATE_BAND_LO_ENV,
                                        "band_lo", 1.1)
            hi = _decider.operating_tau("prelim", _PRELIM_GATE_BAND_HI_ENV,
                                        "band_hi", 1.1)
            if lo <= p_ret < hi and _decider._env() != "off":
                d = _decider.score("prelim", text)
                if d is not None:
                    dtau = _decider.tau_with_fallback(
                        "prelim", _PRELIM_GATE_TAU_ENV, "tau_retrieve", "decider", tau)
                    return float((d["probs"] or {}).get("retrieve") or 0.0) < dtau
        return p_ret < tau
    except Exception:
        return False


def _make_retrieval_provider(holder: dict, *, agent_path: bool,
                             on_event: Callable | None, state: dict) -> Callable[[], dict | None]:
    """understand 入口的 retrieval provider：join（deferred 在此同步
    补跑）→ 闸过则发射（主路径唯一发射点）→ 返回摘要 dict。图线程调用。"""
    def provider() -> dict | None:
        flight = holder.get("flight")
        if flight is None:
            return None
        summary = flight.join()
        _emit_preliminary(flight, agent_path=agent_path,
                          on_event=on_event, state=state)
        return summary
    return provider


def _keyword_count_summary(text: str, *, sources: Any = None,
                           search_params: dict | None = None) -> dict:
    """杠杆②（设计 v3.2 §4.3）：图起跑前的**同步关键词快速计数段** → 摘要 dict。

    调用既有 pre-loop 同一段 `rule_match_summary`，但 strategy 强制 "fixed"、recall
    强制 "off"（纯关键词，实测热态毫秒级——§2 表末行 + §5 复验）。只供共识上下文
    拼装使用，**绝不影响** flight 的 auto 全量检索（无标记分支 flight 仍以用户
    search_params 全量跑，供 understand/display）。fail-open：fixed 摘要异常/失败
    按既有 status="error" 形状返回，绝不抛、绝不成为新故障源。
    """
    sp = dict(search_params or {})
    sp["strategy"] = "fixed"
    sp["recall"] = "off"
    try:
        return rule_match_summary(text, sources=sources, search_params=sp)
    except Exception:
                                                              
                                       
        return {"status": "error", "total": 0, "top_titles": [],
                "abstain_reason": "", "unresolved_terms": [], "note": "keyword_count_error"}


def _consensus_extra_zh(text: str, *, sources: Any = None,
                        search_params: dict | None = None) -> str:
    """route_extra_zh 内容生成（v3.2 修正，盲跑实测处置 + §5 复验裁定）。

    盲跑实测证明「机械标记行 + 裸关键词段行」的追加式格式会引入偏置（u03/u04 真
    回归未被修复、u05/u06/u16 新分歧）——补偿必须**逐字复刻**今天串行路径共识实际
    看到的检索概览段（`_route_context_zh` 的 retrieval 位同款文案：前缀 +
    `agent_exec._route_retrieval_zh`），使共识输入与今天**逐位同构**（字节级一致 →
    共识行为与今天一致 → 零回归由构造保证）。不再拼机械标记行（实测其 action 偏置
    是 u03/u04 回归主因；今天串行路径共识本就无该行）。

    有标记/无标记分支输出同一行。`_route_retrieval_zh` 只报 status/total 与弃权
    原因，绝不含结果集标题（诚实不变量 tests/test_scoped_routing.py:236 同族）。
    """
    summary = _keyword_count_summary(text, sources=sources, search_params=search_params)
    retrieval_zh = _agent._route_retrieval_zh(summary)
    return "**这句话**过规则匹配（关键词检索第一段）的结果：" + retrieval_zh


def _recipe_narrow_plan(plan: dict | None, recipe_verbs: "frozenset[str] | None") -> dict | None:
    """suggested_recipe 机械收窄（2026-08-22 只缩小、不扩权）。

    agent 路径产出的 plan 不受 action_plan.plan_action 的 allowed_verbs 闸约束（图内
    动词表独立），这里在 turn 层补同一道机械闸：工具类 plan（EXEC）的 verb 不在 recipe
    允许集 → 降 none 并附如实注记（`recipe_narrowed` 记录被收窄掉的 verb，供调试/审计，
    不进回执文案；也不进 rejected——那不是模型造词，是「用户选定的建议动作与实际路由
    不符」）。search/none 路由不动（hint 只约束执行动词面）；参数校验/执行开关/安全闸
    （cancelled / 极性门 / blocked_reason）一字不动——只动 verb 一个键。
    """
    if plan is None or not recipe_verbs or plan.get("kind") != _ap.EXEC:
        return plan
    verb = str(plan.get("verb") or "")
    if verb and verb not in recipe_verbs:
        if plan.get("steps"):
                                                 
                                                              
                                    
            return plan
        narrowed = _ap._blank_plan(
            source=str(plan.get("source") or "llm"),
            llm_status=str(plan.get("llm_status") or "ok"),
            reason_zh="这条建议动作没能路由到对应的能力，已按未执行处理。",
        )
        narrowed["recipe_narrowed"] = verb                                      
        return narrowed
    return plan

def _route_turn_impl(
    utterance: str,
    *,
    has_results: bool = False,
    result_total: int = 0,
    current_query: str = "",
    current_filters: Any = None,
    sources: Any = None,
    config: LLMConfig | None = None,
    llm_call: Callable[[str], str | None] | None = None,
    keyword_mapping: dict | None = None,
    use_agent: bool = True,
    on_event: Callable[[str, dict], None] | None = None,
    principal: str = "",
    search_params: dict | None = None,
    artifact_context: str = "",
    suggested_recipe: str = "",
    ask_answer: dict | None = None,
    conversation_history: list[dict] | None = None,
    prefetched_out: list | None = None,
) -> dict[str, Any]:
    """一句话 → `{route, query, plan, echo_zh, retrieval, via, needs_agent, suggestions}`。永不执行、永不抛（除入参非法）。

    （2026-08-17 本体从 `route_turn` 改名而来——公开入口是文件底部的同名薄壳
    只多一圈 trace recorder 绑定与 route_decision 落账，管线逻辑一字未动。）

    `principal`：成功经验库分区主体（会话账户 id，空 → anonymous）——
    只在 agent 路径透传给 plan_with_agent(_events)；规则/保底路径不读它。

    - route == "search"：`query` 是拿去 `/api/recommend` 的完整检索句
      （LLM 改写的 effective_query；没改写就是用户原话）。
    - route == "tool"：`plan` 是 EXEC 动词的执行 plan（前端 act 结构派发）。
    - route == "none"：`echo_zh` 是给用户的如实回音；`plan` 一并带回（含规则兜底信息）。
      `needs_agent=True` 是其中的降级气泡专档（「AI 执行」关 + 规则检出操作意图）：
      前端据它渲染成带设置指路的美观气泡。
      `suggestions`（2026-08-09 五机制批）只在 **LLM 真判的 none**（「没听懂」死胡同）
      非空：2~3 颗机械生成的候选动作 `[{label, utterance}]`，前端渲染成可点 chip、
      点击即把该 utterance 重新入环；其余路由恒空列表（LLM 缺席的兜底 none 不带——
      管护动词没有大模型到场判不了，给候选也是死路）。
    - `retrieval`：规则匹配概览（回显/调试；「AI 执行」关的规则直达档不付这次匹配，恒 None）。

    `use_agent` 是「AI 执行」开关（维度 C）的请求标志——**LLM 分流器的总闸**：
    关闭时（且未注入 llm_call）LLM 分流器永不启动（不拼装提示词、不发调用）。
    开启时优先走 `agent_exec.plan_with_agent`（langgraph 编排）：**agent 可用 ∧
    大模型总开关开且非 mock 且有 key** 时先试它；它抛出 AgentError 或**任何**异常都
    原样回退 `action_plan.plan_action` 保底——agent 路径绝不成为新的单点。
    `llm_call` 注入时按「开启」处理且永不走 langgraph agent（测试隔离，与 plan_action 同纪律）。
    `on_event`（2026-08-03 流式）只在 agent 路径透传给 `plan_with_agent_events`
    （每节点落定时回调 trace 条目）；保底路径**不回调**——它是一次性 LLM 调用，没有节点可播。

    `search_params`（2026-08-16 初步结果先行，并发分流 v3.1
    重定发射语义）：/api/utterance 端点收敛后的真实检索参数（top_k/recall/strategy/
    facet_filters/suppressed_constraints/lenient_dims/date_from/date_to/rerank）——
    pre-loop 规则段用它跑完整确定性管线（rerank 恒 off、不付 LLM 重排/润色）；
    None = 旧轻量概览逐位不变。初步结果先行改为 **verdict-gated**：
    marker 命中 → 不起 RAG flight 直接进图（**永不发射**，纯执行不打印）；无标记 →
    flight 起跑（准入信号量 ≤3；池满 deferred）∥ 图起跑（共识盲跑）；发射点全在图线程
    ——主路径唯一发射点 = understand 节点入口（join/补跑完成后、构造 prompt 前，
    闸 = agent_path ∧ on_event 在场 ∧ flight 完成 ∧ status==results ∧ total>0 ∧
    ¬abandoned），保底分支在 plan_action 判出非 EXEC（search/general 向）后同法发射、
    EXEC/none 一律抑制。action 路线永不发射。返回值 additive 两键：`result_payload`
    （环内 search.rerun 采纳步 / rank·rerank display 批次的 recommend_payload；无环内
    上屏批时镜像 active 批——仅 preliminary 批也镜像，「legacy 镜像 active
    batch」；无批则 None）与 `preliminary_final`
    （保守判定，见 search 分支注释——任一条件不明即 False，宁可重检不跳检）。
    轮内有批时再加 `result_batches`
    （preliminary 在前、环内上屏批在后，各补 batch_id/seq/created_at/turn_id）与
    `active_batch`（默认最后一批）；无批时两键均不出现。

迭代杠杆②（设计 v3.2 §4.3/§5）：并发路径在**图起跑前**同步跑关键词快速计数段
    （`_consensus_extra_zh`，strategy 强制 fixed + recall 强制 off = 纯关键词热态毫秒级），
    把「状态/命中数」信号以**逐字复刻今天串行路径共识检索概览段**的文案注入
    `route_extra_zh`（`**这句话**过规则匹配（关键词检索第一段）的结果：规则匹配…`）
    ——补共识盲跑丢失的检索信号，共识输入与今天逐位同构（§0 盲跑实测 u03/u04 真
    回归的根因补偿；盲跑裁定追加式机械行格式有偏置，§5 记录）。
    """
    text = _ap.normalize_utterance(utterance)
                                                 
                                                                         
                                               
                                                      
    _ask_answer = ask_answer if isinstance(ask_answer, dict) else None

                                                     
                                                 
                                      
    if _classify_identifier(text) and not _ap.rule_curate_op_marker(text):
        return {
            "route": ROUTE_SEARCH, "query": text, "plan": None,
            "echo_zh": "", "retrieval": None, "via": "identifier", "needs_agent": False,
            "suggestions": [],
            "result_payload": None, "preliminary_final": False,
        }

                                        
    if not use_agent and llm_call is None:
        marker = _ap.rule_operation_marker(text)
        if marker:
                                               
            return {
                "route": ROUTE_NONE, "query": "",
                "plan": _ap._blank_plan(
                    source="rule", llm_status="agent_off",
                    reason_zh=f"「AI 执行」未开启：规则认到操作词「{marker}」，不执行、也不当检索处理。",
                ),
                "echo_zh": (
                    f"这句像是一个操作指令（我认到了「{marker}」），不是检索。"
                    "「AI 执行」现在没有开启，我不能替你动手——"
                    "到「设置」里打开「AI 执行」，再把这句说一次就行。"
                ),
                "retrieval": None, "via": "agent_off", "needs_agent": True,
                "suggestions": [],
                "result_payload": None, "preliminary_final": False,
            }
        return {
            "route": ROUTE_SEARCH, "query": text, "plan": None,
            "echo_zh": "", "retrieval": None, "via": "rule_direct", "needs_agent": False,
            "suggestions": [],
            "result_payload": None, "preliminary_final": False,
        }

                                                 
                                               
                                                        
                                                  
                                  
    agent_cfg = None
    agent_path = False
    if use_agent and llm_call is None and _agent.agent_available():
        try:
            agent_cfg = config or load_llm_config()
        except Exception:
            agent_cfg = None
                                                                  
                                            
        if agent_cfg is not None and _ap.should_use_llm(agent_cfg)[0]:
            agent_path = True

    if _ask_answer is not None and not agent_path:
                                                                  
                                                        
                                                            
        return {
            "route": ROUTE_SEARCH, "query": text, "plan": None,
            "echo_zh": "", "retrieval": None, "via": "ask_answer_fallback",
            "needs_agent": False, "suggestions": [],
            "result_payload": None, "preliminary_final": False,
        }

    markers = detect_action_markers(text)
                                                                 
                                                          
                                                      
    holder: dict[str, Any] = {"flight": None, "markers": markers, "text": text,
                              "sources": sources, "search_params": search_params}
                                                          
                                                             
                                                       
                                                   
                                           
                                                     
                                                
                    
    route_extra_zh = ""
    if agent_path and _ask_answer is None:
        if not markers:
                                             
                                                                     
            if (not _agent._hybrid_intent_gate(text)) and _prelim_gate_skip(text):
                holder["prelim_gated"] = True
            else:
                                                                  
                _warmup_rag_environment(search_params)
                flight = _RagFlight(text, sources=sources, search_params=search_params)
                flight.start()
                holder["flight"] = flight
        route_extra_zh = _consensus_extra_zh(
            text, sources=sources, search_params=search_params)
                                                                    
                                                     

    emit_state: dict[str, Any] = {"preliminary_sent": False, "prelim_payload": None}
                                                           
                                                            
    recipe_verbs = _ap.resolve_suggested_recipe(suggested_recipe)
    plan: dict[str, Any] | None = None
    agent_used = False
    agent_fell_back = False
    if agent_path:
                                        
                                                           
                                                
                                                        
        _intent_checklist: list[dict[str, Any]] | None = None
        _pending_frontend: list[dict[str, Any]] | None = None
        if markers and _ask_answer is None:
            try:
                _intents = _ap.plan_action_intents(
                    text, has_results=has_results, result_total=result_total,
                    config=config, llm_call=llm_call, retrieval=None,
                    current_query=current_query, current_filters=current_filters,
                    allowed_verbs=recipe_verbs,
                )
            except Exception:                                       
                _intents = None
            if _intents is None:
                                                       
                try:
                    _fe_probe = _ap.plan_action(
                        text, has_results=has_results, result_total=result_total,
                        config=config, llm_call=llm_call, retrieval=None,
                        current_query=current_query, current_filters=current_filters,
                        allowed_verbs=recipe_verbs,
                    )
                except Exception:                                           
                    _fe_probe = None
                if (_fe_probe is not None
                        and str(_fe_probe.get("kind") or "") == _ap.EXEC):
                    _, _intent_checklist, _pending_frontend = _partition_intents([_fe_probe])
            else:
                plan, _intent_checklist, _pending_frontend = _partition_intents(_intents)
        if plan is None:
            try:
                                                                   
                                                                   
                _sp = search_params or {}
                _agent_search_kwargs = {
                    "search_sources": sources,
                    "search_facet_filters": _sp.get("facet_filters"),
                    "search_suppressed_constraints": _sp.get("suppressed_constraints"),
                    "search_lenient_dims": _sp.get("lenient_dims"),
                    "search_date_from": str(_sp.get("date_from") or ""),
                    "search_date_to": str(_sp.get("date_to") or ""),
                                                                
                                                                             
                                                        
                                                 
                    "search_top_k": _sp.get("top_k"),
                    "search_recall": str(_sp.get("recall") or ""),
                    "search_strategy": str(_sp.get("strategy") or ""),
                }
                if conversation_history is not None:
                    _agent_search_kwargs["conversation_history"] = conversation_history
                                                               
                                                                             
                                                                          
                                                            
                                                             
                                                                            
                                                              
                if _ask_answer is not None:
                    _provider = None
                    _verdict_hook = None
                else:
                    _provider = _make_retrieval_provider(
                        holder, agent_path=True, on_event=on_event, state=emit_state)
                    _verdict_hook = _make_route_verdict_hook(holder)
                if on_event is not None:
                    plan, _trace = _agent.plan_with_agent_events(
                        text,
                        has_results=has_results, result_total=result_total,
                        config=agent_cfg, retrieval=None,
                        current_query=current_query, current_filters=current_filters,
                        on_event=on_event, principal=principal,
                        retrieval_provider=_provider, on_route_verdict=_verdict_hook,
                        route_extra_zh=route_extra_zh,
                        route_scope=("search" if _ask_answer is not None else ""),
                        ask_answer=_ask_answer,
                                                       
                                               
                        allowed_verbs=recipe_verbs,
                                               
                        artifact_context=artifact_context,
                        intent_checklist=_intent_checklist,
                        **_agent_search_kwargs,
                    )
                else:
                                                          
                                                                    
                    plan, _trace = _agent.plan_with_agent(
                        text,
                        has_results=has_results, result_total=result_total,
                        config=agent_cfg, retrieval=None,
                        current_query=current_query, current_filters=current_filters,
                        principal=principal,
                        retrieval_provider=_provider, on_route_verdict=_verdict_hook,
                        route_extra_zh=route_extra_zh,
                        route_scope=("search" if _ask_answer is not None else ""),
                        ask_answer=_ask_answer,
                                           
                        allowed_verbs=recipe_verbs,
                                               
                        artifact_context=artifact_context,
                        intent_checklist=_intent_checklist,
                        **_agent_search_kwargs,
                    )
                agent_used = True
                if _pending_frontend:
                                                         
                                                             
                    plan["pending_frontend"] = _pending_frontend
            except _agent.AgentError:
                plan = None                              
                agent_fell_back = True
            except Exception as exc:
                                           
                                                          
                                                         
                plan = None
                agent_fell_back = True
                try:
                    _agent._audit_fallback(
                        _agent._agent_project_root(), "turn",
                        f"unhandled:{type(exc).__name__}:{str(exc)[:160]}", text, "")
                except Exception as audit_exc:
                                     
                                                                
                    _agent._warn_once(
                        f"turn_audit_fallback::{type(audit_exc).__name__}",
                        "turn 的降级审计落盘失败（仅记异常类型，不含正文/密钥）。")
    hybrid_abstain = False
    if plan is None:
                                          
                                                                    
                                                                   
                                                           
        if holder["flight"] is None and not markers:
            _warmup_rag_environment(search_params)
            flight = _RagFlight(text, sources=sources, search_params=search_params)
            flight.start()
            holder["flight"] = flight
        _flight = holder["flight"]
        _retrieval_for_plan = _flight.join() if _flight is not None else None
                                               
                                                         
        hybrid_abstain = _agent._hybrid_intent_gate(text)
        if hybrid_abstain:
            plan = _ap._blank_plan(
                source="rule",
                llm_status="hybrid_abstain",
                reason_zh="混合诉求(操作+检索串联):单次单动词保底通道做不全,机械弃权。",
            )
            plan["hybrid_abstain"] = True
        else:
            plan = _ap.plan_action(
                text,
                has_results=has_results, result_total=result_total,
                config=config, llm_call=llm_call,
                retrieval=_retrieval_for_plan,
                current_query=current_query, current_filters=current_filters,
                allowed_verbs=recipe_verbs,                           
                **({"conversation_history": conversation_history} if conversation_history is not None else {}),
            )
        if agent_fell_back:
                                                                
                                                            
                                               
            plan["agent_fallback"] = True
                                                  
            _te.emit_finish_reason(kind="agent_fallback")
                                                                
        if (str(plan.get("kind") or "") != _ap.EXEC
                and str(plan.get("verb") or "") != "none"):
            _emit_preliminary(_flight, agent_path=agent_path,
                              on_event=on_event, state=emit_state)

                                               
                                                 
    plan = _recipe_narrow_plan(plan, recipe_verbs)

                                                       
                                                                  
                                                                             
                                       
                                                             
    _flight = holder["flight"]
    is_tool_route = bool(plan is not None and str(plan.get("kind") or "") == _ap.EXEC)
    retrieval: dict | None = None
    retrieval_note = ""
    if is_tool_route:
        if _flight is None:
            retrieval_note = ("skipped_gate" if holder.get("prelim_gated")
                              else "skipped_action_marker")
        else:
            retrieval_note = "discarded_action_route"
            if _flight.done():
                _flight.join()                                    
    elif _flight is not None:
        retrieval = _flight.join()
        if prefetched_out is not None and _flight.meta is not None:
            prefetched_out.append({"query": text, "payload": _flight.ensure_payload()})
    preliminary_sent = bool(emit_state["preliminary_sent"])
    prelim_payload = emit_state["prelim_payload"]
    trace_preliminary = (
        "emitted" if preliminary_sent
        else ("skipped_marker" if markers
              else ("suppressed_action" if is_tool_route
                    else ("suppressed_gate" if holder.get("prelim_gated") else ""))))

                                                  
                                                             
                                                                    
                                                       
                                    
                                                      
    result_payload: dict[str, Any] | None = None
                                 
                                                           
                              
    loop_payload: dict[str, Any] | None = None
    batches: list[dict[str, Any]] = []
    if prelim_payload is not None:
        batches.append({
            "kind": "preliminary",
                                          
                                                                                   
                                        
            "label": text[:20],
            "query_raw": text,
            "query_effective": text,
            "payload": prelim_payload,
        })
    for _step in list(plan.get("steps") or []):
        _res = _step.get("result") if isinstance(_step, dict) else None
        if (isinstance(_res, dict) and _step.get("ok")
                and str(_step.get("verb") or "") == "search.rerun"
                and _res.get("adopted") and isinstance(_res.get("payload"), dict)):
            loop_payload = _res["payload"]
            _rerun_batch = {
                                                                     
                "kind": "rescue" if _res.get("replace_screen") else "search_rerun",
                "label": str(_res.get("query") or "")[:20],
                                                              
                "query_raw": text,
                "query_effective": str(_res.get("query") or ""),
                "payload": _res["payload"],
            }
                                                  
                                                        
            _rerun_disc = str(_res.get("disclosure_zh") or "").strip()
            if _rerun_disc:
                _rerun_batch["disclosure_zh"] = _rerun_disc
            batches.append(_rerun_batch)
        elif (isinstance(_res, dict) and _step.get("ok")
                and str(_step.get("verb") or "") in ("rank", "rerank", "results.show")
                and isinstance(_res.get("batch"), dict)
                and isinstance(_res["batch"].get("payload"), dict)):
            loop_payload = _res["batch"]["payload"]
                                                                                   
                                                                        
                                               
            batches.append({
                "kind": str(_res["batch"].get("kind") or _step.get("verb")),
                "label": str(_res["batch"].get("label") or "")[:20],
                "query_raw": str(_res["batch"].get("query_raw") or ""),
                "query_effective": str(_res["batch"].get("query_effective") or ""),
                "payload": _res["batch"]["payload"],
            })
                                              
                                                             
                                                       
                                                         
                                               
    result_payload = loop_payload
    if result_payload is None and batches:
        result_payload = batches[-1]["payload"]
    extra: dict[str, Any] = {}
                                                              
                                                              
                                                         
    for _s in reversed(list(plan.get("steps") or [])):
        if not (isinstance(_s, dict) and _s.get("ok")):
            continue
        if str(_s.get("verb") or "") != "ask.relax":
            continue
        _ar = _s.get("result") if isinstance(_s.get("result"), dict) else {}
        if _ar.get("asked") and isinstance(_ar.get("ask_relax"), dict):
            extra["ask_relax"] = _ar["ask_relax"]
        break
    if batches:
        _turn_id = uuid4().hex
        _now = datetime.now(timezone.utc).isoformat()
        for _i, _b in enumerate(batches):
            _b["batch_id"] = f"b{_i + 1}"
            _b["seq"] = _i + 1
            _b["created_at"] = _now
            _b["turn_id"] = _turn_id
                                   
            _b["scope_fingerprint"] = _batch_scope_fingerprint(
                str(_b.get("query_effective") or ""), sources, search_params)
                                                             
        extra.update({"result_batches": batches,
                      "active_batch": batches[-1]["batch_id"]})

    if plan.get("kind") == _ap.EXEC:
        return {
            "route": ROUTE_TOOL, "query": "", "plan": plan,
            "echo_zh": "", "retrieval": None, "retrieval_note": retrieval_note,
            "via": str(plan.get("source") or ""), "needs_agent": False,
            "suggestions": [],
            "result_payload": result_payload, "preliminary_final": False,
            "_preliminary_trace": trace_preliminary,
            **extra,
        }
    if str(plan.get("verb") or "") in _ap.ROUTE_QUERY_VERBS:
                                                                 
        final_query = str(plan.get("effective_query") or "").strip() or text
                                                       
                                                           
                                                          
                                                                 
                                                                         
                                                                     
                                                          
                                                      
        polish_flag = bool((search_params or {}).get("polish", True))
        llm_armed = True            
        try:
            llm_armed = bool(_ap.should_use_llm(config or load_llm_config())[0])
        except Exception:
            llm_armed = True
        polish_will_run = polish_flag and llm_armed
        preliminary_final = bool(
            preliminary_sent and loop_payload is None and final_query == text
            and str((search_params or {}).get("rerank") or "off") == "off"
            and not polish_will_run)
        return {
            "route": ROUTE_SEARCH,
            "query": final_query,
            "plan": plan, "echo_zh": "", "retrieval": retrieval,
            "via": "agent" if agent_used else "llm", "needs_agent": False,
            "suggestions": [],
            "result_payload": result_payload,
            "preliminary_final": preliminary_final,
            "_preliminary_trace": trace_preliminary,
            **extra,
        }
                                                  
                                             
    if hybrid_abstain:
        return {
            "route": ROUTE_NONE, "query": "", "plan": plan,
                                                               
            "echo_zh": _HYBRID_ABSTAIN_AGENT_DOWN_ZH if agent_fell_back else _HYBRID_ABSTAIN_ZH,
            "retrieval": retrieval, "via": str(plan.get("source") or ""),
            "needs_agent": False, "suggestions": [],
            "result_payload": result_payload, "preliminary_final": False,
            "_preliminary_trace": trace_preliminary,
            **extra,
        }
                                                       
                                                   
                                                      
                                            
                                            
    if str(plan.get("verb") or "") == "none" and str(plan.get("source") or "") == "rule":
        if search_shaped(text, keyword_mapping=keyword_mapping) and not _ap.rule_operation_marker(text):
            return {
                "route": ROUTE_SEARCH, "query": text, "plan": plan,
                "echo_zh": "", "retrieval": retrieval, "via": "rule_fallback",
                "needs_agent": False, "suggestions": [],
                "result_payload": result_payload, "preliminary_final": False,
                "_preliminary_trace": trace_preliminary,
                **extra,
            }
        return {
            "route": ROUTE_NONE, "query": "", "plan": plan,
            "echo_zh": (
                "大模型这次没有接上（" + _llm_absent_zh(plan.get("llm_status")) + "），"
                "这句我只能按关键词猜，没有认出能直接帮你做的事。"
                "检查更新、删除、联网搜这类操作必须大模型在场才能判断——"
                "可以再发一次试试，或到「设置」里检查大模型连接。"
            ),
            "retrieval": retrieval, "via": str(plan.get("source") or ""),
                                                  
            "needs_agent": False, "suggestions": [],
            "result_payload": result_payload, "preliminary_final": False,
            "_preliminary_trace": trace_preliminary,
            **extra,
        }
                                                                
                                                            
                                            
    downgraded_reason = (
        str(plan.get("reason_zh") or "").strip() if plan.get("downgraded_from") else ""
    )
    return {
        "route": ROUTE_NONE, "query": "", "plan": plan,
        "echo_zh": downgraded_reason or "这句话我没有听懂，什么都没有做。",
        "retrieval": retrieval, "via": str(plan.get("source") or ""),
        "needs_agent": False,
                                                    
        "suggestions": _none_route_suggestions(has_results=bool(has_results)),
        "result_payload": result_payload, "preliminary_final": False,
        "_preliminary_trace": trace_preliminary,
        **extra,
    }


def route_turn(
    utterance: str,
    *,
    has_results: bool = False,
    result_total: int = 0,
    current_query: str = "",
    current_filters: Any = None,
    sources: Any = None,
    config: LLMConfig | None = None,
    llm_call: Callable[[str], str | None] | None = None,
    keyword_mapping: dict | None = None,
    use_agent: bool = True,
    on_event: Callable[[str, dict], None] | None = None,
    principal: str = "",
    search_params: dict | None = None,
    artifact_context: str = "",
    suggested_recipe: str = "",
    ask_answer: dict | None = None,
    conversation_history: list[dict] | None = None,
    prefetched_out: list | None = None,
) -> dict[str, Any]:
    """turn 入口（签名/返回契约与 `_route_turn_impl` 逐位一致——本壳只做 trace 接线）。

    `suggested_recipe`（2026-08-22 additive）：结果页阶梯 chip /
    任务卡**未经编辑**的模板文本随请求携带的建议动作 id。allowlist 校验在
    `action_plan.resolve_suggested_recipe`（单一真源）：非法 → 忽略按普通路由处理，
    响应带 additive `recipe_note` 如实记录；合法 → 只缩小动词选择面（plan_action
    allowed_verbs + agent 路径产出的 plan 机械收窄），绝不扩权/不绕执行开关/安全闸。


    入口建 `TraceRecorder` 并经 contextvars 绑进本线程 context（webapp 的 SSE worker
    线程内天然隔离；既有函数零签名变更），图内各挂钩点（llm_call/tool_call/…）据此
    落盘；impl 返回后统一发一次 route_decision（含 understand/route_consensus 原始
    投票）。`AGENT_TRACE` OFF 时 recorder.enabled=False——零落盘、不加 trace_turn_id
    键、行为与旧版逐位一致。trace 一切故障 fail-soft，绝不掀翻路由。
    """
    rec = _recorder_for_turn(_agent._agent_project_root(),
                             session_id=principal or "anonymous")
                                                                
                                                                    
                                                   
                                                              
    from ..corpus.patch_package import bind_patch_scope
    with _bind_recorder(rec), bind_patch_scope(
            principal if principal and principal != "anonymous" else None):
        result = _route_turn_impl(
            utterance,
            has_results=has_results, result_total=result_total,
            current_query=current_query, current_filters=current_filters,
            sources=sources, config=config, llm_call=llm_call,
            keyword_mapping=keyword_mapping, use_agent=use_agent,
            on_event=on_event, principal=principal, search_params=search_params,
            artifact_context=artifact_context,
            suggested_recipe=suggested_recipe,
            ask_answer=ask_answer,
            **({"conversation_history": conversation_history} if conversation_history is not None else {}),
            **({"prefetched_out": prefetched_out} if prefetched_out is not None else {}),
        )
        if rec.enabled:
                                                    
            result["trace_turn_id"] = rec.turn_id
            _te.emit_route_decision(result)
                                              
                                                      
        recipe_verbs = _ap.resolve_suggested_recipe(suggested_recipe)
        if suggested_recipe and recipe_verbs is None:
            result["recipe_note"] = (
                f"suggested_recipe「{str(suggested_recipe).strip()}」不在 allowlist，"
                "已忽略按普通路由处理。"
            )
        elif recipe_verbs is not None:
            _plan = result.get("plan") if isinstance(result.get("plan"), dict) else None
            if _plan and _plan.get("recipe_narrowed"):
                result["recipe_note"] = (
                    f"建议动作路由到能力外动词「{str(_plan['recipe_narrowed'])}」，"
                    "已收窄为不执行。"
                )
    return result
