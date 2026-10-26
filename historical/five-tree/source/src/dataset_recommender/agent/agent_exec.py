                       
"""执行侧 Agent 规划与有界多步执行（langgraph 编排）。

设计蓝本：设计文档 + 设计文档
+ 设计文档（本版 = 换装后形态）。图拓扑与全部机械护栏自 2026-08-04
起逐位未动；2026-08-07 换装把「怎么用 langgraph」迁到 1.x 地道用法：

    understand → validate →（repair ≤1 次 → validate）→ execute → decide
                      ↑__________________________________|（还有下一步）
                      （done / 停环）→ narrate

换装五刀（与旧实现的差异面，全部经差分 harness + 真机探针双闸验收）：

1. **图模块级编译一次**：六节点是模块级函数（不再是每请求重建的闭包），`_get_graph()` 懒加载
   单例 + threading.Lock 护首轮构建；每请求依赖（chat_model / model_name）走
   **Context API**（`StateGraph(..., context_schema=_AgentContext)` + 节点 kwarg `runtime`
   注入——官方 v0.6+ 的 run-dependencies 通道），state 保持纯数据。
2. **reducer**：`trace`/`steps`/`observations`/`dead_ends` 四个 key 是
   `Annotated[list, operator.add]`——节点只返回**增量**；plan["steps"] 仍写全量快照
   （前端把非空 steps 当「后端已执行」的所有权令牌，契约不变）。
3. **流式 = `stream_mode="values"`**：每帧是 reducer 应用后的全量 state，末帧即终态
   （删掉手工 merged 合并循环）；`on_event` 按 trace 长度 diff 发增量事件，事件协议不变；
   `config={"recursion_limit": 50}` 框架级保险（MAX_STEPS=8 下最深路径 ≈26-30 节点次）。
4. **decide 迁 tool-calling 主通道**：绑 5 个 loop 工具 + `finish` + `unsupported_next_step`
   （非 loop 动词真枚举）共 7 个工具（**不是** 18 动词全表——不把不可在循环内执行的动作
   伪装成可调用工具），`tool_choice="auto"` + `parallel_tool_calls=False`：回 loop
   工具=续步、finish=done、unsupported=婉拒（declined_zh 语义与旧版
   逐位一致）、幻觉工具名/散文=非法=停环；**多 tool_call 取第一个**（DeepSeek 不遵守 parallel_tool_calls=False，多调用是常态且第一个调用实证
   17/17 合法，循环会再判断后续，不吞事）——**2026-08-14 起追加同批只读消费**：
   第 2..N 个调用里只读白名单（check_updates/db_status）且互相独立、逐个过全套机械闸的
   续步随首步同批执行（探针实测 A 类批量 2..N 合法率 100%，n=371）；写动词/未知工具名
   出现即截断，其余回炉由循环带新状态再判（`_batch_readonly_extras`）。**仅当调用本身
   抛异常**才跌散文 JSON 兜底
   （按当前面装配的 JSON 壳全文再问一次，记 agent_fallbacks.jsonl node="decide"）；拿到但非法
   的回答维持旧版「直接停环不再问」的 fail-safe。
5. **工具调用三级通道收口为共享助手** `_invoke_tool_channel`：understand 用 required 起、
   decide/repair 用 auto 起，同一份降档/兜底代码（repair 从此也走结构化通道，不再是
   无 bind 的散文调用）。

各节点职责（与旧版逐位一致）：

- **understand**：`bind_tools` 挂**由 `action_plan.VERB_SPECS` 程序生成的能力工具表**
  （单一真源不漂移；parameters 由 `agent_schemas` 逐动词 pydantic 生成），三级通道——
  `tool_choice="required"` 强制档优先；模型 400 拒收强制档 → 自动档 `"auto"` 重试；
  provider 彻底不支持 tool-calling / 应答不可用 → **图内降级** JSON-in-prompt 再解析一次。
- **validate**：机械护栏镜像校验——**安全围栏不是 LLM 自觉，是代码**。违规写进
  `violations`；通过则当场用同一套 `build_plan_from_raw` 产出 plan。循环续步时
  产出挂 `loop_plan`（plan.verb 恒为首步动词，前端契约不炸）。
- **repair**（≤1 次，**只服务首步**）：把 violations 反馈给 LLM 自修；仍非法 →
  `AgentPlanInvalid`。
- **execute**：动词命中 `LOOP_TOOLS` 注册表且未取消 → **图内真跑工具**；run() 出口过
  `agent_schemas.LOOP_RESULT_MODELS` 形状闸；step 经 Step 模型构造再 model_dump 实录；
  工具抛异常 → ok=False 记 error，不炸图；每跑一个工具往联网账本追加一行审计；
  终态码失败记死路账。注册表之外动词 → 空过。
- **decide**（仅 execute 真跑过工具后进入）：LLM 看「原话 + 已完成步骤紧凑投影」决定
  finish/续步/婉拒。机械校验双保险 + MAX_STEPS=8 硬上界（2026-08-15 由 6 放宽，
  同批落地**到顶结算闸**：跑满上限时 pending 硬闸 + 清单对账双双结清则不谎报
  「没做完」；写步 MAX_WRITE_STEPS=2、写条数 MAX_WRITE_RECORDS=40 双独立预算不随
  总步数放宽——search_online 的 network_error 失败可证零副作用、不占写步预算）
  + 连续失败处置二分（联网二连败改联网暂停——
  离线快照源检查不连坐；非网络二连败禁提失败动作，均不硬停；**同批只读续步逐个用
  当前实录重过这两道闸**——批内熔断，初筛对批前状态不够）。两类可修情形各给
  **一次**重问（共享同一份预算，每次 decide 调用至多一次）：主通道非法应答（散文拒答/
  幻觉工具名）重问一次；续步提议没通过 `_validate_raw` 校验（violation）
  带检查意见重问一次；**机械闸拒收同样回灌一次**（
  拒绝当前提议 ≠ 整条请求完成，尾随只读事项必须有机会推进）。重问后放行的写动词落
  强制核销账，finish 报告必须引用其步骤号单独交代。仍非法 → 当 done 处理
  （fail-safe 停环，不走 repair）。
- **narrate**：确定性拼接收口（`source="agent"` 标记）；steps 非空时由 LLM 据步骤紧凑投影
  写整段汇报 plan.report_zh，LLM 缺席/失败回退确定性拼接（同一批事实）；单步 db_status
  的既有汇报路径逐位保留。**机械后检** `_report_contradiction_reason` 五路拦「与实录矛盾」
  的汇报 → 弃用回退，原文与原因码留痕 trace。

## 三条刻意守住的边界

1. **图内执行仅限 `LOOP_TOOLS` 注册表**：db_status（只读）、check_updates（只读在线比对）、
   search_online（联网搜 + 入库——写操作，依据是产品方 2026-08-03 的全自动化预先授权：
   confirm_token 图内闭环回传 = 机械确认，后端重算指纹 fail-closed；每步记账 + 回收站
   可撤销）、sync_updates（检查更新→能闭环来源自动入库的复合流，2026-08-06「工作流即工具」）、
   search.rerun（换词重检，只读本地检索 + 机械择优闸，2026-08-16 检索工具化 Phase 1）。
   注册表之外的动词**一律不执行**，产物照旧交前端 runner（plan 契约因此对那些动词零变化）。
2. **plan 契约 additive**：产出经过 `action_plan.build_plan_from_raw` / `_finalize` **同一套**
   机械护栏（复用 import，不复制逻辑），形状与 action_plan 输出逐位同形，仅 `source="agent"`
   并附 `trace`；observation/report_zh/steps 是 additive 字段，plan.verb 恒为首步动词。
3. **可选依赖 + 保底回退**：langgraph/langchain 只在函数内**惰性 import**（模块级 import
   不许碰 langchain）；未安装或被 `BIODATA_AGENT_EXEC=off` 关停时 `agent_available()` 为 False，
   调用方（turn.route_turn）原样回退 `action_plan.plan_action`——本模块抛出的 `AgentError`
   也由调用方捕获回退，agent 路径**绝不**成为新的单点。

失败契约：`AgentUnavailable`（依赖缺席 / 大模型关、mock、无 key / 客户端构造失败）与
`AgentPlanInvalid`（repair 后仍过不了护栏），共同基类 `AgentError`。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import operator
import os
import re
import sys
import threading
import time
import uuid
from . import learning_scope as _learning_scope
from . import run_control as _run_control
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, TypedDict

from . import action_plan as _ap
from . import collab_rules as _collab_rules
from . import hybrid_policy as _hybrid_policy
from . import decider as _decider
from . import l0_router as _l0
from ..retrieval import search_request as _sr
from ..domain.registry import domain_catalog as _domain_catalog, domain_labels_zh as _domain_labels_zh, get_domain as _get_domain
from ..domain import vocab_tools as _vocab_tools
                                                             
                                                         
from ..retrieval.query_parser import _alias_occurrences as _qp_alias_occurrences
from .agent_schemas import (
    LOOP_RESULT_MODELS as _LOOP_RESULT_MODELS,
    Step as _Step,
    source_candidates_zh as _source_candidates_zh,
    verb_parameters_schema as _verb_parameters_schema,
)
from ..llm.llm_client import LLMConfig, complex_model_name as _complex_model_name
from ..app.runtime_paths import instance_data_dir_for, resource_file_for

                                          
                                                                 
                                                            
from .trace import snapshot_store as _trace_snapshot_store
from .trace import events as _te
from .trace import rollback as _trace_rollback
from .trace import SnapshotError as _SnapshotError

                                                        
                                                            
from pydantic import ValidationError as _ValidationError

__all__ = [
    "AgentError",
    "AgentUnavailable",
    "AgentPlanInvalid",
    "LOOP_TOOLS",
    "MAX_STEPS",
    "agent_available",
    "plan_with_agent",
    "plan_with_agent_events",
]


class AgentError(Exception):
    """Agent 规划失败的共同基类——调用方（turn.route_turn）捕获后回退 action_plan 保底。"""


class AgentUnavailable(AgentError):
    """Agent 不可用：langchain 依赖缺席 / 被 env 关停 / 大模型关、mock 或无 key / 客户端构造失败。"""


class AgentPlanInvalid(AgentError):
    """repair（≤1 次）后 LLM 输出仍过不了机械护栏。`violations` 带人读违规清单（供日志/调试）。"""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        super().__init__("；".join(self.violations))


class _AgentState(TypedDict, total=False):
    """langgraph 图状态。六个累积型 key 带 reducer（后增补 reask_writes 与 usage_ledger）：

    `trace`/`steps`/`observations`/`dead_ends`/`reask_writes`/`usage_ledger` 是 `Annotated[list, operator.add]`——
    节点只返回**增量**，reducer 负责合并（reducer 必须是 Annotated 的最后一项，
    否则静默退化为覆盖且建图不报错——社区实测 footgun）；节点内读到的 state 是
    reducer 应用后的全量（`_is_duplicate_step`/接地语料等读 steps 的代码零改动）。
    其余 key 保持默认覆盖语义（单一归属节点的标量/最终产物）。"""

    utterance: str
    retrieval: dict | None
    current_query: str
    current_filters: Any
    has_results: bool
    result_total: int
    mode: str                                                          
    raw: dict[str, Any]                                                     
    violations: list[str]                              
    repairs: int                                  
    plan: dict[str, Any]                                                       
    loop_plan: dict[str, Any]                                                     
    raw_batch: list[dict]                                                                                               
    loop_batch: list[dict]                                                          
    loop_next: bool                                                            
    last_ran: bool                                                           
    truncated: bool                                                          
    truncated_settled: bool                                                      
    trace: Annotated[list[dict], operator.add]                                                 
    observations: Annotated[list[dict], operator.add]                         
    steps: Annotated[list[dict], operator.add]                                     
    declined_zh: str                                                   
    dead_ends: Annotated[list[dict], operator.add]                                   
    finish_vetoes: int                                                    
    reask_writes: Annotated[list[dict], operator.add]                                           
    pending_reask_write: bool                                                     
    usage_ledger: Annotated[list[dict], operator.add]                                                 
    checklist: list[dict]                                                             
    checklist_unavailable: str                                             
    checklist_dropped: int                                                          
    search_sources: Any                                                                                    
                                                     
                                                                    
    search_facet_filters: Any
    search_suppressed_constraints: Any
    search_lenient_dims: Any
    search_date_from: str
    search_date_to: str
                                                           
                                                              
                                               
                           
    search_top_k: Any
    search_recall: str
    search_strategy: str
    route_scope: str                                                                                                                                 
    required_capabilities: list[dict]                                                                                                                      
    artifact_context: str                                                                                     
    intent_checklist: list[dict]                                                                                             
    ask_answer: dict | None                                                                                                                                                                     
    allowed_verbs: "frozenset[str] | None"                                                                          


@dataclass(frozen=True)
class _AgentContext:
    """langgraph 的 `context_schema`：每请求的 run 依赖（官方 v0.6+ 的依赖注入通道，
    取代旧的 config["configurable"] 模式）。frozen=True 让「immutable context」不只是注释。
    chat_model 是请求级现建的实例（config 是请求级快照，不缓存 client 是刻意的防串配设计）；
    model_name 供跌 JSON 兜底时的审计行如实标注。
    decide_model/decide_model_name/decide_lane（2026-08-07 复杂度路由）：
    decide 节点专用档——仅当 utterance 评分进 complex 车道且配置了 LLM_MODEL_COMPLEX 时
    非 None；understand/validate/repair/narrate 恒走 chat_model（问题数据在案）。"""

    chat_model: Any
    model_name: str = ""
    decide_model: Any = None
    decide_model_name: str = ""
    decide_lane: str = "simple"
                                 
                                                                            
                                                        
                            
    principal: str = ""
    endpoint_fp: str = ""
    project_id: str = ""
    cancel_check: Any = None
                                                      
                                                                      
                                                            
    on_progress: Any = None
                                         
                               
                                                                                
                                                                  
                                                                                 
                                                      
                                                        
    retrieval_provider: Any = None
    on_route_verdict: Any = None
    route_extra_zh: str = ""
                                          
                                                   
    artifact_context: str = ""
    history_context: str = ""


def agent_available() -> bool:
    """langgraph + langchain-openai 已安装、且未被 `BIODATA_AGENT_EXEC=off` 强制关停。

    只用 find_spec 探测、**不 import**——本模块的 langchain 依赖全部在函数内惰性加载，
    未装扩展时 import 本模块零代价（turn.py 模块级 import 本模块因此恒安全）。"""
    if str(os.environ.get("BIODATA_AGENT_EXEC") or "").strip().lower() == "off":
        return False
    try:
        return bool(
            importlib.util.find_spec("langgraph") and importlib.util.find_spec("langchain_openai")
        )
    except (ImportError, ValueError):                                                
        return False


def _build_chat_model(config: LLMConfig) -> Any:
    """按次构造 ChatOpenAI（**不缓存 client**）：config 是请求级快照，设置页的 key/端点
    经 env override 进来，缓存会把上一个请求的配置错发给下一个请求。

    DeepSeek V4 思考旋钮：config.thinking / reasoning_effort 非 None
    时经 `extra_body` 透传（langchain_openai 1.4.1 原生字段；未配置 = 请求体逐位不变）。"""
    from langchain_openai import ChatOpenAI

    extra_body: dict[str, Any] = {}
    if config.thinking is not None:
        extra_body["thinking"] = {"type": "enabled" if config.thinking else "disabled"}
        if config.thinking and config.reasoning_effort:
            extra_body["reasoning_effort"] = config.reasoning_effort
    try:
        return ChatOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=float(config.timeout),
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
            **({"extra_body": extra_body} if extra_body else {}),
        )
    except Exception as exc:
        raise AgentUnavailable(
            f"Agent 的 LLM 客户端构造失败（{type(exc).__name__}）。"
        ) from exc


                                                                   
_COMPLEX_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _complex_thinking_env() -> tuple[bool | None, str | None]:
    """复杂度车道的思考旋钮 env（`LLM_COMPLEX_THINKING` / `LLM_COMPLEX_EFFORT`，任务1）。

    未配置 → (None, None)：complex 车道行为与「只换模型名」的旧版逐位一致。
    thinking=on 时 effort 才随附（非法值钳 None——坏配置静默降级为官方默认档，
    与 `LLM_MODEL_COMPLEX` 写错名「调用期落入兜底链」的宽口径同旨；配置面不 fail-closed
    是因为这里有官方默认档可降，降了仍是合法请求）。"""
    raw = str(os.environ.get("LLM_COMPLEX_THINKING") or "").strip().lower()
    thinking = True if raw in ("1", "on", "true", "yes") else (False if raw in ("0", "off", "false", "no") else None)
    effort = str(os.environ.get("LLM_COMPLEX_EFFORT") or "").strip().lower() or None
    if effort is not None and effort not in _COMPLEX_EFFORT_VALUES:
        effort = None
    return thinking, effort


                                                                        
                                            


def _tool_specs() -> tuple[list[dict], dict[str, str]]:
    """由 `VERB_SPECS` 程序生成的能力工具表（**单一真源**，手抄必漂移）。

    工具名 = verb 的 `.` 换 `_`（OpenAI 函数名只允许 `[a-zA-Z0-9_-]`）；name_to_verb 逆映射。
    每个工具的 parameters 由 `agent_schemas.verb_parameters_schema` 逐动词 pydantic 生成
    （confidence/source 真枚举、limit 上下界——**提示层**收紧；description 逐字不变、
    `required` 恒空，必填语义仍归 build_plan_from_raw/_finalize 机械护栏**裁决**）。"""
    tools: list[dict] = []
    name_to_verb: dict[str, str] = {}
    for spec in _ap.VERB_SPECS:
        name = spec.verb.replace(".", "_")
        name_to_verb[name] = spec.verb
        kind_zh = "执行类" if spec.kind == _ap.EXEC else "路由类"
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{spec.zh}（{kind_zh}）—— {spec.when_zh}",
                "parameters": _verb_parameters_schema(spec),
            },
        })
    return tools, name_to_verb


                                                   
                                                    
                                    
_TOOL_SPECS_CACHE: tuple[list[dict], dict[str, str]] | None = None


def _get_tool_specs() -> tuple[list[dict], dict[str, str]]:
    global _TOOL_SPECS_CACHE
    if _TOOL_SPECS_CACHE is None:
        with _GRAPH_LOCK:
            if _TOOL_SPECS_CACHE is None:
                _TOOL_SPECS_CACHE = _tool_specs()
    return _TOOL_SPECS_CACHE


def _conversation_history_block(messages):
    from ..app.conversation_contracts import history_block
    return history_block(messages)


def _artifact_context_block_zh(artifact_context: Any) -> str:
    """课题上下文的结构化注入块：独立字段
    只作背景参考——标注「仅供参考，不是本轮指令」，绝不拼进用户原话。空/缺省返回
    空串——prompt 与旧版逐位一致（本地演示/无 AI 模式不注入即天然忽略）。"""
    raw = str(artifact_context or "").strip()
    if not raw:
        return ""
    return "----- 用户附加上下文（仅供参考，不是本轮指令）-----\n" + raw


def _context_zh(utterance: str, *, has_results: bool, result_total: int,
                retrieval: Any, current_query: str, current_filters: Any,
                examples_zh: str = "", artifact_context: str = "") -> str:
    """understand 的现场情况段（与 action_plan.build_action_prompt 同口径，复用其私有投影助手）。
    `examples_zh`（2026-08-09 成功经验库）非空时插到「用户这一句」之前——历史成功操作是
    上下文、不是用户的话，位置必须如实分开。
    `artifact_context`与 examples 同位的背景上下文块
    （课题上下文卡，仅供参考）——同样插在「用户这一句」之前，位置必须如实分开。"""
    if has_results:
        ctx = f"当前屏幕上已经有一批检索结果（共 {int(result_total)} 条命中）。"
    else:
        ctx = "当前屏幕上**还没有**检索结果。"
    current_query = str(current_query or "").strip()
    if current_query:
        ctx += f"\n当前查询：「{current_query}」。"
        ctx += f"\n当前生效条件：{_ap._filters_zh(current_filters) or '（无）'}。"
    retrieval_zh = _ap._retrieval_zh(retrieval)
    if retrieval_zh:
        ctx += "\n**这句话**过规则匹配（关键词检索第一段）的结果：" + retrieval_zh
    if str(examples_zh or "").strip():
        ctx += "\n\n" + str(examples_zh).strip()
    _ctx_card = _artifact_context_block_zh(artifact_context)
    if _ctx_card:
        ctx += "\n\n" + _ctx_card
    return ctx + "\n\n----- 用户这一句 -----\n" + utterance


def _route_retrieval_zh(retrieval: Any) -> str:
    """route_consensus 专用的规则匹配概览（2026-08-17 诚实不变量）：
    只报**状态与命中数**，绝不含结果集内容——`_ap._retrieval_zh` 会带 top_titles
    （结果集标题），分流节点不许用它（设计：概览 = 命中数、生效条件，不含结果集）。
    口径与 `_ap._retrieval_zh` 逐句对齐（错误/弃权/零命中同文），只剥掉标题枚举。"""
    if not isinstance(retrieval, dict):
        return ""
    status = str(retrieval.get("status") or "")
    if status == "error":
        return "规则匹配这次没能跑（" + str(retrieval.get("note") or "内部原因") + "）。"
    if status == "abstained":
        terms = "、".join("「" + str(t) + "」" for t in (retrieval.get("unresolved_terms") or [])[:6])
        return ("规则匹配**整句弃权**（" + str(retrieval.get("abstain_reason") or "")
                + (("；未收录词：" + terms) if terms else "") + "）。")
    total = int(retrieval.get("total") or 0)
    if total <= 0:
        return "规则匹配**零命中**（库中没有同时满足所有条件的记录）。"
    return f"规则匹配命中 {total} 条。"


def _route_context_zh(utterance: str, *, has_results: bool, result_total: int,
                      retrieval: Any, current_query: str, current_filters: Any) -> str:
    """route_consensus 的现场段：与 `_context_zh` 同上下文面，两个刻意
    差别——检索概览用 `_route_retrieval_zh`（不含结果集标题，诚实不变量）；原话由调用方
    自拼「用户原话」尾段，本函数不附带（避免 `_context_zh` 自带的「用户这一句」造成
    原话进 prompt 两次）。route_consensus 常驻环首，本函数是其唯一现场段。"""
    if has_results:
        ctx = f"当前屏幕上已经有一批检索结果（共 {int(result_total)} 条命中）。"
    else:
        ctx = "当前屏幕上**还没有**检索结果。"
    current_query = str(current_query or "").strip()
    if current_query:
        ctx += f"\n当前查询：「{current_query}」。"
        ctx += f"\n当前生效条件：{_ap._filters_zh(current_filters) or '（无）'}。"
    retrieval_zh = _route_retrieval_zh(retrieval)
    if retrieval_zh:
        ctx += "\n**这句话**过规则匹配（关键词检索第一段）的结果：" + retrieval_zh
    return ctx


def _message_text(message: Any) -> str:
    """AIMessage → 文本（content blocks 形态拼出文本段）。三处调用点共用一份，不抄第三份。"""
    content = getattr(message, "content", "")
    if isinstance(content, list):                           
        content = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def _raw_from_message(message: Any, name_to_verb: dict[str, str]) -> dict[str, Any]:
    """AIMessage → raw dict：优先 tool_calls（逆映射回 verb），其次 content 里的 JSON
    （复用 action_plan 的容错解析真源）。取不到 → {}（交给 validate 记违规，不在此抛）。

    **多 tool_call 取第一个**：DeepSeek 实测**不遵守** `parallel_tool_calls=False`
    ——decide 不可读 17/17、understand「no_tool_calls」跌兜底 8/8 全是模型一次回了
    ≥2 个调用，且实测**第一个调用 17/17 是合法续步**（模型的批量调用是「规划
    先行」，后续动作本就不由本次调用决定——循环带着新状态会再判断，吃第一个不吞
    任何事；旧「静默只吃第一个会把后续调用吞掉」的顾虑在循环架构下不成立）。
    请求侧 `parallel_tool_calls=False` 保留恒上（provider 哪天遵守了，多调用自然消失）。"""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        if isinstance(call, dict):
            name, args = call.get("name"), call.get("args")
        else:                                                  
            name, args = getattr(call, "name", ""), getattr(call, "args", None)
        verb = name_to_verb.get(str(name or ""))
        if verb and isinstance(args, dict):
            return {"verb": verb, **args}
        return {}
    return _ap.parse_action_response(_message_text(message))


def _usage_record(answer: Any, node: str) -> dict | None:
    """从 AIMessage 抠 DeepSeek 磁盘缓存用量（2026-08-08 埋点）。

    langchain_openai 1.4.1 把 `usage.prompt_cache_hit_tokens` 透传到
    `usage_metadata.input_token_details.cache_read`（实测确认）。读不到（FakeModel、
    非 DeepSeek provider、老版本）→ None——调用方跳过，usage_ledger 保持缺席，
    离线测试路径的 plan 键集逐位不变。"""
    usage = getattr(answer, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    if not isinstance(input_tokens, int):
        return None
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read") if isinstance(details, dict) else None
    return {"node": node, "input": input_tokens,
            "cache_read": int(cache_read or 0),
            "output": int(usage.get("output_tokens") or 0)}


def _invoke_text_with_continuation(
    chat_model: Any,
    messages: list,
    *,
    usage_sink: list | None = None,
    usage_node: str = "",
    max_continuations: int = 1,
) -> Any:
    """文本应答调用 + **截断两段式续写**（2026-08-08 有界自愈重试器的唯一增量）。

    适用面：**content 文本**截断（finish_reason=length）——narrate 汇报、JSON 兜底档。
    续写把已收到的前半段回贴、请模型「接着写」，拼接两段后返回（response_metadata 取
    末次调用的，finish_reason 如实反映终态）。**tool_calls 截断不可续**（arguments 是
    残缺 JSON，协议层无法拼接）——原样返回，由调用方现有容错链处理；续写后再截断
    也不再续（上限 max_continuations 次，默认 1——总调用数有界 ≤2）。
    V4 思考模式的 reasoning_tokens 吃 max_tokens 额度，截断率比 V3 时代高，这一档
    从「罕见」升格为「必须有兜底」。"""
    from langchain_core.messages import AIMessage, HumanMessage

    def _sink(answer: Any) -> None:
        if usage_sink is not None and answer is not None:
            rec = _usage_record(answer, usage_node)
            if rec is not None:
                usage_sink.append(rec)

    answer = chat_model.invoke(messages)
    _sink(answer)
    for _ in range(max_continuations):
        finish = str((getattr(answer, "response_metadata", None) or {}).get("finish_reason") or "")
        if finish != "length":
            break
        if getattr(answer, "tool_calls", None):
            break            
        head = _message_text(answer)
        if not head.strip():
            break                                               
        cont = chat_model.invoke(list(messages) + [
            AIMessage(content=head),
            HumanMessage(content="接着刚才的继续，别重复。"),
        ])
        _sink(cont)
        tail = _message_text(cont)
        answer = AIMessage(
            content=head + tail,
            response_metadata=getattr(cont, "response_metadata", None) or {},
            usage_metadata=getattr(cont, "usage_metadata", None),
        )
    return answer


def _tool_choice_rejected(exc: Exception) -> bool:
    """这个异常是不是「模型不收这个 tool_choice」？思考模式模型（deepseek-v4-flash /
    deepseek-reasoner，2026-08-05 实测）对 `tool_choice="required"` 回 400，报文点名
    tool_choice："Thinking mode does not support this tool_choice"。按报文字样判定——
    本请求里只有这一个参数名会被点名，跨 provider 通用，不误伤超时/断网类异常
    （那些报文不含它）。"""
    return "tool_choice" in str(exc)


def _invoke_tool_channel_impl(
    chat_model: Any,
    *,
    tools: list[dict],
    messages: list,
    choice: str,
    json_prompt: str | None = None,
    refallback_on_empty: bool = False,
    name_to_verb: dict[str, str] | None = None,
    usage_sink: list | None = None,
    usage_node: str = "",
) -> tuple[Any | None, str, str, str, str]:
    """`_invoke_tool_channel` 的本体（2026-08-17 拆壳）：返回五元组
    `(answer, note, fallback_reason, json_error_type, channel)`——channel 是实际产出
    应答的档位（choice 原档 / "auto" 降档 / "json_fallback" / 全灭 ""），供公开壳的
    trace llm_call 留痕；语义与下方公开壳 docstring 逐字一致。"""
    from langchain_core.messages import HumanMessage

    def _attempt(ch: str) -> Any:
        bound = chat_model.bind_tools(tools, tool_choice=ch, parallel_tool_calls=False)
        return bound.invoke(messages)

    def _usable(answer: Any) -> bool:
        return bool(_raw_from_message(answer, name_to_verb or {}))

    fallback_reason = ""
    note = ""

    def _sink(answer: Any) -> None:
        if usage_sink is not None and answer is not None:
            rec = _usage_record(answer, usage_node)
            if rec is not None:
                usage_sink.append(rec)

    try:
        answer = _attempt(choice)
        _sink(answer)
        if refallback_on_empty and not _usable(answer):
            fallback_reason = "no_tool_calls"
        else:
            return answer, note, fallback_reason, "", choice
    except Exception as exc:
        fallback_reason = type(exc).__name__
        if _tool_choice_rejected(exc) and choice != "auto":
            try:
                answer = _attempt("auto")
                _sink(answer)
                note = "（模型不收强制档，已用自动档）"
                if refallback_on_empty and not _usable(answer):
                    fallback_reason = "no_tool_calls(auto)"
                else:
                                                                    
                                                               
                    return answer, note, "", "", "auto"
            except Exception as exc2:
                fallback_reason = f"{type(exc).__name__}→auto 仍 {type(exc2).__name__}"
    if json_prompt is None:
        return None, "", fallback_reason, "", ""
    try:
        answer = _invoke_text_with_continuation(
            chat_model, [HumanMessage(content=json_prompt)],
            usage_sink=usage_sink, usage_node=usage_node)
        return answer, "", fallback_reason, "", "json_fallback"
    except Exception as exc:
        return None, "", fallback_reason, type(exc).__name__, "json_fallback"


def _invoke_tool_channel(
    chat_model: Any,
    *,
    tools: list[dict],
    messages: list,
    choice: str,
    json_prompt: str | None = None,
    refallback_on_empty: bool = False,
    name_to_verb: dict[str, str] | None = None,
    usage_sink: list | None = None,
    usage_node: str = "",
) -> tuple[Any | None, str, str, str]:
    """工具调用通道的**统一三级实现**（2026-08-07 understand/repair/decide 共用）。

    档位语义：`choice` 档先跑；模型 400 拒收该 tool_choice（`_tool_choice_rejected`）且
    首档不是 auto → 降 auto 档重试（留在结构化通道，槽位抽取质量不滑坡）；仍不行且给了
    `json_prompt` → 散文 JSON 兜底再问一次。**注意：只有「调用本身抛异常」才降档/兜底**——
    拿到了回答但内容不可用，由 `refallback_on_empty` 决定是否视为「没拿到」继续降级
    （understand 是 True：拿不到动词判定值得再问一次；decide 是 False：拿到但非法 = 停环，
    绝不因为「没读懂」就多给一次提议写操作的机会）。

    `parallel_tool_calls=False` 恒上（契约层仍要求一次一个动作；DeepSeek 实测不遵守该
    参数——多调用的处置在 `_raw_from_message` / `_decide_answer_kind`：取第一个，
    证据见两函数 docstring）。

    返回 `(answer, note, fallback_reason, json_error_type)`：
    - answer：AIMessage；全部通道都没拿到 → None；
    - note：通道标注（「（模型不收强制档，已用自动档）」——直取成功/JSON 兜底为空串）；
    - fallback_reason：跌了 JSON 兜底时为非空（审计行如实标注的档位史）；
    - json_error_type：JSON 兜底这一问本身抛异常时的异常类型名（trace 措辞用）。

    trace 留痕：本壳在 impl 返回后统一发一次 trace llm_call
    （digest+延迟+档位+用量合计）——understand/repair/decide 单点覆盖；fail-soft，
    AGENT_TRACE OFF 时零构造零落盘。
    """
    t0 = time.monotonic()
    u0 = len(usage_sink) if usage_sink is not None else 0
    answer, note, fallback_reason, json_error_type, channel = _invoke_tool_channel_impl(
        chat_model, tools=tools, messages=messages, choice=choice, json_prompt=json_prompt,
        refallback_on_empty=refallback_on_empty, name_to_verb=name_to_verb,
        usage_sink=usage_sink, usage_node=usage_node)
    if _te.recorder_active():
        usage = None
        if usage_sink is not None and len(usage_sink) > u0:
            new_recs = usage_sink[u0:]
            usage = {"node": usage_node,
                     "input": sum(int(r.get("input") or 0) for r in new_recs),
                     "cache_read": sum(int(r.get("cache_read") or 0) for r in new_recs),
                     "output": sum(int(r.get("output") or 0) for r in new_recs)}
        _te.emit_llm_call(
            node=usage_node,
            model=str(getattr(chat_model, "model_name", "")
                      or getattr(chat_model, "model", "") or ""),
            prompt=(str(json_prompt or "") if channel == "json_fallback"
                    else "\n".join(str(getattr(m, "content", "") or "") for m in messages)),
            response=_message_text(answer) if answer is not None else "",
            ms=int((time.monotonic() - t0) * 1000),
            channel=channel, fallback_reason=fallback_reason, usage=usage)
    return answer, note, fallback_reason, json_error_type


                                                                         
                                                
_TOOLS_SYSTEM_ZH = (
    _ap._RULES_ZH
    + "铁律（违反任一条都是错误）：\n"
    "1. 从工具表里挑工具调用；表里没有对应动作时选 none。原话一口气要求多件"
    "**彼此独立且只读**的事（如「检查 A、B、C 有没有更新」）时，一次为每件事各发"
    "一个调用（同一工具可发多次，每个来源一个）；其余情况恰好一个。\n"
    "2. quoted 必须是用户原话里**逐字出现**的一段连续文字，不要改写、不要加字、不要翻译；"
    "选了执行类动作却给不出原文依据时，改选 none。\n"
    "3. 用户**明确说不做**某个动作时 → 动词照选，并填 cancelled=true；「能不能/要不要…吧」是征询，照常执行。\n"
    "4. limit 只在用户**明确说了条数**时填，否则不填；不要把年份、编号、版本号当条数。\n"
    "5. 只是在**描述要找什么数据** → 那是检索需求，选 none 或 search.new。\n"
    "6. 规则匹配零命中或整句弃权 **不等于** 这句话无效——工具调用句往往零命中。\n"
    "7. 选 search.new / refine.conditions / lookup.identifier 时必须同时填 effective_query："
    "一句完整、可独立执行的检索句；none 和执行类一律不填。\n"
)


                                    
                                                                  
_NAMED_SOURCE_VERBS: tuple[str, ...] = ("curate.check_updates", "curate.search_online")

                        
                                                   
                                                    
                                              
_NAMED_SOURCE_OPTIONAL_FILL_VERBS: tuple[str, ...] = ("curate.sync_updates",)


                                                                                            
                                                              
                                                   
 
                                                                  
                                                          
                       
                                                                 
                                                                      
                                                                                 
                             
                                      
                                                           
                                                                                
                                                           
                                                     
                                                                          
                                                                             
                                
 
                     
                                                                            
                                                              
                                                   
 
                                                                 
                                                                  
                                                       
                                                                                            

                                                      
                                                                     
                                                
                                                
                                        
                        
                                                   
                                                 
                                          
                                                 
                                                        
MAX_STEPS = 8

                                           
                                              
                                                          
                       
MAX_WRITE_STEPS = 2

                                  
                                                
                                                                   
MAX_WRITE_RECORDS = 40

                                             
                                                                     
                                                             
_WRITE_LOOP_TOOLS: frozenset[str] = frozenset(
    {"curate.search_online", "curate.sync_updates"})

_ROLLBACK_LOOP_TOOL = "curate.rollback"
MAX_ROLLBACK = 2


def _rollback_used(steps: list[dict]) -> int:
    """本轮已执行/尝试的回滚次数；独立预算避免 rollback↔rollback 空转。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == _ROLLBACK_LOOP_TOOL)

                                                               
                                                             
                                
MAX_SEARCH_RERUN = 1


def _search_rerun_used(steps: list[dict]) -> int:
    """已用换词重检次数（**从 steps 现算，不加新状态**）：search.rerun 步不论成败都计——
    提议过即消耗预算，防「失败换个说法再提」绕过上限空转。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "search.rerun")

                                                                    
                                              
                                                            
                                          
MAX_RANK = 2
MAX_RERANK = 1


def _rank_used(steps: list[dict]) -> int:
    """已用裸新检索次数（从 steps 现算，同 `_search_rerun_used` 口径：提议过即消耗）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "rank")


def _rerank_used(steps: list[dict]) -> int:
    """已用优化重检次数（从 steps 现算，同 `_search_rerun_used` 口径：提议过即消耗）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "rerank")


                                                                 
                                                    
                                                                  
MAX_SHOW = 2


def _show_used(steps: list[dict]) -> int:
    """已用上屏次数（从 steps 现算，同 `_search_rerun_used` 口径：提议过即消耗）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "results.show")


                                                          
                                         
                                                         
                             
MAX_ASK_RELAX = 1


def _ask_relax_used(steps: list[dict]) -> int:
    """已用放宽询问次数（从 steps 现算，同 `_show_used` 口径：提议过即消耗）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "ask.relax")


def _ask_relax_pending(steps: list[dict]) -> dict | None:
    """环是否正停在一张等作答的放松卡上（2026-09-14 ask 暂停/重启批）：从 steps
    现算——存在 ok 且 result.asked=True 的 ask.relax 步即「环已暂停等作答」
    （不落新状态键，与 `_screen_baseline` 的机械现取同哲学）。返回该步（取最近一次）。"""
    pending: dict | None = None
    for s in steps:
        if str(s.get("verb") or "") != "ask.relax":
            continue
        result = s.get("result")
        if s.get("ok") and isinstance(result, dict) and result.get("asked"):
            pending = s
    return pending


                                                   
           
                                                           
                          
                                                                            
                                                                            
MAX_COMPARE = 1
MAX_CITE_EXPORT = 1
MAX_COMPAT = 2
MAX_FAIR = 2


def _compare_used(steps: list[dict]) -> int:
    """已用数据集对比次数（从 steps 现算，同 `_search_rerun_used` 口径）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "compare.datasets")


def _cite_export_used(steps: list[dict]) -> int:
    """已用引文导出次数（从 steps 现算）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "cite.export")


def _compat_used(steps: list[dict]) -> int:
    """已用兼容查找次数（从 steps 现算）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "compat.find")


def _fair_used(steps: list[dict]) -> int:
    """已用 FAIR 自检次数（从 steps 现算）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "fair.check")

                                     
                                                                  
                                    
MAX_ROUTE_REQUEST = 1


def _route_request_used(steps: list[dict]) -> int:
    """已用换路线次数（从 steps 现算，同 `_search_rerun_used` 口径：提议过即消耗）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") == "route.request")

                                                                       
                                                                       
                                                          
                                                           
                                                                
                         
_TERMINAL_STEP_CODES: frozenset[str] = frozenset({"source_not_registered"})

                                               
                                           
                              
                              
                                           
                                             
                                          
_CONSECUTIVE_FAILURE_BREAKER = 2

                                                          
                                      
_NETWORK_LOOP_TOOLS: frozenset[str] = frozenset(
    {"curate.check_updates", "curate.search_online", "curate.sync_updates"})

                                      
                                                  
                                                     
_NETWORK_MORATORIUM_BLOCK_ZH = (
    "\n----- 联网暂停（机械约束）-----\n"
    "联网已连续失败两次：本回合不许再提议联网工具（check_updates/search_online/sync_updates），"
    "离线工具 curate.db_status 与**离线快照源**（CELLxGENE / EBI SCEA / HuBMAP / "
    "Single Cell Portal）的 check_updates 仍可用；没有可做的离线事就 finish"
    "（JSON 通道回 {\"done\": true}）。"
)

                                
                                                     
                           
_FAILED_STEP_BLOCK_ZH = (
    "\n----- 有步骤失败时的次序 -----\n"
    "先把没做过的**新事**做完；重试失败的事最多一次，放在最后。"
)

                               
                                           
_WRITE_BUDGET_BLOCK_ZH = (
    "\n----- 写步预算已用完（机械约束）-----\n"
    f"本次请求的写步（search_online/sync_updates）已达上限 {MAX_WRITE_STEPS} 次："
    "本回合不许再提议写工具；只读工具（check_updates/db_status）仍可用；"
    "没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                            
                                                              
_SEARCH_RERUN_BUDGET_BLOCK_ZH = (
    "\n----- 换词重检预算已用完（机械约束）-----\n"
    f"本次请求已换词重检 {MAX_SEARCH_RERUN} 次（一次请求最多 {MAX_SEARCH_RERUN} 次）："
    "本回合不许再提议 search.rerun；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                                    
_RANK_BUDGET_BLOCK_ZH = (
    "\n----- 新检索预算已用完（机械约束）-----\n"
    f"本次请求已新检索 {MAX_RANK} 次（一次请求最多 {MAX_RANK} 次）："
    "本回合不许再提议 rank；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)
_RERANK_BUDGET_BLOCK_ZH = (
    "\n----- 优化重检预算已用完（机械约束）-----\n"
    f"本次请求已优化重检 {MAX_RERANK} 次（一次请求最多 {MAX_RERANK} 次）："
    "本回合不许再提议 rerank；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                                
_SHOW_BUDGET_BLOCK_ZH = (
    "\n----- 上屏预算已用完（机械约束）-----\n"
    f"本次请求已上屏 {MAX_SHOW} 次（一次请求最多 {MAX_SHOW} 次）："
    "本回合不许再提议 results.show；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                         
_ASK_RELAX_BUDGET_BLOCK_ZH = (
    "\n----- 放宽询问预算已用完（机械约束）-----\n"
    f"本次请求已询问放宽条件 {MAX_ASK_RELAX} 次（一次请求最多 {MAX_ASK_RELAX} 次）："
    "本回合不许再提议 ask.relax；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                          
                                             
                                             
                                        
def _ask_answer_resume_block_zh(ask_answer: Any) -> str:
    if not isinstance(ask_answer, dict):
        return ""
    label = str(ask_answer.get("label") or "").strip()
    message = str(ask_answer.get("message") or "").strip()
    chosen = label or message or "（选项文案缺失）"
    return (
        "\n----- 用户已在放松卡上作答（机械事实）-----\n"
        f"用户选择了「{chosen}」：第 1 步已按此选择忽略对应条件重跑了原查询"
        "（结果见上方已完成步骤）。接下来由你裁决：重跑结果健康且已自动上屏 → finish "
        "并如实说明忽略了哪些条件；仍不健康 → 可考虑 rerank 改写、（预算内）再发起 "
        "ask.relax，或如实 finish 说明现状。"
    )


def _screen_alert_block_zh(steps: list[dict]) -> str:
    """「结果不健康」告警注入段（2026-09-13 rerank 处置批——**纯情报员**：只注入上下文，
    不改 workflow 走向）：环内最近一次检索试跑（rank/rerank）被机械健康度谓词判为
    degenerate（零命中/解析弃权/向量兜底/条件丢失）时成段，把事实与可选对策摆给模型
    （rerank 是对策之一），选哪个、动不动都归模型裁决。最近试跑非 degenerate 或还没有
    试跑时恒空段。"""
    for s in reversed(steps):
        if not (isinstance(s, dict) and s.get("ok")):
            continue
        v = str(s.get("verb") or "")
        if v not in ("rank", "rerank"):
            continue
        r = s.get("result") if isinstance(s.get("result"), dict) else {}
        if str(r.get("health") or "") != "degenerate":
            return ""
        status = str(r.get("resolution_status") or "")
        total = int(r.get("total") or 0)
        if r.get("structured_context_lost"):
            detail = "新查询没能完整保留当前筛选条件（为避免放宽条件，未上屏）"
        elif status == "abstained":
            detail = "解析层弃权，检索未真正执行"
        elif status == "vector_fallback":
            detail = "条件未能执行，走的是向量语义兜底（未经条件核验）"
        elif status == "clarification_required":
            detail = "查询有歧义，需要先澄清"
        elif total <= 0:
            detail = "命中 0 条"
        else:
            detail = f"命中 {total} 条但未经条件核验"
        q = str(r.get("query") or r.get("rewritten_query") or "")
                                                               
                                             
                                                      
                                       
        relax_live = r.get("relax_live")
        relax_note = ""
        ask_relax_hint = ("ask.relax（病根在用户条件互相打架或过严时，"
                          "弹卡问用户放松哪个）/ ")
        if isinstance(relax_live, list):
            if relax_live:
                opts = "；".join(
                    f"{o.get('key')}（放松「{o.get('label')}」）→ {o.get('count')} 条"
                    for o in relax_live[:6])
                relax_note = (
                    f"确定性预演（零 LLM）：{opts}——放松用户条件能救回结果，"
                    "病根可能在**用户条件过严/互相打架**；这种情况 ask.relax 是对因对策"
                    "（把放松选项做成选择卡弹给用户，点选后自动重搜）。\n")
            else:
                relax_note = ("确定性预演（零 LLM）：放松任何单一条件都救不回结果——"
                              "ask.relax 不适用（选了也会被机械拒绝）。\n")
                ask_relax_hint = ""
        return (
            "\n----- 结果不健康告警（机械事实，仅供决策）-----\n"
            f"最近一次检索「{q}」的结果**不健康**：{detail}——屏面上没有经条件核验的结果。\n"
            + relax_note
            + "可选对策：rerank（优化检索词重查，是常用对策之一）/ "
            + ask_relax_hint
            + "search.rerun（换词重检）/ rank（换个方向新检索）/ "
              "如实告知用户后 finish——选哪个由你按原因判断。"
        )
    return ""

                                            
_COMPARE_BUDGET_BLOCK_ZH = (
    "\n----- 数据集对比预算已用完（机械约束）-----\n"
    f"本次请求已对比 {MAX_COMPARE} 次（一次请求最多 {MAX_COMPARE} 次）："
    "本回合不许再提议 compare.datasets；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)
_CITE_EXPORT_BUDGET_BLOCK_ZH = (
    "\n----- 引文导出预算已用完（机械约束）-----\n"
    f"本次请求已导出引文 {MAX_CITE_EXPORT} 次（一次请求最多 {MAX_CITE_EXPORT} 次）："
    "本回合不许再提议 cite.export；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)
_COMPAT_BUDGET_BLOCK_ZH = (
    "\n----- 兼容查找预算已用完（机械约束）-----\n"
    f"本次请求已查找兼容数据集 {MAX_COMPAT} 次（一次请求最多 {MAX_COMPAT} 次）："
    "本回合不许再提议 compat.find；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)
_FAIR_BUDGET_BLOCK_ZH = (
    "\n----- FAIR 自检预算已用完（机械约束）-----\n"
    f"本次请求已 FAIR 自检 {MAX_FAIR} 次（一次请求最多 {MAX_FAIR} 次）："
    "本回合不许再提议 fair.check；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                      
                                                  
                                      
_ROUTE_REQUEST_BUDGET_BLOCK_ZH = (
    "\n----- 换路线机会已用完（机械约束）-----\n"
    "本次请求已换过一次处理路线（每轮最多 1 次）：本回合不许再提议 route.request，"
    "按当前路线的工具继续；没有可做的事就 finish（JSON 通道回 {\"done\": true}）。"
)

                                                              
                                                       
                                   
def _rescue_block_zh(unresolved_terms: Any) -> str:
    """rescue 注入段（动态：原检索投影的未收录词逐字进提示）。

    放宽依据（取证，设计决定）：5 次真触发全被择优闸「改空拒」挡下——
    「语义等价」要求下模型只敢近乎原样改写（4 次逐字相同、1 次加空格），而能救活的
    改写（「小鼠神经胶质瘤」→「小鼠胶质瘤」）都必须丢弃**库中未收录**的用户词。
    放宽只开这一道口：未收录词可丢弃/映射为收录近义词；**已收录条件必须全部保留**、
    不许新增——丢弃了什么由择优闸机械比对（dropped_terms）如实披露，不靠模型自报。"""
    terms = [str(t).strip() for t in (unresolved_terms or []) if str(t).strip()]
    if terms:
        shown = "、".join(f"「{t}」" for t in terms[:8])
        relax = (
            f"规则匹配已确认：{shown}在库里**没有收录**——正是它们让这句话搜不到。"
            "改写时可以**丢弃这些词**，或把它们**映射为库里收录的近义词**"
            "（如「神经胶质瘤」→「胶质瘤」）；丢了哪些词，系统会机械比对后如实告知用户。"
            "除此之外：查询里**已收录的条件必须全部保留**，也不许新增用户没表达的条件。"
        )
    else:
        relax = (
            "规则匹配确认：这句话的条件**全部已收录**（没有可丢弃的未收录词），"
            "搜不到是条件组合本身没有数据。改写只允许**等价换说法**——已收录条件"
            "一个不能少、不能改，也不许新增；没有真正的等价改写就选 none 如实放弃。"
        )
    return (
        "\n----- 本回合限制（机械约束）-----\n"
        "这是一次「检索救回」回合：用户当前的查询没有搜到结果。本回合只允许两个选择——\n"
        "① 选 search.rerun，query 槽填改写后的检索句；\n"
        "② 选 none——没有可改的查询就如实放弃。\n"
        + relax + "\n"
        "其余任何动作本回合都不允许（系统会机械拒绝）。"
    )

                                                                        
_RESCUE_DECIDE_BLOCK_ZH = (
    "\n----- 本回合限制（机械约束）-----\n"
    "这是「检索救回」回合：只允许 search.rerun（换词重检）或 finish 收尾——"
    "其余工具的提议会被机械拒绝。"
)


def _write_steps_used(steps: list[dict]) -> int:
    """已用**正向写**步数（从 steps 现算，不加新状态）：写工具步不论成败都计——
    每一步都可能已产生副作用（sync 中途失败前可能已写过部分来源的批次文件）。
    2026-08-15 豁免：`search_online` 的 **network_error**
    失败可证零副作用（异常在 plan 取数阶段抛出，apply 入库一行都没跑，见
    `_loop_search_online` 的 plan→apply 顺序）——不计入写步预算，与指纹去重的
    network_error 豁免同一「唯一真·可重试码」哲学。sync 网络失败按契约如实降级
    不抛、不产生 network_error 步，无需区分；除 network_error 外的失败码照旧计入
    （bad_result_shape 等无法自证零写入）。"""
    return sum(1 for s in steps if str(s.get("verb") or "") in _WRITE_LOOP_TOOLS
               and not (str(s.get("verb") or "") == "curate.search_online"
                        and str(s.get("error_code") or "") == "network_error"))


def _written_files_of_result(verb: str, result: Any) -> list[str] | None:
    """本步工具**自报**的落盘产物名清单（快照归属过滤用）——单一映射锚点。

    - 联网入库：`filename`（单文件；零写入 → 空清单，明确「本步没有产物」）；
    - 同步入库：`created_files` 回执（缺省回退逐源 `filename`）；
    - 其余动词 → None：不设清单（快照保持全目录差分旧行为，不假装知道产物）。"""
    if not isinstance(result, dict):
        return None
    if verb == "curate.search_online":
        fn = str(result.get("filename") or "").strip()
        return [fn] if fn else []
    if verb == "curate.sync_updates":
        names: list[str] = []
        for item in list(result.get("created_files") or []):
            s = str(item or "").strip()
            if s and s not in names:
                names.append(s)
        if names:
            return names
        for ent in list(result.get("sources") or []):
            fn = str((ent or {}).get("filename") or "").strip()
            if fn and fn not in names:
                names.append(fn)
        return names
    return None


def _mutation_receipt_of_result(verb: str, result: Any) -> dict[str, list[str]] | None:
    """本步工具**自报的变更收据**（归属纪律）。

    - `created` 复用 `_written_files_of_result` 的产物映射；
    - `modified`/`deleted` 对两个写漏斗恒为空——它们都是「新建一个批次文件」的语义，
      不会改写/删除既有文件；配成显式空清单后，快照不再把 capture—finalize 窗口内
      **他人的改写/删除**归给本步（全目录差分只作留痕，不进回滚执行清单）；
    - 认不出的动词 → None（不设收据；快照保持旧全目录差分行为）。"""
    created = _written_files_of_result(verb, result)
    if created is None:
        return None
    return {"created": list(created), "modified": [], "deleted": []}


def _empty_mutation_receipt() -> dict[str, list[str]]:
    """失败步的收据 = **全部为空**：工具没跑完、产物清单不可知时，任何窗口内变化都
    不归给本步（宁可少退不毁数据）；未归因项由快照如实留痕（unattributed_*）。"""
    return {"created": [], "modified": [], "deleted": []}


def _write_records_used(steps: list[dict]) -> int:
    """已写入条数（**从 steps 真实结果现算**）：search 取 record_count
    sync 取 imported_total；无结果（失败/异常步）计 0——能查到账的才计入，查不到
    如实少报（副作用账在回收站/流水账里另有真源，这里只是预算近似）。"""
    total = 0
    for s in steps:
        if str(s.get("verb") or "") not in _WRITE_LOOP_TOOLS:
            continue
        result = s.get("result")
        if not isinstance(result, dict):
            continue
        for key in ("record_count", "imported_total"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                total += value
                break
    return total


def _failed_tool_ban(steps: list[dict]) -> frozenset[str]:
    """非网络二连败的禁提面（**从 steps 现算**）：最近两步均失败、且非联网暂停形态
    （不全是 network_error）→ 禁提这两步各自的 verb（指纹到动作）。
    旧设计「任意二连败硬停环」会把链上剩余独立事项（如 db_status）一起误杀
    联网二连败不在此列——网络在抖时整族都该停，走
    `_network_moratorium` 的整族禁提。"""
    if len(steps) < _CONSECUTIVE_FAILURE_BREAKER:
        return frozenset()
    tail = steps[-_CONSECUTIVE_FAILURE_BREAKER:]
    if not all(not s.get("ok") for s in tail):
        return frozenset()
    if all(str(s.get("error_code") or "") == "network_error" for s in tail):
        return frozenset()                          
    return frozenset(str(s.get("verb") or "") for s in tail if s.get("verb"))


def _failed_tool_ban_block_zh(banned: frozenset[str]) -> str:
    """失败动作禁提的 prompt 机械注入段（与联网暂停段同型，禁提面是具体动作）。
    文案如实：两个**不同**动作各败一次不许说成「连续失败两次」——
    那是同一动作连败两次的专属表述。"""
    names = sorted(
        (_ap.VERB_BY_NAME.get(v).zh if _ap.VERB_BY_NAME.get(v) else v) for v in banned)
    if len(names) == 1:
        lead = f"{names[0]} 刚刚已连续失败两次"
    else:
        lead = "最近两步（" + "、".join(names) + "）均失败"
    return (
        "\n----- 失败工具禁提（机械约束）-----\n"
        f"{lead}：本回合不许再提议它们；其余工具仍可用；"
        "没有可做的事就 finish（JSON 通道回 {\"done\": true}）。")


def _is_network_call(verb: str, source: Any) -> bool:
    """该次调用是否触网：search/sync 恒触网；check_updates 按
    解析源判定——离线快照源（CHECK_UPDATE_SOURCES 里 online=False 的 CELLxGENE /
    EBI SCEA / HuBMAP / Single Cell Portal）只读本地快照，联网暂停不得连坐。
    source 缺省/认不出 = 查全部（含在线源）= 按触网处理（保守同旧口径）。"""
    if verb != "curate.check_updates":
        return True
    from ..corpus import corpus_curation as _cc
    from ..corpus import corpus_net as _cn
    key = _cn.resolve_source_key(source, valid_keys=_cc.CHECK_UPDATE_SOURCES)
    if key is None:
        return True
    return bool(_cc.CHECK_UPDATE_SOURCES.get(key, {}).get("online", True))


def _network_moratorium(steps: list[dict]) -> bool:
    """联网暂停判定（**从 steps 现算，不加新状态**）：最近两步均以 network_error 失败。
    联网二连败是网络在抖、不是「没路可走」——离线工具本可做；非网络码的二连败
    （形状闸等）不在此列，维持原硬停。"""
    if len(steps) < _CONSECUTIVE_FAILURE_BREAKER:
        return False
    return all(
        not s.get("ok") and str(s.get("error_code") or "") == "network_error"
        for s in steps[-_CONSECUTIVE_FAILURE_BREAKER:])


def _agent_project_root() -> Path:
    """agent 图内工具的项目根：**复用 corpus_status 的默认根真源**（起 = runtime_paths
    实例数据根：source/portable = 项目根；frozen = data_root）——不发明第二份解析逻辑。
    数据写盘侧（账本/引文/管护）以它为基；随包静态资源（prompts/官方快照）另行经
    resource_file_for 走 resource 层。"""
    from ..corpus.corpus_status import _default_project_root

    return _default_project_root()


def _loop_db_status(slots: dict, root: Path) -> dict:
    from ..corpus.corpus_status import db_status

    return db_status(project_root=root)


def _loop_check_updates(slots: dict, root: Path) -> dict:
    from ..corpus import corpus_curation as cc

    source = str((slots or {}).get("source") or "").strip()
    return cc.check_updates([source] if source else None, project_root=root)


def _loop_search_online(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """联网搜 + 立刻入库（plan → apply 图内闭环）：confirm_token 由 plan 结果内部回传——
    这就是全自动化授权的机械确认（后端重算指纹 fail-closed）。keywords 缺失 → CurateError
    如实失败；未注册源（如 10x）在**任何网络请求之前**抛 source_not_registered（fail-fast）。
    返回 plan/apply 两侧字段的合并 dict（前端 actSearchCardHtml 吃同一份）。

    `ctx.write_records_remaining` = 执行前现算的本轮剩余写入额度——联网候选数按
    min(默认 20, 剩余) 收窄，本轮实际写入因此不会越过 40 条上限（旧实现只在写完后扣账，
    30 + 20 = 50 条）。额度缺席（直调/旧 ctx）→ 默认 20，行为逐位不变。"""
    from ..corpus import corpus_curation as cc

    slots = slots or {}
    keywords = str(slots.get("keywords") or "").strip()
    if not keywords:
        raise cc.CurateError(
            "bad_param",
            "这一步没有拿到能当联网搜索关键词的内容，没有发起搜索，也没有写入任何内容。"
            "修正方法：从用户原话提取主题词（优先英文，如「人类肺」→ human lung）"
            "作为 keywords 重新提议这一步。",
        )
    source = str(slots.get("source") or "").strip().lower() or "arrayexpress"
    species = str(slots.get("species") or "").strip() or None
    limit = 20
    remaining = (ctx or {}).get("write_records_remaining")
    capped = False
    remaining_n = 20
    if remaining is not None:
        try:
            remaining_n = max(0, int(remaining))
        except (TypeError, ValueError):
            remaining_n = 20
        if remaining_n <= 0:
            raise cc.CurateError(
                "too_large",
                "本次请求的写入额度已用完（一次请求最多 40 条），没有执行联网入库，"
                "也没有写入任何内容；还要入库可以再说一次。")
        if remaining_n < limit:
            limit = remaining_n
            capped = True
    pr = cc.plan_search_online(keywords, source, species, limit=limit, project_root=root)
    applied = cc.apply_search_online(pr, confirm_token=pr["confirm_token"], project_root=root)
    warnings = list(pr.get("warnings") or []) + list(applied.get("warnings") or [])
    if capped:
        warnings.append(
            f"本次请求的写入额度只剩 {remaining_n} 条，联网候选按 {limit} 条上限执行"
            "（其余候选可以再说一次「联网搜…」继续）。")
    return {
        "source_label": pr.get("source_label"),
        "query": pr.get("query"),
        "species": pr.get("species"),
        "sample_titles": pr.get("sample_titles"),
        "record_count": applied.get("record_count"),
        "filename": applied.get("filename"),
                                              
        "skipped_existing": applied.get("skipped_existing") or pr.get("skipped_existing") or [],
        "skipped_existing_count": applied.get("skipped_existing_count")
            if applied.get("skipped_existing_count") is not None
            else pr.get("skipped_existing_count") or 0,
        "warnings": warnings,
    }


def _loop_sync_updates(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """检查更新 → 有新增则自动入库的复合流（corpus_curation.sync_updates 一次原子调用）。

    按 `ctx.write_records_remaining` 在**写前**同时收窄单源（max_import）与
    跨来源总预算（total_max_import）——先 search 后 sync 的两步写链合计不越过 40 条；
    被预算挡下的新增由语料侧 note_zh 如实点名（不静默截断）。"""
    from ..corpus import corpus_curation as cc

    source = str((slots or {}).get("source") or "").strip()
    kwargs: dict = {}
    remaining = (ctx or {}).get("write_records_remaining")
    if remaining is not None:
        try:
            remaining_n = max(1, int(remaining))
        except (TypeError, ValueError):
            remaining_n = None
        if remaining_n is not None:
            kwargs["max_import"] = min(10, remaining_n)
            kwargs["total_max_import"] = remaining_n
    return cc.sync_updates([source] if source else None, project_root=root, **kwargs)


class _SearchRerunParamError(Exception):
    """search.rerun 的 query 槽为空。execute 的错误提取读 hint/code 属性
    （与 corpus_curation.CurateError 同约：str 是「code: hint」，hint 才是人读）。"""

    code = "bad_param"
    hint = ("这一步没有拿到改写后的检索句（query 槽为空），没有发起重检。"
            "修正方法：query 槽填改写后的检索句重新提议这一步。")

    def __init__(self) -> None:
        super().__init__(f"{self.code}: {self.hint}")


def _rescue_disclosure_zh(rewrite: str, n_after: int | None, dropped_terms: list) -> str:
    """rescue 采纳的**确定性披露句**（2026-08-17 rescue2）：丢弃词/改写词/命中数全部
    机械实算（dropped_terms 是「改写句里消失的未收录词」子串比对），不采信 LLM 自报。
    端点把它当 report_zh 下发——前端 `handleSearchRescue` 优先显示 report_zh，零改动上屏。"""
    n = "若干" if n_after is None else str(int(n_after))
    if dropped_terms:
        shown = "、".join(f"「{t}」" for t in dropped_terms[:8])
        return f"{shown}在库里没有收录，已按「{rewrite}」重查，找到 {n} 条，结果区已更新。"
    return f"已按「{rewrite}」重查，找到 {n} 条，结果区已更新。"


def _loop_search_rerun(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """换一组查询词把本地库重新检索一遍（2026-08-16 检索工具化 Phase 1，机械择优闸）。

    跑的是与主检索**同一条管线**（`workflow._prepare_context` / `run_with_meta`，规则检索，
    查询改写只在本工具环发生，不存在 workflow 审核→重搜的第二通道。采纳与否由机械闸裁定：
      ① 改写解析出的硬过滤与基准同集（`_same_hard_filter`，存活集不变）→ 拒
         （rewrite_no_change_kept_original）；
      ② 载荷弄丢已生效的结构化条件（分面/抑制/宽容/日期）→ 拒
         （structured_context_lost_kept_original，防误换屏安全闸，见下）；
      ③ 否则采纳——**命中 0 条同样采纳**（设计决定：条件变更重检的空结果就是
         诚实答案，空结果集照常上屏，绝不「结果不如当前就否决、保持不变」）——载荷 =
         /api/recommend 同形 dict（`app.recommend_rows.recommend_payload`，采纳档第三次跑
         run_with_meta 取全量投影；两次规则检索，本地便宜）。
    基准 = 现场查询（ctx.current_query；rescue 端点恒有，链内空现场时无基准）：有基准时
    对基准查询先跑一次同口径 prep 取 intent/candidates——n_before 实算，不采信任何自述；
    无基准 → n_before 如实记 None、同集闸无对象可比而跳过（改写即采纳，含零命中）。
    query 槽空 → `_SearchRerunParamError`（bad_param，与 _loop_search_online 同纪律）。

    rescue2：ctx.unresolved_terms（原检索投影的未收录词，rescue 端点链下入；
    缺省恒 []——链内调用与旧测试逐位不变）。采纳时机械比对 `dropped_terms` = 改写句里
    **消失**的未收录词（子串比对，双向都逐字来自真实串），附 `disclosure_zh` 确定性
    披露句；未采纳两档 dropped_terms 恒 []。

    nl-A（2026-08-17 挂账，择优闸与屏口径一致）：n_before/n_after 保持**择优闸口径**
    （prep 候选数，top-k 截断——步骤卡注释明示、机械三态的裁决语境）；additive 的
    n_before_total/n_after_total 是**屏口径**（未截断命中总数）——采纳档 n_after_total
    与屏单源（= payload.result_total），n_before_total 为基准查询同管线硬过滤存活数
    （matched_survivors 一次扫描，语料已在内存、零额外装载）；采纳档 payload.audit 的
    n_before/n_after 也用屏口径——那对键喂的是整屏替换后的 audit 横幅（与结果区
    「库中共 N 条匹配」、披露句同屏，截断口径上去就是同屏打架：实测 10 → 10 条 vs
    屏上 93 条）。无基准时 n_before_total 与 n_before 同如实记 None。"""
    from ..app import workflow as wf

    slots = slots or {}
    ctx = ctx or {}
    query = str(slots.get("query") or "").strip()
    if not query:
        raise _SearchRerunParamError()
    base_query = str(ctx.get("current_query") or "").strip()
    sources = ctx.get("search_sources")
    replace_screen = bool(ctx.get("replace_screen"))
    unresolved = [str(t).strip() for t in (ctx.get("unresolved_terms") or []) if str(t).strip()]

                                                            
                                                         
                                                                
                                               
    structured_kwargs = _loop_structured_kwargs(ctx)
    facet_filters = structured_kwargs["facet_filters"]

    flow = wf.DatasetRecommendationWorkflow()
    n_before: int | None = None
    n_before_total: int | None = None
    intent_base = None
    if base_query:
        bctx = flow._prepare_context(
            query=base_query, sources=sources, auto_parse_sources=False,
            **structured_kwargs)
        intent_base, candidates_base = bctx[0], bctx[1]
        n_before = len(candidates_base)
        n_before_total = len(
            flow.retriever.matched_survivors(bctx[2], intent_base, facet_filters))
        if sources is None:
                                               
                                                    
            sources = bctx[6].sources

    rctx = flow._prepare_context(
        query=query, sources=sources, auto_parse_sources=False,
        **structured_kwargs)
    intent2, candidates2 = rctx[0], rctx[1]
    n_after = len(candidates2)
                                                
             
    n_after_total = len(flow.retriever.matched_survivors(rctx[2], intent2, facet_filters))

    def _rejected(reason: str, disclosure_zh: str | None = None) -> dict:
        result = {
            "adopted": False, "reason": reason, "query": query,
            "n_before": n_before, "n_after": n_after,
            "n_before_total": n_before_total, "n_after_total": n_after_total,
            "replace_screen": replace_screen, "payload": None,
            "dropped_terms": [],
        }
        if disclosure_zh:
            result["disclosure_zh"] = disclosure_zh
        return result

    if intent_base is not None and wf._same_hard_filter(intent_base, intent2):
        return _rejected("rewrite_no_change_kept_original")

    meta = flow.run_with_meta(
        wf.RecommendParams(
            query=query, use_llm=False, sources=sources,
            auto_parse_sources=False,
            **structured_kwargs))
                                                                       
                                                                      
                                                       
               
    payload, scope_kept = _verified_recommend_payload(meta, structured_kwargs)
    if not scope_kept:
        return _rejected(
            "structured_context_lost_kept_original",
            "这次重新检索没有执行：新查询没能完整保留你当前生效的筛选条件，"
            "为避免条件被放宽、结果失真，当前结果区未改动。")
                                                
                                              
    n_after_total = int(payload.get("result_total") or n_after_total)
                                                                      
                                                                      
                                                                
    payload["audit"] = {
        "triggered": True, "verdict": False, "rewritten_query": query, "used": True,
        "reason": "rewritten", "mode": "rerank",
        "n_before": n_before_total, "n_after": n_after_total,
        "was_no_result": n_before_total == 0,
    }
                                                       
                                                               
                                                     
                                
    dropped = [t for t in unresolved if t not in query]
    return {
        "adopted": True, "reason": "adopted", "query": query,
        "n_before": n_before, "n_after": n_after,
        "n_before_total": n_before_total, "n_after_total": n_after_total,
        "replace_screen": replace_screen, "payload": payload,
        "dropped_terms": dropped,
        "disclosure_zh": _rescue_disclosure_zh(query, payload.get("result_total"), dropped),
    }


                                                                                

                                                             
                                                        
_QUERY_REWRITE_PROMPT_FALLBACK_ZH = (
    "你是生物数据检索 query 优化器。把用户给的检索句改写成更适合数据集目录检索的形式："
    "物种/组织/疾病/平台等实体规范化，去口语化操作词，中文主题词补齐英文对照。"
    "只输出优化后的检索句一行，不要任何解释；改不出更好的就原样输出原句。"
)


def _prompt_resource_text(name: str) -> str:
    """提示词文本解析（**领域包优先**、骨架 `prompts/` 兜底）。

    领域包 `prompts_dir` 两种形态：绝对路径（外部领域包直读）或 resource 相对路径
    （与骨架 prompts/ 同一打包通道：source=项目根、frozen=_MEIPASS）。
    biodata 包指向 `prompts/domains/biodata/`，命中即读；其余名称回落骨架目录。
    """
    pack_dir = (_get_domain().prompts_dir or "").strip()
    if pack_dir:
        p = Path(pack_dir)
        cand = (p / name) if p.is_absolute() else resource_file_for(
            _agent_project_root(), f"{p.as_posix()}/{name}")
        if cand.is_file():
            return cand.read_text(encoding="utf-8").strip()
    return resource_file_for(_agent_project_root(), f"prompts/{name}").read_text(encoding="utf-8").strip()


def _query_rewrite_prompt() -> str:
    """rerank 改写器的系统提示词（领域包 `query_rewrite.md`，文件即真源）——
    直接委托 `_prompt_md`（同一份缓存/降级/告警机制，不养第二份实现）。"""
    return _prompt_md("query_rewrite.md", _QUERY_REWRITE_PROMPT_FALLBACK_ZH)


                                              
                                               
_LOOP_CORE_FALLBACK_ZH = (
    "你在执行用户的一句话指令，可能要做连续多件事。诚实不变量（违反任一条都是错误）："
    "只据已完成步骤的真实结果判断，不许编造；verb 只能从工具表选；quoted 必须逐字出自原话；"
    "独立的事逐件完成；机械预算段是硬上限。收尾必须附 completion_report 逐件核销。"
)
_LOOP_DELTA_FALLBACK_ZH = "按工具表内各工具的专职描述逐步推进；路线明显不对就 route_request。"
_ROUTE_CONSENSUS_FALLBACK_ZH = (
    "你是分流器，只判断这句话走哪条路线：search=找数据/改检索条件/贴编号直链；"
    "action=要动作（下载/联网搜库/检查更新/入库/管护）；general=闲聊/概念问答/混合/拿不准。"
    "只输出一个 JSON 对象：{\"route\": \"search|action|general\", \"reason\": \"一句中文理由\"}。"
)
_ROUTE_WRITE_GATE_FALLBACK_ZH = (
    "你是网关，只判断处理这句话是否需要用到会产生本地下载或数据写入后果的工具："
    "write=打包下载/导出引文/投稿材料/入库/删除/恢复/同步进外部库；"
    "read=找数据/改条件/查看重排/对比/兼容/FAIR/查库容/检查更新(只看不同步)/"
    "联网搜库(只搜不入库)/概念问答/闲聊；拿不准判 read（只读是安全默认）。"
    "只输出一个 JSON 对象：{\"gate\": \"write|read\", \"reason\": \"一句中文理由\"}。"
)

                                                               
                                                                
                                                                       
                                            
                                                     
                                     
_ROUTE_MODE_ENV = "BIODATA_ROUTE_MODE"
_ROUTE_MODE_DEFAULT = "binary"
                                                  
                                                
                                                       
                                            
_ROUTE_L0_ENV = "BIODATA_ROUTE_L0"
_ROUTE_L0_TAU_ENV = "BIODATA_ROUTE_L0_TAU"


def _route_mode() -> str:
    raw = os.environ.get(_ROUTE_MODE_ENV, "").strip().lower()
    if not raw:
        return _ROUTE_MODE_DEFAULT
    if raw in ("three_class", "binary"):
        return raw
    _warn_once("route_mode", f"{_ROUTE_MODE_ENV}={raw!r} 非法，回默认 {_ROUTE_MODE_DEFAULT}。")
    return _ROUTE_MODE_DEFAULT


def _env_default_on(name: str) -> bool:
    """默认开开关：未设/空白 → True；truthy → True；其余显式值（0/off/false/no/…）→ False。"""
    raw = os.environ.get(name, "").strip().lower()
    return (not raw) or raw in {"1", "true", "yes", "on"}
_PROMPT_MD_CACHE: dict[tuple[str, str], str] = {}


def _prompt_md(name: str, fallback_zh: str) -> str:
    """`prompts/<name>`（领域包目录优先，见 `_prompt_resource_text`）文件即真源的提示词加载
    （懒加载缓存，按 (domain_id, name) 分键——换域不串内容）；缺失/读取失败退回内置最小版。"""
    key = (_get_domain().domain_id, name)
    if key not in _PROMPT_MD_CACHE:
        try:
                                              
            _PROMPT_MD_CACHE[key] = _prompt_resource_text(name)
        except Exception:
            _warn_once(f"prompt_md::{name}", f"prompts/{name} 读取失败，用内置最小提示词。")
            _PROMPT_MD_CACHE[key] = fallback_zh
    return _PROMPT_MD_CACHE[key]


def _route_consensus_prompt() -> str:
    """分流共识的系统提示词（`prompts/route_consensus.md`，文件即真源）。"""
    return _prompt_md("route_consensus.md", _ROUTE_CONSENSUS_FALLBACK_ZH)


def _route_write_gate_prompt() -> str:
    """二元写闸的系统提示词（`prompts/route_write_gate.md`，文件即真源）——
    判定定义与 research/route_eval/label_binary_gate.py 的打标口径逐字一致
    （该脚本内置漂移校验，prompt 改了打标脚本会报错要求重打标）。"""
    return _prompt_md("route_write_gate.md", _ROUTE_WRITE_GATE_FALLBACK_ZH)


def _rewrite_query(model: Any, query: str,
                   usage_sink: list | None = None,
                   probe_zh: str = "", health_zh: str = "",
                   extra_system_zh: str = "", anchor_zh: str = "",
                   repair_zh: str = "") -> tuple[str, bool]:
    """rerank 的独立 LLM 改写调用（独立上下文 + 独立系统提示词）+ **机械健全性检查**：
    非空、≤200 字符、与原句不同——任一不过（含模型缺席/调用异常）退回原 query 并
    rewritten=False 如实标注，不靠 LLM 自评。`usage_sink`：
    给了就把本次改写调用的缓存用量经 `_usage_record` 追加进去（读不到用量自然跳过）。
    `probe_zh`（2026-09-13 夜批）：约束松弛反事实探测报告，非空时拼在 query 后一并
    注入（**仅建议**——模型可照建议走 relax 通道、可自由改写、也可弃权）；缺省空串
    与历史行为逐位一致。
    `health_zh`（2026-09-16 改写环臂批）：机械健康快照（`_rewrite_health_zh`），非空时
    按 query → 健康快照 → 探测报告顺序拼进 human；缺省空串逐位不变。
    `extra_system_zh`（同批，研究臂用）：追加在系统提示词之后（协议包装/few-shot），
    缺省空串逐位不变。
    `anchor_zh`（2026-09-17 上下文工程批，借鉴 Aivis 经验八「重点信息多次增强」）：
    环内 decide 传给 rerank 的 query 可能已被模型转述——与原话有出入时把用户原话
    作为锚点块注入（紧跟 query 之后），改写语义以原话为准；缺省空串逐位不变。
    `repair_zh`（2026-09-18 专家化批 WP3）：repair 路由未采纳时的候选菜单
    （`_rewrite_repair_zh`），非空时按 query → 锚点 → 健康快照 → 候选菜单 →
    探测报告顺序拼进 human；缺省空串逐位不变。"""
    if model is None:
        return query, False
    t0 = time.monotonic()                              
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        parts = [query]
        if anchor_zh:
            parts.append(anchor_zh)
        if health_zh:
            parts.append(health_zh)
        if repair_zh:
            parts.append(repair_zh)
        if probe_zh:
            parts.append(probe_zh)
        human = "\n\n".join(parts)
        system = _query_rewrite_prompt() + (
            ("\n\n" + extra_system_zh) if extra_system_zh else "")
        answer = model.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        rec = _usage_record(answer, "rerank_rewrite")
        if rec is not None and usage_sink is not None:
            usage_sink.append(rec)
        content = getattr(answer, "content", "")
        if isinstance(content, list):
                                        
            content = " ".join(
                str(p.get("text") or "") for p in content if isinstance(p, dict))
        candidate = str(content or "").strip()
        _te.emit_llm_call(
            node="rerank_rewrite",
            model=str(getattr(model, "model_name", "") or getattr(model, "model", "") or ""),
            prompt=query, response=str(content or ""),
            ms=int((time.monotonic() - t0) * 1000), channel="text")
    except Exception as exc:
        _te.emit_llm_call(
            node="rerank_rewrite",
            model=str(getattr(model, "model_name", "") or getattr(model, "model", "") or ""),
            prompt=query, response="",
            ms=int((time.monotonic() - t0) * 1000), channel="text",
            fallback_reason=type(exc).__name__)
        return query, False
                                            
    candidate = candidate.splitlines()[0].strip().strip("\"'") if candidate else ""
    if not candidate or len(candidate) > 200 or candidate == query:
        return query, False
    return candidate, True


def _rewrite_health_zh(*, health: str, total: int, resolution_status: str = "",
                       filters: Any = (), top_titles: Any = (),
                       label: str = "当前结果", lite: bool = False) -> str:
    """rerank 改写/试算反馈的**机械健康快照**（零 LLM，2026-09-16 改写环臂批）：把
    当前屏面或一次试跑的健康事实压成中文紧凑块注入改写上下文——健康度三态、解析状态、
    结果数、执行的条件、top 示例。全是机械事实，不评内容质量（与 `_health_of_meta`
    同纪律）；某一面不可得就如实略过（结果数是底线事实，恒在），不编造。
    `lite=True`（2026-09-21 注入稀释消融批）：只留健康度+结果数两行最小屏面信号，
    略去解析状态/执行的条件/top 示例——供「注入量是否稀释关键信号」的消融实验与
    （若验证成立后的）生产最小注入使用；默认 False 行为逐位不变。"""
    lines: list[str] = []
    health_zh = _HEALTH_ZH.get(str(health or ""), "")
    if health_zh:
        lines.append(f"- 健康度：{health_zh}")
    status = str(resolution_status or "").strip()
    if status and not lite:
        lines.append(f"- 解析状态：{status}")
    lines.append(f"- 结果数：{int(total or 0)}")
    if lite:
        return f"参考：{label}的健康快照（机械事实，仅供判断）：\n" + "\n".join(lines)
    conds = "、".join(
        f"{'排除' if f.get('polarity') == 'exclude' else ''}{f.get('label', '')}="
        f"{'/'.join(str(v) for v in (f.get('values') or []))}"
        for f in list(filters or []) if isinstance(f, dict))
    if conds.strip("、"):
        lines.append(f"- 执行的条件：{conds.strip('、')}")
    titles = [str(t or "").strip() for t in list(top_titles or [])][:3]
    titles = [t for t in titles if t]
    if titles:
        lines.append(f"- top 示例：{'；'.join(titles)}")
    return f"参考：{label}的健康快照（机械事实，仅供判断）：\n" + "\n".join(lines)


                                                                                  

_DIFF_SCENE_KEYS = ("facet_filters", "suppressed_constraints", "lenient_dims",
                    "date_from", "date_to")


def _dim_set_diff(before: dict, after: dict) -> dict:
    """两个 {dim: [canonical values]} 的值级集合差：kept/added/removed（空类略去）。"""
    out: dict[str, dict] = {"kept": {}, "added": {}, "removed": {}}
    for dim in sorted(set(before or {}) | set(after or {})):
        b, a = set((before or {}).get(dim) or []), set((after or {}).get(dim) or [])
        for key, vals in (("kept", sorted(b & a)), ("added", sorted(a - b)),
                          ("removed", sorted(b - a))):
            if vals:
                out[key][dim] = vals
    return out


def _role_shift(b_map: dict, a_map: dict) -> dict:
    """同一 (dim, value) 在两视图间的跨界：b_map→a_map 方向的角色迁移清单。"""
    out: dict[str, list[str]] = {}
    for dim in sorted(set(b_map or {}) & set(a_map or {})):
        moved = sorted(set((b_map or {}).get(dim) or []) & set((a_map or {}).get(dim) or []))
        if moved:
            out[dim] = moved
    return out


def _constraint_diff(flow: Any, original: str, rewritten: str,
                     structured_kwargs: dict | None = None) -> dict:
    """改写前后约束语义的机械对比（零 LLM，2026-09-17 实验 J）：规则解析器在同一结构化
    现场下分别解析原句与改写句，逐项比对硬约束（含排除极性）/软偏好/原始数据要求/
    时间过滤/自由主题词/解析状态/未解析词。
    `would_block`/`block_reasons` 是**影子闸**（只记录不执行）：硬约束增删、排除变化、
    软硬角色转换、正负能量翻转、硬日期/原始数据硬要求变化 = 未授权语义变化。
    解析异常如实入 diff（parse_error + would_block=True），不抛出、不编造。"""
    scene = {k: v for k, v in dict(structured_kwargs or {}).items() if k in _DIFF_SCENE_KEYS}
    diff: dict[str, Any] = {"original": original, "rewritten": rewritten}
    intents: list[Any] = []
    for q in (original, rewritten):
        try:
            intent, *_ = flow._prepare_context(
                q, sources=None, auto_parse_sources=False, **scene)
            intents.append(intent)
        except Exception as exc:                                    
            intents.append(None)
            diff["parse_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if intents[0] is None or intents[1] is None:
        diff["would_block"] = True
        diff["block_reasons"] = ["parse_error"]
        return diff
    b, a = intents
    diff["status"] = {
        "before": str(getattr(b, "parse_status", "") or ""),
        "after": str(getattr(a, "parse_status", "") or ""),
        "before_reason": str(getattr(b, "abstain_reason", "") or
                             getattr(b, "clarification_reason", "") or ""),
        "after_reason": str(getattr(a, "abstain_reason", "") or
                            getattr(a, "clarification_reason", "") or ""),
    }
    hard = _dim_set_diff(getattr(b, "constraints", None), getattr(a, "constraints", None))
    excl = _dim_set_diff(getattr(b, "excluded_constraints", None),
                         getattr(a, "excluded_constraints", None))
    soft = _dim_set_diff(getattr(b, "preferred_constraints", None),
                         getattr(a, "preferred_constraints", None))
    diff["hard"], diff["excluded"], diff["soft"] = hard, excl, soft
    diff["hard_to_soft"] = _role_shift(getattr(b, "constraints", None),
                                       getattr(a, "preferred_constraints", None))
    diff["soft_to_hard"] = _role_shift(getattr(b, "preferred_constraints", None),
                                       getattr(a, "constraints", None))
    diff["polarity_flip"] = {
        "include_to_exclude": _role_shift(getattr(b, "constraints", None),
                                          getattr(a, "excluded_constraints", None)),
        "exclude_to_include": _role_shift(getattr(b, "excluded_constraints", None),
                                          getattr(a, "constraints", None)),
    }
    diff["raw_required"] = [getattr(b, "has_raw_data_required", None),
                            getattr(a, "has_raw_data_required", None)]
    diff["date"] = [[getattr(b, "date_from", ""), getattr(b, "date_to", "")],
                    [getattr(a, "date_from", ""), getattr(a, "date_to", "")]]
    diff["soft_raw"] = [getattr(b, "preferred_raw", None), getattr(a, "preferred_raw", None)]
    diff["soft_date"] = [[getattr(b, "preferred_date_from", ""), getattr(b, "preferred_date_to", "")],
                         [getattr(a, "preferred_date_from", ""), getattr(a, "preferred_date_to", "")]]
    diff["soft_sources"] = [list(getattr(b, "preferred_sources", None) or []),
                            list(getattr(a, "preferred_sources", None) or [])]
    fb = set(getattr(b, "free_text_terms", None) or [])
    fa = set(getattr(a, "free_text_terms", None) or [])
    diff["free_text"] = {"kept": sorted(fb & fa), "added": sorted(fa - fb),
                         "removed": sorted(fb - fa)}
    diff["unresolved"] = {"before": list(getattr(b, "unresolved_terms", None) or []),
                          "after": list(getattr(a, "unresolved_terms", None) or [])}
    reasons: list[str] = []
    if hard["added"]:
        reasons.append(f"硬约束新增:{hard['added']}")
    if hard["removed"]:
        reasons.append(f"硬约束丢失:{hard['removed']}")
    if excl["added"] or excl["removed"]:
        reasons.append(f"排除变化:加{excl['added']}减{excl['removed']}")
    if diff["hard_to_soft"]:
        reasons.append(f"硬转软:{diff['hard_to_soft']}")
    if diff["soft_to_hard"]:
        reasons.append(f"软转硬:{diff['soft_to_hard']}")
    if diff["polarity_flip"]["include_to_exclude"] or diff["polarity_flip"]["exclude_to_include"]:
        reasons.append(f"极性翻转:{diff['polarity_flip']}")
    if diff["raw_required"][0] != diff["raw_required"][1]:
        reasons.append(f"原始数据硬要求:{diff['raw_required'][0]}→{diff['raw_required'][1]}")
    if diff["date"][0] != diff["date"][1]:
        reasons.append(f"时间硬过滤:{diff['date'][0]}→{diff['date'][1]}")
    diff["block_reasons"] = reasons
    diff["would_block"] = bool(reasons)
    return diff


def _fmt_dim_map_zh(d: dict) -> str:
    labels = _domain_labels_zh()
    return "；".join(f"{labels.get(dim, dim)}＝{'/'.join(str(v) for v in vals)}"
                     for dim, vals in (d or {}).items())


def _constraint_diff_zh(diff: dict) -> str:
    """`_constraint_diff` 的人读版（注入改写循环上下文用）。机械事实，不评好坏、
    不给改写建议（与 `_rewrite_health_zh` 同纪律）；空类如实略过。"""
    if not diff:
        return ""
    _STATUS_ZH = {"executable": "可执行", "abstained": "弃权",
                  "clarification_required": "需澄清"}
    lines = ["参考：改写前后约束对比（机械事实，仅供判断）："]
    st = diff.get("status") or {}
    if st:
        def _st_zh(side: str, reason_key: str) -> str:
            zh = _STATUS_ZH.get(st.get(side), str(st.get(side) or "未知"))
            reason = str(st.get(reason_key) or "")
            return f"{zh}（{reason}）" if reason else zh
        lines.append(f"- 解析状态：原句 {_st_zh('before', 'before_reason')} → "
                     f"改写 {_st_zh('after', 'after_reason')}")
    for key, name in (("hard", "硬约束"), ("excluded", "排除"), ("soft", "软偏好")):
        seg = diff.get(key) or {}
        parts = []
        if seg.get("kept"):
            parts.append(f"保留[{_fmt_dim_map_zh(seg['kept'])}]")
        if seg.get("added"):
            parts.append(f"新增[{_fmt_dim_map_zh(seg['added'])}]")
        if seg.get("removed"):
            parts.append(f"丢失[{_fmt_dim_map_zh(seg['removed'])}]")
        if parts:
            lines.append(f"- {name}：" + "；".join(parts))
    for key, name in (("hard_to_soft", "硬转软"), ("soft_to_hard", "软转硬")):
        if diff.get(key):
            lines.append(f"- {name}：{_fmt_dim_map_zh(diff[key])}")
    flips = diff.get("polarity_flip") or {}
    if flips.get("include_to_exclude") or flips.get("exclude_to_include"):
        lines.append(f"- 极性翻转：正→负 {_fmt_dim_map_zh(flips.get('include_to_exclude')) or '无'}"
                     f"；负→正 {_fmt_dim_map_zh(flips.get('exclude_to_include')) or '无'}")
    def _bool_zh(v: Any) -> str:
        return {None: "无", True: "要", False: "不要"}.get(v, str(v))
    rr = diff.get("raw_required") or [None, None]
    if rr[0] != rr[1]:
        lines.append(f"- 原始数据硬要求：{_bool_zh(rr[0])} → {_bool_zh(rr[1])}")
    dt = diff.get("date") or [["", ""], ["", ""]]
    if dt[0] != dt[1]:
        lines.append(f"- 时间硬过滤：{dt[0]} → {dt[1]}")
    sr = diff.get("soft_raw") or [None, None]
    if sr[0] != sr[1]:
        lines.append(f"- 软偏好·原始数据：{_bool_zh(sr[0])} → {_bool_zh(sr[1])}")
    sd = diff.get("soft_date") or [["", ""], ["", ""]]
    if sd[0] != sd[1]:
        lines.append(f"- 软偏好·时间：{sd[0]} → {sd[1]}")
    ss = diff.get("soft_sources") or [[], []]
    if ss[0] != ss[1]:
        lines.append(f"- 软偏好·来源：{ss[0]} → {ss[1]}")
    ft = diff.get("free_text") or {}
    if ft.get("kept") or ft.get("added") or ft.get("removed"):
        parts = []
        if ft.get("kept"):
            parts.append(f"保留[{'、'.join(ft['kept'])}]")
        if ft.get("added"):
            parts.append(f"新增[{'、'.join(ft['added'])}]")
        if ft.get("removed"):
            parts.append(f"丢失[{'、'.join(ft['removed'])}]")
        lines.append("- 自由主题词：" + "；".join(parts))
    un = diff.get("unresolved") or {}
    if un.get("before") or un.get("after"):
        lines.append(f"- 未解析词：原句 {un.get('before') or '[]'} → 改写 {un.get('after') or '[]'}")
    if diff.get("parse_error"):
        lines.append(f"- 解析异常：{diff['parse_error']}")
    return "\n".join(lines) if len(lines) > 1 else ""


                                                                                  
 
                                                        
                                                         
                                                 
                                                     
                                         
                                                    
                                                                          
                                                            
                                               

_REPAIR_MAX_CANDIDATES = 3


def _repair_norm(s: str) -> str:
    """纠错匹配的归一化：NFKC（全半角/兼容字符）→ 小写 → 去全部空白。"""
    import unicodedata
    return "".join(unicodedata.normalize("NFKC", str(s or "")).lower().split())


def _edit_distance(a: str, b: str) -> int:
    """经典 Levenshtein（两段短串，O(len·len) 足够）。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def _sub_distance(text: str, pat: str) -> int:
    """pat 在 text 内的近似**包含**距离（Sellers 变体：D[i][0]=0 免费跳过 text 前缀、
    答案取末列最小值免费跳过后缀，且**不允许跳过 pat 字符**——pat 的每个字符都必须
    对齐到 text 的某个字符）——未收录词常带上下文残渣（『里贫雪病』←『脐带血里贫雪
    病人』），整串距离会被残渣撑大，包含距离才是 typo 的真实距离。允许跳过 pat 字符
    会把『干性→脑干』一类假包含放进来（2026-09-18 WP3 评测实证），故收紧。"""
    n, m = len(text), len(pat)
    if not m:
        return n
    if not n:
        return m
    prev = list(range(m + 1))
    best = prev[m]
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ca = text[i - 1]
        for j in range(1, m + 1):
                                                            
            cur[j] = min(prev[j] + 1, prev[j - 1] + (ca != pat[j - 1]))
        best = min(best, cur[m])
        prev = cur
    return best


def _repair_eligible(term: str) -> bool:
    """可纠词的资格闸（对抗纪律：正确缩写/编号/大写混写一律不纠——对抗集误纠率必须 0）：
    归一化后 ≥2 字符、不含数字（GSE/E-MTAB/SRP 等编号形一票否决）、不含 ASCII 大写
    （HCC/LUAD/snRNA 等缩写形一票否决）。"""
    norm = _repair_norm(term)
    if len(norm) < 2:
        return False
    if any(ch.isdigit() for ch in norm):
        return False
    if any("A" <= ch <= "Z" for ch in str(term)):
        return False
    return True


def _repair_candidates(term: str) -> list[dict]:
    """单个未收录词的词表近邻（零 LLM，单一真源 `_domain_catalog()`）：全部 alias 做归一化
    编辑距离扫描，两档命中——整串距离 ≤1（参与比较两串较短者 ≤6 字符；更长放宽到 2）
    且**较短者 ≥3 字符**（两字词单字符替换 = 换掉一半内容，证据不足以自动纠错——
    2026-09-18 WP3 评测实证『开头→舌头』即此档假命中）；或**近似包含**距离 ≤1
    （别名嵌在词内、词严格长于别名且残渣前后缀合计 ≤2 字符、别名 ≥2 字符——
    『里贫雪病』→『贫血』一类，残渣是解析器切分的副产品不是语义；纯 ASCII 词
    （拼音形）别名须 ≥4 字符，防「pinxue→pig」一类假命中）。
    同一词条多 alias 命中只留最近一个；按 (距离, alias) 升序取前 `_REPAIR_MAX_CANDIDATES`。
    返回 [{dim, display, alias, targets, absent, dist, via}]——via ∈ edit（整串档）/
    substring（近似包含档，`_repair_route` 一律不自动采纳、仅 advisory：残渣嵌影
    证据弱，2026-09-18 WP5『黏液纤毛→血液』红线实证）/pinyin（拼音档）。
    （absent=词表已知的库中缺席项，
    纠过去得到「正确地无结果」，合法不算失败）。"""
    norm = _repair_norm(term)
    ascii_term = norm.isascii()
    hits: list[dict] = []
    for dim, entries in _domain_catalog().items():
        for e in entries or []:
            best: dict | None = None
            for alias in (e.get("aliases") or []):
                a = _repair_norm(alias)
                if not a or a == norm:
                    continue                             
                limit = 1 if min(len(norm), len(a)) <= 6 else 2
                d = _edit_distance(norm, a) if min(len(norm), len(a)) >= 3 else 10 ** 9
                via = "edit"
                if (d > limit and len(norm) > len(a) >= 2
                        and len(norm) - len(a) <= 2
                        and (not ascii_term or len(a) >= 4)):
                    d_sub = _sub_distance(norm, a)
                    if d_sub <= 1:
                        d = d_sub
                        via = "substring"
                if d <= limit and (best is None or d < best["dist"]):
                    best = {"dim": dim, "display": str(e.get("display") or alias),
                            "alias": str(alias),
                            "targets": [str(t) for t in (e.get("targets") or [])],
                            "absent": bool(e.get("absent")), "dist": d, "via": via}
            if best is not None:
                hits.append(best)
                                                      
                                              
                                                     
    if ascii_term and norm.isalpha() and len(norm) >= 5:
        for h in _repair_pinyin_hits(norm):
            for old in hits:
                if old["dim"] == h["dim"] and old["display"] == h["display"]:
                    if h["dist"] < old["dist"]:
                        hits.remove(old)
                        hits.append(h)
                    break
            else:
                hits.append(h)
    hits.sort(key=lambda h: (h["dist"], h["alias"]))
    return hits[:_REPAIR_MAX_CANDIDATES]


def _repair_pinyin_hits(norm: str) -> list[dict]:
    """拼音连写索引查找（`domain.vocab_tools.pinyin_alias_index` 单通道）：精确 key 命中 dist=0；
    ≥8 字符时同首字母 key 允许 1 编辑距离（dist=1）。返回与 `_repair_candidates`
    同形的候选（带 via=\"pinyin\" 标记）。"""
    index = _vocab_tools.pinyin_alias_index()
    if not index:
        return []
    out: list[dict] = []
    for h in index.get(norm) or []:
        out.append({**h, "dist": 0, "via": "pinyin"})
    if not out and len(norm) >= 8:
        first = norm[0]
        for key, bucket in index.items():
            if not key or key[0] != first or abs(len(key) - len(norm)) > 1:
                continue
            if _edit_distance(norm, key) <= 1:
                for h in bucket:
                    out.append({**h, "dist": 1, "via": "pinyin"})
    return out


def _repair_pinyin_spans(query: str, terms: list[str]) -> list[dict]:
    """相邻纯拼音未收录词的连写重组纠错（2026-09-18 WP5）：解析器按空格切分会把
    整词拼音切碎（fei xian wei hua → 4 个未收录词，各自太短不够单字纠错资格），
    本通道把**查询中相邻**（间隔只含空白/标点，不含任何字母数字——「hua de danxibao」
    里的 de 会正确断开）的纯拼音词逐段拼回长串查拼音索引。
    只产出候选与坐标，不改动文本；采纳纪律在 `_repair_route`（精确唯一才自动采纳）。
    返回 [{term(原文片段), candidates, span:(start,end), term_idx:[...]}]。"""
    if not terms:
        return []
    low = str(query or "").lower()
    pos: list[tuple[int, int]] = []
    cur = 0
    for t in terms:
        p = low.find(str(t).lower(), cur)
        if p < 0:
            return []                                       
        pos.append((p, p + len(str(t))))
        cur = p + len(str(t))
    index = _vocab_tools.pinyin_alias_index()
    if not index:
        return []

    def _ascii_alpha(t: str) -> bool:
        n = _repair_norm(t)
        return bool(n) and n.isascii() and n.isalpha()

                                          
    runs: list[list[int]] = []
    for i in range(len(terms)):
        if not _ascii_alpha(terms[i]):
            continue
        if runs and pos[i][0] - pos[runs[-1][-1]][1] >= 0:
            gap = low[pos[runs[-1][-1]][1]:pos[i][0]]
            prev_end = pos[runs[-1][-1]][1]
            if pos[i][0] >= prev_end and gap and not any(ch.isalnum() for ch in gap):
                runs[-1].append(i)
                continue
        runs.append([i])

    spans: list[dict] = []
    for run in runs:
        if len(run) < 2:
            continue
        best: dict | None = None
        for size in range(len(run), 1, -1):
            for off in range(0, len(run) - size + 1):
                window = run[off:off + size]
                joined = "".join(_repair_norm(terms[i]) for i in window)
                if len(joined) < 5:
                    continue
                exact = [dict(h, dist=0, via="pinyin") for h in index.get(joined) or []]
                cands = exact
                if not cands and size == len(run) and len(joined) >= 8:
                    first = joined[0]
                    for key, bucket in index.items():
                        if not key or key[0] != first or abs(len(key) - len(joined)) > 1:
                            continue
                        if _edit_distance(joined, key) <= 1:
                            cands.extend(dict(h, dist=1, via="pinyin") for h in bucket)
                if not cands:
                    continue
                cands.sort(key=lambda h: (h["dist"], h["alias"]))
                cands = cands[:_REPAIR_MAX_CANDIDATES]
                i0, i1 = window[0], window[-1]
                cand = {"term": str(query)[pos[i0][0]:pos[i1][1]],
                        "candidates": cands, "span": (pos[i0][0], pos[i1][1]),
                        "term_idx": window}
                                               
                if (best is None or len(window) > len(best["term_idx"])
                        or (len(window) == len(best["term_idx"])
                            and min(c["dist"] for c in cands)
                            < min(c["dist"] for c in best["candidates"]))):
                    best = cand
            if (best is not None
                    and min(c["dist"] for c in best["candidates"]) == 0):
                break                                 
        if best is not None:
            spans.append(best)
    return spans


_REPAIR_ABSORB_WINDOW = 2


def _repair_absorbed(original: str, fixes: list[dict], dim: str, val: str) -> bool:
    """丢失约束的「被吸收」判定（2026-09-18 专家化批 WP3）：纠错打补丁会让替换词与
    相邻字粘连成更长实体（『肺』+纤微化→肺纤维化），原句中紧贴替换点的旧条件随之
    被合并消费——这是纠错的忠实语义产物，不是语义丢失。机械口径：丢失的
    (dim, target) 对应词表条目的某个 alias 在原句中出现在某处替换点的
    ±`_REPAIR_ABSORB_WINDOW` 字符窗口内。查不到出处/不在窗口内 → False（不采纳）。"""
    spans: list[tuple[int, int]] = []
    for f in fixes or []:
        term = str(f.get("term") or "")
        p = original.find(term) if term else -1
        if p >= 0:
            spans.append((p, p + len(term)))
    if not spans:
        return False
    low = str(val).lower()
    for e in (_domain_catalog().get(dim) or []):
        if low not in {str(t).lower() for t in (e.get("targets") or [])}:
            continue
        for alias in (e.get("aliases") or []):
            a = str(alias)
            if not a:
                continue
            start = original.find(a)
            while start >= 0:
                end = start + len(a)
                if any(s - _REPAIR_ABSORB_WINDOW <= start and end <= t + _REPAIR_ABSORB_WINDOW
                       for s, t in spans):
                    return True
                start = original.find(a, start + 1)
    return False


def _repair_diff_ok(flow: Any, original: str, patched: str, fixes: list[dict],
                    structured_kwargs: dict) -> "tuple[bool, dict]":
    """纠错补丁的白名单语义校验（零 LLM，2026-09-18 专家化批 WP3）。

    **对比基线是降级解析，不是弃权解析**：原句弃权时约束为空，拿它当 before 会把
    上下文条件全记成「新增」而误杀正确纠错（探针实证：『纤微化』→『纤维化』被拒，
    根因是词表 entry.targets 与解析器语境内解析口径不同——『肺纤维化』语境解析出
    disease=[pulmonary fibrosis, interstitial lung]，与 alias 条目 targets=[fibrosis]
    对不上）。诚实口径 = 主管线自动降级同一段 `strip_terms`：before = 剥掉未收录词后
    的原句解析；after = 补丁句（仍弃权时同样剥掉残余未收录词）的解析。两侧差集即
    纠错的真实语义产物——凡是纠错引入的新增必然显形，凡与纠错无关的条件必须原样
    保留。

    白名单：硬/软**新增**必须落在被纠实体的维度内（跨维出现=粘连出意外语义，拒）；
    硬/软**丢失**逐项过 `_repair_absorbed`（不被吸收的丢失=纠错动了上下文，拒）；
    排除增删、软硬迁移、极性翻转、原始数据/时间硬要求变化、自由词新增、解析异常
    一律拒。另要求解析状态改善：未收录词减少或补丁句可执行化。
    返回 (是否采纳, 记账 dict)——带 authorized_added/lost_absorbed/unresolved 明细。"""
    info: dict[str, Any] = {"authorized_added": {}, "lost_absorbed": {},
                            "unresolved": {"before": [], "after": []},
                            "reject_reason": ""}
    scene = {k: v for k, v in dict(structured_kwargs or {}).items() if k in _DIFF_SCENE_KEYS}

    def _parse(q: str):
        intent, *_ = flow._prepare_context(q, sources=None,
                                           auto_parse_sources=False, **scene)
        return intent

    def _degraded(q: str):
        """与主管线自动降级同口径：unresolved_term 弃权时剥词重解析（至多 3 轮）。"""
        from ..app.workflow import strip_terms
        live = str(q or "").strip()
        if not live:
            return None
        for _ in range(3):
            intent = _parse(live)
            if not (getattr(intent, "abstain", False)
                    and str(getattr(intent, "abstain_reason", "") or "") == "unresolved_term"):
                return intent
            terms = [str(t) for t in (getattr(intent, "unresolved_terms", None) or [])
                     if str(t).strip()]
            nxt = " ".join(strip_terms(live, terms).split()).strip()
            if not nxt or nxt == live:
                return intent
            live = nxt
        return _parse(live)

    try:
        b0, a0 = _parse(original), _parse(patched)
    except Exception as exc:                                    
        info["reject_reason"] = f"parse_error: {type(exc).__name__}"
        return False, info
    before_terms = [str(t) for t in (getattr(b0, "unresolved_terms", None) or [])
                    if str(t).strip()]
    after_terms = [str(t) for t in (getattr(a0, "unresolved_terms", None) or [])
                   if str(t).strip()]
    info["unresolved"] = {"before": before_terms, "after": after_terms}
    a_exec = (not getattr(a0, "abstain", False)
              or str(getattr(a0, "parse_status", "") or "") == "executable")
    if not (a_exec or len(after_terms) < len(before_terms)):
        info["reject_reason"] = "no_parse_improvement"
        return False, info
    try:
        base, new = _degraded(original), _degraded(patched)
    except Exception as exc:                
        info["reject_reason"] = f"parse_error: {type(exc).__name__}"
        return False, info

    def _view(intent: Any) -> dict:
        if intent is None:
            return {"hard": {}, "excluded": {}, "soft": {}, "raw": None,
                    "date": ["", ""], "free": set()}
        return {"hard": dict(getattr(intent, "constraints", None) or {}),
                "excluded": dict(getattr(intent, "excluded_constraints", None) or {}),
                "soft": dict(getattr(intent, "preferred_constraints", None) or {}),
                "raw": getattr(intent, "has_raw_data_required", None),
                "date": [str(getattr(intent, "date_from", "") or ""),
                         str(getattr(intent, "date_to", "") or "")],
                "free": set(getattr(intent, "free_text_terms", None) or [])}

    vb, va = _view(base), _view(new)
    fix_dims = {str(f.get("dim") or "") for f in fixes or []}
    hard = _dim_set_diff(vb["hard"], va["hard"])
    soft = _dim_set_diff(vb["soft"], va["soft"])
    for side, seg in (("hard", hard), ("soft", soft)):
        for dim, vals in (seg.get("added") or {}).items():
            if dim not in fix_dims:
                info["reject_reason"] = f"added_outside_fix_dims:{dim}"
                return False, info
            info["authorized_added"].setdefault(side, {}).setdefault(dim, []).extend(vals)
        for dim, vals in (seg.get("removed") or {}).items():
            kept = [v for v in vals if _repair_absorbed(original, fixes, dim, v)]
            if len(kept) != len(vals):
                info["reject_reason"] = f"lost_not_absorbed:{dim}:{vals}"
                return False, info
            if kept:
                info["lost_absorbed"].setdefault(side, {}).setdefault(dim, []).extend(kept)
    excl = _dim_set_diff(vb["excluded"], va["excluded"])
    if excl.get("added") or excl.get("removed"):
        info["reject_reason"] = "excluded_changed"
        return False, info
    if (_role_shift(vb["hard"], va["soft"]) or _role_shift(vb["soft"], va["hard"])
            or _role_shift(vb["hard"], va["excluded"])
            or _role_shift(vb["excluded"], va["hard"])):
        info["reject_reason"] = "role_or_polarity_shift"
        return False, info
    if vb["raw"] != va["raw"] or vb["date"] != va["date"]:
        info["reject_reason"] = "raw_or_date_changed"
        return False, info
    if va["free"] - vb["free"]:
        info["reject_reason"] = "free_text_added"
        return False, info
    return True, info


def _repair_route(flow: Any, query: str, structured_kwargs: dict) -> dict | None:
    """repair 路由（改写通道前的零 LLM 局部纠错，2026-09-18 专家化批 WP3）：
    原句因未收录词弃权（unresolved_term）时，逐词在受控词表上找最近邻 alias——
    **无歧义且至少一词可唯一采纳**才打补丁（唯一最近邻的词 str.replace 单次替换、
    无候选的词原样保留——下游自动降级会忽略它并显著标注，纠错+降级两机制接力），
    再过 `_repair_diff_ok` 白名单校验；有等距多候选的歧义词/白名单不过 →
    adopted=False + 候选清单（advisory，歧义不靠命中数裁决、「不改」恒保留），
    调用方走改写兜底。非 unresolved_term 弃权/无未收录词 → None（本术式不适用，
    照常改写）。返回 {terms: [{term, candidates}], adopted, patched_query,
    adopted_fixes, diff}。"""
    scene = {k: v for k, v in dict(structured_kwargs or {}).items() if k in _DIFF_SCENE_KEYS}
    try:
        intent, *_ = flow._prepare_context(query, sources=None,
                                           auto_parse_sources=False, **scene)
    except Exception:                                      
        return None
    if not (getattr(intent, "abstain", False)
            and str(getattr(intent, "abstain_reason", "") or "") == "unresolved_term"):
        return None
    terms = [str(t).strip() for t in (getattr(intent, "unresolved_terms", None) or [])
             if str(t).strip()]
    if not terms:
        return None
    per_term: list[dict] = []
    for t in terms:
        cands = _repair_candidates(t) if _repair_eligible(t) else []
        per_term.append({"term": t, "candidates": cands})
                                               
                                   
    spans = _repair_pinyin_spans(query, terms)
    info: dict[str, Any] = {"terms": per_term + [{"term": s["term"],
                                                  "candidates": s["candidates"]}
                                                 for s in spans],
                            "adopted": False,
                            "patched_query": "", "adopted_fixes": [], "diff": None}
    adoptable, ambiguous = [], False
    span_fixes: list[dict] = []
    covered: set[int] = set()
    for s in spans:
        cands = s["candidates"]
                                                         
                                   
        if cands and cands[0]["dist"] == 0 and len(cands) == 1:
            span_fixes.append(s)
            covered.update(s["term_idx"])
        elif len(cands) >= 2 and cands[0]["dist"] == cands[1]["dist"]:
            ambiguous = True
    for idx_t, c in enumerate(per_term):
        if idx_t in covered:
            continue
        cands = c["candidates"]
        if cands and (len(cands) == 1 or cands[0]["dist"] < cands[1]["dist"]):
                                                  
            if cands[0].get("via") == "pinyin" and cands[0]["dist"] > 0:
                continue
                                                         
                                                         
                                                 
                                         
            if cands[0].get("via") == "substring":
                continue
            adoptable.append(c)
        elif len(cands) >= 2:                          
            ambiguous = True
    if ambiguous or (not adoptable and not span_fixes):
        return info
    patched = query
    adopted: list[dict] = []
    for s in sorted(span_fixes, key=lambda x: -x["span"][0]):           
        pick = s["candidates"][0]
        start, end = s["span"]
        patched = patched[:start] + pick["alias"] + patched[end:]
        adopted.append({"term": s["term"], "alias": pick["alias"],
                        "dim": pick["dim"], "display": pick["display"],
                        "targets": pick["targets"]})
    for c in adoptable:
        pick = c["candidates"][0]
        if c["term"] not in patched:                      
            return info
        patched = patched.replace(c["term"], pick["alias"], 1)
        adopted.append({"term": c["term"], "alias": pick["alias"],
                        "dim": pick["dim"], "display": pick["display"],
                        "targets": pick["targets"]})
    if patched == query:
        return info
    ok, account = _repair_diff_ok(flow, query, patched, adopted, structured_kwargs)
    info["patched_query"] = patched
    info["adopted_fixes"] = adopted
    info["diff"] = account
    if not ok:
        return info
    info["adopted"] = True
    return info


def _repair_note_zh(repair: dict, empty: bool = False) -> str:
    """纠错事实句（机械事实，不评好坏）：采纳档列「原词 → 纠正词（维度）」清单，
    `empty=True`（纠正后试跑零存活）时降级为「仅供参考」措辞——覆盖校验不过
    的纠正不配断言式呈现（2026-09-18 WP3 评测实证：误纠全部伴随零存活）；
    未采纳档列候选 advisory（交模型/用户裁决）。"""
    if not repair:
        return ""
    if repair.get("adopted"):
        fixes = "；".join(f"「{f['term']}」→「{f['alias']}」（{_domain_labels_zh().get(f['dim'], f['dim'])}）"
                          for f in (repair.get("adopted_fixes") or []))
        if not fixes:
            return ""
        base = f"未收录词已按词表就近纠正：{fixes}。"
        if empty:
            return (base + "但纠正后的条件在本库没有存活结果——该纠正仅供参考，"
                    "请核实是否符合你的意图。")
        return base
    parts = []
    for c in (repair.get("terms") or []):
        cands = c.get("candidates") or []
        if cands:
            opts = "、".join(f"「{h['alias']}」（{_domain_labels_zh().get(h['dim'], h['dim'])}）"
                             for h in cands)
            parts.append(f"「{c['term']}」可指 {opts}")
        else:
            parts.append(f"「{c['term']}」词表无近邻")
    return ("未收录词纠错候选（歧义/无把握，未自动采纳）：" + "；".join(parts) + "。"
            if parts else "")


def _rewrite_repair_zh(repair: dict) -> str:
    """repair 候选菜单的改写注入块（机械事实，2026-09-18 专家化批 WP3）：
    repair 路由有候选但未自动采纳（歧义/白名单不过）时，把候选清单作为参考事实
    注入改写调用，**交模型裁决**——可采用候选、可自由改写、也可忽略；无候选的词
    如实标注「词表无近邻」。空清单/已采纳返回空串（已采纳根本不会走到改写通道）。"""
    if not repair or repair.get("adopted"):
        return ""
    lines: list[str] = []
    for c in (repair.get("terms") or []):
        cands = c.get("candidates") or []
        if cands:
            opts = "、".join(
                f"「{h['alias']}」（{_domain_labels_zh().get(h['dim'], h['dim'])}）"
                for h in cands)
            lines.append(f"- 「{c['term']}」可指：{opts}")
        else:
            lines.append(f"- 「{c['term']}」：词表无近邻")
    if not lines:
        return ""
    return ("参考：原句未收录词的词表近邻候选（机械事实，仅供判断——有把握的可在改写中"
            "采用，没把握的忽略，不许照抄自己不确定的词）：\n" + "\n".join(lines))


def _latest_screen_facts(ctx: dict) -> dict:
    """`_rewrite_health_zh` 的环内取数器（2026-09-16 改写环臂批）：倒序扫环内 steps 取
    最近一次检索事实——rank/rerank 试跑结果 dict 直接带出健康度/结果数/解析状态/执行的
    条件/top 标题；search.rerun 采纳档与 results.show 成功档视同 healthy（与
    `_screen_baseline` 同口径）；无环内事实按屏面状态退化（有结果 unknown——不假装
    知道它健不健康；无结果 degenerate）。只读机械事实，不评内容质量。"""
    for s in reversed(list(ctx.get("steps") or [])):
        if not (isinstance(s, dict) and s.get("ok")):
            continue
        v = str(s.get("verb") or "")
        r = s.get("result") if isinstance(s.get("result"), dict) else {}
        if v in ("rank", "rerank") and r.get("health"):
            return {"health": str(r.get("health")), "total": int(r.get("total") or 0),
                    "resolution_status": str(r.get("resolution_status") or ""),
                    "filters": list(r.get("filters") or []),
                    "top_titles": [t.get("dataset_name") for t in list(r.get("top") or [])
                                   if isinstance(t, dict)]}
        if v == "search.rerun" and r.get("adopted"):
            return {"health": "healthy", "total": int(r.get("n_after_total") or 0)}
        if v == "results.show" and r.get("shown"):
            return {"health": "healthy", "total": int(r.get("total") or 0)}
    if ctx.get("has_results"):
        return {"health": "unknown", "total": int(ctx.get("result_total") or 0)}
    return {"health": "degenerate", "total": 0}


def _top_digest(rows: Any, n: int = 3) -> list[dict[str, Any]]:
    """top 条目紧凑 digest（rank/rerank 返回契约的 top 键）：七字段——
    dataset_uid（2026-08-20 批补：同批依赖占位的解析源，`$<N>.top[<i>].dataset_uid`）/
    dataset_name / species / tissue / disease / source / rank（1 起序号，批补——
    decide 据此知道 top[0]/top[1] 指哪条）。原始候选 dict 与卡片行 dict 的键名在这几
    字段上同形，两种输入通吃。"""
    digest: list[dict[str, Any]] = []
    for idx, r in enumerate(list(rows or [])[:n], start=1):
        if not isinstance(r, dict):
            continue
        digest.append({
            "dataset_uid": r.get("dataset_uid", ""),
            "dataset_name": r.get("dataset_name", ""),
            "species": r.get("species", ""),
            "tissue": r.get("tissue", ""),
            "disease": r.get("disease", ""),
            "source": r.get("source", ""),
            "rank": idx,
        })
    return digest


                                                                                  
 
                                                  
                                                             
                                                                          
                                                                
                                                             
                                                                         
                                                               
                        


def _health_of_meta(meta: Any, *, structured_context_lost: bool = False) -> str:
    """检索结果的**规则可断言健康度**（三态）——只看机械事实，不评内容质量：
    "degenerate"（零命中 / 解析弃权 / 向量语义兜底 / 查询待澄清 / 结构化现场丢失）
    ——屏面上没有经条件核验的结果；"healthy"（有命中、未降级、现场完整）——规则程序
    可断言的好，auto-show 只在本态触发；其余 "watch"（有结果但走了自动降级）。
    「更好」是内容判断，不归本谓词——双方都健康时机械层只报事实。"""
    if structured_context_lost:
        return "degenerate"
    total = int(getattr(meta, "result_total", 0) or 0)
    status = str(getattr(meta, "resolution_status", "") or "")
    if total <= 0 or status in ("abstained", "vector_fallback", "clarification_required"):
        return "degenerate"
    if status == "degraded":
        return "watch"
    return "healthy"


def _verified_recommend_payload(meta: Any, structured_kwargs: dict
                                ) -> "tuple[dict | None, bool]":
    """recommend 同形载荷 + 应用态回填 + scope_kept 机械复核——rank auto-show /
    rerank auto-show / search.rerun 采纳档 / results.show 重放共用（单通道）。
    显式日期必须出现在 interpretation，三个应用态必须逐位等于输入；任一丢失 →
    (None, False)，调用方 fail-closed 不出批，宁可保留原结果也不放宽重跑。"""
    from ..app.conversation_results import complete_payload

    payload = complete_payload(meta, params=structured_kwargs)
                                                           
                           
    payload["applied_facets"] = structured_kwargs["facet_filters"]
    payload["applied_suppressed"] = structured_kwargs["suppressed_constraints"]
    payload["applied_lenient"] = structured_kwargs["lenient_dims"]
    interpretation = payload.get("interpretation") if isinstance(payload.get("interpretation"), dict) else {}
    intent_projection = interpretation.get("intent") \
        if isinstance(interpretation.get("intent"), dict) else {}
    scope_kept = (
        payload.get("applied_facets") == structured_kwargs["facet_filters"]
        and payload.get("applied_suppressed") == structured_kwargs["suppressed_constraints"]
        and payload.get("applied_lenient") == structured_kwargs["lenient_dims"]
        and (not structured_kwargs["date_from"]
             or str(intent_projection.get("date_from") or "") == structured_kwargs["date_from"])
        and (not structured_kwargs["date_to"]
             or str(intent_projection.get("date_to") or "") == structured_kwargs["date_to"])
    )
    if not scope_kept:
        return None, False
    return payload, True


def _build_show_batch(origin: str, query_effective: str, query_raw: str,
                      payload: dict) -> dict:
    """上屏批次原料（turn.py 组卷只认这个形状）：kind = **内容来源**动词
    （rank/rerank——batch_select/results 两处前端消费点按内容来源分派，show 只是出口，
    前端无感）；label 截 20 字；query_raw = 本轮用户原话（契约）。"""
    return {
        "kind": origin,
        "label": query_effective[:20],
        "query_raw": query_raw,
        "query_effective": query_effective,
        "payload": payload,
    }


def _screen_baseline(ctx: dict) -> dict:
    """rerank/ask.relax 闸门的**屏面基线**（机械现取，不落状态）：倒序扫环内 steps，取最近
    **真实上屏**的批次——
      · rank/rerank 的 auto-show 成功档（displayed=True 且 batch 非空）；
      · results.show 的成功档（shown=True 且 batch 非空；幂等档 batch=None 跳过——
        原批次步在更前面，内容同源）；
      · search.rerun 采纳档（已过择优/应用态闸门上屏）。
    2026-09-18 专家化批修正：此前取「最近一次带 health 的 rank/rerank 试跑」，无论它有没有
    上屏——被拒/仅 advisory 的试跑会冒充屏面（剧本：健康屏 A 在显示 → 试跑 B 被拒留在
    steps → 下一次调用拿 B 当基线，把健康屏误判为空屏 → auto-show 绕过 LLM 裁决顶掉 A）。
    试跑只是情报，屏面只认上屏事实。
    无环内上屏事实时按屏面状态退化：有结果 → "unknown"（屏面内容未经本环核验，机械层
    不假装知道它健不健康——生死线不触发，全走 advisory）；无结果 → "degenerate"
    （空屏面：任何健康结果都严格更好，0→N 自动上屏的合法触发面）。
    合成健康度纪律：results.show/search.rerun 不自带 health 键——按上屏条数合成
    （total>0 → healthy；==0 → degenerate，空屏不许冒充健康屏）。"""
    for s in reversed(list(ctx.get("steps") or [])):
        if not (isinstance(s, dict) and s.get("ok")):
            continue
        v = str(s.get("verb") or "")
        r = s.get("result") if isinstance(s.get("result"), dict) else {}
        if v in ("rank", "rerank") and r.get("displayed") and r.get("batch"):
            return {"health": str(r.get("health") or "healthy"),
                    "total": int(r.get("total") or 0),
                    "origin": v,
                    "query": str(r.get("query") or r.get("rewritten_query") or "")}
        if v == "results.show" and r.get("shown") and r.get("batch"):
            total = int(r.get("total") or 0)
            return {"health": "healthy" if total > 0 else "degenerate",
                    "total": total, "origin": v,
                    "query": str(r.get("query_effective") or "")}
        if v == "search.rerun" and r.get("adopted"):
            total = int(r.get("n_after_total") or 0)
            return {"health": "healthy" if total > 0 else "degenerate",
                    "total": total, "origin": v,
                    "query": str(r.get("query") or "")}
    if ctx.get("has_results"):
        return {"health": "unknown", "total": int(ctx.get("result_total") or 0),
                "origin": "screen", "query": str(ctx.get("current_query") or "")}
    return {"health": "degenerate", "total": 0, "origin": "screen", "query": ""}


_HEALTH_ZH = {"healthy": "条件核验通过", "watch": "走了自动降级（忽略部分条件）",
              "degenerate": "没有经条件核验的结果", "unknown": "未经本环核验"}


def _screen_comparison_zh(baseline: dict, trial_health: str, trial_total: int,
                          trial_query: str) -> str:
    """rerank 试跑的新旧对照句（机械事实清单，**不判「更好」**——内容优劣的裁决归模型，
    经 results.show 表达）。"""
    if baseline.get("origin") == "screen" and baseline.get("health") == "degenerate":
        base_desc = "当前屏面：无结果"
    elif baseline.get("health") == "unknown":
        base_desc = (f"当前屏面：「{baseline.get('query') or '—'}」"
                     f"{int(baseline.get('total') or 0)} 条（{_HEALTH_ZH['unknown']}）")
    else:
        base_desc = (f"当前屏面：「{baseline.get('query') or '—'}」"
                     f"{int(baseline.get('total') or 0)} 条"
                     f"（{_HEALTH_ZH.get(str(baseline.get('health') or ''), '未知')}）")
    trial_desc = (f"本次试跑：「{trial_query}」命中 {trial_total} 条"
                  f"（{_HEALTH_ZH.get(trial_health, '未知')}）")
    return f"新旧对照——{base_desc}；{trial_desc}。"


                                                                                    
 
                                 
                                                  
                                            
                                                  


class _RerankRelaxParamError(Exception):
    """rerank 的 relax 槽含非法 key（不是探测报告里的项/该维当前无约束可松）。"""

    code = "bad_param"

    def __init__(self, key: str):
        super().__init__(key)
        self.hint = (f"relax 槽里的「{key}」不是本次探测报告给出的可放松项。"
                     "修正方法：relax 只填报告里列出的 key；或者清空 relax 槽改用改写。")


def _relaxation_probe_zh(flow: Any, query: str, sources: Any,
                         structured_kwargs: dict) -> str:
    """rerank 改写调用前的**约束松弛反事实探测报告**（零 LLM、纯确定性）：对当前查询
    解析出的约束集逐维试算「放松它能救回多少条 + top 示例」，作为**建议**注入。
    复用 `retriever.relaxation_options`（单通道原则；include_dead=True 保留死维——
    「放松也无救」是信息量）。弃权态/无约束/探测异常都如实说明，不静默吞掉。"""
    try:
        intent, _cands, normalized, *_ = flow._prepare_context(
            query, sources=sources, auto_parse_sources=False, **structured_kwargs)
    except Exception as exc:                                  
        return f"（松弛试算未能完成：{type(exc).__name__}——请按原句自行判断是否改写）"
    if getattr(intent, "abstain", False):
        return ("参考：本查询未能完整解析出可执行条件，没有可放松的维度——"
                "这种情况优先考虑改写查询本身（规范实体名、中文主题词补英文对照）。")
    options = flow.retriever.relaxation_options(
        normalized, intent, top_k=3, max_options=6, include_dead=True)
    if not options:
        return "参考：本查询没有可放松的条件（未解析出硬约束）。"
    lines = ["参考：当前条件的松弛试算（确定性预演，仅建议——可照做、可自由改写、也可都不做）："]
    for o in options:
        names = "、".join(
            str(getattr(getattr(c, "record", None), "dataset_name", "") or "")
            for c in list(o.get("candidates") or [])[:2])
        names = names.strip("、")
        verb = "只按" if o.get("kind") == "only" else "放松"
        suffix = "搜" if o.get("kind") == "only" else ""
        if o.get("count"):
            example = f"（例：{names}）" if names else ""
            lines.append(f"- {o['key']}：{verb}「{o['label']}」{suffix}→ {o['count']} 条{example}")
        else:
            lines.append(f"- {o['key']}：{verb}「{o['label']}」{suffix}→ 0 条（放松也无救）")
    lines.append("若决定照某条建议放松，把它加到 relax 槽（可多选）后重发本工具；"
                 "否则正常输出改写句；两者都不合适就原样输出原句。")
    return "\n".join(lines)


def _relax_live_digest(flow: Any, query: str, sources: Any,
                       structured_kwargs: dict) -> list[dict] | None:
    """ask.relax 适用性情报（2026-09-14 YOLO 批；零 LLM 确定性预演）：对查询解析出的
    约束集逐维试算「放松它能救回几条」，回传紧凑 digest（key/label/count；
    include_dead=False——「放松也无救」的死维不列）。解析弃权/预演异常 → None
    （情报不可得，调用方保持通用文案，不妄断）。

    嵌进 degenerate 试跑结果（rank/rerank）的 `relax_live` 键：decide 的
    「结果不健康告警」（`_screen_alert_block_zh`）据此点名活选项（ask.relax 是
    对因对策）或如实标注「ask.relax 不适用」——模型不再凭通用对策清单盲选，
    也不会空撞 no_options 机械闸。与 `_loop_ask_relax` 的弹卡预演同一
    `retriever.relaxation_options` 真源（单通道原则），此处只做紧凑投影。"""
    try:
        intent, _cands, normalized, *_ = flow._prepare_context(
            query, sources=sources, auto_parse_sources=False, **structured_kwargs)
    except Exception:                              
        return None
    if getattr(intent, "abstain", False):
        return None
    try:
        options = flow.retriever.relaxation_options(
            normalized, intent, top_k=3, max_options=6, include_dead=False)
    except Exception:                                  
        return None
    return [{"key": str(o.get("key") or ""), "label": str(o.get("label") or ""),
             "count": int(o.get("count") or 0)} for o in options]


def _relax_to_suppressed(intent: Any, keys: Any) -> list[str]:
    """把 rerank relax 槽的探测报告 key 翻译成 suppressed_constraints 口径（filter_id
    极性形式，单通道复用条件板「忽略此条件」的既有执行与呈现）：dim:X→include:X /
    exclude:X→exclude:X / raw→按当前值 raw:required|raw:forbidden / date→date:range /
    only:X→删其余全部（其它正向维+全部排除维+FASTQ+时间范围）。
    翻译真源是 `query_parser.relax_key_to_suppressed`（2026-09-14 收编上移，单锚点）；
    本层只做 LLM 入参侧的契约适配——非法 key 抛 `_RerankRelaxParamError`（如实拒绝，不静默忽略）。"""
    from ..retrieval.query_parser import relax_key_to_suppressed

    suppressed: list[str] = []
    for raw in list(keys or []):
        key = str(raw or "").strip()
        if not key:
            continue
        try:
            suppressed.extend(relax_key_to_suppressed(intent, key))
        except ValueError:
            raise _RerankRelaxParamError(key)
    out: list[str] = []
    for s in suppressed:
        if s not in out:
            out.append(s)
    return out


def _resolve_show_handle(handle: str, steps: list[dict]) -> "tuple[int | None, dict | None]":
    """results.show 的句柄解析：latest（缺省）= 最近一次 rank/rerank 试跑；
    latest_rank / latest_rerank = 该动词最近一次；数字 = 步号（1 起，与 decide 投影
    里的执行序号同口径）。只认 ok 且带 health 键的试跑结果（rank/rerank）。
    返回 (步号, 步实录)；找不到 → (None, None)。"""
    def _is_trial(s: Any, verb: str | None = None) -> bool:
        if not (isinstance(s, dict) and s.get("ok")):
            return False
        v = str(s.get("verb") or "")
        if v not in ("rank", "rerank") or (verb and v != verb):
            return False
        r = s.get("result")
        return isinstance(r, dict) and bool(r.get("health"))

    steps = list(steps or [])
    h = str(handle or "").strip() or "latest"
    if h.isdigit():
        idx = int(h)
        if 1 <= idx <= len(steps) and _is_trial(steps[idx - 1]):
            return idx, steps[idx - 1]
        return None, None
    verb = {"latest": None, "latest_rank": "rank", "latest_rerank": "rerank"}.get(h)
    if h not in ("latest", "latest_rank", "latest_rerank"):
        return None, None
    for i in range(len(steps) - 1, -1, -1):
        if _is_trial(steps[i], verb):
            return i + 1, steps[i]
    return None, None


def _loop_show(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """results.show：把环内一次 rank/rerank **试跑**结果放上结果区——屏态唯一出口
    （2026-09-13 rerank 处置批；display 槽已退役，其它工具没有上屏权限，rank/rerank 的
    auto-show 是本路径不经 LLM 的机械直调）。

    试跑缓存不落 state：句柄解析到 steps 里的试跑记录（rank/rerank 结果 dict 自带
    生效查询/命中数/health），用记录的生效查询 + **当前**结构化现场重放同一条确定性
    管线（use_llm=False）重建 recommend 同形载荷，并做指纹复核（命中总数一致 +
    应用态逐位保留）；无此试跑、记录缺检索句、指纹不符都 fail-closed（shown=False +
    note_zh 如实说明），绝不拿旧现场的内容冒充现在的。幂等：试跑已上屏（auto-show）
    时不重复出批。只读；批次 kind = 试跑来源动词，前端按 kind 渲染无感。"""
    from ..app import workflow as wf

    slots = slots or {}
    ctx = ctx or {}
    handle = str(slots.get("handle") or "").strip() or "latest"
    steps = list(ctx.get("steps") or [])
    _no, record = _resolve_show_handle(handle, steps)
    if record is None:
        return {"shown": False, "handle": handle,
                "note_zh": f"上屏没有执行：没有找到「{handle}」对应的检索/重查试跑记录。"}
    origin = str(record.get("verb") or "")
    res = record.get("result") if isinstance(record.get("result"), dict) else {}
    query_effective = str(res.get("query") or res.get("rewritten_query") or "").strip()
    if not query_effective:
        return {"shown": False, "handle": handle, "origin": origin,
                "note_zh": "上屏没有执行：试跑记录里没有可重放的检索句。"}
    if res.get("displayed"):
                                                             
                    
        return {"shown": True, "handle": handle, "origin": origin,
                "query_effective": query_effective,
                "total": int(res.get("total") or 0), "overridden": False,
                "batch": None, "displayed": True,
                "note_zh": "该结果已在结果区，未重复上屏。"}
                                                  
                                           
    structured_kwargs = _replay_structured_kwargs(ctx, record)
    flow = wf.DatasetRecommendationWorkflow()
    meta = flow.run_with_meta(
        wf.RecommendParams(
            query=query_effective, use_llm=False,
            sources=ctx.get("search_sources"),
            auto_parse_sources=False, **structured_kwargs))
    payload, scope_kept = _verified_recommend_payload(meta, structured_kwargs)
    fingerprint_ok = scope_kept and int(meta.result_total or 0) == int(res.get("total") or 0)
    if not fingerprint_ok:
        return {"shown": False, "handle": handle, "origin": origin,
                "query_effective": query_effective,
                "total": int(res.get("total") or 0),
                "note_zh": "上屏没有执行：重放结果与试跑时已不一致（结果已过期或现场已变），"
                           "当前结果区未改动——请重新检索后再上屏。"}
    batch = _build_show_batch(
        origin, query_effective,
        str(ctx.get("utterance") or "").strip() or query_effective, payload)
    return {"shown": True, "handle": handle, "origin": origin,
            "query_effective": query_effective,
            "total": int(meta.result_total or 0),
            "overridden": bool(ctx.get("has_results")),
            "batch": batch, "displayed": True}


def _loop_rank(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """裸新检索（**试跑化**，2026-09-13 rerank 处置批）：以 query 跑标准 RAG 管线
    （与 /api/recommend 同核心的确定性段——`run_with_meta` 规则检索，与 search.rerun
    同口径），如实回报 {total, 生效条件, top digest} + 机械健康度（health/
    resolution_status，`_health_of_meta` 谓词）。
    **healthy 即自动上屏**（results.show 路径不经 LLM 的机械直调：构造 recommend
    同形载荷 + 批次原料，displayed=True；应用态丢失则 fail-closed 不上屏并如实标注
    structured_context_lost）；其余健康度只回报、不占屏面——屏态唯一出口是
    results.show 动词（display 槽已退役）。
    与 search.rerun 的界限不变：不做机械择优、不与基准比对。
    只读；query 槽空 → `_SearchRerunParamError`（bad_param，与 search.rerun 同纪律）。"""
    from ..app import workflow as wf

    slots = slots or {}
    ctx = ctx or {}
    query = str(slots.get("query") or "").strip()
    if not query:
        raise _SearchRerunParamError()
    sources = ctx.get("search_sources")
                                                
                                                                  
                                                
    structured_kwargs = _loop_structured_kwargs(ctx)

    flow = wf.DatasetRecommendationWorkflow()
    meta = flow.run_with_meta(
        wf.RecommendParams(
            query=query, use_llm=False, sources=sources,
            auto_parse_sources=False, **structured_kwargs))
    health = _health_of_meta(meta)
    batch = None
    displayed = False
    structured_lost = False
    rows: Any = meta.retrieved_data
    if health == "healthy":
                                                         
                                                                                      
        payload, scope_kept = _verified_recommend_payload(meta, structured_kwargs)
        if scope_kept:
            rows = payload.get("results")
            displayed = True
            batch = _build_show_batch(
                "rank", query, str(ctx.get("utterance") or "").strip() or query, payload)
        else:
                                                              
                                                       
            structured_lost = True
            health = "degenerate"
    result = {
        "query": query,
        "total": int(meta.result_total or 0),
                                                                     
                                                                     
        "filters": list(meta.active_filters or []),
        "top": _top_digest(rows),
        "displayed": displayed,
        "batch": batch,
        "health": health,
        "resolution_status": str(getattr(meta, "resolution_status", "") or ""),
    }
    if structured_lost:
        result["structured_context_lost"] = True
        result["disclosure_zh"] = ("这次检索没能完整保留当前筛选条件，为避免放宽条件后"
                                   "误出结果，已保留原结果。")
        return result
    if health == "degenerate":
                                                                    
                                                              
                                                     
        digest = _relax_live_digest(flow, query, sources, structured_kwargs)
        if digest is not None:
            result["relax_live"] = digest
    return result


                                                    
                                 
                                                                                      
                                                           
                                  
                                                               
                                                                              
                                                                  
                                                                         
                                                          
                                                                
                                                               
                            
_RERANK_TRIGGER_MAX_TOTAL = 3


def _loop_rerank(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """坏 query 优化重检（**试跑化**，2026-09-13 rerank 处置批；双通道 2026-09-13 夜批）：
    **双通道入口**——
      · relax 槽非空（结构化放松）：跳过改写（零 LLM），把探测报告 key 翻译成
        suppressed_constraints 后经标准管线重检**原句**；非法 key 抛
        `_RerankRelaxParamError`（bad_param，如实拒绝）；
      · relax 空（默认）：先过 **repair 路由**（2026-09-18 专家化批 WP3，零 LLM
        局部纠错——原句因未收录词弃权时，受控词表唯一最近邻打补丁试跑，比整句改写
        更窄的干预优先；不适用/不采纳则候选 advisory 随结果回传），再走
        **约束松弛反事实探测**（零 LLM，报告仅建议）+ **独立 LLM 改写调用**兜底
        （报告随调用注入；机械健全性检查兜底，不过退回原句且 rewritten=False
        如实标注）→ 跑标准管线（与 `_loop_rank` 同口径）。
    两通道汇流后**只试跑不直接上屏**（display 槽已退役），机械对照新旧结果健康度。
    **触发闸**（2026-09-18 触发子集批，阈值 `_RERANK_TRIGGER_MAX_TOTAL`）：屏面已核验
    健康或主检索直出（healthy/unknown）且结果充足（>3 条）时，本工具只有下行风险——
    如实拒绝（refused=true + refuse_reason=screen_healthy，拒绝是数据不是故障，
    同 ask.relax 的 screen_not_degenerate 先例；零 LLM、零试跑，换措辞重搜请走
    search.rerun）。
    过闸后的三态出口（**语义闸扩档**，2026-09-18 触发子集批——改写定稿与原句经
    `_constraint_diff` 零 LLM 机械对比，未授权语义变化（硬约束增删/排除变化/软硬转换/
    极性翻转/原始数据·时间硬要求变化/解析异常）即 semantic_blocked）：
      ① 屏面健康 + （试跑 degenerate 或 语义被改）→ rejected=true + reject_reason
         （trial_degenerate / semantic_change）+ comparison_zh 对照（无 batch，屏面不动）；
      ② 屏面 degenerate + 试跑 healthy → 自动上屏（auto_shown=true，results.show
         路径不经 LLM 的机械直调；应用态丢失则 fail-closed 退回 advisory）；
         空屏面下语义被改**放行但如实标注**（disclosure_zh——可能的错误结果远大于
         什么结果都不给，标注义务不省）；
      ③ 其余 → 仅 advisory 对照（baseline/comparison_zh 如实给事实，语义被改时附
         警示，上屏与否归模型经 results.show 裁决；双方都健康时机械层只报事实不判
         「更好」）。
    返回 {original_query, rewritten_query, rewritten, applied, relaxed, total, filters,
    top digest, health, resolution_status, baseline, comparison_zh, rejected, auto_shown,
    reject_reason, semantic_blocked（+ 拒绝档 refused/refuse_reason/note_zh，
    + 放行标注 disclosure_zh，+ repair 纠错情报——None=未触发；applied="repair" 时
    为采纳档：terms/adopted_fixes/patched_query，未采纳档为候选 advisory）}。
    只读；query 槽空 → `_SearchRerunParamError`（bad_param，同纪律）。"""
    from ..app import workflow as wf

    slots = slots or {}
    ctx = ctx or {}
    query = str(slots.get("query") or "").strip()
    if not query:
        raise _SearchRerunParamError()
    sources = ctx.get("search_sources")
                                                  
                                                                 
                                            
    structured_kwargs = _loop_structured_kwargs(ctx)

                                                
                                                  
                                                                     
                                                         
                                                
    baseline = _screen_baseline(ctx)
    if (str(baseline.get("health") or "") in ("healthy", "unknown")
            and int(baseline.get("total") or 0) > _RERANK_TRIGGER_MAX_TOTAL):
        return {
            "original_query": query, "rewritten_query": query, "rewritten": False,
            "applied": "none", "relaxed": [], "probe_zh": "", "total": 0,
            "filters": [], "top": [], "displayed": False, "batch": None,
            "health": "", "resolution_status": "",
            "baseline": baseline, "comparison_zh": "", "rejected": False,
            "auto_shown": False, "reject_reason": "", "semantic_blocked": False,
            "refused": True, "refuse_reason": "screen_healthy",
            "note_zh": (f"当前结果区已有不错的结果（{int(baseline.get('total') or 0)} 条、"
                        "状态健康）——优化检索词重检只适用于结果差/零结果的场景，本次"
                        "没有执行。想换个措辞重搜请走 search.rerun；结果变差或零结果时"
                        "再用我不迟。"),
        }

    relax_keys = slots.get("relax")
    if isinstance(relax_keys, str):
        relax_keys = [relax_keys]
    relax_keys = [str(k).strip() for k in (relax_keys or []) if str(k).strip()]

    flow = wf.DatasetRecommendationWorkflow()
    relaxed: list[str] = []
    probe_zh = ""
    repair: dict | None = None
    effective_kwargs = structured_kwargs
    if relax_keys:
                                                                 
        intent, _c, _n, *_ = flow._prepare_context(
            query, sources=sources, auto_parse_sources=False, **structured_kwargs)
        relaxed = _relax_to_suppressed(intent, relax_keys)                        
        effective_kwargs = {
            **structured_kwargs,
            "suppressed_constraints": list(structured_kwargs["suppressed_constraints"]) + relaxed,
        }
        rewritten_query, rewritten, applied = query, False, "relax"
    else:
                                                          
                                                        
                                                    
                                   
        repair = _repair_route(flow, query, structured_kwargs)
        if repair and repair.get("adopted"):
            rewritten_query, rewritten, applied = repair["patched_query"], True, "repair"
        else:
                                                   
                                                                      
                                                                     
            probe_zh = _relaxation_probe_zh(flow, query, sources, structured_kwargs)
            health_zh = _rewrite_health_zh(**_latest_screen_facts(ctx), label="当前结果")
                                                        
                                                           
                                           
            utterance = str(ctx.get("utterance") or "").strip()
            anchor_zh = (f"用户原话（转述如有出入，语义以此为准，不许偏离它）：{utterance}"
                         if utterance and utterance != query else "")
                                                                
                                                             
            repair_zh = _rewrite_repair_zh(repair) if repair else ""
            rewritten_query, rewritten = _rewrite_query(
                ctx.get("chat_model"), query, usage_sink=ctx.get("usage_sink"),
                probe_zh=probe_zh, health_zh=health_zh, anchor_zh=anchor_zh,
                repair_zh=repair_zh)
            applied = "rewrite" if rewritten else "none"

    meta = flow.run_with_meta(
        wf.RecommendParams(
            query=rewritten_query, use_llm=False, sources=sources,
            auto_parse_sources=False, **effective_kwargs))
    health = _health_of_meta(meta)
    total = int(meta.result_total or 0)
    comparison_zh = _screen_comparison_zh(baseline, health, total, rewritten_query)
    if applied == "relax":
        comparison_zh += f"本次为定向放松重检（已放松：{'、'.join(relax_keys)}），条件放松部分未经原条件核验。"
    if repair:
                                                         
                                                  
                             
        comparison_zh += _repair_note_zh(repair, empty=(health == "degenerate"))
                                                 
                                                                
                                                              
                                        
    semantic_blocked = False
    reject_reason = ""
    diff_zh = ""
    if applied == "rewrite" and rewritten:
        _diff = _constraint_diff(flow, query, rewritten_query, structured_kwargs)
        semantic_blocked = bool(_diff.get("would_block"))
        if semantic_blocked:
            diff_zh = _constraint_diff_zh(_diff)
    rejected = False
    auto_shown = False
    batch = None
    displayed = False
    structured_lost = False
    rows: Any = meta.retrieved_data
    if str(baseline.get("health")) == "healthy" and (health == "degenerate" or semantic_blocked):
                                                    
                                          
        rejected = True
        reject_reason = "trial_degenerate" if health == "degenerate" else "semantic_change"
        if diff_zh:
            comparison_zh += diff_zh
    elif str(baseline.get("health")) == "degenerate" and health == "healthy":
                                                     
                                                                                
        payload, scope_kept = _verified_recommend_payload(meta, effective_kwargs)
        if scope_kept:
            rows = payload.get("results")
            displayed = True
            auto_shown = True
            batch = _build_show_batch(
                "rerank", rewritten_query,
                str(ctx.get("utterance") or "").strip() or query, payload)
        else:
            structured_lost = True
            health = "degenerate"
            comparison_zh = _screen_comparison_zh(baseline, health, total, rewritten_query)
    elif semantic_blocked and diff_zh:
                                                        
                               
        comparison_zh += "改写相对原句有约束语义变化（未经屏面核验不建议直接上屏）：" + diff_zh
    result = {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "rewritten": rewritten,
        "applied": applied,
        "relaxed": relaxed,
                                                     
                                                          
        "probe_zh": probe_zh,
                                                              
                                               
        "repair": ({k: v for k, v in repair.items() if k != "diff"}
                   if repair else None),
        "total": total,
                                                             
        "filters": list(meta.active_filters or []),
        "top": _top_digest(rows),
        "displayed": displayed,
        "batch": batch,
        "health": health,
        "resolution_status": str(getattr(meta, "resolution_status", "") or ""),
        "baseline": baseline,
        "comparison_zh": comparison_zh,
        "rejected": rejected,
        "auto_shown": auto_shown,
        "reject_reason": reject_reason,
        "semantic_blocked": semantic_blocked,
    }
    if auto_shown and semantic_blocked:
                                                  
                                             
        result["disclosure_zh"] = (
            "这次改写相对原查询有语义变化（"
            + "；".join(str(r)[:120] for r in (_diff.get("block_reasons") or []))
            + "）。因原结果区没有结果，仍按改写句上屏——请对照原查询核实是否符合你的意图。")
    if auto_shown and applied == "repair":
                                                        
                                          
        result["disclosure_zh"] = (
            _repair_note_zh(repair)
            + "因原结果区没有结果，已按纠正后的查询上屏——请核实纠正是否符合你的意图。")
    if structured_lost:
        result["structured_context_lost"] = True
        result["disclosure_zh"] = ("这次检索没能完整保留当前筛选条件，为避免放宽条件后"
                                   "误出结果，已保留原结果。")
        return result
    if health == "degenerate":
                                                                       
                                                                 
                                                                  
        digest = _relax_live_digest(flow, rewritten_query, sources, structured_kwargs)
        if digest is not None:
            result["relax_live"] = digest
    return result


def _loop_ask_relax(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """ask.relax（2026-09-14 YOLO 批）：「问用户放松哪个条件」——LLM 判定坏结果的
    病根在**用户条件本身**（互相打架/过严，而非措辞差）时，把「放松哪个条件、各能救回
    多少条」做成选择卡弹给用户（载荷经结果 `ask_relax` 键 → turn.py 组卷透传 →
    前端渲染；用户点选后前端以原查询 + 所选 suppressed_constraints 重搜，零 LLM，
    复用条件板既有通道）。

    机械闸（与 rerank 生死线同哲学——提示不是围栏，代码是）：只在屏面
    **degenerate**（`_screen_baseline`：无经条件核验的结果）时放行。屏面健康时
    问用户没有意义；unknown/watch 档系统已自动降级过、用户能看到结果，也不打扰——
    均如实拒绝（refuse_reason=screen_not_degenerate）。
    选项由 `retriever.relaxation_options` 确定性预演（零 LLM；include_dead=False——
    「放松也无救」的死维不给用户选）；每选项的 suppress 清单随预演携带（翻译真源
    `query_parser.relax_key_to_suppressed` 单锚点），此处只**与当前已生效的
    suppressed_constraints 合并去重**（前端整组替换语义，服务端合并好不丢既有忽略）。
    本工具不上屏（屏态唯一出口仍是 results.show）、不改查询、不落状态。
    拒绝档 ok 恒 True——拒绝是数据不是故障（同 rerank rejected 哲学）。

    环暂停语义（2026-09-14 ask 暂停/重启批，取代「之后请调 finish 收尾」旧口径）：
    asked=True 一落地，本批剩余续步被跳过、环直去 narrate 做确定性说明，**不再发起
    下一次 LLM 调用**；用户作答经 /api/utterance 的 `ask_answer` 字段以新回合重新入环
    （route_consensus 免投票直定 search、understand 零 LLM 强置首步 rank 按所选
    suppressed_constraints 重跑，decide 注入 resume_block 后由 LLM 裁决上屏/finish/
    再追问）——环能看到作答与重跑结果，不再是「弹卡后环外零 LLM 直落屏」。"""
    from ..app import workflow as wf

    slots = slots or {}
    ctx = ctx or {}
    reason = str(slots.get("reason") or "").strip()

    def _refuse(code: str, note: str, query: str = "") -> dict:
        return {"asked": False, "refuse_reason": code, "query": query,
                "reason": reason, "options": [], "ask_relax": None,
                "note_zh": note}

    query = _loop_effective_query(ctx)
    if not query:
        return _refuse("no_query", "没有可询问的原查询（环内还没有检索事实，"
                                 "现场也没有当前检索句）——放宽询问没有执行。")
    baseline = _screen_baseline(ctx)
    if str(baseline.get("health") or "") != "degenerate":
        return _refuse("screen_not_degenerate",
                       "屏面上已有可看的结果（未经核验的坏屏面才值得打扰用户）——"
                       "放宽询问没有执行；结果质量问题请走 rerank/results.show。",
                       query)
    structured_kwargs = _loop_structured_kwargs(ctx)
    flow = wf.DatasetRecommendationWorkflow()
    try:
        intent, _cands, normalized, *_ = flow._prepare_context(
            query, sources=ctx.get("search_sources"),
            auto_parse_sources=False, **structured_kwargs)
    except Exception as exc:                                   
        return _refuse("parse_error",
                       f"条件预演未能完成（{type(exc).__name__}）——放宽询问没有执行；"
                       "可改用 rerank 优化检索词重查。", query)
    if getattr(intent, "abstain", False):
        return _refuse("abstain",
                       "本查询未能完整解析出可执行条件，没有可放松的维度——"
                       "放宽询问没有执行；这种情况请改用 rerank 改写查询本身。", query)
    options = flow.retriever.relaxation_options(
        normalized, intent, top_k=3, max_options=6, include_dead=False)
    if not options:
        return _refuse("no_options",
                       "本查询没有放松后能救回结果的条件（各维试算均零新增）——"
                       "放宽询问没有执行；可改用 rerank 或如实告知用户后 finish。", query)
    card_options: list[dict] = []
    digest_options: list[dict] = []
    for o in options:
        key = str(o.get("key") or "")
        kind = str(o.get("kind") or "drop")
        label = str(o.get("label") or "")
        count = int(o.get("count") or 0)
        relaxed = [str(s) for s in (o.get("suppress") or [])]                          
        suppress = list(dict.fromkeys(
            list(structured_kwargs["suppressed_constraints"]) + relaxed))
                                                         
                                                          
        card_options.append({"key": key, "kind": kind, "label": label,
                             "count": count, "suppress": suppress})
        digest_options.append({"key": key, "kind": kind, "label": label,
                               "count": count})
    payload = {"query": query, "reason": reason, "options": card_options}
    return {"asked": True, "query": query, "reason": reason,
            "options": digest_options, "ask_relax": payload,
            "note_zh": (f"已向用户发起放宽询问（{len(card_options)} 个选项）——"
                        "环已暂停等作答，不要再调用任何工具：用户选完会按其选择自动"
                        "重跑检索并回到环内，届时再裁决上屏/继续改写/收尾。")}


                                                                            
 
                                                                         
                                                        
                                                             
                                                                 
                                                        
                                                                     
               

def _loop_structured_kwargs(ctx: dict) -> dict:
    """环内检索的**完整确定性入参**——与 search.rerun / pre-loop 摘要 / /api/recommend
    同口径（`wf.sanitize_*` 公共真源净化）：结构化现场（facet/suppressed/lenient/date）
    + 排序杠杆（top_k/recall/strategy，2026-09-17 三径合一——此前环内试跑静默回默认，
    请求级开向量召回时新旧屏面健康对照口径不一）。环内确定性段恒 rerank=off /
    use_llm=False / llm_available=False（与 pre-loop 同纪律：不付 LLM 重排）。
    ctx 缺键（旧会话/测试/harness 空 ctx）→ 全部回 workflow 默认，逐位不变。"""
    from ..app import workflow as wf

    kw = {
        "facet_filters": wf.sanitize_facet_filters(ctx.get("search_facet_filters")),
        "suppressed_constraints": wf.sanitize_suppressed(ctx.get("search_suppressed_constraints")),
        "lenient_dims": sorted(wf.sanitize_lenient_dims(ctx.get("search_lenient_dims"))),
        "date_from": str(ctx.get("search_date_from") or "").strip(),
        "date_to": str(ctx.get("search_date_to") or "").strip(),
        "llm_available": False,
    }
    top_k = ctx.get("search_top_k")
    try:
        top_k = int(top_k)
        if not 1 <= top_k <= 50:
            top_k = None
    except (TypeError, ValueError):
        top_k = None
    if top_k is not None:
        kw["top_k"] = top_k
    recall = str(ctx.get("search_recall") or "").strip().lower()
                                                                           
                                                         
    if recall not in ("off", "dense", "cross_encoder", "vector"):
        recall = "off"
    kw["recall_backend"] = recall
    if str(ctx.get("search_strategy") or "").strip().lower() == "auto":
        kw["strategy"] = "auto"
                                                                             
        from ..retrieval.vector_recall import resolve_auto_recall_preference
        preferred, available = resolve_auto_recall_preference()
        kw["preferred_recall"] = preferred
        kw["recall_available"] = available
    return kw


def _recent_effective_step(ctx: dict) -> tuple[dict | None, str]:
    """「最近一批检索结果」的**生效检索步**与其生效查询：最近一次成功的环内检索步
    （rank 的 query / rerank 的 rewritten_query / search.rerun 采纳档的 query——
    从 steps 实录现取，不采信自述）；无 → (None, "")。"""
    for step in reversed(list((ctx or {}).get("steps") or [])):
        if not step.get("ok"):
            continue
        result = step.get("result") or {}
        kind = step.get("card_kind")
        if kind == "rank":
            q = str(result.get("query") or "").strip()
        elif kind == "rerank":
            q = str(result.get("rewritten_query") or "").strip()
        elif kind == "search_rerun":
            q = str(result.get("query") or "").strip() if result.get("adopted") else ""
        else:
            continue
        if q:
            return step, q
    return None, ""


def _loop_effective_query(ctx: dict) -> str:
    """「最近一批检索结果」的生效查询；无 → 退回现场 current_query（调用方如实降级）。"""
    _step, q = _recent_effective_step(ctx)
    if q:
        return q
    return str((ctx or {}).get("current_query") or "").strip()


def _replay_structured_kwargs(ctx: dict, record: dict | None = None) -> dict:
    """重放/默认对象定位的**完整有效检索入参** = 结构化现场 + 该次试跑自报的生效放松项。

    rerank 放松档只在局部 `effective_kwargs` 追加 suppression；其后「导出当前结果 /
    对比前两条」若只继承 query（`_loop_structured_kwargs`）会恢复刚放松的条件，
    重跑出空集/异集，批次范围对不上。`record` 精确给出试跑记录（results.show）；
    缺省取最近一次生效检索步的实录（默认对象定位/重放同源）。"""
    kw = _loop_structured_kwargs(ctx)
    res = record.get("result") if isinstance(record, dict) else None
    if not isinstance(res, dict):
        step, _q = _recent_effective_step(ctx)
        res = ((step or {}).get("result") or {})
    relaxed = [str(t).strip() for t in (res.get("relaxed") or []) if str(t).strip()]
    if relaxed:
        base = list(kw.get("suppressed_constraints") or [])
        for term in relaxed:
            if term not in base:
                base.append(term)
        kw["suppressed_constraints"] = base
    return kw


def _recent_result_records(ctx: dict, root: Path) -> tuple[list | None, Any]:
    """当前结果集的**序列化候选 dict**（重跑标准规则管线，`use_llm=False`，与 rank 同口径）；
    无生效查询 → (None, None)——调用方如实降级。

    **注意**：管线产出的 dict 只是展示形状（`_serialize_retrieved_data`），**不是**
    `DatasetRecord`——需要记录本体（build_item / find_compatible / locate）时一律从
    `_full_corpus` 取，本函数只提供**默认对象的 uid 清单**（`_recent_result_uids`）。"""
    query = _loop_effective_query(ctx)
    if not query:
        return None, None
    from ..app import workflow as wf

    flow = wf.DatasetRecommendationWorkflow()
    meta = flow.run_with_meta(
        wf.RecommendParams(
            query=query, use_llm=False, sources=(ctx or {}).get("search_sources"),
            auto_parse_sources=False,
            **_replay_structured_kwargs(ctx)))
    return list(meta.retrieved_data or []), meta


def _recent_result_uids(ctx: dict, root: Path) -> list[str] | None:
    """最近一批检索结果的 dataset_uid 清单（保序）。无生效查询 → None（不是 []——
    调用方要区分「没查过」与「查了但没有」，降级句两说）。"""
    records, _ = _recent_result_records(ctx, root)
    if records is None:
        return None
    return _result_uids(records)


def _full_corpus(root: Path) -> list:
    """全量语料（base + 外部库）——与 /api/fair、/api/compatible 同口径的定位语料。"""
    from ..corpus.corpus import load_full_corpus

    return load_full_corpus(root / "database" / "base", root)


def _result_uids(records: Any) -> list[str]:
    """记录列表 → 保序 dataset_uid 清单（去空）。**dict 与 DatasetRecord 通吃**——
    管线序列化 dict（`r.get("dataset_uid")`）与语料记录（`r.raw["dataset_uid"]`）两形态
    都可能是输入（前者来自 `_recent_result_records`，后者来自 `_full_corpus`）。"""
    out: list[str] = []
    for r in list(records or []):
        if isinstance(r, dict):
            uid = str(r.get("dataset_uid") or "").strip()
        else:
            raw = r.raw if isinstance(r.raw, dict) else {}
            uid = str(raw.get("dataset_uid") or "").strip()
        if uid and uid not in out:
            out.append(uid)
    return out


def _record_by_uid(records: Any, uid: str) -> Any:
    """按 dataset_uid 取记录（保序列表线性扫；None = 找不到）。"""
    for r in list(records or []):
        raw = r.raw if isinstance(r.raw, dict) else {}
        if str(raw.get("dataset_uid") or "") == uid:
            return r
    return None


def _locate_loop_dataset(records: Any, ident: str) -> tuple[Any, list[dict]]:
    """编号/名称/链接 → 单条记录（`corpus.locate_record` 单一真源：uid 精确 > url 精确 >
    name 精确+source 消歧）。返回 (record, candidates)；record=None 且 candidates 非空 =
    歧义（如实报歧义，绝不静默任取）。"""
    from ..corpus.corpus import locate_record

    ident = str(ident or "").strip()
    if not ident:
        return None, []
    return locate_record(list(records or []), uid=ident, url=ident, name=ident)


def _item_digest(item: dict) -> dict[str, str]:
    """条目 digest（结果契约的 a/b/seed 用）：uid + 名称 + 来源。"""
    return {
        "dataset_uid": str(item.get("dataset_uid") or ""),
        "dataset_name": str(item.get("dataset_name") or ""),
        "source": str(item.get("source") or ""),
    }


                                                        
                                       
_COMPARE_PROMPT_FALLBACK_ZH = (
    "你是生物数据检索工具的数据集对比撰稿人。把给定的两个数据集字段差异 JSON 翻译成一段"
    "中文对比结论：先总述相同/不同字段数，再挑 2-4 个不同字段各说一句；只允许措辞加工，"
    "不得新增/修改/推断任何数字或事实；缺失字段如实写「未标注」；不评价哪个更好；"
    "不超过 400 字。"
)


def _compare_wording_prompt() -> str:
    """compare 措辞器的系统提示词（`prompts/compare.md`，文件即真源）——懒加载缓存。"""
    return _prompt_md("compare.md", _COMPARE_PROMPT_FALLBACK_ZH)


def _render_compare_with_llm(model: Any, diff: dict, name_a: str, name_b: str,
                             usage_sink: list | None = None) -> str | None:
    """compare.datasets 的**独立 LLM 措辞调用**（独立上下文 + 独立系统提示词，与 rerank
    改写同纪律）+ 机械健全性检查：非空、≤`compare.MAX_WORDING_CHARS` 字符、不引入
    diff 事实之外的新数字（`compare.introduces_foreign_numbers` 交叉核验——数字和事实
    必须来自确定性 diff）。任一不过（含模型缺席/调用异常）→ None，调用方退回确定性
    拼接（`compare.render_deterministic`），`wording_source` 如实标注。"""
    from . import compare as _cmp

    if model is None:
        return None
    t0 = time.monotonic()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        answer = model.invoke([
            SystemMessage(content=_compare_wording_prompt()),
            HumanMessage(content=_cmp.build_prompt(diff, name_a, name_b)),
        ])
        rec = _usage_record(answer, "compare_wording")
        if rec is not None and usage_sink is not None:
            usage_sink.append(rec)
        content = getattr(answer, "content", "")
        if isinstance(content, list):
            content = " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
        candidate = str(content or "").strip()
        _te.emit_llm_call(
            node="compare_wording",
            model=str(getattr(model, "model_name", "") or getattr(model, "model", "") or ""),
            prompt=_cmp.build_prompt(diff, name_a, name_b), response=candidate,
            ms=int((time.monotonic() - t0) * 1000), channel="text")
    except Exception as exc:
        _te.emit_llm_call(
            node="compare_wording",
            model=str(getattr(model, "model_name", "") or getattr(model, "model", "") or ""),
            prompt=_cmp.build_prompt(diff, name_a, name_b), response="",
            ms=int((time.monotonic() - t0) * 1000), channel="text",
            fallback_reason=type(exc).__name__)
        return None
    if not candidate or len(candidate) > _cmp.MAX_WORDING_CHARS:
        return None
    if _cmp.introduces_foreign_numbers(candidate, diff, name_a, name_b):
        return None
    return candidate


def _compare_degraded(reason: str, note: str) -> dict:
    """compare.datasets 的降级出口（形状与成功档同一契约——降级是数据不是故障）。"""
    return {
        "a": {}, "b": {}, "assumption_zh": "", "fields": [], "n_same": 0, "n_diff": 0,
        "n_unknown": 0, "identical": False, "comparison_zh": note,
        "wording_source": "deterministic", "degraded": True, "degrade_reason": reason,
        "caveat_zh": "",
    }


def _loop_compare_datasets(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """对比两个数据集：确定性字段 diff（`compare.diff_items`，
    零 LLM，事实层）+ 一次独立 LLM 措辞调用（数字交叉核验）→ 中文对比结论。
    槽位 a/b 可选（编号/名称）；没点名 → 当前结果前两条（结论里说明这个假设）。
    记录本体一律走 `_full_corpus`（DatasetRecord）——管线序列化 dict 只提供默认 uid。
    降级路径（如实）：无结果 / 指定对象找不到 / 歧义 / a==b 只有一条可比 / 字段全同——
    都产出诚实的句子，不假装对比成功、不编造差异。"""
    from . import compare as _cmp
    from ..content.item_view import build_item

    slots = slots or {}
    ctx = ctx or {}
    a_ident = str(slots.get("a") or "").strip()
    b_ident = str(slots.get("b") or "").strip()
    default_uids = _recent_result_uids(ctx, root)
    if default_uids is None and not (a_ident or b_ident):
        return _compare_degraded(
            "no_results",
            "当前没有可对比的检索结果（也没有指定数据集编号/名称），无法对比。")
    records_all = _full_corpus(root)
    if default_uids is None:
        default_uids = []

    def _pick(pos: int, ident: str, exclude_uid: str = "") -> tuple[Any, str | None]:
        """第 pos 个对比对象：点名的先定位（locate 单一真源），没点名取当前结果第 pos 条
        （与已定对象去重）。返回 (record, degrade_reason or None)。"""
        if ident:
            record, candidates = _locate_loop_dataset(records_all, ident)
            if record is not None:
                return record, None
            return None, ("ambiguous" if candidates else "not_found")
        if pos == 0 and default_uids:
            return _record_by_uid(records_all, default_uids[0]), None
        if pos == 1 and default_uids:
            for uid in default_uids[1:]:
                if uid != exclude_uid:
                    return _record_by_uid(records_all, uid), None
        return None, ("no_results" if not default_uids else "only_one_result")

    record_a, degrade_a = _pick(0, a_ident)
    if degrade_a is not None:
        note = _compare_ident_note(a_ident or "当前结果第一条", degrade_a)
        return _compare_degraded(f"{degrade_a}_a", note)
    item_a = build_item(record_a)
    uid_a = str(item_a.get("dataset_uid") or "")
    record_b, degrade_b = _pick(1, b_ident, exclude_uid=uid_a)
    if degrade_b is not None:
        note = _compare_ident_note(b_ident or "当前结果第二条", degrade_b)
        return _compare_degraded(f"{degrade_b}_b", note)
    item_b = build_item(record_b)
    uid_b = str(item_b.get("dataset_uid") or "")
    name_a = str(item_a.get("dataset_name") or a_ident or "数据集A")
    name_b = str(item_b.get("dataset_name") or b_ident or "数据集B")
    if uid_a and uid_b and uid_a == uid_b:
        return _compare_degraded(
            "same_dataset",
            f"「{name_a}」与「{name_b}」是同一个数据集（{uid_a}），无需对比。")

    diff = _cmp.diff_items(item_a, item_b)
    comparison = _render_compare_with_llm(ctx.get("chat_model"), diff, name_a, name_b,
                                          usage_sink=ctx.get("usage_sink"))
    wording_source = "llm" if comparison is not None else "deterministic"
    if comparison is None:
        comparison = _cmp.render_deterministic(diff, name_a, name_b)
    if not a_ident and not b_ident:
        assumption_zh = "未指定对比对象，默认取当前结果的前两条进行对比。"
    elif not a_ident:
        assumption_zh = "未指定第一个对比对象，默认取当前结果第一条。"
    elif not b_ident:
        assumption_zh = "未指定第二个对比对象，默认取当前结果第二条。"
    else:
        assumption_zh = ""
    return {
        "a": _item_digest(item_a), "b": _item_digest(item_b),
        "assumption_zh": assumption_zh,
        "fields": diff["fields"], "n_same": diff["n_same"], "n_diff": diff["n_diff"],
        "n_unknown": diff["n_unknown"], "identical": diff["identical"],
        "comparison_zh": comparison, "wording_source": wording_source,
        "degraded": False, "degrade_reason": "",
        "caveat_zh": "对比只覆盖本目录收录的元数据字段；缺失字段按「未标注」处理，"
                     "不代表数据集没有。",
    }


def _compare_ident_note(shown: str, reason: str) -> str:
    """compare 对象定位失败的如实句（降级出口的 comparison_zh）。"""
    if reason == "ambiguous":
        return f"「{shown}」命中多条同名数据集，无法确定是哪一个；请改用 dataset_uid 精确指定。"
    if reason == "not_found":
        return f"「{shown}」在当前库中找不到，无法参与对比。"
    if reason == "no_results":
        return "当前没有可用的检索结果，取不到默认对比对象（也请确认指定了编号/名称）。"
    if reason == "only_one_result":
        return "当前结果只有一条可比数据集，无法对比（对比需要两个不同的数据集）。"
    return f"{shown}不可用，无法对比。"


def _sanitize_uids_slot(value: Any, limit: int = 20) -> list[str]:
    """cite.export 的 uids 槽清洗（2026-08-20 批）：列表 / 单字符串 →
    去空去重保序的 uid 清单，上限 20。占位引用已由 execute 解析层替换为真实 uid 才到
    这里（本函数不碰占位——解析失败在 execute 已跳过）。"""
    if isinstance(value, (list, tuple)):
        raw_items = [str(x) for x in value]
    elif isinstance(value, str) and value.strip():
        raw_items = [value]
    else:
        return []
    items: list[str] = []
    for raw in raw_items:
        s = str(raw).strip()
        if s and s not in items:
            items.append(s)
    return items[:limit]


def _loop_cite_export(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """导出当前结果（或指定条数 / 指定编号清单）的 **RIS + BibTeX 双格式**引文并落盘
    （2026-08-18；2026-08-20 批扩 uids 槽）：产物落
    `.userdata/citations/`（本机运行产物目录，gitignored），回执带完整路径与字节数——
    前端 runner 旧路径只下 .ris、把响应里的 bibtex 丢掉的缺口在此补上（两种格式都落盘，
    用户都拿得到）。write 语义（readonly=False）由 trace 快照锚定、可被 curate.rollback
    的机械闸看到；文件不落 database/、不碰基准。
    `uids` 槽（可选，≤20）：提供时**按 UID 列表导出**（真实消费——用户点名/依赖占位
    解析后的编号清单；数组元素可为真实编号或同批前序检索结果的占位引用，占位已由
    execute 解析层替换为真实 uid 才到这里）；未提供时保持现状（当前结果集 + limit）。"""
    from ..content.reuse_pack import build_pack_for_uids, to_bibtex, to_ris

    slots = slots or {}
    ctx = ctx or {}
    limit = slots.get("limit")
    explicit_uids = _sanitize_uids_slot(slots.get("uids"))
    if explicit_uids:
                                                     
        uids = explicit_uids
    else:
        default_uids = _recent_result_uids(ctx, root)
        uids = list(default_uids or [])
        if not uids:
            return {
                "n_datasets": 0, "uids": [], "files": [], "out_dir": "",
                "note_zh": "当前没有可导出的检索结果，没有生成引文文件。",
            }
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            uids = uids[:limit]
                                                                   
                                
    pack = build_pack_for_uids(uids, _full_corpus(root))
    ris = to_ris(pack)
    bibtex = to_bibtex(pack)
    if not ris.strip() and not bibtex.strip():
        return {
            "n_datasets": 0, "uids": [], "files": [], "out_dir": "",
            "note_zh": "这些数据集在当前库中都解析不到，没有可导出的引文。",
        }
    from ..app.runtime_paths import account_data_dir_for
    from ..corpus.patch_package import current_patch_scope

    out_dir = account_data_dir_for(Path(root), ".userdata/citations", current_patch_scope())
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    artifact_id = uuid.uuid4().hex
    ris_name = f"reused-public-datasets-{ts}-{artifact_id}.ris"
    bib_name = f"reused-public-datasets-{ts}-{artifact_id}.bib"
    ris_path = out_dir / ris_name
    bib_path = out_dir / bib_name
    ris_path.write_text(ris, encoding="utf-8")
    bib_path.write_text(bibtex, encoding="utf-8")
                                                              
                                                     
    ris_bytes = ris_path.stat().st_size
    bib_bytes = bib_path.stat().st_size
    unresolved = list(pack.get("unresolved") or [])
    note = (f"已导出 {len(uids)} 个数据集的引文，RIS 与 BibTeX 两种格式都已落盘："
            f"{out_dir} 下的 {ris_name}、{bib_name}。")
    if unresolved:
        note += (f"其中 {len(unresolved)} 个编号在当前库中找不到，未纳入引文"
                 f"（{('、'.join(unresolved[:3]))}{'…' if len(unresolved) > 3 else ''}）。")
    return {
        "n_datasets": len(uids), "uids": uids,
        "files": [
            {"filename": ris_name, "format": "ris", "bytes": int(ris_bytes)},
            {"filename": bib_name, "format": "bibtex", "bytes": int(bib_bytes)},
        ],
        "out_dir": str(out_dir),
        "note_zh": note,
    }


def _loop_compat_find(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """给一个数据集找**元数据兼容**同伴：包
    `compatibility.find_compatible`（同物种 且 chemistry 或平台相同——可整合的必要非充分
    条件，caveat 恒带）。种子 = uid 槽点名（编号/名称/链接，locate 单一真源）或当前结果
    第一条；找不到/无结果 → 如实降级，不硬找。"""
    from ..content import compatibility as _compat

    slots = slots or {}
    ctx = ctx or {}
    ident = str(slots.get("uid") or "").strip()
    records_all = _full_corpus(root)
    seed_uid = ""
    assumed = False
    degrade = ""
    if ident:
        record, candidates = _locate_loop_dataset(records_all, ident)
        if record is None:
            degrade = "ambiguous" if candidates else "not_found"
        else:
            raw = record.raw if isinstance(record.raw, dict) else {}
            seed_uid = str(raw.get("dataset_uid") or "")
    else:
        default_uids = _recent_result_uids(ctx, root)
        if default_uids is None or not default_uids:
            degrade = "no_results"
        else:
            record = _record_by_uid(records_all, default_uids[0])
            if record is None:
                degrade = "not_found"
            else:
                raw = record.raw if isinstance(record.raw, dict) else {}
                seed_uid = str(raw.get("dataset_uid") or "")
                assumed = True
    if degrade or not seed_uid:
        note = _compat_note_zh("", 0, assumed, degrade, ident)
        return {
            "seed": {}, "criteria": {}, "total": 0, "compatible": [],
            "caveat": _compat.CAVEAT_ZH, "note_zh": note,
            "degraded": True, "degrade_reason": degrade or "not_found",
        }
    result = _compat.find_compatible(seed_uid, records_all, limit=20)
    if result is None:
                                                      
        return {
            "seed": {}, "criteria": {}, "total": 0, "compatible": [],
            "caveat": _compat.CAVEAT_ZH,
            "note_zh": f"「{ident or seed_uid}」无法定位到可用记录，没有找兼容同伴。",
            "degraded": True, "degrade_reason": "not_found",
        }
    note = _compat_note_zh(str(result["seed"].get("dataset_name") or ""),
                           result["total"], assumed, None, ident)
    return {
        "seed": result["seed"], "criteria": result["criteria"],
        "total": result["total"], "compatible": result["compatible"],
        "caveat": result["caveat"], "note_zh": note,
        "degraded": False, "degrade_reason": "",
    }


def _compat_note_zh(seed_name: str, total: int, assumed: bool,
                    degrade: str | None, ident: str) -> str:
    """compat.find 的人读句：成功档（种子名 + 兼容数 + 缺省说明）与降级档（如实原因）。"""
    if degrade == "ambiguous":
        return f"「{ident}」命中多条同名数据集，无法确定是哪一个；请改用 dataset_uid 精确指定。"
    if degrade == "not_found":
        return f"「{ident}」在当前库中找不到，没有可找兼容同伴的对象。"
    if degrade == "no_results":
        return "当前没有可用的检索结果（也没有指定数据集编号），没有可找兼容同伴的对象。"
    prefix = "（未指定数据集，默认取当前结果第一条）" if assumed else ""
    return (f"已按「{seed_name}」的元数据找到 {total} 个兼容数据集"
            f"（共享物种，且 chemistry 或平台相同）。{prefix}").strip()


def _loop_fair_check(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """对单个数据集做 13 项 FAIR 元数据自检：包
    `fair.build_fair_report`。**边界纪律**：衡量的是「这份公开元数据够不够引用/写方法学」
    ——复用者视角就绪度，**不是**官方 FAIR 认证，也不是对数据质量的评价；note_zh 与
    工具描述都不越过这个定义。对象 = uid 槽点名或当前结果第一条。"""
    from ..content.item_view import build_item
    from ..retrieval.fair import build_fair_report

    slots = slots or {}
    ctx = ctx or {}
    ident = str(slots.get("uid") or "").strip()
    record = None
    assumed = False
    degrade = ""
    records_all = _full_corpus(root)
    if ident:
        record, candidates = _locate_loop_dataset(records_all, ident)
        if record is None:
            degrade = "ambiguous" if candidates else "not_found"
    else:
        default_uids = _recent_result_uids(ctx, root)
        if default_uids is None or not default_uids:
            degrade = "no_results"
        else:
            record = _record_by_uid(records_all, default_uids[0])
            if record is None:
                degrade = "not_found"
            else:
                assumed = True
    if degrade or record is None:
        return {
            "dataset_name": "", "source": "", "fair": {}, "data_availability": {},
            "note_zh": _fair_note_zh("", "", None, degrade, ident),
            "degraded": True, "degrade_reason": degrade or "not_found",
        }
    item = build_item(record, include_introduction=True)
    report = build_fair_report(item)
    summary = (report.get("fair") or {}).get("summary") or {}
    note = _fair_note_zh(str(report.get("dataset_name") or ""),
                         str(summary.get("readiness_pct") or ""), summary,
                         None, "", assumed=assumed)
    return {
        "dataset_name": str(report.get("dataset_name") or ""),
        "source": str(report.get("source") or ""),
        "fair": report.get("fair") or {},
        "data_availability": report.get("data_availability") or {},
        "note_zh": note, "degraded": False, "degrade_reason": "",
    }


def _fair_note_zh(name: str, readiness: str, summary: dict | None,
                  degrade: str | None, ident: str, *, assumed: bool = False) -> str:
    """fair.check 的人读句：成功档（就绪度 + 边界句 + 缺省说明）与降级档（如实原因）。"""
    if degrade == "ambiguous":
        return f"「{ident}」命中多条同名数据集，无法确定是哪一个；请改用 dataset_uid 精确指定。"
    if degrade == "not_found":
        return f"「{ident}」在当前库中找不到，无法做 FAIR 自检。"
    if degrade == "no_results":
        return "当前没有可用的检索结果（也没有指定数据集编号），无法做 FAIR 自检。"
    prefix = "（未指定数据集，默认取当前结果第一条）" if assumed else ""
    if summary is None:
        p = u = pct = 0
        s = 0
    else:
        s = int(summary.get("pass") or 0)
        p = int(summary.get("partial") or 0)
        u = int(summary.get("unknown") or 0)
        pct = str(summary.get("readiness_pct") or "")
    return (f"「{name}」的 FAIR 复用就绪度：{pct}%（{s} 项充分 / {p} 项部分 / "
            f"{u} 项未知）——这是复用者视角的就绪度自检，不是官方 FAIR 认证，"
            f"也不是对数据质量的评价。{prefix}").strip()


LOOP_TOOLS: dict[str, dict[str, Any]] = {
    "curate.db_status": {
        "run": _loop_db_status, "label_zh": _ap.VERB_BY_NAME["curate.db_status"].zh, "card_kind": "db_status",
        "readonly": True, "report": True, "observation": True,
                                                 
        "decide_zh": ("汇报数据库当前状态（各来源条数、外部库与回收站、近期变动）：只读，无槽位。"
                      "「看看库里多少条 / 现在有什么」是一件**独立的事**，要单独一步做——"
                      "不许并入检查或搜索步里顺带回答"),
    },
    "curate.check_updates": {
        "run": _loop_check_updates, "label_zh": _ap.VERB_BY_NAME["curate.check_updates"].zh, "card_kind": "check_updates",
        "readonly": True,
                                                  
                                         
                                                                                
        "decide_zh": ("检查来源更新：只读在线比对某个来源有没有新数据（不搜关键词、不入库）；"
                      "一个调用只查一个来源——原话点名多个来源时，可为每个来源各发一个本工具调用（它们彼此独立）；"
                      "槽位 source，不填查全部。问某个**主题**在网上有没有数据不是本工具——"
                      "那是 curate.search_online；「检查更新，有的话直接入库」（不限定主题）"
                      "不是本工具——那是 curate.sync_updates（检查+入库一步做完）"),
    },
    "curate.search_online": {
        "run": _loop_search_online, "label_zh": _ap.VERB_BY_NAME["curate.search_online"].zh, "card_kind": "search_online",
        "readonly": False,
                                                      
                                               
        "needs_context": True,
                                                    
                                               
                                                                        
                                                                      
                                   
        "decide_zh": ("联网搜官方源并入库；槽位 keywords 必填——从原话提取**主题词**"
                      "（疾病/组织/物种/技术等），**优先英文**（联网源都是英文源，"
                      "「人类肺」→ human lung），source / species 可选；来源名不是主题词，"
                      "**不许拿来源名当 keywords**（来源名放 source 槽）。"
                      f"本地库已收录来源：{_source_candidates_zh('curate.sync_updates')}——"
                      "名单内来源的「检查更新/有没有新发布/更新入库」走 curate.check_updates"
                      "（只查）/ curate.sync_updates（检查+入库一步完成），**不要联网搜**；"
                      "本工具只按主题词找新数据集。"
                      "检查更新发现疑似新增、且用户说了「若有 X 数据就下载/入库」时，条件成立后"
                      "下一步就是本工具（keywords 逐字取疑似新增条目标题）——不许拿「新增里"
                      "已经有了」提前替系统放弃"),
    },
    "curate.sync_updates": {
        "run": _loop_sync_updates, "label_zh": _ap.VERB_BY_NAME["curate.sync_updates"].zh, "card_kind": "sync_updates",
        "readonly": False,
                                                 
        "needs_context": True,
                                              
                                            
                                                                  
        "decide_zh": ("检查更新并把能自动入库的疑似新增直接入库（先比对后入库的复合流，"
                      "一步做完，不要拆成 check_updates + search_online 两步）；"
                      "**原话限定了主题的下载不选我**——选 search_online（我不过滤主题，"
                      "会把所有疑似新增都入库）；例：「检查有没有更新，有的话都下载下来」选我，"
                      "「联网搜 human lung 数据入库」含主题词——那是 search_online，不是我；"
                      "槽位 source 可选；本工具做完即闭环——检查更新+入库一步完成，"
                      "不要再叠加 search_online 或 check_updates"),
    },
                                                                 
                                                                
                                           
                                                                           
    "search.rerun": {
        "run": _loop_search_rerun, "label_zh": _ap.VERB_BY_NAME["search.rerun"].zh, "card_kind": "search_rerun",
        "readonly": True, "needs_context": True,
        "decide_zh": ("换一组查询词把本地库重新检索一遍（只读语义：不改库，跑的是与主检索同一条"
                      "管线；结果集没变会如实拒绝并保留当前结果；命中 0 条也如实采纳上屏——"
                      "空结果集就是新条件的真实答案）。槽位 query 必填。"
                      "只在三种情况提议：① 当前零命中或结果明显跑偏；② 用户要换方向重查；"
                      "③ 动作链中途需要另查一批数据做判断。同一查询不许重复提议（机械闸）。"),
    },
}

                                       
                                                                    
                                                                    
                                                        
                                                                     
                                                        
LOOP_TOOLS["rank"] = {
    "run": _loop_rank, "label_zh": _ap.VERB_BY_NAME["rank"].zh, "card_kind": "rank",
    "readonly": True, "needs_context": True,
    "decide_zh": ("用一条检索句在本地库做**新检索**（只读，跑与主检索同一条管线，如实回报"
                  "命中总数/生效条件/健康度/top 条目——top 每条含 dataset_uid 与 rank 序号，"
                  "后续对比/FAIR/兼容/引文可引用）。槽位 query 必填（完整检索句）。"
                  "结果健康（有命中且条件核验通过）时会**自动上屏**，无需额外动作；"
                  "不健康时会附健康度标注与告警，按告警给出的对策处理。"
                  "换词重检已有查询是 search.rerun，坏 query 先优化再查是 rerank——"
                  "裸新检索才选我。"),
}
LOOP_TOOLS["rerank"] = {
    "run": _loop_rerank, "label_zh": _ap.VERB_BY_NAME["rerank"].zh, "card_kind": "rerank",
    "readonly": True, "needs_context": True,
    "decide_zh": ("当前检索句**质量差**（太口语化、实体写法不规范、中英错位）或**条件过严"
                  "零结果**时重查（只读）。**收到「结果不健康」告警时本工具是对策之一**。"
                  "执行前系统先给出当前条件的松弛试算报告（确定性预演，仅建议）——"
                  "两通道任选：①照报告建议放松：relax 槽填报告里的 key（可多选），"
                  "零改写确定性重检原句；②不给 relax：独立改写器优化检索句再重查"
                  "（改写不过机械健全性检查会如实退回原句、rewritten=false）。"
                  "槽位 query 必填（填**原始**的差 query，不要自己先改）、reason 可选。"
                  "执行后只**试跑**、不直接上屏：系统给出新旧结果的健康度对照——"
                  "试跑明显更好时用 results.show 上屏；试跑会把健康结果搞没时机械驳回"
                  "（rejected=true），当前结果区不动。"
                  "两道机械闸（2026-09-18 起）：①当前结果区健康且充足（>3 条）时本工具"
                  "直接拒绝执行（refused=true, refuse_reason=screen_healthy）——好结果"
                  "不需要重查，换措辞请走 search.rerun；②改写若引入未授权的约束语义变化"
                  "（硬约束增删/排除变化/软硬偏好互转/极性翻转/原始数据或时间硬要求变化），"
                  "健康屏面上机械驳回（rejected=true, reject_reason=semantic_change），"
                  "空屏面上放行但如实标注（disclosure_zh）。"
                  "top 条目同 rank 含 dataset_uid 与 "
                  "rank 序号，可被后续结果处理引用。"),
}
LOOP_TOOLS["results.show"] = {
    "run": _loop_show, "label_zh": _ap.VERB_BY_NAME["results.show"].zh,
    "card_kind": "results_show",
    "readonly": True, "needs_context": True,
    "decide_zh": ("把环内一次检索/重查的**试跑结果**放上结果区——屏面唯一出口（只读，"
                  "重放与试跑同一条确定性管线重建载荷 + 指纹复核，现场已变会如实拒绝）。"
                  "槽位 handle 可选：latest（缺省 = 最近一次试跑）/ latest_rank / "
                  "latest_rerank / 步号。仅当试跑结果**明显好于**当前屏面时才调用；"
                  "屏面已是该结果时本工具幂等，不要重复调用。"),
}
                                                        
                                                       
                                                        
LOOP_TOOLS["ask.relax"] = {
    "run": _loop_ask_relax, "label_zh": _ap.VERB_BY_NAME["ask.relax"].zh,
    "card_kind": "ask_relax",
    "readonly": True, "needs_context": True,
    "decide_zh": ("屏面**没有经条件核验的结果**、且病根在**用户条件互相打架或过严**"
                  "（如同时限定矛盾条件、年份过严）时，弹一张「放松哪个条件」选择卡问用户"
                  "（只读，不改查询、不上屏）。选项与各选项能救回多少条由系统确定性预演，"
                  "用户点选后自动按所选放松条件重搜。槽位 reason 可选（为什么判定条件"
                  "不合理，一句中文，会展示给用户）。**收到「结果不健康」告警且原因在"
                  "条件本身时本工具是对策之一**；屏面健康、或问题在措辞/实体写法/"
                  "中英错位时不选我（那是 rerank）；解析弃权时也别选我（没有可放松的"
                  "条件，会被拒绝）。每轮至多 1 次；调用后请调 finish 并说明正在等"
                  "用户作答。"),
}

                                                                            
                                                                            
                                                          
                                                                 
                                                                              
LOOP_TOOLS["compare.datasets"] = {
    "run": _loop_compare_datasets, "label_zh": _ap.VERB_BY_NAME["compare.datasets"].zh, "card_kind": "compare",
    "readonly": True, "needs_context": True,
    "decide_zh": ("把两个数据集放在一起做**元数据字段**对比（名称/来源/物种/组织/疾病/"
                  "平台/技术/chemistry/模态/样本量/发表时间/文件数）：对比前两条、比较这两个"
                  "数据集、它们有什么不同。槽位 a/b 填用户点名的数据集编号或名称（原话点名了"
                  "才填；没点名就不填 = 当前结果前两条，结论里会说明这个假设）。字段差异由"
                  "系统确定性比对（数字与事实以此为准），结论措辞由独立改写器生成；"
                  "不评价哪个数据集更好。「换一批查询词重新检索」不是本工具——那是 "
                  "search.rerun / rank。"),
}
LOOP_TOOLS["cite.export"] = {
    "run": _loop_cite_export, "label_zh": _ap.VERB_BY_NAME["cite.export"].zh, "card_kind": "cite_export",
    "readonly": False, "needs_context": True,
    "decide_zh": ("把当前结果（或指定条数 / 指定编号清单）生成 **RIS + BibTeX 两种格式**"
                  "的引文文件落盘，并回执文件路径与字节数（limit 用户说了条数才填；"
                  "uids 用户点名了编号/要引用同批前序检索结果时才填，没填 = 当前结果第 1 条）。"
                  "导出的**引文文本**（数据集条目 TY-DATA / @misc，不是论文条目），"
                  "不含数据集文件本身——「打包下载数据集文件」不是本工具（那是前端打包 "
                  "pack.download，本环不做）。"),
}
LOOP_TOOLS["compat.find"] = {
    "run": _loop_compat_find, "label_zh": _ap.VERB_BY_NAME["compat.find"].zh, "card_kind": "compat_find",
    "readonly": True, "needs_context": True,
    "decide_zh": ("给一个数据集找**元数据上兼容**的其它数据集（同物种 且 chemistry 或平台"
                  "相同——可整合的**必要非充分**条件，结论恒带诚实边界句，绝不说「可整合」）。"
                  "槽位 uid 填用户点名的数据集编号或名称（原话点名了才填；没点名就不填 = "
                  "当前结果第一条）。「换一批查询词重新检索找数据」不是本工具——那是 "
                  "search.rerun / rank；本工具按兼容判据找同伴，不重跑检索。"),
}
LOOP_TOOLS["fair.check"] = {
    "run": _loop_fair_check, "label_zh": _ap.VERB_BY_NAME["fair.check"].zh, "card_kind": "fair_check",
    "readonly": True, "needs_context": True,
    "decide_zh": ("对单个数据集做 **13 项 FAIR 元数据自检**（Findable/Accessible/"
                  "Interoperable/Reusable，每项 pass/partial/unknown + 改进建议 + 投稿"
                  "数据可用性声明）。衡量「这份公开元数据够不够引用/写方法学」——"
                  "复用者视角的就绪度，**不是**官方 FAIR 认证，也不是对数据质量的评价。"
                  "槽位 uid 填用户点名的数据集编号或名称（原话点名了才填；没点名就不填 = "
                  "当前结果第一条）。"),
}


                                                                                

                                                    
_CONSENSUS_ROUTES: tuple[str, ...] = ("search", "action", "general")
_SCOPED_ROUTES: tuple[str, ...] = (*_CONSENSUS_ROUTES, "rescue", "read")
                                           
_ROUTE_REQUEST_TARGETS: tuple[str, ...] = _CONSENSUS_ROUTES


class _RouteRequestParamError(Exception):
    """route.request 的 target_route 槽为空或非法。execute 的错误提取读 hint/code 属性
    （与 `_SearchRerunParamError` 同约）。"""

    code = "bad_param"
    hint = ("这一步没有拿到合法的目标路线（target_route 槽为空或不在 "
            "search/action/general 里），没有切换路线。")

    def __init__(self) -> None:
        super().__init__(f"{self.code}: {self.hint}")


def _loop_route_request(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """逃生口元动词：如实记录换线请求——真正的路线切换由 execute
    据本步结果写 state.route_scope（下一轮 decide 按新套件装面）。target_route 非法 →
    bad_param（预算闸/同线闸在 `_adjudicate_decide_obj`，本函数只管形状）。"""
    slots = slots or {}
    target = str(slots.get("target_route") or "").strip()
    if target not in _ROUTE_REQUEST_TARGETS:
        raise _RouteRequestParamError()
    return {
        "requested_route": target, "switched": True,
        "reason": str(slots.get("reason") or "").strip(),
    }


                                          
                                                             
                                                                        
LOOP_TOOLS["route.request"] = {
    "run": _loop_route_request, "label_zh": _ap.VERB_BY_NAME["route.request"].zh, "card_kind": "route_request",
    "readonly": True,
    "decide_zh": ("发现当前处理路线不对时，请求换到另一条路线：search=检索向（找数据/改条件/"
                  "贴编号）、action=动作向（下载/联网搜库/检查更新/入库/管护）、"
                  "general=全能兜底（拿不准就走它）。每轮至多 1 次（机械闸），换线后按"
                  "新路线的工具与纪律继续。槽位 target_route 必填、reason 可选。"),
}


                                                                                                  

def _rollback_refuse(sid: str | None, reason: str, note_zh: str) -> dict:
    """回滚机械闸的拒绝出口（形状与成功档同一契约——拒绝是数据不是故障）。"""
    return {"snapshot_id": sid, "rolled_back": False, "reason": reason, "verb": "",
            "recycled": [], "restored": [], "skipped": [], "unrestorable": [],
            "errors": [], "note_zh": note_zh}


def _loop_curate_rollback(slots: dict, root: Path, ctx: dict | None = None) -> dict:
    """回滚动词（2026-08-17 rb1）：把本轮**最近一步带快照且未回过**的写步骤回退掉。

    机械闸（零槽位——回哪一步由本闸现定，模型发明不了快照 id）：
    - 候选 = ctx.steps 里的**正向写步**且有 snapshot_id（成败都算——失败写步也可能
      finalize 了半写现场）；curate.rollback 自己的 trace 快照明确跳过——本工具不支持
      「回滚回滚」，避免 ping-pong；
    - 逐新到旧只允许已 rolled_back_at 的正向写步跳过；meta 缺失/损坏与未 finalize
      都 fail-closed，绝不越过不可确认的最新写步去动更早现场；
    - 最近可用锚**未 finalize → fail-closed 如实拒绝**——不越过它去回更早的步
      （乱序回退会把现场搞得更难读；宁可少退不毁数据，与 rollback.py 同纪律）；
    - 没有可回滚步 → 如实拒绝。拒绝不抛异常（search.rerun adopted=False 同哲学：
      闸的裁定是数据，step.ok 恒 True）。
    只许回滚**本轮** session 的写步——ctx.steps 是本轮实录，结构性保证够不到历史快照。"""
    steps = list((ctx or {}).get("steps") or [])
    store = _trace_snapshot_store(root)
    candidates = [s for s in reversed(steps)
                  if isinstance(s, dict)
                  and str(s.get("verb") or "") in _WRITE_LOOP_TOOLS
                  and str(s.get("snapshot_id") or "")]
    for step in candidates:
        sid = str(step["snapshot_id"])
        try:
            meta = store.load(sid)
        except _SnapshotError:
            return _rollback_refuse(
                sid, "snapshot_unavailable",
                "最近一步写操作的快照缺失或损坏，无法确认它是否已经回滚；为安全起见"
                "不越过它回退更早的步骤。")
        if meta.get("rolled_back_at"):
            continue                  
        if not meta.get("finalized"):
            return _rollback_refuse(
                sid, "snapshot_not_finalized",
                "最近一步写操作的快照不完整（执行后未 finalize），为安全起见不越过它"
                "回退更早的步骤。")
        try:
            res = _trace_rollback.apply_rollback(root, sid)
        except _SnapshotError as exc:
            return _rollback_refuse(sid, "snapshot_error",
                                    f"回退时快照不可用：{exc}")
        recycled = [str(e["name"]) for e in res["applied"]["recycled"]]
        restored = [str(e["name"]) for e in res["applied"]["restored"]]
        verb_zh = str(step.get("verb_zh") or step.get("verb") or "写操作")
        segs: list[str] = []
        if recycled:
            segs.append(f"{len(recycled)} 个新文件移入回收站")
        if restored:
            segs.append(f"{len(restored)} 个文件恢复原字节")
        n_bad = len(res.get("unrestorable") or []) + len(res.get("errors") or [])
                                  
        rolled_back = bool(res.get("rolled_back"))
        skipped = list(res.get("skipped") or [])
        if rolled_back:
            prefix = f"已回滚「{verb_zh}」这一步："
            note = prefix + "、".join(segs) + "。"
            reason = "rolled_back"
        else:
            facts = ("、".join(segs) if segs else "未实际恢复任何文件")
            tails: list[str] = []
            if n_bad:
                tails.append(f"{n_bad} 项未能恢复")
            if skipped:
                tails.append(f"{len(skipped)} 项因现场已变化而跳过")
            note = (f"没有完成回滚「{verb_zh}」：{facts}"
                    + (("；" + "、".join(tails)) if tails else "") + "。")
            reason = "rollback_incomplete"
        return {"snapshot_id": sid, "rolled_back": rolled_back, "reason": reason,
                "verb": str(step.get("verb") or ""),
                "recycled": recycled, "restored": restored,
                "skipped": skipped,
                "unrestorable": list(res.get("unrestorable") or []),
                "errors": list(res.get("errors") or []),
                "note_zh": note}
    return _rollback_refuse(
        None, "no_rollbackable_step",
        "本轮还没有可回滚的写操作——需要有已执行、留了快照且尚未回滚的写步骤。")


                                                          
                                                                
                                            
                                                            
                                                                     
                                                                  
LOOP_TOOLS["curate.rollback"] = {
    "run": _loop_curate_rollback, "label_zh": _ap.VERB_BY_NAME["curate.rollback"].zh, "card_kind": "rollback",
    "readonly": False, "needs_context": True,
    "decide_zh": ("撤销本轮**最近一步写操作**（联网搜入库/同步入库等），把它动过的文件回到"
                  "动手前的样子（新文件移入回收站、被改动/删除的写回原字节；回收站式可逆，"
                  "绝不真删）。零槽位——回哪一步由机械闸定（最新一步带快照的写步），不用也"
                  "不能指定；没有可回滚的步、快照缺失/损坏/不完整都会如实拒绝，不硬来。"
                  "不支持回滚回滚；重复调用只会依次回退更早的正向写步。"),
}


def _execute_detail_zh(spec: dict, result: dict) -> str:
    """execute 成功步的 trace 摘要（人读一句；db_status 沿用原 observe 的口径）。"""
    kind = spec.get("card_kind")
    if kind == "db_status":
        total = result.get("total_records")
        return (f"读到 {len(result.get('sources') or [])} 个来源"
                + (f"、共 {total} 条" if total is not None else "") + "。")
    if kind == "check_updates":
        return f"检查了 {len(result.get('sources') or [])} 个来源的更新。"
    if kind == "search_online":
        filename = result.get("filename")
        if not filename:
                                               
                                                    
            warn = next((str(w).strip() for w in (result.get("warnings") or [])
                         if str(w or "").strip()), "")
            return warn if warn else "联网搜索的候选均已在库中，没有重复入库。"
        return (f"联网搜到 {int(result.get('record_count') or 0)} 条，"
                f"已写入外部库 {str(filename)}。")
    if kind == "sync_updates":
        imported = int(result.get("imported_total") or 0)
        base = f"检查了 {len(result.get('sources') or [])} 个来源的更新"
        return base + (f"，新入库 {imported} 条。" if imported else "，没有需要入库的新增。")
    if kind == "search_rerun":
        if result.get("adopted"):
                                                        
                                       
                                                       
            nb = result.get("n_before_total") if result.get("n_before_total") is not None \
                else result.get("n_before")
            na = result.get("n_after_total") if result.get("n_after_total") is not None \
                else result.get("n_after")
            return (f"按「{str(result.get('query') or '')}」重新检索："
                    + (f"原来 {int(nb)} 条" if isinstance(nb, int) and not isinstance(nb, bool)
                       else "原结果")
                    + f" → {int(na or 0)} 条，结果已更新。")
        reason = str(result.get("reason") or "")
        if reason == "rewrite_no_change_kept_original":
            return (f"按「{str(result.get('query') or '')}」重新检索的结果与当前相同，"
                    "结果区未改动。")
        if reason == "structured_context_lost_kept_original":
            return str(result.get("disclosure_zh") or "") or \
                "这次重新检索没有执行：新查询没能完整保留当前筛选条件，结果区未改动。"
        return "重新检索未执行，结果区未改动。"
    if kind == "rank":
        shown = str(result.get("query") or "")
        suffix = "，结果区已自动上屏" if result.get("displayed") else ""
        return f"检索（{shown}）：命中 {int(result.get('total') or 0)} 条{suffix}。"
    if kind == "rerank":
        if result.get("refused"):
                                                       
                                               
            return str(result.get("note_zh") or "") or \
                "优化检索词重查没有执行：当前结果区状态良好，不需要重查。"
        total = int(result.get("total") or 0)
        if result.get("rejected"):
                                                  
                           
            return str(result.get("comparison_zh") or "") or \
                "优化检索词重查被驳回：新结果不如当前结果健康，结果区未改动。"
        suffix = "，结果区已自动上屏" if result.get("auto_shown") else ""
        if result.get("rewritten"):
            return (f"优化检索词重查（{str(result.get('original_query') or '')} → "
                    f"{str(result.get('rewritten_query') or '')}）：命中 {total} 条{suffix}。")
        return (f"优化检索词重查：改写未通过检查，按原句"
                f"（{str(result.get('original_query') or '')}）重查，命中 {total} 条{suffix}。")
    if kind == "results_show":
        if result.get("shown"):
            return (f"结果区已更新为「{str(result.get('query_effective') or '')}」"
                    f"（{int(result.get('total') or 0)} 条）。")
        return str(result.get("note_zh") or "") or "上屏没有执行。"
    if kind == "route_request":
        target = str(result.get("requested_route") or "")
        return f"已切换处理路线（→ {target}），按新路线继续。"
    if kind == "rollback":
                                                            
                                 
        return str(result.get("note_zh") or "") or "回滚写操作完成。"
                                              
                                        
    if kind == "compare":
        if result.get("degraded"):
            return str(result.get("comparison_zh") or "") or "对比没有完成。"
        return (f"对比完成（{(result.get('a') or {}).get('dataset_name')} vs "
                f"{(result.get('b') or {}).get('dataset_name')}）："
                f"{int(result.get('n_same') or 0)} 个字段一致、"
                f"{int(result.get('n_diff') or 0)} 个字段不同。")
    if kind == "cite_export":
        return str(result.get("note_zh") or "引文导出完成。")
    if kind == "compat_find":
        if result.get("degraded"):
            return str(result.get("note_zh") or "") or "兼容查找没有完成。"
        return f"找到 {int(result.get('total') or 0)} 个元数据兼容的数据集。"
    if kind == "fair_check":
        if result.get("degraded"):
            return str(result.get("note_zh") or "") or "FAIR 自检没有完成。"
        summary = (result.get("fair") or {}).get("summary") or {}
        return (f"FAIR 复用就绪度 {int(summary.get('readiness_pct') or 0)}%"
                f"（{int(summary.get('pass') or 0)} 充分 / {int(summary.get('partial') or 0)} 部分 / "
                f"{int(summary.get('unknown') or 0)} 未知）。")
    return "工具执行完成。"


                                  
                                                             
                                                
                                         
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """同一原因只在 stderr 提示一次；绝不抛异常、绝不打断请求。"""
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[agent_exec] {message}", file=sys.stderr)


def _audit_loop_tool(root: Path, verb: str, slots: dict, ok: bool, note: str,
                     records: int | None = None) -> None:
    """图内执行的审计行：与联网 fetch **同一本账**（.userdata/curate_net_ledger.jsonl，
    复用 corpus_curation 的落账小助手——import 单向：agent_exec → corpus_curation 合法）。
    审计自身失败绝不掀翻执行（账本是审计，不是执行的前提）。
    records 只在工具真带回条数时写实数（如 search_online 入库 N 条）；只读工具没有
    「写入条数」可记，字段整个省略——恒 0 会让账本统计口径失真。"""
    try:
        from ..corpus.net_ledger import append_net_ledger, net_ledger_path, now_iso

        row = {
            "ts": now_iso(),
            "endpoint": f"agent_exec:{verb}",
            "query": json.dumps(dict(slots or {}), ensure_ascii=False)[:120],
            "ok": bool(ok),
            "error": "" if ok else str(note)[:200],
        }
        if records is not None:
            row["records"] = int(records)
                                          
        append_net_ledger(net_ledger_path(Path(root)), row)
    except Exception as exc:
                                    
        _warn_once(f"net_ledger_write::{type(exc).__name__}",
                   f"联网账本写盘失败（{type(exc).__name__}），审计行可能缺行。")


def _audit_fallback(root: Path, node: str, reason: str, utterance: str, model: str) -> None:
    """跌 JSON-in-prompt 兜底的**抓现场账**（2026-08-06 B4；2026-08-07 换装起 node 参数化——
    decide 迁工具通道后也有了兜底档，账本如实标注是哪个节点跌的）。

    生产偶发「直连通道不可用→回退问法」在离线 A/B 里复现不了（required→auto 档 48/48
    成功）——既然复现不了，就让真机自己留证：每次跌兜底落一行
    `.userdata/agent_fallbacks.jsonl`（ts/node/reason/model/utterance 截断）。
    与联网账本同目录同纪律：审计自身失败绝不掀翻主流程。"""
    try:
        from ..corpus.net_ledger import append_jsonl, net_ledger_path, now_iso

        append_jsonl(net_ledger_path(Path(root)).parent / "agent_fallbacks.jsonl", {
            "ts": now_iso(),
            "node": str(node or ""),
            "reason": str(reason or "")[:200],
            "model": str(model or ""),
            "utterance": str(utterance or "")[:120],
        })
    except Exception as exc:
                                             
        _warn_once(f"fallback_audit_write::{type(exc).__name__}",
                   f"降级审计账本写盘失败（{type(exc).__name__}），兜底现场可能缺行。")


                                                                                    
                                                         
                                                        
                           
                                                      
                                                                   
_EXAMPLES_LEDGER_NAME = "curate_examples.jsonl"                                
_EXAMPLE_CANDIDATES_NAME = "curate_example_candidates.jsonl"                   
_EXAMPLES_MAX_ROWS = 200                             
_EXAMPLES_INJECT_LIMIT = 3                           
_EXAMPLES_MIN_OVERLAP = 2                                


def _examples_ledger_path(root: Path) -> Path:
    from ..corpus.net_ledger import net_ledger_path

    return net_ledger_path(Path(root)).parent / _EXAMPLES_LEDGER_NAME


def _example_candidates_path(root: Path) -> Path:
    return _examples_ledger_path(Path(root)).parent / _EXAMPLE_CANDIDATES_NAME


def _iter_example_rows(path: Path) -> Iterator[dict]:
    """JSONL 共用流式解析锚点；坏行留痕但不回显正文，内存不随历史行数增长。"""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("ledger_row_not_object")
            except Exception as exc:
                _warn_once(f"example_ledger_bad_line::{type(exc).__name__}",
                           f"经验库账本 {path.name} 有无法解析的行（{type(exc).__name__}），"
                           "已跳过——若跳光等价于账本损坏，请检查该文件。")
                continue
            yield row


def _read_example_rows(path: Path, *, streaming: bool = False) -> list[dict] | Iterator[dict]:
    """旧调用仍返回可修改列表；长历史读面显式选流式并先过滤/预算，不另建解析器。"""
    rows = _iter_example_rows(path)
    return rows if streaming else list(rows)


def _write_example_rows(path: Path, rows: list[dict]) -> None:
    from ..app.conversation_runs import RunError
    raise RunError('knowledge_legacy_write_disabled', '原账本仅作迁移留档，修改须经过统一知识道闸。')


def _endpoint_fp_from_config(config: Any) -> str:
    """端点指纹（成功经验库分区键）：sha256（base_url|model) 前 12 位。
    api_key 永不参与——换 key 不该换分区，key 材料也绝不进任何账本邻接面。"""
    material = str(getattr(config, "base_url", "") or "") + "|" + str(getattr(config, "model", "") or "")
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def _example_args_zh(verb: str, slots: dict) -> str:
    """动作参数的紧凑人读投影：只收安全标量槽（target/keywords/source/species/limit），
    其余槽型（dict/list）不注——prompt 体积有预算，且这些槽对表达习惯对齐没有增量。"""
    parts: list[str] = []
    for key in ("target", "keywords", "source", "species", "limit"):
        value = (slots or {}).get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            parts.append(f"{key}={str(value).strip()[:40]}")
    return f"{verb}（{'，'.join(parts)}）" if parts else verb


@_learning_scope.synchronized_bank
def _maybe_record_success(root: Path, utterance: str, plan: dict, steps: list,
                          *, principal: str = "anonymous", endpoint_fp: str = "",
                          mode: str = "tools", repairs: int = 0, finish_vetoes: int = 0,
                          reask_write_count: int = 0, truncated: bool = False,
                          checklist_dropped: int = 0, project_id: str | None = None) -> None:
    """成功收尾机械追加一行进**候选池**（2026-08-13 起不直接进正式库——用户在记忆模块
    预览勾选后才由 `approve_example_candidates` 迁入，注入侧只读正式库）。
    **机械判据**（不问 LLM）：有真跑的工具步、每步都 ok、至少一步是 curate.* 管护动词、
    plan 非取消态。账本自身失败绝不掀翻主流程。
    每行打**分区标**（principal 会话账户 + endpoint_fp 端点指纹）——
    原话是隐私面，候选展示与注入都只取同分区行。
    收录质量闸：**「跑通」不等于「干得漂亮」**——下列信号任一命中即不录：
    理解通道跌 JSON 兜底（mode != "tools"）、首步被 repair 修过、finish 核销被打回/
    写步被重问（finish_vetoes / reask_write_count）、被步数上限掐停（truncated）、
    清单幻觉条目被剔（checklist_dropped）。被判据修好的执行，其「原话→动作」映射正是
    会教偏模型的毒样例；宁可少录，不录脏。kw-only 缺省全为「干净」——内部直调/旧钉
    行为逐位不变。"""
    try:
        if mode != "tools":
            return
        if repairs or finish_vetoes or reask_write_count or truncated or checklist_dropped:
            return
        if not steps or (plan or {}).get("cancelled"):
            return
        if not all(bool(s.get("ok")) for s in steps):
            return
        verbs = [str(s.get("verb") or "") for s in steps]
        if not any(v.startswith("curate.") for v in verbs):
            return
        text = str(utterance or "").strip()[:120]
        if not text:
            return
        principal = str(principal or "anonymous")
        endpoint_fp = str(endpoint_fp or "")
        project_id = _learning_scope.resolve_project_id(project_id)
        step_rows = [{"verb": v, "args": _example_args_zh(v, dict(s.get("slots") or {}))}
                     for s, v in zip(steps, verbs)]
        service = _example_knowledge(root)
        after = {'kind': 'example', 'title': '', 'text': text, 'trigger': '', 'tier': 'explicit',
            'permanence': 'persistent', 'payload': {'utterance': text, 'steps': step_rows, 'verified_success': False}, 'links': []}
        service.propose(principal, endpoint_fp, project_id,
            request_id='runtime-example:' + _collab_rules.content_fingerprint([project_id, after])[:32],
            operations=[{'op': 'create', 'after': after}], actor='model',
            source_refs=[{'type': 'user', 'text': text, 'project_id': project_id}])
    except Exception as exc:
                                        
                                
        _warn_once(f"example_candidate_write::{type(exc).__name__}",
                   f"成功经验候选池写盘失败（{type(exc).__name__}），本次收录可能未落盘。")


def _example_partition_of(row: dict) -> tuple[str, str, str]:
    """分区标：缺字段的存量行按 （"anonymous","") 计——与 `_load_success_examples` 同口径。"""
    return _learning_scope.partition_of(row)


def _example_same_partition(row: dict, principal: "str | None", endpoint_fp: "str | None",
                            project_id: str | None = None, *, inherit_global: bool = False) -> bool:
    """principal=None = 内部直调/测试不过滤（旧行为逐位不变）；否则双键全等才同区。"""
    return _learning_scope.matches(row, principal, endpoint_fp, project_id, inherit_global=inherit_global)


def _example_knowledge(root):
    from ..app.conversation_store import ConversationStore
    from ..app.knowledge_service import KnowledgeService
    return KnowledgeService(ConversationStore(Path(root)))


def list_example_candidates(root: Path, *, principal: "str | None" = None,
                            endpoint_fp: "str | None" = None, project_id: str | None = None) -> list[dict]:
    if principal is None:
        return []
    service = _example_knowledge(root)
    project = _learning_scope.resolve_project_id(project_id)
    items, offset = [], 0
    while True:
        page = service.changes(principal, endpoint_fp or '', project, offset=offset, limit=200)
        for change in page['items']:
            if change['status'] in {'committed', 'rejected'}:
                continue
            for op in change['operations']:
                after = op.get('after', {})
                if after.get('kind') == 'example':
                    items.append(dict(after['payload'], id=change['id'], status=change['status'],
                        change_id=change['id'], change_revision=change['revision'], fingerprint=change['fingerprint'],
                        assessment=change.get('assessment'), assessment_id=change.get('assessment_id'), project_id=project))
        offset = page['next_offset']
        if offset is None:
            return items


def _example_change(service, principal, endpoint, change_id, project):
    from ..app.conversation_runs import RunError
    try:
        row = service.change(principal, endpoint, change_id)
    except RunError as exc:
        if exc.code != 'knowledge_change_not_found':
            raise
        migrated = service.store.resolve_legacy_id(principal, endpoint, change_id, filename='curate_example_candidates.jsonl')
        row = service.change(principal, endpoint, migrated)
    if row['project_id'] != project or not any(op.get('after', {}).get('kind') == 'example' for op in row['operations']):
        raise RunError('knowledge_scope_conflict', '这不是当前项目的示例提案。')
    return row


def approve_example_candidates(root: Path, ids: Any, *, principal: "str | None" = None,
                               endpoint_fp: "str | None" = None, project_id: str | None = None) -> dict:
    from ..app.conversation_runs import RunError
    if principal is None or type(ids) is not list or not 1 <= len(ids) <= 32:
        raise RunError('knowledge_invalid_operations', '示例提案需要本人身份及有界编号清单。')
    service, project = _example_knowledge(root), _learning_scope.resolve_project_id(project_id)
    rows = [_example_change(service, principal, endpoint_fp or '', cid, project) for cid in ids]
    return {'approved': 0, 'duplicated': 0, 'stage': 'draft', 'changes': rows,
        'detail': '示例仍未生效；请在知识审查中评估并真人提交。'}


def dismiss_example_candidates(root: Path, ids: Any, *, principal: "str | None" = None,
                               endpoint_fp: "str | None" = None, project_id: str | None = None) -> dict:
    from ..app.conversation_runs import RunError
    if principal is None or type(ids) is not list or not 1 <= len(ids) <= 32:
        raise RunError('knowledge_invalid_operations', '驳回需要本人身份及有界编号清单。')
    service, project = _example_knowledge(root), _learning_scope.resolve_project_id(project_id)
    receipts = []
    for cid in ids:
        row = _example_change(service, principal, endpoint_fp or '', cid, project)
        receipts.append(service.reject(principal, endpoint_fp or '', project,
            change_id=row['id'], expected_change_revision=row['revision']))
    return {'ok': all(row['ok'] for row in receipts), 'dismissed': sum(row['ok'] for row in receipts), 'changes': receipts}


def _char_bigrams(text: str) -> set:
    t = "".join(ch for ch in str(text or "").lower() if ch.strip())
    if not t:
        return set()
    if len(t) == 1:
        return {t}
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _load_success_examples(root: Path, utterance: str,
                           limit: int = _EXAMPLES_INJECT_LIMIT,
                           *, principal: "str | None" = None,
                           endpoint_fp: "str | None" = None, project_id: str | None = None) -> list[dict]:
    """旧读取名仅投影已确认知识，不把迁移示例当成可核验成功。"""
    from .knowledge import legacy_examples
    if principal is None:
        return []
    snapshot = _example_knowledge(root).snapshot(principal, endpoint_fp or '',
        _learning_scope.resolve_project_id(project_id))
    return legacy_examples(snapshot['items'])


def _knowledge_context_safe(root, utterance, *, principal, endpoint_fp, project_id,
                            phase, available_tools) -> str:
    """统一知识投影的消费层护栏：道闸 RunError（超预算/作用域失效等）降级为空段并经
    _warn_once 留痕，绝不抛穿主轮（知识是增益注入，不是执行前提）；编程错误
    （非 RunError）仍原样抛出。降级不等于截断——条目一条不丢，恢复后完整注入。"""
    from .knowledge import knowledge_context
    from ..app.conversation_runs import RunError
    try:
        return knowledge_context(root, utterance, principal=principal,
            endpoint_fp=endpoint_fp or '', project_id=project_id, phase=phase,
            available_tools=available_tools)
    except RunError as exc:
        _warn_once(f"knowledge_context::{exc.code}",
                   f"已确认知识投影本轮降级为空（{exc.code}）；未截断未丢弃，恢复后完整注入。")
        return ''


def _examples_prompt_zh(root: Path, utterance: str, *,
                        principal: "str | None" = None,
                        endpoint_fp: "str | None" = None, project_id: str | None = None,
                        available_tools=None) -> str:
    from .knowledge import registered_tools
    return _knowledge_context_safe(root, utterance, principal=principal,
        endpoint_fp=endpoint_fp or '',
        project_id=_learning_scope.resolve_project_id(project_id), phase='understand',
        available_tools=registered_tools() if available_tools is None else available_tools)


def _step_projection(step: dict, *, reasked_write: bool = False) -> dict:
    """单步结果的**紧凑投影**（decide/narrate 的 prompt 素材）：只留「判断下一步 / 写汇报」
    要用的字段——原始 result 整个塞进去会把 prompt 吹爆（db_status 带全量清单）。
    `reasked_write=True` 时打标（2026-08-07 B 方案：重问后放行的写步——模型据此知道
    finish 的 completion_report 必须引用本步步骤号并单独交代结果，否则核销硬闸拒收）。"""
    out: dict[str, Any] = {
        "verb": step.get("verb"),
        "verb_zh": step.get("verb_zh"),
        "ok": bool(step.get("ok")),
    }
    if reasked_write:
        out["reasked_write"] = True
    if not step.get("ok"):
        out["error"] = str(step.get("error") or "")[:300]
        return out
    kind = step.get("card_kind")
    r = step.get("result") or {}
    if kind == "db_status":
        out["result"] = {
            "total_records": r.get("total_records"),
            "sources": r.get("sources"),
            "external_files": len(r.get("external_files") or []),
            "recycle": len(r.get("recycle") or []),
            "ledger": r.get("ledger"),
        }
    elif kind == "check_updates":
        entries: list[dict[str, Any]] = []
        for s in (r.get("sources") or []):
            entry = {
                k: s.get(k)
                for k in ("source", "label", "mode", "local_count", "online_recent",
                          "new_count", "snapshot_date", "note_zh")
                if s.get(k) is not None
            }
            titles = [str(c.get("title") or "") for c in (s.get("new_candidates") or [])[:5]]
            if titles:
                entry["new_titles"] = titles
            entries.append(entry)
        out["result"] = {"sources": entries}
    elif kind == "search_online":
        out["result"] = {
            k: r.get(k)
            for k in ("source_label", "query", "species", "record_count", "filename",
                      "sample_titles", "warnings")
            if r.get(k) is not None
        }
    elif kind == "sync_updates":
                                                              
                                                           
        entries2: list[dict[str, Any]] = []
        for s in (r.get("sources") or []):
            entries2.append({
                k: s.get(k)
                for k in ("source", "label", "mode", "new_count", "imported_count",
                          "filename", "note_zh")
                if s.get(k) is not None
            })
        out["result"] = {"imported_total": r.get("imported_total"), "sources": entries2}
    elif kind == "search_rerun":
                                                                     
                                                             
        proj: dict[str, Any] = {
            k: r.get(k)
            for k in ("adopted", "reason", "query", "n_before", "n_after",
                      "n_before_total", "n_after_total", "replace_screen")
            if r.get(k) is not None
        }
        if r.get("adopted") and isinstance(r.get("payload"), dict):
            proj["result_total"] = (r["payload"] or {}).get("result_total")
        out["result"] = proj
    elif kind in ("rank", "rerank"):
                                                                
                                                                           
                                                                                
                                                          
                                                                     
                                                                   
        proj_rr: dict[str, Any] = {
            k: r.get(k)
            for k in ("query", "original_query", "rewritten_query", "rewritten",
                      "total", "filters", "top", "displayed", "health",
                      "resolution_status", "rejected", "auto_shown", "baseline",
                      "relax_live", "refused", "refuse_reason", "reject_reason",
                      "semantic_blocked")
            if r.get(k) is not None
        }
        if r.get("comparison_zh"):
            proj_rr["comparison_zh"] = str(r.get("comparison_zh"))[:400]
        for k in ("note_zh", "disclosure_zh"):
            if r.get(k):
                proj_rr[k] = str(r.get(k))[:300]
        out["result"] = proj_rr
    elif kind == "results_show":
                                                       
                                                                      
        out["result"] = {
            k: r.get(k)
            for k in ("shown", "handle", "origin", "query_effective", "total",
                      "overridden", "note_zh")
            if r.get(k) is not None
        }
    elif kind == "route_request":
                                         
        out["result"] = {
            k: r.get(k)
            for k in ("requested_route", "switched", "reason")
            if r.get(k) is not None
        }
    elif kind == "compare":
                                                 
                                                                       
        proj: dict[str, Any] = {
            k: r.get(k)
            for k in ("assumption_zh", "n_same", "n_diff", "n_unknown", "identical",
                      "wording_source", "degraded", "degrade_reason")
            if r.get(k) is not None
        }
        proj["comparison_zh"] = str(r.get("comparison_zh") or "")[:400]
        proj["a"] = r.get("a")
        proj["b"] = r.get("b")
        out["result"] = proj
    elif kind == "cite_export":
                                                    
        out["result"] = {
            k: r.get(k)
            for k in ("n_datasets", "uids", "files", "out_dir", "note_zh")
            if r.get(k) is not None
        }
    elif kind == "compat_find":
                                                   
        names = [str(c.get("dataset_name") or "") for c in (r.get("compatible") or [])[:5]]
        out["result"] = {
            "total": r.get("total"),
            "seed": r.get("seed"),
            "top_names": names,
            "caveat": str(r.get("caveat") or "")[:200],
            "degraded": r.get("degraded"),
            "degrade_reason": r.get("degrade_reason"),
            "note_zh": str(r.get("note_zh") or "")[:200],
        }
    elif kind == "fair_check":
                                                                
        summary = (r.get("fair") or {}).get("summary") or {}
        gaps = [(str(g.get("id") or ""), str(g.get("label") or ""))
                for g in (r.get("fair") or {}).get("gaps") or []]
        out["result"] = {
            "dataset_name": r.get("dataset_name"),
            "summary": summary,
            "gaps": gaps[:6],
            "degraded": r.get("degraded"),
            "degrade_reason": r.get("degrade_reason"),
            "note_zh": str(r.get("note_zh") or "")[:200],
        }
    else:
        out["result"] = {}
    return out


def _report_fallback_zh(obs: dict) -> str:
    """数据库状态的**确定性**汇报（LLM 缺席时的兜底——与 narrate 的 LLM 汇报同一批事实）。"""
    sources = list(obs.get("sources") or [])
    total = int(obs.get("total_records") or 0)
    top = "、".join(f"{s['label']} {s['local_count']}" for s in sources[:3])
    ext = len(obs.get("external_files") or [])
    rec = len(obs.get("recycle") or [])
    ledger = dict(obs.get("ledger") or {})
    parts = [f"目录共收录 {total} 条、来自 {len(sources)} 个来源（{top} 等）",
             f"你上传或联网搜来的文件 {ext} 个、回收站 {rec} 个"]
    if ledger.get("entries"):
        parts.append(f"近期联网操作记录 {int(ledger['entries'])} 条")
    else:
        parts.append("近期没有联网操作记录")
    return "；".join(parts) + "。"


_REPORT_RULES_ZH = (
    "你是数据库状态汇报员。据随附的**真实数据**写一段简明中文汇报（不超过 120 字）。\n"
    "铁律（违反任一条都是错误）：\n"
    "1. 只用给定的事实与数字——不得编造、不得改写数字、不得估算；\n"
    "2. snapshot_date 为 null 的来源**不许提日期**；\n"
    "3. 结构：先一句总条数与来源数，再点用户最可能关心的两三点（外部库/回收站有几个文件、"
    "近期有没有联网操作、某源条数）；不要建议、不要评论、不要客套。\n"
)


def _report_with_llm(chat_model: Any, obs: dict, usage_sink: list | None = None) -> str | None:
    """narrate 的 LLM 汇报：observation 紧凑投影 → 一段简明中文。失败/空回 → None（兜底确定性）。"""
    from langchain_core.messages import HumanMessage

    compact = {
        "total_records": obs.get("total_records"),
        "sources": obs.get("sources"),
        "external_files": obs.get("external_files"),
        "recycle": obs.get("recycle"),
        "ledger": obs.get("ledger"),
    }
    prompt = _REPORT_RULES_ZH + "\n----- 真实数据（JSON）-----\n" + json.dumps(
        compact, ensure_ascii=False)
    try:
        answer = _invoke_text_with_continuation(
            chat_model, [HumanMessage(content=prompt)],
            usage_sink=usage_sink, usage_node="narrate")
    except Exception:
        return None
    text = _message_text(answer).strip()
    return text or None


                                                                                            
                                                                     
 
                                                   
                                                                                      
                                             
                                                                
                                                            
           
                                                                                            

                                                 
                                                 
                                                                                
                                                    
_DECIDE_VERB_ORDER: tuple[str, ...] = (
    "curate.check_updates", "curate.search_online", "curate.sync_updates", "curate.db_status",
    "curate.rollback", "search.rerun", "rank", "rerank", "results.show", "ask.relax",
    "route.request",
    "compare.datasets", "cite.export", "compat.find", "fair.check",
)


def _decide_tool_table_zh(verbs: Any = None) -> str:
    """decide prompt 的动词清单——**由 LOOP_TOOLS 注册表程序生成**（注册表是唯一真源；
    此前 prompt 里这份三动词清单是手抄的第二份拷贝，2026-08-06 消漂移面）。
    各行描述取注册项的 `decide_zh`；模块加载期从真实注册表取一次（各面规则壳均为
    加载期装配的常量）——测试整体替换 LOOP_TOOLS 不受影响，注册表将来加工具时清单自动跟随。
    带 source 槽的动词行尾程序拼候选清单（与 schema 枚举
    同出 `agent_schemas.source_candidates_zh`——prompt 名单与 schema 枚举不再有第二份拷贝）。
    `verbs`（2026-08-17 scoped 路由）：显式给顺序子集时按子集出表（套件收窄面）；
    缺省 None = `_DECIDE_VERB_ORDER` 全表，与历史输出**逐位一致**。"""
    rows = []
    for verb in (verbs if verbs is not None else _DECIDE_VERB_ORDER):
        row = f"   - {verb}（{LOOP_TOOLS[verb]['decide_zh']}"
        if "source" in (_ap.VERB_BY_NAME[verb].slots if verb in _ap.VERB_BY_NAME else ()):
            row += f"；source 候选：{_source_candidates_zh(verb)}"
        rows.append(row + "）")
    return "\n".join(rows)


                                                         
                                        
                                           
                                              
                                                                           
                                                         
_DECIDE_JSON_BULLETS_ZH = (
    "- 已完成；或剩下的事**做不到 / 条件不成立**（例如检查结果是「没有新增」，那「若有则下载」"
    "就不用做；或用户要的来源本工具接不了）→ 只回 {\"done\": true}\n"
    "- 还需要再做一步 → 只回**一个 JSON 对象**（不要任何多余文字）："
    "{\"verb\": \"…\", \"quoted\": \"…\", 该动作需要的槽位}；"
    "本通道一次只能发一个调用——「先检索再对比/导出」这类顺序诉求**分步发**：先回检索步，"
    "等它执行完再回下一步（JSON 通道不支持同批依赖占位）\n"
)

                                                                   
                                                    
                                                           
            
_DECIDE_TOOLS_CHANNEL_BULLETS_ZH = (
    "- 已完成；或剩下的事**做不到 / 条件不成立**（例如检查结果是「没有新增」，那「若有则下载」"
    "就不用做；或用户要的来源本工具接不了）→ 调用 finish（**必须附 completion_report**，"
    "逐件核销要求见上方「finish 契约」）\n"
    "- 还需要再做一步 → 调用对应的工具；若接下来要做的几件事**彼此独立且都是"
    "只读**（如逐来源检查更新、读库容），一次把它们各发一个调用；"
    "有先后依赖或会写库的动作仍一次只发一个\n"
    "- 用户还要求了本循环做不到的事（例如打包下载、删除文件）→ 调用 unsupported_next_step"
    " 说明是那一件\n"
)

                                                  
_DECIDE_FINISH_TOOL = "finish"
_DECIDE_UNSUPPORTED_TOOL = "unsupported_next_step"


def _build_decide_tool_specs() -> tuple[list[dict], dict[str, str]]:
    """decide 的 tool-calling 工具面（2026-08-07 评审改案）：

    **5 个 loop 工具**（2026-08-16 检索工具化起含 search.rerun；schema 由
    `agent_schemas.verb_parameters_schema` 程序生成、单一真源，
    description 取注册表的循环语境专职描述 `decide_zh`）+ **2 个控制工具**：
    - `finish`：结构化 done——「要做的事已全部完成，或剩下的做不到/条件不成立」；
      **必填** `completion_report` 参数承载逐件核销报告（2026-08-08 起机械层消费：
      `_unfinished_business` 扫出报告里自认「没做」的事项 → decide 拒收收尾并回灌重问
      一次，见 decide 节点注释；`_decide_answer_kind` 因此把 finish 的 args 带出来）；
    - `unsupported_next_step`：承载「用户还要一件本图做不了的事」——婉拒能力的正式通道
      （verb 槽是**非 loop 动词的真枚举**，程序取自 VERB_SPECS − LOOP_TOOLS）。

    刻意**不**绑 18 动词全表：那把 pack.download 等不可在循环内执行的动作伪装成可调用
    工具，显著提高误选率、还吹大 prompt。

    **模块加载期构建一次**（与各面规则壳同纪律）：读的是真实 LOOP_TOOLS——
    测试整体替换注册表（替身项没有 decide_zh）不影响本常量；构建后只读。"""
    tools: list[dict] = []
    name_to_verb: dict[str, str] = {}
    for verb in _DECIDE_VERB_ORDER:
        spec = _ap.VERB_BY_NAME[verb]
        name = verb.replace(".", "_")
        name_to_verb[name] = verb
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": LOOP_TOOLS[verb]["decide_zh"],
                "parameters": _verb_parameters_schema(spec),
            },
        })
    unsupported = [s.verb for s in _ap.VERB_SPECS if s.verb not in LOOP_TOOLS]
    tools.append({
        "type": "function",
        "function": {
            "name": _DECIDE_FINISH_TOOL,
                                                   
            "description": ("判断用户要求的事已经全部完成，或剩下的事做不到/条件不成立时，"
                          "调用它收尾（必须附 completion_report 核销报告）。"
                          "还有没核销的事项时不许调用——做完一件就收尾与没做同罪。"),
                                               
                                                              
                                                                 
            "parameters": {
                "type": "object",
                "properties": {
                    "completion_report": {
                        "type": "string",
                                              
                                                                            
                        "description": ("逐项核销报告：把用户原话要求的事逐件列出，每件标注"
                                      "「已做（第几步）/ 条件不成立（据第几步的结果）/ 做不到"
                                      "（据第几步的结果）」——豁免同样必须写明是据第几步的真实"
                                      "结果得出的。有一件没交代就不许调用 finish。"),
                    },
                },
                "required": ["completion_report"],
            },
        },
    })
    tools.append({
        "type": "function",
        "function": {
            "name": _DECIDE_UNSUPPORTED_TOOL,
            "description": ("用户还要求了一件本循环做不到的事（如打包下载、删除文件）时，"
                          "调用它来说明是哪一件；表里四个工具能做的事不许用它——"
                          "例：「先检查更新再把数据搜来入库」里的搜来入库是 curate.search_online，"
                          "不许用它婉拒。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "verb": {
                        "type": "string",
                        "enum": unsupported,
                        "description": "做不到的那个动作（受控动词名）。",
                    },
                },
                "required": [],
            },
        },
    })
    return tools, name_to_verb


                                                        
_DECIDE_TOOL_SPECS, _DECIDE_TOOL_NAME_TO_VERB = _build_decide_tool_specs()

                                                                               
 
                                                       
                                                  
                                  

                                             
                                                                  
                                                  
                                                         
                                                             
                                 
                                                                   
                                                 
                                                               
                                                
                                                                
_SUITE_LOOP_VERBS: dict[str, tuple[str, ...]] = {
    "search": tuple(v for v in ("rank", "rerank", "results.show", "ask.relax",
                                "search.rerun",
                                "curate.db_status",
                                "compare.datasets", "cite.export",
                                "compat.find", "fair.check")
                    if v in LOOP_TOOLS),
    "action": tuple(v for v in ("curate.check_updates", "curate.search_online",
                                "curate.sync_updates", "curate.db_status",
                                "curate.rollback",
                                "compare.datasets", "cite.export",
                                "compat.find", "fair.check")
                    if v in LOOP_TOOLS),
    "general": tuple(LOOP_TOOLS),
                                     
    "rescue": tuple(v for v in ("search.rerun",) if v in LOOP_TOOLS),
                                                       
                                                                    
                                       
                                                    
    "read": tuple(v for v in LOOP_TOOLS if v not in _WRITE_LOOP_TOOLS),
}

                                                     
                                                          
_DECIDE_TOOL_SPECS_BY_SUITE: dict[str, list[dict]] = {}
_DECIDE_TOOL_NAME_TO_VERB_BY_SUITE: dict[str, dict[str, str]] = {}
for _suite, _suite_verbs in _SUITE_LOOP_VERBS.items():
    _allowed = set(_suite_verbs)
    if _suite != "rescue":
        _allowed.add("route.request")
    _DECIDE_TOOL_SPECS_BY_SUITE[_suite] = [
        t for t in _DECIDE_TOOL_SPECS
        if str((t.get("function") or {}).get("name") or "")
        in ((_DECIDE_FINISH_TOOL,) if _suite == "rescue"
            else (_DECIDE_FINISH_TOOL, _DECIDE_UNSUPPORTED_TOOL))
        or _DECIDE_TOOL_NAME_TO_VERB.get(
            str((t.get("function") or {}).get("name") or "")) in _allowed
    ]
    _DECIDE_TOOL_NAME_TO_VERB_BY_SUITE[_suite] = {
        n: v for n, v in _DECIDE_TOOL_NAME_TO_VERB.items() if v in _allowed}

def _understand_suite_verbs(suite: str) -> tuple[str, ...]:
    """understand 首步投影的套件动词集（ROUTE 投影退役：search.new/refine.conditions/
    lookup.identifier 不再投影——检索由 rank/rerank agentic 覆盖；route.request 刻意
    不进任何首步面：首步没有「发现路错」可言）。none 恒在（诚实的「不是执行诉求」出口）。
    2026-08-18 四工具批：search/action 两线都装结果处理四工具（检索后追问
    「对比/引文/兼容/FAIR」在两条线都真实发生，真机分流实测见 `_SUITE_LOOP_VERBS` 注释）。"""
    if suite == "search":
        verbs = [v for v in ("rank", "rerank", "search.rerun", "curate.db_status",
                             "compare.datasets", "cite.export", "compat.find", "fair.check")
                 if v in _ap.VERB_BY_NAME]
    elif suite == "action":
                                                         
                                             
        verbs = list(_SUITE_LOOP_VERBS["action"]) + [
            s.verb for s in _ap.VERB_SPECS
            if s.kind == _ap.EXEC and s.verb not in LOOP_TOOLS]
    elif suite == "rescue":
        verbs = ["search.rerun"]
    elif suite == "read":
                                                           
        verbs = [s.verb for s in _ap.VERB_SPECS
                 if s.kind == _ap.EXEC and s.verb not in _WRITE_LOOP_TOOLS]
    else:
                                                       
        verbs = [s.verb for s in _ap.VERB_SPECS if s.kind == _ap.EXEC]
    return tuple(dict.fromkeys([*verbs, "none"]))

_SUITE_UNDERSTAND_VERBS: dict[str, tuple[str, ...]] = {
    s: _understand_suite_verbs(s) for s in _SCOPED_ROUTES}

                                                      
                                                  
                                               
_SCOPED_CORE_ZH: str = _prompt_md("loop_core.md", _LOOP_CORE_FALLBACK_ZH)
_SCOPED_DECIDE_RULES_BY_SUITE: dict[str, dict[str, str]] = {}
for _suite in _SCOPED_ROUTES:
    _suite_set = set(_SUITE_LOOP_VERBS[_suite])
    if _suite != "rescue":
        _suite_set.add("route.request")
    _table = _decide_tool_table_zh(
        tuple(v for v in _DECIDE_VERB_ORDER if v in _suite_set))
    _base = (
        _SCOPED_CORE_ZH + "\n\n"
        + _prompt_md(f"loop_{_suite}.md", _LOOP_DELTA_FALLBACK_ZH)
        + "\n\n## 可用工具（verb 只能从这张表里选）\n" + _table + "\n\n## 输出方式\n"
    )
    if _suite == "rescue":
                                                            
        _SCOPED_DECIDE_RULES_BY_SUITE[_suite] = {
            "tools": _base + (
                "- 需要换词重检 → 调用 search_rerun\n"
                "- 不需要或无法再改写 → 调用 finish（必须附 completion_report）\n"),
            "json": _base + (
                "- 需要换词重检 → 只回 search.rerun 的 JSON 对象\n"
                "- 不需要或无法再改写 → 只回 {\"done\": true}\n"),
        }
    else:
        _SCOPED_DECIDE_RULES_BY_SUITE[_suite] = {
            "tools": _base + _DECIDE_TOOLS_CHANNEL_BULLETS_ZH,
            "json": _base + _DECIDE_JSON_BULLETS_ZH,
        }

                                                         
                                                                        
                                                             
                         
_SCOPED_TOOLS_SYSTEM_ZH = (
    _ap._RULES_ZH
    + "铁律（违反任一条都是错误）：\n"
    + "".join(
        f"{i}. {body}\n" for i, body in enumerate(_ap._TOOLS_CHANNEL_RULE_BODIES_ZH, 1))
)


def _loop_slots_fingerprint(verb: str, raw_or_slots: dict) -> tuple:
    """verb + 声明槽位的归一指纹（decide 的「不许重复已执行步骤」机械比对用）。

    槽位值一律过 `_norm_source`（小写 + 去全部空白）：查重的语义是「同一件事不许做两遍」，
    LLM 换个大小写/多空一个空格（ArrayExpress → arrayexpress）并不能让同一步变成新步骤
    （2026-08-04 对抗评审坐实变体穿透：同一检查真跑两遍、账本两行）。"""
    spec = _ap.VERB_BY_NAME.get(str(verb or ""))
    names = spec.slots if spec else ()
    items = []
    for name in names:
        value = _norm_source((raw_or_slots or {}).get(name))
        if value:
            items.append((name, value))
    return str(verb or ""), tuple(sorted(items))


def _is_duplicate_step(verb: str, raw: dict, steps: list[dict]) -> bool:
    """是否与既往步同 verb 同参数（decide 的「不许重复已执行步骤」机械比对用）。

    比对集 = 成功步 + **非 network_error** 失败的步（
    W2a4 曾一律豁免失败步，把 bad_result_shape 这类确定性失败也放去重试白烧一步）：
    - network_error 是唯一真·可重试码——失败步什么都没做成，同指纹重试放行
      （check 只有 source 槽，同源重查恒同指纹，重试上界由连续失败
      处置二分天然兜底（再败即联网暂停/硬停）；
    - bad_result_shape 是确定性失败（形状不合契约），同指纹重试必败 → 照样拦截；
    - bad_param / no_candidates 的合法重试要换参，指纹天然不同，不受本闸影响；
    - 终态码（source_not_registered）的同目标重试仍由死路闸拦截，不经本闸。"""
    fp = _loop_slots_fingerprint(verb, raw)
    return any(
        _loop_slots_fingerprint(s.get("verb"), s.get("slots") or {}) == fp
        for s in steps
        if s.get("ok") or str(s.get("error_code") or "") != "network_error"
    )


def _search_coverage_violation(obj: dict, steps: list[dict]) -> str | None:
    """搜索覆盖闸（首搜成功后 decide 换措辞/加过滤再搜
    一遍同主题，rule 10 没拦住、第三步才被指纹去重拦下——白烧真联网搜索与 LLM 往返）。

    只管 `curate.search_online`；返回违规原因码（None = 放行）。判定口径（拿不准放行）：
      - 只看**同 source**（`_norm_source` 归一，空=空）的既往**成功**搜索步——首搜失败
        （ok=False）不计入覆盖（重试放行）；换 source 再搜是另一条路（放行）。
      - tokens_new ⊆ 既往 record_count>0 步的 token 并集 → "covered"：主题已被有结果的
        搜索覆盖，换措辞/加过滤不会有新结论；
      - tokens_new ⊆（上述并集 ∪ record_count==0 步的 token 并集）且同 source 成功搜索
        步数 ≥ 2 → "retry_exhausted"：零结果后只允许一次换措辞重试；
      - tokens_new 为空：既往同 source 有成功搜索 → "covered_empty"，否则放行（已在上面
        `prior` 为空时返回）。
    多主题合法用例不得误伤：「human lung 和 mouse brain 都要」第二搜 tokens 不含于第一搜
    的并集 → 子集关系不成立 → 放行。"""
    if str(obj.get("verb") or "") != "curate.search_online":
        return None
    src = _norm_source(obj.get("source"))
    prior = [s for s in steps
             if s.get("ok") and str(s.get("verb") or "") == "curate.search_online"
             and _norm_source((s.get("slots") or {}).get("source")) == src]
    if not prior:
        return None
    tokens_new = set(_keyword_content_tokens(obj.get("keywords")))
    if not tokens_new:
        return "covered_empty"
    pos: set[str] = set()
    zero: set[str] = set()
    for s in prior:
        n = (s.get("result") or {}).get("record_count")
        toks = set(_keyword_content_tokens((s.get("slots") or {}).get("keywords")))
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            pos |= toks
        elif isinstance(n, int) and not isinstance(n, bool) and n == 0:
            zero |= toks
    if tokens_new <= pos:
        return "covered"
    if len(prior) >= 2 and tokens_new <= (pos | zero):
        return "retry_exhausted"
    return None


                                                                                                    
                                                            
                                                       
_UNFINISHED_MARKERS_ZH: tuple[str, ...] = ("没做", "未做", "还没有做", "还没做", "待做")
                                                    
                                     
_UNFINISHED_EXEMPT_ZH: tuple[str, ...] = ("条件不成立", "做不到", "无法", "不需要")
                            
                                               
                                             
_DEPENDENCY_EXCUSE_ZH: tuple[str, ...] = ("前置", "前面", "前件", "该步骤", "上一步")
                                        
                                                     
                                        
                               
_DONE_MARKERS_ZH: tuple[str, ...] = ("已做",)
                         
_STEP_REF_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十]+)\s*步")
_CN_DIGITS_ZH = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9}


def _step_ref_number(text: str) -> int | None:
    """步骤号引用 → 数字：阿拉伯直取；中文数字认 一..九 与「十」组合（十/十一/二十）；
    认不出（「第一二步」式非法形）→ None——拿不出合法号码按「无合法步骤号」处理。"""
    text = text.strip()
    if text.isdigit():
        return int(text)
    if not text or any(ch not in _CN_DIGITS_ZH and ch != "十" for ch in text):
        return None
    if text.count("十") > 1:
        return None
    if "十" in text:
        left, _, right = text.partition("十")
        if len(left) > 1 or len(right) > 1:
            return None
        tens = _CN_DIGITS_ZH.get(left, 1) if left else 1
        ones = _CN_DIGITS_ZH.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS_ZH[text]
    return None


def _completion_report_veto(report: str, n_steps: int) -> tuple[str | None, str]:
    """逐行扫描 finish 的 completion_report，返回 (否决行原文, 形态码)；无否决 → (None, "")。

    三种形态（逐行判定，返回第一行否决行；行内先查「没做」族、合法豁免行不再查「已做」）：
    - "unfinished"（v3 旧闸）：命中「没做/未做/还没有做/还没做/待做」且同行无豁免词；
    - "dependency_excuse"（形态 B）：豁免词命中行同时夹带依赖借口词
      （前置/前面/前件/该步骤/上一步）——「因前置步骤失败而未做」不是合法豁免；
    - "exempt_without_step"（2026-08-08 提前收工残余）：
      豁免行的**举证责任**——「条件不成立/做不到」必须引用合法步骤号（据第几步的真实
      结果得出），空口豁免与没做同罪；
    - "done_without_step"（形态 A）：标注「已做」的行必须引用真实存在的步骤号
      （第 N 步、1 ≤ N ≤ n_steps）——没跑过的步骤不许自称已做。
    空报告/缺席 → (None, "")：拿不到核销结论时维持 fail-safe 接受（收尾方向是安全侧）。

    **只管 tool 通道的 finish**：散文 JSON 兜底档的 {"done": true} 没有 completion_report，
    照旧接受——这条不对称是刻意的：兜底档是通道异常时的保命档，不再加门槛。"""
    for raw_line in str(report or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(m in line for m in _UNFINISHED_MARKERS_ZH):
            if not any(e in line for e in _UNFINISHED_EXEMPT_ZH):
                return line, "unfinished"
            if any(d in line for d in _DEPENDENCY_EXCUSE_ZH):
                return line, "dependency_excuse"
            refs = [_step_ref_number(t) for t in _STEP_REF_RE.findall(line)]
            if not any(n is not None and 1 <= n <= n_steps for n in refs):
                return line, "exempt_without_step"
            continue                            
        if any(m in line for m in _DONE_MARKERS_ZH):
            refs = [_step_ref_number(t) for t in _STEP_REF_RE.findall(line)]
            if not any(n is not None and 1 <= n <= n_steps for n in refs):
                return line, "done_without_step"
    return None, ""


def _unfinished_business(report: str, n_steps: int = 0) -> str | None:
    """completion_report 的否决行（原文引用进回灌反馈）；无否决 → None。
    保持「报告进、否决行/None 出」的原签名语义；形态判定收在 `_completion_report_veto`
    （decide 要按形态拼反馈文案）。`n_steps` = 已完成步数，形态 A 的步骤号越界判定用。"""
    line, _shape = _completion_report_veto(report, n_steps)
    return line


def _reask_write_veto(report: str, reask_writes: list[dict]) -> tuple[str | None, str]:
    """重问写步的**强制核销复核**（重问后的写动词从
    「机械拒绝」改为「放行 + 强制核销」）：台账里每一个重问后放行的写步，finish 的
    completion_report 都必须引用其步骤号单独交代结果；缺一个 → 返回 (缺口描述, 形态码)。
    只查「步骤号引用存在性」，不查措辞——结果好坏由 LLM 自述、faithful 机械后检兜谎。"""
    refs: set[int] = set()
    for token in _STEP_REF_RE.findall(str(report or "")):
        n = _step_ref_number(token)
        if n is not None:
            refs.add(n)
    for entry in reask_writes:
        step_no = entry.get("step_no")
        if not isinstance(step_no, int) or isinstance(step_no, bool):
            continue
        if step_no not in refs:
            shown = str(entry.get("verb_zh") or entry.get("verb") or "")
            return (f"第 {step_no} 步「{shown}」（重问后放行的写操作）",
                    "reask_write_unaccounted")
    return None, ""


def _finish_veto(report: str, n_steps: int,
                 reask_writes: list[dict]) -> tuple[str | None, str]:
    """finish 核销报告的合并否决口：先跑 `_completion_report_veto` 三形态，再跑重问写步
    强制核销（`_reask_write_veto`）——decide 的首次拒收与第二次 fail-safe 判定都用
    这同一个入口，口径不分叉。"""
    line, shape = _completion_report_veto(report, n_steps)
    if not line:
        line, shape = _reask_write_veto(report, reask_writes)
    return line, shape


def _searched_topics_block_zh(steps: list[dict]) -> str:
    """decide prompt 的「已搜主题清单」段（2026-08-08 rule 10 后半句禁令的机械事实面）——
    仅收录 ok 的 search_online 步，把「搜过什么主题、搜到几条」摆在明面上，同一主题不许
    换措辞或加过滤条件再搜。没有成功搜索步 → 空串（整段不出现，两个壳同口径）。"""
    lines: list[str] = []
    for s in steps:
        if not s.get("ok") or str(s.get("verb") or "") != "curate.search_online":
            continue
        keywords = str((s.get("slots") or {}).get("keywords") or "").strip()
        count = (s.get("result") or {}).get("record_count")
        count_zh = str(count) if isinstance(count, int) and not isinstance(count, bool) else "?"
        source = str((s.get("slots") or {}).get("source") or "").strip() or "全部"
        lines.append(f"- 「{keywords}」→ 搜到 {count_zh} 条（来源：{source}）")
    if not lines:
        return ""
    return ("\n----- 已经联网搜过的主题（同一主题不许换措辞或加过滤条件再搜）-----\n"
            + "\n".join(lines))


                                 
                                                            
                                               
                        
                                           
                            
                                                     
                                               
_PENDING_COUNT_RE = re.compile(r"多少条|(?<![这那好十几多数])几条|多少数据|多少记录|库容")
_PENDING_IMPORT_RE = re.compile(r"下载|入库|拿回|拿回来|搜")


                                                        
                                                          
                                                         
                                            
                                                        
                                                          
_CURATE_EXTRA_NAMED_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Zenodo", ("Zenodo",)),
    ("refine.bio", ("refine.bio", "refinebio")),
    ("NCBI GEO", ("NCBI GEO", "GEO")),
    ("HuBMAP", ("HuBMAP",)),
    ("Broad Single Cell Portal", ("Broad Single Cell Portal", "Single Cell Portal")),
    ("EBI Single Cell Expression Atlas", ("EBI Single Cell Expression Atlas",)),
)


def _named_sources_in(utterance: str) -> list[str]:
    """原话点名的来源集合（保序去重）：别名匹配 + 受控规范名逐字 + curate 补充表三趟
    （逐字趟与 `_named_source_violation` 的豁免同口径：词表刻意不收裸「encode」
    这类普通英文词，但用户原样写出全大写 ENCODE 就是点名）。
    从 `_pending_hints_block_zh` 规则 1 抽出的共享真源——
    未决提示与复杂度路由吃同一份「点名」口径，防两份拷贝漂移。"""
    named: list[str] = []
    for span in _sr.source_alias_spans(utterance):
        source = str(span["source"])
        if source not in named:
            named.append(source)
    for source, _aliases in _sr.SOURCE_ALIASES:
        if str(source) in (utterance or "") and str(source) not in named:
            named.append(str(source))
    for label, forms in _CURATE_EXTRA_NAMED_SOURCES:
        if any(f in (utterance or "") for f in forms) and label not in named:
            named.append(label)
    return named


def _pending_hints_block_zh(utterance: str, steps: list[dict]) -> str:
    """decide prompt 的「未决事项机械提示」段（**提示不是闸**——不拦截任何东西）。
    只报机械可判定的三条（两条现成真源 + 实录步），拿不准绝不报；三条都无命中 → 空串
    （整段不出现）：
      1. **点名源未触碰**：原话点名来源集合（`_named_sources_in`）− steps 实际触碰
         来源集（`_step_touched_sources`）非空 → 逐来源一行；
      2. **库容问句缺 db_status**：原话命中库容问法且没有 ok 的 db_status 步；
      3. **检出未入库**：存在 ok 的 check_updates 步其某 online 源 new_count>0，且原话有
         下载/入库/搜索诉求（含「同步」二字时**不报**——sync 本身就是入库路径；否定极性
         豁免见 `_import_hint_request`），且没有任何 ok 的 search_online/sync_updates 步。
    规则 1 刻意不过滤失败步：失败也算「处理过」的一种（重试与否由失败语义闸管），
    拿不准的场景一律不报，保住「提示零谎报」的可信度。"""
    lines: list[str] = []
                 
    touched = _step_touched_sources(steps)
    for source in _named_sources_in(utterance):
        if source not in touched:
            lines.append(f"- 原话点名的来源「{source}」还没有任何一步处理过")
                          
    if _PENDING_COUNT_RE.search(utterance or "") and not any(
            s.get("ok") and str(s.get("verb") or "") == "curate.db_status" for s in steps):
        lines.append("- 原话要求说明库里条数，但还没有执行过 db_status")
                
    found_new = any(
        s.get("ok") and str(s.get("verb") or "") == "curate.check_updates"
        and any(
            isinstance(e, dict) and str(e.get("mode") or "") == "online"
            and isinstance(e.get("new_count"), int) and not isinstance(e.get("new_count"), bool)
            and e.get("new_count") > 0
            for e in ((s.get("result") or {}).get("sources") or []))
        for s in steps)
    imported = any(
        s.get("ok") and str(s.get("verb") or "") in ("curate.search_online", "curate.sync_updates")
        for s in steps)
    if (found_new and not imported and "同步" not in (utterance or "")
            and _import_hint_request(utterance)):
        lines.append("- 检查发现了疑似新增，但「检出」不等于「入库」：还没有执行搜索/同步入库步")
    if not lines:
        return ""
    return ("\n----- 机械提示：以下事项可能还没做（逐项核对，确实做了就忽略）-----\n"
            + "\n".join(lines)
                                                  
                                                      
            + "\n核对完仍按上面的规则回答：调用恰好一个工具，或调用 finish 并填好核销报告。")


def _import_hint_request(utterance: str) -> bool:
    """提示层规则 3 的入库诉求**极性判定**（2026-08-15 ，与已修硬闸
    `_import_hard_request` 同思路、复用文件内既有 `_DENIAL_MORPH_RE` 语素表）：
    逐命中回看**同小句**前缀，窗口内有否定语素 → 该命中不计入入库诉求。
    「有新增也不要入库」是**拒绝**入库，此前裸子串词表把它当成诉求、提示模型去入库——
    提示不是闸（危害低），但「提示零谎报」的可信度同样守。粒度与硬闸侧一致：
    一否定一肯定的混合句（「别重复入库，有新增就下载」）不被整句豁免。"""
    text = str(utterance or "")
    for m in _PENDING_IMPORT_RE.finditer(text):
        head = text[:m.start()]
        head = head[max(head.rfind(p) for p in "，。；！？,.;!?\n") + 1:]
        if not _DENIAL_MORPH_RE.search(head):
            return True
    return False


                                                                         
 
                      
                                             
                                                           
                                                       
                                                      
                                                         

_CHECKLIST_MAX_ITEMS = 8                                       
_CHECKLIST_VERBS = (                                           
    "curate.check_updates", "curate.search_online", "curate.sync_updates",
    "curate.db_status", "rank", "unsupported")
_CHECKLIST_ANCHOR_MIN = 4                        

                                                 
                                                     
                                
_SEARCH_SETTLE_VERBS: tuple[str, ...] = ("rank", "rerank", "search.rerun")


def _is_search_settle_step(s: dict) -> bool:
    """检索半核销的**证据步判定**（2026-09-14 YOLO 批把消费口径从裸元组收进本助手）：
    ok ∧（verb ∈ `_SEARCH_SETTLE_VERBS`，或 verb=="ask.relax" 且真发起了询问
    （result.asked 为真）——混合句的「找数据」一半在弹卡等用户作答时也算有交代，
    不许 finish 闸把「等作答」误杀成「没做」；被闸拒绝（asked=False）的提议不算核销）。"""
    if not bool(s.get("ok")):
        return False
    v = str(s.get("verb") or "")
    if v in _SEARCH_SETTLE_VERBS:
        return True
    if v == "ask.relax":
        r = s.get("result") if isinstance(s.get("result"), dict) else {}
        return bool(r.get("asked"))
    return False

                                             
                                                
                                              
_WIDTH_FOLD = str.maketrans(
    "，。；！？（）【】、：‘’“”　",
    ",.;!?()[]、:" + "''" + '""' + " ")


def _agent_status_block_zh(state: "_AgentState", *, steps: list[dict], moratorium: bool,
                           ban_verbs: frozenset[str] = frozenset()) -> str:
    """decide 双壳恒注入的**执行状态栏**（2026-08-08 epub 第2章状态栏思想）。

    三条铁律（书 2.6）：①**代码维护**——全部字段从图状态确定性现算，模型不许统计；
    ②有损投影要谨慎选维度——只放真实机械状态（步数/失败/联网暂停/已搜主题/
    finish_vetoes/reask_writes/清单未决；虚构的「统一重问预算」不进——repairs/finish_vetoes/
    decide 局部 reasked 是三份独立预算）；③读数配操作策略——「剩余=1 时必须 finish」
    只在可达且真实时出现（MAX_STEPS 提前返回在 prompt 构建之前，「6/6 到顶」永不可达、不写）。
    understand 不注：首步无循环状态，`_context_zh` 已含结果态——注入只会重复。
    """
    n = len(steps)
    remaining = max(0, MAX_STEPS - n)
    failed = sum(1 for s in steps if not s.get("ok"))
    searched = sum(1 for s in steps if s.get("ok") and s.get("verb") == "curate.search_online")
    writes = _write_steps_used(steps)
    records = _write_records_used(steps)
    vetoes = int(state.get("finish_vetoes") or 0)
    reask = len(state.get("reask_writes") or [])
    parts = [
        f"已执行步数 {n}（最多 {MAX_STEPS} 步）",
        f"剩余步数 {remaining}",
        f"写步数 {writes}（最多 {MAX_WRITE_STEPS} 次写）",
        f"已写入 {records} 条（最多 {MAX_WRITE_RECORDS} 条）",
        f"失败步数 {failed}",
        f"联网暂停 {'是' if moratorium else '否'}",
        f"已搜主题 {searched}",
        f"finish 核销被拒 {vetoes}",
        f"重问放行写步 {reask}",
    ]
    if ban_verbs:
        names = "、".join(sorted(
            (_ap.VERB_BY_NAME.get(v).zh if _ap.VERB_BY_NAME.get(v) else v) for v in ban_verbs))
        parts.append(f"二连败禁提 {names}")
    checklist = list(state.get("checklist") or [])
    item_lines: list[str] = []
    if checklist:
        declined_zh = str(state.get("declined_zh") or "")
        item_states = _checklist_item_states(checklist, steps, declined_zh)
        missing_n = sum(1 for it in item_states if it["status"] == "missing")
        parts.append(f"清单未决 {missing_n}")
                                                             
                                   
        for it in item_states:
            if it["status"] == "done":
                mark = f"已做（第{it['step_no']}步）" if it.get("step_no") else "已做"
            elif it["status"] == "exempt":
                mark = "豁免（零新增）"
            elif it["status"] == "declined":
                mark = "已婉拒（表外）"
            else:
                mark = "未做"
            item_lines.append(f"[{it['task_id']}] {mark} {it['text']}")
    cap_lines: list[str] = []
    caps = list(state.get("required_capabilities") or [])
    if caps:
        cap_states = _capability_item_states(caps, steps,
                                             str(state.get("declined_zh") or ""))
        cap_missing = sum(1 for it in cap_states if it["status"] == "missing")
        parts.append(f"混合诉求能力未决 {cap_missing}")
        for it in cap_states:
            if it["status"] == "done":
                mark = f"已做（第{it['step_no']}步）" if it.get("step_no") else "已做"
            elif it["status"] == "exempt":
                mark = "豁免（零新增，前提不成立）"
            elif it["status"] == "declined":
                mark = "已交代（本环做不到）"
            else:
                mark = "未做"
            cap_lines.append(f"[{it['capability']}] {mark} {it['label_zh']}")
    intent_lines: list[str] = []
    intent_cl = list(state.get("intent_checklist") or [])
    if intent_cl:
                                                       
                                                        
        intent_states = _intent_checklist_item_states(intent_cl, steps)
        intent_missing = sum(1 for it in intent_states if it["status"] == "missing")
        parts.append(f"句内动作未决 {intent_missing}")
        for it in intent_states:
            label = it["verb_zh"] or it["verb"]
            if it["status"] == "done":
                mark = f"已做（第{it['step_no']}步）"
            elif it["status"] == "delegated":
                mark = "已交前端（环内别再做）"
            elif it["status"] == "accounted":
                mark = "已在核销报告交代"
            else:
                mark = "未做"
            frag = f"「{it['quoted'][:20]}」" if it.get("quoted") else ""
            intent_lines.append(f"[{it['verb']}] {mark} {label}{frag}")
    lines = ["\n----- 执行状态（系统机械账本，实时）-----", "；".join(parts) + "。"]
    if item_lines:
        lines.append("清单逐项（代码对账，勿凭印象改判）：" + "；".join(item_lines) + "。")
    if cap_lines:
        lines.append("混合诉求能力逐项（代码对账，勿凭印象改判）："
                     + "；".join(cap_lines) + "。")
    if intent_lines:
        lines.append("句内动作逐项（代码对账，勿凭印象改判；全部核销后才许 finish）："
                     + "；".join(intent_lines) + "。")
    if remaining == 1:
        lines.append("剩余步数 = 1：下一步执行后就必须调用 finish——这是最后一次执行机会。")
    return "\n".join(lines) + "\n"


def _fold_width(text: Any) -> str:
    return str(text or "").translate(_WIDTH_FOLD)


def _parse_checklist(payload: Any, utterance: str) -> tuple[list[dict], int]:
    """清单 LLM 应答 → 合法条目列表 + 剔除数。纯函数（形状/枚举/锚点三层机械校验）。

    校验（逐条，不合规剔除计数——宁缺毋滥，烂条目比对账误判安全）：
    - 形状：dict 且 text/anchor/expect_verb 三键齐备；
    - expect_verb ∈ `_CHECKLIST_VERBS`（受控枚举）；
    - anchor 全半角归一后必须是 utterance 的**子串**且 ≥4 字（评审：anchor 机械可验
      幻觉锚点直接剔除——这是「专名幻觉查形状」的可机械实现面）；
    - 条目里的来源名经 `_named_sources_in` 受控词表提取（词表外的来源写法不进 sources，
      对账时按无来源条目处理——不剔除整条，锚点已保证条目植根原话）。
    """
    items = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None)
    if not isinstance(items, list):
        return [], 0
    utterance_folded = _fold_width(utterance)
    out: list[dict] = []
    dropped = 0
    for raw in items[: _CHECKLIST_MAX_ITEMS * 2]:                     
        if not isinstance(raw, dict):
            dropped += 1
            continue
        text = str(raw.get("text") or "").strip()
        anchor = str(raw.get("anchor") or "").strip()
        expect_verb = str(raw.get("expect_verb") or "").strip()
        if not text or len(_fold_width(anchor)) < _CHECKLIST_ANCHOR_MIN:
            dropped += 1
            continue
        if _fold_width(anchor) not in utterance_folded:
            dropped += 1
            continue
        if expect_verb not in _CHECKLIST_VERBS:
            dropped += 1
            continue
        out.append({"task_id": f"t{len(out) + 1}", "text": text[:60], "anchor": anchor,
                    "expect_verb": expect_verb,
                                                              
                                                           
                                              
                    "sources": _named_sources_in(text + " " + anchor)})
                                             
                                                       
                                               
    valid_truncated = max(0, len(out) - _CHECKLIST_MAX_ITEMS)
    return out[:_CHECKLIST_MAX_ITEMS], dropped + valid_truncated


def _step_ok_verb(step: dict, verb: str) -> bool:
    return bool(step.get("ok")) and str(step.get("verb") or "") == verb


def _step_source(step: dict) -> str:
    """步骤槽位里的来源（规范名口径，空串=没填/全量）。"""
    slots = step.get("slots") or {}
    canon = _canonical_source(slots.get("source"))
    return str(canon or "").strip()


def _step_covered_sources(step: dict) -> set[str]:
    """步骤实际覆盖的来源集合（规范名口径）：
    - slots.source 填了 → 单源集合；
    - 空 source 的 check/sync 步 → 从 result.sources 的条目还原（评审：
      全来源成功步不许被误杀——空槽位不等于「没覆盖」，结果里写着覆盖了谁）。"""
    direct = _step_source(step)
    if direct:
        return {direct}
    covered: set[str] = set()
    for e in ((step.get("result") or {}).get("sources") or []):
        if not isinstance(e, dict):
            continue
        name = str(_canonical_source(e.get("source")) or "") or str(e.get("label") or "")
        if name:
            covered.add(name)
    return covered


def _task_settled_by(task: dict, steps: list[dict]) -> bool:
    """单条清单的动词级对账（零信任：只认 steps 实录，不认模型文本）。

    评审修复：**按来源覆盖核销**——条目点了 N 个来源，就要 N 个来源
    各自都有 ok 同动词步骤覆盖（任一来源型「一步核销全任务」的洞已堵；步骤侧来源
    判定走 `_step_covered_sources`，空槽位全量步按结果还原覆盖集）。
    条目没点名来源 → 任一 ok 同动词步即核销（对账只验「做了这件事」）。
    条件豁免（expect=search_online）：**每个**点名来源都要有一个 ok check 步给出
    零新增证据（new_candidates 是空列表**且** new_count 严格整数 0——双重严格且，
    任一来源缺证据/有新增都不豁免）；未点名来源 → 任一 ok check 零新增即豁免。
    失败步/无关来源的 check 不出具豁免资格。
    expect=rank：混合句的检索半——任一 ok 的 rank/rerank/search.rerun 步
    即核销；ask.relax 真发起询问（弹卡等用户作答）也算有交代
    （`_is_search_settle_step` 单点口径；检索覆盖全库，**忽略来源对账**）。"""
    expect = str(task.get("expect_verb") or "")
    wanted_sources = set(task.get("sources") or [])

    if expect == "rank":
        return any(_is_search_settle_step(s) for s in steps)
    if not wanted_sources:
        if any(_step_ok_verb(step, expect) for step in steps):
            return True
    else:
        remaining = set(wanted_sources)
        for step in steps:
            if not _step_ok_verb(step, expect):
                continue
            remaining -= _step_covered_sources(step)
        if not remaining:
            return True
    if expect == "curate.search_online":
        zero_sources: set[str] = set()                   
        for step in steps:
            if not _step_ok_verb(step, "curate.check_updates"):
                continue
            for e in ((step.get("result") or {}).get("sources") or []):
                if not isinstance(e, dict) or str(e.get("mode") or "") != "online":
                    continue
                cands = e.get("new_candidates")
                new_count = e.get("new_count")
                if (isinstance(cands, list) and not cands) and new_count == 0 \
                        and isinstance(new_count, int) and not isinstance(new_count, bool):
                    name = str(_canonical_source(e.get("source")) or "") \
                        or str(e.get("label") or "")
                    if name:
                        zero_sources.add(name)
        if not wanted_sources:
            return bool(zero_sources)
        return zero_sources >= wanted_sources
    return False


def _checklist_item_states(checklist: list[dict], steps: list[dict],
                           declined_zh: str = "") -> list[dict]:
    """清单逐项的**代码现算**状态（2026-08-09 调研-长程agent批 候选4：核销状态栏逐项化——
    Anthropic harness「JSON-over-Markdown + 全 false 初始化」与 Manus 尾部复述的机械化版；
    模型只读不写）。返回 [{task_id, text, status, step_no}]：
    status ∈ done(第N步) / exempt(零新增豁免) / declined(表外已婉拒) / missing(未做)。
    判定复用 `_task_settled_by` 与 `_checklist_unsettled` 的同一套零信任对账，绝不另立口径。"""
    out: list[dict] = []
    declined_pool = 1 if str(declined_zh or "").strip() else 0
    for task in checklist:
        tid = str(task.get("task_id") or "")
        text = str(task.get("text") or "")
        expect = str(task.get("expect_verb") or "")
        if expect == "unsupported":
            if declined_pool > 0:
                declined_pool -= 1
                out.append({"task_id": tid, "text": text, "status": "declined", "step_no": None})
            else:
                out.append({"task_id": tid, "text": text, "status": "missing", "step_no": None})
            continue
                                                 
                                                                         
                                                         
        if expect == "rank":
            step_no = next((i for i, s in enumerate(steps, 1)
                            if _is_search_settle_step(s)), None)
        else:
            step_no = next((i for i, s in enumerate(steps, 1)
                            if _step_ok_verb(s, expect)), None)
        if _task_settled_by(task, steps):
            if expect == "curate.search_online" and step_no is None:
                out.append({"task_id": tid, "text": text, "status": "exempt", "step_no": None})
            else:
                out.append({"task_id": tid, "text": text, "status": "done", "step_no": step_no})
        else:
            out.append({"task_id": tid, "text": text, "status": "missing", "step_no": None})
    return out


def _checklist_unsettled(checklist: list[dict], steps: list[dict],
                         declined_zh: str = "") -> list[dict]:
    """清单对账器：返回**未决**条目列表（[{task_id, text, reason}]），空 = 全部核销。

    reason 码：`step_missing`（没有匹配的 ok 步骤）/ `unsupported_unaddressed`
    （表外事项没有 decide 婉拒记录兜底）。条件豁免在 `_task_settled_by` 内判定。"""
    unsettled: list[dict] = []
                                                      
                                              
                                                            
                                            
    declined_pool = 1 if str(declined_zh or "").strip() else 0
    for task in checklist:
        expect = str(task.get("expect_verb") or "")
        if expect == "unsupported":
            if declined_pool > 0:
                declined_pool -= 1
                continue
            unsettled.append({"task_id": task["task_id"], "text": task["text"],
                              "reason": "unsupported_unaddressed"})
            continue
        if not _task_settled_by(task, steps):
            unsettled.append({"task_id": task["task_id"], "text": task["text"],
                              "reason": "step_missing"})
    return unsettled


def _capability_item_states(caps: list[dict], steps: list[dict],
                            declined_zh: str = "") -> list[dict]:
    """混合诉求能力账的逐项**代码现算**状态（与 `_checklist_item_states`
    同哲学：零信任 steps 实录对账，模型只读不写）。

    返回 [{capability, label_zh, verbs, status, step_no}]，status ∈
    done(第N步) / exempt(条件不成立天然豁免) / declined(环做不到已交代) / missing(未做)。
    核销口径：
    - verbs 非空：任一 ok 步动词 ∈ verbs 即 done（证据步 = 首个匹配步号）；
    - verbs 空（action.generic，本环无对应工具）：靠 decide 婉拒记录交代（pool=1，
      与清单 unsupported 同口径——每轮至多婉拒一件，pool 不膨胀）；
    - action.import 的**零新增豁免**：存在 ok 的 curate.check_updates 步、且没有任何
      ok check 步报出新增（new_count>0 或非空 new_candidates）→ 入库前提不成立，
      天然豁免（「有新的就入库」而确实没有新的，入库半不算欠账）。"""
    out: list[dict] = []
    declined_pool = 1 if str(declined_zh or "").strip() else 0
    for cap in caps:
        cid = str(cap.get("capability") or "")
        label = str(cap.get("label_zh") or "")
        verbs = [str(v) for v in (cap.get("verbs") or [])]
        if not verbs:
            if declined_pool > 0:
                declined_pool -= 1
                out.append({"capability": cid, "label_zh": label, "verbs": verbs,
                            "status": "declined", "step_no": None})
            else:
                out.append({"capability": cid, "label_zh": label, "verbs": verbs,
                            "status": "missing", "step_no": None})
            continue
        step_no = next((i for i, s in enumerate(steps, 1)
                        if bool(s.get("ok")) and str(s.get("verb") or "") in verbs), None)
        if step_no is not None:
            out.append({"capability": cid, "label_zh": label, "verbs": verbs,
                        "status": "done", "step_no": step_no})
            continue
        if cid == "action.import" and _import_precondition_absent(steps):
            out.append({"capability": cid, "label_zh": label, "verbs": verbs,
                        "status": "exempt", "step_no": None})
            continue
        out.append({"capability": cid, "label_zh": label, "verbs": verbs,
                    "status": "missing", "step_no": None})
    return out


def _import_precondition_absent(steps: list[dict]) -> bool:
    """action.import 零新增豁免的机械判定：有 ok check 步 ∧ 全部 ok check 步零新增。
    严格口径与 `_task_settled_by` 的豁免段一致（new_count 严格整数 0 ∧ new_candidates
    空列表；bool 不算 int）。"""
    saw_check = False
    for step in steps:
        if not _step_ok_verb(step, "curate.check_updates"):
            continue
        saw_check = True
        for e in ((step.get("result") or {}).get("sources") or []):
            if not isinstance(e, dict):
                continue
            cands = e.get("new_candidates")
            new_count = e.get("new_count")
            has_new = (isinstance(cands, list) and bool(cands)) or (
                isinstance(new_count, int) and not isinstance(new_count, bool)
                and new_count > 0)
            if has_new:
                return False
    return saw_check


def _capabilities_unsettled(caps: list[dict], steps: list[dict],
                            declined_zh: str = "") -> list[dict]:
    """能力账对账器：返回**未决**能力项 [{capability, label_zh, verbs, reason}]，空 = 全部核销。
    reason 码：`step_missing`（没有匹配的 ok 步骤）/ `generic_unaddressed`
    （环做不到的事收尾前没有明确交代）。"""
    unsettled: list[dict] = []
    for it in _capability_item_states(caps, steps, declined_zh):
        if it["status"] == "missing":
            unsettled.append({
                "capability": it["capability"], "label_zh": it["label_zh"],
                "verbs": it["verbs"],
                "reason": "generic_unaddressed" if it["capability"] == "action.generic"
                else "step_missing"})
    return unsettled


def _intent_checklist_item_states(intent_checklist: list[dict], steps: list[dict],
                                  report: str = "") -> list[dict]:
    """句内动作清单（「不少于我」下限合同，2026-09-01）逐项**代码现算**状态——
    与 `_checklist_item_states`/`_capability_item_states` 同哲学：零信任 steps 实录
    对账，模型只读不写。

    返回 [{verb, verb_zh, quoted, plane, status, step_no}]：
    - plane="frontend"（前端直派面：已交前端接力）→ status 恒 "delegated"，环内免核销；
    - plane="inloop"：存在 ok=True 且动词匹配的步 → done（证据步=首个匹配步号）；
      否则核销报告（finish 的 completion_report）点名它（verb_zh 或 quoted 字面
      子串，全角折叠口径）→ "accounted"（模型已如实交代，例如条件不成立/做不到）；
      两皆无 → "missing"。"""
    report_folded = str(report or "").translate(_WIDTH_FOLD)
    out: list[dict] = []
    for item in intent_checklist:
        if not isinstance(item, dict):
            continue
        verb = str(item.get("verb") or "")
        verb_zh = str(item.get("verb_zh") or "")
        quoted = str(item.get("quoted") or "")
        plane = str(item.get("plane") or "inloop")
        if plane == "frontend":
            out.append({"verb": verb, "verb_zh": verb_zh, "quoted": quoted,
                        "plane": plane, "status": "delegated", "step_no": None})
            continue
        step_no = next((i for i, s in enumerate(steps, 1)
                        if _step_ok_verb(s, verb)), None) if verb else None
        if step_no is not None:
            out.append({"verb": verb, "verb_zh": verb_zh, "quoted": quoted,
                        "plane": plane, "status": "done", "step_no": step_no})
            continue
        named = any(frag and frag.translate(_WIDTH_FOLD) in report_folded
                    for frag in (verb_zh, quoted))
        out.append({"verb": verb, "verb_zh": verb_zh, "quoted": quoted,
                    "plane": plane,
                    "status": "accounted" if named else "missing",
                    "step_no": None})
    return out


def _intent_checklist_unsettled(intent_checklist: list[dict], steps: list[dict],
                                report: str = "") -> list[dict]:
    """句内动作清单对账器：返回**未决**项（status=="missing" 的子集），空 = 全部核销。
    缺项语义：既没有成功步骤、收尾报告也没点名交代——「做一个勾一个」的机械检出。"""
    return [it for it in _intent_checklist_item_states(intent_checklist, steps, report)
            if it["status"] == "missing"]
    "把用户这句话拆成「要做的几件事」的清单。输出一个 JSON 数组，每项：\n"
_CHECKLIST_PROMPT_ZH = (
    '{"text": "这件事的简明中文（≤40字）", "anchor": "原话里对应这句话的逐字片段（≥4字，'
    '必须逐字摘自原话，不许改写）", "expect_verb": 预期执行动作}。\n'
    "expect_verb 只能取：curate.check_updates（检查来源更新）/ curate.search_online"
    "（联网搜索入库）/ curate.sync_updates（检查并同步入库）/ curate.db_status（汇报库里"
    "条数）/ rank（在本地库检索数据）/ unsupported（本工具做不到的事）。\n"
    "规则：宁少勿多，只拆原话明确说了的事；同一件事只列一次；只输出 JSON 数组，无其它文字。")


def _task_checklist_call(chat_model: Any, utterance: str,
                         usage_sink: list | None = None,
                         feedback: str = "") -> tuple[list[dict], int, str]:
    """清单轻量产出：一次 chat 调用拆事项 → （合法条目, 剔除数, 失败原因)。

    失败原因空串 = 成功（条目可为空——空清单由调用方按 unavailable 处置）；
    非空 = 调用异常或应答不可解析（调用方走一次有界 repair 的决策素材，repair 时把
    失败原文经 `feedback` 回灌）。只产清单不做任何执行决策——核销判定零信任模型，
    全在 `_checklist_unsettled`。"""
    from langchain_core.messages import HumanMessage

    prompt = (_CHECKLIST_PROMPT_ZH + "\n----- 用户原话 -----\n" + str(utterance or ""))
    if feedback:
        prompt += ("\n\n你上一次的回答没能用：\n" + feedback
                   + "\n请按上面的格式重新输出（只输出 JSON 数组）。")
    try:
        answer = chat_model.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        return [], 0, type(exc).__name__
    if usage_sink is not None:
        rec = _usage_record(answer, "checklist")
        if rec is not None:
            usage_sink.append(rec)
                                                                   
                                         
    text = _message_text(answer).strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
                                                 
        left, right = text.find("["), text.rfind("]")
        if 0 <= left < right:
            try:
                payload = json.loads(text[left:right + 1])
            except (ValueError, TypeError):
                payload = None
    if payload is None:
        return [], 0, "unparseable"
    tasks, dropped = _parse_checklist(payload, utterance)
    if not tasks:
                                                   
                                                           
        return [], dropped, "empty" if not dropped else "all_dropped"
    return tasks, dropped, ""


                                       
                                         
_VETO_TEACHING_SUFFIX = {
    "done_without_step": "——「已做」必须写明是第几步的结果（步骤号不许超过已完成的步数）",
    "dependency_excuse": "——彼此独立的事不许拿前件失败当理由",
    "exempt_without_step": "——「条件不成立/做不到」必须写明是据第几步的结果得出的",
    "reask_write_unaccounted": "——重问后放行的写操作，必须写明它的步骤号并单独交代结果",
    "checklist_unsettled": "——清单里的这件事没有对应的成功步骤",
    "capability_unsettled": "——混合诉求里的这一半没有对应的成功步骤（或明确交代做不到）",
    "intent_unsettled": "——句内交代的这个动作没有对应的成功步骤；做不了就在核销报告里点名它并说明原因",
    "unfinished": "——你的核销报告里写着还有没做的事",
}

                                                           
                                                     
                                      
_PENDING_VETO_FORCED_VERB: dict[str, tuple[str, str]] = {
    "pending_count_query": ("curate.db_status", "原话要求说明库里条数"),
    "pending_new_not_imported": ("curate.search_online", "检出疑似新增但还没搜来入库"),
    "pending_source_untouched": ("curate.check_updates", "点名的来源还没检查"),
}


def _finish_veto_all(report: str, n_steps: int, reask_writes: list[dict],
                     checklist: list[dict], steps: list[dict],
                     declined_zh: str, utterance: str,
                     capabilities: tuple | list = (),
                     intent_checklist: tuple | list = ()) -> list[tuple[str, str]]:
    """finish 否决的**聚合口**（分次否决会耗尽重试额度）——
    一次返回全部 (否决描述, 形态码)，去重、稳定排序。调用方把整张列表一次回灌。

    汇聚六路（原文本闸与重问写步闸维持单条语义，清单对账与 pending 硬闸可多条）：
    - `_completion_report_veto` 三形态（报告文本自认的缺口）；
    - `_reask_write_veto`（重问写步未单独交代）；
    - 清单对账 `_checklist_unsettled`（有清单时；形态码 checklist_unsettled）；
    - 混合诉求能力账 `_capabilities_unsettled`（机械闸产出 capabilities 时
      形态码 capability_unsettled）——混合句只做一半不许收尾；
    - 句内动作清单对账 `_intent_checklist_unsettled`（2026-09-01「不少于我」下限合同，
      turn 枚举探测注入时；形态码 intent_unsettled）——枚举出的每个环内动作须有归宿：
      成功步骤或报告点名交代，缺一即拒收；
    - pending 硬闸 `_pending_violations`（机械可判的未决事实升闸；码见各规则）。
    `capabilities`/`intent_checklist` 为关键字默认参数（缺省空 = 旧调用零变化）。"""
    out: list[tuple[str, str]] = []
    line, shape = _completion_report_veto(report, n_steps)
    if line:
        out.append((line, shape))
    line2, shape2 = _reask_write_veto(report, reask_writes)
    if line2:
        out.append((line2, shape2))
    if checklist:
        for item in _checklist_unsettled(checklist, steps, declined_zh):
            text = str(item.get("text") or "")[:40]
            if str(item.get("reason") or "") == "unsupported_unaddressed":
                out.append((f"清单「{text}」是做不到的事，但收尾前没有交代过它",
                            "checklist_unsettled"))
            else:
                out.append((f"清单「{text}」没有对应的成功步骤", "checklist_unsettled"))
    if capabilities:
        for item in _capabilities_unsettled(list(capabilities), steps, declined_zh):
            label = str(item.get("label_zh") or "")[:40]
            if str(item.get("reason") or "") == "generic_unaddressed":
                out.append((f"混合诉求里的「{label}」这一半本环做不到，但收尾前没有明确交代",
                            "capability_unsettled"))
            else:
                out.append((f"混合诉求里的「{label}」这一半没有对应的成功步骤",
                            "capability_unsettled"))
    if intent_checklist:
        for item in _intent_checklist_unsettled(list(intent_checklist), steps, report):
            label = str(item.get("verb_zh") or item.get("verb") or "")[:40]
            out.append((f"句内交代的「{label}」没有对应的成功步骤，核销报告也没点名交代它",
                        "intent_unsettled"))
    out.extend(_pending_violations(utterance, steps))
                                         
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for text, code in out:
        if text in seen:
            continue
        seen.add(text)
        deduped.append((text[:120], code))
    return deduped[:_CHECKLIST_MAX_ITEMS]


def _pending_violations(utterance: str, steps: list[dict]) -> list[tuple[str, str]]:
    """pending 机械判定**升硬闸**（原 `_pending_hints_block_zh` 的三条提示）——
    判定口径与提示段共享同一份真源（本函数），提示段只做格式化。

    升闸收窄（v2.1 裁决）：规则 3 的入库诉求词表**剔「搜」**——「搜搜有没有新数据」是
    检索语境不是入库诉求，提示层误报只是误导，闸层误报是误杀。规则 1/2 口径不变
    （失败步算「处理过」——重试与否归失败语义闸管，不在本闸重复执法）。
    规则 3 另有否定极性豁免（`_import_hard_request`）：
    「有新增也不要入库」是拒绝入库，不许被词表子串当成入库诉求。"""
    out: list[tuple[str, str]] = []
    touched = _step_touched_sources(steps)
    for source in _named_sources_in(utterance):
        if source not in touched:
            out.append((f"原话点名的来源「{source}」还没有任何一步处理过",
                        "pending_source_untouched"))
    if _PENDING_COUNT_RE.search(utterance or "") and not any(
            _step_ok_verb(s, "curate.db_status") for s in steps):
        out.append(("原话要求说明库里条数，但还没有执行过 db_status", "pending_count_query"))
                                          
                                  
    new_sources: set[str] = set()
    for s in steps:
        if not _step_ok_verb(s, "curate.check_updates"):
            continue
        for e in ((s.get("result") or {}).get("sources") or []):
            if (isinstance(e, dict) and str(e.get("mode") or "") == "online"
                    and isinstance(e.get("new_count"), int)
                    and not isinstance(e.get("new_count"), bool)
                    and e.get("new_count") > 0):
                name = str(_canonical_source(e.get("source")) or "") or str(e.get("label") or "")
                if name:
                    new_sources.add(name)
    imported_sources: set[str] = set()
    for s in steps:
        if _step_ok_verb(s, "curate.search_online") or _step_ok_verb(s, "curate.sync_updates"):
            imported_sources |= _step_covered_sources(s)
    missing_import = new_sources - imported_sources
    if (missing_import and "同步" not in (utterance or "")
            and _import_hard_request(utterance)):
        out.append(("检查发现疑似新增且原话要求入库，但 "
                    + "、".join(sorted(missing_import)) + " 还没有执行搜索/同步入库步",
                    "pending_new_not_imported"))
    return out


                                                                        
_PENDING_IMPORT_HARD_RE = re.compile(r"下载|入库|拿回|拿回来")

                                              
                                                 
                                                
                                           
                                                   
_PENDING_IMPORT_DENY_MORPH_RE = re.compile(
    r"不要|不用|不许|不可|不再|先不|不会|无需|无须|不必|不准|甭|勿|切勿|禁止|拒绝"
    r"|(?<!分)别(?![人的])")


def _import_hard_request(utterance: str) -> bool:
    """原话是否含**非否定**的入库诉求（pending 硬闸规则 3 的极性判定）：
    逐个命中回看**同小句**前缀（最后一个句读之后），窗口内有否定语素 → 该命中不计入
    （逐命中粒度：「别重复入库，有新增就下载」这类一否定一肯定的混合句不被整句豁免）。"""
    text = str(utterance or "")
    for m in _PENDING_IMPORT_HARD_RE.finditer(text):
        head = text[:m.start()]
        head = head[max(head.rfind(p) for p in "，。；！？,.;!?\n") + 1:]
        if not _PENDING_IMPORT_DENY_MORPH_RE.search(head):
            return True
    return False


                                                                                           

                       
                                                                 
                                                           
                                           
                                                        
_COMPLEX_CONN_RE = re.compile(
    r"然后|接着|随后|之后|完了|最后|顺便|还要|再把|再帮|再检查|再搜|再告诉|也来|，再|；"
    r"|也给|也搜|也检查|也顺便|另外|同时|一并|以及|并且|分别")
_COMPLEX_COND_RE = re.compile(
    r"如果|若|有的话|没有的话|有新增|有新的[^，。]*就|有[^，。]*就(?:搜|下载|入库|同步)")

                                                
                                                 
                                                           
                                                     
_COMPLEX_RESTRAINT_RE = re.compile(r"不用|别再|不要再|(?<!分)别|算了|取消|先不")

                                                              
                                                    
                                                   
_PENDING_COUNT_RE_LANE = _PENDING_COUNT_RE

_COMPLEX_ROUTE_THRESHOLD = 2


def decide_complexity_score(utterance: str) -> tuple[int, int, int, int]:
    """复杂度评分拆解 (score, 连接词数, 条件词数, 点名来源数)——测试与审计用同一份拆解。"""
    text = str(utterance or "")
    n_conn = len(_COMPLEX_CONN_RE.findall(text))
    n_cond = len(_COMPLEX_COND_RE.findall(text))
    n_src = len(_named_sources_in(text))
    return n_conn + n_cond + max(0, n_src - 1), n_conn, n_cond, n_src


def decide_lane(utterance: str) -> str:
    """复杂度路由车道："complex"（decide/repair 走 LLM_MODEL_COMPLEX 档）/ "simple"（现状）。

    实证（chat 与 reasoner 逐例对照）：chat 的长链失败以 **decide 断链**为主、
    understand/repair 首步误判为辅；reasoner 的克制类回落全在
    短链场景。因此 decide 与 repair 吃路由，understand/validate/narrate 恒走 chat。
    判罚次序（先守后放）：克制语素 → simple；库容问句+另一事项信号 → complex；score≥2 →
    complex；其余 simple。"""
    text = str(utterance or "")
    if _COMPLEX_RESTRAINT_RE.search(text):
        return "simple"
    score, _nc, _nd, _ns = decide_complexity_score(text)
    if score >= _COMPLEX_ROUTE_THRESHOLD:
        return "complex"
    if score >= 1 and _PENDING_COUNT_RE_LANE.search(text):
        return "complex"
    return "simple"


def _adjudicate_decide_obj(obj: dict, state: "_AgentState", *,
                           allow_placeholders: bool = False
                           ) -> tuple[dict | None, str, str, str]:
    """decide 提议的**机械裁决**（tool-call 与 JSON 兜底两通道共用同一套机械闸；
    fail-safe：任何非法/超界都当 done 停环，repair 预算在 decide 节点层至多一次）。

    返回 (下一个工具调用 raw 或 None, 人读说明, 被婉拒动作的汇报句, violation 反馈)——
    raw 为 None 即 done；第三件在「提议被机械层拦下」（范围外动词 / 完全重复 / 搜索
    覆盖闸 / 死路）时非空，narrate 的确定性兜底汇报要点名这件没做的事（LLM 汇报路径有
    规则 2/3 兜底，不需要它）；**第四件只在 `_validate_raw` 校验违规这一种停法下非空**
    （人读违规清单，供 decide 节点带反馈重问一次——2026-08-08 violation 重问对称化，
    与非法应答重问同型；去重/覆盖闸/死路/暂停令是**刻意的机械停**，重问只会再撞同一
    道闸，不给反馈、不重问）。

    `allow_placeholders`（2026-08-20 批）：True = 占位接地续步（`_batch_readonly_extras`
    调用）——占位槽值已在静态阶段（正则/序号/矩阵）校验过，本裁决不再重复判定；
    False（默认，主步与单步路径）时，raw 里出现形似/越界占位引用 → 记 violation，走
    repair/回炉——占位只能引用**本轮次已执行**的 rank/rerank 步（主步序号 = 已执行步数
    +1，施工修正见 `_placeholder_static_violations` 注释）。"""
    done = obj.get("done")
    if done is True or str(done or "").strip().lower() == "true":
        return None, "大模型判断：要求的事已经完成（或条件不成立，没有要做的下一步）。", "", ""
    verb = str(obj.get("verb") or "").strip()
    if not verb:
        return None, "大模型没说完成了，也没说下一步做什么，按「已完成」收尾。", "", ""
    if verb == "none":
                                                         
                                                   
        return None, "大模型判断：要求的事已经完成（或条件不成立，没有要做的下一步）。", "", ""
    if verb not in LOOP_TOOLS:
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, (f"大模型提议的「{verb}」不在允许自动执行的范围内，按「已完成」收尾"
                      "（范围外的动作绝不会在这里执行）。"), (
                      f"你要的「{shown}」这一步没有做——它不在允许自动执行的范围内。"), ""
                                          
                                                         
    scope = str(state.get("route_scope") or "general")
    suite = set(_SUITE_LOOP_VERBS.get(scope) or _SUITE_LOOP_VERBS["general"])
    if verb != "route.request" and verb not in suite:
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, (f"本回合走「{scope}」路线——「{shown}」不在本路线的"
                      "工具面内，按「已完成」收尾。"), (
                      f"你要的「{shown}」这一步没有做——它不属于本回合的处理路线。"), ""
    if verb == "route.request":
        target = str(obj.get("target_route") or "").strip()
        if scope == "rescue":
            return None, ("检索救回路线不允许换线，这一步没有执行，"
                          "按「已完成」收尾。"), (
                          "你要的「切换处理路线」这一步没有做——救回路线不开放换线。"), ""
        if _route_request_used(list(state.get("steps") or [])) >= MAX_ROUTE_REQUEST:
            return None, ("本次请求已换过一次处理路线（每轮最多 1 次），这一步没有再"
                          "执行，按「已完成」收尾。"), (
                          "你要的「切换处理路线」这一步没有做——每轮最多换 1 次路线，"
                          "机会已用完。"), ""
        if target not in _ROUTE_REQUEST_TARGETS or target == scope:
            return None, ("切换处理路线的目标不成立（目标不在 search/action/general 里，"
                          "或与当前路线相同），这一步没有执行，按「已完成」收尾。"), (
                          "你要的「切换处理路线」这一步没有做——目标路线不成立。"), ""
                                       
                                                             
                                              
    if (verb in _NETWORK_LOOP_TOOLS and _is_network_call(verb, obj.get("source"))
            and _network_moratorium(list(state.get("steps") or []))):
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, (f"联网已连续失败两次，联网暂停中——「{shown}」是联网工具，"
                      "这一步本回合不再尝试，按「已完成」收尾。"), "", ""
                                            
    if verb == _ROLLBACK_LOOP_TOOL and _rollback_used(list(state.get("steps") or [])) >= MAX_ROLLBACK:
        return None, (f"本次请求的回滚预算已用完（每轮最多 {MAX_ROLLBACK} 次），这一步没有再执行，"
                      "按「已完成」收尾。"), (
                      f"你要的「回滚写操作」这一步没有做——每轮最多回滚 {MAX_ROLLBACK} 次，"
                      "回滚预算已用完。"), ""
                                            
                                                 
                                                           
                  
    if verb in _WRITE_LOOP_TOOLS and (
            _write_steps_used(list(state.get("steps") or [])) >= MAX_WRITE_STEPS
            or _write_records_used(list(state.get("steps") or [])) >= MAX_WRITE_RECORDS):
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        _steps_used = _write_steps_used(list(state.get("steps") or []))
        _recs_used = _write_records_used(list(state.get("steps") or []))
        reason = (f"写步已达上限 {MAX_WRITE_STEPS} 次" if _steps_used >= MAX_WRITE_STEPS
                  else f"本次请求已累计写入 {_recs_used} 条（上限 {MAX_WRITE_RECORDS} 条）")
        return None, (f"本次请求的{reason}——「{shown}」是写工具，"
                      "这一步没有做（其余已入库的可在回收站账本里查到），按「已完成」收尾。"), (
                      f"你要的「{shown}」这一步没有做——一次请求最多自动写 {MAX_WRITE_STEPS} 次 "
                      f"/ {MAX_WRITE_RECORDS} 条，预算已用完；还要入库可以再说一次。"), ""
                                                                 
                                                        
    if verb == "search.rerun" and \
            _search_rerun_used(list(state.get("steps") or [])) >= MAX_SEARCH_RERUN:
        return None, (f"本次请求已换词重检 {MAX_SEARCH_RERUN} 次（一次请求最多 "
                      f"{MAX_SEARCH_RERUN} 次），这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「检索新查询」这一步没有做——一次请求最多换词重检 "
                      f"{MAX_SEARCH_RERUN} 次，预算已用完。"), ""
                                                                 
                                                     
    if verb == "rank" and \
            _rank_used(list(state.get("steps") or [])) >= MAX_RANK:
        return None, (f"本次请求已新检索 {MAX_RANK} 次（一次请求最多 {MAX_RANK} 次），"
                      "这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「检索数据集」这一步没有做——一次请求最多新检索 "
                      f"{MAX_RANK} 次，预算已用完。"), ""
    if verb == "rerank" and \
            _rerank_used(list(state.get("steps") or [])) >= MAX_RERANK:
        return None, (f"本次请求已优化重检 {MAX_RERANK} 次（一次请求最多 {MAX_RERANK} 次），"
                      "这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「优化检索词重查」这一步没有做——一次请求最多优化重检 "
                      f"{MAX_RERANK} 次，预算已用完。"), ""
                                                                   
                                                    
    if verb == "results.show" and \
            _show_used(list(state.get("steps") or [])) >= MAX_SHOW:
        return None, (f"本次请求已上屏 {MAX_SHOW} 次（一次请求最多 {MAX_SHOW} 次），"
                      "这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「更新结果区」这一步没有做——一次请求最多上屏 "
                      f"{MAX_SHOW} 次，预算已用完。"), ""
                                                                
                                             
    if verb == "ask.relax" and \
            _ask_relax_used(list(state.get("steps") or [])) >= MAX_ASK_RELAX:
        return None, (f"本次请求已询问放宽条件 {MAX_ASK_RELAX} 次（一次请求最多 "
                      f"{MAX_ASK_RELAX} 次），这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「询问放宽条件」这一步没有做——一次请求最多询问放宽条件 "
                      f"{MAX_ASK_RELAX} 次，预算已用完。"), ""
                                              
                                                         
                                                     
    if verb == "compare.datasets" and \
            _compare_used(list(state.get("steps") or [])) >= MAX_COMPARE:
        return None, (f"本次请求已对比数据集 {MAX_COMPARE} 次（一次请求最多 {MAX_COMPARE} 次），"
                      "这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「对比数据集」这一步没有做——一次请求最多对比 "
                      f"{MAX_COMPARE} 次，预算已用完。"), ""
    if verb == "cite.export" and \
            _cite_export_used(list(state.get("steps") or [])) >= MAX_CITE_EXPORT:
        return None, (f"本次请求已导出引文 {MAX_CITE_EXPORT} 次（一次请求最多 "
                      f"{MAX_CITE_EXPORT} 次），这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「导出引文」这一步没有做——一次请求最多导出引文 "
                      f"{MAX_CITE_EXPORT} 次，预算已用完。"), ""
    if verb == "compat.find" and \
            _compat_used(list(state.get("steps") or [])) >= MAX_COMPAT:
        return None, (f"本次请求已查找兼容数据集 {MAX_COMPAT} 次（一次请求最多 "
                      f"{MAX_COMPAT} 次），这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「查找兼容数据集」这一步没有做——一次请求最多查找 "
                      f"{MAX_COMPAT} 次，预算已用完。"), ""
    if verb == "fair.check" and \
            _fair_used(list(state.get("steps") or [])) >= MAX_FAIR:
        return None, (f"本次请求已 FAIR 自检 {MAX_FAIR} 次（一次请求最多 {MAX_FAIR} 次），"
                      "这一步没有再执行，按「已完成」收尾。"), (
                      f"你要的「检查 FAIR 就绪度」这一步没有做——一次请求最多 FAIR 自检 "
                      f"{MAX_FAIR} 次，预算已用完。"), ""
                         
                                                          
    if verb in _failed_tool_ban(list(state.get("steps") or [])):
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, (f"「{shown}」这一步刚失败过（最近两步均失败），本回合不再尝试，"
                      "按「已完成」收尾。"), (
                      f"你要的「{shown}」这一步没有做——它刚失败过，本回合不再重试。"), ""
                                                          
                                                         
    autofill_note = ""
    if _autofill_named_source(verb, obj, state["utterance"]):
        autofill_note = f"source 槽位按 quoted 点名补为 {obj['source']}；"
    sync_all_named = _sync_all_online_named(verb, obj, state["utterance"])
    if sync_all_named:
                                                         
                                                                   
        autofill_note += (f"原话点名的是{sync_all_named}，本步 source 未填——"
                          "按全部在线源同步（写面超出点名范围）；")
    violations = _validate_raw(obj, state["utterance"], steps=list(state.get("steps") or []))
    if not allow_placeholders:
                                                  
                                                             
                                                    
                                                                         
                      
        _exec_verbs = [str(s.get("verb") or "") for s in (state.get("steps") or [])]
        ph_extra = _placeholder_static_violations(
            verb, obj, _exec_verbs + [verb], len(_exec_verbs) + 1)
        if ph_extra:
            violations = list(violations) + ph_extra
    if violations:
                                                   
        return None, "大模型提议的下一步没通过检查（" + "；".join(violations) + "），按「已完成」收尾。", \
            "", "；".join(violations)
    if _is_duplicate_step(verb, obj, list(state.get("steps") or [])):
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, "大模型提议的下一步与已执行步骤重复，按「已完成」收尾。", (
            f"「{shown}」与已完成的步骤完全重复，按去重策略没有重复执行。"), ""
                              
                                              
    coverage = _search_coverage_violation(obj, list(state.get("steps") or []))
    if coverage:
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, ("大模型提议的下一步与已完成的搜索主题重复（换措辞不会改变结论），"
                      "按「已完成」收尾。"), (
                      f"「{shown}」没有再执行——同一主题刚才已经搜过。"), ""
                                                           
                                      
    src_norm = _norm_source(obj.get("source"))
    for dead in state.get("dead_ends") or []:
        if str(dead.get("verb") or "") != verb:
            continue
        dead_src = str(dead.get("source") or "")
        if dead_src and dead_src != src_norm:
            continue
        verb_spec = _ap.VERB_BY_NAME.get(verb)
        shown = verb_spec.zh if verb_spec else verb
        return None, (f"「{shown}」刚才已经试过并如实失败（该来源本工具接不了），"
                      "换参数也是同样的结果，按「已完成」收尾。"), (
                      f"你要的「{shown}」这一步没有做——它指向的来源本工具接不了。"), ""
    return obj, autofill_note, "", ""


def _readonly_loop_verbs() -> frozenset:
    """只读白名单（2026-08-14 批）：LOOP_TOOLS 注册表 `readonly=True` 的动词，**减去**
    下方显式排除项——代码口径（2026-08-22 修正 docstring，与代码逐位对齐）现况
    = curate.check_updates / curate.db_status / search.rerun / rank / rerank（均无写库
    副作用、槽位全部来自原话、不依赖前序结果；rank/rerank/search.rerun 是只读本地检索，
    2026-08-16/17 检索工具化与 scoped 路由批入注册表后即自然落进本白名单——本 docstring
    此前滞后写作「现况=check_updates/db_status」，以代码为准）。写动词
    （search_online / sync_updates）**永不**进白名单： 实测预发写调用逐位保真
    仅 45%（参数要等前序真实结果接地、条件要等前步判定）。
    route.request虽只读也**永不**进批：它是控制面元动词——同批多调用
    里混进换线会让面切换时序不可判，换线必须独占一轮。
    compare.datasets / compat.find / fair.check同样只读但**永不**
    进白名单：三个工具的**缺省对象依赖前序结果**（当前结果前两条/第一条——要从环内最近检索
    步的现场现取），同批执行时前序检索步未必已跑，缺省语义不可判；宁让它们独占一轮
    （批内剔出即回炉，下一轮 decide 带新状态重判，不丢事）。cite.export 是写工具
    （readonly=False），天然不进白名单。
    2026-08-20：四工具的**缺省依赖**由「同批**占位引用**」显式接地后
    即可进批——白名单之外另开 `_PLACEHOLDER_SLOTS` 通道（`_batch_readonly_extras` v2
    判断），本函数口径不变。"""
    return frozenset(v for v, s in LOOP_TOOLS.items()
                     if s.get("readonly")
                     and v not in ("route.request", "compare.datasets", "compat.find",
                                   "fair.check"))


                                                                                             
                            
                                                       
                                             

                                                          
                                                         
                                               
_PLACEHOLDER_RE = re.compile(r"^\$([1-9]\d*)\.top\[(\d+)\]\.dataset_uid$")

                                                   
_PLACEHOLDER_PRODUCERS: frozenset[str] = frozenset({"rank", "rerank"})

                                  
                                                                                     
                                              
_PLACEHOLDER_SLOTS: dict[str, tuple[str, ...]] = {
    "compare.datasets": ("a", "b"),
    "compat.find": ("uid",),
    "fair.check": ("uid",),
    "cite.export": ("uids",),
}

                                                 
_PLACEHOLDER_META_SLOTS: frozenset[str] = frozenset(
    {"quoted", "confidence", "reason", "cancelled"})

                                              
                                                               
                                                                  
                                                     
_NEVER_BATCH_LOOP_VERBS: frozenset[str] = frozenset(
    {"route.request", "curate.rollback", "curate.search_online", "curate.sync_updates"})

                                         
_PH_BAD: tuple[str, ...] = ("__ph_bad__",)


def _placeholder_ref(value: Any) -> tuple[int, int] | tuple[str, ...] | None:
    """单值占位识别：返回 （N, i) = 合法占位；`_PH_BAD` = 形似占位但不合规
    （以 `$` 开头、或含 `.top[` / `.dataset_uid` 段却拼不成唯一形状——内嵌/路径错/缺段）；
    None = 普通字面量。"""
    s = str(value)
    m = _PLACEHOLDER_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.startswith("$") or ".top[" in s or ".dataset_uid" in s:
        return _PH_BAD
    return None


def _iter_placeholder_candidates(raw: dict) -> Iterator[tuple[str, str]]:
    """枚举 raw 里可能承载占位的 (槽位, 值串)——uids 数组槽逐个元素，其余槽整体为一个值。
    元槽位（quoted/confidence/reason/cancelled）也枚举（静态校验要拦它们）；verb 不算槽。"""
    for slot, value in (raw or {}).items():
        if slot == "verb":
            continue
        if slot == "uids" and isinstance(value, list):
            for item in value:
                yield slot, str(item)
        elif isinstance(value, str):
            yield slot, value


def _raw_has_placeholder(raw: dict) -> bool:
    """raw 里是否存在**合法**占位引用（静态阶段已校验过，这里只做存在性探测——
    validate 的延迟通道与 decide 的采纳留痕用）。"""
    for _slot, value in _iter_placeholder_candidates(raw):
        if isinstance(_placeholder_ref(value), tuple):
            return True
    return False


def _slots_has_placeholder(slots: Any) -> bool:
    """计划 slots 里是否存在合法占位引用（execute 主步解析的探测——主步是 plan 形态，
    `_iter_placeholder_candidates` 对 slots dict 同样适用：verb 键不存在、uids 数组逐元素）。"""
    if not isinstance(slots, dict):
        return False
    for _slot, value in _iter_placeholder_candidates(slots):
        if isinstance(_placeholder_ref(value), tuple):
            return True
    return False


def _placeholder_static_violations(verb: str, raw: dict, position_verbs: list[str],
                                   pos: int) -> list[str]:
    """**静态阶段**占位校验（2026-08-20 施工修正：`pos` 与 `N` 的编号口径
    为**本轮次执行步序号**，1 起——真机探针实测模型自然形态是「understand 先 rank、decide
    单发 compare($1)」，批内局域编号会把该形态误杀成死路；零前序时与设计示例的批内编号
    逐位一致，见设计文档「施工修正」节）：只查正则形状 + 序号越界 + 流向矩阵——**不走**
    `_validate_raw` / `build_plan_from_raw`（它们会拒掉或毁形占位）。返回人读违规清单，
    空表 = 通过。position_verbs = 按执行序号排列的动词清单（下标 0 = 本轮第 1 步：
    已执行步在前，主步/同批续步按执行顺序接后）；pos = 当前步的执行序号（1 起）。
    任一违规 → 调用方截断整尾回炉（依赖链已不可信）。"""
    violations: list[str] = []
    for slot, value in _iter_placeholder_candidates(raw):
        ref = _placeholder_ref(value)
        if ref is _PH_BAD:
            violations.append(
                f"槽位 {slot} 的「{value}」不是合法的占位引用——唯一合法形状是 "
                "$<N>.top[<i>].dataset_uid（整体为一个值，不支持内嵌或其它路径）。")
            continue
        if not isinstance(ref, tuple):
            continue         
        n, i = ref
        if slot in _PLACEHOLDER_META_SLOTS:
            violations.append(f"元槽位 {slot} 不允许占位引用。")
            continue
        allowed = _PLACEHOLDER_SLOTS.get(verb, ())
        if slot not in allowed:
            violations.append(
                f"槽位 {slot} 不允许占位引用（{verb} 只有"
                f"{'、'.join(allowed) if allowed else '没有'}槽可占位）。")
            continue
        if n >= pos:
            violations.append(
                f"占位 ${n}.top[{i}].dataset_uid 引用了执行序号 {n}——必须小于当前步"
                f"的执行序号 {pos}（序号从 1 起，本轮第 1 步是 $1）。")
            continue
        producer = position_verbs[n - 1] if 1 <= n <= len(position_verbs) else ""
        if producer not in _PLACEHOLDER_PRODUCERS:
            violations.append(
                f"占位 ${n}.top[{i}].dataset_uid 引用的第 {n} 步是"
                f"「{producer or '未知'}」——只允许引用 rank / rerank 的 top digest。")
    return violations


                                                 
_BATCH_EXEC_SEQ: list[int] = [0]


def _next_batch_id() -> str:
    """下一个占位批的批 id（execute 局部，trace 里定位「批内解析/跳过」用）。"""
    _BATCH_EXEC_SEQ[0] += 1
    return f"batch-{_BATCH_EXEC_SEQ[0]}"


def _resolve_placeholder_slots(raw: dict, resolved: dict[int, dict],
                               batch_id: str, pos: int
                               ) -> tuple[dict[str, Any], str, str | None, list[str]]:
    """execute **解析阶段**：把 raw 里占位槽替换为真实 dataset_uid。

    resolved = {执行序号(1 起): 该步成功 result}（execute 局部变量，见 execute 节点注释）。
    返回 (解析后的槽位 dict, trace 细节串, 跳过原因码 or None, 被解析的占位槽名清单)：
      - skip_reason=None：全部占位解析成功（槽里可混字面量，原样透传）；
      - "dependency_unavailable"：引用的前序步骤未成功执行（没进 resolved）→ 本步跳过；
      - "resolver_error"：top 下标越界 / top 缺失 / 条目无 dataset_uid → 本步跳过。
    后两种**不算工具失败**：不触发 `_failed_tool_ban`、不吃失败预算、留 trace（含批 id/
    计划位置/依赖位置/原引用/跳过原因）、不写假 ok=False step（「不执行不记步」口径）。
    占位不在槽里（普通字面量/其它键）原样透传；`resolved_slots` 供 `_resolve_placeholder_plan`
    只覆写占位槽（其余槽保留 build_plan_from_raw 的净化：limit 丢弃、display 归一等）。"""
    slots: dict[str, Any] = {}
    resolved_slots: list[str] = []
    resolve_notes: list[str] = []
    for slot, value in (raw or {}).items():
        if slot in _PLACEHOLDER_META_SLOTS or slot == "verb" or slot == "_batch_pos":
            continue
        if slot == "uids" and isinstance(value, list):
            resolved_uids: list[str] = []
            for item in value:
                item = str(item)
                ref = _placeholder_ref(item)
                if isinstance(ref, tuple):
                    uid, note, skip = _resolve_one(ref, slot, resolved, batch_id, pos)
                    if skip:
                        return {}, note, skip, []
                    resolved_uids.append(uid)
                    if note:
                        resolve_notes.append(note)
                    resolved_slots.append(slot)
                else:
                    resolved_uids.append(item)            
            if resolved_uids:
                slots["uids"] = resolved_uids
            continue
        if not isinstance(value, str):
            continue
        ref = _placeholder_ref(value)
        if isinstance(ref, tuple):
            uid, note, skip = _resolve_one(ref, slot, resolved, batch_id, pos)
            if skip:
                return {}, note, skip, []
            slots[slot] = uid
            if note:
                resolve_notes.append(note)
            resolved_slots.append(slot)
        else:
            slots[slot] = value
    return slots, "；".join(resolve_notes), None, resolved_slots


def _resolve_placeholder_plan(raw0: dict, resolved: dict[int, dict], batch_id: str,
                              ordinal: int, utterance: str, has_results: bool,
                              result_total: int, *, steps: list[dict] | None = None
                              ) -> tuple[dict | None, str, str | None]:
    """ **解析后全闸**：占位替换为真实值 → `_validate_raw` →
    `build_plan_from_raw`——与单步调用同口径，一步不少。raw0 = 带占位的 raw（verb + 槽位）。

    返回 (plan or None, trace 细节串, 跳过原因码 or None)：
      - None/None = 成功（plan 可执行，细节串含「占位 → uid」映射留 trace）；
      - "dependency_unavailable" / "resolver_error" / "cancelled" / "downgraded" =
        解析/闸失败，调用方跳过（不执行、不记步、trace 留原因）。
    主步（decide 单发续步）与同批续步共用本助手——两处行为逐位一致。"""
    slots_r, note, skip, resolved_slots = _resolve_placeholder_slots(
        raw0, resolved, batch_id, ordinal)
    if skip:
        return None, note, skip
    raw2: dict[str, Any] = {"verb": str(raw0.get("verb") or ""), **slots_r}
    for _k in ("quoted", "confidence", "reason"):
        if _k in raw0:
            raw2[_k] = raw0[_k]
    if raw0.get("cancelled") is True:
        return None, (
            f"（批 {batch_id}，计划位置 {ordinal}）你说了不做，这一步已取消。"), "cancelled"
    violations = _validate_raw(raw2, utterance, steps=steps)
    if violations:
        return None, (
            f"（批 {batch_id}，计划位置 {ordinal}）解析后实参未通过检查"
            f"（{'；'.join(violations)}）——本步不执行、不记步，留待下一轮带新状态重判。"), \
            "validation"
    plan2 = _ap.build_plan_from_raw(
        raw2, utterance, has_results=has_results, result_total=result_total,
        llm_status="ok")
    if str(plan2.get("verb") or "") != str(raw0.get("verb") or "") or plan2.get("cancelled"):
                                                 
        return None, (
            f"（批 {batch_id}，计划位置 {ordinal}）解析后计划被机械层降级/取消"
            "（verb 变了或 cancelled），本步不执行、不记步。"), "downgraded"
                                                           
                                                               
                                                                
                                                          
    plan2 = dict(plan2)
    if resolved_slots:
        plan2["slots"] = {**dict(plan2.get("slots") or {}),
                          **{k: slots_r[k] for k in resolved_slots if k in slots_r}}
    return plan2, note, None


def _resolve_one(ref: tuple[int, int], slot: str, resolved: dict[int, dict],
                 batch_id: str, pos: int) -> tuple[str, str, str | None]:
    """单条占位引用的解析：返回 (uid, trace 细节串, 跳过原因码 or None)。"""
    n, i = ref
    ref_zh = f"${n}.top[{i}].dataset_uid"
    dep = resolved.get(n)
    if dep is None:
        return "", (
            f"（批 {batch_id}，计划位置 {pos}，依赖位置 {n}，原引用 {ref_zh}）依赖不可用："
            f"第 {n} 个调用未成功执行（或未在批内执行），没有可解析的 top 结果——"
            "本步不执行、不记步，留待下一轮带新状态重判。"), "dependency_unavailable"
    top = dep.get("top") or []
    if not isinstance(top, list) or i >= len(top):
        got = len(top) if isinstance(top, list) else 0
        return "", (
            f"（批 {batch_id}，计划位置 {pos}，依赖位置 {n}，原引用 {ref_zh}）解析失败："
            f"第 {n} 个调用的 top 只有 {got} 条，下标 {i} 越界——本步不执行、不记步，"
            "留待下一轮带新状态重判。"), "resolver_error"
    entry = top[i] if isinstance(top[i], dict) else {}
    uid = str(entry.get("dataset_uid") or "").strip()
    if not uid:
        return "", (
            f"（批 {batch_id}，计划位置 {pos}，依赖位置 {n}，原引用 {ref_zh}）解析失败："
            f"第 {n} 个调用 top 的第 {i} 条没有 dataset_uid（生产者未按新契约输出）——"
            "本步不执行、不记步，留待下一轮带新状态重判。"), "resolver_error"
    return uid, f"（批 {batch_id}，计划位置 {pos}，依赖位置 {n}）占位 {ref_zh} → {uid}", None


def _batch_readonly_extras(calls: list, first_raw: dict,
                           state: "_AgentState") -> tuple[list[dict], int]:
    """多调用的**同批消费**过滤（2026-08-14 ；2026-08-20 升级为支持
    **占位接地**批量，§2。实测依据见 `_readonly_loop_verbs` 注释与
    `research/reports/multicall-legality-probe/summary.md`——A 类独立只读
    批量第 2..N 个调用 schema/语义合法率 100%（n=371））。

    decide 一次回 N 个调用时，第一个照旧走「裁决 → validate → execute」主路径；本函数从
    第 2..N 个里筛出可以**同批安全执行**的续步。返回 (采纳的 raw 列表, 回炉个数)。
    全部机械判定、宁严勿宽——被剔/回炉的调用不会丢：下一轮 decide 带新状态重判（与
    「取第一个」旧策的唯一差别是**独立续步不再白等一轮模型往返**）。规则（v2）：
    - **永不进批清单**（`_NEVER_BATCH_LOOP_VERBS`：route.request / curate.rollback /
      一切写库动词）出现即**截断**：其后的调用可能是「写给后状态」的（如入库后再报数），
      语义依赖顺序，整尾回炉再判；写库动词无占位也截断——v2 不做写入链；
    - **占位静态校验**：占位引用只过正则 + 序号越界 + 流向矩阵——违规即
      **截断**整尾（模型的依赖链已不可信，后续调用可能依赖它）；通过即放行（raw 形态
      原样暂存，带 `_batch_pos` 标批内序号，执行时解析后再过全闸）；
    - **无占位**的写动词（cite.export）与 compare/compat/fair（缺省对象依赖前序结果）
      照旧截断回炉——只有占位接地才进批；
    - 幻觉工具名 / finish / unsupported 出现即截断（与既有口径一致）；
    - 参数键 ⊆ 该动词 decide 工具 schema 的声明键（实测模型偶把 keywords/species
      涂抹进 check_updates——批量路径从严，脏参数调用回炉）；
    - 与首步 / 已执行步骤 / 已采纳同批步**去重**（同 verb 同槽位指纹只留第一个）；
    - 逐个过 `_adjudicate_decide_obj` 全套机械闸（联网暂停/失败禁提/校验/覆盖/死路——
      与单呼叫续步完全同口径；被闸住的剔除但不截断后续；占位续步放行 `allow_placeholders`）；
    - MAX_STEPS 预算：本批采纳后总步数不许越界。"""
    steps = list(state.get("steps") or [])
    budget = MAX_STEPS - len(steps) - 1            
    if budget <= 0 or len(calls or []) < 2:
        return [], 0
    props_by_name = {
        str((spec.get("function") or {}).get("name") or ""):
            set((((spec.get("function") or {}).get("parameters") or {}).get("properties") or {}))
        for spec in _DECIDE_TOOL_SPECS
    }
    readonly = _readonly_loop_verbs()
    accepted: list[dict] = []
    dropped = 0
                                                           
    seen_steps = steps + [{"verb": str(first_raw.get("verb") or ""),
                           "slots": {k: v for k, v in first_raw.items() if k != "verb"},
                           "ok": True}]
                                                
                                                            
                                       
    n_exec = len(steps)
    position_verbs: list[str] = [str(s.get("verb") or "") for s in steps]
    position_verbs.append(str(first_raw.get("verb") or ""))       
    for i in range(1, len(calls)):
        call = calls[i]
        if isinstance(call, dict):
            name, args = call.get("name"), call.get("args")
        else:                                       
            name, args = getattr(call, "name", ""), getattr(call, "args", None)
        verb = _DECIDE_TOOL_NAME_TO_VERB.get(str(name or ""))
                                                
        if verb in _NEVER_BATCH_LOOP_VERBS or (
                verb not in readonly and verb not in _PLACEHOLDER_SLOTS):
            dropped += len(calls) - i
            break
        if not isinstance(args, dict) or not set(args) <= props_by_name.get(str(name), set()):
            dropped += 1
            continue
        raw = {"verb": verb, **args}
                                           
        ph_violations = _placeholder_static_violations(
            verb, raw, position_verbs, n_exec + 1 + len(accepted) + 1)
        if ph_violations:
            dropped += len(calls) - i
            break
        has_ph = _raw_has_placeholder(raw)
        if not has_ph and verb not in readonly:
                                                                  
                                           
            dropped += len(calls) - i
            break
        if _is_duplicate_step(verb, raw, seen_steps):
            dropped += 1
            continue
                                     
                                                                   
                                                                    
                                                        
                                                              
        nxt2, _note2, _dec2, _vfb2 = _adjudicate_decide_obj(
            raw, dict(state, steps=seen_steps), allow_placeholders=True)
        if nxt2 is None:
            dropped += 1
            continue
                                                                   
                                                            
        ordinal = n_exec + 1 + len(accepted) + 1
        accepted.append({**nxt2, "_batch_pos": ordinal})
        position_verbs.append(str(verb or ""))
        seen_steps.append({"verb": verb,
                           "slots": {k: v for k, v in nxt2.items() if k != "verb"},
                           "ok": True})
        if len(accepted) >= budget:
            dropped += len(calls) - i - 1
            break
    return accepted, dropped


def _parse_decide_answer(content: str, state: "_AgentState") -> tuple[dict | None, str, str, str]:
    """decide **散文 JSON 通道**（兜底档）的入口：容错解析 + `_adjudicate_decide_obj` 机械裁决。
    保留本函数签名是既有测试接缝（tests/test_agent_failure_semantics.py 直接调用）。"""
    obj = _ap.parse_action_response(content or "")
    if not obj:
        return None, "大模型没给出能读懂的答复，按「已完成」收尾。", "", ""
    return _adjudicate_decide_obj(obj, state)


def _decide_answer_kind(answer: Any, name_to_verb: dict[str, str]) -> tuple[str, Any]:
    """decide 主通道（tool-calling）应答的分诊：返回 (kind, payload)。

    kind ∈ loop（续步提议，payload = raw dict）/ finish（结构化 done，payload = 工具
    args dict——completion_report 所在，decide 的核销否决闸要读它）/
    unsupported（婉拒表外动作，payload = verb 名）/ json（content 里的可解析 JSON，
    payload = obj——模型没调工具但回了 JSON，与 understand 同一份双通道解析真源）/
    invalid（幻觉工具名 / 参数不是对象 / 啥也解析不出——fail-safe 停环）。
    **多 tool_call 取第一个**（DeepSeek 不遵守
    `parallel_tool_calls=False`，decide 不可读 17/17 全是多调用且第一个 17/17 合法；
    循环带着新状态会再判断后续，吃第一个不吞任何事）——多调用留痕由 decide 节点
    拼进 trace detail（`_classify`）；2026-08-14 起第 2..N 个里的只读独立续步
    由 decide 节点的 `_batch_readonly_extras` 同批采纳（本函数的分诊语义不变）。
    """
    tool_calls = getattr(answer, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        if isinstance(call, dict):
            name, args = call.get("name"), call.get("args")
        else:
            name, args = getattr(call, "name", ""), getattr(call, "args", None)
        if str(name or "") == _DECIDE_FINISH_TOOL:
            return "finish", args if isinstance(args, dict) else {}
        if str(name or "") == _DECIDE_UNSUPPORTED_TOOL:
            return "unsupported", str((args or {}).get("verb") or "") if isinstance(args, dict) else ""
        verb = name_to_verb.get(str(name or ""))
        if verb and isinstance(args, dict):
            return "loop", {"verb": verb, **args}
        return "invalid", None                  
    obj = _ap.parse_action_response(_message_text(answer))
    if obj:
        return "json", obj
    return "invalid", None


                                                                                            
                                     
                                                                                            

_STEPS_REPORT_RULES_ZH = (
    "你是执行汇报员。据随附的**用户原话**与各步骤的**真实执行结果**写一段简明中文汇报"
    "（不超过 150 字）。\n"
    "铁律（违反任一条都是错误）：\n"
    "1. 只用给定的事实与数字——不得编造、不得改写数字、不得估算；\n"
    "2. **steps 列表就是全部做过的事**——没出现在列表里的动作（下载、删除等）"
    "**一律没有发生**，汇报里绝不能说做了；「已下载/已入库」这类话只允许按第 3 条的"
    "既遂含义说；\n"
                                    
                                                           
                           
    "3. 步骤动词的既遂含义（ok=true 时必须按此说，不许反着说）：\n"
    "   - curate.search_online ＝ 已联网搜索**并把 record_count 条记录写入本地库**——"
    "入库**已经发生**（filename 是写入的文件），不许说「未入库 / 未执行入库 / 没有导入」；\n"
    "   - curate.sync_updates ＝ 已检查更新并按需入库，imported_total 是实际入库条数"
    "（为 0 就如实报 0——这是正常完成，不是失败）；\n"
    "   - curate.check_updates / curate.db_status ＝ **只读**动作，本身不做任何入库或下载；\n"
                                                     
                                                 
                                            
                                                  
                                                
                   
    "   - search.rerun ＝ 换查询词把本地库**重新检索**了一遍（只读重跑检索，不写库）；"
    "adopted=true 表示新条件的结果已上屏——如实报命中条数（result_total 为 0 就是"
    "新条件没有匹配，这是新条件的真实答案，照实直说，绝不能说成「没查」）；结果区"
    "已更新就只描述新结果是什么，绝不再提旧结果，更不许说「保持不变」「没有更合适"
    "的改写」这类与实际矛盾的话；"
    "adopted=false 只有两种：结果与当前相同（结果区未改动）、或为保住筛选条件没有执行"
    "（如实说明哪一档）；\n"
    "4. 说清每一步做了什么、结果如何；失败或没做到的步骤必须**如实写明原因**；\n"
    "5. snapshot_date 为 null 的来源**不许提日期**；\n"
    "6. 「条件没成立所以没做」是正常完成，不是失败（如「没有新增，不需要下载」）；\n"
                         
                                                      
    "7. 每个来源只在一个地方说结果：**来源名第一次出现的小句里就要带上它的数字（含 0）**——"
    "「ENCODE 检到 3 条疑似新增、实际入库 0 条」；不许先单独说一句「ENCODE 检查完成」、"
    "把数字留到别的小句，更不许只在句尾裸写数字；\n"
    "8. 不要建议、不要评论、不要客套。\n"
                                     
                                                   
                               
    "9. 检索类步骤（rank / rerank / search.rerun）的汇报要自然交代：用了哪些关键词、"
    "按什么方式在本地库检索（关键词规则检索，必要时先优化了检索词）、命中多少条、"
    "结果区是否更新——用自己的话连贯地说，不许照本段句式逐字套用。整段汇报只说"
    "实际发生的事：结果已更新就只描述新结果；确实什么都没执行才说结果未变——"
    "「结果已更新」与「保持不变」两种话绝不出现在同一段汇报里；\n"
    "10. 汇报里绝不出现内部工程术语——「重检 / 择优 / 采纳 / 救回 / 闸 / 批次」这类词"
    "对用户没有意义，一律改说「重新检索 / 没有匹配 / 结果已更新 / 结果区未改动」这类"
    "日常说法。\n"
)


def _steps_report_with_llm(chat_model: Any, utterance: str, steps: list[dict],
                           usage_sink: list | None = None) -> str | None:
    """LLM 据 steps 紧凑投影写整段汇报。失败/空回 → None（兜底确定性拼接，同一批事实）。"""
    from langchain_core.messages import HumanMessage

    payload = {
        "utterance": utterance,
        "steps": [_step_projection(s) for s in steps],
    }
    prompt = _STEPS_REPORT_RULES_ZH + "\n----- 原话与步骤结果（JSON）-----\n" + json.dumps(
        payload, ensure_ascii=False)
    try:
        answer = _invoke_text_with_continuation(
            chat_model, [HumanMessage(content=prompt)],
            usage_sink=usage_sink, usage_node="narrate")
    except Exception:
        return None
    text = _message_text(answer).strip()
    return text or None


                                                
                                                   
                                        
                                          
                                               
                             
_WRITE_ACTIONS_ZH: tuple[str, ...] = ("下载", "入库", "写入", "导入", "保存", "存",
                                      "删除", "恢复", "安装", "联网搜")
                                             
                                             
                                          
                                            
_WRITE_DONE_PREFIX_ZH: tuple[str, ...] = ("已完成", "已经完成", "已经", "已", "并", "搞定")
_WRITE_DONE_SUFFIX_ZH: tuple[str, ...] = ("完成", "好了", "好啦", "成功", "完毕", "搞定", "了")
_WRITE_NEG_ZH: tuple[str, ...] = ("未", "没", "不", "无")
                                               
                                             
                                          
                   
_WRITE_UNDONE_ZH: tuple[str, ...] = ("失败", "取消", "吗")
                                             
                       
_CLAUSE_BREAKS_ZH: tuple[str, ...] = ("，", "。", "；", "！", "？", "：", "\n")


def _clause_head(report: str, end: int, limit: int = 8) -> str:
    """report[:end] 末尾一小句（小句隔断之后、最长 limit 字）——完成态语素的归属窗口。"""
    head = report[max(0, end - limit):end]
    cut = max(head.rfind(p) for p in _CLAUSE_BREAKS_ZH)
    return head[cut + 1:] if cut >= 0 else head


                                                             
                                             
                                        
_WRITE_ZERO_ZH: tuple[str, ...] = ("0 条", "0条", "为 0", "为零", "=0", "0 个", "0个")


def _claims_done_write(report: str) -> bool:
    """汇报里是否有「写动作既遂」声称：写动作词的小句窗口内出现完成态语素（前后两种语序），
    且小句窗口内没有否定/非既遂/零数量语素（豁免看整个小句，含语素之后——「下载完成不了」
    「合并下载失败了」的否定都在完成态语素之后）。"""
    exempt = _WRITE_NEG_ZH + _WRITE_UNDONE_ZH + _WRITE_ZERO_ZH
    for action in _WRITE_ACTIONS_ZH:
        start = 0
        while True:
            i = report.find(action, start)
            if i < 0:
                break
            start = i + 1
            if action == "存" and report[i + 1:i + 2] == "在":
                continue                                     
            if action == "入库" and report[max(0, i - 2):i] == "同步":
                continue                                     
                                             
                                                     
            head = _clause_head(report, i)
                                                    
                                               
            tail = report[i + len(action):]
            cut = min([tail.find(p) for p in _CLAUSE_BREAKS_ZH if tail.find(p) >= 0]
                      or [len(tail)])
            tail = tail[:cut]
            for marker in _WRITE_DONE_PREFIX_ZH:
                j = head.rfind(marker)
                if j >= 0:
                    window = head[:j] + head[j + len(marker):] + tail
                    if not any(m in window for m in exempt):
                        return True
            for marker in _WRITE_DONE_SUFFIX_ZH:
                k = tail.find(marker)
                if k >= 0:
                    window = head + tail
                    if not any(m in window for m in exempt):
                        return True
    return False


                                            
                                              
                                                        
                                    
                                        
def _denies_done_write(report: str) -> bool:
    """汇报里是否有「写动作否认」声称：写动作词的小句窗口内出现否定语素（头窗：未/没/不/无；
    尾窗：失败/取消）。与 `_claims_done_write` 同构的模式化判定——调用场景是「入库步明明
    成功」，任何对写动作的否认都该判矛盾；误伤（「不需要下载」式措辞被拦）的代价只是
    回退确定性拼接，安全侧。"""
    for action in _WRITE_ACTIONS_ZH:
        start = 0
        while True:
            i = report.find(action, start)
            if i < 0:
                break
            start = i + 1
            if action == "存" and report[i + 1:i + 2] == "在":
                continue                           
            head = _clause_head(report, i)
            if any(m in head for m in _WRITE_NEG_ZH):
                return True
            tail = report[i + len(action):]
            cut = min([tail.find(p) for p in _CLAUSE_BREAKS_ZH if tail.find(p) >= 0]
                      or [len(tail)])
            if any(m in tail[:cut] for m in _WRITE_UNDONE_ZH):
                return True
    return False


                                                     
                                                      
_READ_ACTIONS_ZH: tuple[str, ...] = ("联网搜索", "联网检索", "搜索", "检索", "检查", "查询", "查")
                                                    
                                                          
          
_READ_RESULT_SUFFIX_ZH: tuple[str, ...] = ("到", "出")
                                                        
               
_READ_NEG_ZH: tuple[str, ...] = ("没有", "未", "没")
                                      
_READ_STEP_VERBS: tuple[str, ...] = ("curate.search_online", "curate.check_updates", "search.rerun")
                                              
_READ_TAIL_PARTICLES_ZH: tuple[str, ...] = ("过", "了", "的")


def _denies_done_read(report: str, steps: list[dict]) -> bool:
    """汇报里是否有「读动作否认」声称（第五路机械后检
    明明搜过 X，LLM 汇报却说「未搜索 X」——既有四路只管写动作否认与只读假性声称，
    管不到只读动作的否认）。

    检出「否认语素（未/没有/没，在小句头窗内）+ 读动作词」的小句后看**尾窗**
    （到下一小句隔断为止，隔断表复用 `_CLAUSE_BREAKS_ZH`）：
      a. 尾窗点名的来源 ∈ 成功读步实际触碰的来源集（与 `_report_names_untouched_source`
         同真源 `_sr.source_alias_spans`）→ 否认了真做过的检查/搜索；
      b. 尾窗无来源但有内容 token（`_keyword_content_tokens`），且 tokens ⊆ 成功 search 步
         keywords 的 token 并集 → 否认了真搜过的主题；
      c. 尾窗既无来源也无内容 token（裸否认「没有搜索」）且存在成功读步 → 直接矛盾；
      d. 其余不拦——「没搜 multiome」但搜索关键词确不含 multiome 是实话。**拿不准不拦**：
         误伤的代价只是回退确定性拼接（同一批事实），与既有四路同一条安全论证。
    """
    done_reads = [s for s in steps
                  if s.get("ok") and str(s.get("verb") or "") in _READ_STEP_VERBS]
    if not done_reads:
        return False
    touched = _step_touched_sources(done_reads)
    searched: set[str] = set()
    for s in done_reads:
        if s.get("verb") == "curate.search_online":
            searched |= set(_keyword_content_tokens((s.get("slots") or {}).get("keywords")))
    for action in _READ_ACTIONS_ZH:
        start = 0
        while True:
            i = report.find(action, start)
            if i < 0:
                break
            start = i + 1
            if report[i + len(action):i + len(action) + 1] in _READ_RESULT_SUFFIX_ZH:
                continue                                
            head = _clause_head(report, i)
            if not any(m in head for m in _READ_NEG_ZH):
                continue
            tail = report[i + len(action):]
            cut = min([tail.find(p) for p in _CLAUSE_BREAKS_ZH if tail.find(p) >= 0]
                      or [len(tail)])
            tail = tail[:cut]
            named = {str(span["source"]) for span in _sr.source_alias_spans(tail)}
            if named:
                if named & touched:
                    return True
                continue                            
            tokens = [t for t in _keyword_content_tokens(tail)
                      if any(ch not in _READ_TAIL_PARTICLES_ZH for ch in t)]
            if tokens:
                if set(tokens) <= searched:
                    return True
                continue                        
            return True                         
    return False


                                                
                                                                  
                                                          
                                      
_TOPIC_GATE_DIMS: tuple[str, ...] = ("species", "tissue", "disease", "assay")

                                               
                                                           
                                                                  
                                             
                                            
                                           
_DENIAL_MORPH_RE = re.compile(
    r"(?<!分)别|算了|取消|不用|先不|不再|不听|停(?:下|止|手)|(?<!要)不要|(?<!是)不是[要让]"
    r"|don'?t|do\s+not|stop|cancel", re.IGNORECASE)


def _utterance_topic_alias(utterance: str) -> str | None:
    """原话里命中的第一个主题维度**值**别名（无 → None）——sync 主题闸的机械判定。
    复用同一份词表与匹配真源（`vocabulary.CATALOG` × `query_parser._alias_occurrences`），
    口径与检索侧解析逐位一致：原话能解析出物种/组织/疾病/技术约束 = 限定了主题。
    先**等长遮蔽来源专名**再扫描——「Human Cell Atlas」自带 human
    「检查HCA更新并同步」曾被误判成限定主题而拦死 sync（来源名不是主题）。"""
    low = _sr.mask_source_spans(str(utterance or "")).lower()
    for dim in _TOPIC_GATE_DIMS:
        for entry in _domain_catalog().get(dim, []):
            for alias in entry.get("aliases", []):
                alias = str(alias).lower().strip()
                if alias and _qp_alias_occurrences(low, alias):
                    return alias
    return None


                       
_SYNC_GATE_CLAUSE_RE = re.compile(r"[，。；！？!?\n]+")

                                                      
                                                    
_SYNC_GATE_ANAPHORA_RE = re.compile(r"^\s*(?:有[^，。]*就|有的话|若|如果|要是)")


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """硬标点分句的 (start, end) 区间表（sync 主题闸两个判定共用）。"""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SYNC_GATE_CLAUSE_RE.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _sync_gate_scope_zh(utterance: str, quoted: str) -> str:
    """sync 主题闸的**分句作用域**（坐实）：主题词只有落在 sync 所引
    片段的分句里（条件回指句再往前扩一句）才算限定 sync 本身；主题词在**别的分支**时，
    对 sync 的拦截口径另走「别支主题」消息（见 `_validate_raw` 的 sync 闸）。
    quoted 拿不到/对不上原话 → 退回整句（fail-closed，与旧版全域扫描逐位一致）。"""
    text = str(utterance or "")
    q = str(quoted or "").strip()
    if not q or q not in text:
        return text
    qs, qe = text.index(q), text.index(q) + len(q)
    spans = _clause_spans(text)
    hit = [i for i, (a, b) in enumerate(spans) if a < qe and qs < b]
    if not hit:
        return text
    first = hit[0]
    if _SYNC_GATE_ANAPHORA_RE.match(text[spans[first][0]:spans[first][1]]) and first > 0:
        first -= 1
    return text[spans[first][0]:spans[hit[-1]][1]]


                                                       
                                                   
_SYNC_BRANCH_INTENT_RE = re.compile(r"同步|下载|入库|拿回|拿来|拉取|sync|download|pull", re.IGNORECASE)


def _sync_topic_elsewhere_exempt(utterance: str, raw: dict[str, Any]) -> bool:
    """sync 主题闸的**别支豁免**（2026-08-08 坐实：「检查CELLxGENE更新，有新增就
    同步，然后检查下ArrayExpress，有新的人类肺数据就搜来入库」——主题词「人类」限定的是
    ArrayExpress 分支，拿它拦 CELLxGENE 分支的 sync 是张冠李戴；reasoner 被误导理由反复
    拉扯后整计划崩成 AgentPlanInvalid 0/3）。豁免四条件（缺一照拦，fail-closed）：
    ① sync 的 quoted 作用域本身无主题词（调用方已判，这里不重查）；
    ② 作用域里本身有同步/下载/入库意图——sync 是该分支的合法动词，不是误选；
    ③ sync 填了可识别的 source 槽位——不带 source 的全量 sync 永不在豁免之列；
    ④ 主题词归属分支（主题分句 + 条件回指前句）点名的来源集合非空、且不含 sync 的 source
       ——归属判不出来（空集）时照拦。"""
    scope = _sync_gate_scope_zh(utterance, str(raw.get("quoted") or ""))
    if not _SYNC_BRANCH_INTENT_RE.search(scope):
        return False
    filled = _canonical_source(raw.get("source"))
    if not filled:
        return False
    topic = _utterance_topic_alias(utterance)
    if not topic:
        return False                         
    text = str(utterance or "")
    idx = text.lower().find(topic)
    spans = _clause_spans(text)
    ci = next((i for i, (a, b) in enumerate(spans) if a <= idx < b), None)
    if idx < 0 or ci is None:
        return False
    lo = ci
    if _SYNC_GATE_ANAPHORA_RE.match(text[spans[ci][0]:spans[ci][1]]) and ci > 0:
        lo = ci - 1
    branch_sources = set(_named_sources_in(text[spans[lo][0]:spans[ci][1]]))
    return bool(branch_sources) and filled not in branch_sources


def _canonical_source(value: Any) -> str | None:
    """任意写法 → 规范来源名（检索侧 SOURCE_ALIASES 同一份词表真源）；认不出 → None。
    2026-08-09 自对抗复查补第三趟：curate-only 源（当时 Zenodo 等不在检索词表）必须认识——
    否则 Zenodo 检查步 canon=None → 清单核销永远落空（`_step_covered_sources` 连同
    result.sources 条目回退一起失灵），点名闸的别支豁免也跟着 fail-closed 误拦。
    2026-08-14 Zenodo 已登记进检索 SOURCE_ALIASES（首批入库），第三趟对它变为冗余兜底，
    保留无伤（前两趟先命中）。"""
    v = _norm_source(value)
    if not v:
        return None
    for source, aliases in _sr.SOURCE_ALIASES:
        if v == _norm_source(source) or v in {_norm_source(a) for a in aliases}:
            return source
    for label, forms in _CURATE_EXTRA_NAMED_SOURCES:
        if v == _norm_source(label) or v in {_norm_source(f) for f in forms}:
            return label
    return None


def _step_touched_sources(steps: list[dict]) -> set[str]:
    """steps 实际触碰过的来源（规范名集合）：各步 slots.source + 结果 payload 的来源项。"""
    touched: set[str] = set()
    for s in steps:
        c = _canonical_source((s.get("slots") or {}).get("source"))
        if c:
            touched.add(c)
        r = s.get("result") or {}
        c = _canonical_source(r.get("source_label"))
        if c:
            touched.add(c)
        for e in (r.get("sources") or []):
            for key in ("source", "label"):
                c = _canonical_source(e.get(key))
                if c:
                    touched.add(c)
    return touched


def _report_names_untouched_source(report: str, steps: list[dict]) -> bool:
    """只读侧的**假性声称**（批B 集成抓到：「检查了10x和ArrayExpress的更新」
    与「ArrayExpress未在步骤中检查」同屏——写动作后检管不到只读声称）。
    汇报点名的来源若没有任何一步真碰过（slots.source / 结果 payload 都不算），即与实录矛盾。
    误伤（「你没点名 CELLxGENE」式如实说明被拦）的代价只是回退确定性拼接，安全侧。"""
    named = {str(span["source"]) for span in _sr.source_alias_spans(report)}
    if not named:
        return False
    return bool(named - _step_touched_sources(steps))


                                                                                            
                              
 
                                                
                             
                                                   
                                          
                                                              
                                                          
                                                       
                                                                        
                                                           
                                               
                                                     
                                                                        
                                 
                                         
                                                       
                                                                                            

                                   
_COUNT_CLAIM_VERB_RE = re.compile(
    r"(?:搜到|搜索到|检索到|找到|入库|下载|导入|写入|新增|发现)\s*了?\s*(\d+)\s*条")
                                                 
_COUNT_CLAIM_NOUN_RE = re.compile(r"(\d+)\s*条\s*(?:疑似)?(?:新增|新数据|候选)")


def _step_true_counts(steps: list[dict]) -> set[int]:
    """steps 实录里的**真实关键计数**集合（数字交叉核验的比对基准；只收 ok 步——
    失败步没有可信计数可引，它的 error 是系统的话、不是数据）。"""
    counts: set[int] = set()
    for s in steps:
        if not s.get("ok"):
            continue
        r = s.get("result") or {}
        kind = s.get("card_kind")
        if kind == "search_online":
            n = r.get("record_count")
            if isinstance(n, int) and not isinstance(n, bool):
                counts.add(n)
        elif kind == "check_updates":
            for e in r.get("sources") or []:
                if not isinstance(e, dict):
                    continue
                n = e.get("new_count")
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
                                                                           
                for k in ("online_recent", "local_count"):
                    n = e.get(k)
                    if isinstance(n, int) and not isinstance(n, bool):
                        counts.add(n)
                cands = e.get("new_candidates")
                if isinstance(cands, list):
                    counts.add(len(cands))
        elif kind == "sync_updates":
                                                
                                                             
            total = r.get("imported_total")
            if isinstance(total, int) and not isinstance(total, bool):
                counts.add(total)
            for e in r.get("sources") or []:
                if not isinstance(e, dict):
                    continue
                for k in ("new_count", "imported_count"):
                    n = e.get(k)
                    if isinstance(n, int) and not isinstance(n, bool):
                        counts.add(n)
                titles = e.get("imported_titles")
                if isinstance(titles, list):
                    counts.add(len(titles))
        elif kind == "db_status":
                                                                  
            n = r.get("total_records")
            if isinstance(n, int) and not isinstance(n, bool):
                counts.add(n)
        elif kind == "search_rerun":
                                                              
            for k in ("n_before", "n_after", "n_before_total", "n_after_total"):
                n = r.get(k)
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
            n = (r.get("payload") or {}).get("result_total") if isinstance(r.get("payload"), dict) else None
            if isinstance(n, int) and not isinstance(n, bool):
                counts.add(n)
        elif kind in ("compare", "compat_find", "cite_export", "fair_check"):
                                   
                                                             
            for key in ("n_same", "n_diff", "n_unknown"):
                n = r.get(key)
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
            if kind == "compat_find":
                n = r.get("total")
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
                n = len(r.get("compatible") or [])
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
            elif kind == "cite_export":
                n = r.get("n_datasets")
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
                n = len(r.get("uids") or [])
                if isinstance(n, int) and not isinstance(n, bool):
                    counts.add(n)
                for f in r.get("files") or []:
                    n = (f or {}).get("bytes")
                    if isinstance(n, int) and not isinstance(n, bool):
                        counts.add(n)
            elif kind == "fair_check":
                summary = (r.get("fair") or {}).get("summary") or {}
                for key in ("pass", "partial", "unknown", "total", "readiness_pct"):
                    n = summary.get(key)
                    if isinstance(n, int) and not isinstance(n, bool):
                        counts.add(n)
    return counts


def _report_miscounts_steps(report: str, steps: list[dict]) -> bool:
    """汇报里紧贴计数语境的阿拉伯数字与 steps 真实计数矛盾？（判定逻辑见上方区块注释）"""
    true_counts = _step_true_counts(steps)
    if not true_counts:
        return False
    claimed = {int(m.group(1)) for m in _COUNT_CLAIM_VERB_RE.finditer(report)}
    claimed |= {int(m.group(1)) for m in _COUNT_CLAIM_NOUN_RE.finditer(report)}
    return any(n not in true_counts for n in claimed)


def _report_contradiction_reason(report: str, steps: list[dict]) -> str | None:
    """LLM 汇报与 steps 实录的矛盾**原因码**（None = 不矛盾）。机械后检逐路判定：

    - `untouched_source`：汇报点名了任何一步都没碰过的来源（`_report_names_untouched_source`）
      → 只读侧假性声称；
    - `count_mismatch`：汇报里紧贴计数语境的数字不属于任何真实关键计数
      （`_report_miscounts_steps` 数字交叉核验）→ 谎称/改写数量；
    - `denied_read`：明明有成功的读步（联网搜索/检查更新），汇报却出现「未搜索 X」式
      读动作否认（`_denies_done_read` 模式化判定；结果动词「没搜到」式诚实措辞刻意豁免）
      → 谎称没做做了的（只读侧，2026-08-08）
    - `claimed_write`：steps 里没有成功的入库步，汇报却出现既遂写动作声称
      （`_claims_done_write` 模式化判定，否定/非既遂语素豁免）→ 谎称做了没做的；
    - `denied_write`：入库步明明成功，汇报却出现写动作否认表述 → 谎称没做做了的。
    判矛盾 → 调用方弃用该汇报、回退确定性拼接（同一批事实），原因码进 trace 留痕
    （复盘「真谎称」还是「误伤」要靠它定位是哪一路拦的）。"""
    if _report_names_untouched_source(report, steps):
        return "untouched_source"
    if _report_miscounts_steps(report, steps):
        return "count_mismatch"
    if _denies_done_read(report, steps):
        return "denied_read"
                                                                 
                                                
    wrote = any(
        s.get("ok") and (
            (s.get("card_kind") == "search_online"
             and int((s.get("result") or {}).get("record_count") or 0) > 0)
            or (s.get("card_kind") == "sync_updates"
                and int((s.get("result") or {}).get("imported_total") or 0) > 0)
        )
        for s in steps
    )
    if wrote:
        return "denied_write" if _denies_done_write(report) else None
    return "claimed_write" if _claims_done_write(report) else None


def _report_contradicts_steps(report: str, steps: list[dict]) -> bool:
    """LLM 汇报与 steps 实录**互相矛盾**？机械后检（prompt 铁律挡不住全部幻觉，安全围栏是代码）。
    逐路判定口径见 `_report_contradiction_reason`（本谓词是它的布尔薄封装，行为不变）。"""
    return _report_contradiction_reason(report, steps) is not None


def _steps_report_fallback_zh(steps: list[dict]) -> str:
    """全程汇报的**确定性**拼接（LLM 缺席时的兜底——与 LLM 汇报同一批事实）。"""
    parts: list[str] = []
    for s in steps:
        zh = str(s.get("verb_zh") or s.get("verb") or "这一步")
        if not s.get("ok"):
            parts.append(f"{zh}没有完成：{str(s.get('error') or '未知原因').rstrip('。')}")
            continue
        kind = s.get("card_kind")
        r = s.get("result") or {}
        if kind == "db_status":
            parts.append(_report_fallback_zh(r).rstrip("。"))
        elif kind == "check_updates":
            segs: list[str] = []
            for e in (r.get("sources") or []):
                label = str(e.get("label") or e.get("source") or "")
                mode = str(e.get("mode") or "")
                if mode == "online":
                    new = e.get("new_count")
                    if new is None:
                        segs.append(f"{label}的在线比对这次没能完成")
                    elif int(new):
                        segs.append(f"{label}疑似新增 {int(new)} 条")
                    else:
                        segs.append(f"{label}没有疑似新增")
                elif mode == "snapshot":
                    segs.append(f"{label} 只有本地副本（{int(e.get('local_count') or 0)} 条），"
                                "不能在线核对更新")
                else:
                    segs.append(f"{label}无法检查（{str(e.get('note_zh') or '没拿到具体原因')}）")
            parts.append("检查了来源更新——" + "；".join(segs) if segs else "检查了来源更新")
        elif kind == "search_online":
            filename = r.get("filename")
            if not filename:
                                            
                                                  
                warn = next((str(w).strip() for w in (r.get("warnings") or [])
                             if str(w or "").strip()), "")
                parts.append(warn.rstrip("。") if warn
                             else "联网搜索的候选均已在库中，没有重复入库")
            else:
                parts.append(f"联网搜到 {int(r.get('record_count') or 0)} 条，"
                             f"已入库到 {str(filename)}")
        elif kind == "sync_updates":
            segs2 = []
            for e in (r.get("sources") or []):
                label = str(e.get("label") or e.get("source") or "")
                note = str(e.get("note_zh") or "").strip().rstrip("。")
                                                                           
                                                      
                segs2.append(f"{label}：{note}" if note else f"{label}没有需要入库的新增")
            parts.append("检查更新并同步入库——" + "；".join(segs2) if segs2 else f"{zh}完成")
        elif kind == "search_rerun":
            rq = str(r.get("query") or "")
            if r.get("adopted"):
                                                            
                                                           
                nb = r.get("n_before_total") if r.get("n_before_total") is not None \
                    else r.get("n_before")
                na = r.get("n_after_total") if r.get("n_after_total") is not None \
                    else r.get("n_after")
                parts.append(
                    f"按「{rq}」重新检索——结果已更新："
                    + (f"原来 {int(nb)} 条" if isinstance(nb, int) and not isinstance(nb, bool)
                       else "原结果")
                    + f" → {int(na or 0)} 条")
            else:
                reason = str(r.get("reason") or "")
                why = ("结果与当前相同，结果区未改动"
                       if reason == "rewrite_no_change_kept_original"
                       else str(r.get("disclosure_zh")
                                or "新查询没能完整保留当前筛选条件，这次重新检索没有执行")
                       if reason == "structured_context_lost_kept_original"
                       else "这次重新检索没有执行，结果区未改动")
                parts.append(f"按「{rq}」重新检索——{why}")
        elif kind == "rollback":
                                                     
            parts.append(str(r.get("note_zh") or f"{zh}完成").rstrip("。"))
        elif kind == "compare":
                                                           
            note = str(r.get("comparison_zh") or "")
            if note:
                parts.append(note.rstrip("。"))
            elif r.get("identical"):
                parts.append(f"对比完成：可比字段完全相同（{zh}）")
            else:
                parts.append(f"{zh}完成（{int(r.get('n_same') or 0)} 个字段一致、"
                             f"{int(r.get('n_diff') or 0)} 个字段不同）")
        elif kind == "cite_export":
                                              
            parts.append(str(r.get("note_zh") or f"{zh}完成").rstrip("。"))
        elif kind == "compat_find":
                                               
            parts.append(str(r.get("note_zh") or f"{zh}完成").rstrip("。"))
        elif kind == "fair_check":
                                                 
            parts.append(str(r.get("note_zh") or f"{zh}完成").rstrip("。"))
        else:
            parts.append(f"{zh}完成")
    return "；".join(parts) + "。"


def _norm_source(value: Any) -> str:
    """来源名归一：小写 + 去全部空白——「10x Genomics」「10x genomics」「10xgenomics」同槽。"""
    return re.sub(r"\s+", "", str(value or "")).lower()


def _autofill_named_source(verb: str, raw: dict[str, Any], utterance: str) -> str | None:
    """点名源缺槽位的**确定性补位**：understand 掉 source
    槽位时，旧路只有 violation→repair——chat 会乖乖补，reasoner 却借题重推、把整个动词
    换成 sync（k11：补槽位 violation → repair 改判 sync+source → 撞 sync 主题闸 → repair
    预算耗尽 → AgentPlanInvalid 0/3）。quoted 里**逐字**点名了恰好一个来源时，补哪一个是
    唯一确定解——直接补上（validate trace 明示，不静默），把「掉槽位→repair 扯皮」整条
    失败链连根拔掉。quoted 没点名/点名多个 → 交回 violation 通道（歧义不由机械裁决）。"""
    if verb not in _NAMED_SOURCE_VERBS:
        return None
    if str(raw.get("source") or "").strip():
        return None
    quoted = str(raw.get("quoted") or "").strip()
    if not quoted or quoted not in (utterance or ""):
        return None
    in_quote = _named_sources_in(quoted)
    if len(in_quote) != 1:
        return None
    raw["source"] = in_quote[0]
    return in_quote[0]


def _sync_all_online_named(verb: str, raw: dict[str, Any], utterance: str) -> str:
    """点名单源场景下 sync 空槽的**如实留痕**（2026-08-15 审计 exec-gates）：
    半闸（`_NAMED_SOURCE_OPTIONAL_FILL_VERBS`）刻意放行空槽 sync——「不填 = 同步全部」是
    sync 的合法形态；但原话点名了来源时，实际写面是**全部在线源**、超出点名范围。语义
    不动（补确定解还是拒需先定口径），但 trace 与最终汇报都必须明说
    「按全部在线源同步」——不让「只同步了点名的那个源」成为可误读的默认。
    返回点名录（空串 = 无需留痕：非 sync / source 已填 / 原话没点名）。"""
    if verb != "curate.sync_updates":
        return ""
    if str(raw.get("source") or "").strip():
        return ""
    named = _named_sources_in(utterance)
    return "、".join(named) if named else ""


def _named_source_violation(verb: str, raw: dict[str, Any], utterance: str) -> str | None:
    """点名源一致性**机械校验**（安全围栏是代码，不依赖 LLM 自觉）。

    点名口径 = `_named_sources_in` 共享真源（别名匹配 + **受控规范名逐字**两趟）：原话点名的
    来源集合（**可能多个**——「先检查10x和ArrayExpress」的多步续步各有合法对象，
    2026-08-06 批B P1：只认第一个会把第二个点名源的续步恒误判违规、结构性截断多步链）；
    集合非空时 slots.source 归一后必须落在**任一点名源**的 {规范名 ∪ 全部别名} 归一集合里
    （填别名「10x」与填规范名「10x Genomics」都算对），否则返回人读 violation——
    走既有 violations→repair 通道，不在这里静默改写。单源场景行为与旧版逐位一致。
    规范名逐字（2026-08-08 探针豁免、2026-08-08 b08 升格为点名判定）：词表
    刻意不收裸「encode」（普通英文动词，见 search_request.SOURCE_ALIASES 注释），但用户
    **原样写出受控规范名**（全大写 ENCODE）就是点名——坐实：understand 槽位落空时
    此处无闸可拦，白跑一遍全量检查、把 decide 续步顶超 max_steps。"""
    if verb not in _NAMED_SOURCE_VERBS and verb not in _NAMED_SOURCE_OPTIONAL_FILL_VERBS:
        return None
    named = _named_sources_in(utterance)
    if not named:
        return None                                     
    canonicals: list[str] = list(named)
    acceptable: set[str] = set()
    for canonical in canonicals:
        acceptable.add(_norm_source(canonical))
        for source, aliases in _sr.SOURCE_ALIASES:
            if source == canonical:
                acceptable.update(_norm_source(alias) for alias in aliases)
                break
    names = "、".join(canonicals)
    filled = str(raw.get("source") or "").strip()
    if not filled:
                                          
        if verb in _NAMED_SOURCE_OPTIONAL_FILL_VERBS:
            return None
                                             
                                                                      
                                                  
        return (f"用户点名的是{names}，source 没填——点名来源时必填（填规范名；"
                "只补 source 槽位，动词本身不用换）。")
    if _norm_source(filled) not in acceptable:
        return (f"用户点名的是{names}，你填的 source 是「{filled}」——"
                "必须填用户点名的来源（填规范名）。")
    return None


                                     
_KW_STOP_TOKENS = ("data", "dataset", "datasets")


def _keyword_content_tokens(keywords: Any) -> list[str]:
    """keywords 的内容 token 化（**共享助手**，2026-08-08 从 `_ungrounded_keyword_tokens` 抽出：
    搜索覆盖闸 `_search_coverage_violation` 与汇报后检 `_denies_done_read` 复用同一份切词口径）。
    小写后按「英数段 / 连续中文段」切词，剔弱停用词——行为与原处逐位不变。"""
    return [t for t in re.findall(r"[a-z0-9]+|[一-鿿]+", str(keywords or "").lower())
            if t not in _KW_STOP_TOKENS]


def _ungrounded_keyword_tokens(keywords: Any, utterance: str) -> list[str]:
    """keywords 里**无法在接地语料里找到出处**的 token 清单（全接地 → 空表）。

    2026-08-06 批A D5（真机）：原话「看看有没有什么新数据，有的话拿回来」零主题词，
    decide 却提议 search_online(keywords="single cell")——**臆造参数触发真写库**（20 条落盘）。
    quoted 槽有逐字校验、keywords 槽此前只有「从原话提取」的 prompt 自觉——写操作的参数
    必须有机械校验。接地判定（宽进、只拦发明）：
      1. token 逐字出现在接地语料里（中英文原样都算）；
      2. token 命中 `vocabulary.CATALOG` 某词条的 alias/target，且该词条有别名出现在语料里
         （「人类肺」→ human/lung 合法翻译；语料没提单细胞时 single/cell 不接地）；
      3. 中文复合 token 内含接地词条别名（「人类肺数据」含「人类」）。

    接地语料 = 用户原话；**decide 续步时**追加已完成步骤的真实结果文本
    （2026-08-06 批C：「若有则下载」要下载的正是检查步骤发现的条目——出处之二，
    见 `_step_grounding_texts`；首步 steps 恒空，语料退化为纯原话，D5 口径不变）。
    """
    low = (utterance or "").lower()
    tokens = _keyword_content_tokens(keywords)
    if not tokens:
        return []
    entries: list[tuple[set[str], bool]] = []                                 
    for catalog_entries in _domain_catalog().values():
        for entry in catalog_entries:
            aliases = {str(a).lower() for a in entry.get("aliases", [])}
            aliases |= {str(t).lower() for t in entry.get("targets", [])}
            aliases = {a for a in aliases if len(a) >= 2 or any("一" <= ch <= "鿿" for ch in a)}
            if not aliases:
                continue
            mentioned = any(a in low for a in aliases)
            entries.append((aliases, mentioned))
    bad: list[str] = []
    for token in tokens:
        if token in low:
            continue
        ok = False
        for aliases, mentioned in entries:
            if not mentioned:
                continue
            if token in aliases or any(a in token or token in a for a in aliases):
                ok = True
                break
        if not ok:
            bad.append(token)
    return bad


def _step_grounding_texts(steps: list[dict] | None) -> list[str]:
    """已完成步骤**真实结果**里可供 keywords 接地的文本（decide 续步的出处之二，2026-08-06）。

    只收成功步骤的实有字段：check_updates 发现的疑似新增条目（accession + title）、
    search_online 的 query / sample_titles——「若有则下载」要下载的往往就是这些条目。
    失败步骤不产出处（error 是系统的话，不是数据）；db_status 的汇总数同样不产主题词。"""
    texts: list[str] = []
    for step in steps or []:
        if not step.get("ok"):
            continue
        result = step.get("result") or {}
        kind = step.get("card_kind")
        if kind == "check_updates":
            for src in result.get("sources") or []:
                for cand in src.get("new_candidates") or []:
                    for key in ("accession", "title"):
                        value = str(cand.get(key) or "").strip()
                        if value:
                            texts.append(value)
        elif kind == "search_online":
            query = str(result.get("query") or "").strip()
            if query:
                texts.append(query)
            for title in result.get("sample_titles") or []:
                value = str(title or "").strip()
                if value:
                    texts.append(value)
    return texts


def _validate_raw(raw: dict[str, Any], utterance: str,
                  steps: list[dict] | None = None) -> list[str]:
    """机械护栏的镜像校验：公共形状三条走 `action_plan.raw_shape_violations` 单一真源
    （2026-08-10 架构评审裁决落地——此前此处私拷一份还自称与 `_finalize`「口径
    一一对应」，评审坐实两路径已漂移）。本函数只叠加多步执行专属闸：点名源 /
    幻觉取消 / sync 主题 / keywords 接地。

    与 fallback 的 cancelled 语义分流是**刻意的设计决定**（不是漂移）：单步 fallback
    错收取消的最坏结果是用户再说一次（`action_plan._finalize` 照收 LLM 自报）；多步链
    里幻觉取消会当场杀链（j03 坐实），故本路径把「原话无否定语素的 cancelled=true」
    当 violation 拦下走 repair。"""
    violations: list[str] = _ap.raw_shape_violations(raw, utterance)
    verb = str(raw.get("verb") or "").strip() if raw else ""
    if verb and verb in _ap.VERB_BY_NAME:
        spec = _ap.VERB_BY_NAME[verb]
        named = _named_source_violation(verb, raw, utterance)
        if named:
            violations.append(named)
        if (spec.kind == _ap.EXEC and raw.get("cancelled") is True
                and not _DENIAL_MORPH_RE.search(utterance or "")):
                                               
                                                           
            violations.append(
                "cancelled 填了 true，但原话里没有任何「不做」的否定语素"
                "（不/别/取消/算了/停）——只有用户明确说不做某个动作时才能填 true；"
                "「要不要/能不能」是征询，照常执行、cancelled 填 false。")
        if verb == "curate.sync_updates":
                                      
                                                              
                                                       
                                         
                                                            
                                                           
                                                                    
                                  
                                                             
                                                               
                                                                      
                                                 
                                                            
                                                                   
            scope = _sync_gate_scope_zh(utterance, str(raw.get("quoted") or ""))
            topic = _utterance_topic_alias(scope)
            if topic:
                violations.append(
                    f"原话限定了主题（命中主题词「{topic}」）——sync_updates 不限主题，"
                    "会把所有疑似新增都入库；限定主题应该先用 curate.check_updates 检查、"
                    "再由 curate.search_online 按主题联网搜。")
            else:
                topic_else = _utterance_topic_alias(utterance)
                if topic_else and not _sync_topic_elsewhere_exempt(utterance, raw):
                    violations.append(
                        f"原话里有诉求限定了主题（命中主题词「{topic_else}」）——"
                        "sync_updates 不会按主题过滤，全量同步会把主题外的数据也入库；"
                        "且「有新增就同步」这类有条件的话以**先检查**为前提。"
                        "请先 curate.check_updates（填上它那一支点名的 source），"
                        "确认有新增后再按原话决定下一步。")
        if verb == "curate.search_online":
                                                          
                                                             
                                                        
                                                
            grounding = str(utterance or "")
            step_texts = _step_grounding_texts(steps)
            if step_texts:
                grounding += "\n" + "\n".join(step_texts)
            bad = _ungrounded_keyword_tokens(raw.get("keywords"), grounding)
            if bad:
                violations.append(
                    f"keywords 里的「{'、'.join(bad)}」在用户原话和已完成步骤的真实结果里都找不到出处——"
                    "主题词必须出自原话（可以翻译成英文），或逐字取自已完成步骤发现的真实条目；"
                    "两头都没有出处的不许搜。")
    return violations


                                                                                            
                                                         
                                                                                            


def _trace_entry(node: str, label_zh: str, detail: str, ok: bool, started: float) -> list[dict]:
    """一条 trace **增量**（reducer 语义：节点只返回新增的一条，合并由 channel 负责）。"""
    return [{
        "node": node,
        "label_zh": label_zh,
        "detail": detail,
        "ok": bool(ok),
        "ms": int((time.monotonic() - started) * 1000),
    }]


                                                                                          
 
                                                    
                                                                            
                                                                 
                                                               
                                                            
                                                    
                                                      

                                                                
                                               
                                                              
_HYBRID_ACTION_FAMILIES = _hybrid_policy.HYBRID_ACTION_FAMILIES
_HYBRID_ACTION_RES_EXTRA = _hybrid_policy.HYBRID_ACTION_RES_EXTRA
_HYBRID_ACTION_RES = _hybrid_policy.HYBRID_ACTION_RES
_HYBRID_LEXICON_VERSION = _hybrid_policy.HYBRID_LEXICON_VERSION
_split_hybrid_clauses = _hybrid_policy.split_clauses
_hybrid_action_hit = _hybrid_policy.action_hit
_hybrid_search_hit = _hybrid_policy.search_hit
_hybrid_intent_gate = _hybrid_policy.hybrid_intent_gate
_hybrid_required_capabilities = _hybrid_policy.required_capabilities


def _parse_route_vote(content: str, *, mode: str = "three_class") -> tuple[str, str, bool]:
    """单票解析（route_consensus）：{"route": ..., "reason": ...}——route 非法/解析不出
    都记**废票**（("", "", False)），废票不投给任何路线（不许把「没读懂」折算成 general
    的一票——兜底是显式规则，不是暗箱加权）。
    二元写闸模式：票面是 {"gate": "write"|"read"}——write → 「general」套件（全集含
    写工具），read → 「read」套件（general − 写工具）；返回的 route 是**套件名**，
    下游装配路径与三分类同构。"""
    obj = _ap.parse_action_response(str(content or ""))
    if mode == "binary":
        gate = str(obj.get("gate") or "").strip().lower()
        if gate not in ("write", "read"):
            return "", "", False
        return ("general" if gate == "write" else "read",
                str(obj.get("reason") or "").strip()[:80], True)
    route = str(obj.get("route") or "").strip()
    if route not in _CONSENSUS_ROUTES:
        return "", "", False
    return route, str(obj.get("reason") or "").strip()[:80], True


def _run_route_consensus(model: Any, prompt: str,
                         usage_sink: list | None = None,
                         *, mode: str = "three_class") -> tuple[str, list[dict]]:
    """分流共识（2026-08-17 钉死点 1：共识帽 + 机械兜底）：**并行** 2 次独立调用
    （温度岔开 0.0/0.8 保独立性——bind 不支持温度就退回原模型并在票上如实记 bound=False）；
    两张有效票一致即定；不一致加第 3 次；多数决（唯一最高票且 ≥2 张有效票）；三方平票、
    有效票不足或无有效票 → 机械兜底 general（安全地板，不许临场发挥）。返回 (route, votes)
    ——votes 是**全部原始投票**（温度/原文/解析结果），由节点落 trace 附加字段（错误
    分析的第一现场）。`usage_sink`：给了就把每票的缓存
    用量经 `_usage_record` 追加进去（读不到用量的替身/老 provider 自然跳过）。
    `mode`（2026-09-19 分流三改批）：binary → 系统提示词与票面解析走二元写闸
    （`prompts/route_write_gate.md`），机械兜底同样是 general（全集安全地板：
    能力不缺位，写工具另有 MAX_WRITE_STEPS/MAX_WRITE_RECORDS 预算闸硬约束）。"""
    from concurrent.futures import ThreadPoolExecutor

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (_route_write_gate_prompt() if mode == "binary"
                     else _route_consensus_prompt())

    def _vote(temp: float) -> dict:
        m, bound = model, False
        try:
            m = model.bind(temperature=temp)
            bound = True
        except Exception:
            m, bound = model, False
        vote: dict[str, Any] = {"temperature": temp, "bound": bound}
        t0 = time.monotonic()                                  
        try:
            answer = m.invoke([SystemMessage(content=system_prompt),
                               HumanMessage(content=prompt)])
            rec = _usage_record(answer, "route_consensus")
            if rec is not None and usage_sink is not None:
                usage_sink.append(rec)                                 
            content = getattr(answer, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(p.get("text") or "") for p in content if isinstance(p, dict))
            raw = str(content or "")
        except Exception as exc:
            vote.update({"raw": "", "route": "", "reason": "", "ok": False,
                         "error": type(exc).__name__,
                         "ms": int((time.monotonic() - t0) * 1000)})
            return vote
        route, reason, ok = _parse_route_vote(raw, mode=mode)
        vote.update({"raw": raw[:500], "route": route, "reason": reason, "ok": ok,
                     "ms": int((time.monotonic() - t0) * 1000)})
        return vote

    with ThreadPoolExecutor(max_workers=2) as pool:
        votes: list[dict] = list(pool.map(_vote, [0.0, 0.8]))
    valid = [v for v in votes if v["ok"]]
    if len(valid) == 2 and valid[0]["route"] == valid[1]["route"]:
        return valid[0]["route"], votes
    votes.append(_vote(0.5))
    valid = [v for v in votes if v["ok"]]
    if not valid:
        return "general", votes
    counts: dict[str, int] = {}
    for v in valid:
        counts[v["route"]] = counts.get(v["route"], 0) + 1
    top = max(counts.values())
    winners = [r for r, c in counts.items() if c == top]
                                                
                                            
                                               
    return (winners[0] if len(winners) == 1 and top >= 2 else "general"), votes


def _notify_route_verdict(ctx: Any, route: str) -> None:
    """：verdict hook **只做 abandoned/lazy 标记，不发射**（r3 ：
    节点内发射会产生 tool_start→preliminary→step 乱序；主路径唯一发射点 =
    understand 入口）。机械闸快进与 LLM 共识两条路径同调本助手。"""
    if getattr(ctx, "on_route_verdict", None) is not None:
        try:
            ctx.on_route_verdict(route)
        except Exception:
                                               
            _warn_once("route_verdict_hook",
                       "route_consensus verdict hook 抛异常（仅记类型），已忽略。")


def _l0_route_verdict(utterance: str, mode: str) -> tuple[str, dict] | None:
    """小模型级联第一级（2026-09-19 分流三改批·任务2；同日复刻批升级**双判官**）：
    高置信直采，返回 (套件名, 实录)；低置信/分歧/显式关闭/模型缺席/任何异常
    → None（升级 LLM 共识，现状行为逐位不变——fail-open 不成为新单点）。

    - binary（默认通道）：**双判官一致闸**——decider（Jev 复刻，语义模型）与
      L0（字符 n-gram 线性）判决一致即采（评测：OOD 一致 88.9% 覆盖下 96/96=100%、
      ID 98.2% 覆盖下 604/605=99.83%，一致性本身就是闸，不再叠 τ）；分歧 → None
      升级 LLM（分歧案例对错各半，正是需要 LLM 的地方）。只有一官可用时
      （CPU 部署无 GPU decider / 工件缺席）退化为单官 τ 闸（各自工件工作点）。
    - route3（兜底通道）：单链 prefer L0（其 OOD 83.3% 优于 decider 76.9%），τ 闸。
    τ 默认取工件内嵌工作点（cal 选 test 报），env `BIODATA_ROUTE_L0_TAU` 显式覆盖；
    级联默认开（`BIODATA_ROUTE_L0=0/off` 显式关闭）；decider 可用性见 decider.py。"""
    if not _env_default_on(_ROUTE_L0_ENV):
        return None
    try:
        if mode == "binary":
            decider_mode = _decider._env()
            d = _decider.score("binary", utterance) if decider_mode != "off" else None
            l = (None if decider_mode == "only"
                 else _l0.score(_l0.ROUTE_BINARY, utterance))
            hit: dict | None = None
            if d is not None and l is not None:
                if str(d["label"]) != str(l["label"]):
                    return None                     
                hit = {"label": d["label"], "impl": "dual",
                       "confidence": min(float(d["confidence"]), float(l["confidence"])),
                       "probs": d["probs"],
                       "dual": {"decider_conf": round(float(d["confidence"]), 6),
                                "l0_conf": round(float(l["confidence"]), 6)}}
                tau = 0.0            
            elif d is not None or l is not None:
                single, impl = (d, "decider") if d is not None else (l, "l0")
                assert single is not None
                tau = _decider.tau_with_fallback(
                    "binary", _ROUTE_L0_TAU_ENV, "tau", impl, 1.1)
                if float(single["confidence"]) < tau:
                    return None
                hit = dict(single)
                hit["impl"] = impl
            else:
                return None
        else:
            hit = _decider.score_with_fallback("route3", utterance,
                                               prefer=("l0", "decider"))
            if hit is None:
                return None
            tau = _decider.tau_with_fallback("route3", _ROUTE_L0_TAU_ENV, "tau",
                                             str(hit.get("impl") or ""), 1.1)
            if float(hit["confidence"]) < tau:
                return None
        label = str(hit["label"])
        suite = ({"write": "general", "read": "read"}.get(label) if mode == "binary"
                 else (label if label in _CONSENSUS_ROUTES else None))
        if suite is None:
            return None
        info = {"label": label, "confidence": round(float(hit["confidence"]), 6),
                "tau": round(tau, 6), "impl": str(hit.get("impl") or ""),
                "model": str(hit.get("impl") or "")}
        if hit.get("dual"):
            info["dual"] = hit["dual"]
        return suite, info
    except Exception:
        return None


def route_consensus(state: _AgentState, *, runtime: Any) -> dict:
    """环首分流共识节点（转正后常驻入图）：无条件进环后的第一次 LLM
    调用，只决定「装哪套工具 + 哪套系统提示词」——路由定义与示例在
    `prompts/route_consensus.md`（文件即真源）。输入 = 原话 + 会话短期上下文
    （与 understand 同上下文面）+ 初步检索概览（命中数/生效条件；**不含结果集**——
    诚实不变量：模型永远不直接碰结果集内容）。救回端点会把
    `route_scope="rescue"` 作为第四套件预置；该套件无需投票，但与其它套件共用同一图。
    2026-08-22：**混合诉求机械预闸**——`_hybrid_intent_gate` 检出同句同时含
    动作与检索信号时，跳过 LLM 投票直接 general（确定性、零调用成本；单意图句不触发，
    误伤率 0 优先）。"""
    started = time.monotonic()
    ctx: _AgentContext = runtime.context
    if str(state.get("route_scope") or "") == "rescue":
        _notify_route_verdict(ctx, "rescue")
        return {"route_scope": "rescue",
                "trace": _trace_entry("route_consensus", "分流共识",
                                      "端点已预置「rescue」套件，不发起分流投票。", True, started)}
    if isinstance(state.get("ask_answer"), dict):
                                                          
                                                     
        _notify_route_verdict(ctx, "search")
        return {"route_scope": "search",
                "trace": _trace_entry("route_consensus", "分流共识",
                                      "放松卡作答续跑：直定「search」套件，不发起分流投票。",
                                      True, started)}
    if getattr(ctx, "on_progress", None) is not None:
        ctx.on_progress("tool_start", {"verb": "node", "label_zh": "分流共识", "detail": ""})
                                        
                                            
                                                          
                                                                       
    if _hybrid_intent_gate(state["utterance"]):
        caps = _hybrid_required_capabilities(state["utterance"])
        entry = _trace_entry(
            "route_consensus", "分流共识",
            "机械意图闸：同句检出动作与检索两类信号 → 走「general」路线"
            "（混合诉求，未发起分流投票）。", True, started)[0]
        entry["route_votes"] = []               
        entry["required_capabilities"] = caps
        _te.emit_route_consensus_votes("general", [])
        _notify_route_verdict(ctx, "general")
        return {"route_scope": "general", "trace": [entry], "usage_ledger": [],
                "required_capabilities": caps}
    model = ctx.decide_model or ctx.chat_model
                                                         
                                         
    mode = _route_mode()
    l0_hit = _l0_route_verdict(state["utterance"], mode)
    if l0_hit is not None:
        l0_route, l0_info = l0_hit
        entry = _trace_entry(
            "route_consensus", "分流共识",
            f"L0 小模型高置信直采（{l0_info['label']} conf={l0_info['confidence']:.4f} "
            f"≥ τ={l0_info['tau']:.4f}）→ 走「{l0_route}」路线（未发起 LLM 投票）。",
            True, started)[0]
        entry["route_votes"] = []              
        entry["route_l0"] = l0_info
        _te.emit_route_consensus_votes(l0_route, [])
        _notify_route_verdict(ctx, l0_route)
        return {"route_scope": l0_route, "trace": [entry], "usage_ledger": []}
                                    
                                                       
                                           
                   
    context = _route_context_zh(state["utterance"],
                                has_results=state["has_results"],
                                result_total=state["result_total"],
                                retrieval=state.get("retrieval"),
                                current_query=state.get("current_query") or "",
                                current_filters=state.get("current_filters"))
                                         
                                             
    if getattr(ctx, "route_extra_zh", None):
        context = context + "\n" + str(ctx.route_extra_zh).strip()
                                 
                                    
    _ctx_card = _artifact_context_block_zh(state.get("artifact_context"))
    if _ctx_card:
        context = context + "\n\n" + _ctx_card
    context += getattr(ctx, "history_context", "")
    prompt = context + "\n\n----- 用户原话 -----\n" + state["utterance"]
    usage_local: list = []                                      
    route, votes = _run_route_consensus(model, prompt, usage_sink=usage_local, mode=mode)
    n_valid = sum(1 for v in votes if v.get("ok"))
    agreed = (n_valid == 2 and len(votes) == 2)
    detail = (f"{len(votes)} 票（有效 {n_valid}）→ 走「{route}」路线"
              + ("（二元写闸）" if mode == "binary" else "")
              + ("。" if agreed else "（分歧加投/机械兜底）。"))
    entry = _trace_entry("route_consensus", "分流共识", detail, True, started)[0]
    entry["route_votes"] = votes                                  
                                              
                                                             
                                      
    _te.emit_route_consensus_votes(route, votes)
    if _te.recorder_active():
        rc_model = ctx.decide_model_name if ctx.decide_model is not None else ctx.model_name
        for _v in votes:
            _te.emit_llm_call(
                node="route_consensus", model=rc_model, prompt=prompt,
                response=str(_v.get("raw") or ""), ms=int(_v.get("ms") or 0),
                channel="consensus_vote",
                fallback_reason=str(_v.get("error") or ""))
                                                       
                                     
    _notify_route_verdict(ctx, route)
    return {"route_scope": route, "trace": [entry], "usage_ledger": usage_local}


def _scoped_face_verbs(state: "_AgentState") -> frozenset[str]:
    """套件面 ∩ 请求级建议动作允许集（`state.allowed_verbs`）。

    understand/repair 的工具面、validate 的首步复核与 decide 的续步面共用这一处——
    不允许子集只进枚举/保底规划而不进执行边界：首步、续步、批量执行都必须吃同一道
    执行前机械闸（工具先执行、事后收窄会把「不执行」答成发生过的事）。"""
    scope = str(state.get("route_scope") or "")
    face = set(_SUITE_UNDERSTAND_VERBS.get(scope) or _SUITE_UNDERSTAND_VERBS["general"])
    allowed = state.get("allowed_verbs")
    if allowed:
        face &= set(allowed)
    return frozenset(face)


def _scoped_understand_face(state: "_AgentState"
                            ) -> tuple[list[dict] | None, dict[str, str] | None, list | None]:
    """按 state.route_scope 返回 understand/repair 套件收窄面。

    search/action/general/rescue 均经过这一条装配路径。"""
    tools, name_to_verb = _get_tool_specs()
    face = _scoped_face_verbs(state)
    tools = [t for t in tools
             if name_to_verb.get(str((t.get("function") or {}).get("name") or "")) in face]
    name_to_verb = {n: v for n, v in name_to_verb.items() if v in face}
    face_specs = [s for s in _ap.VERB_SPECS if s.verb in face]
    return tools, name_to_verb, face_specs


def understand(state: _AgentState, *, runtime: Any) -> dict:
    """意图理解，三级通道（共享助手 `_invoke_tool_channel`）：`tool_choice="required"` 强制档
    优先；模型 400 拒收强制档（思考模式模型实测如此）→ 自动档重试——留在结构化通道，
    槽位抽取质量不滑坡；provider 彻底不支持 tool-calling / 应答不可用 → 图内降级
    JSON-in-prompt（同一套 prompt 真源）再解析一次。
    通道如实标注：raw 来自 tool_calls 才算工具调用模式；provider 没发 tool_call 但 content
    本身是可解析 JSON 时走的是内容解析——mode/trace 不许谎称通道。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    started = time.monotonic()
    ctx: _AgentContext = runtime.context
                                      
                                                              
    if getattr(ctx, "on_progress", None) is not None:
        ctx.on_progress("tool_start", {"verb": "node", "label_zh": "理解意图", "detail": ""})
    if isinstance(state.get("ask_answer"), dict):
                                                          
                                                     
                                                           
                                                                    
                                                                       
                                                             
        answer = state.get("ask_answer") or {}
        label = str(answer.get("label") or "").strip()
        raw = {"verb": "rank", "query": state["utterance"], "quoted": state["utterance"]}
        _te.emit_understand_vote(raw, "forced")
        return {
            "raw": raw, "mode": "forced",
            "retrieval": state.get("retrieval"),
            "raw_batch": [], "usage_ledger": [],
            "trace": _trace_entry(
                "understand", "理解意图",
                "放松卡作答续跑：零 LLM 强置首步「关键词检索」"
                + (f"（用户选择：{label}）" if label else "")
                + "，按其忽略条件重跑原查询。", True, started),
        }
                                                        
    rescue = str(state.get("route_scope") or "") == "rescue"
    scoped_tools, scoped_names, face_specs = _scoped_understand_face(state)
    tools, name_to_verb = scoped_tools, scoped_names
                                              
                                                              
                                                                   
                                                    
    resolved = state.get("retrieval")
    if (resolved is None and getattr(ctx, "retrieval_provider", None) is not None
            and str(state.get("route_scope") or "") != "action"):
        try:
            resolved = ctx.retrieval_provider()
        except Exception:
                                                                      
                                                      
            resolved = {"status": "error", "total": 0, "top_titles": [],
                        "abstain_reason": "", "unresolved_terms": [], "note": "provider_error"}
    call_kwargs = dict(
        has_results=state["has_results"], result_total=state["result_total"],
        retrieval=resolved, current_query=state.get("current_query") or "",
        current_filters=state.get("current_filters"),
    )
    examples_zh = _examples_prompt_zh(
        _agent_project_root(), state["utterance"],
        principal=ctx.principal or "anonymous", endpoint_fp=ctx.endpoint_fp,
        available_tools=[spec.verb for spec in face_specs] if face_specs is not None else LOOP_TOOLS,
        **_learning_scope.context_kwargs(ctx))
    context = _context_zh(state["utterance"], examples_zh=examples_zh,
                          artifact_context=str(state.get("artifact_context") or ""),
                          **call_kwargs)
    prompt_kwargs: dict[str, Any] = dict(call_kwargs, examples_zh=examples_zh)
    if face_specs is not None:
                                                      
        prompt_kwargs["verbs"] = face_specs
    context += getattr(ctx, "history_context", "")
    json_prompt = _ap.build_action_prompt(state["utterance"], **prompt_kwargs) + getattr(ctx, "history_context", "")
                                            
                                                          
                                             
    _ctx_card = _artifact_context_block_zh(state.get("artifact_context"))
    if _ctx_card:
        json_prompt = json_prompt + "\n\n" + _ctx_card
    if rescue:
                                             
                                                    
                                                         
                                                            
        block = _rescue_block_zh((resolved or {}).get("unresolved_terms"))
        context += block
        json_prompt += block
    usage_local: list = []
                               
    system_zh = _SCOPED_TOOLS_SYSTEM_ZH
    answer, note, fb_reason, json_err = _invoke_tool_channel(
        ctx.chat_model, tools=tools,
        messages=[SystemMessage(content=system_zh), HumanMessage(content=context)],
        choice="required", json_prompt=json_prompt, refallback_on_empty=True,
        name_to_verb=name_to_verb, usage_sink=usage_local, usage_node="understand")
    if fb_reason:
        _audit_fallback(_agent_project_root(), "understand", fb_reason,
                        state["utterance"], ctx.model_name)
    raw = _raw_from_message(answer, name_to_verb) if answer is not None else {}
    had_tool_call = bool(getattr(answer, "tool_calls", None) or []) if answer is not None else False
    verb = str(raw.get("verb") or "")
    verb_spec = _ap.VERB_BY_NAME.get(verb)
    shown = verb_spec.zh if verb_spec else (verb or "（空）")
    if not fb_reason:
                                       
        detail = ("工具调用模式" if had_tool_call else "内容 JSON 模式") + note + f"，判为「{shown}」。"
        ok = True
        mode = "tools" if had_tool_call else "json"
    else:
                                                 
        detail = f"大模型的直连通道不可用（{fb_reason}），换一种问法再试一次。"
        if raw:
            detail += f"判为「{shown}」。"
        elif json_err:
            detail += f"再试一次也没拿到能用的回答（{json_err}），转入检查。"
        else:
            detail += "再试一次也没拿到能用的回答，转入检查。"
        ok = bool(raw)
        mode = "json"
                                               
                                                            
                                                 
                                                                
                                                       
                                                           
                                                  
                                   
    batch_extras: list[dict] = []
    answer_calls = list(getattr(answer, "tool_calls", None) or []) if answer is not None else []
    if not rescue and raw and len(answer_calls) > 1:
        batch_extras, _batch_dropped = _batch_readonly_extras(answer_calls, raw, state)
        if batch_extras:
            _rest = len(answer_calls) - 1 - len(batch_extras)
            detail += (f"（模型一次给了 {len(answer_calls)} 个调用，其中 {len(batch_extras)}"
                       f" 个只读且互相独立，随首步同批执行"
                       + (f"，其余 {_rest} 个回炉再判" if _rest > 0 else "") + "。）")
                                                     
                                                          
                                                            
                                            
                                                  
                                   
    checklist_update: dict[str, Any] = {}
    _score, _n_conn, _n_cond, _n_src = decide_complexity_score(state["utterance"])
    _multi = (_n_conn >= 1) or (_n_cond >= 1) or (_n_src >= 2)
    if (ctx.decide_lane == "complex" and raw and verb_spec is not None
            and verb_spec.kind == _ap.EXEC and _multi):
        tasks, dropped, err = _task_checklist_call(ctx.chat_model, state["utterance"],
                                                   usage_sink=usage_local)
        if err:
            tasks2, dropped2, err2 = _task_checklist_call(
                ctx.chat_model, state["utterance"], usage_sink=usage_local,
                feedback=f"失败原因：{err}")
            tasks, dropped = tasks2, dropped + dropped2
            err = err2
        if err:
            checklist_update["checklist_unavailable"] = err
        else:
            checklist_update["checklist"] = tasks
        if dropped:
            checklist_update["checklist_dropped"] = dropped
                                      
                                                          
    entry = _trace_entry("understand", "理解意图", detail, ok, started)[0]
    if checklist_update.get("checklist"):
        entry["checklist"] = checklist_update["checklist"]
    if checklist_update.get("checklist_unavailable"):
        entry["checklist_unavailable"] = checklist_update["checklist_unavailable"]
    if checklist_update.get("checklist_dropped"):
        entry["checklist_dropped"] = checklist_update["checklist_dropped"]
                                        
                                                                    
    _te.emit_understand_vote(raw, mode)
                                                                 
                                                                    
                                   
    return {
        "raw": raw, "mode": mode,
        "retrieval": resolved,
        "raw_batch": batch_extras,
        "usage_ledger": usage_local,
        **checklist_update,
        "trace": [entry],
    }


def validate(state: _AgentState, *, runtime: Any) -> dict:
    """护栏校验：机械口径（镜像 build_plan_from_raw 的处置），违规进 violations。
    通过则当场用**同一套** build_plan_from_raw 产出 plan（复用 import，不复制逻辑）。
    循环续步（steps 已非空）时产出挂 `loop_plan`——plan.verb 恒为首步动词（前端契约不炸）。
    校验前先跑 `_autofill_named_source`：quoted 逐字点名唯一来源时的 source 缺槽位是
    确定解补位（trace 明示），不再走 violation→repair 让强模型借题重推动词。"""
    started = time.monotonic()
    raw = dict(state.get("raw") or {})
    autofilled = _autofill_named_source(str(raw.get("verb") or ""), raw, state["utterance"])
    violations = _validate_raw(
        raw, state["utterance"], steps=list(state.get("steps") or []))
                                      
                                                
                                   
    if not list(state.get("steps") or []):
        verb_now = str(raw.get("verb") or "")
        if verb_now:
            face = _scoped_face_verbs(state)
            if verb_now not in face:
                scope = str(state.get("route_scope") or "")
                violations = list(violations) + [
                    f"本回合走「{scope or 'general'}」路线，首步只能选本路线的动作"
                    f"（{'、'.join(sorted(face))}）——「{verb_now}」不在本路线面内"]
    if violations:
        return {
            "violations": violations,
                                                   
            "raw_batch": None, "loop_batch": None,
            "trace": _trace_entry("validate", "合规检查",
                                  "；".join(violations), False, started),
        }
    plan = _ap.build_plan_from_raw(
        raw, state["utterance"],
        has_results=state["has_results"], result_total=state["result_total"],
        llm_status="ok",
    )
    key = "loop_plan" if state.get("steps") else "plan"
    detail = f"通过：{plan['verb_zh']}。"
    if autofilled:
        detail += f"（source 槽位按 quoted 点名补为 {autofilled}。）"
    sync_all_named = _sync_all_online_named(str(raw.get("verb") or ""), raw, state["utterance"])
    if sync_all_named:
                                                             
        detail += (f"（原话点名了{sync_all_named}，本步 source 未填——"
                   "按全部在线源同步，写面超出点名范围，如实留痕。）")
                                                        
                                                                 
                                                    
                                              
                                                         
                                                          
                                                                
                           
    batch_plans: list[dict] = []
    batch_drop_notes: list[str] = []
    for extra in list(state.get("raw_batch") or []):
        if not isinstance(extra, dict):
            batch_drop_notes.append("同批续步形状不是对象，已回炉")
            continue
        extra = dict(extra)
        if _raw_has_placeholder(extra):
            spec = LOOP_TOOLS.get(str(extra.get("verb") or ""))
            if spec is None:
                batch_drop_notes.append("同批续步不在注册表，已回炉")
                continue
            pos = extra.pop("_batch_pos", 0)
            batch_plans.append({
                "verb": str(extra.get("verb") or ""),
                "verb_zh": str(spec["label_zh"]),
                "deferred": True,
                "pos": int(pos or 0),
                "raw": extra,
            })
            continue
        _autofill_named_source(str(extra.get("verb") or ""), extra, state["utterance"])
        extra.pop("_batch_pos", None)
        extra_violations = _validate_raw(
            extra, state["utterance"], steps=list(state.get("steps") or []))
        if extra_violations:
            verb_spec = _ap.VERB_BY_NAME.get(str(extra.get("verb") or ""))
            batch_drop_notes.append(
                f"同批续步「{(verb_spec.zh if verb_spec else extra.get('verb'))}」没通过检查"
                f"（{'；'.join(extra_violations)}），已回炉待重判")
            continue
        batch_plans.append(_ap.build_plan_from_raw(
            extra, state["utterance"],
            has_results=state["has_results"], result_total=state["result_total"],
            llm_status="ok"))
    if batch_plans:
        n_deferred = sum(1 for p in batch_plans if p.get("deferred"))
        if n_deferred:
            detail += (f"同批另有 {len(batch_plans)} 个续步一并过检"
                       f"（其中 {n_deferred} 个占位接地，执行时解析）。")
        else:
            detail += f"同批另有 {len(batch_plans)} 个只读续步一并过检。"
    if batch_drop_notes:
        detail += "（" + "；".join(batch_drop_notes) + "。）"
    return {
        "violations": [], key: plan, "raw": raw,
        "raw_batch": None, "loop_batch": batch_plans,
        "trace": _trace_entry("validate", "合规检查", detail, True, started),
    }


def repair(state: _AgentState, *, runtime: Any) -> dict:
    """自我修正（≤1 次）：把 violations 原样喂回，让 LLM 在知情的前提下重答。
    2026-08-07 换装起同样走结构化工具通道（共享助手 auto 档）——旧版是无 bind 的散文
    调用，「结构化通道治槽位落空」在首次校验失败后又退回散文，正是槽位落空的问题温床。
    complex 车道的 repair 走
    decide 同档模型——understand 首答仍走 chat（快、克制类零暴露），但首答违规后的
    修正机会给强模型，把「误判→错误执行」拦在 execute 之前。"""
    from langchain_core.messages import HumanMessage

    started = time.monotonic()
    ctx: _AgentContext = runtime.context
    model = ctx.decide_model or ctx.chat_model
    lane_note = "长链档｜" if ctx.decide_model is not None else ""
                                     
    scoped_tools, scoped_names, face_specs = _scoped_understand_face(state)
    tools, name_to_verb = scoped_tools, scoped_names
    feedback = "；".join(state.get("violations") or [])
    prompt_kwargs: dict[str, Any] = dict(
        has_results=state["has_results"], result_total=state["result_total"],
        retrieval=state.get("retrieval"), current_query=state.get("current_query") or "",
        current_filters=state.get("current_filters"))
    if face_specs is not None:
        prompt_kwargs["verbs"] = face_specs
    json_prompt = _ap.build_action_prompt(state["utterance"], **prompt_kwargs) + getattr(ctx, "history_context", "")
    prompt = (
        json_prompt
        + "\n\n你上一次的输出违反了护栏：\n" + feedback
        + "\n请修正后重新作答（工具调用或一个 JSON 对象均可）。"
    )
    usage_local: list = []
    answer, _note, _fb, _je = _invoke_tool_channel(
        model, tools=tools, messages=[HumanMessage(content=prompt)],
        choice="auto", json_prompt=prompt, refallback_on_empty=False,
        name_to_verb=name_to_verb, usage_sink=usage_local, usage_node="repair")
    raw = _raw_from_message(answer, name_to_verb) if answer is not None else {}
    detail = ("已把不合规的地方告诉大模型，拿到了修正后的回答。" if raw
              else "已把不合规的地方告诉大模型，但修正后的回答还是读不懂。")
    return {
        "raw": raw,
        "repairs": int(state.get("repairs") or 0) + 1,
        "usage_ledger": usage_local,
        "trace": _trace_entry("repair", "让大模型改一版", lane_note + detail, bool(raw), started),
    }


def execute(state: _AgentState, *, runtime: Any) -> dict:
    """图内工具执行（2026-08-04 长程多步执行；取代 2026-08-03 的 observe 只读节点）：
    当前动词（首步 plan，循环续步 loop_plan）命中 LOOP_TOOLS 且未取消 → **真跑工具**
    （slots + 项目根）；run() 出口过 `agent_schemas.LOOP_RESULT_MODELS` 的**形状闸**，
    step 经 Step 模型构造再 model_dump 实录——state.steps 只返**增量**（reducer 合并），
    plan.steps 写**全量快照**（前端把非空 steps 当「后端已执行」的所有权令牌，契约不变）；
    工具抛异常 → ok=False 记 error（hint 原样），不炸图；每跑一个工具往联网账本追加一行
    审计。db_status 的产出同时挂 plan.observation + state.observations（既有契约不回归）。
    动词不在注册表 → 空过（同原 observe 语义：不为没有的事伪造步骤）。
    **同批只读续步**：decide 多调用采纳、validate 过检的 `loop_batch`
    紧随本步**逐个同口径执行**（同一道形状闸/审计/实录，一步不少、顺序不变）——只读
    白名单由 `_batch_readonly_extras` 把守，写库动词/回滚/换线永不进批（cite.export 带
    占位接地可进批，2026-08-20 ）；某一步失败不连坐其余独立只读步（各自如实记
    ok=False，decide 带全量结果再判）。**占位接地续步**：主步与续步的占位在
    本节点解析（`_resolve_placeholder_plan` 全闸助手），解析失败（resolver_error /
    dependency_unavailable）不执行、不记步、留 trace。"""
                                                
    on_progress = getattr(runtime.context, "on_progress", None)
    plan = dict(state.get("plan") or {})
    active = dict(state.get("loop_plan") or plan)
    verb = str(active.get("verb") or "")
    spec = LOOP_TOOLS.get(verb)
    if not spec:
        return {"last_ran": False, "pending_reask_write": False, "loop_batch": None}
    started = time.monotonic()
    if active.get("cancelled"):
        return {
            "last_ran": False,
            "pending_reask_write": False,
            "loop_batch": None,
            "trace": _trace_entry("execute", f"执行工具 · {spec['label_zh']}",
                                  "你说了不做，这一步已取消。", True, started),
        }
                                                      
                                                     
                                                              
    reask_increment: list[dict] = []
    if state.get("pending_reask_write") and not bool(spec.get("readonly")):
        reask_increment = [{"verb": verb, "verb_zh": str(active.get("verb_zh") or spec["label_zh"]),
                            "step_no": len(list(state.get("steps") or [])) + 1}]
    root = _agent_project_root()
    base_steps = list(state.get("steps") or [])
    new_steps: list[dict] = []
    new_trace: list[dict] = []
    dead_increments: list[dict] = []
    obs_increments: list[dict] = []
    any_failure = False
                                   
                                                            
                                                    
    usage_local: list = []
                                                
                                                                    
                                                              
                                                        
                                                            
    resolved_results: dict[int, dict] = {
        i: (s.get("result") or {})
        for i, s in enumerate(base_steps, start=1)
        if s.get("ok") and isinstance(s.get("result"), dict)
    }
    main_ordinal = len(base_steps) + 1                                
    has_ph = _slots_has_placeholder(active.get("slots")) or any(
        isinstance(x, dict) and x.get("deferred")
        for x in (state.get("loop_batch") or []))
    batch_id = _next_batch_id() if has_ph else ""
    if has_ph and _slots_has_placeholder(active.get("slots")):
                                                         
                                                          
                                                                            
        plan2, ph_note, skip_reason = _resolve_placeholder_plan(
            {"verb": verb, **dict(active.get("slots") or {}),
             **{k: active.get(k) for k in ("quoted", "confidence", "reason")
                if active.get(k) is not None}},
            resolved_results, batch_id, main_ordinal,
            state["utterance"], bool(state["has_results"]), int(state["result_total"] or 0),
            steps=base_steps)
        if skip_reason:
            new_trace.extend(_trace_entry(
                "execute", f"批内依赖跳过 · {spec['label_zh']}", ph_note, False, started))
                                                                 
                                                   
            return {"last_ran": True, "pending_reask_write": False,
                    "loop_batch": None, "trace": new_trace,
                    "usage_ledger": usage_local}
        active = plan2
        verb = str(active.get("verb") or "")
        spec = LOOP_TOOLS.get(verb)
        if spec is None:
            return {"last_ran": False, "pending_reask_write": False,
                    "loop_batch": None, "trace": new_trace,
                    "usage_ledger": usage_local}
        if ph_note:
            new_trace.extend(_trace_entry(
                "execute", f"批内依赖解析 · {spec['label_zh']}", ph_note, True, started))

    def _trace_step(step: dict) -> None:
        """**全部**动词的工具步留痕——step 实录的机器可读持久化
        副本 + 预算计数现场（与裁决层同一批现算函数、同一真源）。fail-soft，OFF 零构造。"""
        if not _te.recorder_active():
            return
        live = base_steps + new_steps
        _te.emit_tool_call(
            verb=str(step.get("verb") or ""), slots=step.get("slots"),
            ok=bool(step.get("ok")), error_code=step.get("error_code"),
            ms=int(step.get("ms") or 0), card_kind=str(step.get("card_kind") or ""),
            readonly=bool(step.get("readonly")),
            budgets={"steps": len(live),
                     "write_steps": _write_steps_used(live),
                     "write_records": _write_records_used(live),
                     "search_rerun": _search_rerun_used(live)})

    def _trace_snapshot_finalize(sid: str | None, v: str,
                                 mutation: dict | None = None,
                                 patch_mutations: list | None = None) -> None:
        """写动词操作后 finalize 并发 state_snapshot；trace 故障不掀翻主流程。
        目录模式按工具结果归属；补丁模式只认 RMW 临界区的真实收据，失败后已提交的
        写入也如实留锚，无写入的失败/零写成功用空收据。"""
        if sid is None:
            return
        try:
            if patch_mutations is not None:
                mutation = dict(mutation or {}, patch=patch_mutations)
            store = _trace_snapshot_store(root)
            diff = store.finalize(sid, mutation=mutation)
            meta = store.load(sid)
            _te.emit_state_snapshot(
                snapshot_id=sid, verb=v,
                created=diff["created"], modified=diff["modified"],
                deleted=diff["deleted"],
                preimage_missing=list(meta.get("preimage_missing") or []))
        except Exception as exc:
            _warn_once(f"trace_snapshot_finalize::{type(exc).__name__}",
                       "trace 快照 finalize 失败（仅记异常类型），本步回退锚缺失。")

    def _run_one(act: dict) -> dict | None:
        """单个 loop 工具的执行全路径（形状闸/审计/实录/trace）——主步与同批续步共用，
        每调用一次，单呼叫路径的产出与旧版逐位一致。返回本步实录（成功/失败都返回，
        供批的解析源 `resolved_results` 现取 result——不进 state reducer）。"""
        nonlocal any_failure
        v = str(act.get("verb") or "")
        sp = LOOP_TOOLS[v]
                                          
                                                             
                                   
        allowed = state.get("allowed_verbs")
        if allowed and v not in set(allowed):
            new_trace.extend(_trace_entry(
                "execute", f"建议动作收窄 · {sp['label_zh']}",
                "这一步不在本次建议动作允许的能力里，没有执行；按建议动作面继续。",
                True, time.monotonic()))
            return None
        slots = dict(act.get("slots") or {})
        verb_zh = str(act.get("verb_zh") or sp["label_zh"])
                                                     
                                                           
                                           
        if on_progress is not None:
            on_progress("tool_start", {
                "verb": v, "label_zh": f"执行工具 · {sp['label_zh']}", "detail": ""})
        t0 = time.monotonic()
                                                    
                                                                  
                                        
        sid: str | None = None
        if not sp.get("readonly") and _te.recorder_active():
            try:
                sid = _trace_snapshot_store(root).capture(v, preimage_paths=[])
            except Exception as exc:
                _warn_once(f"trace_snapshot_capture::{type(exc).__name__}",
                           "trace 快照 capture 失败（仅记异常类型），本步无回退锚。")
        patch_mutations: list = []

        def _run_tool(*args):
            if sid is None:
                return sp["run"](*args)
            from ..corpus.patch_package import collect_patch_mutations

            with collect_patch_mutations() as receipts:
                try:
                    return sp["run"](*args)
                finally:
                    patch_mutations.extend(receipts)

        try:
            if sp.get("needs_context"):
                                                                   
                                                                         
                                                 
                                                             
                                                          
                                                        
                                              
                                                        
                                                          
                                                       
                                                                             
                                                                         
                result = _run_tool(slots, root, {
                    "current_query": str(state.get("current_query") or ""),
                    "search_sources": state.get("search_sources"),
                    "search_facet_filters": state.get("search_facet_filters"),
                    "search_suppressed_constraints": state.get("search_suppressed_constraints"),
                    "search_lenient_dims": state.get("search_lenient_dims"),
                    "search_date_from": str(state.get("search_date_from") or ""),
                    "search_date_to": str(state.get("search_date_to") or ""),
                    "search_top_k": state.get("search_top_k"),
                    "search_recall": str(state.get("search_recall") or ""),
                    "search_strategy": str(state.get("search_strategy") or ""),
                    "replace_screen": str(state.get("route_scope") or "") == "rescue",
                    "unresolved_terms": list(
                        (state.get("retrieval") or {}).get("unresolved_terms") or []),
                    "chat_model": getattr(runtime.context, "chat_model", None),
                    "utterance": str(state.get("utterance") or ""),
                    "usage_sink": usage_local,
                    "steps": base_steps + new_steps,
                    "has_results": bool(state.get("has_results")),
                    "result_total": int(state.get("result_total") or 0),
                                                   
                                                        
                    "write_records_remaining": max(
                        0, MAX_WRITE_RECORDS - _write_records_used(base_steps + new_steps)),
                })
            else:
                result = _run_tool(slots, root)
                                     
                                                                  
                                                         
                                                     
                                         
            result_model = _LOOP_RESULT_MODELS.get(v)
            if result_model is not None:
                result_model.model_validate(result)
        except Exception as exc:
                                                                  
                                                               
            if isinstance(exc, _ValidationError):
                                                         
                                                                
                hint = "工具返回的结果不符合登记的形状契约，已按失败如实记录。"
                code = "bad_result_shape"
            else:
                hint = str(getattr(exc, "hint", None) or str(exc))
                code = str(getattr(exc, "code", None) or type(exc).__name__)
                                                                        
                                                               
            step = _Step(
                verb=v, verb_zh=verb_zh, slots=slots, ok=False,
                error=hint, error_code=code,
                card_kind=sp["card_kind"], readonly=bool(sp.get("readonly")),
                ms=int((time.monotonic() - t0) * 1000),
                snapshot_id=sid,
            ).model_dump(exclude_none=True)
            new_steps.append(step)
            _audit_loop_tool(root, v, slots, False, str(exc))
            any_failure = True
            _trace_snapshot_finalize(sid, v, mutation=_empty_mutation_receipt(),
                                     patch_mutations=patch_mutations)
            _trace_step(step)
                                                           
                                                               
            if code in _TERMINAL_STEP_CODES:
                dead_increments.append({
                    "verb": v, "code": code,
                    "source": _norm_source(slots.get("source")),
                })
            new_trace.extend(_trace_entry("execute", f"执行工具 · {sp['label_zh']}",
                                          f"工具执行失败：{hint}", False, t0))
            return step
        step = _Step(
            verb=v, verb_zh=verb_zh, slots=slots, ok=True,
            result=result, card_kind=sp["card_kind"],
            readonly=bool(sp.get("readonly")),
            ms=int((time.monotonic() - t0) * 1000),
            snapshot_id=sid,
        ).model_dump(exclude_none=True)
        new_steps.append(step)
                                                    
                                                  
        _trace_snapshot_finalize(sid, v, mutation=_mutation_receipt_of_result(v, result),
                                 patch_mutations=patch_mutations)
        _trace_step(step)
        if sp.get("observation"):
                                                      
                                                                       
                          
            plan["observation"] = result
            obs_increments.append({"verb": v, "data": result})
                                                                      
        rec_count = result.get("record_count") if isinstance(result, dict) else None
        _audit_loop_tool(root, v, slots, True, "ok", records=rec_count)
        new_trace.extend(_trace_entry("execute", f"执行工具 · {sp['label_zh']}",
                                      _execute_detail_zh(sp, result), True, t0))
        return step
                                          
    main_step = _run_one(active)
    if main_step and main_step.get("ok") and isinstance(main_step.get("result"), dict):
        resolved_results[main_ordinal] = main_step["result"]
                                                           
                                                      
                                                   
                                    
    extras = list(state.get("loop_batch") or [])
    if _ask_relax_pending(base_steps + new_steps) is not None and extras:
        new_trace.extend(_trace_entry(
            "execute", "环已暂停等作答",
            f"放松选择卡已弹出：本批剩余 {len(extras)} 个续步不再执行，"
            "待用户作答后以新回合回环续跑。", True, time.monotonic()))
        extras = []
                                                         
                        
                                       
                                                 
                                                     
                                                      
    for loop_idx, extra in enumerate(extras):
        if not (isinstance(extra, dict) and not extra.get("cancelled")):
            continue
        ev = str(extra.get("verb") or "")
        espec = LOOP_TOOLS.get(ev)
        if espec is None:
            continue
        live_steps = base_steps + new_steps
        act = extra
                                                             
                                                                 
        batch_pos = len(base_steps) + 1 + (loop_idx + 1)
        if extra.get("deferred"):
                                                            
                                                                          
                                   
            raw0 = dict(extra.get("raw") or {})
            plan2, resolve_note, skip_reason = _resolve_placeholder_plan(
                raw0, resolved_results, batch_id, batch_pos,
                state["utterance"], bool(state["has_results"]),
                int(state["result_total"] or 0), steps=live_steps)
            if skip_reason:
                                                                            
                                                                 
                                                                    
                           
                new_trace.extend(_trace_entry(
                    "execute", f"批内依赖跳过 · {espec['label_zh']}",
                    resolve_note, False, time.monotonic()))
                continue
            act = plan2
            if resolve_note:
                                                     
                                                       
                new_trace.extend(_trace_entry(
                    "execute", f"批内依赖解析 · {espec['label_zh']}",
                    resolve_note, True, time.monotonic()))
            ev = str(act.get("verb") or "")
            espec = LOOP_TOOLS.get(ev)
            if espec is None:
                continue
                                            
                                                           
                                                
                                                         
        if ev in _failed_tool_ban(live_steps):
            new_trace.extend(_trace_entry(
                "execute", f"批内熔断 · {espec['label_zh']}",
                "同批续步未执行：该动作此前已连续失败两次被禁提，留待下一轮带新状态重判。",
                False, time.monotonic()))
            continue
        if ev in _NETWORK_LOOP_TOOLS \
                and _is_network_call(ev, (act.get("slots") or {}).get("source")) \
                and _network_moratorium(live_steps):
            new_trace.extend(_trace_entry(
                "execute", f"批内熔断 · {espec['label_zh']}",
                "同批续步未执行：联网二连败触发联网暂停，留待下一轮带新状态重判。",
                False, time.monotonic()))
            continue
                                                      
                                                    
                                                        
        if ev == "rank" and _rank_used(live_steps) >= MAX_RANK:
            new_trace.extend(_trace_entry(
                "execute", f"批内熔断 · {espec['label_zh']}",
                "同批续步未执行：新检索预算已用完，留待下一轮带新状态重判。",
                False, time.monotonic()))
            continue
        if ev == "rerank" and _rerank_used(live_steps) >= MAX_RERANK:
            new_trace.extend(_trace_entry(
                "execute", f"批内熔断 · {espec['label_zh']}",
                "同批续步未执行：优化重检预算已用完，留待下一轮带新状态重判。",
                False, time.monotonic()))
            continue
                                            
                                                  
        if ev in _PLACEHOLDER_SLOTS and (
                (ev == "compare.datasets" and _compare_used(live_steps) >= MAX_COMPARE)
                or (ev == "cite.export" and _cite_export_used(live_steps) >= MAX_CITE_EXPORT)
                or (ev == "compat.find" and _compat_used(live_steps) >= MAX_COMPAT)
                or (ev == "fair.check" and _fair_used(live_steps) >= MAX_FAIR)):
            new_trace.extend(_trace_entry(
                "execute", f"批内熔断 · {espec['label_zh']}",
                "同批续步未执行：该工具的独立预算已用完，留待下一轮带新状态重判。",
                False, time.monotonic()))
            continue
        step = _run_one(act)
        if step and step.get("ok") and isinstance(step.get("result"), dict):
            resolved_results[batch_pos] = step["result"]
                                              
        if _ask_relax_pending(base_steps + new_steps) is not None:
            remaining = len(extras) - (loop_idx + 1)
            if remaining > 0:
                new_trace.extend(_trace_entry(
                    "execute", "环已暂停等作答",
                    f"放松选择卡已弹出：本批剩余 {remaining} 个续步不再执行，"
                    "待用户作答后以新回合回环续跑。", True, time.monotonic()))
            break
                                                           
                                
    plan["steps"] = base_steps + new_steps
    out: dict[str, Any] = {
        "plan": plan, "steps": new_steps, "loop_plan": None, "loop_batch": None,
        "last_ran": True,
        "reask_writes": reask_increment, "pending_reask_write": False,
        "usage_ledger": usage_local,
    }
    if any_failure:
        out["dead_ends"] = dead_increments
    if obs_increments:
        out["observations"] = obs_increments
                                          
                                                              
                                                  
               
    for s in new_steps:
        if str(s.get("verb") or "") == "route.request" and s.get("ok"):
            target = str(((s.get("result") or {}).get("requested_route")) or "")
            if target in _ROUTE_REQUEST_TARGETS:
                out["route_scope"] = target
    out["trace"] = new_trace
    return out


def _narrow_decide_face(specs: list[dict], names: dict[str, str],
                        allowed_verbs: Any) -> tuple[list[dict], dict[str, str]]:
    """decide 工具面按建议动作允许集收窄（续步/批量共用首步的同一道面）。

    - 控制工具（finish / unsupported_next_step 不在 `names` 映射里）与 route.request
      逃生口保留——它们不直接执行动作；
    - 动作工具须在允许集内，面外工具从工具表里消失（提示层收窄；执行面兜底在 execute）。"""
    if not allowed_verbs:
        return specs, names
    allowed = set(allowed_verbs)

    def _kept(name: str) -> bool:
        verb = names.get(name)
        return verb is None or verb in allowed or verb == "route.request"

    return (
        [t for t in specs if _kept(str((t.get("function") or {}).get("name") or ""))],
        {n: v for n, v in names.items() if _kept(n)},
    )


def decide(state: _AgentState, *, runtime: Any) -> dict:
    """判断下一步（仅 execute 真跑过工具后进入）：**主通道 tool-calling**（2026-08-07 换装；
    取代 2026-08-04 的散文 JSON 主通道）——LLM 看「原话 + 已完成步骤紧凑投影」，绑
    5 个 loop 工具 + finish + unsupported_next_step（`_DECIDE_TOOL_SPECS`）：
    回 loop 工具 = 续步提议；finish = done；unsupported = 婉拒表外动作；
    幻觉工具名 / 散文 / 全通道没拿到 = 非法；**多 tool_call 取第一个**（DeepSeek 不遵守 parallel_tool_calls=False，实测 decide 不可读
    17/17 全是多调用、第一个调用 17/17 合法续步——批量调用是「规划先行」，后续动作
    循环会带新状态再判断，吃第一个不吞事；留痕「一次给了 N 个」拼进 trace）。
    **2026-08-14 起同批只读消费**：被采纳应答的第 2..N 个调用里，只读白名单
    （check_updates/db_status）且互相独立的续步经 `_batch_readonly_extras` 机械过滤
    （参数键严格/逐个裁决闸/去重/步数预算）后随首步**同批执行**（raw_batch → validate
    同口径复检 → execute 逐个真跑留痕）；写动词/幻觉名出现即截断、整尾回炉再判——
    与「取第一个」旧策的唯一差别是只读独立续步不再白等一轮模型往返。
    **仅调用本身抛异常**才跌散文 JSON 兜底。主通道拿到非法应答 → **重问一次**
    （非法一刀切停环是长链断裂的最大
    单因）。重问后的写动词处置：**放行 + 强制核销复核**
    （纯只读闸会把「首答散文→重问答对但首选写动词」的长链稳定截断：误杀代价每次
    必发生，而放行风险有参数校验 / faithful 机械后检 / 核销硬闸三道防线）：放行的写步
    落 `reask_writes` 台账，finish 的 completion_report 必须引用其步骤号单独交代结果，
    否则核销硬闸拒收收尾（每请求至多回灌一次，第二次 fail-safe 接受并如实标注）。
    重问仍非法 → 照旧停环；JSON 兜底档的非法应答不多问——保命档不加门槛。
    机械校验双保险 + MAX_STEPS 硬上界
    + 连续失败处置二分（联网二连败改**联网暂停**：prompt 注入禁令 + 联网提议机械拒绝，
    离线事项不误伤；非网络码二连败维持硬停）。
    「若有则…」的条件语义由 LLM 看真实结果自然处理（如 new_count=0 → done）。
    2026-08-08 核销硬闸（finish 必附 completion_report
    `_unfinished_business` 扫出报告自认「没做」的事项且本请求尚未否决过 → 拒收收尾、
    把缺口回灌重问一次（同一 `_invoke_tool_channel` 通道）；第二次 finish 仍自认未完成
    → fail-safe 接受，trace 如实标注。JSON 兜底档的 {"done": true} 不过本闸（保命档
    不再加门槛，见 `_unfinished_business` docstring）。
    核销闸升级（`_completion_report_veto` 三形态，反馈文案按形态区分）：
    形态 A「已做」无合法步骤号（b08/k01——没跑 db_status 却自称告知库容）与形态 B
    豁免行夹带依赖借口词（k03/k08——彼此独立的事拿前件失败当理由）与「自认没做」同罪。"""
    from langchain_core.messages import AIMessage, HumanMessage

    started = time.monotonic()
    ctx: _AgentContext = runtime.context
                                             
                                              
    model = ctx.decide_model or ctx.chat_model
    model_name = ctx.decide_model_name if ctx.decide_model is not None else ctx.model_name
    lane_note = "长链档｜" if ctx.decide_model is not None else ""
    usage_local: list = []                                      
    steps = list(state.get("steps") or [])
    if len(steps) >= MAX_STEPS:
                                      
                                                            
                                                   
                                             
                                                 
                                                       
                                                 
                                                      
                                                     
        settled = False
        try:
            settled = (not _pending_violations(str(state.get("utterance") or ""), steps)
                       and not _checklist_unsettled(list(state.get("checklist") or []), steps,
                                                    str(state.get("declined_zh") or ""))
                       and not _capabilities_unsettled(
                           list(state.get("required_capabilities") or []), steps,
                           str(state.get("declined_zh") or ""))
                                                            
                                                           
                       and not _intent_checklist_unsettled(
                           list(state.get("intent_checklist") or []), steps, "")
                       and all(s.get("ok") for s in steps))
        except Exception:
            settled = False
                                                     
                                                                  
                                   
                                                                         
                                     
        degrade_note = ""
        if state.get("checklist_unavailable"):
            degrade_note = ("；注意：事项清单当时没建成（checklist_unavailable），"
                            "本次结算只按 pending 三道口径核验，清单对账缺席。")
        return {
            "loop_next": False,
            "usage_ledger": usage_local,
                                                              
            "truncated": True,
            "truncated_settled": settled,
            "trace": _trace_entry("decide", "判断下一步",
                                  f"已连续执行 {len(steps)} 步（最多 {MAX_STEPS} 步），到此收尾。"
                                  + ("结算：原话交代的事已全部执行。" if settled else "")
                                  + degrade_note,
                                  True, started),
        }
                                        
                                                            
                                                                       
                                                                  
                                                                  
                                                     
                                             
                                                               
                                                
                                                      
    moratorium = _network_moratorium(steps)
    ban_verbs = frozenset() if moratorium else _failed_tool_ban(steps)
    write_budget_out = _write_steps_used(steps) >= MAX_WRITE_STEPS
    rerun_budget_out = _search_rerun_used(steps) >= MAX_SEARCH_RERUN
                                   
                                                     
                                                  
    scope = str(state.get("route_scope") or "general")
    route_req_out = _route_request_used(steps) >= MAX_ROUTE_REQUEST
    decide_specs = list(_DECIDE_TOOL_SPECS_BY_SUITE.get(scope)
                        or _DECIDE_TOOL_SPECS_BY_SUITE["general"])
    decide_names = dict(_DECIDE_TOOL_NAME_TO_VERB_BY_SUITE.get(scope)
                        or _DECIDE_TOOL_NAME_TO_VERB_BY_SUITE["general"])
                                                
                                                                         
                    
    decide_specs, decide_names = _narrow_decide_face(
        decide_specs, decide_names, state.get("allowed_verbs"))
    if route_req_out:
        decide_specs = [t for t in decide_specs
                        if str((t.get("function") or {}).get("name") or "")
                        != "route_request"]
        decide_names.pop("route_request", None)
    rules_pair = (_SCOPED_DECIDE_RULES_BY_SUITE.get(scope)
                  or _SCOPED_DECIDE_RULES_BY_SUITE["general"])
    tools_rules, json_rules = rules_pair["tools"], rules_pair["json"]
                                                    
    reask_nos = {int(m.get("step_no")) for m in (state.get("reask_writes") or [])
                 if isinstance(m.get("step_no"), int) and not isinstance(m.get("step_no"), bool)}
    projections = json.dumps([_step_projection(s, reasked_write=((i + 1) in reask_nos))
                              for i, s in enumerate(steps)], ensure_ascii=False)
                                                       
                              
    searched_block = _searched_topics_block_zh(steps)
                                
    moratorium_block = _NETWORK_MORATORIUM_BLOCK_ZH if moratorium else ""
                                   
    ban_block = _failed_tool_ban_block_zh(ban_verbs) if ban_verbs else ""
                                 
    write_block = _WRITE_BUDGET_BLOCK_ZH if write_budget_out else ""
                                              
    rerun_block = _SEARCH_RERUN_BUDGET_BLOCK_ZH if rerun_budget_out else ""
                                                   
    rank_block = _RANK_BUDGET_BLOCK_ZH if _rank_used(steps) >= MAX_RANK else ""
    rerank_block = _RERANK_BUDGET_BLOCK_ZH if _rerank_used(steps) >= MAX_RERANK else ""
                                                              
                                                  
                                        
    show_block = _SHOW_BUDGET_BLOCK_ZH if _show_used(steps) >= MAX_SHOW else ""
                                                      
    ask_relax_block = (_ASK_RELAX_BUDGET_BLOCK_ZH
                       if _ask_relax_used(steps) >= MAX_ASK_RELAX else "")
                                              
    resume_block = _ask_answer_resume_block_zh(state.get("ask_answer"))
    screen_alert_block = _screen_alert_block_zh(steps)
                                              
    compare_block = _COMPARE_BUDGET_BLOCK_ZH if _compare_used(steps) >= MAX_COMPARE else ""
    cite_block = _CITE_EXPORT_BUDGET_BLOCK_ZH if _cite_export_used(steps) >= MAX_CITE_EXPORT else ""
    compat_block = _COMPAT_BUDGET_BLOCK_ZH if _compat_used(steps) >= MAX_COMPAT else ""
    fair_block = _FAIR_BUDGET_BLOCK_ZH if _fair_used(steps) >= MAX_FAIR else ""
                                               
    route_req_block = (_ROUTE_REQUEST_BUDGET_BLOCK_ZH
                       if scope != "rescue" and route_req_out else "")
                                                 
    rescue_block = _RESCUE_DECIDE_BLOCK_ZH if scope == "rescue" else ""
                                      
    failed_block = _FAILED_STEP_BLOCK_ZH if any(not s.get("ok") for s in steps) else ""
                                      
    pending_block = _pending_hints_block_zh(state["utterance"], steps)
                                                           
    status_block = _agent_status_block_zh(state, steps=steps, moratorium=moratorium,
                                          ban_verbs=ban_verbs)
    rules_block = _knowledge_context_safe(
        _agent_project_root(), state["utterance"],
        principal=ctx.principal or "anonymous", endpoint_fp=ctx.endpoint_fp,
        project_id=_learning_scope.resolve_project_id(getattr(ctx, 'project_id', None)),
        phase='decide', available_tools=set(decide_names.values()))
    tools_prompt = (
        tools_rules
        + rules_block
        + getattr(ctx, "history_context", "")
        + "\n----- 用户原话 -----\n" + state["utterance"]
        + "\n----- 已完成步骤（JSON）-----\n" + projections
        + searched_block
        + moratorium_block
        + ban_block
        + write_block
        + rerun_block
        + rank_block
        + rerank_block
        + show_block
        + ask_relax_block
        + resume_block
        + screen_alert_block
        + compare_block
        + cite_block
        + compat_block
        + fair_block
        + route_req_block
        + rescue_block
        + failed_block
        + pending_block
        + status_block
    )
    json_prompt = (
        json_rules
        + rules_block
        + getattr(ctx, "history_context", "")
        + "\n----- 用户原话 -----\n" + state["utterance"]
        + "\n----- 已完成步骤（JSON）-----\n" + projections
        + searched_block
        + moratorium_block
        + ban_block
        + write_block
        + rerun_block
        + rank_block
        + rerank_block
        + show_block
        + ask_relax_block
        + resume_block
        + screen_alert_block
        + compare_block
        + cite_block
        + compat_block
        + fair_block
        + route_req_block
        + rescue_block
        + failed_block
        + pending_block
        + status_block
    )
    answer, _note, fb_reason, _je = _invoke_tool_channel(
        model, tools=decide_specs,
        messages=[HumanMessage(content=tools_prompt)],
        choice="auto", json_prompt=json_prompt, refallback_on_empty=False,
        usage_sink=usage_local, usage_node="decide")
    if fb_reason:
                                                                       
                                  
        _audit_fallback(_agent_project_root(), "decide", fb_reason,
                        state["utterance"], model_name)
    multi_holder: list[str] = []                                
    batch_holder: list[list] = []                                  
    ncalls_holder: list[int] = []                                  

    def _classify(ans: Any) -> tuple[str, Any]:
        """应答分诊 + 多调用留痕（DeepSeek 不遵守
        `parallel_tool_calls=False`，一次回 ≥2 个调用是常态而非非法——`_decide_answer_kind`
        取第一个，这里把「一次给了 N 个」如实拼进 trace）。每次分诊先清 holder：
        最终采用的那次分诊（最后一次调用）决定留痕与否，中间被重问掉的应答不留痕。"""
        multi_holder.clear()
        batch_holder.clear()
        ncalls_holder.clear()
        k, p = ("invalid", None) if ans is None else _decide_answer_kind(
            ans, decide_names)
        n_calls = len(getattr(ans, "tool_calls", None) or [])
        if n_calls > 1 and k != "invalid":
            multi_holder.append(f"模型一次给了 {n_calls} 个调用，按顺序先执行第一个；")
            ncalls_holder.append(n_calls)
        if k == "loop" and n_calls > 1:
            batch_holder.append(list(getattr(ans, "tool_calls", None) or []))
        return k, p

    kind, payload = _classify(answer)
    reask_note = ""                                             
    reasked = False                                    
    pending_write = False                                     
    write_note = ""                                    
    if kind == "invalid" and answer is not None and not fb_reason:
                          
                                                       
                                                         
                                                       
                                                          
        reask_note = "第一次回答没读懂，已重问一次；"
        prev = _message_text(answer).strip()
        prev_calls = getattr(answer, "tool_calls", None) or []
        if prev_calls:                                    
            names = "、".join(str(c.get("name") if isinstance(c, dict)
                                  else getattr(c, "name", "")) for c in prev_calls)
            prev = (prev + f"\n（你上一条消息里调用了：{names}）").strip()
        answer2, _note3, fb_reason3, _je3 = _invoke_tool_channel(
            model, tools=decide_specs,
            messages=[HumanMessage(content=tools_prompt),
                      AIMessage(content="你上一轮的回答没能读懂，原文：\n" + (prev or "（空）")),
                      HumanMessage(content=("你刚才的回答没能读懂。只需要再做一步就调用对应工具；"
                                            "做完了就调用 finish 并填好核销报告。"))],
            choice="auto", json_prompt=json_prompt, refallback_on_empty=False,
            usage_sink=usage_local, usage_node="decide")
        if fb_reason3:
            _audit_fallback(_agent_project_root(), "decide", fb_reason3,
                            state["utterance"], model_name)
        kind, payload = _classify(answer2)
        reasked = True
                                                            
                                                       
                                                 
                                                        
                                                                     
                                                        
                                       
    adjudicated: tuple[dict | None, str, str, str] | None = None
    if kind in ("loop", "json"):
        adjudicated = _adjudicate_decide_obj(payload, state)
        _nxt0, _note0, _dec0, violation_fb = adjudicated
        if _nxt0 is None and violation_fb and not reasked and not fb_reason:
            if kind == "loop":                                            
                prev = (f"调用工具 {str(payload.get('verb') or '')}，参数："
                        + json.dumps({k: v for k, v in payload.items() if k != "verb"},
                                     ensure_ascii=False))
            else:
                prev = json.dumps(payload, ensure_ascii=False)
            answer2, _note4, fb_reason4, _je4 = _invoke_tool_channel(
                model, tools=decide_specs,
                messages=[HumanMessage(content=tools_prompt),
                          AIMessage(content="你上一轮提议的下一步没通过检查。你的提议：\n" + prev
                                            + "\n检查意见：\n" + violation_fb),
                          HumanMessage(content=("请按检查意见修正后重新提议下一步；"
                                                "做完了就调用 finish 并填好核销报告。"))],
                choice="auto", json_prompt=json_prompt, refallback_on_empty=False,
                usage_sink=usage_local, usage_node="decide")
            if fb_reason4:
                _audit_fallback(_agent_project_root(), "decide", fb_reason4,
                                state["utterance"], model_name)
            kind, payload = _classify(answer2)
            reask_note = f"第一次提议没通过检查（{violation_fb}），已带检查意见重问一次；"
            reasked = True
            adjudicated = (_adjudicate_decide_obj(payload, state)
                           if kind in ("loop", "json") else None)
                                               
                                                  
                                                    
                                                 
                                            
    if kind in ("loop", "json") and not reasked and not fb_reason:
        _nxt_r, _note_r, _dec_r, _vfb_r = adjudicated if adjudicated is not None else (None, "", "", "")
        if _nxt_r is None and str(payload.get("verb") or "") in LOOP_TOOLS:
            verb_spec0 = _ap.VERB_BY_NAME.get(str(payload.get("verb") or ""))
            shown0 = verb_spec0.zh if verb_spec0 else str(payload.get("verb") or "")
            prev = (f"调用工具 {payload.get('verb')}，参数："
                    + json.dumps({k: v for k, v in payload.items() if k != "verb"},
                                 ensure_ascii=False)) if kind == "loop" \
                else json.dumps(payload, ensure_ascii=False)
            answer3, _note5, fb_reason5, _je5 = _invoke_tool_channel(
                model, tools=decide_specs,
                messages=[HumanMessage(content=tools_prompt),
                          AIMessage(content="你上一轮提议的下一步被系统拒绝了。你的提议：\n" + prev
                                            + "\n拒绝原因：\n" + _note_r),
                          HumanMessage(content=(
                              f"「{shown0}」这一步这次不会再执行，不许再提同一个动作。"
                              "用户原话里还有没做完的**其他**事（例如只读的检查/库容汇报）就提议那件；"
                              "确实没有可做的了，就调用 finish 并在核销报告里如实写明哪些事没做、为什么。"))],
                choice="auto", json_prompt=json_prompt, refallback_on_empty=False,
                usage_sink=usage_local, usage_node="decide")
            if fb_reason5:
                _audit_fallback(_agent_project_root(), "decide", fb_reason5,
                                state["utterance"], model_name)
            kind, payload = _classify(answer3)
            reask_note = f"提议「{shown0}」被系统拒绝（{_note_r}），已回灌重问一次；"
            reasked = True
            adjudicated = (_adjudicate_decide_obj(payload, state)
                           if kind in ("loop", "json") else None)
    vetoes = int(state.get("finish_vetoes") or 0)
    veto_note = ""                                          
    veto_accept = ""                                     
    reask_list = list(state.get("reask_writes") or [])                 
    if kind == "finish":
        while kind == "finish":
            report = str((payload or {}).get("completion_report") or "")
                                                       
                                                         
            veto_list = _finish_veto_all(report, len(steps), reask_list,
                                         list(state.get("checklist") or []), steps,
                                         str(state.get("declined_zh") or ""), state["utterance"],
                                         capabilities=list(
                                             state.get("required_capabilities") or []),
                                         intent_checklist=list(
                                             state.get("intent_checklist") or []))
            line, shape = veto_list[0] if veto_list else (None, "")
            if not veto_list:
                break
            if vetoes >= 2:
                                                           
                veto_accept = ("核销报告仍未单独交代重问后放行的写步结果，按大模型最终判断收尾"
                               f"（已回灌重问 {vetoes} 次）。"
                               if shape == "reask_write_unaccounted" else
                               "核销报告仍标注有未完成事项，按大模型最终判断收尾"
                               f"（已回灌重问 {vetoes} 次）。")
                break
                                                  
                                                  
                                                                
                                                                      
                                               
                                                             
                                                                    
                                            
            vetoes += 1
            if len(veto_list) > 1 or shape in ("checklist_unsettled", "capability_unsettled",
                                               "intent_unsettled",
                                               "pending_source_untouched",
                                               "pending_count_query", "pending_new_not_imported"):
                gap = ("你的核销报告里写着还有没交代的事：\n"
                       + "\n".join(f"- {text}{_VETO_TEACHING_SUFFIX.get(code, '')}"
                                   for text, code in veto_list) + "\n")
            elif shape == "done_without_step":
                gap = (f"你的核销报告把「{line}」标成「已做」却没有合法的步骤号——"
                       "「已做」必须写明是第几步的结果（步骤号不许超过已完成的步数）。")
            elif shape == "dependency_excuse":
                gap = (f"你的核销报告里「{line}」拿前件失败当借口——"
                       "彼此独立的事不许拿前件失败当理由。")
            elif shape == "exempt_without_step":
                gap = (f"你的核销报告里「{line}」的豁免没有举证——"
                       "「条件不成立/做不到」必须写明是据第几步的结果得出的。")
            elif shape == "reask_write_unaccounted":
                gap = (f"你的核销报告没有单独交代{line}的结果——重问后放行的写操作，"
                       "报告里必须写明它的步骤号并单独交代结果。")
            else:
                gap = f"你的核销报告里写着还有没做的事：「{line}」。"
            forced_note = ""
            if vetoes == 2:
                _task_by_id = {str(t.get("task_id") or ""): t
                               for t in (state.get("checklist") or []) if isinstance(t, dict)}
                _cap_forced = None                                  
                _forced = next(
                    (_task_by_id[u["task_id"]] for u in _checklist_unsettled(
                        list(state.get("checklist") or []), steps,
                        str(state.get("declined_zh") or ""))
                     if str(_task_by_id.get(u["task_id"], {}).get("expect_verb") or "")
                     in LOOP_TOOLS),
                    None)
                if _forced is not None:
                    _fspec = _ap.VERB_BY_NAME.get(str(_forced.get("expect_verb")))
                    forced_note = (
                        f"下一步你必须提议「{(_fspec.zh if _fspec else _forced.get('expect_verb'))}」"
                        f"（事项：{_forced.get('text')}）——这是硬性要求，不是建议；"
                        "除此之外的收尾一律不收。")
                else:
                                                 
                                                           
                    _cap_forced = next(
                        (c for c in _capabilities_unsettled(
                            list(state.get("required_capabilities") or []), steps,
                            str(state.get("declined_zh") or ""))
                         if (c.get("verbs") or [""])[0] in LOOP_TOOLS), None)
                    if _cap_forced is not None:
                        _fverb = str((_cap_forced.get("verbs") or [""])[0])
                        _fspec = _ap.VERB_BY_NAME.get(_fverb)
                        forced_note = (
                            f"下一步你必须提议「{(_fspec.zh if _fspec else _fverb)}」"
                            f"（混合诉求的「{_cap_forced.get('label_zh')}」这一半还没做）"
                            "——这是硬性要求，不是建议；除此之外的收尾一律不收。")
                if _forced is None and _cap_forced is None:
                                                          
                                                                                           
                                                  
                    _pend = next((code for _text, code in veto_list
                                  if code in _PENDING_VETO_FORCED_VERB), "")
                    if _pend:
                        _fverb, _ftext = _PENDING_VETO_FORCED_VERB[_pend]
                        _fspec = _ap.VERB_BY_NAME.get(_fverb)
                        forced_note = (
                            f"下一步你必须提议「{(_fspec.zh if _fspec else _fverb)}」"
                            f"（{_ftext}）——这是硬性要求，不是建议；"
                            "除此之外的收尾一律不收。")
            veto_note += gap + forced_note + "已拒收收尾并把缺口回灌重问一次；"
            feedback = (gap + forced_note +
                        "只有条件不成立或做不到的事才允许收尾；否则请继续提议下一步。")
                                                                       
                                                                  
            answer, _note2, fb_reason2, _je2 = _invoke_tool_channel(
                model, tools=decide_specs,
                messages=[HumanMessage(content=tools_prompt),
                          AIMessage(content="你上一轮调用 finish 收尾，核销报告原文：\n"
                                            + (report or "（空）")),
                          HumanMessage(content=feedback)],
                choice="auto", json_prompt=json_prompt, refallback_on_empty=False,
                usage_sink=usage_local, usage_node="decide")
            if fb_reason2:
                _audit_fallback(_agent_project_root(), "decide", fb_reason2,
                                state["utterance"], model_name)
            kind, payload = _classify(answer)
                                                   
                                                
                                               
            reasked = False
    if kind == "finish":
        nxt, note, declined = None, ("大模型判断：要求的事已经完成"
                                     "（或条件不成立，没有要做的下一步）。" + veto_accept), ""
    elif kind == "unsupported":
                                                       
        nxt, note, declined, _vfb = _adjudicate_decide_obj({"verb": payload}, state)
    elif kind in ("loop", "json"):
                                                      
                                                       
                                                                       
                                                          
                                                                 
                                    
        nxt, note, declined, _vfb = (adjudicated if adjudicated is not None
                                     else _adjudicate_decide_obj(payload, state))
        if reasked and nxt is not None:
            reask_verb = str(nxt.get("verb") or "")
            reask_spec = LOOP_TOOLS.get(reask_verb)
            if reask_spec is not None and not reask_spec.get("readonly"):
                pending_write = True
                write_note = "重问后放行的写动作已记入强制核销账，收尾报告须单独交代本步；"
    else:                                                     
        nxt, note, declined = None, "大模型没给出能读懂的答复，按「已完成」收尾。", ""
                                                
                                                               
                                           
    batch_extras: list[dict] = []
    if kind == "loop" and nxt is not None and batch_holder:
        batch_extras, _batch_dropped = _batch_readonly_extras(batch_holder[0], nxt, state)
    multi_note = multi_holder[0] if multi_holder else ""
    if batch_extras:
        n_calls = len(batch_holder[0])
        rest = n_calls - 1 - len(batch_extras)
        if any(_raw_has_placeholder(e) for e in batch_extras):
                                              
                                                   
            multi_note = (f"模型一次给了 {n_calls} 个调用，其中 {len(batch_extras)} 个通过"
                          "占位依赖检查，同批按顺序执行（依赖引用执行时解析）"
                          + (f"，其余 {rest} 个回炉再判" if rest > 0 else "") + "；")
        else:
            multi_note = (f"模型一次给了 {n_calls} 个调用，其中 {len(batch_extras)} 个只读且"
                          "互相独立，同批采纳执行"
                          + (f"，其余 {rest} 个回炉再判" if rest > 0 else "") + "；")
                                     
                                                       
                                                                
    if ncalls_holder:
        _n_calls = ncalls_holder[0]
        _te.emit_batch_emission(n_calls=_n_calls, adopted=len(batch_extras),
                                dropped=_n_calls - 1 - len(batch_extras), note=multi_note,
                                n_placeholder=sum(
                                    1 for e in batch_extras if _raw_has_placeholder(e)))
    note_full = lane_note + multi_note + reask_note + veto_note + write_note + note
    veto_state = ({"finish_vetoes": vetoes}
                  if vetoes != int(state.get("finish_vetoes") or 0) else {})
    if nxt is None:
        out: dict[str, Any] = {
            "loop_next": False,
            "pending_reask_write": False,
            "usage_ledger": usage_local,
            "trace": _trace_entry("decide", "判断下一步", note_full, True, started),
            **veto_state,
        }
        if declined:
                                                         
                                                 
            out["declined_zh"] = declined
        return out
    verb = str(nxt.get("verb") or "")
    return {
        "loop_next": True, "raw": nxt, "loop_plan": None,
        "raw_batch": batch_extras,
        "pending_reask_write": pending_write,
        "usage_ledger": usage_local,
        **veto_state,
        "trace": _trace_entry("decide", "判断下一步",
                              f"{lane_note}{multi_note}{reask_note}{veto_note}{write_note}{note}"
                              f"还需要一步：{_ap.VERB_BY_NAME[verb].zh}。",
                              True, started),
    }


def narrate(state: _AgentState, *, runtime: Any) -> dict:
    """生成说明：**确定性拼接优先**（规则同 act_summary_llm：只用事实，不新增断言）。
    uncertainty_zh/回执素材已由 build_plan_from_raw 的机械口径产出（它产出时 source
    还是 llm，「大模型从你这句话里读出来的」的归因逐字正确）；这里打上 agent 来源标记。
    steps 非空（图内真跑过工具）时写整段汇报 plan.report_zh：**单步 db_status 走既有
    observation 汇报路径逐位保留**；其余（多步 / 其它工具 / 含失败步）由 LLM 据 steps
    紧凑投影组织（LLM 缺席/失败回退确定性拼接，同一批事实两条措辞路径）。"""
    started = time.monotonic()
                                      
                                                              
    _on_progress = getattr(runtime.context, "on_progress", None)
    if _on_progress is not None:
        _on_progress("tool_start", {"verb": "node", "label_zh": "生成说明", "detail": ""})
    chat_model = runtime.context.chat_model
    usage_local: list = []                              
    plan = dict(state["plan"])
    plan["source"] = "agent"
    plan["llm_status"] = "ok"
    detail = str(plan["verb_zh"])
    discarded_report: str | None = None
    discard_reason: str | None = None                                    
    if plan.get("cancelled"):
        detail += "，你说了不做，这一步已取消。"
    if plan.get("uncertainty_zh"):
        detail += "；附「未另外核对」标注。"
    steps = list(state.get("steps") or [])
    spec = LOOP_TOOLS.get(str(plan.get("verb") or ""))
    if _ask_relax_pending(steps) is not None:
                                                        
                                                          
                                                 
        ask_payload = ((_ask_relax_pending(steps).get("result") or {})
                       .get("ask_relax") or {})
        reason = str(ask_payload.get("reason") or "").strip()
        waiting = "选完我会按你的选择自动重跑检索并继续。"
        plan["report_zh"] = (reason.rstrip("。") + "。" if reason else "") + waiting
        plan["report_source"] = "deterministic"
        detail += "；环已暂停等作答，说明按确定事实如实写出。"
    elif (len(steps) <= 1 and spec and spec.get("report") and plan.get("observation")
            and not plan.get("cancelled")):
                                                        
                                                     
        report = _report_with_llm(chat_model, plan["observation"], usage_sink=usage_local)
        if report:
            plan["report_zh"] = report
            plan["report_source"] = "llm"
            detail += "；汇报由大模型据真实数据组织。"
        else:
            plan["report_zh"] = _report_fallback_zh(plan["observation"])
            plan["report_source"] = "deterministic"
            detail += "；大模型没接上，汇报按确定事实如实写出。"
    elif steps:
        report = _steps_report_with_llm(chat_model, state["utterance"], steps,
                                        usage_sink=usage_local)
                                                          
        contradicted = False
        discard_reason = _report_contradiction_reason(report, steps) if report else None
        if report and discard_reason:
                                                     
                                          
                                                    
                                 
            discarded_report = report
            report = None
            contradicted = True
            detail += "；大模型的汇报措辞可能越界（疑似与实录不符），按确定事实如实写出。"
        if report:
            plan["report_zh"] = report
            plan["report_source"] = "llm"
            detail += "；全程汇报由大模型据真实步骤结果组织。"
        else:
            plan["report_zh"] = _steps_report_fallback_zh(steps)
            plan["report_source"] = "deterministic"
            declined = str(state.get("declined_zh") or "").strip()
            if declined:
                                                    
                                                 
                plan["report_zh"] = plan["report_zh"].rstrip("。") + "；" + declined
            if not contradicted:
                detail += "；大模型没接上，全程汇报按确定事实如实写出。"
    elif plan.get("cancelled"):
                                                         
                           
        plan["report_zh"] = f"{str(plan['verb_zh']).rstrip('。')}：已按你的要求取消，没有执行任何操作。"
        plan["report_source"] = "deterministic"
    if state.get("truncated") and plan.get("report_zh"):
        if state.get("truncated_settled"):
                                                   
                                                
            plan["report_zh"] = str(plan["report_zh"]).rstrip("。") + (
                f"。（一句话能连续做的事较多，一次最多 {MAX_STEPS} 步的预算刚好用完，"
                "你交代的事已全部完成。）")
            detail += "；到顶结算：原话交代的事已全部执行，汇报已如实标注预算刚好用完。"
        else:
                                                                 
                                                      
            plan["report_zh"] = str(plan["report_zh"]).rstrip("。") + (
                f"；要连续做的事超过一次最多能做的 {MAX_STEPS} 步，剩下的没有执行。")
            detail += "；汇报已标注：还有事被步数上限截断，没做完。"
                                              
                                                
                                                        
    if plan.get("report_zh"):
        named_all = _named_sources_in(str(state.get("utterance") or ""))
        if named_all and any(
                s.get("ok") and str(s.get("verb") or "") == "curate.sync_updates"
                and not str((s.get("slots") or {}).get("source") or "").strip()
                for s in steps):
            plan["report_zh"] = str(plan["report_zh"]).rstrip("。") + (
                f"；你点名的是{'、'.join(named_all)}，同步步未指定来源，"
                "实际按全部在线源检查并同步。")
            detail += "；点名源下 sync 空槽按全部在线源同步，汇报已如实标注。"
                                                  
                                          
                                                         
                                   
    checklist = list(state.get("checklist") or [])
    if checklist and plan.get("report_zh"):
        unsettled = _checklist_unsettled(checklist, steps, str(state.get("declined_zh") or ""))
        if unsettled:
            tails = "、".join(f"「{str(t.get('text') or '')[:30]}」" for t in unsettled[:3])
                                                             
                                       
            plan["report_zh"] = str(plan["report_zh"]).rstrip("。") + (
                f"；按开头拆出的事项清单核对，这些事还没做：{tails}"
                + ("等。" if len(unsettled) > 3 else "。"))
            detail += f"；清单核对：还有 {len(unsettled)} 件没做，已如实标注。"
    entry = _trace_entry("narrate", "生成说明", detail, True, started)[0]
    if discarded_report is not None:
                                                   
                                
        entry["discarded_report_zh"] = discarded_report
                                                                            
                                                                 
        entry["discard_reason"] = discard_reason
    return {
                                                                           
                                                     
                                                                 
        "violations": [],
        "plan": plan,
        "usage_ledger": usage_local,
        "trace": [entry],
    }


def _route_after_validate(state: _AgentState) -> str:
    if not state.get("violations"):
        return "execute"
    if state.get("steps"):
                                                  
                                                                       
                                                     
                                                        
                                                  
        return "narrate"
    if int(state.get("repairs") or 0) < 1:
        return "repair"
    return "fail"


def _route_after_execute(state: _AgentState) -> str:
                                                                  
                                                       
    if _ask_relax_pending(list(state.get("steps") or [])) is not None:
        return "narrate"
                                                           
    return "decide" if state.get("last_ran") else "narrate"


def _route_after_decide(state: _AgentState) -> str:
    return "validate" if state.get("loop_next") else "narrate"


                                         
_GRAPH_LOCK = threading.Lock()
_COMPILED_GRAPH: Any = None
_GRAPH_BUILDS = 0                        


def _get_graph() -> Any:
    """模块级编译单例：首次调用构建 + compile 一次，之后复用；
    双检锁护首轮并发。compile 产物跨请求共享是 langgraph 的设计意图——每次
    stream/invoke 新建 loop/channels/Runtime（源码实证见换装调研底稿），每请求依赖
    走 context 注入、不进闭包。"""
    global _COMPILED_GRAPH, _GRAPH_BUILDS
    if _COMPILED_GRAPH is None:
        with _GRAPH_LOCK:
            if _COMPILED_GRAPH is None:
                from langgraph.graph import END, START, StateGraph

                graph = StateGraph(_AgentState, context_schema=_AgentContext)
                graph.add_node("understand", _run_control.cancellable_node(understand))
                graph.add_node("validate", _run_control.cancellable_node(validate))
                graph.add_node("repair", _run_control.cancellable_node(repair))
                graph.add_node("execute", _run_control.cancellable_node(execute))
                graph.add_node("decide", _run_control.cancellable_node(decide))
                graph.add_node("narrate", _run_control.cancellable_node(narrate))
                                             
                                                
                graph.add_node("route_consensus", _run_control.cancellable_node(route_consensus))
                graph.add_edge(START, "route_consensus")
                graph.add_edge("route_consensus", "understand")
                graph.add_edge("understand", "validate")
                graph.add_conditional_edges(
                    "validate", _route_after_validate,
                    {"execute": "execute", "repair": "repair", "narrate": "narrate", "fail": END},
                )
                graph.add_edge("repair", "validate")
                graph.add_conditional_edges(
                    "execute", _route_after_execute,
                    {"decide": "decide", "narrate": "narrate"},
                )
                graph.add_conditional_edges(
                    "decide", _route_after_decide,
                    {"validate": "validate", "narrate": "narrate"},
                )
                graph.add_edge("narrate", END)
                _COMPILED_GRAPH = graph.compile()
                _GRAPH_BUILDS += 1
    return _COMPILED_GRAPH


def plan_with_agent_events(
    utterance: str,
    *,
    has_results: bool,
    result_total: int,
    config: LLMConfig,
    retrieval: dict | None,
    current_query: str,
    current_filters: Any,
    chat_model: Any | None = None,
    decide_model: Any | None = None,
    on_event: Callable[[str, dict], None] | None = None,
    principal: str = "",
    route_scope: str = "",
    search_sources: Any = None,
    search_facet_filters: Any = None,
    search_suppressed_constraints: Any = None,
    search_lenient_dims: Any = None,
    search_date_from: str = "",
    search_date_to: str = "",
                                          
    search_top_k: Any = None,
    search_recall: str = "",
    search_strategy: str = "",
                                             
                                    
    retrieval_provider: Any = None,
    on_route_verdict: Any = None,
    route_extra_zh: str = "",
                        
                                                                   
    artifact_context: str = "",
                                                    
                                                                     
                                                                  
    intent_checklist: list[dict] | None = None,
                                                 
                                                                                
                                                             
                              
    ask_answer: dict | None = None,
                                                  
                                            
    allowed_verbs: "frozenset[str] | None" = None,
    conversation_history: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """一句话 → (plan, trace)。plan 形状与 `action_plan.plan_action` 输出逐位同形，
    仅 `source="agent"`、`llm_status="ok"`，并附 `trace`（节点步骤，供前端行动流渲染）。

    与 `plan_with_agent` 的唯一区别是 `on_event`（流式）：给了回调就在逐节点推进时
    （`stream_mode="values"`——每帧是 reducer 应用后的全量 state，末帧即终态）
    按 trace 长度 diff 回调 `on_event("step", trace_entry)`（顺序 = 节点执行序）；
    没给回调也走同一条流式收集路径——**不跑两遍图**，两个入口共享同一张编译单例、
    同一份终态口径。

    失败抛 `AgentUnavailable` / `AgentPlanInvalid`——调用方（turn.route_turn）捕获后回退
    `action_plan.plan_action` 保底。`chat_model` 可注入（测试替身，需有 `bind_tools`/`invoke`），
    注入时跳过 `should_use_llm` 闸（与 action_plan 的 `llm_call` 注入同纪律：测试隔离）。
    `decide_model`（2026-08-07 复杂度路由）是 decide/repair 档的**显式注入缝**：
    给了就用它（不再按 env 自建），不给且车道为 complex 且配置了 LLM_MODEL_COMPLEX 才自建。
    `principal`：成功经验库分区主体（会话账户 id；空 → anonymous）。
    与 config 派生的 endpoint_fp 一起，既打进 understand 注入过滤，也打进成功收尾的账本行标。
    `route_scope="rescue"` = 检索救回回合——作为正式第四套件，understand/decide
    工具面收敛到 search.rerun（+none/finish），validate/裁决层共用套件机械闸；
    `search_sources` 与五个 `search_*` 结构化条件是 search.rerun 的完整当前屏范围；
    rescue 与常规 /api/utterance 都从同一份检索参数透传，缺省全空保持旧调用兼容。
三缝：`retrieval_provider` 在场时图 initial state 的 retrieval 取 None（并发
    pre-loop 的摘要由 understand 入口经 provider 局部汇合）；`on_route_verdict` 在
    route_consensus 算完 route 后回调（只标记不发射）；`route_extra_zh` 拼进共识上下文
    尾部。三者缺省 = 旧行为逐位不变。
    """
    if not agent_available():
                                                  
        _te.emit_finish_reason(kind="unavailable")
        raise AgentUnavailable("langchain 扩展未安装，或已被 BIODATA_AGENT_EXEC=off 强制关停。")
    text = _ap.normalize_utterance(utterance)
    total = max(0, int(result_total or 0))

    injected = chat_model is not None
    if chat_model is None:
        ok, reason = _ap.should_use_llm(config)
        if not ok:
                                
            _te.emit_finish_reason(kind="unavailable")
            raise AgentUnavailable(f"大模型不可用（{reason}）——agent 路径与单次分类同闸。")
        chat_model = _build_chat_model(config)

            
                                                                               
                                                              
                                                                
                                                         
                                                       
                                                                         
                                                                       
    lane = decide_lane(text)
    decide_model_name = ""
    if decide_model is None and not injected and lane == "complex":
        thinking, effort = _complex_thinking_env()
        decide_model_name = _complex_model_name()
                                                       
                                                            
                                       
        if decide_model_name or thinking:
            decide_model = _build_chat_model(
                _dataclass_replace(config,
                                   model=decide_model_name or config.model,
                                   thinking=thinking, reasoning_effort=effort))
            if not decide_model_name:
                decide_model_name = f"{config.model}+thinking"
    elif decide_model is not None:
        decide_model_name = "injected"

    graph = _get_graph()
                                                                         
    principal_norm = (principal or "").strip() or "anonymous"
    endpoint_fp = _endpoint_fp_from_config(config)
    context = _AgentContext(
        chat_model=chat_model, model_name=str(getattr(config, "model", "") or ""),
        decide_model=decide_model, decide_model_name=decide_model_name, decide_lane=lane,
        principal=principal_norm, endpoint_fp=endpoint_fp, project_id=_learning_scope.current_project_id(),
        cancel_check=_run_control.current_check(),
                                                       
        on_progress=on_event,
            
        retrieval_provider=retrieval_provider,
        on_route_verdict=on_route_verdict,
        route_extra_zh=route_extra_zh,
                                        
        artifact_context=str(artifact_context or ""),
        history_context=_conversation_history_block(conversation_history))
    initial: _AgentState = {
        "utterance": text,
        "artifact_context": str(artifact_context or ""),
                                                                       
                                                         
        "retrieval": (None if retrieval_provider is not None else retrieval),
        "current_query": str(current_query or ""),
        "current_filters": current_filters,
        "has_results": bool(has_results),
        "result_total": total,
        "route_scope": (str(route_scope or "") if str(route_scope or "") in _SCOPED_ROUTES else ""),
        "search_sources": search_sources,
        "search_facet_filters": search_facet_filters,
        "search_suppressed_constraints": search_suppressed_constraints,
        "search_lenient_dims": search_lenient_dims,
        "search_date_from": str(search_date_from or ""),
        "search_date_to": str(search_date_to or ""),
        "search_top_k": search_top_k,
        "search_recall": str(search_recall or ""),
        "search_strategy": str(search_strategy or ""),
        "raw": {},
        "violations": [],
        "repairs": 0,
        "trace": [],
        "observations": [],                                   
        "steps": [],                                                  
        "loop_next": False,
        "last_ran": False,
                                                        
        "intent_checklist": [dict(it) for it in (intent_checklist or []) if isinstance(it, dict)],
                                                            
        "ask_answer": dict(ask_answer) if isinstance(ask_answer, dict) else None,
                                          
        "allowed_verbs": (frozenset(allowed_verbs) if allowed_verbs else None),
    }

                                                     
                                                       
                                                        
    final: dict[str, Any] = {}
    emitted = 0
    for frame in graph.stream(initial, config={"recursion_limit": 50}, context=context,
                              stream_mode="values"):
        final = frame
        trace_so_far = list(frame.get("trace") or [])
        if on_event is not None and len(trace_so_far) > emitted:
            for entry in trace_so_far[emitted:]:
                on_event("step", entry)
        emitted = max(emitted, len(trace_so_far))

    if final.get("violations"):
                                       
        _te.emit_finish_reason(kind="plan_invalid",
                               repairs=int(final.get("repairs") or 0))
        raise AgentPlanInvalid(list(final["violations"]))
    plan = dict(final.get("plan") or {})
    if not plan:
                                                            
                                              
        _te.emit_finish_reason(kind="unavailable")        
        raise AgentUnavailable("agent 图没能产出 plan（编排异常）。")
    trace = list(final.get("trace") or [])
    plan["trace"] = trace
                                                        
                                                                          
                                                                      
                                                   
                                                           
                                                
                                          
    _maybe_record_success(_agent_project_root(), text, plan, list(final.get("steps") or []),
                          principal=principal_norm, endpoint_fp=endpoint_fp,
                          mode=str(final.get("mode") or "tools"),
                          repairs=int(final.get("repairs") or 0),
                          finish_vetoes=int(final.get("finish_vetoes") or 0),
                          reask_write_count=len(final.get("reask_writes") or []),
                          truncated=bool(final.get("truncated")),
                          checklist_dropped=int(final.get("checklist_dropped") or 0))
                                                        
                                                            
    usage_ledger = list(final.get("usage_ledger") or [])
    if usage_ledger:
        input_total = sum(int(r.get("input") or 0) for r in usage_ledger)
        cache_total = sum(int(r.get("cache_read") or 0) for r in usage_ledger)
        plan["llm_usage"] = {
            "calls": usage_ledger,
            "input_total": input_total,
            "cache_read_total": cache_total,
            "cache_hit_rate": round(cache_total / input_total, 4) if input_total else 0.0,
        }
                                                            
                                                              
    _truncated = bool(final.get("truncated"))
    _te.emit_finish_reason(
        kind=(("truncated_settled" if final.get("truncated_settled") else "truncated")
              if _truncated else "completed"),
        steps=len(list(final.get("steps") or [])),
        repairs=int(final.get("repairs") or 0),
        finish_vetoes=int(final.get("finish_vetoes") or 0),
        reask_write_count=len(final.get("reask_writes") or []),
        declined=str(final.get("declined_zh") or ""),
        truncated=_truncated,
        truncated_settled=bool(final.get("truncated_settled")))
    return plan, trace


def plan_with_agent(
    utterance: str,
    *,
    has_results: bool,
    result_total: int,
    config: LLMConfig,
    retrieval: dict | None,
    current_query: str,
    current_filters: Any,
    chat_model: Any | None = None,
    decide_model: Any | None = None,
    principal: str = "",
    route_scope: str = "",
    search_sources: Any = None,
    search_facet_filters: Any = None,
    search_suppressed_constraints: Any = None,
    search_lenient_dims: Any = None,
    search_date_from: str = "",
    search_date_to: str = "",
    search_top_k: Any = None,
    search_recall: str = "",
    search_strategy: str = "",
                                                     
    retrieval_provider: Any = None,
    on_route_verdict: Any = None,
    route_extra_zh: str = "",
                                         
    artifact_context: str = "",
                                                                
    intent_checklist: list[dict] | None = None,
                                                                 
    ask_answer: dict | None = None,
                                          
    allowed_verbs: "frozenset[str] | None" = None,
    conversation_history: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """`plan_with_agent_events` 的无回调薄封装——
    返回值/异常契约与旧版逐位不变，调用方（turn.route_turn 非流式路径）无需任何改动。"""
    return plan_with_agent_events(
        utterance,
        has_results=has_results, result_total=result_total,
        config=config, retrieval=retrieval,
        current_query=current_query, current_filters=current_filters,
        chat_model=chat_model, decide_model=decide_model, on_event=None,
        principal=principal,
        route_scope=route_scope,
        search_sources=search_sources,
        search_facet_filters=search_facet_filters,
        search_suppressed_constraints=search_suppressed_constraints,
        search_lenient_dims=search_lenient_dims,
        search_date_from=search_date_from, search_date_to=search_date_to,
        search_top_k=search_top_k, search_recall=search_recall,
        search_strategy=search_strategy,
        retrieval_provider=retrieval_provider,
        on_route_verdict=on_route_verdict,
        route_extra_zh=route_extra_zh,
        artifact_context=artifact_context,
        intent_checklist=intent_checklist,
        ask_answer=ask_answer,
        allowed_verbs=allowed_verbs,
        conversation_history=conversation_history,
    )
