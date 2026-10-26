from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import html as _html
import io
import ipaddress
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback
import urllib.parse
import zipfile
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..corpus.corpus import available_sources, corpus_cache_generation, corpus_snapshot, invalidate_external_cache, known_source_values, load_full_corpus, load_normalized_corpus, locate_record, publish_corpus_write, _base_source as _corpus_base_source
from ..corpus.downloads import files_for, primary_url
from . import accounts
from . import llm_quota
from . import mcp_tokens                                 
from ..llm import act_summary_llm, dream, search_reply_llm
from ..content import item_view
from ..llm.config import get_settings
from ..domain.registry import get_domain
from .accounts import AccountError, SESSION_COOKIE
from ..retrieval.fair import build_fair_report
from ..llm.llm_client import LLMConfig, diagnose_network, healthcheck, load_llm_config, should_use_llm
from ..llm.llm_client import _normalize_provider as _llm_normalize_provider
from ..retrieval.query_parser import parse_query
from ..content.reuse_pack import ReusePackError, build_pack_for_uids, sanitize_uids, to_bibtex, to_ris
from ..content.reuse_pack import to_markdown as pack_to_markdown
from ..retrieval.search_request import resolve_search_request
from ..corpus.uploads import (
    UploadError,
    ingest_dataset,
    new_upload_name as _new_upload_name,
)
from .recommend_rows import rows_from_retrieved
from .workflow import (
    DatasetRecommendationWorkflow,
    RecommendParams,
    intent_projection,
    sanitize_facet_filters,
    sanitize_suppressed,
    sanitize_lenient_dims,
)
from .request_validation import (
    ParamValidationError,
    validate_date_window,
    validate_iso_date,
    validate_query,
    validate_sources,
)


def _validate_or_400(fn, *args, **kwargs):
    """共享校验束（app/request_validation）→ Web 翻译：ParamValidationError → 400 + hint
    + X-Error-Code 头（机器码此前只在 MCP 端保留，Web 端一律被剥掉，前端无法按码
    编程；本头 additive，不改变 detail 字符串文案）。"""
    try:
        return fn(*args, **kwargs)
    except ParamValidationError as exc:
        raise HTTPException(
            status_code=400, detail=exc.hint, headers={"X-Error-Code": exc.code}
        ) from exc


def _status_for(code: str, table: dict[str, int]) -> int:
    """机器码 → HTTP 状态的唯一翻译口（缺省 400）：账户/令牌/dream/下载各错误码表
    都经它查表，不再各写 `.get(code, 400)` 或内联三元。"""
    return table.get(code, 400)


from .runtime_paths import get_app_paths, instance_data_dir_for
from .model_installer import cancel_model_install, model_install_status, start_model_install
from .limits import MAX_DATASETS_LIMIT                                          
from .limits import CORPUS_SYNC_VECTOR_TIMEOUT_SECONDS

                                                       
                                                                                             
                                                                          
                                                                                                  
                                                                                         
                                                                                                  
PATHS = get_app_paths()
PROJECT_ROOT = PATHS.data_root
CONFIG_ROOT = PATHS.config_root
RESOURCE_ROOT = PATHS.resource_root
STATIC_DIR = RESOURCE_ROOT / "web" / "static"
DATA_DIR = PATHS.shipped_base_dir
                                                    
                                                                    
                                                                                    

from .llm_admission import ENV_LOCK

                                                                      
                                                                               
                                                                                   
                                                                   
                                                     
                                                                        
                                                                        
                                                                                                       
                                                                             
                                                                                                          
                                                                                         
                                                       
WEB_API_VERSION = "3.7.0"


class _CspStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        if path.lower().endswith((".html", ".htm")):
            full_path, _ = await run_in_threadpool(self.lookup_path, path)
            if full_path:
                page_html = await run_in_threadpool(Path(full_path).read_text, encoding="utf-8")
                response.headers["Content-Security-Policy"] = _content_security_policy(page_html)
        return response


app = FastAPI(title=f"{get_domain().branding.product_name} Web UI", version=WEB_API_VERSION)
                                                      
                                                       
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", _CspStaticFiles(directory=str(STATIC_DIR)), name="static")

                                                                         
                                                                     
                                                           
                                                         
                                              
                                                            
                                                      
                                                       
from starlette.routing import Route as _Route              

try:
    from . import mcp_server as _mcp_server                                       
except ModuleNotFoundError as exc:                                        
    if exc.name != "mcp":
        raise
    _mcp_server = None

if _mcp_server is not None:
    _ONLINE_MCP_APP = _mcp_server.build_online_mcp_app()
    app.router.routes.append(_Route("/mcp", endpoint=_ONLINE_MCP_APP, methods=["GET", "POST", "DELETE"]))


@asynccontextmanager
async def _lifespan_with_mcp(_app):
    """宿主驱动 MCP 会话管理器及已有会话维护；无 MCP 时仍恢复持久清理。
    无待清理项保持 runtime 惰性；本生命周期创建的 runtime 最后一个宿主退出时关闭。"""
    import anyio
    token = object()
    try:
        await run_in_threadpool(_enter_conversation_lifespan, token)
        if _mcp_server is None:
            yield
        else:
            _mcp_server.reset_online_runtime()
            async with _mcp_server.mcp.session_manager.run():
                yield
    finally:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(_leave_conversation_lifespan, token)


app.router.lifespan_context = _lifespan_with_mcp

                                                
                                                       
                                                                       
                                                            
                                                        
                                                                  
logger = logging.getLogger(__name__)
if not logger.handlers:
    _stderr_handler = logging.StreamHandler()
    _stderr_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_stderr_handler)
    logger.setLevel(logging.INFO)


@app.exception_handler(RequestValidationError)
def _redacted_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """安全：不回显用户提交的原始值（如密码 / API Key）——请求体校验失败时只保留 loc/msg/type，
    剔除 pydantic 默认带回的 `input` / `ctx`（对抗评审 #3；覆盖全部端点）。"""
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors}, media_type="application/json; charset=utf-8")


                                                    
                                                       
                                            
_MAX_SOURCES_ITEMS = 50                                         
_MAX_FACET_FILTERS_ITEMS = 64                                                
_MAX_SUPPRESSED_ITEMS = 32                                    
_MAX_LENIENT_ITEMS = 16                             
_MAX_CURRENT_FILTERS_ITEMS = 100                        
_MAX_UIDS_ITEMS = 500                                          
_MAX_API_KEY_CHARS = 512                           
_MAX_MODEL_CHARS = 200                       

                                                              
_DESC_PROVIDER = "mock / zhipuai / openai-compatible / trial（T3 限量试用）"
_DESC_REQUEST_API_KEY = "本次请求临时 key，不持久化"
_DESC_CUSTOM_BASE_URL = "自定义 API 接口地址"
_DESC_CUSTOM_MODEL = "自定义模型名"


class _ExperimentContract(BaseModel):
    """实验归因三件套必须同在；缺一就拒绝，绝不把观察流量误标为随机实验。"""
    experiment_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    experiment_arm: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    propensity: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def _complete_experiment(self):
        present = (self.experiment_id is not None, self.experiment_arm is not None, self.propensity is not None)
        if any(present) and not all(present):
            raise ValueError("experiment_id/experiment_arm/propensity must be provided together")
        return self


class RecommendRequest(_ExperimentContract):
    query: str = Field(..., min_length=1, description="用户查询")
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False, description="大模型总开关：门控润色与一切请求级 LLM 能力")
    mock_llm: bool = Field(default=False)
    polish: bool = Field(default=True, description="AI 润色推荐说明（只改说明文字，不动结果与排序）。总开关 use_llm 之下的独立子开关")
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    top_k: int | None = Field(default=None, ge=1, le=50, description="返回结果最大数量（默认 10，最大 50）")
    rerank: str = Field(default="off", description="可选 LLM 重排：off / llm")
    rerank_top_n: int | None = Field(default=None, ge=1, le=50, description="启用重排时喂给 LLM 的候选池大小")
    recall: str = Field(default="off", description="可选向量召回：off / dense（稠密嵌入） / cross_encoder（本地重排器，推荐）")
    strategy: str = Field(default="fixed", description="检索策略：fixed=使用显式 recall/rerank；auto=综合候选数量、语义信息量和可用后端，选择纯规则、本地语义、LLM 或两层组合。auto 会覆盖 recall/rerank 的显式取值")
    auto_allow_llm: bool = Field(default=False, description="strategy=auto 时是否允许在复杂查询中自动使用已配置 LLM；默认 false 避免意外联网")
    date_from: str | None = Field(default=None, description="发表时间范围起（ISO YYYY-MM-DD，含）；空=不限；格式或日历不合法 → 400（不静默忽略）")
    date_to: str | None = Field(default=None, description="发表时间范围止（ISO YYYY-MM-DD，含）；空=不限；格式或日历不合法 → 400（不静默忽略）")
    sources: list[str] | None = Field(default=None, max_length=_MAX_SOURCES_ITEMS, description="在哪些数据来源中检索（如 ['10x Genomics','CELLxGENE Discover']）；None=仅基础语料（官方评测不受影响）")
    auto_parse_sources: bool = Field(default=False, description="是否从 query 自动识别数据来源专名；前端自动来源模式开启，手动模式关闭")
    base_url: str | None = Field(default=None, description="自定义 API 接口地址（base_url）；留空则用服务器/预设默认")
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description="自定义模型名；留空则用服务器/预设默认")
    facet_filters: list[dict] | None = Field(default=None, max_length=_MAX_FACET_FILTERS_ITEMS, description="分面细化：在当前结果集上按维度精确收窄。value 取 facets[].values[].value 里回传的**分面键**（物种/组织/疾病为小写归一键，如 'homo sapiens'）；None=不细化")
    suppressed_constraints: list[str] | None = Field(default=None, max_length=_MAX_SUPPRESSED_ITEMS, description="忽略查询里已识别出的某个筛选条件（对应网页「查询条件」标签上的「忽略」）：取 query_constraints[].dim（物种/组织/疾病/平台/技术 + has_raw_data + date）。传入则该条件在检索前被放宽；None/空=全部条件照常生效，返回结果与不传本参数完全一致")
    lenient_dims: list[str] | None = Field(default=None, max_length=_MAX_LENIENT_ITEMS, description="把「未标注」的也纳入（对应网页结果区的「也纳入」按钮）：对这些维度（species/tissue/disease/platform/assay/modality）**字段为空的记录视作通过**（无法核验≠不匹配），已知是别的值仍然排除。取 coverage_caveats[].dim；None/空=不放宽，返回结果与不传本参数完全一致")
                                                                            
    conv_id: str | None = Field(default=None, max_length=80, description="前端生成的对话 id：本次检索记入该对话的 session log（复盘数据面）；None=不记录")
    replay_key: str | None = Field(default=None, max_length=64, description="对话的复盘访问 key（匿名分区下防串号的第二道闸）；随 conv_id 生效")


class TaskPackPreviewRequest(RecommendRequest):
    """一句话任务包的预览入参。沿用检索入参，另加条数与范围。"""
    limit: int | None = Field(default=None, description="打进包里的数据集条数（10 / 20 / 50，默认 10）。服务端在全部命中里按当前排序取前 N，与页面上显示几张卡无关")
    scope: str = Field(default="primary", description="primary=每个数据集只取一个代表性主文件（默认）；all=取全部文件")
    query_effective: str | None = Field(default=None, description="本工具实际用于检索的句子（开了关键词审核且被采纳时与原句不同）")
    keep_selected: list[str] | None = Field(default=None, description="重新预览时保留的勾选；新候选里还在的会继续勾上")


class TaskPackBuildRequest(BaseModel):
    """产包入参。**必须原样回传预览时拿到的那一整套**，服务端会用同样的参数重跑一遍再比对。"""
    model_config = ConfigDict(extra="forbid")

    plan_token: str = Field(..., description="预览时返回的 plan_token")
    selected_uids: list[str] = Field(..., max_length=_MAX_UIDS_ITEMS, description="用户勾选的数据集编号；必须是候选池的子集（少选合法）")
    snapshot_id: str = Field(default="", description="预览时的目录快照编号；指纹三件套缺一即 400（空=没走过预览，绝不放行）")
    content_digest: str = Field(default="", description="预览时的内容指纹；同上")
    retrieval_date: str = Field(default="", description="预览时的检索日期（ISO YYYY-MM-DD）；非法即 400——该值会拼进响应头文件名，绝不原样信任")
    scope: str = Field(default="primary")
    retrieval_params: dict = Field(..., description="预览时返回的 retrieval_params，原样回传")
    format: str = Field(default="zip", description="zip=直接下载压缩包；json=逐个文件的文本，由调用方自行落盘")


class BoardPlanRequest(BaseModel):
    """「再说一句话改条件」的规划入参。**无状态**：当前条件由调用方每次原样带上来。

    服务端刻意不存会话——存了就等于事实上的使用行为采集，而「要不要做使用数据采集」
    是产品方明确保留的未决策项。
    """
    query: str = Field(default="", description="产生当前这批结果的那句话（开了关键词审核时是改写后那句），不是输入框里的当前值")
    utterance: str = Field(default="", description="用户这一句原话，例如「换成小鼠」「去掉组织限制」")
    forced_op: str = Field(default="", description="点按钮时直接指定改动类型：replace / widen / restart / suggest / add / remove / lenient；留空则由 utterance 判断")
    dim: str = Field(default="", description="点按钮时直接指定要改哪一项（species / tissue / disease / platform / assay / modality / has_raw_data / date）")
    candidate_override: str = Field(default="", description="用户在预览框里手改后的整句；给了就以它为准")
    current_filters: list[dict] | None = Field(default=None, max_length=_MAX_CURRENT_FILTERS_ITEMS, description="上一次 /api/recommend 返回的 active_filters 原样。这是「现在按什么在筛」的唯一真源")
    resolution: dict | None = Field(default=None, description="上一次 /api/recommend 返回的 interpretation.resolution 原样，用于核对数据来源专名")
    suppressed_constraints: list[str] | None = Field(default=None, max_length=_MAX_SUPPRESSED_ITEMS, description="当前被忽略的筛选项编号")
    lenient_dims: list[str] | None = Field(default=None, max_length=_MAX_LENIENT_ITEMS, description="当前已放宽的项")
    facet_filters: list[dict] | None = Field(default=None, max_length=_MAX_FACET_FILTERS_ITEMS, description="当前「在结果里再缩小」的选择")
    coverage_dims: list[str] | None = Field(default=None, max_length=_MAX_CURRENT_FILTERS_ITEMS, description="上一次返回的 coverage_caveats 里的项，用来判断「放宽某一项」有没有意义")
    date_from: str | None = Field(default=None, description="年份下拉框的起始值")
    date_to: str | None = Field(default=None, description="年份下拉框的结束值")


class ActionPlanRequest(BaseModel):
    """一句话执行层的入参。**只规划，不执行**——见 `/api/action/plan` 的 docstring。"""
    model_config = ConfigDict(extra="forbid")

    utterance: str = Field(..., min_length=1, description="用户这一句原话")
    has_results: bool = Field(default=False, description="调用方屏幕上当前是否已有一批检索结果")
    result_total: int = Field(default=0, ge=0, description="当前这批结果的命中总数（调用方自述）")
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


class UtteranceRequest(_ExperimentContract):
    """统一路由端点 `/api/utterance` 的入参（turn pipeline：规则匹配 → LLM 分流）。

    无状态：LLM 分流所需的现场（有无结果 / 当前查询 / 当前条件 / 来源池）由调用方
    每次原样带上来；LLM 配置覆盖沿用 `/api/recommend` 的请求级契约。
    """
    model_config = ConfigDict(extra="forbid")

    utterance: str = Field(..., min_length=1, description="用户这一句原话")
    has_results: bool = Field(default=False, description="调用方屏幕上当前是否已有一批检索结果")
    result_total: int = Field(default=0, ge=0, description="当前这批结果的命中总数（调用方自述）")
                         
    query: str = Field(default="", description="产生当前这批结果的那句话（refine 改写以它为底）")
    current_filters: list[dict] | None = Field(default=None, max_length=_MAX_CURRENT_FILTERS_ITEMS, description="上一次 /api/recommend 返回的 active_filters 原样")
    sources: list[str] | None = Field(default=None, max_length=_MAX_SOURCES_ITEMS, description="当前选中的来源池（规则匹配概览按它取）")
                                          
    artifact_context: str | None = Field(
        default=None,
        max_length=2000,
        description="用户附加上下文（追踪/数据集快照，前端已截断 ≤2000 Unicode 字符）。"
                    "独立字段、不拼进用户原话：只进入 agent prompt 作结构化上下文块（首行带类型，标注「仅供参考」），"
                    "不进 identifier 快速道/query parser/quoted 证据/检索 query；本地演示/无 AI 模式被安全忽略",
    )
                                             
    suggested_recipe: str | None = Field(
        default=None,
        max_length=64,
        description="结果页阶梯 chip / 任务卡**未经编辑**的模板文本随请求携带的建议动作 id。"
                    "allowlist 校验在 action_plan.SUGGESTED_RECIPES（单一真源）：不在表 → "
                    "忽略回普通路由并如实记录（响应 recipe_note）；合法 → 只缩小到既有能力"
                    "（动词选择面收窄），不得绕过参数校验/执行开关/安全闸。编辑过模板 = 普通路由不携带",
    )
                                                    
    ask_answer: dict | None = Field(
        default=None,
        description="用户在环内放松卡（ask.relax）上的作答：{kind: 恒 \"ask.relax\", "
                    "choice_key, label, message, suppress}。在场时本回合语义恒为「按所选忽略条件"
                    "重跑原查询」：agent 可用 → 续跑 ReAct 环（route_consensus 免投票直定 search、"
                    "understand 零 LLM 强置首步 rank、decide 注入作答事实段）；不可用 → 回退纯检索"
                    "路由（等同旧零 LLM 重搜）。suppress 清单过与 suppressed_constraints 同一净化闸",
    )
                                 
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)
                                                      
    agent: bool = Field(
        default=True,
        description="开：所有消息 100% 过 LLM 分流（langgraph agent 优先、单次分类保底）；"
                    "关：LLM 分流器永不启动——一切输入按规则检索处理（操作句只回降级气泡）",
    )
                   
    stream: bool = Field(
        default=False,
        description="true 时改发 text/event-stream：agent 规划节点 step* → final（final 体与非流式逐位同形）",
    )

    project_id: str | None = Field(default=None, max_length=80, description="对话项目编号；空串为未归类")
    background: bool = Field(default=False, description="以有界持久任务执行，并返回可恢复的任务编号")
    shared: bool = Field(default=False, description="显式共享会话；服务器消息、结果帧与现场为真源")
    expected_revision: int | None = Field(default=None, ge=0)
    state_patch: dict | None = Field(default=None)
    ask_choice: str | None = Field(default=None, max_length=80)
                                   
    req_id: str | None = Field(
        default=None,
        description="调用方生成的请求号：断流重发**同一句**时原样回传，服务端占用去重（重发拿缓存结果，"
                    "不二次执行写工具）；两次独立提交必须各自新号。None=无幂等，行为与不传逐位一致",
    )
                                                   
                                                                        
                                                            
    top_k: int | None = Field(default=None, ge=1, le=50, description="返回结果最大数量（默认 10，最大 50）")
    rerank: str = Field(default="off", description="可选 LLM 重排：off / llm")
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    auto_parse_sources: bool = Field(default=False)
    auto_allow_llm: bool = Field(default=False)
    recall: str = Field(default="off", description="可选向量召回：off / dense / cross_encoder")
    strategy: str = Field(default="fixed", description="检索策略：fixed / auto（同 /api/recommend 口径）")
    facet_filters: list[dict] | None = Field(default=None, max_length=_MAX_FACET_FILTERS_ITEMS, description="分面细化（同 /api/recommend）")
    suppressed_constraints: list[str] | None = Field(default=None, max_length=_MAX_SUPPRESSED_ITEMS, description="忽略的筛选条件 dim（同 /api/recommend）")
    lenient_dims: list[str] | None = Field(default=None, max_length=_MAX_LENIENT_ITEMS, description="「也纳入未标注」的维度（同 /api/recommend）")
    date_from: str | None = Field(default=None, description="发表时间范围起（ISO YYYY-MM-DD，含）；非法 → 400（同 /api/recommend）")
    date_to: str | None = Field(default=None, description="发表时间范围止（ISO YYYY-MM-DD，含）；非法 → 400（同 /api/recommend）")
    polish: bool = Field(default=True, description="AI 润色推荐说明子开关（同 /api/recommend 缺省口径）；只用于 preliminary_final 判定——polish 实际会跑 = use_llm ∧ polish")
                                                                            
    conv_id: str | None = Field(default=None, max_length=80, description="前端生成的对话 id：本轮对话记入该对话的 session log（复盘数据面）；None=不记录")
    replay_key: str | None = Field(default=None, max_length=64, description="对话的复盘访问 key（匿名分区下防串号的第二道闸）；随 conv_id 生效")


class SearchRescueRequest(BaseModel):
    """检索救回端点 `/api/agent/search-rescue` 的入参（2026-08-16 检索工具化 Phase 1）。

    当前查询零命中时，让 agent 在**收敛面**（只允许 search.rerun / none）下尝试
    换一组查询词重跑本地检索；采纳与否由工具内机械择优闸裁定。LLM 配置覆盖沿用
    `/api/utterance` 的请求级契约。"""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="当前（零命中的）查询句")
    sources: list[str] | None = Field(default=None, max_length=_MAX_SOURCES_ITEMS, description="当前选中的来源池（重检按它取，不因改写漂移换池）")
    current_filters: list[dict] | None = Field(default=None, max_length=_MAX_CURRENT_FILTERS_ITEMS, description="当前生效条件（现场回显用）")
    result_total: int = Field(default=0, ge=0, description="当前结果的命中总数（调用方自述，仅回显）")
                                                                 
                                           
    facet_filters: list[dict] | None = Field(default=None, max_length=_MAX_FACET_FILTERS_ITEMS, description="分面细化（同 /api/recommend）")
    suppressed_constraints: list[str] | None = Field(default=None, max_length=_MAX_SUPPRESSED_ITEMS, description="忽略的筛选条件 dim（同 /api/recommend）")
    lenient_dims: list[str] | None = Field(default=None, max_length=_MAX_LENIENT_ITEMS, description="「也纳入未标注」的维度（同 /api/recommend）")
    date_from: str | None = Field(default=None, description="发表时间范围起（ISO YYYY-MM-DD，含）；非法 → 400（同 /api/recommend）")
    date_to: str | None = Field(default=None, description="发表时间范围止（ISO YYYY-MM-DD，含）；非法 → 400（同 /api/recommend）")
                                                     
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


class _ReplayBase(BaseModel):
    """复盘端点公共入参（2026-09-28 复盘批）：对话 id + 访问 key。
    key 是匿名分区下防串号的第二道闸——服务端首写落 key，之后读写必须匹配，不匹配一律 404。"""
    model_config = ConfigDict(extra="forbid")

    conv_id: str = Field(..., min_length=1, max_length=80, description="前端生成的对话 id")
    key: str = Field(default="", max_length=64, description="复盘访问 key（localStorage 随机生成）")


class _ReplayLlmBase(_ReplayBase):
    """带 LLM 配置覆盖的复盘入参（手动整理 / 复盘对话 / 补建；与 /api/recommend 同契约）。"""
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


class ReplayDigestRequest(_ReplayLlmBase):
    """手动「重新整理」入参（用户显式触发，同步执行）。"""


class ReplayChatRequest(_ReplayLlmBase):
    text: str = Field(..., min_length=1, max_length=4000, description="sidechat 这一轮的用户文本（标注 pill 已展平拼入）")


class ReplayBackfillRequest(_ReplayLlmBase):
    turns: list[dict] = Field(default_factory=list, max_length=60, description="本机历史补建：[{role: user|assistant, text}]；仅服务端尚无记录时生效")


class ReplaySeenRequest(_ReplayBase):
    """开窗已读的核销入参。`observed_version`/`observed_pids` 是可选的版本绑定（additive）：
    只核销用户实际看到的那一版条目，后台新落的成果不被吞（缺省＝旧调用语义，核销当前全部）。"""
    observed_version: "int | None" = Field(
        default=None, ge=0,
        description="points 返回的 digest_version（用户实际渲染的那一版）；缺省＝旧调用语义")
    observed_pids: "list[str] | None" = Field(
        default=None, max_length=2000,
        description="可选：用户实际看到并已渲染的 pid 集（精确回显，优先于 observed_version）")


class ReplayAnnotateRequest(_ReplayBase):
    pid: str = Field(..., min_length=1, max_length=32, description="图上条目/边 id（pid 或 eid）")
    text: str = Field(..., min_length=1, max_length=2000, description="一条用户标注（只记录，不发 LLM）")


class InterpretRequest(BaseModel):
    query: str = Field(..., min_length=1, description="待预览解析的用户查询")
    sources: list[str] | None = Field(default=None, max_length=_MAX_SOURCES_ITEMS, description="自动来源模式下的允许来源池，或手动模式下的显式来源")
    auto_parse_sources: bool = Field(default=False, description="是否从 query 自动识别数据来源专名")


class DiagnoseRequest(BaseModel):
    provider: str = Field(default="zhipuai", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=True)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description="本次诊断临时 key，不持久化")
    base_url: str | None = Field(default=None, description="待诊断的 API 接口地址")
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description="待诊断的模型名")


class CuratePlanRequest(BaseModel):
    """对话式数据库管护的 plan 入参（preview，**零写盘**；search_online 的 plan 会联网查官方源并记请求账本）。"""
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., description="管护动作：list / import / search_online / remove / restore")
    query: str | None = Field(default=None, description="search_online 的搜索关键词")
    source: str | None = Field(default=None, description="import 的归属来源名 / search_online 的官方源键（默认 arrayexpress）")
    species: str | None = Field(default=None, description="search_online 的物种过滤（本地子串过滤）")
    limit: int | None = Field(default=None, ge=1, le=100, description="search_online 候选上限（默认 20，最大 100）")
    filename: str | None = Field(default=None, description="import 的落盘名 / remove 的 external 文件名 / restore 的回收站文件名（含时间戳前缀）")
    payload_json: str | None = Field(default=None, description="import 的数据集 JSON 文本（记录数组或 {\"records\":[…]} 对象）")


class CurateApplyRequest(CuratePlanRequest):
    """管护 apply 入参：**必须回传 plan 给的 confirm_token**；search_online 还需原样回传 plan_result。"""
    confirm_token: str = Field(..., description="plan 返回的 confirm_token；重算比对不一致 → token_mismatch，零写入")
    force: bool = Field(default=False, description="import 撞内容重复时仍确认入库（默认拒绝）")
    plan_result: dict | None = Field(default=None, description="search_online apply：plan 返回的完整结果（含 candidates），原样回传")


class CurateCheckUpdatesRequest(BaseModel):
    """`POST /api/curate/check-updates` 入参。**只读**：无 confirm_token、不落盘。"""
    model_config = ConfigDict(extra="forbid")

    sources: list[str] | None = Field(
        default=None,
        description="要检查的来源名（口语名即可，如「10x」「ArrayExpress」）；null/缺省 = 检查全部已注册来源",
    )


class CurateSyncUpdatesRequest(BaseModel):
    """`POST /api/curate/sync-updates` 入参（2026-08-06 `curate.sync_updates`）。

    检查更新 → 有新增则自动入库的复合流：写侧是 uploads 管线 + 账本 + 回收站可撤回
    （与 search_online 同一授权口径），故无 confirm_token——原子调用没有信任边界要跨。"""
    model_config = ConfigDict(extra="forbid")

    sources: list[str] | None = Field(
        default=None,
        description="要检查并同步的来源名（口语名即可）；null/缺省 = 全部已注册来源",
    )


class CurateRecallRequest(BaseModel):
    """`POST /api/curate/recall` 入参：要撤回的 sync operation_id。

    整次撤回 = 把该 operation 的 created_files[] 全部移入回收站（可重入；operation 不存在 →
    unknown_operation 400）。operation_id 从 sync_updates 返回的 receipt 或 `GET /api/curate/sync-status`
    的 last_operation_id 取。"""
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(
        ...,
        description="sync_updates 返回的 operation_id（sync_ 开头）；最近一次可用 /api/curate/sync-status 查",
    )


class WatchCheckRequest(BaseModel):
    """`POST /api/watch/check` 入参：课题保存的**确定性检索 spec**。

    与 /api/recommend 入参子集同构（关键词/来源/分面/已删约束/宽放维度/日期 + spec_version）。
    端点**强制** strategy=fixed、recall=off、rerank=off、polish=false 重跑（基线语义 §4.1：
    不拿可能经 LLM/语义重排的显示结果当基线）。spec_version 当前只支持 v1（与
    record_fingerprint_schema 同版——检索规格与指纹 schema 一起版本化，未来升级同步加档）。"""
    model_config = ConfigDict(extra="forbid")

    spec_version: str = Field(
        default="v1",
        description="确定性检索规格版本（当前支持 v1；与 record_fingerprint_schema 同版）",
    )
    query: str = Field(default="", description="检索关键词（display_query 的解析结果）")
    sources: list[str] | None = Field(default=None, max_length=_MAX_SOURCES_ITEMS, description="来源白名单；null/缺省 = 默认检索池")
    facet_filters: list[dict] | None = Field(
        default=None, max_length=_MAX_FACET_FILTERS_ITEMS, description="分面细化过滤（同 /api/recommend 口径）")
    suppressed_constraints: list[str] | None = Field(
        default=None, max_length=_MAX_SUPPRESSED_ITEMS, description="「已命中里被删掉的」维度（同 /api/recommend 口径）")
    lenient_dims: list[str] | None = Field(
        default=None, max_length=_MAX_LENIENT_ITEMS, description="宽放维度——字段为空的记录视作通过（同 /api/recommend 口径）")
    date_from: str = Field(default="", description="发表时间起（YYYY-MM-DD）")
    date_to: str = Field(default="", description="发表时间止（YYYY-MM-DD）")


class ExportPackRequest(BaseModel):
    """`POST /api/artifacts/export-pack` 入参（2026-08-22 导出中心）。

    入参是课题**当前状态快照**：导出类型 + 课题的键与状态（name/goal/纳入排除条件/
    candidates（uid+status+reason+verified_at）/check_condition/provenance）。
    **数据集元数据全文由服务端从本地语料解析**（reuse-pack 的 keys-only 哲学）——
    前端不传元数据，杜绝「把数据集描述贴进来」吃进未发表工作的路径。
    `project` 是自由 dict、由 `export_pack.sanitize_snapshot` 逐字段归一（课程快照结构
    由前端 artifacts.js 保证，服务端只做防御性清洗，不复制一遍数据层 schema）。"""
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        ..., description="导出类型：download_list（下载清单）/ citations（引文）/ "
                         "screening_record（筛选记录）/ full（全部研究材料）")
    project: dict = Field(
        ..., description="课题当前状态快照（manifest 所需 uid+状态+理由+核验时间 + "
                         "check_condition + provenance；数据集内容由服务端从本地语料读取）")


class ActSummaryRequest(BaseModel):
    """执行结果 LLM 总结的入参（p10）。**只总结，不执行**——done/gap/policy 行由调用方
    从真实返回值构造后上报，本端点不核实、也不新增任何事实；LLM 缺席/失败 → fail-open，
    `summary_zh=None`，调用方原样保留自己的事实句。"""
    model_config = ConfigDict(extra="forbid")

    verb_zh: str = Field(..., min_length=1, max_length=100, description="动作的中文名（如「联网检索并入库」）")
    utterance: str = Field(default="", max_length=500, description="用户那句原话")
    ok: bool = Field(..., description="本次动作是否成功；False 时总结绝不能说「已」完成")
    done_lines: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=30, description="做到的事实行（每条 ≤500 字）")
    gap_lines: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=30, description="没做到的事实行（每条 ≤500 字）")
    policy_lines: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=30, description="口径说明行（每条 ≤500 字）")
                      
    brief: bool = Field(
        default=False,
        description="true 时走一句话模式（≤35 字、只用事实、ok=false 直说没做成，铁律写死在 prompt）；响应形状不变",
    )
                                                        
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


class SearchReplyRequest(BaseModel):
    """检索回执 LLM 改写的入参（2026-08-30 p11）。**只解说，不检索、不执行**——命中数/展示数/
    命中关键词/解析状态由调用方（前端 board.js，数字全部取自真实检索响应）构造后上报，
    本端点不核实、也不新增任何事实；LLM 缺席/失败 → fail-open，`reply_zh=None`，
    调用方原样保留自己的确定性事实句。`can_suggest` 是下一步建议的**白名单**：
    LLM 只许从中原样挑一条，为空则禁止给任何建议（护栏写进 `search_reply_llm._SEARCH_REPLY_RULES_ZH`）。"""
    model_config = ConfigDict(extra="forbid")

    utterance: str = Field(default="", max_length=500, description="用户那句原话")
    query: str = Field(default="", max_length=500, description="实际检索词（可能被改写过）")
    note: str = Field(default="", max_length=200, description="前置说明（如「我把这句按『X』检索」/改条件留痕）；空=无")
    total: int = Field(default=0, ge=0, description="库中命中总条数")
    shown: int = Field(default=0, ge=0, description="结果区实际展示条数")
    hit_keywords: list[Annotated[str, Field(max_length=100)]] = Field(default_factory=list, max_length=30, description="命中的关键词（原文）")
    resolution_status: str = Field(default="", max_length=50, description="解析状态（空=正常；abstained / clarification_required 等）")
    has_relax: bool = Field(default=False, description="结果区是否给出了可以直接点的放宽方式")
    can_suggest: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=5, description="可建议动作白名单（空=禁止给建议）")
                                                        
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    use_llm: bool = Field(default=False)
    mock_llm: bool = Field(default=False)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


                                                                               
                                  
def _sanitize_facet_filters(raw: object) -> list[dict]:
    """委托 workflow.sanitize_facet_filters（单一真源）。"""
    return sanitize_facet_filters(raw)


def _sanitize_suppressed(raw: object) -> list[str]:
    """委托 workflow.sanitize_suppressed（单一真源）。"""
    return sanitize_suppressed(raw)


def _sanitize_ask_answer(raw: dict | None) -> dict | None:
    """放松卡作答净化（2026-09-14 ask 暂停/重启批）：kind 恒 "ask.relax"——
    不符即 400（与端点其它入参校验同径；合法前端只产这一种）。suppress 过
    `_sanitize_suppressed`（与 suppressed_constraints 同一真源）；label/message/
    choice_key 收编为定长截断的纯文本（注入 decide prompt 的机械事实段，先裁再用）。
    None → None（未携带，绝大多数回合）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict) or str(raw.get("kind") or "") != "ask.relax":
        raise HTTPException(
            status_code=400,
            detail="ask_answer.kind 目前只支持 \"ask.relax\"（环内放松卡作答）。",
            headers={"X-Error-Code": "bad_ask_answer"})
    return {
        "kind": "ask.relax",
        "choice_key": str(raw.get("choice_key") or "")[:64],
        "label": str(raw.get("label") or "")[:200],
        "message": str(raw.get("message") or "")[:500],
        "suppress": _sanitize_suppressed(raw.get("suppress")),
    }


def _sanitize_lenient_dims(raw: object) -> list[str]:
    """委托 workflow.sanitize_lenient_dims（单一真源）；回传排序 list 供 JSON 回显稳定。"""
    return sorted(sanitize_lenient_dims(raw))


def _require_iso_date(value: str | None, *, name: str) -> str:
    """发表时间入参：空 → ""（不限）；否则必须是格式与日历都合法的 YYYY-MM-DD，非法 → 400。

    为什么不能静默吞（旧行为：非「年份打头」一律当没传）：用户给了筛选条件、系统悄悄丢掉，
    结果和预期对不上却无任何提示；更糟的是 "2020-13-45" 这种不存在的日期曾被当作已生效
    条件回显上屏。诚实方向只有一个：给了就校验，不合法就明说。
    （本函数为薄委托：真源在 app/request_validation.validate_iso_date，与 MCP 同一份。）"""
    return _validate_or_400(validate_iso_date, value, name=name)


def _normalize_provider(provider: str | None) -> str:
    """webapp 设置面口径（封闭集）：默认 zhipuai、额外收编 trial，其余未知值一律回落 zhipuai
    （产品意图：用户输入静默收敛到已知通道，不报错）。真源在 llm_client._normalize_provider，
    勿再抄映射表。trial 必须独立成类——漏掉会落进 zhipuai 兜底，试用请求错烧正式 key。"""
    return _llm_normalize_provider(provider, default="zhipuai", extra_aliases={"trial"})


                                                                             
                                                                        
                                                                         
                                 
_rows_from_retrieved = rows_from_retrieved


                                                 
                                                                               
                                                              


_LLM_SECRET_ENV_KEYS = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "ZAI_API_KEY",
    "ZHIPUAI_API_KEY",
    "ZHIPUAI_TOKEN",
)
_NONPUBLIC_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_LEGACY_NUMERIC_LABEL_PATTERN = re.compile(r"^(?:[0-9]+|0x[0-9a-f]+)$")


def _endpoint_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=f"不安全的 API 接口地址：{message}")


def _validate_endpoint_url(raw_url: str | None) -> str:
    """Validate a request-controlled LLM endpoint without resolving DNS.

    Public endpoints must use HTTPS. Plain HTTP is intentionally limited to the
    three explicit local-development hosts supported by the UI. Literal IPs are
    classified locally, while domain names are validated syntactically only.
    The shared HTTP client separately disables automatic redirects.
    """
    value = (raw_url or "").strip()
    if not value:
        return ""
                                                    
                                                     
                                                                          
                                                           
    if _account_gate_required():
        raise HTTPException(status_code=400, detail="网页版不支持自定义接口地址。")
    if any(char.isspace() or ord(char) < 32 for char in value):
        raise _endpoint_error("地址不能包含空白或控制字符。")

    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise _endpoint_error("端口或 URL 格式无效。") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise _endpoint_error("只接受 https 地址（本机调试可用 http）。")
    if parsed.username is not None or parsed.password is not None:
        raise _endpoint_error("地址中不能包含用户名或密码。")
    if not parsed.hostname:
        raise _endpoint_error("缺少主机名。")
    if parsed.query or parsed.fragment or "?" in value or "#" in value:
        raise _endpoint_error("地址里不要带 ? 或 # 及之后的内容。")
    if port is not None and not (1 <= port <= 65535):
        raise _endpoint_error("端口必须在 1 到 65535 之间。")

    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise _endpoint_error("主机名不是有效的 IDNA 域名。") from exc

    if hostname == "localhost":
        if scheme != "http":
            raise _endpoint_error("localhost 仅允许使用 http 本地回环模式。")
        return value

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None

    if address is not None:
        explicit_loopbacks = {ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")}
        if address in explicit_loopbacks:
            if scheme != "http":
                raise _endpoint_error("本机地址（127.0.0.1 等）只能用 http。")
            return value
        if scheme != "https":
            raise _endpoint_error("非回环地址必须使用 https。")
        if (
            not address.is_global
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise _endpoint_error("这个 IP 属于内网或保留地址，不能用作接口地址。")
        return value

    if scheme != "https":
        raise _endpoint_error("http 只允许 localhost、127.0.0.1 或 ::1。")
    if hostname.endswith(_NONPUBLIC_HOST_SUFFIXES):
        raise _endpoint_error("不能使用内网域名。")
    labels = hostname.split(".")
    if (
        len(labels) < 2
        or all(_LEGACY_NUMERIC_LABEL_PATTERN.fullmatch(label) for label in labels)
        or len(hostname) > 253
        or any(not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels)
    ):
        raise _endpoint_error("主机名必须是有效的公网域名或公网 IP。")
    return value


def _endpoint_identity(url: str | None) -> tuple[str, str, int, str] | None:
    """Canonical comparison key for an already trusted or validated endpoint."""
    value = (url or "").strip()
    if not value:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except (UnicodeError, ValueError):
        return None
    path = parsed.path.rstrip("/") or "/"
    return parsed.scheme.lower(), hostname, port, path


                                                             
                                                   
                                                         
                                         
_TRUSTED_HOSTS_ENV = "BIODATA_TRUSTED_HOSTS"


def _parse_trusted_request_hosts(raw: str | None) -> frozenset[str]:
    """解析 `BIODATA_TRUSTED_HOSTS` 为规范小写主机名集合；非法条目抛 ValueError。"""
    entries: list[str] = []
    for part in (raw or "").replace(",", " ").split():
        value = part.strip().lower().rstrip(".")
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            address = None
        if address is not None:
            entries.append(str(address))
            continue
        labels = value.split(".")
        if (
            not value
            or len(labels) < 2
            or len(value) > 253
            or value.endswith(_NONPUBLIC_HOST_SUFFIXES)
            or any(not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels)
        ):
            raise ValueError(
                f"{_TRUSTED_HOSTS_ENV} 含非法主机名（只接受公网 IP 或 ≥2 段域名）: {part!r}"
            )
        entries.append(value)
    return frozenset(entries)


_TRUSTED_REQUEST_HOSTS: frozenset[str] = _parse_trusted_request_hosts(os.getenv(_TRUSTED_HOSTS_ENV))


def _is_supported_local_request_host(hostname: str | None) -> bool:
    """Accept loopback hosts, plus any host explicitly trusted for server deployment.

    本机形态（缺省）只认 loopback；服务器部署经 ``BIODATA_TRUSTED_HOSTS`` 显式放行
    灰度入口（公网 IP / 未来域名），白名单逐项校验过、不含通配。"""
    normalized = (hostname or "").strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    if normalized in _TRUSTED_REQUEST_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _raw_request_host(request: Request) -> tuple[str, int | None] | None:
    """从**原始 Host 头**解析 (hostname, port)；缺失/解析失败 → None（fail-closed）。

    为什么不读 ``request.base_url``：starlette 对匹配不上 ``_HOST_RE`` 的畸形 Host
    （如 ``127.0.0.1:not-a-port``）会静默回退成 uvicorn 绑定地址——畸形输入被「洗」成
    合法 loopback，基于 base_url 的守卫恒放行（真服已验证）。
    守卫必须自己看原始头：``urlsplit("//" + host)`` 后访问 ``port``，畸形端口在这里抛
    ValueError，一律 403，绝不 fail-open 成受信值。
    """
    raw = (request.headers.get("host") or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(f"//{raw}")
        port = parsed.port                                                            
    except ValueError:
        return None
                                                          
                                                             
    if parsed.username is not None or parsed.password is not None:
        return None
    if not parsed.hostname:
        return None
    return parsed.hostname, port


def _cross_origin_detail(origin: str, *, scheme: str, hostname: str, port: int | None) -> str | None:
    """Origin 与受信请求源比对：可放行 → None；拒绝 → 403 文案。

    同源判定的单一真源：路由前 middleware（`_require_loopback_host`）与端点内纵深防御
    （`_require_same_origin`）共用，防两处规则分叉。"""
    try:
        supplied = urllib.parse.urlsplit(origin)
        supplied_port = supplied.port or (443 if supplied.scheme.lower() == "https" else 80)
    except ValueError:
        return "Origin 无效。"
    expected_port = port or (443 if scheme == "https" else 80)
    supplied_origin = (supplied.scheme.lower(), (supplied.hostname or "").lower(), supplied_port)
    expected_origin = (scheme, hostname.lower(), expected_port)
    if (
        supplied.scheme.lower() not in {"http", "https"}
        or supplied.username is not None
        or supplied.password is not None
        or supplied.path not in {"", "/"}
        or supplied.query
        or supplied.fragment
        or supplied_origin != expected_origin
    ):
        return "拒绝非同源请求。"
    return None


                                                                  
                         
_HOST_INVALID_DETAIL = "Host 无效。"
_UNTRUSTED_HOST_DETAIL = "仅接受本机 loopback Host。"


                                                                                  
                                                    
                                                                                    
                                                         
_MAX_RAW_BODY_BYTES = 64 * 1024 * 1024
_UPLOAD_TOO_LARGE_DETAIL = "文件超过 64 MB 上限；单细胞元数据 JSON 一般远小于这个量级，请检查是否传错了文件。"


def _raw_body_too_large_response() -> JSONResponse:
    """413 响应体（中间件 Content-Length 预检直返用；计数路径走 HTTPException 同 detail）。"""
    return JSONResponse(
        status_code=413,
        content={"detail": _UPLOAD_TOO_LARGE_DETAIL},
        media_type="application/json; charset=utf-8",
    )


class _RawBodyLimitMiddleware:
    """原始请求 body 字节上限（体积闸）：Content-Length 预检 + 实际字节计数双闸。

    - Content-Length 头存在且超限 → 不读 body、直接 413 —— JSON/表单在 FastAPI/Pydantic
      模型解析**之前**被拒绝（payload_json 的 64 MB 检查不再发生在解析之后）。
    - chunked / 缺失 / 非数字 / 谎报 Content-Length：包一层计数 receive，累计超过上限
      立即抛 `HTTPException(413)` 中断下游 —— FastAPI 的 body 解析对 HTTPException 原样
      重抛（见 fastapi.routing 的 `except HTTPException: raise`），路由层把它转成 413
      响应；**拒绝前绝不占用完整内存**（multipart 临时文件同理不再先落满盘）。
    - 只作用于 POST/PUT/PATCH/DELETE；GET/HEAD/静态资源不拦。
    - 纯 ASGI 中间件（不走 BaseHTTPMiddleware）：计数 receive 原样透传给下游路由，端点
      读 body 的瞬间即被计数；中断发生在响应开始之前，回 413 安全无竞态。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") not in _UNSAFE_METHODS:
            await self.app(scope, receive, send)
            return
        max_bytes = _MAX_RAW_BODY_BYTES
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > max_bytes:
                    await _raw_body_too_large_response()(scope, receive, send)
                    return
                break
        total = 0

        async def limited_receive() -> dict:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body") or b"")
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_DETAIL)
            return message

        await self.app(scope, limited_receive, send)


                                                 
                                                       
                                          
                                
app.add_middleware(_RawBodyLimitMiddleware)


                                                                       
                                                               
                                                                      
                                                      
                                                         
                                        
_REQUIRE_ACCOUNT_ENV = "BIODATA_REQUIRE_ACCOUNT"
_INVITE_CODE_ENV = "BIODATA_INVITE_CODE"
_AUTH_OPEN_PATHS = frozenset({
    "/api/health",
    "/api/account/register",
    "/api/account/login",
    "/api/account/logout",
    "/api/account/whoami",
                                                        
                                                                       
                                                                     
                                     
    "/api/admin/corpus-sync",
    "/api/admin/corpus-sync/status",
})


def _account_gate_required() -> bool:
    from .llm_admission import account_gate_required
    return account_gate_required()


def _request_user(request: Request) -> "accounts.PublicUser | None":
    if not hasattr(request.state, "account_user"):
        request.state.account_user = accounts.resolve_session(
            request.cookies.get(SESSION_COOKIE), sessions_path=_sessions_store())
    return request.state.account_user


@app.middleware("http")
async def _account_gate(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    user = None
    if path.startswith("/api/"):
        user = await run_in_threadpool(_request_user, request)
    if (_account_gate_required() and path.startswith("/api/")
            and path not in _AUTH_OPEN_PATHS and user is None):
        return _json_utf8(
            {"ok": False, "error": "auth_required", "detail": "请先登录后再使用。"},
            status_code=401)
    if path.startswith("/api/"):
        await run_in_threadpool(_refresh_shared_corpus_cache)
    if user is not None:
        from ..corpus.patch_package import bind_patch_scope                             
        with bind_patch_scope(user.id):
            return await call_next(request)
    return await call_next(request)


@app.middleware("http")
async def _require_loopback_host(request: Request, call_next):
    """Reject DNS-rebinding Host values and cross-origin writes before any route is dispatched.

    同源闸放在这里（而不是只在端点函数体内第一行）的原因：FastAPI 的 pydantic 校验先于
    函数体执行，跨源请求带非法 body 时会先吃到 422 的参数结构细节（xss-sec）——
    middleware 在路由分发与校验之前拦，跨源 + 非法 body 得到 403 而不是 422。
    端点内的 `_require_same_origin` 保留作纵深防御（AST 门
    `test_every_post_route_checks_the_origin` 也要求每个写端点都调它）。"""
    parts = _raw_request_host(request)
    if parts is None:
        return JSONResponse(status_code=403, content={"detail": _HOST_INVALID_DETAIL})
    hostname, port = parts
    if not _is_supported_local_request_host(hostname):
        return JSONResponse(
            status_code=403,
            content={"detail": _UNTRUSTED_HOST_DETAIL},
        )
    if request.method in _UNSAFE_METHODS:
        try:
            _require_same_origin(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


def _content_security_policy(page_html: str = "") -> str:
    hashes = set()
    normalized = page_html.replace("\r\n", "\n").replace("\r", "\n")
    for attributes, script in re.findall(r"<script\b([^>]*)>(.*?)</script\s*>", normalized, re.I | re.S):
        if re.search(r"\bsrc\s*=", attributes, re.I):
            continue
        digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
        hashes.add(f"'sha256-{digest}'")
    scripts = " ".join(["'self'", *sorted(hashes)])
    return f"script-src {scripts}; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """安全头覆盖错误响应；内联脚本按实际内容哈希授权，不限制下载与遥测连接来源。"""
    try:
        response = await call_next(request)
    except Exception:
        traceback.print_exc()
        response = JSONResponse(status_code=500, content={"detail": "服务器内部错误，细节见服务端日志。"},
                                media_type="application/json; charset=utf-8")
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not response.headers.get("content-type", "").lower().startswith("text/html"):
        response.headers.setdefault("Content-Security-Policy", _content_security_policy())
                                                    
                                                 
                                                            
                                                                
    if request.url.path.startswith("/static/"):
        if request.query_params.get("v"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
    return response


@app.middleware("http")
async def _request_logging(request: Request, call_next):
    """统一请求日志（2026-08-15 触发点/ "webapp 整体无 logging" 收口）。

    每请求一行：method + 路径 + 状态码 + 耗时。注册在最后 → 栈上最外层，Host 守卫直返的
    403、`_security_headers` 兜出的 500 也能记到最终状态码；5xx 升 WARNING 便于定位。
    脱敏口径：只取 `request.url.path`——不带 query string、绝不读 body，api_key / 密码
    等敏感字段因此结构上不可能进日志。流式端点的耗时只算到 StreamingResponse 交付
    （不含逐帧推送全程），日志行不改任何响应内容。"""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    log = logger.warning if response.status_code >= 500 else logger.info
    log("HTTP %s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _require_same_origin(request: Request) -> None:
    """Reject unsupported Host values and cross-origin browser POSTs.

    端点内的纵深防御：主闸已在路由前的 `_require_loopback_host` middleware
    （那里用**原始 Host 头**判定），这里仍按 base_url 复核一遍，双重口径。"""
    try:
        expected = urllib.parse.urlsplit(str(request.base_url))
        expected_port = expected.port                                     
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=_HOST_INVALID_DETAIL) from exc
    if not _is_supported_local_request_host(expected.hostname):
        raise HTTPException(status_code=403, detail=_UNTRUSTED_HOST_DETAIL)

    fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        raise HTTPException(status_code=403, detail="禁止跨源请求。")
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return
    detail = _cross_origin_detail(
        origin,
        scheme=expected.scheme.lower(),
        hostname=expected.hostname or "",
        port=expected_port,
    )
    if detail is not None:
        raise HTTPException(status_code=403, detail=detail)


                                                                              
                                                               
                                                           
                                               
                                          
_LLM_INTRO_RATE_LIMIT = 30                            
_LLM_INTRO_RATE_WINDOW = 60.0        
_rate_buckets: "dict[str, deque[float]]" = {}
_rate_lock = threading.Lock()


def _rate_limited(key: str, *, limit: int, window: float) -> bool:
    """滑动窗口频率限制：`key` 在 `window` 秒内已有 ≥`limit` 次 → 拒绝（返回 False）。

    进程内实现（报告建议的「简单频率限制」档）；成功调用会记录一次，超限调用不计数
    （不计失败，避免攻击者用超限请求把配额推满饿死正常用户）。线程安全：`_rate_lock`
    只保护这个进程内字典，不碰 ENV_LOCK——限流绝不能反过来阻塞 LLM 配置物化。"""
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(key, deque())
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


from .llm_admission import temporary_env as _temporary_env


def _build_request_overrides(
    provider: str,
    use_llm: bool,
    mock_llm: bool,
    api_key: str | None,
    base_url: str | None = None,
    model: str | None = None,
    server_provider: str | None = None,
    server_base_url: str | None = None,
) -> dict[str, str | None]:
    overrides: dict[str, str | None] = {
        "LLM_PROVIDER": provider,
        "ENABLE_LLM": "true" if use_llm else "false",
        "MOCK_LLM": "true" if mock_llm else "false",
    }
    api_key = (api_key or "").strip() or None
    if provider in ("mock", "trial"):
                                                           
                                                          
        base_url = ""
        model = ""
    else:
        base_url = _validate_endpoint_url(base_url)
        model = (model or "").strip()

    if api_key and provider != "trial":
                                                                            
                                                                             
                                                         
                                                                
                                             
        overrides["LLM_API_KEY"] = api_key
        if provider == "zhipuai":
            overrides["ZAI_API_KEY"] = api_key
        elif provider == "openai-compatible":
            overrides["OPENAI_API_KEY"] = api_key
    else:
                                                                         
                                                                                
                                                                                  
                                                                                
                                                
                                                                                
                                                               
                                                                                 
                                                                                  
                                                                                 
                                              
                                                                                
                                                                               
                                                                                  
                                                                                  
                                                             
                                                                            
                                                                                
        effective_endpoint = base_url or server_base_url
        provider_differs = provider != _normalize_provider(server_provider)
        endpoint_differs = _endpoint_identity(effective_endpoint) != _endpoint_identity(server_base_url)
        if provider_differs or endpoint_differs:
            overrides.update({key: "" for key in _LLM_SECRET_ENV_KEYS})

                                                     
                                                                           
    if provider == "zhipuai":
        if base_url:
            overrides["LLM_BASE_URL"] = base_url
            overrides["ZHIPUAI_BASE_URL"] = base_url
        if model:
            overrides["LLM_MODEL"] = model
            overrides["ZHIPUAI_MODEL"] = model
    elif provider == "openai-compatible":
        if base_url:
            overrides["LLM_BASE_URL"] = base_url
            overrides["OPENAI_BASE_URL"] = base_url
        if model:
            overrides["LLM_MODEL"] = model
            overrides["OPENAI_MODEL"] = model
    return overrides


def _resolve_llm_request(
    provider: str | None,
    *,
    use_llm: bool,
    mock_llm: bool,
    base_url: str | None,
) -> tuple[str, str, bool, bool]:
    """请求级 LLM 四元组归一化（各端点同径）：provider 归一化 → mock/trial 丢弃请求级
    地址 → mock 推导 → use_llm 推导，返回 ``(provider, requested_base_url, mock_llm, use_llm)``。

    provider 只在启用 LLM 时才有意义：provider==mock 仅当 use_llm 时才当作 mock，
    否则 use_llm=false 会被 "provider==mock" 强行拉回 mock（那样"关闭 LLM"就形同虚设）。
    mock 从不联网、trial 地址锁定服务端托管值，两者的请求级 base_url 一律为空
    （也不经 `_validate_endpoint_url`——锁定的通道不接受任何请求级端点）。"""
    provider = _normalize_provider(provider)
    requested_base_url = "" if provider in ("mock", "trial") else _validate_endpoint_url(base_url)
    mock_llm = bool(mock_llm) or (provider == "mock" and bool(use_llm))
    use_llm = bool(use_llm or mock_llm)
    return provider, requested_base_url, mock_llm, use_llm


def _materialize_request_llm_config(
    provider: str,
    *,
    use_llm: bool,
    mock_llm: bool,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    project_root: Path = CONFIG_ROOT,
    within: Callable[[], None] | None = None,
) -> LLMConfig:
    """请求级 LLM 配置物化唯一通道：ENV_LOCK 内以服务器真实配置为信任基准构造 env 覆盖，
    `_temporary_env` 窗口内物化不可变 LLMConfig——请求级 provider/key/endpoint 冻结进配置
    对象，并发请求的 env 覆盖串行安全，锁外下游不再读 os.environ（锁内最小化：锁内只做
    「读配置/物化」，网络 I/O 一律在锁外）。

    `within`（可选）在同一 env 窗口内执行——需顺带物化其它对象的调用方
    （/api/recommend 的 workflow 与 auto_llm_available）经它挂上，不另开窗口。"""
    from .llm_admission import materialize
    return materialize(config_root=CONFIG_ROOT, project_root=project_root,
        loader=load_llm_config, lock=ENV_LOCK, within=within,
        overrides_factory=lambda server: _build_request_overrides(
            provider=provider, use_llm=use_llm, mock_llm=mock_llm,
            api_key=api_key, base_url=base_url, model=model,
            server_provider=server.provider, server_base_url=server.base_url))


                                                                                
                                                                     
                                                            
                                                       
_LLM_DAILY_PER_USER_ENV = "BIODATA_LLM_DAILY_PER_USER"
_LLM_DAILY_GLOBAL_ENV = "BIODATA_LLM_DAILY_GLOBAL"
_LLM_QUOTA_EXEMPT_ENV = "BIODATA_LLM_QUOTA_EXEMPT"
_TRIAL_DAILY_PER_USER_ENV = "BIODATA_TRIAL_DAILY_PER_USER"
_TRIAL_DAILY_GLOBAL_ENV = "BIODATA_TRIAL_DAILY_GLOBAL"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s=%r 不是整数，按默认值 %d 处理", name, raw, default)
        return default


def _quota_exempt_users() -> frozenset[str]:
    """豁免名单（逗号/空白分隔用户名，产品方用）；比较一律小写（用户名本就规范化小写）。"""
    return frozenset(
        part.strip().lower()
        for part in os.getenv(_LLM_QUOTA_EXEMPT_ENV, "").replace(",", " ").split()
        if part.strip()
    )


def _gate_llm_quota(
    request: Request,
    *,
    cfg: LLMConfig | None,
    provider: str,
    use_llm: bool,
    mock_llm: bool,
    api_key: str | None,
    requested_base_url: str = "",
    require_enabled: bool = True,
) -> None:
    """按执行侧有效配置准入；不受 enable 开关控制的能力显式传 require_enabled=False。"""
    if not _account_gate_required():
        return
    if not use_llm:
        return
    if cfg is None:
        cfg = _materialize_request_llm_config(
            _normalize_provider(provider), use_llm=use_llm, mock_llm=mock_llm,
            api_key=api_key, base_url=requested_base_url, model=None)
    from .llm_admission import gate_quota
    from .durable_jobs import JobStoreError
    user = _request_user(request)
    try:
        gate_quota(PROJECT_ROOT, username=getattr(user, "username", ""), cfg=cfg,
            api_key=api_key, use_llm=use_llm, require_enabled=require_enabled,
            account_required=_account_gate_required())
    except JobStoreError as exc:
        status = {"auth_required": 401, "llm_quota_exceeded": 429}.get(exc.code, 503)
        raise HTTPException(status_code=status, detail=exc.message) from None


def _json_utf8(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
                                                                                   
                                                                      
    return JSONResponse(content=payload, status_code=status_code,
                        media_type="application/json; charset=utf-8")


                                       
                                                
_TELEMETRY_META_ENV = {
    "http://<server-ip>:8471/v1/ingest": "BIODATA_TELEMETRY_ENDPOINT",
    "<your-ingest-token>": "BIODATA_TELEMETRY_TOKEN",
    "<server-ip>": "BIODATA_TELEMETRY_ALLOW_INSECURE",
}

                                            
                                                      
_DEPLOY_FOOTER_PLACEHOLDER = "<!--deploy-footer-->"


def _index_html() -> str:
    text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for placeholder, env_name in _TELEMETRY_META_ENV.items():
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            text = text.replace(placeholder, _html.escape(value, quote=True))
                                                
                                                
                                       
    footer = str(os.environ.get("BIODATA_SITE_FOOTER_HTML") or "").strip()
    text = text.replace(_DEPLOY_FOOTER_PLACEHOLDER, footer)
    return text


                                                                   
                                                                                 
                                                                 
                                                       
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index() -> Response:
    page_html = _index_html()
    return Response(page_html, media_type="text/html; charset=utf-8",
                    headers={"Cache-Control": "no-cache",
                             "Content-Security-Policy": _content_security_policy(page_html)})


                                                               
                                                                                                  
                                                        
@app.api_route("/dataset", methods=["GET", "HEAD"], include_in_schema=False)
def dataset_page() -> FileResponse:
    path = STATIC_DIR / "dataset.html"
    return FileResponse(path, headers={"Cache-Control": "no-cache",
                        "Content-Security-Policy": _content_security_policy(path.read_text(encoding="utf-8"))})


                                                           
_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 22">'
    b'<rect width="22" height="22" rx="5" fill="#0d9488"/>'
    b'<circle cx="11" cy="11" r="5.4" fill="none" stroke="#ffffff" stroke-width="1.9"/>'
    b'<path d="M11 5.6 C6 8.6 16 12.4 11 15.4" fill="none" stroke="#ffffff" stroke-width="1.3" opacity=".7"/>'
    b"</svg>"
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/api/domain/manifest")
def api_domain_manifest() -> JSONResponse:
    """活动领域清单（additive、只读）：品牌/维度/基础来源，供前端与接入方动态化。

    骨肉分离架构（领域包契约见内部架构文档）：前端品牌文案不再硬编码，经本端点
    读取当前领域包声明；缺省 biodata 与旧版逐位一致。
    """
    pack = get_domain()
    b = pack.branding
    return JSONResponse({
        "ok": True,
        "domain": {
            "domain_id": pack.domain_id,
            "name_zh": pack.name_zh,
            "tagline_zh": pack.tagline_zh,
            "base_source": pack.base_source,
            "branding": {
                "product_name": b.product_name,
                "page_title": b.page_title,
                "dataset_page_title": b.dataset_page_title,
                "download_dir_prefix": b.download_dir_prefix,
                "corpus_blurb_zh": b.corpus_blurb_zh,
            },
            "dimensions": [{"name": d.name, "label_zh": d.label_zh} for d in pack.dimensions],
            "raw_data_labels": list(pack.raw_data_labels),
        },
    })


@app.get("/api/health")
def api_health(request: Request) -> JSONResponse:
    with ENV_LOCK:
        server_cfg = load_llm_config(project_root=CONFIG_ROOT)
        zhipu_cfg = load_llm_config(project_root=CONFIG_ROOT, provider_override="zhipuai")
        trial_cfg = load_llm_config(project_root=CONFIG_ROOT, provider_override="trial")
    from ..agent import agent_exec as _agent_exec
    body = {
            "ok": True,
            "service": "dataset-recommender-web",
            "version": WEB_API_VERSION,
                                                   
                                                   
                                                                                
                                              
                                               
                                 
            "install_root": str(PATHS.install_root),
                                                                  
            "runtime_mode": PATHS.runtime_mode,
            "zhipu_config_detected": bool(zhipu_cfg.api_key),
                                                         
                                                                
            "account": {
                "required": _account_gate_required(),
                "invite": bool(os.getenv(_INVITE_CODE_ENV, "").strip()),
            },
                                                                    
                                                                           
            "llm_server": {
                "key_detected": bool(server_cfg.api_key) and server_cfg.enable_llm and server_cfg.provider != "mock",
                "provider": server_cfg.provider,
                "base_url": server_cfg.base_url,
                                                             
                                                               
                "trial": {
                    "available": bool(trial_cfg.api_key) and trial_cfg.enable_llm,
                    "model": trial_cfg.model,
                    "daily_limit": _env_int(_TRIAL_DAILY_PER_USER_ENV, 30),
                },
            },
                                                             
                                                       
            "extensions": {"agent": _agent_exec.agent_available()},
                                                                   
                                                        
                                                            
            "recall_api": _recall_api_health(),
                                         
                                                     
                                                                   
            "corpus": {"gen": _corpus_gen_sentinel()},
        }
    if _account_gate_required():
                                             
        body.pop("install_root", None)
                                                         
                                                         
                                           
        if _request_user(request) is None:
            llm = body.get("llm_server") or {}
            trial = llm.get("trial") or {}
            body["llm_server"] = {
                "key_detected": bool(llm.get("key_detected")),
                "trial": {
                    "available": bool(trial.get("available")),
                    "daily_limit": trial.get("daily_limit", 30),
                },
            }
    return _json_utf8(body)


def _recall_api_health() -> dict:
    """recall_api.api_status 的防爆包装：健康端点绝不因召回探测异常 500。"""
    try:
        from ..retrieval import recall_api                             

        return recall_api.api_status()
    except Exception:
        return {"embed": False, "rerank": False, "model": "", "dimensions": 0}


def _corpus_gen_sentinel() -> "str | None":
    """语料代变更哨兵：**无绑定** corpus_cache_generation 元组
    repr 的 sha256[:12]——不绑补丁作用域（健康端点在登录请求下会带账户补丁作用域，不带掉
    会变成每账户一个哨兵、且被用户上传扰动）。只是变更哨兵，不作对外可复现指纹
    （corpus_cache_generation 注释同口径）；语料未装载/任何异常 → None，health 绝不因此 500。"""
    try:
        from ..corpus.patch_package import unbound_patch_scope                             

        with unbound_patch_scope():
            gen = corpus_cache_generation(DATA_DIR, PROJECT_ROOT)
        return hashlib.sha256(repr(gen).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return None


                                                      
_LOCAL_MODEL_WEB_DETAIL = "网页版使用在线向量服务，无需安装本地模型。"


@app.get("/api/local-model/status")
def api_local_model_status() -> JSONResponse:
    """只回普通状态/体积，不暴露本机路径、uv 输出或下载源原始错误。"""
    return _json_utf8({"ok": True, **model_install_status(PATHS)})


@app.post("/api/local-model/install")
def api_local_model_install(request: Request) -> JSONResponse:
    """用户显式触发后台在线安装；单飞，失败不影响规则排序。"""
    _require_same_origin(request)
                                     
                                            
    if _account_gate_required():
        raise HTTPException(status_code=403, detail=_LOCAL_MODEL_WEB_DETAIL)
    return _json_utf8({"ok": True, **start_model_install(PATHS)})


@app.post("/api/local-model/cancel")
def api_local_model_cancel(request: Request) -> JSONResponse:
    _require_same_origin(request)
    if _account_gate_required():
        raise HTTPException(status_code=403, detail=_LOCAL_MODEL_WEB_DETAIL)
    return _json_utf8({"ok": True, **cancel_model_install(PATHS)})


class AccountCredentials(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="用户名")
    password: str = Field(..., min_length=1, max_length=200, description="密码（仅本机 scrypt 哈希存储，绝不明文/回显/记录）")
    remember: bool = Field(True, description="记住我：30 天免登录；false = 浏览器会话级 cookie")
    invite_code: str | None = Field(default=None, max_length=200, description="邀请码（仅护栏模式 BIODATA_REQUIRE_ACCOUNT=1 的注册需要；闸关时忽略）")


class AccountSwitchPayload(BaseModel):
    token: str = Field(..., min_length=1, max_length=200, description="待切换账号的会话 token（前端按账号记住的）")


_SESSION_MAX_AGE = accounts.SESSION_TTL_DAYS * 24 * 3600                                            
_ACCOUNT_ERROR_STATUS = {
    "bad_username": 400, "weak_password": 400, "store_full": 400,
    "username_taken": 409, "invalid_credentials": 401, "locked": 429,
    "store_corrupt": 503, "store_unavailable": 503,
    "session_store_unavailable": 503, "redis_package_missing": 503,
}


def _accounts_store() -> Path:
    return accounts.default_store_path(PROJECT_ROOT)


def _sessions_store() -> Path:
    return accounts.default_sessions_path(PROJECT_ROOT)


def _account_http_error(exc: AccountError) -> HTTPException:
    return HTTPException(status_code=_status_for(exc.code, _ACCOUNT_ERROR_STATUS), detail=exc.message)


def _set_session_cookie(resp: JSONResponse, token: str, *, remember: bool = True) -> None:
                                                                             
                                                                     
                                                                            
                                                          
                
    resp.set_cookie(key=SESSION_COOKIE, value=token,
                    max_age=_SESSION_MAX_AGE if remember else None,
                    httponly=True, samesite="strict", path="/",
                    secure=os.getenv(_COOKIE_SECURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"})


                                                              
_COOKIE_SECURE_ENV = "BIODATA_COOKIE_SECURE"

                                          
                                                      
_ACCOUNT_LOGIN_RATE_LIMIT = 10                      
_ACCOUNT_REGISTER_RATE_LIMIT = 5                    
_ACCOUNT_RATE_WINDOW = 60.0            


@app.post("/api/account/register")
def api_account_register(payload: AccountCredentials, request: Request) -> JSONResponse:
    """注册本地账户 → 立即登录（下发会话 cookie）。密码仅 scrypt 哈希本地存储，绝不明文/回显。
    响应带 `session_token`：loopback 本地工具，前端按账号记住它用于一键切换（效力等同 cookie）；
    护栏模式（公网）不下发该字段、一键切换端点同时关闭。"""
    _require_same_origin(request)
    if _account_gate_required():
                                                      
        host = request.client.host if request.client else "-"
        if not _rate_limited(f"account-register:{host}",
                             limit=_ACCOUNT_REGISTER_RATE_LIMIT, window=_ACCOUNT_RATE_WINDOW):
            raise HTTPException(status_code=429, detail="注册尝试过于频繁，请稍后再试。")
                                                         
                                                                   
                                                                  
    if _account_gate_required():
        invite = os.getenv(_INVITE_CODE_ENV, "").strip()
        if not invite:
            raise HTTPException(status_code=403, detail="本站暂未开放注册，请联系管理员开通账户。")
        if not hmac.compare_digest((payload.invite_code or "").strip(), invite):
            raise HTTPException(status_code=403, detail="邀请码不正确，请向管理员核对后再试。")
    try:
        user = accounts.register(payload.username, payload.password, store_path=_accounts_store())
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    token = accounts.create_session(user, sessions_path=_sessions_store())
    body: dict[str, Any] = {"ok": True, "user": user.as_dict()}
    if not _account_gate_required():
                                                              
        body["session_token"] = token
    resp = _json_utf8(body)
    _set_session_cookie(resp, token, remember=payload.remember)
    return resp


@app.post("/api/account/login")
def api_account_login(payload: AccountCredentials, request: Request) -> JSONResponse:
    """登录本地账户 → 下发会话 cookie。失败信息不区分「用户不存在 / 密码错」（防枚举）。
    `session_token` 同 register：仅本机形态下发。"""
    _require_same_origin(request)
    if _account_gate_required():
                                              
        host = request.client.host if request.client else "-"
        if not _rate_limited(f"account-login:{host}",
                             limit=_ACCOUNT_LOGIN_RATE_LIMIT, window=_ACCOUNT_RATE_WINDOW):
            raise HTTPException(status_code=429, detail="登录尝试过于频繁，请稍后再试。")
    try:
        user = accounts.authenticate(payload.username, payload.password, store_path=_accounts_store())
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    token = accounts.create_session(user, sessions_path=_sessions_store())
    body = {"ok": True, "user": user.as_dict()}
    if not _account_gate_required():
        body["session_token"] = token
    resp = _json_utf8(body)
    _set_session_cookie(resp, token, remember=payload.remember)
    return resp


@app.post("/api/account/logout")
def api_account_logout(request: Request) -> JSONResponse:
    """登出：销毁服务端会话（含落盘快照）+ 清除 cookie。"""
    _require_same_origin(request)
    try:
        accounts.destroy_session(request.cookies.get(SESSION_COOKIE), sessions_path=_sessions_store())
    except AccountError as exc:
        raise _account_http_error(exc) from exc
    resp = _json_utf8({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/account/whoami")
def api_account_whoami(request: Request) -> JSONResponse:
    """据会话 cookie 返回当前登录账户（无 / 过期 → user:null）；前端据此选记忆命名空间。"""
    user = _request_user(request)
    return _json_utf8({"ok": True, "user": user.as_dict() if user else None})


def _require_session_user(request: Request) -> "accounts.PublicUser":
    """护栏形态端点共用闸：护栏形态 + 有效会话，否则 404/401
    （trial-quota 与在线 MCP 令牌端点共用）。"""
    if not _account_gate_required():
        raise HTTPException(status_code=404, detail="not found")
    user = _request_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="auth_required")
    return user


@app.get("/api/account/trial-quota")
def api_account_trial_quota(request: Request) -> JSONResponse:
    """T3 限量试用通道的当日额度回显（设置界面「今日剩余」数据源，2026-08-25 夜）。

    additive：仅护栏形态提供（试用通道本就只存在于部署形态；本机形态 404，前端按
    「通道不可用」隐藏额度块）。remaining=None 表示该账号不限量（豁免名单或上限≤0），
    前端显示「不限量」；其余字段只报状态，key 绝不回显。"""
    user = _require_session_user(request)
    with ENV_LOCK:
        trial_cfg = load_llm_config(project_root=CONFIG_ROOT, provider_override="trial")
    daily_limit = _env_int(_TRIAL_DAILY_PER_USER_ENV, 30)
    exempt = user.username.lower() in _quota_exempt_users()
    used = llm_quota.usage_snapshot(PROJECT_ROOT, user.username, trial=True)
    unlimited = exempt or daily_limit <= 0
    return _json_utf8({
        "ok": True,
        "available": bool(trial_cfg.api_key) and trial_cfg.enable_llm,
        "model": trial_cfg.model,
        "daily_limit": daily_limit,
        "used": used,
        "remaining": None if unlimited else max(0, daily_limit - used),
        "unlimited": unlimited,
    })


@app.post("/api/account/switch")
def api_account_switch(payload: AccountSwitchPayload, request: Request) -> JSONResponse:
    """一键切换账号：校验前端记住的会话 token → 有效则把 cookie 重设到该账号。
    token 无效/过期 → 401（前端丢弃该条记忆、退回密码登录）。
    公网护栏硬化：护栏模式下一键切换整体关闭（session_token 不再下发，
    前端也不记 token——共用浏览器的公网场景里「点一下换成别人」不成立）。"""
    _require_same_origin(request)
    if _account_gate_required():
        raise HTTPException(status_code=403, detail="网页版不支持一键切换账号。")
    user = accounts.resolve_session(payload.token, sessions_path=_sessions_store())
    if user is None:
        raise HTTPException(status_code=401, detail="这个账号的登录状态已失效，请重新登录。")
    resp = _json_utf8({"ok": True, "user": user.as_dict()})
    _set_session_cookie(resp, payload.token, remember=True)
    return resp


                                                                              
                                                       
                                                     
                                            


def _mcp_tokens_store() -> Path:
    return mcp_tokens.default_tokens_path(PROJECT_ROOT)


def _online_mcp_config_hint(request: Request, raw_token: str) -> dict[str, Any]:
    """给 MCP 客户端的接入配置片段（streamable-HTTP 通用形状，Kimi Code / Claude 等同构）。
    Host 头已经过了受信主机闸（护栏形态），直接回显无风险。"""
    base = f"{request.url.scheme}://{request.headers.get('host', '')}"
    return {"url": f"{base}/mcp", "headers": {"Authorization": f"Bearer {raw_token}"}}


class McpTokenMintPayload(BaseModel):
    """铸币请求。label 仅用户自辨识（列表回显用），不落敏感信息。"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=40)


class McpTokenRevokePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_id: str = Field(min_length=1, max_length=32)


                                          
_MCP_TOKEN_ERROR_STATUS = {"too_many_tokens": 409}


@app.post("/api/account/mcp-token")
def api_account_mcp_token_mint(payload: McpTokenMintPayload, request: Request) -> JSONResponse:
    """铸币 → 明文令牌仅此一次返回 + 可直接复制的客户端配置片段。"""
    _require_same_origin(request)
    user = _require_session_user(request)
    try:
        raw, rec = mcp_tokens.mint_token(user.id, user.username, payload.label,
                                         store_path=_mcp_tokens_store())
    except mcp_tokens.McpTokenError as exc:
        raise HTTPException(
            status_code=_status_for(exc.code, _MCP_TOKEN_ERROR_STATUS), detail=exc.message) from exc
    return _json_utf8({"ok": True, "token": raw, "record": rec,
                       "config": _online_mcp_config_hint(request, raw)})


@app.get("/api/account/mcp-tokens")
def api_account_mcp_tokens_list(request: Request) -> JSONResponse:
    """本账户令牌列表（公开字段：token_id/label/prefix/created_at；绝无明文/摘要）。"""
    user = _require_session_user(request)
    return _json_utf8({"ok": True, "tokens": mcp_tokens.list_tokens(user.id, store_path=_mcp_tokens_store())})


@app.post("/api/account/mcp-token/revoke")
def api_account_mcp_token_revoke(payload: McpTokenRevokePayload, request: Request) -> JSONResponse:
    """吊销（仅属主）：立即生效——verifier 每次请求实时查库，无缓存延迟。"""
    _require_same_origin(request)
    user = _require_session_user(request)
    if not mcp_tokens.revoke_token(user.id, payload.token_id, store_path=_mcp_tokens_store()):
        raise HTTPException(status_code=404, detail="令牌不存在或不属于当前账户。")
    return _json_utf8({"ok": True})


class DreamRequest(BaseModel):
    """dream 记忆整理（手动「整理记忆」按钮）：对话快照 + LLM 配置覆盖。

    服务端零存储：conversations 由前端从本机历史快照组织；LLM 配置随请求带来、绝不持久化
    （与 /api/utterance 同契约：请求级 key 优先，缺省回落服务端配置）。"""

    model_config = ConfigDict(extra="forbid")

    conversations: list[dict] = Field(default_factory=list, max_length=24, description="对话快照 [{query, chat:[{k,t,n}]}]")
    provider: str = Field(default="mock", description=_DESC_PROVIDER)
    api_key: str | None = Field(default=None, max_length=_MAX_API_KEY_CHARS, description=_DESC_REQUEST_API_KEY)
    base_url: str | None = Field(default=None, description=_DESC_CUSTOM_BASE_URL)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS, description=_DESC_CUSTOM_MODEL)


_DREAM_ERROR_STATUS = {"empty_input": 400, "no_key": 400, "llm_failed": 502}


@app.post("/api/dream")
def api_dream(payload: DreamRequest, request: Request) -> JSONResponse:
    """dream 记忆整理：历史对话 → 封闭 JSON 记忆候选（generated:true；解析失败=空清单，绝不编造）。"""
    _require_same_origin(request)
    if payload.api_key:
                                                                  
        provider = "zhipuai" if payload.provider == "zhipuai" else "openai-compatible"
        cfg = LLMConfig(
            enable_llm=True,
            provider=provider,
            api_key=payload.api_key.strip(),
            base_url=_validate_endpoint_url(payload.base_url),
            model=(payload.model or "").strip(),
        )
    else:
                                                                   
                                                             
                                                                        
                                                              
                               
        with ENV_LOCK:
            if _normalize_provider(payload.provider) == "trial":
                cfg = load_llm_config(project_root=CONFIG_ROOT, provider_override="trial")
            else:
                cfg = load_llm_config(project_root=CONFIG_ROOT)
                                                            
        _gate_llm_quota(request, cfg=cfg, provider=payload.provider, use_llm=True,
                        mock_llm=False, api_key=None)
    try:
        result = dream.dream_from_conversations(payload.conversations, config=cfg)
    except dream.DreamError as exc:
        raise HTTPException(status_code=_status_for(exc.code, _DREAM_ERROR_STATUS), detail=exc.message) from exc
    return _json_utf8(result)


class CurateExamplesRequest(BaseModel):
    """成功经验库候选的勾选/忽略（2026-08-13 起用户挑选入库）：候选 id 清单 + 端点坐标。

    base_url/model 只用于算端点指纹（分区键，同口径）；api_key 永不进这个端点。"""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list, max_length=200, description="候选行 id 清单")
    base_url: str = Field(default="", description="当前 AI 端点地址（算端点指纹用）")
    model: str = Field(default="", description="当前模型名（算端点指纹用）")
    provider: str = Field(default="", max_length=64,
                          description="当前 provider 预设（算端点指纹用；缺省空=服务端配置）")
    project_id: str = Field(default="", max_length=80)


def _effective_form_endpoint(provider: str, base_url: str, model: str) -> tuple[str, str]:
    """表单坐标 → **有效端点身份**（管理/收录与 Agent 注入同一套指纹口径）。

    - base_url 非空 = 请求级自定义端点（BYOK）：原样使用；
    - base_url 空（试用预设清空地址 / 护栏模式 / 用服务端默认）＝表单坐标**不是** Agent
      实际使用的端点（Agent 按服务端物化配置算指纹）——按配置物化通道解析出非空地址与
      模型；不解析的话规则/样例存进另一个分区，界面显示「生效中」而 decide 永不注入。
    - `provider`（additive，缺省空 = 服务端配置）：请求显式给了 provider 时经同一物化
      通道解析（试用/自定义 provider 的地址与模型以物化为准）。"""
    url = str(base_url or "").strip()
    mdl = str(model or "").strip()
    if url:
        return url, mdl
    try:
        if str(provider or "").strip():
            cfg = _materialize_request_llm_config(
                _normalize_provider(provider), use_llm=True, mock_llm=False,
                api_key=None, base_url="", model=mdl, project_root=CONFIG_ROOT)
            return str(cfg.base_url or ""), str(cfg.model or "")
        with ENV_LOCK:
            cfg = load_llm_config(project_root=CONFIG_ROOT)
                                              
        return str(cfg.base_url or ""), (mdl or str(cfg.model or ""))
    except Exception:
        return url, mdl


def _userdata_partition(request: Request, base_url: str, model: str,
                        provider: str = "") -> tuple[str, str]:
    """`.userdata/` 账本分区键（成功经验库与协作规则卡**共用**同一套口径）：
    会话账户 principal + 端点指纹 endpoint_fp（有效端点身份，见 `_effective_form_endpoint`）。"""
    from types import SimpleNamespace

    from ..agent import agent_exec as _ax

    eff_url, eff_model = _effective_form_endpoint(provider, base_url, model)
    return _utterance_principal(request), _ax._endpoint_fp_from_config(
        SimpleNamespace(base_url=eff_url, model=eff_model))


def _checked_learning_project(request: Request, project_id: str | None) -> str:
    from .workspace_store import WorkspaceError, WorkspaceStore
    if not project_id:
        return ""
    from ..agent.learning_scope import principal_variants
    for principal in principal_variants(_utterance_principal(request)):
        try:
            return WorkspaceStore(PROJECT_ROOT).get_project(principal, project_id)["id"]
        except WorkspaceError as exc:
            failure = exc
    raise HTTPException(status_code=404, detail=failure.message, headers={"X-Error-Code": failure.code}) from failure


@app.get("/api/curate-examples/pending")
def api_curate_examples_pending(request: Request, base_url: str = "", model: str = "",
                                provider: str = "", project_id: str = "") -> JSONResponse:
    """成功操作样例的候选池待选清单：只有本分区（同账户 + 同端点指纹）的行可见。"""
    from ..agent import agent_exec as _ax

    principal, fp = _userdata_partition(request, base_url, model, provider)
    rows = _ax.list_example_candidates(PROJECT_ROOT, principal=principal, endpoint_fp=fp,
                                       project_id=_checked_learning_project(request, project_id))
    return _json_utf8({"ok": True, "candidates": rows})


@app.post("/api/curate-examples/approve")
def api_curate_examples_approve(payload: CurateExamplesRequest, request: Request) -> JSONResponse:
    """勾选入库：候选迁入正式库（注入侧只读正式库）；返回 approved/duplicated 计数。"""
    _require_same_origin(request)
    from ..agent import agent_exec as _ax

    principal, fp = _userdata_partition(request, payload.base_url, payload.model,
                                        payload.provider)
    result = _ax.approve_example_candidates(PROJECT_ROOT, payload.ids,
                                            principal=principal, endpoint_fp=fp,
                                            project_id=_checked_learning_project(request, payload.project_id))
    return _json_utf8({"ok": True, **result})


@app.post("/api/curate-examples/dismiss")
def api_curate_examples_dismiss(payload: CurateExamplesRequest, request: Request) -> JSONResponse:
    """忽略：候选从池里删除（不进正式库）；返回 dismissed 计数。"""
    _require_same_origin(request)
    from ..agent import agent_exec as _ax

    principal, fp = _userdata_partition(request, payload.base_url, payload.model,
                                        payload.provider)
    result = _ax.dismiss_example_candidates(PROJECT_ROOT, payload.ids,
                                            principal=principal, endpoint_fp=fp,
                                            project_id=_checked_learning_project(request, payload.project_id))
    return _json_utf8({"ok": True, **result})


class CollabRulesRequest(BaseModel):
    """协作规则卡的逐条确认 / 撤回（2026-09-29 记忆层 RSI 落地）。

    候选来自 `/api/dream` 的 `rules` 字段（**预览期 ephemeral、不落地**，与记忆候选同一
    范式）；用户点「确认」才把条目写进 `.userdata/collab_rules.jsonl`。
    base_url/model 只用于算端点指纹（分区键，同口径）；api_key 永不进这个端点。"""

    model_config = ConfigDict(extra="forbid")

    rules: list[dict] = Field(default_factory=list, max_length=20,
                              description="规则候选 [{text,trigger,tier,evidence}]（approve 用）")
    ids: list[str] = Field(default_factory=list, max_length=200,
                           description="规则 id 清单（revoke 用）")
    base_url: str = Field(default="", description="当前 AI 端点地址（算端点指纹用）")
    model: str = Field(default="", description="当前模型名（算端点指纹用）")
    provider: str = Field(default="", max_length=64,
                          description="当前 provider 预设（算端点指纹用；缺省空=服务端配置）")
    project_id: str = Field(default="", max_length=80)


@app.get("/api/rules/list")
def api_rules_list(request: Request, base_url: str = "", model: str = "",
                   provider: str = "", project_id: str = "") -> JSONResponse:
    """本分区（同账户 + 同端点指纹）的全部协作规则：active + retracted
    （撤回是 tombstone，行留在账本里、只是不再注入）。"""
    from ..agent import collab_rules as _cr

    principal, fp = _userdata_partition(request, base_url, model, provider)
    rows = _cr.list_collab_rules(PROJECT_ROOT, principal=principal, endpoint_fp=fp,
                                project_id=_checked_learning_project(request, project_id))
    return _json_utf8({"ok": True, "rules": rows})


@app.post("/api/rules/approve")
def api_rules_approve(payload: CollabRulesRequest, request: Request) -> JSONResponse:
    """逐条确认：写 active；返回 approved/duplicated/rejected 三计数
    （rejected = 收口不过的条数——诚实报数，界面据此说明有几条没入账）。"""
    _require_same_origin(request)
    from ..agent import collab_rules as _cr

    principal, fp = _userdata_partition(request, payload.base_url, payload.model,
                                        payload.provider)
    result = _cr.approve_collab_rules(PROJECT_ROOT, payload.rules,
                                      principal=principal, endpoint_fp=fp,
                                      project_id=_checked_learning_project(request, payload.project_id))
    return _json_utf8({"ok": True, **result})


@app.post("/api/rules/revoke")
def api_rules_revoke(payload: CollabRulesRequest, request: Request) -> JSONResponse:
    """撤回：打 tombstone（不物理删），此后不再注入；返回 revoked 计数。"""
    _require_same_origin(request)
    from ..agent import collab_rules as _cr

    principal, fp = _userdata_partition(request, payload.base_url, payload.model,
                                        payload.provider)
    result = _cr.revoke_collab_rules(PROJECT_ROOT, payload.ids,
                                     principal=principal, endpoint_fp=fp,
                                     project_id=_checked_learning_project(request, payload.project_id))
    return _json_utf8({"ok": True, **result})


@app.post("/api/interpret")
def api_interpret(payload: InterpretRequest, request: Request) -> JSONResponse:
    """只解析查询，不装载语料或执行排序；供输入框实时预览来源与时间约束。"""
    _require_same_origin(request)
    query = payload.query.strip()
                                                                   
    _validate_or_400(validate_query, query)
    workflow = DatasetRecommendationWorkflow()
    resolution = resolve_search_request(
        query,
        payload.sources,
        known_source_values(workflow.settings.data_dir, workflow.settings.project_root),
        auto_parse_sources=bool(payload.auto_parse_sources),
    )
    intent = parse_query(resolution.parsed_query, workflow.settings.keyword_mapping)
    interpretation = resolution.as_dict()
    interpretation["effective_sources"] = list(resolution.sources or ["10x Genomics"])
    interpretation["intent"] = intent_projection(intent)
    return _json_utf8({"ok": True, "interpretation": interpretation})


                                                      
                                                               
                                                  
from .conversation_results import POLICY_ID_SCHEMA, ROUTE_POLICY_VERSION


def _build_policy_id(*, sources, strategy: str, rerank: str, recall: str, model) -> dict:
    """组装 policy_id（additive；任何一步失败都不掀翻检索——调用方包 try）。语料快照
    复用 corpus.corpus_snapshot 单一真源（snapshot_id 由内容决定，语料变了它才变）。"""
    records = load_normalized_corpus(DATA_DIR, PROJECT_ROOT, list(sources) if sources else None)
    snap = corpus_snapshot(records)
    from .conversation_results import policy_from_snapshot
    return policy_from_snapshot(snap, sources=sources, strategy=strategy, rerank=rerank,
        recall=recall, model=model, app_version=WEB_API_VERSION, base_source=_corpus_base_source())


def _policy_id_or_none(**kwargs) -> dict | None:
    """_build_policy_id 的安全壳：组装失败（语料装载异常等）降级 None，绝不掀翻检索。"""
    try:
        return _build_policy_id(**kwargs)
    except Exception:
        return None


from .conversation_results import policy_token as _policy_token, policy_id_string as _policy_id_string


def _policy_response_fields(policy_id: dict | None) -> dict[str, Any]:
    return {"policy_id": policy_id, "policy_id_str": _policy_id_string(policy_id)}


def _experiment_response(payload) -> dict[str, Any] | None:
    """回显服务端实际收到的完整随机化合同；普通流量返回 None。"""
    if not getattr(payload, "experiment_id", None):
        return None
    return {"id": payload.experiment_id, "arm": payload.experiment_arm,
            "propensity": payload.propensity}


@app.post("/api/recommend")
def api_recommend(payload: RecommendRequest, request: Request) -> JSONResponse:
    _require_same_origin(request)
    query = payload.query.strip()
                                                       
                                                      
    _validate_or_400(validate_query, query)

    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)
                                                 
    rerank_backend = "llm" if str(payload.rerank or "").strip().lower() == "llm" else "off"
                                                           
    _recall = str(payload.recall or "").strip().lower()
    recall_backend = _recall if _recall in ("dense", "cross_encoder", "vector") else "off"
                                                           
                                                  
    date_from = _require_iso_date(payload.date_from, name="date_from")
    date_to = _require_iso_date(payload.date_to, name="date_to")
                                                     
                                             
                                                 
    _validate_or_400(validate_date_window, date_from, date_to)
                                                             
                                                        
                                                         
    if payload.sources:
        _validate_or_400(validate_sources, payload.sources,
                         known=known_source_values(DATA_DIR, PROJECT_ROOT))
                                                                  
    facet_filters = _sanitize_facet_filters(payload.facet_filters)
                                                          
    suppressed_constraints = _sanitize_suppressed(payload.suppressed_constraints)
                                                                    
    lenient_dims = _sanitize_lenient_dims(payload.lenient_dims)
                                                                                     
    strategy = "auto" if str(payload.strategy or "").strip().lower() == "auto" else "fixed"
                                                                                
                                                                      
    recall_available = None
    preferred_recall = "cross_encoder"
    if strategy == "auto":
        from ..retrieval.vector_recall import resolve_auto_recall_preference
        preferred_recall, recall_available = resolve_auto_recall_preference()

    _workflow_box: dict[str, Any] = {}

    def _attach_workflow() -> None:
                                                         
                                  
        _workflow_box["workflow"] = DatasetRecommendationWorkflow()
        auto_llm_available = None
        if strategy == "auto":
            auto_llm_available = bool(load_llm_config(project_root=CONFIG_ROOT).api_key) if payload.auto_allow_llm else False
        _workflow_box["auto_llm_available"] = auto_llm_available

                                                                         
                                                                    
    request_llm_config = _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model,
        project_root=get_settings().project_root, within=_attach_workflow)
    workflow = _workflow_box["workflow"]
    auto_llm_available = _workflow_box["auto_llm_available"]

                                    
    llm_admitted = False

    def _before_llm(config: LLMConfig) -> None:
        nonlocal llm_admitted
        if llm_admitted:
            return
        _gate_llm_quota(request, cfg=config, provider=provider, use_llm=True,
                        mock_llm=config.mock_llm, api_key=payload.api_key)
        llm_admitted = True

    meta = workflow.run_with_meta(
        RecommendParams(
            query=query,
            top_k=payload.top_k,
                                                              
                                                                 
            use_llm=use_llm and bool(payload.polish),
            mock_llm=mock_llm,
            provider=provider,
            rerank_backend=rerank_backend,
            rerank_top_n=payload.rerank_top_n,
            recall_backend=recall_backend,
            date_from=date_from,
            date_to=date_to,
            sources=payload.sources,
            auto_parse_sources=bool(payload.auto_parse_sources),
            facet_filters=facet_filters,
            suppressed_constraints=suppressed_constraints,
            lenient_dims=lenient_dims,
            strategy=strategy,
            recall_available=recall_available,
            llm_available=auto_llm_available,
            preferred_recall=preferred_recall,
            base_llm_config=request_llm_config,
        ),
        before_llm=_before_llm,
    )

                                           
                                                  
    results = _rows_from_retrieved(meta.retrieved_data)
    warnings: list[str] = []
    if meta.fallback_reason:
        warnings.append(meta.fallback_reason)
                                                              
                                                   
    if (provider == "mock" and bool(payload.use_llm) and not bool(payload.mock_llm)
            and "provider" not in payload.model_fields_set):
        warnings.append("未指定 provider，本次按 mock 处理（如需真实 LLM，请显式指定 provider）")

                                             
                                                                           
                                                           
                                                     
                                                             
                                                            
    try:
        _iproj = (getattr(meta, "interpretation", {}) or {}).get("intent") or {}
        _oov: list[str] = []
        if (getattr(meta, "resolution_status", "") == "abstained"
                and _iproj.get("abstain_reason") == "unresolved_term"):
            _oov = [str(t).strip() for t in (_iproj.get("unresolved_terms") or []) if str(t).strip()]
        elif getattr(meta, "resolution_status", "") == "degraded":
            _oov = [str(t).strip()
                    for t in ((getattr(meta, "degraded_search", None) or {}).get("ignored_terms") or [])
                    if str(t).strip()]
        elif (getattr(meta, "resolution_status", "") == "vector_fallback"
                and (getattr(meta, "vector_fallback", None) or {}).get("reason") == "unresolved_term"):
                                                         
            _oov = [str(t).strip()
                    for t in ((getattr(meta, "vector_fallback", None) or {}).get("ignored_terms") or [])
                    if str(t).strip()]
        if _oov:
                from ..corpus.net_ledger import append_jsonl, net_ledger_path, now_iso
                append_jsonl(net_ledger_path(PROJECT_ROOT).parent / "oov_terms.jsonl", {
                    "ts": now_iso(), "query": str(query or "")[:120], "terms": _oov[:8],
                })
    except Exception as exc:
                                                 
                                                           
        print(f"OOV 词表日志写入失败（{type(exc).__name__}: {exc}），本次跳过，不影响检索。", file=sys.stderr)

                                                                           
                                                             
    from ..content import identifiers
    identifier_lookup = identifiers.lookup(query, lambda: load_full_corpus(DATA_DIR, PROJECT_ROOT))

                                                  
                                                               
                                     
    relaxation_options = [
        {
            "key": opt.get("key", ""),
            "label": opt.get("label", ""),
            "kind": opt.get("kind", "drop"),                                       
            "count": opt.get("count", 0),
            "suppress": list(dict.fromkeys(
                suppressed_constraints + [str(s) for s in (opt.get("suppress") or [])])),
            "results": _rows_from_retrieved(opt.get("retrieved_data", [])),
        }
        for opt in getattr(meta, "relaxation_options", [])
    ]
                                                               
                                                 
    _deg = getattr(meta, "degraded_search", None)
    degraded_search = None
    if _deg:
        degraded_search = {
            "ignored_terms": list(_deg.get("ignored_terms", [])),
            "query": _deg.get("query", ""),
            "count": _deg.get("count", 0),
            "results": _rows_from_retrieved(_deg.get("results", [])),
            "active_filters": _deg.get("active_filters", []),
        }

                                                                          
    _replay_record_safe(_utterance_principal(request), payload.conv_id, payload.replay_key, "search", {
        "query": query[:500],
        "total": int(getattr(meta, "result_total", len(results)) or 0),
        "status": str(getattr(meta, "resolution_status", "results") or ""),
        "top": [str(r.get("dataset_name") or "")[:80] for r in results[:5] if r.get("dataset_name")],
    }, request_llm_config, before_llm=_before_llm)

    from .conversation_results import complete_payload
    body = complete_payload(meta, query=query, provider=provider, sources=payload.sources,
        params={"top_k": payload.top_k, "strategy": strategy, "rerank": rerank_backend,
                "recall": recall_backend, "facet_filters": facet_filters,
                "suppressed_constraints": suppressed_constraints, "lenient_dims": lenient_dims,
                "date_from": date_from, "date_to": date_to, "polish": payload.polish},
        policy=_policy_response_fields(_policy_id_or_none(sources=payload.sources, strategy=strategy,
            rerank=rerank_backend, recall=recall_backend, model=request_llm_config.model)),
        experiment=_experiment_response(payload))
    body["warnings"] = warnings
    return _json_utf8(body)


@app.post("/api/feasibility")
def api_feasibility(payload: RecommendRequest, request: Request) -> JSONResponse:
    """N12 可行性概览：一个研究问题 → 有多少可复用数据、够不够、缺口在哪。

    聚合**通过硬过滤的全部候选**（top_k 放大到 5000 抓全命中集，非 top-k 排序结果），给确定性概览：
    候选数 / 总细胞量**下限** / 物种·平台·年份·来源分布 / 可下载率 / 缺口。不调用 LLM、不写盘。
    与 MCP `assess_feasibility` 共用 `feasibility.build_report` 单一真源；检索器/评测从不 import feasibility。
    """
    _require_same_origin(request)
    query = payload.query.strip()
    _validate_or_400(validate_query, query)
    from ..retrieval import feasibility
    _df = _require_iso_date(payload.date_from, name="date_from")
    _dt = _require_iso_date(payload.date_to, name="date_to")
                                                     
                                                             
                                              
    _validate_or_400(validate_date_window, _df, _dt)
    if payload.sources:
        _validate_or_400(validate_sources, payload.sources,
                         known=known_source_values(DATA_DIR, PROJECT_ROOT))
    _TOP = 5000
    meta = DatasetRecommendationWorkflow().run_with_meta(
        RecommendParams(
            query=query,
            top_k=_TOP,
            use_llm=False,
            date_from=_df,
            date_to=_dt,
            sources=payload.sources,
            auto_parse_sources=bool(payload.auto_parse_sources),
            facet_filters=_sanitize_facet_filters(payload.facet_filters),
            suppressed_constraints=_sanitize_suppressed(payload.suppressed_constraints),
            lenient_dims=_sanitize_lenient_dims(payload.lenient_dims),
        )
    )
                                                                          
                                                                
    survivors = meta.retrieved_data
    result_total = getattr(meta, "result_total", len(survivors))
    truncated = result_total > len(survivors)
    report = feasibility.build_report(survivors, result_total, truncated)
    return _json_utf8({
        "ok": True,
        "report": report,
        "resolution_status": getattr(meta, "resolution_status", "results"),
    })


def _task_pack_retrieval(params: dict) -> dict:
    """按给定的检索参数跑一次检索，把任务包需要的一切一次性算齐。

    预览与产包**走同一个函数**：产包时用回传的同一套参数重跑，再比对指纹。
    两处各写一遍检索调用，就是给「包里叙述的检索与实际跑的检索不是同一次」留口子。
    """
    from ..content import item_view, task_pack
    from ..corpus.corpus import load_normalized_corpus

    _validate_or_400(validate_query, params.get("query"))
    if params.get("query_effective"):
        _validate_or_400(validate_query, params["query_effective"])
    sources = params.get("sources") or None
                                            
                                                                
                                
    _validate_or_400(validate_date_window,
                     params.get("date_from") or "", params.get("date_to") or "")
    meta = DatasetRecommendationWorkflow().run_with_meta(
        RecommendParams(
            query=params["query_effective"] or params["query"],
            top_k=5000,
            use_llm=False,
            date_from=params.get("date_from") or "",
            date_to=params.get("date_to") or "",
            sources=sources,
            auto_parse_sources=bool(params.get("auto_parse_sources")),
            facet_filters=_sanitize_facet_filters(params.get("facet_filters")),
            suppressed_constraints=_sanitize_suppressed(params.get("suppressed_constraints")),
            lenient_dims=_sanitize_lenient_dims(params.get("lenient_dims")),
        )
    )
    records = load_normalized_corpus(DATA_DIR, PROJECT_ROOT, sources)
    by_uid = {}
    for record in records:
        raw = record.raw if isinstance(record.raw, dict) else {}
        uid = str(raw.get("dataset_uid") or "")
        if uid and uid not in by_uid:
            by_uid[uid] = record
    ordered_uids = [str(row.get("dataset_uid") or "") for row in meta.retrieved_data]
    ordered_uids = [uid for uid in ordered_uids if uid in by_uid]
    limit = task_pack.sanitize_limit(params.get("limit"))
    candidate_uids = ordered_uids[:limit]
    items = [item_view.build_item(by_uid[uid], include_introduction=True) for uid in candidate_uids]
    honesty = {
        "active_filters": meta.active_filters,
        "coverage_caveats": meta.coverage_caveats,
        "unused_query_terms": meta.unused_query_terms,
        "or_handling": meta.or_handling,
        "search_trace_summary": (meta.search_trace or {}).get("summary", ""),
        "result_total": meta.result_total,
    }
    return {"meta": meta, "records": records, "items": items,
            "candidate_uids": candidate_uids, "by_uid": by_uid,
            "result_uids": set(ordered_uids), "honesty": honesty}


def _task_pack_params(payload: "TaskPackPreviewRequest") -> dict:
    """把检索口径固化成一个可原样回传的块。auto_parse_sources 必须带上——
    少这一个字段，产包时重跑的检索就可能与预览时不是同一次。"""
    return {
        "query": (payload.query or "").strip(),
        "query_effective": (payload.query_effective or "").strip(),
        "sources": list(payload.sources) if payload.sources else None,
        "auto_parse_sources": bool(payload.auto_parse_sources),
        "facet_filters": _sanitize_facet_filters(payload.facet_filters),
        "suppressed_constraints": _sanitize_suppressed(payload.suppressed_constraints),
        "lenient_dims": _sanitize_lenient_dims(payload.lenient_dims),
        "date_from": (payload.date_from or "").strip(),
        "date_to": (payload.date_to or "").strip(),
        "limit": payload.limit,
        "scope": payload.scope,
    }


@app.post("/api/task-pack/preview")
def api_task_pack_preview(payload: TaskPackPreviewRequest, request: Request) -> JSONResponse:
    """一句话任务包 · 第一步：先看清单。

    **刻意不是一句话直接产包。** 用户没勾过任何东西就打包，等于替他决定了「哪 10 条算数」；
    先把「会包含什么、装不了什么、缺什么」摊开，他确认过的材料才敢拿去投稿。

    只读、离线、不调用 LLM、不写盘。候选池由服务端在**全部命中**里按当前排序截取，
    与页面上恰好显示了几张卡无关。
    """
    _require_same_origin(request)
    from ..content import task_pack
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    _validate_pack_sources(payload.sources)
                                                          
                                  
    _require_iso_date(payload.date_from, name="date_from")
    _require_iso_date(payload.date_to, name="date_to")
    try:
        params = _task_pack_params(payload)
        task_pack.sanitize_limit(params["limit"])
        scope = task_pack.sanitize_scope(payload.scope)
        run = _task_pack_retrieval(params)
        if not run["items"]:
            return _json_utf8({
                "ok": True, "plan": None,
                "resolution_status": getattr(run["meta"], "resolution_status", "no_match"),
                "message_zh": "这次检索没有命中任何数据集，没有可以打包的内容。",
            })
        pack = task_pack.build_task_pack(
            query=query, items=run["items"], records=run["records"],
            scope=scope, retrieval_params=params, honesty=run["honesty"],
            membership={uid: "in_result_set" for uid in run["candidate_uids"]},
        )
    except task_pack.TaskPackError as exc:
        raise HTTPException(status_code=400, detail=str(exc),
                            headers={"X-Error-Code": exc.code}) from None
    keep = [uid for uid in (payload.keep_selected or []) if uid in run["candidate_uids"]]
    dropped = [uid for uid in (payload.keep_selected or []) if uid not in run["candidate_uids"]]
    return _json_utf8({
        "ok": True,
        "resolution_status": getattr(run["meta"], "resolution_status", "results"),
        "result_total": run["honesty"]["result_total"],
        "plan": _preview_projection(pack, run),
        "keep_selected": keep,
        "dropped_selected": dropped,
        "message_zh": ("你之前勾选的 %d 条已不在新的候选里，已自动取消勾选。" % len(dropped)) if dropped else "",
    })


def _preview_projection(pack: dict, run: dict) -> dict:
    """预览面板要的东西。**坏消息排在最前面**——装不了什么，先说。"""
    from ..corpus import download_script
    from ..content import task_pack as task_pack_module
    plan = pack["plan"]
    return {
        "plan_token": pack["plan_token"],
        "scope": pack["scope"],
        "retrieval": pack["retrieval"],
        "retrieval_params": pack["retrieval_params"],
        "cannot_include": plan["manual"],
        "primary_only_zh": download_script.primary_only_sentence(plan),
                                                    
                                                   
                                                               
        "primary_only_policy_zh": (download_script.PRIMARY_ONLY_POLICY_ZH
                                   if pack["scope"] == "primary"
                                   else download_script.ALL_FILES_POLICY_ZH),
        "tiers": plan["tiers"],
        "estimate": plan["estimate"],
        "ledger": plan["ledger"],
        "inspection": plan["inspection"],
        "items": [{
            "dataset_uid": it["dataset_uid"], "dataset_name": it["dataset_name"],
            "source": it["source"], "tier": it["tier"], "tier_evidence": it["tier_evidence"],
            "n_files_selected": it["n_files_selected"], "n_files_total": it["n_files_total"],
            "bytes_selected": it["bytes_selected"], "rows_planned": it["rows_planned"],
            "page_url": it["page_url"],
        } for it in plan["items"]],
        "todo": pack["todo"],
        "candidate_uids": run["candidate_uids"],
        "pack_files": list(task_pack_module.PACK_FILES),
    }


def _validate_pack_sources(sources) -> None:
    """拼错的来源名必须显式报错，绝不静默判空（同型 bug 已经在可行性概览上出过一次）。
    （批起为薄委托：形状闸+空白闸+未知闸的真源在 app/request_validation.validate_sources，
    与 /api/recommend、MCP 同一份。）"""
    if not sources:
        return
    _validate_or_400(validate_sources, sources, known=known_source_values(DATA_DIR, PROJECT_ROOT))


@app.post("/api/task-pack/build")
def api_task_pack_build(payload: TaskPackBuildRequest, request: Request) -> Response:
    """一句话任务包 · 第二步：产包。

    **用回传的同一套检索参数重跑一遍，再算一次指纹。** 对不上就明确说哪里变了、
    一个字节都不产出——绝不发一份与用户看过的预览不符的材料。

    锁的是**候选池**不是勾选：少勾几条完全合法，不会触发不一致。
    """
    _require_same_origin(request)
    from ..content import task_pack
                                                                  
                                                    
                                        
    retrieval_date = _require_iso_date(payload.retrieval_date, name="retrieval_date")
    try:
                                                  
                                                           
                                       
        missing_fp = [name for name, value in (("plan_token", payload.plan_token),
                                               ("snapshot_id", payload.snapshot_id),
                                               ("content_digest", payload.content_digest))
                      if not (value or "").strip()]
        if missing_fp:
            raise task_pack.TaskPackError(
                "bad_param",
                "缺少指纹字段：" + "、".join(missing_fp) + "；请先预览（preview）一次，再原样回传。",
            )
        selected = task_pack.sanitize_uids_for_pack(payload.selected_uids)
        scope = task_pack.sanitize_scope(payload.scope)
        params = dict(payload.retrieval_params or {})
        params.setdefault("query_effective", "")
        if not str(params.get("query") or "").strip():
            raise task_pack.TaskPackError("bad_param", "retrieval_params 里缺少 query。")
        _validate_pack_sources(params.get("sources"))
        run = _task_pack_retrieval(params)
        rebuilt = task_pack.build_task_pack(
            query=params["query"], items=run["items"], records=run["records"],
            scope=scope, retrieval_params=params, honesty=run["honesty"],
            today=retrieval_date or None,
            membership={uid: "in_result_set" for uid in run["candidate_uids"]},
        )
    except task_pack.TaskPackError as exc:
        raise HTTPException(status_code=400, detail=str(exc),
                            headers={"X-Error-Code": exc.code}) from None

    mismatch = _pack_mismatch(payload, rebuilt, run)
    if mismatch:
        return _json_utf8({"ok": False, **mismatch}, status_code=409)

    missing = [uid for uid in selected if uid not in run["candidate_uids"]]
    if missing:
        return _json_utf8({
            "ok": False, "code": "plan_mismatch",
            "message_zh": "这份清单和刚才预览的不是同一批了。请重新预览一次，再生成。",
            "hint_zh": "本次没有生成任何文件。",
        }, status_code=409)

    chosen_items = [it for it in run["items"] if it["dataset_uid"] in set(selected)]
    try:
        pack = task_pack.build_task_pack(
            query=params["query"], items=chosen_items, records=run["records"],
            scope=scope, retrieval_params=params, honesty=run["honesty"],
            today=retrieval_date or None,
            membership={uid: "in_result_set" for uid in selected},
        )
        files = task_pack.render_files(pack)
        if (payload.format or "zip").lower() == "json":
                                                 
                                                           
                                                  
            total_bytes = sum(len(f["text"].encode("utf-8")) for f in files)
            if total_bytes > task_pack.MAX_ZIP_BYTES:
                return _json_utf8({
                    "ok": False, "code": "too_large",
                    "message_zh": (f"文本版共约 {total_bytes // 1024} KB，超过上限 "
                                   f"{task_pack.MAX_ZIP_BYTES // 1024 // 1024} MB。请少选几个数据集，"
                                   "或改用 format=zip（压缩后体积小得多）。")}, status_code=413)
            return _json_utf8({"ok": True, "plan_token": rebuilt["plan_token"],
                               "files": [{"name": f["path"], "text": f["text"]} for f in files]})
        blob = task_pack.files_to_zip_bytes(files)
    except task_pack.TaskPackError as exc:
        status = 413 if exc.code == "too_large" else 400
        return _json_utf8({"ok": False, "code": exc.code, "message_zh": str(exc)}, status_code=status)
    name = f"biodata-task-pack-{pack['retrieval']['date']}-{pack['plan_token'][:8]}.zip"
    return Response(content=blob, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _pack_mismatch(payload: "TaskPackBuildRequest", rebuilt: dict, run: dict) -> "dict | None":
    """哪里变了，就说哪里变了。四种情况的处理方式对用户完全不同，不能糊成一句「变了」。

    空指纹在端点入口已被拒（bad_param），走到这里的三件套**必然非空**——直接比对，
    不再有「if 给了才比」的短路（那曾把「没核对」当成「核对通过」， T2）。"""
    retrieval = rebuilt["retrieval"]
    if payload.snapshot_id != retrieval["snapshot_id"]:
        return {"code": "snapshot_changed",
                "message_zh": "数据集清单在你预览之后变过（例如有人上传了新数据）。请重新预览一次。",
                "hint_zh": "本次没有生成任何文件。"}
    if payload.content_digest != retrieval["content_digest"]:
        return {"code": "catalog_content_changed",
                "message_zh": "数据集编号没变，但其中一些记录的内容在你预览之后被更新过"
                              "（例如来源补录了组织字段）。请重新预览一次。",
                "hint_zh": "本次没有生成任何文件。"}
    if payload.plan_token != rebuilt["plan_token"]:
        return {"code": "configuration_mismatch",
                "message_zh": "生成时用的检索条件和预览时不一样。请重新预览一次。",
                "hint_zh": "本次没有生成任何文件。"}
    return None


@app.post("/api/board/plan")
def api_board_plan(payload: BoardPlanRequest, request: Request) -> JSONResponse:
    """条件板：把「再说一句话改条件」规划成一次具体的改动。**只规划，不检索。**

    无状态、纯计算：不装载数据集目录、不落盘、不写日志、不联网、不用大模型。
    返回里的 `next_request` 才是拿去调 `/api/recommend` 的东西；
    `status` 不是 auto_apply / needs_confirm 时，`next_request` 恒为 None —— 一个字节都没改。

    与 MCP `plan_query_edit` 共用 `board.plan_edit` 单一真源；
    检索器 / 编排 / 冻结评测从不 import board → 767 基准结构性不受影响。
    """
    _require_same_origin(request)
    from . import board
    try:
        plan = board.plan_edit(
            payload.query or "",
            payload.utterance or "",
            forced_op=payload.forced_op or "",
            dim=payload.dim or "",
            candidate_override=payload.candidate_override or "",
            current_filters=payload.current_filters,
            resolution=payload.resolution,
            suppressed_constraints=payload.suppressed_constraints,
            lenient_dims=payload.lenient_dims,
            facet_filters=payload.facet_filters,
            coverage_dims=payload.coverage_dims,
            date_from=payload.date_from or "",
            date_to=payload.date_to or "",
                                                            
                                                        
            keyword_mapping=get_settings().keyword_mapping,
        )
    except board.BoardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _json_utf8({"ok": True, **plan})


def _run_action_plan(
    utterance: str,
    *,
    has_results: bool,
    result_total: int,
    provider: str,
    use_llm: bool,
    mock_llm: bool,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> dict:
    """按请求级覆盖跑 `action_plan.plan_action`（ENV_LOCK + `_temporary_env` 链）。

    `/api/action/plan` 与 `/api/utterance` 的 action/歧义分支共用这一条——LLM 配置链
    （请求级 key 隔离、endpoint 校验、server 配置对照）只有一份，不手抄。
    """
    from ..agent import action_plan as _ap

    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        provider, use_llm=use_llm, mock_llm=mock_llm, base_url=base_url)

                                                         
                                                                            
    cfg = _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=api_key, base_url=requested_base_url, model=model)
    try:
        return _ap.plan_action(
            utterance,
            has_results=bool(has_results),
            result_total=int(result_total or 0),
            config=cfg,
        )
    except _ap.ActionPlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc),
                            headers={"X-Error-Code": exc.code}) from None


@app.post("/api/action/plan")
def api_action_plan(payload: ActionPlanRequest, request: Request) -> JSONResponse:
    """一句话 → **该做哪一个动作**（封闭动词表里的一个）。**只出 plan，不执行任何动作。**

    这条分层是整个「说了就做」方案里最值钱的判断：本端点不检索、不落盘、不产任何交付物。
    真正的执行发生在调用方（网页 `act.js` 派发既有能力、MCP 调用方自己决定）。
    于是 `tests/test_action_markers.py` 那条「检索端点永不夹带产物」的诚实性分层原样成立，
    自动执行落地**不需要重基线任何既有的门**。

    `has_results` / `result_total` 由调用方自述、服务端无从核实 —— 所以它们只用来置
    `blocked_reason`，返回体里**不生成任何带这两个数字的成句断言**。

    与 MCP `plan_action` 共用 `action_plan.plan_action` 单一真源；
    检索器 / 编排 / 冻结评测从不 import action_plan → 冻结 767 基准结构性不受影响。
    """
    _require_same_origin(request)
                                                                   
                                
    _gate_llm_quota(
        request, cfg=None, provider=payload.provider,
        use_llm=bool(payload.use_llm or payload.mock_llm), mock_llm=bool(payload.mock_llm),
        api_key=payload.api_key,
        requested_base_url="" if _normalize_provider(payload.provider) in ("mock", "trial")
        else _validate_endpoint_url(payload.base_url))
    plan = _run_action_plan(
        payload.utterance,
        has_results=bool(payload.has_results),
        result_total=int(payload.result_total or 0),
        provider=payload.provider,
        use_llm=bool(payload.use_llm),
        mock_llm=bool(payload.mock_llm),
        api_key=payload.api_key,
        base_url=payload.base_url,
        model=payload.model,
    )
    return _json_utf8({"ok": True, "plan": plan})


def _utterance_principal(request: Request) -> str:
    """请求身份快照决定分区；只有准入时没有有效会话才使用匿名分区。"""
    user = _request_user(request)
    return str(getattr(user, "id", "") or "") or "anonymous"


                                                                                         
                                                             
                                            
                                 

def _replay_record_safe(principal: str, conv_id: str | None, replay_key: str | None,
                        kind: str, record: dict, cfg: "LLMConfig | None", *,
                        before_llm: Callable[[LLMConfig], None] | None = None) -> None:
    """复盘落痕挂钩：fail-soft 总闸——任何异常只进日志，绝不掀翻主路径。
    收 principal 而不是 request，供流式 worker 共用；before_llm 原样透传请求级准入。"""
    if not conv_id:
        return
    try:
        from . import replay_service
        replay_service.record_turn(
            PROJECT_ROOT, principal, str(conv_id)[:80],
            str(replay_key or "")[:64], kind, record, cfg, before_llm=before_llm)
    except Exception:
        logger.exception("复盘落痕失败（不影响主路径）：kind=%s", kind)


def _replay_record_utterance(principal: str, payload: UtteranceRequest,
                             text: str, result: dict, cfg: "LLMConfig | None") -> None:
    """一轮 /api/utterance → chat 记录（用户原话 + 路由 + 步骤 + 回复；浓缩截断在 replay_digest）。"""
    if not isinstance(result, dict):
        return
    plan = result.get("plan") or {}
    steps = [
        {"verb_zh": str(s.get("verb_zh") or s.get("verb") or "")[:60], "ok": bool(s.get("ok", True))}
        for s in (plan.get("steps") or [])[:20] if isinstance(s, dict)
    ]
    reply = str(plan.get("report_zh") or "") or str(result.get("echo_zh") or "")
    _replay_record_safe(principal, payload.conv_id, payload.replay_key, "chat", {
        "user": text[:500], "route": str(result.get("route") or ""),
        "reply": reply[:1500], "steps": steps,
    }, cfg)


                                                                                      
 
                                                         
                                                    
                                                
                                            
_UTT_IDEM_TTL_SECONDS = 3600.0                           
_UTT_IDEM_RUNNING_TTL_SECONDS = 24 * 3600.0                                  
                                                                    
                                                                         
_UTT_IDEM_MAX_ENTRIES = 256                         
_UTT_IDEM_WAIT_TIMEOUT = 180.0                                      
_UTT_IDEM_LOCK = threading.Lock()
_UTT_IDEM: "OrderedDict[tuple[str, str], dict]" = OrderedDict()


def _utterance_request_fp(text: str, payload: Any, provider: str,
                          mock_llm: bool, use_llm: bool) -> str:
    """占用条目的请求指纹：同 req_id 但**内容不同**的请求不得
    共享幂等槽（此前只看 req_id——两个不同 utterance/config 复用同号会错等/错收响应；
    撞指纹 → 409 如实说明，不当成同一次重发）。

    2026-08-10 二轮评审补全：指纹材料加入**现场字段**（has_results/result_total/query/
    current_filters/sources/base_url）——断流重发会把它们原样带上（同指纹合法复用），
    而「同号不同现场」必是另一次请求（撞指纹 409）。api_key 只以其哈希入料——
    指纹材料虽不落盘，也绝不给任何调试/日志路径留原文的机会。"""
    api_key = str(getattr(payload, "api_key", None) or "")
    material = json.dumps({
        "text": text,
                                                    
                                                   
        "provider": provider, "mock_llm": bool(mock_llm), "use_llm": bool(use_llm),
        "model": str(getattr(payload, "model", "") or ""),
        "agent": bool(getattr(payload, "agent", False)),
        "has_results": bool(getattr(payload, "has_results", False)),
        "result_total": int(getattr(payload, "result_total", 0) or 0),
        "query": str(getattr(payload, "query", "") or ""),
        "current_filters": getattr(payload, "current_filters", None),
        "sources": getattr(payload, "sources", None),
                                            
                                                 
        "artifact_context": str(getattr(payload, "artifact_context", "") or ""),
                                                    
                                                   
        "suggested_recipe": str(getattr(payload, "suggested_recipe", "") or ""),
        "base_url": str(getattr(payload, "base_url", "") or ""),
                                   
                                             
        "top_k": getattr(payload, "top_k", None),
        "rerank": str(getattr(payload, "rerank", "") or ""),
        "recall": str(getattr(payload, "recall", "") or ""),
        "strategy": str(getattr(payload, "strategy", "") or ""),
        "facet_filters": getattr(payload, "facet_filters", None),
        "suppressed_constraints": getattr(payload, "suppressed_constraints", None),
        "lenient_dims": getattr(payload, "lenient_dims", None),
        "date_from": str(getattr(payload, "date_from", "") or ""),
        "date_to": str(getattr(payload, "date_to", "") or ""),
        "polish": bool(getattr(payload, "polish", True)),
        "ask_answer": getattr(payload, "ask_answer", None),
        "conv_id": str(getattr(payload, "conv_id", "") or ""),
        "replay_key_fp": hashlib.sha256(str(getattr(payload, "replay_key", "") or "").encode("utf-8")).hexdigest(),
        "api_key_fp": hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16],
        "project_id": str(getattr(payload, "project_id", "") or ""),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _sanitize_req_id(raw: str | None) -> str | None:
    """兼容既有入口名；提交与恢复的规范化真源在 conversation_runs。"""
    from .conversation_runs import normalize_request_id
    return normalize_request_id(raw)


def _prune_utterance_idem(now: float) -> None:
    """TTL + 上限淘汰。**只在持 `_UTT_IDEM_LOCK` 时调用**。
    TTL 分态：done 1h 淘汰（缓存体到期）；running 只在 24h 泄漏兜底线上
    淘汰（owner 崩掉占坑）——正常长链执行/锁排队远低于 24h，绝不按 1h 误杀在途 owner。
    上限淘汰只动 done 条目：FIFO 淘汰 running owner =
    同号请求随后成为新 owner、同一句话的写工具真实执行两遍——那才是重复写窗口。"""
    stale = [rid for rid, e in _UTT_IDEM.items()
             if (e["state"] == "done" and now - e["ts"] > _UTT_IDEM_TTL_SECONDS)
             or (e["state"] != "done" and now - e["ts"] > _UTT_IDEM_RUNNING_TTL_SECONDS)]
    for rid in stale:
        del _UTT_IDEM[rid]
    while len(_UTT_IDEM) > _UTT_IDEM_MAX_ENTRIES:
        done_oldest = next((rid for rid, e in _UTT_IDEM.items() if e["state"] == "done"), None)
        if done_oldest is None:
            break                                                     
        del _UTT_IDEM[done_oldest]


def _utterance_idem_claim(req_id: str, fp: str = "", *, principal: str = "anonymous") -> tuple[dict, bool]:
    """按账户占用 req_id：没见过 → 注册 running 条目并返回 (entry, True=owner)；
    已见过 → 返回 (entry, False=非 owner，调用方等/取缓存体)。
    满员且全 running → 503 拒绝新占用（宁可拒新、绝不淘汰在途 owner）。
    同号但请求指纹不同（fp 非空且不匹配）→ 409（评审：那不是重发，是撞号）。"""
    now = time.monotonic()
    key = (str(principal or "anonymous"), req_id)
    with _UTT_IDEM_LOCK:
        _prune_utterance_idem(now)                             
        entry = _UTT_IDEM.get(key)
        if entry is not None:
            if fp and entry.get("fp") and entry["fp"] != fp:
                raise HTTPException(
                    status_code=409,
                    detail="这个 req_id 已对应另一句话（内容指纹不同）。"
                           "如果确实要重发，请换一个新的 req_id。")
            return entry, False
        if len(_UTT_IDEM) >= _UTT_IDEM_MAX_ENTRIES:
            done_oldest = next((rid for rid, e in _UTT_IDEM.items() if e["state"] == "done"), None)
            if done_oldest is not None:
                del _UTT_IDEM[done_oldest]
            else:
                raise HTTPException(status_code=503, detail="在途请求过多，请稍后重试。")
        entry = {"state": "running", "body": None, "event": threading.Event(),
                 "ts": now, "fp": fp}
        _UTT_IDEM[key] = entry
        return entry, True


                                                           
                                        
from .conversation_runs import RUN_INTERNAL_ERROR_DETAIL
_UTTERANCE_INTERNAL_ERROR_DETAIL = RUN_INTERNAL_ERROR_DETAIL


def _utterance_idem_store(entry: dict, body: dict) -> None:
    """owner 收尾：存最终响应体并唤醒等待者。锁只罩字典写，event.set 放锁外（等待方不持本锁）。"""
    with _UTT_IDEM_LOCK:
        entry["body"] = body
        entry["state"] = "done"
    entry["event"].set()


def _utterance_idem_abort(entry: dict | None, exc: BaseException) -> None:
    """执行前失败：释放本人占用，并让已附着的等待方收到同一个公开错误。"""
    if entry is None:
        return
    error = exc if isinstance(exc, HTTPException) else HTTPException(500, _UTTERANCE_INTERNAL_ERROR_DETAIL)
    with _UTT_IDEM_LOCK:
        for key, current in list(_UTT_IDEM.items()):
            if current is entry:
                del _UTT_IDEM[key]
                break
        entry.update(state="done", error=error)
    entry["event"].set()


def _utterance_idem_wait_response(entry: dict, *, stream: bool) -> Response:
    """非 owner 分支：done → 直接回缓存体（HTTP 200 原样）；running → 等 owner 收尾，
    超时仍无体 → 503（客户端稍后重试）。**绝不持 ENV_LOCK / 不载 env override**——
    非 owner 不执行路由，只是等一个已经跑起来的结果。"""
    body = entry["body"]
    if entry["state"] != "done" or body is None:
        entry["event"].wait(timeout=_UTT_IDEM_WAIT_TIMEOUT)
        body = entry["body"]
    if entry.get("error") is not None:
        raise entry["error"]
    if body is None:
        raise HTTPException(status_code=503, detail="同一句话还在处理中，请稍等。")
    if stream:
                                                                     
                                                            
        if body.get("ok"):
            frames = [_sse_line("final", body)]
        else:
            frames = [_sse_line("error", {"detail": body.get("detail") or _UTTERANCE_INTERNAL_ERROR_DETAIL})]
        return StreamingResponse(iter(frames), media_type="text/event-stream")
    return _json_utf8(body)


def _utterance_response_body(result: dict[str, Any]) -> dict[str, Any]:
    """`/api/utterance` 的响应体（2026-08-03 抽出）：非流式响应与流式 final 事件
    **共用这一个真源**——SSE 契约要求 final 体与非流式逐位同形，抄两份必漂移。"""
    from ..agent import agent_exec as _agent_exec
    body = {
        "ok": True,
        "route": result["route"],
        "query": result["query"],
        "plan": result["plan"],
        "echo_zh": result["echo_zh"],
        "retrieval": result["retrieval"],
        "via": result["via"],
                                            
                                                 
        "needs_agent": bool(result.get("needs_agent")),
                                                                 
                                                                
        "suggestions": list(result.get("suggestions") or []),
                                                                 
        "agent": {"available": _agent_exec.agent_available(), "used": result["via"] == "agent"},
                                                                 
                                                             
        "result_payload": result.get("result_payload"),
                                                                 
                                    
        "preliminary_final": bool(result.get("preliminary_final")),
    }
                                                     
                                                                                 
                                                                     
                         
    if result.get("retrieval") is None and result.get("retrieval_note"):
        body["retrieval_note"] = result["retrieval_note"]
                                                             
                                                     
    if result.get("result_batches"):
        body["result_batches"] = result["result_batches"]
        body["active_batch"] = result.get("active_batch")
                                           
                                                  
    if result.get("trace_turn_id"):
        body["trace_turn_id"] = result["trace_turn_id"]
                                                          
                                 
    if result.get("policy_id") is not None:
        body["policy_id"] = result["policy_id"]
        body["policy_id_str"] = result.get("policy_id_str") or _policy_id_string(result["policy_id"])
    if result.get("experiment") is not None:
        body["experiment"] = result["experiment"]
                                                      
                                                  
    if result.get("recipe_note"):
        body["recipe_note"] = result["recipe_note"]
                                                             
                                        
    if result.get("ask_relax") is not None:
        body["ask_relax"] = result["ask_relax"]
    return body


def _inject_utterance_policy_id(result: dict[str, Any], payload, search_params: dict | None, model) -> None:
    """route=="search" 时把 policy_id 塞进 result（additive；流式/非流式共用
    注入点，透传在 `_utterance_response_body`）。组装失败降级 None（键不出现，不掀翻路由）。"""
    if not isinstance(result, dict) or result.get("route") != "search":
        return
    sp = search_params or {}
    policy_id = _policy_id_or_none(
        sources=payload.sources,
        strategy=sp.get("strategy", ""),
        rerank=sp.get("rerank", ""),
        recall=sp.get("recall", ""),
        model=model,
    )
    result.update(_policy_response_fields(policy_id))
    result["experiment"] = _experiment_response(payload)


def _sse_line(event: str, data: Any) -> str:
    """一帧 SSE：`data: {json}\\n\\n`（ensure_ascii=False：中文原样发，客户端按 utf-8 解）。"""
    return "data: " + json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n\n"


def _route_turn_with_project(project_id: str, text: str, **kwargs):
    from ..agent import turn
    from ..agent.learning_scope import bind_project_scope
    with bind_project_scope(project_id):
        return turn.route_turn(text, **kwargs)


_CONVERSATION_RUNTIME_LOCK = threading.Lock()
_CONVERSATION_RUNTIME = None
_CONVERSATION_LIFESPANS = set()
_CONVERSATION_LIFESPAN_RUNTIME = None


def _make_conversation_runtime(store):
    from .conversation_runs import ConversationRuntime
    return ConversationRuntime(store, workers=max(1, min(_env_int("BIODATA_RUN_WORKERS", 2), 8)),
        max_inflight=max(1, _env_int("BIODATA_RUN_MAX_INFLIGHT", 16)),
        max_per_owner=max(1, _env_int("BIODATA_RUN_MAX_PER_USER", 1)),
        max_seconds=max(30, _env_int("BIODATA_RUN_TIMEOUT_SECONDS", 900)))


def _enter_conversation_lifespan(token):
    from .conversation_store import ConversationStore
    global _CONVERSATION_RUNTIME, _CONVERSATION_LIFESPAN_RUNTIME
    with _CONVERSATION_RUNTIME_LOCK:
        _CONVERSATION_LIFESPANS.add(token)
        if _CONVERSATION_RUNTIME is None:
            store = ConversationStore(PROJECT_ROOT)
            if store.has_pending_cleanup():
                _CONVERSATION_RUNTIME = _make_conversation_runtime(store)
                _CONVERSATION_LIFESPAN_RUNTIME = _CONVERSATION_RUNTIME


def _leave_conversation_lifespan(token):
    global _CONVERSATION_RUNTIME, _CONVERSATION_LIFESPAN_RUNTIME
    runtime = None
    with _CONVERSATION_RUNTIME_LOCK:
        if token not in _CONVERSATION_LIFESPANS:
            return
        _CONVERSATION_LIFESPANS.remove(token)
        if not _CONVERSATION_LIFESPANS:
            runtime, _CONVERSATION_LIFESPAN_RUNTIME = _CONVERSATION_LIFESPAN_RUNTIME, None
            if _CONVERSATION_RUNTIME is runtime:
                _CONVERSATION_RUNTIME = None
    if runtime is not None:
        runtime.close()


def _get_conversation_runtime():
    from .conversation_store import ConversationStore
    global _CONVERSATION_RUNTIME, _CONVERSATION_LIFESPAN_RUNTIME
    with _CONVERSATION_RUNTIME_LOCK:
        if _CONVERSATION_RUNTIME is None:
            _CONVERSATION_RUNTIME = _make_conversation_runtime(ConversationStore(PROJECT_ROOT))
            if _CONVERSATION_LIFESPANS:
                _CONVERSATION_LIFESPAN_RUNTIME = _CONVERSATION_RUNTIME
        return _CONVERSATION_RUNTIME


def _execute_utterance(text, payload, cfg, principal, project_id, search_params, ask_answer, on_event=None):
    from .conversation_service import execute_turn
    body = execute_turn(text, cfg=cfg, principal=principal, project_id=project_id,
        state={"has_results": bool(payload.has_results), "result_total": int(payload.result_total or 0),
               "query": payload.query or "", "current_filters": payload.current_filters,
               "sources": payload.sources, "search_params": search_params},
        use_agent=bool(payload.agent), artifact_context=str(payload.artifact_context or ""),
        suggested_recipe=str(payload.suggested_recipe or ""), ask_answer=ask_answer,
        on_event=on_event, route=_route_turn_with_project)
    _inject_utterance_policy_id(body, payload, search_params, cfg.model)
    try:
        _replay_record_utterance(principal, payload, text, body, cfg)
    except Exception:
        logger.exception("复盘落痕挂钩失败（不影响主路径）")
    return _utterance_response_body(body)


def _utterance_event_stream(
    text, payload, *, provider, requested_base_url, mock_llm, use_llm,
    idem_entry=None, principal="", project_id="", search_params=None, ask_answer=None,
    runtime=None, run=None,
):
    """Legacy SSE rendering over the same bounded durable execution channel."""
    from .conversation_runs import RunError, TERMINAL
    import uuid as _stream_uuid
    runtime = runtime or _get_conversation_runtime()
    if run is None:
        def prepare():
            cfg = _materialize_request_llm_config(provider, use_llm=use_llm, mock_llm=mock_llm,
                api_key=payload.api_key, base_url=requested_base_url, model=payload.model)
            return lambda emit: _execute_utterance(text, payload, cfg, principal, project_id,
                                                   search_params, ask_answer, emit)
        try:
            run = runtime.submit(principal or "anonymous", _sanitize_req_id(payload.req_id) or _stream_uuid.uuid4().hex,
                                 _utterance_request_fp(text, payload, provider, mock_llm, use_llm), prepare=prepare)
        except BaseException as exc:
            _utterance_idem_abort(idem_entry, exc)
            raise
    after = 0
    try:
        with runtime.waiting_transport(principal or "anonymous"):
            while True:
                state = runtime.store.load(principal or "anonymous", run["id"])
                batch = runtime.store.events(principal or "anonymous", run["id"], after)
                for event in batch:
                    after = event["seq"]
                    yield _sse_line(event["event"], event["data"])
                if state["status"] in TERMINAL and len(batch) < 100:
                    body = state["result"] or {"ok": False, "detail": _UTTERANCE_INTERNAL_ERROR_DETAIL}
                    if idem_entry is not None:
                        _utterance_idem_store(idem_entry, body)
                    yield _sse_line("final" if body.get("ok") else "error",
                                   body if body.get("ok") else {"detail": body.get("detail", _UTTERANCE_INTERNAL_ERROR_DETAIL)})
                    break
                time.sleep(.05)
    except RunError as exc:
        yield _sse_line("error", {"detail": exc.message})


def _shared_principal(request):
    principal = _utterance_principal(request)
    return "local" if principal == "anonymous" else principal


def _get_conversation_service():
    from .conversation_service import ConversationService
    return ConversationService(PROJECT_ROOT, runtime=_get_conversation_runtime())


def _shared_utterance(payload, request):
    from .conversation_api import invoke
    from .llm_admission import materialize
    owner = _shared_principal(request)
    explicit = bool(payload.model_fields_set & {"provider", "use_llm", "mock_llm", "api_key", "base_url", "model"})
    def configure():
        if not explicit:
            return materialize(config_root=CONFIG_ROOT, loader=load_llm_config, lock=ENV_LOCK)
        provider, base, mock, enabled = _resolve_llm_request(payload.provider,
            use_llm=payload.use_llm, mock_llm=payload.mock_llm, base_url=payload.base_url)
        return _materialize_request_llm_config(provider, use_llm=enabled, mock_llm=mock,
            api_key=payload.api_key, base_url=base, model=payload.model)
    def quota(cfg):
        _gate_llm_quota(request, cfg=cfg, provider=cfg.provider, use_llm=True,
            mock_llm=cfg.mock_llm, api_key=payload.api_key)
    def operation():
        service = _get_conversation_service()
        patch = dict(payload.state_patch or {})
        controls = {key: getattr(payload, key) for key in (
            'top_k', 'rerank', 'rerank_top_n', 'recall', 'strategy', 'facet_filters',
            'suppressed_constraints', 'lenient_dims', 'date_from', 'date_to', 'polish',
            'auto_parse_sources', 'auto_allow_llm') if key in payload.model_fields_set}
        if controls:
            existing = dict(patch.get('search_params') or {})
            if any(key in existing and existing[key] != value for key, value in controls.items()):
                from .conversation_runs import RunError
                raise RunError('conflicting_search_parameter', '顶层参数与 state_patch 的同名检索参数不一致。')
            patch['search_params'] = {**existing, **controls}
        if 'sources' in payload.model_fields_set:
            patch['sources'] = payload.sources
        run = service.chat(owner, payload.utterance, request_id=payload.req_id,
            conv_id=payload.conv_id, expected_revision=payload.expected_revision,
            project_id=payload.project_id, state_patch=patch or None, ask_choice=payload.ask_choice,
            use_agent=payload.agent, configure=configure, quota=quota,
            config_fingerprint=hashlib.sha256(json.dumps({key: getattr(payload, key)
                for key in ("provider", "use_llm", "mock_llm", "api_key", "base_url", "model")
                if key in payload.model_fields_set}, sort_keys=True).encode()).hexdigest() if explicit else "")
        if payload.background:
            return {"ok": True, "async": True, "shared": True, "run": run}
        return service.runtime.wait_result(owner, run["id"])
    if payload.stream:
        from .conversation_runs import TERMINAL
        captured = {}
        def submit_only():
            saved = payload.background
            payload.background = True
            try:
                captured.update(operation())
                return captured
            finally:
                payload.background = saved
        response = invoke(submit_only)
        runtime, run = _get_conversation_runtime(), captured["run"]
        def events():
            after = 0
            with runtime.waiting_transport(owner):
                while True:
                    state = runtime.store.load(owner, run["id"])
                    batch = runtime.store.events(owner, run["id"], after)
                    for event in batch:
                        after = event["seq"]
                        yield _sse_line(event["event"], event["data"])
                    if state["status"] in TERMINAL and len(batch) < 100:
                        yield _sse_line("final", state["result"])
                        break
                    time.sleep(.05)
        return StreamingResponse(events(), media_type="text/event-stream")
    response = invoke(operation)
    if payload.background:
        response.status_code = 202
    return response


@app.post("/api/utterance")
def api_utterance(payload: UtteranceRequest, request: Request) -> Response:
    """统一对话窗口的**后端路由**（turn pipeline）。**路由是真，执行也是真——
    AI 执行开启时，EXEC 动词经 agent 图（langgraph）在本端点内真实执行**
    （search_online/sync_updates 会写外部库 upload_*，记账 + 回收站可撤 +
    2026-08-09 起写入汇强制流水账）；「只出计划不执行」的是 `/api/action/plan`。

    管线（2026-08-03 用户定稿，唯一设计，无并行短路）——`turn.route_turn` 单一真源：
    **规则匹配（一切指令都过；零命中/弃权 ≠ 无效）→ LLM 分流 →
    检索指令（search，带 effective_query）/ 工具调用（tool，EXEC plan）/ none（如实回音）**。
    LLM 缺席/失败时规则兜底（动作词 → tool 规则档；search_shaped → search；其余 → none），
    绝不把零命中/带执行标记的句子 fail-open 成检索。

    LLM 配置链与 `/api/recommend` 同径：ENV_LOCK + `_temporary_env` 请求级覆盖
    （请求级 key 隔离、endpoint 校验、server 配置对照），锁内只载配置，
    锁外才跑规则匹配与 LLM 分流（不把网络 I/O 关在 ENV_LOCK 里）。
    执行回执与进度落入账户隔离的任务库；请求级密钥只保留在执行闭包内。
    浏览器侧动作仍由调用方消费回执后派发。

    `stream:true`改发 `text/event-stream`：preliminary? →
    tool_start* → step* → final（2026-08-16）
    final 体与本响应逐位同形；细节见 `_utterance_event_stream`。

    `req_id` 按账户持久去重：同号同请求共享一次执行与费用检查，同号异请求返回
    冲突。JSON、SSE 与 `background:true` 共享 ConversationRuntime 的有界队列；
    后台模式立即返回任务编号，后续通过 `/api/runs` 查询、恢复进度或申请取消。
    进程重启后未完成任务标为 interrupted，不自动重复可能已经生效的操作。
    """
    _require_same_origin(request)
    if payload.shared:
        return _shared_utterance(payload, request)
    if payload.query:
        _validate_or_400(validate_query, payload.query)
    from ..agent import action_plan as _ap
    from ..agent import turn

    principal = _utterance_principal(request)                               
    project_id = ""

    try:
        text = _ap.normalize_utterance(payload.utterance)
    except _ap.ActionPlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc),
                            headers={"X-Error-Code": exc.code}) from None

    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)

                                             
                                                     
                                                   
    sp_rerank = "llm" if str(payload.rerank or "").strip().lower() == "llm" else "off"
    _rec = str(payload.recall or "").strip().lower()
    sp_recall = _rec if _rec in ("dense", "cross_encoder", "vector") else "off"
    sp_date_from = _require_iso_date(payload.date_from, name="date_from")
    sp_date_to = _require_iso_date(payload.date_to, name="date_to")
    _validate_or_400(validate_date_window, sp_date_from, sp_date_to)
    search_params = {
        "top_k": payload.top_k,
        "rerank": sp_rerank,
        "recall": sp_recall,
        "strategy": "auto" if str(payload.strategy or "").strip().lower() == "auto" else "fixed",
        "facet_filters": _sanitize_facet_filters(payload.facet_filters),
        "suppressed_constraints": _sanitize_suppressed(payload.suppressed_constraints),
        "lenient_dims": _sanitize_lenient_dims(payload.lenient_dims),
        "date_from": sp_date_from,
        "date_to": sp_date_to,
        "polish": bool(payload.polish),
    }
    search_params.update({key: getattr(payload, key) for key in (
        "auto_parse_sources", "auto_allow_llm", "rerank_top_n") if key in payload.model_fields_set})
                                                      
    ask_answer = _sanitize_ask_answer(payload.ask_answer)
    if payload.project_id is not None:
        from .workspace_store import WorkspaceError, WorkspaceStore
        from ..secret_patterns import redact
        try:
            project_id = WorkspaceStore(PROJECT_ROOT).resolve_conversation(
                principal, str(getattr(payload, "conv_id", "") or ""), payload.project_id,
                title=redact(text)[:180])
        except WorkspaceError as exc:
            status = 409 if exc.code == "conversation_project_changed" else 404
            raise HTTPException(status_code=status, detail=exc.message,
                                headers={"X-Error-Code": exc.code}) from exc

                                                      
                                                        
    from .conversation_runs import RunError
    import uuid as _run_uuid
    request_id = _sanitize_req_id(payload.req_id) or _run_uuid.uuid4().hex
    def prepare():
        cfg = _materialize_request_llm_config(
            provider, use_llm=use_llm, mock_llm=mock_llm,
            api_key=payload.api_key, base_url=requested_base_url, model=payload.model)
        _gate_llm_quota(request, cfg=cfg, provider=provider, use_llm=use_llm, mock_llm=mock_llm,
                        api_key=payload.api_key, requested_base_url=requested_base_url)
        return lambda emit: _execute_utterance(
            text, payload, cfg, principal, project_id, search_params, ask_answer, emit)
    runtime = _get_conversation_runtime()
    try:
        run = runtime.submit(
            principal, request_id, _utterance_request_fp(text, payload, provider, mock_llm, use_llm),
            prepare=prepare, conversation_id=str(getattr(payload, "conv_id", "") or ""),
            project_id=project_id, request={"text": text, "query": payload.query or "",
                "search_params": search_params, "sources": payload.sources})
        if payload.background:
            return _json_utf8({"ok": True, "async": True, "run": run}, status_code=202)
        if payload.stream:
            return StreamingResponse(_utterance_event_stream(
                text, payload, provider=provider, requested_base_url=requested_base_url,
                mock_llm=mock_llm, use_llm=use_llm, principal=principal, project_id=project_id,
                search_params=search_params, ask_answer=ask_answer, runtime=runtime, run=run),
                media_type="text/event-stream")
        return _json_utf8(runtime.wait_result(principal, run["id"]))
    except RunError as exc:
        status = 409 if exc.code == "run_request_conflict" else 503 if exc.code in {
            "run_capacity", "run_wait_timeout", "run_transport_capacity"} else 400
        raise HTTPException(status_code=status, detail=exc.message,
                            headers={"X-Error-Code": exc.code}) from exc


                                                                                            
                                                        
                                                                  
                                                            
                                                            
                                    

def _replay_cfg_or_none(payload) -> "LLMConfig | None":
    """复盘端点的请求级 LLM 物化（与 /api/utterance 同径）。"""
    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)
    return _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model)


def _replay_principal(request, conv_id):
    principal = _utterance_principal(request)
    if principal == 'anonymous':
        from .conversation_store import ConversationStore
        if ConversationStore(PROJECT_ROOT).exists('local', conv_id, include_deleted=True):
            return 'local'
    return principal


@app.get("/api/replay/status")
def api_replay_status(request: Request, conv_id: str = "", key: str = "") -> JSONResponse:
    """角标轮询（轻量）：{ok, n_turns, has_digest, parse_failed, unseen, unseen_knowledge,
    drawing, skip_reason, lint_count}。conv 从未落痕 → 空态（has_digest=false），不是 404。"""
    _require_same_origin(request)
    from . import replay_service
    out = replay_service.status(PROJECT_ROOT, _replay_principal(request, conv_id), conv_id[:80], key[:64])
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.get("/api/replay/points")
def api_replay_points(request: Request, conv_id: str = "", key: str = "") -> JSONResponse:
    """复盘窗口全量载荷：{ok, digest, threads, snapshots, unseen, unseen_knowledge,
    n_turns, drawing, skip_reason}。"""
    _require_same_origin(request)
    from . import replay_service
    out = replay_service.points(PROJECT_ROOT, _replay_principal(request, conv_id), conv_id[:80], key[:64])
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.get("/api/replay/snapshots")
def api_replay_snapshots(request: Request, conv_id: str = "", key: str = "") -> JSONResponse:
    """标注快照清单（pill 找回用）：{ok, snapshots}。"""
    _require_same_origin(request)
    from . import replay_service
    out = replay_service.points(PROJECT_ROOT, _replay_principal(request, conv_id), conv_id[:80], key[:64])
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8({"ok": True, "snapshots": out["snapshots"]})


@app.post("/api/replay/digest")
def api_replay_digest(payload: ReplayDigestRequest, request: Request) -> JSONResponse:
    """手动「重新整理」（用户显式触发，同步执行；试用通道也放行但计日配额）。"""
    _require_same_origin(request)
    from . import replay_service
    _gate_llm_quota(
        request, cfg=None, provider=payload.provider, use_llm=payload.use_llm,
        mock_llm=payload.mock_llm, api_key=payload.api_key)
    out = replay_service.manual_digest(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        _replay_cfg_or_none(payload))
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.post("/api/replay/annotate")
def api_replay_annotate(payload: ReplayAnnotateRequest, request: Request) -> JSONResponse:
    """图上标注一条（只记录，不发 LLM——发送权只属于 composer）。"""
    _require_same_origin(request)
    from . import replay_service
    ok = replay_service.annotate(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        payload.pid, payload.text)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8({"ok": True})


@app.post("/api/replay/snapshot")
def api_replay_snapshot(payload: _ReplayBase, request: Request) -> JSONResponse:
    """「添加到对话」：打包全部用户标注为快照并清空图上标注 → {ok, snap_id?, count, items, text}。"""
    _require_same_origin(request)
    from . import replay_service
    snap = replay_service.take_snapshot(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key)
    if snap is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8({"ok": True, **snap})


@app.post("/api/replay/snapshot/{snap_id}/annotate")
def api_replay_snapshot_annotate(snap_id: str, payload: ReplayAnnotateRequest,
                                 request: Request) -> JSONResponse:
    """往已打包的快照里追加标注（pill 找回后的继续编辑）。"""
    _require_same_origin(request)
    from . import replay_service
    snap = replay_service.annotate_snapshot(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        snap_id[:80], payload.pid, payload.text)
    if not snap:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8({"ok": True, "snapshot": snap})


@app.post("/api/replay/seen")
def api_replay_seen(payload: ReplaySeenRequest, request: Request) -> JSONResponse:
    """开窗即已读：核销用户**实际看到的版本**（observed_version/observed_pids，additive）。
    返回 {ok, marked, digest_version, stale}——stale=true 表示窗口版本已过期，前端应刷新后重发。"""
    _require_same_origin(request)
    from . import replay_service
    out = replay_service.mark_seen(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        observed_version=payload.observed_version, observed_pids=payload.observed_pids)
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.post("/api/replay/chat")
def api_replay_chat(payload: ReplayChatRequest, request: Request) -> JSONResponse:
    """sidechat 一轮：全上下文（digest + 浓缩痕迹 + 标注 + 复盘史）→ {ok, reply}。"""
    _require_same_origin(request)
    from . import replay_service
    _gate_llm_quota(
        request, cfg=None, provider=payload.provider, use_llm=payload.use_llm,
        mock_llm=payload.mock_llm, api_key=payload.api_key)
    out = replay_service.chat(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        payload.text, _replay_cfg_or_none(payload))
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.post("/api/replay/backfill")
def api_replay_backfill(payload: ReplayBackfillRequest, request: Request) -> JSONResponse:
    """本机历史补建（用户显式点按钮才上传，对齐 /api/dream 先例；仅服务端无记录时生效）。
    确认本次会真整理后、落痕前过 LLM 配额闸（复用 /api/replay/digest 同款闸；被拒 → 429 零写入）。"""
    _require_same_origin(request)
    from . import replay_service
    out = replay_service.backfill(
        PROJECT_ROOT, _replay_principal(request, payload.conv_id), payload.conv_id, payload.key,
        payload.turns, _replay_cfg_or_none(payload),
        quota_gate=lambda: _gate_llm_quota(
            request, cfg=None, provider=payload.provider, use_llm=payload.use_llm,
            mock_llm=payload.mock_llm, api_key=payload.api_key))
    if out is None:
        raise HTTPException(status_code=404, detail="not found")
    return _json_utf8(out)


@app.post("/api/agent/search-rescue")
def api_agent_search_rescue(payload: SearchRescueRequest, request: Request) -> JSONResponse:
    """检索救回（2026-08-16 检索工具化 Phase 1）：零命中查询的「换词重检」入口。

    agent 在 rescue 收敛面（工具面只有 search.rerun / none，validate 与裁决层各有一道
    机械闸兜底）下跑一次图；search.rerun 的机械闸裁定采纳与否：
    采纳 → `adopted=true` + /api/recommend 同形 `payload`（调用方整屏替换结果）；
    结果集同集 / 筛选条件丢失 / LLM 选 none 放弃 → `adopted=false` + 诚实 reason，
    当前结果保持不变。设计决定：**改空（新条件 0 命中）也采纳**——空结果集
    照常上屏，是条件变更重检的诚实答案，不再否决后「保持不变」。
    rescue2：原检索投影（rule_match_summary 同径）注入提示词——未收录词
    允许丢弃/映射，已收录条件必须全保留；采纳时响应带 `dropped_terms`（机械比对）且
    `report_zh` 用确定性披露句点名丢弃词/改写词/命中数，前端零改动上屏。
    **绝不 500**：agent 缺席 / LLM 未配 / 图内任何异常都回 200 fail-open（reason 如实），
    本端点不产交付物、不落盘（工具内每次重检照常记联网账本——只读检索记 0 条口径）。"""
    _require_same_origin(request)
    from ..agent import action_plan as _ap
    from ..agent import agent_exec as _ax

                                                              
                                    
    rescue_facets = _sanitize_facet_filters(payload.facet_filters)
    rescue_suppressed = _sanitize_suppressed(payload.suppressed_constraints)
    rescue_lenient = _sanitize_lenient_dims(payload.lenient_dims)
    rescue_date_from = _require_iso_date(payload.date_from, name="date_from")
    rescue_date_to = _require_iso_date(payload.date_to, name="date_to")
    _validate_or_400(validate_date_window, rescue_date_from, rescue_date_to)
    rescue_search_params = {
        "facet_filters": rescue_facets,
        "suppressed_constraints": rescue_suppressed,
        "lenient_dims": rescue_lenient,
        "date_from": rescue_date_from,
        "date_to": rescue_date_to,
    }

    def _fail_open(reason: str, report_zh: str, *, available: bool) -> JSONResponse:
        return _json_utf8({
            "ok": True, "attempted": False, "reason": reason, "adopted": False,
            "query": payload.query, "rewrite": "", "n_before": None, "n_after": None,
            "payload": None, "report_zh": report_zh, "trace": [], "dropped_terms": [],
            "agent": {"available": available, "used": False},
        })

    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)
                                                                   
                                           
    cfg = _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model)

    if not _ax.agent_available():
        return _fail_open("agent_unavailable",
                          "没有重新检索：agent 扩展不可用，当前结果未变。",
                          available=False)
    llm_ok, why = _ap.should_use_llm(cfg)
    if not llm_ok:
        return _fail_open("llm_unavailable",
                          f"没有重新检索：大模型不可用（{why}），当前结果未变。",
                          available=True)
                                                          
    _gate_llm_quota(request, cfg=cfg, provider=provider, use_llm=use_llm,
                    mock_llm=mock_llm, api_key=payload.api_key)

                                 
                                                                     
                                                                   
    from ..agent import trace as _trace
    _principal = _utterance_principal(request)
    _rec = _trace.recorder_for_turn(PROJECT_ROOT, session_id=_principal or "anonymous")

    def _with_tid(body: dict) -> dict:
                                      
        if _rec.enabled:
            body["trace_turn_id"] = _rec.turn_id
        return body

    try:
                                                                
                                                   
                                                               
        try:
            from ..agent.turn import rule_match_summary as _rms
            retrieval_summary = _rms(
                payload.query, sources=payload.sources, search_params=rescue_search_params)
        except Exception:
            logger.exception("search-rescue 原检索投影失败（fail-open：按无投影继续）")
            retrieval_summary = None
        with _trace.bind_recorder(_rec):
            plan, trace = _ax.plan_with_agent_events(
                payload.query,
                has_results=False,
                result_total=int(payload.result_total or 0),
                config=cfg,
                retrieval=retrieval_summary,
                current_query=payload.query,
                current_filters=payload.current_filters,
                route_scope="rescue",
                search_sources=payload.sources,
                search_facet_filters=rescue_facets,
                search_suppressed_constraints=rescue_suppressed,
                search_lenient_dims=rescue_lenient,
                search_date_from=rescue_date_from,
                search_date_to=rescue_date_to,
                principal=_principal,
            )
    except Exception:
                                                      
        logger.exception("/api/agent/search-rescue 兜底触发：agent 图抛出异常")
        return _fail_open("agent_error",
                          "这次重新检索没有跑成（内部错误已记录），当前结果未变。",
                          available=True)

    rerun_step = next(
        (s for s in list(plan.get("steps") or [])
         if str(s.get("verb") or "") == "search.rerun"),
        None,
    )
    report_zh = str(plan.get("report_zh") or "")
    if rerun_step is None or not rerun_step.get("ok"):
                                                                  
        if rerun_step is None:
            return _json_utf8(_with_tid({
                "ok": True, "attempted": True, "reason": "no_rewrite", "adopted": False,
                "query": payload.query, "rewrite": "", "n_before": None, "n_after": None,
                "payload": None, "dropped_terms": [],
                "report_zh": report_zh or "没有更合适的换法可试，当前结果未变。",
                "trace": trace, "agent": {"available": True, "used": True},
            }))
        return _json_utf8(_with_tid({
            "ok": True, "attempted": True, "reason": "agent_error", "adopted": False,
            "query": payload.query,
            "rewrite": str((rerun_step.get("slots") or {}).get("query") or ""),
            "n_before": None, "n_after": None, "payload": None, "dropped_terms": [],
            "report_zh": report_zh or (
                "重新检索没有跑成：" + str(rerun_step.get("error") or "未知原因")),
            "trace": trace, "agent": {"available": True, "used": True},
        }))
    r = rerun_step.get("result") or {}
    dropped = [str(t) for t in (r.get("dropped_terms") or []) if str(t).strip()]
    return _json_utf8(_with_tid({
        "ok": True,
        "attempted": True,
        "reason": "adopted" if r.get("adopted") else str(r.get("reason") or "no_rewrite"),
        "adopted": bool(r.get("adopted")),
        "query": payload.query,
        "rewrite": str(r.get("query") or ""),
        "n_before": r.get("n_before"),
        "n_after": r.get("n_after"),
        "payload": r.get("payload"),
                                                            
                                                                           
        "report_zh": str(r.get("disclosure_zh") or "") or report_zh,
        "dropped_terms": dropped,
        "trace": trace,
        "agent": {"available": True, "used": True},
    }))


@app.post("/api/act/summary")
def api_act_summary(payload: ActSummaryRequest, request: Request) -> JSONResponse:
    """执行结果的 LLM 中文总结（p10）。**只总结，不执行、不检索、不落盘。**

    done/gap/policy 事实行由调用方（前端 act_core，数字全部取自真实返回值）构造后上报，
    本端点不核实、也绝不新增事实——护栏写进 user prompt（`act_summary_llm._RULES_ZH`），
    ok=False 时总结绝不能说「已」。LLM 缺席/失败/空回 → fail-open：`summary_zh=None`，
    调用方原样保留自己的确定性事实句。

    LLM 配置链与 `/api/action/plan` 同径：ENV_LOCK + `_temporary_env` 请求级覆盖
    （请求级 key 隔离、endpoint 校验、server 配置对照），锁内只载 config，锁外才调 provider
    （不把网络 I/O 关在 ENV_LOCK 里）。

    `brief:true`走一句话模式（≤35 字），响应形状不变。
    """
    _require_same_origin(request)
    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)

    cfg = _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model)

                                                 
    _gate_llm_quota(request, cfg=cfg, provider=provider, use_llm=use_llm,
                    mock_llm=mock_llm, api_key=payload.api_key)

    facts = {
        "verb_zh": payload.verb_zh,
        "utterance": payload.utterance,
        "ok": bool(payload.ok),
        "done_lines": payload.done_lines,
        "gap_lines": payload.gap_lines,
        "policy_lines": payload.policy_lines,
    }
    if payload.brief:
                                    
                                                                       
                                                                   
                                             
        brief = act_summary_llm.summarize_brief_with_llm(facts, config=cfg)
        if brief:
            brief_status = "ok"
        else:
            gate_ok, gate_reason = act_summary_llm.should_use_llm(cfg)
            brief_status = "failed" if gate_ok else gate_reason
        return _json_utf8({
            "ok": True,
            "summary_zh": brief,
            "summary_source": "llm" if brief else None,
            "llm_status": brief_status,
            "llm_model": None,
        })
    result = act_summary_llm.summarize_action_with_llm(facts, config=cfg)
    return _json_utf8({
        "ok": True,
        "summary_zh": result["summary_zh"],
        "summary_source": result["summary_source"],
        "llm_status": result["llm_status"],
        "llm_model": result["llm_model"],
    })


@app.post("/api/search/reply")
def api_search_reply(payload: SearchReplyRequest, request: Request) -> JSONResponse:
    """检索回执的 LLM 中文改写（2026-08-30 p11）。**只解说，不检索、不执行、不落盘。**

    命中数/展示数/命中关键词/解析状态等事实由调用方（前端 board.js cbPushCurrent，数字全部
    取自真实检索响应）构造后上报，本端点不核实、也绝不新增事实——护栏写进 user prompt
    （`search_reply_llm._SEARCH_REPLY_RULES_ZH`），0 命中直说没找到、建议只许从 can_suggest
    白名单原样挑一条。LLM 缺席/失败/空回 → fail-open：`reply_zh=None`，调用方原样保留
    自己的确定性事实句（不挂「AI 总结」标，归因诚实）。

    LLM 配置链与 `/api/act/summary` 同径：ENV_LOCK + `_temporary_env` 请求级覆盖
    （请求级 key 隔离、endpoint 校验、server 配置对照），锁内只载 config，锁外才调 provider
    （不把网络 I/O 关在 ENV_LOCK 里）；T3 配额闸同口径。
    """
    _require_same_origin(request)
    provider, requested_base_url, mock_llm, use_llm = _resolve_llm_request(
        payload.provider, use_llm=payload.use_llm, mock_llm=payload.mock_llm,
        base_url=payload.base_url)

    cfg = _materialize_request_llm_config(
        provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model)

                                                 
    _gate_llm_quota(request, cfg=cfg, provider=provider, use_llm=use_llm,
                    mock_llm=mock_llm, api_key=payload.api_key)

    facts = {
        "utterance": payload.utterance,
        "query": payload.query,
        "note": payload.note,
        "total": int(payload.total),
        "shown": int(payload.shown),
        "hit_keywords": payload.hit_keywords,
        "resolution_status": payload.resolution_status,
        "has_relax": bool(payload.has_relax),
        "can_suggest": payload.can_suggest,
    }
    result = search_reply_llm.search_reply_with_llm(facts, config=cfg)
    return _json_utf8({
        "ok": True,
        "reply_zh": result["reply_zh"],
        "reply_source": result["reply_source"],
        "llm_status": result["llm_status"],
        "llm_model": result["llm_model"],
    })


                                       
                                                       
                                                         
_MAX_UPLOAD_BYTES = _MAX_RAW_BODY_BYTES
                                  
_MAX_UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _read_upload_bounded(file: UploadFile, *, max_bytes: int) -> bytes:
    """分块流式读取上传文件，累计超过 `max_bytes` 立即关闭并中止（413），绝不整读复制。

    体积闸：旧实现 `await file.read()` 先占满内存（或 multipart 临时文件先落满盘）才检查
    `len(raw_bytes)`；缺失/非数字/谎报 Content-Length 或 chunked 的请求会在拒绝前占用完整内存。
    本函数每次只读固定块并累计，上限一到立即关闭 `UploadFile`（释放 multipart 临时文件）。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_MAX_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            await file.close()
            raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_DETAIL)
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/upload")
async def api_upload(
    request: Request,
    file: UploadFile = File(...),
    source: str = Query(default=""),
) -> JSONResponse:
    """上传一个数据集 JSON → 落入**外部库目录** `database/external/`（**绝不入** `database/base/` 基准语料）→ 即时可检索。
    - 确定性安全：官方 54 题评测走 base-only，永不读外部库 → 上传不影响 767 基准。
    - 逐条打来源标签（每条自带 `source` > 表单 `source` > 文件包裹层 `source` > 默认「用户上传」），
      使其作为一个**可勾选的来源**出现在浏览/查询里，且计数口径正确（不被误判为 10x Genomics）。
    - 返回解析条数、归属来源、可读校验提示（缺 dataset_name / 物种非通用名），让格式错误当场可见、不被误读。"""
    _require_same_origin(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名。")

                                                                        
                                                                
                                                              
                                               
    declared = (request.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_DETAIL)

                                                                         
                                                                       
                                                                     
                                                                    
    try:
                                                                              
                                                                   
        safe_name = _new_upload_name(file.filename)
                                               
        raw_bytes = await _read_upload_bounded(file, max_bytes=_MAX_UPLOAD_BYTES)
                                                                     
                                                        
        result = await run_in_threadpool(
            ingest_dataset,
            raw_bytes=raw_bytes,
            safe_name=safe_name,
            project_root=PROJECT_ROOT,
            form_source=source,
        )
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=exc.hint,
                            headers={"X-Error-Code": exc.code}) from exc

    return _json_utf8(
        {
        "ok": True,
        "filename": result.filename,
        "saved_to": result.saved_to,
        "record_count": result.record_count,
        "sources": result.sources,              
        "warnings": result.warnings,                   
        "joined_knowledge_base": True,
        }
    )


                                                                                          
                                         
                                                                          
                                                                                
                                                                    
                                                              
_MAX_CURATE_PAYLOAD_BYTES = _MAX_UPLOAD_BYTES                                      


def _curate_payload_bytes(payload_json: str | None) -> bytes | None:
    """import 的 JSON 文本 → 字节；空 → None（由真源报 bad_param），超上限 → bad_param。"""
    if payload_json is None or not payload_json.strip():
        return None
    raw = payload_json.encode("utf-8")
    if len(raw) > _MAX_CURATE_PAYLOAD_BYTES:
        from ..corpus.corpus_curation import CurateError
        raise CurateError(
            "bad_param",
            f"导入的 JSON 文本超过上限（64 MB）；单细胞元数据 JSON 一般远小于此。",
        )
    return raw


@app.post("/api/curate/plan")
def api_curate_plan(payload: CuratePlanRequest, request: Request) -> JSONResponse:
    """对话式数据库管护第一步：preview + confirm_token。**零写盘**
    （search_online 的 plan 会真实联网查官方源，并记一行请求账本 .userdata/curate_net_ledger.jsonl）。

    动作：list 清点 / import 本地导入（内容 hash 去重）/ search_online 联网搜官方源（候选不落盘）/
    remove 回收站式删除预览 / restore 恢复预览。拿到 preview 与 confirm_token 后念给用户确认，
    再走 `/api/curate/apply`。"""
    _require_same_origin(request)
    from ..corpus import corpus_curation as cc
    try:
        result = cc.run_curate_action(
            payload.action,
            dry_run=True,
            query=payload.query,
            source=payload.source,
            species=payload.species,
            limit=payload.limit,
            filename=payload.filename,
            payload_bytes=_curate_payload_bytes(payload.payload_json),
            project_root=PROJECT_ROOT,
        )
    except cc.CurateError as exc:
        raise HTTPException(status_code=400, detail=exc.hint,
                            headers={"X-Error-Code": exc.code}) from exc
    return _json_utf8({"ok": True, "dry_run": True, "result": result})


@app.post("/api/curate/apply")
def api_curate_apply(payload: CurateApplyRequest, request: Request) -> JSONResponse:
    """对话式数据库管护第二步：回传 confirm_token **真执行**（import/search_online 入库写盘；
    remove 移入回收站；restore 移回）。token 重算比对不一致 → token_mismatch，**一个字节不动**。
    search_online 的 apply 还需把 plan 返回的完整结果作为 plan_result 原样回传。"""
    _require_same_origin(request)
    from ..corpus import corpus_curation as cc
    try:
        result = cc.run_curate_action(
            payload.action,
            dry_run=False,
            query=payload.query,
            source=payload.source,
            species=payload.species,
            limit=payload.limit,
            filename=payload.filename,
            payload_bytes=_curate_payload_bytes(payload.payload_json),
            plan_result=payload.plan_result,
            confirm_token=payload.confirm_token,
            force=payload.force,
            project_root=PROJECT_ROOT,
        )
    except cc.CurateError as exc:
        raise HTTPException(status_code=400, detail=exc.hint,
                            headers={"X-Error-Code": exc.code}) from exc
    return _json_utf8({"ok": True, "dry_run": False, "result": result})


@app.post("/api/curate/check-updates")
def api_curate_check_updates(payload: CurateCheckUpdatesRequest, request: Request) -> JSONResponse:
    """检查来源更新（2026-08-03 新能力 `curate.check_updates`，扩在线源）。**只读**：无 confirm_token、
    不落盘（在线比对会经限速唯一出口联网，并记一行请求账本）。

    有在线通道的来源（ArrayExpress / ENCODE / 10x）真在线拉最新清单与本地库比对；
    离线快照源（CELLxGENE/EBI SCEA/HCA）如实报告本地快照条数/日期 + 官网核对入口，
    不伪造在线比对能力。网络失败不 5xx——结果里该来源的 note_zh 如实写明
    （ENCODE/10x 拉不到时 mode 如实降级 "snapshot"）。"""
    _require_same_origin(request)
    from ..corpus import corpus_curation as cc
    result = cc.check_updates(payload.sources, project_root=PROJECT_ROOT)
    return _json_utf8({"ok": True, "result": result})


                                                                                                       
                                                                
                                                        
                                                            
                                                           
                                                                   
                                  
                                                              
 
                                                 
                                                            
                                                       
                                                                                    
                                                              
                                   

_CORPUS_SYNC_JOB_LOCK = threading.RLock()
_CORPUS_SYNC_JOB: dict = {
    "status": "idle",                                            
    "started_at": None,
    "finished_at": None,
    "result": None,                                                                  
    "error": None,
}
                                          
_CORPUS_SYNC_VECTOR_TIMEOUT_S = CORPUS_SYNC_VECTOR_TIMEOUT_SECONDS
                                        
_CORPUS_SYNC_JOB_KIND = "corpus_sync"
                                             
_CORPUS_SYNC_RECONCILED = False


def _corpus_sync_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _corpus_sync_jobs_db() -> Path:
    """持久镜像库路径（实例 userdata 层 / env 覆盖，规则真源在 durable_jobs）。"""
    from . import durable_jobs                           
    return durable_jobs.default_db_path(PROJECT_ROOT)


def _corpus_sync_mirror_warn(step: str, exc: Exception) -> None:
    """持久镜像写失败只记告警，绝不反过来炸掉 job 本体/状态端点（镜像是增强，不是真源）。"""
    print(f"[corpus-sync] 持久镜像 {step} 失败（不影响任务本身）：{type(exc).__name__}: {exc}",
          file=sys.stderr)


def _corpus_sync_reconcile_once() -> None:
    """进程启动后首次触碰 job 状态时镜像对账：镜像停在 running = 上一进程被打断，如实翻
    failed（单 worker 前提下成立——本进程没发起它，running 镜像即孤儿）。"""
    global _CORPUS_SYNC_RECONCILED
    with _CORPUS_SYNC_JOB_LOCK:
        if _CORPUS_SYNC_RECONCILED:
            return
        if _CORPUS_SYNC_JOB["status"] != "idle":
            _CORPUS_SYNC_RECONCILED = True
            return
        from . import durable_jobs                           
        try:
            durable_jobs.reconcile_interrupted(
                _CORPUS_SYNC_JOB_KIND, "进程重启，上一次同步任务被中断。", db_path=_corpus_sync_jobs_db())
        except Exception as exc:
            _corpus_sync_mirror_warn("reconcile", exc)
        else:
            _CORPUS_SYNC_RECONCILED = True


def _invalidate_shared_corpus_caches() -> None:
    invalidate_external_cache()
    publish_corpus_write(PROJECT_ROOT)              
    from ..retrieval import recall_api
    recall_api.invalidate_vectors()


def _refresh_shared_corpus_cache() -> None:
    from . import rq_jobs
    if rq_jobs.backend_active():
        rq_jobs.consume_web_invalidate_flag(_invalidate_shared_corpus_caches)


def _corpus_sync_job_snapshot() -> dict:
    """job 状态快照。RQ 后端点亮时真源在 Redis（读不到如实 503，不伪造 idle）；默认路径
    返回内存快照，全新进程（idle 且从未跑过）时回落到持久镜像——重启不丢上一轮终态。"""
    from . import rq_jobs                           
    if rq_jobs.backend_active():
        try:
            snap = rq_jobs.snapshot()
        except rq_jobs.JobBackendError as exc:
            raise HTTPException(status_code=503, detail=exc.message,
                                headers={"X-Error-Code": exc.code}) from exc
        _refresh_shared_corpus_cache()
        return snap
    with _CORPUS_SYNC_JOB_LOCK:
        _corpus_sync_reconcile_once()
        snap = dict(_CORPUS_SYNC_JOB)
        if snap["status"] == "idle" and snap["finished_at"] is None:
            from . import durable_jobs                           
            try:
                mirrored = durable_jobs.load(_CORPUS_SYNC_JOB_KIND, db_path=_corpus_sync_jobs_db())
            except Exception as exc:
                _corpus_sync_mirror_warn("load", exc)
                mirrored = None
            if mirrored is not None:
                return mirrored
        return snap


def _corpus_sync_rebuild_vectors() -> "str | None":
    """语料向量重建（job 第二阶段，同在后台线程）：子进程跑 build_corpus_vectors.py
    --include-uploads 到临时文件，退出码 0（全成）或 2（留缺口，运行期查询侧补嵌兜住）都算
    可用 → os.replace 原子替换目标文件 → recall_api.invalidate_vectors()。
    目标路径取 env BIODATA_EMBED_VECTOR_FILE（部署侧指到 /data 持久路径；代码不硬编码）；
    未配置（本机形态默认）→ 无可重建，None 跳过不算失败。
    返回 None=成功/跳过；str=失败摘要（只留子进程输出 tail——构建脚本绝不打印 key，
    这里也不再经手 key 本身）。"""
    from ..retrieval import recall_api                             

    target = recall_api._vector_file_path()
    if target is None:
        return None
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_corpus_vectors.py"
    if not script.is_file():
        return f"语料向量构建脚本缺失（{script.name} 不在安装内）"
    import subprocess                

    tmp_path = target.with_name(target.name + ".rebuild-tmp")
    cmd = [sys.executable, str(script), "--include-uploads", "--out", str(tmp_path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_CORPUS_SYNC_VECTOR_TIMEOUT_S, cwd=str(PROJECT_ROOT))
    except subprocess.TimeoutExpired:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return f"语料向量重建超时（{_CORPUS_SYNC_VECTOR_TIMEOUT_S // 60} 分钟上限）"
    tail = (((proc.stdout or "") + "\n" + (proc.stderr or "")).strip())[-1500:]
    if proc.returncode not in (0, 2):
        return f"语料向量重建失败（退出码 {proc.returncode}）：{tail or '无输出'}"
    os.replace(tmp_path, target)
    recall_api.invalidate_vectors()
    return None


def _corpus_sync_execute(sources: "list[str] | None") -> dict:
    """job 第一阶段执行体：sync_updates（显式无补丁作用域 → 共享写层 upload_*）。单通道——
    默认线程路径与 RQ worker 路径（rq_jobs.corpus_sync_entry）共用这一份实现，禁止平行
    写第二份。返回 operation receipt；异常原样上抛，由调用方收口为 failed。"""
    from ..corpus import corpus_curation as cc
    from ..corpus.patch_package import unbound_patch_scope

    with unbound_patch_scope():
        return cc.sync_updates(sources, project_root=PROJECT_ROOT)


def _corpus_sync_job_finalize(status: str, result: "dict | None", error: "str | None") -> None:
    """终态落账（默认线程路径）：内存快照更新 + 持久镜像 best-effort 写穿透。镜像失败只
    告警——运行期事实源仍是内存快照，镜像只服务重启对账。"""
    finished_at = _corpus_sync_now()
    with _CORPUS_SYNC_JOB_LOCK:
        _CORPUS_SYNC_JOB.update(
            status=status, finished_at=finished_at, result=result, error=error)
        from . import durable_jobs                           
        try:
            durable_jobs.record_terminal(
                _CORPUS_SYNC_JOB_KIND, status, finished_at, result, error,
                db_path=_corpus_sync_jobs_db())
        except Exception as exc:
            _corpus_sync_mirror_warn("record_terminal", exc)


def _corpus_sync_job_run(sources: "list[str] | None") -> None:
    """job 线程体（默认线程路径）：sync → 外部库缓存失效 →（有新增）向量重建。一切异常收口
    为 failed 状态，绝不漏栈到线程外；结果/错误都进 _CORPUS_SYNC_JOB 供状态端点如实呈现，
    并经 _corpus_sync_job_finalize 写穿透到持久镜像。"""
    result = None
    try:
        result = _corpus_sync_execute(sources)
        invalidate_external_cache()
        publish_corpus_write(PROJECT_ROOT)              
        vec_err = None
        if int(result.get("imported_total") or 0) > 0:
            vec_err = _corpus_sync_rebuild_vectors()
        _corpus_sync_job_finalize("failed" if vec_err else "done", result, vec_err)
    except Exception as exc:                                                
        hint = getattr(exc, "hint", "") or str(exc) or type(exc).__name__
        _corpus_sync_job_finalize("failed", result, hint)


def _corpus_sync_job_start(sources: "list[str] | None") -> dict:
    """启动或附着语料同步 job（单飞：running 中不新建、不抛 sync_busy，直接返回现状快照）。

    env BIODATA_JOB_BACKEND=rq → 委托 rq_jobs（跨进程 worker 执行）；后端不可用如实 503，
    不静默回落内存路径（用户明明开了 MQ，静默降级会变成查不出来的错觉）。默认路径先对
    持久镜像做一次重启对账，置 running 后 best-effort 写穿透。"""
    from . import rq_jobs                           
    if rq_jobs.backend_active():
        try:
            return rq_jobs.start(sources)
        except rq_jobs.JobBackendError as exc:
            raise HTTPException(status_code=503, detail=exc.message,
                                headers={"X-Error-Code": exc.code}) from exc
    with _CORPUS_SYNC_JOB_LOCK:
        _corpus_sync_reconcile_once()
        if _CORPUS_SYNC_JOB["status"] == "running":
            return dict(_CORPUS_SYNC_JOB)
        started_at = _corpus_sync_now()
        _CORPUS_SYNC_JOB.update(
            status="running", started_at=started_at,
            finished_at=None, result=None, error=None)
        from . import durable_jobs                           
        try:
            durable_jobs.record_running(
                _CORPUS_SYNC_JOB_KIND, started_at, db_path=_corpus_sync_jobs_db())
        except Exception as exc:
            _corpus_sync_mirror_warn("record_running", exc)
    try:
        worker = threading.Thread(
            target=_corpus_sync_job_run, args=(sources,), name="corpus-sync-job", daemon=True)
        worker.start()
    except Exception as exc:
        _corpus_sync_job_finalize("failed", None, "同步任务线程启动失败，请稍后重试。")
        raise HTTPException(status_code=503, detail="同步任务线程启动失败，请稍后重试。",
                            headers={"X-Error-Code": "job_start_failed"}) from exc
    return _corpus_sync_job_snapshot()


                                                                                                   

_ADMIN_TOKEN_ENV = "BIODATA_ADMIN_TOKEN"
_LOOPBACK_CLIENT_HOSTS = frozenset({"127.0.0.1", "::1"})


def _admin_client_host(request: Request) -> str:
    """对端地址（独立小函数：测试 monkeypatch 这里；真实部署 cron 从容器内 127.0.0.1 打来）。"""
    return str(request.client.host) if request.client else ""


def _require_admin(request: Request) -> None:
    """管理端点双闸：仅 loopback 对端 + env BIODATA_ADMIN_TOKEN（未配置 → 403 fail-closed
    「未启用」；X-Admin-Token 头经 hmac.compare_digest 比对，防时序侧信道）。
    token 绝不进日志/响应（请求日志中间件只记 method+path，不带头）。"""
    if _admin_client_host(request) not in _LOOPBACK_CLIENT_HOSTS:
        raise HTTPException(status_code=403, detail="管理端点仅接受本机回环调用。")
    token = os.getenv(_ADMIN_TOKEN_ENV, "").strip()
    if not token:
        raise HTTPException(status_code=403, detail="管理端点未启用（服务端未配置管理令牌）。")
    given = (request.headers.get("x-admin-token") or "").strip()
    if not given or not hmac.compare_digest(given, token):
        raise HTTPException(status_code=403, detail="管理令牌无效。")


@app.post("/api/admin/corpus-sync")
def api_admin_corpus_sync(request: Request) -> JSONResponse:
    """cron 用：启动/附着语料同步 job（恒全源），202 立即返回，不阻塞。双闸见 _require_admin。"""
    _require_same_origin(request)
    _require_admin(request)
    job = _corpus_sync_job_start(None)
    return _json_utf8({"ok": True, "job": job}, status_code=202)


@app.get("/api/admin/corpus-sync/status")
def api_admin_corpus_sync_status(request: Request) -> JSONResponse:
    """cron 用：查语料同步 job 状态。双闸见 _require_admin。"""
    _require_same_origin(request)
    _require_admin(request)
    return _json_utf8({"ok": True, "job": _corpus_sync_job_snapshot()})


@app.post("/api/curate/sync-updates")
def api_curate_sync_updates(payload: CurateSyncUpdatesRequest, request: Request) -> JSONResponse:
    """检查更新 → 有新增则自动入库的复合流（2026-08-06 `curate.sync_updates`）。

    双消费点：① 未装 langchain 扩展时前端 runner 从这里跑同一份复合流（agent 图内 execute
    节点调的是同一个 `corpus_curation.sync_updates` 真源）；② CLI/外部直接调用。
    写侧经 uploads 管线 + 账本，可经回收站撤回；网络失败不 5xx——该来源的 note_zh 如实写明。

    整任务跨进程锁——另一进程/线程的 sync 正在跑 → CurateError
    (sync_busy) → HTTP 400（fail-closed 拒绝并发写，不排队）；返回扩展 operation receipt
    （operation_id / created_files[] / failed_sources[] / skipped_existing，既有字段兼容）。

    2026-08-26：**仅 guard on（网页形态）**改为异步——启动/附着进程内单飞
    job（见上方「语料同步后台任务」区块），202 立即返回 `{ok, job, async: true}`（additive
    新键），前端轮询 `GET /api/curate/sync-updates/status`；guard off（本机形态）下方阻塞
    路径逐字节不变。"""
    _require_same_origin(request)
    if _account_gate_required():
        job = _corpus_sync_job_start(payload.sources)
        return _json_utf8({"ok": True, "job": job, "async": True}, status_code=202)
    from ..corpus import corpus_curation as cc
    try:
        result = cc.sync_updates(payload.sources, project_root=PROJECT_ROOT)
    except cc.CurateError as exc:
        raise HTTPException(status_code=400, detail=exc.hint,
                            headers={"X-Error-Code": exc.code}) from exc
    return _json_utf8({"ok": True, "result": result})


@app.get("/api/curate/sync-updates/status")
def api_curate_sync_updates_status(request: Request) -> JSONResponse:
    """语料同步 job 状态轮询（2026-08-26 登录即可——中间件闸，无 token 闸）。

    guard off（本机形态）下也可用：job 恒 idle（本机路径走上方阻塞端点，不起 job），
    前端据此自然降级。"""
    _require_same_origin(request)
    return _json_utf8({"ok": True, "job": _corpus_sync_job_snapshot()})


@app.get("/api/curate/sync-status")
def api_curate_sync_status(request: Request) -> JSONResponse:
    """实例级同步状态（2026-08-22 §7）：上次同步时间 / 上次 operation / 是否 busy。

    **只读、不写盘**（busy 实时探测 sync 整任务锁，跨进程准确；「上次同步」是**实例级事实**，
    不得存 per-profile localStorage——评审①#6 裁决）。"""
    _require_same_origin(request)
    from ..corpus import corpus_curation as cc
    return _json_utf8({"ok": True, "result": cc.sync_status(project_root=PROJECT_ROOT)})


@app.post("/api/curate/recall")
def api_curate_recall(payload: CurateRecallRequest, request: Request) -> JSONResponse:
    """按 operation_id **整次撤回**一次 sync 的全部成功写入（2026-08-22 §7）。

    回收站语义：把该 operation 的 created_files[] 移入 `.userdata/recycle/`（可逆、可重入、
    单文件失败不连累其余）；operation 不存在 → 400 unknown_operation。"""
    _require_same_origin(request)
    from ..corpus import corpus_curation as cc
    try:
        result = cc.recall_sync_operation(payload.operation_id, project_root=PROJECT_ROOT)
    except cc.CurateError as exc:
        raise HTTPException(status_code=400, detail=exc.hint,
                            headers={"X-Error-Code": exc.code}) from exc
    return _json_utf8({"ok": True, "result": result})


@app.post("/api/curate/status")
def api_curate_status(request: Request) -> JSONResponse:
    """数据库状态汇报（2026-08-03 `curate.db_status` 的能力端点）。**只读、离线、不抛**：
    各源条数/快照日期 + 外部库与回收站清单 + 近期联网审计摘要。

    双消费点：① 未装 langchain 扩展时前端 runner 从这里取同一份事实（agent 图内
    execute 节点调的是同一个 `corpus_status.db_status` 真源）；② CLI/外部只读探查。
    汇报措辞不归本端点（LLM 组织在 agent narrate / `/api/act/summary`）。"""
    _require_same_origin(request)
    from ..corpus import corpus_status as cs
    return _json_utf8({"ok": True, "result": cs.db_status(project_root=PROJECT_ROOT)})


                                                                                                
                               
                                                                    
                                                           
                                                                                                     
                                                                              
                             
                                                           

                                                       
                                                                      
_WATCH_CHECK_MAX_UIDS = 200

                                                                
                                                                                                  
                                                                             
                                                      
                                                                         
RECORD_FINGERPRINT_SCHEMA = "v1"


def _norm_fingerprint_component(value: Any) -> str:
    """指纹组分的稳定规范化：None → ""；bool/数字 → 稳定字面量；字符串去首尾空白（大小写不归一——
    unit 等刻意归一化字段由调用方另行 .lower()，避免所有字段被无差别折叠）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def record_fingerprint_v1(uid: Any, item: dict) -> str:
    """record_fingerprint_schema v1：一条检索结果的语义指纹（sha256 hex，全 64 位）。

    输入取 workflow `_serialize_retrieved_data` 的 payload 项（含 dataset_uid / count / unit /
    raw_data_status 结构化字段）。缺字段按空值规范化参与哈希（不抛）——语料字段演进时旧记录
    仍可与新记录比较。raw_data_status 若为字符串（老格式）按 code=该字符串、authoritative=false
    处理（本项目检索结果恒为结构化 dict，此处只是防御）。"""
    raw = item.get("raw_data_status") or {}
    code = raw.get("code") if isinstance(raw, dict) else raw
    authoritative = raw.get("authoritative") if isinstance(raw, dict) else False
    payload = {
        "schema": RECORD_FINGERPRINT_SCHEMA,
        "dataset_uid": _norm_fingerprint_component(uid),
        "sample_size": {
            "count": _norm_fingerprint_component(item.get("count")),
            "unit": _norm_fingerprint_component(item.get("unit")).lower(),                         
        },
        "raw_data_status": {
            "code": _norm_fingerprint_component(code).lower(),
            "authoritative": bool(authoritative),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_watch_spec(payload: WatchCheckRequest) -> dict:
    """watch-check 入参校验 + 规范化（返回 executed_spec 的确定性组成部分；非法 → HTTPException 400）。

    与 /api/recommend 同口径：sources 未知名/空白名显式 400 点名；非法日期/倒挂窗口 400；
    facet/suppressed/lenient 走同一 sanitize 白名单收敛；空 spec（无关键词/来源/分面）400——
    没有可重跑的检索条件时不许拿「全库浏览」冒充检查结果。"""
    spec_version = str(payload.spec_version or "").strip()
    if spec_version != RECORD_FINGERPRINT_SCHEMA:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 spec_version「{spec_version}」（当前支持：{RECORD_FINGERPRINT_SCHEMA}）。",
        )
    query = payload.query.strip()
    if not query and not payload.sources and not payload.facet_filters:
        raise HTTPException(
            status_code=400,
            detail="检查条件为空（没有关键词/来源/分面条件），没有可重跑的确定性检索。",
        )
    if payload.sources:
        _validate_or_400(validate_sources, payload.sources,
                         known=known_source_values(DATA_DIR, PROJECT_ROOT))
    facet_filters = _sanitize_facet_filters(payload.facet_filters)
    suppressed_constraints = _sanitize_suppressed(payload.suppressed_constraints)
    lenient_dims = _sanitize_lenient_dims(payload.lenient_dims)
    date_from = _require_iso_date(payload.date_from, name="date_from")
    date_to = _require_iso_date(payload.date_to, name="date_to")
    _validate_or_400(validate_date_window, date_from, date_to)
    return {
        "spec_version": RECORD_FINGERPRINT_SCHEMA,
        "query": query,
        "sources": [str(x).strip() for x in payload.sources] if payload.sources else [],
        "facet_filters": facet_filters,
        "suppressed_constraints": suppressed_constraints,
        "lenient_dims": lenient_dims,
        "date_from": date_from,
        "date_to": date_to,
    }


@app.post("/api/watch/check")
def api_watch_check(payload: WatchCheckRequest, request: Request) -> JSONResponse:
    """课题更新检查的**确定性重跑**端点（2026-08-22 §4.1/§4.2，评审①阻断3）。

    入参为课题保存的确定性检索 spec；端点以 strategy=fixed、recall=off、rerank=off、polish=false
    重跑与 /api/recommend 同一条确定性管线（**零 LLM**，不读 LLM 配置），返回：
      result_total    未截断命中总数（作「是否收窄」判断的事实）；
      uids[]          ≤200 的 uid 集合（无序集合语义：实现保留检索器稳定顺序以支持确定性对拍，
                      调用方不得依赖顺序）；
      fingerprints    {uid: record_fingerprint_schema v1 语义指纹}（material change 比较用）；
      truncated       result_total > 200 时为 true（此时不得断言「某条已从全部结果消失」）；
      executed_spec   规范化后的执行规格（回显用户实际保存/生效的条件，不只是原始自然语言）；
      checked_at      本次检查时间（实例级事实）。
    baseline 生成（保存检查条件时）与逐次检查共用本端点口径，保证可比。"""
    _require_same_origin(request)
    spec = _validate_watch_spec(payload)
    workflow = DatasetRecommendationWorkflow(settings=get_settings())
    meta = workflow.run_with_meta(
        RecommendParams(
            query=spec["query"],
            top_k=_WATCH_CHECK_MAX_UIDS,
            use_llm=False,                                                 
            mock_llm=False,
            rerank_backend="off",
            recall_backend="off",
            strategy="fixed",
            llm_available=False,                                          
            date_from=spec["date_from"],
            date_to=spec["date_to"],
            sources=spec["sources"] or None,
            facet_filters=spec["facet_filters"],
            suppressed_constraints=spec["suppressed_constraints"],
            lenient_dims=spec["lenient_dims"],
        )
    )
    uids: list[str] = []
    fingerprints: dict[str, str] = {}
    for item in meta.retrieved_data:
        uid = str(item.get("dataset_uid") or "").strip()
        if not uid:
            continue
        if len(uids) >= _WATCH_CHECK_MAX_UIDS:
            break
        uids.append(uid)
        fingerprints[uid] = record_fingerprint_v1(uid, item)
    result_total = int(meta.result_total or len(uids))
    from ..corpus.net_ledger import now_iso
    return _json_utf8({"ok": True, "result": {
        "result_total": result_total,
        "uids": uids,
        "fingerprints": fingerprints,
        "truncated": result_total > _WATCH_CHECK_MAX_UIDS,
        "executed_spec": {
            "spec_version": spec["spec_version"],
            "query": spec["query"],
            "sources": spec["sources"],
            "facet_filters": spec["facet_filters"],
            "suppressed_constraints": spec["suppressed_constraints"],
            "lenient_dims": spec["lenient_dims"],
            "date_from": spec["date_from"],
            "date_to": spec["date_to"],
            "strategy": "fixed",
            "recall": "off",
            "rerank": "off",
            "polish": False,
        },
        "checked_at": now_iso(),
    }})


@app.post("/api/artifacts/export-pack")
def api_artifacts_export_pack(payload: ExportPackRequest, request: Request) -> Response:
    """课题导出中心（2026-08-22 「研究包」的落地形态）。

    入参 = 课题**当前状态快照**（前端组装：candidates 的 uid+status+reason+verified_at +
    check_condition + provenance + 导出类型 kind），服务端从**本地语料**解析数据集元数据
    （uid 的数据集内容不信任调用方、前端不传），生成研究材料 ZIP 下载：

      - manifest（稳定标识符 + 来源 URL + 最后核验时间）
      - 纳入/排除表（含理由与核验时间）
      - 下载任务包（复用 `task_pack` 的下载子集，放 `download/` 子目录）
      - 三格式引文：RIS / BibTeX（`reuse_pack` 既有）+ GB/T 7714-2015 新 formatter（[DS/OL]）
      - 检索与核验溯源（来自 provenance，只列在场字段）
      - 「数据发现与筛选方法」草稿（只从真实 provenance/search_trace 生成、明确标注草稿、
        不编造样本数/访问日期/数据库范围）
      - recipe.json（可重跑：check_condition.spec + provenance 关键参数 + 候选快照）

    `kind` 决定 ZIP 内容子集：download_list / citations / screening_record 是**单项轻量导出**
    （内容如实标注在包内 README「本包里有什么」）；full = 全部研究材料。

    **只读、离线、不调用 LLM、不写盘**（内存拼 ZIP，同 task_pack 范式）。目录版本
    （snapshot_id/content_digest）经 `X-Biodata-Export-Meta` 响应头回给前端写台账——
    那是实例级事实，前端不自己造。检索器/编排/冻结评测从不 import `export_pack`。"""
    _require_same_origin(request)
    from ..content import export_pack
    from ..content import task_pack as _task_pack
    try:
        kind = export_pack.sanitize_kind(payload.kind)
        snapshot = export_pack.sanitize_snapshot(payload.project)
    except export_pack.ExportPackError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    records = load_full_corpus(DATA_DIR, PROJECT_ROOT)
    pack = export_pack.build_export_pack(snapshot, records, today=datetime.date.today().isoformat())
    files = export_pack.render_files(pack, kind)
    try:
        blob = _task_pack.files_to_zip_bytes(files)
    except _task_pack.TaskPackError as exc:
        status = 413 if exc.code == "too_large" else 400
        return _json_utf8({"ok": False, "code": exc.code, "message_zh": str(exc)}, status_code=status)
    name = f"biodata-export-{pack['exported_at']}-{pack['corpus']['snapshot_id'][:8] or 'nocorpus'}.zip"
                                                       
    meta = json.dumps({
        "dataset_version": pack["corpus"]["snapshot_id"],
        "content_digest": pack["corpus"]["content_digest"],
        "exported_at": pack["exported_at"],
        "kind": kind,
    }, ensure_ascii=True, separators=(",", ":"))
    return Response(
        content=blob, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"',
                 "X-Biodata-Export-Meta": meta},
    )


@app.get("/spec/upload")
def upload_spec() -> FileResponse:
    """把《数据集上传规范》原文以纯文本内联呈现（前端「查看完整规范」链接指向此处）。
    只暴露这一份自撰的**公开**规范文件，**不挂载整个 `使用教程/` 目录**（避免泄漏组内其它材料）。"""
                                                                     
    spec = RESOURCE_ROOT / "使用教程" / "数据集上传" / "数据集上传规范.md"
    if not spec.exists():
        raise HTTPException(status_code=404, detail="规范文件未找到。")
    return FileResponse(spec, media_type="text/plain; charset=utf-8")


@app.post("/api/diagnose")
def api_diagnose(payload: DiagnoseRequest, request: Request) -> JSONResponse:
    _require_same_origin(request)
    normalized_provider = _normalize_provider(payload.provider)
                                                         
                                                     
                                  
                                                                                   
    requested_base_url = "" if normalized_provider in ("mock", "trial") else _validate_endpoint_url(payload.base_url)
                                                                           
    mock_llm = bool(payload.mock_llm) or normalized_provider == "mock"
    use_llm = bool(payload.use_llm or mock_llm)

                                                                                
                                                                            
                                                                             
                                                                    
    cfg = _materialize_request_llm_config(
        normalized_provider, use_llm=use_llm, mock_llm=mock_llm,
        api_key=payload.api_key, base_url=requested_base_url, model=payload.model)
                                                  
                                                                   
                                                    
                                                                    
                                                    
    _gate_llm_quota(
        request, cfg=cfg, provider=normalized_provider, use_llm=True,
        mock_llm=mock_llm, api_key=payload.api_key, require_enabled=False)
    health = healthcheck(cfg)
    network = diagnose_network(cfg)
    return _json_utf8(
        {
        "ok": True,
        "provider": normalized_provider,
        "healthcheck": health,
        "network_diagnose": network,
        }
    )


@app.get("/api/sources")
def api_sources() -> JSONResponse:
    """可选数据来源清单 + 各自数据集数（供智能查询页「按来源勾选」渲染）。
    10x Genomics 恒在且置顶；外部平台库来自静态快照。"""
    return _json_utf8({"ok": True, "sources": available_sources(DATA_DIR, PROJECT_ROOT)})


                                                                 
                                                             
                                                             
                                                   
_web_item_from_record = item_view.build_item

                                             
_facet = item_view.facet_list
_interleave_by_source = item_view.interleave_by_source

                                                     
                                                                       
                        
_MAX_DATASETS_LIMIT = MAX_DATASETS_LIMIT

                                              
                                               
                                                   
_DATASETS_FULL_CACHE_LOCK = threading.Lock()
_DATASETS_FULL_CACHE: "tuple[tuple, bytes, str] | None" = None


def _reset_datasets_response_cache() -> None:
    """清理 Web 响应缓存（主要给契约测试；生产通过语料代际键自动失效）。"""
    global _DATASETS_FULL_CACHE
    with _DATASETS_FULL_CACHE_LOCK:
        _DATASETS_FULL_CACHE = None


def _etag_matches(request: Request, etag: str) -> bool:
    values = [part.strip() for part in (request.headers.get("if-none-match") or "").split(",")]
    return "*" in values or etag in values


def _dataset_facet_bits(record: Any) -> tuple[str, str, str, int | None]:
    """单记录 → 浏览页分面四元组 (species, platform, source, published_year)。

    PERF-M01：分面计数不再依赖构造完整展示 item（此前 limit=1 也为全库每条构造 item，
    O(N) CPU/内存/JSON 分配被公网重复请求放大）。四项派生与 `item_view.build_item`
    逐字同源——platform/source/published_year 与 build_item 同式，species 与旧浏览循环
    同式；改 build_item 派生时这里必须同步（tests/test_webapp_security.py 有 parity 钉）。"""
    raw = record.raw if isinstance(record.raw, dict) else {}
    species = (record.species or "").strip()
    platform = (record.platform_family or "").strip()
    source = str(raw.get("source") or "").strip() or "10x Genomics"
    published_date = str(raw.get("published_date") or "").strip()
    return species, platform, source, item_view.published_year(published_date)


def _datasets_payload(records: list[Any], *, limit: int | None, offset: int) -> dict[str, Any]:
    """从已交错的全库记录生成浏览载荷；分面口径与旧端点逐位一致。"""
    species_counter: dict[str, int] = {}
    platform_counter: dict[str, int] = {}
    source_counter: dict[str, int] = {}
    year_counter: dict[str, int] = {}
    unknown_year_count = 0
    for record in records:
        species, platform, source, year = _dataset_facet_bits(record)
        if species:
            species_counter[species] = species_counter.get(species, 0) + 1
        if platform:
            platform_counter[platform] = platform_counter.get(platform, 0) + 1
        source_counter[source] = source_counter.get(source, 0) + 1
        if year is not None:
            ykey = str(year)
            year_counter[ykey] = year_counter.get(ykey, 0) + 1
        else:
            unknown_year_count += 1
    selected = records[offset:] if limit is None else records[offset:offset + limit]
    items = [_web_item_from_record(r, include_introduction=False) for r in selected]
    return {
        "ok": True,
        "count": len(records),
        "facets": {
            "species": _facet(species_counter),
            "platform": _facet(platform_counter),
            "source": _facet(source_counter),
            "published_year": _facet(year_counter),
        },
        "unknown_year_count": unknown_year_count,
        "records": items,
    }


def _cached_datasets_full_response(records: list[Any], generation: tuple) -> tuple[bytes, str]:
    global _DATASETS_FULL_CACHE
    with _DATASETS_FULL_CACHE_LOCK:
        cached = _DATASETS_FULL_CACHE
        if cached is not None and cached[0] == generation:
            return cached[1], cached[2]
        payload = _datasets_payload(records, limit=None, offset=0)
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
                                                       
        etag = f'W/"{hashlib.sha256(body).hexdigest()}"'
        _DATASETS_FULL_CACHE = (generation, body, etag)
        return body, etag


@app.get("/api/datasets")
def api_datasets(request: Request = None, limit: int | None = None, offset: int = 0) -> Response:
    """全库浏览：**所有来源并列**返回归一化记录 + 物种/平台/来源分面（前端做筛选与分页）。

    基础语料每次现算（767 条，开销可忽略），故上传新增的数据即时可见；
    外部平台库为静态快照（缓存），与基础语料并列展示、每条带来源标签。

    可选 `limit`/`offset`（2026-08-06 additive）：只截 `records` 当前页；
    `count` 与 `facets` 恒按全库计算（前端整拉一次做客户端筛选，不受影响）。
    不传时行为逐位不变。对齐 MCP `browse_datasets` 的分页入参。

    PERF-M01：`limit` 显式传入时有上限（`_MAX_DATASETS_LIMIT`，与 MCP browse_datasets 同源
    于 `app/limits.MAX_DATASETS_LIMIT`）；分面计数走
    `_dataset_facet_bits` 轻量投影（不再为全库每条构造完整 item），展示 item 只对
    当前页构造——公网重复请求不再放大 O(N) CPU/内存/JSON 分配。
    """
    if limit is not None:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit 需 ≥ 1。")
        if limit > _MAX_DATASETS_LIMIT:
                                                         
                                                            
            raise HTTPException(status_code=422, detail=f"limit 最大 {_MAX_DATASETS_LIMIT}。")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset 需 ≥ 0。")
    if limit is None and offset == 0:
        generation = corpus_cache_generation(DATA_DIR, PROJECT_ROOT)
        records = _interleave_by_source(load_full_corpus(DATA_DIR, PROJECT_ROOT))
        generation_after_load = corpus_cache_generation(DATA_DIR, PROJECT_ROOT)
        if generation_after_load != generation:
                                                       
                                      
            generation = generation_after_load
            records = _interleave_by_source(load_full_corpus(DATA_DIR, PROJECT_ROOT))
        body, etag = _cached_datasets_full_response(records, generation)
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request is not None and _etag_matches(request, etag):
            return Response(status_code=304, headers=headers)
        return Response(
            content=body, media_type="application/json; charset=utf-8", headers=headers
        )
    records = _interleave_by_source(load_full_corpus(DATA_DIR, PROJECT_ROOT))
    return _json_utf8(_datasets_payload(records, limit=limit, offset=offset))


def _locate_record_or_409(uid: str, url: str, name: str, source: str):
    """introduction/fair 共用的定位前言：三键全空 → 400；`locate_record`（Web 与 MCP
    共用单一真源）按 uid 精确 > url 精确 > name 精确（source 消歧）定位，name 命中多条
    且消歧失败 → 409 + candidates（如实报歧义，绝不静默任取第一条；载荷形状是前端契约）；
    定位不到 → 404。命中则返回记录。"""
    if not (uid or url or name):
        raise HTTPException(status_code=400, detail="uid, url or name is required")
    record, ambiguous = locate_record(
        load_full_corpus(DATA_DIR, PROJECT_ROOT), uid=uid, url=url, name=name, source=source)
    if ambiguous:
        raise HTTPException(status_code=409, detail={
            "error": "ambiguous_name",
            "message": "该名称命中多条同名数据集，无法确定是哪一条；请改用 dataset_uid 精确指定。",
            "candidates": ambiguous,
        })
    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return record


@app.get("/api/introduction")
def api_introduction(
    request: Request = None,                                                             
    uid: str = Query(default="", max_length=240),
    url: str = Query(default="", max_length=1200),
    name: str = Query(default="", max_length=600),
    source: str = Query(default="", max_length=200),
    llm: int = Query(default=0),
) -> JSONResponse:
    """按需返回单个数据集介绍，避免浏览列表为 5,000+ 条记录复制大段摘要。

    定位走 `corpus.locate_record`（Web 与 MCP 共用单一真源）：**严格优先级 uid 精确 > url 精确 >
    name 精确（source 消歧）**，每档键全扫一遍语料再退化——语料存在同名两条时（如 …-ff-ultima /
    …-ff-ultima-4），uid 全参请求绝不会被靠前那条的 name 命中截胡。name 命中多条且 source 消歧
    失败 → 409 + candidates（如实报歧义，绝不静默任取第一条）。刻意**不**建跨请求缓存索引做 O(1)：
    基础语料每次现算以保证上传即时可见（见 /api/datasets），缓存索引会在上传后静默过期，
    得不偿失（评审结论）。

    `llm=1`（N5，opt-in）：在确定性介绍上 additive 叠加 LLM 中文导读（`introduction.llm_summary`）。
    **双层门**：需服务端 `ENABLE_LLM` 开 **且** 本请求显式 `llm=1` 才会真调；否则 `llm_summary=None`、
    确定性介绍逐字不变（fail-open）。`llm=0`（默认）时响应与从前逐字节一致。

    限流闸：`llm=1` 是 GET 但会产生费用的外部请求——加与写端点同级的同源检查
    （`_require_same_origin`）与简单频率限制（`_rate_limited`）；`llm=0` 路径逐位不变
    （纯确定性只读，无成本，不加闸）。"""
    uid, url, name, source = (str(v or "").strip() for v in (uid, url, name, source))
    if llm:
                                                   
                                                             
                                              
        if request is not None:
            _require_same_origin(request)
        if not _rate_limited("introduction:llm", limit=_LLM_INTRO_RATE_LIMIT, window=_LLM_INTRO_RATE_WINDOW):
            raise HTTPException(status_code=429, detail="AI 介绍生成过于频繁，请稍后再试。")
    record = _locate_record_or_409(uid, url, name, source)
    item = _web_item_from_record(record, include_introduction=True)
    intro = item["introduction"]
    if llm:
        from ..llm.intro_llm import enrich_introduction_with_llm
                                                                       
                                                                 
                                                                      
                                                                          
        with ENV_LOCK:
            intro_cfg = load_llm_config(project_root=CONFIG_ROOT)
                                           
                                                                  
                                                               
        if request is not None:
            _gate_llm_quota(request, cfg=intro_cfg, provider=intro_cfg.provider,
                            use_llm=True, mock_llm=False, api_key=None)
        intro = enrich_introduction_with_llm(item, intro=intro, config=intro_cfg)
    return _json_utf8({"ok": True, "introduction": intro})


@app.get("/api/fair")
def api_fair(
    uid: str = Query(default="", max_length=240),
    url: str = Query(default="", max_length=1200),
    name: str = Query(default="", max_length=600),
    source: str = Query(default="", max_length=200),
) -> JSONResponse:
    """按需返回单个数据集的 **FAIR 元数据自检 + 投稿数据可用性说明（DAS）**。

    只读、纯元数据、确定性、离线（同 `/api/introduction`：不调用 LLM、不联网）；定位/消歧口径
    一致——`corpus.locate_record` 单一真源（uid 精确 > url 精确 > name 精确+source 消歧；
    name 多条消歧失败 → 409 + candidates 如实报歧义）。核心走 `fair.build_fair_report`
    （Web 与 MCP 共用单一真源）；检索器/编排/冻结评测从不 import `fair` → 冻结 767 基准结构性不受影响。"""
    uid, url, name, source = (str(v or "").strip() for v in (uid, url, name, source))
    record = _locate_record_or_409(uid, url, name, source)
    item = _web_item_from_record(record, include_introduction=True)
    return _json_utf8({"ok": True, "fair_report": build_fair_report(item)})


@app.post("/api/reuse-pack")
async def api_reuse_pack(request: Request) -> JSONResponse:
    """N 个复用的公开数据集 → 一份投稿材料（英文段落 + 补充表 + 待办清单）。

    **为什么是 POST 而不是 GET**：GET 会把用户勾选的每个 dataset_uid 打进 uvicorn 的
    access log —— 那是**事实上的埋点**，而「要不要做使用数据采集」是产品方明确保留的
    未决策项。同时也违反「个人数据不进 URL 参数」。
    这里没有权限问题（本机 loopback），纯粹是**不许把用户行为落进日志**。

    入参 **keys-only**：`{"uids": ["cxg:...", "ae:..."]}`。服务端自己按 uid 解析记录，
    **不接受**调用方传数据集内容——一旦开了「把数据集描述贴进来」的口子，产品就有了吃进
    未发表工作的路径。入参是键不是内容 → IP 红线是结构性的。

    只读、离线、不调用 LLM、**不写盘**（导出走前端 Blob）。核心走
    `reuse_pack.build_pack_for_uids`（Web 与 MCP 共用单一真源）；检索器/编排/冻结评测
    从不 import `reuse_pack` → 冻结 767 基准结构性不受影响。
    """
    _require_same_origin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON：{\"uids\": [...]}") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象：{\"uids\": [...]}")
    try:
        uids = sanitize_uids(payload.get("uids"))
    except ReusePackError as exc:
        raise HTTPException(status_code=400, detail=str(exc),
                            headers={"X-Error-Code": exc.code}) from None
                                                       
                                                            
    def _build_pack_payload() -> dict[str, Any]:
        pack = build_pack_for_uids(uids, load_full_corpus(DATA_DIR, PROJECT_ROOT))
        return {
            "ok": True,
            "pack": pack,
            "markdown": pack_to_markdown(pack),
            "ris": to_ris(pack),                                    
            "bibtex": to_bibtex(pack),                                      
        }

    return _json_utf8(await run_in_threadpool(_build_pack_payload))


@app.get("/api/citations/download")
def api_citations_download(f: str = Query(default="", max_length=200)) -> Response:
    """把环内 `cite.export`（图内 LOOP_TOOLS）落盘在 `.userdata/citations/` 的引文文件
    发回浏览器（2026-08-19 批；additive 契约）。

    **为什么需要它**：图内 `cite.export` 的文件写在**服务端** `.userdata/citations/`
    （write 语义由 trace 快照锚定、可被 curate.rollback 看到），浏览器拿不到——旧的
    runner 路径（`actRunCiteExport`，前端 Blob）只下 RIS、把 BibTeX 丢了。本端点把已写出
    的文件按名发回，卡片「下载」按钮直接链到这里。

    **白名单纪律**：入参 `f` 只接受**裸文件名**（basename）——含路径分隔符（`/` `\\`）、
    `..`、`.`、空值一律 400 拒绝；`resolve()` 后必须仍落在 `.userdata/citations/` 内
    （前缀判定，防符号链接/目录穿越）；文件不存在 → 404。响应 `Content-Disposition:
    attachment` + 按扩展名给 Content-Type（RIS / BibTeX / 文本兜底）。
    只读端点，不写盘、不调 LLM。"""
    name = str(f or "").strip()
    if not name or "/" in name or "\\" in name or name in (".", "..") or Path(name).name != name:
        raise HTTPException(status_code=400, detail="f 必须是 .userdata/citations/ 下的裸文件名。")
    from .runtime_paths import account_data_dir_for
    from ..corpus.patch_package import current_patch_scope

    citations_dir = account_data_dir_for(PROJECT_ROOT, ".userdata/citations", current_patch_scope())
    base = citations_dir.resolve()
    candidate = (citations_dir / name).resolve()
    if candidate == base or base not in candidate.parents:
        raise HTTPException(status_code=404, detail="引文文件不在允许的目录内。")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="引文文件不存在或已被清理。")
    ext = candidate.suffix.lower()
    media_type = {
        ".ris": "application/x-research-info-systems",
        ".bib": "application/x-bibtex",
        ".txt": "text/plain; charset=utf-8",
    }.get(ext, "text/plain; charset=utf-8")
                                                             
                                                      
    cd_name = name.replace('"', "")
    return Response(
        content=candidate.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{cd_name}"'},
    )


                                                                                 
                                                                       
                                                                     
                                                                                   
                                                                                   
                                                        
                                                               
                                                                  
                                              
_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_ACK_SCHEMA = "biodata-mcp-upload-cursor/v1"

                                                             
                                                                
                                                
_MCP_CALLS_DEFAULT_LIMIT = 100
_MCP_CALLS_MAX_LIMIT = 200
_MCP_CALLS_DEFAULT_MAX_BYTES = 500_000
_MCP_CALLS_MAX_BYTES = 2 * 1024 * 1024


def _mcp_calls_log_path() -> Path:
    return instance_data_dir_for(PROJECT_ROOT, ".userdata") / "mcp_calls.jsonl"


def _mcp_calls_upload_cursor_path() -> Path:
    return instance_data_dir_for(PROJECT_ROOT, ".userdata") / "mcp_calls_uploaded.json"


def _legacy_mcp_call_id(line: str) -> str:
    """旧日志行（schema v0，无 call_id）按**行原文**合成稳定 call_id 并标 legacy:true——
    同一行每次读到都合成出同一个键，接收端幂等去重才有意义。"""
    return "legacy-" + hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:32]


def _parse_since_ts(raw: str) -> datetime.datetime | None:
    """since_ts 查询参数（ISO8601）；非法值 400 由调用方抛。naive 按 UTC 计。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="since_ts 需为 ISO8601 时间串。")
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


                            
_TELEMETRY_RELAY_WEB_DETAIL = "网页版不由本端点中继遥测。"


@app.get("/api/telemetry/mcp-calls")
def api_telemetry_mcp_calls(
    request: Request = None,
    after: int = Query(default=0),
    limit: int = Query(default=_MCP_CALLS_DEFAULT_LIMIT),
    max_bytes: int = Query(default=_MCP_CALLS_DEFAULT_MAX_BYTES),
    since_ts: str = Query(default=""),
) -> JSONResponse:
    """读 data_root/.userdata/mcp_calls.jsonl 的**行号增量**（批；additive 契约
    加分页与 legacy 键合成）。

    `after` = 已消费的最后一条记录行号（1-based；0/缺省 = 从头）。响应：
      - `records`：`after` 之后的记录数组（原样 JSONL 行；**无 call_id 的旧行** additive
        合成 `call_id="legacy-"+sha256(行原文)[:32]` 并标 `legacy:true`，幂等键稳定）；
      - `next_offset`：调用方下一次应传回的 `after`（幂等推进，绝不回退）；
      - `truncated`：本次因 limit/max_bytes 截断（还有未返回记录）时为 true。
    分页/过滤参数（全部 additive，缺省行为与旧版逐位一致）：
      - `limit`：单次最多返回记录数（默认 100，上限 200——对齐接收端 MAX_MCP_RECORDS）；
      - `max_bytes`：records 原文行 UTF-8 字节预算（默认 ~500KB，上限 2MB）；
      - `since_ts`：ISO8601；只返回 `ts` ≥ 该值的记录。**无 ts/不可解析 ts 的 legacy 行
        一律保留**（不过滤——它们是未消费数据，滤掉就永远传不出去了）；被 since_ts 滤掉的
        行视为已消费（offset 照常前进——传 since_ts 就是明确放弃旧数据）。
    截断/过滤都只影响 records 内容；`next_offset` 恒为本轮**实际检查过**的最后一条成功解析
    记录的行号。文件不存在 → `records: []`、`next_offset: 0`（正常业务结果，非报错）。
    仅本机 loopback 可达（middleware 强制）；同源检查与既有端点同风格（浏览器跨站不可读）。
    """
    if request is not None:
        _require_same_origin(request)
                                        
                                             
    if _account_gate_required():
        raise HTTPException(status_code=403, detail=_TELEMETRY_RELAY_WEB_DETAIL)
    if after < 0:
        raise HTTPException(status_code=400, detail="after 需 ≥ 0。")
    if limit < 1 or limit > _MCP_CALLS_MAX_LIMIT:
        raise HTTPException(status_code=400, detail=f"limit 需在 1–{_MCP_CALLS_MAX_LIMIT} 之间。")
    if max_bytes < 1 or max_bytes > _MCP_CALLS_MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"max_bytes 需在 1–{_MCP_CALLS_MAX_BYTES} 之间。")
    since = _parse_since_ts(since_ts)
    path = _mcp_calls_log_path()
    if not path.is_file():
        return _json_utf8({"ok": True, "records": [], "next_offset": 0, "truncated": False})
    records: list = []
    used_bytes = 0
    last_examined = after
    truncated = False
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno <= after:
                continue
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                                  
            if not isinstance(rec, dict):
                continue
                                                         
            if len(records) >= limit or used_bytes + len(line.encode("utf-8")) > max_bytes:
                truncated = True
                break
            last_examined = lineno
            if since is not None:
                rec_ts = None
                if rec.get("ts"):
                    try:
                        rec_ts = _parse_since_ts(str(rec.get("ts")))
                    except HTTPException:
                        rec_ts = None                                
                if rec_ts is not None and rec_ts < since:
                    continue                                  
            if not rec.get("call_id"):
                rec = dict(rec)
                rec["call_id"] = _legacy_mcp_call_id(line)
                rec["legacy"] = True
            records.append(rec)
            used_bytes += len(line.encode("utf-8"))
    return _json_utf8({"ok": True, "records": records, "next_offset": max(after, last_examined),
                       "truncated": truncated})


class TelemetryAckRequest(BaseModel):
    offset: int = Field(..., ge=0, description="已成功上传的最后一条 MCP 调用记录行号（1-based；0 = 未消费）")


@app.post("/api/telemetry/mcp-calls/ack")
def api_telemetry_mcp_calls_ack(payload: TelemetryAckRequest, request: Request) -> JSONResponse:
    """把已上传游标持久化到 data_root/.userdata/mcp_calls_uploaded.json（批
    改 CAS：**游标只前进**）。

    读-比-写整个在进程内锁内：已存 offset ≥ 请求值时不落盘（回退请求视为已达成——
    中继重放旧进度不会把游标拉回去造成整段重传），响应恒为 max(请求, 已存)。
    写盘走临时文件 + `os.replace` 原子落位，防并发 ack 互相撕扯。供中继/接收端在成功
    消费 records 后推进游标，本地进程重启后仍可续传。
    """
    _require_same_origin(request)
                                        
    if _account_gate_required():
        raise HTTPException(status_code=403, detail=_TELEMETRY_RELAY_WEB_DETAIL)
    target = _mcp_calls_upload_cursor_path()
    with _TELEMETRY_LOCK:
        stored = 0
        if target.exists():
            try:
                prev = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(prev, dict):
                    stored = max(0, int(prev.get("offset") or 0))
            except (OSError, ValueError, TypeError):
                stored = 0                              
        effective = max(payload.offset, stored)
        if effective > stored:
            value = {
                "schema": _TELEMETRY_ACK_SCHEMA,
                "offset": effective,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(temporary, target)
    return _json_utf8({"ok": True, "offset": effective})


@app.get("/api/guide/agent-prompt")
def api_guide_agent_prompt(request: Request = None) -> Response:
    """返回 MCP 接入提示词全文（text/markdown；用户复制给自家 agent 代完成接入用）。

    只读资源（使用教程/MCP安装/agent接入提示词.md）经 runtime_paths 的 resource_root 解析
    （frozen = _MEIPASS；source/portable = 项目根），**绝不拼 cwd 相对路径**——与 /spec/upload 同风格。
    """
    if request is not None:
        _require_same_origin(request)
    prompt_file = RESOURCE_ROOT / "使用教程" / "MCP安装" / "agent接入提示词.md"
    if not prompt_file.is_file():
        raise HTTPException(status_code=404, detail="接入提示词文件不存在。")
    return Response(
        content=prompt_file.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
                                                                          
                                                     
        headers={"Content-Disposition": 'inline; filename="agent-mcp-prompt.md"'},
    )


@app.get("/api/guide/online-prompt")
def api_guide_online_prompt(request: Request = None) -> Response:
    """在线 MCP 接入提示词模板（text/markdown；含 __BIODATA_MCP_URL__/__BIODATA_MCP_TOKEN__
    占位符，由前端在铸币成功后代入真实值再复制给用户）。

    与 api_guide_agent_prompt 同约束：只读资源经 RESOURCE_ROOT 解析、同源闸、ASCII 文件名。
    """
    if request is not None:
        _require_same_origin(request)
    prompt_file = RESOURCE_ROOT / "使用教程" / "MCP安装" / "在线接入提示词模板.md"
    if not prompt_file.is_file():
        raise HTTPException(status_code=404, detail="在线接入提示词模板文件不存在。")
    return Response(
        content=prompt_file.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="online-mcp-prompt.md"'},
    )


@app.get("/api/guide/skill.zip")
def api_guide_skill_zip(request: Request = None) -> Response:
    """把随包 skill 目录现场打成 zip 返回（Content-Disposition 附件；批
    改**确定性构建**）。

    安装版用户不拿源码包时，从这里下载 `biodata-dataset-discovery` 技能目录解压到自家
    agent 的 skills 目录（Skill 安装教程路线 A/B）。zip 在内存构建、零落盘；只读资源目录
    缺失 → 404。arcname 以 `.agents/skills/` 为根，解压后即得可被客户端发现的目录形态。
    确定性：条目按 arcname 排序、ZipInfo 手工构造（date_time 固定 1980-01-01、
    create_system=3、external_attr=0644）、compresslevel=9——同一棵目录树两次构建
    字节完全一致，响应头 `X-SHA256` 给出本包摘要（中继/客户端可校验缓存与完整性）。
    """
    if request is not None:
        _require_same_origin(request)
    skill_dir = RESOURCE_ROOT / ".agents" / "skills" / "biodata-dataset-discovery"
    if not skill_dir.is_dir():
        raise HTTPException(status_code=404, detail="skill 目录不存在。")
    entries: list[tuple[str, bytes]] = []
    for path in sorted(skill_dir.rglob("*")):
        if path.is_file():
            entries.append((path.relative_to(skill_dir.parent).as_posix(), path.read_bytes()))
    entries.sort(key=lambda item: item[0])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arcname, data in entries:
            zinfo = zipfile.ZipInfo(filename=arcname, date_time=(1980, 1, 1, 0, 0, 0))
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.create_system = 3
            zinfo.external_attr = (0o644 & 0xFFFF) << 16
            zf.writestr(zinfo, data)
    payload = buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="biodata-dataset-discovery.zip"',
            "X-SHA256": hashlib.sha256(payload).hexdigest(),
        },
    )


@app.get("/api/compatible")
def api_compatible(uid: str = "", limit: int = 20) -> JSONResponse:
    """N13 元数据兼容分组：给一个数据集找**同物种 + 兼容 chemistry/platform** 的其它数据集。

    **诚实边界**：只回「元数据兼容」（可整合的必要非充分条件），始终附带 caveat，绝不说「可整合」。
    只读、确定性、离线；与 MCP `find_compatible_datasets` 共用 `compatibility.find_compatible` 单一真源。
    检索器/编排/冻结评测从不 import compatibility → 冻结 767 基准结构性不受影响。
    """
    from ..content import compatibility
    u = (uid or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="uid is required")
    try:
        lim = max(1, min(100, int(limit)))
    except (TypeError, ValueError):
        lim = 20
    result = compatibility.find_compatible(u, load_full_corpus(DATA_DIR, PROJECT_ROOT), limit=lim)
    if result is None:
        raise HTTPException(status_code=404, detail="没有找到对应的数据集")
    return _json_utf8({"ok": True, **result})


@app.get("/api/files")
def api_files(
    uid: str = Query(default="", description="dataset_uid（首选）"),
    url: str = Query(default="", description="数据集页面 url（uid 缺失时的回退键）"),
) -> JSONResponse:
    """某数据集的**全部**真实文件下载直链——前端点开卡片时按需拉取。

    默认查询/浏览响应只带代表性主文件（primary），避免每条结果拖着几十个直链。
    这里按 uid（或页面 url）现取该数据集完整 files 列表；查不到 → 空列表（降级，不报错）。
    """
    key = (uid or "").strip() or (url or "").strip()
                                                                         
    from dataset_recommender.corpus import downloads as _dl, inspection as _insp
    _rec = _dl.get(key)
    resolved_uid = _rec.get("dataset_uid") if _rec else None
    primary = primary_url(key)
    items: list[dict[str, Any]] = []
    for f in files_for(key):
        if not isinstance(f, dict):
            continue
        download_url = f.get("download_url")
        if not download_url:
            continue
                                                                         
        st = _insp.status_for(resolved_uid, download_url) if resolved_uid else None
        items.append(
            {
                "title": f.get("title") or f.get("filename") or "文件",
                "filename": f.get("filename") or "",
                "category": f.get("category") or "",
                "pipeline": f.get("pipeline") or "",
                "size_human": f.get("size_human") or "",
                "bytes": f.get("bytes") or 0,
                "download_url": download_url,
                "is_primary": bool(primary) and download_url == primary,
                "problem": bool(st and st.get("problem")),
                "problem_reason": (st or {}).get("problem_reason"),
                "last_verified": (st or {}).get("last_verified"),
            }
        )
                                                  
    items.sort(key=lambda x: not x["is_primary"])
    return _json_utf8({"ok": True, "count": len(items), "files": items})


                                                                                       

class DownloadPlanRequest(BaseModel):
    """POST /api/download/plan 入参。`uids: Any` 故意放宽形状、端点内手工校验成 400：
    pydantic 的 list[str] 严格校验会给 422，与「坏入参 → 400」的契约口径不一致。"""
    uids: Any = None


class DownloadStartRequest(BaseModel):
    uids: Any = None


class DownloadCancelRequest(BaseModel):
    job_id: str = ""


class DownloadUpdateRequest(BaseModel):
    """POST /api/download/update 入参（dl-auto-1 在途增删）。
    `add/remove: Any` 故意放宽形状、端点内手工校验成 400。"""
    add: Any = None
    remove: Any = None


def _sanitize_download_uids(raw: Any) -> list[str]:
    """入参严格校验：uids 必须是非空字符串数组；去重、去空白。坏形状一律 400。

    体积闸：新增最大数量上限（`_MAX_DOWNLOAD_UIDS`，超限 422）——此前 uids 只要求非空，
    攻击者可控页面可一次性塞成千上万个编号，plan/start 都会随之放大 CPU/内存/磁盘。
    """
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="uids 必须是非空数组（数据集编号列表）。")
    if len(raw) > _MAX_DOWNLOAD_UIDS:
        raise HTTPException(
            status_code=422,
            detail=f"uids 数量超过上限（最多 {_MAX_DOWNLOAD_UIDS} 个）；请分批下载。",
        )
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            raise HTTPException(status_code=400,
                                detail=f"uids 里的每一项必须是数据集编号字符串，收到 {type(value).__name__}。")
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    if not out:
        raise HTTPException(status_code=400, detail="uids 里的数据集编号不能全为空字符串。")
    return out


_DOWNLOAD_ERROR_STATUS = {
    "bad_param": 400,
    "no_downloadable": 400,
    "disk_space_insufficient": 507,
    "job_conflict": 409,
    "job_not_running": 409,
    "unknown_job": 404,
}

                                                 
                                               
_MAX_DOWNLOAD_UIDS = 100

                                                       
_DOWNLOAD_JOB_NOT_FOUND_DETAIL = "没有这个下载任务：{job_id}。"


def _download_error_response(exc: BaseException) -> Response:
    code = getattr(exc, "code", "bad_param")
    status = _status_for(code, _DOWNLOAD_ERROR_STATUS)
    return _json_utf8({"ok": False, "code": code, "message_zh": str(exc)}, status_code=status)


def _reject_server_side_download() -> None:
    """公网护栏硬化：护栏模式下服务端代下数据整体关闭——真下载会把数据集
    拉进容器 home（不在 /data 持久卷），多用户公网形态语义错误且可被刷磁盘；网页版走任务包
    （客户端直连原站）。闸关（本机形态）零影响。"""
    if _account_gate_required():
        raise HTTPException(status_code=403, detail="网页版暂不支持服务端代下数据，请使用任务包。")


@app.post("/api/download/plan")
def api_download_plan(payload: DownloadPlanRequest, request: Request) -> Response:
    """真下载第一步：uids → 可下载清单（不落盘、不下载、不起线程，零网络）。

    响应：{ok, items:[{dataset_uid, dataset_title, source, tier, page_url, bytes,
    files:[{filename,url,bytes}]}], total_bytes, unsupported:[{dataset_uid,title,reason}]}。
    """
    _require_same_origin(request)
    _reject_server_side_download()
    from ..corpus import download_manager as DM
    uids = _sanitize_download_uids(payload.uids)
    plan = DM.build_download_plan(uids)
    return _json_utf8({"ok": True, "items": plan["items"], "total_bytes": plan["total_bytes"],
                       "unsupported": plan["unsupported"]})


@app.post("/api/download/start")
def api_download_start(payload: DownloadStartRequest, request: Request) -> Response:
    """真下载第二步：预检（无可下载项 400 / 磁盘不足 507 / 已有任务 409）→ 建目录 → 起线程。

    响应：{ok, job_id, dir, total_bytes}；目录 = ~/Downloads/BioData数据-<时间戳>/，
    每个数据集一个子文件夹（编号_标题），根目录随进度写 README.txt + manifest.tsv。
    """
    _require_same_origin(request)
    _reject_server_side_download()
    from ..corpus import download_manager as DM
    uids = _sanitize_download_uids(payload.uids)
    try:
        job = DM.start_job(uids)
    except DM.DownloadManagerError as exc:
        return _download_error_response(exc)
    return _json_utf8({"ok": True, "job_id": job["job_id"], "dir": job["dir"],
                       "total_bytes": job["total_bytes"]})


@app.get("/api/download/status")
def api_download_status(job: str = Query(default="")) -> Response:
    """真下载状态轮询：{ok, ...状态 dict}；任务不存在 → 404。"""
    _reject_server_side_download()
    from ..corpus import download_manager as DM
    job_id = (job or "").strip()
    state = DM.get_status(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=_DOWNLOAD_JOB_NOT_FOUND_DETAIL.format(job_id=job_id or "(空)"))
    return _json_utf8({"ok": True, **state})


@app.post("/api/download/cancel")
def api_download_cancel(payload: DownloadCancelRequest, request: Request) -> Response:
    """取消下载：置取消标志，chunk 间/文件间停手；已保留的 .part 可续传。

    响应：{ok, state}；任务不存在 → 404。
    """
    _require_same_origin(request)
    _reject_server_side_download()
    from ..corpus import download_manager as DM
    job_id = (payload.job_id or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id 不能为空。")
    state = DM.cancel_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=_DOWNLOAD_JOB_NOT_FOUND_DETAIL.format(job_id=job_id))
    return _json_utf8({"ok": True, "state": state["state"]})


@app.post("/api/download/update")
def api_download_update(payload: DownloadUpdateRequest, request: Request) -> Response:
    """在途增删（dl-auto-1）：对**当前运行中的**下载任务做增量 add/remove。

    入参 `{add: [uid...], remove: [uid...]}`（至少一个非空）。语义：
      - remove 排队中条目 → 跳过；remove 正在下载的条目 → 中止该数据集当前文件并清理它的
        未完成部分（.part 与子目录）、继续队列下一条；remove 已完成条目 → 409 级拒绝（如实说）。
      - add → 追加到当前 job 队列尾部（不可下载的如实标注）。
    无运行中任务 → 409 `job_not_running`；坏入参 → 400。响应除增删结果汇总外带当前快照，
    供前端直接刷新队列与进度。
    """
    _require_same_origin(request)
    _reject_server_side_download()
    from ..corpus import download_manager as DM
    add = _sanitize_download_uids(payload.add) if payload.add else []
    remove = _sanitize_download_uids(payload.remove) if payload.remove else []
    try:
        result = DM.update_job(add=add, remove=remove)
    except DM.DownloadManagerError as exc:
        return _download_error_response(exc)
    return _json_utf8({"ok": True, **result})


from .workspace_api import build_workspace_router

app.include_router(build_workspace_router(
    project_root=lambda: PROJECT_ROOT,
    principal=_utterance_principal,
    require_origin=_require_same_origin,
    service=_get_conversation_service,
))


from .conversation_runs_api import build_runs_router
app.include_router(build_runs_router(
    runtime=_get_conversation_runtime,
    principal=_utterance_principal,
    require_origin=_require_same_origin,
))

from .conversation_api import build_conversation_router


def _owned_replay_config(request, payload):
    from .llm_admission import materialize
    if not payload.model_fields_set & {'provider', 'use_llm', 'mock_llm', 'api_key', 'base_url', 'model'}:
        return materialize(config_root=CONFIG_ROOT, loader=load_llm_config, lock=ENV_LOCK)
    provider, base, mock, enabled = _resolve_llm_request(payload.provider,
        use_llm=True if payload.use_llm is None else payload.use_llm,
        mock_llm=bool(payload.mock_llm), base_url=payload.base_url)
    return _materialize_request_llm_config(provider, use_llm=enabled, mock_llm=mock,
        api_key=payload.api_key, base_url=base, model=payload.model)


def _owned_replay_quota(request, payload, cfg):
    _gate_llm_quota(request, cfg=cfg, provider=cfg.provider, use_llm=cfg.enable_llm,
        mock_llm=cfg.mock_llm, api_key=payload.api_key)


app.include_router(build_conversation_router(
    service=_get_conversation_service, principal=_shared_principal,
    require_origin=_require_same_origin,
    configure=_owned_replay_config, quota=_owned_replay_quota,
))

from .learning_api import build_learning_router
app.include_router(build_learning_router(
    service=_get_conversation_service, principal=_shared_principal,
    endpoint=lambda request, base, model, provider: _userdata_partition(request, base, model, provider)[1],
    require_origin=_require_same_origin,
    configure=_owned_replay_config, quota=_owned_replay_quota,
))

from .knowledge_api import build_knowledge_router
app.include_router(build_knowledge_router(
    service=_get_conversation_service, principal=_shared_principal,
    endpoint=lambda request, base, model, provider: _userdata_partition(request, base, model, provider)[1],
    require_origin=_require_same_origin,
    configure=_owned_replay_config, quota=_owned_replay_quota,
))
