                       
"""BioData Agent · 本地 stdio MCP 服务器（零改动封装现有确定性管线）。

把「中文自然语言 → 单细胞数据集」管线暴露成 25 个 MCP 工具（在线23个）：
19 个基础工具（16 个只读 + 3 个写盘工具：upload_dataset 写外部库、provision_dataset 写调用方指定目录、
curate_datasets 管护外部库 upload_* 与回收站）+ 6 个高层 Agent 工具（agent_chat / agent_conversations /
agent_projects / agent_task / agent_replay / agent_learning，持久会话、项目、复盘与经授权的独立学习；
调用会消耗模型额度，学习批准/替代须网页真人逐条确认），
供 Claude Code / Codex 这类支持 MCP 的 agent 直接调用（与网页端能力对齐）：

  1. recommend_datasets(query, sources=None, top_k=None, recall/use_llm/rerank=opt-in,
     facet_filters/suppressed_constraints/date_from/date_to=opt-in)  —— 完整推荐管线
     （约束抽取 fail-closed → 硬过滤 → 存活集排序 → 反捏造终检）。**默认全程确定性、离线、无需 key**；
     质量杠杆（recall 本地重排 / use_llm 润色 / rerank）与前端「数据细化 / 忽略已命中 / 发表时间范围」
     （facet_filters / suppressed_constraints / date_*）均 opt-in、默认关。stdio 稳定性见文件头注释。
  2. get_file_manifest(dataset_uid)  —— 【核心能力】某数据集的真实文件级下载清单：fileset / 角色 /
     直链 / md5 / 大小 / pipeline / FASTQ。竞品只给到数据集页面，这里给到每个文件。
  3. parse_constraints(query)  —— 只解析、不检索：看清系统从这句话里抽出了哪些**硬约束**、
     是否 fail-closed 弃权（abstain）。便宜、供父 agent 预检。
  4. browse_datasets(species/platform/source/year=None, limit/offset)  —— 全库浏览（对齐网页 `/api/datasets`）：
     所有来源并列的归一化记录 + 物种/平台/来源/年份分面，按维度轻过滤 + 分页。无需 query。
  5. get_dataset_introduction(uid/url/name/source)  —— 单数据集确定性介绍（对齐网页 `/api/introduction`）：
     只整理现有元数据、不调用 LLM。给 browse/recommend 拿到 uid 后按需取介绍。
  6. upload_dataset(records|path, filename=None, source=None)  —— 【写工具 ①】把一份数据集 JSON
     摄取进外部库 database/external/（对齐网页 `/api/upload`），即时可被 browse/recommend(sources=[…])
     检索到。写入**只进** database/external/、**绝不**碰冻结基准 database/base/ → 774 评测不受影响。
     按用户明确授权补入：打破「只读」但保「离线 / 永不崩 / base 不动」；写盘=非确定性（时间戳文件名）。
  7. biodata_status()  —— 健康自检 + 库存清单（各来源计数、下载索引是否就绪、快照日期）。
     装完照着教程让 agent 调一次这个，就知道服务器活没活、数据挂没挂。
  8. biodata_llm_status(check_connection=False) —— 脱敏检查 LLM API 配置；显式传 True 才联网做最小探测。
  9. assess_dataset_fair(uid/url/name/source) —— 单数据集**复用就绪度**检查 + 「复用公开数据」声明
     （对齐网页 `/api/fair`）：13 项 F/A/I/R 检查 + 一段可放进稿件的英文声明；确定性、离线、不调用 LLM。
     受众是**复用者**（本项目语料 100% 是别人已公开的数据）：每项给的是**你能做的事**
     （去哪核实、稿件里怎么写），不是「请改进你的数据」。
 10. build_reuse_pack(uids) —— 【复用出处清单】N 个复用的公开数据集 → 投稿材料（对齐网页
     `POST /api/reuse-pack`）：英文段落 + 补充表 + 中文待办 + Markdown + RIS/BibTeX 引文导出。keys-only
     入参（只收 dataset_uid、不收数据集内容）。**不是 DAS 主句**：你自己产出的数据那一句本工具
     帮不上也不会写——它需要你未发表的实验信息，本工具按设计不接收。
 11. lookup_identifier(identifier) —— 按标识符精确反查 + 诚实 fail-closed（对齐网页 `/api/recommend`
     的 identifier_lookup）：贴 UUID / E-XXXX-N / ENCSR 实验编号 / 论文 DOI / GSE → 直达本目录记录；
     贴 GSM/SRA 等 Sample 与原始 reads 编号 → 如实告知不在本目录 11 来源收录范围、指向原库，绝不静默返回 0。只读、离线、确定性。
 12. find_compatible_datasets(uid) —— 元数据兼容分组（对齐网页 `/api/compatible`）：给一个数据集找
     同物种 + 兼容 chemistry/platform 的其它数据集。**只回「元数据兼容」（必要非充分），始终附带 caveat，
     绝不说「可整合」**。只读、离线、确定性。
 13. assess_feasibility(query) —— 可行性概览（对齐网页 `/api/feasibility`）：研究问题 → 候选数 + 总细胞量
     **下限** + 物种·平台·年份·来源分布 + 可下载率 + 缺口。总细胞量**必然是下限**（AE 等无细胞数）、显式标注。
     只读、离线、不调用 LLM。
 14. verify_local_assets(directory) —— 本地实验室资产台账（对齐 CLI `scripts/scan_lab_assets.py`）：扫描
     一个本地目录树 → 与文件级清单（10x 基础语料 15119 文件 md5/大小/文件名）比对 → 哪些已知数据集在本地、
     md5 核验通过没有、哪些内容不符。**只算 md5、绝不读数据内容；只读、不联网、不改文件**。诚实边界：md5
     清单只覆盖 10x 基础语料，外部库无单文件清单→无法 md5 核验（见 report.caveat）。
 15. plan_query_edit(query, utterance, …)  —— 接着改条件（对齐网页 `/api/board/plan`）：把「换成小鼠 /
     再加一条：要有 FASTQ / 去掉组织限制」规划成一次具体改动。**只规划、不检索、不装载目录**。
 16. plan_action(utterance, has_results, result_total)  —— 一句话要做什么（对齐网页 `/api/action/plan`）：
     归一化成封闭动词表里的一个动作，**只出计划、不执行**；判定依据保证是用户原话字面子串。
 17. build_task_pack(mode=preview|build, query, …)  —— 一句话任务包（对齐网页 `/api/task-pack/*`）：
     一次检索 → 结果清单 + 下载脚本 + FAIR 自检 + 引文，四件套口径一致；先预览、确认后再产包。
     MCP 只返文本、绝不写盘。
 18. provision_dataset(dataset_uid, dest_dir, scope/max_files/dry_run)  —— 【写工具 ②】把数据集文件
     **真正下载到调用方指定的 dest_dir**（对齐 CLI `scripts/provision_dataset.py`）：https+白名单、
     .part→原子改名、md5/大小核对、.corrupt 留证据。**仅在调用方给定 dest_dir 时写盘，绝不写 database/；
     默认只下 1 个主文件（scope=primary）**，dry_run=True 先给计划不下载。
 19. curate_datasets(action, …, dry_run/confirm_token)  —— 【写工具 ③】对话式数据库管护
     （对齐 Web `/api/curate/*` 与 CLI `scripts/curate_datasets.py`）：list 清点 / import 本地导入 /
     search_online 联网搜索官方源 / remove 回收站式删除 / restore 恢复，管护对象限
     database/external/ 的 upload_* 命名空间。**plan 默认（dry_run=True）零写盘，apply 才写盘/联网**
     且必须回传 confirm_token（重算比对，不一致零写入）。search_online 的 plan 会真实联网并记请求账本。
     **check_updates / sync_updates**（2026-08-22 补齐）：只读检查来源更新 / 检查+自动
     入库复合流（sync 有整任务文件锁，冲突 → `sync_busy`；返回 operation receipt，可经 Web
     `/api/curate/recall` 整次撤回），修复 plan_action 已宣称而工具不接的 plan→execute 断链。
 20. agent_chat(message, conversation_id/request_id/expected_revision/project_id/state_patch/ask_choice/
     wait_seconds)  —— 【高层】直接对知识库 Agent 说一句话，不必由宿主拼装底层工具：一次调用即
     走完整轮（解析→检索→可选 LLM），并自己保存消息、结果帧与检索现场。同 `request_id` 重试即幂等
     （不重复执行写工具）；后台未跑完时返回 `run_id` + `pending=true`，用 agent_task 续读，**不要另起请求重跑**。
     `state_patch` 只接受 query/sources/search_params，`ask_choice` 只取当前选择卡的真实 key。
     对齐 Web `/?conv=` 同一现场：同实例 + 同身份即同 scope（普通 MCP 不复制图谱 GUI）。
 21. agent_conversations(action, conversation_id/expected_revision/offset/limit/after_message/after_frame/
     include_archived/frame_id/title/import_id/data)  —— 【高层】会话 list/get/create/archive/restore/
     rename/delete/select_frame/branch/import。list/get 有分页游标；写动作携 `expected_revision`
     防旧客户端覆盖新轮；`branch` 必须给真实 frame_id；`import` 只提交用户**逐条预览确认**的一条旧史
     并给稳定 import_id——**绝不自动扫描本机历史**（导入是幂等的，原件不动；保存会话 ≠ 训练 consent）。
 22. agent_projects(action, project_id/name/color/conversation_id/expected_revision/project_ids)  ——
     【高层】共享项目 list/create/update/archive/restore/reorder/assign，与 Web 项目机制、学习作用域同源；
     归档项目保留会话；不上传记忆。
 23. agent_task(action=status/events/cancel/resolve, run_id/request_id/after)  —— 【高层】读写
     已有后台任务的运行现场：`status` 读快照、`events` 按 `seq` 游标增量、`resolve` 按 `request_id`
     找回、`cancel` 只在安全边界停止（已完成操作不会撤销；`interrupted` 不自动重演）。
 24. agent_replay(conversation_id, action=status/points/annotate/snapshot/annotate_snapshot/seen/chat/
     digest, data)  —— 【高层】复盘（own replay）：`points` 返回结构化图 + 出处 + 标注，`snapshot`
     打包标注供继续讨论，`chat`/`digest` 会使用可用模型与同一额度。`data` 仅允许
     pid/text/snap_id/observed_version/observed_pids（strict `{data: …}` 信封，旧 anonymous key 保留）。
     不需要客户端互传 key；仅本人共享会话可复盘。
 25. agent_learning(action=status/control/trigger/candidates/review) —— 经scope授权的独立ReAct学习，
     默认关闭，每调用独立额度；候选/审核反馈与真实trace持久可回查。AI关/trial/无配置不后台调用；
     重启等待同身份端点授权配置。批准/替代须网页真人逐条确认，MCP不能代批；暂停/删除阻止迟回候选。

设计原则（与主线「永不崩→降级」合同一致）：
  * 传输 = stdio：由客户端本地 spawn 一个进程；**默认检索/解析不联网**。旧的 19 个基础工具仍只在
    调用方显式开启 LLM 参数或 biodata_llm_status(check_connection=True) 时，才向已配置的 API 发请求。
    高层 Agent 五工具是**如实声明的例外**：agent_chat 与 agent_replay 的 chat·digest 会主动使用
    可用模型（服务端配置的，或调用方本轮的请求级配置）并计入与 Web 同一身份额度；经它们规划的轮次
    也可执行调用方**本轮已授权**的联网工具（如 curate_datasets 的 search_online / check_updates /
    sync_updates），执行与否在结果里如实回显。**这不是预先授权**：保存会话/项目/复盘只是给你自己
    看的持久现场，不等于训练 consent，也不构成对未来联网或写盘的授权。
  * 非法请求与内部异常转换为 ToolError（客户端 isError=true），合法业务结果保持 ok=true；
    FastMCP 兜住单次工具错误，不让它掀翻 MCP 进程。
  * 数据路径锚定在**源码位置**（config.py: project_root = 源文件父级），与启动 cwd 无关 →
    Claude Code / Codex 从任意目录 spawn 都能找到 database/ 与下载索引。
  * 重资源（语料、下载索引）一律**懒加载**，import 期不做 I/O → 启动秒开，不触发 Codex 启动超时。
  * 只有三个**语料写盘**工具：upload_dataset（按用户授权补入，**只写** database/external/、绝不碰 database/base/）
    与 provision_dataset（**只写**调用方显式给定的 dest_dir，executor 对 database/base 与在仓数据目录
    fail-closed 拒绝）、curate_datasets（管护写工具，apply 才写 database/external/ 的 upload_* 命名空间
    与 .userdata/recycle/ 回收站；search_online apply 时联网）；其余工具调用不写语料/冻结基准。默认路径不启用 LLM，774 冻结评测的确定性与
    硬违规 0% 不受影响（官方评测走 base-only、永不读外部库，故上传即便入库也不影响基准）。
    高层 Agent 五工具（agent_chat / agent_conversations / agent_projects / agent_task / agent_replay）
    另有一层**持久现场**副作用（会话/项目/任务/复盘落实例数据库，**不写语料、不碰冻结基准**），
    且 agent_chat / agent_replay 的 chat·digest 会使用已配置模型、消耗与 Web 同一身份额度
    （缺模型如实降级，绝不静默假装成功）。
  * 身份与 scope（高层工具专用）：stdio 无 `BIODATA_MCP_ACCESS_TOKEN` 时身份为本机 `local`，
    与同一实例的 Web「本机 shared」同 scope；显式设了该环境变量则**必须**是同一实例令牌库里已有、
    未被撤销的令牌（否则 fail-closed，绝不降级冒用本机身份），**不接受裸 account_id**，
    也不得把线上服务器令牌用于另一套本地库。在线形态（Bearer）高层工具走服务端配置、同一账户额度。
  * 工具调用留痕：每次调用 append 一行脱敏 JSON 到 .userdata/mcp_calls.jsonl（gitignored、零网络、
    写失败静默降级绝不影响调用；env BIODATA_MCP_CALL_LOG=off 关闭，默认开）。
"""
from __future__ import annotations

import datetime
import functools
import inspect
import json
import os
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

                                                                     
                                                            
                                                                         
                                                      
                                                                  
                                                            
for _k, _v in {
    "CUDA_VISIBLE_DEVICES": "",                                         
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}.items():
    os.environ.setdefault(_k, _v)
                                                                                

                                                                     
                                                         
                                                                    
                                                                  
                                                               
                                                                
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2] if (_HERE.parents[2] / "dataset_recommender").is_dir() else _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcp.server.fastmcp import FastMCP              
from mcp.server.fastmcp.exceptions import ToolError              
from mcp.server.transport_security import TransportSecuritySettings              
from pydantic import StrictBool, StrictInt                                              
from dataset_recommender.content.introduction import build_dataset_introduction              
                                                                             
                                                                                
from dataset_recommender.corpus.uploads import (              
    UploadError,
    ingest_dataset,
    new_upload_name,
)
from dataset_recommender.secret_patterns import redact as _redact_secret_values              
from dataset_recommender.app.runtime_paths import get_app_paths, instance_data_dir_for              
from dataset_recommender.app import mcp_tokens                                          
                    
                                                                    
from dataset_recommender.app.limits import MAX_DATASETS_LIMIT              
                                                   
                                                  
from dataset_recommender.app.request_validation import (              
    ParamValidationError,
    validate_date_window,
    validate_iso_date,
    validate_query,
    validate_sources,
)

                                                               
                                                                               
                                                         
                                                                           
                                                                
                                                            
try:
    from mcp.server.fastmcp.utilities import func_metadata as _fm
    from pydantic import ConfigDict as _ConfigDict
    _base_cfg = dict(getattr(_fm.ArgModelBase, "model_config", {}) or {})
    _base_cfg["extra"] = "forbid"
    _fm.ArgModelBase.model_config = _ConfigDict(**_base_cfg)
except Exception:                                   
    pass
                                                                                

                                                                
                                                       
_LEDGER_SNAPSHOT = "2026-07-10"

                                                  
                                                                               
                                                                          
                                                            
                                                                          
                                                                 
                                                                      
                                                                                
                                                                        
                                                                            
                                                                                  
                                                     
                                                                           
                                                        
                                                                    
                                                                                         
                                                                                   
                                                                                           
                                                                                                
                                                                                     
                                                                 
                                                                             
                                                                                  
                                            
                                                          
                                                                                    
                                                              
                                                                                 
                                                          
                                                                                   
                                                                              
                                                                                      
                                                                                 
                                                                          
                                                                      
                                                                                
                                                                 
                                                                                
                                                                                   
                                                           
                                                                        
                                                                 
                                                                                
                                                                                     
                                                         
                                                                                                     
                                                         
                                                                              
                                                                                   
                                                                          
                                                                                   
                                                                      
                                                                               
                                                              
                                                                               
                                                                                    
                                                                             
                                                                                                
                                                                    
                                                                                        
                                                  
                                                                             
                                                                                      
                                                                     
                                                                                    
                                                                                    
                                                                                    
                                                               
                                                                     
                                                                                                    
                                                                                        
                                                                                         
                                                                         
                                                                           
                                                                                            
                                                                                               
                                                                                     
                                                                         
                                            
                                                                               
                                                             
                                                                           
                                                                            
                                                                
                            
                                                                         
                                                      
                                                              
                                                             
                                              
                                                              
                                                             
                                                                
                                                                                 
                                                                        
                                                                                        
                                                                                     
                                                     
                                    
                                                       
                                                           
                                                        
                                                            
                                                                          
                                                         
                                                              
                                                                                 
                                                                                   
                                                                   
                                                                                   
                                                                     
                                                              
                                                                              
                                                                        
                                                                
                                                                                 
                                                                         
                                       
                                                                   
                                                                  
                                                                         
                                                                              
                                                                               
                                                                             
                                                                      
                                                                              
                                                                                    
                                                                                        
                                                                                          
                                                                  
                                                                               
                                                             
                                                                      
                                                                        
                                                                    
                                   
                                                                        
                   
                                                                   
                                                                                 
                                                                   
                                                               
                                                 
                                                                         
                                                        
                                                                                   
                                                                                              
                                                                           
                                                               
                                                             
                                                                   
                                                                       
                                                                           
                                                                       
                                         
                                                                              
                                                    
                                                                                 
                                                                              
                                                                 
                                                                          
                                                            
                                                                
_SERVER_VERSION = "1.38.0"


                                                                                      
                                                                   
                                                                                
                                                           
                                                                  
                                                           
                                                       
 
                                  
_ONLINE_DISABLED_TOOLS: dict[str, str] = {
    "provision_dataset": "它把文件下载到 dest_dir——这是调用方**本机**路径语义；在线形态下写到的是服务器磁盘，无意义且绝不开放",
    "verify_local_assets": "它扫描调用方**本机**目录树与本地资产比对；在线形态下扫到的是服务器文件系统，无意义且绝不开放",
}

                              
_ONLINE_LOCAL_PATH_PARAMS: dict[str, tuple[str, ...]] = {"upload_dataset": ("path",)}

                                                  
                                                    
_ONLINE_LLM_SAFE: dict[str, dict[str, object]] = {
    "recommend_datasets": {
        "use_llm": False, "rerank": "off", "rerank_top_n": None,
        "auto_allow_llm": False, "strategy": "fixed",
    },
    "get_dataset_introduction": {"llm": False},
    "biodata_llm_status": {"check_connection": False},
}


def _enforce_online_policy(name: str, arguments: dict) -> None:
    """在线形态工具策略：禁用工具 / LLM 参数被推离安全档 → 显式 ToolError（不静默偏离）。"""
    reason = _ONLINE_DISABLED_TOOLS.get(name)
    if reason is not None:
        raise ToolError(f"online_tool_disabled: {name} 在线形态不可用——{reason}。")
    for param in _ONLINE_LOCAL_PATH_PARAMS.get(name, ()):
        if arguments.get(param):
            raise ToolError(
                f"online_path_disabled: 在线形态不接受 {param} 本机路径；请直接提交结构化内容。")
    for param, safe_value in (_ONLINE_LLM_SAFE.get(name) or {}).items():
        value = arguments.get(param, safe_value)
        if value is None or value == safe_value:
            continue
        raise ToolError(
            f"online_llm_disabled: 在线 MCP 形态为纯确定性服务（不消耗服务端 LLM 额度），参数 "
            f"{param}={value!r} 不被接受；请去掉该参数后重试，或改用本地版获得完整能力。")


class _ScopedFastMCP(FastMCP):
    """两用 FastMCP：stdio 直通；在线（HTTP request 在调用上下文里）执行账户作用域绑定与在线策略。"""

    async def call_tool(self, name, arguments):
        request = None
        try:
            ctx = self.get_context()
            request = ctx.request_context.request if ctx.request_context else None
        except ValueError:
            request = None
        if request is None:
            from dataset_recommender.app.mcp_agent_tools import bind_identity, stdio_identity
            from dataset_recommender.app.conversation_runs import RunError
            from dataset_recommender.corpus.patch_package import bind_patch_scope
            try:
                owner, username = stdio_identity(get_app_paths().data_root)
            except RunError as exc:
                raise ToolError(f'{exc.code}: {exc.message}') from None
            with bind_identity(owner, username), bind_patch_scope(None if owner == 'local' else owner):
                return await super().call_tool(name, arguments)
        _enforce_online_policy(name, arguments)
        account_id = request.scope.get("biodata.mcp_account_id", "")                           
        from dataset_recommender.corpus.patch_package import bind_patch_scope            
        from dataset_recommender.llm.scope_gate import force_llm_off
        from dataset_recommender.app.mcp_agent_tools import AGENT_TOOLS, bind_identity
        if name in AGENT_TOOLS:
            if not account_id:
                raise ToolError("auth_required: 高层 Agent 需要有效传输身份。")
            with bind_patch_scope(account_id), bind_identity(account_id, request.scope.get("biodata.mcp_username", "")):
                return await super().call_tool(name, arguments)
        with bind_patch_scope(account_id or None), force_llm_off():
            return await super().call_tool(name, arguments)

    async def list_tools(self):
        """在线形态的 tools/list 直接摘掉禁用工具——客户端连「看得到但一调就错」的机会都没有；
        stdio（无请求上下文）返回全集，与历史一致。"""
        tools = await super().list_tools()
        request = None
        try:
            ctx = self.get_context()
            request = ctx.request_context.request if ctx.request_context else None
        except ValueError:
            request = None
        if request is None:
            return tools
        return [t for t in tools if t.name not in _ONLINE_DISABLED_TOOLS]


from dataset_recommender.domain.registry import get_domain as _get_domain

mcp = _ScopedFastMCP(_get_domain().branding.mcp_server_name, stateless_http=True, json_response=True,
                     transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
                                                             
                                                         
                                                                        
                                                                     
                                                                  
                               

                                              
_WF = None
_SETTINGS = None


def _settings():
    global _SETTINGS
    if _SETTINGS is None:
        from dataset_recommender.llm.config import get_settings
        _SETTINGS = get_settings()
    return _SETTINGS


def _workflow():
    global _WF
    if _WF is None:
        from dataset_recommender.app.workflow import DatasetRecommendationWorkflow
        _WF = DatasetRecommendationWorkflow()
    return _WF


                                              
_RECALL_CHOICES = ("off", "cross_encoder", "dense")                  
_RERANK_CHOICES = ("off", "llm")                                       

                                                                           
                                                              
                                                 
_DEFAULT_RECALL = (os.environ.get("BIODATA_MCP_RECALL", "off") or "off").strip().lower()
if _DEFAULT_RECALL not in _RECALL_CHOICES:
    _DEFAULT_RECALL = "off"


                                                                         
                                                     
                                                                   
                                                             
                                                      
                                                     
                                                                            
                                                      
def _bad_request(code: str, hint: str) -> ToolError:
    return ToolError(f"{code}: {hint}")


def _clip(value: object, limit: int = 120) -> str:
    """错误消息回显入参的统一截断，2026-08-21）：调用方贴多长的输入，
    错误串曾原样回多长（超长 uid/枚举值→响应放大器）。统一截到 120 字符加省略号；
    非字符串先取 repr（与历史 {value!r} 口径一致）。"""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


                                                                
                                                         
                                                                             
                                              
_MAX_TOP_K = 100


def _require_query(query: str) -> None:
    """query 四道闸（空/控制字符/纯符号/超长）。

    批起闸体下沉为薄委托：真源在 ``app/request_validation.validate_query``
    （与 Web /api/recommend·/api/interpret 同一份，两端口径同源同措辞）；此处只做
    ParamValidationError → ToolError 的 MCP 翻译。历史裁决记录见该模块 docstring
    （反馈：\0/零宽/双向字符可混入、纯符号输入被包装成推荐、无长度上限）。"""
    try:
        validate_query(query)
    except ParamValidationError as exc:
        raise _bad_request(exc.code, exc.hint) from exc


def _validate_enum(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise _bad_request("bad_param", f"{name} 只能是 {list(choices)} 之一，收到「{_clip(value)}」。")


def _validate_positive(name: str, value: "int | None") -> None:
                                                                       
                                                  
                                                                            
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _bad_request("bad_param", f"{name} 必须是正整数（≥1）或省略，收到「{_clip(value)}」。")
                                    
    if value > _MAX_TOP_K:
        raise _bad_request(
            "bad_param",
            f"{name}={value} 超过上限 {_MAX_TOP_K}（防响应放大/吃爆上下文）；请用 ≤{_MAX_TOP_K} 的值并收窄查询。",
        )


def _validate_sources(sources: "list[str] | None") -> None:
                                                                    
                                                                           
                                                                  
    if not sources:                                           
        return
    from dataset_recommender.corpus.corpus import known_source_values
    s = _settings()
    try:
        validate_sources(sources, known=known_source_values(s.data_dir, s.project_root))
    except ParamValidationError as exc:
        raise _bad_request(exc.code, exc.hint) from exc


def _require_iso_date(value: "str | None", *, name: str) -> str:
    """发表时间入参：空 → ""（不限）；否则必须是格式与日历都合法的 YYYY-MM-DD，非法 → bad_param。

    2026-08-08 产品方拍板：MCP 与 Web 日期口径**拉齐为严格档**（登记册清零）——
    旧宽松口径「非年份打头一律忽略＝不限」把用户给了的条件悄悄丢掉；日历不存在的日期
    （如 2026-02-30）曾按字面值透传检索层。批起闸体下沉：真源在
    ``app/request_validation.validate_iso_date``（与 Web 同一份，原两份实现逐行同构）。"""
    try:
        return validate_iso_date(value, name=name)
    except ParamValidationError as exc:
        raise _bad_request(exc.code, exc.hint) from exc


def _validate_date_window(date_from: str, date_to: str) -> None:
    """倒挂窗口（from > to）→ bad_param 当场点名（真源在 app/request_validation.validate_date_window，
    批起 Web 的 recommend/feasibility/task-pack 与本端同一份闸）。"""
    try:
        validate_date_window(date_from, date_to)
    except ParamValidationError as exc:
        raise _bad_request(exc.code, exc.hint) from exc


def _enrich_candidates(candidates: list[dict]) -> None:
    """Attach shared introductions and data-quality caveats to candidates in place."""
    from dataset_recommender.domain.registry import domain_catalog
    from dataset_recommender.corpus.data_quality import record_caveats
    from types import SimpleNamespace
    for c in candidates:
        if not isinstance(c, dict):
            continue
        rec = SimpleNamespace(
            species=c.get("species", ""),
            tissue=c.get("tissue", ""),
            description=c.get("description", ""),
        )
        c["caveats"] = record_caveats(rec, domain_catalog())
        c["introduction"] = build_dataset_introduction(c)


                                                                     
                                                                     
                                                      
                                                                                
                                                        
                                                                 
                                                              
                                                                            
                                                        
                                                
                                                              
                                                      
_CALL_LOG_SCHEMA = "biodata-mcp-calls/v1"                                         
_CALL_LOG_ENV = "BIODATA_MCP_CALL_LOG"
                                                                   
                                                                  
_CALL_LOG_FILE = instance_data_dir_for(get_app_paths().data_root, ".userdata") / "mcp_calls.jsonl"
                                                 
                                                             
                                                            
                                                       
                                 
_SENSITIVE_PARAM_RE = re.compile(
    r"(api_?key|token|secret|password|passwd|credential|authorization|auth_?header)", re.I)
_MAX_PARAM_CHARS = 400                                         


def _param_snapshot(fn, args: tuple, kwargs: dict) -> dict:
    """入参摘要：按签名绑定参数名 + 双层脱敏（名 + 值）+ 截断。绑定失败（直接 Python 调用给了意外参数）退回 kwargs 原样脱敏。"""
    try:
        items = list(inspect.signature(fn).bind_partial(*args, **kwargs).arguments.items())
    except (TypeError, ValueError):
        items = list(kwargs.items())
    out: dict = {}
    for name, value in items:
        if _SENSITIVE_PARAM_RE.search(str(name)):
            out[str(name)] = "<redacted>"
            continue
        if isinstance(value, str):
            v = _redact_secret_values(value) or ""                   
            out[str(name)] = v if len(v) <= _MAX_PARAM_CHARS else v[:_MAX_PARAM_CHARS] + "…"
            continue
        try:
            v = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            v = f"<{type(value).__name__}>"
        v = _redact_secret_values(v) or ""                                   
        out[str(name)] = v if len(v) <= _MAX_PARAM_CHARS else v[:_MAX_PARAM_CHARS] + "…"
    return out


def _log_error_code(exc: BaseException) -> str:
    """ToolError 消息约定是「code: hint」→ 取机器码；其余异常记类型名（脱敏，不带异常正文）。"""
    if isinstance(exc, ToolError):
        head = str(exc).split(":", 1)[0].strip()
        if re.fullmatch(r"[a-z][a-z0-9_]*", head):
            return head
        return "tool_error"
    return f"exception:{type(exc).__name__}"


def _internal_error(exc: Exception) -> ToolError:
    """内部异常 → 客户端可见的 ToolError（**脱敏**：只报异常类型名，不带异常正文）。

    为什么脱敏：异常正文常带本机绝对路径（如 PermissionError 的 Errno 13 附带语料目录全路径，
    对抗评审 p0-record -c 实测外泄），MCP 客户端不该看到。完整堆栈打到本进程 stderr
    （stdio 协议里 stderr 只在服务器侧可见，不进协议帧）——排查看服务器日志，客户端只见通用提示。
    机器码保持 internal_error 不变：`_log_error_code` 与客户端按码分类的逻辑不受影响。
    """
    traceback.print_exc()
    return ToolError(f"internal_error: {type(exc).__name__}（细节见本服务器 stderr 日志）")


def _record_tool_call(tool: str, params: dict, start: float, *, ok: bool, error: "str | None") -> None:
    """append-only 落盘一行；env 关闭即不写；任何 I/O 失败静默降级——日志绝不能影响工具调用本身。"""
    try:
        if (os.environ.get(_CALL_LOG_ENV, "on") or "on").strip().lower() in ("off", "0", "false", "no"):
            return
        record = {
            "schema": _CALL_LOG_SCHEMA,
            "call_id": uuid.uuid4().hex,                               
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "tool": tool,
            "params": params,
            "duration_ms": round((time.monotonic() - start) * 1000, 1),
            "ok": ok,                                                         
            "error": error,                                                
        }
        _CALL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _CALL_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _logged(fn):
    """工具入口统一留痕：时间戳(ISO8601 UTC)/工具名/参数摘要(脱敏)/耗时 ms/ok-isError/错误码。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            _record_tool_call(fn.__name__, _param_snapshot(fn, args, kwargs), start,
                              ok=False, error=_log_error_code(exc))
            raise
        _record_tool_call(fn.__name__, _param_snapshot(fn, args, kwargs), start, ok=True, error=None)
        return result
    return wrapper


                                                                   
@mcp.tool()
@_logged
def recommend_datasets(
    query: str,
    sources: list[str] | None = None,
    top_k: StrictInt | None = None,
    recall: str | None = None,
    use_llm: bool = False,
    rerank: str = "off",
    rerank_top_n: StrictInt | None = None,
    strategy: str = "fixed",
    auto_parse: bool = True,
    auto_allow_llm: bool = False,
    facet_filters: list[dict] | None = None,
    suppressed_constraints: list[str] | None = None,
    lenient_dims: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """中文自然语言 → 单细胞/生信数据集推荐。

    默认（不传下面任何质量参数）跑完整检索流程：抽取查询条件（能做就做、做了如实说清做法）→
    硬过滤 → 对通过过滤的结果排序 → 输出前逐条核对（结果里的每项说法都必须在元数据里有依据），
    **全程确定性、离线、无需任何 API key**。
    下面几个可选杠杆是 opt-in：默认全关＝确定性；只在你需要时打开。

    参数：
      query:   一句中文（或中英混合）查询，如「人类肺癌的单细胞数据，要有 FASTQ 原始数据」。
      sources: 数据来源过滤。None（默认）= 只用 10x Genomics 基础语料（有文件级清单、字段最全）。
               传来源名列表可扩展到其它公开目录，如 ["10x Genomics","CELLxGENE Discover"]；
               合法来源名调 biodata_status() 查看。注意：外部来源的元数据完备度较低，
               字段显示 unknown 只表示来源没标注，**不等于**不满足该条件。
      top_k:   返回条数上限，None = 用配置默认（10）；**硬上限 100**（防响应放大），超限报 bad_param。
               须为严格整数（不接受 "3" / true 等强转）。
      recall:  召回/重排后端，opt-in。
               "off"（默认）= 纯词面确定性排序；
               "cross_encoder" = **本地** cross-encoder 重排，**离线、无需 key**，但语义更贴合中文查询；
               "dense" = 本地稠密向量召回。
               ⚠️ 注意①：cross_encoder / dense 需先 `py scripts/fetch_embedding_model.py --cross-encoder`
               （或对应嵌入模型）**装本地模型**；未装时**静默回退规则序**（结果=确定性词面序、不报错、
               不联网）。meta.recall_used 会回显你请求的 backend，装没装看 meta.notes。
      use_llm: 是否对结果做 **LLM 中文润色**。默认 False。
               ⚠️ 告诫②：True 需 **API key + 联网**，且输出**非确定性**；缺 key 时 fail-open 回退
               确定性结果（不崩），meta.llm_used 反映是否真的用上。
               ⚠️ 告诫③：**本工具的调用方本身就是 LLM**——通常**无需** use_llm 润色；直接读结构化
               `candidates`（含 score / matched_fields / 下载直链）自己叙述，更快、更可控、可确定复现。
      rerank:  "off"（默认）| "llm"（**LLM 重排**存活集）。
               ⚠️ 同告诫②：llm 需 key + 联网、非确定性；缺 key 静默回退原序。
      rerank_top_n: 仅对存活集前 N 条做重排；None = 用默认。
      strategy: "fixed"（默认）| "auto"。
               "fixed" = 用上面显式的 recall/rerank（逐位复刻旧行为）。
               "auto" = 综合候选压力、自由语义信息与可用后端，选择纯规则、本地语义、LLM 或两层组合；
               紧查询保持确定性序不动。auto **覆盖** recall/rerank
               的显式取值。⚠️ MCP 下 auto 只用**已预热**的本地后端（需在服务器 env 设 BIODATA_MCP_RECALL 并重启，
               否则回退确定性词面序）。默认不引 LLM 路径；只有同时传 auto_allow_llm=true、检测到 API 配置，
               且查询确属高候选压力+丰富自由语义时，才可能选「语义排序→LLM 精排」。决策见 meta.strategy。
      auto_parse: True（默认）= 自动从 query 识别高辨识度数据来源专名，并与时间/实体规则一起返回解释；
               来源排除语义会触发安全守卫、不自动反选。False = query 原样交给规则解析器。
      auto_allow_llm: False（默认）= strategy=auto 保持离线；True 才授权自动策略在已配置 API 时选择
               LLM 重排。该参数不等于 use_llm，后者只控制推荐说明润色。
      facet_filters: 分面细化（对齐网页「数据细化」侧栏；默认 None＝不细化）。在**当前命中集**上按维度做
               精确等值后置收窄：传 [{"dim": 维度, "value": 分面键}, ...]，跨维度 AND、同维度多值 OR。
               合法 dim：source/species/tissue/disease/modality/platform/assay/has_raw_data/year（见返回 facets）；
               value 取 facets[].values[].value 回传的**分面键**（物种/组织/疾病为小写归一键，如 "homo sapiens"，
               本工具会自动小写归一）。非法/未知维度项被静默丢弃（安全默认）。典型用法：先不传拿到 facets，
               再挑某个 value 传回来收窄。分面把结果收窄到 0 属正常业务结果（非报错）。
      suppressed_constraints: 忽略「已命中」里的某维硬约束→检索前放宽（对齐网页删「已命中」chip；默认 None＝不放宽）。
               取值为**极性 filter_id**：include:<dim> / exclude:<dim>（dim∈物种/组织/疾病/平台/技术/模态）/
               raw:required / raw:forbidden / date:range，或**裸 dim**（该维正负全放宽）。取自本次返回的
               query_constraints[].filter_id。非法项静默丢弃。默认不传＝不放宽任何约束（官方评测/旧调用逐位不变）。
      lenient_dims: 诚实降级——对这些维度（物种/组织/疾病/平台/技术/模态）上**字段为空**的记录视作通过
               （无法核验≠不匹配），已知不同值仍排除。取自本次返回的 coverage_caveats[].dim。与
               suppressed_constraints 的区别：suppress **整维放宽**（含已知不同值），lenient **只纳未标注的**。
               默认 None＝不宽容（官方评测/旧调用逐位不变）。
      date_from / date_to: 发表时间范围（ISO YYYY-MM-DD，含端点；对齐网页年份选择器）。**显式传入优先**，
               覆盖从 query 自然语言解析出的相对时间；不传则保留 query 里解析出的范围（如「近三年」）。
               严格校验（与网页端同口径，2026-08-08 产品方拍板拉齐、登记册清零）：给了就
               必须是格式与日历都合法的日期，非法 → bad_param 点名参数；from > to 倒挂窗口同样
               bad_param，不再静默忽略或按字面值执行。

    错误契约：**非法请求** → 客户端 `isError=true`，消息形如
      「Error executing tool recommend_datasets: bad_param: top_k 必须是正整数…」：
        · empty_query（query 空）
        · bad_query（query 含控制/不可见字符、纯符号/emoji 无检索内容、或超长 >2000 字）
        · bad_param（recall/rerank/strategy 越界、top_k/rerank_top_n 非正整数或超上限 100、
          date_from/date_to 非法日期或 from>to 倒挂窗口）
        · bad_source（sources 有未知来源名或空/空白来源名；合法来源见 biodata_status）
        · 未知参数名（如 topk 拼错）→ 校验错误（extra=forbid），不再静默用默认值。
      **合法请求的业务结果**（无匹配 / fail-closed 弃权 / 需澄清）仍是正常返回、`ok=true`、`isError=false`
      ——那不是错误。非法时间表达（近0/负年、非法日历日、并列年份歧义）走**弃权**（understood.abstain=true），
      非报错。
      注：result_total=满足约束的命中总数（同一数据集的近重复条目合并前）；returned=合并去重并截取 top_k 后
                           实际返回的条数，二者不等属正常。

    返回 dict（ok=true 时）：
      understood        —— 系统从 query 抽到的硬约束 + 是否 fail-closed 弃权。
      result_total      —— 满足约束的**命中总数**（未截断）。
      returned          —— 本次实际返回条数。
      candidates        —— 结构化候选，每条含 dataset_uid（喂给 get_file_manifest）、真实下载直链、
                           n_files、score、matched_fields/missing_fields、推荐理由、确定性 **introduction**，
                           以及 **caveats**
                           （数据一致性建议信号：**物种**标注与描述文本不一致；空列表=无问题。
                           n_files=0 提示见 get_file_manifest）。
      facets            —— 可继续收窄的维度（按物种/组织/…）；值里的 value 即 facet_filters 该维要回传的分面键。
      query_constraints —— 本次查询**已命中**的硬约束（放宽后的真源）：每项含 filter_id（喂给 suppressed_constraints
                           以放宽该维）、dim、label、values。等价网页「已命中」区。
      applied_facets    —— 本次真正生效的分面细化项（净化去噪后回显；空=未细化）。
      applied_suppressed—— 本次真正生效的放宽维度（净化去噪后回显；空=未放宽）。
      coverage_caveats  —— 诚实降级缺口 [{dim,label,count,by_source:[{source,count}]}]：满足其它条件、
                           但该维**未标注**（无法核验）的记录计数，按源分组。回传其 dim 给 lenient_dims 即纳入。
      unused_query_terms—— ["免疫","儿童",…]：查询里**没有对应筛选维度**的实义描述词
                           （性别/年龄/受试者/功能类）；系统没有用它们过滤，也没有因此弃权。空=无。
      action_markers    —— ["打包","下载脚本",…]：用户在同一句里说出的**执行类**诉求。本工具只做检索，
                           不会替他执行；要打包用 build_task_pack、要文件清单用 get_file_manifest。
                           非空时应主动告诉用户「检索到了 N 条，要打包吗」，而不是假装没看见。空=没提。
      applied_lenient   —— 本次真正宽容的维度（净化后回显；空=未宽容）。
      relaxation_options—— 0 结果时的放宽建议，**两档**，靠每项的 `kind` 区分（转述时必须照它措辞，
                           说反了就是把系统没做的事说成做了）：
                           `kind="drop"` = **去掉「label」这一个条件**，用户写的其余条件都还在；
                           `kind="only"` = **只按「label」搜**，其余条件（含 FASTQ / 时间范围 /
                           排除项）这次全都没参与筛选。`count` 是该档能救回多少条。
                           第二档只在条件够多时才出现，它是「去掉任何单独一个仍是 0 条」时唯一还救得回来的一档。
                           `suppress` 是该档对应的 suppressed_constraints filter_id 清单
                           （2026-09-14 收编新增；要执行该档就把它整组作为 suppressed_constraints 重调）。
      degraded_search   —— 规则因**未收录词**弃权时的降级建议 {ignored_terms, query, count, results,
                           active_filters}：「先忽略这几个不认识的词，能搜到 N 条，实际按这几个条件筛」。
                           **服务器不会自动应用**：实测「翼龙的单细胞数据」忽略「翼龙」后是 3473 条无关
                           数据，自动降级等于把「查无此物」答成「给你一堆别的」。要用就带着 query 再调一次，
                           并把 ignored_terms 如实告诉用户。null=没有可给的降级（忽略后一个条件都不剩，
                           或忽略了也搜不到）。
      advisory          —— **诚实提示**（非 None 时应转述给用户）：查询较宽泛/未识别到精确约束时提示
                           「本批只按 <维度> 粗筛、靠前顺序未必精确」，别把它当精确匹配转述。
      interpretation    —— 原句、送入规则解析的清洗句、实际来源与完整 intent 投影（核对系统把 query 理解成什么）。
      query             —— 原始查询字符串（回显）。
      answer_markdown   —— 现成中文 Markdown 结果表，父 agent 可直接转述给用户。
      meta              —— 如实动态标注本次调用：deterministic / llm_used / offline /
                           recall_used / rerank_used / links_snapshot / evidence_scope / notes(注意事项)；
                           另含 **recall_applied / rerank_applied / llm_applied**（**实际采用的后端**——是否发生
                           回退看这些，而非 *_used/*_requested 这类「请求形态」字段）、strategy（auto 决策）、
                           search_trace（执行轨迹）。
    """
    try:
                                                                                 
        _require_query(query)
                                                            
        eff_recall = (recall if recall is not None else _DEFAULT_RECALL)
        eff_recall = (eff_recall or "off").strip().lower()
        if eff_recall not in _RECALL_CHOICES:
            raise _bad_request(
                "bad_param",
                f"recall 只能是 {list(_RECALL_CHOICES)} 之一（off=默认确定性；"
                f"cross_encoder=本地重排，离线无 key；dense=本地稠密召回），收到「{_clip(recall)}」。",
            )
        _validate_enum("rerank", rerank, _RERANK_CHOICES)
        _validate_positive("top_k", top_k)
        _validate_positive("rerank_top_n", rerank_top_n)
        _validate_sources(sources)
        strategy_norm = str(strategy or "fixed").strip().lower()
        _validate_enum("strategy", strategy_norm, ("fixed", "auto"))

        notes: list[str] = []
                                                                            
                                                                     
        if eff_recall in ("cross_encoder", "dense"):
            from dataset_recommender.retrieval.vector_recall import recall_backend_ready
            if not recall_backend_ready(eff_recall):
                notes.append(
                    f"recall={eff_recall} 未在本 MCP 服务器预热 → 已回退规则序（避免 stdio 下的模型加载死锁）。"
                    f"要在 MCP 里启用本地重排：于服务器配置的 env 设 BIODATA_MCP_RECALL={eff_recall} 并重启"
                    f"（启动时预热约 10-20s，之后每次调用只打分、不再加载）。CLI / Web 无此限制、可直接用。"
                )
                eff_recall = "off"

                                                                       
                                                    
        strat_preferred = _DEFAULT_RECALL if _DEFAULT_RECALL in ("cross_encoder", "dense") else "cross_encoder"
        strat_recall_available = False
        strat_llm_available = False
        if strategy_norm == "auto":
            from dataset_recommender.retrieval.vector_recall import recall_backend_ready
            strat_recall_available = recall_backend_ready(strat_preferred)
            if auto_allow_llm:
                from dataset_recommender.llm.llm_client import load_llm_config
                strat_llm_available = bool(load_llm_config(project_root=get_app_paths().config_root).api_key)

                                                                        
                                                                         
                                                      
        from dataset_recommender.app.workflow import sanitize_facet_filters, sanitize_suppressed, sanitize_lenient_dims
        eff_facet_filters = sanitize_facet_filters(facet_filters)
        eff_suppressed = sanitize_suppressed(suppressed_constraints)
        eff_lenient = sorted(sanitize_lenient_dims(lenient_dims))                          
                                                      
                                                                       
                                                                           
                                                    
        eff_date_from = _require_iso_date(date_from, name="date_from")
        eff_date_to = _require_iso_date(date_to, name="date_to")
        _validate_date_window(eff_date_from, eff_date_to)

        wf = _workflow()
        from dataset_recommender.app.workflow import RecommendParams
        res = wf.run_with_meta(
            RecommendParams(
                query=query,
                top_k=top_k,
                sources=sources,
                auto_parse_sources=bool(auto_parse),
                use_llm=use_llm,                                             
                mock_llm=False,
                recall_backend=eff_recall,                                                   
                rerank_backend=rerank,                      
                rerank_top_n=rerank_top_n,
                date_from=eff_date_from,                              
                date_to=eff_date_to,
                facet_filters=eff_facet_filters,                              
                suppressed_constraints=eff_suppressed,                            
                lenient_dims=eff_lenient,                                                  
                strategy=strategy_norm,                                                    
                recall_available=(strat_recall_available if strategy_norm == "auto" else None),
                llm_available=strat_llm_available,
                preferred_recall=strat_preferred,
            )
        )
        interpretation = getattr(res, "interpretation", None) or {}
        intent = interpretation.get("intent") or _parse_intent(query, sources=sources, auto_parse=bool(auto_parse))
                                                                     
        eff_recall_used = eff_recall
        eff_rerank_used = rerank
        strategy_result = getattr(res, "strategy", None)
        if strategy_norm == "auto" and strategy_result:
            eff_recall_used = strategy_result.get("recall_backend", eff_recall)
            eff_rerank_used = strategy_result.get("rerank_backend", "off")
                                              
        search_trace = getattr(res, "search_trace", None) or {}
        trace_by_id = {x.get("id"): x for x in search_trace.get("steps", []) if isinstance(x, dict)}
        recall_actual = eff_recall_used if trace_by_id.get("local_semantic", {}).get("status") == "used" else "off"
        rerank_actual = eff_rerank_used if trace_by_id.get("llm_rerank", {}).get("status") == "used" else "off"
                                                                                
                                                                
        deterministic = (eff_recall_used == "off") and (eff_rerank_used == "off") and (not use_llm)
        llm_used = bool(res.llm_response_used) or (eff_rerank_used == "llm")
                                                                       
        offline = (not use_llm) and (eff_rerank_used != "llm")
        if use_llm or eff_rerank_used == "llm":
            notes.append(
                "已请求 LLM 路径（use_llm 润色 / rerank=llm）：需 key + 联网且**非确定性**；"
                "缺 key 时 fail-open——use_llm 回退确定性结果（llm_used=False），rerank=llm 静默回退原序。"
            )
        if strategy_norm == "auto" and strategy_result:
            notes.append(
                f"strategy=auto：分类器按存活集大小选 recall={eff_recall_used}/rerank={eff_rerank_used}"
                f"（tier={strategy_result.get('tier')}）。MCP 只使用已预热本地后端；LLM 还需"
                " auto_allow_llm=true 与可用 API 配置，未满足则回退安全路径。详见 meta.strategy。"
            )
        meta = {
            "deterministic": deterministic,
            "llm_used": llm_used,
            "offline": offline,
            "recall_used": eff_recall_used,
            "rerank_used": eff_rerank_used,
            "recall_requested": eff_recall_used,
            "rerank_requested": eff_rerank_used,
            "recall_applied": recall_actual,
            "rerank_applied": rerank_actual,
            "llm_applied": bool(res.llm_response_used) or (rerank_actual == "llm"),
                                                                                                     
            "strategy": (strategy_result if strategy_norm == "auto" else None),
            "search_trace": search_trace,
            "links_snapshot": _LEDGER_SNAPSHOT,
            "evidence_scope": (
                "有文件级校验的来源只有 10x Genomics；其它来源只用于发现、元数据完备度较低，"
                "字段显示 unknown 只表示来源没标注，不等于不满足该条件。"
            ),
        }
        if notes:
            meta["notes"] = notes
                                                    
        _enrich_candidates(res.retrieved_data)
        return {
            "ok": True,
            "query": query,
            "understood": intent,
            "interpretation": interpretation,
            "advisory": _advisory(intent, res.result_total),
            "result_total": res.result_total,
            "returned": len(res.retrieved_data),
            "candidates": res.retrieved_data,
            "facets": res.facets,
                                                                            
            "query_constraints": getattr(res, "active_filters", []),
                                                  
            "applied_facets": eff_facet_filters,
            "applied_suppressed": eff_suppressed,
                                                                            
                                                                                      
            "coverage_caveats": res.coverage_caveats,
                                                                
            "unused_query_terms": getattr(res, "unused_query_terms", []),
                                                       
                                                               
            "action_markers": getattr(res, "action_markers", []),
            "applied_lenient": eff_lenient,
            "relaxation_options": res.relaxation_options,
                                                        
                                                           
            "degraded_search": getattr(res, "degraded_search", None),
            "answer_markdown": res.answer,
            "meta": meta,
        }
    except ToolError:
        raise                                               
    except Exception as exc:                                                
        raise _internal_error(exc) from exc


                                                               
@mcp.tool()
@_logged
def get_file_manifest(dataset_uid: str) -> dict:
    """取某数据集的**文件级**下载清单。

    不只是数据集页面链接：这里给到**每个文件**的直链、md5、大小、fileset 归属与 pipeline。
    dataset_uid 从 recommend_datasets 返回的 candidates[*].dataset_uid 取。
    什么时候**别**用：它只拿清单、**不下载文件**——真要下载到本地用 provision_dataset；
    还没拿到 dataset_uid 时先用 recommend_datasets / browse_datasets 定位，不要拿数据集名字瞎猜 uid。

    返回 dict：
      platform / page_url / fetch_status / pipelines
      counts   —— n_files / n_filesets / n_fastq_files
      raw_data —— has_raw_data_actual（实测更正信号）、fastq_present、fastq_url、fastq_bytes
      primary  —— 代表性主文件（分析师最先要的那份）：url / filename / title / bytes
      files[]  —— 每个文件：fileset / category / filename / title / download_url / host /
                  bytes / size_human / md5sum / pipeline，外加 **status**（链接存活台账里最近一次实测的结论：
                  reachable / size / integrity / load / problem / problem_reason / last_verified；
                  无实测记录则 status=None）。
      problems —— 本数据集**问题文件**精简清单（reachable=dead 或 size-mismatch）：filename / url / reason。
                  下载前一眼可见的告警；无问题则为空列表。
      provisioning —— 数据集级下载可用性摘要（字段名保持 provisioning）：n_files / n_problem / provisioning_ok / last_verified。
      verification —— 校验语义 + 最近快照 id/日期 + 问题文件计数；md5 为 10x 声明值、尚未逐字节重算；
                     本工具**不做实时联网探活**（遵守「URL 存在 ≠ 当前可加载」契约）。
      caveats  —— 数据集级质量提示：n_files=0（无文件级清单）时给出说明；空=无。
    空 / 查不到的 dataset_uid = 非法请求 → raise ToolError（`empty_uid` / `bad_uid`，客户端 isError=true），
    提示先用 recommend_datasets 拿 uid。
    """
    try:
        from dataset_recommender.corpus import downloads, inspection
        if not dataset_uid or not str(dataset_uid).strip():
            raise _bad_request(
                "empty_uid",
                "dataset_uid 不能为空；取自 recommend_datasets 返回的 candidates[*].dataset_uid。",
            )
                                                            
        dataset_uid = str(dataset_uid).strip()
        rec = downloads.get(dataset_uid)
        if not rec:
            raise _bad_request(
                "bad_uid",
                f"未找到 dataset_uid「{_clip(dataset_uid)}」的文件清单；"
                "先调 recommend_datasets，用返回里 candidates[*].dataset_uid 作为本工具入参。",
            )
        uid = rec.get("dataset_uid", dataset_uid)
                                                                      
                                                                        
        files: list = []
        problems: list = []
        for f in (rec.get("files", []) or []):
            if not isinstance(f, dict):
                files.append(f)
                continue
            st = inspection.status_for(uid, f.get("download_url"))
            files.append({**f, "status": st})
            if st and st.get("problem"):
                problems.append({
                    "filename": f.get("filename"),
                    "download_url": f.get("download_url"),
                    "reason": st.get("problem_reason"),
                })
        prov = inspection.dataset_summary(uid)
                                               
        manifest_caveats: list[str] = []
        if not rec.get("n_files", 0):
            manifest_caveats.append(
                "n_files=0：该数据集无文件级清单（多为发现层外部来源未落实文件直链）；"
                "下载与文件级校验请优先选 10x 基础语料记录。"
            )
        return {
            "ok": True,
            "dataset_uid": uid,
            "caveats": manifest_caveats,
            "platform": rec.get("platform"),
            "page_url": rec.get("url"),
            "fetch_status": rec.get("fetch_status"),
            "pipelines": rec.get("pipelines", []),
            "counts": {
                "n_files": rec.get("n_files", 0),
                "n_filesets": rec.get("n_filesets", 0),
                "n_fastq_files": rec.get("n_fastq_files", 0),
            },
            "raw_data": {
                "has_raw_data_actual": rec.get("has_raw_data_actual"),
                "has_any_raw_input": rec.get("has_any_raw_input"),
                "fastq_present": bool(rec.get("fastq_download_url")),
                "fastq_url": rec.get("fastq_download_url"),
                "fastq_bytes": rec.get("fastq_bytes"),
            },
            "primary": {
                "url": rec.get("primary_download_url"),
                "filename": rec.get("primary_filename"),
                "title": rec.get("primary_title"),
                "bytes": rec.get("primary_bytes"),
            },
            "provisioning": prov,
            "problems": problems,
            "files": files,
            "verification": {
                "md5": "每个文件随清单给出 md5sum，可做下载后完整性校验（md5 为 10x 声明值，尚未逐字节重算）。",
                "links_snapshot": _LEDGER_SNAPSHOT,
                "last_verified": (prov or {}).get("last_verified"),
                "snapshot_id": (prov or {}).get("snapshot_id"),
                "n_problem_files": (prov or {}).get("n_problem", 0),
                "note": (
                    "每个文件 status 给出最近一次实测的 reachable / size 结论"
                    "（integrity=md5 未重算、load 未加载，均 unknown≠失败）。"
                    + (f" ⚠ 本数据集有 {len(problems)} 个问题文件（见 problems），下载前请留意。"
                       if problems else " 本快照下全部文件无 problem。")
                ),
            },
        }
    except ToolError:
        raise                                   
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                             
@mcp.tool()
@_logged
def parse_constraints(
    query: str,
    sources: list[str] | None = None,
    auto_parse: bool = True,
) -> dict:
    """只解析、不检索：看系统从 query 抽出了哪些**硬约束**、有没有因为拿不准而弃权。

    便宜（不加载候选、不排序），供父 agent 在真正推荐前预检「机器听懂了什么、会按什么硬过滤」。
    默认 auto_parse=True，会与 recommend_datasets 一样识别 CELLxGENE、HCA、EBI SCEA、ArrayExpress、
    10x Genomics、ENCODE、HuBMAP、Single Cell Portal 和 NCBI GEO 等高辨识度来源专名（ENCODE 只认
    encode project / encodeproject / encode portal / encode 数据库 等复合写法，刻意不收裸
    encode——它是普通英文动词，收了会把无关查询静默改成只查 ENCODE）；返回 source_resolution
    和 effective_sources。
    弃权（abstain）规则（2026-07-25 大幅收窄）：
      · 「或」组合与「最好 / 尽量 / 如果可以」这类倾向说法**照做**——同一维度多个值本来就是「或」，
        倾向说法按软偏好执行（只加权、不筛掉任何数据）；`or_handling` 会如实说明实际执行方式。
      · 仍然弃权的只有两类：① 句子里剩下**系统未收录的实义词**（此时同时给出「先忽略这几个词」
        的降级检索，用户一步就能拿到结果）；② 表达不出来且**会误删数据**的结构，
        如「优先不要 X」（软性排除做出来就是硬排除）、条件句 / 疑问句里的否定。
      · 硬条件筛完为空（`resolution_status == "no_match"`）不是弃权：如实报缺哪一条 + 给放宽建议，
        不硬凑不相关的结果。
    """
    try:
        _require_query(query)
        _validate_sources(sources)
        return {
            "ok": True,
            "query": query,
            "understood": _parse_intent(query, sources=sources, auto_parse=bool(auto_parse)),
        }
    except ToolError:
        raise                                                    
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                        
                                                                 
                                                         
                                                                 
                                                   
                                                     
_DEFAULT_BROWSE_LIMIT = 50
_MAX_BROWSE_LIMIT = MAX_DATASETS_LIMIT                                            


def _browse_facet(counter: dict[str, int]) -> list[dict]:
    """分面计数投影：单一真源 `item_view.facet_list`（与 Web 共用一份，不再各抄）。"""
    from dataset_recommender.content.item_view import facet_list
    return facet_list(counter)


def _browse_item(record, include_introduction: bool = False) -> dict:
    """归一化记录 → 浏览卡片字段。**与网页共用 `item_view.build_item` 单一真源。**

    2026-07-17 之前这里是一份**手抄的平行投影**，docstring 声称与 webapp「同口径」，
    实际缺 `modality` / `collection_doi`——于是 `assess_dataset_fair` 在 Web 修好之后、
    在 MCP 依旧 5667/5667 印泛泛的 "dataset"，DOI 句永远印不出。两份手抄的东西迟早漂移，
    且漂移时没有任何测试会响。现在只有一处实现，想漂移都没地方漂。
    """
    from dataset_recommender.content.item_view import build_item
    return build_item(record, include_introduction=include_introduction)


def _interleave_by_source(records: list) -> list:
    """按来源轮转交错，让浏览首屏就同时看到各库：单一真源 `item_view.interleave_by_source`
    （与 Web 共用一份）。各库内部保持原顺序。"""
    from dataset_recommender.content.item_view import interleave_by_source
    return interleave_by_source(records)


def _browse_matches(item: dict, species, platform, source, year) -> bool:
    """轻量过滤谓词：species/platform/source 大小写不敏感等值；year 前缀等值。空过滤=放行。"""
    def _eq(a, b) -> bool:
        return str(a or "").strip().lower() == str(b or "").strip().lower()
    if species and not _eq(item["species"], species):
        return False
    if platform and not _eq(item["platform"], platform):
        return False
    if source and not _eq(item["source"], source):
        return False
    if year:
        y = str(year).strip()
        if str(item.get("published_year") or "") != y and not item["published_date"].startswith(y):
            return False
    return True


@mcp.tool()
@_logged
def browse_datasets(
    species: str | None = None,
    platform: str | None = None,
    source: str | None = None,
    year: str | None = None,
    limit: StrictInt | None = None,
    offset: StrictInt | None = None,
) -> dict:
    """全库浏览（对齐网页 `/api/datasets`）：所有来源并列的数据集 + 物种/平台/来源/年份分面，
    可按维度轻过滤 + 分页。**无需 query**——用于「让我看看目录里有什么」这类浏览意图。

    与推荐管线正交：不做 NL 约束抽取、不排序，只做等值过滤 + 分页 + 分面计数；确定性、离线、只读。
    与 `recommend_datasets` 的分面（facet_filters）不同：那是在**一句 query 的命中集**上收窄，
    这是在**全库**上浏览。

    参数（全部可选）：
      species / platform / source / year: 轻量等值过滤（大小写不敏感；year 为发表年份如 "2023"）。
               可先不传拿到 facets、再挑 facets[dim][].value 传回。None/空=该维不过滤。
      limit:   本页返回记录数上限，默认 50，**硬上限 100**（防响应放大；与网页 `/api/datasets`
               同源常量）。须严格整数、正数。
      offset:  分页偏移，默认 0；≥ total 时返回空 records（非报错）。须严格整数、非负。

    返回 dict：
      total    —— 过滤后命中总数（未分页）。
      returned / offset / limit —— 本页实际返回条数 / 偏移 / 页大小。
      facets   —— {species/platform/source/published_year: [{value,count}]}，按**过滤后**集合计数
                  （facet 计数之和 = total，自洽）；unknown_year_count 单列（无法解析年份的条数）。
      records  —— 本页记录（精简字段：名称/物种/组织/疾病/平台/技术/来源/页面 url/下载直链/
                  dataset_uid/n_files/发表时间；**不含**大段介绍——要介绍用 get_dataset_introduction）。
      filters_applied —— 回显本次生效的过滤（归一后）。
      caveats  —— 数据一致性提示（如外部来源完备度较低）。
    非法请求（limit/offset 非严格整数、非正/负、超上限）→ raise ToolError（bad_param，isError=true）；
    未知过滤取值 → 命中 0 条属正常业务结果、非报错。
    """
    try:
        from dataset_recommender.corpus.corpus import load_full_corpus
                                                                                
        eff_limit = _DEFAULT_BROWSE_LIMIT if limit is None else limit
        if isinstance(eff_limit, bool) or not isinstance(eff_limit, int) or eff_limit <= 0:
            raise _bad_request("bad_param", f"limit 必须是正整数（≥1）或省略，收到「{_clip(limit)}」。")
        if eff_limit > _MAX_BROWSE_LIMIT:
            raise _bad_request(
                "bad_param",
                f"limit={eff_limit} 超过上限 {_MAX_BROWSE_LIMIT}（防响应放大/吃爆上下文）；请用 ≤{_MAX_BROWSE_LIMIT} 并分页。",
            )

        eff_offset = 0 if offset is None else offset
        if isinstance(eff_offset, bool) or not isinstance(eff_offset, int) or eff_offset < 0:
            raise _bad_request("bad_param", f"offset 必须是非负整数（≥0）或省略，收到「{_clip(offset)}」。")

        s = _settings()
        records = _interleave_by_source(load_full_corpus(s.data_dir, s.project_root))
        f_species = (species or "").strip() or None
        f_platform = (platform or "").strip() or None
        f_source = (source or "").strip() or None
        f_year = (year or "").strip() or None

        species_counter: dict[str, int] = {}
        platform_counter: dict[str, int] = {}
        source_counter: dict[str, int] = {}
        year_counter: dict[str, int] = {}
        unknown_year = 0
        matched: list[dict] = []
        for record in records:
            item = _browse_item(record, include_introduction=False)
            if not _browse_matches(item, f_species, f_platform, f_source, f_year):
                continue
            matched.append(item)
            sp = (item["species"] or "").strip()
            if sp:
                species_counter[sp] = species_counter.get(sp, 0) + 1
            pf = item["platform"]
            if pf:
                platform_counter[pf] = platform_counter.get(pf, 0) + 1
            src = item["source"]
            source_counter[src] = source_counter.get(src, 0) + 1
            yr = item["published_year"]
            if yr is not None:
                year_counter[str(yr)] = year_counter.get(str(yr), 0) + 1
            else:
                unknown_year += 1

        total = len(matched)
        page = matched[eff_offset:eff_offset + eff_limit]
        caveats: list[str] = []
        if f_source is None:
            caveats.append(
                "外部平台来源完备度较低（文件级信息以 10x Genomics 基础语料最完整）；字段 unknown≠不满足。"
            )
        return {
            "ok": True,
            "total": total,
            "returned": len(page),
            "offset": eff_offset,
            "limit": eff_limit,
            "filters_applied": {"species": f_species, "platform": f_platform, "source": f_source, "year": f_year},
            "facets": {
                "species": _browse_facet(species_counter),
                "platform": _browse_facet(platform_counter),
                "source": _browse_facet(source_counter),
                "published_year": _browse_facet(year_counter),
            },
            "unknown_year_count": unknown_year,
            "records": page,
            "caveats": caveats,
        }
    except ToolError:
        raise                                            
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                      
@mcp.tool()
@_logged
def get_dataset_introduction(
    uid: str | None = None,
    url: str | None = None,
    name: str | None = None,
    source: str | None = None,
    llm: bool = False,
) -> dict:
    """单数据集介绍（对齐网页 `/api/introduction`）：按 uid（首选）/ url / name 定位一条记录，
    默认返回**只整理现有元数据、不调用 LLM** 的结构化介绍（summary + facts + caveat；与网页显示的内容一致）。

    典型用法：`browse_datasets` / `recommend_datasets` 拿到 dataset_uid 后按需取该数据集介绍
    （browse 列表刻意不内嵌介绍以免响应放大；recommend 候选已内嵌，此工具便于按 uid 单点回查）。

    参数（至少给一个 uid/url/name）：
      uid:    dataset_uid（精确键，首选）。
      url:    数据集页面 url（uid 缺失时的回退键，精确匹配）。
      name:   数据集名称（精确匹配；可配 source 消歧，避免同名不同库串卡）。
      source: 配合 name 消歧的来源名。
      llm:    True（可选）→ 在确定性介绍之外，额外附一段 LLM 生成的中文导读（`introduction.llm_summary`）。
              需要**两个条件同时满足**才会真的调用：服务端开启 `ENABLE_LLM`，且本次传 `llm=True`；
              否则 `llm_summary=None`，确定性介绍的内容不受任何影响。默认 False。
    返回 dict：matched（dataset_name / dataset_uid / source / url）+ introduction（summary/facts/caveat/…）。
    空（uid/url/name 全空）→ 报 `empty_key` 错误；查不到 → 报 `not_found` 错误；
    name 命中多条同名且 source 消歧失败 → 报 `ambiguous_name` 错误（消息附各候选 uid，改用 uid 重调）。
    均置客户端 isError=true。
    """
    try:
        from dataset_recommender.corpus.corpus import load_full_corpus, locate_record
        uid_s = str(uid or "").strip()
        url_s = str(url or "").strip()
        name_s = str(name or "").strip()
        source_s = str(source or "").strip()
        if not (uid_s or url_s or name_s):
            raise _bad_request(
                "empty_key",
                "需给出 uid、url 或 name 之一（uid 取自 recommend_datasets / browse_datasets 的 dataset_uid）。",
            )
        s = _settings()
                                                               
                                                                           
        record, ambiguous = locate_record(
            load_full_corpus(s.data_dir, s.project_root), uid=uid_s, url=url_s, name=name_s, source=source_s)
        if ambiguous:
            cands = "；".join(
                f"uid={c['dataset_uid']}（{c['source']}，{c['url']}）" for c in ambiguous)
            raise _bad_request(
                "ambiguous_name",
                f"该名称命中 {len(ambiguous)} 条同名数据集，无法确定是哪一条：{cands}。"
                "请改用 dataset_uid 重新调用。",
            )
        if record is None:
            raise _bad_request(
                "not_found",
                "未找到匹配的数据集；核对 uid/url/name（可先用 browse_datasets 或 recommend_datasets 定位）。",
            )
        item = _browse_item(record, include_introduction=True)
        intro = item["introduction"]
        if llm:
            from dataset_recommender.llm.intro_llm import enrich_introduction_with_llm
            intro = enrich_introduction_with_llm(item, intro=intro)
        return {
            "ok": True,
            "matched": {
                "dataset_name": item["dataset_name"],
                "dataset_uid": item["dataset_uid"],
                "source": item["source"],
                "url": item["url"],
            },
            "introduction": intro,
        }
    except ToolError:
        raise                                     
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                  
@mcp.tool()
@_logged
def assess_dataset_fair(
    uid: str | None = None,
    url: str | None = None,
    name: str | None = None,
    source: str | None = None,
) -> dict:
    """单数据集 **FAIR 元数据自检 + 投稿数据可用性说明（DAS）**（对齐网页 `/api/fair`）：
    按 uid（首选）/ url / name 定位一条记录，返回**只据现有公开元数据、不调用 LLM** 的确定性结果——
    13 项 FAIR 自检（F/A/I/R，每项 pass|partial|unknown；unknown = 来源未标注，不等于不满足）+ 一段可
    直接投稿的**英文**数据可用性说明（缺失的子句会略去，并列进 missing）。与网页 `/api/fair` 使用同一套
    实现，结果逐字段一致。

    典型用法：`browse_datasets` / `recommend_datasets` 拿 dataset_uid 后按需评估该数据集投稿就绪度。
    什么时候**别**用：它只查**元数据完备度**（有没有标注），不评价数据本身的科学质量——
    「这条数据靠不靠谱、能不能信」不是它能答的。
    参数（至少给一个 uid/url/name；name 可配 source 消歧）。空 → 报 `empty_key` 错误；查不到 → 报 `not_found` 错误；
    name 命中多条同名且 source 消歧失败 → 报 `ambiguous_name` 错误（消息附各候选 uid，改用 uid 重调）。
    均置客户端 isError=true。
    返回 dict：ok + matched（dataset_name/dataset_uid/source/url）+ fair_report
    （fair.{checks,summary,gaps} + data_availability.{statement,missing,notes}）。
    """
    try:
        from dataset_recommender.corpus.corpus import load_full_corpus, locate_record
        from dataset_recommender.retrieval.fair import build_fair_report
        uid_s = str(uid or "").strip()
        url_s = str(url or "").strip()
        name_s = str(name or "").strip()
        source_s = str(source or "").strip()
        if not (uid_s or url_s or name_s):
            raise _bad_request(
                "empty_key",
                "需给出 uid、url 或 name 之一（uid 取自 recommend_datasets / browse_datasets 的 dataset_uid）。",
            )
        s = _settings()
                                                                        
                                                         
        record, ambiguous = locate_record(
            load_full_corpus(s.data_dir, s.project_root), uid=uid_s, url=url_s, name=name_s, source=source_s)
        if ambiguous:
            cands = "；".join(
                f"uid={c['dataset_uid']}（{c['source']}，{c['url']}）" for c in ambiguous)
            raise _bad_request(
                "ambiguous_name",
                f"该名称命中 {len(ambiguous)} 条同名数据集，无法确定是哪一条：{cands}。"
                "请改用 dataset_uid 重新调用。",
            )
        if record is None:
            raise _bad_request(
                "not_found",
                "未找到匹配的数据集；核对 uid/url/name（可先用 browse_datasets 或 recommend_datasets 定位）。",
            )
        item = _browse_item(record, include_introduction=True)
        return {
            "ok": True,
            "matched": {
                "dataset_name": item["dataset_name"],
                "dataset_uid": item["dataset_uid"],
                "source": item["source"],
                "url": item["url"],
            },
            "fair_report": build_fair_report(item),
        }
    except ToolError:
        raise                                     
    except Exception as exc:
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def build_reuse_pack(uids: list[str]) -> dict:
    """N 个**复用的公开数据集** → 一份投稿材料（对齐网页 `POST /api/reuse-pack`）：
    一段可放进 Data Availability 小节的**英文段落**（按来源汇总）+ 一张**补充表**
    （每个数据集一行：编号 / 链接 / 原始数据状态）+ 一份**中文待办**（本工具无法提供、
    必须你自己核实的事）+ Markdown 导出串。确定性、离线、不调用 LLM、**不写盘**。

    **这不是 Data Availability Statement 的主句**。主句讲的是你自己产出的数据去向；
    那句本工具帮不上也不会写——它需要你未发表的实验信息，本工具按设计不接收
    （这条边界是**结构性的**：入参 keys-only，见下。2026-07-29 起产物里不再另附一段
    重复陈述它的中文 `boundary` 字段）。

    入参 **keys-only**：`uids` = dataset_uid 列表（取自 recommend_datasets /
    browse_datasets 的 `dataset_uid`）。服务端自己解析记录，**不接受**调用方传数据集内容——
    入参是键不是内容 → 不存在「把未发表工作贴进来」的路径。

    诚实约束：没有真 accession 就不写 accession（全库仅 38.3% 有）；「本工具没查过」
    （如 ArrayExpress 的原始数据可及性）**绝不**写成「没有原始数据」，改列进 gaps；
    查不到的 uid 会明确列进 `unresolved`，绝不静默丢弃。

    空/非法/超量 → ToolError（empty_input / bad_param / too_many），客户端 isError=true。
    返回 dict：ok + pack（n_datasets/paragraph/table/sources/raw_state_counts/
    gaps/unresolved/**retrieval** 复现出处）+ markdown + **ris** + **bibtex**（数据集引文导出。
    注意：collection_doi 指向的是**产出这批数据的论文**、不是数据集本身 → 条目一律用数据集类型
    （RIS `TY-DATA` / BibTeX `@misc`），
    DOI 只作关联论文放 note）。
    """
    try:
        from dataset_recommender.corpus.corpus import load_full_corpus
        from dataset_recommender.content.reuse_pack import (
            ReusePackError,
            build_pack_for_uids,
            to_bibtex,
            to_markdown,
            to_ris,
        )
        try:
            eff_uids = sanitize_uids_shared(uids)
        except ReusePackError as exc:
            raise _bad_request(exc.code, str(exc)) from None
        s = _settings()
        pack = build_pack_for_uids(eff_uids, load_full_corpus(s.data_dir, s.project_root))
        return {
            "ok": True,
            "pack": pack,
            "markdown": to_markdown(pack),
            "ris": to_ris(pack),                              
            "bibtex": to_bibtex(pack),                                      
        }
    except ToolError:
        raise
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                        
                                                   
_MAX_UID_CHARS = 200


def sanitize_uids_shared(uids):
    """与 Web `/api/reuse-pack` 共用同一套入参归一（`reuse_pack.sanitize_uids` 单一真源）。"""
    from dataset_recommender.content.reuse_pack import ReusePackError, sanitize_uids
    out = sanitize_uids(uids)
    for u in out:
        if len(u) > _MAX_UID_CHARS:
            raise ReusePackError(
                "bad_param",
                f"单条 dataset_uid 过长（{len(u)} 字符，上限 {_MAX_UID_CHARS}）——这不是合法编号。",
            )
    return out


@mcp.tool()
@_logged
def lookup_identifier(identifier: str) -> dict:
    """按**标识符**精确反查数据集 + 诚实 fail-closed（对齐 Web `/api/recommend` 的 `identifier_lookup`）。

    贴 CELLxGENE/HCA 的 **UUID**、ArrayExpress/SCEA 的 **E-XXXX-N** 编号、ENCODE 的 **ENCSR…**
    实验编号、GEO 的 **GSE** 系列号、HuBMAP 的 **HBM…** 编号、Single Cell Portal 的 **SCP…** 编号、
    或**关联论文 DOI** → 直达本目录对应记录（match 为该数据集 item）；贴 **GSM / SRA** 号
    → **如实告知**它不在本目录收录范围（GEO 侧只收 Series 级、SRA 原始 reads 编号不索引）、指向来源原库，
    而**不是静默返回 0**（省去你先搜一遍才确定本目录没有）。也不要指望用任意论文信息反查到数据集。

    只读、离线、确定性。返回 `{ok, lookup}`：`lookup` = {is_identifier,kind,value,indexed,match,external_url,message}；
    identifier 不是一个可识别标识符 → `lookup=None`（这时该走 recommend_datasets 做语义检索）。
    空/空白 identifier → ToolError(`empty_query`)，客户端 isError=true。
    """
    try:
        ident = (identifier or "").strip()
        if not ident:
            raise _bad_request("empty_query", "identifier 不能为空。")
        from dataset_recommender.corpus.corpus import load_full_corpus
        from dataset_recommender.content import identifiers as _idf
        s = _settings()
        result = _idf.lookup(ident, lambda: load_full_corpus(s.data_dir, s.project_root))
        return {"ok": True, "lookup": result}
    except ToolError:
        raise
    except Exception as exc:
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def find_compatible_datasets(uid: str, limit: int = 20) -> dict:
    """给一个数据集找**元数据兼容**（同物种 + 兼容 chemistry/platform）的其它数据集（对齐网页
    `/api/compatible`）。

    **诚实边界**：只回「元数据兼容」——这是可整合的**必要非充分**条件；批次效应、注释体系、参考基因组
    版本、质控口径都可能让实际整合不可行。结果**始终附带这条 caveat**，**绝不**说「这些可以合并分析」。
    什么时候**别**用：它只按元数据字段比对、**不懂语义**——「找和这条研究内容/实验设计相似的数据集」
    该走 recommend_datasets；「这几条能不能真合并分析」也超出它的能力（批次/注释/参考基因组它不判）。

    只读、离线、确定性。返回 `{ok, seed, criteria, compatible[], total, caveat}`；每个 compatible 项带
    `_compat_basis`（凭 chemistry 还是 platform 相同）。空/空白 uid → ToolError(`empty_key`)；查不到 →
    `not_found`。与网页 `/api/compatible` 使用同一套实现，结果一致。

    参数 `limit`：返回兼容数据集条数上限，默认 20，范围 1-100（超出范围自动取最近的边界值，无法解析的值按 20 处理）。
    """
    try:
        u = (uid or "").strip()
        if not u:
            raise _bad_request("empty_key", "uid 不能为空。")
        try:
            lim = max(1, min(100, int(limit)))
        except (TypeError, ValueError):
            lim = 20
        from dataset_recommender.corpus.corpus import load_full_corpus
        from dataset_recommender.content import compatibility
        s = _settings()
        result = compatibility.find_compatible(u, load_full_corpus(s.data_dir, s.project_root), limit=lim)
        if result is None:
            raise _bad_request("not_found", f"没有找到 uid={u} 对应的数据集。")
        return {"ok": True, **result}
    except ToolError:
        raise
    except Exception as exc:
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def assess_feasibility(query: str, sources: list[str] | None = None) -> dict:
    """研究问题 → **可行性概览**（对齐网页 `/api/feasibility`）：有多少可复用数据、够不够、缺口在哪。

    聚合**通过硬过滤的全部候选**（放大 top_k 抓全命中集）：候选数 / 总细胞量**下限** / 物种·平台·年份·
    来源分布 / 可下载率 / 缺口。**诚实约束**：ArrayExpress 无研究级细胞数（覆盖率 0%、源库结构性
    缺失），全库 31% 无细胞数 → 「总细胞量」**必然是下限**，报告显式标注有多少数据集没有细胞数，绝不把下限
    当总量。`sources` 不传 → 仅 10x Genomics 基础语料；要跨来源检索，请把来源名列表传进来。确定性、离线、不调用 LLM、只读。
    什么时候**别**用：它只回**聚合概览**、不返回逐条数据集清单——要逐条结果用 recommend_datasets。
    例：「人类肺癌单细胞数据够不够做队列研究、缺口在哪」→ 用它；「列出这些数据集」→ 不用它。
    空/空白 query → ToolError(`empty_query`)。返回 `{ok, report, resolution_status}`。
    """
    try:
        q = (query or "").strip()
        if not q:
            raise _bad_request("empty_query", "query 不能为空。")
                                                                          
                                                               
        _validate_sources(sources)
        from dataset_recommender.app.workflow import DatasetRecommendationWorkflow, RecommendParams
        from dataset_recommender.retrieval import feasibility
        meta = DatasetRecommendationWorkflow().run_with_meta(
            RecommendParams(
                query=q, top_k=5000, use_llm=False, sources=sources, auto_parse_sources=True,
            )
        )
        total = getattr(meta, "result_total", len(meta.retrieved_data))
        truncated = total > len(meta.retrieved_data)
        report = feasibility.build_report(meta.retrieved_data, total, truncated)
        return {"ok": True, "report": report, "resolution_status": getattr(meta, "resolution_status", "results")}
    except ToolError:
        raise
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                             
_MCP_MAX_PACK_CHARS = 200_000


@mcp.tool()
@_logged
def build_task_pack(
    mode: str = "preview",
    query: str = "",
    selected_uids: "list | None" = None,
    sources: "list | None" = None,
    limit: "int | None" = None,
    scope: str = "primary",
    plan_token: str = "",
    snapshot_id: str = "",
    content_digest: str = "",
    retrieval_date: str = "",
    auto_parse_sources: bool = False,
    facet_filters: "list | None" = None,
    suppressed_constraints: "list | None" = None,
    lenient_dims: "list | None" = None,
    date_from: str = "",
    date_to: str = "",
) -> dict:
    """一句话任务包：一次检索 → 结果清单 + 下载脚本 + FAIR 自检 + 引文，**四件套口径一致**。

    分两步，`mode` 选：

    - `preview`（默认）：先给清单——会收录哪些数据集、**装不了什么**、只取主文件的声明、
      四档下载方式的计数、下载量下限、需要用户自己确认的事。**不产出任何文件。**
      把这些念给用户听、让他确认或勾掉几条，再进第二步。
    - `build`：产出 14 个文件的文本内容。必须带上预览返回的 `plan_token` / `snapshot_id` /
      `content_digest` / `retrieval_date` 与**同一套检索参数**；服务端会用它们重跑一遍再比对，
      对不上就明确告诉你哪里变了、**一个字节都不产出**。少勾几条是合法的，不会触发不一致。

    ## 必须念给用户听的三件事

    1. **scope 声明**：`primary_only_zh` 会按本次 scope 如实写好——`scope='primary'`（默认）
       是每个数据集只列 1 个代表性主文件、不含 FASTQ 等原始测序数据；`scope='all'` 是
       列出来源清单里的全部文件。用户如果搜的是「含 FASTQ 的数据」而本次是 primary，
       务必把这句原话读出来——否则他会以为包里有 FASTQ。
    2. **装不了什么**：`cannot_include` 里的数据集脚本下不了，只能到页面自己取。
       这不表示它们不能下载，只表示本工具没有这些来源的文件级清单。
    3. **md5 是来源公布的声明值**，本工具没有重新算过。

    ## 这个工具与网页版的不对称

    MCP 只返**文本**：`files[].content` 是每个文件的完整内容，由你或用户自己落盘。
    绝不返回二进制、绝不写盘。产物超过 20 万字符会报 `too_large` 而**不截断**——
    被截断的 download.sh 是一个会被执行的危险产物。

    参数：
      mode: 'preview' 或 'build'。
      query: 检索语句（preview 必填；build 时同样必填，用于重跑同一次检索）。
      selected_uids: build 时用户确认要打包的数据集编号，必须是预览候选池的子集。
      limit: 候选池条数（10 / 20 / 50，默认 10）。服务端在全部命中里按当前排序取前 N。
      其余参数与 recommend_datasets 同名同义，原样传即可。
    空 query → `empty_query`；条数/日期非法（含 from>to 倒挂）→ `bad_param`；来源拼错 → `bad_source`；
    指纹对不上 → `plan_mismatch` / `configuration_mismatch` / `snapshot_changed` / `catalog_content_changed`。
    """
    try:
        mode = str(mode or "preview").strip().lower()
        if mode not in ("preview", "build"):
            raise _bad_request("bad_param", "mode 只能是 preview 或 build。")
        _require_query(query)
        _validate_sources(sources)
        from dataset_recommender.content import item_view, task_pack
        from dataset_recommender.corpus.corpus import load_normalized_corpus

        try:
            capped = task_pack.sanitize_limit(limit)
            scope_value = task_pack.sanitize_scope(scope)
        except task_pack.TaskPackError as exc:
            raise _bad_request(exc.code, str(exc)) from None

        settings = _settings()
                                                                                       
                                                           
                                                
        eff_date_from = _require_iso_date(date_from, name="date_from")
        eff_date_to = _require_iso_date(date_to, name="date_to")
        _validate_date_window(eff_date_from, eff_date_to)
                                                                                
                                                                          
                                                                                                              
                                                             
                                              
        from dataset_recommender.app.workflow import sanitize_facet_filters, sanitize_suppressed, sanitize_lenient_dims
        facet_filters = sanitize_facet_filters(facet_filters)
        suppressed_constraints = sanitize_suppressed(suppressed_constraints)
        lenient_dims = sorted(sanitize_lenient_dims(lenient_dims))
        params = {
            "query": query.strip(), "query_effective": "",
            "sources": list(sources) if sources else None,
            "auto_parse_sources": bool(auto_parse_sources),
            "facet_filters": facet_filters or [], "suppressed_constraints": suppressed_constraints or [],
            "lenient_dims": lenient_dims or [], "date_from": eff_date_from, "date_to": eff_date_to,
            "limit": capped, "scope": scope_value,
        }
        from dataset_recommender.app.workflow import RecommendParams
        meta = _workflow().run_with_meta(
            RecommendParams(
                query=query.strip(), top_k=5000, use_llm=False,
                date_from=eff_date_from, date_to=eff_date_to, sources=sources,
                auto_parse_sources=bool(auto_parse_sources), facet_filters=facet_filters,
                suppressed_constraints=suppressed_constraints, lenient_dims=lenient_dims,
            )
        )
        records = load_normalized_corpus(settings.data_dir, settings.project_root, sources)
        by_uid = {}
        for record in records:
            raw = record.raw if isinstance(record.raw, dict) else {}
            uid = str(raw.get("dataset_uid") or "")
            if uid and uid not in by_uid:
                by_uid[uid] = record
        ordered = [str(r.get("dataset_uid") or "") for r in meta.retrieved_data]
        candidates = [uid for uid in ordered if uid in by_uid][:capped]
        honesty = {
            "active_filters": meta.active_filters, "coverage_caveats": meta.coverage_caveats,
            "unused_query_terms": meta.unused_query_terms, "result_total": meta.result_total,
            "search_trace_summary": (meta.search_trace or {}).get("summary", ""),
        }
        if not candidates:
                                                     
                                              
            return {"ok": True, "mode": mode, "plan": None, "result_total": meta.result_total,
                    "retrieval_params": params,
                    "message_zh": "这次检索没有命中任何数据集，没有可以打包的内容。",
                    "meta": {"deterministic": True, "offline": True, "llm_used": False}}

        wanted = candidates
        if mode == "build":
            try:
                wanted = task_pack.sanitize_uids_for_pack(selected_uids)
            except task_pack.TaskPackError as exc:
                raise _bad_request(exc.code, str(exc)) from None
            missing = [uid for uid in wanted if uid not in candidates]
            if missing:
                raise _bad_request("plan_mismatch",
                                   "这些数据集不在当前候选池里，请重新 preview 一次："
                                   + "、".join(missing[:5]))

        items = [item_view.build_item(by_uid[uid], include_introduction=True) for uid in candidates]
        try:
            full = task_pack.build_task_pack(
                query=query.strip(), items=items, records=records, scope=scope_value,
                retrieval_params=params, honesty=honesty, today=retrieval_date or None,
                membership={uid: "in_result_set" for uid in candidates},
            )
        except task_pack.TaskPackError as exc:
            raise _bad_request(exc.code, str(exc)) from None

        if mode == "build":
                                                     
                                                      
            missing_fp = [name for name, value in (("plan_token", plan_token),
                                                   ("snapshot_id", snapshot_id),
                                                   ("content_digest", content_digest))
                          if not (value or "").strip()]
            if missing_fp:
                raise _bad_request(
                    "bad_param",
                    "mode='build' 必须带上预览返回的指纹字段：" + "、".join(missing_fp)
                    + "；请先 mode='preview' 拿一次。",
                )
            for given, actual, code, why in (
                (snapshot_id, full["retrieval"]["snapshot_id"], "snapshot_changed", "数据集编号集合变过了"),
                (content_digest, full["retrieval"]["content_digest"], "catalog_content_changed", "有记录的内容被更新过"),
                (plan_token, full["plan_token"], "configuration_mismatch", "检索条件与预览时不一样"),
            ):
                if given != actual:
                    raise _bad_request(code, f"{why}，本次没有生成任何文件。请重新 preview 一次。")

        if mode == "preview":
            plan = full["plan"]
            from dataset_recommender.corpus import download_script
            return {
                "ok": True, "mode": "preview", "plan_token": full["plan_token"],
                "result_total": meta.result_total,
                "candidate_uids": candidates,
                "primary_only_zh": download_script.primary_only_sentence(plan),
                "cannot_include": plan["manual"],
                "tiers": plan["tiers"], "estimate": plan["estimate"],
                "ledger": plan["ledger"], "inspection": plan["inspection"],
                "retrieval": full["retrieval"], "retrieval_params": params,
                "items": plan["items"], "todo": full["todo"],
                "pack_files": list(task_pack.PACK_FILES),
                "note": ("这一步只给清单，没有产出任何文件。把 scope 声明（primary_only_zh）与"
                         "「装不了什么」念给用户听、拿到确认之后再用 mode='build' 产包。"
                         if scope_value == "primary" else
                         "这一步只给清单，没有产出任何文件。本次 scope=all：包里会列出每个数据集的"
                         "全部文件（primary_only_zh 已按此口径写好，照念即可），文件多、体积大，"
                         "提醒用户 build 的文本产物可能超过本通道 20 万字符上限；"
                         "把「装不了什么」念给用户听、拿到确认之后再用 mode='build' 产包。"),
                "meta": {"deterministic": True, "offline": True, "llm_used": False},
            }

        chosen = [it for it in items if it["dataset_uid"] in set(wanted)]
        try:
            pack = task_pack.build_task_pack(
                query=query.strip(), items=chosen, records=records, scope=scope_value,
                retrieval_params=params, honesty=honesty, today=retrieval_date or None,
                membership={uid: "in_result_set" for uid in wanted},
            )
            files = task_pack.render_files(pack)
        except task_pack.TaskPackError as exc:
            raise _bad_request(exc.code, str(exc)) from None
        total = sum(len(f["text"]) for f in files)
        if total > _MCP_MAX_PACK_CHARS:
            raise _bad_request("too_large",
                               f"产物共 {total} 字符，超过本通道上限 {_MCP_MAX_PACK_CHARS}。"
                               "请少选几个数据集——被截断的下载脚本是会被执行的危险产物，所以这里不截断。")
        return {
            "ok": True, "mode": "build", "plan_token": pack["plan_token"],
            "files": [{"path": f["path"], "content": f["text"], "bytes": len(f["text"].encode("utf-8"))}
                      for f in files],
            "todo": pack["todo"], "retrieval": pack["retrieval"],
            "note": "这些是文件内容，请原样写到同一个目录里（download.ps1 需要 UTF-8 BOM）。"
                    "本工具不写盘、不返回二进制。",
            "meta": {"deterministic": True, "offline": True, "llm_used": False},
        }
    except ToolError:
        raise
    except Exception as exc:                             
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def plan_query_edit(
    query: str = "",
    utterance: str = "",
    forced_op: str = "",
    dim: str = "",
    candidate_override: str = "",
    current_filters: "list | None" = None,
    resolution: "dict | None" = None,
    suppressed_constraints: "list | None" = None,
    lenient_dims: "list | None" = None,
    facet_filters: "list | None" = None,
    coverage_dims: "list | None" = None,
    date_from: str = "",
    date_to: str = "",
) -> dict:
    """把用户「再说一句话改条件」规划成一次具体改动。**只说明会改成什么，不执行检索、不装载数据集目录。**

    典型用法：先 `recommend_datasets` 拿到一批结果，用户接着说「换成小鼠」「再加一条：要有 FASTQ 的」
    「去掉组织限制」「放宽疾病」——把那句原话连同**上一次返回的条件**交给本工具，拿回下一步该怎么检索。

    ## 调用时必须守住的三件事

    1. **`current_filters` 与 `resolution` 必须原样取自上一次 `recommend_datasets` 的返回**
       （`understood.active_filters` 与 `interpretation.resolution`），**不要自己重新解析用户原话**。
       「现在按什么在筛」的唯一真源是那次返回：数据来源专名在解析之前就被整段剥掉了，
       自己重新解析会凭空多出一条物种条件，改写时还会把来源专名打碎、静默换掉检索范围。
    2. **`query` 是产生当前这批结果的那句话**，不是用户最新输入的那句。
    3. **`status='needs_confirm'` 时不要当成可以直接执行**：那意味着这一步会改到句子里的字，
       必须把 `message` 与 `removed_text` 念给用户听、拿到同意，再用 `next_request` 去调 `recommend_datasets`。
       只有 `status='auto_apply'`（只改了开关、一键可逆）才可以直接执行。

    ## 返回

    - `status`：`auto_apply`（只改开关，可直接执行）/ `needs_confirm`（改了句子，要用户确认）/
      `needs_choice`（同一项上冲突，把 `choices` 三个选项念给用户挑）/ `suggest`（给候选说法）/
      `not_understood`（没听懂，条件一个字都没改）/ `rejected`（改完连带动到别处，已拒绝执行）。
    - `next_request`：只有 auto_apply / needs_confirm 才非空，字段可直接喂给 `recommend_datasets`
      （query / suppressed_constraints / lenient_dims / facet_filters / date_from / date_to）。
      其余状态恒为 None —— 一个字节都没有改。
    - `board_view.rows`：当前条件分成四区（你这次要找的条件 / 在已筛出的结果里再缩小 / 已放宽 /
      这次没有用它筛），可直接念给用户听。
    - `removed_text`：这一步会从原句里中和掉的片段，带前后各几个字的上下文。
    - `verify.mismatches`：被拒绝时逐条说明哪里连带变了。

    参数：
      query: 产生当前这批结果的那句话。
      utterance: 用户这一句原话。
      forced_op: 直接指定改动类型（replace / widen / restart / suggest / add / remove / lenient）；
                 通常用于把 needs_choice 的某个选项执行下去，此时 utterance 传该选项给出的取值。
      dim: 直接指定要改哪一项（species / tissue / disease / platform / assay / modality / has_raw_data / date）。
      candidate_override: 用户手改过的整句；给了就以它为准（仍会再确认一次）。
      current_filters / resolution / suppressed_constraints / lenient_dims / facet_filters / coverage_dims:
                 上一次 recommend_datasets 返回里的对应内容，原样回传。
      date_from / date_to: 当前生效的发表时间范围。
    utterance 与 forced_op 同时为空 → ToolError(`empty_input`)；过长 → `too_large`；
    forced_op 非法 → `bad_param`。
    """
    try:
        from dataset_recommender.app import board
        try:
            plan = board.plan_edit(
                query or "",
                utterance or "",
                forced_op=forced_op or "",
                dim=dim or "",
                candidate_override=candidate_override or "",
                current_filters=current_filters,
                resolution=resolution,
                suppressed_constraints=suppressed_constraints,
                lenient_dims=lenient_dims,
                facet_filters=facet_filters,
                coverage_dims=coverage_dims,
                date_from=date_from or "",
                date_to=date_to or "",
                keyword_mapping=_settings().keyword_mapping,
            )
        except board.BoardError as exc:
            raise _bad_request(exc.code, str(exc)) from None
        plan["note"] = (
            "本工具只说明这一步会把条件改成什么，不执行检索。"
            "要拿结果，请用返回的 next_request 去调 recommend_datasets；"
            "next_request 为空就表示这一步没有执行、条件保持原样。"
        )
        return {"ok": True, **plan}
    except ToolError:
        raise
    except Exception as exc:                             
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def plan_action(utterance: str, has_results: StrictBool = False, result_total: StrictInt = 0) -> dict:
    """把用户随口说的一句话归一化成**封闭动词表里的一个动作**。**只出计划，不执行任何动作。**

    典型用法：用户说「把前 5 条打包下载」「这几个导出成引文」「存成压缩包给我」——
    把那句原话交给本工具，拿回一个结构化的 plan，再由**你**决定去调 `build_task_pack` /
    `build_reuse_pack` 等真正做事的工具。

    ## 本工具刻意**不**返回成句中文

    `has_results` / `result_total` 是**你自述**的，服务端无从核实。拿没核实的数字去发
    「已为你打包 20 条」这种断言，就是编造。成句文案留给能核实的那一端（真正执行完的人）渲染。

    ## 调用时必须守住的四件事

    1. **`kind == "route"` 不是执行**：`none`（这句话不是执行诉求）/ `search.new` / `refine.conditions`
       / `lookup.identifier` 四档要交回对应管线，不要当成「做了一件事」。
    2. **`blocked_reason == "no_results"` 时不要照做**：那是「读懂了，但现在没东西可做」。
       先去检索，别改说法——问题不在说法上。
    3. **`cancelled == true` 时不要照做**：用户**取消**了这个动作（「先别导入」「不要打包」）。
       动词保留是为了让你能如实回音「好，不做了」，不是授权——跳过一切执行，只回话。
    4. **`deltas` 与 `slot_sources` 必须念给用户听**：`clamped`（被收到上限）/ `guessed`
       （大模型替他填的数，他没说过）/ `dropped`（这一项没用上）三态是「不许静默偏离」的落点。

    ## 返回

    - `verb`：封闭词表里的一个（执行类 `pack.download` / `pack.preview` / `cite.export` /
      `reuse.pack` / `feasibility.run` / `files.show` / `curate.list` / `curate.import` /
      `curate.search_online` / `curate.check_updates` / `curate.sync_updates` / `curate.db_status` /
      `curate.remove` / `curate.restore`；路由类 `none` / `search.new` /
      `refine.conditions` / `lookup.identifier`）。`kind` 区分两类。curate.* 动词只出管护
      plan，真正执行由你另调 `curate_datasets`（plan→confirm_token→apply 两步）。
      环内专属动词（`search.rerun` / `rank` / `rerank` / `route.request`）**不在**本工具
      词表——它们只属于多步 agent 环；万一出现会进 `rejected` 并降 `none`。
    - `cancelled`：用户是否取消了这个动作（否定取消 → 动词照留 + 此标记，见上「守住的四件事」）。
    - `quoted`：判定依据，**保证是用户原话的字面子串**；执行类没有依据就不会给执行类 verb。
    - `slots` / `slot_sources` / `deltas`：参数、参数从哪来、与用户说的差在哪。
    - `rejected`：这句话里提到、但本工具不做的动作（只报不做）。
    - `confidence`：`low` **不表示要停下来问**——同样的 verb、同样的参数照做，只是更该核对一眼。
    - `source`：`llm`（大模型归一化）/ `rule`（大模型缺席时按关键词猜的，此时只会给
      `pack.preview` 这类不落盘的动作，且带 `caveat_zh`）。

    参数：
      utterance: 用户这一句原话（必填，上限 500 字）。
      has_results: 你手上现在是否已有一批检索结果。
      result_total: 那批结果的命中总数。
    空 utterance → ToolError(`empty_input`)；过长 → `too_large`。
    """
    try:
        from dataset_recommender.agent import action_plan as _ap
        try:
            plan = _ap.plan_action(
                utterance or "",
                has_results=bool(has_results),
                result_total=max(0, int(result_total or 0)),
            )
        except _ap.ActionPlanError as exc:
            raise _bad_request(exc.code, str(exc)) from None
        return {"ok": True, "plan": plan}
    except ToolError:
        raise
    except Exception as exc:                             
        raise _internal_error(exc) from exc


@mcp.tool()
@_logged
def verify_local_assets(directory: str, max_files: int = 20000) -> dict:
    """扫描一个**本地目录树** → 与本目录文件级清单（10x 基础语料 15119 个文件的 md5/大小/文件名）比对 →
    实验室资产台账（**只读、纯校验和、不联网、不改任何文件**；对齐 CLI `scripts/scan_lab_assets.py`）。

    回答：本地这堆文件里哪些属于本目录已知数据集、各 md5 核验通过没有、哪些名字大小对上但内容不符
    （md5_mismatch=本地副本与来源声明不一致，可能损坏/换版）、哪些本目录不认识。

    **只算 md5、绝不读数据内容**（md5 本就是来源清单里的校验和）；巨文件（>8 GiB）跳 md5、只按名+大小
    比对、hashed=false，不冒充已校验。**诚实边界**：md5 清单只覆盖 10x 基础语料；外部库（AE/HCA/CELLxGENE）
    只有数据集页面 url、无单文件清单 → 那些数据集的本地文件无法 md5 核验（见 report.caveat）。

    参数：
      directory: 要扫描的本地目录路径。
      max_files: 扫描文件数上限（默认 20000；硬上限 = 台账模块扫描上限 200000，超限静默钳位——
                  防误扫超大目录与全盘 md5 DoS；达到上限时 report.truncated=true）。
    空 directory → ToolError(`empty_path`)；目录不存在 → {ok:true, report:{root_exists:false,…}}（不算错误）。
    返回 `{ok, report}`。为控制响应体积、也为不把本地路径全量外传：每个数据集不回传逐文件路径
    （只回计数汇总）；数据集列表只回前 100 个（`report.n_datasets_present` 是完整计数）；
    unmatched 只回前 20 条样本（`report.n_unmatched` 是完整计数）。
    """
    try:
        d = str(directory or "").strip()
        if not d:
            raise _bad_request("empty_path", "directory 不能为空。")
        from dataset_recommender.content import lab_ledger
        try:
            mf = int(max_files)
        except (TypeError, ValueError):
            mf = lab_ledger.MAX_SCAN_FILES
        if mf <= 0:
            mf = lab_ledger.MAX_SCAN_FILES
                                                                
                                                                       
                                                  
        if mf > lab_ledger.MAX_SCAN_FILES:
            mf = lab_ledger.MAX_SCAN_FILES
        index = lab_ledger.build_manifest_index()
        scan = lab_ledger.scan_directory(d, index, max_files=mf)
        report = lab_ledger.build_ledger_report(scan, index)
                                                             
        for ds in report.get("datasets", []):
            ds.pop("files", None)
        report["datasets"] = report["datasets"][:100]
        report["unmatched"] = report["unmatched"][:20]
        return {"ok": True, "report": report}
    except ToolError:
        raise
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                                  
                                                              
                                                                  
                                                      
                                                                        
                                                          
                                                                      
                                    
_MAX_PROVISION_FILES = 500
_DEFAULT_PROVISION_FILES = 50


@mcp.tool()
@_logged
def provision_dataset(
    dataset_uid: str,
    dest_dir: str,
    scope: str = "primary",
    include_flagged: bool = False,
    max_files: StrictInt | None = None,
    dry_run: bool = False,
) -> dict:
    """把一个数据集的文件**真正下载到调用方指定的目录**（对齐 CLI `scripts/provision_dataset.py`）。

    ⚠️ **联网 + 写盘**工具（本服务器三个写盘工具之一，另两个是 upload_dataset 与 curate_datasets）：
    **仅在调用方给定 dest_dir 时写盘，绝不写 database/**（executor 对冻结基准 database/base/ 与在仓
    数据目录 fail-closed 拒绝）；只下 https 且主机在计划白名单内的文件；先写 `.part`、核对通过才原子
    改名，核对不符改名 `.corrupt` 留证据；巡检旗标（dead/size_mismatch）文件默认跳过。
    下载结果可用 `scripts/record_provision_results.py` 回写活台账（本工具自己不写台账）。

    参数：
      dataset_uid: 取自 recommend_datasets 返回的 candidates[*].dataset_uid。本工具只服务本机文件清单
                   覆盖到的数据集（10x 基础语料）；查不到 uid → `unknown_uid`，不降级猜页面链接。
      dest_dir:    下载目标目录，**必须是绝对路径**（相对路径会把文件下到意想不到的地方 → `bad_out_dir`）。
                   不存在会自动创建。落 database/base/ 或在仓数据目录内 → `protected_out_dir`。
      scope:       "primary"（默认）= **只下 1 个代表性主文件**（分析师最先要的那份，不含 FASTQ）；
                   "all" = 该数据集文件清单里的全部文件（请配合 max_files）。
      include_flagged: False（默认）跳过巡检旗标文件；True 才连旗标文件一起下。
      max_files:   单次执行文件数上限（默认 50，硬上限 500）。计划文件数超限 → `bad_param` **报错**，
                   不静默截断——防止一次误拉整库（全库 15119 文件）。
      dry_run:     True = 只返回将要下载的文件清单与总量，**不下载任何文件**（先念给用户确认再真下）。

    返回 dict（ok=true 时）：
      dry_run / scope / dest_dir / planned_files / planned_bytes（已知字节合计，缺声明的不计）
      counts     —— 逐文件状态计数（ok / size_ok / unverified / md5_mismatch / size_mismatch /
                    unreachable / rejected / skipped_flagged；dry_run 时全 0）。
      files      —— 逐文件结果（status / bytes / md5 核对 / saved_as 相对落盘路径 / error 摘要）；
                    dry_run 时为计划清单（filename / bytes / md5sum / download_url / flag_kind）。
      summary_zh —— 人类可读摘要（与 CLI 同口径）。
      write_boundary / next —— 写盘边界声明 + 下一步提示（verify_local_assets 复核；台账回写走 CLI）。
    错误契约：非法请求 → isError=true（empty_uid / unknown_uid / bad_out_dir / protected_out_dir /
    bad_param(scope 越界、max_files 非法或计划超限) / no_allowed_hosts）。
    """
    try:
        from dataset_recommender.corpus import download_executor as DE
        from dataset_recommender.corpus import download_plan as DP
        uid = str(dataset_uid or "").strip()
        if not uid:
            raise _bad_request(
                "empty_uid",
                "dataset_uid 不能为空；取自 recommend_datasets 返回的 candidates[*].dataset_uid。",
            )
        scope_norm = str(scope or "primary").strip().lower()
        if scope_norm not in DP.SCOPES:
            raise _bad_request(
                "bad_param",
                f"scope 只能是 {list(DP.SCOPES)} 之一（primary=只下主文件，默认；all=全部文件），收到「{_clip(scope)}」。",
            )
        eff_max = _DEFAULT_PROVISION_FILES if max_files is None else max_files
        if isinstance(eff_max, bool) or not isinstance(eff_max, int) or eff_max <= 0:
            raise _bad_request("bad_param", f"max_files 必须是正整数（≥1）或省略，收到「{_clip(max_files)}」。")
        if eff_max > _MAX_PROVISION_FILES:
            raise _bad_request(
                "bad_param",
                f"max_files={eff_max} 超过硬上限 {_MAX_PROVISION_FILES}（防一次误拉整库）；请分批下载或改用 CLI。",
            )
        try:
            out_root = DE.resolve_out_dir(dest_dir)                                             
            rows, _allowed = DE.build_rows([uid], scope=scope_norm)                               
        except DE.ProvisionError as exc:
            raise _bad_request(exc.code, str(exc)) from None
        if len(rows) > eff_max:
            raise _bad_request(
                "bad_param",
                f"本次计划 {len(rows)} 个文件，超过 max_files={eff_max}：宁可报错也不静默截断。"
                f"请缩小范围（scope='primary'）或显式调大 max_files（硬上限 {_MAX_PROVISION_FILES}）。",
            )
        planned_bytes = sum(int(r["bytes"]) for r in rows if r.get("bytes"))
        if dry_run:
            return {
                "ok": True, "dry_run": True, "dataset_uid": uid, "scope": scope_norm,
                "dest_dir": str(out_root), "planned_files": len(rows), "planned_bytes": planned_bytes,
                "counts": {s: 0 for s in DE.STATUSES},
                "files": [{"filename": r["filename"], "bytes": r.get("bytes"), "md5sum": r.get("md5sum"),
                           "download_url": r["download_url"], "flag_kind": r.get("flag_kind")}
                          for r in rows],
                "summary_zh": (
                    f"预演：将下载 {len(rows)} 个文件（声明字节合计约 {planned_bytes}）到 {out_root}。"
                    "本次没有下载任何文件。"
                ),
                "write_boundary": "仅在调用方给定 dest_dir 时写盘；绝不写 database/ 与在仓数据目录。",
                "next": ["把计划念给用户确认后，用 dry_run=False 再调一次即真下载。"],
            }
        report = DE.provision([uid], str(out_root), scope=scope_norm,
                              include_flagged=bool(include_flagged))
        return {
            "ok": True, "dry_run": False, "dataset_uid": uid, "scope": scope_norm,
            "dest_dir": str(out_root), "planned_files": len(rows), "planned_bytes": planned_bytes,
            "counts": report.counts(),
            "files": [r.to_dict() for r in report.results],
            "summary_zh": report.summary_zh(),
            "write_boundary": "只在 dest_dir 内写盘；绝不写 database/ 与在仓数据目录。",
            "next": [
                f"可用 verify_local_assets(directory=\"{out_root}\") 复核这些文件与 10x 清单的 md5 台账。",
                "要把本次实测回写活台账：CLI 跑 scripts/provision_dataset.py 加 --report-json 产出报告，"
                "再用 scripts/record_provision_results.py 回写（本工具不写台账）。",
            ],
        }
    except ToolError:
        raise                                                                                             
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                           
                                         
                                                    
                                                                         
                                                            
                                                                          


@mcp.tool()
@_logged
def curate_datasets(
    action: str,
    query: str | None = None,
    source: str | None = None,
    species: str | None = None,
    limit: StrictInt | None = None,
    filename: str | None = None,
    payload_json: str = "",
    confirm_token: str | None = None,
    force: bool = False,
    dry_run: bool = True,
) -> dict:
    """管护写工具（写工具 ③）：**plan 默认（dry_run=True）零写盘，apply（dry_run=False + confirm_token）才写盘/联网**。

    对话式数据库管护（对齐 Web `/api/curate/*` 与 CLI `scripts/curate_datasets.py`），管护对象限
    `database/external/` 的 `upload_*` 命名空间（用户上传 + 联网搜索入库文件；官方五源快照
    `not_curatable`，仍走人工流水线），**绝不**碰冻结基准 `database/base/`。

    action（必填，未知动作 → `bad_action`）：
      list           —— 清点外部库（文件/条数/来源/修改时间）+ 回收站内容。纯只读，dry_run 参数对它无意义。
      check_updates  —— **只读**检查来源更新（`source` 可选，不填查全部；在线比对失败按既有契约
                        逐源如实降级为 snapshot + note，不抛）。无 confirm_token、不落盘
                        （联网比对记一行请求账本）。
      sync_updates   —— 检查更新 → 有新增则自动入库的复合流（`source` 可选，不填查全部；写侧
                        uploads 管线 + 回收站可撤回，**整次可撤回**——返回 `operation_id` +
                        `created_files[]`，用 Web `/api/curate/recall` 可整次撤回）。
                        **整任务文件锁**：另一个 sync 正在跑 → `sync_busy`（fail-closed，不排队）。
                        必须显式 dry_run=False；dry_run=True 拒绝执行，预览请用 check_updates。
                        无 confirm_token——原子调用没有信任边界要跨。
      import         —— 本地 JSON 导入 + 内容 hash 去重（撞已有 external 记录默认拒绝，`force=true` 可覆盖）。
                        `payload_json` = 数据集 JSON 文本（记录数组或 {"records":[…]} 对象）；`filename` 落盘名
                        （自动加 upload_<时间戳>_ 前缀，省略时默认 curate_import.json）；`source` 归属来源名。
      search_online  —— **联网**按关键词搜官方源（`source` 默认 "arrayexpress"，可选
                        arrayexpress / cellxgene / hubmap / single_cell_portal / hca / 10x / geo，
                        别名如 scp、ncbi geo、cellxgene discover 可识别；`species` 本地子串过滤；
                        `limit` 候选上限 1–100 默认 20）。plan 返回候选 preview（**不落盘**，只记请求账本
                        `.userdata/curate_net_ledger.jsonl`）；apply 时把 plan 的整个返回 JSON 原样作为
                        `payload_json` 回传（候选在 `candidates` 键里，调包/改动 → `token_mismatch` 零写入）。
      remove         —— 回收站式**可逆**删除：`filename`（external 库里的叶子文件名）移入
                        `.userdata/recycle/` + manifest 账本 + 缓存即时失效；不是真删除。
      restore        —— 把回收站文件移回 external 原文件名；`filename` = 回收站里的文件名
                        （含时间戳前缀，用 list 查）。

    两步确认（复用任务包指纹模式）：plan（dry_run=True，默认）返回 preview + `confirm_token`；
    apply（dry_run=False）**必须回传 confirm_token**（缺 → `bad_param`），服务端用当前状态重算比对，
    不一致 → `token_mismatch`、**一个字节不动**。

    返回 dict（ok=true 时）：`action` / `dry_run` / 各动作的 preview 或执行回执字段 /
    `write_boundary`（写盘/联网边界声明）/ `next`（下一步提示）。
    错误契约：非法请求 → isError=true（`bad_action` / `bad_param` / `unknown_file` / `not_curatable` /
    `token_mismatch` / `duplicate_content` / `source_not_registered` / `network_error` /
    `no_candidates` / `too_large`；摄取层 `bad_file` / `bad_encoding` / `invalid_json` / `no_records` 原码透传）。
    """
    from dataset_recommender.corpus import corpus_curation as cc
    try:
        action_name = str(action or "").strip()
                                                                  
                                                                                  
                                                                      
                                                                    
                                                                                  
                            
        if action_name in ("check_updates", "sync_updates"):
            if action_name == "sync_updates" and dry_run:
                raise _bad_request(
                    "bad_param", "sync_updates 不支持 dry_run；预览请用 check_updates，"
                    "确认执行入库后显式传 dry_run=False。")
            src_list = [source] if str(source or "").strip() else None
            if action_name == "check_updates":
                result = cc.check_updates(src_list, project_root=_settings().project_root)
            else:
                result = cc.sync_updates(src_list, project_root=_settings().project_root)
            nxt: list[str] = []
            if action_name == "check_updates":
                nxt.append("发现疑似新增、用户说「有就入库」？用 sync_updates 一步检查+入库。")
            elif result.get("created_files"):
                nxt.append("如需撤回本次全部导入，用 Web 端点 POST /api/curate/recall（传 operation_id）。")
            return {"ok": True, **result,
                    "write_boundary": cc.write_boundary_zh(action_name, dry_run=bool(dry_run)),
                    "next": nxt}
        payload_bytes: bytes | None = None
        plan_result: dict | None = None
        raw_payload = str(payload_json or "").strip()
        if raw_payload:
            if len(raw_payload.encode("utf-8")) > _MAX_UPLOAD_BYTES:
                raise _bad_request(
                    "too_large",
                    f"payload_json 超过上限 {_MAX_UPLOAD_BYTES}（64 MB）；单细胞元数据 JSON 远小于此。",
                )
            if str(action or "").strip() == "search_online":
                try:
                    plan_result = json.loads(raw_payload)
                except ValueError as exc:
                    raise _bad_request("bad_param", f"payload_json 不是合法 JSON：{exc}") from None
                if not isinstance(plan_result, dict):
                    raise _bad_request(
                        "bad_param",
                        "search_online 的 payload_json 必须是 plan 返回的完整 JSON 对象（含 candidates）。",
                    )
            else:
                payload_bytes = raw_payload.encode("utf-8")
        result = cc.run_curate_action(
            action,
            dry_run=bool(dry_run),
            query=query,
            source=source,
            species=species,
            limit=limit,
            filename=filename,
            payload_bytes=payload_bytes,
            plan_result=plan_result,
            confirm_token=confirm_token,
            force=bool(force),
            project_root=_settings().project_root,
        )
        nxt: list[str] = []
        if result.get("dry_run") and result.get("confirm_token"):
            nxt.append("把预览念给用户确认后，用 dry_run=False + 回传 confirm_token 再调一次即真执行。")
            if result.get("action") == "curate.search_online":
                nxt.append("apply 时把本次返回的整个 JSON 原样作为 payload_json 回传（候选在 candidates 键里）。")
        return {"ok": True, **result, "next": nxt}
    except ToolError:
        raise
    except cc.CurateError as exc:                                                      
        raise _bad_request(exc.code, exc.hint) from exc
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                  
                                                                                    
                                                         
                                                                        
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024                                              

                                                        
                                        
                                                                            
                                                           
                                              
_SENSITIVE_UPLOAD_DIRS = (".userdata", ".git", ".venv", "database")


def _reject_sensitive_upload_path(raw_path: str, p: "Path") -> None:
    """敏感路径拒绝（raise ToolError`sensitive_path`）：① .env* 密钥文件（任意位置）；
    ② 仓库内 .userdata/.git/.venv/database 四目录（运行状态/版本库/冻结与外部数据——从 database/
    搬运毫无意义且污染检索面）；③ 仓库外同名敏感目录（.git/.userdata/.venv，纵深防御）。"""
    try:
        resolved = p.resolve()
    except OSError:                                         
        resolved = p.absolute()
    if resolved.name.startswith(".env"):
        raise _bad_request("sensitive_path", "path 不能指向 .env* 配置/密钥文件；只接受数据集 JSON。")
    root = _settings().project_root
    try:
        root_resolved = Path(str(root)).resolve()
    except OSError:
        root_resolved = Path(str(root)).absolute()
    rel = resolved.relative_to(root_resolved) if resolved.is_relative_to(root_resolved) else None
    if rel is not None and rel.parts and rel.parts[0] in _SENSITIVE_UPLOAD_DIRS:
        raise _bad_request(
            "sensitive_path",
            f"path 不能指向仓库内 {rel.parts[0]}/ 目录（运行状态/版本库/语料数据）。"
            "请上传来自外部目录（如下载/导出目录）的数据集 JSON。",
        )
    for anc in resolved.parents:
        if anc.name in (".git", ".userdata", ".venv"):
            raise _bad_request("sensitive_path", f"path 位于敏感目录 {anc.name} 内，拒绝读取。")


@mcp.tool()
@_logged
def upload_dataset(
    records: list | dict | None = None,
    path: str | None = None,
    filename: str | None = None,
    source: str | None = None,
) -> dict:
    """把一份数据集 JSON 摄取进**外部库** database/external/（对齐网页 `/api/upload`）——
    本服务器三个写盘工具之一（另两个是 provision_dataset，写调用方指定目录；curate_datasets，管护外部库）；摄取后即时可被
    browse_datasets / recommend_datasets(sources=[…]) 查到。

    安全边界：内容**只写入** database/external/，**绝不**改动内置的基准数据 database/base/；默认检索不读
    外部库 → 上传不会影响内置数据的检索结果。上传的数据会成为一个**可勾选的来源**（逐条打来源标签，
    计数口径正确、不会被算成 10x Genomics 的数据）。
    ⚠️ 与 16 个只读工具不同：本工具**会写磁盘**，且文件名带时间戳 → 同一份输入上传两次会落成两个文件。

    输入（records 与 path **二选一**）：
      records:  数据集 JSON，**直接传结构化值**（首选，自足离线）：记录数组 `[ {…}, … ]`（首选），
                或包裹对象 `{ "records": [ {…} ], "source": "…" }`。**不要传 JSON 字符串**——直接传数组/对象即可
                （MCP 会按结构校验；字符串反而会被框架预解析，易踩坑）。每条记录建议含 dataset_name / species /
                url 等字段（缺 dataset_name 会在 warnings 提示）。
      path:     仅本地 stdio 可用的 JSON 文件**路径**；在线 MCP 仅接受 records，不读取服务器路径。
                仅用于**读取**输入字节（utf-8 / 带 BOM 均可）；
                写入永远落 database/external/，与 path 所在目录无关。
      filename: 落盘用文件名（会自动加 upload_<时间戳>_ 前缀）。**须以 .json 结尾**；
                省略时取 path 的文件名，若无 path 则默认 "upload.json"。
      source:   归属来源名（对齐网页表单 ?source=）。优先级：每条自带 source > 本参 > 包裹对象 source >
                默认「用户上传」。给个有意义的来源名，便于之后按来源勾选筛选。

    返回 dict（ok=true 时）：
      filename / saved_to —— 实际落盘文件名（含 upload_ 前缀）/ 相对仓库根路径（database/external/upload_…json）。
      record_count        —— 摄取的记录条数。
      sources             —— {来源名: 条数}（逐条打标后的计数）。
      warnings            —— 可读校验提示（缺 dataset_name / 物种非通用名等；空数组=无问题）。
      next                —— 下一步提示：用 browse_datasets(source="…") 或 recommend_datasets(sources=["…"]) 查刚上传的数据。

    错误契约：**非法请求** → 客户端 isError=true（消息形如「Error executing tool
    upload_dataset: <码>: <提示>」）：
        · empty_input（records 与 path 都没给）
        · bad_param（records 与 path 同时给了；二选一）
        · not_found（path 指向的文件不存在/不可读）
        · sensitive_path（path 指向敏感位置：.env* 密钥文件、仓库内 .userdata/.git/.venv/database
          或任意 .git/.userdata/.venv 目录——运行状态/版本库/语料数据不是上传源，fail-closed）
        · too_large（内容超过 64 MB 上限）
        · bad_file（文件名/路径不以 .json 结尾）
        · bad_encoding（path 文件不是合法 UTF-8）
        · invalid_json（path 文件不是合法 JSON）
        · no_records（没解析出任何数据集记录，如空数组）
      **写盘只在全部校验通过后发生**（校验失败绝不留半个文件）。
    """
    try:
        import json as _json
        records_provided = records is not None
        raw_path = (path or "").strip()
        if not records_provided and not raw_path:
            raise _bad_request(
                "empty_input",
                "需给出 records（数据集 JSON：记录数组或 {records:[…]} 对象）或 path（本地 JSON 文件路径）之一。",
            )
        if records_provided and raw_path:
            raise _bad_request(
                "bad_param",
                "records 与 path 二选一：同时给了会有歧义；只传其中一个。",
            )

        if records_provided:
                                                                                
            raw_bytes = _json.dumps(records, ensure_ascii=False).encode("utf-8")
            default_name = "upload.json"
        else:
            p = Path(raw_path).expanduser()
            _reject_sensitive_upload_path(raw_path, p)                                                  
            if not p.is_file():
                raise _bad_request("not_found", f"path 指向的文件不存在或不是文件：{_clip(raw_path)}")
            try:
                size = p.stat().st_size
            except OSError as exc:
                raise _bad_request("not_found", f"无法读取 path（{type(exc).__name__}）：{_clip(raw_path)}")
            if size > _MAX_UPLOAD_BYTES:
                raise _bad_request(
                    "too_large",
                    f"文件 {size} 字节，超过上限 {_MAX_UPLOAD_BYTES}（64 MB）；单细胞元数据 JSON 远小于此。",
                )
            raw_bytes = p.read_bytes()
            default_name = p.name

        if len(raw_bytes) > _MAX_UPLOAD_BYTES:
            raise _bad_request(
                "too_large",
                f"content {len(raw_bytes)} 字节，超过上限 {_MAX_UPLOAD_BYTES}（64 MB）；请分批上传。",
            )

        eff_filename = (filename or "").strip() or default_name
                                                                                        
        safe_name = new_upload_name(eff_filename)

        s = _settings()
        result = ingest_dataset(
            raw_bytes=raw_bytes,
            safe_name=safe_name,
            project_root=s.project_root,                                                    
            form_source=(source or ""),
            note="用户上传（本地 MCP upload_dataset 工具）。",
        )
        first_source = next(iter(result.sources), "用户上传")
        return {
            "ok": True,
            "filename": result.filename,
            "saved_to": result.saved_to,
            "record_count": result.record_count,
            "sources": result.sources,
            "warnings": result.warnings,
            "next": (
                f"已入外部库、即时可检索：browse_datasets(source=\"{first_source}\") 或 "
                f"recommend_datasets(sources=[\"{first_source}\", …]) 即可查到刚上传的数据。"
            ),
        }
    except UploadError as exc:
                                                                                      
        raise _bad_request(exc.code, exc.hint) from exc
    except ToolError:
        raise                                                                       
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                         
def _llm_failure_code(error: str | None) -> str:
    text = (error or "").lower()
    if "missing" in text and ("key" in text or "token" in text):
        return "missing_key"
    if "httperror 401" in text or "httperror 403" in text or "unauthorized" in text:
        return "authentication_failed"
    if "httperror 429" in text or "rate limit" in text:
        return "rate_limited"
    if "httperror 400" in text or "httperror 404" in text or "model" in text:
        return "endpoint_or_model_error"
    if "url error" in text or "timeout" in text or "dns" in text or "connection" in text:
        return "network_error"
    if "unsupported provider" in text:
        return "unsupported_provider"
    if "non-json" in text or "missing choices" in text or "content is empty" in text:
        return "invalid_response"
    return "provider_error"


def _llm_status_payload(check_connection: bool = False) -> dict:
    """Return a secret-free LLM configuration summary and optionally perform one tiny API call."""
    from dataset_recommender.llm.config import external_llm_env_status
    from dataset_recommender.llm.llm_client import call_llm, load_llm_config

    config = load_llm_config(project_root=get_app_paths().config_root)
    external = external_llm_env_status()
                                           
    is_mock = bool(config.mock_llm) or (config.provider or "").strip().lower() == "mock"
    configured = bool(config.api_key) or is_mock
    payload: dict = {
        "ok": True,
        "provider": config.provider,
        "model": config.model,
        "configured": configured,
        "mock_mode": is_mock,
        "enabled_by_default": bool(config.enable_llm),
        "external_env": external,
        "checked": bool(check_connection),
        "connection": "not_checked",
        "error_code": None,
    }
    if external["configured"] and not external["readable"]:
        payload["configuration_warning"] = "external_env_file_unavailable"

    if not check_connection:
        return payload
                                                    
                                                                                   
    if is_mock:
        payload["connection"] = "mock"
        payload["error_code"] = None
        return payload
    if not configured:
        payload["connection"] = "failed"
        payload["error_code"] = "missing_key"
        payload["hint"] = "按安装教程重新配置 API Key（Windows 可直接重跑 scripts/setup_mcp.ps1；macOS / Linux 见教程附录 A），然后重启客户端。"
        return payload

                                                                                          
                                                                                                 
    config.enable_llm = True
    result = call_llm("Reply with exactly one word: OK", config)
    if result.succeeded:
        payload["connection"] = "success"
        return payload

    code = _llm_failure_code(result.error)
    payload["connection"] = "failed"
    payload["error_code"] = code
    payload["hint"] = {
        "authentication_failed": "检查 API Key 是否有效、是否属于当前服务商。",
        "rate_limited": "检查账户额度或稍后重试。",
        "endpoint_or_model_error": "检查 Base URL 与模型名是否匹配服务商。",
        "network_error": "检查网络、代理、证书与防火墙。",
        "unsupported_provider": "Provider 只能使用 openai-compatible 或 zhipuai。",
        "invalid_response": "接口已响应，但不是兼容的 Chat Completions 返回格式。",
        "missing_key": "重新运行安装脚本配置 API Key。",
    }.get(code, "接口调用失败；请用网页 /api/diagnose 或服务商控制台继续定位。")
    return payload


@mcp.tool()
@_logged
def biodata_llm_status(check_connection: bool = False) -> dict:
    """脱敏检查 MCP 的 LLM API 配置。

    默认 check_connection=False：只读本地配置，**不联网、不消耗额度**；返回 provider/model、是否检测到
    key、项目外密钥文件是否可读。显式传 True 才发送一次最小 Chat Completions 探测；只返回稳定错误码，
    不返回 API Key、密钥文件路径、请求体或服务商原始错误正文。
    """
    try:
        return _llm_status_payload(bool(check_connection))
    except Exception as exc:
        raise _internal_error(exc) from exc


                                                                  
@mcp.tool()
@_logged
def biodata_status() -> dict:
    """健康自检 + 库存清单。装完让 agent 调一次即可确认服务器与数据是否就绪。

    返回：各来源计数（10x 置顶）、下载直链索引是否就绪、直链快照日期、确定性/离线声明。
    `tools` 是**服务器全部工具清单**（单一真源 `_EXPECTED_TOOLS`）：在线形态下 `tools/list`
    会再摘掉两个本机文件系统语义工具（`provision_dataset` / `verify_local_assets`），
    因此在线时这份清单可能比实际可调用的多两个——以 `tools/list` 为准。
    """
    try:
        from dataset_recommender.corpus import downloads, inspection
        from dataset_recommender.corpus.corpus import available_sources
        s = _settings()
        srcs = available_sources(s.data_dir, s.project_root)
        snap = inspection.snapshot_info() or {}
        return {
            "ok": True,
            "server": "biodata",
            "sources": srcs,
            "corpus_total": sum(x["count"] for x in srcs),
            "download_index_ready": downloads.is_available(),
            "links_snapshot": _LEDGER_SNAPSHOT,
            "inspection_ready": inspection.is_available(),
            "inspection_snapshot": snap.get("snapshot_date"),
            "server_version": _SERVER_VERSION,
            "deterministic_offline": True,
            "llm": _llm_status_payload(False),
            "tools": list(_EXPECTED_TOOLS),
        }
    except Exception as exc:                                         
        raise _internal_error(exc) from exc


                                                                    
def _parse_intent(
    query: str,
    *,
    sources: "list[str] | None" = None,
    auto_parse: bool = True,
) -> dict:
    """用 Web/MCP 共用入口解释查询，再投影成 JSON 友好 dict。"""
    from dataset_recommender.corpus.corpus import known_source_values
    from dataset_recommender.retrieval.query_parser import parse_query
    from dataset_recommender.retrieval.search_request import resolve_search_request
    from dataset_recommender.app.workflow import intent_projection
    s = _settings()
    resolution = resolve_search_request(
        query,
        sources,
        known_source_values(s.data_dir, s.project_root),
        auto_parse_sources=auto_parse,
    )
    intent = parse_query(resolution.parsed_query, s.keyword_mapping)
    payload = intent_projection(intent)
    payload["source_resolution"] = resolution.as_dict()
    payload["effective_sources"] = list(resolution.sources or ["10x Genomics"])
    return payload


def _advisory(intent: dict, result_total: int) -> str | None:
    """诚实提示：查询未被有效约束时，明确告知结果是通用/宽泛的，而非精确匹配。

    背景（试跑发现）：受控词表不建模细胞类型等维度，故「人类免疫细胞」这类查询里的「免疫」
    会被当填充词丢掉，只剩「物种+平台」这类宽泛约束 → 命中面极大、靠前结果排序可能与用户真实
    意图无关。此提示把这种「宽泛/未精确筛选」的情形**显式**告诉父 agent，不改动管线、不影响确定性。
    仅依据已抽取的约束判断（纯函数，便于单测）。"""
    from dataset_recommender.retrieval.query_parser import _DIM_LABEL_CN                        
                                        
    if intent.get("abstain") or intent.get("parse_status") in ("abstained", "clarification_required"):
        return None
    constraints = intent.get("constraints") or {}
    excluded = intent.get("excluded_constraints") or {}
    dims = [d for d in constraints if constraints.get(d)]
    ex_dims = [d for d in excluded if excluded.get(d)]
    has_raw = intent.get("has_raw_data_required") is not None
                                                            
    if not dims and not ex_dims and not has_raw:
        return ("未从查询中识别到任何可过滤条件（物种/组织/疾病/平台/技术/原始数据），"
                "以下为通用结果、未按你的查询精确筛选；请补充条件以获得相关结果。")
    if set(dims) <= {"species", "platform", "modality"} and not has_raw and result_total >= 100:
        named = "、".join(_DIM_LABEL_CN.get(d, d) for d in dims)
        return (f"查询较宽泛：仅按 {named} 过滤，命中 {result_total} 条，靠前结果排序可能并不精确；"
                "建议追加 组织/疾病/技术 等条件收窄（本工具没有「细胞类型」这类筛选维度，这类词不会参与过滤）。")
    return None


from dataset_recommender.app.mcp_agent_tools import AGENT_TOOLS, register_agent_tools
register_agent_tools(mcp, root=lambda: get_app_paths().data_root)

_EXPECTED_TOOLS = (
    "agent_chat",
    "agent_conversations",
    "agent_projects",
    "agent_task",
    "agent_replay",
    "agent_learning",
    "recommend_datasets",
    "get_file_manifest",
    "parse_constraints",
    "browse_datasets",
    "get_dataset_introduction",
    "assess_dataset_fair",
    "build_reuse_pack",
    "lookup_identifier",
    "find_compatible_datasets",
    "assess_feasibility",
    "plan_query_edit",
    "plan_action",
    "build_task_pack",
    "verify_local_assets",
    "provision_dataset",
    "curate_datasets",
    "upload_dataset",
    "biodata_status",
    "biodata_llm_status",
)

_CLI_FLAGS = {"--help", "-h", "--version", "-V", "--selfcheck", "--llm-check"}

_USAGE = """biodata-mcp —— BioData Agent 本地 stdio MCP 服务器

用法：
  <python> mcp_server.py                无参数 = 以 stdio 启动服务器（由 MCP 客户端 spawn，勿手动常驻）
  <python> mcp_server.py --selfcheck    内置真协议自检：另起子进程走 initialize→tools/list→biodata_status，
                                        一条命令搞定，不改动项目文件。装完或排错时先跑它。退出码 0=通过 / 1=失败
  <python> mcp_server.py --llm-check    显式联网做一次最小 LLM API 探测；输出脱敏 JSON。退出码 0=成功
  <python> mcp_server.py --version      打印服务器版本、MCP SDK 版本、链接快照
  <python> mcp_server.py --help         打印本帮助

环境变量：
  BIODATA_LLM_ENV_FILE=<绝对路径>          项目外 LLM 配置文件；客户端配置只需保存此路径，不保存 API Key。
  BIODATA_MCP_RECALL=cross_encoder|dense   启动期预热本地重排模型（默认 off）。详见使用教程 §5.1。
  BIODATA_MCP_ACCESS_TOKEN=<在线接入令牌>   可选。不设 = 高层工具以本机身份运行，与同一实例的
                                           网页版「本机」同 scope；设了则必须是**同一实例**「接入
                                           AI 助手」里生成且未撤销的令牌，否则启动即 fail-closed
                                           （绝不降级冒用本机身份，也别把线上服务器的令牌用在这里）。
"""


def _tokens_store() -> Path:
    """在线接入令牌库路径（单一真源 mcp_tokens.default_tokens_path；根与 webapp 账户/会话库
    同一 PROJECT_ROOT = get_app_paths().data_root，两端读写同一份文件）。"""
    return mcp_tokens.default_tokens_path(get_app_paths().data_root)


class _OnlineMcpAsgi:
    """在线形态 ASGI 包装：Bearer 令牌闸 → 账户注入 scope → 内层 streamable-HTTP 应用。

    为什么自写而不用 SDK 的 token_verifier：SDK 认证链要求 AuthSettings（issuer_url /
    resource_server_url 都是必填的 URL）——本项目没有授权服务器，填假 URL 是捏造。自写
    30 行、401 形状自控、账户直接进 scope 供 `_ScopedFastMCP.call_tool` 读取，零假配置。
    类形态（而非闭包函数）是因为 Starlette `Route` 按 inspect.isfunction 判定——函数会被
    当成 request-response 端点再包一层，ASGI 三参签名会被调崩。
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self._inner(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        raw = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        rec = mcp_tokens.resolve_token(raw, store_path=_tokens_store())
        if rec is None:
            from starlette.responses import JSONResponse
            resp = JSONResponse(
                {"ok": False, "error": "invalid_token",
                 "detail": "缺少或无效的在线接入令牌；请到网页版「接入 AI 助手」生成。"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="biodata-mcp"'},
            )
            await resp(scope, receive, send)
            return
        scope["biodata.mcp_account_id"] = rec["account_id"]
        scope["biodata.mcp_username"] = rec["username"]
        await self._inner(scope, receive, send)


_online_asgi: _OnlineMcpAsgi | None = None


def reset_online_runtime() -> None:
    """重建在线运行时（session manager + 内层 ASGI 应用）。

    `StreamableHTTPSessionManager.run()` 每个实例只能进一次（SDK 实读：`_has_started`
    守卫，二次进入抛 RuntimeError）。宿主应用每次 lifespan 进入前必须调本函数——
    测试反复进出 lifespan、进程内重挂都靠它幂等。做法是把 `mcp._session_manager`
    置 None 触发 `streamable_http_app()` 的惰性重建（同一份 settings，语义不变），
    再把新内层应用换进稳定的路由包装器（Route 持有的 `_OnlineMcpAsgi` 实例不变）。
    """
    global _online_asgi
    mcp._session_manager = None                                                  
    inner = mcp.streamable_http_app()
    if _online_asgi is None:
        _online_asgi = _OnlineMcpAsgi(inner)
    else:
        _online_asgi._inner = inner


def build_online_mcp_app():
    """在线形态 ASGI 应用（/mcp 单端点）。调用方负责：①把返回值挂到路由（Path 保持 /mcp 不变，
    内层 Starlette 的路由就是 settings.streamable_http_path="/mcp"）；②在宿主应用 lifespan 里
    先 `reset_online_runtime()` 再驱动 `mcp.session_manager.run()`（挂载不走子应用 lifespan，
    stateless 模式也需要任务组）。返回的是模块级稳定实例，重复调用返回同一个。"""
    reset_online_runtime()
    assert _online_asgi is not None
    return _online_asgi


def _mcp_sdk_version() -> str:
    try:
        import importlib.metadata as _md
        return _md.version("mcp")
    except Exception:
        return "unknown"


def _print_version() -> int:
    print(
        f"biodata-mcp {_SERVER_VERSION} | MCP SDK {_mcp_sdk_version()} | "
        f"Python {sys.version.split()[0]} | links_snapshot {_LEDGER_SNAPSHOT}"
    )
    return 0


def _print_llm_check() -> int:
    import json as _json

    payload = _llm_status_payload(True)
    print(_json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
                                                 
    return 0 if payload.get("connection") in {"success", "mock"} else 1


def _selfcheck() -> int:
    """客户端侧「真协议」自检：spawn 本服务器的另一个进程，完成 initialize → tools/list →
    调用 biodata_status，验证全部工具（`_EXPECTED_TOOLS`）可见且库存就绪。一条命令、零拷贝、**不写项目**
    （子进程带 -B + PYTHONDONTWRITEBYTECODE=1，不落 __pycache__）。

    设计为教程里那段手抄 Python 探针的「零依赖内置替身」：装完或排错时
      <venv-python> mcp_server.py --selfcheck
    即可，无需把多行脚本粘进 PowerShell（规避引号/编码坑）。退出码 0=通过 / 1=失败；
    人读结论走 stdout，逐项 PASS/FAIL 与子进程 stderr 一并可见。
    """
    import json as _json

    def _line(ok: bool, msg: str) -> None:
        print(("  [PASS] " if ok else "  [FAIL] ") + msg, flush=True)

    try:
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
                                                                       
                                                                            
        print(
            f"SELFCHECK_FAIL 无法导入 mcp 客户端 API（{type(exc).__name__}: {exc}）。\n"
            f"  → mcp 核心已装（否则本文件导入 FastMCP 时就已报错），此处失败多为 mcp 版本不兼容 / 安装不完整。\n"
            f"    请核对该 venv 的 mcp 版本（教程固定 mcp==1.28.1）：<那个 python> -m pip show mcp，必要时重装。",
            flush=True,
        )
        return 1

    async def _run() -> int:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"                    
        env["PYTHONUTF8"] = "1"
                                               
        env.setdefault(_CALL_LOG_ENV, "off")
        if getattr(sys, "frozen", False):
                                                                         
                                                                               
            command, args = sys.executable, []
        else:
            command, args = sys.executable, ["-B", str(Path(__file__).resolve())]
        params = StdioServerParameters(command=command, args=args, env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                names = {t.name for t in (await session.list_tools()).tools}
                missing = [t for t in _EXPECTED_TOOLS if t not in names]
                _line(not missing, f"tools/list 可见 {len(names)} 个工具"
                                   + (f"，缺 {missing}" if missing else ""))
                if missing:
                    return 1

                r = await session.call_tool("biodata_status", {})
                if r.isError or not r.content:
                    _line(False, "biodata_status 返回 MCP 错误或空内容")
                    return 1
                payload = _json.loads(r.content[0].text)
                if not payload.get("ok"):
                    _line(False, f"biodata_status ok=false：{payload.get('error')}")
                    return 1
                ct = payload.get("corpus_total", 0)
                di = bool(payload.get("download_index_ready"))
                _line(ct > 0, f"corpus_total={ct}")
                _line(di, f"download_index_ready={di}")
                if ct <= 0 or not di:
                    return 1

                                                            
                r_llm = await session.call_tool("biodata_llm_status", {"check_connection": False})
                if r_llm.isError or not r_llm.content:
                    _line(False, "biodata_llm_status 返回 MCP 错误或空内容")
                    return 1
                llm_payload = _json.loads(r_llm.content[0].text)
                llm_configured = bool(llm_payload.get("configured"))
                _line(True, f"LLM 配置可读取（configured={str(llm_configured).lower()}；未联网）")

                                                          
                r2 = await session.call_tool("parse_constraints", {"query": "人类肺癌单细胞，要有 FASTQ"})
                ok2 = (not r2.isError) and bool(r2.content) and _json.loads(r2.content[0].text).get("ok")
                _line(bool(ok2), "parse_constraints 核心路径可调用")
                if not ok2:
                    return 1

                                                                        
                r3 = await session.call_tool("curate_datasets", {"action": "list"})
                ok3 = (not r3.isError) and bool(r3.content) and _json.loads(r3.content[0].text).get("ok")
                _line(bool(ok3), "curate_datasets(action=list) 管护入口可调用")
                if not ok3:
                    return 1

                print(
                    f"SELFCHECK_OK tools={len(names)} corpus_total={ct} download_index_ready=true "
                    f"llm_configured={str(llm_configured).lower()}",
                    flush=True,
                )
                return 0

    print("[biodata-mcp] 自检中：spawn 子进程走真 MCP 协议 …", flush=True)

    async def _guarded() -> int:
                                                             
        try:
            return await asyncio.wait_for(_run(), timeout=90)
        except asyncio.TimeoutError:
            print(
                "SELFCHECK_FAIL 90s 内未完成协议握手（服务器可能卡在启动/预热）。"
                "若配了 BIODATA_MCP_RECALL，预热较慢属正常，可重试或临时清空该 env。",
                flush=True,
            )
            return 1

    try:
        return asyncio.run(_guarded())
    except Exception as exc:
        print(f"SELFCHECK_FAIL 协议自检异常（{type(exc).__name__}: {exc}）", flush=True)
        return 1


if __name__ == "__main__":
                                                                
                                                                
                                                          
                                                    
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")                
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")                
    _argv = sys.argv[1:]
                                                                   
                                                                        
                          
    if _argv:
        for _s in (sys.stdout, sys.stderr):
            try:
                _s.reconfigure(encoding="utf-8")                            
            except Exception:
                pass
    if "--help" in _argv or "-h" in _argv:
        print(_USAGE)
        raise SystemExit(0)
    if "--version" in _argv or "-V" in _argv:
        raise SystemExit(_print_version())
    if "--selfcheck" in _argv:
        raise SystemExit(_selfcheck())
    if "--llm-check" in _argv:
        raise SystemExit(_print_llm_check())
                                                               
                                                                    
                                                    
    if _argv:
        print(f"未识别的参数：{_argv}。本服务器无参数即启动；可用子命令：{sorted(_CLI_FLAGS)}\n", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)

                                                                 
                                                          
                                                                            
    if _DEFAULT_RECALL != "off":
        try:
            from dataset_recommender.retrieval.vector_recall import warm_recall_backend
            _ready = warm_recall_backend(_DEFAULT_RECALL)
            print(f"[biodata-mcp] 预热 recall='{_DEFAULT_RECALL}' → ready={_ready}", file=sys.stderr, flush=True)
        except Exception as _e:                                  
            print(f"[biodata-mcp] 预热 recall 失败（{_e}）→ 回退 off", file=sys.stderr, flush=True)
                                                                  
                                    
    mcp.run()
