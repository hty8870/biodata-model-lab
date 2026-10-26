"""封装好的知识库 Agent MCP 面；身份由传输层注入，不接受 owner 或执行事实。"""
from __future__ import annotations

import threading
import uuid
import os

_STDIO_TOKEN_AT_STARTUP = os.environ.get('BIODATA_MCP_ACCESS_TOKEN', '')
from contextlib import contextmanager
from contextvars import ContextVar

AGENT_TOOLS = ('agent_chat', 'agent_conversations', 'agent_projects', 'agent_task', 'agent_replay', 'agent_learning')
_IDENTITY = ContextVar('mcp_agent_identity', default=('local', ''))
_SERVICES = {}
_LOCK = threading.Lock()


@contextmanager
def bind_identity(owner, username=''):
    token = _IDENTITY.set((owner, username))
    try:
        yield
    finally:
        _IDENTITY.reset(token)


def stdio_identity(project_root):
    if not _STDIO_TOKEN_AT_STARTUP:
        return 'local', ''
    from . import mcp_tokens
    from .conversation_runs import RunError
    record = mcp_tokens.resolve_token(_STDIO_TOKEN_AT_STARTUP,
        store_path=mcp_tokens.default_tokens_path(project_root))
    if record is None:
        raise RunError('invalid_token', '显式 stdio 接入凭据无效或已撤销；不会降级冒用本机身份。')
    return record['account_id'], record['username']


def service_for(project_root):
    from .conversation_service import ConversationService
    from .conversation_store import ConversationStore
    path = ConversationStore(project_root).path
    key = (str(project_root), str(path))
    with _LOCK:
        if key not in _SERVICES:
            _SERVICES[key] = ConversationService(project_root)
        return _SERVICES[key]


def register_agent_tools(mcp, *, root):
    from mcp.server.fastmcp.exceptions import ToolError
    from .conversation_runs import RunError
    from .durable_jobs import JobStoreError
    from .llm_admission import gate_quota, materialize
    from .request_validation import ParamValidationError
    from .runtime_paths import get_app_paths
    import time
    from functools import wraps
    import anyio

    slow_admission = anyio.CapacityLimiter(6)
    control_admission = anyio.CapacityLimiter(2)
    workers = anyio.CapacityLimiter(8)

    def transport(operation):
        admission = control_admission if operation.__name__ == 'agent_task' else slow_admission
        @wraps(operation)
        async def dispatch(*args, **kwargs):
            borrower = object()
            try:
                admission.acquire_on_behalf_of_nowait(borrower)
            except anyio.WouldBlock:
                raise ToolError('mcp_transport_busy: 同时处理的连接已达上限，请稍后重试或核对任务状态。') from None
            lock = threading.Lock()
            started = withdrawn = False
            def execute():
                nonlocal started
                with lock:
                    if withdrawn:
                        return None
                    started = True
                try:
                    return operation(*args, **kwargs)
                finally:
                    anyio.from_thread.run_sync(admission.release_on_behalf_of, borrower)
            try:
                return await anyio.to_thread.run_sync(execute, limiter=workers, abandon_on_cancel=True)
            finally:
                with lock:
                    if not started:
                        withdrawn = True
                        admission.release_on_behalf_of(borrower)
        return dispatch

    def invoke(operation):
        try:
            return operation()
        except (JobStoreError, ParamValidationError, ValueError) as exc:
            raise ToolError(f"{getattr(exc, 'code', 'bad_param')}: {exc}") from None

    def context():
        owner, username = _IDENTITY.get()
        app = service_for(root())
        def quota(cfg):
            gate_quota(root(), username=username, cfg=cfg)
        return app, owner, quota

    @mcp.tool()
    @transport
    def agent_chat(message: str, conversation_id: str | None = None, request_id: str | None = None,
                   expected_revision: int | None = None, project_id: str | None = None,
                   state_patch: dict | None = None, ask_choice: str | None = None, wait_seconds: float = 2) -> dict:
        """直接对 BioData 知识库 Agent 说一句话，不需要宿主拼接底层工具。支持自然话多轮，自己保存消息、结果帧和检索现场。重试须复用 request_id；后台返回 run_id 后用 agent_task 续读，不要另起请求重跑。state_patch 只接受 query/sources/search_params，ask_choice 仅取当前选择卡真实 key。默认运行可用 Agent 配置并使用 Web 同一身份额度；缺模型如实降级。仅同实例、同身份共享。"""
        def operation():
            app, owner, quota = context()
            req_id = request_id or uuid.uuid4().hex
            wait = max(0, min(float(wait_seconds), 30))
            from contextlib import nullcontext
            from .conversation_runs import TERMINAL
            waiting = app.runtime.waiting_transport(owner) if wait else nullcontext()
            with waiting:
                run = app.chat(owner, message, conv_id=conversation_id, request_id=req_id,
                    expected_revision=expected_revision, project_id=project_id, state_patch=state_patch,
                    ask_choice=ask_choice, quota=quota)
                deadline = time.monotonic() + wait
                while run['status'] not in TERMINAL and time.monotonic() < deadline:
                    time.sleep(.03)
                    run = app.store.load(owner, run['id'])
                return {'ok': True, 'request_id': req_id, 'run': run,
                        'conversation_id': run['conversation_id'], 'scope': 'local' if owner == 'local' else 'account',
                        'web_path': '/?conv=' + run['conversation_id'],
                        'result': run['result'], 'pending': run['status'] not in TERMINAL}
        return invoke(operation)

    @mcp.tool()
    @transport
    def agent_conversations(action: str = 'list', conversation_id: str = '', expected_revision: int | None = None,
                            offset: int = 0, limit: int = 100, after_message: int = 0, after_frame: int = 0,
                            include_archived: bool = False, frame_id: str = '', title: str = '',
                            import_id: str = '', data: dict | None = None, project_id: str | None = None) -> dict:
        """会话 list/get/create/archive/restore/rename/delete/select_frame/branch/import。list/get 含可信项目归属与分页游标；修改携 expected_revision 防止旧客户端覆盖新轮。create/import/branch 可给 project_id。branch 必须给真实 frame_id。import 仅提交用户预览确认的一条旧史，提供稳定 import_id；绝不自动扫描本机历史。删除后不可复盘。"""
        def operation():
            app, owner, _ = context()
            if action == 'list':
                return {'ok': True, **app.list_conversations(owner, offset=offset, limit=limit, include_archived=include_archived)}
            if action == 'get':
                return {'ok': True, 'conversation': app.get_conversation(owner, conversation_id,
                    after_message=after_message, after_frame=after_frame, limit=limit)}
            return app.conversation_action(owner, conversation_id, action, expected_revision=expected_revision,
                frame_id=frame_id, title=title, import_id=import_id, data=data, project_id=project_id)
        return invoke(operation)

    @mcp.tool()
    @transport
    def agent_projects(action: str = 'list', project_id: str = '', name: str = '', color: str | None = None,
                       conversation_id: str = '', expected_revision: int | None = None, project_ids: list[str] | None = None) -> dict:
        """共享项目 list/create/update/archive/restore/reorder/assign。归档项目保留会话；assign 需真实会话及 expected_revision。与 Web 项目机制、学习作用域同源，不上传记忆。"""
        def operation():
            app, owner, _ = context()
            store = app.workspace
            if action == 'list':
                return {'ok': True, 'projects': store.list_projects(owner, include_archived=True), 'assignments': store.assignments(owner)}
            if action == 'create':
                from .workspace_store import DEFAULT_PROJECT_COLOR
                result = store.create_project(owner, name, color=DEFAULT_PROJECT_COLOR if color is None else color)
            elif action == 'update':
                result = store.update_project(owner, project_id, name=name or None, color=color)
            elif action in {'archive', 'restore'}:
                result = store.archive_project(owner, project_id, archived=action == 'archive')
            elif action == 'reorder':
                return {'ok': True, 'projects': store.reorder_projects(owner, project_ids or [])}
            elif action == 'assign':
                return app.conversation_action(owner, conversation_id, 'project', expected_revision=expected_revision, project_id=project_id)
            else:
                raise RunError('invalid_project_action', '不支持这个项目动作。')
            return {'ok': True, 'project': result}
        return invoke(operation)

    @mcp.tool()
    @transport
    def agent_task(action: str = 'status', run_id: str = '', request_id: str = '', after: int = 0) -> dict:
        """已有任务 status/events/cancel/resolve；events 返回 seq 游标。取消只在安全边界停止，已完成操作不会撤销。interrupted 不自动重演。"""
        def operation():
            app, owner, _ = context()
            if action == 'status':
                return {'ok': True, 'run': app.store.load(owner, run_id)}
            if action == 'resolve':
                return {'ok': True, 'run': app.store.find_request(owner, request_id)}
            if action == 'events':
                events = app.store.events(owner, run_id, after)
                return {'ok': True, 'events': events, 'next_after': events[-1]['seq'] if events else after}
            if action == 'cancel':
                return {'ok': True, 'run': app.runtime.cancel(owner, run_id)}
            raise RunError('invalid_task_action', '不支持这个任务动作。')
        return invoke(operation)

    @mcp.tool()
    @transport
    def agent_replay(conversation_id: str, action: str = 'points', data: dict | None = None) -> dict:
        """复盘 status/points/annotate/snapshot/annotate_snapshot/seen/chat/digest。points 返回结构化图、出处、标注；snapshot 打包标注供继续讨论；chat/digest 会使用可用模型与同一额度。data 仅 pid/text/snap_id/observed_version/observed_pids。不需要客户端互传 key；仅授权本人共享会话。"""
        def operation():
            app, owner, quota = context()
            return app.replay(owner, conversation_id, action, data,
                configure=lambda: materialize(config_root=get_app_paths().config_root), quota=quota)
        return invoke(operation)

    @mcp.tool()
    @transport
    def agent_learning(action: str = 'status', project_id: str = '', enabled: bool | None = None,
                       threshold: int = 3, request_id: str = '', feedback: str = '', conversation_id: str = '',
                       candidate_id: str = '', decision: str = '', expected_fingerprint: str = '',
                       expected_rules_fingerprint: str = '', offset: int = 0, limit: int = 100,
                       record_type: str = 'candidate', status: str = '', job_id: str = '', after: int = 0,
                       base_url: str | None = None, model: str | None = None,
                       selection: list | None = None, source_ids: list | None = None,
                       local_materials: list | None = None, manual: bool = False,
                       operations: list | None = None, source_refs: list | None = None,
                       category_definitions: list | None = None,
                       change_id: str = '', expected_change_revision: int = 0,
                       scope_level: str = 'user', kind: str = '') -> dict:
        """独立学习 status/control/release/trigger/candidates/review + 来源读面 sources + 统一知识 entries/catalog/changes/propose/assess。catalog读取分类目录，status.knowledge列分类定义；propose可携category_definitions与条目的disclosure分类引用，仍须网页确认。release可给旧base_url/model，只释放当前trusted身份该端点全部项目的短期配置，保留长期consent/规则/候选；不接受key/owner/confirmed。默认需要当前scope授权；非试用AI、独立调用预算。只读同身份、同端点、同项目的候选/审阅记录与真实执行trace。control开关。trigger支持固定选择（selection=[{project_id,conversation_ids}]、source_ids、local_materials=[{client_id,project_id,conversation_id?,text}]）与manual一次性授权；跨项目用户级汇总（allow_cross_project）只在Web同源真人入口确认，MCP不提供。review可驳回并保存反馈；approve/replace仅返回网页真人确认路径，MCP不能替用户激活规则。知识propose只建草稿、assess只保存服务端评估；commit永不在MCP开放（human_confirmed仅同源Web真人审核入口）。项目不自动提升全局。重启无密钥不自动执行，需同owner端点恢复授权。"""
        def operation():
            app, owner, quota = context()
            from .learning_service import service_for as learning_for
            from ..agent.agent_exec import _endpoint_fp_from_config
            learning = learning_for(app.store, app.workspace)
            snapshot = learning.store.authorization_snapshot(owner) if action in {'control', 'trigger', 'release', 'assess'} else None
            cfg = materialize(config_root=get_app_paths().config_root)
            fp = _endpoint_fp_from_config(cfg)
            if action == 'release':
                from dataclasses import replace
                if (base_url is not None and len(base_url) > 2048) or (model is not None and len(model) > 256):
                    raise RunError('bad_param', '端点地址或模型名过长。')
                selected = replace(cfg, base_url=cfg.base_url if base_url is None else base_url,
                    model=cfg.model if model is None else model)
                selected_fp = _endpoint_fp_from_config(selected)
                return learning.release(owner, selected_fp, expected_identity=owner,
                    expected_config_epoch=snapshot['config_epochs'].get(selected_fp, 0))
            if base_url is not None or model is not None:
                raise RunError('bad_param', '端点选择只用于释放旧配置授权。')
            if action == 'status':
                return {'ok': True, **learning.status(owner, fp, project_id, job_id=job_id, after=after),
                        'knowledge': learning.knowledge.status(owner, fp, project_id)}
            if action == 'catalog':
                return learning.knowledge.directory(owner, fp, project_id, offset=offset, limit=limit)
            if action == 'sources':
                return learning.sources(owner, fp, project_id or None, offset=offset, limit=limit)
            if action == 'candidates':
                return {'ok': True, **learning.candidates(owner, fp, project_id, record_type=record_type, status=status, offset=offset, limit=limit)}
            if action in {'entries', 'changes'}:
                knowledge = learning.knowledge
                if action == 'entries':
                    return knowledge.entries(owner, fp, project_id, scope_level=scope_level, kind=kind,
                        status=status or 'active', offset=offset, limit=limit)
                return knowledge.changes(owner, fp, project_id, status=status, offset=offset, limit=limit)
            if action == 'propose':
                if not request_id or operations is None:
                    raise RunError('bad_param', '知识提案需要稳定 request_id 与明确 operations。')
                return learning.knowledge.propose(owner, fp, project_id, request_id=request_id,
                    operations=operations, source_refs=source_refs or (), expected_identity=owner,
                    category_definitions=category_definitions or ())
            if action == 'assess':
                if not change_id:
                    raise RunError('bad_param', '评估需要明确 change_id 与 expected_change_revision。')
                epoch = snapshot['config_epochs'].get(fp, 0)
                scope_epoch = snapshot['scope_epochs'].get((fp, project_id), 0)
                def check(*, conn=None):
                    learning.store.authorization(owner, fp, project_id,
                        expected_config_epoch=epoch, expected_scope_epoch=scope_epoch, conn=conn)
                return learning.knowledge.assess(owner, fp, project_id, change_id=change_id,
                    expected_change_revision=expected_change_revision, cfg=cfg, check=check,
                    before_llm=lambda actual: quota(actual))
            if action in {'control', 'trigger'}:
                epoch = snapshot['config_epochs'].get(fp, 0)
                scope_epoch = snapshot['scope_epochs'].get((fp, project_id), 0)
                if action == 'control':
                    if enabled is None:
                        raise RunError('invalid_learning_control', '请明确开启或暂停学习。')
                    return learning.control(owner, cfg, project_id=project_id, enabled=enabled, threshold=threshold,
                        quota=quota, expected_identity=owner, expected_config_epoch=epoch, expected_scope_epoch=scope_epoch)
                if type(manual) is not bool:
                    raise RunError('invalid_learning_trigger', '一次性授权标记须为严格布尔值。')
                captured_epochs = {project: value for (endpoint_fp, project), value
                    in snapshot['scope_epochs'].items() if endpoint_fp == fp}
                return learning.trigger(owner, cfg, project_id=project_id, request_id=request_id, feedback=feedback,
                    conversation_id=conversation_id, quota=quota, expected_identity=owner,
                    expected_config_epoch=epoch, expected_scope_epoch=scope_epoch,
                    selection=selection, source_ids=source_ids, local_materials=local_materials,
                    manual=manual, allow_cross_project=False,
                    expected_source_scope_epochs=captured_epochs)
            if action == 'review':
                if not decision:
                    return {'ok': True, **learning.candidates(owner, fp, project_id, record_type='audit', offset=offset, limit=limit)}
                return learning.review(owner, fp, project_id, candidate_id=candidate_id, decision=decision,
                    expected_fingerprint=expected_fingerprint, expected_rules_fingerprint=expected_rules_fingerprint,
                    feedback=feedback, human_confirmed=False)
            raise RunError('invalid_learning_action', '不支持这个学习动作。')
        return invoke(operation)
