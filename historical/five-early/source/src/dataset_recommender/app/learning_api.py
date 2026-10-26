"""同源学习 API；可信身份注入，批准边界只由网页逐条确认入口开放。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from .conversation_api import invoke
from .conversation_runs import RunError
from .learning_service import LearningService, service_for

LearningId = Annotated[str, Field(min_length=1, max_length=128)]


class LearningSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    project_id: str = Field(max_length=80)
    conversation_ids: list[LearningId] = Field(max_length=64)


class LearningMaterial(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    client_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(default='', max_length=80)
    conversation_id: str = Field(default='', max_length=80)
    text: str = Field(min_length=1, max_length=4000)


class LearningRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    selection: list[LearningSelection] | None = Field(default=None, max_length=64)
    source_ids: list[LearningId] | None = Field(default=None, max_length=200)
    local_materials: list[LearningMaterial] | None = Field(default=None, max_length=12)
    manual: bool = False
    allow_cross_project: bool = False
    project_id: str = Field(default='', max_length=80)
    provider: str | None = Field(default=None, max_length=64)
    use_llm: bool | None = None
    mock_llm: bool | None = None
    api_key: str | None = Field(default=None, max_length=4096, repr=False, exclude=True)
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    action: str = Field(default='set', max_length=32)
    expected_identity: str = Field(default='', max_length=128)
    expected_config_epoch: int | None = Field(default=None, ge=0)
    enabled: bool | None = None
    threshold: int = Field(default=3, ge=1, le=20)
    request_id: str = Field(default='', max_length=128)
    feedback: str = Field(default='', max_length=2000)
    conversation_id: str = Field(default='', max_length=80)
    candidate_id: str = Field(default='', max_length=80)
    decision: str = Field(default='', max_length=32)
    expected_fingerprint: str = Field(default='', max_length=64)
    expected_rules_fingerprint: str = Field(default='', max_length=64)
    confirm: bool = False

    def check_trigger_flags(self):
        if self.allow_cross_project and (not self.manual or self.project_id or self.feedback
            or not (self.selection or self.source_ids or self.local_materials)):
            raise RunError('invalid_learning_trigger', '跨项目用户层汇总仅用于明确选来源的一次性手动授权，不用于自动、反馈或项目层。')


def build_learning_router(*, service, principal, endpoint, require_origin, configure, quota):
    router = APIRouter()

    def owned(request):
        shared = service()
        return service_for(shared.store, shared.workspace), principal(request)

    def binding(request, payload, *, for_trigger=False):
        owner = principal(request)
        if not payload.expected_identity:
            raise RunError('learning_identity_required', '学习授权操作需要当前账户的条件标记。')
        LearningService.check_identity(owner, payload.expected_identity)
        if for_trigger:
            payload.check_trigger_flags()
        elif payload.allow_cross_project:
            raise RunError('invalid_learning_control', '跨项目汇总标记只用于手动trigger。')
        if payload.expected_config_epoch is None:
            raise RunError('learning_config_required', '请先读取学习状态并携带配置授权版本。')
        shared = service()
        app = service_for(shared.store, shared.workspace)
        captured = app.store.authorization_snapshot(owner)
        fp = endpoint(request, payload.base_url or '', payload.model or '', payload.provider or '')
        if captured['config_epochs'].get(fp, 0) != payload.expected_config_epoch:
            raise RunError('learning_config_conflict', '配置授权已变化，请刷新后重试。')
        scope, epoch = app.store.authorization(owner, fp, payload.project_id,
            expected_config_epoch=payload.expected_config_epoch,
            expected_scope_epoch=captured['scope_epochs'].get((fp, payload.project_id), 0))
        return app, owner, fp, scope['epoch'], epoch, {project: value for (endpoint_fp, project), value
            in captured['scope_epochs'].items() if endpoint_fp == fp}

    def configured(request, payload, app, owner, fp):
        from ..agent.agent_exec import _endpoint_fp_from_config
        cfg = configure(request, payload)
        app.check_identity(principal(request), owner)
        if _endpoint_fp_from_config(cfg) != fp:
            raise RunError('learning_config_conflict', '有效AI端点已变化，请刷新学习状态后重新操作。')
        return cfg

    @router.get('/api/learning/status')
    def status(request: Request, project_id: str = '', base_url: str = '', model: str = '', provider: str = '',
               job_id: str = '', after: int = Query(default=0, ge=0)):
        def operation():
            app, owner = owned(request)
            return {'ok': True, **app.status(owner, endpoint(request, base_url, model, provider), project_id, job_id=job_id, after=after)}
        return invoke(operation)

    @router.get('/api/learning/sources')
    def sources(request: Request, project_id: str | None = Query(default=None, max_length=80),
                base_url: str = Query(default='', max_length=2048), model: str = Query(default='', max_length=256),
                provider: str = Query(default='', max_length=64), offset: int = Query(default=0, ge=0),
                limit: int = Query(default=100, ge=1, le=200)):
        def operation():
            app, owner = owned(request)
            return app.sources(owner, endpoint(request, base_url, model, provider), project_id, offset=offset, limit=limit)
        return invoke(operation)

    @router.get('/api/learning/candidates')
    def candidates(request: Request, project_id: str = '', base_url: str = '', model: str = '', provider: str = '',
                   record_type: str = 'candidate', status: str = '', offset: int = Query(default=0, ge=0),
                   limit: int = Query(default=100, ge=1, le=200)):
        def operation():
            app, owner = owned(request)
            fp = endpoint(request, base_url, model, provider)
            from ..agent.collab_rules import list_collab_rules, content_fingerprint
            return {'ok': True, **app.candidates(owner, fp, project_id, record_type=record_type, status=status, offset=offset, limit=limit),
                    'active_rules': [dict(r, learning_rule=True, fingerprint=content_fingerprint(r)) for r in list_collab_rules(app.root, principal=owner, endpoint_fp=fp, project_id=project_id) if r.get('status', 'active') == 'active']}
        return invoke(operation)

    @router.post('/api/learning/control')
    def control(payload: LearningRequest, request: Request):
        require_origin(request)
        def operation():
            app, owner, fp, scope_epoch, config_epoch, source_epochs = binding(request, payload)
            if payload.action == 'release':
                if payload.api_key is not None:
                    raise RunError('invalid_learning_release', '释放授权不需要密钥，请只发送端点与账户条件标记。')
                app.check_identity(principal(request), owner)
                return app.release(owner, fp, expected_identity=payload.expected_identity,
                    expected_config_epoch=config_epoch)
            if payload.action != 'set':
                raise RunError('invalid_learning_control', '不支持这个授权生命周期操作。')
            if payload.enabled is None:
                raise RunError('invalid_learning_control', '请明确开启或暂停学习。')
            cfg = configured(request, payload, app, owner, fp)
            return app.control(owner, cfg, project_id=payload.project_id,
                enabled=payload.enabled, threshold=payload.threshold, quota=lambda effective: quota(request, payload, effective),
                expected_identity=payload.expected_identity, expected_config_epoch=config_epoch, expected_scope_epoch=scope_epoch)
        return invoke(operation)

    @router.post('/api/learning/trigger')
    def trigger(payload: LearningRequest, request: Request):
        require_origin(request)
        def operation():
            app, owner, fp, scope_epoch, config_epoch, source_epochs = binding(request, payload, for_trigger=True)
            cfg = configured(request, payload, app, owner, fp)
            return app.trigger(owner, cfg, project_id=payload.project_id,
                request_id=payload.request_id, feedback=payload.feedback, conversation_id=payload.conversation_id,
                quota=lambda effective: quota(request, payload, effective), expected_identity=payload.expected_identity,
                expected_config_epoch=config_epoch, expected_scope_epoch=scope_epoch,
                selection=[item.model_dump() for item in payload.selection] if payload.selection is not None else None,
                source_ids=payload.source_ids,
                local_materials=[item.model_dump() for item in payload.local_materials] if payload.local_materials is not None else None,
                manual=payload.manual, allow_cross_project=payload.allow_cross_project,
                expected_source_scope_epochs=source_epochs)
        return invoke(operation)

    @router.post('/api/learning/review')
    def review(payload: LearningRequest, request: Request):
        require_origin(request)
        def operation():
            app, owner = owned(request)
            if payload.decision != 'reject' and not payload.confirm:
                raise RunError('human_confirmation_required', '请逐条核对并确认这条规则。')
            return app.review(owner, endpoint(request, payload.base_url or '', payload.model or '', payload.provider or ''),
                payload.project_id, candidate_id=payload.candidate_id, decision=payload.decision,
                expected_fingerprint=payload.expected_fingerprint, expected_rules_fingerprint=payload.expected_rules_fingerprint,
                feedback=payload.feedback, human_confirmed=payload.confirm)
        return invoke(operation)

    return router
