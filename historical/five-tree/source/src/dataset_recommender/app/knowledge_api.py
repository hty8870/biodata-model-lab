"""统一知识 HTTP 适配：可信请求身份、同源写入、配置 CAS 与真人提交边界。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ..agent.agent_exec import _endpoint_fp_from_config
from .conversation_api import invoke
from .conversation_runs import RunError
from .learning_service import service_for


class KnowledgeEndpoint(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    project_id: str = Field(default='', max_length=80)
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    provider: str | None = Field(default=None, max_length=64)


class KnowledgePropose(KnowledgeEndpoint):
    expected_identity: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    operations: list[dict] = Field(min_length=1, max_length=32)
    source_refs: list[dict] = Field(default_factory=list, max_length=8)


class KnowledgeAssess(KnowledgeEndpoint):
    use_llm: bool | None = None
    mock_llm: bool | None = None
    api_key: str | None = Field(default=None, max_length=4096, repr=False, exclude=True)
    expected_identity: str = Field(min_length=1, max_length=128)
    expected_config_epoch: int = Field(ge=0)
    change_id: str = Field(min_length=1, max_length=128)
    expected_change_revision: int = Field(ge=0)


class KnowledgeCommit(KnowledgeEndpoint):
    expected_identity: str = Field(min_length=1, max_length=128)
    change_id: str = Field(min_length=1, max_length=128)
    expected_change_revision: int = Field(ge=0)
    assessment_id: str = Field(min_length=1, max_length=128)
    expected_catalog_generation: int = Field(ge=0)
    confirm: bool = False


class KnowledgeReject(KnowledgeEndpoint):
    change_id: str = Field(min_length=1, max_length=128)
    expected_change_revision: int = Field(ge=0)
    feedback: str = Field(default='', max_length=2000)


EndpointBase = Annotated[str, Query(max_length=2048)]
EndpointModel = Annotated[str, Query(max_length=256)]
EndpointProvider = Annotated[str, Query(max_length=64)]
Project = Annotated[str, Query(max_length=80)]


def build_knowledge_router(*, service, principal, endpoint, require_origin, configure, quota):
    router = APIRouter()

    def owned(request, base='', model='', provider=''):
        shared = service()
        owner = principal(request)
        learning = service_for(shared.store, shared.workspace)
        return learning.knowledge, learning, owner, endpoint(request, base or '', model or '', provider or '')

    @router.get('/api/knowledge/status')
    def status(request: Request, project_id: Project = '', base_url: EndpointBase = '',
               model: EndpointModel = '', provider: EndpointProvider = ''):
        def operation():
            app, learning, owner, fp = owned(request, base_url, model, provider)
            result = app.status(owner, fp, project_id)
            result['config_epoch'] = learning.store.config_epoch(owner, fp)
            return result
        return invoke(operation)

    @router.get('/api/knowledge/entries')
    def entries(request: Request, project_id: Project = '', base_url: EndpointBase = '',
                model: EndpointModel = '', provider: EndpointProvider = '',
                scope_level: str = Query(default='user', max_length=16), kind: str = Query(default='', max_length=16),
                status: str = Query(default='active', max_length=16), include_inherited: bool = False,
                offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)):
        def operation():
            app, _, owner, fp = owned(request, base_url, model, provider)
            return app.entries(owner, fp, project_id, scope_level=scope_level, kind=kind, status=status,
                include_inherited=include_inherited, offset=offset, limit=limit)
        return invoke(operation)

    @router.get('/api/knowledge/entries/{entry_id}')
    def entry(entry_id: str, request: Request, base_url: EndpointBase = '',
              model: EndpointModel = '', provider: EndpointProvider = ''):
        def operation():
            app, _, owner, fp = owned(request, base_url, model, provider)
            return {'ok': True, 'entry': app.entry(owner, fp, entry_id)}
        return invoke(operation)

    @router.get('/api/knowledge/changes')
    def changes(request: Request, project_id: Project = '', base_url: EndpointBase = '',
                model: EndpointModel = '', provider: EndpointProvider = '', status: str = Query(default='', max_length=16),
                offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)):
        def operation():
            app, _, owner, fp = owned(request, base_url, model, provider)
            return app.changes(owner, fp, project_id, status=status, offset=offset, limit=limit)
        return invoke(operation)

    @router.post('/api/knowledge/propose')
    def propose(payload: KnowledgePropose, request: Request):
        require_origin(request)
        def operation():
            app, _, owner, fp = owned(request, payload.base_url, payload.model, payload.provider)
            return app.propose(owner, fp, payload.project_id, request_id=payload.request_id,
                operations=payload.operations, source_refs=payload.source_refs, expected_identity=payload.expected_identity)
        return invoke(operation)

    @router.post('/api/knowledge/assess')
    def assess(payload: KnowledgeAssess, request: Request):
        require_origin(request)
        def operation():
            app, learning, owner, fp = owned(request, payload.base_url, payload.model, payload.provider)
            learning.check_identity(owner, payload.expected_identity)
            scope, epoch = learning.store.authorization(owner, fp, payload.project_id,
                expected_config_epoch=payload.expected_config_epoch)
            cfg = configure(request, payload)
            if _endpoint_fp_from_config(cfg) != fp:
                raise RunError('learning_config_conflict', '有效 AI 端点已变化，请刷新状态。')
            def check(*, conn=None):
                learning.check_identity(principal(request), owner)
                learning.store.authorization(owner, fp, payload.project_id,
                    expected_config_epoch=epoch, expected_scope_epoch=scope['epoch'], conn=conn)
            return app.assess(owner, fp, payload.project_id, change_id=payload.change_id,
                expected_change_revision=payload.expected_change_revision, cfg=cfg, check=check,
                before_llm=lambda effective: quota(request, payload, effective))
        return invoke(operation)

    @router.post('/api/knowledge/commit')
    def commit(payload: KnowledgeCommit, request: Request):
        require_origin(request)
        def operation():
            app, _, owner, fp = owned(request, payload.base_url, payload.model, payload.provider)
            return app.commit(owner, fp, payload.project_id, change_id=payload.change_id,
                expected_change_revision=payload.expected_change_revision, assessment_id=payload.assessment_id,
                expected_catalog_generation=payload.expected_catalog_generation,
                human_confirmed=payload.confirm is True, expected_identity=payload.expected_identity)
        return invoke(operation)

    @router.post('/api/knowledge/reject')
    def reject(payload: KnowledgeReject, request: Request):
        require_origin(request)
        def operation():
            app, _, owner, fp = owned(request, payload.base_url, payload.model, payload.provider)
            return app.reject(owner, fp, payload.project_id, change_id=payload.change_id,
                expected_change_revision=payload.expected_change_revision, feedback=payload.feedback)
        return invoke(operation)

    return router
