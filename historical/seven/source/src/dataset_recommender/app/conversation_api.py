"""共享会话 HTTP 薄适配；旧浏览器历史只在显式 import 动作中上传。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .durable_jobs import JobStoreError
from .request_validation import ParamValidationError


class ConversationAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: str
    expected_revision: int | None = Field(default=None, ge=0)
    frame_id: str = ''
    title: str = Field(default='', max_length=180)
    project_id: str = ''
    import_id: str = ''
    data: dict | None = None


class OwnedReplayData(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    pid: str = Field(default='', max_length=128)
    text: str = Field(default='', max_length=8000)
    snap_id: str = Field(default='', max_length=128)
    observed_version: int | None = Field(default=None, ge=0)
    observed_pids: list[str] | None = Field(default=None, max_length=4096)


class OwnedReplayRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    data: OwnedReplayData = Field(default_factory=OwnedReplayData)
    provider: str | None = Field(default=None, max_length=64)
    use_llm: bool | None = None
    mock_llm: bool | None = None
    api_key: str | None = Field(default=None, max_length=4096, repr=False, exclude=True)
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)


def invoke(operation):
    try:
        return JSONResponse(operation(), headers={'Cache-Control': 'no-store'})
    except (JobStoreError, ParamValidationError) as exc:
        code = exc.code
        status = 404 if code.endswith('not_found') else 409 if code.endswith('conflict') or code in {'conversation_busy', 'conversation_capacity'} else 400
        if code in {'auth_required', 'invalid_token'}:
            status = 401
        elif code == 'llm_quota_exceeded':
            status = 429
        elif code in {'run_enqueue_failed', 'run_capacity', 'run_transport_capacity', 'llm_quota_unavailable'}:
            status = 503
        raise HTTPException(status_code=status, detail=str(exc), headers={'X-Error-Code': code}) from exc


def build_conversation_router(*, service, principal, require_origin, configure=None, quota=None):
    router = APIRouter()

    @router.get('/api/conversations')
    def conversations(request: Request, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200), include_archived: bool = False):
        def operation():
            app, owner = service(), principal(request)
            result = app.list_conversations(owner, offset=offset, limit=limit, include_archived=include_archived)
            return {'ok': True, **result, 'scope': 'local' if owner == 'local' else 'account'}
        return invoke(operation)

    @router.get('/api/conversations/{conv_id}')
    def conversation(conv_id: str, request: Request, after_message: int = Query(default=0, ge=0), after_frame: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)):
        def operation():
            app, owner = service(), principal(request)
            result = app.get_conversation(owner, conv_id, after_message=after_message, after_frame=after_frame, limit=limit)
            return {'ok': True, 'conversation': result}
        return invoke(operation)

    @router.post('/api/conversations/actions')
    def create_or_import(payload: ConversationAction, request: Request):
        require_origin(request)
        return invoke(lambda: service().conversation_action(principal(request), '', **payload.model_dump()))

    @router.post('/api/conversations/{conv_id}/actions')
    def change(conv_id: str, payload: ConversationAction, request: Request):
        require_origin(request)
        return invoke(lambda: service().conversation_action(principal(request), conv_id, **payload.model_dump()))

    @router.get('/api/conversations/{conv_id}/replay')
    def replay(conv_id: str, request: Request, action: str = 'points'):
        return invoke(lambda: service().replay(principal(request), conv_id, action))

    @router.post('/api/conversations/{conv_id}/replay/{action}')
    def replay_write(conv_id: str, action: str, payload: OwnedReplayRequest, request: Request):
        require_origin(request)
        return invoke(lambda: service().replay(principal(request), conv_id, action,
            payload.data.model_dump(exclude_unset=True),
            configure=(lambda: configure(request, payload)) if configure else None,
            quota=(lambda cfg: quota(request, payload, cfg)) if quota else None))

    return router
