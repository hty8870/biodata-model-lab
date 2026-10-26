"""HTTP adapters for account-owned conversation project metadata."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .workspace_store import DEFAULT_PROJECT_COLOR, MAX_PROJECTS, WorkspaceError, WorkspaceStore


class ProjectNameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    color: str = Field(default=DEFAULT_PROJECT_COLOR, pattern=r"^#[0-9a-fA-F]{6}$")


class ProjectUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")

    @model_validator(mode="after")
    def require_update(self):
        if self.name is None and self.color is None:
            raise ValueError("请提供项目名称或颜色。")
        return self


class ProjectArchiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archived: bool = True


class ProjectOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_ids: list[str] = Field(max_length=MAX_PROJECTS)


class ConversationProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(default="", max_length=80)
    expected_revision: int | None = Field(default=None, ge=0)


def build_workspace_router(*, project_root: Callable[[], Path],
                           principal: Callable[[Request], str],
                           require_origin: Callable[[Request], None], service=None) -> APIRouter:
    router = APIRouter()

    identity = principal
    def request_owner(request):
        value = identity(request)
        return 'local' if value == 'anonymous' and request.query_params.get('shared', '').lower() in {'true', '1'} else value

    def invoke(operation):
        try:
            return JSONResponse(operation(), headers={"Cache-Control": "no-store"})
        except WorkspaceError as exc:
            status = {"project_not_found": 404, "conversation_project_changed": 409,
                      "project_limit": 409, "project_order_changed": 409}.get(exc.code, 400)
            raise HTTPException(status_code=status, detail=exc.message,
                                headers={"X-Error-Code": exc.code}) from exc

    def store():
        return WorkspaceStore(project_root())

    @router.get("/api/conversation-projects")
    def workspace_snapshot(request: Request):
        def load():
            database, owner = store(), request_owner(request)
            return {"ok": True, "projects": database.list_projects(owner, include_archived=True),
                    "assignments": database.assignments(owner)}
        return invoke(load)

    @router.post("/api/conversation-projects")
    def project_create(payload: ProjectNameRequest, request: Request):
        require_origin(request)
        return invoke(lambda: {"ok": True, "project": store().create_project(
            request_owner(request), payload.name, color=payload.color)})

    @router.post("/api/conversation-projects/reorder")
    def project_reorder(payload: ProjectOrderRequest, request: Request):
        require_origin(request)
        return invoke(lambda: {"ok": True, "projects": store().reorder_projects(request_owner(request), payload.project_ids)})

    @router.patch("/api/conversation-projects/{project_id}")
    def project_update(project_id: str, payload: ProjectUpdateRequest, request: Request):
        require_origin(request)
        return invoke(lambda: {"ok": True, "project": store().update_project(
            request_owner(request), project_id, name=payload.name, color=payload.color)})

    @router.post("/api/conversation-projects/{project_id}/archive")
    def project_archive(project_id: str, payload: ProjectArchiveRequest, request: Request):
        require_origin(request)
        return invoke(lambda: {"ok": True, "project": store().archive_project(
            request_owner(request), project_id, archived=payload.archived)})

    @router.put("/api/conversations/{conversation_id}/project")
    def conversation_move(conversation_id: str, payload: ConversationProjectRequest, request: Request):
        require_origin(request)
        def move():
            owner = request_owner(request)
            if service is not None and service().store.exists(owner, conversation_id, include_deleted=True):
                return service().conversation_action(owner, conversation_id, 'project',
                    expected_revision=payload.expected_revision, project_id=payload.project_id)
            return {"ok": True, **store().assign_conversation(owner, conversation_id, payload.project_id)}
        from .conversation_api import invoke as shared_invoke
        return shared_invoke(move)

    return router
