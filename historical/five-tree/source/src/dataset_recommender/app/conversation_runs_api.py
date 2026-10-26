"""Owned task status, event replay and cooperative cancellation HTTP adapters."""
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .conversation_runs import RunError


class ResolveRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    req_id: str = Field(min_length=1, max_length=128)
    shared: bool = False


def build_runs_router(*, runtime, principal, require_origin):
    router = APIRouter()

    def invoke(operation):
        try:
            return JSONResponse(operation(), headers={"Cache-Control": "no-store"})
        except RunError as exc:
            raise HTTPException(status_code=404 if exc.code == "run_not_found" else 400,
                                detail=exc.message, headers={"X-Error-Code": exc.code}) from exc

    def owner(request, shared):
        value = principal(request)
        return "local" if shared and value == "anonymous" else value

    @router.post("/api/runs/resolve")
    def resolve(payload: ResolveRunRequest, request: Request):
        require_origin(request)
        return invoke(lambda: {"ok": True, "run": runtime().store.find_request(owner(request, payload.shared), payload.req_id)})

    @router.get("/api/runs/{run_id}")
    def status(run_id: str, request: Request, shared: bool = False):
        return invoke(lambda: {"ok": True, "run": runtime().store.load(owner(request, shared), run_id)})

    @router.get("/api/runs/{run_id}/events")
    def events(run_id: str, request: Request, after: int = Query(default=0, ge=0), shared: bool = False):
        return invoke(lambda: {"ok": True, "events": runtime().store.events(owner(request, shared), run_id, after)})

    @router.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str, request: Request, shared: bool = False):
        require_origin(request)
        return invoke(lambda: {"ok": True, "run": runtime().cancel(owner(request, shared), run_id)})

    return router
