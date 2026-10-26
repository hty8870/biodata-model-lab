"""Cooperative cancellation at safe Agent execution boundaries."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from functools import wraps

_CHECK = contextvars.ContextVar("biodata_run_check", default=None)
_COMPLETED = contextvars.ContextVar("biodata_run_completed", default=None)


class _RunStopped(BaseException):
    def __init__(self, completed_result=None):
        super().__init__()
        self.completed_result = completed_result


class RunCancelled(_RunStopped):
    """Not a model failure: bypass fallback handlers that would otherwise keep executing."""


class RunTimedOut(_RunStopped):
    """A resource deadline stops further work at the next safe boundary."""


def current_check():
    return _CHECK.get()


def completed_snapshot():
    return _COMPLETED.get()


def _check_boundary(check):
    try:
        check()
    except (RunCancelled, RunTimedOut) as exc:
        if exc.completed_result is None:
            exc.completed_result = completed_snapshot()
        raise


def checkpoint() -> None:
    check = current_check()
    if check is not None:
        _check_boundary(check)


@contextmanager
def bind_run_check(check):
    token = _CHECK.set(check)
    completed_token = _COMPLETED.set(None)
    try:
        yield
    finally:
        _COMPLETED.reset(completed_token)
        _CHECK.reset(token)


def cancellable_node(function):
    @wraps(function)
    def invoke(state, *, runtime):
        check = getattr(runtime.context, "cancel_check", None)
        if check is not None:
            _check_boundary(check)
        result = function(state, runtime=runtime)
        if isinstance(result, dict):
            plan = result.get("plan")
            if result.get("steps") or (isinstance(plan, dict) and plan.get("steps")):
                _COMPLETED.set(result)
        if check is not None:
            try:
                _check_boundary(check)
            except (RunCancelled, RunTimedOut) as exc:
                if exc.completed_result is None:
                    exc.completed_result = result
                raise
        return result
    return invoke
