"""Shared project/global learning partition semantics for Agent evidence banks."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

_PROJECT = contextvars.ContextVar("biodata_learning_project", default="")


@contextmanager
def bank_transaction(root):
    from ..corpus.fs_utils import file_critical_section
    from ..app.runtime_paths import instance_data_dir_for
    with file_critical_section(instance_data_dir_for(Path(root), ".userdata") / "learning-ledgers.lock"):
        yield


def synchronized_bank(function):
    """Serialize read-modify-write operations on the shared learning ledgers."""
    @wraps(function)
    def invoke(root, *args, **kwargs):
        with bank_transaction(root):
            return function(root, *args, **kwargs)
    return invoke


def current_project_id() -> str:
    return _PROJECT.get()


def resolve_project_id(value: str | None) -> str:
    return current_project_id() if value is None else str(value or "")


def context_kwargs(context) -> dict:
    project_id = str(getattr(context, "project_id", "") or "")
    return {"project_id": project_id} if project_id else {}


@contextmanager
def bind_project_scope(project_id: str):
    token = _PROJECT.set(str(project_id or ""))
    try:
        yield
    finally:
        _PROJECT.reset(token)


_PRINCIPAL_ALIASES = {'local': 'anonymous'}


def canonical_principal(value) -> str:
    """同一本机身份的新旧账本名兼容；不改变账户身份或持久行，不泛化到其它账户。"""
    name = str(value or 'anonymous')
    return _PRINCIPAL_ALIASES.get(name, name)


def principal_variants(value) -> tuple[str, ...]:
    name = str(value or 'anonymous')
    canonical = canonical_principal(name)
    return tuple(dict.fromkeys([name, canonical] + [key for key, target in _PRINCIPAL_ALIASES.items() if target == canonical]))


def partition_of(row: dict) -> tuple[str, str, str]:
    return (canonical_principal(row.get("principal")), str(row.get("endpoint_fp") or ""),
            str(row.get("project_id") or ""))


def matches(row: dict, principal: str | None, endpoint_fp: str | None,
            project_id: str | None = None, *, inherit_global: bool = False) -> bool:
    if principal is None:
        return True                                                               
    owner, endpoint, project = partition_of(row)
    selected = resolve_project_id(project_id)
    return bool(owner == canonical_principal(principal) and endpoint == str(endpoint_fp or "")
                and (project == selected or (inherit_global and selected and project == "")))


def retain_per_partition(rows: list[dict], limit: int) -> list[dict]:
    """Keep recent evidence without allowing one project/account to evict another."""
    counts, retained = {}, []
    for row in reversed(rows):
        key = partition_of(row)
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0) + 1
            retained.append(row)
    return list(reversed(retained))
