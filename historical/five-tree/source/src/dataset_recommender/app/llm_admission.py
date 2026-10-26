"""Web/MCP 共用配置物化与身份额度；密钥仅在执行闭包内，网络调用不持环境锁。"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from ..llm.llm_client import load_llm_config, should_use_llm
from . import llm_quota
from .durable_jobs import JobStoreError

ENV_LOCK = threading.Lock()


@contextmanager
def temporary_env(overrides):
    old = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def materialize(*, config_root, project_root=None, overrides_factory=None, within=None,
                loader=load_llm_config, lock=ENV_LOCK):
    with lock:
        server = loader(project_root=config_root)
        overrides = overrides_factory(server) if overrides_factory else {}
        with temporary_env(overrides):
            if within:
                within()
            return loader(project_root=project_root or config_root)


def account_gate_required():
    return os.getenv('BIODATA_REQUIRE_ACCOUNT', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def gate_quota(project_root, *, username, cfg, api_key=None, use_llm=True,
               require_enabled=True, account_required=None):
    required = account_gate_required() if account_required is None else account_required
    if not required or not use_llm:
        return
    provider = str(cfg.provider or '')
    if provider != 'trial' and (api_key or '').strip():
        return
    if not should_use_llm(cfg, require_enabled=require_enabled)[0]:
        return
    if not username:
        raise JobStoreError('auth_required', 'auth_required')
    exempt = {s.lower() for s in os.getenv('BIODATA_LLM_QUOTA_EXEMPT', '').replace(',', ' ').split()}
    if username.lower() in exempt:
        return
    def integer(name, default):
        try:
            return max(0, int(os.getenv(name, '') or default))
        except ValueError:
            return default
    trial = provider == 'trial'
    prefix = 'BIODATA_TRIAL' if trial else 'BIODATA_LLM'
    try:
        llm_quota.check_and_increment(project_root, username, trial=trial,
            per_user_limit=integer(prefix + '_DAILY_PER_USER', 30 if trial else 100),
            global_limit=integer(prefix + '_DAILY_GLOBAL', 500 if trial else 1000))
    except llm_quota.QuotaExceeded as exc:
        raise JobStoreError('llm_quota_exceeded', str(exc)) from None
    except llm_quota.QuotaPathError:
        raise JobStoreError('llm_quota_unavailable', 'AI 配额存储配置不可用，请联系管理员。') from None


class Admission:
    def __init__(self, gate):
        self.gate = gate
        self._admitted = False
        self._lock = threading.Lock()

    def __call__(self, cfg):
        with self._lock:
            if not self._admitted:
                self.gate(cfg)
                self._admitted = True
