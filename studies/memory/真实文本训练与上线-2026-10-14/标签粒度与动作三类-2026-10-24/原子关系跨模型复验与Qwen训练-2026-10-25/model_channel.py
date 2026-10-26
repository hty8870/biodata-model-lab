                      
"""统一模型通道：一次调用 → 一个 outcome。

重试、预算、断点与落盘全部归采集器；本模块只把一次请求变成统一结果。
通道：codex（CLI）、chat（OpenAI 兼容）、anthropic（messages）、ollama（本机 /api/chat）。
纪律：
- 密钥只按环境变量名读取，不读/不建 .env 或 auth 文件，绝不写进 outcome/ledger。
- Ollama 无需密钥；thinking 只进 diagnostics，绝不拼进 text。
- 子进程一律 argv 列表 + shell=False，prompt 只走 stdin。
- stdout/stderr 有界保存，落盘前先脱敏。
- complete() 每次只发一次请求，失败不自动重试；availability 不探测远端。

用法：`python model_channel.py` 只跑离线 selftest，不联网。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

EXP = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 300.0
DIAG_LIMIT = 1200
_RAW_INPUT_LIMIT = 200_000
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"

                                   
OUTCOME_KEYS = frozenset({"text", "succeeded", "status", "finish_reason", "usage",
                          "latency_s", "provider", "model", "error_kind", "attempted",
                          "diagnostics"})

                                     
LEDGER_KEYS = ("schema", "ts", "case_id", "split", "arm", "atomic_version", "attempt",
               "channel", "model", "provider", "prompt_sha256", "prompt_chars",
               "attempted", "succeeded", "status", "finish_reason", "error_kind",
               "usage", "latency_s", "raw_text_sha256", "raw_text_chars",
               "cumulative_calls", "cumulative_tokens")

_SECRET_ENV_NAMES = ("OPENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "DASHSCOPE_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY",
                     "MOONSHOT_API_KEY", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY")

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._\-]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|authorization)\b"
               r"\s*[:=]\s*[\"']?[^\s\"',;]{8,}"),
)


class ChannelError(Exception):
    """带机器码的通道配置错误；消息不含密钥。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _known_secrets(extra: tuple[str, ...] = ()) -> list[str]:
    values = [item for item in extra if isinstance(item, str) and len(item) >= 8]
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            values.append(value)
    return values


def redact(text: Any, *, extra: tuple[str, ...] = (), limit: int = DIAG_LIMIT) -> str:
    """先按已知密钥值替换，再按模式脱敏，最后截尾到 limit。"""
    value = "" if text is None else str(text)
    value = value[-_RAW_INPUT_LIMIT:]
    for secret in sorted(_known_secrets(extra), key=len, reverse=True):
        value = value.replace(secret, "<redacted>")
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("<redacted>", value)
    if len(value) > limit:
        value = "…" + value[-limit:]
    return value


def normalize_usage(*, prompt_tokens: Any = None, completion_tokens: Any = None,
                    total_tokens: Any = None, source: str = "provider") -> dict:
    """input/output/total 三键统一口径；缺 total 时由 input+output 推导。"""

    def clean(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value <= 10 ** 9 else None

    prompt, completion, total = clean(prompt_tokens), clean(completion_tokens), clean(total_tokens)
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is None and prompt is None and completion is None:
        source = "none"
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "source": source}


def estimate_usage(prompt_text: str, completion_text: str = "") -> dict:
    """无 usage 的通道（Codex CLI）用字符数估算；带 source 标记，不冒充当计费值。"""
    def half(text: str) -> int:
        return int(math.ceil(len(text or "") / 2.0))

    prompt, completion = half(prompt_text), half(completion_text)
    return normalize_usage(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=prompt + completion,
                           source="estimated_chars_div_2")


def make_outcome(*, provider: str, model: str, text: Any = None, attempted: bool = True,
                 succeeded: bool = False, finish_reason: str | None = None,
                 usage: dict | None = None, latency_s: float = 0.0,
                 error_kind: str | None = None,
                 diagnostics: dict | None = None) -> dict:
    clean_text = text if isinstance(text, str) and text.strip() else None
    if succeeded and clean_text is None:
        succeeded, error_kind = False, error_kind or "empty_output"
    status = "ok" if succeeded else (error_kind or "llm_failure")
    outcome = {"text": clean_text, "succeeded": bool(succeeded), "status": status,
               "finish_reason": finish_reason, "usage": usage or normalize_usage(),
               "latency_s": round(float(latency_s), 3), "provider": provider,
               "model": model, "error_kind": error_kind,
               "attempted": bool(attempted), "diagnostics": dict(diagnostics or {})}
    assert set(outcome) == OUTCOME_KEYS
    return outcome


def ledger_row(*, case_id: str, split: str, arm: str, atomic_version: str, attempt: int,
               channel: Any, outcome: dict, prompt_sha256: str, prompt_chars: int,
               cumulative_calls: int, cumulative_tokens: int) -> dict:
    """ledger 只存元数据与脱敏诊断；prompt 正文留在 raw 记录里。"""
    text = outcome.get("text")
    row = {
        "schema": "atom-cross-model-call-v1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_id": case_id, "split": split, "arm": arm,
        "atomic_version": atomic_version, "attempt": int(attempt),
        "channel": channel.name, "model": channel.model,
        "provider": outcome["provider"],
        "prompt_sha256": prompt_sha256, "prompt_chars": int(prompt_chars),
        "attempted": outcome["attempted"], "succeeded": outcome["succeeded"],
        "status": outcome["status"], "finish_reason": outcome["finish_reason"],
        "error_kind": outcome["error_kind"], "usage": outcome["usage"],
        "latency_s": outcome["latency_s"],
        "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "raw_text_chars": len(text) if text else 0,
        "cumulative_calls": int(cumulative_calls),
        "cumulative_tokens": int(cumulative_tokens),
    }
    assert tuple(row) == LEDGER_KEYS
    return row


                                                     


def _run_subprocess(args: list[str], *, input: str, timeout: float, cwd: str,
                    env: dict | None = None):
    return subprocess.run(args, input=input, text=True, encoding="utf-8", errors="replace",
                          capture_output=True, timeout=timeout, cwd=cwd, env=env,
                          shell=False, check=False)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


class CodexCLIChannel:
    """`codex exec` 一次性调用；最终文本只从 -o 文件取。"""

    name = "codex"
    provider = "codex-cli"

    def __init__(self, model: str, *, effort: str = "high", timeout: float = DEFAULT_TIMEOUT,
                 cwd: Path | str | None = None, binary: str | None = None,
                 runner: Callable[..., Any] | None = None) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", str(model or "")):
            raise ChannelError("invalid_model", f"模型 ID 不合法：{model!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(effort or "")):
            raise ChannelError("invalid_effort", f"reasoning effort 不合法：{effort!r}")
        self.model = str(model)
        self.effort = str(effort)
        self.timeout = float(timeout)
        self.cwd = Path(cwd or EXP)
        self.binary = str(binary or "codex")
        self.runner = runner or _run_subprocess

    def resolve_binary(self) -> str | None:
        if os.path.dirname(self.binary):
            return self.binary if os.path.isfile(self.binary) else None
        return shutil.which(self.binary)

    def availability(self) -> tuple[bool, str]:
        return (True, "ready") if self.resolve_binary() else (False, "binary_missing")

    def command(self, out_path: str | Path, binary: str | None = None) -> list[str]:
        return [binary or self.resolve_binary() or self.binary, "exec", "--ephemeral",
                "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never",
                "-m", self.model, "-c", f'model_reasoning_effort="{self.effort}"',
                "-o", str(out_path), "-"]

    def complete(self, prompt: str) -> dict:
        started = time.perf_counter()
        binary = self.resolve_binary()
        if not binary:
            return make_outcome(provider=self.provider, model=self.model, attempted=False,
                                error_kind="binary_missing",
                                latency_s=time.perf_counter() - started,
                                diagnostics={"binary": redact(self.binary)})
        with tempfile.TemporaryDirectory(prefix="atomize-codex-") as tmp:
            out_path = Path(tmp) / "last-message.txt"
            args = self.command(out_path, binary)
            try:
                proc = self.runner(args, input=prompt, timeout=self.timeout,
                                   cwd=str(self.cwd), env=None)
            except subprocess.TimeoutExpired:
                return make_outcome(provider=self.provider, model=self.model,
                                    attempted=True, error_kind="timeout",
                                    usage=estimate_usage(prompt),
                                    latency_s=time.perf_counter() - started,
                                    diagnostics={"stderr_tail": f"timeout after {self.timeout}s"})
            except (OSError, subprocess.SubprocessError) as exc:
                return make_outcome(provider=self.provider, model=self.model,
                                    attempted=False, error_kind="spawn_error",
                                    latency_s=time.perf_counter() - started,
                                    diagnostics={"error": redact(str(exc))})
            latency = time.perf_counter() - started
            stdout = _as_text(getattr(proc, "stdout", ""))
            stderr = _as_text(getattr(proc, "stderr", ""))
            returncode = getattr(proc, "returncode", None)
            diag = {"returncode": returncode, "stdout_tail": redact(stdout),
                    "stderr_tail": redact(stderr)}
            if returncode != 0:
                return make_outcome(provider=self.provider, model=self.model,
                                    attempted=True, error_kind="nonzero_exit",
                                    usage=estimate_usage(prompt), latency_s=latency,
                                    diagnostics=diag)
            text = _read_text(out_path)
            if text is None:
                return make_outcome(provider=self.provider, model=self.model,
                                    attempted=True, error_kind="no_output",
                                    usage=estimate_usage(prompt), latency_s=latency,
                                    diagnostics=diag)
            text = text.strip()
            if not text:
                return make_outcome(provider=self.provider, model=self.model,
                                    attempted=True, error_kind="empty_output",
                                    usage=estimate_usage(prompt), latency_s=latency,
                                    diagnostics=diag)
            return make_outcome(provider=self.provider, model=self.model, text=text,
                                succeeded=True, finish_reason="stop",
                                usage=estimate_usage(prompt, text), latency_s=latency,
                                diagnostics=diag)


                                                              


def _check_endpoint(endpoint: str) -> str:
    value = str(endpoint or "").strip()
    if value and not re.match(r"^https?://", value, re.IGNORECASE):
        raise ChannelError("endpoint_invalid", "端点须为 http(s) 完整 URL。")
    return value


def _urllib_transport(endpoint: str, headers: dict, payload: dict, timeout: float):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def _body(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _openai_finish(finish: str | None) -> tuple[bool, str | None, str | None]:
    """返回 (可用, finish_reason, error_kind)。"""
    if finish in ("length", "max_tokens"):
        return False, "length", "truncated"
    if finish == "content_filter":
        return False, "content_filter", "content_filter"
    return True, finish, None


class ChatCompletionsChannel:
    """OpenAI 兼容 chat/completions；endpoint 必须是完整 URL（含路径）。"""

    name = "chat"
    provider = "openai-chat"
    default_key_env = "OPENCODE_API_KEY"

    def __init__(self, model: str, *, endpoint: str = "", api_key_env: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, max_tokens: int = 1600,
                 temperature: float = 0.2,
                 transport: Callable[..., Any] | None = None) -> None:
        if not str(model or "").strip():
            raise ChannelError("invalid_model", "模型 ID 为空。")
        self.model = str(model)
        self.endpoint = _check_endpoint(endpoint)
        self.api_key_env = api_key_env or self.default_key_env
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.transport = transport or _urllib_transport

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def availability(self) -> tuple[bool, str]:
        if not self.endpoint:
            return False, "endpoint_missing"
        if not self.api_key():
            return False, f"no_key:{self.api_key_env}"
        return True, "ready"

    def complete(self, prompt: str) -> dict:
        started = time.perf_counter()
        endpoint_ok = bool(self.endpoint)
        if not endpoint_ok or not self.api_key():
            return make_outcome(provider=self.provider, model=self.model, attempted=False,
                                error_kind="endpoint_missing" if not endpoint_ok else "no_key",
                                latency_s=time.perf_counter() - started,
                                diagnostics={"api_key_env": self.api_key_env})
        key = self.api_key() or ""
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "stream": False}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            status, body = self.transport(self.endpoint, headers, payload, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="network", latency_s=time.perf_counter() - started,
                                diagnostics={"error": redact(str(exc))})
        latency = time.perf_counter() - started
        if not 200 <= int(status) < 300:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="http", latency_s=latency,
                                diagnostics={"http_status": int(status),
                                             "body_tail": redact(body)})
        try:
            value = json.loads(body)
            choice = value["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason")
        except (json.JSONDecodeError, LookupError, TypeError, AttributeError):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        usage_raw = value.get("usage") if isinstance(value, dict) else None
        usage = normalize_usage(
            prompt_tokens=(usage_raw or {}).get("prompt_tokens"),
            completion_tokens=(usage_raw or {}).get("completion_tokens"),
            total_tokens=(usage_raw or {}).get("total_tokens"))
        if not isinstance(text, str):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", usage=usage, latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        usable, finish, error_kind = _openai_finish(finish)
        if not usable:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind=error_kind, finish_reason=finish, usage=usage,
                                latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        return make_outcome(provider=self.provider, model=self.model, text=text,
                            succeeded=True, finish_reason=finish, usage=usage,
                            latency_s=latency, diagnostics={})


class AnthropicMessagesChannel:
    """Anthropic messages 形状；Qwen 的 Anthropic 兼容端点必须由调用方显式给出。"""

    name = "anthropic"
    provider = "anthropic-messages"
    default_key_env = "ANTHROPIC_API_KEY"
    anthropic_version = "2023-06-01"

    def __init__(self, model: str, *, endpoint: str = "", api_key_env: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, max_tokens: int = 1600,
                 temperature: float = 0.2,
                 transport: Callable[..., Any] | None = None) -> None:
        if not str(model or "").strip():
            raise ChannelError("invalid_model", "模型 ID 为空。")
        self.model = str(model)
        self.endpoint = _check_endpoint(endpoint)
        self.api_key_env = api_key_env or self.default_key_env
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.transport = transport or _urllib_transport

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def availability(self) -> tuple[bool, str]:
        if not self.endpoint:
            return False, "endpoint_missing"
        if not self.api_key():
            return False, f"no_key:{self.api_key_env}"
        return True, "ready"

    def complete(self, prompt: str) -> dict:
        started = time.perf_counter()
        endpoint_ok = bool(self.endpoint)
        if not endpoint_ok or not self.api_key():
            return make_outcome(provider=self.provider, model=self.model, attempted=False,
                                error_kind="endpoint_missing" if not endpoint_ok else "no_key",
                                latency_s=time.perf_counter() - started,
                                diagnostics={"api_key_env": self.api_key_env})
        key = self.api_key() or ""
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "temperature": self.temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": key, "anthropic-version": self.anthropic_version,
                   "Content-Type": "application/json"}
        try:
            status, body = self.transport(self.endpoint, headers, payload, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="network", latency_s=time.perf_counter() - started,
                                diagnostics={"error": redact(str(exc))})
        latency = time.perf_counter() - started
        if not 200 <= int(status) < 300:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="http", latency_s=latency,
                                diagnostics={"http_status": int(status),
                                             "body_tail": redact(body)})
        try:
            value = json.loads(body)
            blocks = value["content"]
            text = "\n".join(str(block.get("text") or "") for block in blocks
                             if isinstance(block, dict) and block.get("type") == "text")
            stop = value.get("stop_reason")
        except (json.JSONDecodeError, LookupError, TypeError, AttributeError):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        usage_raw = value.get("usage") or {}
        usage = normalize_usage(prompt_tokens=usage_raw.get("input_tokens"),
                                completion_tokens=usage_raw.get("output_tokens"),
                                total_tokens=usage_raw.get("total_tokens"))
        finish = "length" if stop in ("max_tokens", "length") else stop
        if not text.strip():
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="truncated" if finish == "length" else "empty_output",
                                finish_reason=finish, usage=usage, latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        return make_outcome(provider=self.provider, model=self.model, text=text,
                            succeeded=True, finish_reason=finish, usage=usage,
                            latency_s=latency, diagnostics={})


def _ollama_endpoint(value: str | None) -> str:
    """默认本地 /api/chat；只接受 http(s)。裸 host 补文档路径，不猜其它端点。"""
    url = str(value or "").strip() or DEFAULT_OLLAMA_ENDPOINT
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ChannelError("endpoint_invalid", "Ollama 端点须为 http(s) URL。")
    if parsed.path in ("", "/"):
        url = url.rstrip("/") + "/api/chat"
    return url


def _is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(str(url or ""))
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class OllamaChannel:
    """本机 Ollama `/api/chat`；无鉴权，content 与 thinking 严格分离。

    availability 只看端点形状，绝不探测 daemon：真实连通由主 Agent 负责。
    """

    name = "ollama"
    provider = "ollama"

    def __init__(self, model: str, *, endpoint: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, num_predict: int = 2048,
                 transport: Callable[..., Any] | None = None) -> None:
        text = str(model or "").strip()
        if not text:
            raise ChannelError("invalid_model", "模型 ID 为空。")
        if len(text) > 200:
            raise ChannelError("invalid_model", "模型 ID 过长。")
        self.model = text
        self.endpoint = _ollama_endpoint(endpoint)
        self.timeout = float(timeout)
        self.num_predict = int(num_predict)
        self.transport = transport or _urllib_transport

    def availability(self) -> tuple[bool, str]:
        return ((True, "configured") if _is_http_url(self.endpoint)
                else (False, "endpoint_invalid"))

    def complete(self, prompt: str) -> dict:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }
        headers = {"Content-Type": "application/json"}
        try:
            status, body = self.transport(self.endpoint, headers, payload, self.timeout)
        except TimeoutError as exc:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="timeout",
                                latency_s=time.perf_counter() - started,
                                diagnostics={"error": redact(str(exc))})
        except (urllib.error.URLError, OSError) as exc:
            text = str(exc).lower()
            kind = "timeout" if "timed out" in text or "timeout" in text else "network"
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind=kind, latency_s=time.perf_counter() - started,
                                diagnostics={"error": redact(str(exc))})
        latency = time.perf_counter() - started
        if not 200 <= int(status) < 300:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="http", latency_s=latency,
                                diagnostics={"http_status": int(status),
                                             "body_tail": redact(body)})
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        if not isinstance(value, dict):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        try:
            message = value["message"]
            content = message.get("content")
            thinking = message.get("thinking")
        except (KeyError, TypeError, AttributeError):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", latency_s=latency,
                                diagnostics={"body_tail": redact(body)})
        done = value.get("done")
        done_reason = value.get("done_reason")
        finish = done_reason if isinstance(done_reason, str) else None
        usage = normalize_usage(prompt_tokens=value.get("prompt_eval_count"),
                                completion_tokens=value.get("eval_count"))
        diag: dict[str, Any] = {}
        if isinstance(thinking, str) and thinking.strip():
            diag["thinking_chars"] = len(thinking)
            diag["thinking_sha256"] = hashlib.sha256(thinking.encode("utf-8")).hexdigest()
            diag["thinking_tail"] = redact(thinking, limit=400)
        for key, label in (("total_duration", "total_duration_ms"),
                           ("load_duration", "load_duration_ms"),
                           ("prompt_eval_duration", "prompt_eval_duration_ms"),
                           ("eval_duration", "eval_duration_ms")):
            raw = value.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                diag[label] = round(raw / 1_000_000, 3)
        if done is False:
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", finish_reason=finish,
                                usage=usage, latency_s=latency,
                                diagnostics={**diag, "body_tail": redact(body)})
        if not isinstance(content, str):
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="bad_response", finish_reason=finish,
                                usage=usage, latency_s=latency,
                                diagnostics={**diag, "body_tail": redact(body)})
        if finish in ("length", "max_tokens"):
            diag.update({"partial_text_chars": len(content),
                         "partial_text_sha256":
                             hashlib.sha256(content.encode("utf-8")).hexdigest(),
                         "partial_text_tail": redact(content, limit=400)})
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="truncated", finish_reason="length",
                                usage=usage, latency_s=latency, diagnostics=diag)
        if finish == "content_filter":
            diag.update({"partial_text_chars": len(content),
                         "partial_text_sha256":
                             hashlib.sha256(content.encode("utf-8")).hexdigest(),
                         "partial_text_tail": redact(content, limit=400)})
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="content_filter", finish_reason=finish,
                                usage=usage, latency_s=latency, diagnostics=diag)
        if not content.strip():
            return make_outcome(provider=self.provider, model=self.model, attempted=True,
                                error_kind="empty_output", finish_reason=finish,
                                usage=usage, latency_s=latency, diagnostics=diag)
        return make_outcome(provider=self.provider, model=self.model, text=content,
                            succeeded=True, finish_reason=finish, usage=usage,
                            latency_s=latency, diagnostics=diag)


def make_channel(name: str, model: str, *, endpoint: str | None = None,
                 api_key_env: str | None = None, effort: str = "high",
                 timeout: float = DEFAULT_TIMEOUT, max_tokens: int = 1600,
                 temperature: float = 0.2, cwd: Path | str | None = None,
                 binary: str | None = None, runner: Callable[..., Any] | None = None,
                 transport: Callable[..., Any] | None = None):
    if name == "codex":
        return CodexCLIChannel(model, effort=effort, timeout=timeout, cwd=cwd,
                               binary=binary, runner=runner)
    if name == "chat":
        return ChatCompletionsChannel(model, endpoint=endpoint or "",
                                      api_key_env=api_key_env, timeout=timeout,
                                      max_tokens=max_tokens, temperature=temperature,
                                      transport=transport)
    if name == "anthropic":
        return AnthropicMessagesChannel(model, endpoint=endpoint or "",
                                        api_key_env=api_key_env, timeout=timeout,
                                        max_tokens=max_tokens, temperature=temperature,
                                        transport=transport)
    if name == "ollama":
        return OllamaChannel(model, endpoint=endpoint, timeout=timeout,
                             transport=transport)
    raise ChannelError("unknown_channel", f"未知通道：{name}")


def channel_status(*, endpoint: str | None = None, api_key_env: str | None = None,
                   binary: str | None = None) -> dict:
    """通道可用性（不联网）：API 通道必须有显式端点 + 进程环境里的密钥。"""
    chat_env = api_key_env or ChatCompletionsChannel.default_key_env
    anth_env = api_key_env or AnthropicMessagesChannel.default_key_env
    if binary:
        codex_bin = binary if (os.path.isfile(binary) or shutil.which(binary)) else None
    else:
        codex_bin = shutil.which("codex")
    detail = {
        "codex": "ready" if codex_bin else "binary_missing",
        "chat": ("ready" if endpoint and os.environ.get(chat_env)
                 else "endpoint_missing" if not endpoint else f"no_key:{chat_env}"),
        "anthropic": ("ready" if endpoint and os.environ.get(anth_env)
                      else "endpoint_missing" if not endpoint else f"no_key:{anth_env}"),
        "ollama": ("configured" if _is_http_url(DEFAULT_OLLAMA_ENDPOINT)
                   else "endpoint_invalid"),
    }
    return {"available": {key: value in ("ready", "configured")
                          for key, value in detail.items()},
            "detail": detail}


                                                       


def _fake_runner(text: str | None = None, *, returncode: int = 0, stdout: str = "",
                 stderr: str = "", raises: Exception | None = None):
    from types import SimpleNamespace

    def runner(args, *, input, timeout, cwd, env=None):
        if raises is not None:
            raise raises
        if text is not None:
            out = Path(args[args.index("-o") + 1])
            out.write_text(text, encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return runner


def selftest() -> dict:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            checks[name] = True
        except Exception as exc:                 
            checks[name] = False
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    secret = "sk-" + "Z" * 24

    def command_shape() -> None:
        channel = CodexCLIChannel("gpt-6.1-sol", effort="high", runner=_fake_runner("[]"))
        args = channel.command("C:/tmp/out.txt")
        assert args[1] == "exec"
        assert args[-1] == "-"
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("--color") + 1] == "never"
        assert args[args.index("-m") + 1] == "gpt-6.1-sol"
        assert args[args.index("-c") + 1] == 'model_reasoning_effort="high"'
        assert args[args.index("-o") + 1].endswith("out.txt")
        for flag in ("--ephemeral", "--skip-git-repo-check"):
            assert flag in args

    def codex_success() -> None:
        channel = CodexCLIChannel("m", runner=_fake_runner("[]"))
        out = channel.complete("prompt")
        assert out["succeeded"] is True and out["text"] == "[]"
        assert out["status"] == "ok"
        assert out["usage"]["source"] == "estimated_chars_div_2"
        assert set(out) == OUTCOME_KEYS

    def codex_error_redaction() -> None:
        channel = CodexCLIChannel("m", runner=_fake_runner(
            None, returncode=2, stderr=f"Authorization: Bearer {secret}"))
        out = channel.complete("p")
        assert out["succeeded"] is False and out["error_kind"] == "nonzero_exit"
        blob = json.dumps(out, ensure_ascii=False)
        assert secret not in blob and "sk-" not in blob, blob

    def codex_timeout() -> None:
        channel = CodexCLIChannel("m", timeout=1, runner=_fake_runner(
            raises=subprocess.TimeoutExpired("codex", 1)))
        out = channel.complete("p")
        assert out["error_kind"] == "timeout" and out["attempted"] is True

    def codex_empty_and_missing() -> None:
        assert CodexCLIChannel("m", runner=_fake_runner("")).complete("p")["error_kind"] == "empty_output"
        assert CodexCLIChannel("m", runner=_fake_runner(None)).complete("p")["error_kind"] == "no_output"

    def chat_success() -> None:
        env_name = "MODEL_CHANNEL_SELFTEST_DUMMY"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "sk-selftest-not-a-real-key"
        seen: dict = {}

        def transport(url, headers, payload, timeout):
            seen.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, json.dumps({"choices": [{"message": {"content": "[]"},
                                                "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                              "total_tokens": 12}})

        try:
            channel = ChatCompletionsChannel(
                "qwen-test", endpoint="https://example.invalid/v1/chat/completions",
                api_key_env=env_name, transport=transport)
            assert channel.availability() == (True, "ready")
            out = channel.complete("prompt")
            assert out["succeeded"] and out["text"] == "[]"
            assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 2,
                                    "total_tokens": 12, "source": "provider"}
            assert seen["payload"]["messages"][0]["content"] == "prompt"
            assert seen["headers"]["Authorization"].startswith("Bearer ")
            assert "sk-selftest-not-a-real-key" not in json.dumps(out)
        finally:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

    def chat_no_key() -> None:
        channel = ChatCompletionsChannel(
            "q", endpoint="https://example.invalid/v1/chat/completions",
            api_key_env="MODEL_CHANNEL_SELFTEST_ABSENT")
        out = channel.complete("p")
        assert out["attempted"] is False and out["error_kind"] == "no_key"
        assert channel.availability()[0] is False

    def chat_http_error_redacted() -> None:
        env_name = "MODEL_CHANNEL_SELFTEST_DUMMY2"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "sk-selftest-not-a-real-key"
        try:
            channel = ChatCompletionsChannel(
                "q", endpoint="https://example.invalid/v1/chat/completions",
                api_key_env=env_name,
                transport=lambda *_: (401, '{"error":"invalid ' + secret + '"}'))
            out = channel.complete("p")
            assert out["error_kind"] == "http"
            assert out["diagnostics"]["http_status"] == 401
            assert secret not in json.dumps(out, ensure_ascii=False)
        finally:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

    def anthropic_success() -> None:
        env_name = "MODEL_CHANNEL_SELFTEST_DUMMY3"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "sk-selftest-not-a-real-key"
        try:
            channel = AnthropicMessagesChannel(
                "qwen-test", endpoint="https://example.invalid/v1/messages",
                api_key_env=env_name,
                transport=lambda *_: (200, json.dumps({
                    "content": [{"type": "text", "text": "[]"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1}})))
            out = channel.complete("prompt")
            assert out["succeeded"] and out["text"] == "[]"
            assert out["usage"]["prompt_tokens"] == 5
            assert "x-api-key" not in json.dumps(out)
        finally:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

    def ollama_endpoint_and_availability() -> None:
        calls: list[int] = []

        def transport(*_args):
            calls.append(1)
            raise AssertionError("availability 不应发起请求")

        channel = OllamaChannel("qwen3:0.6b", transport=transport)
        assert channel.endpoint == DEFAULT_OLLAMA_ENDPOINT
        assert channel.availability() == (True, "configured")
        assert calls == []
        assert OllamaChannel("qwen3:0.6b",
                             endpoint="http://127.0.0.1:11434").endpoint == DEFAULT_OLLAMA_ENDPOINT
        assert OllamaChannel("qwen3:0.6b",
                             endpoint="http://host:11434/custom").endpoint == "http://host:11434/custom"
        assert OllamaChannel("batiai/qwen3.8-27b:iq3").model == "batiai/qwen3.8-27b:iq3"

    def ollama_body_and_success() -> None:
        thinking = "内部推理，不应进 text"
        seen: dict = {}

        def transport(url, headers, payload, timeout):
            seen.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, json.dumps({
                "message": {"role": "assistant", "content": "[]", "thinking": thinking},
                "done": True, "done_reason": "stop",
                "prompt_eval_count": 12, "eval_count": 3,
                "total_duration": 1_500_000_000,
            }, ensure_ascii=False)

        channel = OllamaChannel("qwen3:0.6b", transport=transport)
        out = channel.complete("prompt")
        assert seen["url"] == DEFAULT_OLLAMA_ENDPOINT
        assert "Authorization" not in seen["headers"]
        payload = seen["payload"]
        assert payload["model"] == "qwen3:0.6b"
        assert payload["messages"] == [{"role": "user", "content": "prompt"}]
        assert payload["stream"] is False and payload["think"] is False
        assert payload["keep_alive"] == "10m"
        assert payload["options"] == {"temperature": 0, "num_predict": 2048}
        assert out["succeeded"] is True and out["text"] == "[]"
        assert thinking not in out["text"]
        assert out["finish_reason"] == "stop"
        assert out["usage"] == {"prompt_tokens": 12, "completion_tokens": 3,
                                "total_tokens": 15, "source": "provider"}
        assert out["diagnostics"]["thinking_chars"] == len(thinking)
        assert out["diagnostics"]["thinking_sha256"]
        assert out["diagnostics"]["thinking_tail"] == thinking
        assert out["diagnostics"]["total_duration_ms"] == 1500.0
        assert set(out) == OUTCOME_KEYS

    def ollama_thinking_only() -> None:
        thinking = "只有思考"

        def transport(*_args):
            return 200, json.dumps({"message": {"content": "", "thinking": thinking},
                                    "done": True, "done_reason": "stop",
                                    "prompt_eval_count": 5, "eval_count": 2},
                                   ensure_ascii=False)

        out = OllamaChannel("qwen3:0.6b", transport=transport).complete("p")
        assert out["succeeded"] is False and out["error_kind"] == "empty_output"
        assert out["text"] is None
        assert out["diagnostics"]["thinking_chars"] == len(thinking)
        assert out["diagnostics"]["thinking_tail"] == thinking

    def ollama_length_and_usage() -> None:
        partial = "部分输出"

        def transport(*_args):
            return 200, json.dumps({"message": {"content": partial},
                                    "done": True, "done_reason": "length",
                                    "prompt_eval_count": 5, "eval_count": 9},
                                   ensure_ascii=False)

        out = OllamaChannel("qwen3:0.6b", transport=transport).complete("p")
        assert out["error_kind"] == "truncated" and out["text"] is None
        assert out["finish_reason"] == "length"
        assert out["usage"]["total_tokens"] == 14
        assert out["diagnostics"]["partial_text_chars"] == len(partial)
        assert out["diagnostics"]["partial_text_sha256"]
        assert partial not in (out["text"] or "")

    def ollama_errors() -> None:
        local_secret = "sk-" + "O" * 24

        def http_error(*_args):
            return 500, '{"error":"boom ' + local_secret + '"}'

        out = OllamaChannel("m", transport=http_error).complete("p")
        assert out["error_kind"] == "http" and out["diagnostics"]["http_status"] == 500
        assert local_secret not in json.dumps(out, ensure_ascii=False)

        def network_error(*_args):
            raise urllib.error.URLError("connection refused")

        out = OllamaChannel("m", transport=network_error).complete("p")
        assert out["error_kind"] == "network" and out["attempted"] is True

        def timeout_error(*_args):
            raise TimeoutError("timed out")

        out = OllamaChannel("m", transport=timeout_error).complete("p")
        assert out["error_kind"] == "timeout"

        def bad_json(*_args):
            return 200, "<html>not json</html>"

        out = OllamaChannel("m", transport=bad_json).complete("p")
        assert out["error_kind"] == "bad_response"

        def done_false(*_args):
            return 200, json.dumps({"message": {"content": "[]"}, "done": False})

        out = OllamaChannel("m", transport=done_false).complete("p")
        assert out["error_kind"] == "bad_response"

        def missing_message(*_args):
            return 200, json.dumps({"done": True})

        out = OllamaChannel("m", transport=missing_message).complete("p")
        assert out["error_kind"] == "bad_response"

        def no_usage(*_args):
            return 200, json.dumps({"message": {"content": "[]"}, "done": True,
                                    "done_reason": "stop"})

        out = OllamaChannel("m", transport=no_usage).complete("p")
        assert out["succeeded"] is True and out["usage"]["source"] == "none"

    def redaction_helper() -> None:
        text = redact("k=supersecretvalue other sk-" + "Q" * 24, extra=("supersecretvalue",))
        assert "supersecretvalue" not in text and "Q" * 24 not in text

    def ledger_shape() -> None:
        channel = CodexCLIChannel("m", runner=_fake_runner("[]"))
        outcome = channel.complete("prompt")
        row = ledger_row(case_id="c1", split="dev", arm="A1", atomic_version="v2",
                         attempt=1, channel=channel, outcome=outcome,
                         prompt_sha256="0" * 64, prompt_chars=6,
                         cumulative_calls=1, cumulative_tokens=3)
        assert tuple(row) == LEDGER_KEYS
        assert "sk-" not in json.dumps(row, ensure_ascii=False)

    def status_shape() -> None:
        status = channel_status(endpoint=None)
        assert set(status["available"]) == {"codex", "chat", "anthropic", "ollama"}
        assert all(isinstance(value, bool) for value in status["available"].values())
        assert status["detail"]["ollama"] == "configured"
        assert status["available"]["ollama"] is True

    check("codex_command_shape", command_shape)
    check("codex_success", codex_success)
    check("codex_error_redaction", codex_error_redaction)
    check("codex_timeout", codex_timeout)
    check("codex_empty_and_missing", codex_empty_and_missing)
    check("chat_success", chat_success)
    check("chat_no_key", chat_no_key)
    check("chat_http_error_redacted", chat_http_error_redacted)
    check("anthropic_success", anthropic_success)
    check("ollama_endpoint_and_availability", ollama_endpoint_and_availability)
    check("ollama_body_and_success", ollama_body_and_success)
    check("ollama_thinking_only", ollama_thinking_only)
    check("ollama_length_and_usage", ollama_length_and_usage)
    check("ollama_errors", ollama_errors)
    check("redaction_helper", redaction_helper)
    check("ledger_shape", ledger_shape)
    check("status_shape", status_shape)
    return {"ok": not failures, "checks": checks, "failures": failures}


if __name__ == "__main__":
    result = selftest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 2)
