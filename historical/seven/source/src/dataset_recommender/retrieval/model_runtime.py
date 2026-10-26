                       
"""隔离本地模型运行时的路径与 JSONL 客户端。

在线安装的 torch/transformers 全部住 data_root/model-runtime/venv，主 FastAPI 进程
不修改 sys.path。打分通过该 venv 的常驻 worker 进行，依赖冲突或 worker 崩溃只会让
vector_recall 回退规则顺序。
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

from ..app.runtime_paths import AppPaths, get_app_paths
from .model_worker import (
    MAX_DOCUMENT_CHARS,
    MAX_PAIRS,
    MAX_QUERY_CHARS,
    MAX_REQUEST_BYTES,
    MAX_TEXTS,
    MODEL_ID,
    model_files_ready,
)
from .vector_recall import DEFAULT_CROSS_ENCODER_MODEL, DEFAULT_EMBEDDING_MODEL

READY_SCHEMA = "biodata-model-runtime/v1"
STARTUP_TIMEOUT_S = 300.0
SCORE_TIMEOUT_S = 180.0
                                         
                                                              
                                    
REQUEST_BYTES_BUDGET = MAX_REQUEST_BYTES - 64 * 1024


def _json_size(item: object) -> int:
    """序列化**字节**数（UTF-8）：中文一字三字节，按字符数计预算会低估三倍。"""
    return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _split_by_budget(items: Sequence, *, limit: int = REQUEST_BYTES_BUDGET) -> "list[list]":
    """按序列化字节预算把 items 切块（pairs/texts 共用同一机制）。

    单条自身超预算 → `ValueError`（fail-fast：绝不发出去换一个必然被丢弃的等待）。"""
    chunks: "list[list]" = []
    current: list = []
    size = 2                
    for item in items:
        piece = _json_size(item) + 1            
        if piece + 2 > limit:
            raise ValueError("request_item_too_large")
        if current and size + piece > limit:
            chunks.append(current)
            current, size = [], 2
        current.append(item)
        size += piece
    if current:
        chunks.append(current)
    return chunks


def runtime_root(paths: "AppPaths | None" = None) -> Path:
    return (paths or get_app_paths()).data_root / "model-runtime"


def runtime_python(paths: "AppPaths | None" = None) -> Path:
    return runtime_root(paths) / "venv" / "Scripts" / "python.exe"


def ready_manifest_path(paths: "AppPaths | None" = None) -> Path:
    return runtime_root(paths) / "READY.json"


def worker_script(paths: "AppPaths | None" = None) -> Path:
    resolved = paths or get_app_paths()
    if resolved.runtime_mode == "frozen":
        return resolved.resource_root / "tools" / "model_worker.py"
    return Path(__file__).resolve().with_name("model_worker.py")


def model_dir(paths: "AppPaths | None" = None) -> Path:
    return (paths or get_app_paths()).model_root / "cross_encoders" / DEFAULT_CROSS_ENCODER_MODEL


def embed_model_dir(paths: "AppPaths | None" = None) -> Path:
    return (paths or get_app_paths()).model_root / "embeddings" / DEFAULT_EMBEDDING_MODEL


def read_ready_manifest(paths: "AppPaths | None" = None) -> dict:
    path = ready_manifest_path(paths)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(value, dict) or value.get("schema") != READY_SCHEMA or value.get("model_id") != MODEL_ID:
        return {}
    return value


def _manifest_files_intact(manifest: dict, paths: "AppPaths | None" = None) -> bool:
    """核对逐文件 SHA256 证据；无哈希的旧清单不认证 READY，显式重装后恢复。"""
    files = manifest.get("model_files")
    if not isinstance(files, dict) or not files:
        return False
    return _files_intact(files, model_dir(paths))


def _files_intact(files, target: Path) -> bool:
    """统一 SHA256 证据校验；尺寸清单本身不再构成可信完整性证据。"""
    from .model_worker import verify_model_manifest
    return verify_model_manifest(target, files)


def _embed_manifest_files_intact(manifest: dict, paths: "AppPaths | None" = None) -> bool:
    """嵌入模型的同款完整性核对。双模型同批安装之前的旧 manifest 没有 ``embed_files``
    → 嵌入侧一律不就绪（fail-closed）；重跑安装器补齐后自动转就绪。"""
    return _files_intact(manifest.get("embed_files"), embed_model_dir(paths))


def external_embed_ready(paths: "AppPaths | None" = None) -> bool:
    """嵌入侧就绪闸（与 external_runtime_ready 同族）：运行时要件 + 嵌入模型文件与
    manifest 完整性。嵌入侧不就绪不影响重排侧就绪判定。"""
    resolved = paths or get_app_paths()
    manifest = read_ready_manifest(resolved)
    if not manifest:
        return False
    if not (runtime_python(resolved).is_file() and worker_script(resolved).is_file()):
        return False
    if not model_files_ready(embed_model_dir(resolved)):
        return False
    return _embed_manifest_files_intact(manifest, resolved)


def external_runtime_ready(paths: "AppPaths | None" = None) -> bool:
    resolved = paths or get_app_paths()
    manifest = read_ready_manifest(resolved)
    if not manifest:
        return False
    if not (runtime_python(resolved).is_file() and worker_script(resolved).is_file()):
        return False
    if not model_files_ready(model_dir(resolved)):
        return False
    return _manifest_files_intact(manifest, resolved)


class _WorkerProcess:
    """常驻 worker 子进程共享基座：启动/读写/关闭一套实现，子类只声明就绪闸与服务命令。"""

    def __init__(self, paths: "AppPaths | None" = None) -> None:
        self.paths = paths or get_app_paths()
        self.process: "subprocess.Popen[str] | None" = None
        self.responses: "queue.Queue[dict]" = queue.Queue()
        self.stderr_lines: "queue.Queue[str]" = queue.Queue(maxsize=40)
        self.lock = threading.Lock()
        from .model_worker import model_generation
        self.model_generation = model_generation(self._model_dir())

    def _model_dir(self) -> Path:
        raise NotImplementedError

    def _generation(self):
        from .model_worker import model_generation
        path = ready_manifest_path(self.paths)
        try:
            stat = path.stat()
            ready = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino, stat.st_dev)
        except OSError:
            ready = None
        return ready, model_generation(self._model_dir())

    def _reader(self, stream, responses=None) -> None:
        responses = self.responses if responses is None else responses
        for line in stream:
            try:
                payload = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                responses.put(payload)

    def _stderr_reader(self, stream) -> None:
        for line in stream:
            text = str(line).strip()[:500]
            if not text:
                continue
            if self.stderr_lines.full():
                try:
                    self.stderr_lines.get_nowait()
                except queue.Empty:
                    pass
            self.stderr_lines.put(text)

    def _ready(self) -> bool:
        raise NotImplementedError

    def _serve_command(self) -> "list[str]":
        raise NotImplementedError

    def _start(self) -> None:
        generation = self._generation()
        if generation[1] != self.model_generation:
            self.close()
            raise RuntimeError("encoder_generation_changed")
        if not self._ready():
            raise RuntimeError("model_runtime_not_ready")
        if self.process is not None and self.process.poll() is None:
            if getattr(self, "_loaded_generation", None) == generation:
                return
            self.close()
        self.responses = queue.Queue()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self._serve_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(target=self._reader, args=(self.process.stdout, self.responses), daemon=True).start()
        threading.Thread(target=self._stderr_reader, args=(self.process.stderr,), daemon=True).start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("model_worker_start_failed")
            try:
                payload = self.responses.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if payload.get("ready") is True:
                if generation != self._generation():
                    self.close()
                    raise RuntimeError("encoder_generation_changed")
                self._loaded_generation = generation
                return
        self.close()
        raise TimeoutError("model_worker_start_timeout")

    def _roundtrip(self, request: dict) -> dict:
        with self.lock:
            self._start()
            assert self.process is not None and self.process.stdin is not None
            request_id = str(request.get("id") or "")
            self.process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + SCORE_TIMEOUT_S
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("model_worker_exited")
                try:
                    payload = self.responses.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if payload.get("id") != request_id:
                                                                          
                                                                     
                                                           
                    if payload.get("ok") is False:
                        raise RuntimeError("model_worker_request_rejected")
                    continue
                if payload.get("ok") is not True:
                    raise RuntimeError("model_worker_bad_response")
                if self._loaded_generation != self._generation():
                    self.close()
                    raise RuntimeError("encoder_generation_changed")
                return payload
            self.close()
            raise TimeoutError("model_worker_score_timeout")

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:                
            try:
                process.kill()
            except Exception:
                pass


class ExternalCrossScorer(_WorkerProcess):
    """cross_encoder 打分客户端：request ``{"pairs"}`` → response ``{"scores"}``。"""

    def _ready(self) -> bool:
        return external_runtime_ready(self.paths)

    def _model_dir(self) -> Path:
        return model_dir(self.paths)

    def _serve_command(self) -> "list[str]":
        return [str(runtime_python(self.paths)), str(worker_script(self.paths)),
                "--serve", str(model_dir(self.paths))]

    def __call__(self, pairs: "Sequence[tuple[str, str]]") -> "list[float]":
        prepared = [[str(a)[:MAX_QUERY_CHARS], str(b)[:MAX_DOCUMENT_CHARS]] for a, b in pairs]
        scores: "list[float]" = []
                                                                   
        for chunk in _split_by_budget(prepared[:MAX_PAIRS]):
            payload = self._roundtrip({"id": uuid.uuid4().hex, "pairs": chunk})
            part = payload.get("scores")
            if not isinstance(part, list) or len(part) != len(chunk):
                raise RuntimeError("model_worker_bad_response")
            scores.extend(float(value) for value in part)
        return scores


class ExternalEmbedder(_WorkerProcess):
    """dense 嵌入客户端（``--embed`` worker）：request ``{"texts"}`` → response ``{"vectors"}``。

    文本截断与 vector_index.MAX_TEXT_CHARS / recall_api 的 12000 防御逐字同源。"""

    MAX_TEXT_CHARS = 12000

    def _ready(self) -> bool:
        return external_embed_ready(self.paths)

    def _model_dir(self) -> Path:
        return embed_model_dir(self.paths)

    def _serve_command(self) -> "list[str]":
        return [str(runtime_python(self.paths)), str(worker_script(self.paths)),
                "--serve", str(embed_model_dir(self.paths)), "--embed"]

    def __call__(self, texts: "Sequence[str]") -> "list[list[float]]":
        batch = [str(t)[: self.MAX_TEXT_CHARS] for t in texts]
        out: "list[list[float]]" = []
                                                             
                                                       
        for chunk in _split_by_budget(batch):
            payload = self._roundtrip({"id": uuid.uuid4().hex, "texts": chunk})
            vectors = payload.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(chunk):
                raise RuntimeError("model_worker_bad_response")
            out.extend([float(x) for x in vec] for vec in vectors)
        return out


_WORKERS: "list[_WorkerProcess]" = []


def external_cross_scorer(paths: "AppPaths | None" = None) -> ExternalCrossScorer:
    scorer = ExternalCrossScorer(paths)
    _WORKERS.append(scorer)
    return scorer


def external_embedder(paths: "AppPaths | None" = None) -> ExternalEmbedder:
    embedder = ExternalEmbedder(paths)
    _WORKERS.append(embedder)
    return embedder


@atexit.register
def _close_workers() -> None:
    for worker in list(_WORKERS):
        worker.close()
