                       
"""隔离本地模型运行时的路径与 JSONL 客户端。

在线安装的 torch/transformers 全部住 data_root/model-runtime/venv，主 FastAPI 进程
不修改 sys.path。打分通过该 venv 的常驻 worker 进行，依赖冲突或 worker 崩溃只会让
vector_recall / 关系语义层回退规则顺序。

关系任务与 cross/embed 共用同一个 ``_WorkerProcess`` 启动、串行锁、超时、关闭和世代
契约；模型只来自服务端显式配置的受信 bundle（``BIODATA_RELATION_BUNDLE_DIR`` +
``BIODATA_RELATION_BUNDLE_SHA256``），不接受 HTTP 传入的模型路径。
"""
from __future__ import annotations

import atexit
import json
import math
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
    RELATION_CLASS_COUNT,
    model_files_ready,
    relation_bundle_ready,
    relation_generation,
)
from .vector_recall import DEFAULT_CROSS_ENCODER_MODEL, DEFAULT_EMBEDDING_MODEL

READY_SCHEMA = "biodata-model-runtime/v1"
STARTUP_TIMEOUT_S = 300.0
SCORE_TIMEOUT_S = 180.0
                                         
                                                              
                                    
REQUEST_BYTES_BUDGET = MAX_REQUEST_BYTES - 64 * 1024

                                                              
MODEL_RUNTIME_ROOT_ENV = "BIODATA_MODEL_RUNTIME_ROOT"
                                          
RELATION_BUNDLE_DIR_ENV = "BIODATA_RELATION_BUNDLE_DIR"
RELATION_BUNDLE_SHA_ENV = "BIODATA_RELATION_BUNDLE_SHA256"


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
    override = os.environ.get(MODEL_RUNTIME_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return (paths or get_app_paths()).data_root / "model-runtime"


def _venv_python_parts() -> "tuple[str, ...]":
    if os.name == "nt":
        return ("venv", "Scripts", "python.exe")
    return ("venv", "bin", "python")


def runtime_python(paths: "AppPaths | None" = None) -> Path:
    """Windows 保持 ``venv/Scripts/python.exe``；POSIX 使用 ``venv/bin/python``。"""
    return runtime_root(paths).joinpath(*_venv_python_parts())


def relation_bundle_root() -> "Path | None":
    value = os.environ.get(RELATION_BUNDLE_DIR_ENV, "").strip()
    return Path(value).expanduser() if value else None


def relation_bundle_pin() -> str:
    return os.environ.get(RELATION_BUNDLE_SHA_ENV, "").strip().lower()


def relation_backend_configured() -> bool:
    """是否显式配置了关系 bundle；缺省未配置时关系能力保持关闭。

    父层用它保留「配置了但不可用」的回退原因：目录已给却缺 SHA/venv 时仍返回 True，
    真正失败在 ``score_pairs`` 以 ``RuntimeError`` 显式暴露，绝不伪装成未配置。
    """
    return relation_bundle_root() is not None


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

    GENERATION_CHANGED = "encoder_generation_changed"

    def __init__(self, paths: "AppPaths | None" = None) -> None:
        self.paths = paths or get_app_paths()
        self.process: "subprocess.Popen[str] | None" = None
        self.responses: "queue.Queue[dict]" = queue.Queue()
        self.stderr_lines: "queue.Queue[str]" = queue.Queue(maxsize=40)
        self.lock = threading.Lock()
        self.model_generation = self._initial_generation()

    def _model_dir(self) -> Path:
        raise NotImplementedError

    def _initial_generation(self):
        from .model_worker import model_generation
        return model_generation(self._model_dir())

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
            raise RuntimeError(self.GENERATION_CHANGED)
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
                    raise RuntimeError(self.GENERATION_CHANGED)
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
                    raise RuntimeError(self.GENERATION_CHANGED)
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


def _valid_relation_classes(value: object) -> bool:
    return (isinstance(value, list) and len(value) == RELATION_CLASS_COUNT
            and all(isinstance(item, str) and item for item in value)
            and len(set(value)) == len(value))


def _normalize_sha256(value: object) -> "str | None":
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        return None
    return text


class ExternalRelationScorer(_WorkerProcess):
    """关系 bundle 打分客户端：request ``{"pairs"}`` → 五叶 classes/probabilities/complete。"""

    GENERATION_CHANGED = "relation_generation_changed"

    def __init__(self, paths: "AppPaths | None" = None, *, bundle_root=None,
                 bundle_sha256=None) -> None:
        root = bundle_root if bundle_root is not None else relation_bundle_root()
        if root is None:
            raise ValueError("relation_bundle_not_configured")
        try:
            self.bundle_root = Path(root).expanduser().resolve()
        except OSError:
            self.bundle_root = Path(root).expanduser()
        self.bundle_sha256 = (str(bundle_sha256).strip().lower()
                              if bundle_sha256 is not None else relation_bundle_pin())
        self.identity = None
        super().__init__(paths)

    def _model_dir(self) -> Path:
        return self.bundle_root

    def _initial_generation(self):
        return relation_generation(self.bundle_root, self.bundle_sha256)

    def _generation(self):
        return (None, relation_generation(self.bundle_root, self.bundle_sha256))

    def _ready(self) -> bool:
        return (runtime_python(self.paths).is_file() and worker_script(self.paths).is_file()
                and relation_bundle_ready(self.bundle_root, self.bundle_sha256))

    def _serve_command(self) -> "list[str]":
        return [str(runtime_python(self.paths)), str(worker_script(self.paths)),
                "--serve", str(self.bundle_root), "--relation",
                "--bundle-sha256", self.bundle_sha256]

    def score_pairs(self, pairs: "list[dict]") -> dict:
        """输入纯语义 pair；输出只含父层消费的五个字段，顺序与输入一致。"""
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("relation_pairs_empty")
        if len(pairs) > MAX_PAIRS:
            raise ValueError("relation_pairs_too_many")
        if any(not isinstance(pair, dict) for pair in pairs):
            raise ValueError("relation_pairs_invalid")
        try:
            chunks = _split_by_budget(pairs)
        except TypeError as exc:
            raise ValueError("relation_pairs_invalid") from exc

        classes = None
        model_id = None
        meta_sha256 = None
        probabilities: "list[list[float]]" = []
        input_complete: "list[bool]" = []
        for chunk in chunks:
            payload = self._roundtrip({"id": uuid.uuid4().hex, "pairs": chunk})
            part_classes = payload.get("classes")
            part_probabilities = payload.get("probabilities")
            part_complete = payload.get("input_complete")
            part_model = payload.get("model_id")
            part_meta = _normalize_sha256(payload.get("meta_sha256"))
            if not _valid_relation_classes(part_classes):
                raise RuntimeError("model_worker_bad_response")
            if (not isinstance(part_probabilities, list) or len(part_probabilities) != len(chunk)):
                raise RuntimeError("model_worker_bad_response")
            if (not isinstance(part_complete, list) or len(part_complete) != len(chunk)
                    or any(type(value) is not bool for value in part_complete)):
                raise RuntimeError("model_worker_bad_response")
            if (not isinstance(part_model, str) or not part_model or len(part_model) > 256
                    or part_meta is None):
                raise RuntimeError("model_worker_bad_response")
            if classes is None:
                classes, model_id, meta_sha256 = list(part_classes), part_model, part_meta
            elif part_classes != classes or part_model != model_id or part_meta != meta_sha256:
                raise RuntimeError("model_worker_bad_response")
            for row in part_probabilities:
                if not isinstance(row, list) or len(row) != len(classes):
                    raise RuntimeError("model_worker_bad_response")
                checked: "list[float]" = []
                for value in row:
                    if type(value) is bool or not isinstance(value, (int, float)):
                        raise RuntimeError("model_worker_bad_response")
                    number = float(value)
                    if not math.isfinite(number) or number < 0.0 or number > 1.0:
                        raise RuntimeError("model_worker_bad_response")
                    checked.append(number)
                probabilities.append(checked)
            input_complete.extend(part_complete)
        return {"classes": classes, "probabilities": probabilities,
                "input_complete": input_complete, "model_id": model_id,
                "meta_sha256": meta_sha256}


_WORKERS: "list[_WorkerProcess]" = []
_WORKERS_LOCK = threading.Lock()
_RELATION_LOCK = threading.Lock()
_RELATION_SCORER: "ExternalRelationScorer | None" = None


def external_cross_scorer(paths: "AppPaths | None" = None) -> ExternalCrossScorer:
    scorer = ExternalCrossScorer(paths)
    with _WORKERS_LOCK:
        _WORKERS.append(scorer)
    return scorer


def external_embedder(paths: "AppPaths | None" = None) -> ExternalEmbedder:
    embedder = ExternalEmbedder(paths)
    with _WORKERS_LOCK:
        _WORKERS.append(embedder)
    return embedder


def external_relation_scorer(paths: "AppPaths | None" = None) -> "ExternalRelationScorer | None":
    """关系打分单例；未配置返回 None，配置变化关闭旧实例并换新。

    bundle 身份 = 显式路径 + bundle.json pin + manifest 代际；每次调用只做清单/stat
    核对，不会逐请求重启子进程，也不会在 ``_WORKERS`` 里累积旧实例。
    """
    global _RELATION_SCORER
    root = relation_bundle_root()
    if root is None:
        with _RELATION_LOCK:
            current = _RELATION_SCORER
            if current is not None:
                current.close()
                with _WORKERS_LOCK:
                    try:
                        _WORKERS.remove(current)
                    except ValueError:
                        pass
                _RELATION_SCORER = None
        return None
    pin = relation_bundle_pin()
    identity = (str(root), pin, str(runtime_root(paths)), str(worker_script(paths)),
                relation_generation(root, pin))
    with _RELATION_LOCK:
        current = _RELATION_SCORER
        if current is not None and getattr(current, "identity", None) == identity:
            return current
        if current is not None:
            current.close()
            with _WORKERS_LOCK:
                try:
                    _WORKERS.remove(current)
                except ValueError:
                    pass
        scorer = ExternalRelationScorer(paths, bundle_root=root, bundle_sha256=pin)
        scorer.identity = identity
        with _WORKERS_LOCK:
            _WORKERS.append(scorer)
        _RELATION_SCORER = scorer
        return scorer


@atexit.register
def _close_workers() -> None:
    with _WORKERS_LOCK:
        workers = list(_WORKERS)
    for worker in workers:
        worker.close()
