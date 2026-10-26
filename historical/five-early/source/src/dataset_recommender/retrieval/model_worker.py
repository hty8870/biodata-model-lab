                       
"""本地语义模型隔离 worker。

本文件会被 PyInstaller 作为普通 data 复制到 ``_internal/tools/model_worker.py``，
再由 data_root/model-runtime/venv 的独立 Python 执行。它刻意不 import 主项目，避免
在线安装的 torch/transformers 依赖进入 FastAPI/pywebview 主进程。

两种模式（``--embed`` 切换为稠密嵌入模型）：
- ``--download DIR [--embed]``：从固定 ModelScope id 下载，失败回退固定 HuggingFace id；
- ``--serve DIR [--embed]``：加载本地 CrossEncoder（或 SentenceTransformer 嵌入器），
  以 JSONL stdin/stdout 提供有界打分/嵌入服务。
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

                                                                              
                                                          
MODEL_ID = "BAAI/" + "bge-reranker-v2-m3"
                                                                                  
                                          
EMBED_MODEL_ID = "sentence-transformers/" + "paraphrase-multilingual-MiniLM-L12-v2"
IGNORE_PATTERNS = ["*.onnx", "onnx/*", "*openvino*", "*.gguf", "*imatrix*", "*.h5", "*.msgpack"]
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_PAIRS = 5000
MAX_QUERY_CHARS = 4096
MAX_DOCUMENT_CHARS = 8192
MAX_TEXTS = 512
                                                                                
MAX_EMBED_TEXT_CHARS = 12000
                                                
                                       
COMPLETION_MARKER = ".download-complete.json"


_INTEGRITY_CACHE = OrderedDict()
_INTEGRITY_LOCK = threading.Lock()


_MODEL_FILE_PATTERNS = (
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.model", "*.tiktoken",
    "*.index.json", "config.json", "*_config.json", "config_*.json", "modules.json",
    "tokenizer*.json", "special_tokens_map.json", "added_tokens.json", "sentence_*.json",
    "vocab.*", "merges.txt", "sentencepiece.*",
)


def _managed_model_file(name):
    return any(fnmatch.fnmatchcase(Path(name).name.lower(), pattern) for pattern in _MODEL_FILE_PATTERNS)


def _completion_evidence(target):
    try:
        value = json.loads((Path(target) / COMPLETION_MARKER).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _model_files(target, *, names=None, include_unmanaged=False):
    root = Path(target).resolve()
    inventory = _completion_evidence(root) if names is None else names
    return {path.relative_to(root).as_posix(): path for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in {COMPLETION_MARKER, COMPLETION_MARKER + ".tmp"}
            and path.resolve().is_relative_to(root)
            and (include_unmanaged or _managed_model_file(path.name)
                 or path.relative_to(root).as_posix() in inventory)}


def model_generation(target):
    """可观察模型代际；Windows 缺可信变更 token 时每次完整 SHA，不能只信 stat。"""
    root = Path(target).resolve()
    if not root.is_dir():
        return None
    try:
        marker = root / COMPLETION_MARKER
        return (str(root), _weight_sig(marker) if marker.is_file() else None,
                tuple((name, _weight_sig(path)) for name, path in _model_files(root).items()))
    except OSError:
        raise RuntimeError("model_changed_during_generation")


def _file_stat_signature(path):
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino, stat.st_dev)


def _parse_usn_record(raw):
    if not 8 <= len(raw) <= 4096:
        return None
    length = int.from_bytes(raw[:4], "little")
    major = int.from_bytes(raw[4:6], "little")
    minor = int.from_bytes(raw[6:8], "little")
    if length != len(raw) or length % 8 or minor != 0:
        return None
    if major == 2:
        header_size, reference_size, usn_offset = 60, 8, 24
    elif major == 3:
        header_size, reference_size, usn_offset = 76, 16, 40
    else:
        return None
    if length < header_size:
        return None
    reference = int.from_bytes(raw[8:8 + reference_size], "little")
    usn = int.from_bytes(raw[usn_offset:usn_offset + 8], "little", signed=True)
    name_size = int.from_bytes(raw[header_size - 4:header_size - 2], "little")
    name_offset = int.from_bytes(raw[header_size - 2:header_size], "little")
    if (reference <= 0 or usn <= 0 or name_size <= 0 or name_size % 2
            or name_offset < header_size or name_offset % 2 or name_offset + name_size > length):
        return None
    return ("usn", major, reference, usn)


def _windows_file_api():
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    api.CreateFileW.restype = wintypes.HANDLE
    api.DeviceIoControl.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
    api.DeviceIoControl.restype = wintypes.BOOL
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    api.CloseHandle.restype = wintypes.BOOL
    return api


def _windows_file_change_token(path):
    import ctypes
    from ctypes import wintypes
    try:
        api = _windows_file_api()
        handle = api.CreateFileW(str(path), 0x80000000, 7, None, 3, 0, None)
        if handle in (None, 0, ctypes.c_void_p(-1).value):
            return None
        try:
            versions = (wintypes.WORD * 2)(2, 3)
            output = ctypes.create_string_buffer(4096)
            returned = wintypes.DWORD()
            if not api.DeviceIoControl(handle, 0x900EB, versions, ctypes.sizeof(versions),
                                       output, len(output), ctypes.byref(returned), None):
                return None
            if not 8 <= returned.value <= len(output):
                return None
            return _parse_usn_record(output.raw[:returned.value])
        finally:
            if not api.CloseHandle(handle):
                raise OSError("model_change_token_handle_close_failed")
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def _weight_sig(path):
    signature = _file_stat_signature(path)
    if os.name != "nt":
        return signature
    token = _windows_file_change_token(path)
    if token is None:
        token = ("sha256", _weight_sha(path))
        if signature != _file_stat_signature(path):
            raise OSError("model_changed_during_hash")
    return signature + (token,)


def _weight_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_manifest(target: Path, *, integrity=True, include_unmanaged=False) -> dict:
    manifest = {}
    for name, path in _model_files(target, include_unmanaged=include_unmanaged).items():
        before = _weight_sig(path)
        if integrity:
            sha = _weight_sha(path)
            if before != _weight_sig(path):
                raise OSError("model_changed_during_hash")
            manifest[name] = {"size": before[2], "sha256": sha}
        else:
            manifest[name] = before[2]
    return manifest


def _model_layout_ready(target, sizes):
    if sizes.get("config.json", 0) <= 0:
        return False
    if not any(sizes.get(name, 0) > 0 for name in ("tokenizer.json", "tokenizer_config.json")):
        return False
    if not any(name.endswith((".safetensors", ".bin")) and size > 0 for name, size in sizes.items()):
        return False
    root = Path(target).resolve()
    for name in sizes:
        if not name.endswith((".safetensors.index.json", ".bin.index.json")):
            continue
        weight_map = json.loads((root / name).read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard.endswith((".safetensors", ".bin")):
                return False
            path = (root / name).parent / shard
            if not path.resolve().is_relative_to(root):
                return False
            if sizes.get(path.relative_to(root).as_posix(), 0) <= 0:
                return False
    return True


def verify_model_manifest(target, evidence):
    if not isinstance(evidence, dict) or not evidence:
        return False
    try:
        expected = []
        for name, item in sorted(evidence.items()):
            if (not isinstance(name, str) or not isinstance(item, dict) or not isinstance(item.get("size"), int)
                    or isinstance(item["size"], bool) or item["size"] < 0
                    or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in item["sha256"])):
                return False
            expected.append((name, item["size"], item["sha256"]))
        files = _model_files(target, names=evidence)
        if set(files) != set(evidence):
            return False
        if not _model_layout_ready(target, {name: size for name, size, _ in expected}):
            return False
        signature = tuple((name, _weight_sig(path)) for name, path in sorted(files.items()))
        if any(sig[2] != evidence[name]["size"] for name, sig in signature):
            return False
        key = (str(Path(target).resolve()), tuple(expected), signature)
        with _INTEGRITY_LOCK:
            if key in _INTEGRITY_CACHE:
                _INTEGRITY_CACHE.move_to_end(key)
                return True
        if any(_weight_sha(files[name]) != sha for name, _, sha in expected):
            return False
        if signature != tuple((name, _weight_sig(path)) for name, path in sorted(files.items())):
            return False
        with _INTEGRITY_LOCK:
            _INTEGRITY_CACHE[key] = True
            while len(_INTEGRITY_CACHE) > 16:
                _INTEGRITY_CACHE.popitem(last=False)
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def model_files_ready(target: Path) -> bool:
    """配置、分词器及完整权重分片的尺寸预闸；仍不构成 SHA 完成证据。"""
    try:
        return _model_layout_ready(target, model_file_manifest(target, integrity=False))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _copy_snapshot(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _completion_marker_path(target: Path) -> Path:
    return Path(target) / COMPLETION_MARKER


def completed_model_manifest(target: Path) -> dict:
    recorded = _completion_evidence(target)
    return recorded if verify_model_manifest(target, recorded) else {}


def _completion_valid(target: Path) -> bool:
    return bool(completed_model_manifest(target))


                                                                                    
                                                            
                                                        
_MARKER_PUBLISH_RETRIES = 5
_MARKER_PUBLISH_RETRY_SLEEP_S = 0.05


def _replace_with_retry(tmp: Path, marker: Path) -> None:
    """有界重试的原子替换：只在 Windows 句柄竞态（PermissionError）上重试，绝不静默伪成功。"""
    for attempt in range(_MARKER_PUBLISH_RETRIES):
        try:
            os.replace(tmp, marker)
            return
        except PermissionError:
            if attempt == _MARKER_PUBLISH_RETRIES - 1:
                raise
            time.sleep(_MARKER_PUBLISH_RETRY_SLEEP_S * (attempt + 1))


def _write_completion_marker(target: Path, evidence=None) -> None:
    marker = _completion_marker_path(target)
    fd, name = tempfile.mkstemp(prefix=marker.name + ".", suffix=".tmp", dir=target)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(model_file_manifest(target) if evidence is None else evidence,
                      handle, ensure_ascii=False, separators=(",", ":"))
        _replace_with_retry(tmp, marker)
    finally:
        tmp.unlink(missing_ok=True)


def _publish_staging(staging: Path, target: Path, evidence) -> None:
    """备份旧管理文件与本次覆盖的同名文件；故障逐项恢复，不退役其他用户文件。"""
    target.mkdir(parents=True, exist_ok=True)
    old = {name: path for name, path in _model_files(target, include_unmanaged=True).items()
           if _managed_model_file(name) or name in evidence}
    marker = _completion_marker_path(target)
    if marker.is_file():
        old[COMPLETION_MARKER] = marker
    names = set(old) | set(evidence) | {COMPLETION_MARKER}
    for name in names:
        path = target / name
        if path.is_dir() or not path.resolve().is_relative_to(target.resolve()):
            raise OSError("model_publication_path_conflict")
    backup = Path(tempfile.mkdtemp(prefix="." + target.name + ".rollback-", dir=target.parent))
    keep_backup = False
    try:
        for name, path in old.items():
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, saved)
        for name in evidence:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, dest)
        if not verify_model_manifest(target, evidence):
            raise OSError("model_publication_integrity_failed")
        _write_completion_marker(target, evidence)
        if not _completion_valid(target):
            raise OSError("model_completion_integrity_failed")
    except BaseException as primary:
        failures = []
        for name in sorted(names, reverse=True):
            try:
                dest, saved = target / name, backup / name
                if saved.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved, dest)
                elif name not in old:
                    dest.unlink(missing_ok=True)
            except OSError:
                failures.append(name)
        if failures:
            keep_backup = True
            raise OSError("model_publication_restore_incomplete: " + ", ".join(failures)) from primary
        raise
    finally:
        if not keep_backup:
            shutil.rmtree(backup)


def _download_via_mirrors(staging: Path, model_id: str) -> bool:
    """每个镜像各用干净 staging，失败文件不得混入另一个镜像的完成文件集。"""
    try:
        from modelscope import snapshot_download as modelscope_download

        print(json.dumps({"stage": "model", "message": "正在从 ModelScope 下载模型权重…"}, ensure_ascii=False), flush=True)
        source = Path(modelscope_download(model_id, ignore_file_pattern=IGNORE_PATTERNS))
        _copy_snapshot(source, staging)
        if model_files_ready(staging):
            print(json.dumps({"stage": "model", "message": "ModelScope 模型下载完成。"}, ensure_ascii=False), flush=True)
            return True
        print("ModelScope 返回目录缺少完整权重，尝试 HuggingFace。", file=sys.stderr, flush=True)
    except Exception as exc:                                 
        print(f"ModelScope 下载失败（{type(exc).__name__}），尝试 HuggingFace。", file=sys.stderr, flush=True)

    shutil.rmtree(staging)
    staging.mkdir()
    try:
        from huggingface_hub import snapshot_download as huggingface_download

        print(json.dumps({"stage": "model", "message": "正在从 HuggingFace 下载模型权重…"}, ensure_ascii=False), flush=True)
        huggingface_download(repo_id=model_id, local_dir=str(staging), ignore_patterns=IGNORE_PATTERNS)
        if model_files_ready(staging):
            print(json.dumps({"stage": "model", "message": "HuggingFace 模型下载完成。"}, ensure_ascii=False), flush=True)
            return True
        print("HuggingFace 下载完成但模型文件不完整。", file=sys.stderr, flush=True)
    except Exception as exc:                
        print(f"HuggingFace 下载失败（{type(exc).__name__}）。", file=sys.stderr, flush=True)
    return False


def download_model(target: Path, model_id: str = MODEL_ID) -> bool:
    """临时完整文件集 → SHA 校验 → 可恢复发布；成功标记仅来自本次 staging。"""
    target = Path(target)
    if _completion_valid(target):
        print(json.dumps({"stage": "model", "message": "模型已经就绪，无需重复下载。"}, ensure_ascii=False), flush=True)
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + target.name + ".incoming-", dir=target.parent))
    try:
        if not _download_via_mirrors(staging, model_id) or not model_files_ready(staging):
            return False
        evidence = model_file_manifest(staging, include_unmanaged=True)
        if not verify_model_manifest(staging, evidence):
            return False
        _publish_staging(staging, target, evidence)
        return True
    finally:
        shutil.rmtree(staging)


def _valid_pairs(value: Any) -> "list[tuple[str, str]] | None":
    if not isinstance(value, list) or not value or len(value) > MAX_PAIRS:
        return None
    out: "list[tuple[str, str]]" = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            return None
        query, document = row
        if not isinstance(query, str) or not isinstance(document, str):
            return None
        if len(query) > MAX_QUERY_CHARS or len(document) > MAX_DOCUMENT_CHARS:
            return None
        out.append((query, document))
    return out


def _valid_texts(value: Any) -> "list[str] | None":
    if not isinstance(value, list) or not value or len(value) > MAX_TEXTS:
        return None
    out: "list[str]" = []
    for text in value:
        if not isinstance(text, str) or len(text) > MAX_EMBED_TEXT_CHARS:
            return None
        out.append(text)
    return out


def serve(model_dir: Path, *, embed: bool = False) -> int:
    if not model_files_ready(model_dir):
        print("模型文件不完整。", file=sys.stderr)
        return 2
    model_id = EMBED_MODEL_ID if embed else MODEL_ID
    try:
        if embed:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(str(model_dir), local_files_only=True)
            except TypeError:
                model = SentenceTransformer(str(model_dir))
        else:
            from sentence_transformers import CrossEncoder

            try:
                model = CrossEncoder(str(model_dir), max_length=512, local_files_only=True)
            except TypeError:
                model = CrossEncoder(str(model_dir), max_length=512)
    except Exception as exc:                
        print(f"模型加载失败：{type(exc).__name__}", file=sys.stderr)
        return 3

    print(json.dumps({"ready": True, "model": model_id}), flush=True)
    for raw in sys.stdin.buffer:
        if len(raw) > MAX_REQUEST_BYTES:
            print(json.dumps({"id": "", "ok": False, "error": "request_too_large"}), flush=True)
            continue
        try:
            request = json.loads(raw.decode("utf-8"))
            request_id = str(request.get("id", ""))[:96]
            if embed:
                texts = _valid_texts(request.get("texts"))
                if not request_id or texts is None:
                    raise ValueError("bad_request")
                vectors = model.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                )
                payload = {"id": request_id, "ok": True,
                           "vectors": [[float(x) for x in vec] for vec in vectors]}
            else:
                pairs = _valid_pairs(request.get("pairs"))
                if not request_id or pairs is None:
                    raise ValueError("bad_request")
                scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
                payload = {"id": request_id, "ok": True, "scores": [float(value) for value in scores]}
        except Exception as exc:                                       
            payload = {"id": str(locals().get("request_id", "")), "ok": False, "error": type(exc).__name__}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="BioData Agent 本地模型隔离 worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", metavar="DIR")
    group.add_argument("--serve", metavar="DIR")
    parser.add_argument("--embed", action="store_true",
                        help="使用稠密嵌入模型（缺省为 cross-encoder 重排器）")
    args = parser.parse_args(argv)
    model_id = EMBED_MODEL_ID if args.embed else MODEL_ID
    if args.download:
        return 0 if download_model(Path(args.download), model_id=model_id) else 1
    return serve(Path(args.serve), embed=bool(args.embed))


if __name__ == "__main__":
    raise SystemExit(main())
