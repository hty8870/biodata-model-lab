"""语料级向量索引（vector 召回后端的本地数据源，确定性可复现）。

与 recall_api 的 API 语料向量文件同构的**本地**对应物：把全语料（基础 + 外部库）的候选
打分文本用本地 MiniLM 嵌成向量，持久化到 userdata_dir/vector_index/；查询时按
「uid + 文本指纹」双校验命中取用，未覆盖/指纹漂移的条目现场补嵌（**仅内存、不持久化**，
与 recall_api.api_dense_vectors 同一契约）。

设计约束（与 vector_recall / recall_api 同源，defense-in-depth）：
- 文本序列化唯一真源是 vector_recall._candidate_text（外加与 API 侧逐字同源的 12000 截断），
  索引条目与查询侧打分文本指纹一致；模板漂移由 sha 检出并按条目重建。
- 首用构建、按条目指纹**增量**重建：model/dimensions 不符或坏档 → 整档重建；语料增删/文本
  变化 → 只重嵌受影响条目、剔除已消失的 uid。
- 任何失败（模型不可用/嵌入畸形/写盘异常）→ None：调用方 fail-closed 回退规则序，绝不抛错、
  绝不阻塞请求、绝不联网。
- 官方评测不传 recall 参数 → 结构性不触碰本模块；本模块导入期不引入任何重依赖
  （sentence-transformers 仅在真正建索引/补嵌时经 load_embedder 惰性加载）。
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path
from typing import Sequence

from .recall_api import candidate_text_sha
from .model_worker import MAX_TEXTS
from .retriever import RetrievedCandidate
from .candidate_text import candidate_text as _candidate_text
from .vector_recall import (
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    load_embedder,
    valid_vector,
)
from .warn_once import warn_once_prefixed as _warn_once_prefixed
from ..app.runtime_paths import get_app_paths

                                                                                              
EMBED_DIMENSIONS = 384
                                                                                    
MAX_TEXT_CHARS = 12000

_INDEX_LOCK = threading.Lock()
                                                       
_INDEX_CACHE: "dict[str, tuple[tuple, dict]]" = {}
                                                           
                                                              
                              
_EMBEDDER_CACHE: "list[tuple[tuple, Embedder]]" = []
                                      
                                       
_EMBEDDER_LOCK = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    _warn_once_prefixed("[vector_index]", key, message)


def index_path(paths=None) -> Path:
    """索引文件唯一位置：userdata_dir/vector_index/corpus_vectors.<model>.<dims>.json.gz。"""
    p = paths or get_app_paths()
    return (Path(p.userdata_dir) / "vector_index"
            / f"corpus_vectors.{DEFAULT_EMBEDDING_MODEL}.{EMBED_DIMENSIONS}.json.gz")


def _file_sig(path: Path) -> "tuple[int, int] | None":
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino, st.st_dev)
    except OSError:
        return None


def load_index(path: "Path | None" = None, *, expected_model=None) -> "dict | None":
    """读索引文件并做版本校验（model/dimensions 不匹配、坏档 → None，触发整档重建）。

    文件格式（gzip JSON，与 API 语料向量文件同构）：
      {"meta": {"model", "dimensions", "created_at", "count"},
       "vectors": {"<uid>": {"h": "<candidate_text_sha>", "v": [floats]}}}
    """
    p = Path(path) if path is not None else index_path()
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload.get("meta") or {}
        vectors = payload.get("vectors") or {}
        if (
            meta.get("model") != DEFAULT_EMBEDDING_MODEL
            or int(meta.get("dimensions") or 0) != EMBED_DIMENSIONS
            or not isinstance(vectors, dict)
            or (expected_model is not None and meta.get("model_state") is not None
                and meta["model_state"] != json.dumps(expected_model))
            or (expected_model is not None and meta.get("model_state") is None
                and any(part is not None for part in expected_model[2:]))
        ):
            return None
        return vectors
    except Exception:
        return None


def _record_text(record) -> str:
    """索引文本 = 查询侧打分文本（_candidate_text 唯一真源）+ API 侧同款 12000 截断。"""
    t = _candidate_text(RetrievedCandidate(record=record, score=0.0))
    return t[:MAX_TEXT_CHARS] if len(t) > MAX_TEXT_CHARS else t


def _encoder_matches(enc, model):
    return getattr(enc, "model_generation", model[-1]) == model[-1]


def _local_embedder() -> "Embedder | None":
    """本地优先、隔离运行时兜底；单飞正缓存绑定实际模型代际，换代即退休旧客户端。"""
    with _EMBEDDER_LOCK:
        try:
            model = _model_state()
            if _EMBEDDER_CACHE:
                captured, cached = _EMBEDDER_CACHE[0]
                if captured == model and _encoder_matches(cached, model):
                    return cached
                _EMBEDDER_CACHE.clear()
                close = getattr(cached, "close", None)
                if callable(close):
                    close()
            try:
                enc = load_embedder()
            except Exception:
                enc = None
            if enc is None:
                from .model_runtime import external_embed_ready, external_embedder
                if external_embed_ready():
                    enc = external_embedder()
            if enc is not None:
                if model != _model_state() or not _encoder_matches(enc, model):
                    close = getattr(enc, "close", None)
                    if callable(close):
                        close()
                    return None
                _EMBEDDER_CACHE.append((model, enc))
            return enc
        except Exception:
            return None


def get_embedder() -> "Embedder | None":
    """统一嵌入器获取入口（单一真源）：本地优先、隔离运行时兜底、进程内单例。
    消费方（索引构建/查询、语义兜底召回）一律走这里，不再各自 load_embedder。"""
    return _local_embedder()


def _load_corpus(paths=None) -> list:
    """语料装载唯一真源（corpus.load_full_corpus）：基础语料 + 全部外部库（shipped+user 双层）。"""
    from ..corpus.corpus import load_full_corpus
    p = paths or get_app_paths()
    root = Path(p.data_root)
    return load_full_corpus(root / "database" / "base", root)


def _write_index(path: Path, vectors: dict, *, commit_check=None, build_check=None, model=None) -> None:
    """序列化在读锁外；所有发布者共用文件锁，条件核对后仅原子替换持短读锁。"""
    import uuid
    from ..corpus.fs_utils import acquire_file_lock, release_file_lock
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = acquire_file_lock(path.with_name(path.name + ".lock"), timeout=1.0) if commit_check is None else None
    payload = {"meta": {"model": DEFAULT_EMBEDDING_MODEL, "dimensions": EMBED_DIMENSIONS,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "count": len(vectors),
                        "model_state": json.dumps(model if model is not None else _model_state())},
               "vectors": vectors}
    tmp = path.parent / (path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if build_check is not None and not build_check():
            raise RuntimeError("index_source_changed")
        with _INDEX_LOCK:
            if commit_check is not None and not commit_check():
                raise RuntimeError("index_generation_changed")
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
        if handle is not None:
            release_file_lock(handle)


def _embed_batches(enc: Embedder, texts: Sequence[str]) -> "list | None":
    """构建和查询补嵌共用 worker 上限；任一批失配则整组失败，不交付部分向量。"""
    fresh: list = []
    for start in range(0, len(texts), MAX_TEXTS):
        chunk = texts[start:start + MAX_TEXTS]
        part = enc(chunk)
        if not part or len(part) != len(chunk):
            _warn_once("embed_shape", "嵌入结果数量不符 —— vector 召回回退规则序。")
            return None
        fresh.extend(part)
    return fresh


_REQUESTS = {}


def _texts_for(records):
    texts = {}
    for record in records:
        raw = getattr(record, "raw", None)
        uid = str(raw.get("dataset_uid") or "") if isinstance(raw, dict) else ""
        if uid:
            texts[uid] = _record_text(record)
    return texts


def _model_state(paths=None):
    from .model_runtime import ready_manifest_path, embed_model_dir
    from .model_worker import model_generation
    resolved = paths or get_app_paths()
    return (DEFAULT_EMBEDDING_MODEL, EMBED_DIMENSIONS, _file_sig(ready_manifest_path(resolved)),
            model_generation(embed_model_dir(resolved)))


def _freeze_store(store):
    from types import MappingProxyType
    from .vector_recall import VectorSnapshot
    return MappingProxyType({uid: MappingProxyType({"h": entry["h"], "v": VectorSnapshot(entry["v"])})
                             for uid, entry in store.items()
                             if isinstance(entry, dict) and isinstance(entry.get("h"), str)
                             and valid_vector(entry.get("v"), EMBED_DIMENSIONS)})


def _owned_store(store):
    return {uid: {"h": entry["h"], "v": list(entry["v"])} for uid, entry in store.items()}


def _index_snapshot(path, model):
    sig = _file_sig(path)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(str(path))
        if cached is not None and cached[0] == (sig, model):
            return sig, cached[1]
    raw = load_index(path, expected_model=model) or {}
    snapshot = _freeze_store(raw)
    if sig != _file_sig(path):
        return None, None
    with _INDEX_LOCK:
        _INDEX_CACHE[str(path)] = ((sig, model), snapshot)
    return sig, snapshot


def _covered(store, texts):
    return all(uid in store and store[uid]["h"] == candidate_text_sha(text) for uid, text in texts.items())


def ensure_index(records=None, *, embedder: "Embedder | None" = None, paths=None,
                 scope: str = "full") -> "dict | None":
    """构建在短读锁外；按路径单飞，提交用 OS 文件锁及起点/代际/模型/语料条件校验。"""
    from ..corpus.fs_utils import path_lock, acquire_file_lock, release_file_lock
    view_only = str(scope).strip().lower() == "view"
    supplied = records
    try:
        enc = embedder if embedder is not None else _local_embedder()
        if enc is None:
            return None
        path = index_path(paths)
        from ..corpus.corpus import corpus_cache_generation
        resolved = paths or get_app_paths()
        def source_generation():
            return corpus_cache_generation(resolved.data_root / "database" / "base", resolved.data_root)
        generation = source_generation()
        source = records if records is not None else _load_corpus(paths)
        if not isinstance(source, (list, tuple)):
            source = tuple(source)
        texts = _texts_for(source)
        model = _model_state(paths)
        if not _encoder_matches(enc, model):
            return None
        signature = (model, generation, tuple(sorted((uid, candidate_text_sha(text)) for uid, text in texts.items())))
        with _INDEX_LOCK:
            if not view_only:
                _REQUESTS[str(path)] = signature
        sig, old = _index_snapshot(path, model)
        if old is None:
            return None
        if _covered(old, texts) and (view_only or set(old) == set(texts)):
            return _owned_store(old)
        lock = path_lock(path)
        if not lock.acquire(timeout=1.0):
            return None
        try:
            sig, old = _index_snapshot(path, model)
            if old is None:
                return None
            if _covered(old, texts) and (view_only or set(old) == set(texts)):
                return _owned_store(old)
            vectors = {uid: {"h": entry["h"], "v": list(entry["v"])} for uid, entry in old.items()
                       if uid in texts and entry["h"] == candidate_text_sha(texts[uid])}
            stale = [uid for uid in texts if uid not in vectors]
            fresh = _embed_batches(enc, [texts[uid] for uid in stale]) if stale else []
            if fresh is None:
                return None
            for uid, vector in zip(stale, fresh):
                values = [float(value) for value in vector]
                if not valid_vector(values, EMBED_DIMENSIONS):
                    return None
                vectors[uid] = {"h": candidate_text_sha(texts[uid]), "v": values}
            current_source = source if supplied is not None else _load_corpus(paths)
            current_texts = _texts_for(current_source)
            current = (_model_state(paths), source_generation(), tuple(sorted((uid, candidate_text_sha(text)) for uid, text in current_texts.items())))
            if signature != current or sig != _file_sig(path):
                return None
            if view_only:
                merged = _owned_store(old)
                merged.update(vectors)
                return merged
            snapshot = _freeze_store(vectors)
            handle = acquire_file_lock(path.with_name(path.name + ".lock"), timeout=1.0)
            try:
                def current_commit():
                    return (_REQUESTS.get(str(path)) == signature and sig == _file_sig(path)
                            and model == _model_state(paths) and generation == source_generation())
                def current_source_signature():
                    final_source = source if supplied is not None else _load_corpus(paths)
                    final_texts = _texts_for(final_source)
                    return signature[-1] == tuple(sorted((uid, candidate_text_sha(text)) for uid, text in final_texts.items()))
                _write_index(path, vectors, commit_check=current_commit, build_check=current_source_signature, model=model)
                with _INDEX_LOCK:
                    _INDEX_CACHE[str(path)] = ((_file_sig(path), model), snapshot)
                return _owned_store(snapshot)
            finally:
                release_file_lock(handle)
        finally:
            lock.release()
    except Exception as exc:
        _warn_once(f"ensure_exc::{type(exc).__name__}", "索引构建未提交，回退规则顺序。")
        return None


def index_dense_vectors(
    query: str,
    items: "Sequence[object]",
    *,
    embedder: "Embedder | None" = None,
    paths=None,
) -> "list[list[float]] | None":
    """vector 本地数据源：返回 [查询向量, *文档向量]（与 recall_api.api_dense_vectors 同契约）。

    文档向量按「uid + 文本指纹」双校验查语料索引；未覆盖/指纹漂移的条目（用户上传、模板
    更新）现场补嵌——**仅内存、不持久化**。任何环节失败 → None（调用方回退规则序，绝不错位）。"""
    try:
        enc = embedder if embedder is not None else _local_embedder()
        if enc is None:
            _warn_once("embedder_unavailable", "本地嵌入模型不可用 —— vector 召回回退规则序。")
            return None
        model = _model_state(paths)
        if not _encoder_matches(enc, model):
            return None
        sig, snapshot = _index_snapshot(index_path(paths), model)
        requested = _texts_for([getattr(c, "record", c) for c in items])
        store = snapshot if snapshot is not None and _covered(snapshot, requested) else ensure_index(embedder=enc, paths=paths)
        if store is None:
            return None
        texts = [_record_text(getattr(c, "record", c)) for c in items]
        doc_vecs: "list[list[float] | None]" = []
        missing: "list[int]" = []
        for i, cand in enumerate(items):
            raw = getattr(getattr(cand, "record", None), "raw", None)
            uid = str(raw.get("dataset_uid") or "") if isinstance(raw, dict) else ""
            entry = store.get(uid) if uid else None
            if (
                hasattr(entry, "get") and entry.get("h") == candidate_text_sha(texts[i])
                and valid_vector(entry.get("v"), EMBED_DIMENSIONS)
            ):
                doc_vecs.append(entry["v"])
            else:
                doc_vecs.append(None)
                missing.append(i)
        queue = [query] + [texts[i] for i in missing]
        fresh = _embed_batches(enc, queue)
        if fresh is None:
            return None
        qv = [float(x) for x in fresh[0]]
        if not valid_vector(qv, EMBED_DIMENSIONS):
            return None
        for pos, i in enumerate(missing):
            v = [float(x) for x in fresh[pos + 1]]
            if not valid_vector(v, EMBED_DIMENSIONS):
                return None
            doc_vecs[i] = v
        if model != _model_state(paths) or not _encoder_matches(enc, model):
            return None
        return [qv, *doc_vecs]
    except Exception as exc:
        _warn_once(
            f"query_exc::{type(exc).__name__}",
            f"vector 查询向量组装异常（{exc!r}）—— 回退规则序。",
        )
        return None


def reset_caches_for_test() -> None:
    """测试专用：清空进程内索引缓存（生产代码绝不调用）。"""
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()
    _EMBEDDER_CACHE.clear()
