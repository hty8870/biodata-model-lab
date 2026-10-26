"""可选「向量召回」层（默认关，本地稠密嵌入，确定性可复现）。

既定方向「规则过滤 + 向量召回 + llm 重排」里的**向量召回**通道。落地为：用本地稠密句向量
模型（sentence-transformers，权重预先下到本地目录、运行时不联网），对**已过硬过滤的存活集**
按与查询的语义相似度重排；可与词面分做融合（hybrid：`fusion="linear"` 为 min-max+α 线性融合，
`fusion="rrf"` 为 RRF 名次融合 k=60——尺度无关、零调参，见 RECALL_FUSIONS 注释）。

设计约束（与 rerank 层同源，defense-in-depth）：
- 只对存活集排序：输入恒为 survivors 子集，输出是其**排列/截断**，绝不引入集合外记录
  → 0% 硬违规由 retriever 终检(passes_hard_filter)结构性保证，本层的排列守卫是冗余。
- 后端 `off`：原样返回（与未启用时**字节等价**）。
- 后端 `dense`：本地嵌入 → 余弦 → 融合 → 稳定排序；模型缺失/依赖未装/任何异常一律回退原序
  （永不报错、永不违规、永不阻塞请求）。
- 后端 `vector`：语料级向量召回——候选向量优先命中本地语料索引（vector_index，首用构建）或
  API 向量文件（未覆盖条目现场补嵌），查询向量现编；缺省 RRF 名次融合。同一套回退合同。
- **可复现**：给定同一模型与输入，稠密嵌入确定 → 适合进分级评测（不像 LLM 重排不可复现）。

依赖边界：本模块**不在导入期引入任何重依赖**；sentence-transformers 仅在真正走 dense 且模型
可用时**惰性导入**。官方评测入口不传 recall 参数 → 结构性走 off，绝不加载模型。

「召回 vs 重排」的诚实说明：本架构里硬过滤已产出**完整**合法集，故稠密向量在此承担的是
「存活集内的语义**召回排序**」，而非「突破规则过滤的扩召回」——后者与 0% 硬违规不变量结构性
冲突（违规判定本身即词面约束匹配，纳入词面漏掉的记录=制造违规）。
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

from .ltr_features import ltr_feature_rows
from .retriever import RetrievedCandidate
from ..app.runtime_paths import get_app_paths

RECALL_BACKENDS = ("off", "dense", "cross_encoder", "vector")
DEFAULT_RECALL_ALPHA = 0.5
                                                 
                                                
                                                                  
                                                  
                                                   
                                                           
                                                                     
                                                                        
                                                            
                              
RECALL_FUSIONS = ("linear", "rrf", "ltr")
                                                                     
DEFAULT_RECALL_FUSION = "linear"
RRF_K = 60
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
                                                              
DEFAULT_CROSS_ENCODER_MODEL = "bge-reranker-v2-m3"

                                      
Embedder = Callable[[Sequence[str]], "list[list[float]]"]
                                                     
CrossScorer = Callable[["Sequence[tuple[str, str]]"], "list[float]"]

                                          
                                           
                                               
                                          
_EMBEDDER_CACHE: "dict[str, Embedder]" = {}
_CROSS_CACHE: "dict[str, CrossScorer]" = {}
                                                                        
                                                 
from .warn_once import WARNED as _WARNED              
from .warn_once import warn_once_prefixed as _warn_once_prefixed              
_DETERMINISM_DONE = False
                                              
                                                         
                                                 
_MODEL_LOCK = threading.Lock()

                                                                      
                                               
                                            
                                                             
                                                     
                                                           
                                            
                                                            
                                                                   
                  
                                          
                                                                       
                                                  
                  
_DOC_VECTOR_CACHE: "OrderedDict[tuple[str, str], list[list[float]]]" = OrderedDict()
_DOC_VECTOR_CACHE_MAX = 2048
_DOC_VECTOR_LOCK = threading.Lock()


def default_model_dir(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    """dense 嵌入模型目录 = model_root/embeddings/<name>（经 runtime_paths 单一真源：
    source/portable = 项目根/models/…，frozen = data_root/models/…，历史路径逐字节一致）。"""
    return get_app_paths().model_root / "embeddings" / model_name


def default_cross_encoder_dir(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> Path:
    """cross-encoder 重排模型目录 = model_root/cross_encoders/<name>（语义同上）。"""
    return get_app_paths().model_root / "cross_encoders" / model_name


def default_ltr_dir() -> Path:
    """LTR 融合模型目录 = model_root/ltr（含 ltr_mlp.pt 与 ltr_spec.json 特征契约）。"""
    return get_app_paths().model_root / "ltr"


                                       
_LTR_SCORER_CACHE: "OrderedDict[str, Callable[[Sequence[Sequence[float]]], list[float]]]" = OrderedDict()


def load_ltr_scorer(model_dir: "str | Path | None" = None):
    """惰性加载 LTR 融合打分器（torch 仅在此刻才导入）。返回 callable(batch_features)->scores。

    契约：目录须含 ltr_spec.json（feature_order/feature_mean/feature_std/model 结构）与
    ltr_mlp.pt（nn.Sequential state_dict，Linear/ReLU 交替）。任何缺失/异常 → None（调用方回退）。
    """
    root = Path(model_dir) if model_dir else default_ltr_dir()
    key = str(root)
    if key in _LTR_SCORER_CACHE:
        return _LTR_SCORER_CACHE[key]
    spec_path = root / "ltr_spec.json"
    weights_path = root / "ltr_mlp.pt"
    if not spec_path.is_file() or not weights_path.is_file():
        return None
    try:
        import json as _json

        import torch                       

        spec = _json.loads(spec_path.read_text(encoding="utf-8"))
        model_spec = spec["model"]
        layers: list = []
        last = int(model_spec["input_dim"])
        for width in model_spec["hidden_layers"]:
            layers += [torch.nn.Linear(last, int(width)), torch.nn.ReLU()]
            last = int(width)
        layers.append(torch.nn.Linear(last, int(model_spec["output_dim"])))
        net = torch.nn.Sequential(*layers)
        net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        net.eval()
        mean = torch.tensor(spec["feature_mean"], dtype=torch.float32)
        std = torch.tensor(spec["feature_std"], dtype=torch.float32)
        order = list(spec["feature_order"])

        def _score(batch: "Sequence[Sequence[float]]") -> "list[float]":
            with torch.no_grad():
                x = torch.tensor([list(map(float, row)) for row in batch], dtype=torch.float32)
                x = (x - mean) / std
                return [float(v) for v in net(x).squeeze(-1).tolist()]

        _score.ltr_feature_order = order                 
        _LTR_SCORER_CACHE[key] = _score
        return _score
    except Exception as exc:                                      
        _warn_once(f"ltr_load::{key}", f"LTR 融合模型加载失败，回退默认融合：{exc!r}")
        return None


def _ltr_fused_order(
    items: "list[RetrievedCandidate]",
    dense: "list[float]",
    intent: object,
    trace: "dict | None",
) -> "list[int] | None":
    """LTR 融合排序：9 维推理期特征 → MLP 分 → 稳定降序。任何一步不可用 → None（调用方回退）。

    特征构造本体已单源化到 `ltr_features.ltr_feature_rows`（2026-09-30 队列② R1：
    生产/离线同一函数，语义与 scripts/train_ltr.py 的训练侧逐一对齐，契约名见
    spec.feature_order）——本函数只保留打分器加载、回退语义与畸形分守卫。
    """
    if intent is None:
        return None                          
    scorer = load_ltr_scorer()
    if scorer is None:
        return None
    rows = ltr_feature_rows(items, dense, intent, scorer.ltr_feature_order)
    scores = scorer(rows)
    if not scores or len(scores) != len(items) or not all(math.isfinite(s) for s in scores):
        _warn_once("ltr_bad_scores", "LTR 融合打分畸形（长度不符或含 NaN/Inf）—— 回退默认融合。")
        return None
    return sorted(range(len(items)), key=lambda i: (-scores[i], i))


def _setup_determinism() -> None:
    """best-effort 可复现（幂等）：eval 语义 + 关 TF32 + cudnn 确定。不启用会崩的严格算法开关，
    因为向量召回只做**排列**，确定性关乎结果可复现而非 0% 违规不变量（那由终检结构性保证）。
    check-then-set 全程持模块级单飞锁（并发下只有第一个线程执行初始化与 os.environ
    写入，其余线程看到 _DETERMINISM_DONE 即返回——flight 线程零 env 触碰）。"""
    global _DETERMINISM_DONE
    if _DETERMINISM_DONE:
        return
    with _MODEL_LOCK:
        if _DETERMINISM_DONE:
            return
        try:
            import os as _os
            _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        _DETERMINISM_DONE = True


def _warn_once(key: str, message: str) -> None:
    """同一原因只在 stderr 提示一次；绝不抛异常、绝不打断请求。"""
    _warn_once_prefixed("[vector_recall]", key, message)


def load_embedder(
    model_dir: "str | Path | None" = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> "Embedder | None":
    """惰性单飞正缓存；同路径模型换代即重载，失败及加载中换代不入缓存。"""
    from .model_worker import model_generation
    resolved = Path(model_dir) if model_dir is not None else default_model_dir(model_name)
    cache_key = str(resolved)
    try:
        generation = model_generation(resolved)
    except (OSError, RuntimeError):
        return None
    cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None and getattr(cached, "model_generation", generation) == generation:
        return cached

    _setup_determinism()
    with _MODEL_LOCK:
        embedder: "Embedder | None" = None
        try:
            generation = model_generation(resolved)
            cached = _EMBEDDER_CACHE.get(cache_key)
            if cached is not None and getattr(cached, "model_generation", generation) == generation:
                return cached
            _EMBEDDER_CACHE.pop(cache_key, None)
            if not resolved.exists():
                _warn_once(
                    cache_key,
                    f"本地嵌入模型不存在：{resolved} —— 向量召回回退为规则顺序。"
                    "先安装 requirements-embeddings.txt 再运行 scripts/fetch_embedding_model.py。",
                )
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except Exception:
                    _warn_once(
                        cache_key,
                        "未安装 sentence-transformers（可选依赖）—— 向量召回回退为规则顺序。"
                        "安装：pip install -r requirements-embeddings.txt。",
                    )
                    SentenceTransformer = None                            

                if SentenceTransformer is not None:
                    try:
                        model = SentenceTransformer(str(resolved), local_files_only=True)
                    except TypeError:
                        model = SentenceTransformer(str(resolved))
                    if generation != model_generation(resolved):
                        return None

                    def _encode(texts: "Sequence[str]") -> "list[list[float]]":
                        if generation != model_generation(resolved):
                            raise RuntimeError("encoder_generation_changed")
                        vecs = model.encode(
                            list(texts), normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False,
                        )
                        if generation != model_generation(resolved):
                            raise RuntimeError("encoder_generation_changed")
                        return [[float(x) for x in v] for v in vecs]

                    _encode.model_generation = generation
                    embedder = _encode
        except Exception as exc:
            _warn_once(cache_key, f"加载嵌入模型失败（{exc}）—— 向量召回回退为规则顺序。")
            embedder = None

        if embedder is not None:
            _EMBEDDER_CACHE[cache_key] = embedder
        return embedder


def load_cross_encoder(
    model_dir: "str | Path | None" = None,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> "CrossScorer | None":
    """惰性单飞正缓存；与嵌入器共用模型代际合同，缺模型或依赖时诚实降级。"""
    from .model_worker import model_generation
    resolved = Path(model_dir) if model_dir is not None else default_cross_encoder_dir(model_name)
    cache_key = "ce::" + str(resolved)
    try:
        generation = model_generation(resolved)
    except (OSError, RuntimeError):
        return None
    cached = _CROSS_CACHE.get(cache_key)
    if cached is not None and getattr(cached, "model_generation", generation) == generation:
        return cached

    _setup_determinism()
    with _MODEL_LOCK:
        scorer: "CrossScorer | None" = None
        try:
            generation = model_generation(resolved)
            cached = _CROSS_CACHE.get(cache_key)
            if cached is not None and getattr(cached, "model_generation", generation) == generation:
                return cached
            _CROSS_CACHE.pop(cache_key, None)
            close = getattr(cached, "close", None)
            if callable(close):
                close()
            if not resolved.exists():
                _warn_once(
                    cache_key,
                    f"本地重排模型不存在：{resolved} —— 向量召回(cross_encoder)回退为规则顺序。"
                    "先装 requirements-embeddings.txt 再运行 scripts/fetch_embedding_model.py --cross-encoder。",
                )
            else:
                try:
                    from sentence_transformers import CrossEncoder
                except Exception:
                    CrossEncoder = None                            

                if CrossEncoder is not None:
                    try:
                        model = CrossEncoder(str(resolved), max_length=512, local_files_only=True)
                    except TypeError:
                        model = CrossEncoder(str(resolved), max_length=512)
                    if generation != model_generation(resolved):
                        return None

                    def _score(pairs: "Sequence[tuple[str, str]]") -> "list[float]":
                        if generation != model_generation(resolved):
                            raise RuntimeError("encoder_generation_changed")
                        scores = model.predict(list(pairs), batch_size=32, show_progress_bar=False)
                        if generation != model_generation(resolved):
                            raise RuntimeError("encoder_generation_changed")
                        return [float(x) for x in scores]

                    _score.model_generation = generation
                    scorer = _score
                else:
                    from .model_runtime import external_cross_scorer, external_runtime_ready
                    if external_runtime_ready():
                        scorer = external_cross_scorer()
                    if scorer is None:
                        _warn_once(cache_key, "未安装可用的本地模型运行组件—— cross_encoder 回退为规则顺序。")
        except Exception as exc:
            _warn_once(cache_key, f"加载重排模型失败（{exc}）—— cross_encoder 回退为规则顺序。")
            scorer = None

        if scorer is not None:
            _CROSS_CACHE[cache_key] = scorer
        return scorer


def semantic_fallback_recall(
    records,
    query: str,
    intent: object = None,
    top_k: int = 12,
    model_dir: "str | Path | None" = None,
) -> "list | None":
    """解析失败场景的语义兜底召回：全库向量相似度 top_k（不经任何条件硬过滤）。

    供 workflow 在「解析弃权且自动降级救不回」时调用（产品裁定 2026-09-12：
    失败路径从沉默改为「向量语义兜底 + 诚实标注」）。与 recall_rerank 的区别：
    本函数没有候选集可重排，直接在语料级向量索引上按查询余弦排序产出候选。
    输出不宣称满足任何条件——标注与展示口径由调用方（workflow/前端）负责。

    嵌入器/索引不可用、语料为空、任何异常 → None（调用方回退旧的诚实弃权，绝不抛错）。
    """
    try:
        from . import vector_index                           
        enc = load_embedder(model_dir)
        if enc is None and model_dir is None:
                                                            
                                                         
                                      
            enc = vector_index.get_embedder()
        if enc is None:
            return None
                                                      
        store = vector_index.ensure_index(records, embedder=enc, scope="view")
        if not store:
            return None
        by_uid = {}
        for r in records:
            raw = getattr(r, "raw", None)
            uid = str(raw.get("dataset_uid") or "") if isinstance(raw, dict) else ""
            if uid:
                by_uid[uid] = r
        q_text = expand_query_bilingual(query, intent) if intent is not None else query
        fresh = enc([q_text])
        dims = vector_index.EMBED_DIMENSIONS
        if not fresh or len(fresh[0]) != dims:
            return None
        qv = [float(x) for x in fresh[0]]
        if not valid_vector(qv, dims):
            return None
        from .retriever import RetrievedCandidate             
        pairs = [(entry.get("v"), by_uid[uid]) for uid, entry in store.items()
                 if uid in by_uid and hasattr(entry, "get") and valid_vector(entry.get("v"), dims)]
        values = batch_cosine(qv, [vector for vector, _ in pairs], failsoft=True)
        scored = [(score, record) for score, (_, record) in zip(values, pairs) if score is not None]
        if not scored:
            return None
        scored.sort(key=lambda t: (-t[0], getattr(t[1], "family_id", ""), getattr(t[1], "dataset_name", "")))
        return [
            RetrievedCandidate(record=rec, score=round(score, 6),
                               matched_fields=[], missing_fields=[], reason="vector_fallback")
            for score, rec in scored[: max(1, int(top_k or 12))]
        ]
    except Exception as exc:
        _warn_once(f"semantic_fallback::{type(exc).__name__}", f"语义兜底召回异常（{exc!r}）—— 回退诚实弃权。")
        return None


def recall_backend_ready(backend: str) -> bool:
    """该 recall 后端是否**已就绪**（模型已在缓存里），**不触发任何加载**。
    'off' / 未知 → True（无需模型）。供 MCP 服务器在 piped-stdio 下判断能否安全使用：
    未预热的 torch 后端若在请求内首次加载会死锁，调用方据此回退规则序。"""
    b = (backend or "off").strip().lower()
    if b == "cross_encoder":
        return _CROSS_CACHE.get("ce::" + str(default_cross_encoder_dir())) is not None
    if b == "dense":
        return _EMBEDDER_CACHE.get(str(default_model_dir())) is not None
    if b == "vector":
                                                                  
                                                                       
        return False
    return True


def recall_backend_local_available(backend: str) -> bool:
    """该 recall 后端的本地模型是否**可加载**（本地模型目录存在 + sentence-transformers 已装），**不触发加载**。

    三态语义：
    - `ready`（recall_backend_ready）= 模型**已在缓存**里（piped-stdio 安全，MCP 用它——未预热的 torch 后端在请求内首次加载会死锁）。
    - `local_available`（本函数）= 本地模型**能加载**（首次加载有开销但不死锁；暖机闸用它——API 数据源无需预热）。
    - `available`（recall_backend_available）= 有**任一**可执行后端（本地可加载或 API 数据源就绪），auto 策略判定用它。
    'off' / 未知后端 → False（无需/无从加载语义模型）。仅做 find_spec + 目录存在性检查，无重导入。
    """
    b = (backend or "off").strip().lower()
    if b not in ("cross_encoder", "dense", "vector"):
        return False
    target = default_cross_encoder_dir() if b == "cross_encoder" else default_model_dir()
    if not target.exists():
        return False
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            return True
    except Exception:
        pass
    if b == "cross_encoder":
        try:
            from .model_runtime import external_runtime_ready
            return external_runtime_ready()
        except Exception:
            return False
    if b == "vector":
                                                         
                                                         
        try:
            from .model_runtime import external_embed_ready
            return external_embed_ready()
        except Exception:
            return False
    return False


def recall_backend_available(backend: str) -> bool:
    """该 recall 后端是否**可执行**（本地模型可加载，或 API 数据源就绪），**不触发加载、不触网**。

    auto 策略判定用它：本机形态（API 闸默认 off）语义与「本地可加载」逐位一致；
    服务器形态（无本地模型、API 闸开且 key 在位）同样计为可用——cross_encoder 执行时
    走智谱 rerank API、dense 走语料向量文件 + 查询侧 API 嵌入（见 recall_rerank 的 API 分支，
    任一失败 fail-closed 回退规则序）。'off' / 未知后端 → False。只探存在性，绝不回显 key。
    """
    b = (backend or "off").strip().lower()
    if b == "cross_encoder":
        if recall_backend_local_available(b):
            return True
        from . import recall_api                                            
        return recall_api.rerank_api_ready()
    if b == "dense":
        if recall_backend_local_available(b):
            return True
        from . import recall_api
        return recall_api.embed_api_ready()
    if b == "vector":
        if recall_backend_local_available(b):
            return True
        from . import recall_api
        return recall_api.embed_api_ready()
    return False


def resolve_auto_recall_preference() -> "tuple[str, bool]":
    """auto 策略的语义后端偏好解析（Web/CLI/Agent 三调用点唯一真源）：返回 (preferred, available)。

    偏好序 vector > cross_encoder：vector 每个查询只现编 1 条查询向量（语料向量走本地索引
    或 API 向量文件），成本远低于 cross_encoder 的全存活集逐对打分；vector 不可执行时回退
    cross_encoder；两者皆无 → ("cross_encoder", False)，调用方按「无语义后端」走确定性词面序。
    判定走 recall_backend_available 语义：不触发加载、不触网。"""
    if recall_backend_available("vector"):
        return ("vector", True)
    return ("cross_encoder", recall_backend_available("cross_encoder"))


def warm_recall_backend(backend: str) -> bool:
    """在**主线程、启动期**预热指定 recall 后端的本地模型（加载进缓存），返回是否就绪。
    'off' / 未知 → True（无需预热）。必须在事件循环 / 请求处理**之外**调用
    （piped-stdio 下唯一不死锁的加载时机；Web 后台预热线程同语义——它只是一条
    非事件循环的普通线程）。"""
    b = (backend or "off").strip().lower()
    if b == "cross_encoder":
        return load_cross_encoder() is not None
    if b == "dense":
        return load_embedder() is not None
    if b == "vector":
                                                                 
                                           
                                                      
                                              
        if load_embedder() is None:
            return False
        from . import vector_index                        
        if vector_index.ensure_index() is None:
            return False
                                         
        load_ltr_scorer()
        return True
    return True


def expand_query_bilingual(query: str, intent: object) -> str:
    """确定性双语扩展（消融证明有效）：中文原 query 后附受控词表解析出的英文 display 术语，
    **仅进重排打分文本，不回灌硬过滤**。帮 cross-encoder 对齐「中文查询 ↔ 英文文档」。"""
    dm = getattr(intent, "display_map", None) or {}
    terms: "list[str]" = []
    for dim in ("species", "tissue", "disease", "platform", "assay"):
        for t in dm.get(dim, []) or []:
            if t and t not in terms:
                terms.append(t)
    if getattr(intent, "has_raw_data_required", None) is True:
        terms.append("with raw FASTQ data")
    return f"{query} ({', '.join(terms)})" if terms else query


from .candidate_text import candidate_text as _candidate_text                                         


def valid_vector(vector: object, dimensions: int | None = None) -> bool:
    """索引与打分共用的形状/有限性守卫；零向量可用，畸形值不可用。"""
    if not isinstance(vector, (list, tuple)) or not vector:
        return False
    if dimensions is not None and len(vector) != dimensions:
        return False
    try:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) for x in vector)
    except (TypeError, ValueError, OverflowError):
        return False


from functools import lru_cache


@lru_cache(maxsize=8192)
def _snapshot_norm(values):
    return math.sqrt(sum(x * x for x in values))


class VectorSnapshot(tuple):
    __slots__ = ()

    def __new__(cls, values):
        if not valid_vector(values):
            raise ValueError("invalid_vector")
        instance = super().__new__(cls, values)
        _snapshot_norm(instance)
        return instance

    @property
    def norm(self):
        return _snapshot_norm(self)

    def __setattr__(self, name, value):
        raise AttributeError("immutable_vector_snapshot")

    def __eq__(self, other):
        return tuple.__eq__(self, tuple(other)) if isinstance(other, (list, tuple)) else NotImplemented

    __hash__ = tuple.__hash__


def batch_cosine(query, documents, *, failsoft=False) -> list:
    if not valid_vector(query):
        raise ValueError("invalid_vector")
    na = math.sqrt(sum(x * x for x in query))
    scores = []
    for document in documents:
        try:
            if type(document) is VectorSnapshot:
                if len(document) != len(query):
                    raise ValueError("invalid_vector")
                nb = document.norm
            else:
                if not valid_vector(document, len(query)):
                    raise ValueError("invalid_vector")
                nb = math.sqrt(sum(y * y for y in document))
            dot = sum(x * y for x, y in zip(query, document))
            score = 0.0 if na <= 0.0 or nb <= 0.0 else dot / (na * nb)
            if not math.isfinite(score):
                raise ValueError("nonfinite_cosine")
        except ValueError:
            if not failsoft:
                raise
            score = None
        scores.append(score)
    return scores


def _cosine(a: "list[float]", b: "list[float]") -> float:
    return batch_cosine(a, [b])[0]


def _minmax_norm(values: "list[float]") -> "list[float]":
    """min-max 归一化到 [0,1]；全相等时返回全 0（不引入词面偏好，交给稠密分/稳定序）。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _rrf_fused_order(lex_scores: "list[float]", dense_scores: "list[float]", k: int = RRF_K) -> "list[int]":
    """RRF 融合（k=60）：`score(d) = Σ 1/(k + rank_i(d))`——只用两路名次、零尺度假设。

    名次并列按原下标稳定（与 linear 路的稳定序同口径）；返回按融合分降序的下标序列
    （同分按下标，确定性可复现）。"""
    n = len(lex_scores)
    lex_order = sorted(range(n), key=lambda i: (-float(lex_scores[i]), i))
    dense_order = sorted(range(n), key=lambda i: (-float(dense_scores[i]), i))
    lex_rank = [0] * n
    dense_rank = [0] * n
    for r, i in enumerate(lex_order):
        lex_rank[i] = r
    for r, i in enumerate(dense_order):
        dense_rank[i] = r
    fused = [1.0 / (k + lex_rank[i]) + 1.0 / (k + dense_rank[i]) for i in range(n)]
    return sorted(range(n), key=lambda i: (-fused[i], i))


def _dense_vectors_cached(
    enc: Embedder,
    model_dir: "str | Path | None",
    query: str,
    texts: "list[str]",
) -> "list[list[float]] | None":
    """默认加载路径的稠密向量组装：查询向量现编，文档向量查 _DOC_VECTOR_CACHE（缺失才编码）。

    返回 [查询向量, *文档向量]；**仅当**文档批编码输出长度不符时返回 None（对齐外层
    invalid_vectors「畸形输出→回退」语义）。本函数不吞任何异常——查询批为空/畸形、enc 抛错
    一律上抛，交给 recall_rerank 外层宽 except 兜底（既定「永不阻塞」合同）。
    字节等价红线：缓存命中返回的是**当初编码出的同一个 list 对象**（绝不重算），融合数学与
    _cosine 一行未动 → linear 融合输出逐位不变。
    """
                                                            
                                                      
    qvecs = enc([query])
    cache_key_model = str(Path(model_dir) if model_dir is not None else default_model_dir())
    if hasattr(enc, "model_generation"):
        cache_key_model = (cache_key_model, enc.model_generation)
    doc_vecs: "list[list[float] | None]" = []
    missing: "list[int]" = []
    with _DOC_VECTOR_LOCK:
        for t in texts:
            key = (cache_key_model, t)
            if key in _DOC_VECTOR_CACHE:
                _DOC_VECTOR_CACHE.move_to_end(key)                           
                doc_vecs.append(_DOC_VECTOR_CACHE[key])
            else:
                doc_vecs.append(None)                                       
                missing.append(len(doc_vecs) - 1)
    if missing:
                                                    
        fresh = enc([texts[i] for i in missing])
        if len(fresh) != len(missing):
            return None                                                                         
        with _DOC_VECTOR_LOCK:
            for pos, vec in zip(missing, fresh):
                doc_vecs[pos] = vec
                _DOC_VECTOR_CACHE[(cache_key_model, texts[pos])] = vec
                while len(_DOC_VECTOR_CACHE) > _DOC_VECTOR_CACHE_MAX:
                    _DOC_VECTOR_CACHE.popitem(last=False)                       
    return [qvecs[0], *doc_vecs]


def recall_rerank(
    query: str,
    candidates: "Sequence[RetrievedCandidate]",
    backend: str = "off",
    alpha: float = DEFAULT_RECALL_ALPHA,
    embedder: "Embedder | None" = None,
    cross_scorer: "CrossScorer | None" = None,
    intent: object | None = None,
    model_dir: "str | Path | None" = None,
    top_k: "int | None" = None,
    trace: "dict | None" = None,
    fusion: "str | None" = None,
) -> "list[RetrievedCandidate]":
    """可选向量召回（存活集内语义重排）。四种后端，输出恒为输入 candidates 的**排列/截断**。

    backend="off"（默认）    → 原样返回（截断到 top_k，如未传则不截）。
    backend="dense"          → 本地稠密嵌入余弦 × 词面分融合排序：缺省 `fusion="ltr"`（学习排序融合；
                               工件缺失时 fail-soft 回退 `linear`）；`fusion="linear"` 为 min-max
                               归一化 + alpha 线性融合（历史行为字节等价）；
                               `fusion="rrf"` 为 RRF 名次融合（k=60，尺度无关、零调参）。
    backend="vector"         → 语料级向量索引召回：候选向量命中本地语料索引 / API 向量文件
                               （未覆盖条目现场补嵌），余弦 × 词面分融合；缺省 `fusion="ltr"`
                               （fail-soft 回退 `rrf`）。
    backend="cross_encoder"  → 本地 cross-encoder 对全存活集逐对打分，按**纯分**排序（词面原序仅同分 tie-break）；
                               中文 query 附英文别名（双语扩展）以对齐英文文档。**基准选型胜出的默认路径**。

    任一后端：模型缺失/依赖未装/任何异常/畸形输出 → 回退传入顺序（永不报错、永不违规、永不阻塞）。
    embedder / cross_scorer 可注入（便于单测/评测共享一次加载）。intent 用于双语扩展（cross_encoder）。
    """
    items = list(candidates)
    started_at = time.perf_counter()
                                                               
                                                                   
                                                     
    resolved_fusion = str(fusion or "").strip().lower() or ("ltr" if backend in ("dense", "vector") else "linear")

    def _duration_ms() -> int:
        return max(0, int(round((time.perf_counter() - started_at) * 1000)))

    if trace is not None:
        trace.update({
            "backend": backend, "status": "skipped", "reason": "disabled",
            "candidate_count": len(items), "duration_ms": 0,
            "fusion": (resolved_fusion if backend in ("dense", "vector") else ""),
        })
    if not items or backend == "off" or backend not in RECALL_BACKENDS:
        if trace is not None and not items:
            trace.update({"status": "skipped", "reason": "no_candidates", "duration_ms": _duration_ms()})
        return items[:top_k] if top_k is not None else items

    def _mark(status: str, reason: str) -> None:
        if trace is not None:
            trace.update({"status": status, "reason": reason, "duration_ms": _duration_ms()})
            if status == "fallback" and backend in ("dense", "vector"):
                trace["fusion"] = "rule-order"                     

    def _cut(seq: "list[RetrievedCandidate]") -> "list[RetrievedCandidate]":
        return seq[:top_k] if top_k is not None else seq

    try:
        if backend == "cross_encoder":
                                                                       
                                                               
            api_scores: "list[float] | None" = None
            if cross_scorer is None:
                from . import recall_api                         
                if recall_api.api_rerank_enabled():
                    q_api = expand_query_bilingual(query, intent) if intent is not None else query
                    api_scores = recall_api.api_rerank_scores(
                        q_api, [_candidate_text(c) for c in items]
                    )
                    if api_scores is None:
                                                                  
                        _mark("fallback", "api_unavailable")
                        return _cut(items)
            if api_scores is not None:
                scores = api_scores
            else:
                score_fn = cross_scorer if cross_scorer is not None else load_cross_encoder(model_dir)
                if score_fn is None:
                    _mark("fallback", "model_or_dependency_unavailable")
                    return _cut(items)                                        
                q = expand_query_bilingual(query, intent) if intent is not None else query
                pairs = [(q, _candidate_text(c)) for c in items]
                scores = score_fn(pairs)
            if not scores or len(scores) != len(items) or not all(math.isfinite(float(s)) for s in scores):
                                                                    
                                             
                _warn_once("ce_bad_scores", "cross_encoder 打分畸形（长度不符或含 NaN/Inf）—— 回退规则序。")
                _mark("fallback", "invalid_scores")
                return _cut(items)
                                                                    
            order = sorted(range(len(items)), key=lambda i: (-float(scores[i]), i))
            _mark("used", "completed")
            return _cut([items[i] for i in order])

                                                                   
                                                                  
        use_rrf = resolved_fusion == "rrf"
        a = max(0.0, min(1.0, float(alpha)))
        if backend == "vector":
                                                              
                                                          
            vectors: "list[list[float]] | None" = None
            if embedder is not None:
                texts = [_candidate_text(c) for c in items]
                vectors = embedder([query, *texts])
            else:
                from . import recall_api                         
                if recall_api.api_embed_enabled():
                    vectors = recall_api.api_dense_vectors(query, items)
                    if vectors is None:
                                                                    
                        _mark("fallback", "api_unavailable")
                        return _cut(items)
                else:
                    from . import vector_index                         
                    vectors = vector_index.index_dense_vectors(query, items)
                    if vectors is None:
                        _mark("fallback", "index_unavailable")
                        return _cut(items)
        else:
                                
                                                                        
                                                               
            api_vectors: "list[list[float]] | None" = None
            if embedder is None:
                from . import recall_api                         
                if recall_api.api_embed_enabled():
                    api_vectors = recall_api.api_dense_vectors(query, items)
                    if api_vectors is None:
                                                                    
                        _mark("fallback", "api_unavailable")
                        return _cut(items)
            enc: "Embedder | None" = None
            if api_vectors is None:
                enc = embedder if embedder is not None else load_embedder(model_dir)
                if enc is None:
                    _mark("fallback", "model_or_dependency_unavailable")
                    return _cut(items)                                            
            if api_vectors is not None:
                vectors = api_vectors
            elif embedder is None:
                                                             
                                                                  
                texts = [_candidate_text(c) for c in items]
                vectors = _dense_vectors_cached(enc, model_dir, query, texts)
            else:
                                                                     
                                                      
                texts = [_candidate_text(c) for c in items]
                vectors = enc([query, *texts])
        if not vectors or len(vectors) != len(items) + 1:                            
            _mark("fallback", "invalid_vectors")
            return _cut(items)
        qv = vectors[0]
        try:
            dense = batch_cosine(qv, vectors[1:])
        except ValueError:
            _warn_once("dense_bad_vectors", "dense 向量维度不一致或含非有限值 —— 回退规则序。")
            _mark("fallback", "invalid_vectors")
            return _cut(items)
        if not all(math.isfinite(d) for d in dense):                                 
            _warn_once("dense_bad_scores", "dense 稠密分含 NaN/Inf —— 回退规则序。")
            _mark("fallback", "invalid_scores")
            return _cut(items)
        if resolved_fusion == "ltr":
                                                         
                                                         
            ltr_order = _ltr_fused_order(items, dense, intent, trace)
            if ltr_order is not None:
                if trace is not None:
                    trace["fusion"] = "ltr"
                _mark("used", "completed")
                return _cut([items[i] for i in ltr_order])
            resolved_fusion = "rrf" if backend == "vector" else "linear"
            use_rrf = resolved_fusion == "rrf"
            if trace is not None:
                trace["fusion"] = resolved_fusion
                trace["ltr_status"] = "fallback"
        if use_rrf:
                                                    
            lex_raw = [float(getattr(c, "score", 0.0) or 0.0) for c in items]
            if not all(math.isfinite(v) for v in lex_raw):
                _warn_once("dense_bad_scores", "dense 词面分含 NaN/Inf —— 回退规则序。")
                _mark("fallback", "invalid_scores")
                return _cut(items)
            order = _rrf_fused_order(lex_raw, dense)
            if trace is not None:
                trace["fusion"] = "rrf"
        else:
            lex = _minmax_norm([float(getattr(c, "score", 0.0) or 0.0) for c in items])
            blended = [(1.0 - a) * lex[i] + a * dense[i] for i in range(len(items))]
            if not all(math.isfinite(b) for b in blended):                                     
                _warn_once("dense_bad_scores", "dense 融合分含 NaN/Inf —— 回退规则序。")
                _mark("fallback", "invalid_scores")
                return _cut(items)
            order = sorted(range(len(items)), key=lambda i: (-blended[i], i))
        _mark("used", "completed")
        return _cut([items[i] for i in order])
    except Exception as exc:
                                                              
                                                            
        _warn_once(f"recall_exc::{backend}", f"向量召回({backend})运行期异常，回退规则序：{exc!r}")
        _mark("fallback", "runtime_error")
        if trace is not None and resolved_fusion == "ltr":
            trace["ltr_status"] = "fallback"
        return _cut(items)
