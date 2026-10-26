"""记忆关系轻量模型的特征与推理封装（研究-only，生产不引用）。

冻结 MiniLM 只编码不微调：候选 u、旧规则聚合（mean v、mean|u-v|、mean u*v）、
出处证据同构，加少量文本结构特征；char n-gram 基线用确定性请求文本。
特征面只读真实投影会保留的字段：candidate/existing 的 text、trigger、candidate
evidence 引用与 evidence 出处文本——绝不读 id、tier、permanence、标签、来源、
分区分组与 context（替换权限判据）；训练/回放同一投影口径，避免构造先验（如
implicit 与答案相关）混进特征。RelationScorer.classes 恒为相邻 spec.json 的
classes 原序，predict_proba 恒返回 n×spec 类数；训练缺席的类别概率恰为 0。
spec.json 是历史七类数据格式的唯一真源，不随当前生产 RELATIONS 变化。
嵌入缓存键 = sha256(模型内容指纹 + 文本)，换模型内容或改文本都重算，绝不按行 id 缓存。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

_SPEC_PATH = Path(__file__).resolve().with_name("spec.json")


def _spec_classes() -> tuple[str, ...]:
    """历史七类格式的唯一真源：相邻 spec.json 的 classes 原序。

    生产 memory_relations 已换五叶；研究数据/manifest 仍按这份冻结七类格式构造，
    因此不能把当前生产 RELATIONS 当研究类序，也不在这里做 7↔5 映射。
    """
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    classes = spec.get("classes")
    if not isinstance(classes, list) or not classes \
            or any(not isinstance(name, str) or not name for name in classes) \
            or len(set(classes)) != len(classes):
        raise RuntimeError("spec.json classes 必须是唯一的非空字符串列表")
    return tuple(classes)


                                                    
CLASSES = _spec_classes()
ARTIFACT_SCHEMA = "memory-relations-scorer-v1"
                                             
FEATURE_RECIPE = "memory-relations-features-v2"
EMBEDDING_DEFAULT_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEVICE_DEFAULT = "cpu"
THREADS_DEFAULT = 6
BATCH_DEFAULT = 16
                                           
_COUNT_CAP = 8.0
_EPS = 1e-12
_TRIGGER_MARK = "\n[trigger] "


def model_fingerprint(model_dir: "str | Path") -> str:
    """模型目录内容指纹：分块读取每个文件字节的 sha256（含全部权重/tokenizer/配置）。

    同大小同 mtime 的内容变化也必须失效缓存，故不做 size/mtime 快路径；不读取
    文件标识（inode 等），只信内容。首次 ~1GB 读取（一次性成本）。
    """
    root = Path(model_dir)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()[:32]


def _text(value: object, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串，实际是 {type(value).__name__}")
    return value


def request_view(request: dict) -> dict:
    """请求 → 特征视图：只取真实投影保留的 text/trigger/evidence 类字段。

    id、tier、permanence、scope、labels、provenance、group/split、context（替换权限判据）
    一律不读取——训练侧与生产 `_rule` 投影同口径，回放/训练特征一致。
    返回的字符串在后续特征里只做长度、拼接与嵌入，不做任何身份复原。
    """
    if not isinstance(request, dict):
        raise ValueError("request 必须是字典")
    candidate = request.get("candidate")
    existing = request.get("existing")
    if not isinstance(candidate, dict):
        raise ValueError("request.candidate 必须是字典")
    if not isinstance(existing, list):
        raise ValueError("request.existing 必须是列表")
    evidence = request.get("evidence")
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise ValueError("request.evidence 必须是列表")
    quotes = candidate.get("evidence")
    if quotes is None:
        quotes = []
    if not isinstance(quotes, list):
        raise ValueError("candidate.evidence 必须是列表")
    rule_texts: list[str] = []
    rule_triggers: list[str] = []
    for row in existing:
        if not isinstance(row, dict):
            raise ValueError("existing 行必须是字典")
        rule_texts.append(_text(row.get("text"), "existing.text"))
        rule_triggers.append(_text(row.get("trigger"), "existing.trigger"))
    source_texts: list[str] = []
    for row in evidence:
        if not isinstance(row, dict):
            raise ValueError("evidence 行必须是字典")
        source_texts.append(_text(row.get("text"), "evidence.text"))
    return {
        "candidate_text": _text(candidate.get("text"), "candidate.text"),
        "candidate_trigger": _text(candidate.get("trigger"), "candidate.trigger"),
        "quotes": [_text(item, "candidate.evidence[]") for item in quotes],
        "rule_texts": rule_texts,
        "rule_triggers": rule_triggers,
        "evidence_texts": source_texts,
    }


def request_has_trigger(request: dict) -> bool:
    """候选是否带非空 trigger（门策略 contextual_requires_trigger 的唯一实现，宽容解析）。"""
    candidate = request.get("candidate") if isinstance(request, dict) else None
    trigger = candidate.get("trigger") if isinstance(candidate, dict) else None
    return isinstance(trigger, str) and bool(trigger.strip())


def request_text(request: dict) -> str:
    """char n-gram 基线的确定性输入串：标记词与文本同源可读、行序稳定。"""
    view = request_view(request)
    lines = [view["candidate_text"]]
    if view["candidate_trigger"]:
        lines.append("candidate trigger " + view["candidate_trigger"])
    for index, text in enumerate(view["quotes"]):
        lines.append(f"quote{index} {text}")
    for index, text in enumerate(view["rule_texts"]):
        lines.append(f"existing{index} {text}")
        if view["rule_triggers"][index]:
            lines.append(f"existing{index} trigger {view['rule_triggers'][index]}")
    for index, text in enumerate(view["evidence_texts"]):
        lines.append(f"evidence{index} {text}")
    return "\n".join(lines)


def _structural_from_view(view: dict) -> np.ndarray:
    rule_lengths = [len(text) for text in view["rule_texts"]]
    evidence_lengths = [len(text) for text in view["evidence_texts"]]
    features = [
        np.log1p(len(view["candidate_text"])),
        np.log1p(len(view["candidate_trigger"])),
        min(len(view["rule_texts"]), _COUNT_CAP) / _COUNT_CAP,
        np.log1p(max(rule_lengths)) if rule_lengths else 0.0,
        float(np.mean([np.log1p(length) for length in rule_lengths])) if rule_lengths else 0.0,
        1.0 if any(view["rule_triggers"]) else 0.0,
        min(sum(1 for trigger in view["rule_triggers"] if trigger), _COUNT_CAP) / _COUNT_CAP,
        min(len(view["evidence_texts"]), _COUNT_CAP) / _COUNT_CAP,
        min(len(view["quotes"]), _COUNT_CAP) / _COUNT_CAP,
        np.log1p(max(evidence_lengths)) if evidence_lengths else 0.0,
    ]
    return np.asarray(features, dtype=np.float64)


def structural_features(request: dict) -> np.ndarray:
    """少量文本结构特征：长度与条数（截断），全部由请求文本直接数出（v2 起无元数据标志）。"""
    return _structural_from_view(request_view(request))


class Embedder:
    """本地 SentenceTransformer（CPU、禁止下载）+（模型指纹, 文本）两级缓存。

    磁盘缓存 = npz(keys=sha256(指纹+文本), vectors=float32)；指纹不符即视为无缓存。
    缓存只加速，不改变结果；encode 不做自动落盘，由调用方显式 save_cache()。
    """

    def __init__(
        self,
        model_dir: "str | Path",
        *,
        device: str = DEVICE_DEFAULT,
        threads: int = THREADS_DEFAULT,
        batch_size: int = BATCH_DEFAULT,
        cache_dir: "str | Path | None" = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"本地嵌入模型目录不存在：{self.model_dir}（禁止下载）")
        self.device = device
        self.threads = int(threads)
        self.batch_size = int(batch_size)
        self.fingerprint = model_fingerprint(self.model_dir)
        self.cache_path = Path(cache_dir) / "embeddings.npz" if cache_dir else None
        self.versions: dict[str, str] = {}
        self._model = None
        self._memory: dict[str, np.ndarray] = {}
        self._load_cache()

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256((self.fingerprint + "\x00" + text).encode("utf-8")).hexdigest()

    def _ensure_model(self):
        if self._model is None:
            import os

            import torch
            from sentence_transformers import SentenceTransformer

            os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
            torch.set_num_threads(self.threads)
            try:
                model = SentenceTransformer(str(self.model_dir), device=self.device,
                                            local_files_only=True)
            except TypeError:                                               
                model = SentenceTransformer(str(self.model_dir), device=self.device)
            import sentence_transformers

            self.versions = {"sentence_transformers": sentence_transformers.__version__,
                             "torch": torch.__version__}
            self._model = model
        return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure_model().get_sentence_embedding_dimension())

    def _load_cache(self) -> None:
        path = self.cache_path
        if path is None or not path.is_file():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self.fingerprint:
                    return
                keys = [str(key) for key in data["keys"]]
                vectors = np.asarray(data["vectors"], dtype=np.float32)
        except Exception:                     
            return
        if vectors.ndim == 2 and len(keys) == vectors.shape[0]:
            for key, vector in zip(keys, vectors):
                self._memory[key] = vector

    def save_cache(self) -> None:
        if self.cache_path is None or not self._memory:
            return
        keys = sorted(self._memory)
        vectors = np.stack([self._memory[key] for key in keys])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, fingerprint=np.array(self.fingerprint),
                     keys=np.array(keys, dtype="U64"), vectors=vectors)
        temporary.replace(self.cache_path)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """文本 → 规范化 float32 向量 (n, dim)；重复文本只编码一次。"""
        unique: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("encode 只接受字符串")
            if text not in seen:
                seen.add(text)
                unique.append(text)
        keyed = [(self._cache_key(text), text) for text in unique]
        missing = [(key, text) for key, text in keyed if key not in self._memory]
        if missing:
            vectors = np.asarray(
                self._ensure_model().encode(
                    [text for _, text in missing], batch_size=self.batch_size,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
                dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(missing):
                raise RuntimeError("嵌入输出形状异常")
            for (key, _), vector in zip(missing, vectors):
                self._memory[key] = vector
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._memory[self._cache_key(text)] for text in texts])


def _embed_key(text: str, trigger: str) -> str:
    return f"{text}{_TRIGGER_MARK}{trigger}" if trigger else text


def minilm_feature_matrix(embedder: Embedder, requests: Sequence[dict]) -> np.ndarray:
    """冻结 MiniLM 特征矩阵，块序固定：

    [u, mean(v), mean|u-v|, mean(u*v), mean(e), mean|u-e|, mean(u*e), 结构特征]
    其中 u=候选、v=各旧规则、e=各出处文本，均按指纹+文本缓存去重编码；
    无旧规则/无出处时相应块取零，由结构特征的条数特征表达缺失。
    """
    views = [request_view(request) for request in requests]
    prewarm: list[str] = []
    for view in views:
        prewarm.append(_embed_key(view["candidate_text"], view["candidate_trigger"]))
        prewarm.extend(_embed_key(text, trigger)
                       for text, trigger in zip(view["rule_texts"], view["rule_triggers"]))
        prewarm.extend(view["evidence_texts"])
    if prewarm:
        embedder.encode(prewarm)
    rows: list[np.ndarray] = []
    for view in views:
        candidate = embedder.encode(
            [_embed_key(view["candidate_text"], view["candidate_trigger"])])[0]
        blocks: list[np.ndarray] = [candidate]
        rule_keys = [_embed_key(text, trigger)
                     for text, trigger in zip(view["rule_texts"], view["rule_triggers"])]
        for keys in (rule_keys, view["evidence_texts"]):
            if keys:
                matrix = embedder.encode(keys)
                blocks.extend([matrix.mean(axis=0), np.abs(candidate - matrix).mean(axis=0),
                               (candidate * matrix).mean(axis=0)])
            else:
                zeros = np.zeros_like(candidate)
                blocks.extend([zeros, zeros, zeros])
        blocks.append(_structural_from_view(view))
        rows.append(np.concatenate(blocks))
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


class RelationScorer:
    """轻量分类器的推理封装：classes 恒为 spec 类序，输出恒为 n×7。

    temperature 只来自 calibration；predict_proba 输出校准后概率，
    predict_proba_raw 输出校准前概率（供 ECE/Brier 对照）。
    """

    def __init__(
        self,
        *,
        kind: str,
        model,
        class_indices: Sequence[int],
        temperature: float = 1.0,
        embedder: "Embedder | None" = None,
    ) -> None:
        if kind not in ("majority", "char-logreg", "minilm-logreg", "minilm-mlp"):
            raise ValueError(f"未知模型类型：{kind}")
        if float(temperature) <= 0.0:
            raise ValueError("temperature 必须为正")
        self.kind = kind
        self.classes = CLASSES
        self.model = model
        self.class_indices = tuple(int(index) for index in class_indices)
        self.temperature = float(temperature)
        self._embedder = embedder

    @property
    def embedder(self) -> "Embedder | None":
        return self._embedder

    def _full_logits(self, requests: Sequence[dict]) -> np.ndarray:
        count = len(requests)
        logits = np.full((count, len(CLASSES)), -np.inf, dtype=np.float64)
        if not count:
            return logits
        if self.kind == "majority":
            prior = np.asarray(self.model, dtype=np.float64)
            with np.errstate(divide="ignore"):
                return np.log(prior)[None, :].repeat(count, axis=0)
        if self.kind == "char-logreg":
            raw = self.model.decision_function([request_text(item) for item in requests])
        else:
            raw = np.log(np.clip(self.model.predict_proba(
                minilm_feature_matrix(self._embedder, list(requests))), _EPS, 1.0))
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = np.stack([-raw, raw], axis=1)
        logits[:, list(self.class_indices)] = raw
        return logits

    def predict_proba_raw(self, requests) -> np.ndarray:
        items = [requests] if isinstance(requests, dict) else list(requests)
        if not items:
            return np.zeros((0, len(CLASSES)), dtype=np.float64)
        return softmax(self._full_logits(items))

    def predict_proba(self, requests) -> np.ndarray:
        """requests：请求字典序列（也接受单个请求）→ (n, spec 类序) 概率。"""
        items = [requests] if isinstance(requests, dict) else list(requests)
        if not items:
            return np.zeros((0, len(CLASSES)), dtype=np.float64)
        return softmax(self._full_logits(items) / self.temperature)


def load_scorer(model_dir: "str | Path",
                device: str = DEVICE_DEFAULT) -> "RelationScorer | CrossEncoderScorer":
    """加载训练输出的打分器（adapter/evaluate 消费的唯一公共 loader）。

    先按 finetune 工件识别（meta.json schema）惰性分派到 finetune.load_scorer（cross-encoder），
    否则按 sklearn 工件（model.joblib）加载；契约不符即抛错，绝不静默降级，也不另建第二回放入口。
    """
    from research.memory_relations import finetune

    if finetune.is_finetune_artifact(model_dir):
        return finetune.load_scorer(model_dir, device=device)
    import joblib

    path = Path(model_dir) / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"打分器工件不存在：{path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("打分器工件 schema 不符")
    if tuple(payload.get("classes") or ()) != CLASSES:
        raise ValueError("打分器类别序与 spec 类序不符")
    if payload.get("feature_recipe") != FEATURE_RECIPE:
        raise ValueError("打分器特征配方与当前实现不符")
    kind = payload["kind"]
    embedder = None
    if kind in ("minilm-logreg", "minilm-mlp"):
        embedder_path = payload.get("embedder_path")
        if not embedder_path or not Path(embedder_path).is_dir():
            raise FileNotFoundError(f"嵌入模型缺失：{embedder_path!r}（禁止下载）")
        embedder = Embedder(embedder_path, device=device,
                            threads=payload.get("threads", THREADS_DEFAULT),
                            batch_size=payload.get("batch_size", BATCH_DEFAULT))
        expected = payload.get("embedding_fingerprint")
        if expected and expected != embedder.fingerprint:
            raise ValueError("嵌入模型文件指纹与训练时不一致；换模型必须重新训练")
    return RelationScorer(kind=kind, model=payload["model"],
                          class_indices=payload["class_indices"],
                          temperature=payload.get("temperature", 1.0), embedder=embedder)
