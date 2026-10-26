"""关系选择器的配置与保存评估指纹；不加载模型，不含路径或凭据。

memory_relations 消费阈值，knowledge_service 在读取、保存和提交时核对同一策略。
softmax 最大值未经校准，只用于路由，不能解释为语义判断准确率。
"""
from __future__ import annotations

import hashlib
import json
import math
import os

from ..retrieval.model_runtime import relation_backend_configured, relation_bundle_pin, relation_bundle_root
from ..retrieval.model_worker import RelationBundleError, read_relation_manifest

DEFAULT_CONFIDENCE_THRESHOLD = 0.80


def confidence_threshold() -> float:
    try:
        value = float(os.environ.get("BIODATA_RELATION_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))
    except (TypeError, ValueError):
        raise ValueError("relation_invalid_threshold") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("relation_invalid_threshold")
    return value


def expected_model_identity() -> tuple[str, str] | None:
    root = relation_bundle_root()
    if root is None:
        return None
    try:
        manifest = read_relation_manifest(root, relation_bundle_pin(), verify_files=False)
    except (RelationBundleError, OSError, ValueError):
        return None
    return manifest["model_id"], manifest["meta_sha256"]


def expected_meta_sha256() -> str | None:
    identity = expected_model_identity()
    return identity[1] if identity is not None else None


def policy_hash() -> str:
    enabled = relation_backend_configured()
    policy = {"version": "biodata-relation-policy/1", "mode": "hybrid" if enabled else "llm"}
    if enabled:
        try:
            threshold = confidence_threshold()
        except ValueError:
            threshold = None
        policy.update(threshold=threshold, bundle_sha256=relation_bundle_pin(),
                      classes="semantic-relations/5", score="softmax_max_uncalibrated")
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matches_assessment_policy(assessment: dict) -> bool:
    saved = assessment.get("relation_policy_hash")
    return saved == policy_hash() if saved is not None else not relation_backend_configured()
