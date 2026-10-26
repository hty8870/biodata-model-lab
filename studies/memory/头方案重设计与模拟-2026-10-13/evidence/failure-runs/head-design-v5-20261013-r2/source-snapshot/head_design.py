"""记忆语义头 v5（r2）的研究专用模拟 harness（研究-only）。

- 形式面：8 世界（3 布尔原子）、255 个非空满足集，两个非空 bitmap 的 5 叶
  （equivalent/old_entails_new_strict/new_entails_old_strict/disjoint/overlap_incomparable）
  在全部 65025 有序对上穷举；唯一 leaf->parent 配置 E/E/N/C/N；未知/空输入/背景不一致/
  条件交集未知/空交集单列原因码，不当 neutral；完整条件句与条件交叠区间内容关系不等价。
- 训练/损失全在 log 空间：flat 头用 log_softmax 与分组 logsumexp，H 用 log_softmax(parent)
  + logsigmoid(±branch)；λ=1 时"粗分组+条件细"恒等于对应 logits 的交叉熵；训练全程
  loss/grad/params/dev finite 检查，非有限即非零退出且不产出成功模型。所有臂的粗监督是
  整张有序表（含逆向）；fine 只是每次呈现的监督粒度与显式 coupling，不是新的整表标注。
- 目标口径是互补族隔离设计下的 conditional partition distribution；不声称全 65025 均匀或
  NLP 泛化；finaltoy 封存（封存是不用于选择的流程承诺；生成器公开，非信息论秘密）。
- 候选 A/B/C/H/D：A=3 logits 粗 CE；B=5 logits 粗分组；C=B+同 λ 条件细；H=3 父+E/N 两条件
  二分支（5 logits/4 自由度）；D 是五叶图的粗消费面。B 的 fine 是 unidentified diagnostic。
  选点分粗/细两个 estimand（同分取最早），结构比较只比较 C/H 的 fine-estimand。
- CLI 阶段：formal / prepare / train-dev / freeze / finalize / status；`freeze` 从验证过的
  train-dev 历史与权重生成冻结 decision（不是自报字符串）；finalize 校验完整链路、真实加载
  权重、登记 final-access attempt（失败也记录，同 binding 需 --reproduce）；prepare 不导出、
  不评分 finaltoy；新 out 必须不存在（拒绝覆盖已有 run）；运行产物默认只写 runs/ 之下。
- 编辑合同审计是抽象 oracle case + 单故障 mutations（合同覆盖用），不复制生产提交/CAS 状态机，
  也不声称 LLM 编辑成功；重复 ID 直接拒绝，冲突按 pair+fingerprint 记账。
- 边界：形式 oracle 不是自然语言评测；bitmap 不证明 NL 转换；五叶只回答语义关系，不承担
  定位/授权/执行成功证明；哈希指纹是完整性流程，不声称防恶意整套重写或替代认证。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WORLD_BITS = 8
WORLD_MASK = (1 << WORLD_BITS) - 1
TRAINABLE_ARMS = ("A", "B", "C", "H")
MANDATORY_PREPARE_ARTIFACTS = (
    "protocol-frozen.json",
    "partition.json",
    "exposures/train.jsonl",
    "exposures/dev.jsonl",
    "exposures/calibration_prob.jsonl",
    "exposures/calibration_policy.jsonl",
)
ATTEMPTS_FILENAME = "final-access-attempts.jsonl"
MANDATORY_MANIFEST_KEYS = ("stage", "created_at", "protocol_sha256", "source_sha256", "artifacts", "budget")
AUDIT_CODES = (
    "unauthorized_strengthening",
    "unauthorized_weakening",
    "unauthorized_readonly_change",
    "unauthorized_condition_change",
    "unauthorized_ttl_change",
    "scope_escalation",
    "stale_read_basis",
    "unknown_read_basis",
    "joint_inconsistent",
    "mixed_object_classes",
    "evidence_lineage_loss",
    "undeclared_change",
    "unmentioned_field_changed",
    "introduced_conflict",
)


class HarnessError(Exception):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def protocol_default_path() -> Path:
    return Path(__file__).with_name("head-design-v5.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return _repo_root() / "research" / "memory_relations" / "runs"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _write_json(path: Path, obj: object) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    except ValueError as exc:
        raise HarnessError(f"refusing success write with non-finite value for {path}: {exc}", 1) from exc
    return _write_bytes(path, (text + "\n").encode("utf-8"))


def _read_json(path: Path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read JSON {path}: {exc}", 2) from exc


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def runtime_info() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
    }


def _array_digest(parts: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(parts):
        value = np.ascontiguousarray(parts[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def require_mask(value, *, name: str = "mask", allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise HarnessError(f"{name} must be a plain int in [0, {WORLD_MASK}]", 2)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(f"{name} must be a plain int in [0, {WORLD_MASK}]", 2)
    if not 0 <= value <= WORLD_MASK:
        raise HarnessError(f"{name} out of range [0, {WORLD_MASK}]: {value}", 2)
    return value


def load_protocol(path: Path | str | None = None) -> tuple[dict, str]:
    target = Path(path) if path is not None else protocol_default_path()
    payload = target.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    validate_protocol(protocol)
    return protocol, sha256_bytes(payload)


def validate_protocol(protocol: dict) -> None:
    if protocol.get("revision") != "r2":
        raise HarnessError("protocol revision must be r2", 2)
    formal = protocol["formal"]
    if formal["world_count"] != WORLD_BITS or formal["full_mask"] != WORLD_MASK:
        raise HarnessError("protocol formal world geometry mismatch", 2)
    leaves = formal["leaves"]
    parents = formal["parents"]
    if len(set(leaves)) != len(leaves) or len(set(parents)) != len(parents):
        raise HarnessError("protocol duplicate leaf/parent names", 2)
    if set(formal["leaf_parent"]) != set(leaves):
        raise HarnessError("protocol leaf_parent must cover exactly the leaves", 2)
    if not set(formal["leaf_parent"].values()) <= set(parents):
        raise HarnessError("protocol leaf_parent maps outside parents", 2)
    if set(formal["expected_leaf_counts"]) != set(leaves):
        raise HarnessError("protocol expected leaf counts mismatch", 2)
    if set(formal["expected_parent_counts"]) != set(parents):
        raise HarnessError("protocol expected parent counts mismatch", 2)
    total = formal["pair_total"]
    if sum(formal["expected_leaf_counts"].values()) != total:
        raise HarnessError("protocol expected leaf counts do not sum to pair total", 2)
    if sum(formal["expected_parent_counts"].values()) != total:
        raise HarnessError("protocol expected parent counts do not sum to pair total", 2)
    for leaf, count in formal["expected_leaf_counts"].items():
        if count < 0:
            raise HarnessError("protocol negative leaf count", 2)
    from_leaf = {parent: 0 for parent in parents}
    for leaf in leaves:
        from_leaf[formal["leaf_parent"][leaf]] += formal["expected_leaf_counts"][leaf]
    if from_leaf != formal["expected_parent_counts"]:
        raise HarnessError("protocol parent counts inconsistent with leaf->parent sums", 2)
    for combo, leaf in formal["combination_leaf"].items():
        if len(combo) != 2 or combo[0] not in parents or combo[1] not in parents or leaf not in leaves:
            raise HarnessError("protocol combination_leaf entry invalid", 2)
    if not set(formal["combination_leaf"]) <= set(formal["legal_bidirectional"]):
        raise HarnessError("protocol combination_leaf must stay within legal bidirectional combos", 2)
    partition = protocol["partition"]
    pairs_per_stratum = {k: sum(1 for a, _ in complement_pairs() if bin(a).count("1") == k) for k in (1, 2, 3, 4)}
    for k, counts in partition["strata_min_side"].items():
        if sum(counts.values()) != pairs_per_stratum[int(k)]:
            raise HarnessError(f"protocol strata_min_side[{k}] does not partition the {pairs_per_stratum[int(k)]} pairs", 2)
    expected_pairs = partition["expected_pairs"]
    by_role = {role: sum(partition["strata_min_side"][k][role] for k in partition["strata_min_side"]) for role in ("train", "dev", "calibration", "finaltoy")}
    for role, count in by_role.items():
        if expected_pairs[role] != count:
            raise HarnessError(f"protocol expected_pairs[{role}] inconsistent with strata", 2)
    if expected_pairs["total"] != 127:
        raise HarnessError("protocol complement pair total must be 127", 2)
    for role in ("train", "dev", "calibration", "finaltoy"):
        head = expected_pairs[role] * 2 + (1 if role == "train" else 0)
        if partition["expected_entities"][role] != head:
            raise HarnessError(f"protocol expected_entities[{role}] inconsistent", 2)
    pools = partition["calibration_pools"]
    if pools["prob_pairs"] + pools["policy_pairs"] != expected_pairs["calibration"]:
        raise HarnessError("protocol calibration pools must cover the calibration pairs", 2)
    temperature = protocol["evaluation"]["temperature"]
    low, high = temperature["range"]
    if not 0 < low < 1 < high:
        raise HarnessError("protocol temperature range must straddle 1.0", 2)
    rule = protocol["training"]["architecture_rule"]
    if rule["default"] not in TRAINABLE_ARMS or rule["challenger"] not in TRAINABLE_ARMS:
        raise HarnessError("protocol architecture rule arms invalid", 2)
    extension = protocol["training"]["extension"]
    if extension["decide_before_step"] != extension["trigger_step"] + 1:
        raise HarnessError("protocol extension must decide before any step beyond the trigger", 2)
    if extension["windows"]["early"] >= extension["windows"]["late"]:
        raise HarnessError("protocol extension windows invalid", 2)
    selection = protocol["training"]["selection"]
    if selection["fine_estimand"]["arms"] != ["C", "H"] or selection["coarse_estimand"]["metric"] != "dev_coarse_nll":
        raise HarnessError("protocol selection rules invalid", 2)


@dataclass(frozen=True)
class Verdict:
    status: str
    leaf: str | None
    reason: str | None


def leaf_of(p_mask: int, h_mask: int) -> str:
    require_mask(p_mask, name="p_mask")
    require_mask(h_mask, name="h_mask")
    if p_mask == h_mask:
        return "equivalent"
    overlap = p_mask & h_mask
    if overlap == p_mask:
        return "old_entails_new_strict"
    if overlap == h_mask:
        return "new_entails_old_strict"
    if overlap == 0:
        return "disjoint"
    return "overlap_incomparable"


def parent_of_masks(p_mask: int, h_mask: int) -> str:
    require_mask(p_mask, name="p_mask")
    require_mask(h_mask, name="h_mask")
    if p_mask & (WORLD_MASK ^ h_mask) == 0:
        return "E"
    if p_mask & h_mask == 0:
        return "C"
    return "N"


def classify(p_mask: int | None, h_mask: int | None, *, background_consistent: bool = True) -> Verdict:
    p_value = require_mask(p_mask, name="p_mask", allow_none=True)
    h_value = require_mask(h_mask, name="h_mask", allow_none=True)
    if p_value is None or h_value is None:
        return Verdict("unresolved", None, "input_unknown")
    if p_value == 0 or h_value == 0:
        return Verdict("unresolved", None, "empty_satisfier")
    if not background_consistent:
        return Verdict("unresolved", None, "background_mismatch")
    return Verdict("relation", leaf_of(p_value, h_value), None)


def conditional_full_relation(
    p_mask: int,
    h_mask: int,
    cond_p_mask: int,
    cond_h_mask: int,
    *,
    overlap_known: bool = True,
    background_consistent: bool = True,
) -> Verdict:
    p_value = require_mask(p_mask, name="p_mask")
    h_value = require_mask(h_mask, name="h_mask")
    cond_p = require_mask(cond_p_mask, name="cond_p_mask")
    cond_h = require_mask(cond_h_mask, name="cond_h_mask")
    if not overlap_known:
        return Verdict("unresolved", None, "condition_overlap_unknown")
    if cond_p & cond_h == 0:
        return Verdict("unresolved", None, "empty_condition_overlap")
    statement_p = (p_value & cond_p) | (WORLD_MASK ^ cond_p)
    statement_h = (h_value & cond_h) | (WORLD_MASK ^ cond_h)
    return classify(statement_p, statement_h, background_consistent=background_consistent)


def interval_content_relation(
    p_mask: int,
    h_mask: int,
    cond_p_mask: int,
    cond_h_mask: int,
    *,
    overlap_known: bool = True,
) -> Verdict:
    p_value = require_mask(p_mask, name="p_mask")
    h_value = require_mask(h_mask, name="h_mask")
    cond_p = require_mask(cond_p_mask, name="cond_p_mask")
    cond_h = require_mask(cond_h_mask, name="cond_h_mask")
    if not overlap_known:
        return Verdict("unresolved", None, "condition_overlap_unknown")
    overlap = cond_p & cond_h
    if overlap == 0:
        return Verdict("unresolved", None, "empty_condition_overlap")
    return classify(p_value & overlap, h_value & overlap)


def combination_leaf(label_a: str, label_b: str, protocol: dict) -> str | None:
    return protocol["formal"]["combination_leaf"].get(label_a + label_b)


def joint_marginals_feasible(marginal_a: Sequence[float], marginal_b: Sequence[float], protocol: dict) -> bool:
    parents = protocol["formal"]["parents"]
    legal = [combo for combo in protocol["formal"]["legal_bidirectional"]]
    if len(marginal_a) != len(parents) or len(marginal_b) != len(parents):
        raise HarnessError("marginal vectors must match the parent count", 2)
    if any(value < -1e-9 or not math.isfinite(float(value)) for value in list(marginal_a) + list(marginal_b)):
        return False
    source = len(parents) * 2
    sink = source + 1
    capacity: dict[tuple[int, int], float] = {}

    def add_edge(fr: int, to: int, cap: float) -> None:
        capacity[(fr, to)] = capacity.get((fr, to), 0.0) + cap
        capacity.setdefault((to, fr), 0.0)

    for index, value in enumerate(marginal_a):
        add_edge(source, index, max(0.0, float(value)))
    for index, value in enumerate(marginal_b):
        add_edge(len(parents) + index, sink, max(0.0, float(value)))
    for combo in legal:
        row = parents.index(combo[0])
        col = parents.index(combo[1])
        add_edge(row, len(parents) + col, 1.0)
    flow = 0.0
    while True:
        parent = {source: None}
        queue = [source]
        while queue and sink not in parent:
            node = queue.pop(0)
            for (fr, to), cap in capacity.items():
                if fr == node and cap > 1e-12 and to not in parent:
                    parent[to] = fr
                    queue.append(to)
        if sink not in parent:
            break
        path_bottleneck = float("inf")
        node = sink
        while parent[node] is not None:
            path_bottleneck = min(path_bottleneck, capacity[(parent[node], node)])
            node = parent[node]
        node = sink
        while parent[node] is not None:
            capacity[(parent[node], node)] -= path_bottleneck
            capacity[(node, parent[node])] = capacity.get((node, parent[node]), 0.0) + path_bottleneck
            node = parent[node]
        flow += path_bottleneck
    return flow >= 1.0 - 1e-9


def bidirectional_summary(
    hard_a: Sequence[str],
    hard_b: Sequence[str],
    *,
    protocol: dict,
    marginal_a: Sequence[Sequence[float]] | None = None,
    marginal_b: Sequence[Sequence[float]] | None = None,
    c_edge_tolerance: float | None = None,
) -> dict:
    if len(hard_a) != len(hard_b) or not hard_a:
        raise HarnessError("bidirectional summary needs equal non-empty label sequences", 2)
    tolerance = float(c_edge_tolerance if c_edge_tolerance is not None else protocol["evaluation"]["joint_marginals"].get("c_edge_tolerance", 0.5))
    leaves = protocol["formal"]["leaves"]
    fine_counts = {leaf: 0 for leaf in leaves}
    legal = 0
    for label_a, label_b in zip(hard_a, hard_b):
        leaf = combination_leaf(label_a, label_b, protocol)
        if leaf is None:
            continue
        legal += 1
        fine_counts[leaf] += 1
    total = len(hard_a)
    summary = {
        "n": total,
        "legal": legal,
        "illegal": total - legal,
        "hard_combo_unmappable": total - legal,
        "coverage": legal / total,
        "fine_counts": fine_counts,
        "denominator": "full",
    }
    if (marginal_a is None) != (marginal_b is None):
        raise HarnessError("joint feasibility needs both marginal vectors", 2)
    if marginal_a is not None and marginal_b is not None:
        if len(marginal_a) != total or len(marginal_b) != total:
            raise HarnessError("marginals must align with the hard labels", 2)
        c_index = protocol["formal"]["parents"].index("C")
        gaps = [abs(float(a[c_index]) - float(b[c_index])) for a, b in zip(marginal_a, marginal_b)]
        incompatible = sum(1 for a, b in zip(marginal_a, marginal_b) if not joint_marginals_feasible(a, b, protocol))
        summary["c_edge_mismatch_count"] = int(sum(1 for gap in gaps if gap > tolerance))
        summary["c_edge_mean_abs_gap"] = float(sum(gaps) / total)
        summary["c_edge_tolerance"] = tolerance
        summary["prob_joint_incompatible"] = int(incompatible)
        summary["prob_joint_incompatible_fraction"] = incompatible / total
    return summary


def exhaustive_census(protocol: dict) -> dict:
    formal = protocol["formal"]
    leaves = formal["leaves"]
    parents = formal["parents"]
    leaf_counts = {leaf: 0 for leaf in leaves}
    parent_counts = {parent: 0 for parent in parents}
    combo_counts: dict[str, int] = {}
    swap_violations = 0
    leaf_parent_violations = 0
    conservation_violations = 0
    illegal_combinations = 0
    combination_mismatches = 0
    legal = set(formal["legal_bidirectional"])
    for p_mask in range(1, WORLD_MASK + 1):
        for h_mask in range(1, WORLD_MASK + 1):
            leaf = leaf_of(p_mask, h_mask)
            leaf_counts[leaf] += 1
            parent = parent_of_masks(p_mask, h_mask)
            parent_counts[parent] += 1
            if formal["leaf_parent"][leaf] != parent:
                leaf_parent_violations += 1
            if formal["swap_map"][leaf] != leaf_of(h_mask, p_mask):
                swap_violations += 1
            combo = parent + parent_of_masks(h_mask, p_mask)
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
            if combo not in legal:
                illegal_combinations += 1
            if combination_leaf(parent, parent_of_masks(h_mask, p_mask), protocol) != leaf:
                combination_mismatches += 1
    for parent in parents:
        from_leaf = sum(leaf_counts[leaf] for leaf in leaves if formal["leaf_parent"][leaf] == parent)
        if from_leaf != parent_counts[parent]:
            conservation_violations += 1
    matches_expected = leaf_counts == formal["expected_leaf_counts"] and parent_counts == formal["expected_parent_counts"]
    return {
        "pair_total": WORLD_MASK * WORLD_MASK,
        "leaf_counts": leaf_counts,
        "parent_counts": parent_counts,
        "combo_counts": dict(sorted(combo_counts.items())),
        "swap_violations": swap_violations,
        "leaf_parent_violations": leaf_parent_violations,
        "conservation_violations": conservation_violations,
        "illegal_combinations": illegal_combinations,
        "combination_mismatches": combination_mismatches,
        "matches_expected": matches_expected,
    }
def complement_pairs() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    for mask in range(1, WORLD_MASK + 1):
        partner = WORLD_MASK ^ mask
        if partner == 0 or partner in seen:
            continue
        seen.add(mask)
        seen.add(partner)
        low, high = sorted((mask, partner), key=lambda m: (bin(m).count("1"), m))
        pairs.append((low, high))
    return pairs


def build_partition(protocol: dict) -> dict:
    partition = protocol["partition"]
    rng = random.Random(partition["seed"])
    pairs = complement_pairs()
    strata = {k: [pair for pair in pairs if bin(pair[0]).count("1") == k] for k in (1, 2, 3, 4)}
    roles: dict[str, list[tuple[int, int]]] = {role: [] for role in ("train", "dev", "calibration", "finaltoy")}
    for k in (1, 2, 3, 4):
        bucket = list(strata[k])
        rng.shuffle(bucket)
        counts = partition["strata_min_side"][str(k)]
        offset = 0
        for role in ("train", "dev", "calibration", "finaltoy"):
            take = counts[role]
            roles[role].extend(bucket[offset:offset + take])
            offset += take
        if offset != len(bucket):
            raise HarnessError(f"protocol strata_min_side[{k}] does not fill the {len(bucket)} pairs", 2)
    calibration = list(roles["calibration"])
    rng.shuffle(calibration)
    half = partition["calibration_pools"]["prob_pairs"]
    if half * 2 != len(calibration):
        raise HarnessError("protocol calibration pools must split the calibration pairs evenly", 2)
    pair_pools = {
        "train": roles["train"],
        "dev": roles["dev"],
        "calibration_prob": calibration[:half],
        "calibration_policy": calibration[half:],
        "finaltoy": roles["finaltoy"],
    }
    entity_pools: dict[str, list[int]] = {}
    for pool, pool_pairs in pair_pools.items():
        entities = [mask for pair in pool_pairs for mask in pair]
        if pool == "train":
            entities.append(WORLD_MASK)
        entity_pools[pool] = sorted(entities)
    return {
        "seed": partition["seed"],
        "stratum_pairs": {str(k): len(strata[k]) for k in (1, 2, 3, 4)},
        "pair_pools": pair_pools,
        "entity_pools": entity_pools,
    }


def ordered_exposures(entities: Sequence[int]) -> list[tuple[int, int]]:
    return [(p_mask, h_mask) for p_mask in entities for h_mask in entities]


def exposure_jsonl(pairs: Iterable[tuple[int, int]]) -> bytes:
    return "".join(json.dumps({"p": p, "h": h}, separators=(",", ":")) + "\n" for p, h in pairs).encode("utf-8")


def entity_digest(entities: Sequence[int]) -> str:
    return sha256_bytes(",".join(str(mask) for mask in sorted(entities)).encode("utf-8"))


def pool_statistics(entities: Sequence[int], pairs: Sequence[tuple[int, int]], protocol: dict) -> dict:
    leaf_counts = {leaf: 0 for leaf in protocol["formal"]["leaves"]}
    parent_counts = {parent: 0 for parent in protocol["formal"]["parents"]}
    diagonal = 0
    complement = 0
    for p_mask, h_mask in pairs:
        leaf_counts[leaf_of(p_mask, h_mask)] += 1
        parent_counts[parent_of_masks(p_mask, h_mask)] += 1
        if p_mask == h_mask:
            diagonal += 1
        if p_mask ^ h_mask == WORLD_MASK:
            complement += 1
    return {
        "leaf_counts": leaf_counts,
        "parent_counts": parent_counts,
        "diagonal_pairs": diagonal,
        "diagonal_fraction": diagonal / len(pairs),
        "complement_ordered_pairs": complement,
        "complement_fraction": complement / len(pairs),
        "cardinality_support": sorted({bin(mask).count("1") for mask in entities}),
    }


def partition_report(protocol: dict) -> dict:
    partition = protocol["partition"]
    built = build_partition(protocol)
    expected = partition["expected_pairs"]
    actual_by_role = {role: len(pool) for role, pool in built["pair_pools"].items()}
    if actual_by_role["train"] != expected["train"] or actual_by_role["dev"] != expected["dev"]:
        raise HarnessError("partition train/dev pair counts mismatch", 2)
    if actual_by_role["calibration_prob"] + actual_by_role["calibration_policy"] != expected["calibration"]:
        raise HarnessError("partition calibration pair counts mismatch", 2)
    if actual_by_role["finaltoy"] != expected["finaltoy"]:
        raise HarnessError("partition finaltoy pair counts mismatch", 2)
    pools: dict[str, dict] = {}
    expected_counts = partition["expected_entities"]
    expected_pool_pairs = partition["expected_pool_pairs"]
    for pool, entities in built["entity_pools"].items():
        sealed = pool == "finaltoy"
        wanted = expected_counts["calibration"] // 2 if pool in ("calibration_prob", "calibration_policy") else expected_counts[pool]
        if len(entities) != wanted:
            raise HarnessError(f"partition entity count mismatch for {pool}", 2)
        exposures = ordered_exposures(entities)
        if len(exposures) != expected_pool_pairs[pool]:
            raise HarnessError(f"partition pool pair count mismatch for {pool}", 2)
        pools[pool] = {
            "entities": None if sealed else entities,
            "entity_count": len(entities),
            "entity_sha256": entity_digest(entities),
            "pairs": len(exposures),
            "exposure_sha256": sha256_bytes(exposure_jsonl(exposures)),
            "sealed": sealed,
            "exported": not sealed,
            "stats": pool_statistics(entities, exposures, protocol),
        }
    train_entities = pools["train"]["entity_count"]
    dev_entities = pools["dev"]["entity_count"]
    cal_entities = pools["calibration_prob"]["entity_count"] + pools["calibration_policy"]["entity_count"]
    final_entities = pools["finaltoy"]["entity_count"]
    if (train_entities, dev_entities, cal_entities, final_entities) != (
        partition["expected_entities"]["train"],
        partition["expected_entities"]["dev"],
        partition["expected_entities"]["calibration"],
        partition["expected_entities"]["finaltoy"],
    ):
        raise HarnessError("partition entity counts do not match protocol", 2)
    universe = WORLD_MASK * WORLD_MASK
    included = sum(row["pairs"] for row in pools.values())
    cross_partition = universe - (train_entities ** 2 + dev_entities ** 2 + cal_entities ** 2 + final_entities ** 2)
    calibration_cross_pool = cal_entities ** 2 - (pools["calibration_prob"]["entity_count"] ** 2 + pools["calibration_policy"]["entity_count"] ** 2)
    excluded = {"cross_partition": cross_partition, "calibration_cross_pool": calibration_cross_pool, "total": universe - included}
    if excluded != partition["excluded_pairs"]:
        raise HarnessError("partition exclusions do not match protocol (real denominators changed)", 2)
    return {
        "seed": built["seed"],
        "stratum_pairs": built["stratum_pairs"],
        "pools": pools,
        "denominators": {
            "universe_pairs": universe,
            "included_pairs": {**{pool: row["pairs"] for pool, row in pools.items()}, "total": included},
            "excluded_pairs": excluded,
        },
        "distribution_target": partition["distribution_target"],
    }


def index_maps(protocol: dict) -> dict:
    formal = protocol["formal"]
    leaves = formal["leaves"]
    parents = formal["parents"]
    groups = {parent: [i for i, leaf in enumerate(leaves) if formal["leaf_parent"][leaf] == parent] for parent in parents}
    for parent, members in groups.items():
        if not members:
            raise HarnessError(f"parent {parent} has no leaves", 2)
        if len(members) not in (1, 2):
            raise HarnessError(f"parent {parent} needs one or two leaves for this design", 2)
    for parent in protocol["training"]["candidates"]["H"]["branches"]:
        if len(groups[parent]) != 2:
            raise HarnessError(f"H branch parent {parent} must hold exactly two leaves", 2)
    return {
        "leaves": leaves,
        "parents": parents,
        "groups": groups,
        "parent_of_leaf": [parents.index(formal["leaf_parent"][leaf]) for leaf in leaves],
        "leaf_index": {leaf: i for i, leaf in enumerate(leaves)},
    }


def encode_pairs(pairs: Sequence[tuple[int, int]], protocol: dict) -> np.ndarray:
    spec = protocol["training"]["input"]
    p_bits = spec["p_bits"]
    h_bits = spec["h_bits"]
    for p_mask, h_mask in pairs:
        require_mask(p_mask, name="p_mask")
        require_mask(h_mask, name="h_mask")
    p_masks = np.asarray([p for p, _ in pairs], dtype=np.int64)
    h_masks = np.asarray([h for _, h in pairs], dtype=np.int64)
    p_low = ((p_masks[:, None] >> np.arange(p_bits)[None, :]) & 1).astype(np.float32)
    h_low = ((h_masks[:, None] >> np.arange(h_bits)[None, :]) & 1).astype(np.float32)
    return np.concatenate([p_low, h_low], axis=1)


def label_indices(pairs: Sequence[tuple[int, int]], protocol: dict, maps: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    maps = maps or index_maps(protocol)
    leaf_idx = np.asarray([maps["leaf_index"][leaf_of(p, h)] for p, h in pairs], dtype=np.int64)
    parent_idx = np.asarray([maps["parent_of_leaf"][i] for i in leaf_idx], dtype=np.int64)
    return leaf_idx, parent_idx


def leaf_counts_from_pairs(pairs: Sequence[tuple[int, int]], protocol: dict) -> dict:
    counts = {leaf: 0 for leaf in protocol["formal"]["leaves"]}
    for p, h in pairs:
        counts[leaf_of(p, h)] += 1
    return counts


def parent_counts_from_pairs(pairs: Sequence[tuple[int, int]], protocol: dict) -> dict:
    counts = {parent: 0 for parent in protocol["formal"]["parents"]}
    for p, h in pairs:
        counts[parent_of_masks(p, h)] += 1
    return counts


def pregenerate_common(seed: int, n_train: int, protocol: dict) -> dict:
    spec = protocol["training"]
    backbone_rng = np.random.default_rng(np.random.PCG64([seed, 0xB0]))
    batches_rng = np.random.default_rng(np.random.PCG64([seed, 0xBA]))
    hidden = spec["backbone"]["hidden"]
    upstream = spec["input"]["p_bits"] + spec["input"]["h_bits"]
    w1 = backbone_rng.normal(0.0, math.sqrt(2.0 / upstream), (hidden[0], upstream)).astype(np.float32)
    b1 = np.zeros(hidden[0], dtype=np.float32)
    w2 = backbone_rng.normal(0.0, math.sqrt(2.0 / hidden[0]), (hidden[1], hidden[0])).astype(np.float32)
    b2 = np.zeros(hidden[1], dtype=np.float32)
    updates = max(spec["updates"], spec["extension"]["target_updates"])
    batches = batches_rng.integers(0, n_train, size=(updates, spec["batch_size"]))
    backbone = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return {
        "backbone": backbone,
        "batches": batches,
        "backbone_sha256": _array_digest(backbone),
        "batches_sha256": _array_digest({"batches": batches}),
        "streams": {
            "backbone": f"PCG64(seed={seed}, tag=0xB0) independent of batches",
            "batches": f"PCG64(seed={seed}, tag=0xBA) independent of backbone",
        },
    }


def initial_priors(protocol: dict, parent_counts: dict) -> dict:
    parents = protocol["formal"]["parents"]
    raw = np.asarray([max(float(parent_counts.get(parent, 0)), 1e-9) for parent in parents], dtype=np.float64)
    pi_parent = raw / raw.sum()
    pi_leaf = np.zeros(len(protocol["formal"]["leaves"]), dtype=np.float64)
    maps = index_maps(protocol)
    for parent_index, parent in enumerate(parents):
        members = maps["groups"][parent]
        for member in members:
            pi_leaf[member] = pi_parent[parent_index] / len(members)
    return {"pi_parent": pi_parent, "pi_leaf": pi_leaf}


def head_init(arm: str, seed: int, protocol: dict, parent_counts: dict) -> dict:
    if arm not in TRAINABLE_ARMS:
        raise HarnessError(f"head_init got non-trainable arm {arm}", 2)
    priors = initial_priors(protocol, parent_counts)
    rng = np.random.default_rng(np.random.PCG64([seed, 1049 + ord(arm)]))
    scale = float(protocol["training"]["init"]["tiny_random_scale"])

    def tiny_weights(out_dim: int) -> np.ndarray:
        return rng.normal(0.0, scale, (out_dim, 64)).astype(np.float32)

    state: dict[str, np.ndarray] = {}
    if arm == "A":
        state["parent_w"] = tiny_weights(3)
        state["parent_b"] = np.log(priors["pi_parent"]).astype(np.float32)
    elif arm in ("B", "C"):
        state["leaf_w"] = tiny_weights(5)
        state["leaf_b"] = np.log(priors["pi_leaf"]).astype(np.float32)
    else:
        state["parent_w"] = tiny_weights(3)
        state["parent_b"] = np.log(priors["pi_parent"]).astype(np.float32)
        for parent in protocol["training"]["candidates"]["H"]["branches"]:
            key = parent.lower()
            state[f"branch_{key}_w"] = rng.normal(0.0, scale, (1, 64)).astype(np.float32)
            state[f"branch_{key}_b"] = np.zeros(1, dtype=np.float32)
    return state


class TinyHeadModel(nn.Module):
    def __init__(self, arm: str, protocol: dict, backbone_state: dict, head_state: dict):
        super().__init__()
        self.arm = arm
        self.protocol = protocol
        self.maps = index_maps(protocol)
        spec = protocol["training"]
        upstream = spec["input"]["p_bits"] + spec["input"]["h_bits"]
        hidden = spec["backbone"]["hidden"]
        self.backbone = nn.Sequential(
            nn.Linear(upstream, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        with torch.no_grad():
            self.backbone[0].weight.copy_(torch.as_tensor(backbone_state["w1"]))
            self.backbone[0].bias.copy_(torch.as_tensor(backbone_state["b1"]))
            self.backbone[2].weight.copy_(torch.as_tensor(backbone_state["w2"]))
            self.backbone[2].bias.copy_(torch.as_tensor(backbone_state["b2"]))
        if arm == "A":
            self.parent_head = nn.Linear(hidden[1], 3)
            with torch.no_grad():
                self.parent_head.weight.copy_(torch.as_tensor(head_state["parent_w"]))
                self.parent_head.bias.copy_(torch.as_tensor(head_state["parent_b"]))
        elif arm in ("B", "C"):
            self.leaf_head = nn.Linear(hidden[1], len(self.maps["leaves"]))
            with torch.no_grad():
                self.leaf_head.weight.copy_(torch.as_tensor(head_state["leaf_w"]))
                self.leaf_head.bias.copy_(torch.as_tensor(head_state["leaf_b"]))
        elif arm == "H":
            self.parent_head = nn.Linear(hidden[1], len(self.maps["parents"]))
            self.branch_e = nn.Linear(hidden[1], 1)
            self.branch_n = nn.Linear(hidden[1], 1)
            with torch.no_grad():
                self.parent_head.weight.copy_(torch.as_tensor(head_state["parent_w"]))
                self.parent_head.bias.copy_(torch.as_tensor(head_state["parent_b"]))
                for parent in spec["candidates"]["H"]["branches"]:
                    key = parent.lower()
                    module = getattr(self, f"branch_{key}")
                    module.weight.copy_(torch.as_tensor(head_state[f"branch_{key}_w"]))
                    module.bias.copy_(torch.as_tensor(head_state[f"branch_{key}_b"]))
        else:
            raise HarnessError(f"unknown arm {arm}", 2)


def forward_log(model: TinyHeadModel, x: torch.Tensor) -> dict:
    features = model.backbone(x)
    if model.arm == "A":
        parent_logits = model.parent_head(features)
        return {
            "parent_logits": parent_logits,
            "parent_logp": torch.log_softmax(parent_logits, dim=-1),
            "leaf_logits": None,
            "leaf_logp": None,
            "branch_logits": None,
        }
    if model.arm in ("B", "C"):
        leaf_logits = model.leaf_head(features)
        return {
            "parent_logits": None,
            "parent_logp": None,
            "leaf_logits": leaf_logits,
            "leaf_logp": torch.log_softmax(leaf_logits, dim=-1),
            "branch_logits": None,
        }
    parent_logits = model.parent_head(features)
    parent_logp = torch.log_softmax(parent_logits, dim=-1)
    branch_logits = {
        parent: getattr(model, f"branch_{parent.lower()}")(features).squeeze(-1)
        for parent in model.protocol["training"]["candidates"]["H"]["branches"]
    }
    out = torch.full((parent_logits.shape[0], len(model.maps["leaves"])), float("-inf"), dtype=parent_logits.dtype)
    for parent_index, parent in enumerate(model.maps["parents"]):
        members = model.maps["groups"][parent]
        if parent in branch_logits:
            branch = branch_logits[parent]
            out[:, members[0]] = parent_logp[:, parent_index] + F.logsigmoid(branch)
            out[:, members[1]] = parent_logp[:, parent_index] + F.logsigmoid(-branch)
        else:
            out[:, members[0]] = parent_logp[:, parent_index]
    return {
        "parent_logits": parent_logits,
        "parent_logp": parent_logp,
        "leaf_logits": None,
        "leaf_logp": out,
        "branch_logits": branch_logits,
    }


def log_group_sums(logp5: torch.Tensor, maps: dict) -> torch.Tensor:
    return torch.stack([torch.logsumexp(logp5[:, maps["groups"][parent]], dim=-1) for parent in maps["parents"]], dim=-1)


def grouped_coarse_nll(logp5: torch.Tensor, leaf_idx: torch.Tensor, maps: dict) -> torch.Tensor:
    log_groups = log_group_sums(logp5, maps)
    gold_parent = torch.as_tensor(maps["parent_of_leaf"], dtype=torch.long)[leaf_idx]
    gold = log_groups.gather(1, gold_parent.unsqueeze(1)).squeeze(1)
    return -gold.mean()


def conditional_fine_nll(logp5: torch.Tensor, leaf_idx: torch.Tensor, maps: dict) -> torch.Tensor:
    log_groups = log_group_sums(logp5, maps)
    gold_parent = torch.as_tensor(maps["parent_of_leaf"], dtype=torch.long)[leaf_idx]
    gold_group = log_groups.gather(1, gold_parent.unsqueeze(1)).squeeze(1)
    gold_leaf = logp5.gather(1, leaf_idx.unsqueeze(1)).squeeze(1)
    return (gold_group - gold_leaf).mean()


def five_ce(logits5: torch.Tensor, leaf_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits5, leaf_idx)


def coarse_ce(parent_logits: torch.Tensor, parent_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(parent_logits, parent_idx)


def arm_loss(arm: str, output: dict, leaf_idx: torch.Tensor, parent_idx: torch.Tensor, protocol: dict) -> torch.Tensor:
    if arm == "A":
        return coarse_ce(output["parent_logits"], parent_idx)
    maps = index_maps(protocol)
    lam = protocol["training"]["lambda_fine"]
    loss = grouped_coarse_nll(output["leaf_logp"], leaf_idx, maps)
    if arm in ("C", "H"):
        loss = loss + lam * conditional_fine_nll(output["leaf_logp"], leaf_idx, maps)
    return loss


ALLOWED_OBJECTIVE_SETS = {
    ("coarse_ce",),
    ("coarse_grouped",),
    ("coarse_grouped", "conditional_fine"),
}


def objective_loss(
    objectives: Sequence[str],
    *,
    protocol: dict,
    leaf_idx: torch.Tensor,
    parent_idx: torch.Tensor,
    parent_logits: torch.Tensor | None = None,
    leaf_logp: torch.Tensor | None = None,
) -> torch.Tensor:
    key = tuple(objectives)
    if key not in ALLOWED_OBJECTIVE_SETS:
        raise HarnessError(f"objective combination not allowed (double-count risk): {key}", 2)
    maps = index_maps(protocol)
    lam = protocol["training"]["lambda_fine"]
    if key == ("coarse_ce",):
        if parent_logits is None:
            raise HarnessError("coarse_ce needs parent logits", 2)
        return coarse_ce(parent_logits, parent_idx)
    if leaf_logp is None:
        raise HarnessError("grouped objectives need five-leaf log probabilities", 2)
    loss = grouped_coarse_nll(leaf_logp, leaf_idx, maps)
    if "conditional_fine" in key:
        loss = loss + lam * conditional_fine_nll(leaf_logp, leaf_idx, maps)
    return loss


def _assert_finite(value, what: str, exit_code: int = 1) -> None:
    if isinstance(value, torch.Tensor):
        ok = bool(torch.isfinite(value).all().item())
    else:
        ok = math.isfinite(float(value))
    if not ok:
        raise HarnessError(f"non-finite {what}; refusing to continue", exit_code)


def _check_params_finite(model: nn.Module, what: str) -> None:
    for name, parameter in model.named_parameters():
        _assert_finite(parameter, f"{what}:{name}")


def _macro_f1_and_recall(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> tuple[float, list[float]]:
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(n_classes):
        pred_mask = pred == index
        gold_mask = gold == index
        tp = int((pred_mask & gold_mask).sum())
        fp = int((pred_mask & ~gold_mask).sum())
        fn = int((~pred_mask & gold_mask).sum())
        recalls.append(tp / (tp + fn) if tp + fn else float("nan"))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = recalls[-1]
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            f1s.append(float("nan"))
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    usable = [value for value in f1s if not math.isnan(value)]
    macro = float(sum(usable) / len(usable)) if usable else float("nan")
    return macro, recalls


def dev_metrics(model: TinyHeadModel, x: torch.Tensor, leaf_idx: torch.Tensor, parent_idx: torch.Tensor, protocol: dict) -> dict:
    model.eval()
    with torch.no_grad():
        output = forward_log(model, x)
        if model.arm == "A":
            logp3 = output["parent_logp"]
            coarse_nll = float(-logp3.gather(1, parent_idx.unsqueeze(1)).squeeze(1).mean())
            fine_nll = None
            leaf_logp = None
        else:
            leaf_logp = output["leaf_logp"]
            logp3 = log_group_sums(leaf_logp, model.maps)
            coarse_nll = float(-logp3.gather(1, parent_idx.unsqueeze(1)).squeeze(1).mean())
            fine_nll = float(-leaf_logp.gather(1, leaf_idx.unsqueeze(1)).squeeze(1).mean())
        _assert_finite(coarse_nll, "dev coarse NLL")
        if fine_nll is not None:
            _assert_finite(fine_nll, "dev fine NLL")
        p3 = torch.exp(logp3).numpy()
        p5 = None if leaf_logp is None else torch.exp(leaf_logp).numpy()
    parent_pred = p3.argmax(axis=1)
    gold_parent = parent_idx.numpy()
    coarse_accuracy = float((parent_pred == gold_parent).mean())
    coarse_macro_f1, coarse_recall = _macro_f1_and_recall(parent_pred, gold_parent, len(model.maps["parents"]))
    _assert_finite(coarse_accuracy, "dev coarse accuracy")
    _assert_finite(coarse_macro_f1, "dev coarse macro F1")
    if p5 is None:
        fine_accuracy = None
        fine_macro_f1 = None
        fine_recall = None
    else:
        leaf_pred = p5.argmax(axis=1)
        gold_leaf = leaf_idx.numpy()
        fine_accuracy = float((leaf_pred == gold_leaf).mean())
        fine_macro_f1, fine_recall = _macro_f1_and_recall(leaf_pred, gold_leaf, len(model.maps["leaves"]))
        _assert_finite(fine_accuracy, "dev fine accuracy")
        _assert_finite(fine_macro_f1, "dev fine macro F1")
    return {
        "dev_coarse_nll": coarse_nll,
        "dev_fine_nll": fine_nll,
        "coarse_accuracy": coarse_accuracy,
        "fine_accuracy": fine_accuracy,
        "coarse_macro_f1": coarse_macro_f1,
        "fine_macro_f1": fine_macro_f1,
        "coarse_recall": coarse_recall,
        "fine_recall": fine_recall,
    }


def select_coarse_checkpoint(history: Sequence[dict]) -> dict:
    if not history:
        raise HarnessError("coarse checkpoint selection needs a non-empty dev history", 2)
    return min(history, key=lambda row: (row["dev_coarse_nll"], row["step"]))


def select_fine_checkpoint(history: Sequence[dict], arm: str, protocol: dict) -> dict | None:
    if arm not in protocol["training"]["selection"]["fine_estimand"]["arms"]:
        return None
    rows = [row for row in history if row.get("dev_fine_nll") is not None]
    if not rows:
        raise HarnessError("fine checkpoint selection needs a fine dev history", 2)
    return min(rows, key=lambda row: (row["dev_fine_nll"], row["step"]))


def _history_at(history: Sequence[dict], step: int) -> dict:
    for row in history:
        if row["step"] == step:
            return row
    raise HarnessError(f"history lacks the pre-registered step {step}", 2)


def extension_gains(histories: dict, protocol: dict) -> dict:
    extension = protocol["training"]["extension"]
    early = extension["windows"]["early"]
    late = extension["windows"]["late"]
    coarse: dict[str, float] = {}
    fine: dict[str, float] = {}
    for arm in extension["arms"]:
        seeds = histories[arm]
        coarse[arm] = float(np.mean([_history_at(rows, early)["dev_coarse_nll"] - _history_at(rows, late)["dev_coarse_nll"] for rows in seeds.values()]))
        if arm in extension["fine_arms"]:
            fine[arm] = float(np.mean([_history_at(rows, early)["dev_fine_nll"] - _history_at(rows, late)["dev_fine_nll"] for rows in seeds.values()]))
    max_gain = max(list(coarse.values()) + list(fine.values()))
    minimum = float(extension["metric_gain_min"])
    return {
        "coarse": coarse,
        "fine": fine,
        "max_gain": max_gain,
        "min_gain": minimum,
        "triggered": max_gain >= minimum,
        "windows": {"early": early, "late": late},
        "note": "decided before any step beyond the trigger; when not triggered no extension work is performed",
    }
def train_arm(
    arm: str,
    seed: int,
    train_pairs: Sequence[tuple[int, int]],
    dev_pairs: Sequence[tuple[int, int]],
    protocol: dict,
    *,
    updates: int | None = None,
    resume: dict | None = None,
) -> dict:
    if arm not in TRAINABLE_ARMS:
        raise HarnessError(f"unknown arm {arm}", 2)
    torch.set_num_threads(protocol["training"]["cpu_threads"])
    maps = index_maps(protocol)
    spec = protocol["training"]
    eval_every = spec["eval_every"]
    target_updates = int(updates if updates is not None else spec["updates"])
    train_x = torch.as_tensor(encode_pairs(train_pairs, protocol))
    dev_x = torch.as_tensor(encode_pairs(dev_pairs, protocol))
    train_leaf, train_parent = (torch.as_tensor(arr) for arr in label_indices(train_pairs, protocol, maps))
    dev_leaf, dev_parent = (torch.as_tensor(arr) for arr in label_indices(dev_pairs, protocol, maps))
    parent_counts = parent_counts_from_pairs(train_pairs, protocol)
    common = pregenerate_common(seed, len(train_pairs), protocol)
    model = TinyHeadModel(arm, protocol, common["backbone"], head_init(arm, seed, protocol, parent_counts))
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["optimizer"]["lr"])
    if resume is not None:
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
    priors = initial_priors(protocol, parent_counts)
    probe = train_x[: min(256, len(train_pairs))]
    with torch.no_grad():
        probe_output = forward_log(model, probe)
        if arm == "A":
            p5_mean = None
            p3_mean = torch.exp(probe_output["parent_logp"]).mean(dim=0).tolist()
        else:
            p5_mean = torch.exp(probe_output["leaf_logp"]).mean(dim=0).tolist()
            p3_mean = torch.exp(log_group_sums(probe_output["leaf_logp"], maps)).mean(dim=0).tolist()
    initial_q = {
        "theoretical": {"pi_parent": priors["pi_parent"].tolist(), "pi_leaf": priors["pi_leaf"].tolist()},
        "actual_forward_mean": {"p3": p3_mean, "p5": p5_mean},
        "probe_rows": int(probe.shape[0]),
        "source": "resumed-terminal" if resume is not None else "fresh-init",
    }
    history: list[dict] = []
    checkpoints: dict[int, dict] = {}
    start_step = 1 if resume is None else int(resume["next_step"])
    started = time.time()
    for step in range(start_step, target_updates + 1):
        model.train()
        optimizer.zero_grad()
        index = torch.as_tensor(common["batches"][step - 1])
        output = forward_log(model, train_x[index])
        loss = arm_loss(arm, output, train_leaf[index], train_parent[index], protocol)
        _assert_finite(loss, f"{arm}-{seed} training loss at step {step}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                _assert_finite(parameter.grad, f"{arm}-{seed} gradient {name} at step {step}")
        optimizer.step()
        if step % eval_every == 0:
            metrics = dev_metrics(model, dev_x, dev_leaf, dev_parent, protocol)
            history.append({"step": step, **metrics})
            checkpoints[step] = copy.deepcopy(model.state_dict())
    _check_params_finite(model, f"{arm}-{seed} final parameters")
    if not history:
        raise HarnessError(f"{arm}-{seed} produced no dev evaluation points", 1)
    coarse = select_coarse_checkpoint(history)
    fine = select_fine_checkpoint(history, arm, protocol)
    return {
        "arm": arm,
        "seed": seed,
        "updates": target_updates,
        "start_step": start_step,
        "resumed": resume is not None,
        "eval_every": eval_every,
        "history": history,
        "coarse": {"step": coarse["step"], "dev_coarse_nll": coarse["dev_coarse_nll"]},
        "fine": None if fine is None else {"step": fine["step"], "dev_fine_nll": fine["dev_fine_nll"]},
        "checkpoint_states": {
            "coarse": checkpoints[coarse["step"]],
            "fine": None if fine is None else checkpoints[fine["step"]],
        },
        "terminal": {
            "state": copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "step": target_updates,
            "next_batch_index": target_updates,
        },
        "initial_q": initial_q,
        "rng": {
            "backbone_sha256": common["backbone_sha256"],
            "batches_sha256": common["batches_sha256"],
            "streams": common["streams"],
        },
        "eval_calls": len(history),
        "wall_seconds": round(time.time() - started, 3),
        "torch_threads": torch.get_num_threads(),
    }


def build_model_skeleton(arm: str, seed: int, train_pairs: Sequence[tuple[int, int]], protocol: dict) -> TinyHeadModel:
    parent_counts = parent_counts_from_pairs(train_pairs, protocol)
    common = pregenerate_common(seed, len(train_pairs), protocol)
    return TinyHeadModel(arm, protocol, common["backbone"], head_init(arm, seed, protocol, parent_counts))


def load_state_strict(model: TinyHeadModel, state: dict, what: str) -> None:
    expected = model.state_dict()
    if set(state) != set(expected):
        raise HarnessError(f"{what}: state keys do not match arm {model.arm} parameterization", 4)
    for key, tensor in expected.items():
        candidate = state[key]
        if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != tuple(tensor.shape):
            raise HarnessError(f"{what}: shape mismatch at {key}", 4)
        if not bool(torch.isfinite(candidate).all().item()):
            raise HarnessError(f"{what}: non-finite weights at {key}", 4)
    model.load_state_dict(state, strict=True)
    model.eval()


def _log_probs(model: TinyHeadModel, pairs: Sequence[tuple[int, int]], protocol: dict) -> tuple[np.ndarray, np.ndarray | None]:
    x = torch.as_tensor(encode_pairs(pairs, protocol))
    with torch.no_grad():
        output = forward_log(model, x)
        if model.arm == "A":
            logp3 = output["parent_logp"].numpy().astype(np.float64)
            logp5 = None
        else:
            logp5 = output["leaf_logp"].numpy().astype(np.float64)
            logp3 = _np_log_group_sums(logp5, model.maps)
    return logp3, logp5


def _np_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = values - maximum
        total = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return (maximum + total).squeeze(axis)


def _np_log_group_sums(logp5: np.ndarray, maps: dict) -> np.ndarray:
    return np.stack([_np_logsumexp(logp5[:, maps["groups"][parent]], axis=1) for parent in maps["parents"]], axis=1)


def fit_temperature(logp3: np.ndarray, parent_idx: np.ndarray, protocol: dict) -> dict:
    temperature_spec = protocol["evaluation"]["temperature"]
    low, high = temperature_spec["range"]
    grid = np.geomspace(float(low), float(high), int(temperature_spec["grid_points"]))
    logp3 = np.asarray(logp3, dtype=np.float64)
    parent_idx = np.asarray(parent_idx, dtype=np.int64)
    usable = np.isfinite(logp3).all(axis=1)
    usable_count = int(usable.sum())
    if usable_count == 0:
        raise HarnessError("no usable calibration rows for temperature fitting", 1)
    usable_logp3 = logp3[usable]
    usable_gold = parent_idx[usable]
    rows = np.arange(len(usable_gold))

    def nll(values: float) -> float:
        scaled = usable_logp3 / values
        scaled = scaled - _np_logsumexp(scaled, axis=1)[:, None]
        return float(-scaled[rows, usable_gold].mean())

    scored = [(float(t), nll(float(t))) for t in grid]
    best = min(scored, key=lambda pair: (pair[1], pair[0]))
    return {
        "temperature": best[0],
        "nll_before": nll(1.0),
        "nll_after": best[1],
        "grid_points": len(scored),
        "range": [float(low), float(high)],
        "usable_rows": usable_count,
        "excluded_non_finite": int(len(logp3) - usable_count),
    }


def apply_temperature_to_logp3(logp3: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logp3, dtype=np.float64) / float(temperature)
    return scaled - _np_logsumexp(scaled, axis=1)[:, None]


def apply_temperature_to_logp5(logp5: np.ndarray, temperature: float, maps: dict) -> np.ndarray:
    logp3 = _np_log_group_sums(logp5, maps)
    logp3_scaled = apply_temperature_to_logp3(logp3, temperature)
    parent_of_leaf = np.asarray(maps["parent_of_leaf"], dtype=np.int64)
    return logp3_scaled[:, parent_of_leaf] + (logp5 - logp3[:, parent_of_leaf])


def policy_coverage_description(logp3: np.ndarray, parent_idx: np.ndarray, protocol: dict) -> dict:
    coverage_target = float(protocol["evaluation"]["policy_coverage"])
    finite = np.isfinite(logp3).all(axis=1)
    excluded = int((~finite).sum())
    if not bool(finite.any()):
        raise HarnessError("no usable policy rows for the coverage description", 1)
    logp3 = logp3[finite]
    gold = parent_idx[finite]
    probs = np.exp(logp3)
    max_prob = probs.max(axis=1)
    threshold = float(np.quantile(max_prob, 1.0 - coverage_target))
    covered = max_prob >= threshold - 1e-12
    predictions = probs.argmax(axis=1)
    covered_accuracy = float((predictions[covered] == gold[covered]).mean()) if covered.any() else None
    return {
        "n": int(finite.sum()),
        "excluded_non_finite": excluded,
        "coverage_target": coverage_target,
        "threshold": threshold,
        "empirical_coverage": float(covered.mean()),
        "covered_accuracy": covered_accuracy,
        "descriptive_only": True,
        "note": "empirical coverage description; not a risk certificate",
    }
@dataclass(frozen=True)
class EditObject:
    object_id: str
    claim_key: str
    rel_class: str
    content_format: str
    guard_days: int
    readonly_until: str
    condition: str
    ttl: str
    scope: tuple[str, ...]
    evidence: tuple[str, ...]
    lineage: tuple[str, ...]
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class EditLibrary:
    version: int
    objects: tuple[EditObject, ...]


@dataclass(frozen=True)
class EditIntent:
    target_ids: tuple[str, ...]
    allowed_ops: frozenset
    declared_scope: tuple[str, ...]
    read_basis_version: int
    joint_claims: tuple[tuple[str, ...], ...] = ()
    supersede_ref: str | None = None


def _objects_by_id(library: EditLibrary) -> dict[str, EditObject]:
    return {obj.object_id: obj for obj in library.objects}


def _validate_unique_ids(library: EditLibrary, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for obj in library.objects:
        if obj.object_id in seen:
            duplicates.add(obj.object_id)
        seen.add(obj.object_id)
    if duplicates:
        raise HarnessError(f"{label}: duplicate object ids {sorted(duplicates)} (rejected regardless of order)", 2)


def conflict_inventory(library: EditLibrary) -> dict:
    by_claim: dict[str, list[tuple[str, str]]] = {}
    for obj in library.objects:
        by_claim.setdefault(obj.claim_key, []).append((obj.object_id, obj.rel_class))
    inventory = {}
    for claim, members in by_claim.items():
        classes = {rel_class for _, rel_class in members}
        if len(classes) > 1:
            ordered = sorted(members)
            fingerprint = sha256_bytes(";".join(f"{oid}:{rel_class}" for oid, rel_class in ordered).encode("utf-8"))
            inventory[claim] = {
                "claim": claim,
                "fingerprint": fingerprint,
                "members": [{"object_id": oid, "rel_class": rel_class} for oid, rel_class in ordered],
            }
    return inventory


def conflicted_claims(library: EditLibrary) -> set[str]:
    return set(conflict_inventory(library))


def _clauses_satisfiable(clauses: Sequence[Sequence[str]]) -> bool:
    variables = sorted({literal.lstrip("!") for clause in clauses for literal in clause})
    if not variables:
        return not any(len(clause) == 0 for clause in clauses)
    for values in itertools.product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, values))
        if all(any(assignment[literal.lstrip("!")] != literal.startswith("!") for literal in clause) for clause in clauses):
            return True
    return False


def _field_codes(before: EditObject, after: EditObject, intent: EditIntent) -> list[str]:
    codes: list[str] = []
    for field in ("content_format", "guard_days", "readonly_until", "condition", "ttl", "scope", "evidence", "lineage", "dependencies", "rel_class", "claim_key"):
        old = getattr(before, field)
        new = getattr(after, field)
        if old == new:
            continue
        if field == "content_format":
            continue
        if field == "guard_days":
            if new > old and "strengthen" not in intent.allowed_ops:
                codes.append("unauthorized_strengthening")
            if new < old and "weaken" not in intent.allowed_ops:
                codes.append("unauthorized_weakening")
        elif field == "readonly_until":
            if "readonly" not in intent.allowed_ops:
                codes.append("unauthorized_readonly_change")
        elif field == "condition":
            if "condition" not in intent.allowed_ops:
                codes.append("unauthorized_condition_change")
        elif field == "ttl":
            if "ttl" not in intent.allowed_ops:
                codes.append("unauthorized_ttl_change")
        elif field == "scope":
            continue
        elif field in ("evidence", "lineage", "dependencies"):
            if not set(new) >= set(old) and ("drop_refs" not in intent.allowed_ops or not intent.supersede_ref):
                codes.append("evidence_lineage_loss")
        elif field == "rel_class":
            continue
        else:
            codes.append("unmentioned_field_changed")
    return codes


def _class_codes(before: EditLibrary, after: EditLibrary, intent: EditIntent) -> list[str]:
    before_by_id = _objects_by_id(before)
    after_by_id = _objects_by_id(after)
    changed = [
        oid
        for oid in intent.target_ids
        if oid in before_by_id and oid in after_by_id and before_by_id[oid].rel_class != after_by_id[oid].rel_class
    ]
    if not changed:
        return []
    touched_claims: set[str] = set()
    for oid in intent.target_ids:
        if oid in before_by_id:
            touched_claims.add(before_by_id[oid].claim_key)
        if oid in after_by_id:
            touched_claims.add(after_by_id[oid].claim_key)
    codes: list[str] = []
    for claim in sorted(touched_claims):
        pre_classes = {obj.rel_class for obj in before.objects if obj.claim_key == claim}
        post_classes = {obj.rel_class for obj in after.objects if obj.claim_key == claim}
        if len(post_classes) == 1 and len(pre_classes) >= 2 and "merge_classes" not in intent.allowed_ops:
            codes.append("mixed_object_classes")
    if codes:
        return list(dict.fromkeys(codes))
    return [] if "reclass" in intent.allowed_ops else ["undeclared_change"]


def audit_edit(before: EditLibrary, after: EditLibrary, intent: EditIntent) -> dict:
    _validate_unique_ids(before, "before library")
    _validate_unique_ids(after, "after library")
    codes: list[str] = []
    before_by_id = _objects_by_id(before)
    after_by_id = _objects_by_id(after)
    if intent.read_basis_version != before.version:
        codes.append("stale_read_basis" if intent.read_basis_version < before.version else "unknown_read_basis")
    for oid in sorted(set(before_by_id) | set(after_by_id)):
        b = before_by_id.get(oid)
        a = after_by_id.get(oid)
        if b is not None and a is not None and b == a:
            continue
        if oid not in intent.target_ids:
            codes.append("undeclared_change")
            continue
        if b is None:
            if "create" not in intent.allowed_ops:
                codes.append("undeclared_change")
            continue
        if a is None:
            if "delete" not in intent.allowed_ops:
                codes.append("undeclared_change")
            continue
        codes.extend(_field_codes(b, a, intent))
    codes.extend(_class_codes(before, after, intent))
    for oid in intent.target_ids:
        target = after_by_id.get(oid)
        if target is not None and not set(target.scope) <= set(intent.declared_scope):
            codes.append("scope_escalation")
    if intent.joint_claims and not _clauses_satisfiable(intent.joint_claims):
        codes.append("joint_inconsistent")
    before_conflicts = conflict_inventory(before)
    after_conflicts = conflict_inventory(after)
    introduced = [after_conflicts[claim] for claim in sorted(after_conflicts) if claim not in before_conflicts or before_conflicts[claim]["fingerprint"] != after_conflicts[claim]["fingerprint"]]
    retained = [after_conflicts[claim] for claim in sorted(after_conflicts) if claim in before_conflicts and before_conflicts[claim]["fingerprint"] == after_conflicts[claim]["fingerprint"]]
    resolved = [before_conflicts[claim] for claim in sorted(before_conflicts) if claim not in after_conflicts]
    if introduced:
        codes.append("introduced_conflict")
    ordered = list(dict.fromkeys(codes))
    return {
        "codes": ordered,
        "pre_existing_conflicts": retained,
        "introduced_conflicts": introduced,
        "resolved_conflicts": resolved,
    }


def _object(oid: str, claim: str, rel_class: str) -> EditObject:
    return EditObject(
        object_id=oid,
        claim_key=claim,
        rel_class=rel_class,
        content_format="csv",
        guard_days=30,
        readonly_until="2026-12-31",
        condition="always",
        ttl="-",
        scope=("project-1",),
        evidence=("ev-1",),
        lineage=("ln-1",),
        dependencies=("dep-1",),
    )


def _base_objects() -> dict[str, EditObject]:
    return {
        "obj_alpha": _object("obj_alpha", "claim-A", "equivalent"),
        "obj_beta": _object("obj_beta", "claim-B", "overlap_incomparable"),
        "obj_beta2": _object("obj_beta2", "claim-B", "overlap_incomparable"),
        "obj_gamma": _object("obj_gamma", "claim-C", "equivalent"),
        "obj_delta": _object("obj_delta", "claim-C", "disjoint"),
    }


def _library(objects: dict[str, EditObject], version: int = 7) -> EditLibrary:
    return EditLibrary(version=version, objects=tuple(objects.values()))


def _replace(library: EditLibrary, updated: dict[str, EditObject]) -> EditLibrary:
    return EditLibrary(version=library.version, objects=tuple(updated.get(obj.object_id, obj) for obj in library.objects))


def _intent(target_ids: tuple[str, ...], **kwargs) -> EditIntent:
    defaults = {
        "allowed_ops": frozenset(),
        "declared_scope": ("project-1",),
        "read_basis_version": 7,
    }
    defaults.update(kwargs)
    return EditIntent(target_ids=tuple(target_ids), **defaults)


def _single_field_case(oid: str, field: str, value, *, targets: tuple[str, ...] | None = None, **intent_kwargs):
    before = _library(_base_objects())
    updated = replace(_objects_by_id(before)[oid], **{field: value})
    after = _replace(before, {oid: updated})
    return before, after, _intent(tuple(targets or (oid,)), **intent_kwargs)


def _case_content_format():
    return _single_field_case("obj_alpha", "content_format", "tsv")


def _case_guard_90():
    return _single_field_case("obj_alpha", "guard_days", 90)


def _case_guard_7():
    return _single_field_case("obj_alpha", "guard_days", 7)


def _case_readonly_shortened():
    return _single_field_case("obj_alpha", "readonly_until", "2026-10-20")


def _case_condition_edit():
    return _single_field_case("obj_alpha", "condition", "region=eu")


def _case_ttl_edit():
    return _single_field_case("obj_alpha", "ttl", "30d")


def _case_scope_escalation():
    return _single_field_case("obj_alpha", "scope", ("project-1", "project-2"))


def _case_stale_basis():
    return _single_field_case("obj_alpha", "content_format", "tsv", read_basis_version=6)


def _case_future_basis():
    return _single_field_case("obj_alpha", "content_format", "tsv", read_basis_version=8)


def _case_joint_inconsistent():
    before, after, _ = _single_field_case("obj_alpha", "content_format", "tsv")
    return before, after, _intent(("obj_alpha",), joint_claims=(("a", "b"), ("!a",), ("!b",)))


def _mixed_library() -> EditLibrary:
    objects = _base_objects()
    objects["obj_m1"] = _object("obj_m1", "claim-M", "equivalent")
    objects["obj_m2"] = _object("obj_m2", "claim-M", "disjoint")
    return _library(objects)


def _case_mixed_classes():
    before = _mixed_library()
    after = _replace(before, {"obj_m2": replace(_objects_by_id(before)["obj_m2"], rel_class="equivalent")})
    return before, after, _intent(("obj_m1", "obj_m2"))


def _case_mixed_single_target():
    before = _mixed_library()
    after = _replace(before, {"obj_m2": replace(_objects_by_id(before)["obj_m2"], rel_class="equivalent")})
    return before, after, _intent(("obj_m2",))


def _case_evidence_dropped():
    return _single_field_case("obj_alpha", "evidence", ())


def _case_lineage_dropped():
    return _single_field_case("obj_alpha", "lineage", ())


def _case_dependencies_dropped():
    return _single_field_case("obj_alpha", "dependencies", ())


def _case_unmentioned_object():
    return _single_field_case("obj_beta", "guard_days", 90, targets=("obj_alpha",))


def _case_unmentioned_target_field():
    return _single_field_case("obj_alpha", "claim_key", "claim-A2")


def _case_introduced_conflict():
    before = _library(_base_objects())
    after = _replace(before, {"obj_beta": replace(_objects_by_id(before)["obj_beta"], rel_class="disjoint")})
    return before, after, _intent(("obj_beta",), allowed_ops=frozenset({"reclass"}))


def _case_third_conflict():
    before = _library(_base_objects())
    added = _object("obj_epsilon", "claim-C", "new_entails_old_strict")
    after = EditLibrary(version=before.version, objects=tuple(before.objects) + (added,))
    return before, after, _intent(("obj_epsilon",), allowed_ops=frozenset({"create"}))


MUTATIONS = {
    "content_format_csv_to_tsv": _case_content_format,
    "guard_30_to_90": _case_guard_90,
    "guard_30_to_7": _case_guard_7,
    "readonly_until_shortened": _case_readonly_shortened,
    "condition_edit": _case_condition_edit,
    "ttl_edit": _case_ttl_edit,
    "scope_escalation": _case_scope_escalation,
    "stale_read_basis": _case_stale_basis,
    "future_read_basis": _case_future_basis,
    "joint_inconsistent": _case_joint_inconsistent,
    "mixed_object_classes": _case_mixed_classes,
    "mixed_class_single_target": _case_mixed_single_target,
    "evidence_ref_dropped": _case_evidence_dropped,
    "lineage_ref_dropped": _case_lineage_dropped,
    "dependency_ref_dropped": _case_dependencies_dropped,
    "unmentioned_object_change": _case_unmentioned_object,
    "unmentioned_target_field": _case_unmentioned_target_field,
    "introduced_conflict": _case_introduced_conflict,
    "old_claim_third_conflict": _case_third_conflict,
}


def _check_case_asserts(name: str, before, after, report) -> bool:
    before_alpha = _objects_by_id(before)["obj_alpha"]
    after_alpha = _objects_by_id(after)["obj_alpha"]
    if name == "readonly_until_preserved":
        return after_alpha.readonly_until == before_alpha.readonly_until == "2026-12-31"
    if name == "guard_days_preserved":
        return after_alpha.guard_days == before_alpha.guard_days == 30
    if name == "pre_existing_conflicts_reported_separately":
        return any(entry["claim"] == "claim-C" for entry in report["pre_existing_conflicts"])
    if name == "no_introduced_conflicts":
        return not report["introduced_conflicts"]
    raise HarnessError(f"unknown audit case assertion {name}", 2)


def run_edit_audit(protocol: dict) -> dict:
    catalog = protocol["edit_audit"]
    results = []
    for case in catalog["cases"]:
        builder = MUTATIONS.get(case["mutation"])
        if builder is None:
            raise HarnessError(f"protocol references unknown mutation {case['mutation']}", 2)
        before, after, intent = builder()
        report = audit_edit(before, after, intent)
        actual = sorted(report["codes"])
        expected = sorted(case["expected_codes"])
        asserts_ok = all(_check_case_asserts(name, before, after, report) for name in case.get("asserts", []))
        results.append({
            "id": case["id"],
            "mutation": case["mutation"],
            "expected_codes": expected,
            "actual_codes": actual,
            "asserts": list(case.get("asserts", [])),
            "passed": actual == expected and asserts_ok,
            "pre_existing_conflicts": report["pre_existing_conflicts"],
            "introduced_conflicts": report["introduced_conflicts"],
            "resolved_conflicts": report["resolved_conflicts"],
        })
    expected_union = set().union(*[set(case["expected_codes"]) for case in catalog["cases"]]) if catalog["cases"] else set()
    uncovered = sorted(set(catalog["codes"]) - expected_union)
    return {
        "cases": results,
        "total": len(results),
        "passed": sum(1 for row in results if row["passed"]),
        "failed": [row["id"] for row in results if not row["passed"]],
        "codes_uncovered": uncovered,
    }
def _check_fresh_out(out_path, *, allow_external: bool = False) -> Path:
    out = Path(out_path)
    if out.exists():
        raise HarnessError(f"refusing to overwrite existing run directory: {out}", 2)
    if not allow_external:
        root = runs_root().resolve()
        target = out.resolve()
        if root not in target.parents:
            raise HarnessError(f"refusing out outside the ignored runs root {root}: {out} (pass --allow-external-out to override)", 2)
    return out


def _create_out(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    return out


def _artifact_hashes(directory: Path, relpaths: Iterable[str]) -> dict:
    return {rel: sha256_file(directory / rel) for rel in relpaths}


def _read_exposure(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append((int(row["p"]), int(row["h"])))
    return pairs


def expected_exposure(protocol: dict, pool: str) -> list[tuple[int, int]]:
    built = build_partition(protocol)
    return ordered_exposures(built["entity_pools"][pool])


def load_prepare_context(prepare_dir, protocol_path=None) -> dict:
    prepare_dir = Path(prepare_dir)
    manifest_path = prepare_dir / "manifest.json"
    if not manifest_path.exists():
        raise HarnessError(f"prepare manifest missing: {manifest_path}", 2)
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != "prepare":
        raise HarnessError(f"not a prepare directory: {prepare_dir}", 2)
    artifacts = manifest.get("artifacts") or {}
    missing = [rel for rel in MANDATORY_PREPARE_ARTIFACTS if rel not in artifacts]
    if missing:
        raise HarnessError(f"prepare manifest lacks mandatory artifacts: {missing}", 4)
    for rel, expected in artifacts.items():
        path = prepare_dir / rel
        if not path.exists() or sha256_file(path) != expected:
            raise HarnessError(f"prepare artifact hash mismatch: {rel}", 4)
    for pool, row in (manifest.get("pools") or {}).items():
        if row.get("exported"):
            rel = f"exposures/{pool}.jsonl"
            if artifacts.get(rel) != row.get("exposure_sha256"):
                raise HarnessError(f"prepare pool/artifact exposure SHA mismatch: {pool}", 4)
    frozen = _read_json(prepare_dir / "protocol-frozen.json")
    if frozen.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise HarnessError("frozen protocol identity mismatch with prepare manifest", 4)
    if manifest.get("protocol_frozen_sha256") != artifacts.get("protocol-frozen.json"):
        raise HarnessError("frozen protocol file hash mismatch with prepare manifest", 4)
    repo_protocol = Path(protocol_path) if protocol_path else protocol_default_path()
    if sha256_file(repo_protocol) != manifest["protocol_sha256"]:
        raise HarnessError("repo protocol bytes changed since prepare; re-prepare required", 4)
    protocol = frozen["content"]
    validate_protocol(protocol)
    if _canonical(protocol) != _canonical(json.loads(repo_protocol.read_text(encoding="utf-8"))):
        raise HarnessError("frozen protocol content does not equal the verified repo protocol", 4)
    if sha256_file(Path(__file__)) != manifest.get("source_sha256"):
        raise HarnessError("harness source changed since prepare; re-prepare required", 4)
    return {"protocol": protocol, "manifest": manifest, "frozen": frozen, "prepare_dir": prepare_dir}


def _read_verified_exposure(prepare_dir: Path, manifest: dict, rel: str, expected_pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    expected_sha = (manifest.get("artifacts") or {}).get(rel)
    if expected_sha is None:
        raise HarnessError(f"prepare manifest lacks {rel}", 4)
    path = prepare_dir / rel
    if not path.exists() or sha256_file(path) != expected_sha:
        raise HarnessError(f"exposure hash mismatch: {rel}", 4)
    pairs = _read_exposure(path)
    if pairs != list(expected_pairs):
        raise HarnessError(f"{rel} does not equal the deterministic expected exposure (order/length/duplicates checked)", 4)
    return pairs


def preflight_train_dev(prepare_dir, protocol_path=None) -> tuple[dict, list[tuple[int, int]], list[tuple[int, int]]]:
    context = load_prepare_context(prepare_dir, protocol_path)
    manifest = context["manifest"]
    protocol = context["protocol"]
    train_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/train.jsonl", expected_exposure(protocol, "train"))
    dev_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/dev.jsonl", expected_exposure(protocol, "dev"))
    return context, train_pairs, dev_pairs


def stage_formal(out_path, protocol_path=None, *, allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    protocol, protocol_sha = load_protocol(protocol_path)
    census = exhaustive_census(protocol)
    failed = (
        not census["matches_expected"]
        or census["swap_violations"]
        or census["leaf_parent_violations"]
        or census["conservation_violations"]
        or census["illegal_combinations"]
        or census["combination_mismatches"]
    )
    if failed:
        raise HarnessError("formal census failed; refusing to write results", 1)
    unresolved_cases = {
        "unknown_input": (classify(None, 5), "input_unknown"),
        "empty_input": (classify(0, 5), "empty_satisfier"),
        "inconsistent_background": (classify(3, 5, background_consistent=False), "background_mismatch"),
        "condition_overlap_unknown": (conditional_full_relation(3, 5, 1, 3, overlap_known=False), "condition_overlap_unknown"),
        "empty_condition_overlap": (conditional_full_relation(3, 5, 1, 2), "empty_condition_overlap"),
    }
    recorded_unresolved = {}
    for name, (verdict, reason) in unresolved_cases.items():
        if verdict.status != "unresolved" or verdict.leaf is not None or verdict.reason != reason:
            raise HarnessError(f"unresolved case {name} did not stay single-column: {verdict}", 1)
        recorded_unresolved[name] = {"status": verdict.status, "leaf": verdict.leaf, "reason": verdict.reason}
    conditional_full = conditional_full_relation(3, 1, 1, 3)
    conditional_interval = interval_content_relation(3, 1, 1, 3)
    if conditional_full.leaf == conditional_interval.leaf or None in (conditional_full.leaf, conditional_interval.leaf):
        raise HarnessError("conditional full/interval non-equivalence fixture collapsed", 1)
    if {"supersede", "temporary"} & set(protocol["formal"]["leaves"]):
        raise HarnessError("lifecycle categories must not be mixed into the leaf layer", 1)
    joint_demo = joint_marginals_feasible([0.45, 0.15, 0.40], [0.33, 0.27, 0.40], protocol)
    if not joint_demo:
        raise HarnessError("joint feasibility fixture for the legal EC-marginal example collapsed", 1)
    audit = run_edit_audit(protocol)
    if audit["failed"] or audit["codes_uncovered"]:
        raise HarnessError(f"edit audit failed: {audit['failed']} uncovered={audit['codes_uncovered']}", 1)
    formal = {
        "stage": "formal",
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha,
        "source_sha256": sha256_file(Path(__file__)),
        "created_at": _now_iso(),
        "duration_seconds": round(time.time() - started, 3),
        "runtime": runtime_info(),
        "census": census,
        "unresolved_checks": recorded_unresolved,
        "conditional_non_equivalence": {
            "example": {"p": 3, "h": 1, "cond_p": 1, "cond_h": 3},
            "full_conditional_relation": conditional_full.leaf,
            "interval_content_relation": conditional_interval.leaf,
            "equivalent": False,
        },
        "joint_marginal_demo": {
            "example_marginals": {"a": [0.45, 0.15, 0.40], "b": [0.33, 0.27, 0.40]},
            "map_labels": ["E", "C"],
            "joint_feasible": joint_demo,
            "note": "hard combo EC is unmappable while the same marginals are jointly feasible; the three notions stay separate",
        },
        "edit_audit": audit,
        "boundary": protocol["boundary"],
    }
    _create_out(out)
    formal_sha = _write_json(out / "formal.json", formal)
    manifest = {
        "stage": "formal",
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha,
        "source_sha256": formal["source_sha256"],
        "created_at": formal["created_at"],
        "duration_seconds": formal["duration_seconds"],
        "runtime": formal["runtime"],
        "updates": None,
        "steps": None,
        "inference_calls": {"dev_forward_passes": 0},
        "denominators": {"universe_pairs": census["pair_total"]},
        "budget": {"updates": None, "steps": None, "duration_seconds": formal["duration_seconds"]},
        "artifacts": {"formal.json": formal_sha},
    }
    _write_json(out / "manifest.json", manifest)
    return {
        "stage": "formal",
        "out": str(out),
        "formal_sha256": formal_sha,
        "pair_total": census["pair_total"],
        "leaf_counts": census["leaf_counts"],
        "parent_counts": census["parent_counts"],
        "edit_audit_passed": audit["passed"],
        "edit_audit_total": audit["total"],
        "duration_seconds": formal["duration_seconds"],
    }


def stage_prepare(out_path, protocol_path=None, *, allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    protocol, protocol_sha = load_protocol(protocol_path)
    report = partition_report(protocol)
    created = _now_iso()
    source_sha = sha256_file(Path(__file__))
    frozen = {
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "frozen_at": created,
        "content": protocol,
        "sealing": "finaltoy exposure content is not exported; the generator stays public and sealing is a process commitment, not information-theoretic secrecy",
    }
    _create_out(out)
    _write_json(out / "protocol-frozen.json", frozen)
    partition_payload = {
        "seed": report["seed"],
        "stratum_pairs": report["stratum_pairs"],
        "pools": report["pools"],
        "denominators": report["denominators"],
        "distribution_target": report["distribution_target"],
    }
    _write_json(out / "partition.json", partition_payload)
    relpaths = ["protocol-frozen.json", "partition.json"]
    for pool, row in report["pools"].items():
        if row["exported"]:
            rel = f"exposures/{pool}.jsonl"
            _write_bytes(out / rel, exposure_jsonl(ordered_exposures(row["entities"])))
            relpaths.append(rel)
    artifacts = _artifact_hashes(out, relpaths)
    for rel in relpaths:
        if not rel.startswith("exposures/"):
            continue
        pool = Path(rel).stem
        rows = (out / rel).read_bytes().count(b"\n")
        if rows != report["pools"][pool]["pairs"]:
            raise HarnessError(f"written exposure row count mismatch for {pool}", 1)
    pools_summary = {}
    for pool, row in report["pools"].items():
        entry = {key: row[key] for key in ("entity_count", "entity_sha256", "pairs", "sealed", "exported", "stats")}
        entry["exposure_sha256"] = artifacts[f"exposures/{pool}.jsonl"] if row["exported"] else row["exposure_sha256"]
        pools_summary[pool] = entry
    duration = round(time.time() - started, 3)
    manifest = {
        "stage": "prepare",
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": protocol_sha,
        "protocol_frozen_sha256": artifacts["protocol-frozen.json"],
        "source_sha256": source_sha,
        "created_at": created,
        "duration_seconds": duration,
        "runtime": runtime_info(),
        "seed": report["seed"],
        "pools": pools_summary,
        "denominators": report["denominators"],
        "distribution_target": report["distribution_target"],
        "updates": None,
        "steps": None,
        "budget": {"updates": None, "steps": None, "eval_every": protocol["training"]["eval_every"], "duration_seconds": duration},
        "artifacts": artifacts,
    }
    manifest_sha = _write_json(out / "manifest.json", manifest)
    return {
        "stage": "prepare",
        "out": str(out),
        "manifest_sha256": manifest_sha,
        "pools": {pool: {"entity_count": row["entity_count"], "pairs": row["pairs"], "sealed": row["sealed"]} for pool, row in pools_summary.items()},
        "denominators": report["denominators"],
        "duration_seconds": duration,
    }


def stage_train_dev(
    out_path,
    prepare_dir,
    protocol_path=None,
    *,
    allow_external: bool = False,
    updates: int | None = None,
    seeds: Sequence[int] | None = None,
) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    context, train_pairs, dev_pairs = preflight_train_dev(prepare_dir, protocol_path)
    protocol = context["protocol"]
    manifest = context["manifest"]
    spec = protocol["training"]
    target_updates = int(updates if updates is not None else spec["updates"])
    if target_updates <= 0 or target_updates % spec["eval_every"] != 0:
        raise HarnessError("updates override must be a positive multiple of eval_every", 2)
    seed_list = [int(seed) for seed in (seeds if seeds is not None else spec["seeds"])]
    if not seed_list:
        raise HarnessError("at least one seed is required", 2)
    overrides = {"updates": updates is not None, "seeds": seeds is not None}
    formal_budget = (not any(overrides.values())) and target_updates == spec["updates"]
    runs: dict[str, dict] = {}
    _create_out(out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "terminal").mkdir(parents=True, exist_ok=True)
    extension: dict = {
        "applicable": formal_budget,
        "triggered": None,
        "gains": None,
        "continued": False,
        "keeps_1500_results": spec["extension"]["keeps_1500_results"],
        "reason": None if formal_budget else "budget override: the pre-registered extension decision applies to the formal 1500-update grid only",
    }
    for seed in seed_list:
        for arm in TRAINABLE_ARMS:
            run = train_arm(arm, seed, train_pairs, dev_pairs, protocol, updates=target_updates)
            histories = run.pop("history")
            checkpoint_states = run.pop("checkpoint_states")
            terminal = run.pop("terminal")
            models: dict[str, dict] = {}
            coarse_path = out / "models" / f"{arm}-{seed}-coarse.pt"
            torch.save(checkpoint_states["coarse"], coarse_path)
            models["coarse"] = {"path": str(coarse_path.relative_to(out)).replace("\\", "/"), "sha256": sha256_file(coarse_path), "step": run["coarse"]["step"]}
            if checkpoint_states["fine"] is not None:
                fine_path = out / "models" / f"{arm}-{seed}-fine.pt"
                torch.save(checkpoint_states["fine"], fine_path)
                models["fine"] = {"path": str(fine_path.relative_to(out)).replace("\\", "/"), "sha256": sha256_file(fine_path), "step": run["fine"]["step"]}
            else:
                models["fine"] = None
            terminal_payload = {
                "model_state": terminal["state"],
                "optimizer_state": terminal["optimizer_state"],
                "step": terminal["step"],
                "next_batch_index": terminal["next_batch_index"],
            }
            terminal_path = out / "terminal" / f"{arm}-{seed}.pt"
            torch.save(terminal_payload, terminal_path)
            runs[f"{arm}-{seed}"] = {
                **run,
                "history": histories,
                "models": models,
                "terminal": {
                    "path": str(terminal_path.relative_to(out)).replace("\\", "/"),
                    "sha256": sha256_file(terminal_path),
                    "step": terminal["step"],
                    "next_batch_index": terminal["next_batch_index"],
                },
            }
    if formal_budget:
        histories = {arm: {seed: runs[f"{arm}-{seed}"]["history"] for seed in seed_list} for arm in TRAINABLE_ARMS}
        gains = extension_gains(histories, protocol)
        extension["gains"] = gains
        extension["triggered"] = bool(gains["triggered"])
        if gains["triggered"]:
            for seed in seed_list:
                for arm in TRAINABLE_ARMS:
                    key = f"{arm}-{seed}"
                    saved = torch.load(out / runs[key]["terminal"]["path"], map_location="cpu")
                    resume_payload = {
                        "model_state": saved["model_state"],
                        "optimizer_state": saved["optimizer_state"],
                        "next_step": int(saved["step"]) + 1,
                    }
                    extended = train_arm(arm, seed, train_pairs, dev_pairs, protocol, updates=spec["extension"]["target_updates"], resume=resume_payload)
                    extended_histories = extended.pop("history")
                    extended.pop("checkpoint_states")
                    extended_terminal = extended.pop("terminal")
                    path = out / "terminal" / f"{arm}-{seed}-extended.pt"
                    torch.save({"model_state": extended_terminal["state"], "optimizer_state": extended_terminal["optimizer_state"], "step": extended_terminal["step"], "next_batch_index": extended_terminal["next_batch_index"]}, path)
                    runs[key]["extended"] = {
                        **extended,
                        "history": extended_histories,
                        "terminal": {"path": str(path.relative_to(out)).replace("\\", "/"), "sha256": sha256_file(path), "step": extended_terminal["step"], "next_batch_index": extended_terminal["next_batch_index"]},
                    }
            extension["continued"] = True
    runs_payload = {
        "stage": "train-dev",
        "created_at": _now_iso(),
        "protocol_sha256": manifest["protocol_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "exposures": {
            "train": {"sha256": manifest["pools"]["train"]["exposure_sha256"], "rows": manifest["pools"]["train"]["pairs"]},
            "dev": {"sha256": manifest["pools"]["dev"]["exposure_sha256"], "rows": manifest["pools"]["dev"]["pairs"]},
        },
        "selection": spec["selection"],
        "fine_diagnostic_arms": ["B"],
        "extension": extension,
        "runs": runs,
    }
    runs_sha = _write_json(out / "runs.json", runs_payload)
    refs = ["runs.json"]
    for run in runs.values():
        for view in ("coarse", "fine"):
            if run["models"].get(view):
                refs.append(run["models"][view]["path"])
        refs.append(run["terminal"]["path"])
        if "extended" in run:
            refs.append(run["extended"]["terminal"]["path"])
    artifacts = _artifact_hashes(out, refs)
    duration = round(time.time() - started, 3)
    manifest_out = {
        "stage": "train-dev",
        "protocol_sha256": manifest["protocol_sha256"],
        "source_sha256": runs_payload["source_sha256"],
        "created_at": runs_payload["created_at"],
        "duration_seconds": duration,
        "runtime": runtime_info(),
        "updates": target_updates,
        "steps": [step for step in range(spec["eval_every"], target_updates + 1, spec["eval_every"])],
        "inference_calls": {
            "optimizer_updates": sum(run["updates"] + (run["extended"]["updates"] if "extended" in run else 0) for run in runs.values()),
            "dev_forward_batches": sum(run["eval_calls"] + (run["extended"]["eval_calls"] if "extended" in run else 0) for run in runs.values()),
        },
        "denominators": {"train_pairs": len(train_pairs), "dev_pairs": len(dev_pairs)},
        "budget": {"updates": target_updates, "seeds": seed_list, "overrides": overrides, "formal": formal_budget, "runs": len(runs), "duration_seconds": duration},
        "artifacts": artifacts,
    }
    manifest_sha = _write_json(out / "manifest.json", manifest_out)
    return {
        "stage": "train-dev",
        "out": str(out),
        "manifest_sha256": manifest_sha,
        "budget": manifest_out["budget"],
        "extension": extension,
        "runs": {key: {"coarse": run["coarse"], "fine": run["fine"], "model_sha256": run["models"]["coarse"]["sha256"]} for key, run in runs.items()},
        "duration_seconds": duration,
    }
def _run_file_refs(run: dict) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for view in ("coarse", "fine"):
        entry = (run.get("models") or {}).get(view)
        if entry:
            refs.append((entry["path"], entry["sha256"]))
    refs.append((run["terminal"]["path"], run["terminal"]["sha256"]))
    if "extended" in run:
        refs.append((run["extended"]["terminal"]["path"], run["extended"]["terminal"]["sha256"]))
    return refs


def load_train_dev_context(prepare_context: dict, train_dev_dir) -> dict:
    train_dev_dir = Path(train_dev_dir)
    manifest_path = train_dev_dir / "manifest.json"
    if not manifest_path.exists():
        raise HarnessError(f"train-dev manifest missing: {manifest_path}", 2)
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != "train-dev":
        raise HarnessError(f"not a train-dev directory: {train_dev_dir}", 2)
    if manifest.get("protocol_sha256") != prepare_context["manifest"]["protocol_sha256"]:
        raise HarnessError("train-dev does not bind this prepare protocol", 4)
    if manifest.get("source_sha256") != sha256_file(Path(__file__)):
        raise HarnessError("harness source changed since train-dev; re-run train-dev", 4)
    artifacts = manifest.get("artifacts") or {}
    if "runs.json" not in artifacts:
        raise HarnessError("train-dev manifest lacks runs.json", 4)
    for rel, expected in artifacts.items():
        path = train_dev_dir / rel
        if not path.exists() or sha256_file(path) != expected:
            raise HarnessError(f"train-dev artifact hash mismatch: {rel}", 4)
    runs_payload = _read_json(train_dev_dir / "runs.json")
    if runs_payload.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise HarnessError("runs.json protocol binding mismatch", 4)
    for key, run in (runs_payload.get("runs") or {}).items():
        for rel, sha in _run_file_refs(run):
            if artifacts.get(rel) != sha:
                raise HarnessError(f"train-dev artifact list inconsistent for {key}: {rel}", 4)
    return {
        "dir": train_dev_dir,
        "manifest": manifest,
        "runs": runs_payload,
        "manifest_sha256": sha256_file(manifest_path),
        "runs_sha256": artifacts["runs.json"],
    }


def verify_selections(runs: dict, protocol: dict) -> dict:
    report = {}
    for key in sorted(runs):
        run = runs[key]
        history = run["history"]
        coarse = select_coarse_checkpoint(history)
        if coarse["step"] != run["coarse"]["step"] or abs(coarse["dev_coarse_nll"] - run["coarse"]["dev_coarse_nll"]) > 1e-9:
            raise HarnessError(f"train-dev selection mismatch for {key}: coarse", 4)
        fine = select_fine_checkpoint(history, run["arm"], protocol)
        recorded = run.get("fine")
        if (fine is None) != (recorded is None):
            raise HarnessError(f"train-dev selection mismatch for {key}: fine presence", 4)
        if fine is not None and (fine["step"] != recorded["step"] or abs(fine["dev_fine_nll"] - recorded["dev_fine_nll"]) > 1e-9):
            raise HarnessError(f"train-dev selection mismatch for {key}: fine", 4)
        report[key] = {"coarse": run["coarse"], "fine": recorded}
    return report


def verify_model_file(arm: str, seed: int, rel_path: str, expected_sha: str, train_dev_dir: Path, train_pairs, protocol: dict) -> TinyHeadModel:
    path = train_dev_dir / rel_path
    if not path.exists():
        raise HarnessError(f"model file missing: {rel_path}", 4)
    if sha256_file(path) != expected_sha:
        raise HarnessError(f"model weights hash mismatch: {rel_path}", 4)
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise HarnessError(f"model file is not a state dict: {rel_path}", 4)
    model = build_model_skeleton(arm, seed, train_pairs, protocol)
    load_state_strict(model, state, rel_path)
    return model


def final_access_identity(context: dict) -> str:
    manifest = context["manifest"]
    return sha256_bytes(_canonical({
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_frozen_sha256": manifest["protocol_frozen_sha256"],
        "source_sha256": manifest["source_sha256"],
        "artifacts": manifest["artifacts"],
    }).encode("utf-8"))


def _row_at(history, step: int) -> dict:
    return _history_at(history, step)


def candidate_views_from_runs(runs: dict, protocol: dict) -> list[dict]:
    views: list[dict] = []
    for key in sorted(runs):
        run = runs[key]
        arm = run["arm"]
        seed = int(run["seed"])
        coarse_row = _row_at(run["history"], run["coarse"]["step"])
        views.append({
            "candidate": f"{arm}-{seed}-coarse",
            "arm": arm,
            "seed": seed,
            "view": "coarse",
            "step": run["coarse"]["step"],
            "model_path": run["models"]["coarse"]["path"],
            "model_sha256": run["models"]["coarse"]["sha256"],
            "dev_coarse_nll": run["coarse"]["dev_coarse_nll"],
            "dev_fine_nll_at_step": coarse_row.get("dev_fine_nll"),
        })
        if run["models"].get("fine"):
            fine_row = _row_at(run["history"], run["fine"]["step"])
            views.append({
                "candidate": f"{arm}-{seed}-fine",
                "arm": arm,
                "seed": seed,
                "view": "fine",
                "step": run["fine"]["step"],
                "model_path": run["models"]["fine"]["path"],
                "model_sha256": run["models"]["fine"]["sha256"],
                "dev_fine_nll": run["fine"]["dev_fine_nll"],
                "dev_coarse_nll_at_step": fine_row.get("dev_coarse_nll"),
            })
    return views


def evaluate_architecture_rule(runs: dict, protocol: dict) -> dict:
    rule = protocol["training"]["architecture_rule"]
    default_arm = rule["default"]
    challenger = rule["challenger"]
    seeds = sorted({int(run["seed"]) for run in runs.values()})
    paired_seeds = [seed for seed in seeds if f"{challenger}-{seed}" in runs and f"{default_arm}-{seed}" in runs]
    if len(paired_seeds) != len(seeds) or len(seeds) < 3:
        return {
            "recommended": default_arm,
            "reason": "insufficient paired seeds for the pre-registered rule; default kept",
            "paired_seeds": paired_seeds,
            "criteria": None,
            "paired_differences": [],
            "scope": rule["scope"],
        }
    paired = []
    for seed in paired_seeds:
        c_run = runs[f"{default_arm}-{seed}"]
        h_run = runs[f"{challenger}-{seed}"]
        c_fine = c_run["fine"]["dev_fine_nll"]
        h_fine = h_run["fine"]["dev_fine_nll"]
        c_coarse_at = _row_at(c_run["history"], c_run["fine"]["step"])["dev_coarse_nll"]
        h_coarse_at = _row_at(h_run["history"], h_run["fine"]["step"])["dev_coarse_nll"]
        paired.append({
            "seed": seed,
            "c_fine_nll": c_fine,
            "h_fine_nll": h_fine,
            "fine_improvement": c_fine - h_fine,
            "c_coarse_nll_at_fine_step": c_coarse_at,
            "h_coarse_nll_at_fine_step": h_coarse_at,
            "coarse_delta": h_coarse_at - c_coarse_at,
        })
    mean_c_fine = float(np.mean([row["c_fine_nll"] for row in paired]))
    mean_improvement = float(np.mean([row["fine_improvement"] for row in paired]))
    threshold = max(
        float(rule["min_fine_improvement"]["absolute_nats"]),
        float(rule["min_fine_improvement"]["relative_to_c_fine_nll"]) * mean_c_fine,
    )
    all_positive = all(row["fine_improvement"] > 0 for row in paired)
    mean_coarse_delta = float(np.mean([row["coarse_delta"] for row in paired]))
    coarse_ok = mean_coarse_delta <= float(rule["max_coarse_degradation_nats"])
    criteria = {
        "mean_fine_improvement": mean_improvement,
        "threshold": threshold,
        "all_seeds_positive": all_positive,
        "mean_coarse_delta": mean_coarse_delta,
        "coarse_degradation_ok": coarse_ok,
        "same_checkpoint": rule["same_checkpoint_comparison"],
    }
    recommended = challenger if (mean_improvement >= threshold and all_positive and coarse_ok) else default_arm
    return {
        "recommended": recommended,
        "reason": "pre-registered criteria applied to the paired fine-estimand comparison",
        "paired_seeds": paired_seeds,
        "criteria": criteria,
        "paired_differences": paired,
        "scope": rule["scope"],
    }


def stage_freeze(out_path, prepare_dir, train_dev_dir, *, protocol_path=None, allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    context = load_prepare_context(prepare_dir, protocol_path)
    protocol = context["protocol"]
    td = load_train_dev_context(context, train_dev_dir)
    runs = td["runs"]["runs"]
    if not runs:
        raise HarnessError("train-dev run set is empty; refusing to freeze", 3)
    verify_selections(runs, protocol)
    train_pairs = _read_verified_exposure(context["prepare_dir"], context["manifest"], "exposures/train.jsonl", expected_exposure(protocol, "train"))
    loaded_models = 0
    for key in sorted(runs):
        run = runs[key]
        for view in ("coarse", "fine"):
            entry = run["models"].get(view)
            if entry:
                verify_model_file(run["arm"], int(run["seed"]), entry["path"], entry["sha256"], td["dir"], train_pairs, protocol)
                loaded_models += 1
    views = candidate_views_from_runs(runs, protocol)
    architecture = evaluate_architecture_rule(runs, protocol)
    created = _now_iso()
    payload = {
        "stage": "freeze",
        "created_at": created,
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": context["manifest"]["protocol_sha256"],
        "protocol_frozen_sha256": context["manifest"]["protocol_frozen_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "prepare": {
            "final_access_id": final_access_identity(context),
            "manifest_sha256": sha256_file(context["prepare_dir"] / "manifest.json"),
        },
        "train_dev": {
            "manifest_sha256": td["manifest_sha256"],
            "runs_sha256": td["runs_sha256"],
            "updates": td["manifest"]["updates"],
            "seeds": td["manifest"]["budget"]["seeds"],
            "overrides": td["manifest"]["budget"]["overrides"],
            "formal_budget": td["manifest"]["budget"]["formal"],
        },
        "candidate_views": views,
        "architecture_rule": architecture,
        "runtime": runtime_info(),
        "boundary": [
            "freeze is computed from verified train-dev history and weights, not from a self-reported string",
            "tiny or overridden budgets are mechanism evidence only, not formal results",
            "the recommendation is a fixed-budget result, not a global optimum",
        ],
    }
    _create_out(out)
    freeze_sha = _write_json(out / "freeze.json", payload)
    duration = round(time.time() - started, 3)
    manifest = {
        "stage": "freeze",
        "created_at": created,
        "protocol_sha256": payload["protocol_sha256"],
        "source_sha256": payload["source_sha256"],
        "duration_seconds": duration,
        "runtime": payload["runtime"],
        "updates": payload["train_dev"]["updates"],
        "steps": None,
        "inference_calls": {"model_loads": loaded_models},
        "denominators": {"candidate_views": len(views), "paired_seeds": len(architecture.get("paired_seeds") or [])},
        "budget": {"updates": payload["train_dev"]["updates"], "formal": payload["train_dev"]["formal_budget"], "duration_seconds": duration},
        "artifacts": {"freeze.json": freeze_sha},
    }
    _write_json(out / "manifest.json", manifest)
    return {
        "stage": "freeze",
        "out": str(out),
        "freeze_sha256": freeze_sha,
        "recommended": architecture["recommended"],
        "candidate_views": len(views),
        "formal_budget": payload["train_dev"]["formal_budget"],
        "duration_seconds": duration,
    }
def _np_project(p5: np.ndarray, maps: dict) -> np.ndarray:
    return np.stack([p5[:, maps["groups"][parent]].sum(axis=1) for parent in maps["parents"]], axis=1)


def project_5_to_3(p5, maps: dict):
    if isinstance(p5, torch.Tensor):
        return torch.stack([p5[:, maps["groups"][parent]].sum(dim=-1) for parent in maps["parents"]], dim=-1)
    return _np_project(np.asarray(p5, dtype=np.float64), maps)


def _two_way_entries(pairs, logp3_forward: np.ndarray, logp3_reverse: np.ndarray, protocol: dict) -> list:
    parents = protocol["formal"]["parents"]
    c_index = parents.index("C")
    entries: list = []
    for index in range(len(pairs)):
        forward = logp3_forward[index]
        reverse = logp3_reverse[index]
        if not (bool(np.isfinite(forward).all()) and bool(np.isfinite(reverse).all())):
            entries.append(None)
            continue
        probs_forward = np.exp(forward)
        probs_reverse = np.exp(reverse)
        hard_forward = parents[int(np.argmax(probs_forward))]
        hard_reverse = parents[int(np.argmax(probs_reverse))]
        entries.append({
            "hard_a": hard_forward,
            "hard_b": hard_reverse,
            "combo": hard_forward + hard_reverse,
            "combo_legal": combination_leaf(hard_forward, hard_reverse, protocol) is not None,
            "joint_feasible": bool(joint_marginals_feasible(probs_forward.tolist(), probs_reverse.tolist(), protocol)),
            "c_edge_gap": float(abs(probs_forward[c_index] - probs_reverse[c_index])),
        })
    return entries


def _question_rows(view: dict, pairs, logp3_raw, logp5_raw, logp3_cal, logp5_cal, protocol: dict, maps: dict, two_way=None) -> list[dict]:
    parents = maps["parents"]
    leaves = maps["leaves"]
    parent_of_leaf = maps["parent_of_leaf"]
    rows: list[dict] = []
    for index, (p_mask, h_mask) in enumerate(pairs):
        gold_leaf_index = maps["leaf_index"][leaf_of(p_mask, h_mask)]
        gold_parent_index = parent_of_leaf[gold_leaf_index]
        forward_raw = logp3_raw[index]
        reverse_raw = None if logp5_raw is None else logp5_raw[index]
        forward_cal = None if logp3_cal is None else logp3_cal[index]
        reverse_cal = None if logp5_cal is None else logp5_cal[index]
        finite = bool(np.isfinite(forward_raw).all())
        if reverse_raw is not None:
            finite = finite and bool(np.isfinite(reverse_raw).all())
        if forward_cal is not None:
            finite = finite and bool(np.isfinite(forward_cal).all())
        if reverse_cal is not None:
            finite = finite and bool(np.isfinite(reverse_cal).all())
        row = {
            "candidate": view["candidate"],
            "arm": view["arm"],
            "seed": view["seed"],
            "view": view["view"],
            "step": view["step"],
            "model_sha256": view["model_sha256"],
            "index": index,
            "p": int(p_mask),
            "h": int(h_mask),
            "gold_leaf": leaves[gold_leaf_index],
            "gold_parent": parents[gold_parent_index],
        }
        if finite:
            p3_raw = np.exp(forward_raw)
            p5_raw = None if reverse_raw is None else np.exp(reverse_raw)
            p3_cal = None if forward_cal is None else np.exp(forward_cal)
            p5_cal = None if reverse_cal is None else np.exp(reverse_cal)
            parent_hit = int(np.argmax(p3_raw))
            fine_hit = None if p5_raw is None else int(np.argmax(p5_raw))
            parent_first_leaf = None
            if p5_raw is not None:
                members = maps["groups"][parents[parent_hit]]
                parent_first_leaf = leaves[int(max(members, key=lambda member: p5_raw[member]))]
            projection = None
            if p5_raw is not None:
                projected = _np_project(p5_raw[None, :], maps)[0]
                projection = float(np.max(np.abs(p3_raw - projected)))
            row.update({
                "p3": [float(value) for value in p3_raw],
                "p5": None if p5_raw is None else [float(value) for value in p5_raw],
                "coarse_logp_gold": float(forward_raw[gold_parent_index]),
                "fine_logp_gold": None if reverse_raw is None else float(reverse_raw[gold_leaf_index]),
                "coarse_nll": float(-forward_raw[gold_parent_index]),
                "fine_nll": None if reverse_raw is None else float(-reverse_raw[gold_leaf_index]),
                "coarse_nll_calibrated": None if forward_cal is None else float(-forward_cal[gold_parent_index]),
                "fine_nll_calibrated": None if reverse_cal is None else float(-reverse_cal[gold_leaf_index]),
                "parent_map": parents[parent_hit],
                "fine_map": None if fine_hit is None else leaves[fine_hit],
                "parent_first": parent_first_leaf,
                "map_disagreement": None if fine_hit is None else bool(parent_of_leaf[fine_hit] != parent_hit),
                "projection_residual": projection,
                "simplex_ok": bool(
                    (p3_raw >= 0).all()
                    and abs(float(p3_raw.sum()) - 1.0) < 1e-6
                    and (p5_raw is None or ((p5_raw >= 0).all() and abs(float(p5_raw.sum()) - 1.0) < 1e-6))
                ),
            })
        else:
            for key in (
                "p3", "p5", "coarse_logp_gold", "fine_logp_gold", "coarse_nll", "fine_nll",
                "coarse_nll_calibrated", "fine_nll_calibrated", "parent_map", "fine_map",
                "parent_first", "map_disagreement", "projection_residual",
            ):
                row[key] = None
            row["simplex_ok"] = False
        if two_way is not None:
            entry = two_way[index]
            row["two_way"] = entry
        else:
            row["two_way"] = None
        if not finite:
            failure = "non_finite"
        elif row["two_way"] is not None and not row["two_way"]["combo_legal"]:
            failure = "hard_combo_unmappable"
        else:
            failure = "ok"
        row["failure_reason"] = failure
        row["finite"] = finite
        rows.append(row)
    return rows


def _class_stats(rows: list[dict], gold_key: str, pred_key: str, names: Sequence[str]) -> dict:
    index = {name: i for i, name in enumerate(names)}
    gold = np.asarray([index[row[gold_key]] for row in rows], dtype=np.int64)
    pred = np.asarray([index[row[pred_key]] for row in rows], dtype=np.int64)
    macro, recalls = _macro_f1_and_recall(pred, gold, len(names))
    confusion = {f"{names[g]}->{names[p]}": 0 for g in range(len(names)) for p in range(len(names))}
    for gold_value, pred_value in zip(gold.tolist(), pred.tolist()):
        confusion[f"{names[gold_value]}->{names[pred_value]}"] += 1
    return {
        "macro_f1": None if math.isnan(macro) else float(macro),
        "per_class_recall": {names[i]: (None if math.isnan(recalls[i]) else float(recalls[i])) for i in range(len(names))},
        "confusion": confusion,
    }


def summarize_rows(rows: list[dict], protocol: dict, maps: dict) -> dict:
    parents = maps["parents"]
    leaves = maps["leaves"]
    n = len(rows)
    finite_rows = [row for row in rows if row["finite"]]
    fine_rows = [row for row in finite_rows if row.get("fine_map") is not None]

    def mean_of(key: str, subset: list[dict]):
        values = [row[key] for row in subset if row.get(key) is not None and math.isfinite(row[key])]
        return float(np.mean(values)) if values else None

    coarse_stats = _class_stats(finite_rows, "gold_parent", "parent_map", parents) if finite_rows else {"macro_f1": None, "per_class_recall": {}, "confusion": {}}
    fine_stats = _class_stats(fine_rows, "gold_leaf", "fine_map", leaves) if fine_rows else {"macro_f1": None, "per_class_recall": {}, "confusion": {}}
    projection_values = [row["projection_residual"] for row in rows if row.get("projection_residual") is not None]
    two_rows = [row for row in rows if row.get("two_way") is not None]
    two_way = None
    if two_rows:
        tolerance = float(protocol["evaluation"]["joint_marginals"]["c_edge_tolerance"])
        legal = sum(1 for row in two_rows if row["two_way"]["combo_legal"])
        gaps = [row["two_way"]["c_edge_gap"] for row in two_rows]
        two_way = {
            "n": len(two_rows),
            "coverage": legal / len(two_rows),
            "invalid": len(two_rows) - legal,
            "c_edge_mismatch": int(sum(1 for gap in gaps if gap > tolerance)),
            "c_edge_mean_gap": float(np.mean(gaps)),
            "prob_joint_incompatible": int(sum(1 for row in two_rows if not row["two_way"]["joint_feasible"])),
        }
    return {
        "n": n,
        "usable": len(finite_rows),
        "excluded_non_finite": n - len(finite_rows),
        "coarse_nll": mean_of("coarse_nll", finite_rows),
        "coarse_nll_calibrated": mean_of("coarse_nll_calibrated", finite_rows),
        "fine_nll": mean_of("fine_nll", fine_rows),
        "fine_nll_calibrated": mean_of("fine_nll_calibrated", fine_rows),
        "coarse_accuracy": (sum(1 for row in finite_rows if row["parent_map"] == row["gold_parent"]) / len(finite_rows)) if finite_rows else None,
        "fine_accuracy": (sum(1 for row in fine_rows if row["fine_map"] == row["gold_leaf"]) / len(fine_rows)) if fine_rows else None,
        "parent_first_accuracy": (sum(1 for row in fine_rows if row["parent_first"] == row["gold_leaf"]) / len(fine_rows)) if fine_rows else None,
        "coarse_macro_f1": coarse_stats["macro_f1"],
        "fine_macro_f1": fine_stats["macro_f1"],
        "coarse_per_class_recall": coarse_stats["per_class_recall"],
        "fine_per_class_recall": fine_stats["per_class_recall"],
        "coarse_confusion": coarse_stats["confusion"],
        "fine_confusion": fine_stats["confusion"],
        "map_disagreement": int(sum(1 for row in fine_rows if row["map_disagreement"])),
        "projection_residual_max": max(projection_values) if projection_values else None,
        "finite_violations": n - len(finite_rows),
        "simplex_violations": int(sum(1 for row in finite_rows if not row["simplex_ok"])),
        "hard_combo_unmappable": int(sum(1 for row in rows if row["failure_reason"] == "hard_combo_unmappable")),
        "two_way": two_way,
    }


def _rows_csv(rows: list[dict]) -> str:
    import csv
    import io

    columns = [
        "candidate", "arm", "seed", "view", "step", "model_sha256", "index", "p", "h",
        "gold_leaf", "gold_parent", "p3", "p5", "coarse_nll", "coarse_nll_calibrated",
        "fine_nll", "fine_nll_calibrated", "parent_map", "fine_map", "parent_first",
        "map_disagreement", "projection_residual", "finite", "simplex_ok", "failure_reason",
        "two_way_combo", "two_way_joint_feasible", "two_way_c_edge_gap",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        two_way = row.get("two_way")
        flat = {
            "candidate": row["candidate"],
            "arm": row["arm"],
            "seed": row["seed"],
            "view": row["view"],
            "step": row["step"],
            "model_sha256": row["model_sha256"],
            "index": row["index"],
            "p": row["p"],
            "h": row["h"],
            "gold_leaf": row["gold_leaf"],
            "gold_parent": row["gold_parent"],
            "p3": "" if row.get("p3") is None else ";".join(f"{value:.9g}" for value in row["p3"]),
            "p5": "" if row.get("p5") is None else ";".join(f"{value:.9g}" for value in row["p5"]),
            "coarse_nll": row.get("coarse_nll"),
            "coarse_nll_calibrated": row.get("coarse_nll_calibrated"),
            "fine_nll": row.get("fine_nll"),
            "fine_nll_calibrated": row.get("fine_nll_calibrated"),
            "parent_map": row.get("parent_map"),
            "fine_map": row.get("fine_map"),
            "parent_first": row.get("parent_first"),
            "map_disagreement": row.get("map_disagreement"),
            "projection_residual": row.get("projection_residual"),
            "finite": row["finite"],
            "simplex_ok": row["simplex_ok"],
            "failure_reason": row["failure_reason"],
            "two_way_combo": "" if two_way is None else two_way["combo"],
            "two_way_joint_feasible": "" if two_way is None else two_way["joint_feasible"],
            "two_way_c_edge_gap": "" if two_way is None else f"{two_way['c_edge_gap']:.9g}",
        }
        writer.writerow(flat)
    return buffer.getvalue()


def _read_attempts(prepare_dir: Path) -> list[dict]:
    path = prepare_dir / ATTEMPTS_FILENAME
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_attempt(prepare_dir: Path, record: dict) -> None:
    path = prepare_dir / ATTEMPTS_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def stage_finalize(
    out_path,
    prepare_dir,
    train_dev_dir,
    freeze_path,
    *,
    verify_only: bool = False,
    reproduce: bool = False,
    protocol_path=None,
    allow_external: bool = False,
) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    context = load_prepare_context(prepare_dir, protocol_path)
    protocol = context["protocol"]
    manifest = context["manifest"]
    td = load_train_dev_context(context, train_dev_dir)
    freeze_file = Path(freeze_path)
    if not freeze_file.exists():
        raise HarnessError("refusing finalize: freeze file missing", 3)
    freeze = _read_json(freeze_file)
    if freeze.get("stage") != "freeze":
        raise HarnessError("refusing finalize: not a freeze file", 3)
    freeze_sha = sha256_file(freeze_file)
    if freeze.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise HarnessError("refusing finalize: freeze does not bind this prepare protocol", 4)
    if freeze.get("protocol_frozen_sha256") != manifest["protocol_frozen_sha256"]:
        raise HarnessError("refusing finalize: freeze does not bind this frozen protocol file", 4)
    if freeze.get("source_sha256") != sha256_file(Path(__file__)):
        raise HarnessError("harness source changed since freeze; re-freeze required", 4)
    if freeze.get("train_dev", {}).get("manifest_sha256") != td["manifest_sha256"]:
        raise HarnessError("refusing finalize: freeze does not bind this train-dev manifest", 4)
    if freeze.get("train_dev", {}).get("runs_sha256") != td["runs_sha256"]:
        raise HarnessError("refusing finalize: freeze does not bind this runs.json", 4)
    access_id = final_access_identity(context)
    if freeze.get("prepare", {}).get("final_access_id") != access_id:
        raise HarnessError("refusing finalize: freeze does not bind this final-access identity", 4)
    runs = td["runs"]["runs"]
    recomputed_views = candidate_views_from_runs(runs, protocol)
    if _canonical(recomputed_views) != _canonical(freeze.get("candidate_views")):
        raise HarnessError("frozen candidate set does not match the train-dev evidence (selection or weights changed)", 4)
    train_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/train.jsonl", expected_exposure(protocol, "train"))
    models_by_view: dict[str, TinyHeadModel] = {}
    for view in freeze["candidate_views"]:
        models_by_view[view["candidate"]] = verify_model_file(view["arm"], int(view["seed"]), view["model_path"], view["model_sha256"], td["dir"], train_pairs, protocol)
    checks = {
        "freeze_binding": True,
        "train_dev_chain": True,
        "candidate_set_matches": True,
        "model_weights_loadable_finite_shape_checked": True,
    }
    if verify_only:
        _create_out(out)
        payload = {
            "stage": "finalize-precheck",
            "mode": "preflight",
            "created_at": _now_iso(),
            "checks": checks,
            "freeze_sha256": freeze_sha,
            "final_access_id": access_id,
            "candidate_views": [view["candidate"] for view in freeze["candidate_views"]],
            "recommended": freeze["architecture_rule"]["recommended"],
            "runtime": runtime_info(),
            "note": "full chain verified including real weight loading; scoring not run; no attempt registered",
        }
        precheck_sha = _write_json(out / "finalize-precheck.json", payload)
        duration = round(time.time() - started, 3)
        manifest_out = {
            "stage": "finalize",
            "mode": "preflight",
            "protocol_sha256": manifest["protocol_sha256"],
            "source_sha256": sha256_file(Path(__file__)),
            "created_at": payload["created_at"],
            "duration_seconds": duration,
            "runtime": payload["runtime"],
            "updates": td["manifest"]["budget"]["updates"],
            "steps": None,
            "inference_calls": {"forward_batches": 0, "forward_rows": 0},
            "denominators": {"candidate_views": len(freeze["candidate_views"])},
            "budget": {"updates": None, "steps": None, "formal": td["manifest"]["budget"]["formal"], "duration_seconds": duration},
            "artifacts": {"finalize-precheck.json": precheck_sha},
        }
        _write_json(out / "manifest.json", manifest_out)
        return {"stage": "finalize-precheck", "out": str(out), "checks": checks, "precheck_sha256": precheck_sha, "duration_seconds": duration}
    attempts = _read_attempts(context["prepare_dir"])
    same_access = [row for row in attempts if row.get("final_access_id") == access_id]
    if same_access:
        if any(row.get("freeze_sha256") != freeze_sha for row in same_access):
            raise HarnessError("final-access decision changed after scoring started; refused (start a new prepare/amendment route instead)", 5)
        if not reproduce:
            raise HarnessError("final-access already attempted for this binding; pass --reproduce for an explicit reproduction", 5)
        mode = "reproduce"
    else:
        mode = "score"
    _append_attempt(context["prepare_dir"], {
        "record": "attempt",
        "final_access_id": access_id,
        "freeze_sha256": freeze_sha,
        "mode": mode,
        "registered_at": _now_iso(),
        "out": str(out),
    })
    try:
        result = _score_frozen_set(out, context, td, freeze, models_by_view, mode, freeze_sha, access_id, started)
    except Exception as exc:
        _append_attempt(context["prepare_dir"], {
            "record": "outcome",
            "final_access_id": access_id,
            "freeze_sha256": freeze_sha,
            "status": "failed",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "recorded_at": _now_iso(),
        })
        raise
    _append_attempt(context["prepare_dir"], {
        "record": "outcome",
        "final_access_id": access_id,
        "freeze_sha256": freeze_sha,
        "status": "success",
        "summary_sha256": result["summary_sha256"],
        "recorded_at": _now_iso(),
    })
    return result


def _score_frozen_set(out: Path, context: dict, td: dict, freeze: dict, models_by_view: dict, mode: str, freeze_sha: str, access_id: str, started: float) -> dict:
    protocol = context["protocol"]
    manifest = context["manifest"]
    prepare_dir = context["prepare_dir"]
    maps = index_maps(protocol)
    prob_pairs = _read_verified_exposure(prepare_dir, manifest, "exposures/calibration_prob.jsonl", expected_exposure(protocol, "calibration_prob"))
    policy_pairs = _read_verified_exposure(prepare_dir, manifest, "exposures/calibration_policy.jsonl", expected_exposure(protocol, "calibration_policy"))
    final_pairs = expected_exposure(protocol, "finaltoy")
    if sha256_bytes(exposure_jsonl(final_pairs)) != manifest["pools"]["finaltoy"]["exposure_sha256"]:
        raise HarnessError("regenerated finaltoy fingerprint does not match the frozen prepare manifest", 4)
    _, prob_parent = label_indices(prob_pairs, protocol, maps)
    _, policy_parent = label_indices(policy_pairs, protocol, maps)
    inference_calls = 0
    inference_rows = 0

    def log_probs(model: TinyHeadModel, pairs):
        nonlocal inference_calls, inference_rows
        inference_calls += 1
        inference_rows += len(pairs)
        return _log_probs(model, pairs, protocol)

    cache: dict = {}
    for view in freeze["candidate_views"]:
        key = (view["arm"], int(view["seed"]), view["step"], view["model_sha256"])
        if key in cache:
            continue
        model = models_by_view[view["candidate"]]
        prob_logp3, _ = log_probs(model, prob_pairs)
        policy_logp3, _ = log_probs(model, policy_pairs)
        final_logp3, final_logp5 = log_probs(model, final_pairs)
        temperature = fit_temperature(prob_logp3, prob_parent, protocol)
        policy_calibrated = apply_temperature_to_logp3(policy_logp3, temperature["temperature"])
        policy_description = policy_coverage_description(policy_calibrated, policy_parent, protocol)
        if final_logp5 is None:
            cal_logp3 = apply_temperature_to_logp3(final_logp3, temperature["temperature"])
            cal_logp5 = None
        else:
            cal_logp5 = apply_temperature_to_logp5(final_logp5, temperature["temperature"], maps)
            cal_logp3 = _np_log_group_sums(cal_logp5, maps)
        two_way = None
        if view["arm"] == "A":
            swapped = [(h_mask, p_mask) for p_mask, h_mask in final_pairs]
            swap_logp3, _ = log_probs(model, swapped)
            two_way = _two_way_entries(final_pairs, final_logp3, swap_logp3, protocol)
        cache[key] = {
            "final_logp3": final_logp3,
            "final_logp5": final_logp5,
            "cal_logp3": cal_logp3,
            "cal_logp5": cal_logp5,
            "temperature": temperature,
            "policy": policy_description,
            "two_way": two_way,
        }
    rows: list[dict] = []
    view_summaries: dict = {}
    for view in freeze["candidate_views"]:
        key = (view["arm"], int(view["seed"]), view["step"], view["model_sha256"])
        entry = cache[key]
        rows_view = _question_rows(
            view,
            final_pairs,
            entry["final_logp3"],
            entry["final_logp5"],
            entry["cal_logp3"],
            entry["cal_logp5"],
            protocol,
            maps,
            two_way=entry["two_way"],
        )
        rows.extend(rows_view)
        view_summaries[view["candidate"]] = {
            **summarize_rows(rows_view, protocol, maps),
            "temperature": entry["temperature"],
            "policy": entry["policy"],
        }
    _create_out(out)
    rows_sha = _write_bytes(out / "questions.jsonl", "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode("utf-8"))
    csv_sha = _write_bytes(out / "questions.csv", _rows_csv(rows).encode("utf-8"))
    read_back = [json.loads(line) for line in (out / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(read_back) != len(rows):
        raise HarnessError("questions file round-trip lost rows", 1)
    recomputed = {}
    for view in freeze["candidate_views"]:
        subset = [row for row in read_back if row["candidate"] == view["candidate"]]
        recomputed[view["candidate"]] = {
            **summarize_rows(subset, protocol, maps),
            "temperature": view_summaries[view["candidate"]]["temperature"],
            "policy": view_summaries[view["candidate"]]["policy"],
        }
    if _canonical(recomputed) != _canonical(view_summaries):
        raise HarnessError("summary recomputation from per-question files drifted from the in-memory computation", 1)
    duration = round(time.time() - started, 3)
    payload = {
        "stage": "finalize",
        "mode": mode,
        "created_at": _now_iso(),
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_frozen_sha256": manifest["protocol_frozen_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "freeze_sha256": freeze_sha,
        "final_access_id": access_id,
        "recommended": freeze["architecture_rule"]["recommended"],
        "formal_budget": td["manifest"]["budget"]["formal"],
        "budget_overrides": td["manifest"]["budget"]["overrides"],
        "views": view_summaries,
        "rows": len(rows),
        "questions_file": "questions.jsonl",
        "questions_sha256": rows_sha,
        "csv_sha256": csv_sha,
        "inference_calls": {"forward_batches": inference_calls, "forward_rows": inference_rows},
        "runtime": runtime_info(),
        "boundary": [
            "toy finaltoy metrics are research-only simulation numbers, not natural-language results and not production gains",
            "the reported candidate set is the complete frozen set; C/H views sharing one artifact are deduplicated for compute but reported separately",
            "policy coverage is a descriptive 90% empirical rule, not a risk certificate",
            "tiny or overridden budgets are mechanism evidence only",
        ],
    }
    summary_sha = _write_json(out / "summary.json", payload)
    manifest_out = {
        "stage": "finalize",
        "mode": mode,
        "protocol_sha256": manifest["protocol_sha256"],
        "source_sha256": payload["source_sha256"],
        "created_at": payload["created_at"],
        "duration_seconds": duration,
        "runtime": payload["runtime"],
        "updates": td["manifest"]["budget"]["updates"],
        "steps": None,
        "inference_calls": payload["inference_calls"],
        "denominators": {"rows": len(rows), "candidate_views": len(freeze["candidate_views"])},
        "budget": {"updates": td["manifest"]["budget"]["updates"], "steps": None, "formal": td["manifest"]["budget"]["formal"], "duration_seconds": duration},
        "artifacts": {"questions.jsonl": rows_sha, "questions.csv": csv_sha, "summary.json": summary_sha},
    }
    _write_json(out / "manifest.json", manifest_out)
    return {
        "stage": "finalize",
        "mode": mode,
        "out": str(out),
        "summary_sha256": summary_sha,
        "recommended": payload["recommended"],
        "rows": len(rows),
        "views": sorted(view_summaries),
        "duration_seconds": duration,
    }


def stage_status(run_dir) -> dict:
    root = Path(run_dir)
    stages: dict = {}
    failures: list[str] = []
    for stage in ("formal", "prepare", "train-dev", "freeze", "finalize"):
        stage_dir = root / stage
        manifest_path = stage_dir / "manifest.json"
        if not manifest_path.exists():
            if stage_dir.exists():
                stages[stage] = {"stage": stage, "verified": False, "reason": "manifest missing"}
                failures.append(stage)
            continue
        manifest = _read_json(manifest_path)
        missing_keys = [key for key in MANDATORY_MANIFEST_KEYS if key not in manifest]
        artifacts = manifest.get("artifacts") or {}
        checks = {}
        problems: list[str] = []
        if missing_keys:
            problems.append("schema")
        if not artifacts:
            problems.append("artifacts-empty")
        for rel, expected in artifacts.items():
            path = stage_dir / rel
            actual = sha256_file(path) if path.exists() else None
            ok = actual == expected
            checks[rel] = ok
        if checks and not all(checks.values()):
            problems.append("hash")
        verified = not problems
        if problems:
            failures.append(stage)
        stages[stage] = {
            "stage": manifest.get("stage"),
            "mode": manifest.get("mode"),
            "problems": problems,
            "schema_ok": not missing_keys,
            "hashes_ok": bool(checks) and all(checks.values()),
            "created_at": manifest.get("created_at"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "source_sha256": manifest.get("source_sha256"),
            "manifest_keys_missing": missing_keys,
            "artifacts_verified": verified,
            "artifact_checks": checks,
            "denominators": manifest.get("denominators"),
            "updates": manifest.get("updates"),
            "steps": manifest.get("steps"),
            "runtime": manifest.get("runtime"),
        }
    ok = bool(stages) and all(row.get("artifacts_verified") for row in stages.values())
    return {"run_dir": str(root), "read_only": True, "ok": ok, "failures": failures, "stages": stages}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.memory_relations.head_design")
    sub = parser.add_subparsers(dest="stage", required=True)
    formal = sub.add_parser("formal", help="exhaustive census + edit mutation audit")
    formal.add_argument("--out", required=True)
    formal.add_argument("--protocol")
    formal.add_argument("--allow-external-out", action="store_true")
    prepare = sub.add_parser("prepare", help="freeze protocol, entity partition and ordered exposure fingerprints")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--protocol")
    prepare.add_argument("--allow-external-out", action="store_true")
    train_dev = sub.add_parser("train-dev", help="train candidates on train/dev only (no calibration/finaltoy parsing)")
    train_dev.add_argument("--out", required=True)
    train_dev.add_argument("--prepare", required=True)
    train_dev.add_argument("--protocol")
    train_dev.add_argument("--updates", type=int, default=None, help="tiny-budget override for closed-loop mechanism runs; recorded as an override")
    train_dev.add_argument("--seeds", default=None, help="comma-separated seed override; recorded as an override")
    train_dev.add_argument("--allow-external-out", action="store_true")
    freeze = sub.add_parser("freeze", help="compute the frozen decision from verified train-dev history and weights")
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--prepare", required=True)
    freeze.add_argument("--train-dev", required=True)
    freeze.add_argument("--protocol")
    freeze.add_argument("--allow-external-out", action="store_true")
    finalize = sub.add_parser("finalize", help="score the frozen candidate set once (coarse + fine views, A two-way diagnostic)")
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--prepare", required=True)
    finalize.add_argument("--train-dev", required=True)
    finalize.add_argument("--freeze", required=True)
    finalize.add_argument("--verify-only", action="store_true")
    finalize.add_argument("--reproduce", action="store_true")
    finalize.add_argument("--protocol")
    finalize.add_argument("--allow-external-out", action="store_true")
    status = sub.add_parser("status", help="read-only inspection of an existing run (non-zero exit on verification failure)")
    status.add_argument("--run", required=True)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.stage == "formal":
            result = stage_formal(args.out, args.protocol, allow_external=args.allow_external_out)
        elif args.stage == "prepare":
            result = stage_prepare(args.out, args.protocol, allow_external=args.allow_external_out)
        elif args.stage == "train-dev":
            seeds = [int(part) for part in args.seeds.split(",")] if args.seeds else None
            result = stage_train_dev(args.out, args.prepare, args.protocol, allow_external=args.allow_external_out, updates=args.updates, seeds=seeds)
        elif args.stage == "freeze":
            result = stage_freeze(args.out, args.prepare, getattr(args, "train_dev"), protocol_path=args.protocol, allow_external=args.allow_external_out)
        elif args.stage == "finalize":
            result = stage_finalize(
                args.out,
                args.prepare,
                getattr(args, "train_dev"),
                args.freeze,
                verify_only=args.verify_only,
                reproduce=args.reproduce,
                protocol_path=args.protocol,
                allow_external=args.allow_external_out,
            )
        else:
            result = stage_status(args.run)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0 if result["ok"] else 4
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
