"""研究-only 分区与模型数值：固定注册类指标、log 空间损失、真实增量续训与逐题诊断。

输入仅原始 bitmap；目标为隔离池的 conditional partition distribution，不声称 NLP 泛化。
非有限训练/校准拒绝；诊断保留全尝试分母；温度只作用父边缘，不 epsilon 造概率。
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from research.memory_relations.head_design_semantics import (
    Geometry,
    HarnessError,
    _array_digest,
    _now_iso,
    combination_leaf,
    joint_marginals_feasible,
    leaf_of,
    parent_of_masks,
    probability_simplex,
    sha256_bytes,
    strata_families,
    study_geometry,
)

TRAINABLE_ARMS = ("A", "B", "C", "H")


def _sealed_pool_name(protocol: dict, study: str) -> str:
    return protocol["semantics"]["studies"][study]["sealed_pool"]


def build_partition(protocol: dict, study: str = "formal") -> dict:
    geometry = study_geometry(protocol, study)
    config = protocol["partition"]["studies"][study]
    rng = random.Random(protocol["partition"]["seed"])
    strata = strata_families(geometry)
    roles: dict[str, list[tuple[int, int]]] = {"train": [], "dev": [], "calibration": [], "final": []}
    if config.get("final_first"):
        for key in sorted(strata, key=lambda value: int(value)):
            counts = config["strata_min_side"][str(key)]
            bucket = sorted(strata[key])
            candidates = [family for family in bucket if family[0] >= 256 and family[1] >= 256]
            others = [family for family in bucket if not (family[0] >= 256 and family[1] >= 256)]
            rng.shuffle(candidates)
            final_take = candidates[: counts["final"]]
            pool = others + candidates[counts["final"]:]
            rng.shuffle(pool)
            roles["final"].extend(final_take)
            cal_take = pool[: counts["calibration"]]
            dev_take = pool[len(cal_take):len(cal_take) + counts["dev"]]
            train_take = pool[len(cal_take) + len(dev_take):]
            if len(cal_take) != counts["calibration"] or len(dev_take) != counts["dev"] or len(train_take) != counts["train"]:
                raise HarnessError(f"partition stratum {key}: role counts do not fit the family pool", 2)
            roles["calibration"].extend(cal_take)
            roles["dev"].extend(dev_take)
            roles["train"].extend(train_take)
    else:
        for key in sorted(strata, key=lambda value: int(value)):
            counts = config["strata_min_side"][str(key)]
            bucket = list(strata[key])
            rng.shuffle(bucket)
            offset = 0
            for role in ("train", "dev", "calibration", "final"):
                take = counts[role]
                roles[role].extend(bucket[offset:offset + take])
                offset += take
            if offset != len(bucket):
                raise HarnessError(f"partition stratum {key}: role counts do not fill the family bucket", 2)
    calibration = list(roles["calibration"])
    rng.shuffle(calibration)
    prob_families = config["calibration_pools"]["prob_families"]
    if prob_families + config["calibration_pools"]["policy_families"] != len(calibration):
        raise HarnessError("calibration pools must split the calibration families", 2)
    sealed = _sealed_pool_name(protocol, study)
    pair_pools = {
        "train": roles["train"],
        "dev": roles["dev"],
        "calibration_prob": calibration[:prob_families],
        "calibration_policy": calibration[prob_families:],
        sealed: roles["final"],
    }
    entity_pools: dict[str, list[int]] = {}
    for pool, pool_families in pair_pools.items():
        entities = [mask for family in pool_families for mask in family]
        if pool == "train":
            entities.append(geometry.full_mask)
        entity_pools[pool] = sorted(entities)
    novelty = None
    if config.get("final_first"):
        members = [mask for family in pair_pools[sealed] for mask in family]
        minimum = min(members) if members else None
        required = geometry.final_novelty_min_mask
        novelty = {
            "min_member_value": minimum,
            "required_at_least": required,
            "passed": bool(minimum is not None and required is not None and minimum >= required),
            "rule": config["final_constraint"],
        }
        if not novelty["passed"]:
            raise HarnessError("formal final families violate the >255 novelty constraint", 2)
    return {
        "study": study,
        "geometry": geometry,
        "stratum_families": {str(key): len(strata[key]) for key in sorted(strata)},
        "pair_pools": pair_pools,
        "entity_pools": entity_pools,
        "sealed_pool": sealed,
        "novelty": novelty,
    }


def ordered_exposures(entities: Sequence[int]) -> list[tuple[int, int]]:
    return [(p_mask, h_mask) for p_mask in entities for h_mask in entities]


def exposure_jsonl(pairs) -> bytes:
    return "".join(json.dumps({"p": p, "h": h}, separators=(",", ":")) + "\n" for p, h in pairs).encode("utf-8")


def entity_digest(entities: Sequence[int]) -> str:
    return sha256_bytes(",".join(str(mask) for mask in sorted(entities)).encode("utf-8"))


def pool_statistics(entities: Sequence[int], pairs: Sequence[tuple[int, int]], protocol: dict, geometry: Geometry) -> dict:
    leaf_counts = {leaf: 0 for leaf in protocol["semantics"]["leaves"]}
    parent_counts = {parent: 0 for parent in protocol["semantics"]["parents"]}
    diagonal = 0
    complement = 0
    for p_mask, h_mask in pairs:
        leaf_counts[leaf_of(p_mask, h_mask)] += 1
        parent_counts[parent_of_masks(p_mask, h_mask, full_mask=geometry.full_mask)] += 1
        if p_mask == h_mask:
            diagonal += 1
        if p_mask ^ h_mask == geometry.full_mask:
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


def partition_report(protocol: dict, study: str = "formal") -> dict:
    config = protocol["partition"]["studies"][study]
    built = build_partition(protocol, study)
    geometry = built["geometry"]
    expected_families = config["expected_families"]
    actual_by_role = {
        "train": len(built["pair_pools"]["train"]),
        "dev": len(built["pair_pools"]["dev"]),
        "calibration": len(built["pair_pools"]["calibration_prob"]) + len(built["pair_pools"]["calibration_policy"]),
        "final": len(built["pair_pools"][built["sealed_pool"]]),
    }
    for role, count in actual_by_role.items():
        if count != expected_families[role]:
            raise HarnessError(f"partition family count mismatch for {role}", 2)
    if sum(actual_by_role.values()) != expected_families["total"]:
        raise HarnessError("partition family totals mismatch", 2)
    pools: dict[str, dict] = {}
    expected_entities = config["expected_entities"]
    expected_pool_pairs = config["expected_pool_pairs"]
    calibration_pools = config["calibration_pools"]
    for pool, entities in built["entity_pools"].items():
        sealed = pool == built["sealed_pool"]
        if pool == "calibration_prob":
            expected_count = calibration_pools["prob_families"] * 2
        elif pool == "calibration_policy":
            expected_count = calibration_pools["policy_families"] * 2
        else:
            expected_count = expected_entities[_role_of_pool(pool, built["sealed_pool"])]
        if len(entities) != expected_count:
            raise HarnessError(f"partition entity count mismatch for {pool}", 2)
        exposures = ordered_exposures(entities)
        expected_pairs_key = "final" if sealed else pool
        if len(exposures) != expected_pool_pairs[expected_pairs_key]:
            raise HarnessError(f"partition pool pair count mismatch for {pool}", 2)
        pools[pool] = {
            "entities": None if sealed else entities,
            "entity_count": len(entities),
            "entity_sha256": entity_digest(entities),
            "pairs": len(exposures),
            "exposure_sha256": sha256_bytes(exposure_jsonl(exposures)),
            "sealed": sealed,
            "exported": not sealed,
            "pilot": bool(protocol["semantics"]["studies"][study]["pilot"]),
            "stats": pool_statistics(entities, exposures, protocol, geometry),
        }
    train_entities = pools["train"]["entity_count"]
    dev_entities = pools["dev"]["entity_count"]
    cal_entities = pools["calibration_prob"]["entity_count"] + pools["calibration_policy"]["entity_count"]
    final_entities = pools[built["sealed_pool"]]["entity_count"]
    universe = geometry.full_mask * geometry.full_mask
    included = sum(row["pairs"] for row in pools.values())
    cross_partition = universe - (train_entities ** 2 + dev_entities ** 2 + cal_entities ** 2 + final_entities ** 2)
    calibration_cross_pool = cal_entities ** 2 - (pools["calibration_prob"]["entity_count"] ** 2 + pools["calibration_policy"]["entity_count"] ** 2)
    excluded = {"cross_partition": cross_partition, "calibration_cross_pool": calibration_cross_pool, "total": universe - included}
    if excluded != config["excluded_pairs"]:
        raise HarnessError("partition exclusions do not match protocol (real denominators changed)", 2)
    if config.get("final_first") and not pools[built["sealed_pool"]]["stats"]["cardinality_support"]:
        raise HarnessError("sealed pool statistics missing", 2)
    return {
        "study": study,
        "geometry": {
            "world_bits": geometry.world_bits,
            "world_count": geometry.world_count,
            "full_mask": geometry.full_mask,
            "input_bits_per_side": geometry.input_bits_per_side,
        },
        "seed": protocol["partition"]["seed"],
        "stratum_families": built["stratum_families"],
        "pools": pools,
        "sealed_pool": built["sealed_pool"],
        "novelty": built["novelty"],
        "denominators": {
            "universe_pairs": universe,
            "included_pairs": {**{pool: row["pairs"] for pool, row in pools.items()}, "total": included},
            "excluded_pairs": excluded,
        },
        "distribution_target": protocol["partition"]["distribution_target"],
    }


def _role_of_pool(pool: str, sealed_pool: str) -> str:
    if pool == sealed_pool:
        return "final"
    if pool in ("calibration_prob", "calibration_policy"):
        return "calibration"
    return pool


def index_maps(protocol: dict) -> dict:
    semantics = protocol["semantics"]
    leaves = semantics["leaves"]
    parents = semantics["parents"]
    groups = {parent: [i for i, leaf in enumerate(leaves) if semantics["leaf_parent"][leaf] == parent] for parent in parents}
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
        "parent_of_leaf": [parents.index(semantics["leaf_parent"][leaf]) for leaf in leaves],
        "leaf_index": {leaf: i for i, leaf in enumerate(leaves)},
    }


def encode_pairs(pairs: Sequence[tuple[int, int]], geometry: Geometry) -> np.ndarray:
    bits = geometry.input_bits_per_side
    for p_mask, h_mask in pairs:
        if isinstance(p_mask, bool) or not isinstance(p_mask, int) or not 0 <= p_mask <= geometry.full_mask:
            raise HarnessError(f"p_mask out of range [0, {geometry.full_mask}]", 2)
        if isinstance(h_mask, bool) or not isinstance(h_mask, int) or not 0 <= h_mask <= geometry.full_mask:
            raise HarnessError(f"h_mask out of range [0, {geometry.full_mask}]", 2)
    p_masks = np.asarray([p for p, _ in pairs], dtype=np.int64)
    h_masks = np.asarray([h for _, h in pairs], dtype=np.int64)
    p_low = ((p_masks[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.float32)
    h_low = ((h_masks[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.float32)
    return np.concatenate([p_low, h_low], axis=1)


def label_indices(pairs: Sequence[tuple[int, int]], protocol: dict, maps: dict | None = None, geometry: Geometry | None = None) -> tuple[np.ndarray, np.ndarray]:
    maps = maps or index_maps(protocol)
    full_mask = geometry.full_mask if geometry is not None else None
    leaf_idx = np.asarray([maps["leaf_index"][leaf_of(p, h, full_mask=full_mask)] for p, h in pairs], dtype=np.int64)
    parent_idx = np.asarray([maps["parent_of_leaf"][i] for i in leaf_idx], dtype=np.int64)
    return leaf_idx, parent_idx


def leaf_counts_from_pairs(pairs: Sequence[tuple[int, int]], protocol: dict) -> dict:
    counts = {leaf: 0 for leaf in protocol["semantics"]["leaves"]}
    for p, h in pairs:
        counts[leaf_of(p, h)] += 1
    return counts


def parent_counts_from_pairs(pairs: Sequence[tuple[int, int]], protocol: dict, geometry: Geometry) -> dict:
    counts = {parent: 0 for parent in protocol["semantics"]["parents"]}
    for p, h in pairs:
        counts[parent_of_masks(p, h, full_mask=geometry.full_mask)] += 1
    return counts


def pregenerate_common(seed: int, n_train: int, protocol: dict, geometry: Geometry) -> dict:
    backbone_rng = np.random.default_rng(np.random.PCG64([seed, 0xB0]))
    batches_rng = np.random.default_rng(np.random.PCG64([seed, 0xBA]))
    spec = protocol["training"]
    hidden = spec["backbone"]["hidden"]
    upstream = 2 * geometry.input_bits_per_side
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
    parents = protocol["semantics"]["parents"]
    raw = np.asarray([max(float(parent_counts.get(parent, 0)), 1e-9) for parent in parents], dtype=np.float64)
    pi_parent = raw / raw.sum()
    pi_leaf = np.zeros(len(protocol["semantics"]["leaves"]), dtype=np.float64)
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
    tags = protocol["training"]["init"]["head_rng_tags"]
    rng = np.random.default_rng(np.random.PCG64([seed, tags["leaf"] if arm in ("B", "C") else tags["parent"]]))
    scale = float(protocol["training"]["init"]["tiny_random_scale"])
    hidden = int(protocol["training"]["backbone"]["hidden"][1])

    def tiny_weights(out_dim: int) -> np.ndarray:
        return rng.normal(0.0, scale, (out_dim, hidden)).astype(np.float32)

    state: dict[str, np.ndarray] = {}
    if arm == "A":
        state["parent_w"] = tiny_weights(3)
        state["parent_b"] = np.log(priors["pi_parent"]).astype(np.float32)
    elif arm in ("B", "C"):
        state["leaf_w"] = tiny_weights(len(priors["pi_leaf"]))
        state["leaf_b"] = np.log(priors["pi_leaf"]).astype(np.float32)
    else:
        state["parent_w"] = tiny_weights(3)
        state["parent_b"] = np.log(priors["pi_parent"]).astype(np.float32)
        for parent in protocol["training"]["candidates"]["H"]["branches"]:
            key = parent.lower()
            branch_rng = np.random.default_rng(np.random.PCG64([seed, tags[f"branch_{parent}"]]))
            state[f"branch_{key}_w"] = branch_rng.normal(0.0, scale, (1, hidden)).astype(np.float32)
            state[f"branch_{key}_b"] = np.zeros(1, dtype=np.float32)
    return state
class TinyHeadModel(nn.Module):
    def __init__(self, arm: str, protocol: dict, geometry: Geometry, backbone_state: dict, head_state: dict):
        super().__init__()
        self.arm = arm
        self.protocol = protocol
        self.maps = index_maps(protocol)
        spec = protocol["training"]
        upstream = 2 * geometry.input_bits_per_side
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


def confusion_metrics(confusion: np.ndarray) -> dict:
    matrix = np.asarray(confusion)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise HarnessError("invalid confusion matrix", 2)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    return {"macro_f1": float(f1.mean()), "recall": recall.tolist(), "f1": f1.tolist(), "support": support.astype(int).tolist(), "zero_division": 0}


def classification_metrics(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> dict:
    pred, gold = np.asarray(pred), np.asarray(gold)
    if type(n_classes) is not int or n_classes < 1 or pred.ndim != 1 or gold.shape != pred.shape or not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(gold.dtype, np.integer):
        raise HarnessError("invalid registered-class prediction vectors", 2)
    if ((pred < 0) | (pred >= n_classes) | (gold < 0) | (gold >= n_classes)).any():
        raise HarnessError("prediction outside registered classes", 2)
    matrix = np.bincount(gold * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return {**confusion_metrics(matrix), "confusion": matrix.tolist()}


def _macro_f1_and_recall(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> tuple[float, list[float]]:
    stats = classification_metrics(pred, gold, n_classes)
    return stats["macro_f1"], stats["recall"]


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
        "coarse_support": np.bincount(gold_parent, minlength=len(model.maps["parents"])).tolist(),
        "fine_support": np.bincount(leaf_idx.numpy(), minlength=len(model.maps["leaves"])).tolist(),
        "registered_classes": {"coarse": model.maps["parents"], "fine": model.maps["leaves"]},
        "zero_division": 0,
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


def build_model_skeleton(arm: str, seed: int, train_pairs: Sequence[tuple[int, int]], protocol: dict, geometry: Geometry) -> TinyHeadModel:
    parent_counts = parent_counts_from_pairs(train_pairs, protocol, geometry)
    common = pregenerate_common(seed, len(train_pairs), protocol, geometry)
    return TinyHeadModel(arm, protocol, geometry, common["backbone"], head_init(arm, seed, protocol, parent_counts))


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


def state_digest(state: dict) -> str:
    return _array_digest({key: value.detach().cpu().numpy() for key, value in state.items()})


def train_arm(
    arm: str,
    seed: int,
    train_pairs: Sequence[tuple[int, int]],
    dev_pairs: Sequence[tuple[int, int]],
    protocol: dict,
    geometry: Geometry,
    *,
    updates: int | None = None,
    resume: dict | None = None,
) -> dict:
    if arm not in TRAINABLE_ARMS:
        raise HarnessError(f"unknown arm {arm}", 2)
    torch.set_num_threads(protocol["training"]["cpu_threads"])
    maps, spec = index_maps(protocol), protocol["training"]
    eval_every = spec["eval_every"]
    target_updates = int(updates if updates is not None else spec["updates"])
    if target_updates < 1 or target_updates % eval_every or not train_pairs or not dev_pairs:
        raise HarnessError("training requires non-empty exposures and a positive grid endpoint", 2)
    train_x = torch.as_tensor(encode_pairs(train_pairs, geometry))
    dev_x = torch.as_tensor(encode_pairs(dev_pairs, geometry))
    train_leaf, train_parent = (torch.as_tensor(arr) for arr in label_indices(train_pairs, protocol, maps, geometry))
    dev_leaf, dev_parent = (torch.as_tensor(arr) for arr in label_indices(dev_pairs, protocol, maps, geometry))
    parent_counts = parent_counts_from_pairs(train_pairs, protocol, geometry)
    common = pregenerate_common(seed, len(train_pairs), protocol, geometry)
    if target_updates > len(common["batches"]):
        raise HarnessError("target exceeds pre-generated batch stream", 2)
    rng = {"backbone_sha256": common["backbone_sha256"], "batches_sha256": common["batches_sha256"], "streams": common["streams"]}
    exposure_sha = sha256_bytes(exposure_jsonl(train_pairs))
    model = TinyHeadModel(arm, protocol, geometry, common["backbone"], head_init(arm, seed, protocol, parent_counts))
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["optimizer"]["lr"])
    start_step = 1
    if resume is not None:
        start_step = int(resume["step"]) + 1
        if resume.get("next_batch_index") != start_step - 1 or resume.get("rng") != rng or resume.get("exposure_sha256") != exposure_sha or resume.get("arm") != arm or resume.get("seed") != seed:
            raise HarnessError("resume terminal/batch/exposure/arm binding mismatch", 4)
        load_state_strict(model, resume["model_state"], "resume terminal")
        if state_digest(resume["model_state"]) != resume.get("state_sha256"):
            raise HarnessError("resume state fingerprint mismatch", 4)
        optimizer.load_state_dict(resume["optimizer_state"])
        if not optimizer.state or any(int(value["step"].item()) != start_step - 1 for value in optimizer.state.values()):
            raise HarnessError("resume Adam step mismatch", 4)
    if start_step > target_updates or (start_step - 1) % eval_every:
        raise HarnessError("resume must continue a terminal grid endpoint", 2)
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
    initial_q = {"theoretical": {"pi_parent": priors["pi_parent"].tolist(), "pi_leaf": priors["pi_leaf"].tolist()},
                 "actual_forward_mean": {"p3": p3_mean, "p5": p5_mean}, "probe_rows": int(probe.shape[0]),
                 "source": "resumed-terminal" if resume is not None else "fresh-init"}
    history, checkpoints = [], {}
    started, cpu_started = time.time(), time.process_time()
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
            checkpoints[step] = copy.deepcopy(model.state_dict())
            history.append({"step": step, "state_sha256": state_digest(checkpoints[step]), **metrics})
    _check_params_finite(model, f"{arm}-{seed} final parameters")
    if not history:
        raise HarnessError(f"{arm}-{seed} produced no dev evaluation points", 1)
    coarse, fine = select_coarse_checkpoint(history), select_fine_checkpoint(history, arm, protocol)
    final_state = copy.deepcopy(model.state_dict())
    terminal = {"state": final_state, "optimizer_state": copy.deepcopy(optimizer.state_dict()), "step": target_updates,
                "next_batch_index": target_updates, "state_sha256": state_digest(final_state), "arm": arm, "seed": seed,
                "rng": rng, "exposure_sha256": exposure_sha}
    return {"arm": arm, "seed": seed, "updates": target_updates, "start_step": start_step,
            "executed_updates": target_updates - start_step + 1, "resumed": resume is not None, "eval_every": eval_every,
            "history": history, "coarse": {"step": coarse["step"], "dev_coarse_nll": coarse["dev_coarse_nll"]},
            "fine": None if fine is None else {"step": fine["step"], "dev_fine_nll": fine["dev_fine_nll"]},
            "checkpoint_states": {"coarse": checkpoints[coarse["step"]], "fine": None if fine is None else checkpoints[fine["step"]]},
            "all_checkpoint_states": checkpoints, "terminal": terminal, "initial_q": initial_q, "rng": rng,
            "batch_slice_sha256": _array_digest({"batches": common["batches"][start_step - 1:target_updates]}),
            "eval_calls": len(history), "wall_seconds": round(time.time() - started, 3),
            "cpu_seconds": round(time.process_time() - cpu_started, 3), "torch_threads": torch.get_num_threads()}
def _np_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = values - np.where(np.isneginf(maximum), 0.0, maximum)
        total = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        result = maximum + total
    return np.where(np.isneginf(maximum), -np.inf, result).squeeze(axis)


def _np_log_group_sums(logp5: np.ndarray, maps: dict) -> np.ndarray:
    return np.stack([_np_logsumexp(logp5[:, maps["groups"][parent]], axis=1) for parent in maps["parents"]], axis=1)


def _np_project(p5: np.ndarray, maps: dict) -> np.ndarray:
    return np.stack([p5[:, maps["groups"][parent]].sum(axis=1) for parent in maps["parents"]], axis=1)


def project_5_to_3(p5, maps: dict):
    if isinstance(p5, torch.Tensor):
        return torch.stack([p5[:, maps["groups"][parent]].sum(dim=-1) for parent in maps["parents"]], dim=-1)
    return _np_project(np.asarray(p5, dtype=np.float64), maps)


def log_probs(model: TinyHeadModel, pairs: Sequence[tuple[int, int]], geometry: Geometry) -> tuple[np.ndarray, np.ndarray | None]:
    x = torch.as_tensor(encode_pairs(pairs, geometry))
    with torch.no_grad():
        output = forward_log(model, x)
        if model.arm == "A":
            logp3 = output["parent_logp"].numpy().astype(np.float64)
            logp5 = None
        else:
            logp5 = output["leaf_logp"].numpy().astype(np.float64)
            logp3 = _np_log_group_sums(logp5, model.maps)
    return logp3, logp5


def validate_log_prob_matrix(values: np.ndarray, classes: int, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != classes or len(array) == 0 or not np.isfinite(array).all():
        raise HarnessError(f"{what}: non-finite/empty/wrong-shape log probabilities (including zero/all-Inf groups); no epsilon fabrication", 1)
    if (array > 1e-6).any() or not np.all(np.abs(_np_logsumexp(array, axis=1)) <= 1e-6):
        raise HarnessError(f"{what}: log probabilities violate simplex", 1)
    return array


def fit_temperature(logp3: np.ndarray, parent_idx: np.ndarray, protocol: dict) -> dict:
    spec = protocol["evaluation"]["temperature"]
    low, high = spec["range"]
    grid = np.geomspace(float(low), float(high), int(spec["grid_points"]))
    logp3 = validate_log_prob_matrix(logp3, 3, "temperature calibration")
    parent_idx = np.asarray(parent_idx, dtype=np.int64)
    if parent_idx.shape != (len(logp3),) or ((parent_idx < 0) | (parent_idx >= 3)).any():
        raise HarnessError("calibration gold shape/class mismatch", 2)
    rows = np.arange(len(parent_idx))

    def nll(value):
        scaled = apply_temperature_to_logp3(logp3, value)
        return float(-scaled[rows, parent_idx].mean())

    scored = [(float(t), nll(float(t))) for t in grid]
    best = min(scored, key=lambda pair: (pair[1], pair[0]))
    return {"temperature": best[0], "nll_before": nll(1.0), "nll_after": best[1], "grid_points": len(scored),
            "range": [float(low), float(high)], "usable_rows": len(logp3), "excluded_non_finite": 0,
            "invalid_rows_policy": "fail the scoring attempt; never drop or fabricate calibration rows"}


def apply_temperature_to_logp3(logp3: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(float(temperature)) or not 0.25 <= temperature <= 4.0:
        raise HarnessError("temperature must be finite in [0.25,4]", 2)
    scaled = validate_log_prob_matrix(logp3, 3, "parent temperature") / float(temperature)
    return scaled - _np_logsumexp(scaled, axis=1)[:, None]


def apply_temperature_to_logp5(logp5: np.ndarray, temperature: float, maps: dict) -> np.ndarray:
    logp5 = validate_log_prob_matrix(logp5, len(maps["leaves"]), "leaf temperature")
    logp3 = _np_log_group_sums(logp5, maps)
    logp3_scaled = apply_temperature_to_logp3(logp3, temperature)
    parent_of_leaf = np.asarray(maps["parent_of_leaf"], dtype=np.int64)
    return logp3_scaled[:, parent_of_leaf] + (logp5 - logp3[:, parent_of_leaf])


def policy_coverage_description(logp3: np.ndarray, parent_idx: np.ndarray, protocol: dict) -> dict:
    coverage_target = float(protocol["evaluation"]["policy_coverage"])
    logp3 = validate_log_prob_matrix(logp3, 3, "policy description")
    finite = np.ones(len(logp3), dtype=bool)
    excluded = 0
    gold = np.asarray(parent_idx)
    if gold.shape != (len(logp3),) or ((gold < 0) | (gold >= 3)).any():
        raise HarnessError("policy gold shape/class mismatch", 2)
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


def _two_way_entries(pairs, logp3_forward: np.ndarray, logp3_reverse: np.ndarray, protocol: dict) -> list:
    parents = protocol["semantics"]["parents"]
    if np.asarray(logp3_forward).shape != (len(pairs), len(parents)) or np.asarray(logp3_reverse).shape != (len(pairs), len(parents)):
        raise HarnessError("two-way matrix shape mismatch", 4)
    c_index, entries = parents.index("C"), []
    for forward, reverse in zip(logp3_forward, logp3_reverse):
        reason = None
        if not np.isfinite(forward).all():
            reason = "forward_non_finite"
        elif not np.isfinite(reverse).all():
            reason = "reverse_non_finite"
        elif not probability_simplex(np.exp(forward)) or not probability_simplex(np.exp(reverse)):
            reason = "two_way_simplex_violation"
        if reason:
            entries.append({"status": "failed", "reason": reason, "hard_a": None, "hard_b": None, "combo": None,
                            "combo_legal": False, "joint_feasible": None, "c_edge_gap": None,
                            "logp3_reverse": None if not np.isfinite(reverse).all() else reverse.tolist()})
            continue
        probs_forward, probs_reverse = np.exp(forward), np.exp(reverse)
        hard_forward, hard_reverse = parents[int(np.argmax(forward))], parents[int(np.argmax(reverse))]
        entries.append({"status": "evaluated", "reason": None, "hard_a": hard_forward, "hard_b": hard_reverse,
                        "combo": hard_forward + hard_reverse, "combo_legal": combination_leaf(hard_forward, hard_reverse, protocol) is not None,
                        "joint_feasible": joint_marginals_feasible(probs_forward.tolist(), probs_reverse.tolist(), protocol),
                        "c_edge_gap": float(abs(probs_forward[c_index] - probs_reverse[c_index])), "logp3_reverse": reverse.tolist()})
    return entries


def _question_rows(view: dict, pairs, logp3_raw, logp5_raw, logp3_cal, logp5_cal, protocol: dict, maps: dict, geometry: Geometry, two_way=None) -> list[dict]:
    parents, leaves, parent_of_leaf = maps["parents"], maps["leaves"], maps["parent_of_leaf"]
    matrices = {"logp3": logp3_raw, "logp5": logp5_raw, "logp3_calibrated": logp3_cal, "logp5_calibrated": logp5_cal}
    for name, matrix in matrices.items():
        if matrix is not None and np.asarray(matrix).shape != (len(pairs), len(leaves) if "5" in name else len(parents)):
            raise HarnessError(f"question matrix shape mismatch: {name}", 4)
    if two_way is not None and len(two_way) != len(pairs):
        raise HarnessError("two-way entries do not align with all questions", 4)
    rows = []
    for index, (p_mask, h_mask) in enumerate(pairs):
        gold_leaf = maps["leaf_index"][leaf_of(p_mask, h_mask, full_mask=geometry.full_mask)]
        gold_parent = parent_of_leaf[gold_leaf]
        vectors = {name: None if matrix is None else np.asarray(matrix[index], dtype=np.float64) for name, matrix in matrices.items()}
        finite = all(value is None or bool(np.isfinite(value).all()) for value in vectors.values())
        row = {"candidate": view["candidate"], "arm": view["arm"], "seed": view["seed"], "view": view["view"], "step": view["step"],
               "model_sha256": view["model_sha256"], "index": index, "p": int(p_mask), "h": int(h_mask),
               "gold_leaf": leaves[gold_leaf], "gold_parent": parents[gold_parent]}
        for name, value in vectors.items():
            row[name] = None if value is None or not np.isfinite(value).all() else value.tolist()
        p3, p5 = None, None
        simplex = finite
        if finite:
            for name, value in vectors.items():
                probs = None if value is None else np.exp(value)
                row[name.replace("logp", "p")] = None if probs is None else probs.tolist()
                simplex = simplex and (probs is None or probability_simplex(probs))
            p3 = np.exp(vectors["logp3"])
            p5 = None if vectors["logp5"] is None else np.exp(vectors["logp5"])
        else:
            for name in vectors:
                row[name.replace("logp", "p")] = None
        usable = finite and simplex
        if usable:
            parent_hit = int(np.argmax(p3))
            fine_hit = None if p5 is None else int(np.argmax(p5))
            parent_first = None if p5 is None else leaves[max(maps["groups"][parents[parent_hit]], key=lambda member: p5[member])]
            row.update({"coarse_logp_gold": float(vectors["logp3"][gold_parent]), "fine_logp_gold": None if p5 is None else float(vectors["logp5"][gold_leaf]),
                        "coarse_nll": float(-vectors["logp3"][gold_parent]), "fine_nll": None if p5 is None else float(-vectors["logp5"][gold_leaf]),
                        "coarse_nll_calibrated": None if vectors["logp3_calibrated"] is None else float(-vectors["logp3_calibrated"][gold_parent]),
                        "fine_nll_calibrated": None if vectors["logp5_calibrated"] is None else float(-vectors["logp5_calibrated"][gold_leaf]),
                        "parent_map": parents[parent_hit], "fine_map": None if fine_hit is None else leaves[fine_hit], "parent_first": parent_first,
                        "map_disagreement": None if fine_hit is None else bool(parent_of_leaf[fine_hit] != parent_hit),
                        "projection_residual": None if p5 is None else float(np.max(np.abs(p3 - _np_project(p5[None, :], maps)[0])))})
        else:
            for name in ("coarse_logp_gold", "fine_logp_gold", "coarse_nll", "fine_nll", "coarse_nll_calibrated", "fine_nll_calibrated",
                         "parent_map", "fine_map", "parent_first", "map_disagreement", "projection_residual"):
                row[name] = None
        cal_parent = None if not usable or vectors["logp3_calibrated"] is None else int(np.argmax(vectors["logp3_calibrated"]))
        cal_fine = None if not usable or vectors["logp5_calibrated"] is None else int(np.argmax(vectors["logp5_calibrated"]))
        row["parent_map_calibrated"] = None if cal_parent is None else parents[cal_parent]
        row["fine_map_calibrated"] = None if cal_fine is None else leaves[cal_fine]
        row["parent_first_calibrated"] = None if cal_parent is None or cal_fine is None else leaves[max(maps["groups"][parents[cal_parent]], key=lambda member: vectors["logp5_calibrated"][member])]
        entry = None if two_way is None else two_way[index]
        if two_way is not None and entry is None:
            entry = {"status": "failed", "reason": "missing_reverse_result", "combo_legal": False, "joint_feasible": None, "c_edge_gap": None}
        row["two_way"] = entry
        row["two_way_status"] = "not_requested" if two_way is None else entry.get("status", "evaluated")
        failure = "ok"
        if not finite:
            failure = "non_finite"
        elif not simplex:
            failure = "simplex_violation"
        elif row["two_way_status"] == "failed":
            failure = entry["reason"]
        elif entry is not None and not entry["combo_legal"]:
            failure = "hard_combo_unmappable"
        row.update({"finite": finite, "simplex_ok": simplex, "failure_reason": failure})
        rows.append(row)
    return rows


def _class_stats(rows: list[dict], gold_key: str, pred_key: str, names: Sequence[str]) -> dict:
    index = {name: i for i, name in enumerate(names)}
    gold = np.asarray([index[row[gold_key]] for row in rows], dtype=np.int64)
    pred = np.asarray([index[row[pred_key]] for row in rows], dtype=np.int64)
    stats = classification_metrics(pred, gold, len(names))
    return {"macro_f1": stats["macro_f1"], "per_class_recall": dict(zip(names, stats["recall"])),
            "support": dict(zip(names, stats["support"])), "zero_division": stats["zero_division"],
            "confusion": {f"{names[g]}->{names[p]}": stats["confusion"][g][p] for g in range(len(names)) for p in range(len(names))}}


def summarize_rows(rows: list[dict], protocol: dict, maps: dict) -> dict:
    n = len(rows)
    finite_rows = [row for row in rows if row["finite"]]
    coarse_rows = [row for row in finite_rows if row["simplex_ok"] and row.get("parent_map") is not None]
    fine_rows = [row for row in coarse_rows if row.get("fine_map") is not None]

    def mean_of(key, subset):
        values = [row[key] for row in subset if row.get(key) is not None and math.isfinite(row[key])]
        return float(np.mean(values)) if values else None

    coarse_stats = _class_stats(coarse_rows, "gold_parent", "parent_map", maps["parents"])
    fine_stats = _class_stats(fine_rows, "gold_leaf", "fine_map", maps["leaves"])
    two_rows = [row for row in rows if row.get("two_way") is not None]
    two_way = None
    if two_rows:
        evaluated = [row for row in two_rows if row.get("two_way_status", "evaluated") != "failed"]
        legal = sum(row["two_way"]["combo_legal"] for row in evaluated)
        gaps = [row["two_way"]["c_edge_gap"] for row in evaluated]
        tolerance = float(protocol["evaluation"]["joint_marginals"]["c_edge_tolerance"])
        two_way = {"n": len(two_rows), "attempted": len(two_rows), "evaluated": len(evaluated), "failed": len(two_rows) - len(evaluated),
                   "coverage": legal / len(two_rows), "invalid": len(two_rows) - legal,
                   "hard_combo_unmappable": len(evaluated) - legal,
                   "c_edge_mismatch": sum(gap > tolerance for gap in gaps), "c_edge_mean_gap": float(np.mean(gaps)) if gaps else None,
                   "prob_joint_evaluated": len(evaluated), "prob_joint_incompatible": sum(not row["two_way"]["joint_feasible"] for row in evaluated),
                   "denominator": "all attempted bidirectional questions, including failed reverse calls"}
    reasons = {}
    for row in rows:
        reasons[row["failure_reason"]] = reasons.get(row["failure_reason"], 0) + 1
    coarse_correct = sum(row["parent_map"] == row["gold_parent"] for row in coarse_rows)
    fine_correct = sum(row["fine_map"] == row["gold_leaf"] for row in fine_rows)
    projection = [row["projection_residual"] for row in rows if row.get("projection_residual") is not None]
    calibrated = {}
    for kind, gold_key, classes in (("coarse", "gold_parent", maps["parents"]), ("fine", "gold_leaf", maps["leaves"])):
        pred_key = "parent_map_calibrated" if kind == "coarse" else "fine_map_calibrated"
        subset = [row for row in coarse_rows if row.get(pred_key) is not None]
        stats = _class_stats(subset, gold_key, pred_key, classes)
        calibrated[kind] = {**stats, "n": len(subset), "nll": mean_of(f"{kind}_nll_calibrated", subset),
                            "accuracy": sum(row[pred_key] == row[gold_key] for row in subset) / len(subset) if subset else None,
                            "macro_f1": stats["macro_f1"] if subset else None}
    return {"n": n, "usable": len(coarse_rows), "excluded_non_finite": n - len(finite_rows), "failure_reasons": reasons, "calibrated_metrics": calibrated,
            "coarse_nll": mean_of("coarse_nll", coarse_rows), "coarse_nll_calibrated": mean_of("coarse_nll_calibrated", coarse_rows),
            "fine_nll": mean_of("fine_nll", fine_rows), "fine_nll_calibrated": mean_of("fine_nll_calibrated", fine_rows),
            "coarse_accuracy": coarse_correct / len(coarse_rows) if coarse_rows else None,
            "fine_accuracy": fine_correct / len(fine_rows) if fine_rows else None,
            "coarse_accuracy_all_attempt": coarse_correct / n if n else None,
            "fine_correct_all_attempt": fine_correct / n if n else None,
            "parent_first_accuracy": sum(row["parent_first"] == row["gold_leaf"] for row in fine_rows) / len(fine_rows) if fine_rows else None,
            "coarse_macro_f1": coarse_stats["macro_f1"] if coarse_rows else None, "fine_macro_f1": fine_stats["macro_f1"] if fine_rows else None,
            "coarse_per_class_recall": coarse_stats["per_class_recall"], "fine_per_class_recall": fine_stats["per_class_recall"],
            "coarse_support": coarse_stats["support"], "fine_support": fine_stats["support"], "zero_division": 0,
            "coarse_confusion": coarse_stats["confusion"], "fine_confusion": fine_stats["confusion"],
            "map_disagreement": sum(row["map_disagreement"] for row in fine_rows), "projection_residual_max": max(projection) if projection else None,
            "finite_violations": n - len(finite_rows), "simplex_violations": sum(not row["simplex_ok"] for row in finite_rows),
            "hard_combo_unmappable": sum(row["failure_reason"] == "hard_combo_unmappable" for row in rows), "two_way": two_way}


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return str(value)


def rows_csv(rows: list[dict]) -> str:
    import csv
    import io

    if not rows:
        raise HarnessError("empty per-question output", 4)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(value) for key, value in row.items()})
    return buffer.getvalue()


def verify_question_roundtrip(expected: list[dict], json_rows: list[dict], csv_text: str, views: Sequence[dict], pairs, maps: dict) -> dict:
    import csv
    import io

    if not expected or json_rows != expected or len(expected) != len(views) * len(pairs):
        raise HarnessError("JSONL question id/order/matrix round-trip mismatch", 4)
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(csv_rows) != len(expected):
        raise HarnessError("CSV question count mismatch", 4)
    for position, row in enumerate(json_rows):
        view, index = views[position // len(pairs)], position % len(pairs)
        if row["candidate"] != view["candidate"] or row["index"] != index or (row["p"], row["h"]) != pairs[index]:
            raise HarnessError("question candidate/id/membership mismatch", 4)
        for key in ("p3", "p5", "logp3", "logp5", "p3_calibrated", "p5_calibrated", "logp3_calibrated", "logp5_calibrated"):
            value = row.get(key)
            if value is not None and (len(value) != (len(maps["leaves"]) if "5" in key else len(maps["parents"])) or not np.isfinite(value).all()):
                raise HarnessError(f"question vector shape/finite mismatch: {key}", 4)
        for suffix in ("3", "5", "3_calibrated", "5_calibrated"):
            prob, logp = row.get(f"p{suffix}"), row.get(f"logp{suffix}")
            if prob is not None and (logp is None or not np.array_equal(np.asarray(prob), np.exp(np.asarray(logp)))):
                raise HarnessError("question probability/log-vector mismatch", 4)
        if csv_rows[position] != {key: _csv_cell(value) for key, value in row.items()}:
            raise HarnessError("CSV question id/shape/matrix/scalar round-trip mismatch", 4)
    return {"jsonl_rows_verified": len(json_rows), "csv_rows_verified": len(csv_rows), "full_matrix_and_id_comparison": True}
