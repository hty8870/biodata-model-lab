"""头方案 v5（r3）研究-only 的模型学习数值模块（研究-only）。

- 分区与有序曝光：study 驱动的互补 families（formal=N10 final-first 且 final 成员均 >255；smoke=N8 legacy 排布），
  实体/互补/cross-role 严格隔离；输入是原始 bitmap 拼接（无 relational features）。
- 模型数值：共同骨干/损失/训练/dev 指标/粗·细双 estimand 选点/G-based 延长/温度/逐题与汇总。
  损失全 log 空间；NaN/非有限非零退出；成功 JSON allow_nan=False。
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
    rng = np.random.default_rng(np.random.PCG64([seed, 1049 + ord(arm)]))
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
            state[f"branch_{key}_w"] = rng.normal(0.0, scale, (1, hidden)).astype(np.float32)
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
    maps = index_maps(protocol)
    spec = protocol["training"]
    eval_every = spec["eval_every"]
    target_updates = int(updates if updates is not None else spec["updates"])
    train_x = torch.as_tensor(encode_pairs(train_pairs, geometry))
    dev_x = torch.as_tensor(encode_pairs(dev_pairs, geometry))
    train_leaf, train_parent = (torch.as_tensor(arr) for arr in label_indices(train_pairs, protocol, maps, geometry))
    dev_leaf, dev_parent = (torch.as_tensor(arr) for arr in label_indices(dev_pairs, protocol, maps, geometry))
    parent_counts = parent_counts_from_pairs(train_pairs, protocol, geometry)
    common = pregenerate_common(seed, len(train_pairs), protocol, geometry)
    model = TinyHeadModel(arm, protocol, geometry, common["backbone"], head_init(arm, seed, protocol, parent_counts))
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
def _np_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = values - maximum
        total = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return (maximum + total).squeeze(axis)


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

    def nll(value: float) -> float:
        scaled = usable_logp3 / value
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


def _two_way_entries(pairs, logp3_forward: np.ndarray, logp3_reverse: np.ndarray, protocol: dict) -> list:
    parents = protocol["semantics"]["parents"]
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


def _question_rows(view: dict, pairs, logp3_raw, logp5_raw, logp3_cal, logp5_cal, protocol: dict, maps: dict, geometry: Geometry, two_way=None) -> list[dict]:
    parents = maps["parents"]
    leaves = maps["leaves"]
    parent_of_leaf = maps["parent_of_leaf"]
    rows: list[dict] = []
    for index, (p_mask, h_mask) in enumerate(pairs):
        gold_leaf_index = maps["leaf_index"][leaf_of(p_mask, h_mask, full_mask=geometry.full_mask)]
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


def rows_csv(rows: list[dict]) -> str:
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
