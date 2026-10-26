"""研究-only 有限集合语义、完整 cohort 合取与受控最小 core、实值投影回执。

关系标签不代表知识库状态或编辑授权。Core 身份绑定 frame/cohort/成员语义，变化单列 review；
资源不足显式拒绝，不产“无新增冲突”证书。回执按 content/full_conditional/overlap 模式求 oracle。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

SOURCE_MODULES = ("head_design.py", "head_design_learning.py", "head_design_semantics.py")


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
    import torch

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


def source_bundle_info() -> dict:
    directory = Path(__file__).parent
    files = {}
    digest = hashlib.sha256()
    for name in SOURCE_MODULES:
        path = directory / name
        if not path.exists():
            raise HarnessError(f"source bundle module missing: {name}", 2)
        payload = path.read_bytes()
        files[name] = sha256_bytes(payload)
        digest.update(name.encode("utf-8"))
        digest.update(payload)
    return {"sha256": digest.hexdigest(), "files": files}


def source_bundle_sha256() -> str:
    return source_bundle_info()["sha256"]


def require_mask(value, *, full_mask: int, name: str = "mask", allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise HarnessError(f"{name} must be a plain int in [0, {full_mask}]", 2)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(f"{name} must be a plain int in [0, {full_mask}]", 2)
    if not 0 <= value <= full_mask:
        raise HarnessError(f"{name} out of range [0, {full_mask}]: {value}", 2)
    return value


def load_protocol(path: Path | str | None = None) -> tuple[dict, str]:
    target = Path(path) if path is not None else protocol_default_path()
    payload = target.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    validate_protocol(protocol)
    return protocol, sha256_bytes(payload)


@dataclass(frozen=True)
class Geometry:
    study: str
    world_bits: int
    world_count: int
    full_mask: int
    input_bits_per_side: int
    final_novelty_min_mask: int | None


def study_geometry(protocol: dict, study: str) -> Geometry:
    studies = protocol["semantics"]["studies"]
    if study not in studies:
        raise HarnessError(f"unknown study {study}", 2)
    domain = studies[study]["domain"]
    return Geometry(
        study=study,
        world_bits=int(domain["world_bits"]),
        world_count=int(domain["world_count"]),
        full_mask=int(domain["full_mask"]),
        input_bits_per_side=int(domain["input_bits_per_side"]),
        final_novelty_min_mask=domain.get("final_novelty_min_mask"),
    )


def study_roles(protocol: dict) -> tuple[str, ...]:
    return tuple(sorted(protocol["semantics"]["studies"]))


def complement_families(geometry: Geometry) -> list[tuple[int, int]]:
    full = geometry.full_mask
    families: list[tuple[int, int]] = []
    seen: set[int] = set()
    for mask in range(1, full + 1):
        partner = full ^ mask
        if partner == 0 or partner in seen:
            continue
        seen.add(mask)
        seen.add(partner)
        low, high = sorted((mask, partner), key=lambda m: (bin(m).count("1"), m))
        families.append((low, high))
    return families


def strata_families(geometry: Geometry) -> dict[int, list[tuple[int, int]]]:
    strata: dict[int, list[tuple[int, int]]] = {}
    for family in complement_families(geometry):
        key = bin(family[0]).count("1")
        strata.setdefault(key, []).append(family)
    return strata


def validate_protocol(protocol: dict) -> None:
    if protocol.get("revision") != "r4":
        raise HarnessError("protocol revision must be r4", 2)
    if protocol.get("source_bundle", {}).get("modules") != list(SOURCE_MODULES):
        raise HarnessError("protocol source module list inconsistent with the complete bundle", 2)
    semantics = protocol["semantics"]
    leaves = semantics["leaves"]
    parents = semantics["parents"]
    if len(set(leaves)) != len(leaves) or len(set(parents)) != len(parents):
        raise HarnessError("protocol duplicate leaf/parent names", 2)
    if set(semantics["leaf_parent"]) != set(leaves):
        raise HarnessError("protocol leaf_parent must cover exactly the leaves", 2)
    if not set(semantics["leaf_parent"].values()) <= set(parents):
        raise HarnessError("protocol leaf_parent maps outside parents", 2)
    for combo, leaf in semantics["combination_leaf"].items():
        if len(combo) != 2 or combo[0] not in parents or combo[1] not in parents or leaf not in leaves:
            raise HarnessError("protocol combination_leaf entry invalid", 2)
    if not set(semantics["combination_leaf"]) <= set(semantics["legal_bidirectional"]):
        raise HarnessError("protocol combination_leaf must stay within legal bidirectional combos", 2)
    if set(semantics["swap_map"]) != set(leaves):
        raise HarnessError("protocol swap_map must cover the leaves", 2)
    for study, config in semantics["studies"].items():
        if any(type(config["domain"].get(key)) is not int for key in ("world_bits", "world_count", "full_mask", "input_bits_per_side")):
            raise HarnessError(f"study {study}: geometry fields must be plain integers", 2)
        geometry = study_geometry(protocol, study)
        if geometry.world_count != geometry.world_bits or geometry.world_bits < 1 or geometry.input_bits_per_side != geometry.world_bits:
            raise HarnessError(f"study {study}: positive world_bits, world_count and input_bits_per_side must agree", 2)
        if geometry.full_mask != (1 << geometry.world_bits) - 1:
            raise HarnessError(f"study {study}: full_mask inconsistent", 2)
        expected_leaf = config["expected_leaf_counts"]
        expected_parent = config["expected_parent_counts"]
        if set(expected_leaf) != set(leaves) or set(expected_parent) != set(parents):
            raise HarnessError(f"study {study}: expected counts must cover leaves/parents", 2)
        total = int(config["pair_total"])
        if sum(expected_leaf.values()) != total or sum(expected_parent.values()) != total:
            raise HarnessError(f"study {study}: expected counts do not sum to pair_total", 2)
        from_leaf = {parent: 0 for parent in parents}
        for leaf in leaves:
            from_leaf[semantics["leaf_parent"][leaf]] += expected_leaf[leaf]
        if from_leaf != expected_parent:
            raise HarnessError(f"study {study}: parent counts inconsistent with leaves", 2)
        partition = protocol["partition"]["studies"][study]
        strata = strata_families(geometry)
        for key, counts in partition["strata_min_side"].items():
            if sum(counts.values()) != len(strata.get(int(key), [])):
                raise HarnessError(f"study {study}: strata_min_side[{key}] does not partition its families", 2)
        role_totals = {role: sum(partition["strata_min_side"][k][role] for k in partition["strata_min_side"]) for role in ("train", "dev", "calibration", "final")}
        for role, count in role_totals.items():
            if partition["expected_families"][role] != count:
                raise HarnessError(f"study {study}: expected_families[{role}] inconsistent", 2)
        if partition["expected_families"]["total"] != len(complement_families(geometry)):
            raise HarnessError(f"study {study}: family total mismatch", 2)
        for role in ("train", "dev", "calibration", "final"):
            head = role_totals[role] * 2 + (1 if role == "train" else 0)
            if partition["expected_entities"][role] != head:
                raise HarnessError(f"study {study}: expected_entities[{role}] inconsistent", 2)
        pools = partition["calibration_pools"]
        if pools["prob_families"] + pools["policy_families"] != role_totals["calibration"]:
            raise HarnessError(f"study {study}: calibration pools must cover the calibration families", 2)
        if partition["final_first"]:
            candidates = partition.get("final_candidate_families") or {}
            for key in partition["strata_min_side"]:
                needed = partition["strata_min_side"][key]["final"]
                available = int(candidates.get(key, 0))
                if available < needed:
                    raise HarnessError(f"study {study}: stratum {key} lacks enough novel final candidates", 2)
    temperature = protocol["evaluation"]["temperature"]
    low, high = temperature["range"]
    if not 0 < low < 1 < high:
        raise HarnessError("protocol temperature range must straddle 1.0", 2)
    extension = protocol["training"]["extension"]
    if extension["decide_before_step"] != extension["trigger_step"] + 1:
        raise HarnessError("protocol extension must decide before any step beyond the trigger", 2)
    if extension["windows"]["early"] >= extension["windows"]["late"]:
        raise HarnessError("protocol extension windows invalid", 2)
    selection = protocol["training"]["selection"]
    if selection["fine_estimand"]["arms"] != ["C", "H"] or selection["coarse_estimand"]["metric"] != "dev_coarse_nll":
        raise HarnessError("protocol selection rules invalid", 2)
    if set(protocol["edit_audit"]["codes"]) != set(AUDIT_CODES):
        raise HarnessError("protocol audit codes do not match the harness catalog", 2)


@dataclass(frozen=True)
class Verdict:
    status: str
    leaf: str | None
    reason: str | None


def leaf_of(p_mask: int, h_mask: int, *, full_mask: int | None = None) -> str:
    if full_mask is not None:
        require_mask(p_mask, full_mask=full_mask, name="p_mask")
        require_mask(h_mask, full_mask=full_mask, name="h_mask")
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


def parent_of_masks(p_mask: int, h_mask: int, *, full_mask: int) -> str:
    require_mask(p_mask, full_mask=full_mask, name="p_mask")
    require_mask(h_mask, full_mask=full_mask, name="h_mask")
    if p_mask & (full_mask ^ h_mask) == 0:
        return "E"
    if p_mask & h_mask == 0:
        return "C"
    return "N"


def classify(p_mask: int | None, h_mask: int | None, *, full_mask: int, background_consistent: bool = True) -> Verdict:
    p_value = require_mask(p_mask, full_mask=full_mask, name="p_mask", allow_none=True)
    h_value = require_mask(h_mask, full_mask=full_mask, name="h_mask", allow_none=True)
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
    full_mask: int,
    overlap_known: bool = True,
    background_consistent: bool = True,
) -> Verdict:
    p_value = require_mask(p_mask, full_mask=full_mask, name="p_mask")
    h_value = require_mask(h_mask, full_mask=full_mask, name="h_mask")
    cond_p = require_mask(cond_p_mask, full_mask=full_mask, name="cond_p_mask")
    cond_h = require_mask(cond_h_mask, full_mask=full_mask, name="cond_h_mask")
    if not overlap_known:
        return Verdict("unresolved", None, "condition_overlap_unknown")
    if cond_p & cond_h == 0:
        return Verdict("unresolved", None, "empty_condition_overlap")
    statement_p = (p_value & cond_p) | (full_mask ^ cond_p)
    statement_h = (h_value & cond_h) | (full_mask ^ cond_h)
    return classify(statement_p, statement_h, full_mask=full_mask, background_consistent=background_consistent)


def interval_content_relation(
    p_mask: int,
    h_mask: int,
    cond_p_mask: int,
    cond_h_mask: int,
    *,
    full_mask: int,
    overlap_known: bool = True,
) -> Verdict:
    p_value = require_mask(p_mask, full_mask=full_mask, name="p_mask")
    h_value = require_mask(h_mask, full_mask=full_mask, name="h_mask")
    cond_p = require_mask(cond_p_mask, full_mask=full_mask, name="cond_p_mask")
    cond_h = require_mask(cond_h_mask, full_mask=full_mask, name="cond_h_mask")
    if not overlap_known:
        return Verdict("unresolved", None, "condition_overlap_unknown")
    overlap = cond_p & cond_h
    if overlap == 0:
        return Verdict("unresolved", None, "empty_condition_overlap")
    return classify(p_value & overlap, h_value & overlap, full_mask=full_mask)


def combination_leaf(label_a: str, label_b: str, protocol: dict) -> str | None:
    return protocol["semantics"]["combination_leaf"].get(label_a + label_b)


SIMPLEX_TOLERANCE = 1e-6


def probability_simplex(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 1 and array.size and np.isfinite(array).all() and (array >= 0).all()
                and (array <= 1 + SIMPLEX_TOLERANCE).all() and abs(float(array.sum()) - 1.0) <= SIMPLEX_TOLERANCE)


def joint_marginals_feasible(marginal_a: Sequence[float], marginal_b: Sequence[float], protocol: dict) -> bool:
    parents = protocol["semantics"]["parents"]
    legal = list(protocol["semantics"]["legal_bidirectional"])
    if len(marginal_a) != len(parents) or len(marginal_b) != len(parents):
        raise HarnessError("marginal vectors must match the parent count", 2)
    if not probability_simplex(marginal_a) or not probability_simplex(marginal_b):
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
        previous = {source: None}
        queue = [source]
        while queue and sink not in previous:
            node = queue.pop(0)
            for (fr, to), cap in capacity.items():
                if fr == node and cap > 1e-12 and to not in previous:
                    previous[to] = fr
                    queue.append(to)
        if sink not in previous:
            break
        bottleneck = float("inf")
        node = sink
        while previous[node] is not None:
            bottleneck = min(bottleneck, capacity[(previous[node], node)])
            node = previous[node]
        node = sink
        while previous[node] is not None:
            capacity[(previous[node], node)] -= bottleneck
            capacity[(node, previous[node])] = capacity.get((node, previous[node]), 0.0) + bottleneck
            node = previous[node]
        flow += bottleneck
    return flow >= 1.0 - SIMPLEX_TOLERANCE


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
    leaves = protocol["semantics"]["leaves"]
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
        c_index = protocol["semantics"]["parents"].index("C")
        gaps = [abs(float(a[c_index]) - float(b[c_index])) for a, b in zip(marginal_a, marginal_b)]
        incompatible = sum(1 for a, b in zip(marginal_a, marginal_b) if not joint_marginals_feasible(a, b, protocol))
        summary["c_edge_mismatch_count"] = int(sum(1 for gap in gaps if gap > tolerance))
        summary["c_edge_mean_abs_gap"] = float(sum(gaps) / total)
        summary["c_edge_tolerance"] = tolerance
        summary["prob_joint_incompatible"] = int(incompatible)
        summary["prob_joint_incompatible_fraction"] = incompatible / total
    return summary


_VECTOR_LEAF_ORDER = (
    "equivalent",
    "old_entails_new_strict",
    "new_entails_old_strict",
    "disjoint",
    "overlap_incomparable",
)


def _vector_leaf_codes(p, h):
    overlap = p & h
    code = np.full(p.shape, 4, dtype=np.int8)
    code = np.where(overlap == 0, 3, code)
    code = np.where((overlap == h) & (p != h), 2, code)
    code = np.where((overlap == p) & (p != h), 1, code)
    code = np.where(p == h, 0, code)
    return code


def _vector_parent_codes(p, h, full_mask: int):
    code = np.full(p.shape, 1, dtype=np.int8)
    code = np.where((p & h) == 0, 2, code)
    code = np.where((p & (full_mask ^ h)) == 0, 0, code)
    return code


def exhaustive_census(protocol: dict, study: str = "formal", *, chunk: int = 256) -> dict:
    semantics = protocol["semantics"]
    leaves = semantics["leaves"]
    parents = semantics["parents"]
    geometry = study_geometry(protocol, study)
    full = geometry.full_mask
    masks = np.arange(1, full + 1, dtype=np.int64)
    leaf_code_of = {leaf: index for index, leaf in enumerate(_VECTOR_LEAF_ORDER)}
    protocol_leaf_index = {leaf: index for index, leaf in enumerate(leaves)}
    leaf_counts = np.zeros(len(leaves), dtype=np.int64)
    parent_counts = np.zeros(len(parents), dtype=np.int64)
    combo_counts: dict[str, int] = {}
    swap_violations = 0
    leaf_parent_violations = 0
    illegal_combinations = 0
    combination_mismatches = 0
    swap_expected = np.asarray([leaf_code_of[semantics["swap_map"][leaf]] for leaf in _VECTOR_LEAF_ORDER], dtype=np.int8)
    parent_expected = np.asarray([parents.index(semantics["leaf_parent"][leaf]) for leaf in _VECTOR_LEAF_ORDER], dtype=np.int8)
    legal = set(semantics["legal_bidirectional"])
    combo_code = {combo: index for index, combo in enumerate(f"{a}{b}" for a in parents for b in parents)}
    combo_expected_leaf = np.full(len(combo_code), -1, dtype=np.int8)
    for combo, leaf in semantics["combination_leaf"].items():
        combo_expected_leaf[combo_code[combo]] = leaf_code_of[leaf]
    for start in range(0, len(masks), chunk):
        p = masks[start:start + chunk][:, None]
        h = masks[None, :]
        codes = _vector_leaf_codes(p, h)
        pcodes = _vector_parent_codes(p, h, full)
        leaf_counts += np.bincount(codes.ravel(), minlength=len(_VECTOR_LEAF_ORDER))[: len(_VECTOR_LEAF_ORDER)]
        parent_counts += np.bincount(pcodes.ravel(), minlength=len(parents))
        swap_codes = _vector_leaf_codes(h, p)
        swap_violations += int((swap_expected[codes] != swap_codes).sum())
        leaf_parent_violations += int((parent_expected[codes] != pcodes).sum())
        pcodes_swap = _vector_parent_codes(h, p, full)
        combos = pcodes.astype(np.int64) * len(parents) + pcodes_swap.astype(np.int64)
        for index, (combo, code_index) in enumerate(combo_code.items()):
            combo_counts[combo] = combo_counts.get(combo, 0) + int((combos == index).sum())
        for combo, index in combo_code.items():
            if combo not in legal:
                illegal_combinations += int((combos == index).sum())
        expected_leaf = combo_expected_leaf[combos]
        combination_mismatches += int(((expected_leaf >= 0) & (expected_leaf != codes)).sum())
    leaf_counts_by_name = {leaf: int(leaf_counts[protocol_leaf_index[leaf]]) for leaf in leaves}
    parent_counts_by_name = {parent: int(parent_counts[parents.index(parent)]) for parent in parents}
    conservation_violations = 0
    for parent in parents:
        from_leaf = sum(leaf_counts_by_name[leaf] for leaf in leaves if semantics["leaf_parent"][leaf] == parent)
        if from_leaf != parent_counts_by_name[parent]:
            conservation_violations += 1
    config = semantics["studies"][study]
    matches_expected = leaf_counts_by_name == config["expected_leaf_counts"] and parent_counts_by_name == config["expected_parent_counts"]
    return {
        "study": study,
        "geometry": {"world_bits": geometry.world_bits, "full_mask": geometry.full_mask, "pair_total": geometry.full_mask ** 2},
        "pair_total": full * full,
        "leaf_counts": leaf_counts_by_name,
        "parent_counts": parent_counts_by_name,
        "combo_counts": dict(sorted(combo_counts.items())),
        "swap_violations": swap_violations,
        "leaf_parent_violations": leaf_parent_violations,
        "conservation_violations": conservation_violations,
        "illegal_combinations": illegal_combinations,
        "combination_mismatches": combination_mismatches,
        "matches_expected": matches_expected,
    }


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
    "evidence_lineage_loss",
    "undeclared_change",
    "unmentioned_field_changed",
    "unauthorized_semantics_change",
    "new_semantic_conflict",
    "changed_existing_conflict_review",
)


@dataclass(frozen=True)
class EditObject:
    object_id: str
    cohort: str
    binding_id: str
    version: int
    mask: int
    guard: int
    conjunctive: bool
    content_format: str
    guard_days: int
    readonly_until: str
    condition: str
    ttl: str
    scope: tuple[str, ...]
    evidence: tuple[str, ...]
    lineage: tuple[str, ...]
    dependencies: tuple[str, ...]
    notes: str = ""
    frame_id: str = "finite-frame"


@dataclass(frozen=True)
class EditLibrary:
    version: int
    full_mask: int
    objects: tuple[EditObject, ...]
    frame_id: str = "finite-frame"


@dataclass(frozen=True)
class EditIntent:
    target_ids: tuple[str, ...]
    allowed_ops: frozenset
    declared_scope: tuple[str, ...]
    read_basis_version: int
    joint_claims: tuple[tuple[str, ...], ...] = ()
    supersede_ref: str | None = None


@dataclass(frozen=True)
class RelationReceipt:
    object_id_a: str
    object_id_b: str
    version_a: int
    version_b: int
    projection: str
    context: str
    declared_label: str


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


def _validate_object_domain(library: EditLibrary, label: str) -> None:
    full = library.full_mask
    if type(full) is not int or full < 1 or full & (full + 1) or not library.frame_id:
        raise HarnessError(f"{label}: full_mask must be positive 2^N-1 with a non-empty frame", 2)
    for obj in library.objects:
        if obj.frame_id != library.frame_id:
            raise HarnessError(f"{label}:{obj.object_id}: cross-frame object rejected", 2)
        require_mask(obj.mask, full_mask=library.full_mask, name=f"{label}:{obj.object_id}:mask")
        require_mask(obj.guard, full_mask=library.full_mask, name=f"{label}:{obj.object_id}:guard")
        if not isinstance(obj.conjunctive, bool):
            raise HarnessError(f"{label}:{obj.object_id}:conjunctive must be a bool", 2)
        if not isinstance(obj.version, int) or isinstance(obj.version, bool) or obj.version < 1:
            raise HarnessError(f"{label}:{obj.object_id}:version must be a positive int", 2)
        if not obj.cohort or not obj.binding_id:
            raise HarnessError(f"{label}:{obj.object_id}:cohort/binding_id must be non-empty", 2)


def assertion_closure(obj: EditObject, full_mask: int) -> int:
    require_mask(obj.mask, full_mask=full_mask, name=f"{obj.object_id}:mask")
    require_mask(obj.guard, full_mask=full_mask, name=f"{obj.object_id}:guard")
    return (full_mask ^ obj.guard) | obj.mask


def _member_semantics(obj: EditObject) -> dict:
    fields = ("object_id", "version", "frame_id", "cohort", "binding_id", "mask", "guard", "conjunctive", "condition", "ttl", "scope", "guard_days", "readonly_until")
    return {field: getattr(obj, field) for field in fields}


def cohort_conjunctions(library: EditLibrary) -> dict[str, int]:
    _validate_unique_ids(library, "library")
    _validate_object_domain(library, "library")
    conjunctions: dict[str, int] = {}
    for obj in library.objects:
        if obj.conjunctive:
            conjunctions[obj.cohort] = conjunctions.get(obj.cohort, library.full_mask) & assertion_closure(obj, library.full_mask)
    return conjunctions


def minimal_inconsistent_cores(library: EditLibrary, *, max_core_size: int | None = None) -> list[dict]:
    conjunctions = cohort_conjunctions(library)
    if max_core_size is not None and (type(max_core_size) is not int or max_core_size < 1):
        raise HarnessError("max_core_size must be a positive integer or None", 2)
    cohorts: dict[str, list[EditObject]] = {}
    for obj in library.objects:
        if obj.conjunctive:
            cohorts.setdefault(obj.cohort, []).append(obj)
    cores: list[dict] = []
    for cohort in sorted(cohorts):
        objects = sorted(cohorts[cohort], key=lambda obj: obj.object_id)
        if len(objects) > 16 or library.full_mask.bit_length() > 16:
            raise HarnessError("unsupported research cohort: at most 16 members/worlds; core review incomplete", 3)
        if conjunctions[cohort] != 0:
            continue
        bound = min(len(objects), library.full_mask.bit_length())
        if max_core_size is not None and max_core_size < bound:
            raise HarnessError(f"cohort {cohort} is jointly inconsistent; core search incomplete at cap {max_core_size}, needs review", 3)
        found: list[frozenset] = []
        for size in range(1, bound + 1):
            for combo in itertools.combinations(objects, size):
                ids = frozenset(obj.object_id for obj in combo)
                if any(existing <= ids for existing in found):
                    continue
                conjunction = library.full_mask
                for obj in combo:
                    conjunction &= assertion_closure(obj, library.full_mask)
                if conjunction == 0:
                    found.append(ids)
                    semantic = {"frame_id": library.frame_id, "full_mask": library.full_mask, "cohort": cohort, "members": [_member_semantics(obj) for obj in combo]}
                    cores.append({**semantic, "fingerprint": sha256_bytes(_canonical(semantic).encode("utf-8"))})
    return cores


def _core_key(core: dict) -> str:
    return core["fingerprint"]


def _core_ids(core: dict) -> frozenset:
    return frozenset(member["object_id"] for member in core["members"])


def classify_cores(before_cores: list[dict], after_cores: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    before_keys = {_core_key(core) for core in before_cores}
    before_ids = [_core_ids(core) for core in before_cores]
    new, retained, changed = [], [], []
    for core in after_cores:
        ids = _core_ids(core)
        if _core_key(core) in before_keys:
            retained.append(core)
        elif any(old_ids <= ids or ids <= old_ids for old_ids in before_ids):
            changed.append(core)
        else:
            new.append(core)
    return new, retained, changed


def relation_receipt_binding(library: EditLibrary, object_id_a: str, object_id_b: str, mode: str) -> tuple[str, str]:
    _validate_unique_ids(library, "receipt library")
    _validate_object_domain(library, "receipt library")
    if mode not in ("content", "full_conditional", "overlap"):
        raise HarnessError(f"unknown receipt projection mode {mode}", 2)
    objects = _objects_by_id(library)
    try:
        pair = [objects[object_id_a], objects[object_id_b]]
    except KeyError as exc:
        raise HarnessError("receipt references an unknown object", 2) from exc
    projection = {"mode": mode, "objects": [_member_semantics(obj) for obj in pair]}
    context = {"frame_id": library.frame_id, "full_mask": library.full_mask, "library_version": library.version,
               "bindings": [{"object_id": obj.object_id, "cohort": obj.cohort, "binding_id": obj.binding_id} for obj in pair]}
    return _canonical(projection), _canonical(context)


def verify_relation_receipt(receipt: RelationReceipt, objects_by_id: dict, *, protocol: dict, library: EditLibrary | None = None) -> dict:
    def reject(reason, oracle=None):
        return {"ok": False, "reason": reason, "oracle_label": oracle}

    if receipt.declared_label not in protocol["semantics"]["leaves"]:
        return reject("invalid_label")
    if not receipt.projection or not receipt.context or library is None:
        return reject("missing_binding")
    obj_a, obj_b = objects_by_id.get(receipt.object_id_a), objects_by_id.get(receipt.object_id_b)
    if obj_a is None or obj_b is None:
        return reject("unknown_object")
    if receipt.version_a != obj_a.version or receipt.version_b != obj_b.version:
        return reject("version_mismatch")
    try:
        projected = json.loads(receipt.projection)
        bound_context = json.loads(receipt.context)
        mode = projected["mode"]
        expected_projection, expected_context = relation_receipt_binding(library, receipt.object_id_a, receipt.object_id_b, mode)
    except (ValueError, KeyError, TypeError, HarnessError):
        return reject("invalid_binding")
    actual_objects = _objects_by_id(library)
    if actual_objects.get(obj_a.object_id) != obj_a or actual_objects.get(obj_b.object_id) != obj_b:
        return reject("object_binding_mismatch")
    if _canonical(projected) != expected_projection or _canonical(bound_context) != expected_context:
        return reject("projection_context_mismatch")
    full = library.full_mask
    if mode == "content":
        verdict = classify(obj_a.mask, obj_b.mask, full_mask=full)
    elif mode == "full_conditional":
        verdict = conditional_full_relation(obj_a.mask, obj_b.mask, obj_a.guard, obj_b.guard, full_mask=full)
    else:
        verdict = interval_content_relation(obj_a.mask, obj_b.mask, obj_a.guard, obj_b.guard, full_mask=full)
    if verdict.status != "relation":
        return reject(verdict.reason)
    if receipt.declared_label != verdict.leaf:
        return reject("label_mismatch", verdict.leaf)
    return {"ok": True, "reason": None, "oracle_label": verdict.leaf}


def relation_receipt_diagnostics(receipts: Sequence[RelationReceipt], library: EditLibrary, *, protocol: dict) -> dict:
    objects_by_id = _objects_by_id(library)
    label_counts: dict[str, int] = {}
    mismatches = []
    for receipt in receipts:
        result = verify_relation_receipt(receipt, objects_by_id, protocol=protocol, library=library)
        if not result["ok"]:
            mismatches.append({"object_id_a": receipt.object_id_a, "object_id_b": receipt.object_id_b,
                               "declared_label": receipt.declared_label, "reason": result["reason"]})
        label_counts[receipt.declared_label] = label_counts.get(receipt.declared_label, 0) + 1
    return {"pairs": len(receipts), "labels": dict(sorted(label_counts.items())), "mixed": len(label_counts) > 1, "mismatches": mismatches}


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
    semantic_fields = ("mask", "guard", "cohort", "binding_id", "version", "conjunctive")
    for field in semantic_fields + (
        "content_format",
        "guard_days",
        "readonly_until",
        "condition",
        "ttl",
        "scope",
        "evidence",
        "lineage",
        "dependencies",
        "notes",
    ):
        old = getattr(before, field)
        new = getattr(after, field)
        if old == new:
            continue
        if field in semantic_fields:
            if "semantics" not in intent.allowed_ops:
                codes.append("unauthorized_semantics_change")
        elif field == "content_format":
            continue
        elif field == "guard_days":
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
        else:
            codes.append("unmentioned_field_changed")
    return codes


def audit_edit(
    before: EditLibrary,
    after: EditLibrary,
    intent: EditIntent,
    *,
    receipts: Sequence[RelationReceipt] | None = None,
    max_core_size: int | None = None,
    protocol: dict | None = None,
) -> dict:
    _validate_unique_ids(before, "before library")
    _validate_unique_ids(after, "after library")
    _validate_object_domain(before, "before library")
    _validate_object_domain(after, "after library")
    if before.full_mask != after.full_mask or before.frame_id != after.frame_id:
        raise HarnessError("frame migration is unsupported by this finite edit contract", 3)
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
    for oid in intent.target_ids:
        target = after_by_id.get(oid)
        if target is not None and not set(target.scope) <= set(intent.declared_scope):
            codes.append("scope_escalation")
    if intent.joint_claims and not _clauses_satisfiable(intent.joint_claims):
        codes.append("joint_inconsistent")
    before_cores = minimal_inconsistent_cores(before, max_core_size=max_core_size)
    after_cores = minimal_inconsistent_cores(after, max_core_size=max_core_size)
    new_cores, retained_cores, changed_cores = classify_cores(before_cores, after_cores)
    if new_cores:
        codes.append("new_semantic_conflict")
    if changed_cores:
        codes.append("changed_existing_conflict_review")
    diagnostics = None
    if receipts is not None:
        if protocol is None:
            raise HarnessError("relationship receipts need the protocol for label validation", 2)
        diagnostics = relation_receipt_diagnostics(receipts, after, protocol=protocol)
    ordered = list(dict.fromkeys(codes))
    return {
        "codes": ordered,
        "new_minimal_cores": new_cores,
        "retained_minimal_cores": retained_cores,
        "changed_existing_cores": changed_cores,
        "review_required": bool(changed_cores),
        "core_search_complete": True,
        "before_cohort_conjunctions": cohort_conjunctions(before),
        "after_cohort_conjunctions": cohort_conjunctions(after),
        "no_new_conflict_certificate": False,
        "mixed_relation_diagnostic": diagnostics,
    }
_FIXTURE_FULL_MASK = 0b1111


def _object(oid: str, cohort: str, mask: int, *, guard: int = _FIXTURE_FULL_MASK, version: int = 1, binding: str = "b1", conjunctive: bool = True, **overrides) -> EditObject:
    defaults = dict(
        content_format="csv",
        guard_days=30,
        readonly_until="2026-12-31",
        condition="always",
        ttl="-",
        scope=("project-1",),
        evidence=("ev-1",),
        lineage=("ln-1",),
        dependencies=("dep-1",),
        notes="",
    )
    defaults.update(overrides)
    return EditObject(
        object_id=oid,
        cohort=cohort,
        binding_id=binding,
        version=version,
        mask=mask,
        guard=guard,
        conjunctive=conjunctive,
        **defaults,
    )


def _base_objects() -> dict[str, EditObject]:
    return {
        "obj_alpha": _object("obj_alpha", "c1", 0b0111),
        "obj_beta": _object("obj_beta", "c1", 0b0011),
        "obj_gamma": _object("obj_gamma", "c2", 0b0001),
        "obj_delta": _object("obj_delta", "c2", 0b0010),
        "obj_beta2": _object("obj_beta2", "c3", 0b0011),
        "obj_example": _object("obj_example", "c1", 0b0000, conjunctive=False),
    }


def _library(objects: dict[str, EditObject], version: int = 7, full_mask: int = _FIXTURE_FULL_MASK) -> EditLibrary:
    return EditLibrary(version=version, full_mask=full_mask, objects=tuple(objects.values()))


def _replace(library: EditLibrary, updated: dict[str, EditObject]) -> EditLibrary:
    return replace(library, objects=tuple(updated.get(obj.object_id, obj) for obj in library.objects))


def _extend(library: EditLibrary, extra: Sequence[EditObject]) -> EditLibrary:
    return replace(library, objects=tuple(library.objects) + tuple(extra))


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


def _case_multi_class_consistent():
    before = _library(_base_objects())
    after = _extend(before, [
        _object("obj_h1", "cmix", 0b0111),
        _object("obj_h2", "cmix", 0b0011),
        _object("obj_h3", "cmix", 0b1001),
    ])
    return before, after, _intent(("obj_h1", "obj_h2", "obj_h3"), allowed_ops=frozenset({"create"}))


def _case_same_class_jointly_inconsistent():
    before = _library(_base_objects())
    after = _extend(before, [
        _object("obj_cs1", "cs", 0b0011),
        _object("obj_cs2", "cs", 0b0110),
        _object("obj_cs3", "cs", 0b0101),
    ])
    return before, after, _intent(("obj_cs1", "obj_cs2", "obj_cs3"), allowed_ops=frozenset({"create"}))


def _case_exclusive_guards_consistent():
    before = _library(_base_objects())
    after = _extend(before, [
        _object("obj_g1", "cg", 0b0010, guard=0b0001),
        _object("obj_g2", "cg", 0b0001, guard=0b0010),
    ])
    return before, after, _intent(("obj_g1", "obj_g2"), allowed_ops=frozenset({"create"}))


def _case_old_cohort_new_core():
    before = _extend(_library(_base_objects()), [
        _object("obj_s1", "c5", 0b0011),
        _object("obj_s2", "c5", 0b0110),
    ])
    after = _extend(before, [_object("obj_s3", "c5", 0b0101)])
    return before, after, _intent(("obj_s3",), allowed_ops=frozenset({"create"}))


def _case_old_conflict_unrelated_change():
    return _single_field_case("obj_alpha", "content_format", "tsv")


def _case_changed_existing_core_review():
    before = _extend(_library(_base_objects()), [
        _object("obj_t1", "c6", 0b0001),
        _object("obj_t2", "c6", 0b0010),
    ])
    after = _replace(before, {"obj_t2": replace(_objects_by_id(before)["obj_t2"], mask=0b0100, version=2)})
    return before, after, _intent(("obj_t2",), allowed_ops=frozenset({"semantics"}))


def _case_fragment_moved():
    return _single_field_case("obj_alpha", "notes", "relocated")


def _case_allowed_different_relations():
    before = _library(_base_objects())
    after = _extend(before, [
        _object("obj_u1", "c7", 0b0011),
        _object("obj_u2", "c7", 0b1010),
    ])
    return before, after, _intent(("obj_u1", "obj_u2"), allowed_ops=frozenset({"create"}))


def _case_mask_change_unauthorized():
    return _single_field_case("obj_alpha", "mask", 0b0101)


def _case_evidence_dropped():
    return _single_field_case("obj_alpha", "evidence", ())


def _case_lineage_dropped():
    return _single_field_case("obj_alpha", "lineage", ())


def _case_dependencies_dropped():
    return _single_field_case("obj_alpha", "dependencies", ())


def _case_unmentioned_object():
    return _single_field_case("obj_beta", "guard_days", 90, targets=("obj_alpha",))


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
    "multi_class_consistent": _case_multi_class_consistent,
    "same_class_jointly_inconsistent": _case_same_class_jointly_inconsistent,
    "exclusive_guards_consistent": _case_exclusive_guards_consistent,
    "old_cohort_new_core": _case_old_cohort_new_core,
    "old_conflict_unrelated_change": _case_old_conflict_unrelated_change,
    "changed_existing_core_review": _case_changed_existing_core_review,
    "fragment_moved": _case_fragment_moved,
    "allowed_different_relations": _case_allowed_different_relations,
    "mask_change_unauthorized": _case_mask_change_unauthorized,
    "evidence_ref_dropped": _case_evidence_dropped,
    "lineage_ref_dropped": _case_lineage_dropped,
    "dependency_ref_dropped": _case_dependencies_dropped,
    "unmentioned_object_change": _case_unmentioned_object,
}


def _check_case_asserts(name: str, before: EditLibrary, after: EditLibrary, report: dict) -> bool:
    before_alpha = _objects_by_id(before)["obj_alpha"]
    after_alpha = _objects_by_id(after)["obj_alpha"]
    if name == "readonly_until_preserved":
        return after_alpha.readonly_until == before_alpha.readonly_until == "2026-12-31"
    if name == "guard_days_preserved":
        return after_alpha.guard_days == before_alpha.guard_days == 30
    if name == "retained_core_reported":
        return any(core["cohort"] == "c2" for core in report["retained_minimal_cores"])
    if name == "no_certificate":
        return report["no_new_conflict_certificate"] is False
    if name == "review_required_flag":
        return report["review_required"] is True and bool(report["changed_existing_cores"])
    raise HarnessError(f"unknown audit case assertion {name}", 2)


def run_edit_audit(protocol: dict) -> dict:
    catalog = protocol["edit_audit"]
    results = []
    for case in catalog["cases"]:
        builder = MUTATIONS.get(case["mutation"])
        if builder is None:
            raise HarnessError(f"protocol references unknown mutation {case['mutation']}", 2)
        before, after, intent = builder()
        report = audit_edit(before, after, intent, protocol=protocol)
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
            "new_cores": len(report["new_minimal_cores"]),
            "retained_cores": len(report["retained_minimal_cores"]),
            "changed_cores": len(report["changed_existing_cores"]),
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
