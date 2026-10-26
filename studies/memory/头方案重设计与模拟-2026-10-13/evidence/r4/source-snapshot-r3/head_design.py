"""头方案 v5（r3）研究-only 的 CLI 与 artifact 阶段入口（研究-only）。

- 三模块：`head_design_semantics.py`（基座 + 有限语义 + 编辑 oracle）、`head_design_learning.py`
  （分区/曝光 + 模型学习数值）、本文件（CLI/artifact 阶段）；本文件 import 再导出既有研究 API，
  不复制实现。
- source_bundle：所有源码模块统一哈希（`source_bundle_sha256`）写入 r3 manifest 并作为 gate 依据；
  不再只绑 CLI 文件 SHA。r1/r2 旧制品按 legacy schema 只核其原 hash，不因 r3 schema 判坏。
- study_role：formal（N10、封存池 `final`，final 评分需正式 1500 预算）与 smoke（N8 pilot、封存池
  `finalpilot`，仅机制用；小预算可评 pilot 但输出标 pilot，不构成正式终测）。
- 门禁：新 `--out` 必须不存在；prepare 校验 mandatory artifacts/池-制品 SHA 一致/曝光与确定性期望相等/
  frozen canonical=protocol/source bundle 一致；finalize 需真实 freeze（从验证过的 train-dev 证据计算）、
  attempt 登记（失败也记，同 binding 需 --reproduce）、逐题 CSV/JSONL 为真源且 summary 从逐题重算。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from research.memory_relations.head_design_learning import (              
    ALLOWED_OBJECTIVE_SETS,
    TRAINABLE_ARMS,
    TinyHeadModel,
    _assert_finite,
    _check_params_finite,
    _class_stats,
    _history_at,
    _macro_f1_and_recall,
    _np_log_group_sums,
    _np_logsumexp,
    _np_project,
    _question_rows,
    _two_way_entries,
    apply_temperature_to_logp3,
    apply_temperature_to_logp5,
    arm_loss,
    build_model_skeleton,
    build_partition,
    coarse_ce,
    conditional_fine_nll,
    dev_metrics,
    encode_pairs,
    entity_digest,
    exposure_jsonl,
    extension_gains,
    fit_temperature,
    five_ce,
    forward_log,
    grouped_coarse_nll,
    head_init,
    index_maps,
    initial_priors,
    label_indices,
    leaf_counts_from_pairs,
    load_state_strict,
    log_group_sums,
    log_probs,
    objective_loss,
    ordered_exposures,
    parent_counts_from_pairs,
    partition_report,
    policy_coverage_description,
    pool_statistics,
    pregenerate_common,
    project_5_to_3,
    rows_csv,
    select_coarse_checkpoint,
    select_fine_checkpoint,
    summarize_rows,
    train_arm,
)
from research.memory_relations.head_design_semantics import (              
    AUDIT_CODES,
    EditIntent,
    EditLibrary,
    EditObject,
    Geometry,
    HarnessError,
    MUTATIONS,
    RelationReceipt,
    Verdict,
    _array_digest,
    _base_objects,
    _canonical,
    _check_case_asserts,
    _extend,
    _intent,
    _library,
    _now_iso,
    _object,
    _objects_by_id,
    _read_json,
    _replace,
    _repo_root,
    _write_bytes,
    _write_json,
    assertion_closure,
    audit_edit,
    bidirectional_summary,
    classify,
    classify_cores,
    combination_leaf,
    complement_families,
    conditional_full_relation,
    exhaustive_census,
    interval_content_relation,
    joint_marginals_feasible,
    leaf_of,
    load_protocol,
    minimal_inconsistent_cores,
    parent_of_masks,
    protocol_default_path,
    relation_receipt_diagnostics,
    require_mask,
    run_edit_audit,
    runs_root,
    runtime_info,
    sha256_bytes,
    sha256_file,
    source_bundle_info,
    source_bundle_sha256,
    strata_families,
    study_geometry,
    study_roles,
    validate_protocol,
    verify_relation_receipt,
)

SCHEMA_MARKER = "hd5-r3"
ATTEMPTS_FILENAME = "final-access-attempts.jsonl"
MANDATORY_PREPARE_ARTIFACTS = (
    "protocol-frozen.json",
    "partition.json",
    "exposures/train.jsonl",
    "exposures/dev.jsonl",
    "exposures/calibration_prob.jsonl",
    "exposures/calibration_policy.jsonl",
)
R3_MANIFEST_KEYS = ("stage", "schema", "created_at", "protocol_sha256", "source_bundle_sha256", "artifacts", "budget", "study_role")
LEGACY_MANIFEST_KEYS = ("stage", "protocol_sha256", "artifacts")
LEGACY_ARTIFACT_KEYS = ("stage", "protocol_sha256", "source_sha256", "artifacts")


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


def _artifact_hashes(directory: Path, relpaths) -> dict:
    return {rel: sha256_file(directory / rel) for rel in relpaths}


def _read_exposure(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append((int(row["p"]), int(row["h"])))
    return pairs


def expected_exposure(protocol: dict, study: str, pool: str) -> list[tuple[int, int]]:
    built = build_partition(protocol, study)
    return ordered_exposures(built["entity_pools"][pool])


def _study_role_of(context: dict) -> str:
    return context["study_role"]


def _data_identity(protocol_sha: str, study: str, report: dict) -> str:
    payload = {
        "protocol_sha256": protocol_sha,
        "study_role": study,
        "geometry": report["geometry"],
        "seed": report["seed"],
        "stratum_families": report["stratum_families"],
        "pools": {pool: {"entity_count": row["entity_count"], "entity_sha256": row["entity_sha256"], "pairs": row["pairs"], "exposure_sha256": row["exposure_sha256"]} for pool, row in report["pools"].items()},
        "denominators": report["denominators"],
    }
    return sha256_bytes(_canonical(payload).encode("utf-8"))


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
    manifest_bundle = manifest.get("source_bundle_sha256")
    if manifest_bundle != source_bundle_sha256():
        raise HarnessError("source bundle changed since prepare; re-prepare required", 4)
    study_role = manifest.get("study_role")
    if study_role not in study_roles(protocol):
        raise HarnessError(f"prepare study_role invalid: {study_role}", 4)
    geometry = study_geometry(protocol, study_role)
    return {"protocol": protocol, "manifest": manifest, "frozen": frozen, "prepare_dir": prepare_dir, "study_role": study_role, "geometry": geometry}


def _read_verified_exposure(prepare_dir: Path, manifest: dict, rel: str, expected_pairs) -> list[tuple[int, int]]:
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
    study = context["study_role"]
    protocol = context["protocol"]
    train_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/train.jsonl", expected_exposure(protocol, study, "train"))
    dev_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/dev.jsonl", expected_exposure(protocol, study, "dev"))
    return context, train_pairs, dev_pairs


def stage_formal(out_path, protocol_path=None, *, study: str = "formal", allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    protocol, protocol_sha = load_protocol(protocol_path)
    geometry = study_geometry(protocol, study)
    census = exhaustive_census(protocol, study)
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
    full_mask = geometry.full_mask
    unresolved_cases = {
        "unknown_input": (classify(None, 5, full_mask=full_mask), "input_unknown"),
        "empty_input": (classify(0, 5, full_mask=full_mask), "empty_satisfier"),
        "inconsistent_background": (classify(3, 5, full_mask=full_mask, background_consistent=False), "background_mismatch"),
        "condition_overlap_unknown": (conditional_full_relation(3, 5, 1, 3, full_mask=full_mask, overlap_known=False), "condition_overlap_unknown"),
        "empty_condition_overlap": (conditional_full_relation(3, 5, 1, 2, full_mask=full_mask), "empty_condition_overlap"),
    }
    recorded_unresolved = {}
    for name, (verdict, reason) in unresolved_cases.items():
        if verdict.status != "unresolved" or verdict.leaf is not None or verdict.reason != reason:
            raise HarnessError(f"unresolved case {name} did not stay single-column: {verdict}", 1)
        recorded_unresolved[name] = {"status": verdict.status, "leaf": verdict.leaf, "reason": verdict.reason}
    conditional_full = conditional_full_relation(3, 1, 1, 3, full_mask=full_mask)
    conditional_interval = interval_content_relation(3, 1, 1, 3, full_mask=full_mask)
    if conditional_full.leaf == conditional_interval.leaf or None in (conditional_full.leaf, conditional_interval.leaf):
        raise HarnessError("conditional full/interval non-equivalence fixture collapsed", 1)
    if {"supersede", "temporary"} & set(protocol["semantics"]["leaves"]):
        raise HarnessError("lifecycle categories must not be mixed into the leaf layer", 1)
    joint_demo = joint_marginals_feasible([0.45, 0.15, 0.40], [0.33, 0.27, 0.40], protocol)
    if not joint_demo:
        raise HarnessError("joint feasibility fixture for the legal EC-marginal example collapsed", 1)
    audit = run_edit_audit(protocol)
    if audit["failed"] or audit["codes_uncovered"]:
        raise HarnessError(f"edit audit failed: {audit['failed']} uncovered={audit['codes_uncovered']}", 1)
    bundle = source_bundle_info()
    formal = {
        "stage": "formal",
        "schema": SCHEMA_MARKER,
        "study_role": study,
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "created_at": _now_iso(),
        "duration_seconds": round(time.time() - started, 3),
        "runtime": runtime_info(),
        "geometry": {"world_bits": geometry.world_bits, "world_count": geometry.world_count, "full_mask": geometry.full_mask, "input_bits_per_side": geometry.input_bits_per_side},
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
        "schema": SCHEMA_MARKER,
        "study_role": study,
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
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
        "study_role": study,
        "out": str(out),
        "formal_sha256": formal_sha,
        "pair_total": census["pair_total"],
        "leaf_counts": census["leaf_counts"],
        "parent_counts": census["parent_counts"],
        "edit_audit_passed": audit["passed"],
        "edit_audit_total": audit["total"],
        "duration_seconds": formal["duration_seconds"],
    }


def stage_prepare(out_path, protocol_path=None, *, study_role: str = "formal", allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    protocol, protocol_sha = load_protocol(protocol_path)
    if study_role not in study_roles(protocol):
        raise HarnessError(f"unknown study_role {study_role}", 2)
    report = partition_report(protocol, study_role)
    created = _now_iso()
    bundle = source_bundle_info()
    frozen = {
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "study_role": study_role,
        "frozen_at": created,
        "content": protocol,
        "sealing": "the sealed pool content is not exported; the generator stays public and sealing is a process commitment, not information-theoretic secrecy",
    }
    _create_out(out)
    _write_json(out / "protocol-frozen.json", frozen)
    partition_payload = {
        "study_role": study_role,
        "geometry": report["geometry"],
        "seed": report["seed"],
        "stratum_families": report["stratum_families"],
        "pools": report["pools"],
        "sealed_pool": report["sealed_pool"],
        "novelty": report["novelty"],
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
        entry = {key: row[key] for key in ("entity_count", "entity_sha256", "pairs", "sealed", "exported", "pilot", "stats")}
        entry["exposure_sha256"] = artifacts[f"exposures/{pool}.jsonl"] if row["exported"] else row["exposure_sha256"]
        pools_summary[pool] = entry
    duration = round(time.time() - started, 3)
    data_identity = _data_identity(protocol_sha, study_role, report)
    manifest = {
        "stage": "prepare",
        "schema": SCHEMA_MARKER,
        "study_role": study_role,
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": protocol_sha,
        "protocol_frozen_sha256": artifacts["protocol-frozen.json"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "created_at": created,
        "duration_seconds": duration,
        "runtime": runtime_info(),
        "geometry": report["geometry"],
        "seed": report["seed"],
        "stratum_families": report["stratum_families"],
        "sealed_pool": report["sealed_pool"],
        "novelty": report["novelty"],
        "data_identity": data_identity,
        "pools": pools_summary,
        "denominators": report["denominators"],
        "distribution_target": report["distribution_target"],
        "updates": None,
        "steps": None,
        "inference_calls": {"dev_forward_passes": 0},
        "budget": {"updates": None, "steps": None, "eval_every": protocol["training"]["eval_every"], "duration_seconds": duration},
        "artifacts": artifacts,
        "scored": False,
    }
    manifest_sha = _write_json(out / "manifest.json", manifest)
    return {
        "stage": "prepare",
        "study_role": study_role,
        "out": str(out),
        "manifest_sha256": manifest_sha,
        "data_identity": data_identity,
        "sealed_pool": report["sealed_pool"],
        "novelty": report["novelty"],
        "pools": {pool: {"entity_count": row["entity_count"], "pairs": row["pairs"], "sealed": row["sealed"]} for pool, row in pools_summary.items()},
        "denominators": report["denominators"],
        "duration_seconds": duration,
    }


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
    if manifest.get("study_role") != prepare_context["study_role"]:
        raise HarnessError("train-dev study_role does not match prepare", 4)
    if manifest.get("data_identity") != prepare_context["manifest"].get("data_identity"):
        raise HarnessError("train-dev data identity does not match prepare", 4)
    if manifest.get("source_bundle_sha256") != source_bundle_sha256():
        raise HarnessError("source bundle changed since train-dev; re-run train-dev", 4)
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


def verify_model_file(arm: str, seed: int, rel_path: str, expected_sha: str, train_dev_dir: Path, train_pairs, protocol: dict, geometry: Geometry) -> TinyHeadModel:
    path = train_dev_dir / rel_path
    if not path.exists():
        raise HarnessError(f"model file missing: {rel_path}", 4)
    if sha256_file(path) != expected_sha:
        raise HarnessError(f"model weights hash mismatch: {rel_path}", 4)
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise HarnessError(f"model file is not a state dict: {rel_path}", 4)
    model = build_model_skeleton(arm, seed, train_pairs, protocol, geometry)
    load_state_strict(model, state, rel_path)
    return model


def final_access_identity(context: dict) -> str:
    manifest = context["manifest"]
    return sha256_bytes(_canonical({
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_frozen_sha256": manifest["protocol_frozen_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "study_role": manifest["study_role"],
        "data_identity": manifest["data_identity"],
        "artifacts": manifest["artifacts"],
    }).encode("utf-8"))


def candidate_views_from_runs(runs: dict, protocol: dict) -> list[dict]:
    views: list[dict] = []
    for key in sorted(runs):
        run = runs[key]
        arm = run["arm"]
        seed = int(run["seed"])
        coarse_row = _history_at(run["history"], run["coarse"]["step"])
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
            fine_row = _history_at(run["history"], run["fine"]["step"])
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


def _history_at(history, step: int) -> dict:
    for row in history:
        if row["step"] == step:
            return row
    raise HarnessError(f"history lacks the pre-registered step {step}", 2)


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
        c_coarse_at = _history_at(c_run["history"], c_run["fine"]["step"])["dev_coarse_nll"]
        h_coarse_at = _history_at(h_run["history"], h_run["fine"]["step"])["dev_coarse_nll"]
        paired.append({
            "seed": seed,
            "c_fine_nll": c_fine,
            "h_fine_nll": h_fine,
            "fine_improvement": c_fine - h_fine,
            "c_coarse_nll_at_fine_step": c_coarse_at,
            "h_coarse_nll_at_fine_step": h_coarse_at,
            "coarse_delta": h_coarse_at - c_coarse_at,
        })
    mean_c_fine = float(sum(row["c_fine_nll"] for row in paired) / len(paired))
    mean_improvement = float(sum(row["fine_improvement"] for row in paired) / len(paired))
    threshold = max(
        float(rule["min_fine_improvement"]["absolute_nats"]),
        float(rule["min_fine_improvement"]["relative_to_c_fine_nll"]) * mean_c_fine,
    )
    all_positive = all(row["fine_improvement"] > 0 for row in paired)
    mean_coarse_delta = float(sum(row["coarse_delta"] for row in paired) / len(paired))
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


def stage_train_dev(
    out_path,
    prepare_dir,
    protocol_path=None,
    *,
    allow_external: bool = False,
    updates: int | None = None,
    seeds=None,
) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    context, train_pairs, dev_pairs = preflight_train_dev(prepare_dir, protocol_path)
    protocol = context["protocol"]
    geometry = context["geometry"]
    manifest = context["manifest"]
    study_role = context["study_role"]
    spec = protocol["training"]
    target_updates = int(updates if updates is not None else spec["updates"])
    if target_updates <= 0 or target_updates % spec["eval_every"] != 0:
        raise HarnessError("updates override must be a positive multiple of eval_every", 2)
    seed_list = [int(seed) for seed in (seeds if seeds is not None else spec["seeds"])]
    if not seed_list:
        raise HarnessError("at least one seed is required", 2)
    overrides = {"updates": updates is not None, "seeds": seeds is not None}
    formal_budget = (not any(overrides.values())) and target_updates == spec["updates"]
    bundle = source_bundle_info()
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
            run = train_arm(arm, seed, train_pairs, dev_pairs, protocol, geometry, updates=target_updates)
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
                    extended = train_arm(arm, seed, train_pairs, dev_pairs, protocol, geometry, updates=spec["extension"]["target_updates"], resume=resume_payload)
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
        "schema": SCHEMA_MARKER,
        "study_role": study_role,
        "data_identity": manifest["data_identity"],
        "created_at": _now_iso(),
        "protocol_sha256": manifest["protocol_sha256"],
        "source_bundle_sha256": bundle["sha256"],
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
        "schema": SCHEMA_MARKER,
        "study_role": study_role,
        "data_identity": manifest["data_identity"],
        "protocol_sha256": manifest["protocol_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
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
        "study_role": study_role,
        "out": str(out),
        "manifest_sha256": manifest_sha,
        "budget": manifest_out["budget"],
        "extension": extension,
        "runs": {key: {"coarse": run["coarse"], "fine": run["fine"], "model_sha256": run["models"]["coarse"]["sha256"]} for key, run in runs.items()},
        "duration_seconds": duration,
    }
def stage_freeze(out_path, prepare_dir, train_dev_dir, *, protocol_path=None, allow_external: bool = False) -> dict:
    out = _check_fresh_out(out_path, allow_external=allow_external)
    started = time.time()
    context = load_prepare_context(prepare_dir, protocol_path)
    protocol = context["protocol"]
    geometry = context["geometry"]
    study_role = context["study_role"]
    td = load_train_dev_context(context, train_dev_dir)
    runs = td["runs"]["runs"]
    if not runs:
        raise HarnessError("train-dev run set is empty; refusing to freeze", 3)
    verify_selections(runs, protocol)
    train_pairs = _read_verified_exposure(context["prepare_dir"], context["manifest"], "exposures/train.jsonl", expected_exposure(protocol, study_role, "train"))
    loaded_models = 0
    for key in sorted(runs):
        run = runs[key]
        for view in ("coarse", "fine"):
            entry = run["models"].get(view)
            if entry:
                verify_model_file(run["arm"], int(run["seed"]), entry["path"], entry["sha256"], td["dir"], train_pairs, protocol, geometry)
                loaded_models += 1
    views = candidate_views_from_runs(runs, protocol)
    architecture = evaluate_architecture_rule(runs, protocol)
    bundle = source_bundle_info()
    created = _now_iso()
    payload = {
        "stage": "freeze",
        "schema": SCHEMA_MARKER,
        "created_at": created,
        "study_role": study_role,
        "data_identity": context["manifest"]["data_identity"],
        "sealed_pool": context["manifest"]["sealed_pool"],
        "pilot": bool(protocol["semantics"]["studies"][study_role]["pilot"]),
        "protocol": protocol["protocol"],
        "revision": protocol["revision"],
        "protocol_sha256": context["manifest"]["protocol_sha256"],
        "protocol_frozen_sha256": context["manifest"]["protocol_frozen_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
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
        "novelty": context["manifest"]["novelty"],
        "runtime": runtime_info(),
        "boundary": [
            "freeze is computed from verified train-dev history and weights, not from a self-reported string",
            "tiny or overridden budgets are mechanism evidence only, not formal results",
            "the recommendation is a fixed-budget result inside one study, not a global optimum",
            "N8 pilot and N10 formal are not a causal difference",
        ],
    }
    _create_out(out)
    freeze_sha = _write_json(out / "freeze.json", payload)
    duration = round(time.time() - started, 3)
    manifest = {
        "stage": "freeze",
        "schema": SCHEMA_MARKER,
        "study_role": study_role,
        "data_identity": context["manifest"]["data_identity"],
        "created_at": created,
        "protocol_sha256": payload["protocol_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
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
        "study_role": study_role,
        "out": str(out),
        "freeze_sha256": freeze_sha,
        "recommended": architecture["recommended"],
        "candidate_views": len(views),
        "formal_budget": payload["train_dev"]["formal_budget"],
        "duration_seconds": duration,
    }


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
    geometry = context["geometry"]
    study_role = context["study_role"]
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
    if freeze.get("source_bundle_sha256") != source_bundle_sha256():
        raise HarnessError("source bundle changed since freeze; re-freeze required", 4)
    if freeze.get("study_role") != study_role or freeze.get("data_identity") != manifest.get("data_identity"):
        raise HarnessError("refusing finalize: freeze does not bind this study identity", 4)
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
    train_pairs = _read_verified_exposure(context["prepare_dir"], manifest, "exposures/train.jsonl", expected_exposure(protocol, study_role, "train"))
    models_by_view: dict[str, TinyHeadModel] = {}
    for view in freeze["candidate_views"]:
        models_by_view[view["candidate"]] = verify_model_file(view["arm"], int(view["seed"]), view["model_path"], view["model_sha256"], td["dir"], train_pairs, protocol, geometry)
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
            "schema": SCHEMA_MARKER,
            "mode": "preflight",
            "study_role": study_role,
            "pilot": bool(protocol["semantics"]["studies"][study_role]["pilot"]),
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
        bundle = source_bundle_info()
        manifest_out = {
            "stage": "finalize",
            "schema": SCHEMA_MARKER,
            "mode": "preflight",
            "study_role": study_role,
            "data_identity": manifest["data_identity"],
            "protocol_sha256": manifest["protocol_sha256"],
            "source_bundle_sha256": bundle["sha256"],
            "source_files": bundle["files"],
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
        return {"stage": "finalize-precheck", "study_role": study_role, "out": str(out), "checks": checks, "precheck_sha256": precheck_sha, "duration_seconds": duration}
    formal_budget = bool(td["manifest"]["budget"]["formal"])
    if study_role == "formal" and not formal_budget:
        raise HarnessError("formal final scoring requires the formal 1500-update budget; tiny or overridden runs are pilot-only", 3)
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
        "study_role": study_role,
        "pilot": bool(protocol["semantics"]["studies"][study_role]["pilot"]),
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
    geometry = context["geometry"]
    manifest = context["manifest"]
    prepare_dir = context["prepare_dir"]
    study_role = context["study_role"]
    sealed_pool = manifest["sealed_pool"]
    maps = index_maps(protocol)
    prob_pairs = _read_verified_exposure(prepare_dir, manifest, "exposures/calibration_prob.jsonl", expected_exposure(protocol, study_role, "calibration_prob"))
    policy_pairs = _read_verified_exposure(prepare_dir, manifest, "exposures/calibration_policy.jsonl", expected_exposure(protocol, study_role, "calibration_policy"))
    sealed_pairs = expected_exposure(protocol, study_role, sealed_pool)
    if sha256_bytes(exposure_jsonl(sealed_pairs)) != manifest["pools"][sealed_pool]["exposure_sha256"]:
        raise HarnessError("regenerated sealed-pool fingerprint does not match the frozen prepare manifest", 4)
    _, prob_parent = label_indices(prob_pairs, protocol, maps, geometry)
    _, policy_parent = label_indices(policy_pairs, protocol, maps, geometry)
    inference_calls = 0
    inference_rows = 0

    def log_probs_counted(model: TinyHeadModel, pairs):
        nonlocal inference_calls, inference_rows
        inference_calls += 1
        inference_rows += len(pairs)
        return log_probs(model, pairs, geometry)

    cache: dict = {}
    for view in freeze["candidate_views"]:
        key = (view["arm"], int(view["seed"]), view["step"], view["model_sha256"])
        if key in cache:
            continue
        model = models_by_view[view["candidate"]]
        prob_logp3, _ = log_probs_counted(model, prob_pairs)
        policy_logp3, _ = log_probs_counted(model, policy_pairs)
        sealed_logp3, sealed_logp5 = log_probs_counted(model, sealed_pairs)
        temperature = fit_temperature(prob_logp3, prob_parent, protocol)
        policy_calibrated = apply_temperature_to_logp3(policy_logp3, temperature["temperature"])
        policy_description = policy_coverage_description(policy_calibrated, policy_parent, protocol)
        if sealed_logp5 is None:
            cal_logp3 = apply_temperature_to_logp3(sealed_logp3, temperature["temperature"])
            cal_logp5 = None
        else:
            cal_logp5 = apply_temperature_to_logp5(sealed_logp5, temperature["temperature"], maps)
            cal_logp3 = _np_log_group_sums(cal_logp5, maps)
        two_way = None
        if view["arm"] == "A":
            swapped = [(h_mask, p_mask) for p_mask, h_mask in sealed_pairs]
            swap_logp3, _ = log_probs_counted(model, swapped)
            two_way = _two_way_entries(sealed_pairs, sealed_logp3, swap_logp3, protocol)
        cache[key] = {
            "final_logp3": sealed_logp3,
            "final_logp5": sealed_logp5,
            "cal_logp3": cal_logp3,
            "cal_logp5": cal_logp5,
            "temperature": temperature,
            "policy": policy_description,
            "two_way": two_way,
        }
    from research.memory_relations.head_design_learning import _question_rows

    rows: list[dict] = []
    view_summaries: dict = {}
    for view in freeze["candidate_views"]:
        key = (view["arm"], int(view["seed"]), view["step"], view["model_sha256"])
        entry = cache[key]
        rows_view = _question_rows(
            view,
            sealed_pairs,
            entry["final_logp3"],
            entry["final_logp5"],
            entry["cal_logp3"],
            entry["cal_logp5"],
            protocol,
            maps,
            geometry,
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
    csv_sha = _write_bytes(out / "questions.csv", rows_csv(rows).encode("utf-8"))
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
    bundle = source_bundle_info()
    pilot = bool(protocol["semantics"]["studies"][study_role]["pilot"])
    payload = {
        "stage": "finalize",
        "schema": SCHEMA_MARKER,
        "mode": mode,
        "study_role": study_role,
        "pilot": pilot,
        "created_at": _now_iso(),
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_frozen_sha256": manifest["protocol_frozen_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "data_identity": manifest["data_identity"],
        "freeze_sha256": freeze_sha,
        "final_access_id": access_id,
        "recommended": freeze["architecture_rule"]["recommended"],
        "formal_budget": bool(td["manifest"]["budget"]["formal"]),
        "budget_overrides": td["manifest"]["budget"]["overrides"],
        "sealed_pool": sealed_pool,
        "views": view_summaries,
        "rows": len(rows),
        "questions_file": "questions.jsonl",
        "questions_sha256": rows_sha,
        "csv_sha256": csv_sha,
        "inference_calls": {"forward_batches": inference_calls, "forward_rows": inference_rows},
        "runtime": runtime_info(),
        "boundary": [
            "pilot scores are mechanism/development numbers; the retired N8 sealed pool is not a formal final test" if pilot else "formal final is a programmatic holdout of the public oracle structure, not a human/NLP final test",
            "toy metrics are research-only simulation numbers, not natural-language results and not production gains",
            "the reported candidate set is the complete frozen set; C/H views sharing one artifact are deduplicated for compute but reported separately",
            "policy coverage is a descriptive 90% empirical rule, not a risk certificate",
            "tiny or overridden budgets are mechanism evidence only",
        ],
    }
    summary_sha = _write_json(out / "summary.json", payload)
    manifest_out = {
        "stage": "finalize",
        "schema": SCHEMA_MARKER,
        "mode": mode,
        "study_role": study_role,
        "pilot": pilot,
        "data_identity": manifest["data_identity"],
        "protocol_sha256": manifest["protocol_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "created_at": payload["created_at"],
        "duration_seconds": duration,
        "runtime": payload["runtime"],
        "updates": td["manifest"]["budget"]["updates"],
        "steps": None,
        "inference_calls": payload["inference_calls"],
        "denominators": {"rows": len(rows), "candidate_views": len(freeze["candidate_views"])},
        "budget": {"updates": td["manifest"]["budget"]["updates"], "steps": None, "formal": bool(td["manifest"]["budget"]["formal"]), "duration_seconds": duration},
        "artifacts": {"questions.jsonl": rows_sha, "questions.csv": csv_sha, "summary.json": summary_sha},
    }
    _write_json(out / "manifest.json", manifest_out)
    return {
        "stage": "finalize",
        "mode": mode,
        "study_role": study_role,
        "pilot": pilot,
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
        if manifest.get("schema") == SCHEMA_MARKER:
            era = SCHEMA_MARKER
            required = R3_MANIFEST_KEYS
        else:
            era = "legacy"
            required = LEGACY_MANIFEST_KEYS if all(key in manifest for key in LEGACY_MANIFEST_KEYS) else LEGACY_ARTIFACT_KEYS
        missing_keys = [key for key in required if key not in manifest]
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
            checks[rel] = actual == expected
        if checks and not all(checks.values()):
            problems.append("hash")
        verified = not problems
        if problems:
            failures.append(stage)
        stages[stage] = {
            "stage": manifest.get("stage"),
            "era": era,
            "mode": manifest.get("mode"),
            "study_role": manifest.get("study_role"),
            "pilot": manifest.get("pilot"),
            "problems": problems,
            "schema_ok": not missing_keys,
            "hashes_ok": bool(checks) and all(checks.values()),
            "created_at": manifest.get("created_at"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "source_bundle_sha256": manifest.get("source_bundle_sha256") or manifest.get("source_sha256"),
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
    formal = sub.add_parser("formal", help="exhaustive census + edit mutation audit for a study")
    formal.add_argument("--out", required=True)
    formal.add_argument("--study", default="formal", choices=("formal", "smoke"))
    formal.add_argument("--protocol")
    formal.add_argument("--allow-external-out", action="store_true")
    prepare = sub.add_parser("prepare", help="freeze protocol, study partition and ordered exposure fingerprints")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--study-role", default="formal", choices=("formal", "smoke"))
    prepare.add_argument("--protocol")
    prepare.add_argument("--allow-external-out", action="store_true")
    train_dev = sub.add_parser("train-dev", help="train candidates on train/dev only (no calibration/sealed parsing)")
    train_dev.add_argument("--out", required=True)
    train_dev.add_argument("--prepare", required=True)
    train_dev.add_argument("--protocol")
    train_dev.add_argument("--updates", type=int, default=None, help="tiny-budget override for mechanism runs; recorded as an override")
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
    status = sub.add_parser("status", help="read-only inspection of an existing run (legacy schema aware; non-zero exit on verification failure)")
    status.add_argument("--run", required=True)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.stage == "formal":
            result = stage_formal(args.out, args.protocol, study=args.study, allow_external=args.allow_external_out)
        elif args.stage == "prepare":
            result = stage_prepare(args.out, args.protocol, study_role=args.study_role, allow_external=args.allow_external_out)
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
