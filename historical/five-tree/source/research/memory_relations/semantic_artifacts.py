"""训练审计与推理导出的薄策略；模型保存/加载只委托原NR参数化引擎。

导出复用调用者按SHA审定的历史brief/compact证明，不重跑原完整训练guard，不冒充initializer。
证书是本机受信操作声明而非ACL；原审计工件不修改，新schema无continuation、不可续训。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch

from research.memory_relations import semantic_model as SM

SCHEMA = "semantic-text-control-inference-export-v1"
CERTIFICATE = "semantic-inference-certificate-v1"
HEADER_LIMIT = 65536
META_LIMIT = 1024 * 1024
RECOVERY_KIND = "post-save-selected-snapshot-v1"
RECOVERY_SCHEMA = "semantic-postsave-recovery-v1"
TENSOR_MANIFEST = "named-tensor-data-sha256-v1"


class ArtifactError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise ArtifactError(code)


def _small_json(path, limit=META_LIMIT):
    path = Path(path)
    _require(path.stat().st_size <= limit, "artifact_json_too_large")
    return SM.D.read_json(path)


def project_header(path, *, stop="train", limit=HEADER_LIMIT):
    """标准decoder逐顶层字段；有界、无缓冲逐字节，在stop冒号后且值之前停止。"""
    decoder = json.JSONDecoder(object_pairs_hook=_unique_pairs,
                               parse_constant=lambda value: _require(False, "invalid_header_number"))
    fields, count = {}, 0
    with Path(path).open("rb", buffering=0) as handle:
        def take():
            nonlocal count
            _require(count < limit, "header_limit_exceeded")
            byte = handle.read(1)
            count += len(byte)
            _require(bool(byte), "invalid_header")
            return byte
        def nonspace():
            byte = take()
            while byte in b" \r\n\t":
                byte = take()
            return byte
        def value(delimiters):
            raw = bytearray(nonspace())
            while True:
                try:
                    text = raw.decode("utf-8")
                    parsed, end = decoder.raw_decode(text)
                    tail = text[end:].strip()
                    if len(tail) == 1 and tail.encode() in delimiters:
                        return parsed, tail
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                raw.extend(take())
        _require(nonspace() == b"{", "invalid_header")
        while True:
            key, _ = value(b":")
            _require(type(key) is str and key not in fields, "invalid_header_key")
            if key == stop:
                return fields
            item, separator = value(b",}")
            fields[key] = item
            _require(separator == ",", "header_stop_missing")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _tensor_chunks(tensor):
    _require(torch.is_tensor(tensor), "invalid_selected_state")
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    for start in range(0, data.numel(), 1024 * 1024):
        yield memoryview(data[start:start + 1024 * 1024].cpu().numpy())


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _require(torch.is_tensor(tensor), "invalid_selected_state")
        digest.update(SM._digest([name, list(tensor.shape), str(tensor.dtype)]).encode("ascii"))
        for chunk in _tensor_chunks(tensor):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_manifest(state):
    """只有tensor原字节的SHA；清单摘要不是state_sha256，也不采用外部aggregate算法。"""
    result = []
    for name, tensor in sorted(state.items()):
        digest = hashlib.sha256()
        for chunk in _tensor_chunks(tensor):
            digest.update(chunk)
        result.append({"key": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "bytes": tensor.numel() * tensor.element_size(), "data_sha256": digest.hexdigest()})
    return result


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _recovery_reference(reference, base):
    try:
        _require(type(reference) is dict and type(reference.get("path")) is str
                 and _is_sha256(reference.get("sha256")), "recovery_reference_invalid")
        path = Path(reference["path"])
        if not path.is_absolute():
            path = base / path
        _require(path.stat().st_size <= META_LIMIT, "recovery_reference_too_large")
        _require(SM.FT.file_sha256(path) == reference["sha256"], "recovery_reference_sha_mismatch")
        return path.absolute()
    except OSError as error:
        raise ArtifactError("recovery_reference_missing") from error


def _recovery_proof(reference, base, brief, run, head):
    _require(type(reference) is dict and reference.get("kind") == RECOVERY_KIND, "recovery_kind_invalid")
    path = _recovery_reference(reference, base)
    try:
        record = _small_json(path)
    except ValueError as error:
        raise ArtifactError("recovery_record_invalid") from error
    _require(type(record) is dict and record.get("schema") == RECOVERY_SCHEMA
             and record.get("kind") == RECOVERY_KIND, "recovery_record_invalid")
    _require(record.get("run_id") == run["run_id"] and record.get("arm") == head["arm"]
             and record.get("source_sha256") == SM.source_binding(), "recovery_origin_binding_mismatch")
    synthetic = brief.get("synthetic", False)
    _require(type(synthetic) is bool and type(record.get("synthetic")) is bool
             and record["synthetic"] is synthetic, "recovery_synthetic_mismatch")
    core = run.get("core_files")
    _require(type(core) is dict and record.get("core_files") == core
             and all(type(core.get(name)) is dict and _is_sha256(core[name].get("sha256"))
                     for name in (SM.META_FILE, SM.WEIGHTS_FILE)), "recovery_core_mismatch")
    origin, counters, validation = (record.get(name) for name in ("origin_process", "counters", "validation"))
    _require(all(type(value) is dict for value in (origin, counters, validation)), "recovery_record_invalid")
    _require(type(run.get("exit_code")) is int and run["exit_code"] != 0 and run.get("validated") is False
             and origin.get("exit_code") == run["exit_code"] and type(origin.get("exit_code")) is int
             and "reload_max_abs_diff" in run and run["reload_max_abs_diff"] is None
             and "reload_max_abs_diff" in origin and origin["reload_max_abs_diff"] is None,
             "recovery_original_failure_missing")
    _require(type(origin.get("prior_failed_updates")) is int and origin["prior_failed_updates"] >= 0
             and type(origin.get("total_accounted_updates")) is int
             and type(run.get("prior_failed_updates")) is int and type(run.get("total_accounted_updates")) is int
             and origin["prior_failed_updates"] == run.get("prior_failed_updates")
             and origin["total_accounted_updates"] == run.get("total_accounted_updates")
             == origin["prior_failed_updates"] + run["updates"], "recovery_cost_mismatch")
    expected_counters = {"updates": run["updates"], "selected_step": run["best_step"],
                         "terminal_step": run["updates"], "Adam_steps": [run["updates"]]}
    _require(counters == expected_counters and all(type(counters.get(name)) is int
             for name in ("updates", "selected_step", "terminal_step"))
             and all(type(value) is int for value in counters["Adam_steps"]), "recovery_counters_mismatch")
    device = validation.get("reference_device")
    _require(type(device) is str and (device in ("cpu", "cuda") or
             (device.startswith("cuda:") and device[5:].isdigit())), "recovery_device_invalid")
    _require(validation.get("status") == "completed_passed" and type(validation.get("exit_code")) is int
             and validation["exit_code"] == 0 and validation.get("native_full_guard_passed") is True
             and type(validation.get("native_loader_calls")) is int and validation["native_loader_calls"] > 0,
             "recovery_validation_incomplete")
    _require(validation.get("scope") == "post_save_selected_snapshot_vs_native_reload"
             and validation.get("historical_live_reexecuted") is False
             and validation.get("native_device") == "cpu"
             and validation.get("native_state_method") == "all_tensor_torch_equal"
             and validation.get("native_state_digest_computed") is False
             and validation.get("reference_method") == "independent_original_factory_selected_snapshot"
             and validation.get("reference_storage_independent") is True
             and validation.get("directions") == ["forward"]
             and validation.get("input_row_indices") in ([0], [0, 1])
             and all(type(value) is int for value in validation["input_row_indices"])
             and _is_sha256(validation.get("input_encoding_sha256"))
             and validation.get("actual_encoded_inputs_equal") is True,
             "recovery_validation_scope_mismatch")
    _require(all(type(validation.get(name)) is int and validation[name] == 0
                 for name in ("new_optimizer_updates", "resave_count")), "recovery_added_training_or_resave")
    difference, threshold = validation.get("max_abs_diff"), validation.get("threshold")
    _require(type(threshold) in (int, float) and threshold == 1e-6
             and type(difference) in (int, float) and 0 <= difference <= threshold
             and math.isfinite(difference), "recovery_parity_not_passed")
    evidence = record.get("evidence")
    _require(type(evidence) is dict and {"origin_failure", "native_validation", "comparison", "state_identity"} <= set(evidence),
             "recovery_evidence_missing")
    for item in evidence.values():
        _recovery_reference(item, path.parent)
    state = record.get("state_evidence")
    checks = {"saved_equals_selected_best": "per_tensor_metadata_and_data_sha256",
              "independent_reference_equals_selected": "per_tensor_metadata_and_data_sha256",
              "device_reference_equals_selected": "per_tensor_metadata_and_data_sha256",
              "native_cpu_equals_selected": "all_tensor_torch_equal"}
    _require(type(state) is dict and set(state) == {"format", "tensors", "checks"}
             and state.get("format") == TENSOR_MANIFEST and state.get("checks") == checks
             and type(state.get("tensors")) is list and bool(state["tensors"]), "recovery_state_evidence_invalid")
    for item in state["tensors"]:
        _require(type(item) is dict and set(item) == {"key", "shape", "dtype", "bytes", "data_sha256"}
                 and type(item["key"]) is str and bool(item["key"]) and type(item["shape"]) is list
                 and all(type(value) is int and value >= 0 for value in item["shape"])
                 and type(item["dtype"]) is str and item["dtype"].startswith("torch.")
                 and type(item["bytes"]) is int and item["bytes"] >= 0 and _is_sha256(item["data_sha256"]),
                 "recovery_tensor_manifest_invalid")
    keys = [item["key"] for item in state["tensors"]]
    _require(keys == sorted(set(keys)), "recovery_tensor_manifest_invalid")
    return {"kind": RECOVERY_KIND, "recovery_validation": {"kind": RECOVERY_KIND, "path": str(path), "sha256": reference["sha256"]},
            "origin_process": origin, "recheck": validation, "counters": counters,
            "selected_tensor_manifest_format": TENSOR_MANIFEST,
            "selected_tensor_manifest_sha256": SM._digest(state["tensors"])}


def _check_recovery_state(approval, state):
    if approval.get("kind") == RECOVERY_KIND:
        _require(SM._digest(tensor_manifest(state)) == approval["selected_tensor_manifest_sha256"],
                 "recovery_selected_tensor_mismatch")


def _check_recovery_continuation(approval, continuation):
    if approval.get("kind") != RECOVERY_KIND:
        return
    binding = continuation.get("training_binding")
    dev_hashes = binding.get("dev_input_hashes") if type(binding) is dict else None
    _require(type(dev_hashes) is list and bool(dev_hashes) and len(dev_hashes) % 2 == 0
             and all(type(row) is dict and all(_is_sha256(row.get(key)) for key in ("raw", "visible", "view"))
                     for row in dev_hashes), "recovery_source_dev_binding_invalid")
    _require(approval["recheck"]["input_row_indices"] == list(range(min(2, len(dev_hashes) // 2))),
             "recovery_input_prefix_mismatch")
    optimizer_state = continuation.get("optimizer_state")
    _require(type(optimizer_state) is dict, "recovery_source_Adam_missing")
    optimizer = optimizer_state.get("state")
    _require(type(optimizer) is dict and bool(optimizer), "recovery_source_Adam_missing")
    steps = []
    for values in optimizer.values():
        _require(type(values) is dict, "recovery_source_Adam_mismatch")
        value = values.get("step")
        if torch.is_tensor(value):
            _require(value.numel() == 1, "recovery_source_Adam_mismatch")
            value = value.item()
        _require(type(value) in (int, float) and value == approval["counters"]["terminal_step"]
                 and math.isfinite(value), "recovery_source_Adam_mismatch")
        steps.append(int(value))
    _require(sorted(set(steps)) == approval["counters"]["Adam_steps"], "recovery_source_Adam_mismatch")
    mode = binding.get("checkpoint_selection")
    selected = continuation.get("best_state" if mode == "best-dev" else "terminal_state")
    _require(isinstance(selected, dict) and bool(selected), "recovery_source_selected_snapshot_missing")
    _check_recovery_state(approval, selected)


def _proof(brief_path, approved_sha256, run_id):
    brief_path = Path(brief_path).absolute()
    _require(SM.FT.file_sha256(brief_path) == approved_sha256, "approval_receipt_sha_mismatch")
    brief = _small_json(brief_path, 262144)
    _require(brief.get("source_sha256") == SM.source_binding(), "historical_source_drift")
    matches = [run for run in brief.get("runs", []) if run.get("run_id") == run_id]
    _require(len(matches) == 1, "historical_run_not_unique")
    run = matches[0]
    recovery = "recovery_validation" in run
    _require(run.get("complete_core_delivered") is True, "historical_run_not_validated")
    if not recovery:
        _require(brief.get("first_full_pool_streamed_save_loader_reload_passed") is True,
                 "historical_full_validation_missing")
        _require(run.get("validated") is True and run.get("exit_code") == 0, "historical_run_not_validated")
        difference = run.get("reload_max_abs_diff")
        _require(type(difference) in (int, float) and math.isfinite(difference) and 0 <= difference <= 1e-6,
                 "historical_reload_not_passed")
    _require(type(run.get("updates")) is int and run["updates"] > 0
             and type(run.get("best_step")) is int and 0 < run["best_step"] <= run["updates"], "historical_training_counters")
    reference = run["receipt"]
    compact = Path(reference["path"])
    if not compact.is_absolute():
        compact = brief_path.parent / compact
    _require(SM.FT.file_sha256(compact) == reference["sha256"], "compact_receipt_sha_mismatch")
    head = project_header(compact, stop="training")
    _require(head.get("run_id") == run_id and head.get("arm") in SM.ARM_CLASSES
             and type(head.get("original_loader_calls")) is int and head["original_loader_calls"] > 0,
             "compact_historical_loader_missing")
    approval = {"brief_path": str(brief_path), "brief_sha256": approved_sha256,
                "compact_path": str(compact.absolute()), "compact_sha256": reference["sha256"]}
    if recovery:
        approval.update(_recovery_proof(run["recovery_validation"], brief_path.parent, brief, run, head))
    return brief, run, head, approval


def validate_certificate(report, continuation, *, state_dict=None, input_params=None):
    _require(type(report) is dict and report.get("schema") == CERTIFICATE, "invalid_inference_certificate")
    _require(continuation is None and report.get("resumable") is False, "inference_export_not_resumable")
    _require(report.get("reused_historical_validation") is True
             and report.get("original_guard_ran_this_export") is False
             and report.get("full_pool_revalidated") is False, "inference_validation_scope_mismatch")
    proof, origin = report["approval"], report["origin"]
    brief, run, head, measured = _proof(proof["brief_path"], proof["brief_sha256"], origin["run_id"])
    synthetic = brief.get("synthetic", False)
    _require(type(synthetic) is bool and report.get("synthetic_proof") is synthetic,
             "inference_synthetic_proof_mismatch")
    _require(proof == measured, "inference_approval_binding_mismatch")
    for name, field in ((SM.META_FILE, "meta_sha256"), (SM.WEIGHTS_FILE, "weights_sha256")):
        _require(origin[field] == run["core_files"][name]["sha256"], "inference_origin_sha_mismatch")
    _require(origin["updates"] == run["updates"] and origin["selected_step"] == run["best_step"]
             and origin["arm"] == head["arm"] and report["source_sha256"] == SM.source_binding(),
             "inference_training_origin_mismatch")
    _require(type(origin.get("train_sha256")) is str and len(origin["train_sha256"]) == 64,
             "inference_origin_train_sha_missing")
    params = report["input_parameters"]
    _require(params["recipe"] == SM.INPUT_RECIPE and params["input_limits"] == dict(SM.INPUT_LIMITS)
             and type(params["max_length"]) is int and params["max_length"] > 0, "inference_input_mismatch")
    if input_params is not None:
        _require(params == input_params, "inference_input_mismatch")
    if state_dict is not None:
        _require(state_sha256(state_dict) == origin["selected_state_sha256"], "inference_selected_state_mismatch")
        _check_recovery_state(proof, state_dict)


def artifact_spec(arm):
    return dict(SM.artifact_spec(arm), schema=SCHEMA, validate_training=validate_certificate)


def read_metadata(directory):
    directory = Path(directory)
    meta = _small_json(directory / SM.META_FILE)
    _require(meta.get("schema") == SCHEMA, "inference_export_required")
    arm = meta.get("bindings", {}).get("arm")
    _require(arm in SM.ARM_CLASSES, "invalid_inference_arm")
    spec = artifact_spec(arm)
    expected = {"kind": spec["kind"], "classes": list(spec["classes"]), "label_space": spec["label_space"],
                "input_recipe": SM.INPUT_RECIPE, "input_limits": dict(SM.INPUT_LIMITS),
                "direction": SM.DIRECTION, "joint_roles": SM.ROLES[SM.DIRECTION], "bindings": SM.artifact_binding(arm)}
    _require(all(meta.get(key) == value for key, value in expected.items()), "inference_metadata_binding_mismatch")
    _require(SM.FT.file_sha256(directory / SM.WEIGHTS_FILE) == meta["weights_sha256"], "inference_weights_sha_mismatch")
    _require(meta.get("tokenizer_dir") == SM.NR.NLI_TOKENIZER_DIR
             and SM.NR.FEAT.model_fingerprint(directory / meta["tokenizer_dir"]) == meta["tokenizer_files_sha256"],
             "inference_tokenizer_files_mismatch")
    _require(meta.get("effective_config_dir") == "config"
             and SM.NR.FEAT.model_fingerprint(directory / "config") == meta["effective_config_files_sha256"],
             "inference_config_files_mismatch")
    validate_certificate(meta["train"], None, input_params={"recipe": meta["input_recipe"], "max_length": meta["max_length"],
                                                          "input_limits": meta["input_limits"]})
    _require(meta["train_sha256"] == SM.NR._tokenizer_state_sha256(meta["train"]), "inference_certificate_sha_mismatch")
    return meta


def identity(directory):
    directory = Path(directory)
    meta = read_metadata(directory)
    return {"meta_sha256": SM.FT.file_sha256(directory / SM.META_FILE), "weights_sha256": meta["weights_sha256"],
            "arm": meta["bindings"]["arm"], "schema": SCHEMA, "certificate_sha256": meta["train_sha256"],
            "policy_sha256": SM.FT.file_sha256(__file__),
            "input": {"recipe": meta["input_recipe"], "max_length": meta["max_length"], "limits": meta["input_limits"]}}


def load_inference_artifact(directory, *, device="cpu"):
    meta = read_metadata(directory)
    arm = meta["bindings"]["arm"]
    del meta
    scorer, meta, payload = SM.NR.load_joint_artifact(directory, spec=artifact_spec(arm), bindings=SM.artifact_binding(arm), device=device)
    _require(state_sha256(scorer.model.state_dict()) == meta["train"]["origin"]["selected_state_sha256"],
             "inference_loaded_state_mismatch")
    return scorer, meta, payload


def export_inference(source, directory, *, brief_receipt, approved_receipt_sha256, run_id):
    source, directory = Path(source).absolute(), Path(directory).absolute()
    _require(not directory.exists() and not directory.is_relative_to(source), "fresh_inference_destination_required")
    _, run, compact, approval = _proof(brief_receipt, approved_receipt_sha256, run_id)
    measured = {name: SM.FT.file_sha256(source / name) for name in (SM.META_FILE, SM.WEIGHTS_FILE)}
    _require(all(measured[name] == run["core_files"][name]["sha256"] for name in measured), "source_artifact_sha_mismatch")
    head = project_header(source / SM.META_FILE)
    arm = compact["arm"]
    original = SM.artifact_spec(arm)
    for key, value in (("schema", SM.ARTIFACT_SCHEMA), ("kind", original["kind"]), ("classes", list(original["classes"])),
                       ("label_space", original["label_space"]), ("input_recipe", SM.INPUT_RECIPE),
                       ("direction", SM.DIRECTION), ("joint_roles", SM.ROLES[SM.DIRECTION]),
                       ("weights_file", SM.WEIGHTS_FILE), ("weights", "full")):
        _require(head.get(key) == value, "source_header_binding_mismatch")
    _require(head["weights_sha256"] == measured[SM.WEIGHTS_FILE], "source_header_weights_mismatch")
    payload = torch.load(source / SM.WEIGHTS_FILE, weights_only=True, map_location="cpu", mmap=True)
    continuation = payload.pop("continuation", None)
    _require(type(continuation) is dict and continuation.get("completed_steps") == run["updates"],
             "source_training_counters_mismatch")
    selection_mode = continuation.get("training_binding", {}).get("checkpoint_selection")
    selected_step = continuation.get("selected_step") if selection_mode == "best-dev" else continuation["completed_steps"]
    _require(selected_step == run["best_step"], "source_selected_step_mismatch")
    _check_recovery_continuation(approval, continuation)
    del continuation
    state = payload.pop("state_dict")
    _check_recovery_state(approval, state)
    _require(payload.get("bindings") == SM.artifact_binding(arm), "source_payload_bindings_mismatch")
    for key in ("schema", "kind", "classes", "label_space", "weights", "direction", "joint_roles",
                "backbone_config_sha256", "tokenizer_state_sha256", "dropout", "max_length", "hidden_size"):
        _require(payload.get(key) == head.get(key), "source_payload_header_mismatch:" + key)
    _require(payload.get("recipe") == head["input_recipe"] and payload.get("input_limits") == dict(SM.INPUT_LIMITS),
             "source_payload_input_mismatch")
    _require(payload.get("effective_config_dir") == "config"
             and SM.NR.FEAT.model_fingerprint(source / "config") == payload["effective_config_files_sha256"],
             "source_config_files_mismatch")
    _require(SM.NR.FEAT.model_fingerprint(source / head["tokenizer_dir"]) == head["tokenizer_files_sha256"],
             "source_tokenizer_files_mismatch")
    tokenizer = SM.NR._reconstruct_tokenizer(source / head["tokenizer_dir"], use_fast=head["tokenizer_runtime"]["is_fast"], where="inference export")
    _require(SM.NR._tokenizer_state_sha256(SM.NR._tokenizer_state(tokenizer, where="inference export")) == head["tokenizer_state_sha256"]
             and SM.NR._tokenizer_runtime_summary(tokenizer) == head["tokenizer_runtime"], "source_tokenizer_state_mismatch")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(source / "config"), local_files_only=True)
    _require(SM.NR._config_digest(config) == head["backbone_config_sha256"], "source_config_value_mismatch")
    backbone = Path(head["backbone_path"])
    _require(not directory.is_relative_to(backbone.absolute()), "fresh_inference_destination_required")
    _require(SM.NR.FEAT.model_fingerprint(backbone) == head["backbone_fingerprint"], "source_backbone_files_mismatch")
    model, actual_config = SM.FT.build_joint_model(backbone, dropout=head["dropout"], num_labels=len(original["classes"]))
    _require(SM.NR._config_digest(actual_config) == head["backbone_config_sha256"], "source_backbone_config_mismatch")
    model.load_state_dict(state, strict=True, assign=True)
    SM.require_model(model, arm=arm)
    selected_sha = state_sha256(state)
    certificate = {"schema": CERTIFICATE, "resumable": False, "reused_historical_validation": True,
                   "original_guard_ran_this_export": False, "full_pool_revalidated": False,
                   "synthetic_proof": _small_json(approval["brief_path"], 262144).get("synthetic", False),
                   "approval": approval, "source_sha256": SM.source_binding(),
                   "input_parameters": {"recipe": payload["recipe"], "max_length": payload["max_length"], "input_limits": payload["input_limits"]},
                   "origin": {"run_id": run_id, "arm": arm, "updates": run["updates"], "selected_step": run["best_step"],
                              "train_sha256": payload["train_sha256"], "meta_sha256": measured[SM.META_FILE],
                              "weights_sha256": measured[SM.WEIGHTS_FILE], "selected_state_sha256": selected_sha,
                              "config_files_sha256": payload["effective_config_files_sha256"],
                              "tokenizer_files_sha256": head["tokenizer_files_sha256"]}}
    del payload
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, max_length=head["max_length"], arm=arm,
                               batch_size=head["batch_size"], direction=head["direction"], device="cpu")
    saved = False
    try:
        meta = SM.NR.save_joint_artifact(scorer, directory, spec=artifact_spec(arm), backbone_dir=backbone,
                                         train_meta=certificate, bindings=SM.artifact_binding(arm))
        saved = True
        del scorer, model, state
        reloaded, _, exported = load_inference_artifact(directory)
        _require(state_sha256(exported["state_dict"]) == selected_sha, "export_selected_state_mismatch")
        del reloaded, exported
        _require(all(SM.FT.file_sha256(source / name) == digest for name, digest in measured.items()), "source_changed_during_export")
    except BaseException:
        if saved:
            shutil.rmtree(directory)
        raise
    return {"schema": SCHEMA, "directory": str(directory), "run_id": run_id,
            "origin_meta_sha256": measured[SM.META_FILE], "origin_weights_sha256": measured[SM.WEIGHTS_FILE],
            "meta_sha256": SM.FT.file_sha256(directory / SM.META_FILE), "weights_sha256": meta["weights_sha256"],
            "selected_state_sha256": selected_sha, "original_guard_ran_this_export": False, "full_pool_revalidated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--brief-receipt", required=True)
    parser.add_argument("--approved-receipt-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = export_inference(args.source, args.out, brief_receipt=args.brief_receipt,
                                  approved_receipt_sha256=args.approved_receipt_sha256, run_id=args.run_id)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(json.dumps({"ok": False, "code": getattr(error, "code", type(error).__name__)}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
