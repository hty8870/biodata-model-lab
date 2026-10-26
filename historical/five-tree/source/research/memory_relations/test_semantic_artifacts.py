"""仅合成trained tiny证明；不生成真实研究通过凭据、不打开原巨工件或题行。"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from research.memory_relations import semantic_artifacts as A
from research.memory_relations import semantic_model as SM

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    torch.set_num_threads(1)
    def denied(*args, **kwargs):
        raise AssertionError("synthetic artifact mechanics must remain offline")
    monkeypatch.setattr(socket.socket, "connect", denied)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_source(tmp_path, arm, *, tiny=None, updates=1):
    from research.memory_relations.test_semantic_model import prepared_canonical_fixture, canonical_rows, args
    from research.memory_relations.test_finetune import _tiny_model_dir
    from transformers import AutoTokenizer
    tmp_path.mkdir(parents=True, exist_ok=True)
    training_dir = tmp_path / "training-data"
    training_dir.mkdir()
    fixture = prepared_canonical_fixture(training_dir)
    if tiny is None:
        backbone = _tiny_model_dir(tmp_path, corpus=json.dumps(canonical_rows(), ensure_ascii=False))
        tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
        mother, _ = SM.FT.build_joint_model(backbone, num_labels=3)
    else:
        _, backbone, tokenizer, mother = tiny
    tokenizer.add_tokens(["train", "dev", "alpha", "beta"])
    tokenizer.save_pretrained(str(backbone))
    mother.backbone.resize_token_embeddings(len(tokenizer))
    mother.backbone.save_pretrained(str(backbone))
    model = SM.initialize_arm(mother, arm)
    training = SM.train_semantic(model, tokenizer, fixture["fit"], arm=arm, **args(fixture, updates=updates, eval_every_steps=1))
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, max_length=64, arm=arm, device="cpu")
    source = tmp_path / ("source-" + arm)
    meta = SM.save_semantic_artifact(scorer, source, backbone_dir=backbone, training=training)
    loaded, _, payload = SM.load_semantic_artifact(source, device="cpu")
    probes = [row["pair"] for row in fixture["dev"][:2]]
    original = scorer.predict_both_proba(probes, role_independent_frame=True)
    restored = loaded.predict_both_proba(probes, role_independent_frame=True)
    difference = max(float(np.abs(original[key] - restored[key]).max()) for key in original)
    assert difference <= 1e-6
    assert A.state_sha256(payload["state_dict"]) == SM._state_sha256(payload["state_dict"])
    del payload
    run_id = "synthetic-" + arm
    compact = tmp_path / "synthetic-compact.json"
    _json(compact, {"run_id": run_id, "arm": arm, "original_loader_calls": 1,
                    "training": {"budget": training["budget"], "selection": training["selection"]},
                    "original_loader_and_ledger_passed": True, "original_cli_reload_parity_passed": difference <= 1e-6})
    brief = tmp_path / "synthetic-brief.json"
    core = {name: {"sha256": SM.FT.file_sha256(source / name)} for name in (SM.META_FILE, SM.WEIGHTS_FILE)}
    _json(brief, {"schema": "synthetic-semantic-validation-brief-v1", "synthetic": True,
                  "first_full_pool_streamed_save_loader_reload_passed": True, "source_sha256": SM.source_binding(),
                  "runs": [{"run_id": run_id, "exit_code": 0, "validated": True, "complete_core_delivered": True,
                            "updates": training["budget"]["actual_updates"], "best_step": training["selection"]["selected_step"],
                            "reload_max_abs_diff": difference, "core_files": core,
                            "receipt": {"path": compact.name, "sha256": SM.FT.file_sha256(compact)}}]})
    return {"source": source, "scorer": loaded, "fixture": fixture, "meta": meta, "probes": probes,
            "brief_receipt": brief, "approved_receipt_sha256": SM.FT.file_sha256(brief), "run_id": run_id}


def export_source(source, out):
    return A.export_inference(source["source"], out, **{key: source[key] for key in ("brief_receipt", "approved_receipt_sha256", "run_id")})


def make_export(tmp_path, arm, *, tiny=None):
    source = make_source(tmp_path, arm, tiny=tiny)
    out = tmp_path / ("inference-" + arm)
    result = export_source(source, out)
    return dict(source, artifact=out, result=result)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_trained_export_tensor_bytes_inputs_probabilities_and_marker(arm, tmp_path):
    data = make_export(tmp_path, arm)
    loaded, meta, payload = A.load_inference_artifact(data["artifact"])
    original = torch.load(data["source"] / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True, mmap=True)
    assert set(payload["state_dict"]) == set(original["state_dict"])
    for name, value in payload["state_dict"].items():
        other = original["state_dict"][name]
        assert value.dtype == other.dtype and value.shape == other.shape and torch.equal(value.view(torch.uint8), other.view(torch.uint8))
        actual = loaded.model.state_dict()[name]
        assert actual.dtype == other.dtype and actual.shape == other.shape
        assert torch.equal(actual.view(torch.uint8), other.view(torch.uint8))
    assert "continuation" not in payload and meta["schema"] == A.SCHEMA
    cert = meta["train"]
    assert cert["synthetic_proof"] is True and cert["origin"]["updates"] == 1 and cert["origin"]["selected_step"] == 1
    assert cert["original_guard_ran_this_export"] is cert["full_pool_revalidated"] is False
    assert cert["reused_historical_validation"] is True and cert["origin"]["train_sha256"] == original["train_sha256"]
    before = data["scorer"].prepare_inputs(data["probes"], role_independent_frame=True)
    after = loaded.prepare_inputs(data["probes"], role_independent_frame=True)
    assert before == after
    left = data["scorer"].predict_both_proba(data["probes"], role_independent_frame=True)
    right = loaded.predict_both_proba(data["probes"], role_independent_frame=True)
    for direction in left:
        np.testing.assert_allclose(left[direction], right[direction], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(left[direction].argmax(axis=1), right[direction].argmax(axis=1))
    with pytest.raises(ValueError):
        SM.load_semantic_artifact(data["artifact"])
    with pytest.raises(ValueError):
        SM.resume_continuation(data["artifact"])
    assert data["result"]["selected_state_sha256"] == A.state_sha256(original["state_dict"])


def test_bounded_header_never_reads_train_value_and_rejects_bad_header(tmp_path, monkeypatch):
    prefix = b'{"schema":"example","a":{"b":[1,2]},"train":'
    path = tmp_path / "meta.json"
    path.write_bytes(prefix + b'{"forbidden": "body"}')
    real_open = Path.open
    reads = []
    class Guard(io.BytesIO):
        def read(self, n=-1):
            assert n == 1 and self.tell() < len(prefix)
            reads.append(n)
            return super().read(n)
    raw = path.read_bytes()
    monkeypatch.setattr(Path, "open", lambda p, *args, **kwargs: Guard(raw) if p == path else real_open(p, *args, **kwargs))
    assert A.project_header(path) == {"schema": "example", "a": {"b": [1, 2]}}
    assert len(reads) == len(prefix)
    monkeypatch.setattr(Path, "open", real_open)
    for value in (b'[]', b'{"a":1}', b'{"a":1,"a":2,"train":{}}', b'{"a":"' + b'x' * 200):
        path.write_bytes(value)
        with pytest.raises(A.ArtifactError):
            A.project_header(path, limit=100)


@pytest.mark.parametrize("fault", ["approval", "compact", "meta", "weights", "tokenizer", "config", "source7", "state", "input"])
def test_export_refuses_tampered_origin_and_proof(tmp_path, fault):
    data = make_source(tmp_path, "c5")
    source = data["source"]
    if fault == "approval":
        data["approved_receipt_sha256"] = "0" * 64
    elif fault == "compact":
        with (tmp_path / "synthetic-compact.json").open("a", encoding="utf-8") as h:
            h.write(" ")
    elif fault in ("meta", "weights"):
        with (source / (SM.META_FILE if fault == "meta" else SM.WEIGHTS_FILE)).open("ab") as h:
            h.write(b" ")
    elif fault in ("tokenizer", "config"):
        path = next((source / fault).glob("*.json"))
        with path.open("a", encoding="utf-8") as h:
            h.write(" ")
    else:
        payload = torch.load(source / SM.WEIGHTS_FILE, weights_only=True, map_location="cpu")
        if fault == "source7":
            payload["bindings"]["source_sha256"]["semantic_model"] = "0" * 64
        elif fault == "input":
            payload["max_length"] = 32
        else:
            payload["state_dict"]["classifier.weight"] = payload["state_dict"]["classifier.weight"][:1]
        torch.save(payload, source / SM.WEIGHTS_FILE)
        meta = SM.D.read_json(source / SM.META_FILE)
        meta["weights_sha256"] = SM.FT.file_sha256(source / SM.WEIGHTS_FILE)
        _json(source / SM.META_FILE, meta)
        brief = SM.D.read_json(data["brief_receipt"])
        for name in (SM.META_FILE, SM.WEIGHTS_FILE):
            brief["runs"][0]["core_files"][name]["sha256"] = SM.FT.file_sha256(source / name)
        _json(data["brief_receipt"], brief)
        data["approved_receipt_sha256"] = SM.FT.file_sha256(data["brief_receipt"])
    with pytest.raises((ValueError, RuntimeError)):
        export_source(data, tmp_path / "denied")
    assert not (tmp_path / "denied").exists()


def test_export_uses_single_NR_save_load_engine_and_no_original_guard_this_export(tmp_path):
    data = make_source(tmp_path, "dual3")
    calls = []
    def profile(frame, event, arg):
        if event == "call" and frame.f_code in (SM.NR.save_joint_artifact.__code__, SM.NR.load_joint_artifact.__code__, SM.validate_training_ledger.__code__):
            calls.append(frame.f_code.co_name)
    old = sys.getprofile()
    try:
        sys.setprofile(profile)
        export_source(data, tmp_path / "out")
    finally:
        sys.setprofile(old)
    assert calls.count("save_joint_artifact") == calls.count("load_joint_artifact") == 1
    assert "validate_training_ledger" not in calls
    with pytest.raises(A.ArtifactError, match="fresh_inference_destination"):
        export_source(data, tmp_path / "out")


def test_fake_certificate_cannot_replace_caller_pinned_receipt(tmp_path):
    data = make_export(tmp_path, "c5")
    cert = copy.deepcopy(SM.D.read_json(data["artifact"] / SM.META_FILE)["train"])
    cert["approval"]["brief_sha256"] = "0" * 64
    with pytest.raises(A.ArtifactError, match="approval_receipt_sha_mismatch"):
        A.validate_certificate(cert, None)
    cert["original_guard_ran_this_export"] = True
    with pytest.raises(A.ArtifactError, match="inference_validation_scope"):
        A.validate_certificate(cert, None)


def test_export_real_CLI(tmp_path):
    data = make_source(tmp_path, "c5")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))), PYTHONSAFEPATH="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    process = subprocess.run([sys.executable, "-B", "-m", "research.memory_relations.semantic_artifacts", "--source", str(data["source"]),
                              "--out", str(tmp_path / "cli-export"), "--brief-receipt", str(data["brief_receipt"]),
                              "--approved-receipt-sha256", data["approved_receipt_sha256"], "--run-id", data["run_id"]],
                             cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result["schema"] == A.SCHEMA and result["original_guard_ran_this_export"] is False


def test_original_training_audit_guard_and_new_inference_schema_remain_separate(tmp_path, record_testsuite_property):
    data = make_source(tmp_path, "c5")
    path = data["source"] / SM.META_FILE
    meta = SM.D.read_json(path)
    weights_sha = SM.FT.file_sha256(data["source"] / SM.WEIGHTS_FILE)
    record_testsuite_property("synthetic_metadata_pretty_bytes", path.stat().st_size)
    path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    record_testsuite_property("synthetic_metadata_compact_bytes", path.stat().st_size)
    assert path.stat().st_size <= A.META_LIMIT
    assert SM.D.read_json(path) == meta and meta["train"]["budget"]["actual_updates"] == 1
    assert SM.FT.file_sha256(data["source"] / SM.WEIGHTS_FILE) == weights_sha
    with pytest.raises(A.ArtifactError, match="inference_export_required"):
        A.load_inference_artifact(data["source"])
    payload = torch.load(data["source"] / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    meta["train"]["budget"]["actual_updates"] = 99
    meta["train_sha256"] = payload["train_sha256"] = SM.NR._tokenizer_state_sha256(meta["train"])
    torch.save(payload, data["source"] / SM.WEIGHTS_FILE)
    meta["weights_sha256"] = SM.FT.file_sha256(data["source"] / SM.WEIGHTS_FILE)
    _json(path, meta)
    with pytest.raises(ValueError, match="training ledger mismatch: completed/actual updates"):
        SM.load_semantic_artifact(data["source"])


def test_exported_state_bytes_cannot_be_changed_by_rehashing_only_new_files(tmp_path):
    data = make_export(tmp_path, "dual3")
    path = data["artifact"] / SM.META_FILE
    meta = SM.D.read_json(path)
    payload = torch.load(data["artifact"] / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    payload["state_dict"]["classifier.weight"][0, 0] += 1
    torch.save(payload, data["artifact"] / SM.WEIGHTS_FILE)
    meta["weights_sha256"] = SM.FT.file_sha256(data["artifact"] / SM.WEIGHTS_FILE)
    _json(path, meta)
    with pytest.raises(A.ArtifactError, match="inference_selected_state_mismatch"):
        A.load_inference_artifact(data["artifact"])


def test_export_failed_reload_removes_only_its_own_new_target(tmp_path, monkeypatch):
    data = make_source(tmp_path, "c5")
    before = {name: SM.FT.file_sha256(data["source"] / name) for name in (SM.META_FILE, SM.WEIGHTS_FILE)}
    def fail(*args, **kwargs):
        raise A.ArtifactError("synthetic_reload_failure")
    monkeypatch.setattr(A, "load_inference_artifact", fail)
    with pytest.raises(A.ArtifactError, match="synthetic_reload_failure"):
        export_source(data, tmp_path / "failed-target")
    assert not (tmp_path / "failed-target").exists()
    assert all(SM.FT.file_sha256(data["source"] / name) == digest for name, digest in before.items())


def test_loaded_model_bytes_are_checked_after_original_engine_return(tmp_path):
    data = make_export(tmp_path, "c5")
    def profile(frame, event, result):
        if event == "return" and frame.f_code is SM.NR.load_joint_artifact.__code__ and result is not None:
            with torch.no_grad():
                result[0].model.classifier.weight[0, 0] += 1
    old = sys.getprofile()
    try:
        sys.setprofile(profile)
        with pytest.raises(A.ArtifactError, match="inference_loaded_state_mismatch"):
            A.load_inference_artifact(data["artifact"])
    finally:
        sys.setprofile(old)


@pytest.mark.parametrize("fault", ["missing", "flipped"])
def test_certificate_synthetic_marker_cannot_be_removed_or_flipped(tmp_path, fault):
    data = make_export(tmp_path, "c5")
    report = SM.D.read_json(data["artifact"] / SM.META_FILE)["train"]
    if fault == "missing":
        report.pop("synthetic_proof")
    else:
        report["synthetic_proof"] = False
    with pytest.raises(A.ArtifactError, match="inference_synthetic_proof_mismatch"):
        A.validate_certificate(report, None)


def test_brief_synthetic_marker_is_strict_bool_and_absent_defaults_false(tmp_path):
    data = make_export(tmp_path, "c5")
    report = SM.D.read_json(data["artifact"] / SM.META_FILE)["train"]
    brief = SM.D.read_json(data["brief_receipt"])
    brief.pop("synthetic")
    path = tmp_path / "synthetic-default-brief.json"
    _json(path, brief)
    report["approval"]["brief_path"] = str(path.absolute())
    report["approval"]["brief_sha256"] = SM.FT.file_sha256(path)
    report["synthetic_proof"] = False
    A.validate_certificate(report, None)
    report.pop("synthetic_proof")
    with pytest.raises(A.ArtifactError, match="inference_synthetic_proof_mismatch"):
        A.validate_certificate(report, None)
    brief["synthetic"] = 1
    _json(path, brief)
    report["approval"]["brief_sha256"] = SM.FT.file_sha256(path)
    report["synthetic_proof"] = 1
    with pytest.raises(A.ArtifactError, match="inference_synthetic_proof_mismatch"):
        A.validate_certificate(report, None)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_export_preserves_selected_state_when_terminal_is_later_and_different(tmp_path, monkeypatch, arm):
    evaluate = SM.eval_semantic
    dev_scores = iter((0.1, 0.2))
    observed = []
    def synthetic_dev_sequence(*args, **kwargs):
        actual = evaluate(*args, **kwargs)
        score = next(dev_scores)
        observed.append(score)
        return dict(actual, loss=score, row_loss=score)
    with monkeypatch.context() as patch:
        patch.setattr(SM, "eval_semantic", synthetic_dev_sequence)
        data = make_source(tmp_path, arm, updates=2)
    assert observed == [0.1, 0.2]
    original_scorer, meta, original = SM.load_semantic_artifact(data["source"])
    continuation = original["continuation"]
    selected = meta["train"]["selection"]["selected_step"]
    terminal = continuation["completed_steps"]
    assert selected == 1 and terminal == 2 and selected < terminal
    assert meta["train"]["budget"]["actual_updates"] == terminal
    assert all(int(value["step"]) == 2 for value in continuation["optimizer_state"]["state"].values())
    assert any(not torch.equal(original["state_dict"][key], continuation["terminal_state"][key]) for key in original["state_dict"])
    selected_sha = A.state_sha256(original["state_dict"])
    terminal_sha = A.state_sha256(continuation["terminal_state"])
    assert selected_sha == A.state_sha256(continuation["best_state"]) != terminal_sha
    out = tmp_path / "selected-export"
    result = export_source(data, out)
    scorer, exported_meta, exported = A.load_inference_artifact(out)
    assert result["selected_state_sha256"] == A.state_sha256(exported["state_dict"]) == selected_sha
    assert A.state_sha256(exported["state_dict"]) != terminal_sha
    assert all(torch.equal(exported["state_dict"][key], value) for key, value in original["state_dict"].items())
    assert exported_meta["train"]["origin"]["selected_step"] == selected
    assert exported_meta["train"]["origin"]["updates"] == terminal
    assert exported_meta["train"]["synthetic_proof"] is True
    before = original_scorer.predict_both_proba(data["probes"], role_independent_frame=True)
    after = scorer.predict_both_proba(data["probes"], role_independent_frame=True)
    for direction in before:
        np.testing.assert_allclose(before[direction], after[direction], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(before[direction].argmax(axis=1), after[direction].argmax(axis=1))


def _external_tensor_identity(state):
    combined, records = hashlib.sha256(), []
    for key, tensor in sorted(state.items()):
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        header = {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        combined.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        combined.update(data)
        records.append(dict(header, bytes=len(data), data_sha256=hashlib.sha256(data).hexdigest()))
    return combined.hexdigest(), records


def postsave_cpu_recheck(source, output):
    """新CPU进程真实native guard和独立selected参考；不模拟CUDA执行。"""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source, output = Path(source), Path(output)
    before = {name: SM.FT.file_sha256(source / name) for name in (SM.META_FILE, SM.WEIGHTS_FILE)}
    native, meta, payload = SM.load_semantic_artifact(source, device="cpu")
    continuation, saved = payload["continuation"], payload["state_dict"]
    best = continuation["best_state"]
    assert all(torch.equal(native.model.state_dict()[key], value) for key, value in saved.items())
    external, records = _external_tensor_identity(saved)
    assert _external_tensor_identity(best) == (external, records)
    assert A.tensor_manifest(saved) == records and A.state_sha256(saved) == SM._state_sha256(saved) != external
    arm = meta["bindings"]["arm"]
    model, config = SM.FT.build_joint_model(meta["backbone_path"], dropout=meta["dropout"], num_labels=len(SM.ARM_CLASSES[arm]))
    assert SM.NR._config_digest(config) == meta["backbone_config_sha256"]
    model.load_state_dict(best, strict=True)
    assert all(model.state_dict()[key].data_ptr() != native.model.state_dict()[key].data_ptr()
               for key, value in saved.items() if value.numel())
    assert _external_tensor_identity(model.state_dict()) == (external, records)
    reference = SM.SemanticScorer(model=model, tokenizer=native.tokenizer, arm=arm, max_length=native.max_length, device="cpu")
    _, dev = SM.read_prepared(meta["train"]["training_binding"]["prepared_data"]["directory"])
    rows = dev[:2]
    def predict(scorer):
        events = []
        def actual_inputs(module, args_, kwargs):
            assert all(value.device.type == "cpu" for value in kwargs.values())
            events.append({key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                 "values_sha256": SM.D.digest(value.detach().cpu().tolist())} for key, value in kwargs.items()})
        handle = scorer.model.register_forward_pre_hook(actual_inputs, with_kwargs=True)
        try:
            probabilities = scorer.predict_rows(rows)
        finally:
            handle.remove()
        assert events and np.isfinite(probabilities).all()
        return probabilities, events
    left, left_inputs = predict(native)
    right, right_inputs = predict(reference)
    assert left_inputs == right_inputs
    difference = float(np.max(np.abs(left - right)))
    assert difference <= 1e-6 and not torch.cuda.is_initialized()
    steps = sorted({int(value["step"]) for value in continuation["optimizer_state"]["state"].values()})
    validation = {"status": "completed_passed", "exit_code": 0, "native_full_guard_passed": True,
                  "native_loader_calls": 1, "native_device": "cpu", "reference_device": "cpu",
                  "native_state_method": "all_tensor_torch_equal", "native_state_digest_computed": False,
                  "reference_method": "independent_original_factory_selected_snapshot", "reference_storage_independent": True,
                  "scope": "post_save_selected_snapshot_vs_native_reload", "historical_live_reexecuted": False,
                  "directions": ["forward"], "input_row_indices": list(range(len(rows))),
                  "input_encoding_sha256": SM._digest(left_inputs), "actual_encoded_inputs_equal": True,
                  "max_abs_diff": difference, "threshold": 1e-6, "new_optimizer_updates": 0, "resave_count": 0}
    checks = {"saved_equals_selected_best": "per_tensor_metadata_and_data_sha256",
              "independent_reference_equals_selected": "per_tensor_metadata_and_data_sha256",
              "device_reference_equals_selected": "per_tensor_metadata_and_data_sha256",
              "native_cpu_equals_selected": "all_tensor_torch_equal"}
    _json(output / "state-evidence.json", {"synthetic": True, "external_aggregate_sha256": external,
          "tensors": records, "checks": checks, "native_cpu_independent_digest_computed": False})
    _json(output / "cpu-recheck.json", {"synthetic": True, "validation": validation, "counters": {
          "updates": meta["train"]["budget"]["actual_updates"], "selected_step": continuation["selected_step"],
          "terminal_step": continuation["completed_steps"], "Adam_steps": steps}})
    assert all(SM.FT.file_sha256(source / name) == digest for name, digest in before.items())


def _recovery_environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))), PYTHONSAFEPATH="1",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def _make_recovery_source(directory, arm):
    evaluate = SM.eval_semantic
    scores = iter((0.1, 0.2))
    def nonterminal(*args, **kwargs):
        return dict(evaluate(*args, **kwargs), loss=next(scores))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SM, "eval_semantic", nonterminal)
        data = make_source(directory / "origin", arm, updates=2)
    legacy_path, legacy_sha = data["brief_receipt"], data["approved_receipt_sha256"]
    env = _recovery_environment()
    failure = directory / "origin-failure.json"
    command = [sys.executable, "-B", "-c",
               "import json,sys; json.dump({'synthetic':True,'kind':'CPU_only_postsave_failure_fixture','exit_code':1,'reload_max_abs_diff':None},open(sys.argv[1],'w')); raise SystemExit(1)", str(failure)]
    failed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1
    _json(directory / "origin-process.json", {"command": command, "exit_code": failed.returncode,
          "stdout": failed.stdout, "stderr": failed.stderr, "synthetic": True})
    command = [sys.executable, "-B", "-c",
               "import sys; from research.memory_relations.test_semantic_artifacts import postsave_cpu_recheck; postsave_cpu_recheck(*sys.argv[1:])",
               str(data["source"]), str(directory)]
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    _json(directory / "recheck-process.json", {"command": command, "exit_code": process.returncode,
          "stdout": process.stdout, "stderr": process.stderr, "synthetic": True})
    assert process.returncode == 0, process.stdout + process.stderr
    recheck = SM.D.read_json(directory / "cpu-recheck.json")
    state = SM.D.read_json(directory / "state-evidence.json")
    compact = directory / "failed-origin-compact.json"
    _json(compact, {"run_id": data["run_id"], "arm": arm, "original_loader_calls": 1,
                   "training": {"budget": data["meta"]["train"]["budget"], "selection": data["meta"]["train"]["selection"]},
                   "original_cli_reload_parity_passed": None, "original_process_exit_code": failed.returncode})
    brief = SM.D.read_json(legacy_path)
    run = brief["runs"][0]
    run.update(exit_code=failed.returncode, validated=False, reload_max_abs_diff=None,
               prior_failed_updates=0, total_accounted_updates=run["updates"],
               receipt={"path": compact.name, "sha256": SM.FT.file_sha256(compact)})
    brief["first_full_pool_streamed_save_loader_reload_passed"] = False
    program_snapshot = directory / "validation-program.py"
    program_snapshot.write_bytes(Path(__file__).read_bytes())
    evidence = {role: {"path": str(path.absolute()), "sha256": SM.FT.file_sha256(path)} for role, path in {
        "origin_failure": failure, "origin_process": directory / "origin-process.json",
        "native_validation": directory / "cpu-recheck.json", "comparison": directory / "cpu-recheck.json",
        "state_identity": directory / "state-evidence.json", "validation_process": directory / "recheck-process.json",
        "validation_program": program_snapshot}.items()}
    record = {"schema": A.RECOVERY_SCHEMA, "kind": A.RECOVERY_KIND, "synthetic": True,
              "run_id": run["run_id"], "arm": arm, "source_sha256": SM.source_binding(), "core_files": run["core_files"],
              "origin_process": {"exit_code": failed.returncode, "reload_max_abs_diff": None,
                                 "prior_failed_updates": 0, "total_accounted_updates": run["updates"]},
              "counters": recheck["counters"], "validation": recheck["validation"],
              "state_evidence": {"format": A.TENSOR_MANIFEST, "tensors": state["tensors"], "checks": state["checks"]},
              "evidence": evidence}
    record_path = directory / "normalized-recovery.json"
    _json(record_path, record)
    run["recovery_validation"] = {"kind": A.RECOVERY_KIND, "path": record_path.name, "sha256": SM.FT.file_sha256(record_path)}
    brief_path = directory / "recovery-brief.json"
    _json(brief_path, brief)
    assert SM.FT.file_sha256(legacy_path) == legacy_sha
    return dict(data, brief_receipt=brief_path, approved_receipt_sha256=SM.FT.file_sha256(brief_path),
                recovery_record=record_path, legacy_brief_receipt=legacy_path, legacy_brief_sha256=legacy_sha)


@pytest.fixture(scope="module")
def recovered_sources(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("recovery-sources")
    return {arm: _make_recovery_source(directory / arm, arm) for arm in SM.ARM_CLASSES}


def _recovery_case(data, directory):
    directory.mkdir()
    brief = SM.D.read_json(data["brief_receipt"])
    record = SM.D.read_json(data["recovery_record"])
    brief["runs"][0]["receipt"]["path"] = str((data["brief_receipt"].parent / brief["runs"][0]["receipt"]["path"]).absolute())
    record_path = directory / "normalized-recovery.json"
    _json(record_path, record)
    brief["runs"][0]["recovery_validation"] = {"kind": A.RECOVERY_KIND, "path": record_path.name,
                                               "sha256": SM.FT.file_sha256(record_path)}
    path = directory / "brief.json"
    _json(path, brief)
    return dict(data, brief_receipt=path, approved_receipt_sha256=SM.FT.file_sha256(path), recovery_record=record_path)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_recovery_real_CPU_process_cli_export_reload_state_inputs_and_probabilities(recovered_sources, tmp_path, arm):
    data = _recovery_case(recovered_sources[arm], tmp_path / "case")
    legacy = A._proof(data["legacy_brief_receipt"], data["legacy_brief_sha256"], data["run_id"])
    assert len(legacy) == 4 and set(legacy[3]) == {"brief_path", "brief_sha256", "compact_path", "compact_sha256"}
    proof = A._proof(data["brief_receipt"], data["approved_receipt_sha256"], data["run_id"])
    assert len(proof) == 4 and proof[1]["exit_code"] == 1 and proof[1]["reload_max_abs_diff"] is None
    original = {name: SM.FT.file_sha256(data["source"] / name) for name in (SM.META_FILE, SM.WEIGHTS_FILE)}
    output = tmp_path / "export"
    process = subprocess.run([sys.executable, "-B", "-m", "research.memory_relations.semantic_artifacts",
        "--source", str(data["source"]), "--out", str(output), "--brief-receipt", str(data["brief_receipt"]),
        "--approved-receipt-sha256", data["approved_receipt_sha256"], "--run-id", data["run_id"]],
        cwd=ROOT, env=_recovery_environment(), capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    loaded, meta, payload = A.load_inference_artifact(output)
    cert = meta["train"]
    assert cert["synthetic_proof"] is True and cert["approval"]["kind"] == A.RECOVERY_KIND
    assert cert["approval"]["origin_process"]["exit_code"] == 1
    assert cert["approval"]["origin_process"]["reload_max_abs_diff"] is None
    assert cert["approval"]["recheck"]["exit_code"] == 0 and cert["approval"]["recheck"]["reference_device"] == "cpu"
    assert cert["approval"]["recheck"]["historical_live_reexecuted"] is False
    assert cert["approval"]["recheck"]["native_state_digest_computed"] is False
    assert cert["origin"]["selected_step"] == 1 and cert["origin"]["updates"] == cert["approval"]["counters"]["terminal_step"] == 2
    assert cert["approval"]["counters"]["Adam_steps"] == [2] and "continuation" not in payload
    assert A.state_sha256(payload["state_dict"]) == cert["origin"]["selected_state_sha256"]
    assert A.tensor_manifest(payload["state_dict"]) == SM.D.read_json(data["recovery_record"])["state_evidence"]["tensors"]
    assert data["scorer"].prepare_inputs(data["probes"], role_independent_frame=True) == loaded.prepare_inputs(data["probes"], role_independent_frame=True)
    left = data["scorer"].predict_both_proba(data["probes"], role_independent_frame=True)
    right = loaded.predict_both_proba(data["probes"], role_independent_frame=True)
    for direction in left:
        np.testing.assert_allclose(left[direction], right[direction], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(left[direction].argmax(axis=1), right[direction].argmax(axis=1))
    assert all(SM.FT.file_sha256(data["source"] / name) == value for name, value in original.items())
    cert = copy.deepcopy(cert)
    cert["synthetic_proof"] = False
    with pytest.raises(A.ArtifactError, match="inference_synthetic_proof_mismatch"):
        A.validate_certificate(cert, None)


@pytest.mark.parametrize("fault,code", [
    ("no_reference", "historical_full_validation_missing"), ("missing_file", "recovery_reference_missing"),
    ("bad_kind", "recovery_kind_invalid"), ("stale_sha", "recovery_reference_sha_mismatch"),
    ("missing_evidence", "recovery_evidence_missing"), ("bad_evidence", "recovery_reference_sha_mismatch"),
    ("source", "recovery_origin_binding_mismatch"), ("run", "recovery_origin_binding_mismatch"),
    ("arm", "recovery_origin_binding_mismatch"), ("core", "recovery_core_mismatch"),
    ("counter", "recovery_counters_mismatch"), ("cost", "recovery_cost_mismatch"),
    ("erased_exit", "recovery_original_failure_missing"), ("erased_null", "recovery_original_failure_missing"),
    ("recheck_failed", "recovery_validation_incomplete"), ("recheck_unfinished", "recovery_validation_incomplete"),
    ("old_live_scope", "recovery_validation_scope_mismatch"), ("native_digest_claim", "recovery_validation_scope_mismatch"),
    ("wrong_device", "recovery_device_invalid"), ("input_scope", "recovery_validation_scope_mismatch"),
    ("new_updates", "recovery_added_training_or_resave"), ("resave", "recovery_added_training_or_resave"),
    ("nan", "recovery_record_invalid"), ("inf", "recovery_record_invalid"),
    ("wide_threshold", "recovery_parity_not_passed"), ("tensor", "recovery_selected_tensor_mismatch"),
    ("aggregate", "recovery_state_evidence_invalid"), ("synthetic", "recovery_synthetic_mismatch"),
])
def test_recovery_rejects_invalid_explicit_evidence(recovered_sources, tmp_path, fault, code):
    data = _recovery_case(recovered_sources["c5"], tmp_path / "case")
    brief = SM.D.read_json(data["brief_receipt"])
    record = SM.D.read_json(data["recovery_record"])
    run, validation = brief["runs"][0], record["validation"]
    if fault == "no_reference":
        run.pop("recovery_validation")
    elif fault == "missing_file":
        run["recovery_validation"]["path"] = "missing.json"
    elif fault == "bad_kind":
        run["recovery_validation"]["kind"] = "implicit_success"
    elif fault == "missing_evidence":
        record["evidence"].pop("origin_failure")
    elif fault == "bad_evidence":
        record["evidence"]["comparison"]["sha256"] = "0" * 64
    elif fault == "source":
        record["source_sha256"]["semantic_model"] = "0" * 64
    elif fault in ("run", "arm"):
        record["run_id" if fault == "run" else "arm"] = "wrong"
    elif fault == "core":
        record["core_files"][SM.META_FILE]["sha256"] = "0" * 64
    elif fault == "counter":
        record["counters"]["selected_step"] = 2
    elif fault == "cost":
        record["origin_process"]["prior_failed_updates"] = 1
    elif fault == "erased_exit":
        run["exit_code"] = 0
    elif fault == "erased_null":
        run["reload_max_abs_diff"] = 0
    elif fault in ("recheck_failed", "recheck_unfinished"):
        validation["exit_code" if fault == "recheck_failed" else "status"] = 1 if fault == "recheck_failed" else "running"
    elif fault == "old_live_scope":
        validation["scope"] = "original_live_probabilities_replayed"
    elif fault == "native_digest_claim":
        validation["native_state_digest_computed"] = True
    elif fault == "wrong_device":
        validation["reference_device"] = "unknown"
    elif fault == "input_scope":
        validation["input_row_indices"] = [1, 0]
    elif fault in ("new_updates", "resave"):
        validation["new_optimizer_updates" if fault == "new_updates" else "resave_count"] = 1
    elif fault in ("nan", "inf", "wide_threshold"):
        validation["max_abs_diff" if fault != "wide_threshold" else "threshold"] = {
            "nan": float("nan"), "inf": float("inf"), "wide_threshold": 1e-3}[fault]
    elif fault == "tensor":
        record["state_evidence"]["tensors"][0]["data_sha256"] = "0" * 64
    elif fault == "aggregate":
        record["state_evidence"]["aggregate_sha256"] = "external_algorithm_not_A_state_sha"
    elif fault == "synthetic":
        record["synthetic"] = False
    _json(data["recovery_record"], record)
    if fault not in ("no_reference", "missing_file", "bad_kind", "stale_sha"):
        run["recovery_validation"]["sha256"] = SM.FT.file_sha256(data["recovery_record"])
    elif fault == "stale_sha":
        with data["recovery_record"].open("a", encoding="utf-8") as stream:
            stream.write(" ")
    _json(data["brief_receipt"], brief)
    data["approved_receipt_sha256"] = SM.FT.file_sha256(data["brief_receipt"])
    with pytest.raises(A.ArtifactError, match=code):
        export_source(data, tmp_path / "denied")
    assert not (tmp_path / "denied").exists()


@pytest.mark.parametrize("fault,code", [("Adam", "recovery_source_Adam_mismatch"),
                                        ("selected_best", "recovery_selected_tensor_mismatch")])
def test_recovery_checks_already_loaded_continuation_without_new_loader(recovered_sources, fault, code):
    data = recovered_sources["c5"]
    approval = A._proof(data["brief_receipt"], data["approved_receipt_sha256"], data["run_id"])[3]
    payload = torch.load(data["source"] / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True, mmap=True)
    continuation = copy.deepcopy(payload["continuation"])
    if fault == "Adam":
        next(iter(continuation["optimizer_state"]["state"].values()))["step"] += 1
    else:
        continuation["best_state"]["classifier.weight"][0, 0] += 1
    with pytest.raises(A.ArtifactError, match=code):
        A._check_recovery_continuation(approval, continuation)


def test_recovery_rejects_shortened_real_dev_prefix(recovered_sources, tmp_path):
    data = _recovery_case(recovered_sources["c5"], tmp_path / "case")
    dev_hashes = data["meta"]["train"]["training_binding"]["dev_input_hashes"]
    assert len(data["probes"]) == min(2, len(dev_hashes) // 2) == 2
    record = SM.D.read_json(data["recovery_record"])
    assert record["synthetic"] is True and record["validation"]["input_row_indices"] == [0, 1]
    assert record["counters"]["selected_step"] == 1 and record["counters"]["terminal_step"] == 2
    record["validation"]["input_row_indices"] = [0]
    _json(data["recovery_record"], record)
    brief = SM.D.read_json(data["brief_receipt"])
    brief["runs"][0]["recovery_validation"]["sha256"] = SM.FT.file_sha256(data["recovery_record"])
    _json(data["brief_receipt"], brief)
    data["approved_receipt_sha256"] = SM.FT.file_sha256(data["brief_receipt"])
    with pytest.raises(A.ArtifactError, match="recovery_input_prefix_mismatch"):
        export_source(data, tmp_path / "denied")
    assert not (tmp_path / "denied").exists()


@pytest.mark.parametrize("case", ["single_dev", "empty", "odd", "invalid"])
def test_recovery_dev_prefix_binding_mechanics(recovered_sources, case):
    data = recovered_sources["c5"]
    approval = A._proof(data["brief_receipt"], data["approved_receipt_sha256"], data["run_id"])[3]
    assert SM.D.read_json(data["recovery_record"])["synthetic"] is True
    payload = torch.load(data["source"] / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True, mmap=True)
    continuation = copy.deepcopy(payload["continuation"])
    binding = continuation["training_binding"]
    binding["dev_input_hashes"] = binding["dev_input_hashes"][:2]
    approval["recheck"]["input_row_indices"] = [0]
    if case == "single_dev":
        A._check_recovery_continuation(approval, continuation)
    else:
        binding["dev_input_hashes"] = {"empty": [], "odd": binding["dev_input_hashes"][:1], "invalid": [{}, {}]}[case]
        with pytest.raises(A.ArtifactError, match="recovery_source_dev_binding_invalid"):
            A._check_recovery_continuation(approval, continuation)
