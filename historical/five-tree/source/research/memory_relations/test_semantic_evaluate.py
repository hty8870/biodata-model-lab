"""五叶consumer的合成CPU合同测试；不读真实开发/封存题、不证明模型质量。"""
from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from research.memory_relations import metrics as M
from research.memory_relations import semantic_data as D
from research.memory_relations import semantic_evaluate as E
from research.memory_relations import semantic_model as SM
from research.memory_relations.test_finetune import _tiny_model_dir
from research.memory_relations.test_semantic_data import (
    _allocated_row, _qualified_construction_evidence, construction,
)
from research.memory_relations.test_semantic_model import qualified_consumer_fixture

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    def denied(*args, **kwargs):
        raise AssertionError("synthetic evaluation must be offline")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    torch.set_num_threads(1)


@pytest.fixture
def tiny(tmp_path):
    from transformers import AutoTokenizer
    from research.memory_relations.test_semantic_model import canonical_rows
    fixture = qualified_consumer_fixture(tmp_path)
    corpus = json.dumps([fixture["source"], canonical_rows(), canonical_rows("dev")], ensure_ascii=False)
    backbone = _tiny_model_dir(tmp_path, corpus=corpus)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    tokenizer.add_tokens(["same_domain"])
    tokenizer.save_pretrained(str(backbone))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        mother, _ = SM.FT.build_joint_model(backbone, num_labels=3)
        mother.backbone.resize_token_embeddings(len(tokenizer))
        mother.backbone.save_pretrained(str(backbone))
    return fixture, backbone, tokenizer, mother


def _artifact(tiny, tmp_path, arm):
    from research.memory_relations.test_semantic_artifacts import make_export
    from research.memory_relations import semantic_artifacts as A
    data = make_export(tmp_path / "artifact-export", arm, tiny=tiny)
    return A.load_inference_artifact(data["artifact"])[0], data["artifact"]


def _onehot(classes, label):
    return np.eye(len(classes))[classes.index(label)]


@pytest.mark.parametrize("leaf", SM.SR.SEMANTIC_RELATIONS)
def test_dyad_mapping_is_derived_and_all_legal_vertices_decode(leaf):
    forward_parent = SM.SR.SEMANTIC_TO_NLI[leaf]
    reverse_parent = SM.SR.SEMANTIC_TO_NLI[SM.SR.swap_relation(leaf)]
    result = E.decode_both({"forward": _onehot(SM.SR.NLI_CLASSES, forward_parent)[None, :],
                            "reverse": _onehot(SM.SR.NLI_CLASSES, reverse_parent)[None, :]}, "dual3")
    assert E.legal_dyads()[(forward_parent, reverse_parent)] == leaf
    assert result["forward"]["prediction"] == leaf
    assert result["reverse"]["prediction"] == SM.SR.swap_relation(leaf)
    assert result["forward"]["valid_mass"] == 1 and result["forward"]["invalid_mass"] == 0
    assert result["forward"]["probabilities"] == _onehot(SM.SR.SEMANTIC_RELATIONS, leaf).tolist()


def test_invalid_hard_dyad_keeps_probabilities_but_never_forces_a_legal_leaf():
    result = E.decode_both({"forward": [[.1, .2, .7]], "reverse": [[.7, .2, .1]]}, "dual3")
    row = result["forward"]
    assert result["parents"]["forward"]["prediction"] == "contradiction"
    assert result["parents"]["reverse"]["prediction"] == "entailment"
    assert row["prediction"] is None and row["status"] == "invalid_dyad"
    q = [.07, .02, .14, .07, .04]
    assert row["legal_cell_mass"] == pytest.approx(q)
    assert row["valid_mass"] == pytest.approx(.34)
    assert row["invalid_mass"] == pytest.approx(.66)
    assert row["probabilities"] == pytest.approx(np.asarray(q) / .34)
    assert result["reverse"]["probabilities"] == pytest.approx(SM.SR.swap_probabilities(row["probabilities"]))
    hard = M.classification_summary([0], [-1], classes=SM.SR.SEMANTIC_RELATIONS)
    probability = M.probability_summary([row["probabilities"]], [0], n_classes=5)
    assert hard["failure_weight"] == 1 and probability["coverage"] == 1


def test_zero_legal_mass_does_not_fabricate_probabilities_or_fix_invalid_hard_label():
    result = E.decode_both({"forward": [[0., 0., 1.]], "reverse": [[1., 0., 0.]]}, "dual3")["forward"]
    assert result["status"] == "invalid_dyad" and result["prediction"] is None
    assert result["valid_mass"] == 0 and result["invalid_mass"] == 1
    assert result["probabilities"] is None and result["probability_status"] == "zero_valid_mass"


@pytest.mark.parametrize("arm", ["c5", "dual3"])
@pytest.mark.parametrize("fault,status", [(None, "missing_probability"), ([float("nan")], "nonfinite"),
                                           ([.5], "invalid_simplex"), ([], "invalid_shape")])
def test_missing_nonfinite_simplex_and_shape_are_explicit(arm, fault, status):
    size = len(SM.ARM_CLASSES[arm])
    value = fault
    if status in ("nonfinite", "invalid_simplex"):
        value = [fault[0]] * size
    result = E.decode_both({"forward": [value] if value is not None else None,
                            "reverse": [np.ones(size) / size]}, arm)
    assert result["forward"]["status"] == status
    assert result["forward"]["prediction"] is None and result["forward"]["probabilities"] is None
    if arm == "c5":
        assert result["reverse"]["status"] == "ok"


def test_c5_uses_forward_five_class_prediction_and_keeps_reverse_separate():
    values = {"forward": [[.1, .6, .1, .1, .1]], "reverse": [[.9, .01, .02, .03, .04]]}
    result = E.decode_both(values, "c5")
    assert result["forward"]["prediction"] == SM.SR.SEMANTIC_RELATIONS[1]
    assert result["reverse"]["prediction"] == SM.SR.SEMANTIC_RELATIONS[0]
    assert result["forward"]["probabilities"] == values["forward"][0]


def test_public_preparation_audits_the_same_shared_window_in_both_directions(tiny, monkeypatch):
    fixture, _, tokenizer, mother = tiny
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer,
                              max_length=64, arm="c5")
    bundle = D.load_qualified_source(fixture["manifest"])
    row = bundle["rows"][0]
    real = scorer.prepare_inputs
    calls = []
    def tracked(pairs, **kwargs):
        output = real(pairs, **kwargs)
        calls.append((copy.deepcopy(pairs), kwargs, output))
        return output
    monkeypatch.setattr(scorer, "prepare_inputs", tracked)
    monkeypatch.setattr(scorer, "predict_rows", lambda *_: pytest.fail("training-row path forbidden"))
    record = E.predict_one(scorer, row, bundle)
    assert len(calls) == 2 and calls[0][0] == calls[1][0] == D.student_pair(row)
    assert calls[0][1] == calls[1][1] == {"role_independent_frame": True}
    assert calls[0][2] == calls[1][2]
    audit = record["input_audit"]
    assert audit["raw_pair_sha256"] == calls[1][2][0]["raw_pair_sha256"]
    prepared = calls[1][2][0]
    assert audit["max_length"] == 64
    assert record["truncation"] == ("complete" if prepared["complete"] else "truncated")
    assert prepared["recipe"] == SM.SR.SEMANTIC_INPUT_RECIPE
    assert prepared["complete"] == (max(prepared["full_visible_token_lengths"]) <= 64)
    assert audit["visible_information_sha256"] == audit["directions"]["forward"]["visible_information_sha256"] \
        == audit["directions"]["reverse"]["visible_information_sha256"]
    for direction, view in calls[1][2][0]["directions"].items():
        assert audit["directions"][direction]["encoding_sha256"] == D.digest(view["encoding"])
        assert audit["directions"][direction]["fields"] == view["fields"]
        assert audit["directions"][direction]["role_body_covered"] == {"old": True, "new": True}
        assert view["token_length"] <= 64


def _record(root, cluster, *, prediction=None, probabilities=None, status="invalid_dyad"):
    return {"root_id": root, "cluster_id": cluster, "gold": "equivalent", "reverse_gold": "equivalent",
            "split": "dev", "family": "fixture", "language": "zh", "kind": "memory/memory", "truncation": "unknown",
            "technical_failure": status in E.TECHNICAL,
            **{direction: {"status": status, "prediction": prediction, "probabilities": probabilities}
               for direction in ("forward", "reverse")}}


def test_failures_keep_resolved_denominator_and_only_gold_fn_and_zero_gold_nll(monkeypatch):
    monkeypatch.setitem(E.STATISTICS, "bootstrap_samples", 25)
    records = [_record("a", "x", prediction="equivalent", probabilities=[1, 0, 0, 0, 0], status="ok"),
               _record("b", "x", probabilities=[0, 1, 0, 0, 0]),
               _record("c", "y", status="timeout"), _record("d", "y", status="abstain")]
    report = E.summarize(records)
    main = report["primary_forward"]
    assert report["resolved_roots"] == main["classification"]["n"] == 4
    assert main["classification"]["accuracy"] == .25
    assert main["classification"]["failure_weight"] == 3
    assert main["classification"]["per_class"]["equivalent"]["fn"] == 3
    assert main["probability"]["coverage"] == .5
    assert main["probability"]["nll"]["status"] == "positive_infinity"
    assert report["technical_failures"] == 1
    assert report["primary_cluster_bootstrap"]["nll"]["ci"] is None
    assert report["primary_cluster_bootstrap"]["nll"]["invalid"] == 25
    json.dumps(report, allow_nan=False)


def test_root_uniform_point_and_cluster_not_root_resampling_use_only_metrics_engine(monkeypatch):
    monkeypatch.setitem(E.STATISTICS, "bootstrap_samples", 37)
    records = [_record(str(i), "large" if i < 9 else "small", prediction="equivalent" if i < 9 else None,
                       probabilities=[1, 0, 0, 0, 0] if i < 9 else None,
                       status="ok" if i < 9 else "invalid_dyad") for i in range(10)]
    engine = M.cluster_bootstrap
    calls = []
    def tracked(groups, metric, **kwargs):
        calls.append((groups, kwargs))
        return engine(groups, metric, **kwargs)
    monkeypatch.setattr(M, "cluster_bootstrap", tracked)
    report = E.summarize(records)
    point = report["primary_cluster_bootstrap"]["accuracy"]
    expected = engine(["large"] * 9 + ["small"], lambda index, weights: float(np.average(index < 9, weights=weights)),
                       samples=37, seed=20261014, confidence=.95, weighting="row")
    assert point == expected and point["point"] == .9 and point["group_clusters"] == 2
    assert len(calls) == 5 and all(call[1]["weighting"] == "row" for call in calls)
    assert report["strata"]["family"]["fixture"]["classification"]["n"] == 10
    assert report["strata"]["family"]["fixture"]["clusters"] == 2
    with pytest.raises(E.SemanticEvaluationError, match="one_canonical"):
        E.summarize(records + [records[0]])


@pytest.mark.parametrize("stage,error,expected", [("prepare", ValueError("input"), "input_failure"),
                                                   ("predict", RuntimeError("inference"), "inference_failure"),
                                                   ("predict", TimeoutError("timeout"), "timeout")])
def test_per_root_technical_failure_is_not_retried_or_dropped(tiny, monkeypatch, stage, error, expected):
    fixture, _, tokenizer, mother = tiny
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, max_length=64, arm="c5")
    bundle = D.load_qualified_source(fixture["manifest"])
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(scorer, "prepare_inputs" if stage == "prepare" else "predict_both_proba", fail)
    result = E.predict_one(scorer, bundle["rows"][0], bundle)
    assert calls == [1] and result["technical_failure"] is True
    assert result["forward"]["status"] == result["reverse"]["status"] == expected
    assert result["forward"]["prediction"] is None
    assert (result["input_audit"] is None) == (stage == "prepare")


def _cli(*arguments, expected=0):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))),
               PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, "-B", "-m", "research.memory_relations.semantic_evaluate", *map(str, arguments)],
                             cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert process.returncode == expected, process.stdout + process.stderr
    return json.loads(process.stderr if expected == 2 else process.stdout)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_real_cli_qualified_source_artifact_predictions_report_and_no_overwrite(tiny, tmp_path, arm):
    fixture = tiny[0]
    _, artifact = _artifact(tiny, tmp_path, arm)
    before = SM.source_binding()
    weights = SM.FT.file_sha256(artifact / SM.WEIGHTS_FILE)
    output = tmp_path / "evaluation"
    arguments = ("evaluate", "--model", artifact, "--source-manifest", fixture["manifest"], "--allocation", fixture["allocation"],
                 "--split", "dev", "--out", output)
    result = _cli(*arguments)
    assert set(result) == {"status", "resolved_roots", "binding_sha256", "predictions", "report", "metadata"}
    assert result["status"] == "completed" and result["resolved_roots"] == 5
    records = D.load_rows(output / "predictions.jsonl")
    assert len({row["root_id"] for row in records}) == len(records) == 5
    assert all(row["split"] == "dev" and row["original_split"] == "pilot" for row in records)
    report, meta = D.read_json(output / "report.json"), D.read_json(output / "metadata.json")
    assert report["primary_forward"]["classification"]["n"] == 5
    assert report["primary_forward"]["probability"]["coverage"] == 1
    assert report["statistics"]["temperature"] == 1 and report["statistics"]["temperature_status"] == "fixed_uncalibrated"
    assert report["primary_cluster_bootstrap"]["accuracy"]["samples"] == 2000
    assert report["primary_cluster_bootstrap"]["accuracy"]["weighting"] == "row"
    assert meta["binding"]["scientific_sources"] == before == SM.source_binding()
    assert meta["binding"]["model"]["weights_sha256"] == weights == SM.FT.file_sha256(artifact / SM.WEIGHTS_FILE)
    for name, sha in meta["files_sha256"].items():
        assert SM.FT.file_sha256(output / name) == sha
    old = (output / "predictions.jsonl").read_bytes()
    assert _cli(*arguments, expected=2)["code"] == "FileExistsError"
    assert (output / "predictions.jsonl").read_bytes() == old
    missing_split = _cli("evaluate", "--model", artifact, "--source-manifest", fixture["manifest"],
                         "--split", "train", "--out", tmp_path / "bad-split", expected=2)
    assert missing_split["code"] == "development_requires_dev"


def _sealed_fixture(construction):
    fixture = construction
    rows = [_allocated_row(slot) for slot in (fixture["slots"][0], fixture["slots"][2])]
    context = D._construction_context(fixture["scope"], fixture["run"].name)
    evidence = _qualified_construction_evidence(fixture["run"] / "sealed", rows, context["audit"])
    return evidence


def _read_guard(monkeypatch, paths):
    forbidden = {os.path.normcase(os.path.abspath(path)) for path in paths}
    touched = []
    for name in ("open", "read_bytes", "read_text", "stat", "is_file", "exists"):
        original = getattr(Path, name)
        def guarded(path, *args, _original=original, **kwargs):
            if os.path.normcase(os.path.abspath(path)) in forbidden:
                touched.append(str(path))
                raise AssertionError("sealed content accessed before final lock")
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guarded)
    return touched


def test_missing_confirmation_bad_lock_and_development_read_zero_sealed_assets(tiny, tmp_path, construction, monkeypatch):
    evidence = _sealed_fixture(construction)
    _, artifact = _artifact(tiny, tmp_path, "c5")
    lock = tmp_path / "final-lock.json"
    kwargs = {"splits": ["blind_final"], "construction_scope": construction["scope"], "run_id": construction["run"].name}
    E.write_final_lock(artifact, evidence["manifest"], lock, **kwargs)
    with pytest.raises(FileExistsError):
        E.write_final_lock(artifact, evidence["manifest"], lock, **kwargs)
    forbidden = [*evidence["paths"].values(), construction["allocation"],
                 Path(evidence["document"]["blind_shards"]["fixture-task"]["path"])]
    touched = _read_guard(monkeypatch, forbidden)
    monkeypatch.setattr(E, "_load_scorer", lambda *_a, **_k: pytest.fail("model load before valid lock"))
    for confirm, lock_path in ((False, lock), (True, None)):
        with pytest.raises(E.SemanticEvaluationError, match="confirmation_and_lock"):
            E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="final", lock_path=lock_path,
                       confirm_final=confirm, **kwargs)
    with pytest.raises(E.SemanticEvaluationError, match="sealed_source_in_development"):
        E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="development", splits=["dev"])
    with pytest.raises(E.SemanticEvaluationError, match="development_scope_forbidden"):
        E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="development", splits=["dev"],
                   construction_scope=construction["scope"])
    for field, value in (("run_id", "other-run"), ("splits", ["ood_final"])):
        bad = dict(kwargs, **{field: value})
        with pytest.raises(E.SemanticEvaluationError, match="final_lock_mismatch"):
            E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="final", lock_path=lock,
                       confirm_final=True, **bad)
    broken = D.read_json(lock)
    broken["binding"]["statistics"]["temperature"] = 2
    other = tmp_path / "tampered-lock.json"
    other.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(E.SemanticEvaluationError, match="invalid_final_lock"):
        E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="final", lock_path=other,
                   confirm_final=True, **kwargs)
    assert not touched and not (tmp_path / "denied").exists()


def test_synthetic_final_cli_uses_real_construction_loader_without_stage_relabel(tiny, tmp_path, construction):
    evidence = _sealed_fixture(construction)
    _, artifact = _artifact(tiny, tmp_path, "dual3")
    lock, output = tmp_path / "final-lock.json", tmp_path / "final-evaluation"
    common = ("--model", artifact, "--source-manifest", evidence["manifest"], "--split", "blind_final",
              "--construction-scope", construction["scope"], "--run-id", construction["run"].name)
    result = _cli("lock-final", *common, "--out-lock", lock)
    assert result["binding_sha256"] == D.read_json(lock)["binding_sha256"]
    result = _cli("evaluate", *common, "--mode", "final", "--lock", lock, "--confirm-final", "--out", output)
    assert result["status"] == "completed" and result["resolved_roots"] == 2
    records = D.load_rows(output / "predictions.jsonl")
    assert all(row["split"] == row["original_split"] == "blind_final" for row in records)
    assert D.read_json(output / "metadata.json")["qualified_lineage"]["construction_scope"]["run_id"] == construction["run"].name
    assert _cli("evaluate", *common, "--mode", "final", "--lock", lock, "--out", tmp_path / "denied", expected=2)["code"] \
        == "final_confirmation_and_lock_required"


@pytest.mark.parametrize("fault", ["source", "meta", "weights", "statistics", "window", "science", "scope"])
def test_source_or_model_drift_after_lock_refused_before_loader(tiny, tmp_path, construction, monkeypatch, fault):
    evidence = _sealed_fixture(construction)
    _, artifact = _artifact(tiny, tmp_path, "c5")
    lock = tmp_path / "final-lock.json"
    kwargs = {"splits": ["blind_final"], "construction_scope": construction["scope"], "run_id": construction["run"].name}
    E.write_final_lock(artifact, evidence["manifest"], lock, **kwargs)
    monkeypatch.setattr(D, "load_qualified_source", lambda *_a, **_k: pytest.fail("source content read after lock drift"))
    if fault == "source":
        evidence["manifest"].write_text(json.dumps(evidence["document"], indent=2), encoding="utf-8")
    elif fault in ("meta", "window"):
        path = artifact / SM.META_FILE
        meta = D.read_json(path)
        meta["max_length" if fault == "window" else "diagnostic_fixture"] = 32
        path.write_text(json.dumps(meta), encoding="utf-8")
    elif fault == "weights":
        with (artifact / SM.WEIGHTS_FILE).open("ab") as handle:
            handle.write(b"changed-fixture")
    elif fault == "scope":
        with construction["scope"].open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        document = D.read_json(lock)
        if fault == "statistics":
            document["binding"]["statistics"]["n_bins"] = 11
        else:
            document["binding"]["scientific_sources"]["semantic_model"] = "0" * 64
        document["binding_sha256"] = D.digest(document["binding"])
        lock.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(E.SemanticEvaluationError, match="final_lock_mismatch|inference_weights_sha_mismatch|inference_input_mismatch"):
        E.evaluate(artifact, evidence["manifest"], tmp_path / "denied", mode="final", lock_path=lock,
                   confirm_final=True, **kwargs)
    assert not (tmp_path / "denied").exists()


def test_evaluation_writes_all_attempts_and_nonzero_exit_for_technical_failure(tiny, tmp_path, monkeypatch, capsys):
    fixture = tiny[0]
    scorer, artifact = _artifact(tiny, tmp_path, "c5")
    capsys.readouterr()
    real_predict = scorer.predict_both_proba
    calls = []
    def first_timeout(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("synthetic timeout")
        return real_predict(*args, **kwargs)
    monkeypatch.setattr(scorer, "predict_both_proba", first_timeout)
    monkeypatch.setattr(E, "_load_scorer", lambda *_a, **_k: scorer)
    monkeypatch.setitem(E.STATISTICS, "bootstrap_samples", 23)
    output = tmp_path / "failed-evaluation"
    assert E.main(["evaluate", "--model", str(artifact), "--source-manifest", str(fixture["manifest"]),
                   "--allocation", str(fixture["allocation"]), "--split", "dev", "--out", str(output)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    report = D.read_json(output / "report.json")
    rows = D.load_rows(output / "predictions.jsonl")
    assert receipt["status"] == "completed_with_technical_failures"
    assert len(rows) == len(calls) == report["resolved_roots"] == 5
    assert report["technical_failures"] == 1
    assert rows[0]["forward"]["status"] == "timeout"
    assert rows[0]["forward"]["hard_success"] is rows[0]["forward"]["probability_available"] is False
    assert rows[0]["input_audit"] is not None
    assert report["primary_forward"]["classification"]["failure_weight"] == 1
    assert report["primary_forward"]["probability"]["coverage"] == .8
    assert D.read_json(output / "metadata.json")["status"] == receipt["status"]


def test_original_dev_split_selection_does_not_need_allocation_or_scope(tiny, tmp_path, monkeypatch):
    fixture = tiny[0]
    rows = fixture["source"]
    for row in rows:
        row["split"] = "dev" if "-dev-" in row["root_id"] else "train"
    by_id = {row["id"]: row for row in rows}
    fixture["paths"]["source_rows"].write_bytes(D.B._jsonl_bytes(rows))
    for record in fixture["qualifications"]:
        record["source_row_sha256"] = D.digest(by_id[record["id"]])
    fixture["paths"]["qualifications"].write_bytes(D.B._jsonl_bytes(fixture["qualifications"]))
    packet = D.read_json(fixture["paths"]["packet_map"])
    for item in packet["items"]:
        item["source_row_sha256"] = D.digest(by_id[item["source_id"]])
    fixture["paths"]["packet_map"].write_bytes(D.B._json_bytes(packet))
    imported = D.read_json(fixture["paths"]["import_manifest"])
    imported["files_sha256"]["qualifications.jsonl"] = SM.FT.file_sha256(fixture["paths"]["qualifications"])
    fixture["paths"]["import_manifest"].write_bytes(D.B._json_bytes(imported))
    for name in ("source_rows", "qualifications", "packet_map", "import_manifest"):
        fixture["document"]["assets"][name]["sha256"] = SM.FT.file_sha256(fixture["paths"][name])
    fixture["manifest"].write_bytes(D.B._json_bytes(fixture["document"]))
    selected, bundle, lineage = E._source_rows(fixture["manifest"], mode="development", splits=["dev"],
                                              allocation=None, construction_scope=None, run_id=None)
    assert len(selected) == 5 and lineage is None and len(bundle["rows"]) == 10
    assert all(row["split"] == "dev" for row in selected)


def _identity_index(artifact, path):
    E._write_json(path, {"schema": E.IDENTITY_FORMAT, "model": E._model_identity(artifact)})
    return path


def _cohort_source(tmp_path, name, members):
    from research.memory_relations.test_semantic_data import row as data_row
    directory = tmp_path / name
    directory.mkdir()
    canonical = []
    for identifier, root, cluster, split, label in members:
        item = data_row(root, label=label)
        item.update(id=identifier, root_id=root, cluster_id=cluster, split=split)
        item["pair"]["old"]["text"] += name + root
        canonical.append(item)
    fixture = qualified_consumer_fixture(directory, canonical=canonical)
    return {"source_id": name, "source_manifest": str(fixture["manifest"]), "allocation": str(fixture["allocation"])}


def _write_cohort(tmp_path, sources):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"schema": E.COHORT_FORMAT, "sources": sources}), encoding="utf-8")
    return path


def test_combined_cohort_calls_summary_once_and_resamples_shared_clusters(tiny, tmp_path, monkeypatch):
    first = _cohort_source(tmp_path, "first", [("a", "a", "shared", "dev", "equivalent"),
                                               ("b", "b", "solo", "dev", "equivalent")])
    second = _cohort_source(tmp_path, "second", [("c", "c", "shared", "dev", "disjoint")])
    cohort = _write_cohort(tmp_path, [first, second])
    scorer, artifact = _artifact(tiny, tmp_path, "c5")
    with torch.no_grad():
        scorer.model.classifier.weight.zero_()
        scorer.model.classifier.bias.fill_(-20)
        scorer.model.classifier.bias[0] = 20
    monkeypatch.setattr(E, "_load_scorer", lambda *_args: scorer)
    monkeypatch.setitem(E.STATISTICS, "bootstrap_samples", 51)
    real_summary, summaries = E.summarize, []
    def tracked(records):
        summaries.append(copy.deepcopy(records))
        return real_summary(records)
    monkeypatch.setattr(E, "summarize", tracked)
    out = tmp_path / "cohort-evaluation"
    E.evaluate(artifact, None, out, mode="development", splits=["dev"], cohort=cohort)
    assert len(summaries) == 1 and len(summaries[0]) == 3
    report = D.read_json(out / "report.json")
    assert report["primary_forward"]["classification"]["accuracy"] == pytest.approx(2 / 3)
    assert report["primary_forward"]["classification"]["accuracy"] != .5
    expected = M.cluster_bootstrap(["shared", "solo", "shared"],
                                    lambda index, weights: float(np.average(index < 2, weights=weights)),
                                    samples=51, seed=20261014, confidence=.95, weighting="row")
    assert report["primary_cluster_bootstrap"]["accuracy"] == expected
    assert report["primary_forward"]["clusters"] == 2
    records = D.load_rows(out / "predictions.jsonl")
    assert [row["source_id"] for row in records] == ["first", "first", "second"]
    assert all(row["evidence"]["source_row_sha256"] and row["source_manifest_sha256"] for row in records)
    assert len(D.read_json(out / "metadata.json")["cohort_sources"]) == 2


@pytest.mark.parametrize("fault,code", [("id", "duplicate_cohort_canonical_id"), ("root", "duplicate_cohort_root"),
                                         ("split", "cohort_cluster_crosses_splits"), ("other_split_root", "duplicate_cohort_root"),
                                         ("other_split_id", "duplicate_cohort_canonical_id")])
def test_cohort_rejects_cross_source_duplicates_and_split_clusters(tmp_path, fault, code):
    first = _cohort_source(tmp_path, "first", [("a", "root-a", "shared", "dev", "equivalent")])
    members = [("a" if fault == "id" else "b", "root-a" if fault == "root" else "root-b",
                "other", "dev", "equivalent")]
    if fault == "split":
        members.append(("train-c", "root-c", "shared", "train", "equivalent"))
    elif fault.startswith("other_split"):
        members.append(("a" if fault == "other_split_id" else "train-c",
                        "root-a" if fault == "other_split_root" else "root-c", "train-only", "train", "equivalent"))
    second = _cohort_source(tmp_path, "second", members)
    cohort = _write_cohort(tmp_path, [first, second])
    specs = E._source_specs(None, cohort, mode="development")
    with pytest.raises(E.SemanticEvaluationError, match=code):
        E._cohort_rows(specs, mode="development", splits=["dev"])


def _sealed_cohort(tmp_path, construction):
    audit = D._construction_context(construction["scope"], construction["run"].name)["audit"]
    sources, forbidden = [], [construction["allocation"]]
    for name, slot in zip(("first", "second"), (construction["slots"][0], construction["slots"][2])):
        directory = tmp_path / ("sealed-" + name)
        directory.mkdir()
        fixture = _qualified_construction_evidence(directory, [_allocated_row(slot)], audit)
        sources.append({"source_id": name, "source_manifest": str(fixture["manifest"]),
                        "construction_scope": str(construction["scope"]), "run_id": construction["run"].name})
        forbidden.extend(fixture["paths"].values())
        forbidden.append(Path(fixture["document"]["blind_shards"]["fixture-task"]["path"]))
    return _write_cohort(tmp_path, sources), sources, forbidden


def test_final_lock_binds_entire_cohort_before_either_loader(tiny, tmp_path, construction, monkeypatch):
    cohort, sources, forbidden = _sealed_cohort(tmp_path, construction)
    _, artifact = _artifact(tiny, tmp_path, "c5")
    index = _identity_index(artifact, tmp_path / "identity.json")
    lock = tmp_path / "cohort-lock.json"
    E.write_final_lock(artifact, None, lock, splits=["blind_final"], cohort=cohort, identity_index=index)
    binding = D.read_json(lock)["binding"]
    assert len(binding["sources"]) == 2 and binding["cohort_sha256"] == SM.FT.file_sha256(cohort)
    touched = _read_guard(monkeypatch, forbidden)
    monkeypatch.setattr(E, "_load_scorer", lambda *_args: pytest.fail("model load after cohort lock drift"))
    for changed in (sources[:1], list(reversed(sources))):
        cohort.write_text(json.dumps({"schema": E.COHORT_FORMAT, "sources": changed}), encoding="utf-8")
        with pytest.raises(E.SemanticEvaluationError, match="final_lock_mismatch"):
            E.evaluate(artifact, None, tmp_path / "denied", mode="final", splits=["blind_final"],
                       cohort=cohort, identity_index=index, lock_path=lock, confirm_final=True)
    cohort.write_text(json.dumps({"schema": E.COHORT_FORMAT, "sources": sources}), encoding="utf-8")
    with Path(sources[1]["source_manifest"]).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(E.SemanticEvaluationError, match="final_lock_mismatch"):
        E.evaluate(artifact, None, tmp_path / "denied", mode="final", splits=["blind_final"],
                   cohort=cohort, identity_index=index, lock_path=lock, confirm_final=True)
    assert not touched


def test_real_cli_final_cohort_retains_both_sources_and_one_denominator(tiny, tmp_path, construction):
    cohort, _, _ = _sealed_cohort(tmp_path, construction)
    _, artifact = _artifact(tiny, tmp_path, "dual3")
    index = _identity_index(artifact, tmp_path / "identity.json")
    lock, out = tmp_path / "cohort-lock.json", tmp_path / "cohort-final"
    common = ("--model", artifact, "--cohort", cohort, "--identity-index", index, "--split", "blind_final")
    _cli("lock-final", *common, "--out-lock", lock)
    result = _cli("evaluate", *common, "--mode", "final", "--lock", lock, "--confirm-final", "--out", out)
    report = D.read_json(out / "report.json")
    assert result["resolved_roots"] == report["primary_forward"]["classification"]["n"] == 2
    assert {row["source_id"] for row in D.load_rows(out / "predictions.jsonl")} == {"first", "second"}


def _trained_artifact(tiny, tmp_path, arm):
    from research.memory_relations.test_semantic_artifacts import make_export
    data = make_export(tmp_path / "trained-export", arm, tiny=tiny)
    return data["fixture"], data["artifact"]


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_real_cli_trained_inference_export_retains_training_origin(tiny, tmp_path, arm):
    fixture, artifact = _trained_artifact(tiny, tmp_path, arm)
    index = _identity_index(artifact, tmp_path / "trained-identity.json")
    result = _cli("evaluate", "--model", artifact, "--identity-index", index,
                  "--source-manifest", fixture["manifest"], "--allocation", fixture["allocation"],
                  "--split", "dev", "--device", "cpu", "--out", tmp_path / "trained-evaluation")
    assert result["resolved_roots"] == 5 and result["status"] == "completed"
    certificate = D.read_json(artifact / SM.META_FILE)["train"]
    assert certificate["origin"]["updates"] == 1 and certificate["full_pool_revalidated"] is False


def test_inference_policy_calls_NR_once_and_never_reads_original_train_metadata(tiny, tmp_path, monkeypatch):
    from research.memory_relations import semantic_artifacts as A
    fixture, artifact = _trained_artifact(tiny, tmp_path, "c5")
    index = _identity_index(artifact, tmp_path / "identity.json")
    original_meta = artifact.parent / "source-c5" / SM.META_FILE
    original_read = Path.read_text
    def guarded_read(path, *args, **kwargs):
        assert path != original_meta, "consumer must not read original full train metadata"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    calls, state_checks = [], []
    def profile(frame, event, arg):
        if event == "call" and frame.f_code in (SM.NR.load_joint_artifact.__code__, SM.validate_training_ledger.__code__):
            calls.append((frame.f_code.co_name, frame.f_locals.get("device")))
        if event == "call" and frame.f_code is A.validate_certificate.__code__ and frame.f_locals.get("state_dict") is not None:
            state_checks.append(frame.f_locals["report"])
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        E.evaluate(artifact, fixture["manifest"], tmp_path / "indexed", mode="development", splits=["dev"],
                   allocation=fixture["allocation"], identity_index=index)
    finally:
        sys.setprofile(previous)
    assert calls == [("load_joint_artifact", "cpu")]
    assert len(state_checks) == 1 and state_checks[0]["origin"]["updates"] == 1
    assert state_checks[0]["original_guard_ran_this_export"] is False


@pytest.mark.parametrize("fault", ["meta", "weights", "facts"])
def test_identity_index_drift_or_false_facts_cannot_reach_data_loader(tiny, tmp_path, monkeypatch, fault):
    _, artifact = _artifact(tiny, tmp_path, "c5")
    index = _identity_index(artifact, tmp_path / "identity.json")
    if fault in ("meta", "weights"):
        with (artifact / (SM.META_FILE if fault == "meta" else SM.WEIGHTS_FILE)).open("ab") as handle:
            handle.write(b" ")
    else:
        document = D.read_json(index)
        document["model"]["input"]["max_length"] = 32
        index.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(D, "load_qualified_source", lambda *_a, **_k: pytest.fail("index drift opened source rows"))
    with pytest.raises(E.SemanticEvaluationError, match="model_identity_index_drift|inference_weights_sha_mismatch"):
        E.evaluate(artifact, tiny[0]["manifest"], tmp_path / "denied", mode="development", splits=["dev"],
                   allocation=tiny[0]["allocation"], identity_index=index)


def test_inference_origin_counters_cannot_be_rewritten_as_new_training(tiny, tmp_path, monkeypatch):
    fixture, artifact = _trained_artifact(tiny, tmp_path, "c5")
    index = _identity_index(artifact, tmp_path / "identity.json")
    path = artifact / SM.META_FILE
    meta = D.read_json(path)
    payload = torch.load(artifact / SM.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    meta["train"]["origin"]["updates"] = 99
    meta["train_sha256"] = payload["train_sha256"] = SM.NR._tokenizer_state_sha256(meta["train"])
    torch.save(payload, artifact / SM.WEIGHTS_FILE)
    meta["weights_sha256"] = SM.FT.file_sha256(artifact / SM.WEIGHTS_FILE)
    path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(D, "load_qualified_source", lambda *_a, **_k: pytest.fail("bad certificate opened source rows"))
    with pytest.raises(ValueError, match="inference_training_origin_mismatch"):
        E.evaluate(artifact, fixture["manifest"], tmp_path / "denied", mode="development", splits=["dev"],
                   allocation=fixture["allocation"], identity_index=index)


def test_cuda_argument_reaches_original_loader_without_cpu_fallback(tiny, tmp_path):
    _, artifact = _artifact(tiny, tmp_path, "c5")
    identity = E._model_identity(artifact)
    devices = []
    def profile(frame, event, arg):
        if event == "call" and frame.f_code is SM.NR.load_joint_artifact.__code__:
            devices.append(frame.f_locals["device"])
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        with pytest.raises((RuntimeError, ValueError), match="CUDA|cuda"):
            E._load_scorer(artifact, identity, "cuda")
    finally:
        sys.setprofile(previous)
    assert devices == ["cuda"]


def test_cohort_empty_contributor_keeps_binding_lineage_and_full_members(tiny, tmp_path, monkeypatch):
    first = _cohort_source(tmp_path, "selected", [("a", "a", "ca", "dev", "equivalent")])
    second = _cohort_source(tmp_path, "empty", [("b", "b", "cb", "train", "equivalent")])
    cohort = _write_cohort(tmp_path, [first, second])
    _, artifact = _artifact(tiny, tmp_path, "c5")
    monkeypatch.setitem(E.STATISTICS, "bootstrap_samples", 13)
    out = tmp_path / "partial-cohort"
    result = E.evaluate(artifact, None, out, mode="development", splits=["dev"], cohort=cohort)
    assert result["resolved_roots"] == 1
    meta = D.read_json(out / "metadata.json")
    assert len(meta["binding"]["sources"]) == 2
    assert {item["source_id"]: item["selected_roots"] for item in meta["cohort_sources"]} == {"empty": 0, "selected": 1}
    assert all(item["qualified_lineage"]["manifest_sha256"] for item in meta["cohort_sources"])


def test_cohort_total_empty_rejected_and_single_source_still_rejects_empty(tmp_path):
    sources = [_cohort_source(tmp_path, name, [(name, name, name, "train", "equivalent")]) for name in ("a", "b")]
    cohort = _write_cohort(tmp_path, sources)
    specs = E._source_specs(None, cohort, mode="development")
    with pytest.raises(E.SemanticEvaluationError, match="empty_selected_source"):
        E._cohort_rows(specs, mode="development", splits=["dev"], allow_empty_sources=True)
    with pytest.raises(E.SemanticEvaluationError, match="empty_selected_source"):
        E._source_rows(sources[0]["source_manifest"], mode="development", splits=["dev"],
                       allocation=sources[0]["allocation"], construction_scope=None, run_id=None)


@pytest.mark.parametrize("fault,code", [("root", "duplicate_cohort_root"), ("id", "duplicate_cohort_canonical_id"),
                                         ("cluster", "cohort_cluster_crosses_splits")])
def test_empty_contributor_still_participates_in_duplicate_and_split_checks(tmp_path, fault, code):
    first = _cohort_source(tmp_path, "first", [("a", "root-a", "shared", "dev", "equivalent")])
    second = _cohort_source(tmp_path, "empty", [("a" if fault == "id" else "b", "root-a" if fault == "root" else "root-b",
                                                "shared" if fault == "cluster" else "other", "train", "equivalent")])
    specs = E._source_specs(None, _write_cohort(tmp_path, [first, second]), mode="development")
    with pytest.raises(E.SemanticEvaluationError, match=code):
        E._cohort_rows(specs, mode="development", splits=["dev"], allow_empty_sources=True)
