"""候选多run与study唯一锁审计；纯合成fixture不是实际研究分布/独立金标。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.memory_relations import evaluate as E
from research.memory_relations import features as F
from research.memory_relations import metrics as M
from research.memory_relations import train as T
from research.memory_relations.test_metrics import partition_case
from research.memory_relations.test_training import _row, _write_jsonl


def _cal_row(index, role, label, *, root=None, primary=True):
    root = root or f"{role}-root-{index}"
    row = _row(str(1000 + index), root, "calibration", label, index)
    row["phenomena"] = {"root_id": root, "calibration_role": role, "operating_primary": primary,
                        "domain": "writing", "family": "synthetic-contract", "author": "fixture",
                        "rationale": "合成统计合同，不是语义正确性证据", "variant": f"v{index}"}
    return row


def _candidate_rows():
    rows = []
    for i, label in enumerate(F.CLASSES):
        rows += [_row(f"train-{i}-{n}", f"train-root-{n}", "train", label, n) for n in range(3)]
        rows += [_row(f"dev-{i}", f"dev-{i}", "dev", label, i)]
    for n in range(3):
        rows += [_row(f"train-extra-{n}-{i}", f"train-root-{n}", "train", "complement", i) for i in range(3)]
    rows += [_cal_row(1, "prob", "complement"), _cal_row(2, "prob", "conflict")]
    rows += [_cal_row(3, "policy", "complement"), _cal_row(4, "policy", "complement")]
    for row in rows:
        if not isinstance(row["phenomena"], dict):
            row["phenomena"] = {}
    return rows


@pytest.fixture
def candidate(tmp_path):
    data, out = tmp_path / "development.jsonl", tmp_path / "candidates"
    _write_jsonl(data, _candidate_rows())
    assert T.main(["--data", str(data), "--out", str(out), "--models", "majority", "--sizes", "21",
                   "--seeds", "13", "--eval-splits", "none", "--calibration-protocol", "partitioned"]) == 0
    model = out / "scorers" / "majority-size21-seed13"
    meta = json.loads((model / "meta.json").read_text(encoding="utf-8"))
    assert meta["calibration"]["stage"] == "candidate"
    assert meta["calibration"]["selection"]["candidate_threshold"] == 0.0
    assert meta["threshold"] is None and meta["threshold_feasible"] is False
    return model, data


def test_candidate_api_does_not_call_audit_and_keeps_top_threshold_null(monkeypatch):
    z, y, trigger, groups, roles, primary = partition_case()
    mask = np.asarray(roles) != "audit"
    monkeypatch.setattr(M, "_audit_at_locked_point", lambda *args, **kwargs: pytest.fail("candidate cannot audit"))
    result = M.calibrate_partitioned(z[mask], y[mask], trigger[mask], np.asarray(groups)[mask],
                                     np.asarray(roles)[mask], primary[mask], stage="candidate")
    assert result["selection"]["candidate_threshold"] == 0.0
    assert result["audit"]["status"] == "deferred" and result["audit"]["primary_count"] is None
    assert result["threshold"] is None and result["threshold_feasible"] is False
    with pytest.raises(ValueError, match="禁止audit"):
        M.calibrate_partitioned(z, y, trigger, groups, roles, primary, stage="candidate")


def test_locked_audit_never_refits_or_selects_grid(monkeypatch):
    z, y, trigger, groups, roles, primary = partition_case(audit_allow=500)
    mask = np.asarray(roles) == "audit"
    monkeypatch.setattr(M, "fit_temperature", lambda *args, **kwargs: pytest.fail("must not refit"))
    monkeypatch.setattr(M, "select_threshold", lambda *args, **kwargs: pytest.fail("must not reselect"))
    result = M.audit_locked_partition(z[mask], y[mask], trigger[mask], np.asarray(groups)[mask],
                                      np.asarray(roles)[mask], primary[mask], temperature=1.7,
                                      candidate_threshold=0.0, point_feasible=True)
    assert result["temperature"] == 1.7 and result["threshold_feasible"] is True
    assert result["audit"]["primary_count"] == 500


def test_calibration_primary_and_roles_are_counted_per_root_not_group_alias():
    rows = [_cal_row(1, "policy", "complement", root="one", primary=True),
            _cal_row(2, "policy", "complement", root="one", primary=False)]
    rows[1]["group_id"] = "different-group-alias"
    roles, primary = T._calibration_roles(rows)
    assert roles == ["policy", "policy"] and primary.tolist() == [True, False]
    rows[1]["phenomena"]["operating_primary"] = True
    with pytest.raises(ValueError, match="恰有一个"):
        T._calibration_roles(rows)
    rows[1]["phenomena"].update(operating_primary=False, calibration_role="audit")
    with pytest.raises(ValueError, match="语义根"):
        T._calibration_roles(rows)


def test_nine_candidates_do_not_receive_audit(candidate, tmp_path, monkeypatch):
    _model, data = candidate
    monkeypatch.setattr(M, "audit_locked_partition", lambda *args, **kwargs: pytest.fail("no audit"))
    assert T.main(["--data", str(data), "--out", str(tmp_path / "nine"), "--models", "majority,char-logreg,minilm-logreg",
                   "--sizes", "21", "--seeds", "13,20260919,424242", "--eval-splits", "none",
                   "--calibration-protocol", "partitioned", "--embedding-model", str(Path(T.__file__).parents[2]
                   / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"), "--threads", "2"]) == 0
    report = json.loads((tmp_path / "nine" / "metrics.json").read_text(encoding="utf-8"))
    assert len(report["runs"]) == 9
    assert all(r["calibration"]["audit"]["status"] == "deferred" for r in report["runs"])
    assert all(r["operating_threshold"] is None for r in report["runs"])


def test_candidate_mixed_audit_file_fails_before_fitting(tmp_path, monkeypatch):
    rows = _candidate_rows() + [_cal_row(5, "audit", "complement")]
    data = tmp_path / "bad-development.jsonl"
    _write_jsonl(data, rows)
    monkeypatch.setattr(T, "fit_model", lambda *args, **kwargs: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="物理分离"):
        T.main(["--data", str(data), "--out", str(tmp_path / "out"), "--models", "majority", "--sizes", "21",
                "--seeds", "13", "--eval-splits", "none", "--calibration-protocol", "partitioned"])
    assert not (tmp_path / "out").exists()


def test_single_study_lock_once_audit_and_real_public_loader_consumer(candidate, tmp_path):
    model, _data = candidate
    lock, out, audit = tmp_path / "study-lock", tmp_path / "audit-out", tmp_path / "audit.jsonl"
    assert T.main(["--mode", "lock", "--model-dir", str(model), "--lock-dir", str(lock), "--study-id", "qa-fixed-study"]) == 0
    with pytest.raises(FileExistsError):
        T.lock_candidate(model, lock, study_id="qa-fixed-study")
    rows = [_cal_row(100 + i, "audit", "complement") for i in range(500)]
    _write_jsonl(audit, rows)
    assert T.main(["--mode", "audit", "--lock-dir", str(lock), "--data", str(audit), "--out", str(out)]) == 0
    result = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert result["threshold"] == 0.0 and result["threshold_feasible"] is True
    assert result["audit"]["primary_count"] == result["audit"]["independent_groups"] == 500
    with pytest.raises(FileExistsError):
        T.audit_candidate(lock, tmp_path / "must-not-open.jsonl", tmp_path / "second-audit")
    scorer = F.load_scorer(out / "scorer", device="cpu")
    assert np.isfinite(scorer.predict_proba([rows[0]["request"]])).all()
    report = tmp_path / "evaluated.json"
    assert E.main(["--model-dir", str(out / "scorer"), "--data", str(audit), "--out", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["operation"]["calibrated"] is True
    assert json.loads((model / "meta.json").read_text())["threshold"] is None


@pytest.mark.parametrize("target", ["artifact", "lock"])
def test_changed_artifact_or_commitment_fails_before_audit_data_read(candidate, tmp_path, monkeypatch, target):
    model, _data = candidate
    lock = tmp_path / "study-lock"
    T.lock_candidate(model, lock, study_id="fixed")
    path = model / "meta.json" if target == "artifact" else lock / "lock.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(T, "load_rows", lambda *args, **kwargs: pytest.fail("must not open audit"))
    with pytest.raises(ValueError, match="改|变动"):
        T.audit_candidate(lock, tmp_path / "sealed-audit.jsonl", tmp_path / "audit-out")


def test_null_policy_candidate_never_opens_audit(candidate, tmp_path, monkeypatch):
    model, _data = candidate
    path = model / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["calibration"]["selection"].update(candidate_threshold=None, point_feasible=False, point=None)
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    lock = tmp_path / "null-lock"
    T.lock_candidate(model, lock, study_id="null-fixed")
    monkeypatch.setattr(T, "load_rows", lambda *args, **kwargs: pytest.fail("null point cannot read audit"))
    result = T.audit_candidate(lock, tmp_path / "sealed-must-not-probe.jsonl", tmp_path / "null-out")
    assert result["audit_data_read"] is False and result["threshold"] is None
    assert result["audit"]["status"] == "no_candidate"


def test_shared_dev_prob_policy_hashes_are_independent_of_arm_train_and_order(tmp_path):
    first, second = _candidate_rows(), copy.deepcopy(_candidate_rows())
    for row in second:
        if row["split"] == "train":
            row["request"]["candidate"]["text"] += "另一训练臂"
    contracts = []
    for index, rows in enumerate((first, list(reversed(second)))):
        path = tmp_path / f"arm-{index}.jsonl"
        _write_jsonl(path, rows)
        args = T.build_parser().parse_args(["--data", str(path), "--eval-splits", "none",
                                           "--calibration-protocol", "partitioned"])
        contracts.append(T._data_contract(T.prepare_data(args)))
    assert contracts[0]["sha256_by_partition"]["train"] != contracts[1]["sha256_by_partition"]["train"]
    for partition in ("dev", "prob", "policy"):
        for field in ("sha256_by_partition", "root_ids_by_partition", "row_counts"):
            assert contracts[0][field][partition] == contracts[1][field][partition]
    assert contracts[0]["audit_visible"] is False
    assert contracts[0]["unit"] == "semantic_root_not_descendant_rows"


def test_audit_descendants_and_group_aliases_do_not_inflate_independent_evidence(candidate, tmp_path):
    model, _data = candidate
    lock, audit = tmp_path / "study-lock", tmp_path / "variants-audit.jsonl"
    T.lock_candidate(model, lock, study_id="variants-fixed")
    rows = [_cal_row(2000 + i, "audit", "complement", root=f"audit-root-{i // 400}", primary=i % 400 == 0)
            for i in range(800)]
    for i, row in enumerate(rows):
        row["group_id"] = f"alias-{i}"
    _write_jsonl(audit, rows)
    result = T.audit_candidate(lock, audit, tmp_path / "variants-out")
    assert result["audit"]["sample_count"] == 800
    assert result["audit"]["primary_count"] == result["audit"]["independent_groups"] == 2
    assert result["audit"]["allow_count"] == 2 and result["threshold"] is None
    assert result["threshold_feasible"] is False


@pytest.mark.parametrize("fault, message", [("overlap", "重叠"), ("role", "audit"), ("primary", "恰有一个")])
def test_failed_audit_attempt_is_consumed_before_loading_or_scoring_again(candidate, tmp_path, monkeypatch, fault, message):
    model, _data = candidate
    lock, audit, out = tmp_path / "study-lock", tmp_path / "bad-audit.jsonl", tmp_path / "bad-out"
    T.lock_candidate(model, lock, study_id="bad-fixed")
    rows = [_cal_row(100, "audit", "complement")]
    if fault == "overlap":
        rows[0]["phenomena"]["root_id"] = "train-root-0"
    elif fault == "role":
        rows[0]["phenomena"]["calibration_role"] = "policy"
    else:
        rows += [_cal_row(101, "audit", "complement", root=rows[0]["phenomena"]["root_id"])]
    _write_jsonl(audit, rows)
    monkeypatch.setattr(F.RelationScorer, "predict_proba_raw", lambda *args, **kwargs: pytest.fail("must not score"))
    with pytest.raises(ValueError, match=message):
        T.audit_candidate(lock, audit, out)
    assert (lock / "audit-attempt.json").is_file() and not out.exists()
    monkeypatch.setattr(T, "load_rows", lambda *args, **kwargs: pytest.fail("failed attempt cannot reopen audit"))
    with pytest.raises(FileExistsError):
        T.audit_candidate(lock, audit, tmp_path / "retry-out")


def test_malformed_candidate_audit_contract_fails_with_value_error(candidate, tmp_path):
    model, _data = candidate
    path = model / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["calibration"]["audit"] = None
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate"):
        T.lock_candidate(model, tmp_path / "bad-lock", study_id="bad-meta")


def test_explicit_root_and_legacy_group_fallback_cannot_leak_across_splits(tmp_path):
    rows = _candidate_rows()
    dev = next(row for row in rows if row["split"] == "dev")
    dev["phenomena"]["root_id"] = rows[0]["group_id"]
    path = tmp_path / "mixed-roots.jsonl"
    _write_jsonl(path, rows)
    args = T.build_parser().parse_args(["--data", str(path), "--eval-splits", "none",
                                       "--calibration-protocol", "partitioned"])
    with pytest.raises(ValueError, match="跨分区重叠"):
        T.prepare_data(args)


def test_real_joint_candidate_plan_lock_null_audit_and_public_consumer(tmp_path, monkeypatch):
    from research.memory_relations import finetune as FT
    from research.memory_relations.test_finetune import _tiny_model_dir

    rows = _candidate_rows()
    for row in rows:
        if row["phenomena"].get("calibration_role") == "policy":
            row["label"] = "conflict"
    data, out = tmp_path / "joint-development.jsonl", tmp_path / "joint-out"
    _write_jsonl(data, rows)
    assert FT.main(["--data", str(data), "--out", str(out), "--models", "finetuned-joint", "--sizes", "21",
                    "--seeds", "13", "--embedding-model", str(_tiny_model_dir(tmp_path)), "--device", "cpu",
                    "--threads", "1", "--batch-size", "8", "--grad-accum", "2", "--max-length", "48",
                    "--max-steps", "2", "--eval-every-steps", "1", "--checkpoint-selection", "best-dev",
                    "--selection-metric", "dev-loss", "--patience", "0", "--eval-splits", "none",
                    "--calibration-protocol", "partitioned", "--calibration-stage", "candidate", "--smoke"]) == 0
    plan = json.loads((out / "planned_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert plan["args"]["calibration_stage"] == report["args"]["calibration_stage"] == "candidate"
    model = out / "scorers" / "finetuned-joint-size21-seed13"
    meta = json.loads((model / "meta.json").read_text(encoding="utf-8"))
    assert plan["training_input_contract"] == report["training_input_contract"] == meta["train"]["training_input_contract"]
    assert meta["train"]["calibration"]["selection"]["candidate_threshold"] is None
    original = T._artifact_binding(model)[1]
    lock = tmp_path / "joint-lock"
    T.lock_candidate(model, lock, study_id="joint-fixed")
    monkeypatch.setattr(T, "load_rows", lambda *args, **kwargs: pytest.fail("null joint cannot open/probe audit"))
    audited = tmp_path / "joint-audited"
    result = T.audit_candidate(lock, tmp_path / "must-not-probe.jsonl", audited)
    assert result["audit_data_read"] is False and result["threshold"] is None
    assert T._artifact_binding(model)[1] == original
    before = F.load_scorer(model).predict_proba([rows[0]["request"]])
    after = F.load_scorer(audited / "scorer").predict_proba([rows[0]["request"]])
    np.testing.assert_allclose(before, after, atol=1e-6)
    assert E.main(["--model-dir", str(audited / "scorer"), "--data", str(data),
                   "--out", str(tmp_path / "invalid-eval.json")]) == 2
    assert not (tmp_path / "invalid-eval.json").exists()
    assert E.main(["--model-dir", str(audited / "scorer"), "--data", str(data),
                   "--out", str(tmp_path / "joint-eval.json"), "--diagnostic", "--threshold", "0"]) == 0
