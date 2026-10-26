"""真实研究模块闭环与根等权反例；合成回执是合同探针，不是独立教师判读。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from research.memory_relations import build_data as B
from research.memory_relations import evaluate as E
from research.memory_relations import features as F
from research.memory_relations import finetune as FT
from research.memory_relations import metrics as M
from research.memory_relations import schema as S
from research.memory_relations import teachers as K
from research.memory_relations import train as T
from research.memory_relations.test_finetune import _tiny_model_dir, _train_pair_rows
from research.memory_relations.test_source_v3 import _row


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("integration contract must not call a network teacher")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(B.M, "call_llm", denied)


def test_v3_spec_frozen_protocol_and_fixed_budget_are_strict():
    directory = Path(B.__file__).parent
    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    protocol = json.loads(B.SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    frozen = directory / "runs" / "v3-coordination" / "source-protocol-frozen.json"
    audit = json.loads((frozen.parent / "preallocation-audit.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == spec["source_v3"]["frozen_protocol_sha256"] \
        == audit["protocol_sha256"] == "d2394abb73c985a7bd2878b327d208040be773a5a5d14646a75b6151cbda5cc6"
    assert spec["schema_fields"] == protocol["row_contract"]["fields"] == list(S.FIELDS)
    assert spec["schema_field_count"] == protocol["row_contract"]["field_count"] == 9
    assert set(spec["splits"]) == set(S.SPLITS) == set(T.SPLITS) == set(FT.SPLITS)
    assert spec["source_v3"]["roots_by_split"] == protocol["targets"]["roots_by_split"] == audit["roots_by_split"]
    for key in ("arms", "seeds", "max_steps", "effective_batch", "patience", "selection_metric",
                "checkpoint_selection", "max_length", "eval_every_steps"):
        assert spec["main_comparison"][key] == audit["main_comparison"][key]
    assert spec["main_comparison"]["dev_weighting"] == "root_equal"
    assert spec["calibration_v3"]["roles"] == list(S.CALIBRATION_ROLES)
    assert spec["calibration_v3"]["primary_unit"] == audit["calibration_primary_unit"] == "pair"
    assert spec["calibration_v3"]["certifies_set_risk"] is False
    assert spec["safety_metrics"]["calibration_target_dangerous_allow"] == M.TARGET_DANGEROUS_ALLOW == .01
    assert spec["safety_metrics"]["calibration_min_coverage"] == M.MIN_COVERAGE == .5
    assert protocol["root_assignments"] == [] and protocol["human_gold_available"] is False


def test_root_ids_use_semantic_root_with_explicit_legacy_group_fallback():
    rows = [{"group_id": "a", "phenomena": {"root_id": "one-root"}},
            {"group_id": "b", "phenomena": {"root_id": "one-root"}},
            {"group_id": "legacy"}]
    assert T._root_ids(rows) == ["one-root", "one-root", "legacy"]
    for invalid in ("", None, [], True):
        rows[0]["phenomena"]["root_id"] = invalid
        with pytest.raises(ValueError, match="root_id"):
            T._root_ids(rows)


def _loss_logits(losses, labels):
    probabilities = np.empty((len(labels), len(F.CLASSES)))
    for index, (loss, label) in enumerate(zip(losses, labels)):
        probability = np.exp(-loss)
        probabilities[index] = (1 - probability) / (len(F.CLASSES) - 1)
        probabilities[index, label] = probability
    return np.log(probabilities)


def test_dev_root_equal_loss_does_not_change_when_one_root_is_repeated(monkeypatch):
    logits = _loss_logits([.3, 1.5], [0, 1])
    monkeypatch.setattr(FT, "logits_for", lambda *args, **kwargs: logits)
    kwargs = {"max_length": 48, "device": "cpu"}
    original = FT.eval_mode_metrics(None, None, [{}, {}], np.asarray([0, 1]),
                                    group_ids=["a", "b"], **kwargs)
    logits = np.concatenate([np.repeat(logits[:1], 9, axis=0), logits[1:]])
    repeated = FT.eval_mode_metrics(None, None, [{}] * 10, np.asarray([0] * 9 + [1]),
                                    group_ids=["a"] * 9 + ["b"], **kwargs)
    assert repeated["loss"] == pytest.approx(original["loss"])
    assert repeated["loss"] == pytest.approx(.9)
    assert repeated["row_loss"] == pytest.approx(.42)
    assert repeated["weighting"] == "root_equal" and repeated["n_groups"] == 2


def test_best_dev_chooses_root_loss_even_when_row_loss_prefers_other_checkpoint(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    directory = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model, _ = FT.build_joint_model(directory)
    rows = _train_pair_rows()
    dev = [copy.deepcopy(rows[0]["request"])] * 10
    labels = np.asarray([0] * 9 + [1])
    states = []

    def dev_logits(model, tokenizer, requests, **kwargs):
        assert requests == dev
        states.append(FT._cpu_state(model))
        losses = ([1.] * 9 + [.001]) if len(states) == 1 else ([.3] * 9 + [1.])
        return _loss_logits(losses, labels)

    monkeypatch.setattr(FT, "logits_for", dev_logits)
    result = FT.train_model(
        "frozen-joint", model, tokenizer, [row["request"] for row in rows], T._labels(rows),
        seed=13, epochs=1, batch_size=4, grad_accum=2, max_length=48,
        device=torch.device("cpu"), bf16=False, max_steps=2, eval_every_steps=1,
        checkpoint_selection="best-dev", selection_metric="dev-loss", patience=0,
        dev_requests=dev, dev_labels=labels, dev_group_ids=["a"] * 9 + ["b"])
    first, second = result["history"]
    assert first["dev_loss"] < second["dev_loss"]
    assert first["dev_row_loss"] > second["dev_row_loss"]
    assert result["selection"]["selected_step"] == 1
    assert result["selection"]["weighting"] == "root_equal"
    assert result["selection"]["secondary_metric"] == "dev_row_loss"
    assert result["budget"]["actual_updates"] == 2
    assert all(torch.equal(value, states[0][name]) for name, value in model.state_dict().items())
    assert any(not torch.equal(states[0][name], states[1][name]) for name in states[0]
               if name.startswith("classifier."))


@pytest.mark.parametrize("groups", [None, [""], ["a", "b"], [1]])
def test_dev_monitor_refuses_missing_or_invalid_roots_before_optimizer(groups, monkeypatch):
    rows = _train_pair_rows()
    monkeypatch.setattr(FT, "build_optimizer", lambda *args, **kwargs: pytest.fail("must not build optimizer"))
    with pytest.raises(ValueError, match="dev_group_ids"):
        FT.train_model(
            "frozen-joint", None, None, [row["request"] for row in rows], T._labels(rows),
            seed=13, epochs=1, batch_size=4, grad_accum=2, max_length=48,
            device=torch.device("cpu"), bf16=False, checkpoint_selection="best-dev",
            dev_requests=[rows[0]["request"]], dev_labels=np.asarray([0]), dev_group_ids=groups)


@pytest.mark.parametrize("all_audit", [False, True])
def test_schema_build_blind_receipts_train_calibrate_loader_adapter_evaluate_without_terminal_reads(tmp_path, monkeypatch, all_audit):
    roots = {f"qa{index}": [f"qa{index}-root{n:02d}" for n in range(20)] for index in range(5)}
    assignments = B.preallocate_author_roots(roots)
    if all_audit:
        for entry in assignments:
            if entry["split"] == "calibration":
                entry["calibration_role"] = "audit"
    protocol = json.loads(B.SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["root_assignments"] = assignments
    protocol_path, shard = tmp_path / "protocol.json", tmp_path / "author-fixture.jsonl"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    rows = [_row(entry, 10000 + index) for index, entry in enumerate(assignments)]
    for index, row in enumerate(rows):
        row["id"] = str(9000000 + index)
        S.validate_row(row)
    shard.write_bytes(B._jsonl_bytes(rows))
    assembled = B.build_author_shards([shard], protocol_path)
    trainable, sealed = tmp_path / "trainable", tmp_path / "sealed"
    B.write_author_outputs(assembled, trainable, sealed)
    assert not assembled["sealed_rows"] and not assembled["quarantine_rows"]
    assert len(assembled["training_rows"]) == 100

    ledger, mapping, blind = tmp_path / "private" / "ledger.jsonl", tmp_path / "private" / "map.json", tmp_path / "blind"
    K.export_blind(trainable / "pairs.jsonl", ledger, blind, mapping, teacher_id="offline-contract-judge",
                   teacher_provider="kimi", teacher_model="fixture-receipt", shard_size=200,
                   synthetic_only=True)
    private = json.loads(mapping.read_text(encoding="utf-8"))
    receipts = []
    for item in private["entries"]:
        source = item["source_row"]
        receipts.append({"blind_id": item["entry"]["blind_id"], "input_sha256": item["entry"]["input_sha256"],
                         "teacher": item["teacher"], "agent_run_id": "synthetic-contract-fixture",
                         "judgement": {"relation": source["label"], "reason": "合成参考回执，只验证接口，不执行独立推理。",
                                       "related_ids": [handle for handle, original in item["reference_map"]["existing"].items()
                                                       if original in source["related_ids"]],
                                       "evidence_refs": [handle for handle, original in item["reference_map"]["evidence"].items()
                                                         if original in source["evidence_refs"]],
                                       "error_code": "model_uncertain" if source["label"] == "uncertain" else None}})
    reply = tmp_path / "receipts.jsonl"
    reply.write_bytes(B._jsonl_bytes(receipts))
    imported = K.import_agent_labels(reply, ledger, blind / private["entries"][0]["shard"], mapping,
                                    agent_run_id="synthetic-contract-fixture")
    assert imported["accepted"] == 100 and imported["api_calls"] == 0 and imported["usage"] is None
    judged = [record["row"] for record in K.load_records(ledger).values()]
    for row in judged:
        S.validate_row(row)
    data = tmp_path / "judged-pairs.jsonl"
    audit_rows = [row for row in judged if row["split"] == "calibration"
                  and row["phenomena"]["calibration_role"] == "audit"]
    audit_path = tmp_path / "isolated-audit.jsonl"
    audit_path.write_bytes(B._jsonl_bytes(audit_rows))
    data.write_bytes(B._jsonl_bytes([row for row in judged if row not in audit_rows]))
    evaluation_rows = [row for row in judged if row["split"] == "dev"]
    evaluation_path = tmp_path / "dev-replay.jsonl"
    evaluation_path.write_bytes(B._jsonl_bytes(evaluation_rows))
    terminal = Path(B.__file__).parent / "data" / "challenge.jsonl"
    sealed_absolute, terminal_absolute, audit_absolute = sealed.resolve(), terminal.resolve(), audit_path.resolve()

    for method in ("open", "read_bytes", "read_text", "is_file", "exists", "stat"):
        original = getattr(Path, method)

        def guarded(path, *args, _original=original, **kwargs):
            resolved = Path(os.path.abspath(str(path)))
            if resolved in (terminal_absolute, sealed_absolute, audit_absolute) or sealed_absolute in resolved.parents:
                pytest.fail(f"audit/terminal probe/hash/read forbidden in candidate none mode: {path.name}")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, guarded)

    common = ["--data", str(data), "--challenge", str(terminal), "--sizes", "75", "--seeds", "13",
              "--eval-splits", "none", "--calibration-protocol", "partitioned", "--bootstrap-samples", "5"]
    char_out = tmp_path / "char"
    if all_audit:
        monkeypatch.setattr(T, "fit_model", lambda *args, **kwargs: pytest.fail("no prob/policy cannot fit"))
        monkeypatch.setattr(FT, "build_joint_model", lambda *args, **kwargs: pytest.fail("cannot build"))
        with pytest.raises(ValueError, match="calibration"):
            T.main(common + ["--out", str(char_out), "--models", "char-logreg", "--threads", "2"])
        with pytest.raises(ValueError, match="calibration"):
            FT.main(common + ["--out", str(tmp_path / "joint"), "--models", "frozen-joint"])
        diagnostic = T.calibrate(np.zeros((len(audit_rows), len(F.CLASSES))), audit_rows, protocol="partitioned")
        assert diagnostic["temperature"] == 1.0 and diagnostic["threshold"] is None
        assert diagnostic["selection"]["temperature_fitted"] is False
        assert not char_out.exists()
        return
    assert T.main(common + ["--out", str(char_out), "--models", "char-logreg", "--threads", "2"]) == 0
    char_metrics = json.loads((char_out / "metrics.json").read_text(encoding="utf-8"))
    assert char_metrics["runs"][0]["calibration"]["protocol"] == "partitioned"
    assert char_metrics["runs"][0]["calibration"]["audit"]["status"] == "deferred"
    assert char_metrics["runs"][0]["calibration"]["audit"]["primary_count"] is None
    assert char_metrics["data"]["challenge"] is None and char_metrics["runs"][0]["eval"] == {}

    audit_texts = {F.request_text(row["request"]) for row in judged if row["split"] == "calibration"
                   and row["phenomena"]["calibration_role"] == "audit"}
    audit_predictions = []
    original_logits = FT.CrossEncoderScorer._logits

    def observed_logits(scorer, requests):
        audit_predictions.extend(F.request_text(request) for request in requests
                                 if F.request_text(request) in audit_texts)
        return original_logits(scorer, requests)

    monkeypatch.setattr(FT.CrossEncoderScorer, "_logits", observed_logits)
    tiny = _tiny_model_dir(tmp_path)
    joint_out = tmp_path / "joint"
    assert FT.main(common + ["--out", str(joint_out), "--models", "frozen-joint", "--device", "cpu",
                            "--embedding-model", str(tiny), "--batch-size", "4", "--grad-accum", "2",
                            "--max-length", "48", "--checkpoint-selection", "best-dev", "--selection-metric", "dev-loss",
                            "--max-steps", "2", "--eval-every-steps", "1", "--patience", "0", "--threads", "2", "--smoke"]) == 0
    joint_metrics = json.loads((joint_out / "metrics.json").read_text(encoding="utf-8"))
    run = joint_metrics["runs"][0]
    assert run["selection"]["weighting"] == "root_equal" and run["selection"]["dev_root_count"] == 10
    assert all("dev_row_loss" in record for record in run["train_history"])
    assert run["budget"]["actual_updates"] == 2 and run["budget"]["actual_sample_instances"] == 16
    assert run["operating_threshold"] is None and run["threshold_feasible"] is False
    assert run["calibration"]["audit"]["status"] == "deferred" and run["eval"] == {}
    assert run["calibration"]["audit"]["primary_count"] is None
    assert len(audit_predictions) == 0
    assert run["calibration"]["metrics_scope"] == "prob_policy_only"
    assert run["calibration"]["nll_rows"] == 10
    assert run["calibration"]["metrics"]["n"] == 10
    assert run["parity"]["ok"] is True

    for directory in (char_out / "scorers" / "char-logreg-size75-seed13",
                      joint_out / "scorers" / "frozen-joint-size75-seed13"):
        scorer = F.load_scorer(directory, device="cpu")
        matrix = scorer.predict_proba([row["request"] for row in evaluation_rows])
        assert matrix.shape == (10, 7) and np.isfinite(matrix).all()
        report_path = tmp_path / (directory.name + ".json")
        args = ["--model-dir", str(directory), "--data", str(evaluation_path), "--out", str(report_path)]
        assert E.main(args) == 2 and not report_path.exists()
        assert E.main(args + ["--diagnostic", "--threshold", "0", "--oracle-aggregation"]) == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["operation"]["calibrated"] is False
        assert report["summary"]["all_attempt"]["n"] == 10
        assert report["summary"]["classifier"]["observed"] is True
        assert report["summary"]["classifier"]["pair_attempts"] == 10
        assert report["summary"]["oracle_aggregation"]["rows"] == 10
        assert set(report["summary"]["refs_by_role"]["gated"]) == {"related_ids", "evidence_refs"}
