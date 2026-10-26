                       
"""`evaluate.py` 的正反自测：数据池绑定、manifest/选择唯一入口、真实（tiny）加载器与计分、
七类外部诊断（不产准确率）、原生臂绑定/HANS 归约/oracle 跨度、断点与 resume、截断证据、
GPU 硬闸、崩溃治理与 CLI。

全部离线（autouse 禁 socket/urlopen），不触网、不依赖真实公开数据与正式工件。
默认由 tiny BERT 创建机械工件；显式 BIODATA_SEMANTIC_TINY_ARTIFACTS 可只读复用
source_binding匹配的现成tiny两臂，避免重复fit。视图/选择取冻结study。fixture 记录是合成合同
样例，不是任何真实成绩；正例证明链路可承载，反例证明篡改/省行/子集计分/绑定漂移/设备回退
都会 fail loud。
"""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from research.memory_relations import features as FEAT              
from research.memory_relations import finetune as FT              
from research.memory_relations import nli_reference as NLI              
from research.memory_relations import semantic_artifacts as A              
from research.memory_relations.public_benchmarks import evaluate as E              
from research.memory_relations.public_benchmarks import protocol as P              
from research.memory_relations.public_benchmarks import test_protocol as TP              
from research.memory_relations.public_benchmarks.test_protocol import (              
    make_record, records_for_purpose)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network call in offline evaluate test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)


                                                                             
                                        
                                                                             

def _rows_with_purpose(dataset: str, purpose: str, count: int, *, start: int = 0) -> list:
    """按确定性 cluster 分区构造真实落在指定用途的合法行；`start` 用于与既有行不重叠。"""
    study = P.load_study()
    rows: list = []
    index = start
    while len(rows) < count:
        group, premise, hypothesis = TP._candidate(dataset, index)
        index += 1
        record = make_record(dataset, group=group, premise=premise, hypothesis=hypothesis)
        if P.assign_purpose(study, dataset, TP._cluster_id(record)) == purpose:
            rows.append(record)
    return rows


def _ocnli_rows(diagnostic: int = 3, sealed: int = 1, *, start: int = 0) -> list:
    return (_rows_with_purpose("ocnli", "diagnostic", diagnostic, start=start)
            + _rows_with_purpose("ocnli", "sealed_acceptance", sealed, start=start + 500))


def _write_producer_run(directory: Path, files: dict) -> Path:
    """合成 datasets.py 产出的 run（manifest + normalized/*.jsonl，哈希/计数真实）。"""
    normalized = directory / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    datasets: dict = {}
    total = 0
    for (dataset, split), rows in sorted(files.items()):
        payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                          for row in rows)
        path = normalized / f"{dataset}.{split.replace('/', '_')}.jsonl"
        path.write_bytes(payload.encode("utf-8"))
        total += len(rows)
        datasets.setdefault(dataset, {"splits": {}})
        datasets[dataset]["splits"][split] = {
            "official_split": split,
            "counts": {"rows": len(rows)},
            "normalized": {"path": f"normalized/{path.name}",
                           "bytes": len(payload.encode("utf-8")),
                           "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                           "records": len(rows)},
        }
    manifest = {
        "schema_version": 1,
        "tool": "research/memory_relations/public_benchmarks/datasets.py",
        "run": directory.name, "generated_at_utc": "2026-01-01T00:00:00Z",
        "datasets": datasets, "totals": {"records": total},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                             encoding="utf-8")
    return directory


def _tiny_tokenizer(tmp_path: Path, *, name: str):
    from research.memory_relations.test_finetune import _tiny_model_dir
    from transformers import AutoTokenizer

    directory = _tiny_model_dir(tmp_path, name=name)
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


def _tiny_seven_artifact(tmp_path: Path, *, max_length: int | None = None,
                         name: str = "a") -> Path:
    """真实 tiny 七类工件：tiny BERT + 真实 save_scorer 合同（features.load_scorer 可真实加载）。"""
    from research.memory_relations.test_finetune import _tiny_model_dir
    from transformers import AutoTokenizer

    backbone = _tiny_model_dir(tmp_path, name=f"seven-backbone-{name}")
    model, _ = FT.build_joint_model(backbone)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    if max_length is None:
        max_length = int(P.load_study()["native_reference"]["max_tokens_standard"])
    scorer = FT.CrossEncoderScorer(kind="frozen-joint", model=model, tokenizer=tokenizer,
                                   max_length=max_length, device="cpu",
                                   threshold=None, threshold_feasible=False)
    out = tmp_path / f"seven-artifact-{name}-{max_length}"
    FT.save_scorer(scorer, out, weights="head_only", backbone_dir=backbone,
                   backbone_fingerprint=FEAT.model_fingerprint(backbone), train_meta={})
    return out


def _tiny_native_artifact(tmp_path: Path) -> Path:
    """真实 tiny 原生工件：`nli_reference.load_nli_scorer` 可真实加载的独立 schema 工件。"""
    from research.memory_relations.test_finetune import _tiny_model_dir
    from transformers import AutoTokenizer

    backbone = _tiny_model_dir(tmp_path, name="nli-backbone")
    model, _ = FT.build_joint_model(backbone, num_labels=len(NLI.NLI_CLASSES))
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    max_length = int(P.load_study()["native_reference"]["max_tokens_standard"])
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=max_length, device="cpu")
    out = tmp_path / "nli-artifact"
    NLI.save_nli_artifact(scorer, out, backbone_dir=backbone, train_meta={})
    return out


def _fake_native_artifact(tmp_path: Path, *, max_length: int | None = None,
                          classes: list | None = None) -> Path:
    """只满足注册比对的假原生工件（绑定/归约测试用；权重文件是占位字节）。"""
    directory = tmp_path / "fake-nli"
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / NLI.NLI_WEIGHTS_FILE
    weights.write_bytes(b"placeholder-weights")
    if max_length is None:
        max_length = int(P.load_study()["native_reference"]["max_tokens_standard"])
    meta = {
        "schema": NLI.NLI_SCHEMA, "kind": NLI.NLI_KIND,
        "classes": list(classes or NLI.NLI_CLASSES), "label_space": "nli3",
        "input_recipe": FT.INPUT_RECIPE, "direction": NLI.NLI_PRIMARY_DIRECTION,
        "joint_roles": NLI.NLI_JOINT_ROLES[NLI.NLI_PRIMARY_DIRECTION],
        "max_length": int(max_length),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "backbone_fingerprint": "f" * 32,
    }
    (directory / NLI.NLI_META_FILE).write_text(json.dumps(meta, ensure_ascii=False),
                                               encoding="utf-8")
    return directory


class FakeScorer:
    """按规则的假打分器：classes 可控、max_length 可设、输出真实 (n, k) 概率（测试用）。

    仍暴露 `.model`（微型真实张量）与 `.tokenizer`——加载器把「实际参数设备/真实 tokenizer」
    作为合同要求，mock 也必须满足，避免凭声明伪造设备证明。
    """

    def __init__(self, classes, rule, *, max_length: int, tokenizer, fail_on=()):
        import torch

        self.classes = tuple(classes)
        self.rule = rule
        self.max_length = int(max_length)
        self.tokenizer = tokenizer
        self.model = torch.nn.Linear(1, 1)
        self.direction = NLI.NLI_PRIMARY_DIRECTION
        self.calls: list = []
        self.fail_on = set(fail_on)

    def predict_proba(self, requests):
        items = [requests] if isinstance(requests, dict) else list(requests)
        self.calls.extend(items)
        rows = []
        for request in items:
            text = request["candidate"]["text"]
            if text in self.fail_on:
                raise RuntimeError("boom")
            label = self.rule(request)
            probs = np.zeros(len(self.classes), dtype=np.float64)
            probs[self.classes.index(label)] = 1.0
            rows.append(probs)
        return np.asarray(rows, dtype=np.float64)


def _results(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


                                                                             
     
                                                                             

def test_pool_loads_official_non_train_members_only(tmp_path):
    rows = _ocnli_rows(2, 1)
    train_rows = _rows_with_purpose("ocnli", "diagnostic", 3, start=900)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows,
                                                 ("ocnli", "train.50k"): train_rows})
    pool = E.load_data_run(run)
    assert pool["accounting"]["rows"] == len(rows)
    assert pool["accounting"]["labeled"] == len(rows)
    assert pool["train_excluded_rows"] == len(train_rows)
    assert [item["official_split"] for item in pool["files"]] == ["dev"]
    assert pool["producer_totals"]["records"] == len(rows) + len(train_rows)


def test_pool_binding_fails_loud_on_tamper(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    path = run / "normalized/ocnli.dev.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(E.EvalError, match="normalized_sha_mismatch"):
        E.load_data_run(run)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(E.EvalError, match="missing_normalized_file"):
        E.load_data_run(run)
    path.write_bytes(original)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["datasets"]["ocnli"]["splits"]["dev"]["normalized"]["records"] += 1
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(E.EvalError, match="normalized_count_mismatch"):
        E.load_data_run(run)


def test_frozen_manifest_must_cover_pool_exactly(tmp_path):
    rows = _ocnli_rows(2, 1)
    manifest = P.partition_records(rows)
    frozen = _write(manifest, tmp_path / "frozen.json")
    assert E.build_manifest(rows, frozen_manifest_path=frozen)["units"] == manifest["units"]
    dropped = copy.deepcopy(manifest)
    dropped["diagnostic"] = dropped["diagnostic"][:-1] or dropped["sealed_acceptance"][:-1]
    with pytest.raises((P.ContractError, E.EvalError)):
        E.build_manifest(rows, frozen_manifest_path=_write(dropped, tmp_path / "dropped.json"))
    tampered = copy.deepcopy(manifest)
    target = tampered["sealed_acceptance"][0] if tampered["sealed_acceptance"] \
        else tampered["diagnostic"][0]
    target["hypothesis"] = "tampered hypothesis"
    with pytest.raises((P.ContractError, E.EvalError)):
        E.build_manifest(rows, frozen_manifest_path=_write(tampered, tmp_path / "edited.json"))


def _write(payload: dict, path: Path) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


                                                                             
          
                                                                             

def test_selection_uses_manifest_scope_and_select_for_phase():
    study = P.load_study()
    rows = _ocnli_rows(3, 2)
    manifest = P.partition_records(rows, study)
    diagnostic = E.build_selection(manifest, study=study, view_id="ocnli_diag7_w384",
                                   phase="diagnostic")
    assert diagnostic["binding"] == "select_for_phase+dataset"
    expected = sorted(row["id"] for row in manifest["diagnostic"])
    assert diagnostic["ids"] == expected
    assert diagnostic["accounting"]["rule"] == "all_rows"
    final = E.build_selection(manifest, study=study, view_id="ocnli_diag7_w384",
                              phase="final_acceptance")
    assert len(final["ids"]) == len(rows) and set(final["ids"]) == {row["id"] for row in rows}
    native = E.build_selection(manifest, study=study, view_id="ocnli_nli3_native",
                               phase="diagnostic")
    assert native["binding"] == "manifest_scope"
    assert native["ids"] == expected                                  
    assert native["accounting"]["rule"] == "all_rows"


                                                                             
                     
                                                                             

def _diag_run(tmp_path, *, limit=None, score=True, out_name="diag-out"):
    rows = _ocnli_rows(3, 1)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    result = E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                        scorer_dir=artifact, out_dir=tmp_path / out_name, arm="seven-class",
                        device="cpu", limit=limit, score=score)
    return result, artifact


def test_seven_class_arm_real_loader_report_and_evidence(tmp_path):
    result, artifact = _diag_run(tmp_path)
    selection = json.loads(Path(result["out_dir"], "selection.json").read_text(encoding="utf-8"))
    entries = _results(Path(result["out_dir"], "results.jsonl"))
    assert len(entries) == len(selection["ids"])
    for entry in entries:
        assert entry["error"] is None
        assert set(entry["probs"]) == set(P.DIRECTIONS)
        for direction in P.DIRECTIONS:
            probs = np.asarray(entry["probs"][direction], dtype=np.float64)
            assert probs.shape == (7,)
            assert np.isfinite(probs).all()
            assert abs(probs.sum() - 1.0) <= 1e-9
        assert isinstance(entry["truncated"]["hypothesis_as_candidate"], bool)
    report = result["report"]["view_report"]
    assert report["not_accuracy"] is True and report["task_equivalence"] == "none"
    assert "accuracy" not in report
    assert set(report["by_direction"]) == set(P.DIRECTIONS)
    assert report["directions"]["primary"] == P.PRIMARY_DIRECTION
    assert report["flip_rate"]["n"] == len(entries)
    evidence = result["report"]["runner_evidence"]
    assert evidence["denominators"]["completion"]["complete"] is True
    assert evidence["denominators"]["n_scored"] == len(entries)
    assert evidence["artifact_integrity"]["unchanged"] is True
    assert evidence["device"]["matmul_device"] == "cpu"
    assert evidence["scorer"]["classes"] == list(FEAT.CLASSES)
    assert evidence["scorer"]["window"] == 384
    assert evidence["scorer"]["weights_sha256"] == hashlib.sha256(
        (artifact / FT.WEIGHTS_FILE).read_bytes()).hexdigest()
    assert result["progress"]["infer_complete"] is True
    assert result["progress"]["truncation"]["rows"] == 2 * len(entries)
    assert "by_source" in result["progress"]["truncation"]


def test_seven_class_window_conflict_and_expansion(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    small = _tiny_seven_artifact(tmp_path, max_length=64, name="small")
    with pytest.raises(E.EvalError, match="window_conflict"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=small, out_dir=tmp_path / "conflict", arm="seven-class")
    big = _tiny_seven_artifact(tmp_path, name="big")
    result = E.run_view(data_dir=run, view_id="ocnli_diag7_w512", phase="diagnostic",
                        scorer_dir=big, out_dir=tmp_path / "expanded", arm="seven-class")
    assert result["run"]["scorer"]["window"] == 512
    assert result["run"]["scorer"]["artifact_max_length"] == 384


def test_seven_class_arm_never_scores_native_views(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    with pytest.raises(E.EvalError, match="arm_view_mismatch"):
        E.run_view(data_dir=run, view_id="ocnli_nli3_native", phase="diagnostic",
                   scorer_dir=artifact, out_dir=tmp_path / "wrong-arm", arm="seven-class")
    with pytest.raises(E.EvalError, match="arm_view_mismatch"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=artifact, out_dir=tmp_path / "wrong-arm2", arm="native")


def test_seven_class_failures_are_kept_in_denominators(tmp_path, monkeypatch):
    rows = _ocnli_rows(3, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    failing = rows[0]["hypothesis"]
    original = E.load_seven_class_scorer

    def loader(model_dir, *, view, device):
        handle = original(model_dir, view=view, device=device)
        inner = handle["scorer"].predict_proba

        def broken(requests):
            items = [requests] if isinstance(requests, dict) else list(requests)
            if any(item["candidate"]["text"] == failing for item in items):
                raise RuntimeError("synthetic failure")
            return inner(items)

        handle["scorer"].predict_proba = broken
        return handle

    monkeypatch.setattr(E, "load_seven_class_scorer", loader)
    result = E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                        scorer_dir=artifact, out_dir=tmp_path / "failing", arm="seven-class")
    evidence = result["report"]["runner_evidence"]
    assert evidence["denominators"]["n_failed"] == 1
    assert evidence["denominators"]["n_scored"] == len(rows) - 1
    assert evidence["denominators"]["completion"]["complete"] is False
    assert result["progress"]["n_failed"] == 1
    report = result["report"]["view_report"]
    assert report["n"] == len(rows) - 1                        


                                                                             
                        
                                                                             

def test_native_arm_binding_and_real_loader(tmp_path):
    rows = _ocnli_rows(3, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_native_artifact(tmp_path)
    result = E.run_view(data_dir=run, view_id="ocnli_nli3_native", phase="diagnostic",
                        scorer_dir=artifact, out_dir=tmp_path / "native", arm="auto",
                        device="cpu")
    report = result["report"]["view_report"]
    assert report["view_id"] == "ocnli_nli3_native" and report["n_total"] == len(rows)
    assert report["accuracy"]["denominator"] == len(rows)
    assert set(report["bootstrap"]) == {"row", "group_equal"}
    assert report["completion"]["complete"] is True
    evidence = result["report"]["runner_evidence"]
    assert evidence["scorer"]["classes"] == list(NLI.NLI_CLASSES)
    assert evidence["scorer"]["metadata_check"] == "matched_registered_native_reference"
    assert evidence["scorer"]["artifact_integrity"] == "verified_weights_sha256"


def test_native_arm_rejects_unregistered_artifact(tmp_path):
    view = P.view_spec(P.load_study(), "ocnli_nli3_native")
    wrong_window = _fake_native_artifact(tmp_path, max_length=512)
    with pytest.raises(P.ContractError, match="artifact_window_mismatch"):
        E.load_native_scorer(wrong_window, view=view, device="cpu")
    wrong_classes = _fake_native_artifact(tmp_path, classes=["entailment", "non-entailment"])
    with pytest.raises(P.ContractError):
        E.load_native_scorer(wrong_classes, view=view, device="cpu")
    missing = tmp_path / "missing-artifact"
    missing.mkdir()
    with pytest.raises(E.EvalError, match="artifact_missing"):
        E.load_native_scorer(missing, view=view, device="cpu")


def _premise_index(record: dict) -> int:
    return int(record["premise"].split()[-1].rstrip("."))


def test_hans_reduction_maps_three_class_to_entailment2(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path, name="hans-tokenizer")
    candidates = records_for_purpose("hans", "diagnostic", 8)
    even = [row for row in candidates if _premise_index(row) % 2 == 0][:2]
    odd = [row for row in candidates if _premise_index(row) % 2 == 1][:2]
    assert even and odd, "夹具必须同时覆盖 entailment 与 non-entailment 归约路径"
    rows = even + odd
    run = _write_producer_run(tmp_path / "run", {("hans", "evaluation"): rows})

    def rule(request):
        text = request["existing"][0]["text"]
        index = int(text.split()[-1].rstrip("."))
        return "entailment" if index % 2 == 0 else "neutral"

    def loader(directory, device):
        return FakeScorer(NLI.NLI_CLASSES, rule, max_length=384, tokenizer=tokenizer)

    artifact = _fake_native_artifact(tmp_path)
    result = E.run_view(data_dir=run, view_id="hans_entailment2_native",
                        phase="final_acceptance", scorer_dir=artifact,
                        out_dir=tmp_path / "hans", arm="native",
                        confirm_final_acceptance=True, scorer_loader=loader)
    entries = {entry["record_id"]: entry["pred"]
               for entry in _results(Path(result["out_dir"], "results.jsonl"))}
    expected = {}
    for row in rows:
        index = int(row["premise"].split()[-1].rstrip("."))
        expected[row["id"]] = "entailment" if index % 2 == 0 else "non-entailment"
    assert entries == expected
    report = result["report"]["view_report"]
    assert report["macro_f1"]["classes"] == ["entailment", "non-entailment"]
    assert report["accuracy"]["denominator"] == len(rows)
    assert report["completion"]["complete"] is True
    assert set(report["by_stratum"]) == {"heuristic", "subcase"}                  


def test_diag_arm_missing_gold_and_strata(tmp_path, monkeypatch):
    """缺标行只进分布/覆盖记账、绝不当 AUC 负例；分层按 study stratify_by 只作描述统计。"""
    tokenizer = _tiny_tokenizer(tmp_path, name="diag-tokenizer")
    rows = _rows_with_purpose("hans", "diagnostic", 3, start=700)
    rows = [dict(row, gold_label=None, metadata=dict(row["metadata"], gold_status="missing",
                                                     label_method="no_label"))
            for row in rows]
    for row in rows:
        row["metadata"].pop("original_label", None)
    run = _write_producer_run(tmp_path / "run", {("hans", "evaluation"): rows})
    fake = FakeScorer(FEAT.CLASSES, lambda request: "conflict", max_length=384,
                      tokenizer=tokenizer)
    original = E.load_seven_class_scorer

    def loader(model_dir, *, view, device):
        handle = original(model_dir, view=view, device=device)
        handle["scorer"] = fake
        return handle

    artifact = _tiny_seven_artifact(tmp_path)
    monkeypatch.setattr(E, "load_seven_class_scorer", loader)
    result = E.run_view(data_dir=run, view_id="hans_diag7_w384", phase="diagnostic",
                        scorer_dir=artifact, out_dir=tmp_path / "diag", arm="seven-class")
    report = result["report"]["view_report"]
    assert report["not_accuracy"] is True
    assert report["n"] == len(rows)
    assert report["n_with_gold"] == 0
    assert report["n_missing_gold"] == len(rows)
    primary = report["by_direction"]["hypothesis_as_candidate"]
    assert primary["conflict_signal"]["rank_auc"] is None
    assert primary["conflict_signal"]["n_missing_excluded"] == len(rows)
    assert primary["conflict_signal"]["bootstrap"] is None
    assert set(primary["mean_probability_by_stratum"]) == {"heuristic", "subcase"}
    assert result["report"]["runner_evidence"]["denominators"]["completion"]["complete"] is True


def test_oracle_view_uses_gold_spans_and_accounting(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path, name="oracle-tokenizer")
    span = [{"text": "the parties shall keep information confidential", "start": 0,
             "end": 47}]
    rows = [
        make_record("contractnli", gold="entailment", split="test", group="doc-1",
                    premise="CONTRACT text one. " * 4, hypothesis="CONTRACT hypothesis one.",
                    rid="contractnli:test:1:a", spans=span),
        make_record("contractnli", gold="contradiction", split="test", group="doc-2",
                    premise="CONTRACT text two. " * 4, hypothesis="CONTRACT hypothesis two.",
                    rid="contractnli:test:2:a", spans=span),
        make_record("contractnli", gold="neutral", split="test", group="doc-3",
                    premise="CONTRACT text three. " * 4, hypothesis="CONTRACT hypothesis three.",
                    rid="contractnli:test:3:a", spans=span),
        make_record("contractnli", gold="entailment", split="test", group="doc-4",
                    premise="CONTRACT text four. " * 4, hypothesis="CONTRACT hypothesis four.",
                    rid="contractnli:test:4:a"),
    ]
    run = _write_producer_run(tmp_path / "run", {("contractnli", "test"): rows})
    scorer = FakeScorer(NLI.NLI_CLASSES, lambda request: "entailment", max_length=384,
                        tokenizer=tokenizer)

    artifact = _fake_native_artifact(tmp_path)
    result = E.run_view(data_dir=run, view_id="contractnli_nli3_oracle_evidence",
                        phase="final_acceptance", scorer_dir=artifact,
                        out_dir=tmp_path / "oracle", arm="native",
                        confirm_final_acceptance=True,
                        scorer_loader=lambda directory, device: scorer)
    selection = json.loads(Path(result["out_dir"], "selection.json").read_text(encoding="utf-8"))
    assert selection["accounting"]["rule"] == "gold_evidence_two_class"
    assert selection["accounting"]["n_official"] == 4
    assert selection["accounting"]["n_eligible"] == 2
    assert selection["accounting"]["n_neutral_ineligible"] == 1
    assert selection["accounting"]["n_span_unavailable"] == 1
    assert len(selection["ids"]) == 2
    premises = {request["existing"][0]["text"] for request in scorer.calls}
    assert premises == {span[0]["text"]}                                    
    report = result["report"]["view_report"]
    assert report["layer"] == "oracle_evidence"
    assert report["claim"] == "oracle_evidence"
    assert report["accuracy"]["denominator"] == 2


                                                                             
                    
                                                                             

def test_probe_limit_then_resume_and_incompatible_resume(tmp_path):
    rows = _ocnli_rows(4, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "run-out"
    first = E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                       scorer_dir=artifact, out_dir=out, arm="seven-class", limit=1,
                       score=False)
    assert first["progress"]["n_done"] == 1
    assert first["progress"]["selection_total"] == len(rows)
    with pytest.raises(E.EvalError, match="incomplete_selection"):
        E.score_view(out_dir=out)
    second = E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                        scorer_dir=artifact, out_dir=out, arm="seven-class")
    assert second["progress"]["n_done"] == len(rows)
    assert second["progress"]["infer_complete"] is True
    lines = Path(out, "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(rows)
    assert len({json.loads(line)["record_id"] for line in lines}) == len(rows)
    with pytest.raises(E.EvalError, match="incompatible_resume"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w512", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="seven-class")
    other = _write_producer_run(tmp_path / "run2", {("ocnli", "dev"): rows})
    with pytest.raises(E.EvalError, match="incompatible_resume"):
        E.run_view(data_dir=other, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="seven-class")


def test_results_tamper_refuses_duplicates_and_missing_rows(tmp_path):
    rows = _ocnli_rows(3, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "tamper-out"
    E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
               scorer_dir=artifact, out_dir=out, arm="seven-class")
    path = Path(out, "results.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(E.EvalError, match="duplicate_results"):
        E.score_view(out_dir=out)
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(E.EvalError, match="incomplete_selection"):
        E.score_view(out_dir=out)


def test_final_acceptance_requires_confirmation_and_keeps_report_pure(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_native_artifact(tmp_path)
    out = tmp_path / "final-out"
    with pytest.raises(E.EvalError, match="final_acceptance_not_confirmed"):
        E.run_view(data_dir=run, view_id="ocnli_nli3_native", phase="final_acceptance",
                   scorer_dir=artifact, out_dir=out, arm="native")
    result = E.run_view(data_dir=run, view_id="ocnli_nli3_native", phase="final_acceptance",
                        scorer_dir=artifact, out_dir=out, arm="native",
                        confirm_final_acceptance=True)
    manifest = json.loads(Path(out, "manifest.json").read_text(encoding="utf-8"))
    entries = _results(Path(out, "results.jsonl"))
    results_map = {entry["record_id"]: {key: entry[key]
                                        for key in ("pred", "error", "truncated")}
                   for entry in entries}
    report = result["report"]["view_report"]
    P.assert_final_complete(report, manifest=manifest, results=results_map)


def test_artifact_mutation_detected_at_score_and_resume(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "mut-out"
    E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
               scorer_dir=artifact, out_dir=out, arm="seven-class")
    with open(artifact / FT.WEIGHTS_FILE, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(E.EvalError, match="artifact_mutated"):
        E.score_view(out_dir=out)
    with pytest.raises(E.EvalError, match="incompatible_resume"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="seven-class")


def test_out_dir_must_be_fresh_or_own_run(tmp_path):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(E.EvalError, match="out_dir_not_empty"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="seven-class")


                                                                             
             
                                                                             

def test_truncation_projection_matches_single_source(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path, name="truncation-tokenizer")
    long_row = make_record("ocnli", premise="long premise " * 40,
                           hypothesis="short hypothesis", group="prem-9001",
                           rid="ocnli:dev:prem-9001")
    short_row = make_record("ocnli", group="prem-9002", rid="ocnli:dev:prem-9002")
    requests = [P.diagnostic_request(long_row), P.diagnostic_request(short_row)]
    lengths = [E.row_token_length(tokenizer, request) for request in requests]
    assert lengths[0] > lengths[1]
    window = lengths[1]
    flags = [E.row_truncated(tokenizer, request, window) for request in requests]
    assert flags == [True, False]
    profile = FT.truncation_profile(tokenizer, requests, window)
    assert sum(flags) == profile["truncated_rows"] == 1
    tracker = E.TruncationTracker(max_length=window)
    tracker.add(tokenizer, requests, lengths)
    summary = tracker.summary()
    assert summary["truncated_rows"] == 1
    assert summary["rows"] == 2
    assert summary["attribution"] in ("offset_mapping", "token_prefix_boundary")
    assert summary["by_source"], "角色级覆盖必须来自 truncation_profile"
    assert set(summary["by_source"]) <= {"candidate", "existing"}


def test_device_gate_never_falls_back_to_cpu(tmp_path):
    import torch

    assert E.resolve_device("cpu") == "cpu"
    assert E.device_proof("cpu")["matmul_device"] == "cpu"
    with pytest.raises(E.EvalError, match="invalid_device"):
        E.resolve_device("tpu")
    with pytest.raises(E.EvalError, match="device_unavailable"):
        E.resolve_device("cuda:99999")
    if not torch.cuda.is_available():
        with pytest.raises(E.EvalError, match="device_unavailable"):
            E.resolve_device("cuda")
        rows = _ocnli_rows(1, 0)
        run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
        artifact = _tiny_seven_artifact(tmp_path)
        with pytest.raises(E.EvalError, match="device_unavailable"):
            E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                       scorer_dir=artifact, out_dir=tmp_path / "gpu", arm="seven-class",
                       device="cuda")


def test_resume_rejects_device_change_when_gpu_available(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("需要可用 GPU 才能构造设备变更 resume 反例")
    rows = _ocnli_rows(1, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "device-out"
    E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
               scorer_dir=artifact, out_dir=out, arm="seven-class", device="cpu")
    with pytest.raises(E.EvalError, match="incompatible_resume"):
        E.run_view(data_dir=run, view_id="ocnli_diag7_w384", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="seven-class", device="cuda")


                                                                             
     
                                                                             

def test_cli_run_score_status(tmp_path, capsys):
    rows = _ocnli_rows(2, 0)
    run = _write_producer_run(tmp_path / "run", {("ocnli", "dev"): rows})
    artifact = _tiny_seven_artifact(tmp_path)
    out = tmp_path / "cli-out"
    code = E.main(["run", "--data", str(run), "--view", "ocnli_diag7_w384",
                   "--phase", "diagnostic", "--scorer", str(artifact), "--out", str(out),
                   "--arm", "seven-class"])
    assert code == 0, capsys.readouterr()
    capsys.readouterr()                   
    code = E.main(["status", "--out", str(out)])
    assert code == 0
    status = json.loads(capsys.readouterr().out)
    assert status["view_id"] == "ocnli_diag7_w384"
    assert status["progress"]["infer_complete"] is True
    assert status["has_report"] is True
    code = E.main(["score", "--out", str(out)])
    assert code == 0
    code = E.main(["run", "--data", str(run), "--view", "no_such_view",
                   "--phase", "diagnostic", "--scorer", str(artifact),
                   "--out", str(tmp_path / "cli-bad")])
    assert code == 1
    assert "ERROR" in capsys.readouterr().err


@pytest.fixture
def bound_native_run(tmp_path):
    rows = _ocnli_rows(3, 1)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): rows})
    artifact = _tiny_native_artifact(tmp_path)
    out = tmp_path / "bound-out"
    E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
               scorer_dir=artifact, out_dir=out, score=False)
    return data, artifact, out


def test_score_rejects_coherent_subset(bound_native_run):
    _, _, out = bound_native_run
    entries = _results(out / "results.jsonl")[:1]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["diagnostic"] if row["id"] == entries[0]["record_id"]]
    subset = P.partition_records(rows)
    selection = E.build_selection(subset, view_id="ocnli_nli3_native", phase="diagnostic")
    _write(subset, out / "manifest.json")
    _write(selection, out / "selection.json")
    (out / "results.jsonl").write_text(E._results_line(entries[0]) + "\n", encoding="utf-8")
    with pytest.raises(E.EvalError, match="incompatible_resume|manifest_pool_mismatch"):
        E.score_view(out_dir=out)


@pytest.mark.parametrize("drift", ["study", "source", "manifest", "selection", "scorer"])
def test_score_rechecks_run_binding(bound_native_run, drift):
    data, _, out = bound_native_run
    if drift == "study":
        path = out / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["study_sha256"] = "0" * 64
        _write(run, path)
    elif drift == "source":
        with open(data / "normalized/ocnli.dev.jsonl", "ab") as handle:
            handle.write(b" ")
    elif drift in ("manifest", "selection"):
        with open(out / f"{drift}.json", "ab") as handle:
            handle.write(b" ")
    else:
        path = out / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["scorer"]["classes"] = list(FEAT.CLASSES)
        _write(run, path)
    with pytest.raises(E.EvalError, match="incompatible_resume|normalized_sha_mismatch|scorer"):
        E.score_view(out_dir=out)


def test_status_rederives_completion_and_stale_report(bound_native_run):
    _, _, out = bound_native_run
    E.score_view(out_dir=out)
    path = out / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    progress.update(n_done=999, n_scored=999, infer_complete=True)
    _write(progress, out / "progress.json")
    before = {item.name: item.read_bytes() for item in out.iterdir() if item.is_file()}
    status = E.status_view(out_dir=out)
    assert status["progress"]["n_done"] == len(lines) - 1
    assert status["progress"]["n_unrecorded"] == 1
    assert status["progress"]["infer_complete"] is False
    assert status["progress"]["completion"]["complete"] is False
    assert status["has_report"] is True and status["report_current"] is False
    after = {item.name: item.read_bytes() for item in out.iterdir() if item.is_file()}
    assert after == before


def test_resume_checks_supplied_manifest(bound_native_run, tmp_path):
    data, artifact, out = bound_native_run
    stored = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    subset = P.partition_records(stored["diagnostic"][:1])
    frozen = _write(subset, tmp_path / "subset.json")
    with pytest.raises(E.EvalError, match="manifest_pool_mismatch|incompatible_resume"):
        E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, manifest_path=frozen, score=False)


def test_resume_keeps_all_truncation_evidence(tmp_path):
    result, artifact = _diag_run(tmp_path, limit=1, score=False)
    out = Path(result["out_dir"])
    assert E.status_view(out_dir=out)["progress"]["infer_complete"] is False
    resumed = E.run_view(data_dir=tmp_path / "run", view_id="ocnli_diag7_w384",
                         phase="diagnostic", scorer_dir=artifact, out_dir=out)
    assert resumed["progress"]["truncation"]["rows"] == 2 * 3
    again = E.run_view(data_dir=tmp_path / "run", view_id="ocnli_diag7_w384",
                       phase="diagnostic", scorer_dir=artifact, out_dir=out)
    assert again["progress"]["truncation"] == resumed["progress"]["truncation"]


def test_oracle_reduction_uses_registered_label_names():
    view = P.view_spec(P.load_study(), "contractnli_nli3_oracle_evidence")
    scorer = type("Scorer", (), {"classes": ("contradiction", "entailment", "neutral")})()
    assert E._native_prediction(np.array([0.3, 0.1, 0.6]), scorer, view) == "contradiction"
    assert E._native_prediction(np.array([0.1, 0.3, 0.6]), scorer, view) == "entailment"


def test_full_document_and_oracle_keep_separate_inputs_and_claims(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path, name="layers-tokenizer")
    span = [{"text": "the parties shall keep information confidential"}]
    rows = [make_record("contractnli", gold="entailment", split="test", group="doc-long",
                        premise="CONTRACT text " * 400, hypothesis="CONTRACT hypothesis.",
                        rid="contractnli:test:long", spans=span)]
    data = _write_producer_run(tmp_path / "data", {("contractnli", "test"): rows})
    artifact = _fake_native_artifact(tmp_path)
    calls = {}
    for view_id in ("contractnli_nli3_full_document", "contractnli_nli3_oracle_evidence"):
        scorer = FakeScorer(NLI.NLI_CLASSES, lambda request: "entailment", max_length=384,
                            tokenizer=tokenizer)
        result = E.run_view(data_dir=data, view_id=view_id, phase="final_acceptance",
                            scorer_dir=artifact, out_dir=tmp_path / view_id,
                            confirm_final_acceptance=True,
                            scorer_loader=lambda directory, device: scorer)
        report = result["report"]["view_report"]
        calls[report["layer"]] = scorer.calls
        if report["layer"] == "full_document":
            assert report["claim"] == "windowed_not_full"
            assert report["truncation"]["count"] == 1
        else:
            assert report["claim"] == "oracle_evidence"
            assert report["truncation"]["count"] == 0
    assert calls["full_document"][0]["existing"][0]["text"] == rows[0]["premise"]
    assert calls["oracle_evidence"][0]["existing"][0]["text"] == span[0]["text"]


def test_score_rejects_modified_study_object(bound_native_run):
    _, _, out = bound_native_run
    study = P.load_study()
    study["metrics"]["bootstrap"]["samples"] += 1
    with pytest.raises(E.EvalError, match="study_drift"):
        E.score_view(out_dir=out, study=study)


def test_native_gpu_channel_rejects_cpu_parameters_without_gpu_work(tmp_path, monkeypatch):
    rows = _ocnli_rows(1, 0)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): rows})
    artifact = _fake_native_artifact(tmp_path)
    tokenizer = _tiny_tokenizer(tmp_path, name="device-tokenizer")
    cpu = FakeScorer(NLI.NLI_CLASSES, lambda request: "entailment", max_length=384,
                     tokenizer=tokenizer)
    monkeypatch.setattr(E, "resolve_device", lambda device: "cuda:0")

    def proof_must_not_run(device):
        pytest.fail("CPU model passed the requested GPU channel")

    monkeypatch.setattr(E, "device_proof", proof_must_not_run)
    with pytest.raises(E.EvalError, match="scorer_device_mismatch"):
        E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                   scorer_dir=artifact, out_dir=tmp_path / "wrong-device", device="cuda:0",
                   scorer_loader=lambda directory, device: cpu)
    assert not (tmp_path / "wrong-device").exists()


def test_native_failures_and_missing_gold_keep_full_denominator(tmp_path):
    rows = _ocnli_rows(3, 0)
    rows[0]["gold_label"] = None
    rows[0]["metadata"].update(gold_status="missing", label_method="no_label")
    rows[0]["metadata"].pop("original_label", None)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): rows})
    artifact = _fake_native_artifact(tmp_path)
    tokenizer = _tiny_tokenizer(tmp_path, name="failures-tokenizer")
    scorer = FakeScorer(NLI.NLI_CLASSES, lambda request: "entailment", max_length=384,
                        tokenizer=tokenizer, fail_on=[rows[1]["hypothesis"]])
    out = tmp_path / "fail-out"
    result = E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                        scorer_dir=artifact, out_dir=out,
                        scorer_loader=lambda directory, device: scorer)
    report = result["report"]["view_report"]
    assert report["n_total"] == 3
    assert report["counts"]["without_gold"] == 1
    assert report["counts"]["failed"] == 1
    assert report["accuracy"]["denominator"] == 1
    assert report["coverage"]["denominator"] == 2
    assert report["failure_rate"]["denominator"] == 3
    assert report["completion"]["complete"] is False
    assert E.status_view(out_dir=out)["progress"]["completion"]["complete"] is False


def test_status_rejects_cross_task_prediction(bound_native_run):
    _, _, out = bound_native_run
    entries = _results(out / "results.jsonl")
    entries[0]["pred"] = "conflict"
    (out / "results.jsonl").write_text(
        "".join(E._results_line(entry) + "\n" for entry in entries), encoding="utf-8")
    with pytest.raises(E.EvalError, match="cross_task_prediction"):
        E.status_view(out_dir=out)


@pytest.fixture(scope="module")
def semantic_artifacts(tmp_path_factory):
    import os
    import torch
    from research.memory_relations import semantic_model as SM
    from research.memory_relations import semantic_artifacts as A
    from research.memory_relations.test_semantic_artifacts import make_export

    source = os.environ.get("BIODATA_SEMANTIC_TINY_ARTIFACTS")
    if source:
        paths = {arm: Path(source) / f"{arm}-qualified-2" for arm in SM.ARM_CLASSES}
        for path in paths.values():
            assert A.read_metadata(path)["schema"] == A.SCHEMA
        return paths
    torch.set_num_threads(1)
    base = tmp_path_factory.mktemp("semantic-artifacts")
    return {arm: make_export(base / arm, arm)["artifact"] for arm in SM.ARM_CLASSES}


def _semantic_rows(dataset, count):
    rows, index = [], 0
    while len(rows) < count:
        suffix = " ".join(str(index))
        row = make_record(dataset, premise=f"旧命题甲 {suffix}", hypothesis=f"新命题乙 {suffix}",
                          group=None if dataset == "anli" else f"synthetic-{index}",
                          rid=f"{dataset}:dev:synthetic-{index}")
        index += 1
        if P.partition_records([row])["diagnostic"]:
            rows.append(row)
    return rows


def test_semantic_parent_sum_and_forward_only(semantic_artifacts, monkeypatch):
    from research.memory_relations import semantic_model as SM

    pair = {"old": {"kind": "proposition", "text": "旧命题甲"},
            "new": {"kind": "proposition", "text": "新命题乙"}}
    scorer, _, _ = A.load_inference_artifact(semantic_artifacts["c5"])
    monkeypatch.setattr(scorer, "_predict_views", lambda views: np.tile([.3, .3, .35, .02, .03], (len(views), 1)))
    outcome = E.predict_requests(scorer, [pair], predict=scorer.predict_nli_proba,
                                 output_classes=SM.SR.NLI_CLASSES)[0][0]
    np.testing.assert_allclose(outcome, [.6, .38, .02])
    view = P.view_spec(P.load_study(), "ocnli_nli3_native")
    assert E._native_prediction(outcome, scorer, view, output_classes=SM.SR.NLI_CLASSES) == "entailment"
    monkeypatch.setattr(scorer, "_predict_views", lambda views: np.tile([0., 0., 1., 0., 0.], (len(views), 1)))
    np.testing.assert_array_equal(scorer.predict_nli_proba([pair]), [[0., 1., 0.]])
    dual, _, _ = A.load_inference_artifact(semantic_artifacts["dual3"])
    monkeypatch.setattr(dual, "_predict_views", lambda views: np.tile([.1, .6, .3], (len(views), 1)))
    monkeypatch.setattr(dual, "predict_both_proba", lambda *args, **kwargs: pytest.fail("directions averaged"))
    np.testing.assert_array_equal(dual.predict_nli_proba([pair]), [[.1, .6, .3]])
    assert len(scorer.classes) == 5 and len(dual.classes) == 3


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_semantic_real_run_score_status_resume(tmp_path, semantic_artifacts, arm):
    rows = _semantic_rows("ocnli", 3)
    rows[0]["gold_label"] = None
    rows[0]["metadata"].update(gold_status="missing", label_method="no_label")
    rows[0]["metadata"].pop("original_label", None)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): rows})
    out = tmp_path / arm
    artifact = semantic_artifacts[arm]
    first = E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                       scorer_dir=artifact, out_dir=out, arm="semantic", limit=1, score=False)
    assert first["progress"]["n_unrecorded"] == 2
    with pytest.raises((E.EvalError, P.ContractError), match="missing_results|incomplete_selection"):
        E.score_view(out_dir=out)
    result = E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                        scorer_dir=artifact, out_dir=out, arm="semantic")
    assert result["run"]["scorer"]["schema"] == A.SCHEMA
    assert len(result["run"]["scorer"]["classes"]) == (5 if arm == "c5" else 3)
    assert result["run"]["scorer"]["output_classes"] == list(NLI.NLI_CLASSES)
    assert result["report"]["view_report"]["n_total"] == 3
    assert result["report"]["view_report"]["counts"]["without_gold"] == 1
    assert result["report"]["view_report"]["accuracy"]["denominator"] == 2
    assert result["progress"]["truncation"]["rows"] == 3
    assert E.status_view(out_dir=out)["report_current"] is True
    entries = _results(out / "results.jsonl")
    assert len(entries) == 3 and all(len(entry["p3"]) == 3 for entry in entries)
    with pytest.raises(E.EvalError):
        E.project_native_artifact(artifact)
    with pytest.raises(E.EvalError, match="incompatible_resume"):
        E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                   scorer_dir=artifact, out_dir=out, arm="native")
    tampered = json.loads((out / "run.json").read_text(encoding="utf-8"))
    tampered["scorer"]["projection"] = "fine_argmax_then_map"
    _write(tampered, out / "run.json")
    with pytest.raises(E.EvalError, match="scorer_binding_mismatch"):
        E.score_view(out_dir=out)


@pytest.mark.parametrize("view_id", ["ocnli_nli3_native", "anli_nli3_native", "hans_entailment2_native",
                                      "contractnli_nli3_full_document", "contractnli_nli3_oracle_evidence"])
def test_semantic_existing_five_views(tmp_path, semantic_artifacts, monkeypatch, view_id):
    from research.memory_relations import semantic_model as SM

    dataset = P.view_spec(P.load_study(), view_id)["dataset"]
    rows = _semantic_rows(dataset, 2)
    if dataset == "contractnli":
        for row in rows:
            row["metadata"]["spans"] = [{"text": "旧命题甲"}]
    data = _write_producer_run(tmp_path / "data", {(dataset, "dev"): rows})
    calls = []
    original = SM.SemanticScorer.predict_nli_proba

    def constant(self, pairs, **kwargs):
        calls.extend(pairs if isinstance(pairs, list) else [pairs])
        return np.tile([.2, .7, .1], (len(pairs) if isinstance(pairs, list) else 1, 1))

    monkeypatch.setattr(SM.SemanticScorer, "predict_nli_proba", constant)
    result = E.run_view(data_dir=data, view_id=view_id, phase="diagnostic",
                        scorer_dir=semantic_artifacts["c5"], out_dir=tmp_path / "out", arm="semantic")
    report = result["report"]["view_report"]
    assert report["n_total"] == len(result["run"]["data"]["pool"]["by_dataset"]) * 2
    if dataset == "hans":
        assert {entry["pred"] for entry in _results(tmp_path / "out/results.jsonl")} == {"non-entailment"}
    if "oracle" in view_id:
        assert report["claim"] == "oracle_evidence"
        assert {pair["old"]["text"] for pair in calls} == {"旧命题甲"}
    assert all(set(pair) == {"old", "new", "frame"} for pair in calls)
    assert all(pair["frame"] == {"background": ""} for pair in calls)
    monkeypatch.setattr(SM.SemanticScorer, "predict_nli_proba", original)


def test_semantic_long_same_visible_input_and_actual_profile(tmp_path, semantic_artifacts, monkeypatch):
    from research.memory_relations import semantic_model as SM
    view = P.view_spec(P.load_study(), "contractnli_nli3_full_document")
    handles = [E.load_semantic_scorer(semantic_artifacts[arm], view=view, device="cpu") for arm in ("c5", "dual3")]
    row = make_record("contractnli", premise="旧命题甲" * 2500, hypothesis="新命题乙", group="long-document")
    pairs = [E.semantic_pair(row, view)]
    prepared = [handle["scorer"].prepare_inputs(pairs)[0] for handle in handles]
    assert len(pairs[0]["old"]["text"]) > 8192
    assert prepared[0] == prepared[1]
    for item in prepared:
        assert item["complete"] is False and max(item["full_visible_token_lengths"]) > 64
        assert item["recipe"] == SM.SR.SEMANTIC_INPUT_RECIPE
        for direction in item["directions"].values():
            assert direction["token_length"] <= 64
            for field in direction["fields"]:
                assert "value_type" in field and "omitted_ranges" in field and "content_token_indices" in field
                for segment in field["rendered_segments"]:
                    a, b = segment["source_range"]
                    c, d = segment["rendered_range"]
                    assert item["raw_text"][a:b] == direction["text"][c:d]
        assert item["directions"]["forward"]["visible_information_sha256"] == item["directions"]["reverse"]["visible_information_sha256"]
        assert all(direction["role_body_covered"] == {"old": True, "new": True} for direction in item["directions"].values())
    monkeypatch.setattr(FT, "joint_text", lambda *args, **kwargs: pytest.fail("legacy semantic serializer"))
    monkeypatch.setattr(FT, "truncation_profile", lambda *args, **kwargs: pytest.fail("legacy semantic profile"))
    entries, audits, lengths = E._infer_entries(arm="semantic", view=view, scorer=handles[0]["scorer"],
                                               rows=[row], tokenizer=handles[0]["scorer"].tokenizer, max_length=64)
    assert entries[0]["truncated"] is True and entries[0]["error"] is None
    tracker = E.TruncationTracker(max_length=64)
    tracker.add(handles[0]["scorer"].tokenizer, audits, lengths, semantic=True)
    assert tracker.summary()["truncated_rows"] == 1
    manifest = P.partition_records([row])
    phase = "diagnostic" if manifest["diagnostic"] else "final_acceptance"
    mapped = {row["id"]: {key: entries[0][key] for key in ("pred", "error", "truncated")}}
    assert P.evaluate_view(None, mapped, view_id=view["view_id"], phase=phase, manifest=manifest)["claim"] == "windowed_not_full"


@pytest.mark.parametrize("fault", ["classes", "source", "limits", "config", "tokenizer", "weights"])
def test_semantic_artifact_drift_refused(tmp_path, semantic_artifacts, fault):
    import shutil
    from research.memory_relations import semantic_model as SM

    artifact = tmp_path / "artifact"
    shutil.copytree(semantic_artifacts["c5"], artifact)
    meta = json.loads((artifact / SM.META_FILE).read_text(encoding="utf-8"))
    if fault == "classes":
        meta["classes"] = meta["classes"][::-1]
    elif fault == "source":
        meta["bindings"]["source_sha256"]["semantic_model"] = "0" * 64
    elif fault == "limits":
        meta["input_limits"]["max_text_chars"] = 8192
    elif fault in ("config", "tokenizer"):
        path = next((artifact / fault).glob("*.json"))
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = artifact / SM.WEIGHTS_FILE
        path.write_bytes(path.read_bytes() + b" ")
    _write(meta, artifact / SM.META_FILE)
    with pytest.raises(E.EvalError, match="scorer_binding_mismatch|artifact_hash_mismatch|semantic_artifact_invalid"):
        E.load_semantic_scorer(artifact, view=P.view_spec(P.load_study(), "ocnli_nli3_native"), device="cpu")


def test_semantic_failure_and_input_drift_kept(tmp_path, semantic_artifacts, monkeypatch):
    from research.memory_relations import semantic_model as SM

    rows = _semantic_rows("ocnli", 3)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): rows})
    original = SM.SemanticScorer.predict_nli_proba

    def fail_one(self, pairs, **kwargs):
        items = [pairs] if isinstance(pairs, dict) else pairs
        if any(pair["new"]["text"] == rows[0]["hypothesis"] for pair in items):
            raise RuntimeError("synthetic failure")
        return original(self, pairs, **kwargs)

    monkeypatch.setattr(SM.SemanticScorer, "predict_nli_proba", fail_one)
    out = tmp_path / "out"
    result = E.run_view(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic",
                        scorer_dir=semantic_artifacts["dual3"], out_dir=out, arm="semantic")
    assert result["report"]["view_report"]["n_total"] == 3
    assert result["report"]["view_report"]["counts"]["failed"] == 1
    assert result["progress"]["completion"]["complete"] is False
    entries = _results(out / "results.jsonl")
    entries[1]["input"]["raw_pair_sha256"] = "0" * 64
    (out / "results.jsonl").write_text("".join(E._results_line(entry) + "\n" for entry in entries), encoding="utf-8")
    with pytest.raises(E.EvalError, match="input_drift"):
        E.score_view(out_dir=out)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_inference_export_public_real_subprocess_run_resume_score_status(tmp_path, semantic_artifacts, arm):
    import os
    import subprocess
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): _semantic_rows("ocnli", 3)})
    out = tmp_path / "cli-run"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))), PYTHONSAFEPATH="1",
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def cli(*arguments):
        result = subprocess.run([sys.executable, "-B", "-m", "research.memory_relations.public_benchmarks.evaluate", *map(str, arguments)],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    common = ("--data", data, "--view", "ocnli_nli3_native", "--phase", "diagnostic", "--scorer", semantic_artifacts[arm],
              "--out", out, "--arm", "semantic", "--device", "cpu")
    cli("run", *common, "--limit", "1", "--no-score")
    partial = json.loads(cli("status", "--out", out))
    assert partial["progress"]["n_done"] == 1
    cli("run", *common, "--no-score")
    cli("score", "--out", out)
    final = json.loads(cli("status", "--out", out))
    assert final["report_current"] is True and final["progress"]["n_done"] == 3
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["scorer"]["schema"] == A.SCHEMA


@pytest.mark.parametrize("part", ["tokenizer", "config"])
def test_semantic_score_status_resume_share_export_asset_policy(tmp_path, semantic_artifacts, part):
    import shutil
    artifact = tmp_path / "export"
    shutil.copytree(semantic_artifacts["c5"], artifact)
    data = _write_producer_run(tmp_path / "data", {("ocnli", "dev"): _semantic_rows("ocnli", 2)})
    out = tmp_path / "run"
    arguments = dict(data_dir=data, view_id="ocnli_nli3_native", phase="diagnostic", scorer_dir=artifact, out_dir=out, arm="semantic")
    E.run_view(**arguments)
    path = next((artifact / part).glob("*.json"))
    path.write_bytes(path.read_bytes() + b" ")
    for action in (lambda: E.run_view(**arguments), lambda: E.score_view(out_dir=out), lambda: E.status_view(out_dir=out)):
        with pytest.raises(E.EvalError, match="semantic_artifact_invalid"):
            action()
