"""finetune.py 的研究专项测试：自建 tiny BERT（含 fast tokenizer）+ CPU 前向/后向/端到端。

跑法：pytest research/memory_relations/test_finetune.py -q（需 PYTHONPATH=src;仓库根）。
真实 MiniLM 只用一条 gated 前向/冻结/offset 归属测试（可见时），不做 GPU 训练；不可见则 skip。
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from research.memory_relations import features as FEAT
from research.memory_relations import finetune as TRAIN
from research.memory_relations import metrics as EM

_CLASS_TEXTS = {
    "complement": ("附件加一页核对表。", "先做校正，再做汇总。"),
    "conflict": ("汇总必须两行。", "汇总必须只有一行。"),
    "contextual": ("小样本模式下写两行；常规仍一行。", "汇总写一行。"),
    "duplicate": ("先做校正，再做汇总。", "做校正，然后汇总。"),
    "supersede": ("汇总永久改两行，不再一行。", "汇总写一行。"),
    "temporary": ("本轮先写两行，下轮恢复。", "汇总写一行。"),
    "uncertain": ("有时两行有时一行。", "汇总写一行。"),
}


def _build_request(candidate_text: str, trigger: str, base_text: str, index: int,
                   *, empty_existing: bool = False) -> dict:
    quote = f"出处{index}：{base_text}"
    existing = [] if empty_existing else [
        {"id": "r0", "text": base_text, "trigger": "汇总时", "tier": "explicit", "evidence": []}]
    return {
        "candidate": {"text": candidate_text, "trigger": trigger, "tier": "implicit",
                      "permanence": "persistent", "scope": "project", "evidence": [quote]},
        "existing": existing,
        "evidence": [{"id": "e0", "text": quote}],
        "context": {"scope": "project"},
    }


def _row(row_id: str, group_id: str, split: "str | None", label: str, index: int, *,
         empty_existing: bool = False, label_origin: str = "construction") -> dict:
    candidate_text, base_text = _CLASS_TEXTS[label]
    row = {"id": row_id, "group_id": group_id, "label": label, "label_origin": label_origin,
           "related_ids": [], "evidence_refs": [], "phenomena": {},
           "request": _build_request(f"{candidate_text}（版本{index}）",
                                     "汇总时" if index % 2 else "", base_text, index,
                                     empty_existing=empty_existing)}
    if split is not None:
        row["split"] = split
    return row


def _multi_rule_row(row_id: str, group_id: str) -> dict:
    row = _row(row_id, group_id, "train", "duplicate", 0)
    row["request"]["existing"].append({"id": "r1", "text": "汇总写一行。", "trigger": "汇总时",
                                       "tier": "explicit", "evidence": []})
    return row


def _set_row(row_id: str, group_id: str) -> dict:
    return {"id": row_id, "group_id": group_id, "split": "set", "label": "duplicate",
            "label_origin": "construction", "related_ids": [], "evidence_refs": [],
            "phenomena": {}, "request": {"candidate": {"text": "集合样例"}, "existing": [],
                                          "evidence": []}}


def _fixture() -> "tuple[list[dict], list[dict]]":
    data_rows: list[dict] = []
    challenge_rows: list[dict] = []
    for label in sorted(_CLASS_TEXTS):
        for index in range(3):
            data_rows.append(_row(f"train-{label}-{index}", f"train-g-{label}", "train",
                                  label, index))
        for index in range(2):
            data_rows.append(_row(f"cal-{label}-{index}", f"cal-g-{label}", "calibration",
                                  label, index))
        data_rows.append(_row(f"heldout-{label}", f"heldout-g-{label}", "heldout", label, 0))
        for index in range(2):
            challenge_rows.append(_row(f"challenge-{label}-{index}", f"challenge-g-{label}",
                                       None, label, index, label_origin="agent_review"))
    data_rows.append(_row("train-empty", "train-g-empty", "train", "duplicate", 0,
                          empty_existing=True))
    data_rows.append(_row("cal-empty", "cal-g-empty", "calibration", "complement", 0,
                          empty_existing=True))
    data_rows.append(_row("heldout-empty", "heldout-g-empty", "heldout", "temporary", 0,
                          empty_existing=True))
    challenge_rows.append(_row("challenge-empty", "challenge-g-empty", None, "conflict", 0,
                               empty_existing=True, label_origin="agent_review"))
    data_rows.append(_multi_rule_row("train-multi", "train-g-multi"))
    data_rows.append(_set_row("set-0", "set-g-0"))
    challenge_rows.append(_multi_rule_row("challenge-multi", "challenge-g-multi"))
    return data_rows, challenge_rows


def _train_pair_rows() -> "list[dict]":
    return [row for row in _fixture()[0]
            if row.get("split") == "train" and len(row["request"]["existing"]) <= 1]


def _write_jsonl(path: Path, rows: "list[dict]") -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _tiny_model_dir(tmp_path: Path, *, corpus: str = "", name: str = "tiny-model") -> Path:
    """自建 tiny BERT（2 层 hidden 32）+ 字符级 fast tokenizer：真实前向/后向/offset，毫秒级。

    corpus 可换成调用方语料（字符级词表按该语料构建）；name 供同一 tmp_path 下建多个 tiny 模型。
    """
    from transformers import BertConfig, BertModel, BertTokenizerFast

    if not corpus:
        data_rows, challenge_rows = _fixture()
        corpus = "".join(json.dumps(row, ensure_ascii=False) for row in data_rows + challenge_rows)
    alphabet = sorted(set(corpus + "[]_abcdefghijklmnopqrstuvwxyz0123456789"))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet
    directory = tmp_path / name
    directory.mkdir()
    (directory / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(str(directory))
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=64,
                        max_position_embeddings=64)
    BertModel(config).save_pretrained(str(directory))
    return directory


def _slow_tokenizer(directory: Path):
    from transformers import BertTokenizer

    return BertTokenizer(vocab_file=str(directory / "vocab.txt"), do_lower_case=True)


def _embedding_model_path() -> "Path | None":
    candidates = [
        os.environ.get("MEMORY_RELATIONS_EMBEDDING_MODEL", ""),
        str(Path(__file__).resolve().parents[2] / "models" / "embeddings" / FEAT.EMBEDDING_DEFAULT_NAME),
        str(Path(__file__).resolve().parents[5] / "models" / "embeddings" / FEAT.EMBEDDING_DEFAULT_NAME),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def test_joint_serialization_spans_and_template_reads_only_allowed_fields():
    base = _build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0)
    dirty = copy.deepcopy(base)
    dirty["candidate"].update({"id": "cand-secret", "label": "conflict", "scope": "global"})
    dirty["existing"][0].update({"id": "rule-secret", "permanence": "persistent"})
    dirty["teacher_vote"] = "supersede"
    text = TRAIN.joint_text(dirty)
    assert TRAIN.joint_text(dirty) == TRAIN.joint_text(base)
    for marker in ("cand-secret", "rule-secret", "conflict", "persistent", "global", "supersede"):
        assert marker not in text
    assert "[candidate] " in text and "[existing] " in text and "[quote] " in text \
        and "[evidence] " in text
    assert text.index("[existing]") < text.index("[quote]") < text.index("[evidence]")
                                           
    blocks = TRAIN.joint_blocks(base)
    serialized, spans = TRAIN._serialize(blocks)
    assert serialized == text
    assert len(spans) == len(blocks)
    for (role, start, end), block in zip(spans, blocks):
        assert role == block.role and serialized[start:end] == f"[{block.role}] {block.text}"
    empty = _build_request("先做校正。", "", "x", 0, empty_existing=True)
    assert "[existing] [empty]" in TRAIN.joint_text(empty)
    with pytest.raises(ValueError):
        TRAIN.joint_text(_multi_rule_row("m", "g")["request"])


def test_truncation_attribution_is_precise(tmp_path):
    directory = _tiny_model_dir(tmp_path)
    from transformers import AutoTokenizer

    fast = AutoTokenizer.from_pretrained(str(directory))
    assert fast.is_fast, "tiny 资产应提供 fast tokenizer 以覆盖 offset 归属路径"
    slow = _slow_tokenizer(directory)
    request = _build_request("短。", "", "行。", 0)
    request["candidate"]["evidence"] = []
    request["evidence"] = [{"id": "e0", "text": "证" * 12}, {"id": "e1", "text": "据" * 4}]
    text, spans = TRAIN._serialize(TRAIN.joint_blocks(request))
    owners, special, method = TRAIN._attributed_owners(fast, text, spans)
    assert method == "offset_mapping"
    full = len(fast(text, add_special_tokens=True)["input_ids"])
    assert len(owners) == full - special
                                
    exact = TRAIN.truncation_profile(fast, [request], full)
    assert exact["truncated_rows"] == 0 and exact["by_source"] == {}
    assert exact["attribution"] == "offset_mapping"
    assert exact["counting_unit"].startswith("row")
                                                  
    cut = TRAIN.truncation_profile(fast, [request], full - 1)
    assert cut["truncated_rows"] == 1
    assert cut["by_source"] == {"evidence": {"cut": 1, "dropped": 0}}
    kept = fast(text, add_special_tokens=True, truncation=True, max_length=full - 1)["input_ids"]
    assert len(kept) == full - 1, "与 tokenizer 自身截断语义一致（保留 max_length 个 token）"
    assert {owner for owner in owners[full - 1 - special:] if owner} == {"evidence"}
                                                           
    boundaries = [end for _, _, end in spans]
    prefix_counts = [len(fast(text[:end], add_special_tokens=False)["input_ids"])
                     for end in boundaries]
    last_block_tokens = prefix_counts[-1] - prefix_counts[-2]
    deep = TRAIN.truncation_profile(fast, [request], full - last_block_tokens - 3)
    assert deep["truncated_rows"] == 1
    assert deep["by_source"] == {"evidence": {"cut": 1, "dropped": 0}}, \
        "多 same-role block 不得重复计行（旧实现会记 2）"
                                                                               
    candidate_tokens = prefix_counts[0]
    early = TRAIN.truncation_profile(fast, [request], special + candidate_tokens + 1)
    assert early["by_source"]["existing"] == {"cut": 1, "dropped": 0}
    assert early["by_source"]["evidence"] == {"cut": 0, "dropped": 1}
                                                   
    slow_full = len(slow(text, add_special_tokens=True)["input_ids"])
    fallback = TRAIN.truncation_profile(slow, [request], slow_full - 1)
    assert fallback["attribution"] == "token_prefix_boundary"
    assert fallback["by_source"] == {"evidence": {"cut": 1, "dropped": 0}}


def test_require_seven_classes_fails_loud():
    with pytest.raises(ValueError, match="未覆盖全部七类"):
        TRAIN._require_seven_classes(np.asarray([0, 1, 0]))
    TRAIN._require_seven_classes(np.arange(len(FEAT.CLASSES)))


def test_resolve_device_never_falls_back(monkeypatch):
    assert TRAIN.resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        TRAIN.resolve_device("mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="不回退 CPU"):
        TRAIN.resolve_device("cuda")
    assert TRAIN.resolve_device("cpu").type == "cpu"


def test_optimizer_groups_and_updates_differ_between_arms(tmp_path):
    model_dir = _tiny_model_dir(tmp_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    rows = _train_pair_rows()
    requests = [row["request"] for row in rows]
    labels = np.asarray([FEAT.CLASSES.index(row["label"]) for row in rows])
    device = torch.device("cpu")
                                         
    model_f, _ = TRAIN.build_joint_model(model_dir)
    optimizer = TRAIN.build_optimizer("frozen-joint", model_f)
    head_ids = {id(param) for param in model_f.classifier.parameters()}
    assert {id(param) for param in model_f.parameters() if param.requires_grad} == head_ids
    grouped = [param for group in optimizer.param_groups for param in group["params"]]
    assert {id(param) for param in grouped} == head_ids and len(grouped) == len(head_ids)
    assert {round(float(group["lr"]), 8) for group in optimizer.param_groups} == \
        {round(TRAIN.LR_HEAD, 8)}
    before = {name: tensor.detach().clone() for name, tensor in model_f.state_dict().items()}
    TRAIN.train_model("frozen-joint", model_f, tokenizer, requests, labels, seed=13, epochs=1,
                      batch_size=8, grad_accum=1, max_length=48, device=device, bf16=False)
    after = model_f.state_dict()
    assert all(torch.equal(after[name], before[name])
               for name in before if name.startswith("backbone.")), "冻结骨干必须逐位不变"
    assert any(not torch.equal(after[name], before[name])
               for name in before if name.startswith("classifier.")), "head 必须更新"
                                                
    model_t, _ = TRAIN.build_joint_model(model_dir)
    optimizer = TRAIN.build_optimizer("finetuned-joint", model_t)
    assert {round(float(group["lr"]), 8) for group in optimizer.param_groups} == \
        {round(TRAIN.LR_BACKBONE, 8), round(TRAIN.LR_HEAD, 8)}
    assert any(group["weight_decay"] == 0.0 for group in optimizer.param_groups), \
        "bias/LayerNorm 必须走无衰减组"
    trainable = {id(param) for param in model_t.parameters() if param.requires_grad}
    grouped = [param for group in optimizer.param_groups for param in group["params"]]
    assert {id(param) for param in grouped} == trainable and len(grouped) == len(trainable)
    before_t = {name: tensor.detach().clone() for name, tensor in model_t.state_dict().items()}
    TRAIN.train_model("finetuned-joint", model_t, tokenizer, requests, labels, seed=13, epochs=1,
                      batch_size=8, grad_accum=1, max_length=48, device=device, bf16=False)
    after_t = model_t.state_dict()
    assert any(not torch.equal(after_t[name], before_t[name])
               for name in before_t if name.startswith("backbone.")), "端到端骨干必须更新"
    assert any(not torch.equal(after_t[name], before_t[name])
               for name in before_t if name.startswith("classifier.")), "head 必须更新"


def test_tiny_model_cpu_end_to_end(tmp_path):
    model_dir = _tiny_model_dir(tmp_path)
    data_rows, challenge_rows = _fixture()
    data_path, challenge_path = tmp_path / "data.jsonl", tmp_path / "challenge.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(challenge_path, challenge_rows)
    out = tmp_path / "out"
    code = TRAIN.main([
        "--data", str(data_path), "--challenge", str(challenge_path), "--out", str(out),
        "--models", "frozen-joint,finetuned-joint", "--sizes", "22", "--seeds", "13",
        "--epochs", "2", "--batch-size", "4", "--grad-accum", "2", "--max-length", "48",
        "--device", "cpu", "--embedding-model", str(model_dir),
        "--bootstrap-samples", "12", "--smoke",
    ])
    assert code == 0
    manifest = json.loads((out / "planned_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned_before_fit"
    assert manifest["args"]["epochs"] == 1, "--smoke 只允许 1 epoch（测试模式）"
    assert manifest["args"]["smoke"] is True
    assert manifest["args"]["max_length"] == 48 and manifest["args"]["device"] == "cpu"
    assert manifest["source_sha256"]["data"] == TRAIN._file_sha256(data_path)
    assert manifest["model"]["hidden_size"] == 32 and manifest["model"]["num_hidden_layers"] == 2
    assert len(manifest["model"]["backbone_fingerprint"]) == 32
    semantics = manifest["semantics"]
    assert semantics["joint_max_length"] == 48
    assert "sentence_transformer_defaults" in semantics
    assert "唯一归因" in semantics["comparison_scope"], "必须写明 v1→v2 不能唯一归因 finetune"
    contract = manifest["artifact_contract"]
    assert contract["kind_weights_pairing"] == TRAIN.EXPECTED_WEIGHTS
    assert contract["meta_top_level"] == ["threshold", "threshold_feasible"]

    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert document["schema"] == TRAIN.TRAINING_SCHEMA
    assert document["planned_manifest"] == "planned_manifest.json"
    assert [run["model"] for run in document["runs"]] == ["frozen-joint", "finetuned-joint"]
    assert document["truncation"]["challenge"]["rows"] == 15
    assert document["truncation"]["challenge"]["counting_unit"].startswith("row")
    assert document["data"]["excluded_rows"]["train"]["multi_rule"] == 1
    assert set(document["data"]["group_id_enforced"]) == set(TRAIN.T.SPLITS) | {"challenge"}
    assert document["environment"]["bf16_used"] is False
    assert document["semantics"]["joint_max_length"] == 48
    for run in document["runs"]:
        assert run["parity"]["ok"] is True
        assert run["n_train"] == 22 and run["epochs"] == 1 and run["grad_accum"] == 2
        assert len(run["train_history"]) == 1
        assert np.isfinite(run["train_history"][0]["loss"])
        assert 0.0 <= run["train_history"][0]["accuracy"] <= 1.0
        train_eval = run["train_eval"]
        assert train_eval["mode"].startswith("eval") and train_eval["n"] == 22
        assert np.isfinite(train_eval["loss"])
        assert 0.0 <= train_eval["accuracy"] <= 1.0
        assert set(train_eval["prediction_counts"]) == set(FEAT.CLASSES)
        sample_profile = run["truncation_train_sample"]
        assert sample_profile["rows"] == 22
        assert sample_profile["counting_unit"].startswith("row")
        assert sample_profile["attribution"] in ("offset_mapping", "token_prefix_boundary")
        assert run["temperature"] > 0
        assert np.isfinite(run["nll_raw"]) and np.isfinite(run["nll_scaled"])
        assert set(run["eval"]) == {"heldout", "challenge"}
        calibration = run["calibration"]
        assert calibration["metrics"]["n"] == 15
        for name, result in run["eval"].items():
            assert result["n"] == (8 if name == "heldout" else 15)
            assert set(result["per_class"]) == set(FEAT.CLASSES)
            if run["operating_threshold"] is None:
                assert result["operating"] == {"threshold": None, "selected": False}
            else:
                assert result["operating"]["selected"] is True
            assert result["diagnostic_tau0"]["scope"] == "uncalibrated_tau0_diagnostic"
        artifact = out / "scorers" / f"{run['model']}-size22-seed13"
        meta = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
        assert meta["weights"] == TRAIN.EXPECTED_WEIGHTS[run["model"]]
        assert meta["classes"] == list(FEAT.CLASSES)
        assert "threshold" in meta and "threshold_feasible" in meta
        assert meta["threshold"] == run["operating_threshold"], \
            "meta 顶层 threshold 必须来自 calibration 选点且 null 保持 null"
        assert meta["threshold_feasible"] is run["threshold_feasible"]
        assert TRAIN.is_finetune_artifact(artifact) is True
        scorer = TRAIN.load_scorer(artifact, device="cpu")
        assert scorer.classes == FEAT.CLASSES
        assert scorer.threshold == meta["threshold"]
        probe = [row["request"] for row in challenge_rows[:2]]
        probs = scorer.predict_proba(probe)
        assert probs.shape == (2, len(FEAT.CLASSES))
        assert np.allclose(probs.sum(axis=1), 1.0)

    predictions = [json.loads(line) for line in
                   (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    assert len(predictions) == 2 * (15 + 8 + 15)
    first_run_challenge = [row for row in predictions
                           if row["model"] == "frozen-joint" and row["eval_set"] == "challenge"]
    assert len(first_run_challenge) == 15
    reloaded = TRAIN.load_scorer(out / "scorers" / "frozen-joint-size22-seed13")
    recorded = first_run_challenge[0]
    source = next(row for row in challenge_rows if row.get("id") == recorded["id"])
    assert np.allclose(reloaded.predict_proba([source["request"]])[0], recorded["probs"],
                       atol=2e-6), "工件重载应与 predictions.jsonl 记录一致"
    for row in predictions:
        assert len(row["probs"]) == len(FEAT.CLASSES)
        assert abs(sum(row["probs"]) - 1.0) < 1e-3
        assert row["verdict"] is None or row["verdict"] in FEAT.CLASSES
        assert row["verdict_tau0"] in FEAT.CLASSES
    curves = [json.loads(line) for line in
              (out / "curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(curves) == 2 * 3 * len(EM.default_thresholds()), "2 臂 × 3 个非空评估集 × τ 网格"
    assert all("eval_set" in point and "conditional_risk" in point for point in curves)


def test_artifact_contract_meta_threshold_pairing_and_dispatch(tmp_path):
    model_dir = _tiny_model_dir(tmp_path)
    from transformers import AutoTokenizer

    model, _ = TRAIN.build_joint_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    fingerprint = FEAT.model_fingerprint(model_dir)
    probe = [_build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0)]
    artifacts: dict[str, Path] = {}
    for kind, weights in (("frozen-joint", "head_only"), ("finetuned-joint", "full")):
        scorer = TRAIN.CrossEncoderScorer(kind=kind, model=model, tokenizer=tokenizer,
                                          max_length=48, device="cpu", temperature=1.5,
                                          threshold=0.25, threshold_feasible=True)
        directory = tmp_path / f"artifact-{kind}"
        TRAIN.save_scorer(scorer, directory, weights=weights, backbone_dir=model_dir,
                          backbone_fingerprint=fingerprint, train_meta={"seed": 13})
        artifacts[kind] = directory
        meta = json.loads((directory / TRAIN.META_FILE).read_text(encoding="utf-8"))
        assert meta["threshold"] == 0.25 and meta["threshold_feasible"] is True
        assert meta["kind"] == kind and meta["weights"] == weights
        assert TRAIN.is_finetune_artifact(directory) is True
                                                         
        loaded = TRAIN.load_scorer(directory, device="cpu")
        assert loaded.classes == FEAT.CLASSES
        assert loaded.threshold == 0.25 and loaded.threshold_feasible is True
        assert np.array_equal(scorer.predict_proba(probe), loaded.predict_proba(probe))
        assert TRAIN.check_reload_parity(scorer, directory, probe, device="cpu")["ok"] is True
    assert TRAIN.is_finetune_artifact(tmp_path / "nothing") is False
                                             
    null_scorer = TRAIN.CrossEncoderScorer(kind="frozen-joint", model=model, tokenizer=tokenizer,
                                           max_length=48, device="cpu")
    null_dir = tmp_path / "artifact-null"
    TRAIN.save_scorer(null_scorer, null_dir, weights="head_only", backbone_dir=model_dir,
                      backbone_fingerprint=fingerprint, train_meta={})
    null_meta = json.loads((null_dir / TRAIN.META_FILE).read_text(encoding="utf-8"))
    assert null_meta["threshold"] is None and null_meta["threshold_feasible"] is False
    assert TRAIN.load_scorer(null_dir).threshold is None
                                                  
    with pytest.raises(ValueError, match="必须配"):
        TRAIN.save_scorer(null_scorer, tmp_path / "bad-pair", weights="full",
                          backbone_dir=model_dir, backbone_fingerprint=fingerprint,
                          train_meta={})
    with pytest.raises(ValueError):
        TRAIN.CrossEncoderScorer(kind="frozen-joint", model=model, tokenizer=tokenizer,
                                 max_length=48, device="cpu", threshold=0.1,
                                 threshold_feasible=False)

    def _tampered_copy(name: str, source: Path) -> Path:
        target = tmp_path / name
        shutil.copytree(source, target)
        return target

    def _rewrite_meta(directory: Path, **updates) -> dict:
        path = directory / TRAIN.META_FILE
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(updates)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    frozen_dir = artifacts["frozen-joint"]
    hidden = json.loads((frozen_dir / TRAIN.META_FILE).read_text(encoding="utf-8"))["hidden_size"]
    kind_swap = _tampered_copy("artifact-kind-swap", frozen_dir)
    _rewrite_meta(kind_swap, kind="finetuned-joint")
    with pytest.raises(ValueError, match="不配对"):
        TRAIN.load_scorer(kind_swap)
    bad_weights = _tampered_copy("artifact-bad-weights", frozen_dir)
    _rewrite_meta(bad_weights, weights="half")
    with pytest.raises(ValueError, match="weights 无效"):
        TRAIN.load_scorer(bad_weights)
    bad_kind = _tampered_copy("artifact-bad-kind", frozen_dir)
    _rewrite_meta(bad_kind, kind="half-joint")
    with pytest.raises(ValueError, match="kind 无效"):
        TRAIN.load_scorer(bad_kind)
    missing_threshold = _tampered_copy("artifact-missing-threshold", frozen_dir)
    path = missing_threshold / TRAIN.META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["threshold"]
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少顶层 threshold"):
        TRAIN.load_scorer(missing_threshold)
    inconsistent = _tampered_copy("artifact-inconsistent-threshold", frozen_dir)
    _rewrite_meta(inconsistent, threshold_feasible=False)
    with pytest.raises(ValueError):
        TRAIN.load_scorer(inconsistent)
    hidden_shift = _tampered_copy("artifact-hidden", frozen_dir)
    _rewrite_meta(hidden_shift, hidden_size=int(hidden) + 1)
    with pytest.raises(ValueError, match="hidden_size"):
        TRAIN.load_scorer(hidden_shift)
    extra_key = _tampered_copy("artifact-backbone-key", frozen_dir)
    payload = torch.load(extra_key / TRAIN.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    payload["state_dict"]["backbone.probe"] = torch.zeros(1)
    torch.save(payload, extra_key / TRAIN.WEIGHTS_FILE)
    _rewrite_meta(extra_key,
                  weights_sha256=TRAIN._file_sha256(extra_key / TRAIN.WEIGHTS_FILE))
    with pytest.raises(ValueError, match="head_only 载荷"):
        TRAIN.load_scorer(extra_key)
    full_source = artifacts["finetuned-joint"]
    missing_backbone = _tampered_copy("artifact-full-missing", full_source)
    payload = torch.load(missing_backbone / TRAIN.WEIGHTS_FILE, map_location="cpu",
                         weights_only=True)
    payload["state_dict"] = {name: tensor for name, tensor in payload["state_dict"].items()
                             if not name.startswith("backbone.")}
    torch.save(payload, missing_backbone / TRAIN.WEIGHTS_FILE)
    _rewrite_meta(missing_backbone,
                  weights_sha256=TRAIN._file_sha256(missing_backbone / TRAIN.WEIGHTS_FILE))
    with pytest.raises(ValueError, match="full 载荷"):
        TRAIN.load_scorer(missing_backbone)
    tampered = _tampered_copy("artifact-tamper", frozen_dir)
    data = bytearray((tampered / TRAIN.WEIGHTS_FILE).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (tampered / TRAIN.WEIGHTS_FILE).write_bytes(bytes(data))
    with pytest.raises(ValueError, match="哈希不符"):
        TRAIN.load_scorer(tampered)
    with pytest.raises(FileNotFoundError):
        TRAIN.load_scorer(tmp_path / "missing")


def test_eval_group_id_presence_is_enforced(tmp_path):
    data_rows, challenge_rows = _fixture()
    for row in data_rows:
        if row.get("id") == "heldout-complement":
            del row["group_id"]
    data_path, challenge_path = tmp_path / "data.jsonl", tmp_path / "challenge.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(challenge_path, challenge_rows)
    with pytest.raises(ValueError, match="heldout 分区.*group_id"):
        TRAIN.main(["--data", str(data_path), "--challenge", str(challenge_path),
                    "--out", str(tmp_path / "out"), "--models", "frozen-joint",
                    "--sizes", "22", "--seeds", "13"])
    data_rows2, challenge_rows2 = _fixture()
    del challenge_rows2[0]["group_id"]
    _write_jsonl(data_path, data_rows2)
    _write_jsonl(challenge_path, challenge_rows2)
    with pytest.raises(ValueError, match="challenge 分区.*group_id"):
        TRAIN.main(["--data", str(data_path), "--challenge", str(challenge_path),
                    "--out", str(tmp_path / "out2"), "--models", "frozen-joint",
                    "--sizes", "22", "--seeds", "13"])


def test_real_backbone_forward_freeze_and_offsets_if_available():
    embedding = _embedding_model_path()
    if embedding is None:
        pytest.skip("本地 MiniLM 权重不可见（禁止下载）；设 MEMORY_RELATIONS_EMBEDDING_MODEL 后重跑")
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(str(embedding), local_files_only=True)
    assert config.hidden_size == 384 and config.num_hidden_layers == 12
    assert config.max_position_embeddings >= 384
    model, _ = TRAIN.build_joint_model(embedding, dropout=0.1)
    assert int(model.classifier.in_features) == 384
    requests = [_build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0),
                _build_request("汇总必须两行。", "", "汇总必须只有一行。", 1)]
    labels = torch.tensor([FEAT.CLASSES.index("duplicate"), FEAT.CLASSES.index("conflict")])
    tokenizer = AutoTokenizer.from_pretrained(str(embedding), local_files_only=True)
    assert tokenizer.is_fast, "本地 MiniLM 应提供 fast tokenizer（offset 归属路径）"
    profile = TRAIN.truncation_profile(tokenizer, requests, 384)
    assert profile["attribution"] == "offset_mapping"
    assert profile["truncated_rows"] == 0, "预检请求远短于 384，不得出现截断"
    encoded = TRAIN._encode(tokenizer, [TRAIN.joint_text(request) for request in requests], 64,
                            torch.device("cpu"))
    loss = torch.nn.CrossEntropyLoss()(model(**encoded), labels)
    loss.backward()
    assert all(p.grad is not None for p in model.classifier.parameters()), "head 必须有梯度"
    assert any(p.grad is not None for p in model.backbone.parameters()), "未冻结骨干应有梯度"
    for param in model.backbone.parameters():
        param.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    loss = torch.nn.CrossEntropyLoss()(model(**encoded), labels)
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters()), "冻结后骨干不得再有梯度"
    assert all(p.grad is not None for p in model.classifier.parameters())


def _dev_rows() -> list[dict]:
    rows = []
    for index, label in enumerate(FEAT.CLASSES):
        row = _row(f"dev-{label}", f"dev-g-{label}", "dev", label, index)
        row["request"]["candidate"]["text"] += "仅开发集。"
        row["phenomena"] = {"root_id": f"dev-root-{label}", "domain": "writing",
                            "family": "validation", "author": "synthetic-author",
                            "label_rationale": "合成fixture，不是真人金标"}
        rows.append(row)
    return rows


def _train_args(rows, **overrides) -> dict:
    arguments = {"seed": 13, "epochs": 1, "batch_size": 8, "grad_accum": 2,
                 "max_length": 48, "device": torch.device("cpu"), "bf16": False}
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize("kind", TRAIN.MODEL_KINDS)
@pytest.mark.parametrize("metric", ["dev-loss", "dev-macro-f1"])
def test_best_nonfinal_epoch_cpu_clone_restore_and_reload(tmp_path, monkeypatch, kind, metric):
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model, _ = TRAIN.build_joint_model(model_dir)
    rows, dev = _train_pair_rows(), _dev_rows()
    snapshots = []
    original = TRAIN.eval_mode_metrics

    def controlled(model, tokenizer, requests, labels, **kwargs):
        metrics = original(model, tokenizer, requests, labels, **kwargs)
        snapshots.append(TRAIN._cpu_state(model))
        index = len(snapshots) - 1
        metrics.update({"loss": [1.2, 0.2, 0.7, 0.9][index],
                        "macro_f1": [0.1, 0.8, 0.5, 0.3][index]})
        return metrics

    monkeypatch.setattr(TRAIN, "eval_mode_metrics", controlled)
    result = TRAIN.train_model(
        kind, model, tokenizer, [row["request"] for row in rows], TRAIN.T._labels(rows),
        **_train_args(rows, epochs=4, checkpoint_selection="best-dev", selection_metric=metric,
                      dev_requests=[row["request"] for row in dev], dev_labels=TRAIN.T._labels(dev), dev_group_ids=TRAIN.T._root_ids(dev)))
    selection = result["selection"]
    assert selection["selected_epoch"] == 2 and selection["selected_step"] == 4
    assert selection["state_restored"] and not selection["stopped_early"]
    assert result["budget"]["actual_sample_instances"] == 4 * len(rows)
    assert all(tensor.device.type == "cpu" for tensor in snapshots[1].values())
    assert all(torch.equal(tensor, snapshots[1][name]) for name, tensor in model.state_dict().items())
    assert any(not torch.equal(snapshots[-1][name], snapshots[1][name])
               for name in snapshots[1] if name.startswith("classifier."))
    scorer = TRAIN.CrossEncoderScorer(kind=kind, model=model, tokenizer=tokenizer, max_length=48,
                                      batch_size=3, temperature=1.4)
    directory = tmp_path / "selected-artifact"
    TRAIN.save_scorer(scorer, directory, weights=TRAIN.EXPECTED_WEIGHTS[kind], backbone_dir=model_dir,
                      backbone_fingerprint=FEAT.model_fingerprint(model_dir), train_meta={"selection": selection})
    payload = torch.load(directory / TRAIN.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in payload["state_dict"].values())
    assert all(torch.equal(tensor, snapshots[1][name]) for name, tensor in payload["state_dict"].items())
    reloaded = TRAIN.load_scorer(directory)
    assert reloaded.batch_size == 3
    probe = [row["request"] for row in dev]
    assert TRAIN.check_reload_parity(scorer, directory, probe, device="cpu")["ok"]
    assert np.array_equal(scorer.predict_proba(probe), reloaded.predict_proba(probe))


@pytest.mark.parametrize("metric", ["dev-loss", "dev-macro-f1"])
def test_patience_min_delta_resets_after_improvement_then_stops(tmp_path, monkeypatch, metric):
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model, _ = TRAIN.build_joint_model(model_dir)
    rows, dev = _train_pair_rows(), _dev_rows()
    observed = []

    def controlled(model, *args, **kwargs):
        observed.append(TRAIN._cpu_state(model))
        index = len(observed) - 1
        return {"loss": [1.0, 0.8, 0.79, 0.78][index],
                "row_loss": [1.0, 0.8, 0.79, 0.78][index],
                "macro_f1": [0.4, 0.6, 0.61, 0.62][index]}

    monkeypatch.setattr(TRAIN, "eval_mode_metrics", controlled)
    result = TRAIN.train_model(
        "frozen-joint", model, tokenizer, [row["request"] for row in rows], TRAIN.T._labels(rows),
        **_train_args(rows, epochs=6, checkpoint_selection="best-dev", selection_metric=metric,
                      patience=2, min_delta=0.05, dev_requests=[row["request"] for row in dev],
                      dev_labels=TRAIN.T._labels(dev), dev_group_ids=TRAIN.T._root_ids(dev)))
    assert len(result["history"]) == 4 and result["selection"]["selected_epoch"] == 2
    assert result["selection"]["stopped_early"] is True
    assert result["budget"]["actual_updates"] == 8
    assert result["budget"]["actual_sample_instances"] == 4 * len(rows)
    assert result["budget"]["planned_sample_instances"] == 6 * len(rows)
    assert all(torch.equal(value, observed[1][name]) for name, value in model.state_dict().items())


def test_unequal_tail_microbatches_match_true_sample_mean_gradients(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    initial, _ = TRAIN.build_joint_model(model_dir, dropout=0.0)
    rows = [next(row for row in _train_pair_rows() if row["label"] == label) for label in FEAT.CLASSES]
    requests, labels = [row["request"] for row in rows], TRAIN.T._labels(rows)
    gradients = []
    original_step = torch.optim.AdamW.step

    def capture(optimizer, *args, **kwargs):
        gradients.append([parameter.grad.detach().clone() for group in optimizer.param_groups
                          for parameter in group["params"]])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", capture)
    accumulated, direct = copy.deepcopy(initial), copy.deepcopy(initial)
    result = TRAIN.train_model("frozen-joint", accumulated, tokenizer, requests, labels,
                               **_train_args(rows, batch_size=4, grad_accum=2))
    TRAIN.train_model("frozen-joint", direct, tokenizer, requests, labels,
                      **_train_args(rows, batch_size=7, grad_accum=1))
    assert result["updates"][0]["microbatch_sizes"] == [4, 3]
    assert result["budget"]["actual_sample_instances"] == 7
    assert len(gradients) == 2
    assert all(torch.allclose(left, right, atol=2e-7, rtol=2e-6)
               for left, right in zip(gradients[0], gradients[1]))
    assert all(torch.allclose(accumulated.state_dict()[name], direct.state_dict()[name], atol=1e-7)
               for name in initial.state_dict())


def test_fixed_updates_fill_across_epoch_boundary_and_count_exact_tokens(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model, _ = TRAIN.build_joint_model(model_dir)
    rows = [next(row for row in _train_pair_rows() if row["label"] == label) for label in FEAT.CLASSES]
    rows[0]["request"] = {"candidate": {"text": "短", "trigger": ""}, "existing": [], "evidence": []}
    encoded_counts = []
    original_encode = TRAIN._encode

    def encode(tokenizer, texts, max_length, device):
        result = original_encode(tokenizer, texts, max_length, device)
        real = int(result["attention_mask"].sum())
        encoded_counts.append((len(texts), real, result["input_ids"].numel() - real))
        return result

    monkeypatch.setattr(TRAIN, "_encode", encode)
    result = TRAIN.train_model("finetuned-joint", model, tokenizer, [row["request"] for row in rows],
                               TRAIN.T._labels(rows), **_train_args(rows, batch_size=4, grad_accum=2,
                                                                   max_steps=5, eval_every_steps=2))
    budget = result["budget"]
    assert budget["actual_updates"] == budget["planned_updates"] == 5
    assert budget["actual_sample_instances"] == budget["planned_sample_instances"] == 40
    assert len(encoded_counts) == 10 and all(count[0] == 4 for count in encoded_counts)
    assert budget["real_tokens"] == sum(count[1] for count in encoded_counts)
    assert budget["padding_tokens"] == sum(count[2] for count in encoded_counts) > 0
    assert [record["step"] for record in result["history"]] == [2, 4, 5]
    assert all(update["microbatch_sizes"] == [4, 4] and update["sample_instances"] == 8
               for update in result["updates"])
    assert result["selection"]["selected_step"] == 5


def test_tuning_cli_fixed_budget_default_patience_zero_and_zero_terminal_access(tmp_path, monkeypatch):
    model_dir = _tiny_model_dir(tmp_path)
    rows = [row for row in _fixture()[0] if row["split"] in ("train", "calibration")] + _dev_rows()
    path, out = tmp_path / "development.jsonl", tmp_path / "out"
    _write_jsonl(path, rows)
    original_metrics, original_parity, original_load = TRAIN.eval_mode_metrics, TRAIN.check_reload_parity, TRAIN.T.load_rows
    dev_evaluations, parity_requests = [], []

    def load(*args, **kwargs):
        assert kwargs["source"] != "challenge"
        return original_load(*args, **kwargs)

    def metrics(model, tokenizer, requests, labels, **kwargs):
        result = original_metrics(model, tokenizer, requests, labels, **kwargs)
        assert kwargs["batch_size"] == 3
        if all("仅开发集" in request["candidate"]["text"] for request in requests):
            dev_evaluations.append(result)
            result["loss"] = [0.3, 0.1, 0.8][len(dev_evaluations) - 1]
        return result

    def parity(scorer, directory, requests, **kwargs):
        assert all("仅开发集" in request["candidate"]["text"] for request in requests)
        parity_requests.extend(requests)
        return original_parity(scorer, directory, requests, **kwargs)

    monkeypatch.setattr(TRAIN.T, "load_rows", load)
    monkeypatch.setattr(TRAIN, "eval_mode_metrics", metrics)
    monkeypatch.setattr(TRAIN, "check_reload_parity", parity)
    assert TRAIN.main(["--data", str(path), "--challenge", str(tmp_path / "sealed-do-not-open.jsonl"),
                       "--out", str(out), "--models", "frozen-joint", "--sizes", "22", "--seeds", "13",
                       "--epochs", "1", "--checkpoint-selection", "best-dev", "--max-steps", "5",
                       "--eval-every-steps", "2", "--eval-batch-size", "3", "--batch-size", "4",
                       "--grad-accum", "2", "--max-length", "48", "--device", "cpu",
                       "--embedding-model", str(model_dir), "--eval-splits", "none",
                       "--bootstrap-samples", "5", "--threads", "2"]) == 0
    manifest = json.loads((out / "planned_manifest.json").read_text(encoding="utf-8"))
    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    run = document["runs"][0]
    assert manifest["args"]["patience"] == 0 and manifest["epoch_selection"] == "best-dev"
    assert manifest["source_sha256"]["challenge"] is None and run["eval"] == {}
    assert manifest["budgets"][0]["updates"] == run["budget"]["actual_updates"] == 5
    assert run["budget"]["actual_sample_instances"] == 40
    assert run["selection"]["selected_step"] == 4 and not run["selection"]["stopped_early"]
    assert run["train_eval"]["n"] == 22 and run["train_eval"]["mode"].startswith("eval")
    assert set(document["truncation"]) == {"train", "dev", "calibration"}
    assert document["truncation"]["dev"]["rows"] == 7
    assert all(profile["rows"] == 1 for profile in document["truncation"]["dev"]["by_class"].values())
    assert set(run["memory_stages"]) == {"build", "train_and_dev", "selected_train_eval", "calibration",
                                         "eval_calibration", "export", "reload_parity"}
    assert all(stage["peak_allocated_mb"] is None for stage in run["memory_stages"].values())
    assert parity_requests and len(dev_evaluations) == 3
    artifact = out / "scorers" / "frozen-joint-size22-seed13"
    meta = json.loads((artifact / TRAIN.META_FILE).read_text(encoding="utf-8"))
    assert meta["train"]["selection"] == run["selection"]
    assert TRAIN.load_scorer(artifact).batch_size == 3
    trace = [json.loads(line) for line in (out / run["training_trace"]).read_text(encoding="utf-8").splitlines()]
    assert trace[0]["event"] == "planned" and trace[-1]["event"] == "completed_run"
    assert [record["step"] for record in trace if record["event"] == "update"] == [1, 2, 3, 4, 5]
    assert len([record for record in trace if record["event"] == "dev_evaluation"]) == 3
    assert trace[-1]["selection"] == run["selection"] and trace[-1]["budget"] == run["budget"]
    assert all("request" not in record and "labels" not in record for record in trace)


@pytest.mark.parametrize("extra", [["--checkpoint-selection", "best-dev"], ["--max-steps", "-1"],
                                   ["--eval-every-steps", "-1"], ["--eval-batch-size", "0"],
                                   ["--min-delta", "nan"], ["--patience", "1"], ["--sizes", "3,22"],
                                   ["--seeds", "13,-1"]])
def test_joint_preflight_refuses_before_model_build(tmp_path, monkeypatch, extra):
    path, out = tmp_path / "development.jsonl", tmp_path / "out"
    rows = [row for row in _fixture()[0] if row["split"] in ("train", "calibration")]
    if extra != ["--checkpoint-selection", "best-dev"]:
        rows += _dev_rows()
    _write_jsonl(path, rows)
    monkeypatch.setattr(TRAIN, "build_joint_model", lambda *args, **kwargs: pytest.fail("must not build"))
    with pytest.raises(ValueError):
        TRAIN.main(["--data", str(path), "--out", str(out), "--models", "frozen-joint", "--sizes", "22",
                    "--seeds", "13", "--eval-splits", "none"] + extra)
    assert not out.exists()


def test_failed_training_keeps_committed_update_trace_without_artifact(tmp_path, monkeypatch):
    model_dir = _tiny_model_dir(tmp_path)
    path, out = tmp_path / "development.jsonl", tmp_path / "failed-run"
    _write_jsonl(path, [row for row in _fixture()[0] if row["split"] in ("train", "calibration")] + _dev_rows())
    monkeypatch.setattr(TRAIN, "eval_mode_metrics", lambda *args, **kwargs: {"loss": float("nan"), "macro_f1": 0.0})
    with pytest.raises(ValueError, match="dev指标非有限"):
        TRAIN.main(["--data", str(path), "--out", str(out), "--models", "frozen-joint", "--sizes", "22",
                    "--seeds", "13", "--eval-splits", "none", "--checkpoint-selection", "best-dev",
                    "--max-steps", "2", "--eval-every-steps", "1", "--batch-size", "4", "--grad-accum", "2",
                    "--max-length", "48", "--embedding-model", str(model_dir), "--threads", "2"])
    trace_path = out / "training" / "frozen-joint-size22-seed13.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in trace] == ["planned", "update", "failed"]
    assert trace[1]["sample_instances"] == 8 and trace[2]["error_type"] == "ValueError"
    assert (out / "planned_manifest.json").is_file()
    assert not (out / "scorers").exists() and not (out / "metrics.json").exists()


def test_default_head_stays_seven_and_public_primitives_delegate(tmp_path, monkeypatch):
    """公共原语增补的七类回归：默认头数/空输入/私有名薄封装语义全部不变。"""
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    default_model, config = TRAIN.build_joint_model(model_dir)
    assert int(default_model.classifier.out_features) == len(FEAT.CLASSES)
    assert int(default_model.classifier.in_features) == int(config.hidden_size)
    small_model, _ = TRAIN.build_joint_model(model_dir, num_labels=3)
    assert int(small_model.classifier.out_features) == 3
    with pytest.raises(ValueError):
        TRAIN.build_joint_model(model_dir, num_labels=0)
                                        
    assert TRAIN.logits_for(default_model, tokenizer, [], max_length=48, device="cpu").shape == (0, 7)
    assert TRAIN.logits_for(small_model, tokenizer, [], max_length=48, device="cpu").shape == (0, 3)
                                                    
    seen = []
    original_encode = TRAIN._encode

    def encode(tokenizer, texts, max_length, device):
        seen.append(list(texts))
        return original_encode(tokenizer, texts, max_length, device)

    monkeypatch.setattr(TRAIN, "_encode", encode)
    TRAIN.encode_batch(tokenizer, ["公共原语"], 48, torch.device("cpu"))
    assert seen == [["公共原语"]]
    assert TRAIN.file_sha256(model_dir / "vocab.txt") == TRAIN._file_sha256(model_dir / "vocab.txt")
    windows = list(TRAIN.training_windows(7, epochs=2, batch_size=3, grad_accum=1, max_steps=0,
                                          rng=np.random.default_rng(13)))
    reference = list(TRAIN._training_windows(7, epochs=2, batch_size=3, grad_accum=1, max_steps=0,
                                             rng=np.random.default_rng(13)))
    assert [step for step, _, _, _ in windows] == [step for step, _, _, _ in reference]
    assert all(np.array_equal(left, right) for (_, _, chunks, _), (_, _, chunks2, _)
               in zip(windows, reference) for left, right in zip(chunks, chunks2))
    assert TRAIN.cpu_state(default_model)["classifier.weight"].device.type == "cpu"
    assert TRAIN.resolve_bf16("off", torch.device("cpu")) is False
    assert hasattr(TRAIN.autocast_for(torch.device("cpu"), True), "__enter__")
    TRAIN.validate_training_options(
        epochs=1, batch_size=2, grad_accum=1, max_length=48, checkpoint_selection="final",
        selection_metric="dev-loss", patience=0, min_delta=0.0, max_steps=0, eval_every_steps=0,
        eval_batch_size=2)
    with pytest.raises(ValueError):
        TRAIN.validate_training_options(
            epochs=1, batch_size=2, grad_accum=1, max_length=48, checkpoint_selection="final",
            selection_metric="dev-loss", patience=1, min_delta=0.0, max_steps=0, eval_every_steps=0,
            eval_batch_size=2)


def test_build_joint_optimizer_is_shared_by_both_seven_class_arms(tmp_path):
    model_dir = _tiny_model_dir(tmp_path)
    model, _ = TRAIN.build_joint_model(model_dir)
    frozen = TRAIN.build_joint_optimizer(model, train_backbone=False)
    head_ids = {id(param) for param in model.classifier.parameters()}
    assert {id(param) for param in model.parameters() if param.requires_grad} == head_ids
    grouped = [param for group in frozen.param_groups for param in group["params"]]
    assert {id(param) for param in grouped} == head_ids
    tuned, _ = TRAIN.build_joint_model(model_dir)
    full = TRAIN.build_joint_optimizer(tuned, train_backbone=True)
    assert {round(float(group["lr"]), 8) for group in full.param_groups} == \
        {round(TRAIN.LR_BACKBONE, 8), round(TRAIN.LR_HEAD, 8)}
    assert all(param.requires_grad for param in tuned.parameters())
    with pytest.raises(ValueError):
        TRAIN.build_optimizer("half-joint", tuned)


def test_train_model_still_requires_seven_classes(tmp_path):
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    model, _ = TRAIN.build_joint_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    requests = [row["request"] for row in _train_pair_rows()]
    three = np.asarray([index % 3 for index in range(len(requests))])
    with pytest.raises(ValueError, match="未覆盖全部七类"):
        TRAIN.train_model("frozen-joint", model, tokenizer, requests, three, seed=13, epochs=1,
                          batch_size=8, grad_accum=1, max_length=48, device=torch.device("cpu"),
                          bf16=False)


def test_seven_class_save_scorer_rejects_foreign_head(tmp_path):
    """七类 saver 不得接受非 7 类头：其它标签空间必须用自己的工件合同（如 NLI 参考臂）。"""
    from transformers import AutoTokenizer

    model_dir = _tiny_model_dir(tmp_path)
    model, _ = TRAIN.build_joint_model(model_dir, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    scorer = TRAIN.CrossEncoderScorer(kind="finetuned-joint", model=model, tokenizer=tokenizer,
                                      max_length=48, device="cpu")
    with pytest.raises(ValueError, match="类头"):
        TRAIN.save_scorer(scorer, tmp_path / "bad-head", weights="full", backbone_dir=model_dir,
                          backbone_fingerprint=FEAT.model_fingerprint(model_dir), train_meta={})


def test_seven_class_dev_callback_retains_root_count_without_dev_groups(tmp_path):
    from transformers import AutoTokenizer

    directory = _tiny_model_dir(tmp_path)
    model, effective = TRAIN.build_joint_model(directory)
    assert effective is model.backbone.config
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    rows = _train_pair_rows()
    requests = [row["request"] for row in rows]
    labels = np.asarray([FEAT.CLASSES.index(row["label"]) for row in rows])
    groups = [row["group_id"] for row in rows]
    events = []
    result = TRAIN.train_model("frozen-joint", model, tokenizer, requests, labels, seed=13, epochs=1,
                                batch_size=8, grad_accum=1, max_length=48, max_steps=1,
                                device=torch.device("cpu"), bf16=False, dev_requests=requests,
                                dev_labels=labels, dev_group_ids=groups, event_callback=events.append)
    dev = [event for event in events if event["event"] == "dev_evaluation"]
    assert len(dev) == 1 and dev[0]["dev_root_count"] == len(set(groups))
    assert "dev_groups" not in dev[0] and "dev_groups" not in result["history"][0]
