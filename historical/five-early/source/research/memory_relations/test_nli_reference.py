"""nli_reference.py 的研究专项测试：tiny CPU 模型上的真实梯度/预算/加载保真与正反用例。

跑法：pytest -p no:cacheprovider research/memory_relations/test_nli_reference.py -q
（需 PYTHONPATH=src;仓库根）。全部离线（autouse 置 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1）、
CPU、只写 pytest tmp_path；不做 GPU 训练、不下载模型、不读旧运行/旧工件、不触碰公开验收数据
（本模块没有验收数据入口）。真实 MiniLM 仅一条 gated 加载/单步 smoke（可见时），不做正式训练。
"""
from __future__ import annotations

import copy
import inspect
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
from research.memory_relations import nli_reference as NLI
from research.memory_relations.test_finetune import _build_request, _embedding_model_path, _tiny_model_dir

_NLI_TEXTS = {
    "entailment": [("一个男人正在外面走路。", "一个男人在户外行走。"),
                   ("小孩在吃苹果。", "孩子在吃水果。")],
    "neutral": [("一个男人正在外面走路。", "一个男人正在跑步。"),
                ("猫睡在沙发上。", "狗睡在地板上。")],
    "contradiction": [("一个男人正在外面走路。", "一个男人在办公室坐着。"),
                      ("小孩在吃苹果。", "孩子什么也没吃。")],
}


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """禁网：本模块测试只吃本地路径权重，transformers 离线标志兜底。"""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _nli_corpus() -> str:
    texts = [text for pairs in _NLI_TEXTS.values() for pair in pairs for text in pair]
    return " ".join(texts) + " [candidate] [existing] [quote] [evidence] " + " ".join(NLI.NLI_CLASSES)


def _nli_request(premise: str, hypothesis: str) -> dict:
    """共同 memory-shaped request：主向 existing=[premise]、candidate=hypothesis（无 trigger/evidence）。"""
    return {"candidate": {"text": hypothesis, "trigger": ""},
            "existing": [{"text": premise, "trigger": ""}],
            "evidence": []}


def _nli_rows(*, split: str = "train") -> list[dict]:
    rows = []
    for label, pairs in sorted(_NLI_TEXTS.items()):
        for index, (premise, hypothesis) in enumerate(pairs):
            rows.append({"split": split, "label": label, "group_id": f"{split}-g-{label}-{index}",
                         "request": _nli_request(premise, hypothesis)})
    return rows


def _train_args(**overrides) -> dict:
    arguments = {"seed": 13, "epochs": 1, "batch_size": 3, "grad_accum": 1, "max_length": 48,
                 "device": torch.device("cpu"), "bf16": False}
    arguments.update(overrides)
    return arguments


def _tiny_nli_model(tmp_path):
    directory = _tiny_model_dir(tmp_path, corpus=_nli_corpus(), name="tiny-nli-model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model, _ = TRAIN.build_joint_model(directory, num_labels=len(NLI.NLI_CLASSES))
    return directory, tokenizer, model


def test_nli_label_space_rejects_hard_mapping():
    assert NLI.label_indexes(["entailment", "neutral", "contradiction"]).tolist() == [0, 1, 2]
    assert NLI.label_indexes(np.asarray([2, 1, 0])).tolist() == [2, 1, 0]
    assert NLI.row_labels(_nli_rows()).tolist() == [2, 2, 0, 0, 1, 1]
    for label in ("duplicate", "uncertain", "conflict", "non-entailment", "complement"):
        with pytest.raises(ValueError, match="禁止七类硬映射"):
            NLI.label_indexes([label])
    with pytest.raises(ValueError, match="越界"):
        NLI.label_indexes([3])
    with pytest.raises(ValueError, match="越界"):
        NLI.label_indexes([-1])
    with pytest.raises(ValueError):
        NLI.label_indexes([True])
    with pytest.raises(ValueError):
        NLI.label_indexes([None])
    with pytest.raises(ValueError, match="未覆盖全部三类"):
        NLI.require_three_classes([0, 1, 0])
    NLI.require_three_classes([0, 1, 2])


def test_train_nli_gates_label_space_and_three_class_coverage(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    missing_neutral = [row for row in _nli_rows() if row["label"] != "neutral"]
    with pytest.raises(ValueError, match="未覆盖全部三类"):
        NLI.train_nli(model, tokenizer, missing_neutral, **_train_args())
    hard_mapped = copy.deepcopy(_nli_rows())
    hard_mapped[0]["label"] = "duplicate"
    with pytest.raises(ValueError, match="禁止七类硬映射"):
        NLI.train_nli(model, tokenizer, hard_mapped, **_train_args())
    out_of_space = copy.deepcopy(_nli_rows())
    out_of_space[0]["label"] = 3
    with pytest.raises(ValueError, match="越界"):
        NLI.train_nli(model, tokenizer, out_of_space, **_train_args())
    with pytest.raises(ValueError, match="方向"):
        NLI.train_nli(model, tokenizer, _nli_rows(), direction="七类硬映射", **_train_args())


def test_train_nli_refuses_window_beyond_backbone_positions(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    assert int(model.backbone.config.max_position_embeddings) == 64, "tiny fixture 位置上限固定"
    with pytest.raises(ValueError, match="max_position_embeddings"):
        NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args(max_length=65))
    result = NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args(max_length=64, max_steps=1))
    assert result["truncation"]["train"]["max_length"] == 64


def test_train_nli_train_only_and_sealed_rows_are_rejected(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    base = _nli_rows()
    for split in ("heldout", "ood", "set", "test", "dev", "calibration"):
        rows = [dict(row, split=split) for row in base]
        with pytest.raises(ValueError, match="不允许"):
            NLI.train_nli(model, tokenizer, rows, **_train_args())
    without_split = [{key: value for key, value in row.items() if key != "split"} for row in base]
    with pytest.raises(ValueError, match="缺少 split"):
        NLI.train_nli(model, tokenizer, without_split, **_train_args())
    sealed = [dict(row, sealed=True) for row in base]
    with pytest.raises(ValueError, match="sealed"):
        NLI.train_nli(model, tokenizer, sealed, **_train_args())
    no_request = [dict(row, request=None) for row in base]
    with pytest.raises(ValueError, match="request"):
        NLI.train_nli(model, tokenizer, no_request, **_train_args())
    with pytest.raises(ValueError, match="行不能为空"):
        NLI.train_nli(model, tokenizer, [], **_train_args())
                                                          
    result = NLI.train_nli(model, tokenizer, base, dev_rows=_nli_rows(split="dev"), **_train_args())
    assert result["data"]["dev_splits"] == {"dev": len(base)}
    for split in ("heldout", "ood", "test"):
        dev_bad = [dict(row, split=split) for row in _nli_rows(split="dev")]
        with pytest.raises(ValueError, match="不允许"):
            NLI.train_nli(model, tokenizer, base, dev_rows=dev_bad, **_train_args())
    dev_sealed = [dict(row, sealed=True) for row in _nli_rows(split="dev")]
    with pytest.raises(ValueError, match="sealed"):
        NLI.train_nli(model, tokenizer, base, dev_rows=dev_sealed, **_train_args())
    with pytest.raises(ValueError, match="best-dev 必须提供独立 dev 行"):
        NLI.train_nli(model, tokenizer, base, checkpoint_selection="best-dev", **_train_args())


def test_train_nli_fixed_budget_updates_backbone_and_counts_tokens(tmp_path, monkeypatch):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    rows = _nli_rows()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    encoded = []
    original_encode = TRAIN.encode_batch

    def encode(tokenizer, texts, max_length, device):
        result = original_encode(tokenizer, texts, max_length, device)
        real = int(result["attention_mask"].sum())
        encoded.append((len(texts), real, result["input_ids"].numel() - real))
        return result

    monkeypatch.setattr(TRAIN, "encode_batch", encode)
    result = NLI.train_nli(model, tokenizer, rows, **_train_args(max_steps=2, batch_size=2, grad_accum=2))
    budget = result["budget"]
    assert budget["mode"] == "fixed-updates" and budget["actual_updates"] == budget["planned_updates"] == 2
    assert budget["actual_sample_instances"] == budget["planned_sample_instances"] == 8
    assert len(encoded) == 4 and all(count[0] == 2 for count in encoded), "NLI 必须走共享 encode_batch 原语"
    assert budget["real_tokens"] == sum(count[1] for count in encoded)
    assert budget["padding_tokens"] == sum(count[2] for count in encoded) > 0
    assert [update["microbatch_sizes"] for update in result["updates"]] == [[2, 2], [2, 2]]
    assert all(update["sample_instances"] == 4 for update in result["updates"])
    after = model.state_dict()
    assert any(not torch.equal(after[name], before[name])
               for name in before if name.startswith("backbone.")), "NLI 参考臂必须端到端更新骨干"
    assert any(not torch.equal(after[name], before[name])
               for name in before if name.startswith("classifier.")), "head 必须更新"
    assert len(result["history"]) == 1 and np.isfinite(result["history"][0]["loss"])
    assert result["selection"]["mode"] == "final" and result["selection"]["state_restored"] is True
    assert result["selection"]["weighting"] == "row_native" and result["selection"]["dev_rows"] == 0
    assert result["data"]["train_splits"] == {"train": len(rows)}
    assert result["data"]["train_counts"] == {"entailment": 2, "neutral": 2, "contradiction": 2}
    assert result["data"]["direction"] == NLI.NLI_PRIMARY_DIRECTION
    assert result["data"]["joint_roles"] == NLI.NLI_JOINT_ROLES[NLI.NLI_PRIMARY_DIRECTION]
    assert result["truncation"]["train"]["rows"] == len(rows) and "dev" not in result["truncation"]
    assert result["truncation"]["train"]["counting_unit"].startswith("row")
    assert result["device_log"]["resolved"] == "cpu" and result["device_log"]["bf16"] is False


@pytest.mark.parametrize("metric", ["dev-loss", "dev-macro-f1"])
def test_train_nli_best_dev_selection_is_dev_only_with_patience(tmp_path, monkeypatch, metric):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    rows, dev = _nli_rows(), _nli_rows(split="dev")
    snapshots, seen = [], []
    original = NLI.eval_nli_metrics

    def controlled(model, tokenizer, rows_arg, **kwargs):
        metrics = original(model, tokenizer, rows_arg, **kwargs)
        snapshots.append(TRAIN.cpu_state(model))
        seen.append([row["split"] for row in rows_arg])
        index = len(snapshots) - 1
        metrics.update({"loss": [1.0, 0.2, 0.9, 0.7, 0.8, 0.85][index],
                        "macro_f1": [0.1, 0.9, 0.2, 0.3, 0.35, 0.4][index]})
        return metrics

    monkeypatch.setattr(NLI, "eval_nli_metrics", controlled)
    result = NLI.train_nli(model, tokenizer, rows, dev_rows=dev, checkpoint_selection="best-dev",
                           selection_metric=metric, patience=2, min_delta=0.05,
                           **_train_args(epochs=3, eval_every_steps=1, max_steps=0))
    selection = result["selection"]
    assert selection["selected_epoch"] == 1 and selection["selected_step"] == 2
    assert selection["best_dev_value"] == (0.2 if metric == "dev-loss" else 0.9)
    assert selection["stopped_early"] is True and selection["state_restored"] is True
    assert selection["weighting"] == "root_equal" and selection["dev_groups"] == len(dev)
    assert len(result["history"]) == 4
    assert result["budget"]["planned_updates"] == 6 and result["budget"]["actual_updates"] == 4
    assert result["budget"]["actual_sample_instances"] == 4 * 3, "4 updates × batch 3 × accum 1"
    assert result["budget"]["planned_sample_instances"] == 6 * 3
    assert all(torch.equal(tensor, snapshots[1][name]) for name, tensor in model.state_dict().items())
    assert seen == [["dev"] * len(dev)] * 4, "选择只允许看 dev 行；封存/终测不得进入 checkpoint 选择"
    assert result["truncation"]["dev"]["rows"] == len(dev)


def test_train_nli_dev_group_weighting_is_all_or_none(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    rows = _nli_rows()
    plain_dev = [{key: value for key, value in row.items() if key != "group_id"}
                 for row in _nli_rows(split="dev")]
    result = NLI.train_nli(model, tokenizer, rows, dev_rows=plain_dev, **_train_args())
    assert result["selection"]["weighting"] == "row_native"
    assert result["selection"]["dev_groups"] is None
    mixed_dev = copy.deepcopy(_nli_rows(split="dev"))
    mixed_dev[0].pop("group_id")
    with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
        NLI.train_nli(model, tokenizer, rows, dev_rows=mixed_dev, **_train_args())
    for malformed in (123, "", "   ", None, ["g"], {"g": 1}):
        broken = copy.deepcopy(_nli_rows(split="dev"))
        for row in broken:
            row["group_id"] = malformed
        with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
            NLI.train_nli(model, tokenizer, rows, dev_rows=broken, **_train_args())
    mixed_malformed = copy.deepcopy(_nli_rows(split="dev"))
    mixed_malformed[-1]["group_id"] = None
    with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
        NLI.train_nli(model, tokenizer, rows, dev_rows=mixed_malformed, **_train_args())


def test_train_nli_api_has_no_acceptance_entry():
    """验收边界：训练入口不暴露任何验收/终测开关，独立验收无法进入 fit 或选择。"""
    parameters = set(inspect.signature(NLI.train_nli).parameters)
    forbidden = {"acceptance", "accept", "accept_rows", "test", "test_rows", "heldout",
                 "heldout_rows", "challenge", "challenge_rows", "sealed", "sealed_rows",
                 "eval_splits", "final_acceptance", "audit_rows"}
    assert not parameters & forbidden
    assert {"rows", "dev_rows", "max_steps", "checkpoint_selection", "direction", "seed"} <= parameters


def test_nli_scorer_uses_shared_joint_text_and_memory_requests(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu")
    requests = [_build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0),
                _build_request("汇总必须两行。", "", "汇总必须只有一行。", 1)]
    probs = scorer.predict_proba(requests)
    assert probs.shape == (len(requests), len(NLI.NLI_CLASSES))
    assert np.allclose(probs.sum(axis=1), 1.0)
    logits = TRAIN.logits_for(model, tokenizer, requests, max_length=48, device="cpu")
    assert np.allclose(probs, FEAT.softmax(logits)), "NLI 概率必须来自共享 joint_text/logits_for 通道"
    assert scorer.predict_proba(requests[0]).shape == (1, 3)
    assert scorer.predict_proba([]).shape == (0, 3)
    assert scorer.predict([]) == []
    assert set(scorer.predict(requests)) <= set(NLI.NLI_CLASSES)
    assert scorer.classes is NLI.NLI_CLASSES
    dirty = copy.deepcopy(requests[0])
    dirty["candidate"].update({"id": "cand-secret", "label": "conflict"})
    dirty["teacher_vote"] = "supersede"
    assert np.allclose(scorer.predict_proba([dirty]), probs[:1]), "只读文本面，忽略 id/label/provenance"


def test_nli_artifact_roundtrip_hash_class_order_and_cross_contract_rejection(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    training = NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args())
    fingerprint = FEAT.model_fingerprint(directory)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, batch_size=3)
    artifact = tmp_path / "nli-artifact"
    meta = NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory,
                                 backbone_fingerprint=fingerprint,
                                 train_meta={"budget": training["budget"],
                                             "selection": training["selection"]})
    assert meta["schema"] == NLI.NLI_SCHEMA and meta["kind"] == NLI.NLI_KIND
    assert meta["classes"] == list(NLI.NLI_CLASSES) and meta["label_space"] == NLI.NLI_LABEL_SPACE
    assert meta["direction"] == NLI.NLI_PRIMARY_DIRECTION
    assert meta["joint_roles"] == NLI.NLI_JOINT_ROLES[NLI.NLI_PRIMARY_DIRECTION]
    assert meta["input_recipe"] == TRAIN.INPUT_RECIPE
    assert meta["weights"] == "full" and meta["weights_file"] == NLI.NLI_WEIGHTS_FILE
    assert meta["weights_sha256"] == TRAIN.file_sha256(artifact / NLI.NLI_WEIGHTS_FILE)
    assert meta["train"]["budget"] == training["budget"] and meta["train"]["selection"] == training["selection"]
    assert NLI.is_nli_artifact(artifact) is True and NLI.is_nli_artifact(tmp_path / "nothing") is False
    loaded = NLI.load_nli_scorer(artifact, device="cpu")
    assert loaded.classes == NLI.NLI_CLASSES and loaded.direction == NLI.NLI_PRIMARY_DIRECTION
    assert loaded.batch_size == 3 and loaded.max_length == 48
    probe = [row["request"] for row in _nli_rows(split="dev")]
    assert np.array_equal(scorer.predict_proba(probe), loaded.predict_proba(probe))
    assert NLI.check_nli_reload_parity(scorer, artifact, probe, device="cpu")["ok"] is True
                                            
    assert TRAIN.is_finetune_artifact(artifact) is False
    with pytest.raises(FileNotFoundError):
        TRAIN.load_scorer(artifact)
    with pytest.raises(FileNotFoundError):
        FEAT.load_scorer(artifact)
                                                     
    seven_model, _ = TRAIN.build_joint_model(directory)
    with pytest.raises(ValueError, match="3 类头"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=seven_model, tokenizer=tokenizer, max_length=48),
                              tmp_path / "bad-head", backbone_dir=directory,
                              backbone_fingerprint=fingerprint, train_meta={})
    with pytest.raises(ValueError, match="weights=full"):
        NLI.save_nli_artifact(scorer, tmp_path / "bad-weights", weights="head_only",
                              backbone_dir=directory, backbone_fingerprint=fingerprint, train_meta={})
    with pytest.raises(ValueError, match="方向"):
        NLI.save_nli_artifact(scorer, tmp_path / "bad-direction", backbone_dir=directory,
                              backbone_fingerprint=fingerprint, train_meta={}, direction="nli7")

    def tamper(name: str) -> "tuple[object, object]":
        target = tmp_path / name
        shutil.copytree(artifact, target)
        path = target / NLI.NLI_META_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        return target, payload

    def rewrite(directory_path, payload: dict) -> None:
        (directory_path / NLI.NLI_META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    target, payload = tamper("nli-schema")
    payload["schema"] = TRAIN.ARTIFACT_SCHEMA
    rewrite(target, payload)
    with pytest.raises(ValueError, match="schema 不符"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-class-order")
    payload["classes"] = ["entailment", "contradiction", "neutral"]
    rewrite(target, payload)
    with pytest.raises(ValueError, match="类别序"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-roles")
    payload["joint_roles"] = {"premise": "candidate", "hypothesis": "existing"}
    rewrite(target, payload)
    with pytest.raises(ValueError, match="joint_roles"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-direction")
    payload["direction"] = "premise_as_candidate"
    rewrite(target, payload)
    with pytest.raises(ValueError, match="joint_roles"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-recipe")
    payload["input_recipe"] = "memory-relations-joint-input-v2"
    rewrite(target, payload)
    with pytest.raises(ValueError, match="输入配方"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-payload-classes")
    weights_path = target / NLI.NLI_WEIGHTS_FILE
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    saved["classes"] = ["entailment", "contradiction", "neutral"]
    torch.save(saved, weights_path)
    payload["weights_sha256"] = TRAIN.file_sha256(weights_path)
    rewrite(target, payload)
    with pytest.raises(ValueError, match="载荷"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-hash")
    data = bytearray((target / NLI.NLI_WEIGHTS_FILE).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (target / NLI.NLI_WEIGHTS_FILE).write_bytes(bytes(data))
    with pytest.raises(ValueError, match="哈希不符"):
        NLI.load_nli_scorer(target)
    target, payload = tamper("nli-missing-weights")
    (target / NLI.NLI_WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        NLI.load_nli_scorer(target)
    with pytest.raises(FileNotFoundError):
        NLI.load_nli_scorer(tmp_path / "no-such-artifact")
                                  
    assert NLI.load_nli_scorer(artifact).classes == NLI.NLI_CLASSES


def test_nli_gpu_request_never_silently_falls_back(tmp_path, monkeypatch):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    artifact = tmp_path / "nli-gpu-artifact"
    NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48),
                          artifact, backbone_dir=directory,
                          backbone_fingerprint=FEAT.model_fingerprint(directory), train_meta={})
    with pytest.raises(ValueError, match="只支持 cpu 或 cuda"):
        NLI.load_nli_scorer(artifact, device="mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="不回退 CPU"):
        NLI.load_nli_scorer(artifact, device="cuda:0")
    with pytest.raises(RuntimeError, match="不回退 CPU"):
        NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cuda")
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="不回退 CPU"):
        NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args(device=torch.device("cuda")))
    assert all(torch.equal(tensor, before[name]) for name, tensor in model.state_dict().items()), \
        "GPU 请求失败必须先于任何参数更新，绝不静默在 CPU 上训练"


def test_export_backbone_writes_new_dir_only_and_stays_loadable(tmp_path):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from research.memory_relations import train as T7

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    source_fingerprint = FEAT.model_fingerprint(directory)
    initial = {name: tensor.detach().clone() for name, tensor in model.backbone.state_dict().items()}
    NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args())
    export_dir = tmp_path / "exported-nli-backbone"
    manifest = NLI.export_backbone(model, tokenizer, export_dir,
                                   direction=NLI.NLI_PRIMARY_DIRECTION)
    assert manifest["schema"] == NLI.NLI_BACKBONE_EXPORT_SCHEMA
    assert manifest["fingerprint"] == FEAT.model_fingerprint(export_dir)
    assert manifest["config"]["hidden_size"] == 32 and manifest["config"]["num_hidden_layers"] == 2
    assert manifest["config_sha256"] == NLI._config_digest(model.backbone.config)
    assert manifest["source_backbone_path"] == str(directory)
    assert manifest["tokenizer_source"]["vocab_size"] == len(tokenizer)
    assert manifest["direction"] == NLI.NLI_PRIMARY_DIRECTION
    assert NLI.NLI_WEIGHTS_FILE not in manifest["files"] and NLI.NLI_META_FILE not in manifest["files"]
    assert FEAT.model_fingerprint(directory) == source_fingerprint, "导出不得改动原骨干目录"
    exported = AutoModel.from_pretrained(str(export_dir), local_files_only=True)
    trained = model.backbone.state_dict()
    assert all(torch.equal(exported.state_dict()[name], tensor) for name, tensor in trained.items())
    assert any(not torch.equal(exported.state_dict()[name], initial[name]) for name in initial), \
        "导出的必须是被 NLI 训练更新过的 backbone，而不是原权重"
    assert AutoTokenizer.from_pretrained(str(export_dir), local_files_only=True).is_fast
    warm, _ = TRAIN.build_joint_model(export_dir)
    assert int(warm.classifier.in_features) == 32
    assert int(warm.classifier.out_features) == len(FEAT.CLASSES), "未来七类 warm-start 自建 7 类头"
    assert T7.resolve_embedding_model(str(export_dir), tmp_path) == export_dir
    assert AutoConfig.from_pretrained(str(export_dir), local_files_only=True).hidden_size == 32
    assert NLI.export_backbone(model, tokenizer, export_dir / "child")["config"]["hidden_size"] == 32
    with pytest.raises(FileExistsError, match="全新目录"):
        NLI.export_backbone(model, tokenizer, export_dir)
    with pytest.raises(FileExistsError, match="全新目录"):
        NLI.export_backbone(model, tokenizer, directory)
    empty_target = tmp_path / "empty-target"
    empty_target.mkdir()
                                       
    with pytest.raises(FileExistsError, match="全新目录"):
        NLI.export_backbone(model, tokenizer, empty_target)
    assert list(empty_target.iterdir()) == []
    with pytest.raises(ValueError, match="源骨干目录"):
        NLI.export_backbone(model, tokenizer, directory / "sub-export")
    assert not (directory / "sub-export").exists()
    assert FEAT.model_fingerprint(directory) == source_fingerprint


def test_eval_nli_metrics_empty_and_group_weighting(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    empty = NLI.eval_nli_metrics(model, tokenizer, [], max_length=48, device=torch.device("cpu"))
    assert empty["n"] == 0 and empty["loss"] is None and empty["weighting"] == "row_native"
    rows = _nli_rows(split="dev")
    plain = [{key: value for key, value in row.items() if key != "group_id"} for row in rows]
    row_native = NLI.eval_nli_metrics(model, tokenizer, plain, max_length=48,
                                      device=torch.device("cpu"))
    grouped = NLI.eval_nli_metrics(model, tokenizer, rows, max_length=48, device=torch.device("cpu"))
    assert row_native["weighting"] == "row_native" and grouped["weighting"] == "root_equal"
    assert grouped["n_groups"] == len(rows)
    assert 0.0 <= row_native["accuracy"] <= 1.0 and 0.0 <= grouped["macro_f1"] <= 1.0
    assert set(row_native["prediction_counts"]) == set(NLI.NLI_CLASSES)
    assert np.isfinite(row_native["loss"]) and np.isfinite(row_native["row_loss"])
    with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
        NLI.eval_nli_metrics(model, tokenizer, rows[:5] + [plain[5]], max_length=48,
                             device=torch.device("cpu"))
    malformed = [{**row, "group_id": None} for row in rows]
    with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
        NLI.eval_nli_metrics(model, tokenizer, malformed, max_length=48,
                             device=torch.device("cpu"))
    assert NLI.eval_nli_metrics(model, tokenizer, plain, max_length=48,
                                device=torch.device("cpu"))["weighting"] == "row_native"


def test_real_backbone_nli_head_if_available():
    embedding = _embedding_model_path()
    if embedding is None:
        pytest.skip("本地 MiniLM 权重不可见（禁止下载）；设 MEMORY_RELATIONS_EMBEDDING_MODEL 后重跑")
    from transformers import AutoTokenizer

    model, config = TRAIN.build_joint_model(embedding, num_labels=len(NLI.NLI_CLASSES))
    assert int(config.hidden_size) == 384 and int(config.num_hidden_layers) == 12
    assert int(model.classifier.in_features) == 384
    assert int(model.classifier.out_features) == len(NLI.NLI_CLASSES)
    tokenizer = AutoTokenizer.from_pretrained(str(embedding), local_files_only=True)
    rows = _nli_rows()
    result = NLI.train_nli(model, tokenizer, rows, **_train_args(max_steps=1, batch_size=3,
                                                                grad_accum=1, max_length=64))
    assert result["budget"]["actual_updates"] == 1
    assert result["truncation"]["train"]["attribution"] in ("offset_mapping",
                                                            "token_prefix_boundary")
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=64, device="cpu")
    probs = scorer.predict_proba([row["request"] for row in rows])
    assert probs.shape == (len(rows), len(NLI.NLI_CLASSES)) and np.allclose(probs.sum(axis=1), 1.0)


                                                                             
                                                       
                                                                             

def _frozen_backbone_model(model_dir, *, frozen_fraction: float):
    model, _ = TRAIN.build_joint_model(model_dir, num_labels=len(NLI.NLI_CLASSES))
    params = list(model.backbone.named_parameters())
    count = len(params) if frozen_fraction >= 1.0 else max(1, int(len(params) * frozen_fraction))
    for _, param in params[:count]:
        param.requires_grad_(False)
    return model


def _clone_with_config(tmp_path, source, *, name: str, **overrides):
    """同形状但配置不同的复制体（只改不影响力张量形状的字段），用于 config 替换类反例。"""
    from transformers import AutoConfig

    target = tmp_path / name
    shutil.copytree(source, target)
    config = AutoConfig.from_pretrained(str(target), local_files_only=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    config.save_pretrained(str(target))
    return target


def _reordered_tokenizer_dir(tmp_path, source, *, name: str):
    """词表内容重排的本地 tokenizer 目录（文件集合与词表规模不变，id 映射不同）。"""
    from transformers import BertTokenizerFast

    target = tmp_path / name
    shutil.copytree(source, target)
    lines = [line for line in (target / "vocab.txt").read_text(encoding="utf-8").splitlines() if line]
    reordered = lines[:5] + list(reversed(lines[5:]))
    (target / "vocab.txt").write_text("\n".join(reordered), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(target / "vocab.txt"), do_lower_case=True).save_pretrained(str(target))
    return target


def _artifact_copy(tmp_path, source, name: str):
    target = tmp_path / name
    shutil.copytree(source, target)
    return target


def _rewrite_meta(directory, **updates) -> dict:
    path = directory / NLI.NLI_META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(updates)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _make_junction(link, target) -> bool:
    """Windows 目录 junction（mklink /J 不需提权）；创建失败返回 False 由调用方 skip。"""
    import subprocess

    if os.name != "nt":
        return False
    completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               capture_output=True, text=True)
    return completed.returncode == 0 and link.exists()


def test_nli_head_and_hidden_are_enforced_before_any_update(tmp_path):
    """R1：非 3 类头/头宽不符/输出形状不符在训练、scorer、指标、导出各处写前 fail-loud。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    seven_model, _ = TRAIN.build_joint_model(directory)
    before = {name: tensor.detach().clone() for name, tensor in seven_model.state_dict().items()}
    with pytest.raises(ValueError, match="3 类头"):
        NLI.train_nli(seven_model, tokenizer, _nli_rows(), **_train_args())
    assert all(torch.equal(tensor, before[name]) for name, tensor in seven_model.state_dict().items()), \
        "非 3 类头必须在任何参数更新前被拒"
    with pytest.raises(ValueError, match="3 类头"):
        NLI.NLIScorer(model=seven_model, tokenizer=tokenizer, max_length=48, device="cpu")
    with pytest.raises(ValueError, match="3 类头"):
        NLI.eval_nli_metrics(seven_model, tokenizer, _nli_rows(split="dev"), max_length=48,
                             device=torch.device("cpu"))
    with pytest.raises(ValueError, match="3 类头"):
        NLI.export_backbone(seven_model, tokenizer, tmp_path / "seven-export")
    shifted, _ = TRAIN.build_joint_model(directory, num_labels=3)
    shifted.classifier = torch.nn.Linear(64, 3)
    with pytest.raises(ValueError, match="hidden_size"):
        NLI.train_nli(shifted, tokenizer, _nli_rows(), **_train_args())
    with pytest.raises(ValueError, match="hidden_size"):
        NLI.NLIScorer(model=shifted, tokenizer=tokenizer, max_length=48, device="cpu")
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu")
    scorer.model.classifier = torch.nn.Linear(int(model.classifier.in_features), len(FEAT.CLASSES))
    with pytest.raises(RuntimeError, match="输出形状"):
        scorer.predict_proba([row["request"] for row in _nli_rows()])


def test_nli_rejects_frozen_and_partially_frozen_backbones(tmp_path):
    """R2：NLI 参考臂按声明端到端；冻结/部分冻结骨干在更新前拒绝，正常运行的覆盖与梯度有记录。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    frozen = _frozen_backbone_model(directory, frozen_fraction=1.0)
    partial = _frozen_backbone_model(directory, frozen_fraction=0.25)
    for suspect in (frozen, partial):
        before = {name: tensor.detach().clone() for name, tensor in suspect.state_dict().items()}
        with pytest.raises(ValueError, match="冻结骨干参数"):
            NLI.train_nli(suspect, tokenizer, _nli_rows(), **_train_args())
        assert all(torch.equal(tensor, before[name]) for name, tensor in suspect.state_dict().items()), \
            "冻结骨干必须在更新前被拒"
        with pytest.raises(ValueError, match="冻结骨干参数"):
            NLI.export_backbone(suspect, tokenizer, tmp_path / f"export-{id(suspect)}")
    shared, _ = TRAIN.build_joint_model(directory, num_labels=3)
    TRAIN.build_optimizer("frozen-joint", shared)
    with pytest.raises(ValueError, match="冻结骨干参数"):
        NLI.train_nli(shared, tokenizer, _nli_rows(), **_train_args())
    with pytest.raises(ValueError, match="冻结骨干参数"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=shared, tokenizer=tokenizer, max_length=48),
                              tmp_path / "frozen-artifact", backbone_dir=directory, train_meta={})
    assert not (tmp_path / "frozen-artifact").exists(), "冻结骨干必须在写入前被拒"
    result = NLI.train_nli(model, tokenizer, _nli_rows(), **_train_args(max_steps=1))
    guard = result["guard"]
    assert guard["head_classes"] == len(NLI.NLI_CLASSES) and guard["hidden_size"] == 32
    assert guard["backbone_trainable"] is True and guard["full_optimizer_coverage"] is True
    assert guard["backbone_grad_seen"] is True
    assert guard["backbone_parameters"] > 0
    assert guard["optimizer_parameters"] >= guard["backbone_parameters"]


def test_nli_direction_is_derived_from_scorer(tmp_path):
    """R3：方向只从 scorer 派生；显式冲突在任何写入前拒绝，逆向臂不得被写成主向。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    reverse = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu",
                            direction="premise_as_candidate")
    artifact = tmp_path / "reverse-artifact"
    meta = NLI.save_nli_artifact(reverse, artifact, backbone_dir=directory, train_meta={"seed": 13})
    assert meta["direction"] == "premise_as_candidate"
    assert meta["joint_roles"] == NLI.NLI_JOINT_ROLES["premise_as_candidate"]
    loaded = NLI.load_nli_scorer(artifact, device="cpu")
    assert loaded.direction == reverse.direction == "premise_as_candidate"
    probe = [row["request"] for row in _nli_rows(split="dev")]
    assert NLI.check_nli_reload_parity(reverse, artifact, probe, device="cpu")["ok"] is True
    conflict = tmp_path / "conflict-artifact"
    with pytest.raises(ValueError, match="方向"):
        NLI.save_nli_artifact(reverse, conflict, backbone_dir=directory, train_meta={},
                              direction=NLI.NLI_PRIMARY_DIRECTION)
    assert not conflict.exists(), "方向冲突必须在任何写入前失败"
    forward_meta = NLI.save_nli_artifact(
        NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu"),
        tmp_path / "forward-artifact", backbone_dir=directory, train_meta={},
        direction=NLI.NLI_PRIMARY_DIRECTION)
    assert forward_meta["direction"] == NLI.NLI_PRIMARY_DIRECTION
    assert forward_meta["joint_roles"] == NLI.NLI_JOINT_ROLES[NLI.NLI_PRIMARY_DIRECTION]


def test_nli_artifact_binds_actual_config_and_hidden_size(tmp_path):
    """R4：config 按实际内容摘要绑定（同形状替换也拒），hidden 以实际 config 为准而非自述。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    artifact = tmp_path / "config-artifact"
    NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48),
                          artifact, backbone_dir=directory, train_meta={})
    decoy = _clone_with_config(tmp_path, directory, name="decoy-config",
                               num_attention_heads=8, layer_norm_eps=1e-9)
    from transformers import AutoConfig

    decoy_config = AutoConfig.from_pretrained(str(decoy), local_files_only=True)
    assert decoy_config.hidden_size == model.backbone.config.hidden_size, "反例必须同形状"
    assert NLI._config_digest(decoy_config) != NLI._config_digest(model.backbone.config)
    with pytest.raises(ValueError, match="config"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48),
                              tmp_path / "decoy-artifact", backbone_dir=decoy, train_meta={})
    assert not (tmp_path / "decoy-artifact").exists(), "config 不一致必须在写入前被拒"
    swapped = _artifact_copy(tmp_path, artifact, "config-swap")
    _rewrite_meta(swapped, backbone_path=str(decoy))
    with pytest.raises(ValueError, match="config"):
        NLI.load_nli_scorer(swapped)
    hidden_tampered = _artifact_copy(tmp_path, artifact, "hidden-tamper")
    payload = torch.load(hidden_tampered / NLI.NLI_WEIGHTS_FILE, map_location="cpu", weights_only=True)
    payload["hidden_size"] = 999
    torch.save(payload, hidden_tampered / NLI.NLI_WEIGHTS_FILE)
    _rewrite_meta(hidden_tampered, hidden_size=999,
                  weights_sha256=TRAIN.file_sha256(hidden_tampered / NLI.NLI_WEIGHTS_FILE))
    with pytest.raises(ValueError, match="hidden_size"):
        NLI.load_nli_scorer(hidden_tampered)
    assert NLI.load_nli_scorer(artifact).classes == NLI.NLI_CLASSES, "原工件仍可加载"


def _live_reordered_tokenizer(source):
    """同规模内存词表改序的 live tokenizer（保留原 name_or_path，改的是实际 id 映射）。"""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    entries = sorted(data["model"]["vocab"].items(), key=lambda pair: pair[1])
    values = [value for _, value in entries]
    tail = values[5:]
    for index in range(0, len(tail) - 1, 2):
        tail[index], tail[index + 1] = tail[index + 1], tail[index]
    data["model"]["vocab"] = {token: new_id
                              for (token, _), new_id in zip(entries, values[:5] + tail)}
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(data))
    return tokenizer


def test_nli_artifact_binds_live_tokenizer_state_and_roundtrips_exactly(tmp_path):
    """R5（r2 同族反例）：绑定 scorer 实际 tokenizer 的序列化状态，而不是来源目录 + 词表规模。

    同规模内存词表改序（保留原 name_or_path）、可序列化 normalizer 修改、公开 padding_side
    构造参数与 add_tokens 规模变化都必须精确 round-trip；无名内存态 tokenizer 写前拒绝。
    """
    from tokenizers import normalizers
    from transformers import AutoTokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    probe = [row["request"] for row in _nli_rows(split="dev")]

    def save_and_check(prefix, live):
        scorer = NLI.NLIScorer(model=model, tokenizer=live, max_length=48, device="cpu")
        artifact = tmp_path / f"{prefix}-artifact"
        meta = NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={})
        assert meta["tokenizer_dir"] == NLI.NLI_TOKENIZER_DIR
        assert (artifact / NLI.NLI_TOKENIZER_DIR).is_dir()
        reloaded = NLI.load_nli_scorer(artifact, device="cpu")
        difference = float(np.max(np.abs(scorer.predict_proba(probe) - reloaded.predict_proba(probe))))
        assert difference <= 1e-6, f"{prefix}: 实际 tokenizer 必须 round-trip 精确（旧行为会丢失实际状态）"
        assert NLI.check_nli_reload_parity(scorer, artifact, probe, device="cpu")["ok"] is True
        return meta, reloaded

    base = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    meta, _ = save_and_check("plain", base)
    assert meta["tokenizer_runtime"]["vocab_size"] == len(base)
    assert meta["tokenizer_path"] == str(directory)

    reordered_dir = _reordered_tokenizer_dir(tmp_path, directory, name="reordered-tokenizer")
    reordered_disk = AutoTokenizer.from_pretrained(str(reordered_dir), local_files_only=True)
    assert reordered_disk("一个男人正在外面走路。")["input_ids"] \
        != base("一个男人正在外面走路。")["input_ids"], "磁盘改序 tokenizer 必须真的不同"
    save_and_check("disk-reorder", reordered_disk)

    reorder = _live_reordered_tokenizer(directory)
    assert len(reorder) == len(base), "反例必须同规模"
    assert reorder.name_or_path == str(directory), "反例必须保留有效来源声明"
    assert reorder("一个男人正在外面走路。")["input_ids"] != base("一个男人正在外面走路。")["input_ids"], \
        "反例必须真的改变实际编码"
    meta, reloaded = save_and_check("mem-reorder", reorder)
    assert meta["tokenizer_path"] == str(directory), "写出处仍是实际对象的声明来源（仅供审计）"
    assert reloaded.tokenizer("一个男人正在外面走路。")["input_ids"] \
        == reorder("一个男人正在外面走路。")["input_ids"], "重载必须复现改序后的实际 ID 映射"

                                              
    moved_source = tmp_path / "moved-tokenizer"
    shutil.copytree(reordered_dir, moved_source)
    moved = AutoTokenizer.from_pretrained(str(moved_source), local_files_only=True)
    moved_scorer = NLI.NLIScorer(model=model, tokenizer=moved, max_length=48, device="cpu")
    owned_artifact = tmp_path / "owned-state-artifact"
    NLI.save_nli_artifact(moved_scorer, owned_artifact, backbone_dir=directory, train_meta={})
    (moved_source / "vocab.txt").write_text("only [PAD] [UNK] [CLS] [SEP] [MASK]", encoding="utf-8")
    (moved_source / "tokenizer.json").unlink()
    owned_reloaded = NLI.load_nli_scorer(owned_artifact, device="cpu")
    owned_difference = float(np.max(np.abs(moved_scorer.predict_proba(probe)
                                           - owned_reloaded.predict_proba(probe))))
    assert owned_difference <= 1e-6, "加载不得回读来源目录：绑定必须是工件自有的序列化状态"

    normalized = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    before_ids = normalized("一个男人正在外面走路。")["input_ids"]
    normalized.backend_tokenizer.normalizer = normalizers.Sequence([normalizers.Replace("男", "女")])
    assert normalized("一个男人正在外面走路。")["input_ids"] != before_ids, "反例必须真的改变归一化行为"
    save_and_check("normalizer", normalized)

    left = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, padding_side="left")
    assert left.padding_side == "left"
    meta, reloaded = save_and_check("padding-left", left)
    assert meta["tokenizer_runtime"]["padding_side"] == "left"
    assert reloaded.tokenizer.padding_side == "left", "加载不得把 padding_side 静默还原为默认"

    added = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    added.add_tokens(["[zzz]"])
    meta, _ = save_and_check("added", added)
    assert meta["tokenizer_runtime"]["vocab_size"] == len(added) == len(base) + 1

    nameless = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    nameless.name_or_path = ""
    with pytest.raises(ValueError, match="本地来源目录"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=nameless, max_length=48),
                              tmp_path / "nameless-artifact", backbone_dir=directory, train_meta={})
    assert not (tmp_path / "nameless-artifact").exists()


def test_nli_artifact_binds_slow_tokenizer_without_silent_fast_swap(tmp_path):
    """R5 边界：slow tokenizer 按 is_fast 对齐重建（绝不静默换成 fast 类），且精确 round-trip。"""
    from research.memory_relations.test_finetune import _slow_tokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    slow = _slow_tokenizer(directory)
    slow.name_or_path = str(directory)
    scorer = NLI.NLIScorer(model=model, tokenizer=slow, max_length=48, device="cpu")
    artifact = tmp_path / "slow-tokenizer-artifact"
    meta = NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={})
    assert meta["tokenizer_runtime"]["is_fast"] is False
    reloaded = NLI.load_nli_scorer(artifact, device="cpu")
    assert reloaded.tokenizer.is_fast is False, "slow tokenizer 不得被静默换成 fast 类"
    assert type(reloaded.tokenizer).__name__ == type(slow).__name__
    probe = [row["request"] for row in _nli_rows(split="dev")]
    assert np.array_equal(scorer.predict_proba(probe), reloaded.predict_proba(probe))


def test_nli_tokenizer_binding_rejects_unreconstructable_and_tampered_state(tmp_path):
    """R5 负向：实际状态不可重建时写前拒绝且不留半成品；工件自有 tokenizer 状态被换/被改 fail-loud。"""
    from transformers import AutoTokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu")
    artifact = tmp_path / "binding-artifact"
    NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={})

    sabotaged = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    swapped = dict(sabotaged.get_vocab())
    ordered = sorted(swapped, key=lambda token: swapped[token])
    first, second = ordered[6], ordered[7]
    swapped[first], swapped[second] = swapped[second], swapped[first]
    sabotaged.get_vocab = lambda: dict(swapped)                             
    bad = tmp_path / "unreconstructable-artifact"
    with pytest.raises(ValueError, match="精确重建"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=sabotaged, max_length=48),
                              bad, backbone_dir=directory, train_meta={})
    assert not bad.exists(), "重建互验失败不得留下半成品工件目录"

    swapped_artifact = _artifact_copy(tmp_path, artifact, "tokenizer-swap")
    replacement = tmp_path / "replacement-tokenizer"
    reorder = _live_reordered_tokenizer(directory)
    reorder.save_pretrained(str(replacement))
    shutil.rmtree(swapped_artifact / NLI.NLI_TOKENIZER_DIR)
    shutil.copytree(replacement, swapped_artifact / NLI.NLI_TOKENIZER_DIR)
    _rewrite_meta(swapped_artifact,
                  tokenizer_files_sha256=FEAT.model_fingerprint(swapped_artifact / NLI.NLI_TOKENIZER_DIR))
    with pytest.raises(ValueError, match="语义状态"):
        NLI.load_nli_scorer(swapped_artifact)

    payload_tampered = _artifact_copy(tmp_path, artifact, "payload-tokenizer-hash")
    saved = torch.load(payload_tampered / NLI.NLI_WEIGHTS_FILE, map_location="cpu", weights_only=True)
    saved["tokenizer_state_sha256"] = "0" * 64
    torch.save(saved, payload_tampered / NLI.NLI_WEIGHTS_FILE)
    _rewrite_meta(payload_tampered,
                  weights_sha256=TRAIN.file_sha256(payload_tampered / NLI.NLI_WEIGHTS_FILE))
    with pytest.raises(ValueError, match="tokenizer 状态摘要"):
        NLI.load_nli_scorer(payload_tampered)
    assert NLI.load_nli_scorer(artifact).classes == NLI.NLI_CLASSES, "原工件仍可加载"


def test_nli_artifact_save_requires_fresh_target_and_preserves_existing(tmp_path):
    """R3 生命周期：保存只写全新目录——既有目标（有效工件/空目录/普通文件/链接）一律写前拒绝，
    拒绝不得改动有效旧工件的任何字节或可加载性（绝不先覆写再拒绝）。"""
    from tokenizers import normalizers
    from transformers import AutoTokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    source_fingerprint = FEAT.model_fingerprint(directory)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu")
    artifact = tmp_path / "existing-artifact"
    NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={"seed": 13})
    before = {
        "fingerprint": FEAT.model_fingerprint(artifact),
        "weights": (artifact / NLI.NLI_WEIGHTS_FILE).read_bytes(),
        "meta": (artifact / NLI.NLI_META_FILE).read_bytes(),
        "tokenizer_files": sorted(path.name for path in (artifact / NLI.NLI_TOKENIZER_DIR).iterdir()),
    }
    assert NLI.load_nli_scorer(artifact).classes == NLI.NLI_CLASSES

                                     
    with pytest.raises(ValueError, match="全新目录"):
        NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={})
                                                  
    unsupported = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    unsupported.backend_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
    with pytest.raises(ValueError, match="全新目录"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=unsupported, max_length=48),
                              artifact, backbone_dir=directory, train_meta={})
                              
    empty = tmp_path / "empty-target"
    empty.mkdir()
    with pytest.raises(ValueError, match="全新目录"):
        NLI.save_nli_artifact(scorer, empty, backbone_dir=directory, train_meta={})
    file_target = tmp_path / "file-target"
    file_target.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="全新目录"):
        NLI.save_nli_artifact(scorer, file_target, backbone_dir=directory, train_meta={})

    assert FEAT.model_fingerprint(artifact) == before["fingerprint"]
    assert (artifact / NLI.NLI_WEIGHTS_FILE).read_bytes() == before["weights"]
    assert (artifact / NLI.NLI_META_FILE).read_bytes() == before["meta"]
    assert sorted(path.name for path in (artifact / NLI.NLI_TOKENIZER_DIR).iterdir()) \
        == before["tokenizer_files"]
    assert NLI.load_nli_scorer(artifact).classes == NLI.NLI_CLASSES, "有效旧工件必须保持可加载"
    assert list(empty.iterdir()) == [] and file_target.read_text(encoding="utf-8") == "occupied"
    assert FEAT.model_fingerprint(directory) == source_fingerprint


def test_nli_artifact_save_cleans_failed_fresh_creation(tmp_path, monkeypatch):
    """R3 生命周期：新目标上 tokenizer/checkpoint/meta 任一阶段失败都撤回整棵新目录，不留半成品。"""
    from tokenizers import normalizers
    from transformers import AutoTokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    source_fingerprint = FEAT.model_fingerprint(directory)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=48, device="cpu")

    unsupported = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    unsupported.backend_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
    tokenizer_failure = tmp_path / "tokenizer-failure-artifact"
    with pytest.raises(ValueError, match="精确重建"):
        NLI.save_nli_artifact(NLI.NLIScorer(model=model, tokenizer=unsupported, max_length=48),
                              tokenizer_failure, backbone_dir=directory, train_meta={})
    assert not tokenizer_failure.exists(), "tokenizer 阶段失败必须撤回新目标"

    original_save = torch.save

    def failing_save(payload, destination, *args, **kwargs):
        Path(destination).write_bytes(b"incomplete-checkpoint")
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr(NLI.torch, "save", failing_save)
    checkpoint_failure = tmp_path / "checkpoint-failure-artifact"
    with pytest.raises(OSError, match="injected checkpoint failure"):
        NLI.save_nli_artifact(scorer, checkpoint_failure, backbone_dir=directory, train_meta={})
    assert not checkpoint_failure.exists(), "checkpoint 写失败必须撤回新目标"
    monkeypatch.setattr(NLI.torch, "save", original_save)

    original_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self.name == NLI.NLI_META_FILE:
            original_write_text(self, "{incomplete", encoding="utf-8")
            raise OSError("injected meta failure")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    meta_failure = tmp_path / "meta-failure-artifact"
    with pytest.raises(OSError, match="injected meta failure"):
        NLI.save_nli_artifact(scorer, meta_failure, backbone_dir=directory, train_meta={})
    assert not meta_failure.exists(), "meta 写失败必须撤回新目标"
    monkeypatch.setattr(Path, "write_text", original_write_text)
    assert FEAT.model_fingerprint(directory) == source_fingerprint


def test_export_backbone_refuses_reparse_and_unresolved_provenance(tmp_path):
    """R6：junction/链接目标、已存在目录、源内路径与不可解析来源全部写前拒绝，受害者目录零写入。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    victim = tmp_path / "victim-empty"
    victim.mkdir()
    junction = tmp_path / "target-junction"
    if not _make_junction(junction, victim):
        pytest.skip("当前环境无法创建 Windows 目录 junction（cmd mklink /J 不可用）")
    assert junction.exists() and junction.is_dir()
    with pytest.raises(FileExistsError, match="全新目录"):
        NLI.export_backbone(model, tokenizer, junction)
    assert list(victim.iterdir()) == [], "junction 指向的目录不得被写入"
                                             
    with pytest.raises(ValueError, match="reparse"):
        NLI.export_backbone(model, tokenizer, junction / "sub-export")
    assert list(victim.iterdir()) == [], "经 junction 祖先的写入不得发生"
    dangling = tmp_path / "dangling-link"
    try:
        os.symlink(tmp_path / "no-such-victim", dangling, target_is_directory=True)
    except OSError:
        dangling = None
    if dangling is not None:
        with pytest.raises(FileExistsError, match="全新目录"):
            NLI.export_backbone(model, tokenizer, dangling)
    blind = copy.deepcopy(model)
    blind.backbone.config._name_or_path = ""
    with pytest.raises(ValueError, match="fail-closed"):
        NLI.export_backbone(blind, tokenizer, tmp_path / "blind-export")
    phantom = copy.deepcopy(model)
    phantom.backbone.config._name_or_path = str(tmp_path / "no-such-source")
    with pytest.raises(ValueError, match="fail-closed"):
        NLI.export_backbone(phantom, tokenizer, tmp_path / "phantom-export")
    assert not (tmp_path / "blind-export").exists() and not (tmp_path / "phantom-export").exists()
    assert junction.is_dir() and list(victim.iterdir()) == []


def test_export_backbone_source_alias_fence_and_dotdot(tmp_path):
    """R6（r2 同族反例）：源经真实 junction/alias 加载时按实际源身份围栏——真实源子树与别名子树
    都写前拒绝且源指纹不变；旁路全新目录仍可正常导出；.. 组件写前拒绝。"""
    from transformers import AutoModel, AutoTokenizer

    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    actual_source = tmp_path / "actual-source"
    shutil.copytree(directory, actual_source)
    junction = tmp_path / "source-junction"
    if not _make_junction(junction, actual_source):
        pytest.skip("当前环境无法创建 Windows 目录 junction（cmd mklink /J 不可用）")
    aliased_model, _ = TRAIN.build_joint_model(junction, num_labels=len(NLI.NLI_CLASSES))
    aliased_tokenizer = AutoTokenizer.from_pretrained(str(junction), local_files_only=True)
    assert str(aliased_model.backbone.config._name_or_path) == str(junction), \
        "反例必须经 alias 加载（name_or_path=alias）"
    alias_fingerprint = FEAT.model_fingerprint(junction)
    NLI.train_nli(aliased_model, aliased_tokenizer, _nli_rows(), **_train_args(max_steps=1))
    assert FEAT.model_fingerprint(junction) == alias_fingerprint, "训练不得改动源目录"
    for target, pattern in ((actual_source / "forbidden-nested-export", "源骨干目录"),
                            (junction / "sub-export", "reparse|源骨干目录")):
        with pytest.raises(ValueError, match=pattern):
            NLI.export_backbone(aliased_model, aliased_tokenizer, target)
        assert not target.exists()
        assert FEAT.model_fingerprint(junction) == alias_fingerprint, "拒绝路径不得改动真实源"
    outside = tmp_path / "outside-export"
    manifest = NLI.export_backbone(aliased_model, aliased_tokenizer, outside)
    assert manifest["source_backbone_path"] == str(junction)
    assert os.path.normcase(manifest["source_backbone_resolved"]) \
        == os.path.normcase(os.path.realpath(str(actual_source)))
    assert FEAT.model_fingerprint(junction) == alias_fingerprint
    exported = AutoModel.from_pretrained(str(outside), local_files_only=True)
    assert all(torch.equal(exported.state_dict()[name], tensor)
               for name, tensor in aliased_model.backbone.state_dict().items())
    with pytest.raises(ValueError, match=r"\.\."):
        NLI.export_backbone(aliased_model, aliased_tokenizer, tmp_path / "parent" / ".." / "escape")
    assert not (tmp_path / "escape").exists()


def test_nli_group_ids_malformed_vs_absent_and_root_weights(tmp_path):
    """R7：缺席=row_native、存在却畸形=fail-loud；重复根按 1/根行数等权（与 metrics 同口径）。"""
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    duplicated = copy.deepcopy(_nli_rows(split="dev"))
    for index, row in enumerate(duplicated):
        row["group_id"] = "root-a" if index < 4 else "root-b"
    probe = [row["request"] for row in duplicated]
    logits = TRAIN.logits_for(model, tokenizer, probe, max_length=48, device="cpu")
    losses = torch.nn.functional.cross_entropy(
        torch.from_numpy(logits).float(), torch.from_numpy(NLI.row_labels(duplicated)),
        reduction="none").numpy()
    weights = EM._group_weights([row["group_id"] for row in duplicated])
    assert not np.allclose(weights, 1.0), "反例必须真的根不等权（否则测不到口径）"
    grouped = NLI.eval_nli_metrics(model, tokenizer, duplicated, max_length=48,
                                   device=torch.device("cpu"))
    assert grouped["weighting"] == "root_equal" and grouped["n_groups"] == 2
    assert np.isclose(grouped["loss"], float(np.average(losses, weights=weights)))
    plain = [{key: value for key, value in row.items() if key != "group_id"} for row in duplicated]
    row_native = NLI.eval_nli_metrics(model, tokenizer, plain, max_length=48,
                                      device=torch.device("cpu"))
    assert row_native["weighting"] == "row_native" and row_native["n_groups"] is None
    assert np.isclose(row_native["loss"], float(losses.mean()))
    result = NLI.train_nli(model, tokenizer, _nli_rows(), dev_rows=duplicated, **_train_args())
    assert result["selection"]["weighting"] == "root_equal"
    assert result["selection"]["dev_groups"] == 2
    training_plain = NLI.train_nli(model, tokenizer, _nli_rows(), dev_rows=plain, **_train_args())
    assert training_plain["selection"]["weighting"] == "row_native"
    for malformed in (123, "", "   ", None, ["g"], {"g": 1}, True):
        broken = [{**row, "group_id": malformed} for row in duplicated]
        with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
            NLI.eval_nli_metrics(model, tokenizer, broken, max_length=48, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="合法非空字符串或全行不带"):
            NLI.train_nli(model, tokenizer, _nli_rows(), dev_rows=broken, **_train_args())


def test_nli_input_budget_is_bound_to_actual_scorer_payload(tmp_path):
    directory, tokenizer, model = _tiny_nli_model(tmp_path)
    scorer = NLI.NLIScorer(model=model, tokenizer=tokenizer, max_length=64)
    artifact = tmp_path / "bound-input"
    NLI.save_nli_artifact(scorer, artifact, backbone_dir=directory, train_meta={})
    payload = torch.load(artifact / NLI.NLI_WEIGHTS_FILE, map_location="cpu", weights_only=True)
    assert payload["max_length"] == 64 and payload["recipe"] == TRAIN.INPUT_RECIPE
    assert NLI.load_nli_scorer(artifact).max_length == 64
    path = artifact / NLI.NLI_META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert "input_limits" not in meta and "input_limits" not in payload
    meta["max_length"] = 56
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="max_length"):
        NLI.load_nli_scorer(artifact)
