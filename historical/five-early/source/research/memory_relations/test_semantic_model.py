"""CPU/offline mechanics, not model-quality evidence; no CUDA allocation or final-data reads."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR
from research.memory_relations import semantic_model as SM
from research.memory_relations.test_finetune import _tiny_model_dir


def canonical_rows(split="train"):
    rows = []
    for space, classes in (("nli3", SM.SR.NLI_CLASSES), ("semantic5", SM.SR.SEMANTIC_RELATIONS)):
        for index, label in enumerate(classes):
            key = f"{split}-{space}-{index}"
            rows.append({"id": key, "root_id": key, "cluster_id": key, "split": split,
                         "label_space": space, "label": label, "label_origin": "mechanics_fixture",
                         "pair": {"old": {"kind": "memory", "text": f"{split}旧命题{index}",
                                          "trigger": "仅本轮", "permanence": "temporary",
                                          "payload": {"steps": [{"verb": "read", "args": {"values": [2, 1]}},
                                                                {"verb": "write", "args": "  result  "}]}},
                                  "new": {"kind": "proposition", "text": f"新命题{index}"},
                                  "frame": {"background": "固定解释域"}}})
    return rows


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    directory = _tiny_model_dir(tmp_path, corpus=json.dumps(canonical_rows(), ensure_ascii=False))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    mother, _ = FT.build_joint_model(directory, num_labels=3)
    return directory, tokenizer, mother


def args(**override):
    result = {"updates": 2, "seed": 13, "canonical_batch": 4, "microbatch": 2, "grad_accum": 4,
              "max_length": 64, "device": torch.device("cpu"), "bf16": False,
              "dev_rows": canonical_rows("dev")}
    result.update(override)
    return result


def test_semantic_serializer_preserves_parameters_and_excludes_identity():
    row = canonical_rows()[0]
    dirty = copy.deepcopy(row)
    dirty["pair"]["old"].update({"owner_id": "not-input", "label": "not-input", "id": "not-input"})
    orientations = SM.oriented_rows([dirty])
    assert orientations[0]["text"] == SM.SR.semantic_pair_text(row["pair"])
    text = orientations[0]["text"]
    assert "not-input" not in text and "temporary" in text and "仅本轮" in text
    assert '[2,1]' in text and '  result  ' in text
    assert text.index('"read"') < text.index('"write"')
    assert orientations[1]["text"] == SM.SR.semantic_pair_text(SM.SR.swap_pair(row["pair"]))
    assert orientations[0]["label"] == row["label"] and orientations[1]["label"] is None


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_partial_gold_reverse_mask_and_source_half_means(arm):
    rows = canonical_rows()
    orientations = SM.oriented_rows(rows)
    logits = torch.randn(len(orientations), len(SM.ARM_CLASSES[arm]), requires_grad=True)
    losses, _ = SM.orientation_losses(logits, orientations, arm)
    weights = SM.source_weights(orientations)
    (losses * torch.tensor(weights)).sum().backward()
    assert torch.equal(logits.grad[1:6:2], torch.zeros_like(logits.grad[1:6:2]))
    assert np.isclose(weights[:6].sum(), 0.5) and np.isclose(weights[6:].sum(), 0.5)
    coarse = copy.deepcopy(rows[0])
    coarse["label"] = "equivalent"
    with pytest.raises(ValueError, match="coarse gold"):
        SM.validate_row(coarse)
    if arm == "c5":
        p = torch.softmax(logits.detach()[0], dim=0)
        assert float(losses[0].detach()) == pytest.approx(float(-torch.log(p[0] + p[1])), abs=1e-6)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_whole_canonical_gradient_equals_oriented_accumulation(arm):
    items = SM.oriented_rows(canonical_rows())
    weights = torch.tensor(SM.source_weights(items), dtype=torch.float64)
    full = torch.randn(len(items), len(SM.ARM_CLASSES[arm]), dtype=torch.float64, requires_grad=True)
    chunks = full.detach().clone().requires_grad_(True)
    loss, _ = SM.orientation_losses(full, items, arm)
    (loss * weights).sum().backward()
    for start in range(0, len(items), 3):
        loss, _ = SM.orientation_losses(chunks[start:start + 3], items[start:start + 3], arm)
        (loss * weights[start:start + 3]).sum().backward()
    torch.testing.assert_close(full.grad, chunks.grad, rtol=0, atol=0)


def test_initial_parent_probability_parity_and_fine_breaks_symmetry(tiny):
    _, tokenizer, mother = tiny
    c5, dual = SM.initialize_arm(mother, "c5"), SM.initialize_arm(mother, "dual3")
    pairs = [row["pair"] for row in canonical_rows()]
    left = SM.SemanticScorer(model=c5, tokenizer=tokenizer, arm="c5", max_length=64)
    right = SM.SemanticScorer(model=dual, tokenizer=tokenizer, arm="dual3", max_length=64)
    np.testing.assert_allclose(left.predict_nli_proba(pairs), right.predict_proba(pairs), rtol=0, atol=1e-6)
    assert torch.equal(c5.classifier.weight[0], c5.classifier.weight[1])
    item = SM.oriented_rows([canonical_rows()[3]])[:1]
    SM.orientation_losses(c5.classifier(torch.randn(1, c5.classifier.in_features)), item, "c5")[0].sum().backward()
    assert not torch.equal(c5.classifier.weight.grad[0], c5.classifier.weight.grad[1])


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_cpu_real_training_budget_and_full_gradient(tiny, arm):
    _, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, arm)
    before = FT.cpu_state(model)
    result = SM.train_semantic(model, tokenizer, canonical_rows(), arm=arm, **args())
    assert result["budget"]["actual_updates"] == 2
    assert result["budget"]["actual_canonical_instances"] == 8
    assert result["budget"]["actual_sample_instances"] == 16
    assert result["budget"]["supervised_directions_per_update"] == 6
    assert result["budget"]["real_tokens"] > 0 and result["guard"]["backbone_grad_seen"]
    assert all(update["microbatch_sizes"] == [2] * 4 for update in result["updates"])
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items() if key.startswith("backbone."))
    assert result["continuation"]["optimizer_state"]["state"]
    assert result["continuation"]["best_state"] and result["truncation"]["train"]["truncated_rows"] > 0


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_terminal_adam_continuation_matches_uninterrupted_and_keeps_best(tiny, tmp_path, arm, monkeypatch):
    directory, tokenizer, mother = tiny
    def controlled(*_args, **_kwargs):
        return {"loss": 0.1, "row_loss": 0.1, "macro_f1": None}
    monkeypatch.setattr(SM, "eval_semantic", controlled)
    full = SM.initialize_arm(mother, arm)
    initial = SM.initialize_arm(mother, arm)
    uninterrupted = SM.train_semantic(full, tokenizer, canonical_rows(), arm=arm, **args(updates=3))
    first = SM.train_semantic(initial, tokenizer, canonical_rows(), arm=arm, **args(updates=1))
    scorer = SM.SemanticScorer(model=initial, tokenizer=tokenizer, arm=arm, max_length=64)
    target = tmp_path / "first"
    SM.save_semantic_artifact(scorer, target, backbone_dir=directory, training=first)
    loaded, meta, payload = SM.load_semantic_artifact(target)
    assert meta["schema"] == SM.ARTIFACT_SCHEMA and not NR.is_nli_artifact(target)
    assert not FT.is_finetune_artifact(target)
    np.testing.assert_array_equal(scorer.predict_proba([canonical_rows()[0]["pair"]]),
                                  loaded.predict_proba([canonical_rows()[0]["pair"]]))
    continued = SM.train_semantic(SM.initialize_arm(mother, arm), tokenizer, canonical_rows(), arm=arm,
                                   resume_state=payload["continuation"], **args(updates=3))
    assert continued["selection"]["selected_step"] == 1
    for key in uninterrupted["continuation"]["terminal_state"]:
        torch.testing.assert_close(uninterrupted["continuation"]["terminal_state"][key],
                                   continued["continuation"]["terminal_state"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="drift"):
        SM.train_semantic(SM.initialize_arm(mother, arm), tokenizer, canonical_rows(), arm=arm,
                           resume_state=payload["continuation"], **args(updates=3, seed=424242))


def test_artifact_rejects_source_weight_tokenizer_config_drift_and_old_loaders(tiny, tmp_path, monkeypatch):
    directory, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, canonical_rows(), arm="c5", **args(updates=1))
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, arm="c5", max_length=64)
    target = tmp_path / "artifact"
    SM.save_semantic_artifact(scorer, target, backbone_dir=directory, training=training)
    for load in (NR.load_nli_scorer, FT.load_scorer):
        with pytest.raises(FileNotFoundError):
            load(target)
    with pytest.raises(ValueError, match="全新目录"):
        SM.save_semantic_artifact(scorer, target, backbone_dir=directory, training=training)
    with monkeypatch.context() as patch:
        patch.setattr(SM, "source_binding", lambda: {"semantic_model": "changed"})
        with pytest.raises(ValueError, match="source drift"):
            SM.load_semantic_artifact(target)
    weights = target / SM.WEIGHTS_FILE
    original = weights.read_bytes()
    weights.write_bytes(original + b"drift")
    with pytest.raises(ValueError, match="哈希"):
        SM.load_semantic_artifact(target)
    weights.write_bytes(original)
    vocab = target / "tokenizer" / "tokenizer_config.json"
    text = vocab.read_text(encoding="utf-8")
    vocab.write_text(text + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="指纹"):
        SM.load_semantic_artifact(target)
    vocab.write_text(text, encoding="utf-8")
    path = directory / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["layer_norm_eps"] *= 2
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        SM.load_semantic_artifact(target)


def test_sealed_final_nan_shape_and_budget_rejections(tiny):
    _, tokenizer, mother = tiny
    for key, value in (("split", "final"), ("sealed", True), ("label", None)):
        rows = canonical_rows()
        rows[0][key] = value
        with pytest.raises(ValueError):
            SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, rows, arm="c5", **args())
        dev = canonical_rows("dev")
        dev[0][key] = value
        with pytest.raises(ValueError):
            SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, canonical_rows(), arm="c5", **args(dev_rows=dev))
    orientations = SM.oriented_rows(canonical_rows())
    for logits in (torch.full((16, 5), float("nan")), torch.zeros(16, 3)):
        with pytest.raises(ValueError):
            SM.orientation_losses(logits, orientations, "c5")
    with pytest.raises(ValueError, match="budget"):
        list(SM.balanced_windows(canonical_rows(), seed=13, updates=1, microbatch=8, grad_accum=6))
    bad = SM.initialize_arm(mother, "c5")
    bad.classifier = nn.Linear(32, 7)
    with pytest.raises(ValueError, match="shape"):
        SM.train_semantic(bad, tokenizer, canonical_rows(), arm="c5", **args())


def test_prepare_fit_budget_status_cli(tiny, tmp_path, capsys):
    directory, tokenizer, mother = tiny
    mother_dir = tmp_path / "mother3"
    NR.save_nli_artifact(NR.NLIScorer(model=mother, tokenizer=tokenizer, max_length=64),
                         mother_dir, backbone_dir=directory, train_meta={"mechanics_only": True})
    raw = tmp_path / "generic.jsonl"
    raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical_rows() + canonical_rows("dev")), encoding="utf-8")
    prepared = tmp_path / "prepared"
    assert SM.main(["prepare", "--data", str(raw), "--out", str(prepared)]) == 0
    common = ["--prepared", str(prepared), "--mother3", str(mother_dir), "--updates", "1",
              "--canonical-batch", "4", "--microbatch", "2", "--grad-accum", "4", "--max-length", "64", "--tiny-cpu"]
    assert SM.main(["budget", *common]) == 0
    target = tmp_path / "fit"
    assert SM.main(["fit", *common, "--out", str(target), "--arm", "dual3", "--device", "cpu", "--bf16", "off"]) == 0
    assert SM.main(["status", "--artifact", str(target)]) == 0
    assert '"reload_max_abs_diff": 0.0' in capsys.readouterr().out
    with pytest.raises(ValueError, match="main budget"):
        SM.main(["budget", *[item for item in common if item != "--tiny-cpu"]])
    path = prepared / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash drift"):
        SM.read_prepared(prepared)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_canonical32_full_encoder_gradient_matches_micro8_accum8(tiny, arm):
    _, tokenizer, mother = tiny
    for module in mother.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, "dropout_prob"):
            module.dropout_prob = 0.0
    whole, accumulated = SM.initialize_arm(mother, arm), SM.initialize_arm(mother, arm)
    options = args(updates=1, canonical_batch=32, checkpoint_selection="final", dev_rows=())
    first = SM.train_semantic(whole, tokenizer, canonical_rows(), arm=arm,
                              **dict(options, microbatch=64, grad_accum=1))
    second = SM.train_semantic(accumulated, tokenizer, canonical_rows(), arm=arm,
                               **dict(options, microbatch=8, grad_accum=8))
    assert first["budget"]["actual_sample_instances"] == second["budget"]["actual_sample_instances"] == 64
    assert second["updates"][0]["microbatch_sizes"] == [8] * 8
    assert second["budget"]["supervised_directions_per_update"] == 48
    for (name, left), (other, right) in zip(whole.named_parameters(), accumulated.named_parameters()):
        assert name == other
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    assert second["guard"]["backbone_grad_seen"]


def test_equal_parent_distributions_have_same_proper_dev_nll(tiny, monkeypatch):
    _, tokenizer, mother = tiny
    c5, dual = SM.initialize_arm(mother, "c5"), SM.initialize_arm(mother, "dual3")
    def constant(values):
        return lambda **encoded: torch.tensor(values).repeat(len(encoded["input_ids"]), 1)
    monkeypatch.setattr(c5, "forward", constant(np.log([1/6, 1/6, 1/6, 1/3, 1/6]).tolist()))
    monkeypatch.setattr(dual, "forward", constant([0., 0., 0.]))
    options = dict(max_length=64, device=torch.device("cpu"))
    left = SM.eval_semantic(c5, tokenizer, canonical_rows("dev"), arm="c5", **options)
    right = SM.eval_semantic(dual, tokenizer, canonical_rows("dev"), arm="dual3", **options)
    assert left["loss"] == pytest.approx(right["loss"], abs=1e-6)
    assert left["loss"] == pytest.approx(np.log(3), abs=1e-6)
    assert left["diagnostics"]["fine_leaf_nll"] > left["diagnostics"]["fine_parent_nll"]


def test_loader_status_refuse_three_updates_claiming_fifty(tiny, tmp_path):
    directory, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, canonical_rows(), arm="c5", **args(updates=3, eval_every_steps=1))
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, arm="c5", max_length=64)
    artifact = tmp_path / "three"
    SM.save_semantic_artifact(scorer, artifact, backbone_dir=directory, training=training)
    meta_path = artifact / SM.META_FILE
    original = meta_path.read_text(encoding="utf-8")
    meta = json.loads(original)
    for section, field, value in (("budget", "actual_updates", 50), ("budget", "real_tokens", 1),
                                   ("budget", "actual_sample_instances", 400), ("selection", "selected_step", 50),
                                   ("training_binding", "seed", 424242)):
        edited = copy.deepcopy(meta)
        edited["train"][section][field] = value
        meta_path.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(ValueError, match="training ledger"):
            SM.main(["status", "--artifact", str(artifact)])
        bad_report = copy.deepcopy(edited["train"])
        with pytest.raises(ValueError, match="ledger"):
            SM.validate_training_ledger(bad_report, training["continuation"])
    meta_path.write_text(original, encoding="utf-8")
    assert SM.load_semantic_artifact(artifact)[2]["continuation"]["completed_steps"] == 3


@pytest.mark.parametrize("seed", SM.SEEDS)
def test_fine_roots_uniform_before_variant_sampling(seed):
    rows = canonical_rows()
    for variant in range(9):
        rows.append(dict(rows[-1], id=f"variant-{variant}"))
    counts = {}
    for _, _, chunks, _ in SM.balanced_windows(rows, seed=seed, updates=100):
        indexes = np.concatenate(chunks)[::2] // 2
        for index in indexes:
            row = rows[index]
            if row["label_space"] == "semantic5":
                counts[row["root_id"]] = counts.get(row["root_id"], 0) + 1
    assert len(counts) == 5 and set(counts.values()) == {320}


@pytest.mark.parametrize("reverse", [False, True])
def test_renamed_semantic_pair_cannot_cross_fit_dev(reverse):
    fit, dev = canonical_rows(), canonical_rows("dev")
    dev[0]["pair"] = SM.SR.swap_pair(fit[0]["pair"]) if reverse else fit[0]["pair"]
    with pytest.raises(ValueError, match="pair leakage"):
        SM.checked_pools(fit, dev)


def test_source_frozen_at_start_and_resume_without_payload_rejected(tiny, monkeypatch):
    _, tokenizer, mother = tiny
    original = SM.source_binding()
    calls = []
    def drift():
        calls.append(1)
        return original if len(calls) == 1 else dict(original, semantic_model="changed")
    monkeypatch.setattr(SM, "source_binding", drift)
    with pytest.raises(ValueError, match="during training"):
        SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, canonical_rows(), arm="c5", **args(updates=1))
    monkeypatch.setattr(SM, "source_binding", lambda: original)
    with pytest.raises(ValueError, match="continuation missing"):
        SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, canonical_rows(), arm="c5", resume_state={}, **args(updates=1))
    monkeypatch.setattr(SM, "load_semantic_artifact", lambda *a, **k: (None, {}, {}))
    with pytest.raises(ValueError, match="cold start"):
        SM.resume_continuation("does-not-matter")


def test_cli_default_grid_and_real_clipping(tiny, monkeypatch):
    _, tokenizer, mother = tiny
    parsed = SM.build_parser().parse_args(["fit", "--prepared", "p", "--mother3", "m", "--out", "o", "--arm", "c5"])
    assert (parsed.updates, parsed.eval_every_steps, parsed.clip_grad_norm) == (50, 25, 1.0)
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(parameters, norm, **kwargs):
        calls.append(norm)
        return original(parameters, norm, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    result = SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, canonical_rows(), arm="c5",
                                **args(updates=50, eval_every_steps=25))
    assert [item["step"] for item in result["history"]] == [25, 50]
    assert calls == [1.0] * 50
    assert all(update["clipped_gradient_norm"] <= 1 + 1e-6 for update in result["updates"])
    SM.validate_training_ledger({k: v for k, v in result.items() if k != "continuation"}, result["continuation"])


def test_raw_initializer_and_producer_membership_hash_binding(tiny, tmp_path):
    directory, _, _ = tiny
    mother_path = tmp_path / "raw-mother"
    raw = SM.initialize_untrained_mother(directory, mother_path, seed=13, max_length=64)
    assert raw["train"]["initialization"]["optimizer_updates"] == 0
    assert raw["train"]["initialization"]["kind"] == "raw-backbone-untrained"
    rows = canonical_rows() + canonical_rows("dev")
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assets = []
    for role, split in (("protected-fit", "train"), ("protected-dev", "dev"), ("protection-manifest", "audit")):
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps([row["id"] for row in rows if row["split"] == split and row["label_space"] == "nli3"]
                                   if split != "audit" else {"fixture": True}), encoding="utf-8")
        assets.append({"role": role, "path": str(path), "sha256": FT.file_sha256(path)})
    manifest = tmp_path / "producer.json"
    manifest.write_text(json.dumps({"schema": "semantic-producer-assets-v1", "producer": "mechanics_fixture", "assets": assets}), encoding="utf-8")
    prepared = tmp_path / "prepared"
    meta = SM.prepare(data, prepared, producer_manifest=manifest)
    assert meta["producer_lineage"]["assets"]["protected-fit"]["sha256"] == assets[0]["sha256"]
    SM.read_prepared(prepared)
    bad_rows = copy.deepcopy(rows)
    bad_rows[0]["source_id"] = "not-protected-member"
    with pytest.raises(ValueError, match="member set"):
        SM.producer_lineage(manifest, bad_rows)
    Path(assets[0]["path"]).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="asset drift"):
        SM.read_prepared(prepared)


def test_shared_views_masks_lean_export_and_raw_visible_hashes(tiny):
    _, tokenizer, mother = tiny
    rows = canonical_rows()
    rows[0]["pair"]["old"]["text"] = "alpha beta " * 600
    oriented = SM.oriented_rows(rows, tokenizer, 64)
    for forward, reverse in zip(oriented[::2], oriented[1::2]):
        assert forward["prepared"] is reverse["prepared"]
        assert forward["input_view"]["visible_information_sha256"] == reverse["input_view"]["visible_information_sha256"]
        assert forward["input_view"]["role_body_covered"] == reverse["input_view"]["role_body_covered"] == {"old": True, "new": True}
        assert len(forward["encoding"]["input_ids"]) <= 64 and len(reverse["encoding"]["input_ids"]) <= 64
        if forward["source"] == "public":
            assert reverse["loss_mask"] == 0 and reverse["label"] is None
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, arm="c5", max_length=64)
    prepared = scorer.prepare_inputs([rows[0]["pair"]])[0]
    lean = SM.SR.prepare_semantic_input(rows[0]["pair"], tokenizer, max_length=64, role_independent_frame=True)
    assert prepared == lean
    assert prepared["raw_pair_sha256"] != prepared["directions"]["forward"]["view_sha256"]
    both = scorer.predict_both_proba([rows[0]["pair"]])
    assert both["forward"].shape == both["reverse"].shape == (1, 5)
    np.testing.assert_allclose(scorer.predict_nli_proba([rows[0]["pair"]])[0],
                               SM.SR.project_nli_probabilities(both["forward"][0].tolist()), atol=1e-6, rtol=0)


def test_visible_recipe_drift_and_initializer_cannot_resume_or_claim_updates(tiny, tmp_path):
    directory, tokenizer, mother = tiny
    raw = {"kind": "raw-backbone-untrained", "optimizer_updates": 0, "seed": 13}
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, arm="c5", max_length=64)
    target = tmp_path / "initializer"
    meta = SM.save_semantic_initializer(scorer, target, backbone_dir=directory, mother_initialization=raw)
    reloaded = SM.load_semantic_artifact(target)[0]
    np.testing.assert_array_equal(scorer.predict_proba([canonical_rows()[0]["pair"]]), reloaded.predict_proba([canonical_rows()[0]["pair"]]))
    with pytest.raises(ValueError, match="continuation"):
        SM.resume_continuation(target)
    meta_path = target / SM.META_FILE
    original = meta_path.read_text(encoding="utf-8")
    edited = json.loads(original)
    edited["input_recipe"] = "semantic-pair-json-v1"
    meta_path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="配方"):
        SM.load_semantic_artifact(target)
    meta_path.write_text(original, encoding="utf-8")
    bad = copy.deepcopy(meta["train"])
    bad["budget"]["actual_updates"] = 50
    with pytest.raises(ValueError, match="initializer"):
        SM.validate_training_ledger(bad, None)
    data = tmp_path / "raw.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows() + canonical_rows("dev")), encoding="utf-8")
    prepared = tmp_path / "prepared-recipe"
    SM.prepare(data, prepared)
    document = json.loads((prepared / "prepare.json").read_text(encoding="utf-8"))
    document["input_recipe"] = "semantic-pair-json-v1"
    (prepared / "prepare.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe"):
        SM.read_prepared(prepared)


def test_semantic_inference_budget_cannot_tamper_training_certificate(tiny, tmp_path):
    directory, tokenizer, mother = tiny
    fit, dev = canonical_rows(), canonical_rows("dev")
    for rows, old, new in ((fit, "旧", "新"), (dev, "命", "题")):
        for row in rows:
            row["pair"] = {"old": {"kind": "proposition", "text": old}, "new": {"kind": "proposition", "text": new}}
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, fit, arm="c5", **args(updates=1, dev_rows=dev))
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, arm="c5", max_length=64)
    artifact = tmp_path / "trained-budget"
    SM.save_semantic_artifact(scorer, artifact, backbone_dir=directory, training=training)
    assert SM.load_semantic_artifact(artifact)[0].max_length == 64
    meta_path = artifact / SM.META_FILE
    original = meta_path.read_text(encoding="utf-8")
    meta = json.loads(original)
    meta["max_length"] = 56
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="max_length"):
        SM.main(["status", "--artifact", str(artifact)])
    meta_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="scorer input budget"):
        SM.validate_training_ledger({k: v for k, v in training.items() if k != "continuation"}, training["continuation"],
                                    input_params={"recipe": SM.INPUT_RECIPE, "max_length": 56})
    with pytest.raises(ValueError, match="scorer input budget"):
        SM.save_semantic_artifact(SM.SemanticScorer(model=model, tokenizer=tokenizer, arm="c5", max_length=56),
                                   tmp_path / "different-scorer", backbone_dir=directory, training=training)


def qualified_consumer_fixture(tmp_path):
    from research.memory_relations.test_semantic_data import row as data_row
    D = SM.D
    root = tmp_path.resolve() / "qualification-fixture"
    root.mkdir()
    rows, qualifications, reviews, scope, packet, allocation, public, entries = [], [], [], [], [], [], [], []
    instructions = "mechanics fixture only; review the complete forward and candidate inverse pairs"
    reviewer = {"id": "mechanics-reviewer", "provider": "offline-fixture", "model": "fixture", "family": "fixture", "revision": "1"}
    for split in ("train", "dev"):
        for index, label in enumerate(SM.SR.SEMANTIC_RELATIONS):
            original = data_row(f"fixture-{split}-{index}", label=label)
            original["pair"]["old"]["text"] = f"旧{split}甲命题{index}。"
            original["pair"]["new"]["text"] = f"新{split}乙命题{index}。"
            D.validate_row(original)
            blind = "blind-" + original["id"]
            entry = {"blind_id": blind, "forward_pair": D.student_pair(original),
                     "candidate_inverse_pair": SM.SR.swap_pair(D.student_pair(original)), "instructions": instructions}
            input_sha = D.digest(entry)
            entry["input_sha256"] = input_sha
            entries.append(entry)
            review = {"blind_id": blind, "input_sha256": input_sha, "reviewer": reviewer,
                      "review": {"inverse_decision": "accept", "forward_label": label,
                                 "inverse_label": SM.SR.swap_relation(label), "actually_reversed": True,
                                 "same_domain": True, "nl_fidelity": True, "complete_semantics": True,
                                 "role_reference_conflict": False, "reason": "mechanics fixture only, not actual60 quality"}}
            encoded = json.dumps(review, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            review_sha = __import__("hashlib").sha256(encoded).hexdigest()
            reviews.append(encoded)
            scope.append({"agent": "fixture-owner", "item": "fixture-task", "line": len(reviews), "line_sha256": review_sha})
            qualifications.append({"id": original["id"], "kind": "explicit_inverse", "inverse_pair": entry["candidate_inverse_pair"],
                                   "source_row_sha256": D.digest(original), "review_kind": "mechanics_fixture", "review_record_sha256": review_sha})
            packet.append({"source_id": original["id"], "blind_id": blind, "input_sha256": input_sha,
                           "forward_pair_sha256": D.digest(entry["forward_pair"]),
                           "candidate_inverse_pair_sha256": D.digest(entry["candidate_inverse_pair"]),
                           "source_row_sha256": D.digest(original), "label": label, "root_id": original["root_id"], "cluster_id": original["cluster_id"]})
            allocation.append({"source_id": original["id"], "source_row_sha256": D.digest(original),
                               "root_id": original["root_id"], "cluster_id": original["cluster_id"], "stage_split": split})
            rows.append(original)
        for index, label in enumerate(SM.SR.NLI_CLASSES):
            identifier = f"public-{split}-{index}"
            public.append({"id": identifier, "root_id": identifier, "cluster_id": identifier, "split": split,
                           "label_space": "nli3", "label": label, "label_origin": "mechanics_fixture",
                           "pair": {"old": {"kind": "proposition", "text": f"公开{split}旧句{index}"},
                                    "new": {"kind": "proposition", "text": f"公开{split}新句{index}"}}})
    def dump_json(name, value):
        path = root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path
    def dump_rows(name, values):
        path = root / name
        path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
        return path
    source = dump_rows("original.jsonl", rows)
    qualifications_path = dump_rows("qualifications.jsonl", qualifications)
    reviews_path = root / "reviews.jsonl"
    reviews_path.write_bytes(b"\n".join(reviews) + b"\n")
    scope_path = dump_json("scope.json", {"occurrences": scope})
    packet_path = dump_json("packet.json", {"items": packet, "reviewer": reviewer})
    import_path = dump_json("import.json", {"scope_corrected": True, "files_sha256": {"qualifications.jsonl": FT.file_sha256(qualifications_path)}})
    blind_path = dump_rows("blind.jsonl", entries)
    blind_manifest = dump_json("blind-manifest.json", {"shards_sha256": {"fixture-task": FT.file_sha256(blind_path)},
                                                       "instructions_sha256": D.digest(instructions)})
    paths = {"source_rows": source, "qualifications": qualifications_path, "reviews": reviews_path,
             "review_scope": scope_path, "packet_map": packet_path, "import_manifest": import_path,
             "blind_manifest": blind_manifest}
    document = {"schema": "semantic-qualified-source-v2", "assets": {key: {"path": str(path), "sha256": FT.file_sha256(path)} for key, path in paths.items()},
                "blind_shards": {"fixture-task": {"path": str(blind_path), "sha256": FT.file_sha256(blind_path)}},
                "source_boundary": {"schema": "semantic-source-boundary-v1", "tasks": [
                    {"owner_id": "fixture-owner", "task_id": "fixture-task", "member_ids": [item["blind_id"] for item in packet]}]}}
    manifest = dump_json("fine-manifest.json", document)
    allocated = dump_json("allocation.json", {"schema": "semantic-source-allocation-v1", "items": allocation})
    public_path = dump_rows("public.jsonl", public)
    return {"root": root, "source": rows, "manifest": manifest, "allocation": allocated, "public": public_path,
            "document": document, "paths": paths, "qualifications": qualifications, "entries": entries}


def test_qualified_source_allocation_keeps_raw_hashes_and_shared_masks(tmp_path):
    fixture = qualified_consumer_fixture(tmp_path)
    bundle = SM.D.load_qualified_source(fixture["manifest"])
    assert all(row["split"] == "pilot" for row in bundle["rows"])
    assert len(bundle["rows"]) == 10
    derived, lineage = SM.qualified_stage_rows(fixture["manifest"], fixture["allocation"])
    originals = {row["id"]: row for row in fixture["source"]}
    for row in derived:
        original = originals[row["id"]]
        assert row["source_binding"]["evidence"]["source_row_sha256"] == SM.D.digest(original)
        assert row["source_binding"]["evidence"]["source_row_sha256"] != SM.D.digest(row)
        assert row["pair"] == original["pair"] and row["label"] == original["label"]
        assert row["root_id"] == original["root_id"] and row["cluster_id"] == original["cluster_id"]
    out = tmp_path / "prepared"
    SM.prepare(fixture["public"], out, fine_manifest=fixture["manifest"], allocation=fixture["allocation"])
    fit, dev = SM.read_prepared(out)
    oriented = SM.oriented_rows(fit + dev)
    for left, right in zip(oriented[::2], oriented[1::2]):
        assert right["loss_mask"] == int(left["source"] == "fine")
        assert right["label"] is None if left["source"] == "public" else right["label"] is not None
    assert lineage["qualified_source"]["proxy_only"]


def test_qualified_consumer_rejects_unqualified_stage_and_fake_evidence(tmp_path):
    fixture = qualified_consumer_fixture(tmp_path)
    original = fixture["source"][0]
    with pytest.raises(ValueError, match="qualification"):
        SM.validate_row(dict(original, split="train"), stage="fit")
    derived, _ = SM.qualified_stage_rows(fixture["manifest"], fixture["allocation"])
    for field, value in (("split", "dev"), ("label", "disjoint"), ("id", "washed-id")):
        edited = copy.deepcopy(derived[0])
        edited[field] = value
        with pytest.raises(ValueError, match="drift"):
            SM.validate_row(edited, stage="fit")
    document = copy.deepcopy(fixture["document"])
    document["source_boundary"]["tasks"][0]["owner_id"] = "out-of-scope"
    fixture["manifest"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SM.D.SemanticDataError, match="authorized_review"):
        SM.D.load_qualified_source(fixture["manifest"])
    fixture["manifest"].write_text(json.dumps(fixture["document"]), encoding="utf-8")
    qualification = copy.deepcopy(fixture["qualifications"])
    qualification[0]["review_record_sha256"] = "0" * 64
    fixture["paths"]["qualifications"].write_text("".join(json.dumps(item) + "\n" for item in qualification), encoding="utf-8")
    document = copy.deepcopy(fixture["document"])
    document["assets"]["qualifications"]["sha256"] = FT.file_sha256(fixture["paths"]["qualifications"])
    import_data = {"scope_corrected": True, "files_sha256": {"qualifications.jsonl": document["assets"]["qualifications"]["sha256"]}}
    fixture["paths"]["import_manifest"].write_text(json.dumps(import_data), encoding="utf-8")
    document["assets"]["import_manifest"]["sha256"] = FT.file_sha256(fixture["paths"]["import_manifest"])
    fixture["manifest"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SM.D.SemanticDataError, match="review_bytes_drift"):
        SM.D.load_qualified_source(fixture["manifest"])


def test_qualified_fixture_tiny_two_arms_share_effective_sources(tmp_path):
    import shutil
    fixture = qualified_consumer_fixture(tmp_path)
    prepared = tmp_path / "prepared-tiny"
    SM.prepare(fixture["public"], prepared, fine_manifest=fixture["manifest"], allocation=fixture["allocation"])
    fit, dev = SM.read_prepared(prepared)
    directory = _tiny_model_dir(tmp_path, corpus=json.dumps(fit + dev, ensure_ascii=False), name="qualified-tiny")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    mother, _ = FT.build_joint_model(directory, num_labels=3)
    results = {}
    for arm in ("c5", "dual3"):
        model = SM.initialize_arm(mother, arm)
        options = dict(updates=2, seed=13, canonical_batch=32, microbatch=8, grad_accum=8,
                       max_length=64, device=torch.device("cpu"), bf16=False, eval_every_steps=1,
                       prepared_directory=prepared, mechanics_only=True)
        with pytest.raises(ValueError, match="formal training"):
            SM.train_semantic(model, tokenizer, fit, arm=arm, dev_rows=dev, **dict(options, mechanics_only=False))
        results[arm] = SM.train_semantic(model, tokenizer, fit, arm=arm, dev_rows=dev, **options)
        scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, arm=arm, max_length=64)
        artifact = tmp_path / (arm + "-qualified-2")
        SM.save_semantic_artifact(scorer, artifact, backbone_dir=directory, training=results[arm])
        loaded, meta, payload = SM.load_semantic_artifact(artifact)
        assert meta["train"]["budget"]["qualified_fine_only"] is True
        assert meta["train"]["purpose"] == "mechanics-fixture" and meta["train"]["mechanics_only"] is True
        assert meta["input_limits"] == payload["input_limits"] == meta["train"]["input_limits"] == dict(SM.INPUT_LIMITS)
        assert meta["train"]["training_binding"]["input_limits"] == meta["train"]["truncation"]["train"]["input_limits"] == dict(SM.INPUT_LIMITS)
        np.testing.assert_array_equal(scorer.predict_rows(dev[:2]), loaded.predict_rows(dev[:2]))
        if arm == "c5":
            for fault, message in (("empty", "fine evidence"), ("missing_member", "fine evidence"),
                                   ("invented_batch", "participating"), ("data_digest", "immutable prepared"),
                                   ("input_limits", "input limits")):
                invalid = copy.deepcopy(results[arm])
                binding = invalid["training_binding"]
                if fault == "empty":
                    binding["fine_evidence"] = []
                    invalid["budget"]["qualification_manifest_sha256"] = []
                    invalid["budget"]["allocation_sha256"] = []
                elif fault == "missing_member":
                    binding["fine_evidence"] = binding["fine_evidence"][1:]
                elif fault == "invented_batch":
                    invalid["source_batch_order"][0]["source_ids"][0] = "invented-member"
                    invalid["source_batch_order"][0]["root_ids"][0] = "invented-root"
                    invalid["source_batch_order"][0]["member_sha256"] = "0" * 64
                elif fault == "input_limits":
                    binding["input_limits"] = dict(SM.SR.SEMANTIC_LIMITS_PRODUCT)
                else:
                    binding["data_sha256"] = "0" * 64
                invalid["continuation"]["training_binding"] = copy.deepcopy(binding)
                invalid["continuation"]["source_batch_order"] = copy.deepcopy(invalid["source_batch_order"])
                invalid["budget"]["source_batch_order_sha256"] = SM._digest(invalid["source_batch_order"])
                with pytest.raises(ValueError, match=message):
                    SM.save_semantic_artifact(scorer, tmp_path / (fault + "-save-rejected"), backbone_dir=directory, training=invalid)
                tampered = tmp_path / (fault + "-load-rejected")
                shutil.copytree(artifact, tampered)
                changed_meta, changed_payload = copy.deepcopy(meta), copy.deepcopy(payload)
                changed_meta["train"] = {key: value for key, value in invalid.items() if key != "continuation"}
                changed_payload["continuation"] = invalid["continuation"]
                changed_meta["train_sha256"] = changed_payload["train_sha256"] = NR._tokenizer_state_sha256(changed_meta["train"])
                torch.save(changed_payload, tampered / SM.WEIGHTS_FILE)
                changed_meta["weights_sha256"] = FT.file_sha256(tampered / SM.WEIGHTS_FILE)
                (tampered / SM.META_FILE).write_text(json.dumps(changed_meta), encoding="utf-8")
                with pytest.raises(ValueError, match=message):
                    SM.load_semantic_artifact(tampered)
    assert results["c5"]["source_batch_order"] == results["dual3"]["source_batch_order"]
    assert results["c5"]["training_binding"]["train_input_hashes"] == results["dual3"]["training_binding"]["train_input_hashes"]
    for result in results.values():
        assert result["budget"]["supervised_directions_per_update"] == 48
        assert result["budget"]["actual_sample_instances"] == 128
        assert result["budget"]["source_weights"] == {"public": 0.5, "fine": 0.5}
        assert len(result["training_binding"]["fine_evidence"]) == 10
        assert all(len(batch["source_ids"]) == len(batch["root_ids"]) == 32 for batch in result["source_batch_order"])


def test_qualified_initializer_zero_does_not_require_fine_certificate(tiny, tmp_path):
    directory, tokenizer, mother = tiny
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, arm="c5", max_length=64)
    artifact = tmp_path / "initializer-zero"
    SM.save_semantic_initializer(scorer, artifact, backbone_dir=directory,
                                  mother_initialization={"kind": "raw-backbone-untrained", "optimizer_updates": 0})
    _, meta, payload = SM.load_semantic_artifact(artifact)
    assert meta["train"]["budget"]["actual_updates"] == 0 and "continuation" not in payload
    assert "fine_evidence" not in meta["train"] and "training_binding" not in meta["train"]


def test_qualified_source_rejects_same_label_blind_source_substitution(tmp_path):
    fixture = qualified_consumer_fixture(tmp_path)
    D, paths = SM.D, fixture["paths"]
    document = copy.deepcopy(fixture["document"])
    packet = D.read_json(paths["packet_map"])
    qualifications = D.load_rows(paths["qualifications"])
    blind_sha = document["blind_shards"]["fixture-task"]["sha256"]
    review_sha = FT.file_sha256(paths["reviews"])
    left, right = 0, 5
    assert packet["items"][left]["label"] == packet["items"][right]["label"]
    assert fixture["entries"][left]["forward_pair"] != fixture["entries"][right]["forward_pair"]
    for field in ("source_id", "source_row_sha256", "root_id", "cluster_id", "forward_pair_sha256", "candidate_inverse_pair_sha256"):
        packet["items"][left][field], packet["items"][right][field] = packet["items"][right][field], packet["items"][left][field]
    qualifications[left]["review_record_sha256"], qualifications[right]["review_record_sha256"] = qualifications[right]["review_record_sha256"], qualifications[left]["review_record_sha256"]
    paths["packet_map"].write_text(json.dumps(packet), encoding="utf-8")
    paths["qualifications"].write_text("".join(json.dumps(row) + "\n" for row in qualifications), encoding="utf-8")
    paths["import_manifest"].write_text(json.dumps({"scope_corrected": True, "files_sha256": {
        "qualifications.jsonl": FT.file_sha256(paths["qualifications"])}}), encoding="utf-8")
    for key, path in paths.items():
        document["assets"][key]["sha256"] = FT.file_sha256(path)
    fixture["manifest"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(D.SemanticDataError, match="actual_blind_source_pair_drift"):
        D.load_qualified_source(fixture["manifest"])
    assert FT.file_sha256(paths["reviews"]) == review_sha
    assert FT.file_sha256(document["blind_shards"]["fixture-task"]["path"]) == blind_sha


@pytest.mark.parametrize("fault", ["extra_top", "missing_top", "extra_review", "missing_reason", "empty_reason", "long_reason",
                                   "numeric_bool", "bad_label", "bad_inverse", "null_accept", "false_accept", "role_conflict",
                                   "extra_identity", "missing_identity", "wrong_identity", "wrong_input"])
def test_strict_inverse_review_validator(tmp_path, fault):
    fixture = qualified_consumer_fixture(tmp_path)
    D = SM.D
    receipt = D.load_rows(fixture["paths"]["reviews"])[0]
    identity = copy.deepcopy(receipt["reviewer"])
    if fault == "extra_top":
        receipt["extra"] = True
    elif fault == "missing_top":
        del receipt["input_sha256"]
    elif fault == "extra_review":
        receipt["review"]["extra"] = True
    elif fault == "missing_reason":
        del receipt["review"]["reason"]
    elif fault == "empty_reason":
        receipt["review"]["reason"] = "  "
    elif fault == "long_reason":
        receipt["review"]["reason"] = "甲" * 2001
    elif fault == "numeric_bool":
        receipt["review"]["actually_reversed"] = 1
    elif fault == "bad_label":
        receipt["review"]["forward_label"] = "entailment"
    elif fault == "bad_inverse":
        receipt["review"]["inverse_label"] = "disjoint"
    elif fault == "null_accept":
        receipt["review"]["forward_label"] = None
    elif fault == "false_accept":
        receipt["review"]["nl_fidelity"] = False
    elif fault == "role_conflict":
        receipt["review"]["role_reference_conflict"] = True
    elif fault == "extra_identity":
        receipt["reviewer"]["extra"] = "fixture"
    elif fault == "missing_identity":
        del receipt["reviewer"]["revision"]
    elif fault == "wrong_identity":
        receipt["reviewer"]["model"] = "different-model"
    else:
        receipt["input_sha256"] = "0" * 64
    with pytest.raises(D.SemanticDataError):
        D.validate_inverse_review(receipt, identity=identity, entry=fixture["entries"][0])


@pytest.mark.parametrize("fault", ["pair", "label", "row", "frame_inverse", "unknown_status", "missing_reason", "missing_bytes"])
def test_qualified_source_producer_evidence_faults_rejected(tmp_path, fault):
    fixture = qualified_consumer_fixture(tmp_path)
    D = SM.D
    document = copy.deepcopy(fixture["document"])
    paths = fixture["paths"]
    if fault in ("pair", "label", "row"):
        rows = D.load_rows(paths["source_rows"])
        if fault == "pair":
            rows[0]["pair"]["old"]["text"] += "漂移"
        elif fault == "label":
            rows[0]["label"] = "disjoint"
        else:
            rows[0]["NL_fidelity_origin"]["kind"] = "different-origin"
        paths["source_rows"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        document["assets"]["source_rows"]["sha256"] = FT.file_sha256(paths["source_rows"])
    elif fault == "frame_inverse":
        qualifiers = D.load_rows(paths["qualifications"])
        qualifiers[0]["inverse_pair"]["frame"]["background"] += "改frame"
        paths["qualifications"].write_text("".join(json.dumps(row) + "\n" for row in qualifiers), encoding="utf-8")
        document["assets"]["qualifications"]["sha256"] = FT.file_sha256(paths["qualifications"])
        paths["import_manifest"].write_text(json.dumps({"scope_corrected": True, "files_sha256": {
            "qualifications.jsonl": document["assets"]["qualifications"]["sha256"]}}), encoding="utf-8")
        document["assets"]["import_manifest"]["sha256"] = FT.file_sha256(paths["import_manifest"])
    elif fault in ("unknown_status", "missing_reason"):
        records = D.load_rows(paths["reviews"])
        if fault == "missing_reason":
            del records[0]["review"]["reason"]
        else:
            records[0]["review"]["inverse_decision"] = "qualified"
        lines = [json.dumps(row).encode("utf-8") for row in records]
        paths["reviews"].write_bytes(b"\n".join(lines) + b"\n")
        document["assets"]["reviews"]["sha256"] = FT.file_sha256(paths["reviews"])
        hashes = [__import__("hashlib").sha256(line).hexdigest() for line in lines]
        scope = {"occurrences": [{"agent": "fixture-owner", "item": "fixture-task", "line": index+1,
                                  "line_sha256": value} for index, value in enumerate(hashes)]}
        paths["review_scope"].write_text(json.dumps(scope), encoding="utf-8")
        document["assets"]["review_scope"]["sha256"] = FT.file_sha256(paths["review_scope"])
        qualifications = D.load_rows(paths["qualifications"])
        for qualification, value in zip(qualifications, hashes):
            qualification["review_record_sha256"] = value
        paths["qualifications"].write_text("".join(json.dumps(row) + "\n" for row in qualifications), encoding="utf-8")
        document["assets"]["qualifications"]["sha256"] = FT.file_sha256(paths["qualifications"])
        paths["import_manifest"].write_text(json.dumps({"scope_corrected": True, "files_sha256": {
            "qualifications.jsonl": document["assets"]["qualifications"]["sha256"]}}), encoding="utf-8")
        document["assets"]["import_manifest"]["sha256"] = FT.file_sha256(paths["import_manifest"])
    else:
        paths["reviews"].write_bytes(b"")
        document["assets"]["reviews"]["sha256"] = FT.file_sha256(paths["reviews"])
    fixture["manifest"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(D.SemanticDataError):
        D.load_qualified_source(fixture["manifest"])


def test_prediction_none_label_does_not_pretend_qualification_and_requires_frame_declaration(tiny):
    _, tokenizer, mother = tiny
    row = {"id": "prediction", "root_id": "prediction", "cluster_id": "prediction", "split": "inference",
           "label_space": "semantic5", "label": None, "label_origin": "unknown",
           "pair": {"old": {"kind": "proposition", "text": "旧命题"},
                    "new": {"kind": "proposition", "text": "新命题"},
                    "frame": {"background": "未知是否角色相关的背景"}}}
    assert SM.validate_row(row)["label"] is None
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, arm="c5", max_length=64)
    with pytest.raises(SM.SR.SemanticRelationError, match="明确声明"):
        scorer.prepare_inputs([row["pair"]])
    prepared = scorer.prepare_inputs([row["pair"]], role_independent_frame=True)[0]
    assert prepared["use"] == "prediction_not_formal_qualification" and prepared["inference_frame_declaration"] is True


def test_research_input_limits_public_prepare_readback_and_visible_pair(tmp_path, tiny):
    _, tokenizer, _ = tiny
    row = {"id": "long-public", "root_id": "long-public", "cluster_id": "long-public", "split": "train",
           "label_space": "nli3", "label": "entailment", "label_origin": "mechanics_fixture",
           "pair": {"old": {"kind": "proposition", "text": "旧" * 60000},
                    "new": {"kind": "proposition", "text": "新" * 7000}}}
    dev = dict(row, id="short-dev", root_id="short-dev", cluster_id="short-dev", split="dev",
               pair={"old": {"kind": "proposition", "text": "甲"}, "new": {"kind": "proposition", "text": "乙"}})
    assert SM.validate_row(row)["pair"]["old"]["text"] == row["pair"]["old"]["text"]
    data = tmp_path / "long-public.jsonl"
    data.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in (row, dev)), encoding="utf-8")
    output = tmp_path / "long-prepared"
    meta = SM.prepare(data, output)
    fit, observed_dev = SM.read_prepared(output)
    assert meta["input_limits"] == dict(SM.SR.SEMANTIC_LIMITS_RESEARCH)
    assert fit[0]["pair"]["old"]["text"] == row["pair"]["old"]["text"] and observed_dev[0]["id"] == dev["id"]
    assert len(SM.oriented_rows(fit)[0]["text"]) > 65536
    prepared = SM.prepare_input_pairs([fit[0]["pair"]], tokenizer, 64)[0]
    assert prepared["input_limits"] == dict(SM.INPUT_LIMITS)
    assert prepared["directions"]["forward"]["visible_information_sha256"] == prepared["directions"]["reverse"]["visible_information_sha256"]
    assert all(view["token_length"] <= 64 for view in prepared["directions"].values())
