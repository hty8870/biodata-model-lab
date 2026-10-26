"""CPU/offline mechanics, not model-quality evidence; no CUDA allocation or final-data reads."""
from __future__ import annotations

import copy
import io
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR
from research.memory_relations import semantic_model as SM
from research.memory_relations.test_finetune import _tiny_model_dir


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_loader_releases_arm_probe_metadata_before_shared_loader(tmp_path, monkeypatch, arm):
    class TrackedMetadata(dict):
        pass

    path = tmp_path / SM.META_FILE
    path.write_text(json.dumps({"bindings": {"arm": arm}, "train": {"history": [{"loss": 0.25}]}}), encoding="utf-8")
    original_loads = json.loads
    references, calls = [], []
    result = (object(), {}, {})
    expected_binding = SM.artifact_binding(arm)

    def tracked_loads(*args, **kwargs):
        meta = TrackedMetadata(original_loads(*args, **kwargs))
        references.append(weakref.ref(meta))
        return meta

    def shared_loader(directory, *, spec, device, bindings):
        assert len(references) == 1 and references[0]() is None
        assert directory == tmp_path and device == "cpu"
        assert spec["schema"] == SM.ARTIFACT_SCHEMA and spec["classes"] == SM.ARM_CLASSES[arm]
        assert bindings == expected_binding
        calls.append(directory)
        return result

    monkeypatch.setattr(SM.json, "loads", tracked_loads)
    monkeypatch.setattr(NR, "load_joint_artifact", shared_loader)
    assert SM.load_semantic_artifact(tmp_path) is result
    assert calls == [tmp_path]


@pytest.mark.parametrize("newline", [None, "\n", "\r\n"])
def test_cli_stream_json_matches_legacy_print(monkeypatch, newline):
    result = {"文字": "中文 café ☕ 𠮷", "nested": [{"values": [None, True, False, -0.0, 1.25, 1e-12, 1e20]}, {}, []],
              "lines": "first\r\nsecond\n\tlast"}
    expected = io.BytesIO()
    legacy = io.TextIOWrapper(expected, encoding="utf-8", newline=newline)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), file=legacy)
    legacy.flush()
    actual = io.BytesIO()
    stdout = io.TextIOWrapper(actual, encoding="utf-8", newline=newline)
    original_dumps = json.dumps

    def no_whole_result(value, *args, **kwargs):
        if value is result:
            raise AssertionError("CLI must not allocate a complete JSON result string")
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(SM, "prepare", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(SM.json, "dumps", no_whole_result)
    with monkeypatch.context() as patch:
        patch.setattr("sys.stdout", stdout)
        assert SM.main(["prepare", "--data", "unused.jsonl", "--out", "unused-directory"]) == 0
    stdout.flush()
    assert actual.getvalue() == expected.getvalue()
    assert actual.getvalue().endswith(b"\n")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cli_stream_json_rejects_nonfinite(monkeypatch, capsys, value):
    monkeypatch.setattr(SM, "prepare", lambda *_args, **_kwargs: {"value": value})
    with pytest.raises(ValueError, match="Out of range float"):
        SM.main(["prepare", "--data", "unused.jsonl", "--out", "unused-directory"])
    capsys.readouterr()


def canonical_rows(split="train"):
    rows = []
    for space, classes in (("nli3", SM.SR.NLI_CLASSES), ("semantic5", SM.SR.SEMANTIC_RELATIONS)):
        for index, label in enumerate(classes):
            key = f"{split}-{space}-{index}"
            rows.append({"id": key, "root_id": key, "cluster_id": key, "split": split,
                         "label_space": space, "label": label, "label_origin": "mechanics_fixture",
                         "frame_role_independent": True,
                         "pair": {"old": {"kind": "memory", "text": f"{split}旧命题{index}",
                                          "trigger": "仅本轮", "permanence": "temporary",
                                          "payload": {"steps": [{"verb": "read", "args": {"values": [2, 1]}},
                                                                {"verb": "write", "args": "  result  "}]}},
                                  "new": {"kind": "proposition", "text": f"新命题{index}"},
                                  "frame": {"background": "固定解释域"}}})
    return rows


def prepared_canonical_fixture(tmp_path, *, fit=None, dev=None, producer_manifest=None):
    fit = canonical_rows() if fit is None else list(fit)
    dev = canonical_rows("dev") if dev is None else list(dev)
    fixture = qualified_consumer_fixture(tmp_path, canonical=fit + dev)
    prepared = tmp_path / "prepared-canonical"
    meta = SM.prepare(fixture["public"], prepared, fine_manifest=fixture["manifest"],
                      allocation=fixture["allocation"], producer_manifest=producer_manifest)
    fit, dev = SM.read_prepared(prepared)
    return dict(fixture, prepared=prepared, prepared_meta=meta, fit=fit, dev=dev)


@pytest.fixture
def qualified_canonical(tmp_path):
    return prepared_canonical_fixture(tmp_path)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    corpus = json.dumps([canonical_rows(), "未知是否角色相关的背景"], ensure_ascii=False)
    directory = _tiny_model_dir(tmp_path, corpus=corpus)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    tokenizer.add_tokens(["train", "dev", "alpha", "beta", "same_domain"])
    tokenizer.save_pretrained(str(directory))
    backbone = AutoModel.from_pretrained(str(directory), local_files_only=True)
    backbone.resize_token_embeddings(len(tokenizer))
    backbone.save_pretrained(str(directory))
    mother, _ = FT.build_joint_model(directory, num_labels=3)
    return directory, tokenizer, mother


def args(fixture, **override):
    result = {"updates": 2, "seed": 13, "canonical_batch": 4, "microbatch": 2, "grad_accum": 4,
              "max_length": 64, "device": torch.device("cpu"), "bf16": False,
              "dev_rows": fixture["dev"], "prepared_directory": fixture["prepared"], "mechanics_only": True,
              "producer_lineage": fixture["prepared_meta"]["producer_lineage"]}
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
def test_partial_gold_reverse_mask_and_source_half_means(arm, qualified_canonical):
    rows = qualified_canonical["fit"]
    with pytest.raises(ValueError, match="qualification"):
        SM.oriented_rows([canonical_rows()[3]])
    orientations = SM.oriented_rows(rows)
    logits = torch.randn(len(orientations), len(SM.ARM_CLASSES[arm]), requires_grad=True)
    losses, _ = SM.orientation_losses(logits, orientations, arm)
    weights = SM.source_weights(orientations)
    (losses * torch.tensor(weights)).sum().backward()
    assert torch.equal(logits.grad[1:6:2], torch.zeros_like(logits.grad[1:6:2]))
    assert np.isclose(weights[:6].sum(), 0.5) and np.isclose(weights[6:].sum(), 0.5)
    assert all(item["loss_mask"] == 1 for item in orientations[6:])
    coarse = copy.deepcopy(rows[0])
    coarse["label"] = "equivalent"
    with pytest.raises(ValueError, match="coarse gold"):
        SM.validate_row(coarse)
    if arm == "c5":
        p = torch.softmax(logits.detach()[0], dim=0)
        assert float(losses[0].detach()) == pytest.approx(float(-torch.log(p[0] + p[1])), abs=1e-6)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_whole_canonical_gradient_equals_oriented_accumulation(arm, qualified_canonical):
    items = SM.oriented_rows(qualified_canonical["fit"])
    weights = torch.tensor(SM.source_weights(items), dtype=torch.float64)
    full = torch.randn(len(items), len(SM.ARM_CLASSES[arm]), dtype=torch.float64, requires_grad=True)
    chunks = full.detach().clone().requires_grad_(True)
    loss, _ = SM.orientation_losses(full, items, arm)
    (loss * weights).sum().backward()
    for start in range(0, len(items), 3):
        loss, _ = SM.orientation_losses(chunks[start:start + 3], items[start:start + 3], arm)
        (loss * weights[start:start + 3]).sum().backward()
    torch.testing.assert_close(full.grad, chunks.grad, rtol=0, atol=0)


def test_initial_parent_probability_parity_and_fine_breaks_symmetry(tiny, qualified_canonical):
    _, tokenizer, mother = tiny
    c5, dual = SM.initialize_arm(mother, "c5"), SM.initialize_arm(mother, "dual3")
    pairs = [row["pair"] for row in qualified_canonical["fit"]]
    left = SM.SemanticScorer(model=c5, tokenizer=tokenizer, arm="c5", max_length=64)
    right = SM.SemanticScorer(model=dual, tokenizer=tokenizer, arm="dual3", max_length=64)
    np.testing.assert_allclose(left.predict_nli_proba(pairs, role_independent_frame=True),
                               right.predict_proba(pairs, role_independent_frame=True), rtol=0, atol=1e-6)
    assert torch.equal(c5.classifier.weight[0], c5.classifier.weight[1])
    item = SM.oriented_rows([qualified_canonical["fit"][3]])[:1]
    SM.orientation_losses(c5.classifier(torch.randn(1, c5.classifier.in_features)), item, "c5")[0].sum().backward()
    assert not torch.equal(c5.classifier.weight.grad[0], c5.classifier.weight.grad[1])


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_cpu_real_training_budget_and_full_gradient(tiny, arm, qualified_canonical):
    _, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, arm)
    before = FT.cpu_state(model)
    result = SM.train_semantic(model, tokenizer, qualified_canonical["fit"], arm=arm, **args(qualified_canonical))
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
def test_terminal_adam_continuation_matches_uninterrupted_and_keeps_best(tiny, tmp_path, arm, monkeypatch, qualified_canonical):
    directory, tokenizer, mother = tiny
    evaluate = SM.eval_semantic
    def controlled(*_args, **_kwargs):
        result = evaluate(*_args, **_kwargs)
        return dict(result, loss=0.1, row_loss=0.1)
    monkeypatch.setattr(SM, "eval_semantic", controlled)
    full = SM.initialize_arm(mother, arm)
    initial = SM.initialize_arm(mother, arm)
    fit = qualified_canonical["fit"]
    options = args(qualified_canonical, updates=3, eval_every_steps=1)
    uninterrupted = SM.train_semantic(full, tokenizer, fit, arm=arm, **options)
    first = SM.train_semantic(initial, tokenizer, fit, arm=arm, **dict(options, updates=1))
    assert first["history"][0]["dev_diagnostics"]["parent_nll_by_root"]
    scorer = SM.SemanticScorer(model=initial, tokenizer=tokenizer, arm=arm, max_length=64)
    target = tmp_path / "first"
    SM.save_semantic_artifact(scorer, target, backbone_dir=directory, training=first)
    loaded, meta, payload = SM.load_semantic_artifact(target)
    assert meta["schema"] == SM.ARTIFACT_SCHEMA and not NR.is_nli_artifact(target)
    assert not FT.is_finetune_artifact(target)
    assert meta["train"]["history"] == payload["continuation"]["history"] == first["history"]
    np.testing.assert_array_equal(scorer.predict_proba([fit[0]["pair"]], role_independent_frame=True),
                                  loaded.predict_proba([fit[0]["pair"]], role_independent_frame=True))
    continued = SM.train_semantic(SM.initialize_arm(mother, arm), tokenizer, fit, arm=arm,
                                   resume_state=payload["continuation"], **options)
    assert continued["selection"]["selected_step"] == 1
    assert [record["step"] for record in continued["history"]] == [1, 2, 3]
    assert continued["history"][:len(first["history"])] == first["history"]
    for actual, expected in zip(continued["history"], uninterrupted["history"]):
        assert actual["dev_diagnostics"] == expected["dev_diagnostics"]
    for key in uninterrupted["continuation"]["terminal_state"]:
        torch.testing.assert_close(uninterrupted["continuation"]["terminal_state"][key],
                                   continued["continuation"]["terminal_state"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="drift"):
        SM.train_semantic(SM.initialize_arm(mother, arm), tokenizer, fit, arm=arm,
                           resume_state=payload["continuation"], **dict(options, seed=424242))


def test_artifact_rejects_source_weight_tokenizer_config_drift_and_old_loaders(tiny, tmp_path, monkeypatch, qualified_canonical):
    directory, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, qualified_canonical["fit"], arm="c5", **args(qualified_canonical, updates=1))
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
    path = target / "config" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["layer_norm_eps"] *= 2
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        SM.load_semantic_artifact(target)


def test_sealed_final_nan_shape_and_budget_rejections(tiny, qualified_canonical):
    _, tokenizer, mother = tiny
    for key, value in (("split", "final"), ("sealed", True), ("label", None)):
        rows = copy.deepcopy(qualified_canonical["fit"])
        rows[0][key] = value
        with pytest.raises(ValueError):
            SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, rows, arm="c5", **args(qualified_canonical))
        dev = copy.deepcopy(qualified_canonical["dev"])
        dev[0][key] = value
        with pytest.raises(ValueError):
            SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, qualified_canonical["fit"], arm="c5",
                               **args(qualified_canonical, dev_rows=dev))
    orientations = SM.oriented_rows(qualified_canonical["fit"])
    for logits in (torch.full((16, 5), float("nan")), torch.zeros(16, 3)):
        with pytest.raises(ValueError):
            SM.orientation_losses(logits, orientations, "c5")
    with pytest.raises(ValueError, match="budget"):
        list(SM.balanced_windows(qualified_canonical["fit"], seed=13, updates=1, microbatch=8, grad_accum=6))
    bad = SM.initialize_arm(mother, "c5")
    bad.classifier = nn.Linear(32, 7)
    with pytest.raises(ValueError, match="shape"):
        SM.train_semantic(bad, tokenizer, qualified_canonical["fit"], arm="c5", **args(qualified_canonical))


def test_prepare_fit_budget_status_cli(tiny, tmp_path, capsys):
    directory, tokenizer, mother = tiny
    mother_dir = tmp_path / "mother3"
    NR.save_nli_artifact(NR.NLIScorer(model=mother, tokenizer=tokenizer, max_length=64),
                         mother_dir, backbone_dir=directory, train_meta={"mechanics_only": True})
    raw = tmp_path / "generic.jsonl"
    rows = canonical_rows() + canonical_rows("dev")
    raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="public nli3 only"):
        SM.main(["prepare", "--data", str(raw), "--out", str(tmp_path / "unqualified")])
    fixture = qualified_consumer_fixture(tmp_path, canonical=rows)
    prepared = tmp_path / "prepared"
    assert SM.main(["prepare", "--data", str(fixture["public"]), "--fine-manifest", str(fixture["manifest"]),
                    "--allocation", str(fixture["allocation"]), "--out", str(prepared)]) == 0
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
def test_canonical32_full_encoder_gradient_matches_micro8_accum8(tiny, arm, tmp_path):
    _, tokenizer, mother = tiny
    fixture = prepared_canonical_fixture(tmp_path, dev=())
    for module in mother.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, "dropout_prob"):
            module.dropout_prob = 0.0
    whole, accumulated = SM.initialize_arm(mother, arm), SM.initialize_arm(mother, arm)
    options = args(fixture, updates=1, canonical_batch=32, checkpoint_selection="final")
    first = SM.train_semantic(whole, tokenizer, fixture["fit"], arm=arm,
                              **dict(options, microbatch=64, grad_accum=1))
    second = SM.train_semantic(accumulated, tokenizer, fixture["fit"], arm=arm,
                               **dict(options, microbatch=8, grad_accum=8))
    assert first["budget"]["actual_sample_instances"] == second["budget"]["actual_sample_instances"] == 64
    assert second["updates"][0]["microbatch_sizes"] == [8] * 8
    assert second["budget"]["supervised_directions_per_update"] == 48
    for (name, left), (other, right) in zip(whole.named_parameters(), accumulated.named_parameters()):
        assert name == other
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    assert second["guard"]["backbone_grad_seen"]


def test_equal_parent_distributions_have_same_proper_dev_nll(tiny, monkeypatch, qualified_canonical):
    _, tokenizer, mother = tiny
    c5, dual = SM.initialize_arm(mother, "c5"), SM.initialize_arm(mother, "dual3")
    def constant(values):
        return lambda **encoded: torch.tensor(values).repeat(len(encoded["input_ids"]), 1)
    monkeypatch.setattr(c5, "forward", constant(np.log([1/6, 1/6, 1/6, 1/3, 1/6]).tolist()))
    monkeypatch.setattr(dual, "forward", constant([0., 0., 0.]))
    options = dict(max_length=64, device=torch.device("cpu"))
    left = SM.eval_semantic(c5, tokenizer, qualified_canonical["dev"], arm="c5", **options)
    right = SM.eval_semantic(dual, tokenizer, qualified_canonical["dev"], arm="dual3", **options)
    assert left["loss"] == pytest.approx(right["loss"], abs=1e-6)
    assert left["loss"] == pytest.approx(np.log(3), abs=1e-6)
    assert left["diagnostics"]["fine_leaf_nll"] > left["diagnostics"]["fine_parent_nll"]
    fields = ("source", "root_id", "cluster_id", "weight", "n_rows", "n_directions")
    left_units = left["diagnostics"]["parent_nll_by_root"]
    right_units = right["diagnostics"]["parent_nll_by_root"]
    assert [[row[key] for key in fields] for row in left_units] == [[row[key] for key in fields] for row in right_units]
    np.testing.assert_allclose([row["loss"] for row in left_units], [row["loss"] for row in right_units], atol=1e-6, rtol=0)


@pytest.mark.parametrize("arm", ["c5", "dual3"])
def test_dev_parent_nll_observations_keep_root_means_masks_and_weights(tiny, tmp_path, monkeypatch, arm):
    _, tokenizer, mother = tiny
    dev = canonical_rows("dev")
    for row in dev:
        row["cluster_id"] = "cluster-" + row["root_id"]
    dev[4]["cluster_id"] = dev[3]["cluster_id"]
    for index in (0, 3):
        for variant in range(2):
            dev.append(dict(dev[index], id=f"{dev[index]['id']}-variant-{variant}"))
    rows = prepared_canonical_fixture(tmp_path, dev=dev)["dev"]
    for index in (0, 3):
        masked = copy.deepcopy(rows[index])
        masked.update(id="masked-" + masked["label_space"], root_id="masked-" + masked["label_space"],
                      cluster_id="masked-" + masked["label_space"], label=None)
        masked.pop("source_binding", None)
        rows.append(masked)
    orientations = SM.oriented_rows(rows, tokenizer, 64)
    weights = SM.source_weights(orientations, root_equal=True)
    probabilities = []
    for index in range(len(orientations)):
        parent = np.asarray([index + 1, index % 4 + 2, len(orientations) - index + 1], dtype=np.float64)
        parent /= parent.sum()
        probabilities.append([parent[0] / 2, parent[0] / 2, parent[1] / 2, parent[2], parent[1] / 2]
                             if arm == "c5" else parent.tolist())
    logits = torch.tensor(np.log(probabilities), dtype=torch.float32)
    losses, hits = SM.joint_parent_losses(logits, orientations, arm)
    model = SM.initialize_arm(mother, arm)
    batches, offset = [], 0
    def forward(**encoded):
        nonlocal offset
        count = len(encoded["input_ids"])
        batches.append(count)
        result = logits[offset:offset + count]
        offset += count
        return result
    monkeypatch.setattr(model, "forward", forward)
    result = SM.eval_semantic(model, tokenizer, rows, arm=arm, max_length=64,
                              device=torch.device("cpu"), batch_size=3)
    units = result["diagnostics"]["parent_nll_by_root"]
    assert sum(batches) == len(orientations) and len(batches) == (len(orientations) + 2) // 3
    expected = {}
    for index, row in enumerate(rows):
        if row["label"] is not None:
            source = "fine" if row["label_space"] == "semantic5" else "public"
            expected.setdefault((source, row["root_id"]), []).append(index)
    assert {(unit["source"], unit["root_id"]) for unit in units} == set(expected)
    canonical_losses = losses.numpy().astype(np.float64).reshape(-1, 2)
    for unit in units:
        indexes = expected[(unit["source"], unit["root_id"])]
        oriented_indexes = [index for index, row in enumerate(orientations)
                            if (row["source"], row["root_id"]) == (unit["source"], unit["root_id"]) and row["loss_mask"]]
        row_losses = canonical_losses[indexes].mean(axis=1) if unit["source"] == "fine" else canonical_losses[indexes, 0]
        assert unit["loss"] == pytest.approx(float(row_losses.mean()), abs=1e-12)
        assert unit["cluster_id"] == rows[indexes[0]]["cluster_id"]
        assert unit["weight"] == float(weights[oriented_indexes].sum())
        assert unit["n_rows"] == len(indexes)
        assert unit["n_directions"] == len(oriented_indexes) == len(indexes) * (2 if unit["source"] == "fine" else 1)
    for source in ("fine", "public"):
        assert sum(unit["weight"] for unit in units if unit["source"] == source) == pytest.approx(0.5)
    assert result["loss"] == float(np.dot(weights, losses.numpy()))
    assert result["accuracy"] == float(np.dot(weights, hits.numpy()))
    assert sum(unit["weight"] * unit["loss"] for unit in units) == pytest.approx(result["loss"], abs=1e-12)
    assert json.loads(json.dumps(result["diagnostics"])) == result["diagnostics"]


def test_loader_status_refuse_three_updates_claiming_fifty(tiny, tmp_path, qualified_canonical):
    directory, tokenizer, mother = tiny
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, qualified_canonical["fit"], arm="c5",
                                 **args(qualified_canonical, updates=3, eval_every_steps=1))
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
def test_fine_roots_uniform_before_variant_sampling(seed, tmp_path):
    rows = canonical_rows()
    for variant in range(9):
        rows.append(dict(rows[-1], id=f"variant-{variant}"))
    rows = prepared_canonical_fixture(tmp_path, fit=rows, dev=())["fit"]
    counts = {}
    for _, _, chunks, _ in SM.balanced_windows(rows, seed=seed, updates=100):
        indexes = np.concatenate(chunks)[::2] // 2
        for index in indexes:
            row = rows[index]
            if row["label_space"] == "semantic5":
                counts[row["root_id"]] = counts.get(row["root_id"], 0) + 1
    assert len(counts) == 5 and set(counts.values()) == {320}


@pytest.mark.parametrize("reverse", [False, True])
def test_renamed_semantic_pair_cannot_cross_fit_dev(reverse, qualified_canonical):
    fit, dev = copy.deepcopy(qualified_canonical["fit"]), copy.deepcopy(qualified_canonical["dev"])
    dev[0]["pair"] = SM.SR.swap_pair(fit[0]["pair"]) if reverse else fit[0]["pair"]
    with pytest.raises(ValueError, match="pair leakage"):
        SM.checked_pools(fit, dev)


def test_source_frozen_at_start_and_resume_without_payload_rejected(tiny, monkeypatch, qualified_canonical):
    _, tokenizer, mother = tiny
    original = SM.source_binding()
    changed = [False]
    def drift():
        return dict(original, semantic_model="changed") if changed[0] else original
    def during_fit(_event):
        changed[0] = True
    monkeypatch.setattr(SM, "source_binding", drift)
    with pytest.raises(ValueError, match="during training"):
        SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, qualified_canonical["fit"], arm="c5",
                           **args(qualified_canonical, updates=1, event_callback=during_fit))
    monkeypatch.setattr(SM, "source_binding", lambda: original)
    with pytest.raises(ValueError, match="continuation missing"):
        SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, qualified_canonical["fit"], arm="c5",
                           resume_state={}, **args(qualified_canonical, updates=1))
    monkeypatch.setattr(SM, "load_semantic_artifact", lambda *a, **k: (None, {}, {}))
    with pytest.raises(ValueError, match="cold start"):
        SM.resume_continuation("does-not-matter")


def test_cli_default_grid_and_real_clipping(tiny, monkeypatch, qualified_canonical):
    _, tokenizer, mother = tiny
    parsed = SM.build_parser().parse_args(["fit", "--prepared", "p", "--mother3", "m", "--out", "o", "--arm", "c5"])
    assert (parsed.updates, parsed.eval_every_steps, parsed.clip_grad_norm) == (50, 25, 1.0)
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(parameters, norm, **kwargs):
        calls.append(norm)
        return original(parameters, norm, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    result = SM.train_semantic(SM.initialize_arm(mother, "c5"), tokenizer, qualified_canonical["fit"], arm="c5",
                                **args(qualified_canonical, updates=50, eval_every_steps=25))
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
    fixture = qualified_consumer_fixture(tmp_path, canonical=rows)
    assets = []
    for role, split in (("protected-fit", "train"), ("protected-dev", "dev"), ("protection-manifest", "audit")):
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps([row["id"] for row in rows if row["split"] == split and row["label_space"] == "nli3"]
                                   if split != "audit" else {"fixture": True}), encoding="utf-8")
        assets.append({"role": role, "path": str(path), "sha256": FT.file_sha256(path)})
    manifest = tmp_path / "producer.json"
    manifest.write_text(json.dumps({"schema": "semantic-producer-assets-v1", "producer": "mechanics_fixture", "assets": assets}), encoding="utf-8")
    prepared = tmp_path / "prepared"
    meta = SM.prepare(fixture["public"], prepared, producer_manifest=manifest,
                      fine_manifest=fixture["manifest"], allocation=fixture["allocation"])
    assert meta["producer_lineage"]["assets"]["protected-fit"]["sha256"] == assets[0]["sha256"]
    SM.read_prepared(prepared)
    bad_rows = copy.deepcopy(rows)
    bad_rows[0]["source_id"] = "not-protected-member"
    with pytest.raises(ValueError, match="member set"):
        SM.producer_lineage(manifest, bad_rows)
    Path(assets[0]["path"]).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="asset drift"):
        SM.read_prepared(prepared)


def test_shared_views_masks_lean_export_and_raw_visible_hashes(tiny, qualified_canonical):
    _, tokenizer, mother = tiny
    rows = copy.deepcopy(qualified_canonical["fit"])
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
    prepared = scorer.prepare_inputs([rows[0]["pair"]], role_independent_frame=True)[0]
    lean = SM.SR.prepare_semantic_input(rows[0]["pair"], tokenizer, max_length=64,
                                       role_independent_frame=True, limits=SM.INPUT_LIMITS)
    assert prepared == dict(lean, inference_frame_declaration=True, use="prediction_not_formal_qualification")
    assert not prepared["complete"]
    assert prepared["raw_pair_sha256"] != prepared["directions"]["forward"]["view_sha256"]
    short = {"old": {"kind": "memory", "text": "alpha"}, "new": {"kind": "memory", "text": "beta"}}
    full = SM.prepare_input_pairs([short], tokenizer, 384, frame_declarations=[True])[0]
    assert full["complete"] and full["raw_pair_sha256"] == full["directions"]["forward"]["view_sha256"]
    assert full["directions"]["forward"]["text"] == SM.SR.semantic_pair_text(short)
    both = scorer.predict_both_proba([rows[0]["pair"]], role_independent_frame=True)
    assert both["forward"].shape == both["reverse"].shape == (1, 5)
    np.testing.assert_allclose(scorer.predict_nli_proba([rows[0]["pair"]], role_independent_frame=True)[0],
                               SM.SR.project_nli_probabilities(both["forward"][0].tolist()), atol=1e-6, rtol=0)


@pytest.mark.parametrize("old_recipe", ["semantic-pair-json-v1", "semantic-role-balanced-fragments-v2"])
def test_visible_recipe_drift_and_initializer_cannot_resume_or_claim_updates(tiny, tmp_path, old_recipe):
    directory, tokenizer, mother = tiny
    raw = {"kind": "raw-backbone-untrained", "optimizer_updates": 0, "seed": 13}
    scorer = SM.SemanticScorer(model=SM.initialize_arm(mother, "c5"), tokenizer=tokenizer, arm="c5", max_length=64)
    target = tmp_path / "initializer"
    meta = SM.save_semantic_initializer(scorer, target, backbone_dir=directory, mother_initialization=raw)
    reloaded = SM.load_semantic_artifact(target)[0]
    np.testing.assert_array_equal(scorer.predict_proba([canonical_rows()[0]["pair"]], role_independent_frame=True),
                                  reloaded.predict_proba([canonical_rows()[0]["pair"]], role_independent_frame=True))
    with pytest.raises(ValueError, match="continuation"):
        SM.resume_continuation(target)
    meta_path = target / SM.META_FILE
    original = meta_path.read_text(encoding="utf-8")
    edited = json.loads(original)
    edited["input_recipe"] = old_recipe
    meta_path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="配方"):
        SM.load_semantic_artifact(target)
    meta_path.write_text(original, encoding="utf-8")
    bad = copy.deepcopy(meta["train"])
    bad["budget"]["actual_updates"] = 50
    with pytest.raises(ValueError, match="initializer"):
        SM.validate_training_ledger(bad, None)
    fixture = qualified_consumer_fixture(tmp_path, canonical=canonical_rows() + canonical_rows("dev"))
    prepared = tmp_path / "prepared-recipe"
    SM.prepare(fixture["public"], prepared, fine_manifest=fixture["manifest"], allocation=fixture["allocation"])
    document = json.loads((prepared / "prepare.json").read_text(encoding="utf-8"))
    document["input_recipe"] = old_recipe
    (prepared / "prepare.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe"):
        SM.read_prepared(prepared)


def test_semantic_inference_budget_cannot_tamper_training_certificate(tiny, tmp_path):
    directory, tokenizer, mother = tiny
    fit, dev = canonical_rows(), canonical_rows("dev")
    for rows, old, new in ((fit, "旧", "新"), (dev, "命", "题")):
        for row in rows:
            row["pair"] = {"old": {"kind": "proposition", "text": old}, "new": {"kind": "proposition", "text": new},
                           "frame": {"background": "固定解释域"}}
    fixture = prepared_canonical_fixture(tmp_path, fit=fit, dev=dev)
    model = SM.initialize_arm(mother, "c5")
    training = SM.train_semantic(model, tokenizer, fixture["fit"], arm="c5", **args(fixture, updates=1))
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


def qualified_consumer_fixture(tmp_path, *, canonical=None):
    from research.memory_relations.test_semantic_data import row as data_row
    D = SM.D
    root = tmp_path.resolve() / "qualification-fixture"
    root.mkdir()
    rows, qualifications, reviews, scope, packet, allocation, public, entries = [], [], [], [], [], [], [], []
    instructions = "mechanics fixture only; review the complete forward and candidate inverse pairs"
    reviewer = {"id": "mechanics-reviewer", "provider": "offline-fixture", "model": "fixture", "family": "fixture", "revision": "1"}
    if canonical is None:
        canonical = []
        for split in ("train", "dev"):
            for index, label in enumerate(SM.SR.SEMANTIC_RELATIONS):
                original = data_row(f"fixture-{split}-{index}", label=label)
                original["pair"]["old"]["text"] = f"旧{split}甲命题{index}。"
                original["pair"]["new"]["text"] = f"新{split}乙命题{index}。"
                canonical.append(dict(original, split=split))
            for index, label in enumerate(SM.SR.NLI_CLASSES):
                identifier = f"public-{split}-{index}"
                canonical.append({"id": identifier, "root_id": identifier, "cluster_id": identifier, "split": split,
                                  "label_space": "nli3", "label": label, "label_origin": "mechanics_fixture",
                                  "pair": {"old": {"kind": "proposition", "text": f"公开{split}旧句{index}"},
                                           "new": {"kind": "proposition", "text": f"公开{split}新句{index}"}}})
    for template in canonical:
        if template["label_space"] == "nli3":
            public.append(copy.deepcopy(template))
            continue
        label = template["label"]
        original = data_row(template["root_id"], label=label)
        original.update(id=template["id"], root_id=template["root_id"], cluster_id=template["cluster_id"],
                        pair=SM.SR.validate_semantic_pair(template["pair"], limits=SM.INPUT_LIMITS))
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
                           "root_id": original["root_id"], "cluster_id": original["cluster_id"], "stage_split": template["split"]})
        rows.append(original)
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
    tokenizer.add_tokens(["same_domain"])
    tokenizer.save_pretrained(str(directory))
    mother, _ = FT.build_joint_model(directory, num_labels=3)
    mother.backbone.resize_token_embeddings(len(tokenizer))
    mother.backbone.save_pretrained(str(directory))
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
        assert meta["train"]["history"] == payload["continuation"]["history"] == results[arm]["history"]
        for record in meta["train"]["history"]:
            units = record["dev_diagnostics"]["parent_nll_by_root"]
            assert sum(unit["loss"] * unit["weight"] for unit in units) == pytest.approx(record["dev_loss"], abs=1e-12)
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


def test_input_cache_full_view_token_audit_parity_and_bounded_batches(tiny, tmp_path, monkeypatch):
    _, tokenizer, mother = tiny
    fit = canonical_rows()
    fit[0]["pair"]["old"]["text"] = "train café 中文 空白\t\n" * 80
    fit[0]["pair"]["new"]["text"] = "新命题 Unicode café " * 30
    for index in range(27):
        fit.append(dict(fit[1], id=f"extra-{index}", root_id=f"extra-{index}", cluster_id=f"extra-{index}"))
    fixture = prepared_canonical_fixture(tmp_path, fit=fit)
    groups = {"train": fixture["fit"], "dev": fixture["dev"]}
    expected = {name: SM.oriented_rows(rows, tokenizer, 64) for name, rows in groups.items()}
    original = SM.prepare_input_pairs
    batches = []
    def bounded(pairs, *params, **options):
        batches.append(len(pairs))
        assert len(pairs) <= 32
        return original(pairs, *params, **options)
    monkeypatch.setattr(SM, "prepare_input_pairs", bounded)
    cache = tmp_path / "input-cache"
    consume = lambda name, items: SM.oriented_rows(groups[name], prepared_inputs=items)
    disabled, disabled_report = SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=fixture["prepared_meta"], consume=consume)
    cold, cold_report = SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=fixture["prepared_meta"], input_cache=cache, consume=consume)
    assert max(batches) == 32 and disabled_report["mode"] == "disabled" and cold_report["mode"] == "cold"
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot inputs/dev/truncation must not prepare again")
    monkeypatch.setattr(SM, "prepare_input_pairs", forbidden)
    monkeypatch.setattr(SM.SR, "prepare_semantic_input", forbidden)
    hot, hot_report = SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=fixture["prepared_meta"], input_cache=cache, consume=consume)
    assert hot_report["mode"] == "hot"
    assert {path.name for path in cache.iterdir()} == {"input-cache.json", "train.jsonl.gz", "dev.jsonl.gz"}
    for name, rows in expected.items():
        assert rows == disabled[name] == cold[name] == hot[name]
        assert hot_report["groups"][name]["prepare_seconds"] == hot_report["groups"][name]["write_seconds"] == 0
        assert cold_report["groups"][name]["rows"] == hot_report["groups"][name]["rows"] == len(groups[name])
        assert cold_report["groups"][name]["sha256"] == hot_report["groups"][name]["sha256"]
        for items in (disabled[name], cold[name], hot[name]):
            for index in range(0, len(items), 2):
                assert items[index]["prepared"] is items[index + 1]["prepared"]
                for row in items[index:index + 2]:
                    assert row["encoding"] is row["input_view"]["encoding"]
                    assert row["input_view"] is row["prepared"]["directions"]["reverse" if row["reverse"] else "forward"]
            assert SM.truncation_profile(tokenizer, items, 64) == SM.truncation_profile(tokenizer, rows, 64)
            left = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in items], torch.device("cpu"))
            right = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in rows], torch.device("cpu"))
            assert left.keys() == right.keys()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    model = SM.initialize_arm(mother, "c5")
    evaluated = [SM.eval_semantic(model, tokenizer, groups["dev"], arm="c5", max_length=64,
                                 device=torch.device("cpu"), orientations=items)
                 for items in (expected["dev"], cold["dev"], hot["dev"], hot["dev"])]
    assert all(item == evaluated[0] for item in evaluated)
    with SM.gzip.open(cache / "train.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert all(set(item) == {"id", "prepared"} for item in records)
    assert records[0]["prepared"] == expected["train"][0]["prepared"]
    with pytest.raises(ValueError, match="shorter"):
        SM.oriented_rows(groups["train"], prepared_inputs=iter([]))
    with pytest.raises(ValueError, match="exceeds"):
        SM.oriented_rows(groups["dev"], prepared_inputs=iter([item["prepared"] for item in hot["dev"]]))


def _profile_without_storage(profile):
    result = {key: value for key, value in profile.items() if key not in ("audit_mode", "audit_reference", "audit")}
    result["audit"] = [{key: value for key, value in audit.items() if key != "fields"} for audit in profile["audit"]]
    return result


def _assert_exact_training(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact_training(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact_training(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ["c5", "dual3"])
@pytest.mark.parametrize("seed", SM.SEEDS)
def test_cache_training_gradients_adam_selection_and_cross_cache_resume(tiny, qualified_canonical, tmp_path, monkeypatch, arm, seed):
    _, tokenizer, mother = tiny
    fit = qualified_canonical["fit"]
    options = args(qualified_canonical, seed=seed, eval_every_steps=1)
    cache = tmp_path / "cache"
    encoded_batches, gradients = [], []
    real_encode = FT.encode_prepared_views
    def encode(*params, **kwargs):
        result = real_encode(*params, **kwargs)
        encoded_batches.append({key: value.detach().cpu().clone() for key, value in result.items()})
        return result
    monkeypatch.setattr(FT, "encode_prepared_views", encode)
    def train(*, input_cache=None, updates=2, resume_state=None):
        nonlocal encoded_batches, gradients
        encoded_batches, gradients = [], []
        model = SM.initialize_arm(mother, arm)
        handles = [parameter.register_hook(lambda gradient, name=name: gradients.append((name, gradient.detach().cpu().clone())))
                   for name, parameter in model.named_parameters() if parameter.requires_grad]
        try:
            result = SM.train_semantic(model, tokenizer, fit, arm=arm,
                                       **dict(options, updates=updates, input_cache=input_cache, resume_state=resume_state))
        finally:
            for handle in handles:
                handle.remove()
        return model, result
    def comparable(result):
        value = {key: value for key, value in result.items() if key not in ("seconds", "input_preparation")}
        value["truncation"] = {split: _profile_without_storage(profile) for split, profile in result["truncation"].items()}
        return value
    uncached_model, uncached = train()
    uncached_batches, uncached_gradients = encoded_batches, gradients
    cold_model, cold = train(input_cache=cache)
    assert encoded_batches and gradients
    _assert_exact_training(uncached_batches, encoded_batches)
    _assert_exact_training(uncached_gradients, gradients)
    assert cold["input_preparation"]["mode"] == "cold"
    assert all(profile["audit_mode"] == "inline" for profile in uncached["truncation"].values())
    assert all(profile["audit_mode"] == "cache_ref" and all("fields" not in audit for audit in profile["audit"])
               for profile in cold["truncation"].values())
    _assert_exact_training(comparable(uncached), comparable(cold))
    for (name, left), (other, right) in zip(uncached_model.named_parameters(), cold_model.named_parameters()):
        assert name == other
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    first_uncached = train(updates=1)[1]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot fit/dev/truncation called input preparation")
    with monkeypatch.context() as patch:
        patch.setattr(SM, "prepare_input_pairs", forbidden)
        patch.setattr(SM.SR, "prepare_semantic_input", forbidden)
        _, hot = train(input_cache=cache)
        _, continued_cached = train(input_cache=cache, resume_state=first_uncached["continuation"])
        _, first_cached = train(input_cache=cache, updates=1)
    _, continued_uncached = train(resume_state=first_cached["continuation"])
    assert hot["input_preparation"]["mode"] == "hot"
    for result in (hot, continued_cached, continued_uncached):
        _assert_exact_training(comparable(uncached), comparable(result))
    assert not any("cache" in key for key in hot["training_binding"])
    SM.validate_training_ledger(hot, hot["continuation"])


@pytest.mark.parametrize("fault", ["window", "tokenizer", "model_input_names", "source", "declaration", "order", "prepared_hash", "count"])
def test_input_cache_identity_mismatches_require_new_directory(tiny, qualified_canonical, tmp_path, monkeypatch, fault):
    _, tokenizer, _ = tiny
    groups = {"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}
    meta = copy.deepcopy(qualified_canonical["prepared_meta"])
    cache = tmp_path / "cache"
    SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=meta, input_cache=cache)
    before = {path.name: FT.file_sha256(path) for path in cache.iterdir()}
    max_length = 64
    if fault == "window":
        max_length = 65
    elif fault == "tokenizer":
        tokenizer.add_tokens(["new-cache-token"])
    elif fault == "model_input_names":
        tokenizer.model_input_names = list(reversed(tokenizer.model_input_names))
    elif fault == "source":
        source = SM.source_binding()
        monkeypatch.setattr(SM, "source_binding", lambda: dict(source, semantic_model="changed"))
    elif fault == "declaration":
        groups = dict(groups, train=[dict(groups["train"][0], frame_role_independent=False), *groups["train"][1:]])
    elif fault == "order":
        groups = dict(groups, train=list(reversed(groups["train"])))
    elif fault == "prepared_hash":
        meta["files_sha256"]["train"] = "changed"
    else:
        meta["counts"]["train"] += 1
    with pytest.raises(ValueError, match="input cache.*new directory|input group/prepared count"):
        SM.prepare_inputs(groups, tokenizer, max_length, prepared_meta=meta, input_cache=cache)
    assert before == {path.name: FT.file_sha256(path) for path in cache.iterdir()}


@pytest.mark.parametrize("fault", ["manifest", "missing_train", "missing_dev", "corrupt", "order", "short", "extra"])
def test_input_cache_missing_corrupt_or_misaligned_payload_rejected(tiny, qualified_canonical, tmp_path, fault):
    _, tokenizer, _ = tiny
    groups = {"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}
    meta = qualified_canonical["prepared_meta"]
    cache = tmp_path / "cache"
    SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=meta, input_cache=cache)
    manifest_path = cache / "input-cache.json"
    if fault == "manifest":
        manifest_path.unlink()
    elif fault.startswith("missing_"):
        (cache / (fault.removeprefix("missing_") + ".jsonl.gz")).unlink()
    elif fault == "corrupt":
        with (cache / "train.jsonl.gz").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        path = cache / "train.jsonl.gz"
        with SM.gzip.open(path, "rt", encoding="utf-8") as stream:
            lines = stream.readlines()
        if fault == "order":
            lines = list(reversed(lines))
        elif fault == "short":
            lines.pop()
        else:
            lines.append(lines[-1])
        with SM.gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.writelines(lines)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["payloads"]["train"]["sha256"] = FT.file_sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {path.name: FT.file_sha256(path) for path in cache.iterdir()}
    with pytest.raises(ValueError, match="input cache.*new directory"):
        SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=meta, input_cache=cache)
    assert before == {path.name: FT.file_sha256(path) for path in cache.iterdir()}


def test_input_cache_failed_build_never_publishes_partial_directory(tiny, qualified_canonical, tmp_path):
    _, tokenizer, _ = tiny
    cache = tmp_path / "interrupted-cache"
    def interrupted(_name, items):
        next(items)
        raise RuntimeError("interrupted consumer")
    with pytest.raises(RuntimeError, match="interrupted consumer"):
        SM.prepare_inputs({"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}, tokenizer, 64,
                          prepared_meta=qualified_canonical["prepared_meta"], input_cache=cache, consume=interrupted)
    assert not cache.exists() and not list(tmp_path.glob(".interrupted-cache-*"))


def test_prepare_inputs_cli_is_tokenizer_only_cpu_local(tiny, qualified_canonical, tmp_path, monkeypatch, capsys):
    import transformers

    _, tokenizer, _ = tiny
    directory = tmp_path / "tokenizer-only"
    tokenizer.save_pretrained(directory)
    cache = tmp_path / "cache"
    common = ["prepare-inputs", "--prepared", str(qualified_canonical["prepared"]), "--tokenizer", str(directory),
              "--max-length", "64", "--tiny-cpu"]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepare-inputs must not load model/optimizer/CUDA")
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(FT, "build_joint_model", forbidden)
    monkeypatch.setattr(FT, "build_joint_optimizer", forbidden)
    monkeypatch.setattr(NR, "load_nli_scorer", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    assert SM.main(common) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "disabled"
    assert SM.main([*common, "--input-cache", str(cache)]) == 0
    cold = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(SM, "prepare_input_pairs", forbidden)
    assert SM.main([*common, "--input-cache", str(cache)]) == 0
    hot = json.loads(capsys.readouterr().out)
    assert cold["mode"] == "cold" and hot["mode"] == "hot"
    assert cold["identity_sha256"] == hot["identity_sha256"]
    assert hot["mechanics_only"] and hot["command_total_seconds"] >= hot["total_seconds"]
    with pytest.raises(ValueError, match="formal training requires"):
        SM.main([arg for arg in common if arg != "--tiny-cpu"])


def test_budget_and_fit_cli_consume_shared_input_cache(tiny, qualified_canonical, tmp_path, monkeypatch, capsys):
    directory, tokenizer, mother = tiny
    mother_dir = tmp_path / "mother"
    NR.save_nli_artifact(NR.NLIScorer(model=mother, tokenizer=tokenizer, max_length=64), mother_dir,
                         backbone_dir=directory, train_meta={"mechanics_only": True})
    cache = tmp_path / "cache"
    common = ["--prepared", str(qualified_canonical["prepared"]), "--mother3", str(mother_dir), "--updates", "1",
              "--canonical-batch", "4", "--microbatch", "2", "--grad-accum", "4", "--max-length", "64", "--tiny-cpu"]
    assert SM.main(["budget", *common]) == 0
    uncached = json.loads(capsys.readouterr().out)
    assert SM.main(["budget", *common, "--input-cache", str(cache)]) == 0
    cold = json.loads(capsys.readouterr().out)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot budget must reuse prepared inputs")
    with monkeypatch.context() as patch:
        patch.setattr(SM, "prepare_input_pairs", forbidden)
        assert SM.main(["budget", *common, "--input-cache", str(cache)]) == 0
    hot = json.loads(capsys.readouterr().out)
    assert _profile_without_storage(uncached["truncation"]) == _profile_without_storage(cold["truncation"]) == _profile_without_storage(hot["truncation"])
    assert uncached["truncation"]["audit_mode"] == "inline"
    assert cold["truncation"] == hot["truncation"] and hot["truncation"]["audit_mode"] == "cache_ref"
    assert hot["input_preparation"]["mode"] == "hot"
    assert SM.main(["fit", *common, "--input-cache", str(cache), "--arm", "c5", "--out", str(tmp_path / "fit"),
                    "--device", "cpu", "--bf16", "off"]) == 0
    output = capsys.readouterr().out
    assert '"mode": "hot"' in output and '"reload_max_abs_diff": 0.0' in output


@pytest.mark.parametrize("fault", ["v2_recipe", "v2_source"])
def test_v3_rejects_stale_input_cache_without_rewriting_it(tiny, qualified_canonical, tmp_path, fault):
    _, tokenizer, _ = tiny
    groups = {"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}
    meta = qualified_canonical["prepared_meta"]
    cache = tmp_path / "old-cache"
    SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=meta, input_cache=cache)
    path = cache / "input-cache.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if fault == "v2_recipe":
        document["identity"]["input_recipe"] = "semantic-role-balanced-fragments-v2"
    else:
        document["identity"]["source_sha256"]["semantic_relations"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    before = {item.name: FT.file_sha256(item) for item in cache.iterdir()}
    with pytest.raises(ValueError, match="input cache identity mismatch.*new directory"):
        SM.prepare_inputs(groups, tokenizer, 64, prepared_meta=meta, input_cache=cache)
    assert before == {item.name: FT.file_sha256(item) for item in cache.iterdir()}


def _replay_binding(fixture, tokenizer):
    return {"source_sha256": SM.source_binding(), "max_length": 64,
            "prepared_data": {"files_sha256": fixture["prepared_meta"]["files_sha256"]},
            "tokenizer_encoding_sha256": NR.tokenizer_encoding_sha256(tokenizer),
            "model_input_names": list(tokenizer.model_input_names)}


def test_training_projection_preserves_science_and_releases_full_trees(tiny, qualified_canonical):
    import gc
    class TrackedDict(dict):
        pass
    class TrackedList(list):
        pass
    _, tokenizer, _ = tiny
    rows = qualified_canonical["fit"]
    references = []
    def track(item):
        result = TrackedDict(item)
        result["directions"] = {name: TrackedDict(view) for name, view in item["directions"].items()}
        references.append(weakref.ref(result))
        for view in result["directions"].values():
            view["fields"] = TrackedList(view["fields"])
            references.extend((weakref.ref(view), weakref.ref(view["fields"])))
        return result
    prepared = [track(item) for item in SM.prepare_input_pairs([row["pair"] for row in rows], tokenizer, 64,
                                                              frame_declarations=[True] * len(rows))]
    snapshot = copy.deepcopy(prepared)
    full = SM.oriented_rows(rows, prepared_inputs=prepared)
    thin = SM.oriented_rows(rows, prepared_inputs=prepared, training_only=True)
    assert prepared == snapshot
    for index in range(len(full)):
        assert thin[index]["encoding"] is full[index]["encoding"]
        assert thin[index]["text"] is full[index]["text"]
        assert thin[index]["prepared"] is not full[index]["prepared"]
        assert thin[index]["input_view"] is not full[index]["input_view"]
        assert "raw_text" not in thin[index]["prepared"] and "fields" not in thin[index]["input_view"]
        assert {key: value for key, value in thin[index].items() if key not in ("prepared", "input_view")} \
            == {key: value for key, value in full[index].items() if key not in ("prepared", "input_view")}
        assert thin[index]["input_view"] is thin[index]["prepared"]["directions"]["reverse" if index % 2 else "forward"]
    assert all(thin[index]["prepared"] is thin[index + 1]["prepared"] for index in range(0, len(thin), 2))
    del full, prepared
    gc.collect()
    assert all(reference() is None for reference in references)
    with pytest.raises(ValueError, match="requires prepared"):
        SM.oriented_rows(rows, training_only=True)


def test_training_inputs_inline_cold_hot_and_full_audit_replay(tiny, qualified_canonical, tmp_path, monkeypatch):
    _, tokenizer, _ = tiny
    groups = {"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}
    options = {"prepared_meta": qualified_canonical["prepared_meta"]}
    full, inline, _ = SM._prepare_training_inputs(groups, tokenizer, 64, **options)
    cache = tmp_path / "cache"
    thin, compact, cold = SM._prepare_training_inputs(groups, tokenizer, 64, **options, input_cache=cache)
    assert cold["mode"] == "cold" and Path(cold["audit_reference"]["directory"]) == cache
    assert (cache / "input-cache.json").is_file()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot materialization/profile/replay must not reprepare or rerender")
    monkeypatch.setattr(SM, "prepare_input_pairs", forbidden)
    monkeypatch.setattr(SM.SR, "prepare_semantic_input", forbidden)
    hot, hot_profiles, report = SM._prepare_training_inputs(groups, tokenizer, 64, **options, input_cache=cache)
    assert report["mode"] == "hot" and hot == thin and hot_profiles == compact
    binding = _replay_binding(qualified_canonical, tokenizer)
    for split in groups:
        assert _profile_without_storage(inline[split]) == _profile_without_storage(compact[split])
        assert all("fields" not in audit for audit in compact[split]["audit"])
        recovered = list(SM.replay_input_audit(compact[split]["audit_reference"], groups, binding=binding))
        expected = [row["prepared"] for row in full[split][::2]]
        assert recovered == expected
        serialized = "".join(json.dumps({"id": row["id"], "prepared": prepared}, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False) + "\n"
                             for row, prepared in zip(groups[split], expected))
        with SM.gzip.open(cache / (split + ".jsonl.gz"), "rt", encoding="utf-8") as stream:
            assert stream.read() == serialized
        left = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in full[split]], torch.device("cpu"))
        right = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in thin[split]], torch.device("cpu"))
        _assert_exact_training(left, right)


def _rewrite_audit_payload(cache, reference, records):
    path = cache / "train.jsonl.gz"
    with SM.gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path = cache / "input-cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payloads"]["train"].update(sha256=FT.file_sha256(path), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference["payloads"] = manifest["payloads"]
    reference["manifest_sha256"] = FT.file_sha256(manifest_path)


@pytest.mark.parametrize("fault,message", [
    ("reference_missing", "audit reference missing"), ("manifest", "audit reference mismatch"),
    ("payload", "payload hash/count/bytes mismatch"), ("source", "audit reference mismatch"),
    ("prepared_hash", "audit reference mismatch"), ("id", "canonical order mismatch"),
    ("order", "canonical order mismatch"), ("short", "record count mismatch"),
    ("extra", "record count mismatch"), ("summary", "stored audit summary mismatch"),
    ("fields", "stored audit summary mismatch"), ("model_input_names", "identity mismatch"),
])
def test_cache_audit_replay_rejects_corruption_without_fallback(tiny, qualified_canonical, tmp_path, monkeypatch, fault, message):
    _, tokenizer, _ = tiny
    groups = {"train": qualified_canonical["fit"], "dev": qualified_canonical["dev"]}
    cache = tmp_path / "cache"
    _, profiles, _ = SM._prepare_training_inputs(groups, tokenizer, 64,
        prepared_meta=qualified_canonical["prepared_meta"], input_cache=cache)
    profile = copy.deepcopy(profiles["train"])
    reference = profile["audit_reference"]
    binding = _replay_binding(qualified_canonical, tokenizer)
    if fault == "reference_missing":
        del profile["audit_reference"]
    elif fault == "manifest":
        with (cache / "input-cache.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif fault == "payload":
        with (cache / "train.jsonl.gz").open("ab") as stream:
            stream.write(b"x")
    elif fault == "source":
        reference["identity"]["source_sha256"]["semantic_model"] = "0" * 64
    elif fault == "prepared_hash":
        reference["identity"]["groups"]["train"]["prepared_file_sha256"] = "0" * 64
    elif fault == "model_input_names":
        binding["model_input_names"] = list(reversed(binding["model_input_names"]))
    else:
        with SM.gzip.open(cache / "train.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        if fault == "id":
            records[0]["id"] = "wrong-canonical-id"
        elif fault == "order":
            records[0], records[1] = records[1], records[0]
        elif fault == "short":
            records.pop()
        elif fault == "extra":
            records.append(copy.deepcopy(records[-1]))
        elif fault == "summary":
            records[0]["prepared"]["full_visible_token_lengths"][0] += 1
        elif fault == "fields":
            del records[0]["prepared"]["directions"]["forward"]["fields"]
        _rewrite_audit_payload(cache, reference, records)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("corrupt audit reference must never fall back to inline")
    monkeypatch.setattr(SM, "prepare_input_pairs", forbidden)
    with pytest.raises(ValueError, match=message):
        list(SM._validated_profile_audits(profile, groups, binding, "train"))


@pytest.mark.parametrize("arm", ["c5", "dual3"])
@pytest.mark.parametrize("tie", [False, True])
def test_cache_nonterminal_best_tie_save_reload_and_bounded_audit_lifetime(tiny, qualified_canonical, tmp_path, monkeypatch, arm, tie):
    class TrackedFields(list):
        pass
    directory, tokenizer, mother = tiny
    real_evaluate = SM.eval_semantic
    values = iter(())
    def evaluate(*args_, **kwargs):
        result = real_evaluate(*args_, **kwargs)
        result["loss"] = next(values)
        return result
    monkeypatch.setattr(SM, "eval_semantic", evaluate)
    outputs = []
    cache = tmp_path / "cache"
    for input_cache in (None, cache):
        values = iter([0.8, 0.8 if tie else 0.9])
        model = SM.initialize_arm(mother, arm)
        training = SM.train_semantic(model, tokenizer, qualified_canonical["fit"], arm=arm,
            **args(qualified_canonical, eval_every_steps=1, input_cache=input_cache))
        assert training["selection"]["selected_step"] == 1 and training["continuation"]["completed_steps"] == 2
        assert any(not torch.equal(training["continuation"]["best_state"][key], value)
                   for key, value in training["continuation"]["terminal_state"].items())
        outputs.append(training)
    for key in ("budget", "selection", "updates", "history", "continuation", "training_binding"):
        _assert_exact_training(outputs[0][key], outputs[1][key])
    references, peak = [], [0]
    original = SM._cached_group_inputs
    def tracked(*args_, **kwargs):
        for item in original(*args_, **kwargs):
            for view in item["directions"].values():
                view["fields"] = TrackedFields(view["fields"])
                references.append(weakref.ref(view["fields"]))
            peak[0] = max(peak[0], sum(reference() is not None for reference in references))
            yield item
    monkeypatch.setattr(SM, "_cached_group_inputs", tracked)
    scorer = SM.SemanticScorer(model=model, tokenizer=tokenizer, arm=arm, max_length=64)
    target = tmp_path / "trained"
    meta = SM.save_semantic_artifact(scorer, target, backbone_dir=directory, training=outputs[1])
    loaded, loaded_meta, payload = SM.load_semantic_artifact(target)
    assert meta == loaded_meta and all(profile["audit_mode"] == "cache_ref" and
        all("fields" not in audit for audit in profile["audit"]) for profile in meta["train"]["truncation"].values())
    before = scorer.predict_both_proba([qualified_canonical["dev"][0]["pair"]], role_independent_frame=True)
    after = loaded.predict_both_proba([qualified_canonical["dev"][0]["pair"]], role_independent_frame=True)
    _assert_exact_training(before, after)
    _assert_exact_training(payload["continuation"], outputs[1]["continuation"])
    resumed = []
    for input_cache in (None, cache):
        values = iter([1.0])
        resumed_model = SM.initialize_arm(mother, arm)
        resumed.append(SM.train_semantic(resumed_model, tokenizer, qualified_canonical["fit"], arm=arm,
            **args(qualified_canonical, updates=3, eval_every_steps=1, input_cache=input_cache,
                   resume_state=copy.deepcopy(payload["continuation"]))))
    for key in ("budget", "selection", "updates", "history", "continuation", "training_binding"):
        _assert_exact_training(resumed[0][key], resumed[1][key])
    assert resumed[0]["selection"]["selected_step"] == 1
    assert 0 < peak[0] <= 8 and all(reference() is None for reference in references)
    bad = copy.deepcopy(outputs[1])
    del bad["truncation"]["train"]["audit_reference"]
    with pytest.raises(ValueError, match="audit reference missing"):
        SM.validate_training_ledger(bad, bad["continuation"])
    path = cache / "train.jsonl.gz"
    hidden = cache / "temporarily-unavailable.gz"
    path.rename(hidden)
    try:
        with pytest.raises(ValueError, match="manifest or payload missing/corrupt"):
            SM.load_semantic_artifact(target)
    finally:
        hidden.rename(path)
