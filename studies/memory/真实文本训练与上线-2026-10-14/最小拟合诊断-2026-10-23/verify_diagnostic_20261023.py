from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from research.memory_relations import finetune as FT
from research.memory_relations import semantic_model as SM
from diagnostic_common_20261023 import CLASSES, ROOT, analysis_selftest, dump, evaluate, load_inputs, sha, summarize


def surface_label(pair):
    old, new = pair["old"]["text"], pair["new"]["text"]
    counts = old.count("，并且"), new.count("，并且")
    if counts == (1, 1):
        return CLASSES[0]
    if counts == (1, 0):
        return CLASSES[1]
    if counts == (0, 1):
        return CLASSES[2]
    groups = pair["frame"]["background"].split("属性：", 1)[1].split("。", 1)[0].split("；")
    first_axis_literals = groups[0].split("或")
    return CLASSES[3] if old.rstrip("。") in first_axis_literals and new.rstrip("。") in first_axis_literals else CLASSES[4]


def public_gradient_check():
    logits = torch.tensor([[0.1, 0.7, -0.2, 0.3, -0.6]], requires_grad=True)
    loss = SM.orientation_losses(logits, [{"label_space": "nli3", "label": "entailment"}], "c5")[0].sum()
    gradient = torch.autograd.grad(loss, logits)[0][0]
    probabilities = logits.detach().softmax(dim=-1)[0]
    parent_mass = probabilities[0] + probabilities[1]
    expected = (probabilities[0] - probabilities[1]) * (1 - 1 / parent_mass)
    actual = gradient[0] - gradient[1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    equal_logits = torch.zeros((1, 5), requires_grad=True)
    equal_loss = SM.orientation_losses(equal_logits, [{"label_space": "nli3", "label": "entailment"}], "c5")[0].sum()
    equal_gradient = torch.autograd.grad(equal_loss, equal_logits)[0][0]
    assert float(equal_gradient[0] - equal_gradient[1]) == 0
    return {"unequal_sibling_gradient_difference": float(actual), "formula_verified": True,
            "equal_sibling_gradient_difference": float(equal_gradient[0] - equal_gradient[1]),
            "conclusion": "parent loss lacks fine identity labels; sibling gradient difference is zero at equality, not identically zero"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    run = ROOT / "run-seed13"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    config = json.loads((run / "run-config.json").read_text(encoding="utf-8"))
    assert sha(run / "diagnostic-terminal.pt") == result["checkpoint_sha256"]
    for name, expected in config["source_sha256"].items():
        assert sha(args.repo_root / "research/memory_relations" / name) == expected
    for name, expected in config["script_sha256"].items():
        assert sha(ROOT / name) == expected
    rows = [json.loads(line) for line in (ROOT / "data-100-2026-10-23.jsonl").read_text(encoding="utf-8").splitlines()]
    surface = [surface_label(row["pair"]) for row in rows]
    surface_correct = sum(pred == row["label"] for pred, row in zip(surface, rows))
    analysis = analysis_selftest()
    predicted = json.loads((run / "final-predictions.json").read_text(encoding="utf-8"))
    probabilities = np.asarray([row["probabilities"] for row in predicted])
    labels = np.asarray([CLASSES.index(row["gold"]) for row in predicted])
    assert summarize(probabilities[::2], labels[::2]) == result["final"]["canonical"]
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA unavailable; no CPU fallback"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    base = args.repo_root / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model, _ = FT.build_joint_model_from_config(AutoConfig.from_pretrained(base, local_files_only=True), num_labels=5)
    state = torch.load(run / "diagnostic-terminal.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    _, encodings, reload_labels, ids, audit = load_inputs(tokenizer)
    assert audit == config["input_audit"]
    metrics, reloaded = evaluate(model, tokenizer, encodings, reload_labels, ids, device)
    difference = float(np.abs(probabilities - np.asarray([row["probabilities"] for row in reloaded])).max())
    assert difference <= 1e-6
    assert [row["prediction"] for row in reloaded] == [row["prediction"] for row in predicted]
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(updates) == result["fit"]["budget"]["actual_updates"]
    assert all(row["backbone_gradient_norm"] > 0 and row["head_gradient_norm"] > 0 for row in updates)
    assert all(row["parameter_samples"][part]["step_delta_abs_max"] > 0 for row in updates for part in ("encoder", "head"))
    check = {"schema": "relation-fit-diagnostic-verification-v1", "passed": True,
             "loss_and_metric_selftests": analysis, "public_gradient": public_gradient_check(),
             "surface_only_baseline": {"correct": surface_correct, "n": len(rows), "uses_only_pair_text_and_frame": True,
                                       "limitation": "template shortcut, not generic language parser"},
             "recomputed_final_canonical": result["final"]["canonical"], "strict_reload_max_probability_difference": difference,
             "strict_reload_argmax_all_equal": True, "positive_encoder_and_head_gradient_updates": len(updates),
             "nonzero_sampled_encoder_and_head_parameter_delta_updates": len(updates),
             "checkpoint_and_source_sha256_verified": True,
             "input_audit_equal_after_reload": True, "reload_metrics": metrics}
    dump(run / "verification.json", check)
    print(json.dumps({"passed": True, "strict_reload_max_probability_difference": difference,
                      "surface_correct_of_100": surface_correct, "positive_gradient_updates": len(updates),
                      "public_gradient": check["public_gradient"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
