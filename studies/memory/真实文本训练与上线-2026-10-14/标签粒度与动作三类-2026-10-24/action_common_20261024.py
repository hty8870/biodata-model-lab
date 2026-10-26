from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.memory_relations import finetune as FT
from research.memory_relations import metrics as EM
from research.memory_relations import nli_reference as NR

ROOT = Path(__file__).resolve().parent
ACTIONS = ("coexist", "replace_all", "merge_preserve")
FINE = ("coexist_equivalent", "coexist_old_strict", "coexist_new_strict", "coexist_overlap", "replace_conflict", "merge_conflict")
PARENTS = (0, 0, 0, 0, 1, 2)
MOTHER = "artifacts/worktrees/semantic-formal-train-20261015/research/memory_relations/runs/v8-formal-preparation-20261015/raw-mother3-seed-13"


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_rows(split):
    rows = [json.loads(line) for line in (ROOT / "inputs-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    source = {row["id"]: row for row in (json.loads(line) for line in (ROOT / "data-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines())}
    for row in rows:
        formal = source[row["id"]]["formal"]
        row["uncovered_old_count"] = len(set(formal["old"]) - set(formal["new"]))
    return rows


def coarse_logits(logits, arm):
    if arm == "a3":
        assert logits.shape[1] == 3
        return logits.float()
    assert arm in ("f6", "r6") and logits.shape[1] == 6
    return torch.stack((torch.logsumexp(logits[:, :4].float(), dim=1), logits[:, 4].float(), logits[:, 5].float()), dim=1)


def initialize(repo, arm, seed):
    mother = Path(repo) / MOTHER
    assert FT.file_sha256(mother / "nli_model.pt") == "99e9b75f826533289bf7b43eb5336916c7929c6bf3f08b65750e431ebedcf26c"
    scorer = NR.load_nli_scorer(mother, device="cpu")
    torch.manual_seed(seed)
    base = nn.Linear(scorer.model.classifier.in_features, 3)
    nn.init.normal_(base.weight, mean=0, std=.02)
    nn.init.zeros_(base.bias)
    if arm == "a3":
        scorer.model.classifier = base
    else:
        head = nn.Linear(base.in_features, 6)
        with torch.no_grad():
            for index, parent in enumerate(PARENTS):
                head.weight[index].copy_(base.weight[parent])
                head.bias[index].copy_(base.bias[parent] - (np.log(4) if parent == 0 else 0))
        features = torch.randn(9, base.in_features)
        assert torch.allclose(base(features), coarse_logits(head(features), arm), atol=1e-6, rtol=0)
        scorer.model.classifier = head
    torch.manual_seed(seed + 100000)
    random.seed(seed)
    np.random.seed(seed)
    return scorer


def labels_for(rows, arm, seed):
    targets = np.asarray([row["action_index"] if arm == "a3" else row["fine_index"] for row in rows], dtype=np.int64)
    if arm == "r6":
        rng = np.random.default_rng(seed + 2718)
        for group in sorted({row["cluster_id"] for row in rows}):
            indexes = [i for i, row in enumerate(rows) if row["cluster_id"] == group and row["action_index"] == 0]
            targets[indexes] = rng.permutation(targets[indexes])
        assert np.array_equal(np.bincount(targets, minlength=6), np.bincount([row["fine_index"] for row in rows], minlength=6))
        assert all(PARENTS[int(label)] == row["action_index"] for label, row in zip(targets, rows))
    return targets


def summarize(probabilities, rows):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray([row["action_index"] for row in rows])
    assert probabilities.shape == (len(rows), 3) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    guess = probabilities.argmax(axis=1)
    result = EM.classification_summary(gold, guess, classes=ACTIONS)
    ce = float(-np.log(np.clip(probabilities[np.arange(len(rows)), gold], 1e-30, 1)).mean())
    uncovered = np.asarray([row["uncovered_old_count"] for row in rows])
    unsafe = (guess == 1) & (uncovered > 0)
    result.update(correct=int((gold == guess).sum()), loss=ce, row_loss=ce,
                  n_groups=len({row["domain"] for row in rows}), weighting="row_equal_class_balanced",
                  unsafe_replace_count=int(((gold == 2) & (guess == 1)).sum()), merge_denominator=int((gold == 2).sum()),
                  coexist_replace_loss_count=int(((gold == 0) & unsafe).sum()),
                  unsafe_replace_total_count=int(unsafe.sum()), preservation_denominator=int((uncovered > 0).sum()),
                  missed_conflict_count=int(((gold != 0) & (guess == 0)).sum()), conflict_denominator=int((gold != 0).sum()),
                  extra_merge_count=int(((gold != 2) & (guess == 2)).sum()),
                  lost_preserved_items=int(uncovered[guess == 1].sum()))
    return result


def evaluate(model, tokenizer, rows, arm, device):
    model.eval()
    probabilities, fine_values = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in rows[start:start + 16]], device)
            logits = model(**batch).float()
            probabilities.extend(coarse_logits(logits, arm).softmax(dim=-1).cpu().tolist())
            fine_values.extend(logits.softmax(dim=-1).cpu().tolist())
    result = summarize(probabilities, rows)
    predictions = [{"id": row["id"], "cluster_id": row["cluster_id"], "domain": row["domain"],
                    "gold": ACTIONS[row["action_index"]], "prediction": ACTIONS[int(np.argmax(values))],
                    "probabilities": values, "fine_probabilities": fine,
                    "preserve_count": row["preserve_count"], "conflict_count": row["conflict_count"]}
                   for row, values, fine in zip(rows, probabilities, fine_values)]
    if arm != "a3":
        labels = [row["fine_index"] for row in rows]
        fine_ce = float(-np.log(np.clip(np.asarray(fine_values)[np.arange(len(rows)), labels], 1e-30, 1)).mean())
        result.update(true_fine_ce=fine_ce, true_intra_ce=fine_ce - result["loss"])
    return result, predictions


def selftest():
    rows = [{"action_index": i, "fine_index": [0, 4, 5][i], "domain": "test", "preserve_count": int(i == 2), "uncovered_old_count": int(i == 2)} for i in range(3)]
    good = np.full((3, 3), .005)
    good[np.arange(3), np.arange(3)] = .99
    assert summarize(good, rows)["correct"] == 3
    assert summarize(np.roll(good, 1, axis=1), rows)["correct"] == 0
    coexist = [{"action_index": 0, "domain": "test", "uncovered_old_count": n} for n in (0, 1, 0, 1)]
    wrong_replace = np.tile([.005, .99, .005], (4, 1))
    loss_check = summarize(wrong_replace, coexist)
    assert loss_check["coexist_replace_loss_count"] == 2 and loss_check["unsafe_replace_total_count"] == 2
    assert loss_check["lost_preserved_items"] == 2 and loss_check["unsafe_replace_count"] == 0
    logits = torch.randn(12, 6, requires_grad=True)
    fine = torch.arange(12) % 6
    parent = torch.tensor(PARENTS)[fine]
    fine_loss = torch.nn.functional.cross_entropy(logits, fine, reduction="none")
    parent_loss = torch.nn.functional.cross_entropy(coarse_logits(logits, "f6"), parent, reduction="none")
    log_prob = logits.log_softmax(dim=1)
    parent_log_prob = coarse_logits(logits, "f6").log_softmax(dim=1)
    conditional = -log_prob[torch.arange(12), fine] + parent_log_prob[torch.arange(12), parent]
    assert torch.allclose(fine_loss, parent_loss + conditional, atol=1e-6)
    assert torch.all(conditional >= -1e-6)
    assert torch.allclose(torch.autograd.grad(fine_loss.mean(), logits, retain_graph=True)[0],
                          torch.autograd.grad((parent_loss + conditional).mean(), logits)[0], atol=1e-6)
    return {"perfect_3_of_3": True, "wrong_0_of_3": True, "coexist_replacement_information_loss": True,
            "equivalent_and_new_stronger_no_false_loss": True, "fine_coarse_conditional_loss_identity": True, "gradient_identity": True}
