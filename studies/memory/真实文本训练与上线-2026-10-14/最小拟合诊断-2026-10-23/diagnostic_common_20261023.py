from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from research.memory_relations import metrics as EM

from build_diagnostic_20261023 import selftest

ROOT = Path(__file__).resolve().parent
CLASSES = SR.SEMANTIC_RELATIONS


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return FT.file_sha256(path)


def load_inputs(tokenizer):
    rows = [json.loads(line) for line in (ROOT / "data-100-2026-10-23.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = selftest(rows)
    prepared = [SR.prepare_semantic_input(row["pair"], tokenizer, max_length=384, role_independent_frame=True) for row in rows]
    assert all(item["complete"] for item in prepared)
    encodings, labels, ids = [], [], []
    seen = {}
    for row, item in zip(rows, prepared):
        for direction in ("forward", "reverse"):
            encoding = item["directions"][direction]["encoding"]
            label = row["label"] if direction == "forward" else SR.swap_relation(row["label"])
            key = tuple(encoding["input_ids"])
            assert key not in seen or seen[key] == label, "same input with inconsistent labels"
            seen[key] = label
            encodings.append(encoding)
            labels.append(CLASSES.index(label))
            ids.append(row["id"] + ":" + direction)
    lengths = [len(item["input_ids"]) for item in encodings]
    audit = {
        "canonical_rows": len(rows), "scenario_clusters": len({row["cluster_id"] for row in rows}),
        "oriented_rows": len(encodings), "unique_oriented_encodings": len(seen), "complete_canonical": sum(item["complete"] for item in prepared),
        "canonical_class_counts": dict(Counter(row["label"] for row in rows)), "length_min": min(lengths), "length_max": max(lengths),
        "nonpadding_tokens_one_oriented_epoch": sum(lengths), "oracle_selftest": checks,
        "recipe": SR.SEMANTIC_INPUT_RECIPE, "data_sha256": sha(ROOT / "data-100-2026-10-23.jsonl"),
        "oriented_input_sha256": hashlib.sha256(json.dumps(encodings, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "inconsistent_label_collisions": 0,
    }
    dump(ROOT / "input-audit-2026-10-23.json", audit)
    return rows, encodings, np.asarray(labels, dtype=np.int64), ids, audit


def summarize(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert probabilities.shape == (len(labels), len(CLASSES))
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    result = EM.classification_summary(labels, probabilities.argmax(axis=1), classes=CLASSES)
    maxima = probabilities.max(axis=1)
    result.update({"ce": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-30, 1)).mean()),
                   "correct": int((probabilities.argmax(axis=1) == labels).sum()),
                   "max_softmax_min_mean_max": [float(maxima.min()), float(maxima.mean()), float(maxima.max())],
                   "above_080": int((maxima >= 0.80).sum())})
    return result


def analysis_selftest():
    labels = np.arange(5)
    perfect = np.full((5, 5), 0.005)
    perfect[np.arange(5), labels] = 0.98
    good = summarize(perfect, labels)
    bad = summarize(np.roll(perfect, 1, axis=1), labels)
    assert good["correct"] == 5 and good["accuracy"] == 1 and good["macro_f1"] == 1
    assert bad["correct"] == 0 and bad["accuracy"] == 0 and bad["macro_f1"] == 0
    assert bad["ce"] > good["ce"]
    return {"perfect_5_of_5": True, "intentionally_wrong_0_of_5": True, "loss_ordering": True}


def evaluate(model, tokenizer, encodings, labels, ids, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(encodings), 16):
            batch = FT.encode_prepared_views(tokenizer, encodings[start:start + 16], device)
            outputs.append(model(**batch).float().softmax(dim=-1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    forward, reverse = probabilities[0::2], probabilities[1::2]
    mapped = np.asarray([SR.swap_probabilities(row.tolist()) for row in reverse])
    metrics = {"canonical": summarize(forward, labels[0::2]), "reverse": summarize(reverse, labels[1::2]),
               "all_oriented": summarize(probabilities, labels),
               "swap_probability_max_abs": float(np.abs(forward - mapped).max()),
               "swap_hard_consistent": int((forward.argmax(axis=1) == mapped.argmax(axis=1)).sum())}
    predictions = [{"id": item, "gold": CLASSES[int(label)], "prediction": CLASSES[int(proba.argmax())],
                    "probabilities": proba.astype(float).tolist()} for item, label, proba in zip(ids, labels, probabilities)]
    return metrics, predictions
