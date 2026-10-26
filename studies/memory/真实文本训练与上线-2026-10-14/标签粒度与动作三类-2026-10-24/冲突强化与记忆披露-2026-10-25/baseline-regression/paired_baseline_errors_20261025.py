from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ("equivalent", "compatible", "conflict")


def read_seed(seed: int) -> tuple[dict[str, dict], dict]:
    path = HERE / f"direct3-seed{seed}-terminal-ocnli-diagnostic.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in data["predictions"]}
    assert len(rows) == len(data["predictions"]) == 1502
    assert data["checkpoint"] == "terminal-300" and data["seed"] == seed
    return rows, {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def decisions(row: dict) -> tuple[str, float, float]:
    probabilities = np.asarray(row["coarse_probabilities"], dtype=float)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability vector")
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("Invalid probability simplex")
    index = int(probabilities.argmax())
    return CLASSES[index], float(probabilities[2]), float(probabilities[index])


def combine(left: dict[str, dict], right: dict[str, dict]) -> tuple[dict, list[dict]]:
    assert left.keys() == right.keys()
    details = []
    for item_id in left:
        a, b = left[item_id], right[item_id]
        assert a["gold_nli"] == b["gold_nli"] and a["cluster_id"] == b["cluster_id"]
        pa, ca, ma = decisions(a)
        pb, cb, mb = decisions(b)
        gold = a["gold_nli"] == "contradiction"
        correct_a, correct_b = (pa == "conflict") == gold, (pb == "conflict") == gold
        state = "both_correct" if correct_a and correct_b else (
            "seed13_only_correct" if correct_a else "seed42_only_correct" if correct_b else "both_wrong")
        details.append({"id": item_id, "cluster_id": a["cluster_id"], "gold_nli": a["gold_nli"],
                        "seed13_class": pa, "seed42_class": pb,
                        "seed13_p_conflict": ca, "seed42_p_conflict": cb,
                        "seed13_max_probability": ma, "seed42_max_probability": mb,
                        "paired_binary_state": state})
    strata = {}
    for label in ("contradiction", "entailment", "neutral"):
        selected = [row for row in details if row["gold_nli"] == label]
        strata[label] = {"n": len(selected), "states": dict(Counter(r["paired_binary_state"] for r in selected)),
                         "clusters": len({r["cluster_id"] for r in selected})}
        for seed in (13, 42):
            mistakes = [r for r in selected if (r[f"seed{seed}_class"] == "conflict") != (label == "contradiction")]
            scores = [r[f"seed{seed}_p_conflict"] for r in mistakes]
            strata[label][f"seed{seed}"] = {
                "binary_errors": len(mistakes),
                "coarse_prediction_counts": dict(Counter(r[f"seed{seed}_class"] for r in selected)),
                "error_p_conflict_quantiles": dict(zip(("min", "p25", "median", "p75", "max"),
                    np.quantile(scores, (0, .25, .5, .75, 1)).tolist())) if scores else None,
                "errors_max_probability_ge_0_8": sum(r[f"seed{seed}_max_probability"] >= .8 for r in mistakes),
            }
    shared_misses = [r for r in details if r["gold_nli"] == "contradiction" and r["paired_binary_state"] == "both_wrong"]
    summary = {"n": len(details), "clusters": len({r["cluster_id"] for r in details}),
               "states": dict(Counter(r["paired_binary_state"] for r in details)),
               "coarse_prediction_disagreement_n": sum(r["seed13_class"] != r["seed42_class"] for r in details),
               "shared_contradiction_misses_n": len(shared_misses),
               "shared_contradiction_misses_clusters": len({r["cluster_id"] for r in shared_misses}),
               "strata": strata}
    return summary, details


def selftest() -> None:
    def row(gold: str, p: list[float]) -> dict:
        return {"gold_nli": gold, "cluster_id": "one", "coarse_probabilities": p}
    a = {"good": row("contradiction", [.1, .1, .8]), "bad": row("contradiction", [.8, .1, .1]),
         "neutral": row("neutral", [.1, .8, .1])}
    b = {"good": row("contradiction", [.1, .1, .8]), "bad": row("contradiction", [.1, .8, .1]),
         "neutral": row("neutral", [.1, .1, .8])}
    summary, details = combine(a, b)
    assert summary["states"] == {"both_correct": 1, "both_wrong": 1, "seed13_only_correct": 1}
    assert summary["shared_contradiction_misses_n"] == 1
    assert summary["strata"]["contradiction"]["seed13"]["errors_max_probability_ge_0_8"] == 1
    assert len(details) == 3
    try:
        decisions(row("neutral", [1, 1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid simplex was accepted")
    print(json.dumps({"selftest_checks": 5, "passed": 5}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    left, source13 = read_seed(13)
    right, source42 = read_seed(42)
    summary, details = combine(left, right)
    assert summary["strata"]["contradiction"]["n"] == 452
    for seed, expected in ((13, (293, 44)), (42, (225, 104))):
        fn = summary["strata"]["contradiction"][f"seed{seed}"]["binary_errors"]
        fp = sum(summary["strata"][label][f"seed{seed}"]["binary_errors"] for label in ("entailment", "neutral"))
        assert (fn, fp) == expected
    result = {"schema": "paired-terminal-baseline-errors/1", "sources": [source13, source42],
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "posthoc_baseline_diagnostic_not_new_test_or_threshold_selection",
              "confidence_0_8_is_descriptive_only": True, "summary": summary, "details": details}
    output = HERE / "paired_baseline_errors_20261025.json"
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output.exists():
        assert output.read_text(encoding="utf-8") == text, "Existing result differs; do not overwrite"
    else:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(json.dumps({"output": output.name, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
