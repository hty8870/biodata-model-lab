from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ACTIONS = ("coexist", "replace_all", "merge_preserve")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def label(row):
    return ACTIONS.index(row["prediction"])


def main():
    source = rows(ROOT / "data-v3/rows.jsonl")
    summary = read(ROOT / "evaluation/summary.json")
    evaluations = {name: read(ROOT / "evaluation" / (name + ".json")) for name in summary["models"]}
    output = {}
    for name, item in evaluations.items():
        predictions = item["counterfactual"]["predictions"]
        by_id = {row["id"]: row for row in predictions}
        groups = {}
        for row in source:
            groups.setdefault(row["cluster_id"], []).append(row["id"])
        hist = Counter(row["prediction"] for row in predictions)
        diversity = Counter(len({by_id[key]["prediction"] for key in group}) for group in groups.values())
        max_values = np.array([max(row["probabilities"]) for row in predictions])
        correct = np.array([row["prediction"] == row["gold"] for row in predictions])
        output[name] = {"prediction_histogram": dict(hist), "matrix": item["counterfactual"]["manual"]["matrix"],
                        "triplet_distinct_predictions": {str(k): v for k, v in diversity.items()},
                        "all_three_same": diversity.get(1, 0), "triplets_n": len(groups),
                        "raw_score_ge_0_8": int((max_values >= .8).sum()),
                        "wrong_with_raw_score_ge_0_8": int(((max_values >= .8) & ~correct).sum()),
                        "mean_raw_max_probability": float(max_values.mean()),
                        "coexist_merge_transition_both_correct": item["counterfactual"]["triplet"]["conflict_transition_both_correct"],
                        "merge_replace_transition_both_correct": item["counterfactual"]["triplet"]["preservation_transition_both_correct"]}
    agreement = {}
    for seed in (13, 20260919):
        left, right = evaluations[f"original-a3-seed{seed}"], evaluations[f"asym-c6-seed{seed}"]
        agreement[str(seed)] = {}
        for dataset in ("counterfactual", "used_confirmation"):
            lp, rp = left[dataset]["predictions"], right[dataset]["predictions"]
            assert [row["id"] for row in lp] == [row["id"] for row in rp]
            agreement[str(seed)][dataset] = {"same_label": sum(l["prediction"] == r["prediction"] for l, r in zip(lp, rp)), "n": len(lp)}
    training = rows(ROOT.parent / "data-v1/train.jsonl")
    coverage = {}
    for action in range(3):
        selected = [row for row in training if row["action_index"] == action]
        two_by_two = [row for row in selected if len(row["formal"]["old"]) == len(row["formal"]["new"]) == 2]
        target_pattern = []
        for row in two_by_two:
            old, new = row["formal"]["old"], row["formal"]["new"]
            overlap = set(old) & set(new)
            if action == 0:
                ok = len(overlap) == 1 and all(old[key] == new[key] for key in overlap)
            elif action == 1:
                ok = set(old) == set(new) and any(old[key] != new[key] for key in old)
            else:
                ok = len(overlap) == 1 and any(old[key] != new[key] for key in overlap)
            if ok:
                target_pattern.append(row)
        coverage[ACTIONS[action]] = {"all_training_n": len(selected), "two_old_two_new": len(two_by_two),
                                    "same_abstract_key_pattern_as_diagnostic": len(target_pattern),
                                    "source_has_context_prefixes": True}
    examples = []
    for orientation in (0, 1):
        group = [row for row in source if row["family"] == "language_code" and row["orientation"] == orientation]
        group.sort(key=lambda row: row["action_index"])
        for row in group:
            answers = {}
            for name in ("asym-c6-seed13", "asym-f6-seed13", "asym-c6-seed20260919", "asym-f6-seed20260919"):
                answer = next(value for value in evaluations[name]["counterfactual"]["predictions"] if value["id"] == row["id"])
                answers[name] = {"prediction": answer["prediction"], "raw_max_probability": max(answer["probabilities"])}
            examples.append({"id": row["id"], "orientation": orientation, "old": row["pair"]["old"]["text"],
                             "new": row["pair"]["new"]["text"], "gold": row["action"], "retained_old_text": row["retained_old_text"],
                             "answers": answers})
    baselines = {name: {"correct": value["summary"]["correct"], "n": 120, "macro_f1": value["summary"]["macro_f1"],
                       "triplets_correct": value["triplet_summary"]["triplets_all_correct"]} for name, value in summary["baselines"].items()}
    result = {"status": "completed_posthoc_diagnostic", "models": output, "a3_asym_c6_prediction_agreement": agreement,
              "training_abstract_pattern_coverage": coverage, "baselines": baselines, "preselected_language_code_examples": examples,
              "limits": ["posthoc interpretation only, no model reselection or new training", "abstract pattern count is not a causal explanation",
                         "raw softmax is uncalibrated and is not the production C5 confidence or fallback threshold",
                         "all new examples share a simplified fixed context; no human gold"],
              "source_summary_sha256": hashlib.sha256((ROOT / "evaluation/summary.json").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "failure-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "preselected_language_code_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
