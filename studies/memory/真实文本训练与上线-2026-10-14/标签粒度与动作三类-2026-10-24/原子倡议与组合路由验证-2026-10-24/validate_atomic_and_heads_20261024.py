from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FINE = ("coexist_equivalent", "coexist_old_strict", "coexist_new_strict", "coexist_overlap", "replace_conflict", "merge_conflict")
RELATIONS = ("equivalent", "compatible", "conflict")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_state(old, new, entries):
    keys = [key for key, _ in entries]
    final = dict(entries)
    return {
        "unsatisfied_new": sorted(key for key, value in new.items() if final.get(key) != value),
        "lost_untouched_old": sorted(key for key, value in old.items() if key not in new and final.get(key) != value),
        "extra_keys": sorted(set(final) - set(old) - set(new)),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "duplicate_atom_count": len(entries) - len(set(entries)),
    }


def atomic_apply(old, new):
    entries = list(old.items())
    operations = []
    for key, value in sorted(new.items()):
        if key not in old:
            operations.append({"op": "create", "key": key, "value": value})
            entries.append((key, value))
        elif old[key] != value:
            operations.append({"op": "update", "key": key, "value": value})
            entries = [(other, value if other == key else old_value) for other, old_value in entries]
        else:
            operations.append({"op": "no_op", "key": key, "value": value})
    return entries, operations


def classification(gold, guess):
    assert len(gold) == len(guess) and gold
    matrix = [[0 for _ in RELATIONS] for _ in RELATIONS]
    per_class = {}
    f1_values = []
    for index, name in enumerate(RELATIONS):
        tp = sum(g == index and p == index for g, p in zip(gold, guess))
        fp = sum(g != index and p == index for g, p in zip(gold, guess))
        fn = sum(g == index and p != index for g, p in zip(gold, guess))
        support = sum(g == index for g in gold)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall, "f1": f1}
    for g, p in zip(gold, guess):
        matrix[g][p] += 1
    return {"n": len(gold), "correct": sum(g == p for g, p in zip(gold, guess)),
            "accuracy": sum(g == p for g, p in zip(gold, guess)) / len(gold),
            "macro_f1": sum(f1_values) / len(f1_values), "per_class": per_class,
            "confusion_matrix": {"classes": list(RELATIONS), "matrix": matrix}}


def binary_at_threshold(scores, positive, threshold):
    tp = sum(score >= threshold and gold for score, gold in zip(scores, positive))
    fp = sum(score >= threshold and not gold for score, gold in zip(scores, positive))
    fn = sum(score < threshold and gold for score, gold in zip(scores, positive))
    tn = sum(score < threshold and not gold for score, gold in zip(scores, positive))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"threshold": threshold, "support": sum(positive), "predicted_positive": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def selftest():
    old = {"A": 1, "B": 1}
    entries, operations = atomic_apply(old, {"A": 1})
    assert [row["op"] for row in operations] == ["no_op"]
    assert not any(audit_state(old, {"A": 1}, entries).values())
    entries, operations = atomic_apply({"A": 1}, {"A": 1, "B": 0})
    assert [row["op"] for row in operations] == ["no_op", "create"]
    assert not any(audit_state({"A": 1}, {"A": 1, "B": 0}, entries).values())
    old = {"A": 1, "B": 1}
    new = {"A": 0}
    entries, operations = atomic_apply(old, new)
    assert [row["op"] for row in operations] == ["update"]
    assert not any(audit_state(old, new, entries).values())
    whole_replace = list(new.items())
    assert audit_state(old, new, whole_replace)["lost_untouched_old"] == ["B"]
    blind_append = list(old.items()) + list(new.items())
    assert audit_state(old, new, blind_append)["duplicate_key_count"] == 1
    perfect = classification([0, 1, 2], [0, 1, 2])
    wrong = classification([0, 1, 2], [1, 2, 0])
    assert perfect["macro_f1"] == 1.0 and wrong["macro_f1"] == 0.0
    binary = binary_at_threshold([0.9, 0.8, 0.2, 0.1], [True, True, False, False], 0.3)
    inverted = binary_at_threshold([0.1, 0.2, 0.8, 0.9], [True, True, False, False], 0.3)
    assert binary["precision"] == 1.0 and binary["recall"] == 1.0
    assert inverted["tp"] == 0 and inverted["fp"] == 2 and inverted["fn"] == 2
    return {"atomic_positive_cases": 3, "whole_replace_loss_detected": True,
            "blind_append_duplicate_detected": True, "perfect_metric": True,
            "deliberately_wrong_metric": True, "binary_positive_and_negative_controls": True}


def atomic_experiment():
    rows = []
    source_paths = []
    for split in ("train", "dev", "heldout"):
        path = BASE / "data-v1" / f"{split}.jsonl"
        source_paths.append(path)
        rows.extend(load_jsonl(path))
    assert len(rows) == 1200
    expected_fine = dict(zip(FINE, (100, 100, 100, 100, 400, 400)))
    assert Counter(row["fine"] for row in rows) == expected_fine
    grouped = {name: {"rows": 0, "rows_with_no_op": 0, "rows_with_create": 0, "rows_with_update": 0,
                      "operations": Counter(), "naive_coexist_rows_with_duplicates": 0,
                      "naive_coexist_duplicate_atoms": 0, "preserved_untouched_old_atoms": 0} for name in FINE}
    failures = []
    merge_gold_mismatches = []
    totals = Counter()
    for row in rows:
        old = {str(key): int(value) for key, value in row["formal"]["old"].items()}
        new = {str(key): int(value) for key, value in row["formal"]["new"].items()}
        entries, operations = atomic_apply(old, new)
        audit = audit_state(old, new, entries)
        if any(audit.values()) or dict(entries) != {**old, **new}:
            failures.append({"id": row["id"], "audit": audit})
        fine = grouped[row["fine"]]
        fine["rows"] += 1
        op_names = [operation["op"] for operation in operations]
        fine["operations"].update(op_names)
        totals.update(op_names)
        for name in ("no_op", "create", "update"):
            fine[f"rows_with_{name}"] += int(name in op_names)
        preserved = sorted(set(old) - set(new))
        fine["preserved_untouched_old_atoms"] += len(preserved)
        totals["preserved_untouched_old_atoms"] += len(preserved)
        if row["action_index"] == 0:
            naive_entries = list(old.items()) + list(new.items())
            naive_audit = audit_state(old, new, naive_entries)
            fine["naive_coexist_rows_with_duplicates"] += int(naive_audit["duplicate_atom_count"] > 0)
            fine["naive_coexist_duplicate_atoms"] += naive_audit["duplicate_atom_count"]
            totals["naive_coexist_duplicate_atoms"] += naive_audit["duplicate_atom_count"]
        if row["fine"] == "merge_conflict":
            totals["compound_merge_rows"] += 1
            update_keys = sorted(int(operation["key"]) for operation in operations if operation["op"] == "update")
            preserved_keys = [int(key) for key in preserved]
            if update_keys != sorted(row["formal"]["conflict_keys"]) or preserved_keys != sorted(row["formal"]["preserve_keys"]):
                merge_gold_mismatches.append(row["id"])
            else:
                totals["merge_rows_matching_generator_decomposition"] += 1
        if row["fine"] == "replace_conflict":
            totals["replace_all_removed_old_atoms"] += len(preserved)
        whole_replace_audit = audit_state(old, new, list(new.items()))
        totals["whole_replace_rows_with_loss"] += int(bool(whole_replace_audit["lost_untouched_old"]))
        totals["whole_replace_lost_old_atoms"] += len(whole_replace_audit["lost_untouched_old"])
    for fine in grouped.values():
        fine["operations"] = dict(sorted(fine["operations"].items()))
    return {"n_pairs": len(rows), "n_clusters": len({row["cluster_id"] for row in rows}),
            "implementation_regression_passed": not failures and not merge_gold_mismatches,
            "failure_count": len(failures), "failures": failures,
            "merge_gold_mismatch_count": len(merge_gold_mismatches), "merge_gold_mismatches": merge_gold_mismatches,
            "totals": dict(sorted(totals.items())), "by_fine": grouped,
            "interpretation": "constructive implementation regression under perfect atomization, finite independent slots and upsert-only; not independent empirical contract validation"}, source_paths


def relation_gold(fine_index):
    return 0 if fine_index == 0 else 1 if fine_index in (1, 2, 3) else 2


def grouped_probabilities(fine_probabilities):
    assert len(fine_probabilities) == 6
    assert math.isclose(sum(fine_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
    return (fine_probabilities[0], sum(fine_probabilities[1:4]), sum(fine_probabilities[4:6]))


def join_predictions(data_path, prediction_path):
    rows = {row["id"]: row for row in load_jsonl(data_path)}
    payload = load_json(prediction_path)
    predictions = payload["predictions"]
    assert len(rows) == len(predictions)
    assert set(rows) == {row["id"] for row in predictions}
    joined = []
    for prediction in predictions:
        source = rows[prediction["id"]]
        joined.append({"id": prediction["id"], "gold": relation_gold(source["fine_index"]),
                       "probabilities": grouped_probabilities(prediction["fine_probabilities"])})
    return joined, source_paths_for(data_path, prediction_path)


def source_paths_for(*paths):
    return [Path(path) for path in paths]


def joint_predict(probabilities):
    return max(range(3), key=lambda index: probabilities[index])


def cascade_predict(probabilities, equivalent_threshold, conflict_threshold):
    equivalent, _, conflict = probabilities
    if equivalent >= equivalent_threshold:
        return 0
    conditional_conflict = conflict / (1.0 - equivalent)
    return 2 if conditional_conflict >= conflict_threshold else 1


def relation_metrics(rows, arm, thresholds=None):
    joint_guess = [joint_predict(row["probabilities"]) for row in rows]
    if arm == "joint_argmax":
        guess = joint_guess
    else:
        equivalent_threshold, conflict_threshold = thresholds
        guess = [cascade_predict(row["probabilities"], equivalent_threshold, conflict_threshold) for row in rows]
    result = classification([row["gold"] for row in rows], guess)
    result["changed_vs_joint"] = sum(candidate != baseline for candidate, baseline in zip(guess, joint_guess))
    return result


def head_metrics(rows):
    gold_equivalent = [row["gold"] == 0 for row in rows]
    gold_conflict = [row["gold"] == 2 for row in rows]
    equivalent = [row["probabilities"][0] for row in rows]
    conflict = [row["probabilities"][2] for row in rows]
    states = Counter()
    for e_score, c_score in zip(equivalent, conflict):
        e_high = e_score >= 0.30
        c_high = c_score >= 0.30
        states["both_high" if e_high and c_high else "equivalent_only" if e_high else "conflict_only" if c_high else "neither_high"] += 1
    return {"equivalent_at_030": binary_at_threshold(equivalent, gold_equivalent, 0.30),
            "conflict_at_030": binary_at_threshold(conflict, gold_conflict, 0.30),
            "joint_states_at_030": dict(sorted(states.items()))}


def router_experiment():
    specifications = {
        "seed13": {"dev_step": "0200"},
        "seed20260919": {"dev_step": "0300"},
    }
    models = {}
    source_paths = []
    for seed, specification in specifications.items():
        dev_data = BASE / "data-v1" / "dev.jsonl"
        dev_predictions = BASE / "runs" / f"f6-{seed}" / f"dev-step-{specification['dev_step']}.json"
        confirmation_data = BASE / "confirmation-v2" / "rows.jsonl"
        confirmation_predictions = BASE / "confirmation-v2" / f"f6-{seed}.json"
        dev, dev_paths = join_predictions(dev_data, dev_predictions)
        confirmation, confirmation_paths = join_predictions(confirmation_data, confirmation_predictions)
        source_paths.extend(dev_paths + confirmation_paths)
        models[seed] = {"dev": dev, "confirmation": confirmation}
    fixed_arms = {"joint_argmax": None, "cascade_e030_c050": (0.30, 0.50), "cascade_e030_c030": (0.30, 0.30)}
    candidates = []
    for equivalent_step in range(1, 20):
        equivalent_threshold = equivalent_step / 20
        for conflict_threshold in (0.30, 0.50, 0.70):
            dev_scores = [relation_metrics(model["dev"], "cascade", (equivalent_threshold, conflict_threshold)) for model in models.values()]
            candidates.append({"thresholds": (equivalent_threshold, conflict_threshold),
                               "mean_macro_f1": sum(score["macro_f1"] for score in dev_scores) / len(dev_scores),
                               "mean_accuracy": sum(score["accuracy"] for score in dev_scores) / len(dev_scores)})
    selected = max(candidates, key=lambda row: (row["mean_macro_f1"], row["mean_accuracy"], row["thresholds"]))
    arms = dict(fixed_arms)
    arms["dev_tuned_cascade"] = selected["thresholds"]
    result_models = {}
    for seed, model in models.items():
        result_models[seed] = {}
        for split, rows in model.items():
            result_models[seed][split] = {
                "n": len(rows),
                "arms": {name: relation_metrics(rows, "joint_argmax" if name == "joint_argmax" else "cascade", thresholds)
                         for name, thresholds in arms.items()},
                "heads": head_metrics(rows),
            }
    means = {}
    for name in arms:
        scores = [result_models[seed]["confirmation"]["arms"][name] for seed in models]
        means[name] = {"accuracy": sum(score["accuracy"] for score in scores) / len(scores),
                       "macro_f1": sum(score["macro_f1"] for score in scores) / len(scores)}
    return {"arms": {name: {"equivalent_threshold": None if thresholds is None else thresholds[0],
                             "conditional_conflict_threshold": None if thresholds is None else thresholds[1]}
                     for name, thresholds in arms.items()},
            "dev_tuning": {"grid_size": len(candidates), "selected": selected,
                           "selection": "maximum two-seed mean dev macro-F1; ties mean accuracy, then higher thresholds"},
            "models": result_models, "confirmation_two_seed_mean": means,
            "interpretation": "same-frozen-F6-probability router counterfactual; not independently trained or calibrated two-model evidence"}, source_paths


def concise(result):
    return {"selftest": result["selftest"],
            "atomic": {"n_pairs": result["atomic"]["n_pairs"], "regression_passed": result["atomic"]["implementation_regression_passed"],
                       "totals": result["atomic"]["totals"]},
            "router": {"selected": result["router"]["dev_tuning"]["selected"],
                       "confirmation_two_seed_mean": result["router"]["confirmation_two_seed_mean"],
                       "joint_states_at_030": {seed: model["confirmation"]["heads"]["joint_states_at_030"]
                                                for seed, model in result["router"]["models"].items()}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    atomic, atomic_paths = atomic_experiment()
    router, router_paths = router_experiment()
    paths = atomic_paths + router_paths
    unique_paths = []
    for path in paths:
        if path not in unique_paths:
            unique_paths.append(path)
    result = {"schema": "atomic-proposal-and-head-router-validation-v1", "selftest": selftest(),
              "atomic": atomic, "router": router,
              "source_sha256": {str(path.relative_to(BASE)): sha256(path) for path in unique_paths}}
    canonical = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    output = ROOT / "data" / "results.json"
    if args.check:
        assert output.exists(), "results.json missing; run the generator first"
        assert load_json(output) == canonical, "recomputed results differ"
        print(json.dumps({"check": "passed", **concise(canonical)}, ensure_ascii=False))
        return
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; use --check for deterministic verification")
    output.write_text(json.dumps(canonical, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(concise(canonical), ensure_ascii=False))


if __name__ == "__main__":
    main()
