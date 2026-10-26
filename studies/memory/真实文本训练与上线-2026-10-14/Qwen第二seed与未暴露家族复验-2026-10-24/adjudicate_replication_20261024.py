from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_common_20261024 as C
from verify_run_20261024 import metric_check

sys.path.insert(0, str(C.ANCHOR))
from verify_action_experiment_20261024 import manual

RUN42 = HERE / "runs" / "qwen3-lora-seed42"
BASELINE = HERE / "baseline" / "qwen3-lora-seed13-best-dev"
SOURCE13 = HERE.parent / "固定数据拟合与基座对照-2026-10-24" / "runs" / "qwen3-lora-seed13"
RESULTS = HERE / "results"


def decide(dev_macro, new_macro, missed, unsafe, baseline_macro):
    gap = abs(new_macro - baseline_macro)
    positive = {
        "seed42_dev_macro_f1_at_least_0_70": dev_macro >= 0.70,
        "seed42_new_macro_f1_at_least_0_65": new_macro >= 0.65,
        "seed42_missed_conflict_at_most_8_of_80": missed <= 8,
        "seed42_unsafe_replace_at_most_12_of_80": unsafe <= 12,
        "seed_macro_f1_absolute_gap_at_most_0_10": gap <= 0.10,
    }
    stop = {
        "seed42_new_macro_f1_below_0_55": new_macro < 0.55,
        "seed42_missed_conflict_above_16_of_80": missed > 16,
        "seed_macro_f1_absolute_gap_above_0_15": gap > 0.15,
    }
    status = "stop" if any(stop.values()) else ("positive_replication" if all(positive.values()) else "gray")
    return {"status": status, "seed_macro_f1_absolute_gap": gap,
            "positive_checks": positive, "stop_signals": stop,
            "all_positive_checks_passed": all(positive.values()), "any_stop_signal": any(stop.values())}


def selftest():
    rows = [
        {"id": "r0", "action_index": 0, "formal": {"old": {"0": 0, "1": 0}, "new": {"0": 0}}},
        {"id": "r1", "action_index": 1, "formal": {"old": {"0": 0, "1": 0}, "new": {"0": 1, "1": 1}}},
        {"id": "r2", "action_index": 2, "formal": {"old": {"0": 0, "1": 0}, "new": {"0": 1}}},
    ]

    def predictions(guesses):
        output = []
        for row, guess in zip(rows, guesses):
            probabilities = [0.01, 0.01, 0.01]
            probabilities[guess] = 0.97 if row["action_index"] == guess else 0.80
            remainder = (1.0 - probabilities[guess]) / 2
            probabilities = [value if index == guess else remainder for index, value in enumerate(probabilities)]
            output.append({"id": row["id"], "gold": C.ACTIONS[row["action_index"]],
                           "prediction": C.ACTIONS[guess], "probabilities": probabilities})
        return output

    perfect = manual(rows, predictions([0, 1, 2]))
    wrong = manual(rows, predictions([1, 0, 0]))
    assert perfect["macro_f1"] == 1.0 and perfect["correct"] == 3
    assert perfect["missed_conflict_count"] == perfect["unsafe_replace_total_count"] == 0
    assert wrong["macro_f1"] == wrong["correct"] == 0
    assert wrong["missed_conflict_count"] == 2 and wrong["unsafe_replace_total_count"] == 1
    assert wrong["coexist_replace_loss_count"] == 1 and wrong["unsafe_replace_count"] == 0
    positive = decide(0.70, 0.65, 8, 12, 0.70)
    low = decide(0.90, 0.549, 0, 0, 0.549)
    missed = decide(0.90, 0.80, 17, 0, 0.80)
    gap = decide(0.90, 0.80, 0, 0, 0.96)
    gray = decide(0.90, 0.60, 9, 0, 0.61)
    assert positive["status"] == "positive_replication" and all(positive["positive_checks"].values())
    assert low["status"] == "stop" and low["stop_signals"]["seed42_new_macro_f1_below_0_55"]
    assert missed["status"] == "stop" and missed["stop_signals"]["seed42_missed_conflict_above_16_of_80"]
    assert gap["status"] == "stop" and gap["stop_signals"]["seed_macro_f1_absolute_gap_above_0_15"]
    assert gray["status"] == "gray" and not gray["any_stop_signal"]
    return {"known_answer_metric_pass": True, "known_answer_metric_fail": True,
            "positive_boundary": True, "three_stop_branches": True, "gray_branch": True}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    tests = selftest()
    if args.selftest:
        print(json.dumps({"selftest": tests}, ensure_ascii=False))
        return 0
    result42 = C.read_json(RUN42 / "result.json")
    verification = C.read_json(RUN42 / "verification.json")
    result13 = C.read_json(SOURCE13 / "result.json")
    baseline_receipt = C.read_json(BASELINE / "receipt.json")
    assert verification["passed"] is True
    assert result42["config"]["seed"] == 42 and result13["config"]["seed"] == 13
    lock = C.make_or_check_lock()
    assert result42["config"]["settings"] == lock["protocol"]["qwen3"]
    lock_sha = C.sha256_file(HERE / "run-lock.json")
    assert result42["run_lock_sha256"] == baseline_receipt["run_lock_sha256"] == lock_sha
    code_paths = {"base": Path(C.__file__), "driver": HERE / "run_fit_bases_20261024.py",
                  "finetune.py": Path(C.FT.__file__), "action_common": C.ANCHOR / "action_common_20261024.py",
                  "semantic_relations.py": Path(C.SR.__file__)}
    assert set(code_paths) == set(lock["inputs"]["code"])
    for name, path in code_paths.items():
        assert C.sha256_file(path) == lock["inputs"]["code"][name], f"code drift after lock: {name}"
    for name, digest in baseline_receipt["data"].items():
        assert C.sha256_file(C.DIAGNOSTIC_DIR / name) == digest, f"new-set drift: {name}"
    assert C.sha256_file(SOURCE13 / "result.json") == baseline_receipt["source_result_sha256"]

    rows = C.diagnostic_rows()
    problems = []
    predictions13 = C.read_json(BASELINE / "predictions.json")
    predictions42 = C.read_json(RUN42 / "predictions" / "best-dev" / "diagnostic-120.json")
    metrics13 = C.read_json(BASELINE / "metrics.json")
    metrics42 = result42["diagnostics"]["metrics"]["diagnostic-120"]
    independent13 = metric_check(rows, predictions13, metrics13, "seed13/new120", problems)
    independent42 = metric_check(rows, predictions42, metrics42, "seed42/new120", problems)
    assert not problems, problems
    decision = decide(result42["best_dev"]["metrics"]["dev"]["macro_f1"],
                      independent42["macro_f1"], metrics42["missed_conflict_count"],
                      metrics42["unsafe_replace_total_count"], independent13["macro_f1"])
    by_id13 = {row["id"]: row for row in predictions13}
    by_id42 = {row["id"]: row for row in predictions42}
    assert set(by_id13) == set(by_id42) == {row["id"] for row in rows}
    disagreement = sum(by_id13[item]["prediction"] != by_id42[item]["prediction"] for item in by_id13)
    both_correct = sum(by_id13[item]["gold"] == by_id13[item]["prediction"] and
                       by_id42[item]["gold"] == by_id42[item]["prediction"] for item in by_id13)
    seed13_only = sum(by_id13[item]["gold"] == by_id13[item]["prediction"] and
                      by_id42[item]["gold"] != by_id42[item]["prediction"] for item in by_id13)
    seed42_only = sum(by_id13[item]["gold"] != by_id13[item]["prediction"] and
                      by_id42[item]["gold"] == by_id42[item]["prediction"] for item in by_id13)
    outcome_counts = {"both_correct": both_correct, "seed13_only_correct": seed13_only,
                      "seed42_only_correct": seed42_only,
                      "neither_correct": len(rows) - both_correct - seed13_only - seed42_only}
    assert sum(outcome_counts.values()) == len(rows)

    summary_rows = []
    for seed, dev, metrics in ((13, result13["best_dev"]["metrics"]["dev"], metrics13),
                               (42, result42["best_dev"]["metrics"]["dev"], metrics42)):
        summary_rows.append({"seed": seed, "dev_n": dev["n"], "dev_macro_f1": dev["macro_f1"],
                             "new_n": metrics["n"], "new_correct": metrics["correct"],
                             "new_accuracy": metrics["accuracy"], "new_macro_f1": metrics["macro_f1"],
                             "new_ce": metrics["loss"], "missed_conflict_count": metrics["missed_conflict_count"],
                             "conflict_denominator": metrics["conflict_denominator"],
                             "unsafe_replace_total_count": metrics["unsafe_replace_total_count"],
                             "preservation_denominator": metrics["preservation_denominator"],
                             "triplets_all_correct": metrics["triplets"]["triplets_all_correct"],
                             "triplets_n": metrics["triplets"]["triplets_n"],
                             **{f"f1_{name}": metrics["per_class"][name]["f1"] for name in C.ACTIONS}})
    fields = list(summary_rows[0])
    write_csv(RESULTS / "seed-comparison.csv", summary_rows, fields)
    family_rows = []
    for family in sorted(metrics13["triplets"]["by_family"]):
        left = metrics13["triplets"]["by_family"][family]
        right = metrics42["triplets"]["by_family"][family]
        assert left["n"] == right["n"]
        family_rows.append({"family": family, "n": left["n"], "seed13_correct": left["correct"],
                            "seed42_correct": right["correct"], "delta_seed42_minus_seed13": right["correct"] - left["correct"]})
    write_csv(RESULTS / "family-comparison.csv", family_rows,
              ["family", "n", "seed13_correct", "seed42_correct", "delta_seed42_minus_seed13"])
    next_steps = {
        "positive_replication": "more seeds, separate calibration/threshold data, and resource validation only; no deployment from this experiment",
        "stop": "preserve the negative result and analyze semantics/errors; no base or recipe expansion",
        "gray": "inconclusive; no deployment and no automatic configuration expansion",
    }
    receipt = {"date": "2026-10-24", "decision": decision,
               "thresholds": {"dev_macro_f1_min": 0.70, "new_macro_f1_min": 0.65,
                              "missed_conflict_max_of_80": 8, "unsafe_replace_max_of_80": 12,
                              "seed_gap_positive_max": 0.10, "new_macro_f1_stop_below": 0.55,
                              "missed_conflict_stop_above_of_80": 16, "seed_gap_stop_above": 0.15},
               "next_step": next_steps[decision["status"]], "no_deploy": True,
               "integrity": {"verification_passed": True, "lock_sha256": lock_sha,
                             "locked_code_unchanged": True, "new_data_unchanged": True,
                             "source13_result_unchanged": True, "metric_recompute_problems": problems,
                             "independent_metric_max_differences": {
                                 "seed13": max(independent13["differences"].values(), default=0.0),
                                 "seed42": max(independent42["differences"].values(), default=0.0)}},
               "analysis_selftest": tests, "seed_comparison": summary_rows,
               "paired_prediction_outcomes": outcome_counts, "argmax_disagreements": disagreement,
               "family_comparison": family_rows,
               "exposed_confirmation_300_secondary_only": {
                   "seed13": result13["diagnostics"]["metrics"]["confirmation-300"]["macro_f1"],
                   "seed42": result42["diagnostics"]["metrics"]["confirmation-300"]["macro_f1"]},
               "artifacts": {"seed_comparison_csv": "results/seed-comparison.csv",
                             "family_comparison_csv": "results/family-comparison.csv"},
               "script_sha256": C.sha256_file(Path(__file__))}
    C.dump(RESULTS / "replication-decision.json", receipt)
    print(json.dumps({"decision": decision, "paired_prediction_outcomes": outcome_counts,
                      "argmax_disagreements": disagreement, "selftest": tests}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
