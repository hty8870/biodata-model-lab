from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from verify_action_experiment_20261024 import manual, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def independent_triplets(source, predictions):
    correct = {row["id"]: row["prediction"] == row["gold"] for row in predictions}
    groups = {row["cluster_id"] for row in source}
    ids = {name: {row["action_index"]: row["id"] for row in source if row["cluster_id"] == name} for name in groups}
    return {"triplets_all_correct": sum(all(correct[key] for key in group.values()) for group in ids.values()),
            "triplets_n": len(ids),
            "conflict_transition_both_correct": sum(correct[group[0]] and correct[group[2]] for group in ids.values()),
            "preservation_transition_both_correct": sum(correct[group[1]] and correct[group[2]] for group in ids.values())}


def main():
    summary = read(ROOT / "evaluation/summary.json")
    lock = read(ROOT / "training-lock.json")
    assert sha(ROOT / "run_matched_heads_20261024.py") == lock["new_sources_sha256"]["run_matched_heads_20261024.py"]
    assert sha(ROOT / "设计-破缺六头与最小变化诊断-2026-10-24.md") == lock["new_sources_sha256"]["设计-破缺六头与最小变化诊断-2026-10-24.md"]
    for name, expected in lock["old_sources_sha256"].items():
        assert sha(ROOT.parent / name) == expected
    assert sha(ROOT / "data-v2/rows.jsonl") == lock["new_sources_sha256"]["rows.jsonl"]
    assert sha(ROOT / "data-v2/inputs.jsonl") == lock["new_sources_sha256"]["inputs.jsonl"]
    assert sha(ROOT / "data-v2/manifest.json") == lock["new_sources_sha256"]["manifest.json"]
    evaluation_lock = read(ROOT / "evaluation/evaluation-lock.json")
    assert sha(ROOT / "training-lock.json") == evaluation_lock["training_lock_sha256"]
    assert sha(ROOT / "evaluate_matched_heads_20261024.py") == evaluation_lock["script_sha256"]
    for name, expected in evaluation_lock["new_counterfactual_sha256"].items():
        assert sha(ROOT / "data-v3" / name) == expected
    train_source, dev_source = (rows(ROOT.parent / "data-v1" / (name + ".jsonl")) for name in ("train", "dev"))
    diagnostic = rows(ROOT / "data-v3/rows.jsonl")
    confirmation = rows(ROOT.parent / "confirmation-v2/rows.jsonl")
    fit_checks = {}
    for seed in (13, 20260919):
        paired = []
        for arm in ("c6", "f6"):
            directory = ROOT / "runs" / f"{arm}-seed{seed}"
            value = read(directory / "result.json")
            assert value["fit"]["budget"]["actual_updates"] == 300 and value["fit"]["budget"]["actual_sample_instances"] == 9600
            assert value["fit"]["backbone_grad_seen"] and value["config"]["device"] == "cuda:0"
            assert value["cuda_peak_allocated_bytes"] > 1024 ** 3 and not value["new_diagnostic_opened"]
            assert value["config"]["initial_coarse_gradient_spread"] > 1e-9 and value["selected_copy_parameter_spread"] > 1e-5
            assert sha(directory / "selected.pt") == value["checkpoint_sha256"]
            splits = {}
            for name, source in (("train", train_source), ("dev", dev_source)):
                predictions = read(directory / (name + "-predictions.json"))
                direct = manual(source, predictions)
                assert all(np.isclose(v, value["selected_" + name][k], atol=1e-10, rtol=0) for k, v in direct.items() if k != "matrix")
                splits[name] = direct
            fit_checks[f"{arm}-seed{seed}"] = {"selected_step": value["fit"]["selection"]["selected_step"], "splits": splits,
                                              "seconds": value["seconds"], "cuda_peak_allocated_bytes": value["cuda_peak_allocated_bytes"],
                                              "checkpoint_bytes": value["checkpoint_bytes"],
                                              "initial_coarse_gradient_spread": value["config"]["initial_coarse_gradient_spread"],
                                              "selected_copy_parameter_spread": value["selected_copy_parameter_spread"]}
            paired.append(value)
        assert paired[0]["config"]["initial_state_sha256"] == paired[1]["config"]["initial_state_sha256"]
        assert paired[0]["fit"]["budget"] == paired[1]["fit"]["budget"]
        assert paired[0]["batch_order_sha256"] == paired[1]["batch_order_sha256"]
        assert paired[0]["exposure_sha256"] == paired[1]["exposure_sha256"]
        left, right = (read(ROOT / "runs" / f"{arm}-seed{seed}" / "initial.json")["dev_predictions"] for arm in ("c6", "f6"))
        assert [row["id"] for row in left] == [row["id"] for row in right]
        assert np.max(np.abs(np.array([row["probabilities"] for row in left]) - np.array([row["probabilities"] for row in right]))) <= 1e-6
    evaluations = {}
    for name in summary["models"]:
        value = read(ROOT / "evaluation" / (name + ".json"))
        assert value["reload_dev_probability_max_delta"] <= 1e-6
        for key, source in (("counterfactual", diagnostic), ("used_confirmation", confirmation)):
            direct = manual(source, value[key]["predictions"])
            assert all(np.isclose(v, value[key]["summary"][k], atol=1e-10, rtol=0) for k, v in direct.items() if k != "matrix")
            assert direct["matrix"] == value[key]["summary"]["confusion_matrix"]["matrix"]
        triplet = independent_triplets(diagnostic, value["counterfactual"]["predictions"])
        assert all(value["counterfactual"]["triplet"][key] == count for key, count in triplet.items())
        evaluations[name] = {"manual": manual(diagnostic, value["counterfactual"]["predictions"]), "triplet": triplet}
    with (ROOT / "evaluation/counterfactual-all-answers.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len({row["id"] for row in csv_rows}) == 120
    by_id = {row["id"]: row for row in diagnostic}
    answers = {name: {row["id"]: row for row in read(ROOT / "evaluation" / (name + ".json"))["counterfactual"]["predictions"]}
               for name in summary["models"]}
    answers.update({name: {row["id"]: row for row in read(ROOT / "evaluation" / ("baseline-" + name + ".json"))["predictions"]}
                    for name in summary["baselines"]})
    for row in csv_rows:
        expected = by_id[row["id"]]
        assert row["gold"] == expected["action"] and row["old"] == expected["pair"]["old"]["text"] and row["new"] == expected["pair"]["new"]["text"]
        assert row["retained_old_text"] == expected["retained_old_text"]
        for name, predictions in answers.items():
            actual = predictions[row["id"]]
            assert row[name] == actual["prediction"]
            assert np.isclose(float(row[name + "_raw_max_probability"]), max(actual["probabilities"]), atol=1e-12, rtol=0)
    means = {}
    for group in ("original-a3", "original-f6", "asym-c6", "asym-f6"):
        values = [summary["models"][f"{group}-seed{seed}"] for seed in (13, 20260919)]
        means[group] = {"cf_accuracy": float(np.mean([value["counterfactual_summary"]["accuracy"] for value in values])),
                        "cf_macro_f1": float(np.mean([value["counterfactual_summary"]["macro_f1"] for value in values])),
                        "cf_triplets": float(np.mean([value["triplet_summary"]["triplets_all_correct"] for value in values])),
                        "cf_unsafe_replace": float(np.mean([value["counterfactual_summary"]["unsafe_replace_total_count"] for value in values])),
                        "cf_missed_conflict": float(np.mean([value["counterfactual_summary"]["missed_conflict_count"] for value in values])),
                        "cf_extra_merge": float(np.mean([value["counterfactual_summary"]["extra_merge_count"] for value in values])),
                        "confirmation_accuracy": float(np.mean([value["used_confirmation_summary"]["accuracy"] for value in values])),
                        "confirmation_macro_f1": float(np.mean([value["used_confirmation_summary"]["macro_f1"] for value in values]))}
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    result = {"status": "passed", "matched_pair_checks": True, "four_actual_cuda_fits": fit_checks,
              "independent_evaluation_recheck": evaluations, "two_seed_means": means, "csv_rows_verified": 120,
              "total_new_checkpoint_bytes": sum(value["checkpoint_bytes"] for value in fit_checks.values()),
              "evaluation_summary_sha256": sha(ROOT / "evaluation/summary.json"), "verification_script_sha256": sha(Path(__file__)),
              "human_gold": False, "production_changed": False}
    (ROOT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("independent_evaluation_recheck", "four_actual_cuda_fits")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
