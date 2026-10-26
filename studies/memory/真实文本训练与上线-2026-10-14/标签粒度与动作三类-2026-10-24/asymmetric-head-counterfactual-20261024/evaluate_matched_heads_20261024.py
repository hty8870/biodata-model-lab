from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import action_common_20261024 as AC
from analyze_supplemental_results_20261024 import length_features
from verify_action_experiment_20261024 import manual
from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from run_matched_heads_20261024 import SEEDS, initialize, make_lock


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def records(source, probabilities):
    return [{"id": row["id"], "gold": AC.ACTIONS[row["action_index"]], "prediction": AC.ACTIONS[int(np.argmax(values))],
             "probabilities": list(map(float, values))} for row, values in zip(source, probabilities)]


def triplet_metrics(source, predictions):
    assert len(source) == len(predictions)
    guesses = {row["id"]: row["prediction"] for row in predictions}
    grouped = {}
    for row in source:
        grouped.setdefault(row["cluster_id"], []).append(row)
    full, conflict_transition, preservation_transition = 0, 0, 0
    details = []
    for name, group in sorted(grouped.items()):
        assert len(group) == 3 and {row["action_index"] for row in group} == {0, 1, 2}
        correct = {row["action_index"]: guesses[row["id"]] == AC.ACTIONS[row["action_index"]] for row in group}
        full += int(all(correct.values()))
        conflict_transition += int(correct[0] and correct[2])
        preservation_transition += int(correct[2] and correct[1])
        details.append({"group": name, "all_three_correct": all(correct.values()),
                        "by_action_correct": correct, "predictions": {str(row["action_index"]): guesses[row["id"]] for row in group}})
    slices = {}
    for family in sorted({row["family"] for row in source}):
        subset = [row for row in source if row["family"] == family]
        slices[family] = {"correct": sum(guesses[row["id"]] == row["action"] for row in subset), "n": len(subset)}
    whole = {kind: {"n": sum(row["whole_mode"] == kind for row in source),
                    "correct": sum(guesses[row["id"]] == "replace_all" for row in source if row["whole_mode"] == kind)}
             for kind in ("both_changed", "carry_valid_old")}
    return {"triplets_all_correct": full, "triplets_n": len(grouped), "conflict_transition_both_correct": conflict_transition,
            "preservation_transition_both_correct": preservation_transition, "transition_n": len(grouped),
            "by_family": slices, "whole_replacement_slices": whole, "triplet_details": details}


def checked_summary(raw, prepared, predictions, summary):
    direct = manual(raw, predictions)
    for key, value in direct.items():
        if key != "matrix":
            assert np.isclose(value, summary[key], atol=1e-10, rtol=0), (key, value, summary[key])
    assert direct["matrix"] == summary["confusion_matrix"]["matrix"]
    return direct


def selftest(source):
    gold = np.array([row["action_index"] for row in source])
    probabilities = np.full((len(source), 3), .005)
    probabilities[np.arange(len(source)), gold] = .99
    good = records(source, probabilities)
    wrong = records(source, np.roll(probabilities, 1, axis=1))
    all_replace = records(source, np.tile([.005, .99, .005], (len(source), 1)))
    assert manual(source, good)["correct"] == 120 and manual(source, wrong)["correct"] == 0
    assert triplet_metrics(source, good)["triplets_all_correct"] == 40
    assert triplet_metrics(source, wrong)["triplets_all_correct"] == 0
    assert manual(source, all_replace)["unsafe_replace_total_count"] == 80
    return {"perfect_120": True, "wrong_zero": True, "perfect_40_triplets": True, "wrong_zero_triplets": True,
            "coexist_and_partial_unsafe_replace_80": True}


def paired_bootstrap(evaluated, source):
    families = sorted({row["family"] for row in source})
    indexes = {family: np.array([i for i, row in enumerate(source) if row["family"] == family]) for family in families}
    gold = np.array([row["action_index"] for row in source])
    guesses = {(arm, seed): np.array([AC.ACTIONS.index(row["prediction"]) for row in evaluated[f"asym-{arm}-seed{seed}"]["counterfactual"]["predictions"]])
               for arm in ("c6", "f6") for seed in SEEDS}
    rng = np.random.default_rng(20261024 + 97)
    differences = []
    for _ in range(1000):
        selected = np.concatenate([indexes[key] for key in rng.choice(families, len(families), replace=True)])
        differences.append(float(np.mean([f1_score(gold[selected], guesses["f6", seed][selected], labels=[0, 1, 2], average="macro", zero_division=0)
                                          - f1_score(gold[selected], guesses["c6", seed][selected], labels=[0, 1, 2], average="macro", zero_division=0) for seed in SEEDS])))
    return {"unit": "semantic_preference_family", "n_units": len(families), "fixed_seeds": list(SEEDS), "replicates": 1000,
            "f6_minus_c6_macro_f1_95_percentile": np.quantile(differences, [.025, .975]).tolist(),
            "scope": "conditional on fixed two trained seeds, these 20 families, synthetic oracle and AI text review; not production safety"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    assert (ROOT / "training-complete.json").exists(), "all four fits must finish before diagnosis"
    make_lock(args.repo_root)
    audit = read(ROOT / "data-audit.json")
    assert audit["status"] == "passed"
    raw = rows(ROOT / "data-v3/rows.jsonl")
    prepared = rows(ROOT / "data-v3/inputs.jsonl")
    assert [row["id"] for row in raw] == [row["id"] for row in prepared]
    assert all(row["complete"] for row in prepared)
    checks = selftest(raw)
    output = ROOT / "evaluation"
    output.mkdir(exist_ok=False)
    models = []
    for seed in SEEDS:
        for arm in ("c6", "f6"):
            models.append({"name": f"asym-{arm}-seed{seed}", "arm": arm, "seed": seed, "directory": ROOT / "runs" / f"{arm}-seed{seed}", "new": True})
        for arm in ("a3", "f6"):
            models.append({"name": f"original-{arm}-seed{seed}", "arm": arm, "seed": seed, "directory": ROOT.parent / "runs" / f"{arm}-seed{seed}", "new": False})
    locked = []
    for item in models:
        result = read(item["directory"] / "result.json")
        assert FT.file_sha256(item["directory"] / "selected.pt") == result["checkpoint_sha256"]
        locked.append({"name": item["name"], "arm": item["arm"], "seed": item["seed"], "checkpoint_sha256": result["checkpoint_sha256"],
                       "selected_step": result["fit"]["selection"]["selected_step"], "new": item["new"]})
    lock = {"models": locked, "new_counterfactual_sha256": audit["final_source_sha256"],
            "training_lock_sha256": FT.file_sha256(ROOT / "training-lock.json"), "audit_sha256": FT.file_sha256(ROOT / "data-audit.json"),
            "confirmation_rows_sha256": FT.file_sha256(ROOT.parent / "confirmation-v2/rows.jsonl"),
            "script_sha256": FT.file_sha256(__file__), "counterfactual_not_for_selection": True,
            "old_confirmation_is_used_diagnostic_not_blind": True, "data_v3_revision_note": FT.file_sha256(ROOT / "评分前输入契约修订-2026-10-24.md")}
    assert all(FT.file_sha256(ROOT / "data-v3" / name) == digest for name, digest in audit["final_source_sha256"].items())
    AC.dump(output / "evaluation-lock.json", lock)
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    old_confirmation = rows(ROOT.parent / "confirmation-v2/rows.jsonl")
    confirmation_inputs = None
    dev = AC.load_rows("dev")
    evaluated = {}
    for item in models:
        run = read(item["directory"] / "result.json")
        scorer = initialize(args.repo_root, item["seed"]) if item["new"] else AC.initialize(args.repo_root, item["arm"], item["seed"])
        scorer.model.load_state_dict(torch.load(item["directory"] / "selected.pt", map_location="cpu", weights_only=True), strict=True)
        scorer.model.to(device)
        helper_arm = "a3" if item["arm"] == "a3" else "f6"
        dev_summary, dev_predictions = AC.evaluate(scorer.model, scorer.tokenizer, dev, helper_arm, device)
        previous = read(item["directory"] / "dev-predictions.json")
        assert [row["id"] for row in previous] == [row["id"] for row in dev_predictions]
        delta = float(np.abs(np.asarray([row["probabilities"] for row in previous]) - np.asarray([row["probabilities"] for row in dev_predictions])).max())
        assert delta <= 1e-6 and [row["prediction"] for row in previous] == [row["prediction"] for row in dev_predictions]
        cf_summary, cf_predictions = AC.evaluate(scorer.model, scorer.tokenizer, prepared, helper_arm, device)
        cf_manual = checked_summary(raw, prepared, cf_predictions, cf_summary)
        if confirmation_inputs is None:
            confirmation_inputs = []
            for row in old_confirmation:
                item_input = SR.prepare_semantic_input(row["pair"], scorer.tokenizer, max_length=384, role_independent_frame=True,
                                                       limits=SR.SEMANTIC_LIMITS_RESEARCH)["directions"]["forward"]
                assert item_input["complete"]
                confirmation_inputs.append({"id": row["id"], "cluster_id": row["cluster_id"], "domain": row["domain"],
                                            "action_index": row["action_index"], "fine_index": row["fine_index"], "encoding": item_input["encoding"],
                                            "preserve_count": len(row["formal"]["preserve_keys"]), "conflict_count": len(row["formal"]["conflict_keys"]),
                                            "uncovered_old_count": len(set(row["formal"]["old"]) - set(row["formal"]["new"]))})
        confirmation_summary, confirmation_predictions = AC.evaluate(scorer.model, scorer.tokenizer, confirmation_inputs, helper_arm, device)
        checked_summary(old_confirmation, confirmation_inputs, confirmation_predictions, confirmation_summary)
        if not item["new"]:
            expected = read(ROOT.parent / "confirmation-v2" / f"{item['arm']}-seed{item['seed']}.json")["predictions"]
            assert [row["id"] for row in expected] == [row["id"] for row in confirmation_predictions]
            assert [row["prediction"] for row in expected] == [row["prediction"] for row in confirmation_predictions]
            old_delta = float(np.abs(np.array([row["probabilities"] for row in expected]) - np.array([row["probabilities"] for row in confirmation_predictions])).max())
            assert old_delta <= 1e-6
        else:
            old_delta = None
        result = {"name": item["name"], "arm": item["arm"], "seed": item["seed"], "checkpoint_sha256": run["checkpoint_sha256"],
                  "selected_step": run["fit"]["selection"]["selected_step"], "selected_train": run["selected_train"], "selected_dev": dev_summary,
                  "reload_dev_probability_max_delta": delta, "original_confirmation_probability_max_delta": old_delta,
                  "counterfactual": {"summary": cf_summary, "manual": cf_manual, "triplet": triplet_metrics(raw, cf_predictions), "predictions": cf_predictions},
                  "used_confirmation": {"summary": confirmation_summary, "predictions": confirmation_predictions}}
        evaluated[item["name"]] = result
        AC.dump(output / (item["name"] + ".json"), result)
        print(json.dumps({"name": item["name"], "selected_step": result["selected_step"], "cf_correct": cf_summary["correct"],
                          "cf_macro_f1": cf_summary["macro_f1"], "cf_triplets": result["counterfactual"]["triplet"]["triplets_all_correct"],
                          "confirmation_correct": confirmation_summary["correct"], "reload_delta": delta}), flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    training = rows(ROOT.parent / "data-v1/train.jsonl")
    text = lambda row: row["pair"]["old"]["text"] + " [NEW] " + row["pair"]["new"]["text"]
    char = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=20000),
                         LogisticRegression(C=1, max_iter=1000, random_state=13))
    char.fit([text(row) for row in training], [row["action_index"] for row in training])
    length = LogisticRegression(max_iter=1000, random_state=13).fit(length_features(training), [row["action_index"] for row in training])
    baseline_probabilities = {"char_tfidf": char.predict_proba([text(row) for row in raw]),
                              "length_clause_count": length.predict_proba(length_features(raw)),
                              "constant_coexist": np.tile([.99, .005, .005], (len(raw), 1))}
    baselines = {}
    for name, probabilities in baseline_probabilities.items():
        predictions = records(raw, probabilities)
        baselines[name] = {"summary": AC.summarize(probabilities, prepared), "manual": manual(raw, predictions),
                           "triplet": triplet_metrics(raw, predictions), "predictions": predictions}
        AC.dump(output / f"baseline-{name}.json", baselines[name])
    paired = []
    for seed in SEEDS:
        left, right = (evaluated[f"asym-{arm}-seed{seed}"] for arm in ("c6", "f6"))
        paired.append({"seed": seed, "cf_macro_f1_f6_minus_c6": right["counterfactual"]["summary"]["macro_f1"] - left["counterfactual"]["summary"]["macro_f1"],
                       "cf_correct_f6_minus_c6": right["counterfactual"]["summary"]["correct"] - left["counterfactual"]["summary"]["correct"],
                       "cf_triplets_f6_minus_c6": right["counterfactual"]["triplet"]["triplets_all_correct"] - left["counterfactual"]["triplet"]["triplets_all_correct"],
                       "confirmation_macro_f1_f6_minus_c6": right["used_confirmation"]["summary"]["macro_f1"] - left["used_confirmation"]["summary"]["macro_f1"]})
    summary = {"counterfactual_n": 120, "families": 20, "triplets": 40, "selftests": checks,
               "models": {name: {key: value for key, value in item.items() if key not in ("counterfactual", "used_confirmation")} |
                                {"counterfactual_summary": item["counterfactual"]["summary"],
                                 "triplet_summary": {k: v for k, v in item["counterfactual"]["triplet"].items() if k not in ("by_family", "triplet_details")},
                                 "used_confirmation_summary": item["used_confirmation"]["summary"]} for name, item in evaluated.items()},
               "baselines": {name: {"summary": item["summary"], "triplet_summary": {k: v for k, v in item["triplet"].items() if k not in ("by_family", "triplet_details")}} for name, item in baselines.items()},
               "paired": paired, "paired_family_bootstrap": paired_bootstrap(evaluated, raw), "human_gold": False, "production_changed": False}
    AC.dump(output / "summary.json", summary)
    fields = ["id", "family", "orientation", "whole_mode", "gold", "old", "new", "retained_old_text"]
    names = list(evaluated) + list(baselines)
    fields += [field for name in names for field in (name, name + "_raw_max_probability")]
    by_name = {name: {row["id"]: row for row in value["counterfactual"]["predictions"]} for name, value in evaluated.items()}
    by_name.update({name: {row["id"]: row for row in value["predictions"]} for name, value in baselines.items()})
    with (output / "counterfactual-all-answers.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in raw:
            record = {key: row[key] for key in ("id", "family", "orientation", "whole_mode", "retained_old_text")}
            record.update(gold=row["action"], old=row["pair"]["old"]["text"], new=row["pair"]["new"]["text"])
            for name in names:
                answer = by_name[name][row["id"]]
                record[name] = answer["prediction"]
                record[name + "_raw_max_probability"] = max(answer["probabilities"])
            writer.writerow(record)
    print(json.dumps({"evaluation_completed": True, "paired": paired, "bootstrap": summary["paired_family_bootstrap"]}), flush=True)


if __name__ == "__main__":
    main()
