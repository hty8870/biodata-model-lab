from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from research.memory_relations import finetune as FT

from action_common_20261024 import ACTIONS, ROOT, dump, evaluate, initialize, load_rows


def paired_interval(records, rows, seeds):
    domains = sorted({row["domain"] for row in rows})
    by_domain = {name: [i for i, row in enumerate(rows) if row["domain"] == name] for name in domains}
    gold = np.asarray([row["action_index"] for row in rows])
    predictions = {(arm, seed): np.asarray([ACTIONS.index(row["prediction"]) for row in records[f"{arm}-seed{seed}"]["predictions"]])
                   for seed in seeds for arm in ("a3", "f6")}
    rng = np.random.default_rng(20261024)
    differences = []
    for _ in range(1000):
        indexes = np.concatenate([by_domain[name] for name in rng.choice(domains, len(domains), replace=True)])
        differences.append(float(np.mean([f1_score(gold[indexes], predictions["f6", seed][indexes], average="macro", labels=[0, 1, 2], zero_division=0)
                                          - f1_score(gold[indexes], predictions["a3", seed][indexes], average="macro", labels=[0, 1, 2], zero_division=0) for seed in seeds])))
    return {"unit": "outer_document_topic", "units": len(domains), "replicates": 1000,
            "seed_set_fixed": seeds, "difference_macro_f1_percentile_95": np.quantile(differences, [.025, .975]).tolist(),
            "scope": "conditional on fixed trained seeds and this synthetic generator; five topics are not production risk certification"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "evaluation-lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "ready-for-one-heldout-pass"
    output = ROOT / "heldout-evaluation"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    dev, heldout = load_rows("dev"), load_rows("heldout")
    records = {}
    for item in lock["runs"]:
        arm, seed = item["arm"], item["seed"]
        name = f"{arm}-seed{seed}"
        directory = ROOT / "runs" / name
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert result["checkpoint_sha256"] == item["checkpoint_sha256"] == FT.file_sha256(directory / "selected.pt")
        assert result["config"]["common_sha256"] == FT.file_sha256(ROOT / "action_common_20261024.py")
        scorer = initialize(args.repo_root, arm, seed)
        scorer.model.load_state_dict(torch.load(directory / "selected.pt", map_location="cpu", weights_only=True), strict=True)
        scorer.model.to(device)
        dev_summary, dev_predictions = evaluate(scorer.model, scorer.tokenizer, dev, arm, device)
        expected = json.loads((directory / "dev-predictions.json").read_text(encoding="utf-8"))
        assert [row["id"] for row in expected] == [row["id"] for row in dev_predictions]
        difference = float(np.abs(np.asarray([row["probabilities"] for row in expected]) - np.asarray([row["probabilities"] for row in dev_predictions])).max())
        assert difference <= 1e-6
        assert [row["prediction"] for row in expected] == [row["prediction"] for row in dev_predictions]
        summary, predictions = evaluate(scorer.model, scorer.tokenizer, heldout, arm, device)
        per_fine = {}
        for leaf in range(6):
            indexes = [i for i, row in enumerate(heldout) if row["fine_index"] == leaf]
            per_fine[str(leaf)] = {"n": len(indexes), "correct": sum(predictions[i]["gold"] == predictions[i]["prediction"] for i in indexes)}
        records[name] = {"arm": arm, "seed": seed, "checkpoint_sha256": item["checkpoint_sha256"],
                         "selected_step": result["fit"]["selection"]["selected_step"], "summary": summary,
                         "dev_reload_max_probability_difference": difference, "dev_reload_summary": dev_summary,
                         "per_fine_action_accuracy": per_fine, "predictions": predictions}
        dump(output / (name + ".json"), records[name])
        print(json.dumps({"run": name, "correct": summary["correct"], "n": len(heldout), "macro_f1": summary["macro_f1"],
                          "unsafe_replace": summary["unsafe_replace_total_count"], "reload_probability_max": difference}), flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    seeds = sorted({item["seed"] for item in lock["runs"] if item["arm"] == "a3"})
    assert all(f"f6-seed{seed}" in records for seed in seeds)
    paired = []
    for seed in seeds:
        left, right = records[f"a3-seed{seed}"]["summary"], records[f"f6-seed{seed}"]["summary"]
        paired.append({"seed": seed, "delta_macro_f1": right["macro_f1"] - left["macro_f1"],
                       "delta_correct": right["correct"] - left["correct"],
                       "delta_unsafe_replace": right["unsafe_replace_total_count"] - left["unsafe_replace_total_count"]})
    dump(output / "summary.json", {"n": len(heldout), "scenario_groups": len({row["cluster_id"] for row in heldout}),
                                   "outer_topics": len({row["domain"] for row in heldout}),
                                   "runs": {name: {key: value for key, value in item.items() if key != "predictions"} for name, item in records.items()},
                                   "paired": paired, "paired_domain_bootstrap": paired_interval(records, heldout, seeds),
                                   "labels": "program oracle, 36 train-pilot AI blind checks; no human gold", "production_changed": False})


if __name__ == "__main__":
    main()
