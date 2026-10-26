from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from verify_action_experiment_20261024 import manual

ROOT = Path(__file__).resolve().parent
NAMES = ("coexist", "replace_all", "merge_preserve")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(rows, probabilities):
    return [{"id": row["id"], "gold": NAMES[row["action_index"]], "prediction": NAMES[int(np.argmax(values))], "probabilities": values}
            for row, values in zip(rows, probabilities)]


def check_extra(name, predictions, summary, rows):
    actual = manual(rows, predictions)
    for key, value in actual.items():
        if key != "matrix":
            assert np.isclose(value, summary[key], atol=1e-10, rtol=0), (name, key, value, summary[key])
    return actual


def length_features(rows):
    return np.asarray([[len(row["pair"]["old"]["text"]), len(row["pair"]["new"]["text"]),
                        row["pair"]["old"]["text"].count("；") + 1, row["pair"]["new"]["text"].count("；") + 1] for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-r6", action="store_true")
    args = parser.parse_args()
    datasets = {split: [json.loads(line) for line in (ROOT / "data-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
                for split in ("train", "dev", "heldout")}
    train, dev, heldout = (datasets[key] for key in ("train", "dev", "heldout"))
    text = lambda row: row["pair"]["old"]["text"] + " [NEW] " + row["pair"]["new"]["text"]
    char = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=20000),
                         LogisticRegression(C=1, max_iter=1000, random_state=13))
    char.fit([text(row) for row in train], [row["action_index"] for row in train])
    length = LogisticRegression(max_iter=1000, random_state=13).fit(length_features(train), [row["action_index"] for row in train])
    baseline_predictions = {"constant_coexist_uniform_forecast": records(heldout, np.full((len(heldout), 3), 1 / 3).tolist()),
                            "char_tfidf": records(heldout, char.predict_proba([text(row) for row in heldout]).tolist()),
                            "length_clause_count": records(heldout, length.predict_proba(length_features(heldout)).tolist())}
    baselines = {name: manual(heldout, predictions) for name, predictions in baseline_predictions.items()}
    primary = read(ROOT / "heldout-evaluation/summary.json")
    means = {arm: {key: float(np.mean([item["summary"][key] for item in primary["runs"].values() if item["arm"] == arm]))
                   for key in ("accuracy", "macro_f1", "unsafe_replace_total_count", "lost_preserved_items", "missed_conflict_count", "extra_merge_count")}
             for arm in ("a3", "f6")}
    mean_result = read(ROOT / "pooling-mean-seed13/result.json")
    mean_evaluation = read(ROOT / "pooling-mean-seed13/heldout.json")
    assert mean_evaluation["dev_reload_max_probability_difference"] <= 1e-6
    extras = {"mean_pool_seed13": check_extra("mean", mean_evaluation["predictions"], mean_evaluation["summary"], heldout)}
    a3 = read(ROOT / "runs/a3-seed13/result.json")
    for key in ("actual_updates", "actual_sample_instances", "real_tokens", "padding_tokens"):
        assert mean_result["fit"]["budget"][key] == a3["fit"]["budget"][key]
    if (ROOT / "posthoc-r6-evaluation.json").exists():
        r6 = read(ROOT / "posthoc-r6-evaluation.json")
        assert r6["dev_reload_max_probability_difference"] <= 1e-6
        extras["posthoc_r6_seed13"] = check_extra("posthoc_r6", r6["predictions"], r6["summary"], heldout)
        for split in ("train", "dev"):
            extra_train = read(ROOT / "runs/r6-seed13/result.json")["selected_" + split]
            extra_predictions = read(ROOT / "runs/r6-seed13" / (split + "-predictions.json"))
            check_extra("posthoc_r6_" + split, extra_predictions, extra_train, datasets[split])
    elif args.require_r6:
        raise AssertionError("posthoc R6 not finished")
    artifacts = list(ROOT.rglob("*"))
    weight_files = [path for path in artifacts if path.is_file() and path.suffix == ".pt"]
    result = {"passed": True, "baseline_hyperparameters_frozen_from_development_pilot": True,
              "baselines": baselines, "baseline_predictions": baseline_predictions, "primary_seed_means_same_300_questions": means,
              "primary_mean_macro_difference": means["f6"]["macro_f1"] - means["a3"]["macro_f1"],
              "primary_conditional_interval": primary["paired_domain_bootstrap"]["difference_macro_f1_percentile_95"],
              "extras": extras, "r6_is_posthoc_not_independent_final": True,
              "weight_files": {path.relative_to(ROOT).as_posix(): path.stat().st_size for path in weight_files},
              "weight_logical_bytes": sum(path.stat().st_size for path in weight_files),
              "all_artifact_logical_bytes": sum(path.stat().st_size for path in artifacts if path.is_file()),
              "logical_bytes_not_physical_volume_change": True}
    (ROOT / "supplemental-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "baselines", "primary_seed_means_same_300_questions", "primary_mean_macro_difference", "primary_conditional_interval", "extras", "weight_logical_bytes", "all_artifact_logical_bytes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
