from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("coexist", "replace_all", "merge_preserve")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual(rows, predictions):
    matrix = np.zeros((3, 3), dtype=np.int64)
    losses = []
    lost, unsafe, coexist_loss, merge_loss, missed, extra = 0, 0, 0, 0, 0, 0
    for row, prediction in zip(rows, predictions):
        assert row["id"] == prediction["id"] and NAMES[row["action_index"]] == prediction["gold"]
        values = np.asarray(prediction["probabilities"])
        assert values.shape == (3,) and np.isfinite(values).all() and np.allclose(values.sum(), 1, atol=1e-6, rtol=0)
        gold, guess = row["action_index"], NAMES.index(prediction["prediction"])
        assert int(values.argmax()) == guess
        matrix[gold, guess] += 1
        losses.append(-float(np.log(values[gold])))
        uncovered = len(set(row["formal"]["old"]) - set(row["formal"]["new"]))
        if guess == 1 and uncovered:
            unsafe += 1
            lost += uncovered
            coexist_loss += int(gold == 0)
            merge_loss += int(gold == 2)
        missed += int(gold != 0 and guess == 0)
        extra += int(gold != 2 and guess == 2)
    f1 = []
    for index in range(3):
        denominator = int(matrix[index, :].sum() + matrix[:, index].sum())
        f1.append(2 * int(matrix[index, index]) / denominator if denominator else 0)
    return {"n": len(rows), "correct": int(matrix.trace()), "accuracy": float(matrix.trace() / matrix.sum()),
            "macro_f1": float(np.mean(f1)), "loss": float(np.mean(losses)), "unsafe_replace_total_count": unsafe,
            "unsafe_replace_count": merge_loss, "coexist_replace_loss_count": coexist_loss,
            "lost_preserved_items": lost, "missed_conflict_count": missed, "extra_merge_count": extra,
            "matrix": matrix.tolist()}


def main():
    progress = read(ROOT / "progress.json")
    assert progress["evaluation_completed"]
    lock = read(ROOT / "design-lock.json")
    assert all(sha(ROOT / name) == expected for name, expected in lock["inputs_sha256"].items())
    evaluation_lock = read(ROOT / "evaluation-lock.json")
    assert evaluation_lock["design_lock_sha256"] == sha(ROOT / "design-lock.json")
    source = {split: [json.loads(line) for line in (ROOT / "data-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
              for split in ("train", "dev", "heldout")}
    assert not ({row["domain"] for row in source["train"]} & {row["domain"] for row in source["dev"]})
    assert not ({row["domain"] for row in source["train"] + source["dev"]} & {row["domain"] for row in source["heldout"]})
    checks = {}
    for item in evaluation_lock["runs"]:
        name = f"{item['arm']}-seed{item['seed']}"
        directory = ROOT / "runs" / name
        result = read(directory / "result.json")
        assert result["fit"]["budget"]["actual_updates"] == 300
        assert result["fit"]["budget"]["actual_sample_instances"] == 9600
        assert result["config"]["device"] == "cuda:0" and result["cuda_peak_allocated_bytes"] > 0
        assert result["heldout_opened"] is False
        assert sha(directory / "selected.pt") == result["checkpoint_sha256"] == item["checkpoint_sha256"]
        checks[name] = {}
        for split in ("train", "dev"):
            predictions = read(directory / (split + "-predictions.json"))
            actual = manual(source[split], predictions)
            summary = result["selected_" + split]
            for key, value in actual.items():
                if key != "matrix":
                    assert np.isclose(value, summary[key], atol=1e-10, rtol=0), (name, split, key, value, summary[key])
            checks[name][split] = actual
        heldout = read(ROOT / "heldout-evaluation" / (name + ".json"))
        actual = manual(source["heldout"], heldout["predictions"])
        for key, value in actual.items():
            if key != "matrix":
                assert np.isclose(value, heldout["summary"][key], atol=1e-10, rtol=0), (name, "heldout", key)
        assert actual["matrix"] == heldout["summary"]["confusion_matrix"]["matrix"]
        checks[name]["heldout"] = actual
        assert heldout["dev_reload_max_probability_difference"] <= 1e-6
    for seed in {item["seed"] for item in evaluation_lock["runs"] if item["arm"] == "a3"}:
        initial = []
        budgets = []
        for arm in ("a3", "f6"):
            directory = ROOT / "runs" / f"{arm}-seed{seed}"
            initial.append(np.asarray([row["probabilities"] for row in read(directory / "initial.json")["dev_predictions"]]))
            budget = read(directory / "result.json")["fit"]["budget"]
            budgets.append((budget["actual_updates"], budget["actual_sample_instances"], budget["real_tokens"], budget["padding_tokens"]))
        assert np.abs(initial[0] - initial[1]).max() <= 1e-6
        assert budgets[0] == budgets[1]
    receipt = {"passed": True, "verification": "manual confusion and information-loss counts, no shared summarizer", "checks": checks,
               "all_fits_cuda": True, "same_budget_and_initial_parent_probabilities": True,
               "immutable_inputs_unchanged": True, "domain_split_disjoint": True, "no_deployment": True}
    (ROOT / "independent-verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "heldout": {name: value["heldout"] for name, value in checks.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
