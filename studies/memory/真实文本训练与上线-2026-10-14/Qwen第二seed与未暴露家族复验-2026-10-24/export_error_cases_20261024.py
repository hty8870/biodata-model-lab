from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / "data-unexposed-v1" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
p13 = {row["id"]: row for row in json.loads((HERE / "baseline" / "qwen3-lora-seed13-best-dev" / "predictions.json").read_text(encoding="utf-8"))}
p42 = {row["id"]: row for row in json.loads((HERE / "runs" / "qwen3-lora-seed42" / "predictions" / "best-dev" / "diagnostic-120.json").read_text(encoding="utf-8"))}
decision = json.loads((HERE / "results" / "replication-decision.json").read_text(encoding="utf-8"))
assert set(p13) == set(p42) == {row["id"] for row in rows}


def correct(prediction):
    return prediction["gold"] == prediction["prediction"]


def export(row):
    left, right = p13[row["id"]], p42[row["id"]]
    if right["prediction"] == "coexist" and row["action_index"] != 0:
        error_type = "missed_conflict"
    elif right["prediction"] == "replace_all" and row["uncovered_old_count"] > 0:
        error_type = "unsafe_replace"
    else:
        error_type = "other_confusion"
    return {"id": row["id"], "family": row["family"], "domain": row["domain"], "cluster_id": row["cluster_id"],
            "gold": row["action"], "seed13_prediction": left["prediction"], "seed42_prediction": right["prediction"],
            "error_type_seed42": error_type, "uncovered_old_count": row["uncovered_old_count"],
            "retained_old_text": row["retained_old_text"], "old": row["pair"]["old"]["text"],
            "new": row["pair"]["new"]["text"], "seed13_probabilities": json.dumps(left["probabilities"]),
            "seed42_probabilities": json.dumps(right["probabilities"])}


shared = [export(row) for row in rows if not correct(p13[row["id"]]) and not correct(p42[row["id"]])]
errors42 = [export(row) for row in rows if not correct(p42[row["id"]])]
assert len(shared) == decision["paired_prediction_outcomes"]["neither_correct"] == 8
assert len(errors42) == 120 - decision["seed_comparison"][1]["new_correct"] == 16
fields = list(shared[0])
for name, selected in (("shared-errors.csv", shared), ("seed42-errors.csv", errors42)):
    with (HERE / "results" / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
print(json.dumps({"shared_errors": len(shared), "seed42_errors": len(errors42),
                  "seed42_error_types": Counter(row["error_type_seed42"] for row in errors42),
                  "shared_error_families": Counter(row["family"] for row in shared),
                  "shared_error_gold": Counter(row["gold"] for row in shared)}, ensure_ascii=False))
