from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.memory_relations import finetune as FT

from action_common_20261024 import ROOT, dump, evaluate, initialize, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "runs/r6-seed13"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert result["checkpoint_sha256"] == FT.file_sha256(directory / "selected.pt")
    assert result["config"]["common_sha256"] == FT.file_sha256(ROOT / "action_common_20261024.py")
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    scorer = initialize(args.repo_root, "r6", 13)
    scorer.model.load_state_dict(torch.load(directory / "selected.pt", map_location="cpu", weights_only=True), strict=True)
    scorer.model.to(device)
    dev_summary, dev_predictions = evaluate(scorer.model, scorer.tokenizer, load_rows("dev"), "r6", device)
    reference = json.loads((directory / "dev-predictions.json").read_text(encoding="utf-8"))
    difference = float(np.abs(np.asarray([row["probabilities"] for row in reference]) - np.asarray([row["probabilities"] for row in dev_predictions])).max())
    assert difference <= 1e-6 and [row["prediction"] for row in reference] == [row["prediction"] for row in dev_predictions]
    summary, predictions = evaluate(scorer.model, scorer.tokenizer, load_rows("heldout"), "r6", device)
    dump(ROOT / "posthoc-r6-evaluation.json", {"scope": "post-hoc mechanism diagnostic on already-inspected validation; not new independent final",
                                             "decision_record_sha256": FT.file_sha256(ROOT / "补充诊断-R6资源规则偏离记录-2026-10-24.md"),
                                             "dev_reload_max_probability_difference": difference, "dev_summary": dev_summary,
                                             "summary": summary, "predictions": predictions})
    print(json.dumps({"scope": "posthoc_r6", "correct": summary["correct"], "n": 300, "macro_f1": summary["macro_f1"],
                      "unsafe_replace": summary["unsafe_replace_total_count"], "reload_max": difference}), flush=True)


if __name__ == "__main__":
    main()
