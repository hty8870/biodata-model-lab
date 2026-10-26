from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_common_20261024 as C
from research.memory_relations import finetune as FT
from run_fit_bases_20261024 import triplet_report, write_predictions_csv
from verify_run_20261024 import reload_qwen

SOURCE_RUN = HERE.parent / "固定数据拟合与基座对照-2026-10-24" / "runs" / "qwen3-lora-seed13"
OUTPUT = HERE / "baseline" / "qwen3-lora-seed13-best-dev"


def main() -> int:
    assert not OUTPUT.exists(), f"baseline output already exists: {OUTPUT}"
    lock = C.make_or_check_lock()
    rows = C.diagnostic_rows()
    problems = []
    model, tokenizer, reload_info = reload_qwen(SOURCE_RUN, "best-dev", problems)
    assert not problems, problems
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    model.to(device)
    evaluate = C.evaluator("qwen3", model, tokenizer, device, C.effective_settings("qwen3")["max_length"])
    summary, predictions = evaluate(rows)
    summary["triplets"] = triplet_report(rows, predictions)
    C.dump(OUTPUT / "predictions.json", predictions)
    C.dump(OUTPUT / "metrics.json", summary)
    csv_receipt = write_predictions_csv(OUTPUT / "predictions.csv", predictions, "diagnostic-120")
    checkpoint = SOURCE_RUN / "checkpoints" / "best-dev"
    checkpoint_files = {str(path.relative_to(checkpoint)): {"bytes": path.stat().st_size,
                                                             "sha256": C.sha256_file(path)}
                        for path in sorted(checkpoint.rglob("*")) if path.is_file()}
    receipt = {
        "source_run": str(SOURCE_RUN), "source_result_sha256": C.sha256_file(SOURCE_RUN / "result.json"),
        "checkpoint_tag": "best-dev", "checkpoint_files": checkpoint_files,
        "run_lock_sha256": C.sha256_file(HERE / "run-lock.json"),
        "data": {name: C.sha256_file(C.DIAGNOSTIC_DIR / name) for name in ("rows.jsonl", "inputs.jsonl", "manifest.json")},
        "script_sha256": C.sha256_file(Path(__file__)), "environment": C.environment(),
        "strict_reload": reload_info, "strict_reload_problems": problems,
        "predictions_sha256": C.sha256_file(OUTPUT / "predictions.json"),
        "metrics_sha256": C.sha256_file(OUTPUT / "metrics.json"), "csv": csv_receipt,
        "prediction_count": len(predictions), "new_set_used_for_training_or_selection": False,
        "no_deploy": True,
    }
    C.dump(OUTPUT / "receipt.json", receipt)
    print(json.dumps({"baseline": "qwen3-lora-seed13/best-dev", "n": len(predictions),
                      "macro_f1": summary["macro_f1"], "accuracy": summary["accuracy"],
                      "missed_conflict_count": summary.get("missed_conflict_count"),
                      "unsafe_replace_total_count": summary.get("unsafe_replace_total_count"),
                      "triplets_all_correct": summary["triplets"]["triplets_all_correct"],
                      "strict_reload_problems": problems}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
