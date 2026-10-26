from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = STUDY.parent / "原子关系跨模型复验与Qwen训练-2026-10-25"
sys.path.insert(0, str(PARENT))

import numpy as np
import torch
import train_relation_heads as T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(13, 42), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run_name = f"direct3-seed{args.seed}"
    lock_path = HERE / f"final-lock-{run_name}-terminal.json"
    lock = T.validate_final_lock(lock_path, run=run_name, checkpoint="terminal")
    if lock["selected_step"] != 300:
        raise RuntimeError("This regression requires the fixed terminal-300 checkpoint")
    rows = T.read_split("heldout", purpose="locked-final", unlock=lock)
    if len(rows) != 150:
        raise RuntimeError("The already exposed regression denominator changed")
    run = T.load_run(run_name)
    identity = {
        "schema": "terminal-baseline-regression/1",
        "run": run_name,
        "checkpoint": "terminal",
        "step": 300,
        "n": len(rows),
        "status": "already_exposed_regression_not_blind_final",
        "lock_sha256": T.sha256_file(lock_path),
        "checkpoint_files_sha256": lock["checkpoint_files_sha256"],
        "data_sha256": lock["data_sha256"]["heldout"],
        "runner_sha256": T.sha256_file(__file__),
    }
    if args.check_only:
        print(json.dumps(identity, ensure_ascii=False), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    start = time.perf_counter()
    built = T.reload_run(run, "terminal", device)
    tokenizer = built["tokenizer"]
    lengths = [len(tokenizer(row["visible_text"], add_special_tokens=True)["input_ids"]) for row in rows]
    if max(lengths) > T.MAX_LENGTH:
        raise RuntimeError("Regression input would be truncated")
    torch.cuda.reset_peak_memory_stats(device)
    result = T.evaluate_split(
        model=built["model"], tokenizer=tokenizer, rows=rows, arm="direct3",
        split="heldout", device=device, max_length=T.MAX_LENGTH, bf16=True,
        encode=T.default_encode(tokenizer, device, T.MAX_LENGTH), batch_size=16,
    )
    torch.cuda.synchronize(device)
    output = {
        **identity,
        "device": torch.cuda.get_device_name(device),
        "max_input_tokens": max(lengths),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "summary": result["summary"],
        "predictions": result["predictions"],
    }
    path = HERE / f"{run_name}-terminal-old150.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("lock_sha256", "data_sha256", "checkpoint_files_sha256"):
            if output[key] != previous[key]:
                raise RuntimeError(f"Existing regression identity differs: {key}")
        before = np.asarray([p["coarse_probabilities"] for p in previous["predictions"]])
        after = np.asarray([p["coarse_probabilities"] for p in output["predictions"]])
        difference = float(np.max(np.abs(before - after)))
        if difference > 1e-6 or output["summary"] != previous["summary"]:
            raise RuntimeError(f"Fresh reload regression differs: max probability delta={difference}")
        print(json.dumps({"verified_existing": str(path), "probability_max_delta": difference,
                          "summary": output["summary"]}, ensure_ascii=False), flush=True)
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"output": str(path), "summary": output["summary"]}, ensure_ascii=False), flush=True)
    del result, built
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
