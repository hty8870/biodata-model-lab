from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import conflict_common as K

ROOT = Path(__file__).resolve().parent
APPROVAL = ROOT / "audits/semantic-review-final-20261026.json"


def verify_run(route, seed):
    directory = K.RUNS_DIR / f"{route}-seed{seed}"
    lock = K.read_json(directory / "run-lock.json")
    result = K.read_json(directory / "result.json")
    assert lock["route"] == route and lock["seed"] == seed
    assert result["selection"]["terminal_step"] == K.UPDATES == 300
    assert lock["data"]["files"] == {s: K.sha256_file(K.PROCESSED_DIR / f"{s}.jsonl") for s in K.SPLITS}
    assert lock["runtime"]["device"] == "cuda:0"
    assert lock["runtime"]["cuda_peak_allocated_bytes"] > 0
    assert lock["runtime"]["parameter_devices_after_fit"] == ["cuda:0"]
    for checkpoint in ("terminal", "best-dev"):
        actual = K.checkpoint_file_shas(directory / "checkpoints" / checkpoint)
        expected = {name: value["sha256"] for name, value in lock["checkpoints"][checkpoint]["files"].items()}
        assert actual == expected
    return lock


def main():
    approval = K.read_json(APPROVAL)
    assert approval["status"] == "accepted_for_fixed_training"
    assert approval["processed_sha256"] == {s: K.sha256_file(K.PROCESSED_DIR / f"{s}.jsonl") for s in K.SPLITS}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    for seed in K.SEEDS:
        for route in K.TRAIN_ROUTES:
            name = f"{route}-seed{seed}"
            directory = K.RUNS_DIR / name
            if directory.exists():
                verify_run(route, seed)
                print(json.dumps({"run": name, "verified_existing_complete": True}), flush=True)
                continue
            log_path = ROOT / f"audits/train-{name}-20261026.log"
            start = time.perf_counter()
            print(json.dumps({"run": name, "event": "started", "log": str(log_path)}), flush=True)
            with log_path.open("x", encoding="utf-8") as stream:
                completed = subprocess.run([sys.executable, "-B", "train_conflict_arms.py", "train",
                                            "--route", route, "--seed", str(seed), "--device", "cuda:0"],
                                           cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{name}: exit {completed.returncode}; preserve partial run and inspect {log_path}")
            lock = verify_run(route, seed)
            print(json.dumps({"run": name, "event": "completed", "seconds": time.perf_counter() - start,
                              "runtime": lock["runtime"]}), flush=True)
        left, right = (K.read_json(K.RUNS_DIR / f"{r}-seed{seed}/run-lock.json") for r in ("r2", "r3"))
        assert left["init"] == right["init"]
        assert left["plan"]["sha256"] == right["plan"]["sha256"]
        assert left["data"]["train"] == right["data"]["train"]
    print(json.dumps({"completed_fits": 6, "updates_per_fit": 300, "new_evaluation_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
