"""九个真实CUDA run与唯一dev锁最终验收；只开发probe，不读audit/终测，不重训练。"""
from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from research.memory_relations import features as F

BASE = Path(__file__).resolve().parent
PREFLIGHT = BASE.parent / "v3-training-preflight-20261012"
spec = importlib.util.spec_from_file_location("mr3_final_launcher", PREFLIGHT / "formal_launcher.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    destination = BASE / "training-verification-final.json"
    if destination.exists():
        raise FileExistsError(destination)
    final, selection = load(BASE / "final-study-result.json"), load(BASE / "dev-only-selection.json")
    if final["status"] != "completed" or final["completed_candidate_runs"] != 9 or final["audit_performed"]:
        raise AssertionError("九run未完整，不验收")
    dispatch = load(PREFLIGHT / "formal-dispatch-20261012.json")
    intake = load(BASE / "intake-verification.json")
    original = load(BASE / "launch-manifest.json")
    continuation = load(BASE / "attempts/attempt-002/launch-manifest.json")
    if intake["status"] != "passed" or any(L.sha(L.ROOT / "research/memory_relations" / name) != value for name, value in original["source_sha256"].items()):
        raise AssertionError("派发/主源码保护不符")
    frozen = Path(dispatch["frozen_root"])
    for arm, digest in dispatch["development_sha256"].items():
        if L.sha(frozen / arm / "development.jsonl") != digest:
            raise AssertionError("冻结开发文件SHA变动")
        for role, shared_sha in dispatch["common_partition_sha256"].items():
            if L.sha(frozen / arm / f"{role}.jsonl") != shared_sha:
                raise AssertionError("共享开发原文件SHA变动")
    dev = [json.loads(line) for line in (frozen / L.ARMS[0] / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    probes = [row["request"] for row in dev[:8]]
    events = []
    for path in (BASE / "launch-events.jsonl", BASE / "attempts/attempt-002/launch-events.jsonl"):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    spawn = {value["arm"]: value["child_pid"] for value in events if value["event"] == "child_started"}
    records, rows, fingerprints, telemetry = [], [], set(), {}
    for arm in L.ARMS:
        directory = BASE / arm
        n_train = dispatch["train_rows_roots"][arm][0]
        checked = L.result_records(arm, directory, n_train, spawn[arm])
        records.extend(checked)
        report = load(directory / "metrics.json")
        manifest = load(directory / "planned_manifest.json")
        fingerprints.add(manifest["model"]["backbone_fingerprint"])
        for run in report["runs"]:
            if run["budget"]["planned_updates"] != 400 or run["budget"]["planned_sample_instances"] != 6400 \
                    or len(run["updates"]) != 400 or [row["step"] for row in run["updates"]] != list(range(1, 401)) \
                    or any(row["sample_instances"] != 16 for row in run["updates"]) \
                    or run["selection"]["dev_root_count"] != 40 or run["selection"]["snapshot_device"] != "cpu" \
                    or not run["parity"]["ok"]:
                raise AssertionError("真实完整更新/root监控/原loader parity不符")
            monitor = [value for value in run["train_history"] if value.get("dev_loss") is not None]
            minimum = min(value["dev_loss"] for value in monitor)
            if run["selection"]["best_dev_value"] != minimum:
                raise AssertionError("best-dev不是实际根loss最小点")
            artifact = directory / "scorers" / f"finetuned-joint-size{n_train}-seed{run['seed']}"
            meta = load(artifact / "meta.json")
            if L.sha(artifact / "model.pt") != meta["weights_sha256"] or meta["threshold"] is not None or meta["threshold_feasible"] is not False:
                raise AssertionError("保存工件hash/未审计候选形状不符")
            scorer = F.load_scorer(artifact, device="cpu")
            probabilities = scorer.predict_proba(probes)
            if tuple(scorer.classes) != F.CLASSES or probabilities.shape != (8, 7) or not np.isfinite(probabilities).all():
                raise AssertionError("最终真实公共loader CPU验证失败")
            del scorer
            gc.collect()
            proof = next(value["gpu_training_proof"] for value in checked if value["seed"] == run["seed"])
            rows.append({"arm": arm, "seed": run["seed"], "train_rows": n_train, "train_roots": dispatch["train_rows_roots"][arm][1],
                         "best_dev_root_loss": minimum, "selected_step": run["selection"]["selected_step"], "device": run["device"],
                         "actual_updates": 400, "actual_instances": 6400, "real_tokens": run["budget"]["real_tokens"],
                         "padding_tokens": run["budget"]["padding_tokens"], "train_and_dev_peak_allocated_mb": proof["train_and_dev_peak_allocated_mb"],
                         "peak_allocated_mb": run["memory_peak_mb"], "actual_interpreter_pid": proof["child_pid"], "spawn_pid": proof["spawn_pid"],
                         "pid_binding": proof["process_identity"], "backbone_fingerprint": manifest["model"]["backbone_fingerprint"],
                         "artifact_weights_sha256": meta["weights_sha256"], "public_loader_cpu_validation_only": True,
                         "original_training_parity": run["parity"], "training_seconds": run["train_seconds"]})
        samples = [json.loads(line) for line in (BASE / f"{arm}.gpu-telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        measured = [value["gpu"] for value in samples if value.get("gpu")]
        utilization = [value["utilization_gpu_pct"] for value in measured if value.get("utilization_gpu_pct") is not None]
        power = [value["power_w"] for value in measured if value.get("power_w") is not None]
        telemetry[arm] = {"samples": len(samples), "valid_samples": len(measured), "utilization_peak_pct": max(utilization),
                          "utilization_mean_pct": sum(utilization) / len(utilization), "power_peak_w": max(power),
                          "scope": "aggregate shared physical GPU, not per-process memory attribution"}
    if len(rows) != 9 or len(fingerprints) != 1:
        raise AssertionError("九run缺失或本地骨干指纹不一致")
    recomputed = L.selection(records)
    if recomputed != selection:
        raise AssertionError("不是固定dev均值/median规则选择")
    lock = load(L.FIXED_LOCK_DIR / "lock.json")
    commitment = load(L.FIXED_LOCK_DIR / "commitment.json")
    if lock["study_id"] != L.STUDY_ID or commitment["lock_sha256"] != L.sha(L.FIXED_LOCK_DIR / "lock.json") \
            or Path(lock["model_dir"]) != Path(selection["chosen"]["model_dir"]) or (L.FIXED_LOCK_DIR / "audit-attempt.json").exists():
        raise AssertionError("固定study唯一锁/未audit约束不符")
    for name, digest in lock["artifact_sha256"].items():
        if L.sha(Path(lock["model_dir"]) / name) != digest:
            raise AssertionError("锁定工件变动")
    size = sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file())
    result = {"status": "VALIDATED_NINE_REAL_CUDA_RUNS_AND_ONE_DEV_LOCK", "study_id": L.STUDY_ID, "runs": rows,
              "selection": selection, "fixed_lock_dir": str(L.FIXED_LOCK_DIR), "lock_sha256": commitment["lock_sha256"],
              "gpu_telemetry": telemetry, "formal_output_bytes": size, "current_disk_free_bytes": shutil.disk_usage(BASE).free,
              "same_updates_and_instances_not_tokens": True, "main_source_sha256_unchanged": original["source_sha256"],
              "development_sha256": dispatch["development_sha256"], "backbone_fingerprints": sorted(fingerprints),
              "original_failed_attempt_preserved": load(BASE / "launch-result.json"), "original_complete_arm_not_retrained": True,
              "continuation_source_fingerprints": {key: continuation[key] for key in ("launcher_sha256", "wrapper_sha256", "registration_sha256")},
              "audit_or_terminal_read": False, "calibration_used_for_final_selection": False, "human_gold": False,
              "cpu_parity_is_validation_not_training": True, "production_commit_push_deploy": False}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
