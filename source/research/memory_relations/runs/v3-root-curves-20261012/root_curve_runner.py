"""预注册not-main根规模曲线：label-blind nested根、完整后代、六CUDA run，主锁永不改。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random
import sys

BASE = Path(__file__).resolve().parent
PREFLIGHT = BASE.parent / "v3-training-preflight-20261012"
MAIN = BASE.parent / "v3-data-ablation-20261012"
spec = importlib.util.spec_from_file_location("mr3_curve_launcher", PREFLIGHT / "formal_launcher.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
SAMPLING_SEED = 20261012
SIZES = (60, 120)
DIMENSIONS = ("author", "domain")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def root(row):
    value = row.get("phenomena", {}).get("root_id")
    if not isinstance(value, str) or not value:
        raise ValueError("accepted新根必须有显式root_id，不能行/别名抽样")
    return value


def label_blind_order(rows, seed=SAMPLING_SEED):
    metadata = {}
    for row in rows:
        if row["split"] != "train":
            continue
        identity = root(row)
        values = tuple(row["phenomena"].get(field) for field in DIMENSIONS)
        if any(not isinstance(value, str) or not value for value in values) or identity in metadata and metadata[identity] != values:
            raise ValueError("根内作者/领域必须一致且完整")
        metadata[identity] = values
    shuffled = sorted(metadata)
    random.Random(seed).shuffle(shuffled)
    rank = {identity: i for i, identity in enumerate(shuffled)}
    counts = [Counter() for _ in DIMENSIONS]
    remaining, ordered = set(shuffled), []
    while remaining:
        def score(identity):
            values = metadata[identity]
            unseen = sum(counts[i][value] == 0 for i, value in enumerate(values))
            return (-unseen, sum(counts[i][value] for i, value in enumerate(values)), rank[identity])
        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        ordered.append(chosen)
        for i, value in enumerate(metadata[chosen]):
            counts[i][value] += 1
    return ordered, metadata


def subset_bytes(records, selected_roots):
    selected = set(selected_roots)
    output = []
    for raw, row in records:
        split = row["split"]
        if split == "train":
            if root(row) in selected:
                output.append(raw)
        elif split == "dev":
            output.append(raw)
        elif split == "calibration" and row["phenomena"].get("calibration_role") in ("prob", "policy"):
            output.append(raw)
        else:
            raise ValueError("曲线开发源包含audit/终测/set或非candidate角色，拒绝")
    return b"".join(output)


def protected_binding():
    lock = load(L.FIXED_LOCK_DIR / "lock.json")
    model = Path(lock["model_dir"])
    paths = [L.FIXED_LOCK_DIR / "lock.json", L.FIXED_LOCK_DIR / "commitment.json", model / "model.pt", model / "meta.json",
             MAIN / "dev-only-selection.json", MAIN / "final-study-result.json"]
    return {str(path): L.sha(path) for path in paths}


def check_protected(manifest):
    if any(L.sha(path) != digest for path, digest in manifest["protected_main_binding"].items()):
        raise AssertionError("主工件/metadata/lock/commitment/选择结果变动，曲线停止")
    if any(L.sha(L.ROOT / "research/memory_relations" / name) != digest for name, digest in manifest["main_source_sha256"].items()):
        raise AssertionError("主源码变动")
    if L.sha(manifest["source"]) != manifest["source_sha256"]:
        raise AssertionError("accepted开发源变动")


def prepare():
    from research.memory_relations import train as T
    path = BASE / "sampling-manifest.json"
    if path.exists():
        raise FileExistsError("根采样已锁定，不再次采样/覆盖")
    dispatch = load(PREFLIGHT / "formal-dispatch-20261012.json")
    source = Path(dispatch["frozen_root"]) / "new-adjudicated/development.jsonl"
    if L.sha(source) != dispatch["development_sha256"]["new-adjudicated"]:
        raise AssertionError("accepted235开发源SHA不符")
    records = [(raw, json.loads(raw)) for raw in source.read_bytes().splitlines(keepends=True) if raw.strip()]
    ordered, metadata = label_blind_order([row for _raw, row in records])
    if len(ordered) != 235:
        raise AssertionError("完整accepted根必须235，不扩到未决240")
    manifest = {"schema": "memory-relations-root-size-diagnostic-v1", "status": "sampling_locked_before_fit",
                "source": str(source), "source_sha256": L.sha(source), "sampling_seed": SAMPLING_SEED,
                "root_order": ordered, "label_or_score_used_to_select_roots": False,
                "sampling_rule": "maximize uncovered author/domain then minimize accumulated root counts; seeded random rank breaks ties",
                "dimensions": list(DIMENSIONS), "not_main": True, "eligible_for_final_selection": False,
                "root_counts": [60, 120, 235], "all_descendants_included": True, "source_line_bytes_preserved": True,
                "protected_main_binding": protected_binding(),
                "main_source_sha256": load(MAIN / "launch-manifest.json")["source_sha256"],
                "full235_reuse": str(MAIN / "new-adjudicated"), "full235_additional_training": False,
                "created_at": datetime.now().isoformat(timespec="seconds"), "subsets": {}}
    L.write_new(BASE / "root-order-before-label-validation.json", {key: manifest[key] for key in
                ("root_order", "sampling_seed", "sampling_rule", "dimensions", "label_or_score_used_to_select_roots")})
    original_train_counts = Counter(root(row) for _raw, row in records if row["split"] == "train")
    common_contract = load(MAIN / "launch-manifest.json")["training_input_contract"]["new-adjudicated"]
    for size in SIZES:
        chosen = ordered[:size]
        folder = BASE / f"subsets/roots-{size}"
        folder.mkdir(parents=True, exist_ok=False)
        development = folder / "development.jsonl"
        with development.open("xb") as handle:
            handle.write(subset_bytes(records, chosen))
        args = T.build_parser().parse_args(["--data", str(development), "--eval-splits", "none", "--calibration-protocol", "partitioned"])
        prepared = T.prepare_data(args)
        contract = T._data_contract(prepared)
        train = prepared["pairs"]["train"]
        T.preflight_training(train, [len(train)], L.SEEDS, BASE / f"validation-only-{size}-no-write")
        if set(T._root_ids(train)) != set(chosen) or Counter(T._root_ids(train)) != {identity: original_train_counts[identity] for identity in chosen}:
            raise AssertionError("完整根/后代漏行或越界")
        if any(contract["sha256_by_partition"][role] != common_contract["sha256_by_partition"][role] for role in ("dev", "prob", "policy")):
            raise AssertionError("共同开发分区被改动")
        coverage = {field: dict(Counter(metadata[identity][i] for identity in chosen)) for i, field in enumerate(DIMENSIONS)}
        manifest["subsets"][str(size)] = {"root_ids": chosen, "development": str(development), "sha256": L.sha(development),
                                        "train_rows": len(train), "coverage_roots": coverage, "training_input_contract": contract,
                                        "seven_classes_checked_after_root_lock_no_reselection": True}
    check_protected(manifest)
    L.write_new(path, manifest)
    return manifest


def single_result(slug, directory, n_train, seed, spawn_pid):
    original_seeds = L.SEEDS
    try:
        L.SEEDS = (seed,)
        return L.result_records(slug, directory, n_train, spawn_pid)[0]
    finally:
        L.SEEDS = original_seeds


def command(data, output, n_train, seed):
    embedding = L.ROOT / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
    cmd = L.command(sys.executable, data, output, n_train, embedding)
    cmd[cmd.index("--seeds") + 1] = str(seed)
    return cmd


def execute(manifest, *, resume=False):
    from scripts import quality_gate as Q
    check_protected(manifest)
    existing = sorted((BASE / "attempts").glob("attempt-*")) if (BASE / "attempts").exists() else []
    if existing and not resume:
        raise FileExistsError("已有曲线attempt，不能覆盖；失败后显式--resume原目录")
    if resume and (not existing or load(existing[-1] / "result.json")["status"] != "failed"):
        raise ValueError("只能续接同目录failed attempt")
    attempt = BASE / "attempts" / f"attempt-{len(existing) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    runtime = attempt / "runtime"
    runtime.mkdir()
    env, _sanitized, _cleared = Q._sanitized_environment(runtime)
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", PYTHONIOENCODING="utf-8", BIODATA_SEARCH_LIVE="0")
    resolved = load(existing[-1] / "result.json") if existing else {"runs": []}
    completed = {(row["roots"], row["seed"]): row for row in resolved["runs"]}
    runner_sha, wrapper_sha, launcher_sha = L.sha(__file__), L.sha(PREFLIGHT / "cuda_training_wrapper.py"), L.sha(PREFLIGHT / "formal_launcher.py")
    L.write_new(attempt / "plan.json", {"not_main": True, "eligible_for_final_selection": False, "root_counts": list(SIZES),
                "seeds": list(L.SEEDS), "max_steps": 400, "instances": 6400, "runner_sha256": runner_sha,
                "wrapper_sha256": wrapper_sha, "launcher_sha256": launcher_sha, "sampling_manifest_sha256": L.sha(BASE / "sampling-manifest.json")})
    with L.CudaLease(L.RUNS / ".cuda-exclusive-0.lock"), (attempt / "events.jsonl").open("x", encoding="utf-8") as journal:
        def event(value):
            journal.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **value}, ensure_ascii=False) + "\n")
            journal.flush()
        try:
            for size in SIZES:
                info = manifest["subsets"][str(size)]
                for seed in L.SEEDS:
                    check_protected(manifest)
                    if L.sha(info["development"]) != info["sha256"] or L.sha(__file__) != runner_sha \
                            or L.sha(PREFLIGHT / "cuda_training_wrapper.py") != wrapper_sha or L.sha(PREFLIGHT / "formal_launcher.py") != launcher_sha:
                        raise AssertionError("冻结子集/运行脚本变动")
                    if (size, seed) in completed:
                        row = completed[size, seed]
                        single_result(row["slug"], Path(row["output"]), info["train_rows"], seed, row["spawn_pid"])
                        event({"event": "reuse_completed_run_no_retrain", "roots": size, "seed": seed})
                        continue
                    slug = f"roots{size}-seed{seed}-a{len(existing) + 1}"
                    output = BASE / "runs" / f"roots-{size}" / f"seed-{seed}" / attempt.name
                    if output.exists():
                        raise FileExistsError("partial目录存在，保留不覆盖")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if L.shutil.disk_usage(BASE).free < L.disk_requirement(L.GPU_QA.stat().st_size):
                        raise RuntimeError("曲线空间不足")
                    gpu = L.record_gpu_preflight(0, allow_idle_contexts=True)
                    cmd = command(info["development"], output, info["train_rows"], seed)
                    event({"event": "starting_diagnostic_run", "roots": size, "seed": seed, "command": cmd, "gpu": gpu,
                           "not_main": True, "eligible_for_final_selection": False})
                    process = L.run_child(cmd, arm=slug, output=BASE, env=env, physical_index=0, physical_uuid=gpu["physical_uuid"], event=event)
                    if process["exit_code"]:
                        raise RuntimeError("真实曲线run失败，保全部partial/attempt")
                    checked = single_result(slug, output, info["train_rows"], seed, process["pid"])
                    row = {"roots": size, "seed": seed, "slug": slug, "output": str(output), "spawn_pid": process["pid"],
                           "record": checked, "not_main": True, "eligible_for_final_selection": False}
                    completed[size, seed] = row
                    checkpoint = attempt / f"completed-{len(completed):02d}.json"
                    L.write_new(checkpoint, {"runs": list(completed.values())})
                    event({"event": "completed_real_diagnostic_run", "roots": size, "seed": seed,
                           "actual_interpreter_pid": checked["gpu_training_proof"]["child_pid"], "updates": 400, "instances": 6400})
            check_protected(manifest)
            final = {"status": "completed", "runs": list(completed.values()), "not_main": True,
                     "eligible_for_final_selection": False, "main_lock_unchanged": True, "full235_retrained": False}
            L.write_new(attempt / "result.json", final)
            L.write_new(BASE / "completed-runs-final.json", {"runs": list(completed.values())})
        except Exception as error:
            L.write_new(attempt / "result.json", {"status": "failed", "error_type": type(error).__name__, "runs": list(completed.values()),
                        "all_partial_attempts_retained": True, "not_main": True})
            if not (BASE / "completed-runs.json").exists():
                L.write_new(BASE / "completed-runs.json", {"runs": list(completed.values())})
            event({"event": "failed", "error_type": type(error).__name__})
            raise
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = load(BASE / "sampling-manifest.json") if args.resume else prepare()
    print(json.dumps({"sampling_locked": True, "root_counts": [60, 120, 235], "subsets": manifest["subsets"],
                      "not_main": True, "fixed_main_lock_unchanged": True}, ensure_ascii=False), flush=True)
    if args.execute:
        execute(manifest, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
