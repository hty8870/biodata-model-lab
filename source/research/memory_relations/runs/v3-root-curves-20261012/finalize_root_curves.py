"""根规模曲线研究聚合/CPU诊断验证；235复用，绝不更改主候选/lock或读取终测。"""
from __future__ import annotations

from collections import Counter
import gc
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
from research.memory_relations import features as F
from research.memory_relations import metrics as M
from research.memory_relations import train as T
from research.memory_relations import finetune as FT

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mr3_curve_finalizer", BASE / "root_curve_runner.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
L = C.L


def diagnostic(model_dir, dev):
    scorer = F.load_scorer(model_dir, device="cpu")
    probabilities = scorer.predict_proba_raw([row["request"] for row in dev])
    labels = T._labels(dev)
    predicted = probabilities.argmax(axis=1)
    if probabilities.shape != (405, 7) or not np.isfinite(probabilities).all():
        raise AssertionError("公共loader开发分类诊断不符")
    weights = M._group_weights(T._root_ids(dev))
    result = {"scope": "raw classifier dev diagnostic only, non-human proxy labels, not final selection",
              "n_rows": len(dev), "n_roots": len(set(T._root_ids(dev))), "row_accuracy": float((predicted == labels).mean()),
              "root_equal_accuracy": float(np.average(predicted == labels, weights=weights)),
              "row_macro_f1": M.macro_f1(labels, predicted), "root_equal_macro_f1": M.macro_f1(labels, predicted, sample_weight=weights),
              "public_loader_device": "cpu", "cpu_is_verification_not_training": True}
    del scorer
    gc.collect()
    return result


def presentation_counts(train, seed):
    indices = T.nested_group_indices(train, [len(train)], seed)[len(train)]
    sample = [train[index] for index in indices]
    roots = T._root_ids(sample)
    counts = Counter()
    for _step, _epoch, chunks, _end in FT._training_windows(len(sample), epochs=3, batch_size=8, grad_accum=2,
                                                        max_steps=400, rng=np.random.default_rng(seed)):
        for chunk in chunks:
            counts.update(roots[index] for index in chunk)
    if sum(counts.values()) != 6400:
        raise AssertionError("呈现实例重放不符")
    return {"source": "deterministic exact frozen nested sampling/update-window replay, actual total verified in GPU metrics",
            "counts_by_root": dict(counts), "total_instances": 6400, "minimum_per_root": min(counts.values()),
            "maximum_per_root": max(counts.values()), "mean_per_root": sum(counts.values()) / len(counts)}


def main():
    destination = BASE / "curve-verification-final.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = C.load(BASE / "sampling-manifest.json")
    C.check_protected(manifest)
    complete = C.load(BASE / "completed-runs-final.json")["runs"]
    if len(complete) != 6 or {(row["roots"], row["seed"]) for row in complete} != {(size, seed) for size in C.SIZES for seed in L.SEEDS}:
        raise AssertionError("六曲线真实run缺失")
    source_rows = T.load_rows(manifest["source"], source="data", selected_splits=("train", "dev", "calibration"), calibration_stage="candidate")
    dev = [row for row in source_rows if row["split"] == "dev"]
    all_rows, source_protection = [], {}
    for size in (60, 120, 235):
        if size == 235:
            references = C.load(C.MAIN / "training-verification-final.json")["runs"]
            selected = [row for row in references if row["arm"] == "new-adjudicated"]
            train = [row for row in source_rows if row["split"] == "train"]
            sources = [(seed, C.MAIN / "new-adjudicated", None) for seed in L.SEEDS]
        else:
            info = manifest["subsets"][str(size)]
            if L.sha(info["development"]) != info["sha256"]:
                raise AssertionError("冻结根子集变动")
            subset = T.load_rows(info["development"], source="data", selected_splits=("train", "dev", "calibration"), calibration_stage="candidate")
            train = [row for row in subset if row["split"] == "train"]
            chosen = [row for row in complete if row["roots"] == size]
            sources = [(row["seed"], Path(row["output"]), row) for row in chosen]
        for seed, directory, reference in sources:
            report = C.load(directory / "metrics.json")
            run = next(row for row in report["runs"] if row["seed"] == seed)
            if reference is not None:
                C.single_result(reference["slug"], directory, len(train), seed, reference["spawn_pid"])
            if run["budget"]["actual_updates"] != 400 or run["budget"]["actual_sample_instances"] != 6400 \
                    or len(run["updates"]) != 400 or not run["parity"]["ok"] \
                    or run["selection"]["weighting"] != "root_equal" or run["selection"]["state_restored"] is not True:
                raise AssertionError("真实固定预算/rootdev restore/parity不符")
            model = directory / "scorers" / f"finetuned-joint-size{len(train)}-seed{seed}"
            meta = C.load(model / "meta.json")
            if L.sha(model / "model.pt") != meta["weights_sha256"] or meta["threshold"] is not None or meta["threshold_feasible"] is not False:
                raise AssertionError("候选保存hash/未审计形状不符")
            artifact = {"model.pt": L.sha(model / "model.pt"), "meta.json": L.sha(model / "meta.json")}
            classification = diagnostic(model, dev)
            if artifact != {name: L.sha(model / name) for name in artifact}:
                raise AssertionError("CPU读取修改工件")
            trace = C.load(directory / "device-traces" / f"finetuned-joint-size{len(train)}-seed{seed}.json")
            all_rows.append({"roots": size, "seed": seed, "not_main": True, "eligible_for_final_selection": False,
                             "train_rows": len(train), "reused_main235": size == 235, "extra_training_for235": False,
                             "best_dev_root_loss": run["selection"]["best_dev_value"], "selected_step": run["selection"]["selected_step"],
                             "device": run["device"], "actual_interpreter_pid": trace["pid"], "parent_pid": trace.get("parent_pid"),
                             "actual_optimizer_steps": trace["actual_optimizer_steps"], "actual_instances": run["budget"]["actual_sample_instances"],
                             "real_tokens": run["budget"]["real_tokens"], "padding_tokens": run["budget"]["padding_tokens"],
                             "train_and_dev_peak_allocated_mb": run["memory_stages"]["train_and_dev"]["peak_allocated_mb"],
                             "presentation_counts": presentation_counts(train, seed), "truncation_train_sample": run["truncation_train_sample"],
                             "classification_diagnostic": classification, "weights_sha256": meta["weights_sha256"],
                             "model_dir": str(model), "training_seconds": run["train_seconds"], "reload_parity": run["parity"]})
    aggregate = {}
    for size in (60, 120, 235):
        records = [row for row in all_rows if row["roots"] == size]
        values = [row["best_dev_root_loss"] for row in records]
        aggregate[str(size)] = {"n_seeds": 3, "root_dev_loss_mean": statistics.mean(values), "root_dev_loss_sample_sd": statistics.stdev(values),
                               "row_accuracy_mean": statistics.mean(row["classification_diagnostic"]["row_accuracy"] for row in records),
                               "row_accuracy_sample_sd": statistics.stdev(row["classification_diagnostic"]["row_accuracy"] for row in records),
                               "root_equal_macro_f1_mean": statistics.mean(row["classification_diagnostic"]["root_equal_macro_f1"] for row in records),
                               "root_equal_macro_f1_sample_sd": statistics.stdev(row["classification_diagnostic"]["root_equal_macro_f1"] for row in records)}
    C.check_protected(manifest)
    result = {"status": "VALIDATED_ROOT_CURVE_DIAGNOSTIC_ONLY", "runs": all_rows, "aggregate": aggregate,
              "sampling_seed": C.SAMPLING_SEED, "nested_60_120_235": True, "source_line_bytes_preserved": True,
              "six_new_real_cuda_runs": 6, "main235_three_reused_no_retrain": 3, "not_main": True, "eligible_for_final_selection": False,
              "protected_main_before": manifest["protected_main_binding"], "protected_main_after": C.protected_binding(),
              "fixed_main_lock_and_selected_weights_meta_unchanged": True, "no_selection_lock_or_audit_called": True,
              "main_source_sha256": manifest["main_source_sha256"], "source_sha256": manifest["source_sha256"],
              "audit_terminal_old_challenge_read": False, "human_gold": False, "training_stage_complete": True,
              "diagnostic_interpretation": "same400updates/6400instances, different roots/tokens/repetitions; dev used for tuning, no terminal generalization or risk claim"}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "aggregate": aggregate, "fixed_main_lock_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
