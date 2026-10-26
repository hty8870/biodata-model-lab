"""Independent strict reload + metric recomputation for one study run.

This script never reuses the training-time model object or the stored metrics: it
rebuilds each checkpoint from disk, re-encodes the frozen inputs, re-runs the
forward pass, recomputes every metric from raw predictions (including a second,
independent implementation in verify_action_experiment_20261024.manual) and checks
the CSV exports.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_common_20261024 as C              
from research.memory_relations import finetune as FT              

sys.path.insert(0, str(C.ANCHOR))
from verify_action_experiment_20261024 import manual              

TOLERANCE = 1e-5


def read(path):
    return C.read_json(path)


def compare_floats(recorded: dict, recomputed: dict, keys, prefix: str, problems: list) -> dict:
    checked = {}
    for key in keys:
        if key not in recorded or key not in recomputed:
            continue
        left, right = recorded[key], recomputed[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            difference = abs(float(left) - float(right))
            checked[f"{prefix}.{key}"] = difference
            if difference > TOLERANCE:
                problems.append(f"{prefix}.{key}: recorded {left} vs recomputed {right}")
        elif left != right:
            problems.append(f"{prefix}.{key}: recorded {left!r} vs recomputed {right!r}")
    return checked


def metric_check(rows, predictions, recorded: dict, label: str, problems: list) -> dict:
    probabilities = np.asarray([prediction["probabilities"] for prediction in predictions], dtype=np.float64)
    independent = C.AC.summarize(probabilities, rows)
    second = manual(rows, predictions)
    differences = compare_floats(recorded, independent, list(independent), label + ".summarize", problems)
    differences.update(compare_floats(recorded, second, list(second), label + ".manual", problems))
    if "matrix" in second:
        if recorded.get("confusion_matrix", {}).get("matrix") != second["matrix"]:
            problems.append(f"{label}: confusion matrix differs from manual recomputation")
    if not np.isclose(independent["loss"], second["loss"], atol=1e-9, rtol=0):
        problems.append(f"{label}: CE differs between summarize and manual")
    return {"differences": differences, "n": independent["n"],
            "accuracy": independent["accuracy"], "macro_f1": independent["macro_f1"],
            "ce": independent["loss"]}


def compare_predictions(stored, fresh, label: str, problems: list) -> dict:
    if len(stored) != len(fresh):
        problems.append(f"{label}: prediction count {len(stored)} vs {len(fresh)}")
        return {"rows": len(stored), "ids_equal": False}
    id_mismatch = [left["id"] for left, right in zip(stored, fresh) if left["id"] != right["id"]]
    if id_mismatch:
        problems.append(f"{label}: id order differs at {id_mismatch[:3]}")
    gold_mismatch = [left["id"] for left, right in zip(stored, fresh) if left["gold"] != right["gold"]]
    if gold_mismatch:
        problems.append(f"{label}: gold differs at {gold_mismatch[:3]}")
    argmax_mismatch = [left["id"] for left, right in zip(stored, fresh)
                       if left["prediction"] != right["prediction"]]
    if argmax_mismatch:
        problems.append(f"{label}: argmax differs at {argmax_mismatch[:3]}")
    maximum = 0.0
    for left, right in zip(stored, fresh):
        maximum = max(maximum, float(np.max(np.abs(np.asarray(left["probabilities"])
                                                   - np.asarray(right["probabilities"])))))
    if maximum > TOLERANCE:
        problems.append(f"{label}: probability max abs difference {maximum}")
    return {"rows": len(stored), "ids_equal": not id_mismatch, "argmax_equal": not argmax_mismatch,
            "probability_max_abs_difference": maximum}


def reload_qwen(run_dir: Path, tag: str, problems: list):
    """Independent reconstruction: plain AutoModel + adapter files + head, not the training builder."""
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    directory = run_dir / "checkpoints" / tag
    tokenizer = AutoTokenizer.from_pretrained(str(C.QWEN["dir"]), local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModel.from_pretrained(str(C.QWEN["dir"]), local_files_only=True, dtype=torch.bfloat16)
    base.config.use_cache = False
    adapter = directory / "adapter"
    backbone = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    for name, parameter in backbone.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    file_state = load_file(adapter / "adapter_model.safetensors")
    model_keys = {name: parameter for name, parameter in backbone.named_parameters() if "lora_" in name}
    file_to_model = {key: key.replace(".lora_A.weight", ".lora_A.default.weight")
                        .replace(".lora_B.weight", ".lora_B.default.weight") for key in file_state}
    missing = sorted(key for key, target in file_to_model.items() if target not in model_keys)
    unexpected = sorted(name for name in model_keys if name not in set(file_to_model.values()))
    if missing or unexpected:
        problems.append(f"qwen adapter key mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    for key, tensor in file_state.items():
        target = file_to_model[key]
        if target in model_keys and not torch.equal(tensor.float(),
                                                    model_keys[target].detach().cpu().float()):
            problems.append(f"qwen adapter tensor differs after load: {key}")
            break
    wrapper = C.PooledClassifier(backbone, backbone.config.hidden_size, pool="last_nonpad", seed=13,
                                 extra_forward={"use_cache": False})
    head = torch.load(directory / "head.pt", map_location="cpu", weights_only=False)
    wrapper.classifier.load_state_dict(head["classifier"], strict=True)
    return wrapper, tokenizer, {"adapter_keys": len(file_state), "lora_dtype": str(
        next(parameter.dtype for name, parameter in backbone.named_parameters() if "lora_" in name)),
        "file_state_tensors": len(file_state)}


def rebuild(run_dir: Path, kind: str, tag: str, problems: list):
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    if kind == "qwen3":
        model, tokenizer, extra = reload_qwen(run_dir, tag, problems)
        model.to(device)
        return model, tokenizer, extra
    model, tokenizer = C.build(kind)
    model.to(device)
    payload = torch.load(run_dir / "checkpoints" / tag / "weights.pt", map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, tokenizer, {"state_dict_tensors": len(payload["state_dict"]),
                              "meta": payload["meta"]}


def csv_check(path: Path, predictions, problems: list, label: str) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(predictions):
        problems.append(f"{label}: csv rows {len(rows)} vs predictions {len(predictions)}")
        return {"path": str(path), "rows": len(rows)}
    mismatch = 0
    correct = 0
    for row, prediction in zip(rows, predictions):
        values = [float(row["p_coexist"]), float(row["p_replace_all"]), float(row["p_merge_preserve"])]
        guess = C.ACTIONS[int(np.argmax(values))]
        if guess != prediction["prediction"] or row["id"] != prediction["id"]:
            mismatch += 1
        correct += int(row["gold"] == row["prediction"])
    if mismatch:
        problems.append(f"{label}: csv argmax/id mismatch in {mismatch} rows")
    return {"path": str(path), "rows": len(rows), "argmax_mismatch": mismatch,
            "csv_correct": correct, "sha256": C.sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = C.RUNS / args.run
    result = read(run_dir / "result.json")
    kind = result["kind"]
    problems: list[str] = []
    checks: dict = {}
    device = FT.resolve_device("cuda:0")

                                                  
    lock = C.make_or_check_lock()
    lock_now = C.sha256_file(HERE / "run-lock.json")
    checks["lock_sha256"] = {"recorded": result["run_lock_sha256"], "actual": lock_now}
    if result["run_lock_sha256"] != lock_now:
        problems.append("run-lock.json changed after this run")
    for name, digest in lock["inputs"]["actual_sha256"].items():
        if C.sha256_file(C.ANCHOR / name) != digest:
            problems.append(f"frozen input drift: {name}")
    for tag in ("terminal", "best-dev"):
        recorded = result["terminal" if tag == "terminal" else "best_dev"]["checkpoint"]
        if kind == "qwen3":
            if C.sha256_file(Path(recorded["directory"]) / "head.pt") != recorded["files"]["head.pt"]["sha256"]:
                problems.append(f"{tag}: head.pt sha mismatch")
            for name, item in recorded["files"].items():
                if C.sha256_file(Path(recorded["directory"]) / name) != item["sha256"]:
                    problems.append(f"{tag}: {name} sha mismatch")
        else:
            if C.sha256_file(recorded["path"]) != recorded["sha256"]:
                problems.append(f"{tag}: checkpoint sha mismatch")
    checks["checkpoint_hashes"] = "checked"

                                              
    groups = {"train": C.load_rows("train"), "dev": C.load_rows("dev")}
    for split, rows in groups.items():
        source = {row["id"]: row for row in C.read_jsonl(C.ANCHOR / "data-v1" / f"{split}.jsonl")}
        for row in rows:
            row["formal"] = source[row["id"]]["formal"]
    diagnostics = {"diagnostic-120": C.diagnostic_rows(),
                   "confirmation-300": C.confirmation_rows(C.mother_tokenizer())}
    if kind.startswith("minilm"):
        from run_fit_bases_20261024 import attach_minilm_encodings
        attach_minilm_encodings(diagnostics["confirmation-300"], C.mother_tokenizer(), 384)
    all_groups = dict(groups, **diagnostics)
    checks["reload"] = {}
    for tag, group_names in (("terminal", ("train", "dev")), ("best-dev", ("train", "dev"))):
        model, tokenizer, extra = rebuild(run_dir, kind, tag, problems)
        key = "terminal" if tag == "terminal" else "best_dev"
        checks["reload"][tag] = extra
        evaluate = C.evaluator(kind, model, tokenizer, device, C.effective_settings(kind)["max_length"])
        for name in group_names:
            rows = all_groups[name]
            summary, predictions = evaluate(rows)
            stored = read(run_dir / "predictions" / tag / f"{name}.json")
            checks.setdefault("predictions", {})[f"{tag}/{name}"] = compare_predictions(
                stored, predictions, f"{tag}/{name}", problems)
            checks.setdefault("metrics", {})[f"{tag}/{name}"] = metric_check(
                rows, stored, result[key]["metrics"][name], f"{tag}/{name}", problems)
            checks["metrics"][f"{tag}/{name}"]["fresh_accuracy"] = summary["accuracy"]
            checks["metrics"][f"{tag}/{name}"]["fresh_vs_recorded"] = compare_floats(
                result[key]["metrics"][name], summary, list(summary), f"{tag}/{name}.fresh", problems)
            checks["csv"] = checks.get("csv", {})
            checks["csv"][f"{tag}/{name}"] = csv_check(
                C.HERE / "results" / f"rows-{args.run}-{tag}-{name}.csv", stored,
                problems, f"{tag}/{name}")
        del model
        torch.cuda.empty_cache()

                                                                           
    model, tokenizer, extra = rebuild(run_dir, kind, "best-dev", problems)
    evaluate = C.evaluator(kind, model, tokenizer, device, C.effective_settings(kind)["max_length"])
    for name, rows in diagnostics.items():
        summary, predictions = evaluate(rows)
        stored = read(run_dir / "predictions" / "best-dev" / f"{name}.json")
        checks["predictions"][f"best-dev/{name}"] = compare_predictions(stored, predictions,
                                                                        f"best-dev/{name}", problems)
        checks["metrics"][f"best-dev/{name}"] = metric_check(
            rows, stored, result["diagnostics"]["metrics"][name], f"best-dev/{name}", problems)
        checks["csv"][f"best-dev/{name}"] = csv_check(
            C.HERE / "results" / f"rows-{args.run}-best-dev-{name}.csv", stored, problems, f"best-dev/{name}")
        if name == "diagnostic-120":
            sys.path.insert(0, str(C.ANCHOR / "asymmetric-head-counterfactual-20261024"))
            from evaluate_matched_heads_20261024 import triplet_metrics
            triplets = triplet_metrics(rows, stored)
            recorded = result["diagnostics"]["metrics"][name]["triplets"]
            for key in ("triplets_all_correct", "triplets_n", "conflict_transition_both_correct",
                        "preservation_transition_both_correct"):
                if triplets[key] != recorded[key]:
                    problems.append(f"triplets {key}: recorded {recorded[key]} vs recomputed {triplets[key]}")
            checks["triplets"] = {key: triplets[key] for key in ("triplets_all_correct", "triplets_n")}
    del model
    torch.cuda.empty_cache()

                                                                                          
    if kind == "qwen3":
        terminal_state = torch.load(run_dir / "checkpoints" / "terminal" / "head.pt", map_location="cpu",
                                    weights_only=False)["classifier"]
        best_state = torch.load(run_dir / "checkpoints" / "best-dev" / "head.pt", map_location="cpu",
                                weights_only=False)["classifier"]
        head_changed = any(not torch.equal(terminal_state[key], best_state[key]) for key in terminal_state)
    else:
        left = torch.load(run_dir / "checkpoints" / "terminal" / "weights.pt", map_location="cpu",
                          weights_only=False)["state_dict"]
        right = torch.load(run_dir / "checkpoints" / "best-dev" / "weights.pt", map_location="cpu",
                           weights_only=False)["state_dict"]
        head_changed = any(not torch.equal(left[key], right[key]) for key in left if key.startswith("classifier."))
    checks["terminal_differs_from_best_dev"] = bool(head_changed)
    checks["selection"] = result["training"]["selection"]
    if result["training"]["selection"]["mode"] != "final":
        problems.append("training selection mode is not final; terminal would not be the last update")
    if result["terminal"]["step"] < result["best_dev"]["step"]:
        problems.append("terminal step precedes best-dev step")

                                                                                      
    if True:
        from run_fit_bases_20261024 import frozen_base_sha256
        fresh, _ = C.build(kind)
        fresh_sha = C.state_sha256(FT.cpu_state(fresh))
        checks["initial_state_sha256"] = {"recorded": result["initial_state_sha256"], "rebuilt": fresh_sha}
        if fresh_sha != result["initial_state_sha256"]:
            problems.append("initial state is not reproducible from the recorded builder")
        if kind == "qwen3":
            rebuilt = frozen_base_sha256(fresh)
            checks["frozen_base_sha256"] = {"recorded": result["frozen_base_sha256"]["before"],
                                            "rebuilt": rebuilt,
                                            "train_end": result["frozen_base_sha256"]["after"]}
            if rebuilt != result["frozen_base_sha256"]["before"]:
                problems.append("Qwen base weights differ from the downloaded base")
            if result["frozen_base_sha256"]["before"] != result["frozen_base_sha256"]["after"]:
                problems.append("Qwen base weights changed during training")
        del fresh

    receipt = {"run": args.run, "kind": kind, "passed": not problems, "problems": problems,
               "checks": checks, "environment": C.environment(),
               "script_sha256": C.sha256_file(Path(__file__)),
               "locked": {"action_common": lock["inputs"]["code"]["action_common"],
                          "finetune": lock["inputs"]["code"]["finetune.py"],
                          "semantic_relations": lock["inputs"]["code"]["semantic_relations.py"]}}
    C.dump(run_dir / "verification.json", receipt)
    print(json.dumps({"run": args.run, "passed": receipt["passed"], "problems": problems[:10],
                      "diagnostic_macro_f1": checks["metrics"]["best-dev/diagnostic-120"]["macro_f1"],
                      "confirmation_macro_f1": checks["metrics"]["best-dev/confirmation-300"]["macro_f1"]},
                     ensure_ascii=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
