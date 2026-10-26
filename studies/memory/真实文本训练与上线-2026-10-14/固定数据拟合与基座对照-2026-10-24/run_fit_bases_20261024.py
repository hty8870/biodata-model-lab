"""Minimal reusable driver for the 2026-10-24 fixed-fit / base comparison study.

Subcommands (all GPU work is explicit; nothing falls back to CPU):
  selftest       loss-target / perfect-wrong / summarize self-tests
  audit          frozen-input, duplicate-label and renderer checks
  smoke          short real CUDA check for an alternative base (gate before training)
  train          the one configuration per base; MiniLM runs segmented 300..1500
  control        the single MiniLM head-lr 1e-4 comparison (only if 1500 is insufficient)
  replay         re-run frozen-checkpoint diagnostics from a finished run
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_common_20261024 as C              
from research.memory_relations import finetune as FT              

RESULTS = HERE / "results"
PREDICTIONS = HERE / "predictions"

RUN_NAMES = {"minilm": "minilm-a3-seed13",
             "minilm-head-lr-1e-4": "minilm-headlr1e-4-seed13",
             "macbert": "macbert-seed13",
             "qwen3": "qwen3-lora-seed13"}


def read(path):
    return C.read_json(path)


def summarize_for_csv(summary: dict) -> dict:
    row = {"n": summary["n"], "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
           "ce": summary["loss"], "correct": summary["correct"]}
    for name in C.ACTIONS:
        item = summary["per_class"][name]
        row[f"recall_{name}"] = item["recall"]
        row[f"f1_{name}"] = item["f1"]
    for key in ("missed_conflict_count", "conflict_denominator", "unsafe_replace_count", "merge_denominator",
                "coexist_replace_loss_count", "preservation_denominator", "unsafe_replace_total_count",
                "lost_preserved_items", "extra_merge_count"):
        row[key] = summary.get(key)
    return row


def write_predictions_csv(path: Path, predictions, split_label: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "split_group", "gold", "prediction", "correct", "p_coexist", "p_replace_all",
              "p_merge_preserve", "cluster_id", "domain"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            values = prediction["probabilities"]
            writer.writerow({"id": prediction["id"], "split_group": split_label, "gold": prediction["gold"],
                             "prediction": prediction["prediction"],
                             "correct": int(prediction["gold"] == prediction["prediction"]),
                             "p_coexist": f"{values[0]:.10g}", "p_replace_all": f"{values[1]:.10g}",
                             "p_merge_preserve": f"{values[2]:.10g}",
                             "cluster_id": prediction["cluster_id"], "domain": prediction["domain"]})
    return {"path": str(path), "rows": len(predictions), "bytes": path.stat().st_size,
            "sha256": C.sha256_file(path)}


def triplet_report(rows, predictions) -> dict:
    """Reuse the already-reviewed triplet logic instead of writing a second implementation."""
    sys.path.insert(0, str(C.ANCHOR / "asymmetric-head-counterfactual-20261024"))
    from evaluate_matched_heads_20261024 import triplet_metrics
    result = triplet_metrics(rows, predictions)
    return {key: result[key] for key in ("triplets_all_correct", "triplets_n",
                                          "conflict_transition_both_correct",
                                          "preservation_transition_both_correct", "transition_n",
                                          "by_family", "whole_replacement_slices")}


def attach_minilm_encodings(rows, tokenizer, max_length: int) -> dict:
    profile = C.assert_fits(tokenizer, [row["visible_text"] for row in rows], max_length, "minilm/confirmation")
    for row in rows:
        encoded = tokenizer(row["visible_text"], add_special_tokens=True)
        keys = [key for key in tokenizer.model_input_names if key in encoded]
        row["encoding"] = {key: list(encoded[key]) for key in keys}
    return profile


def smoke_gate(base: str) -> dict:
    path = C.SMOKES / f"{base}.json"
    assert path.is_file(), f"no successful smoke for {base}: run `smoke --base {base}` first"
    record = read(path)
    assert record["ok"] is True, f"smoke for {base} failed"
    for split, profile in record["token_profiles"].items():
        assert profile["max"] <= record["max_length"], f"{base}/{split} needs truncation"
    return record


def save_checkpoint(kind: str, directory: Path, model, state: dict, meta: dict) -> dict:
    model.load_state_dict(state, strict=True)
    if kind == "qwen3":
        head = {name: tensor.detach().cpu() for name, tensor in model.classifier.state_dict().items()}
        return C.save_qwen_checkpoint(directory, model, head, meta)
    return C.save_full_checkpoint(directory / "weights.pt", state, meta)


def evaluate_all(kind, model, tokenizer, device, max_length, groups: dict) -> dict:
    evaluate = C.evaluator(kind, model, tokenizer, device, max_length)
    return {name: evaluate(rows) for name, rows in groups.items()}


def persist_evaluation(run_dir: Path, tag: str, kind, model, tokenizer, device, max_length,
                       groups: dict) -> dict:
    evaluated = evaluate_all(kind, model, tokenizer, device, max_length, groups)
    payload, csvs = {}, {}
    for name, (summary, predictions) in evaluated.items():
        payload[name] = summary
        C.dump(run_dir / "predictions" / tag / f"{name}.json", predictions)
        csvs[name] = write_predictions_csv(RESULTS / f"rows-{run_dir.name}-{tag}-{name}.csv", predictions, name)
    C.dump(run_dir / f"metrics-{tag}.json", payload)
    return {"metrics": payload, "predictions": {name: f"predictions/{tag}/{name}.json" for name in evaluated},
            "csv": csvs}


def diagnostic_groups(kind: str, tokenizer) -> dict:
    groups = {"diagnostic-120": C.diagnostic_rows()}
    confirmations = C.confirmation_rows(C.mother_tokenizer())
    if kind.startswith("minilm"):
        attach_minilm_encodings(confirmations, tokenizer, 384)
    groups["confirmation-300"] = confirmations
    return groups


def cmd_selftest(args) -> int:
    C.AUDITS.mkdir(parents=True, exist_ok=True)
    reference = C.AC.selftest()
    rows = C.load_rows("train")
    targets = [row["action_index"] for row in rows]
    device = FT.resolve_device("cuda:0")
    model, tokenizer = C.build_minilm()
    model.to(device)
    optimizer = C.build_optimizer("minilm", model)
    encode_fn = C.make_encode_fn("minilm", tokenizer, rows, device, 384)
    loss_fn = C.make_loss(targets, device)
    window = next(FT.training_windows(len(rows), epochs=1, batch_size=8, grad_accum=4, max_steps=1,
                                      rng=np.random.default_rng(C.SEED)))
    chunk = window[2][0]
    loss_check = C.loss_target_selftest(loss_fn, targets, rows, chunk, device)
    probe = C.gradient_and_update_probe(model=model, optimizer=optimizer, encode_fn=encode_fn, loss_fn=loss_fn,
                                        chunk=chunk, device=device, bf16=True, frozen=False, label="minilm-selftest")
    payload = {"anchor_action_common_selftest": reference, "loss_target": loss_check,
               "gradient_update": {key: probe[key] for key in probe if key not in ("stats", "deltas")},
               "initial_state_sha256": probe["initial_state_sha256"]}
    C.dump(C.AUDITS / "selftest-minilm.json", payload)
    print(json.dumps({"selftest": True, "loss_target": loss_check,
                      "gradient_update": {key: payload["gradient_update"][key] for key in
                                          ("nonzero_grad_tensors", "changed_tensors", "head_changed_tensors",
                                           "backbone_changed_tensors")}}, ensure_ascii=False))
    return 0


def cmd_audit(args) -> int:
    C.AUDITS.mkdir(parents=True, exist_ok=True)
    mother_tokenizer = C.mother_tokenizer()
    lock = C.make_or_check_lock()
    report = {"lock_sha256": C.sha256_file(HERE / "run-lock.json"),
              "frozen": lock["inputs"]["actual_sha256"], "mother": lock["inputs"]["mother"],
              "models": {name: {"revision": item["revision"], "files": item["files"]}
                         for name, item in lock["inputs"]["models"].items()},
              "renderer_reproduction": C.renderer_reproduction_check(mother_tokenizer)}
    rows = {split: C.load_rows(split) for split in ("train", "dev", "heldout")}
    report["splits"] = {split: {"n": len(group),
                                "labels": {C.ACTIONS[i]: sum(1 for row in group if row["action_index"] == i)
                                           for i in range(3)},
                                "domains": len({row["domain"] for row in group}),
                                "clusters": len({row["cluster_id"] for row in group})}
                        for split, group in rows.items()}
    report["minilm_profiles"] = {split: {"n": len(group),
                                         "sum": int(sum(len(row["encoding"]["input_ids"]) for row in group)),
                                         "max": max(len(row["encoding"]["input_ids"]) for row in group),
                                         "min": min(len(row["encoding"]["input_ids"]) for row in group)}
                                 for split, group in rows.items()}
    profiles = {"minilm": report["minilm_profiles"]}
    for name, builder, max_length in (("macbert", C.build_macbert, 512), ("qwen3", C.build_qwen, 512)):
        _, tokenizer = builder()
        profiles[name] = {split: C.assert_fits(tokenizer, [row["visible_text"] for row in group], max_length,
                                               f"{name}/{split}")
                          for split, group in rows.items()}
        profiles[name]["diagnostic-120"] = C.assert_fits(tokenizer, [row["visible_text"] for row in C.diagnostic_rows()],
                                                         max_length, f"{name}/diagnostic")
        profiles[name]["confirmation-300"] = C.assert_fits(
            tokenizer, [row["visible_text"] for row in C.confirmation_rows(mother_tokenizer)], max_length,
            f"{name}/confirmation")
        del tokenizer
        gc.collect()
    report["token_profiles"] = profiles
    report["duplicate_labels"] = C.duplicate_label_audit(mother_tokenizer)
    assert report["duplicate_labels"]["identical_text_conflicting_labels"] == 0
    assert report["duplicate_labels"]["identical_encoding_conflicting_labels"] == 0
    assert report["duplicate_labels"]["identical_text_across_splits"] == 0
    C.dump(C.AUDITS / "input-audit.json", report)
    print(json.dumps({"audit": True, "token_max": {name: {split: profile["max"] for split, profile in value.items()}
                                                   for name, value in profiles.items()},
                      "duplicates": report["duplicate_labels"]["groups"]}, ensure_ascii=False))
    return 0


def cmd_smoke(args) -> int:
    base = args.base
    assert base in ("macbert", "qwen3")
    C.SMOKES.mkdir(parents=True, exist_ok=True)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = C.build(base)
    settings = C.effective_settings(base)
    max_length = settings["max_length"]
    rows = C.load_rows("train")
    groups = {"train": rows, "dev": C.load_rows("dev"), "diagnostic": C.diagnostic_rows()}
    confirmations = C.confirmation_rows(C.mother_tokenizer())
    if base.startswith("minilm"):
        attach_minilm_encodings(confirmations, tokenizer, max_length)
    groups["confirmation"] = confirmations
    profiles = {name: C.assert_fits(tokenizer, [row["visible_text"] for row in group], max_length,
                                    f"{base}/{name}") for name, group in groups.items()}
    model.to(device)
    optimizer = C.build_optimizer(base, model)
    audit = C.parameter_audit(model, optimizer, f"smoke-{base}")
    targets = [row["action_index"] for row in rows]
    encode_fn = C.make_encode_fn(base, tokenizer, rows, device, max_length)
    loss_fn = C.make_loss(targets, device)
    window = next(FT.training_windows(len(rows), epochs=1, batch_size=settings["micro"],
                                      grad_accum=settings["accum"], max_steps=1, rng=np.random.default_rng(C.SEED)))
    chunk = window[2][0]
    loss_check = C.loss_target_selftest(loss_fn, targets, rows, chunk, device)
    probe = C.gradient_and_update_probe(model=model, optimizer=optimizer, encode_fn=encode_fn, loss_fn=loss_fn,
                                        chunk=chunk, device=device, bf16=settings["bf16"],
                                        frozen=base == "qwen3", label=f"smoke-{base}")
    record = {"ok": True, "base": base, "settings": settings, "environment": C.environment(),
              "token_profiles": profiles, "max_length": max_length,
              "model": {"class": type(model).__name__, "backbone": type(model.backbone).__name__,
                        "trainable_parameters": audit["trainable_parameters"],
                        "total_parameters": audit["total_parameters"], "dtypes": audit["dtypes"]},
              "lora": audit.get("lora"), "optimizer_groups": audit["groups"],
              "loss_target": loss_check,
              "gradient_update": {key: probe[key] for key in probe if key not in ("stats", "deltas")},
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "seconds": round(time.perf_counter() - started, 2),
              "script_sha256": C.sha256_file(Path(__file__))}
    C.dump(C.SMOKES / f"{base}.json", record)
    print(json.dumps({"smoke_ok": True, "base": base,
                      "max_tokens": {name: profile["max"] for name, profile in profiles.items()},
                      "gradient": {key: record["gradient_update"][key] for key in
                                   ("nonzero_grad_tensors", "changed_tensors", "frozen_params_with_grad")},
                      "peak_gib": round(record["cuda_peak_allocated_bytes"] / 2 ** 30, 2),
                      "seconds": record["seconds"]}, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    kind = args.kind
    run_name = RUN_NAMES[kind]
    run_dir = C.RUNS / run_name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"run directory already exists: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    lock = C.make_or_check_lock()
    settings = C.effective_settings(kind)
    segments = settings["updates"] if isinstance(settings["updates"], list) else [settings["updates"]]
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    smoke = smoke_gate(kind) if kind in ("macbert", "qwen3") else None
    train_rows, dev_rows = C.load_rows("train"), C.load_rows("dev")
    targets = [row["action_index"] for row in train_rows]
    model, tokenizer = C.build(kind)
    max_length = settings["max_length"]
    groups = {"train": train_rows, "dev": dev_rows}
    profiles = {}
    if not kind.startswith("minilm"):
        profiles = {name: C.assert_fits(tokenizer, [row["visible_text"] for row in group], max_length,
                                        f"{kind}/{name}") for name, group in groups.items()}
    model.to(device)
    base_sha_before = frozen_base_sha256(model) if kind == "qwen3" else None
    optimizer = C.build_optimizer(kind, model)
    audit = C.parameter_audit(model, optimizer, f"train-{kind}")
    encode_fn = C.make_encode_fn(kind, tokenizer, train_rows, device, max_length)
    loss_fn = C.make_loss(targets, device)
    window = next(FT.training_windows(len(train_rows), epochs=1, batch_size=settings["micro"],
                                      grad_accum=settings["accum"], max_steps=1, rng=np.random.default_rng(C.SEED)))
    chunk = window[2][0]
    loss_check = C.loss_target_selftest(loss_fn, targets, train_rows, chunk, device)
    probe = C.gradient_and_update_probe(model=model, optimizer=optimizer, encode_fn=encode_fn, loss_fn=loss_fn,
                                        chunk=chunk, device=device, bf16=settings["bf16"],
                                        frozen=kind == "qwen3", label=f"train-{kind}")
    initial_state_sha256 = probe["initial_state_sha256"]
    if kind == "minilm-head-lr-1e-4":
        main = read(C.RUNS / RUN_NAMES["minilm"] / "result.json")
        assert initial_state_sha256 == main["initial_state_sha256"], \
            "head-lr control does not start from the main run's initial state"
    evaluate = C.evaluator(kind, model, tokenizer, device, max_length)
    initial = {"train": evaluate(train_rows)[0], "dev": evaluate(dev_rows)[0]}
    C.dump(run_dir / "initial.json", initial)
    training = C.train_segments(kind=kind, model=model, tokenizer=tokenizer, rows=train_rows, targets=targets,
                                seed=C.SEED, segments=segments, micro=settings["micro"], accum=settings["accum"],
                                max_length=max_length, device=device, bf16=settings["bf16"], optimizer=optimizer,
                                encode_fn=encode_fn, eval_dev=lambda: evaluate(dev_rows)[0],
                                eval_train=lambda: evaluate(train_rows)[0], output=run_dir,
                                frozen=kind == "qwen3", require_backbone_grad=True,
                                clip_norm=settings["clip_norm"], eval_every=50,
                                fit_target=C.PROTOCOL["fit_target"] if kind.startswith("minilm") else None)
    state = training["state"]
    terminal_state = state["terminal_state"]
    assert C.state_sha256(model.state_dict()) == C.state_sha256(terminal_state), \
        "model is not holding the terminal weights"
    terminal = persist_evaluation(run_dir, "terminal", kind, model, tokenizer, device, max_length,
                                  {"train": train_rows, "dev": dev_rows})
    terminal_meta = C.checkpoint_meta(kind, int(state["updates"][-1]["step"]), "terminal",
                                      {"updates": len(state["updates"]), "fit_target_reached": training["stopped"]})
    terminal_ckpt = save_checkpoint(kind, run_dir / "checkpoints" / "terminal", model, terminal_state, terminal_meta)
    best_state = state["best_state"]
    assert best_state is not None, "no dev snapshot was selected"
    model.load_state_dict(best_state, strict=True)
    best = persist_evaluation(run_dir, "best-dev", kind, model, tokenizer, device, max_length,
                              {"train": train_rows, "dev": dev_rows})
    best_meta = C.checkpoint_meta(kind, int(state["selected_step"]), "best-dev",
                                  {"selected_epoch": state["selected_epoch"], "best_dev_macro_f1": state["best_value"]})
    best_ckpt = save_checkpoint(kind, run_dir / "checkpoints" / "best-dev", model, best_state, best_meta)
                                                                               
    diagnostics = diagnostic_groups(kind, tokenizer)
    diag_payload, diag_csv = {}, {}
    evaluate_frozen = C.evaluator(kind, model, tokenizer, device, max_length)
    for name, rows in diagnostics.items():
        summary, predictions = evaluate_frozen(rows)
        diag_payload[name] = summary
        C.dump(run_dir / "predictions" / "best-dev" / f"{name}.json", predictions)
        diag_csv[name] = write_predictions_csv(RESULTS / f"rows-{run_name}-best-dev-{name}.csv", predictions, name)
        if name == "diagnostic-120":
            diag_payload[name]["triplets"] = triplet_report(rows, predictions)
    C.dump(run_dir / "diagnostics.json", diag_payload)
    base_sha_after = frozen_base_sha256(model) if kind == "qwen3" else None
    if kind == "qwen3":
        assert base_sha_before == base_sha_after, "frozen Qwen base weights changed during training"
    fit = read(run_dir / "fit.json")
    result = {
        "run": run_name, "kind": kind, "environment": C.environment(),
        "run_lock_sha256": C.sha256_file(HERE / "run-lock.json"),
        "config": {"settings": settings, "segments": segments, "seed": C.SEED,
                   "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                   "label_counts": {C.ACTIONS[i]: sum(1 for t in targets if t == i) for i in range(3)},
                   "max_length": max_length, 
                   "optimizer_groups": [{"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
                                        for group in optimizer.param_groups],
                   "smoke_gate": None if smoke is None else {"path": str(C.SMOKES / f"{kind}.json"),
                                                             "sha256": C.sha256_file(C.SMOKES / f"{kind}.json"),
                                                             "cuda_peak_allocated_bytes": smoke["cuda_peak_allocated_bytes"]},
                   "token_profiles": profiles, "script_sha256": C.sha256_file(Path(__file__)),
                   "model_builder": {"class": type(model).__name__, "backbone": type(model.backbone).__name__}},
        "initial": initial, "initial_state_sha256": initial_state_sha256,
        "frozen_base_sha256": {"before": base_sha_before, "after": base_sha_after},
        "audits": {"parameters": audit, "loss_target": loss_check,
                   "gradient_update": {key: probe[key] for key in probe if key not in ("stats", "deltas")}},
        "training": {"segments": training["records"], "stopped_at_target": training["stopped"],
                     "reached_at_update": training["reached_at"],
                     "seconds": round(training["seconds"], 3), "selection": fit["selection"],
                     "budget": fit["budget"], "segments_run": fit["segments_run"]},
        "terminal": {"step": terminal_meta["step"], **terminal, "checkpoint": terminal_ckpt},
        "best_dev": {"step": best_meta["step"], "best_dev_macro_f1": state["best_value"],
                     "selected_epoch": state["selected_epoch"], **best, "checkpoint": best_ckpt},
        "diagnostics": {"metrics": diag_payload, "csv": diag_csv,
                        "note": "already-exposed diagnostic replay; never used for training or selection"},
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "seconds": round(time.perf_counter() - started, 3),
        "no_deploy": True,
    }
    C.dump(run_dir / "result.json", result)
    write_metrics_table()
    print(json.dumps({"completed": True, "run": run_name,
                      "fit_target_reached": training["stopped"], "reached_at": training["reached_at"],
                      "terminal": {"train_accuracy": terminal["metrics"]["train"]["accuracy"],
                                   "train_ce": terminal["metrics"]["train"]["loss"],
                                   "dev_macro_f1": terminal["metrics"]["dev"]["macro_f1"]},
                      "best_dev": {"step": best_meta["step"],
                                   "dev_macro_f1": best["metrics"]["dev"]["macro_f1"],
                                   "dev_accuracy": best["metrics"]["dev"]["accuracy"]},
                      "diagnostic_macro_f1": diag_payload["diagnostic-120"]["macro_f1"],
                      "confirmation_macro_f1": diag_payload["confirmation-300"]["macro_f1"],
                      "seconds": result["seconds"]}, ensure_ascii=False))
    return 0


def frozen_base_sha256(model) -> str:
    state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
             if "lora_" not in name and not name.startswith("classifier.")}
    return C.state_sha256(state)


def write_metrics_table() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in sorted(C.RUNS.glob("*/result.json")):
        result = read(result_path)
        for tag, key in (("terminal", "terminal"), ("best-dev", "best_dev")):
            for split, summary in result[key]["metrics"].items():
                rows.append({"run": result["run"], "checkpoint": tag, "split": split,
                             **summarize_for_csv(summary)})
        for split, summary in result["diagnostics"]["metrics"].items():
            row = {"run": result["run"], "checkpoint": "best-dev-frozen", "split": split,
                   **summarize_for_csv(summary)}
            if "triplets" in summary:
                for key in ("triplets_all_correct", "triplets_n", "conflict_transition_both_correct",
                            "preservation_transition_both_correct"):
                    row[key] = summary["triplets"][key]
            rows.append(row)
    fields = ["run", "checkpoint", "split"] + [key for key in summarize_for_csv(
        {"n": 0, "accuracy": 0, "macro_f1": 0, "loss": 0, "correct": 0,
         "per_class": {name: {"recall": 0, "f1": 0} for name in C.ACTIONS}}) ]
    fields += ["triplets_all_correct", "triplets_n", "conflict_transition_both_correct",
               "preservation_transition_both_correct"]
    with (RESULTS / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_replay(args) -> int:
    run_dir = C.RUNS / args.run
    result = read(run_dir / "result.json")
    kind = result["kind"]
    settings = C.effective_settings(kind)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    model, tokenizer = C.build(kind)
    model.to(device)
    load_checkpoint(kind, run_dir / "checkpoints" / "best-dev", model)
    diagnostics = diagnostic_groups(kind, tokenizer)
    evaluate = C.evaluator(kind, model, tokenizer, device, settings["max_length"])
    payload = {}
    for name, rows in diagnostics.items():
        summary, predictions = evaluate(rows)
        payload[name] = summary
        C.dump(run_dir / "predictions" / "best-dev" / f"{name}.json", predictions)
        write_predictions_csv(RESULTS / f"rows-{args.run}-best-dev-{name}.csv", predictions, name)
        if name == "diagnostic-120":
            payload[name]["triplets"] = triplet_report(rows, predictions)
    C.dump(run_dir / "diagnostics.json", payload)
    print(json.dumps({"replayed": args.run, "diagnostic_macro_f1": payload["diagnostic-120"]["macro_f1"],
                      "confirmation_macro_f1": payload["confirmation-300"]["macro_f1"]}, ensure_ascii=False))
    return 0


def load_checkpoint(kind: str, directory: Path, model) -> None:
    if kind == "qwen3":
        from peft import PeftModel
        adapter = directory / "adapter"
        assert adapter.is_dir(), f"missing adapter: {adapter}"
                                                                                   
        base = model.backbone
        loaded = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        for name, parameter in loaded.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        model.backbone = loaded.to(model.classifier.weight.device)
        head = torch.load(directory / "head.pt", map_location="cpu", weights_only=False)
        model.classifier.load_state_dict(head["classifier"], strict=True)
        return
    payload = torch.load(directory / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--base", required=True, choices=("macbert", "qwen3"))
    smoke.set_defaults(func=cmd_smoke)
    train = sub.add_parser("train")
    train.add_argument("--kind", required=True, choices=tuple(RUN_NAMES))
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(func=cmd_train)
    replay = sub.add_parser("replay")
    replay.add_argument("--run", required=True)
    replay.set_defaults(func=cmd_replay)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
