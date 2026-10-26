from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR
from research.memory_relations import semantic_model as SM

from diagnostic_common_20261023 import CLASSES, ROOT, analysis_selftest, dump, evaluate, load_inputs, sha

SEED = 13
MAX_STEPS = 300
MICROBATCH = 8
GRAD_ACCUM = 4
OBSERVE = {1, 10, 25, *range(50, MAX_STEPS + 1, 25)}
MOTHER = "artifacts/worktrees/semantic-formal-train-20261015/research/memory_relations/runs/v8-formal-preparation-20261015/raw-mother3-seed-13"


def sample_parameters(model):
    return {"encoder": model.backbone.encoder.layer[0].attention.self.query.weight,
            "head": model.classifier.weight}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / "run-seed13"
    run.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    assert torch.cuda.is_available(), "CUDA unavailable; no CPU fallback"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    mother_dir = args.repo_root / MOTHER
    mother = NR.load_nli_scorer(mother_dir, device="cpu")
    tokenizer = mother.tokenizer
    model = SM.initialize_arm(mother.model, "c5")
    assert torch.equal(mother.model.backbone.encoder.layer[0].attention.self.query.weight,
                       model.backbone.encoder.layer[0].attention.self.query.weight)
    del mother
    model.to(device)
    rows, encodings, labels, ids, audit = load_inputs(tokenizer)
    orients = [{"label": CLASSES[int(label)], "label_space": "semantic5"} for label in labels]
    optimizer = FT.build_joint_optimizer(model, train_backbone=True)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
    bf16 = FT.resolve_bf16("on", device)
    config = {"schema": "relation-fit-diagnostic-run-v1", "seed": SEED, "max_steps": MAX_STEPS,
              "microbatch": MICROBATCH, "grad_accum": GRAD_ACCUM, "oriented_batch": MICROBATCH * GRAD_ACCUM,
              "observe_steps": sorted(OBSERVE), "loss": "uniform semantic5 CE over all oriented rows; no public mixing",
              "criterion": {"canonical_correct_at_least": 99, "all_oriented_accuracy_at_least": 0.99, "ce_ratio_at_most": 0.2},
              "initialization": "original raw mother3 seed13 -> SM.initialize_arm(c5)",
              "mother_meta_sha256": sha(mother_dir / "nli_meta.json"), "mother_weights_sha256": sha(mother_dir / "nli_model.pt"),
              "model_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "optimizer": [{"lr": group["lr"], "weight_decay": group["weight_decay"], "n": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups],
              "clip_grad_norm": 1.0, "bf16_training": bf16, "eval_precision": "float32", "device": str(device),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
              "source_sha256": {name: sha(args.repo_root / "research/memory_relations" / name) for name in ("semantic_model.py", "finetune.py", "nli_reference.py", "metrics.py")},
              "script_sha256": {name: sha(ROOT / name) for name in ("build_diagnostic_20261023.py", "diagnostic_common_20261023.py", Path(__file__).name)},
              "input_audit": audit, "analysis_selftest": analysis_selftest(),
              "bypassed": ["formal all-pool prepare/cache", "qualified source_binding intake", "public/fine 50/50 sampler", "parent-NLL dev checkpoint selection", "production artifact export"],
              "scope": "train-set fit only, not generalization or original formal-entry end-to-end proof"}
    dump(run / "run-config.json", config)
    initial, initial_predictions = evaluate(model, tokenizer, encodings, labels, ids, device)
    dump(run / "eval-step-0000.json", {"step": 0, "metrics": initial, "predictions": initial_predictions})
    sampled = sample_parameters(model)
    initial_parameters = {name: value.detach().clone() for name, value in sampled.items()}
    previous_parameters = {name: value.detach().clone() for name, value in sampled.items()}
    exposure = Counter()
    summaries = [{"step": 0, "metrics": initial}]
    stop = {"success": False}
    trace_path = run / "updates.jsonl"

    def loss_fn(logits, chunk, window):
        losses, hits = SM.orientation_losses(logits, [orients[int(i)] for i in chunk], "c5")
        return losses.sum(), hits.sum()

    check_logits = torch.tensor([[0.2, -0.1, 0.4, 0.1, -0.4], [-0.2, 0.5, 0.1, -0.3, 0.0]], device=device, requires_grad=True)
    check_labels = torch.tensor([0, 2], device=device)
    check_rows = [{"label_space": "semantic5", "label": CLASSES[int(i)]} for i in check_labels]
    check_loss = SM.orientation_losses(check_logits, check_rows, "c5")[0].mean()
    assert torch.allclose(check_loss, torch.nn.functional.cross_entropy(check_logits, check_labels))
    assert torch.allclose(torch.autograd.grad(check_loss, check_logits, retain_graph=True)[0],
                          torch.autograd.grad(torch.nn.functional.cross_entropy(check_logits, check_labels), check_logits)[0])
    dump(run / "loss-selftest.json", {"same_as_torch_ce": True, "same_gradient_as_torch_ce": True,
                                    "fit_joint_normalization": "sum(chunk CE)/len(full window), accumulated across chunks"})

    def windows():
        for window in FT.training_windows(len(encodings), epochs=1, batch_size=MICROBATCH,
                                          grad_accum=GRAD_ACCUM, max_steps=MAX_STEPS, rng=np.random.default_rng(SEED)):
            if stop["success"]:
                break
            for chunk in window[2]:
                exposure.update(int(i) for i in chunk)
            yield window

    def observe(event):
        if event["event"] != "update":
            return
        step = event["step"]
        changes = {}
        for name, value in sampled.items():
            delta = value.detach() - previous_parameters[name]
            changes[name] = {"step_delta_abs_max": float(delta.abs().max()),
                             "from_initial_abs_max": float((value.detach() - initial_parameters[name]).abs().max()),
                             "grad_norm_after_clip": float(value.grad.detach().norm()) if value.grad is not None else None}
            previous_parameters[name].copy_(value.detach())
        event = dict(event, parameter_samples=changes, elapsed_seconds=time.perf_counter() - started,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        if step in OBSERVE:
            metrics, predictions = evaluate(model, tokenizer, encodings, labels, ids, device)
            dump(run / f"eval-step-{step:04d}.json", {"step": step, "metrics": metrics, "predictions": predictions})
            summaries.append({"step": step, "metrics": metrics})
            canonical = metrics["canonical"]
            passed = canonical["correct"] >= 99 and metrics["all_oriented"]["accuracy"] >= 0.99 and canonical["ce"] <= initial["canonical"]["ce"] * 0.2
            print(json.dumps({"step": step, "train_correct_of_100": canonical["correct"], "train_ce": canonical["ce"], "criterion_passed": passed}, ensure_ascii=False), flush=True)
            if passed:
                confirmation, _ = evaluate(model, tokenizer, encodings, labels, ids, device)
                assert confirmation == metrics
                stop["success"] = True

    fitted = FT.fit_joint(model, tokenizer, [""] * len(encodings), optimizer=optimizer, windows=windows(),
                          planned_steps=MAX_STEPS, planned_instances=MAX_STEPS * MICROBATCH * GRAD_ACCUM,
                          max_length=384, device=device, bf16=bf16, loss_fn=loss_fn,
                          evaluate=None, checkpoint_selection="final", require_backbone_grad=True,
                          clip_grad_norm=1.0, event_callback=observe, eval_every_steps=25,
                          encode_fn=lambda chunk: FT.encode_prepared_views(tokenizer, [encodings[int(i)] for i in chunk], device))
    continuation = fitted.pop("continuation")
    torch.save(continuation["terminal_state"], run / "diagnostic-terminal.pt")
    del continuation
    final, final_predictions = evaluate(model, tokenizer, encodings, labels, ids, device)
    dump(run / "final-predictions.json", final_predictions)
    result = {"schema": "relation-fit-diagnostic-result-v1", "criterion_passed": stop["success"],
              "initial": initial, "final": final, "observations": summaries, "fit": fitted,
              "exposure": {"distinct_oriented_rows": len(exposure), "distinct_canonical_rows": len({i // 2 for i in exposure}),
                           "sample_instances": sum(exposure.values()), "min_per_oriented_row": min(exposure.values()),
                           "max_per_oriented_row": max(exposure.values()), "oriented_counts": dict(exposure)},
              "seconds_total": time.perf_counter() - started, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "checkpoint_bytes": (run / "diagnostic-terminal.pt").stat().st_size,
              "checkpoint_sha256": sha(run / "diagnostic-terminal.pt"), "optimizer_saved": False}
    dump(run / "result.json", result)
    print(json.dumps({"criterion_passed": stop["success"], "steps": fitted["budget"]["actual_updates"], "correct_of_100": final["canonical"]["correct"], "ce": final["canonical"]["ce"], "seconds_total": result["seconds_total"]}), flush=True)


if __name__ == "__main__":
    main()
