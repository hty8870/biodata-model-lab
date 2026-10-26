from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from research.memory_relations import finetune as FT

from action_common_20261024 import ROOT, dump, evaluate, initialize, load_rows, selftest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    output = ROOT / "probe-fit-36"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    checks = selftest()
    pool = load_rows("train")
    rows = [row for action in range(3) for row in [item for item in pool if item["action_index"] == action][:12]]
    assert len(rows) == 36
    scorer = initialize(args.repo_root, "a3", 13)
    model, tokenizer = scorer.model.to(device), scorer.tokenizer
    optimizer = FT.build_joint_optimizer(model, train_backbone=True)
    labels = np.asarray([row["action_index"] for row in rows])
    initial, _ = evaluate(model, tokenizer, rows, "a3", device)
    observations = []
    passed = {"value": False}

    def windows():
        for window in FT.training_windows(len(rows), epochs=1, batch_size=8, grad_accum=4, max_steps=100, rng=np.random.default_rng(13)):
            if passed["value"]:
                break
            yield window

    def loss_fn(logits, chunk, window):
        target = torch.as_tensor(labels[chunk], device=device)
        return torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum"), (logits.argmax(dim=1) == target).sum()

    def observe(event):
        if event["event"] == "update" and event["step"] in (25, 50, 75, 100):
            result, _ = evaluate(model, tokenizer, rows, "a3", device)
            observations.append({"step": event["step"], "metrics": result})
            passed["value"] = event["step"] >= 50 and result["correct"] >= 35 and result["loss"] <= initial["loss"] * .2
            print({"step": event["step"], "correct_of_36": result["correct"], "ce": result["loss"], "passed": passed["value"]}, flush=True)

    fitted = FT.fit_joint(model, tokenizer, [""] * len(rows), optimizer=optimizer, windows=windows(), planned_steps=100,
                          planned_instances=3200, max_length=384, device=device, bf16=FT.resolve_bf16("on", device),
                          loss_fn=loss_fn, checkpoint_selection="final", require_backbone_grad=True, clip_grad_norm=1,
                          event_callback=observe, eval_every_steps=25,
                          encode_fn=lambda chunk: FT.encode_prepared_views(tokenizer, [rows[int(i)]["encoding"] for i in chunk], device))
    del fitted["continuation"]
    final, predictions = evaluate(model, tokenizer, rows, "a3", device)
    dump(output / "result.json", {"passed": passed["value"], "scope": "36 train-only balanced examples; weights discarded; not generalization",
                                 "initial": initial, "final": final, "observations": observations, "fit": fitted,
                                 "selftest": checks, "predictions": predictions, "seconds": time.perf_counter() - started,
                                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "weights_saved": False})
    assert passed["value"], "train-only fitting probe failed: diagnose before main contrast"


if __name__ == "__main__":
    main()
