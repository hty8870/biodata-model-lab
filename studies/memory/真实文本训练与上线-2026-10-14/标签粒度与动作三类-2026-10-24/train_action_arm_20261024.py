from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.memory_relations import finetune as FT

from action_common_20261024 import ROOT, coarse_logits, dump, evaluate, initialize, labels_for, load_rows, selftest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("a3", "f6", "r6"), required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    lock = json.loads((ROOT / "design-lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "go" and [args.arm, args.seed] in lock["runs"]
    for name, digest in lock["inputs_sha256"].items():
        assert FT.file_sha256(ROOT / name) == digest, name
    output = ROOT / "runs" / f"{args.arm}-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    checks = selftest()
    train, dev = load_rows("train"), load_rows("dev")
    scorer = initialize(args.repo_root, args.arm, args.seed)
    model, tokenizer = scorer.model, scorer.tokenizer
    targets = labels_for(train, args.arm, args.seed)
    actions = np.asarray([row["action_index"] for row in train])
    model.to(device)
    optimizer = FT.build_joint_optimizer(model, train_backbone=True)
    steps = lock["updates"]
    config = {"arm": args.arm, "seed": args.seed, "updates": steps, "microbatch": 8, "accumulation": 4,
              "selection": "best action dev macro-F1; ties earliest; no per-arm early stop",
              "init": "raw pretrained mother encoder, shared random action head; fine rows copy parents minus log4",
              "device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "python": platform.python_version(),
              "model_parameters": sum(p.numel() for p in model.parameters()), "labels_used": dict(Counter(targets.tolist())),
              "shuffled_fine_labels_changed": int((targets != np.asarray([row["fine_index"] for row in train])).sum()) if args.arm == "r6" else 0,
              "script_sha256": FT.file_sha256(__file__), "common_sha256": FT.file_sha256(ROOT / "action_common_20261024.py"),
              "design_lock_sha256": FT.file_sha256(ROOT / "design-lock.json"), "selftest": checks,
              "optimizer": [{"lr": group["lr"], "weight_decay": group["weight_decay"]} for group in optimizer.param_groups],
              "clip_norm": 1, "eval_precision": "float32", "train_bf16": True, "optimizer_saved": False}
    dump(output / "config.json", config)
    initial_train, _ = evaluate(model, tokenizer, train, args.arm, device)
    initial_dev, initial_predictions = evaluate(model, tokenizer, dev, args.arm, device)
    dump(output / "initial.json", {"train": initial_train, "dev": initial_dev, "dev_predictions": initial_predictions})
    observations = []
    exposure = Counter()

    def windows():
        for window in FT.training_windows(len(train), epochs=1, batch_size=8, grad_accum=4,
                                          max_steps=steps, rng=np.random.default_rng(args.seed)):
            for chunk in window[2]:
                exposure.update(int(i) for i in chunk)
            yield window

    def loss_fn(logits, chunk, window):
        selected = torch.as_tensor(targets[chunk], device=device)
        loss = torch.nn.functional.cross_entropy(logits.float(), selected, reduction="sum")
        parent = coarse_logits(logits, args.arm)
        hits = (parent.detach().argmax(dim=1) == torch.as_tensor(actions[chunk], device=device)).sum()
        return loss, hits

    def on_event(event):
        record = dict(event, elapsed_seconds=time.perf_counter() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if event["event"] == "dev_evaluation":
            train_result, _ = evaluate(model, tokenizer, train, args.arm, device)
            dev_result, predictions = evaluate(model, tokenizer, dev, args.arm, device)
            observations.append({"step": event["step"], "train": train_result, "dev": dev_result})
            dump(output / f"dev-step-{event['step']:04d}.json", {"step": event["step"], "metrics": dev_result, "predictions": predictions})
            print(json.dumps({"arm": args.arm, "seed": args.seed, "step": event["step"],
                              "train_correct": train_result["correct"], "train_n": len(train),
                              "dev_correct": dev_result["correct"], "dev_n": len(dev),
                              "dev_macro_f1": dev_result["macro_f1"], "seconds": record["elapsed_seconds"]}), flush=True)

    fitted = FT.fit_joint(model, tokenizer, [""] * len(train), optimizer=optimizer, windows=windows(),
                          planned_steps=steps, planned_instances=steps * 32, max_length=384, device=device,
                          bf16=FT.resolve_bf16("on", device), loss_fn=loss_fn,
                          evaluate=lambda: evaluate(model, tokenizer, dev, args.arm, device)[0],
                          checkpoint_selection="best-dev", selection_metric="dev-macro-f1", patience=0,
                          min_delta=0, eval_every_steps=50, require_backbone_grad=True, clip_grad_norm=1,
                          event_callback=on_event,
                          encode_fn=lambda chunk: FT.encode_prepared_views(tokenizer, [train[int(i)]["encoding"] for i in chunk], device))
    del fitted["continuation"]
    torch.save(FT.cpu_state(model), output / "selected.pt")
    final_train, train_predictions = evaluate(model, tokenizer, train, args.arm, device)
    final_dev, dev_predictions = evaluate(model, tokenizer, dev, args.arm, device)
    dump(output / "train-predictions.json", train_predictions)
    dump(output / "dev-predictions.json", dev_predictions)
    result = {"config": config, "initial_train": initial_train, "initial_dev": initial_dev,
              "selected_train": final_train, "selected_dev": final_dev, "observations": observations,
              "fit": fitted, "seconds_total": time.perf_counter() - started,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "checkpoint_bytes": (output / "selected.pt").stat().st_size,
              "checkpoint_sha256": FT.file_sha256(output / "selected.pt"),
              "exposure": {"instances": sum(exposure.values()), "distinct_rows": len(exposure),
                           "min": min(exposure.values()), "max": max(exposure.values())},
              "heldout_opened": False}
    dump(output / "result.json", result)
    print(json.dumps({"completed": True, "arm": args.arm, "seed": args.seed, "selected": fitted["selection"],
                      "dev_macro_f1": final_dev["macro_f1"], "seconds": result["seconds_total"]}), flush=True)


if __name__ == "__main__":
    main()
