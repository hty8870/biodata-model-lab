from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import action_common_20261024 as AC
from research.memory_relations import finetune as FT

SEEDS = (13, 20260919)
ARMS = ("c6", "f6")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_sha(model):
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def copy_spread(head, gradients=False):
    weight = head.weight.grad if gradients else head.weight
    bias = head.bias.grad if gradients else head.bias
    return max(float((weight[:4] - weight[0]).detach().abs().max()),
               float((bias[:4] - bias[0]).detach().abs().max()))


def initialize(repo, seed):
    scorer = AC.initialize(repo, "f6", seed)
    generator = torch.Generator().manual_seed(seed + 418)
    noise = torch.randn(4, scorer.model.classifier.in_features, generator=generator) * .005
    noise -= noise.mean(dim=0, keepdim=True)
    with torch.no_grad():
        scorer.model.classifier.weight[:4].add_(noise)
    assert copy_spread(scorer.model.classifier) > 1e-5
    return scorer


def make_lock(repo):
    old = read(ROOT.parent / "design-lock.json")
    files = ["data-v1/train.jsonl", "data-v1/dev.jsonl", "inputs-v1/train.jsonl", "inputs-v1/dev.jsonl", "action_common_20261024.py"]
    for name in files:
        assert FT.file_sha256(ROOT.parent / name) == old["inputs_sha256"][name], name
    new_files = [Path(__file__), ROOT / "设计-破缺六头与最小变化诊断-2026-10-24.md", ROOT / "data-v2/manifest.json",
                 ROOT / "data-v2/rows.jsonl", ROOT / "data-v2/inputs.jsonl"]
    value = {"seeds": list(SEEDS), "arms": list(ARMS), "updates": 300, "microbatch": 8, "grad_accum": 4,
             "planned_instances": 9600, "noise_std": .005, "noise_seed_offset": 418,
             "training_data": "original 720 train / 180 dev; new counterfactual diagnostic never used",
             "selection": "same action dev macroF1 every 50 updates; ties earliest; no early stop",
             "old_sources_sha256": {name: FT.file_sha256(ROOT.parent / name) for name in files},
             "new_sources_sha256": {path.name: FT.file_sha256(path) for path in new_files},
             "finetune_sha256": FT.file_sha256(FT.__file__), "repo_head_expected": "86a982ff7d441c019378e3c2f81132b5751ef44c",
             "deploy": False, "save_optimizer": False}
    path = ROOT / "training-lock.json"
    if path.exists():
        assert read(path) == value, "training protocol changed"
    else:
        AC.dump(path, value)
    return value


def train(repo, arm, seed, lock):
    output = ROOT / "runs" / f"{arm}-seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    checks = AC.selftest()
    train_rows, dev = AC.load_rows("train"), AC.load_rows("dev")
    assert len(train_rows) == 720 and len(dev) == 180
    scorer = initialize(repo, seed)
    model, tokenizer = scorer.model, scorer.tokenizer
    initial_sha = state_sha(model)
    initial_parameter_spread = copy_spread(model.classifier)
    model.to(device)
    model.eval()
    encoded = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in train_rows[:8]], device)
    probe = torch.nn.functional.cross_entropy(AC.coarse_logits(model(**encoded), "f6"),
                                              torch.tensor([row["action_index"] for row in train_rows[:8]], device=device))
    probe.backward()
    initial_gradient_spread = copy_spread(model.classifier, gradients=True)
    assert initial_gradient_spread > 1e-9, "coarse-supervised child rows still symmetric"
    model.zero_grad(set_to_none=True)
    del probe, encoded
    initial_train, _ = AC.evaluate(model, tokenizer, train_rows, "f6", device)
    initial_dev, initial_predictions = AC.evaluate(model, tokenizer, dev, "f6", device)
    AC.dump(output / "initial.json", {"state_sha256": initial_sha, "copy_parameter_spread": initial_parameter_spread,
                                     "coarse_gradient_copy_spread": initial_gradient_spread,
                                     "train": initial_train, "dev": initial_dev, "dev_predictions": initial_predictions})
    optimizer = FT.build_joint_optimizer(model, train_backbone=True)
    targets = np.array([row["action_index"] if arm == "c6" else row["fine_index"] for row in train_rows])
    actions = np.array([row["action_index"] for row in train_rows])
    config = {"arm": arm, "seed": seed, "initial_state_sha256": initial_sha, "initial_parameter_spread": initial_parameter_spread,
              "initial_coarse_gradient_spread": initial_gradient_spread, "training_lock_sha256": FT.file_sha256(ROOT / "training-lock.json"),
              "script_sha256": FT.file_sha256(__file__), "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "python": platform.python_version(), "torch": str(torch.__version__), "parameters": sum(p.numel() for p in model.parameters()),
              "targets": dict(Counter(map(int, targets))), "bf16_training": True, "float32_evaluation": True,
              "clip_norm": 1, "optimizer": [{"lr": group["lr"], "weight_decay": group["weight_decay"], "eps": group["eps"]} for group in optimizer.param_groups],
              "new_diagnostic_used_in_training_or_selection": False, "optimizer_saved": False}
    AC.dump(output / "config.json", config)
    exposure, order_hash, observations = Counter(), hashlib.sha256(), []

    def windows():
        for window in FT.training_windows(len(train_rows), epochs=1, batch_size=8, grad_accum=4, max_steps=300,
                                          rng=np.random.default_rng(seed)):
            order_hash.update(b"window")
            for chunk in window[2]:
                exposure.update(map(int, chunk))
                order_hash.update(np.asarray(chunk, dtype="<i8").tobytes())
                order_hash.update(b"chunk")
            yield window

    def loss_fn(logits, chunk, window):
        parent = AC.coarse_logits(logits, "f6")
        selected = torch.as_tensor(targets[chunk], device=device)
        loss = torch.nn.functional.cross_entropy(parent if arm == "c6" else logits.float(), selected, reduction="sum")
        hits = (parent.detach().argmax(dim=1) == torch.as_tensor(actions[chunk], device=device)).sum()
        return loss, hits

    def on_event(event):
        record = dict(event, seconds=time.perf_counter() - started)
        if event["event"] == "update" and event["step"] in (1, 50, 100, 150, 200, 250, 300):
            record.update(copy_parameter_spread=copy_spread(model.classifier), copy_gradient_spread=copy_spread(model.classifier, gradients=True))
        with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if event["event"] == "dev_evaluation":
            train_summary, _ = AC.evaluate(model, tokenizer, train_rows, "f6", device)
            dev_summary, predictions = AC.evaluate(model, tokenizer, dev, "f6", device)
            observations.append({"step": event["step"], "train": train_summary, "dev": dev_summary})
            AC.dump(output / f"dev-{event['step']:04d}.json", {"metrics": dev_summary, "predictions": predictions})
            print(json.dumps({"arm": arm, "seed": seed, "step": event["step"], "train_correct": train_summary["correct"],
                              "dev_correct": dev_summary["correct"], "dev_macro_f1": dev_summary["macro_f1"],
                              "seconds": time.perf_counter() - started}), flush=True)

    fitted = FT.fit_joint(model, tokenizer, [""] * len(train_rows), optimizer=optimizer, windows=windows(), planned_steps=300,
                          planned_instances=9600, max_length=384, device=device, bf16=FT.resolve_bf16("on", device),
                          loss_fn=loss_fn, evaluate=lambda: AC.evaluate(model, tokenizer, dev, "f6", device)[0],
                          checkpoint_selection="best-dev", selection_metric="dev-macro-f1", patience=0, min_delta=0,
                          eval_every_steps=50, require_backbone_grad=True, clip_grad_norm=1, event_callback=on_event,
                          encode_fn=lambda chunk: FT.encode_prepared_views(tokenizer, [train_rows[int(i)]["encoding"] for i in chunk], device))
    del fitted["continuation"]
    torch.save(FT.cpu_state(model), output / "selected.pt")
    train_summary, train_predictions = AC.evaluate(model, tokenizer, train_rows, "f6", device)
    dev_summary, dev_predictions = AC.evaluate(model, tokenizer, dev, "f6", device)
    AC.dump(output / "train-predictions.json", train_predictions)
    AC.dump(output / "dev-predictions.json", dev_predictions)
    exposure_sha = hashlib.sha256(json.dumps(sorted(exposure.items())).encode()).hexdigest()
    assert len(fitted["updates"]) == 300 and sum(exposure.values()) == 9600 and len(exposure) == 720
    result = {"config": config, "initial_train": initial_train, "initial_dev": initial_dev, "selected_train": train_summary,
              "selected_dev": dev_summary, "observations": observations, "fit": fitted,
              "selected_copy_parameter_spread": copy_spread(model.classifier), "batch_order_sha256": order_hash.hexdigest(),
              "exposure_sha256": exposure_sha, "exposure": {"instances": sum(exposure.values()), "distinct": len(exposure),
                                                           "min": min(exposure.values()), "max": max(exposure.values())},
              "checkpoint_sha256": FT.file_sha256(output / "selected.pt"), "checkpoint_bytes": (output / "selected.pt").stat().st_size,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "seconds": time.perf_counter() - started,
              "new_diagnostic_opened": False}
    AC.dump(output / "result.json", result)
    print(json.dumps({"completed": True, "arm": arm, "seed": seed, "selected_step": fitted["selection"]["selected_step"],
                      "dev_macro_f1": dev_summary["macro_f1"], "seconds": result["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    lock = make_lock(args.repo_root)
    for seed in SEEDS:
        for arm in ARMS:
            train(args.repo_root, arm, seed, lock)
            gc.collect()
            torch.cuda.empty_cache()
        left = read(ROOT / "runs" / f"c6-seed{seed}" / "result.json")
        right = read(ROOT / "runs" / f"f6-seed{seed}" / "result.json")
        for key in ("batch_order_sha256", "exposure_sha256"):
            assert left[key] == right[key], key
        assert left["config"]["initial_state_sha256"] == right["config"]["initial_state_sha256"]
        assert left["fit"]["budget"] == right["fit"]["budget"]
        li = read(ROOT / "runs" / f"c6-seed{seed}" / "initial.json")["dev_predictions"]
        ri = read(ROOT / "runs" / f"f6-seed{seed}" / "initial.json")["dev_predictions"]
        delta = float(np.max(np.abs(np.asarray([x["probabilities"] for x in li]) - np.asarray([x["probabilities"] for x in ri]))))
        assert delta <= 1e-6
        AC.dump(ROOT / f"pair-verification-seed{seed}.json", {"initial_state_equal": True, "batch_order_equal": True,
                                                             "exposure_and_budget_equal": True, "initial_dev_probability_max_delta": delta})
    AC.dump(ROOT / "training-complete.json", {"runs": [f"{arm}-seed{seed}" for seed in SEEDS for arm in ARMS],
                                             "new_diagnostic_scored": False, "production_changed": False})


if __name__ == "__main__":
    main()
