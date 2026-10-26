from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from research.memory_relations import finetune as FT

from action_common_20261024 import ROOT, dump, evaluate, initialize, load_rows, selftest


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def mean_forward(self, **batch):
    hidden = self.backbone(**batch).last_hidden_state
    return self.classifier(self.dropout(masked_mean(hidden, batch["attention_mask"])))


def configure(repo, device):
    scorer = initialize(repo, "a3", 13)
    scorer.model.forward = MethodType(mean_forward, scorer.model)
    scorer.model.to(device)
    return scorer


def pooling_selftest():
    hidden = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 0]])
    expected = torch.tensor([[2., 4.]])
    assert torch.equal(masked_mean(hidden, mask), expected)
    hidden[:, 2] = -100000
    assert torch.equal(masked_mean(hidden, mask), expected)
    assert torch.equal(masked_mean(hidden[:, :2], torch.ones(1, 2)), hidden[:, :2].mean(dim=1))
    return {"padding_invariant": True, "all_valid_equals_mean": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    args = parser.parse_args()
    shared_lock = json.loads((ROOT / "design-lock.json").read_text(encoding="utf-8"))
    assert shared_lock["updates"] == 300 and shared_lock["status"] == "go"
    for name, digest in shared_lock["inputs_sha256"].items():
        assert FT.file_sha256(ROOT / name) == digest, name
    directory = ROOT / "pooling-mean-seed13"
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    checks = {"pooling": pooling_selftest(), "metrics": selftest()}
    scorer = configure(args.repo_root, device)
    model, tokenizer = scorer.model, scorer.tokenizer
    dev = load_rows("dev")
    if args.mode == "evaluate":
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert result["script_sha256"] == FT.file_sha256(__file__)
        assert result["checkpoint_sha256"] == FT.file_sha256(directory / "selected.pt")
        model.load_state_dict(torch.load(directory / "selected.pt", map_location=device, weights_only=True), strict=True)
        dev_summary, dev_predictions = evaluate(model, tokenizer, dev, "a3", device)
        reference = json.loads((directory / "dev-predictions.json").read_text(encoding="utf-8"))
        difference = float(np.abs(np.asarray([row["probabilities"] for row in reference]) - np.asarray([row["probabilities"] for row in dev_predictions])).max())
        assert difference <= 1e-6 and [row["prediction"] for row in reference] == [row["prediction"] for row in dev_predictions]
        heldout = load_rows("heldout")
        summary, predictions = evaluate(model, tokenizer, heldout, "a3", device)
        dump(directory / "heldout.json", {"summary": summary, "predictions": predictions, "dev_reload_max_probability_difference": difference,
                                           "scope": "exploratory pooling contrast registered before parent inspected heldout outcomes; seed13 only"})
        print(json.dumps({"mode": "evaluate", "correct": summary["correct"], "n": len(heldout), "macro_f1": summary["macro_f1"], "reload_max": difference}), flush=True)
        return
    directory.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train = load_rows("train")
    labels = np.asarray([row["action_index"] for row in train])
    optimizer = FT.build_joint_optimizer(model, train_backbone=True)
    initial, _ = evaluate(model, tokenizer, dev, "a3", device)
    dump(directory / "config.json", {"pooling": "attention-mask mean including normal special tokens", "seed": 13, "updates": 300,
                                      "script_sha256": FT.file_sha256(__file__), "selftest": checks,
                                      "model_parameters": sum(p.numel() for p in model.parameters()), "initial_dev": initial,
                                      "registered_design_sha256": FT.file_sha256(ROOT / "补充设计-预训练读出一致性-2026-10-24.md")})
    observations = []

    def loss_fn(logits, chunk, window):
        gold = torch.as_tensor(labels[chunk], device=device)
        return torch.nn.functional.cross_entropy(logits.float(), gold, reduction="sum"), (logits.detach().argmax(dim=1) == gold).sum()

    def event(record):
        if record["event"] == "dev_evaluation":
            train_summary, _ = evaluate(model, tokenizer, train, "a3", device)
            dev_summary, _ = evaluate(model, tokenizer, dev, "a3", device)
            observations.append({"step": record["step"], "train": train_summary, "dev": dev_summary})
            print(json.dumps({"step": record["step"], "train_correct": train_summary["correct"], "dev_correct": dev_summary["correct"],
                              "dev_macro_f1": dev_summary["macro_f1"]}), flush=True)

    fitted = FT.fit_joint(model, tokenizer, [""] * len(train), optimizer=optimizer,
                          windows=FT.training_windows(len(train), epochs=1, batch_size=8, grad_accum=4, max_steps=300, rng=np.random.default_rng(13)),
                          planned_steps=300, planned_instances=9600, max_length=384, device=device, bf16=FT.resolve_bf16("on", device),
                          loss_fn=loss_fn, evaluate=lambda: evaluate(model, tokenizer, dev, "a3", device)[0],
                          checkpoint_selection="best-dev", selection_metric="dev-macro-f1", patience=0, min_delta=0, eval_every_steps=50,
                          require_backbone_grad=True, clip_grad_norm=1, event_callback=event,
                          encode_fn=lambda chunk: FT.encode_prepared_views(tokenizer, [train[int(i)]["encoding"] for i in chunk], device))
    del fitted["continuation"]
    torch.save(FT.cpu_state(model), directory / "selected.pt")
    train_summary, train_predictions = evaluate(model, tokenizer, train, "a3", device)
    dev_summary, dev_predictions = evaluate(model, tokenizer, dev, "a3", device)
    dump(directory / "train-predictions.json", train_predictions)
    dump(directory / "dev-predictions.json", dev_predictions)
    dump(directory / "result.json", {"script_sha256": FT.file_sha256(__file__), "fit": fitted, "selected_train": train_summary,
                                      "selected_dev": dev_summary, "observations": observations, "seconds_total": time.perf_counter() - started,
                                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "optimizer_saved": False,
                                      "checkpoint_sha256": FT.file_sha256(directory / "selected.pt"), "heldout_opened": False})
    print(json.dumps({"mode": "train", "completed": True, "dev_macro_f1": dev_summary["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
