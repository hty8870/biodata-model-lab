from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_common_20261024 as C              
from research.memory_relations import finetune as FT              

OUT = HERE / "probes"


def describe(name: str, model) -> dict:
    base = model.backbone
    inner = getattr(base, "base_model", None)
    inner_name = None
    if inner is not None and hasattr(inner, "model"):
        inner_name = type(inner.model).__name__
    dtypes = {}
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
    return {"name": name, "wrapper": type(model).__name__, "backbone": type(base).__name__,
            "inner": inner_name,
            "hidden_size": int(model.classifier.in_features),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "dtype_numel": dtypes, "pool": model.pool,
            "accepts_token_type_ids": bool(model.accepts_token_type_ids)}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    report = {"environment": C.environment(), "bases": {}}
    mother_tokenizer = C.build_minilm()[1]
    rows = {split: C.load_rows(split) for split in ("train", "dev")}
    diagnostics = C.diagnostic_rows()
    confirmations = C.confirmation_rows(mother_tokenizer)
    report["renderer_reproduction"] = C.renderer_reproduction_check(mother_tokenizer)
    texts = {name: [row["visible_text"] for row in group] for name, group in
             dict(rows, diagnostic=diagnostics, confirmation=confirmations).items()}
    report["token_profiles"] = {
        "minilm_stored": {split: {"n": len(group), "sum": int(sum(len(r["encoding"]["input_ids"]) for r in group)),
                                  "max": max(len(r["encoding"]["input_ids"]) for r in group)}
                          for split, group in rows.items()},
    }

                                                                                      
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = C.build_macbert()
    info = describe("macbert", model)
    info["tokenizer"] = {"class": type(tokenizer).__name__, "model_input_names": list(tokenizer.model_input_names),
                         "pad_token_id": tokenizer.pad_token_id, "vocab_size": tokenizer.vocab_size}
    info["profiles"] = {name: C.assert_fits(tokenizer, group, 512, f"macbert/{name}") for name, group in texts.items()}
    model.to(device)
    optimizer = C.build_optimizer("macbert", model)
    train_rows = rows["train"]
    targets = [row["action_index"] for row in train_rows]
    encode_fn = C.make_encode_fn("macbert", tokenizer, train_rows, device, 512)
    loss_fn = C.make_loss(targets, device)
    window = next(FT.training_windows(len(train_rows), epochs=1, batch_size=8, grad_accum=4, max_steps=1,
                                      rng=np.random.default_rng(13)))
    chunk = window[2][0]
    smoke = C.gradient_and_update_probe(model=model, optimizer=optimizer, encode_fn=encode_fn, loss_fn=loss_fn,
                                        chunk=chunk, device=device, bf16=True, frozen=False, label="macbert-smoke")
    smoke.pop("stats")
    smoke.pop("deltas")
    info["smoke"] = smoke
    info["smoke"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    info["smoke"]["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    info["smoke"]["seconds"] = round(time.perf_counter() - started, 2)
    report["bases"]["macbert"] = info
    del model, optimizer
    torch.cuda.empty_cache()

                                                                                       
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = C.build_qwen()
    info = describe("qwen3", model)
    info["tokenizer"] = {"class": type(tokenizer).__name__, "model_input_names": list(tokenizer.model_input_names),
                         "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
                         "padding_side": tokenizer.padding_side, "vocab_size": tokenizer.vocab_size}
    info["lora"] = {"tensors": len(C.lora_parameter_names(model)), "dtypes": C.adapter_dtypes(model),
                    "names_sample": C.lora_parameter_names(model)[:4]}
    info["profiles"] = {name: C.assert_fits(tokenizer, group, 512, f"qwen3/{name}") for name, group in texts.items()}
    model.to(device)
    optimizer = C.build_optimizer("qwen3", model)
    encode_fn = C.make_encode_fn("qwen3", tokenizer, train_rows, device, 512)
    chunk = window[2][0]
    smoke = C.gradient_and_update_probe(model=model, optimizer=optimizer, encode_fn=encode_fn, loss_fn=loss_fn,
                                        chunk=chunk, device=device, bf16=True, frozen=True, label="qwen3-smoke")
    smoke.pop("stats")
    smoke.pop("deltas")
    info["smoke"] = smoke
    info["smoke"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    info["smoke"]["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    info["smoke"]["seconds"] = round(time.perf_counter() - started, 2)
    adapter_dir = OUT / "qwen-lora-probe"
    info["save_probe"] = C.save_qwen_checkpoint(adapter_dir, model, model.classifier.state_dict(),
                                               {"probe": True})
    info["save_probe"].pop("files", None)
    report["bases"]["qwen3"] = info
    C.dump(OUT / "base-probe.json", report)
    print(json.dumps({"ok": True, "macbert": report["bases"]["macbert"]["smoke"],
                      "qwen3": report["bases"]["qwen3"]["smoke"],
                      "token_max": {name: {"macbert": report["bases"]["macbert"]["profiles"][name]["max"],
                                           "qwen3": report["bases"]["qwen3"]["profiles"][name]["max"]}
                                    for name in texts}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
