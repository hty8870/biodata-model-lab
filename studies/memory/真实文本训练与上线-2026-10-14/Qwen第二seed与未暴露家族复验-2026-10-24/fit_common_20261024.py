"""Public primitives for the 2026-10-24 fixed-fit / base comparison study.

Everything here is research-only. The repo, the frozen data locks and the old runs
are read-only inputs; this module writes nothing outside the study directory.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ANCHOR = HERE.parent / "标签粒度与动作三类-2026-10-24"
REPO = Path(__file__).resolve().parents[4] / 'source'
MODELS = Path("<archive-root>/真实文本训练与上线-2026-10-14/固定数据拟合与基座对照-2026-10-24/models")
RUNS = HERE / "runs"
AUDITS = HERE / "audits"
SMOKES = HERE / "smokes"

for extra in (str(REPO), str(ANCHOR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import action_common_20261024 as AC              
from dataset_recommender.agent import semantic_relations as SR              
from research.memory_relations import finetune as FT              

ACTIONS = AC.ACTIONS
SEED = 42
MOTHER_SHA = "99e9b75f826533289bf7b43eb5336916c7929c6bf3f08b65750e431ebedcf26c"
MACBERT = {"repo_id": "hfl/chinese-macbert-base",
           "revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
           "dir": MODELS / "macbert"}
QWEN = {"repo_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "dir": MODELS / "qwen3"}
DIAGNOSTIC_DIR = HERE / "data-unexposed-v1"
CONFIRMATION_ROWS = ANCHOR / "confirmation-v2" / "rows.jsonl"


def read_json(path: "str | Path") -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: "str | Path") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: "str | Path", value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                 default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n",
                      encoding="utf-8")


def append_jsonl(path: "str | Path", value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def sha256_file(path: "str | Path") -> str:
    return FT.file_sha256(Path(path))


def state_sha256(state: dict) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        raw = tensor.view(torch.int16).numpy().tobytes() if tensor.dtype == torch.bfloat16 \
            else tensor.numpy().tobytes()
        hasher.update(raw)
    return hasher.hexdigest()


def environment() -> dict:
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(), "bf16_supported": torch.cuda.is_bf16_supported(),
            "platform": platform.platform()}


                                                                                        
               
                                                                                        

def locked_inputs() -> dict:
    """Read the anchor design lock and re-verify every SHA it pins."""
    lock = read_json(ANCHOR / "design-lock.json")
    assert lock["status"] == "go", "anchor design lock is not go"
    actual = {name: sha256_file(ANCHOR / name) for name in lock["inputs_sha256"]}
    for name, digest in lock["inputs_sha256"].items():
        assert actual[name] == digest, f"frozen input drift: {name}"
    mother = REPO / AC.MOTHER / "nli_model.pt"
    assert sha256_file(mother) == MOTHER_SHA, "mother weights SHA drift"
    return {"anchor_lock": lock, "actual_sha256": actual,
            "mother": {"path": str(mother), "sha256": MOTHER_SHA},
            "diagnostic": {name: sha256_file(DIAGNOSTIC_DIR / name)
                           for name in ("rows.jsonl", "inputs.jsonl", "manifest.json")},
            "confirmation": {name: sha256_file(CONFIRMATION_ROWS.parent / name)
                             for name in ("rows.jsonl", "lock.json")},
            "code": {"finetune.py": sha256_file(FT.__file__),
                     "action_common": sha256_file(ANCHOR / "action_common_20261024.py"),
                     "semantic_relations.py": sha256_file(SR.__file__),
                     "base": sha256_file(Path(__file__)),
                     "driver": sha256_file(HERE / "run_fit_bases_20261024.py")},
            "models": {name: {"dir": str(spec["dir"]), "revision": spec["revision"],
                              "files": {item.name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
                                        for item in sorted(spec["dir"].iterdir())
                                        if item.is_file()}}
                       for name, spec in (("macbert", MACBERT), ("qwen3", QWEN))}}


PROTOCOL = {
    "study": "Qwen second-seed and unexposed-family replication 2026-10-24",
    "seed": SEED,
    "source": "data-v1 720 train / 180 dev, inputs-v1 frozen MiniLM encodings, three actions",
    "diagnostic_replay": "data-unexposed-v1 120 (20 families / 40 triplets), confirmation-v2 300 exposed secondary; never train/select",
    "segments": [300, 600, 900, 1200, 1500],
    "fit_target": {"train_accuracy": 0.99, "train_ce": 0.10},
    "minilm": {"lr_backbone": 2e-5, "lr_head": 1e-3, "weight_decay": 0.01, "clip_norm": 1.0,
               "micro": 8, "accum": 4, "max_length": 384, "bf16": True},
    "minilm_head_lr_control": {"lr_backbone": 2e-5, "lr_head": 1e-4, "steps": 600},
    "macbert": {"repo_id": MACBERT["repo_id"], "revision": MACBERT["revision"], "pool": "cls",
                "lr_backbone": 2e-5, "lr_head": 1e-3, "weight_decay": 0.01, "clip_norm": 1.0,
                "micro": 8, "accum": 4, "max_length": 512, "bf16": True, "updates": 300},
    "qwen3": {"repo_id": QWEN["repo_id"], "revision": QWEN["revision"], "pool": "last_nonpad",
              "base_dtype": "bfloat16", "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
              "lr": 1e-4, "weight_decay": 0.01, "clip_norm": 1.0, "micro": 2, "accum": 16,
              "max_length": 512, "gradient_checkpointing": True, "bf16": True, "updates": 300},
    "evaluation": "terminal weights for fit; dev-best (macro-F1, ties earliest) frozen for generalization replay",
    "no_grid": True,
}


def make_or_check_lock() -> dict:
    path = HERE / "run-lock.json"
    value = {"protocol": PROTOCOL, "inputs": locked_inputs(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if path.exists():
        previous = read_json(path)
        assert previous["protocol"] == value["protocol"], "protocol drift versus run-lock.json"
        for key in ("actual_sha256", "mother", "diagnostic", "confirmation", "models"):
            assert previous["inputs"][key] == value["inputs"][key], f"input drift: {key}"
        for key in ("finetune.py", "action_common", "semantic_relations.py"):
            assert previous["inputs"]["code"][key] == value["inputs"]["code"][key], \
                f"frozen code drift: {key}"
        return previous
    dump(path, value)
    return value


                                                                                        
      
                                                                                        

def load_rows(split: str) -> list[dict]:
    """Frozen MiniLM inputs (encoding + visible_text) merged with labels/uncovered counts."""
    rows = AC.load_rows(split)
    assert len(rows) == {"train": 720, "dev": 180, "heldout": 300}[split]
    for row in rows:
        assert row["complete"] is True and row["visible_text"] and row["encoding"]["input_ids"]
    return rows


def diagnostic_rows() -> list[dict]:
    """120-row minimal-change diagnostic: rows + stored visible_text/MiniLM encodings."""
    rows = read_jsonl(DIAGNOSTIC_DIR / "rows.jsonl")
    inputs = {row["id"]: row for row in read_jsonl(DIAGNOSTIC_DIR / "inputs.jsonl")}
    assert len(rows) == 120 and len(inputs) == 120
    for row in rows:
        item = inputs[row["id"]]
        assert item["visible_text"] and item["encoding"]["input_ids"]
        row["visible_text"] = item["visible_text"]
        row["encoding"] = item["encoding"]
        row["uncovered_old_count"] = int(row.get("uncovered_old_count",
                                                 len(set(row["formal"]["old"]) - set(row["formal"]["new"]))))
    assert len({row["cluster_id"] for row in rows}) == 40
    assert all(sum(1 for other in rows if other["cluster_id"] == group) == 3
               for group in {row["cluster_id"] for row in rows})
    return rows


def confirmation_rows(mother_tokenizer) -> list[dict]:
    """300 already-exposed confirmation rows: canonical visible_text rendered with the mother tokenizer."""
    rows = read_jsonl(CONFIRMATION_ROWS)
    assert len(rows) == 300
    for row in rows:
        prepared = SR.prepare_semantic_input(row["pair"], mother_tokenizer, max_length=384,
                                             role_independent_frame=True, limits=SR.SEMANTIC_LIMITS_RESEARCH)
        forward = prepared["directions"]["forward"]
        assert forward["complete"], f"confirmation row needs budget adaptation: {row['id']}"
        row["visible_text"] = forward["text"]
        row["view_sha256"] = forward["view_sha256"]
        row["uncovered_old_count"] = len(set(row["formal"]["old"]) - set(row["formal"]["new"]))
        row.setdefault("preserve_count", len(row["formal"].get("preserve_keys", [])))
        row.setdefault("conflict_count", len(row["formal"].get("conflict_keys", [])))
    return rows


def renderer_reproduction_check(mother_tokenizer) -> dict:
    """Prove the canonical renderer reproduces data-v3 stored texts before trusting it for confirmation."""
    rows = read_jsonl(DIAGNOSTIC_DIR / "rows.jsonl")
    stored = {row["id"]: row for row in read_jsonl(DIAGNOSTIC_DIR / "inputs.jsonl")}
    mismatched, adapted, encoding_mismatch = [], [], []
    for row in rows:
        prepared = SR.prepare_semantic_input(row["pair"], mother_tokenizer, max_length=384,
                                             role_independent_frame=True, limits=SR.SEMANTIC_LIMITS_RESEARCH)
        forward = prepared["directions"]["forward"]
        if not forward["complete"]:
            adapted.append(row["id"])
        if forward["text"] != stored[row["id"]]["visible_text"]:
            mismatched.append(row["id"])
        ids = mother_tokenizer(forward["text"], add_special_tokens=True)["input_ids"]
        if ids != stored[row["id"]]["encoding"]["input_ids"]:
            encoding_mismatch.append(row["id"])
    assert not mismatched and not adapted and not encoding_mismatch, \
        f"renderer reproduction failed: {mismatched[:3]} {adapted[:3]} {encoding_mismatch[:3]}"
    return {"n": len(rows), "visible_text_identical": True, "encoding_identical": True,
            "budget_adapted": 0}


                                                                                        
                       
                                                                                        

def token_lengths(tokenizer, texts: Sequence[str]) -> dict:
    lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
    values = np.asarray(lengths)
    return {"n": len(lengths), "special_tokens_included": True, "sum": int(values.sum()),
            "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "max": int(values.max())}


def assert_fits(tokenizer, texts: Sequence[str], max_length: int, label: str) -> dict:
    profile = token_lengths(tokenizer, texts)
    if profile["max"] > max_length:
        raise ValueError(f"{label}: max tokens {profile['max']} > {max_length}; refusing to truncate conditions")
    profile["max_length"] = max_length
    profile["complete"] = True
    return profile


def encode_batch_visible(tokenizer, texts: Sequence[str], device: torch.device, max_length: int) -> dict:
    encoded = tokenizer(list(texts), padding="longest", truncation=False, return_tensors="pt")
    width = int(encoded["input_ids"].shape[1])
    if width > max_length:
        raise ValueError(f"encoded width {width} exceeds {max_length}: refuse to truncate")
    keys = set(tokenizer.model_input_names)
    return {key: value.to(device) for key, value in encoded.items() if key in keys}


class PooledClassifier(nn.Module):
    """Backbone + dropout + 3-way linear head; the only study-specific read-out parameter is the pool."""

    def __init__(self, backbone: nn.Module, hidden_size: int, *, pool: str, seed: int,
                 num_labels: int = 3, dropout: float = FT.DROPOUT,
                 extra_forward: "dict | None" = None) -> None:
        super().__init__()
        if pool not in ("cls", "last_nonpad"):
            raise ValueError(pool)
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(int(hidden_size), int(num_labels))
        self.pool = pool
        self.extra_forward = dict(extra_forward or {})
        self.accepts_token_type_ids = "token_type_ids" in set(inspect.signature(
            self._forward_target().forward).parameters)
        torch.manual_seed(int(seed))
        nn.init.normal_(self.classifier.weight, mean=0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _forward_target(self) -> nn.Module:
        model = self.backbone
        return getattr(model, "base_model", model)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **extra):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, **self.extra_forward}
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        hidden = outputs.last_hidden_state
        if self.pool == "cls":
            pooled = hidden[:, 0]
        else:
            index = attention_mask.to(torch.long).sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
        return self.classifier(self.dropout(pooled))


_MOTHER_TOKENIZER = None


def mother_tokenizer():
    """Tokenizer of the frozen mother artifact, loaded without pulling the 470 MB weights."""
    global _MOTHER_TOKENIZER
    if _MOTHER_TOKENIZER is None:
        from transformers import AutoTokenizer
        _MOTHER_TOKENIZER = AutoTokenizer.from_pretrained(str(REPO / AC.MOTHER / "tokenizer"),
                                                          local_files_only=True)
    return _MOTHER_TOKENIZER


def build_minilm(seed: int = SEED):
    scorer = AC.initialize(REPO, "a3", seed)
    assert isinstance(scorer.model.classifier, nn.Linear) and scorer.model.classifier.out_features == 3
    return scorer.model, scorer.tokenizer


def build_macbert(seed: int = SEED):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MACBERT["dir"]), local_files_only=True)
    backbone = AutoModel.from_pretrained(str(MACBERT["dir"]), local_files_only=True,
                                         add_pooling_layer=False)
    backbone.config.use_cache = False
    model = PooledClassifier(backbone, backbone.config.hidden_size, pool="cls", seed=seed)
    return model, tokenizer


def build_qwen(seed: int = SEED, *, gradient_checkpointing: bool = True):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN["dir"]), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    backbone = AutoModel.from_pretrained(str(QWEN["dir"]), local_files_only=True,
                                         dtype=torch.bfloat16)
    backbone.config.use_cache = False
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                        task_type="FEATURE_EXTRACTION",
                        target_modules=list(PROTOCOL["qwen3"]["target_modules"]))
    torch.manual_seed(int(seed))
    peft_model = get_peft_model(backbone, config)
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
    model = PooledClassifier(peft_model, backbone.config.hidden_size, pool="last_nonpad", seed=seed,
                             extra_forward={"use_cache": False})
    return model, tokenizer


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if "lora_" in name]


def adapter_dtypes(model: nn.Module) -> dict:
    return {str(dtype): sum(1 for name, param in model.named_parameters() if "lora_" in name
                            and param.dtype == dtype)
            for dtype in {param.dtype for name, param in model.named_parameters() if "lora_" in name}}


def build_optimizer(kind: str, model: nn.Module) -> torch.optim.Optimizer:
    if kind in ("minilm", "minilm-head-lr-1e-4", "macbert"):
        settings = PROTOCOL["minilm"] if kind.startswith("minilm") else PROTOCOL["macbert"]
        return FT.build_joint_optimizer(model, train_backbone=True,
                                        lr_backbone=settings["lr_backbone"],
                                        lr_head=effective_settings(kind)["lr_head"],
                                        weight_decay=settings["weight_decay"])
    if kind == "qwen3":
        settings = PROTOCOL["qwen3"]
        return FT.build_joint_optimizer(model, train_backbone=True, lr_backbone=settings["lr"],
                                        lr_head=settings["lr"], weight_decay=settings["weight_decay"])
    raise ValueError(kind)


def make_encode_fn(kind: str, tokenizer, rows: Sequence[dict], device: torch.device, max_length: int):
    if kind.startswith("minilm"):
        return lambda chunk: FT.encode_prepared_views(tokenizer, [rows[int(i)]["encoding"] for i in chunk], device)
    return lambda chunk: encode_batch_visible(tokenizer, [rows[int(i)]["visible_text"] for i in chunk],
                                              device, max_length)


def make_loss(targets, device: torch.device):
    targets = np.asarray(targets, dtype=np.int64)

    def loss_fn(logits, chunk, window):
        selected = torch.as_tensor(targets[chunk], device=device)
        loss = F.cross_entropy(logits.float(), selected, reduction="sum")
        hits = (logits.detach().float().argmax(dim=1) == selected).sum()
        return loss, hits

    return loss_fn


                                                                                        
            
                                                                                        

def build_predictions(probabilities, rows: Sequence[dict]) -> list[dict]:
    predictions = []
    for row, values in zip(rows, probabilities):
        values = [float(value) for value in values]
        predictions.append({
            "id": row["id"], "cluster_id": row["cluster_id"], "domain": row["domain"],
            "gold": ACTIONS[int(row["action_index"])], "prediction": ACTIONS[int(np.argmax(values))],
            "probabilities": values, "fine_probabilities": values,
            "preserve_count": int(row.get("preserve_count", 0)),
            "conflict_count": int(row.get("conflict_count", 0))})
    return predictions


def evaluate_visible(model, tokenizer, rows: Sequence[dict], device: torch.device, max_length: int,
                     batch_size: int = 16, bf16: bool = True) -> tuple[dict, list[dict]]:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = encode_batch_visible(tokenizer, [row["visible_text"] for row in rows[start:start + batch_size]],
                                         device, max_length)
            with FT.autocast_for(device, bf16):
                logits = model(**batch)
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise RuntimeError(f"logits shape {tuple(logits.shape)} is not (n, 3)")
            probabilities.extend(logits.float().softmax(dim=-1).cpu().tolist())
    summary = AC.summarize(probabilities, rows)
    return summary, build_predictions(probabilities, rows)


def evaluate_minilm(model, tokenizer, rows: Sequence[dict], device: torch.device) -> tuple[dict, list[dict]]:
    return AC.evaluate(model, tokenizer, rows, "a3", device)


def evaluator(kind: str, model, tokenizer, device: torch.device, max_length: int):
    if kind.startswith("minilm"):
        return lambda rows: evaluate_minilm(model, tokenizer, rows, device)
    return lambda rows: evaluate_visible(model, tokenizer, rows, device, max_length)


                                                                                        
        
                                                                                        

def duplicate_label_audit(mother_tokenizer) -> dict:
    """Same input must never carry two labels; identical text must not cross splits."""
    splits = {split: load_rows(split) for split in ("train", "dev", "heldout")}
    diagnostics = diagnostic_rows()
    confirmations = confirmation_rows(mother_tokenizer)
    groups = dict(splits, diagnostic=diagnostics, confirmation=confirmations)

    text_labels: dict[str, set[int]] = {}
    text_splits: dict[str, set[str]] = {}
    encoding_labels: dict[str, set[int]] = {}
    for split, rows in groups.items():
        for row in rows:
            text_labels.setdefault(row["visible_text"], set()).add(int(row["action_index"]))
            text_splits.setdefault(row["visible_text"], set()).add(split)
            if "encoding" in row:
                key = hashlib.sha256(json.dumps(row["encoding"], sort_keys=True).encode()).hexdigest()
                encoding_labels.setdefault(key, set()).add(int(row["action_index"]))
    conflicting_texts = {key: sorted(values) for key, values in text_labels.items() if len(values) > 1}
    conflicting_encodings = {key: sorted(values) for key, values in encoding_labels.items() if len(values) > 1}
    cross_split_texts = {key: sorted(values) for key, values in text_splits.items() if len(values) > 1}
    return {"groups": {name: len(rows) for name, rows in groups.items()},
            "identical_text_conflicting_labels": len(conflicting_texts),
            "identical_encoding_conflicting_labels": len(conflicting_encodings),
            "identical_text_across_splits": len(cross_split_texts),
            "examples": {"text_conflict": list(conflicting_texts.items())[:2],
                         "encoding_conflict": list(conflicting_encodings.items())[:2],
                         "cross_split": list(cross_split_texts.items())[:2]}}


def parameter_audit(model: nn.Module, optimizer: torch.optim.Optimizer, label: str) -> dict:
    group_of: dict[int, int] = {}
    for index, group in enumerate(optimizer.param_groups):
        for param in group["params"]:
            assert id(param) not in group_of, "a parameter appears in two optimizer groups"
            group_of[id(param)] = index
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    assert trainable == set(group_of), "optimizer parameter set differs from requires_grad set"
    tensors = []
    for name, param in model.named_parameters():
        tensors.append({"name": name, "numel": param.numel(), "dtype": str(param.dtype),
                        "requires_grad": bool(param.requires_grad),
                        "optimizer_group": group_of.get(id(param)),
                        "is_lora": "lora_" in name,
                        "is_head": name.startswith("classifier.")})
    groups = [{"index": index, "tensors": len(group["params"]),
               "numel": int(sum(param.numel() for param in group["params"])),
               "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
               "eps": float(group["eps"])} for index, group in enumerate(optimizer.param_groups)]
    summary = {
        "total_parameters": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameters": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "trainable_tensors": len(trainable),
        "frozen_tensors": sum(1 for param in model.parameters() if not param.requires_grad),
        "dtypes": dict(Counter(str(param.dtype) for param in model.parameters())),
        "groups": groups,
    }
    if any("lora_" in name for name, _ in model.named_parameters()):
        lora = [(name, param) for name, param in model.named_parameters() if "lora_" in name]
        summary["lora"] = {"tensors": len(lora), "numel": int(sum(param.numel() for _, param in lora)),
                           "dtypes": adapter_dtypes(model),
                           "base_trainable": int(sum(param.numel() for name, param in model.named_parameters()
                                                     if "lora_" not in name and not name.startswith("classifier."))),
                           "base_frozen": all(not param.requires_grad for name, param in model.named_parameters()
                                              if "lora_" not in name and not name.startswith("classifier."))}
    dump(AUDITS / f"parameter-audit-{label}.json", {"summary": summary, "tensors": tensors})
    return summary


def gradient_and_update_probe(*, model, optimizer, encode_fn, loss_fn, chunk, device, bf16, frozen: bool,
                              label: str, unused_prefixes: "tuple[str, ...]" = ("backbone.pooler.",)) -> dict:
    """One real CUDA step: prove non-zero gradients AND that parameters actually move, then restore."""
    initial = FT.cpu_state(model)
    initial_sha = state_sha256(initial)
    model.train()
    if frozen:
        model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    batch = encode_fn(chunk)
    with FT.autocast_for(device, bf16):
        logits = model(**batch)
        loss, _ = loss_fn(logits, chunk, chunk)
    if not torch.isfinite(loss):
        raise ValueError("probe loss is not finite")
    (loss / len(chunk)).backward()
    gradients, stats = {}, []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        entry = {"name": name, "grad_none": grad is None,
                 "grad_nonzero_tensors": None, "grad_abs_max": None, "grad_abs_mean": None}
        if grad is not None:
            gradients[name] = grad
            entry["grad_nonzero_tensors"] = int((grad != 0).sum().item())
            entry["grad_abs_max"] = float(grad.abs().max().item())
            entry["grad_abs_mean"] = float(grad.abs().mean().item())
        stats.append(entry)
    optimizer.step()
    deltas = []
    for name, param in model.named_parameters():
        before = initial[name].to(param.dtype)
        delta = (param.detach().cpu() - before.float()).abs()
        deltas.append({"name": name, "requires_grad": bool(param.requires_grad),
                       "changed_tensors": int((delta != 0).sum().item()),
                       "delta_abs_max": float(delta.max().item()),
                       "delta_abs_mean": float(delta.mean().item())})
    frozen_with_grad = [name for name, param in model.named_parameters()
                        if not param.requires_grad and param.grad is not None]
    model.load_state_dict(initial, strict=True)
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)

    by_name = {item["name"]: item for item in stats}
    delta_by_name = {item["name"]: item for item in deltas}
    head_grad = [item for item in stats if item["name"].startswith("classifier.")]
    backbone_grad = [item for item in stats if not item["name"].startswith("classifier.")]
    head_changed = sum(1 for item in deltas if item["name"].startswith("classifier.") and item["changed_tensors"] > 0)
    backbone_changed = sum(1 for item in deltas if not item["name"].startswith("classifier.")
                           and item["changed_tensors"] > 0)
    nonzero_head = sum(1 for item in head_grad if item["grad_nonzero_tensors"])
    nonzero_backbone = sum(1 for item in backbone_grad if item["grad_nonzero_tensors"])
    all_trainable = {item["name"] for item in stats}
    zero_grad_missing_update = [name for name in all_trainable
                                if by_name[name]["grad_nonzero_tensors"] and delta_by_name[name]["changed_tensors"] == 0]
    report = {
        "label": label, "device": str(device), "bf16_autocast": bf16, "probe_batch": len(chunk),
        "initial_state_sha256": initial_sha, "loss": float(loss.detach()),
        "grads_present_tensors": len(gradients), "trainable_tensors": len(all_trainable),
        "nonzero_grad_tensors": sum(1 for item in stats if item["grad_nonzero_tensors"]),
        "grad_none_tensors": [item["name"] for item in stats if item["grad_none"]],
        "unused_trainable_tensors": [item["name"] for item in stats if item["grad_none"]
                                      and item["name"].startswith(unused_prefixes)],
        "unexpected_grad_none_tensors": [item["name"] for item in stats if item["grad_none"]
                                          and not item["name"].startswith(unused_prefixes)],
        "nonzero_head_tensors": nonzero_head, "nonzero_backbone_tensors": nonzero_backbone,
        "head_tensors": len(head_grad), "backbone_trainable_tensors": len(backbone_grad),
        "changed_tensors": sum(1 for item in deltas if item["changed_tensors"] > 0),
        "head_changed_tensors": head_changed, "backbone_changed_tensors": backbone_changed,
        "frozen_params_with_grad": frozen_with_grad,
        "nonzero_grad_without_update": zero_grad_missing_update,
        "max_abs_delta": max(item["delta_abs_max"] for item in deltas),
        "stats": stats, "deltas": deltas,
    }
    assert not report["unexpected_grad_none_tensors"],         f"trainable tensors without grad outside the known-unused pooler: {report['unexpected_grad_none_tensors'][:5]}"
    assert nonzero_head and head_changed, "head did not receive a non-zero gradient or did not move"
    assert not zero_grad_missing_update, f"non-zero grad but no update: {zero_grad_missing_update[:5]}"
    if label.startswith("qwen3"):
        assert report["frozen_params_with_grad"] == [], "frozen base received gradients"
        assert nonzero_backbone and backbone_changed, "LoRA adapters did not receive gradients/update"
    else:
        assert nonzero_backbone and backbone_changed, "encoder did not receive gradients or did not move"
    return report


def loss_target_selftest(loss_fn, targets, rows, chunk, device) -> dict:
    """Perfect vs deliberately wrong logits, sum/mean identity, and real-chunk target mapping."""
    selected = np.asarray(targets)[chunk]
    n = len(chunk)
    gold = torch.as_tensor(selected, device=device)
    perfect = torch.full((n, 3), -10.0, device=device)
    perfect[torch.arange(n, device=device), gold] = 10.0
    wrong_index = torch.as_tensor((selected + 1) % 3, device=device)
    wrong = torch.full((n, 3), -10.0, device=device)
    wrong[torch.arange(n, device=device), wrong_index] = 10.0
    perfect_logits = perfect.clone().requires_grad_(True)
    wrong_logits = wrong.clone().requires_grad_(True)
    perfect_loss, perfect_hits = loss_fn(perfect_logits, chunk, chunk)
    wrong_loss, wrong_hits = loss_fn(wrong_logits, chunk, chunk)
    mean_loss = F.cross_entropy(perfect_logits.float(), gold, reduction="mean")
    assert torch.allclose(perfect_loss, mean_loss * n, atol=1e-4), "loss_fn is not a window sum"
    probabilities = perfect.softmax(dim=-1).detach().cpu().numpy()
    subset = [rows[int(i)] for i in chunk]
    summary = AC.summarize(probabilities, subset)
    assert summary["correct"] == n and summary["accuracy"] == 1.0
    assert summary["loss"] < 1e-6, summary["loss"]
    wrong_probabilities = wrong.softmax(dim=-1).detach().cpu().numpy()
    wrong_summary = AC.summarize(wrong_probabilities, subset)
    assert wrong_summary["correct"] == 0 and wrong_summary["loss"] > 1.0
    perfect_grad = torch.autograd.grad(perfect_loss, perfect_logits)[0]
    wrong_grad = torch.autograd.grad(wrong_loss, wrong_logits)[0]
    assert torch.isfinite(perfect_grad).all() and perfect_grad.abs().sum() > 0
    assert torch.isfinite(wrong_grad).all() and wrong_grad.abs().sum() > 0
    return {"n": n, "perfect_ce": float(perfect_loss.detach()), "perfect_hits": float(perfect_hits),
            "wrong_ce": float(wrong_loss.detach()), "wrong_hits": float(wrong_hits),
            "perfect_accuracy": summary["accuracy"], "wrong_accuracy": wrong_summary["accuracy"],
            "sum_equals_mean_times_n": True, "loss_finite_gradient": True,
            "gold_labels_in_chunk": [int(v) for v in selected[:8]],
            "gold_actions_in_chunk": [ACTIONS[int(v)] for v in selected[:8]]}


                                                                                        
          
                                                                                        

def train_segments(*, kind: str, model, tokenizer, rows, targets, seed: int, segments: Sequence[int],
                   micro: int, accum: int, max_length: int, device: torch.device, bf16: bool,
                   optimizer: torch.optim.Optimizer, encode_fn, eval_dev, eval_train, output: Path,
                   frozen: bool, require_backbone_grad: bool, clip_norm: float,
                   eval_every: int = 50, fit_target: "dict | None" = None) -> dict:
    """Continue the frozen-update budget in segments, keeping optimizer state in memory only."""
    assert segments == sorted(segments) and segments[-1] == max(segments)
    output.mkdir(parents=True, exist_ok=True)
    curve_path, updates_path = output / "curve.jsonl", output / "updates.jsonl"
    for path in (curve_path, updates_path):
        path.unlink(missing_ok=True)
    targets_array = np.asarray(targets, dtype=np.int64)
    loss_fn = make_loss(targets_array, device)
    state: "dict | None" = None
    started = time.perf_counter()
    segment_records, reached_at, stopped = [], None, False
    for target in segments:
        windows = FT.training_windows(len(rows), epochs=1, batch_size=micro, grad_accum=accum,
                                      max_steps=target, rng=np.random.default_rng(seed))
        segment_started = time.perf_counter()

        def on_event(event: dict) -> None:
            record = dict(event, seconds=round(time.perf_counter() - started, 3))
            if event["event"] == "update":
                append_jsonl(updates_path, record)
                return
            if event["event"] == "dev_evaluation":
                train_summary = eval_train()
                dev_summary = eval_dev()
                entry = {"step": event["step"], "segment_target": target, "seconds": round(time.perf_counter() - started, 3),
                         "interval_loss": event["loss"], "interval_accuracy": event["accuracy"],
                         "train": train_summary, "dev": dev_summary,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                append_jsonl(curve_path, entry)
                segment_records.append(entry)
                print(json.dumps({"kind": kind, "step": event["step"], "segment_target": target,
                                  "train_accuracy": train_summary["accuracy"], "train_ce": train_summary["loss"],
                                  "dev_macro_f1": dev_summary["macro_f1"],
                                  "seconds": entry["seconds"]}), flush=True)

        fitted = FT.fit_joint(model, tokenizer, [""] * len(rows), optimizer=optimizer, windows=windows,
                              planned_steps=target, planned_instances=target * micro * accum,
                              max_length=max_length, device=device, bf16=bf16, loss_fn=loss_fn,
                              evaluate=eval_dev, checkpoint_selection="final", selection_metric="dev-macro-f1",
                              patience=0, min_delta=0.0, eval_every_steps=eval_every, frozen=frozen,
                              require_backbone_grad=require_backbone_grad, resume_state=state,
                              event_callback=on_event, clip_grad_norm=clip_norm, encode_fn=encode_fn)
        state = fitted["continuation"]
        last = segment_records[-1]
        assert last["step"] == target, "segment did not reach its planned update count"
        segment_records[-1]["segment_seconds"] = round(time.perf_counter() - segment_started, 3)
        reached = (fit_target is not None and float(last["train"]["accuracy"]) >= fit_target["train_accuracy"]
                   and float(last["train"]["loss"]) <= fit_target["train_ce"])
        segment_records[-1]["fit_target_reached_here"] = bool(reached)
        if reached:
            reached_at, stopped = target, True
            break
    fitted_done = {"kind": kind, "segments_run": [record["step"] for record in segment_records],
                   "stopped_at_target": stopped, "reached_at_update": reached_at,
                   "seconds_total": round(time.perf_counter() - started, 3),
                   "final": {key: value for key, value in fitted.items() if key != "continuation"},
                   "selection": fitted["selection"], "budget": fitted["budget"],
                   "curve": segment_records, "encoder_targets": None}
    dump(output / "fit.json", fitted_done)
    return {"fitted": fitted, "state": state, "records": segment_records,
            "reached_at": reached_at, "stopped": stopped, "seconds": time.perf_counter() - started}


def effective_settings(kind: str) -> dict:
    if kind == "minilm":
        settings = dict(PROTOCOL["minilm"])
        settings["updates"] = PROTOCOL["segments"]
    elif kind == "minilm-head-lr-1e-4":
        settings = dict(PROTOCOL["minilm"])
        settings.update(lr_head=PROTOCOL["minilm_head_lr_control"]["lr_head"],
                        updates=PROTOCOL["minilm_head_lr_control"]["steps"])
    elif kind == "macbert":
        settings = dict(PROTOCOL["macbert"])
    elif kind == "qwen3":
        settings = dict(PROTOCOL["qwen3"])
    else:
        raise ValueError(kind)
    return settings


def build(kind: str, seed: int = SEED):
    if kind.startswith("minilm"):
        return build_minilm(seed)
    if kind == "macbert":
        return build_macbert(seed)
    if kind == "qwen3":
        return build_qwen(seed)
    raise ValueError(kind)


def checkpoint_meta(kind: str, step: "int | None", selection: str, extra: "dict | None" = None) -> dict:
    meta = {"kind": kind, "step": step, "selection": selection, "seed": SEED,
            "targets": "action_index (coexist/replace_all/merge_preserve)", "protocol": PROTOCOL["study"]}
    if extra:
        meta.update(extra)
    return meta


def save_full_checkpoint(path: Path, state: dict, meta: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "meta": meta}, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_qwen_checkpoint(directory: Path, model, head_state: dict, meta: dict) -> dict:
    """Adapter + head only: never copy the 0.6B base."""
    directory.mkdir(parents=True, exist_ok=True)
    adapter = directory / "adapter"
    adapter.mkdir(exist_ok=True)
    try:
        model.backbone.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        used_embedding_flag = True
    except TypeError:
        model.backbone.save_pretrained(adapter, safe_serialization=True)
        used_embedding_flag = False
    torch.save({"classifier": head_state, "meta": meta}, directory / "head.pt")
    files = {item.relative_to(directory).as_posix(): {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
             for item in sorted(directory.rglob("*")) if item.is_file()}
    weight_files = [name for name in files if name != "head.pt" and not name.endswith(".json")]
    assert weight_files, "adapter weights were not written"
    total = sum(item["bytes"] for item in files.values())
    base_bytes = sum(item.stat().st_size for item in QWEN["dir"].glob("*.safetensors"))
    assert total < base_bytes / 10, "saved Qwen checkpoint is suspiciously close to base size"
    return {"directory": str(directory), "files": files, "bytes": total,
            "save_embedding_layers_false": used_embedding_flag,
            "fraction_of_base_weight_bytes": total / base_bytes}
