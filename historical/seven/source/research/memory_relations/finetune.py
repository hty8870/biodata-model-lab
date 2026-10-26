"""记忆关系联合输入 cross-encoder 微调实验（研究-only；v1 首轮之后的第二臂）。

两臂同输入模板（角色分隔 joint text）+ 同 7 类 head：
  frozen-joint    冻结 12 层 MiniLM 骨干（BertModel, hidden 384），只训 head（head_only 权重）；
  finetuned-joint 端到端微调（backbone lr 2e-5 / head lr 1e-3，AdamW，wd 0.01；full 权重）。
默认仍取 final 权重；显式 best-dev 仅按独立dev loss/macro-F1选CPU快照并恢复。
固定update模式默认patience=0，每步跨epoch填满实例预算；partitioned candidate只见prob/policy，
顶层阈值null且audit deferred；唯一dev候选经train --mode lock/audit独立一次审计。
--eval-splits none 不打开独立终测文件，不用终测作parity；参数、预算与数据指纹fit前落盘。
请求 GPU 不可用或显存分配失败如实报错，绝不静默回退 CPU 冒充 GPU。
输入模板只读真实投影的 text/trigger/candidate evidence 引用/有序 evidence 出处文本
（无 tier/permanence/id/label/scope）；截断按 joint 序列真实字符边界 + tokenizer
offset_mapping（slow tokenizer 回退前缀 token 边界）逐 role 归属，同一 role 多 block
每行至多计一次（counting_unit=row），无法精确归属时明确标注，不假称输入完整。
语义边界：v1 学生臂是 SentenceTransformer 句向量口径（官方 mean pooling、其
sentence_bert_config max_seq_length=128），v2 两臂是 384 + CLS 池化 joint 输入——
只比较 v2 内两臂解释冻结效应，v1→v2 的差异不能唯一归因于 finetune。

Usage:
  python -m research.memory_relations.finetune --data pairs.jsonl --challenge challenge.jsonl \\
      --out runs/v2-train --models frozen-joint,finetuned-joint --sizes 2000 \\
      --seeds 13,20260919,424242 --epochs 3 --device cuda --batch-size 8 \\
      --embedding-model <本地 MiniLM 目录绝对路径> [--grad-accum 2] [--max-length 384] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, namedtuple
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from research.memory_relations import features as FEAT
from research.memory_relations import metrics as EM
from research.memory_relations import train as T

MODEL_KINDS = ("frozen-joint", "finetuned-joint")
KINDS = MODEL_KINDS
                                          
EXPECTED_WEIGHTS = {"frozen-joint": "head_only", "finetuned-joint": "full"}
ARTIFACT_SCHEMA = "memory-relations-finetune-scorer-v1"
TRAINING_SCHEMA = "memory-relations-finetune-v1"
INPUT_RECIPE = "memory-relations-joint-input-v1"
WEIGHTS_FILE = "model.pt"
META_FILE = "meta.json"
COUNTING_UNIT = "row（同一 role 的多个 block 每行至多计一次）"

EPOCHS_DEFAULT = 3
BATCH_DEFAULT = 8
GRAD_ACCUM_DEFAULT = 2
MAX_LENGTH_DEFAULT = 384
LR_BACKBONE = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
INFER_BATCH = 16
EMPTY_MARKER = "[empty]"
SPLITS = T.SPLITS

_Block = namedtuple("_Block", "role text")


def joint_blocks(request: dict) -> "list[_Block]":
    """真实投影字段 → 有序 block 列表（序列化与截断统计共用同一实现）。"""
    view = FEAT.request_view(request)
    rules = view["rule_texts"]
    if len(rules) > 1:
        raise ValueError("finetune 只接受 existing ≤1 的配对请求（多规则整集行应在数据侧排除）")
    blocks = [_Block("candidate", view["candidate_text"])]
    if view["candidate_trigger"]:
        blocks.append(_Block("candidate_trigger", view["candidate_trigger"]))
    if rules:
        blocks.append(_Block("existing", rules[0]))
        if view["rule_triggers"][0]:
            blocks.append(_Block("existing_trigger", view["rule_triggers"][0]))
    else:
        blocks.append(_Block("existing", EMPTY_MARKER))
    blocks.extend(_Block("quote", text) for text in view["quotes"])
    blocks.extend(_Block("evidence", text) for text in view["evidence_texts"])
    return blocks


def _serialize(blocks: Sequence[_Block]) -> "tuple[str, list[tuple[str, int, int]]]":
    """唯一序列化真源：joint text 与逐 block 字符区间（块间 '\\n' 属于间隙，不归块）。"""
    pieces: list[str] = []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for index, block in enumerate(blocks):
        if index:
            pieces.append("\n")
            cursor += 1
        piece = f"[{block.role}] {block.text}"
        spans.append((block.role, cursor, cursor + len(piece)))
        pieces.append(piece)
        cursor += len(piece)
    return "".join(pieces), spans


def joint_text(request: dict) -> str:
    """角色分隔 joint input（训练与截断统计共用同一序列化）。"""
    return _serialize(joint_blocks(request))[0]


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _prefix_owners(tokenizer, text: str, spans: Sequence[tuple[str, int, int]]) -> "list[str] | None":
    """slow tokenizer 回退：按 block 字符边界的 token 前缀数切分归属（前缀边界口径）。"""
    owners: list[str] = []
    previous = 0
    for role, _, end in spans:
        count = len(tokenizer(text[:end], add_special_tokens=False)["input_ids"])
        if count < previous:
            return None                   
        owners.extend([role] * (count - previous))
        previous = count
    total = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if total > previous:
        owners.extend([None] * (total - previous))
    return owners


def _attributed_owners(tokenizer, text: str, spans: Sequence[tuple[str, int, int]]):
    """真实 token → role 归属：offset_mapping 优先，slow tokenizer 回退前缀边界。

    返回 (owners|None, special_count, attribution)；owners 为按序列顺序的真实 token
    归属标签（None/separator 表示不可归属），无法精确时为 None 并标注 unavailable。
    """
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    try:
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    except (TypeError, NotImplementedError, ValueError):
        encoded = None
    if encoded is not None and "offset_mapping" in encoded:
        owners = []
        for start, end in encoded["offset_mapping"]:
            if start == end:                                 
                continue
            owner = None
            for role, block_start, block_end in spans:
                if block_start <= start < block_end:
                    owner = role
                    break
            owners.append(owner if owner is not None else "separator")
        if len(owners) + special_count == len(encoded["input_ids"]):
            return owners, special_count, "offset_mapping"
    owners = _prefix_owners(tokenizer, text, spans)
    if owners is None:
        return None, special_count, "unavailable"
    return owners, special_count, "token_prefix_boundary"


def truncation_profile(tokenizer, requests: Sequence[dict], max_length: int) -> dict:
    """真实截断画像：整行真实 token 长度 + 逐 role 的 cut/dropped 行数归属。

    行截断按整行（含特殊 token）真实长度判定；role 归属基于 joint 序列的真实字符区间
    （offset_mapping，或 slow tokenizer 的前缀 token 边界）；同一 role 的多个 block 在
    同一行至多计一次（counting_unit=row）；无法精确归属时 attribution=unavailable，
    只保留行级计数。
    """
    requests = list(requests)
    if not requests:
        return {"rows": 0, "truncated_rows": 0, "truncation_rate": None,
                "max_length": int(max_length), "budget_tokens": None,
                "attribution": "unavailable", "counting_unit": COUNTING_UNIT,
                "unattributed_tokens": 0, "token_length": None, "by_source": {}}
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = max(0, int(max_length) - special_count)
    lengths: list[int] = []
    truncated_rows = 0
    unattributed = 0
    by_source: dict[str, dict] = {}
    methods: set[str] = set()
    for request in requests:
        text, spans = _serialize(joint_blocks(request))
        full_length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        lengths.append(full_length)
        if full_length > int(max_length):
            truncated_rows += 1
        owners, _, method = _attributed_owners(tokenizer, text, spans)
        methods.add(method)
        if owners is None:
            continue
        counts: dict[str, list[int]] = {}
        for index, owner in enumerate(owners):
            if owner is None or owner == "separator":
                unattributed += 1
                continue
            kept, total = counts.setdefault(owner, [0, 0])
            counts[owner] = [kept + (1 if index < budget else 0), total + 1]
        for role, (kept, total) in counts.items():
            if total > kept:                     
                record = by_source.setdefault(role, {"cut": 0, "dropped": 0})
                record["dropped" if kept == 0 else "cut"] += 1
    values = np.asarray(lengths, dtype=float)
    attribution = "unavailable" if methods == {"unavailable"} else \
        "mixed" if "unavailable" in methods or len(methods) > 1 else methods.pop()
    return {
        "rows": len(requests),
        "truncated_rows": truncated_rows,
        "truncation_rate": truncated_rows / len(requests),
        "max_length": int(max_length),
        "budget_tokens": budget,
        "attribution": attribution,
        "counting_unit": COUNTING_UNIT,
        "unattributed_tokens": unattributed,
        "token_length": {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                         "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
                         "max": float(values.max())},
        "by_source": {role: dict(record) for role, record in sorted(by_source.items())},
    }


class _JointClassifier(nn.Module):
    """Backbone [CLS] 池化 + dropout + 7 类线性 head（两臂唯一差别在是否冻结骨干）。"""

    def __init__(self, backbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), len(FEAT.CLASSES))

    def forward(self, **batch):
        outputs = self.backbone(**batch)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def resolve_device(name: str) -> torch.device:
    """显式解析设备；请求 GPU 不可用或越界一律如实报错，绝不回退 CPU。"""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or re.fullmatch(r"cuda:\d+", name):
        if not torch.cuda.is_available():
            raise RuntimeError(f"请求 GPU（--device {name}）但 CUDA 不可用：如实报错，不回退 CPU")
        index = int(name.split(":")[1]) if ":" in name else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"请求 GPU {name} 但只有 {torch.cuda.device_count()} 张卡：如实报错，不回退 CPU")
        return torch.device(name)
    raise ValueError(f"--device 只支持 cpu 或 cuda[:n]，实际 {name!r}")


def _resolve_bf16(mode: str, device: torch.device) -> bool:
    supported = device.type == "cuda" and bool(torch.cuda.is_bf16_supported())
    if mode == "off":
        return False
    if mode == "on" and not supported:
        raise ValueError("--bf16 on 但当前设备不支持 bf16")
    return supported


def _autocast(device: torch.device, bf16: bool):
    if bf16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _encode(tokenizer, texts: Sequence[str], max_length: int, device: torch.device) -> dict:
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=int(max_length),
                        return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _param_groups(module: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": float(lr), "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0})
    return groups


def build_optimizer(
    kind: str,
    model: _JointClassifier,
    *,
    lr_backbone: float = LR_BACKBONE,
    lr_head: float = LR_HEAD,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.optim.AdamW:
    """两臂唯一优化器构造：frozen 冻结骨干且只含 head 组；finetuned 含骨干+head 组。"""
    if kind not in MODEL_KINDS:
        raise ValueError(f"未知 cross-encoder 类型：{kind}")
    if kind == "frozen-joint":
        for param in model.backbone.parameters():
            param.requires_grad_(False)
    groups = [] if kind == "frozen-joint" else \
        _param_groups(model.backbone, lr_backbone, weight_decay)
    groups += _param_groups(model.classifier, lr_head, weight_decay)
    if not groups:
        raise ValueError("优化器没有可训练参数")
    return torch.optim.AdamW(groups)


def _cpu_state(model: nn.Module) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _training_windows(n: int, *, epochs: int, batch_size: int, grad_accum: int,
                      max_steps: int, rng):
    effective_batch = batch_size * grad_accum
    if max_steps:
        order, cursor, consumed = rng.permutation(n), 0, 0
        for step in range(1, max_steps + 1):
            chunks, previous_epoch = [], consumed // n
            for _ in range(grad_accum):
                parts, needed = [], batch_size
                while needed:
                    if cursor == n:
                        order, cursor = rng.permutation(n), 0
                    count = min(needed, n - cursor)
                    parts.append(order[cursor:cursor + count])
                    cursor += count
                    needed -= count
                chunks.append(np.concatenate(parts))
            consumed += effective_batch
            yield step, (consumed + n - 1) // n, chunks, consumed // n > previous_epoch
        return
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, effective_batch):
            selected = order[start:start + effective_batch]
            chunks = [selected[offset:offset + batch_size] for offset in range(0, len(selected), batch_size)]
            step += 1
            yield step, epoch, chunks, start + effective_batch >= n


def _validate_training_options(*, epochs, batch_size, grad_accum, max_length,
                               checkpoint_selection, selection_metric, patience,
                               min_delta, max_steps, eval_every_steps, eval_batch_size) -> None:
    if min(epochs, batch_size, grad_accum, max_length, eval_batch_size) < 1 \
            or min(patience, max_steps, eval_every_steps) < 0 \
            or not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("训练预算须为正，patience/max-steps/eval-every-steps/min-delta须非负有限")
    if checkpoint_selection not in ("final", "best-dev") \
            or selection_metric not in ("dev-loss", "dev-macro-f1"):
        raise ValueError("checkpoint_selection/selection_metric 无效")
    if patience and checkpoint_selection != "best-dev":
        raise ValueError("patience 只允许用于 best-dev；final 保留完整预算")


def train_model(
    kind: str,
    model: _JointClassifier,
    tokenizer,
    requests: Sequence[dict],
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    device: torch.device,
    bf16: bool,
    lr_backbone: float = LR_BACKBONE,
    lr_head: float = LR_HEAD,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_selection: str = "final",
    selection_metric: str = "dev-loss",
    patience: int = 0,
    min_delta: float = 0.0,
    max_steps: int = 0,
    eval_every_steps: int = 0,
    eval_batch_size: int = INFER_BATCH,
    dev_requests: "Sequence[dict] | None" = None,
    dev_labels: "np.ndarray | None" = None,
    dev_group_ids: "Sequence[str] | None" = None,
    event_callback=None,
) -> dict:
    """固定实例预算，尾批样本归一；独立dev根等权选CPU快照，row-loss仅辅助。"""
    _validate_training_options(epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                               max_length=max_length, checkpoint_selection=checkpoint_selection,
                               selection_metric=selection_metric, patience=patience, min_delta=min_delta,
                               max_steps=max_steps, eval_every_steps=eval_every_steps,
                               eval_batch_size=eval_batch_size)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(requests) or labels.shape != (len(requests),) or not 0 <= seed < 2 ** 32:
        raise ValueError("训练请求/labels长度或seed无效")
    T._require_seven_classes(labels)
    dev_requests = list(dev_requests) if dev_requests is not None else []
    dev_groups = list(dev_group_ids) if dev_group_ids is not None else []
    if dev_requests:
        if dev_labels is None or np.asarray(dev_labels).shape != (len(dev_requests),):
            raise ValueError("dev请求/labels长度不符")
        if len(dev_groups) != len(dev_requests) or any(
                not isinstance(group, str) or not group.strip() for group in dev_groups):
            raise ValueError("dev必须提供等长非空语义根dev_group_ids，禁止行等权回退")
    elif checkpoint_selection == "best-dev":
        raise ValueError("best-dev 必须提供独立dev配对行")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    frozen = kind == "frozen-joint"
    optimizer = build_optimizer(kind, model, lr_backbone=lr_backbone, lr_head=lr_head,
                                weight_decay=weight_decay)
    model.to(device)
    texts = [joint_text(request) for request in requests]
    planned_steps = max_steps or epochs * ((len(texts) + batch_size * grad_accum - 1) // (batch_size * grad_accum))
    planned_instances = max_steps * batch_size * grad_accum if max_steps else epochs * len(texts)
    history, updates = [], []
    best_state, best_value, selected_epoch, selected_step = None, None, None, None
    bad_evaluations, instances, real_tokens, padding_tokens = 0, 0, 0, 0
    interval_loss, interval_correct, interval_seen = 0.0, 0, 0
    stopped_early = False
    started = time.perf_counter()
    windows = _training_windows(len(texts), epochs=epochs, batch_size=batch_size,
                                grad_accum=grad_accum, max_steps=max_steps, rng=np.random.default_rng(seed))
    for step, epoch, chunks, epoch_end in windows:
        model.train()
        if frozen:
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        window_n = sum(len(chunk) for chunk in chunks)
        loss_sum, correct, window_real, window_padding = 0.0, 0, 0, 0
        for chunk in chunks:
            encoded = _encode(tokenizer, [texts[index] for index in chunk], max_length, device)
            targets = torch.as_tensor(labels[chunk], dtype=torch.long, device=device)
            count_real = int(encoded["attention_mask"].sum().item())
            window_real += count_real
            window_padding += encoded["input_ids"].numel() - count_real
            with _autocast(device, bf16):
                logits = model(**encoded)
                loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("训练loss非有限：停止，不导出工件")
            (loss / window_n).backward()
            loss_sum += float(loss.detach())
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
        optimizer.step()
        instances += window_n
        real_tokens += window_real
        padding_tokens += window_padding
        updates.append({"step": step, "epoch": epoch, "sample_instances": window_n,
                        "cumulative_sample_instances": instances, "microbatch_sizes": [len(chunk) for chunk in chunks],
                        "real_tokens": window_real, "padding_tokens": window_padding,
                        "loss": loss_sum / window_n, "accuracy": correct / window_n})
        if event_callback is not None:
            event_callback({"event": "update", **updates[-1]})
        interval_loss += loss_sum
        interval_correct += correct
        interval_seen += window_n
        evaluate_now = (step % eval_every_steps == 0) if eval_every_steps else epoch_end
        if evaluate_now or step == planned_steps:
            record = {"epoch": epoch, "step": step, "sample_instances": interval_seen,
                      "loss": interval_loss / interval_seen, "accuracy": interval_correct / interval_seen}
            if dev_requests:
                dev = eval_mode_metrics(model, tokenizer, dev_requests, dev_labels, max_length=max_length,
                                        device=str(device), batch_size=eval_batch_size, group_ids=dev_groups)
                value = float(dev["loss"] if selection_metric == "dev-loss" else dev["macro_f1"])
                if not np.isfinite(value):
                    raise ValueError("dev指标非有限：停止，不导出工件")
                record.update({"dev_loss": dev["loss"], "dev_row_loss": dev["row_loss"],
                               "dev_macro_f1": dev["macro_f1"], "dev_weighting": "root_equal",
                               "dev_root_count": len(set(dev_groups))})
                improved = best_value is None or (best_value - value > min_delta if selection_metric == "dev-loss"
                                                  else value - best_value > min_delta)
                if improved:
                    best_value, bad_evaluations = value, 0
                    if checkpoint_selection == "best-dev":
                        best_state, selected_epoch, selected_step = _cpu_state(model), epoch, step
                else:
                    bad_evaluations += 1
                if patience and bad_evaluations >= patience and step < planned_steps:
                    stopped_early = True
            history.append(record)
            if event_callback is not None:
                event_callback({"event": "dev_evaluation" if dev_requests else "training_interval", **record})
            print(f"    epoch {epoch} step {step}/{planned_steps}: loss={record['loss']:.4f} "
                  f"train_acc={record['accuracy']:.4f} dev_loss={record.get('dev_loss')}", flush=True)
            interval_loss, interval_correct, interval_seen = 0.0, 0, 0
        if stopped_early:
            break
    if checkpoint_selection == "final":
        best_state = _cpu_state(model)
        selected_epoch, selected_step = updates[-1]["epoch"], updates[-1]["step"]
    model.load_state_dict(best_state, strict=True)
    model.eval()
    result = {"history": history, "updates": updates, "seconds": round(time.perf_counter() - started, 3),
              "budget": {"mode": "fixed-updates" if max_steps else "epochs", "planned_updates": planned_steps,
                         "planned_sample_instances": planned_instances, "actual_updates": len(updates),
                         "actual_sample_instances": instances, "real_tokens": real_tokens, "padding_tokens": padding_tokens,
                         "token_counting": "实际训练编码含special tokens；padding单列；不含dev/eval"},
              "selection": {"mode": checkpoint_selection, "metric": selection_metric,
                            "weighting": "root_equal" if dev_requests else None,
                            "dev_root_count": len(set(dev_groups)),
                            "secondary_metric": "dev_row_loss" if dev_requests else None,
                            "selected_epoch": selected_epoch, "selected_step": selected_step,
                            "best_dev_value": best_value, "patience": patience, "min_delta": min_delta,
                            "patience_unit": "dev evaluations", "stopped_early": stopped_early,
                            "state_restored": True, "snapshot_device": "cpu"}}
    if event_callback is not None:
        event_callback({"event": "selection_restored", **result["selection"], "budget": result["budget"]})
    return result


def logits_for(model: _JointClassifier, tokenizer, requests: Sequence[dict], *,
               max_length: int, device: str, batch_size: int = INFER_BATCH) -> np.ndarray:
    """eval 模式（无 dropout）推理 logits：scorer 与训练后 train_eval 共用同一实现。"""
    rows = [joint_text(request) for request in requests]
    if not rows:
        return np.zeros((0, len(FEAT.CLASSES)), dtype=np.float64)
    model.to(device)
    model.eval()
    collected = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            encoded = _encode(tokenizer, rows[start:start + int(batch_size)], max_length,
                              torch.device(device))
            collected.append(model(**encoded).detach().float().cpu().numpy())
    return np.concatenate(collected, axis=0).astype(np.float64)


def eval_mode_metrics(model: _JointClassifier, tokenizer, requests: Sequence[dict],
                      labels: np.ndarray, *, max_length: int, device: str,
                      batch_size: int = INFER_BATCH,
                      group_ids: "Sequence[str] | None" = None) -> dict:
    """同一eval通道；传语义根时主指标根等权，原行等权loss保留为secondary。"""
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(requests),):
        raise ValueError("eval请求/labels长度不符")
    groups = list(group_ids) if group_ids is not None else None
    if groups is not None and (len(groups) != len(y) or any(
            not isinstance(group, str) or not group.strip() for group in groups)):
        raise ValueError("eval语义根必须为等长非空字符串向量")
    weighting = "root_equal" if groups is not None else "row"
    if not len(y):
        return {"n": 0, "loss": None, "row_loss": None, "macro_f1": None,
                "accuracy": None, "weighting": weighting, "n_groups": 0,
                "mode": "eval(no dropout)", "prediction_counts": {}}
    logits = logits_for(model, tokenizer, requests, max_length=max_length, device=device,
                        batch_size=batch_size)
    targets = torch.from_numpy(y)
    losses = nn.functional.cross_entropy(torch.from_numpy(logits).float(), targets,
                                        reduction="none").numpy()
    weights = EM._group_weights(groups) if groups is not None else np.ones(len(y))
    predictions = logits.argmax(axis=1)
    return {
        "n": int(len(y)), "n_groups": len(set(groups)) if groups is not None else None,
        "loss": float(np.average(losses, weights=weights)),
        "row_loss": float(losses.mean()), "weighting": weighting,
        "accuracy": float(np.average(predictions == y, weights=weights)),
        "macro_f1": EM.macro_f1(y, predictions, sample_weight=weights),
        "mode": "eval(no dropout)",
        "prediction_counts": {FEAT.CLASSES[index]: int((predictions == index).sum())
                              for index in range(len(FEAT.CLASSES))},
    }


class CrossEncoderScorer:
    """联合输入 cross-encoder 的推理封装：classes 恒为 spec 类序，输出恒为 n×7。"""

    classes = FEAT.CLASSES

    def __init__(
        self,
        *,
        kind: str,
        model: _JointClassifier,
        tokenizer,
        max_length: int,
        device: str = "cpu",
        temperature: float = 1.0,
        batch_size: int = INFER_BATCH,
        threshold: "float | None" = None,
        threshold_feasible: bool = False,
    ) -> None:
        if kind not in MODEL_KINDS:
            raise ValueError(f"未知 cross-encoder 类型：{kind}")
        if float(temperature) <= 0.0:
            raise ValueError("temperature 必须为正")
        _check_threshold(threshold, threshold_feasible)
        self.kind = kind
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = str(device)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        self.threshold = threshold
        self.threshold_feasible = bool(threshold_feasible)
        self.model.to(self.device)
        self.model.eval()

    def _logits(self, requests: Sequence[dict]) -> np.ndarray:
        return logits_for(self.model, self.tokenizer, requests, max_length=self.max_length,
                          device=self.device, batch_size=self.batch_size)

    @staticmethod
    def _items(requests) -> list:
        return [requests] if isinstance(requests, dict) else list(requests)

    def predict_proba_raw(self, requests) -> np.ndarray:
        items = self._items(requests)
        if not items:
            return np.zeros((0, len(FEAT.CLASSES)), dtype=np.float64)
        return FEAT.softmax(self._logits(items))

    def predict_proba(self, requests) -> np.ndarray:
        """requests：请求字典序列（也接受单个请求）→ (n, spec 类序) 校准后概率。"""
        items = self._items(requests)
        if not items:
            return np.zeros((0, len(FEAT.CLASSES)), dtype=np.float64)
        return FEAT.softmax(self._logits(items) / self.temperature)


def _check_threshold(threshold, threshold_feasible) -> None:
    """threshold/threshold_feasible 一致性（null ⇔ false；非 null 必须非负有限）。"""
    if not isinstance(threshold_feasible, bool):
        raise ValueError("threshold_feasible 必须是布尔")
    if threshold is None:
        if threshold_feasible:
            raise ValueError("threshold 为 null 时 threshold_feasible 必须为 false")
        return
    value = float(threshold)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("threshold 必须是非负有限数字或 null")
    if not threshold_feasible:
        raise ValueError("threshold 非 null 时 threshold_feasible 必须为 true")


def build_joint_model(backbone_dir: "str | Path", *, dropout: float = DROPOUT):
    """本地权重构建 backbone+head；返回 (model, config)；禁止下载。"""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(backbone_dir), local_files_only=True)
    backbone = AutoModel.from_pretrained(str(backbone_dir), config=config, local_files_only=True)
    model = _JointClassifier(backbone, int(config.hidden_size), dropout)
    return model, config


def save_scorer(
    scorer: CrossEncoderScorer,
    directory: Path,
    *,
    weights: str,
    backbone_dir: Path,
    backbone_fingerprint: str,
    train_meta: dict,
) -> dict:
    """持久化 state_dict + meta + 内容指纹；kind 与 weights 严格配对。

    meta 顶层保存 calibration 选点 threshold/threshold_feasible（null 保持 null），
    上游 evaluate 直接读取，不因缺键误报“无可行”。
    """
    if scorer.kind not in MODEL_KINDS:
        raise ValueError(f"未知 cross-encoder 类型：{scorer.kind}")
    if weights not in ("head_only", "full"):
        raise ValueError("weights 只支持 head_only | full")
    if EXPECTED_WEIGHTS[scorer.kind] != weights:
        raise ValueError(f"{scorer.kind} 必须配 weights={EXPECTED_WEIGHTS[scorer.kind]}，实际 {weights}")
    _check_threshold(scorer.threshold, scorer.threshold_feasible)
    hidden_size = int(scorer.model.classifier.in_features)
    dropout = float(scorer.model.dropout.p)
    state = _cpu_state(scorer.model)
    if weights == "head_only":
        state = {name: tensor for name, tensor in state.items() if name.startswith("classifier.")}
        if not state:
            raise ValueError("head_only 载荷为空（没有 classifier.* 权重）")
    else:
        if not any(name.startswith("backbone.") for name in state) \
                or not any(name.startswith("classifier.") for name in state):
            raise ValueError("full 载荷必须同时含 backbone.* 与 classifier.* 权重")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "classes": list(FEAT.CLASSES), "recipe": INPUT_RECIPE,
                "kind": scorer.kind, "weights": weights,
                "hidden_size": hidden_size, "dropout": dropout}, directory / WEIGHTS_FILE)
    meta = {
        "schema": ARTIFACT_SCHEMA,
        "kind": scorer.kind,
        "classes": list(FEAT.CLASSES),
        "input_recipe": INPUT_RECIPE,
        "max_length": int(scorer.max_length),
        "batch_size": int(scorer.batch_size),
        "temperature": float(scorer.temperature),
        "threshold": scorer.threshold,
        "threshold_feasible": bool(scorer.threshold_feasible),
        "weights": weights,
        "weights_sha256": _file_sha256(directory / WEIGHTS_FILE),
        "backbone_path": str(backbone_dir),
        "backbone_fingerprint": backbone_fingerprint,
        "tokenizer_path": str(backbone_dir),
        "tokenizer_fingerprint": backbone_fingerprint,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train": train_meta,
    }
    (directory / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return meta


def is_finetune_artifact(model_dir: "str | Path") -> bool:
    """工件识别（供 features.load_scorer 分派用）：meta.json 且 schema 匹配。"""
    meta_path = Path(model_dir) / META_FILE
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(meta, dict) and meta.get("schema") == ARTIFACT_SCHEMA


def load_scorer(model_dir: "str | Path", device: str = "cpu") -> CrossEncoderScorer:
    """由features.load_scorer惰性分派的joint工件加载器；契约不符即抛错，不静默降级。"""
    from transformers import AutoModel, AutoTokenizer

    directory = Path(model_dir)
    meta_path, weights_path = directory / META_FILE, directory / WEIGHTS_FILE
    if not meta_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"finetune 工件不完整：{directory}（需要 {META_FILE} 与 {WEIGHTS_FILE}）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("finetune 工件 schema 不符")
    if tuple(meta.get("classes") or ()) != FEAT.CLASSES:
        raise ValueError("finetune 工件类别序与 spec 类序不符")
    if meta.get("input_recipe") != INPUT_RECIPE:
        raise ValueError("finetune 工件输入配方与当前实现不符")
    kind, weights = meta.get("kind"), meta.get("weights")
    if kind not in MODEL_KINDS:
        raise ValueError(f"finetune kind 无效：{kind!r}（只允许 {MODEL_KINDS}）")
    if weights not in ("head_only", "full"):
        raise ValueError(f"finetune weights 无效：{weights!r}")
    if EXPECTED_WEIGHTS[kind] != weights:
        raise ValueError(f"finetune kind 与 weights 不配对：{kind} ↔ {weights}")
    for key in ("threshold", "threshold_feasible"):
        if key not in meta:
            raise ValueError(f"finetune meta 缺少顶层 {key}（evaluate 依赖，缺键会误报无可行）")
    _check_threshold(meta["threshold"], meta["threshold_feasible"])
    if _file_sha256(weights_path) != meta.get("weights_sha256"):
        raise ValueError("finetune 权重文件哈希不符（损坏或被替换）")
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_path"], local_files_only=True)
    if FEAT.model_fingerprint(meta["tokenizer_path"]) != meta.get("tokenizer_fingerprint"):
        raise ValueError("tokenizer 内容指纹与训练时不一致")
    if weights == "head_only" and \
            FEAT.model_fingerprint(meta["backbone_path"]) != meta.get("backbone_fingerprint"):
        raise ValueError("冻结骨干内容指纹与训练时不一致；换模型必须重新训练")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload \
            or tuple(payload.get("classes") or ()) != FEAT.CLASSES \
            or payload.get("recipe") != INPUT_RECIPE:
        raise ValueError("finetune 权重载荷结构与当前实现不符")
    if payload.get("kind") != kind or payload.get("weights") != weights:
        raise ValueError("finetune 载荷 kind/weights 与 meta 不一致")
    if int(payload.get("hidden_size", -1)) != int(meta["hidden_size"]) \
            or float(payload.get("dropout", -1.0)) != float(meta["dropout"]):
        raise ValueError("finetune 载荷与 meta 的 hidden_size/dropout 不一致")
    state_keys = set(payload["state_dict"])
    if weights == "head_only":
        if not state_keys or any(not name.startswith("classifier.") for name in state_keys):
            raise ValueError(f"head_only 载荷只允许 classifier.* 键：{sorted(state_keys)[:5]}")
    elif not any(name.startswith("backbone.") for name in state_keys) \
            or not any(name.startswith("classifier.") for name in state_keys):
        raise ValueError("full 载荷必须同时含 backbone.* 与 classifier.* 键")
    backbone = AutoModel.from_pretrained(meta["backbone_path"], local_files_only=True)
    model = _JointClassifier(backbone, int(meta["hidden_size"]), float(meta["dropout"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"finetune 权重含多余键：{sorted(unexpected)[:5]}")
    if weights == "full":
        if missing:
            raise ValueError(f"finetune 全量权重缺键：{sorted(missing)[:5]}")
    elif any(not name.startswith("backbone.") for name in missing):
        raise ValueError(f"head_only 工件缺少非骨干权重：{sorted(missing)[:5]}")
    model.eval()
    if kind == "frozen-joint":
        for param in model.backbone.parameters():
            param.requires_grad_(False)
    return CrossEncoderScorer(kind=kind, model=model, tokenizer=tokenizer,
                              max_length=int(meta["max_length"]), device=device,
                              temperature=float(meta.get("temperature", 1.0)),
                              batch_size=int(meta.get("batch_size", INFER_BATCH)),
                              threshold=meta["threshold"],
                              threshold_feasible=meta["threshold_feasible"])


def check_reload_parity(scorer: CrossEncoderScorer, directory: Path,
                        probe_requests: Sequence[dict], *, device: str, atol: float = 1e-6) -> dict:
    """state_dict 重载 parity：概率/argmax/threshold 元数据必须一致，否则 fail-loud。"""
    reloaded = load_scorer(directory, device=device)
    before = scorer.predict_proba(list(probe_requests))
    after = reloaded.predict_proba(list(probe_requests))
    difference = float(np.max(np.abs(before - after))) if len(before) else 0.0
    argmax_match = bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1)))
    ok = (argmax_match and difference <= atol
          and tuple(reloaded.classes) == FEAT.CLASSES
          and reloaded.temperature == scorer.temperature
          and reloaded.max_length == scorer.max_length
          and reloaded.threshold == scorer.threshold
          and reloaded.threshold_feasible == scorer.threshold_feasible)
    if not ok:
        raise RuntimeError(f"finetune 重载 parity 失败：max_abs_diff={difference}, argmax_match={argmax_match}")
    return {"max_abs_diff": difference, "argmax_match": argmax_match, "ok": True}


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


_require_seven_classes = T._require_seven_classes
_require_group_ids = T._require_group_ids


class _MemoryStage:
    def __init__(self, records: dict, name: str, device: torch.device) -> None:
        self.records, self.name, self.device = records, name, device

    def __enter__(self):
        self.start = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.start = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, exc_type, exc, traceback):
        record = {"start_allocated_mb": self.start, "peak_allocated_mb": None,
                  "end_allocated_mb": None, "peak_reserved_mb": None, "failed": exc_type is not None}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            record.update({"peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                           "end_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
                           "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)})
        self.records[self.name] = record
        return False


def _append_training_event(path: Path, event: dict) -> None:
    record = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _split_truncation(tokenizer, rows: Sequence[dict], max_length: int) -> dict:
    profile = truncation_profile(tokenizer, [row["request"] for row in rows], max_length)
    profile["by_class"] = {label: truncation_profile(
        tokenizer, [row["request"] for row in rows if row["label"] == label], max_length)
        for label in FEAT.CLASSES}
    return profile


def _sentence_transformer_facts(embedding_dir: Path) -> dict:
    """官方 sentence-transformers 默认（max_seq_length / mean pooling）与 joint 口径对照。"""
    facts: dict = {"config_file": "sentence_bert_config.json", "max_seq_length": None,
                   "mean_pooling_default": None}
    config_path = embedding_dir / "sentence_bert_config.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("max_seq_length"), int):
                facts["max_seq_length"] = payload["max_seq_length"]
        except (OSError, json.JSONDecodeError):
            pass
    pooling_path = embedding_dir / "1_Pooling" / "config.json"
    if pooling_path.is_file():
        try:
            payload = json.loads(pooling_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                facts["mean_pooling_default"] = bool(payload.get("pooling_mode_mean_tokens"))
        except (OSError, json.JSONDecodeError):
            pass
    return facts


def _parse_models(text: str) -> list[str]:
    models = list(dict.fromkeys(T._parse_csv(text)))
    unknown = [kind for kind in models if kind not in MODEL_KINDS]
    if not models or unknown:
        raise ValueError(f"--models 只支持 {MODEL_KINDS}，实际 {unknown or text!r}")
    return models


def _default_out(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "memory-relations" / datetime.now().strftime("finetune-%Y%m%d-%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="配对 JSONL（pairs.jsonl；含 train/calibration[/heldout/ood]）")
    parser.add_argument("--challenge", default="", help="配对 challenge JSONL（只用于最终评测）")
    parser.add_argument("--out", default="", help="输出目录（正式运行由父侧指定 runs/ 下的路径）")
    parser.add_argument("--models", default="frozen-joint,finetuned-joint")
    parser.add_argument("--sizes", default="2000")
    parser.add_argument("--seeds", default="13,20260919,424242")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda[:n]；请求 GPU 失败即报错，不回退")
    parser.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_DEFAULT)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT)
    parser.add_argument("--embedding-model", default="", help="本地 MiniLM 目录绝对路径（骨干+tokenizer；禁止下载）")
    parser.add_argument("--bf16", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=EM.BOOTSTRAP_SAMPLES)
    parser.add_argument("--checkpoint-selection", choices=("final", "best-dev"), default="final")
    parser.add_argument("--selection-metric", choices=("dev-loss", "dev-macro-f1"), default="dev-loss",
                        help="dev指标按语义根等权；dev-loss主监控，dev_row_loss仅辅助，不用终测选权重")
    parser.add_argument("--patience", type=int, default=None,
                        help="dev评估连续未改善次数；fixed-updates/final默认0，epoch best-dev默认2")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0, help=">0时覆盖epochs，每update恰batch×accum个实例")
    parser.add_argument("--eval-every-steps", type=int, default=0, help="0为每epoch，>0为update间隔；末步必评")
    parser.add_argument("--eval-batch-size", type=int, default=INFER_BATCH)
    parser.add_argument("--eval-splits", default="heldout,ood,challenge",
                        help="none（调参）或 heldout,ood,challenge 的子集；parity只用dev/train")
    parser.add_argument("--calibration-protocol", choices=("legacy", "partitioned"), default="legacy")
    parser.add_argument("--calibration-stage", choices=("candidate",), default="candidate",
                        help="partitioned训练只读prob/policy；audit须train --mode lock/audit独立一次执行")
    parser.add_argument("--smoke", action="store_true",
                        help="测试模式：epochs≤1、fixed updates≤3、bootstrap≤50，明确标注非正式实验")
    return parser


def main(argv: "Sequence[str] | None" = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    models = _parse_models(args.models)
    sizes = sorted(T._parse_ints(args.sizes, "--sizes"))
    seeds = T._parse_ints(args.seeds, "--seeds")
    epochs, batch_size, grad_accum, max_length = args.epochs, args.batch_size, args.grad_accum, args.max_length
    patience = args.patience if args.patience is not None else (
        0 if args.max_steps or args.checkpoint_selection == "final" else 2)
    _validate_training_options(epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                               max_length=max_length, checkpoint_selection=args.checkpoint_selection,
                               selection_metric=args.selection_metric, patience=patience, min_delta=args.min_delta,
                               max_steps=args.max_steps, eval_every_steps=args.eval_every_steps,
                               eval_batch_size=args.eval_batch_size)
    if args.bootstrap_samples < 1 or args.threads < 1:
        raise ValueError("--bootstrap-samples/--threads 必须 ≥ 1")
    if args.smoke:
        epochs = min(epochs, 1)
        args.max_steps = min(args.max_steps, 3)
        args.bootstrap_samples = min(args.bootstrap_samples, 50)
    out_dir = Path(args.out) if args.out else _default_out(repo_root)
    if out_dir.exists():
        raise FileExistsError(f"输出已存在：{out_dir}；禁止覆盖历史实验")
    data_path = Path(args.data)
    prepared_data = T.prepare_data(args)
    input_contract = T._data_contract(prepared_data)
    by_split, pair_by_split = prepared_data["by_split"], prepared_data["pairs"]
    excluded_by_split = prepared_data["excluded"]
    train_rows, calibration_rows, dev_rows = (pair_by_split[name] for name in ("train", "calibration", "dev"))
    if args.checkpoint_selection == "best-dev" and not dev_rows:
        raise ValueError("best-dev 必须提供独立dev配对行")
    sample_maps = T.preflight_training(train_rows, sizes, seeds, out_dir)
    challenge_path, challenge_rows = prepared_data["challenge_path"], pair_by_split["challenge"]
    challenge_excluded = excluded_by_split["challenge"]
    diagnostic_calibration = prepared_data["calibration_diagnostic_rows"]
    calibration_diagnostic_mask = prepared_data["calibration_diagnostic_mask"]
    calibration_metric_scope = "prob_policy_only" if args.calibration_protocol == "partitioned" else "legacy_calibration"
    eval_sets = [("calibration", diagnostic_calibration)] + prepared_data["eval_sets"]
    overlaps = prepared_data["overlaps"]
    device = resolve_device(args.device)
    bf16 = _resolve_bf16(args.bf16, device)
    torch.set_num_threads(int(args.threads))

    embedding_dir = T.resolve_embedding_model(args.embedding_model, repo_root)
    from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

    config = AutoConfig.from_pretrained(str(embedding_dir), local_files_only=True)
    if max_length > int(config.max_position_embeddings):
        raise ValueError(f"--max-length {max_length} 超过骨干 max_position_embeddings "
                         f"{config.max_position_embeddings}")
    tokenizer = AutoTokenizer.from_pretrained(str(embedding_dir), local_files_only=True)
    started_plan = time.perf_counter()
    fingerprint = FEAT.model_fingerprint(embedding_dir)
    semantics = {
        "joint_pooling": "CLS（v2 两臂同输入、同 head；唯一差异是骨干是否冻结）",
        "joint_max_length": max_length,
        "v1_reference": "v1 学生臂是 SentenceTransformer 句向量口径（mean pooling）",
        "sentence_transformer_defaults": _sentence_transformer_facts(embedding_dir),
        "comparison_scope": "只比较 v2 内两臂解释冻结效应；v1→v2 同时改变输入序列、池化与"
                            "序列长度（官方 sentence_bert_config 与其 mean pooling），"
                            "v1→v2 的差异不能唯一归因于 finetune。",
    }
    training_options = {"checkpoint_selection": args.checkpoint_selection, "selection_metric": args.selection_metric,
                        "patience": patience, "min_delta": args.min_delta, "max_steps": args.max_steps,
                        "eval_every_steps": args.eval_every_steps, "eval_batch_size": args.eval_batch_size}
    manifest = {
        "schema": TRAINING_SCHEMA + "-planned",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "planned_before_fit",
        "training_input_contract": input_contract,
        "args": {
            "data": str(data_path), "challenge": str(challenge_path) if challenge_path else None,
            "out": str(out_dir), "models": models, "sizes": sizes, "seeds": seeds,
            "epochs": epochs, "batch_size": batch_size, "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum, "max_length": max_length,
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD, "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT, "device": str(device), "bf16": bf16, "bf16_requested": args.bf16,
            "threads": int(args.threads), "smoke": bool(args.smoke), **training_options,
            "eval_splits": list(prepared_data["eval_splits"]), "calibration_protocol": args.calibration_protocol,
            "calibration_stage": args.calibration_stage,
        },
        "budgets": [{"seed": seed, "size_requested": size, "n_train": len(indices),
                     "n_groups": len({train_rows[index]["group_id"] for index in indices}),
                     "updates": args.max_steps or epochs * ((len(indices) + batch_size * grad_accum - 1) // (batch_size * grad_accum)),
                     "sample_instances": args.max_steps * batch_size * grad_accum if args.max_steps else epochs * len(indices)}
                    for seed, size_map in sample_maps.items() for size, indices in size_map.items()],
        "source_sha256": {
            "data": _file_sha256(data_path),
            "challenge": _file_sha256(challenge_path) if challenge_path else None,
        },
        "sampling": {"function": "train.nested_group_indices", "group_key": "group_id",
                     "sizes": sizes, "seeds": seeds,
                     "granularity_note": "整数 size 为下界：按整组取行，actual n（n_train）可能不同，"
                                         "运行记录同时给出 size_requested 与 n_train"},
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                      "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
                      "weight_decay": WEIGHT_DECAY, "bias_and_layernorm_no_decay": True,
                      "loss": "CrossEntropyLoss（无类别权重）"},
        "epoch_selection": args.checkpoint_selection,
        "selection_rules": {
            "weights": "final保留末步；best-dev仅按独立dev的预定指标/min-delta选择，CPU clone恢复后导出",
            "temperature": "legacy仅calibration；partitioned candidate仅prob温度、policy primary候选点，audit隔离给唯一study锁",
            "audit": "candidate拒绝audit输入且阈值null；独立train --mode lock/audit锁工件/温度/点后只审计一次，失败也不重试",
            "heldout_ood_challenge": "只评显式选中的终测；none不打开独立终测文件；parity仅dev/train",
            "patience": "连续未改善dev评估次数；固定步默认0跑满，与epochs独立",
        },
        "artifact_contract": {
            "kind_weights_pairing": EXPECTED_WEIGHTS,
            "meta_top_level": ["threshold", "threshold_feasible"],
            "meta_note": "threshold/threshold_feasible 来自 calibration 选点，null 保持 null；"
                         "上游 evaluate 直接读取，缺键即 fail-loud 而不是误报无可行",
        },
        "classes": list(FEAT.CLASSES),
        "input_recipe": INPUT_RECIPE,
        "semantics": semantics,
        "model": {
            "backbone_path": str(embedding_dir), "tokenizer_path": str(embedding_dir),
            "backbone_fingerprint": fingerprint, "tokenizer_fingerprint": fingerprint,
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "max_position_embeddings": int(config.max_position_embeddings),
        },
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "planned_runs": len(models) * len(sizes) * len(seeds),
        "notes": [
            "参数/随机 seed/数据 sha256 在 fit 前落盘（本文件）；权重选择以checkpoint_selection为准，训练预算固定。",
            "GPU 不可用或显存分配失败一律报错，不静默回退 CPU。",
            "截断按 joint 序列真实字符边界与 tokenizer offset（slow 回退前缀边界）逐 role 归属，"
            "同一 role 多 block 每行至多计一次；无法精确归属时明确标注。",
            "v1 与 v2 的输入/池化口径不同：只在 v2 内比较两臂解释冻结效应。",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "planned_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"planned manifest -> {out_dir / 'planned_manifest.json'}"
          f"（fingerprint {fingerprint[:12]}，{time.perf_counter() - started_plan:.1f}s）", flush=True)

    truncation = {name: _split_truncation(tokenizer, source_rows, max_length)
                  for name, source_rows in [("train", train_rows), ("dev", dev_rows)] + eval_sets}
    for name, profile in truncation.items():
        if profile["truncation_rate"]:
            print(f"truncation[{name}]: {profile['truncated_rows']}/{profile['rows']} "
                  f"({profile['truncation_rate']:.1%}) by_source={profile['by_source']} "
                  f"attribution={profile['attribution']}", flush=True)

    y_calibration = T._labels(calibration_rows)
    calibration_requests = [row["request"] for row in calibration_rows]
    dev_requests, y_dev = [row["request"] for row in dev_rows], T._labels(dev_rows)
    group_ids = {name: T._group_ids(eval_rows) for name, eval_rows in eval_sets}
    requests_by_set = {name: [row["request"] for row in eval_rows] for name, eval_rows in eval_sets}
    labels_by_set = {name: T._labels(eval_rows) for name, eval_rows in eval_sets}
    triggers_by_set = {name: T._triggers(eval_rows) for name, eval_rows in eval_sets}
    probe_requests = [row["request"] for row in (dev_rows or train_rows)[:8]]

    runs: list[dict] = []
    curve_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for seed in seeds:
        for size in sizes:
            indices = sample_maps[seed][size]
            sample = [train_rows[index] for index in indices]
            y_sample = T._labels(sample)
            sample_requests = [row["request"] for row in sample]
            sample_truncation = _split_truncation(tokenizer, sample, max_length)
            for kind in models:
                started = time.perf_counter()
                memory_stages = {}
                trace_path = out_dir / "training" / f"{kind}-size{size}-seed{seed}.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)

                def record_event(event):
                    _append_training_event(trace_path, event)

                record_event({"event": "planned", "model": kind, "seed": seed, "size_requested": size,
                              "n_train": len(sample), **training_options})
                build_started = time.perf_counter()
                random.seed(seed)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                with _MemoryStage(memory_stages, "build", device):
                    model, _ = build_joint_model(embedding_dir, dropout=DROPOUT)
                    model.to(device)
                build_seconds = round(time.perf_counter() - build_started, 3)
                print(f"[{seed}] size={size} {kind}: training {epochs} epoch(s), "
                      f"max_steps={args.max_steps} on {len(sample)} rows ({device})", flush=True)
                try:
                    with _MemoryStage(memory_stages, "train_and_dev", device):
                        training = train_model(
                            kind, model, tokenizer, sample_requests, y_sample,
                            seed=seed, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                            max_length=max_length, device=device, bf16=bf16, **training_options,
                            dev_requests=dev_requests, dev_labels=y_dev,
                            dev_group_ids=T._root_ids(dev_rows), event_callback=record_event)
                except Exception as error:
                    record_event({"event": "failed", "error_type": type(error).__name__, "memory_stages": memory_stages})
                    raise
                with _MemoryStage(memory_stages, "selected_train_eval", device):
                    train_eval = eval_mode_metrics(model, tokenizer, sample_requests, y_sample,
                                                   max_length=max_length, device=str(device),
                                                   batch_size=args.eval_batch_size)
                scorer = CrossEncoderScorer(kind=kind, model=model, tokenizer=tokenizer,
                                            max_length=max_length, device=str(device), batch_size=args.eval_batch_size)
                with _MemoryStage(memory_stages, "calibration", device):
                    calibration_logits = scorer._logits(calibration_requests)
                    raw_calibration = FEAT.softmax(calibration_logits)
                    calibration = T.calibrate(calibration_logits, calibration_rows, protocol=args.calibration_protocol,
                                              stage=args.calibration_stage)
                    scorer.temperature = float(calibration["temperature"])
                    scaled_calibration = FEAT.softmax(calibration_logits / scorer.temperature)
                    operating_threshold, feasible_point = calibration["threshold"], calibration["selection"]
                    scorer.threshold, scorer.threshold_feasible = operating_threshold, calibration["threshold_feasible"]
                if operating_threshold is None:
                    print(f"    {kind}: 无可行阈值（不声明已选操作策略）", flush=True)

                eval_metrics: dict[str, dict] = {}
                for name, eval_rows in eval_sets:
                    if not eval_rows:
                        continue
                    requests = requests_by_set[name]
                    with _MemoryStage(memory_stages, f"eval_{name}", device):
                        logits = scorer._logits(requests)
                        probs, raw_probs = FEAT.softmax(logits / scorer.temperature), FEAT.softmax(logits)
                        result = EM.evaluate(
                            probs, labels_by_set[name], raw_probs=raw_probs,
                            has_trigger=triggers_by_set[name], group_ids=group_ids[name],
                            threshold=operating_threshold,
                            bootstrap_samples=args.bootstrap_samples, seed=seed)
                    eval_metrics[name] = result
                    for point in result["risk_coverage"]:
                        curve_rows.append({"model": kind, "size": size, "seed": seed,
                                           "eval_set": name, **point})
                    for position, row in enumerate(eval_rows):
                        verdict = (EM.pair_verdict(probs[position], row["request"], operating_threshold)
                                   if operating_threshold is not None else None)
                        verdict_tau0 = EM.pair_verdict(probs[position], row["request"], 0.0)
                        prediction_rows.append({
                            "model": kind, "size": size, "seed": seed, "eval_set": name,
                            "id": row.get("id"), "group_id": row.get("group_id"),
                            "label": row["label"], "pred": FEAT.CLASSES[int(probs[position].argmax())],
                            "has_trigger": bool(triggers_by_set[name][position]),
                            "verdict": verdict,
                            "allow": None if verdict is None else verdict in EM.ALLOWING_CLASSES,
                            "verdict_tau0": verdict_tau0,
                            "allow_tau0": verdict_tau0 in EM.ALLOWING_CLASSES,
                            "probs": [round(float(value), 6) for value in probs[position]],
                        })

                weights = EXPECTED_WEIGHTS[kind]
                artifact_dir = out_dir / "scorers" / f"{kind}-size{size}-seed{seed}"
                with _MemoryStage(memory_stages, "export", device):
                    meta = save_scorer(scorer, artifact_dir, weights=weights,
                                       backbone_dir=embedding_dir, backbone_fingerprint=fingerprint,
                                       train_meta={"size_requested": size, "n_train": int(len(sample)),
                                                   "seed": seed, "epochs": epochs, **training_options,
                                                   "batch_size": batch_size, "grad_accum": grad_accum,
                                                   "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
                                                   "threshold": operating_threshold,
                                                   "threshold_feasible": scorer.threshold_feasible,
                                                   "calibration": calibration, "calibration_stage": args.calibration_stage,
                                                   "training_input_contract": input_contract,
                                                   "train_history": training["history"],
                                                   "selection": training["selection"], "budget": training["budget"],
                                                   "train_eval": train_eval,
                                                   "eval_batch_size": args.eval_batch_size,
                                                   "device_at_train": str(device), "bf16": bf16,
                                                   "smoke": bool(args.smoke)})
                with _MemoryStage(memory_stages, "reload_parity", device):
                    parity = check_reload_parity(scorer, artifact_dir, probe_requests, device=str(device))
                record_event({"event": "completed_run", "selection": training["selection"],
                              "budget": training["budget"], "calibration": calibration,
                              "train_eval": train_eval, "parity": parity, "memory_stages": memory_stages})
                peaks = [value["peak_allocated_mb"] for value in memory_stages.values()
                         if value["peak_allocated_mb"] is not None]
                memory_peak_mb = max(peaks) if peaks else None
                runs.append({
                    "training_trace": str(trace_path.relative_to(out_dir)).replace("\\", "/"),
                    "model": kind, "size_requested": size, "n_train": int(len(sample)),
                    "n_train_groups": len({train_rows[index]["group_id"] for index in indices}),
                    "seed": seed, "epochs": epochs, "batch_size": batch_size,
                    "grad_accum": grad_accum, "effective_batch": batch_size * grad_accum,
                    "max_length": max_length, "device": str(device), "bf16": bf16,
                    "build_seconds": build_seconds, **training_options,
                    "train_seconds": training["seconds"], "train_history": training["history"],
                    "selection": training["selection"], "budget": training["budget"], "updates": training["updates"],
                    "train_eval": train_eval,
                    "truncation_train_sample": sample_truncation,
                    "temperature": scorer.temperature,
                    "nll_raw": (_nll(raw_calibration[calibration_diagnostic_mask], y_calibration[calibration_diagnostic_mask])
                                if diagnostic_calibration else None),
                    "nll_scaled": (_nll(scaled_calibration[calibration_diagnostic_mask], y_calibration[calibration_diagnostic_mask])
                                   if diagnostic_calibration else None),
                    "operating_threshold": operating_threshold,
                    "threshold_feasible": scorer.threshold_feasible,
                    "calibration": {
                        "n": len(calibration_rows), "protocol": args.calibration_protocol,
                        "target_conditional_risk": EM.TARGET_DANGEROUS_ALLOW,
                        "min_coverage": EM.MIN_COVERAGE,
                        "selected_point": feasible_point,
                        "selection": calibration["selection"], "audit": calibration["audit"],
                        "metrics": eval_metrics.get("calibration"), "metrics_scope": calibration_metric_scope,
                        "nll_rows": len(diagnostic_calibration),
                    },
                    "eval": {name: value for name, value in eval_metrics.items() if name != "calibration"},
                    "weights": {"kind": weights, "sha256": meta["weights_sha256"]},
                    "parity": parity,
                    "memory_peak_mb": memory_peak_mb, "memory_stages": memory_stages,
                    "seconds": round(time.perf_counter() - started, 3),
                })
                summary = " ".join(f"{name}:F1={value['macro_f1']:.3f}"
                                   for name, value in eval_metrics.items())
                print(f"    {kind}: threshold={operating_threshold} "
                      f"feasible={scorer.threshold_feasible} T={scorer.temperature:.3f} "
                      f"NLL {runs[-1]['nll_raw']}->{runs[-1]['nll_scaled']} "
                      f"train_eval_acc={train_eval['accuracy']:.3f} "
                      f"peak_mb={memory_peak_mb} {summary}", flush=True)
                del scorer, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    document = {
        "schema": TRAINING_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "planned_manifest": "planned_manifest.json",
        "classes": list(FEAT.CLASSES),
        "safety": {
            "allowing_classes": list(EM.ALLOWING_CLASSES),
            "blocking_classes": list(EM.BLOCKING_CLASSES),
            "contextual_requires_trigger": EM.CONTEXTUAL_REQUIRES_TRIGGER,
            "target_conditional_risk": EM.TARGET_DANGEROUS_ALLOW,
            "min_coverage": EM.MIN_COVERAGE,
            "calibration_used_for": ["temperature", "operating_threshold"],
            "eval_only_splits": ["heldout", "ood", "challenge"],
        },
        "semantics": semantics,
        "data": {
            "data": str(data_path),
            "challenge": str(challenge_path) if challenge_path else None,
            "splits": {name: len(split_rows) for name, split_rows in by_split.items()},
            "pair_rows": {**{name: len(split_rows) for name, split_rows in pair_by_split.items()},
                          "challenge": len(challenge_rows)},
            "excluded_rows": {**excluded_by_split, "challenge": dict(challenge_excluded)},
            "exclusion_note": "set 形状与 existing>1 的整集行不进入配对训练/配对指标；set 级评测在 adapter.evaluate_sets",
            "train_label_origins": dict(Counter(row.get("label_origin", "unknown")
                                                for row in train_rows)),
            "group_overlaps": overlaps, "root_overlaps": prepared_data["root_overlaps"],
            "group_id_enforced": list(SPLITS) + ["challenge"],
            "bootstrap_cluster_field": "group_id",
            "source_sha256": manifest["source_sha256"],
        },
        "truncation": truncation,
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_used": bf16, "bf16_requested": args.bf16,
            "backbone": manifest["model"],
        },
        "training_input_contract": input_contract,
        "args": {
            "models": models, "sizes": sizes, "seeds": seeds, "epochs": epochs,
            "batch_size": batch_size, "grad_accum": grad_accum, "max_length": max_length,
            "device": str(device), "smoke": bool(args.smoke),
            "bootstrap_samples": args.bootstrap_samples, "threads": int(args.threads), **training_options,
            "eval_splits": list(prepared_data["eval_splits"]), "calibration_protocol": args.calibration_protocol,
            "calibration_stage": args.calibration_stage,
        },
        "runs": runs,
        "notes": [
            "合成/教师标签不是真人金标；旧challenge已曝光，不得声称盲测或生产安全证明。",
            "每个update与dev评估增量落training/*.jsonl，失败保留已执行预算；成功后保存恢复的选中权重。",
            "partitioned candidate拒绝audit且只用prob/policy：audit deferred、顶层threshold null；唯一dev候选经train --mode lock/audit独立一次审计。",
            "调参none要求独立train/dev/calibration文件；不打开独立终测；parity只用dev或train。",
            "指标仅为该研究实验的离线结果，不代表生产路由能力，也不得与非同类口径混用。",
            "预注册：参数/种子/数据 sha256/开发分区血缘在 fit 前落盘 planned_manifest.json；权重只按dev选择，"
            "partitioned温度只用prob、候选点只用policy；heldout/OOD/challenge只最终评测。",
            "工件 meta 顶层携带 threshold/threshold_feasible（candidate未审计固定null/false）；"
            "本模块提供 is_finetune_artifact 与 load_scorer，保持公共loader分派合同。",
            "未选到可行阈值时 operating_threshold 为 null，不声明已选操作策略（τ=0 只在 diagnostic_tau0 作未校准诊断）。",
            "截断按 joint 序列真实字符边界 + tokenizer offset（slow 回退前缀边界）归属 role；"
            "同一 role 多 block 每行至多计一次（counting_unit=row）；无法精确时标注 unavailable；不假称输入完整。",
            "train_history 是记录区间内在线（含 dropout）值；train_eval 是恢复后的选中模型在同一nested sample上的eval模式值，两者不可混用。",
            "语义边界：v1 是 SentenceTransformer mean-pool 句向量口径；v2 两臂是 384 + CLS joint 输入——"
            "只比较 v2 内两臂解释冻结效应，v1→v2 差异不能唯一归因 finetune。",
            "请求 GPU 失败或显存不足如实报错，绝不静默回退 CPU 冒充 GPU。",
            "零分母指标记 null；bootstrap 零分母复制记 invalid；零错误时读单侧稀有事件上界，不得把 percentile CI 的 [0,0] 当安全证据。",
            "predictions.jsonl 概率四舍五入到 6 位；verdict 按未舍入概率计算；verdict_tau0 是未校准诊断，不是运行点。",
        ],
    }
    if args.smoke:
        document["notes"].append("本次为 --smoke 测试模式（epochs≤1 / bootstrap≤50），不是正式实验。")
    (out_dir / "metrics.json").write_text(json.dumps(document, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / "curve.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in curve_rows), encoding="utf-8")
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8")
    print(f"-> {out_dir}（{len(runs)} runs）", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
