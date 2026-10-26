"""原子关系输出头：direct3 vs fine5 的离线审计、真实 CUDA 拟合与重放。

研究属性：repo 外内部 RA 实验。repo、旧实验与生产模型全部只读；本脚本只写
``EXP/runs``、``EXP/audits`` 下的运行产物，不部署、不改阈值、不改 repo。

CLI
---
    python train_relation_heads.py                       # 默认离线 selftest（不建模型）
    python train_relation_heads.py selftest
    python train_relation_heads.py audit [--allow-transformed-visible] [--no-heldout-integrity]
    python train_relation_heads.py smoke --arm direct3|fine5|all [--seed 13|42] [--device cuda:0]
    python train_relation_heads.py train --arm direct3|fine5 --seed 13|42 [--resume] [--device cuda:0]
    python train_relation_heads.py replay --run <arm>-seed<seed> [--checkpoint best-dev|terminal]
                                         --split train|dev|calibration|heldout [--unlock-final runs/final-lock.json]

数据与预算（预注册，不允许命令行放宽）
------------------------------------
* 输入 ``data/processed/{train,dev,calibration,heldout}.jsonl``，行形状见 ``ROW_KEYS``。
* 两臂共享 Qwen3-0.6B、LoRA(r16/alpha32/dropout0) FP32、BF16 冻结基座、head FP32、
  lr 1e-4 / wd .01 / clip 1.0 / micro 2 x accum 16 / max_length 768 / 300 updates，
  seed 13 与 42、numpy 数据顺序由 seed 决定；本脚本不提供改动这些常量的开关。
* 每 50 updates 同时评 train 与 dev 的**共同 coarse macro-F1**；dev-best 平局取最早。
  terminal 与 dev-best 分存 adapter+head，不保存 Adam、不复制基座。

heldout 访问纪律
--------------
* ``read_split("heldout", purpose=...)`` 是唯一入口。``purpose="train"`` 永远拒绝；
  ``purpose="locked-final"`` 需要主 Agent 手工创建的 ``runs/final-lock.json``，并当场
  重算 heldout 文件 SHA 与锁内值比对；``purpose="audit"`` 只用于 ``audit`` 的跨 split
  泄漏/契约完整性检查（不建模型、不算任何指标、不落任何行文本）。
* ``train`` 只解析 train/dev 行；heldout 只以字节 SHA 形式进入 run-lock（不解析行）。
* 本脚本从不创建 final-lock.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

EXP = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5] / 'source'
FIT_DIR = EXP.parents[1] / "固定数据拟合与基座对照-2026-10-24"

for _extra in (str(REPO / "src"), str(REPO), str(FIT_DIR)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import numpy as np              
import torch              

import fit_common_20261024 as C              
from research.memory_relations import finetune as FT              
from research.memory_relations import metrics as EM              

PROCESSED = EXP / "data" / "processed"
RUNS = EXP / "runs"
AUDITS = EXP / "audits"
SPLITS = ("train", "dev", "calibration", "heldout")

FINE = tuple(C.SR.SEMANTIC_RELATIONS)
COARSE = ("equivalent", "compatible", "conflict")
FINE_TO_COARSE = {
    "equivalent": "equivalent",
    "old_entails_new_strict": "compatible",
    "new_entails_old_strict": "compatible",
    "overlap_incomparable": "compatible",
    "disjoint": "conflict",
}
ARMS = {"direct3": 3, "fine5": 5}
SEEDS = (13, 42)

FINE_EQUIVALENT = FINE.index("equivalent")
FINE_CONFLICT = FINE.index("disjoint")
FINE_STRICT = (FINE.index("old_entails_new_strict"), FINE.index("new_entails_old_strict"))
COARSE_EQUIVALENT = COARSE.index("equivalent")
COARSE_CONFLICT = COARSE.index("conflict")

ROW_KEYS = ("id", "root_id", "cluster_id", "family_id", "split", "label", "coarse_label",
            "label_origin", "source", "author", "pair", "visible_text", "phenomena")

QWEN_SPEC = dict(C.PROTOCOL["qwen3"])
UPDATES = 300
MICRO = QWEN_SPEC["micro"]
ACCUM = QWEN_SPEC["accum"]
MAX_LENGTH = 768
EVAL_EVERY = 50
CLIP_NORM = QWEN_SPEC["clip_norm"]
assert (MICRO, ACCUM, MAX_LENGTH, UPDATES, EVAL_EVERY) == (2, 16, 768, 300, 50), "protocol drift"
assert (QWEN_SPEC["lr"], QWEN_SPEC["weight_decay"], CLIP_NORM) == (1e-4, 0.01, 1.0), "protocol drift"
assert (QWEN_SPEC["lora_r"], QWEN_SPEC["lora_alpha"], QWEN_SPEC["lora_dropout"]) == (16, 32, 0.0), "protocol drift"
assert FINE == ("equivalent", "old_entails_new_strict", "new_entails_old_strict", "disjoint",
                "overlap_incomparable"), "five-leaf order drift"

PROTOCOL = {
    "study": "atomic relation cross-model replication and Qwen head training 2026-10-25",
    "arms": {name: {"num_labels": labels, "targets": ("coarse_label" if name == "direct3" else "label")}
             for name, labels in ARMS.items()},
    "seeds": list(SEEDS),
    "updates": UPDATES, "micro": MICRO, "accum": ACCUM, "max_length": MAX_LENGTH,
    "lr": QWEN_SPEC["lr"], "weight_decay": QWEN_SPEC["weight_decay"], "clip_norm": CLIP_NORM,
    "lora": {"r": QWEN_SPEC["lora_r"], "alpha": QWEN_SPEC["lora_alpha"],
             "dropout": QWEN_SPEC["lora_dropout"], "target_modules": list(QWEN_SPEC["target_modules"])},
    "base": {"repo_id": C.QWEN["repo_id"], "revision": C.QWEN["revision"], "dtype": "bfloat16",
             "pool": QWEN_SPEC["pool"], "gradient_checkpointing": QWEN_SPEC["gradient_checkpointing"]},
    "eval_every": EVAL_EVERY, "selection": "coarse macro-F1 on dev, ties keep the earliest",
    "fine_to_coarse": dict(FINE_TO_COARSE),
}


class HeldoutLockError(RuntimeError):
    """heldout 读取被纪律拒绝（缺锁、锁过期、用途不符）。"""


@contextmanager
def anchor_audit_sink(directory: Path):
    """``C.parameter_audit`` 会写 ``C.AUDITS``（旧实验目录）；这里把它导到本实验目录。"""
    directory.mkdir(parents=True, exist_ok=True)
    previous = C.AUDITS
    C.AUDITS = Path(directory)
    try:
        yield Path(directory)
    finally:
        C.AUDITS = previous


                                                                                        
              
                                                                                        

def read_json(path: "str | Path") -> Any:
    return C.read_json(path)


def dump(path: "str | Path", value: Any) -> None:
    C.dump(path, value)


def append_jsonl(path: "str | Path", value: dict) -> None:
    C.append_jsonl(path, value)


def write_jsonl(path: "str | Path", rows: Sequence[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: "str | Path") -> list[dict]:
    return C.read_jsonl(path)


def sha256_file(path: "str | Path") -> str:
    return C.sha256_file(path)


def optional_sha(path: "str | Path") -> "str | None":
    path = Path(path)
    return C.sha256_file(path) if path.is_file() else None


def state_sha256(state: dict) -> str:
    return C.state_sha256(state)


def environment() -> dict:
    info = dict(C.environment())
    info["script_sha256"] = sha256_file(Path(__file__))
    return info


                                                                                        
                        
                                                                                        

def split_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return PROCESSED / f"{split}.jsonl"


def read_split(split: str, *, purpose: str, unlock: "dict | None" = None,
               require_exists: bool = True) -> list[dict]:
    """唯一 split 入口；heldout 的用途与锁在这里集中裁决。"""
    if purpose not in ("train", "audit", "locked-final"):
        raise ValueError(f"unknown purpose {purpose!r}")
    path = split_path(split)
    if split == "heldout":
        if purpose == "train":
            raise HeldoutLockError(
                "heldout 未锁定：train/smoke/metrics/calibrate 一律不得读取 heldout 行；"
                "只允许 runs/final-lock.json 通过后的 final-eval/replay")
        if purpose == "locked-final":
            if not isinstance(unlock, dict):
                raise HeldoutLockError("heldout 解锁需要已校验的 final-lock")
            expected = (unlock.get("data_sha256") or {}).get("heldout")
            if not isinstance(expected, str) or len(expected) != 64:
                raise HeldoutLockError("final-lock 缺少 heldout 的 data_sha256")
            if not path.is_file():
                raise FileNotFoundError(f"missing split file: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise HeldoutLockError(
                    f"heldout SHA 与 final-lock 不符：锁 {expected[:12]} 盘上 {actual[:12]}")
        elif purpose != "audit":
            raise HeldoutLockError(f"heldout 不允许 purpose={purpose!r}")
    if not path.is_file():
        if require_exists:
            raise FileNotFoundError(
                f"missing split file: {path}（data/processed 由 build_relation_data.py 生成）")
        return []
    return read_jsonl(path)


def side_text(side: Any) -> str:
    if isinstance(side, str):
        return side
    if isinstance(side, dict) and isinstance(side.get("text"), str):
        return side["text"]
    raise TypeError(f"pair side must be str or {{'text': str}}, got {type(side).__name__}")


def pair_sides(pair: Any) -> tuple[str, str]:
    if not isinstance(pair, dict) or "old" not in pair or "new" not in pair:
        raise TypeError("pair must be an object with old/new")
    return side_text(pair["old"]), side_text(pair["new"])


def normalize(text: str) -> str:
    return " ".join(text.split())


def pair_key(pair: Any) -> str:
    old, new = pair_sides(pair)
    payload = json.dumps({"old": normalize(old), "new": normalize(new)},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def label_index(row: dict, arm: str) -> int:
    return COARSE.index(row["coarse_label"]) if arm == "direct3" else FINE.index(row["label"])


def targets_for(rows: Sequence[dict], arm: str) -> list[int]:
    return [label_index(row, arm) for row in rows]


def gold_indices(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fine = np.asarray([FINE.index(row["label"]) for row in rows], dtype=int)
    coarse = np.asarray([COARSE.index(row["coarse_label"]) for row in rows], dtype=int)
    strict = np.isin(fine, list(FINE_STRICT))
    return fine, coarse, strict


def validate_row(row: Any, split: str) -> list[str]:
    """行契约：字段齐全、类型正确、fine/coarse 映射一致、pair 与 visible_text 完整。"""
    if not isinstance(row, dict):
        return ["row is not an object"]
    errors = []
    missing = [key for key in ROW_KEYS if key not in row]
    if missing:
        errors.append(f"missing keys {missing}")
        return errors
    for key in ("id", "root_id", "cluster_id", "label_origin", "source"):
        if not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"{key} must be a non-empty string")
    for key in ("family_id", "author"):
        if row[key] is not None and (not isinstance(row[key], str) or not row[key].strip()):
            errors.append(f"{key} must be null or a non-empty string (legacy rows carry null)")
    if row["split"] != split:
        errors.append(f"split field {row['split']!r} != file split {split!r}")
    if row["label"] not in FINE_TO_COARSE:
        errors.append(f"label {row['label']!r} not in five-leaf space")
    if row["coarse_label"] not in COARSE:
        errors.append(f"coarse_label {row['coarse_label']!r} not in {COARSE}")
    elif row["label"] in FINE_TO_COARSE and FINE_TO_COARSE[row["label"]] != row["coarse_label"]:
        errors.append(f"coarse_label {row['coarse_label']!r} != map[{row['label']!r}]")
    if not isinstance(row["phenomena"], (list, dict)):
        errors.append("phenomena must be a list or an object")
    try:
        old, new = pair_sides(row["pair"])
    except TypeError as error:
        errors.append(str(error))
        old = new = ""
    if not old.strip() or not new.strip():
        errors.append("pair sides must be non-empty")
    if not isinstance(row["visible_text"], str) or not row["visible_text"].strip():
        errors.append("visible_text must be a non-empty string")
    return errors


def visible_text_contains_sides(row: dict) -> bool:
    try:
        old, new = pair_sides(row["pair"])
    except TypeError:
        return False
    visible = normalize(row["visible_text"])
    return normalize(old) in visible and normalize(new) in visible


def duplicate_label_audit(groups: dict[str, list[dict]], *, allow_transformed: bool) -> dict:
    """跨 split 泄漏与同文异标：只回计数与少量 id 举例，不回行文本。"""
    text_splits: dict[str, set[str]] = {}
    text_labels: dict[str, set[str]] = {}
    text_ids: dict[str, list[str]] = {}
    pair_splits: dict[str, set[str]] = {}
    pair_ids: dict[str, list[str]] = {}
    fields = {"root_id": {}, "family_id": {}, "cluster_id": {}}
    ids: dict[str, list[str]] = {}
    transformed: list[dict] = []
    skipped = 0
    required = ("id", "label", "visible_text", "pair", "root_id", "family_id", "cluster_id")
    for split, rows in groups.items():
        for row in rows:
            if not isinstance(row, dict) or any(key not in row for key in required):
                skipped += 1
                continue
            key = text_key(row["visible_text"])
            text_splits.setdefault(key, set()).add(split)
            text_labels.setdefault(key, set()).add(row["label"])
            text_ids.setdefault(key, []).append(f"{split}:{row['id']}")
            row_pair_key = pair_key(row["pair"])
            pair_splits.setdefault(row_pair_key, set()).add(split)
            pair_ids.setdefault(row_pair_key, []).append(f"{split}:{row['id']}")
            for name in fields:
                value = row[name]
                if isinstance(value, str) and value.strip():
                    fields[name].setdefault(value, set()).add(split)
            ids.setdefault(row["id"], []).append(split)
            if not visible_text_contains_sides(row):
                transformed.append({"id": row["id"], "split": split})
    conflicts = {key: sorted(values) for key, values in text_labels.items() if len(values) > 1}
    cross_text = {key: sorted(value) for key, value in text_splits.items() if len(value) > 1}
    cross_pair = {key: sorted(value) for key, value in pair_splits.items() if len(value) > 1}
    cross_fields = {name: {key: sorted(value) for key, value in table.items() if len(value) > 1}
                    for name, table in fields.items()}
    duplicate_ids = {key: value for key, value in ids.items() if len(value) > 1}
    report = {
        "groups": {name: len(rows) for name, rows in groups.items()},
        "skipped_malformed_rows": skipped,
        "identical_text_conflicting_labels": len(conflicts),
        "identical_text_across_splits": len(cross_text),
        "identical_pair_across_splits": len(cross_pair),
        "duplicate_ids": len(duplicate_ids),
        "cross_split": {name: len(table) for name, table in cross_fields.items()},
        "visible_text_missing_side": len(transformed),
        "examples": {
            "text_conflict": [{"text_sha": key[:12], "labels": value, "rows": text_ids[key][:4]}
                              for key, value in list(conflicts.items())[:3]],
            "cross_split_text": [{"text_sha": key[:12], "splits": value, "rows": text_ids[key][:4]}
                                 for key, value in list(cross_text.items())[:3]],
            "cross_split_pair": [{"pair_sha": key[:12], "splits": value, "rows": pair_ids[key][:4]}
                                 for key, value in list(cross_pair.items())[:3]],
            "cross_split_fields": {name: [{"value": key, "splits": value}
                                          for key, value in list(table.items())[:3]]
                                   for name, table in cross_fields.items()},
            "duplicate_ids": list(duplicate_ids.items())[:3],
            "visible_text_missing_side": transformed[:3],
        },
        "transformed_visible_text_allowed": bool(allow_transformed),
    }
    report["passed"] = bool(
        not report["identical_text_conflicting_labels"]
        and not report["identical_text_across_splits"]
        and not report["identical_pair_across_splits"]
        and not report["duplicate_ids"]
        and not any(report["cross_split"].values())
        and (allow_transformed or not report["visible_text_missing_side"]))
    return report


                                                                                        
                               
                                                                                        

LOCK_SCHEMA = "relation-final-lock/1"


def checkpoint_file_shas(directory: "str | Path") -> dict:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {directory}")
    files = {item.relative_to(directory).as_posix(): sha256_file(item)
             for item in sorted(directory.rglob("*")) if item.is_file()}
    if not files:
        raise FileNotFoundError(f"empty checkpoint directory: {directory}")
    return files


def validate_final_lock(path: "str | Path", *, run: "str | None" = None,
                        checkpoint: "str | None" = None,
                        require_checkpoint_files: bool = True) -> dict:
    """校验主 Agent 的 final-lock；本函数只读，绝不创建或补写锁。"""
    path = Path(path)
    if not path.is_file():
        raise HeldoutLockError(
            f"final-lock 不存在：{path}。该文件只由主 Agent 在看完 dev 后手工创建，"
            f"脚本不得自行生成（schema={LOCK_SCHEMA}）")
    lock = read_json(path)
    if not isinstance(lock, dict):
        raise HeldoutLockError("final-lock 不是对象")
    if lock.get("schema") != LOCK_SCHEMA:
        raise HeldoutLockError(f"final-lock schema 必须是 {LOCK_SCHEMA}")
    for key in ("run", "checkpoint", "arm", "seed", "data_sha256", "checkpoint_files_sha256"):
        if key not in lock:
            raise HeldoutLockError(f"final-lock 缺少字段 {key}")
    if lock["arm"] not in ARMS:
        raise HeldoutLockError(f"final-lock arm 非法：{lock['arm']!r}")
    if int(lock["seed"]) not in SEEDS:
        raise HeldoutLockError(f"final-lock seed 非法：{lock['seed']!r}")
    data = lock["data_sha256"]
    if not isinstance(data, dict) or any(not isinstance(data.get(split), str)
                                         or len(data.get(split, "")) != 64 for split in SPLITS):
        raise HeldoutLockError(f"final-lock data_sha256 必须含 {list(SPLITS)} 的 64 位 SHA")
    for split in SPLITS:
        actual = optional_sha(split_path(split))
        if actual != data[split]:
            raise HeldoutLockError(
                f"final-lock 数据 SHA 与盘上 {split} 不符：锁 {data[split][:12]} 盘上 "
                f"{str(actual)[:12]}")
    if run is not None and lock["run"] != run:
        raise HeldoutLockError(f"final-lock 属于 run={lock['run']}，不是 {run}")
    if checkpoint is not None and lock["checkpoint"] != checkpoint:
        raise HeldoutLockError(f"final-lock 锁定 checkpoint={lock['checkpoint']}，不是 {checkpoint}")
    if require_checkpoint_files:
        run_dir = RUNS / lock["run"]
        lock_sha = optional_sha(run_dir / "run-lock.json")
        if lock.get("train_run_lock_sha256") and lock_sha != lock["train_run_lock_sha256"]:
            raise HeldoutLockError("final-lock 的 train_run_lock_sha256 与盘上 run-lock.json 不符")
        actual_files = checkpoint_file_shas(run_dir / "checkpoints" / lock["checkpoint"])
        if actual_files != lock["checkpoint_files_sha256"]:
            raise HeldoutLockError("final-lock 的 checkpoint 文件 SHA 与盘上 checkpoint 不符")
    return lock


def lock_template(run: str, checkpoint: str) -> dict:
    """打印给主 Agent 的锁模板；只返回内容，不落盘。"""
    run_dir = RUNS / run
    lock = read_json(run_dir / "run-lock.json")
    result = read_json(run_dir / "result.json")
    return {
        "schema": LOCK_SCHEMA,
        "created": "<YYYY-MM-DDTHH:MM:SS>",
        "run": run,
        "checkpoint": checkpoint,
        "arm": lock["identity"]["config"]["arm"],
        "seed": lock["identity"]["config"]["seed"],
        "selected_step": result["selection"]["selected_step"],
        "train_run_lock_sha256": sha256_file(run_dir / "run-lock.json"),
        "checkpoint_files_sha256": checkpoint_file_shas(run_dir / "checkpoints" / checkpoint),
        "data_sha256": {split: sha256_file(split_path(split)) for split in SPLITS},
        "note": "hand-created by the main agent after dev review; scripts never create this file",
    }


                                                                                        
                           
                                                                                        

def lora_state(model: torch.nn.Module) -> dict:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_" in name}


def build_arm(arm: str, seed: int, *, builder: "Any | None" = None) -> dict:
    """Qwen3-0.6B + LoRA(r16/alpha32) + head(3 or 5)。

    两臂 LoRA 权重来自同一个 ``build_qwen(seed)``：fine5 只替换 head（5 类），
    因此同 seed 的 LoRA 初始化 SHA 相同；替换后必须重建 optimizer（见 fit_arm）。
    """
    if arm not in ARMS:
        raise ValueError(arm)
    build = builder or C.build_qwen
    model, tokenizer = build(seed)
    initial = {
        "classifier_in_features": int(model.classifier.in_features),
        "classifier_out_features": int(model.classifier.out_features),
        "classifier_weight_shape": list(model.classifier.weight.shape),
        "classifier_bias_shape": list(model.classifier.bias.shape),
    }
    lora_init = state_sha256(lora_state(model)) if lora_state(model) else None
    replaced = False
    if int(model.classifier.out_features) != ARMS[arm]:
        torch.manual_seed(int(seed))
        head = torch.nn.Linear(int(model.classifier.in_features), ARMS[arm])
        with torch.no_grad():
            head.weight.normal_(mean=0.0, std=0.02)
            head.bias.zero_()
        model.classifier = head
        replaced = True
    if int(model.classifier.out_features) != ARMS[arm]:
        raise RuntimeError(f"{arm}: classifier out_features {model.classifier.out_features} != {ARMS[arm]}")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "arm": arm,
        "seed": int(seed),
        "init": {
            **initial,
            "classifier_replaced": replaced,
            "classifier_init": ("seeded normal(std=0.02) + zero bias" if replaced
                                else "PooledClassifier(seed) normal(std=0.02) + zero bias"),
            "final_out_features": int(model.classifier.out_features),
            "final_weight_shape": list(model.classifier.weight.shape),
            "final_bias_shape": list(model.classifier.bias.shape),
            "head_state_sha256": state_sha256(model.classifier.state_dict()),
            "lora_init_sha256": lora_init,
        },
    }


def frozen_base_state(model: torch.nn.Module) -> dict:
    return {name: param.detach().cpu() for name, param in model.named_parameters()
            if "lora_" not in name and not name.startswith("classifier.")}


def frozen_base_sha256(model: torch.nn.Module) -> str:
    return state_sha256(frozen_base_state(model))


def head_buffers(model: torch.nn.Module) -> dict:
    return {name: value.detach().cpu() for name, value in model.named_buffers()}


def save_checkpoint(directory: Path, model: torch.nn.Module, head_state: dict,
                    meta: dict) -> dict:
    """Qwen：adapter + head 两件套（不保存 Adam、不复制基座）；selftest 用 fake-full。"""
    directory.mkdir(parents=True, exist_ok=True)
    if meta.get("format") == "qwen-lora":
        return C.save_qwen_checkpoint(directory, model, head_state,
                                      dict(meta, head_state_buffers=sorted(head_buffers(model))))
    torch.save({"state_dict": model.state_dict(), "meta": meta}, directory / "weights.pt")
    files = {item.relative_to(directory).as_posix(): {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
             for item in sorted(directory.rglob("*")) if item.is_file()}
    return {"directory": str(directory), "files": files,
            "bytes": sum(item["bytes"] for item in files.values())}


def load_checkpoint(model: torch.nn.Module, directory: "str | Path") -> dict:
    """严格重载：adapter 权重 SHA / 全量 state SHA 必须与保存时一致，否则拒绝出分。"""
    directory = Path(directory)
    head_path, weights_path = directory / "head.pt", directory / "weights.pt"
    if head_path.is_file():
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta") or {}
        if meta.get("format") != "qwen-lora":
            raise RuntimeError(f"{directory}: head.pt is not a qwen-lora checkpoint")
        if int(model.classifier.out_features) != int(meta["num_labels"]):
            raise RuntimeError(f"head shape mismatch: model {model.classifier.out_features} "
                               f"vs checkpoint {meta['num_labels']}")
        from peft import PeftModel
        adapter = directory / "adapter"
        if not adapter.is_dir():
            raise FileNotFoundError(f"missing adapter directory: {adapter}")
        base = model.backbone.get_base_model() if hasattr(model.backbone, "get_base_model") else model.backbone
        loaded = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        for name, parameter in loaded.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        model.backbone = loaded.to(model.classifier.weight.device)
        model.classifier.load_state_dict(payload["classifier"], strict=True)
        actual_lora = state_sha256(lora_state(model))
        if meta.get("adapter_lora_sha256") and actual_lora != meta["adapter_lora_sha256"]:
            raise RuntimeError("reloaded LoRA weights do not match the checkpoint SHA")
        actual_head = state_sha256(model.classifier.state_dict())
        if meta.get("head_sha256") and actual_head != meta["head_sha256"]:
            raise RuntimeError("reloaded head weights do not match the checkpoint SHA")
        return {"format": "qwen-lora", "meta": meta, "adapter_lora_sha256": actual_lora,
                "head_sha256": actual_head}
    if weights_path.is_file():
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta") or {}
        model.load_state_dict(payload["state_dict"], strict=True)
        actual = state_sha256(model.state_dict())
        if meta.get("full_state_sha256") and actual != meta["full_state_sha256"]:
            raise RuntimeError("reloaded state does not match the checkpoint SHA")
        return {"format": "fake-full", "meta": meta, "full_state_sha256": actual}
    raise FileNotFoundError(f"no checkpoint payload under {directory}")


                                                                                        
                                                
                                                                                        

def rate(hits: int, n: int) -> dict:
    return {"value": (None if n == 0 else float(hits) / float(n)), "numerator": int(hits), "n": int(n)}


def metric_block(gold: np.ndarray, pred: np.ndarray, probabilities: np.ndarray,
                 classes: Sequence[str], *, equivalent: int, conflict: int,
                 strict_mask: np.ndarray) -> dict:
    """分类汇总 + 安全率；strict_mask 由 fine gold 决定（coarse 域也必须传 fine 派生的掩码）。"""
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(gold), len(classes)):
        raise ValueError(f"probabilities shape {probabilities.shape} != {(len(gold), len(classes))}")
    summary = EM.classification_summary(gold, pred, classes=list(classes))
    n = int(len(gold))
    rows = np.arange(n)
    loss = float(-np.log(np.clip(probabilities[rows, gold], 1e-30, 1.0)).mean()) if n else None
    non_equivalent = gold != equivalent
    conflicts = gold == conflict
    summary.update({
        "classes": list(classes),
        "correct": int((gold == pred).sum()),
        "loss": loss,
        "row_loss": loss,
        "conflict_recall": rate(int((pred[conflicts] == conflict).sum()), int(conflicts.sum())),
        "false_equivalence": rate(int((pred[non_equivalent] == equivalent).sum()),
                                  int(non_equivalent.sum())),
        "strict_subset_redundancy_accept": rate(int((pred[strict_mask] == equivalent).sum()),
                                                int(strict_mask.sum())),
        "definitions": {
            "conflict_recall": "pred == conflict among gold conflict（分母=gold conflict 行数）",
            "false_equivalence": "pred == equivalent among gold != equivalent（原 gold 类别的误等价接受）",
            "strict_subset_redundancy_accept": (
                "pred == equivalent among gold 五叶严格单向包含（old_entails_new_strict / "
                "new_entails_old_strict）：把严格子集当完全冗余接受的比率"),
        },
        "n_classes": len(classes),
    })
    return summary


def coarse_probabilities(probabilities: np.ndarray, arm: str) -> np.ndarray:
    """fine5 五叶概率按固定映射聚合 coarse；direct3 原样返回。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if arm == "direct3":
        return probabilities
    if probabilities.ndim != 2 or probabilities.shape[1] != len(FINE):
        raise ValueError(f"fine5 probabilities shape {probabilities.shape} invalid")
    columns = [probabilities[:, FINE_EQUIVALENT],
               probabilities[:, FINE.index("old_entails_new_strict")]
               + probabilities[:, FINE.index("new_entails_old_strict")]
               + probabilities[:, FINE.index("overlap_incomparable")],
               probabilities[:, FINE_CONFLICT]]
    return np.stack(columns, axis=1)


def forward_probabilities(model, tokenizer, rows: Sequence[dict], arm: str, device: torch.device,
                          max_length: int, bf16: bool, encode, batch_size: int = 16) -> np.ndarray:
    model.eval()
    collected: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            chunk = np.arange(start, stop)
            encoded = encode(chunk, rows)
            with FT.autocast_for(device, bf16):
                logits = model(**encoded)
            if logits.ndim != 2 or logits.shape != (len(chunk), ARMS[arm]):
                raise RuntimeError(f"logits shape {tuple(logits.shape)} != {(len(chunk), ARMS[arm])}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits")
            collected.extend(logits.float().softmax(dim=-1).cpu().tolist())
    return np.asarray(collected, dtype=np.float64).reshape(len(rows), ARMS[arm])


def evaluate_split(*, model, tokenizer, rows: Sequence[dict], arm: str, split: str,
                   device: torch.device, max_length: int, bf16: bool, encode,
                   batch_size: int = 16) -> dict:
    """一次前向 → coarse 指标（+ fine5 的 fine 指标）与逐题概率/argmax。"""
    probabilities = forward_probabilities(model, tokenizer, rows, arm, device, max_length, bf16,
                                          encode, batch_size)
    fine_gold, coarse_gold, strict_mask = gold_indices(rows)
    coarse = coarse_probabilities(probabilities, arm)
    coarse_pred = coarse.argmax(axis=1)
    summary = {
        "split": split,
        "arm": arm,
        "n": len(rows),
        "coarse": metric_block(coarse_gold, coarse_pred, coarse, COARSE,
                               equivalent=COARSE_EQUIVALENT, conflict=COARSE_CONFLICT,
                               strict_mask=strict_mask),
        "fine": None,
        "fine_classes": list(FINE) if arm == "fine5" else None,
    }
    fine_pred = probabilities.argmax(axis=1)
    if arm == "fine5":
        summary["fine"] = metric_block(fine_gold, fine_pred, probabilities, FINE,
                                       equivalent=FINE_EQUIVALENT, conflict=FINE_CONFLICT,
                                       strict_mask=strict_mask)
    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "id": row["id"], "split": split,
            "family_id": row["family_id"], "cluster_id": row["cluster_id"],
            "root_id": row["root_id"], "source": row["source"], "author": row["author"],
            "label_origin": row["label_origin"],
            "gold_fine": row["label"], "gold_coarse": row["coarse_label"],
            "gold_fine_index": int(fine_gold[index]), "gold_coarse_index": int(coarse_gold[index]),
            "pred_coarse": COARSE[int(coarse_pred[index])],
            "pred_coarse_index": int(coarse_pred[index]),
            "pred_fine": FINE[int(fine_pred[index])] if arm == "fine5" else None,
            "pred_fine_index": int(fine_pred[index]) if arm == "fine5" else None,
            "coarse_probabilities": [float(value) for value in coarse[index]],
            "fine_probabilities": ([float(value) for value in probabilities[index]]
                                   if arm == "fine5" else None),
            "max_coarse_probability": float(coarse[index].max()),
            "max_fine_probability": float(probabilities[index].max()),
        })
    return {"summary": summary, "predictions": predictions,
            "coarse_probabilities": coarse, "fine_probabilities": probabilities}


def default_encode(tokenizer, device: torch.device, max_length: int):
    def encode(chunk, rows):
        return C.encode_batch_visible(tokenizer, [rows[int(index)]["visible_text"] for index in chunk],
                                      device, max_length)
    return encode


                                                                                        
                                          
                                                                                        

THRESHOLDS = [round(index / 100.0, 2) for index in range(100)]
CONFLICT_RECALL_FLOOR = 0.90
FALSE_EQUIVALENCE_CEILING = 0.05


def gate_row(coarse: np.ndarray, coarse_gold: np.ndarray, threshold: float) -> dict:
    """单阈值 risk-coverage 行；confidence 用 coarse 概率的 max（accept 或 fallback 给 LLM）。"""
    coarse = np.asarray(coarse, dtype=np.float64)
    gold = np.asarray(coarse_gold, dtype=int)
    pred = coarse.argmax(axis=1)
    accepted = coarse.max(axis=1) >= float(threshold)
    accepted_n = int(accepted.sum())
    hits = int((pred[accepted] == gold[accepted]).sum())
    conflicts = gold == COARSE_CONFLICT
    non_equivalent = gold != COARSE_EQUIVALENT
    accepted_conflict_gold = conflicts & accepted
    accepted_non_equivalent = non_equivalent & accepted
    return {
        "threshold": float(threshold),
        "accepted": accepted_n,
        "coverage": rate(accepted_n, len(gold)),
        "fallback": rate(len(gold) - accepted_n, len(gold)),
        "accepted_accuracy": rate(hits, accepted_n),
        "conflict_recall": rate(int((pred[accepted_conflict_gold] == COARSE_CONFLICT).sum()),
                                int(conflicts.sum())),
        "conflict_denominator": int(conflicts.sum()),
        "false_equivalence": rate(int((pred[accepted_non_equivalent] == COARSE_EQUIVALENT).sum()),
                                  int(non_equivalent.sum())),
        "non_equivalent_denominator": int(non_equivalent.sum()),
    }


def calibrate_gate(coarse: np.ndarray, coarse_gold: np.ndarray,
                   thresholds: Sequence[float] = tuple(THRESHOLDS)) -> dict:
    """calibration-only 研究性阈值：risk-coverage 全表 + 预注册唯一阈值规则。"""
    coarse = np.asarray(coarse, dtype=np.float64)
    gold = np.asarray(coarse_gold, dtype=int)
    table = [gate_row(coarse, gold, threshold) for threshold in thresholds]
    feasible = [row for row in table
                if row["accepted"] > 0
                and (row["conflict_recall"]["value"] or 0.0) >= CONFLICT_RECALL_FLOOR
                and (row["false_equivalence"]["value"] if row["false_equivalence"]["value"] is not None
                     else 1.0) <= FALSE_EQUIVALENCE_CEILING]
    selected = None
    if feasible:
        selected = sorted(feasible,
                          key=lambda row: (-row["accepted_accuracy"]["value"],
                                           -row["coverage"]["value"],
                                           -row["threshold"]))[0]
    return {
        "domain": "coarse",
        "n": int(len(gold)),
        "thresholds": table,
        "rule": {
            "constraint_conflict_recall_min": CONFLICT_RECALL_FLOOR,
            "constraint_false_equivalence_max": FALSE_EQUIVALENCE_CEILING,
            "objective": "max accepted accuracy",
            "tie_break": ["higher coverage", "higher threshold"],
            "accept_rule": "accept the local argmax when max_probability >= threshold, else fallback",
            "fit_on": "calibration only; heldout never used",
        },
        "feasible_thresholds": len(feasible),
        "selected": selected,
        "local_accept": selected is not None,
        "threshold": None if selected is None else selected["threshold"],
    }


                                                                                        
     
                                                                                        

def loss_target_selftest(loss_fn, targets: Sequence[int], rows: Sequence[dict], arm: str,
                         chunk: np.ndarray, device: torch.device) -> dict:
    """任意类数的 sum-loss 与 target 映射自检（C 版本硬编码 3 类，fine5 不能直接用）。"""
    count = len(chunk)
    width = ARMS[arm]
    selected = np.asarray(targets, dtype=int)[chunk]
    gold = torch.as_tensor(selected, device=device)
    perfect = torch.full((count, width), -10.0, device=device)
    perfect[torch.arange(count, device=device), gold] = 10.0
    wrong_index = torch.as_tensor((selected + 1) % width, device=device)
    wrong = torch.full((count, width), -10.0, device=device)
    wrong[torch.arange(count, device=device), wrong_index] = 10.0
    perfect_logits = perfect.clone().requires_grad_(True)
    wrong_logits = wrong.clone().requires_grad_(True)
    perfect_loss, perfect_hits = loss_fn(perfect_logits, chunk, chunk)
    wrong_loss, wrong_hits = loss_fn(wrong_logits, chunk, chunk)
    mean_loss = torch.nn.functional.cross_entropy(perfect_logits.float(), gold, reduction="mean")
    if not torch.allclose(perfect_loss, mean_loss * count, atol=1e-4):
        raise AssertionError("loss_fn is not a window sum")
    if float(perfect_hits) != count or float(wrong_hits) != 0:
        raise AssertionError("argmax hits mismatch")
    perfect_grad = torch.autograd.grad(perfect_loss, perfect_logits)[0]
    if not (torch.isfinite(perfect_grad).all() and float(perfect_grad.abs().sum()) > 0):
        raise AssertionError("perfect-logit gradient is degenerate")
    return {"arm": arm, "classes": width, "n": count,
            "perfect_ce": float(perfect_loss.detach()), "wrong_ce": float(wrong_loss.detach()),
            "perfect_hits": float(perfect_hits), "wrong_hits": float(wrong_hits),
            "sum_equals_mean_times_n": True, "gold_labels": [int(value) for value in selected[:8]],
            "gold_names": [COARSE[int(value)] if arm == "direct3" else FINE[int(value)]
                           for value in selected[:8]]}


def best_dev_from_curve(history: Sequence[dict]) -> "tuple[int, float] | None":
    """平局取最早：只有严格更大才替换。"""
    best = None
    for record in history:
        value = record.get("dev_macro_f1")
        if value is None or not math.isfinite(float(value)):
            continue
        if best is None or float(value) > best[1]:
            best = (int(record["step"]), float(value))
    return best


def fit_arm(*, model, tokenizer, train_rows: Sequence[dict], dev_rows: Sequence[dict],
            arm: str, seed: int, device: torch.device, bf16: bool, max_length: int,
            out_dir: Path, encode, updates: int = UPDATES, eval_every: int = EVAL_EVERY,
            frozen: bool = True, verbose: bool = True) -> dict:
    """固定 updates 拟合；每 eval_every 步同时评 train/dev 的 coarse macro-F1。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    updates_path, curve_path = out_dir / "updates.jsonl", out_dir / "curve.jsonl"
    for path in (updates_path, curve_path):
        path.unlink(missing_ok=True)
    targets = targets_for(train_rows, arm)
    loss_fn = C.make_loss(targets, device)
    optimizer = C.build_optimizer("qwen3", model)
    if not any(group["params"] for group in optimizer.param_groups):
        raise RuntimeError("optimizer has no parameter groups")
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    covered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not head_ids <= covered:
        raise RuntimeError("optimizer does not cover the classifier head")
    windows = FT.training_windows(len(train_rows), epochs=1, batch_size=MICRO, grad_accum=ACCUM,
                                  max_steps=int(updates), rng=np.random.default_rng(int(seed)))
    started = time.perf_counter()

    def evaluate_dev() -> dict:
        result = evaluate_split(model=model, tokenizer=tokenizer, rows=dev_rows, arm=arm, split="dev",
                                device=device, max_length=max_length, bf16=bf16, encode=encode)
        return result["summary"]["coarse"]

    def on_event(event: dict) -> None:
        if event["event"] == "update":
            append_jsonl(updates_path, dict(event, seconds=round(time.perf_counter() - started, 3)))
            return
        if event["event"] != "dev_evaluation":
            return
        eval_started = time.perf_counter()
        train_summary = evaluate_split(model=model, tokenizer=tokenizer, rows=train_rows, arm=arm,
                                       split="train", device=device, max_length=max_length,
                                       bf16=bf16, encode=encode)["summary"]
        dev_summary = evaluate_split(model=model, tokenizer=tokenizer, rows=dev_rows, arm=arm,
                                     split="dev", device=device, max_length=max_length,
                                     bf16=bf16, encode=encode)["summary"]
        entry = {
            "step": int(event["step"]), "epoch": event.get("epoch"),
            "interval_loss": event.get("loss"), "interval_accuracy": event.get("accuracy"),
            "train_coarse_macro_f1": train_summary["coarse"]["macro_f1"],
            "dev_coarse_macro_f1": dev_summary["coarse"]["macro_f1"],
            "train": train_summary, "dev": dev_summary,
            "eval_seconds": round(time.perf_counter() - eval_started, 3),
            "seconds": round(time.perf_counter() - started, 3),
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                          if device.type == "cuda" else 0),
        }
        append_jsonl(curve_path, entry)
        if verbose:
            print(json.dumps({"arm": arm, "seed": seed, "step": entry["step"],
                              "dev_coarse_macro_f1": entry["dev_coarse_macro_f1"],
                              "train_coarse_macro_f1": entry["train_coarse_macro_f1"]},
                             ensure_ascii=False), flush=True)

    fitted = FT.fit_joint(model, tokenizer, [""] * len(train_rows), optimizer=optimizer,
                          windows=windows, planned_steps=int(updates),
                          planned_instances=int(updates) * MICRO * ACCUM,
                          max_length=max_length, device=device, bf16=bf16, loss_fn=loss_fn,
                          evaluate=evaluate_dev, checkpoint_selection="best-dev",
                          selection_metric="dev-macro-f1", patience=0, min_delta=0.0,
                          eval_every_steps=int(eval_every), frozen=frozen,
                          require_backbone_grad=True, event_callback=on_event,
                          clip_grad_norm=CLIP_NORM,
                          encode_fn=lambda chunk: encode(chunk, train_rows))
    best = best_dev_from_curve(fitted["history"])
    selection = fitted["selection"]
    if best is None:
        raise RuntimeError("no dev evaluation happened; cannot select a checkpoint")
    if (int(selection["selected_step"]), float(selection["best_dev_value"])) != best:
        raise RuntimeError(f"dev-best tie handling drifted: selection {selection['selected_step']} "
                           f"vs earliest-max {best[0]}")
    history = fitted["history"]
    payload = {
        "arm": arm, "seed": int(seed), "updates": int(updates), "micro": MICRO, "accum": ACCUM,
        "max_length": int(max_length), "bf16": bool(bf16), "frozen_backbone": bool(frozen),
        "seconds": round(time.perf_counter() - started, 3),
        "selection": {"mode": "best-dev", "metric": "coarse macro-F1 on dev",
                      "selected_step": int(selection["selected_step"]),
                      "best_dev_macro_f1": float(selection["best_dev_value"]),
                      "earliest_max_step": int(best[0]), "ties_keep_earliest": True,
                      "terminal_step": int(fitted["updates"][-1]["step"])},
        "budget": fitted["budget"],
        "history": history,
        "optimizer_groups": [{"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]),
                              "tensors": len(group["params"])} for group in optimizer.param_groups],
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    dump(out_dir / "fit.json", payload)
    return {"fitted": fitted, "payload": payload, "optimizer": optimizer,
            "state": fitted["continuation"], "loss_fn": loss_fn, "targets": targets,
            "seconds": payload["seconds"]}


                                                                                        
                     
                                                                                        

def data_identity() -> dict:
    """四份 split 的字节 SHA：只做完整性锚点，不解析 heldout 行。"""
    missing = [split for split in SPLITS if not split_path(split).is_file()]
    if missing:
        raise FileNotFoundError(
            f"processed data 不完整，缺少 {missing}；先跑 build_relation_data.py 与 audit")
    return {split: sha256_file(split_path(split)) for split in SPLITS}


def model_identity() -> dict:
    directory = Path(C.QWEN["dir"])
    files = {item.name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
             for item in sorted(directory.iterdir()) if item.is_file()}
    return {"repo_id": C.QWEN["repo_id"], "revision": C.QWEN["revision"],
            "dir": str(directory), "files": files}


def code_identity() -> dict:
    return {
        "train_relation_heads.py": optional_sha(Path(__file__)),
        "score_and_verify.py": optional_sha(EXP / "score_and_verify.py"),
        "fit_common_20261024.py": optional_sha(FIT_DIR / "fit_common_20261024.py"),
        "finetune.py": optional_sha(Path(FT.__file__)),
        "metrics.py": optional_sha(Path(EM.__file__)),
        "semantic_relations.py": optional_sha(Path(C.SR.__file__)),
        "model_worker.py": optional_sha(REPO / "src" / "dataset_recommender" / "retrieval" / "model_worker.py"),
        "action_common_20261024.py": optional_sha(EXP.parent / "action_common_20261024.py"),
    }


def run_config(arm: str, seed: int) -> dict:
    return {"arm": arm, "seed": int(seed), "num_labels": ARMS[arm],
            "targets": PROTOCOL["arms"][arm]["targets"],
            "updates": UPDATES, "micro": MICRO, "accum": ACCUM, "max_length": MAX_LENGTH,
            "lr": QWEN_SPEC["lr"], "weight_decay": QWEN_SPEC["weight_decay"], "clip_norm": CLIP_NORM,
            "lora": dict(PROTOCOL["lora"]), "base": dict(PROTOCOL["base"]),
            "eval_every": EVAL_EVERY, "selection": PROTOCOL["selection"],
            "label_space": {"fine": list(FINE), "coarse": list(COARSE),
                            "fine_to_coarse": dict(FINE_TO_COARSE)},
            "split_files": {split: f"data/processed/{split}.jsonl" for split in SPLITS}}


def identity(arm: str, seed: int) -> dict:
    return {"config": run_config(arm, seed), "data_sha256": data_identity(),
            "model": model_identity(), "code": code_identity()}


def identity_mismatch(previous: dict, current: dict) -> list[str]:
    reasons = []
    for key in ("config", "data_sha256", "model", "code"):
        if previous.get(key) != current.get(key):
            reasons.append(key)
    return reasons


def run_dir_for(arm: str, seed: int) -> Path:
    return RUNS / f"{arm}-seed{int(seed)}"


def load_run(run: str, *, require_completed: bool = True) -> dict:
    run_dir = RUNS / run
    lock_path = run_dir / "run-lock.json"
    if not lock_path.is_file():
        raise SystemExit(f"missing run-lock.json: {lock_path}")
    lock = read_json(lock_path)
    if require_completed and lock.get("status") != "completed":
        raise SystemExit(f"run {run} is not completed (status={lock.get('status')!r})")
    result_path = run_dir / "result.json"
    result = read_json(result_path) if result_path.is_file() else None
    return {"run": run, "dir": run_dir, "lock": lock, "result": result,
            "arm": lock["identity"]["config"]["arm"], "seed": int(lock["identity"]["config"]["seed"])}


def reload_run(run: dict, checkpoint: str, device: torch.device) -> dict:
    """在新进程重建 Qwen3-0.6B 并严格载入 adapter+head（SHA 校验失败即拒绝）。"""
    built = build_arm(run["arm"], run["seed"])
    model = built["model"].to(device)
    loaded = load_checkpoint(model, run["dir"] / "checkpoints" / checkpoint)
    model.eval()
    return {"model": model, "tokenizer": built["tokenizer"], "arm": run["arm"],
            "seed": run["seed"], "init": built["init"], "reload": loaded,
            "checkpoint": checkpoint}


                                                                                        
          
                                                                                        

def cmd_selftest(args) -> int:
    """离线自检：不建 Qwen、不碰 CUDA、不读 processed 数据、不写 EXP。"""
    import tempfile
    torch.set_num_threads(1)
    checks: dict[str, Any] = {}

                                    
    checks["fine_mapping"] = {name: FINE_TO_COARSE[name] for name in FINE}
    fine_probs = np.asarray([[0.40, 0.30, 0.10, 0.10, 0.10],
                             [0.10, 0.20, 0.20, 0.40, 0.10]])
    coarse = coarse_probabilities(fine_probs, "fine5")
    expected = np.asarray([[0.40, 0.50, 0.10], [0.10, 0.50, 0.40]])
    assert np.allclose(coarse, expected, atol=1e-12), coarse
    assert int(fine_probs[0].argmax()) == FINE_EQUIVALENT and int(coarse[0].argmax()) == COARSE.index("compatible")
    checks["aggregation"] = {"fine_argmax": FINE[int(fine_probs[0].argmax())],
                             "coarse_argmax": COARSE[int(coarse[0].argmax())]}

              
    good = {"id": "x1", "root_id": "r1", "cluster_id": "c1", "family_id": "f1", "split": "dev",
            "label": "old_entails_new_strict", "coarse_label": "compatible", "label_origin": "spec",
            "source": "unit", "author": "qwen", "pair": {"old": {"text": "A 和 B。"}, "new": {"text": "A。"}},
            "visible_text": "old: A 和 B。 new: A。", "phenomena": []}
    assert validate_row(good, "dev") == []
    bad = dict(good, coarse_label="conflict")
    assert any("coarse_label" in error for error in validate_row(bad, "dev"))
    bad2 = dict(good, visible_text="")
    assert any("visible_text" in error for error in validate_row(bad2, "dev"))
    missing = {key: value for key, value in good.items() if key not in ("phenomena",)}
    assert any("missing keys" in error for error in validate_row(missing, "dev"))
    checks["row_contract"] = {"good_errors": 0, "bad_coarse_detected": True, "bad_visible_detected": True}

                            
    leaked = [dict(good, id="a", split="train"), dict(good, id="b", split="dev",
                                                     label="disjoint", coarse_label="conflict"),
              dict(good, id="c", split="calibration")]
    report = duplicate_label_audit({"train": [leaked[0]], "dev": [leaked[1]],
                                    "calibration": [leaked[2]], "heldout": []},
                                   allow_transformed=False)
    assert report["identical_text_conflicting_labels"] == 1, report
    assert report["identical_text_across_splits"] == 1
    assert report["identical_pair_across_splits"] == 1
    assert report["cross_split"]["family_id"] == 1
    assert report["duplicate_ids"] == 0
    assert report["passed"] is False
    checks["leakage"] = {key: report[key] for key in
                         ("identical_text_conflicting_labels", "identical_text_across_splits",
                          "identical_pair_across_splits", "visible_text_missing_side")}

                     
    denials = {}
    for purpose in ("train",):
        try:
            read_split("heldout", purpose=purpose)
        except HeldoutLockError as error:
            denials[purpose] = str(error)[:40]
        else:
            raise AssertionError(f"heldout must be refused for purpose={purpose}")
    try:
        validate_final_lock(EXP / "runs" / "__no_such_lock__.json")
    except HeldoutLockError as error:
        denials["missing_lock"] = str(error)[:40]
    else:
        raise AssertionError("missing final-lock must be refused")
    checks["heldout_denials"] = denials

              
    history = [{"step": 50, "dev_macro_f1": 0.5}, {"step": 100, "dev_macro_f1": 0.7},
               {"step": 150, "dev_macro_f1": 0.7}, {"step": 200, "dev_macro_f1": 0.6}]
    assert best_dev_from_curve(history) == (100, 0.7), best_dev_from_curve(history)
    checks["tie_earliest"] = best_dev_from_curve(history)

                                                    
    with tempfile.TemporaryDirectory(prefix="relation-heads-selftest-") as temporary:
        root = Path(temporary)
        rows = fake_rows()
        device = torch.device("cpu")
        built = build_arm("fine5", 13, builder=lambda seed: build_fake_model("fine5", seed))
        encode = default_encode(built["tokenizer"], device, 64)
        training = fit_arm(model=built["model"], tokenizer=built["tokenizer"], train_rows=rows,
                           dev_rows=rows[:4], arm="fine5", seed=13, device=device, bf16=False,
                           max_length=64, out_dir=root / "fake-run", encode=encode, updates=2,
                           eval_every=1, frozen=False, verbose=False)
        state = training["state"]
        model = built["model"]
        best_state = state["best_state"]
        assert best_state is not None
        model.load_state_dict(best_state, strict=True)
        best_eval = evaluate_split(model=model, tokenizer=built["tokenizer"], rows=rows, arm="fine5",
                                   split="dev", device=device, max_length=64, bf16=False, encode=encode)
        meta = {"format": "fake-full", "num_labels": 5, "selection": "best-dev",
                "full_state_sha256": state_sha256(model.state_dict())}
        save_checkpoint(root / "fake-run" / "checkpoints" / "best-dev", model, {}, meta)
        model.load_state_dict(state["terminal_state"], strict=True)
        meta_terminal = dict(meta, selection="terminal", full_state_sha256=state_sha256(model.state_dict()))
        save_checkpoint(root / "fake-run" / "checkpoints" / "terminal", model, {}, meta_terminal)
        rebuilt = build_arm("fine5", 13, builder=lambda seed: build_fake_model("fine5", seed))
        reloaded = load_checkpoint(rebuilt["model"], root / "fake-run" / "checkpoints" / "best-dev")
        replay = evaluate_split(model=rebuilt["model"], tokenizer=rebuilt["tokenizer"], rows=rows,
                                arm="fine5", split="dev", device=device, max_length=64,
                                bf16=False, encode=encode)
        delta = float(np.abs(replay["coarse_probabilities"] - best_eval["coarse_probabilities"]).max())
        assert delta <= 1e-5, delta
        assert [row["pred_coarse"] for row in replay["predictions"]] == [
            row["pred_coarse"] for row in best_eval["predictions"]]
        curve = read_jsonl(root / "fake-run" / "curve.jsonl")
        updates = read_jsonl(root / "fake-run" / "updates.jsonl")
        assert len(curve) == 2 and len(updates) == 2, (len(curve), len(updates))
        assert curve[0]["dev"]["coarse"]["macro_f1"] is not None
        checks["fake_reload"] = {"reload_format": reloaded["format"], "max_probability_delta": delta,
                                 "curve_entries": len(curve), "update_entries": len(updates),
                                 "selected_step": training["payload"]["selection"]["selected_step"]}
        checks["fake_fit"] = {"budget_updates": training["payload"]["budget"]["actual_updates"],
                              "sample_instances": training["payload"]["budget"]["actual_sample_instances"],
                              "real_tokens": training["payload"]["budget"]["real_tokens"]}

    print(json.dumps({"selftest": True, "ok": True, "checks": checks}, ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    groups = {}
    for split in SPLITS:
        if split == "heldout" and getattr(args, "no_heldout_integrity", False):
            continue
        purpose = "audit"
        groups[split] = read_split(split, purpose=purpose)
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {"dir": str(PROCESSED), "splits": {split: str(split_path(split)) for split in SPLITS}},
        "protocol": PROTOCOL,
        "heldout_access": {"read": "heldout" in groups, "purpose": "integrity-only",
                           "model_built": False, "metrics_computed": False,
                           "rows_logged": False},
        "row_keys": list(ROW_KEYS),
        "splits": {},
        "errors": [],
    }
    contract_errors: dict[str, list] = {}
    extra_keys: dict[str, list] = {}
    for split, rows in groups.items():
        errors: list[dict] = []
        extras: set[str] = set()
        usable: list[dict] = []
        for row in rows:
            row_errors = validate_row(row, split)
            if row_errors:
                errors.append({"id": row.get("id") if isinstance(row, dict) else None,
                               "errors": row_errors})
            if isinstance(row, dict) and all(key in row for key in
                                             ("label", "coarse_label", "family_id", "cluster_id",
                                              "root_id", "source", "author", "label_origin",
                                              "phenomena")):
                usable.append(row)
            if isinstance(row, dict):
                extras |= set(row) - set(ROW_KEYS)
        contract_errors[split] = errors
        extra_keys[split] = sorted(extras)
        labels = {name: sum(1 for row in usable if row["label"] == name) for name in FINE}
        coarse = {name: sum(1 for row in usable if row["coarse_label"] == name) for name in COARSE}
        families: dict[str, list[str]] = {}
        without_family = 0
        for row in usable:
            family = row["family_id"]
            if not isinstance(family, str) or not family.strip():
                without_family += 1
                continue
            families.setdefault(family, []).append(row["label"])
        unbalanced = {name: {leaf: values.count(leaf) for leaf in FINE}
                      for name, values in families.items() if len(values) != 15
                      or any(values.count(leaf) != 3 for leaf in FINE)}

        def distinct(key: str, rows=usable) -> int:
            return len({row[key] for row in rows if isinstance(row[key], str) and row[key]})

        report["splits"][split] = {
            "n": len(rows),
            "usable_rows": len(usable),
            "labels": labels, "coarse": coarse,
            "families": len(families), "rows_without_family_id": without_family,
            "clusters": distinct("cluster_id"), "roots": distinct("root_id"),
            "sources": sorted({row["source"] for row in usable if isinstance(row["source"], str)}),
            "authors": sorted({row["author"] for row in usable if isinstance(row["author"], str)}),
            "label_origins": sorted({row["label_origin"] for row in usable
                                     if isinstance(row["label_origin"], str)}),
            "phenomena_nonempty": sum(1 for row in usable if row["phenomena"]),
            "families_not_15_or_unbalanced": len(unbalanced),
            "unbalanced_examples": list(unbalanced.items())[:2],
            "extra_keys": extra_keys[split],
            "contract_errors": len(errors),
            "contract_error_examples": errors[:3],
        }
    report["leakage"] = duplicate_label_audit(groups, allow_transformed=getattr(
        args, "allow_transformed_visible", False))

                                         
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(C.QWEN["dir"]), local_files_only=True)
        profiles, failures = {}, []
        for split, rows in groups.items():
            if not rows:
                profiles[split] = {"n": 0, "note": "empty split", "max_length_budget": MAX_LENGTH,
                                   "fits": True}
                continue
            texts = [row["visible_text"] for row in rows]
            profile = C.token_lengths(tokenizer, texts)
            profile["max_length_budget"] = MAX_LENGTH
            profile["fits"] = profile["max"] <= MAX_LENGTH
            profiles[split] = profile
            if not profile["fits"]:
                over = [row["id"] for row in rows
                        if len(tokenizer(row["visible_text"], add_special_tokens=True,
                                         truncation=False)["input_ids"]) > MAX_LENGTH]
                failures.append({"split": split, "over_budget_ids": over[:5], "count": len(over)})
        report["tokenizer"] = {"dir": str(C.QWEN["dir"]), "tokenizer_class": type(tokenizer).__name__,
                               "max_length": MAX_LENGTH, "truncation": False,
                               "profiles": profiles, "over_budget": failures}
    finally:
        tokenizer = None

    contract_ok = not any(contract_errors.values())
    tokenizer_ok = bool(report.get("tokenizer")) and not report["tokenizer"]["over_budget"]
    report["passed"] = bool(contract_ok and tokenizer_ok and report["leakage"]["passed"])
    if not contract_ok:
        report["errors"].append({"code": "row_contract",
                                 "splits": {split: len(errors) for split, errors in contract_errors.items()}})
    if not tokenizer_ok:
        report["errors"].append({"code": "tokenizer_over_budget"})
    if not report["leakage"]["passed"]:
        report["errors"].append({"code": "leakage", "detail": {
            key: report["leakage"][key] for key in
            ("identical_text_conflicting_labels", "identical_text_across_splits",
             "identical_pair_across_splits", "duplicate_ids", "cross_split",
             "visible_text_missing_side")}})
    dump(AUDITS / "input-audit.json", report)
    print(json.dumps({"audit": True, "passed": report["passed"],
                      "splits": {split: report["splits"][split]["n"] for split in groups},
                      "labels": {split: report["splits"][split]["labels"] for split in groups},
                      "token_max": {split: report["tokenizer"]["profiles"][split].get("max")
                                    for split in groups},
                      "leakage": {key: report["leakage"][key] for key in
                                  ("identical_text_conflicting_labels", "identical_text_across_splits",
                                   "identical_pair_across_splits", "duplicate_ids")},
                      "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


def smoke_one(arm: str, seed: int, device: torch.device) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    built = build_arm(arm, seed)
    model, tokenizer = built["model"], built["tokenizer"]
    train_rows = read_split("train", purpose="train")
    dev_rows = read_split("dev", purpose="train")
    profiles = {name: C.assert_fits(tokenizer, [row["visible_text"] for row in rows], MAX_LENGTH,
                                    f"qwen3/{name}")
                for name, rows in (("train", train_rows), ("dev", dev_rows))}
    model.to(device)
    optimizer = C.build_optimizer("qwen3", model)
    with anchor_audit_sink(AUDITS):
        audit = C.parameter_audit(model, optimizer, f"smoke-{arm}-seed{seed}")
    targets = targets_for(train_rows, arm)
    encode = default_encode(tokenizer, device, MAX_LENGTH)
    loss_fn = C.make_loss(targets, device)
    window = next(FT.training_windows(len(train_rows), epochs=1, batch_size=MICRO, grad_accum=ACCUM,
                                      max_steps=1, rng=np.random.default_rng(seed)))
    chunk = window[2][0]
    loss_check = loss_target_selftest(loss_fn, targets, train_rows, arm, chunk, device)
    probe = C.gradient_and_update_probe(model=model, optimizer=optimizer,
                                        encode_fn=lambda indices: encode(indices, train_rows),
                                        loss_fn=loss_fn, chunk=chunk, device=device,
                                        bf16=QWEN_SPEC["bf16"], frozen=True,
                                        label=f"qwen3-{arm}-seed{seed}")
    assert not probe["frozen_params_with_grad"], "frozen base received gradients"
    assert probe["nonzero_backbone_tensors"] and probe["backbone_changed_tensors"], \
        "LoRA adapters did not receive gradients or did not move"
    record = {
        "ok": True, "arm": arm, "seed": int(seed), "environment": environment(),
        "token_profiles": profiles, "max_length": MAX_LENGTH,
        "init": built["init"],
        "model": {"trainable_parameters": audit["trainable_parameters"],
                  "total_parameters": audit["total_parameters"], "dtypes": audit["dtypes"],
                  "lora": audit.get("lora")},
        "optimizer_groups": audit["groups"],
        "loss_target": loss_check,
        "gradient_update": {key: probe[key] for key in probe if key not in ("stats", "deltas")},
        "frozen_base_sha256": frozen_base_sha256(model),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device": str(device), "script_sha256": sha256_file(Path(__file__)),
    }
    del optimizer, model, built
    torch.cuda.empty_cache()
    return record


def cmd_smoke(args) -> int:
    device = FT.resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("smoke 必须显式跑 CUDA；本脚本拒绝 CPU 冒烟")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    records = {}
    for arm in arms:
        records[arm] = smoke_one(arm, args.seed, device)
        dump(AUDITS / f"smoke-{arm}-seed{int(args.seed)}.json", records[arm])
    lora_shas = {arm: records[arm]["init"]["lora_init_sha256"] for arm in arms}
    same_lora = len(set(lora_shas.values())) <= 1
    combined = {"seed": int(args.seed), "arms": arms, "lora_init_sha256": lora_shas,
                "same_seed_lora_init_identical": same_lora,
                "cuda_peak_allocated_bytes": max(record["cuda_peak_allocated_bytes"]
                                                 for record in records.values()),
                "records": {arm: {"gradient": {key: records[arm]["gradient_update"][key] for key in
                                               ("nonzero_grad_tensors", "changed_tensors",
                                                "frozen_params_with_grad", "nonzero_backbone_tensors",
                                                "backbone_changed_tensors")},
                                  "trainable_parameters": records[arm]["model"]["trainable_parameters"],
                                  "head_out_features": records[arm]["init"]["final_out_features"]}
                            for arm in arms}}
    dump(AUDITS / f"smoke-summary-seed{int(args.seed)}.json", combined)
    print(json.dumps({"smoke_ok": True, **combined}, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    arm, seed = args.arm, int(args.seed)
    assert arm in ARMS and seed in SEEDS
    device = FT.resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("train 必须显式跑 CUDA；本脚本拒绝 CPU 训练（不做静默回退）")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    run_dir = run_dir_for(arm, seed)
    lock_path = run_dir / "run-lock.json"
    current = identity(arm, seed)
    previous = None
    if run_dir.exists():
        if not args.resume:
            raise SystemExit(f"run 目录已存在：{run_dir}（输出不得覆盖；续跑请显式 --resume）")
        if not lock_path.is_file():
            raise SystemExit(f"拒绝续跑：缺少 {lock_path}，无法证明是同配置同 SHA")
        previous = read_json(lock_path)
        if previous.get("status") == "completed":
            raise SystemExit(f"拒绝续跑：{run_dir} 已完成；复评请用 replay / score_and_verify")
        reasons = identity_mismatch(previous.get("identity", {}), current)
        if reasons:
            raise SystemExit(f"拒绝续跑：{reasons} 与上次不一致（断点只接受同配置同 SHA）")
    else:
        run_dir.mkdir(parents=True)
    lock = {
        "schema": "relation-run-lock/1", "status": "in_progress",
        "created": (previous or {}).get("created", time.strftime("%Y-%m-%dT%H:%M:%S")),
        "run": run_dir.name, "identity": current,
        "heldout": {"file_sha256": current["data_sha256"]["heldout"],
                    "rows_parsed": False,
                    "note": "heldout 只以字节 SHA 进入身份；train 从不解析其行"},
        "attempts": list((previous or {}).get("attempts", [])) + [
            {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "resume": bool(args.resume)}],
        "environment": environment(), "protocol": PROTOCOL,
    }
    dump(lock_path, lock)
    started = time.perf_counter()
    try:
        train_rows = read_split("train", purpose="train")
        dev_rows = read_split("dev", purpose="train")
        if len(train_rows) < MICRO * ACCUM:
            raise SystemExit(f"train split too small: {len(train_rows)}")
        built = build_arm(arm, seed)
        model, tokenizer = built["model"], built["tokenizer"]
        sibling = run_dir_for("fine5" if arm == "direct3" else "direct3", seed) / "run-lock.json"
        sibling_init = (read_json(sibling).get("init") or {}) if sibling.is_file() else {}
        sibling_lora = sibling_init.get("lora_init_sha256")
        lora_matches_sibling = (None if sibling_lora is None
                                else bool(sibling_lora == built["init"]["lora_init_sha256"]))
        if lora_matches_sibling is False:
            raise RuntimeError("同 seed 两臂的 LoRA 初始化 SHA 不同：初始化不可比，拒绝开训")
        token_profiles = {name: C.assert_fits(tokenizer, [row["visible_text"] for row in rows],
                                              MAX_LENGTH, f"qwen3/{name}")
                          for name, rows in (("train", train_rows), ("dev", dev_rows))}
        model.to(device)
        base_before = frozen_base_sha256(model)
        optimizer = C.build_optimizer("qwen3", model)
        with anchor_audit_sink(AUDITS):
            audit = C.parameter_audit(model, optimizer, f"train-{arm}-seed{seed}")
        targets = targets_for(train_rows, arm)
        encode = default_encode(tokenizer, device, MAX_LENGTH)
        loss_fn = C.make_loss(targets, device)
        window = next(FT.training_windows(len(train_rows), epochs=1, batch_size=MICRO, grad_accum=ACCUM,
                                          max_steps=1, rng=np.random.default_rng(seed)))
        chunk = window[2][0]
        loss_check = loss_target_selftest(loss_fn, targets, train_rows, arm, chunk, device)
        probe = C.gradient_and_update_probe(model=model, optimizer=optimizer,
                                            encode_fn=lambda indices: encode(indices, train_rows),
                                            loss_fn=loss_fn, chunk=chunk, device=device,
                                            bf16=QWEN_SPEC["bf16"], frozen=True,
                                            label=f"qwen3-{arm}-seed{seed}")
        initial_sha = probe["initial_state_sha256"]
        if probe["frozen_params_with_grad"]:
            raise RuntimeError("frozen base received gradients during the probe")
        initial = {"train": evaluate_split(model=model, tokenizer=tokenizer, rows=train_rows, arm=arm,
                                           split="train", device=device, max_length=MAX_LENGTH,
                                           bf16=QWEN_SPEC["bf16"], encode=encode)["summary"],
                   "dev": evaluate_split(model=model, tokenizer=tokenizer, rows=dev_rows, arm=arm,
                                         split="dev", device=device, max_length=MAX_LENGTH,
                                         bf16=QWEN_SPEC["bf16"], encode=encode)["summary"]}
        dump(run_dir / "initial.json", initial)
        dump(run_dir / "probes.json", {"parameter_audit": audit, "loss_target": loss_check,
                                       "gradient_update": {key: probe[key] for key in probe
                                                           if key not in ("stats", "deltas")},
                                       "token_profiles": token_profiles,
                                       "init": built["init"], "initial_state_sha256": initial_sha})
        training = fit_arm(model=model, tokenizer=tokenizer, train_rows=train_rows, dev_rows=dev_rows,
                           arm=arm, seed=seed, device=device, bf16=QWEN_SPEC["bf16"],
                           max_length=MAX_LENGTH, out_dir=run_dir, encode=encode, frozen=True)
        state = training["state"]
        terminal_state, best_state = state["terminal_state"], state["best_state"]
        if best_state is None:
            raise RuntimeError("no dev-best snapshot was captured")
        model.load_state_dict(best_state, strict=True)
        if state_sha256(model.state_dict()) != state_sha256(best_state):
            raise RuntimeError("model does not hold the dev-best weights")
        best_eval = {name: evaluate_split(model=model, tokenizer=tokenizer, rows=rows, arm=arm,
                                          split=name, device=device, max_length=MAX_LENGTH,
                                          bf16=QWEN_SPEC["bf16"], encode=encode)
                     for name, rows in (("train", train_rows), ("dev", dev_rows))}
        best_meta = {"format": "qwen-lora", "arm": arm, "seed": seed,
                     "num_labels": ARMS[arm], "classes": list(COARSE) if arm == "direct3" else list(FINE),
                     "step": int(state["selected_step"]), "selection": "best-dev",
                     "best_dev_coarse_macro_f1": float(state["best_value"]),
                     "adapter_lora_sha256": state_sha256(lora_state(model)),
                     "head_sha256": state_sha256(model.classifier.state_dict()),
                     "targets": PROTOCOL["arms"][arm]["targets"], "protocol": PROTOCOL["study"]}
        best_ckpt = save_checkpoint(run_dir / "checkpoints" / "best-dev", model,
                                    model.classifier.state_dict(), best_meta)
        model.load_state_dict(terminal_state, strict=True)
        if state_sha256(model.state_dict()) != state_sha256(terminal_state):
            raise RuntimeError("model does not hold the terminal weights")
        terminal_step = int(training["fitted"]["updates"][-1]["step"])
        terminal_eval = {name: evaluate_split(model=model, tokenizer=tokenizer, rows=rows, arm=arm,
                                              split=name, device=device, max_length=MAX_LENGTH,
                                              bf16=QWEN_SPEC["bf16"], encode=encode)
                         for name, rows in (("train", train_rows), ("dev", dev_rows))}
        terminal_meta = {"format": "qwen-lora", "arm": arm, "seed": seed,
                         "num_labels": ARMS[arm], "classes": list(COARSE) if arm == "direct3" else list(FINE),
                         "step": terminal_step, "selection": "terminal",
                         "adapter_lora_sha256": state_sha256(lora_state(model)),
                         "head_sha256": state_sha256(model.classifier.state_dict()),
                         "targets": PROTOCOL["arms"][arm]["targets"], "protocol": PROTOCOL["study"]}
        terminal_ckpt = save_checkpoint(run_dir / "checkpoints" / "terminal", model,
                                        model.classifier.state_dict(), terminal_meta)
        base_after = frozen_base_sha256(model)
        if base_before != base_after:
            raise RuntimeError("frozen Qwen base changed during training")
        for tag, payload in (("best-dev", best_eval), ("terminal", terminal_eval)):
            dump(run_dir / "scores" / tag / "train" / "summary.json", payload["train"]["summary"])
            dump(run_dir / "scores" / tag / "dev" / "summary.json", payload["dev"]["summary"])
            write_jsonl(run_dir / "scores" / tag / "train" / "predictions.jsonl", payload["train"]["predictions"])
            write_jsonl(run_dir / "scores" / tag / "dev" / "predictions.jsonl", payload["dev"]["predictions"])
        result = {
            "run": run_dir.name, "arm": arm, "seed": seed, "environment": environment(),
            "run_lock_sha256_at_training_start": sha256_file(lock_path),
            "config": current["config"], "identity": {"data_sha256": current["data_sha256"],
                                                     "model": current["model"], "code": current["code"]},
            "data_rows": {"train": len(train_rows), "dev": len(dev_rows)},
            "label_counts": {name: int(sum(1 for value in targets if value == index))
                             for index, name in enumerate(COARSE if arm == "direct3" else FINE)},
            "init": built["init"], "initial_state_sha256": initial_sha,
            "lora_init_matches_sibling_arm": lora_matches_sibling,
            "frozen_base_sha256": {"before": base_before, "after": base_after},
            "probes": {"parameter_audit": audit, "loss_target": loss_check,
                       "gradient_update": {key: probe[key] for key in probe
                                           if key not in ("stats", "deltas")}},
            "fit": {key: value for key, value in training["payload"].items() if key != "history"},
            "selection": training["payload"]["selection"],
            "initial": initial,
            "checkpoints": {"best-dev": {"step": best_meta["step"], "checkpoint": best_ckpt,
                                         "metrics": {name: best_eval[name]["summary"]
                                                     for name in ("train", "dev")}},
                            "terminal": {"step": terminal_step, "checkpoint": terminal_ckpt,
                                         "metrics": {name: terminal_eval[name]["summary"]
                                                     for name in ("train", "dev")}}},
            "token_profiles": token_profiles,
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                          if device.type == "cuda" else 0),
            "cuda_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                         if device.type == "cuda" else 0),
            "seconds": round(time.perf_counter() - started, 3),
            "heldout_scored": False,
            "no_deploy": True,
        }
        dump(run_dir / "result.json", result)
        lock.update({
            "status": "completed",
            "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "selected_step": int(state["selected_step"]),
            "terminal_step": terminal_step,
            "selection": {"metric": "dev coarse macro-F1", "ties": "earliest",
                          "best_dev_coarse_macro_f1": float(state["best_value"])},
            "checkpoints": {"best-dev": best_ckpt, "terminal": terminal_ckpt},
            "init": built["init"],
            "lora_init_matches_sibling_arm": lora_matches_sibling,
            "run_lock_sha256_before_completion": result["run_lock_sha256_at_training_start"],
            "seconds": result["seconds"],
            "heldout_metrics": None,
        })
        dump(lock_path, lock)
        print(json.dumps({"completed": True, "run": run_dir.name,
                          "selected_step": int(state["selected_step"]),
                          "terminal_step": terminal_step,
                          "best_dev_coarse_macro_f1": float(state["best_value"]),
                          "terminal_dev_coarse_macro_f1": terminal_eval["dev"]["summary"]["coarse"]["macro_f1"],
                          "budget": training["payload"]["budget"],
                          "cuda_peak_gib": round(result["cuda_peak_allocated_bytes"] / 2 ** 30, 2),
                          "seconds": result["seconds"]}, ensure_ascii=False))
        return 0
    except BaseException as error:
        lock["status"] = "failed"
        lock["failed"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "error": f"{type(error).__name__}: {error}"[:2000]}
        dump(lock_path, lock)
        raise


def cmd_replay(args) -> int:
    run = load_run(args.run)
    checkpoint = args.checkpoint
    split = args.split
    unlock = None
    if split == "heldout":
        unlock = validate_final_lock(args.unlock_final, run=run["run"], checkpoint=checkpoint)
        if int(unlock.get("selected_step", -1)) != int(run["result"]["selection"]["selected_step"]):
            raise SystemExit("final-lock selected_step 与 run 记录不一致")
    device = FT.resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("replay 必须显式跑 CUDA（0.6B + LoRA 不做 CPU 回退）")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    rows = read_split(split, purpose=("locked-final" if split == "heldout" else "train"),
                      unlock=unlock)
    loaded = reload_run(run, checkpoint, device)
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    ev = evaluate_split(model=model, tokenizer=tokenizer, rows=rows, arm=loaded["arm"], split=split,
                        device=device, max_length=MAX_LENGTH, bf16=QWEN_SPEC["bf16"],
                        encode=default_encode(tokenizer, device, MAX_LENGTH))
    out_dir = run["dir"] / "replay" / f"{checkpoint}-{split}"
    write_jsonl(out_dir / "predictions.jsonl", ev["predictions"])
    payload = {"run": run["run"], "checkpoint": checkpoint, "split": split,
               "reload": loaded["reload"], "summary": ev["summary"],
               "environment": environment(),
               "checkpoint_files_sha256": checkpoint_file_shas(run["dir"] / "checkpoints" / checkpoint),
               "final_lock": None if unlock is None else {
                   "path": str(Path(args.unlock_final)), "sha256": sha256_file(args.unlock_final),
                   "data_sha256": unlock["data_sha256"]},
               "heldout_rows_parsed": split == "heldout"}
    if split in ("train", "dev"):
        stored = run["result"]["checkpoints"][checkpoint]["metrics"][split]
        deltas = {}
        for key in ("accuracy", "macro_f1", "loss"):
            if stored["coarse"].get(key) is not None:
                deltas[key] = abs(float(stored["coarse"][key]) - float(ev["summary"]["coarse"][key]))
        stored_pred = {row["id"]: row for row in read_jsonl(
            run["dir"] / "scores" / checkpoint / split / "predictions.jsonl")}
        max_delta, argmax_mismatch = compare_predictions(stored_pred, ev["predictions"])
        payload["stored_comparison"] = {"summary_abs_delta": deltas, "max_probability_delta": max_delta,
                                        "argmax_mismatch": argmax_mismatch}
        if argmax_mismatch or max_delta > 1e-5 or max(deltas.values(), default=0.0) > 1e-6:
            dump(out_dir / "replay.json", payload)
            raise SystemExit(f"replay 与存档不一致：{payload['stored_comparison']}")
    dump(out_dir / "replay.json", payload)
    print(json.dumps({"replayed": run["run"], "checkpoint": checkpoint, "split": split,
                      "coarse": {key: ev["summary"]["coarse"][key]
                                 for key in ("n", "accuracy", "macro_f1", "loss")},
                      "fine": None if ev["summary"]["fine"] is None else
                      {key: ev["summary"]["fine"][key] for key in ("n", "accuracy", "macro_f1")},
                      "comparison": payload.get("stored_comparison")}, ensure_ascii=False))
    return 0


def compare_predictions(stored: dict[str, dict], fresh: Sequence[dict]) -> tuple[float, int]:
    """按 id 对齐比较 coarse/fine 概率与 argmax；返回 (最大概率差, argmax 不一致数)。"""
    max_delta, mismatches = 0.0, 0
    for row in fresh:
        previous = stored.get(row["id"])
        if previous is None:
            raise RuntimeError(f"stored predictions missing id {row['id']}")
        for key in ("coarse_probabilities", "fine_probabilities"):
            left, right = previous.get(key), row.get(key)
            if left is None or right is None:
                if (left is None) != (right is None):
                    raise RuntimeError(f"{row['id']}: {key} presence differs")
                continue
            max_delta = max(max_delta, max(abs(float(a) - float(b)) for a, b in zip(left, right)))
        for key in ("pred_coarse", "pred_fine"):
            if previous.get(key) != row.get(key):
                mismatches += 1
    if set(stored) != {row["id"] for row in fresh}:
        raise RuntimeError("stored prediction id set differs from the fresh one")
    return max_delta, mismatches


                                                                                        
                                                                  
                                                                                        

class FakeTokenizer:
    """字符级假 tokenizer：只实现 encode_batch_visible / token 长度所需的接口。"""

    model_input_names = ["input_ids", "attention_mask"]
    pad_token = "<pad>"
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        ids = [min(ord(character) % 97 + 2, 63) for character in text]
        return [1] + (ids or [2]) + [2]

    def __call__(self, texts, padding="longest", truncation=False, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        width = max(len(item) for item in encoded) if padding == "longest" else max(len(item) for item in encoded)
        input_ids, attention = [], []
        for item in encoded:
            pad = width - len(item)
            input_ids.append(item + [self.pad_token_id] * pad)
            attention.append([1] * len(item) + [0] * pad)
        result = {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                  "attention_mask": torch.tensor(attention, dtype=torch.long)}
        return result if return_tensors == "pt" else {key: value.tolist() for key, value in result.items()}


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, hidden)
        self.proj = torch.nn.Linear(hidden, hidden)

    def forward(self, input_ids=None, attention_mask=None, **extra):
        hidden = torch.tanh(self.proj(self.embedding(input_ids)))

        class Output:
            pass

        output = Output()
        output.last_hidden_state = hidden
        return output


def build_fake_model(arm: str, seed: int):
    torch.manual_seed(int(seed))
    backbone = FakeBackbone()
    model = C.PooledClassifier(backbone, 8, pool="last_nonpad", seed=seed, num_labels=3)
    return model, FakeTokenizer()


def fake_rows() -> list[dict]:
    texts = [("所有结果必须按时间升序排列。", "所有结果必须按时间降序排列。"),
             ("报告必须包含研究方法和局限。", "报告必须包含研究方法。"),
             ("报告必须包含研究方法。", "报告必须包含研究方法和局限。"),
             ("默认使用中文回复。", "默认使用中文回复。"),
             ("报告需要说明研究方法。", "使用图表展示计算资源消耗。"),
             ("每次汇报都要说明方法。", "每次汇报都要详细解释方法。")]
    labels = ["disjoint", "old_entails_new_strict", "new_entails_old_strict",
              "equivalent", "overlap_incomparable", "old_entails_new_strict"]
    rows = []
    for index, ((old, new), label) in enumerate(zip(texts, labels)):
        rows.append({"id": f"fake-{index}", "root_id": f"root-{index}", "cluster_id": f"cluster-{index}",
                     "family_id": f"family-{index}", "split": "dev", "label": label,
                     "coarse_label": FINE_TO_COARSE[label], "label_origin": "spec",
                     "source": "selftest", "author": "fake",
                     "pair": {"old": {"kind": "memory", "text": old},
                              "new": {"kind": "memory", "text": new}},
                     "visible_text": f"old: {old} new: {new}", "phenomena": []})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    audit = sub.add_parser("audit")
    audit.add_argument("--allow-transformed-visible", action="store_true",
                       help="把 visible_text 未包含原始两侧降级为警告（默认硬失败）")
    audit.add_argument("--no-heldout-integrity", action="store_true",
                       help="跳过 heldout 的完整性/泄漏检查（默认做，只算哈希与计数）")
    audit.set_defaults(func=cmd_audit)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--arm", default="all", choices=("direct3", "fine5", "all"))
    smoke.add_argument("--seed", type=int, default=13, choices=SEEDS)
    smoke.add_argument("--device", default="cuda:0")
    smoke.set_defaults(func=cmd_smoke)
    train = sub.add_parser("train")
    train.add_argument("--arm", required=True, choices=tuple(ARMS))
    train.add_argument("--seed", type=int, required=True, choices=SEEDS)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(func=cmd_train)
    replay = sub.add_parser("replay")
    replay.add_argument("--run", required=True)
    replay.add_argument("--checkpoint", default="best-dev", choices=("best-dev", "terminal"))
    replay.add_argument("--split", default="dev", choices=SPLITS)
    replay.add_argument("--unlock-final", default=None)
    replay.add_argument("--device", default="cuda:0")
    replay.set_defaults(func=cmd_replay)
    args = parser.parse_args()
    return (getattr(args, "func", None) or cmd_selftest)(args)


if __name__ == "__main__":
    raise SystemExit(main())
