"""冲突识别线公共层：路径、预算常量、路由、行契约、锁与 io。

唯一真源复用（本模块不复制下列实现）：

* ``../原子关系跨模型复验与Qwen训练-2026-10-25/train_relation_heads.py``：Qwen3-0.6B
  + LoRA 构建、``last_nonpad`` 编码、coarse 指标、严格 checkpoint 存取、dev 选点平局规则；
* ``../../固定数据拟合与基座对照-2026-10-24/fit_common_20261024.py``：``build_qwen``、
  sum-CE 损失、优化器与 ``PROTOCOL['qwen3']``；
* repo ``dataset_recommender.agent.semantic_relations``：pair 校验与 ``visible_text`` 序列化；
* repo ``research.memory_relations.finetune``：``fit_joint`` 固定实例预算训练循环。

本实验只读上述文件；repo 与旧实验一律不写。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path(__file__).resolve().parents[6] / 'source'
PARENT_EXP = ROOT.parent / "原子关系跨模型复验与Qwen训练-2026-10-25"
FIT_DIR = ROOT.parents[1] / "固定数据拟合与基座对照-2026-10-24"

for _extra in (str(REPO / "src"), str(REPO), str(FIT_DIR), str(PARENT_EXP)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import numpy as np              

import train_relation_heads as TRH              

C = TRH.C
SR = C.SR
FT = TRH.FT

FINE = TRH.FINE
COARSE = TRH.COARSE
FINE_TO_COARSE = TRH.FINE_TO_COARSE
ARMS = TRH.ARMS
SEEDS = TRH.SEEDS
UPDATES = TRH.UPDATES
MICRO = TRH.MICRO
ACCUM = TRH.ACCUM
MAX_LENGTH = TRH.MAX_LENGTH
EVAL_EVERY = TRH.EVAL_EVERY
QWEN_SPEC = TRH.QWEN_SPEC

                                                                                        
                         
                                                                                        

SPLITS = ("train", "dev", "heldout")
FAMILIES_PER_SPLIT = {"train": 24, "dev": 12, "heldout": 24}
FAMILY_TOTAL = sum(FAMILIES_PER_SPLIT.values())                              
                                                                 
ROWS_PER_FAMILY = 18
SHARED_PER_FAMILY = 6
CONTROL_PER_FAMILY = 6
DYADS_PER_FAMILY = 3
DYAD_ROWS_PER_FAMILY = 2 * DYADS_PER_FAMILY
NATURAL_PER_FAMILY = SHARED_PER_FAMILY + CONTROL_PER_FAMILY
ROWS_PER_CLASS_PER_FAMILY = 6

                                                                          
                                                                   
                            
SHARED_LABEL_PLAN = ("conflict", "conflict", "compatible", "compatible", "equivalent", "equivalent")
DYAD_LABEL_PLAN = (
                                                                   
    ("D2", "compatible", "conflict"),
    ("D1", "equivalent", "conflict"),
    ("D7", "equivalent", "compatible"),
)

SOURCE_NEW = "cf-new"
MATCHED_WINDOW_FIELD = "matched_window_id"
                                                       
AUTHOR_PLAN = {
    "train": ("deepseek-v4-flash",),
    "dev": ("deepseek-v4-flash",),
    "heldout": ("batiai/qwen3.8-27b:iq3",),
    "composite": ("deepseek-v4-flash",),
}
POOL_SHARED = "shared"
POOL_CONTROL = "control"
POOL_COUNTERFACTUAL = "counterfactual"
POOLS = (POOL_SHARED, POOL_CONTROL, POOL_COUNTERFACTUAL)
SOURCE_LEGACY = "legacy-514"
LABEL_ORIGIN = "spec_rule"
FIXTURE_AUTHOR = "fixture-renderer-v0"
FIXTURE_SOURCE = "cf-fixture"

NEW_ROW_KEYS = ("id", "root_id", "cluster_id", "family_id", "split", "label", "coarse_label",
                "label_origin", "source", "author", "pair", "visible_text", "phenomena", "dyad")

                                                                                        
         
                                                                                        

TRAIN_ROUTES = ("r1", "r2", "r3")
SCORE_ROUTES = ("r0", "r1", "r2", "r3")
MARGIN = 0.5
MARGIN_WEIGHT = 0.2
MICRO_PER_WINDOW = ACCUM                           
WINDOW_ROWS = MICRO * MICRO_PER_WINDOW             
DYAD_SLOTS_PER_WINDOW = 4                                                 
NATURAL_SLOTS_PER_WINDOW = MICRO_PER_WINDOW - DYAD_SLOTS_PER_WINDOW

CONTROL_POLICY_VERSION = "control-matching-v2.2-r2"
CONTROL_POLICY = {
    "version": CONTROL_POLICY_VERSION,
    "decision": "设计-2026-10-25.md §2.2 and second independent semantic review; before real collection",
    "equivalent": {"rows": 120, "both_equivalent_to_old": True,
                   "same_matched_semantic_pair": True, "new_leaf_differences": 0,
                   "independent_semantic_novelty": False},
    "non_equivalent": {"rows": 240, "minimum_new_leaf_differences": 2,
                       "minimum_effective_factor_changes": 2,
                       "nonempty_applicability": True, "exceptions_must_be_active_members": True,
                       "forbid_semantic_pair_twins_with": ["shared", "counterfactual"]},
    "matching_scope": "same coarse label and recipe mechanism; fine subtypes are measured, not matched",
    "reference_partition": "both named reference lists are explicitly disjoint in every shared background",
    "literal_novelty": "report separately; fixture wording is not author novelty evidence",
    "cross_split": "pair/old/new/any-side semantic uniqueness unchanged",
    "causal_scope": "R2-R1 combines data organization, effective novelty, fine composition and authors; R3-R2 isolates the margin objective",
}

PROTOCOL = {
    "version": "conflict-protocol-v2.2-r2", "control_matching": CONTROL_POLICY,
    "study": "conflict strengthening and memory disclosure 2026-10-25 / conflict line",
    "base": {"repo_id": C.QWEN["repo_id"], "revision": C.QWEN["revision"], "dtype": "bfloat16"},
    "arm": "direct3",
    "labels": list(COARSE),
    "seeds": list(SEEDS),
    "updates": UPDATES, "micro": MICRO, "accum": ACCUM, "max_length": MAX_LENGTH,
    "eval_every": EVAL_EVERY,
    "lr": QWEN_SPEC["lr"], "weight_decay": QWEN_SPEC["weight_decay"], "clip_norm": QWEN_SPEC["clip_norm"],
    "lora": {"r": QWEN_SPEC["lora_r"], "alpha": QWEN_SPEC["lora_alpha"],
             "dropout": QWEN_SPEC["lora_dropout"], "target_modules": list(QWEN_SPEC["target_modules"])},
    "selection": "coarse macro-F1 on dev (legacy dev + new dev), ties keep the earliest",
    "save": "adapter + head only (no Adam, no base copy)",
    "dyad_slots_per_window": DYAD_SLOTS_PER_WINDOW,
    "margin": {"formula": "mean(relu(0.5 - (p_conflict(conflict) - p_conflict(nonconflict))))",
               "weight": MARGIN_WEIGHT, "applies_to": ["r3"]},
}

                                                                                        
    
                                                                                        

SPECS_DIR = HERE / "data" / "specs"
SPECS_PATH = SPECS_DIR / "specs.jsonl"
AUTHORS_DIR = HERE / "data" / "authors"
PROCESSED_DIR = HERE / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
ROUTES_PATH = HERE / "data" / "routes.json"
COMPOSITE_SPECS = SPECS_DIR / "composite_specs.jsonl"
PLANS_DIR = HERE / "plans"
RUNS_DIR = HERE / "runs"
AUDITS_DIR = HERE / "audits"
SELFTEST_DIR = HERE / "_selftest"
FIXTURE_DIR = SELFTEST_DIR / CONTROL_POLICY_VERSION
PARENT_PROCESSED = PARENT_EXP / "data" / "processed"

                                                                                        
              
                                                                                        

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                  separators=(",", ":")))


def sha256_file(path: "str | Path") -> str:
    return C.sha256_file(Path(path))


def optional_sha(path: "str | Path") -> "str | None":
    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def read_json(path: "str | Path") -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: "str | Path") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                               separators=(",", ":")) + "\n" for row in rows))


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(value.split())


                                                                                        
     
                                                                                        

def pair_sides(pair: Any) -> tuple[str, str]:
    return TRH.pair_sides(pair)


def pair_digest(row: dict) -> str:
    old, new = pair_sides(row["pair"])
    background = row["pair"].get("frame", {}).get("background", "")
    return sha256_text("\u241f".join([normalize_text(old), normalize_text(new),
                                      normalize_text(background)]))


def content_digest(row: dict) -> str:
    """v2.2：纯 old/new 归一化摘要（不含 background/JSON 包装），用于跨 split 与同 pair 异标。"""
    old, new = pair_sides(row["pair"])
    return sha256_text(normalize_text(old) + "\u241f" + normalize_text(new))


def new_text_digest(row: dict) -> str:
    old, new = pair_sides(row["pair"])
    del old
    return sha256_text(normalize_text(new))


def validate_new_row(row: Any, split: str) -> list[str]:
    """在父实验行契约之上追加：新数据来源、dyad 结构、phenomena 机制字段。"""
    errors = list(TRH.validate_row(row, split))
    if errors:
        return errors
    for key in NEW_ROW_KEYS:
        if key not in row:
            errors.append(f"missing new key {key}")
    if row.get("source") not in (SOURCE_NEW, FIXTURE_SOURCE):
        errors.append(f"source {row.get('source')!r} is not a conflict-line source")
    if row.get("label_origin") != LABEL_ORIGIN:
        errors.append(f"label_origin {row.get('label_origin')!r} != {LABEL_ORIGIN!r}")
    phenomena = row.get("phenomena")
    if not isinstance(phenomena, dict):
        errors.append("phenomena must be an object for new rows")
    else:
        for key in ("mechanism", "setting", "scenario", "variant"):
            if not isinstance(phenomena.get(key), str) or not phenomena[key]:
                errors.append(f"phenomena.{key} must be a non-empty string")
    dyad = row.get("dyad")
    pool = row_pool(row)
    matched = matched_window_id(row)
    if pool == POOL_CONTROL:
        if dyad is not None:
            errors.append("control row must not carry dyad")
        if not isinstance(matched, str) or not matched:
            errors.append("control row needs phenomena.matched_window_id")
    elif pool == POOL_COUNTERFACTUAL:
        if dyad is None:
            errors.append("counterfactual row must carry dyad")
        if matched:
            errors.append("counterfactual row must not carry matched_window_id")
    else:
        if dyad is not None:
            errors.append("shared row must not carry dyad")
        if matched:
            errors.append("shared row must not carry matched_window_id")
    if dyad is not None:
        if not isinstance(dyad, dict):
            errors.append("dyad must be null or an object")
        else:
            for key in ("dyad_id", "role", "partner_id", "factor"):
                if key not in dyad:
                    errors.append(f"dyad missing {key}")
            if dyad.get("role") not in ("conflict", "nonconflict"):
                errors.append(f"dyad role {dyad.get('role')!r} invalid")
            if row.get("label") == "disjoint" and dyad.get("role") != "conflict":
                errors.append("disjoint row must carry dyad.role=conflict")
            if row.get("label") != "disjoint" and dyad.get("role") != "nonconflict":
                errors.append("non-disjoint dyad row must carry dyad.role=nonconflict")
    return errors


def matched_window_id(row: dict) -> "str | None":
    return (row.get("phenomena") or {}).get(MATCHED_WINDOW_FIELD)


def row_has_conflict_side(row: dict) -> bool:
    return row.get("dyad") is not None and row["dyad"].get("role") == "conflict"


def row_pool(row: dict) -> str:
    pool = (row.get("phenomena") or {}).get("pool")
    if pool in POOLS:
        return pool
    return POOL_COUNTERFACTUAL if row.get("dyad") else POOL_SHARED


def family_balance(rows: Sequence[dict]) -> dict:
    counts = {name: 0 for name in COARSE}
    for row in rows:
        counts[row["coarse_label"]] += 1
    return counts


                                                                                        
                
                                                                                        

def _parent_rows(split: str) -> list[dict]:
    path = PARENT_PROCESSED / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing parent split: {path}")
    return read_jsonl(path)


def load_new_split(split: str, *, purpose: str, lock: "dict | None" = None) -> list[dict]:
    """新 split 的唯一入口；heldout 只允许 audit 与锁定终评。"""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = PROCESSED_DIR / f"{split}.jsonl"
    if split == "heldout":
        if purpose == "train":
            raise PermissionError("new heldout 未锁定：训练/选点/calibrate 一律不得读取")
        if purpose == "locked-final":
            if not isinstance(lock, dict):
                raise PermissionError("new heldout 解锁需要已校验的 final-lock")
            expected = (lock.get("data_sha256") or {}).get("new-heldout")
            actual = sha256_file(path)
            if not expected or expected != actual:
                raise PermissionError(f"new heldout SHA 与锁不符：锁 {str(expected)[:12]} 盘上 {actual[:12]}")
        elif purpose not in ("audit", "fixture"):
            raise PermissionError(f"new heldout 不允许 purpose={purpose!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    return read_jsonl(path)


def route_definition(route: str) -> dict:
    """路由定义是唯一真源：训练/评分/审计都从这里取，不允许各自拼数据。"""
    if route == "r1":
        return {"route": "r1", "role": "train", "new_splits": ("train",), "include_dyads": False,
                "legacy_splits": ("train",), "allowed_pools": (POOL_SHARED, POOL_CONTROL),
                "description": "既有 train 1001 + 新自然（shared + control；不碰 counterfactual）"}
    if route == "r2":
        return {"route": "r2", "role": "train", "new_splits": ("train",), "include_dyads": True,
                "legacy_splits": ("train",), "allowed_pools": (POOL_SHARED, POOL_COUNTERFACTUAL),
                "description": "既有 train 1001 + 新 shared + 新反事实（不碰 control）"}
    if route == "r3":
        return {"route": "r3", "role": "train", "new_splits": ("train",), "include_dyads": True,
                "legacy_splits": ("train",), "allowed_pools": (POOL_SHARED, POOL_COUNTERFACTUAL),
                "description": "同 r2 数据、窗口与初始化；CE + 0.2×margin"}
    if route == "dev":
        return {"route": "dev", "role": "dev", "new_splits": ("dev",), "include_dyads": True,
                "legacy_splits": ("dev",), "allowed_pools": POOLS,
                "description": "既有 dev 263 + 新 dev（自然 + dyad）"}
    if route == "old-train":
        return {"route": "old-train", "role": "regression", "new_splits": (), "legacy_splits": ("train",),
                "description": "旧训练集回归（只评分）"}
    if route == "old-dev":
        return {"route": "old-dev", "role": "regression", "new_splits": (), "legacy_splits": ("dev",),
                "description": "旧 dev 回归（只评分）"}
    if route == "old-calibration":
        return {"route": "old-calibration", "role": "locked-regression", "new_splits": (),
                "legacy_splits": ("calibration",), "description": "旧 calibration 锁后回归，不参与拟合/选点"}
    if route == "old-heldout":
        return {"route": "old-heldout", "role": "locked-regression", "new_splits": (),
                "legacy_splits": ("heldout",), "description": "旧 heldout 锁后回归，不是新终测"}
    if route == "dev-natural":
        return {"route": "dev-natural", "role": "dev", "new_splits": ("dev",), "include_dyads": False,
                "legacy_splits": ("dev",), "allowed_pools": (POOL_SHARED, POOL_CONTROL),
                "description": "R1 dev-best 用：既有 dev 263 + 新 dev 自然（shared+control）"}
    if route == "dev-counterfactual":
        return {"route": "dev-counterfactual", "role": "dev", "new_splits": ("dev",),
                "include_dyads": True, "legacy_splits": ("dev",),
                "allowed_pools": (POOL_SHARED, POOL_COUNTERFACTUAL),
                "description": "R2/R3 dev-best 用：既有 dev 263 + 新 dev shared + CF"}
    if route == "new-dev":
        return {"route": "new-dev", "role": "regression", "new_splits": ("dev",), "legacy_splits": (),
                "description": "新 dev 诊断（已参与 dev 选点，故只作诊断）"}
    if route == "new-heldout":
        return {"route": "new-heldout", "role": "locked-final", "new_splits": ("heldout",),
                "legacy_splits": (), "allowed_pools": POOLS,
                "description": "唯一新终测；锁后只评一次"}
    if route == "new-heldout-natural":
        return {"route": "new-heldout-natural", "role": "locked-final", "new_splits": ("heldout",),
                "legacy_splits": (), "allowed_pools": (POOL_SHARED, POOL_CONTROL),
                "description": "新终测的 pair 子集：自然 heldout（主指标）"}
    if route == "new-heldout-counterfactual":
        return {"route": "new-heldout-counterfactual", "role": "locked-final",
                "new_splits": ("heldout",), "legacy_splits": (),
                "allowed_pools": (POOL_COUNTERFACTUAL,),
                "description": "新终测的 pair 子集：反事实 heldout（dyad 双方同时判对率）"}
    raise KeyError(f"unknown route {route!r}")


def route_rows(route: str, *, purpose: str, lock: "dict | None" = None,
               manifest: "dict | None" = None) -> list[dict]:
    """按路由装配行；fixture 数据一律拒绝（只允许 selftest 显式放行）。"""
    definition = route_definition(route)
    rows: list[dict] = []
    for split in definition.get("legacy_splits", ()):
        rows.extend(_parent_rows(split))
    allowed = definition.get("allowed_pools")
    for split in definition.get("new_splits", ()):
        new_rows = load_new_split(split, purpose=purpose, lock=lock)
        if not definition.get("include_dyads", True):
            new_rows = [row for row in new_rows if row.get("dyad") is None]
        if allowed is not None:
            new_rows = [row for row in new_rows if row_pool(row) in allowed]
        rows.extend(new_rows)
    if manifest is not None and manifest.get("fixture") and purpose != "fixture":
        raise PermissionError("fixture 数据不得进入真实训练/评分路由")
    rows.sort(key=lambda row: (0 if row.get("source") == SOURCE_LEGACY else 1, str(row["id"])))
    return rows


def route_fingerprint(route: str, rows: Sequence[dict]) -> str:
    """只哈希行内容（id + pair digest + 标签）。r2/r3 同数据必须同指纹。"""
    del route
    payload = {"rows": [[row["id"], pair_digest(row), row["label"], row["coarse_label"]] for row in rows]}
    return sha256_json(payload)


def route_table(*, purpose: str = "audit", lock: "dict | None" = None,
                manifest: "dict | None" = None) -> dict:
    table = {}
    for route in ("r1", "r2", "r3", "dev", "old-train", "old-dev", "new-dev",
                  "old-calibration", "old-heldout", "new-heldout"):
        rows = route_rows(route, purpose=purpose, lock=lock, manifest=manifest)
        definition = route_definition(route)
        table[route] = {
            "role": definition["role"], "description": definition["description"],
            "n": len(rows), "fingerprint": route_fingerprint(route, rows),
            "coarse": family_balance(rows),
            "sources": sorted({row.get("source") for row in rows}),
            "families": len({row.get("family_id") for row in rows if row.get("family_id")}),
        }
    return table


                                                                                        
                             
                                                                                        

LOCK_SCHEMA = "conflict-final-lock-v1"


class LockError(RuntimeError):
    """锁缺失/过期/与盘上数据或 checkpoint 不符。"""


def checkpoint_file_shas(directory: "str | Path") -> dict:
    directory = Path(directory)
    return {item.relative_to(directory).as_posix(): sha256_file(item)
            for item in sorted(directory.rglob("*")) if item.is_file()}


def load_lock(path: "str | Path") -> dict:
    path = Path(path)
    if not path.is_file():
        raise LockError(f"missing final-lock: {path}")
    lock = read_json(path)
    if lock.get("schema") != LOCK_SCHEMA:
        raise LockError(f"final-lock schema {lock.get('schema')!r} != {LOCK_SCHEMA!r}")
    return lock


def composite_rows_path() -> Path:
    return HERE / "data" / "composite" / "final" / "rows.jsonl"


def composite_fingerprint(rows: Sequence[dict]) -> str:
    return sha256_json([[row["id"], pair_digest(row), row["coarse_label"]] for row in rows])


def route_fingerprint_any(route: str) -> "tuple[str, dict]":
    """复合集合不在 60 family 路由表里，读法与指纹在这里单独钉死。"""
    if route == "composite":
        path = composite_rows_path()
        if not path.is_file():
            raise FileNotFoundError(f"composite 尚未装配：{path}")
        rows = read_jsonl(path)
        return composite_fingerprint(rows), {"rows": len(rows), "path": str(path)}
    rows = route_rows(route, purpose="audit")
    return route_fingerprint(route, rows), {"rows": len(rows)}


def check_lock(lock: dict, *, route: str, run_dir: "str | Path | None" = None,
               checkpoint: "str | None" = None) -> dict:
    """校验：路由允许、行指纹、文件 SHA、（可选）checkpoint 文件 SHA。"""
    allowed = lock.get("locked_routes") or []
    if route not in allowed:
        raise LockError(f"route {route!r} 不在锁的 locked_routes={allowed}")
    actual_fingerprint, detail = route_fingerprint_any(route)
    expected = (lock.get("route_fingerprint") or {}).get(route)
    if expected != actual_fingerprint:
        raise LockError(f"route {route!r} 行指纹与锁不符")
    definition = (route_definition(route) if route != "composite"
                  else {"new_splits": (), "legacy_splits": ()})
    file_shas = {}
    for split in definition.get("new_splits", ()):
        path = PROCESSED_DIR / f"{split}.jsonl"
        file_shas[f"new-{split}"] = sha256_file(path)
    for split in definition.get("legacy_splits", ()):
        file_shas[f"old-{split}"] = sha256_file(PARENT_PROCESSED / f"{split}.jsonl")
    if route == "composite":
        file_shas["composite"] = sha256_file(composite_rows_path())
    for key, value in file_shas.items():
        if (lock.get("data_sha256") or {}).get(key) != value:
            raise LockError(f"{key} 文件 SHA 与锁不符")
    if run_dir is not None and checkpoint is not None:
        actual = checkpoint_file_shas(Path(run_dir) / "checkpoints" / checkpoint)
        expected_files = (lock.get("checkpoint_files_sha256") or {}).get(f"{route}:{checkpoint}")
        if expected_files is not None and expected_files != actual:
            raise LockError("checkpoint 文件 SHA 与锁不符")
    return {"route": route, "fingerprint": actual_fingerprint, "file_sha256": file_shas, **detail}


def lock_requirements(*, run: str, checkpoint: str = "best-dev") -> dict:
    """主 Agent 手工建锁时需要的全部字段；本函数不落盘。"""
    run_dir = RUNS_DIR / run
    lock = read_json(run_dir / "run-lock.json")
    rows_fingerprints = {}
    data_shas = {}
    for route in ("new-heldout", "new-heldout-natural", "new-heldout-counterfactual",
                  "old-heldout", "old-calibration"):
        rows = route_rows(route, purpose="audit")
        rows_fingerprints[route] = route_fingerprint(route, rows)
    for name, path in (("new-heldout", PROCESSED_DIR / "heldout.jsonl"),
                       ("old-heldout", PARENT_PROCESSED / "heldout.jsonl"),
                       ("old-calibration", PARENT_PROCESSED / "calibration.jsonl")):
        data_shas[name] = sha256_file(path)
    locked_routes = ["new-heldout", "new-heldout-natural", "new-heldout-counterfactual",
                     "old-heldout", "old-calibration"]
    if composite_rows_path().is_file():
        fingerprint, detail = route_fingerprint_any("composite")
        rows_fingerprints["composite"] = fingerprint
        data_shas["composite"] = sha256_file(composite_rows_path())
        locked_routes.append("composite")
    file_shas = {}
    for name in ("best-dev", "terminal"):
        directory = run_dir / "checkpoints" / name
        if directory.is_dir():
            file_shas[f"new-heldout:{name}"] = checkpoint_file_shas(directory)
    return {
        "schema": LOCK_SCHEMA,
        "created": "<YYYY-MM-DDTHH:MM:SS>",
        "run": run,
        "note": "hand-created by the main agent after dev review; scripts never create this file",
        "locked_routes": locked_routes,
        "route_fingerprint": rows_fingerprints,
        "data_sha256": data_shas,
        "checkpoint_files_sha256": file_shas,
        "run_lock_sha256": sha256_file(run_dir / "run-lock.json"),
        "plan_sha256": (lock.get("plan") or {}).get("sha256"),
        "init": lock.get("init"),
    }


def plan_path(route: str, seed: int, *, scope: str = "train") -> Path:
    return PLANS_DIR / f"{scope}-{route}-seed{int(seed)}.json"


PLAN_HASH_EXCLUDED = ("sha256", "route")


def plan_hash(plan: dict) -> str:
    """计划内容哈希：``route`` 只是文件名标签（r2/r3 共享同一窗），不参与哈希。"""
    return sha256_json({key: value for key, value in plan.items() if key not in PLAN_HASH_EXCLUDED})


def load_plan(path: "str | Path", *, verify: bool = True) -> dict:
    path = Path(path)
    plan = read_json(path)
    if verify and plan_hash(plan) != plan.get("sha256"):
        raise ValueError(f"plan SHA mismatch: {path}")
    return plan
