"""R1/R2/R3 等预算训练：共享 paired microbatch 窗口、CE 与反事实 margin、严格拒绝 CPU fit。

CLI
---
    python train_conflict_arms.py                          # 默认 selftest（伪模型 + 合成 logits）
    python train_conflict_arms.py selftest
    python train_conflict_arms.py plan --route r1|r2|r3 --seed 13|42
    python train_conflict_arms.py audit
    python train_conflict_arms.py smoke --route r3 --seed 13 [--device cuda:0]
    python train_conflict_arms.py train --route r1|r2|r3 --seed 13|42 [--device cuda:0] [--resume]
    python train_conflict_arms.py init-check --left r2-seed13 --right r3-seed13
    python train_conflict_arms.py plan-check --left r2 --right r3 --seed 13

协议（预注册，无命令行放宽开关）
-------------------------------
* Qwen3-0.6B + LoRA r16/alpha32 + 3 类 head；300 updates × micro2 × accum16 = 9600 instances；
  max_length 768、BF16、lr 1e-4 / wd .01 / clip 1.0；只保存 adapter+head。
* R1 = 既有 train + shared + control；R2 = 既有 train + shared + CF；R3 与 R2 同数据、同窗口、同初始化。
* R2/R3 共用同一份窗口计划（SHA 相同）：每窗 16 个 micro-batch，其中 4 个放整只 dyad，
  其余 12 个按固定规则两两配对自然行。R3 的额外项只加在窗口均值上：
  ``loss = mean_CE + 0.2 * mean(relu(0.5 - (p_conflict(conflict) - p_conflict(nonconflict))))``。
* ``train``/``smoke`` 在 CPU、无 CUDA、fixture 数据、缺计划或已存在 run 目录时直接拒绝。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import conflict_common as K

TRH = K.TRH
C = K.C
FT = K.FT

CONFLICT_INDEX = K.COARSE.index("conflict")
RUN_SCHEMA = "conflict-run-lock-v1"
PLAN_SCHEMA = "conflict-window-plan-v1"


                                                                                        
                    
                                                                                        

def _plan_code(route: str) -> str:
    return {"r1": "natural", "r2": "paired", "r3": "paired"}[route]


def _row_layout(rows: Sequence[dict]) -> dict:
    """行索引 ⇄ id、dyad 归组；conflict-bearing 与 E↔P dyad 都合法，半只 dyad 直接报错。"""
    index_of = {row["id"]: index for index, row in enumerate(rows)}
    if len(index_of) != len(rows):
        raise ValueError("duplicate row ids in route")
    dyads: dict[str, list[int]] = {}
    dyad_of: dict[int, str] = {}
    role_of: dict[int, str] = {}
    for index, row in enumerate(rows):
        dyad = row.get("dyad")
        if not dyad:
            continue
        dyads.setdefault(dyad["dyad_id"], []).append(index)
        dyad_of[index] = dyad["dyad_id"]
        role_of[index] = dyad["role"]
    conflict_dyads = set()
    for dyad_id, members in dyads.items():
        if len(members) != 2:
            raise ValueError(f"dyad {dyad_id} has {len(members)} rows in route")
        labels = {rows[index]["coarse_label"] for index in members}
        roles = {rows[index]["dyad"]["role"] for index in members}
        if "conflict" in labels:
            if roles != {"conflict", "nonconflict"} or sum(
                    rows[index]["dyad"]["role"] == "conflict" for index in members) != 1:
                raise ValueError(f"dyad {dyad_id} conflict roles {roles}")
            conflict_dyads.add(dyad_id)
        elif labels != {"compatible", "equivalent"} or roles != {"nonconflict"}:
            raise ValueError(f"dyad {dyad_id} is neither conflict-bearing nor E↔P: {labels}/{roles}")
    return {"index_of": index_of, "dyads": dyads, "dyad_of": dyad_of, "role_of": role_of,
            "conflict_dyads": conflict_dyads}


def _shuffled_cycle(pool: Sequence[Any], rng: np.random.Generator) -> Iterable[Any]:
    while True:
        order = list(pool)
        rng.shuffle(order)
        for item in order:
            yield item


def _paired_cycle(pool: Sequence[Any], rng: np.random.Generator) -> Iterable[list]:
    """自然行按固定规则两两配对：洗牌后顺次取两个；奇数尾巴与下一轮首项配对。"""
    carry: "Any | None" = None
    while True:
        order = list(pool)
        rng.shuffle(order)
        if carry is not None:
            if carry in order:
                order.remove(carry)
            order.insert(0, carry)
            carry = None
        cursor = 0
        while cursor + 1 < len(order):
            yield [order[cursor], order[cursor + 1]]
            cursor += 2
        if cursor < len(order):
            carry = order[cursor]


def _special_units(route: str, rows: Sequence[dict], layout: dict) -> tuple[dict, str]:
    """R1 用 matched control pair；R2/R3 用 dyad pair。键统一是 dyad_id。"""
    if route == "r1":
        units: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            matched = K.matched_window_id(row)
            if matched:
                units.setdefault(matched, []).append(index)
        if not units:
            raise ValueError("r1 has no matched control rows")
        for key, members in units.items():
            if len(members) != 2:
                raise ValueError(f"control slot {key} has {len(members)} rows")
            if any(rows[index]["dyad"] is not None for index in members):
                raise ValueError(f"control slot {key} contains a dyad row")
        return units, K.POOL_CONTROL
    return {key: list(members) for key, members in layout["dyads"].items()}, K.POOL_COUNTERFACTUAL


def build_plan(route: str, seed: int, rows: Sequence[dict], *,
               updates: int = K.UPDATES, special_slots: int = K.DYAD_SLOTS_PER_WINDOW) -> dict:
    """三臂同构：每窗 4 个 special micro-batch + 12 个自然配对 micro-batch。

    special 槽位由**同一份 dyad_id 洗牌序列**驱动：R1 放该 dyad 的 matched control pair，
    R2/R3 放该 dyad 本身，因此三臂槽位与曝光一一对应（R2/R3 计划 SHA 相同）。
    """
    if route not in K.TRAIN_ROUTES:
        raise ValueError(f"route {route!r} has no training plan")
    layout = _row_layout(rows)
    units, special_pool = _special_units(route, rows, layout)
    if special_slots * 2 > K.WINDOW_ROWS:
        raise ValueError("special slots exceed window")
    natural_ids = sorted(row["id"] for row in rows
                         if K.row_pool(row) == K.POOL_SHARED or row["source"] == K.SOURCE_LEGACY)
    if len(natural_ids) < 2 * (K.MICRO_PER_WINDOW - special_slots):
        raise ValueError(f"natural pool too small: {len(natural_ids)}")
    rng_special = np.random.default_rng(int(seed) * 1000003)
    rng_natural = np.random.default_rng(int(seed) * 1000003 + 7)
    keys = sorted(units)
    special_stream = _shuffled_cycle(keys, rng_special)
    natural_stream = _paired_cycle(natural_ids, rng_natural)
    windows, schedule = [], []
    exposure = {row["id"]: 0 for row in rows}
    for step in range(1, int(updates) + 1):
        chunks: list[list[int]] = []
        window_keys = []
        for _ in range(special_slots):
            key = next(special_stream)
            window_keys.append(key)
            chunks.append([int(index) for index in units[key]])
        for _ in range(K.MICRO_PER_WINDOW - special_slots):
            pair = next(natural_stream)
            chunks.append([layout["index_of"][row_id] for row_id in pair])
        flat = [index for chunk in chunks for index in chunk]
        if len(flat) != K.WINDOW_ROWS:
            raise ValueError(f"window {step} has {len(flat)} rows")
        special_here = {layout["dyad_of"][index] for index in flat if index in layout["dyad_of"]}
        control_here = {K.matched_window_id(rows[index]) for index in flat
                        if K.matched_window_id(rows[index])}
        if len(window_keys) != len(set(window_keys)):
            raise ValueError(f"window {step} repeats a special slot")
        windows.append(chunks)
        schedule.append(window_keys)
        for index in flat:
            exposure[rows[index]["id"]] += 1
    payload = {
        "schema": PLAN_SCHEMA, "route": route, "plan_code": "paired-window", "seed": int(seed),
        "updates": int(updates), "micro": K.MICRO, "accum": K.ACCUM,
        "rows_per_window": K.WINDOW_ROWS, "special_slots": int(special_slots),
        "special_pool": special_pool,
        "natural_slots": K.MICRO_PER_WINDOW - int(special_slots),
        "rows_sha256": K.sha256_json([row["id"] for row in rows]),
        "natural_ids_sha256": K.sha256_json(natural_ids),
        "n_rows": len(rows), "n_special": len(keys),
        "special_schedule": schedule,
        "repeat_instances": int(sum(len(flat) - len(set(flat))
                                    for flat in ([index for chunk in window for index in chunk]
                                                 for window in windows))),
        "windows": [[list(chunk) for chunk in window] for window in windows],
        "exposure": _exposure_summary(exposure, rows),
    }
    payload["sha256"] = K.plan_hash(payload)
    return payload


def _exposure_summary(exposure: dict, rows: Sequence[dict]) -> dict:
    pools = {row["id"]: K.row_pool(row) for row in rows}
    special = sorted({count for row_id, count in exposure.items()
                      if pools[row_id] in (K.POOL_CONTROL, K.POOL_COUNTERFACTUAL)})
    natural = sorted({count for row_id, count in exposure.items()
                      if pools[row_id] == K.POOL_SHARED or pools[row_id] not in
                      (K.POOL_CONTROL, K.POOL_COUNTERFACTUAL)})
    by_pool = {}
    for pool in K.POOLS:
        counts = sorted({count for row_id, count in exposure.items() if pools[row_id] == pool})
        by_pool[pool] = counts
    from collections import Counter
    origin = {row["id"]: (K.row_pool(row) if (row.get("phenomena") or {}).get("pool") in K.POOLS
                           else "existing_train") for row in rows}
    details = {}
    for pool in ("existing_train", *K.POOLS):
        values = [exposure[row_id] for row_id, name in origin.items() if name == pool]
        details[pool] = {"rows": len(values), "instances": sum(values),
                         "histogram": {str(count): n for count, n in sorted(Counter(values).items())},
                         "min": min(values) if values else None,
                         "max": max(values) if values else None,
                         "mean": sum(values) / len(values) if values else None}
    return {"by_pool": by_pool, "by_origin_pool": details,
            "special_rows": special, "natural_rows": natural,
            "max": max(exposure.values()), "min": min(exposure.values())}


def plan_windows(plan: dict, rows: Sequence[dict]) -> Iterable[tuple]:
    """把计划变成 ``FT.fit_joint`` 需要的 (step, epoch, chunks, epoch_end)。"""
    if plan["rows_sha256"] != K.sha256_json([row["id"] for row in rows]):
        raise ValueError("plan rows_sha256 != current route rows（路由/顺序变了，拒绝复用计划）")
    n = len(rows)
    consumed = 0
    previous_epoch = 0
    for step, window in enumerate(plan["windows"], start=1):
        chunks = [np.asarray(chunk, dtype=int) for chunk in window]
        consumed += sum(len(chunk) for chunk in chunks)
        epoch = (consumed + n - 1) // n
        epoch_end = consumed // n > previous_epoch
        previous_epoch = consumed // n
        yield step, epoch, chunks, epoch_end


def plan_identity(plan: dict) -> dict:
    return {"path": None, "sha256": plan["sha256"], "windows": len(plan["windows"]),
            "rows_per_window": plan["rows_per_window"], "special_slots": plan["special_slots"],
            "special_pool": plan["special_pool"], "n_rows": plan["n_rows"],
            "n_special": plan["n_special"], "exposure": plan["exposure"]}


def plan_checks(plan: dict, rows: Sequence[dict]) -> dict:
    """v2.2 计划结构：每窗 4 个 special micro-batch（同槽）+ 12 个自然配对 micro-batch。"""
    problems = []
    layout = _row_layout(rows)
    if len(plan["windows"]) != plan["updates"]:
        problems.append("window count")
    if plan["rows_sha256"] != K.sha256_json([row["id"] for row in rows]):
        problems.append("rows_sha256 mismatch")
    schedule = plan.get("special_schedule") or []
    if len(schedule) != len(plan["windows"]):
        problems.append("special_schedule length")
    expected_pool = plan["special_pool"]
    seen_special = set()
    for step, window in enumerate(plan["windows"], start=1):
        flat = [index for chunk in window for index in chunk]
        if len(flat) != plan["rows_per_window"]:
            problems.append(f"step {step}: {len(flat)} rows")
        for position, chunk in enumerate(window):
            if len(chunk) != 2:
                problems.append(f"step {step}: microbatch size {len(chunk)}")
                continue
            if position < plan["special_slots"]:
                pools = {K.row_pool(rows[index]) for index in chunk}
                if pools != {expected_pool}:
                    problems.append(f"step {step}: special pool {pools}")
                    continue
                if expected_pool == K.POOL_CONTROL:
                    keys = {K.matched_window_id(rows[index]) for index in chunk}
                    if len(keys) != 1 or None in keys:
                        problems.append(f"step {step}: control slot mixed")
                    else:
                        seen_special.add(next(iter(keys)))
                else:
                    keys = {layout["dyad_of"].get(index) for index in chunk}
                    if len(keys) != 1 or None in keys:
                        problems.append(f"step {step}: dyad slot mixed")
                    else:
                        seen_special.add(next(iter(keys)))
            else:
                pools = {K.row_pool(rows[index]) for index in chunk}
                if pools != {K.POOL_SHARED} and not all(
                        rows[index]["source"] == K.SOURCE_LEGACY or K.row_pool(rows[index]) == K.POOL_SHARED
                        for index in chunk):
                    problems.append(f"step {step}: natural slot pools {pools}")
        if step - 1 < len(schedule):
            keys_here = []
            for chunk in window[: plan["special_slots"]]:
                if expected_pool == K.POOL_CONTROL:
                    keys_here.append(K.matched_window_id(rows[chunk[0]]))
                else:
                    keys_here.append(layout["dyad_of"][chunk[0]])
            if keys_here != schedule[step - 1]:
                problems.append(f"step {step}: schedule mismatch")
    expected_keys = (set(layout["dyads"]) if expected_pool == K.POOL_COUNTERFACTUAL
                     else {K.matched_window_id(row) for row in rows if K.matched_window_id(row)})
    if seen_special != expected_keys:
        problems.append(f"scheduled {len(seen_special)}/{len(expected_keys)} special slots")
    return {"ok": not problems, "problems": problems[:10], "special_scheduled": len(seen_special),
            "windows": len(plan["windows"]), "special_pool": expected_pool,
            "exposure": plan["exposure"], "repeat_instances": plan.get("repeat_instances")}


                                                                                        
                                
                                                                                        

def margin_loss_factory(targets: Sequence[int], rows: Sequence[dict], device: torch.device,
                        *, weight: float = K.MARGIN_WEIGHT, margin: float = K.MARGIN):
    """返回 ``loss_fn(logits, chunk, window) -> (sum_loss, hits)``。

    尺度：``fit_joint`` 对每个 chunk 调 ``loss_fn`` 后做 ``(loss / len(window)).backward()``，
    因此要得到 ``mean_CE + weight * mean_dyad_margin``，chunk 内的 margin 项须乘
    ``weight * len(window) / n_dyads_in_window``。无 dyad 的窗口 strictly 退化为纯 CE。
    """
    targets = np.asarray(targets, dtype=np.int64)
    layout = _row_layout(rows)
    dyad_of, role_of = layout["dyad_of"], layout["role_of"]
    conflict_dyads = set(layout["conflict_dyads"])

    def loss_fn(logits, chunk, window):
        selected = torch.as_tensor(targets[chunk], device=device)
        ce = torch.nn.functional.cross_entropy(logits.float(), selected, reduction="sum")
        hits = (logits.detach().float().argmax(dim=1) == selected).sum()
        window_dyads = {dyad_of.get(int(index)) for index in window if int(index) in dyad_of}
        window_dyads.discard(None)
        window_dyads &= conflict_dyads                              
        if not window_dyads:
            return ce, hits
        pairs = []
        for left in range(len(chunk)):
            for right in range(left + 1, len(chunk)):
                first, second = int(chunk[left]), int(chunk[right])
                if first not in dyad_of or dyad_of[first] != dyad_of.get(second):
                    continue
                if dyad_of[first] not in conflict_dyads:
                    continue                                
                if role_of[first] == role_of[second]:
                    continue
                conflict_side = left if role_of[first] == "conflict" else right
                nonconflict_side = right if conflict_side == left else left
                pairs.append((conflict_side, nonconflict_side))
        if not pairs:
            return ce, hits
        probabilities = logits.float().softmax(dim=-1)[:, CONFLICT_INDEX]
        total = logits.new_zeros(())
        for conflict_side, nonconflict_side in pairs:
            gap = probabilities[conflict_side] - probabilities[nonconflict_side]
            total = total + torch.relu(margin - gap)
        scale = weight * len(window) / len(window_dyads)
        return ce + scale * total, hits

    return loss_fn


def window_loss(loss_fn, logits: torch.Tensor, window: Sequence[int],
                chunks: Sequence[Sequence[int]]) -> torch.Tensor:
    """复算 ``fit_joint`` 的窗口尺度：sum(chunk loss) / len(window)。"""
    total = None
    for chunk in chunks:
        index = np.asarray(chunk, dtype=int)
        part, _ = loss_fn(logits[index], index, np.asarray(window, dtype=int))
        total = part if total is None else total + part
    return total / len(window)


                                                                                        
    
                                                                                        

def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda":
        raise SystemExit("refuse: 训练只允许真实 CUDA；CPU fit 被显式禁止")
    if not torch.cuda.is_available():
        raise SystemExit("refuse: torch.cuda.is_available() 为 False，训练先阻塞而不是静默回退 CPU")
    return device


def _manifest() -> dict:
    if not K.MANIFEST_PATH.is_file():
        raise SystemExit(f"missing manifest: {K.MANIFEST_PATH}（先 assemble）")
    return K.read_json(K.MANIFEST_PATH)


def load_plan_for(route: str, seed: int, rows: Sequence[dict], *, create: bool = True) -> dict:
    path = K.plan_path(route, seed)
    if path.is_file():
        plan = K.load_plan(path)
        if plan["rows_sha256"] != K.sha256_json([row["id"] for row in rows]):
            raise SystemExit(f"{path} 与当前路由行不一致；删除旧计划后重新生成")
        return plan
    if not create:
        raise SystemExit(f"missing plan {path}; 先运行 plan --route {route} --seed {seed}")
    plan = build_plan(route, seed, rows)
    K.write_json_atomic(path, plan)
    return plan


def _snapshot(model) -> dict:
    return {"head_state_sha256": TRH.state_sha256(model.classifier.state_dict()),
            "lora_state_sha256": TRH.state_sha256(TRH.lora_state(model)),
            "base_state_sha256": TRH.frozen_base_sha256(model)}


def fit_route(*, route: str, seed: int, device: torch.device, run_dir: Path,
              updates: int = K.UPDATES, eval_every: int = K.EVAL_EVERY,
              smoke: bool = False, verbose: bool = True) -> dict:
    manifest = _manifest()
    if manifest.get("fixture"):
        raise SystemExit("refuse: manifest.fixture=true；fixture 数据不得训练")
    from build_conflict_data import check_spec_set, require_feasible_controls
    current_specs = K.read_jsonl(K.SPECS_PATH)
    check_spec_set(current_specs)
    require_feasible_controls(current_specs)
    if manifest.get("spec_sha256") != K.sha256_json(current_specs):
        raise SystemExit("refuse: dataset/spec SHA mismatch")
    train_rows = K.route_rows(route, purpose="train", manifest=manifest)
    dev_route = "dev-natural" if route == "r1" else "dev-counterfactual"
    dev_rows = K.route_rows(dev_route, purpose="dev", manifest=manifest)
    if route in ("r2", "r3"):
        if not any(row.get("dyad") for row in train_rows):
            raise SystemExit("refuse: r2/r3 训练集没有 dyad")
    else:
        if any(row.get("dyad") for row in train_rows):
            raise SystemExit("refuse: r1 训练集混入 dyad")
        if not any(K.row_pool(row) == K.POOL_CONTROL for row in train_rows):
            raise SystemExit("refuse: r1 训练集没有 matched control 行")
    plan = load_plan_for(route, seed, train_rows)
    checks = plan_checks(plan, train_rows)
    if not checks["ok"]:
        raise SystemExit(f"plan invalid: {checks['problems']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    built = TRH.build_arm("direct3", seed)
    model, tokenizer = built["model"], built["tokenizer"]
    init = _snapshot(model)
    base_before = init["base_state_sha256"]
    optimizer = C.build_optimizer("qwen3", model)
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    covered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not head_ids <= covered:
        raise SystemExit("refuse: optimizer 未覆盖 head")
    targets = TRH.targets_for(train_rows, "direct3")
    if route == "r3":
        loss_fn = margin_loss_factory(targets, train_rows, device)
    else:
        loss_fn = C.make_loss(targets, device)
    encode = TRH.default_encode(tokenizer, device, K.MAX_LENGTH)

    def evaluate_dev() -> dict:
        return TRH.evaluate_split(model=model, tokenizer=tokenizer, rows=dev_rows, arm="direct3",
                                  split="dev", device=device, max_length=K.MAX_LENGTH,
                                  bf16=True, encode=encode)["summary"]["coarse"]

    started = time.perf_counter()
    fitted = FT.fit_joint(
        model, tokenizer, [""] * len(train_rows), optimizer=optimizer,
        windows=plan_windows(plan, train_rows), planned_steps=int(updates),
        planned_instances=int(updates) * K.WINDOW_ROWS,
        max_length=K.MAX_LENGTH, device=device, bf16=True, loss_fn=loss_fn,
        evaluate=evaluate_dev, checkpoint_selection="best-dev", selection_metric="dev-macro-f1",
        patience=0, min_delta=0.0, eval_every_steps=int(eval_every), frozen=True,
        require_backbone_grad=True, clip_grad_norm=K.PROTOCOL["clip_norm"],
        encode_fn=lambda chunk: encode(chunk, train_rows),
    )
    best = TRH.best_dev_from_curve(fitted["history"])
    selection = fitted["selection"]
    if best is None:
        raise RuntimeError("no dev evaluation happened")
    if (int(selection["selected_step"]), float(selection["best_dev_value"])) != best:
        raise RuntimeError("dev-best tie handling drifted")
    base_after = TRH.frozen_base_sha256(model)
    if base_before != base_after:
        raise RuntimeError("frozen base weights changed during training")

    def save(name: str, state: dict, step: int, label: str) -> dict:
        model.load_state_dict(state, strict=True)
        meta = {"format": "qwen-lora", "num_labels": int(model.classifier.out_features),
                "step": int(step), "selection": label, "route": route, "seed": int(seed),
                "adapter_lora_sha256": TRH.state_sha256(TRH.lora_state(model)),
                "head_sha256": TRH.state_sha256(model.classifier.state_dict()),
                "protocol": K.PROTOCOL}
        return TRH.save_checkpoint(run_dir / "checkpoints" / name, model,
                                   model.classifier.state_dict(), meta)

    best_files = save("best-dev", fitted["continuation"]["best_state"],
                      int(selection["selected_step"]), "best-dev")
    terminal_files = save("terminal", fitted["continuation"]["terminal_state"],
                          int(fitted["updates"][-1]["step"]), "terminal")
    model.load_state_dict(fitted["continuation"]["best_state"], strict=True)
    dev_final = evaluate_dev()

    data = {
        "train": {"rows": len(train_rows), "sha256": K.route_fingerprint(route, train_rows)},
        "dev": {"route": dev_route, "rows": len(dev_rows),
                "sha256": K.route_fingerprint(dev_route, dev_rows)},
        "files": {split: K.sha256_file(K.PROCESSED_DIR / f"{split}.jsonl") for split in K.SPLITS},
    }
    lock = {
        "schema": RUN_SCHEMA, "created": K.utc_now(), "route": route, "seed": int(seed),
        "run": run_dir.name, "protocol": K.PROTOCOL, "data": data,
        "plan": plan_identity(plan),
        "init": init, "frozen_base_sha256_after": base_after,
        "checkpoints": {"best-dev": best_files, "terminal": terminal_files},
        "code_sha256": {name: K.sha256_file(Path(__file__).resolve().parent / name)
                        for name in ("conflict_common.py", "conflict_semantics.py",
                                     "build_conflict_data.py", "train_conflict_arms.py")},
        "budget": fitted["budget"],
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "parameter_devices_after_fit": sorted({str(p.device) for p in model.parameters()})},
        "seconds": round(time.perf_counter() - started, 3),
    }
    K.write_json_atomic(run_dir / "run-lock.json", lock)
    result = {
        "route": route, "seed": int(seed), "updates": int(updates), "dev_route": dev_route,
        "selection": {"selected_step": int(selection["selected_step"]),
                      "best_dev_macro_f1": float(selection["best_dev_value"]),
                      "terminal_step": int(fitted["updates"][-1]["step"])},
        "dev_final": dev_final,
        "history": fitted["history"],
        "budget": fitted["budget"],
        "loss": {"kind": "ce+margin" if route == "r3" else "ce",
                 "margin_weight": K.MARGIN_WEIGHT if route == "r3" else 0.0,
                 "margin": K.MARGIN if route == "r3" else None},
        "smoke": bool(smoke),
    }
    K.write_json_atomic(run_dir / "result.json", result)
    return {"run_dir": run_dir, "lock": lock, "result": result, "dev_final": dev_final,
            "plan": plan_identity(plan)}


def cmd_train(args) -> int:
    device = _device(args.device)
    if args.route not in K.TRAIN_ROUTES:
        raise SystemExit("r0 不训练：只复用既有 direct3 结果做评分")
    run_dir = K.RUNS_DIR / f"{args.route}-seed{args.seed}"
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run 目录已存在：{run_dir}；不覆盖既有结果（换目录或显式 --resume）")
    output = fit_route(route=args.route, seed=args.seed, device=device, run_dir=run_dir,
                       updates=K.UPDATES, eval_every=K.EVAL_EVERY, smoke=False)
    print(json.dumps({"command": "train", "route": args.route, "seed": args.seed,
                      "run": str(output["run_dir"]),
                      "selected_step": output["result"]["selection"]["selected_step"],
                      "dev_macro_f1": output["result"]["dev_final"]["macro_f1"],
                      "budget": output["result"]["budget"],
                      "plan": output["plan"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args) -> int:
    device = _device(args.device)
    run_dir = K.RUNS_DIR / "_smoke" / f"{args.route}-seed{args.seed}"
    if run_dir.exists():
        raise SystemExit(f"smoke 目录已存在：{run_dir}")
    output = fit_route(route=args.route, seed=args.seed, device=device, run_dir=run_dir,
                       updates=3, eval_every=3, smoke=True)
    lock = output["lock"]
    files = set()
    for entry in lock["checkpoints"]["best-dev"]["files"]:
        files.add(entry)
    if any(name.endswith(".safetensors") and "adapter" not in name for name in files):
        raise SystemExit(f"smoke save looks like a base copy: {sorted(files)}")
    print(json.dumps({"command": "smoke", "ok": True, "run": str(run_dir),
                      "files": sorted(files), "init": lock["init"],
                      "frozen_base_equal": lock["init"]["base_state_sha256"] == lock["frozen_base_sha256_after"],
                      "budget": lock["budget"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args) -> int:
    manifest = _manifest()
    if manifest.get("fixture"):
        raise SystemExit("refuse: fixture 数据没有训练计划")
    rows = K.route_rows(args.route, purpose="train", manifest=manifest)
    plan = build_plan(args.route, args.seed, rows)
    checks = plan_checks(plan, rows)
    if not checks["ok"]:
        raise SystemExit(f"plan invalid: {checks['problems']}")
    path = K.plan_path(args.route, args.seed)
    K.write_json_atomic(path, plan)
    print(json.dumps({"command": "plan", "path": str(path), "identity": plan_identity(plan),
                      "checks": checks}, ensure_ascii=False, indent=2))
    return 0


def cmd_plan_dry(args) -> int:
    """在已装配数据（可 fixture）上建三臂 300 窗计划，输出槽位/曝光/ SHA 对账；不建模型。"""
    fixture = K.FIXTURE_DIR / "processed"
    if not (fixture / "manifest.json").is_file():
        raise SystemExit(f"missing fixture bundle: {fixture}（先跑 fixture-bundle）")
    manifest = K.read_json(fixture / "manifest.json")
    original = K.PROCESSED_DIR
    K.PROCESSED_DIR = fixture
    try:
        report = {"fixture": manifest.get("fixture"), "arms": {}}
        plans = {}
        for route in K.TRAIN_ROUTES:
            rows = K.route_rows(route, purpose="fixture", manifest=manifest)
            plan = build_plan(route, args.seed, rows)
            plans[route] = plan
            report["arms"][route] = {
                "rows": len(rows), "n_special": plan["n_special"],
                "special_pool": plan["special_pool"], "sha256": plan["sha256"],
                "exposure": plan["exposure"], "checks": plan_checks(plan, rows),
            }
            K.write_json_atomic(K.FIXTURE_DIR / "plans" / f"{route}-seed{args.seed}.json", plan)
        report["r2_r3_sha_equal"] = plans["r2"]["sha256"] == plans["r3"]["sha256"]
        report["r1_r2_special_schedule_equal"] = (plans["r1"]["special_schedule"]
                                                   == plans["r2"]["special_schedule"])
        report["r1_r2_special_count_equal"] = plans["r1"]["n_special"] == plans["r2"]["n_special"]
        origin_r1 = plans["r1"]["exposure"]["by_origin_pool"]
        origin_r2 = plans["r2"]["exposure"]["by_origin_pool"]
        report["r1_r2_special_exposure_equal"] = origin_r1[K.POOL_CONTROL] == origin_r2[K.POOL_COUNTERFACTUAL]
        report["natural_exposure_equal"] = all(origin_r1[pool] == origin_r2[pool]
                                                for pool in ("existing_train", K.POOL_SHARED))
        report["origin_pool_counts_correct"] = all(
            plan["n_rows"] == 1289
            and plan["exposure"]["by_origin_pool"]["existing_train"]["rows"] == 1001
            and plan["exposure"]["by_origin_pool"][K.POOL_SHARED]["rows"] == 144
            for plan in plans.values())
        report["ok"] = (report["r2_r3_sha_equal"] and report["r1_r2_special_schedule_equal"]
                         and report["r1_r2_special_exposure_equal"] and report["natural_exposure_equal"]
                         and report["origin_pool_counts_correct"] and report["r1_r2_special_count_equal"]
                         and all(entry["checks"]["ok"] for entry in report["arms"].values()))
        print(json.dumps({"command": "plan-dry-run", **report}, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    finally:
        K.PROCESSED_DIR = original


def cmd_plan_check(args) -> int:
    manifest = _manifest()
    left = K.load_plan(K.plan_path(args.left, args.seed))
    right = K.load_plan(K.plan_path(args.right, args.seed))
    same = left["sha256"] == right["sha256"]
    print(json.dumps({"command": "plan-check", "left": left["route"], "right": right["route"],
                      "seed": args.seed, "same_sha256": same,
                      "left_sha256": left["sha256"], "right_sha256": right["sha256"],
                      "special_pool": [left["special_pool"], right["special_pool"]],
                      "special_schedule_equal": left.get("special_schedule") == right.get("special_schedule"),
                      "exposure_equal": left["exposure"] == right["exposure"],
                      "exposure": [left["exposure"], right["exposure"]]},
                     ensure_ascii=False, indent=2))
    return 0 if same else 1


def cmd_init_check(args) -> int:
    left = K.read_json(K.RUNS_DIR / args.left / "run-lock.json")
    right = K.read_json(K.RUNS_DIR / args.right / "run-lock.json")
    fields = ("head_state_sha256", "lora_state_sha256", "base_state_sha256")
    equal = all(left["init"][field] == right["init"][field] for field in fields)
    plan_equal = left["plan"]["sha256"] == right["plan"]["sha256"]
    data_equal = left["data"]["train"]["sha256"] == right["data"]["train"]["sha256"]
    print(json.dumps({"command": "init-check", "left": left["run"], "right": right["run"],
                      "init_equal": equal, "plan_equal": plan_equal, "train_data_equal": data_equal,
                      "init": {field: [left["init"][field], right["init"][field]] for field in fields}},
                     ensure_ascii=False, indent=2))
    return 0 if (equal and plan_equal and data_equal) else 1


def cmd_audit(args) -> int:
    manifest = _manifest()
    report = {"fixture": manifest.get("fixture"), "routes": {}, "problems": []}
    if manifest.get("fixture"):
        report["problems"].append("fixture manifest：训练被拒绝（这是预期行为）")
    for route in K.TRAIN_ROUTES:
        try:
            rows = K.route_rows(route, purpose="train", manifest=manifest)
        except PermissionError as error:
            report["routes"][route] = {"refused": str(error)}
            continue
        try:
            plan = load_plan_for(route, args.seed, rows, create=False)
        except SystemExit as error:
            report["routes"][route] = {"rows": len(rows), "missing_plan": str(error)}
            report["problems"].append(f"{route}: missing plan")
            continue
        checks = plan_checks(plan, rows)
        report["routes"][route] = {
            "rows": len(rows), "dyads": sum(1 for row in rows if row.get("dyad")),
            "plan": plan_identity(plan), "checks": checks,
            "fingerprint": K.route_fingerprint(route, rows),
        }
        if not checks["ok"]:
            report["problems"].extend(f"{route}: {problem}" for problem in checks["problems"])
    report["ok"] = not report["problems"]
    print(json.dumps({"command": "audit", **report}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


                                                                                        
                                  
                                                                                        

def _synthetic_rows(count_dyads: int = 2) -> list[dict]:
    rows = []
    for index in range(count_dyads):
        dyad_id = f"syn-d{index}"
        rows.append({"id": f"syn-d{index}-c", "cluster_id": f"syn-{index}", "family_id": f"syn-{index}",
                     "root_id": f"syn-{index}", "split": "train", "label": "disjoint",
                     "coarse_label": "conflict", "source": "selftest", "author": "fake",
                     "label_origin": "selftest", "visible_text": f"syn conflict {index}",
                     "dyad": {"dyad_id": dyad_id, "role": "conflict", "partner_id": f"syn-d{index}-n",
                              "factor": {"field": "new.action", "from": "require", "to": "prohibit",
                                         "from_render": "必须", "to_render": "不得"}}})
        rows.append({"id": f"syn-d{index}-n", "cluster_id": f"syn-{index}", "family_id": f"syn-{index}",
                     "root_id": f"syn-{index}", "split": "train", "label": "overlap_incomparable",
                     "coarse_label": "compatible", "source": "selftest", "author": "fake",
                     "label_origin": "selftest", "visible_text": f"syn nonconflict {index}",
                     "dyad": {"dyad_id": dyad_id, "role": "nonconflict", "partner_id": f"syn-d{index}-c",
                              "factor": {"field": "new.action", "from": "require", "to": "prohibit",
                                         "from_render": "必须", "to_render": "不得"}}})
    return rows


def _mixed_rows(dyads: int = 16, naturals: int = 16) -> list[dict]:
    """伪模型集成用：足够填满一个窗口的唯一行（dyad 与自然行混合）。"""
    rows = _synthetic_rows(dyads)
    labels = [("equivalent", "equivalent"), ("old_entails_new_strict", "compatible"),
              ("overlap_incomparable", "compatible"), ("disjoint", "conflict")]
    for index in range(naturals):
        fine, coarse = labels[index % len(labels)]
        rows.append({"id": f"syn-nat{index}", "family_id": f"syn-nat{index}",
                     "cluster_id": f"syn-nat{index}", "root_id": f"syn-nat{index}",
                     "split": "train", "label": fine, "coarse_label": coarse,
                     "label_origin": "selftest", "source": "selftest", "author": "fake",
                     "visible_text": f"syn natural {index}", "dyad": None})
    return rows


def _v22_rows(dyads: int = 4, shared: int = 24, legacy: int = 6) -> list[dict]:
    """v2.2 形状的合成行：conflict dyad / E↔P dyad / matched control / shared / legacy。"""
    rows: list[dict] = []
    for index in range(shared):
        fine, coarse = (("equivalent", "equivalent"), ("old_entails_new_strict", "compatible"),
                        ("disjoint", "conflict"))[index % 3]
        rows.append({"id": f"syn-shared{index}", "family_id": f"syn-s{index}",
                     "cluster_id": f"syn-s{index}", "root_id": f"syn-s{index}", "split": "train",
                     "label": fine, "coarse_label": coarse, "label_origin": "selftest",
                     "source": "selftest", "author": "fake", "dyad": None,
                     "phenomena": {"pool": K.POOL_SHARED}, "visible_text": f"shared {index}"})
    for index in range(legacy):
        rows.append({"id": f"syn-legacy{index}", "family_id": f"syn-l{index}",
                     "cluster_id": f"syn-l{index}", "root_id": f"syn-l{index}", "split": "train",
                     "label": "overlap_incomparable", "coarse_label": "compatible",
                     "label_origin": "selftest", "source": K.SOURCE_LEGACY, "author": None,
                     "dyad": None, "visible_text": f"legacy {index}"})
    for index in range(dyads):
        dyad_id = f"syn-dy{index}"
        e_to_p = index % 2 == 1
        if e_to_p:
            pairs = [("compatible", "nonconflict"), ("equivalent", "nonconflict")]
        else:
            pairs = [("conflict", "conflict"), ("compatible", "nonconflict")]
        for side, (label, role) in zip(("a", "b"), pairs):
            row_id = f"syn-d{index}-{side}"
            rows.append({
                "id": row_id, "family_id": f"syn-f{index}", "cluster_id": f"syn-f{index}",
                "root_id": f"syn-f{index}", "split": "train",
                "label": "disjoint" if label == "conflict" else
                         ("equivalent" if label == "equivalent" else "overlap_incomparable"),
                "coarse_label": label, "label_origin": "selftest", "source": "selftest",
                "author": "fake",
                "phenomena": {"pool": K.POOL_COUNTERFACTUAL, "dyad": dyad_id},
                "dyad": {"dyad_id": dyad_id, "role": role,
                         "partner_id": f"syn-d{index}-{'b' if side == 'a' else 'a'}",
                         "factor": {"field": "new.action", "from": "require", "to": "prohibit",
                                    "from_render": "必须", "to_render": "不得"}},
                "visible_text": f"dyad {index} {side}"})
                                                            
        for side, (label, _role) in zip(("a", "b"), pairs):
            rows.append({
                "id": f"syn-c{index}-{side}", "family_id": f"syn-f{index}",
                "cluster_id": f"syn-f{index}", "root_id": f"syn-c{index}-{side}",
                "split": "train",
                "label": "disjoint" if label == "conflict" else
                         ("equivalent" if label == "equivalent" else "overlap_incomparable"),
                "coarse_label": label, "label_origin": "selftest", "source": "selftest",
                "author": "fake", "dyad": None,
                "phenomena": {"pool": K.POOL_CONTROL, "matched_window_id": dyad_id,
                              "matched_side": side,
                              "matched_pair_id": f"syn-d{index}-{side}"},
                "visible_text": f"control {index} {side}"})
    return rows


def _logits(conflict_probability: float, nonconflict_probability: float) -> torch.Tensor:
    """构造 [conflict, nonconflict] 两行 logits，使 p_conflict 恰为给定值（其余概率均分）。"""
    def one(target: float) -> list[float]:
        rest = (1.0 - target) / (K.COARSE.index("conflict"))
        values = [rest] * len(K.COARSE)
        values[CONFLICT_INDEX] = target
        return [math.log(max(value, 1e-9)) for value in values]
    return torch.tensor([one(conflict_probability), one(nonconflict_probability)], dtype=torch.float32)


def _bad_loss_scales(targets, rows, device):
    """三种故意写错的损失，用来证明自检有牙齿。"""
    layout = _row_layout(rows)
    targets_array = np.asarray(targets, dtype=np.int64)

    def missing_scale(logits, chunk, window):
        selected = torch.as_tensor(targets_array[chunk], device=device)
        ce = torch.nn.functional.cross_entropy(logits.float(), selected, reduction="sum")
        probabilities = logits.float().softmax(dim=-1)[:, CONFLICT_INDEX]
        total = logits.new_zeros(())
        for left in range(len(chunk)):
            for right in range(left + 1, len(chunk)):
                first, second = int(chunk[left]), int(chunk[right])
                if first in layout["dyad_of"] and layout["dyad_of"][first] == layout["dyad_of"].get(second):
                    conflict_side = left if layout["role_of"][first] == "conflict" else right
                    nonconflict_side = right if conflict_side == left else left
                    total = total + torch.relu(K.MARGIN - (probabilities[conflict_side]
                                                          - probabilities[nonconflict_side]))
        return ce + K.MARGIN_WEIGHT * total, (logits.detach().argmax(dim=1) == selected).sum()

    def wrong_sign(logits, chunk, window):
        selected = torch.as_tensor(targets_array[chunk], device=device)
        ce = torch.nn.functional.cross_entropy(logits.float(), selected, reduction="sum")
        probabilities = logits.float().softmax(dim=-1)[:, CONFLICT_INDEX]
        total = logits.new_zeros(())
        window_dyads = {layout["dyad_of"].get(int(index)) for index in window} - {None}
        for left in range(len(chunk)):
            for right in range(left + 1, len(chunk)):
                first, second = int(chunk[left]), int(chunk[right])
                if first in layout["dyad_of"] and layout["dyad_of"][first] == layout["dyad_of"].get(second):
                    conflict_side = left if layout["role_of"][first] == "conflict" else right
                    nonconflict_side = right if conflict_side == left else left
                    total = total + torch.relu(K.MARGIN + (probabilities[conflict_side]
                                                          - probabilities[nonconflict_side]))
        scale = K.MARGIN_WEIGHT * len(window) / max(len(window_dyads), 1)
        return ce + scale * total, (logits.detach().argmax(dim=1) == selected).sum()

    def scale_by_rows(logits, chunk, window):
        selected = torch.as_tensor(targets_array[chunk], device=device)
        ce = torch.nn.functional.cross_entropy(logits.float(), selected, reduction="sum")
        probabilities = logits.float().softmax(dim=-1)[:, CONFLICT_INDEX]
        total = logits.new_zeros(())
        for left in range(len(chunk)):
            for right in range(left + 1, len(chunk)):
                first, second = int(chunk[left]), int(chunk[right])
                if first in layout["dyad_of"] and layout["dyad_of"][first] == layout["dyad_of"].get(second):
                    conflict_side = left if layout["role_of"][first] == "conflict" else right
                    nonconflict_side = right if conflict_side == left else left
                    total = total + torch.relu(K.MARGIN - (probabilities[conflict_side]
                                                          - probabilities[nonconflict_side]))
        return ce + K.MARGIN_WEIGHT * len(window) * total, (logits.detach().argmax(dim=1) == selected).sum()

    return {"missing_scale": missing_scale, "wrong_sign": wrong_sign, "scale_by_rows": scale_by_rows}


def run_selftest() -> dict:
    checks = []
    device = torch.device("cpu")
    rows = _synthetic_rows(2)
    targets = [K.COARSE.index(row["coarse_label"]) for row in rows]
    good = margin_loss_factory(targets, rows, device)
    window = [0, 1, 2, 3]
    chunks = [[0, 1], [2, 3]]

    def record(name: str, ok: bool, **extra) -> None:
        checks.append({"check": name, "ok": bool(ok), **extra})
        if not ok:
            raise AssertionError(f"{name}: {extra}")

                                             
    satisfied = _logits(0.9, 0.1)
    violated = _logits(0.1, 0.9)
    loss_sat, _ = good(satisfied, np.asarray([0, 1]), np.asarray(window))
    loss_vio, _ = good(violated, np.asarray([0, 1]), np.asarray(window))
    ce_only = torch.nn.functional.cross_entropy(satisfied, torch.tensor([2, 1]), reduction="sum")
    margin_sat = float(loss_sat - ce_only)
    ce_only_vio = torch.nn.functional.cross_entropy(violated, torch.tensor([2, 1]), reduction="sum")
    margin_vio = float(loss_vio - ce_only_vio)
    record("margin_zero_when_satisfied", abs(margin_sat) < 1e-6, margin=margin_sat)
    expected = K.MARGIN_WEIGHT * len(window) / 2 * (K.MARGIN + 0.8)
    record("margin_positive_when_violated", abs(margin_vio - expected) < 1e-4,
           margin=margin_vio, expected=expected)

                                           
    layout_rows = _synthetic_rows(2)
    split_window = [0, 1, 2, 3]
    partial, _ = good(satisfied[:1], np.asarray([0]), np.asarray(split_window))
    ce_partial = torch.nn.functional.cross_entropy(
        satisfied[:1], torch.tensor([2]), reduction="sum")
    record("no_partner_no_margin", abs(float(partial - ce_partial)) < 1e-6)

                                                        
    mismatched = _synthetic_rows(2)                                         
    mismatch_targets = [K.COARSE.index(row["coarse_label"]) for row in mismatched]
    mismatch_loss = margin_loss_factory(mismatch_targets, mismatched, device)
    mismatch_logits = _logits(0.1, 0.9)
    value, _ = mismatch_loss(mismatch_logits, np.asarray([0, 2]), np.asarray([0, 1, 2, 3]))
    ce_value = torch.nn.functional.cross_entropy(mismatch_logits, torch.tensor([2, 2]), reduction="sum")
    record("must_not_pair_mismatched_dyads", torch.allclose(value, ce_value),
           value=float(value.detach()), ce=float(ce_value.detach()))

                                   
    natural_rows = _synthetic_rows(0) + [
        {"id": "plain-1", "coarse_label": "compatible", "label": "overlap_incomparable",
         "dyad": None, "visible_text": "p1"},
        {"id": "plain-2", "coarse_label": "conflict", "label": "disjoint",
         "dyad": None, "visible_text": "p2"},
    ]
    plain_targets = [1, 2]
    plain_loss = margin_loss_factory(plain_targets, natural_rows, device)
    logits = _logits(0.4, 0.6).clone().requires_grad_(True)
    value, _ = plain_loss(logits, np.asarray([0, 1]), np.asarray([0, 1]))
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor(plain_targets), reduction="sum")
    record("no_dyad_window_is_pure_ce", torch.allclose(value, ce),
           value=float(value.detach()), ce=float(ce.detach()))
    grad_margin = torch.autograd.grad(value, logits)[0]
    grad_ce = torch.autograd.grad(ce, logits)[0]
    record("no_dyad_gradient_equal", torch.allclose(grad_margin, grad_ce))

                                                              
    violated_logits = torch.cat([_logits(0.1, 0.9), _logits(0.1, 0.9)], dim=0)
    effective = float(window_loss(good, violated_logits, window, chunks))
    probabilities = violated_logits.softmax(dim=-1)[:, CONFLICT_INDEX]
    expected_margin = (torch.relu(K.MARGIN - (probabilities[0] - probabilities[1]))
                       + torch.relu(K.MARGIN - (probabilities[2] - probabilities[3]))) / 2
    expected_ce = torch.nn.functional.cross_entropy(violated_logits, torch.tensor([2, 1, 2, 1]))
    record("window_scale_matches_recipe",
           abs(effective - float(expected_ce + K.MARGIN_WEIGHT * expected_margin)) < 1e-5,
           effective=effective, expected=float(expected_ce + K.MARGIN_WEIGHT * expected_margin))

                                                    
    trainable = _logits(0.2, 0.8).clone().requires_grad_(True)
    value, _ = good(trainable, np.asarray([0, 1]), np.asarray(window))
    gradient = torch.autograd.grad(value, trainable)[0]
    before = trainable.detach().softmax(dim=-1)[:, CONFLICT_INDEX]
    updated = trainable.detach() - 0.05 * gradient
    after = updated.softmax(dim=-1)[:, CONFLICT_INDEX]
    record("gradient_pushes_conflict_up",
           float(after[0]) > float(before[0]) and float(after[1]) < float(before[1]),
           before=[float(value) for value in before], after=[float(value) for value in after])

                            
    bad = _bad_loss_scales(targets, rows, device)
    def fails_scale(fn) -> bool:
        value = float(window_loss(fn, violated_logits, window, chunks))
        return abs(value - float(expected_ce + K.MARGIN_WEIGHT * expected_margin)) >= 1e-5
    record("must_fail_missing_scale", fails_scale(bad["missing_scale"]))
    record("must_fail_scale_by_rows", fails_scale(bad["scale_by_rows"]))
    value_bad, _ = bad["wrong_sign"](satisfied.clone(), np.asarray([0, 1]), np.asarray(window))
    record("must_fail_wrong_sign", float(value_bad) > float(loss_sat) + 1e-3,
           wrong_sign=float(value_bad), correct=float(loss_sat))

                                                                 
    synthetic = _v22_rows(dyads=6, shared=24, legacy=6)
    plan_r1 = build_plan("r1", 13, synthetic, updates=4)
    plan_r2 = build_plan("r2", 13, synthetic, updates=4)
    plan_r3 = build_plan("r3", 13, synthetic, updates=4)
    record("r2_r3_plans_identical", plan_r2["sha256"] == plan_r3["sha256"], sha=plan_r2["sha256"])
    record("r1_r2_special_schedule_identical",
           plan_r1["special_schedule"] == plan_r2["special_schedule"],
           r1=plan_r1["special_schedule"][:1], r2=plan_r2["special_schedule"][:1])
    record("r1_r2_special_exposure_identical",
           plan_r1["exposure"]["by_pool"][K.POOL_CONTROL] ==
           plan_r2["exposure"]["by_pool"][K.POOL_COUNTERFACTUAL],
           r1=plan_r1["exposure"]["by_pool"][K.POOL_CONTROL],
           r2=plan_r2["exposure"]["by_pool"][K.POOL_COUNTERFACTUAL])
    for name, plan in (("r1", plan_r1), ("r2", plan_r2)):
        plan_report = plan_checks(plan, synthetic)
        record(f"plan_structure_{name}", plan_report["ok"], problems=plan_report["problems"])
    record("plan_budget", all(len(plan["windows"]) == 4 for plan in (plan_r1, plan_r2))
           and all(sum(len(chunk) for chunk in window) == K.WINDOW_ROWS
                   for plan in (plan_r1, plan_r2) for window in plan["windows"]))
    fixture = K.FIXTURE_DIR / "processed"
    if (fixture / "manifest.json").is_file():
        manifest = K.read_json(fixture / "manifest.json")
        original = K.PROCESSED_DIR
        K.PROCESSED_DIR = fixture
        try:
            dev_natural = K.route_rows("dev-natural", purpose="fixture", manifest=manifest)
            dev_cf = K.route_rows("dev-counterfactual", purpose="fixture", manifest=manifest)
            dev_all = K.route_rows("dev", purpose="fixture", manifest=manifest)
        finally:
            K.PROCESSED_DIR = original
        record("dev_routes_arm_specific",
               len(dev_natural) == 407 and len(dev_cf) == 407 and len(dev_all) == 479
               and {row["family_id"] for row in dev_cf} == {row["family_id"] for row in dev_natural},
               natural=len(dev_natural), counterfactual=len(dev_cf), combined=len(dev_all))
        record("dev_natural_has_no_cf",
               all(row.get("dyad") is None for row in dev_natural))
        record("dev_cf_has_dyads", any(row.get("dyad") for row in dev_cf))
    record("r1_uses_control_slots",
           plan_r1["special_pool"] == K.POOL_CONTROL and plan_r1["n_special"] == 6
           and plan_r2["special_pool"] == K.POOL_COUNTERFACTUAL)
                                          
    real_manifest = K.read_json(K.MANIFEST_PATH) if K.MANIFEST_PATH.is_file() else None
    if real_manifest and not real_manifest.get("fixture"):
        rows_r1 = K.route_rows("r1", purpose="train", manifest=real_manifest)
        rows_r2 = K.route_rows("r2", purpose="train", manifest=real_manifest)
        record("three_arms_1289", len(rows_r1) == 1289 and len(rows_r2) == 1289,
               r1=len(rows_r1), r2=len(rows_r2))
        full_r1 = build_plan("r1", 13, rows_r1)
        full_r2 = build_plan("r2", 13, rows_r2)
        record("full_plan_schedule_equal",
               full_r1["special_schedule"] == full_r2["special_schedule"]
               and len(full_r1["windows"]) == K.UPDATES)
        record("full_plan_structure", plan_checks(full_r1, rows_r1)["ok"]
               and plan_checks(full_r2, rows_r2)["ok"])

                                           
    e2p_rows = [row for row in synthetic if (row.get("phenomena") or {}).get("dyad") == "syn-dy1"]
    conflict_rows = [row for row in synthetic
                     if (row.get("phenomena") or {}).get("dyad") == "syn-dy0"]
    window_rows = conflict_rows + e2p_rows
    window_targets = [K.COARSE.index(row["coarse_label"]) for row in window_rows]
    window_index = {row["id"]: position for position, row in enumerate(window_rows)}
    chunk = [window_index[row["id"]] for row in conflict_rows]
    e2p_chunk = [window_index[row["id"]] for row in e2p_rows]
    window_indices = list(range(len(window_rows)))
    mixed_loss = margin_loss_factory(window_targets, window_rows, device)
    logits = torch.cat([_logits(0.1, 0.9), _logits(0.05, 0.05)], dim=0)
    effective = float(window_loss(mixed_loss, logits, window_indices, [chunk, e2p_chunk]))
    probabilities = logits.softmax(dim=-1)[:, CONFLICT_INDEX]
    expected_margin = torch.relu(K.MARGIN - (probabilities[0] - probabilities[1]))
    expected_ce = torch.nn.functional.cross_entropy(
        logits, torch.tensor(window_targets), reduction="mean")
    record("e_to_p_not_counted_in_d",
           abs(effective - float(expected_ce + K.MARGIN_WEIGHT * expected_margin)) < 1e-5,
           effective=effective,
           expected=float(expected_ce + K.MARGIN_WEIGHT * expected_margin))
    e2p_only, _ = mixed_loss(logits[e2p_chunk], np.asarray(e2p_chunk), np.asarray(window_indices))
    e2p_ce = torch.nn.functional.cross_entropy(logits[e2p_chunk],
                                               torch.tensor([window_targets[i] for i in e2p_chunk]),
                                               reduction="sum")
    record("e_to_p_never_forced_conflict_side", torch.allclose(e2p_only, e2p_ce))

    grad_rows = _v22_rows(dyads=6, shared=24, legacy=6)
    grad_plan = build_plan("r3", 13, grad_rows, updates=4)
    grad_chunks = grad_plan["windows"][0]
    grad_window = [i for chunk in grad_chunks for i in chunk]
    grad_targets = [K.COARSE.index(row["coarse_label"]) for row in grad_rows]
    grad_logits = torch.randn((len(grad_rows), 3), generator=torch.Generator().manual_seed(919),
                              requires_grad=True)
    fn = margin_loss_factory(grad_targets, grad_rows, device)
    chunked_loss = window_loss(fn, grad_logits, grad_window, grad_chunks)
    hinges = []
    for chunk in grad_chunks[:K.DYAD_SLOTS_PER_WINDOW]:
        members = [i for i in chunk if grad_rows[i]["dyad"]["role"] == "conflict"]
        if members:
            conflict_index = members[0]
            other_index = next(i for i in chunk if i != conflict_index)
            p = grad_logits.softmax(dim=-1)[:, CONFLICT_INDEX]
            hinges.append(torch.relu(K.MARGIN - (p[conflict_index] - p[other_index])))
    direct_loss = torch.nn.functional.cross_entropy(
        grad_logits[grad_window], torch.tensor([grad_targets[i] for i in grad_window]))
    if hinges:
        direct_loss = direct_loss + K.MARGIN_WEIGHT * torch.stack(hinges).mean()
    chunked_gradient = torch.autograd.grad(chunked_loss, grad_logits, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(direct_loss, grad_logits)[0]
    record("full_32_window_loss_and_gradient", len(grad_window) == 32
           and torch.allclose(chunked_loss, direct_loss, atol=1e-6)
           and torch.allclose(chunked_gradient, direct_gradient, atol=1e-6),
           conflict_bearing_D=len(hinges), max_gradient_delta=float((chunked_gradient - direct_gradient).abs().max()))
    e_only_value, _ = mixed_loss(logits[e2p_chunk], np.asarray(e2p_chunk), np.asarray(e2p_chunk))
    record("E_P_only_D_zero_pure_CE", torch.allclose(e_only_value, e2p_ce))

    import tempfile
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conflict-train-selftest-", dir=K.SELFTEST_DIR) as temporary:
        root = Path(temporary)
        fake_rows = _mixed_rows(16, 40)
        built = TRH.build_arm("direct3", 13, builder=lambda seed: TRH.build_fake_model("direct3", seed))
        model, tokenizer = built["model"], built["tokenizer"]
        init_two = TRH.build_arm("direct3", 13, builder=lambda seed: TRH.build_fake_model("direct3", seed))
        record("init_same_seed",
               TRH.state_sha256(model.classifier.state_dict())
               == TRH.state_sha256(init_two["model"].classifier.state_dict()))
        other = TRH.build_arm("direct3", 42, builder=lambda seed: TRH.build_fake_model("direct3", seed))
        record("init_differs_by_seed",
               TRH.state_sha256(model.classifier.state_dict())
               != TRH.state_sha256(other["model"].classifier.state_dict()))
        encode = TRH.default_encode(tokenizer, device, 64)
        before_sha = TRH.state_sha256(model.state_dict())
        before = TRH.evaluate_split(model=model, tokenizer=tokenizer, rows=fake_rows, arm="direct3",
                                    split="dev", device=device, max_length=64, bf16=False,
                                    encode=encode)
        meta = {"format": "fake-full", "num_labels": 3, "selection": "untrained-selftest",
                "full_state_sha256": before_sha}
        TRH.save_checkpoint(root / "checkpoints" / "untrained", model, {}, meta)
        rebuilt = TRH.build_arm("direct3", 13, builder=lambda seed: TRH.build_fake_model("direct3", seed))
        reloaded = TRH.load_checkpoint(rebuilt["model"], root / "checkpoints" / "untrained")
        after = TRH.evaluate_split(model=rebuilt["model"], tokenizer=rebuilt["tokenizer"],
                                   rows=fake_rows, arm="direct3", split="dev", device=device,
                                   max_length=64, bf16=False, encode=encode)
        delta = float(np.abs(after["coarse_probabilities"] - before["coarse_probabilities"]).max())
        record("untrained_fake_save_reload", reloaded["format"] == "fake-full" and delta <= 1e-5
               and before_sha == TRH.state_sha256(model.state_dict()), delta=delta, updates=0)

                      
    refused = False
    try:
        _device("cpu")
    except SystemExit as error:
        refused = "CPU" in str(error) or "cuda" in str(error)
    record("must_refuse_cpu_fit", refused)

    print(json.dumps({"selftest": True, "ok": True, "checks": checks}, ensure_ascii=False, indent=2))
    return {"ok": True, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=lambda args: (run_selftest(), 0)[1])
    plan = sub.add_parser("plan")
    plan.add_argument("--route", required=True, choices=K.TRAIN_ROUTES)
    plan.add_argument("--seed", type=int, required=True, choices=K.SEEDS)
    plan.set_defaults(func=cmd_plan)
    audit = sub.add_parser("audit")
    audit.add_argument("--seed", type=int, default=13, choices=K.SEEDS)
    audit.set_defaults(func=cmd_audit)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--route", required=True, choices=K.TRAIN_ROUTES)
    smoke.add_argument("--seed", type=int, default=13, choices=K.SEEDS)
    smoke.add_argument("--device", default="cuda:0")
    smoke.set_defaults(func=cmd_smoke)
    train = sub.add_parser("train")
    train.add_argument("--route", required=True, choices=K.TRAIN_ROUTES)
    train.add_argument("--seed", type=int, required=True, choices=K.SEEDS)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", action="store_true")
    train.set_defaults(func=cmd_train)
    init_check = sub.add_parser("init-check")
    init_check.add_argument("--left", required=True)
    init_check.add_argument("--right", required=True)
    init_check.set_defaults(func=cmd_init_check)
    plan_dry = sub.add_parser("plan-dry-run")
    plan_dry.add_argument("--seed", type=int, default=13, choices=K.SEEDS)
    plan_dry.set_defaults(func=cmd_plan_dry)
    plan_check = sub.add_parser("plan-check")
    plan_check.add_argument("--left", required=True)
    plan_check.add_argument("--right", required=True)
    plan_check.add_argument("--seed", type=int, default=13, choices=K.SEEDS)
    plan_check.set_defaults(func=cmd_plan_check)
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        return (run_selftest(), 0)[1]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
