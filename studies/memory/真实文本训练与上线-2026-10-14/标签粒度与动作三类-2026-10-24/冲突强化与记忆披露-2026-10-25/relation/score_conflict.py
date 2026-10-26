"""冲突识别线评分面：原始训练回归、旧集回归、新 heldout、复合集合整组路由。

CLI
---
    python score_conflict.py                          # 默认 selftest（不建模型、不读 heldout）
    python score_conflict.py routes
    python score_conflict.py score --run r2-seed13 --route dev [--checkpoint best-dev|terminal] [--device cuda:0]
    python score_conflict.py score --run r2-seed13 --route new-heldout --unlock-final runs/final-lock.json
    python score_conflict.py score --external-run <DIR> --route old-train [--device cuda:0]
    python score_conflict.py compare --left r2-seed13 --right r3-seed13 --route new-heldout \
                                     --unlock-final runs/final-lock.json [--bootstrap 1000]
    python score_conflict.py verify --run r2-seed13 --route dev [--device cuda:0]
    python score_conflict.py lock-template --run r2-seed13

纪律
----
* 指标、coarse 聚合、checkpoint 重载全部走 ``train_relation_heads``（唯一实现，不在本文件复制）。
* ``new-heldout`` / ``old-heldout`` / ``old-calibration`` 必须通过手工 final-lock：路由允许、
  行指纹、文件 SHA、checkpoint 文件 SHA 任一不符即拒绝。
* 本文件从不创建锁；``lock-template`` 只把主 Agent 需要写入锁的字段打印出来。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import conflict_common as K

TRH = K.TRH
C = K.C

LOCKED_ROUTES = ("new-heldout", "new-heldout-natural", "new-heldout-counterfactual",
                 "old-heldout", "old-calibration", "composite")
BOOTSTRAP_ROUNDS = 1000
SCORE_SCHEMA = "conflict-score-v2.2"


                                                                                        
      
                                                                                        

def resolve_run(run: "str | None", external: "str | None") -> dict:
    if bool(run) == bool(external):
        raise SystemExit("必须且只能给 --run 或 --external-run 之一")
    if external:
        directory = Path(external).resolve()
        lock = K.read_json(directory / "run-lock.json")
        return {"run": directory.name, "directory": directory, "lock": lock,
                "seed": int(lock.get("identity", {}).get("config", {}).get("seed")
                            or lock.get("seed") or 13)}
    directory = K.RUNS_DIR / run
    if not directory.is_dir():
        raise SystemExit(f"missing run directory: {directory}")
    lock = K.read_json(directory / "run-lock.json")
    return {"run": run, "directory": directory, "lock": lock, "seed": int(lock["seed"])}


def load_model(entry: dict, *, checkpoint: str, device):
    built = TRH.build_arm("direct3", entry["seed"])
    model, tokenizer = built["model"], built["tokenizer"]
    info = TRH.load_checkpoint(model, entry["directory"] / "checkpoints" / checkpoint)
    model.to(device)
    return model, tokenizer, info


def encode_for(tokenizer, device):
    return TRH.default_encode(tokenizer, device, K.MAX_LENGTH)


                                                                                        
       
                                                                                        

def metric_block(gold: np.ndarray, pred: np.ndarray, probabilities: np.ndarray,
                 rows: Sequence[dict]) -> dict:
    fine = np.asarray([K.FINE.index(row["label"]) for row in rows], dtype=int)
    strict = np.isin(fine, [K.FINE.index("old_entails_new_strict"),
                            K.FINE.index("new_entails_old_strict")])
    return TRH.metric_block(np.asarray(gold, dtype=int), np.asarray(pred, dtype=int),
                            np.asarray(probabilities, dtype=np.float64), list(K.COARSE),
                            equivalent=K.COARSE.index("equivalent"),
                            conflict=K.COARSE.index("conflict"), strict_mask=strict)


def per_family_table(rows: Sequence[dict], gold: np.ndarray, pred: np.ndarray) -> dict:
    table = {}
    for index, row in enumerate(rows):
        key = row.get("family_id") or row.get("cluster_id") or "unknown"
        entry = table.setdefault(key, {"n": 0, "correct": 0, "conflict_gold": 0, "conflict_hit": 0})
        entry["n"] += 1
        entry["correct"] += int(pred[index] == gold[index])
        if gold[index] == K.COARSE.index("conflict"):
            entry["conflict_gold"] += 1
            entry["conflict_hit"] += int(pred[index] == K.COARSE.index("conflict"))
    for entry in table.values():
        entry["accuracy"] = entry["correct"] / entry["n"]
        entry["conflict_recall"] = (None if entry["conflict_gold"] == 0
                                    else entry["conflict_hit"] / entry["conflict_gold"])
    return table


def dyad_table(rows: Sequence[dict], gold: np.ndarray, pred: np.ndarray,
               probabilities: np.ndarray) -> dict:
    """分开统计 conflict-bearing dyad 与 E↔P dyad；后者没有 conflict 侧，不强行指派。"""
    groups = {}
    for index, row in enumerate(rows):
        dyad = row.get("dyad")
        if not dyad:
            continue
        group = groups.setdefault(dyad["dyad_id"], {"sides": [], "ids": []})
        group["sides"].append({
            "role": dyad["role"],
            "gold": K.COARSE[int(gold[index])], "pred": K.COARSE[int(pred[index])],
            "p_conflict": float(probabilities[index][K.COARSE.index("conflict")]),
        })
        group["ids"].append(row["id"])
    summary = {"n": len(groups), "conflict_dyads": 0, "e_to_p_dyads": 0,
               "both_labels_correct": 0, "conflict_dyads_both_labels_correct": 0,
               "conflict_detected_and_other_not_conflict": 0,
               "conflict_side_correct": 0, "other_label_correct": 0,
               "other_not_conflict": 0, "e_to_p_both_labels_correct": 0,
               "margin_satisfied": 0}
    for dyad_id, group in groups.items():
        conflict = [side for side in group["sides"] if side["role"] == "conflict"]
        other = [side for side in group["sides"] if side["role"] == "nonconflict"]
        if len(group["sides"]) != 2:
            raise ValueError(f"incomplete dyad {dyad_id}")
        exact = int(all(side["pred"] == side["gold"] for side in group["sides"]))
        summary["both_labels_correct"] += exact
        if len(conflict) == 1 and len(other) == 1 and conflict[0]["gold"] == "conflict" \
                and other[0]["gold"] != "conflict":
            summary["conflict_dyads"] += 1
            summary["conflict_dyads_both_labels_correct"] += exact
            summary["conflict_side_correct"] += int(conflict[0]["pred"] == "conflict")
            summary["other_label_correct"] += int(other[0]["pred"] == other[0]["gold"])
            summary["other_not_conflict"] += int(other[0]["pred"] != "conflict")
            summary["conflict_detected_and_other_not_conflict"] += int(
                conflict[0]["pred"] == "conflict" and other[0]["pred"] != "conflict")
            summary["margin_satisfied"] += int(
                conflict[0]["p_conflict"] - other[0]["p_conflict"] >= K.MARGIN)
        elif not conflict and len(other) == 2 \
                and {side["gold"] for side in other} == {"equivalent", "compatible"}:
            summary["e_to_p_dyads"] += 1
            summary["e_to_p_both_labels_correct"] += exact
        else:
            raise ValueError(f"invalid roles or labels in dyad {dyad_id}")
    denominators = {
        "both_labels_correct": summary["n"],
        "conflict_dyads_both_labels_correct": summary["conflict_dyads"],
        "conflict_detected_and_other_not_conflict": summary["conflict_dyads"],
        "conflict_side_correct": summary["conflict_dyads"],
        "other_label_correct": summary["conflict_dyads"],
        "other_not_conflict": summary["conflict_dyads"],
        "e_to_p_both_labels_correct": summary["e_to_p_dyads"],
        "margin_satisfied": summary["conflict_dyads"],
    }
    for name, denominator in denominators.items():
        summary[name + "_rate"] = None if denominator == 0 else summary[name] / denominator
    summary["denominators"] = denominators
    return summary


def composite_table(rows: Sequence[dict], gold: np.ndarray, pred: np.ndarray) -> dict:
    groups = {}
    for index, row in enumerate(rows):
        key = (row.get("phenomena") or {}).get("composite_group") or row.get("family_id")
        groups.setdefault(key, []).append(index)
    table = {"groups": len(groups), "all_correct": 0, "conflict_groups": 0,
             "conflict_groups_detected": 0, "group_rows": 0}
    for key, indices in groups.items():
        table["group_rows"] += len(indices)
        correct = all(pred[index] == gold[index] for index in indices)
        table["all_correct"] += int(correct)
        has_conflict = any(gold[index] == K.COARSE.index("conflict") for index in indices)
        if has_conflict:
            table["conflict_groups"] += 1
            detected = all(pred[index] == K.COARSE.index("conflict")
                           for index in indices if gold[index] == K.COARSE.index("conflict"))
            table["conflict_groups_detected"] += int(detected)
    table["all_correct_rate"] = (None if not groups else table["all_correct"] / len(groups))
    table["conflict_group_detection"] = (None if not table["conflict_groups"]
                                         else table["conflict_groups_detected"] / table["conflict_groups"])
    return table


                                                                                        
    
                                                                                        

def score_rows(*, rows: Sequence[dict], model, tokenizer, device, split_label: str) -> dict:
    encode = encode_for(tokenizer, device)
    evaluation = TRH.evaluate_split(model=model, tokenizer=tokenizer, rows=list(rows), arm="direct3",
                                    split=split_label, device=device, max_length=K.MAX_LENGTH,
                                    bf16=True, encode=encode)
    coarse = evaluation["coarse_probabilities"]
    gold = np.asarray([K.COARSE.index(row["coarse_label"]) for row in rows], dtype=int)
    pred = coarse.argmax(axis=1)
    summary = metric_block(gold, pred, coarse, rows)
    result = {
        "schema": SCORE_SCHEMA, "split": split_label, "n": len(rows),
        "coarse": summary,
        "per_family": per_family_table(rows, gold, pred),
        "strata": {field: per_family_table([
            dict(row, family_id=(row.get("phenomena") or {}).get(field, "unknown")) for row in rows],
            gold, pred) for field in ("setting", "recipe")},
        "dyad": dyad_table(rows, gold, pred, coarse),
        "predictions": evaluation["predictions"],
    }
    if any((row.get("phenomena") or {}).get("composite_group") for row in rows):
        result["composite"] = composite_table(rows, gold, pred)
    return result


def route_purpose(route: str) -> str:
    return "locked-final" if route in LOCKED_ROUTES else "audit"


def cmd_score(args) -> int:
    device = _cuda(args.device)
    entry = resolve_run(args.run, args.external_run)
    if args.route in LOCKED_ROUTES:
        if not args.unlock_final:
            raise SystemExit(f"route {args.route} 需要 --unlock-final（手工 final-lock）")
        lock = K.load_lock(args.unlock_final)
        run_lock_sha = K.sha256_file(entry["directory"] / "run-lock.json")
        if lock.get("run_lock_sha256") and lock["run_lock_sha256"] != run_lock_sha:
            raise SystemExit("final-lock 的 run_lock_sha256 与盘上 run-lock.json 不符")
        K.check_lock(lock, route=args.route, run_dir=entry["directory"], checkpoint=args.checkpoint)
        rows = load_rows_for_scoring(args.route, lock=lock)
    else:
        rows = load_rows_for_scoring(args.route, lock=None)
    model, tokenizer, reload_info = load_model(entry, checkpoint=args.checkpoint, device=device)
    result = score_rows(rows=rows, model=model, tokenizer=tokenizer, device=device,
                        split_label=args.route)
    result.update({"run": entry["run"], "route": args.route, "checkpoint": args.checkpoint,
                   "reload": reload_info, "unlock_final": bool(args.unlock_final)})
    out_dir = entry["directory"] / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.route}-{args.checkpoint}.json"
    K.write_json_atomic(out, result)
    K.write_jsonl_atomic(out_dir / f"{args.route}-{args.checkpoint}-predictions.jsonl",
                         result["predictions"])
    print(json.dumps({"command": "score", "run": entry["run"], "route": args.route,
                      "n": result["n"], "coarse": result["coarse"],
                      "dyad": result["dyad"], "composite": result.get("composite"),
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


def _cuda(name: str):
    import torch
    device = torch.device(name)
    if device.type != "cuda":
        raise SystemExit("refuse: 评分也走真实 CUDA；CPU 只允许 selftest")
    if not torch.cuda.is_available():
        raise SystemExit("refuse: torch.cuda.is_available() 为 False")
    return device


def load_rows_for_scoring(route: str, *, lock: "dict | None") -> list[dict]:
    if route == "composite":
        path = K.composite_rows_path()
        if not path.is_file():
            raise SystemExit("复合集合尚未装配：data/composite/final/rows.jsonl 不存在（先 specs+author+assemble）")
        return K.read_jsonl(path)
    manifest = K.read_json(K.MANIFEST_PATH) if K.MANIFEST_PATH.is_file() else None
    return K.route_rows(route, purpose=route_purpose(route), lock=lock, manifest=manifest)


                                                                                        
                         
                                                                                        

def _paired_bootstrap(gold: np.ndarray, left_pred: np.ndarray, right_pred: np.ndarray,
                      rows: Sequence[dict], *, rounds: int, seed: int = 20261025,
                      cluster_ids=None, kind: str = "family") -> dict:
    if not rows or rounds < 2:
        raise ValueError("bootstrap requires rows and at least two rounds")
    clusters = list(cluster_ids) if cluster_ids is not None else [
        row.get("family_id") or row["id"] for row in rows]
    if len(clusters) != len(rows):
        raise ValueError("cluster ids length mismatch")
    unique = sorted(set(clusters))
    index_by_cluster = {name: np.asarray([i for i, value in enumerate(clusters) if value == name])
                        for name in unique}
    rng = np.random.default_rng(int(seed))
    metrics = ("macro_f1", "accuracy", "conflict_recall", "false_equivalence")
    series = {label: {name: [] for name in metrics} for label in ("left", "right", "delta")}
    for _ in range(int(rounds)):
        picked = rng.choice(len(unique), size=len(unique), replace=True)
        indices = np.concatenate([index_by_cluster[unique[position]] for position in picked])
        blocks = {}
        for label, predictions in (("left", left_pred), ("right", right_pred)):
            blocks[label] = TRH.metric_block(
                gold[indices], predictions[indices], np.eye(len(K.COARSE))[predictions[indices]],
                list(K.COARSE), equivalent=K.COARSE.index("equivalent"),
                conflict=K.COARSE.index("conflict"), strict_mask=np.zeros(len(indices), dtype=bool))
        for name in metrics:
            values = {label: (blocks[label][name]["value"] if isinstance(blocks[label][name], dict)
                              else blocks[label][name]) for label in ("left", "right")}
            for label in ("left", "right"):
                series[label][name].append(values[label])
            series["delta"][name].append(None if any(value is None for value in values.values())
                                         else values["right"] - values["left"])
    def interval(values):
        return None if not values else [float(np.percentile(values, 2.5)),
                                        float(np.percentile(values, 97.5))]
    summary = {}
    for name in metrics:
        valid = {label: [v for v in series[label][name] if v is not None]
                 for label in series}
        deltas = valid["delta"]
        sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
        summary[name] = {
            "delta_ci95": interval(deltas), "delta_sd": sd,
            "retrospective_mdes_pp_approx": None if sd is None else 2.802 * sd * 100,
            "left_ci95": interval(valid["left"]), "right_ci95": interval(valid["right"]),
            "valid_rounds": len(deltas), "undefined_rounds": rounds - len(deltas),
        }
    return {"kind": kind, "clusters": len(unique), "rounds": int(rounds), "seed": int(seed),
            "independent_rows_assumed": False, "metrics": summary}


def compare_runs(*, left: dict, right: dict, rows: Sequence[dict],
                 rounds: int = BOOTSTRAP_ROUNDS, specs=None) -> dict:
    from conflict_diagnostics import semantic_clusters
    gold = np.asarray([K.COARSE.index(row["coarse_label"]) for row in rows], dtype=int)
    def aligned(result):
        predictions = result["predictions"]
        by_id = {row["id"]: row for row in predictions}
        if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in rows}:
            raise ValueError("prediction ids mismatch or duplicated")
        return np.asarray([K.COARSE.index(by_id[row["id"]]["pred_coarse"]) for row in rows])
    left_pred, right_pred = aligned(left), aligned(right)
    flips = {"left_only_correct": 0, "right_only_correct": 0, "both_correct": 0, "both_wrong": 0}
    per_example = []
    for index, row in enumerate(rows):
        left_hit, right_hit = left_pred[index] == gold[index], right_pred[index] == gold[index]
        name = ("both_correct" if left_hit and right_hit else "left_only_correct" if left_hit
                else "right_only_correct" if right_hit else "both_wrong")
        flips[name] += 1
        per_example.append({"id": row["id"], "family_id": row.get("family_id"),
                            "gold": K.COARSE[int(gold[index])],
                            "left": K.COARSE[int(left_pred[index])],
                            "right": K.COARSE[int(right_pred[index])], "outcome": name})
    block = {}
    for name, predictions in (("left", left_pred), ("right", right_pred)):
        block[name] = TRH.metric_block(gold, predictions, np.eye(len(K.COARSE))[predictions],
                                       list(K.COARSE), equivalent=K.COARSE.index("equivalent"),
                                       conflict=K.COARSE.index("conflict"),
                                       strict_mask=np.zeros(len(gold), dtype=bool))
    point = {"left": {}, "right": {}, "delta": {}}
    for name in ("macro_f1", "accuracy", "conflict_recall", "false_equivalence"):
        values = {side: block[side][name]["value"] if isinstance(block[side][name], dict)
                  else block[side][name] for side in ("left", "right")}
        for side in ("left", "right"):
            point[side][name] = values[side]
        point["delta"][name] = (None if any(value is None for value in values.values())
                                else float(values["right"] - values["left"]))
    if specs is None:
        specs = K.read_jsonl(K.SPECS_PATH) if K.SPECS_PATH.is_file() else []
    dependencies = semantic_clusters(rows, specs)
    point.update({"flips": flips, "per_example": per_example, "n": len(gold),
                  "bootstrap_family": _paired_bootstrap(gold, left_pred, right_pred, rows, rounds=rounds),
                  "bootstrap_semantic_cluster": _paired_bootstrap(
                      gold, left_pred, right_pred, rows, rounds=rounds,
                      cluster_ids=dependencies["ids"], kind="family_plus_semantic_components"),
                  "semantic_dependencies": dependencies})
    return point


def cmd_compare(args) -> int:
    device = _cuda(args.device)
    if args.route not in LOCKED_ROUTES:
        raise SystemExit("compare 只用于锁后路由（new-heldout/old-heldout/old-calibration/composite）")
    if not args.unlock_final:
        raise SystemExit("compare 需要 --unlock-final")
    lock = K.load_lock(args.unlock_final)
    rows = load_rows_for_scoring(args.route, lock=lock)
    results = {}
    for label, run in (("left", args.left), ("right", args.right)):
        entry = resolve_run(run, None)
        K.check_lock(lock, route=args.route, run_dir=entry["directory"], checkpoint=args.checkpoint)
        model, tokenizer, _ = load_model(entry, checkpoint=args.checkpoint, device=device)
        scored = score_rows(rows=rows, model=model, tokenizer=tokenizer, device=device,
                            split_label=args.route)
        results[label] = {"run": run, **scored}
    comparison = compare_runs(left=results["left"], right=results["right"], rows=rows, rounds=args.bootstrap)
    out = K.RUNS_DIR / f"compare-{args.route}-{args.left}-vs-{args.right}.json"
    K.write_json_atomic(out, {"route": args.route, "left": results["left"]["coarse"],
                              "right": results["right"]["coarse"], **comparison})
    print(json.dumps({"command": "compare", "route": args.route,
                      "delta": comparison["delta"], "flips": comparison["flips"],
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args) -> int:
    """同一 checkpoint 重新加载后逐题概率/argmax 必须与已存评分一致。"""
    device = _cuda(args.device)
    entry = resolve_run(args.run, args.external_run)
    stored_path = entry["directory"] / "scores" / f"{args.route}-{args.checkpoint}.json"
    if not stored_path.is_file():
        raise SystemExit(f"missing stored score: {stored_path}")
    stored = K.read_json(stored_path)
    rows = load_rows_for_scoring(args.route, lock=(K.load_lock(args.unlock_final)
                                                   if args.unlock_final else None))
    model, tokenizer, _ = load_model(entry, checkpoint=args.checkpoint, device=device)
    fresh = score_rows(rows=rows, model=model, tokenizer=tokenizer, device=device,
                       split_label=args.route)
    delta, mismatches = TRH.compare_predictions(
        {row["id"]: row for row in stored["predictions"]}, fresh["predictions"])
    print(json.dumps({"command": "verify", "run": entry["run"], "route": args.route,
                      "max_probability_delta": delta, "argmax_mismatches": mismatches,
                      "ok": mismatches == 0 and delta <= 1e-6}, ensure_ascii=False, indent=2))
    return 0 if (mismatches == 0 and delta <= 1e-6) else 1


def cmd_routes(args) -> int:
    manifest = K.read_json(K.MANIFEST_PATH) if K.MANIFEST_PATH.is_file() else None
    table = {}
    for route in ("r1", "r2", "r3", "dev", "old-train", "old-dev", "new-dev",
                  "old-calibration", "old-heldout", "new-heldout",
                  "new-heldout-natural", "new-heldout-counterfactual",
                  "dev-natural", "dev-counterfactual"):
        try:
            rows = K.route_rows(route, purpose="audit", manifest=manifest)
            table[route] = {"n": len(rows), "fingerprint": K.route_fingerprint(route, rows),
                            "role": K.route_definition(route)["role"]}
        except Exception as error:
            table[route] = {"error": str(error)[:80]}
    composite = K.composite_rows_path()
    table["composite"] = {"n": len(K.read_jsonl(composite)) if composite.is_file() else 0,
                          "role": "policy", "built": composite.is_file()}
    print(json.dumps({"command": "routes", "routes": table}, ensure_ascii=False, indent=2))
    return 0


def cmd_lock_template(args) -> int:
    print(json.dumps(K.lock_requirements(run=args.run, checkpoint=args.checkpoint),
                     ensure_ascii=False, indent=2))
    return 0


                                                                                        
                             
                                                                                        

def _synthetic_scored_rows() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows, gold, pred = [], [], []
    plan = [("disjoint", "conflict", "conflict"), ("overlap_incomparable", "compatible", "compatible"),
            ("equivalent", "equivalent", "equivalent"), ("old_entails_new_strict", "compatible", "conflict"),
            ("disjoint", "conflict", "compatible"), ("disjoint", "conflict", "conflict")]
    for index, (fine, coarse, predicted) in enumerate(plan):
        rows.append({"id": f"syn-{index}", "family_id": f"fam-{index % 3}", "label": fine,
                     "coarse_label": coarse,
                     "dyad": ({"dyad_id": f"dy-{index // 2}",
                               "role": "conflict" if coarse == "conflict" else "nonconflict"}
                              if index < 4 else None)})
        gold.append(K.COARSE.index(coarse))
        pred.append(K.COARSE.index(predicted))
                                         
    for offset, (fine, coarse) in enumerate((("equivalent", "equivalent"),
                                             ("old_entails_new_strict", "compatible"))):
        rows.append({"id": f"syn-e2p-{offset}", "family_id": "fam-e2p", "label": fine,
                     "coarse_label": coarse,
                     "dyad": {"dyad_id": "dy-e2p", "role": "nonconflict",
                              "partner_id": f"syn-e2p-{1 - offset}"}})
        gold.append(K.COARSE.index(coarse))
        pred.append(K.COARSE.index("equivalent" if offset == 0 else "conflict"))
    return rows, np.asarray(gold), np.asarray(pred)


def run_selftest() -> dict:
    checks = []

    def record(name: str, ok: bool, **extra) -> None:
        checks.append({"check": name, "ok": bool(ok), **extra})
        if not ok:
            raise AssertionError(f"{name}: {extra}")

    rows, gold, pred = _synthetic_scored_rows()
    probabilities = np.eye(len(K.COARSE))[pred]
    block = metric_block(gold, pred, probabilities, rows)
    record("metrics_hand_computed",
           block["conflict_recall"]["value"] == 2 / 3 and block["conflict_recall"]["n"] == 3
           and block["false_equivalence"]["value"] == 0.0
           and abs(block["accuracy"] - 5 / 8) < 1e-9,
           conflict_recall=block["conflict_recall"], accuracy=block["accuracy"])
    record("accuracy_is_not_macro_f1", abs(block["macro_f1"] - block["accuracy"]) > 1e-6,
           macro_f1=block["macro_f1"], accuracy=block["accuracy"])

                                                  
    composite_rows = []
    for group in range(2):
        for position in range(2):
            composite_rows.append({
                "id": f"cfx-{group}-{position}", "family_id": f"cfx-{group}",
                "label": "disjoint" if position == 0 else "equivalent",
                "coarse_label": "conflict" if position == 0 else "equivalent",
                "phenomena": {"composite_group": f"cfx-{group}"}})
    composite_gold = np.asarray([2, 0, 2, 0])
    composite_pred = np.asarray([2, 0, 2, 1])           
    table = composite_table(composite_rows, composite_gold, composite_pred)
    record("composite_group_all_correct",
           table["groups"] == 2 and table["all_correct"] == 1
           and table["all_correct_rate"] == 0.5
           and table["conflict_group_detection"] == 1.0, table=table)

    dyad = dyad_table(rows, gold, pred, probabilities)
    record("dyad_table", dyad["n"] == 3 and dyad["conflict_dyads"] == 1
           and dyad["e_to_p_dyads"] == 2 and dyad["both_labels_correct"] == 1
           and dyad["conflict_detected_and_other_not_conflict"] == 1
           and dyad["e_to_p_both_labels_correct"] == 0, dyad=dyad)

    import tempfile
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conflict-score-selftest-", dir=K.SELFTEST_DIR) as temporary:
        root = Path(temporary)
        processed = root / "processed"
        processed.mkdir()
        for split in K.SPLITS:
            K.write_jsonl_atomic(processed / f"{split}.jsonl", [])
        original_processed = K.PROCESSED_DIR
        original_parent = K.PARENT_PROCESSED
        K.PROCESSED_DIR = processed
        K.PARENT_PROCESSED = root / "parent"
        (root / "parent").mkdir()
        for split in ("train", "dev", "calibration", "heldout"):
            K.write_jsonl_atomic(root / "parent" / f"{split}.jsonl", [])
        try:
            lock = {"schema": K.LOCK_SCHEMA, "locked_routes": ["new-heldout"],
                    "route_fingerprint": {"new-heldout": "deadbeef"},
                    "data_sha256": {"new-heldout": K.sha256_file(processed / "heldout.jsonl")}}
            refused = 0
            for candidate in ({}, dict(lock, locked_routes=["old-heldout"]),
                              dict(lock, route_fingerprint={"new-heldout": "wrong"}),
                              dict(lock, data_sha256={"new-heldout": "0" * 64})):
                try:
                    K.check_lock(candidate, route="new-heldout")
                except K.LockError:
                    refused += 1
            record("must_fail_lock_variants", refused == 4, refused=refused)
            correct = dict(lock, route_fingerprint={"new-heldout": K.route_fingerprint(
                "new-heldout", K.route_rows("new-heldout", purpose="audit"))})
            K.check_lock(correct, route="new-heldout")
            record("lock_accepts_correct", True)
        finally:
            K.PROCESSED_DIR = original_processed
            K.PARENT_PROCESSED = original_parent

                                                         
    left = {"predictions": [{"id": row["id"], "pred_coarse": K.COARSE[int(p)]} for row, p in zip(rows, pred)]}
    right = {"predictions": [{"id": row["id"], "pred_coarse": K.COARSE[int(p)]} for row, p in zip(rows, pred)]}
    same = compare_runs(left=left, right=right, rows=rows)
    record("compare_identical_zero_effect",
           same["flips"]["left_only_correct"] == 0 and same["flips"]["right_only_correct"] == 0
           and same["delta"]["macro_f1"] == 0.0)
    flipped = list(pred)
    flipped[0] = K.COARSE.index("compatible")
    right_bad = {"predictions": [{"id": row["id"], "pred_coarse": K.COARSE[int(p)]}
                                 for row, p in zip(rows, flipped)]}
    comparison = compare_runs(left=left, right=right_bad, rows=rows)
    record("compare_detects_regression",
           comparison["flips"]["left_only_correct"] == 1
           and comparison["delta"]["conflict_recall"] < 0,
           flips=comparison["flips"], delta=comparison["delta"]["conflict_recall"])

    from conflict_selftest import run_score_checks
    run_score_checks(sys.modules[__name__], record)
    print(json.dumps({"selftest": True, "ok": True, "checks": checks}, ensure_ascii=False, indent=2))
    return {"ok": True, "checks": checks}


def cmd_mdes_prospective(args) -> int:
    from conflict_power import prospective_mdes
    specs = K.read_jsonl(K.SPECS_PATH)
    result = prospective_mdes(specs)
    result["spec_file_sha256"] = K.sha256_file(K.SPECS_PATH)
    if args.out:
        K.write_json_atomic(Path(args.out), result)
    print(json.dumps({"command": "mdes-prospective", **result}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=lambda args: (run_selftest(), 0)[1])
    sub.add_parser("routes").set_defaults(func=cmd_routes)
    power = sub.add_parser("mdes-prospective")
    power.add_argument("--out", default=None)
    power.set_defaults(func=cmd_mdes_prospective)
    score = sub.add_parser("score")
    score.add_argument("--run", default=None)
    score.add_argument("--external-run", default=None)
    score.add_argument("--route", required=True)
    score.add_argument("--checkpoint", default="terminal", choices=("terminal", "best-dev"),
                       help="主因果比较用 terminal-300（设计 v2 §3.2）；best-dev 只作次要路径")
    score.add_argument("--unlock-final", default=None)
    score.add_argument("--device", default="cuda:0")
    score.set_defaults(func=cmd_score)
    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--route", required=True)
    compare.add_argument("--checkpoint", default="terminal", choices=("terminal", "best-dev"))
    compare.add_argument("--unlock-final", required=True)
    compare.add_argument("--device", default="cuda:0")
    compare.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ROUNDS)
    compare.set_defaults(func=cmd_compare)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", default=None)
    verify.add_argument("--external-run", default=None)
    verify.add_argument("--route", required=True)
    verify.add_argument("--checkpoint", default="terminal", choices=("terminal", "best-dev"))
    verify.add_argument("--unlock-final", default=None)
    verify.add_argument("--device", default="cuda:0")
    verify.set_defaults(func=cmd_verify)
    template = sub.add_parser("lock-template")
    template.add_argument("--run", required=True)
    template.add_argument("--checkpoint", default="terminal", choices=("terminal", "best-dev"))
    template.set_defaults(func=cmd_lock_template)
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        return (run_selftest(), 0)[1]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
