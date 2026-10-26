"""统一指标、生产基线、研究性阈值、锁定终评与严格重载核验。

本文件是 ``train_relation_heads.py`` 的评分面：所有指标、coarse 聚合、阈值规则、
final-lock 校验与 heldout 通道都只有一份实现（``import train_relation_heads`` 复用），
本文件不再复制第二套训练或指标框架。

CLI
---
    python score_and_verify.py                       # 默认离线 selftest（不建模型）
    python score_and_verify.py selftest
    python score_and_verify.py metrics --run <arm>-seed<seed> [--checkpoint best-dev|terminal]
                                      [--split train|dev|calibration] [--device cuda:0]
    python score_and_verify.py baseline --split dev|calibration|heldout [--unlock-final PATH]
                                       [--device cpu]
    python score_and_verify.py calibrate --run <arm>-seed<seed> [--device cuda:0]
    python score_and_verify.py final-eval --run <arm>-seed<seed> --unlock-final runs/final-lock.json
                                        [--show-requirements] [--device cuda:0]
    python score_and_verify.py verify --run <arm>-seed<seed> [--unlock-final PATH] [--device cuda:0]

纪律
----
* ``metrics``/``calibrate`` 只读 train/dev/calibration；heldout 一行都不读。
* ``baseline`` 用生产 bundle ``repo/artifacts/relation-confidence-20261023/bundle-v2``，
  走 ``probe_runtime.py`` 的 direct 加载方式（``model_worker._relation_import_paths`` +
  ``load_runtime_artifact``），并在**子进程**里执行，保证解析到 bundle 自带的冻结源码而不是
  repo 现行代码。身份必须是 ``v3-c5-seed13`` 且与 release summary（``bundle-build.json``）
  的 ``meta_sha256`` / ``bundle_manifest_sha256`` / ``weights_sha256`` 全部对齐。
* ``heldout`` 只有 ``final-eval`` 与 ``verify --unlock-final`` 能读，且必须通过
  ``runs/final-lock.json``（主 Agent 手工创建）。本文件从不创建锁；``--show-requirements``
  只把该锁应包含的内容打印到 stdout。
* 阈值只在 calibration 上选，heldout 只做报告，绝不回选。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5] / 'source'


def _bundle_worker(argv: list[str]) -> int:
    """子进程内按生产 direct 路径加载 bundle；先于本模块任何 repo import 执行。"""
    import hashlib
    import importlib.util

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    bundle = Path(args.bundle).resolve()
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location(
        "relation_worker_reference", Path(REPO) / "src" / "dataset_recommender" / "retrieval" / "model_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._relation_import_paths(bundle / "source")

    import torch

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from research.memory_relations.semantic_artifacts import load_runtime_artifact

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    scorer, meta, payload = load_runtime_artifact(bundle / "runtime",
                                                  expected_meta_sha256=manifest["meta_sha256"],
                                                  device=args.device)
    del payload
    items = [json.loads(line) for line in
             Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [item["pair"] for item in items]
    declarations = [item["pair"].get("frame", {}).get("mode", "same_domain") == "same_domain"
                    for item in items]
    if not all(declarations):
        raise ValueError("baseline only accepts pairs with the existing same_domain declaration")
    prepared = scorer.prepare_inputs(pairs, role_independent_frame=True)
    probabilities = scorer.predict_proba(pairs, role_independent_frame=True)
    rows = []
    for item, view, values in zip(items, prepared, probabilities):
        rendered = view["directions"]["forward"]["text"]
        rows.append({
            "id": item["id"],
            "probabilities": [float(value) for value in values],
            "complete": bool(view["complete"]),
            "requires_llm_fallback": bool(view.get("requires_llm_fallback", not view["complete"])),
            "full_visible_token_lengths": view.get("full_visible_token_lengths"),
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "rendered_matches_visible": rendered == item.get("visible_text"),
            "rendered_normalized_matches_visible": " ".join(rendered.split()) == " ".join(
                str(item.get("visible_text", "")).split()),
        })
    output = {
        "model_id": manifest.get("model_id"),
        "meta_sha256": manifest.get("meta_sha256"),
        "classes": list(scorer.classes),
        "runtime_classes": list(meta.get("classes") or scorer.classes),
        "device": args.device,
        "role_independent_frame": True,
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 3),
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_worker_ok": True, "rows": len(rows), "seconds": output["seconds"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_bundle-worker":
    raise SystemExit(_bundle_worker(sys.argv[2:]))

import numpy as np              
import torch              
from typing import Any, Sequence              

import train_relation_heads as T              

FT = T.FT
C = T.C

BASELINE_ROOT = REPO / "artifacts" / "relation-confidence-20261023"
BUNDLE = BASELINE_ROOT / "bundle-v2"
RELEASE = BASELINE_ROOT / "bundle-build.json"
BASELINE_OUT = EXP / "baseline"
EXPECTED_MODEL_ID = "v3-c5-seed13"
EXPECTED_META_SHA256 = "f600b2d46fa41f0fc18d697b7b1f139a3cabb457340f89ff30547bb89c39d742"
PROBABILITY_TOLERANCE = 1e-5
SUMMARY_TOLERANCE = 1e-6
GATE_THRESHOLD = 0.8
INCOMPLETE_ID_OUTPUT_LIMIT = 20


                                                                                        
                        
                                                                                        

def bf16_for(device: torch.device) -> bool:
    return bool(T.QWEN_SPEC["bf16"]) and device.type == "cuda"


def gate_row(coarse: np.ndarray, gold: np.ndarray, threshold: float = GATE_THRESHOLD) -> dict:
    row = T.gate_row(coarse, gold, threshold)
    row["note"] = ("research-only max-prob accept gate; fitted on calibration, "
                   "reported on heldout without re-selection")
    row["confusion_at_acceptance"] = {
        "classes": list(T.COARSE),
        "matrix": acceptance_table(coarse, gold, threshold),
    }
    return row


def acceptance_table(coarse: np.ndarray, gold: np.ndarray, threshold: float) -> list[list[int]]:
    accepted = np.asarray(coarse).max(axis=1) >= float(threshold)
    pred = np.asarray(coarse).argmax(axis=1)
    table = np.zeros((len(T.COARSE), len(T.COARSE)), dtype=int)
    for gold_index, pred_index, ok in zip(np.asarray(gold, dtype=int), pred, accepted):
        if ok:
            table[gold_index, pred_index] += 1
    return table.tolist()


def incomplete_id_block(rows: Sequence[dict], complete_mask: np.ndarray, *,
                        limit: int = INCOMPLETE_ID_OUTPUT_LIMIT) -> dict:
    """不完整输入的 ID 有界输出；行本身仍全部保留在 predictions 里，不静默删行。"""
    flags = np.asarray(complete_mask, dtype=bool)
    ids = [row["id"] for row, ok in zip(rows, flags) if not ok]
    return {"complete_n": int(flags.sum()), "incomplete_n": len(ids),
            "incomplete_ids": ids[:int(limit)],
            "incomplete_ids_truncated": len(ids) > int(limit),
            "incomplete_ids_output_limit": int(limit)}


def complete_scope_metrics(*, coarse: np.ndarray, coarse_pred: np.ndarray, coarse_gold: np.ndarray,
                           fine: np.ndarray, fine_pred: np.ndarray, fine_gold: np.ndarray,
                           strict_mask: np.ndarray, complete_mask: np.ndarray) -> dict:
    """只对 input_complete=true 的行重算 coarse/fine；空子集返回 null 而不是崩溃。"""
    index = np.flatnonzero(np.asarray(complete_mask, dtype=bool))
    if index.size == 0:
        return {"complete_n": 0, "coarse_complete": None, "fine_complete": None,
                "empty_reason": "no input_complete=true rows: complete-subset metrics are null"}
    return {
        "complete_n": int(index.size),
        "coarse_complete": T.metric_block(
            np.asarray(coarse_gold)[index], np.asarray(coarse_pred)[index],
            np.asarray(coarse)[index], T.COARSE, equivalent=T.COARSE_EQUIVALENT,
            conflict=T.COARSE_CONFLICT, strict_mask=np.asarray(strict_mask)[index]),
        "fine_complete": T.metric_block(
            np.asarray(fine_gold)[index], np.asarray(fine_pred)[index],
            np.asarray(fine)[index], T.FINE, equivalent=T.FINE_EQUIVALENT,
            conflict=T.FINE_CONFLICT, strict_mask=np.asarray(strict_mask)[index]),
    }


def scoring_block(*, run: dict, checkpoint: str) -> dict:
    return {
        "run": run["run"], "arm": run["arm"], "seed": run["seed"], "checkpoint": checkpoint,
        "run_lock_sha256": T.sha256_file(run["dir"] / "run-lock.json"),
        "checkpoint_files_sha256": T.checkpoint_file_shas(run["dir"] / "checkpoints" / checkpoint),
        "data_file_sha256": T.data_identity(),
        "code_sha256": T.code_identity(),
        "environment": T.environment(),
    }


def evaluate_candidate(*, run: dict, checkpoint: str, split: str, device: torch.device,
                       purpose: str = "train", unlock: "dict | None" = None) -> dict:
    rows = T.read_split(split, purpose=purpose, unlock=unlock)
    loaded = T.reload_run(run, checkpoint, device)
    tokenizer = loaded["tokenizer"]
    evaluation = T.evaluate_split(model=loaded["model"], tokenizer=tokenizer, rows=rows,
                                  arm=loaded["arm"], split=split, device=device,
                                  max_length=T.MAX_LENGTH, bf16=bf16_for(device),
                                  encode=T.default_encode(tokenizer, device, T.MAX_LENGTH))
    return {"rows": rows, "loaded": loaded, "eval": evaluation}


def save_split_result(directory: Path, *, evaluation: dict, payload: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    T.write_jsonl(directory / "predictions.jsonl", evaluation["predictions"])
    T.dump(directory / "summary.json", payload)
    return {"predictions": str(directory / "predictions.jsonl"),
            "predictions_sha256": T.sha256_file(directory / "predictions.jsonl"),
            "summary": str(directory / "summary.json")}


                                                                                        
                     
                                                                                        

def baseline_identity() -> dict:
    """生产 bundle 身份链：bundle.json / release summary / runtime 文件三方 SHA 必须一致。"""
    if not (BUNDLE / "bundle.json").is_file():
        raise SystemExit(f"missing production bundle: {BUNDLE}")
    manifest = T.read_json(BUNDLE / "bundle.json")
    release = T.read_json(RELEASE)
    problems = []
    if manifest.get("model_id") != EXPECTED_MODEL_ID:
        problems.append(f"model_id {manifest.get('model_id')!r} != {EXPECTED_MODEL_ID!r}")
    if manifest.get("meta_sha256") != EXPECTED_META_SHA256:
        problems.append("bundle.json meta_sha256 != expected pin")
    if release.get("meta_sha256") != EXPECTED_META_SHA256:
        problems.append("bundle-build.json meta_sha256 != expected pin")
    bundle_sha = T.sha256_file(BUNDLE / "bundle.json")
    if bundle_sha != release.get("bundle_manifest_sha256"):
        problems.append("bundle.json bytes SHA != release bundle_manifest_sha256")
    meta_sha = T.sha256_file(BUNDLE / "runtime" / "semantic_meta.json")
    if meta_sha != EXPECTED_META_SHA256:
        problems.append("runtime/semantic_meta.json SHA != expected pin")
    weights_sha = T.sha256_file(BUNDLE / "runtime" / "semantic_model.pt")
    if weights_sha != release.get("weights_sha256"):
        problems.append("runtime/semantic_model.pt SHA != release weights_sha256")
    if list(manifest.get("classes") or []) != list(T.FINE):
        problems.append(f"bundle classes {manifest.get('classes')} != five-leaf order")
    if problems:
        raise SystemExit("production baseline identity check failed: " + "; ".join(problems))
    return {
        "bundle": str(BUNDLE), "release_summary": str(RELEASE),
        "model_id": manifest["model_id"], "meta_sha256": manifest["meta_sha256"],
        "bundle_manifest_sha256": bundle_sha, "weights_sha256": weights_sha,
        "classes": list(manifest["classes"]),
        "bundle_build": {key: release.get(key) for key in
                         ("archive", "archive_sha256", "files", "bytes", "threshold", "confidence_kind")},
        "files": {name: T.sha256_file(BUNDLE / "runtime" / name)
                  for name in ("config/config.json", "semantic_meta.json", "semantic_model.pt")},
    }


SIDE_KEYS = ("kind", "text", "title", "trigger", "permanence", "payload")


def baseline_pair(row: dict) -> tuple[dict, bool, int]:
    """把 processed 行里的 pair 收敛到生产 renderer 接受的语义 pair（不推断语义，只补默认 kind）。"""
    pair = row["pair"]
    defaulted, dropped = False, 0
    result = {}
    for role in ("old", "new"):
        value = pair[role]
        if isinstance(value, str):
            result[role] = {"kind": "memory", "text": value}
            defaulted = True
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{row['id']}: pair.{role} must be str or object")
        side = {key: value[key] for key in SIDE_KEYS if key in value}
        dropped += len(set(value) - set(SIDE_KEYS))
        if "kind" not in side:
            side["kind"] = "memory"
            defaulted = True
        if "text" not in side:
            raise ValueError(f"{row['id']}: pair.{role} has no text")
        result[role] = side
    if "frame" in pair:
        result["frame"] = pair["frame"]
    return result, defaulted, dropped


def run_baseline_worker(rows: list[dict], device: str) -> dict:
    """在子进程里跑生产 bundle（bundle 自带 source 先于 repo 解析）；返回逐题五叶概率。"""
    BASELINE_OUT.mkdir(parents=True, exist_ok=True)
    payload, defaulted, dropped = [], 0, 0
    for row in rows:
        pair, flag, removed = baseline_pair(row)
        defaulted += int(flag)
        dropped += int(removed)
        payload.append({"id": row["id"], "pair": pair, "visible_text": row["visible_text"]})
    temporary = Path(tempfile.mkdtemp(prefix=".baseline-worker-", dir=str(BASELINE_OUT)))
    try:
        input_path = temporary / "input.jsonl"
        output_path = temporary / "output.json"
        T.write_jsonl(input_path, payload)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "_bundle-worker",
                   "--bundle", str(BUNDLE), "--input", str(input_path),
                   "--output", str(output_path), "--device", device]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                           BIODATA_REPO_ROOT=str(REPO))
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", env=environment)
        if completed.returncode != 0 or not output_path.is_file():
            tail = (completed.stderr or completed.stdout or "")[-3000:]
            raise SystemExit(f"bundle worker failed (exit {completed.returncode}):\n{tail}")
        result = T.read_json(output_path)
        result["stdout"] = (completed.stdout or "").strip()[-500:]
        result["pair_kind_defaulted"] = defaulted
        result["pair_keys_dropped"] = dropped
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def cmd_baseline(args) -> int:
    identity = baseline_identity()
    unlock = None
    if args.split == "heldout":
        if not args.unlock_final:
            raise SystemExit("baseline 对 heldout 必须与候选同一锁：显式 --unlock-final runs/final-lock.json")
        unlock = T.validate_final_lock(args.unlock_final, require_checkpoint_files=False)
    rows = T.read_split(args.split, purpose=("locked-final" if args.split == "heldout" else "train"),
                        unlock=unlock)
    if not rows:
        raise SystemExit(f"empty split {args.split}")
    worker = run_baseline_worker(rows, args.device)
    if worker["model_id"] != EXPECTED_MODEL_ID:
        raise SystemExit(f"bundle worker model_id {worker['model_id']!r} != {EXPECTED_MODEL_ID!r}")
    if worker["meta_sha256"] != EXPECTED_META_SHA256:
        raise SystemExit("bundle worker meta_sha256 differs from the release summary pin")
    if list(worker["classes"]) != list(T.FINE):
        raise SystemExit(f"bundle worker classes {worker['classes']} != five-leaf order")
    by_id = {row["id"]: row for row in worker["rows"]}
    if set(by_id) != {row["id"] for row in rows}:
        raise SystemExit("bundle worker returned a different id set")
    fine = np.asarray([by_id[row["id"]]["probabilities"] for row in rows], dtype=np.float64)
    coarse = T.coarse_probabilities(fine, "fine5")
    coarse_pred = coarse.argmax(axis=1)
    fine_pred = fine.argmax(axis=1)
    fine_gold, coarse_gold, strict_mask = T.gold_indices(rows)
    coarse_summary = T.metric_block(coarse_gold, coarse_pred, coarse, T.COARSE,
                                    equivalent=T.COARSE_EQUIVALENT, conflict=T.COARSE_CONFLICT,
                                    strict_mask=strict_mask)
    fine_summary = T.metric_block(fine_gold, fine_pred, fine, T.FINE,
                                  equivalent=T.FINE_EQUIVALENT, conflict=T.FINE_CONFLICT,
                                  strict_mask=strict_mask)
    complete_mask = np.asarray([by_id[row["id"]]["complete"] for row in rows], dtype=bool)
    scope = {**incomplete_id_block(rows, complete_mask),
             **complete_scope_metrics(coarse=coarse, coarse_pred=coarse_pred, coarse_gold=coarse_gold,
                                      fine=fine, fine_pred=fine_pred, fine_gold=fine_gold,
                                      strict_mask=strict_mask, complete_mask=complete_mask)}
    predictions = []
    for index, row in enumerate(rows):
        record = by_id[row["id"]]
        predictions.append({
            "id": row["id"], "split": args.split,
            "family_id": row["family_id"], "cluster_id": row["cluster_id"],
            "root_id": row["root_id"], "source": row["source"], "author": row["author"],
            "gold_fine": row["label"], "gold_coarse": row["coarse_label"],
            "gold_fine_index": int(fine_gold[index]), "gold_coarse_index": int(coarse_gold[index]),
            "pred_coarse": T.COARSE[int(coarse_pred[index])],
            "pred_coarse_index": int(coarse_pred[index]),
            "pred_fine": T.FINE[int(fine_pred[index])],
            "pred_fine_index": int(fine_pred[index]),
            "coarse_probabilities": [float(value) for value in coarse[index]],
            "fine_probabilities": [float(value) for value in fine[index]],
            "max_coarse_probability": float(coarse[index].max()),
            "max_fine_probability": float(fine[index].max()),
            "input_complete": record["complete"],
            "requires_llm_fallback": record["requires_llm_fallback"],
            "rendered_matches_visible": record["rendered_matches_visible"],
            "rendered_normalized_matches_visible": record["rendered_normalized_matches_visible"],
            "rendered_sha256": record["rendered_sha256"],
        })
    directory = BASELINE_OUT / args.split
    incomplete_all = [row["id"] for row, ok in zip(rows, complete_mask) if not ok]
    incomplete_file = None
    if incomplete_all:
        T.dump(directory / "incomplete-ids.json",
               {"split": args.split, "n": len(incomplete_all), "ids": incomplete_all,
                "reason": ("production renderer marked these pairs incomplete (full pair over its "
                           "max_length); production routes them to LLM fallback")})
        incomplete_file = {"path": str(directory / "incomplete-ids.json"),
                           "sha256": T.sha256_file(directory / "incomplete-ids.json")}
    gate = gate_row(coarse, coarse_gold)
    gate["scope"] = "all rows; threshold-only accept gate"
    gate["incomplete_note"] = ("incomplete inputs already force a production fallback outside this "
                               "threshold; they are not auto-wrong and not counted as local misses")
    summary = {
        "source": "production bundle (five-leaf), coarse = fixed fine aggregation",
        "split": args.split, "n": len(rows),
        "model": {"model_id": identity["model_id"], "meta_sha256": identity["meta_sha256"],
                  "weights_sha256": identity["weights_sha256"],
                  "bundle_manifest_sha256": identity["bundle_manifest_sha256"]},
        "renderer": {"production_renderer": "semantic-json-tree-witness-v3",
                     "role_independent_frame": bool(worker.get("role_independent_frame")),
                     "pair_kind_defaulted": worker["pair_kind_defaulted"],
                     "pair_keys_dropped": worker["pair_keys_dropped"],
                     "rendered_matches_visible": sum(1 for record in predictions
                                                     if record["rendered_matches_visible"]),
                     "rendered_normalized_matches_visible": sum(
                         1 for record in predictions
                         if record["rendered_normalized_matches_visible"]),
                     "input_complete": int(scope["complete_n"]),
                     "incomplete_rows": int(scope["incomplete_n"]),
                     "local_scorable_input": "input_complete=true subset only"},
        "scoring_scope": {
            "local_scorable_subset": "rows with input_complete=true only (coarse_complete/fine_complete)",
            "why": ("the production renderer keeps the full pair only up to its max_length; on "
                    "incomplete inputs production falls back to the LLM (requires_llm_fallback) "
                    "instead of answering locally"),
            "incomplete_is_not_wrong": True,
            "all_row_metrics_meaning": ("coarse/fine/gate are over all rows for traceability; "
                                        "only coarse_complete/fine_complete estimate local semantic "
                                        "capability"),
            "complete_n": int(scope["complete_n"]), "incomplete_n": int(scope["incomplete_n"]),
            "empty_reason": scope.get("empty_reason"),
            "incomplete_ids": scope["incomplete_ids"],
            "incomplete_ids_truncated": scope["incomplete_ids_truncated"],
            "incomplete_ids_output_limit": scope["incomplete_ids_output_limit"],
            "incomplete_ids_file": incomplete_file,
        },
        "coarse": coarse_summary, "fine": fine_summary,
        "coarse_complete": scope["coarse_complete"], "fine_complete": scope["fine_complete"],
        "gate": gate,
        "baseline_identity": identity,
        "final_lock": None if unlock is None else {
            "path": str(Path(args.unlock_final)), "sha256": T.sha256_file(args.unlock_final),
            "data_sha256": unlock["data_sha256"]},
        "environment": T.environment(), "seconds": worker.get("seconds"),
        "note": ("production baseline replay; no threshold tuning here. Read coarse_complete as the "
                 "local-capability estimate; incomplete rows are a production fallback reason, "
                 "not an automatic wrong answer."),
    }
    saved = save_split_result(directory, evaluation={"predictions": predictions}, payload=summary)
    complete_coarse = summary["coarse_complete"]
    print(json.dumps({"baseline": True, "split": args.split, "n": len(rows),
                      "model_id": identity["model_id"],
                      "complete_n": summary["scoring_scope"]["complete_n"],
                      "incomplete_n": summary["scoring_scope"]["incomplete_n"],
                      "coarse_all": {key: coarse_summary[key]
                                     for key in ("accuracy", "macro_f1", "loss")},
                      "coarse_complete": None if complete_coarse is None else {
                          key: complete_coarse[key] for key in ("n", "accuracy", "macro_f1", "loss")},
                      "conflict_recall_all": coarse_summary["conflict_recall"],
                      "conflict_recall_complete": (None if complete_coarse is None
                                                   else complete_coarse["conflict_recall"]),
                      "false_equivalence_all": coarse_summary["false_equivalence"],
                      "rendered_matches_visible": summary["renderer"]["rendered_matches_visible"],
                      "saved": saved}, ensure_ascii=False))
    return 0


                                                                                        
                                 
                                                                                        

def cmd_metrics(args) -> int:
    run = T.load_run(args.run)
    if args.split == "heldout":
        raise SystemExit("heldout 不在 metrics 里评分：必须走 final-eval + runs/final-lock.json")
    checkpoint = args.checkpoint
    if checkpoint not in (run["result"] or {}).get("checkpoints", {}):
        raise SystemExit(f"run {run['run']} has no checkpoint {checkpoint!r}")
    device = FT.resolve_device(args.device)
    result = evaluate_candidate(run=run, checkpoint=checkpoint, split=args.split, device=device)
    payload = dict(result["eval"]["summary"])
    payload.update({"checkpoint": checkpoint, "reload": result["loaded"]["reload"],
                    "scoring": scoring_block(run=run, checkpoint=checkpoint),
                    "rows_parsed": {args.split: len(result["rows"])}})
    directory = run["dir"] / "scores" / checkpoint / args.split
    saved = save_split_result(directory, evaluation=result["eval"], payload=payload)
    coarse = payload["coarse"]
    print(json.dumps({"metrics": True, "run": run["run"], "checkpoint": checkpoint,
                      "split": args.split, "n": payload["n"],
                      "coarse": {key: coarse[key] for key in ("accuracy", "macro_f1", "loss")},
                      "conflict_recall": coarse["conflict_recall"],
                      "false_equivalence": coarse["false_equivalence"],
                      "strict_subset_redundancy_accept": coarse["strict_subset_redundancy_accept"],
                      "reload": {"format": result["loaded"]["reload"].get("format"),
                                 "adapter_lora_sha256": result["loaded"]["reload"].get("adapter_lora_sha256")},
                      "saved": saved}, ensure_ascii=False))
    return 0


def cmd_calibrate(args) -> int:
    run = T.load_run(args.run)
    checkpoint = "best-dev"
    device = FT.resolve_device(args.device)
    result = evaluate_candidate(run=run, checkpoint=checkpoint, split="calibration", device=device)
    coarse = result["eval"]["coarse_probabilities"]
    gold = T.gold_indices(result["rows"])[1]
    gate = T.calibrate_gate(coarse, gold)
    payload = {
        "run": run["run"], "arm": run["arm"], "seed": run["seed"], "checkpoint": checkpoint,
        "split": "calibration", "n": gate["n"],
        "thresholds": gate["thresholds"], "rule": gate["rule"],
        "feasible_thresholds": gate["feasible_thresholds"],
        "selected": gate["selected"], "local_accept": gate["local_accept"],
        "threshold": gate["threshold"],
        "coarse": result["eval"]["summary"]["coarse"],
        "fine": result["eval"]["summary"]["fine"],
        "reload": result["loaded"]["reload"],
        "scoring": scoring_block(run=run, checkpoint=checkpoint),
        "note": ("research-only threshold; fitted on calibration only, never on heldout, "
                 "not a production threshold"),
    }
    saved = save_split_result(run["dir"] / "scores" / checkpoint / "calibration",
                              evaluation=result["eval"], payload=payload)
    T.dump(run["dir"] / "calibration.json", payload)
    print(json.dumps({"calibrate": True, "run": run["run"], "n": payload["n"],
                      "local_accept": payload["local_accept"], "threshold": payload["threshold"],
                      "feasible_thresholds": payload["feasible_thresholds"],
                      "selected": None if payload["selected"] is None else {
                          "threshold": payload["selected"]["threshold"],
                          "accepted": payload["selected"]["accepted"],
                          "accepted_accuracy": payload["selected"]["accepted_accuracy"],
                          "conflict_recall": payload["selected"]["conflict_recall"],
                          "false_equivalence": payload["selected"]["false_equivalence"]},
                      "saved": saved}, ensure_ascii=False))
    return 0


                                                                                        
                         
                                                                                        

def lock_consistency(run: dict, lock: dict) -> None:
    if lock["arm"] != run["arm"] or int(lock["seed"]) != int(run["seed"]):
        raise SystemExit(f"final-lock arm/seed {lock['arm']}/{lock['seed']} != run {run['arm']}/{run['seed']}")
    result = run["result"]
    if result is None:
        raise SystemExit(f"run {run['run']} has no result.json")
    if int(lock.get("selected_step", -1)) != int(result["selection"]["selected_step"]):
        raise SystemExit("final-lock selected_step 与 run 记录的 dev-best step 不一致")
    if lock.get("train_run_lock_sha256"):
        actual = T.sha256_file(run["dir"] / "run-lock.json")
        if actual != lock["train_run_lock_sha256"]:
            raise SystemExit("final-lock train_run_lock_sha256 与盘上 run-lock.json 不一致")


def cmd_final_eval(args) -> int:
    if args.show_requirements:
        print(json.dumps(T.lock_template(args.run, args.checkpoint), ensure_ascii=False, indent=2))
        return 0
    if not args.unlock_final:
        raise SystemExit("final-eval 必须显式 --unlock-final runs/final-lock.json（不自动建锁）")
    lock = T.validate_final_lock(args.unlock_final, run=args.run, checkpoint=None)
    run = T.load_run(args.run)
    lock_consistency(run, lock)
    checkpoint = lock["checkpoint"]
    if checkpoint not in (run["result"] or {}).get("checkpoints", {}):
        raise SystemExit(f"final-lock checkpoint {checkpoint!r} 不在 run 的 checkpoint 里")
    device = FT.resolve_device(args.device)
    result = evaluate_candidate(run=run, checkpoint=checkpoint, split="heldout", device=device,
                                purpose="locked-final", unlock=lock)
    coarse = result["eval"]["coarse_probabilities"]
    gold = T.gold_indices(result["rows"])[1]
    gate = None
    calibration_path = run["dir"] / "calibration.json"
    if calibration_path.is_file():
        calibration = T.read_json(calibration_path)
        if calibration.get("threshold") is not None:
            gate = gate_row(coarse, gold, calibration["threshold"])
            gate["threshold_source"] = {"path": str(calibration_path),
                                        "sha256": T.sha256_file(calibration_path),
                                        "split": "calibration"}
    manifest = {
        "schema": "relation-final-eval-manifest/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": run["run"], "arm": run["arm"], "seed": run["seed"], "checkpoint": checkpoint,
        "final_lock": {"path": str(Path(args.unlock_final)),
                       "sha256": T.sha256_file(args.unlock_final),
                       "created": lock.get("created"),
                       "selected_step": lock.get("selected_step"),
                       "data_sha256": lock["data_sha256"],
                       "checkpoint_files_sha256": lock["checkpoint_files_sha256"]},
        "run_lock_sha256": T.sha256_file(run["dir"] / "run-lock.json"),
        "data_sha256": T.data_identity(),
        "data_rows": {"heldout": len(result["rows"])},
        "model": {"base": T.model_identity(),
                  "checkpoint_files_sha256": T.checkpoint_file_shas(
                      run["dir"] / "checkpoints" / checkpoint),
                  "reload": result["loaded"]["reload"]},
        "code_sha256": T.code_identity(),
        "environment": T.environment(),
        "heldout_metrics_are_final": True,
    }
    payload = dict(result["eval"]["summary"])
    payload.update({"checkpoint": checkpoint, "reload": result["loaded"]["reload"],
                    "gate": gate, "manifest": manifest,
                    "note": "single locked heldout evaluation; no re-selection"})
    directory = run["dir"] / "final"
    saved = save_split_result(directory, evaluation=result["eval"], payload=payload)
    T.dump(directory / "manifest.json", manifest)
    coarse_summary = result["eval"]["summary"]["coarse"]
    print(json.dumps({"final_eval": True, "run": run["run"], "checkpoint": checkpoint,
                      "n": len(result["rows"]),
                      "coarse": {key: coarse_summary[key]
                                 for key in ("accuracy", "macro_f1", "loss")},
                      "conflict_recall": coarse_summary["conflict_recall"],
                      "false_equivalence": coarse_summary["false_equivalence"],
                      "strict_subset_redundancy_accept": coarse_summary["strict_subset_redundancy_accept"],
                      "gate": None if gate is None else {
                          "threshold": gate["threshold"], "coverage": gate["coverage"],
                          "accepted_accuracy": gate["accepted_accuracy"],
                          "conflict_recall": gate["conflict_recall"],
                          "false_equivalence": gate["false_equivalence"]},
                      "saved": saved}, ensure_ascii=False))
    return 0


                                                                                        
                            
                                                                                        

def cmd_verify(args) -> int:
    run = T.load_run(args.run)
    unlock = None
    checkpoint = args.checkpoint
    splits = ["dev", "calibration"]
    if args.unlock_final:
        unlock = T.validate_final_lock(args.unlock_final, run=args.run, checkpoint=None)
        lock_consistency(run, unlock)
        checkpoint = unlock["checkpoint"]
        splits.append("heldout")
    if checkpoint not in (run["result"] or {}).get("checkpoints", {}):
        raise SystemExit(f"run {run['run']} has no checkpoint {checkpoint!r}")
    device = FT.resolve_device(args.device)
    loaded = T.reload_run(run, checkpoint, device)
    tokenizer = loaded["tokenizer"]
    report: dict[str, Any] = {}
    ok = True
    for split in splits:
        directory = (run["dir"] / "final" if split == "heldout"
                     else run["dir"] / "scores" / checkpoint / split)
        predictions_path = directory / "predictions.jsonl"
        summary_path = directory / "summary.json"
        if not predictions_path.is_file() or not summary_path.is_file():
            raise SystemExit(f"missing stored predictions for {split}: {predictions_path}；"
                             f"先跑 metrics（dev/calibration）或 final-eval（heldout）")
        rows = T.read_split(split, purpose=("locked-final" if split == "heldout" else "train"),
                            unlock=unlock)
        evaluation = T.evaluate_split(model=loaded["model"], tokenizer=tokenizer, rows=rows,
                                      arm=loaded["arm"], split=split, device=device,
                                      max_length=T.MAX_LENGTH, bf16=bf16_for(device),
                                      encode=T.default_encode(tokenizer, device, T.MAX_LENGTH))
        stored_predictions = T.read_jsonl(predictions_path)
        stored_summary = T.read_json(summary_path)
        by_id = {row["id"]: row for row in stored_predictions}
        max_delta, mismatches = T.compare_predictions(by_id, evaluation["predictions"])
        summary_delta = {}
        for block in ("coarse", "fine"):
            fresh_block, stored_block = evaluation["summary"][block], stored_summary.get(block)
            if fresh_block is None and stored_block is None:
                continue
            if fresh_block is None or stored_block is None:
                raise SystemExit(f"{split}: {block} summary presence differs")
            for key in ("accuracy", "macro_f1", "loss"):
                if fresh_block.get(key) is None or stored_block.get(key) is None:
                    continue
                summary_delta[f"{block}.{key}"] = abs(float(fresh_block[key]) - float(stored_block[key]))
        split_ok = (mismatches == 0 and max_delta <= PROBABILITY_TOLERANCE
                    and max(summary_delta.values(), default=0.0) <= SUMMARY_TOLERANCE)
        ok = ok and split_ok
        report[split] = {
            "n": len(rows), "rows_parsed": len(rows),
            "max_probability_delta": max_delta,
            "argmax_mismatch": mismatches,
            "summary_abs_delta": summary_delta,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "summary_tolerance": SUMMARY_TOLERANCE,
            "passed": split_ok,
            "stored": {"predictions": str(predictions_path),
                       "predictions_sha256": T.sha256_file(predictions_path),
                       "summary": str(summary_path)},
        }
    manifest = {
        "schema": "relation-verify-manifest/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": run["run"], "arm": run["arm"], "seed": run["seed"], "checkpoint": checkpoint,
        "reload": loaded["reload"],
        "run_lock_sha256": T.sha256_file(run["dir"] / "run-lock.json"),
        "data_sha256": T.data_identity(),
        "data_rows": {split: report[split]["n"] for split in splits},
        "model": {"base": T.model_identity(),
                  "checkpoint_files_sha256": T.checkpoint_file_shas(
                      run["dir"] / "checkpoints" / checkpoint)},
        "code_sha256": T.code_identity(),
        "environment": T.environment(),
        "final_lock": None if unlock is None else {
            "path": str(Path(args.unlock_final)), "sha256": T.sha256_file(args.unlock_final)},
    }
    payload = {"passed": ok, "splits": report, "manifest": manifest,
               "device": str(device), "bf16": bf16_for(device),
               "note": "fresh process; strict adapter+head reload with SHA checks before scoring"}
    T.dump(run["dir"] / "verify.json", payload)
    print(json.dumps({"verify": True, "passed": ok, "run": run["run"], "checkpoint": checkpoint,
                      "splits": {split: {"max_probability_delta": report[split]["max_probability_delta"],
                                         "argmax_mismatch": report[split]["argmax_mismatch"],
                                         "passed": report[split]["passed"]} for split in splits},
                      "reload": {"format": loaded["reload"].get("format"),
                                 "adapter_lora_sha256": loaded["reload"].get("adapter_lora_sha256"),
                                 "head_sha256": loaded["reload"].get("head_sha256")}},
                     ensure_ascii=False))
    return 0 if ok else 1


                                                                                        
          
                                                                                        

def _probabilities(labels: Sequence[str], confidence: float = 0.9) -> np.ndarray:
    """已知概率：argmax = label，其余质量平均分给另外两类。"""
    total = len(T.COARSE)
    rows = []
    for label in labels:
        index = T.COARSE.index(label)
        row = [(1.0 - confidence) / (total - 1)] * total
        row[index] = confidence
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def cmd_selftest(args) -> int:
    """离线自检：已知概率 + 小模型 fake；不建真实模型、不碰 CUDA、不读 processed。"""
    torch.set_num_threads(1)
    checks: dict[str, Any] = {}
    gold = ["equivalent", "compatible", "compatible", "conflict", "equivalent", "conflict"]
    gold_index = np.asarray([T.COARSE.index(value) for value in gold], dtype=int)
    fine_gold = np.asarray([T.FINE.index(value) for value in
                            ("equivalent", "old_entails_new_strict", "new_entails_old_strict",
                             "disjoint", "overlap_incomparable", "disjoint")], dtype=int)
    strict_mask = np.isin(fine_gold, list(T.FINE_STRICT))

                
    perfect = _probabilities(gold)
    block = T.metric_block(gold_index, perfect.argmax(axis=1), perfect, T.COARSE,
                           equivalent=T.COARSE_EQUIVALENT, conflict=T.COARSE_CONFLICT,
                           strict_mask=strict_mask)
    assert block["accuracy"] == 1.0 and block["macro_f1"] == 1.0
    assert block["conflict_recall"] == {"value": 1.0, "numerator": 2, "n": 2}
    assert block["false_equivalence"] == {"value": 0.0, "numerator": 0, "n": 4}
    assert block["strict_subset_redundancy_accept"] == {"value": 0.0, "numerator": 0, "n": 2}
    checks["perfect"] = {"accuracy": block["accuracy"], "macro_f1": block["macro_f1"],
                         "conflict_recall": block["conflict_recall"],
                         "false_equivalence": block["false_equivalence"]}

                                          
    shifted = np.asarray([(value + 1) % 3 for value in gold_index], dtype=int)
    wrong = np.full((len(gold), 3), 0.1 / 2)
    for row, index in enumerate(shifted):
        wrong[row, index] = 0.9
    wrong_block = T.metric_block(gold_index, shifted, wrong, T.COARSE,
                                 equivalent=T.COARSE_EQUIVALENT, conflict=T.COARSE_CONFLICT,
                                 strict_mask=strict_mask)
    assert wrong_block["accuracy"] == 0.0 and wrong_block["correct"] == 0
    assert wrong_block["conflict_recall"]["value"] == 0.0
    assert wrong_block["false_equivalence"]["numerator"] == 2, wrong_block["false_equivalence"]
    assert wrong_block["strict_subset_redundancy_accept"]["numerator"] == 0
                                      
    redundant = _probabilities(["equivalent"] * len(gold))
    redundant_block = T.metric_block(gold_index, redundant.argmax(axis=1), redundant, T.COARSE,
                                     equivalent=T.COARSE_EQUIVALENT, conflict=T.COARSE_CONFLICT,
                                     strict_mask=strict_mask)
    assert redundant_block["strict_subset_redundancy_accept"] == {"value": 1.0, "numerator": 2, "n": 2}
    assert redundant_block["false_equivalence"] == {"value": 1.0, "numerator": 4, "n": 4}
    assert redundant_block["conflict_recall"] == {"value": 0.0, "numerator": 0, "n": 2}
    checks["all_wrong"] = {"accuracy": wrong_block["accuracy"],
                           "conflict_recall": wrong_block["conflict_recall"],
                           "false_equivalence": wrong_block["false_equivalence"],
                           "strict_subset_redundancy_accept":
                               wrong_block["strict_subset_redundancy_accept"],
                           "all_equivalent": {
                               "strict_subset_redundancy_accept":
                                   redundant_block["strict_subset_redundancy_accept"],
                               "false_equivalence": redundant_block["false_equivalence"]}}

                                                                       
    fine = np.asarray([[0.40, 0.30, 0.10, 0.10, 0.10]])
    coarse = T.coarse_probabilities(fine, "fine5")
    assert abs(float(coarse.sum()) - 1.0) < 1e-12
    assert T.FINE[int(fine.argmax())] == "equivalent"
    assert T.COARSE[int(coarse.argmax())] == "compatible"
    assert np.array_equal(coarse, T.coarse_probabilities(fine, "fine5"))
    checks["fine5_aggregation"] = {"fine_probabilities": fine.tolist(),
                                   "coarse_probabilities": coarse.tolist(),
                                   "fine_argmax": "equivalent", "coarse_argmax": "compatible"}

                    
    design = [(0, 1, 0.75), (1, 0, 0.55), (1, 1, 0.96), (2, 2, 0.95), (0, 0, 0.97), (2, 2, 0.93)]
    good = np.zeros((len(design), 3))
    for row, (_, predicted, maximum) in enumerate(design):
        good[row] = [(1.0 - maximum) / 2.0] * 3
        good[row, predicted] = maximum
    assert np.allclose(good.sum(axis=1), 1.0)
    feasible = T.calibrate_gate(good, gold_index)
    assert feasible["local_accept"] is True, feasible
    selected = feasible["selected"]
    assert selected["accepted"] == 4, selected["accepted"]
    assert selected["accepted_accuracy"]["value"] == 1.0
    assert selected["conflict_recall"]["value"] == 1.0 and selected["conflict_recall"]["n"] == 2
    assert selected["false_equivalence"]["value"] == 0.0 and selected["false_equivalence"]["n"] == 4
    assert 0.76 <= selected["threshold"] <= 0.93, selected["threshold"]
    assert feasible["thresholds"][0]["coverage"] == {"value": 1.0, "numerator": 6, "n": 6}
                                                     
    conflict_wrong = _probabilities(["equivalent"] * 6)
    infeasible = T.calibrate_gate(conflict_wrong, gold_index)
    assert infeasible["local_accept"] is False and infeasible["threshold"] is None
    assert all(row["conflict_recall"]["value"] == 0.0 for row in infeasible["thresholds"])
    for row in infeasible["thresholds"]:
        assert row["coverage"]["n"] == 6 and row["coverage"]["numerator"] + row["fallback"]["numerator"] == 6
    checks["gate"] = {"feasible_threshold": selected["threshold"],
                      "feasible_accepted": selected["accepted"],
                      "feasible_accepted_accuracy": selected["accepted_accuracy"],
                      "infeasible_local_accept": infeasible["local_accept"],
                      "thresholds_reported": len(infeasible["thresholds"])}

                           
    temporary = Path(tempfile.mkdtemp(prefix="relation-score-selftest-"))
    try:
        missing = temporary / "no-lock.json"
        try:
            T.validate_final_lock(missing)
        except T.HeldoutLockError as error:
            lock_error = str(error)
        else:
            raise AssertionError("missing final-lock must be refused")
        stale = temporary / "stale-lock.json"
        T.dump(stale, {"schema": T.LOCK_SCHEMA, "run": "x", "checkpoint": "best-dev",
                       "arm": "fine5", "seed": 13, "selected_step": 1,
                       "data_sha256": {split: "0" * 64 for split in T.SPLITS},
                       "checkpoint_files_sha256": {"head.pt": "0" * 64}})
        try:
            T.validate_final_lock(stale)
        except T.HeldoutLockError as error:
            stale_error = str(error)
        else:
            raise AssertionError("stale final-lock must be refused")
        try:
            T.read_split("heldout", purpose="train")
        except T.HeldoutLockError as error:
            early = str(error)
        else:
            raise AssertionError("heldout must be refused before the lock")
        checks["locks"] = {"missing_lock_refused": True, "stale_lock_refused": True,
                           "early_heldout_refused": True,
                           "missing_error": lock_error[:48], "stale_error": stale_error[:48],
                           "early_error": early[:48]}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

                
    rows = T.fake_rows()
    duplicate = dict(rows[1], id="dup", split="train", label="disjoint", coarse_label="conflict")
    report = T.duplicate_label_audit({"train": [duplicate], "dev": rows[:3],
                                      "calibration": [], "heldout": []},
                                     allow_transformed=False)
    assert report["identical_text_conflicting_labels"] == 1, report["identical_text_conflicting_labels"]
    assert report["identical_text_across_splits"] == 1
    assert report["visible_text_missing_side"] == 0
    assert report["passed"] is False
    bad_row = dict(rows[0], coarse_label="equivalent")
    assert T.validate_row(bad_row, "dev"), "coarse mismatch must be caught"
    checks["audit_counterexamples"] = {"conflicting_labels": report["identical_text_conflicting_labels"],
                                       "cross_split_text": report["identical_text_across_splits"],
                                       "row_contract_detected": True}

                                              
    device = torch.device("cpu")
    built = T.build_arm("direct3", 13, builder=lambda seed: T.build_fake_model("direct3", seed))
    encode = T.default_encode(built["tokenizer"], device, 64)
    first = T.evaluate_split(model=built["model"], tokenizer=built["tokenizer"], rows=rows,
                             arm="direct3", split="dev", device=device, max_length=64, bf16=False,
                             encode=encode)
    meta = {"format": "fake-full", "num_labels": 3, "selection": "best-dev",
            "full_state_sha256": T.state_sha256(built["model"].state_dict())}
    temporary = Path(tempfile.mkdtemp(prefix="relation-score-parity-"))
    try:
        T.save_checkpoint(temporary / "best-dev", built["model"], {}, meta)
        rebuilt = T.build_arm("direct3", 13, builder=lambda seed: T.build_fake_model("direct3", seed))
        reloaded = T.load_checkpoint(rebuilt["model"], temporary / "best-dev")
        second = T.evaluate_split(model=rebuilt["model"], tokenizer=rebuilt["tokenizer"], rows=rows,
                                  arm="direct3", split="dev", device=device, max_length=64,
                                  bf16=False, encode=encode)
        delta, mismatches = T.compare_predictions({row["id"]: row for row in first["predictions"]},
                                                  second["predictions"])
        assert reloaded["format"] == "fake-full" and delta <= PROBABILITY_TOLERANCE and mismatches == 0
        checks["reload_parity"] = {"format": reloaded["format"], "max_probability_delta": delta,
                                   "argmax_mismatch": mismatches,
                                   "tolerance": PROBABILITY_TOLERANCE}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

                                                             
    coarse_gold_4 = np.asarray([T.COARSE.index(name) for name in
                                ("equivalent", "compatible", "conflict", "equivalent")], dtype=int)
    coarse_pred_4 = np.asarray([T.COARSE.index(name) for name in
                                ("equivalent", "compatible", "equivalent", "conflict")], dtype=int)
    coarse_4 = _probabilities(["equivalent", "compatible", "equivalent", "conflict"])
    fine_gold_4 = np.asarray([T.FINE.index(name) for name in
                              ("equivalent", "old_entails_new_strict", "disjoint", "equivalent")],
                             dtype=int)
    fine_pred_4 = np.asarray([T.FINE.index(name) for name in
                              ("equivalent", "old_entails_new_strict", "equivalent", "disjoint")],
                             dtype=int)
    fine_4 = np.full((4, len(T.FINE)), 0.04)
    fine_4[np.arange(4), fine_pred_4] = 0.84
    strict_4 = np.isin(fine_gold_4, list(T.FINE_STRICT))
    rows_4 = [{"id": name} for name in
              ("complete-a", "complete-b", "incomplete-c", "incomplete-d")]
    mask_4 = np.asarray([True, True, False, False])
    all_4 = T.metric_block(coarse_gold_4, coarse_pred_4, coarse_4, T.COARSE,
                           equivalent=T.COARSE_EQUIVALENT, conflict=T.COARSE_CONFLICT,
                           strict_mask=strict_4)
    scope_4 = {**incomplete_id_block(rows_4, mask_4),
               **complete_scope_metrics(coarse=coarse_4, coarse_pred=coarse_pred_4,
                                        coarse_gold=coarse_gold_4, fine=fine_4,
                                        fine_pred=fine_pred_4, fine_gold=fine_gold_4,
                                        strict_mask=strict_4, complete_mask=mask_4)}
    assert all_4["n"] == 4 and all_4["accuracy"] == 0.5
    assert scope_4["complete_n"] == 2 and scope_4["incomplete_n"] == 2
    assert scope_4["incomplete_ids"] == ["incomplete-c", "incomplete-d"]
    assert scope_4["incomplete_ids_truncated"] is False
    assert scope_4["coarse_complete"]["n"] == 2
    assert scope_4["coarse_complete"]["accuracy"] == 1.0
    assert scope_4["coarse_complete"]["conflict_recall"] == {"value": None, "numerator": 0, "n": 0}
    assert scope_4["fine_complete"]["n"] == 2
    assert scope_4["fine_complete"]["accuracy"] == 1.0
    truncated = incomplete_id_block(rows_4, mask_4, limit=1)
    assert truncated["incomplete_ids"] == ["incomplete-c"] and truncated["incomplete_ids_truncated"] is True
    empty_4 = {**incomplete_id_block(rows_4, np.zeros(4, dtype=bool)),
               **complete_scope_metrics(coarse=coarse_4, coarse_pred=coarse_pred_4,
                                        coarse_gold=coarse_gold_4, fine=fine_4,
                                        fine_pred=fine_pred_4, fine_gold=fine_gold_4,
                                        strict_mask=strict_4,
                                        complete_mask=np.zeros(4, dtype=bool))}
    assert empty_4["complete_n"] == 0 and empty_4["incomplete_n"] == 4
    assert empty_4["coarse_complete"] is None and empty_4["fine_complete"] is None
    checks["baseline_complete_scope"] = {
        "all_n": all_4["n"], "all_accuracy": all_4["accuracy"],
        "complete_n": scope_4["complete_n"], "incomplete_n": scope_4["incomplete_n"],
        "complete_accuracy": scope_4["coarse_complete"]["accuracy"],
        "complete_conflict_recall": scope_4["coarse_complete"]["conflict_recall"],
        "incomplete_ids": scope_4["incomplete_ids"],
        "ids_bounded_when_limit_small": truncated["incomplete_ids_truncated"],
        "empty_complete_metrics_are_null": empty_4["coarse_complete"] is None,
    }

    print(json.dumps({"selftest": True, "ok": True, "checks": checks}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    metrics = sub.add_parser("metrics")
    metrics.add_argument("--run", required=True)
    metrics.add_argument("--checkpoint", default="best-dev", choices=("best-dev", "terminal"))
    metrics.add_argument("--split", default="dev", choices=("train", "dev", "calibration"))
    metrics.add_argument("--device", default="cuda:0")
    metrics.set_defaults(func=cmd_metrics)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--split", default="dev", choices=T.SPLITS)
    baseline.add_argument("--unlock-final", default=None)
    baseline.add_argument("--device", default="cpu")
    baseline.set_defaults(func=cmd_baseline)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--run", required=True)
    calibrate.add_argument("--device", default="cuda:0")
    calibrate.set_defaults(func=cmd_calibrate)
    final = sub.add_parser("final-eval")
    final.add_argument("--run", required=True)
    final.add_argument("--unlock-final", default=None)
    final.add_argument("--checkpoint", default="best-dev", choices=("best-dev", "terminal"))
    final.add_argument("--show-requirements", action="store_true")
    final.add_argument("--device", default="cuda:0")
    final.set_defaults(func=cmd_final_eval)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", required=True)
    verify.add_argument("--checkpoint", default="best-dev", choices=("best-dev", "terminal"))
    verify.add_argument("--unlock-final", default=None)
    verify.add_argument("--device", default="cuda:0")
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    return (getattr(args, "func", None) or cmd_selftest)(args)


if __name__ == "__main__":
    raise SystemExit(main())
