                       
"""v3 研究后处理（活动实现；v0/v1/v2 全部冻结为历史，不改其产物）。

三件事：
1. `runs/v3-evaluate/` 的 12 次真实回放聚合（6 个 GPU 训练工件 × {set_challenge, challenge}，
   公共 `features.load_scorer` CPU 加载；新风险口径每 rate 带 numerator/denominator，分类只在
   valid_rows，gate/row error 单列；不与 v1/v2 混率）。
2. `runs/v2-train/` 六训练 run 聚合：实际 n/groups、final train acc/loss、cal F1/NLL/ECE 与
   **全 risk curve**、heldout/challenge 三 seed mean/range、峰显存/时长、truncation 计数。
3. **CPU 推理 parity**（不得用 GPU 批内重载 parity 冒充）：对每个工件在 challenge 135 对上做
   CPU `predict_proba`，与 GPU `predictions.jsonl`（6 位概率）比耐差：max|Δ|、argmax 一致率、
   逐行差异；meta/model sha 与样本 hash 钉死；argmax 与历史不一致 → 显式 `parity_complete=false`。
   从 GPU 训练不经过 CPU 训练；本检查只做「同工件同输入、不同设备」的推理对照。

口径：无真人金标、非生产；所有 threshold=null（无可行点）不转生产；v1→v2 差异不能唯一归因
finetune（输入序列/池化/长度均变）；只比较 v2 内两臂解释冻结效应。GPU 模型一律 CPU 推理，
`CUDA_VISIBLE_DEVICES=""`。用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 .venv/Scripts/python.exe \
        research/memory_relations/runs/v3-evaluate/postprocess.py [--skip-parity]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

PARITY_TOLERANCE = 1e-4
PARITY_BATCH = 8
PARITY_THREADS = 6
DISCLOSURES = [
    "全部 threshold=null（无可行校准点）：不转生产、不声明操作策略；一切率为 agreement，"
    "无真人金标、非 accuracy、非独立核验。",
    "v2 两臂（frozen-joint / finetuned-joint）同输入模板、同 head、同超参协议，唯一差异是骨干"
    "是否冻结 → v2 内两臂差值可解释冻结效应；v1（MiniLM 句向量 mean-pooling、max_len 128）→ v2"
    "（CLS、max_len 384、joint 序列）同时改变输入序列/池化/长度，v1→v2 差异不能唯一归因 finetune。",
    "论文报 finetuned challenge F1 用三 seed 均值与范围（≈0.351，[0.308, 0.415]），不得用最大 seed"
    "（0.415）当效果；frozen 同口径均值 ≈0.182。",
    "12 次回放在 CPU 上执行（工件为同一批 GPU 训练的 checkpoint，经公共 features.load_scorer 加载）；"
    "parity 为 CPU vs GPU 推理对照，不是 GPU 批内 reload parity。",
    "截断计数：train/cal/heldout/challenge 全 0（无样本被截断）；ood 行数为 0。",
]


class Fail(Exception):
    def __init__(self, code: str, detail: object = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while (chunk := handle.read(1 << 20)):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def load_jsonl_strict(path: Path) -> list:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise Fail("invalid_json_line", {"path": str(path), "line": number}) from None
            if not isinstance(row, dict):
                raise Fail("non_object_line", {"path": str(path), "line": number})
            rows.append(row)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def spread(values: list) -> dict:
    vals = [float(value) for value in values if value is not None]
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def rate_record(part: int, whole: int) -> dict:
    return {"numerator": int(part), "denominator": int(whole),
            "value": (part / whole) if whole else None}


                                                   

def aggregate_replays(v3_dir: Path) -> dict:
    log_rows = load_jsonl_strict(v3_dir / "eval-run-log.jsonl")
    per_run, failures = [], []
    for entry in log_rows:
        report_path = ROOT / entry["out"]
        if entry.get("exit") != 0 or not Path(report_path).is_file():
            failures.append({"model_dir": entry.get("model_dir"), "dataset": entry.get("dataset"),
                             "exit": entry.get("exit"), "reason": "nonzero_exit_or_missing"})
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        summary, rows = report["summary"], report["rows"]
        valid = [row for row in rows if row.get("row_error") is None and not row.get("error")]
        prim = sum(1 for row in valid if row.get("primary_agree"))
        acc = sum(1 for row in valid if row.get("acceptable_agree"))
        for label, agreed in (("primary", prim), ("acceptable", acc)):
            reported = summary["agreement"][label]
            got = (agreed / len(valid)) if valid else None
            if len(valid) != summary["valid_rows"] or (reported is None) != (got is None) \
                    or (got is not None and abs(reported - got) > 1e-9):
                raise Fail("aggregate_invariant", {"run": entry["out"], "field": label})
        model_dir = entry["model_dir"]
        arm = "frozen-joint" if model_dir.startswith("frozen-joint") else "finetuned-joint"
        per_run.append({
            "model_dir": model_dir, "arm": arm, "dataset": entry["dataset"],
            "seed": model_dir.rsplit("seed", 1)[1], "report": entry["out"],
            "exit": entry["exit"], "elapsed_s": entry.get("elapsed_s"),
            "report_sha256": sha256_file(report_path),
            "attempts": summary["row_count"], "valid_rows": summary["valid_rows"],
            "gate_error_rows": summary["gate_error_rows"], "row_error_rows": summary["row_error_rows"],
            "gate_errors": summary["gate_errors"], "row_errors": summary["row_errors"],
            "primary": rate_record(prim, len(valid)), "acceptable": rate_record(acc, len(valid)),
            "macro_f1": summary["macro_f1"], "macro_f1_scope": "valid_rows（与 gate_error 并列读）",
            "operation": summary["operation"], "refs": summary["refs"],
            "predictor_calls": summary["predictor_calls"], "scored_pairs": summary["scored_pairs"],
            "threshold_operation": report["operation"],
        })
    aggregates = []
    groups: dict = {}
    for item in per_run:
        groups.setdefault((item["arm"], item["dataset"]), []).append(item)
    for (arm, dataset), entries in sorted(groups.items()):
        by_seed = {item["seed"]: item for item in entries}
        values = {name: [by_seed[key][name]["value"] for key in sorted(by_seed)]
                  for name in ("primary", "acceptable")}
        aggregates.append({
            "arm": arm, "dataset": dataset, "n_seeds": len(entries),
            "counts": {"attempts": sum(item["attempts"] for item in entries),
                       "valid_rows": sum(item["valid_rows"] for item in entries),
                       "gate_error_rows": sum(item["gate_error_rows"] for item in entries),
                       "row_error_rows": sum(item["row_error_rows"] for item in entries)},
            "primary": {"per_seed": {key: by_seed[key]["primary"] for key in sorted(by_seed)},
                        **{name: value for name, value in spread(values["primary"]).items()
                           if name != "n"}},
            "acceptable": {"per_seed": {key: by_seed[key]["acceptable"] for key in sorted(by_seed)},
                           **{name: value for name, value in spread(values["acceptable"]).items()
                              if name != "n"}},
            "macro_f1": {"per_seed": {key: by_seed[key]["macro_f1"] for key in sorted(by_seed)},
                         **spread([by_seed[key]["macro_f1"] for key in sorted(by_seed)])},
            "note": "均值是各 seed 运行点率的均值；各率分母随 seed 可能不同（per_seed 带 numerator/"
                    "denominator），不做跨 seed 合并比率；不与 v1/v2 混率。",
        })
    return {
        "schema": "memory-relations-v3-set-replay-aggregate-v1", "generated_at": _now(),
        "source": {"run_log": "research/memory_relations/runs/v3-evaluate/eval-run-log.jsonl",
                   "runs_logged": len(log_rows), "runs_aggregated": len(per_run),
                   "run_failures": failures,
                   "history_role": "v0/v1/v2 产物冻结为历史（口径不同，不重写、不并表）"},
        "per_run": per_run, "aggregates": aggregates,
        "disclosures": DISCLOSURES + [
            "全部回放为 --diagnostic --threshold 0（calibrated=false）分类诊断，不是已校准操作点。"],
    }


                                                             

def aggregate_v2_train(train_dir: Path) -> dict:
    metrics_path = train_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs = metrics["runs"]
    per_run = []
    for run in runs:
        cal = run["calibration"]["metrics"]
        curve = [{"threshold": point["threshold"], "coverage": point["coverage"],
                  "conditional_risk": point["conditional_risk"],
                  "dangerous_allow_rate": point["dangerous_allow_rate"],
                  "false_block_rate": point["false_block_rate"]}
                 for point in cal["risk_coverage"]]
        per_run.append({
            "model": run["model"], "seed": run["seed"], "n_train": run["n_train"],
            "n_train_groups": run["n_train_groups"],
            "train_history_final": run["train_history"][-1],
            "train_eval": {"accuracy": run["train_eval"]["accuracy"],
                           "loss": run["train_eval"]["loss"], "mode": run["train_eval"]["mode"]},
            "calibration": {"n": cal["n"], "macro_f1": cal["macro_f1"],
                            "accuracy": cal["accuracy"], "ece": cal["ece"], "ece_raw": cal["ece_raw"],
                            "brier": cal["brier"], "nll_raw": run["nll_raw"],
                            "nll_scaled": run["nll_scaled"], "temperature": run["temperature"],
                            "risk_curve": curve,
                            "risk_curve_note": "cal 全 risk curve（τ 网格逐点）；阈值只在 cal 选"},
            "eval_final": {name: {"n": value["n"], "macro_f1": value["macro_f1"],
                                  "accuracy": value["accuracy"], "ece": value["ece"],
                                  "brier": value["brier"],
                                  "diagnostic_tau0": {"coverage": value["diagnostic_tau0"]["coverage"],
                                                      "conditional_risk":
                                                          value["diagnostic_tau0"]["conditional_risk"]}}
                           for name, value in run["eval"].items()},
            "operating_threshold": run["operating_threshold"],
            "threshold_feasible": run["threshold_feasible"],
            "weights": run["weights"], "parity_gpu_in_batch": run["parity"],
            "memory_peak_mb": run["memory_peak_mb"],
            "seconds": {"build": run["build_seconds"], "train": run["train_seconds"],
                        "total": run["seconds"]},
            "truncation": {"train_sample_rows": run["truncation_train_sample"]["rows"],
                           "train_sample_truncated": run["truncation_train_sample"]["truncated_rows"]},
        })
    by_arm_eval = []
    for arm in ("frozen-joint", "finetuned-joint"):
        for eval_set in ("heldout", "challenge"):
            entries = [run for run in runs if run["model"] == arm]
            by_arm_eval.append({
                "model": arm, "eval_set": eval_set, "n_seeds": len(entries),
                "macro_f1": {"per_seed": {str(run["seed"]): run["eval"][eval_set]["macro_f1"]
                                          for run in entries},
                             **spread([run["eval"][eval_set]["macro_f1"] for run in entries])},
                "accuracy": spread([run["eval"][eval_set]["accuracy"] for run in entries]),
            })
    truncation = {name: {"rows": value["rows"], "truncated_rows": value["truncated_rows"]}
                  for name, value in metrics["truncation"].items()}
    return {
        "schema": "memory-relations-v2-train-aggregate-v1", "generated_at": _now(),
        "source": {"metrics_json": _rel(metrics_path),
                   "metrics_sha256": sha256_file(metrics_path),
                   "curve_jsonl_sha256": sha256_file(train_dir / "curve.jsonl"),
                   "predictions_jsonl_sha256": sha256_file(train_dir / "predictions.jsonl"),
                   "planned_manifest": metrics.get("planned_manifest"),
                   "planned_source_sha256": metrics["data"].get("source_sha256"),
                   "n_runs": len(runs)},
        "environment": metrics["environment"],
        "semantics": metrics["semantics"],
        "per_run": per_run, "by_arm_eval": by_arm_eval, "truncation": truncation,
        "training_effect": {
            "frozen_challenge_macro_f1": spread([run["eval"]["challenge"]["macro_f1"]
                                                 for run in runs if run["model"] == "frozen-joint"]),
            "finetuned_challenge_macro_f1": spread([run["eval"]["challenge"]["macro_f1"]
                                                    for run in runs if run["model"] == "finetuned-joint"]),
            "note": "同输入/同 head/同超参协议下两臂对比 → 可解释冻结效应；论文用三 seed 均值与范围"
                    "（finetuned ≈0.351，[0.308, 0.415]），不得取最大 seed 0.415 当效果；"
                    "v1→v2 差异（mean-pooling/128→CLS/384/joint 序列）不能唯一归因 finetune。",
        },
        "disclosures": DISCLOSURES,
    }


                                                                           

def _parity_compare(cpu_probs, gpu_probs, gpu_labels: list, ids: list,
                    tolerance: float = PARITY_TOLERANCE) -> dict:
    """CPU/GPU 概率矩阵对照：max|Δ|、argmax 一致率、逐行差异、complete 判定（纯函数）。"""
    cpu = np.asarray(cpu_probs, dtype=np.float64)
    gpu = np.asarray(gpu_probs, dtype=np.float64)
    if cpu.shape != gpu.shape or cpu.shape[0] != len(ids):
        raise Fail("parity_shape_mismatch", {"cpu": list(cpu.shape), "gpu": list(gpu.shape)})
    difference = np.abs(cpu - gpu)
    cpu_argmax = cpu.argmax(axis=1)
    mismatches = [{"id": ids[index], "cpu_argmax": int(cpu_argmax[index]),
                   "gpu_argmax": gpu_labels[index]}
                  for index in range(len(ids)) if int(cpu_argmax[index]) != gpu_labels[index]]
    max_abs = float(difference.max()) if difference.size else 0.0
    complete = not mismatches and max_abs <= tolerance
    return {"max_abs_diff": max_abs, "tolerance": tolerance,
            "argmax_agreement": {"numerator": len(ids) - len(mismatches), "denominator": len(ids),
                                 "value": (len(ids) - len(mismatches)) / len(ids) if ids else None},
            "argmax_mismatch_rows": mismatches, "parity_complete": complete,
            "status": "COMPLETE" if complete else "NOT_COMPLETE",
            "not_complete_reason": None if complete else
                ("argmax 与 GPU 历史不一致" if mismatches else
                 f"max_abs_diff {max_abs:.3g} > tolerance {tolerance:g}")}


def cpu_inference_parity(train_dir: Path, challenge_path: Path) -> dict:
    import time

    import torch

    from research.memory_relations import features as FEAT
    from research.memory_relations import finetune as F
    from research.memory_relations import train as T

    torch.set_num_threads(PARITY_THREADS)
    rows, excluded = T._pair_filter(T.load_rows(challenge_path, source="challenge"))
    ids = [row["id"] for row in rows]
    sample_sha = sha256_json({"ids": ids, "requests": [row["request"] for row in rows]})
    joint_sha = sha256_json([F.joint_text(row["request"]) for row in rows])
    gpu_rows: dict = {}
    for row in load_jsonl_strict(train_dir / "predictions.jsonl"):
        if row.get("eval_set") == "challenge":
            gpu_rows.setdefault((row["model"], row["seed"]), []).append(row)
    metrics = json.loads((train_dir / "metrics.json").read_text(encoding="utf-8"))
    gpu_weights = {(run["model"], run["seed"]): run["weights"]["sha256"]
                   for run in metrics["runs"]}
    gpu_parity = {(run["model"], run["seed"]): run["parity"] for run in metrics["runs"]}

    artifacts = []
    for directory in sorted((train_dir / "scorers").iterdir()):
        if not directory.is_dir():
            continue
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        kind, weights, seed = meta["kind"], meta["weights"], meta["train"]["seed"]
        key = (kind, seed)
        if key not in gpu_rows:
            raise Fail("parity_gpu_rows_missing", {"artifact": directory.name})
        stored = gpu_rows[key]
        if [row["id"] for row in stored] != ids:
            raise Fail("parity_sample_mismatch", {"artifact": directory.name})
        model_sha = sha256_file(directory / "model.pt")
        expected_weights = {"frozen-joint": "head_only", "finetuned-joint": "full"}[kind]
        started = time.perf_counter()
        scorer = FEAT.load_scorer(directory, device="cpu")                           
        load_seconds = time.perf_counter() - started
        scorer.batch_size = PARITY_BATCH
        started = time.perf_counter()
        cpu_probs = scorer.predict_proba([row["request"] for row in rows])
        infer_seconds = time.perf_counter() - started
        cpu_raw = scorer.predict_proba_raw([row["request"] for row in rows[:24]])
        raw_argmax_matches = bool(
            np.array_equal(cpu_raw.argmax(axis=1), cpu_probs[:24].argmax(axis=1)))
        gpu_probs = np.asarray([row["probs"] for row in stored], dtype=np.float64)
        gpu_labels = [FEAT.CLASSES.index(row["pred"]) for row in stored]
        comparison = _parity_compare(cpu_probs, gpu_probs, gpu_labels, ids)
        artifacts.append({
            "artifact": directory.name, "kind": kind, "weights": weights, "seed": seed,
            "checks": {
                "kind_weights_pairing": weights == expected_weights,
                "classes_match": tuple(meta["classes"]) == FEAT.CLASSES,
                "input_recipe": meta["input_recipe"], "max_length": meta["max_length"],
                "threshold": meta["threshold"], "threshold_feasible": meta["threshold_feasible"],
                "meta_sha256": sha256_file(directory / "meta.json"),
                "model_pt_sha256": model_sha,
                "model_sha_matches_meta": model_sha == meta["weights_sha256"],
                "checkpoint_sha_matches_gpu_record": gpu_weights.get(key) == model_sha,
                "backbone_fingerprint": meta["backbone_fingerprint"],
                "tokenizer_fingerprint": meta["tokenizer_fingerprint"],
            },
            "sample": {"rows": len(ids), "sample_sha256": sample_sha,
                       "joint_text_sha256": joint_sha,
                       "gpu_predicted_rows": [row["id"] for row in stored] == ids},
            "cpu_inference": {"batch_size": PARITY_BATCH, "threads": PARITY_THREADS,
                              "load_seconds": round(load_seconds, 3),
                              "infer_seconds": round(infer_seconds, 3),
                              "raw_argmax_equals_scaled_argmax_first24": raw_argmax_matches},
            "comparison": comparison,
            "gpu_in_batch_reload_parity_reference": {
                "value": gpu_parity.get(key),
                "note": "GPU 批内重载 parity（metrics.json 原值）；参考项，不得替代本 CPU-vs-GPU 对照。"},
        })
        del scorer
    complete = all(item["comparison"]["parity_complete"] for item in artifacts)
    return {
        "schema": "memory-relations-cpu-inference-parity-v1", "generated_at": _now(),
        "method": "同一批 GPU 训练 checkpoint（meta/model sha 钉死）→ 公共 features.load_scorer "
                  "CPU 推理 → 与 GPU predictions.jsonl（6 位概率、未舍入 argmax 的 pred 字段）"
                  "对照；challenge 135 对；tolerance 覆盖 fp32 内核差异与 6 位舍入。",
        "environment": {"cuda_visible_devices": "（由调用方置空，脚本不启 GPU）",
                        "cpu_threads": PARITY_THREADS, "cpu_batch_size": PARITY_BATCH,
                        "torch": __import__("torch").__version__},
        "challenge_pairs": len(ids),
        "sample_sha256": sample_sha, "joint_text_sha256": joint_sha,
        "tolerance": PARITY_TOLERANCE,
        "artifacts": artifacts,
        "overall": {"parity_complete": complete,
                    "status": "COMPLETE" if complete else "NOT_COMPLETE",
                    "note": "argmax 与 GPU 历史不一致时显式 NOT_COMPLETE，不掩盖、不写成通过。"},
        "disclosures": [
            "本对照是 CPU vs GPU 的推理 parity，不是 GPU 批内 reload parity（后者仅参考）。",
            "checkpoint/输入两侧钉死：kind/weights 配对、meta sha、model.pt sha == 训练记录与 meta、"
            "样本 id 顺序与请求 hash 全等，确保两设备读的是同一工件同一输入。",
            "概率存 6 位：对 max|Δ| 的贡献 ≤5e-7；若 argmax 变化（边界行）按 NOT_COMPLETE 处理。",
        ],
    }


                                                     

def run_selftest() -> dict:
    import numpy as np

    from research.memory_relations import train as T

    passes, fails = [], []

    def check(name: str, ok: bool) -> None:
        passes.append({"name": name, "ok": bool(ok)})

    gpu = np.asarray([[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03],
                      [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05]])
    labels = [0, 1]
    same = _parity_compare(gpu.copy(), gpu, labels, ["a", "b"])
    check("parity/identical-complete", same["parity_complete"] and same["max_abs_diff"] == 0.0)
    tiny = gpu.copy()
    tiny[0, 0] += 5e-5
    near = _parity_compare(tiny, gpu, labels, ["a", "b"])
    check("parity/within-tolerance-complete", near["parity_complete"])
    shifted = gpu.copy()
    shifted[0, 0] += 0.01
    value_break = _parity_compare(shifted, gpu, labels, ["a", "b"])
    check("parity/value-beyond-tolerance-not-complete", not value_break["parity_complete"])
    flipped = gpu.copy()
    flipped[1] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
    argmax_break = _parity_compare(flipped, gpu, labels, ["a", "b"])
    check("parity/argmax-mismatch-not-complete",
          (not argmax_break["parity_complete"]) and argmax_break["argmax_mismatch_rows"] == [
              {"id": "b", "cpu_argmax": 0, "gpu_argmax": 1}])
    fails.append({
        "name": "negative/value-only-检查不得漏判",
        "rejected": not value_break["parity_complete"] and value_break["argmax_mismatch_rows"] == [],
        "note": "概率被改但 argmax 不变：仅比 argmax 会误报通过，本函数必须按 max|Δ| 判 NOT_COMPLETE"})
    fails.append({
        "name": "negative/argmax-差异必须显式列出",
        "rejected": argmax_break["status"] == "NOT_COMPLETE" and bool(argmax_break["not_complete_reason"]),
        "note": "argmax 与 GPU 历史不一致 → NOT_COMPLETE 且给出原因，不允许写成通过"})
    check("spread/known", spread([0.2, 0.3, 0.4]) == {"n": 3, "mean": 0.3, "min": 0.2, "max": 0.4})
    check("rate/zero-denominator-null", rate_record(0, 0)["value"] is None)
    challenge, _excluded = T._pair_filter(T.load_rows(
        ROOT / "research" / "memory_relations" / "data" / "challenge.jsonl", source="challenge"))
    check("pair-filter/challenge-135", len(challenge) == 135)
    passed = (all(item["ok"] for item in passes) and all(item["rejected"] for item in fails))
    return {"selftest_passed": passed, "pass_cases": passes, "known_fail_cases": fails}


                                            

def render_conclusion(replay: dict, training: dict, parity: dict, date: str) -> str:
    lines = [f"# 记忆关系轻量模型：v3 后处理结论（{date}，脚本自动生成）", "",
             "- 三件套：本结论 + 脚本 `v3-postprocess.py` + 数据聚合 JSON（v3 回放 / v2-train / parity）；",
             "  v0/v1/v2 全部冻结为历史，口径不同不并表；无真人金标、一切率为 agreement，不转生产。", "",
             "## 1. 12 次真实回放（6 工件 × {set_challenge 37, challenge 147}，CPU 公共 loader）", "",
             "| 臂 | 数据集 | attempts/valid | gate_err/row_err | primary 均值[min,max] | acceptable | macroF1(valid) |",
             "|---|---|---|---|---|---|---|"]
    for agg in replay["aggregates"]:
        p, a, m = agg["primary"], agg["acceptable"], agg["macro_f1"]
        lines.append(f"| {agg['arm']} | {agg['dataset']} | {agg['counts']['attempts']}/"
                     f"{agg['counts']['valid_rows']} | {agg['counts']['gate_error_rows']}/"
                     f"{agg['counts']['row_error_rows']} | {p['mean']:.3f}[{p['min']:.3f},{p['max']:.3f}] | "
                     f"{a['mean']:.3f} | {m['mean']:.3f} |")
    lines += ["", f"- run_failures={replay['source']['run_failures']}（12 run exit 全在日志）；"
                  "逐 run gate 码与分母见聚合 JSON。", "",
              "## 2. v2-train（GPU 6 run）冻结 vs 微调（同输入/同 head/同超参协议）", "",
              "| 臂 | challenge F1 mean[min,max] | heldout F1 mean[min,max] | train_eval acc | peak 显存 | train 秒 |",
              "|---|---|---|---|---|---|"]
    for arm in ("frozen-joint", "finetuned-joint"):
        runs = [run for run in training["per_run"] if run["model"] == arm]
        ch = spread([run["eval_final"]["challenge"]["macro_f1"] for run in runs])
        he = spread([run["eval_final"]["heldout"]["macro_f1"] for run in runs])
        acc = spread([run["train_eval"]["accuracy"] for run in runs])
        mem = spread([run["memory_peak_mb"] for run in runs])
        sec = spread([run["seconds"]["train"] for run in runs])
        lines.append(f"| {arm} | {ch['mean']:.3f}[{ch['min']:.3f},{ch['max']:.3f}] | "
                     f"{he['mean']:.3f}[{he['min']:.3f},{he['max']:.3f}] | "
                     f"{acc['mean']:.3f} | {mem['mean']:.0f}MB | {sec['mean']:.1f} |")
    lines += ["", "- 论文口径：finetuned challenge F1 报三 seed 均值 ≈0.351 与范围 [0.308, 0.415]，"
              "**不用最大 seed 0.415 当效果**；frozen 同口径 ≈0.182。",
              "- 归因边界：两臂同协议 → 差值可解释「骨干是否冻结」；v1（mean-pooling/128）→ v2"
              "（CLS/384/joint 序列）多因素同时改变，v1 MiniLM ≈0.28 不能唯一归因 finetune。",
              "- 校准：F1/ECE/NLL 与全 risk curve（每 run 20 点在聚合 JSON）；threshold 6/6 为 null"
              "（无可行点，不转生产）；truncation 全 0（train/cal/heldout/challenge）。", "",
              "## 3. CPU 推理 parity（同一批 GPU checkpoint，CPU 重放 challenge 135 对）", "",
              "| 工件 | max|Δ| | argmax 一致 | 状态 |", "|---|---|---|---|"]
    for item in parity["artifacts"]:
        cmp = item["comparison"]
        lines.append(f"| {item['artifact']} | {cmp['max_abs_diff']:.3g} | "
                     f"{cmp['argmax_agreement']['numerator']}/{cmp['argmax_agreement']['denominator']} | "
                     f"{cmp['status']} |")
    lines += ["", f"- 总判定：**{parity['overall']['status']}**（tolerance {parity['tolerance']:g}；"
                  "argmax 与 GPU 历史不一致 → 显式 NOT_COMPLETE，不写成通过）。",
              "- parity 为 CPU vs GPU 推理对照；GPU 批内 reload parity 仅作参考项，不得替代。", ""]
    return "\n".join(lines) + "\n"


                                        

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="记忆关系 v3 研究后处理（只读输入）")
    parser.add_argument("--report-dir",
                        default="<archive-root>")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args(argv)

    v3_dir = ROOT / "research" / "memory_relations" / "runs" / "v3-evaluate"
    v3_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train_dir) if args.train_dir else \
        ROOT / "research" / "memory_relations" / "runs" / "v2-train"
    report_dir = Path(args.report_dir)
    result_dir = Path(args.result_dir) if args.result_dir else report_dir / "结果数据"

    selftest = run_selftest()
    write_json(v3_dir / "v3-selftest.json", selftest)
    if not selftest["selftest_passed"]:
        print("error: selftest failed; no aggregate published", file=sys.stderr)
        return 3

    replay = aggregate_replays(v3_dir)
    training = aggregate_v2_train(train_dir)
    if args.skip_parity:
        parity = json.loads((v3_dir / "parity-cpu-vs-gpu.json").read_text(encoding="utf-8"))
    else:
        parity = cpu_inference_parity(train_dir, ROOT / "research" / "memory_relations"
                                      / "data" / "challenge.jsonl")
        write_json(v3_dir / "parity-cpu-vs-gpu.json", parity)

    write_json(v3_dir / "set-replay-aggregate-v3.json", replay)
    write_json(v3_dir / "v2-train-aggregate.json", training)
    date = datetime.now().strftime("%Y-%m-%d")
    write_json(result_dir / f"v3-set-replay-aggregate-{date}.json", replay)
    write_json(result_dir / f"v2-train-aggregate-{date}.json", training)
    write_json(result_dir / f"v3-parity-cpu-vs-gpu-{date}.json", parity)
    (result_dir / "v3-postprocess.py").write_bytes(Path(__file__).read_bytes())
    (result_dir / f"v3-结论-{date}.md").write_text(render_conclusion(replay, training, parity, date),
                                                   encoding="utf-8")

    print(json.dumps({
        "eval_runs": {"logged": replay["source"]["runs_logged"],
                      "aggregated": replay["source"]["runs_aggregated"],
                      "failures": replay["source"]["run_failures"]},
        "training_runs": training["source"]["n_runs"],
        "training_effect": {key: training["training_effect"][key] for key in
                            ("frozen_challenge_macro_f1", "finetuned_challenge_macro_f1")},
        "parity": {"status": parity["overall"]["status"], "tolerance": parity["tolerance"],
                   "per_artifact": {item["artifact"]: item["comparison"]["max_abs_diff"]
                                    for item in parity["artifacts"]}},
        "artifacts": {
            "v3_evaluate": ["set-replay-aggregate-v3.json", "v2-train-aggregate.json",
                            "parity-cpu-vs-gpu.json", "v3-selftest.json"],
            "external": sorted(path.name for path in result_dir.glob(f"v3-*{date}*"))
                        + sorted(path.name for path in result_dir.glob(f"v2-train-aggregate-{date}*")),
        },
    }, ensure_ascii=False, indent=2))
    if parity["overall"]["status"] != "COMPLETE":
        print("warning: parity NOT_COMPLETE（详见 parity-cpu-vs-gpu.json）", file=sys.stderr)
    return 4 if replay["source"]["run_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
