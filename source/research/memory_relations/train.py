"""记忆关系轻量模型的训练/校准/最终评测 CLI（研究-only；大实验由根协调者集成后运行）。

Usage:
  python -m research.memory_relations.train --data data.jsonl --challenge challenge.jsonl \\
      --out artifacts/memory-relations/run --models char-logreg,minilm-logreg,minilm-mlp \\
      --sizes 250,500,1000,2000 --seeds 13,20260919,424242 \\
      [--embedding-model <本地模型目录绝对路径>]

纪律：配对模型只接受 existing ≤1 的行（set 形状与多规则整集行显式跳过并报告，set 级评测
属 adapter.evaluate_sets）；采样按 group_id 分组且尺寸前缀嵌套且必须覆盖全部七类；温度缩放
与运行阈值只在 calibration 上选择，未选到可行阈值时记 null、不声明已选操作；heldout/OOD/
challenge 只用于最终评测；dev独立诊断，group/root跨分区泄漏一律 fail-loud（无绕过开关）。
--eval-splits none 必须使用仅train/dev/calibration的独立开发文件；不打开challenge或用终测parity。
partitioned 训练只用prob/policy的candidate阶段，拒绝audit且顶层阈值null；固定study的唯一候选
经--mode lock绑定后才可--mode audit独立一次审计。旧legacy/complete数学合同保留，不作反复认证。
校准缺共享API拒绝训练，不降级。输出已存在、任一seed/size缺类先失败。
嵌入走本地权重、禁止下载；产物只写 --out 目录（显式--cache-dir可复用已有缓存）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.memory_relations import features as F
from research.memory_relations import metrics as EM

                                                           
from research.memory_relations.features import RelationScorer, load_scorer              

MODEL_KINDS = ("majority", "char-logreg", "minilm-logreg", "minilm-mlp")
EMBEDDING_KINDS = ("minilm-logreg", "minilm-mlp")
SPLITS = ("train", "dev", "calibration", "heldout", "ood", "set")
EVAL_SPLITS = ("heldout", "ood", "challenge")
TRAINING_SCHEMA = "memory-relations-training-v1"


def load_rows(path: "str | Path", *, source: str,
              selected_splits: "Sequence[str] | None" = None,
              calibration_stage: "str | None" = None) -> list[dict]:
    """候选开发文件不能混入audit/终测；发现不允许角色即停止，不投影或评分。"""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"JSONL 不存在：{file}")
    rows: list[dict] = []
    with file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{file}:{line_number} JSON 解析失败：{error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{file}:{line_number} 行必须是 JSON 对象")
            split = row.get("split")
            if split is None:
                if source != "challenge":
                    raise ValueError(f"{file}:{line_number} 缺少 split 字段")
            elif split not in SPLITS:
                raise ValueError(f"{file}:{line_number} split 必须是 {SPLITS} 之一，实际 {split!r}")
            if selected_splits is not None and split not in selected_splits:
                if set(selected_splits) == {"train", "dev", "calibration"}:
                    raise ValueError("调参数据含终测/set分区：请提供仅train/dev/calibration的独立开发文件")
                if source == "audit":
                    raise ValueError("独立审计文件仅允许calibration audit，不读取其它分区")
                continue
            meta = row.get("phenomena")
            role = meta.get("calibration_role") if isinstance(meta, dict) else None
            if split == "calibration" and calibration_stage == "candidate" and role == "audit":
                raise ValueError("candidate开发文件禁止audit，请交付物理分离的prob/policy文件")
            if source == "audit" and (split != "calibration" or role != "audit"):
                raise ValueError("独立审计文件只能含calibration audit角色")
            if row.get("label") not in F.CLASSES:
                raise ValueError(f"{file}:{line_number} label 必须是 spec 七类之一")
            if not isinstance(row.get("request"), dict):
                raise ValueError(f"{file}:{line_number} request 必须是对象")
            _require_group_ids([row], "challenge" if source == "challenge" else split)
            if split != "set":
                F.request_view(row["request"])
            rows.append(row)
    return rows


def nested_group_indices(rows: Sequence[dict], sizes: Sequence[int], seed: int) -> dict[int, np.ndarray]:
    """按 group_id 分组、组序由 seed 置换、尺寸前缀嵌套的确定性采样。

    组整体进出（绝不拆组）；实际样本量可能略大于请求尺寸（组粒度），调用方按
    len(indices) 记录真实训练行数。
    """
    unique_sizes = sorted({int(size) for size in sizes})
    if not unique_sizes or unique_sizes[0] < 1:
        raise ValueError("--sizes 必须是非空正整数列表")
    if unique_sizes[-1] > len(rows):
        raise ValueError(f"请求尺寸 {unique_sizes[-1]} 超过 train 可用行数 {len(rows)}")
    _require_group_ids(rows, "train")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(index)
    ordered = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(ordered))
    sampled: list[int] = []
    cursor = 0
    result: dict[int, np.ndarray] = {}
    for size in unique_sizes:
        while len(sampled) < size and cursor < len(order):
            sampled.extend(groups[ordered[order[cursor]]])
            cursor += 1
        result[size] = np.asarray(sorted(sampled), dtype=int)
    return result


def _pair_filter(rows: Sequence[dict]) -> "tuple[list[dict], Counter]":
    """配对模型只接受 existing ≤1 的行；set 形状与多规则整集行显式跳过并计数。

    整集均值特征与 adapter 的逐对口径不一致，多规则行绝不静默参与训练/配对指标；
    set 级评测属于 adapter.evaluate_sets，本模块只报告实际配对行与排除数。
    """
    kept: list[dict] = []
    excluded: Counter = Counter()
    for row in rows:
        if row.get("split") == "set":
            excluded["set_split"] += 1
            continue
        existing = row["request"].get("existing")
        if isinstance(existing, list) and len(existing) > 1:
            excluded["multi_rule"] += 1
            continue
        kept.append(row)
    return kept, excluded


def _group_overlaps(*named_rows: "tuple[str, Sequence[dict]]") -> dict[str, int]:
    groups = {name: {row["group_id"] for row in rows if row.get("group_id")}
              for name, rows in named_rows if rows}
    names = sorted(groups)
    return {f"{left}∩{right}": len(groups[left] & groups[right])
            for position, left in enumerate(names) for right in names[position + 1:]}


def fit_model(kind: str, inputs, y: np.ndarray, seed: int):
    """拟合单个学生模型；样本未覆盖全部七类（含单类别）时 fail-loud，绝不伪造 7 维输出。"""
    y = np.asarray(y, dtype=int)
    present = set(np.unique(y).tolist())
    if len(present) < 2:
        raise ValueError("训练样本只有单一类别：拒绝伪造 7 维输出，请补充其它类别的训练行")
    _require_seven_classes(y)
    if kind == "majority":
        prior = np.bincount(y, minlength=len(F.CLASSES)).astype(np.float64)
        prior /= prior.sum()
        return prior, [int(index) for index in range(len(F.CLASSES)) if prior[index] > 0.0]
    if kind == "char-logreg":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)),
            ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                       random_state=seed)),
        ])
        pipeline.fit(inputs, y)
        return pipeline, [int(label) for label in pipeline.classes_]
    matrix = np.asarray(inputs, dtype=np.float64)
    if kind == "minilm-logreg":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                       random_state=seed)),
        ])
        pipeline.fit(matrix, y)
        return pipeline, [int(label) for label in pipeline.classes_]
    if kind == "minilm-mlp":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                                  random_state=seed)),
        ])
        pipeline.fit(matrix, y)
        return pipeline, [int(label) for label in pipeline.classes_]
    raise ValueError(f"未知模型类型：{kind}")


def save_scorer(scorer: F.RelationScorer, directory: Path, *, threads: int, batch_size: int) -> None:
    """持久化打分器工件（model.joblib）：仅模型参数与推理所需配置，不含训练行内容。"""
    embedder = scorer.embedder
    payload = {
        "schema": F.ARTIFACT_SCHEMA,
        "kind": scorer.kind,
        "classes": list(F.CLASSES),
        "class_indices": list(scorer.class_indices),
        "model": scorer.model,
        "temperature": scorer.temperature,
        "feature_recipe": F.FEATURE_RECIPE,
        "embedder_path": str(embedder.model_dir) if embedder else None,
        "embedding_fingerprint": embedder.fingerprint if embedder else None,
        "threads": int(threads),
        "batch_size": int(batch_size),
    }
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, directory / "model.joblib")


def check_reload_parity(scorer: F.RelationScorer, directory: Path, probe_requests: Sequence[dict],
                        *, device: str, atol: float = 1e-6) -> dict:
    """joblib 重载 parity：校准后概率与 argmax 必须一致，否则 fail-loud。"""
    reloaded = F.load_scorer(directory, device=device)
    before = scorer.predict_proba(list(probe_requests))
    after = reloaded.predict_proba(list(probe_requests))
    difference = float(np.max(np.abs(before - after))) if len(before) else 0.0
    argmax_match = bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1)))
    ok = (argmax_match and difference <= atol
          and tuple(reloaded.classes) == F.CLASSES
          and tuple(reloaded.class_indices) == tuple(scorer.class_indices)
          and reloaded.temperature == scorer.temperature)
    if not ok:
        raise RuntimeError(f"重载 parity 失败：max_abs_diff={difference}, argmax_match={argmax_match}")
    return {"max_abs_diff": difference, "argmax_match": argmax_match, "ok": True}


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_ints(text: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in _parse_csv(text)]
    except ValueError as error:
        raise ValueError(f"{name} 必须是逗号分隔的整数") from error
    if not values:
        raise ValueError(f"{name} 不能为空")
    return values


def _parse_models(text: str) -> list[str]:
    models = list(dict.fromkeys(_parse_csv(text)))
    unknown = [kind for kind in models if kind not in MODEL_KINDS]
    if not models or unknown:
        raise ValueError(f"--models 只支持 {MODEL_KINDS}，实际 {unknown or text!r}")
    return models


def _labels(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([F.CLASSES.index(row["label"]) for row in rows], dtype=int)


def _triggers(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([F.request_has_trigger(row["request"]) for row in rows], dtype=bool)


def _require_group_ids(rows: Sequence[dict], name: str) -> None:
    if any(not isinstance(row.get("group_id"), str) or not row["group_id"].strip()
           for row in rows):
        raise ValueError(f"{name} 分区缺少非空 group_id：禁止单行 cluster 回退")


def _group_ids(rows: Sequence[dict]) -> list[str]:
    _require_group_ids(rows, "输入")
    return [row["group_id"] for row in rows]


def _root_ids(rows: Sequence[dict]) -> list[str]:
    groups = _group_ids(rows)
    roots = []
    for row, group in zip(rows, groups):
        meta = row.get("phenomena")
        root = meta.get("root_id", group) if isinstance(meta, dict) else group
        if not isinstance(root, str) or not root.strip():
            raise ValueError("语义根root_id必须为非空字符串")
        roots.append(root)
    return roots


def _require_seven_classes(labels: np.ndarray) -> None:
    present = set(np.unique(labels).tolist())
    missing = [name for index, name in enumerate(F.CLASSES) if index not in present]
    if missing:
        raise ValueError(f"训练样本未覆盖全部七类，缺 {missing}：拒绝训练")


def preflight_training(rows: Sequence[dict], sizes: Sequence[int], seeds: Sequence[int],
                       out_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    if out_dir.exists():
        raise FileExistsError(f"输出已存在：{out_dir}；禁止覆盖历史实验")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2 ** 32 for seed in seeds):
        raise ValueError("--seeds 必须互异且在 [0, 2**32) 内")
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 1 for size in sizes):
        raise ValueError("--sizes 必须是互异正整数")
    _require_group_ids(rows, "train")
    maps = {seed: nested_group_indices(rows, sizes, seed) for seed in seeds}
    for seed, samples in maps.items():
        for size, indices in samples.items():
            try:
                _require_seven_classes(_labels([rows[index] for index in indices]))
            except ValueError as error:
                raise ValueError(f"seed={seed} size={size}: {error}") from error
    return maps


def parse_eval_splits(text: str) -> tuple[str, ...]:
    if text == "none":
        return ()
    values = _parse_csv(text)
    if not values or len(set(values)) != len(values) or set(values) - set(EVAL_SPLITS):
        raise ValueError(f"--eval-splits 只支持 none 或 {EVAL_SPLITS} 的逗号列表")
    return tuple(name for name in EVAL_SPLITS if name in values)


def _calibration_roles(rows: Sequence[dict]) -> tuple[list[str], np.ndarray]:
    roles, primary = [], []
    role_by_group: dict[str, str] = {}
    primaries: Counter = Counter()
    for row, root in zip(rows, _root_ids(rows)):
        metadata = row.get("phenomena")
        if not isinstance(metadata, dict) or metadata.get("calibration_role") not in ("prob", "policy", "audit") \
                or type(metadata.get("operating_primary")) is not bool:
            raise ValueError("partitioned calibration 必须有 calibration_role 与布尔 operating_primary")
        role, group = metadata["calibration_role"], root
        if group in role_by_group and role_by_group[group] != role:
            raise ValueError("calibration_role 不能在同一个语义根内变化")
        role_by_group[group] = role
        primaries[group] += int(metadata["operating_primary"])
        roles.append(role)
        primary.append(metadata["operating_primary"])
    if any(primaries[group] != 1 for group in role_by_group):
        raise ValueError("partitioned calibration 每组必须恰有一个 operating_primary")
    return roles, np.asarray(primary, dtype=bool)


def prepare_data(args) -> dict:
    evaluation = parse_eval_splits(args.eval_splits)
    stage = getattr(args, "calibration_stage", "candidate") if args.calibration_protocol == "partitioned" else None
    if stage not in (None, "candidate"):
        raise ValueError("训练CLI仅允许candidate校准，审计须对唯一锁定工件独立执行")
    selected = ("train", "dev", "calibration") + tuple(name for name in evaluation if name != "challenge")
    if evaluation:
        selected += ("set",)
    rows = load_rows(args.data, source="data", selected_splits=selected, calibration_stage=stage)
    by_split = {name: [row for row in rows if row["split"] == name] for name in SPLITS}
    pairs, excluded = {}, {}
    for name, source_rows in by_split.items():
        pairs[name], counts = _pair_filter(source_rows)
        excluded[name] = dict(counts)
    challenge_path = Path(args.challenge) if args.challenge and "challenge" in evaluation else None
    challenge_all = load_rows(challenge_path, source="challenge") if challenge_path else []
    challenge_rows, challenge_counts = _pair_filter(challenge_all)
    pairs["challenge"], excluded["challenge"] = challenge_rows, dict(challenge_counts)
    if not pairs["train"]:
        raise ValueError("--data 的 train 分区没有可用的配对行（existing ≤1）")
    if not evaluation and not pairs["dev"]:
        raise ValueError("调参none必须提供独立dev配对行：parity与开发诊断只用dev")
    calibration_rows = pairs["calibration"]
    if not calibration_rows or len({row["label"] for row in calibration_rows}) < 2:
        raise ValueError("calibration 至少需要两个类别的可用配对行")
    calibration_diagnostic_mask = np.ones(len(calibration_rows), dtype=bool)
    if args.calibration_protocol == "partitioned":
        if not callable(getattr(EM, "calibrate_partitioned", None)):
            raise ValueError("metrics.calibrate_partitioned 尚未集成：拒绝训练，禁止回退 legacy")
        roles, _primary = _calibration_roles(calibration_rows)
        calibration_diagnostic_mask = np.asarray(roles) != "audit"
    _root_ids(pairs["dev"])
    named_rows = tuple(by_split.items()) + (("challenge", challenge_all),)
    overlaps = _group_overlaps(*named_rows)
    root_rows = [(name, [{"group_id": root} for root in _root_ids(source_rows)])
                 for name, source_rows in named_rows]
    root_overlaps = _group_overlaps(*root_rows)
    if any(overlaps.values()) or any(root_overlaps.values()):
        raise ValueError(f"group_id/root_id 跨分区重叠（评测泄漏）：{overlaps}, {root_overlaps}")
    eval_sets = [(name, pairs[name]) for name in evaluation]
    if evaluation and not any(source_rows for _, source_rows in eval_sets):
        raise ValueError("没有可评测的配对数据：提供选中的 heldout/ood/--challenge，或用 --eval-splits none")
    return {"by_split": by_split, "pairs": pairs, "excluded": excluded,
            "eval_sets": eval_sets, "challenge_path": challenge_path,
            "calibration_diagnostic_mask": calibration_diagnostic_mask,
            "calibration_diagnostic_rows": [row for row, selected in zip(calibration_rows, calibration_diagnostic_mask)
                                            if selected],
            "overlaps": overlaps, "root_overlaps": root_overlaps, "eval_splits": evaluation}


def calibrate(logits: np.ndarray, rows: Sequence[dict], *, protocol: str,
              fit_temperature: bool = True, stage: str = "complete") -> dict:
    labels, triggers = _labels(rows), _triggers(rows)
    if protocol == "partitioned":
        roles, primary = _calibration_roles(rows)
        arguments = (logits, labels, triggers, _root_ids(rows), roles, primary)
        result = EM.calibrate_partitioned(*arguments, stage=stage) if stage != "complete" else EM.calibrate_partitioned(*arguments)
        required = {"temperature", "threshold", "threshold_feasible", "selection", "audit"}
        if not isinstance(result, dict) or not required <= set(result):
            raise ValueError("calibrate_partitioned 返回合同不完整")
    elif protocol == "legacy":
        temperature = EM.fit_temperature(logits, labels) if fit_temperature else 1.0
        threshold, selection = EM.select_threshold(F.softmax(logits / temperature), labels, triggers)
        result = {"temperature": temperature, "threshold": threshold,
                  "threshold_feasible": selection is not None, "selection": selection, "audit": None}
    else:
        raise ValueError("calibration protocol 无效")
    temperature = float(result["temperature"])
    threshold, feasible = result["threshold"], result["threshold_feasible"]
    if not np.isfinite(temperature) or temperature <= 0 or type(feasible) is not bool \
            or (threshold is not None) != feasible \
            or (threshold is not None and (not np.isfinite(threshold) or threshold < 0)):
        raise ValueError("校准温度或threshold/threshold_feasible合同不符")
    return result


AUDIT_LOCK_SCHEMA = "memory-relations-audit-lock-v1"


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_contract(prepared) -> dict:
    members = {"train": prepared["pairs"]["train"], "dev": prepared["pairs"]["dev"]}
    calibration = prepared["pairs"]["calibration"]
    for role in ("prob", "policy"):
        members[role] = [row for row in calibration if isinstance(row.get("phenomena"), dict)
                         and row["phenomena"].get("calibration_role") == role]
    return {"sha256_by_partition": {key: _json_sha(sorted(rows, key=lambda row: str(row.get("id"))))
                                    for key, rows in members.items()},
            "root_ids_by_partition": {key: sorted(set(_root_ids(rows))) for key, rows in members.items()},
            "row_counts": {key: len(rows) for key, rows in members.items()},
            "unit": "semantic_root_not_descendant_rows", "audit_visible": False}


def _artifact_binding(directory: Path) -> tuple[dict, dict]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("工件meta不是对象")
    filename = "model.pt" if meta.get("schema") == "memory-relations-finetune-scorer-v1" else "model.joblib"
    return meta, {name: _file_sha(directory / name) for name in ("meta.json", filename)}


def _candidate_calibration(meta: dict) -> dict:
    calibration = meta.get("calibration")
    if calibration is None and isinstance(meta.get("train"), dict):
        calibration = meta["train"].get("calibration")
    if not isinstance(calibration, dict) or calibration.get("stage") != "candidate" \
            or not isinstance(calibration.get("audit"), dict) \
            or calibration["audit"].get("status") != "deferred" \
            or meta.get("threshold") is not None or meta.get("threshold_feasible") is not False:
        raise ValueError("唯一锁只接受未审计candidate工件，不接受旧combined/已审计工件")
    temperature = calibration.get("temperature")
    selection = calibration.get("selection")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not np.isfinite(temperature) or temperature <= 0 or meta.get("temperature") != temperature \
            or not isinstance(selection, dict) or type(selection.get("point_feasible")) is not bool:
        raise ValueError("candidate温度/selection合同无效")
    candidate = selection.get("candidate_threshold")
    if (candidate is not None) != selection["point_feasible"] or (candidate is not None and (
            isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not np.isfinite(candidate)
            or not 0 <= candidate <= 1)):
        raise ValueError("candidate阈值与policy可行标记不一致")
    return calibration


def _run_path(path) -> Path:
    resolved = Path(path).resolve()
    allowed = (Path(__file__).parent / "runs").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError("研究锁/审计输出必须是research/memory_relations/runs的新子目录")
    return resolved


def lock_candidate(model_dir, lock_dir, *, study_id: str) -> dict:
    if not isinstance(study_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", study_id):
        raise ValueError("study-id必须为预注册固定标识，不含路径或秘密")
    directory, target = Path(model_dir).resolve(), _run_path(lock_dir)
    if target.exists() or directory == target or directory in target.parents or target in directory.parents:
        raise FileExistsError("study锁已存在或与工件目录重叠；不允许换候选/覆盖")
    meta, binding = _artifact_binding(directory)
    calibration = _candidate_calibration(meta)
    scorer = F.load_scorer(directory, device="cpu")
    if tuple(scorer.classes) != F.CLASSES or scorer.temperature != calibration["temperature"]:
        raise ValueError("公共loader与candidate温度/类序不符")
    contract = meta.get("training_input_contract")
    if contract is None and isinstance(meta.get("train"), dict):
        contract = meta["train"].get("training_input_contract")
    partitions = {"train", "dev", "prob", "policy"}
    if not isinstance(contract, dict) or contract.get("audit_visible") is not False \
            or contract.get("unit") != "semantic_root_not_descendant_rows" \
            or any(not isinstance(contract.get(field), dict) or set(contract[field]) != partitions
                   for field in ("root_ids_by_partition", "sha256_by_partition", "row_counts")):
        raise ValueError("缺少candidate独立开发分区血缘")
    for partition in partitions:
        roots, digest, count = (contract[field][partition] for field in
                                ("root_ids_by_partition", "sha256_by_partition", "row_counts"))
        if not isinstance(roots, list) or any(not isinstance(root, str) or not root.strip() for root in roots) \
                or len(set(roots)) != len(roots) or not isinstance(digest, str) \
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or type(count) is not int or count < len(roots):
            raise ValueError("candidate开发分区血缘结构无效")
    lock = {"schema": AUDIT_LOCK_SCHEMA, "study_id": study_id, "model_dir": str(directory),
            "created_at": datetime.now().isoformat(timespec="seconds"), "artifact_sha256": binding,
            "classes": list(F.CLASSES), "calibration": calibration, "training_input_contract": contract,
            "selection_basis": "pre_registered_dev_only_not_audit_or_terminal",
            "human_gold": False, "one_audit_per_study_lock": True}
    target.mkdir(parents=True, exist_ok=False)
    with (target / "lock.json").open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (target / "commitment.json").open("x", encoding="utf-8") as handle:
        json.dump({"lock_sha256": _file_sha(target / "lock.json")}, handle)
    return {"lock_dir": str(target), "study_id": study_id, "locked": True,
            "candidate_threshold": calibration["selection"]["candidate_threshold"], "audit_performed": False}


def audit_candidate(lock_dir, audit_data, out_dir, *, device: str = "cpu") -> dict:
    target, output = _run_path(lock_dir), _run_path(out_dir)
    commitment = json.loads((target / "commitment.json").read_text(encoding="utf-8"))
    if commitment.get("lock_sha256") != _file_sha(target / "lock.json"):
        raise ValueError("锁内容被改，禁止读取audit")
    lock = json.loads((target / "lock.json").read_text(encoding="utf-8"))
    if lock.get("schema") != AUDIT_LOCK_SCHEMA or lock.get("classes") != list(F.CLASSES):
        raise ValueError("study锁schema/类序不符")
    directory = Path(lock["model_dir"])
    meta, binding = _artifact_binding(directory)
    if binding != lock.get("artifact_sha256") or _candidate_calibration(meta) != lock.get("calibration"):
        raise ValueError("候选工件/温度/阈值变动，禁止读取audit")
    if output.exists() or output == target or target in output.parents or output in target.parents \
            or output == directory or directory in output.parents or output in directory.parents:
        raise FileExistsError("审计输出已存在或与锁/旧工件重叠")
    scorer = F.load_scorer(directory, device=device)
    if scorer.temperature != lock["calibration"]["temperature"]:
        raise ValueError("公共loader温度与锁不符")
    with (target / "audit-attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now().isoformat(timespec="seconds"),
                   "lock_sha256": commitment["lock_sha256"], "audit_data": str(audit_data)}, handle)
    calibration = lock["calibration"]
    selection = calibration["selection"]
    rows = []
    if selection["candidate_threshold"] is not None:
        rows = load_rows(audit_data, source="audit", selected_splits=("calibration",))
        if not rows or any(len(row["request"].get("existing", [])) > 1 for row in rows):
            raise ValueError("audit必须有独立calibration pair行")
        roles, primary = _calibration_roles(rows)
        roots = _root_ids(rows)
        prior_roots = {root for values in lock["training_input_contract"]["root_ids_by_partition"].values()
                       for root in values}
        if prior_roots & set(roots):
            raise ValueError("audit语义根与train/dev/prob/policy重叠")
        raw = scorer.predict_proba_raw([row["request"] for row in rows])
        logits = np.log(np.clip(raw, 1e-12, None))
        result = EM.audit_locked_partition(logits, _labels(rows), _triggers(rows), roots, roles, primary,
                                           temperature=calibration["temperature"],
                                           candidate_threshold=selection["candidate_threshold"],
                                           point_feasible=selection["point_feasible"])
    else:
        result = EM.audit_locked_partition(np.zeros((0, len(F.CLASSES))), np.asarray([], dtype=int),
                                           np.asarray([], dtype=bool), [], [], np.asarray([], dtype=bool),
                                           temperature=calibration["temperature"], candidate_threshold=None,
                                           point_feasible=False)
    result["selection"] = selection
    result.update(study_id=lock["study_id"], lock_sha256=commitment["lock_sha256"],
                  candidate_artifact_sha256=binding, audit_data_read=bool(rows), human_gold=False)
    output.mkdir(parents=True, exist_ok=False)
    exported = output / "scorer"
    exported.mkdir()
    for filename in binding:
        if filename != "meta.json":
            shutil.copyfile(directory / filename, exported / filename)
    audited_meta = copy.deepcopy(meta)
    audited_meta.update(threshold=result["threshold"], threshold_feasible=result["threshold_feasible"],
                        calibration=result, audit_lock_sha256=commitment["lock_sha256"])
    if isinstance(audited_meta.get("train"), dict):
        audited_meta["train"]["calibration"] = result
    (exported / "meta.json").write_text(json.dumps(audited_meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    F.load_scorer(exported, device="cpu")
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"out": str(output), "scorer": str(exported), **result}


def _default_out(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "memory-relations" / datetime.now().strftime("train-%Y%m%d-%H%M%S")


def resolve_embedding_model(explicit: str, repo_root: Path) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"--embedding-model 目录不存在：{path}（禁止下载）")
        return path
    default = repo_root / "models" / "embeddings" / F.EMBEDDING_DEFAULT_NAME
    if default.is_dir():
        return default
    raise FileNotFoundError(
        f"未找到本地嵌入模型 {default}；用 --embedding-model 指定本地目录（禁止下载）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "lock", "audit"), default="train")
    parser.add_argument("--data", default="", help="train独立开发JSONL；audit仅审计角色JSONL；lock不读数据")
    parser.add_argument("--model-dir", default="", help="lock的唯一已按dev选定candidate工件")
    parser.add_argument("--lock-dir", default="", help="固定study独占锁目录，audit只能读取已有锁")
    parser.add_argument("--study-id", default="", help="lock的预注册study标识；不可在audit后改名重选")
    parser.add_argument("--calibration-stage", choices=("candidate",), default="candidate",
                        help="partitioned训练不读audit；审计只用--mode lock/audit独立一次流程")
    parser.add_argument("--challenge", default="",
                        help="配对 challenge JSONL（agent_review 标签；只用于最终评测）")
    parser.add_argument("--out", default="", help="输出目录（默认 artifacts/memory-relations/ 下）")
    parser.add_argument("--models", default="char-logreg,minilm-logreg,minilm-mlp")
    parser.add_argument("--sizes", default="250,500,1000,2000")
    parser.add_argument("--seeds", default="13,20260919,424242")
    parser.add_argument("--embedding-model", default="",
                        help="本地 MiniLM 目录绝对路径（禁止下载）")
    parser.add_argument("--device", default=F.DEVICE_DEFAULT)
    parser.add_argument("--threads", type=int, default=F.THREADS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=F.BATCH_DEFAULT)
    parser.add_argument("--bootstrap-samples", type=int, default=EM.BOOTSTRAP_SAMPLES)
    parser.add_argument("--cache-dir", default="", help="嵌入缓存目录（默认 <out>/embedding_cache）")
    parser.add_argument("--eval-splits", default="heldout,ood,challenge",
                        help="none（调参，不打开独立终测文件）或 heldout,ood,challenge 的子集")
    parser.add_argument("--calibration-protocol", choices=("legacy", "partitioned"), default="legacy")
    return parser


def main(argv: "Sequence[str] | None" = None) -> int:
                                                        
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if args.mode == "lock":
        if not args.model_dir or not args.lock_dir or not args.study_id or args.data or args.challenge:
            raise ValueError("lock只提供--model-dir --lock-dir --study-id，不接受数据/终测")
        print(json.dumps(lock_candidate(args.model_dir, args.lock_dir, study_id=args.study_id), ensure_ascii=False))
        return 0
    if args.mode == "audit":
        if not args.lock_dir or not args.data or not args.out or args.model_dir or args.challenge:
            raise ValueError("audit只提供--lock-dir --data --out，不允许重选模型/终测")
        print(json.dumps(audit_candidate(args.lock_dir, args.data, args.out, device=args.device), ensure_ascii=False))
        return 0
    if not args.data or args.model_dir or args.lock_dir or args.study_id:
        raise ValueError("train必须提供--data，锁/审计另用--mode")
    models = _parse_models(args.models)
    sizes = sorted(_parse_ints(args.sizes, "--sizes"))
    seeds = _parse_ints(args.seeds, "--seeds")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples 必须 ≥ 1")
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("--threads / --batch-size 必须 ≥ 1")

    out_dir = Path(args.out) if args.out else _default_out(repo_root)
    if out_dir.exists():
        raise FileExistsError(f"输出已存在：{out_dir}；禁止覆盖历史实验")
    data_path = Path(args.data)
    prepared_data = prepare_data(args)
    input_contract = _data_contract(prepared_data)
    by_split, pair_by_split = prepared_data["by_split"], prepared_data["pairs"]
    excluded_by_split = prepared_data["excluded"]
    train_rows, calibration_rows = pair_by_split["train"], pair_by_split["calibration"]
    dev_rows = pair_by_split["dev"]
    challenge_path, challenge_rows = prepared_data["challenge_path"], pair_by_split["challenge"]
    challenge_excluded = excluded_by_split["challenge"]
    eval_sets, overlaps = prepared_data["eval_sets"], prepared_data["overlaps"]
    sample_maps = preflight_training(train_rows, sizes, seeds, out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    embedder = None
    if any(kind in EMBEDDING_KINDS for kind in models):
        embedding_dir = resolve_embedding_model(args.embedding_model, repo_root)
        cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "embedding_cache"
        embedder = F.Embedder(embedding_dir, device=args.device, threads=args.threads,
                              batch_size=args.batch_size, cache_dir=cache_dir)

    def build_inputs(kind: str, source_rows: Sequence[dict]):
        if kind == "majority":
            return None
        if kind == "char-logreg":
            return [F.request_text(row["request"]) for row in source_rows]
        return F.minilm_feature_matrix(embedder, [row["request"] for row in source_rows])

    prepared = {kind: build_inputs(kind, train_rows) for kind in models}
    if embedder is not None:
        embedder.save_cache()
    y_train_all = _labels(train_rows)
    calibration_requests = [row["request"] for row in calibration_rows]
    probe_requests = [row["request"] for row in (dev_rows or train_rows)[:8]]

    runs: list[dict] = []
    curve_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for seed in seeds:
        for size in sizes:
            indices = sample_maps[seed][size]
            y_train = y_train_all[indices]
            for kind in models:
                started = time.perf_counter()
                bank = prepared[kind]
                if kind == "majority":
                    train_inputs = None
                elif kind in EMBEDDING_KINDS:
                    train_inputs = bank[indices]
                else:
                    train_inputs = [bank[index] for index in indices]
                model, class_indices = fit_model(kind, train_inputs, y_train, seed)
                scorer = F.RelationScorer(
                    kind=kind, model=model, class_indices=class_indices,
                    embedder=embedder if kind in EMBEDDING_KINDS else None)
                raw_calibration = scorer.predict_proba_raw(calibration_requests)
                calibration = calibrate(np.log(np.clip(raw_calibration, 1e-12, None)), calibration_rows,
                                        protocol=args.calibration_protocol, fit_temperature=kind != "majority",
                                        stage=args.calibration_stage)
                scorer.temperature = float(calibration["temperature"])
                operating_threshold, feasible_point = calibration["threshold"], calibration["selection"]
                if operating_threshold is None:
                    print(f"[{seed}] size={size} {kind}: 无可行阈值（不声明已选操作策略）", flush=True)

                eval_metrics: dict[str, dict] = {}
                for name, eval_rows in eval_sets:
                    if not eval_rows:
                        continue
                    requests = [row["request"] for row in eval_rows]
                    probs = scorer.predict_proba(requests)
                    raw_probs = scorer.predict_proba_raw(requests)
                    trigger_values = _triggers(eval_rows)
                    result = EM.evaluate(
                        probs, _labels(eval_rows), raw_probs=raw_probs,
                        has_trigger=trigger_values, group_ids=_group_ids(eval_rows),
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
                            "label": row["label"], "pred": F.CLASSES[int(probs[position].argmax())],
                            "has_trigger": bool(trigger_values[position]),
                            "verdict": verdict,
                            "allow": None if verdict is None else verdict in EM.ALLOWING_CLASSES,
                            "verdict_tau0": verdict_tau0,
                            "allow_tau0": verdict_tau0 in EM.ALLOWING_CLASSES,
                            "probs": [round(float(value), 6) for value in probs[position]],
                        })

                artifact_dir = out_dir / "scorers" / f"{kind}-size{size}-seed{seed}"
                save_scorer(scorer, artifact_dir, threads=args.threads,
                            batch_size=args.batch_size)
                parity = check_reload_parity(scorer, artifact_dir, probe_requests,
                                             device=args.device)
                (artifact_dir / "meta.json").write_text(json.dumps({
                    "kind": kind, "size_requested": size, "n_train": int(len(indices)),
                    "seed": seed, "temperature": scorer.temperature,
                    "threshold": operating_threshold, "threshold_feasible": calibration['threshold_feasible'],
                    "calibration": calibration, "calibration_stage": args.calibration_stage,
                    "training_input_contract": input_contract,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

                runs.append({
                    "model": kind, "size_requested": size, "n_train": int(len(indices)),
                    "n_train_groups": len({train_rows[index]["group_id"] for index in indices}),
                    "seed": seed,
                    "temperature": scorer.temperature,
                    "operating_threshold": operating_threshold,
                    "threshold_feasible": calibration['threshold_feasible'],
                    "calibration": {
                        "n": len(calibration_rows),
                        "target_conditional_risk": EM.TARGET_DANGEROUS_ALLOW,
                        "min_coverage": EM.MIN_COVERAGE,
                        "selected_point": feasible_point,
                        "protocol": args.calibration_protocol,
                        "selection": calibration["selection"], "audit": calibration["audit"],
                    },
                    "dev": ({"n": len(dev_rows), "macro_f1": EM.macro_f1(
                        _labels(dev_rows), scorer.predict_proba_raw([row["request"] for row in dev_rows]).argmax(axis=1))}
                        if dev_rows else None),
                    "eval": eval_metrics,
                    "parity": parity,
                    "seconds": round(time.perf_counter() - started, 3),
                })
                summary = " ".join(f"{name}:F1={value['macro_f1']:.3f}"
                                   for name, value in eval_metrics.items())
                print(f"[{seed}] size={size} {kind}: threshold={operating_threshold} "
                      f"feasible={calibration['threshold_feasible']} T={scorer.temperature:.3f} {summary}",
                      flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": TRAINING_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": list(F.CLASSES),
        "safety": {
            "allowing_classes": list(EM.ALLOWING_CLASSES),
            "blocking_classes": list(EM.BLOCKING_CLASSES),
            "contextual_requires_trigger": EM.CONTEXTUAL_REQUIRES_TRIGGER,
            "target_dangerous_allow": EM.TARGET_DANGEROUS_ALLOW,
            "min_coverage": EM.MIN_COVERAGE,
            "calibration_used_for": ["temperature", "operating_threshold"],
            "eval_only_splits": ["challenge", "heldout", "ood"],
        },
        "data": {
            "data": str(data_path),
            "challenge": str(challenge_path) if challenge_path else None,
            "splits": {name: len(split_rows) for name, split_rows in by_split.items()},
            "pair_rows": {**{name: len(split_rows) for name, split_rows in pair_by_split.items()},
                          "challenge": len(challenge_rows)},
            "excluded_rows": {**excluded_by_split, "challenge": dict(challenge_excluded)},
            "exclusion_note": "set 形状与 existing>1 的整集行不进入配对训练/配对指标；"
                              "set 级评测在 adapter.evaluate_sets",
            "train_label_origins": dict(Counter(row.get("label_origin", "unknown")
                                                for row in train_rows)),
            "group_overlaps": overlaps,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": _sklearn_version(),
            "embedding": ({"fingerprint": embedder.fingerprint, "directory": str(embedder.model_dir),
                           "device": args.device, "threads": args.threads,
                           "batch_size": args.batch_size, **embedder.versions}
                          if embedder is not None else None),
        },
        "training_input_contract": input_contract,
        "args": {
            "models": models, "sizes": sorted({int(size) for size in sizes}), "seeds": seeds,
            "bootstrap_samples": args.bootstrap_samples, "cache_dir": args.cache_dir or None,
            "eval_splits": list(prepared_data["eval_splits"]),
            "calibration_protocol": args.calibration_protocol, "calibration_stage": args.calibration_stage,
        },
        "runs": runs,
        "notes": [
            "指标仅为该研究实验的离线结果，不代表生产路由能力，也不得与非同类口径混用。",
            "配对模型只接受 existing ≤1 的行；set 形状与多规则整集行已排除（见 excluded_rows），"
            "set 级评测属 adapter.evaluate_sets。",
            "partitioned candidate只见prob/policy且audit deferred、顶层阈值null；固定study唯一候选才独立一次audit；"
            "legacy点估计可行不是独立认证，heldout/OOD/challenge未参与拟合或选阈（τ=0仅未校准diagnostic）。",
            "组泄漏门无绕过开关：train/calibration/heldout/ood/challenge 之间任何 group_id 重叠都 fail-loud。",
            "predictions.jsonl 概率四舍五入到 6 位；verdict 按未舍入概率计算；verdict_tau0 是未校准诊断，不是运行点。",
            "零分母指标记 null；bootstrap 零分母复制记 invalid；零错误时读单侧稀有事件上界，不得把 percentile CI 的 [0,0] 当安全证据。",
        ],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "curve.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in curve_rows),
        encoding="utf-8")
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8")
    print(f"-> {out_dir}（{len(runs)} runs）", flush=True)
    return 0


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


if __name__ == "__main__":
    try:
        code = main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
