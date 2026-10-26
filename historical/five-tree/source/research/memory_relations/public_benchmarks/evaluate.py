                       
"""公开基准评测运行器（research-only）：冻结 protocol 的唯一评测消费入口。

对冻结合同（`protocol.py` + `study-v4.json`）与数据资产（`datasets.py` 产出的 run）：

- **评测池** = 数据 run 的官方非 train 分区（全成员、不删题；train 分区不评测，只计入总量
  记账）。**分区/用途**只用 `protocol.partition_records` 算出的权威 manifest（或调用方以
  `--manifest` 提供的已验证冻结 manifest；两者都必须逐字节覆盖评测池）；**选择**只经
  `manifest_scope`（原生视图）或 `select_for_phase`（七类诊断视图——study 未给其 scope_rule，
  该协议入口是唯一曝光/选择通道，再按视图 dataset 取行）；**计分**只走 `evaluate_view` /
  `external_contradiction_diagnostic`；**最终完成**只经 `assert_final_complete` 由
  manifest + 完整真实 results 重派生。本模块不写第二套得分/区间/完成/准入判据。
- **工件臂物理分开、类空间不得互相冒用**：
  * `semantic`：显式选择 C5/dual3 的真实 semantic 工件；只经
    `semantic_artifacts.load_inference_artifact` / `predict_nli_proba` 进入五个 NLI 任务视图，
    拒绝旧训练审计schema，不静默回退其巨meta。
    C5 按类名求父概率和、dual3 取 forward p3；输入/截断只消费 `prepare_inputs` 的
    same-visible 证据，保留真实工件类序/配方/研究 limits，不伪装 nli_reference。
  * `native`：原生 NLI 参考臂，先经 `protocol.native_view_binding` 绑定注册表（schema/kind/
    类序/标签空间/主向/槽位/配方/窗口）才可运行，加载只走 `nli_reference.load_nli_scorer`；
    输出 n×3、按 meta 实际类名定位列；HANS 按 study 冻结 reduction 归约
    （argmax==entailment → entailment，其余 → non-entailment）。只有「绑定 + 真实加载」
    成功才可计分，绝不冒充成绩，也绝不用七类替代。
  * `seven-class`：原七类工件只经 `features.load_scorer` + `predict_proba` 真实调用
    （权重/元数据 SHA256 前后保真），两方向都跑、不得平均，输出只交
    `external_contradiction_diagnostic`——永不产 NLI 三分类准确率；窗口按视图臂
    （w384/w512）设置并核验（production_contract 臂必须等于工件自身窗口，全文臂不得超过
    tokenizer 上限）。
- **运行治理**：`results.jsonl` 逐 chunk 落盘（flush + fsync）即断点；resume 先逐字节复核绑定
  （数据文件/study/manifest/selection/工件/视图/阶段/臂/设备），不兼容即拒；完成 = results 恰好
  覆盖选择的全部行且无重复、无失败；失败/未尝试逐行记账、保留在分母，`--limit` 只作有界探针
  （partial 结果无法计分，score 会如实拒绝）。
- **设备**：`finetune.resolve_device` 是唯一设备闸——请求 GPU 不可用/越界一律报错，绝不回退
  CPU；run.json 记录真实设备证明（matmul 设备、模型参数设备、CUDA 计数/名称）。
- **截断**：逐行 `truncated` 显式证据；旧臂消费 `finetune.truncation_profile` / `joint_text`，
  semantic臂消费真实 `prepare_inputs` 的完整性、同信息hash和字段审计，不混用序列器口径。
  角色覆盖由相应输入证据聚合；全文声明由 `protocol.claim_status` 按证据自动
  降级（缺证据或有截断只可声明 windowed_not_full，不以 384/512 冒全文）。

CLI（唯一入口，`--out` 只能写新目录或本模块自己创建的 run 目录）：

    python -m research.memory_relations.public_benchmarks.evaluate run \\
        --data <datasets.py run 目录> --view <view_id> --phase diagnostic|final_acceptance \\
        --scorer <工件目录> --out <自己的 run 目录> [--arm auto|native|seven-class|semantic] \\
        [--device cpu|cuda[:n]] [--chunk-rows N] [--limit N] [--manifest FILE] \\
        [--confirm-final-acceptance] [--no-score]
    python -m research.memory_relations.public_benchmarks.evaluate score --out <run 目录>
    python -m research.memory_relations.public_benchmarks.evaluate status --out <run 目录>

不做：下载/解析数据（datasets.py 职责）、训练、任何 poor/重训判定、封存验收的默认放行
（`phase=final_acceptance` 必须显式 `--confirm-final-acceptance`）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from research.memory_relations.public_benchmarks import protocol as P

__all__ = [
    "EvalError", "ARMS", "RUN_SCHEMA", "REPORT_SCHEMA", "SELECTION_SCHEMA", "PROGRESS_SCHEMA",
    "load_data_run", "build_manifest", "verify_manifest_covers_pool", "build_selection",
    "row_token_length", "row_truncated", "TruncationTracker", "view_request",
    "diagnostic_requests", "resolve_device", "device_proof", "load_seven_class_scorer",
    "project_native_artifact", "load_native_scorer", "load_semantic_scorer", "semantic_pair", "predict_requests",
    "run_view", "score_view", "status_view", "read_results", "main",
]

RUN_SCHEMA = "memory-relations-public-eval-run-v1"
REPORT_SCHEMA = "memory-relations-public-eval-report-v1"
SELECTION_SCHEMA = "memory-relations-public-eval-selection-v1"
PROGRESS_SCHEMA = "memory-relations-public-eval-progress-v1"
ARMS = ("seven-class", "native", "semantic")
NLI_ARMS = ("native", "semantic")
DEFAULT_CHUNK_ROWS = 32
_ERROR_MAX = 300


class EvalError(ValueError):
    """评测运行治理错误；`code` 是有界机器码，`detail` 只放有界审计信息。"""

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: object, code: str, detail: str | None = None) -> None:
    if not condition:
        raise EvalError(code, detail)


def _bounded(value: object, limit: int = _ERROR_MAX) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _format_error(exc: BaseException) -> str:
    return _bounded(f"{type(exc).__name__}: {exc}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise EvalError(code, path.name) from None


def _write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    """原子落盘：先写同目录临时文件再 replace，崩溃不产生半个可读 JSON。"""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        if compact:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)
        else:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_row(record: Mapping) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _study_sha256(study: Mapping) -> str:
    """冻结文件与实际消费的 study 必须一致。"""
    _require(study == P.load_study(), "study_drift")
    return P.study_sha256()


                                                                             
                         
                                                                             

def _read_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                raise EvalError("invalid_normalized_jsonl", f"{path.name}:{number}") from None
    return rows


def load_data_run(data_dir: object, *, study: Mapping | None = None,
                  verify_hashes: bool = True) -> dict:
    """载入 datasets.py 产出的官方非 train 评测池（全成员，不删题）。

    逐分区核对 manifest 声明的 normalized SHA256 与行数；逐行 `validate_record_for_dataset`。
    返回的行对象即 manifest 分区的输入；文件/计数/哈希进入运行绑定。
    """
    study = P.load_study() if study is None else study
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    report = _read_json(manifest_path, "data_manifest_unreadable")
    _require(isinstance(report, dict) and report.get("schema_version") == 1,
             "invalid_data_manifest", "schema_version")
    datasets = report.get("datasets")
    _require(isinstance(datasets, dict) and datasets, "invalid_data_manifest", "datasets")
    rows: list = []
    files: list = []
    seen: set = set()
    for name in sorted(datasets):
        spec = datasets[name]
        _require(isinstance(spec, dict) and isinstance(spec.get("splits"), dict),
                 "invalid_data_manifest", name)
        for split_name in sorted(spec["splits"]):
            entry = spec["splits"][split_name]
            _require(isinstance(entry, dict), "invalid_data_manifest", split_name)
            official = entry.get("official_split")
            _require(isinstance(official, str) and official, "invalid_data_manifest", split_name)
            if "train" in official.lower():
                continue                                   
            normalized = entry.get("normalized")
            _require(isinstance(normalized, dict), "invalid_data_manifest", split_name)
            relative, declared_sha, declared_rows = (normalized.get("path"),
                                                     normalized.get("sha256"),
                                                     normalized.get("records"))
            _require(isinstance(relative, str) and relative, "invalid_data_manifest", split_name)
            _require(isinstance(declared_sha, str) and len(declared_sha) == 64,
                     "invalid_data_manifest", split_name)
            _require(isinstance(declared_rows, int) and declared_rows >= 0,
                     "invalid_data_manifest", split_name)
            path = data_dir / relative
            _require(path.is_file(), "missing_normalized_file", relative)
            if verify_hashes:
                _require(_sha256_file(path) == declared_sha, "normalized_sha_mismatch", relative)
            file_rows = _read_jsonl(path)
            _require(len(file_rows) == declared_rows, "normalized_count_mismatch", relative)
            counts = entry.get("counts")
            if isinstance(counts, dict) and isinstance(counts.get("rows"), int):
                _require(counts["rows"] == declared_rows, "normalized_count_mismatch", relative)
            for record in file_rows:
                P.validate_record_for_dataset(record, study)
                _require(isinstance(record, dict) and record.get("dataset") == name,
                         "record_dataset_mismatch", relative)
                _require(record["id"] not in seen, "duplicate_record_id", str(record["id"]))
                seen.add(record["id"])
                rows.append(record)
            files.append({"dataset": name, "official_split": official, "path": relative,
                          "sha256": declared_sha, "records": declared_rows})
    _require(rows, "empty_pool")
    labeled = sum(1 for row in rows if row["gold_label"] is not None)
    missing = sum(1 for row in rows if row["gold_label"] is None)
    no_consensus = sum(1 for row in rows if row["gold_label"] is None
                       and row["metadata"].get("gold_status") == "no_consensus")
    by_dataset: dict = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
    totals = report.get("totals") if isinstance(report.get("totals"), dict) else {}
    producer_rows = totals.get("records")
    return {
        "run_dir": str(data_dir.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "files": files,
        "rows": rows,
        "accounting": {
            "rows": len(rows), "labeled": labeled, "missing_gold": missing,
            "no_consensus": no_consensus, "by_dataset": by_dataset,
            "policy": "官方非 train 分区全成员；train 不评测、只计入总量记账；不删题",
        },
        "producer_totals": dict(totals),
        "train_excluded_rows": (producer_rows - len(rows))
        if isinstance(producer_rows, int) else None,
    }


                                                                             
                              
                                                                             

def verify_manifest_covers_pool(manifest: Mapping, rows: Sequence) -> None:
    """冻结 manifest 必须与评测池逐字节一致：行集合相同 + 逐行 JSON 逐字节相同。"""
    manifest_rows = list(manifest["diagnostic"]) + list(manifest["sealed_acceptance"])
    pool_ids = sorted(row["id"] for row in rows)
    manifest_ids = sorted(row["id"] for row in manifest_rows)
    _require(manifest_ids == pool_ids, "manifest_pool_mismatch", "行集合与评测池不一致")
    by_id = {row["id"]: row for row in manifest_rows}
    for record in rows:
        if _canonical_row(by_id[record["id"]]) != _canonical_row(record):
            raise EvalError("manifest_pool_mismatch", f"行字节不一致：{record['id']}")


def build_manifest(rows: Sequence, *, study: Mapping | None = None,
                   frozen_manifest_path: object | None = None) -> dict:
    """权威 manifest：默认由 `protocol.partition_records` 一次算出；给了冻结 manifest 则核验后消费，
    绝不在子集上重算用途。"""
    study = P.load_study() if study is None else study
    if frozen_manifest_path is not None:
        manifest = _read_json(Path(frozen_manifest_path), "manifest_unreadable")
        validated = P.validate_manifest(manifest, study)
        verify_manifest_covers_pool(validated, list(rows))
        return validated
    return P.partition_records(list(rows), study)


def build_selection(manifest: Mapping, *, study: Mapping | None = None, view_id: str,
                    phase: str) -> dict:
    """视图×阶段预定全行集合：原生视图走 `manifest_scope`；诊断视图走 `select_for_phase`。"""
    study = P.load_study() if study is None else study
    _require(phase in P.PHASES, "invalid_phase", str(phase))
    view = P.view_spec(study, view_id)
    kind = view["kind"]
    if kind == "native_nli":
        scope = P.manifest_scope(manifest, view_id=view_id, phase=phase, study=study)
        selection = {
            "schema": SELECTION_SCHEMA, "view_id": view_id, "phase": phase, "kind": kind,
            "dataset": view["dataset"], "binding": "manifest_scope",
            "ids": list(scope["ids"]), "record_ids_sha256": scope["record_ids_sha256"],
            "accounting": dict(scope["accounting"]),
        }
    elif kind == "seven_class_external_diagnostic":
        rows = P.select_for_phase(manifest, phase)
        ids = sorted(row["id"] for row in rows if row["dataset"] == view["dataset"])
        selection = {
            "schema": SELECTION_SCHEMA, "view_id": view_id, "phase": phase, "kind": kind,
            "dataset": view["dataset"], "binding": "select_for_phase+dataset",
            "ids": ids, "record_ids_sha256": _ids_sha256(ids),
            "accounting": {"rule": "all_rows", "n_official": len(ids), "n_eligible": len(ids),
                           "note": "study 未给诊断视图 scope_rule；曝光/选择只经 "
                                   "select_for_phase（先 validate_manifest 再返回）"},
        }
    else:
        raise EvalError("unknown_view_kind", kind)
    return selection


                                                                             
           
                                                                             

def view_request(record: Mapping, view: Mapping) -> dict:
    """视图行 → 共同 memory-shaped request（唯一适配 `protocol.diagnostic_request`）。

    oracle_evidence 层按冻结规则改 premise 为 `protocol.oracle_premise` 的 gold 证据跨度
    （受控 oracle 取用；缺失跨度 fail loud，绝不用空 premise 泄漏标签）。
    """
    if view.get("layer") == "oracle_evidence":
        record = dict(record)
        record["premise"] = P.oracle_premise(record)
    return P.diagnostic_request(record, direction=P.PRIMARY_DIRECTION)


def diagnostic_requests(record: Mapping) -> dict:
    """七类探针两方向请求（主向预注册；两向都跑、不得平均）。"""
    return {direction: P.diagnostic_request(record, direction=direction)
            for direction in P.DIRECTIONS}


def row_token_length(tokenizer, request: Mapping) -> int:
    """整行真实 token 长度（含特殊 token）：与 `finetune.truncation_profile` 的行级判定
    同一口径（`finetune.joint_text` 序列化 + tokenizer 真实计数），本函数是该判定的逐行投影。"""
    from research.memory_relations import finetune as FT

    text = FT.joint_text(dict(request))
    return int(len(tokenizer(text, add_special_tokens=True)["input_ids"]))


def row_truncated(tokenizer, request: Mapping, max_length: int) -> bool:
    return row_token_length(tokenizer, request) > int(max_length)


class TruncationTracker:
    """聚合各臂真实输入证据：旧臂用 finetune profile，semantic用prepared字段完整性。"""

    def __init__(self, *, max_length: int) -> None:
        self.max_length = int(max_length)
        self.rows = 0
        self.truncated_rows = 0
        self.unattributed_tokens = 0
        self.budget_tokens = None
        self._attributions: set = set()
        self._by_source: dict = {}
        self._lengths: list = []

    def add(self, tokenizer, requests: Sequence[Mapping], lengths: Sequence[int], *,
            semantic: bool = False) -> None:
        from research.memory_relations import finetune as FT

        requests = list(requests)
        if not requests:
            return
        if semantic:
            by_source = {}
            for item in requests:
                for role in ("old", "new", "frame"):
                    fields = [field for field in item["directions"]["forward"]["fields"]
                              if field["path"] and field["path"][0] == role]
                    bucket = by_source.setdefault(role, {"cut": 0, "dropped": 0})
                    bucket["cut"] += int(any(field["lost"] for field in fields))
                    bucket["dropped"] += int(any(field["omitted"] for field in fields))
            profile = {"rows": len(requests), "truncated_rows": sum(not item["complete"] for item in requests),
                       "unattributed_tokens": 0, "budget_tokens": self.max_length,
                       "attribution": "semantic_same_visible", "by_source": by_source}
        else:
            profile = FT.truncation_profile(tokenizer, requests, self.max_length)
        self.rows += int(profile["rows"])
        self.truncated_rows += int(profile["truncated_rows"])
        self.unattributed_tokens += int(profile["unattributed_tokens"])
        self.budget_tokens = profile["budget_tokens"]
        self._attributions.add(str(profile["attribution"]))
        for role, record in profile["by_source"].items():
            bucket = self._by_source.setdefault(role, {"cut": 0, "dropped": 0})
            bucket["cut"] += int(record["cut"])
            bucket["dropped"] += int(record["dropped"])
        self._lengths.extend(int(value) for value in lengths)

    def summary(self) -> dict:
        lengths = np.asarray(self._lengths, dtype=float)
        return {
            "rows": self.rows,
            "truncated_rows": self.truncated_rows,
            "truncation_rate": (self.truncated_rows / self.rows) if self.rows else None,
            "max_length": self.max_length,
            "budget_tokens": self.budget_tokens,
            "attribution": ("unavailable" if not self._attributions
                            else "mixed" if len(self._attributions) > 1
                            else next(iter(self._attributions))),
            "counting_unit": "row（同一 role 的多个 block 每行至多计一次）",
            "unattributed_tokens": self.unattributed_tokens,
            "token_length": ({
                "mean": float(lengths.mean()), "p50": float(np.percentile(lengths, 50)),
                "p90": float(np.percentile(lengths, 90)), "p99": float(np.percentile(lengths, 99)),
                "max": float(lengths.max()),
            } if self.rows else None),
            "by_source": {role: dict(record) for role, record in sorted(self._by_source.items())},
            "note": "计数聚合本臂真实输入接缝证据；逐行truncated及semantic输入审计写入results.jsonl",
        }


                                                                             
              
                                                                             

def resolve_device(device: object) -> str:
    """唯一设备闸：`finetune.resolve_device`（GPU 不可用/越界报错，绝不回退 CPU）。"""
    from research.memory_relations import finetune as FT

    try:
        return str(FT.resolve_device(str(device)))
    except RuntimeError as exc:
        raise EvalError("device_unavailable", _bounded(exc)) from None
    except ValueError as exc:
        raise EvalError("invalid_device", _bounded(exc)) from None


def device_proof(device: str) -> dict:
    """真实设备证明：解析后的设备上做一次真实 matmul，并记录 CUDA 事实。"""
    import torch

    resolved = resolve_device(device)
    left = torch.ones(2, 2, device=resolved)
    product = left @ left
    proof = {
        "requested": str(device), "resolved": resolved, "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "matmul_device": str(product.device), "matmul_value": float(product.sum()),
    }
    if torch.cuda.is_available():
        try:
            proof["cuda_device_name"] = str(torch.cuda.get_device_name(resolved))
        except (RuntimeError, ValueError, AssertionError):
            proof["cuda_device_name"] = None
    return proof


def _artifact_files(model_dir: Path, weights_file: str, meta_file: str) -> dict:
    weights, meta = model_dir / weights_file, model_dir / meta_file
    _require(weights.is_file(), "artifact_missing", weights_file)
    _require(meta.is_file(), "artifact_missing", meta_file)
    return {"weights_file": weights_file, "meta_file": meta_file,
            "weights_sha256": _sha256_file(weights), "meta_sha256": _sha256_file(meta)}


def _window_of(view: Mapping) -> tuple:
    window = view.get("window")
    if not isinstance(window, dict):
        raise EvalError("invalid_view_window", view.get("view_id", "?"))
    kind = window.get("kind")
    _require(kind in P.WINDOW_KINDS, "invalid_view_window", str(kind))
    max_tokens = window.get("max_tokens")
    return kind, (None if max_tokens is None else int(max_tokens))


def _parameter_device(scorer) -> str:
    """实际模型参数所在设备：设备证明的硬要求（mock/异形 scorer 缺 model/tokenizer 即拒）。"""
    model = getattr(scorer, "model", None)
    _require(callable(getattr(model, "parameters", None))
             and getattr(scorer, "tokenizer", None) is not None,
             "scorer_contract_mismatch", "scorer 必须暴露 model.parameters() 与 tokenizer")
    parameters = iter(model.parameters())
    try:
        first = next(parameters)
    except StopIteration:
        raise EvalError("scorer_contract_mismatch", "model 没有参数") from None
    return str(first.device)


def load_seven_class_scorer(model_dir: object, *, view: Mapping, device: str) -> dict:
    """七类臂加载：只经 `features.load_scorer` 真实加载；窗口按视图臂设置并核验。

    production_contract 臂必须等于工件自身窗口（生产一致配方）；backbone_fulltext 臂允许更大
    窗口，但不得超过 tokenizer 上限，也不得小于工件窗口（诊断臂只放宽、不收紧）。
    绝不把 n×7 输出当 NLI 三分类，也绝不产出三分类准确率。
    """
    from research.memory_relations import features as FEAT
    from research.memory_relations import finetune as FT

    directory = Path(model_dir)
    meta = _read_json(directory / FT.META_FILE, "artifact_missing")
    _require(isinstance(meta, dict) and meta.get("schema") == FT.ARTIFACT_SCHEMA,
             "invalid_artifact", "schema")
    files = _artifact_files(directory, FT.WEIGHTS_FILE, FT.META_FILE)
    artifact_max_length = meta.get("max_length")
    _require(isinstance(artifact_max_length, int) and artifact_max_length >= 1,
             "invalid_artifact", "max_length")
    window_kind, window = _window_of(view)
    _require(window is not None, "invalid_view_window", "max_tokens")
    if window_kind == "production_contract":
        _require(window == artifact_max_length, "window_conflict",
                 f"视图窗口 {window} != 工件 {artifact_max_length}")
    else:
        _require(window >= artifact_max_length, "window_conflict",
                 f"全文臂 {window} < 工件 {artifact_max_length}（诊断臂只放宽、不收紧）")
    resolved = resolve_device(device)
    scorer = FEAT.load_scorer(directory, device=resolved)
    _require(tuple(scorer.classes) == FEAT.CLASSES, "scorer_classes_mismatch")
    if window_kind == "backbone_fulltext":
        tokenizer_max = getattr(scorer.tokenizer, "model_max_length", None)
        _require(not isinstance(tokenizer_max, int) or window <= tokenizer_max,
                 "window_exceeds_backbone", f"{window} > {tokenizer_max}")
    scorer.max_length = int(window)
    return {
        "scorer": scorer,
        "identity": {
            "arm": "seven-class", "dir": str(directory.resolve()), "schema": meta["schema"],
            "kind": meta.get("kind"), "classes": list(scorer.classes),
            "input_recipe": meta.get("input_recipe"), "temperature": meta.get("temperature"),
            "artifact_max_length": artifact_max_length, "window": int(window),
            "window_kind": window_kind, "device": resolved,
            "weights_file": files["weights_file"], "meta_file": files["meta_file"],
            "weights_sha256": files["weights_sha256"], "meta_sha256": files["meta_sha256"],
            "parameter_device": _parameter_device(scorer),
        },
    }


def project_native_artifact(model_dir: object) -> dict:
    """nli_reference 工件 meta → `native_view_binding` 要求的注册投影（逐字段显式，不猜）。"""
    from research.memory_relations import nli_reference as NLI

    directory = Path(model_dir)
    meta = _read_json(directory / NLI.NLI_META_FILE, "artifact_missing")
    _require(isinstance(meta, dict) and meta.get("schema") == NLI.NLI_SCHEMA,
             "invalid_artifact", "schema")
    fields = ("schema", "kind", "classes", "label_space", "input_recipe", "direction",
              "joint_roles", "max_length", "weights_sha256", "backbone_fingerprint")
    _require(all(key in meta for key in fields), "invalid_artifact", "meta 字段缺失")
    return {
        "artifact_id": directory.name,
        "schema": meta["schema"], "kind": meta["kind"], "classes": meta["classes"],
        "label_space": meta["label_space"], "input_recipe": meta["input_recipe"],
        "direction": meta["direction"], "joint_roles": meta["joint_roles"],
        "max_length": meta["max_length"], "weights_sha256": meta["weights_sha256"],
        "backbone_fingerprint": meta["backbone_fingerprint"],
    }


def load_native_scorer(model_dir: object, *, view: Mapping, device: str,
                       scorer_loader: Callable | None = None,
                       study: Mapping | None = None) -> dict:
    """原生臂加载：先经 `protocol.native_view_binding` 注册比对（含 weights SHA256），
    再经 `nli_reference.load_nli_scorer` 真实加载；类空间按 meta 实际类名定位。"""
    from research.memory_relations import nli_reference as NLI

    study = P.load_study() if study is None else study
    directory = Path(model_dir)
    artifact = project_native_artifact(directory)
    files = _artifact_files(directory, NLI.NLI_WEIGHTS_FILE, NLI.NLI_META_FILE)
    _require(files["weights_sha256"] == artifact["weights_sha256"],
             "artifact_hash_mismatch", "weights 哈希与 meta 不一致")
    binding = P.native_view_binding(view, artifact, weights_path=directory / NLI.NLI_WEIGHTS_FILE,
                                    study=study)
    resolved = resolve_device(device)
    loader = NLI.load_nli_scorer if scorer_loader is None else scorer_loader
    scorer = loader(directory, resolved)
    _require(tuple(getattr(scorer, "classes", ()) or ()) == tuple(artifact["classes"]),
             "scorer_classes_mismatch")
    _require(getattr(scorer, "direction", None) == artifact["direction"],
             "scorer_direction_mismatch")
    meta = _read_json(directory / NLI.NLI_META_FILE, "artifact_missing")
    _require(int(getattr(scorer, "max_length", -1)) == int(artifact["max_length"]),
             "scorer_window_mismatch", "meta.max_length 与加载后 scorer.max_length 不一致")
    return {
        "scorer": scorer,
        "binding": binding,
        "identity": {
            "arm": "native", "dir": str(directory.resolve()), "schema": artifact["schema"],
            "kind": artifact["kind"], "classes": list(artifact["classes"]),
            "input_recipe": artifact["input_recipe"], "direction": artifact["direction"],
            "joint_roles": artifact["joint_roles"], "artifact_max_length":
                int(artifact["max_length"]), "window": int(scorer.max_length),
            "window_kind": (view.get("window") or {}).get("kind"), "device": resolved,
            "weights_file": files["weights_file"], "meta_file": files["meta_file"],
            "weights_sha256": files["weights_sha256"], "meta_sha256": files["meta_sha256"],
            "parameter_device": _parameter_device(scorer),
            "metadata_check": binding["metadata_check"],
            "artifact_integrity": binding["artifact_integrity"],
            "note": "meta.max_length 应为加载后 scorer.max_length；不一致即拒",
        },
    }


def _semantic_identity(directory: Path, view: Mapping) -> dict:
    from research.memory_relations import semantic_model as SM
    from research.memory_relations import semantic_artifacts as A

    _require(view["kind"] == "native_nli", "arm_view_mismatch")
    try:
        meta = A.read_metadata(directory)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise EvalError("semantic_artifact_invalid", _format_error(exc)) from None
    subtype = meta["bindings"]["arm"]
    fields = ("schema", "kind", "classes", "label_space", "input_recipe", "input_limits", "direction", "joint_roles", "bindings")
    files = {"weights_file": SM.WEIGHTS_FILE, "meta_file": SM.META_FILE,
             "weights_sha256": meta["weights_sha256"], "meta_sha256": _sha256_file(directory / SM.META_FILE)}
    return {"arm": "semantic", "dir": str(directory.resolve()), **{key: meta[key] for key in fields}, **files,
            "certificate_sha256": meta["train_sha256"], "artifact_policy_sha256": SM.FT.file_sha256(A.__file__),
            "artifact_arm": subtype, "artifact_max_length": meta["max_length"],
            "window": meta["max_length"], "window_kind": (view.get("window") or {}).get("kind"),
            "output_classes": list(SM.SR.NLI_CLASSES),
            "projection": "parent_probability_sum" if subtype == "c5" else "forward_nli3",
            "frame_policy": "empty_background", "metadata_check": "matched_semantic_inference_export"}


def load_semantic_scorer(model_dir: object, *, view: Mapping, device: str) -> dict:
    from research.memory_relations import semantic_artifacts as A

    directory = Path(model_dir)
    identity = _semantic_identity(directory, view)
    resolved = resolve_device(device)
    try:
        scorer, _, _ = A.load_inference_artifact(directory, device=resolved)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise EvalError("semantic_artifact_invalid", _format_error(exc)) from None
    _require(list(scorer.classes) == identity["classes"] and scorer.arm == identity["artifact_arm"]
             and scorer.direction == identity["direction"], "scorer_binding_mismatch")
    _require(scorer.max_length == identity["window"], "scorer_window_mismatch")
    _require(scorer.max_length <= scorer.model.backbone.config.max_position_embeddings,
             "window_exceeds_backbone")
    identity.update(device=resolved, parameter_device=_parameter_device(scorer))
    return {"scorer": scorer, "identity": identity}


def semantic_pair(record: Mapping, view: Mapping) -> dict:
    premise = P.oracle_premise(record) if view.get("layer") == "oracle_evidence" else record["premise"]
    return {"old": {"kind": "proposition", "text": premise},
            "new": {"kind": "proposition", "text": record["hypothesis"]},
            "frame": {"background": ""}}


def _check_semantic_results(entries: Sequence[Mapping], bound: Mapping) -> None:
    from research.memory_relations import semantic_model as SM

    by_id = {row["id"]: row for row in bound["manifest"]["diagnostic"] + bound["manifest"]["sealed_acceptance"]}
    identity, view = bound["run"]["scorer"], bound["view"]
    selected = set(bound["selection"]["ids"])
    for entry in entries:
        _require(entry["record_id"] in selected, "results_out_of_selection")
        audit = entry["input"]
        if audit is not None:
            pair = semantic_pair(by_id[entry["record_id"]], view)
            raw = SM.SR.semantic_pair_pieces(pair, limits=identity["input_limits"])
            _require(audit.get("raw_pair_sha256") == raw["raw_pair_sha256"]
                     and audit.get("recipe") == identity["input_recipe"]
                     and audit.get("input_limits") == identity["input_limits"]
                     and audit.get("max_length") == identity["window"], "input_drift")
        if entry["p3"] is not None:
            pred = _native_prediction(np.asarray(entry["p3"]), None, view, output_classes=SM.SR.NLI_CLASSES)
            _require(entry["pred"] == pred, "corrupt_results", "p3/view prediction")


def _semantic_audit(prepared: Mapping) -> dict:
    return {key: prepared[key] for key in ("recipe", "input_limits", "raw_pair_sha256",
            "visible_information_sha256", "max_length", "complete", "full_visible_token_lengths")} | {
        "directions": {name: {key: item[key] for key in ("view_sha256", "visible_information_sha256",
                       "token_length", "role_body_covered", "fields")}
                       for name, item in prepared["directions"].items()}}


def predict_requests(scorer, requests: Sequence[Mapping], *,
                     predict: Callable | None = None, output_classes: Sequence[str] | None = None) -> list:
    """调用本臂真实概率方法；整块技术失败时逐行归因，不丢弃失败行。

    返回等长 [(probs|None, error|None)]；形状按显式输出类序或scorer原生类序校验，
    概率必须 finite 且在 simplex 容差内（非法向量 fail loud，不静默归一化）。
    """
    items = [dict(request) for request in requests]
    if not items:
        return []
    expected = len(scorer.classes if output_classes is None else output_classes)
    method = scorer.predict_proba if predict is None else predict

    def checked(probs, count: int):
        array = np.asarray(probs, dtype=np.float64)
        if array.shape != (count, expected):
            raise EvalError("scorer_output_shape", f"{array.shape} != {(count, expected)}")
        if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any() \
                or bool((np.abs(array.sum(axis=1) - 1.0) > P.SIMPLEX_TOLERANCE).any()):
            raise EvalError("invalid_probs", "非 simplex 概率输出")
        return array

    try:
        array = checked(method(items), len(items))
        return [(array[index], None) for index in range(len(items))]
    except EvalError:
        raise
    except Exception:
        outcomes = []
        for request in items:
            try:
                array = checked(method(request), 1)
                outcomes.append((array[0], None))
            except Exception as exc:                             
                outcomes.append((None, _format_error(exc)))
        return outcomes


                                                                             
                        
                                                                             

def _results_line(entry: Mapping) -> str:
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_entry(entry: object, *, arm: str) -> None:
    _require(isinstance(entry, dict) and isinstance(entry.get("record_id"), str),
             "corrupt_results", "记录形状")
    if arm in NLI_ARMS:
        fields = {"record_id", "pred", "error", "truncated"}
        _require(set(entry) == (fields | {"p3", "input"} if arm == "semantic" else fields),
                 "corrupt_results", "字段集合")
        _require(entry["pred"] is None or isinstance(entry["pred"], str),
                 "corrupt_results", "pred")
        _require(entry["error"] is None or (isinstance(entry["error"], str) and entry["error"]),
                 "corrupt_results", "error")
        _require(isinstance(entry["truncated"], bool), "corrupt_results", "truncated")
        _require(entry["pred"] is None or entry["error"] is None, "corrupt_results", "pred/error")
        if arm == "semantic":
            from research.memory_relations import semantic_model as SM

            if entry["p3"] is not None:
                try:
                    SM.SR.validate_probabilities(entry["p3"], size=3)
                except (ValueError, TypeError) as exc:
                    raise EvalError("corrupt_results", _format_error(exc)) from None
            _require((entry["p3"] is None) == (entry["pred"] is None), "corrupt_results", "p3/pred")
            audit = entry["input"]
            _require(audit is None or isinstance(audit, dict), "corrupt_results", "input")
            if audit is None:
                _require(entry["error"] is not None and entry["truncated"], "corrupt_results", "input failure")
            else:
                _require(type(audit.get("complete")) is bool and entry["truncated"] == (not audit["complete"]),
                         "corrupt_results", "input completeness")
                directions = audit.get("directions", {})
                _require(set(directions) == {"forward", "reverse"}
                         and all(item.get("visible_information_sha256") == audit.get("visible_information_sha256")
                                 for item in directions.values()), "corrupt_results", "same visible input")
    else:
        _require(set(entry) == {"record_id", "probs", "truncated", "error"},
                 "corrupt_results", "字段集合")
        probs, truncated = entry["probs"], entry["truncated"]
        _require(isinstance(probs, dict) and set(probs) == set(P.DIRECTIONS),
                 "corrupt_results", "probs 方向")
        _require(isinstance(truncated, dict) and set(truncated) == set(P.DIRECTIONS)
                 and all(isinstance(value, bool) for value in truncated.values()),
                 "corrupt_results", "truncated 方向")
        for direction in P.DIRECTIONS:
            value = probs[direction]
            if value is not None:
                _require(isinstance(value, list) and len(value) == 7
                         and all(isinstance(item, (int, float)) for item in value),
                         "corrupt_results", "probs 形状")
        _require(entry["error"] is None or (isinstance(entry["error"], str) and entry["error"]),
                 "corrupt_results", "error")
        failed = any(probs[direction] is None for direction in P.DIRECTIONS)
        _require(failed == (entry["error"] is not None), "corrupt_results", "probs/error 一致")


def read_results(path: Path, *, arm: str, repair_tail: bool = False) -> dict:
    """读取 results.jsonl：重复行/坏行 fail loud；崩溃留下的尾部残片只在 resume 时截除。"""
    path = Path(path)
    if not path.is_file():
        return {"entries": [], "repaired_bytes": 0, "sha256": None}
    raw = path.read_bytes()
    repaired = 0
    if raw and not raw.endswith(b"\n"):
        if not repair_tail:
            raise EvalError("corrupt_results", "尾部残片（无换行）")
        cut = raw.rfind(b"\n") + 1
        repaired = len(raw) - cut
        with open(path, "wb") as handle:
            handle.write(raw[:cut])
        raw = raw[:cut]
    entries = []
    seen: set = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EvalError("corrupt_results", f"line {number}") from None
        _validate_entry(entry, arm=arm)
        _require(entry["record_id"] not in seen, "duplicate_results", entry["record_id"])
        seen.add(entry["record_id"])
        entries.append(entry)
    return {"entries": entries, "repaired_bytes": repaired, "sha256": _sha256_file(path)}


def _append_entries(path: Path, entries: Sequence[Mapping]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(_results_line(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _entry_state(entry: Mapping, *, arm: str) -> str:
    if arm in NLI_ARMS:
        if entry["error"] is not None:
            return "failed"
        return "scored" if entry["pred"] is not None else "not_attempted"
    if entry["error"] is not None:
        return "failed"
    return "scored" if all(entry["probs"][direction] is not None for direction in P.DIRECTIONS) \
        else "not_attempted"


def _counts(entries: Sequence[Mapping], *, arm: str) -> dict:
    states = [_entry_state(entry, arm=arm) for entry in entries]
    return {
        "n_done": len(entries),
        "n_scored": states.count("scored"),
        "n_failed": states.count("failed"),
        "n_not_attempted": states.count("not_attempted"),
    }


def _progress_payload(*, run_record: Mapping, selection: Mapping, planned: Sequence[str],
                      entries: Sequence[Mapping], batches: int, started_at: str,
                      infer_complete: bool = False, artifact: dict | None = None) -> dict:
    counts = _counts(entries, arm=run_record["arm"])
    unrecorded = len(selection["ids"]) - counts["n_done"]
    return {
        "schema": PROGRESS_SCHEMA, "view_id": run_record["view_id"],
        "phase": run_record["phase"], "arm": run_record["arm"],
        "dataset": run_record["dataset"], "started_at": started_at, "updated_at": _utc_now_iso(),
        "selection_total": len(selection["ids"]), "n_planned": len(planned),
        "n_done": counts["n_done"], "n_scored": counts["n_scored"],
        "n_failed": counts["n_failed"], "n_not_attempted": counts["n_not_attempted"],
        "n_unrecorded": unrecorded,
        "batches": batches, "infer_complete": bool(infer_complete and not unrecorded
                                                    and not counts["n_not_attempted"]),
        "completion": {"unrecorded": unrecorded, "failed": counts["n_failed"],
                       "not_attempted": counts["n_not_attempted"],
                       "complete": not unrecorded and not counts["n_failed"]
                       and not counts["n_not_attempted"]},
        "artifact": artifact,
        "note": "失败/未尝试/未落盘保留在预定集；infer_complete 表示全选择已尝试，"
                "completion.complete 另要求无失败；均不构成最终验收证明",
    }


                                                                             
      
                                                                             

def _native_prediction(probs: np.ndarray, scorer, view: Mapping, *,
                       output_classes: Sequence[str] | None = None) -> str:
    classes = list(scorer.classes if output_classes is None else output_classes)
    pred = classes[int(np.asarray(probs).argmax())]
    reductions = view.get("reductions") or {}
    allowed = P.view_label_space(view)
    if view.get("task") == "entailment2" and "nli3" in reductions:
        pred = "entailment" if pred == "entailment" else "non-entailment"
    elif view.get("subset_of") == "nli3" and "nli3" in reductions:
        indexes = [index for index, name in enumerate(classes) if name in allowed]
        _require(len(indexes) == len(allowed), "scorer_classes_mismatch")
        pred = classes[indexes[int(np.asarray(probs)[indexes].argmax())]]
    _require(pred in allowed, "cross_task_prediction", pred)
    return pred


def _infer_entries(*, arm: str, view: Mapping, scorer, rows: Sequence[Mapping],
                   tokenizer, max_length: int) -> tuple:
    """一个 chunk 的真实推理 → (entries, requests_all, lengths_all)；chunk 级异常逐行归因。

    七类臂的单次 `predict_proba` 吃整个 chunk 的 2×n 请求（主向/交换向交替），不做逐行调用；
    截断长度与 `requests_all` 同源，供 tracker 复算角色覆盖。
    """
    if arm == "semantic":
        from research.memory_relations import semantic_model as SM

        pairs = [semantic_pair(row, view) for row in rows]
        prepared, errors = [], []
        for pair in pairs:
            try:
                prepared.append(_semantic_audit(scorer.prepare_inputs([pair])[0]))
                errors.append(None)
            except Exception as exc:
                prepared.append(None)
                errors.append(_format_error(exc))
        valid = [index for index, item in enumerate(prepared) if item is not None]
        outcomes = predict_requests(scorer, [pairs[index] for index in valid],
                                    predict=scorer.predict_nli_proba, output_classes=SM.SR.NLI_CLASSES)
        by_index = dict(zip(valid, outcomes))
        entries = []
        for index, row in enumerate(rows):
            audit = prepared[index]
            probs, error = by_index.get(index, (None, errors[index]))
            entries.append({"record_id": row["id"], "pred": None if probs is None else
                            _native_prediction(probs, scorer, view, output_classes=SM.SR.NLI_CLASSES),
                            "error": error, "truncated": audit is None or not audit["complete"],
                            "p3": None if probs is None else [float(value) for value in probs],
                            "input": audit})
        known = [item for item in prepared if item is not None]
        return entries, known, [item["full_visible_token_lengths"][0] for item in known]
    if arm == "native":
        requests = [view_request(row, view) for row in rows]
        lengths = [row_token_length(tokenizer, request) for request in requests]
        outcomes = predict_requests(scorer, requests)
        entries = []
        for row, request, length, (probs, error) in zip(rows, requests, lengths, outcomes):
            truncated = bool(length > max_length)
            if error is not None or probs is None:
                entries.append({"record_id": row["id"], "pred": None,
                                "error": error or "no_probs", "truncated": truncated})
            else:
                entries.append({"record_id": row["id"],
                                "pred": _native_prediction(probs, scorer, view),
                                "error": None, "truncated": truncated})
        return entries, requests, lengths
    requests_by_direction, lengths_by_direction = {}, {}
    for direction in P.DIRECTIONS:
        requests_by_direction[direction] = [diagnostic_requests(row)[direction] for row in rows]
        lengths_by_direction[direction] = [row_token_length(tokenizer, request)
                                           for request in requests_by_direction[direction]]
    requests_all = [requests_by_direction[direction][index]
                    for index in range(len(rows)) for direction in P.DIRECTIONS]
    outcomes = predict_requests(scorer, requests_all)
    entries = []
    for index, row in enumerate(rows):
        probs, errors = {}, []
        for offset, direction in enumerate(P.DIRECTIONS):
            probs_value, error = outcomes[index * len(P.DIRECTIONS) + offset]
            probs[direction] = None if probs_value is None else [float(value)
                                                                 for value in probs_value]
            if error is not None:
                errors.append(f"{direction}: {error}")
        entries.append({
            "record_id": row["id"], "probs": probs,
            "truncated": {direction: bool(lengths_by_direction[direction][index] > max_length)
                          for direction in P.DIRECTIONS},
            "error": "; ".join(errors) if errors else None,
        })
    return entries, requests_all, [value for direction in P.DIRECTIONS
                                   for value in lengths_by_direction[direction]]


def _binding_check(recorded: Mapping, pool: Mapping, *, study: Mapping, view_id: str,
                   phase: str, arm: str, scorer_dir: Path, out: Path,
                   device: str | None) -> None:
    def fail(detail: str):
        raise EvalError("incompatible_resume", detail)

    if recorded.get("schema") != RUN_SCHEMA or recorded.get("view_id") != view_id \
            or recorded.get("phase") != phase or recorded.get("arm") != arm:
        fail("视图/阶段/臂")
    data = recorded.get("data") or {}
    if data.get("run_dir") != pool["run_dir"] \
            or data.get("manifest_sha256") != pool["manifest_sha256"]:
        fail("数据目录/manifest")
    if data.get("files") != pool["files"] or data.get("pool") != pool["accounting"]:
        fail("数据文件 SHA/计数")
    if recorded.get("study_sha256") != _study_sha256(study):
        fail("study 冻结稿")
    if device is not None and (recorded.get("device") or {}).get("resolved") != resolve_device(device):
        fail("设备")
    scorer = recorded.get("scorer")
    if not isinstance(scorer, dict):
        fail("打分器绑定缺失")
    if Path(str(scorer.get("dir", ""))).resolve() != Path(scorer_dir).resolve():
        fail("打分器目录")
    for role in ("weights", "meta"):
        name = scorer.get(f"{role}_file")
        if not isinstance(name, str) or _sha256_file(scorer_dir / name) != scorer.get(
                f"{role}_sha256"):
            fail(f"打分器 {role} 哈希")
    if _sha256_file(out / "manifest.json") != recorded.get("manifest_sha256"):
        fail("本 run manifest.json")
    if _sha256_file(out / "selection.json") != recorded.get("selection_sha256"):
        fail("本 run selection.json")


def _check_artifact_unchanged(run_record: Mapping, scorer_dir: Path) -> dict:
    scorer = run_record["scorer"]
    result = {}
    for role in ("weights", "meta"):
        name, recorded = scorer.get(f"{role}_file"), scorer.get(f"{role}_sha256")
        _require(isinstance(name, str) and Path(name).name == name,
                 "corrupt_run", "artifact file")
        current = _sha256_file(scorer_dir / name)
        result[f"{role}_sha256_after"] = current
        _require(current == recorded, "artifact_mutated", name)
    result["unchanged"] = True
    return result


def _load_bound_run(out: Path, study: Mapping, *, pool: Mapping | None = None) -> dict:
    recorded = _read_json(out / "run.json", "run_missing")
    _require(isinstance(recorded, dict) and recorded.get("schema") == RUN_SCHEMA,
             "corrupt_run", "run.json")
    _require(recorded.get("phase") in P.PHASES and recorded.get("arm") in ARMS
             and isinstance(recorded.get("view_id"), str), "corrupt_run", "view/phase/arm")
    view = P.view_spec(study, recorded["view_id"])
    arm = recorded["arm"]
    _require((arm in NLI_ARMS if view["kind"] == "native_nli" else arm == "seven-class")
             and recorded.get("kind") == view["kind"]
             and recorded.get("dataset") == view["dataset"] and recorded.get("format") == P.FORMAT,
             "arm_view_mismatch")
    data, scorer = recorded.get("data"), recorded.get("scorer")
    _require(isinstance(data, dict) and isinstance(data.get("run_dir"), str)
             and isinstance(scorer, dict) and isinstance(scorer.get("dir"), str),
             "corrupt_run", "data/scorer")
    if pool is None:
        pool = load_data_run(data["run_dir"], study=study)
    directory = Path(scorer["dir"])
    artifact = _check_artifact_unchanged(recorded, directory)
    _binding_check(recorded, pool, study=study, view_id=recorded["view_id"],
                   phase=recorded["phase"], arm=arm, scorer_dir=directory, out=out, device=None)
    manifest = P.validate_manifest(_read_json(out / "manifest.json", "manifest_missing"), study)
    verify_manifest_covers_pool(manifest, pool["rows"])
    selection = _read_json(out / "selection.json", "selection_missing")
    recomputed = build_selection(manifest, study=study, view_id=recorded["view_id"],
                                 phase=recorded["phase"])
    _require(selection == recomputed, "selection_drift")
    from research.memory_relations import finetune as FT

    if arm == "semantic":
        from research.memory_relations import semantic_model as SM

        from research.memory_relations import semantic_artifacts as A

        expected = _semantic_identity(directory, view)
        _require(all(scorer.get(key) == value for key, value in expected.items()), "scorer_binding_mismatch")
        meta = A.read_metadata(directory)
        names = (SM.WEIGHTS_FILE, SM.META_FILE)
        fields = ("schema", "kind", "classes", "input_recipe", "direction", "joint_roles")
        window = meta["max_length"]
    elif arm == "native":
        from research.memory_relations import nli_reference as NLI

        meta = project_native_artifact(directory)
        P.native_view_binding(view, meta, study=study)
        names = (NLI.NLI_WEIGHTS_FILE, NLI.NLI_META_FILE)
        fields = ("schema", "kind", "classes", "input_recipe", "direction", "joint_roles")
        window = meta["max_length"]
    else:
        from research.memory_relations import features as FEAT

        meta = _read_json(directory / FT.META_FILE, "artifact_missing")
        _require(isinstance(meta, dict) and meta.get("schema") == FT.ARTIFACT_SCHEMA
                 and meta.get("classes") == list(FEAT.CLASSES), "scorer_binding_mismatch")
        names = (FT.WEIGHTS_FILE, FT.META_FILE)
        fields = ("schema", "kind", "classes", "input_recipe")
        window_kind, window = _window_of(view)
        length = meta.get("max_length")
        _require(type(length) is int and window is not None
                 and (length == window if window_kind == "production_contract" else length <= window),
                 "scorer_window_mismatch")
    _require((scorer.get("weights_file"), scorer.get("meta_file")) == names
             and all(scorer.get(key) == meta.get(key) for key in fields)
             and scorer.get("arm") == arm
             and scorer.get("artifact_max_length") == meta.get("max_length")
             and scorer.get("window") == window
             and scorer.get("window_kind") == (view.get("window") or {}).get("kind"),
             "scorer_binding_mismatch")
    device = recorded.get("device")
    _require(isinstance(device, dict) and isinstance(device.get("resolved"), str)
             and scorer.get("device") == device["resolved"]
             and scorer.get("parameter_device") == device["resolved"]
             and device.get("matmul_device") == device["resolved"], "scorer_device_mismatch")
    return {"run": recorded, "view": view, "manifest": manifest, "selection": selection,
            "artifact": artifact}


def run_view(*, data_dir: object, view_id: str, phase: str, scorer_dir: object, out_dir: object,
             arm: str = "auto", device: str = "cpu", chunk_rows: int = DEFAULT_CHUNK_ROWS,
             limit: int | None = None, manifest_path: object | None = None,
             confirm_final_acceptance: bool = False, score: bool = True,
             scorer_loader: Callable | None = None, study: Mapping | None = None,
             log: Callable[[str], None] | None = None) -> dict:
    """一条可复用入口：准备（数据→manifest→选择）→ 真实推理（断点）→ 计分（协议报告）。

    绑定不符的 resume、不在选择内的结果、重复行、部分覆盖（`--limit` 探针）都 fail loud；
    `phase=final_acceptance` 需要显式确认，避免误揭封存验收。
    """
    study = P.load_study() if study is None else study
    _study_sha256(study)
    _require(phase in P.PHASES, "invalid_phase", str(phase))
    view = P.view_spec(study, view_id)
    kind = view["kind"]
    if kind == "native_nli":
        _require(arm in ("auto", *NLI_ARMS), "arm_view_mismatch", f"{arm} vs {kind}")
        arm = "native" if arm == "auto" else arm
    elif kind == "seven_class_external_diagnostic":
        _require(arm in ("auto", "seven-class"), "arm_view_mismatch", f"{arm} vs {kind}")
        arm = "seven-class"
    else:
        raise EvalError("unknown_view_kind", str(kind))
    if phase == "final_acceptance" and not confirm_final_acceptance:
        raise EvalError("final_acceptance_not_confirmed",
                        "最终验收必须显式 --confirm-final-acceptance（不默认揭封存）")
    _require(isinstance(chunk_rows, int) and chunk_rows >= 1, "invalid_arguments", "chunk_rows")
    _require(limit is None or (isinstance(limit, int) and limit >= 1),
             "invalid_arguments", "limit")
    out = Path(out_dir)
    output = log or (lambda _message: None)
    pool = load_data_run(data_dir, study=study)
    output(f"pool rows={pool['accounting']['rows']} labeled={pool['accounting']['labeled']} "
           f"missing={pool['accounting']['missing_gold']}")
    scorer_dir = Path(scorer_dir)
    resume = (out / "run.json").is_file()
    run_record, manifest = None, None
    if resume:
        run_record = _read_json(out / "run.json", "run_missing")
        _require(isinstance(run_record, dict), "corrupt_run", "run.json")
        _binding_check(run_record, pool, study=study, view_id=view_id, phase=phase, arm=arm,
                       scorer_dir=scorer_dir, out=out, device=device)
        bound = _load_bound_run(out, study, pool=pool)
        manifest, selection = bound["manifest"], bound["selection"]
        if manifest_path is not None:
            supplied = build_manifest(pool["rows"], study=study,
                                      frozen_manifest_path=manifest_path)
            _require(supplied == manifest, "incompatible_resume", "调用方 manifest")
        output("resume：绑定逐字节复核通过")
    else:
        if out.exists():
            _require(not any(out.iterdir()), "out_dir_not_empty", str(out))
        manifest = build_manifest(pool["rows"], study=study,
                                  frozen_manifest_path=manifest_path)
        selection = build_selection(manifest, study=study, view_id=view_id, phase=phase)
        output(f"selection={len(selection['ids'])} binding={selection['binding']}")
    planned = list(selection["ids"])
    if limit is not None:
        planned = planned[:limit]
    if arm == "seven-class":
        handle = load_seven_class_scorer(scorer_dir, view=view, device=device)
    elif arm == "semantic":
        handle = load_semantic_scorer(scorer_dir, view=view, device=device)
    else:
        handle = load_native_scorer(scorer_dir, view=view, device=device,
                                    scorer_loader=scorer_loader, study=study)
    _require(handle["identity"]["parameter_device"] == handle["identity"]["device"],
             "scorer_device_mismatch")
    proof = device_proof(device)
    if run_record is None:
                                             
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "manifest.json", manifest, compact=True)
        _write_json(out / "selection.json", selection)
        run_record = {
            "schema": RUN_SCHEMA, "created_at": _utc_now_iso(), "view_id": view_id,
            "phase": phase, "arm": arm, "kind": kind, "dataset": view["dataset"],
            "data": {"run_dir": pool["run_dir"], "manifest_sha256": pool["manifest_sha256"],
                     "files": pool["files"], "pool": pool["accounting"],
                     "producer_totals": pool["producer_totals"],
                     "train_excluded_rows": pool["train_excluded_rows"]},
            "study_sha256": _study_sha256(study), "format": P.FORMAT,
            "manifest_sha256": _sha256_file(out / "manifest.json"),
            "selection_sha256": _sha256_file(out / "selection.json"),
            "scorer": handle["identity"], "device": proof,
            "limits": {"chunk_rows": chunk_rows, "limit": limit},
        }
        _write_json(out / "run.json", run_record)
    scorer = handle["scorer"]
    state = read_results(out / "results.jsonl", arm=arm, repair_tail=resume)
    if state["repaired_bytes"]:
        output(f"resume：截除崩溃留下的尾部残片 {state['repaired_bytes']} 字节")
    entries = state["entries"]
    planned_set = set(planned)
    _require(all(entry["record_id"] in planned_set for entry in entries),
             "results_out_of_selection", "已有结果超出本次计划行（--limit 变小？）")
    progress_old = _read_json(out / "progress.json", "progress_unreadable") \
        if (out / "progress.json").is_file() else None
    started_at = (progress_old or {}).get("started_at") or _utc_now_iso()
    batches = int((progress_old or {}).get("batches", 0) or 0)
    done_set = {entry["record_id"] for entry in entries}
    remaining = [record_id for record_id in planned if record_id not in done_set]
    by_id = {row["id"]: row for row in pool["rows"]}
    tokenizer = scorer.tokenizer
    max_length = int(scorer.max_length)
    tracker = TruncationTracker(max_length=max_length)
    for start in range(0, len(entries), chunk_rows):
        prior_entries = entries[start:start + chunk_rows]
        prior_rows = [by_id[entry["record_id"]] for entry in prior_entries]
        if arm == "semantic":
            prior_requests = []
            for row, entry in zip(prior_rows, prior_entries):
                if entry["input"] is None:
                    continue
                current = _semantic_audit(scorer.prepare_inputs([semantic_pair(row, view)])[0])
                _require(current == entry["input"], "input_drift", row["id"])
                prior_requests.append(current)
            prior_lengths = [item["full_visible_token_lengths"][0] for item in prior_requests]
        else:
            prior_requests = ([view_request(row, view) for row in prior_rows] if arm == "native" else
                              [request for row in prior_rows
                               for request in diagnostic_requests(row).values()])
            prior_lengths = [row_token_length(tokenizer, request) for request in prior_requests]
        tracker.add(tokenizer, prior_requests, prior_lengths, semantic=arm == "semantic")
    for start in range(0, len(remaining), chunk_rows):
        chunk_ids = remaining[start:start + chunk_rows]
        rows = [by_id[record_id] for record_id in chunk_ids]
        chunk_entries, requests, lengths = _infer_entries(
            arm=arm, view=view, scorer=scorer, rows=rows, tokenizer=tokenizer,
            max_length=max_length)
        tracker.add(tokenizer, requests, lengths, semantic=arm == "semantic")
        _append_entries(out / "results.jsonl", chunk_entries)
        entries.extend(chunk_entries)
        batches += 1
        progress = _progress_payload(run_record=run_record, selection=selection, planned=planned,
                                     entries=entries, batches=batches, started_at=started_at)
        _write_json(out / "progress.json", progress)
        output(f"batch {batches}: done={progress['n_done']}/{progress['n_planned']} "
               f"scored={progress['n_scored']} failed={progress['n_failed']}")
    done_ids = [entry["record_id"] for entry in entries]
    _require(len(done_ids) == len(set(done_ids)), "duplicate_results")
    _require(set(done_ids) == planned_set and len(done_ids) == len(planned),
             "incomplete_selection",
             f"covered={len(done_ids)} planned={len(planned)}（不得省行/子集完成声称）")
    artifact_after = _check_artifact_unchanged(run_record, scorer_dir)
    progress = _progress_payload(run_record=run_record, selection=selection, planned=planned,
                                 entries=entries, batches=batches, started_at=started_at,
                                 infer_complete=True, artifact=artifact_after)
    progress["truncation"] = tracker.summary()
    _write_json(out / "progress.json", progress)
    wrapper = None
    if score:
        wrapper = score_view(out_dir=out, study=study, log=log)
    return {"out_dir": str(out), "run": run_record, "progress": progress, "report": wrapper}


                                                                             
                  
                                                                             

def _diagnostic_report(validated: Mapping, selection: Mapping, entries: Mapping,
                       study: Mapping, view: Mapping) -> tuple:
    dataset = view["dataset"]
    _require(set(entries) == set(selection["ids"]), "incomplete_selection",
             f"results {len(entries)} != selection {len(selection['ids'])}（不得子集计分）")
    by_id = {row["id"]: row for row in validated["diagnostic"]
             + validated["sealed_acceptance"]}
    strata_keys = list(study["datasets"][dataset].get("stratify_by") or [])
    scored, failed, not_attempted = [], 0, 0
    for record_id in selection["ids"]:
        entry = entries[record_id]
        state = _entry_state(entry, arm="seven-class")
        if state == "scored":
            scored.append((by_id[record_id], entry))
        elif state == "failed":
            failed += 1
        else:
            not_attempted += 1
    if not scored:
        report = {
            "status": "no_scored_observations", "view_id": selection["view_id"],
            "phase": selection["phase"], "task_equivalence": "none", "not_accuracy": True,
            "n_selected": len(selection["ids"]),
            "note": "预定行全部未产出概率（全失败/全未尝试）：不产出诊断数字，不冒充坏 payload",
        }
    else:
        probs_by_direction = {
            direction: np.asarray([entry["probs"][direction] for _, entry in scored],
                                  dtype=np.float64)
            for direction in P.DIRECTIONS}
        gold = [record["gold_label"] for record, _ in scored]
        units = [validated["units"][record["id"]] for record, _ in scored]
        strata = ([{key: str(record["metadata"].get(key, "unknown")) for key in strata_keys}
                   for record, _ in scored] if strata_keys else None)
        report = P.external_contradiction_diagnostic(probs_by_direction, gold, units=units,
                                                     strata=strata, study=study)
        _require(report["n"] == len(scored), "report_inconsistent", "行数")
    denominators = {
        "n_selected": len(selection["ids"]), "n_scored": len(scored), "n_failed": failed,
        "n_not_attempted": not_attempted,
        "attempt_rate": {"value": (len(scored) + failed) / len(selection["ids"])
                         if selection["ids"] else None,
                         "numerator": len(scored) + failed, "denominator": len(selection["ids"])},
        "failure_rate": {"value": failed / len(selection["ids"]) if selection["ids"] else None,
                         "numerator": failed, "denominator": len(selection["ids"])},
        "completion": {"failed": failed, "not_attempted": not_attempted,
                       "complete": not failed and not not_attempted},
        "note": "技术失败/未尝试保留在预定集并单列；协议诊断只吃已产出概率的行，"
                "缺标行由协议 n_missing_gold 分列",
    }
    return report, denominators


def _diag_truncation_evidence(entries: Mapping, selection: Mapping) -> dict:
    evidence = {}
    for direction in P.DIRECTIONS:
        truncated = sum(1 for record_id in selection["ids"]
                        if entries[record_id]["truncated"][direction])
        evidence[direction] = {"truncated": truncated,
                               "untruncated": len(selection["ids"]) - truncated}
    evidence["note"] = "w384/w512 差值只在真实被截断桶报告；本模块不产合并数字、不按分数选向"
    return evidence


def score_view(*, out_dir: object, study: Mapping | None = None,
               log: Callable[[str], None] | None = None) -> dict:
    """由权威 manifest + 完整 results 重新派生报告（可重复执行；篡改/缺行 fail loud）。

    原生视图：唯一计分入口 `protocol.evaluate_view`；`final_acceptance` 再加
    `assert_final_complete`（manifest+results 重派生并逐字段比对）。七类诊断视图：
    `external_contradiction_diagnostic` + 运行侧失败分母（不产量准确率）。
    `view_report` 是协议原样输出（可直接喂 `assert_final_complete`/`admission_check`）；
    运行证据放 `runner_evidence`，两域不互相混写。
    """
    study = P.load_study() if study is None else study
    out = Path(out_dir)
    output = log or (lambda _message: None)
    bound = _load_bound_run(out, study)
    run_record, view = bound["run"], bound["view"]
    validated, selection, artifact = bound["manifest"], bound["selection"], bound["artifact"]
    view_id, phase, arm = run_record["view_id"], run_record["phase"], run_record["arm"]
    state = read_results(out / "results.jsonl", arm=arm, repair_tail=False)
    entries_list = state["entries"]
    entries = {entry["record_id"]: entry for entry in entries_list}
    if arm == "semantic":
        _check_semantic_results(entries_list, bound)
    if arm in NLI_ARMS:
        results_map = {entry["record_id"]: {key: entry[key] for key in
                                            ("pred", "error", "truncated")}
                       for entry in entries_list}
        report = P.evaluate_view(None, results_map, study=study, view_id=view_id, phase=phase,
                                 manifest=validated)
        if phase == "final_acceptance":
            P.assert_final_complete(report, manifest=validated, results=results_map, study=study)
        denominators = None
    else:
        report, denominators = _diagnostic_report(validated, selection, entries, study, view)
    evidence = {
        "view_id": view_id, "phase": phase, "arm": arm, "dataset": view["dataset"],
        "selection": {"binding": selection["binding"], "n_selected": len(selection["ids"]),
                      "record_ids_sha256": selection["record_ids_sha256"],
                      "accounting": selection["accounting"]},
        "results": {"file": "results.jsonl", "sha256": state["sha256"],
                    "n_entries": len(entries_list),
                    "n_scored": sum(1 for entry in entries_list
                                    if _entry_state(entry, arm=arm) == "scored"),
                    "n_failed": sum(1 for entry in entries_list
                                    if _entry_state(entry, arm=arm) == "failed"),
                    "n_not_attempted": sum(1 for entry in entries_list
                                           if _entry_state(entry, arm=arm) == "not_attempted")},
        "artifact_integrity": {
            "weights_sha256_before_infer": run_record["scorer"]["weights_sha256"],
            "meta_sha256_before_infer": run_record["scorer"]["meta_sha256"], **artifact},
        "device": run_record["device"], "scorer": run_record["scorer"],
        "window": {"view_window": (view.get("window") or {}),
                   "applied_max_length": run_record["scorer"]["window"]},
        "scored_at": _utc_now_iso(),
        "sha256": {"manifest": run_record["manifest_sha256"],
                   "selection": run_record["selection_sha256"],
                   "data_manifest": run_record["data"]["manifest_sha256"]},
        "notes": ["view_report 是协议原样输出（未经 runner 编辑）；本域只放运行证据",
                  "失败/未尝试/缺标均保留在预定集分母，不删题、不按成绩挑方向",
                  "七类臂永不产 NLI 准确率；原生臂无绑定工件时无法运行，也不得用七类替代"],
    }
    if denominators is not None:
        evidence["denominators"] = denominators
        evidence["truncation"] = _diag_truncation_evidence(entries, selection)
    else:
        progress = _read_json(out / "progress.json", "progress_unreadable") \
            if (out / "progress.json").is_file() else {}
        evidence["truncation"] = (progress or {}).get("truncation")
    wrapper = {"schema": REPORT_SCHEMA, "view_report": report, "runner_evidence": evidence}
    _write_json(out / "report.json", wrapper)
    output(f"report written: {out / 'report.json'}")
    return wrapper


def status_view(*, out_dir: object) -> dict:
    """只读核验绑定并从 results 重派生进度；不推理、不计分、不写盘。"""
    out = Path(out_dir)
    bound = _load_bound_run(out, P.load_study())
    run_record, selection = bound["run"], bound["selection"]
    state = read_results(out / "results.jsonl", arm=run_record["arm"])
    entries = state["entries"]
    if run_record["arm"] == "semantic":
        _check_semantic_results(entries, bound)
    selected_ids = set(selection["ids"])
    _require(all(entry["record_id"] in selected_ids for entry in entries),
             "results_out_of_selection")
    if run_record["arm"] in NLI_ARMS:
        allowed = P.view_label_space(bound["view"])
        _require(all(entry["pred"] is None or entry["pred"] in allowed for entry in entries),
                 "cross_task_prediction")
    old = _read_json(out / "progress.json", "progress_missing") \
        if (out / "progress.json").is_file() else {}
    _require(isinstance(old, dict), "corrupt_run", "progress.json")
    progress = _progress_payload(run_record=run_record, selection=selection,
                                 planned=selection["ids"], entries=entries,
                                 batches=old.get("batches", 0),
                                 started_at=old.get("started_at", run_record["created_at"]),
                                 infer_complete=True, artifact=bound["artifact"])
    has_report = (out / "report.json").is_file()
    current = False
    if has_report:
        report = _read_json(out / "report.json", "report_unreadable")
        evidence = report.get("runner_evidence", {}) if isinstance(report, dict) else {}
        current = (isinstance(report, dict) and isinstance(evidence, dict)
                   and report.get("schema") == REPORT_SCHEMA
                   and isinstance(evidence.get("results"), dict)
                   and evidence["results"].get("sha256") == state["sha256"]
                   and evidence.get("scorer") == run_record["scorer"]
                   and evidence.get("sha256") == {
                       "manifest": run_record["manifest_sha256"],
                       "selection": run_record["selection_sha256"],
                       "data_manifest": run_record["data"]["manifest_sha256"]})
    return {"out_dir": str(out.resolve()), "view_id": run_record["view_id"],
            "phase": run_record["phase"], "arm": run_record["arm"], "progress": progress,
            "has_report": has_report, "report_current": bool(current)}


                                                                             
     
                                                                             

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research.memory_relations.public_benchmarks.evaluate",
        description="公开基准评测运行器（protocol 的唯一评测消费入口；research-only）")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="准备 → 真实推理（断点）→ 计分")
    run.add_argument("--data", required=True, help="datasets.py 产出的 run 目录（只读）")
    run.add_argument("--view", required=True, help="冻结 study 的 view_id")
    run.add_argument("--phase", required=True, choices=list(P.PHASES))
    run.add_argument("--scorer", required=True, help="工件目录（七类meta.json；native nli_meta.json；semantic semantic_meta.json，保留各自权重/schema）")
    run.add_argument("--out", required=True, help="新 run 目录；已存在且为本工具 run 时按断点 resume")
    run.add_argument("--arm", choices=["auto", *ARMS], default="auto")
    run.add_argument("--device", default="cpu", help="cpu 或 cuda[:n]；GPU 不可用/越界即报错，不回退")
    run.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                     help="每批落盘行数（断点粒度）")
    run.add_argument("--limit", type=int, default=None,
                     help="有界探针：只跑选择的前 N 行；partial 结果无法计分（score 会拒绝）")
    run.add_argument("--manifest", default=None,
                     help="可选：父侧冻结 manifest（必须逐字节覆盖评测池）")
    run.add_argument("--confirm-final-acceptance", action="store_true",
                     help="final_acceptance 必须显式确认，避免误揭封存验收")
    run.add_argument("--no-score", action="store_true", help="只推理不立刻计分")
    score = commands.add_parser("score", help="由 manifest+results 重新派生报告")
    score.add_argument("--out", required=True)
    status = commands.add_parser("status", help="只读进度/分母")
    status.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    log = lambda message: print(message, flush=True)                         
    try:
        if args.command == "run":
            result = run_view(data_dir=args.data, view_id=args.view, phase=args.phase,
                              scorer_dir=args.scorer, out_dir=args.out, arm=args.arm,
                              device=args.device, chunk_rows=args.chunk_rows, limit=args.limit,
                              manifest_path=args.manifest,
                              confirm_final_acceptance=args.confirm_final_acceptance,
                              score=not args.no_score, log=log)
            progress = result["progress"]
            print(f"OK out={result['out_dir']} done={progress['n_done']}/"
                  f"{progress['n_planned']} scored={progress['n_scored']} "
                  f"failed={progress['n_failed']}", flush=True)
        elif args.command == "score":
            wrapper = score_view(out_dir=args.out, log=log)
            report = wrapper["view_report"]
            summary = {key: report.get(key) for key in
                       ("view_id", "probe_kind", "phase", "n", "n_total", "accuracy")
                       if key in report}
            print(f"OK score {summary}", flush=True)
        else:
            status = status_view(out_dir=args.out)
            print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except (EvalError, P.ContractError, FileNotFoundError) as exc:
        code = getattr(exc, "code", None) or str(exc)
        print(f"ERROR {code}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
