                       
"""公开基准（OCNLI/ANLI/HANS/ContractNLI）的记录/任务合同与评测口径（research-only）。

本模块是公开基准研究的协议真源；不接生产路径，不改既有七类模型与 v3 实验 schema：

- **记录合同**：`validate_record` 校验归一化 record 的 10 个顶层键与标签语义。gold 只能是
  `entailment / contradiction / neutral / non-entailment / null`（null=缺标/无共识，必须与
  `gold_status` 严格一致）；任何七类关系名都不是 gold（`forbidden_gold_relation`）；
  `original_label` 保留源标注并经 `canonical_gold_label` 映射（未知值 fail loud，官方无监督
  标记如 OCNLI `-` 映射为 null）；每数据集的 `provenance` 由校验器强制执行——`combinations`
  声明 (gold_status, label_method) 词汇表；证据口径两种：**OCNLI 直接复用生产方公共纯锚点**
  `datasets.ocnli_vote_status`（study 声明 function/参数绑定/默认值/比对字段，校验器只调用该
  函数并逐字段比对记录声明，本模块不含平行公式——严格多数=有效票≥2 且 2×同意票>有效票，
  有效票≥2 无严格多数即 `single/no_majority`、gold 不改），其它数据集走通用计数复算
  （如 ANLI verifier_labels 的 `count_field` 必须等于按 `count_mode` 复算的计数并满足上下界）；
  HANS 必须 generated/template_rule；single 允许无公开复核票的单票/单人标注、missing 行
  label_method=no_label。缺 raw 单位 id 的行（含 OCNLI 官方 test 与 train 极少数）允许
  premise_family_key 派生 group_id，但必须披露 group_id_source 且 group_id 与派生值一致
  ——派生与「gold 是否缺失」是两件事。
- **任务不等价探针**：七类概率只作明示任务不等价的外部矛盾诊断
  （`external_contradiction_diagnostic`）——两方向都跑、主向预注册、报翻转率、不得平均；
  概率必须是 simplex 合法分布（非法向量 fail loud、不静默归一化）；监督 AUC/自助只吃有 gold
  的行（缺标绝不当负例，只留在分布/覆盖记账）；禁止 neutral=uncertain、entailment=duplicate、
  non-entailment=conflict 等任何七类↔NLI 映射，也不得用七类产出三分类准确率或风险/放行/认证表述。
- **独立单位与分区（单一 manifest 机制）**：raw 单位 = `group_id`；`partition_records` 按依赖
  cluster（共享 group_id / 规范化 premise / 有向 pair 指纹的连通分量）确定性切 diagnostic /
  sealed_acceptance，cluster 原子、raw identity 原样保留、不删行。**所有曝光/选择/计分/最终
  完成入口只走同一权威 manifest**：`validate_manifest` 逐项复核 purposes/units/clusters/
  每桶实际成员与其用途一致/内容 SHA/计数（篡改即拒），`manifest_scope` 给出视图×阶段预定
  全行集合，`select_for_phase` 先复核再返回行；`evaluate_view` 不再存在无 manifest 的
  fragment 计分旁路（显式拒绝）。最终完成 `assert_final_complete` 由 manifest+完整 results
  重新派生并逐字段比对报告，不信任报告内的 binding/complete/失败计数旗子。初诊只曝光
  diagnostic，最终验收评完所有可用行。
- **指标与失败分母**：`evaluate_view` 以全部预定行为 `n_total`；失败/未尝试逐项报告；
  准确率 = correct / scored_with_gold；簇自助复用 `metrics.cluster_bootstrap`；截断必须有
  显式证据，缺证据只降级全文声明（`windowed_not_full`），`finetune.truncation_profile`
  是唯一口径。
- **准入（证据报告，不自动许可）**：`admission_check` 只报告证据完整性与人工复核要求
  （manual_evidence_review_required），**不自动产出 poor=True/retrain=True**；**任何带
  manifest/results 的调用（完整/不完整/零观测一致）都先用权威输入重算报告并逐字段比对**
  （陈旧/伪造报告 → `report_inconsistent`）；不完整选择返回 `incomplete_selection`、
  零 scored_with_gold（全失败/全未尝试/全缺标/零 eligible）返回 `no_scored_observations`
  的不足证据 descriptor（`evidence_complete=false`、`comparison=null`、计数守恒，不当坏
  payload）；不带权威输入的裸“不完整”声明只作**未验证回显**（`unverified_caller_declaration`），
  绝不当作已观测技术事实。完整观测下逐行推导真实失败行与唯一 cluster 反例；基线一律由验证器
  按冻结定义计算——majority_class = 选择行上最高频 gold、chance = 1/类数（解析值，分母不整除
  类数时不伪造整数分子）、实证基线必须给出逐题预测并在同一成员集复算（规则实现本身未经本协议
  核验）；归因声明只做结构登记、标注未验证，科学归因由父/独立复核决策。预注册比较方向
  （CI 上界严格小于最强匹配基线）保留为描述性统计，不因数据不完整归罪学生。
- **原生视图可用性**：`native_view_binding(view, artifact)` 把正式 runner 绑定的真实工件
  meta（schema/kind/类名集合/标签空间/主向/槽位/输入配方/窗口/哈希/骨干指纹）与冻结的
  `study.native_reference` 注册表逐项比对——配方必须等于注册的 `finetune.INPUT_RECIPE`
  标识（不是任意非空字符串）；可选 `weights_path` 做文件 SHA256 完整性核验，元数据匹配与
  完整性核验分开报告。`available_in_round=false` 只是冻结稿状态，原生视图不得被静默长期不跑。

`study-v4.json` 是冻结稿（未观测任何成绩前冻结；改动须同步 `FROZEN_STUDY_SHA256` 并经独立
审查）。本模块不发请求、不加载模型、不写文件。
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from research.memory_relations import metrics as METRICS
from research.memory_relations.features import CLASSES as SEVEN_CLASSES
                                                
                                                      
from research.memory_relations.public_benchmarks import datasets as PUBLIC_DATASETS

__all__ = [
    "ContractError", "CONDITIONAL_TASKS", "DATASETS", "DIRECTIONS", "FIELDS", "FORMAT",
    "FORBIDDEN_LABEL_MAPPINGS", "FROZEN_STUDY_SHA256", "GOLD_BY_TASK", "GOLD_LABELS",
    "GOLD_STATUSES", "NULL_GOLD_STATUSES", "ORACLE_GOLD_IN", "PHASES", "PRIMARY_DIRECTION",
    "PURPOSES", "STUDY_FILENAME", "TASKS",
    "admission_check", "assert_final_complete", "assign_purpose", "canonical_gold_label",
    "check_label_catalog", "check_leakage", "claim_status", "cluster_records",
    "diagnostic_request", "evaluate_view", "external_contradiction_diagnostic", "load_study",
    "manifest_scope", "native_view_binding", "NLI_CLASSES", "oracle_available",
    "oracle_eligible", "oracle_premise", "partition_records", "premise_family_key",
    "select_for_phase", "study_sha256", "validate_manifest", "validate_record",
    "validate_record_for_dataset", "validate_study", "view_label_space", "view_spec",
]

FORMAT = "memory-relations-public-benchmarks-v4"
STUDY_FILENAME = "study-v4.json"
DATASETS = ("ocnli", "anli", "hans", "contractnli")
TASKS = ("nli3", "entailment2")
GOLD_BY_TASK = {
    "nli3": ("entailment", "contradiction", "neutral"),
    "entailment2": ("entailment", "non-entailment"),
}
                                                         
CONDITIONAL_TASKS = {"entailment_contradiction": ("entailment", "contradiction")}
ORACLE_GOLD_IN = ("entailment", "contradiction")
GOLD_LABELS = ("entailment", "contradiction", "neutral", "non-entailment")
GOLD_STATUSES = ("adjudicated", "majority", "single", "generated", "no_consensus", "missing")
NULL_GOLD_STATUSES = ("no_consensus", "missing")
                                              
NLI_CLASSES = ("entailment", "neutral", "contradiction")
                                                                     
PRIMARY_JOINT_ROLES = {"premise": "existing", "hypothesis": "candidate"}
                                                        
                                                                          
PROVENANCE_COUNT_MODES = ("valid_labels", "agree_with_original_label")
                                              
BASELINE_DERIVATIONS = ("most_frequent_gold", "uniform_random_1_over_k",
                        "per_record_predictions")
                                                             
                                                                
                                                     
NATIVE_REFERENCE_SCHEMA = "memory-relations-nli-reference-scorer-v1"
NATIVE_REFERENCE_KIND = "nli-reference-joint"
NATIVE_INPUT_RECIPE = "memory-relations-joint-input-v1"
NATIVE_LABEL_SPACE = "nli3"
FIELDS = ("id", "dataset", "official_split", "task", "premise", "hypothesis",
          "gold_label", "group_id", "metadata", "evidence")
PHASES = ("diagnostic", "final_acceptance")
PURPOSES = ("diagnostic", "sealed_acceptance")
                                                               
                                                     
DIRECTIONS = ("hypothesis_as_candidate", "premise_as_candidate")
PRIMARY_DIRECTION = DIRECTIONS[0]
                                          
FORBIDDEN_LABEL_MAPPINGS = (
    ("neutral", "uncertain"),
    ("entailment", "duplicate"),
    ("non-entailment", "conflict"),
)
                                                         
                                                                                    
                                                 
                                                            
NATIVE_LABEL_MAPS = {
    "ocnli": {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction",
              "-": None},
    "anli": {"e": "entailment", "n": "neutral", "c": "contradiction"},
    "hans": {"entailment": "entailment", "non-entailment": "non-entailment"},
    "contractnli": {"Entailment": "entailment", "Contradiction": "contradiction",
                    "NotMentioned": "neutral"},
}
WINDOW_KINDS = ("production_contract", "backbone_fulltext")
METADATA_MAX_BYTES = 16384
EVIDENCE_MAX_BYTES = 16384
MAX_DEPTH = 6
                                                      
SIMPLEX_TOLERANCE = 1e-5
                                                                    
FROZEN_STUDY_SHA256 = "eac11a96906fb87c68812af273308d062107a58912f2be7a35405e9f45439ffb"


class ContractError(ValueError):
    """合同错误；`code` 是有界机器码，未知字段不回显。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


                                                                             
      
                                                                             

def _text(value: object, code: str, *, max_length: int = 65536) -> str:
    if type(value) is not str or not value.strip():
        raise ContractError(code)
    if len(value) > max_length:
        raise ContractError(code)
    return value


def _depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(item) for item in value), default=0)
    return 0


def _jsonable(value: object, code: str, *, max_bytes: int) -> None:
    if _depth(value) > MAX_DEPTH:
        raise ContractError(code)
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ContractError(code) from None
    if len(serialized.encode("utf-8")) > max_bytes:
        raise ContractError(code)


def canonical_gold_label(dataset: str, original_label: str, *,
                         catalog: Mapping | None = None) -> "str | None":
    """源标注 → 规范 gold；`catalog`（运行时唯一维护锚点，如 sources.json label_map）优先，
    缺省用冻结快照；映射值可为 `None`（官方“无监督标签”标记，如 OCNLI 无共识 `-`）；
    未知 dataset/标签一律 fail loud。"""
    if catalog is None:
        if dataset not in NATIVE_LABEL_MAPS:
            raise ContractError("invalid_dataset")
        table = NATIVE_LABEL_MAPS[dataset]
    else:
        table = catalog.get(dataset)
        if type(table) is not dict:
            raise ContractError("invalid_dataset")
    _text(original_label, "unknown_original_label", max_length=64)
    if original_label not in table:
        raise ContractError("unknown_original_label")
    mapped = table[original_label]
    if mapped is not None and (type(mapped) is not str or not mapped):
        raise ContractError("unknown_original_label")
    return mapped


def check_label_catalog(catalog: Mapping) -> None:
    """运行时 label_map 与冻结快照 parity；漂移 fail loud（改映射须走冻结修订）。"""
    if type(catalog) is not dict:
        raise ContractError("invalid_dataset")
    for dataset in DATASETS:
        if catalog.get(dataset) != NATIVE_LABEL_MAPS[dataset]:
            raise ContractError("label_map_drift")


def _check_metadata(record: dict, *, catalog: Mapping | None = None) -> None:
    metadata = record["metadata"]
    if type(metadata) is not dict or not metadata:
        raise ContractError("invalid_metadata")
    _jsonable(metadata, "invalid_metadata", max_bytes=METADATA_MAX_BYTES)
    status = metadata.get("gold_status")
    if status not in GOLD_STATUSES:
        raise ContractError("invalid_gold_status")
    gold = record["gold_label"]
    if (gold is None) != (status in NULL_GOLD_STATUSES):
        raise ContractError("gold_status_mismatch")
    original = metadata.get("original_label")
    if gold is None:
        if original is None:
            return
        if type(original) is not str or not original.strip():
            raise ContractError("invalid_metadata")
        if canonical_gold_label(record["dataset"], original, catalog=catalog) is not None:
            raise ContractError("original_gold_mismatch")
        return
    if type(original) is not str or not original.strip():
        raise ContractError("missing_original_label")
    if canonical_gold_label(record["dataset"], original, catalog=catalog) != gold:
        raise ContractError("original_gold_mismatch")


def _check_evidence(record: dict) -> None:
    evidence = record["evidence"]
    if type(evidence) is not dict:
        raise ContractError("invalid_evidence")
    _jsonable(evidence, "invalid_evidence", max_bytes=EVIDENCE_MAX_BYTES)
    _text(evidence.get("source_url"), "invalid_evidence", max_length=4096)
    _text(evidence.get("file"), "invalid_evidence", max_length=1024)
    sha = evidence.get("sha256")
    if sha is not None and (type(sha) is not str or len(sha) != 64
                            or any(character not in "0123456789abcdef" for character in sha)):
        raise ContractError("invalid_evidence")
    row = evidence.get("row")
    if row is not None and not (type(row) is int
                                or (type(row) is str and row.strip())):
        raise ContractError("invalid_evidence")


def validate_record(record: object, *, catalog: Mapping | None = None) -> None:
    """归一化 record 的核心合同；失败抛 `ContractError`，通过返回 None。

    校验顶层键集合、类型、task↔gold 一致性、gold↔gold_status 一致性、original_label 与
    规范映射一致（`catalog` 为运行时映射锚点，缺省用冻结快照）、七类名不得作 gold、
    metadata/evidence 有界可序列化与 provenance 必填项。
    """
    if type(record) is not dict or set(record) != set(FIELDS):
        raise ContractError("invalid_record")
    _text(record["id"], "invalid_record", max_length=256)
    if record["dataset"] not in DATASETS:
        raise ContractError("invalid_dataset")
    _text(record["official_split"], "invalid_record", max_length=128)
    task = record["task"]
    if task not in TASKS:
        raise ContractError("invalid_task")
    _text(record["premise"], "invalid_record", max_length=500000)
    _text(record["hypothesis"], "invalid_record", max_length=500000)
    gold = record["gold_label"]
    if gold is not None:
        if type(gold) is not str:
            raise ContractError("invalid_gold")
        if gold in SEVEN_CLASSES:
            raise ContractError("forbidden_gold_relation")
        if gold not in GOLD_BY_TASK[task]:
            raise ContractError("invalid_gold")
    _text(record["group_id"], "invalid_record", max_length=256)
    _check_metadata(record, catalog=catalog)
    _check_evidence(record)


def _check_derivation(record: dict, provenance: Mapping, derivation: Mapping) -> None:
    """生产者公共纯锚点复算：直接调用 study 声明的函数（唯一实现，无平行公式），逐字段比对。

    - 参数按 `arguments` 声明从 metadata 取值（缺失/None 用声明的 `default`）；
    - `require` 字段（status/method）必须与锚点返回值完全一致；
    - `require_when_declared` 字段（计数）在记录声明时也必须一致（旧形状未存计数时不强制）。
    """
    function = getattr(PUBLIC_DATASETS, derivation["function"], None)
    if not callable(function):
        raise ContractError("invalid_study")
    metadata = record["metadata"]
    arguments = {}
    for name, item in derivation["arguments"].items():
        value = metadata.get(item["field"])
        arguments[name] = item["default"] if value is None else value
    derived = function(**arguments)
    if type(derived) is not dict:
        raise ContractError("invalid_provenance")
    for field in derivation["require"]:
        if metadata.get(field) != derived.get(field):
            raise ContractError("invalid_provenance")
    for field in derivation.get("require_when_declared") or ():
        declared = metadata.get(field)
        if declared is not None and declared != derived.get(field):
            raise ContractError("invalid_provenance")


def _check_provenance(record: dict, spec: Mapping) -> None:
    """数据集 provenance 的通用执法：词汇表匹配 + （生产锚点复算 或 计数复算/上下界）。

    `provenance.combinations` 声明 (gold_status, label_method) 词汇；同一 gold_status 可多条覆盖，
    按 (status, method) 唯一匹配。两种证据口径：
    - **`derivation`（首选，唯一实现）**：声明生产者公共纯函数名与参数绑定/默认值/比对字段；
      校验器直接调用该函数并比对返回值——本模块不含任何平行公式。
    - **count mechanism**（未声明 derivation 时）：按 `count_mode` 从真实证据字段复算
      `count_field` 并执行 min_count/max_count 上下界。
    未声明的词汇、锚点复算或计数不一致一律 `invalid_provenance`。
    """
    provenance = spec.get("provenance") or {}
    metadata = record["metadata"]
    status = metadata.get("gold_status")
    method = metadata.get("label_method")
    combos = provenance.get("combinations") or ()
    candidates = [item for item in combos if status in item["gold_status"]]
    matched = [item for item in candidates if method in item["label_method"]]
    if len(matched) != 1:
        raise ContractError("invalid_provenance")
    derivation = provenance.get("derivation")
    if derivation is not None:
        _check_derivation(record, provenance, derivation)
        return
    combo = matched[0]
    count_field = provenance.get("count_field")
    if count_field is None:
        return
    count = metadata.get(count_field)
    if count is not None and type(count) is not int:
        raise ContractError("invalid_provenance")
    evidence = metadata.get(provenance["count_of"])
    if evidence is not None:
        if type(evidence) is not list:
            raise ContractError("invalid_provenance")
        if provenance["count_mode"] == "valid_labels":
            valid_values = provenance["valid_values"]
            recomputed = sum(1 for item in evidence
                             if isinstance(item, str) and item in valid_values)
        else:
            original = metadata.get("original_label")
            if type(original) is not str or not original.strip():
                raise ContractError("invalid_provenance")
            recomputed = sum(1 for item in evidence if item == original)
        if count != recomputed:
            raise ContractError("invalid_provenance")
    elif count is not None:
        raise ContractError("invalid_provenance")
    if "min_count" in combo and (count is None or count < combo["min_count"]):
        raise ContractError("invalid_provenance")
    if "max_count" in combo and count is not None and count > combo["max_count"]:
        raise ContractError("invalid_provenance")


def _check_required_metadata(record: dict, spec: Mapping) -> None:
    metadata = record["metadata"]
    unit = spec["unit"]
    unit_key = unit["metadata_key"]
    fallback = unit.get("fallback")
    for item in spec.get("required_metadata") or ():
        key = item["key"]
        if metadata.get(key) is None and key == unit_key and fallback is not None:
            continue
        if key not in metadata:
            raise ContractError("missing_required_metadata")
        value = metadata[key]
        kind = item["type"]
        if kind == "str":
            _text(value, "invalid_metadata_value", max_length=512)
        elif kind == "text":
            if type(value) is int:
                value = str(value)
            _text(value, "invalid_metadata_value", max_length=512)
            allowed = item.get("allowed")
            if allowed is not None and str(value).casefold() not in {str(x).casefold() for x in allowed}:
                raise ContractError("invalid_metadata_value")
        elif kind == "int":
            if type(value) is not int:
                raise ContractError("invalid_metadata_value")
        elif kind == "list":
            if type(value) is not list:
                raise ContractError("invalid_metadata_value")
        else:
            raise ContractError("invalid_study")
    _check_provenance(record, spec)
    unit_value = metadata.get(unit_key)
    derived = premise_family_key(record["premise"])
    has_unit = type(unit_value) is int or (isinstance(unit_value, str)
                                           and bool(unit_value.strip()))
    if has_unit:
        if str(unit_value) != record["group_id"]:
            raise ContractError("unit_mismatch")
        if unit.get("derive") == "premise_family_key" and record["group_id"] != derived:
            raise ContractError("premise_hash_mismatch")
        return
    if fallback is None:
        raise ContractError("missing_unit_metadata")
    if record["group_id"] != derived:
        raise ContractError("premise_hash_mismatch")
    if metadata.get(fallback["source_field"]) != fallback["source_value"]:
        raise ContractError("unit_fallback_missing_source")


def validate_record_for_dataset(record: object, study: Mapping | None = None, *,
                                catalog: Mapping | None = None) -> None:
    """核心合同 + 数据集级 required_metadata 与独立单位一致性。"""
    validate_record(record, catalog=catalog)
    study = load_study() if study is None else study
    spec = study["datasets"].get(record["dataset"])
    if spec is None:
        raise ContractError("invalid_dataset")
    _check_required_metadata(record, spec)


def _span_texts(spans: object) -> list:
    if spans is None:
        return []
    if type(spans) is not list:
        raise ContractError("invalid_spans")
    texts = []
    for span in spans:
        if type(span) is not dict:
            raise ContractError("invalid_spans")
        text = span.get("text")
        if type(text) is not str or not text.strip():
            raise ContractError("invalid_spans")
        start, end = span.get("start"), span.get("end")
        for value in (start, end):
            if value is not None and type(value) is not int:
                raise ContractError("invalid_spans")
        if start is not None and end is not None and start > end:
            raise ContractError("invalid_spans")
        texts.append(text)
    return texts


def oracle_available(record: object) -> bool:
    """该行是否有可用 gold 证据跨度；缺失只影响 oracle 层计数，不删行、不影响其它视图。"""
    validate_record(record)
    return bool(_span_texts(record["metadata"].get("spans")))


def oracle_premise(record: object) -> str:
    """oracle 层受控取用：只读 `metadata.spans` 的 text 按行拼接；缺失 fail loud。"""
    validate_record(record)
    texts = _span_texts(record["metadata"].get("spans"))
    if not texts:
        raise ContractError("oracle_spans_unavailable")
    return "\n".join(texts)


def premise_family_key(premise: str) -> str:
    """ANLI 独立前提族的规范键：NFKC + 空白折叠 + strip 后的内容 hash（精确规范化）。"""
    _text(premise, "invalid_record", max_length=500000)
    normalized = " ".join(unicodedata.normalize("NFKC", premise).split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return "premise:" + digest[:16]


                                                                             
           
                                                                             

def study_path() -> Path:
    return Path(__file__).resolve().parent / STUDY_FILENAME


def study_sha256(path: object | None = None) -> str:
    target = Path(path) if path is not None else study_path()
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_study(path: object | None = None, *, verify_freeze: bool = True) -> dict:
    """读取并校验冻结 study；默认核对 `FROZEN_STUDY_SHA256`（改动即 fail loud）。"""
    target = Path(path) if path is not None else study_path()
    raw = target.read_bytes()
    if verify_freeze and hashlib.sha256(raw).hexdigest() != FROZEN_STUDY_SHA256:
        raise ContractError("study_frozen_mismatch")
    try:
        study = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContractError("invalid_study") from None
    validate_study(study)
    return study


def _require(value: object, code: str) -> None:
    if not value:
        raise ContractError(code)


def _validate_view(view: object, dataset_specs: Mapping, seen: set) -> None:
    if type(view) is not dict:
        raise ContractError("invalid_study")
    view_id = view.get("view_id")
    _text(view_id, "invalid_study", max_length=128)
    if view_id in seen:
        raise ContractError("invalid_study")
    seen.add(view_id)
    dataset = view.get("dataset")
    if dataset not in dataset_specs:
        raise ContractError("invalid_study")
    kind = view.get("kind")
    if kind not in ("native_nli", "seven_class_external_diagnostic"):
        raise ContractError("invalid_study")
    task = view.get("task")
    dataset_task = dataset_specs[dataset]["task"]
    if task in TASKS:
        if task != dataset_task:
            raise ContractError("invalid_study")
    elif task in CONDITIONAL_TASKS:
        if view.get("subset_of") != dataset_task:
            raise ContractError("invalid_study")
    else:
        raise ContractError("invalid_study")
    metrics = view.get("metrics")
    if type(metrics) is not list or not metrics or any(type(x) is not str for x in metrics):
        raise ContractError("invalid_study")
    if type(view.get("available_in_round")) is not bool:
        raise ContractError("invalid_study")
    if kind == "native_nli":
        if any(name in metrics for name in ("output_distribution", "flip_rate")):
            raise ContractError("invalid_study")
        if "accuracy" not in metrics:
            raise ContractError("invalid_study")
        if view["available_in_round"]:
            raise ContractError("invalid_study")
        _text(view.get("unavailable_reason"), "invalid_study", max_length=2000)
        if view.get("availability_rule") != "artifact_binding_required":
            raise ContractError("invalid_study")
        scope_rule = view.get("scope_rule")
        if scope_rule not in ("all_rows", "gold_evidence_two_class"):
            raise ContractError("invalid_study")
        spaces = view.get("arm_spaces")
        if type(spaces) is not list or not spaces:
            raise ContractError("invalid_study")
        expected_space = view.get("subset_of") if task in CONDITIONAL_TASKS else task
        for space in spaces:
            if space not in TASKS or not (space == expected_space
                                          or (space == "nli3" and expected_space == "entailment2")):
                raise ContractError("invalid_study")
        if type(view.get("reductions")) is not dict or type(view.get("arm_spec")) is not str:
            raise ContractError("invalid_study")
        if type(view.get("input")) is not dict:
            raise ContractError("invalid_study")
        layer = view.get("layer")
        if layer not in ("standard", "full_document", "oracle_evidence"):
            raise ContractError("invalid_study")
        if layer == "full_document":
            if view.get("claim_full_text") is not True or type(view.get("window")) is not dict \
                    or scope_rule != "all_rows":
                raise ContractError("invalid_study")
        if layer == "oracle_evidence":
            eligibility = view.get("eligibility")
            if view.get("oracle") is not True or task not in CONDITIONAL_TASKS \
                    or scope_rule != "gold_evidence_two_class" \
                    or type(eligibility) is not dict \
                    or eligibility.get("gold_in") != list(CONDITIONAL_TASKS[task]) \
                    or eligibility.get("require_spans") is not True:
                raise ContractError("invalid_study")
        if layer == "standard" and scope_rule != "all_rows":
            raise ContractError("invalid_study")
    else:
        if any(name in metrics for name in ("accuracy", "macro_f1")):
            raise ContractError("invalid_study")
        if view.get("task_equivalence") != "none" or not view["available_in_round"]:
            raise ContractError("invalid_study")
        window = view.get("window")
        if type(window) is not dict or window.get("kind") not in WINDOW_KINDS \
                or type(window.get("max_tokens")) is not int:
            raise ContractError("invalid_study")
        directions = view.get("directions")
        if type(directions) is not dict or directions.get("primary") != PRIMARY_DIRECTION \
                or directions.get("swap") != DIRECTIONS[1]:
            raise ContractError("invalid_study")


def _validate_provenance_spec(name: str, spec: Mapping) -> None:
    """数据集 provenance 结构执法（组合词汇表 + 生产锚点 derivation 或计数机制；HANS 精确冻结）。"""
    provenance = spec.get("provenance")
    if type(provenance) is not dict:
        raise ContractError("invalid_study")
    combos = provenance.get("combinations")
    if type(combos) is not list or not combos:
        raise ContractError("invalid_study")
    derivation = provenance.get("derivation")
    if derivation is not None:
        if type(derivation) is not dict \
                or set(derivation) != {"function", "arguments", "require", "require_when_declared"} \
                or type(derivation.get("function")) is not str \
                or not derivation["function"].strip() \
                or type(derivation.get("arguments")) is not dict or not derivation["arguments"] \
                or any(type(argument) is not str or not argument.strip()
                       or type(item) is not dict or set(item) != {"field", "default"}
                       or type(item.get("field")) is not str or not item["field"].strip()
                       for argument, item in derivation["arguments"].items()) \
                or type(derivation.get("require")) is not list or not derivation["require"] \
                or any(type(field) is not str or not field.strip()
                       for field in derivation["require"]) \
                or type(derivation.get("require_when_declared")) is not list \
                or any(type(field) is not str or not field.strip()
                       for field in derivation["require_when_declared"]):
            raise ContractError("invalid_study")
        if not callable(getattr(PUBLIC_DATASETS, derivation["function"], None)):
            raise ContractError("invalid_study")
    by_status: dict = {}
    for combo in combos:
        if type(combo) is not dict \
                or not set(combo) <= {"gold_status", "label_method", "min_count", "max_count"} \
                or type(combo.get("gold_status")) is not list or not combo["gold_status"] \
                or any(status not in GOLD_STATUSES for status in combo["gold_status"]) \
                or type(combo.get("label_method")) is not list or not combo["label_method"] \
                or any(type(method) is not str or not method.strip()
                       for method in combo["label_method"]):
            raise ContractError("invalid_study")
        for status in combo["gold_status"]:
            by_status.setdefault(status, []).append(set(combo["label_method"]))
        for bound in ("min_count", "max_count"):
            if bound in combo and (type(combo[bound]) is not int or combo[bound] < 0):
                raise ContractError("invalid_study")
        if "min_count" in combo and "max_count" in combo \
                and combo["min_count"] > combo["max_count"]:
            raise ContractError("invalid_study")
    for method_sets in by_status.values():
        for index in range(len(method_sets)):
            for other in range(index + 1, len(method_sets)):
                if method_sets[index] & method_sets[other]:
                    raise ContractError("invalid_study")
    if name == "hans":
        if combos != [{"gold_status": ["generated"], "label_method": ["template_rule"]},
                      {"gold_status": ["missing"], "label_method": ["no_label"]}]:
            raise ContractError("invalid_study")
    declared = [key for key in ("count_field", "count_of", "count_mode") if key in provenance]
    if declared and len(declared) != 3:
        raise ContractError("invalid_study")
    if not declared:
        if any("min_count" in combo or "max_count" in combo for combo in combos):
            raise ContractError("invalid_study")
        return
    if type(provenance["count_field"]) is not str or not provenance["count_field"].strip() \
            or type(provenance["count_of"]) is not str or not provenance["count_of"].strip() \
            or provenance["count_mode"] not in PROVENANCE_COUNT_MODES:
        raise ContractError("invalid_study")
    if provenance["count_mode"] == "valid_labels":
        values = provenance.get("valid_values")
        if type(values) is not list or not values \
                or any(type(value) is not str or not value.strip() for value in values):
            raise ContractError("invalid_study")


def validate_study(study: object, *, strict_mirror: bool = True) -> None:
    """校验冻结 study 结构与镜像一致性；失败抛 `ContractError`。"""
    if type(study) is not dict:
        raise ContractError("invalid_study")
    if study.get("format") != FORMAT or study.get("status") != "research-only":
        raise ContractError("invalid_study")
    freeze = study.get("freeze")
    if type(freeze) is not dict or freeze.get("status") != "frozen_before_results":
        raise ContractError("invalid_study")
    for key in ("frozen_at", "basis_commit", "score_exposure", "amendment_policy"):
        _text(freeze.get(key), "invalid_study", max_length=4000)
    history = freeze.get("revision_history")
    if type(history) is not list or not history:
        raise ContractError("invalid_study")
    for item in history:
        if type(item) is not dict or set(item) != {"revision", "study_sha256", "commit", "note"} \
                or any(type(item[key]) is not str or not item[key].strip() for key in item) \
                or len(item["study_sha256"]) != 64:
            raise ContractError("invalid_study")
    dataset_specs = study.get("datasets")
    if type(dataset_specs) is not dict or set(dataset_specs) != set(DATASETS):
        raise ContractError("invalid_study")
    for name, spec in dataset_specs.items():
        if type(spec) is not dict or spec.get("task") not in TASKS:
            raise ContractError("invalid_study")
        _text(spec.get("language"), "invalid_study", max_length=16)
        unit = spec.get("unit")
        if type(unit) is not dict or type(unit.get("metadata_key")) is not str:
            raise ContractError("invalid_study")
        required = spec.get("required_metadata")
        if type(required) is not list or not required:
            raise ContractError("invalid_study")
        for item in required:
            if type(item) is not dict or type(item.get("key")) is not str \
                    or item.get("type") not in ("str", "text", "int", "list"):
                raise ContractError("invalid_study")
        if type(spec.get("stratify_by")) is not list:
            raise ContractError("invalid_study")
        _validate_provenance_spec(name, spec)
        if unit.get("derive") not in (None, "premise_family_key"):
            raise ContractError("invalid_study")
        fallback = unit.get("fallback")
        if fallback is not None:
            if type(fallback) is not dict or fallback.get("derive") != "premise_family_key" \
                    or type(fallback.get("source_field")) is not str \
                    or not fallback["source_field"].strip() \
                    or type(fallback.get("source_value")) is not str \
                    or not fallback["source_value"].strip():
                raise ContractError("invalid_study")
    views = study.get("views")
    if type(views) is not list or not views:
        raise ContractError("invalid_study")
    seen: set = set()
    by_dataset: dict[str, set] = {name: set() for name in DATASETS}
    for view in views:
        _validate_view(view, dataset_specs, seen)
        by_dataset[view["dataset"]].add(view.get("layer") if view["kind"] == "native_nli"
                                        else view.get("window", {}).get("kind"))
    for name, kinds in by_dataset.items():
        if not kinds:
            raise ContractError("invalid_study")
        if name == "contractnli" and not ({"full_document", "oracle_evidence"} <= kinds):
            raise ContractError("invalid_study")
        if name != "contractnli" and "standard" not in kinds:
            raise ContractError("invalid_study")
    partition = study.get("partition")
    if type(partition) is not dict:
        raise ContractError("invalid_study")
    _text(partition.get("salt"), "invalid_study", max_length=128)
    fraction = partition.get("diagnostic_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) \
            or not 0.0 < float(fraction) < 1.0:
        raise ContractError("invalid_study")
    if partition.get("purposes") != list(PURPOSES) or partition.get("phases") != list(PHASES):
        raise ContractError("invalid_study")
    metrics = study.get("metrics")
    if type(metrics) is not dict:
        raise ContractError("invalid_study")
    bootstrap = metrics.get("bootstrap")
    if type(bootstrap) is not dict or bootstrap.get("engine") != "research.memory_relations.metrics.cluster_bootstrap" \
            or type(bootstrap.get("samples")) is not int or bootstrap["samples"] < 1 \
            or not isinstance(bootstrap.get("confidence"), (int, float)) \
            or not 0.0 < float(bootstrap["confidence"]) < 1.0 \
            or type(bootstrap.get("seed")) is not int:
        raise ContractError("invalid_study")
    if metrics.get("exclusion_reasons") != ["gold_no_consensus", "gold_missing"]:
        raise ContractError("invalid_study")
    truncation = metrics.get("truncation")
    if type(truncation) is not dict or truncation.get("buckets") != ["untruncated", "truncated"] \
            or truncation.get("full_text_claim_requires_zero_truncation") is not True \
            or truncation.get("cross_window_delta_only_in_truncated_bucket") is not True \
            or truncation.get("explicit_evidence_required") is not True \
            or truncation.get("missing_evidence_downgrades_full_claim") is not True:
        raise ContractError("invalid_study")
    admission = study.get("admission")
    if type(admission) is not dict or type(admission.get("poor_decision")) is not dict \
            or type(admission.get("retrain_admission")) is not dict:
        raise ContractError("invalid_study")
    if admission.get("mode") != "evidence_report_only" \
            or admission.get("decision_owner") != "parent_with_independent_review" \
            or admission.get("manual_evidence_review_required") is not True:
        raise ContractError("invalid_study")
    profiles = admission.get("baseline_profiles")
    if type(profiles) is not dict or not profiles:
        raise ContractError("invalid_study")
    for profile in profiles.values():
        if type(profile) is not dict or profile.get("supervision") != "gold_supervised" \
                or type(profile.get("recipe")) is not str or not profile["recipe"].strip() \
                or type(profile.get("label_spaces")) is not list or not profile["label_spaces"] \
                or any(space not in TASKS for space in profile["label_spaces"]) \
                or profile.get("derivation") not in BASELINE_DERIVATIONS:
            raise ContractError("invalid_study")
    if admission.get("poor_requires_complete_selection") is not True \
            or admission.get("attribution_evidence_required") is not True:
        raise ContractError("invalid_study")
    if type(admission.get("attribution_classes")) is not dict \
            or "supervision_unfair_gap" not in admission["attribution_classes"]:
        raise ContractError("invalid_study")
    if type(admission.get("language")) is not dict:
        raise ContractError("invalid_study")
    if type(admission.get("min_counterexample_units")) is not int \
            or admission["min_counterexample_units"] < 1:
        raise ContractError("invalid_study")
    native_reference = study.get("native_reference")
    if type(native_reference) is not dict \
            or any(type(native_reference.get(key)) is not str or not native_reference[key].strip()
                   for key in ("schema", "kind", "label_space", "input_recipe", "recipe_source",
                               "direction")):
        raise ContractError("invalid_study")
    if type(native_reference.get("max_tokens_standard")) is not int \
            or native_reference["max_tokens_standard"] < 1:
        raise ContractError("invalid_study")
    roles = native_reference.get("joint_roles")
    if type(roles) is not dict or set(roles) != set(PRIMARY_JOINT_ROLES) \
            or any(roles[key] != PRIMARY_JOINT_ROLES[key] for key in PRIMARY_JOINT_ROLES):
        raise ContractError("invalid_study")
    classes = native_reference.get("classes")
    if type(classes) is not list or len(classes) != len(NLI_CLASSES) \
            or set(classes) != set(NLI_CLASSES):
        raise ContractError("invalid_study")
    forbidden = study.get("forbidden")
    if type(forbidden) is not dict or type(forbidden.get("practices")) is not list \
            or not forbidden["practices"]:
        raise ContractError("invalid_study")
    label_spaces = study.get("label_spaces")
    if type(label_spaces) is not dict:
        raise ContractError("invalid_study")
    if type(study.get("interfaces")) is not dict or type(study.get("axes")) is not dict:
        raise ContractError("invalid_study")
    if strict_mirror:
        if label_spaces.get("canonical_mapping") != NATIVE_LABEL_MAPS:
            raise ContractError("study_mirror_mismatch")
        if label_spaces.get("diagnostic_seven") != list(SEVEN_CLASSES):
            raise ContractError("study_mirror_mismatch")
        if label_spaces.get("tasks") != list(TASKS):
            raise ContractError("study_mirror_mismatch")
        if label_spaces.get("canonical_gold") != list(GOLD_LABELS) + [None]:
            raise ContractError("study_mirror_mismatch")
        if label_spaces.get("gold_statuses") != list(GOLD_STATUSES):
            raise ContractError("study_mirror_mismatch")
        if label_spaces.get("nli_classes") != list(NLI_CLASSES):
            raise ContractError("study_mirror_mismatch")
        native = study["native_reference"]
        if native.get("schema") != NATIVE_REFERENCE_SCHEMA \
                or native.get("kind") != NATIVE_REFERENCE_KIND \
                or native.get("input_recipe") != NATIVE_INPUT_RECIPE \
                or native.get("label_space") != NATIVE_LABEL_SPACE \
                or native.get("classes") != list(NLI_CLASSES) \
                or native.get("direction") != PRIMARY_DIRECTION \
                or native.get("joint_roles") != dict(PRIMARY_JOINT_ROLES):
            raise ContractError("study_mirror_mismatch")
        if forbidden.get("label_mappings") != [list(pair) for pair in FORBIDDEN_LABEL_MAPPINGS]:
            raise ContractError("study_mirror_mismatch")


                                                                             
                
                                                                             

def assign_purpose(study: Mapping, dataset: str, cluster_id: str) -> str:
    """确定性用途分配：sha256(salt \\x1f dataset \\x1f cluster_id) 前 64 bit 落 diagnostic。

    入参是依赖 cluster（连通组）id，不是单行 group_id；同一 cluster 恒同用途。
    """
    if dataset not in DATASETS:
        raise ContractError("invalid_dataset")
    _text(cluster_id, "invalid_record", max_length=256)
    spec = study["partition"]
    digest = hashlib.sha256(
        f"{spec['salt']}\x1f{dataset}\x1f{cluster_id}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(1 << 64)
    return "diagnostic" if value < float(spec["diagnostic_fraction"]) else "sealed_acceptance"


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _row_fingerprint(record: Mapping) -> str:
    return "\x1e".join((record["dataset"], record["group_id"],
                        _normalize_text(record["premise"]),
                        _normalize_text(record["hypothesis"])))


def _cluster_rows(records: Sequence, study: Mapping) -> "tuple[list, list]":
    """依赖连通分量：共享 group_id / 规范化 premise / 有向 pair 指纹的记录合并为一个 cluster。

    raw group_id 原样保留在记录内（审计），cluster 只用于分用途与泄漏验证；合并只为防泄漏，
    不宣称每行独立，也不删除任何行（含缺标行）。
    """
    rows = list(records)
    seen: set = set()
    for record in rows:
        validate_record_for_dataset(record, study)
        if record["id"] in seen:
            raise ContractError("duplicate_record_id")
        seen.add(record["id"])
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    anchors: dict = {}
    for index, record in enumerate(rows):
        tokens = (("group", (record["dataset"], record["group_id"])),
                  ("premise", (record["dataset"], _normalize_text(record["premise"]))),
                  ("pair", (record["dataset"], _normalize_text(record["premise"]),
                            _normalize_text(record["hypothesis"]))))
        for token in tokens:
            if token in anchors:
                union(anchors[token], index)
            else:
                anchors[token] = index
    members: dict = {}
    for index in range(len(rows)):
        members.setdefault(find(index), []).append(index)
    clusters = []
    for root in sorted(members):
        indexes = sorted(members[root])
        fingerprints = sorted({_row_fingerprint(rows[index]) for index in indexes})
        digest = hashlib.sha256("\n".join(fingerprints).encode("utf-8")).hexdigest()
        clusters.append({
            "cluster_id": "cluster:" + digest[:16],
            "dataset": rows[indexes[0]]["dataset"],
            "members": indexes,
            "rows": len(indexes),
            "record_ids": sorted(rows[index]["id"] for index in indexes),
            "raw_units": sorted({rows[index]["group_id"] for index in indexes}),
        })
    clusters.sort(key=lambda item: (item["dataset"], item["cluster_id"]))
    return rows, clusters


def cluster_records(records: Sequence, study: Mapping | None = None) -> list:
    """依赖 cluster 清单（不含内部索引）：cluster_id/dataset/rows/record_ids/raw_units。"""
    study = load_study() if study is None else study
    _rows, clusters = _cluster_rows(records, study)
    return [{key: value for key, value in cluster.items() if key != "members"}
            for cluster in clusters]


def _rows_sha256(rows: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    for record in sorted(rows, key=lambda item: item["id"]):
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def partition_records(records: Sequence, study: Mapping | None = None) -> dict:
    """按依赖 cluster 切 diagnostic / sealed_acceptance（cluster 原子、含内容 SHA）。

    同一 cluster 的行恒同用途：共享 group_id、规范化 premise 或有向 pair 指纹的行不会被
    拆到两个用途（防同文本泄漏、不删行）。返回 `purposes`（record id → 用途）供评测阶段
    显式携带，`clusters` 清单保留成员与 raw group_id 供审计。
    """
    study = load_study() if study is None else study
    rows, clusters = _cluster_rows(records, study)
    buckets: dict[str, list] = {purpose: [] for purpose in PURPOSES}
    purposes: dict = {}
    units: dict = {}
    cluster_view = []
    for cluster in clusters:
        purpose = assign_purpose(study, cluster["dataset"], cluster["cluster_id"])
        for index in cluster["members"]:
            buckets[purpose].append(rows[index])
            purposes[rows[index]["id"]] = purpose
            units[rows[index]["id"]] = cluster["cluster_id"]
        cluster_view.append({"cluster_id": cluster["cluster_id"], "dataset": cluster["dataset"],
                             "purpose": purpose, "rows": cluster["rows"],
                             "record_ids": cluster["record_ids"],
                             "raw_units": cluster["raw_units"]})
    return {
        "diagnostic": buckets["diagnostic"],
        "sealed_acceptance": buckets["sealed_acceptance"],
        "purposes": purposes,
        "units": units,
        "clusters": cluster_view,
        "summary": _summarize(rows, cluster_view),
        "sha256": {purpose: _rows_sha256(buckets[purpose]) for purpose in PURPOSES},
    }


MANIFEST_FIELDS = ("diagnostic", "sealed_acceptance", "purposes", "units", "clusters",
                   "summary", "sha256")


def _summarize(rows: Sequence[Mapping], clusters: Sequence[Mapping]) -> dict:
    """manifest 汇总（partition_records 与 validate_manifest 共用同一实现）。"""
    by_purpose = {}
    for purpose in PURPOSES:
        members = [cluster for cluster in clusters if cluster["purpose"] == purpose]
        by_purpose[purpose] = {
            "rows": sum(cluster["rows"] for cluster in members),
            "clusters": len(members),
            "datasets": {name: sum(cluster["rows"] for cluster in members
                                   if cluster["dataset"] == name) for name in DATASETS},
        }
    largest = max(clusters, key=lambda cluster: cluster["rows"],
                  default={"cluster_id": "", "dataset": "", "rows": 0, "raw_units": []})
    warnings = []
    if largest["rows"] and len(rows) and largest["rows"] / len(rows) > 0.5:
        warnings.append("megacluster_majority_rows")
    if min(by_purpose[purpose]["clusters"] for purpose in PURPOSES) < 2:
        warnings.append("insufficient_cluster_units_for_holdout")
    return {
        "total": len(rows),
        "total_clusters": len(clusters),
        "by_purpose": by_purpose,
        "clusters_with_multiple_raw_units": sum(1 for cluster in clusters
                                                if len(cluster["raw_units"]) > 1),
        "largest_cluster": {"cluster_id": largest["cluster_id"], "dataset": largest["dataset"],
                            "rows": largest["rows"], "raw_units": largest["raw_units"][:16],
                            "raw_units_total": len(largest["raw_units"])},
        "warnings": warnings,
        "note": "cluster=依赖连通分量，合并只为防泄漏、不宣称每行独立也不删行；若留出不足"
                "（warnings 非空）须如实报告，必要时整集作已见压力诊断、独立验收留给人类数据源",
    }


def validate_manifest(manifest: Mapping, study: Mapping | None = None) -> dict:
    """冻结 manifest 的消费端执法（唯一入口）：重算 cluster/purposes/units/每桶实际成员/
    每桶内容 SHA/汇总，与声明逐项比对；任何漂移 fail loud（`manifest_*_mismatch`）。
    返回核验过的规范化 manifest，所有曝光/选择/计分/完成消费点只许使用它的数据。

    桶成员执法：每个桶的实际行集合必须恰好等于按 cluster 重算的该用途行集合——把 sealed 行
    放进 diagnostic 桶（或反向）即使 purposes/units/SHA 自洽也拒（`manifest_bucket_mismatch`）。
    """
    study = load_study() if study is None else study
    if type(manifest) is not dict or set(manifest) != set(MANIFEST_FIELDS):
        raise ContractError("invalid_manifest")
    if type(manifest["diagnostic"]) is not list or type(manifest["sealed_acceptance"]) is not list \
            or type(manifest["purposes"]) is not dict or type(manifest["units"]) is not dict \
            or type(manifest["clusters"]) is not list or type(manifest["summary"]) is not dict \
            or type(manifest["sha256"]) is not dict:
        raise ContractError("invalid_manifest")
    rows, clusters = _cluster_rows(manifest["diagnostic"] + manifest["sealed_acceptance"], study)
    recomputed_purposes: dict = {}
    recomputed_units: dict = {}
    cluster_view = []
    for cluster in clusters:
        purpose = assign_purpose(study, cluster["dataset"], cluster["cluster_id"])
        for index in cluster["members"]:
            recomputed_purposes[rows[index]["id"]] = purpose
            recomputed_units[rows[index]["id"]] = cluster["cluster_id"]
        cluster_view.append({"cluster_id": cluster["cluster_id"], "dataset": cluster["dataset"],
                             "purpose": purpose, "rows": cluster["rows"],
                             "record_ids": cluster["record_ids"],
                             "raw_units": cluster["raw_units"]})
    if manifest["purposes"] != recomputed_purposes:
        raise ContractError("manifest_purposes_mismatch")
    if manifest["units"] != recomputed_units:
        raise ContractError("manifest_units_mismatch")
    if manifest["clusters"] != cluster_view:
        raise ContractError("manifest_clusters_mismatch")
    for purpose in PURPOSES:
        actual = sorted(row["id"] for row in manifest[purpose])
        expected = sorted(record_id for record_id, assigned in recomputed_purposes.items()
                          if assigned == purpose)
        if actual != expected:
            raise ContractError("manifest_bucket_mismatch")
    for purpose in PURPOSES:
        if manifest["sha256"].get(purpose) != _rows_sha256(manifest[purpose]):
            raise ContractError("manifest_sha_mismatch")
    summary = _summarize(rows, cluster_view)
    if manifest["summary"] != summary:
        raise ContractError("manifest_counts_mismatch")
    return {
        "diagnostic": list(manifest["diagnostic"]),
        "sealed_acceptance": list(manifest["sealed_acceptance"]),
        "purposes": dict(recomputed_purposes),
        "units": dict(recomputed_units),
        "clusters": cluster_view,
        "summary": summary,
        "sha256": {purpose: _rows_sha256(manifest[purpose]) for purpose in PURPOSES},
    }


def manifest_scope(manifest: Mapping, *, view_id: str, phase: str,
                   study: Mapping | None = None) -> dict:
    """按冻结 manifest 计算视图×阶段的预定全行集合与显式记账（评测预定集的唯一来源）。

    `all_rows`：该数据集在该阶段的全部官方行；`gold_evidence_two_class`（ContractNLI oracle）：
    只有 gold ∈ {entailment, contradiction} 且有 gold 证据跨度的行入分母，另分列
    n_official / n_eligible / n_neutral_ineligible（真 neutral）/ n_span_unavailable；
    缺标按真实 `gold_status` 分成 `n_missing_gold` / `n_no_consensus_gold`（>0 才出现，
    绝不把 missing 记成 neutral），并做 n_official 守恒检查。不用空 premise 泄漏标签。
    """
    study = load_study() if study is None else study
    validated = validate_manifest(manifest, study)
    view = view_spec(study, view_id)
    if view["kind"] != "native_nli":
        raise ContractError("diagnostic_view_not_scorable")
    if phase not in PHASES:
        raise ContractError("invalid_phase")
    dataset = view["dataset"]
    rows = validated["diagnostic"] if phase == "diagnostic" else \
        validated["diagnostic"] + validated["sealed_acceptance"]
    official = [row for row in rows if row["dataset"] == dataset]
    rule = view.get("scope_rule", "all_rows")
    accounting = {"rule": rule, "n_official": len(official)}
    if rule == "gold_evidence_two_class":
        gold_in = tuple(view["eligibility"]["gold_in"])
        eligible = [row for row in official
                    if row["gold_label"] in gold_in and oracle_available(row)]
        neutral = [row for row in official if row["gold_label"] == "neutral"]
        span_unavailable = [row for row in official if row["gold_label"] in gold_in
                            and not oracle_available(row)]
        missing = [row for row in official if row["gold_label"] is None
                   and row["metadata"].get("gold_status") == "missing"]
        no_consensus = [row for row in official if row["gold_label"] is None
                        and row["metadata"].get("gold_status") == "no_consensus"]
        accounting.update({
            "n_eligible": len(eligible),
            "n_neutral_ineligible": len(neutral),
            "n_span_unavailable": len(span_unavailable),
        })
        if missing:
            accounting["n_missing_gold"] = len(missing)
        if no_consensus:
            accounting["n_no_consensus_gold"] = len(no_consensus)
        conserved = len(eligible) + len(neutral) + len(span_unavailable) + len(missing) \
            + len(no_consensus)
        if conserved != len(official):
            raise ContractError("oracle_accounting_inconsistent")
        ids = [row["id"] for row in eligible]
    else:
        accounting["n_eligible"] = len(official)
        ids = [row["id"] for row in official]
    ids = sorted(ids)
    return {
        "view_id": view_id, "dataset": dataset, "layer": view.get("layer", "standard"),
        "phase": phase, "ids": ids, "accounting": accounting,
        "record_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def select_for_phase(partitioned: Mapping, phase: str) -> list:
    """初诊只看 diagnostic；最终验收评完所有可用行（diagnostic+sealed_acceptance）。

    与所有曝光入口同机制：返回任何行之前先 `validate_manifest` 逐项复核；传未校验/被篡改
    manifest 一律 fail loud，绝不先返回行再由下游补验。
    """
    if phase not in PHASES:
        raise ContractError("invalid_phase")
    validated = validate_manifest(partitioned)
    if phase == "diagnostic":
        return list(validated["diagnostic"])
    return list(validated["diagnostic"]) + list(validated["sealed_acceptance"])


def _canonical_row(record: Mapping) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_leakage(partitioned: Mapping, study: Mapping | None = None) -> dict:
    """分区完整性复核 = `validate_manifest` 逐项执法（purposes/units/clusters/内容 SHA/计数，
    任何篡改 fail loud）+ 泄漏统计：同 premise/pair 由 cluster 合并至单一用途；raw group_id
    原样保留、不删行；合并只防泄漏、不宣称每行独立。"""
    study = load_study() if study is None else study
    validated = validate_manifest(partitioned, study)
    rows = validated["diagnostic"] + validated["sealed_acceptance"]
    by_id = {row["id"]: row for row in rows}
    premise_clusters: dict = {}
    pair_counts: dict = {}
    for cluster in validated["clusters"]:
        for record_id in cluster["record_ids"]:
            record = by_id[record_id]
            premise_key = (record["dataset"], _normalize_text(record["premise"]))
            pair_key = premise_key + (_normalize_text(record["hypothesis"]),)
            premise_clusters.setdefault(premise_key, set()).add(cluster["cluster_id"])
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    return {
        "status": "ok",
        "rows": len(rows),
        "clusters": len(validated["clusters"]),
        "clusters_with_multiple_raw_units":
            validated["summary"]["clusters_with_multiple_raw_units"],
        "premise_texts": len(premise_clusters),
        "premise_texts_in_multiple_clusters": sum(1 for bucket in premise_clusters.values()
                                                  if len(bucket) > 1),
        "pair_texts_in_multiple_records": sum(1 for count in pair_counts.values() if count > 1),
        "warnings": list(validated["summary"]["warnings"]),
        "note": "manifest 已逐项复核（purposes/units/clusters/内容 SHA/计数）；同一 premise/pair 由 "
                "cluster 合并至单一用途；raw group_id 原样保留；合并只防泄漏、不宣称每行独立也不删行",
    }


                                                                             
         
                                                                             

def view_spec(study: Mapping, view_id: str) -> dict:
    for view in study["views"]:
        if view["view_id"] == view_id:
            return view
    raise ContractError("unknown_view")


def view_label_space(view: Mapping) -> tuple:
    """视图的标签空间：原生任务取 GOLD_BY_TASK；条件任务（E/C oracle）取冻结条件子集。"""
    task = view["task"]
    if task in GOLD_BY_TASK:
        return GOLD_BY_TASK[task]
    if task in CONDITIONAL_TASKS:
        return tuple(CONDITIONAL_TASKS[task])
    raise ContractError("invalid_task")


def oracle_eligible(record: object) -> bool:
    """冻结的 ContractNLI oracle 资格：gold ∈ {entailment, contradiction} 且有 gold 证据跨度。
    neutral / 缺标 / 无跨度行不在该二类任务内（只在 n_official / n_neutral_ineligible /
    n_span_unavailable 记账中体现），绝不用空 premise 泄漏标签。"""
    validate_record(record)
    return record["gold_label"] in ORACLE_GOLD_IN and oracle_available(record)


def native_view_binding(view: Mapping, artifact: Mapping, *,
                        weights_path: object | None = None,
                        study: Mapping | None = None) -> dict:
    """正式 runner 绑定真实原生工件（M11 native reference）时的注册与完整性核验。

    - **注册比对**（fail loud）：工件 meta 投影的 schema/kind/类名集合/标签空间/主向/槽位/
      输入配方必须与冻结 `study.native_reference` 逐项一致——输入配方必须等于注册的
      `finetune.INPUT_RECIPE` 标识，不是任意非空字符串；`max_length` 对固定窗口视图必须等于
      冻结窗口（标准视图 `native_reference.max_tokens_standard`），`arm_native`（窗口 null）
      视图只要求正整数并如实报告。
    - **完整性核验分开报告**：给了 `weights_path` 就做文件 SHA256 比对（不符 fail loud
      `artifact_hash_mismatch`），报告 `artifact_integrity="verified_weights_sha256"`；
      没给只报 `not_verified_here`——元数据形状正确不等于权重文件已验证。
    - `runnable` 只表示注册比对通过；`study.available_in_round=false` 只是冻结稿登记状态，
      原生视图不得被静默长期不跑。
    """
    study = load_study() if study is None else study
    if view.get("kind") != "native_nli":
        raise ContractError("invalid_view")
    reference = study["native_reference"]
    required = {"artifact_id", "schema", "kind", "classes", "label_space", "input_recipe",
                "direction", "joint_roles", "max_length", "weights_sha256",
                "backbone_fingerprint"}
    if type(artifact) is not dict or set(artifact) != required:
        raise ContractError("invalid_artifact")
    _text(artifact["artifact_id"], "invalid_artifact", max_length=256)
    _text(artifact["backbone_fingerprint"], "invalid_artifact", max_length=256)
    sha = artifact["weights_sha256"]
    if type(sha) is not str or len(sha) != 64 \
            or any(character not in "0123456789abcdef" for character in sha):
        raise ContractError("invalid_artifact")
    if artifact["schema"] != reference["schema"] or artifact["kind"] != reference["kind"] \
            or artifact["label_space"] != reference["label_space"] \
            or artifact["input_recipe"] != reference["input_recipe"] \
            or artifact["direction"] != reference["direction"] \
            or artifact["joint_roles"] != reference["joint_roles"]:
        raise ContractError("artifact_registration_mismatch")
    classes = artifact["classes"]
    if type(classes) is not list or len(classes) != len(reference["classes"]) \
            or set(classes) != set(reference["classes"]):
        raise ContractError("invalid_artifact")
    max_length = artifact["max_length"]
    if type(max_length) is not int or max_length < 1:
        raise ContractError("invalid_artifact")
    window = view.get("window")
    if type(window) is dict and type(window.get("max_tokens")) is int:
        expected_tokens = window["max_tokens"]
    elif type(window) is dict:
        expected_tokens = None
    else:
        expected_tokens = reference["max_tokens_standard"]
    if expected_tokens is not None and max_length != expected_tokens:
        raise ContractError("artifact_window_mismatch")
    integrity = "not_verified_here"
    if weights_path is not None:
        path = Path(weights_path)
        if not path.is_file():
            raise ContractError("artifact_missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ContractError("artifact_hash_mismatch")
        integrity = "verified_weights_sha256"
    return {
        "runnable": True, "view_id": view["view_id"], "artifact_id": artifact["artifact_id"],
        "classes": list(classes), "direction": artifact["direction"],
        "input_recipe": artifact["input_recipe"], "label_space": artifact["label_space"],
        "max_length": max_length, "weights_sha256": sha,
        "metadata_check": "matched_registered_native_reference",
        "artifact_integrity": integrity,
        "note": "注册比对（schema/kind/类序/标签空间/主向/槽位/配方/窗口）已通过；"
                "artifact_integrity 单列，not_verified_here 时由加载器（load_nli_scorer 校验 "
                "weights_sha256/骨干指纹）在实际加载时核验；study.available_in_round 只是冻结稿状态",
    }


def diagnostic_request(record: object, *, direction: str = PRIMARY_DIRECTION) -> dict:
    """七类探针的学生输入适配：只取 premise/hypothesis 文本面，不借 trigger/evidence 槽。"""
    if direction not in DIRECTIONS:
        raise ContractError("invalid_direction")
    validate_record(record)
    premise = record["premise"]
    hypothesis = record["hypothesis"]
    if direction == "premise_as_candidate":
        candidate, existing = premise, hypothesis
    else:
        candidate, existing = hypothesis, premise
    return {
        "candidate": {"text": candidate, "trigger": ""},
        "existing": [{"text": existing, "trigger": ""}],
        "evidence": [],
    }


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann-Whitney 秩 AUC（带平均秩处理并列）；单边为空时返回 None。"""
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(sorted_scores):
        end = position
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[position]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        ranks[order[position:end + 1]] = average
        position = end + 1
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def external_contradiction_diagnostic(probs_by_direction: Mapping, gold_labels: Sequence,
                                      *, units: Sequence, strata: Sequence | None = None,
                                      study: Mapping | None = None,
                                      samples: int | None = None,
                                      confidence: float | None = None,
                                      seed: int | None = None) -> dict:
    """七类概率的任务不等价探针（外部矛盾诊断）；永不产出准确率/风险/放行/认证。

    两方向（主向预注册）都必需，报各 gold 下的输出分布与 conflict 信号的秩 AUC、
    方向翻转率；两向不得平均。strata 可选（如 HANS heuristic），只作描述性分层。
    """
    study = load_study() if study is None else study
    bootstrap_spec = study["metrics"]["bootstrap"]
    samples = int(bootstrap_spec["samples"]) if samples is None else samples
    confidence = float(bootstrap_spec["confidence"]) if confidence is None else confidence
    seed = int(bootstrap_spec["seed"]) if seed is None else seed
    if set(probs_by_direction) != set(DIRECTIONS):
        raise ContractError("invalid_probs")
    gold_list = [None if gold is None else str(gold) for gold in gold_labels]
    for gold in gold_list:
        if gold is not None and (gold in SEVEN_CLASSES or gold not in GOLD_LABELS):
            raise ContractError("invalid_gold")
    unit_list = [str(unit) for unit in units]
    if not (len(gold_list) == len(unit_list)):
        raise ContractError("invalid_units")
    n = len(unit_list)
    matrices = {}
    for direction in DIRECTIONS:
        matrix = np.asarray(probs_by_direction[direction], dtype=np.float64)
        if matrix.shape != (n, len(SEVEN_CLASSES)) or not np.isfinite(matrix).all() \
                or (matrix < 0).any() or (matrix > 1).any() \
                or (np.abs(matrix.sum(axis=1) - 1.0) > SIMPLEX_TOLERANCE).any():
            raise ContractError("invalid_probs")
        matrices[direction] = matrix
    strata_keys: list[str] = []
    strata_values: list[dict] = [{} for _ in range(n)]
    if strata is not None:
        if len(strata) != n:
            raise ContractError("invalid_units")
        for position, item in enumerate(strata):
            if type(item) is not dict:
                raise ContractError("invalid_units")
            strata_values[position] = {str(key): str(value) for key, value in item.items()}
            for key in strata_values[position]:
                if key not in strata_keys:
                    strata_keys.append(key)
    conflict_index = SEVEN_CLASSES.index("conflict")

    def distribution(index) -> dict:
        return {name: float(matrices[direction][index, position].mean())
                for position, name in enumerate(SEVEN_CLASSES)}

    result: dict = {
        "probe_kind": "task_incompatibility",
        "output_kind": "seven_class_probability_distribution",
        "task_equivalence": "none",
        "not_accuracy": True,
        "classes": list(SEVEN_CLASSES),
        "n": n,
        "n_with_gold": sum(1 for gold in gold_list if gold is not None),
        "n_missing_gold": sum(1 for gold in gold_list if gold is None),
        "units": len(set(unit_list)),
        "directions": {
            "primary": PRIMARY_DIRECTION,
            "swap": DIRECTIONS[1],
            "note": "两向都跑、报翻转率、不得平均；主向预注册为 hypothesis_as_candidate"
                    "（candidate=hypothesis 新候选、existing=premise 既有证据）",
        },
        "by_direction": {},
        "forbidden_label_mappings": [list(pair) for pair in FORBIDDEN_LABEL_MAPPINGS],
        "notes": [
            "七类与 NLI 任务不等价；输出只是七类概率分布，不是 NLI 预测",
            "不得称错误放行/风险/认证/放心；不得 map 到 NLI 标签",
        ],
    }
    all_index = list(range(n))
    for direction in DIRECTIONS:
        by_direction: dict = {
            "mean_probability_all": distribution(all_index),
            "mean_probability_by_gold": {},
        }
        for gold in list(GOLD_LABELS) + [None]:
            index = [position for position, value in enumerate(gold_list) if value == gold]
            if not index:
                continue
            by_direction["mean_probability_by_gold"][gold if gold is not None else "missing"] = \
                distribution(index)
        labeled = [position for position, gold in enumerate(gold_list) if gold is not None]
        positive = np.array([gold_list[position] == "contradiction" for position in labeled],
                            dtype=bool)
        auc = _rank_auc(matrices[direction][labeled, conflict_index], positive) \
            if 0 < positive.sum() < positive.size else None
        signal = {"signal_class": "conflict", "target": "gold == contradiction",
                  "rank_auc": auc, "n_positive": int(positive.sum()),
                  "n_negative": int(positive.size - positive.sum()),
                  "n_labeled": len(labeled), "n_missing_excluded": n - len(labeled),
                  "note": "监督 AUC/自助只吃有 gold 的行；缺标行留在总体分布与覆盖记账，"
                          "绝不当负例"}
        if len(labeled) >= 2 and 0 < positive.sum() < positive.size:
            labeled_units = [unit_list[position] for position in labeled]

            def metric(index, weights, direction=direction, labeled=tuple(labeled)):
                if len(index) == 0:
                    return None
                positions = [labeled[position] for position in index]
                return _rank_auc(matrices[direction][positions, conflict_index],
                                 np.array([gold_list[position] == "contradiction"
                                           for position in positions]))
            record = METRICS.cluster_bootstrap(labeled_units, metric, samples=samples,
                                               confidence=confidence, seed=seed,
                                               weighting="row")
            signal["bootstrap"] = record
        else:
            signal["bootstrap"] = None
        by_direction["conflict_signal"] = signal
        if strata_keys:
            by_stratum = {}
            for key in strata_keys:
                buckets: dict[str, list] = {}
                for position in all_index:
                    buckets.setdefault(strata_values[position].get(key, "missing"), []).append(position)
                by_stratum[key] = {str(value): distribution(index)
                                   for value, index in sorted(buckets.items())}
            by_direction["mean_probability_by_stratum"] = by_stratum
        result["by_direction"][direction] = by_direction
    primary_argmax = matrices[PRIMARY_DIRECTION].argmax(axis=1)
    swap_argmax = matrices[DIRECTIONS[1]].argmax(axis=1)
    result["flip_rate"] = {
        "primary": PRIMARY_DIRECTION,
        "swap": DIRECTIONS[1],
        "flips": int((primary_argmax != swap_argmax).sum()),
        "n": n,
        "rate": float((primary_argmax != swap_argmax).mean()) if n else None,
    }
    return result


                                                                             
                    
                                                                             

def _bucket_stats(rows: Sequence[dict]) -> dict:
    scored_gold = [row for row in rows if row["state"] == "scored" and row["gold"] is not None]
    correct = sum(1 for row in scored_gold if row["pred"] == row["gold"])
    return {
        "n_total": len(rows),
        "scored_with_gold": len(scored_gold),
        "correct": correct,
        "accuracy": {"value": correct / len(scored_gold) if scored_gold else None,
                     "numerator": correct, "denominator": len(scored_gold)},
        "truncated": sum(1 for row in rows if row["truncated"] is True),
        "truncation_unknown": sum(1 for row in rows if row["truncated"] is None),
    }


def claim_status(view: Mapping, truncated_count: int, *, evidence_count: int | None = None,
                 n_total: int | None = None) -> str:
    """全文档层只有「全部预定行都有显式截断证据且零截断」才可声明全文；缺证据一律降级为
    windowed_not_full（不以 absence 充当零截断）；oracle 层显式标识。"""
    layer = view.get("layer", "standard")
    if view["kind"] == "seven_class_external_diagnostic":
        return "task_incompatibility_probe"
    if layer == "full_document":
        if evidence_count is None or n_total is None:
            raise ContractError("missing_truncation")
        if truncated_count == 0 and evidence_count == n_total:
            return "full_document"
        return "windowed_not_full"
    if layer == "oracle_evidence":
        return "oracle_evidence"
    return "standard"


def evaluate_view(records: Sequence | None, results: Mapping, *, study: Mapping | None = None,
                  view_id: str, phase: str, manifest: Mapping | None = None,
                  purposes: Mapping | None = None, units: Mapping | None = None) -> dict:
    """原生视图报告：预定全行覆盖 + 自带分母 + 失败计数 + 截断证据 + 簇自助区间。

    唯一入口是 **manifest 绑定（权威）**：`manifest=partition_records(...)` 先经
    `validate_manifest` 逐项复核（purposes/units/clusters/每桶成员/内容 SHA/计数），预定全行
    集合由 `manifest_scope` 按视图与阶段给出（不是调用方自报子集）。`records` 可省略；给了
    就必须与 manifest 行逐字节一致（否则 `record_drift`）。**无 manifest 的 fragment 计分
    旁路已删除**：只给 `purposes`/`units` 一律 `manifest_required`（与 manifest 同给为
    `invalid_arguments`），不得先按自报集合曝光/计分再由下游补验。`phase=diagnostic` 时
    sealed 行/结果即 `phase_violation`；簇自助按 cluster 重采样。
    """
    study = load_study() if study is None else study
    view = view_spec(study, view_id)
    if view["kind"] != "native_nli":
        raise ContractError("diagnostic_view_not_scorable")
    if phase not in PHASES:
        raise ContractError("invalid_phase")
    if purposes is not None or units is not None:
        raise ContractError("manifest_required" if manifest is None else "invalid_arguments")
    if manifest is None:
        raise ContractError("manifest_required")
    dataset = view["dataset"]
    allowed = view_label_space(view)
    strata_keys = list(study["datasets"][dataset].get("stratify_by") or [])
    validated = validate_manifest(manifest, study)
    scope = manifest_scope(validated, view_id=view_id, phase=phase, study=study)
    by_id = {row["id"]: row for row in validated["diagnostic"] + validated["sealed_acceptance"]}
    rows_source = [by_id[record_id] for record_id in scope["ids"]]
    if records is not None:
        passed = list(records)
        for record in passed:
            if type(record) is not dict or type(record.get("id")) is not str:
                raise ContractError("invalid_record")
        passed_ids = sorted(record["id"] for record in passed)
        if passed_ids != scope["ids"]:
            if phase == "diagnostic" and passed_ids and all(
                    validated["purposes"].get(record_id) == "sealed_acceptance"
                    for record_id in passed_ids):
                raise ContractError("phase_violation")
            raise ContractError("manifest_scope_mismatch")
        for record in passed:
            if _canonical_row(record) != _canonical_row(by_id[record["id"]]):
                raise ContractError("record_drift")
    binding = "manifest"
    declared_purposes = validated["purposes"]
    declared_units = validated["units"]
    scope_ids = [record["id"] for record in rows_source]
    extra = set(results) - set(scope_ids)
    if extra:
        if phase == "diagnostic" and all(
                validated["purposes"].get(record_id) == "sealed_acceptance"
                for record_id in extra):
            raise ContractError("phase_violation")
        raise ContractError("unexpected_results")
    if set(scope_ids) - set(results):
        raise ContractError("missing_results")
    rows = []
    for record in rows_source:
        entry = results[record["id"]]
        if type(entry) is not dict or set(entry) - {"pred", "error", "truncated"}:
            raise ContractError("invalid_outcome")
        pred = entry.get("pred")
        error = entry.get("error")
        if error is not None and (type(error) is not str or not error):
            raise ContractError("invalid_outcome")
        if pred is not None:
            if error is not None:
                raise ContractError("invalid_outcome")
            if pred in SEVEN_CLASSES:
                raise ContractError("cross_task_prediction")
            if pred not in allowed:
                raise ContractError("invalid_prediction")
            state = "scored"
        else:
            state = "failed" if error is not None else "not_attempted"
        truncated = entry.get("truncated")
        if truncated is not None and type(truncated) is not bool:
            raise ContractError("invalid_outcome")
        purpose = declared_purposes.get(record["id"])
        if purpose not in PURPOSES:
            raise ContractError("missing_purpose")
        unit = declared_units.get(record["id"])
        if type(unit) is not str or not unit:
            raise ContractError("missing_unit")
        if phase == "diagnostic" and purpose == "sealed_acceptance":
            raise ContractError("phase_violation")
        metadata = record["metadata"]
        rows.append({
            "record_id": record["id"], "unit": unit, "purpose": purpose,
            "gold": record["gold_label"], "pred": pred, "error": error, "state": state,
            "truncated": truncated,
            "strata": {key: str(metadata.get(key, "unknown")) for key in strata_keys},
        })
    n_total = len(rows)
    scored = [row for row in rows if row["state"] == "scored"]
    failed = [row for row in rows if row["state"] == "failed"]
    not_attempted = [row for row in rows if row["state"] == "not_attempted"]
    with_gold = [row for row in rows if row["gold"] is not None]
    scored_gold = [row for row in scored if row["gold"] is not None]
    correct = [row for row in scored_gold if row["pred"] == row["gold"]]
    truncated = [row for row in rows if row["truncated"] is True]
    truncation_evidence = sum(1 for row in rows if row["truncated"] is not None)
    accuracy = {"value": len(correct) / len(scored_gold) if scored_gold else None,
                "numerator": len(correct), "denominator": len(scored_gold)}
    coverage = {"value": len(scored_gold) / len(with_gold) if with_gold else None,
                "numerator": len(scored_gold), "denominator": len(with_gold)}
    attempt_rate = {"value": (len(scored) + len(failed)) / n_total if n_total else None,
                    "numerator": len(scored) + len(failed), "denominator": n_total}
    failure_rate = {"value": len(failed) / n_total if n_total else None,
                    "numerator": len(failed), "denominator": n_total}
    labels = list(allowed)
    if scored_gold:
        y_true = np.array([labels.index(row["gold"]) for row in scored_gold])
        y_pred = np.array([labels.index(row["pred"]) for row in scored_gold])
        f1 = METRICS.macro_f1(y_true, y_pred, n_classes=len(labels))
    else:
        f1 = None
    per_class = {}
    for label in labels:
        support = sum(1 for row in scored_gold if row["gold"] == label)
        hit = sum(1 for row in scored_gold if row["gold"] == label and row["pred"] == label)
        per_class[label] = {"support": support, "correct": hit,
                            "recall": hit / support if support else None}
    bootstrap = {}
    if scored_gold:
        units = [row["unit"] for row in scored_gold]
        gold_values = np.array([row["gold"] for row in scored_gold], dtype=object)
        pred_values = np.array([row["pred"] for row in scored_gold], dtype=object)
        bootstrap_spec = study["metrics"]["bootstrap"]

        def metric(index, weights):
            if len(index) == 0:
                return None
            hit = (pred_values[index] == gold_values[index]).astype(np.float64)
            if weights is None:
                return float(hit.mean())
            return float(np.average(hit, weights=weights))

        for weighting in ("row", "group_equal"):
            bootstrap[weighting] = METRICS.cluster_bootstrap(
                units, metric, samples=int(bootstrap_spec["samples"]),
                confidence=float(bootstrap_spec["confidence"]), seed=int(bootstrap_spec["seed"]),
                weighting=weighting, numerator=len(correct),
                denominator=len(scored_gold))
    by_stratum = {}
    for key in strata_keys:
        buckets: dict[str, list] = {}
        for row in rows:
            buckets.setdefault(row["strata"][key], []).append(row)
        by_stratum[key] = {value: _bucket_stats(bucket)
                           for value, bucket in sorted(buckets.items())}
    by_truncation = {
        "untruncated": _bucket_stats([row for row in rows if row["truncated"] is False]),
        "truncated": _bucket_stats(truncated),
        "unknown": _bucket_stats([row for row in rows if row["truncated"] is None]),
    }
    by_purpose = None
    if phase == "final_acceptance":
        by_purpose = {purpose: _bucket_stats([row for row in rows if row["purpose"] == purpose])
                      for purpose in PURPOSES}
    report_scope = dict(scope)
    report_scope["binding"] = binding
    report_scope["expected_rows"] = len(scope["ids"])
    report = {
        "view_id": view_id, "dataset": dataset, "task": view["task"], "phase": phase,
        "kind": view["kind"], "layer": view.get("layer", "standard"),
        "n_total": n_total,
        "scope": report_scope,
        "counts": {
            "scored": len(scored), "failed": len(failed), "not_attempted": len(not_attempted),
            "with_gold": len(with_gold), "scored_with_gold": len(scored_gold),
            "without_gold": n_total - len(with_gold), "truncated": len(truncated),
            "units": len({row["unit"] for row in rows}),
        },
        "accuracy": accuracy, "macro_f1": {"value": f1, "n": len(scored_gold), "classes": labels},
        "per_class": per_class,
        "coverage": coverage, "attempt_rate": attempt_rate, "failure_rate": failure_rate,
        "truncation": {
            "count": len(truncated),
            "unknown": n_total - truncation_evidence,
            "evidence": {"explicit": truncation_evidence, "total": n_total},
            "rate": {"value": len(truncated) / n_total if n_total else None,
                     "numerator": len(truncated), "denominator": n_total},
            "buckets": by_truncation,
        },
        "by_truncation": by_truncation, "by_stratum": by_stratum, "by_purpose": by_purpose,
        "stratify_by": strata_keys,
        "bootstrap": bootstrap,
        "claim": claim_status(view, len(truncated), evidence_count=truncation_evidence,
                              n_total=n_total),
        "completion": {"not_attempted": len(not_attempted), "failed": len(failed),
                       "complete": not not_attempted and not failed},
        "denominator_note": "accuracy=correct/scored_with_gold；failed/not_attempted 保留在 "
                            "n_total 并单独报告，零分母记 null；不得删行改善成绩；"
                            "截断缺失证据只降级全文声明，不当作零截断",
    }
    return report


def assert_final_complete(report: Mapping | None = None, *, manifest: Mapping | None = None,
                          results: Mapping | None = None, study: Mapping | None = None) -> None:
    """最终完成回执：由**权威 manifest + 完整真实 results 重新派生**，再与报告逐字段比对。

    不接受报告内的 binding/complete/失败计数旗子作为证据：
    - `manifest`/`results` 必需（缺 → `manifest_required`/`results_required`），先
      `validate_manifest` 再按同一入口重算报告；
    - 报告必须是 `phase=final_acceptance`（否则 `not_final_phase`）且与重算报告
      **逐字段一致**（任何被编辑的旗子/计数/CI/scope → `report_inconsistent`）；
    - 重算的完成状态必须为“全部预定行已尝试且无失败”（否则 `incomplete_final`）。
    旧版只读报告字典的调用方式不再构成完成证明。
    """
    study = load_study() if study is None else study
    if type(report) is not dict:
        raise ContractError("invalid_report")
    if report.get("phase") != "final_acceptance":
        raise ContractError("not_final_phase")
    if manifest is None:
        raise ContractError("manifest_required")
    if results is None:
        raise ContractError("results_required")
    view_id = report.get("view_id")
    if type(view_id) is not str or not view_id:
        raise ContractError("invalid_report")
    validated = validate_manifest(manifest, study)
    internal = evaluate_view(None, results, study=study, view_id=view_id,
                             phase="final_acceptance", manifest=validated)
    if report != internal:
        raise ContractError("report_inconsistent")
    completion = internal["completion"] or {}
    if not completion.get("complete"):
        raise ContractError("incomplete_final")


                                                                             
             
                                                                             

def _validate_admission_report(view_report: Mapping, *, view_id: str, phase: str) -> Mapping:
    """准入报告的 fail-closed 结构与一致性校验：空/伪/不一致报告一律 `invalid_report`。"""
    if type(view_report) is not dict:
        raise ContractError("invalid_report")
    required = {"view_id", "phase", "task", "n_total", "scope", "counts", "accuracy",
                "bootstrap", "completion"}
    if not required <= set(view_report):
        raise ContractError("invalid_report")
    if view_report["view_id"] != view_id or view_report["phase"] != phase:
        raise ContractError("report_mismatch")
    if view_report["task"] not in TASKS and view_report["task"] not in CONDITIONAL_TASKS:
        raise ContractError("invalid_report")
    n_total = view_report["n_total"]
    if type(n_total) is not int or n_total < 0:
        raise ContractError("invalid_report")
    scope = view_report["scope"]
    if type(scope) is not dict:
        raise ContractError("invalid_report")
    if scope.get("binding") != "manifest":
        raise ContractError("unbound_report")
    if scope.get("expected_rows") != n_total \
            or type(scope.get("record_ids_sha256")) is not str \
            or len(scope["record_ids_sha256"]) != 64:
        raise ContractError("invalid_report")
    counts = view_report["counts"]
    keys = ("scored", "failed", "not_attempted", "with_gold", "scored_with_gold",
            "without_gold", "truncated", "units")
    if type(counts) is not dict or set(counts) != set(keys) \
            or any(type(counts[key]) is not int or counts[key] < 0 for key in keys):
        raise ContractError("invalid_report")
    if counts["scored"] + counts["failed"] + counts["not_attempted"] != n_total \
            or counts["with_gold"] + counts["without_gold"] != n_total \
            or counts["scored_with_gold"] > counts["scored"]:
        raise ContractError("invalid_report")
    accuracy = view_report["accuracy"]
    if type(accuracy) is not dict or set(accuracy) != {"value", "numerator", "denominator"} \
            or type(accuracy["numerator"]) is not int \
            or type(accuracy["denominator"]) is not int \
            or accuracy["numerator"] < 0 or accuracy["denominator"] < 0 \
            or accuracy["numerator"] > accuracy["denominator"] \
            or accuracy["denominator"] != counts["scored_with_gold"]:
        raise ContractError("invalid_report")
    if accuracy["denominator"] == 0:
        if accuracy["value"] is not None:
            raise ContractError("invalid_report")
    elif isinstance(accuracy["value"], bool) or not isinstance(accuracy["value"], (int, float)) \
            or abs(float(accuracy["value"])
                   - accuracy["numerator"] / accuracy["denominator"]) > 1e-9:
        raise ContractError("invalid_report")
    bootstrap = view_report["bootstrap"]
    if type(bootstrap) is not dict:
        raise ContractError("invalid_report")
    row = bootstrap.get("row")
    if row is None:
                                                                         
                      
        if bootstrap or accuracy["denominator"] != 0:
            raise ContractError("invalid_report")
    else:
        if type(row) is not dict or type(row.get("denominator")) is not int \
                or row["denominator"] != accuracy["denominator"]:
            raise ContractError("invalid_report")
        ci = row.get("ci")
        if ci is not None and (type(ci) is not list or len(ci) != 2
                               or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                      for value in ci)
                               or float(ci[0]) > float(ci[1])):
            raise ContractError("invalid_report")
    completion = view_report["completion"]
    if type(completion) is not dict or set(completion) != {"not_attempted", "failed", "complete"} \
            or completion["failed"] != counts["failed"] \
            or completion["not_attempted"] != counts["not_attempted"] \
            or completion["complete"] is not (counts["failed"] == 0
                                              and counts["not_attempted"] == 0):
        raise ContractError("invalid_report")
    return view_report


def _compute_baselines(baselines: Sequence, section: Mapping, *, task: str, labels: Sequence,
                       scope: Mapping, rows_by_id: Mapping) -> list:
    """冻结 baseline 的**可复算**描述性统计（唯一机制，不接受标量声明当证据）。

    请求只允许两种形态：
    - 注册名（str）：按冻结 `derivation` 计算——`most_frequent_gold`（majority_class：选择行上
      最高频 gold 的命中数/分母）或 `uniform_random_1_over_k`（chance：解析 1/类数；分母不整除
      类数时不伪造整数分子）；调用方给不了错数，因为它不提供数。
    - 实证基线（dict，仅 `per_record_predictions`）：`{"name", "predictions", "recipe"}`；
      predictions 必须逐题覆盖已验证选择行的全部 id、取值在视图标签空间内，在同一成员集上
      复算分子/分母；recipe 必须等于冻结 profile 的 recipe 标识；规则实现本身未经本协议核验
      （报告明示 `implementation_verified=false`）。
    任务不在 profile.label_spaces 内 → `baseline_label_space_mismatch`；未知名 → `unknown_baseline`；
    形态不符 → `invalid_baseline`。
    """
    profiles = section["baseline_profiles"]
    labeled_ids = [record_id for record_id in scope["ids"]
                   if rows_by_id[record_id]["gold_label"] is not None]
    denominator = len(labeled_ids)
    computed = []
    for item in baselines:
        name = item if type(item) is str else (item.get("name") if type(item) is dict else None)
        profile = profiles.get(name) if type(name) is str else None
        if profile is None:
            raise ContractError("unknown_baseline" if type(name) is str else "invalid_baseline")
        if task not in profile["label_spaces"]:
            raise ContractError("baseline_label_space_mismatch")
        if type(item) is str:
            derivation = profile["derivation"]
            if derivation == "most_frequent_gold":
                counts: dict = {}
                for record_id in labeled_ids:
                    gold = rows_by_id[record_id]["gold_label"]
                    counts[gold] = counts.get(gold, 0) + 1
                order = list(labels)
                if counts and denominator:
                    best = max(order, key=lambda label: (counts.get(label, 0),
                                                         -order.index(label)))
                    numerator = counts[best]
                    value = numerator / denominator
                else:
                    best, numerator, value = None, None, None
                computed.append({
                    "name": item, "derivation": derivation, "value": value,
                    "numerator": numerator, "denominator": denominator,
                    "majority_label": best, "definition": profile["recipe"],
                    "note": "由验证器在选择行上按冻结定义复算",
                })
            elif derivation == "uniform_random_1_over_k":
                classes = len(labels)
                computed.append({
                    "name": item, "derivation": derivation, "value": 1.0 / classes,
                    "classes": classes, "denominator": denominator,
                    "definition": profile["recipe"],
                    "note": "解析 1/类数，不是整数计数；分母不整除类数时不强行整数分子",
                })
            else:
                raise ContractError("invalid_baseline")
            continue
        if type(item) is not dict or set(item) != {"name", "predictions", "recipe"}:
            raise ContractError("invalid_baseline")
        if profile["derivation"] != "per_record_predictions":
            raise ContractError("invalid_baseline")
        predictions = item["predictions"]
        if type(predictions) is not dict or set(predictions) != set(scope["ids"]):
            raise ContractError("baseline_member_mismatch")
        for value in predictions.values():
            if value not in labels:
                raise ContractError("invalid_baseline")
        if item["recipe"] != profile["recipe"]:
            raise ContractError("invalid_baseline")
        correct = sum(1 for record_id in labeled_ids
                      if predictions[record_id] == rows_by_id[record_id]["gold_label"])
        computed.append({
            "name": item["name"], "derivation": profile["derivation"],
            "value": correct / denominator if denominator else None,
            "numerator": correct if denominator else None, "denominator": denominator,
            "predictions_complete": True, "recipe": item["recipe"],
            "implementation_verified": False, "definition": profile["recipe"],
            "note": "逐题预测由调用方提供并在验证成员集上复算；规则实现本身未经本协议核验",
        })
    if not computed:
        raise ContractError("invalid_baseline")
    return computed


def admission_check(study: Mapping | None, *, view_id: str, view_report: Mapping,
                    baselines: Sequence, counterexample_units: Sequence,
                    attribution_classes: Sequence, language_matched: bool,
                    phase: str, manifest: Mapping | None = None,
                    results: Mapping | None = None) -> dict:
    """准入**证据报告**：只报告证据完整性与描述性统计，不自动许可 poor/重训。

    - `view_report` 必须是结构化、内部一致的原生视图报告（空/伪 → `invalid_report`；scope 非
      manifest 绑定 → `unbound_report`）。
    - **任何带 manifest/results 的调用都先用权威输入重算报告并逐字段比对**（完整/不完整/零观测
      一致）：陈旧/伪造报告 → `report_inconsistent`；权威输入缺失其一 → `manifest_required` /
      `results_required`；两者都缺的裸“不完整”声明只作**未验证回显**
      （`unverified_caller_declaration`），完整声明无权威输入一律拒。
    - 不完整选择（failed/not_attempted > 0）→ `evidence_complete=false` + `incomplete_selection`：
      **失败选择只称技术失败，不归罪学生，也不生成 poor/重训许可**。
    - 零 scored_with_gold（全失败/全未尝试/全缺标/零 eligible）→ `evidence_complete=false` +
      `no_scored_observations` 的不足证据 descriptor（`comparison=null`、计数守恒，不当坏 payload）。
    - 完整观测下由真实失败行推导唯一 cluster 反例（每个反例必须是真实失败行）；基线一律由验证器
      按冻结 `derivation` 复算（majority_class/chance）或要求完整逐题预测（实证基线），不接受
      标量声明；比较保留预注册严格方向（CI 上界 < 最强匹配基线）但只作**描述性统计**。归因声明
      只做结构登记并标注未验证（`validated=false`），科学归因由父 + 独立复核按真实报告决策。
    """
    study = load_study() if study is None else study
    view = view_spec(study, view_id)
    if view["kind"] != "native_nli":
        raise ContractError("admission_requires_native_view")
    if phase not in PHASES:
        raise ContractError("invalid_phase")
    if type(language_matched) is not bool:
        raise ContractError("invalid_outcome")
    report = _validate_admission_report(view_report, view_id=view_id, phase=phase)
    section = study["admission"]
    declared = report["counts"]
    base = {
        "view_id": view_id, "phase": phase,
        "manual_evidence_review_required": True,
        "note": "本 API 只报告证据完整性与描述性统计，不自动授权 poor/重训——由父结合真实报告与"
                "独立复核决策；数据不完整只称技术失败，不归学生差",
    }
    if (manifest is None) != (results is None):
        raise ContractError("manifest_required" if manifest is None else "results_required")
    if manifest is None:
                                                       
        if not (declared["failed"] or declared["not_attempted"]):
            raise ContractError("manifest_required")
        return {**base,
                "selection": {"n_total": report["n_total"],
                              "scored_with_gold": declared["scored_with_gold"],
                              "failed": declared["failed"],
                              "not_attempted": declared["not_attempted"]},
                "counts": dict(declared),
                "evidence_complete": False,
                "evidence_source": "unverified_caller_declaration",
                "reasons": ["incomplete_selection", "unverified_caller_declaration"],
                "comparison": None}
    validated = validate_manifest(manifest, study)
    scope = manifest_scope(validated, view_id=view_id, phase=phase, study=study)
    if report["scope"].get("record_ids_sha256") != scope["record_ids_sha256"] \
            or report["n_total"] != len(scope["ids"]):
        raise ContractError("report_scope_mismatch")
    internal = evaluate_view(None, results, study=study, view_id=view_id, phase=phase,
                             manifest=validated)
    if report != internal:
        raise ContractError("report_inconsistent")
    counts = internal["counts"]
    selection = {"n_total": internal["n_total"], "scored_with_gold": counts["scored_with_gold"],
                 "failed": counts["failed"], "not_attempted": counts["not_attempted"]}
    result_base = {**base, "selection": selection, "counts": dict(counts)}
    if section.get("poor_requires_complete_selection") and (
            counts["failed"] or counts["not_attempted"]):
        result_base.update({"evidence_complete": False,
                            "reasons": ["incomplete_selection"], "comparison": None})
        return result_base
    if counts["scored_with_gold"] == 0:
                                                               
                                            
        result_base.update({"evidence_complete": False,
                            "reasons": ["no_scored_observations"], "comparison": None})
        return result_base
    rows_by_id = {row["id"]: row for row in validated["diagnostic"]
                  + validated["sealed_acceptance"]}
    failures = []
    for record_id in scope["ids"]:
        row = rows_by_id[record_id]
        if row["gold_label"] is None:
            continue
        entry = results[record_id]
        if entry.get("error") is not None or (entry.get("pred") is not None
                                              and entry["pred"] != row["gold_label"]):
            failures.append(record_id)
    if type(counterexample_units) not in (list, tuple) \
            or any(type(record_id) is not str for record_id in counterexample_units):
        raise ContractError("invalid_counterexamples")
    examples = list(counterexample_units)
    if len(set(examples)) != len(examples):
        raise ContractError("invalid_counterexamples")
    for record_id in examples:
        if record_id not in failures:
            raise ContractError("counterexample_not_a_failure")
    counterexample_clusters = sorted({validated["units"][record_id] for record_id in examples})
    minimum = int(section["min_counterexample_units"])
    counterexamples_ok = len(counterexample_clusters) >= minimum
    cases = _compute_baselines(baselines, section, task=report["task"],
                               labels=view_label_space(view), scope=scope, rows_by_id=rows_by_id)
    ci = (internal["bootstrap"].get("row") or {}).get("ci")
    ci_upper = float(ci[1]) if ci and ci[1] is not None else None
    numeric = [item for item in cases if item.get("value") is not None]
    strongest = max(numeric, key=lambda item: item["value"]) if numeric else None
    ci_upper_below_strongest = (ci_upper is not None and strongest is not None
                                and ci_upper < float(strongest["value"]))
    attribution = section["attribution_classes"]
    claims = []
    for item in attribution_classes:
        if type(item) is not dict or set(item) != {"class", "evidence", "ref"} \
                or item["class"] not in attribution:
            raise ContractError("invalid_attribution")
        _text(item["evidence"], "invalid_attribution", max_length=2000)
        _text(item["ref"], "invalid_attribution", max_length=512)
        claims.append({"class": item["class"], "evidence": item["evidence"], "ref": item["ref"],
                       "validated": False,
                       "strength_on_record": attribution[item["class"]].get("strength")})
    reasons = []
    if not counterexamples_ok:
        reasons.append("insufficient_pre_registered_counterexample_clusters")
    if not ci_upper_below_strongest:
        reasons.append("ci_upper_not_below_strongest_registered_baseline")
    if not language_matched:
        reasons.append("no_language_matched_evidence")
    if any(claim["class"] == "supervision_unfair_gap" for claim in claims):
        reasons.append("supervision_unfair_gap_is_context_only_never_strong_attribution")
    if any(claim["strength_on_record"] == "admissible" for claim in claims):
        reasons.append("attribution_claims_unvalidated_manual_review_required")
    result = dict(result_base)
    result.update({
        "evidence_complete": True,
        "reasons": reasons,
        "comparison": {
            "rule": "预注册严格方向：accuracy 簇自助 CI 上界 < 最强匹配基线点估计"
                    "（描述性统计，不自动许可）",
            "accuracy": dict(internal["accuracy"]),
            "ci": ci,
            "ci_upper": ci_upper,
            "baselines": cases,
            "strongest_baseline": ({"name": strongest["name"], "value": strongest["value"]}
                                   if strongest else None),
            "ci_upper_below_strongest": ci_upper_below_strongest,
            "counterexample_clusters": counterexample_clusters,
            "min_counterexample_clusters": minimum,
            "counterexamples_complete": counterexamples_ok,
            "language_matched": language_matched,
            "attribution_claims": claims,
            "descriptive_only": True,
        },
    })
    return result
