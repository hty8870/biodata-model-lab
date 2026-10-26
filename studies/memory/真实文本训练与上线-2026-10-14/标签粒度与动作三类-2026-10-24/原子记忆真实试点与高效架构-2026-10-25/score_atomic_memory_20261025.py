                      
"""原子记忆真实试点：独立评分器（repo 外研究，repo 只读）。

本脚本只读 `data/` 下 6 个冻结输入，严格校验后计算 direct-kept 与 loop-visible
两套指标、A1/A2 预注册门、成本与 A2 父来源分解，并写 `data/results.json`。
不调用任何外部模型，不 import repo 代码，不读写秘密，不创建除 results.json 外的文件。

命令：
  check   完全离线：内置合成 raw/case/annotation + 全部指标/门正反自测；失败 exit 2。
  score   只读输入、严格校验、写 data/results.json 并打印摘要；输入缺失即 exit 2。

输入契约以设计 v2 §5–§8、§12 与采集器实际产物为准：
  - cases-v1.jsonl：schema=atomic-pilot-case-v1；gold/forbidden 为冻结独立参考。
  - raw-{P0,A1,A2}.jsonl：parsed candidates、kept_status、usage、elapsed、attempt 等；
    按 (case_id, arm) last-write-wins。A2 候选必须带父 P0 candidate/内容/attempt/闸状态。
  - call-ledger.jsonl：逐 attempt 的成本真源；按 case+arm 汇总所有 attempt。
  - annotations-v1.jsonl：两阶段完成且已 adjudicate 的盲审 atom 级标注；用 candidate
    集合精确匹配唯一 raw output set（空集用采集器 blind_id 哈希公式消歧，无需映射文件）。

字段别名（仅序列化边界，不改变语义）：parsed_candidates|candidates；
elapsed_s|elapsed_seconds；kept_status=kept|dropped_injection|dropped_evidence 与
用户约定的 kept|injection|evidence。

两个真实采集边界：P0 无候选时采集器不运行 A2（可能写 call.attempted=false 的
valid_empty 占位，也可能完全没有行）；评分按显式空输出计，不要求 ledger，也不伪造
候选或语义判断。标注 atom 允许可选 notes；若只填 fully_covered 未填 core，按
full⇒core 语义并入，不拒绝。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CASES_PATH = DATA / "cases-v1.jsonl"
RAW_PATHS = {arm: DATA / f"raw-{arm}.jsonl" for arm in ("P0", "A1", "A2")}
LEDGER_PATH = DATA / "call-ledger.jsonl"
ANNOTATIONS_PATH = DATA / "annotations-v1.jsonl"
RESULTS_PATH = DATA / "results.json"

ARMS = ("P0", "A1", "A2")
SPLITS = ("dev", "confirmation")
MAX_ITEMS = 8
MAX_CONVERSATIONS = 12
MAX_CALLS = 380
MAX_TOTAL_TOKENS = 1_200_000
SCHEMA_CASE = "atomic-pilot-case-v1"
SCHEMA_RESULT = "atomic-pilot-score-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CASE_KEYS = frozenset({"schema", "case_id", "family", "cluster_id", "split",
                       "conversations", "forbidden_inferences", "one_off_hints",
                       "author_notes", "gold_obligations"})
GOLD_KEYS = frozenset({"gold_id", "kind", "tier", "meaning", "evidence_conv_ids",
                       "evidence_spans", "context_requirements"})
SPAN_KEYS = frozenset({"conv_id", "text"})
CONV_KEYS = frozenset({"query", "chat"})
GOLD_KINDS = frozenset({"rule", "memory"})
GOLD_TIERS = frozenset({"explicit", "implicit", "not_applicable"})
SEVERITIES = frozenset({"severe", "major", "minor"})
KEPT_STATUS = frozenset({"kept", "injection", "evidence"})
KEPT_ALIASES = {"kept": "kept", "injection": "injection", "evidence": "evidence",
                "dropped_injection": "injection", "dropped_evidence": "evidence"}
STATUS_VALUES = frozenset({"ok", "partial_gated", "all_gated", "valid_empty",
                           "empty_after_parse", "invalid_json", "llm_failure",
                           "length", "content_filter"})
INVALID_OUTPUT = frozenset({"invalid_json", "llm_failure", "length", "content_filter"})
EMPTY_STATUSES = frozenset({"valid_empty", "empty_after_parse", "invalid_json",
                            "llm_failure", "length", "content_filter"})
CONTEXT_ERRORS = frozenset({"scope", "trigger", "time", "polarity", "condition", "other"})
ENTAILMENTS = frozenset({"supported", "partial", "unsupported"})
ATOM_KEYS = frozenset({"candidate_id", "entailment", "persistence_appropriate", "non_atomic",
                       "over_split", "context_errors", "matched_forbidden_ids",
                       "duplicate_group", "core_entailed_gold_ids",
                       "fully_covered_gold_ids", "alternative_partition"})
ATOM_OPTIONAL = frozenset({"notes"})                     
ANNOTATION_KEYS = frozenset({"blind_id", "stage1_complete", "stage2_complete",
                             "adjudicated", "atoms"})
ANNOTATION_OPTIONAL = frozenset({"case_id", "split"})
LINEAGE_KEYS = ("parent_p0_candidate_id", "parent_content_sha",
                "parent_p0_attempt_sha", "parent_gate_status")
ITEM_KEYS = ("text", "summary", "evidence", "tier", "trigger")
RAW_REQUIRED = frozenset({"case_id", "split", "arm", "attempt", "status", "usage",
                          "finish_reason", "raw_item_count"})
LEDGER_REQUIRED = frozenset({"case_id", "arm", "attempt", "usage"})

                          
THRESHOLDS = {
    "coverage": 0.90,
    "faithfulness": 0.95,
    "complete_set": 0.60,
    "non_atomic_obligation": 0.10,
    "non_atomic_relative_reduction": 0.50,
    "coverage_drop_pp": 2.0,
    "unsupported_increment_pp": 1.0,
    "duplicate_atom": 0.02,
    "over_split": 0.01,
    "a2_coverage_gain_pp": 3.0,
    "a2_non_atomic_relative_reduction": 0.25,
}


class ScoreError(Exception):
    """带机器码的可报告错误；消息不包含模型原文、密钥或端点。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def require(condition: Any, code: str, message: str) -> None:
    if not condition:
        raise ScoreError(code, message)


                                                


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def ratio(num: int, den: int) -> dict:
    return {"num": int(num), "den": int(den),
            "value": (float(num) / float(den)) if den else None}


def percentile(values: Sequence[float], q: float) -> float | None:
    """线性插值分位数（numpy 默认口径）；空序列返回 None。"""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def value_of(node: Any, *keys: str, default: Any = None) -> Any:
    current = node
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def candidate_view(item: Mapping[str, Any]) -> dict:
    row = {"text": str(item.get("text") or ""),
           "summary": str(item.get("summary") or ""),
           "evidence": list(item.get("evidence") or [])}
    if "tier" in item:
        row["tier"] = item["tier"]
    if "trigger" in item:
        row["trigger"] = item["trigger"]
    return row


def item_from_candidate(item: Mapping[str, Any]) -> dict:
    return {key: item[key] for key in ITEM_KEYS if key in item}


def content_sha(items: Iterable[Mapping[str, Any]]) -> str:
    return sha256_text(canon([candidate_view(item_from_candidate(item)) for item in items]))


def candidate_data_view(candidate: "CandidateData") -> dict:
    row = {"text": candidate.text, "summary": candidate.summary,
           "evidence": list(candidate.evidence)}
    if candidate.tier is not None:
        row["tier"] = candidate.tier
    if candidate.trigger is not None:
        row["trigger"] = candidate.trigger
    return row


def candidate_data_content_sha(candidate: "CandidateData") -> str:
    return sha256_text(canon([candidate_data_view(candidate)]))


def blind_id_of(case_id: str, arm: str, attempt: int, content_sha256: str) -> str:
    payload = f"{case_id}|{arm}|{attempt}|{content_sha256}"
    return sha256_text(payload)[:16]


def read_jsonl_file(path: Path) -> tuple[list[dict], dict]:
    if not path.exists():
        raise ScoreError("input_missing", f"输入不存在：{path.name}（{path}）")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ScoreError("input_unreadable", f"无法读取 {path.name}：{type(exc).__name__}") from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScoreError("input_encoding", f"{path.name} 不是 UTF-8：{exc.reason}") from None
    rows: list[dict] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoreError("jsonl_invalid", f"{path.name}:{index} 不是合法 JSON：{exc.msg}") from None
        require(isinstance(row, dict), "jsonl_invalid", f"{path.name}:{index} 不是 JSON 对象")
        rows.append(row)
    meta = {"sha256": sha256_bytes(data), "bytes": len(data), "rows": len(rows)}
    return rows, meta


                                                     


@dataclass
class GoldData:
    gold_id: str
    kind: str
    tier: str
    meaning: str
    context_requirements: tuple[str, ...]

    @property
    def has_context(self) -> bool:
        return bool(self.context_requirements)


@dataclass
class CaseData:
    case_id: str
    family: str
    cluster_id: str
    split: str
    gold: list[GoldData]
    gold_by_id: dict[str, GoldData]
    forbidden: dict[str, str]
    context_gold_ids: set[str]


@dataclass
class CandidateData:
    candidate_id: str
    text: str
    summary: str
    evidence: tuple[str, ...]
    tier: str | None
    trigger: str | None
    kept_status: str
    parent_p0_candidate_id: str | None = None
    parent_content_sha: str | None = None
    parent_p0_attempt_sha: str | None = None
    parent_gate_status: str | None = None


@dataclass
class RawData:
    case_id: str
    split: str
    arm: str
    attempt: int
    status: str
    usage: dict
    elapsed: float
    finish_reason: str | None
    raw_item_count: int
    candidates: list[CandidateData]
    candidate_by_id: dict[str, CandidateData]
    content_sha256: str
    attempt_sha256: str
    blind_id: str
    cap_applied: bool
    at_cap: bool
    p0_attempt_sha256: str | None = None
    p0_content_sha256: str | None = None
    not_run: bool = False
    not_run_reason: str | None = None
    superseded: int = 0


def _alternative_kind_accepts(kind: str) -> bool:
    normalized = kind.strip().lower()
    accepted = ("acceptable_compound", "reasonable_alternative",
                "可接受复合", "合理替代划分")
    rejected = ("reference_omission", "参考遗漏")
    if any(token in normalized for token in rejected):
        return False
    return any(token in normalized for token in accepted)


@dataclass
class AtomData:
    candidate_id: str
    entailment: str
    persistence_appropriate: bool
    non_atomic: bool
    over_split: bool
    context_errors: tuple[str, ...]
    matched_forbidden_ids: tuple[str, ...]
    duplicate_group: Any
    core_entailed_gold_ids: tuple[str, ...]
    fully_covered_gold_ids: tuple[str, ...]
    alternative_partition: Any

    @property
    def alternative_accepted(self) -> bool:
        """只有明确标注“可接受复合项/合理替代划分”才豁免 non-atomic 完整集失败。

        `reference_omission` 是参考遗漏建议，不是对本候选复合形态的接受；不能拿它
        给 non-atomic 洗白。自由文本若含可接受语义也接受，其余为 false。
        """
        value = self.alternative_partition
        if value is None or value is False:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            kind = value.get("type")
            if not isinstance(kind, str):
                return False
            return _alternative_kind_accepts(kind)
        if isinstance(value, str):
            return _alternative_kind_accepts(value)
        return False


@dataclass
class AnnotationData:
    blind_id: str
    atoms: list[AtomData]
    atom_by_id: dict[str, AtomData]


@dataclass
class LedgerData:
    case_id: str
    arm: str
    attempt: int
    split: str
    usage: dict
    elapsed: float


def validate_usage(value: Any, where: str) -> dict:
    require(isinstance(value, dict), "schema_usage", f"{where}.usage 须为对象")
    out: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(key in value, "schema_usage", f"{where}.usage 缺 {key}")
        token = value[key]
        require(token is None or (is_int(token) and 0 <= token <= 10 ** 9),
                "schema_usage", f"{where}.usage.{key} 须为 null 或 0..1e9 整数")
        out[key] = token
    return out


def validate_elapsed(row: Mapping[str, Any], where: str) -> float:
    present = [key for key in ("elapsed_s", "elapsed_seconds") if key in row]
    require(len(present) == 1, "schema_elapsed",
            f"{where}: elapsed_s / elapsed_seconds 必须且只能出现一个（实际 {present}）")
    value = row[present[0]]
    require(is_number(value) and float(value) >= 0,
            "schema_elapsed", f"{where}.{present[0]} 须为非负数")
    return float(value)


def validate_cases(rows: Sequence[dict]) -> list[CaseData]:
    require(rows, "cases_empty", "cases-v1.jsonl 为空")
    cases: list[CaseData] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"cases[{index}]"
        missing = sorted(CASE_KEYS - set(row))
        extra = sorted(set(row) - CASE_KEYS)
        require(not missing and not extra, "schema_case",
                f"{where}: 字段不符（缺 {missing}，多 {extra}）")
        require(row.get("schema") == SCHEMA_CASE, "schema_case",
                f"{where}.schema 须为 {SCHEMA_CASE}")
        case_id = row.get("case_id")
        require(is_nonempty_str(case_id), "schema_case", f"{where}.case_id 须为非空字符串")
        require(case_id not in seen, "schema_case", f"{where}: case_id 重复 {case_id}")
        seen.add(case_id)
        family = row.get("family")
        cluster_id = row.get("cluster_id")
        require(is_nonempty_str(family), "schema_case", f"{where}.family 须为非空字符串")
        require(is_nonempty_str(cluster_id), "schema_case",
                f"{where}.cluster_id 须为非空字符串")
        split = row.get("split")
        require(split in SPLITS, "schema_case", f"{where}.split 须为 {list(SPLITS)}")
        require(isinstance(row.get("author_notes"), str), "schema_case",
                f"{where}.author_notes 须为字符串")

        conversations = row.get("conversations")
        require(isinstance(conversations, list) and 1 <= len(conversations) <= MAX_CONVERSATIONS,
                "schema_case", f"{where}.conversations 须为 1..{MAX_CONVERSATIONS} 段")
        for conv_index, conv in enumerate(conversations):
            spot = f"{where}.conversations[{conv_index}]"
            require(isinstance(conv, dict) and set(conv) <= CONV_KEYS, "schema_case",
                    f"{spot}: 只允许 query/chat 字段")
            require(is_nonempty_str(conv.get("query")), "schema_case",
                    f"{spot}.query 须为非空字符串")
            require(conv.get("chat", []) == [], "schema_case",
                    f"{spot}.chat 必须为空（设计 §2 query-only）")
        conv_count = len(conversations)

        forbidden_rows = row.get("forbidden_inferences")
        require(isinstance(forbidden_rows, list), "schema_case",
                f"{where}.forbidden_inferences 须为数组")
        forbidden: dict[str, str] = {}
        for forbid_index, forbid in enumerate(forbidden_rows):
            spot = f"{where}.forbidden_inferences[{forbid_index}]"
            require(isinstance(forbid, dict), "schema_case", f"{spot} 须为对象")
            fid = forbid.get("forbidden_id")
            require(is_nonempty_str(fid), "schema_case", f"{spot}.forbidden_id 须为非空字符串")
            require(fid not in forbidden, "schema_case", f"{spot}.forbidden_id 重复 {fid}")
            require(is_nonempty_str(forbid.get("description")), "schema_case",
                    f"{spot}.description 须为非空字符串")
            severity = forbid.get("severity")
            require(severity in SEVERITIES, "schema_case",
                    f"{spot}.severity 须为 {sorted(SEVERITIES)} 之一")
            forbidden[fid] = severity

        one_off = row.get("one_off_hints")
        require(isinstance(one_off, list) and all(is_nonempty_str(x) for x in one_off),
                "schema_case", f"{where}.one_off_hints 须为非空字符串数组")
        require(len(set(one_off)) == len(one_off), "schema_case",
                f"{where}.one_off_hints 有重复")

        gold_rows = row.get("gold_obligations")
        require(isinstance(gold_rows, list), "schema_case",
                f"{where}.gold_obligations 须为数组")
        gold: list[GoldData] = []
        gold_by_id: dict[str, GoldData] = {}
        for gold_index, item in enumerate(gold_rows):
            spot = f"{where}.gold_obligations[{gold_index}]"
            require(isinstance(item, dict), "schema_gold", f"{spot} 须为对象")
            missing = sorted(GOLD_KEYS - set(item))
            extra = sorted(set(item) - GOLD_KEYS)
            require(not missing and not extra, "schema_gold",
                    f"{spot}: 字段不符（缺 {missing}，多 {extra}）")
            gold_id = item.get("gold_id")
            require(is_nonempty_str(gold_id), "schema_gold",
                    f"{spot}.gold_id 须为非空字符串")
            require(gold_id not in gold_by_id, "schema_gold",
                    f"{spot}.gold_id 重复 {gold_id}")
            kind = item.get("kind")
            tier = item.get("tier")
            require(kind in GOLD_KINDS, "schema_gold",
                    f"{spot}.kind 须为 {sorted(GOLD_KINDS)} 之一")
            require(tier in GOLD_TIERS, "schema_gold",
                    f"{spot}.tier 须为 {sorted(GOLD_TIERS)} 之一")
            if kind == "rule":
                require(tier in {"explicit", "implicit"}, "schema_gold",
                        f"{spot}: rule 的 tier 须为 explicit|implicit")
            else:
                require(tier == "not_applicable", "schema_gold",
                        f"{spot}: memory 的 tier 须为 not_applicable")
            meaning = item.get("meaning")
            require(is_nonempty_str(meaning), "schema_gold",
                    f"{spot}.meaning 须为非空字符串")
            conv_ids = item.get("evidence_conv_ids")
            require(isinstance(conv_ids, list) and conv_ids
                    and all(is_int(x) and 0 <= x < conv_count for x in conv_ids),
                    "schema_gold", f"{spot}.evidence_conv_ids 须为非空合法整数数组")
            require(len(set(conv_ids)) == len(conv_ids), "schema_gold",
                    f"{spot}.evidence_conv_ids 有重复")
            spans = item.get("evidence_spans")
            require(isinstance(spans, list) and spans, "schema_gold",
                    f"{spot}.evidence_spans 须为非空数组")
            span_ids: set[int] = set()
            for span_index, span in enumerate(spans):
                span_spot = f"{spot}.evidence_spans[{span_index}]"
                require(isinstance(span, dict) and set(span) == SPAN_KEYS, "schema_gold",
                        f"{span_spot}: 字段须恰为 {sorted(SPAN_KEYS)}")
                conv_id = span.get("conv_id")
                require(is_int(conv_id) and 0 <= conv_id < conv_count, "schema_gold",
                        f"{span_spot}.conv_id 非法")
                require(is_nonempty_str(span.get("text")), "schema_gold",
                        f"{span_spot}.text 须为非空字符串")
                span_ids.add(conv_id)
            require(span_ids == set(conv_ids), "schema_gold",
                    f"{spot}: evidence_conv_ids 与 evidence_spans 的 conv_id 不一致")
            context_raw = item.get("context_requirements")
            if isinstance(context_raw, str):
                context = (context_raw,) if context_raw.strip() else ()
            else:
                require(isinstance(context_raw, list)
                        and all(is_nonempty_str(x) for x in context_raw), "schema_gold",
                        f"{spot}.context_requirements 须为字符串或非空字符串数组")
                context = tuple(context_raw)
            gold_item = GoldData(gold_id=gold_id, kind=kind, tier=tier,
                                 meaning=meaning, context_requirements=context)
            gold.append(gold_item)
            gold_by_id[gold_id] = gold_item

        cases.append(CaseData(case_id=case_id, family=family, cluster_id=cluster_id,
                              split=split, gold=gold, gold_by_id=gold_by_id,
                              forbidden=forbidden,
                              context_gold_ids={g.gold_id for g in gold if g.has_context}))

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        cluster_splits[case.cluster_id].add(case.split)
    crossing = sorted(cid for cid, splits in cluster_splits.items() if len(splits) > 1)
    require(not crossing, "schema_cluster_split",
            f"cluster 跨 dev/confirmation：{crossing}")
    return cases


def normalize_kept(value: Any, where: str) -> str:
    require(isinstance(value, str) and value in KEPT_ALIASES, "schema_kept_status",
            f"{where}.kept_status 须为 kept|dropped_injection|dropped_evidence"
            f"（或 kept|injection|evidence）")
    return KEPT_ALIASES[value]


def normalize_parent_gate(value: Any, where: str) -> str:
    require(isinstance(value, str) and value in set(KEPT_ALIASES) | {"unknown"},
            "schema_lineage", f"{where}.parent_gate_status 非法：{value!r}")
    return "unknown" if value == "unknown" else KEPT_ALIASES[value]


def validate_candidate(row: Any, arm: str, where: str) -> CandidateData:
    require(isinstance(row, dict), "schema_candidate", f"{where} 须为对象")
    for key in ("candidate_id", "text", "summary", "evidence", "kept_status"):
        require(key in row, "schema_candidate", f"{where} 缺字段 {key}")
    candidate_id = row.get("candidate_id")
    require(is_nonempty_str(candidate_id), "schema_candidate",
            f"{where}.candidate_id 须为非空字符串")
    text = row.get("text")
    require(is_nonempty_str(text), "schema_candidate", f"{where}.text 须为非空字符串")
    summary = row.get("summary")
    require(isinstance(summary, str), "schema_candidate", f"{where}.summary 须为字符串")
    evidence = row.get("evidence")
    require(isinstance(evidence, list) and all(isinstance(x, str) for x in evidence),
            "schema_candidate", f"{where}.evidence 须为字符串数组")
    kept_status = normalize_kept(row.get("kept_status"), where)
    tier = row.get("tier")
    if "tier" in row:
        require(tier in {"explicit", "implicit"}, "schema_candidate",
                f"{where}.tier 须为 explicit|implicit（memory 应省略该键）")
    trigger = row.get("trigger")
    if "trigger" in row:
        require(is_nonempty_str(trigger), "schema_candidate",
                f"{where}.trigger 须为非空字符串（无触发条件应省略该键）")
    lineage: dict[str, Any] = {}
    if arm == "A2":
        for key in LINEAGE_KEYS:
            require(key in row, "schema_lineage", f"{where} 为 A2 候选但缺 {key}")
        parent_id = row.get("parent_p0_candidate_id")
        require(is_nonempty_str(parent_id), "schema_lineage",
                f"{where}.parent_p0_candidate_id 须为非空字符串")
        parent_content = row.get("parent_content_sha")
        parent_attempt = row.get("parent_p0_attempt_sha")
        parent_gate = normalize_parent_gate(row.get("parent_gate_status"), where)
        if parent_id == "unknown":
            require(parent_content is None and parent_attempt is None
                    and parent_gate == "unknown", "schema_lineage",
                    f"{where}: parent=unknown 时 content/attempt 须为 null 且 gate=unknown")
        else:
            require(is_hex64(parent_content), "schema_lineage",
                    f"{where}.parent_content_sha 须为 64 位小写十六进制")
            require(is_hex64(parent_attempt), "schema_lineage",
                    f"{where}.parent_p0_attempt_sha 须为 64 位小写十六进制")
            require(parent_gate in KEPT_STATUS, "schema_lineage",
                    f"{where}.parent_gate_status 须为 kept|injection|evidence")
        lineage = {"parent_p0_candidate_id": parent_id,
                   "parent_content_sha": parent_content,
                   "parent_p0_attempt_sha": parent_attempt,
                   "parent_gate_status": parent_gate}
    else:
        for key in LINEAGE_KEYS:
            if key in row:
                require(row[key] is None, "schema_lineage",
                        f"{where}: 非 A2 候选不得携带非空 {key}")
    return CandidateData(candidate_id=candidate_id, text=text, summary=summary,
                         evidence=tuple(evidence), tier=tier, trigger=trigger,
                         kept_status=kept_status, **lineage)


def compute_attempt_sha(row: Mapping[str, Any], case_id: str, arm: str, attempt: int,
                        status: str, finish_reason: str | None,
                        succeeded: bool, prompt_sha: str,
                        raw_text_sha: str | None) -> str:
    payload = {"case_id": case_id, "arm": arm, "attempt": attempt,
               "prompt_sha256": prompt_sha, "raw_text_sha256": raw_text_sha,
               "succeeded": succeeded, "finish_reason": finish_reason,
               "status": status}
    return sha256_text(canon(payload))


def parse_raw_latest(row: dict, file_arm: str, case: CaseData) -> RawData:
    where = f"raw[{case.case_id}/{file_arm}]"
    require(row.get("case_id") == case.case_id, "schema_raw",
            f"{where}: case_id 与 cases 不一致")
    require(row.get("arm") == file_arm, "schema_raw",
            f"{where}: arm 与文件名不一致（{row.get('arm')!r}）")
    require(row.get("split") == case.split, "schema_raw",
            f"{where}: split 与 cases 不一致（{row.get('split')!r} vs {case.split}）")
    missing = sorted(RAW_REQUIRED - set(row))
    require(not missing, "schema_raw", f"{where}: 缺字段 {missing}")
    attempt = row.get("attempt")
    require(is_int(attempt) and attempt >= 1, "schema_raw",
            f"{where}.attempt 须为 >=1 整数")
    status = row.get("status")
    require(status in STATUS_VALUES, "schema_raw",
            f"{where}.status 非法：{status!r}（接受 {sorted(STATUS_VALUES)}）")
    usage = validate_usage(row.get("usage"), where)
    elapsed = validate_elapsed(row, where)
    finish_reason = row.get("finish_reason")
    require(finish_reason is None or isinstance(finish_reason, str), "schema_raw",
            f"{where}.finish_reason 须为字符串或 null")
    if status == "length":
        require(finish_reason == "length", "schema_raw",
                f"{where}: status=length 时 finish_reason 须为 length")
    if status == "content_filter":
        require(finish_reason == "content_filter", "schema_raw",
                f"{where}: status=content_filter 时 finish_reason 须为 content_filter")
    raw_item_count = row.get("raw_item_count")
    require(is_int(raw_item_count) and raw_item_count >= 0, "schema_raw",
            f"{where}.raw_item_count 须为非负整数")

    list_keys = [key for key in ("parsed_candidates", "candidates") if key in row]
    require(len(list_keys) == 1, "schema_raw",
            f"{where}: parsed_candidates / candidates 必须且只能出现一个")
    candidates_raw = row[list_keys[0]]
    require(isinstance(candidates_raw, list), "schema_raw",
            f"{where}.{list_keys[0]} 须为数组")
    candidates: list[CandidateData] = []
    candidate_ids: set[str] = set()
    for index, candidate_row in enumerate(candidates_raw):
        spot = f"{where}.{list_keys[0]}[{index}]"
        candidate = validate_candidate(candidate_row, file_arm, spot)
        require(candidate.candidate_id not in candidate_ids, "schema_candidate",
                f"{spot}.candidate_id 重复 {candidate.candidate_id}")
        candidate_ids.add(candidate.candidate_id)
        candidates.append(candidate)

    if status in EMPTY_STATUSES:
        require(not candidates, "schema_raw",
                f"{where}: status={status} 不得携带 parsed candidates")
        if status == "empty_after_parse":
            require(raw_item_count > 0, "schema_raw",
                    f"{where}: empty_after_parse 的 raw_item_count 须 >0")
        else:
            require(raw_item_count == 0, "schema_raw",
                    f"{where}: status={status} 的 raw_item_count 须为 0")
    else:
        require(candidates, "schema_raw",
                f"{where}: status={status} 须携带 parsed candidates")
        kept_count = sum(1 for c in candidates if c.kept_status == "kept")
        if status == "ok":
            require(kept_count == len(candidates), "schema_raw",
                    f"{where}: status=ok 要求全部 kept")
        elif status == "all_gated":
            require(kept_count == 0, "schema_raw",
                    f"{where}: status=all_gated 要求无 kept")
        elif status == "partial_gated":
            require(0 < kept_count < len(candidates), "schema_raw",
                    f"{where}: status=partial_gated 要求部分 kept")
        else:
            raise ScoreError("schema_raw", f"{where}: 非预期 status={status}")
        require(raw_item_count >= len(candidates), "schema_raw",
                f"{where}: raw_item_count 不得小于 parsed candidates 数")

                                      
    raw_text = row.get("raw_text")
    require(raw_text is None or isinstance(raw_text, str), "schema_raw",
            f"{where}.raw_text 须为字符串或 null")
    prompt = row.get("prompt")
    require(prompt is None or isinstance(prompt, str), "schema_raw",
            f"{where}.prompt 须为字符串或 null")
    prompt_sha = row.get("prompt_sha256")
    if prompt_sha is not None:
        require(is_hex64(prompt_sha), "schema_raw", f"{where}.prompt_sha256 非法")
        if isinstance(prompt, str):
            require(sha256_text(prompt) == prompt_sha, "schema_raw",
                    f"{where}: prompt 与 prompt_sha256 不一致")
    elif isinstance(prompt, str):
        prompt_sha = sha256_text(prompt)
    else:
        raise ScoreError("schema_raw", f"{where}: 缺 prompt/prompt_sha256，无法验证 attempt SHA")
    raw_text_sha = row.get("raw_text_sha256")
    if raw_text_sha is not None:
        require(is_hex64(raw_text_sha), "schema_raw", f"{where}.raw_text_sha256 非法")
        if isinstance(raw_text, str):
            require(sha256_text(raw_text) == raw_text_sha, "schema_raw",
                    f"{where}: raw_text 与 raw_text_sha256 不一致")
    elif isinstance(raw_text, str) and raw_text:
        raw_text_sha = sha256_text(raw_text)
    call = row.get("call")
    if call is not None:
        require(isinstance(call, dict), "schema_raw", f"{where}.call 须为对象")
        if "attempted" in call:
            require(isinstance(call["attempted"], bool), "schema_raw",
                    f"{where}.call.attempted 须为布尔")
        if "succeeded" in call:
            require(isinstance(call["succeeded"], bool), "schema_raw",
                    f"{where}.call.succeeded 须为布尔")
    call_attempted = call.get("attempted") if isinstance(call, dict) else None
    succeeded = None
    if isinstance(call, dict) and isinstance(call.get("succeeded"), bool):
        succeeded = call["succeeded"]
    if succeeded is None and isinstance(row.get("succeeded"), bool):
        succeeded = row["succeeded"]
    if succeeded is None:
        succeeded = raw_text is not None
    attempt_sha = compute_attempt_sha(row, case.case_id, file_arm, attempt, status,
                                      finish_reason, succeeded, prompt_sha, raw_text_sha)
    stored_attempt_sha = row.get("attempt_sha256")
    if stored_attempt_sha is not None:
        require(is_hex64(stored_attempt_sha), "schema_raw",
                f"{where}.attempt_sha256 非法")
        require(stored_attempt_sha == attempt_sha, "schema_raw_sha",
                f"{where}: attempt_sha256 与记录内容不一致")
    computed_content_sha = content_sha(candidates_raw)
    stored_content_sha = row.get("content_sha256")
    if stored_content_sha is not None:
        require(is_hex64(stored_content_sha), "schema_raw",
                f"{where}.content_sha256 非法")
        require(stored_content_sha == computed_content_sha, "schema_raw_sha",
                f"{where}: content_sha256 与 parsed candidates 不一致")

    top_attempt: str | None = None
    top_content: str | None = None
    if file_arm == "A2":
        top_attempt = row.get("p0_attempt_sha256")
        top_content = row.get("p0_content_sha256")
        if top_attempt is not None or top_content is not None:
            require(is_hex64(top_attempt) and is_hex64(top_content), "schema_lineage",
                    f"{where}: A2 顶层 p0_attempt_sha256/p0_content_sha256 须同时为 64 hex")
    zero_usage = all(value in (0, None) for value in usage.values())
    not_run = bool(file_arm == "A2" and status == "valid_empty" and not candidates
                   and (call_attempted is False or prompt == ""
                        or (elapsed == 0.0 and zero_usage)))
    return RawData(case_id=case.case_id, split=case.split, arm=file_arm,
                   attempt=attempt, status=status, usage=usage, elapsed=elapsed,
                   finish_reason=finish_reason, raw_item_count=raw_item_count,
                   candidates=candidates,
                   candidate_by_id={c.candidate_id: c for c in candidates},
                   content_sha256=computed_content_sha, attempt_sha256=attempt_sha,
                   blind_id=blind_id_of(case.case_id, file_arm, attempt,
                                        computed_content_sha),
                   cap_applied=raw_item_count > MAX_ITEMS,
                   at_cap=raw_item_count == MAX_ITEMS,
                   p0_attempt_sha256=top_attempt,
                   p0_content_sha256=top_content,
                   not_run=not_run,
                   not_run_reason=("collector_a2_not_attempted" if not_run else None))


def load_raw_sets(raw_rows_by_arm: Mapping[str, Sequence[dict]],
                  cases: Sequence[CaseData]) -> tuple[dict[tuple[str, str], RawData], list[str]]:
    cases_by_id = {case.case_id: case for case in cases}
    latest: dict[tuple[str, str], dict] = {}
    superseded: Counter = Counter()
    warnings: list[str] = []
    for arm in ARMS:
        rows = raw_rows_by_arm.get(arm)
        require(isinstance(rows, list), "schema_raw", f"缺少 raw-{arm}.jsonl 行数组")
        for index, row in enumerate(rows, 1):
            require(isinstance(row, dict), "schema_raw", f"raw-{arm}[{index}] 须为对象")
            case_id = row.get("case_id")
            row_arm = row.get("arm")
            require(is_nonempty_str(case_id), "schema_raw",
                    f"raw-{arm}[{index}].case_id 须为非空字符串")
            require(row_arm == arm, "schema_raw",
                    f"raw-{arm}[{index}]: arm 与文件名不一致（{row_arm!r}）")
            require(case_id in cases_by_id, "schema_raw",
                    f"raw-{arm}[{index}] 引用未知 case：{case_id}")
            key = (case_id, arm)
            if key in latest:
                superseded[key] += 1
            latest[key] = row
    raw_sets: dict[tuple[str, str], RawData] = {}
    for case in cases:
        for arm in ARMS:
            key = (case.case_id, arm)
            if key not in latest:
                if arm == "A2":
                    p0_key = (case.case_id, "P0")
                    require(p0_key in latest, "schema_raw_missing",
                            f"缺少 {case.case_id}/P0 的最新 raw 输出")
                    p0 = raw_sets.get(p0_key)
                    if p0 is None:
                        p0 = parse_raw_latest(latest[p0_key], "P0", case)
                        raw_sets[p0_key] = p0
                    if not p0.candidates:
                        synthetic = synthesize_a2_not_run(case, p0)
                        raw_sets[key] = synthetic
                        warnings.append(
                            f"{case.case_id}/A2 未采集（P0 无候选，采集器按设计跳过）；"
                            "评分按合法空输出 valid_empty 计入，不伪造候选")
                        continue
                require(False, "schema_raw_missing",
                        f"缺少 {case.case_id}/{arm} 的最新 raw 输出")
            raw_sets[key] = parse_raw_latest(latest[key], arm, case)
            raw_sets[key].superseded = superseded.get(key, 0)
            if raw_sets[key].not_run:
                warnings.append(f"{case.case_id}/A2 为采集器未运行占位"
                                "（call.attempted=false, valid_empty）；无 ledger 行")
    for key, count in sorted(superseded.items()):
        if count:
            warnings.append(f"{key[0]}/{key[1]} 有 {count} 条被 last-write-wins 覆盖")
    return raw_sets, warnings


def synthesize_a2_not_run(case: CaseData, p0: RawData) -> RawData:
    """P0 无候选时采集器不跑 A2；为 A2 视图构造显式空输出，不伪造候选。

    负控制（gold 为空）因此按 valid_empty 计 complete=true；正例 P0 空则 A2
    coverage=0、complete=false。成本无 A2 调用，ledger 覆盖检查跳过此合成行。
    """
    content = content_sha([])
    attempt = 1
    attempt_sha = sha256_text(canon({"synthetic": "a2_not_run",
                                     "case_id": case.case_id,
                                     "p0_attempt_sha256": p0.attempt_sha256,
                                     "p0_content_sha256": p0.content_sha256}))
    return RawData(case_id=case.case_id, split=case.split, arm="A2",
                   attempt=attempt, status="valid_empty",
                   usage={"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
                   elapsed=0.0, finish_reason="stop", raw_item_count=0,
                   candidates=[], candidate_by_id={}, content_sha256=content,
                   attempt_sha256=attempt_sha,
                   blind_id=blind_id_of(case.case_id, "A2", attempt, content),
                   cap_applied=False, at_cap=False,
                   p0_attempt_sha256=p0.attempt_sha256,
                   p0_content_sha256=p0.content_sha256,
                   not_run=True, not_run_reason="p0_no_candidates")


def validate_atom(row: Any, where: str) -> AtomData:
    require(isinstance(row, dict), "schema_annotation", f"{where} 须为对象")
    missing = sorted(ATOM_KEYS - set(row))
    extra = sorted(set(row) - ATOM_KEYS - ATOM_OPTIONAL)
    require(not missing and not extra, "schema_annotation",
            f"{where}: atom 字段不符（缺 {missing}，多 {extra}）")
    if "notes" in row:
        require(row["notes"] is None or isinstance(row["notes"], str), "schema_annotation",
                f"{where}.notes 须为字符串或 null")
    candidate_id = row.get("candidate_id")
    require(is_nonempty_str(candidate_id), "schema_annotation",
            f"{where}.candidate_id 须为非空字符串")
    entailment = row.get("entailment")
    require(entailment in ENTAILMENTS, "schema_annotation",
            f"{where}.entailment 须为 {sorted(ENTAILMENTS)} 之一")
    for key in ("persistence_appropriate", "non_atomic", "over_split"):
        require(isinstance(row.get(key), bool), "schema_annotation",
                f"{where}.{key} 须为布尔")
    errors = row.get("context_errors")
    require(isinstance(errors, list)
            and all(isinstance(x, str) and x in CONTEXT_ERRORS for x in errors),
            "schema_annotation",
            f"{where}.context_errors 须为 {sorted(CONTEXT_ERRORS)} 子集")
    require(len(set(errors)) == len(errors), "schema_annotation",
            f"{where}.context_errors 有重复")
    forbidden = row.get("matched_forbidden_ids")
    require(isinstance(forbidden, list) and all(is_nonempty_str(x) for x in forbidden),
            "schema_annotation", f"{where}.matched_forbidden_ids 须为字符串数组")
    require(len(set(forbidden)) == len(forbidden), "schema_annotation",
            f"{where}.matched_forbidden_ids 有重复")
    duplicate_group = row.get("duplicate_group")
    require(duplicate_group is None
            or (isinstance(duplicate_group, str) and bool(duplicate_group.strip()))
            or (is_int(duplicate_group) and duplicate_group >= 0), "schema_annotation",
            f"{where}.duplicate_group 须为 null、非空字符串或非负整数")
    core = row.get("core_entailed_gold_ids")
    full = row.get("fully_covered_gold_ids")
    for key, value in (("core_entailed_gold_ids", core),
                       ("fully_covered_gold_ids", full)):
        require(isinstance(value, list) and all(is_nonempty_str(x) for x in value),
                "schema_annotation", f"{where}.{key} 须为字符串数组")
        require(len(set(value)) == len(value), "schema_annotation",
                f"{where}.{key} 有重复")
                                                                 
    core_ids = tuple(dict.fromkeys(list(core) + list(full)))
    alternative = row.get("alternative_partition")
    require(alternative is None or isinstance(alternative, (bool, str, int, list, dict)),
            "schema_annotation", f"{where}.alternative_partition 类型非法")
    return AtomData(candidate_id=candidate_id, entailment=entailment,
                    persistence_appropriate=row["persistence_appropriate"],
                    non_atomic=row["non_atomic"], over_split=row["over_split"],
                    context_errors=tuple(errors),
                    matched_forbidden_ids=tuple(forbidden),
                    duplicate_group=duplicate_group,
                    core_entailed_gold_ids=core_ids,
                    fully_covered_gold_ids=tuple(full),
                    alternative_partition=alternative)


def infer_case_id_from_blind(blind_id: str, case_ids: Sequence[str]) -> str | None:
    matches = [case_id for case_id in case_ids
               if re.search(r"(?<![A-Za-z0-9])" + re.escape(case_id) + r"(?![A-Za-z0-9])",
                            blind_id)]
    return matches[0] if len(matches) == 1 else None


def load_annotations(rows: Sequence[dict], raw_sets: Mapping[tuple[str, str], RawData],
                     cases: Sequence[CaseData]) -> tuple[dict[tuple[str, str], AnnotationData],
                                                         list[str]]:
    require(rows, "annotations_empty", "annotations-v1.jsonl 为空")
    cases_by_id = {case.case_id: case for case in cases}
    annotations: list[AnnotationData] = []
    annotation_cases: dict[str, str] = {}
    blind_seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"annotations[{index}]"
        require(isinstance(row, dict), "schema_annotation", f"{where} 须为对象")
        missing = sorted(ANNOTATION_KEYS - set(row))
        extra = sorted(set(row) - ANNOTATION_KEYS - ANNOTATION_OPTIONAL)
        require(not missing and not extra, "schema_annotation",
                f"{where}: 字段不符（缺 {missing}，多 {extra}）")
        blind_id = row.get("blind_id")
        require(is_nonempty_str(blind_id), "schema_annotation",
                f"{where}.blind_id 须为非空字符串")
        require(blind_id not in blind_seen, "schema_annotation",
                f"{where}.blind_id 重复 {blind_id}")
        blind_seen.add(blind_id)
        case_id = row.get("case_id")
        if case_id is not None:
            require(is_nonempty_str(case_id) and case_id in cases_by_id,
                    "annotation_case", f"{where}.case_id 未知或非法")
            split = row.get("split")
            require(split is None or split == cases_by_id[case_id].split,
                    "annotation_case", f"{where}.split 与 cases 不一致")
            annotation_cases[blind_id] = case_id
        for key in ("stage1_complete", "stage2_complete", "adjudicated"):
            require(row.get(key) is True, "schema_annotation",
                    f"{where}.{key} 必须为 true（已完成两阶段并 adjudicate）")
        atom_rows = row.get("atoms")
        require(isinstance(atom_rows, list), "schema_annotation",
                f"{where}.atoms 须为数组")
        atoms: list[AtomData] = []
        seen_ids: set[str] = set()
        for atom_index, atom_row in enumerate(atom_rows):
            atom = validate_atom(atom_row, f"{where}.atoms[{atom_index}]")
            require(atom.candidate_id not in seen_ids, "schema_annotation",
                    f"{where}: candidate_id 重复 {atom.candidate_id}")
            seen_ids.add(atom.candidate_id)
            atoms.append(atom)
        annotations.append(AnnotationData(blind_id=blind_id, atoms=atoms,
                                          atom_by_id={a.candidate_id: a for a in atoms}))

                                             
    by_signature: dict[frozenset[str], list[tuple[str, str]]] = defaultdict(list)
    empty_keys: list[tuple[str, str]] = []
    for key, raw in raw_sets.items():
        signature = frozenset(c.candidate_id for c in raw.candidates)
        if signature:
            by_signature[signature].append(key)
        else:
            empty_keys.append(key)
    for signature, keys in by_signature.items():
        require(len(keys) == 1, "raw_candidate_set_ambiguous",
                f"多个 raw output set 拥有相同 candidate 集合：{sorted(keys)}")
    empty_blind = {raw_sets[key].blind_id: key for key in empty_keys}
    require(len(empty_blind) == len(empty_keys), "raw_candidate_set_ambiguous",
            "空输出 raw set 的 blind_id 冲突")

    matched: dict[tuple[str, str], AnnotationData] = {}
    warnings: list[str] = []
    for annotation in annotations:
        if annotation.atoms:
            signature = frozenset(annotation.atom_by_id)
            keys = by_signature.get(signature, [])
            require(len(keys) == 1, "annotation_set_mismatch",
                    f"标注 {annotation.blind_id}: candidate 集合不匹配任何唯一 raw output set"
                    f"（atoms={len(annotation.atoms)}）")
            key = keys[0]
        else:
            key = empty_blind.get(annotation.blind_id)
            if key is None:
                metadata_case = annotation_cases.get(annotation.blind_id)
                if metadata_case is not None:
                    candidates = [candidate_key for candidate_key in empty_keys
                                  if candidate_key[0] == metadata_case]
                    require(len(candidates) == 1, "annotation_empty_case_mismatch",
                            f"标注 {annotation.blind_id}: case={metadata_case} 对应空输出数不是 1")
                    key = candidates[0]
                else:
                    inferred_case = infer_case_id_from_blind(
                        annotation.blind_id, [case.case_id for case in cases])
                    if inferred_case is not None:
                        candidates = [candidate_key for candidate_key in empty_keys
                                      if candidate_key[0] == inferred_case]
                        if len(candidates) == 1:
                            key = candidates[0]
                            warnings.append(f"空输出标注 {annotation.blind_id} 按 blind_id 中的 "
                                            f"case_id={inferred_case} 匹配唯一空集")
                if key is None:
                    require(len(empty_keys) == 1, "annotation_empty_unmatched",
                            f"标注 {annotation.blind_id}: 空 atom 集无法匹配唯一空输出 raw set"
                            f"（空集 {len(empty_keys)} 个，blind_id 哈希不匹配）")
                    key = empty_keys[0]
                    warnings.append(f"空输出标注 {annotation.blind_id} 仅按全局唯一空集匹配")
        if annotation.blind_id in annotation_cases:
            require(key[0] == annotation_cases[annotation.blind_id], "annotation_case_mismatch",
                    f"标注 {annotation.blind_id} 的 case 元数据与 raw 不匹配")
        require(key not in matched, "annotation_duplicate",
                f"raw output set {key} 被重复标注（{annotation.blind_id}）")
        matched[key] = annotation

                                                          
                          
    for key, raw in raw_sets.items():
        if key not in matched and raw.not_run:
            matched[key] = AnnotationData(blind_id=raw.blind_id, atoms=[],
                                          atom_by_id={})
            warnings.append(f"{key[0]}/{key[1]} 为合成空输出，补空 adjudication"
                            "（P0 无候选，A2 未运行）")

    missing = sorted(set(raw_sets) - set(matched))
    extra = sorted(set(matched) - set(raw_sets))
    require(not missing, "annotation_missing",
            f"以下最新 case×arm 没有完整 adjudicated 标注：{missing}")
    require(not extra, "annotation_extra", f"标注匹配到未知 raw output set：{extra}")
    require(len(matched) == len(raw_sets), "annotation_count",
            "标注数与最新 raw output set 数不一致")

                                                       
    for key, annotation in matched.items():
        raw = raw_sets[key]
        case = cases_by_id[key[0]]
        for atom in annotation.atoms:
            require(atom.candidate_id in raw.candidate_by_id, "annotation_candidate",
                    f"{annotation.blind_id}: atom 引用未知 candidate {atom.candidate_id}")
            for gold_id in atom.core_entailed_gold_ids:
                require(gold_id in case.gold_by_id, "annotation_gold_ref",
                        f"{annotation.blind_id}/{atom.candidate_id}: 未知 gold {gold_id}")
            for gold_id in atom.fully_covered_gold_ids:
                require(gold_id in case.gold_by_id, "annotation_gold_ref",
                        f"{annotation.blind_id}/{atom.candidate_id}: 未知 gold {gold_id}")
            for forbidden_id in atom.matched_forbidden_ids:
                require(forbidden_id in case.forbidden, "annotation_forbidden_ref",
                        f"{annotation.blind_id}/{atom.candidate_id}: 未知 forbidden "
                        f"{forbidden_id}")
        require(len(annotation.atoms) == len(raw.candidates), "annotation_candidate",
                f"{annotation.blind_id}: atom 数与 candidates 数不一致")
    return matched, warnings


def load_ledger(rows: Sequence[dict], cases: Sequence[CaseData]) -> tuple[list[LedgerData],
                                                                         list[str]]:
    require(rows, "ledger_empty", "call-ledger.jsonl 为空")
    cases_by_id = {case.case_id: case for case in cases}
    by_attempt: dict[tuple[str, str, int], LedgerData] = {}
    warnings: list[str] = []
    duplicates = 0
    for index, row in enumerate(rows, 1):
        where = f"ledger[{index}]"
        require(isinstance(row, dict), "schema_ledger", f"{where} 须为对象")
        missing = sorted(LEDGER_REQUIRED - set(row))
        require(not missing, "schema_ledger", f"{where}: 缺字段 {missing}")
        case_id = row.get("case_id")
        require(is_nonempty_str(case_id) and case_id in cases_by_id, "schema_ledger",
                f"{where}.case_id 未知或非法")
        arm = row.get("arm")
        require(arm in ARMS, "schema_ledger", f"{where}.arm 非法")
        attempt = row.get("attempt")
        require(is_int(attempt) and attempt >= 1, "schema_ledger",
                f"{where}.attempt 须为 >=1 整数")
        split = row.get("split", cases_by_id[case_id].split)
        require(split in SPLITS and split == cases_by_id[case_id].split, "schema_ledger",
                f"{where}.split 与 cases 不一致")
        usage = validate_usage(row.get("usage"), where)
        elapsed = validate_elapsed(row, where)
        ledger = LedgerData(case_id=case_id, arm=arm, attempt=attempt, split=split,
                            usage=usage, elapsed=elapsed)
        key = (case_id, arm, attempt)
        if key in by_attempt:
            duplicates += 1
        by_attempt[key] = ledger
    if duplicates:
        warnings.append(f"ledger 有 {duplicates} 条重复 (case,arm,attempt)，按 last-write-wins 去重")
    return list(by_attempt.values()), warnings


                                              


def view_measure(case: CaseData, raw: RawData, atoms: Sequence[AtomData]) -> dict:
    full: set[str] = set()
    core: set[str] = set()
    supported = partial = unsupported = 0
    persistence_false_cases = 0
    persistence_false_atoms = 0
    context_error_atoms = 0
    forbidden_matches: list[str] = []
    severe_matches: list[str] = []
    inversion_atoms = 0
    inversion_types: Counter = Counter()
    non_atomic_atoms = 0
    allowed_non_atomic_atoms = 0
    over_split_atoms = 0
    duplicate_groups: dict[Any, int] = defaultdict(int)
    for atom in atoms:
        full.update(atom.fully_covered_gold_ids)
        core.update(atom.core_entailed_gold_ids)
        if atom.entailment == "supported":
            supported += 1
        elif atom.entailment == "partial":
            partial += 1
        else:
            unsupported += 1
        if not atom.persistence_appropriate:
            persistence_false_atoms += 1
            persistence_false_cases = 1
        if atom.context_errors:
            context_error_atoms += 1
            for error in atom.context_errors:
                if error in {"polarity", "condition"}:
                    inversion_atoms += 1
                    inversion_types[error] += 1
        if atom.matched_forbidden_ids:
            for forbidden_id in atom.matched_forbidden_ids:
                forbidden_matches.append(forbidden_id)
                if case.forbidden.get(forbidden_id) == "severe":
                    severe_matches.append(forbidden_id)
        if atom.non_atomic:
            non_atomic_atoms += 1
            if atom.alternative_accepted:
                allowed_non_atomic_atoms += 1
        if atom.over_split:
            over_split_atoms += 1
        if atom.duplicate_group is not None:
            duplicate_groups[atom.duplicate_group] += 1

    duplicate_extra = sum(count - 1 for count in duplicate_groups.values() if count >= 2)
    duplicate_case = 1 if duplicate_extra else 0
    context_lost: set[str] = set()
    for gold_id in case.context_gold_ids:
        if gold_id in full:
            continue
        if any(gold_id in atom.core_entailed_gold_ids and atom.context_errors
               for atom in atoms):
            context_lost.add(gold_id)

    reasons: list[str] = []
    if raw.status == "empty_after_parse":
        reasons.append("empty_after_parse")
    elif raw.status in INVALID_OUTPUT:
        reasons.append(f"invalid_output:{raw.status}")
    else:
        if not set(case.gold_by_id) <= full:
            reasons.append("gold_not_fully_covered")
        if partial:
            reasons.append("partial")
        if unsupported:
            reasons.append("unsupported")
        if persistence_false_atoms:
            reasons.append("false_persistence")
        if context_error_atoms:
            reasons.append("context_error")
        if duplicate_extra:
            reasons.append("duplicate")
        if over_split_atoms:
            reasons.append("over_split")
        if non_atomic_atoms > allowed_non_atomic_atoms:
            reasons.append("disallowed_non_atomic")

    case_full = set(case.gold_by_id) <= full
    return {
        "atoms": atoms,
        "kept_atoms": len(atoms),
        "full": full,
        "core": core,
        "case_full_coverage": case_full,
        "faithfulness": {"supported": supported, "partial": partial,
                         "unsupported": unsupported},
        "persistence_false_cases": persistence_false_cases,
        "persistence_false_atoms": persistence_false_atoms,
        "context_error_atoms": context_error_atoms,
        "context_lost": context_lost,
        "forbidden_matches": forbidden_matches,
        "severe_matches": severe_matches,
        "inversion_atoms": inversion_atoms,
        "inversion_types": dict(inversion_types),
        "non_atomic_atoms": non_atomic_atoms,
        "allowed_non_atomic_atoms": allowed_non_atomic_atoms,
        "over_split_atoms": over_split_atoms,
        "duplicate_extra": duplicate_extra,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_case": duplicate_case,
        "complete": not reasons,
        "complete_reasons": reasons,
        "valid_output": raw.status not in INVALID_OUTPUT,
    }


def raw_measure(case: CaseData, atoms: Sequence[AtomData]) -> dict:
    full: set[str] = set()
    core: set[str] = set()
    supported = 0
    for atom in atoms:
        full.update(atom.fully_covered_gold_ids)
        core.update(atom.core_entailed_gold_ids)
        if atom.entailment == "supported":
            supported += 1
    return {"atoms": len(atoms), "full": full, "core": core, "supported": supported,
            "case_full_coverage": set(case.gold_by_id) <= full}


def measure_case(case: CaseData, raw: RawData, annotation: AnnotationData) -> dict:
    require(set(annotation.atom_by_id) == set(raw.candidate_by_id),
            "annotation_candidate", f"{case.case_id}/{raw.arm}: atom 集合与 candidate 集合不一致")
    raw_atoms = [annotation.atom_by_id[c.candidate_id] for c in raw.candidates]
    direct_atoms = [annotation.atom_by_id[c.candidate_id] for c in raw.candidates
                    if c.kept_status == "kept"]
    loop_atoms = [annotation.atom_by_id[c.candidate_id] for c in raw.candidates
                  if c.kept_status == "kept" and c.tier is not None]
    return {"case": case, "raw": raw, "annotation": annotation,
            "raw_atoms": raw_atoms, "direct_atoms": direct_atoms,
            "loop_atoms": loop_atoms,
            "raw_measure": raw_measure(case, raw_atoms),
            "direct_measure": view_measure(case, raw, direct_atoms),
            "loop_measure": view_measure(case, raw, loop_atoms)}


def aggregate_view(measures: Sequence[dict], view: str) -> dict:
    key = f"{view}_measure"
    n_cases = len(measures)
    kept_atoms = 0
    supported = partial = unsupported = 0
    full_pairs: set[tuple[str, str]] = set()
    all_pairs: set[tuple[str, str]] = set()
    context_pairs: set[tuple[str, str]] = set()
    non_atomic_pairs: set[tuple[str, str]] = set()
    non_atomic_atoms = allowed_non_atomic = 0
    over_split = 0
    duplicate_extra = 0
    duplicate_cases = 0
    context_lost_pairs: set[tuple[str, str]] = set()
    persistence_false_cases = 0
    persistence_false_neg = 0
    negative_cases = 0
    forbidden_cases = 0
    severe_cases = 0
    severe_matches = 0
    inversion_atoms = 0
    inversion_cases = 0
    inversion_types: Counter = Counter()
    complete_cases = 0
    reasons: Counter = Counter()
    full_coverage_cases = 0
    empty_kept_cases = 0

    for measure in measures:
        case = measure["case"]
        measured = measure[key]
        atoms = measured["atoms"]
        kept_atoms += len(atoms)
        if not atoms:
            empty_kept_cases += 1
        supported += measured["faithfulness"]["supported"]
        partial += measured["faithfulness"]["partial"]
        unsupported += measured["faithfulness"]["unsupported"]
        for gold_id in measured["full"]:
            full_pairs.add((case.case_id, gold_id))
        for gold in case.gold:
            all_pairs.add((case.case_id, gold.gold_id))
            if gold.has_context:
                context_pairs.add((case.case_id, gold.gold_id))
        for atom in atoms:
            if atom.non_atomic:
                non_atomic_atoms += 1
                for gold_id in atom.core_entailed_gold_ids:
                    non_atomic_pairs.add((case.case_id, gold_id))
                if atom.alternative_accepted:
                    allowed_non_atomic += 1
            if atom.over_split:
                over_split += 1
        duplicate_extra += measured["duplicate_extra"]
        duplicate_cases += measured["duplicate_case"]
        for gold_id in measured["context_lost"]:
            context_lost_pairs.add((case.case_id, gold_id))
        if measured["persistence_false_cases"]:
            persistence_false_cases += 1
        if not case.gold:
            negative_cases += 1
            if measured["persistence_false_cases"]:
                persistence_false_neg += 1
        if measured["forbidden_matches"]:
            forbidden_cases += 1
        if measured["severe_matches"]:
            severe_cases += 1
            severe_matches += len(measured["severe_matches"])
        if measured["inversion_atoms"]:
            inversion_atoms += measured["inversion_atoms"]
            inversion_cases += 1
            for kind, count in measured["inversion_types"].items():
                inversion_types[kind] += count
        if measured["complete"]:
            complete_cases += 1
        else:
            reasons.update(measured["complete_reasons"])
        if measured["case_full_coverage"]:
            full_coverage_cases += 1

    return {
        "view": "direct-kept" if view == "direct" else "loop-visible",
        "cases": n_cases,
        "kept_atoms": kept_atoms,
        "kept_cases": sum(1 for m in measures if m[key]["atoms"]),
        "empty_kept_cases": empty_kept_cases,
        "faithfulness": {
            "supported": supported, "partial": partial, "unsupported": unsupported,
            "denominator": kept_atoms,
            "value": (float(supported) / kept_atoms) if kept_atoms else None,
            "partial_rate": (float(partial) / kept_atoms) if kept_atoms else None,
            "unsupported_rate": (float(unsupported) / kept_atoms) if kept_atoms else None,
        },
        "source_coverage": ratio(len(full_pairs), len(all_pairs)),
        "full_coverage_case_rate": ratio(full_coverage_cases, n_cases),
        "non_atomic": {
            "obligation_covered": len(non_atomic_pairs),
            "obligation_denominator": len(all_pairs),
            "obligation_rate": (float(len(non_atomic_pairs)) / len(all_pairs))
                               if all_pairs else None,
            "case_count": sum(1 for m in measures
                              if any(a.non_atomic for a in m[key]["atoms"])),
            "case_rate": (float(sum(1 for m in measures
                                    if any(a.non_atomic for a in m[key]["atoms"]))) / n_cases)
                         if n_cases else None,
            "atom_count": non_atomic_atoms,
            "atom_rate": (float(non_atomic_atoms) / kept_atoms) if kept_atoms else None,
            "allowed_atom_count": allowed_non_atomic,
            "disallowed_atom_count": non_atomic_atoms - allowed_non_atomic,
        },
        "over_split": {
            "atom_count": over_split,
            "atom_rate": (float(over_split) / kept_atoms) if kept_atoms else None,
            "case_count": sum(1 for m in measures
                              if any(a.over_split for a in m[key]["atoms"])),
            "case_rate": (float(sum(1 for m in measures
                                    if any(a.over_split for a in m[key]["atoms"]))) / n_cases)
                         if n_cases else None,
        },
        "duplicate": {
            "extra_atoms": duplicate_extra,
            "atom_rate": (float(duplicate_extra) / kept_atoms) if kept_atoms else None,
            "case_count": duplicate_cases,
            "case_rate": (float(duplicate_cases) / n_cases) if n_cases else None,
        },
        "context_loss": {
            "lost_gold": len(context_lost_pairs),
            "denominator": len(context_pairs),
            "rate": (float(len(context_lost_pairs)) / len(context_pairs))
                    if context_pairs else None,
        },
        "false_persistence": {
            "case_count": persistence_false_cases,
            "denominator": n_cases,
            "rate": (float(persistence_false_cases) / n_cases) if n_cases else None,
            "negative_control_count": persistence_false_neg,
            "negative_control_denominator": negative_cases,
            "negative_control_rate": (float(persistence_false_neg) / negative_cases)
                                     if negative_cases else None,
        },
        "forbidden": {
            "case_count": forbidden_cases,
            "case_rate": (float(forbidden_cases) / n_cases) if n_cases else None,
            "severe_case_count": severe_cases,
            "severe_match_count": severe_matches,
        },
        "inversion": {
            "atom_count": inversion_atoms,
            "case_count": inversion_cases,
            "types": dict(sorted(inversion_types.items())),
        },
        "complete_set": {
            "success": complete_cases,
            "denominator": n_cases,
            "rate": (float(complete_cases) / n_cases) if n_cases else None,
            "failure_reasons": dict(sorted(reasons.items())),
            "failed_cases": n_cases - complete_cases,
        },
        "valid_output": {
            "case_count": sum(1 for m in measures if m[key]["valid_output"]),
            "denominator": n_cases,
            "rate": (float(sum(1 for m in measures if m[key]["valid_output"])) / n_cases)
                    if n_cases else None,
        },
    }


def aggregate_raw(measures: Sequence[dict]) -> dict:
    full_pairs: set[tuple[str, str]] = set()
    all_pairs: set[tuple[str, str]] = set()
    full_cases = 0
    atoms = 0
    for measure in measures:
        case = measure["case"]
        measured = measure["raw_measure"]
        atoms += measured["atoms"]
        for gold_id in measured["full"]:
            full_pairs.add((case.case_id, gold_id))
        for gold in case.gold:
            all_pairs.add((case.case_id, gold.gold_id))
        if measured["case_full_coverage"]:
            full_cases += 1
    return {
        "candidates": atoms,
        "source_coverage": ratio(len(full_pairs), len(all_pairs)),
        "full_coverage_case_rate": ratio(full_cases, len(measures)),
    }


def arm_metrics(measures: Sequence[dict], include_case_rows: bool = True) -> dict:
    direct = aggregate_view(measures, "direct")
    loop = aggregate_view(measures, "loop")
    raw = aggregate_raw(measures)
    all_gold = sum(len(m["case"].gold) for m in measures)

    direct["raw_source_coverage"] = raw["source_coverage"]
    direct["raw_to_kept_loss_pp"] = (
        (raw["source_coverage"]["value"] - direct["source_coverage"]["value"]) * 100.0
        if raw["source_coverage"]["value"] is not None
        and direct["source_coverage"]["value"] is not None else None)
    loop["raw_source_coverage"] = raw["source_coverage"]
    loop["raw_to_loop_loss_pp"] = (
        (raw["source_coverage"]["value"] - loop["source_coverage"]["value"]) * 100.0
        if raw["source_coverage"]["value"] is not None
        and loop["source_coverage"]["value"] is not None else None)
    loop["rule_only_coverage"] = copy.deepcopy(loop["source_coverage"])

    memory_full: set[tuple[str, str]] = set()
    memory_only_full: set[tuple[str, str]] = set()
    memory_core: set[tuple[str, str]] = set()
    memory_only_core: set[tuple[str, str]] = set()
    dropped_memory_atoms = 0
    dropped_memory_cases = 0
    direct_complete_cases = 0
    loop_complete_cases = 0
    direct_minus_loop_complete = 0
    for measure in measures:
        case = measure["case"]
        loop_atoms = measure["loop_measure"]["atoms"]
        loop_full = measure["loop_measure"]["full"]
        loop_core = measure["loop_measure"]["core"]
        memory_atoms = [a for a in measure["direct_measure"]["atoms"]
                        if measure["raw"].candidate_by_id[a.candidate_id].tier is None]
        if memory_atoms:
            dropped_memory_cases += 1
        dropped_memory_atoms += len(memory_atoms)
        for atom in memory_atoms:
            for gold_id in atom.fully_covered_gold_ids:
                memory_full.add((case.case_id, gold_id))
                if gold_id not in loop_full:
                    memory_only_full.add((case.case_id, gold_id))
            for gold_id in atom.core_entailed_gold_ids:
                memory_core.add((case.case_id, gold_id))
                if gold_id not in loop_core:
                    memory_only_core.add((case.case_id, gold_id))
        if measure["direct_measure"]["complete"]:
            direct_complete_cases += 1
        if measure["loop_measure"]["complete"]:
            loop_complete_cases += 1
        if measure["direct_measure"]["complete"] and not measure["loop_measure"]["complete"]:
            direct_minus_loop_complete += 1

    loop["memory_projection"] = {
        "memory_full_gold": len(memory_full),
        "memory_only_full_gold": len(memory_only_full),
        "memory_core_gold": len(memory_core),
        "memory_only_core_gold": len(memory_only_core),
        "memory_only_coverage": (float(len(memory_only_full)) / all_gold)
                                if all_gold else None,
        "dropped_memory_atoms": dropped_memory_atoms,
        "dropped_memory_cases": dropped_memory_cases,
        "direct_complete_cases": direct_complete_cases,
        "loop_complete_cases": loop_complete_cases,
        "direct_minus_loop_complete_cases": direct_minus_loop_complete,
        "loop_projection_loss_pp": (
            (direct["source_coverage"]["value"] - loop["source_coverage"]["value"]) * 100.0
            if direct["source_coverage"]["value"] is not None
            and loop["source_coverage"]["value"] is not None else None),
    }

    status_counts: Counter = Counter(m["raw"].status for m in measures)
    valid_cases = sum(1 for m in measures if m["direct_measure"]["valid_output"])
    cap_cases = sum(1 for m in measures if m["raw"].cap_applied)
    at_cap_cases = sum(1 for m in measures if m["raw"].at_cap)
    zero_kept_cases = sum(1 for m in measures if not m["direct_measure"]["atoms"])
    all_gated_cases = sum(1 for m in measures if m["raw"].status == "all_gated")
    empty_cases = sum(1 for m in measures
                      if m["raw"].status in {"valid_empty", "empty_after_parse"})
    result = {
        "arm": measures[0]["raw"].arm if measures else None,
        "cases": len(measures),
        "gold_total": all_gold,
        "context_gold_total": sum(len(m["case"].context_gold_ids) for m in measures),
        "negative_controls": sum(1 for m in measures if not m["case"].gold),
        "raw": raw,
        "direct": direct,
        "loop": loop,
        "valid_output": {
            "case_count": valid_cases,
            "denominator": len(measures),
            "rate": (float(valid_cases) / len(measures)) if measures else None,
            "statuses": dict(sorted(status_counts.items())),
            "invalid_statuses": {k: v for k, v in sorted(status_counts.items())
                                 if k in INVALID_OUTPUT},
            "empty_output_cases": empty_cases,
            "all_gated_cases": all_gated_cases,
            "zero_kept_cases": zero_kept_cases,
        },
        "cap": {
            "applied_cases": cap_cases,
            "at_cap_cases": at_cap_cases,
            "raw_item_count_gt_8_cases": cap_cases,
            "exactly_8_cases": at_cap_cases,
            "limit": MAX_ITEMS,
        },
    }
    if include_case_rows:
        result["case_rows"] = [
            {
                "case_id": m["case"].case_id,
                "arm": m["raw"].arm,
                "attempt": m["raw"].attempt,
                "family": m["case"].family,
                "split": m["case"].split,
                "status": m["raw"].status,
                "not_run": m["raw"].not_run,
                "not_run_reason": m["raw"].not_run_reason,
                "valid_output": m["direct_measure"]["valid_output"],
                "raw_attempt_sha256": m["raw"].attempt_sha256,
                "raw_content_sha256": m["raw"].content_sha256,
                "raw_blind_id": m["raw"].blind_id,
                "superseded_rows": m["raw"].superseded,
                "raw_item_count": m["raw"].raw_item_count,
                "candidate_count": len(m["raw"].candidates),
                "kept_count": len(m["direct_measure"]["atoms"]),
                "loop_kept_count": len(m["loop_measure"]["atoms"]),
                "raw_full_coverage": m["raw_measure"]["case_full_coverage"],
                "direct_coverage": m["direct_measure"]["case_full_coverage"],
                "loop_coverage": m["loop_measure"]["case_full_coverage"],
                "direct_complete": m["direct_measure"]["complete"],
                "loop_complete": m["loop_measure"]["complete"],
                "direct_reasons": list(m["direct_measure"]["complete_reasons"]),
                "loop_reasons": list(m["loop_measure"]["complete_reasons"]),
                "cap_applied": m["raw"].cap_applied,
                "at_cap": m["raw"].at_cap,
            }
            for m in measures
        ]
    return result


def compact_arm_metrics(metrics: dict) -> dict:
    def view(view_name: str) -> dict:
        node = metrics[view_name]
        return {
            "kept_atoms": node["kept_atoms"],
            "coverage": node["source_coverage"],
            "faithfulness": node["faithfulness"]["value"],
            "faithfulness_detail": node["faithfulness"],
            "supported": node["faithfulness"]["supported"],
            "partial": node["faithfulness"]["partial"],
            "unsupported": node["faithfulness"]["unsupported"],
            "complete_set": node["complete_set"],
            "non_atomic_obligation": {
                "num": node["non_atomic"]["obligation_covered"],
                "den": node["non_atomic"]["obligation_denominator"],
                "value": node["non_atomic"]["obligation_rate"],
            },
            "non_atomic_detail": node["non_atomic"],
            "duplicate": node["duplicate"],
            "over_split": node["over_split"],
            "duplicate_rate": node["duplicate"]["atom_rate"],
            "over_split_rate": node["over_split"]["atom_rate"],
            "context_loss": node["context_loss"],
            "false_persistence": node["false_persistence"],
            "forbidden": node["forbidden"],
        }
    return {"cases": metrics["cases"], "gold_total": metrics["gold_total"],
            "raw": metrics["raw"],
            "valid_output": metrics["valid_output"],
            "direct": view("direct"), "loop": view("loop"),
            "statuses": metrics["valid_output"]["statuses"]}


def build_arm_bundles(measures_by_arm: Mapping[str, Sequence[dict]],
                      cases: Sequence[CaseData]) -> tuple[dict, list[dict]]:
    bundles: dict[str, dict] = {}
    stratified: list[dict] = []
    for arm in ARMS:
        measures = measures_by_arm[arm]
        by_split: dict[str, dict] = {}
        for split in SPLITS:
            subset = [m for m in measures if m["case"].split == split]
            by_split[split] = arm_metrics(subset, include_case_rows=False)
        families: list[dict] = []
        for split in SPLITS:
            names = sorted({m["case"].family for m in measures if m["case"].split == split})
            for family in names:
                subset = [m for m in measures
                          if m["case"].split == split and m["case"].family == family]
                node = arm_metrics(subset, include_case_rows=False)
                row = {"arm": arm, "split": split, "family": family,
                       **compact_arm_metrics(node)}
                families.append(row)
                stratified.append(row)
        bundles[arm] = {"arm": arm, "all": arm_metrics(measures, include_case_rows=True),
                        "by_split": by_split, "families": families}
    return bundles, stratified


                                              


def percentile_or_none(values: Sequence[float], q: float) -> float | None:
    value = percentile(values, q)
    return None if value is None else round(value, 6)


def cost_block(rows: Sequence[LedgerData], measures: Sequence[dict]) -> dict:
    calls = len(rows)
    case_latency: dict[str, float] = defaultdict(float)
    attempt_elapsed: list[float] = []
    prompt = completion = total = 0
    unknown_usage = 0
    retry_rows = 0
    for row in rows:
        case_latency[row.case_id] += row.elapsed
        attempt_elapsed.append(row.elapsed)
        if row.attempt > 1:
            retry_rows += 1
        usage = row.usage
        if (usage["prompt_tokens"] is None or usage["completion_tokens"] is None
                or usage["total_tokens"] is None):
            unknown_usage += 1
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]
        if (total_tokens is None and prompt_tokens is not None
                and completion_tokens is not None):
            total_tokens = prompt_tokens + completion_tokens
        prompt += prompt_tokens or 0
        completion += completion_tokens or 0
        total += total_tokens or 0
    case_values = list(case_latency.values())
    return {
        "calls": calls,
        "cases": len(case_values),
        "cases_without_calls": max(0, len(measures) - len(case_values)),
        "retry_rows": retry_rows,
        "calls_per_case": (float(calls) / len(case_values)) if case_values else None,
        "tokens": {"prompt": prompt, "completion": completion, "total": total},
        "unknown_usage_rows": unknown_usage,
        "elapsed_seconds_total": round(sum(attempt_elapsed), 6),
        "case_latency_seconds": {
            "p50": percentile_or_none(case_values, 0.50),
            "p95": percentile_or_none(case_values, 0.95),
            "total": round(sum(case_values), 6),
            "max": round(max(case_values), 6) if case_values else None,
        },
        "attempt_latency_seconds": {
            "p50": percentile_or_none(attempt_elapsed, 0.50),
            "p95": percentile_or_none(attempt_elapsed, 0.95),
            "max": round(max(attempt_elapsed), 6) if attempt_elapsed else None,
        },
    }


def build_costs(ledger: Sequence[LedgerData], measures_by_arm: Mapping[str, Sequence[dict]],
                cases: Sequence[CaseData]) -> dict:
    cases_by_id = {case.case_id: case for case in cases}
    by_arm: dict[str, dict] = {}
    by_arm_split: dict[str, dict] = {}
    by_family: list[dict] = []
    total_rows: list[LedgerData] = []
    for arm in ARMS:
        arm_rows = [row for row in ledger if row.arm == arm]
        total_rows.extend(arm_rows)
        all_block = cost_block(arm_rows, measures_by_arm[arm])
        by_arm[arm] = {"all": all_block}
        by_arm_split[arm] = {}
        for split in SPLITS:
            subset = [row for row in arm_rows if row.split == split]
            subset_measures = [m for m in measures_by_arm[arm] if m["case"].split == split]
            block = cost_block(subset, subset_measures)
            by_arm[arm][split] = block
            by_arm_split[arm][split] = block
            names = sorted({case.family for case in cases if case.split == split})
            for family in names:
                family_cases = {case.case_id for case in cases
                                if case.split == split and case.family == family}
                rows = [row for row in subset if row.case_id in family_cases]
                fam_measures = [m for m in subset_measures
                                if m["case"].family == family]
                fam_block = cost_block(rows, fam_measures)
                by_family.append({"arm": arm, "split": split, "family": family,
                                  "cases": fam_block["cases"],
                                  "calls": fam_block["calls"],
                                  "tokens_total": fam_block["tokens"]["total"],
                                  "elapsed_seconds_total": fam_block["elapsed_seconds_total"],
                                  "case_latency_p50": fam_block["case_latency_seconds"]["p50"],
                                  "case_latency_p95": fam_block["case_latency_seconds"]["p95"]})
    total_block = cost_block(total_rows, [m for arm in ARMS for m in measures_by_arm[arm]])
    return {
        "definition": ("cost from call-ledger per case+arm over all attempts; "
                       "case_latency_seconds = sum(elapsed_s over attempts of that case+arm; "
                       "p50/p95 computed across cases"),
        "by_arm": by_arm,
        "by_arm_split": by_arm_split,
        "by_arm_split_family": by_family,
        "total": total_block,
        "budget": {
            "max_calls": MAX_CALLS,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "calls_remaining": MAX_CALLS - total_block["calls"],
            "tokens_remaining": MAX_TOTAL_TOKENS - total_block["tokens"]["total"],
            "a2_calls": by_arm["A2"]["all"]["calls"],
            "a2_tokens": by_arm["A2"]["all"]["tokens"]["total"],
            "a2_not_run_cases": sum(1 for m in measures_by_arm["A2"]
                                    if m["raw"].not_run),
            "a2_extra_call_per_case": True,
        },
    }


                                                    


def a2_breakdown(measures: Sequence[dict]) -> dict:
    total_gold = sum(len(measure["case"].gold) for measure in measures)
    groups: dict[str, dict] = {
        name: {"candidate_count": 0, "kept_atom_count": 0,
               "core_covered_gold": set(), "fully_covered_gold": set(),
               "non_atomic_atom_count": 0, "cases": set()}
        for name in ("kept", "injection", "evidence", "unknown")
    }
    for measure in measures:
        case = measure["case"]
        raw = measure["raw"]
        for candidate in raw.candidates:
            if candidate.parent_p0_candidate_id == "unknown":
                group_name = "unknown"
            else:
                group_name = candidate.parent_gate_status or "unknown"
            group = groups.setdefault(group_name, {
                "candidate_count": 0, "kept_atom_count": 0,
                "core_covered_gold": set(), "fully_covered_gold": set(),
                "non_atomic_atom_count": 0, "cases": set()})
            group["candidate_count"] += 1
            group["cases"].add(case.case_id)
            if candidate.kept_status != "kept":
                continue
            atom = measure["annotation"].atom_by_id[candidate.candidate_id]
            group["kept_atom_count"] += 1
            for gold_id in atom.core_entailed_gold_ids:
                group["core_covered_gold"].add((case.case_id, gold_id))
            for gold_id in atom.fully_covered_gold_ids:
                group["fully_covered_gold"].add((case.case_id, gold_id))
            if atom.non_atomic:
                group["non_atomic_atom_count"] += 1
    output: dict[str, dict] = {}
    for name, group in groups.items():
        output[name] = {
            "candidate_count": group["candidate_count"],
            "kept_atom_count": group["kept_atom_count"],
            "core_covered_gold": len(group["core_covered_gold"]),
            "fully_covered_gold": len(group["fully_covered_gold"]),
            "core_covered_gold_rate": ((float(len(group["core_covered_gold"])) / total_gold)
                                       if total_gold else None),
            "fully_covered_gold_rate": ((float(len(group["fully_covered_gold"])) / total_gold)
                                        if total_gold else None),
            "non_atomic_atom_count": group["non_atomic_atom_count"],
            "cases": len(group["cases"]),
        }
    return {
        "by_parent_gate_status": output,
        "gold_total": total_gold,
        "pure_atomization_atoms": output["kept"]["kept_atom_count"],
        "evidence_repair_atoms": output["evidence"]["kept_atom_count"],
        "injection_text_change_atoms": output["injection"]["kept_atom_count"],
        "unknown_lineage_atoms": output["unknown"]["kept_atom_count"],
        "unknown_lineage_candidates": output["unknown"]["candidate_count"],
        "note": ("parent_gate_status=kept 才可称纯原子化；evidence/injection 是证据修复或文本变化，"
                 "unknown 不猜测归属。"),
    }


                                                


def gate_item(item_id: str, label: str, passed: bool, value: Any,
              threshold: Any, comparator: str, reason: str) -> dict:
    return {"id": item_id, "label": label, "pass": bool(passed),
            "status": "pass" if passed else "fail",
            "value": value, "threshold": threshold,
            "comparator": comparator, "reason": reason}


def metric_view(bundle: dict, split: str, arm: str, view: str = "direct") -> dict:
    return value_of(bundle, arm, "by_split", split, view, default={}) or {}


def metric_value(node: dict, *keys: str) -> Any:
    return value_of(node, *keys)


def compare_not_increase(new: Any, base: Any, label: str) -> tuple[bool, str]:
    if new is None and base is None:
        return True, f"{label}: 两侧均未定义（无相应分母），按不增加处理"
    if new is None or base is None:
        return False, f"{label}: 比较值未定义（new={new}, base={base}）"
    passed = float(new) <= float(base) + 1e-12
    return passed, f"{label}: {new:.6f} vs {base:.6f}"


def relative_reduction(new: Any, base: Any) -> float | None:
    if new is None or base is None or float(base) == 0.0:
        return None
    return (float(base) - float(new)) / float(base)


def evaluate_a1_gate(bundles: Mapping[str, dict]) -> dict:
    conf = bundles["A1"]["by_split"]["confirmation"]
    p0_conf = bundles["P0"]["by_split"]["confirmation"]
    direct = conf["direct"]
    p0_direct = p0_conf["direct"]
    items: list[dict] = []

    coverage = value_of(direct, "source_coverage", "value")
    items.append(gate_item(
        "confirmation_coverage", "confirmation source_coverage", 
        coverage is not None and coverage >= THRESHOLDS["coverage"], coverage,
        THRESHOLDS["coverage"], ">=",
        f"coverage={coverage}；要求 >= {THRESHOLDS['coverage']}"))
    faithfulness = value_of(direct, "faithfulness", "value")
    items.append(gate_item(
        "confirmation_faithfulness", "confirmation faithfulness",
        faithfulness is not None and faithfulness >= THRESHOLDS["faithfulness"],
        faithfulness, THRESHOLDS["faithfulness"], ">=",
        f"faithfulness={faithfulness}；要求 >= {THRESHOLDS['faithfulness']}"))
    complete = value_of(direct, "complete_set", "rate")
    items.append(gate_item(
        "confirmation_complete_set", "confirmation faithful_complete_set_success",
        complete is not None and complete >= THRESHOLDS["complete_set"], complete,
        THRESHOLDS["complete_set"], ">=",
        f"complete_set={complete}；要求 >= {THRESHOLDS['complete_set']}"))
    non_atomic = value_of(direct, "non_atomic", "obligation_rate")
    items.append(gate_item(
        "confirmation_non_atomic_obligation", "confirmation non_atomic_obligation_rate",
        non_atomic is not None and non_atomic <= THRESHOLDS["non_atomic_obligation"],
        non_atomic, THRESHOLDS["non_atomic_obligation"], "<=",
        f"non_atomic={non_atomic}；要求 <= {THRESHOLDS['non_atomic_obligation']}"))

    p0_non = value_of(p0_direct, "non_atomic", "obligation_rate")
    if p0_non is None or non_atomic is None:
        reduction = None
        rel_pass = False
        rel_reason = "P0/A1 non_atomic 未定义，无法计算相对改善"
    elif p0_non < THRESHOLDS["non_atomic_obligation"]:
        reduction = None
        rel_pass = True
        rel_reason = (f"P0 non_atomic={p0_non:.6f} < {THRESHOLDS['non_atomic_obligation']}，"
                      "按设计不要求相对降幅（P0=0 时相对改善为 null）")
    else:
        reduction = relative_reduction(non_atomic, p0_non)
        rel_pass = reduction is not None and reduction >= THRESHOLDS["non_atomic_relative_reduction"]
        rel_reason = (f"P0={p0_non:.6f}, A1={non_atomic:.6f}，相对下降={reduction}；"
                      f"要求 >= {THRESHOLDS['non_atomic_relative_reduction']}")
    items.append(gate_item(
        "non_atomic_relative_reduction", "non_atomic relative reduction vs P0",
        rel_pass, reduction, THRESHOLDS["non_atomic_relative_reduction"], ">=", rel_reason))

    p0_cov = value_of(p0_direct, "source_coverage", "value")
    if coverage is None or p0_cov is None:
        drop = None
        drop_pass = False
        drop_reason = "coverage 未定义"
    else:
        drop = (p0_cov - coverage) * 100.0
        drop_pass = drop <= THRESHOLDS["coverage_drop_pp"] + 1e-12
        drop_reason = f"P0={p0_cov:.6f}, A1={coverage:.6f}，下降={drop:.6f}pp；要求 <= 2pp"
    items.append(gate_item(
        "coverage_drop_vs_p0", "coverage drop vs P0 (pp)", drop_pass, drop,
        THRESHOLDS["coverage_drop_pp"], "<=", drop_reason))

    unsupported = value_of(direct, "faithfulness", "unsupported_rate")
    p0_unsupported = value_of(p0_direct, "faithfulness", "unsupported_rate")
    if unsupported is None or p0_unsupported is None:
        unsup_inc = None
        unsup_pass = False
        unsup_reason = "unsupported rate 未定义"
    else:
        unsup_inc = (unsupported - p0_unsupported) * 100.0
        unsup_pass = unsup_inc <= THRESHOLDS["unsupported_increment_pp"] + 1e-12
        unsup_reason = (f"P0={p0_unsupported:.6f}, A1={unsupported:.6f}，增量={unsup_inc:.6f}pp；"
                        f"要求 <= 1pp")
    items.append(gate_item(
        "unsupported_increment_vs_p0", "unsupported increment vs P0 (pp)", unsup_pass,
        unsup_inc, THRESHOLDS["unsupported_increment_pp"], "<=", unsup_reason))

    context_new = value_of(direct, "context_loss", "rate")
    context_base = value_of(p0_direct, "context_loss", "rate")
    pass_ctx, reason_ctx = compare_not_increase(context_new, context_base, "context_loss")
    items.append(gate_item("context_loss_not_increased", "context loss not increased",
                           pass_ctx, context_new, context_base, "<=", reason_ctx))

    persistence_new = value_of(direct, "false_persistence", "rate")
    persistence_base = value_of(p0_direct, "false_persistence", "rate")
    pass_pers, reason_pers = compare_not_increase(
        persistence_new, persistence_base, "false_persistence")
    items.append(gate_item("false_persistence_not_increased",
                           "false persistence not increased", pass_pers,
                           persistence_new, persistence_base, "<=", reason_pers))

    duplicate = value_of(direct, "duplicate", "atom_rate")
    duplicate_base = value_of(p0_direct, "duplicate", "atom_rate")
    items.append(gate_item(
        "duplicate_atom_absolute", "duplicate_atom_rate <= .02",
        duplicate is not None and duplicate <= THRESHOLDS["duplicate_atom"],
        duplicate, THRESHOLDS["duplicate_atom"], "<=",
        f"duplicate={duplicate}；要求 <= .02"))
    pass_dup, reason_dup = compare_not_increase(duplicate, duplicate_base, "duplicate_atom")
    items.append(gate_item("duplicate_atom_not_above_p0", "duplicate_atom_rate <= P0",
                           pass_dup, duplicate, duplicate_base, "<=", reason_dup))

    over_split = value_of(direct, "over_split", "atom_rate")
    over_base = value_of(p0_direct, "over_split", "atom_rate")
    items.append(gate_item(
        "over_split_absolute", "over_split_rate <= .01",
        over_split is not None and over_split <= THRESHOLDS["over_split"],
        over_split, THRESHOLDS["over_split"], "<=",
        f"over_split={over_split}；要求 <= .01"))
    pass_over, reason_over = compare_not_increase(over_split, over_base, "over_split")
    items.append(gate_item("over_split_not_above_p0", "over_split_rate <= P0",
                           pass_over, over_split, over_base, "<=", reason_over))

    severe = value_of(direct, "forbidden", "severe_case_count")
    items.append(gate_item("severe_forbidden_zero", "severe forbidden inference cases = 0",
                           severe == 0, severe, 0, "==",
                           f"severe cases={severe}；要求为 0"))
    inversion = value_of(direct, "inversion", "atom_count")
    items.append(gate_item("polarity_condition_inversion_zero",
                           "polarity/condition inversion atoms = 0",
                           inversion == 0, inversion, 0, "==",
                           f"inversion atoms={inversion}；要求为 0"))

    dev = bundles["A1"]["by_split"]["dev"]
    p0_dev = bundles["P0"]["by_split"]["dev"]
    dev_non = value_of(dev, "direct", "non_atomic", "obligation_rate")
    p0_dev_non = value_of(p0_dev, "direct", "non_atomic", "obligation_rate")
    if dev_non is None or p0_dev_non is None or non_atomic is None or p0_non is None:
        direction_pass = False
        direction_reason = (f"dev/confirmation non_atomic 未定义（dev A1={dev_non}, "
                            f"P0={p0_dev_non}; conf A1={non_atomic}, P0={p0_non}）")
        direction_value = None
    else:
        direction_pass = dev_non <= p0_dev_non + 1e-12 and non_atomic <= p0_non + 1e-12
        direction_value = {"dev_delta": dev_non - p0_dev_non,
                           "confirmation_delta": non_atomic - p0_non}
        direction_reason = (f"dev Δ={dev_non - p0_dev_non:.6f}, "
                            f"confirmation Δ={non_atomic - p0_non:.6f}；两处均不得变差")
    items.append(gate_item("non_atomic_direction_consistent",
                           "non-atomic direction consistent dev/confirmation",
                           direction_pass, direction_value, 0.0, "<=", direction_reason))

                                                       
                                                            
    case_rows = {row["case_id"]: row for row in bundles["A1"]["all"]["case_rows"]}
    p0_rows = {row["case_id"]: row for row in bundles["P0"]["all"]["case_rows"]}
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"a1": 0, "p0": 0, "cases": 0})
    for case_id, row in case_rows.items():
        key = (row["split"], row["family"])
        groups[key]["a1"] += row["kept_count"]
        groups[key]["p0"] += p0_rows.get(case_id, {}).get("kept_count", 0)
        groups[key]["cases"] += 1
    loss_groups = sorted(f"{split}/{family}" for (split, family), value in groups.items()
                         if value["a1"] == 0 and value["p0"] > 0)
    cap_loss_cases = sorted(
        row["case_id"] for case_id, row in case_rows.items()
        if row["cap_applied"] and row["candidate_count"] > 0 and row["kept_count"] == 0
        and p0_rows.get(case_id, {}).get("kept_count", 0) > 0)
    systematic_pass = not loss_groups and not cap_loss_cases
    systematic_reason = (f"整组丢失={loss_groups[:8]}；8 条上限整组丢失 case={cap_loss_cases[:8]}"
                         if not systematic_pass else "无 family/上限导致的系统性整组丢失")
    items.append(gate_item("no_systematic_cap_or_gate_loss",
                           "no systematic cap/gate group loss", systematic_pass,
                           {"loss_groups": loss_groups, "cap_loss_cases": cap_loss_cases},
                           [], "==", systematic_reason))

    failed = [item["id"] for item in items if not item["pass"]]
    return {"passed": not failed, "failed_items": failed, "items": items,
            "split": "confirmation", "view": "direct-kept"}


def evaluate_a2_gate(bundles: Mapping[str, dict], a1_gate: Mapping[str, Any]) -> dict:
    conf = bundles["A2"]["by_split"]["confirmation"]
    direct = conf["direct"]
    items: list[dict] = []
    coverage = value_of(direct, "source_coverage", "value")
    items.append(gate_item("confirmation_coverage", "confirmation source_coverage",
                           coverage is not None and coverage >= THRESHOLDS["coverage"],
                           coverage, THRESHOLDS["coverage"], ">=",
                           f"coverage={coverage}；要求 >= .90"))
    faithfulness = value_of(direct, "faithfulness", "value")
    items.append(gate_item("confirmation_faithfulness", "confirmation faithfulness",
                           faithfulness is not None and faithfulness >= THRESHOLDS["faithfulness"],
                           faithfulness, THRESHOLDS["faithfulness"], ">=",
                           f"faithfulness={faithfulness}；要求 >= .95"))
    complete = value_of(direct, "complete_set", "rate")
    items.append(gate_item("confirmation_complete_set",
                           "confirmation faithful_complete_set_success",
                           complete is not None and complete >= THRESHOLDS["complete_set"],
                           complete, THRESHOLDS["complete_set"], ">=",
                           f"complete_set={complete}；要求 >= .60"))
    non_atomic = value_of(direct, "non_atomic", "obligation_rate")
    items.append(gate_item("confirmation_non_atomic_obligation",
                           "confirmation non_atomic_obligation_rate",
                           non_atomic is not None
                           and non_atomic <= THRESHOLDS["non_atomic_obligation"],
                           non_atomic, THRESHOLDS["non_atomic_obligation"], "<=",
                           f"non_atomic={non_atomic}；要求 <= .10"))
    duplicate = value_of(direct, "duplicate", "atom_rate")
    items.append(gate_item("duplicate_atom_absolute", "duplicate_atom_rate <= .02",
                           duplicate is not None and duplicate <= THRESHOLDS["duplicate_atom"],
                           duplicate, THRESHOLDS["duplicate_atom"], "<=",
                           f"duplicate={duplicate}；要求 <= .02"))
    over_split = value_of(direct, "over_split", "atom_rate")
    items.append(gate_item("over_split_absolute", "over_split_rate <= .01",
                           over_split is not None and over_split <= THRESHOLDS["over_split"],
                           over_split, THRESHOLDS["over_split"], "<=",
                           f"over_split={over_split}；要求 <= .01"))
    severe = value_of(direct, "forbidden", "severe_case_count")
    items.append(gate_item("severe_forbidden_zero", "severe forbidden inference cases = 0",
                           severe == 0, severe, 0, "==",
                           f"severe cases={severe}；要求为 0"))
    inversion = value_of(direct, "inversion", "atom_count")
    items.append(gate_item("polarity_condition_inversion_zero",
                           "polarity/condition inversion atoms = 0",
                           inversion == 0, inversion, 0, "==",
                           f"inversion atoms={inversion}；要求为 0"))

    a1_direct = bundles["A1"]["by_split"]["confirmation"]["direct"]
    a1_cov = value_of(a1_direct, "source_coverage", "value")
    a1_non = value_of(a1_direct, "non_atomic", "obligation_rate")
    if coverage is None or a1_cov is None:
        gain = None
        gain_pass = False
        gain_reason = "A1/A2 coverage 未定义"
    else:
        gain = (coverage - a1_cov) * 100.0
        gain_pass = gain >= THRESHOLDS["a2_coverage_gain_pp"] - 1e-12
        gain_reason = f"A1={a1_cov:.6f}, A2={coverage:.6f}, 增益={gain:.6f}pp；要求 >= +3pp"
    items.append(gate_item("a2_coverage_gain_vs_a1", "A2 coverage gain vs A1 (pp)",
                           gain_pass, gain, THRESHOLDS["a2_coverage_gain_pp"], ">=",
                           gain_reason))

    reduction = relative_reduction(non_atomic, a1_non)
    reduction_pass = (reduction is not None
                      and reduction >= THRESHOLDS["a2_non_atomic_relative_reduction"])
    if a1_non is None:
        reduction_reason = "A1 non_atomic 未定义，相对下降为 null"
    elif float(a1_non) == 0.0:
        reduction_reason = "A1 non_atomic=0，相对下降为 null（不写无限改善）"
    else:
        reduction_reason = (f"A1={a1_non:.6f}, A2={non_atomic:.6f}，相对下降={reduction}；"
                            f"要求 >= 25%")
    items.append(gate_item("a2_non_atomic_reduction_vs_a1",
                           "A2 non-atomic relative reduction vs A1",
                           reduction_pass, reduction,
                           THRESHOLDS["a2_non_atomic_relative_reduction"], ">=",
                           reduction_reason))

    improvement_pass = gain_pass or reduction_pass
    improvement_reason = (f"coverage 增益 pass={gain_pass}；non-atomic 相对下降 pass="
                          f"{reduction_pass}；满足任一即可进入成本评估")
    items.append(gate_item("a2_minimum_improvement", "A2 minimum quality improvement",
                           improvement_pass, {"coverage_gain_pp": gain,
                                              "non_atomic_reduction": reduction},
                           "coverage>=+3pp OR non_atomic>=25%", "OR",
                           improvement_reason))

    absolute_ids = {"confirmation_coverage", "confirmation_faithfulness",
                    "confirmation_complete_set", "confirmation_non_atomic_obligation",
                    "duplicate_atom_absolute", "over_split_absolute",
                    "severe_forbidden_zero", "polarity_condition_inversion_zero"}
    absolute_pass = all(item["pass"] for item in items if item["id"] in absolute_ids)
    passed = absolute_pass and improvement_pass
    blocking_ids = absolute_ids | {"a2_minimum_improvement"}
    failed = [item["id"] for item in items
              if not item["pass"] and item["id"] in blocking_ids]
    non_blocking = [item["id"] for item in items
                    if not item["pass"] and item["id"] not in blocking_ids]
    return {"passed": passed, "decision": "enter_cost_evaluation" if passed else "reject",
            "absolute_gate_passed": absolute_pass,
            "minimum_improvement_passed": improvement_pass,
            "failed_items": failed, "non_blocking_failed_items": non_blocking,
            "items": items, "split": "confirmation",
            "view": "direct-kept",
            "a1_gate_passed": bool(a1_gate["passed"])}


                                                    


def validate_a2_lineage(raw_sets: Mapping[tuple[str, str], RawData],
                        cases: Sequence[CaseData]) -> tuple[list[str], dict]:
    warnings: list[str] = []
    rule_counts: Counter = Counter()
    unknown_candidates = 0
    for case in cases:
        p0 = raw_sets[(case.case_id, "P0")]
        a2 = raw_sets[(case.case_id, "A2")]
        if a2.p0_attempt_sha256 is not None:
            require(a2.p0_attempt_sha256 == p0.attempt_sha256, "a2_top_level_parent_mismatch",
                    f"{case.case_id}: A2 顶层 p0_attempt_sha256 与当前 P0 attempt 不匹配")
        if a2.p0_content_sha256 is not None:
            require(a2.p0_content_sha256 == p0.content_sha256, "a2_top_level_parent_mismatch",
                    f"{case.case_id}: A2 顶层 p0_content_sha256 与当前 P0 content 不匹配")
        for candidate in a2.candidates:
            parent_id = candidate.parent_p0_candidate_id
            if parent_id == "unknown":
                unknown_candidates += 1
                require(candidate.parent_content_sha is None
                        and candidate.parent_p0_attempt_sha is None
                        and candidate.parent_gate_status == "unknown", "a2_parent_unknown_shape",
                        f"{case.case_id}/{candidate.candidate_id}: unknown lineage 形状非法")
                continue
            parent = p0.candidate_by_id.get(parent_id)
            require(parent is not None, "a2_parent_not_in_p0",
                    f"{case.case_id}/{candidate.candidate_id}: 父 candidate {parent_id} "
                    "不在当前 P0 输出中")
            expected_content = candidate_data_content_sha(parent)
            require(candidate.parent_content_sha == expected_content,
                    "a2_parent_content_mismatch",
                    f"{case.case_id}/{candidate.candidate_id}: parent_content_sha 不匹配"
                    f"（expected {expected_content[:12]}…）")
            require(candidate.parent_p0_attempt_sha == p0.attempt_sha256,
                    "a2_parent_attempt_mismatch",
                    f"{case.case_id}/{candidate.candidate_id}: parent_p0_attempt_sha "
                    "与当前 P0 attempt 不匹配")
            require(candidate.parent_gate_status == parent.kept_status,
                    "a2_parent_gate_mismatch",
                    f"{case.case_id}/{candidate.candidate_id}: parent_gate_status="
                    f"{candidate.parent_gate_status!r} 与 P0 父状态 {parent.kept_status!r} 不匹配")
            rule_counts["official_content_sha"] += 1
    if unknown_candidates:
        warnings.append(f"A2 有 {unknown_candidates} 个候选为 unknown lineage（不猜测归属）")
    return warnings, {"known_parent_content_rule": "content_sha([candidate_view(parent)])",
                      "known_parent_attempt_rule": "P0 record attempt_sha256",
                      "known_parents_checked": sum(rule_counts.values()),
                      "unknown_lineage_candidates": unknown_candidates}


def validate_ledger_coverage(raw_sets: Mapping[tuple[str, str], RawData],
                             ledger: Sequence[LedgerData]) -> tuple[list[str], dict]:
    ledger_by_attempt = {(row.case_id, row.arm, row.attempt): row for row in ledger}
    missing: list[str] = []
    mismatch: list[str] = []
    skipped_not_run = 0
    for key, raw in sorted(raw_sets.items()):
        if raw.not_run:
            skipped_not_run += 1
            continue
        row = ledger_by_attempt.get((raw.case_id, raw.arm, raw.attempt))
        if row is None:
            missing.append(f"{raw.case_id}/{raw.arm}#{raw.attempt}")
            continue
        raw_total = raw.usage.get("total_tokens")
        ledger_total = row.usage.get("total_tokens")
        if raw_total is not None and ledger_total is not None and raw_total != ledger_total:
            mismatch.append(f"{raw.case_id}/{raw.arm}#{raw.attempt}"
                            f"({raw_total}!={ledger_total})")
    require(not missing, "ledger_missing_attempt",
            f"ledger 缺少以下最新 raw attempt 的成本行：{missing[:12]}"
            + (" …" if len(missing) > 12 else ""))
    warnings: list[str] = []
    if mismatch:
        warnings.append(f"ledger 与 raw 的 total_tokens 有 {len(mismatch)} 处不一致；"
                        "成本以 ledger 为准")
    return warnings, {"latest_attempts_checked": len(raw_sets) - skipped_not_run,
                      "not_run_skipped": skipped_not_run,
                      "usage_mismatch": mismatch[:20],
                      "usage_mismatch_count": len(mismatch)}


                                                


def score_bundle(case_rows: Sequence[dict],
                 raw_rows_by_arm: Mapping[str, Sequence[dict]],
                 ledger_rows: Sequence[dict],
                 annotation_rows: Sequence[dict],
                 *,
                 input_sha256: Mapping[str, dict],
                 script_sha256: str,
                 synthetic_only: bool = False) -> dict:
    cases = validate_cases(case_rows)
    cases_by_id = {case.case_id: case for case in cases}
    raw_sets, raw_warnings = load_raw_sets(raw_rows_by_arm, cases)
    not_run_cases = sorted(raw.case_id for raw in raw_sets.values() if raw.not_run)
    annotations, annotation_warnings = load_annotations(annotation_rows, raw_sets, cases)
    ledger, ledger_warnings = load_ledger(ledger_rows, cases)
    lineage_warnings, lineage_info = validate_a2_lineage(raw_sets, cases)
    ledger_warnings2, ledger_info = validate_ledger_coverage(raw_sets, ledger)

    measures_by_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        measures = [measure_case(case, raw_sets[(case.case_id, arm)],
                                 annotations[(case.case_id, arm)]) for case in cases]
        measures_by_arm[arm] = measures
    bundles, stratified = build_arm_bundles(measures_by_arm, cases)
    costs = build_costs(ledger, measures_by_arm, cases)
    a2_all = a2_breakdown(measures_by_arm["A2"])
    a2_by_split = {
        split: a2_breakdown([m for m in measures_by_arm["A2"] if m["case"].split == split])
        for split in SPLITS
    }

    a1_gate = evaluate_a1_gate(bundles)
    a2_gate = evaluate_a2_gate(bundles, a1_gate)

    a1_direct = bundles["A1"]["by_split"]["confirmation"]["direct"]
    p0_direct = bundles["P0"]["by_split"]["confirmation"]["direct"]
    a2_direct = bundles["A2"]["by_split"]["confirmation"]["direct"]
    a1_loop = bundles["A1"]["by_split"]["confirmation"]["loop"]
    diagnostics = {
        "a1_vs_p0": {
            "coverage_delta_pp": (
                (value_of(a1_direct, "source_coverage", "value")
                 - value_of(p0_direct, "source_coverage", "value")) * 100.0
                if value_of(a1_direct, "source_coverage", "value") is not None
                and value_of(p0_direct, "source_coverage", "value") is not None else None),
            "faithfulness_delta": (
                (value_of(a1_direct, "faithfulness", "value")
                 - value_of(p0_direct, "faithfulness", "value"))
                if value_of(a1_direct, "faithfulness", "value") is not None
                and value_of(p0_direct, "faithfulness", "value") is not None else None),
            "non_atomic_delta": (
                (value_of(a1_direct, "non_atomic", "obligation_rate")
                 - value_of(p0_direct, "non_atomic", "obligation_rate"))
                if value_of(a1_direct, "non_atomic", "obligation_rate") is not None
                and value_of(p0_direct, "non_atomic", "obligation_rate") is not None else None),
        },
        "a2_vs_a1": {
            "coverage_delta_pp": (
                (value_of(a2_direct, "source_coverage", "value")
                 - value_of(a1_direct, "source_coverage", "value")) * 100.0
                if value_of(a2_direct, "source_coverage", "value") is not None
                and value_of(a1_direct, "source_coverage", "value") is not None else None),
            "non_atomic_relative_reduction": relative_reduction(
                value_of(a2_direct, "non_atomic", "obligation_rate"),
                value_of(a1_direct, "non_atomic", "obligation_rate")),
        },
        "loop_projection": {
            "direct_coverage": value_of(a1_direct, "source_coverage"),
            "loop_coverage": value_of(a1_loop, "source_coverage"),
            "rule_only_coverage": value_of(a1_loop, "rule_only_coverage"),
            "coverage_loss_pp": value_of(a1_loop, "memory_projection",
                                         "loop_projection_loss_pp"),
            "direct_complete_set": value_of(a1_direct, "complete_set"),
            "loop_complete_set": value_of(a1_loop, "complete_set"),
            "memory_projection": value_of(a1_loop, "memory_projection"),
        },
        "valid_output": {
            arm: value_of(bundles, arm, "by_split", "confirmation", "valid_output")
            for arm in ARMS
        },
        "a2_lineage": {
            "all": a2_all,
            "by_split": a2_by_split,
            "validation": lineage_info,
        },
        "a2_not_run": {
            "count": len(not_run_cases),
            "cases": not_run_cases,
            "reason": "P0 无候选时采集器按设计不运行 A2；评分按 valid_empty 空输出计入",
        },
        "cost": {
            "total_calls": costs["total"]["calls"],
            "total_tokens": costs["total"]["tokens"]["total"],
            "a2_calls": costs["budget"]["a2_calls"],
            "a2_tokens": costs["budget"]["a2_tokens"],
        },
    }

    counts = {
        "cases": len(cases),
        "dev": sum(1 for case in cases if case.split == "dev"),
        "confirmation": sum(1 for case in cases if case.split == "confirmation"),
        "families": len({case.family for case in cases}),
        "clusters": len({case.cluster_id for case in cases}),
        "gold_total": sum(len(case.gold) for case in cases),
        "negative_controls": sum(1 for case in cases if not case.gold),
        "raw_sets": len(raw_sets),
        "input_raw_sets": len(raw_sets),
        "call_raw_sets": len(raw_sets) - sum(1 for raw in raw_sets.values()
                                             if raw.not_run),
        "a2_not_run": sum(1 for raw in raw_sets.values() if raw.not_run),
        "annotations": len(annotation_rows),
        "matched_output_sets": len(annotations),
        "annotation_atoms": sum(len(annotation.atoms)
                                for annotation in annotations.values()),
        "ledger_rows": len(ledger),
    }
    required_inputs = {"cases-v1.jsonl", "raw-P0.jsonl", "raw-A1.jsonl",
                       "raw-A2.jsonl", "call-ledger.jsonl", "annotations-v1.jsonl"}
    require(set(input_sha256) == required_inputs, "result_input_sha",
            f"input_sha256 须恰含 6 个输入（缺 {sorted(required_inputs - set(input_sha256))}，"
            f"多 {sorted(set(input_sha256) - required_inputs)}）")
    for name, meta in input_sha256.items():
        require(isinstance(meta, dict) and is_hex64(meta.get("sha256")), "result_input_sha",
                f"input_sha256[{name}] 须含 64 hex sha256")
    require(is_hex64(script_sha256), "result_script_sha", "script_sha256 须为 64 hex")
    warnings = (raw_warnings + annotation_warnings + ledger_warnings
                + lineage_warnings + ledger_warnings2)
    if costs["total"]["calls"] > MAX_CALLS:
        warnings.append(f"ledger 调用数 {costs['total']['calls']} 超过硬上限 {MAX_CALLS}")
    if costs["total"]["tokens"]["total"] > MAX_TOTAL_TOKENS:
        warnings.append(f"ledger total tokens {costs['total']['tokens']['total']} "
                        f"超过硬上限 {MAX_TOTAL_TOKENS}")
    return {
        "schema": SCHEMA_RESULT,
        "synthetic_only": bool(synthetic_only),
        "thresholds": dict(THRESHOLDS),
        "counts": counts,
        "inputs": {name: dict(meta) for name, meta in input_sha256.items()},
        "input_sha256": {name: meta["sha256"] for name, meta in input_sha256.items()},
        "script_sha256": script_sha256,
        "arms": bundles,
        "stratified_by_arm_split_family": stratified,
        "a2_parent_breakdown": {"all": a2_all, "by_split": a2_by_split},
        "costs": costs,
        "gates": {"A1": a1_gate, "A2": a2_gate},
        "diagnostics": diagnostics,
        "validation": {"lineage": lineage_info, "ledger": ledger_info,
                       "a2_not_run_cases": not_run_cases,
                       "warnings": warnings},
    }


def result_summary(result: dict) -> dict:
    gates = result["gates"]
    def view(arm: str, split: str, name: str) -> Any:
        return value_of(result, "arms", arm, "by_split", split, name)
    return {
        "schema": result["schema"],
        "cases": result["counts"]["cases"],
        "dev": result["counts"]["dev"],
        "confirmation": result["counts"]["confirmation"],
        "gold_total": result["counts"]["gold_total"],
        "a2_not_run": result["counts"]["a2_not_run"],
        "A1_confirmation": {
            "coverage": view("A1", "confirmation", "direct")["source_coverage"],
            "faithfulness": view("A1", "confirmation", "direct")["faithfulness"],
            "complete_set": view("A1", "confirmation", "direct")["complete_set"],
            "non_atomic": view("A1", "confirmation", "direct")["non_atomic"]["obligation_rate"],
            "loop_coverage": view("A1", "confirmation", "loop")["source_coverage"],
            "loop_complete_set": view("A1", "confirmation", "loop")["complete_set"],
        },
        "A2_confirmation": {
            "coverage": view("A2", "confirmation", "direct")["source_coverage"],
            "complete_set": view("A2", "confirmation", "direct")["complete_set"],
            "non_atomic": view("A2", "confirmation", "direct")["non_atomic"]["obligation_rate"],
        },
        "gates": {
            "A1": {"passed": gates["A1"]["passed"],
                   "failed_items": gates["A1"]["failed_items"]},
            "A2": {"passed": gates["A2"]["passed"],
                   "decision": gates["A2"]["decision"],
                   "absolute_gate_passed": gates["A2"]["absolute_gate_passed"],
                   "minimum_improvement_passed": gates["A2"]["minimum_improvement_passed"],
                   "failed_items": gates["A2"]["failed_items"],
                   "non_blocking_failed_items":
                       gates["A2"].get("non_blocking_failed_items", [])},
        },
        "cost": {
            "calls": result["costs"]["total"]["calls"],
            "tokens": result["costs"]["total"]["tokens"]["total"],
            "a2_calls": result["costs"]["budget"]["a2_calls"],
        },
        "warnings": result["validation"]["warnings"],
    }


                                                      


def load_inputs() -> dict:
    cases_rows, cases_meta = read_jsonl_file(CASES_PATH)
    input_meta = {"cases-v1.jsonl": cases_meta}
    raw_rows: dict[str, list[dict]] = {}
    for arm in ARMS:
        rows, meta = read_jsonl_file(RAW_PATHS[arm])
        raw_rows[arm] = rows
        input_meta[f"raw-{arm}.jsonl"] = meta
    ledger_rows, ledger_meta = read_jsonl_file(LEDGER_PATH)
    input_meta["call-ledger.jsonl"] = ledger_meta
    annotation_rows, annotation_meta = read_jsonl_file(ANNOTATIONS_PATH)
    input_meta["annotations-v1.jsonl"] = annotation_meta
    return {"cases": cases_rows, "raw": raw_rows, "ledger": ledger_rows,
            "annotations": annotation_rows, "input_sha256": input_meta,
            "script_sha256": sha256_file(Path(__file__))}


def command_score() -> int:
    inputs = load_inputs()
    result = score_bundle(inputs["cases"], inputs["raw"], inputs["ledger"],
                          inputs["annotations"],
                          input_sha256=inputs["input_sha256"],
                          script_sha256=inputs["script_sha256"],
                          synthetic_only=False)
    require(DATA.exists(), "output_dir_missing", f"输出目录不存在：{DATA}")
    payload = pretty(result) + "\n"
    try:
        with open(RESULTS_PATH, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
    except OSError as exc:
        raise ScoreError("output_write_failed",
                         f"写入 {RESULTS_PATH.name} 失败：{type(exc).__name__}") from None
    summary = result_summary(result)
    summary["ok"] = True
    summary["output"] = str(RESULTS_PATH)
    summary["results_sha256"] = sha256_file(RESULTS_PATH)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0



                                                         


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> None:
        self.rows.append({"name": name, "pass": bool(passed),
                          **({"detail": detail} if not passed else {})})

    def eq(self, name: str, actual: Any, expected: Any) -> None:
        self.add(name, actual == expected, {"actual": actual, "expected": expected})

    def close(self, name: str, actual: Any, expected: float, tolerance: float = 1e-9) -> None:
        ok = (actual is not None and isinstance(actual, (int, float))
              and abs(float(actual) - float(expected)) <= tolerance)
        self.add(name, ok, {"actual": actual, "expected": expected})

    def none(self, name: str, actual: Any) -> None:
        self.add(name, actual is None, {"actual": actual})

    def truthy(self, name: str, actual: Any) -> None:
        self.add(name, bool(actual), {"actual": actual})

    def raises(self, name: str, code: str, fn) -> None:
        try:
            fn()
        except ScoreError as exc:
            self.add(name, exc.code == code, {"code": exc.code, "message": exc.message,
                                              "expected_code": code})
        except Exception as exc:                               
            self.add(name, False, {"unexpected": type(exc).__name__, "message": str(exc)})
        else:
            self.add(name, False, {"raised": None, "expected_code": code})

    def report(self) -> bool:
        failed = [row for row in self.rows if not row["pass"]]
        print(json.dumps({"check": "atomic-memory-score-selftest",
                          "passed": not failed, "checks": len(self.rows),
                          "failed": failed[:20], "failed_count": len(failed)},
                         ensure_ascii=False, sort_keys=True))
        return not failed


def fixture_case(case_id: str, split: str, gold_ids: Sequence[str] = ("g1",), *,
                 family: str = "fam-a", context_ids: Sequence[str] = (),
                 forbidden: Mapping[str, str] | None = None,
                 kind: str = "rule") -> dict:
    conversation = f"{case_id}：请按对话中的义务执行。"
    golds = []
    for gold_id in gold_ids:
        golds.append({
            "gold_id": gold_id,
            "kind": kind,
            "tier": ("explicit" if kind == "rule" else "not_applicable"),
            "meaning": f"{gold_id} 的原子义务",
            "evidence_conv_ids": [0],
            "evidence_spans": [{"conv_id": 0, "text": conversation}],
            "context_requirements": (["condition"] if gold_id in context_ids else []),
        })
    forbidden_rows = [{"forbidden_id": fid, "description": f"禁止推断 {fid}",
                       "severity": severity}
                      for fid, severity in (forbidden or {}).items()]
    return {
        "schema": SCHEMA_CASE, "case_id": case_id, "family": family,
        "cluster_id": f"{case_id}-cluster", "split": split,
        "conversations": [{"query": conversation, "chat": []}],
        "forbidden_inferences": forbidden_rows,
        "one_off_hints": [], "author_notes": "",
        "gold_obligations": golds,
    }


def fixture_item(candidate_id: str, *, tier: str | None = "explicit",
                 kept: str = "kept", core: Sequence[str] = (),
                 full: Sequence[str] | None = None,
                 entailment: str = "supported",
                 persistence: bool = True, non_atomic: bool = False,
                 over_split: bool = False,
                 context_errors: Sequence[str] = (),
                 forbidden: Sequence[str] = (),
                 duplicate_group: Any = None,
                 alternative: Any = None) -> tuple[dict, dict]:
    full_ids = list(core if full is None else full)
    candidate = {"candidate_id": candidate_id, "text": f"原子候选 {candidate_id}",
                 "summary": "", "evidence": ["证据片段"],
                 "kept_status": kept}
    if tier is not None:
        candidate["tier"] = tier
        candidate["trigger"] = "满足条件时"
    atom = {"candidate_id": candidate_id, "entailment": entailment,
            "persistence_appropriate": persistence, "non_atomic": non_atomic,
            "over_split": over_split, "context_errors": list(context_errors),
            "matched_forbidden_ids": list(forbidden),
            "duplicate_group": duplicate_group,
            "core_entailed_gold_ids": list(core),
            "fully_covered_gold_ids": full_ids,
            "alternative_partition": alternative}
    return candidate, atom


def fixture_raw(case: Mapping[str, Any], arm: str,
                items: Sequence[tuple[dict, dict]], *,
                status: str | None = None, attempt: int = 1,
                raw_item_count: int | None = None,
                usage: Mapping[str, Any] | None = None,
                elapsed: float = 1.0,
                p0_raw: Mapping[str, Any] | None = None,
                parent_ids: Sequence[str] | None = None,
                top_level_p0: bool = True) -> dict:
    candidates = [copy.deepcopy(item[0]) for item in items]
    atoms = [copy.deepcopy(item[1]) for item in items]
    require(len(candidates) == len(atoms), "fixture", "candidate/atom 数量不一致")
    if status is None:
        if not candidates:
            status = "valid_empty"
        else:
            kept_count = sum(1 for c in candidates if c["kept_status"] == "kept")
            status = ("ok" if kept_count == len(candidates)
                      else "all_gated" if kept_count == 0 else "partial_gated")
    if raw_item_count is None:
        raw_item_count = 0 if status in EMPTY_STATUSES and not candidates else len(candidates)
        if status == "empty_after_parse":
            raw_item_count = 1
    if status in {"llm_failure", "length", "content_filter"}:
        raw_text = None
    elif status == "invalid_json":
        raw_text = "not a json array"
    elif status == "valid_empty":
        raw_text = "[]"
    elif status == "empty_after_parse":
        raw_text = "raw nonempty but parser dropped"
    else:
        raw_text = f"raw-{case['case_id']}-{arm}"
    succeeded = status != "llm_failure"
    if usage is None:
        usage = ({"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                 if status == "llm_failure"
                 else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    finish_reason = {"length": "length", "content_filter": "content_filter",
                     "llm_failure": None}.get(status, "stop")
    record = {
        "schema": "atomic-pilot-raw-v1", "case_id": case["case_id"],
        "split": case["split"], "arm": arm, "attempt": attempt, "status": status,
        "prompt": f"prompt-{case['case_id']}-{arm}", "raw_text": raw_text,
        "raw_item_count": raw_item_count, "parsed_candidates": candidates,
        "usage": dict(usage), "elapsed_s": float(elapsed),
        "finish_reason": finish_reason,
        "call": {"attempted": True, "succeeded": succeeded},
    }
    record["content_sha256"] = content_sha(candidates)
    record["attempt_sha256"] = compute_attempt_sha(
        record, case["case_id"], arm, attempt, status, finish_reason, succeeded,
        sha256_text(record["prompt"]),
        sha256_text(raw_text) if isinstance(raw_text, str) and raw_text else None)
    record["blind_id"] = blind_id_of(case["case_id"], arm, attempt,
                                     record["content_sha256"])
    if arm == "A2":
        require(p0_raw is not None, "fixture", "A2 夹具必须有 P0 raw")
        p0_candidates = p0_raw.get("parsed_candidates") or []
        for index, candidate in enumerate(candidates):
            if parent_ids is not None:
                parent_id = parent_ids[index]
            elif p0_candidates:
                parent_id = p0_candidates[min(index, len(p0_candidates) - 1)]["candidate_id"]
            else:
                parent_id = "unknown"
            if parent_id == "unknown":
                candidate.update({"parent_p0_candidate_id": "unknown",
                                  "parent_content_sha": None,
                                  "parent_p0_attempt_sha": None,
                                  "parent_gate_status": "unknown"})
            else:
                parent = next((c for c in p0_candidates
                               if c["candidate_id"] == parent_id), None)
                require(parent is not None, "fixture", f"父 candidate {parent_id} 不存在")
                candidate.update({
                    "parent_p0_candidate_id": parent_id,
                    "parent_content_sha": content_sha([parent]),
                    "parent_p0_attempt_sha": p0_raw["attempt_sha256"],
                    "parent_gate_status": parent["kept_status"],
                })
        record["content_sha256"] = content_sha(candidates)
        record["blind_id"] = blind_id_of(case["case_id"], arm, attempt,
                                         record["content_sha256"])
        if top_level_p0:
            record["p0_attempt_sha256"] = p0_raw.get("attempt_sha256")
            record["p0_content_sha256"] = p0_raw.get("content_sha256")
    return {"raw": record, "atoms": atoms}


def fixture_annotation(raw: Mapping[str, Any], atoms: Sequence[dict]) -> dict:
    return {"blind_id": raw["blind_id"], "stage1_complete": True,
            "stage2_complete": True, "adjudicated": True,
            "atoms": [copy.deepcopy(atom) for atom in atoms]}


def build_bundle_parts(cases: Sequence[dict], specs: Sequence[dict],
                       *, ledger_extra: Sequence[dict] | None = None,
                       ledger_usage: Mapping[str, Any] | None = None,
                       ledger_elapsed: float | None = None) -> dict:
    raw_by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    annotations: list[dict] = []
    latest_raw: dict[tuple[str, str], dict] = {}
    spec_map: dict[tuple[str, str], dict] = {}
    for spec in specs:
        key = (spec["case"]["case_id"], spec["arm"])
        require(key not in spec_map, "fixture", f"重复 spec {key}")
        spec_map[key] = spec
    for case in cases:
        for arm in ARMS:
            require((case["case_id"], arm) in spec_map, "fixture",
                    f"缺少 {case['case_id']}/{arm} spec")
    for case in cases:
        for arm in ARMS:
            spec = spec_map[(case["case_id"], arm)]
            p0_raw = latest_raw.get((case["case_id"], "P0")) if arm == "A2" else None
            built = fixture_raw(case, arm, spec.get("items", []),
                                status=spec.get("status"),
                                attempt=spec.get("attempt", 1),
                                raw_item_count=spec.get("raw_item_count"),
                                usage=spec.get("usage", ledger_usage),
                                elapsed=spec.get("elapsed", ledger_elapsed or 1.0),
                                p0_raw=p0_raw,
                                parent_ids=spec.get("parent_ids"),
                                top_level_p0=spec.get("top_level_p0", True))
            raw_by_arm[arm].append(built["raw"])
            annotations.append(fixture_annotation(built["raw"], built["atoms"]))
            latest_raw[(case["case_id"], arm)] = built["raw"]
    ledger: list[dict] = []
    if ledger_extra is not None:
        ledger.extend(copy.deepcopy(list(ledger_extra)))
    for key, raw in latest_raw.items():
        ledger.append({"schema": "atomic-pilot-call-v1", "case_id": key[0],
                       "split": next(c["split"] for c in cases if c["case_id"] == key[0]),
                       "arm": key[1], "attempt": raw["attempt"],
                       "usage": copy.deepcopy(raw["usage"]),
                       "elapsed_s": raw["elapsed_s"],
                       "attempted": True, "succeeded": raw["call"]["succeeded"]})
    return {"cases": list(cases), "raw": raw_by_arm, "ledger": ledger,
            "annotations": annotations}


def score_parts(parts: Mapping[str, Any]) -> dict:
    input_sha = {name: {"sha256": hashlib.sha256(name.encode()).hexdigest(),
                        "bytes": 0, "rows": 0}
                 for name in ("cases-v1.jsonl", "raw-P0.jsonl", "raw-A1.jsonl",
                              "raw-A2.jsonl", "call-ledger.jsonl",
                              "annotations-v1.jsonl")}
    return score_bundle(parts["cases"], parts["raw"], parts["ledger"],
                        parts["annotations"],
                        input_sha256=input_sha, script_sha256="c" * 64,
                        synthetic_only=True)


def build_bundle(cases: Sequence[dict], specs: Sequence[dict],
                 *, ledger_extra: Sequence[dict] | None = None,
                 ledger_usage: Mapping[str, Any] | None = None,
                 ledger_elapsed: float | None = None) -> dict:
    return score_parts(build_bundle_parts(
        cases, specs, ledger_extra=ledger_extra, ledger_usage=ledger_usage,
        ledger_elapsed=ledger_elapsed))


def perfect_items(case: Mapping[str, Any], arm: str, *, tier: str | None = "explicit",
                  gold_ids: Sequence[str] | None = None,
                  prefix: str | None = None) -> list[tuple[dict, dict]]:
    ids = list(case_gold_ids(case)) if gold_ids is None else list(gold_ids)
    prefix = prefix or f"{case['case_id']}-{arm}"
    return [fixture_item(f"{prefix}-{gold_id}", tier=tier, core=[gold_id],
                         full=[gold_id]) for gold_id in ids]


def case_gold_ids(case: Mapping[str, Any]) -> list[str]:
    return [gold["gold_id"] for gold in case["gold_obligations"]]


def perfect_specs(cases: Sequence[dict], *, tier: str | None = "explicit") -> list[dict]:
    return [{"case": case, "arm": arm, "items": perfect_items(case, arm, tier=tier)}
            for case in cases for arm in ARMS]


def spec(case: Mapping[str, Any], arm: str, items: Sequence[tuple[dict, dict]],
         **kwargs) -> dict:
    return {"case": case, "arm": arm, "items": list(items), **kwargs}


def metric(result: Mapping[str, Any], arm: str, split: str, view: str,
           *keys: str) -> Any:
    return value_of(result, "arms", arm, "by_split", split, view, *keys)


def gate_of(result: Mapping[str, Any], arm: str) -> dict:
    return result["gates"][arm]


def item_of(gate: Mapping[str, Any], item_id: str) -> dict:
    for item in gate["items"]:
        if item["id"] == item_id:
            return item
    raise AssertionError(f"gate item not found: {item_id}")


def run_selftest() -> bool:
    checks = Checks()

                     
    checks.close("percentile_single", percentile([10.0], 0.5), 10.0)
    checks.close("percentile_p50", percentile([1.0, 2.0], 0.5), 1.5)
    checks.close("percentile_p95", percentile([1.0, 2.0], 0.95), 1.95)
    checks.none("ratio_zero_denominator", ratio(0, 0)["value"])
    checks.none("relative_reduction_zero_base", relative_reduction(0.0, 0.0))
    checks.none("relative_reduction_positive_zero_base",
                relative_reduction(1.0, 0.0))

                                                 
    cases = [fixture_case("C-ok-dev", "dev", ("g1", "g2")),
             fixture_case("C-ok-conf", "confirmation", ("g1", "g2"))]
    result = build_bundle(cases, perfect_specs(cases))
    checks.eq("ok_cases", result["counts"]["cases"], 2)
    checks.eq("ok_gold", result["counts"]["gold_total"], 4)
    checks.close("ok_coverage", metric(result, "A1", "confirmation", "direct",
                                       "source_coverage", "value"), 1.0)
    checks.close("ok_full_coverage_case", metric(result, "A1", "confirmation", "direct",
                                                 "full_coverage_case_rate", "value"), 1.0)
    checks.eq("ok_coverage_num", metric(result, "A1", "confirmation", "direct",
                                        "source_coverage", "num"), 2)
    checks.close("ok_faithfulness", metric(result, "A1", "confirmation", "direct",
                                            "faithfulness", "value"), 1.0)
    checks.eq("ok_supported", metric(result, "A1", "confirmation", "direct",
                                     "faithfulness", "supported"), 2)
    checks.close("ok_complete", metric(result, "A1", "confirmation", "direct",
                                       "complete_set", "rate"), 1.0)
    checks.close("ok_non_atomic", metric(result, "A1", "confirmation", "direct",
                                         "non_atomic", "obligation_rate"), 0.0)
    checks.close("ok_loop_coverage", metric(result, "A1", "confirmation", "loop",
                                            "source_coverage", "value"), 1.0)
    checks.close("ok_loop_complete", metric(result, "A1", "confirmation", "loop",
                                            "complete_set", "rate"), 1.0)
    checks.truthy("ok_A1_gate_pass", gate_of(result, "A1")["passed"])
    checks.eq("ok_A1_gate_failed_items", gate_of(result, "A1")["failed_items"], [])
    checks.eq("ok_A1_gate_has_reasons", all(item.get("reason") for item in
                                            gate_of(result, "A1")["items"]), True)
    checks.eq("ok_thresholds", result["thresholds"], THRESHOLDS)
    checks.eq("ok_input_sha_count", len(result["inputs"]), 6)
    checks.eq("ok_script_sha", result["script_sha256"], "c" * 64)
    checks.eq("ok_synthetic", result["synthetic_only"], True)
    checks.none("ok_A1_relative_waiver_value", item_of(gate_of(result, "A1"),
                                                       "non_atomic_relative_reduction")["value"])
    checks.truthy("ok_P0_zero_waiver_reason",
                  "P0" in item_of(gate_of(result, "A1"),
                                  "non_atomic_relative_reduction")["reason"])
    checks.eq("ok_status_counts", metric(result, "A1", "confirmation", "valid_output",
                                         "statuses"), {"ok": 1})
    checks.none("ok_A2_non_atomic_reduction_null",
                item_of(gate_of(result, "A2"), "a2_non_atomic_reduction_vs_a1")["value"])
    checks.eq("ok_A2_decision", gate_of(result, "A2")["decision"], "reject")

                                                            
    case = fixture_case("C-miss", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-miss-A1-g1", core=["g1"], full=["g1"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("miss_coverage", metric(result, "A1", "confirmation", "direct",
                                         "source_coverage", "value"), 0.5)
    checks.close("miss_complete", metric(result, "A1", "confirmation", "direct",
                                         "complete_set", "rate"), 0.0)
    checks.truthy("miss_coverage_gate_fail",
                  not item_of(gate_of(result, "A1"), "confirmation_coverage")["pass"])
    checks.close("miss_drop_pp", item_of(gate_of(result, "A1"),
                                         "coverage_drop_vs_p0")["value"], 50.0)

                                                      
    case = fixture_case("C-fab", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-fab-A1-g1", core=["g1"], full=["g1"]),
          fixture_item("C-fab-A1-x", core=[], full=[], entailment="unsupported")]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("fab_faithfulness", metric(result, "A1", "confirmation", "direct",
                                            "faithfulness", "value"), 0.5)
    checks.eq("fab_unsupported", metric(result, "A1", "confirmation", "direct",
                                        "faithfulness", "unsupported"), 1)
    checks.close("fab_unsupported_increment",
                 item_of(gate_of(result, "A1"), "unsupported_increment_vs_p0")["value"],
                 50.0)
    checks.truthy("fab_unsupported_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "unsupported_increment_vs_p0")["pass"])

                                                                         
    case = fixture_case("C-under", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-under-A1", core=["g1", "g2"], full=["g1", "g2"],
                       non_atomic=True)]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("under_non_atomic_obligation", metric(result, "A1", "confirmation",
                                                       "direct", "non_atomic",
                                                       "obligation_rate"), 1.0)
    checks.close("under_coverage", metric(result, "A1", "confirmation", "direct",
                                          "source_coverage", "value"), 1.0)
    checks.close("under_complete", metric(result, "A1", "confirmation", "direct",
                                          "complete_set", "rate"), 0.0)
    checks.truthy("under_non_atomic_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "confirmation_non_atomic_obligation")["pass"])
    checks.eq("under_non_atomic_reason",
              metric(result, "A1", "confirmation", "direct",
                     "complete_set", "failure_reasons").get("disallowed_non_atomic"), 1)
    checks.close("under_non_atomic_case_rate",
                 metric(result, "A1", "confirmation", "direct",
                        "non_atomic", "case_rate"), 1.0)
    checks.close("under_non_atomic_atom_rate",
                 metric(result, "A1", "confirmation", "direct",
                        "non_atomic", "atom_rate"), 1.0)

                                                          
    case = fixture_case("C-partial", "confirmation", ("g1",))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-partial-A1", core=["g1"], full=["g1"],
                       entailment="partial")]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("partial_faithfulness", metric(result, "A1", "confirmation", "direct",
                                                "faithfulness", "value"), 0.0)
    checks.eq("partial_count", metric(result, "A1", "confirmation", "direct",
                                      "faithfulness", "partial"), 1)
    checks.eq("partial_unsupported", metric(result, "A1", "confirmation", "direct",
                                            "faithfulness", "unsupported"), 0)
    checks.close("partial_complete", metric(result, "A1", "confirmation", "direct",
                                            "complete_set", "rate"), 0.0)

                                                     
    case = fixture_case("C-over", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-over-A1-g1", core=["g1"], full=["g1"], over_split=True),
          fixture_item("C-over-A1-g2", core=["g2"], full=["g2"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("over_rate", metric(result, "A1", "confirmation", "direct",
                                     "over_split", "atom_rate"), 0.5)
    checks.close("over_case_rate", metric(result, "A1", "confirmation", "direct",
                                          "over_split", "case_rate"), 1.0)
    checks.close("over_complete", metric(result, "A1", "confirmation", "direct",
                                         "complete_set", "rate"), 0.0)
    checks.truthy("over_absolute_gate_fail",
                  not item_of(gate_of(result, "A1"), "over_split_absolute")["pass"])
    checks.truthy("over_not_above_gate_fail",
                  not item_of(gate_of(result, "A1"), "over_split_not_above_p0")["pass"])

                                                               
    case = fixture_case("C-ctx", "confirmation", ("g1",), context_ids=("g1",))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-ctx-A1", core=["g1"], full=[], context_errors=["condition"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("ctx_loss_rate", metric(result, "A1", "confirmation", "direct",
                                         "context_loss", "rate"), 1.0)
    checks.eq("ctx_loss_num", metric(result, "A1", "confirmation", "direct",
                                     "context_loss", "lost_gold"), 1)
    checks.eq("ctx_inversion_atoms", metric(result, "A1", "confirmation", "direct",
                                            "inversion", "atom_count"), 1)
    checks.close("ctx_complete", metric(result, "A1", "confirmation", "direct",
                                        "complete_set", "rate"), 0.0)
    checks.truthy("ctx_context_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "context_loss_not_increased")["pass"])
    checks.truthy("ctx_inversion_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "polarity_condition_inversion_zero")["pass"])

                                        
    pos = fixture_case("C-persist", "confirmation", ("g1",))
    neg = fixture_case("C-neg", "confirmation", ())
    p0_pos = perfect_items(pos, "P0")
    p0_neg = []
    a1_pos = [fixture_item("C-persist-A1", core=["g1"], full=["g1"],
                           persistence=False)]
    a2_pos = perfect_items(pos, "A2")
    result = build_bundle([pos, neg], [
        spec(pos, "P0", p0_pos), spec(pos, "A1", a1_pos), spec(pos, "A2", a2_pos),
        spec(neg, "P0", p0_neg), spec(neg, "A1", []), spec(neg, "A2", []),
    ])
    checks.eq("persist_case_num", metric(result, "A1", "confirmation", "direct",
                                         "false_persistence", "case_count"), 1)
    checks.eq("persist_case_den", metric(result, "A1", "confirmation", "direct",
                                         "false_persistence", "denominator"), 2)
    checks.close("persist_case_rate", metric(result, "A1", "confirmation", "direct",
                                             "false_persistence", "rate"), 0.5)
    checks.close("persist_neg_rate", metric(result, "A1", "confirmation", "direct",
                                            "false_persistence",
                                            "negative_control_rate"), 0.0)
    checks.close("persist_complete_rate", metric(result, "A1", "confirmation", "direct",
                                                 "complete_set", "rate"), 0.5)
    checks.truthy("persist_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "false_persistence_not_increased")["pass"])

                                                  
    case = fixture_case("C-dup", "confirmation", ("g1",))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item(f"C-dup-A1-{i}", core=["g1"], full=["g1"],
                       duplicate_group="d") for i in range(3)]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.eq("dup_extra", metric(result, "A1", "confirmation", "direct",
                                  "duplicate", "extra_atoms"), 2)
    checks.eq("dup_kept", metric(result, "A1", "confirmation", "direct",
                                 "kept_atoms"), 3)
    checks.close("dup_rate", metric(result, "A1", "confirmation", "direct",
                                    "duplicate", "atom_rate"), 2.0 / 3.0)
    checks.close("dup_case_rate", metric(result, "A1", "confirmation", "direct",
                                         "duplicate", "case_rate"), 1.0)
    checks.close("dup_complete", metric(result, "A1", "confirmation", "direct",
                                        "complete_set", "rate"), 0.0)
    checks.truthy("dup_absolute_gate_fail",
                  not item_of(gate_of(result, "A1"), "duplicate_atom_absolute")["pass"])
    checks.truthy("dup_not_above_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "duplicate_atom_not_above_p0")["pass"])

                                                            
    pos = fixture_case("C-empty", "confirmation", ("g1",))
    neg = fixture_case("C-empty-neg", "confirmation", ())
    result = build_bundle([pos, neg], [
        spec(pos, arm, []) for arm in ARMS
    ] + [
        spec(neg, arm, []) for arm in ARMS
    ])
    checks.close("empty_coverage", metric(result, "A1", "confirmation", "direct",
                                          "source_coverage", "value"), 0.0)
    checks.none("empty_faithfulness_null", metric(result, "A1", "confirmation", "direct",
                                                  "faithfulness", "value"))
    checks.close("empty_complete_rate", metric(result, "A1", "confirmation", "direct",
                                               "complete_set", "rate"), 0.5)
    checks.close("empty_valid_output_rate", metric(result, "A1", "confirmation", "direct",
                                                   "valid_output", "rate"), 1.0)
    checks.eq("empty_status", metric(result, "A1", "confirmation", "valid_output",
                                     "statuses"), {"valid_empty": 2})
    checks.truthy("empty_positive_fails",
                  not gate_of(result, "A1")["passed"])
    neg_case_row = next(row for row in result["arms"]["A1"]["all"]["case_rows"]
                        if row["case_id"] == "C-empty-neg")
    checks.eq("empty_negative_complete", neg_case_row["direct_complete"], True)

                                                     
    case = fixture_case("C-alt", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-alt-A1", core=["g1", "g2"], full=["g1", "g2"],
                       non_atomic=True,
                       alternative={"type": "acceptable_compound",
                                    "note": "两条义务不可分开更新"})]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("alt_complete", metric(result, "A1", "confirmation", "direct",
                                        "complete_set", "rate"), 1.0)
    checks.close("alt_non_atomic_still_counted",
                 metric(result, "A1", "confirmation", "direct",
                        "non_atomic", "obligation_rate"), 1.0)
    checks.eq("alt_allowed_atoms", metric(result, "A1", "confirmation", "direct",
                                          "non_atomic", "allowed_atom_count"), 1)
    checks.eq("alt_disallowed_atoms", metric(result, "A1", "confirmation", "direct",
                                             "non_atomic", "disallowed_atom_count"), 0)
    checks.truthy("alt_gate_non_atomic_still_fails",
                  not item_of(gate_of(result, "A1"),
                              "confirmation_non_atomic_obligation")["pass"])
    result = build_bundle([case], [
        spec(case, "P0", p0), spec(case, "A1", [
            fixture_item("C-alt-reference", core=["g1", "g2"], full=["g1", "g2"],
                         non_atomic=True,
                         alternative={"type": "reference_omission",
                                      "note": "参考漏了替代划分"})]),
        spec(case, "A2", a2),
    ])
    checks.close("alt_reference_omission_complete",
                 metric(result, "A1", "confirmation", "direct",
                        "complete_set", "rate"), 0.0)
    checks.eq("alt_reference_omission_disallowed",
              metric(result, "A1", "confirmation", "direct",
                     "non_atomic", "disallowed_atom_count"), 1)

                                                                                      
    case = fixture_case("C-compound7", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-compound7-A1-c", core=["g1", "g2"], full=["g1", "g2"],
                       non_atomic=True)]
    a1 += [fixture_item(f"C-compound7-A1-d{i}", core=["g1", "g2"], full=["g1", "g2"],
                        duplicate_group="d7") for i in range(7)]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("compound7_non_atomic_obligation",
                 metric(result, "A1", "confirmation", "direct",
                        "non_atomic", "obligation_rate"), 1.0)
    checks.close("compound7_complete", metric(result, "A1", "confirmation", "direct",
                                              "complete_set", "rate"), 0.0)
    checks.close("compound7_faithfulness", metric(result, "A1", "confirmation", "direct",
                                                  "faithfulness", "value"), 1.0)
    checks.close("compound7_atom_rate", metric(result, "A1", "confirmation", "direct",
                                               "non_atomic", "atom_rate"), 1.0 / 8.0)
    checks.close("compound7_duplicate_rate", metric(result, "A1", "confirmation",
                                                    "direct", "duplicate", "atom_rate"),
                 6.0 / 8.0)
    checks.truthy("compound7_A1_gate_fail", not gate_of(result, "A1")["passed"])
    checks.truthy("compound7_non_atomic_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "confirmation_non_atomic_obligation")["pass"])

                                                              
    case = fixture_case("C-loop-loss", "confirmation", ("g1", "g2"))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-loop-loss-memory", tier=None, core=["g1"], full=["g1"]),
          fixture_item("C-loop-loss-rule", tier="explicit", core=["g2"], full=["g2"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("loop_direct_coverage", metric(result, "A1", "confirmation", "direct",
                                                "source_coverage", "value"), 1.0)
    checks.close("loop_visible_coverage", metric(result, "A1", "confirmation", "loop",
                                                 "source_coverage", "value"), 0.5)
    checks.close("loop_rule_only_coverage", metric(result, "A1", "confirmation", "loop",
                                                   "rule_only_coverage", "value"), 0.5)
    checks.close("loop_projection_loss_pp", metric(result, "A1", "confirmation", "loop",
                                                   "memory_projection",
                                                   "loop_projection_loss_pp"), 50.0)
    checks.eq("loop_memory_only_full", metric(result, "A1", "confirmation", "loop",
                                             "memory_projection",
                                             "memory_only_full_gold"), 1)
    checks.eq("loop_dropped_memory_atoms", metric(result, "A1", "confirmation", "loop",
                                                  "memory_projection",
                                                  "dropped_memory_atoms"), 1)
    checks.close("loop_direct_complete", metric(result, "A1", "confirmation", "direct",
                                                "complete_set", "rate"), 1.0)
    checks.close("loop_visible_complete", metric(result, "A1", "confirmation", "loop",
                                                 "complete_set", "rate"), 0.0)
    checks.eq("loop_complete_gap", metric(result, "A1", "confirmation", "loop",
                                          "memory_projection",
                                          "direct_minus_loop_complete_cases"), 1)

                                                
    case8 = fixture_case("C-cap8", "confirmation", tuple(f"g{i}" for i in range(8)))
    case9 = fixture_case("C-cap9", "confirmation", tuple(f"g{i}" for i in range(8)))
    items8 = perfect_items(case8, "A1")
    items9 = perfect_items(case9, "A1")
    result = build_bundle([case8, case9], [
        spec(case8, "P0", perfect_items(case8, "P0")),
        spec(case8, "A1", items8, raw_item_count=8),
        spec(case8, "A2", perfect_items(case8, "A2")),
        spec(case9, "P0", perfect_items(case9, "P0")),
        spec(case9, "A1", items9, raw_item_count=9),
        spec(case9, "A2", perfect_items(case9, "A2")),
    ])
    checks.eq("cap_applied_cases",
              result["arms"]["A1"]["all"]["cap"]["applied_cases"], 1)
    checks.eq("cap_exactly8_cases", result["arms"]["A1"]["all"]["cap"]["exactly_8_cases"], 1)
    checks.eq("cap_eq8_not_applied",
              next(row["cap_applied"] for row in result["arms"]["A1"]["all"]["case_rows"]
                   if row["case_id"] == "C-cap8"), False)
    checks.eq("cap_gt8_applied",
              next(row["cap_applied"] for row in result["arms"]["A1"]["all"]["case_rows"]
                   if row["case_id"] == "C-cap9"), True)
    checks.close("cap_coverage", metric(result, "A1", "confirmation", "direct",
                                        "source_coverage", "value"), 1.0)

                                                          
    case = fixture_case("C-rawloss", "confirmation", ("g1",))
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-rawloss-A1", kept="dropped_evidence",
                       core=["g1"], full=["g1"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.close("rawloss_raw_coverage", metric(result, "A1", "confirmation", "raw",
                                                "source_coverage", "value"), 1.0)
    checks.close("rawloss_kept_coverage", metric(result, "A1", "confirmation", "direct",
                                                 "source_coverage", "value"), 0.0)
    checks.close("rawloss_loss_pp", metric(result, "A1", "confirmation", "direct",
                                           "raw_to_kept_loss_pp"), 100.0)
    checks.eq("rawloss_all_gated", metric(result, "A1", "confirmation", "valid_output",
                                          "all_gated_cases"), 1)
    checks.close("rawloss_complete", metric(result, "A1", "confirmation", "direct",
                                            "complete_set", "rate"), 0.0)

                                                       
    case = fixture_case("C-severe", "confirmation", ("g1",),
                        forbidden={"F-severe": "severe"})
    p0 = perfect_items(case, "P0")
    a1 = [fixture_item("C-severe-A1", core=["g1"], full=["g1"],
                       forbidden=["F-severe"])]
    a2 = perfect_items(case, "A2")
    result = build_bundle([case], [spec(case, "P0", p0), spec(case, "A1", a1),
                                   spec(case, "A2", a2)])
    checks.eq("severe_cases", metric(result, "A1", "confirmation", "direct",
                                     "forbidden", "severe_case_count"), 1)
    checks.eq("severe_matches", metric(result, "A1", "confirmation", "direct",
                                       "forbidden", "severe_match_count"), 1)
    checks.close("severe_forbidden_case_rate", metric(result, "A1", "confirmation",
                                                      "direct", "forbidden",
                                                      "case_rate"), 1.0)
    checks.close("severe_complete_still", metric(result, "A1", "confirmation", "direct",
                                                 "complete_set", "rate"), 1.0)
    checks.truthy("severe_gate_fail",
                  not item_of(gate_of(result, "A1"), "severe_forbidden_zero")["pass"])

                                                      
    case = fixture_case("C-cost", "confirmation", ("g1",))
    result = build_bundle(
        [case],
        [spec(case, "P0", perfect_items(case, "P0")),
         spec(case, "A1", perfect_items(case, "A1"), attempt=2),
         spec(case, "A2", perfect_items(case, "A2"))],
        ledger_extra=[{"case_id": "C-cost", "split": "confirmation", "arm": "A1",
                       "attempt": 1,
                       "usage": {"prompt_tokens": 20, "completion_tokens": 10,
                                 "total_tokens": 30},
                       "elapsed_s": 3.0}])
    checks.eq("cost_total_calls", result["costs"]["total"]["calls"], 4)
    checks.eq("cost_total_tokens", result["costs"]["total"]["tokens"]["total"], 75)
    checks.eq("cost_a1_calls", result["costs"]["by_arm"]["A1"]["all"]["calls"], 2)
    checks.eq("cost_a1_tokens", result["costs"]["by_arm"]["A1"]["all"]["tokens"]["total"], 45)
    checks.close("cost_a1_case_p50", result["costs"]["by_arm"]["A1"]["all"]
                 ["case_latency_seconds"]["p50"], 4.0)
    checks.close("cost_a1_case_p95", result["costs"]["by_arm"]["A1"]["all"]
                 ["case_latency_seconds"]["p95"], 4.0)
    checks.close("cost_a1_attempt_p50", result["costs"]["by_arm"]["A1"]["all"]
                 ["attempt_latency_seconds"]["p50"], 2.0)
    checks.close("cost_a1_attempt_p95", result["costs"]["by_arm"]["A1"]["all"]
                 ["attempt_latency_seconds"]["p95"], 2.9)
    checks.eq("cost_a2_calls", result["costs"]["budget"]["a2_calls"], 1)

                                                          
    def relative_bundle(a1_non_atomic: int) -> dict:
        ids = [f"g{i}" for i in range(1, 51)]
        conf = fixture_case("C-rel", "confirmation", ids)
        dev = fixture_case("C-rel-dev", "dev", ("g1",))
        def p0_items(case, prefix):
            items = []
            for i, gold_id in enumerate(ids):
                items.append(fixture_item(f"{prefix}-{gold_id}", core=[gold_id],
                                          full=[gold_id],
                                          non_atomic=(i < 5),
                                          alternative=("acceptable_compound"
                                                       if i < 5 else None)))
            return items
        def a1_items(case, prefix):
            items = []
            for i, gold_id in enumerate(ids):
                items.append(fixture_item(f"{prefix}-{gold_id}", core=[gold_id],
                                          full=[gold_id],
                                          non_atomic=(i < a1_non_atomic),
                                          alternative=("acceptable_compound"
                                                       if i < a1_non_atomic else None)))
            return items
        specs = [
            spec(conf, "P0", p0_items(conf, "C-rel-P0")),
            spec(conf, "A1", a1_items(conf, "C-rel-A1")),
            spec(conf, "A2", perfect_items(conf, "A2")),
            spec(dev, "P0", perfect_items(dev, "P0")),
            spec(dev, "A1", perfect_items(dev, "A1")),
            spec(dev, "A2", perfect_items(dev, "A2")),
        ]
        return build_bundle([conf, dev], specs)
    result = relative_bundle(2)
    checks.close("relative_a1_rate", metric(result, "A1", "confirmation", "direct",
                                             "non_atomic", "obligation_rate"), 0.04)
    checks.close("relative_reduction", item_of(gate_of(result, "A1"),
                                               "non_atomic_relative_reduction")["value"], 0.6)
    checks.truthy("relative_gate_pass", gate_of(result, "A1")["passed"])
    result = relative_bundle(3)
    checks.close("relative_fail_a1_rate", metric(result, "A1", "confirmation", "direct",
                                                  "non_atomic", "obligation_rate"), 0.06)
    checks.close("relative_fail_reduction", item_of(gate_of(result, "A1"),
                                                    "non_atomic_relative_reduction")["value"],
                 0.4)
    checks.truthy("relative_fail_item",
                  not item_of(gate_of(result, "A1"),
                              "non_atomic_relative_reduction")["pass"])
    checks.eq("relative_fail_only_item", gate_of(result, "A1")["failed_items"],
              ["non_atomic_relative_reduction"])

                                                           
    dev = fixture_case("C-dir-dev", "dev", ("g1", "g2", "g3", "g4", "g5"))
    conf = fixture_case("C-dir-conf", "confirmation", ("g1", "g2", "g3", "g4", "g5"))
    p0_dev = [fixture_item(f"C-dir-dev-P0-{g}", core=[g], full=[g])
              for g in case_gold_ids(dev)]
    a1_dev = [fixture_item("C-dir-dev-A1-non", core=["g1"], full=["g1"],
                           non_atomic=True, alternative="acceptable_compound")]
    a1_dev += [fixture_item(f"C-dir-dev-A1-{g}", core=[g], full=[g])
               for g in ("g2", "g3", "g4", "g5")]
    p0_conf = [fixture_item("C-dir-conf-P0-non", core=["g1"], full=["g1"],
                            non_atomic=True, alternative="acceptable_compound")]
    p0_conf += [fixture_item(f"C-dir-conf-P0-{g}", core=[g], full=[g])
                for g in ("g2", "g3", "g4", "g5")]
    a1_conf = [fixture_item(f"C-dir-conf-A1-{g}", core=[g], full=[g])
               for g in case_gold_ids(conf)]
    result = build_bundle([dev, conf], [
        spec(dev, "P0", p0_dev), spec(dev, "A1", a1_dev),
        spec(dev, "A2", perfect_items(dev, "A2")),
        spec(conf, "P0", p0_conf), spec(conf, "A1", a1_conf),
        spec(conf, "A2", perfect_items(conf, "A2")),
    ])
    checks.close("direction_dev_delta", item_of(gate_of(result, "A1"),
                                                "non_atomic_direction_consistent")["value"]
                 ["dev_delta"], 0.2)
    checks.truthy("direction_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "non_atomic_direction_consistent")["pass"])
    checks.truthy("direction_in_failed", "non_atomic_direction_consistent"
                  in gate_of(result, "A1")["failed_items"])

                                                         
    case = fixture_case("C-a2", "confirmation", ("g1",))
    parts = build_bundle_parts([case], perfect_specs([case]))
    result = score_parts(parts)
    breakdown = result["a2_parent_breakdown"]["all"]
    checks.eq("a2_parent_kept_atoms",
              breakdown["by_parent_gate_status"]["kept"]["kept_atom_count"], 1)
    checks.eq("a2_parent_kept_full_gold",
              breakdown["by_parent_gate_status"]["kept"]["fully_covered_gold"], 1)
    checks.eq("a2_unknown_atoms", breakdown["unknown_lineage_atoms"], 0)
    checks.eq("a2_pure_atomization", breakdown["pure_atomization_atoms"], 1)

    tampered = copy.deepcopy(parts)
    tampered["raw"]["A2"][0]["parsed_candidates"][0]["parent_content_sha"] = "0" * 64
    checks.raises("a2_content_tamper", "a2_parent_content_mismatch",
                  lambda: score_parts(tampered))
    tampered = copy.deepcopy(parts)
    tampered["raw"]["A2"][0]["parsed_candidates"][0]["parent_p0_attempt_sha"] = "0" * 64
    checks.raises("a2_attempt_tamper", "a2_parent_attempt_mismatch",
                  lambda: score_parts(tampered))
    tampered = copy.deepcopy(parts)
    tampered["raw"]["A2"][0]["parsed_candidates"][0]["parent_gate_status"] = "dropped_evidence"
    checks.raises("a2_gate_tamper", "a2_parent_gate_mismatch",
                  lambda: score_parts(tampered))
    tampered = copy.deepcopy(parts)
    tampered["raw"]["A2"][0]["parsed_candidates"][0]["parent_p0_candidate_id"] = "cand_missing"
    checks.raises("a2_parent_missing", "a2_parent_not_in_p0",
                  lambda: score_parts(tampered))
    tampered = copy.deepcopy(parts)
    tampered["raw"]["A2"][0]["p0_content_sha256"] = "0" * 64
    checks.raises("a2_top_level_tamper", "a2_top_level_parent_mismatch",
                  lambda: score_parts(tampered))

                                 
    case = fixture_case("C-a2-unknown", "confirmation", ("g1",))
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", [fixture_item("C-a2-unknown-A2", core=["g1"], full=["g1"])],
             parent_ids=["unknown"]),
    ])
    breakdown = result["a2_parent_breakdown"]["all"]
    checks.eq("a2_unknown_kept", breakdown["by_parent_gate_status"]["unknown"]
              ["kept_atom_count"], 1)
    checks.eq("a2_unknown_lineage_atoms", breakdown["unknown_lineage_atoms"], 1)
    checks.truthy("a2_unknown_warning", any("unknown lineage" in warning
                                            for warning in result["validation"]["warnings"]))

                                                         
    case = fixture_case("C-a2-break", "confirmation", ("g1", "g2", "g3"))
    p0_items = [fixture_item("C-a2-break-P0-1", core=["g1"], full=["g1"]),
                fixture_item("C-a2-break-P0-2", kept="dropped_evidence",
                             core=["g2"], full=["g2"]),
                fixture_item("C-a2-break-P0-3", kept="dropped_injection",
                             core=["g3"], full=["g3"])]
    a2_items = [fixture_item("C-a2-break-A2-1", core=["g1"], full=["g1"]),
                fixture_item("C-a2-break-A2-2", core=["g2"], full=["g2"]),
                fixture_item("C-a2-break-A2-3", core=["g3"], full=["g3"])]
    result = build_bundle([case], [
        spec(case, "P0", p0_items),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", a2_items,
             parent_ids=["C-a2-break-P0-1", "C-a2-break-P0-2", "C-a2-break-P0-3"]),
    ])
    breakdown = result["a2_parent_breakdown"]["all"]
    checks.eq("a2_break_kept_atoms",
              breakdown["by_parent_gate_status"]["kept"]["kept_atom_count"], 1)
    checks.eq("a2_break_evidence_atoms",
              breakdown["by_parent_gate_status"]["evidence"]["kept_atom_count"], 1)
    checks.eq("a2_break_injection_atoms",
              breakdown["by_parent_gate_status"]["injection"]["kept_atom_count"], 1)
    checks.eq("a2_break_pure", breakdown["pure_atomization_atoms"], 1)
    checks.eq("a2_break_evidence_repair", breakdown["evidence_repair_atoms"], 1)
    checks.eq("a2_break_injection_change", breakdown["injection_text_change_atoms"], 1)
    checks.eq("a2_break_full_each", [
        breakdown["by_parent_gate_status"][name]["fully_covered_gold"]
        for name in ("kept", "evidence", "injection")], [1, 1, 1])
    checks.eq("a2_break_p0_partial_status",
              result["arms"]["P0"]["all"]["valid_output"]["statuses"],
              {"partial_gated": 1})

                                                     
    conf_cases = [fixture_case(f"C-a2gain-{i}", "confirmation", ("g1",))
                  for i in range(10)]
    dev_case = fixture_case("C-a2gain-dev", "dev", ("g1",))
    specs = []
    for index, case in enumerate(conf_cases):
        specs.append(spec(case, "P0",
                          perfect_items(case, "P0") if index != 9 else []))
        specs.append(spec(case, "A1",
                          perfect_items(case, "A1") if index != 9 else []))
        specs.append(spec(case, "A2", perfect_items(case, "A2")))
    specs += [spec(dev_case, "P0", perfect_items(dev_case, "P0")),
              spec(dev_case, "A1", perfect_items(dev_case, "A1")),
              spec(dev_case, "A2", perfect_items(dev_case, "A2"))]
    result = build_bundle(conf_cases + [dev_case], specs)
    checks.close("a2gain_a1_coverage", metric(result, "A1", "confirmation", "direct",
                                               "source_coverage", "value"), 0.9)
    checks.close("a2gain_a2_coverage", metric(result, "A2", "confirmation", "direct",
                                               "source_coverage", "value"), 1.0)
    checks.close("a2gain_pp", item_of(gate_of(result, "A2"),
                                      "a2_coverage_gain_vs_a1")["value"], 10.0)
    checks.truthy("a2gain_absolute", gate_of(result, "A2")["absolute_gate_passed"])
    checks.truthy("a2gain_improvement", gate_of(result, "A2")["minimum_improvement_passed"])
    checks.eq("a2gain_decision", gate_of(result, "A2")["decision"],
              "enter_cost_evaluation")
    checks.eq("a2gain_failed_items", gate_of(result, "A2")["failed_items"], [])
    checks.eq("a2gain_nonblocking",
              set(gate_of(result, "A2")["non_blocking_failed_items"]),
              {"a2_non_atomic_reduction_vs_a1"})

                                                          
    case = fixture_case("C-a2fail", "confirmation", ("g1", "g2"))
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", [fixture_item("C-a2fail-A2", core=["g1", "g2"],
                                       full=["g1", "g2"], non_atomic=True)]),
    ])
    checks.close("a2fail_complete", metric(result, "A2", "confirmation", "direct",
                                           "complete_set", "rate"), 0.0)
    checks.truthy("a2fail_complete_item",
                  not item_of(gate_of(result, "A2"),
                              "confirmation_complete_set")["pass"])
    checks.truthy("a2fail_non_atomic_item",
                  not item_of(gate_of(result, "A2"),
                              "confirmation_non_atomic_obligation")["pass"])
    checks.eq("a2fail_decision", gate_of(result, "A2")["decision"], "reject")

                                                   
    conf_cases = [fixture_case(f"C-a2non-{i}", "confirmation",
                               ("g1", "g2", "g3", "g4", "g5")) for i in range(5)]
    dev_case = fixture_case("C-a2non-dev", "dev", ("g1",))
    specs = []
    for index, case in enumerate(conf_cases):
        p0_items = perfect_items(case, "P0")
        a1_items = []
        a2_items = []
        for gold_index, gold in enumerate(case_gold_ids(case)):
            prefix = case["case_id"]
            if index == 0 and gold_index < 2:
                a1_items.append(fixture_item(f"{prefix}-A1-{gold}", core=[gold],
                                             full=[gold], non_atomic=True,
                                             alternative="acceptable_compound"))
            else:
                a1_items.append(fixture_item(f"{prefix}-A1-{gold}", core=[gold],
                                             full=[gold]))
            if index == 0 and gold_index == 0:
                a2_items.append(fixture_item(f"{prefix}-A2-{gold}", core=[gold],
                                             full=[gold], non_atomic=True,
                                             alternative="acceptable_compound"))
            else:
                a2_items.append(fixture_item(f"{prefix}-A2-{gold}", core=[gold],
                                             full=[gold]))
        specs += [spec(case, "P0", p0_items), spec(case, "A1", a1_items),
                  spec(case, "A2", a2_items)]
    specs += [spec(dev_case, "P0", perfect_items(dev_case, "P0")),
              spec(dev_case, "A1", perfect_items(dev_case, "A1")),
              spec(dev_case, "A2", perfect_items(dev_case, "A2"))]
    result = build_bundle(conf_cases + [dev_case], specs)
    checks.close("a2non_a1_rate", metric(result, "A1", "confirmation", "direct",
                                          "non_atomic", "obligation_rate"), 0.08)
    checks.close("a2non_a2_rate", metric(result, "A2", "confirmation", "direct",
                                          "non_atomic", "obligation_rate"), 0.04)
    checks.close("a2non_reduction", item_of(gate_of(result, "A2"),
                                            "a2_non_atomic_reduction_vs_a1")["value"], 0.5)
    checks.truthy("a2non_absolute", gate_of(result, "A2")["absolute_gate_passed"])
    checks.truthy("a2non_improvement", gate_of(result, "A2")["minimum_improvement_passed"])
    checks.eq("a2non_decision", gate_of(result, "A2")["decision"],
              "enter_cost_evaluation")
    checks.eq("a2non_failed_items", gate_of(result, "A2")["failed_items"], [])
    checks.eq("a2non_nonblocking",
              set(gate_of(result, "A2")["non_blocking_failed_items"]),
              {"a2_coverage_gain_vs_a1"})

                                             
    rows = result["stratified_by_arm_split_family"]
    checks.truthy("stratified_rows", len(rows) > 0)
    checks.eq("stratified_has_arm_split_family",
              all({"arm", "split", "family", "cases", "direct", "loop"} <= set(row)
                  for row in rows), True)
    checks.eq("stratified_arms", sorted({row["arm"] for row in rows}), sorted(ARMS))
    checks.truthy("stratified_ratio_n_over_N",
                  "den" in rows[0]["direct"]["coverage"]
                  and "num" in rows[0]["direct"]["coverage"])

                                                      
    zero_case = fixture_case("C-null", "confirmation", ("g1",))
    zero_result = build_bundle([zero_case], [
        spec(zero_case, arm, []) for arm in ARMS
    ])
    checks.none("zero_duplicate_rate", metric(zero_result, "A1", "confirmation",
                                              "direct", "duplicate", "atom_rate"))
    checks.none("zero_over_split_rate", metric(zero_result, "A1", "confirmation",
                                               "direct", "over_split", "atom_rate"))
    checks.none("zero_non_atomic_atom_rate", metric(zero_result, "A1", "confirmation",
                                                    "direct", "non_atomic",
                                                    "atom_rate"))
    checks.close("zero_complete", metric(zero_result, "A1", "confirmation", "direct",
                                         "complete_set", "rate"), 0.0)

    neg_only = fixture_case("C-only-neg", "confirmation", ())
    neg_only_result = build_bundle([neg_only], [
        spec(neg_only, arm, []) for arm in ARMS
    ])
    checks.none("neg_only_coverage", metric(neg_only_result, "A1", "confirmation",
                                             "direct", "source_coverage", "value"))
    checks.none("neg_only_non_atomic", metric(neg_only_result, "A1", "confirmation",
                                              "direct", "non_atomic",
                                              "obligation_rate"))
    checks.none("neg_only_context", metric(neg_only_result, "A1", "confirmation",
                                           "direct", "context_loss", "rate"))
    checks.close("neg_only_complete", metric(neg_only_result, "A1", "confirmation",
                                             "direct", "complete_set", "rate"), 1.0)
    checks.close("neg_only_false_persistence", metric(neg_only_result, "A1", "confirmation",
                                                      "direct", "false_persistence",
                                                      "rate"), 0.0)
    checks.close("neg_only_valid_output", metric(neg_only_result, "A1", "confirmation",
                                                 "valid_output", "rate"), 1.0)

                                                                       
    status_cases = [
        (fixture_case("C-invalid-json", "confirmation", ("g1",)), "invalid_json"),
        (fixture_case("C-length", "confirmation", ("g1",)), "length"),
        (fixture_case("C-llm", "confirmation", ("g1",)), "llm_failure"),
        (fixture_case("C-parse-empty", "confirmation", ("g1",)), "empty_after_parse"),
        (fixture_case("C-parse-empty-neg", "confirmation", ()), "empty_after_parse"),
    ]
    specs = []
    for status_case, status in status_cases:
        for arm in ARMS:
            specs.append(spec(status_case, arm, [], status=status))
    status_result = build_bundle([case for case, _ in status_cases], specs)
    checks.close("status_valid_output_rate",
                 metric(status_result, "A1", "confirmation", "direct",
                        "valid_output", "rate"), 0.4)
    checks.eq("status_counts", metric(status_result, "A1", "confirmation",
                                      "valid_output", "statuses"),
              {"empty_after_parse": 2, "invalid_json": 1, "length": 1,
               "llm_failure": 1})
    checks.close("status_complete", metric(status_result, "A1", "confirmation",
                                           "direct", "complete_set", "rate"), 0.0)
    checks.close("status_coverage", metric(status_result, "A1", "confirmation",
                                           "direct", "source_coverage", "value"), 0.0)
    checks.eq("status_arm_valid_output_rate",
              metric(status_result, "A1", "confirmation", "valid_output", "rate"), 0.4)
    checks.eq("status_negative_parse_complete",
              next(row["direct_complete"] for row in
                   status_result["arms"]["A1"]["all"]["case_rows"]
                   if row["case_id"] == "C-parse-empty-neg"), False)

                               
    case = fixture_case("C-a2-cov", "confirmation", ("g1", "g2"))
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", [fixture_item("C-a2-cov-A2", core=["g1"], full=["g1"]),
                          fixture_item("C-a2-cov-X", core=[], full=[],
                                       entailment="unsupported")]),
    ])
    checks.close("a2cov_coverage", metric(result, "A2", "confirmation", "direct",
                                          "source_coverage", "value"), 0.5)
    checks.close("a2cov_faithfulness", metric(result, "A2", "confirmation", "direct",
                                              "faithfulness", "value"), 0.5)
    checks.truthy("a2cov_coverage_item",
                  not item_of(gate_of(result, "A2"), "confirmation_coverage")["pass"])
    checks.truthy("a2cov_faith_item",
                  not item_of(gate_of(result, "A2"),
                              "confirmation_faithfulness")["pass"])

                    
    case = fixture_case("C-a2dup", "confirmation", ("g1", "g2"))
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", [fixture_item("C-a2dup-A2-1", core=["g1", "g2"],
                                       full=["g1", "g2"], duplicate_group="d",
                                       over_split=True),
                          fixture_item("C-a2dup-A2-2", core=["g1", "g2"],
                                       full=["g1", "g2"], duplicate_group="d")]),
    ])
    checks.close("a2dup_rate", metric(result, "A2", "confirmation", "direct",
                                      "duplicate", "atom_rate"), 0.5)
    checks.close("a2over_rate", metric(result, "A2", "confirmation", "direct",
                                       "over_split", "atom_rate"), 0.5)
    checks.truthy("a2dup_item",
                  not item_of(gate_of(result, "A2"), "duplicate_atom_absolute")["pass"])
    checks.truthy("a2over_item",
                  not item_of(gate_of(result, "A2"), "over_split_absolute")["pass"])

                                                
    a1_ids = {item["id"] for item in gate_of(result, "A1")["items"]}
    expected_a1 = {
        "confirmation_coverage", "confirmation_faithfulness",
        "confirmation_complete_set", "confirmation_non_atomic_obligation",
        "non_atomic_relative_reduction", "coverage_drop_vs_p0",
        "unsupported_increment_vs_p0", "context_loss_not_increased",
        "false_persistence_not_increased", "duplicate_atom_absolute",
        "duplicate_atom_not_above_p0", "over_split_absolute",
        "over_split_not_above_p0", "severe_forbidden_zero",
        "polarity_condition_inversion_zero", "non_atomic_direction_consistent",
        "no_systematic_cap_or_gate_loss",
    }
    checks.eq("gate_A1_item_ids", a1_ids, expected_a1)
    checks.eq("gate_A1_each_has_reason",
              all(item.get("reason") and item.get("status") in {"pass", "fail"}
                  for item in gate_of(result, "A1")["items"]), True)
    a2_ids = {item["id"] for item in gate_of(result, "A2")["items"]}
    expected_a2 = {
        "confirmation_coverage", "confirmation_faithfulness",
        "confirmation_complete_set", "confirmation_non_atomic_obligation",
        "duplicate_atom_absolute", "over_split_absolute",
        "severe_forbidden_zero", "polarity_condition_inversion_zero",
        "a2_coverage_gain_vs_a1", "a2_non_atomic_reduction_vs_a1",
        "a2_minimum_improvement",
    }
    checks.eq("gate_A2_item_ids", a2_ids, expected_a2)
    checks.eq("gate_A2_each_has_reason",
              all(item.get("reason") and item.get("status") in {"pass", "fail"}
                  for item in gate_of(result, "A2")["items"]), True)

                      
    case = fixture_case("C-invalid", "confirmation", ("g1", "g2"))
    base_parts = build_bundle_parts([case], perfect_specs([case]))
    checks.eq("validation_base_ok", score_parts(copy.deepcopy(base_parts))["counts"]["cases"], 1)

    parts = copy.deepcopy(base_parts)
    for annotation in parts["annotations"]:
        if annotation["blind_id"] == parts["raw"]["A1"][0]["blind_id"]:
            annotation["atoms"][0]["notes"] = "原文依据：对话第二段"
    checks.eq("validation_notes_accepted",
              score_parts(parts)["counts"]["cases"], 1)

    parts = copy.deepcopy(base_parts)
    for annotation in parts["annotations"]:
        if annotation["blind_id"] == parts["raw"]["A1"][0]["blind_id"]:
            for atom in annotation["atoms"]:
                atom["core_entailed_gold_ids"] = []
    full_only = score_parts(parts)
    checks.close("validation_full_implies_core_coverage",
                 metric(full_only, "A1", "confirmation", "direct",
                        "source_coverage", "value"), 1.0)
    checks.close("validation_full_implies_core_complete",
                 metric(full_only, "A1", "confirmation", "direct",
                        "complete_set", "rate"), 1.0)

    parts = copy.deepcopy(base_parts)
    parts["raw"]["A1"] = []
    checks.raises("validation_missing_arm", "schema_raw_missing",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["annotations"] = [ann for ann in parts["annotations"]
                            if ann["blind_id"] != parts["raw"]["A1"][0]["blind_id"]]
    checks.raises("validation_missing_annotation", "annotation_missing",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    for ann in parts["annotations"]:
        if ann["blind_id"] == parts["raw"]["A1"][0]["blind_id"]:
            ann["atoms"].pop()
    checks.raises("validation_missing_atom", "annotation_set_mismatch",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    for ann in parts["annotations"]:
        if ann["blind_id"] == parts["raw"]["A1"][0]["blind_id"]:
            ann["atoms"][0]["core_entailed_gold_ids"].append("NO-SUCH-GOLD")
    checks.raises("validation_gold_ref", "annotation_gold_ref",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    for ann in parts["annotations"]:
        if ann["blind_id"] == parts["raw"]["A1"][0]["blind_id"]:
            ann["atoms"][0]["matched_forbidden_ids"].append("NO-SUCH-FORBIDDEN")
    checks.raises("validation_forbidden_ref", "annotation_forbidden_ref",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["raw"]["A1"][0]["split"] = "dev"
    checks.raises("validation_split_mismatch", "schema_raw",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    if len(parts["raw"]["A1"][0]["parsed_candidates"]) >= 2:
        parts["raw"]["A1"][0]["parsed_candidates"][1]["candidate_id"] = \
            parts["raw"]["A1"][0]["parsed_candidates"][0]["candidate_id"]
    checks.raises("validation_duplicate_candidate", "schema_candidate",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["annotations"][0]["adjudicated"] = False
    checks.raises("validation_not_adjudicated", "schema_annotation",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["annotations"][0]["atoms"][0]["context_errors"] = ["bogus"]
    checks.raises("validation_context_token", "schema_annotation",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["raw"]["A1"][0]["status"] = "ok"
    parts["raw"]["A1"][0]["parsed_candidates"][0]["kept_status"] = "dropped_evidence"
    checks.raises("validation_status_consistency", "schema_raw",
                  lambda: score_parts(parts))

    parts = copy.deepcopy(base_parts)
    parts["ledger"] = [row for row in parts["ledger"]
                       if not (row["case_id"] == "C-invalid" and row["arm"] == "A2")]
    checks.raises("validation_missing_ledger", "ledger_missing_attempt",
                  lambda: score_parts(parts))

                                                  
    pos = fixture_case("C-empty-amb", "confirmation", ("g1",))
    parts = build_bundle_parts([pos], [
        spec(pos, "P0", []), spec(pos, "A1", []),
        spec(pos, "A2", perfect_items(pos, "A2")),
    ])
    parts["annotations"][0]["blind_id"] = "opaque-not-a-real-blind-id"
    checks.raises("validation_empty_ambiguous", "annotation_empty_unmatched",
                  lambda: score_parts(parts))

                                                   
    pos = fixture_case("C-lww", "confirmation", ("g1",))
    parts = build_bundle_parts([pos], [
        spec(pos, "P0", perfect_items(pos, "P0")),
        spec(pos, "A1", []),
        spec(pos, "A2", perfect_items(pos, "A2")),
    ])
    old_blind = parts["raw"]["A1"][0]["blind_id"]
    parts["annotations"] = [ann for ann in parts["annotations"]
                            if ann["blind_id"] != old_blind]
    new_raw = fixture_raw(pos, "A1", perfect_items(pos, "A1"), attempt=2)["raw"]
    parts["raw"]["A1"].append(new_raw)
    parts["annotations"].append(fixture_annotation(
        new_raw, [atom for _, atom in perfect_items(pos, "A1")]))
    parts["ledger"].append({"case_id": "C-lww", "split": "confirmation",
                            "arm": "A1", "attempt": 2,
                            "usage": {"prompt_tokens": 10,
                                      "completion_tokens": 5,
                                      "total_tokens": 15},
                            "elapsed_s": 1.0})
    result = score_parts(parts)
    checks.close("lww_coverage", metric(result, "A1", "confirmation", "direct",
                                         "source_coverage", "value"), 1.0)
    checks.truthy("lww_warning", any("last-write-wins" in warning
                                     for warning in result["validation"]["warnings"]))

                                             
    pos = fixture_case("C-a2nr-pos", "confirmation", ("g1",))
    neg = fixture_case("C-a2nr-neg", "confirmation", ())
    parts = build_bundle_parts([pos, neg], [
        spec(pos, "P0", []), spec(pos, "A1", perfect_items(pos, "A1")),
        spec(pos, "A2", []),
        spec(neg, "P0", []), spec(neg, "A1", []), spec(neg, "A2", []),
    ])
    for case_obj in (pos, neg):
        a2_raw = next(row for row in parts["raw"]["A2"]
                      if row["case_id"] == case_obj["case_id"])
        parts["raw"]["A2"] = [row for row in parts["raw"]["A2"]
                              if row["case_id"] != case_obj["case_id"]]
        parts["annotations"] = [ann for ann in parts["annotations"]
                                if ann["blind_id"] != a2_raw["blind_id"]]
        parts["ledger"] = [row for row in parts["ledger"]
                           if not (row["case_id"] == case_obj["case_id"]
                                   and row["arm"] == "A2")]
    result = score_parts(parts)
    checks.eq("a2_not_run_count", result["counts"]["a2_not_run"], 2)
    checks.eq("a2_not_run_input_raw", result["counts"]["input_raw_sets"], 6)
    checks.eq("a2_not_run_call_raw", result["counts"]["call_raw_sets"], 4)
    checks.eq("a2_not_run_annotations_input", result["counts"]["annotations"], 4)
    checks.eq("a2_not_run_matched_sets", result["counts"]["matched_output_sets"], 6)
    checks.eq("a2_not_run_cases", result["diagnostics"]["a2_not_run"]["cases"],
              ["C-a2nr-neg", "C-a2nr-pos"])
    checks.eq("a2_not_run_cost", result["costs"]["budget"]["a2_not_run_cases"], 2)
    checks.close("a2_not_run_a2_complete", metric(result, "A2", "confirmation",
                                                  "direct", "complete_set", "rate"), 0.5)
    checks.close("a2_not_run_positive_coverage",
                 next(row["direct_coverage"] for row in
                      result["arms"]["A2"]["all"]["case_rows"]
                      if row["case_id"] == "C-a2nr-pos"), False)
    checks.eq("a2_not_run_negative_complete",
              next(row["direct_complete"] for row in
                   result["arms"]["A2"]["all"]["case_rows"]
                   if row["case_id"] == "C-a2nr-neg"), True)
    checks.truthy("a2_not_run_warning",
                  any("未采集" in warning for warning in result["validation"]["warnings"]))

                                                            
                                    
    pos = fixture_case("C-a2ph", "confirmation", ("g1",))
    parts = build_bundle_parts([pos], [
        spec(pos, "P0", []), spec(pos, "A1", perfect_items(pos, "A1")),
        spec(pos, "A2", []),
    ])
    a2_raw = parts["raw"]["A2"][0]
    a2_raw["call"]["attempted"] = False
    a2_raw["elapsed_s"] = 0.0
    a2_raw["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    parts["ledger"] = [row for row in parts["ledger"]
                       if not (row["case_id"] == "C-a2ph" and row["arm"] == "A2")]
    result = score_parts(parts)
    checks.eq("a2_placeholder_not_run", result["counts"]["a2_not_run"], 1)
    checks.eq("a2_placeholder_input_raw", result["counts"]["input_raw_sets"], 3)
    checks.eq("a2_placeholder_call_raw", result["counts"]["call_raw_sets"], 2)
    checks.close("a2_placeholder_complete", metric(result, "A2", "confirmation",
                                                   "direct", "complete_set", "rate"), 0.0)
    checks.truthy("a2_placeholder_warning",
                  any("未运行占位" in warning
                      for warning in result["validation"]["warnings"]))

                                        
    case = fixture_case("C-a2gap", "confirmation", ("g1",))
    parts = build_bundle_parts([case], perfect_specs([case]))
    a2_raw = parts["raw"]["A2"][0]
    parts["raw"]["A2"] = []
    parts["annotations"] = [ann for ann in parts["annotations"]
                            if ann["blind_id"] != a2_raw["blind_id"]]
    parts["ledger"] = [row for row in parts["ledger"]
                       if not (row["case_id"] == "C-a2gap" and row["arm"] == "A2")]
    checks.raises("a2_gap_rejected", "schema_raw_missing",
                  lambda: score_parts(parts))

                                      
    case = fixture_case("C-loss", "confirmation", ("g1",), family="loss-family")
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", [fixture_item("C-loss-A1", kept="dropped_evidence",
                                       core=["g1"], full=["g1"])]),
        spec(case, "A2", perfect_items(case, "A2")),
    ])
    checks.truthy("systematic_gate_fail",
                  not item_of(gate_of(result, "A1"),
                              "no_systematic_cap_or_gate_loss")["pass"])
    checks.eq("systematic_loss_groups",
              item_of(gate_of(result, "A1"),
                      "no_systematic_cap_or_gate_loss")["value"]["loss_groups"],
              ["confirmation/loss-family"])

                                
    case = fixture_case("C-a2-severe", "confirmation", ("g1",),
                        forbidden={"F-severe": "severe"})
    result = build_bundle([case], [
        spec(case, "P0", perfect_items(case, "P0")),
        spec(case, "A1", perfect_items(case, "A1")),
        spec(case, "A2", [fixture_item("C-a2-severe-A2", core=["g1"], full=["g1"],
                                       context_errors=["polarity"],
                                       forbidden=["F-severe"])]),
    ])
    checks.truthy("a2_severe_item_fail",
                  not item_of(gate_of(result, "A2"), "severe_forbidden_zero")["pass"])
    checks.truthy("a2_inversion_item_fail",
                  not item_of(gate_of(result, "A2"),
                              "polarity_condition_inversion_zero")["pass"])
    checks.eq("a2_severe_decision", gate_of(result, "A2")["decision"], "reject")

                               
    missing = ROOT / "data" / "__definitely_missing_score_input__.jsonl"
    checks.raises("missing_input_error", "input_missing",
                  lambda: read_jsonl_file(missing))
    checks.eq("missing_input_not_created", missing.exists(), False)

                                        
    sha_result = score_parts(copy.deepcopy(base_parts))
    rendered = pretty(sha_result)
    checks.truthy("utf8_chinese_visible", "要求" in rendered)
    checks.eq("utf8_no_ascii_escape", "\\u8986" not in rendered, True)
    checks.eq("inputs_all_six", sorted(sha_result["inputs"]),
              sorted(["cases-v1.jsonl", "raw-P0.jsonl", "raw-A1.jsonl",
                      "raw-A2.jsonl", "call-ledger.jsonl", "annotations-v1.jsonl"]))
    checks.eq("input_sha_preserved",
              sha_result["inputs"]["cases-v1.jsonl"]["sha256"],
              hashlib.sha256(b"cases-v1.jsonl").hexdigest())
    checks.eq("input_sha_alias_preserved",
              sha_result["input_sha256"]["cases-v1.jsonl"],
              hashlib.sha256(b"cases-v1.jsonl").hexdigest())
    checks.eq("script_sha_preserved", sha_result["script_sha256"], "c" * 64)

                                                          
    case_a = fixture_case("C-infer-A", "confirmation", ("g1",))
    case_b = fixture_case("C-infer-B", "confirmation", ("g1",))
    parts = build_bundle_parts([case_a, case_b], [
        spec(case_a, "P0", []), spec(case_a, "A1", perfect_items(case_a, "A1")),
        spec(case_a, "A2", perfect_items(case_a, "A2")),
        spec(case_b, "P0", []), spec(case_b, "A1", perfect_items(case_b, "A1")),
        spec(case_b, "A2", perfect_items(case_b, "A2")),
    ])
    for annotation in parts["annotations"]:
        if annotation["atoms"] == []:
            raw = next(raw for arm in ARMS for raw in parts["raw"][arm]
                       if raw["blind_id"] == annotation["blind_id"])
            annotation["blind_id"] = "blind-" + raw["case_id"]
    inferred = score_parts(parts)
    checks.eq("inferred_empty_ok", inferred["counts"]["cases"], 2)
    checks.truthy("inferred_empty_warning", any("case_id" in warning
                                               for warning in inferred["validation"]["warnings"]))

    metadata_parts = copy.deepcopy(parts)
    for annotation in metadata_parts["annotations"]:
        if annotation["atoms"] == []:
            annotation["case_id"] = annotation["blind_id"].replace("blind-", "")
    metadata_result = score_parts(metadata_parts)
    checks.eq("metadata_empty_ok", metadata_result["counts"]["cases"], 2)
    case_c = fixture_case("C-infer-C", "confirmation", ("g1",))
    bad_metadata = build_bundle_parts([case_a, case_b, case_c], [
        spec(case_a, "P0", []), spec(case_a, "A1", perfect_items(case_a, "A1")),
        spec(case_a, "A2", perfect_items(case_a, "A2")),
        spec(case_b, "P0", []), spec(case_b, "A1", perfect_items(case_b, "A1")),
        spec(case_b, "A2", perfect_items(case_b, "A2")),
        spec(case_c, "P0", perfect_items(case_c, "P0")),
        spec(case_c, "A1", perfect_items(case_c, "A1")),
        spec(case_c, "A2", perfect_items(case_c, "A2")),
    ])
    for annotation in bad_metadata["annotations"]:
        if annotation["atoms"] == []:
            raw = next(raw for arm in ARMS for raw in bad_metadata["raw"][arm]
                       if raw["blind_id"] == annotation["blind_id"])
            annotation["blind_id"] = "blind-" + raw["case_id"]
            annotation["case_id"] = raw["case_id"]
    target = next(annotation for annotation in bad_metadata["annotations"]
                  if annotation["atoms"] == [])
    target["case_id"] = case_c["case_id"]
    checks.raises("metadata_empty_mismatch", "annotation_empty_case_mismatch",
                  lambda: score_parts(bad_metadata))

    return checks.report()


                                              


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="原子记忆试点独立评分器")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="离线合成自测，不读真实数据、不创建文件")
    sub.add_parser("score", help="只读冻结输入并写 data/results.json")
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective and effective[0] in {"--check", "-c"}:
        effective[0] = "check"
    elif effective and effective[0] in {"--score", "-s"}:
        effective[0] = "score"
    args = parser.parse_args(effective)
    if args.command == "check":
        try:
            return 0 if run_selftest() else 2
        except Exception as exc:                                 
            print(json.dumps({"check": "atomic-memory-score-selftest", "passed": False,
                              "error": {"type": type(exc).__name__,
                                        "message": str(exc)}},
                             ensure_ascii=False, sort_keys=True))
            return 2
    try:
        return command_score()
    except ScoreError as exc:
        print(json.dumps({"ok": False,
                          "error": {"code": exc.code, "message": exc.message}},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
