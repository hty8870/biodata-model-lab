                       
"""记忆关系研究数据的行契约与访存材料化（research-only）。

每行是 `spec.json` 的 schema：规范脱敏 request（`candidate/existing/evidence/context`）
+ 固定 spec 七类标签（历史数据格式，不跟随当前生产五叶）+ 期望引用。
`canonical_request` 是统一入口：合成原始形状（principal/
endpoint_fp/project_id + 可选权限位）与已脱敏形状都规范成同一形状——身份只用于推导
作用域标签后丢弃，权限位不采信，tier/permanence 属可选构造元数据、不补齐也不按标签
推断，context 未知键（label/expected 等）直接拒绝。`materialize_request` 再恢复固定
合成身份/端点，复用生产投影与输出闸（`assess_relations` 本地 predictor 通道、>64 规则
走真实分批）做契约合法性验证：通过只说明该行能被生产接口如实承载，不证明标签语义正确
——构造标签不是人工金标。旧七类标签回放还要求当前生产仍承载同一关系词表；生产类别合同
与 spec 不相容时 `validate_row` 显式抛 `legacy_production_contract_unavailable`，不拿五叶
结果冒充七类回放。`phenomena` 支持列表（构造标记）或字典（构造/评审元数据）。
模块不发请求、不写记忆、不读环境。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dataset_recommender.agent import memory_relations as M
from dataset_recommender.agent.agent_exec import _endpoint_fp_from_config
from dataset_recommender.llm.llm_client import LLMConfig

from research.memory_relations import features as F

__all__ = [
    "CALIBRATION_ROLES", "CLASSES", "CONTEXT_KEYS", "FIELDS", "LABEL_ORIGINS", "PERMANENCE",
    "REQUEST_FIELDS", "SPLITS", "RowError", "canonical_request", "feature_view", "features_from_request",
    "load_rows", "local_config", "materialize_request", "validate_row",
]

FIELDS = ("id", "group_id", "split", "request", "label", "label_origin",
          "related_ids", "evidence_refs", "phenomena")
REQUEST_FIELDS = ("candidate", "existing", "evidence", "context")
                                                       
CLASSES = F.CLASSES
LABEL_ORIGINS = frozenset({"construction", "teacher_single", "teacher_consensus",
                           "agent_review", "human_anchor"})
SPLITS = frozenset({"train", "dev", "calibration", "heldout", "ood", "set"})
CALIBRATION_ROLES = ("prob", "policy", "audit")
                                                               
PERMANENCE = frozenset({"persistent", "temporary", "unknown"})
TIERS = frozenset({"explicit", "implicit"})
                                 
CONTEXT_KEYS = frozenset({"scope", "replaceable_ids", "inherited_global_ids"})
                           
IDENTITY_FIELDS = ("principal", "endpoint_fp", "project_id")
_RULE_METADATA = ("tier", "permanence")
_RELATED_REQUIRED = frozenset({"duplicate", "contextual", "supersede", "conflict"})
_PHENOMENA_MAX = 8192
_PHENOMENA_DEPTH = 4

SYNTH_PRINCIPAL = "synth-memory-principal"
SYNTH_PROJECT = "synth-memory-project"
LOCAL_PROVIDER = "local"
LOCAL_BASE_URL = "https://model.example/v1"
LOCAL_MODEL = "memory-relations-local"

_REASON = "构造基线行按标签给出关系判断。"


class RowError(ValueError):
    """行契约错误；`code` 是有界机器码，未知字段不回显。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def local_config() -> LLMConfig:
    """本地 predictor 通道的假配置：本机地址、无 key、不触网（model.example）。"""
    return LLMConfig(enable_llm=True, mock_llm=False, provider=LOCAL_PROVIDER,
                     api_key=None, base_url=LOCAL_BASE_URL, model=LOCAL_MODEL)


def _identifier(value: object) -> str:
    if type(value) is not str or not M._ID.fullmatch(value):
        raise RowError("invalid_row")
    return value


def _raw_scope(row: dict) -> tuple[object, object]:
    label = row.get("scope")
    if label is not None and label not in ("project", "global", "same", "inherited_global"):
        raise RowError("invalid_scope")
    project = row.get("project_id")
    if project is not None and (type(project) is not str or len(project) > 128
                                or project != project.strip()):
        raise RowError("invalid_scope")
    return label, project


def _canonical_rule(row: object, *, with_id: bool) -> dict:
    """规范单条规则：只保留投影可见字段；身份字段丢弃，tier/permanence 可选保留。"""
    if type(row) is not dict:
        raise RowError("invalid_rule")
    allowed = {"text", "trigger", "tier", "permanence", "evidence", "scope"} | set(IDENTITY_FIELDS)
    if with_id:
        allowed.add("id")
    if set(row) - allowed or "text" not in row:
        raise RowError("invalid_rule")
    text = row["text"]
    if type(text) is not str or not text.strip():
        raise RowError("invalid_rule")
    trigger = row.get("trigger", "")
    if type(trigger) is not str:
        raise RowError("invalid_rule")
    if row.get("tier") is not None and row["tier"] not in TIERS:
        raise RowError("invalid_rule")
    if row.get("permanence") is not None and row["permanence"] not in PERMANENCE:
        raise RowError("invalid_rule")
    quotes = row.get("evidence", [])
    if type(quotes) is not list or any(type(item) is not str or not item.strip() for item in quotes):
        raise RowError("invalid_rule")
    out = {"text": text, "trigger": trigger, "evidence": copy.deepcopy(quotes)}
    for name in _RULE_METADATA:
        if row.get(name) is not None:
            out[name] = row[name]
    if with_id:
        out["id"] = _identifier(row.get("id"))
    return out


def _rule_scope(row: dict, cand_scope: str, cand_project: object) -> str:
    """作用域标签：优先显式标签，否则由合成 project_id 与候选比较得出。"""
    label, project = _raw_scope(row)
    if label == "inherited_global":
        if cand_scope != "project" or project not in (None, ""):
            raise RowError("scope_mismatch")
        return "inherited_global"
    if label == "same":
        if project is not None and cand_project is not None and project != cand_project:
            raise RowError("scope_mismatch")
        return "same"
    if label is not None:
        raise RowError("invalid_scope")
    if project is None:
        return "same"
    if project == cand_project:
        return "same"
    if project == "" and cand_scope == "project":
        return "inherited_global"
    raise RowError("scope_mismatch")


def canonical_request(request: object) -> dict:
    """统一入口：合成原始形状或脱敏形状 -> 投影可见的规范脱敏 request。

    - 身份（principal/endpoint_fp/project_id）只用于推导作用域标签，规范后丢弃；
      权限位（replaceable_ids/inherited_global_ids）不采信，由生产投影按固定合成身份重算。
    - tier/permanence 是可选构造元数据：原样保留、不补齐、不按标签推断。
    - context 只接受已登记键；label/expected 等未知键直接拒绝。
    """
    if type(request) is not dict or set(request) != set(REQUEST_FIELDS):
        raise RowError("invalid_request")
    candidate, existing, evidence, context = (request[name] for name in REQUEST_FIELDS)
    if type(candidate) is not dict or type(existing) is not list or type(evidence) is not list:
        raise RowError("invalid_request")
    if type(context) is not dict or set(context) - CONTEXT_KEYS:
        raise RowError("invalid_context")
    for name in ("replaceable_ids", "inherited_global_ids"):
        value = context.get(name)
        if value is not None and (type(value) is not list
                                  or any(type(item) is not str for item in value)):
            raise RowError("invalid_context")
    candidate_label, candidate_project = _raw_scope(candidate)
    if candidate_label not in (None, "project", "global"):
        raise RowError("invalid_scope")
    if candidate_project is None and any(type(row) is dict and row.get("project_id") is not None
                                         for row in existing):
        raise RowError("invalid_scope")
    cand_scope = candidate_label if candidate_label is not None else (
        "global" if candidate_project == "" else "project")
    context_scope = context.get("scope")
    if context_scope is not None and context_scope not in ("project", "global"):
        raise RowError("invalid_context")
    if context_scope is not None and context_scope != cand_scope:
        raise RowError("invalid_context")
    candidate_out = _canonical_rule(candidate, with_id=False)
    candidate_out["scope"] = cand_scope
    rules = []
    for row in existing:
        rule = _canonical_rule(row, with_id=True)
        rule["scope"] = _rule_scope(row, cand_scope, candidate_project)
        rules.append(rule)
    sources = []
    for row in evidence:
        if type(row) is not dict or set(row) != {"id", "text"}:
            raise RowError("invalid_request")
        text = row["text"]
        if type(text) is not str or not text.strip():
            raise RowError("invalid_request")
        sources.append({"id": _identifier(row["id"]), "text": text})
    return {"candidate": candidate_out, "existing": rules, "evidence": sources,
            "context": {"scope": cand_scope}}


def _canonical_parts(canonical: dict) -> tuple[dict, list, list, dict]:
    return (canonical["candidate"], canonical["existing"], canonical["evidence"],
            canonical["context"])


def _request_parts(request: object) -> tuple[dict, list, list, dict]:
    return _canonical_parts(canonical_request(request))


def materialize_request(request: object) -> tuple[dict, list, dict, LLMConfig]:
    """原始合成形状或脱敏形状 -> 生产投影入参：固定合成身份/端点 + 作用域映射。

    先过 `canonical_request`（身份丢弃、权限位不采信、字段不按标签补齐），再恢复固定
    合成身份；返回全新对象，调用方改动不回写存储行。
    """
    return _materialize_parts(_request_parts(request))


def _materialize_parts(parts: tuple) -> tuple[dict, list, dict, LLMConfig]:
    candidate, rules, sources, context = parts
    cfg = local_config()
    endpoint_fp = _endpoint_fp_from_config(cfg)
    principal, scope = SYNTH_PRINCIPAL, context["scope"]
    project_id = SYNTH_PROJECT if scope == "project" else ""
    out_candidate = {
        "principal": principal, "endpoint_fp": endpoint_fp, "project_id": project_id,
        "text": candidate["text"], "trigger": candidate["trigger"],
        "evidence": list(candidate["evidence"]),
    }
    for name in _RULE_METADATA:
        if name in candidate:
            out_candidate[name] = candidate[name]
    out_existing = []
    for rule in rules:
        row = {"principal": principal, "endpoint_fp": endpoint_fp,
               "project_id": "" if rule["scope"] == "inherited_global" else project_id,
               "id": rule["id"], "text": rule["text"], "trigger": rule["trigger"],
               "evidence": list(rule["evidence"])}
        for name in _RULE_METADATA:
            if name in rule:
                row[name] = rule[name]
        out_existing.append(row)
    out_evidence = [{"id": source["id"], "text": source["text"]} for source in sources]
    return out_candidate, out_existing, out_evidence, cfg


def _check_projection(parts: tuple, candidate: dict, existing: list, evidence: list) -> None:
    """分批走真实投影：每条存储规则都必须出现在某个批次里且字段逐值一致。

    超单批容量时生产投影会切批（尺寸/字节两维），这里复用同一分批函数，不做 top-k 截断。
    """
    stored_candidate, stored_rules, stored_sources, _context = parts
    try:
        batches = M._size_fit_batches(candidate, existing, evidence, None, False)
    except M._Invalid as exc:
        raise RowError("gate_rejected:" + str(exc.code)) from None
    if not batches:
        batches = [[]]
    stored_by_id = {rule["id"]: rule for rule in stored_rules}
    covered = []
    for batch in batches:
        try:
            projection, _meta = M._request(candidate, batch, evidence, None)
        except M._Invalid as exc:
            raise RowError("gate_rejected:" + str(exc.code)) from None
        for name in ("text", "trigger", "evidence", "scope"):
            if stored_candidate[name] != projection["candidate"][name]:
                raise RowError("projection_mismatch")
        if projection["candidate"].get("tier") != stored_candidate.get("tier"):
            raise RowError("projection_mismatch")
        if stored_sources != projection["evidence"]:
            raise RowError("projection_mismatch")
        for projected in projection["existing"]:
            stored = stored_by_id.get(projected["id"])
            if stored is None:
                raise RowError("projection_mismatch")
            if stored["scope"] != projected["scope"] or stored.get("tier") != projected.get("tier"):
                raise RowError("projection_mismatch")
            for name in ("text", "trigger", "evidence"):
                if stored[name] != projected[name]:
                    raise RowError("projection_mismatch")
            covered.append(projected["id"])
    if sorted(covered) != sorted(stored_by_id):
        raise RowError("projection_mismatch")


def _replay_predictor(row: dict):
    """批次感知回放：批次内只声明该批可见的引用，无目标规则的批次退回 complement。

    逐批判定不能硬套全 set 标签；不确定集行每批都如实返回 uncertain（合并仍为 uncertain）。
    """
    label = row["label"]
    related = list(row["related_ids"])
    refs = list(row["evidence_refs"])

    def predict(batch: dict) -> dict:
        visible = {rule["id"] for rule in batch.get("existing", [])}
        batch_related = [identifier for identifier in related if identifier in visible]
        if label == "uncertain":
            return {"relation": "uncertain", "related_ids": [], "reason": _REASON,
                    "evidence_refs": [], "error_code": None}
        if label in _RELATED_REQUIRED and not batch_related:
            return {"relation": "complement", "related_ids": [], "reason": _REASON,
                    "evidence_refs": refs, "error_code": None}
        return {"relation": label, "related_ids": batch_related, "reason": _REASON,
                "evidence_refs": refs, "error_code": None}

    return predict


def _depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(item) for item in value), default=0)
    return 0


def _check_phenomena(value: object) -> None:
    """phenomena 支持列表（构造标记）或字典（构造/评审元数据），两者都做有界校验。"""
    if type(value) is list:
        if any(type(item) is not str or not item.strip() or len(item) > 64 for item in value) \
                or len(set(value)) != len(value):
            raise RowError("invalid_phenomena")
        return
    if type(value) is not dict or not value:
        raise RowError("invalid_phenomena")
    if any(type(key) is not str or not key.strip() or len(key) > 64 for key in value):
        raise RowError("invalid_phenomena")
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise RowError("invalid_phenomena") from None
    if len(serialized) > _PHENOMENA_MAX or _depth(value) > _PHENOMENA_DEPTH:
        raise RowError("invalid_phenomena")
    labels = value.get("pair_labels")
    if labels is not None:
        if type(labels) is not list or any(
                type(item) is not dict or not {"rule_id", "relation"} <= set(item)
                or set(item) - {"rule_id", "relation", "origin"}
                or type(item["rule_id"]) is not str or item["relation"] not in CLASSES
                or ("origin" in item and item["origin"] not in ("target", "redundant", "distractor"))
                for item in labels):
            raise RowError("invalid_phenomena")
    construction = value.get("construction")
    if construction is not None and (
            type(construction) is not list or any(
                type(item) is not str or not item.strip() or len(item) > 64
                for item in construction)):
        raise RowError("invalid_phenomena")
    if "root_id" in value:
        _identifier(value["root_id"])
        for key in ("domain", "family", "author", "rationale", "variant"):
            item = value.get(key)
            if type(item) is not str or not item.strip() or len(item) > 2000:
                raise RowError("invalid_phenomena")
    if "calibration_role" in value and value["calibration_role"] not in CALIBRATION_ROLES:
        raise RowError("invalid_calibration_metadata")
    if "operating_primary" in value and type(value["operating_primary"]) is not bool:
        raise RowError("invalid_calibration_metadata")


def _legacy_replay_available() -> bool:
    """旧七类回放的前提：当前生产关系空间仍承载 spec 的同一词表。

    生产已换五叶时，`M.assess_relations` 的输出类别与历史标签不相容；此时显式声明
    回放不可用，绝不把五叶结果映射/裁剪成七类。完整旧回放只使用已冻结原 commit。
    """
    return set(CLASSES) <= set(M.RELATIONS)


def validate_row(row: object) -> None:
    """结构 + 分批真闸回放；失败抛 `RowError`，通过返回 None（不声称语义正确）。"""
    if type(row) is not dict or set(row) != set(FIELDS):
        raise RowError("invalid_row")
    _identifier(row["id"])
    _identifier(row["group_id"])
    if any(type(row[key]) is not str for key in ("split", "label", "label_origin")) \
            or row["split"] not in SPLITS or row["label"] not in CLASSES \
            or row["label_origin"] not in LABEL_ORIGINS:
        raise RowError("invalid_row")
    references = {}
    for name in ("related_ids", "evidence_refs"):
        value = row[name]
        if type(value) is not list:
            raise RowError("invalid_row")
        references[name] = [_identifier(item) for item in value]
        if len(set(references[name])) != len(value):
            raise RowError("invalid_row")
    _check_phenomena(row["phenomena"])
    if type(row["phenomena"]) is dict and row["split"] != "calibration" \
            and "calibration_role" in row["phenomena"]:
        raise RowError("invalid_calibration_metadata")
    blank_references = not references["related_ids"] and not references["evidence_refs"]
    if (row["label"] == "uncertain") != blank_references:
        raise RowError("invalid_row")
    parts = _request_parts(row["request"])
    stored_rules, stored_sources = parts[1], parts[2]
    rule_ids = {rule["id"] for rule in stored_rules}
    source_ids = {source["id"] for source in stored_sources}
    if any(identifier not in rule_ids for identifier in references["related_ids"]) \
            or any(identifier not in source_ids for identifier in references["evidence_refs"]):
        raise RowError("unknown_reference")
    candidate, existing, evidence, cfg = _materialize_parts(parts)
    _check_projection(parts, candidate, existing, evidence)
    if not _legacy_replay_available():
        raise RowError("legacy_production_contract_unavailable")
    out = M.assess_relations(candidate, existing, evidence, cfg,
                             predictor=_replay_predictor(row))
    if row["label"] == "uncertain":
        if out["relation"] != "uncertain" or out["error_code"] != "model_uncertain" \
                or out["related_ids"] or out["evidence_refs"]:
            raise RowError("gate_rejected:" + str(out["error_code"]))
    elif out["error_code"] is not None or out["relation"] != row["label"] \
            or out["related_ids"] != references["related_ids"] \
            or out["evidence_refs"] != references["evidence_refs"]:
        raise RowError("gate_rejected:" + str(out["error_code"]))


def feature_view(request: object) -> dict:
    """只投影真实可见语义面：candidate/existing 的 text、trigger 与 evidence 原文。

    tier/permanence 属构造元数据、真实 predictor 不必然提供，不进入特征；id、标签、
    group/split、作用域权限位与引用集合同样不投影。候选引文是证据原文子串，不重复携带。
    """
    candidate, rules, sources, _context = _request_parts(request)

    def rule_view(rule: dict) -> dict:
        return {"text": rule["text"], "trigger": rule["trigger"]}

    return {
        "candidate": rule_view(candidate),
        "existing": [rule_view(rule) for rule in rules],
        "evidence": [source["text"] for source in sources],
    }


features_from_request = feature_view


def load_rows(path: object) -> list[dict]:
    """按行读 JSONL；空行跳过，其它解析错误按行号 fail-closed。"""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise RowError("invalid_json_line:" + str(number)) from None
            if type(row) is not dict:
                raise RowError("invalid_json_line:" + str(number))
            rows.append(row)
    return rows
