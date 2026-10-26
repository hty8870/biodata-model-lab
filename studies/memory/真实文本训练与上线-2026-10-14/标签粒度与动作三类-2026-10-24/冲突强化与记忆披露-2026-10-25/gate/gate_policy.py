"""现行闸与三态闸的研究侧实现（生产代码只读复用，不写入 repo）。

三态策略（预注册设计 §5.1）：
1. **硬约束**先行且对三态一律生效：schema、逐字证据、覆盖数、作用域、预算。
2. 高置信**完整注入形态**才 `reject`；词表命中但形态不完整 → `quarantine`；
   其余 `keep`。
3. `quarantine` 永不自动 active；只能经 pending/assessment/human-confirm 离开。

生产接缝（本原型不修改生产，因此研究侧只能模拟）：
- `dream._apply_gates` 只有 kept/dropped 两个出口，命中词表即丢弃，没有 quarantine 通道；
- `dream` 的记忆档词表含裸 `忽略`、`system`、`prompt`、`ignore`，合法界面/英文句子会无声消失；
- `knowledge.normalize_body` 用规则档词表（裸 `prompt|system|ignore`），与 dream 记忆档
  口径不一致：同一句合法英文能过 dream 记忆档却过不了知识正文闸，反之合法动画句能过
  正文闸却被 dream 记忆档丢弃；
- 作用域在 `_apply_gates` 层不校验（生产在 `KnowledgeService.propose/_operations` 才校验），
  故现行闸在本层对 scope 越界无感。
"""
from __future__ import annotations

import hashlib
import json
import re

from dataset_recommender.agent import knowledge as K
from dataset_recommender.app.conversation_runs import RunError
from dataset_recommender.llm import dream

GATE_VERSION = "three-state/1"
DEFAULT_MAX_CHARS = 24_000
DEFAULT_MAX_ITEMS = 32

                                                  
                                    
REJECT_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("zh_override_scope", re.compile(
        r"(?:忽略|无视|绕过|跳过|推翻|覆盖|作废|丢弃)[^。；;，,\n]{0,12}"
        r"(?:之前|以上|上面|前面|此前|所有|全部|原有|既有|先前)[^。；;，,\n]{0,8}"
        r"(?:指令|提示词|提示|规则|约束|设定|要求|消息|上下文)")),
    ("zh_role_switch", re.compile(
        r"(?:你现在是|从现在开始你是|假装你是|扮演)[^。；;\n]{0,16}"
        r"(?:开发者模式|管理员模式|DAN|不受(?:任何)?限制|没有(?:任何)?限制|越狱|无限制)")),
    ("zh_prompt_exfil", re.compile(
        r"(?:输出|打印|显示|复述|泄露|告诉我|给我看|贴出)[^。；;\n]{0,10}"
        r"(?:系统提示|提示词|系统指令|隐藏指令|初始指令|开发者指令)"
        r"|(?:系统提示|提示词|系统指令|隐藏指令|初始指令|开发者指令)[^。；;\n]{0,10}"
        r"(?:输出|打印|显示|复述|泄露|给我|贴出)")),
    ("zh_governance_bypass", re.compile(
        r"(?:跳过|绕过|免去|不需要|无需|省去|取消)[^。；;\n]{0,8}"
        r"(?:人工(?:审核|确认|审批|复核|批准)|审核|审批|复核|批准)")),
    ("en_override_scope", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\b[^.;\n]{0,24}"
        r"\b(?:all|any|the|your|these|those)?\s*"
        r"(?:previous|prior|above|earlier|system|developer|initial|safety)\b[^.;\n]{0,16}"
        r"\b(?:instructions?|prompts?|rules?|guidelines?|directives?|messages?)\b", re.I)),
    ("en_role_switch", re.compile(
        r"\b(?:you are now|from now on you are|act as|pretend to be)\b[^.;\n]{0,24}"
        r"\b(?:developer mode|dan|unrestricted|unfiltered|jailbroken)\b"
        r"|\bdo anything now\b", re.I)),
    ("en_prompt_exfil", re.compile(
        r"\b(?:reveal|print|show|repeat|leak|output|display)\b[^.;\n]{0,20}"
        r"\b(?:system|developer|hidden|initial)\s+(?:prompt|message|instructions?)\b"
        r"|\b(?:system|developer|hidden|initial)\s+(?:prompt|message|instructions?)\b"
        r"[^.;\n]{0,20}\b(?:reveal|print|show|repeat|leak|output|display)\b", re.I)),
)

                                  
PRODUCTION_SEAMS: tuple[str, ...] = (
    "dream._apply_gates 只返回 kept/dropped，无 quarantine 通道，三态只能研究侧模拟。",
    "dream 记忆档词表含裸 忽略/覆盖/prompt/system/ignore，合法界面与英文句子被无声丢弃。",
    "knowledge.normalize_body 用规则档词表，裸 prompt/system/ignore 一律 knowledge_injection_blocked；"
    "与 dream 记忆档口径不一致，quarantine 的 keep 建议可能根本进不了草稿。",
    "_apply_gates 不校验作用域；scope 硬约束在本层由研究侧补，生产在 propose/_operations 才拦。",
    "生产 assess 的每日预算按真实日期计（learning_usage），研究 runner 另有自己的假时钟预算，两者不可混算。",
    "生产每日预算拒绝码是 learning_budget_exceeded（LearningStore.charge）；knowledge_budget_exceeded 只在 before_llm 额度回调拒绝时出现，不能混称。",
)

_CANDIDATE_KEYS = ("text", "summary", "trigger", "tier", "evidence")


def candidate_of(item: dict) -> dict:
    """研究条目 → 生产 dream 候选形状（去掉研究标签与作用域）。"""
    return {key: item[key] for key in _CANDIDATE_KEYS if key in item}


def _sha(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


                                                                      

def validate_schema(item: dict) -> list[str]:
    """按生产 dream 的封闭输出上限校验候选形状；返回问题码清单。"""
    problems: list[str] = []
    unknown = set(item) - set(_CANDIDATE_KEYS)
    if unknown:
        problems.append("unknown_fields")
    if "text" not in item:
        problems.append("text_missing")
    else:
        try:
            K.text_value(item["text"], dream.DREAM_TEXT_MAX)
        except RunError:
            problems.append("text_invalid")
    for key, limit in (("summary", dream.DREAM_SUMMARY_MAX), ("trigger", dream.DREAM_TRIGGER_MAX)):
        if key in item:
            try:
                K.text_value(item[key], limit, empty=True)
            except RunError:
                problems.append(f"{key}_invalid")
    evidence = item.get("evidence")
    if type(evidence) is not list or not 1 <= len(evidence) <= dream.DREAM_EVIDENCE_SPANS:
        problems.append("evidence_shape")
    else:
        for span in evidence:
            try:
                K.text_value(span, dream.DREAM_EVIDENCE_MAX)
            except RunError:
                problems.append("evidence_invalid")
                break
    if "tier" in item and item["tier"] not in {dream.DREAM_TIER_EXPLICIT, dream.DREAM_TIER_IMPLICIT}:
        problems.append("tier_invalid")
    return problems


def scope_violations(item: dict, conversations: list[dict], allowed: dict) -> list[str]:
    """研究 batch 作用域语义（与生产 user source 语义对齐，不是放松生产 scope）。

    - `allowed.project == ''`：研究 batch **不限制目的地项目**，不是“只允许全局条目”；
      条目可以是全局或任一项目。
    - `allowed.project != ''`：条目 project 必须精确匹配该 batch 允许的项目。
    - 证据 owner 必须与 batch owner 一致。
    - 条目为 project-specific 时，证据 project 只能是该项目或全局空；
      条目为全局空时，同一 owner 的项目证据均可作为 user 出处（生产
      `KnowledgeService._sources` 对 user 出处允许独立 project_id，不要求与 change project 相同）。
    """
    out: list[str] = []
    scope = item.get("scope") or {}
    if scope.get("owner") != allowed.get("owner"):
        out.append("item_owner")
    allowed_project = allowed.get("project", "")
    item_project = scope.get("project", "")
    if allowed_project and item_project != allowed_project:
        out.append("item_project")
    pools = dream._conversation_evidence_pools(conversations)
    hit_indices: set[int] = set()
    for span in item.get("evidence") or []:
        needle = dream._norm(span)
        if not needle:
            continue
        for index, unit in pools:
            if needle in unit:
                hit_indices.add(index)
    for index in sorted(hit_indices):
        if index >= len(conversations):
            continue
        conv_scope = conversations[index].get("scope") or {}
        if conv_scope.get("owner") != allowed.get("owner"):
            out.append("evidence_owner")
        conv_project = conv_scope.get("project", "")
        if item_project and conv_project not in {item_project, ""}:
            out.append("evidence_project")
    return sorted(set(out))


                                                                       

def injection_form(item: dict) -> tuple[str | None, str | None]:
    """返回 (形态名, 命中片段)；只有完整形态才非空。三个字段都查（text/summary/trigger）。"""
    for field in ("text", "summary", "trigger"):
        value = item.get(field)
        if not value:
            continue
        for name, pattern in REJECT_FORMS:
            match = pattern.search(str(value))
            if match:
                return name, match.group(0)
    return None, None


def meta_matches(item: dict) -> list[dict]:
    """原词表命中（现行动词表逐位复用），返回字段/片段，供审计与 quarantine 依据。"""
    profile = dream._gate_profile_of(item)
    out = []
    for field in ("text", "summary", "trigger"):
        value = item.get(field)
        if not value:
            continue
        match = profile.injection_re.search(str(value))
        if match:
            out.append({"field": field, "match": match.group(0)})
    return out


def production_body_block(item: dict) -> str | None:
    """生产知识正文闸对同一文本的结论（只读探针，不写库）。"""
    try:
        K.normalize_body({"kind": "memory", "title": "", "text": item.get("text", ""),
                          "trigger": item.get("trigger", ""), "payload": {},
                          "permanence": "persistent"})
    except RunError as exc:
        return exc.code
    return None


def _evidence_state(item: dict, pools, profile) -> tuple[bool, int, int]:
    ok, coverage = dream._evidence_checks(item.get("evidence") or [], pools)
    return ok, coverage, dream._required_coverage(profile, item)


                                                                          

ARM_NAMES = ("current", "current_plus_hard", "three_state")


def _payload_chars(cases: list[dict], conversations: list[dict]) -> int:
    return (sum(len(json.dumps(conv, ensure_ascii=False)) for conv in conversations)
            + sum(len(json.dumps(entry["item"], ensure_ascii=False)) for entry in cases))


def _hard_state(item: dict, candidate: dict, conversations: list[dict], pools, allowed: dict) -> dict:
    """schema/scope/evidence 的唯一实现；三臂共享，避免“硬约束”口径漂移。"""
    profile = dream._gate_profile_of(candidate)
    ev_ok, coverage, required = _evidence_state(candidate, pools, profile)
    return {
        "schema_problems": validate_schema(candidate),
        "scope_problems": scope_violations(item, conversations, allowed),
        "evidence_ok": ev_ok,
        "evidence_coverage": coverage,
        "required_coverage": required,
    }


def _hard_reasons(hard: dict) -> list[str]:
    reasons = []
    if hard["schema_problems"]:
        reasons.append("schema")
    if hard["scope_problems"]:
        reasons.append("scope")
    if not hard["evidence_ok"] or hard["evidence_coverage"] < hard["required_coverage"]:
        reasons.append("evidence")
    return reasons


def _row(item: dict, batch: dict, candidate: dict, hard: dict, *,
         decision: str, primary: str, reasons: list[str], form, form_text, metas) -> dict:
    return {
        "id": item["id"], "batch": batch["id"], "label": item["label"],
        "evidence_feasible": item["evidence_feasible"],
        "scope_valid": item["scope_valid"],
        "schema_valid": item.get("schema_valid", True),
        "attack": item.get("attack", "none"), "tier": item.get("tier", ""),
        "decision": decision, "primary_reason": primary, "reasons": reasons,
        "injection_form": form, "injection_form_text": form_text,
        "meta_matches": metas, "evidence_ok": hard["evidence_ok"],
        "evidence_coverage": hard["evidence_coverage"],
        "required_coverage": hard["required_coverage"],
        "scope_problems": hard["scope_problems"],
        "schema_problems": hard["schema_problems"],
        "production_body_blocked": production_body_block(candidate),
        "text_sha256": _sha(item["text"]),
        "raw": candidate,
    }


def _counts(rows: list[dict]) -> dict:
    return {key: sum(1 for row in rows if row["decision"] == key)
            for key in ("keep", "quarantine", "reject")}


                                                                     

def apply_three_state(batch: dict, *, max_chars: int = DEFAULT_MAX_CHARS,
                      max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    conversations = batch["conversations"]
    allowed = batch["allowed"]
    pools = dream._conversation_evidence_pools(conversations)
    cases = batch["cases"]
    payload_chars = _payload_chars(cases, conversations)
    budget_exceeded = payload_chars > max_chars or len(cases) > max_items
    rows = []
    for entry in cases:
        item = entry["item"]
        candidate = candidate_of(item)
        hard = _hard_state(item, candidate, conversations, pools, allowed)
        form, form_text = injection_form(candidate)
        metas = meta_matches(candidate)
        reasons = _hard_reasons(hard)
        if form:
            reasons.append("injection")
        elif metas:
            reasons.append("injection_ambiguous")
        if budget_exceeded:
            reasons.append("budget")
        if budget_exceeded:
            decision, primary = "reject", "budget"
        elif hard["schema_problems"]:
            decision, primary = "reject", "schema"
        elif hard["scope_problems"]:
            decision, primary = "reject", "scope"
        elif not hard["evidence_ok"] or hard["evidence_coverage"] < hard["required_coverage"]:
            decision, primary = "reject", "evidence"
        elif form:
            decision, primary = "reject", "injection"
        elif metas:
            decision, primary = "quarantine", "injection_ambiguous"
        else:
            decision, primary = "keep", "none"
        rows.append(_row(item, batch, candidate, hard, decision=decision, primary=primary,
                         reasons=reasons, form=form, form_text=form_text, metas=metas))
    return {"gate": GATE_VERSION, "batch": batch["id"], "allowed": allowed,
            "budget": {"applicable": True, "chars": payload_chars, "max_chars": max_chars,
                       "items": len(cases), "max_items": max_items, "exceeded": budget_exceeded},
            "counts": _counts(rows), "rows": rows}


def apply_current_plus_hard(batch: dict, *, max_chars: int = DEFAULT_MAX_CHARS,
                            max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    """中间臂：现行动词表丢弃逻辑 + 与三态完全相同的 schema/scope/evidence/budget。

    与 `apply_three_state` 的唯一差别是注入处置：命中词表一律 reject，没有 quarantine，
    也不使用“完整注入形态”结构升级。用于把硬约束收益与三态策略收益拆开归因。
    """
    conversations = batch["conversations"]
    allowed = batch["allowed"]
    pools = dream._conversation_evidence_pools(conversations)
    cases = batch["cases"]
    payload_chars = _payload_chars(cases, conversations)
    budget_exceeded = payload_chars > max_chars or len(cases) > max_items
    rows = []
    for entry in cases:
        item = entry["item"]
        candidate = candidate_of(item)
        hard = _hard_state(item, candidate, conversations, pools, allowed)
        form, form_text = injection_form(candidate)
        metas = meta_matches(candidate)
        reasons = _hard_reasons(hard)
        if metas:
            reasons.append("injection")
        if budget_exceeded:
            reasons.append("budget")
        if budget_exceeded:
            decision, primary = "reject", "budget"
        elif hard["schema_problems"]:
            decision, primary = "reject", "schema"
        elif hard["scope_problems"]:
            decision, primary = "reject", "scope"
        elif not hard["evidence_ok"] or hard["evidence_coverage"] < hard["required_coverage"]:
            decision, primary = "reject", "evidence"
        elif metas:
            decision, primary = "reject", "injection"
        else:
            decision, primary = "keep", "none"
        rows.append(_row(item, batch, candidate, hard, decision=decision, primary=primary,
                         reasons=reasons, form=form, form_text=form_text, metas=metas))
    return {"gate": "current-plus-hard/1", "batch": batch["id"], "allowed": allowed,
            "budget": {"applicable": True, "chars": payload_chars, "max_chars": max_chars,
                       "items": len(cases), "max_items": max_items, "exceeded": budget_exceeded},
            "counts": _counts(rows), "rows": rows}


def apply_current_gate(batch: dict, *, max_chars: int = DEFAULT_MAX_CHARS,
                       max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    """现行 raw 闸（直接调用生产 `dream._apply_gates`），只做理由归因，不改行为。

    该臂没有 schema/scope/budget 约束；`budget.applicable=False` 表示“本臂不执行预算”，
    不是“预算未超”。硬约束收益只能归 current_plus_hard/three_state 的共同硬约束。
    """
    conversations = batch["conversations"]
    allowed = batch["allowed"]
    cases = batch["cases"]
    payload_chars = _payload_chars(cases, conversations)
    candidates, id_by_object = [], {}
    for entry in cases:
        candidate = candidate_of(entry["item"])
        id_by_object[id(candidate)] = entry["item"]["id"]
        candidates.append(candidate)
    kept, dropped = dream._apply_gates(candidates, conversations)
                                                     
    kept_ids = {id_by_object[id(item)] for item in kept}
    pools = dream._conversation_evidence_pools(conversations)
    rows = []
    for entry in cases:
        item = entry["item"]
        candidate = candidate_of(item)
        hard = _hard_state(item, candidate, conversations, pools, allowed)
        form, form_text = injection_form(candidate)
        metas = meta_matches(candidate)
        was_kept = item["id"] in kept_ids
        if was_kept:
            decision, reason = "keep", "none"
        elif metas:
            decision, reason = "reject", "injection"
        else:
            decision, reason = "reject", "evidence"
        rows.append(_row(item, batch, candidate, hard, decision=decision, primary=reason,
                         reasons=[] if was_kept else [reason], form=form, form_text=form_text,
                         metas=metas))
    return {"gate": "current-apply-gates", "batch": batch["id"], "allowed": allowed,
            "budget": {"applicable": False, "chars": payload_chars, "max_chars": max_chars,
                       "items": len(cases), "max_items": max_items, "exceeded": None},
            "counts": _counts(rows), "dropped": dropped, "rows": rows}


def run_dataset(dataset: dict, gate: str, *, max_chars: int = DEFAULT_MAX_CHARS,
                max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    """逐批跑一台闸；预算压力批用更小预算。三臂函数签名一致。"""
    arms = {"current": apply_current_gate, "current_plus_hard": apply_current_plus_hard,
            "three_state": apply_three_state}
    assert gate in arms, gate
    batches = []
    for batch in dataset["batches"]:
        limit = 200 if batch.get("stress") == "budget" else max_chars
        batches.append(arms[gate](batch, max_chars=limit, max_items=max_items))
    return {"gate": gate, "batches": batches}


__all__ = [
    "ARM_NAMES", "GATE_VERSION", "PRODUCTION_SEAMS", "REJECT_FORMS", "apply_current_gate",
    "apply_current_plus_hard", "apply_three_state", "candidate_of", "injection_form",
    "meta_matches", "production_body_block", "run_dataset", "scope_violations", "validate_schema",
]
