"""训练与推理共用的纯语义叶子契约；不加载模型、不判断身份或操作权限。

old 固定为前提 P，new 固定为假设 H。五叶描述同一解释域内非空真值集合：
等价、P 严格蕴含 H、H 严格蕴含 P、不相交、相交且互不包含。公开 NLI
投影以这一解释域假设为前提，只能聚合概率，不能从三类 gold 反推五叶。

pair 只接受 old/new 和可选 frame；条目 kind 为 proposition/memory/skill/example，
text 必填，title/trigger/permanence/payload 可选。条件与时限原样作为语义事实，
不推断有效性；permanence 若提供只接受 persistent/temporary，不猜默认值。
frame.mode 只接受 same_domain；省略 frame 或 mode 是调用者采用同域解释的声明，
不是本模块验证了现实中的身份、作用域或权限相同。background 为共享背景文本。
payload 的语义字段为 tools/utterance/steps；步骤只接受 verb/args，args 可为
有界 JSON。映射键顺序规范化，所有序列保序；仅明确结构层的 metadata 剥离。
args 保留全部功能键和值，不按名字猜身份或授权；凭据键值脱敏，键碰撞报错。
无旧项须由外层 coverage 处理，本模块不构造空标记或语义关系。

错误为带 code/hint 的 SemanticRelationError；调用端自行翻译为 HTTP/MCP 错误
或弃权，不应把输入错误当作某个语义叶子。仅依赖标准库及 secret_patterns 真源。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ..secret_patterns import redact

__all__ = [
    "SEMANTIC_RELATIONS", "NLI_CLASSES", "SEMANTIC_TO_NLI", "SWAPPED_RELATIONS",
    "SemanticRelationError", "validate_probabilities", "project_nli_probabilities",
    "swap_relation", "swap_probabilities", "validate_semantic_pair", "swap_pair",
    "semantic_pair_text", "SEMANTIC_INPUT_RECIPE", "semantic_pair_pieces",
    "render_semantic_fragments", "prepare_semantic_input", "SEMANTIC_LIMITS_PRODUCT", "SEMANTIC_LIMITS_RESEARCH",
]

SEMANTIC_RELATIONS = (
    "equivalent",
    "old_entails_new_strict",
    "new_entails_old_strict",
    "disjoint",
    "overlap_incomparable",
)
NLI_CLASSES = ("entailment", "neutral", "contradiction")
SEMANTIC_TO_NLI: Mapping[str, str] = MappingProxyType({
    "equivalent": "entailment",
    "old_entails_new_strict": "entailment",
    "new_entails_old_strict": "neutral",
    "disjoint": "contradiction",
    "overlap_incomparable": "neutral",
})
SWAPPED_RELATIONS: Mapping[str, str] = MappingProxyType({
    "equivalent": "equivalent",
    "old_entails_new_strict": "new_entails_old_strict",
    "new_entails_old_strict": "old_entails_new_strict",
    "disjoint": "disjoint",
    "overlap_incomparable": "overlap_incomparable",
})

PROBABILITY_ATOL = 1e-6
MAX_TEXT_CHARS = 8192
MAX_PAIR_CHARS = 65536
SEMANTIC_LIMITS_PRODUCT = MappingProxyType({"max_text_chars": 8192, "max_pair_chars": 65536})
SEMANTIC_LIMITS_RESEARCH = MappingProxyType({"max_text_chars": 65536, "max_pair_chars": 262144})
MAX_PAYLOAD_ITEMS = 64
MAX_PAYLOAD_DEPTH = 6
_MAX_JSON_NODES = 512
_KINDS = frozenset({"proposition", "memory", "skill", "example"})
_ENTRY_FIELDS = ("kind", "text", "title", "trigger", "permanence", "payload")
_PAYLOAD_FIELDS = ("tools", "utterance", "steps")
_NONSEMANTIC_FIELDS = frozenset({
    "id", "uid", "owner", "owner_id", "principal", "principal_id", "account_id",
    "actor", "actor_id", "endpoint", "endpoint_id", "endpoint_fp", "project",
    "project_id", "permission", "permissions", "authority", "scope", "scope_level",
    "tier", "label", "gold", "label_origin", "relation", "coarse_label",
    "fine_label", "nli_label", "semantic_label", "verdict", "verified_success",
    "provenance", "evidence", "evidence_refs", "links", "status", "revision",
    "created_at", "updated_at",
})


class SemanticRelationError(ValueError):
    def __init__(self, code: str, hint: str) -> None:
        self.code = code
        self.hint = hint
        super().__init__(hint)


def validate_probabilities(
    probabilities: Sequence[float], *, size: int = 5,
) -> tuple[float, ...]:
    """校验 5 维五叶或 3 维 NLI simplex；容差仅用于总和，不裁剪或归一化。"""
    if (type(size) is not int or size not in (len(SEMANTIC_RELATIONS), len(NLI_CLASSES))
            or not isinstance(probabilities, Sequence)
            or isinstance(probabilities, (str, bytes, bytearray))
            or len(probabilities) != size):
        raise SemanticRelationError("semantic_invalid_probabilities", "概率须为约定维数的数值序列。")
    values = []
    for value in probabilities:
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise SemanticRelationError("semantic_invalid_probabilities", "概率须为 [0, 1] 内的有限数值。")
        values.append(float(value))
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_ATOL):
        raise SemanticRelationError("semantic_invalid_probabilities", "概率总和须在绝对容差内等于 1。")
    return tuple(values)


def project_nli_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    """按 SEMANTIC_RELATIONS 输入，按 NLI_CLASSES 输出父类概率和。"""
    values = validate_probabilities(probabilities)
    projected = tuple(math.fsum(value for name, value in zip(SEMANTIC_RELATIONS, values)
                               if SEMANTIC_TO_NLI[name] == parent) for parent in NLI_CLASSES)
    return validate_probabilities(projected, size=len(NLI_CLASSES))


def swap_relation(relation: str) -> str:
    if type(relation) is not str or relation not in SWAPPED_RELATIONS:
        raise SemanticRelationError("semantic_invalid_relation", "关系须为已定义的五叶名称。")
    return SWAPPED_RELATIONS[relation]


def swap_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    """交换 P/H 的五叶分布；三类分布无法唯一恢复反向，故不接受三类输入。"""
    values = dict(zip(SEMANTIC_RELATIONS, validate_probabilities(probabilities)))
    return tuple(values[SWAPPED_RELATIONS[name]] for name in SEMANTIC_RELATIONS)


def _shape(value: object, fields: Sequence[str], code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or len(value) > MAX_PAYLOAD_ITEMS:
        raise SemanticRelationError(code, "语义输入须为有界字段映射。")
    if any(type(key) is not str or (key not in fields and key.casefold() not in _NONSEMANTIC_FIELDS)
           for key in value):
        raise SemanticRelationError(code, "语义输入含未知字段。")
    return value


def _text(value: object, *, empty: bool = False, maximum: int = MAX_TEXT_CHARS) -> str:
    if type(value) is not str or len(value) > maximum or (not empty and not value.strip()):
        raise SemanticRelationError("semantic_invalid_text", "语义文本类型、非空性或长度不合规。")
    if any((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127
           or 0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise SemanticRelationError("semantic_invalid_text", "语义文本含非法控制字符或编码。")
    clean = redact(value.strip())
    if len(clean) > maximum:
        raise SemanticRelationError("semantic_invalid_text", "脱敏后语义文本超出长度界限。")
    return clean


def _sequence(value: object, *, nonempty: bool = False) -> Sequence[object]:
    if (type(value) not in (list, tuple) or len(value) > MAX_PAYLOAD_ITEMS
            or (nonempty and not value)):
        raise SemanticRelationError("semantic_invalid_payload", "语义序列类型或长度不合规。")
    return value


def _json_semantics(value: object, budget: list[int], depth: int = 0, *, limits) -> object:
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_PAYLOAD_DEPTH:
        raise SemanticRelationError("semantic_invalid_payload", "语义参数嵌套或节点数超出界限。")
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        _text(value, empty=True, maximum=limits["max_text_chars"])
        clean = redact(value)
        if len(clean) > limits["max_text_chars"]:
            raise SemanticRelationError("semantic_invalid_text", "脱敏后语义文本超出长度界限。")
        return clean
    if type(value) is int and -(2 ** 63) <= value < 2 ** 63:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json_semantics(item, budget, depth + 1, limits=limits) for item in _sequence(value)]
    if isinstance(value, Mapping) and len(value) <= MAX_PAYLOAD_ITEMS:
        keys = {}
        for key in value:
            _text(key, maximum=128)
            clean = redact(key)
            if len(clean) > 128 or clean in keys:
                raise SemanticRelationError("semantic_invalid_payload", "脱敏后的参数键超出界限或发生冲突。")
            keys[clean] = key
        return {clean: _json_semantics(value[key], budget, depth + 1, limits=limits)
                for clean, key in sorted(keys.items())}
    raise SemanticRelationError("semantic_invalid_payload", "语义参数须为有限、有界的 JSON 值。")


def _payload(value: object, *, limits) -> dict[str, object]:
    row = _shape(value, _PAYLOAD_FIELDS, "semantic_invalid_payload")
    result = {}
    if "tools" in row:
        result["tools"] = [_text(item, maximum=128) for item in _sequence(row["tools"])]
    if "utterance" in row:
        result["utterance"] = _text(row["utterance"], maximum=limits["max_text_chars"])
    if "steps" in row:
        steps = []
        budget = [_MAX_JSON_NODES]
        for value in _sequence(row["steps"], nonempty=True):
            step = _shape(value, ("verb", "args"), "semantic_invalid_payload")
            steps.append({"verb": _text(step.get("verb"), maximum=128),
                          "args": _json_semantics(step.get("args", ""), budget, limits=limits)})
        result["steps"] = steps
    return result


def _entry(value: object, *, limits) -> dict[str, object]:
    row = _shape(value, _ENTRY_FIELDS, "semantic_invalid_entry")
    kind = row.get("kind")
    if type(kind) is not str or kind not in _KINDS:
        raise SemanticRelationError("semantic_invalid_entry", "条目须声明支持的语义类型。")
    result = {"kind": kind, "text": _text(row.get("text"), maximum=limits["max_text_chars"])}
    for key in ("title", "trigger"):
        if key in row:
            result[key] = _text(row[key], empty=True, maximum=limits["max_text_chars"])
    if "permanence" in row:
        permanence = row["permanence"]
        if type(permanence) is not str or permanence not in ("persistent", "temporary"):
            raise SemanticRelationError("semantic_invalid_entry", "已提供的持久性须明确；未知值不可推断。")
        result["permanence"] = permanence
    if "payload" in row:
        result["payload"] = _payload(row["payload"], limits=limits)
    return result


def _dump(pair: Mapping[str, object], *, limits) -> str:
    text = json.dumps(pair, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text) > limits["max_pair_chars"]:
        raise SemanticRelationError("semantic_input_too_large", "完整语义输入超出序列化长度界限。")
    return text


def validate_semantic_pair(pair: Mapping[str, object], *, limits=None) -> dict[str, object]:
    """返回全新、脱敏且仅含语义字段的同域 pair；limits由调用端提供，不取自pair。"""
    limits = SEMANTIC_LIMITS_PRODUCT if limits is None else limits
    row = _shape(pair, ("old", "new", "frame"), "semantic_invalid_pair")
    frame = _shape(row.get("frame", {}), ("mode", "background"), "semantic_invalid_frame")
    mode = frame.get("mode", "same_domain")
    if type(mode) is not str or mode != "same_domain":
        raise SemanticRelationError("semantic_invalid_frame", "仅可比较显式采用同一解释域的语义。")
    result = {
        "frame": {"mode": mode, "background": _text(frame.get("background", ""), empty=True, maximum=limits["max_text_chars"])},
        "old": _entry(row.get("old"), limits=limits),
        "new": _entry(row.get("new"), limits=limits),
    }
    _dump(result, limits=limits)
    return result


def swap_pair(pair: Mapping[str, object], *, limits=None) -> dict[str, object]:
    result = validate_semantic_pair(pair, limits=limits)
    result["old"], result["new"] = result["new"], result["old"]
    return result


def semantic_pair_text(pair: Mapping[str, object], *, limits=None) -> str:
    """唯一事实序列器：紧凑 JSON，frame → old(P) → new(H)；可见片段不冒充此合同。"""
    limits = SEMANTIC_LIMITS_PRODUCT if limits is None else limits
    return _dump(validate_semantic_pair(pair, limits=limits), limits=limits)


SEMANTIC_INPUT_RECIPE = "semantic-role-balanced-fragments-v2"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_pair_pieces(pair: Mapping[str, object], *, limits=None) -> dict:
    """从唯一事实文本解析通用 JSON 值区间；不再实现一份字段投影或业务序列器。"""
    text = semantic_pair_text(pair, limits=limits)
    decoder, fields = json.JSONDecoder(), []

    def visit(start, path):
        value, end = decoder.raw_decode(text, start)
        field = {"path": list(path), "range": [start, end], "string": isinstance(value, str),
                 "container": "object" if isinstance(value, dict) else "array" if isinstance(value, list) else None,
                 "leaf": not isinstance(value, (dict, list)) or not value}
        fields.append(field)
        if isinstance(value, dict) and value:
            cursor = start + 1
            for _ in value:
                key, cursor = decoder.raw_decode(text, cursor)
                cursor = visit(cursor + 1, path + (key,))
                if text[cursor:cursor + 1] == ",":
                    cursor += 1
        elif isinstance(value, list) and value:
            cursor = start + 1
            for index in range(len(value)):
                cursor = visit(cursor, path + (index,))
                if text[cursor:cursor + 1] == ",":
                    cursor += 1
        return end

    visit(0, ())
    return {"raw_text": text, "raw_pair_sha256": _sha_text(text), "fields": fields}


def _pointer(path) -> str:
    return "/" + "/".join(("i:" if type(part) is int else "k:") + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _piece_complete(field, count):
    return count > 0 if field["container"] and not field["leaf"] else count == field["range"][1] - field["range"][0]


def render_semantic_fragments(source: Mapping, prefixes: Mapping, *, reverse: bool = False) -> dict:
    """通用typed路径及原值/容器起始符进入模型文本；结构不保留也必须记丢失。"""
    raw, pieces = source["raw_text"], [field for field in source["fields"] if field["path"]]
    header = "[visible-fragments]\n@" + raw[0] + "\n"
    parts, rendered, cursor = [header], {(): [len("[visible-fragments]\n@"), len("[visible-fragments]\n@") + 1]}, len(header)
    role_order = ("frame", "new", "old") if reverse else ("frame", "old", "new")
    sections, information = {}, {}
    for source_role, role in zip(role_order, ("frame", "old", "new")):
        section = [f"[{role}]\n"]
        cursor += len(section[0])
        facts = []
        for field in pieces:
            path, (start, end) = tuple(field["path"]), field["range"]
            if path[0] != source_role:
                continue
            count = prefixes.get(path, 0)
            if type(count) is not int or not 0 <= count <= end - start:
                raise SemanticRelationError("semantic_invalid_fragment", "可见区间须位于原始值内。")
            if count == 0:
                continue
            prefix = raw[start:start + count]
            marker = "@" if field["container"] else "=" if count == end - start else "~"
            head = json.dumps(_pointer(path[1:]), ensure_ascii=False) + marker
            line = head + prefix + "\n"
            rendered[path] = [cursor + len(head), cursor + len(head) + count]
            section.append(line)
            cursor += len(line)
            facts.append([list(path[1:]), field["container"], prefix, _piece_complete(field, count)])
        sections[source_role] = "".join(section)
        information[source_role] = facts
        parts.extend(section)
    descendants_by_path = {tuple(field["path"]): [] for field in source["fields"]}
    for piece in pieces:
        path = tuple(piece["path"])
        for depth in range(len(path) + 1):
            descendants_by_path[path[:depth]].append(piece)
    audit = []
    for field in source["fields"]:
        path = tuple(field["path"])
        descendants = descendants_by_path[path]
        visible = [[piece["range"][0], piece["range"][0] + prefixes.get(tuple(piece["path"]), 0)]
                   for piece in descendants if prefixes.get(tuple(piece["path"]), 0)]
        if not path:
            visible.insert(0, [0, 1])
        complete = all(_piece_complete(piece, prefixes.get(tuple(piece["path"]), 0)) for piece in descendants)
        audit.append({"path": list(path), "container": field["container"], "original_range": field["range"],
                      "visible_source_ranges": visible, "rendered_range": rendered.get(path),
                      "complete": complete, "lost": not complete, "omitted": not visible})
    info = [information.get("frame", []), sorted([information.get("old", []), information.get("new", [])],
                                               key=lambda facts: json.dumps(facts, ensure_ascii=False))]
    text = "".join(parts)
    return {"text": text, "view_sha256": _sha_text(text), "sections": sections, "fields": audit,
            "visible_information_sha256": _sha_text(json.dumps(info, ensure_ascii=False, separators=(",", ":"))),
            "complete": all(item["complete"] for item in audit)}


def _tokenized(tokenizer, text: str, *, offsets=False) -> dict:
    try:
        result = tokenizer(text, add_special_tokens=True, padding=False, truncation=False,
                           return_attention_mask=True, return_offsets_mapping=offsets)
    except (TypeError, NotImplementedError, ValueError) as error:
        raise SemanticRelationError("semantic_tokenizer_offsets_unavailable", "需要真实 token/字符边界，不能估算截断。") from error
    if not isinstance(result.get("input_ids"), (list, tuple)):
        raise SemanticRelationError("semantic_invalid_encoding", "接缝只接受单文本的实际 token ID 序列。")
    return dict(result)


def _fragment_ends(tokenizer, literal: str, string: bool) -> list[int]:
    if not string or literal == '""':
        return [0, len(literal)]
    safe, cursor = {len(literal)}, 1
    while cursor < len(literal) - 1:
        if literal[cursor] == "\\":
            cursor += 6 if literal[cursor + 1] == "u" else 2
        else:
            cursor += 1
        safe.add(cursor)
    encoded = _tokenized(tokenizer, literal, offsets=True)
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, (list, tuple)):
        raise SemanticRelationError("semantic_tokenizer_offsets_unavailable", "tokenizer 未提供实际 offset_mapping。")
    ends, safe = {0, len(literal)}, sorted(safe)
    for start, end in offsets:
        if end > start:
            low, high = 0, len(safe)
            while low < high:
                middle = (low + high) // 2
                if safe[middle] <= end:
                    low = middle + 1
                else:
                    high = middle
            if low:
                ends.add(safe[low - 1])
    return sorted(ends)


def prepare_semantic_input(pair: Mapping, tokenizer, *, max_length: int,
                           role_independent_frame: bool = False, limits=None) -> dict:
    """共享 tokenizer 驱动接缝；frame 角色无关是调用者声明，不猜指称、不改 raw/gold。"""
    if type(max_length) is not int or max_length < 1 or type(role_independent_frame) is not bool:
        raise SemanticRelationError("semantic_invalid_budget", "输入预算和 frame 声明无效。")
    limits = SEMANTIC_LIMITS_PRODUCT if limits is None else limits
    source = semantic_pair_pieces(pair, limits=limits)
    raw, fields = source["raw_text"], source["fields"]
    background = next(field for field in fields if field["path"] == ["frame", "background"])
    if raw[slice(*background["range"])] != '""' and not role_independent_frame:
        raise SemanticRelationError("semantic_role_dependent_frame", "交换视图仅支持调用者明确声明的角色无关共享背景。")
    pieces = [field for field in fields if field["path"]]
    options = {tuple(field["path"]): [0, 1] if field["container"] and not field["leaf"] else
               _fragment_ends(tokenizer, raw[slice(*field["range"])], field["string"]) for field in pieces}
    groups = {role: [tuple(field["path"]) for field in pieces if field["path"][0] == role]
              for role in ("frame", "old", "new")}
    base = {path: 0 for path in options}
    base[("frame", "mode")] = len(options[("frame", "mode")]) - 1
    if raw[slice(*background["range"])] != '""':
        base[("frame", "background")] = 1
    for role in ("old", "new"):
        base[(role, "text")] = 1

    def prefixes(state):
        return {path: options[path][index] for path, index in state.items()}

    cache = {}
    def count(text):
        if text not in cache:
            cache[text] = len(_tokenized(tokenizer, text)["input_ids"])
        return cache[text]

    def views(state):
        values = prefixes(state)
        return [render_semantic_fragments(source, values, reverse=reverse) for reverse in (False, True)]

    def total(state):
        return max(count(view["text"]) for view in views(state))

    def state_at(role, units, current):
        keys = groups[role]
        caps = [len(options[path]) - 1 - base[path] for path in keys]
        low, high = 0, max(caps, default=0)
        while low < high:
            level = (low + high + 1) // 2
            if sum(min(cap, level) for cap in caps) <= units:
                low = level
            else:
                high = level - 1
        filled = [min(cap, low) for cap in caps]
        rest = units - sum(filled)
        result = dict(current)
        for path, cap, amount in zip(keys, caps, filled):
            extra = int(rest > 0 and amount < cap)
            rest -= extra
            result[path] = base[path] + amount + extra
        return result

    current = dict(base)
    required = total(current)
    if required > max_length:
        raise SemanticRelationError("semantic_budget_too_small", "预算不足以编码共享框架及 old/new 的实际正文，不能只保留 key 头。")
    role_rank = sorted(("old", "new"), key=lambda role: raw[slice(*next(field["range"] for field in fields if field["path"] == [role]))])
    order = ["frame", *role_rank]
    room = max_length - required
    capacities = {role: sum(len(options[path]) - 1 - base[path] for path in groups[role]) for role in order}
    allocated = {role: 0 for role in order}
    for role in order:
        quota = room // len(order)
        low, high = 0, capacities[role]
        while low < high:
            middle = (low + high + 1) // 2
            proposal = state_at(role, middle, current)
            if total(proposal) <= min(max_length, total(current) + quota):
                low = middle
            else:
                high = middle - 1
        current, allocated[role] = state_at(role, low, current), low
    for _ in range(2):
        for role in sorted(order, key=lambda item: (allocated[item], order.index(item))):
            low, high = allocated[role], capacities[role]
            while low < high:
                middle = (low + high + 1) // 2
                if total(state_at(role, middle, current)) <= max_length:
                    low = middle
                else:
                    high = middle - 1
            current, allocated[role] = state_at(role, low, current), low
    prepared = views(current)
    for view in prepared:
        encoded = _tokenized(tokenizer, view["text"], offsets=True)
        offsets = encoded.pop("offset_mapping", None)
        ids, attention = encoded["input_ids"], encoded.get("attention_mask")
        if len(ids) > max_length or offsets is None or not isinstance(attention, (list, tuple)) \
                or len(attention) != len(ids) or len(offsets) != len(ids) \
                or any(type(value) is not int or value not in (0, 1) for value in attention):
            raise SemanticRelationError("semantic_invalid_encoding", "实际编码预算、attention或字符边界不合规。")
        special = set(getattr(tokenizer, "all_special_ids", ()))
        pad_id = getattr(tokenizer, "pad_token_id", None)
        for role in ("old", "new"):
            origin = "new" if view is prepared[1] and role == "old" else "old" if view is prepared[1] else role
            field = next(item for item in view["fields"] if item["path"] == [origin, "text"])
            start, end = field["rendered_range"]
            body_end = end - int(prefixes(current)[(origin, "text")] == options[(origin, "text")][-1])
            if not any(mask == 1 and token != pad_id and token not in special
                       and left < body_end and right > start + 1 for token, mask, (left, right) in zip(ids, attention, offsets)):
                raise SemanticRelationError("semantic_body_not_visible", "正文须有实际attention且非padding/special token；offset头不算覆盖。")
        view.update({"encoding": encoded, "token_length": len(encoded["input_ids"]),
                     "special_tokens": int(tokenizer.num_special_tokens_to_add(pair=False)),
                     "role_body_covered": {"old": True, "new": True},
                     "attention_mask_sum": sum(encoded["attention_mask"])})
        view.pop("sections")
    if prepared[0]["visible_information_sha256"] != prepared[1]["visible_information_sha256"]:
        raise SemanticRelationError("semantic_information_mismatch", "交换只能置换同一份可见片段。")
    complete = prepared[0]["complete"]
    full = views({path: len(values) - 1 for path, values in options.items()})
    return {"full_visible_token_lengths": [count(view["text"]) for view in full],
            "recipe": SEMANTIC_INPUT_RECIPE, "input_limits": dict(limits), "raw_text": raw, "raw_pair_sha256": source["raw_pair_sha256"],
            "visible_information_sha256": prepared[0]["visible_information_sha256"], "max_length": max_length,
            "complete": complete, "requires_llm_fallback": not complete,
            "frame_condition": "caller-declared role-independent shared frame; no language verification",
            "directions": dict(zip(("forward", "reverse"), prepared))}
