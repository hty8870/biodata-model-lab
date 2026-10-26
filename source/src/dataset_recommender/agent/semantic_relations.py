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


SEMANTIC_INPUT_RECIPE = "semantic-json-tree-witness-v3"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _string_boundaries(literal):
    positions, cursor = [1], 1
    while cursor < len(literal) - 1:
        cursor += (6 if literal[cursor + 1:cursor + 2] == "u" else 2) if literal[cursor] == "\\" else 1
        positions.append(cursor)
    return positions


def semantic_pair_pieces(pair: Mapping[str, object], *, limits=None) -> dict:
    """沿唯一canonical文本的JSONDecoder索引树、key及安全转义边界，不另造parser。"""
    text = semantic_pair_text(pair, limits=limits)
    decoder, fields = json.JSONDecoder(), []

    def visit(start, path, key_range=None):
        value, end = decoder.raw_decode(text, start)
        kind = ("object" if isinstance(value, dict) else "array" if isinstance(value, list) else
                "string" if isinstance(value, str) else "null" if value is None else
                "boolean" if type(value) is bool else "integer" if type(value) is int else "number")
        field = {"path": list(path), "range": [start, end], "string": kind == "string", "value_type": kind,
                 "container": kind if kind in ("object", "array") else None,
                 "leaf": kind not in ("object", "array") or not value, "children": [], "key_range": key_range,
                 "string_boundaries": _string_boundaries(text[start:end]) if kind == "string" else []}
        fields.append(field)
        if isinstance(value, dict) and value:
            cursor = start + 1
            for _ in value:
                key_start = cursor
                key, cursor = decoder.raw_decode(text, cursor)
                child = path + (key,)
                field["children"].append(list(child))
                cursor = visit(cursor + 1, child, [key_start, cursor])
                if text[cursor:cursor + 1] == ",":
                    cursor += 1
        elif isinstance(value, list) and value:
            cursor = start + 1
            for index in range(len(value)):
                child = path + (index,)
                field["children"].append(list(child))
                cursor = visit(cursor, child)
                if text[cursor:cursor + 1] == ",":
                    cursor += 1
        return end

    visit(0, ())
    return {"raw_text": text, "raw_pair_sha256": _sha_text(text), "fields": fields}


def _merged_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        elif start < end:
            result.append([start, end])
    return result


def _missing_ranges(start, end, visible):
    result, cursor = [], start
    for left, right in _merged_ranges(visible):
        if cursor < left:
            result.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


def render_semantic_fragments(source: Mapping, selection: Mapping, *, reverse: bool = False, _audit=True) -> dict:
    """唯一JSON树renderer；nodes[path]=full/hole/partial或相对literal的安全string区间。

    STR/INT/NUM/BOOL/NULL/OBJ/ARR为引号外typed holes，MEMBERSn/ITEMSn为原位置省略计数；
    多段string为独立quoted片段与STR拼接。全选仍只复制canonical，不添加表示头。
    """
    raw = source["raw_text"]
    fields = {tuple(item["path"]): item for item in source["fields"]}
    if not isinstance(selection, Mapping) or not isinstance(selection.get("nodes"), Mapping):
        raise SemanticRelationError("semantic_invalid_selection", "v3选择状态须含nodes；整数prefix合同已退役。")
    nodes, reasons = dict(selection["nodes"]), selection.get("reasons", {})
    if nodes.get(()) not in ("full", "partial"):
        raise SemanticRelationError("semantic_invalid_selection", "根须为full或partial的JSON树。")
    for path, choice in list(nodes.items()):
        if path not in fields:
            raise SemanticRelationError("semantic_invalid_selection", "选择路径不在canonical树内。")
        field = fields[path]
        if isinstance(choice, str) and choice in ("full", "hole", "partial"):
            if choice == "partial" and (not field["container"] or field["leaf"]):
                raise SemanticRelationError("semantic_invalid_selection", "只有非空容器可用partial模式。")
        elif field["string"] and isinstance(choice, (list, tuple)):
            safe = set(field["string_boundaries"])
            if any(not isinstance(span, (list, tuple)) or len(span) != 2 or
                   type(span[0]) is not int or type(span[1]) is not int or
                   span[0] not in safe or span[1] not in safe or span[0] >= span[1] for span in choice):
                raise SemanticRelationError("semantic_invalid_selection", "string片段须使用完整转义/Unicode边界。")
            nodes[path] = _merged_ranges(choice)
        else:
            raise SemanticRelationError("semantic_invalid_selection", "原子非string值只可full或typed hole。")
    full_cache = {}
    def full(path):
        if path not in full_cache:
            value, field = nodes.get(path, "hole"), fields[path]
            if value == "full":
                result = True
            elif field["string"] and isinstance(value, list):
                result = value == [[1, field["range"][1] - field["range"][0] - 1]]
            else:
                result = value == "partial" and bool(field["children"]) and all(full(tuple(child)) for child in field["children"])
            full_cache[path] = result
        return full_cache[path]
    parts, copies, represented, cursor, sections = [], [], set(), 0, {}
    def emit(text):
        nonlocal cursor
        parts.append(text)
        cursor += len(text)
    def copy_range(start, end):
        if _audit and start < end:
            copies.append((start, end, cursor, cursor + end - start))
        emit(raw[start:end])
    holes = {"string": "STR", "integer": "INT", "number": "NUM", "boolean": "BOOL", "null": "NULL", "object": "OBJ", "array": "ARR"}
    def render(path):
        field, choice = fields[path], nodes.get(path, "hole")
        start, end = field["range"]
        represented.add(path)
        if full(path) and not (path == () and reverse):
            copy_range(start, end)
            represented.update(p for p in fields if p[:len(path)] == path)
            return
        if field["string"] and isinstance(choice, list) and choice:
            last = 1
            for left, right in choice:
                if left > last:
                    emit("STR")
                copy_range(start, start + 1)
                copy_range(start + left, start + right)
                copy_range(end - 1, end)
                last = right
            if last < end - start - 1:
                emit("STR")
            return
        if choice != "partial" and not (path == () and reverse and full(path)):
            emit(holes[field["value_type"]])
            return
        copy_range(start, start + 1)
        children = [tuple(child) for child in field["children"]]
        if path == ():
            destinations = children
            origins = [("frame",), ("new",), ("old",)] if reverse else children
        else:
            destinations = children
            origins = children
        entries, omitted = [], 0
        for destination, origin in zip(destinations, origins):
            included = path == () or full(path) or origin in nodes
            if not included:
                omitted += 1
                continue
            if omitted:
                entries.append((None, omitted))
                omitted = 0
            entries.append(((destination, origin), 0))
        if omitted:
            entries.append((None, omitted))
        for index, (member, missing) in enumerate(entries):
            if index:
                if full(path):
                    separator = fields[children[index - 1]]["range"][1]
                    copy_range(separator, separator + 1)
                else:
                    emit(",")
            if member is None:
                emit(("MEMBERS" if field["container"] == "object" else "ITEMS") + str(missing))
                continue
            destination, origin = member
            child = fields[destination]
            if field["container"] == "object":
                copy_range(*child["key_range"])
                copy_range(child["key_range"][1], child["range"][0])
            before = len(parts)
            if full(path):
                previous = nodes.get(origin)
                nodes[origin] = "full"
                full_cache[origin] = True
                render(origin)
                if previous is None:
                    nodes.pop(origin)
                else:
                    nodes[origin] = previous
            else:
                render(origin)
            if path == ():
                sections[origin[0]] = "".join(parts[before:])
        copy_range(end - 1, end)
    if full(()):
        sections = {role: raw[slice(*fields[(role,)]["range"])] for role in ("frame", "old", "new")}
    render(())
    text = "".join(parts)
    information = [sections["frame"], sorted([sections["old"], sections["new"]])]
    result = {"text": text, "view_sha256": _sha_text(text), "sections": sections,
              "visible_information_sha256": _sha_text(json.dumps(information, ensure_ascii=False, separators=(",", ":"))),
              "complete": full(())}
    if not _audit:
        return result
    def segments(start, end):
        result = []
        for a, b, c, d in copies:
            left, right = max(a, start), min(b, end)
            if left < right:
                item = {"source_range": [left, right], "rendered_range": [c + left - a, c + right - a]}
                if result and result[-1]["source_range"][1] == left and result[-1]["rendered_range"][1] == item["rendered_range"][0]:
                    result[-1]["source_range"][1] = right
                    result[-1]["rendered_range"][1] = item["rendered_range"][1]
                else:
                    result.append(item)
        return result
    audits = []
    for path, field in fields.items():
        start, end = field["range"]
        spans = segments(start, end)
        visible = _merged_ranges([item["source_range"] for item in spans])
        omitted = _missing_ranges(start, end, visible)
        reason = next((reasons[path[:depth]] for depth in range(len(path), -1, -1) if path[:depth] in reasons), "unspecified")
        child_paths = [tuple(child) for child in field["children"]]
        structure = path in represented and (not field["container"] or not omitted or
                    (nodes.get(path) == "partial" and all(child in represented for child in child_paths)))
        audits.append({"path": list(path), "container": field["container"], "value_type": field["value_type"],
                       "original_range": [start, end], "visible_source_ranges": visible, "rendered_segments": spans,
                       "rendered_range": spans[0]["rendered_range"] if len(spans) == 1 else None,
                       "key_range": field["key_range"], "key_rendered_segments": segments(*field["key_range"]) if field["key_range"] else [],
                       "omitted_ranges": omitted, "structural_complete": structure, "selection_reason": reason,
                       "complete": not omitted, "lost": bool(omitted), "omitted": not visible})
    result["fields"] = audits
    return result


def _tokenized(tokenizer, text: str, *, offsets=False) -> dict:
    try:
        result = tokenizer(text, add_special_tokens=True, padding=False, truncation=False,
                           return_attention_mask=True, return_offsets_mapping=offsets)
    except (TypeError, NotImplementedError, ValueError) as error:
        raise SemanticRelationError("semantic_tokenizer_offsets_unavailable", "需要真实 token/字符边界，不能估算截断。") from error
    if not isinstance(result.get("input_ids"), (list, tuple)):
        raise SemanticRelationError("semantic_invalid_encoding", "接缝只接受单文本的实际 token ID 序列。")
    return dict(result)


def _checked_encoding(tokenizer, text):
    encoded = _tokenized(tokenizer, text, offsets=True)
    ids, attention, offsets = encoded["input_ids"], encoded.get("attention_mask"), encoded.get("offset_mapping")
    if not isinstance(attention, (list, tuple)) or not isinstance(offsets, (list, tuple)) \
            or len(attention) != len(ids) or len(offsets) != len(ids) \
            or any(type(value) is not int for value in ids) \
            or any(type(value) is not int or value not in (0, 1) for value in attention) \
            or any(not isinstance(span, (list, tuple)) or len(span) != 2 or
                   any(type(value) is not int for value in span) or not 0 <= span[0] <= span[1] <= len(text) for span in offsets):
        raise SemanticRelationError("semantic_invalid_encoding", "实际编码、attention或字符边界不合规。")
    return encoded


def _content_tokens(view, raw, encoded, tokenizer):
    special = set(getattr(tokenizer, "all_special_ids", ()))
    pad = getattr(tokenizer, "pad_token_id", None)
    for field in view["fields"]:
        field["content_token_indices"] = []
        if field["container"]:
            continue
        start, end = field["original_range"]
        if field["value_type"] == "string":
            start, end = start + 1, end - 1
        ranges = []
        for segment in field["rendered_segments"]:
            a, b = segment["source_range"]
            left, right = max(a, start), min(b, end)
            if left < right:
                offset = segment["rendered_range"][0] - a
                ranges.append((left + offset, right + offset))
        for index, (token, mask, (left, right)) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"])):
            if mask == 1 and token != pad and token not in special and left < right \
                    and any(max(left, a) < min(right, b) for a, b in ranges):
                field["content_token_indices"].append(index)


def prepare_semantic_input(pair: Mapping, tokenizer, *, max_length: int,
                           role_independent_frame: bool = False, limits=None) -> dict:
    """同一树renderer、交换不变选择与两向实际单文本token预算。"""
    if type(max_length) is not int or max_length < 1 or type(role_independent_frame) is not bool:
        raise SemanticRelationError("semantic_invalid_budget", "输入预算和 frame 声明无效。")
    limits = SEMANTIC_LIMITS_PRODUCT if limits is None else limits
    source = semantic_pair_pieces(pair, limits=limits)
    raw = source["raw_text"]
    fields = {tuple(field["path"]): field for field in source["fields"]}
    def literal(path):
        return raw[slice(*fields[path]["range"])]
    background = ("frame", "background")
    if literal(background) != '""' and not role_independent_frame:
        raise SemanticRelationError("semantic_role_dependent_frame", "交换视图仅支持调用者明确声明的角色无关共享背景。")
    required_paths = [("old", "text"), ("new", "text")]
    if literal(background) != '""':
        required_paths.append(background)
    order = sorted(("old", "new"), key=lambda role: literal((role,)))
    identical = literal(("old",)) == literal(("new",))
    cache = {}
    def encoding(text):
        if text not in cache:
            cache[text] = _checked_encoding(tokenizer, text)
        return cache[text]
    def views(state, audit=False):
        return [render_semantic_fragments(source, state, reverse=reverse, _audit=audit) for reverse in (False, True)]
    def total(state):
        return max(len(encoding(view["text"])["input_ids"]) for view in views(state))
    def body_covered(state, paths):
        for view in views(state, audit=True):
            encoded = encoding(view["text"])
            _content_tokens(view, raw, encoded, tokenizer)
            audited = {tuple(field["path"]): field for field in view["fields"]}
            if any(not audited[path]["content_token_indices"] for path in paths):
                return False
        return True
    def proposal(state, operations, reason):
        nodes, reasons = dict(state["nodes"]), dict(state["reasons"])
        expanded = []
        for path, choice in operations:
            expanded.append((path, choice))
            if identical and path and path[0] in ("old", "new"):
                expanded.append((("new" if path[0] == "old" else "old", *path[1:]), choice))
        for path, choice in expanded:
            if any(nodes.get(path[:depth]) == "full" for depth in range(len(path) + 1)):
                continue
            for depth in range(len(path)):
                nodes[path[:depth]] = "partial"
            previous = nodes.get(path)
            if isinstance(choice, list):
                choice = _merged_ranges((previous if isinstance(previous, list) else []) + choice)
            elif choice == "hole" and previous is not None:
                continue
            nodes[path] = choice
            previous_reason = reasons.get(path, "")
            reasons[path] = previous_reason if previous_reason.startswith("witness_") and not reason.startswith("witness_") else reason
        return {"nodes": nodes, "reasons": reasons}
    full_state = {"nodes": {(): "full"}, "reasons": {(): "full_fit"}}
    full_lengths = [len(encoding(view["text"])["input_ids"]) for view in views(full_state)]
    current = full_state
    if max(full_lengths) > max_length:
        token_spans = {}
        def spans(path):
            if path not in token_spans:
                text = literal(path)
                encoded = encoding(text)
                token_spans[path] = [(a, b) for a, b in encoded["offset_mapping"] if 1 <= a < b <= len(text) - 1]
            return token_spans[path]
        def safe_span(path, left, right):
            safe = fields[path]["string_boundaries"]
            def floor(value):
                low, high = 0, len(safe)
                while low < high:
                    middle = (low + high) // 2
                    if safe[middle] <= value:
                        low = middle + 1
                    else:
                        high = middle
                return safe[max(0, low - 1)]
            def ceiling(value):
                low, high = 0, len(safe)
                while low < high:
                    middle = (low + high) // 2
                    if safe[middle] < value:
                        low = middle + 1
                    else:
                        high = middle
                return safe[min(low, len(safe) - 1)]
            left, right = floor(left), ceiling(right)
            return [[left, right]] if left < right else []
        def windows(path):
            tokens, size = spans(path), len(literal(path))
            result = []
            for width in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                if tokens:
                    n = min(width, len(tokens))
                    value = _merged_ranges(safe_span(path, tokens[0][0], tokens[n - 1][1]) +
                                           safe_span(path, tokens[-n][0], tokens[-1][1]))
                else:
                    safe = fields[path]["string_boundaries"]
                    n = min(width, len(safe) - 1)
                    value = _merged_ranges([[1, safe[n]], [safe[-n - 1], size - 1]]) if n else []
                if value and value not in result:
                    result.append(value)
            return result
        current = {"nodes": {(): "partial", ("frame",): "partial", ("old",): "partial", ("new",): "partial",
                              ("frame", "mode"): "full"}, "reasons": {(): "base", ("frame", "mode"): "base_frame_mode"}}
        if background not in required_paths:
            current = proposal(current, [(background, "full")], "base_frame_empty")
        base_options = {}
        for path in sorted(required_paths, key=lambda path: (order.index(path[0]) if path[0] in order else 2)):
            choices = windows(path)
            encoded = encoding(literal(path))
            special = set(getattr(tokenizer, "all_special_ids", ()))
            pad = getattr(tokenizer, "pad_token_id", None)
            actual = [(a, b) for token, mask, (a, b) in zip(encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"])
                      if mask == 1 and token != pad and token not in special and 1 <= a < b <= len(literal(path)) - 1]
            anchors = [safe_span(path, *span) for span in (actual[:1] + actual[-1:])]
            if not anchors:
                safe = fields[path]["string_boundaries"]
                anchors = [[[safe[0], safe[1]]]] if len(safe) > 1 else ["full"]
            choice = anchors[0]
            base_options[path] = [*anchors, *choices[:4], "full"]
            current = proposal(current, [(path, choice)], "base_body")
        for path, choices in base_options.items():
            alternatives = []
            for choice in choices:
                candidate = {"nodes": dict(current["nodes"]), "reasons": dict(current["reasons"])}
                candidate["nodes"].pop(path, None)
                if identical and path[0] in ("old", "new"):
                    candidate["nodes"].pop(("new" if path[0] == "old" else "old", *path[1:]), None)
                candidate = proposal(candidate, [(path, choice)], "base_body")
                if body_covered(candidate, [path]):
                    alternatives.append((total(candidate), candidate))
            if not alternatives:
                raise SemanticRelationError("semantic_body_not_visible", "正文须有实际attention且非padding/special token；key/引号/marker不算覆盖。")
            current = min(alternatives, key=lambda item: item[0])[1]
        base_cost = total(current)
        if base_cost > max_length:
            raise SemanticRelationError("semantic_budget_too_small", "预算不足以编码共享框架及 old/new 的实际正文，不能只保留 key 头。")
        witness_ceiling = base_cost + (max_length - base_cost) // 2
        left_role, right_role = order
        relatives, seen = [], set()
        for role in order:
            for path in fields:
                if path and path[0] == role and path[1:] not in seen:
                    relatives.append(path[1:])
                    seen.add(path[1:])
        witnesses = []
        for relative in relatives:
            left, right = (left_role, *relative), (right_role, *relative)
            if left not in fields or right not in fields or literal(left) == literal(right):
                continue
            a, b = fields[left], fields[right]
            if a["value_type"] != b["value_type"] or (a["leaf"] and not a["string"]):
                witnesses.append(([(left, "full"), (right, "full")], "witness_type_or_value"))
            elif a["container"] and b["container"]:
                if a["container"] == "object" and [p[-1] for p in a["children"]] != [p[-1] for p in b["children"]]:
                    operation = [(left, "partial"), (right, "partial")]
                    operation += [(tuple(child), "hole") for child in a["children"] + b["children"]]
                    witnesses.append((operation, "witness_membership"))
            elif a["string"] and b["string"]:
                x, y = literal(left), literal(right)
                prefix = 1
                while prefix < min(len(x), len(y)) - 1 and x[prefix] == y[prefix]:
                    prefix += 1
                suffix = 1
                while suffix < min(len(x) - prefix, len(y) - prefix) and x[-suffix - 1] == y[-suffix - 1]:
                    suffix += 1
                for width in (1, 2, 4):
                    operation = []
                    for path, text in ((left, x), (right, y)):
                        tokens = spans(path)
                        points = (prefix, len(text) - suffix)
                        ranges = []
                        for point in points:
                            near = [index for index, (start, end) in enumerate(tokens) if start <= point <= end]
                            index = near[0] if near else next((i for i, (start, _) in enumerate(tokens) if start >= point), len(tokens) - 1)
                            if tokens:
                                ranges += safe_span(path, tokens[max(0, index - width)][0], tokens[min(len(tokens) - 1, index + width)][1])
                            else:
                                ranges += safe_span(path, max(1, point - width), min(len(text) - 1, point + width))
                        operation.append((path, _merged_ranges(ranges)))
                    witnesses.append((operation, "witness_string_boundary"))
        for operations, reason in witnesses:
            candidate = proposal(current, operations, reason)
            if total(candidate) <= witness_ceiling:
                current = candidate
        queues = {}
        for role in [*order, "frame"]:
            operations = []
            for path, field in fields.items():
                if not path or path[0] != role:
                    continue
                operations.append([(path, "full")])
                if field["container"] and not field["leaf"]:
                    operations.append([(path, "partial")])
                elif field["string"]:
                    operations.extend([[(path, value)] for value in windows(path)])
            queues[role] = operations
        room = max_length - total(current)
        for role, operations in queues.items():
            ceiling = min(max_length, total(current) + room // len(queues))
            for operation in operations:
                candidate = proposal(current, operation, "context_balanced")
                if total(candidate) <= ceiling:
                    current = candidate
        for index in range(max(map(len, queues.values()), default=0)):
            for role, operations in queues.items():
                if index < len(operations):
                    candidate = proposal(current, operations[index], "context_refill")
                    if total(candidate) <= max_length:
                        current = candidate
    prepared = views(current, audit=True)
    for view in prepared:
        encoded = dict(encoding(view["text"]))
        if len(encoded["input_ids"]) > max_length:
            raise SemanticRelationError("semantic_invalid_encoding", "实际编码超过预算。")
        _content_tokens(view, raw, encoded, tokenizer)
        audited = {tuple(field["path"]): field for field in view["fields"]}
        if any(not audited[path]["content_token_indices"] for path in required_paths):
            raise SemanticRelationError("semantic_body_not_visible", "正文须有实际attention且非padding/special token；key/引号/marker不算覆盖。")
        encoded.pop("offset_mapping")
        view.update({"encoding": encoded, "token_length": len(encoded["input_ids"]),
                     "special_tokens": int(tokenizer.num_special_tokens_to_add(pair=False)),
                     "role_body_covered": {"old": True, "new": True},
                     "attention_mask_sum": sum(encoded["attention_mask"])})
        view.pop("sections")
    if prepared[0]["visible_information_sha256"] != prepared[1]["visible_information_sha256"]:
        raise SemanticRelationError("semantic_information_mismatch", "交换只能置换同一份可见片段。")
    complete = prepared[0]["complete"]
    return {"full_visible_token_lengths": full_lengths,
            "recipe": SEMANTIC_INPUT_RECIPE, "input_limits": dict(limits), "raw_text": raw, "raw_pair_sha256": source["raw_pair_sha256"],
            "visible_information_sha256": prepared[0]["visible_information_sha256"], "max_length": max_length,
            "complete": complete, "requires_llm_fallback": not complete,
            "frame_condition": "caller-declared role-independent shared frame; no language verification",
            "directions": dict(zip(("forward", "reverse"), prepared))}
