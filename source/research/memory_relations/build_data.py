                       
"""记忆关系研究的训练数据构造器（research-only，确定性，无网络）。

按 `data/seed_catalog.json` 的需求语义组（主题 × 场景）与模板族（每个 split 专属的
措辞族）程序化构造七类平衡的行：同一组内的同义/相斥/时间/条件最小孪生只落在同一
split；split 以组为单位切分，模板族不跨 split。

默认v1是历史构造臂，字节与旧结果一致（其中无关干扰一律complement的旧标法仅供历史对照）；
显式 `--legacy-relabel` 修复候选temporary/uncertain对所有pair有效，并保存原标/修标审计。
`--mode author-shards` 在同一装配入口聚合预分配根的canonical pair，隔离未裁决，
复用features.request_view/request_text做泄漏和可辨识性审计；训练/封存物理分离、拒绝覆盖。
每行与每个 pair 都经 `schema.validate_row` 做生产投影与分批输出闸回放（契约合法性）；
标签是构造语义、不是人工金标，回放不证明语义正确。种子声明三条语义性质
（a/b 互斥、s 与 a 可并存、c1/c2 互斥）与逐类 `label_recipes`，进 audit 供人工复核；
机械契约边界（implicit 替换、跨作用域替换、外部规则等）不混进语义标签，见 test_data.py。

用法：
    python -m research.memory_relations.build_data --out DIRECTORY --seed 20261011
    python -m research.memory_relations.build_data --out DIRECTORY --seed 20261011 --quick
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset_recommender.agent import memory_relations as M              
from dataset_recommender.llm import dream              
from research.memory_relations import features as F              
from research.memory_relations import schema as S              
from research.memory_relations.metrics import ALLOWING_CLASSES              

SEED_PATH = Path(__file__).resolve().parent / "data" / "seed_catalog.json"
SOURCE_PROTOCOL_PATH = Path(__file__).resolve().parent / "data" / "source_protocol_v3.json"
SOURCE_FORMAT = "memory-relations-source-v3"
TRAINABLE_SPLITS = ("train", "dev", "calibration")
SEALED_SPLITS = ("heldout", "ood")
CALIBRATION_ROLES = S.CALIBRATION_ROLES
_ADJUDICATION_STATES = frozenset({"accepted", "pending", "disputed", "unresolved", "quarantined"})
SEED_FORMAT = "memory-relations-seed-v1"
WIRE_FORMAT = "memory-relations-experiment-v1"
CONSTRUCTION_VERSION = "v1"
SPLIT_ORDER = ("train", "calibration", "heldout")
PLAN = {"train": 300, "calibration": 43, "heldout": 43}
PLAN_QUICK = {"train": 32, "calibration": 16, "heldout": 16}
_EMPTY_CLASSES = ("complement", "temporary", "uncertain")
_IMPLICIT_CLASSES = ("duplicate", "complement")
_SLOT_MIN = {"a": 3, "b": 3, "s": 3, "c1": 2, "c2": 2, "t": 2}
_PHRASE_MIN = {"temporary": 3, "uncertain": 3}
_SEMANTIC_FIELDS = ("a_b_exclusive", "s_compatible", "c1_c2_exclusive")
_SEED_SEMANTIC_KEYS = frozenset(_SEMANTIC_FIELDS) | {"note"}
_PAIR_BUDGET = 2000
_NEAR_DUP_THRESHOLD = 0.85
_PUNCTUATION = re.compile(r"[\s，。；：、,.!?！？;:（）()「」“”\"'·—…]+")

                                           
LABEL_RECIPES = {
    "duplicate": "同条件同持久性下换词重述：candidate 与目标规则语义等价；证据为同一要求的重述原话。",
    "complement": "同条件下新增独立/补充要求，与目标规则可并存（s_compatible）；证据为新增要求的原话。",
    "contextual": "candidate@c2 与目标规则@c1 条件互斥（c1_c2_exclusive），两要求可并存。",
    "temporary": "同条件下本轮例外：证据含只限本轮标记，candidate permanence=temporary，旧长期要求保留。",
    "supersede": "同条件永久变更：证据明确作废旧要求，candidate tier=explicit，目标规则同作用域可替换。",
    "conflict": "同条件下 a 与 b 不可同时满足（a_b_exclusive），且证据不含时间/变更标记。",
    "uncertain": "时间/条件/关系不明：证据含糊，related/evidence_refs 为空（生产 uncertain 约定）。",
}


class BuildError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code


                                                                          

def _check_template(value: object, where: str) -> None:
    if type(value) is not dict or set(value) != {"u", "q"} \
            or any(type(value[key]) is not str or not value[key].strip() for key in ("u", "q")):
        raise BuildError("seed_template", where)


def _check_templates(value: object, where: str, minimum: int) -> None:
    if type(value) is not list or len(value) < minimum:
        raise BuildError("seed_template", where)
    for template in value:
        _check_template(template, where)


def load_seed(path: object = SEED_PATH) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(raw) is not dict or raw.get("format") != SEED_FORMAT:
        raise BuildError("seed_format")
    families, groups = raw.get("families"), raw.get("groups")
    if type(families) is not dict or not families or type(groups) is not list or not groups:
        raise BuildError("seed_shape")
    for name, family in families.items():
        if type(family) is not dict:
            raise BuildError("seed_shape", str(name))
        for key in ("restate_a", "restate_b"):
            _check_template(family.get(key), f"{name}/{key}")
        classes = family.get("classes")
        if type(classes) is not dict:
            raise BuildError("seed_shape", str(name))
        for cls in sorted(S.CLASSES):
            entry = classes.get(cls)
            if type(entry) is not dict or "full" not in entry:
                raise BuildError("seed_missing_class", f"{name}/{cls}")
            _check_templates(entry["full"], f"{name}/{cls}/full", 3)
            if cls in _EMPTY_CLASSES:
                _check_templates(entry.get("empty"), f"{name}/{cls}/empty", 3)
            if cls in _IMPLICIT_CLASSES:
                _check_templates(entry.get("implicit"), f"{name}/{cls}/implicit", 1)
        phrases = family.get("phrases") or {}
        for source, minimum in _PHRASE_MIN.items():
            values = phrases.get(source)
            if type(values) is not list or len(set(values)) != len(values) or len(values) < minimum \
                    or any(type(value) is not str or not value.strip() for value in values):
                raise BuildError("seed_phrases", f"{name}/{source}")
    seen_ids = set()
    for group in groups:
        if type(group) is not dict:
            raise BuildError("seed_shape", "group")
        gid = group.get("id")
        if type(gid) is not str or not M._ID.fullmatch(gid) or gid in seen_ids:
            raise BuildError("seed_group_id", str(gid))
        seen_ids.add(gid)
        if group.get("split") not in SPLIT_ORDER or group.get("family") not in families:
            raise BuildError("seed_group_scope", str(gid))
        for name, minimum in _SLOT_MIN.items():
            values = group.get(name)
            if type(values) is not list or len(set(values)) != len(values) or len(values) < minimum \
                    or any(type(value) is not str or not value.strip() for value in values):
                raise BuildError("seed_slot", f"{gid}/{name}")
        semantic = group.get("semantic")
        if type(semantic) is not dict or set(semantic) != _SEED_SEMANTIC_KEYS \
                or any(semantic[name] is not True for name in _SEMANTIC_FIELDS) \
                or type(semantic.get("note")) is not str or not semantic["note"].strip() \
                or len(semantic["note"]) > 200:
            raise BuildError("seed_semantic", str(gid))
    return raw


                                                                      

def _render(template: dict, slots: dict) -> tuple[str, str]:
    utterance = template["u"].format(**slots)
    quote = template["q"].format(**slots)
    if quote not in utterance:
        raise BuildError("seed_quote_mismatch", template["u"])
    return utterance, quote


def _slots(group: dict, phrases: dict, **picks: int) -> dict:
    slots = {name: group[name][picks.get(name, 0)] for name in ("a", "b", "s", "c1", "c2", "t")}
    slots["marker"] = phrases["temporary"][picks.get("marker", 0)]
    slots["vague"] = phrases["uncertain"][picks.get("vague", 0)]
    return slots


def _rule(text: str, trigger: str, tier: str, permanence: str, *, scope: str = "same") -> dict:
    return {"text": text, "trigger": trigger, "tier": tier, "permanence": permanence,
            "evidence": [], "scope": scope}


def _candidate(text: str, trigger: str, tier: str, permanence: str, quotes: list) -> dict:
    return {"text": text, "trigger": trigger, "tier": tier, "permanence": permanence,
            "evidence": list(quotes), "scope": "project"}


def _source(identifier: str, text: str) -> dict:
    return {"id": identifier, "text": text}


def _case(cls: str, candidate: dict, existing: list, sources: list, related: list, refs: list,
          construction: list, targets: tuple | None = None) -> dict:
    """一行构造结果：phenomena 带逐 existing 规则的 pair_labels（目标=整体标签，干扰=complement）。"""
    rules = [{"id": "r%d" % (position + 1), **rule} for position, rule in enumerate(existing)]
    if targets is None:
        targets = ("r1",) if rules else ()
    pair_labels = []
    for position, rule in enumerate(rules):
        if rule["id"] in targets:
            pair_labels.append({"rule_id": rule["id"], "relation": cls,
                                "origin": "target" if position == 0 else "redundant"})
        else:
            pair_labels.append({"rule_id": rule["id"], "relation": "complement",
                                "origin": "distractor"})
    request = {"candidate": candidate, "existing": rules, "evidence": list(sources),
               "context": {"scope": candidate["scope"]}}
    return {"request": request, "related": list(related), "refs": list(refs),
            "phenomena": {"construction": list(construction), "pair_labels": pair_labels}}


def _shuffled(dimensions: list, rng: random.Random) -> list:
    combos = list(itertools.product(*dimensions))
    rng.shuffle(combos)
    return combos


def _cases_duplicate(group, family, rng_seed):
    templates = family["classes"]["duplicate"]
    choices = [(item, "explicit") for item in templates["full"]] + \
              [(item, "implicit") for item in templates.get("implicit", [])]

    def rows():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["a"])), range(len(group["t"])),
                range(len(choices)), range(2)]
        for incumbent, restated, it, ic, redundant in _shuffled(dims, rng):
            template, tier = choices[ic]
            slots = _slots(group, family["phrases"], t=it, a=restated)
            utterance, quote = _render(template, slots)
            existing = [_rule(group["a"][incumbent], slots["t"], "explicit", "persistent")]
            related = ["r1"]
            construction = ["exact_restatement" if incumbent == restated
                            else "paraphrase_restatement"]
            targets = None
            if redundant:
                existing.append(_rule(group["a"][(incumbent + 1) % len(group["a"])], slots["t"],
                                      "explicit", "persistent"))
                related = ["r1", "r2"]
                targets = ("r1", "r2")
                construction.append("redundant_incumbent")
            if tier == "implicit":
                construction.append("implicit_candidate")
            candidate = _candidate(group["a"][restated], slots["t"], tier, "persistent", [quote])
            yield _case("duplicate", candidate, existing, [_source("e1", utterance)], related,
                        ["e1"], construction, targets=targets)

    return rows(), None


def _cases_complement(group, family, rng_seed):
    templates = family["classes"]["complement"]
    meta = [(item, "explicit") for item in templates["full"]] + \
           [(item, "implicit") for item in templates.get("implicit", [])]

    def full():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["s"])), range(len(group["t"])),
                range(len(meta))]
        for incumbent, addition, it, ic in _shuffled(dims, rng):
            template, tier = meta[ic]
            slots = _slots(group, family["phrases"], t=it, s=addition)
            utterance, quote = _render(template, slots)
            existing = [_rule(group["a"][incumbent], slots["t"], "explicit", "persistent")]
            candidate = _candidate(slots["s"], slots["t"], tier, "persistent", [quote])
            construction = ["compatible_addition"]
            if tier == "implicit":
                construction.append("implicit_candidate")
            yield _case("complement", candidate, existing, [_source("e1", utterance)], ["r1"],
                        ["e1"], construction)

    def empty():
        rng = random.Random(rng_seed + "|empty")
        dims = [range(len(group["s"])), range(len(group["t"])), range(len(templates["empty"]))]
        for addition, it, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], t=it, s=addition)
            utterance, quote = _render(templates["empty"][ic], slots)
            candidate = _candidate(slots["s"], slots["t"], "explicit", "persistent", [quote])
            yield _case("complement", candidate, [], [_source("e1", utterance)], [], ["e1"],
                        ["compatible_addition", "empty_existing"])

    return full(), empty()


def _cases_contextual(group, family, rng_seed):
    templates = family["classes"]["contextual"]["full"]

    def rows():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["b"])), range(len(group["c1"])),
                range(len(group["c2"])), range(len(templates))]
        for ia, ib, ic1, ic2, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], a=ia, b=ib, c1=ic1, c2=ic2)
            utterance, quote = _render(templates[ic], slots)
            existing = [_rule(slots["a"], slots["c1"], "explicit", "persistent")]
            candidate = _candidate(slots["b"], slots["c2"], "explicit", "persistent", [quote])
            yield _case("contextual", candidate, existing, [_source("e1", utterance)], ["r1"],
                        ["e1"], ["exclusive_conditions"])

    return rows(), None


def _two_source(slots: dict, restate: dict, quote: str, sources: list, refs: list,
                construction: list) -> list:
    old_utterance, old_quote = _render(restate, slots)
    sources.insert(0, _source("e1", old_utterance))
    sources[1]["id"] = "e2"
    refs[:] = ["e1", "e2"]
    construction.append("two_source_grounding")
    return [old_quote, quote]


def _cases_temporary(group, family, rng_seed):
    templates = family["classes"]["temporary"]
    restate = family["restate_a"]

    def full():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["b"])), range(len(group["t"])),
                range(len(family["phrases"]["temporary"])), range(len(templates["full"])), range(2)]
        for ia, ib, it, im, ic, two in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], a=ia, b=ib, t=it, marker=im)
            utterance, quote = _render(templates["full"][ic], slots)
            existing = [_rule(slots["a"], slots["t"], "explicit", "persistent")]
            sources, refs = [_source("e1", utterance)], ["e1"]
            construction = ["round_exception"]
            quotes = [quote]
            if two:
                quotes = _two_source(slots, restate, quote, sources, refs, construction)
            candidate = _candidate(slots["b"], slots["t"], "explicit", "temporary", quotes)
            yield _case("temporary", candidate, existing, sources, ["r1"], refs, construction)

    def empty():
        rng = random.Random(rng_seed + "|empty")
        dims = [range(len(group["b"])), range(len(group["t"])),
                range(len(family["phrases"]["temporary"])), range(len(templates["empty"]))]
        for ib, it, im, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], b=ib, t=it, marker=im)
            utterance, quote = _render(templates["empty"][ic], slots)
            candidate = _candidate(slots["b"], slots["t"], "explicit", "temporary", [quote])
            yield _case("temporary", candidate, [], [_source("e1", utterance)], [], ["e1"],
                        ["round_exception", "empty_existing"])

    return full(), empty()


def _cases_supersede(group, family, rng_seed):
    templates = family["classes"]["supersede"]["full"]
    restate = family["restate_a"]

    def rows():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["b"])), range(len(group["t"])),
                range(len(templates)), range(2)]
        for ia, ib, it, ic, two in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], a=ia, b=ib, t=it)
            utterance, quote = _render(templates[ic], slots)
            existing = [_rule(slots["a"], slots["t"], "explicit", "persistent")]
            sources, refs = [_source("e1", utterance)], ["e1"]
            construction = ["permanent_change"]
            quotes = [quote]
            if two:
                quotes = _two_source(slots, restate, quote, sources, refs, construction)
            candidate = _candidate(slots["b"], slots["t"], "explicit", "persistent", quotes)
            yield _case("supersede", candidate, existing, sources, ["r1"], refs, construction)

    return rows(), None


def _cases_conflict(group, family, rng_seed):
    templates = family["classes"]["conflict"]["full"]

    def rows():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["b"])), range(len(group["t"])),
                range(len(templates))]
        for ia, ib, it, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], a=ia, b=ib, t=it)
            utterance, quote = _render(templates[ic], slots)
            existing = [_rule(slots["a"], slots["t"], "explicit", "persistent")]
            candidate = _candidate(slots["b"], slots["t"], "explicit", "persistent", [quote])
            yield _case("conflict", candidate, existing, [_source("e1", utterance)], ["r1"],
                        ["e1"], ["same_condition_contradiction"])

    return rows(), None


def _cases_uncertain(group, family, rng_seed):
    templates = family["classes"]["uncertain"]

    def full():
        rng = random.Random(rng_seed + "|full")
        dims = [range(len(group["a"])), range(len(group["b"])), range(len(group["t"])),
                range(len(family["phrases"]["uncertain"])), range(len(templates["full"]))]
        for ia, ib, it, iv, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], a=ia, b=ib, t=it, vague=iv)
            utterance, quote = _render(templates["full"][ic], slots)
            existing = [_rule(slots["a"], slots["t"], "explicit", "persistent")]
            candidate = _candidate(slots["b"], slots["t"], "explicit", "unknown", [quote])
            yield _case("uncertain", candidate, existing, [_source("e1", utterance)], [], [],
                        ["vague_time_or_condition"])

    def empty():
        rng = random.Random(rng_seed + "|empty")
        dims = [range(len(group["b"])), range(len(group["t"])),
                range(len(family["phrases"]["uncertain"])), range(len(templates["empty"]))]
        for ib, it, iv, ic in _shuffled(dims, rng):
            slots = _slots(group, family["phrases"], b=ib, t=it, vague=iv)
            utterance, quote = _render(templates["empty"][ic], slots)
            candidate = _candidate(slots["b"], slots["t"], "explicit", "unknown", [quote])
            yield _case("uncertain", candidate, [], [_source("e1", utterance)], [], [],
                        ["vague_time_or_condition", "empty_existing"])

    return full(), empty()


_CASE_STREAMS = {
    "duplicate": _cases_duplicate,
    "complement": _cases_complement,
    "contextual": _cases_contextual,
    "temporary": _cases_temporary,
    "supersede": _cases_supersede,
    "conflict": _cases_conflict,
    "uncertain": _cases_uncertain,
}


def _decorate(cls: str, index: int, case: dict, slots: dict) -> dict:
    """按行号施加作用域/干扰项变体：注入不改变标签语义，只扩展访存面。"""
    request = case["request"]
    construction = list(case["phenomena"]["construction"])
    pair_labels = list(case["phenomena"]["pair_labels"])
    existing = request["existing"]
    if index % 17 == 16 and all(rule["scope"] == "same" for rule in existing):
        request["candidate"]["scope"] = "global"
        request["context"]["scope"] = "global"
        construction.append("global_scope")
    if cls != "complement" and index % 5 == 4 \
            and not any(rule["id"] == "r2" for rule in existing):
        scope = "inherited_global" if (index % 10 == 9 and request["context"]["scope"] == "project") \
            else "same"
        existing.append({"id": "r2", **_rule(slots["s"], slots["t"], "explicit", "persistent",
                                            scope=scope)})
        construction.append("distractor_rule")
        pair_labels.append({"rule_id": "r2", "relation": "complement", "origin": "distractor"})
        if scope == "inherited_global":
            construction.append("inherited_global_incumbent")
    case["phenomena"] = {"construction": construction, "pair_labels": pair_labels}
    return case


def _row_signature(request: dict, label: str) -> str:
    return json.dumps({"view": S.feature_view(request), "context": request["context"],
                       "scopes": [rule["scope"] for rule in request["existing"]],
                       "label": label}, ensure_ascii=False, sort_keys=True)


def _feature_key(request: dict) -> str:
    return json.dumps(S.feature_view(request), ensure_ascii=False, sort_keys=True)


def _normalized_key(request: dict) -> str:
    canonical = S.canonical_request(request)
    candidate = canonical["candidate"]
    shaped = [(dream._norm(candidate["text"]), dream._norm(candidate["trigger"]),
               candidate.get("tier", ""), candidate.get("permanence", ""))]
    shaped.extend((dream._norm(rule["text"]), dream._norm(rule["trigger"]), rule.get("tier", ""),
                   rule.get("permanence", "")) for rule in canonical["existing"])
    shaped.append(tuple(dream._norm(source["text"]) for source in canonical["evidence"]))
    text = json.dumps(shaped, ensure_ascii=False, sort_keys=True)
    return _PUNCTUATION.sub("", text)


def _group_class_rows(group: dict, family: dict, cls: str, split: str, count: int,
                      seed: int, seen: set) -> list[dict]:
    full_iter, empty_iter = _CASE_STREAMS[cls](group, family, f"{seed}|{group['id']}|{cls}")
    empty_count = min(count - 1, max(1, count // 5)) if empty_iter is not None else 0
    streams = [("full", full_iter, count - empty_count)]
    if empty_iter is not None:
        streams.append(("empty", empty_iter, empty_count))
    rows, index = [], 0
    for name, iterator, needed in streams:
        taken = 0
        for case in iterator:
            if taken >= needed:
                break
            case = _decorate(cls, index, case, _slots(group, family["phrases"]))
            signature = _row_signature(case["request"], cls)
            if signature in seen:
                continue
            rule_ids = sorted(rule["id"] for rule in case["request"]["existing"])
            labelled = sorted(item["rule_id"] for item in case["phenomena"]["pair_labels"])
            if rule_ids != labelled:
                raise BuildError("pair_labels_mismatch", f"{group['id']}/{cls}: {labelled}!={rule_ids}")
            seen.add(signature)
            rows.append({"id": f"{group['id']}-{cls}-{index:02d}", "group_id": group["id"],
                         "split": split, "request": case["request"], "label": cls,
                         "label_origin": "construction", "related_ids": case["related"],
                         "evidence_refs": case["refs"], "phenomena": case["phenomena"]})
            index += 1
            taken += 1
        if taken < needed:
            raise BuildError("insufficient_cases", f"{group['id']}/{cls}/{name}: {taken}<{needed}")
    return rows


def _grounded_source_ids(row: dict) -> list:
    request = row["request"]
    quotes = request["candidate"]["evidence"]
    return sorted(source["id"] for source in request["evidence"]
                  if any(dream._evidence_checks([quote], [(0, dream._norm(source["text"]))])[0]
                         for quote in quotes))


def _pair_line(row: dict, rule: dict | None, relation: str, related: list, refs: list,
               origin: str) -> dict:
    identifier = "empty" if rule is None else rule["id"]
    request = {"candidate": copy.deepcopy(row["request"]["candidate"]),
               "existing": [] if rule is None else [copy.deepcopy(rule)],
               "evidence": copy.deepcopy(row["request"]["evidence"]),
               "context": copy.deepcopy(row["request"]["context"])}
    return {"id": f"{row['id']}-{identifier}", "group_id": row["group_id"], "split": row["split"],
            "request": request, "label": relation, "label_origin": "construction",
            "related_ids": list(related), "evidence_refs": list(refs),
            "phenomena": {"construction": [f"pair_of:{row['id']}"], "row_id": row["id"],
                          "row_label": row["label"], "rule_id": identifier, "pair_origin": origin}}


def _pair_rows(rows: list, *, legacy_v1: bool = False) -> list:
    """候选的限时/未决性质不因换成无关旧规则消失；v1历史臂须显式保留旧标法。"""
    pairs = []
    for row in rows:
        labels = {item["rule_id"]: item for item in row["phenomena"]["pair_labels"]}
        rules = row["request"]["existing"]
        if not rules:
            pairs.append(_pair_line(row, None, row["label"], [], row["evidence_refs"],
                                    "empty_sentinel"))
            continue
        grounded = _grounded_source_ids(row)
        for rule in rules:
            entry = labels[rule["id"]]
            original = entry["relation"]
            relation = row["label"] if not legacy_v1 and row["label"] in (
                "temporary", "uncertain") else original
            related = [] if relation == "uncertain" else (
                [rule["id"]] if rule["id"] in row["related_ids"] else [])
            refs = [] if relation == "uncertain" else (list(row["evidence_refs"]) or grounded)
            pair = _pair_line(row, rule, relation, related, refs,
                              entry.get("origin", "distractor"))
            if relation != original:
                pair["phenomena"]["legacy_label"] = original
                pair["phenomena"]["relabel_rationale"] = "候选限时/未决性质对每个pair成立，与旧规则相关性无关。"
            pairs.append(pair)
    return pairs


                                                                        

def _near_duplicate_report(rows: list, threshold: float = _NEAR_DUP_THRESHOLD,
                           window: int = 12, example_limit: int = 5) -> dict:
    entries = sorted((_normalized_key(row["request"]), position) for position, row in enumerate(rows))
    pairs, examples = 0, []
    for position, (text, row_index) in enumerate(entries):
        for other_text, other_index in entries[position + 1:position + 1 + window]:
            if rows[row_index]["split"] == rows[other_index]["split"]:
                continue
            ratio = SequenceMatcher(None, text, other_text).ratio()
            if ratio >= threshold:
                pairs += 1
                if len(examples) < example_limit:
                    examples.append({"ratio": round(ratio, 3), "left": rows[row_index]["id"],
                                     "right": rows[other_index]["id"],
                                     "left_text": text[:40], "right_text": other_text[:40]})
    return {"threshold": threshold, "pairs": pairs, "examples": examples}


def _run_checks(rows: list, pairs: list, seed_doc: dict, plan: dict) -> dict:
    failures = []
    for row in rows:
        try:
            S.validate_row(row)
        except S.RowError as exc:
            failures.append({"id": row.get("id"), "code": str(exc)})
    if failures:
        raise BuildError("row_contract_replay", json.dumps(failures[:8], ensure_ascii=False))
    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    group_violations = sorted(gid for gid, splits in group_splits.items() if len(splits) > 1)
    family_splits = {}
    for group in seed_doc["groups"]:
        family_splits.setdefault(group["family"], set()).add(group["split"])
    family_violations = sorted(name for name, splits in family_splits.items() if len(splits) > 1)
    feature_splits, normalized_splits = {}, {}
    for row in rows:
        request = row["request"]
        feature_splits.setdefault(_feature_key(request), set()).add(row["split"])
        normalized_splits.setdefault(_normalized_key(request), set()).add(row["split"])
    feature_overlaps = sorted(key[:80] for key, splits in feature_splits.items() if len(splits) > 1)
    normalized_overlaps = sorted(key[:80] for key, splits in normalized_splits.items() if len(splits) > 1)
    for name, violations in (("group_split_purity", group_violations),
                             ("family_split_purity", family_violations),
                             ("cross_split_feature_overlap", feature_overlaps),
                             ("cross_split_normalized_overlap", normalized_overlaps)):
        if violations:
            raise BuildError(name, json.dumps(violations[:8], ensure_ascii=False))
    counts = {split: {cls: 0 for cls in sorted(S.CLASSES)} for split in SPLIT_ORDER}
    for row in rows:
        counts[row["split"]][row["label"]] += 1
    for split, target in plan.items():
        for cls, actual in counts[split].items():
            if actual != target:
                raise BuildError("class_balance", f"{split}/{cls}: {actual}!={target}")
    empty_counts = {cls: 0 for cls in _EMPTY_CLASSES}
    for row in rows:
        if row["split"] == "train" and not row["request"]["existing"] \
                and row["label"] in _EMPTY_CLASSES:
            empty_counts[row["label"]] += 1
    if any(value == 0 for value in empty_counts.values()):
        raise BuildError("empty_existing_missing", json.dumps(empty_counts))
    attributes = {}
    for cls in sorted(S.CLASSES):
        tiers = {}
        permanence = {}
        for row in rows:
            if row["label"] != cls:
                continue
            tiers[row["request"]["candidate"]["tier"]] = tiers.get(row["request"]["candidate"]["tier"], 0) + 1
            permanence[row["request"]["candidate"]["permanence"]] = \
                permanence.get(row["request"]["candidate"]["permanence"], 0) + 1
        attributes[cls] = {"tier": tiers, "permanence": permanence}
    pair_failures = []
    for pair in pairs:
        try:
            S.validate_row(pair)
        except S.RowError as exc:
            pair_failures.append({"id": pair.get("id"), "code": str(exc)})
    if pair_failures:
        raise BuildError("pair_contract_replay", json.dumps(pair_failures[:8], ensure_ascii=False))
    pair_counts = {split: 0 for split in SPLIT_ORDER}
    origins, pair_labels = {}, {}
    for pair in pairs:
        pair_counts[pair["split"]] += 1
        origin = pair["phenomena"]["pair_origin"]
        origins[origin] = origins.get(origin, 0) + 1
        pair_labels[pair["label"]] = pair_labels.get(pair["label"], 0) + 1
    if pair_counts["train"] <= _PAIR_BUDGET and plan["train"] >= PLAN["train"]:
        raise BuildError("pair_budget", f"train pairs {pair_counts['train']} <= {_PAIR_BUDGET}")
    semantics = [{"id": group["id"], "split": group["split"], "note": group["semantic"]["note"],
                  **{name: group["semantic"][name] for name in _SEMANTIC_FIELDS}}
                 for group in seed_doc["groups"]]
    return {
        "counts": counts,
        "class_totals": {cls: sum(counts[split][cls] for split in SPLIT_ORDER)
                         for cls in sorted(S.CLASSES)},
        "empty_existing_train": empty_counts,
        "candidate_attribute_distribution": attributes,
        "pairs": {"counts": pair_counts, "total": len(pairs), "origins": origins,
                  "labels": pair_labels, "train_budget": _PAIR_BUDGET},
        "group_semantics": semantics,
        "label_recipes": dict(LABEL_RECIPES),
        "checks": {
            "rows_contract_replayed": len(rows),
            "pairs_contract_replayed": len(pairs),
            "group_split_purity": {"violations": []},
            "family_split_purity": {"violations": []},
            "row_signature_unique": {"duplicates": 0},
            "pair_labels_cover_existing": {"violations": []},
            "cross_split_feature_overlap": {"violations": []},
            "cross_split_normalized_overlap": {"violations": []},
            "near_duplicate_pairs": _near_duplicate_report(rows),
        },
    }


                                                                        

def build_corpus(seed: int, plan: dict, seed_path: object = SEED_PATH,
                 *, repair_pair_labels: bool = False) -> dict:
    seed_bytes = Path(seed_path).read_bytes()
    seed_doc = load_seed(seed_path)
    rows, seen = [], set()
    for split in SPLIT_ORDER:
        target = int(plan[split])
        groups = [group for group in seed_doc["groups"] if group["split"] == split]
        if not groups:
            raise BuildError("missing_split", split)
        for cls in sorted(S.CLASSES):
            base, extra = divmod(target, len(groups))
            for position, group in enumerate(groups):
                count = base + (1 if position < extra else 0)
                if count <= 0:
                    continue
                family = seed_doc["families"][group["family"]]
                rows.extend(_group_class_rows(group, family, cls, split, count, seed, seen))
    pairs = _pair_rows(rows, legacy_v1=not repair_pair_labels)
    report = _run_checks(rows, pairs, seed_doc, {split: int(plan[split]) for split in SPLIT_ORDER})
    result = {"seed": seed, "plan": {split: int(plan[split]) for split in SPLIT_ORDER},
              "rows": rows, "pairs": pairs, "report": report, "seed_doc": seed_doc,
              "seed_sha256": hashlib.sha256(seed_bytes).hexdigest()}
    if repair_pair_labels:
        result["construction_version"] = "v1-candidate-status-relabel"
        result["legacy_label_audit"] = [
            {"id": pair["id"], "group_id": pair["group_id"], "split": pair["split"],
             "original_label": pair["phenomena"]["legacy_label"], "repaired_label": pair["label"],
             "row_label": pair["phenomena"]["row_label"],
             "rationale": pair["phenomena"]["relabel_rationale"]}
            for pair in pairs if "legacy_label" in pair["phenomena"]]
    return result


def _jsonl_bytes(rows: list) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")


def write_outputs(result: dict, out_dir: object) -> dict:
    rows, pairs = result["rows"], result["pairs"]
    payloads = {"all.jsonl": _jsonl_bytes(rows), "pairs.jsonl": _jsonl_bytes(pairs)}
    for split in SPLIT_ORDER:
        payloads[f"{split}.jsonl"] = _jsonl_bytes([row for row in rows if row["split"] == split])
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())}
    report = result["report"]
    groups = {}
    for group in result["seed_doc"]["groups"]:
        groups.setdefault(group["split"], []).append(
            {"id": group["id"], "family": group["family"], "need": group["need"]})
    audit = {
        "format": "memory-relations-data-audit-v1",
        "wire_format": WIRE_FORMAT,
        "construction_version": CONSTRUCTION_VERSION,
        "generator": "research/memory_relations/build_data.py",
        "seed": result["seed"],
        "seed_catalog_sha256": result["seed_sha256"],
        "plan": result["plan"],
        "label_origin": "construction",
        "classes": sorted(S.CLASSES),
        "groups": groups,
        "hashes": hashes,
        **{key: report[key] for key in ("counts", "class_totals", "empty_existing_train",
                                        "candidate_attribute_distribution", "pairs",
                                        "group_semantics", "label_recipes", "checks")},
    }
    if "legacy_label_audit" in result:
        audit["construction_version"] = result["construction_version"]
        audit["legacy_relabel_count"] = len(result["legacy_label_audit"])
        payloads["legacy_labels.jsonl"] = _jsonl_bytes(result["legacy_label_audit"])
        hashes["legacy_labels.jsonl"] = hashlib.sha256(payloads["legacy_labels.jsonl"]).hexdigest()
    payloads["audit.json"] = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
                              + "\n").encode("utf-8")
    payloads["hashes.json"] = (json.dumps(
        {"algorithm": "sha256", "seed": result["seed"], "rows_total": len(rows),
         "pairs_total": len(pairs), "files": hashes},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new_directory(Path(out_dir), payloads)
    return {"hashes": hashes, "rows_total": len(rows), "pairs_total": len(pairs),
            "counts": report["counts"]}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _write_new_directory(out: Path, payloads: dict) -> None:
    if out.exists():
        raise BuildError("output_exists")
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as handle:
            handle.write(data)


def preallocate_author_roots(author_roots: dict, *, seed: int = 20261011,
                             dev_roots: int | dict = 2,
                             sealed_authors: dict | None = None) -> list[dict]:
    """只读作者/根编号预分配，不读文本、标签、教师判断或学生预测；在创作前冻结返回值。"""
    sealed_authors = dict(sealed_authors or {})
    development = set(author_roots) - set(sealed_authors)
    dev_counts = dict(dev_roots) if type(dev_roots) is dict else {
        author: dev_roots for author in development}
    if set(dev_counts) != development or any(type(count) is not int or count not in (2, 3)
                                             for count in dev_counts.values()) \
            or set(sealed_authors) - set(author_roots) \
            or any(split not in SEALED_SPLITS for split in sealed_authors.values()):
        raise BuildError("source_allocation")
    allocated, seen = [], set()
    for author, roots in sorted(author_roots.items()):
        if type(author) is not str or not M._ID.fullmatch(author) or type(roots) is not list \
                or len(roots) != 20:
            raise BuildError("source_allocation")
        if any(type(root) is not str or not M._ID.fullmatch(root) for root in roots):
            raise BuildError("source_allocation")
        for index, root in enumerate(sorted(roots)):
            if root in seen:
                raise BuildError("duplicate_root_assignment")
            seen.add(root)
            split = sealed_authors.get(author) or (
                "train" if index < 15 else "dev" if index < 15 + dev_counts[author] else "calibration")
            allocated.append({"root_id": root, "author": author, "split": split})
    calibration = sorted((entry for entry in allocated if entry["split"] == "calibration"),
                         key=lambda entry: hashlib.sha256(
                             f"{seed}|{entry['root_id']}".encode("utf-8")).hexdigest())
    by_role = defaultdict(list)
    for index, entry in enumerate(calibration):
        entry.update(calibration_role=CALIBRATION_ROLES[index % len(CALIBRATION_ROLES)],
                     primary_variant="operating")
        by_role[entry["calibration_role"]].append(entry)
    allow_count = round(len(calibration) * 0.6)
    role_counts = {role: len(entries) * 3 // 5 for role, entries in by_role.items()}
    remaining = allow_count - sum(role_counts.values())
    role_order = sorted(by_role, key=lambda role: (-(len(by_role[role]) * 3 % 5), role))
    for role in role_order[:remaining]:
        role_counts[role] += 1
    for role, entries in by_role.items():
        for index, entry in enumerate(entries):
            entry["operating_allow"] = index < role_counts[role]
    return sorted(allocated, key=lambda entry: entry["root_id"])


def load_source_protocol(path: object = SOURCE_PROTOCOL_PATH) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(raw) is not dict or raw.get("format") != SOURCE_FORMAT \
            or raw.get("human_gold_available") is not False:
        raise BuildError("source_protocol")
    entries = raw.get("root_assignments")
    if type(entries) is not list or not entries:
        raise BuildError("missing_root_preallocation")
    assignments, author_splits = {}, defaultdict(set)
    for entry in entries:
        if type(entry) is not dict or not {"root_id", "author", "split"} <= set(entry):
            raise BuildError("source_allocation")
        split, root, author = (entry[name] for name in ("split", "root_id", "author"))
        if type(root) is not str or not M._ID.fullmatch(root) or type(author) is not str \
                or not M._ID.fullmatch(author) or split not in TRAINABLE_SPLITS + SEALED_SPLITS:
            raise BuildError("source_allocation")
        if root in assignments:
            raise BuildError("duplicate_root_assignment")
        extra = {"calibration_role", "primary_variant", "operating_allow"}
        if set(entry) - ({"root_id", "author", "split"} | extra):
            raise BuildError("source_allocation")
        if split == "calibration":
            if not extra <= set(entry) or entry["calibration_role"] not in CALIBRATION_ROLES \
                    or type(entry["primary_variant"]) is not str \
                    or not entry["primary_variant"].strip() \
                    or type(entry["operating_allow"]) is not bool:
                raise BuildError("source_calibration_allocation")
        elif set(entry) & extra:
            raise BuildError("source_calibration_allocation")
        assignments[root] = dict(entry)
        author_splits[author].add(split)
    for author, splits in author_splits.items():
        if splits & set(SEALED_SPLITS) and len(splits) != 1:
            raise BuildError("sealed_author_mixed_splits")
        counts = Counter(entry["split"] for entry in assignments.values() if entry["author"] == author)
        if sum(counts.values()) != 20:
            raise BuildError("author_root_count")
        if not splits & set(SEALED_SPLITS) and (
                counts["train"] != 15 or counts["dev"] not in (2, 3)
                or counts["calibration"] != 5 - counts["dev"]):
            raise BuildError("author_split_counts")
    return {"document": raw, "assignments": assignments,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _normal_student_value(value: object) -> object:
    if type(value) is str:
        return dream._norm(value)
    if type(value) is list:
        return [_normal_student_value(item) for item in value]
    if type(value) is dict:
        return {key: _normal_student_value(item) for key, item in value.items()}
    return value


def input_fingerprints(request: dict) -> dict[str, str]:
    """只散列现有真实学生投影/序列化；归一化指纹不是语义无泄漏证明。"""
    canonical = S.canonical_request(request)
    view, text = F.request_view(canonical), F.request_text(canonical)
    values = {"canonical_request": canonical, "student_view": view, "student_text": text,
              "normalized_view": _normal_student_value(view),
              "normalized_text": dream._norm(text)}
    return {name: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode(
        "utf-8")).hexdigest() for name, value in values.items()}


def _source_leakage_checks(rows: list) -> dict:
    groups, roots, fingerprints = defaultdict(set), defaultdict(set), {}
    root_groups, group_roots, root_metadata = defaultdict(set), defaultdict(set), defaultdict(set)
    labels, fingerprint_roots = defaultdict(set), defaultdict(set)
    for row in rows:
        root = row["phenomena"]["root_id"]
        split = row["split"]
        groups[row["group_id"]].add(split)
        roots[root].add(split)
        root_groups[root].add(row["group_id"])
        group_roots[row["group_id"]].add(root)
        root_metadata[root].add(tuple(row["phenomena"][key] for key in ("domain", "family", "author")))
        for name, fingerprint in input_fingerprints(row["request"]).items():
            fingerprints.setdefault((name, fingerprint), set()).add(split)
            if name != "canonical_request":
                labels[(name, fingerprint)].add(row["label"])
            if name in ("student_view", "student_text"):
                fingerprint_roots[(name, fingerprint)].add(root)
    checks = {"group_split_leakage": groups, "root_split_leakage": roots,
              "root_group_mismatch": root_groups, "group_root_mismatch": group_roots,
              "root_metadata_mismatch": root_metadata,
              "cross_split_input_fingerprint": fingerprints, "student_label_conflict": labels,
              "cross_root_duplicate_input": fingerprint_roots}
    for code, values in checks.items():
        if any(len(items) > 1 for items in values.values()):
            raise BuildError(code)
    return {code: {"violations": 0} for code in checks}


def _source_near_candidates(rows: list, *, window: int = 12, example_limit: int = 100) -> dict:
    entries = sorted((dream._norm(F.request_text(row["request"])), row["id"], row["split"])
                     for row in rows)
    compared, count, examples = 0, 0, []
    for index, (text, identifier, split) in enumerate(entries):
        for other_text, other_id, other_split in entries[index + 1:index + 1 + window]:
            if split == other_split:
                continue
            compared += 1
            ratio = SequenceMatcher(None, text, other_text).ratio()
            if ratio >= _NEAR_DUP_THRESHOLD:
                count += 1
                if len(examples) < example_limit:
                    examples.append({"left_id": identifier, "right_id": other_id,
                                     "left_split": split, "right_split": other_split,
                                     "ratio": round(ratio, 6)})
    return {"method": "sorted-student-text-window-sequence-matcher", "window": window,
            "threshold": _NEAR_DUP_THRESHOLD, "comparisons": compared, "candidate_pairs": count,
            "examples": examples, "examples_truncated": count > len(examples),
            "exhaustive": False, "semantic_no_leakage_certified": False,
            "review_required": count > 0}


def _source_counts(rows: list) -> dict:
    lengths = sorted(len(F.request_text(row["request"])) for row in rows)
    metadata = ("domain", "family", "author")
    by_label = {}
    for label in sorted(S.CLASSES):
        members = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "rows": len(members),
            "evidence_sources": sum(len(row["request"]["evidence"]) for row in members),
            "candidate_quotes": sum(len(row["request"]["candidate"]["evidence"]) for row in members),
            "related_references": sum(len(row["related_ids"]) for row in members),
            "evidence_references": sum(len(row["evidence_refs"]) for row in members),
            "empty_existing": sum(not row["request"]["existing"] for row in members)}
    return {"by_label": by_label, "rows": len(rows), "groups": len({row["group_id"] for row in rows}),
            "roots": len({row["phenomena"]["root_id"] for row in rows}),
            "labels": {label: sum(row["label"] == label for row in rows) for label in sorted(S.CLASSES)},
            "label_origins": dict(sorted(Counter(row["label_origin"] for row in rows).items())),
            "evidence_sources": sum(len(row["request"]["evidence"]) for row in rows),
            "candidate_quotes": sum(len(row["request"]["candidate"]["evidence"]) for row in rows),
            "related_references": sum(len(row["related_ids"]) for row in rows),
            "evidence_references": sum(len(row["evidence_refs"]) for row in rows),
            "reference_cardinalities": {
                key: dict(sorted(Counter(len(row[key]) for row in rows).items()))
                for key in ("related_ids", "evidence_refs")},
            "existing_cardinalities": dict(sorted(Counter(
                len(row["request"]["existing"]) for row in rows).items())),
            "student_text_chars": {"min": lengths[0] if lengths else None,
                                   "max": lengths[-1] if lengths else None,
                                   "p95": lengths[min(len(lengths) - 1, int(len(lengths) * .95))]
                                   if lengths else None},
            **{key: dict(sorted(Counter(row["phenomena"][key] for row in rows).items()))
               for key in metadata}}


def _calibration_allocation_counts(rows: list, assignments: dict) -> dict:
    primary = [row for row in rows if row["split"] == "calibration"
               and row["phenomena"]["operating_primary"]]
    planned = [entry for entry in assignments.values() if entry["split"] == "calibration"]
    def counts(entries, predicate):
        allowed = sum(predicate(entry) for entry in entries)
        return {"total": len(entries), "allow": allowed, "block": len(entries) - allowed,
                "allow_fraction": allowed / len(entries) if entries else None}
    return {"target_allow_fraction": 0.6, "is_user_distribution": False,
            "selection": "frozen-root-primary-variant-not-predictions",
            "planned": counts(planned, lambda entry: entry["operating_allow"]),
            "accepted": counts(primary, lambda row: row["label"] in ALLOWING_CLASSES),
            "roles": {role: {"groups": len({row["group_id"] for row in rows
                                            if row["split"] == "calibration"
                                            and row["phenomena"]["calibration_role"] == role}),
                             "primary": counts([row for row in primary
                                                if row["phenomena"]["calibration_role"] == role],
                                               lambda row: row["label"] in ALLOWING_CLASSES)}
                      for role in CALIBRATION_ROLES}}


def build_author_shards(shard_paths: list, protocol_path: object = SOURCE_PROTOCOL_PATH) -> dict:
    """协调装配入口；验证所有已给shard，隔离未裁决，不修标签/权限，也不读取其它语料。"""
    protocol = load_source_protocol(protocol_path)
    assignments = protocol["assignments"]
    rows, sources, ids, variants = [], [], set(), set()
    for path in sorted((Path(path) for path in shard_paths), key=lambda path: str(path.resolve())):
        shard = S.load_rows(path)
        if not shard:
            raise BuildError("empty_author_shard")
        source = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "rows": len(shard), "splits": []}
        sources.append(source)
        for original in shard:
            row = copy.deepcopy(original)
            try:
                S.validate_row(row)
            except S.RowError as exc:
                raise BuildError("row_contract_replay", exc.code) from None
            if len(row["request"]["existing"]) > 1 or row["label_origin"] != "agent_review" \
                    or not re.fullmatch(r"[0-9]+", row["id"]):
                raise BuildError("author_pair_contract")
            if row["request"] != S.canonical_request(row["request"]):
                raise BuildError("noncanonical_author_request")
            meta = row["phenomena"]
            if type(meta) is not dict or "root_id" not in meta:
                raise BuildError("missing_author_metadata")
            assignment = assignments.get(meta["root_id"])
            if assignment is None or assignment["author"] != meta["author"] \
                    or assignment["split"] != row["split"]:
                raise BuildError("root_assignment_mismatch")
            key = (meta["root_id"], meta["variant"])
            if row["id"] in ids or key in variants:
                raise BuildError("duplicate_author_row")
            ids.add(row["id"])
            variants.add(key)
            status = meta.get("adjudication_status", "accepted")
            if type(status) is not str or status not in _ADJUDICATION_STATES:
                raise BuildError("invalid_adjudication_status")
            primary = row["split"] == "calibration" and meta["variant"] == assignment["primary_variant"]
            if "operating_primary" in meta and meta["operating_primary"] is not primary:
                raise BuildError("primary_preallocation_mismatch")
            meta["operating_primary"] = primary
            if row["split"] == "calibration":
                role = assignment["calibration_role"]
                if "calibration_role" in meta and meta["calibration_role"] != role:
                    raise BuildError("role_preallocation_mismatch")
                meta["calibration_role"] = role
            elif "calibration_role" in meta:
                raise BuildError("role_outside_calibration")
            rows.append(row)
            if row["split"] not in source["splits"]:
                source["splits"].append(row["split"])
        source["splits"].sort()
    if not rows:
        raise BuildError("empty_author_shards")
    checks = _source_leakage_checks(rows)
    root_rows = defaultdict(list)
    for row in rows:
        root_rows[row["phenomena"]["root_id"]].append(row)
    accepted, quarantine, quarantined_roots = [], [], {}
    for root, members in sorted(root_rows.items()):
        assignment = assignments[root]
        approved = [row for row in members
                    if row["phenomena"].get("adjudication_status", "accepted") == "accepted"]
        reason = None
        if assignment["split"] == "calibration":
            primary = [row for row in approved if row["phenomena"]["operating_primary"]]
            if len(primary) != 1:
                reason = "missing_accepted_operating_primary"
            elif (primary[0]["label"] in ALLOWING_CLASSES) != assignment["operating_allow"]:
                reason = "primary_label_preallocation_mismatch"
            elif primary[0]["label"] == "contextual" and not F.request_has_trigger(primary[0]["request"]):
                reason = "contextual_primary_missing_trigger"
        if reason:
            quarantined_roots[root] = reason
        for row in members:
            if reason or row not in approved:
                row["phenomena"]["quarantine_reason"] = reason or "unadjudicated"
                quarantine.append(row)
            else:
                accepted.append(row)
    accepted.sort(key=lambda row: int(row["id"]))
    quarantine.sort(key=lambda row: int(row["id"]))
    for row in accepted + quarantine:
        S.validate_row(row)
    training = [row for row in accepted if row["split"] in TRAINABLE_SPLITS]
    sealed = [row for row in accepted if row["split"] in SEALED_SPLITS]
    report = {"human_gold_available": False, "author_labels_are_proxy": True,
              "independent_teacher_agreement_measured": False, "semantics_verified": False,
              "rows_contract_replayed": len(rows), "checks": checks, "sources": sources,
              "counts": {split: _source_counts([row for row in accepted if row["split"] == split])
                         for split in TRAINABLE_SPLITS + SEALED_SPLITS},
              "quarantine": {"counts": _source_counts(quarantine), "roots": quarantined_roots,
                             "reasons": dict(Counter(row["phenomena"]["quarantine_reason"]
                                                     for row in quarantine))},
              "missing_roots": sorted(set(assignments) - set(root_rows)),
              "calibration": _calibration_allocation_counts(accepted, assignments),
              "near_neighbor_candidates": _source_near_candidates(rows)}
    return {"training_rows": training, "sealed_rows": sealed, "quarantine_rows": quarantine,
            "report": report, "protocol_sha256": protocol["sha256"], "sources": sources,
            "assignments": assignments}


def write_author_outputs(result: dict, out_dir: object, sealed_dir: object) -> dict:
    """训练者只收到train/dev/calibration目录；完整审计和隔离行仅放封存目录。"""
    out, sealed_out = Path(out_dir).resolve(), Path(sealed_dir).resolve()
    if out == sealed_out or out in sealed_out.parents or sealed_out in out.parents:
        raise BuildError("sealed_output_overlap")
    if out.exists() or sealed_out.exists():
        raise BuildError("output_exists")
    rows, sealed, quarantine = (result[key] for key in (
        "training_rows", "sealed_rows", "quarantine_rows"))
    training_payloads = {f"{split}.jsonl": _jsonl_bytes([row for row in rows if row["split"] == split])
                         for split in TRAINABLE_SPLITS}
    training_payloads["pairs.jsonl"] = _jsonl_bytes(rows)
    sealed_payloads = {f"{split}.jsonl": _jsonl_bytes([row for row in sealed if row["split"] == split])
                       for split in SEALED_SPLITS}
    sealed_payloads["quarantine.jsonl"] = _jsonl_bytes(quarantine)
    for payloads, allowed_rows, visibility in (
            (training_payloads, rows, "trainable-only"), (sealed_payloads, sealed, "sealed-test-only")):
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())}
        manifest = {"format": "memory-relations-author-data-v3", "wire_format": WIRE_FORMAT,
                    "visibility": visibility, "human_gold_available": False,
                    "source_protocol_sha256": result["protocol_sha256"], "files": hashes,
                    "counts": _source_counts(allowed_rows)}
        if visibility == "trainable-only":
            assignments = {root: entry for root, entry in result["assignments"].items()
                           if entry["split"] in TRAINABLE_SPLITS}
            audit = {"human_gold_available": False, "semantics_verified": False,
                     "counts": {split: result["report"]["counts"][split] for split in TRAINABLE_SPLITS},
                     "calibration": result["report"]["calibration"],
                     "missing_roots": [root for root in result["report"]["missing_roots"]
                                       if root in assignments],
                     "checks": _source_leakage_checks(rows),
                     "near_neighbor_candidates": _source_near_candidates(rows),
                     "sources": [source for source in result["sources"]
                                 if set(source["splits"]) <= set(TRAINABLE_SPLITS)]}
        else:
            audit = result["report"]
        payloads["audit.json"] = _json_bytes(audit)
        manifest["files"]["audit.json"] = hashlib.sha256(payloads["audit.json"]).hexdigest()
        payloads["manifest.json"] = _json_bytes(manifest)
    _write_new_directory(sealed_out, sealed_payloads)
    _write_new_directory(out, training_payloads)
    return {"rows_total": len(rows), "pairs_total": len(rows),
            "counts": {split: result["report"]["counts"][split] for split in TRAINABLE_SPLITS},
            "hashes": json.loads(training_payloads["manifest.json"])["files"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="构造/装配记忆关系研究数据（research-only，无网络）")
    parser.add_argument("--mode", choices=("legacy", "author-shards"), default="legacy")
    parser.add_argument("--out", required=True, help="新的训练数据目录，拒绝覆盖")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true", help="小规模旧构造（测试用）")
    parser.add_argument("--legacy-relabel", action="store_true", help="显式修旧干扰pair的候选限时/未决标签")
    parser.add_argument("--source-protocol", default=str(SOURCE_PROTOCOL_PATH))
    parser.add_argument("--shard", action="append", default=[], help="作者canonical pair JSONL，可重复")
    parser.add_argument("--sealed-out", help="与训练目录互不包含的新的封存目录")
    args = parser.parse_args(argv)
    try:
        if args.mode == "author-shards":
            if not args.shard or not args.sealed_out or args.quick or args.legacy_relabel:
                parser.error("author-shards须提供--shard和--sealed-out，不能混用旧构造选项")
            result = build_author_shards(args.shard, args.source_protocol)
            summary = write_author_outputs(result, args.out, args.sealed_out)
        else:
            if args.seed is None or args.shard or args.sealed_out:
                parser.error("legacy须提供--seed，不能混用作者shard选项")
            result = build_corpus(args.seed, PLAN_QUICK if args.quick else PLAN,
                                  repair_pair_labels=args.legacy_relabel)
            summary = write_outputs(result, args.out)
    except (BuildError, S.RowError) as exc:
        print(json.dumps({"error": exc.code}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), **summary}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
