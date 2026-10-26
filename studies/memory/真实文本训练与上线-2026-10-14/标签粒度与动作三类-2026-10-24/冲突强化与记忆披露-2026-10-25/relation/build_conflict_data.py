"""冲突识别线的数据构造、作者接口、装配与泄漏审计。

CLI
---
    python build_conflict_data.py                         # 默认 selftest（不访问任何模型）
    python build_conflict_data.py selftest
    python build_conflict_data.py specs [--out data/specs/specs.jsonl]
    python build_conflict_data.py author --split train --model <name> --channel ollama|file [--out DIR]
                                        [--limit N] [--budget-calls N] [--from-dir DIR] [--resume]
    python build_conflict_data.py assemble [--fixture | --authors DIR] [--out DIR]
    python build_conflict_data.py audit [--processed DIR] [--mode final|fixture] [--with-tokenizer]
    python build_conflict_data.py routes

纪律
----
* 本次不调用真实 LLM 作者：``author`` 必须显式给 ``--model`` 与 ``--channel``；selftest 只用
  ``render_fixture`` 机械句，装配产物写到 ``_selftest/`` 并在 manifest 标 ``fixture: true``。
* 训练/评分路由拒绝 fixture 数据（见 ``conflict_common.route_rows``）。
* 标签来自 ``conflict_semantics.classify``，不来自作者；assembly 的每条行都重算一次并比对。
* 本文件只写 ``relation/`` 下的 data/audits/_selftest，不碰 repo、旧实验或生产。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import conflict_common as K
import conflict_semantics as S
import conflict_diagnostics as DIAG

SPEC_SCHEMA = "conflict-spec-v2.2"
AUTHOR_SCHEMA = "conflict-author-v2.2-r2"
AUTHOR_PROMPT_VERSION = "conflict-author-prompt-v2.2-r2p1"
MANIFEST_SCHEMA = "conflict-dataset-v2.2-r2"
AUDIT_SCHEMA = "conflict-audit-v2.2-r2"
POLICY_IDENTITY = {
    "control_policy_version": K.CONTROL_POLICY_VERSION,
    "control_policy_sha256": K.sha256_json(K.CONTROL_POLICY),
    "protocol_sha256": K.sha256_json(K.PROTOCOL),
    "prompt_version": AUTHOR_PROMPT_VERSION,
}
SIDES_PER_FAMILY = 2 * K.NATURAL_PER_FAMILY + 2 * K.DYADS_PER_FAMILY


                                                                                        
                  
                                                                                        

def _gate_block(f: dict, setting: dict, *, exact: Sequence[str] = ()) -> dict:
    spec = S.expected_tokens(f, setting)
    return {"numbers": spec["numbers"], "polarity_classes": [list(group) for group in spec["polarity_classes"]],
            "scope_tokens": spec["scope_tokens"],
            "measure_kind": spec["measure_kind"],
            "exception_item": spec["exception_item"], "window_tokens": spec["window_tokens"],
            "referent": spec["referent"], "object_phrase": spec["object_phrase"],
            "exact_tokens": list(exact)}


def _item(family_id: str, *, pair_id: str, kind: str, recipe: str, label: str,
          old: dict, new: dict, setting: dict, position: int,
          dyad: "dict | None" = None, derived_from: "str | None" = None,
          change: "dict | None" = None, exact_tokens: Sequence[str] = (),
          matched_window_id: "str | None" = None, matched_side: "str | None" = None,
          surface: "dict | None" = None) -> dict:
    pool = (K.POOL_COUNTERFACTUAL if kind in ("dyad_base", "dyad_flip")
            else K.POOL_CONTROL if kind == "control" else K.POOL_SHARED)
    item = {
        "pair_id": pair_id, "kind": kind, "recipe": recipe, "pool": pool,
        "expected_label": label, "expected_coarse": K.FINE_TO_COARSE[label],
        "position": position,
        "old": old, "new": new,
        "semantic": {"old": S.describe(old, setting), "new": S.describe(new, setting)},
        "gate": {"old": _gate_block(old, setting), "new": _gate_block(new, setting, exact=exact_tokens)},
        "dyad": dyad, "derived_from": derived_from, "change": change,
        "matched_window_id": matched_window_id, "matched_side": matched_side,
        "surface": ({"old": S.surface_for(recipe, "old"),
                     "new": S.surface_for(recipe, "new")}
                    if surface is None or "kind" in surface else surface),
    }
    return item


class SemanticAllocator:
    """v2.2：只保证**跨 split** side/pair 语义 twin=0；同 split 重复允许并另报。"""

    def __init__(self) -> None:
        self.pairs: dict = {}
        self.sides: dict = {}
        self.counters: dict = {}
        self.cf_pairs: set = set()
        self.control_pairs: set = set()

    def _cell(self, setting: dict, recipe: str, n: int) -> dict:
        scopes = setting["scopes"]
        lo_cap = max(setting["qmax"] - 5, 0)
        m = n + 131 * sum(ord(ch) for ch in recipe)
        scope = scopes[m % len(scopes)]
        lo = min(setting["lo"] + (m // len(scopes)) % 8, lo_cap)
        items = setting["items"]
        item = items[(m // 3) % len(items)]
        alt = items[(items.index(item) + 1) % len(items)]
        third = items[(items.index(item) + 2) % len(items)]
        subsets = [(item,), (alt,), (third,), (item, alt), (item, third), (alt, third)]
        items_scope = subsets[(m // 6) % len(subsets)]
        first, second = setting["years"]
        near_year, far_year = (first, second) if (m // 9) % 2 == 0 else (second, first)
        referent = setting["lists"][0] if (m // 18) % 2 == 0 else None
        return {"lo": max(lo, 0), "near_scope": scope,
                "far_scope": scopes[(scopes.index(scope) + 1) % len(scopes)],
                "item": item, "item_alt": alt, "items_scope": items_scope,
                "items_scope_alt": next(s for s in subsets if s != items_scope),
                "near_year": near_year, "far_year": far_year,
                "referent": referent,
                "list_near": setting["lists"][0], "list_far": setting["lists"][1]}

    def _conflict(self, split: str, kind: str, key) -> bool:
        return any(key in registry for other, registry in getattr(self, kind).items() if other != split)

    def allocate(self, setting: dict, recipe: str, builder, *, split: str,
                 allow_old: "str | None" = None):
        key = (setting["id"], recipe, split)
        for _ in range(240):
            n = self.counters.get(key, 0)
            self.counters[key] = n + 1
            cell = self._cell(setting, recipe, n)
            old, new = builder(cell)
            old_key, new_key = DIAG.factor_key(old, setting), DIAG.factor_key(new, setting)
            if old_key != allow_old and self._conflict(split, "sides", old_key):
                continue
            if self._conflict(split, "sides", new_key):
                continue
            pair = (old_key, new_key)
            if self._conflict(split, "pairs", pair):
                continue
            self.sides.setdefault(split, set()).update({old_key, new_key})
            self.pairs.setdefault(split, set()).add(pair)
            return cell, old, new
        raise AssertionError(f"cross-split semantic space exhausted for {setting['id']}/{recipe}/{split}")

    def register(self, old: dict, new: dict, *, split: str, setting: dict) -> bool:
        old_key, new_key = DIAG.factor_key(old, setting), DIAG.factor_key(new, setting)
        pair = (old_key, new_key)
        if self._conflict(split, "sides", old_key) or self._conflict(split, "sides", new_key) \
                or self._conflict(split, "pairs", pair):
            return False
        self.sides.setdefault(split, set()).update({old_key, new_key})
        self.pairs.setdefault(split, set()).add(pair)
        return True


def _leaf_diff(left: dict, right: dict, path: str = "") -> int:
    count = 0
    for key in set(left) | set(right):
        if key.startswith("_"):
            continue
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            count += _leaf_diff(a, b, f"{path}.{key}")
        elif a != b:
            count += 1
    return count


def control_spec_contrasts(specs, *, include_accepted: bool = False) -> list[dict]:
    forbidden = {DIAG.item_keys(spec, item, meaning=True)["pair"]
                 for spec in specs for item in spec["items"]
                 if item["pool"] in (K.POOL_SHARED, K.POOL_COUNTERFACTUAL)}
    comparisons = []
    for spec in specs:
        by_id = {item["pair_id"]: item for item in spec["items"]}
        setting = S.SETTINGS_BY_ID[spec["setting"]]
        for item in spec["items"]:
            if item["kind"] != "control":
                continue
            shadow_id = dyad_side_id(item["matched_window_id"], item["matched_side"])
            shadow = by_id[shadow_id]
            differences = _leaf_diff(item["new"], shadow["new"])
            effective = S.effective_factor_differences(item["new"], shadow["new"], setting)
            domain_problems = {name: S.factor_problems(factor, setting) for name, factor in (
                ("old", item["old"]), ("new", item["new"]),
                ("shadow_old", shadow["old"]), ("shadow_new", shadow["new"]))}
            valid_domains = not any(domain_problems.values())
            pair = DIAG.item_keys(spec, item, meaning=True)["pair"]
            same_pair = pair == DIAG.item_keys(spec, shadow, meaning=True)["pair"]
            twin = pair in forbidden
            control_label = S.classify_with(item["old"], item["new"], setting)
            shadow_label = S.classify_with(shadow["old"], shadow["new"], setting)
            shared_old = K.sha256_json(item["old"]) == K.sha256_json(shadow["old"])
            coarse_matches = (item["expected_coarse"] == shadow["expected_coarse"]
                              == K.FINE_TO_COARSE[control_label] == K.FINE_TO_COARSE[shadow_label])
            equivalent = item["expected_coarse"] == "equivalent"
            if equivalent:
                ok = (valid_domains and coarse_matches and shared_old
                      and control_label == shadow_label == "equivalent" and same_pair and differences == 0)
            else:
                ok = valid_domains and coarse_matches and shared_old and differences >= 2 and len(effective) >= 2 and not twin
            entry = {"id": item["pair_id"], "shadow": shadow_id, "coarse": item["expected_coarse"],
                     "contract": "equivalent_shared_semantics" if equivalent else "non_equivalent_contrast",
                     "ok": bool(ok), "control_label": control_label, "shadow_label": shadow_label,
                     "fine_subtype_matches": control_label == shadow_label,
                     "shared_old_factors": shared_old, "same_matched_semantic_pair": same_pair,
                     "leaf_differences": differences, "effective_factor_changes": effective,
                     "domain_problems": domain_problems, "factor_pair_twin": twin,
                     "unavoidable_equivalent_layer": equivalent and bool(ok)}
            if include_accepted or not ok:
                comparisons.append(entry)
    return comparisons


def require_feasible_controls(specs) -> None:
    failures = control_spec_contrasts(specs)
    if failures:
        raise AuthorError(f"control matching policy {K.CONTROL_POLICY_VERSION} rejected "
                          f"{len(failures)} controls; authoring/assembly requires both explicit contracts")


def _control_news(shadow_old: dict, recipe: str, setting: dict, cell: dict):
    """由影子 old 出发构造控制 new；候选按“基础 / 加例外 / 加度量 / 换作用域”展开。"""
    base = dict(shadow_old)
    item = cell["item"]
    lo = int((base.get("measure") or {}).get("low") or cell["lo"])
    low = max(lo - 2, 0)
    variants = []
    if recipe in ("E1", "E2", "E3"):
        if recipe == "E3":
            base = dict(base, slot=setting["slot_secondary"], object=setting["object_secondary"])
        variants.append(base)
    elif recipe == "P1":
        variants.append(dict(base, exception={"kind": "except", "item": item}))
    elif recipe == "P3":
        variants.append(dict(base, scope=cell["far_scope"]))
    elif recipe == "P4":
        variants.append(dict(base, slot=setting["slot_secondary"],
                             object=setting["object_secondary"]))
    elif recipe == "P5":
        variants.append(dict(base, window={"kind": "one_off", "year": cell["far_year"]}))
    elif recipe == "P6":
        variants.append(dict(base, measure=S._measure("at_most", high=lo + 2)))
    elif recipe == "P7":
        variants.append(dict(base, referent=cell["list_far"]))
    elif recipe == "P8":
        variants.append(dict(base, measure=S._measure("at_least", lo + 1)))
    elif recipe == "P10":
        variants.append(dict(base, measure=S._measure("at_least", low)))
    elif recipe == "C1":
        variants.append(dict(base, action="prohibit" if base["action"] == "require" else "require"))
    elif recipe == "C2":
        variants.append(dict(base, measure=S._measure("at_most", high=low)))
    elif recipe == "C3":
        variants.append(dict(base, scope="全部",
                             action="prohibit" if base["action"] == "require" else "require"))
    elif recipe == "C4":
        variants.append(dict(base, window={"kind": "one_off", "year": cell["near_year"]},
                             action="prohibit" if base["action"] == "require" else "require"))
    elif recipe == "C5":
        variants.append(dict(base, referent=shadow_old.get("referent"),
                             action="prohibit" if base["action"] == "require" else "require"))
    elif recipe == "C6":
        variants.append(dict(base, slot=setting["slot_secondary"],
                             object=setting["object_secondary"],
                             action="prohibit" if base["action"] == "require" else "require"))
    elif recipe == "C7":
        variants.append(dict(base, measure=S._measure("within", 1, max(lo - 1, 1))))
    else:
        raise KeyError(recipe)
    expanded = list(variants)
    for variant in variants:
        if (variant.get("exception") or {}).get("kind") != "except" and \
                variant["action"] in ("require", "prohibit"):
            expanded.append(dict(variant, exception={"kind": "except", "item": item}))
        if (variant.get("measure") or {}).get("kind") == "none":
            expanded.append(dict(variant, measure=S._measure("within", lo, min(lo + 2, setting["qmax"]))))
        if not recipe.startswith("E"):
            expanded.append(dict(variant, scope="全部", items_scope=cell["items_scope"]))
            expanded.append(dict(variant, scope="全部", items_scope=None))
    seen, unique = set(), []
    for variant in expanded:
        key = K.sha256_json(variant)
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def build_family_spec(index: int, split: str, allocator: SemanticAllocator) -> dict:
    plan = S.family_plan(index)
    setting = S.SETTINGS_BY_ID[plan["setting"]]
    mechanism = S.MECHANISMS_BY_ID[plan["mechanism"]]
    family_id = f"cf-{index + 1:03d}"
    items: list[dict] = []
                               
    for position, recipe in enumerate(mechanism["recipes"]):
        cell, old, new = allocator.allocate(
            setting, recipe, lambda c, r=recipe: S.build_natural(r, setting, c), split=split)
        label = S.classify_with(old, new, setting)
        if label != S.NATURAL_RECIPES[recipe]["label"]:
            raise AssertionError(f"{family_id}:{recipe} label {label}")
        items.append(_item(family_id, pair_id=f"{family_id}-s{position + 1}", kind="shared",
                           recipe=recipe, label=label, old=old, new=new, setting=setting,
                           position=position,
                           surface=None))
                                       
    dyad_sides = []
    for slot_index, (recipe, base_label, flip_label) in enumerate(S.dyad_recipes_for_family(index)):
        key = (setting["id"], f"dyad:{recipe}", split)
        for _ in range(240):
            n = allocator.counters.get(key, 0)
            allocator.counters[key] = n + 1
            cell = allocator._cell(setting, recipe, n)
            built = S.build_dyad(recipe, setting, cell)
            old, base_new, flip_new = built["old"], built["base_new"], built["flip_new"]
            base_fine = S.classify_with(old, base_new, setting)
            flip_fine = S.classify_with(old, flip_new, setting)
            if K.FINE_TO_COARSE[base_fine] != base_label or K.FINE_TO_COARSE[flip_fine] != flip_label:
                continue
            keys = [DIAG.factor_key(f, setting) for f in (old, base_new, flip_new)]
            if any(allocator._conflict(split, "sides", k) for k in keys):
                continue
            pair_keys = ((keys[0], keys[1]), (keys[0], keys[2]))
            if any(allocator._conflict(split, "pairs", p) or p in allocator.control_pairs for p in pair_keys):
                continue
            allocator.register(old, base_new, split=split, setting=setting)
            allocator.register(old, flip_new, split=split, setting=setting)
            allocator.cf_pairs.update(pair_keys)
            break
        else:
            raise AssertionError(f"{family_id}:{recipe} no cross-split-free dyad")
        dyad_id = f"{family_id}-dy{slot_index + 1}"
        renders = built["renders"]
        base_id, flip_id = f"{family_id}-d{slot_index + 1}-a", f"{family_id}-d{slot_index + 1}-b"
        factor = dict(built["change"], from_render=renders["from"], to_render=renders["to"])
        items.append(_item(family_id, pair_id=base_id, kind="dyad_base", recipe=recipe,
                           label=base_fine, old=old, new=base_new, setting=setting,
                           position=6 + slot_index,
                           dyad={"dyad_id": dyad_id, "role": "nonconflict",
                                 "partner_id": flip_id, "factor": factor},
                           change=factor, exact_tokens=(renders["from"],),
                           surface=S.surface_for(recipe, "old")))
        items.append(_item(family_id, pair_id=flip_id, kind="dyad_flip", recipe=recipe,
                           label=flip_fine, old=old, new=flip_new, setting=setting,
                           position=6 + slot_index,
                           dyad={"dyad_id": dyad_id,
                                 "role": "conflict" if flip_fine == "disjoint" else "nonconflict",
                                 "partner_id": base_id, "factor": factor},
                           derived_from=base_id, change=factor, exact_tokens=(renders["to"],),
                           surface=S.surface_for(recipe, "new")))
        dyad_sides.append((dyad_id, base_label, flip_label, old, base_new, flip_new, recipe))
                                                     
    for slot_index, (dyad_id, base_label, flip_label, old, base_new, flip_new, recipe) in enumerate(dyad_sides):
        for side, label, shadow_new in (("a", base_label, base_new), ("b", flip_label, flip_new)):
            control_recipe = S.CONTROL_PLAN[(recipe, "base" if side == "a" else "flip")]
            key = (setting["id"], f"control:{control_recipe}")
            chosen = None
            for _ in range(240):
                n = allocator.counters.get(key, 0)
                allocator.counters[key] = n + 1
                cell = allocator._cell(setting, control_recipe, n)
                for candidate in _control_news(old, control_recipe, setting, cell):
                    if K.FINE_TO_COARSE[S.classify_with(old, candidate, setting)] != label:
                        continue
                    if S.factor_problems(old, setting) or S.factor_problems(candidate, setting):
                        continue
                    if label != "equivalent" and (_leaf_diff(candidate, shadow_new) < 2
                            or len(S.effective_factor_differences(candidate, shadow_new, setting)) < 2):
                        continue
                    pair_key = (DIAG.factor_key(old, setting), DIAG.factor_key(candidate, setting))
                    if label != "equivalent" and (pair_key in allocator.cf_pairs or any(
                            pair_key == (DIAG.factor_key(shared["old"], setting),
                                         DIAG.factor_key(shared["new"], setting))
                            for shared in items if shared["kind"] == "shared")):
                        continue
                    if not allocator.register(old, candidate, split=split, setting=setting):
                        continue
                    if label != "equivalent":
                        allocator.control_pairs.add(pair_key)
                    chosen = candidate
                    break
                if chosen is not None:
                    break
            if chosen is None:
                raise AssertionError(f"{family_id}:{dyad_id}-{side} no valid control")
            control_label = S.classify_with(old, chosen, setting)
            k = 2 * slot_index + (0 if side == "a" else 1)
            items.append(_item(family_id, pair_id=f"{family_id}-c{k + 1}", kind="control",
                               recipe=control_recipe, label=control_label, old=old, new=chosen,
                               setting=setting, position=9 + k,
                               matched_window_id=dyad_id, matched_side=side,
                               surface=S.surface_for(control_recipe, "old")))
    counts = Counter(item["expected_coarse"] for item in items)
    if counts != Counter({name: K.ROWS_PER_CLASS_PER_FAMILY for name in K.COARSE}) or \
            len(items) != K.ROWS_PER_FAMILY:
        raise AssertionError(f"{family_id} shape {len(items)} {dict(counts)}")
    background = (f"{setting['name']}场景：{setting['scopes'][0]}、{setting['scopes'][1]}与"
                  f"{setting['scopes'][2]}并存，各条目按自身作用域执行；"
                  f"背景只交代场景，不限制条目的对象类型。{S.reference_partition_text(setting)}")
    if len(background) > 240 or re.search(r"\d", background):
        raise AssertionError(f"{family_id} background violates rules")
    return {
        "schema": SPEC_SCHEMA, "family_id": family_id, "index": index, "split": split,
        "mechanism": plan["mechanism"], "mechanism_name": mechanism["name"],
        "setting": plan["setting"], "setting_name": setting["name"],
        "author_role": "heldout" if split == "heldout" else "train",
        "background_hint": background,
        "label_plan": {
            "shared": dict(Counter(item["expected_coarse"] for item in items if item["kind"] == "shared")),
            "dyads": dict(Counter(item["expected_coarse"] for item in items
                                  if item["kind"] in ("dyad_base", "dyad_flip"))),
            "controls": dict(Counter(item["expected_coarse"] for item in items if item["kind"] == "control")),
            "family_total": {name: counts[name] for name in K.COARSE},
        },
        "items": items,
    }


def build_specs() -> list[dict]:
    """60 family × 18 行（v2.2）；split 由固定盐洗牌决定。"""
    assignment = S.split_assignment(K.FAMILY_TOTAL)
    allocator = SemanticAllocator()
    specs = [build_family_spec(index, assignment[index], allocator) for index in range(K.FAMILY_TOTAL)]
    check_spec_set(specs)
    return specs


def check_spec_set(specs: Sequence[dict]) -> dict:
    problems: list[str] = []
    if any(spec.get("schema") != SPEC_SCHEMA for spec in specs):
        problems.append("obsolete spec schema; regenerate v2.2 before authoring")
    if len(specs) != K.FAMILY_TOTAL:
        problems.append(f"family count {len(specs)} != {K.FAMILY_TOTAL}")
    per_split = Counter(spec["split"] for spec in specs)
    if dict(per_split) != K.FAMILIES_PER_SPLIT:
        problems.append(f"split families {dict(per_split)} != {K.FAMILIES_PER_SPLIT}")
    ids = [spec["family_id"] for spec in specs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate family ids")
    for spec in specs:
        items = spec["items"]
        if len(items) != K.ROWS_PER_FAMILY:
            problems.append(f"{spec['family_id']}: {len(items)} items")
        counts = Counter(item["expected_coarse"] for item in items)
        if counts != Counter({name: K.ROWS_PER_CLASS_PER_FAMILY for name in K.COARSE}):
            problems.append(f"{spec['family_id']}: class mix {dict(counts)}")
        for kind, expected in (("shared", K.SHARED_PER_FAMILY), ("control", K.CONTROL_PER_FAMILY),
                               ("dyad_base", K.DYADS_PER_FAMILY), ("dyad_flip", K.DYADS_PER_FAMILY)):
            if sum(1 for item in items if item["kind"] == kind) != expected:
                problems.append(f"{spec['family_id']}: {kind} count")
        dyads = {}
        for item in items:
            if item["dyad"]:
                dyads.setdefault(item["dyad"]["dyad_id"], []).append(item)
        if len(dyads) != K.DYADS_PER_FAMILY:
            problems.append(f"{spec['family_id']}: dyad ids {len(dyads)}")
        for dyad_id, members in dyads.items():
            if sorted(item["kind"] for item in members) != ["dyad_base", "dyad_flip"]:
                problems.append(f"{spec['family_id']}:{dyad_id} kinds")
                continue
            base = next(item for item in members if item["kind"] == "dyad_base")
            flip = next(item for item in members if item["kind"] == "dyad_flip")
            if base["dyad"]["role"] != "nonconflict":
                problems.append(f"{spec['family_id']}:{dyad_id} base role")
            if (flip["expected_label"] == "disjoint") != (flip["dyad"]["role"] == "conflict"):
                problems.append(f"{spec['family_id']}:{dyad_id} flip role")
        controls = [item for item in items if item["kind"] == "control"]
        for control in controls:
            if control["matched_window_id"] not in dyads:
                problems.append(f"{spec['family_id']}:{control['pair_id']} matched id")
            if control["matched_side"] not in ("a", "b"):
                problems.append(f"{spec['family_id']}:{control['pair_id']} matched side")
        for item in items:
            if item["expected_coarse"] != K.FINE_TO_COARSE[item["expected_label"]]:
                problems.append(f"{spec['family_id']}:{item['pair_id']} label mapping")
            for side in ("old", "new"):
                problems.extend(f"{item['pair_id']}.{side}: {problem}" for problem in
                                S.factor_problems(item[side], S.SETTINGS_BY_ID[spec["setting"]]))
            label = S.classify_with(item["old"], item["new"], S.SETTINGS_BY_ID[spec["setting"]])
            if label != item["expected_label"]:
                problems.append(f"{spec['family_id']}:{item['pair_id']} rule drift {label}")
    if problems:
        raise AssertionError("spec set problems: " + "; ".join(problems[:10]))
    return {"families": len(specs), "splits": dict(per_split),
            "mechanisms": dict(Counter(spec["mechanism"] for spec in specs)),
            "rows": sum(len(spec["items"]) for spec in specs)}


                                                                                        
              
                                                                                        

PROMPT_HEADER = """你在为“记忆关系判断”语料写中文句对。只输出一个 JSON 对象，不要解释。

硬约束：
1. 每个 pair 的 old / new 各是一句 10-70 字的中文记忆陈述，语气自然，像用户自己保存的长期约定。
2. 必须逐字包含给定“必须出现”的词元；数字必须原样出现；不得加入其他数字。
3. 不得出现这些词：冲突、等价、蕴含、矛盾、标签、规格、训练数据、标注规则。
4. 不得提到“旧记忆/新记忆/两条/句对/数据”等元话语。
5. 背景段只交代场景与共享约定，不出现数字，不暗示哪条更强或哪条矛盾。
6. 两个句子都必须能独立看懂，不要互相指代“上一条”“前一句”。
7. old/new 每句都必须明确表达给定极性：require 用“必须/需要/应当”之一，prohibit 用“不得/禁止/不允许”之一；不得只写“默认、考虑、可以、按需”等不含明确极性的措辞。

只输出：
{"family_id": "...", "background": "...", "items": [{"pair_id": "...", "old": "...", "new": "..."}]}
"""


def build_prompt(spec: dict) -> str:
    lines = [f"作者提示版本：{AUTHOR_PROMPT_VERSION}；匹配协议：{K.CONTROL_POLICY_VERSION}；"
             f"协议SHA：{K.sha256_json(K.CONTROL_POLICY)}", PROMPT_HEADER,
             f"\nfamily_id: {spec['family_id']}",
             f"领域：{spec['setting_name']}（{spec['mechanism_name']}）",
             f"背景段要求（清单互斥句必须逐字保留）：{spec['background_hint']}",
             "\nitems（标了 a/b 的只写 a 侧；其余每条都要写；不要自行发明未列出的对照句）："]
    for item in spec["items"]:
        if item["kind"] == "dyad_flip":
            continue
        lines.append(f"- pair_id: {item['pair_id']}")
        lines.append(f"  old 语义：{item['semantic']['old']}")
        lines.append(f"  new 语义：{item['semantic']['new']}")
        for side in ("old", "new"):
            polarity = ("必须/需要/应当" if item[side]["action"] == "require"
                        else "不得/禁止/不允许")
            lines.append(f"  {side} 极性：{item[side]['action']}，一般用「{polarity}」之一明确表达；"
                         "若本侧有逐字且唯一片段，片段要求优先，禁止用上述近义词替换该片段。")
            prompt_gate = dict(item["gate"][side], items_scope=item[side].get("items_scope"))
            lines.append(f"  {side} 必须出现：{'、'.join(_required_phrases(prompt_gate))}")
        if item["expected_label"] == "equivalent":
            lines.append("  本条 old 与 new 是同一要求的两种不同说法：两句都必须完整，禁止逐字相同")
        if item["kind"] == "dyad_base":
            factor = item["dyad"]["factor"]
            lines.append(f"  最高优先级：new 必须逐字包含「{factor['from_render']}」且恰好一次，"
                         "不得同义替换或受一般极性语气建议覆盖；"
                         f"不要写「{factor['to_render']}」，以保持后续机械单点替换。")
        for side, surface in (item.get("surface") or {}).items():
            if surface and surface.get("kind") == "numeric":
                form = "在 X 到 Y 之间" if surface["form"] == "range" else "不少于 X 且不多于 Y"
                lines.append(f"  {side} 表面形式必须使用：{form}")
        if item["kind"] == "control":
            base_id = dyad_side_id(item["matched_window_id"], "a")
            lines.append(f"  old 逐字复用 {base_id} 的 old；装配直接取其已验收文本，不另写 old。"
                         "new 应另写自然措辞变体，总长度约为影子的 0.67–1.50 倍。")
            if item["expected_coarse"] == "equivalent":
                lines.append("  这是不可避免的同语义等价层：两边 new 都须等价 old，保持给定因素完全一致，"
                             "不得为了措辞变化发明语义差异；字面重复另行报告，不计独立语义新颖度。")
            else:
                lines.append("  本条必须保持给定的多因素差异，不得复制影子或同族其他条目的语义。")
    lines.append("\n除声明共享的 old 外，各 pair 的 new 应使用不同自然措辞；不得改变给定语义。")
    return "\n".join(lines)


def _required_phrases(gate: dict) -> list[str]:
    phrases = []
    for number in gate["numbers"]:
        phrases.append(str(number))
    if gate.get("measure_kind") and gate["measure_kind"] != "none":
        phrases.append({"at_least": "不少于/至少", "at_most": "不多于/至多",
                        "within": "在 X 到 Y 之间 或 不少于 X 且不多于 Y"}.get(gate["measure_kind"], ""))
    if gate["scope_tokens"]:
        phrases.append("/".join(gate["scope_tokens"][:4]))
    if gate["exception_item"]:
        phrases.append(f"除…{gate['exception_item']}…外")
    if gate["referent"]:
        phrases.append(gate["referent"])
    if gate.get("items_scope"):
        phrases.append(f"必须逐字使用「只针对」，并完整列出「{'、'.join(gate['items_scope'])}」全部事项；"
                       "不能用「仅适用于」「只对…生效」等近义措辞替代「只针对」")
    phrases.append(gate["object_phrase"])
    for token in gate["exact_tokens"]:
        phrases.append(f"（逐字且唯一，最高优先级）「{token}」必须恰好出现一次，不得按一般语气建议改写")
    return phrases


class AuthorError(RuntimeError):
    pass


def parse_author_response(text: str, spec: dict) -> dict:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise AuthorError("response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise AuthorError("response is not an object")
    if payload.get("family_id") != spec["family_id"]:
        raise AuthorError(f"family_id {payload.get('family_id')!r} != {spec['family_id']!r}")
    background = payload.get("background")
    if not isinstance(background, str):
        raise AuthorError("background missing")
    expected = {item["pair_id"]: item for item in spec["items"] if item["kind"] != "dyad_flip"}
    given = {}
    for entry in payload.get("items") or []:
        pair_id = entry.get("pair_id")
        if pair_id not in expected:
            raise AuthorError(f"unexpected pair_id {pair_id!r}")
        if pair_id in given:
            raise AuthorError(f"duplicate pair_id {pair_id!r}")
        given[pair_id] = {"old": entry.get("old"), "new": entry.get("new")}
    missing = sorted(set(expected) - set(given))
    if missing:
        raise AuthorError(f"missing pair_ids {missing}")
    return {"background": background, "items": given}


def canonical_control_old(spec: dict, payload: dict) -> dict:
    entries = {key: dict(value) for key, value in payload["items"].items()}
    for item in spec["items"]:
        if item["kind"] == "control" and item["pair_id"] in entries:
            base_id = dyad_side_id(item["matched_window_id"], "a")
            if base_id in entries:
                entries[item["pair_id"]]["old"] = entries[base_id]["old"]
    return {"background": payload["background"], "items": entries}


def validate_author_payload(spec: dict, payload: dict) -> list[str]:
    problems = []
    payload = canonical_control_old(spec, payload)
    setting = S.SETTINGS_BY_ID[spec["setting"]]
    background = payload["background"]
    if not isinstance(background, str):
        return ["background must be text"]
    if not (10 <= len(background) <= 240):
        problems.append(f"background length {len(background)}")
    if re.search(r"\d", background):
        problems.append("background contains digits")
    if S.reference_partition_text(setting) not in background:
        problems.append("background must preserve the declared disjoint reference lists")
    for token in S.FORBIDDEN_TOKENS:
        if token in background:
            problems.append(f"background forbidden token {token}")
    expected = {item["pair_id"] for item in spec["items"] if item["kind"] != "dyad_flip"}
    if set(payload["items"]) != expected:
        problems.append("author pair ids do not exactly match spec")
    for item in spec["items"]:
        if item["kind"] == "dyad_flip":
            continue
        entry = payload["items"].get(item["pair_id"])
        if not isinstance(entry, dict):
            problems.append(f"{item['pair_id']}: missing author entry")
            continue
        for side, factor in (("old", item["old"]), ("new", item["new"])):
            problems.extend(
                f"{item['pair_id']}.{side}: {problem}" for problem in
                S.check_text_gates(entry.get(side), factor, setting,
                                   exact_tokens=item["gate"][side]["exact_tokens"],
                                   surface=(item.get("surface") or {}).get(side)))
        if isinstance(entry.get("old"), str) and isinstance(entry.get("new"), str) \
                and K.normalize_text(entry["old"]) == K.normalize_text(entry["new"]):
            problems.append(f"{item['pair_id']}: old == new after normalisation")
    return problems


def derive_conflict_text(item: dict, base_new: str) -> tuple[str, list[str]]:
    factor = item["dyad"]["factor"] if item["dyad"] else item["change"]
    return S.single_substitution(base_new, factor["from_render"], factor["to_render"])


                                                                                        
                              
                                                                                        

def fixture_author(spec: dict) -> dict:
    """selftest 专用机械作者；绝不用于真实 data/processed。"""
    setting = S.SETTINGS_BY_ID[spec["setting"]]
    items = {}
    for item in spec["items"]:
        if item["kind"] == "dyad_flip":
            continue
        style = "dyad" if item["kind"] == "dyad_base" else "new"
        tag = f"第{S.cn_number(spec['index'] + 1)}条·第{S.cn_number(item['position'] + 1)}项"
        surfaces = item.get("surface") or {}
        items[item["pair_id"]] = {
            "old": S.render_fixture(item["old"], setting, style="old", tag=tag,
                                    surface=surfaces.get("old")),
            "new": S.render_fixture(item["new"], setting, style=style, tag=tag,
                                    surface=surfaces.get("new")),
        }
    return canonical_control_old(spec, {"background": spec["background_hint"], "items": items})


class OllamaChannel:
    """最小 Ollama 通道：不读密钥、只连本机；每次调用写一行 ledger。"""

    def __init__(self, model: str, *, host: str = "http://127.0.0.1:11434", timeout: int = 180,
                 num_ctx: int = 16384) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def complete(self, prompt: str, *, num_predict: int = 8000) -> dict:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "format": "json", "think": False,
                              "options": {"temperature": 0.2, "num_predict": num_predict,
                                          "num_ctx": self.num_ctx}}).encode("utf-8")
        request = urllib.request.Request(f"{self.host}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        text, reason = body.get("response"), body.get("done_reason")
        prompt_tokens, completion_tokens = body.get("prompt_eval_count"), body.get("eval_count")
        return {"text": text, "model": body.get("model", self.model), "provider": "ollama",
                "succeeded": body.get("done") is True and reason != "length"
                             and isinstance(text, str) and bool(text.strip()),
                "attempted": True, "finish_reason": reason, "error": None,
                "base_url_sha256": K.sha256_text(self.host), "thinking": False,
                "num_ctx": self.num_ctx, "max_tokens": num_predict,
                "latency_s": round(time.perf_counter() - started, 3),
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens
                          if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) else None}}


def write_author_record(directory: Path, spec: dict, payload: dict, *, model: str,
                        prompt: str, usage: dict, source: str,
                        identity: "dict | None" = None) -> dict:
    payload = canonical_control_old(spec, payload)
    record = {
        "schema": AUTHOR_SCHEMA, **POLICY_IDENTITY, "family_id": spec["family_id"], "split": spec["split"],
        "author_role": spec["author_role"], "model": model, "source": source,
        "created": K.utc_now(), "prompt_sha256": K.sha256_text(prompt),
        "spec_sha256": K.sha256_json(spec), "usage": usage, "identity": identity,
        "background": payload["background"],
        "items": [{"pair_id": key, "old": value["old"], "new": value["new"]}
                  for key, value in sorted(payload["items"].items())],
    }
    record["response_sha256"] = K.sha256_json(payload)
    target = directory / f"{spec['family_id']}.json"
    if target.is_file():
        import uuid
        history = directory / "previous" / f"{spec['family_id']}-{time.time_ns()}-{uuid.uuid4().hex}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        target.replace(history)
    K.write_json_atomic(target, record)
    return record


def load_author_records(directory: Path) -> dict:
    records = {}
    if not Path(directory).is_dir():
        return records
    for path in sorted(Path(directory).glob("*.json")):
        record = K.read_json(path)
        if record.get("schema") != AUTHOR_SCHEMA:
            raise AuthorError(f"{path}: schema {record.get('schema')!r}")
        records[record["family_id"]] = record
    return records


                                                                                        
    
                                                                                        

def dyad_side_id(dyad_id: str, side: str) -> str:
    """``cf-001-dy2`` + ``a`` → ``cf-001-d2-a``（spec 的 pair_id 约定）。"""
    family, number = dyad_id.rsplit("-dy", 1)
    return f"{family}-d{number}-{side}"


def rows_for_family(spec: dict, record: dict) -> tuple[list[dict], list[str]]:
    problems: list[str] = []
    setting = S.SETTINGS_BY_ID[spec["setting"]]
    background = record["background"]
    by_id = {entry["pair_id"]: entry for entry in record["items"]}
    pairs = {}
    for item in spec["items"]:
        if item["kind"] == "dyad_flip":
            base = by_id.get(item["derived_from"])
            if base is None:
                problems.append(f"{item['pair_id']}: missing base {item['derived_from']}")
                continue
            old_text, new_text = base["old"], derive_conflict_text(item, base["new"])[0]
        else:
            entry = by_id.get(item["pair_id"])
            if entry is None:
                problems.append(f"{item['pair_id']}: missing author text")
                continue
            old_text, new_text = entry["old"], entry["new"]
            if item["kind"] == "control":
                shadow_id = dyad_side_id(item["matched_window_id"], item["matched_side"])
                if shadow_id not in pairs:
                    problems.append(f"{item['pair_id']}: missing accepted shadow {shadow_id}")
                    continue
                old_text = pairs[shadow_id][0]["old"]["text"]
        for side, text, factor in (("old", old_text, item["old"]), ("new", new_text, item["new"])):
            problems.extend(f"{item['pair_id']}.{side}: {problem}" for problem in
                            S.check_text_gates(text, factor, setting,
                                               exact_tokens=item["gate"][side]["exact_tokens"],
                                               surface=(item.get("surface") or {}).get(side)))
        try:
            pair = K.SR.validate_semantic_pair({
                "frame": {"mode": "same_domain", "background": background},
                "old": {"kind": "memory", "text": old_text},
                "new": {"kind": "memory", "text": new_text},
            })
            visible = K.SR.semantic_pair_text(pair)
        except Exception as error:
            problems.append(f"{item['pair_id']}: semantic pair rejected: {error}")
            continue
        pairs[item["pair_id"]] = (pair, visible)
    for item in spec["items"]:
        if item["kind"] != "dyad_flip":
            continue
        base_id = item["derived_from"]
        if base_id not in pairs or item["pair_id"] not in pairs:
            continue
        base_new = pairs[base_id][0]["new"]["text"]
        derived, sub_problems = derive_conflict_text(item, base_new)
        problems.extend(f"{item['pair_id']}: {problem}" for problem in sub_problems)
        if derived != pairs[item["pair_id"]][0]["new"]["text"]:
            problems.append(f"{item['pair_id']}: derived text mismatch")
        if pairs[base_id][0]["old"]["text"] != pairs[item["pair_id"]][0]["old"]["text"]:
            problems.append(f"{item['pair_id']}: dyad sides do not share old text")
    for item in spec["items"]:
        if item["kind"] != "control":
            continue
        shadow_id = dyad_side_id(item["matched_window_id"], item["matched_side"])
        if shadow_id not in pairs or item["pair_id"] not in pairs:
            problems.append(f"{item['pair_id']}: control/shadow missing")
            continue
        control_old, control_new = K.pair_sides(pairs[item["pair_id"]][0])
        shadow_old, shadow_new = K.pair_sides(pairs[shadow_id][0])
        if control_old != shadow_old:
            problems.append(f"{item['pair_id']}: control old is not byte-identical to {shadow_id}")
        if item["expected_coarse"] != "equivalent" \
                and K.normalize_text(control_new) == K.normalize_text(shadow_new):
            problems.append(f"{item['pair_id']}: control shares normalised new with {shadow_id}")
        shadow_item = next(x for x in spec["items"] if x["pair_id"] == shadow_id)
        if item["expected_coarse"] != shadow_item["expected_coarse"]:
            problems.append(f"{item['pair_id']}: coarse {item['expected_coarse']} != shadow "
                            f"{shadow_item['expected_coarse']}")
        control_len = len(pairs[item["pair_id"]][0]["old"]["text"]) + len(pairs[item["pair_id"]][0]["new"]["text"])
        shadow_len = len(pairs[shadow_id][0]["old"]["text"]) + len(pairs[shadow_id][0]["new"]["text"])
        ratio = (control_len / shadow_len) if shadow_len else 0.0
        if not (0.67 <= ratio <= 1.50):
            problems.append(f"{item['pair_id']}: length ratio {ratio:.3f} outside [0.67, 1.50]")
    rows = []
    for item in spec["items"]:
        if item["pair_id"] not in pairs:
            continue
        pair, visible = pairs[item["pair_id"]]
        dyad = None
        if item["dyad"]:
            dyad = {"dyad_id": item["dyad"]["dyad_id"], "role": item["dyad"]["role"],
                    "partner_id": item["dyad"]["partner_id"], "factor": item["dyad"]["factor"]}
        rows.append({
            "id": item["pair_id"],
            "root_id": f"{spec['family_id']}-{item['dyad']['dyad_id'] if item['dyad'] else item['pair_id']}",
            "cluster_id": spec["family_id"], "family_id": spec["family_id"],
            "split": spec["split"], "label": item["expected_label"],
            "coarse_label": item["expected_coarse"], "label_origin": K.LABEL_ORIGIN,
            "source": K.FIXTURE_SOURCE if record["source"] == "fixture" else K.SOURCE_NEW,
            "author": record["model"], "pair": pair, "visible_text": visible,
            "phenomena": {
                "mechanism": spec["mechanism"], "setting": spec["setting"],
                "scenario": spec["family_id"], "variant": item["recipe"],
                "recipe": item["recipe"], "language": "zh",
                "expected_label": item["expected_label"],
                "author_role": spec["author_role"], "pool": item["pool"],
                "dyad": item["dyad"]["dyad_id"] if item["dyad"] else None,
                **({"matched_window_id": item["matched_window_id"],
                    "matched_side": item["matched_side"],
                    "matched_pair_id": dyad_side_id(item["matched_window_id"], item["matched_side"])}
                   if item["kind"] == "control" else {}),
            },
            "dyad": dyad,
        })
                          
    for row in rows:
        label = S.classify_with(
            _factors_from_text_row(row, spec, "old"), _factors_from_text_row(row, spec, "new"), setting)
        if label != row["label"]:
            problems.append(f"{row['id']}: re-classify drift {label} != {row['label']}")
    return rows, problems


def _factors_from_text_row(row: dict, spec: dict, side: str) -> dict:
    """行本身不携带因素；装配期用 spec 的因素复算，故这里只供 label 复核占位。"""
    for item in spec["items"]:
        if item["pair_id"] == row["id"]:
            return item[side]
    raise KeyError(row["id"])


def assemble(specs: Sequence[dict], records: dict, *, fixture: bool,
             out_dir: Path) -> dict:
    check_spec_set(specs)
    if not fixture:
        require_feasible_controls(specs)
    problems: list[str] = []
    rows_by_split: dict[str, list[dict]] = {split: [] for split in K.SPLITS}
    authors_by_split: dict[str, set] = {split: set() for split in K.SPLITS}
    for spec in specs:
        record = records.get(spec["family_id"])
        if record is None:
            problems.append(f"{spec['family_id']}: missing author record")
            continue
        if record.get("split") != spec["split"]:
            problems.append(f"{spec['family_id']}: record split {record.get('split')!r}")
        expected_source = "fixture" if fixture else "author"
        if not fixture and (record.get("spec_sha256") != K.sha256_json(spec)
                            or record.get("prompt_sha256") != K.sha256_text(build_prompt(spec))
                            or record.get("model") not in K.AUTHOR_PLAN[spec["split"]]
                            or any(record.get(key) != value for key, value in POLICY_IDENTITY.items())):
            problems.append(f"{spec['family_id']}: author spec/prompt/model provenance mismatch")
        if record.get("source") != expected_source:
            problems.append(f"{spec['family_id']}: record source {record.get('source')!r} "
                            f"!= {expected_source!r}")
        payload_problems = validate_author_payload(spec, {"background": record["background"],
                                                          "items": {entry["pair_id"]: entry
                                                                    for entry in record["items"]}})
        problems.extend(f"{spec['family_id']}: {problem}" for problem in payload_problems)
        rows, row_problems = rows_for_family(spec, record)
        problems.extend(f"{spec['family_id']}: {problem}" for problem in row_problems)
        rows_by_split[spec["split"]].extend(rows)
        authors_by_split[spec["split"]].add(record["model"])
    if problems:
        raise AuthorError("assembly problems:\n" + "\n".join(problems[:40]))
    for split in K.SPLITS:
        rows_by_split[split].sort(key=lambda row: row["id"])
        K.write_jsonl_atomic(out_dir / f"{split}.jsonl", rows_by_split[split])
    manifest = {
        "schema": MANIFEST_SCHEMA, **POLICY_IDENTITY, "created": K.utc_now(), "fixture": bool(fixture),
        "spec_sha256": K.sha256_json(specs),
        "spec_count": len(specs),
        "authors": {split: sorted(authors_by_split[split]) for split in K.SPLITS},
        "files": {split: {"path": f"{split}.jsonl",
                          "sha256": K.sha256_file(out_dir / f"{split}.jsonl"),
                          "rows": len(rows_by_split[split]),
                          "families": len({row["family_id"] for row in rows_by_split[split]}),
                          "coarse": K.family_balance(rows_by_split[split]),
                          "pools": {pool: sum(1 for row in rows_by_split[split]
                                              if K.row_pool(row) == pool)
                                    for pool in K.POOLS}}
                  for split in K.SPLITS},
        "final_test_route": "new-heldout",
        "author_plan": {key: list(value) for key, value in K.AUTHOR_PLAN.items()},
        "author_policy": {
            "train_plus_dev": sorted(authors_by_split["train"] | authors_by_split["dev"]),
            "heldout": sorted(authors_by_split["heldout"]),
            "rule": "heldout 作者不得出现在 train/dev；旧 150 与旧 heldout 只作锁后回归",
        },
        "protocol": K.PROTOCOL,
        "label_rule": "conflict_semantics.classify（因素决定，作者不可见标签）",
    }
    K.write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


                                                                                        
    
                                                                                        

class Checker:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def check(self, name: str, ok: bool, **extra) -> bool:
        self.checks[name] = {"ok": bool(ok), **extra}
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(entry["ok"] for entry in self.checks.values())


def _shingles(text: str, size: int = 4) -> set:
    normalized = K.normalize_text(text)
    if len(normalized) < size:
        return {normalized}
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def near_duplicates(rows: Sequence[dict], *, threshold: float = 0.85,
                    cross_threshold: float = 0.75) -> dict:
    """只对 old+new 文本做 Jaccard；排除 background 与 JSON 包装。"""
    def _text(row: dict) -> str:
        return " ".join(K.pair_sides(row["pair"]))
    shingles = {row["id"]: _shingles(_text(row)) for row in rows}
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    within = []
    across = []
    ordered = sorted(rows, key=lambda row: row["id"])
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left.get("dyad") and right.get("dyad") and \
                    left["dyad"].get("dyad_id") == right["dyad"].get("dyad_id"):
                continue                              
            score = _jaccard(shingles[left["id"]], shingles[right["id"]])
            if left["split"] == right["split"]:
                if score >= threshold:
                    within.append({"left": left["id"], "right": right["id"], "jaccard": round(score, 4)})
            elif score >= cross_threshold:
                across.append({"left": left["id"], "right": right["id"], "jaccard": round(score, 4),
                               "splits": [left["split"], right["split"]]})
    return {"within": within[:40], "across": across[:40],
            "within_count": len(within), "across_count": len(across),
            "thresholds": {"within": threshold, "across": cross_threshold}}


def audit_dataset(*, processed_dir: Path, manifest_path: Path, specs: Sequence[dict],
                  mode: str = "final", tokenizer=None) -> dict:
    checker = Checker()
    rows_by_split = {split: K.read_jsonl(processed_dir / f"{split}.jsonl") for split in K.SPLITS}
    manifest = K.read_json(manifest_path) if manifest_path.is_file() else {}
    all_rows = [row for split in K.SPLITS for row in rows_by_split[split]]
    spec_by_item = {item["pair_id"]: (spec, item) for spec in specs for item in spec["items"]}

                     
    row_problems = []
    for split in K.SPLITS:
        for row in rows_by_split[split]:
            row_problems.extend(f"{row.get('id')}: {problem}" for problem in K.validate_new_row(row, split))
    checker.check("row_contract", not row_problems, problems=row_problems[:20])

    serialize_problems = []
    for row in all_rows:
        try:
            visible = K.SR.semantic_pair_text(K.SR.validate_semantic_pair(row["pair"]))
        except Exception as error:
            serialize_problems.append(f"{row['id']}: {error}")
            continue
        if visible != row["visible_text"]:
            serialize_problems.append(f"{row['id']}: visible_text != serializer output")
        old, new = K.pair_sides(row["pair"])
        normalized = K.normalize_text(row["visible_text"])
        if K.normalize_text(old) not in normalized or K.normalize_text(new) not in normalized:
            serialize_problems.append(f"{row['id']}: visible_text does not contain both sides")
        if len(row["visible_text"]) > K.SR.MAX_PAIR_CHARS:
            serialize_problems.append(f"{row['id']}: visible_text exceeds MAX_PAIR_CHARS")
    checker.check("serialization", not serialize_problems, problems=serialize_problems[:20])

                                                                          
    spec_problems = []
    for split in K.SPLITS:
        rows = rows_by_split[split]
        families = defaultdict(list)
        for row in rows:
            families[row["family_id"]].append(row)
        for family_id, members in families.items():
            counts = K.family_balance(members)
            pools = Counter(K.row_pool(row) for row in members)
            if len(members) != K.ROWS_PER_FAMILY:
                spec_problems.append(f"{family_id}: {len(members)} rows")
            if counts != Counter({name: K.ROWS_PER_CLASS_PER_FAMILY for name in K.COARSE}):
                spec_problems.append(f"{family_id}: class mix {dict(counts)}")
            expected_pools = Counter({K.POOL_SHARED: K.SHARED_PER_FAMILY,
                                      K.POOL_CONTROL: K.CONTROL_PER_FAMILY,
                                      K.POOL_COUNTERFACTUAL: K.DYAD_ROWS_PER_FAMILY})
            if pools != expected_pools:
                spec_problems.append(f"{family_id}: pools {dict(pools)}")
            dyads = {row["dyad"]["dyad_id"] for row in members if row["dyad"]}
            if len(dyads) != K.DYADS_PER_FAMILY:
                spec_problems.append(f"{family_id}: dyads {len(dyads)}")
        for row in rows:
            if row["id"] not in spec_by_item:
                spec_problems.append(f"{row['id']}: not in specs")
                continue
            spec, item = spec_by_item[row["id"]]
            if row["family_id"] != spec["family_id"] or row["split"] != spec["split"]:
                spec_problems.append(f"{row['id']}: family/split mismatch")
            if row["label"] != item["expected_label"]:
                spec_problems.append(f"{row['id']}: label {row['label']} != spec {item['expected_label']}")
            if row["coarse_label"] != K.FINE_TO_COARSE[row["label"]]:
                spec_problems.append(f"{row['id']}: coarse mapping")
            if K.row_pool(row) != item["pool"]:
                spec_problems.append(f"{row['id']}: pool {K.row_pool(row)} != spec {item['pool']}")
            setting = S.SETTINGS_BY_ID[spec["setting"]]
            if S.reference_partition_text(setting) not in row["pair"]["frame"].get("background", ""):
                spec_problems.append(f"{row['id']}: missing disjoint reference declaration")
            for field, value in (("setting", spec["setting"]), ("mechanism", spec["mechanism"]),
                                 ("recipe", item["recipe"]), ("variant", item["recipe"])):
                if row["phenomena"].get(field) != value:
                    spec_problems.append(f"{row['id']}: phenomena.{field} mismatch")
            for side, text in zip(("old", "new"), K.pair_sides(row["pair"])):
                spec_problems.extend(f"{row['id']}.{side}: {problem}" for problem in
                                     S.check_text_gates(text, item[side], setting,
                                                        exact_tokens=item["gate"][side]["exact_tokens"],
                                                        surface=(item.get("surface") or {}).get(side)))
            recomputed = S.classify_with(item["old"], item["new"], setting)
            if recomputed != row["label"]:
                spec_problems.append(f"{row['id']}: re-classify {recomputed} != {row['label']}")
    if {row["id"] for row in all_rows} != set(spec_by_item):
        spec_problems.append("row ids do not exactly match spec ids")
    checker.check("spec_alignment", not spec_problems, problems=spec_problems[:20])
    checker.check("manifest_integrity", manifest.get("spec_sha256") == K.sha256_json(specs)
                  and manifest.get("schema") == MANIFEST_SCHEMA
                  and all(manifest.get(key) == value for key, value in POLICY_IDENTITY.items())
                  and manifest.get("protocol") == K.PROTOCOL
                  and all((manifest.get("files", {}).get(split) or {}).get("sha256") ==
                          K.sha256_file(processed_dir / f"{split}.jsonl") for split in K.SPLITS),
                  spec_sha256=K.sha256_json(specs))

    expected_splits = Counter(spec["split"] for spec in specs)
    actual_splits = {split: len({row["family_id"] for row in rows_by_split[split]}) for split in K.SPLITS}
    checker.check("family_splits", actual_splits == dict(expected_splits),
                  actual=actual_splits, expected=dict(expected_splits))

                                   
    families_by_split = {split: {row["family_id"] for row in rows_by_split[split]} for split in K.SPLITS}
    clusters_by_split = {split: {row["cluster_id"] for row in rows_by_split[split]} for split in K.SPLITS}
    roots_by_split = {split: {row["root_id"] for row in rows_by_split[split]} for split in K.SPLITS}
    overlaps = {}
    for left, right in (("train", "dev"), ("train", "heldout"), ("dev", "heldout")):
        overlaps[f"{left}|{right}"] = {
            "family": sorted(families_by_split[left] & families_by_split[right])[:5],
            "cluster": sorted(clusters_by_split[left] & clusters_by_split[right])[:5],
            "root": sorted(roots_by_split[left] & roots_by_split[right])[:5],
        }
    checker.check("split_isolation", all(not any(value.values()) for value in overlaps.values()),
                  overlaps=overlaps)
    duplicate_ids = [item for item, count in Counter(row["id"] for row in all_rows).items() if count > 1]
    checker.check("id_unique", not duplicate_ids, duplicates=duplicate_ids[:10])

                            
    train_authors = {row["author"] for row in rows_by_split["train"]} | \
                    {row["author"] for row in rows_by_split["dev"]}
    heldout_authors = {row["author"] for row in rows_by_split["heldout"]}
    shared_authors = sorted(train_authors & heldout_authors)
    declared = manifest.get("author_plan") or {}
    checker.check("author_plan_declared",
                  {key: sorted(value) for key, value in declared.items()} ==
                  {key: sorted(value) for key, value in K.AUTHOR_PLAN.items()},
                  declared=declared, expected={k: list(v) for k, v in K.AUTHOR_PLAN.items()})
    if mode == "final":
        trainer = set(K.AUTHOR_PLAN["train"]) | set(K.AUTHOR_PLAN["dev"])
        held = set(K.AUTHOR_PLAN["heldout"])
        checker.check("author_isolation",
                      not shared_authors and train_authors <= trainer and heldout_authors == held,
                      train=sorted(train_authors), heldout=sorted(heldout_authors),
                      shared=shared_authors, expected_train=sorted(trainer), expected_heldout=sorted(held))
    else:
        checker.check("author_isolation", manifest.get("fixture") is True,
                      applicable=False, note="fixture 不要求作者隔离；manifest.fixture 必须为 true",
                      shared=shared_authors)

                                                    
    def cross_split(key_fn) -> dict:
        groups = defaultdict(list)
        for row in all_rows:
            groups[key_fn(row)].append(row)
        return {key: members for key, members in groups.items()
                if len({row["split"] for row in members}) > 1}

    exact_visible = cross_split(lambda row: row["visible_text"])
    normalized_visible = cross_split(lambda row: K.normalize_text(row["visible_text"]))
    content_cross = cross_split(K.content_digest)
    pair_cross = cross_split(K.pair_digest)
    checker.check("cross_split_visible_exact", not exact_visible,
                  groups=[[row["id"] for row in members] for members in list(exact_visible.values())[:5]])
    checker.check("cross_split_visible_normalized", not normalized_visible,
                  groups=[[row["id"] for row in members] for members in list(normalized_visible.values())[:5]])
    checker.check("cross_split_content_digest", not content_cross,
                  groups=[[row["id"] for row in members] for members in list(content_cross.values())[:5]])
    checker.check("cross_split_pair_digest", not pair_cross,
                  groups=[[row["id"] for row in members] for members in list(pair_cross.values())[:5]])

                                                                  
    pair_labels, pair_members = defaultdict(set), defaultdict(list)
    new_labels, new_members = defaultdict(set), defaultdict(list)
    for row in all_rows:
        pair_labels[K.content_digest(row)].add(row["label"])
        pair_members[K.content_digest(row)].append(row["id"])
        new_labels[K.new_text_digest(row)].add(row["label"])
        new_members[K.new_text_digest(row)].append(row["id"])
    same_pair_diff = {key: sorted(labels) for key, labels in pair_labels.items() if len(labels) > 1}
    same_new_diff = {key: sorted(labels) for key, labels in new_labels.items() if len(labels) > 1}
    checker.check("same_pair_different_label", not same_pair_diff,
                  groups=[pair_members[key][:4] for key in list(same_pair_diff)[:5]])
    checker.check("same_new_different_label", not same_new_diff,
                  groups=[new_members[key][:4] for key in list(same_new_diff)[:5]])

                                            
    dyad_problems = []
    dyad_groups = defaultdict(list)
    for row in all_rows:
        if row["dyad"]:
            dyad_groups[row["dyad"]["dyad_id"]].append(row)
    for dyad_id, members in dyad_groups.items():
        if len(members) != 2:
            dyad_problems.append(f"{dyad_id}: {len(members)} rows")
            continue
        conflict = [row for row in members if row["dyad"]["role"] == "conflict"]
        nonconflict = [row for row in members if row["dyad"]["role"] == "nonconflict"]
        labels = {row["coarse_label"] for row in members}
        if labels == {"conflict", "compatible"} or labels == {"conflict", "equivalent"}:
            if len(conflict) != 1 or len(nonconflict) != 1:
                dyad_problems.append(f"{dyad_id}: conflict dyad roles")
        elif labels == {"compatible", "equivalent"}:
            if conflict or len(nonconflict) != 2:
                dyad_problems.append(f"{dyad_id}: E↔P dyad must have two nonconflict sides")
        else:
            dyad_problems.append(f"{dyad_id}: labels {labels}")
            continue
        base = conflict[0] if conflict else next(row for row in members if row["label"] != "equivalent")
        flip = next(row for row in members if row["id"] != base["id"])
        if base["split"] != flip["split"]:
            dyad_problems.append(f"{dyad_id}: split mismatch")
        if K.pair_sides(base["pair"])[0] != K.pair_sides(flip["pair"])[0]:
            dyad_problems.append(f"{dyad_id}: old text differs")
        factor = base["dyad"]["factor"]
        derived, problems = S.single_substitution(K.pair_sides(flip["pair"])[1],
                                                  factor["from_render"], factor["to_render"])
        dyad_problems.extend(f"{dyad_id}: {problem}" for problem in problems)
        if derived != K.pair_sides(base["pair"])[1]:
            dyad_problems.append(f"{dyad_id}: flipped new is not the single substitution")
        if base["dyad"]["partner_id"] != flip["id"] or flip["dyad"]["partner_id"] != base["id"]:
            dyad_problems.append(f"{dyad_id}: partner ids inconsistent")
    checker.check("dyad_integrity", not dyad_problems, problems=dyad_problems[:20])

                                                    
    old_groups = defaultdict(list)
    for row in all_rows:
        old_groups[K.normalize_text(K.pair_sides(row["pair"])[0])].append(row)
    repeated = [members for members in old_groups.values() if len(members) > 1]
    cross_split = [sorted(row["id"] for row in members) for members in repeated
                   if len({row["split"] for row in members}) > 1]
    within_split = [sorted(row["id"] for row in members) for members in repeated
                    if len({row["split"] for row in members}) == 1]
    checker.check("shared_old_split_isolation", not cross_split, groups=cross_split,
                  within_split_group_count=len(within_split), within_split_groups=within_split,
                  policy="cross-split old overlap rejected; within-split repetition reported; dyad/control integrity checked separately")

    control_problems = []
    controls_by_dyad = defaultdict(list)
    rows_by_id = {row["id"]: row for row in all_rows}
    cf_content = {K.content_digest(row) for row in all_rows if row.get("dyad")}
    comparisons = control_spec_contrasts(specs, include_accepted=True)
    contrast_failures = [entry for entry in comparisons if not entry["ok"]]
    equivalent_layer = [entry for entry in comparisons if entry["coarse"] == "equivalent"]
    non_equivalent_layer = [entry for entry in comparisons if entry["coarse"] != "equivalent"]
    layer_counts_ok = (len(equivalent_layer) == K.CONTROL_POLICY["equivalent"]["rows"]
                       and len(non_equivalent_layer) == K.CONTROL_POLICY["non_equivalent"]["rows"])
    literal_novelty = {
        "real_author_evidence": mode == "final" and manifest.get("fixture") is False,
        "note": "wording overlap is separate from semantic novelty; fixture tags do not prove author novelty",
        "equivalent": {"rows": 0, "same_shadow_new": 0, "pair_twin_with_CF": 0},
        "non_equivalent": {"rows": 0, "same_shadow_new": 0, "pair_twin_with_CF": 0},
    }
    for row in all_rows:
        if K.row_pool(row) != K.POOL_CONTROL:
            continue
        matched = K.matched_window_id(row)
        side = (row.get("phenomena") or {}).get("matched_side")
        pair_id = (row.get("phenomena") or {}).get("matched_pair_id")
        shadow = rows_by_id.get(pair_id)
        if shadow is None or not shadow.get("dyad") or row["id"] not in spec_by_item \
                or pair_id not in spec_by_item:
            control_problems.append(f"{row['id']}: shadow/spec missing")
            continue
        if side not in ("a", "b") or pair_id != dyad_side_id(matched, side):
            control_problems.append(f"{row['id']}: matched side/pair id mismatch")
        if shadow["dyad"]["dyad_id"] != matched or shadow["family_id"] != row["family_id"] \
                or shadow["split"] != row["split"]:
            control_problems.append(f"{row['id']}: matched window/family/split mismatch")
        if row["dyad"] is not None:
            control_problems.append(f"{row['id']}: control must not carry dyad")
        if row["coarse_label"] != shadow["coarse_label"]:
            control_problems.append(f"{row['id']}: coarse mismatch")
        if K.pair_sides(row["pair"])[0] != K.pair_sides(shadow["pair"])[0]:
            control_problems.append(f"{row['id']}: control old is not byte-identical to shadow old")
        spec, item = spec_by_item[row["id"]]
        _, shadow_item = spec_by_item[pair_id]
        expected_recipe = S.CONTROL_PLAN.get((shadow_item["recipe"],
                                             "base" if side == "a" else "flip"))
        if item["recipe"] != expected_recipe or row["phenomena"].get("recipe") != expected_recipe:
            control_problems.append(f"{row['id']}: recipe outside CONTROL_PLAN domain")
        if K.sha256_json(item["old"]) != K.sha256_json(shadow_item["old"]):
            control_problems.append(f"{row['id']}: spec old differs from shadow old")
        equivalent = row["coarse_label"] == "equivalent"
        literal = literal_novelty["equivalent" if equivalent else "non_equivalent"]
        literal["rows"] += 1
        literal["same_shadow_new"] += int(K.new_text_digest(row) == K.new_text_digest(shadow))
        text_twin = K.content_digest(row) in cf_content
        literal["pair_twin_with_CF"] += int(text_twin)
        if text_twin and not equivalent:
            control_problems.append(f"{row['id']}: normalized old+new text twin with CF")
        control_len = sum(map(len, K.pair_sides(row["pair"])))
        shadow_len = sum(map(len, K.pair_sides(shadow["pair"])))
        ratio = control_len / shadow_len if shadow_len else 0.0
        if not (0.67 <= ratio <= 1.50):
            control_problems.append(f"{row['id']}: length ratio {ratio:.3f}")
        controls_by_dyad[matched].append((side, row["id"]))
    for dyad_id in dyad_groups:
        entries = controls_by_dyad.get(dyad_id, [])
        if sorted(side for side, _ in entries) != ["a", "b"]:
            control_problems.append(f"{dyad_id}: controls {entries}")
    def fine_strata(field):
        table = {}
        for entry in non_equivalent_layer:
            spec, item = spec_by_item[entry["id"]]
            key = item["recipe"] if field == "recipe" else spec[field]
            group = table.setdefault(key, {"n": 0, "mismatched": 0})
            group["n"] += 1
            group["mismatched"] += int(not entry["fine_subtype_matches"])
        return table
    checker.check("control_matching", not control_problems and not contrast_failures and layer_counts_ok,
                  problems=control_problems[:20], operational_ok=not control_problems,
                  semantic_contracts_ok=not contrast_failures and layer_counts_ok,
                  contrast_failures=contrast_failures, contrast_failure_count=len(contrast_failures),
                  by_coarse=dict(Counter(entry["coarse"] for entry in contrast_failures)),
                  equivalent_layer={"rows": len(equivalent_layer),
                                    "accepted": sum(entry["ok"] for entry in equivalent_layer),
                                    "independent_semantic_novelty": False, "members": equivalent_layer},
                  non_equivalent_layer={"rows": len(non_equivalent_layer),
                                        "accepted": sum(entry["ok"] for entry in non_equivalent_layer),
                                        "minimum_leaf_differences": min((entry["leaf_differences"]
                                                                        for entry in non_equivalent_layer), default=None),
                                        "minimum_effective_factor_changes": min((len(entry["effective_factor_changes"])
                                                                                 for entry in non_equivalent_layer), default=None),
                                        "invalid_domains": sum(any(entry["domain_problems"].values())
                                                               for entry in non_equivalent_layer),
                                        "semantic_pair_twins": sum(entry["factor_pair_twin"]
                                                                   for entry in non_equivalent_layer),
                                        "members": non_equivalent_layer},
                  fine_composition={"fine_subtype_matched": False, "n": len(non_equivalent_layer),
                                    "mismatched": sum(not entry["fine_subtype_matches"] for entry in non_equivalent_layer),
                                    "by_split": fine_strata("split"), "by_recipe": fine_strata("recipe"),
                                    "by_setting": fine_strata("setting")},
                  literal_novelty=literal_novelty, policy=K.CONTROL_POLICY)
    twins = DIAG.semantic_twins(specs)
    meaning_twins = DIAG.semantic_twins(specs, meaning=True)
    checker.check("cross_split_factor_semantic_twins", twins["cross_ok"], detail=twins["cross"])
    checker.check("cross_split_classifier_meaning_twins", meaning_twins["cross_ok"],
                  detail=meaning_twins["cross"])
    checker.check("setting_split_coverage", all(
        block["families"] == (2 if split != "dev" else 1)
        for split, strata in twins["strata"].items() for block in strata["setting"].values()),
        strata=twins["strata"])

                                                         
    near = near_duplicates(all_rows)
    checker.check("near_duplicate", True, applicable=False,
                  policy="v2.2 预注册裁决：4-gram 近重复只报告，不自动阻塞；exact/normalized/pair-digest 仍硬失败",
                  within=near["within"][:5], across=near["across"][:5],
                  counts={"within": near["within_count"], "across": near["across_count"]})

                                                      
    parent_rows, parent_info = [], {}
    for split in ("train", "dev", "calibration", "heldout"):
        path = K.PARENT_PROCESSED / f"{split}.jsonl"
        if path.is_file():
            rows = K.read_jsonl(path)
            parent_info[split] = len(rows)
            for row in rows:
                row = dict(row)
                row["_parent_split"] = split
                parent_rows.append(row)
    parent_pairs = defaultdict(list)
    parent_visible = defaultdict(list)
    for row in parent_rows:
        parent_pairs[K.pair_digest(row)].append(row["_parent_split"])
        parent_visible[K.normalize_text(row["visible_text"])].append(row["_parent_split"])
    contamination = []
    for row in all_rows:
        digest = K.pair_digest(row)
        if digest in parent_pairs:
            contamination.append({"id": row["id"], "kind": "pair_digest", "parent": parent_pairs[digest][:3]})
        visible = K.normalize_text(row["visible_text"])
        if visible in parent_visible:
            contamination.append({"id": row["id"], "kind": "visible_text", "parent": parent_visible[visible][:3]})
    checker.check("parent_contamination", not contamination,
                  parent_rows=parent_info, hits=contamination[:10])

                                                             
    def route_shape(new_splits, include_dyads, legacy_splits, pools=None) -> int:
        total = 0
        for split in legacy_splits:
            total += len(K.read_jsonl(K.PARENT_PROCESSED / f"{split}.jsonl"))
        for split in new_splits:
            total += sum(1 for row in rows_by_split[split]
                         if (include_dyads or row["dyad"] is None)
                         and (pools is None or K.row_pool(row) in pools))
        return total

    natural_rows = K.SHARED_PER_FAMILY + K.CONTROL_PER_FAMILY
    dyad_rows = K.DYAD_ROWS_PER_FAMILY
    expected_shapes = {
        "r1": 1001 + K.FAMILIES_PER_SPLIT["train"] * natural_rows,
        "r2": 1001 + K.FAMILIES_PER_SPLIT["train"] * (K.SHARED_PER_FAMILY + dyad_rows),
        "r3": 1001 + K.FAMILIES_PER_SPLIT["train"] * (K.SHARED_PER_FAMILY + dyad_rows),
        "dev": 263 + K.FAMILIES_PER_SPLIT["dev"] * (natural_rows + dyad_rows),
        "dev-natural": 263 + K.FAMILIES_PER_SPLIT["dev"] * natural_rows,
        "dev-counterfactual": 263 + K.FAMILIES_PER_SPLIT["dev"] * (K.SHARED_PER_FAMILY + dyad_rows),
        "new-heldout": K.FAMILIES_PER_SPLIT["heldout"] * (natural_rows + dyad_rows),
        "new-heldout-natural": K.FAMILIES_PER_SPLIT["heldout"] * natural_rows,
        "new-heldout-counterfactual": K.FAMILIES_PER_SPLIT["heldout"] * dyad_rows,
        "old-heldout": 150, "composite": None,
    }
    shapes = {
        "r1": route_shape(("train",), False, ("train",), (K.POOL_SHARED, K.POOL_CONTROL)),
        "r2": route_shape(("train",), True, ("train",), (K.POOL_SHARED, K.POOL_COUNTERFACTUAL)),
        "r3": route_shape(("train",), True, ("train",), (K.POOL_SHARED, K.POOL_COUNTERFACTUAL)),
        "dev": route_shape(("dev",), True, ("dev",)),
        "dev-natural": route_shape(("dev",), False, ("dev",), (K.POOL_SHARED, K.POOL_CONTROL)),
        "dev-counterfactual": route_shape(("dev",), True, ("dev",),
                                          (K.POOL_SHARED, K.POOL_COUNTERFACTUAL)),
        "new-heldout": route_shape(("heldout",), True, ()),
        "new-heldout-natural": route_shape(("heldout",), True, (), (K.POOL_SHARED, K.POOL_CONTROL)),
        "new-heldout-counterfactual": route_shape(("heldout",), True, (), (K.POOL_COUNTERFACTUAL,)),
        "old-heldout": route_shape((), True, ("heldout",)),
        "composite": None,
    }
    checker.check("route_shapes", shapes == expected_shapes, actual=shapes, expected=expected_shapes)

    heldout = rows_by_split["heldout"]
    natural_heldout = [row for row in heldout if K.row_pool(row) in (K.POOL_SHARED, K.POOL_CONTROL)]
    cf_heldout = [row for row in heldout if K.row_pool(row) == K.POOL_COUNTERFACTUAL]
    denominators = {
        "natural": {"rows": len(natural_heldout),
                    "conflict": sum(1 for row in natural_heldout if row["coarse_label"] == "conflict")},
        "counterfactual": {"rows": len(cf_heldout),
                           "conflict": sum(1 for row in cf_heldout if row["coarse_label"] == "conflict")},
    }
    checker.check("heldout_denominators", denominators == {
        "natural": {"rows": 288, "conflict": 96},
        "counterfactual": {"rows": 144, "conflict": 48}}, actual=denominators)

               
    definition = K.route_definition("new-heldout")
    checker.check("final_test_unique",
                  definition["new_splits"] == ("heldout",) and not definition["legacy_splits"],
                  route=definition, final_test_route=manifest.get("final_test_route"))

                         
    fixture_rows = sum(1 for row in all_rows if row["source"] == K.FIXTURE_SOURCE)
    if mode == "fixture":
        checker.check("fixture_contract",
                      manifest.get("fixture") is True and fixture_rows == len(all_rows),
                      manifest_fixture=manifest.get("fixture"), fixture_rows=fixture_rows,
                      total=len(all_rows))
    else:
        checker.check("fixture_contract",
                      manifest.get("fixture") is False and fixture_rows == 0,
                      manifest_fixture=manifest.get("fixture"), fixture_rows=fixture_rows)

              
    coverage = {}
    for split in K.SPLITS:
        coverage[split] = dict(Counter(row["phenomena"]["mechanism"] for row in rows_by_split[split]))
    missing = [split for split in K.SPLITS if set(coverage[split]) != {m["id"] for m in S.MECHANISMS}]
    checker.check("mechanism_coverage", not missing, coverage=coverage, missing=missing)

                             
    if tokenizer is not None:
        lengths = [len(tokenizer(row["visible_text"], add_special_tokens=True,
                                 truncation=False)["input_ids"]) for row in all_rows]
        maximum = max(lengths) if lengths else 0
        checker.check("token_length", maximum <= K.MAX_LENGTH, max_tokens=maximum,
                      max_length=K.MAX_LENGTH, n=len(lengths))
    else:
        checker.check("token_length", True, applicable=False, note="未传 tokenizer；训练前 audit 必须补跑")

    report = {
        "schema": AUDIT_SCHEMA, **POLICY_IDENTITY, "generated": K.utc_now(), "mode": mode,
        "processed_dir": str(processed_dir), "ok": checker.ok,
        "counts": {split: {"rows": len(rows_by_split[split]),
                           "families": len({row["family_id"] for row in rows_by_split[split]}),
                           "coarse": K.family_balance(rows_by_split[split]),
                           "pools": dict(Counter(K.row_pool(row) for row in rows_by_split[split]))}
                   for split in K.SPLITS},
        "near_duplicate": near,
        "factor_twins": twins, "semantic_twins": meaning_twins, "strata": twins["strata"],
        "semantic_dependencies": {split: DIAG.semantic_clusters(rows_by_split[split], specs)
                                  for split in K.SPLITS},
        "route_shapes": shapes,
        "heldout_denominators": denominators,
        "checks": checker.checks,
    }
    return report


                                                                                        
     
                                                                                        

def cmd_specs(args) -> int:
    specs = build_specs()
    out = Path(args.out) if args.out else K.SPECS_PATH
    K.write_jsonl_atomic(out, specs)
    summary = check_spec_set(specs)
    manifest = {
        "schema": SPEC_SCHEMA, **POLICY_IDENTITY, "created": K.utc_now(), "path": out.name,
        "sha256": K.sha256_file(out), "families": summary["families"],
        "splits": summary["splits"], "mechanisms": summary["mechanisms"],
        "rows": sum(len(spec["items"]) for spec in specs),
        "shared_rows": sum(1 for spec in specs for item in spec["items"] if item["kind"] == "shared"),
        "control_rows": sum(1 for spec in specs for item in spec["items"] if item["kind"] == "control"),
        "dyad_rows": sum(1 for spec in specs for item in spec["items"] if item["dyad"]),
        "author_plan": {key: list(value) for key, value in K.AUTHOR_PLAN.items()},
        "design": "v2.2: 60 families x 18 rows = 1080; 24/12/24 families",
        "factor_twins": DIAG.semantic_twins(specs),
        "semantic_twins": DIAG.semantic_twins(specs, meaning=True),
        "control_contrast_failures": control_spec_contrasts(specs),
        "ready_for_real_author": not control_spec_contrasts(specs),
    }
    K.write_json_atomic(out.parent / "manifest.json", manifest)
    print(json.dumps({"command": "specs", **manifest}, ensure_ascii=False, indent=2))
    return 0


class ChannelError(RuntimeError):
    """通道未就绪 / 配置与预注册不符；绝不静默改用别的模型或端点。"""


def _llm_client():
    """延迟导入生产 LLM 客户端（唯一 HTTP 实现，本仓库不复制第二份）。"""
    from dataset_recommender.llm import llm_client as module
    return module


def usage_from_raw(raw_response) -> dict:
    """从 provider 原始响应里取 usage 数值；缺失即 None，不猜、不伪造。"""
    usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    def _int(key):
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return {"prompt_tokens": _int("prompt_tokens"),
            "completion_tokens": _int("completion_tokens"),
            "total_tokens": _int("total_tokens")}


class ChatChannel:
    """生产 ``dataset_recommender.llm.llm_client`` 的薄适配层。

    * 调用前设 ``os.environ["LLM_MODEL"] = model``，再由生产配置读 key/endpoint；
    * 只接受与预注册模型一致的配置，否则拒绝（防止把 API 模型名发给本机 Ollama 或反向）；
    * 不打印/不保存密钥；base_url 只以 SHA 形式进入产物；
    * 单次调用，不自动重试；是否可用由生产 ``should_use_llm`` 裁决。
    """

    def __init__(self, model: str, *, provider_override: "str | None" = None,
                 config_loader=None, caller=None, gate=None, is_complete=None,
                 max_tokens: int = 8000) -> None:
        self.model = model
        self.provider_override = provider_override
        self._loader = config_loader
        self._caller = caller
        self._gate = gate
        self._is_complete = is_complete
        self.max_tokens = int(max_tokens)
        self.config = None

    def prepare(self) -> dict:
        if not self.model:
            raise ChannelError("chat 需要显式 --model")
        client = _llm_client()
        os.environ["LLM_MODEL"] = self.model
        os.environ["LLM_MAX_TOKENS"] = str(self.max_tokens)
        loader = self._loader or (lambda: client.load_llm_config(provider_override=self.provider_override))
        try:
            config = loader()
        except Exception as error:                    
            raise ChannelError(f"config load failed: {type(error).__name__}") from error
        if getattr(config, "model", None) != self.model:
            raise ChannelError(
                f"configured model {getattr(config, 'model', None)!r} != pre-registered {self.model!r}；"
                "若 provider=trial 会忽略 LLM_MODEL，请检查 LLM_PROVIDER/BIODATA_TRIAL_MODEL")
        config.thinking = False
        gate = self._gate or client.should_use_llm
        verdict, reason = gate(config)
        if not verdict:
            raise ChannelError(f"chat channel not ready: {reason}")
        self.config = config
        self._client = client
        return {"provider": config.provider, "model": config.model,
                "max_tokens": getattr(config, "max_tokens", self.max_tokens),
                "thinking": config.thinking,
                "base_url_sha256": K.sha256_text(getattr(config, "base_url", "") or "")}

    def complete(self, prompt: str) -> dict:
        if self.config is None:
            self.prepare()
        client = self._client
        caller = self._caller or client.call_llm
        complete_check = self._is_complete or client.is_complete_finish_reason
        started = time.perf_counter()
        try:
            result = caller(prompt, self.config)
        except Exception as error:                                             
            return {"text": None, "succeeded": False, "attempted": True,
                    "finish_reason": None, "error": f"caller_exception: {type(error).__name__}",
                    "usage": usage_from_raw(None),
                    "latency_s": round(time.perf_counter() - started, 3),
                    "provider": self.config.provider, "model": self.config.model,
                    "max_tokens": getattr(self.config, "max_tokens", self.max_tokens),
                    "thinking": self.config.thinking,
                    "base_url_sha256": K.sha256_text(getattr(self.config, "base_url", "") or "")}
        latency = round(time.perf_counter() - started, 3)
        text = result.text if isinstance(result.text, str) else None
        finish = getattr(result, "finish_reason", None)
        succeeded = bool(getattr(result, "succeeded", False)) and bool(text and text.strip()) \
            and bool(complete_check(finish))
        return {
            "text": text,
            "succeeded": succeeded,
            "attempted": bool(getattr(result, "attempted", False)),
            "finish_reason": finish,
            "error": getattr(result, "error", None),
            "usage": usage_from_raw(getattr(result, "raw_response", None)),
            "latency_s": latency,
            "provider": getattr(result, "provider", None) or self.config.provider,
            "model": getattr(result, "model", None) or self.config.model,
            "max_tokens": getattr(self.config, "max_tokens", self.max_tokens),
            "thinking": self.config.thinking,
            "base_url_sha256": K.sha256_text(getattr(self.config, "base_url", "") or ""),
        }


def append_call_ledger(path: Path, entry: dict) -> None:
    """调用账本：只存 SHA 与数值，不存密钥、不存 base_url 明文、不存正文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def write_raw_attempt(out_dir: Path, spec: dict, prompt: str, outcome: dict, attempt: int) -> Path:
    import uuid
    attempt_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
    record = {
        "schema": "conflict-author-raw-attempt-v2.2-r2", **POLICY_IDENTITY, "attempt_id": attempt_id,
        "created": K.utc_now(), "family_id": spec["family_id"], "split": spec["split"],
        "process_attempt": int(attempt), "spec_sha256": K.sha256_json(spec),
        "prompt_sha256": K.sha256_text(prompt), "response_text": outcome.get("text"),
        "response_sha256": (K.sha256_text(outcome["text"])
                            if isinstance(outcome.get("text"), str) else None),
        "succeeded": outcome.get("succeeded"), "finish_reason": outcome.get("finish_reason"),
        "error_kind": error_kind(outcome.get("error")), "usage": outcome.get("usage"),
        "latency_s": outcome.get("latency_s"), "max_tokens": outcome.get("max_tokens"),
        "thinking": outcome.get("thinking"), "provider": outcome.get("provider"),
        "model": outcome.get("model"), "base_url_sha256": outcome.get("base_url_sha256"),
    }
    path = out_dir / "raw" / f"{spec['family_id']}-{attempt_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return path


def error_kind(error: "str | None") -> "str | None":
    """只保留错误类别前缀（冒号前），避免把 provider 正文或 URL 落盘。"""
    if not error:
        return None
    return error.split(":", 1)[0][:80]


def verified_record(path: Path, spec: dict, expected_models: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        record = K.read_json(path)
        if record.get("schema") != AUTHOR_SCHEMA or record.get("family_id") != spec["family_id"]:
            return False
        if record.get("split") != spec["split"] or record.get("source") != "author":
            return False
        if any(record.get(key) != value for key, value in POLICY_IDENTITY.items()):
            return False
        if record.get("spec_sha256") != K.sha256_json(spec) \
                or record.get("prompt_sha256") != K.sha256_text(build_prompt(spec)):
            return False
        if expected_models and record.get("model") not in expected_models:
            return False
        items = record.get("items")
        if not isinstance(items, list) or any(not isinstance(entry, dict) for entry in items):
            return False
        payload = {"background": record.get("background", ""),
                   "items": {entry.get("pair_id"): entry for entry in items}}
        if len(payload["items"]) != len(items):
            return False
        if validate_author_payload(spec, payload):
            return False
        normalized = canonical_control_old(spec, payload)
        if any(payload["items"][key].get("old") != value["old"]
               for key, value in normalized["items"].items()):
            return False
        return record.get("response_sha256") == K.sha256_json({
            "background": payload["background"],
            "items": {key: {side: value[side] for side in ("old", "new")}
                      for key, value in payload["items"].items()}})
    except Exception:
        return False


def author_route_guard(split: str, channel: str, model: "str | None", budget: int) -> dict:
    """v2.2 分派：train/dev/composite=chat，heldout=本机 ollama；file 仅非 heldout。"""
    if channel == "fixture":
        return {"channel": "fixture"}
    expected = K.AUTHOR_PLAN.get(split, ())
    if channel == "file":
        if split == "heldout":
            raise SystemExit("refuse: heldout 只允许 --channel ollama（本机 Qwen），file 导入不适用")
        if not model:
            raise SystemExit("--channel file 需要 --model（写入记录的可追溯作者名）")
        if model not in expected:
            raise SystemExit(f"refuse: {split} 的预注册作者是 {list(expected)}，收到 {model!r}")
        return {"channel": "file", "model": model}
    if not model:
        raise SystemExit(f"author 需要显式 --model；{split} 的预注册模型是 {list(expected)}")
    if model not in expected:
        raise SystemExit(f"refuse: {split} 的预注册作者是 {list(expected)}，收到 {model!r}；"
                         "如要改分派请先改设计与 AUTHOR_PLAN")
    if split == "heldout":
        if channel != "ollama":
            raise SystemExit("refuse: heldout 必须 --channel ollama（本机 batiai/qwen3.8-27b:iq3）；"
                             "chat 只用于 train/dev/composite")
    else:
        if channel != "chat":
            raise SystemExit(f"refuse: {split} 的预注册作者是 API 模型；请用 --channel chat --model {model}")
        if not budget:
            raise SystemExit("refuse: --channel chat 必须显式 --budget-calls N（调用上限）")
    return {"channel": channel, "model": model, "budget": int(budget)}


def cmd_author(args) -> int:
    specs = K.read_jsonl(Path(args.specs) if getattr(args, "specs", None) else K.SPECS_PATH)
    check_spec_set(specs)
    if args.channel != "fixture":
        require_feasible_controls(specs)
    selected = [spec for spec in specs if spec["split"] == args.split]
    if args.limit:
        selected = selected[:int(args.limit)]
    route = author_route_guard(args.split, args.channel, args.model, args.budget_calls)
    expected_models = K.AUTHOR_PLAN.get(args.split, ())
    out_dir = Path(args.out) if args.out else K.AUTHORS_DIR / _slug(args.model or "fixture")
    if args.channel == "fixture":
        out_dir = guard_fixture_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger) if args.ledger else out_dir / "calls.jsonl"
    chat = None
    if args.channel == "chat":
        chat = ChatChannel(args.model)
        chat.prepare()                        
    budget = int(args.budget_calls or 0)
    calls = written = skipped = 0
    failures: list[dict] = []
    for spec in selected:
        target = out_dir / f"{spec['family_id']}.json"
        if args.resume and verified_record(target, spec, expected_models):
            skipped += 1
            continue
        if budget and calls >= budget and args.channel in ("chat", "ollama"):
            break
        prompt = build_prompt(spec)
        identity = None
        if args.channel == "fixture":
            payload = fixture_author(spec)
            usage = {"calls": 0}
            source = "fixture"
        elif args.channel == "file":
            source_path = Path(args.from_dir) / f"{spec['family_id']}.json"
            if not source_path.is_file():
                raise SystemExit(f"missing author file {source_path}")
            raw = K.read_json(source_path)
            payload = {"background": raw["background"],
                       "items": {entry["pair_id"]: {"old": entry["old"], "new": entry["new"]}
                                 for entry in raw["items"]}}
            usage = raw.get("usage") or {}
            identity = raw.get("identity")
            source = "author"
        else:
            channel = chat if chat is not None else OllamaChannel(args.model)
            started = time.perf_counter()
            try:
                outcome = channel.complete(prompt)
            except Exception as error:
                outcome = {"succeeded": False, "attempted": True, "text": None,
                           "error": type(error).__name__, "finish_reason": None,
                           "model": args.model, "provider": args.channel,
                           "usage": usage_from_raw(None),
                           "latency_s": round(time.perf_counter() - started, 3),
                           "max_tokens": 8000, "thinking": False}
            calls += 1
            raw_path = write_raw_attempt(out_dir, spec, prompt, outcome, calls)
            status = "ok" if outcome["succeeded"] else "incomplete"
            append_call_ledger(ledger, {
                **POLICY_IDENTITY,
                "family_id": spec["family_id"], "split": args.split, "channel": args.channel,
                "model": outcome.get("model") or args.model, "provider": outcome.get("provider"),
                "status": status, "finish_reason": outcome.get("finish_reason"),
                "error_kind": error_kind(outcome.get("error")),
                "usage": outcome.get("usage"), "latency_s": outcome.get("latency_s"),
                "max_tokens": outcome.get("max_tokens"), "thinking": outcome.get("thinking"),
                "spec_sha256": K.sha256_json(spec), "prompt_sha256": K.sha256_text(prompt),
                "raw_attempt": raw_path.relative_to(out_dir).as_posix(),
                "response_sha256": (K.sha256_text(outcome["text"])
                                    if isinstance(outcome.get("text"), str) else None),
                "base_url_sha256": outcome.get("base_url_sha256"), "created": K.utc_now(),
            })
            if not outcome["succeeded"]:
                failures.append({"family_id": spec["family_id"], "status": status,
                                 "finish_reason": outcome.get("finish_reason"),
                                 "error_kind": error_kind(outcome.get("error"))})
                continue                             
            try:
                payload = parse_author_response(outcome["text"], spec)
            except AuthorError as error:
                failures.append({"family_id": spec["family_id"], "status": "unparsed",
                                 "error_kind": str(error)[:80]})
                continue
            usage = outcome["usage"]
            identity = {"provider": outcome["provider"], "model": outcome["model"],
                        "base_url_sha256": outcome["base_url_sha256"],
                        "max_tokens": outcome.get("max_tokens"),
                "thinking": outcome.get("thinking"),
                        "finish_reason": outcome["finish_reason"],
                        "latency_s": outcome["latency_s"], "channel": args.channel}
            source = "author"
        problems = validate_author_payload(spec, payload)
        if problems:
            if args.channel in ("chat", "ollama"):
                failures.append({"family_id": spec["family_id"], "status": "rejected",
                                 "error_kind": problems[0][:80]})
                continue
            raise SystemExit(f"{spec['family_id']}: author payload rejected:\n" + "\n".join(problems[:20]))
        write_author_record(out_dir, spec, payload, model=args.model or K.FIXTURE_AUTHOR,
                            prompt=prompt, usage=usage, source=source, identity=identity)
        written += 1
    summary = {"command": "author", "split": args.split, "channel": args.channel,
               "model": args.model, "out": str(out_dir), "written": written, "calls": calls,
               "skipped_verified": skipped, "failures": len(failures),
               "failed": failures[:10], "budget": budget or None,
               "route": route, "ledger": str(ledger)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def guard_fixture_output(path: Path) -> Path:
    """fixture 产物只允许落在 ``_selftest/`` 下，避免污染 data/processed 或 data/authors。"""
    resolved = Path(path).resolve()
    if K.SELFTEST_DIR.resolve() not in resolved.parents and resolved != K.SELFTEST_DIR.resolve():
        raise SystemExit(f"refuse: fixture 产物只能写到 {K.SELFTEST_DIR} 下，收到 {resolved}")
    return resolved


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name or "unknown")


def cmd_assemble(args) -> int:
    specs = K.read_jsonl(Path(args.specs) if getattr(args, "specs", None) else K.SPECS_PATH)
    check_spec_set(specs)
    if args.fixture:
        records = {spec["family_id"]: {
            "schema": AUTHOR_SCHEMA, **POLICY_IDENTITY, "family_id": spec["family_id"], "split": spec["split"],
            "author_role": spec["author_role"], "model": K.FIXTURE_AUTHOR, "source": "fixture",
            "created": K.utc_now(), "background": spec["background_hint"],
            "items": [{"pair_id": key, "old": value["old"], "new": value["new"]}
                      for key, value in sorted(fixture_author(spec)["items"].items())],
        } for spec in specs}
    else:
        if not args.authors:
            raise SystemExit("assemble 需要 --authors DIR（真实作者产物）或 --fixture（仅 selftest）")
        records = load_author_records(Path(args.authors))
        missing = [spec["family_id"] for spec in specs if spec["family_id"] not in records]
        if missing:
            raise SystemExit(f"缺少 {len(missing)} 个 family 的作者记录：{missing[:5]}")
    out_dir = Path(args.out) if args.out else K.PROCESSED_DIR
    if args.fixture:
        out_dir = guard_fixture_output(out_dir)
    manifest = assemble(specs, records, fixture=bool(args.fixture), out_dir=out_dir)
    print(json.dumps({"command": "assemble", "out": str(out_dir), "fixture": manifest["fixture"],
                      "files": manifest["files"], "authors": manifest["authors"]},
                     ensure_ascii=False, indent=2))
    return 0


def _load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(K.C.QWEN["dir"]), local_files_only=True)
    return tokenizer


def cmd_fixture_bundle(args) -> int:
    """fixture 与审计按匹配协议版本分路径保存，历史红审计不覆盖。"""
    specs = build_specs()
    records = {spec["family_id"]: _fixture_record(spec) for spec in specs}
    out_dir = guard_fixture_output(K.FIXTURE_DIR / "processed")
    manifest = assemble(specs, records, fixture=True, out_dir=out_dir)
    report = audit_dataset(processed_dir=out_dir, manifest_path=out_dir / "manifest.json",
                           specs=specs, mode="fixture")
    K.write_json_atomic(K.AUDITS_DIR / "fixture-dataset-audit-v2.2-r2.json", report)
    print(json.dumps({"command": "fixture-bundle", "ok": report["ok"],
                      "failed": [name for name, entry in report["checks"].items() if not entry["ok"]],
                      "files": manifest["files"], "route_shapes": report["route_shapes"],
                      "heldout_denominators": report["heldout_denominators"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def cmd_audit(args) -> int:
    specs = K.read_jsonl(Path(args.specs) if getattr(args, "specs", None) else K.SPECS_PATH)
    processed = Path(args.processed) if args.processed else K.PROCESSED_DIR
    missing = [split for split in K.SPLITS if not (processed / f"{split}.jsonl").is_file()]
    if missing:
        raise SystemExit(f"refuse: {processed} 缺 {missing}；先跑 assemble（真实作者产物，不能用 fixture）")
    manifest_path = processed / "manifest.json"
    tokenizer = _load_tokenizer() if args.with_tokenizer else None
    report = audit_dataset(processed_dir=processed, manifest_path=manifest_path, specs=specs,
                           mode=args.mode, tokenizer=tokenizer)
    out = Path(args.out) if args.out else K.AUDITS_DIR / f"audit-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    K.write_json_atomic(out, report)
    print(json.dumps({"command": "audit", "ok": report["ok"], "mode": args.mode,
                      "report": str(out), "counts": report["counts"],
                      "failed": [name for name, entry in report["checks"].items() if not entry["ok"]]},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def cmd_routes(args) -> int:
    manifest = K.read_json(K.MANIFEST_PATH) if K.MANIFEST_PATH.is_file() else None
    table = {}
    for route in ("r1", "r2", "r3", "dev", "old-train", "old-dev", "new-dev",
                  "old-calibration", "old-heldout", "new-heldout"):
        try:
            rows = K.route_rows(route, purpose="audit", manifest=manifest)
        except Exception as error:
            table[route] = {"error": str(error)[:120]}
            continue
        table[route] = {"n": len(rows), "fingerprint": K.route_fingerprint(route, rows),
                        "role": K.route_definition(route)["role"]}
    K.write_json_atomic(K.ROUTES_PATH, {"generated": K.utc_now(), "routes": table})
    print(json.dumps({"command": "routes", "routes": table}, ensure_ascii=False, indent=2))
    return 0


                                                                                        
                  
                                                                                        

def _run_pipeline(root: Path, *, mode: str = "fixture", tamper=None) -> dict:
    specs = build_specs()
    K.write_jsonl_atomic(root / "specs.jsonl", specs)
    if tamper == "missing_author":
        records = {}
        for spec in specs[:-1]:
            records[spec["family_id"]] = _fixture_record(spec)
    else:
        records = {spec["family_id"]: _fixture_record(spec) for spec in specs}
    out_dir = root / "processed"
    manifest = assemble(specs, records, fixture=True, out_dir=out_dir)
    rows = {split: K.read_jsonl(out_dir / f"{split}.jsonl") for split in K.SPLITS}
    if tamper == "label":
        rows["train"][0]["label"] = "disjoint"
        rows["train"][0]["coarse_label"] = "conflict"
        K.write_jsonl_atomic(out_dir / "train.jsonl", rows["train"])
    if tamper == "cross_split":
        duplicate = dict(rows["train"][1])
        duplicate["id"] = "cf-dup-heldout"
        duplicate["split"] = "heldout"
        rows["heldout"].append(duplicate)
        K.write_jsonl_atomic(out_dir / "heldout.jsonl", rows["heldout"])
    if tamper == "dyad":
        target = next(row for row in rows["dev"] if row["dyad"] and row["dyad"]["role"] == "conflict")
        factor = target["dyad"]["factor"]
        target["pair"]["new"]["text"] = target["pair"]["new"]["text"].replace(
            factor["to_render"], factor["from_render"], 1)
        K.write_jsonl_atomic(out_dir / "dev.jsonl", rows["dev"])
    if tamper == "control":
        target = next(row for row in rows["train"] if K.row_pool(row) == K.POOL_CONTROL)
        target["phenomena"]["matched_side"] = "b" if target["phenomena"]["matched_side"] == "a" else "a"
        K.write_jsonl_atomic(out_dir / "train.jsonl", rows["train"])
    if tamper == "author":
        for row in rows["heldout"]:
            row["author"] = rows["train"][0]["author"]
        K.write_jsonl_atomic(out_dir / "heldout.jsonl", rows["heldout"])
    if tamper == "fixture_flag":
        manifest["fixture"] = False
        K.write_json_atomic(out_dir / "manifest.json", manifest)
    audit_mode = "final" if tamper in ("fixture_flag", "author") else "fixture"
    report = audit_dataset(processed_dir=out_dir, manifest_path=out_dir / "manifest.json",
                           specs=specs, mode=audit_mode)
    return {"specs": specs, "manifest": manifest, "report": report, "rows": rows, "out": out_dir}


def _fixture_record(spec: dict) -> dict:
    payload = fixture_author(spec)
    return {
        "schema": AUTHOR_SCHEMA, **POLICY_IDENTITY, "family_id": spec["family_id"], "split": spec["split"],
        "author_role": spec["author_role"], "model": K.FIXTURE_AUTHOR, "source": "fixture",
        "created": K.utc_now(), "background": payload["background"],
        "spec_sha256": K.sha256_json(spec), "prompt_sha256": K.sha256_text(build_prompt(spec)),
        "response_sha256": K.sha256_json(payload),
        "items": [{"pair_id": key, "old": value["old"], "new": value["new"]}
                  for key, value in sorted(payload["items"].items())],
    }


def _chat_selftest(record) -> None:
    """离线伪造 chat 通道：验证映射/账本/拒绝/分派/resume；不联网、不读真实 .env。"""
    import tempfile
    import types

    key = "sk-selftest-DO-NOT-PERSIST"
    base = "https://api.example.invalid/v1"
    model = "deepseek-v4-flash"
    captured = {}

    def loader():
        captured["env_model"] = os.environ.get("LLM_MODEL")
        captured["env_max_tokens"] = os.environ.get("LLM_MAX_TOKENS")
        return types.SimpleNamespace(provider="trial", model=model, base_url=base,
                                     api_key=key, enable_llm=True)

    def caller_ok(prompt, config):
        return types.SimpleNamespace(
            text='{"family_id":"cf-001"}', attempted=True, succeeded=True, finish_reason="stop",
            error=None, raw_response={"usage": {"prompt_tokens": 11, "completion_tokens": 22,
                                                "total_tokens": 33}},
            provider="trial", model=model)

    channel = ChatChannel(model, config_loader=loader, caller=caller_ok,
                          gate=lambda config: (True, "ready"))
    previous = os.environ.get("LLM_MODEL")
    previous_max_tokens = os.environ.get("LLM_MAX_TOKENS")
    try:
        info = channel.prepare()
        outcome = channel.complete("请写句对")
    finally:
        if previous is None:
            os.environ.pop("LLM_MODEL", None)
        else:
            os.environ["LLM_MODEL"] = previous
        if previous_max_tokens is None:
            os.environ.pop("LLM_MAX_TOKENS", None)
        else:
            os.environ["LLM_MAX_TOKENS"] = previous_max_tokens
    record("chat_outcome_mapping",
           captured.get("env_model") == model
           and captured.get("env_max_tokens") == "8000"
           and info["max_tokens"] == 8000
           and outcome["max_tokens"] == 8000
           and info["thinking"] is False
           and outcome["thinking"] is False
           and info["base_url_sha256"] == K.sha256_text(base)
           and outcome["text"] == '{"family_id":"cf-001"}'
           and outcome["succeeded"] is True
           and outcome["usage"] == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
           and outcome["provider"] == "trial" and outcome["model"] == model
           and outcome["latency_s"] >= 0,
           usage=outcome["usage"], env_model=captured.get("env_model"),
           max_tokens=outcome["max_tokens"], thinking=outcome["thinking"])

    with tempfile.TemporaryDirectory(prefix="chat-ledger-") as temporary:
        path = Path(temporary) / "calls.jsonl"
        append_call_ledger(path, {"family_id": "cf-001", "model": model,
                                  "provider": outcome["provider"], "status": "ok",
                                  "finish_reason": outcome["finish_reason"],
                                  "prompt_sha256": K.sha256_text("p"),
                                  "usage": outcome["usage"],
                                  "base_url_sha256": outcome["base_url_sha256"]})
        line = path.read_text(encoding="utf-8")
    record("chat_no_secret_in_ledger",
           key not in line and base not in line and K.sha256_text("p") in line)

    def channel_with(*, finish="stop", succeeded=True, text='{"x":1}',
                     gate=lambda config: (True, "ready"), model_name=model):
        def caller(prompt, config):
            return types.SimpleNamespace(text=text, attempted=True, succeeded=succeeded,
                                         finish_reason=finish, error=None,
                                         raw_response={"usage": {"total_tokens": 3}},
                                         provider="trial", model=config.model)
        return ChatChannel(model_name, config_loader=loader, caller=caller, gate=gate)

    record("chat_length_rejected", channel_with(finish="length").complete("x")["succeeded"] is False)
    record("chat_failed_rejected", channel_with(succeeded=False).complete("x")["succeeded"] is False)
    record("chat_empty_text_rejected", channel_with(text="  ").complete("x")["succeeded"] is False)

    raised = ""
    try:
        channel_with(model_name="some-other-model").prepare()
    except ChannelError as error:
        raised = str(error)
    record("chat_model_mismatch_refused", "pre-registered" in raised, error=raised[:80])

    raised = ""
    try:
        channel_with(gate=lambda config: (False, "no_key")).prepare()
    except ChannelError as error:
        raised = str(error)
    record("chat_not_ready_refused", "no_key" in raised, error=raised[:80])

    record("chat_route_guard",
           author_route_guard("train", "chat", "deepseek-v4-flash", 5)["budget"] == 5
           and author_route_guard("heldout", "ollama", "batiai/qwen3.8-27b:iq3", 5)["channel"] == "ollama")
    refusals = 0
    for split, channel_name, model_name, budget in (
            ("train", "chat", "deepseek-v4-flash", 0),
            ("train", "ollama", "deepseek-v4-flash", 5),
            ("heldout", "chat", "batiai/qwen3.8-27b:iq3", 5),
            ("heldout", "file", "batiai/qwen3.8-27b:iq3", 5)):
        try:
            author_route_guard(split, channel_name, model_name, budget)
        except SystemExit:
            refusals += 1
    record("chat_route_guard_refusals", refusals == 4, refused=refusals)

    spec = next(spec for spec in build_specs() if spec["split"] == "train")
    valid = _fixture_record(spec)
    valid["model"] = "deepseek-v4-flash"
    valid["source"] = "author"
    with tempfile.TemporaryDirectory(prefix="chat-resume-") as temporary:
        path = Path(temporary) / f"{spec['family_id']}.json"
        K.write_json_atomic(path, valid)
        accepted = verified_record(path, spec, ("deepseek-v4-flash",))
        K.write_json_atomic(path, dict(valid, items=valid["items"][:3]))
        truncated_rejected = not verified_record(path, spec, ("deepseek-v4-flash",))
        K.write_json_atomic(path, dict(valid, model="other-model"))
        wrong_model_rejected = not verified_record(path, spec, ("deepseek-v4-flash",))
    record("chat_resume_only_verified",
           accepted and truncated_rejected and wrong_model_rejected,
           accepted=accepted, truncated=truncated_rejected, wrong_model=wrong_model_rejected)
    record("author_plan_unchanged",
           {k: tuple(v) for k, v in K.AUTHOR_PLAN.items()} == {
               "train": ("deepseek-v4-flash",), "dev": ("deepseek-v4-flash",),
               "heldout": ("batiai/qwen3.8-27b:iq3",), "composite": ("deepseek-v4-flash",)},
           plan={key: list(value) for key, value in K.AUTHOR_PLAN.items()})


def run_selftest() -> dict:
    from conflict_selftest import run_data_selftest
    return run_data_selftest(sys.modules[__name__])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=lambda args: (run_selftest(), 0)[1])
    specs = sub.add_parser("specs")
    specs.add_argument("--out", default=None)
    specs.set_defaults(func=cmd_specs)
    author = sub.add_parser("author")
    author.add_argument("--split", required=True, choices=K.SPLITS)
    author.add_argument("--model", default=None)
    author.add_argument("--channel", default="chat", choices=("chat", "ollama", "file", "fixture"),
                        help="chat=复用生产 dataset_recommender.llm.llm_client（train/dev/composite，"
                             "必须 --model 与 --budget-calls）；ollama=本机 Qwen（仅 heldout）；"
                             "file=离线导入；fixture=仅 selftest")
    author.add_argument("--out", default=None)
    author.add_argument("--from-dir", default=None)
    author.add_argument("--limit", type=int, default=0)
    author.add_argument("--budget-calls", type=int, default=0,
                        help="chat/ollama 的最大调用次数；chat 必填（单次失败不自动重试）")
    author.add_argument("--resume", action="store_true")
    author.add_argument("--ledger", default=None)
    author.add_argument("--specs", default=None)
    author.set_defaults(func=cmd_author)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--authors", default=None)
    assemble_parser.add_argument("--fixture", action="store_true")
    assemble_parser.add_argument("--out", default=None)
    assemble_parser.add_argument("--specs", default=None)
    assemble_parser.set_defaults(func=cmd_assemble)
    audit = sub.add_parser("audit")
    audit.add_argument("--processed", default=None)
    audit.add_argument("--mode", default="final", choices=("final", "fixture"))
    audit.add_argument("--with-tokenizer", action="store_true")
    audit.add_argument("--specs", default=None)
    audit.add_argument("--out", default=None)
    audit.set_defaults(func=cmd_audit)
    sub.add_parser("fixture-bundle").set_defaults(func=cmd_fixture_bundle)
    sub.add_parser("routes").set_defaults(func=cmd_routes)
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        return (run_selftest(), 0)[1]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
