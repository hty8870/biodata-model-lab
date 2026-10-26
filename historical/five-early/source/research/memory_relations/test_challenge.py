                       
"""挑战集静态契约与投影测试（研究专项）。

标签来源为代理（AI）审阅（label_origin=agent_review），不是真人金标；
本文件只验证**构造策略与输入合同**（形状、投影、出处闸、合并口径、指纹、哈希），
不构成语义 gold、不证明任何模型的语义准确率。独立复算内容：
- 行/请求/候选/规则/证据形状与引用完整；
- 每行经生产 `memory_relations._request` / `_size_fit_batches` 的投影与出处闸，
  pair 行的期望判定再过生产 `_decision` 输出闸；
- set 行标签由条目级人工关系按公开策略与生产 `_merge_decisions` 重新推导
  （跨批结果不同的行另标 phenomena.merge_policy）；
- 全行指纹唯一与 manifest 冻结哈希一致。
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_recommender.agent import memory_relations as M              
from dataset_recommender.llm.dream import _evidence_checks, _norm              
from dataset_recommender.secret_patterns import find_secret_patterns              

RESEARCH = ROOT / "research" / "memory_relations"
CHALLENGE_PATH = RESEARCH / "data" / "challenge.jsonl"
SET_PATH = RESEARCH / "data" / "set_challenge.jsonl"
MANIFEST_PATH = RESEARCH / "challenge_manifest.json"
SPEC_PATH = RESEARCH / "spec.json"

ROW_FIELDS = {"id", "group_id", "split", "request", "label", "label_origin",
              "related_ids", "evidence_refs", "phenomena"}
REQUEST_FIELDS = {"candidate", "existing", "evidence", "context"}
CANDIDATE_FIELDS = {"principal", "endpoint_fp", "project_id", "text", "trigger",
                    "tier", "evidence"}
RULE_FIELDS = {"principal", "endpoint_fp", "project_id", "id", "text", "trigger",
               "tier", "evidence"}
EVIDENCE_FIELDS = {"id", "text"}
ENTRY_RELATIONS = set(M.RELATIONS) | {"unrelated"}
DOMAINS = ("code_style", "meeting", "travel", "writing", "ranking", "accessibility", "reminder")
SET_SIZES = (0, 1, 8, 64, 65, 129)
REQUIRED_MECHANISMS = ("double_negation", "implicit_mutex", "conditional_scope",
                       "time_bound", "revocation_evidence", "empty_existing",
                       "composite_mixed", "long_distractor")
CANONICAL_FP = hashlib.sha256(b"https://model.example/v1|decision-model").hexdigest()[:12]


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="module")
def spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pairs():
    return _load(CHALLENGE_PATH)


@pytest.fixture(scope="module")
def sets():
    return _load(SET_PATH)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def parts(row):
    request = copy.deepcopy(row["request"])
    return request["candidate"], request["existing"], request["evidence"]


def fingerprint(row):
    request = row["request"]
    payload = json.dumps({
        "candidate": _norm(request["candidate"]["text"]),
        "existing": [_norm(item["text"]) for item in request["existing"]],
        "evidence": [_norm(item["text"]) for item in request["evidence"]],
        "label": row["label"],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def context_of(candidate, existing, evidence):
    """≤64 条取生产整批投影；>64 条取各批投影的并集（与装配披露一致）。"""
    if not existing:
        return M._request(candidate, [], evidence, None)[0]["context"]
    if len(existing) <= M._MAX_ROWS:
        return M._request(candidate, existing, evidence, None)[0]["context"]
    contexts = [M._request(candidate, batch, evidence, None)[0]["context"]
                for batch in M._size_fit_batches(candidate, existing, evidence, None, False)]
    scope = contexts[0]["scope"]
    assert all(item["scope"] == scope for item in contexts)
    return {"scope": scope,
            "replaceable_ids": [i for item in contexts for i in item["replaceable_ids"]],
            "inherited_global_ids": [i for item in contexts for i in item["inherited_global_ids"]]}


def merge(names):
    return M._merge_decisions([{"relation": name, "related_ids": [], "evidence_refs": [],
                                "reason": "challenge-test", "error_code": None}
                               for name in names])["relation"]


def batch_policy(entry_relations):
    rels = [name for name in entry_relations if name != "unrelated"]
    mix = sorted(set(rels))
    if not rels:
        return "complement", ["complement"]
    if len(mix) == 1:
        return mix[0], mix
    return "uncertain", sorted(set(mix) | {"uncertain"})


def expand_plan(plan, size):
    entries = [None] * size
    for span in plan:
        start, end = span["span"]
        assert 1 <= start <= end <= size
        for index in range(start - 1, end):
            assert entries[index] is None, "entry_plan 重叠"
            entries[index] = span["relation"]
    assert all(name is not None for name in entries), "entry_plan 未覆盖全部条目"
    return entries


def batch_of(candidate, existing, evidence):
    return M._size_fit_batches(candidate, existing, evidence, None, False)


def test_spec_contract_matches_production(spec):
    assert spec["format"] == "memory-relations-experiment-v1"
    assert spec["status"] == "research-only"
    assert spec["classes"] == sorted(M.RELATIONS)
    assert "agent_review" in spec["label_origins"]
    assert {"ood", "set"}.issubset(set(spec["splits"]))


def test_pair_rows_shape_and_origin(pairs, manifest):
    assert 140 <= len(pairs) <= 210
    assert len({row["id"] for row in pairs}) == len(pairs)
    for row in pairs:
        assert set(row) == ROW_FIELDS, row["id"]
        assert row["label"] in M.RELATIONS, row["id"]
        assert row["label_origin"] == "agent_review", row["id"]
        assert row["split"] == "ood", row["id"]
        request = row["request"]
        assert set(request) == REQUEST_FIELDS, row["id"]
        candidate = request["candidate"]
        assert set(candidate) == CANDIDATE_FIELDS, row["id"]
        assert candidate["endpoint_fp"] == CANONICAL_FP, row["id"]
        assert candidate["tier"] in {"explicit", "implicit"}, row["id"]
        assert 0 < len(candidate["text"]) <= M._MAX_TEXT, row["id"]
        assert len(candidate["trigger"]) <= M._MAX_TRIGGER, row["id"]
        assert 1 <= len(candidate["evidence"]) <= M._MAX_QUOTES, row["id"]
        ids = []
        for item in request["existing"]:
            assert set(item) == RULE_FIELDS, row["id"]
            assert item["endpoint_fp"] == candidate["endpoint_fp"], row["id"]
            ids.append(item["id"])
            assert 0 < len(item["text"]) <= M._MAX_TEXT, row["id"]
        assert len(set(ids)) == len(ids), row["id"]
        assert set(row["related_ids"]) <= set(ids), row["id"]
        evidence_ids = []
        for item in request["evidence"]:
            assert set(item) == EVIDENCE_FIELDS, row["id"]
            evidence_ids.append(item["id"])
            assert 0 < len(item["text"]) <= M._MAX_EVIDENCE_TEXT, row["id"]
        assert len(set(evidence_ids)) == len(evidence_ids), row["id"]
        assert set(row["evidence_refs"]) <= set(evidence_ids), row["id"]
        phenomena = row["phenomena"]
        assert phenomena["domain"] in DOMAINS, row["id"]
        assert phenomena["relational_cases"] and phenomena["rationale"].strip(), row["id"]
    assert manifest["pair_challenge"]["rows"] == len(pairs)


def test_pair_class_balance_and_domain_coverage(pairs):
    counts = {name: 0 for name in M.RELATIONS}
    domains = {name: 0 for name in DOMAINS}
    for row in pairs:
        counts[row["label"]] += 1
        domains[row["phenomena"]["domain"]] += 1
    assert all(18 <= count <= 24 for count in counts.values()), counts
    assert all(domains[name] >= 15 for name in DOMAINS), domains


def test_pair_mechanism_coverage(pairs, sets):
    counts = {}
    for row in list(pairs) + list(sets):
        for name in row["phenomena"]["mechanism"]:
            counts[name] = counts.get(name, 0) + 1
    missing = [name for name in REQUIRED_MECHANISMS if counts.get(name, 0) < 4]
    assert not missing, (missing, counts)


def test_pair_rows_pass_production_request_gate(pairs):
    for row in pairs:
        candidate, existing, evidence = parts(row)
        request, meta = M._request(candidate, existing, evidence, None)
        assert request["context"] == row["request"]["context"], row["id"]
        pools = [(0, _norm(item["text"])) for item in evidence]
        for quote in candidate["evidence"]:
            assert _evidence_checks([quote], pools)[0], (row["id"], quote)
        raw = {"relation": row["label"], "related_ids": list(row["related_ids"]),
               "evidence_refs": list(row["evidence_refs"]),
               "reason": row["phenomena"]["rationale"], "error_code": None}
        out = M._decision(raw, request, meta, None)
        assert out["relation"] == row["label"], row["id"]
        if row["label"] == "uncertain":
            assert out["error_code"] in M._UNCERTAIN_CODES, row["id"]
        else:
            assert out["error_code"] is None, row["id"]
        if row["label"] in {"duplicate", "contextual", "supersede", "conflict"}:
            assert row["related_ids"], row["id"]


def test_pair_uncertain_mix_declares_acceptable_set(pairs):
    for row in pairs:
        mix = row["phenomena"].get("mix")
        if mix is None:
            continue
        assert row["label"] == "uncertain", row["id"]
        assert len(set(mix)) >= 2 and set(mix) <= M.RELATIONS, row["id"]
        assert row["phenomena"]["acceptable_labels"] == sorted(set(mix) | {"uncertain"}), row["id"]


def test_pair_expected_refs_are_local(pairs):
    for row in pairs:
        request = row["request"]
        rule_ids = {item["id"] for item in request["existing"]}
        evidence_ids = {item["id"] for item in request["evidence"]}
        assert set(row["related_ids"]) <= rule_ids, row["id"]
        assert set(row["evidence_refs"]) <= evidence_ids, row["id"]
        if row["label"] == "uncertain":
            assert row["related_ids"] == row["evidence_refs"] == [], row["id"]


def test_set_rows_shape_and_sizes(sets, manifest):
    assert len(sets) >= 35
    assert len({row["id"] for row in sets}) == len(sets)
    sizes = {}
    for row in sets:
        assert set(row) == ROW_FIELDS, row["id"]
        assert row["label"] in M.RELATIONS, row["id"]
        assert row["label_origin"] == "agent_review", row["id"]
        assert row["split"] == "set", row["id"]
        request = row["request"]
        assert set(request) == REQUEST_FIELDS, row["id"]
        size = len(request["existing"])
        assert size in SET_SIZES, (row["id"], size)
        sizes[size] = sizes.get(size, 0) + 1
        assert request["candidate"]["endpoint_fp"] == CANONICAL_FP, row["id"]
        assert len({item["id"] for item in request["existing"]}) == size, row["id"]
        assert all(0 < len(item["text"]) <= M._MAX_TEXT for item in request["existing"]), row["id"]
        assert row["phenomena"]["rationale"].strip(), row["id"]
    assert set(sizes) == set(SET_SIZES)
    assert {str(key): value for key, value in sizes.items()} == manifest["set_challenge"]["size_counts"]


def test_set_entry_plans_derive_labels_via_production_merge(sets):
    for row in sets:
        candidate, existing, evidence = parts(row)
        size = len(existing)
        phenomena = row["phenomena"]
        entries = expand_plan(phenomena["entry_plan"], size)
        assert set(entries) <= ENTRY_RELATIONS, row["id"]
        assert sorted({name for name in entries if name != "unrelated"}) == \
            phenomena["relational_cases"], row["id"]
        batches = batch_of(candidate, existing, evidence)
        if size == 0:
            assert batches == [] and phenomena["entry_plan"] == [], row["id"]
            assert row["label"] in {"complement", "temporary", "uncertain"}, row["id"]
            assert phenomena.get("acceptable_labels", [row["label"]]) == [row["label"]], row["id"]
            continue
        primaries, acceptables, plans = [], [], []
        cursor = 0
        for batch in batches:
            stop = cursor + len(batch)
            primary, acceptable = batch_policy(entries[cursor:stop])
            primaries.append(primary)
            acceptables.append(acceptable)
            plans.append({"span": [cursor + 1, stop], "primary": primary,
                          "acceptable": acceptable})
            cursor = stop
        assert cursor == size, row["id"]
        assert plans == phenomena["batch_plan"], row["id"]
        assert row["label"] == merge(primaries), row["id"]
        if len(set(primaries)) > 1:
            assert "merge_policy" in phenomena, row["id"]
        outcomes = {merge(list(combo)) for combo in product(*acceptables)}
        outcomes.add(row["label"])
        stored = phenomena.get("acceptable_labels", [row["label"]])
        assert stored == sorted(outcomes), row["id"]
        assert row["label"] in stored, row["id"]
        if row["label"] != "uncertain":
            expected_related = [item["id"] for item, rel in zip(existing, entries)
                                if rel not in ("unrelated", "uncertain")]
            assert row["related_ids"] == expected_related, row["id"]


def test_set_scale_variants_share_group_id(sets):
    by_id = {row["id"]: row for row in sets}
    assert by_id["set-006"]["group_id"] == by_id["set-012"]["group_id"]
    assert by_id["set-015"]["group_id"] == by_id["set-025"]["group_id"]
    groups = [row["group_id"] for row in sets]
    assert len(groups) == len(set(groups)) + 2, "只有两组共享 group_id"


def test_set_rows_pass_production_projection(sets, manifest):
    for row in sets:
        candidate, existing, evidence = parts(row)
        batches = batch_of(candidate, existing, evidence)
        if len(existing) > M._MAX_ROWS:
            assert len(batches) >= 2, row["id"]
        assert context_of(candidate, existing, evidence) == row["request"]["context"], row["id"]
        rule_ids = {item["id"] for item in existing}
        evidence_ids = {item["id"] for item in evidence}
        assert set(row["related_ids"]) <= rule_ids, row["id"]
        assert set(row["evidence_refs"]) <= evidence_ids, row["id"]
        if row["label"] == "uncertain":
            assert row["related_ids"] == row["evidence_refs"] == [], row["id"]


def test_cross_file_fingerprints_unique(pairs, sets):
    seen = {}
    for row in list(pairs) + list(sets):
        value = fingerprint(row)
        assert value not in seen, (seen.get(value), row["id"])
        seen[value] = row["id"]


def test_set_conflict_positions_and_mixed_families(sets, manifest):
    positions = {"first": [], "middle": [], "last": []}
    mixed = []
    for row in sets:
        size = len(row["request"]["existing"])
        entries = expand_plan(row["phenomena"]["entry_plan"], size) if size else []
        relevant = sorted({name for name in entries if name != "unrelated"})
        if len([name for name in relevant if name in M.RELATIONS]) >= 2:
            mixed.append(row["id"])
        for span in row["phenomena"]["entry_plan"]:
            if span["relation"] != "conflict":
                continue
            start, end = span["span"]
            bucket = "first" if start == 1 else "last" if end == size else "middle"
            positions[bucket].append(f'{row["id"]}@{start}')
    assert positions["first"] and positions["middle"] and positions["last"]
    assert len(mixed) >= 3, mixed
    assert manifest["set_challenge"]["conflict_positions"] == positions


def test_manifest_freezes_data_and_discloses(pairs, sets, manifest, spec):
    pairs_bytes = CHALLENGE_PATH.read_bytes()
    sets_bytes = SET_PATH.read_bytes()
    assert hashlib.sha256(pairs_bytes).hexdigest() == manifest["pair_challenge"]["sha256"]
    assert hashlib.sha256(sets_bytes).hexdigest() == manifest["set_challenge"]["sha256"]
    assert hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() == manifest["spec_sha256"]
    assert manifest["legacy_spec_sha256"] == "602015f70f89bbab7349eaa76d5a98843cb27332928cb3c2c53ee8593b712c31"
    assert manifest["evaluation_role"] == "exposed-diagnostic"
    assert manifest["independent_terminal_test"] is False
    assert manifest["schema_field_count"] == spec["schema_field_count"] == len(ROW_FIELDS) == 9
    assert manifest["compatible_splits"] == spec["splits"]
    assert manifest["label_origin"] == "agent_review"
    assert manifest["human_gold"] is False
    assert "不是真人" in manifest["human_gold_note"]
    assert manifest["external_or_private_data"] is False
    assert manifest["classes"] == sorted(M.RELATIONS)
    class_counts = {}
    for row in pairs:
        class_counts[row["label"]] = class_counts.get(row["label"], 0) + 1
    assert manifest["pair_challenge"]["class_counts"] == class_counts
    assert manifest["pair_challenge"]["rows"] == len(pairs)
    assert manifest["set_challenge"]["rows"] == len(sets)
    disclosures = "".join(manifest["disclosures"])
    for needle in ("填料池", "循环组合", "endpoint_fp", "非全行重复", "group_id"):
        assert needle in disclosures
    policy = json.dumps(manifest["label_policy"], ensure_ascii=False)
    assert "未使用任何模型" in policy
    assert "不构成语义 gold" in policy and "主标签" in policy


def test_no_secret_like_material_in_rows():
    for path in (CHALLENGE_PATH, SET_PATH):
        text = path.read_text(encoding="utf-8")
        assert not find_secret_patterns(text), path.name
        assert "[REDACTED" not in text, path.name


def test_agent_review_rows_are_not_promoted(pairs, sets, spec):
    assert spec["safety_metrics"]["promotion"].startswith("never")
    for row in list(pairs) + list(sets):
        assert row["label_origin"] in spec["label_origins"]
        assert row["label_origin"] != "human_anchor"
