                       
"""研究数据构造器专项测试：契约回放、切分纯净、feature 边界与机械契约探针。

标签是构造语义而不是人工金标；这里只断言可机械复核的性质（生产投影/输出闸回放、
组与模板族不跨 split、feature 面不含标识符与权限位、机械拒绝不冒充语义标签）。
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset_recommender.agent import knowledge              
from dataset_recommender.agent import memory_relations as M              
from dataset_recommender.app.conversation_runs import RunError              
from dataset_recommender.llm import dream              
from research.memory_relations import build_data as B              
from research.memory_relations import schema as S              

SPEC = json.loads((Path(__file__).resolve().parent / "spec.json").read_text(encoding="utf-8"))
PLAN = {"train": 32, "calibration": 16, "heldout": 16}
                                                       
PLAN_WITH_DISTRACTORS = {"train": 96, "calibration": 40, "heldout": 40}
FILES = ("all.jsonl", "pairs.jsonl", "train.jsonl", "calibration.jsonl", "heldout.jsonl",
         "audit.json", "hashes.json")
                                             
MAIN_DATA = Path(os.environ.get("MEMORY_RELATIONS_MAIN_DATA",
                                Path(__file__).resolve().parent / "data"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network or unconfigured LLM call in offline research test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(M, "call_llm", denied)


@pytest.fixture
def corpus():
    return B.build_corpus(20261011, PLAN)


def _probe_request() -> dict:
    return {
        "candidate": {"text": "汇报先给结论，再给依据", "trigger": "汇报时", "tier": "explicit",
                      "permanence": "persistent", "evidence": ["汇报先给结论，再给依据"],
                      "scope": "project"},
        "existing": [{"id": "r1", "text": "汇报先说结论，再解释依据", "trigger": "汇报时",
                      "tier": "explicit", "permanence": "persistent", "evidence": [],
                      "scope": "same"}],
        "evidence": [{"id": "e1", "text": "以后汇报先给结论，再给依据。"}],
        "context": {"scope": "project"},
    }


def _assess_with(request: dict, decision: dict) -> dict:
    candidate, existing, evidence, cfg = S.materialize_request(request)
    return M.assess_relations(candidate, existing, evidence, cfg,
                              predictor=lambda _request: dict(decision))


def test_schema_constants_match_shared_spec_and_production_class_set():
    assert S.CLASSES is M.RELATIONS
    assert sorted(S.CLASSES) == sorted(SPEC["classes"])
    assert list(S.FIELDS) == SPEC["schema_fields"]
    assert sorted(S.LABEL_ORIGINS) == sorted(SPEC["label_origins"])
    assert sorted(S.REQUEST_FIELDS) == sorted(SPEC["request_fields"])
    assert S.features_from_request is S.feature_view


def test_schema_split_parity_with_shared_spec():
    assert sorted(S.SPLITS) == sorted(SPEC["splits"])
    from research.memory_relations import finetune, train
    assert set(train.SPLITS) == set(finetune.SPLITS) == set(S.SPLITS)


def test_permanence_vocabulary_is_the_production_vocabulary():
    for value in sorted(S.PERMANENCE):
        body = knowledge.normalize_body({"kind": "memory", "text": "示例要求",
                                         "permanence": value})
        assert body["permanence"] == value
    with pytest.raises(RunError):
        knowledge.normalize_body({"kind": "memory", "text": "示例要求", "permanence": "long_term"})


def test_module_cli_quick_build_is_deterministic(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "src")]))
    targets = []
    for name in ("cli-a", "cli-b"):
        result = subprocess.run([sys.executable, "-m", "research.memory_relations.build_data",
                                 "--out", str(tmp_path / name), "--seed", "20261011", "--quick"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        targets.append(tmp_path / name)
    for name in FILES:
        assert (targets[0] / name).read_bytes() == (targets[1] / name).read_bytes()


def test_splits_are_group_disjoint_balanced_and_family_pure(corpus):
    rows = corpus["rows"]
    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    family_splits = {}
    for group in corpus["seed_doc"]["groups"]:
        family_splits.setdefault(group["family"], set()).add(group["split"])
    assert all(len(splits) == 1 for splits in family_splits.values())
    for split in B.SPLIT_ORDER:
        counts = collections.Counter(row["label"] for row in rows if row["split"] == split)
        assert set(counts) == set(S.CLASSES)
        assert set(counts.values()) == {PLAN[split]}


def test_feature_view_projects_only_allowed_semantics(corpus):
    row = next(row for row in corpus["rows"] if row["label"] == "supersede")
    view = S.feature_view(row["request"])
    keys = set()

    def collect(value):
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(view)
    assert keys == {"candidate", "existing", "evidence", "text", "trigger"}
    text = json.dumps(view, ensure_ascii=False)
    for identifier in (row["id"], row["group_id"], "r1", "e1"):
        assert identifier not in text


def test_feature_view_ignores_row_metadata_and_scope(corpus):
    row = next(row for row in corpus["rows"] if row["label"] == "contextual")
    baseline = S.feature_view(row["request"])
    stripped = json.loads(json.dumps(row))
    stripped.update(id="other-id", group_id="other-group", split="heldout",
                    label="duplicate", label_origin="agent_review", phenomena=["other"],
                    related_ids=[], evidence_refs=[])
    assert S.feature_view(stripped["request"]) == baseline
    rescoped = json.loads(json.dumps(row["request"]))
    for rule in rescoped["existing"]:
        rule["scope"] = "inherited_global"
    assert S.feature_view(rescoped) == baseline


def test_rows_replay_the_real_gate(corpus):
    for row in corpus["rows"]:
        S.validate_row(row)
        candidate, existing, evidence, cfg = S.materialize_request(row["request"])
        expected = {"relation": row["label"], "related_ids": list(row["related_ids"]),
                    "reason": "回放探针", "evidence_refs": list(row["evidence_refs"]),
                    "error_code": None}
        out = M.assess_relations(candidate, existing, evidence, cfg,
                                 predictor=lambda _request: dict(expected))
        if row["label"] == "uncertain":
            assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
            assert out["related_ids"] == [] and out["evidence_refs"] == []
        else:
            assert out["error_code"] is None
            assert (out["relation"], out["related_ids"], out["evidence_refs"]) == \
                (row["label"], row["related_ids"], row["evidence_refs"])


def test_candidate_quotes_are_grounded_under_production_normalization(corpus):
    for row in corpus["rows"]:
        candidate, _existing, evidence, _cfg = S.materialize_request(row["request"])
        for quote in candidate["evidence"]:
            assert any(dream._evidence_checks([quote], [(0, dream._norm(item["text"]))])[0]
                       for item in evidence), row["id"]


def test_empty_existing_rows_carry_grounded_evidence(corpus):
    for label in ("complement", "temporary", "uncertain"):
        rows = [row for row in corpus["rows"]
                if row["split"] == "train" and row["label"] == label
                and not row["request"]["existing"]]
        assert rows
        for row in rows:
            assert row["related_ids"] == []
            candidate = row["request"]["candidate"]
            assert candidate["evidence"]
            for quote in candidate["evidence"]:
                assert any(dream._norm(quote) in dream._norm(item["text"])
                           for item in row["request"]["evidence"])


def test_uncertain_and_supersede_rows_keep_semantic_prerequisites(corpus):
    for row in corpus["rows"]:
        candidate = row["request"]["candidate"]
        if row["label"] == "uncertain":
            assert candidate["tier"] == "explicit"
            assert candidate["permanence"] == "unknown"
            assert not row["related_ids"] and not row["evidence_refs"]
        if row["label"] == "supersede":
            assert candidate["tier"] == "explicit"
            assert all(rule["scope"] == "same" for rule in row["request"]["existing"])
            assert row["related_ids"] == ["r1"]


def test_mechanical_contract_cases_fail_closed_without_becoming_labels():
    request = _probe_request()
    replacement = {"relation": "supersede", "related_ids": ["r1"], "reason": "机械契约探针",
                   "evidence_refs": ["e1"], "error_code": None}
    assert _assess_with(request, replacement)["error_code"] is None
    implicit = json.loads(json.dumps(request))
    implicit["candidate"]["tier"] = "implicit"
    assert _assess_with(implicit, replacement)["error_code"] == "replacement_requires_explicit"
    inherited = json.loads(json.dumps(request))
    inherited["existing"][0]["scope"] = "inherited_global"
    assert _assess_with(inherited, replacement)["error_code"] == "scope_replacement_forbidden"
    contextual = {"relation": "contextual", "related_ids": ["r1"], "reason": "机械契约探针",
                  "evidence_refs": ["e1"], "error_code": None}
    assert _assess_with(inherited, contextual)["error_code"] is None
    unknown = {"relation": "duplicate", "related_ids": ["r9"], "reason": "机械契约探针",
               "evidence_refs": ["e1"], "error_code": None}
    assert _assess_with(request, unknown)["error_code"] == "unknown_reference"
    candidate, existing, evidence, cfg = S.materialize_request(request)
    foreign = M.assess_relations(candidate, [dict(existing[0], principal="other-principal")],
                                 evidence, cfg,
                                 predictor=lambda _request: dict(replacement))
    assert foreign["error_code"] == "scope_mismatch"
    drifted = S.local_config()
    drifted.base_url = "https://different.example/v1"
    mismatch = M.assess_relations(candidate, existing, evidence, drifted,
                                  predictor=lambda _request: dict(replacement))
    assert mismatch["error_code"] == "endpoint_mismatch"
    ungrounded = M.assess_relations(dict(candidate, evidence=["不存在的引文"]), existing,
                                    evidence, cfg,
                                    predictor=lambda _request: dict(replacement))
    assert ungrounded["error_code"] == "unverified_evidence"
                                                         
    complementary = {"relation": "complement", "related_ids": [], "reason": "机械契约探针",
                     "evidence_refs": ["e1"], "error_code": None}
    assert _assess_with(inherited, complementary)["error_code"] is None


def _probe_row() -> dict:
    return {"id": "probe-row", "group_id": "probe-group", "split": "train",
            "request": _probe_request(), "label": "duplicate",
            "label_origin": "construction", "related_ids": ["r1"], "evidence_refs": ["e1"],
            "phenomena": ["paraphrase_restatement"]}


def _raw_challenge_like_request() -> dict:
    return {
        "candidate": {"principal": "user-a", "endpoint_fp": "c27e1bc81a50",
                      "project_id": "proj-alpha", "text": "提交前跑一遍格式化，行宽控制在100字符以内。",
                      "trigger": "提交代码时", "tier": "explicit",
                      "evidence": ["以后提交代码前都先格式化一遍，行宽别超过100个字符。"]},
        "existing": [{"principal": "user-a", "endpoint_fp": "c27e1bc81a50",
                      "project_id": "proj-alpha", "id": "r1",
                      "text": "提交前运行格式化，保持行宽不超过100列。", "trigger": "提交代码时",
                      "tier": "explicit", "evidence": []}],
        "evidence": [{"id": "e1", "text": "你记一下：以后提交代码前都先格式化一遍，行宽别超过100个字符。"}],
        "context": {"scope": "project", "replaceable_ids": ["r1"], "inherited_global_ids": []},
    }


def test_canonical_request_unifies_raw_synthetic_and_desensitized_shapes():
    canonical = S.canonical_request(_raw_challenge_like_request())
    assert set(canonical) == {"candidate", "existing", "evidence", "context"}
    assert canonical["context"] == {"scope": "project"}
    assert canonical["candidate"]["scope"] == "project"
    assert canonical["existing"][0]["scope"] == "same"
    for key in S.IDENTITY_FIELDS:
        assert key not in canonical["candidate"] and key not in canonical["existing"][0]
    assert S.canonical_request(canonical) == canonical      
    candidate, existing, _evidence, _cfg = S.materialize_request(_raw_challenge_like_request())
    assert candidate["principal"] == S.SYNTH_PRINCIPAL
    assert candidate["project_id"] == S.SYNTH_PROJECT and existing[0]["project_id"] == S.SYNTH_PROJECT


def test_canonical_request_drops_permissions_and_rejects_unknown_keys():
    canonical = S.canonical_request(_raw_challenge_like_request())
    loose = _raw_challenge_like_request()
    loose["context"]["replaceable_ids"] = []
    loose["context"]["inherited_global_ids"] = ["r1"]
    assert S.canonical_request(loose) == canonical               
    for mutate, code in ((lambda r: r["context"].update(expected=["x"]), "invalid_context"),
                         (lambda r: r["context"].update(label="duplicate"), "invalid_context"),
                         (lambda r: r["candidate"].update(label="duplicate"), "invalid_rule")):
        broken = _raw_challenge_like_request()
        mutate(broken)
        with pytest.raises(S.RowError) as failure:
            S.canonical_request(broken)
        assert failure.value.code == code


def test_raw_project_ids_map_to_scope_labels():
    request = _raw_challenge_like_request()
    request["existing"][0]["project_id"] = ""
    assert S.canonical_request(request)["existing"][0]["scope"] == "inherited_global"
    foreign = _raw_challenge_like_request()
    foreign["candidate"]["project_id"] = ""
    foreign["context"]["scope"] = "global"
    with pytest.raises(S.RowError) as failure:
        S.canonical_request(foreign)
    assert failure.value.code == "scope_mismatch"


def test_phenomena_accepts_list_or_bounded_dict():
    S.validate_row(_probe_row())
    challenge_like = _probe_row()
    challenge_like["phenomena"] = {
        "domain": "code_style", "mechanism": ["equivalent_rephrase"],
        "entry_plan": [{"span": [1, 64], "relation": "unrelated"}], "rationale": "同一要求换词"}
    S.validate_row(challenge_like)
    bad_relation = json.loads(json.dumps(challenge_like))
    bad_relation["phenomena"]["pair_labels"] = [{"rule_id": "r1", "relation": "activate"}]
    with pytest.raises(S.RowError):
        S.validate_row(bad_relation)
    too_deep = _probe_row()
    too_deep["phenomena"] = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    with pytest.raises(S.RowError):
        S.validate_row(too_deep)


def test_chunked_validation_handles_more_than_one_batch():
    request = _probe_request()
    template = request["existing"][0]
    request["existing"] = [dict(template, id=f"r{index:03d}",
                                text=f"汇报先给结论再补依据（第{index}条）。", evidence=[])
                           for index in range(1, 130)]
    request["existing"][-1] = dict(template, id="r129", text="汇报先说结论，再解释依据。")
    row = _probe_row()
    row["request"] = request
    row["related_ids"] = ["r129"]
    row["phenomena"] = {"construction": ["large_set"],
                        "batch_plan": [{"span": [1, 64], "primary": "complement"}]}
    candidate, existing, evidence, cfg = S.materialize_request(request)
    batches = M._size_fit_batches(candidate, existing, evidence, None, False)
    assert len(batches) >= 2
    S.validate_row(row)
    forced = M.assess_relations(candidate, existing, evidence, cfg, predictor=lambda batch: {
        "relation": "duplicate", "related_ids": ["r129"], "reason": "把全 set 引用硬塞给每批",
        "evidence_refs": ["e1"], "error_code": None})
    assert forced["error_code"] == "unknown_reference"                  


def test_pair_rows_follow_per_rule_labels_and_candidate_status():
    corpus = B.build_corpus(20261011, PLAN_WITH_DISTRACTORS, repair_pair_labels=True)
    rows, pairs = corpus["rows"], corpus["pairs"]
    by_row = collections.defaultdict(list)
    for pair in pairs:
        by_row[pair["phenomena"]["row_id"]].append(pair)
    for row in rows:
        row_pairs = by_row[row["id"]]
        rules = row["request"]["existing"]
        expected_ids = sorted(rule["id"] for rule in rules) or ["empty"]
        assert sorted(pair["phenomena"]["rule_id"] for pair in row_pairs) == expected_ids
        relations = {item["rule_id"]: item["relation"] for item in row["phenomena"]["pair_labels"]}
        for pair in row_pairs:
            S.validate_row(pair)
            assert pair["split"] == row["split"] and pair["group_id"] == row["group_id"]
            if pair["phenomena"]["pair_origin"] == "empty_sentinel":
                assert not pair["request"]["existing"] and pair["label"] == row["label"]
                continue
            rule_id = pair["phenomena"]["rule_id"]
            expected = row["label"] if row["label"] in ("temporary", "uncertain") else relations[rule_id]
            assert pair["label"] == expected
            if pair["phenomena"]["pair_origin"] == "distractor":
                assert rule_id not in row["related_ids"]
                assert pair["label"] == (row["label"] if row["label"] in (
                    "temporary", "uncertain") else "complement")
            else:
                assert pair["label"] == row["label"]
                if pair["label"] == "uncertain":
                    assert pair["related_ids"] == [] and pair["evidence_refs"] == []
                else:
                    assert rule_id in row["related_ids"]
    origins = collections.Counter(pair["phenomena"]["pair_origin"] for pair in pairs)
    assert origins["target"] and origins["distractor"] and origins["redundant"] \
        and origins["empty_sentinel"]
    assert corpus["legacy_label_audit"]
    assert all(item["original_label"] == "complement" and item["repaired_label"] in (
        "temporary", "uncertain") for item in corpus["legacy_label_audit"])


def test_real_challenge_head_materializes_under_contract():
    path = MAIN_DATA / "challenge.jsonl"
    if not path.exists():
        pytest.skip(f"main-tree challenge data not available: {path}")
    row = S.load_rows(path)[0]
    assert set(row) == set(S.FIELDS)
    S.validate_row(row)
    candidate, existing, evidence, _cfg = S.materialize_request(row["request"])
    projection, _meta = M._request(candidate, existing, evidence, None)
    visible = {"text", "trigger", "tier", "evidence", "scope"}
    assert set(projection["candidate"]) == visible
    assert set(projection["existing"][0]) == visible | {"id"}
                                                            
                                                             
    for key in S.IDENTITY_FIELDS:
        assert key not in projection["candidate"] and key not in projection["existing"][0]


def test_real_challenge_files_validate_and_large_sets_use_batches():
    for name in ("challenge.jsonl", "set_challenge.jsonl"):
        path = MAIN_DATA / name
        if not path.exists():
            pytest.skip(f"main-tree challenge data not available: {path}")
        rows = S.load_rows(path)
        failures = []
        for row in rows:
            try:
                S.validate_row(row)
            except S.RowError as exc:
                failures.append((row["id"], str(exc)))
        assert not failures, failures[:3]
        large = [row for row in rows if len(row["request"]["existing"]) >= 65]
        if name == "set_challenge.jsonl":
            assert large
            for row in large:
                candidate, existing, evidence, _cfg = S.materialize_request(row["request"])
                assert len(M._size_fit_batches(candidate, existing, evidence, None, False)) >= 2


def test_v0_artifacts_still_validate_under_v1_schema():
    v0 = ROOT / "research" / "memory_relations" / "runs" / "v0-data" / "all.jsonl"
    if not v0.exists():
        pytest.skip("v0 artifacts not present")
    rows = S.load_rows(v0)
    assert rows
    for row in rows[:3]:
        S.validate_row(row)                                          


def test_written_files_round_trip_with_audit_hashes(corpus, tmp_path):
    out = tmp_path / "out"
    B.write_outputs(corpus, out)
    rows = S.load_rows(out / "all.jsonl")
    assert rows == corpus["rows"]
    assert S.load_rows(out / "pairs.jsonl") == corpus["pairs"]
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["seed"] == 20261011 and audit["label_origin"] == "construction"
    assert audit["construction_version"] == "v1"
    assert audit["seed_catalog_sha256"] == hashlib.sha256(B.SEED_PATH.read_bytes()).hexdigest()
    for name, digest in audit["hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert audit["checks"]["rows_contract_replayed"] == len(rows)
    assert audit["checks"]["pairs_contract_replayed"] == len(corpus["pairs"])
    assert audit["pairs"]["total"] == len(corpus["pairs"])
    assert audit["counts"]["train"] == {cls: PLAN["train"] for cls in sorted(S.CLASSES)}
    assert all(item["a_b_exclusive"] and item["s_compatible"] and item["c1_c2_exclusive"]
               and item["note"] for item in audit["group_semantics"])
    assert sorted(audit["label_recipes"]) == sorted(S.CLASSES)


def test_legacy_default_quick_hashes_are_unchanged():
    corpus = B.build_corpus(20261011, B.PLAN_QUICK)
    assert hashlib.sha256(B._jsonl_bytes(corpus["rows"])).hexdigest() == \
        "e99e46e8839fe7e8e7249f5e8142c6300b7e98255f8f6b6b25f99c8acd0310cd"
    assert hashlib.sha256(B._jsonl_bytes(corpus["pairs"])).hexdigest() == \
        "478bf2ecb25bacaf4c8159a02446e4585e75a0aebaeff0d381160d2c6dcc058c"
    assert "legacy_label_audit" not in corpus


def test_legacy_relabel_is_explicit_and_outputs_original_and_repaired_labels(tmp_path):
    original = B.build_corpus(20261011, PLAN_WITH_DISTRACTORS)
    repaired = B.build_corpus(20261011, PLAN_WITH_DISTRACTORS, repair_pair_labels=True)
    assert original["rows"] == repaired["rows"]
    assert len(original["pairs"]) == len(repaired["pairs"])
    assert len(repaired["legacy_label_audit"]) > 0
    old_by_id = {row["id"]: row for row in original["pairs"]}
    new_by_id = {row["id"]: row for row in repaired["pairs"]}
    for audit in repaired["legacy_label_audit"]:
        old, new = old_by_id[audit["id"]], new_by_id[audit["id"]]
        assert old["label"] == audit["original_label"] == "complement"
        assert new["label"] == audit["repaired_label"] in ("temporary", "uncertain")
        assert old["request"] == new["request"]
        S.validate_row(new)
    out = tmp_path / "repaired"
    B.write_outputs(repaired, out)
    assert S.load_rows(out / "legacy_labels.jsonl") == repaired["legacy_label_audit"]
    report = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert report["construction_version"] == "v1-candidate-status-relabel"
    assert report["legacy_relabel_count"] == len(repaired["legacy_label_audit"])
    with pytest.raises(B.BuildError, match="output_exists"):
        B.write_outputs(original, out)


def test_validate_row_rejects_contract_violations(corpus):
    source = next(row for row in corpus["rows"] if row["label"] == "duplicate")
    for mutate in (
            lambda row: row.update(split="test"),
            lambda row: row.update(evidence_refs=["e9"]),
            lambda row: row["request"]["candidate"].update(evidence=["不存在的引文"]),
            lambda row: row.update(label="supersede", related_ids=[]),
            lambda row: row["request"].update(context={"scope": "global"}),
    ):
        broken = json.loads(json.dumps(source))
        mutate(broken)
        with pytest.raises(S.RowError):
            S.validate_row(broken)
    mechanical = json.loads(json.dumps(source))
    mechanical["request"]["candidate"]["tier"] = "implicit"
    mechanical["label"] = "supersede"
    with pytest.raises(S.RowError):
        S.validate_row(mechanical)


def test_gate_replay_is_contract_only_not_semantic(corpus):
                                           
    source = next(row for row in corpus["rows"]
                  if row["label"] == "duplicate" and row["request"]["candidate"]["tier"] == "explicit")
    relabelled = json.loads(json.dumps(source))
    relabelled["label"] = "conflict"
    S.validate_row(relabelled)


def test_load_rows_fails_closed_on_broken_line(tmp_path):
    target = tmp_path / "broken.jsonl"
    target.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(S.RowError) as failure:
        S.load_rows(target)
    assert failure.value.code == "invalid_json_line:2"
