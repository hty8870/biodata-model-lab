"""五叶数据离线机械门；oracle 与作者代理不冒充语言人类金标。"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_recommender.agent import semantic_relations as S
from research.memory_relations import semantic_data as D

V = lambda name: ["var", name]
BOOL = {name: {"values": [False, True], "exhaustive": True} for name in ("a", "b")}
TEACHER = {"id": "stub-blind", "provider": "offline-stub", "model": "stub", "family": "stub", "revision": "1"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("offline data test attempted network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def model(p=None, h=None):
    return {"mode": "assertion", "domains": copy.deepcopy(BOOL), "domain_condition": True,
            "old": V("a") if p is None else p, "new": V("b") if h is None else h}


def row(identifier="root-a", *, split="pilot", label="overlap_incomparable", oracle=None):
    result = {"id": identifier + "-primary", "root_id": identifier, "cluster_id": identifier,
              "split": split, "label_space": "semantic5", "label": label,
              "pair": {"frame": {"mode": "same_domain", "background": "陈述真值域：a与b独立二值且非空。"},
                       "old": {"kind": "memory", "text": "甲已确认。", "trigger": "这次确认时", "permanence": "temporary"},
                       "new": {"kind": "memory", "text": "乙已确认。", "trigger": "这次确认时", "permanence": "temporary"}},
              "label_origin": "author_proxy", "oracle_relation": None,
              "NL_fidelity_origin": {"kind": "author_self_checked", "independent": False, "human": False},
              "adjudication_origin": {"kind": "none", "independent": False, "human": False},
              "resolution_status": "resolved",
              "domain": {"consistent": True, "nonempty": True, "old_nonempty": True, "new_nonempty": True, "basis": "author_proxy"},
              "phenomena": {"family": "negation_polarity", "language": "zh", "mode": "assertion",
                            "scenario": identifier, "skeleton": identifier, "source_ids": [identifier + "-source"],
                            "author": {"role": "author", "human": False, "independent": False}}}
    if oracle is not None:
        certificate = D.oracle_certificate(oracle)
        result.update(label=certificate["relation"], oracle_relation=certificate["relation"],
                      label_origin="author_controlled_oracle",
                      oracle={"model": oracle, "certificate": certificate, "mapping_notes": {"a": "甲确认", "b": "乙确认"}})
    return result


@pytest.mark.parametrize("p,h,expected", [
    (V("a"), V("a"), "equivalent"),
    (["and", V("a"), V("b")], V("a"), "old_entails_new_strict"),
    (V("a"), ["and", V("a"), V("b")], "new_entails_old_strict"),
    (V("a"), ["not", V("a")], "disjoint"),
    (V("a"), V("b"), "overlap_incomparable"),
])
def test_known_five_labels_and_reverse_witnesses(p, h, expected):
    original = model(p, h)
    cert = D.oracle_certificate(original)
    assert cert["relation"] == expected
    assert cert["regions"]["old"]["witness"] is not None
    assert cert["regions"]["new"]["witness"] is not None
    reverse = copy.deepcopy(original)
    reverse["old"], reverse["new"] = h, p
    assert D.oracle_certificate(reverse)["relation"] == S.swap_relation(expected)


def test_numeric_infinite_tail_is_not_clipped_at_five():
    integer = {"mode": "assertion", "domains": {"n": {"integer": {"minimum": 0, "maximum": None}, "unit": "条"}},
               "domain_condition": True, "old": ["ge", V("n"), 5], "new": ["eq", V("n"), 5]}
    cert = D.oracle_certificate(integer)
    assert cert["relation"] == "new_entails_old_strict"
    assert cert["regions"]["old_only"]["witness"] == {"n": 6}
    assert cert["abstraction"]["n"]["cells"][-1]["maximum"] is None
    integer["domains"]["n"]["integer"]["maximum"] = 5
    assert D.oracle_certificate(integer)["relation"] == "equivalent"
    clipped = row(oracle=integer)
    with pytest.raises(D.SemanticDataError, match="finite_numeric_domain_without_textual_boundary"):
        D.validate_row(clipped)


@pytest.mark.parametrize("mutation,code", [
    (lambda m: m["domains"]["a"].update(values=[]), "empty_or_nonexhaustive_domain"),
    (lambda m: m.update(domain_condition=False), "empty_common_domain"),
    (lambda m: m.update(old=False), "empty_denotation"),
    (lambda m: m["domains"]["a"].update(exhaustive=False), "empty_or_nonexhaustive_domain"),
])
def test_empty_or_unproved_domain_rejected(mutation, code):
    m = model()
    mutation(m)
    with pytest.raises(D.SemanticDataError, match=code):
        D.oracle_certificate(m)


def test_complete_condition_and_shared_activation():
    m = model(["implies", V("a"), V("b")], V("b"))
    assert D.oracle_certificate(m)["relation"] == "new_entails_old_strict"
    m["domain_condition"] = V("a")
    assert D.oracle_certificate(m)["relation"] == "equivalent"


def test_pair_single_source_preserves_trigger_permanence_order_and_full_args():
    r = row()
    r["pair"]["old"]["payload"] = {"steps": [
        {"verb": "read", "args": {"permission": "readonly", "source_id": "demo", "version": 3,
                                  "scope": ["b", "a"], "project": "demo-project", "gold": "functional-key"}},
        {"verb": "export", "args": {"fields": ["title", "summary"]}}]}
    D.validate_row(r)
    text = D.student_text(r)
    assert json.loads(text) == S.validate_semantic_pair(r["pair"])
    assert '"trigger":"这次确认时"' in text and '"permanence":"temporary"' in text
    assert '"permission":"readonly"' in text and '"gold":"functional-key"' in text
    assert [s["verb"] for s in json.loads(text)["old"]["payload"]["steps"]] == ["read", "export"]
    assert r["root_id"] not in text and "oracle" not in text and "label_origin" not in text


def test_coarse_loss_mask_does_not_reverse_invent_fine():
    r = row(label="entailment")
    r.update(label_space="nli3", label_origin="official_coarse")
    sup = D.supervision(r)
    assert sup == {"fine_targets": [None, None], "nli_targets": ["entailment", None],
                   "loss_mask": [1, 0], "inverse_qualification": {"status": "unverified"}}
    r["oracle_relation"] = "equivalent"
    with pytest.raises(D.SemanticDataError, match="coarse_must_not_invent_fine"):
        D.validate_row(r)
    for parent in S.NLI_CLASSES:
        r.update(label=parent, oracle_relation=None)
        assert D.supervision(r)["fine_targets"] == [None, None]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(label="duplicate"),
    lambda r: r["NL_fidelity_origin"].update(human=True),
    lambda r: r["domain"].update(nonempty=False),
    lambda r: r.update(resolution_status="unresolved"),
    lambda r: r["pair"]["old"].update(proof="author says so"),
])
def test_bad_labels_origins_and_nonempty_boundary(mutation):
    r = row()
    mutation(r)
    with pytest.raises((D.SemanticDataError, S.SemanticRelationError)):
        D.validate_row(r)


def test_unresolved_has_own_table_and_zero_loss(tmp_path):
    r = row(label=None)
    r["resolution_status"] = "ambiguous"
    r["label_origin"] = "author_unresolved"
    r["phenomena"]["nonresolved_reason"] = "指代未消歧。"
    assert D.supervision(r)["loss_mask"] == [0, 0]
    out = tmp_path / "out"
    summary = D.write_dataset([r], out)
    assert summary["counts"]["nonresolved_rows"] == 1
    assert D.load_rows(out / "resolved.jsonl") == []
    assert len(D.load_rows(out / "nonresolved.jsonl")) == 1
    assert D.load_rows(out / "student.jsonl") == []


@pytest.mark.parametrize("link", ["inverse", "source", "skeleton", "root"])
def test_cross_split_inverse_and_lineage_rejected(link):
    a, b = row("alpha", split="train"), row("beta", split="dev")
    b["pair"]["new"]["text"] = "另一条独立事实。"
    if link == "inverse":
        b["pair"] = S.swap_pair(a["pair"])
        b["label"] = S.swap_relation(a["label"])
    elif link == "root":
        b["root_id"] = a["root_id"]
    else:
        key = "source_ids" if link == "source" else "skeleton"
        b["phenomena"][key] = a["phenomena"][key]
    with pytest.raises(D.SemanticDataError, match="cross_split_dependency"):
        D.cluster_rows([a, b], merge=True)


def test_same_skeleton_merges_without_inflating_support():
    a, b = row("alpha"), row("beta")
    b["pair"]["new"]["text"] = "换了名字的同骨架事实。"
    b["phenomena"]["skeleton"] = a["phenomena"]["skeleton"]
    merged, audit = D.cluster_rows([a, b], merge=True)
    assert audit["candidate_root_ids"] == 2 and audit["dependency_clusters"] == 1
    assert merged[0]["cluster_id"] == merged[1]["cluster_id"]
    with pytest.raises(D.SemanticDataError, match="dependency_cluster_not_merged"):
        D.cluster_rows([a, b])


def test_secret_scan_rejects_and_strict_json_no_duplicate_fields():
    r = row()
    r["pair"]["old"]["text"] = "sk-" + "A" * 24
    with pytest.raises(D.SemanticDataError, match="secret_or_oversized_record"):
        D.validate_row(r)
    with pytest.raises(ValueError):
        D.T._strict_json('{"label":"equivalent","label":"disjoint"}')


def test_preallocation_before_authoring_is_900_slots_not_gold():
    spec = D.read_json(D.JOB_SPEC)
    slots = D.preallocate(spec)
    assert len(slots) == 900 and len({s["cluster_id"] for s in slots}) == 180
    counts = {split: sum(s["split"] == split for s in slots) for split in spec["first_wave"]["planned_root_counts"]}
    assert counts == spec["first_wave"]["planned_root_counts"]
    assert not any(s["accepted_root"] for s in slots)
    jobs = D.author_jobs(slots, spec)
    assert len(jobs) == 660
    assert not any("pair" in job or "label" in job for job in jobs)
    assert not set(j["root_id"] for j in jobs) & {s["root_id"] for s in slots if s["sealed"]}


def action():
    return {"before": {"revision": 1, "clauses": ["先写结论", "保留链接"]},
            "after": {"revision": 2, "clauses": ["后写结论", "保留链接"]},
            "proposal": {"operation": "edit", "authorized_paths": ["clauses.0"], "instruction": "只改位置",
                         "patch": {"clauses.0": "后写结论"}},
            "authority": {"authorized": True, "actor": "a", "expected_actor": "a", "project": "p", "expected_project": "p",
                          "endpoint": "e", "expected_endpoint": "e"},
            "source": {"id": "synthetic-source", "expected_version": 1, "observed_version": 1, "fresh": True},
            "cas": {"expected": 1, "observed": 1}, "joint_model": model(True, True)}


def test_action_separate_authority_version_cas_unrelated_and_false_receipt():
    event = action()
    result = D.assess_action(event)
    assert result["action"] == "edit"
    bad = dict(result, preserves_unrelated=False)
    with pytest.raises(D.SemanticDataError, match="false_action_receipt"):
        D.validate_action_receipt(event, bad)
    event["after"]["clauses"].pop()
    assert D.assess_action(event)["reason"] == "unauthorized_clause_change"
    event = action()
    event["source"]["observed_version"] = 2
    assert D.assess_action(event)["action"] == "defer"
    event = action()
    event["cas"]["observed"] = 2
    assert D.assess_action(event)["action"] == "defer"
    event = action()
    event["authority"]["actor"] = "other"
    assert D.assess_action(event)["action"] == "reject"
    event = action()
    event["proposal"]["patch"]["clauses.0"] = "未经提案的内容"
    assert D.assess_action(event)["reason"] == "proposal_patch_mismatch"
    event = action()
    event["source"].pop("expected_version")
    event["source"].pop("observed_version")
    assert D.assess_action(event)["action"] == "defer"
    event = action()
    event["authority"].pop("actor")
    event["authority"].pop("expected_actor")
    assert D.assess_action(event)["action"] == "reject"


def test_joint_three_conflict_not_detectable_as_individual_pair_disjoint():
    event = action()
    clauses = [["or", V("a"), V("b")], ["or", ["not", V("a")], V("b")], ["not", V("b")]]
    for p, h in itertools.combinations(clauses, 2):
        assert D.oracle_certificate(model(p, h))["regions"]["intersection"]["witness"] is not None
    event["joint_model"]["old"] = ["and", *clauses]
    assert D.assess_action(event)["reason"] == "joint_inconsistent"


def blind_fixture(tmp_path, controlled=False):
    source = tmp_path / "source.jsonl"
    r = row(oracle=model()) if controlled else row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "private" / "map.json"
    D.export_blind([r], blind, mapping, teacher=TEACHER, source=source)
    doc = D.read_json(mapping)
    item = doc["items"][0]
    label = S.swap_relation(r["label"]) if item["swapped"] else r["label"]
    reconstruction = copy.deepcopy(r["oracle"]["model"]) if controlled else None
    if controlled and item["swapped"]:
        reconstruction["old"], reconstruction["new"] = reconstruction["new"], reconstruction["old"]
    receipt = {"blind_id": item["entry"]["blind_id"], "input_sha256": item["entry"]["input_sha256"],
               "teacher": TEACHER, "judgement": {"label": label, "resolution_status": "resolved", "reason": "盲读代理判断。",
               "nl_fidelity": True, "complete_semantics": True, "reconstruction": reconstruction}}
    return source, blind, mapping, receipt


@pytest.mark.parametrize("controlled", [False, True])
def test_real_blind_export_import_resume_and_no_gold_in_jobs(tmp_path, controlled):
    source, blind, mapping, receipt = blind_fixture(tmp_path, controlled)
    entry = D.load_rows(blind / "shard-001.jsonl")[0]
    assert set(entry) == {"blind_id", "pair", "instructions", "input_sha256"}
    assert "oracle" not in entry and "root_id" not in entry and "split" not in entry
    ledger = tmp_path / "teacher-ledger.jsonl"
    result = D.import_teacher([receipt], blind, mapping, ledger, run_id="actual-stub-session")
    assert len(result["accepted"]) == 1 and not result["quarantine"]
    assert result["api_calls"] == 0 and result["human_gold"] is False
    assert result["accepted"][0]["label_origin"] == "blind_teacher_proxy"
    before = ledger.read_bytes()
    resumed = D.import_teacher([receipt], blind, mapping, ledger, run_id="actual-stub-session")
    assert resumed["skipped_done"] == 1 and ledger.read_bytes() == before


@pytest.mark.parametrize("mutation", [
    lambda r: r["judgement"].update(label="duplicate"),
    lambda r: r["judgement"].update(nl_fidelity=False),
    lambda r: r["judgement"].update(complete_semantics=False),
    lambda r: r.update(input_sha256="0" * 64),
    lambda r: r["teacher"].update(model="other-model"),
])
def test_bad_and_false_clause_teacher_receipts_quarantined(tmp_path, mutation):
    _, blind, mapping, receipt = blind_fixture(tmp_path)
    mutation(receipt)
    result = D.import_teacher([receipt], blind, mapping, tmp_path / "ledger.jsonl", run_id="stub")
    assert not result["accepted"] and len(result["quarantine"]) == 1


def test_blind_paths_and_shard_tamper_fail_closed(tmp_path):
    source, blind, mapping, receipt = blind_fixture(tmp_path)
    with pytest.raises(ValueError, match="blind_paths_not_separated"):
        D.export_blind([row()], tmp_path / "bad", tmp_path / "bad" / "map.json", teacher=TEACHER, source=source)
    with (blind / "shard-001.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(D.SemanticDataError, match="blind_shard_hash_mismatch"):
        D.import_teacher([receipt], blind, mapping, tmp_path / "ledger.jsonl", run_id="stub")


def test_blind_map_tamper_and_duplicate_receipt_rejected(tmp_path):
    _, blind, mapping, receipt = blind_fixture(tmp_path)
    result = D.import_teacher([receipt, receipt], blind, mapping, tmp_path / "ledger.jsonl", run_id="dup")
    assert result["quarantine"][0]["code"] == "unknown_or_duplicate_blind_id"
    doc = D.read_json(mapping)
    doc["items"][0]["source_row"]["label"] = "disjoint"
    mapping.write_bytes(D.B._json_bytes(doc))
    with pytest.raises(D.SemanticDataError, match="blind_map_hash_mismatch"):
        D.import_teacher([receipt], blind, mapping, tmp_path / "ledger.jsonl", run_id="tamper")


def test_nonresolved_blind_oracle_vote_keeps_null_and_denominator(tmp_path):
    _, blind, mapping, receipt = blind_fixture(tmp_path, controlled=True)
    receipt["judgement"].update(label=None, resolution_status="ambiguous", reason="无法唯一建立语言解释。",
                               nl_fidelity=False, complete_semantics=False, reconstruction=None)
    result = D.import_teacher([receipt], blind, mapping, tmp_path / "ledger.jsonl", run_id="unknown")
    assert len(result["accepted"]) == 1 and result["accepted"][0]["label"] is None
    assert D.supervision(result["accepted"][0])["loss_mask"] == [0, 0]


def test_categorical_name_and_numeric_slot_changes_share_model_skeleton():
    def categorical(values):
        return {"mode": "assertion", "domains": {"category": {"values": values, "exhaustive": True}},
                "domain_condition": True, "old": ["eq", V("category"), values[0]],
                "new": ["or", ["eq", V("category"), values[0]], ["eq", V("category"), values[1]]]}
    assert D._model_skeleton(categorical(["犬", "猫", "鸟"])) == D._model_skeleton(categorical(["苹果", "梨", "松"]))


def test_author_import_real_slot_and_final_guard():
    slots = D.preallocate(D.read_json(D.JOB_SPEC))
    slot = slots[0]
    r = row(slot["root_id"], split=slot["split"])
    r["cluster_id"] = slot["cluster_id"]
    r["phenomena"]["family"] = slot["family"]
    result = D.import_authors([r], slots)
    assert len(result["accepted"]) == 1 and not result["quarantine"]
    sealed = next(s for s in slots if s["sealed"])
    r = row(sealed["root_id"], split=sealed["split"])
    r["cluster_id"] = sealed["cluster_id"]
    r["phenomena"]["family"] = sealed["family"]
    result = D.import_authors([r], slots)
    assert result["quarantine"][0]["code"] == "unauthorized_author_slot"


def test_blind_export_default_canonical_keeps_source_orientation(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "map.json"
    result = D.export_blind([r], blind, mapping, teacher=TEACHER, source=source)
    doc = D.read_json(mapping)
    assert result["swap_policy"] == doc["swap_policy"] == "canonical"
    assert result["swapped_rows"] == 0 and doc["export_rule"] == D.EXPORT_RULE
    item = doc["items"][0]
    assert item["swapped"] is False and item["swap_basis"] == "canonical"
    assert D.digest(item["entry"]["pair"]) == D.digest(r["pair"])


def test_reciprocal_empty_frame_swaps_without_evidence(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    r["pair"]["frame"]["background"] = ""
    r["label"] = None
    r["resolution_status"] = "ambiguous"
    r["phenomena"]["nonresolved_reason"] = "作者未决。"
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "map.json"
    D.export_blind([r], blind, mapping, teacher=TEACHER, source=source, swap_policy="reciprocal")
    item = D.read_json(mapping)["items"][0]
    assert item["swapped"] is True and item["swap_basis"] == "empty_frame"
    assert D.digest(item["entry"]["pair"]) == D.digest(S.swap_pair(r["pair"]))


def test_reciprocal_role_bound_background_defaults_to_no_swap_or_fails_closed(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "map.json"
    with pytest.raises(D.SemanticDataError, match="unproven_swap_safety"):
        D.export_blind([r], blind, mapping, teacher=TEACHER, source=source, swap_policy="reciprocal")
    assert not blind.exists() and not mapping.exists()


def test_reciprocal_role_invariant_and_explicit_inverse_evidence_paths(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "map.json"
    D.export_blind([r], blind, mapping, teacher=TEACHER, source=source, swap_policy="reciprocal",
                   swap_evidence=[{"id": r["id"], "kind": "role_invariant",
                                   "evidence": "作者声明背景与角色无关。"}])
    item = D.read_json(mapping)["items"][0]
    assert item["swapped"] is True and item["swap_basis"] == "role_invariant"
    assert "swap_evidence_sha256" in item

    blind2, mapping2 = tmp_path / "blind2", tmp_path / "map2.json"
    inverse = S.swap_pair(r["pair"])
    D.export_blind([r], blind2, mapping2, teacher=TEACHER, source=source, swap_policy="reciprocal",
                   swap_evidence=[{"id": r["id"], "kind": "explicit_inverse",
                                   "evidence": "作者逐字给出逆向 pair。", "inverse_pair": inverse}])
    item2 = D.read_json(mapping2)["items"][0]
    assert item2["swapped"] is True and item2["swap_basis"] == "explicit_inverse"
    assert D.digest(item2["entry"]["pair"]) == D.digest(inverse)


def test_reciprocal_rejects_mismatched_or_unbound_evidence(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    cases = [
        ([{"id": r["id"], "kind": "explicit_inverse", "evidence": "x",
           "inverse_pair": copy.deepcopy(r["pair"])}], "inverse_pair_not_exact_swap"),
        ([{"id": r["id"], "kind": "role_invariant", "evidence": " "}], "missing_swap_evidence_text"),
        ([{"id": "root-zzz", "kind": "role_invariant", "evidence": "x"}], "unknown_swap_evidence_id"),
        ([{"id": r["id"], "kind": "not_a_kind", "evidence": "x"}], "invalid_swap_evidence_kind"),
    ]
    for index, (evidence, code) in enumerate(cases):
        with pytest.raises(D.SemanticDataError, match=code):
            D.export_blind([r], tmp_path / f"b{index}", tmp_path / f"m{index}.json",
                           teacher=TEACHER, source=source, swap_policy="reciprocal",
                           swap_evidence=evidence)
    with pytest.raises(D.SemanticDataError, match="swap_evidence_without_reciprocal_policy"):
        D.export_blind([r], tmp_path / "b4", tmp_path / "m4.json", teacher=TEACHER, source=source,
                       swap_evidence=[{"id": r["id"], "kind": "role_invariant", "evidence": "x"}])


def test_swap_policy_bound_to_map_and_import_fingerprint(tmp_path):
    source = tmp_path / "source.jsonl"
    r = row()
    source.write_bytes(D.B._jsonl_bytes([r]))
    blind, mapping = tmp_path / "blind", tmp_path / "map.json"
    D.export_blind([r], blind, mapping, teacher=TEACHER, source=source)
    doc = D.read_json(mapping)
    assert doc["swap_policy"] == "canonical" and doc["swap_evidence_sha256"] is None
    doc["swap_policy"] = "reciprocal"
    mapping.write_bytes(D.B._json_bytes(doc))
    with pytest.raises(D.SemanticDataError, match="blind_map_hash_mismatch"):
        D.import_teacher([], blind, mapping, tmp_path / "ledger.jsonl", run_id="tamper")


def role_bound_row():
    r = row()
    r["pair"]["frame"]["background"] = "陈述真值世界。旧句不禁止附录，不附录也可满足旧句。"
    return r


def inverse_qualification(r, **overrides):
    record = {"id": r["id"], "kind": "explicit_inverse", "inverse_pair": S.swap_pair(r["pair"]),
              "source_row_sha256": D.digest(r), "review_kind": "data_side_inverse_review",
              "review_record_sha256": "a" * 64}
    record.update(overrides)
    return record


def test_semantic5_reverse_masked_without_bound_qualification():
    r = role_bound_row()
    sup = D.supervision(r)
    assert sup["fine_targets"] == [r["label"], None]
    assert sup["nli_targets"] == [S.SEMANTIC_TO_NLI[r["label"]], None]
    assert sup["loss_mask"] == [1, 0]
    assert sup["inverse_qualification"] == {"status": "unverified"}


def test_inverse_qualification_positive_yields_dual_targets():
    r = role_bound_row()
    record = inverse_qualification(r)
    sup = D.supervision(r, qualification=record)
    assert sup["loss_mask"] == [1, 1]
    assert sup["fine_targets"] == [r["label"], S.swap_relation(r["label"])]
    assert sup["inverse_qualification"]["status"] == "inverse_qualified"
    assert sup["inverse_qualification"]["format"] == D.INVERSE_QUALIFICATION_FORMAT
    assert sup["inverse_qualification"]["evidence_sha256"] == D.digest(record)
    assert sup["inverse_qualification"]["basis"] == "mechanical_bounds_and_data_side_declaration"


def test_inverse_qualification_pseudo_and_drift_rejected():
    r = role_bound_row()
    cases = [
        (inverse_qualification(r, kind="role_invariant"), "invalid_inverse_qualification_kind"),
        (inverse_qualification(r, review_record_sha256="x"), "invalid_inverse_review_record_sha256"),
        (inverse_qualification(r, source_row_sha256="0" * 64), "inverse_qualification_source_drift"),
        (inverse_qualification(r, inverse_pair=copy.deepcopy(r["pair"])), "inverse_pair_not_exact_swap"),
        (inverse_qualification(r, review_kind="bad kind!"), "invalid_identifier"),
    ]
    for record, code in cases:
        with pytest.raises(D.SemanticDataError, match=code):
            D.supervision(r, qualification=record)
    extra_key = inverse_qualification(r)
    extra_key["extra"] = "x"
    with pytest.raises(D.SemanticDataError, match="invalid_inverse_qualification"):
        D.supervision(r, qualification=extra_key)
    missing_frame = inverse_qualification(r, inverse_pair={"old": "x", "new": "y"})
    with pytest.raises(D.SemanticDataError, match="invalid_inverse_pair"):
        D.supervision(r, qualification=missing_frame)


def test_inverse_qualification_binds_to_exact_row_revision():
    r = role_bound_row()
    record = inverse_qualification(r)
    drifted = copy.deepcopy(r)
    drifted["pair"]["old"]["text"] = drifted["pair"]["old"]["text"] + "改"
    with pytest.raises(D.SemanticDataError, match="inverse_qualification_source_drift"):
        D.supervision(drifted, qualification=record)


def test_write_dataset_threads_qualifications_and_rule(tmp_path):
    qualified = role_bound_row()
    plain = row("root-b")
    out = tmp_path / "data"
    D.write_dataset([qualified, plain], out, qualifications=[inverse_qualification(qualified)])
    audit = D.read_json(out / "audit.json")
    assert audit["supervision_rule"] == D.SUPERVISION_RULE
    assert audit["inverse_qualification_format"] == D.INVERSE_QUALIFICATION_FORMAT
    assert audit["inverse_qualifications"]["qualified_rows"] == 1
    views = {view["id"]: view["supervision"] for view in D.load_rows(out / "student.jsonl")}
    assert views[qualified["id"]]["loss_mask"] == [1, 1]
    assert views[plain["id"]]["loss_mask"] == [1, 0]
    assert views[plain["id"]]["inverse_qualification"] == {"status": "unverified"}
    out2 = tmp_path / "data2"
    D.write_dataset([plain], out2)
    audit2 = D.read_json(out2 / "audit.json")
    assert audit2["inverse_qualifications"] == {"records": 0, "given_sha256": None,
                                                "qualified_rows": 0}
    assert D.load_rows(out2 / "student.jsonl")[0]["supervision"]["loss_mask"] == [1, 0]


def test_write_dataset_rejects_unknown_or_inapplicable_qualifications(tmp_path):
    r = role_bound_row()
    unknown = inverse_qualification(r)
    unknown["id"] = "root-zzz-primary"
    with pytest.raises(D.SemanticDataError, match="unknown_inverse_qualification_id"):
        D.write_dataset([r], tmp_path / "d1", qualifications=[unknown])
    unresolved = row("root-c")
    unresolved.update(label=None, resolution_status="ambiguous")
    unresolved["phenomena"]["nonresolved_reason"] = "作者未决。"
    with pytest.raises(D.SemanticDataError, match="inapplicable_inverse_qualification"):
        D.write_dataset([unresolved], tmp_path / "d2",
                        qualifications=[inverse_qualification(unresolved)])


def _bind_scope(document):
    document["scope_sha256"] = D.digest({key: value for key, value in document.items() if key != "scope_sha256"})
    return document


@pytest.fixture
def construction(tmp_path):
    run = tmp_path / "synthetic-construction-run"
    allocation = run / "sealed" / "plan" / "slots.jsonl"
    allocation.parent.mkdir(parents=True)
    slots = [{"root_id": "fixture-" + name, "cluster_id": "fixture-" + cluster,
              "family": "negation_polarity", "split": split,
              "content_status": "unwritten_slot", "accepted_root": False, "sealed": True, "primary": True}
             for name, cluster, split in (("a", "ab", "blind_final"), ("b", "ab", "blind_final"),
                                          ("c", "c", "blind_final"), ("d", "d", "ood_final"))]
    allocation.write_bytes(D.B._jsonl_bytes(slots))
    members = [{key: item[key] for key in ("root_id", "cluster_id", "family", "split")} for item in slots]
    document = _bind_scope({"schema": D.CONSTRUCTION_SCOPE_FORMAT, "run_id": run.name,
                           "splits": ["blind_final", "ood_final"],
                           "allocation": {"path": "sealed/plan/slots.jsonl",
                                          "sha256": hashlib.sha256(allocation.read_bytes()).hexdigest(),
                                          "members_sha256": D.digest(sorted(members, key=lambda item: item["root_id"]))}})
    scope = run / "construction-scope.json"
    scope.write_bytes(D.B._json_bytes(document))
    return {"run": run, "allocation": allocation, "slots": slots, "scope": scope, "document": document,
            "kwargs": {"construction_scope": scope, "run_id": run.name}}


def _allocated_row(slot):
    result = row(slot["root_id"], split=slot["split"])
    result["cluster_id"] = slot["cluster_id"]
    result["phenomena"]["family"] = slot["family"]
    result["pair"]["old"]["text"] += slot["root_id"]
    return result


def _data_cli(*arguments, expected=0):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    process = subprocess.run([sys.executable, "-B", "-m", "research.memory_relations.semantic_data",
                              *map(str, arguments)], cwd=ROOT, env=env, capture_output=True,
                             text=True, encoding="utf-8", timeout=60)
    assert process.returncode == expected, process.stdout + process.stderr
    return json.loads(process.stderr if process.stderr else process.stdout)


def test_construction_scope_subset_keeps_allocation_and_nonresolved_denominator(construction):
    fixture = construction
    selected = fixture["slots"][:2]
    rows = [_allocated_row(slot) for slot in selected]
    rows[1].update(label=None, resolution_status="ambiguous", label_origin="author_unresolved")
    rows[1]["phenomena"]["nonresolved_reason"] = "合成夹具保留未决项。"
    jobs = D.author_jobs(selected, D.read_json(D.JOB_SPEC), **fixture["kwargs"])
    assert len(jobs) == 2 and jobs[0]["construction_scope"]["run_id"] == fixture["run"].name
    result = D.import_authors(rows, selected, **fixture["kwargs"])
    assert result["accepted"] == rows and result["received"] == result["accepted_count"] == 2
    assert result["audit"]["nonresolved_rows"] == 1 and not result["quarantine"]
    assert D.supervision(result["accepted"][1])["loss_mask"] == [0, 0]
    assert result["audit"]["construction_scope"]["scope_sha256"] == fixture["document"]["scope_sha256"]


def test_construction_scope_does_not_rekey_cross_cluster_dependencies(construction):
    fixture = construction
    selected = [fixture["slots"][0], fixture["slots"][2]]
    rows = [_allocated_row(slot) for slot in selected]
    rows[1]["phenomena"]["source_ids"] = rows[0]["phenomena"]["source_ids"]
    with pytest.raises(D.SemanticDataError, match="dependency_cluster_not_merged"):
        D.import_authors(rows, selected, **fixture["kwargs"])


def test_default_cli_still_skips_or_rejects_sealed(construction):
    fixture = construction
    run = fixture["run"]
    result = _data_cli("preallocate", "--out", run / "default-plan")
    assert result["planned_slots"] == 900 and result["sealed_content_generated"] is False
    assert len(D.load_rows(run / "default-plan" / "author-jobs.jsonl")) == 660
    result = _data_cli("author-jobs", "--allocation", fixture["allocation"], "--out", run / "default-jobs")
    assert result["author_jobs"] == 0
    source = run / "sealed" / "authors.jsonl"
    source.write_bytes(D.B._jsonl_bytes([_allocated_row(fixture["slots"][0])]))
    result = _data_cli("import-authors", "--input", source, "--allocation", fixture["allocation"],
                       "--out", run / "default-import", expected=2)
    assert result["accepted_count"] == 0 and result["quarantine_count"] == 1
    assert result["failed_by_code"] == {"unauthorized_author_slot": 1}
    assert D.load_rows(run / "default-import" / "quarantine.jsonl")[0]["raw_redacted"]
    teacher = run / "identity.json"
    teacher.write_bytes(D.B._json_bytes(TEACHER))
    result = _data_cli("blind-jobs", "--input", source, "--teacher", teacher,
                       "--out", run / "forbidden-blind", "--map", run / "forbidden-map.json", expected=2)
    assert result["code"] == "sealed_content_export_forbidden"
    assert not (run / "forbidden-blind").exists() and not (run / "forbidden-map.json").exists()


@pytest.mark.parametrize("fault,code", [
    ("run", "construction_run_mismatch"),
    ("declaration_hash", "construction_scope_hash_mismatch"),
    ("allocation_hash", "construction_allocation_hash_mismatch"),
    ("members_hash", "construction_members_hash_mismatch"),
    ("split", "invalid_construction_splits"),
    ("extra", "invalid_construction_scope"),
])
def test_construction_scope_declaration_faults_cli(construction, fault, code):
    fixture = construction
    document = copy.deepcopy(fixture["document"])
    if fault == "run":
        document["run_id"] = "another-run"
    elif fault == "allocation_hash":
        document["allocation"]["sha256"] = "0" * 64
    elif fault == "members_hash":
        document["allocation"]["members_sha256"] = "0" * 64
    elif fault == "split":
        document["splits"].append("train")
    elif fault == "extra":
        document["allow_all"] = True
    _bind_scope(document)
    if fault == "declaration_hash":
        document["scope_sha256"] = "0" * 64
    fixture["scope"].write_bytes(D.B._json_bytes(document))
    out = fixture["run"] / "bad-jobs"
    result = _data_cli("author-jobs", "--allocation", fixture["allocation"], "--out", out,
                       "--construction-scope", fixture["scope"], "--run-id", fixture["run"].name, expected=2)
    assert result["code"] == code and not out.exists()


@pytest.mark.parametrize("field,value", [("root_id", "unknown-root"), ("cluster_id", "wrong-cluster"),
                                         ("family", "wrong-family"), ("split", "ood_final")])
def test_construction_member_scope_rejected_by_author_and_export_cli(construction, field, value):
    fixture = construction
    wrong = dict(fixture["slots"][0], **{field: value})
    spec = D.read_json(D.JOB_SPEC)
    with pytest.raises(D.SemanticDataError, match="construction_member_mismatch"):
        D.author_jobs([wrong], spec, **fixture["kwargs"])
    with pytest.raises(D.SemanticDataError, match="construction_member_mismatch"):
        D.import_authors([], [wrong], **fixture["kwargs"])
    r = _allocated_row(wrong)
    source = fixture["run"] / "sealed" / "wrong.jsonl"
    source.write_bytes(D.B._jsonl_bytes([r]))
    teacher = fixture["run"] / "identity.json"
    teacher.write_bytes(D.B._json_bytes(TEACHER))
    out = fixture["run"] / "wrong-blind"
    result = _data_cli("blind-jobs", "--input", source, "--teacher", teacher, "--out", out,
                       "--map", fixture["run"] / "wrong-map.json", "--construction-scope", fixture["scope"],
                       "--run-id", fixture["run"].name, expected=2)
    assert result["code"] == "construction_member_mismatch" and not out.exists()
    result = D.import_authors([r], fixture["slots"], **fixture["kwargs"])
    assert not result["accepted"] and result["failed_by_code"] == {"unauthorized_author_slot": 1}


def test_construction_scope_cli_author_to_teacher_and_inverse_qualification(construction):
    fixture = construction
    run, scope = fixture["run"], fixture["scope"]
    common = ["--construction-scope", scope, "--run-id", run.name]
    selected = [fixture["slots"][0], fixture["slots"][-1]]
    subset = run / "sealed" / "subset.jsonl"
    subset.write_bytes(D.B._jsonl_bytes(selected))
    jobs = run / "sealed" / "author-jobs"
    result = _data_cli("author-jobs", "--allocation", subset, "--out", jobs, *common)
    assert result["author_jobs"] == 2 and result["accepted_roots"] == 0
    assert result["construction_scope"]["members_sha256"] == fixture["document"]["allocation"]["members_sha256"]
    assert all("pair" not in job and "label" not in job for job in D.load_rows(jobs / "author-jobs.jsonl"))
    source = run / "sealed" / "raw-authors.jsonl"
    authored = [_allocated_row(slot) for slot in selected]
    source.write_bytes(D.B._jsonl_bytes(authored))
    imported = run / "sealed" / "imported"
    result = _data_cli("import-authors", "--input", source, "--allocation", subset, "--out", imported, *common)
    assert result["accepted_count"] == 2 and result["missing_slots"] == result["quarantine_count"] == 0
    assert D.load_rows(imported / "all.jsonl") == authored
    assert D.read_json(imported / "manifest.json")["counts"]["construction_scope"]["scope_sha256"] == fixture["document"]["scope_sha256"]
    teacher = run / "identity.json"
    teacher.write_bytes(D.B._json_bytes(TEACHER))
    blind, mapping = run / "sealed" / "blind", run / "sealed" / "private" / "map.json"
    result = _data_cli("blind-jobs", "--input", imported / "all.jsonl", "--out", blind,
                       "--map", mapping, "--teacher", teacher, *common)
    doc = D.read_json(mapping)
    assert result["rows"] == 2 and result["swapped_rows"] == 0
    assert D.load_rows(mapping.with_suffix(".ledger.jsonl"))[0]["construction_scope"] == doc["construction_scope"]
    for entry in D.load_rows(blind / "shard-001.jsonl"):
        assert set(entry) == {"blind_id", "pair", "instructions", "input_sha256"}
        assert not {"label", "split", "root_id", "cluster_id", "family", "teacher", "source_row", "construction_scope"} & set(entry)
    replies = [{"blind_id": item["entry"]["blind_id"], "input_sha256": item["entry"]["input_sha256"],
                "teacher": TEACHER, "judgement": {"label": item["source_row"]["label"], "resolution_status": "resolved",
                "reason": "仅合成机械夹具，不是独立金标。", "nl_fidelity": True, "complete_semantics": True,
                "reconstruction": None}} for item in doc["items"]]
    receipts = run / "sealed" / "raw-teacher.jsonl"
    receipts.write_bytes(D.B._jsonl_bytes(replies))
    ledger = run / "sealed" / "teacher-ledger.jsonl"
    teacher_args = ["import-teacher", "--input", receipts, "--blind", blind, "--map", mapping, "--ledger", ledger]
    result = _data_cli(*teacher_args, "--run-id", run.name, "--out", run / "missing-scope", expected=2)
    assert result["code"] == "sealed_construction_scope_required" and not ledger.exists()
    result = _data_cli(*teacher_args, "--construction-scope", scope, "--run-id", "wrong-run",
                       "--out", run / "wrong-run", expected=2)
    assert result["code"] == "construction_run_mismatch" and not ledger.exists()
    other_scope = run / "other-scope.json"
    other = copy.deepcopy(fixture["document"])
    other["allocation"]["path"] = str(fixture["allocation"])
    other_scope.write_bytes(D.B._json_bytes(_bind_scope(other)))
    result = _data_cli(*teacher_args, "--construction-scope", other_scope, "--run-id", run.name,
                       "--out", run / "different-scope", expected=2)
    assert result["code"] == "blind_construction_scope_mismatch" and not ledger.exists()
    reviewed = run / "sealed" / "reviewed"
    result = _data_cli(*teacher_args, "--out", reviewed, *common)
    assert result["accepted_count"] == 2 and result["quarantine_count"] == result["missing"] == 0
    actual = D.load_rows(reviewed / "all.jsonl")
    assert {r["split"] for r in actual} == {"blind_final", "ood_final"}
    assert all(r["teacher_evidence"]["construction_scope"] == doc["construction_scope"] for r in actual)
    assert all(event["construction_scope"] == doc["construction_scope"] for event in D.load_rows(ledger))
    qualifications = run / "sealed" / "inverse-qualifications.jsonl"
    qualifications.write_bytes(D.B._jsonl_bytes([inverse_qualification(r) for r in actual]))
    before = ledger.read_bytes()
    qualified = run / "sealed" / "qualified"
    result = _data_cli(*teacher_args, "--out", qualified, "--inverse-qualifications", qualifications, *common)
    assert result["skipped_done"] == 2 and ledger.read_bytes() == before
    assert D.read_json(qualified / "audit.json")["inverse_qualifications"]["qualified_rows"] == 2
    for view in D.load_rows(qualified / "student.jsonl"):
        assert view["supervision"]["loss_mask"] == [1, 1]
        assert view["supervision"]["fine_targets"] == [view["label"], S.swap_relation(view["label"])]
    from research.memory_relations import semantic_model as SM
    evidence = _qualified_construction_evidence(run / "sealed", actual, doc["construction_scope"])
    result = _data_cli("qualified-source", "--input", evidence["manifest"], *common)
    assert result["qualified_rows"] == 2 and result["construction_scope"] == doc["construction_scope"]
    assert "rows" not in result and "label" not in result and result["proxy_only"] is True
    bundle = D.load_qualified_source(evidence["manifest"], **fixture["kwargs"])
    assert bundle["rows"] == actual and all(sup["loss_mask"] == [1, 1] for sup in bundle["supervision_by_id"].values())
    result = _data_cli("qualified-source", "--input", evidence["manifest"], expected=2)
    assert result["code"] == "sealed_construction_scope_required"
    result = _data_cli("qualified-source", "--input", evidence["manifest"], "--construction-scope", other_scope,
                       "--run-id", run.name, expected=2)
    assert result["code"] == "qualified_construction_scope_mismatch"
    with pytest.raises(D.SemanticDataError, match="sealed_construction_scope_required"):
        SM.qualified_stage_rows(evidence["manifest"], evidence["allocation"])
    bad = [inverse_qualification(r, source_row_sha256="0" * 64) for r in actual]
    qualifications.write_bytes(D.B._jsonl_bytes(bad))
    result = _data_cli(*teacher_args, "--out", run / "invalid-inverse", "--inverse-qualifications", qualifications,
                       *common, expected=2)
    assert result["code"] == "inverse_qualification_source_drift" and ledger.read_bytes() == before
    assert not (run / "invalid-inverse").exists()
    assert fixture["allocation"].read_bytes() == D.B._jsonl_bytes(fixture["slots"])
    assert D.read_json(scope) == fixture["document"]


@pytest.mark.parametrize("split", ["blind_final", "ood_final"])
@pytest.mark.parametrize("stage", ["fit", "dev"])
def test_construction_scope_does_not_enable_training_stages(construction, split, stage):
    from research.memory_relations import nli_reference as NR
    from research.memory_relations import semantic_model as SM
    context = D._construction_context(**{"scope_path": construction["scope"], "run_id": construction["run"].name})
    r = row(split=split, label="neutral")
    r.update(label_space="nli3", label_origin="official_coarse", construction_scope=context["audit"])
    with pytest.raises(ValueError, match="sealed/final/wrong split"):
        SM.validate_row(r, stage=stage)
    with pytest.raises(ValueError, match="不允许"):
        NR._checked_rows([dict(r, request={})], stage=stage)


def test_qualified_training_source_still_rejects_sealed(construction):
    from research.memory_relations.test_semantic_model import qualified_consumer_fixture
    fixture = qualified_consumer_fixture(construction["run"])
    fixture["source"][0]["split"] = "blind_final"
    fixture["paths"]["source_rows"].write_bytes(D.B._jsonl_bytes(fixture["source"]))
    document = fixture["document"]
    document["assets"]["source_rows"]["sha256"] = hashlib.sha256(fixture["paths"]["source_rows"].read_bytes()).hexdigest()
    fixture["manifest"].write_bytes(D.B._json_bytes(document))
    with pytest.raises(D.SemanticDataError, match="inapplicable_formal_fine_source"):
        D.load_qualified_source(fixture["manifest"])


def _qualified_construction_evidence(directory, actual, scope_audit):
    from research.memory_relations.test_semantic_model import qualified_consumer_fixture
    evidence = qualified_consumer_fixture(directory, canonical=actual)
    evidence["paths"]["source_rows"].write_bytes(D.B._jsonl_bytes(actual))
    by_id = {r["id"]: r for r in actual}
    for record in evidence["qualifications"]:
        record["source_row_sha256"] = D.digest(by_id[record["id"]])
    evidence["paths"]["qualifications"].write_bytes(D.B._jsonl_bytes(evidence["qualifications"]))
    packet = D.read_json(evidence["paths"]["packet_map"])
    for item in packet["items"]:
        item["source_row_sha256"] = D.digest(by_id[item["source_id"]])
    evidence["paths"]["packet_map"].write_bytes(D.B._json_bytes(packet))
    import_doc = D.read_json(evidence["paths"]["import_manifest"])
    import_doc["files_sha256"]["qualifications.jsonl"] = hashlib.sha256(evidence["paths"]["qualifications"].read_bytes()).hexdigest()
    evidence["paths"]["import_manifest"].write_bytes(D.B._json_bytes(import_doc))
    evidence["document"]["construction_scope"] = scope_audit
    for name in ("source_rows", "qualifications", "packet_map", "import_manifest"):
        evidence["document"]["assets"][name]["sha256"] = hashlib.sha256(evidence["paths"][name].read_bytes()).hexdigest()
    evidence["manifest"].write_bytes(D.B._json_bytes(evidence["document"]))
    return evidence


@pytest.mark.parametrize("fault,code", [
    ("cross_cluster", "dependency_cluster_not_merged"),
    ("cross_split", "cross_split_dependency"),
    ("other_run", "teacher_construction_scope_mismatch"),
    ("teacher_run_only", "teacher_construction_scope_mismatch"),
])
def test_construction_qualification_rechecks_assembled_batches(construction, fault, code):
    fixture = construction
    run = fixture["run"]
    selected = [fixture["slots"][0], fixture["slots"][2 if fault == "cross_cluster" else -1]]
    assembled = []
    for number, slot in enumerate(selected):
        r = _allocated_row(slot)
        if fault in ("cross_cluster", "cross_split"):
            r["phenomena"]["source_ids"] = ["shared-across-singleton-batches"]
        imported = D.import_authors([r], [slot], **fixture["kwargs"])
        assert imported["accepted"] == [r] and not imported["quarantine"]
        source = run / "sealed" / f"batch-{number}.jsonl"
        source.write_bytes(D.B._jsonl_bytes(imported["accepted"]))
        blind, mapping = run / "sealed" / f"blind-{number}", run / "sealed" / f"map-{number}.json"
        D.export_blind(imported["accepted"], blind, mapping, teacher=TEACHER, source=source, **fixture["kwargs"])
        item = D.read_json(mapping)["items"][0]
        receipt = {"blind_id": item["entry"]["blind_id"], "input_sha256": item["entry"]["input_sha256"],
                   "teacher": TEACHER, "judgement": {"label": r["label"], "resolution_status": "resolved",
                   "reason": "仅合成机械夹具。", "nl_fidelity": True, "complete_semantics": True, "reconstruction": None}}
        result = D.import_teacher([receipt], blind, mapping, run / "sealed" / f"ledger-{number}.jsonl", **fixture["kwargs"])
        assert result["accepted_count"] == 1 and not result["quarantine"]
        assembled.extend(result["accepted"])
    context = D._construction_context(fixture["scope"], run.name)
    if fault == "teacher_run_only":
        assembled[0]["teacher_evidence"]["run_id"] = "unrelated-run"
    evidence = _qualified_construction_evidence(run / "sealed", assembled, context["audit"])
    kwargs = dict(fixture["kwargs"])
    if fault == "other_run":
        other = copy.deepcopy(fixture["document"])
        other["run_id"] = "another-construction-run"
        other_scope = run / "another-scope.json"
        other_scope.write_bytes(D.B._json_bytes(_bind_scope(other)))
        kwargs = {"construction_scope": other_scope, "run_id": other["run_id"]}
        evidence["document"]["construction_scope"] = D._construction_context(other_scope, other["run_id"])["audit"]
        evidence["manifest"].write_bytes(D.B._json_bytes(evidence["document"]))
    before = {name: path.read_bytes() for name, path in evidence["paths"].items()}
    with pytest.raises(D.SemanticDataError, match=code):
        D.load_qualified_source(evidence["manifest"], **kwargs)
    result = _data_cli("qualified-source", "--input", evidence["manifest"], "--construction-scope",
                       kwargs["construction_scope"], "--run-id", kwargs["run_id"], expected=2)
    assert result["code"] == code
    assert all(path.read_bytes() == before[name] for name, path in evidence["paths"].items())
