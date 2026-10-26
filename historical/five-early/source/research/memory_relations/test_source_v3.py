"""作者装配正反探针：合成代理标签、真实输入复用、根隔离和封存，不训练或触网。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

import pytest

from research.memory_relations import build_data as B
from research.memory_relations import features as F
from research.memory_relations import schema as S


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("unexpected network/teacher call")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(B.M, "call_llm", denied)


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_shard(path, rows):
    path.write_bytes(B._jsonl_bytes(rows))


def _row(entry, index, *, label=None, variant="operating"):
    if label is None:
        label = ("duplicate" if entry["operating_allow"] else "conflict") \
            if entry["split"] == "calibration" else F.CLASSES[index % len(F.CLASSES)]
    obj = f"第{index}份模拟材料"
    text = obj + "用表格列出结论"
    quote = obj + "用表格列出结论"
    if label == "temporary":
        quote = "仅这一次" + quote + "，长期约定不变"
    elif label == "uncertain":
        quote = "可能" + quote + "，适用期限还未决定"
    elif label == "supersede":
        quote = "以后" + quote + "，原段落格式永久取消"
    elif label == "conflict":
        quote = obj + "只用段落，不用表格"
        text = obj + "只用段落"
    elif label == "complement":
        quote = obj + "再添加来源链接，格式不变"
        text = obj + "添加来源链接"
    trigger = "对外报告时" if label != "contextual" else "内部草稿时"
    request = {"candidate": {"text": text, "trigger": trigger,
                              "tier": "explicit", "evidence": [quote], "scope": "project"},
               "existing": [{"id": "r1", "text": obj + "用表格列出结论",
                             "trigger": "对外报告时", "tier": "explicit", "evidence": [],
                             "scope": "same"}],
               "evidence": [{"id": "e1", "text": "模拟发话：" + quote + "。"}],
               "context": {"scope": "project"}}
    return {"id": str(index), "group_id": entry["root_id"], "split": entry["split"],
            "request": S.canonical_request(request), "label": label, "label_origin": "agent_review",
            "related_ids": [] if label == "uncertain" else ["r1"],
            "evidence_refs": [] if label == "uncertain" else ["e1"],
            "phenomena": {"root_id": entry["root_id"], "domain": "writing",
                          "family": "format_condition_time", "author": entry["author"],
                          "rationale": "模拟原话中的对象、条件与时间作为依据，仅作机械合同探针。",
                          "variant": variant}}


@pytest.fixture
def author_data(tmp_path):
    roots = {f"a{index}": [f"a{index}-g{root:02d}" for root in range(20)] for index in range(5)}
    roots.update({"h0": [f"h0-g{root:02d}" for root in range(20)],
                  "o0": [f"o0-g{root:02d}" for root in range(20)]})
    assignments = B.preallocate_author_roots(roots, sealed_authors={"h0": "heldout", "o0": "ood"})
    protocol = json.loads(B.SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["status"] = "preallocated-before-authoring"
    protocol["root_assignments"] = assignments
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    rows = [_row(entry, 10000 + index) for index, entry in enumerate(assignments)]
    shard_paths = []
    for author in sorted(roots):
        path = tmp_path / f"{author}.jsonl"
        _write_shard(path, [row for row in rows if row["phenomena"]["author"] == author])
        shard_paths.append(path)
    return {"rows": rows, "assignments": assignments, "protocol": protocol,
            "protocol_path": protocol_path, "shards": shard_paths, "tmp": tmp_path}


def _assemble(data, rows=None):
    if rows is None:
        return B.build_author_shards(data["shards"], data["protocol_path"])
    path = data["tmp"] / "mutated.jsonl"
    _write_shard(path, rows)
    return B.build_author_shards([path], data["protocol_path"])


def test_schema_dev_and_metadata_are_additive():
    entry = {"root_id": "a-g01", "author": "a", "split": "dev"}
    row = _row(entry, 123, label="duplicate")
    S.validate_row(row)
    assert set(row) == set(S.FIELDS) and len(S.FIELDS) == 9
    for key, bad in (("root_id", ""), ("rationale", ""), ("author", []), ("variant", None),
                     ("operating_primary", 1), ("calibration_role", "train")):
        broken = copy.deepcopy(row)
        broken["phenomena"][key] = bad
        with pytest.raises(S.RowError):
            S.validate_row(broken)
    row["phenomena"]["calibration_role"] = "prob"
    with pytest.raises(S.RowError, match="invalid_calibration_metadata"):
        S.validate_row(row)


def test_preallocation_matches_target_arithmetic_without_labels():
    roots = {f"a{i:02d}": [f"a{i:02d}-g{n:02d}" for n in range(20)] for i in range(20)}
    sealed = {f"a{i:02d}": "heldout" if i < 18 else "ood" for i in range(16, 20)}
    dev = {f"a{i:02d}": 2 if i < 8 else 3 for i in range(16)}
    allocated = B.preallocate_author_roots(roots, dev_roots=dev, sealed_authors=sealed)
    assert allocated == B.preallocate_author_roots(dict(reversed(list(roots.items()))),
                                                  dev_roots=dev, sealed_authors=sealed)
    assert {split: sum(entry["split"] == split for entry in allocated)
            for split in B.TRAINABLE_SPLITS + B.SEALED_SPLITS} == \
        {"train": 240, "dev": 40, "calibration": 40, "heldout": 40, "ood": 40}
    cal = [entry for entry in allocated if entry["split"] == "calibration"]
    assert sum(entry["operating_allow"] for entry in cal) == 24
    assert {entry["calibration_role"] for entry in cal} == set(B.CALIBRATION_ROLES)
    assert all(entry["primary_variant"] == "operating" for entry in cal)
    with pytest.raises(B.BuildError, match="source_allocation"):
        B.preallocate_author_roots({"a": ["a-g01"]})


def test_template_is_targets_not_existing_data():
    template = json.loads(B.SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert template["status"] == "targets-not-generated" and template["human_gold_available"] is False
    assert template["root_assignments"] == []
    assert template["row_contract"]["fields"] == list(S.FIELDS)
    with pytest.raises(B.BuildError, match="missing_root_preallocation"):
        B.load_source_protocol()


def test_source_assembly_calibration_counts_and_replay(author_data):
    result = _assemble(author_data)
    assert len(result["training_rows"]) == 100 and len(result["sealed_rows"]) == 40
    assert not result["quarantine_rows"]
    audit = result["report"]
    assert audit["rows_contract_replayed"] == 140
    assert audit["semantics_verified"] is False and audit["human_gold_available"] is False
    assert all(check["violations"] == 0 for check in audit["checks"].values())
    assert audit["calibration"]["planned"] == audit["calibration"]["accepted"] == \
        {"total": 15, "allow": 9, "block": 6, "allow_fraction": .6}
    assert all(part["primary"]["total"] == 5 for part in audit["calibration"]["roles"].values())
    assert not audit["calibration"]["is_user_distribution"]
    assert audit["counts"]["train"]["evidence_sources"] == 75
    assert sum(audit["counts"]["train"]["labels"].values()) == 75
    for row in result["training_rows"] + result["sealed_rows"]:
        S.validate_row(row)
    assert "operating_primary" not in author_data["rows"][0]["phenomena"]


def test_nonprimary_variants_retain_root_role(author_data):
    rows = copy.deepcopy(author_data["rows"])
    original = next(row for row in rows if row["split"] == "calibration")
    entry = next(entry for entry in author_data["assignments"]
                 if entry["root_id"] == original["group_id"])
    variant = _row(entry, 20001, label="uncertain", variant="time_undecided")
    rows.append(variant)
    result = _assemble(author_data, rows)
    members = [row for row in result["training_rows"] if row["group_id"] == original["group_id"]]
    assert len(members) == 2 and sum(row["phenomena"]["operating_primary"] for row in members) == 1
    assert {row["phenomena"]["calibration_role"] for row in members} == {entry["calibration_role"]}


@pytest.mark.parametrize("which,code", [("root", "root_assignment_mismatch"),
                                       ("group", "group_split_leakage"),
                                       ("exact", "cross_split_input_fingerprint"),
                                       ("normalized", "cross_split_input_fingerprint")])
def test_cross_split_leakage_is_rejected(author_data, which, code):
    rows = copy.deepcopy(author_data["rows"])
    left = next(row for row in rows if row["split"] == "train" and row["label"] == "duplicate")
    right = next(row for row in rows if row["split"] == "dev")
    if which == "root":
        right["phenomena"]["root_id"] = left["phenomena"]["root_id"]
    elif which == "group":
        right["group_id"] = left["group_id"]
    else:
        right["request"] = copy.deepcopy(left["request"])
        right["label"], right["related_ids"], right["evidence_refs"] = "duplicate", ["r1"], ["e1"]
        if which == "normalized":
            for rule in [right["request"]["candidate"]] + right["request"]["existing"]:
                rule["text"] = " " + rule["text"] + " "
    with pytest.raises(B.BuildError, match=code):
        _assemble(author_data, rows)


def test_same_student_input_multi_label_conflict_including_permissions(author_data):
    rows = copy.deepcopy(author_data["rows"])
    original = next(row for row in rows if row["split"] == "train" and row["label"] == "duplicate")
    broken = copy.deepcopy(original)
    broken.update(id="23000", label="conflict")
    broken["request"]["candidate"]["tier"] = "implicit"
    broken["request"]["existing"][0]["scope"] = "inherited_global"
    broken["phenomena"]["variant"] = "contradictory_label"
    assert F.request_text(broken["request"]) == F.request_text(original["request"])
    assert F.request_view(broken["request"]) == F.request_view(original["request"])
    rows.append(broken)
    with pytest.raises(B.BuildError, match="student_label_conflict"):
        _assemble(author_data, rows)


@pytest.mark.parametrize("mutation,code", [
    (lambda row: row.update(id="123-duplicate"), "author_pair_contract"),
    (lambda row: row.update(label_origin="construction"), "author_pair_contract"),
    (lambda row: row["request"]["existing"].append(dict(row["request"]["existing"][0], id="r2")),
     "author_pair_contract"),
    (lambda row: row["request"]["candidate"].update(principal="synthetic"), "noncanonical_author_request"),
    (lambda row: row.update(evidence_refs=["e9"]), "row_contract_replay"),
    (lambda row: row["request"]["candidate"].update(evidence=["不存在的引文"]), "row_contract_replay"),
    (lambda row: row["phenomena"].pop("rationale"), "row_contract_replay"),
])
def test_author_contract_violations_fail_not_relabel(author_data, mutation, code):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["label"] == "duplicate" and row["split"] == "train")
    mutation(row)
    with pytest.raises(B.BuildError, match=code):
        _assemble(author_data, rows)


def test_permission_failure_is_not_semantic_uncertain(author_data):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["label"] == "supersede" and row["split"] == "train")
    row["request"]["existing"][0]["scope"] = "inherited_global"
    with pytest.raises(B.BuildError) as failure:
        _assemble(author_data, rows)
    assert failure.value.code == "row_contract_replay"
    assert "scope_replacement_forbidden" in str(failure.value)
    assert row["label"] == "supersede"


@pytest.mark.parametrize("state", ["pending", "disputed", "unresolved", "quarantined"])
def test_unadjudicated_rows_quarantined_without_becoming_uncertain(author_data, state):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["label"] == "duplicate" and row["split"] == "train")
    row["phenomena"]["adjudication_status"] = state
    result = _assemble(author_data, rows)
    assert len(result["quarantine_rows"]) == 1
    assert result["quarantine_rows"][0]["label"] == "duplicate"
    assert result["quarantine_rows"][0]["phenomena"]["quarantine_reason"] == "unadjudicated"
    assert len(result["training_rows"]) == 99


@pytest.mark.parametrize("change,reason", [("missing", "missing_accepted_operating_primary"),
                                         ("pending", "missing_accepted_operating_primary"),
                                         ("label", "primary_label_preallocation_mismatch")])
def test_operating_primary_never_reselected_from_labels(author_data, change, reason):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["split"] == "calibration" and row["label"] == "duplicate")
    root = row["group_id"]
    extra = copy.deepcopy(row)
    extra["id"] = "22000"
    extra["phenomena"]["variant"] = "other_allowed"
    extra["request"]["evidence"][0]["text"] += " 后续仍保持这项要求。"
    rows.append(extra)
    if change == "missing":
        rows.remove(row)
    elif change == "pending":
        row["phenomena"]["adjudication_status"] = "pending"
    else:
        row["label"] = "conflict"
    result = _assemble(author_data, rows)
    assert not any(item["group_id"] == root for item in result["training_rows"])
    assert result["report"]["quarantine"]["roots"][root] == reason
    assert all(item["phenomena"]["quarantine_reason"] == reason
               for item in result["quarantine_rows"] if item["group_id"] == root)
    assert result["report"]["calibration"]["accepted"]["total"] == 14


def test_fingerprint_reuses_feature_functions_and_ignores_hidden_metadata(author_data, monkeypatch):
    original = author_data["rows"][0]
    called = {"view": 0, "text": 0}
    view, text = F.request_view, F.request_text
    def spy_view(request):
        called["view"] += 1
        return view(request)
    def spy_text(request):
        called["text"] += 1
        return text(request)
    monkeypatch.setattr(F, "request_view", spy_view)
    monkeypatch.setattr(F, "request_text", spy_text)
    first = B.input_fingerprints(original["request"])
    request = copy.deepcopy(original["request"])
    request["candidate"].update(tier="implicit", permanence="unknown")
    request["existing"][0]["scope"] = "inherited_global"
    second = B.input_fingerprints(request)
    assert called["view"] >= 2 and called["text"] == 2
    assert first["canonical_request"] != second["canonical_request"]
    for key in ("student_view", "student_text", "normalized_view", "normalized_text"):
        assert first[key] == second[key]


def test_near_neighbors_are_candidates_not_semantic_certificate(author_data):
    result = _assemble(author_data)
    near = result["report"]["near_neighbor_candidates"]
    assert near["comparisons"] and near["candidate_pairs"]
    assert near["exhaustive"] is False and near["semantic_no_leakage_certified"] is False
    assert near["review_required"] is True
    assert all("left_text" not in example and "right_text" not in example for example in near["examples"])


def test_trainable_manifest_cannot_leak_sealed_or_quarantine(author_data):
    result = _assemble(author_data)
    out, sealed = author_data["tmp"] / "trainable", author_data["tmp"] / "sealed"
    B.write_author_outputs(result, out, sealed)
    assert {path.name for path in out.iterdir()} == \
        {"train.jsonl", "dev.jsonl", "calibration.jsonl", "pairs.jsonl", "audit.json", "manifest.json"}
    pairs = S.load_rows(out / "pairs.jsonl")
    assert {row["split"] for row in pairs} == set(B.TRAINABLE_SPLITS)
    assert len(S.load_rows(sealed / "heldout.jsonl")) == len(S.load_rows(sealed / "ood.jsonl")) == 20
    for directory in (out, sealed):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    trainer_text = (out / "manifest.json").read_text(encoding="utf-8") + \
        (out / "audit.json").read_text(encoding="utf-8")
    assert "heldout" not in trainer_text and '"ood"' not in trainer_text
    assert "h0" not in trainer_text and "o0" not in trainer_text
    old = (out / "manifest.json").read_bytes()
    with pytest.raises(B.BuildError, match="output_exists"):
        B.write_author_outputs(result, out, sealed)
    assert (out / "manifest.json").read_bytes() == old


def test_output_paths_must_be_physically_disjoint(author_data):
    result = _assemble(author_data)
    out = author_data["tmp"] / "target"
    for sealed in (out, out / "sealed", out.parent):
        with pytest.raises(B.BuildError, match="sealed_output_overlap"):
            B.write_author_outputs(result, out, sealed)
    assert not out.exists()


def test_protocol_rejects_reallocated_roles_and_mixed_test_authors(author_data):
    protocol = copy.deepcopy(author_data["protocol"])
    entry = next(entry for entry in protocol["root_assignments"] if entry["author"] == "h0")
    entry["split"] = "train"
    _write_json(author_data["protocol_path"], protocol)
    with pytest.raises(B.BuildError, match="sealed_author_mixed_splits"):
        _assemble(author_data)


def test_cli_author_assembly_and_non_overwrite(author_data):
    out, sealed = author_data["tmp"] / "cli-train", author_data["tmp"] / "cli-sealed"
    command = [sys.executable, "-B", "-m", "research.memory_relations.build_data", "--mode",
               "author-shards", "--source-protocol", str(author_data["protocol_path"]),
               "--out", str(out), "--sealed-out", str(sealed)]
    for shard in author_data["shards"]:
        command += ["--shard", str(shard)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(B.ROOT / "src"), str(B.ROOT))))
    first = subprocess.run(command, cwd=B.ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["rows_total"] == 100
    second = subprocess.run(command, cwd=B.ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert second.returncode == 2 and json.loads(second.stderr)["error"] == "output_exists"


def test_same_split_duplicate_input_cannot_inflate_independent_roots(author_data):
    rows = copy.deepcopy(author_data["rows"])
    left, right = [row for row in rows if row["split"] == "train" and row["label"] == "duplicate"][:2]
    right["request"] = copy.deepcopy(left["request"])
    with pytest.raises(B.BuildError, match="cross_root_duplicate_input"):
        _assemble(author_data, rows)


@pytest.mark.parametrize("label", ["temporary", "uncertain"])
def test_empty_existing_does_not_erase_candidate_status(author_data, label):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["split"] == "train" and row["label"] == label)
    row["request"]["existing"] = []
    row["related_ids"] = []
    result = _assemble(author_data, rows)
    actual = next(item for item in result["training_rows"] if item["id"] == row["id"])
    assert actual["label"] == label
    S.validate_row(actual)


@pytest.mark.parametrize("field,bad,code", [
    ("calibration_role", "audit", "role_preallocation_mismatch"),
    ("operating_primary", False, "primary_preallocation_mismatch"),
])
def test_author_flags_cannot_override_root_preallocation(author_data, field, bad, code):
    rows = copy.deepcopy(author_data["rows"])
    row = next(row for row in rows if row["split"] == "calibration" and
               next(entry for entry in author_data["assignments"]
                    if entry["root_id"] == row["group_id"])["calibration_role"] == "prob")
    row["phenomena"][field] = bad
    with pytest.raises(B.BuildError, match=code):
        _assemble(author_data, rows)
