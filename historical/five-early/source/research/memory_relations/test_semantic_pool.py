"""semantic_pool 机械转换的正/负 known-answer 测试（CPU-only，不训练，不读真实 run）。

覆盖父指派的负例：缺 members、hash drift、错 label、源改动、fresh overwrite、id rename、
重复 id、跨 split 单位；正例为逐字段已知答案 + 真实消费者（SM.prepare/read_prepared）
端到端 smoke（公开 slice + 合成 fine 行 + 本产 producer-assets.json）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.memory_relations import finetune as FT
from research.memory_relations import semantic_model as SM
from research.memory_relations import semantic_pool as SP

OVERLONG_PREMISE = "长" * 8300 + "tail"


def _record(rid, premise, hypothesis, label, *, dataset="anli", official_split="R1/train",
            group="g", gold_status="single", label_method="writer_only", original_label="e"):
    return {"dataset": dataset, "evidence": {"file": "raw/x.jsonl", "row": 7, "sha256": "ab" * 32,
                                             "source_url": "https://example.invalid/x.zip"},
            "gold_label": label, "group_id": group, "hypothesis": hypothesis, "id": rid,
            "metadata": {"gold_status": gold_status, "label_method": label_method,
                         "original_label": original_label},
            "official_split": official_split, "premise": premise, "task": "nli3"}


def _unit(unit_id, cluster, ids, dataset="anli"):
    return {"unit_id": unit_id, "clusters": [cluster], "datasets": [dataset],
            "record_ids": list(ids), "rows": len(ids)}


def _write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
                    encoding="utf-8", newline="\n")


def _pool_rows():
    return [
        _record("anli:R1/train:e1", "Premise one.", "Hypothesis one.", "entailment"),
        _record("anli:R1/train:n1", OVERLONG_PREMISE, "Hypothesis long.", "neutral"),
        _record("anli:R1/train:c1", "Premise three.", "Hypothesis three.", "contradiction"),
        _record("anli:R1/train:n2", "Premise four.", "Hypothesis four.", "neutral"),
        _record("ocnli:train:d1", "前提一。", "假设一。", "entailment", dataset="ocnli",
                official_split="train.50k"),
        _record("ocnli:train:d2", "前提二。", "假设二。", "neutral", dataset="ocnli",
                official_split="train.50k"),
        _record("ocnli:train:d3", "前提三。", "假设三。", "contradiction", dataset="ocnli",
                official_split="train.50k"),
    ]


TRAIN_IDS = ["anli:R1/train:e1", "anli:R1/train:n1", "anli:R1/train:c1", "anli:R1/train:n2"]
DEV_IDS = ["ocnli:train:d1", "ocnli:train:d2", "ocnli:train:d3"]


def _source(tmp_path, *, pool=None, fit=None, dev=None, units=None, name="run"):
    run = tmp_path / name
    run.mkdir()
    pool = _pool_rows() if pool is None else pool
    fit = list(TRAIN_IDS) if fit is None else fit
    dev = list(DEV_IDS) if dev is None else dev
    units = units if units is not None else [
        _unit("devunit:train1", "cluster:t1", ["anli:R1/train:e1", "anli:R1/train:n1"]),
        _unit("devunit:train2", "cluster:t2", ["anli:R1/train:c1", "anli:R1/train:n2"]),
        _unit("devunit:dev1", "cluster:d1", ["ocnli:train:d1"]),
        _unit("devunit:dev2", "cluster:d2", ["ocnli:train:d2", "ocnli:train:d3"], dataset="ocnli"),
    ]
    _write_jsonl(run / "pool.jsonl", pool)
    _write_jsonl(run / "fit-ids.jsonl", fit)
    _write_jsonl(run / "dev-ids.jsonl", dev)
    _write_jsonl(run / "units.jsonl", units)
    _write_jsonl(run / "dependencies.jsonl", units)
    _write_jsonl(run / "protected.jsonl", [{"datasets": ["ocnli"], "labeled": 1, "missing": 0,
                                            "premise_key": "premise:0000", "rows": 1, "splits": ["ocnli:test"]}])
    return run


def _expected(run):
    return {name: FT.file_sha256(run / name) for name in
            ("pool.jsonl", "fit-ids.jsonl", "dev-ids.jsonl", "units.jsonl", "dependencies.jsonl", "protected.jsonl")}


def _convert(run, tmp_path, **kwargs):
    out = tmp_path / "ownrun"
    manifest = SP.convert(run, out, expected=_expected(run), **kwargs)
    return out, manifest


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_lf(path):
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            assert b"\r\n" not in chunk, path


def _fine_rows(split):
    rows = []
    for index, label in enumerate(SM.SR.SEMANTIC_RELATIONS):
        rows.append({"id": f"fine:{split}:{index}", "root_id": f"fineroot:{split}:{index}",
                     "cluster_id": f"finecluster:{split}:{index}", "split": split,
                     "label_space": "semantic5", "label": label, "label_origin": "mechanics_fixture",
                     "pair": {"old": {"kind": "memory", "text": f"{split}旧命题 {index}"},
                              "new": {"kind": "proposition", "text": f"{split}新命题 {index}"},
                              "frame": {"background": "固定解释域"}}})
    return rows


def test_convert_known_answer_and_readonly_source(tmp_path):
    run = _source(tmp_path)
    before = _expected(run)
    listing = sorted(path.name for path in run.iterdir())
    out, manifest = _convert(run, tmp_path)

    rows = _lines(out / "public-pool.jsonl")
    assert [row["source_id"] for row in rows] == [rid for rid in TRAIN_IDS if rid != "anli:R1/train:n1"] + DEV_IDS
    assert [row["split"] for row in rows] == ["train"] * 3 + ["dev"] * 3
    first = rows[0]
    assert first["id"] == "public:anli:R1/train:e1"
    assert first["label"] == "entailment" and first["label_space"] == "nli3"
    assert first["label_origin"] == "public_official"
    assert first["root_id"] == "devunit:train1" and first["cluster_id"] == "cluster:t1"
    assert first["pair"] == {"frame": {"mode": "same_domain", "background": ""},
                             "old": {"kind": "proposition", "text": "Premise one."},
                             "new": {"kind": "proposition", "text": "Hypothesis one."}}
    assert first["lineage"]["protected_role"] == "protected-fit"
    assert first["lineage"]["pool_sha256"] == before["pool.jsonl"]
    assert first["lineage"]["dataset"] == "anli" and first["lineage"]["gold_status"] == "single"
    assert rows[4]["lineage"]["protected_role"] == "protected-dev"

    quarantine = _lines(out / "source-too-large.jsonl")
    assert [row["source_id"] for row in quarantine] == ["anli:R1/train:n1"]
    item = quarantine[0]
    assert item["reason"] == "source_too_large" and item["windowed_applied"] is False
    assert item["windowed_limit_chars"] == SM.SR.MAX_TEXT_CHARS
    assert item["full"]["premise"]["text"] == OVERLONG_PREMISE
    assert item["full"]["premise"]["chars"] == len(OVERLONG_PREMISE)
    assert item["full"]["premise"]["coverage_if_windowed"] == pytest.approx(8192 / len(OVERLONG_PREMISE))

    assert manifest["counts"]["trainable"] == {"train": 3, "dev": 3, "total": 6}
    quarantine_counts = manifest["counts"]["quarantine"]
    assert (quarantine_counts["train"], quarantine_counts["dev"], quarantine_counts["total"]) == (1, 0, 1)
    assert quarantine_counts["reason"] == "source_too_large"
    assert quarantine_counts["by_dataset"] == {"anli": 1} and quarantine_counts["by_label"] == {"neutral": 1}
    assert manifest["counts"]["conservation"]["fit"] == 4
    assert manifest["source_hashes_after"] == before
    assert _expected(run) == before and sorted(path.name for path in run.iterdir()) == listing

    assets = json.loads((out / "producer-assets.json").read_text(encoding="utf-8"))
    assert assets["schema"] == "semantic-producer-assets-v1"
    roles = {item["role"]: item for item in assets["assets"]}
    assert set(roles) == {"protected-fit", "protected-dev", "protection-manifest"}
    assert roles["protected-fit"]["sha256"] == before["fit-ids.jsonl"]
    assert roles["protected-dev"]["sha256"] == before["dev-ids.jsonl"]
    assert roles["protection-manifest"]["sha256"] == before["protected.jsonl"]
    SM.producer_lineage(out / "producer-assets.json", rows)

    second = tmp_path / "ownrun2"
    SP.convert(run, second, expected=_expected(run))
    assert FT.file_sha256(second / "public-pool.jsonl") == FT.file_sha256(out / "public-pool.jsonl")
    for name in ("public-pool.jsonl", "source-too-large.jsonl", "producer-assets.json",
                 "conversion-manifest.json", "protected-fit.jsonl", "protected-dev.jsonl",
                 "protection-manifest.jsonl"):
        _assert_lf(out / name)
    report = SP.verify(out, run)
    assert report["problems"] == []
    assert report["counts"] == {"trainable": 6, "quarantine": 1, "train": 3, "dev": 3,
                                "quarantine_train": 1, "quarantine_dev": 0}


def test_verify_detects_source_change_and_output_tamper(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    with open(out / "public-pool.jsonl", "a", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n")
    assert "output_hash_drift:public-pool.jsonl" in SP.verify(out, run)["problems"]
    with open(run / "pool.jsonl", "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n")
    problems = SP.verify(out, run)["problems"]
    assert "source_hash_drift:pool.jsonl" in problems


def test_cli_path_and_expected_json(tmp_path):
    run = _source(tmp_path)
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(_expected(run), ensure_ascii=False), encoding="utf-8")
    result = SP.main(["convert", "--source-run", str(run), "--out", str(tmp_path / "cli-out"),
                      "--expected-json", str(expected_path)])
    assert result["counts"]["trainable"]["total"] == 6
    verify_result = SP.main(["verify", "--ownrun", str(tmp_path / "cli-out"), "--source-run", str(run)])
    assert verify_result["problems"] == []


def _mutate_units(run, units):
    _write_jsonl(run / "units.jsonl", units)
    _write_jsonl(run / "dependencies.jsonl", units)


@pytest.mark.parametrize("mutate,code", [
    (lambda run: _write_jsonl(run / "fit-ids.jsonl", TRAIN_IDS[1:]), "pool_member_mismatch"),
    (lambda run: _write_jsonl(run / "fit-ids.jsonl", TRAIN_IDS + ["anli:R1/train:ghost"]), "missing_pool_member"),
    (lambda run: _write_jsonl(run / "pool.jsonl", _pool_rows()[:2] + _pool_rows()[:1] + _pool_rows()[3:]),
     "duplicate_source_id"),
    (lambda run: _write_jsonl(run / "pool.jsonl", [_record("anli:R1/train:renamed", "p", "h", "neutral")] + _pool_rows()[1:]),
     "pool_member_mismatch"),
    (lambda run: _write_jsonl(run / "pool.jsonl", _pool_rows()[:1] + [_record("anli:R1/train:n1", "p", "h", "non-entailment")] + _pool_rows()[2:]),
     "invalid_gold_label"),
    (lambda run: _write_jsonl(run / "pool.jsonl", _pool_rows()[:1] + [_record("anli:R1/train:n1", "p", "h", None)] + _pool_rows()[2:]),
     "invalid_gold_label"),
    (lambda run: _mutate_units(run, [
        _unit("devunit:merged", "cluster:shared", TRAIN_IDS + ["ocnli:train:d1"]),
        _unit("devunit:dev2", "cluster:d2", ["ocnli:train:d2", "ocnli:train:d3"], dataset="ocnli")]),
     "cross_split_root"),
    (lambda run: _mutate_units(run, [_unit("devunit:u", "cluster:c", ["anli:R1/train:e1"])]),
     "missing_unit"),
])
def test_convert_rejects_nonconforming_sources(tmp_path, mutate, code):
    run = _source(tmp_path)
    mutate(run)
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=_expected(run))
    assert failure.value.code == code
    assert not (tmp_path / "ownrun").exists()


def test_convert_rejects_hash_drift(tmp_path):
    run = _source(tmp_path)
    frozen = _expected(run)
    expected = dict(frozen)
    expected["pool.jsonl"] = "0" * 64
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=expected)
    assert failure.value.code == "source_hash_drift"
    _write_jsonl(run / "pool.jsonl", _pool_rows() + [_record("anli:R1/train:x", "p", "h", "neutral")])
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=frozen)
    assert failure.value.code == "source_hash_drift"


def test_convert_requires_fresh_output(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, out, expected=_expected(run))
    assert failure.value.code == "output_exists"
    existing = tmp_path / "pre-existing"
    existing.mkdir()
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, existing, expected=_expected(run))
    assert failure.value.code == "output_exists"


def test_joint_prepare_accepts_public_slice_with_producer_assets(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    rows = _lines(out / "public-pool.jsonl")
    rows += _fine_rows("train") + _fine_rows("dev")
    data = tmp_path / "joint.jsonl"
    _write_jsonl(data, rows)
    prepared = tmp_path / "prepared"
    meta = SM.prepare(data, prepared, producer_manifest=out / "producer-assets.json")
    assert meta["counts"] == {"train": 3 + 5, "dev": 3 + 5}
    fit, dev = SM.read_prepared(prepared)
    assert len(fit) == 8 and len(dev) == 8
    assert {row["source_id"] for row in fit if row["label_space"] == "nli3"} == set(TRAIN_IDS) - {"anli:R1/train:n1"}

    broken = tmp_path / "broken-assets"
    broken.mkdir()
    for name in ("protected-dev.jsonl", "protection-manifest.jsonl"):
        (broken / name).write_bytes((out / name).read_bytes())
    _write_jsonl(broken / "protected-fit.jsonl", [])
    document = json.loads((out / "producer-assets.json").read_text(encoding="utf-8"))
    for item in document["assets"]:
        if item["role"] == "protected-fit":
            item["sha256"] = FT.file_sha256(broken / "protected-fit.jsonl")
    (broken / "producer-assets.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="member set"):
        SM.producer_lineage(broken / "producer-assets.json", rows)


def _active_manifest(out):
    revision = out / "conversion-manifest-v2.json"
    return revision if revision.is_file() else out / "conversion-manifest.json"


def _rewrite_manifest(out, mutate):
    path = _active_manifest(out)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n")


def _resign(out, name):
    def mutate(document):
        document["outputs"][name]["sha256"] = FT.file_sha256(out / name)
        document["outputs"][name]["bytes"] = (out / name).stat().st_size

    _rewrite_manifest(out, mutate)


def test_verify_rejects_manifest_schema_and_empty_hashes(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    _rewrite_manifest(out, lambda document: document.__setitem__("schema", "semantic-bogus"))
    assert "manifest_schema" in SP.verify(out, run)["problems"]
    _rewrite_manifest(out, lambda document: (document.__setitem__("schema", SP.CONVERTER_SCHEMA),
                                             document.__setitem__("outputs", {}),
                                             document["source"].__setitem__("hashes", {})))
    problems = SP.verify(out, run)["problems"]
    assert "manifest_outputs_missing" in problems and "manifest_source_hashes_missing" in problems


def test_verify_rederives_mapping_and_ghost_quarantine(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    rows = _lines(out / "public-pool.jsonl")
    rows[0]["label"] = "contradiction"
    _write_jsonl(out / "public-pool.jsonl", rows)
    _resign(out, "public-pool.jsonl")
    assert any(problem.startswith("row_label_mismatch:") for problem in SP.verify(out, run)["problems"])

    rows[1]["pair"]["old"]["text"] = rows[1]["pair"]["new"]["text"]
    _write_jsonl(out / "public-pool.jsonl", rows)
    _resign(out, "public-pool.jsonl")
    assert any(problem.startswith("row_pair_mismatch:") for problem in SP.verify(out, run)["problems"])

    quarantine = _lines(out / "source-too-large.jsonl")
    ghost = json.loads(json.dumps(quarantine[0]))
    ghost["source_id"] = rows[0]["source_id"]
    ghost["id"] = "public:" + rows[0]["source_id"]
    ghost["full"]["premise"] = {"chars": 9, "sha256": "0" * 64,
                                "coverage_if_windowed": 1.0, "text": "short text"}
    _write_jsonl(out / "source-too-large.jsonl", quarantine + [ghost])
    _resign(out, "source-too-large.jsonl")
    problems = SP.verify(out, run)["problems"]
    assert f"quarantine_duplicate:{rows[0]['source_id']}" in problems
    assert any(problem.startswith("quarantine_reason_unjustified:") for problem in problems)


@pytest.mark.parametrize("artifacts,code", [
    ({"../evil.jsonl": "0" * 64}, "source_name_invalid"),
    ({"C:/tmp/pool.jsonl": "0" * 64}, "source_name_invalid"),
    ({"sub/pool.jsonl": "0" * 64}, "source_name_invalid"),
    ({"extra.jsonl": "zz"}, "source_hash_invalid"),
])
def test_convert_rejects_plan_path_escapes(tmp_path, artifacts, code):
    run = _source(tmp_path)
    (run / "plan.json").write_text(json.dumps({"plan_version": "fixture", "artifacts": artifacts}),
                                   encoding="utf-8", newline="\n")
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=_expected(run))
    assert failure.value.code == code
    assert not (tmp_path / "ownrun").exists()


def test_convert_rejects_expected_table_escapes(tmp_path):
    run = _source(tmp_path)
    table = _expected(run)
    table["..\\pool.jsonl"] = table.pop("pool.jsonl")
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=table)
    assert failure.value.code == "source_name_invalid"


def test_convert_rejects_input_reparse(tmp_path, monkeypatch):
    run = _source(tmp_path)
    original = SP.NR._reparse_components
    monkeypatch.setattr(SP.NR, "_reparse_components",
                        lambda path: [path] if Path(path).name == "pool.jsonl" else original(path))
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.convert(run, tmp_path / "ownrun", expected=_expected(run))
    assert failure.value.code == "input_reparse"


def test_cap_knob_removed_and_boundary_rows(tmp_path):
    run = _source(tmp_path)
    with pytest.raises(TypeError):
        SP.convert(run, tmp_path / "ownrun", expected=_expected(run), cap_chars=4096)
    with pytest.raises(SystemExit):
        SP.main(["convert", "--source-run", str(run), "--out", str(tmp_path / "cli-out"), "--cap-chars", "4096"])

    boundary = _source(tmp_path, name="boundary", pool=[
        _record("anli:R1/train:b0", "y" * 5000, "h0", "entailment"),
        _record("anli:R1/train:b1", "x" * 8192, "h1", "neutral"),
        _record("anli:R1/train:b2", "x" * 8193, "h2", "neutral"),
        *_pool_rows()[4:]], fit=["anli:R1/train:b0", "anli:R1/train:b1", "anli:R1/train:b2"], dev=DEV_IDS,
        units=[_unit("devunit:b0", "cluster:b0", ["anli:R1/train:b0"]),
               _unit("devunit:b1", "cluster:b1", ["anli:R1/train:b1"]),
               _unit("devunit:b2", "cluster:b2", ["anli:R1/train:b2"]),
               _unit("devunit:dev1", "cluster:d1", ["ocnli:train:d1"]),
               _unit("devunit:dev2", "cluster:d2", ["ocnli:train:d2", "ocnli:train:d3"], dataset="ocnli")])
    out, manifest = _convert(boundary, tmp_path)
    assert manifest["counts"]["trainable"] == {"train": 2, "dev": 3, "total": 5}
    assert manifest["counts"]["quarantine"]["train"] == 1
    assert [row["source_id"] for row in _lines(out / "source-too-large.jsonl")] == ["anli:R1/train:b2"]
    assert manifest["cap"] == {"chars": 8192, "source": "semantic_relations.MAX_TEXT_CHARS",
                               "windowed_applied": False, "knob": "removed",
                               "note": manifest["cap"]["note"]}


def test_reanchor_writes_current_entry_point(tmp_path):
    run = _source(tmp_path)
    out, manifest = _convert(run, tmp_path)
    legacy = out / "conversion-manifest.json"
    legacy_sha = FT.file_sha256(legacy)
    result = SP.reanchor(out, run)
    assert result["problems"] == [] and result["supersedes"] == legacy_sha and result["revision"] == 2
    assert FT.file_sha256(legacy) == legacy_sha
    revision = json.loads((out / "conversion-manifest-v2.json").read_text(encoding="utf-8"))
    assert revision["manifest_revision"] == 2
    assert "ownruns" not in revision["cli"] and str(out) in revision["cli"]
    assert revision["cli_history"]["historical_generation_cli"] == manifest["cli"]
    assert "退休" in revision["cli_history"]["status"]
    assert revision["producer_lineage"]["manifest_path"] == str(out / "producer-assets.json")
    assert revision["readback"]["outputs"]["public-pool.jsonl"] == FT.file_sha256(out / "public-pool.jsonl")
    report = SP.verify(out, run)
    assert report["manifest"] == "conversion-manifest-v2.json" and report["problems"] == []
    v2_sha = FT.file_sha256(out / "conversion-manifest-v2.json")
    third = SP.reanchor(out, run)
    assert third["revision"] == 3 and third["problems"] == []
    assert FT.file_sha256(out / "conversion-manifest-v2.json") == v2_sha and FT.file_sha256(legacy) == legacy_sha
    assert SP.verify(out, run)["manifest"] == "conversion-manifest-v3.json"


def test_verify_reconciles_quarantine_against_source(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    item = _lines(out / "source-too-large.jsonl")[0]

    tampered = json.loads(json.dumps(item))
    tampered["full"]["premise"]["text"] += "x"
    tampered["full"]["premise"]["chars"] = len(tampered["full"]["premise"]["text"])
    tampered["full"]["premise"]["sha256"] = hashlib.sha256(
        tampered["full"]["premise"]["text"].encode("utf-8")).hexdigest()
    _write_jsonl(out / "source-too-large.jsonl", [tampered])
    _resign(out, "source-too-large.jsonl")
    assert any(problem.startswith("quarantine_text_mismatch:") for problem in SP.verify(out, run)["problems"])

    for mutate, pattern in (
            (lambda row: row.__setitem__("label", "contradiction"), "quarantine_label_mismatch:"),
            (lambda row: row.__setitem__("root_id", "devunit:ghost"), "quarantine_unit_mismatch:"),
            (lambda row: row.__setitem__("cluster_id", "cluster:ghost"), "quarantine_unit_mismatch:"),
            (lambda row: row.__setitem__("split", "dev"), "quarantine_split:"),
    ):
        row = json.loads(json.dumps(item))
        mutate(row)
        _write_jsonl(out / "source-too-large.jsonl", [row])
        _resign(out, "source-too-large.jsonl")
        assert any(problem.startswith(pattern) for problem in SP.verify(out, run)["problems"]), pattern


def test_verify_flags_output_ids_absent_from_source(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    quarantine = _lines(out / "source-too-large.jsonl")
    ghost = json.loads(json.dumps(quarantine[0]))
    ghost["source_id"] = "anli:R1/train:ghost"
    ghost["id"] = "public:anli:R1/train:ghost"
    _write_jsonl(out / "source-too-large.jsonl", quarantine + [ghost])
    _resign(out, "source-too-large.jsonl")
    assert any(problem.startswith("output_ids_not_in_source:") for problem in SP.verify(out, run)["problems"])


def test_signatures_pinned_and_verified(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    result = SP.reanchor(out, run)
    signatures = out / "source-too-large-signatures.jsonl"
    assert result["signatures"]["rows"] == 1
    assert result["signatures"]["sha256"] == FT.file_sha256(signatures)
    rows = _lines(signatures)
    assert rows[0]["source_id"] == "anli:R1/train:n1" and rows[0]["over_fields"] == ["premise"]
    assert rows[0]["label"] == "neutral" and rows[0]["split"] == "train"
    report = SP.verify(out, run, signatures_out=signatures)
    assert report["problems"] == [] and report["signatures_rows"] == 1
    assert report["manifest"] == "conversion-manifest-v2.json"
    with open(signatures, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n")
    assert any(problem.startswith("signatures_hash_drift:") for problem in SP.verify(out, run)["problems"])


def test_reanchor_blocked_on_source_drift(tmp_path):
    run = _source(tmp_path)
    out, _ = _convert(run, tmp_path)
    with open(run / "pool.jsonl", "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n")
    with pytest.raises(SP.SemanticPoolError) as failure:
        SP.reanchor(out, run)
    assert failure.value.code == "reanchor_blocked"
