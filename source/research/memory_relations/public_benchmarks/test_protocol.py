                       
"""`protocol.py` 的正反自测：记录合同与 provenance 组合、冻结 study、单一 manifest 机制
（校验/选择/计分/完成）、cluster 分区、诊断探针、指标/截断、oracle 条件任务与证据报告准入。

fixture 是明确的合成合同样例，不是任何真实数据集或真实成绩；正例证明合同可承载，
反例（含 r1/r2 独立审查的核心攻击面）证明篡改/声明式证据/删难题/封存曝光/自报子集计分/
旗子伪造完成后/标量基线伪造都会 fail loud，且不依赖任何无 manifest 的旁路。
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from research.memory_relations.public_benchmarks import protocol as P

ORIGINALS = {
    "ocnli": {"entailment": "entailment", "contradiction": "contradiction", "neutral": "neutral"},
    "anli": {"entailment": "e", "contradiction": "c", "neutral": "n"},
    "hans": {"entailment": "entailment", "non-entailment": "non-entailment"},
    "contractnli": {"entailment": "Entailment", "contradiction": "Contradiction",
                    "neutral": "NotMentioned"},
}
DEFAULT_PREMISE = {
    "ocnli": "地球是圆的。",
    "anli": "The sky is blue.",
    "hans": "The cat sat on the mat.",
    "contractnli": "This Agreement is made between the parties.",
}
DEFAULT_HYPOTHESIS = {
    "ocnli": "地球是圆的吗？",
    "anli": "The sky is green.",
    "hans": "The cat sat on the mat.",
    "contractnli": "The parties may terminate this Agreement.",
}
LABEL_METHOD = {"ocnli": "single_annotation", "anli": "writer_only",
                "hans": "template_rule", "contractnli": "single_annotation"}


def make_record(dataset="ocnli", *, gold="entailment", split="dev", group=None, premise=None,
                hypothesis=None, rid=None, source_row=1, spans=None):
    task = "entailment2" if dataset == "hans" else "nli3"
    premise = DEFAULT_PREMISE[dataset] if premise is None else premise
    hypothesis = DEFAULT_HYPOTHESIS[dataset] if hypothesis is None else hypothesis
    if group is None:
        group = {"ocnli": "prem-0001", "anli": P.premise_family_key(premise),
                 "hans": "template-1", "contractnli": "doc-1"}[dataset]
    status = {"ocnli": "single", "anli": "single",
              "hans": "generated", "contractnli": "single"}[dataset]
    metadata = {"gold_status": status, "label_method": LABEL_METHOD[dataset]}
    if dataset == "ocnli":
        metadata.update({"prem_id": group, "votes": [], "votes_valid_count": 0})
    elif dataset == "anli":
        metadata.update({"premise_hash": group, "round": "r1", "uid": "uid-0001",
                         "verifier_labels": None, "verifier_agree_count": None})
    elif dataset == "hans":
        metadata.update({"heuristic": "lexical_overlap", "subcase": "subcase-1",
                         "template": group})
    else:
        metadata["doc_id"] = group
        if spans is not None:
            metadata["spans"] = spans
    if gold is not None:
        metadata["original_label"] = ORIGINALS[dataset][gold]
    else:
        metadata["gold_status"] = "missing"
        metadata["label_method"] = "no_label"
    evidence = {"source_url": f"https://example.invalid/{dataset}/{split}.json",
                "file": f"{split}.json", "row": source_row}
    rid = rid or f"{dataset}:{split}:{group}:{hypothesis}"
    return {"id": rid, "dataset": dataset, "official_split": split, "task": task,
            "premise": premise, "hypothesis": hypothesis, "gold_label": gold,
            "group_id": group, "metadata": metadata, "evidence": evidence}


def _candidate(dataset, index):
    if dataset == "anli":
        premise = f"ANLI premise {index}."
        return P.premise_family_key(premise), premise, f"ANLI hypothesis {index}."
    if dataset == "ocnli":
        return f"prem-{index:04d}", f"OCNLI premise {index}.", f"OCNLI hypothesis {index}."
    if dataset == "hans":
        return f"tpl-{index:04d}", f"HANS premise {index}.", f"HANS hypothesis {index}."
    return f"doc-{index:04d}", f"CONTRACT text {index}.", f"CONTRACT hypothesis {index}."


def _cluster_id(*records):
    return P.cluster_records(list(records))[0]["cluster_id"]


def record_for_purpose(dataset, purpose, start=0):
    """构造一个真实落在指定用途的合法记录（按确定性 cluster 分区查找，不伪造分桶）。"""
    study = P.load_study()
    for candidate in range(start, start + 2000):
        group, premise, hypothesis = _candidate(dataset, candidate)
        record = make_record(dataset, group=group, premise=premise, hypothesis=hypothesis)
        if P.assign_purpose(study, dataset, _cluster_id(record)) == purpose:
            return record
    raise AssertionError("no unit found for purpose")


def records_for_purpose(dataset, purpose, count):
    study = P.load_study()
    records = []
    candidate = 0
    while len(records) < count:
        group, premise, hypothesis = _candidate(dataset, candidate)
        candidate += 1
        record = make_record(dataset, group=group, premise=premise, hypothesis=hypothesis)
        if P.assign_purpose(study, dataset, _cluster_id(record)) == purpose:
            records.append(record)
    return records


def plain_records(dataset, count, start=0):
    """不按用途过滤的合法记录（final 阶段含全部行；各记录独立 cluster）。"""
    records = []
    for index in range(start, start + count):
        group, premise, hypothesis = _candidate(dataset, index)
        records.append(make_record(dataset, group=group, premise=premise, hypothesis=hypothesis))
    return records


def manifest_evaluate(records, results, *, view_id, phase, records_arg=None):
    manifest = P.partition_records(records)
    report = P.evaluate_view(records if records_arg is None else records_arg, results,
                             view_id=view_id, phase=phase, manifest=manifest)
    return manifest, report


def attribution_entries(*classes):
    return [{"class": name, "evidence": f"{name} evidence", "ref": f"{name} ref"}
            for name in classes]


def artifact_entry(**overrides):
    """M11 原生工件 meta 投影的合成样例（形状对齐 nli_reference.NLI_*，值全为 fixture）。"""
    artifact = {"artifact_id": "nli-reference-fixture", "schema": P.NATIVE_REFERENCE_SCHEMA,
                "kind": P.NATIVE_REFERENCE_KIND, "classes": list(P.NLI_CLASSES),
                "label_space": P.NATIVE_LABEL_SPACE, "input_recipe": P.NATIVE_INPUT_RECIPE,
                "direction": P.PRIMARY_DIRECTION, "joint_roles": dict(P.PRIMARY_JOINT_ROLES),
                "max_length": 384, "weights_sha256": "a" * 64,
                "backbone_fingerprint": "backbone:fixture"}
    artifact.update(overrides)
    return artifact


def bucket_digest(rows):
    """与 `protocol._rows_sha256` 同口径的桶内容摘要（供篡改复算 SHA 的负例用）。"""
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) + "\n").encode("utf-8")
                       for row in sorted(rows, key=lambda row: row["id"]))
    return hashlib.sha256(payload).hexdigest()


def assert_code(record, code, *, for_dataset=False):
    with pytest.raises(P.ContractError) as error:
        (P.validate_record_for_dataset if for_dataset else P.validate_record)(record)
    assert error.value.code == code


                                                                             
      
                                                                             

@pytest.mark.parametrize("dataset", list(P.DATASETS))
def test_valid_records_for_all_datasets(dataset):
    record = make_record(dataset)
    P.validate_record(record)
    P.validate_record_for_dataset(record)


def test_null_gold_is_kept_with_status_not_fabricated():
    record = make_record("ocnli", gold=None)
    record["metadata"]["gold_status"] = "no_consensus"
    record["metadata"]["label_method"] = "majority_vote"
    record["metadata"]["original_label"] = "-"
    P.validate_record_for_dataset(record)
    assert record["gold_label"] is None
    missing = make_record("ocnli", gold=None)
    P.validate_record_for_dataset(missing)
    bad = copy.deepcopy(record)
    bad["metadata"]["label_method"] = "single_annotation"
    assert_code(bad, "invalid_provenance", for_dataset=True)
    fabricated = copy.deepcopy(record)
    del fabricated["metadata"]["original_label"]
    assert_code(fabricated, "invalid_provenance", for_dataset=True)


def test_ocnli_no_consensus_marker_maps_to_null_gold():
    record = make_record("ocnli", gold=None)
    record["metadata"]["gold_status"] = "no_consensus"
    record["metadata"]["label_method"] = "majority_vote"
    record["metadata"]["original_label"] = "-"
    P.validate_record_for_dataset(record)
    assert P.canonical_gold_label("ocnli", "-") is None
    bad = copy.deepcopy(record)
    bad["gold_label"] = "entailment"
    bad["metadata"]["gold_status"] = "single"
    assert_code(bad, "original_gold_mismatch")


def test_ocnli_missing_prem_id_uses_disclosed_premise_family_key():
    """缺 raw 单位 id 的派生披露与 gold 是否缺失无关（M8 train 有 gold 的派生行也合法）。"""
    premise = "缺少 prem_id 的官方 test 行。"
    group = P.premise_family_key(premise)
    missing_row = make_record("ocnli", gold=None, group=group, premise=premise)
    del missing_row["metadata"]["prem_id"]
    missing_row["metadata"]["group_id_source"] = "derived_premise_family"
    P.validate_record_for_dataset(missing_row)
    labeled = make_record("ocnli", premise=premise, group=group)
    del labeled["metadata"]["prem_id"]
    labeled["metadata"]["group_id_source"] = "derived_premise_family"
    P.validate_record_for_dataset(labeled)
    undisclosed = copy.deepcopy(missing_row)
    del undisclosed["metadata"]["group_id_source"]
    assert_code(undisclosed, "unit_fallback_missing_source", for_dataset=True)
    undisclosed_labeled = copy.deepcopy(labeled)
    del undisclosed_labeled["metadata"]["group_id_source"]
    assert_code(undisclosed_labeled, "unit_fallback_missing_source", for_dataset=True)
    wrong = copy.deepcopy(labeled)
    wrong["group_id"] = "prem-9999"
    assert_code(wrong, "premise_hash_mismatch", for_dataset=True)


def test_ocnli_real_source_shapes_validate():
    """M8 r1fix 真实形状：test 无标/no_label、train 有 gold 派生、no_consensus、票数诚实。"""
    premise = "也可以是这种简素为美的精神"
    derived = P.premise_family_key(premise)
    test_row = make_record("ocnli", gold=None, group=derived, premise=premise)
    test_row["metadata"].update({"prem_id": None, "votes": [None] * 5, "votes_valid_count": 0,
                                 "group_id_source": "derived_premise_family"})
    P.validate_record_for_dataset(test_row)
    train_row = make_record("ocnli", gold="neutral", group=derived, premise=premise)
    train_row["metadata"].update({"prem_id": None, "votes": [None] * 5, "votes_valid_count": 0,
                                  "group_id_source": "derived_premise_family"})
    P.validate_record_for_dataset(train_row)
    no_consensus = make_record("ocnli", gold=None, group="gov_1448",
                               premise="争取三年内基本解决农村债务问题")
    no_consensus["metadata"].update(
        {"gold_status": "no_consensus", "label_method": "majority_vote", "original_label": "-",
         "prem_id": "gov_1448",
         "votes": ["entailment", "contradiction", "neutral", "contradiction", "entailment"],
         "votes_valid_count": 5})
    P.validate_record_for_dataset(no_consensus)
    majority = make_record("ocnli", gold="entailment", group="lit_635",
                           premise="身上裹一件工厂发的棉大衣")
    majority["metadata"].update({"gold_status": "majority", "label_method": "majority_vote",
                                 "votes": ["entailment"] * 5, "votes_valid_count": 5})
    P.validate_record_for_dataset(majority)


def test_dataset_provenance_is_enforced():
    hans = make_record("hans")
    assert hans["metadata"]["gold_status"] == "generated"
    P.validate_record_for_dataset(hans)
    bad = copy.deepcopy(hans)
    bad["metadata"]["gold_status"] = "adjudicated"
    bad["metadata"]["label_method"] = "human_panel"
    assert_code(bad, "invalid_provenance", for_dataset=True)
    bad = copy.deepcopy(hans)
    bad["metadata"]["label_method"] = "human_panel"
    assert_code(bad, "invalid_provenance", for_dataset=True)
    ocnli = make_record("ocnli")
    bad = copy.deepcopy(ocnli)
    bad["metadata"]["gold_status"] = "adjudicated"
    assert_code(bad, "invalid_provenance", for_dataset=True)
    contract = make_record("contractnli")
    bad = copy.deepcopy(contract)
    bad["metadata"]["gold_status"] = "majority"
    assert_code(bad, "invalid_provenance", for_dataset=True)
    anli = make_record("anli")
    item = copy.deepcopy(anli)
    item["metadata"]["gold_status"] = "single"
    item["metadata"]["label_method"] = "writer_only"
    P.validate_record_for_dataset(item)
    adjudicated = copy.deepcopy(anli)
    adjudicated["metadata"].update({"gold_status": "adjudicated",
                                    "label_method": "writer_with_verifiers",
                                    "verifier_labels": ["e", "e"], "verifier_agree_count": 2})
    P.validate_record_for_dataset(adjudicated)
    unproven = copy.deepcopy(anli)
    unproven["metadata"].update({"gold_status": "adjudicated",
                                 "label_method": "writer_with_verifiers"})
    assert_code(unproven, "invalid_provenance", for_dataset=True)


def test_provenance_evidence_count_is_recomputed():
    """status/method/count 三件一起执法：一票 majority、假计数、单票与复核票不诚实都拒。"""
    single_opinion = make_record("ocnli")
    single_opinion["metadata"].update({"gold_status": "majority", "label_method": "majority_vote",
                                       "votes": ["entailment", None, None, None, None],
                                       "votes_valid_count": 1})
    assert_code(single_opinion, "invalid_provenance", for_dataset=True)
    majority = make_record("ocnli")
    majority["metadata"].update({"gold_status": "majority", "label_method": "majority_vote",
                                 "votes": ["entailment", "entailment", None, None, None],
                                 "votes_valid_count": 2})
    P.validate_record_for_dataset(majority)
    dishonest = copy.deepcopy(majority)
    dishonest["metadata"]["votes_valid_count"] = 5
    assert_code(dishonest, "invalid_provenance", for_dataset=True)
    no_votes_as_majority = make_record("ocnli")
    no_votes_as_majority["metadata"].update({"gold_status": "majority",
                                             "label_method": "majority_vote"})
    assert_code(no_votes_as_majority, "invalid_provenance", for_dataset=True)
    anli = make_record("anli")
    anli["metadata"].update({"gold_status": "single", "label_method": "writer_only",
                             "verifier_labels": ["e", "e"], "verifier_agree_count": 2})
    assert_code(anli, "invalid_provenance", for_dataset=True)
    agree_mismatch = make_record("anli")
    agree_mismatch["metadata"].update({"gold_status": "adjudicated",
                                       "label_method": "writer_with_verifiers",
                                       "verifier_labels": ["e", "n"], "verifier_agree_count": 2})
    assert_code(agree_mismatch, "invalid_provenance", for_dataset=True)


def _vote_row(votes, *, count=None, status="majority", method="majority_vote", gold="entailment"):
    record = make_record("ocnli", gold=gold)
    record["metadata"].update({"gold_status": status, "label_method": method, "votes": votes,
                               "votes_valid_count": len(votes) if count is None else count})
    return record


def test_ocnli_majority_requires_strict_support_for_published_gold():
    """r3 P2-1：≥2 有效票不够——必须对 unchanged 官方 gold 严格过半（2×同意票>有效票）。"""
    for votes in (["contradiction", "contradiction"],
                  ["entailment", "contradiction", "contradiction"],
                  ["entailment", "contradiction"]):
        assert_code(_vote_row(votes), "invalid_provenance", for_dataset=True)
    P.validate_record_for_dataset(
        _vote_row(["entailment", "entailment", "contradiction"]))
    assert_code(_vote_row(["entailment", "entailment", "contradiction"], count=2),
                "invalid_provenance", for_dataset=True)
    assert_code(_vote_row(["entailment", "entailment", "contradiction"], gold="contradiction"),
                "invalid_provenance", for_dataset=True)


def test_ocnli_no_majority_defensive_branch_is_bounded():
    """防务分支（镜像 producer parse_ocnli）：有票无严格多数 → single/no_majority，gold 不改；说谎即拒。"""
    P.validate_record_for_dataset(
        _vote_row(["entailment", "contradiction"], status="single", method="no_majority"))
    P.validate_record_for_dataset(
        _vote_row(["contradiction", "contradiction"], status="single", method="no_majority"))
    assert_code(_vote_row(["entailment", "entailment", "contradiction"],
                          status="single", method="no_majority"),
                "invalid_provenance", for_dataset=True)
    assert_code(_vote_row(["entailment"], status="single", method="no_majority"),
                "invalid_provenance", for_dataset=True)


def test_ocnli_missing_follows_producer_anchor_counts():
    """锚点语义：original_label None → missing/no_label（票计数原样保留、计数仍须诚实）。"""
    record = make_record("ocnli", gold=None)
    record["metadata"].update({"gold_status": "missing", "label_method": "no_label",
                               "votes": ["entailment", None, None, None, None],
                               "votes_valid_count": 1})
    P.validate_record_for_dataset(record)
    bad = copy.deepcopy(record)
    bad["metadata"]["votes_valid_count"] = 5
    assert_code(bad, "invalid_provenance", for_dataset=True)


def test_ocnli_provenance_reuses_public_producer_anchor():
    """唯一实现：study 的 derivation 解析到 datasets.ocnli_vote_status；校验器只调用并比对。"""
    from research.memory_relations.public_benchmarks import datasets as public_datasets
    derivation = P.load_study()["datasets"]["ocnli"]["provenance"]["derivation"]
    assert derivation["function"] == "ocnli_vote_status"
    assert getattr(public_datasets, derivation["function"]) is public_datasets.ocnli_vote_status
    cases = ((None, [None, None, None, None, None]),
             ("-", ["entailment", "contradiction"]),
             ("entailment", ["entailment", "entailment", "contradiction"]),
             ("entailment", ["entailment", "contradiction"]))
    for original, votes in cases:
        derived = public_datasets.ocnli_vote_status(original, votes)
        gold = None if original in (None, "-") else original
        record = _vote_row(votes, status=derived["gold_status"],
                           method=derived["label_method"],
                           count=derived["votes_valid_count"], gold=gold)
        if original is not None:
            record["metadata"]["original_label"] = original
        else:
            record["metadata"].pop("original_label", None)
        P.validate_record_for_dataset(record)
        lied = copy.deepcopy(record)
        lied["metadata"]["label_method"] = ("single_annotation"
                                            if derived["label_method"] != "single_annotation"
                                            else "no_majority")
        assert_code(lied, "invalid_provenance", for_dataset=True)


@pytest.mark.parametrize("dataset,original,expected", [
    ("ocnli", "entailment", "entailment"),
    ("ocnli", "neutral", "neutral"),
    ("ocnli", "contradiction", "contradiction"),
    ("ocnli", "-", None),
    ("anli", "e", "entailment"),
    ("anli", "n", "neutral"),
    ("anli", "c", "contradiction"),
    ("hans", "entailment", "entailment"),
    ("hans", "non-entailment", "non-entailment"),
    ("contractnli", "Entailment", "entailment"),
    ("contractnli", "Contradiction", "contradiction"),
    ("contractnli", "NotMentioned", "neutral"),
])
def test_canonical_gold_label_maps_each_dataset(dataset, original, expected):
    assert P.canonical_gold_label(dataset, original) == expected


def test_label_catalog_is_runtime_anchor_with_parity():
    catalog = {dataset: dict(P.NATIVE_LABEL_MAPS[dataset]) for dataset in P.DATASETS}
    P.check_label_catalog(catalog)
    assert P.canonical_gold_label("anli", "e", catalog=catalog) == "entailment"
    assert P.canonical_gold_label("ocnli", "-", catalog=catalog) is None
    record = make_record("anli")
    P.validate_record_for_dataset(record, catalog=catalog)
    drift_missing_marker = copy.deepcopy(catalog)
    drift_missing_marker["ocnli"].pop("-")
    with pytest.raises(P.ContractError) as error:
        P.check_label_catalog(drift_missing_marker)
    assert error.value.code == "label_map_drift"
    drift = copy.deepcopy(catalog)
    drift["anli"]["e"] = "contradiction"
    with pytest.raises(P.ContractError) as error:
        P.check_label_catalog(drift)
    assert error.value.code == "label_map_drift"
    with pytest.raises(P.ContractError) as error:
        P.validate_record(record, catalog=drift)
    assert error.value.code == "original_gold_mismatch"


def test_canonical_gold_label_fails_loud():
    with pytest.raises(P.ContractError) as error:
        P.canonical_gold_label("ocnli", "conflict")
    assert error.value.code == "unknown_original_label"
    with pytest.raises(P.ContractError) as error:
        P.canonical_gold_label("cnli", "entailment")
    assert error.value.code == "invalid_dataset"


def test_premise_family_key_is_normalized_content_hash():
    folded = P.premise_family_key("The  sky\nis   blue.")
    plain = P.premise_family_key("The sky is blue.")
    assert folded == plain and folded.startswith("premise:")
    assert P.premise_family_key("The sky is green.") != plain


def test_record_rejects_shape_label_and_provenance_violations():
    record = make_record("ocnli")
    bad = copy.deepcopy(record)
    del bad["evidence"]
    assert_code(bad, "invalid_record")
    bad = copy.deepcopy(record)
    bad["extra"] = 1
    assert_code(bad, "invalid_record")
    bad = copy.deepcopy(record)
    bad["dataset"] = "cnli"
    assert_code(bad, "invalid_dataset")
    bad = copy.deepcopy(record)
    bad["task"] = "nli2"
    assert_code(bad, "invalid_task")
    bad = copy.deepcopy(record)
    bad["premise"] = "   "
    assert_code(bad, "invalid_record")
    bad = copy.deepcopy(record)
    bad["gold_label"] = "conflict"
    assert_code(bad, "forbidden_gold_relation")
    bad = make_record("hans")
    bad["gold_label"] = "contradiction"
    assert_code(bad, "invalid_gold")
    bad = copy.deepcopy(record)
    bad["gold_label"] = None
    assert_code(bad, "gold_status_mismatch")
    bad = copy.deepcopy(record)
    bad["metadata"]["gold_status"] = "no_consensus"
    assert_code(bad, "gold_status_mismatch")
    bad = copy.deepcopy(record)
    bad["metadata"]["gold_status"] = "maybe"
    assert_code(bad, "invalid_gold_status")
    bad = copy.deepcopy(record)
    bad["metadata"]["original_label"] = "neutral"
    assert_code(bad, "original_gold_mismatch")
    bad = copy.deepcopy(record)
    bad["metadata"]["original_label"] = "n"
    assert_code(bad, "unknown_original_label")
    bad = copy.deepcopy(record)
    del bad["metadata"]["original_label"]
    assert_code(bad, "missing_original_label")
    bad = copy.deepcopy(record)
    del bad["evidence"]["source_url"]
    assert_code(bad, "invalid_evidence")
    bad = copy.deepcopy(record)
    bad["evidence"]["sha256"] = "xyz"
    assert_code(bad, "invalid_evidence")
    bad = copy.deepcopy(record)
    bad["evidence"] = []
    assert_code(bad, "invalid_evidence")
    bad = copy.deepcopy(record)
    bad["metadata"] = {}
    assert_code(bad, "invalid_metadata")
    bad = copy.deepcopy(record)
    bad["metadata"]["blob"] = "y" * 20000
    assert_code(bad, "invalid_metadata")
    bad = copy.deepcopy(record)
    bad["metadata"]["blob"] = {1, 2}
    assert_code(bad, "invalid_metadata")


def test_dataset_level_unit_and_metadata_violations():
    record = make_record("ocnli")
    bad = copy.deepcopy(record)
    del bad["metadata"]["prem_id"]
    assert_code(bad, "premise_hash_mismatch", for_dataset=True)
    bad["group_id"] = P.premise_family_key(bad["premise"])
    assert_code(bad, "unit_fallback_missing_source", for_dataset=True)
    bad["metadata"]["group_id_source"] = "derived_premise_family"
    P.validate_record_for_dataset(bad)
    bad = copy.deepcopy(record)
    bad["group_id"] = "other"
    assert_code(bad, "unit_mismatch", for_dataset=True)
    bad = copy.deepcopy(record)
    del bad["metadata"]["label_method"]
    assert_code(bad, "missing_required_metadata", for_dataset=True)
    anli = make_record("anli")
    bad = copy.deepcopy(anli)
    bad["group_id"] = "premise:0000000000000000"
    bad["metadata"]["premise_hash"] = bad["group_id"]
    assert_code(bad, "premise_hash_mismatch", for_dataset=True)
    bad = copy.deepcopy(anli)
    bad["metadata"]["round"] = "r4"
    assert_code(bad, "invalid_metadata_value", for_dataset=True)
    bad = copy.deepcopy(anli)
    bad["metadata"]["round"] = "R2"
    P.validate_record_for_dataset(bad)
    hans = make_record("hans")
    bad = copy.deepcopy(hans)
    bad["metadata"]["template"] = 99
    assert_code(bad, "unit_mismatch", for_dataset=True)


def test_oracle_views_use_controlled_spans_only():
    record = make_record("contractnli")
    assert P.oracle_available(record) is False
    with pytest.raises(P.ContractError) as error:
        P.oracle_premise(record)
    assert error.value.code == "oracle_spans_unavailable"
    record["metadata"]["spans"] = [{"text": "First clause.", "start": 0, "end": 13},
                                   {"text": "Second clause."}]
    assert P.oracle_available(record) is True
    assert P.oracle_premise(record) == "First clause.\nSecond clause."
    bad = copy.deepcopy(record)
    bad["metadata"]["spans"] = [{"text": "ok"}, {"no_text": 1}]
    with pytest.raises(P.ContractError) as error:
        P.oracle_available(bad)
    assert error.value.code == "invalid_spans"


def test_oracle_eligibility_is_e_c_conditional_only():
    neutral = make_record("contractnli", gold="neutral",
                          spans=[{"text": "Clause."}])
    assert P.oracle_eligible(neutral) is False
    entailment = make_record("contractnli", gold="entailment", spans=[{"text": "Clause."}])
    assert P.oracle_eligible(entailment) is True
    without_span = make_record("contractnli", gold="contradiction")
    assert P.oracle_eligible(without_span) is False


                                                                             
           
                                                                             

def test_load_study_matches_frozen_hash_and_shape():
    study = P.load_study()
    assert P.study_sha256() == P.FROZEN_STUDY_SHA256
    assert study["format"] == P.FORMAT and study["status"] == "research-only"
    assert set(study["datasets"]) == set(P.DATASETS)
    view_ids = [view["view_id"] for view in study["views"]]
    assert len(view_ids) == len(set(view_ids)) and view_ids
    assert any("HANS" in practice for practice in study["forbidden"]["practices"])
    assert len(study["freeze"]["revision_history"]) >= 3


def test_load_study_rejects_tampered_copy(tmp_path):
    tampered = tmp_path / "study-v4.json"
    tampered.write_bytes(P.study_path().read_bytes() + b"\n")
    with pytest.raises(P.ContractError) as error:
        P.load_study(tampered)
    assert error.value.code == "study_frozen_mismatch"


def test_study_mirrors_protocol_constants():
    study = P.load_study()
    assert study["label_spaces"]["canonical_mapping"] == P.NATIVE_LABEL_MAPS
    assert study["label_spaces"]["diagnostic_seven"] == list(P.SEVEN_CLASSES)
    assert study["label_spaces"]["tasks"] == list(P.TASKS)
    assert study["label_spaces"]["gold_statuses"] == list(P.GOLD_STATUSES)
    assert study["label_spaces"]["nli_classes"] == list(P.NLI_CLASSES)
    native = study["native_reference"]
    assert native["schema"] == P.NATIVE_REFERENCE_SCHEMA
    assert native["kind"] == P.NATIVE_REFERENCE_KIND
    assert native["input_recipe"] == P.NATIVE_INPUT_RECIPE
    assert native["label_space"] == P.NATIVE_LABEL_SPACE
    assert native["classes"] == list(P.NLI_CLASSES)
    assert native["direction"] == P.PRIMARY_DIRECTION
    assert native["joint_roles"] == dict(P.PRIMARY_JOINT_ROLES)
    assert study["forbidden"]["label_mappings"] == [list(pair) for pair in P.FORBIDDEN_LABEL_MAPPINGS]
    assert study["partition"]["purposes"] == list(P.PURPOSES)
    assert study["partition"]["phases"] == list(P.PHASES)
    assert study["input_adaptations"]["diagnostic"]["primary_direction"] == P.PRIMARY_DIRECTION


def test_native_recipe_registration_matches_finetune_source():
    """注册的 input_recipe 必须等于 finetune.INPUT_RECIPE（单一来源；M11 工件同源）。"""
    from research.memory_relations import finetune
    assert P.NATIVE_INPUT_RECIPE == finetune.INPUT_RECIPE


def test_forbidden_mapping_pairs_are_frozen():
    for pair in (("neutral", "uncertain"), ("entailment", "duplicate"),
                 ("non-entailment", "conflict")):
        assert pair in P.FORBIDDEN_LABEL_MAPPINGS


def test_study_freezes_views_boundaries():
    study = P.load_study()
    datasets_with_windows = {}
    for view in study["views"]:
        if view["kind"] == "seven_class_external_diagnostic":
            assert "accuracy" not in view["metrics"]
            assert view["directions"]["primary"] == P.PRIMARY_DIRECTION
            assert view["directions"]["swap"] == P.DIRECTIONS[1]
            assert view["task_equivalence"] == "none"
            datasets_with_windows.setdefault(view["dataset"], set()).add(view["window"]["kind"])
        else:
            assert "accuracy" in view["metrics"]
            assert view["available_in_round"] is False
            assert view["unavailable_reason"]
            assert view["availability_rule"] == "artifact_binding_required"
            if view.get("layer") == "oracle_evidence":
                assert view["task"] == "entailment_contradiction"
                assert view["scope_rule"] == "gold_evidence_two_class"
                assert view["eligibility"] == {"gold_in": ["entailment", "contradiction"],
                                               "require_spans": True}
            else:
                assert view["scope_rule"] == "all_rows"
    for dataset in P.DATASETS:
        assert datasets_with_windows[dataset] == {"production_contract", "backbone_fulltext"}
    contract_layers = {view.get("layer") for view in study["views"]
                       if view["dataset"] == "contractnli" and view["kind"] == "native_nli"}
    assert contract_layers == {"full_document", "oracle_evidence"}
    oracle = next(view for view in study["views"]
                  if view["view_id"] == "contractnli_nli3_oracle_evidence")
    assert oracle["task"] != "nli3"
    assert P.view_label_space(oracle) == ("entailment", "contradiction")


def test_native_view_binding_checks_registered_recipe_and_profile():
    study = P.load_study()
    view = P.view_spec(study, "ocnli_nli3_native")
    artifact = artifact_entry()
    binding = P.native_view_binding(view, artifact)
    assert binding["runnable"] is True and binding["artifact_id"] == "nli-reference-fixture"
    assert binding["metadata_check"] == "matched_registered_native_reference"
    assert binding["artifact_integrity"] == "not_verified_here"
    assert binding["input_recipe"] == P.NATIVE_INPUT_RECIPE

    def expect_code(bad, code, target=view):
        with pytest.raises(P.ContractError) as error:
            P.native_view_binding(target, bad)
        assert error.value.code == code

    expect_code(artifact_entry(input_recipe="unrelated-bag-of-words-1024"),
                "artifact_registration_mismatch")
    expect_code(artifact_entry(label_space="entailment2"), "artifact_registration_mismatch")
    expect_code(artifact_entry(direction="premise_as_candidate"), "artifact_registration_mismatch")
    expect_code(artifact_entry(joint_roles={"premise": "candidate", "hypothesis": "existing"}),
                "artifact_registration_mismatch")
    expect_code(artifact_entry(classes=["entailment", "neutral", "other"]), "invalid_artifact")
    expect_code(artifact_entry(max_length=512), "artifact_window_mismatch")
    expect_code(artifact_entry(weights_sha256="short"), "invalid_artifact")
    missing_key = artifact_entry()
    del missing_key["backbone_fingerprint"]
    expect_code(missing_key, "invalid_artifact")
    full_document = P.view_spec(study, "contractnli_nli3_full_document")
    wide = P.native_view_binding(full_document, artifact_entry(max_length=512))
    assert wide["max_length"] == 512
    expect_code(artifact_entry(max_length=0), "invalid_artifact", target=full_document)


def test_native_view_binding_verifies_weights_file_integrity(tmp_path):
    weights = tmp_path / "nli_model.pt"
    weights.write_bytes(b"fixture weights payload")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    view = P.view_spec(P.load_study(), "ocnli_nli3_native")
    binding = P.native_view_binding(view, artifact_entry(weights_sha256=digest),
                                    weights_path=weights)
    assert binding["artifact_integrity"] == "verified_weights_sha256"
    with pytest.raises(P.ContractError) as error:
        P.native_view_binding(view, artifact_entry(), weights_path=weights)
    assert error.value.code == "artifact_hash_mismatch"
    with pytest.raises(P.ContractError) as error:
        P.native_view_binding(view, artifact_entry(weights_sha256=digest),
                              weights_path=tmp_path / "missing.pt")
    assert error.value.code == "artifact_missing"


def assert_study_rejected(mutation, code="invalid_study"):
    study = copy.deepcopy(P.load_study())
    mutation(study)
    with pytest.raises(P.ContractError) as error:
        P.validate_study(study)
    assert error.value.code == code


def test_validate_study_rejects_tampered_structure():
    def set_status(study):
        study["status"] = "production"

    def drop_dataset(study):
        del study["datasets"]["hans"]

    def metric_mix(study):
        view = next(view for view in study["views"]
                    if view["kind"] == "seven_class_external_diagnostic")
        view["metrics"] = view["metrics"] + ["accuracy"]

    def flip_direction(study):
        view = next(view for view in study["views"]
                    if view["kind"] == "seven_class_external_diagnostic")
        view["directions"]["primary"] = "premise_as_candidate"

    def native_available(study):
        view = next(view for view in study["views"] if view["kind"] == "native_nli")
        view["available_in_round"] = True

    def drop_availability_rule(study):
        view = next(view for view in study["views"] if view["kind"] == "native_nli")
        del view["availability_rule"]

    def oracle_back_to_nli3(study):
        view = next(view for view in study["views"] if view.get("layer") == "oracle_evidence")
        view["task"] = "nli3"
        del view["subset_of"]

    def loose_hans_provenance(study):
        study["datasets"]["hans"]["provenance"]["combinations"] = [
            {"gold_status": ["adjudicated"], "label_method": ["human_panel"]}]

    def broken_vote_rule(study):
        study["datasets"]["ocnli"]["provenance"]["count_mode"] = "guessing"

    def impossible_count_bounds(study):
        study["datasets"]["ocnli"]["provenance"]["combinations"][0].update(
            {"min_count": 5, "max_count": 2})

    def bad_derivation_function(study):
        study["datasets"]["ocnli"]["provenance"]["derivation"]["function"] = "missing_anchor"

    def bad_derivation_arguments(study):
        study["datasets"]["ocnli"]["provenance"]["derivation"]["arguments"]["votes"] = "votes"

    def derivation_extra_key(study):
        study["datasets"]["ocnli"]["provenance"]["derivation"]["note"] = "x"

    def overlapping_combos(study):
        study["datasets"]["ocnli"]["provenance"]["combinations"].append(
            {"gold_status": ["single"], "label_method": ["single_annotation"]})

    def no_derivation(study):
        study["admission"]["baseline_profiles"]["chance"]["derivation"] = "magic"

    def drop_profiles(study):
        del study["admission"]["baseline_profiles"]

    def drop_history(study):
        del study["freeze"]["revision_history"]

    def bad_purposes(study):
        study["partition"]["purposes"] = list(reversed(P.PURPOSES))

    def bad_truncation(study):
        study["metrics"]["truncation"]["explicit_evidence_required"] = False

    def bad_admission(study):
        study["admission"]["min_counterexample_units"] = 0

    def empty_practices(study):
        study["forbidden"]["practices"] = []

    def drop_exclusions(study):
        study["metrics"]["exclusion_reasons"] = []

    def broken_fallback(study):
        study["datasets"]["ocnli"]["unit"]["fallback"]["source_value"] = ""

    for mutation in (set_status, drop_dataset, metric_mix, flip_direction, native_available,
                     drop_availability_rule, oracle_back_to_nli3, loose_hans_provenance,
                     broken_vote_rule, impossible_count_bounds, bad_derivation_function,
                     bad_derivation_arguments, derivation_extra_key, overlapping_combos,
                     no_derivation, drop_profiles, drop_history, bad_purposes, bad_truncation,
                     bad_admission, empty_practices, drop_exclusions, broken_fallback):
        assert_study_rejected(mutation)
    assert_study_rejected(
        lambda study: study["native_reference"].__setitem__("input_recipe", "other-recipe"),
        code="study_mirror_mismatch")
    assert_study_rejected(
        lambda study: study["native_reference"]["joint_roles"].__setitem__(
            "premise", "candidate"),
        code="invalid_study")
    assert_study_rejected(
        lambda study: study["label_spaces"].__setitem__("diagnostic_seven", ["conflict"]),
        code="study_mirror_mismatch")
    assert_study_rejected(
        lambda study: study["forbidden"].__setitem__("label_mappings", []),
        code="study_mirror_mismatch")
    assert_study_rejected(
        lambda study: study["label_spaces"].__setitem__("gold_statuses", ["adjudicated"]),
        code="study_mirror_mismatch")
    assert_study_rejected(
        lambda study: study["label_spaces"]["canonical_mapping"]["ocnli"].pop("-"),
        code="study_mirror_mismatch")


                                                                             
                               
                                                                             

def test_clusters_merge_group_premise_and_pair_dependencies():
    first = make_record("ocnli", group="prem-1000", premise="共享前提。", hypothesis="h-a", rid="c-1")
    second = make_record("ocnli", group="prem-2000", premise="共享前提。", hypothesis="h-b", rid="c-2")
    clusters = P.cluster_records([first, second])
    assert len(clusters) == 1 and clusters[0]["rows"] == 2
    assert clusters[0]["raw_units"] == ["prem-1000", "prem-2000"]
    separate = make_record("ocnli", group="prem-5000", premise="别的。", hypothesis="别的假设",
                           rid="c-5")
    assert len(P.cluster_records([first, separate])) == 2


def test_partition_is_deterministic_and_cluster_atomic():
    records = [make_record("ocnli", group="prem-a", hypothesis="h1", rid="id-1"),
               make_record("ocnli", group="prem-a", hypothesis="h2", rid="id-2"),
               record_for_purpose("hans", "diagnostic", 0),
               record_for_purpose("anli", "sealed_acceptance", 0)]
    records[2]["id"] = "id-3"
    records[3]["id"] = "id-4"
    first = P.partition_records(records)
    second = P.partition_records(records)
    assert first["sha256"] == second["sha256"]
    diagnostic_ids = {row["id"] for row in first["diagnostic"]}
    sealed_ids = {row["id"] for row in first["sealed_acceptance"]}
    assert diagnostic_ids | sealed_ids == {"id-1", "id-2", "id-3", "id-4"}
    assert not diagnostic_ids & sealed_ids
    assert first["units"]["id-1"] == first["units"]["id-2"]
    assert first["purposes"]["id-1"] == first["purposes"]["id-2"]
    assert first["summary"]["total"] == 4
    assert first["summary"]["largest_cluster"]["rows"] == 2
    assert P.validate_manifest(first)["summary"] == first["summary"]
    report = P.check_leakage(first)
    assert report["status"] == "ok" and report["clusters"] == 3
    merged = [cluster for cluster in first["clusters"] if "id-1" in cluster["record_ids"]]
    assert merged and merged[0]["record_ids"] == ["id-1", "id-2"]


def test_partition_hashes_bind_content():
    records = [record_for_purpose("ocnli", "diagnostic", 0)]
    before = P.partition_records(records)
    changed = copy.deepcopy(records)
    changed[0]["metadata"]["note"] = "changed"
    after = P.partition_records(changed)
    purpose = before["purposes"][records[0]["id"]]
    assert before["sha256"][purpose] != after["sha256"][purpose]


def test_partition_rejects_duplicate_ids():
    records = [make_record("ocnli", rid="dup", hypothesis="one"),
               make_record("ocnli", rid="dup", hypothesis="two")]
    with pytest.raises(P.ContractError) as error:
        P.partition_records(records)
    assert error.value.code == "duplicate_record_id"


def test_select_for_phase_keeps_sealed_hidden_until_final():
    records = records_for_purpose("hans", "diagnostic", 2) + records_for_purpose(
        "hans", "sealed_acceptance", 2)
    partitioned = P.partition_records(records)
    diagnostic = P.select_for_phase(partitioned, "diagnostic")
    final = P.select_for_phase(partitioned, "final_acceptance")
    assert {row["id"] for row in diagnostic} == {row["id"] for row in partitioned["diagnostic"]}
    assert len(final) == len(records)
    with pytest.raises(P.ContractError) as error:
        P.select_for_phase(partitioned, "maybe")
    assert error.value.code == "invalid_phase"


@pytest.mark.parametrize("field", ["purposes", "units", "clusters", "sha256", "summary", "content"])
def test_selector_revalidates_manifest_before_returning_rows(field):
    """曝光入口与计分入口同一验证：select_for_phase 返回任何行前必须过 validate_manifest。"""
    row = record_for_purpose("ocnli", "diagnostic", 0)
    manifest = P.partition_records([row])
    if field == "purposes":
        manifest["purposes"][row["id"]] = "sealed_acceptance"
    elif field == "units":
        manifest["units"][row["id"]] = "pretend-independent"
    elif field == "clusters":
        manifest["clusters"][0]["raw_units"] = ["pretend-raw"]
    elif field == "sha256":
        manifest["sha256"]["diagnostic"] = "0" * 64
    elif field == "summary":
        manifest["summary"]["total"] += 1
    else:
        manifest["diagnostic"][0]["metadata"]["content_drift"] = "changed"
    with pytest.raises(P.ContractError) as error:
        P.select_for_phase(manifest, "diagnostic")
    assert error.value.code.startswith("manifest_")


def test_cluster_keeps_same_premise_in_one_purpose():
    near = record_for_purpose("ocnli", "diagnostic", 0)
    far = record_for_purpose("ocnli", "sealed_acceptance", 1)
    records = [make_record("ocnli", group=near["group_id"], premise="Merge me.",
                           hypothesis="a", rid="m-1"),
               make_record("ocnli", group=far["group_id"], premise="Merge me.",
                           hypothesis="b", rid="m-2")]
    partitioned = P.partition_records(records)
    assert len(set(partitioned["purposes"].values())) == 1
    report = P.check_leakage(partitioned)
    assert report["status"] == "ok" and report["clusters_with_multiple_raw_units"] == 1
    assert report["rows"] == 2


def test_partition_reports_holdout_warnings_honestly():
    records = [make_record("ocnli", group=f"prem-w{index}", premise="同一前提。",
                           hypothesis=f"h{index}", rid=f"w-{index}") for index in range(4)]
    partitioned = P.partition_records(records)
    assert partitioned["summary"]["total_clusters"] == 1
    assert "megacluster_majority_rows" in partitioned["summary"]["warnings"]
    assert "insufficient_cluster_units_for_holdout" in partitioned["summary"]["warnings"]
    report = P.check_leakage(partitioned)
    assert report["warnings"] and report["status"] == "ok"


def test_validate_manifest_rejects_every_tamper_channel():
    records = records_for_purpose("ocnli", "diagnostic", 1) + \
        records_for_purpose("ocnli", "sealed_acceptance", 1)
    base = P.partition_records(records)
    P.check_leakage(base)

    def expect(tampered, code):
        with pytest.raises(P.ContractError) as error:
            P.check_leakage(tampered)
        assert error.value.code == code

    tampered = copy.deepcopy(base)
    first_id = records[0]["id"]
    flipped = "sealed_acceptance" if base["purposes"][first_id] == "diagnostic" else "diagnostic"
    tampered["purposes"][first_id] = flipped
    expect(tampered, "manifest_purposes_mismatch")

    tampered = copy.deepcopy(base)
    tampered["units"] = {row["id"]: f"pretend-independent-{index}"
                         for index, row in enumerate(records)}
    expect(tampered, "manifest_units_mismatch")

    tampered = copy.deepcopy(base)
    tampered["clusters"] = []
    expect(tampered, "manifest_clusters_mismatch")

    tampered = copy.deepcopy(base)
    tampered["diagnostic"][0]["metadata"]["changed_after_freeze"] = "changed"
    expect(tampered, "manifest_sha_mismatch")

    tampered = copy.deepcopy(base)
    tampered["summary"]["total"] = tampered["summary"]["total"] + 1
    expect(tampered, "manifest_counts_mismatch")

    tampered = copy.deepcopy(base)
    del tampered["sha256"]
    expect(tampered, "invalid_manifest")

    tampered = copy.deepcopy(base)
    tampered["diagnostic"] = []
    expect(tampered, "manifest_purposes_mismatch")

    tampered = copy.deepcopy(base)
    diagnostic_row = tampered["diagnostic"][0]
    sealed_row = tampered["sealed_acceptance"][0]
    tampered["diagnostic"] = [sealed_row]
    tampered["sealed_acceptance"] = [diagnostic_row]
    tampered["sha256"] = {purpose: bucket_digest(tampered[purpose]) for purpose in P.PURPOSES}
    expect(tampered, "manifest_bucket_mismatch")
    with pytest.raises(P.ContractError) as error:
        P.manifest_scope(tampered, view_id="ocnli_nli3_native", phase="diagnostic")
    assert error.value.code == "manifest_bucket_mismatch"
    with pytest.raises(P.ContractError) as error:
        P.select_for_phase(tampered, "diagnostic")
    assert error.value.code == "manifest_bucket_mismatch"


def test_evaluate_view_binds_to_manifest_scope_not_caller_subset():
    records = records_for_purpose("ocnli", "diagnostic", 2)
    manifest = P.partition_records(records)
    results = {record["id"]: {"pred": "entailment", "truncated": False} for record in records}
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records[:1], {records[0]["id"]: results[records[0]["id"]]},
                        view_id="ocnli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "manifest_scope_mismatch"
    report = P.evaluate_view(records, results, view_id="ocnli_nli3_native",
                             phase="diagnostic", manifest=manifest)
    assert report["n_total"] == 2 and report["scope"]["binding"] == "manifest"
    assert report["scope"]["expected_rows"] == 2
    drifted = copy.deepcopy(records)
    drifted[0]["metadata"]["note"] = "mutated"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(drifted, results, view_id="ocnli_nli3_native",
                        phase="diagnostic", manifest=manifest)
    assert error.value.code == "record_drift"
    final = P.evaluate_view(records, results, view_id="ocnli_nli3_native",
                            phase="final_acceptance", manifest=manifest)
    P.assert_final_complete(final, manifest=manifest, results=results)


def test_no_manifest_fragment_path_is_rejected():
    """fragment 计分旁路已删除：自报集合不能绕过权威 manifest 曝光/计分。"""
    records = [make_record("ocnli", group="same-raw", premise="Same premise.", rid="f-1"),
               make_record("ocnli", group="same-raw", premise="Same premise.", rid="f-2")]
    partitioned = P.partition_records(records)
    results = {record["id"]: {"pred": "entailment", "truncated": False} for record in records}
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, results, view_id="ocnli_nli3_native",
                        phase="final_acceptance", purposes=partitioned["purposes"],
                        units=partitioned["units"])
    assert error.value.code == "manifest_required"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, results, view_id="ocnli_nli3_native",
                        phase="final_acceptance", manifest=partitioned,
                        purposes=partitioned["purposes"], units=partitioned["units"])
    assert error.value.code == "invalid_arguments"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, results, view_id="ocnli_nli3_native",
                        phase="final_acceptance")
    assert error.value.code == "manifest_required"


def test_sealed_rows_are_blocked_before_exposure_in_diagnostic_phase():
    row = record_for_purpose("ocnli", "sealed_acceptance", 0)
    manifest = P.partition_records([row])
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view([row], {row["id"]: {"pred": "entailment"}},
                        view_id="ocnli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "phase_violation"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(None, {row["id"]: {"pred": "entailment"}},
                        view_id="ocnli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "phase_violation"


                                                                             
           
                                                                             

def test_diagnostic_request_uses_text_only_and_predeclared_direction():
    record = make_record("ocnli")
    record["id"] = "SENTINEL-ID"
    record["metadata"]["SENTINEL"] = "SENTINEL-META"
    record["evidence"]["SENTINEL"] = "SENTINEL-PROV"
    primary = P.diagnostic_request(record)
    assert primary["candidate"]["text"] == record["hypothesis"]
    assert primary["existing"][0]["text"] == record["premise"]
    assert primary["evidence"] == [] and primary["candidate"]["trigger"] == ""
    swap = P.diagnostic_request(record, direction="premise_as_candidate")
    assert swap["candidate"]["text"] == record["premise"]
    assert swap["existing"][0]["text"] == record["hypothesis"]
    serialized = json.dumps([primary, swap], ensure_ascii=False)
    for sentinel in ("SENTINEL", "gold", "original_label", "prem_id"):
        assert sentinel not in serialized
    with pytest.raises(P.ContractError) as error:
        P.diagnostic_request(record, direction="average")
    assert error.value.code == "invalid_direction"


def _conflict_matrix(conflict_values):
    matrix = np.zeros((len(conflict_values), len(P.SEVEN_CLASSES)), dtype=np.float64)
    conflict = P.SEVEN_CLASSES.index("conflict")
    complement = P.SEVEN_CLASSES.index("complement")
    for index, value in enumerate(conflict_values):
        matrix[index, conflict] = value
        matrix[index, complement] = 1.0 - value
    return matrix


DIAG_GOLD = ["contradiction", "contradiction", "neutral", "entailment", None]
DIAG_UNITS = ["u1", "u2", "u3", "u4", "u5"]


def _diagnostic(scores, golds):
    matrix = _conflict_matrix(scores)
    return P.external_contradiction_diagnostic(
        {direction: matrix for direction in P.DIRECTIONS}, golds,
        units=[f"cluster-{index}" for index in range(len(scores))], samples=40)


def test_external_contradiction_diagnostic_is_task_incompatibility_probe():
    primary = _conflict_matrix([0.9, 0.8, 0.1, 0.2, 0.5])
    swap = _conflict_matrix([0.2, 0.9, 0.1, 0.8, 0.5])
    result = P.external_contradiction_diagnostic(
        {"hypothesis_as_candidate": primary, "premise_as_candidate": swap},
        DIAG_GOLD, units=DIAG_UNITS)
    assert result["task_equivalence"] == "none" and result["not_accuracy"] is True
    assert result["probe_kind"] == "task_incompatibility"
    assert result["n_with_gold"] == 4 and result["n_missing_gold"] == 1
    signal = result["by_direction"]["hypothesis_as_candidate"]["conflict_signal"]
    assert signal["rank_auc"] == pytest.approx(1.0)
    assert result["directions"]["primary"] == P.PRIMARY_DIRECTION
    assert result["flip_rate"]["flips"] == int(
        (primary.argmax(axis=1) != swap.argmax(axis=1)).sum())
    assert result["forbidden_label_mappings"] == [list(pair) for pair in P.FORBIDDEN_LABEL_MAPPINGS]
    dumped = json.dumps(result, ensure_ascii=False)
    assert '"accuracy"' not in dumped
    assert result["by_direction"]["hypothesis_as_candidate"]["mean_probability_by_gold"][
        "contradiction"]["conflict"] == pytest.approx(0.85)


def test_null_gold_is_not_a_negative_auc_example():
    full = _diagnostic([0.7, 0.3, 1.0], ["contradiction", "entailment", None])
    known = _diagnostic([0.7, 0.3], ["contradiction", "entailment"])
    full_signal = full["by_direction"][P.PRIMARY_DIRECTION]["conflict_signal"]
    known_signal = known["by_direction"][P.PRIMARY_DIRECTION]["conflict_signal"]
    assert full["n"] == 3 and full["n_missing_gold"] == 1
    assert full_signal["rank_auc"] == known_signal["rank_auc"] == pytest.approx(1.0)
    assert full_signal["n_negative"] == 1
    assert full_signal["n_missing_excluded"] == 1
    assert full_signal["bootstrap"]["group_clusters"] == 2
    shifted = _diagnostic([0.7, 0.3, 0.05], ["contradiction", "entailment", None])
    assert shifted["by_direction"][P.PRIMARY_DIRECTION]["conflict_signal"]["rank_auc"] == \
        pytest.approx(1.0)


def test_all_missing_gold_has_no_auc():
    output = _diagnostic([0.9, 0.1], [None, None])
    signal = output["by_direction"][P.PRIMARY_DIRECTION]["conflict_signal"]
    assert signal["rank_auc"] is None and signal["bootstrap"] is None
    assert signal["n_labeled"] == 0


def test_probability_distribution_rejects_non_normalized_rows():
    matrix = np.full((2, len(P.SEVEN_CLASSES)), 0.5)
    with pytest.raises(P.ContractError) as error:
        P.external_contradiction_diagnostic({direction: matrix for direction in P.DIRECTIONS},
                                            ["contradiction", "entailment"], units=["a", "b"])
    assert error.value.code == "invalid_probs"


def test_external_contradiction_diagnostic_rejects_seven_class_gold_and_shapes():
    matrix = _conflict_matrix([0.5] * 5)
    payload = {"hypothesis_as_candidate": matrix, "premise_as_candidate": matrix}
    with pytest.raises(P.ContractError) as error:
        P.external_contradiction_diagnostic(payload, ["conflict"] * 5, units=DIAG_UNITS)
    assert error.value.code == "invalid_gold"
    with pytest.raises(P.ContractError) as error:
        P.external_contradiction_diagnostic({"hypothesis_as_candidate": matrix},
                                            DIAG_GOLD, units=DIAG_UNITS)
    assert error.value.code == "invalid_probs"
    with pytest.raises(P.ContractError) as error:
        P.external_contradiction_diagnostic(
            {"hypothesis_as_candidate": np.zeros((5, 3)), "premise_as_candidate": matrix},
            DIAG_GOLD, units=DIAG_UNITS)
    assert error.value.code == "invalid_probs"


def test_external_contradiction_diagnostic_strata_are_descriptive_only():
    primary = _conflict_matrix([0.9, 0.8, 0.1, 0.2, 0.5])
    strata = [{"heuristic": "lexical_overlap"}, {"heuristic": "lexical_overlap"},
              {"heuristic": "subsequence"}, {"heuristic": "subsequence"},
              {"heuristic": "constituent"}]
    result = P.external_contradiction_diagnostic(
        {"hypothesis_as_candidate": primary, "premise_as_candidate": primary},
        DIAG_GOLD, units=DIAG_UNITS, strata=strata)
    by_stratum = result["by_direction"]["hypothesis_as_candidate"]["mean_probability_by_stratum"]
    assert set(by_stratum["heuristic"]) == {"lexical_overlap", "subsequence", "constituent"}


                                                                             
                          
                                                                             

def test_evaluate_view_reports_denominators_failures_and_truncation():
    records = records_for_purpose("ocnli", "diagnostic", 4)
    golds = ["entailment", "contradiction", "neutral", "entailment"]
    for record, gold in zip(records, golds):
        record["gold_label"] = gold
        record["metadata"]["original_label"] = ORIGINALS["ocnli"][gold]
    results = {
        records[0]["id"]: {"pred": "entailment", "truncated": False},                
        records[1]["id"]: {"pred": "contradiction", "truncated": True},                        
        records[2]["id"]: {"error": "inference_failed"},                            
        records[3]["id"]: {},                                                              
    }
    manifest, report = manifest_evaluate(records, results, view_id="ocnli_nli3_native",
                                         phase="diagnostic")
    assert report["n_total"] == 4
    assert report["counts"] == {"scored": 2, "failed": 1, "not_attempted": 1,
                                "with_gold": 4, "scored_with_gold": 2, "without_gold": 0,
                                "truncated": 1, "units": 4}
    assert report["accuracy"] == {"value": 1.0, "numerator": 2, "denominator": 2}
    assert report["coverage"]["value"] == pytest.approx(0.5)
    assert report["attempt_rate"]["value"] == pytest.approx(0.75)
    assert report["failure_rate"]["value"] == pytest.approx(0.25)
    assert report["truncation"]["count"] == 1
    assert report["truncation"]["evidence"] == {"explicit": 2, "total": 4}
    assert report["by_truncation"]["truncated"]["n_total"] == 1
    assert report["by_truncation"]["untruncated"]["n_total"] == 1
    assert report["by_truncation"]["unknown"]["n_total"] == 2
    assert report["completion"]["complete"] is False
    assert report["claim"] == "standard"
    assert report["macro_f1"]["value"] == pytest.approx(2.0 / 3.0)
    assert report["bootstrap"]["row"]["samples"] == 500
    assert report["bootstrap"]["group_equal"]["group_clusters"] == 2
    assert report["bootstrap"]["row"]["denominator"] == 2
    assert report["scope"]["record_ids_sha256"] == \
        P.manifest_scope(manifest, view_id="ocnli_nli3_native", phase="diagnostic")[
            "record_ids_sha256"]


def test_evaluate_view_counts_null_gold_without_fabrication():
    record = record_for_purpose("ocnli", "diagnostic", 0)
    record["gold_label"] = None
    record["metadata"]["gold_status"] = "no_consensus"
    record["metadata"]["label_method"] = "majority_vote"
    record["metadata"]["original_label"] = "-"
    manifest, report = manifest_evaluate([record], {record["id"]: {"pred": None}},
                                         view_id="ocnli_nli3_native", phase="diagnostic")
    assert report["counts"]["with_gold"] == 0 and report["counts"]["without_gold"] == 1
    assert report["accuracy"] == {"value": None, "numerator": 0, "denominator": 0}
    assert report["coverage"]["value"] is None


def test_evaluate_view_stratifies_and_rejects_invalid_outcomes():
    records = records_for_purpose("anli", "diagnostic", 2)
    results = {records[0]["id"]: {"pred": records[0]["gold_label"], "truncated": False},
               records[1]["id"]: {"pred": records[1]["gold_label"], "truncated": False}}
    manifest, report = manifest_evaluate(records, results, view_id="anli_nli3_native",
                                         phase="diagnostic")
    assert "round" in report["by_stratum"]
    record = records[0]

    def tampered(entry):
        merged = dict(results)
        merged[record["id"]] = entry
        return merged

    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, tampered({"pred": "conflict", "truncated": False}),
                        view_id="anli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "cross_task_prediction"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, tampered({"pred": "maybe", "truncated": False}),
                        view_id="anli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "invalid_prediction"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, tampered({"pred": "entailment", "error": "x",
                                           "truncated": False}),
                        view_id="anli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "invalid_outcome"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, {}, view_id="anli_nli3_native", phase="diagnostic",
                        manifest=manifest)
    assert error.value.code == "missing_results"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, dict(results, ghost={}), view_id="anli_nli3_native",
                        phase="diagnostic", manifest=manifest)
    assert error.value.code == "unexpected_results"
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view(records, results, view_id="anli_diag7_w384",
                        phase="diagnostic", manifest=manifest)
    assert error.value.code == "diagnostic_view_not_scorable"


def test_scored_rows_require_valid_truncation_evidence():
    record = record_for_purpose("ocnli", "diagnostic", 0)
    manifest = P.partition_records([record])
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view([record], {record["id"]: {"pred": "entailment", "truncated": "yes"}},
                        view_id="ocnli_nli3_native", phase="diagnostic", manifest=manifest)
    assert error.value.code == "invalid_outcome"
    report = P.evaluate_view([record], {record["id"]: {"pred": "entailment"}},
                             view_id="ocnli_nli3_native", phase="diagnostic", manifest=manifest)
    assert report["truncation"]["unknown"] == 1
    assert report["truncation"]["evidence"]["explicit"] == 0


def test_full_document_claim_requires_explicit_zero_truncation_evidence():
    record = record_for_purpose("contractnli", "diagnostic", 0)
    manifest = P.partition_records([record])
    omitted = P.evaluate_view([record], {record["id"]: {"pred": "entailment"}},
                              view_id="contractnli_nli3_full_document",
                              phase="diagnostic", manifest=manifest)
    assert omitted["claim"] != "full_document"
    truncated = P.evaluate_view([record], {record["id"]: {"pred": "entailment",
                                                          "truncated": True}},
                                view_id="contractnli_nli3_full_document",
                                phase="diagnostic", manifest=manifest)
    assert truncated["claim"] == "windowed_not_full"
    explicit = P.evaluate_view([record], {record["id"]: {"pred": "entailment",
                                                         "truncated": False}},
                               view_id="contractnli_nli3_full_document",
                               phase="diagnostic", manifest=manifest)
    assert explicit["claim"] == "full_document"
    eligible = make_record("contractnli", gold="entailment", group="doc-oracle",
                           premise="Oracle doc premise.", spans=[{"text": "Clause."}])
    oracle_manifest = P.partition_records([eligible])
    oracle = P.evaluate_view([eligible], {eligible["id"]: {"pred": "entailment",
                                                           "truncated": False}},
                             view_id="contractnli_nli3_oracle_evidence",
                             phase="diagnostic", manifest=oracle_manifest)
    assert oracle["claim"] == "oracle_evidence"


def test_assert_final_complete_rederives_from_manifest_and_results():
    """完成不再是报告内旗子：由 manifest+完整 results 重新派生并与报告逐字段比对。"""
    record = record_for_purpose("ocnli", "diagnostic", 0)
    manifest = P.partition_records([record])
    idle = {record["id"]: {}}
    partial = P.evaluate_view([record], idle, view_id="ocnli_nli3_native",
                              phase="final_acceptance", manifest=manifest)
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(partial, manifest=manifest, results=idle)
    assert error.value.code == "incomplete_final"
    edited = copy.deepcopy(partial)
    edited["completion"]["complete"] = True
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(edited, manifest=manifest, results=idle)
    assert error.value.code == "report_inconsistent"
    done = {record["id"]: {"pred": "entailment", "truncated": False}}
    complete = P.evaluate_view([record], done, view_id="ocnli_nli3_native",
                               phase="final_acceptance", manifest=manifest)
    P.assert_final_complete(complete, manifest=manifest, results=done)
    forged = copy.deepcopy(complete)
    forged["scope"]["binding"] = "fragment"
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(forged, manifest=manifest, results=done)
    assert error.value.code == "report_inconsistent"
    forged_ci = copy.deepcopy(complete)
    forged_ci["bootstrap"]["row"]["ci"] = [0.0, 0.01]
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(forged_ci, manifest=manifest, results=done)
    assert error.value.code == "report_inconsistent"
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(complete)
    assert error.value.code == "manifest_required"
    diagnostic_phase = P.evaluate_view([record], done, view_id="ocnli_nli3_native",
                                       phase="diagnostic", manifest=manifest)
    with pytest.raises(P.ContractError) as error:
        P.assert_final_complete(diagnostic_phase, manifest=manifest, results=done)
    assert error.value.code == "not_final_phase"


def test_oracle_scope_accounts_neutral_and_missing_spans():
    neutral = make_record("contractnli", gold="neutral", group="doc-n1",
                          premise="Neutral doc premise.", spans=[])
    entailed = make_record("contractnli", gold="entailment", group="doc-e1",
                           premise="Entailed doc premise.", spans=[{"text": "Clause E."}])
    contradicted = make_record("contractnli", gold="contradiction", group="doc-c1",
                               premise="Contradicted doc premise.",
                               spans=[{"text": "Clause C."}])
    no_span = make_record("contractnli", gold="entailment", group="doc-e2",
                          premise="No span premise.")
    records = [neutral, entailed, contradicted, no_span]
    manifest = P.partition_records(records)
    scope = P.manifest_scope(manifest, view_id="contractnli_nli3_oracle_evidence",
                             phase="final_acceptance")
    assert scope["accounting"] == {"rule": "gold_evidence_two_class", "n_official": 4,
                                   "n_eligible": 2, "n_neutral_ineligible": 1,
                                   "n_span_unavailable": 1}
    assert set(scope["ids"]) == {entailed["id"], contradicted["id"]}
    with pytest.raises(P.ContractError) as error:
        P.evaluate_view([neutral], {neutral["id"]: {"pred": "neutral", "truncated": False}},
                        view_id="contractnli_nli3_oracle_evidence", phase="final_acceptance",
                        manifest=manifest)
    assert error.value.code == "manifest_scope_mismatch"
    report = P.evaluate_view(
        [entailed, contradicted],
        {entailed["id"]: {"pred": "entailment", "truncated": False},
         contradicted["id"]: {"pred": "contradiction", "truncated": False}},
        view_id="contractnli_nli3_oracle_evidence", phase="final_acceptance", manifest=manifest)
    assert report["task"] == "entailment_contradiction"
    assert report["claim"] == "oracle_evidence"
    assert report["scope"]["accounting"]["n_neutral_ineligible"] == 1


def test_oracle_accounting_splits_missing_from_neutral():
    """缺标不是 neutral：missing 单独记账并参与 n_official 守恒。"""
    eligible = make_record("contractnli", gold="entailment", group="doc-e1",
                           premise="Eligible premise.", spans=[{"text": "Clause E."}])
    neutral = make_record("contractnli", gold="neutral", group="doc-n1",
                          premise="Neutral premise.", spans=[])
    missing = make_record("contractnli", gold=None, group="doc-m1", premise="Missing premise.")
    manifest = P.partition_records([eligible, neutral, missing])
    scope = P.manifest_scope(manifest, view_id="contractnli_nli3_oracle_evidence",
                             phase="final_acceptance")
    accounting = scope["accounting"]
    assert accounting["n_official"] == 3 and accounting["n_eligible"] == 1
    assert accounting["n_neutral_ineligible"] == 1
    assert accounting["n_missing_gold"] == 1
    assert accounting["n_span_unavailable"] == 0
    assert "n_no_consensus_gold" not in accounting
    conserved = (accounting["n_eligible"] + accounting["n_neutral_ineligible"]
                 + accounting["n_span_unavailable"] + accounting["n_missing_gold"])
    assert conserved == accounting["n_official"]
    assert scope["ids"] == [eligible["id"]]


                                                                             
                             
                                                                             

def _admission_context(*, count=30, correct=15, language_matched=True,
                       baselines=("majority_class", "chance"), attribution=None,
                       counterexamples="all_failures"):
    """30 行均衡 gold（10/10/10）的合成准入上下文：majority/chance 真值均为 1/3。"""
    labels = ("entailment", "contradiction", "neutral")
    records = [make_record("ocnli", gold=labels[index % len(labels)],
                           group=f"adm-{index}", premise=f"Admission premise {index}.",
                           hypothesis=f"Admission hypothesis {index}.", rid=f"adm-{index}")
               for index in range(count)]
    manifest = P.partition_records(records)
    results = {}
    failures = []
    for index, record in enumerate(records):
        gold = record["gold_label"]
        if index < correct:
            results[record["id"]] = {"pred": gold, "truncated": False}
        else:
            results[record["id"]] = {"pred": labels[(labels.index(gold) + 1) % len(labels)],
                                     "truncated": False}
            failures.append(record["id"])
    report = P.evaluate_view(None, results, view_id="ocnli_nli3_native",
                             phase="final_acceptance", manifest=manifest)
    if attribution is None:
        attribution = attribution_entries("matched_task")
    if counterexamples == "all_failures":
        counterexamples = list(failures)
    return {"records": records, "manifest": manifest, "results": results, "report": report,
            "baselines": list(baselines), "attribution": attribution,
            "counterexamples": counterexamples, "language_matched": language_matched}


def _run_admission(context, **overrides):
    kwargs = {"view_id": "ocnli_nli3_native", "view_report": context["report"],
              "baselines": context["baselines"],
              "counterexample_units": context["counterexamples"],
              "attribution_classes": context["attribution"],
              "language_matched": context["language_matched"],
              "phase": "final_acceptance",
              "manifest": context["manifest"], "results": context["results"]}
    kwargs.update(overrides)
    return P.admission_check(None, **kwargs)


def test_admission_reports_evidence_and_computed_baselines_without_authorization():
    context = _admission_context()
    assert context["report"]["accuracy"]["value"] == 0.5
    verdict = _run_admission(context)
    assert verdict["evidence_complete"] is True
    assert verdict["manual_evidence_review_required"] is True
    assert not ({"poor_confirmed", "retrain_admissible", "sub_capability_signal_only",
                 "evidence_sufficient"} & set(verdict))
    comparison = verdict["comparison"]
    assert comparison["descriptive_only"] is True
    by_name = {item["name"]: item for item in comparison["baselines"]}
    assert by_name["majority_class"]["value"] == pytest.approx(1 / 3)
    assert by_name["majority_class"]["numerator"] == 10
    assert by_name["majority_class"]["denominator"] == 30
    assert by_name["chance"]["value"] == pytest.approx(1 / 3)
    assert by_name["chance"]["classes"] == 3
    assert comparison["strongest_baseline"]["value"] == pytest.approx(1 / 3)
    assert comparison["ci_upper_below_strongest"] is False
    assert "ci_upper_not_below_strongest_registered_baseline" in verdict["reasons"]


def test_admission_worse_than_baseline_still_requires_manual_review():
    context = _admission_context(correct=0)
    verdict = _run_admission(context)
    assert verdict["evidence_complete"] is True
    assert verdict["comparison"]["ci_upper_below_strongest"] is True
    assert "ci_upper_not_below_strongest_registered_baseline" not in verdict["reasons"]
    assert "poor_confirmed" not in verdict and "retrain_admissible" not in verdict


def test_admission_rejects_fabricated_empty_report():
    fake = {"view_id": "ocnli_nli3_native", "phase": "diagnostic",
            "bootstrap": {"row": {"ci": [0.0, 0.1]}}}
    with pytest.raises(P.ContractError) as error:
        P.admission_check(None, view_id="ocnli_nli3_native", view_report=fake,
                          baselines=[{"name": "lexical_overlap", "accuracy": 0.99}],
                          counterexample_units=10, attribution_classes=["matched_task"],
                          language_matched=True, phase="diagnostic")
    assert error.value.code == "invalid_report"


def test_admission_rejects_fabricated_scalar_baselines():
    """标量声明不再能伪造完美基线；chance/majority 由验证器按冻结定义复算。"""
    context = _admission_context()
    forged = [{"name": "chance", "label_space": "nli3", "supervision": "gold_supervised",
               "record_ids_sha256": context["report"]["scope"]["record_ids_sha256"],
               "numerator": 30, "denominator": 30, "accuracy": 1.0}]
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, baselines=forged)
    assert error.value.code == "invalid_baseline"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, baselines=["chance", {"name": "chance", "predictions": {}}])
    assert error.value.code == "invalid_baseline"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, baselines=[])
    assert error.value.code == "invalid_baseline"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, baselines=["my_own_baseline"])
    assert error.value.code == "unknown_baseline"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, baselines=["lexical_overlap"])
    assert error.value.code == "baseline_label_space_mismatch"


def test_admission_empirical_baseline_requires_full_predictions():
    records = plain_records("hans", 6)
    manifest = P.partition_records(records)
    results = {record["id"]: {"pred": "entailment", "truncated": False} for record in records}
    report = P.evaluate_view(None, results, view_id="hans_entailment2_native",
                             phase="final_acceptance", manifest=manifest)
    profile_recipe = P.load_study()["admission"]["baseline_profiles"]["lexical_overlap"]["recipe"]
    predictions = {record["id"]: record["gold_label"] for record in records}
    request = {"name": "lexical_overlap", "predictions": predictions, "recipe": profile_recipe}
    verdict = P.admission_check(None, view_id="hans_entailment2_native", view_report=report,
                                baselines=[request], counterexample_units=[],
                                attribution_classes=[], language_matched=False,
                                phase="final_acceptance", manifest=manifest, results=results)
    baseline = verdict["comparison"]["baselines"][0]
    assert baseline["value"] == 1.0
    assert baseline["numerator"] == 6 and baseline["denominator"] == 6
    assert baseline["implementation_verified"] is False

    def check(bad_request):
        with pytest.raises(P.ContractError) as error:
            P.admission_check(None, view_id="hans_entailment2_native", view_report=report,
                              baselines=[bad_request], counterexample_units=[],
                              attribution_classes=[], language_matched=False,
                              phase="final_acceptance", manifest=manifest, results=results)
        return error.value.code

    assert check(dict(request, recipe="rough-guess")) == "invalid_baseline"
    assert check(dict(request, predictions={records[0]["id"]: "entailment"})) \
        == "baseline_member_mismatch"
    assert check(dict(request, predictions=dict(predictions,
                                                **{records[0]["id"]: "maybe"}))) \
        == "invalid_baseline"


def test_admission_incomplete_selection_is_a_technical_report_only():
    context = _admission_context(count=20, correct=2)
    for record_id in list(context["results"]):
        context["results"][record_id] = {"error": "inference_failed"}
    context["results"][context["records"][0]["id"]] = {"pred": "contradiction",
                                                       "truncated": False}
    context["report"] = P.evaluate_view(None, context["results"], view_id="ocnli_nli3_native",
                                        phase="final_acceptance", manifest=context["manifest"])
    assert context["report"]["counts"]["failed"] == 19
    verdict = _run_admission(context)
    assert verdict["evidence_complete"] is False
    assert verdict["reasons"] == ["incomplete_selection"]
    assert verdict["comparison"] is None
    assert verdict["manual_evidence_review_required"] is True
    assert not ({"poor_confirmed", "retrain_admissible"} & set(verdict))


def test_admission_requires_manifest_results_and_matching_scope():
    context = _admission_context()
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, manifest=None)
    assert error.value.code == "manifest_required"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, results=None)
    assert error.value.code == "results_required"
    wider = P.partition_records(context["records"] + plain_records("ocnli", 1, start=99))
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, manifest=wider)
    assert error.value.code == "report_scope_mismatch"


def test_admission_report_edits_and_fragment_forgery_fail_loud():
    context = _admission_context()
    forged_binding = copy.deepcopy(context["report"])
    forged_binding["scope"]["binding"] = "fragment"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_report=forged_binding)
    assert error.value.code == "unbound_report"
    ci_edit = copy.deepcopy(context["report"])
    ci_edit["bootstrap"]["row"]["ci"] = [0.0, 0.01]
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_report=ci_edit)
    assert error.value.code == "report_inconsistent"
    accuracy_edit = copy.deepcopy(context["report"])
    accuracy_edit["accuracy"] = {"value": 0.9, "numerator": 27, "denominator": 30}
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_report=accuracy_edit)
    assert error.value.code == "report_inconsistent"
    completion_edit = copy.deepcopy(context["report"])
    completion_edit["completion"]["failed"] = 3
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_report=completion_edit)
    assert error.value.code == "invalid_report"


def test_admission_counterexamples_come_from_real_failures():
    context = _admission_context()
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, counterexample_units=["not-a-row"])
    assert error.value.code == "counterexample_not_a_failure"
    few = context["counterexamples"][:3]
    verdict = _run_admission(context, counterexample_units=few)
    assert len(verdict["comparison"]["counterexample_clusters"]) == 3
    assert verdict["comparison"]["counterexamples_complete"] is False
    assert "insufficient_pre_registered_counterexample_clusters" in verdict["reasons"]
    full = _run_admission(context)
    assert full["comparison"]["counterexamples_complete"] is True


def test_admission_claims_are_recorded_but_not_validated():
    context = _admission_context(
        attribution=attribution_entries("supervision_unfair_gap", "matched_task"))
    verdict = _run_admission(context)
    claims = verdict["comparison"]["attribution_claims"]
    assert claims and all(claim["validated"] is False for claim in claims)
    assert any(claim["class"] == "supervision_unfair_gap" for claim in claims)
    assert "supervision_unfair_gap_is_context_only_never_strong_attribution" in verdict["reasons"]
    assert "attribution_claims_unvalidated_manual_review_required" in verdict["reasons"]
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, attribution_classes=["matched_task"])
    assert error.value.code == "invalid_attribution"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, attribution_classes=[{"class": "matched_task", "evidence": "x"}])
    assert error.value.code == "invalid_attribution"
    unmatched = _run_admission(_admission_context(language_matched=False))
    assert "no_language_matched_evidence" in unmatched["reasons"]
    assert unmatched["comparison"]["language_matched"] is False


def test_admission_requires_native_view_and_phase_match():
    context = _admission_context()
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_id="ocnli_diag7_w384")
    assert error.value.code == "admission_requires_native_view"
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, phase="diagnostic")
    assert error.value.code == "report_mismatch"


def test_admission_partial_report_must_match_supplied_authority():
    """r3 P2-2(a)：陈旧/伪造的“不完整”报告配上真实 manifest+results 必须拒，不得回显其计数。"""
    context = _admission_context(correct=30)
    forged = copy.deepcopy(context["report"])
    forged["counts"].update(scored=0, failed=30, not_attempted=0, scored_with_gold=0)
    forged["accuracy"] = {"value": None, "numerator": 0, "denominator": 0}
    forged["bootstrap"]["row"]["denominator"] = 0
    forged["bootstrap"]["row"]["ci"] = None
    forged["completion"] = {"failed": 30, "not_attempted": 0, "complete": False}
    with pytest.raises(P.ContractError) as error:
        _run_admission(context, view_report=forged)
    assert error.value.code == "report_inconsistent"


def test_admission_bare_partial_declaration_is_explicitly_unverified():
    """无权威输入时：只有否定性“不完整”声明可被回显，并明确标注未验证；完整声明一律拒。"""
    records = plain_records("ocnli", 3)
    manifest = P.partition_records(records)
    failed = {record["id"]: {"error": "fixture-failure"} for record in records}
    report = P.evaluate_view(None, failed, view_id="ocnli_nli3_native",
                             phase="final_acceptance", manifest=manifest)
    verdict = P.admission_check(None, view_id="ocnli_nli3_native", view_report=report,
                                baselines=[{"name": "majority_class", "accuracy": 1.0}],
                                counterexample_units=10, attribution_classes=["matched_task"],
                                language_matched=True, phase="final_acceptance")
    assert verdict["evidence_complete"] is False
    assert verdict["manual_evidence_review_required"] is True
    assert verdict["evidence_source"] == "unverified_caller_declaration"
    assert verdict["reasons"] == ["incomplete_selection", "unverified_caller_declaration"]
    assert verdict["comparison"] is None
    assert not ({"poor_confirmed", "retrain_admissible"} & set(verdict))
    complete = {record["id"]: {"pred": "entailment", "truncated": False} for record in records}
    report_ok = P.evaluate_view(None, complete, view_id="ocnli_nli3_native",
                                phase="final_acceptance", manifest=manifest)
    with pytest.raises(P.ContractError) as error:
        P.admission_check(None, view_id="ocnli_nli3_native", view_report=report_ok,
                          baselines=["chance"], counterexample_units=[],
                          attribution_classes=[], language_matched=True,
                          phase="final_acceptance")
    assert error.value.code == "manifest_required"


@pytest.mark.parametrize("state", ["failed", "unattempted", "missing_gold", "zero_eligible"])
def test_admission_zero_scored_gold_is_insufficient_not_malformed(state):
    """r3 P2-2(b)：合法零观测（含空 scope）返回不足证据 descriptor，不当坏 payload、不判许可。"""
    if state == "zero_eligible":
        records = [make_record("contractnli", gold="neutral", group="zero-n1",
                               premise="Zero eligible premise.", spans=[])]
        view_id = "contractnli_nli3_oracle_evidence"
        results = {}
    else:
        gold = None if state == "missing_gold" else "entailment"
        records = [make_record("ocnli", gold=gold, group=f"zero-{state}-{index}",
                               premise=f"Zero premise {state} {index}.",
                               hypothesis=f"Zero hypothesis {state} {index}.",
                               rid=f"zero-{state}-{index}") for index in range(3)]
        view_id = "ocnli_nli3_native"
        if state == "failed":
            results = {record["id"]: {"error": "fixture-failure"} for record in records}
        elif state == "unattempted":
            results = {record["id"]: {} for record in records}
        else:
            results = {record["id"]: {"pred": "entailment", "truncated": False}
                       for record in records}
    manifest = P.partition_records(records)
    report = P.evaluate_view(None, results, view_id=view_id,
                             phase="final_acceptance", manifest=manifest)
    assert report["accuracy"]["denominator"] == 0 and report["bootstrap"] == {}
    verdict = P.admission_check(None, view_id=view_id, view_report=report,
                                baselines=["chance"], counterexample_units=[],
                                attribution_classes=[], language_matched=True,
                                phase="final_acceptance", manifest=manifest, results=results)
    assert verdict["evidence_complete"] is False
    assert verdict["manual_evidence_review_required"] is True
    assert verdict["comparison"] is None
    assert not ({"poor_confirmed", "retrain_admissible"} & set(verdict))
    counts = verdict["counts"]
    assert counts["scored"] + counts["failed"] + counts["not_attempted"] == report["n_total"]
    assert counts["with_gold"] + counts["without_gold"] == report["n_total"]
    assert counts["scored_with_gold"] <= counts["scored"]
    assert verdict["selection"]["n_total"] == report["n_total"]
    if state in ("failed", "unattempted"):
        assert verdict["reasons"] == ["incomplete_selection"]
    else:
        assert verdict["reasons"] == ["no_scored_observations"]
