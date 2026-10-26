"""训练/特征/指标的研究专项测试：tiny 自合成 fixture，不构造正式挑战数据。

跑法：pytest research/memory_relations/test_training.py -q（需 PYTHONPATH=src;仓库根）。
MiniLM 相关用例只在本地权重可见时运行（禁止下载）；不可见则 skip 并写明原因。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dataset_recommender.agent import memory_relations as M
from dataset_recommender.llm.llm_client import LLMConfig
from research.memory_relations import features as F
from research.memory_relations import metrics as EM
from research.memory_relations import train as T

SPEC_PATH = Path(__file__).resolve().parent / "spec.json"

_CLASS_TEXTS = {
    "complement": ("先做校正，再做汇总，并附核对表。", "先做校正，再做汇总。"),
    "conflict": ("汇总必须两行。", "汇总必须只有一行。"),
    "contextual": ("小样本模式下汇总写两行；常规模式仍一行。", "汇总写一行。"),
    "duplicate": ("先做校正，再做汇总。", "做校正，然后汇总。"),
    "supersede": ("汇总永久改为两行，不再一行。", "汇总写一行。"),
    "temporary": ("本轮先写两行汇总，下轮恢复。", "汇总写一行。"),
    "uncertain": ("有时两行有时一行，看情况。", "汇总写一行。"),
}


def _build_request(candidate_text: str, trigger: str, base_text: str, index: int) -> dict:
    quote = f"出处{index}：{base_text}"
    return {
        "candidate": {"text": candidate_text, "trigger": trigger, "tier": "explicit",
                      "evidence": [quote]},
        "existing": [{"id": "r0", "text": base_text, "trigger": "汇总时", "tier": "explicit",
                      "evidence": []}],
        "evidence": [{"id": "e0", "text": quote}],
        "context": {"scope": "project", "replaceable_ids": ["r0"], "inherited_global_ids": []},
    }


def _row(row_id: str, group_id: str, split: "str | None", label: str, index: int, *,
         label_origin: str = "construction") -> dict:
    candidate_text, base_text = _CLASS_TEXTS[label]
    row = {"id": row_id, "group_id": group_id, "label": label, "label_origin": label_origin,
           "related_ids": [], "evidence_refs": [], "phenomena": [],
           "request": _build_request(f"{candidate_text}（版本{index}）",
                                     "汇总时" if index % 2 else "", base_text, index)}
    if split is not None:
        row["split"] = split
    return row


def _multi_rule_row(row_id: str, group_id: str, split: "str | None" = None) -> dict:
    """整集形状行：existing>1，配对模型必须排除（不静默做均值特征）。"""
    request = _build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0)
    request["existing"] = request["existing"] + [
        {"id": "r1", "text": "汇总写一行。", "trigger": "汇总时", "tier": "explicit",
         "evidence": []}]
    row = {"id": row_id, "group_id": group_id, "label": "duplicate", "label_origin": "construction",
           "related_ids": [], "evidence_refs": [], "phenomena": [], "request": request}
    if split is not None:
        row["split"] = split
    return row


def _set_row(row_id: str, group_id: str) -> dict:
    row = _multi_rule_row(row_id, group_id, "set")
    row["request"] = {"candidate": {"text": "集合样例"}, "existing": [], "evidence": []}
    return row


def _fixture() -> "tuple[list[dict], list[dict]]":
    data_rows: list[dict] = []
    challenge_rows: list[dict] = []
    for label in sorted(_CLASS_TEXTS):
        for index in range(3):
            data_rows.append(_row(f"train-{label}-{index}", f"train-g-{label}", "train",
                                  label, index))
        for index in range(2):
            data_rows.append(_row(f"cal-{label}-{index}", f"cal-g-{label}", "calibration",
                                  label, index))
        data_rows.append(_row(f"heldout-{label}", f"heldout-g-{label}", "heldout", label, 0))
        for index in range(2):
            challenge_rows.append(_row(f"challenge-{label}-{index}", f"challenge-g-{label}",
                                       None, label, index, label_origin="agent_review"))
    data_rows.append(_multi_rule_row("train-multi", "train-g-multi", "train"))
    data_rows.append(_set_row("set-0", "set-g-0"))
    challenge_rows.append(_multi_rule_row("challenge-multi", "challenge-g-multi"))
    return data_rows, challenge_rows


def _write_jsonl(path: Path, rows: "list[dict]") -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _train_pair_rows() -> "list[dict]":
    return [row for row in _fixture()[0]
            if row.get("split") == "train" and len(row["request"]["existing"]) <= 1]


def _fit_tiny_char_scorer():
    rows = _train_pair_rows()
    texts = [F.request_text(row["request"]) for row in rows]
    labels = np.asarray([F.CLASSES.index(row["label"]) for row in rows])
    model, class_indices = T.fit_model("char-logreg", texts, labels, 13)
    return F.RelationScorer(kind="char-logreg", model=model, class_indices=class_indices), rows


def _projection_pair() -> "tuple[dict, dict]":
    """stored（落盘 request）与真实 M 投影 request 的同源对，用于特征/概率 parity。"""
    cfg = LLMConfig(enable_llm=True, api_key=None, base_url="https://model.example/v1",
                    model="decision-model")
    endpoint = hashlib.sha256(f"{cfg.base_url}|{cfg.model}".encode()).hexdigest()[:12]
    scope = {"principal": "user-a", "endpoint_fp": endpoint, "project_id": "project-a"}
    quote = "出处0：做校正，然后汇总。"
    candidate = {**scope, "text": "先做校正，再做汇总。", "trigger": "汇总时", "tier": "explicit",
                 "evidence": [quote]}
    existing = [{**scope, "id": "r1", "text": "做校正，然后汇总。", "trigger": "汇总时",
                 "tier": "explicit", "evidence": []}]
    evidence = [{"id": "e1", "text": quote}]
    captured: list[dict] = []

    def predictor(request):
        captured.append(request)
        return {"relation": "uncertain", "related_ids": [], "reason": "parity probe",
                "evidence_refs": [], "error_code": None}

    out = M.assess_relations(candidate, existing, evidence, cfg, predictor=predictor)
    assert out["relation"] == "uncertain" and captured, "真实 M 投影未产生请求"
    stored = {
        "candidate": {"text": candidate["text"], "trigger": candidate["trigger"],
                      "tier": "implicit",                              
                      "evidence": [quote]},
        "existing": [{"id": "r1", "text": existing[0]["text"], "trigger": "汇总时",
                      "tier": "implicit", "evidence": []}],
        "evidence": [{"id": "e1", "text": quote}],
        "context": {"scope": "project", "replaceable_ids": ["r1"], "inherited_global_ids": []},
    }
    return stored, captured[0]


def _embedding_model_path() -> "Path | None":
    candidates = [
        os.environ.get("MEMORY_RELATIONS_EMBEDDING_MODEL", ""),
        str(Path(__file__).resolve().parents[2] / "models" / "embeddings" / F.EMBEDDING_DEFAULT_NAME),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def _probability_row(**values) -> np.ndarray:
    row = np.zeros(len(F.CLASSES))
    for name, value in values.items():
        row[F.CLASSES.index(name)] = value
    return row


def test_classes_and_safety_match_spec_and_relations():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert list(F.CLASSES) == spec["classes"]
    assert set(F.CLASSES) == set(M.RELATIONS)
    assert F.FEATURE_RECIPE == "memory-relations-features-v2"
    assert list(EM.ALLOWING_CLASSES) == spec["safety_metrics"]["allowing_classes"]
    assert list(EM.BLOCKING_CLASSES) == spec["safety_metrics"]["blocking_classes"]
    assert EM.CONTEXTUAL_REQUIRES_TRIGGER is spec["safety_metrics"]["contextual_requires_trigger"]
    assert EM.TARGET_DANGEROUS_ALLOW == spec["safety_metrics"]["calibration_target_dangerous_allow"]
    assert EM.MIN_COVERAGE == spec["safety_metrics"]["calibration_min_coverage"]
    assert EM.BOOTSTRAP_SAMPLES == spec["evaluation"]["bootstrap_samples"]
    assert EM.CONFIDENCE == spec["evaluation"]["confidence"]
    assert spec["evaluation"]["bootstrap_clusters"] == "group_id"


def test_nested_group_sampling_is_nested_and_group_integral():
    rows = [{"group_id": f"g{index // 2}", "id": f"r{index}"} for index in range(10)]
    first = T.nested_group_indices(rows, [4, 7], 13)
    second = T.nested_group_indices(rows, [7, 4], 13)
    assert np.array_equal(first[4], second[4]) and np.array_equal(first[7], second[7])
    assert set(first[4]) <= set(first[7])
    full = Counter(row["group_id"] for row in rows)
    sampled = Counter(rows[index]["group_id"] for index in first[4])
    for group_id, count in sampled.items():
        assert count == full[group_id], "组必须整体进出，不允许拆组"
    with pytest.raises(ValueError):
        T.nested_group_indices(rows, [11], 13)


def test_training_requires_seven_classes_fail_loud():
    with pytest.raises(ValueError, match="单一类别"):
        T.fit_model("majority", None, np.zeros(3, dtype=int), 13)
    with pytest.raises(ValueError, match="未覆盖全部七类"):
        T.fit_model("char-logreg", ["甲方案", "乙方案", "甲方案"],
                    np.asarray([0, 1, 0]), 13)
    scorer, _ = _fit_tiny_char_scorer()
    assert scorer.classes == F.CLASSES
    probs = scorer.predict_proba([_build_request("先校正，再做汇总。", "汇总时", "先校正后汇总。", 0)])
    assert probs.shape == (1, len(F.CLASSES))
    assert probs.sum() == pytest.approx(1.0)
                                            
    legacy = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)),
                       ("clf", LogisticRegression(max_iter=500))])
    legacy.fit(["甲方案：先校正后汇总。", "乙方案：先汇总后校正。", "甲方案：先校正。"],
               np.asarray([0, 1, 0]))
    legacy_scorer = F.RelationScorer(kind="char-logreg", model=legacy, class_indices=[0, 1])
    legacy_probs = legacy_scorer.predict_proba([_build_request("先校正。", "汇总时", "先校正后汇总。", 0)])
    for index in range(len(F.CLASSES)):
        if index not in (0, 1):
            assert legacy_probs[0, index] == 0.0, "训练缺席的类别必须是精确 0"


def test_feature_view_ignores_metadata_tier_and_context():
    base = _build_request("先做校正，再做汇总。", "汇总时", "做校正，然后汇总。", 0)
    dirty = copy.deepcopy(base)
    dirty["wrapped_label"] = "conflict"
    dirty["teacher_vote"] = "supersede"
    dirty["candidate"].update({"id": "cand-secret", "scope": "global", "project_id": "project-x",
                               "permanence": "permanent", "tier": "implicit"})
    dirty["existing"][0].update({"id": "rule-secret", "scope": "inherited_global",
                                 "replaceable": True, "tier": "implicit"})
    dirty["evidence"][0].update({"id": "ev-secret", "principal": "user-x"})
    dirty["context"] = {"scope": "global", "replaceable_ids": ["rule-secret"],
                        "inherited_global_ids": []}
    assert F.request_text(dirty) == F.request_text(base), "tier/permanence/元数据不得进入特征"
    assert np.array_equal(F.structural_features(dirty), F.structural_features(base))
    assert F.structural_features(base).shape == (10,)
    text = F.request_text(dirty)
    for marker in ("cand-secret", "rule-secret", "ev-secret", "project-x", "global", "conflict"):
        assert marker not in text
    changed = copy.deepcopy(base)
    changed["candidate"]["text"] += "补充一句。"
    assert F.request_text(changed) != F.request_text(base), "文本变化必须改变特征（正对照）"
    scorer, _ = _fit_tiny_char_scorer()
    assert np.array_equal(scorer.predict_proba([base]), scorer.predict_proba([dirty]))


def test_stored_request_matches_real_m_projection():
    stored, projected = _projection_pair()
    assert F.request_text(stored) == F.request_text(projected)
    assert np.array_equal(F.structural_features(stored), F.structural_features(projected))
    text = F.request_text(projected)
    for marker in ("user-a", "project-a", "r1", "e1"):
        assert marker not in text, "投影元数据不得暗复制进特征"
    scorer, _ = _fit_tiny_char_scorer()
    assert np.array_equal(scorer.predict_proba([stored]), scorer.predict_proba([projected]))


def test_empty_existing_sentinel_is_supported():
    request = {"candidate": {"text": "先做校正。", "trigger": ""}, "existing": [], "evidence": []}
    assert F.structural_features(request).shape == (10,)
    assert F.request_text(request) == "先做校正。"
    scorer, _ = _fit_tiny_char_scorer()
    probs = scorer.predict_proba([request])
    assert probs.shape == (1, len(F.CLASSES)) and probs.sum() == pytest.approx(1.0)


def test_pair_verdict_is_the_single_gate_with_tie_and_trigger_rules():
    request = {"candidate": {"text": "x", "trigger": "汇总时"}, "existing": [], "evidence": []}
    no_trigger = {"candidate": {"text": "x", "trigger": ""}, "existing": [], "evidence": []}
    strong = _probability_row(duplicate=0.9, conflict=0.05)
    assert EM.pair_verdict(strong, request, 0.5) == "duplicate"
    assert EM.pair_verdict(strong, request, 0.95) == "uncertain", "低于 τ 必须 low-conf uncertain"
    tie = _probability_row(duplicate=0.5, conflict=0.5)
    assert EM.pair_verdict(tie, request, 0.0) == "uncertain", "跨侧平局不得静默 allow"
    same_side = _probability_row(duplicate=0.4, complement=0.4)
    assert EM.pair_verdict(same_side, request, 0.3) == "complement", "同侧平局按 spec 类序"
    contextual = _probability_row(contextual=0.9)
    assert EM.pair_verdict(contextual, request, 0.0) == "contextual"
    assert EM.pair_verdict(contextual, no_trigger, 0.0) == "uncertain", "无 trigger 的 contextual 不操作"
    with pytest.raises(ValueError):
        EM.pair_verdict(np.zeros(3), request, 0.0)


def _denominator_case():
    probs = np.stack([
        _probability_row(duplicate=0.9),
        _probability_row(conflict=0.8),
        _probability_row(duplicate=0.8),
        _probability_row(contextual=0.7),
        _probability_row(conflict=0.6),
    ])
    y = np.asarray([F.CLASSES.index(name) for name in
                    ("conflict", "duplicate", "duplicate", "contextual", "conflict")])
    trigger = np.asarray([True, True, True, True, True])
    return probs, y, trigger


def test_operating_stats_denominators_and_zero_denominator_nulls():
    probs, y, trigger = _denominator_case()
    stats = EM.operating_stats(probs, y, trigger, 0.0)
    assert stats["coverage"] == pytest.approx(0.6)
    assert stats["allow_count"] == 3
    assert stats["conditional_risk"] == pytest.approx(1 / 3), "1 危险放行 / 3 放行决策"
    assert stats["dangerous_allow_rate"] == pytest.approx(1 / 2), "1 危险放行 / 2 真实阻塞"
    assert stats["dangerous_allow_share"] == pytest.approx(1 / 5)
    assert stats["false_block_rate"] == pytest.approx(1 / 3), "1 误阻塞 / 3 真实允许"
    assert stats["false_block_share"] == pytest.approx(1 / 5)
    blocked = np.stack([_probability_row(conflict=0.9), _probability_row(conflict=0.9)])
    y_allowing = np.asarray([F.CLASSES.index("duplicate"), F.CLASSES.index("complement")])
    nulls = EM.operating_stats(blocked, y_allowing, np.asarray([True, True]), 0.0)
    assert nulls["conditional_risk"] is None and nulls["dangerous_allow_rate"] is None
    assert nulls["false_block_rate"] == pytest.approx(1.0)
    allowed = np.stack([_probability_row(duplicate=0.9), _probability_row(duplicate=0.9)])
    y_blocking = np.asarray([F.CLASSES.index("conflict"), F.CLASSES.index("supersede")])
    danger = EM.operating_stats(allowed, y_blocking, np.asarray([True, True]), 0.0)
    assert danger["conditional_risk"] == pytest.approx(1.0)
    assert danger["dangerous_allow_rate"] == pytest.approx(1.0)
    assert danger["false_block_rate"] is None


def test_risk_coverage_and_select_threshold_use_conditional_risk():
    probs, y, trigger = _denominator_case()
    curve = EM.risk_coverage(probs, y, trigger)
    coverages = [entry["coverage"] for entry in curve]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(coverages, coverages[1:]))
    assert EM.select_threshold(probs, y, trigger, target=0.3, min_coverage=0.5) == (None, None)
    threshold, selected = EM.select_threshold(probs, y, trigger, target=0.5, min_coverage=0.5)
    assert threshold == 0.0
    assert selected is not None and selected["conditional_risk"] == pytest.approx(1 / 3)


def test_evaluate_threshold_none_keeps_no_operation_and_tau0_is_diagnostic():
    probs, y, trigger = _denominator_case()
    groups = [f"g{index}" for index in range(len(y))]
    result = EM.evaluate(probs, y, has_trigger=trigger, group_ids=groups, threshold=None,
                         bootstrap_samples=20, seed=1)
    assert result["operating"] == {"threshold": None, "selected": False}
    assert result["diagnostic_tau0"]["threshold"] == 0.0
    assert result["diagnostic_tau0"]["scope"] == "uncalibrated_tau0_diagnostic"
    assert result["bootstrap"]["scope"] == "uncalibrated_tau0_diagnostic"
    assert "macro_f1" in result and "ece" in result
    selected = EM.evaluate(probs, y, has_trigger=trigger, group_ids=groups, threshold=0.0,
                           bootstrap_samples=20, seed=1)
    assert selected["operating"]["selected"] is True
    assert selected["operating"]["threshold"] == 0.0


def test_cluster_bootstrap_reports_denominators_invalids_and_rare_bound():
    probs, y, trigger = _denominator_case()
    groups = [f"g{index}" for index in range(len(y))]
    record = EM.bootstrap_ci(probs, y, trigger, groups, 0.0, samples=50, seed=7)
    assert record["macro_f1"]["ci"] is not None and record["macro_f1"]["denominator"] == 5
    assert record["conditional_risk"]["denominator"] == 3
    assert record["conditional_risk"]["errors"] == 1
    assert record["dangerous_allow_rate"]["denominator"] == 2
    assert record["false_block_rate"]["denominator"] == 3
    allowed = np.stack([_probability_row(duplicate=0.9), _probability_row(duplicate=0.8),
                        _probability_row(duplicate=0.7)])
    y_allowed = np.asarray([F.CLASSES.index("duplicate")] * 3)
    zero = EM.bootstrap_ci(allowed, y_allowed, np.asarray([True] * 3),
                           ["a", "b", "c"], 0.0, samples=50, seed=7)
    risk = zero["conditional_risk"]
    assert risk["errors"] == 0 and risk["zero_errors"] is True
    assert risk["rare_upper_bound"] is not None and 0 < risk["rare_upper_bound"] <= 1, \
        "零错误不得只给 [0,0] 当证据"
    blocked = EM.bootstrap_ci(np.stack([_probability_row(conflict=0.9)] * 3),
                              y_allowed, np.asarray([True] * 3), ["a", "b", "c"], 0.0,
                              samples=50, seed=7)
    assert blocked["conditional_risk"]["invalid"] == 50
    assert blocked["conditional_risk"]["ci"] is None, "零分母复制不当 0 风险"


def test_temperature_scaling_reduces_nll_and_calibration_metrics_bounds():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=1.5, size=(80, len(F.CLASSES)))
    labels = logits.argmax(axis=1).copy()
    flipped = rng.random(len(labels)) < 0.25
    labels[flipped] = (labels[flipped] + 1) % len(F.CLASSES)
    temperature = EM.fit_temperature(logits, labels)
    assert temperature > 0

    def nll(value: float) -> float:
        scaled = logits / value
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(labels)), labels].mean())

    assert nll(temperature) <= nll(1.0) + 1e-9
    probs = F.softmax(logits / temperature)
    assert 0.0 <= EM.ece(probs, labels) <= 1.0
    assert EM.brier(probs, labels) > 0.0
    onehot = np.zeros((4, len(F.CLASSES)))
    onehot[np.arange(4), [0, 1, 2, 3]] = 1.0
    assert EM.brier(onehot, np.asarray([0, 1, 2, 3])) == 0.0
    with pytest.raises(ValueError):
        EM.fit_temperature(logits, np.zeros(len(labels), dtype=int))


def test_load_rows_fails_loud_on_bad_label_and_missing_group(tmp_path):
    bad_label = tmp_path / "bad_label.jsonl"
    bad_label.write_text(
        json.dumps({"split": "train", "label": "activate", "request": {}}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        T.load_rows(bad_label, source="data")
    row = _row("t0", "g0", "train", "duplicate", 0)
    del row["group_id"]
    missing_group = tmp_path / "missing_group.jsonl"
    _write_jsonl(missing_group, [row])
    with pytest.raises(ValueError, match="group_id"):
        T.load_rows(missing_group, source="data")


def test_pair_filter_excludes_set_and_multi_rule_with_counts():
    rows = [_row("t0", "g0", "train", "duplicate", 0),
            _multi_rule_row("t1", "g1", "train"),
            _set_row("s0", "g-set")]
    kept, excluded = T._pair_filter(rows)
    assert len(kept) == 1 and kept[0]["id"] == "t0"
    assert excluded["multi_rule"] == 1 and excluded["set_split"] == 1


def test_model_fingerprint_tracks_content_not_stat(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"arch": "test"}', encoding="utf-8")
    weights = model / "pytorch_model.bin"
    weights.write_bytes(b"\x00" * 64)
    first = F.model_fingerprint(model)
    assert F.model_fingerprint(model) == first
    stat = weights.stat()
    weights.write_bytes(b"\x01" * 64)
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert weights.stat().st_size == stat.st_size
    assert F.model_fingerprint(model) != first, "同大小同 mtime 的内容修改必须换指纹"


def test_char_scorer_persistence_reload_parity(tmp_path):
    assert T.load_scorer is F.load_scorer, "train 只能重导出唯一实现"
    assert T.RelationScorer is F.RelationScorer
    scorer, rows = _fit_tiny_char_scorer()
    directory = tmp_path / "char-scorer"
    T.save_scorer(scorer, directory, threads=6, batch_size=16)
    reloaded = T.load_scorer(directory)
    probe = [row["request"] for row in rows[:3]]
    assert np.array_equal(scorer.predict_proba(probe), reloaded.predict_proba(probe))
    verdict = EM.pair_verdict(reloaded.predict_proba(probe[:1])[0], probe[0], 0.0)
    assert verdict in F.CLASSES
    parity = T.check_reload_parity(scorer, directory, probe, device="cpu")
    assert parity["ok"] is True and parity["max_abs_diff"] == 0.0
    bad = tmp_path / "bad-scorer"
    bad.mkdir()
    joblib.dump({"schema": "not-ours"}, bad / "model.joblib")
    with pytest.raises(ValueError):
        F.load_scorer(bad)


def test_cli_end_to_end_majority_and_char(tmp_path):
    data_rows, challenge_rows = _fixture()
    data_path, challenge_path = tmp_path / "data.jsonl", tmp_path / "challenge.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(challenge_path, challenge_rows)
    out = tmp_path / "out"
    code = T.main(["--data", str(data_path), "--challenge", str(challenge_path),
                   "--out", str(out), "--models", "majority,char-logreg",
                   "--sizes", "21", "--seeds", "13"])
    assert code == 0
    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert document["schema"] == T.TRAINING_SCHEMA
    assert [run["model"] for run in document["runs"]] == ["majority", "char-logreg"]
    data_section = document["data"]
    assert data_section["excluded_rows"]["train"]["multi_rule"] == 1
    assert data_section["excluded_rows"]["challenge"]["multi_rule"] == 1
    assert data_section["excluded_rows"]["set"]["set_split"] == 1
    assert data_section["pair_rows"]["challenge"] == 14
    assert all(count == 0 for count in data_section["group_overlaps"].values())
    expected_eval_size = {"challenge": 14, "heldout": 7}
    for run in document["runs"]:
        assert run["parity"]["ok"] is True
        assert run["n_train"] == 21
        assert set(run["eval"]) == set(expected_eval_size)
        for name, result in run["eval"].items():
            assert result["n"] == expected_eval_size[name]
            assert set(result["per_class"]) == set(F.CLASSES)
            assert len(result["confusion_matrix"]["matrix"]) == len(F.CLASSES)
            assert len(result["risk_coverage"]) == 20
            operating = result["operating"]
            if run["operating_threshold"] is None:
                assert operating == {"threshold": None, "selected": False}
            else:
                assert operating["selected"] is True
                assert operating["threshold"] == run["operating_threshold"]
            assert result["diagnostic_tau0"]["scope"] == "uncalibrated_tau0_diagnostic"
            bootstrap = result["bootstrap"]
            assert {"samples", "confidence", "seed", "threshold", "macro_f1", "coverage",
                    "conditional_risk", "dangerous_allow_rate", "false_block_rate"} <= set(bootstrap)
            assert bootstrap["macro_f1"]["ci"] is not None
            for key in ("conditional_risk", "dangerous_allow_rate", "false_block_rate"):
                assert {"invalid", "errors", "denominator", "zero_errors", "ci", "valid"} <= \
                    set(bootstrap[key])
    predictions = [json.loads(line) for line in
                   (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    assert len(predictions) == 2 * (14 + 7)
    for row in predictions:
        assert len(row["probs"]) == len(F.CLASSES)
        assert abs(sum(row["probs"]) - 1.0) < 1e-3
        assert row["verdict"] is None or row["verdict"] in F.CLASSES
        assert row["verdict_tau0"] in F.CLASSES
        if row["verdict"] is None:
            assert row["allow"] is None
        else:
            assert row["allow"] == (row["verdict"] in EM.ALLOWING_CLASSES)
        assert row["allow_tau0"] == (row["verdict_tau0"] in EM.ALLOWING_CLASSES)
    curves = [json.loads(line) for line in
              (out / "curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert curves and all({"model", "size", "seed", "eval_set", "threshold", "coverage",
                           "conditional_risk", "dangerous_allow_rate", "false_block_rate"}
                          <= set(point) for point in curves)


def test_group_leakage_including_challenge_is_fatal_and_no_escape_flag(tmp_path):
    data_rows, challenge_rows = _fixture()
    challenge_rows[0]["group_id"] = "train-g-complement"
    data_path, challenge_path = tmp_path / "data.jsonl", tmp_path / "challenge.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(challenge_path, challenge_rows)
    with pytest.raises(ValueError, match="泄漏"):
        T.main(["--data", str(data_path), "--challenge", str(challenge_path),
                "--out", str(tmp_path / "out"), "--models", "majority",
                "--sizes", "21", "--seeds", "13"])
    parser = T.build_parser()
    assert not any("--allow-group-overlap" in action.option_strings
                   for action in parser._actions), "组泄漏门不许有覆盖开关"


def test_minilm_cli_small_end_to_end(tmp_path):
    embedding = _embedding_model_path()
    if embedding is None:
        pytest.skip("本地 MiniLM 权重不可见（禁止下载）；设 MEMORY_RELATIONS_EMBEDDING_MODEL 后重跑")
    data_rows, challenge_rows = _fixture()
    data_path, challenge_path = tmp_path / "data.jsonl", tmp_path / "challenge.jsonl"
    _write_jsonl(data_path, data_rows)
    _write_jsonl(challenge_path, challenge_rows)
    out = tmp_path / "out"
    code = T.main(["--data", str(data_path), "--challenge", str(challenge_path),
                   "--out", str(out), "--models", "minilm-logreg,minilm-mlp", "--sizes", "21",
                   "--seeds", "13", "--embedding-model", str(embedding),
                   "--cache-dir", str(tmp_path / "cache"), "--bootstrap-samples", "25"])
    assert code == 0
    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [run["model"] for run in document["runs"]] == ["minilm-logreg", "minilm-mlp"]
    assert all(run["parity"]["ok"] is True for run in document["runs"])
    assert (tmp_path / "cache" / "embeddings.npz").is_file()
    probe = [row["request"] for row in challenge_rows[:3]]
    for kind in ("minilm-logreg", "minilm-mlp"):
        scorer = F.load_scorer(out / "scorers" / f"{kind}-size21-seed13")
        probs = scorer.predict_proba(probe)
        assert probs.shape == (3, len(F.CLASSES))
        assert np.allclose(probs.sum(axis=1), 1.0)
    stored, projected = _projection_pair()
    embedder = F.Embedder(embedding, cache_dir=tmp_path / "parity-cache")
    assert np.array_equal(F.minilm_feature_matrix(embedder, [stored]),
                          F.minilm_feature_matrix(embedder, [projected]))
    assert F.request_text(stored) == F.request_text(projected)


def test_embedder_cache_keys_on_fingerprint_and_text(tmp_path):
    embedding = _embedding_model_path()
    if embedding is None:
        pytest.skip("本地 MiniLM 权重不可见（禁止下载）")
    embedder = F.Embedder(embedding, cache_dir=tmp_path / "cache")
    texts = ["缓存键测试甲。", "缓存键测试乙。"]
    first = embedder.encode(texts)
    assert first.shape == (2, embedder.dim)
    embedder.save_cache()
    reloaded = F.Embedder(embedding, cache_dir=tmp_path / "cache")
    second = reloaded.encode(texts)
    assert np.array_equal(first, second)
    assert len(reloaded._memory) == 2
    assert all(len(key) == 64 for key in reloaded._memory)


def _development_rows() -> list[dict]:
    rows = [row for row in _fixture()[0] if row["split"] in ("train", "calibration")]
    for index, label in enumerate(F.CLASSES):
        row = _row(f"dev-{label}", f"dev-g-{label}", "dev", label, index)
        row["request"]["candidate"]["text"] += "仅开发集。"
        row["phenomena"] = {"root_id": f"dev-root-{label}", "domain": "writing",
                            "family": "validation", "author": "synthetic-author",
                            "label_rationale": "合成fixture，不是真人金标"}
        rows.append(row)
    return rows


def test_development_cli_never_opens_challenge_and_probes_only_dev(tmp_path, monkeypatch):
    rows = _development_rows()
    path, out = tmp_path / "dev-data.jsonl", tmp_path / "out"
    _write_jsonl(path, rows)
    original_load, original_parity = T.load_rows, T.check_reload_parity
    probes = []

    def load(*args, **kwargs):
        assert kwargs["source"] != "challenge"
        return original_load(*args, **kwargs)

    def parity(scorer, directory, requests, **kwargs):
        assert all("仅开发集" in request["candidate"]["text"] for request in requests)
        probes.extend(requests)
        return original_parity(scorer, directory, requests, **kwargs)

    monkeypatch.setattr(T, "load_rows", load)
    monkeypatch.setattr(T, "check_reload_parity", parity)
    assert T.main(["--data", str(path), "--challenge", str(tmp_path / "sealed-do-not-open.jsonl"),
                   "--out", str(out), "--models", "majority,char-logreg", "--sizes", "21",
                   "--seeds", "13", "--eval-splits", "none", "--bootstrap-samples", "5"]) == 0
    document = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert document["args"]["eval_splits"] == [] and document["data"]["challenge"] is None
    assert all(run["eval"] == {} and run["dev"]["n"] == 7 for run in document["runs"])
    assert (out / "predictions.jsonl").read_text() == ""
    assert probes


def test_development_mode_refuses_mixed_test_file_before_fit(tmp_path, monkeypatch):
    rows = _development_rows() + [{"split": "heldout", "request": "never-project"}]
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(path, rows)
    monkeypatch.setattr(T, "fit_model", lambda *args: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="独立开发文件"):
        T.main(["--data", str(path), "--out", str(tmp_path / "out"), "--models", "majority",
                "--sizes", "21", "--seeds", "13", "--eval-splits", "none"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("sizes,seeds", [("0", "13"), ("22", "13"), ("3,21", "13"),
                                         ("21", "-1"), ("21", "4294967296"),
                                         ("21,21", "13"), ("21", "13,13")])
def test_all_invalid_grids_fail_before_fitting_or_outputs(tmp_path, monkeypatch, sizes, seeds):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, _development_rows())
    monkeypatch.setattr(T, "fit_model", lambda *args: pytest.fail("must not fit"))
    with pytest.raises(ValueError):
        T.main(["--data", str(path), "--out", str(tmp_path / "out"), "--models", "majority",
                "--sizes", sizes, "--seeds", seeds, "--eval-splits", "none"])
    assert not (tmp_path / "out").exists()


def test_preflight_checks_later_seed_and_existing_output(tmp_path, monkeypatch):
    rows = _train_pair_rows()

    def samples(rows, sizes, seed):
        return {21: np.arange(21) if seed == 13 else np.arange(3)}

    monkeypatch.setattr(T, "nested_group_indices", samples)
    with pytest.raises(ValueError, match="seed=42.*未覆盖全部七类"):
        T.preflight_training(rows, [21], [13, 42], tmp_path / "out")
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "history.txt"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        T.preflight_training(rows, [21], [13], out)
    assert sentinel.read_text() == "untouched"


@pytest.mark.parametrize("split", ["train", "dev", "calibration", "heldout", "ood", None])
def test_every_consumed_split_requires_group_not_id_fallback(tmp_path, split):
    row = _row("unique-id", "g", split, "duplicate", 0)
    del row["group_id"]
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match="group_id"):
        T.load_rows(path, source="challenge" if split is None else "data")
    with pytest.raises(ValueError, match="group_id"):
        T._group_ids([row])


def test_root_leakage_does_not_hide_behind_different_group_ids(tmp_path):
    rows = _development_rows()
    rows[0]["phenomena"] = {"root_id": "same-root"}
    rows[-1]["phenomena"]["root_id"] = "same-root"
    path = tmp_path / "leak.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="root_id.*泄漏"):
        T.main(["--data", str(path), "--out", str(tmp_path / "out"), "--models", "majority",
                "--sizes", "21", "--seeds", "13", "--eval-splits", "none"])


def _partitioned_rows() -> list[dict]:
    rows = []
    for index, role in enumerate(("prob", "policy", "audit")):
        for suffix in range(2):
            row = _row(f"{role}-{suffix}", f"{role}-g", "calibration", F.CLASSES[suffix], index)
            row["phenomena"] = {"calibration_role": role, "operating_primary": suffix == 0}
            rows.append(row)
    return rows


def test_partitioned_calibration_consumer_passes_roles_and_rejected_audit_without_reselecting(monkeypatch):
    rows = _partitioned_rows()
    logits = np.zeros((6, 7))
    captured = []

    def shared(*args):
        captured.append(args)
        return {"temperature": 1.7, "threshold": None, "threshold_feasible": False,
                "selection": {"threshold": 0.75}, "audit": {"sufficient": False}}

    monkeypatch.setattr(EM, "calibrate_partitioned", shared, raising=False)
    monkeypatch.setattr(EM, "fit_temperature", lambda *args: pytest.fail("legacy temperature forbidden"))
    monkeypatch.setattr(EM, "select_threshold", lambda *args: pytest.fail("must not reselect"))
    result = T.calibrate(logits, rows, protocol="partitioned")
    assert result["threshold"] is None and result["threshold_feasible"] is False
    assert result["selection"]["threshold"] == 0.75 and result["audit"]["sufficient"] is False
    z, labels, trigger, groups, roles, primary = captured[0]
    assert z is logits and np.array_equal(labels, T._labels(rows))
    assert np.array_equal(trigger, T._triggers(rows)) and groups == T._group_ids(rows)
    assert roles == ["prob", "prob", "policy", "policy", "audit", "audit"]
    assert primary.tolist() == [True, False, True, False, True, False]


@pytest.mark.parametrize("failure", ["string-primary", "double-primary", "missing-primary", "mixed-role"])
def test_partition_metadata_fail_closed(failure):
    rows = _partitioned_rows()
    if failure == "string-primary":
        rows[0]["phenomena"]["operating_primary"] = "true"
    elif failure == "double-primary":
        rows[1]["phenomena"]["operating_primary"] = True
    elif failure == "missing-primary":
        rows[0]["phenomena"]["operating_primary"] = False
    else:
        rows[1]["phenomena"]["calibration_role"] = "audit"
    with pytest.raises(ValueError):
        T._calibration_roles(rows)


def test_missing_partitioned_api_fails_before_fit(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, _development_rows())
    monkeypatch.delattr(EM, "calibrate_partitioned", raising=False)
    monkeypatch.setattr(T, "fit_model", lambda *args: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="尚未集成"):
        T.main(["--data", str(path), "--out", str(tmp_path / "out"), "--models", "majority",
                "--sizes", "21", "--seeds", "13", "--eval-splits", "none", "--calibration-protocol", "partitioned"])
    assert not (tmp_path / "out").exists()


def test_tuning_mode_requires_dev_even_with_legacy_final_estimators(tmp_path, monkeypatch):
    path, out = tmp_path / "no-dev.jsonl", tmp_path / "out"
    _write_jsonl(path, [row for row in _development_rows() if row["split"] != "dev"])
    monkeypatch.setattr(T, "fit_model", lambda *args: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="独立dev.*parity"):
        T.main(["--data", str(path), "--out", str(out), "--models", "majority",
                "--sizes", "21", "--seeds", "13", "--eval-splits", "none"])
    assert not out.exists()
