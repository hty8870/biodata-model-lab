"""pair→set 适配器的离线专项测试：替身 scorer 经真实 M 道闸，不代表语义准确率。

覆盖 spec.json 冻结接口与主线评审口径：make_predictor 经 metrics.pair_verdict 单真源
（七类映射、malformed 概率、AI 关闭/trial/mock 边界、引用只取真实输入、implicit/inherited
supersede 交闸判、0/1/8/64/65/129 规则全量逐对无 top-k、跨批 uncertain 覆 conflict/引用并集
污染/批间混杂不检测的如实记录、无隐式网络与确定性）；evaluate_sets 的 all-attempt 完整
分母、语义弃权/系统失败、finally 尝试延迟、classifier/pair/set raw-gated 分离、角色引用
评分、组等权与预选 primary、真实 adapter oracle 聚合探针；保留 secondary valid-only、
0 分母 null 与 CLI meta threshold_feasible 严格 true / --diagnostic 纪律。
临时 SQLite KnowledgeService.assess(predictor=) 实际回放。
历史可选探针保留环境 skip；test_integration_v3 直接加载必需模块，缺模块使集成门失败，不以替身掩盖。
"""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from dataset_recommender.agent import memory_relations as M
from dataset_recommender.app.conversation_store import ConversationStore
from dataset_recommender.app.knowledge_service import KnowledgeService
from dataset_recommender.app.workspace_store import WorkspaceStore
from dataset_recommender.llm.llm_client import LLMConfig

from research.memory_relations import adapter, evaluate
from research.memory_relations import features as FEAT

ROOT = Path(__file__).resolve().parents[2]
for _module in (M, adapter, evaluate):
    assert Path(_module.__file__).resolve().is_relative_to(ROOT), _module.__file__

FIELDS = {"relation", "related_ids", "reason", "evidence_refs", "error_code"}
                                                                            
CLASSES = ("complement", "conflict", "contextual", "duplicate", "supersede", "temporary", "uncertain")
ALLOWING = ("duplicate", "complement", "contextual")
BLOCKING = ("conflict", "supersede", "temporary", "uncertain")


class RowErrorDouble(ValueError):
    """schema.RowError 的测试对应物（真实 schema 就绪时改用真类型）。"""

    __test__ = False

    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _StandinMetrics:
    """预集成替身（仅当 metrics / pair_verdict 未就绪时启用）。

    只提供机械管线所需的最小面：pair_verdict=argmax+阈值门（None=不门控）；类序/允许阻塞
    划分/宏 F1 按 spec 口径。真实操作判定（等分 allow/block 不默放行、无 trigger 的
    contextual 判 uncertain、低置信 max-prob 判 uncertain）以 metrics 团队真实现为准；
    本替身不声称与真实操作语义等价。
    """

    CLASSES = CLASSES
    ALLOWING_CLASSES = ALLOWING
    BLOCKING_CLASSES = BLOCKING
    BOOTSTRAP_SAMPLES = 40
    CONFIDENCE = 0.95

    @staticmethod
    def pair_verdict(probs, request, threshold):
        row = np.asarray(probs, dtype=float)
        best = int(np.argmax(row))
        if threshold is not None and float(row[best]) < float(threshold):
            return "uncertain"
        return CLASSES[best]

    @staticmethod
    def macro_f1(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        size = len(CLASSES)
        table = np.bincount(y_true * size + y_pred, minlength=size * size).reshape(size, size)
        true_positive = np.diag(table).astype(float)
        denominator = 2 * true_positive + (table.sum(axis=0) - true_positive) \
            + (table.sum(axis=1) - true_positive)
        f1 = np.divide(2 * true_positive, denominator, out=np.zeros(size), where=denominator > 0)
        return float(f1.mean())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network or LLM fallback in offline adapter test")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(urllib.request, "urlopen", deny)
    monkeypatch.setattr(M, "call_llm", deny)
    monkeypatch.setattr("dataset_recommender.app.knowledge_service.call_llm", deny)
    for name in ("BIODATA_RUNS_DB", "BIODATA_WORKSPACE_DB", "BIODATA_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def module_bridge(monkeypatch):
    """其它角色模块未合入时的最小替身桥；真模块/真 pair_verdict 就绪后自动退场。"""
    real = None
    try:
        from research.memory_relations import metrics as real                 
    except ImportError:
        real = None
    if real is None or not callable(getattr(real, "pair_verdict", None)):
        monkeypatch.setattr(adapter, "_metrics_module", lambda: _StandinMetrics)
    if real is None:
        monkeypatch.setattr(evaluate, "_metrics_module", lambda: _StandinMetrics)
    try:
        from research.memory_relations import schema as _schema                       
    except ImportError:
        monkeypatch.setattr(evaluate, "_row_error_type", lambda: RowErrorDouble)


@pytest.fixture
def inputs():
    cfg = LLMConfig(enable_llm=True, api_key="test-memory-relations-key",
                    base_url="https://model.example/v1", model="research-scorer")
    fp = hashlib.sha256(f"{cfg.base_url}|{cfg.model}".encode()).hexdigest()[:12]
    scope = {"principal": "user-a", "endpoint_fp": fp, "project_id": "project-a"}
    quote = "以后汇报先给结论，再给依据。"
    candidate = {**scope, "text": "汇报先给结论，再给依据。", "trigger": "汇报时",
                 "tier": "explicit", "evidence": [quote]}
    existing = [{**scope, "id": "r1", "text": "汇报先说结论，再解释原因。",
                 "trigger": "汇报时", "tier": "explicit", "evidence": []}]
    evidence = [{"id": "e1", "text": quote}]
    return candidate, existing, evidence, cfg


class RecordingScorer:
    """按收到的请求回放的替身 scorer；probability 由 matrix_for(requests) 决定。"""

    def __init__(self, matrix_for, *, classes=CLASSES):
        self.classes = list(classes)
        self.matrix_for = matrix_for
        self.calls = []

    def predict_proba(self, requests):
        self.calls.append([copy.deepcopy(request) for request in requests])
        return self.matrix_for(self.calls[-1])


def one_hot(relation, count):
    matrix = np.zeros((count, len(CLASSES)))
    matrix[:, CLASSES.index(relation)] = 1.0
    return matrix


def dist(relation, confidence, count):
    matrix = np.full((count, len(CLASSES)), (1.0 - confidence) / (len(CLASSES) - 1))
    matrix[:, CLASSES.index(relation)] = confidence
    return matrix


def constant(relation):
    return RecordingScorer(lambda requests: one_hot(relation, len(requests)))


def capped(by_id):
    """按规则 id 给 (relation, confidence)；未列出的规则默认 complement。"""
    def matrix_for(requests):
        matrix = np.zeros((len(requests), len(CLASSES)))
        for index, request in enumerate(requests):
            relation, confidence = by_id.get(request["existing"][0]["id"], ("complement", 1.0))
            matrix[index] = dist(relation, confidence, 1)[0]
        return matrix
    return RecordingScorer(matrix_for)


def by_rule(**overrides):
    def matrix_for(requests):
        matrix = np.zeros((len(requests), len(CLASSES)))
        for index, request in enumerate(requests):
            rules = request["existing"]
            relation = overrides.get(rules[0]["id"], "complement") if rules else "complement"
            matrix[index, CLASSES.index(relation)] = 1.0
        return matrix
    return RecordingScorer(matrix_for)


def run(inputs, predictor, **kwargs):
    candidate, existing, evidence, cfg = inputs
    return M.assess_relations(candidate, existing, evidence, cfg, predictor=predictor, **kwargs)


def rules_for(inputs, count):
    template = inputs[1][0]
    return [{**template, "id": f"r{index}"} for index in range(count)]


@pytest.mark.parametrize("relation", CLASSES)
def test_seven_classes_through_real_gate(inputs, relation):
    scorer = constant(relation)
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert set(out) == FIELDS
    assert out["relation"] == relation
    assert 0 < len(out["reason"]) <= 480
    assert "未经独立核验" in out["reason"]
    assert len(scorer.calls) == 1
    assert [rule["id"] for rule in scorer.calls[0][0]["existing"]] == ["r1"]
    if relation == "uncertain":
        assert out["error_code"] == "model_uncertain"
        assert out["related_ids"] == [] and out["evidence_refs"] == []
    else:
        assert out["error_code"] is None
        assert out["related_ids"] == ["r1"]
        assert out["evidence_refs"] == ["e1"]
        assert "非逐条关系实证" in out["reason"], "宽泛引用必须如实说明"


def test_pair_requests_are_projected_without_scope_material(inputs):
    scorer = constant("complement")
    run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    request = scorer.calls[0][0]
    assert set(request) == {"candidate", "existing", "evidence", "context"}
    assert [rule["id"] for rule in request["existing"]] == ["r1"]
    assert request["candidate"]["scope"] == "project"
    text = json.dumps(scorer.calls[0], ensure_ascii=False)
    assert "principal" not in text and "endpoint_fp" not in text
    assert inputs[0]["principal"] not in text


MALFORMED = [
    pytest.param(lambda n: np.zeros((n + 1, len(CLASSES))), id="wrong-row-count"),
    pytest.param(lambda n: np.zeros((n, len(CLASSES) - 1)), id="wrong-width"),
    pytest.param(lambda n: np.full((n, len(CLASSES)), np.nan), id="nan"),
    pytest.param(lambda n: np.full((n, len(CLASSES)), np.inf), id="inf"),
    pytest.param(lambda n: np.full((n, len(CLASSES)), -0.1), id="negative"),
    pytest.param(lambda n: np.full((n, len(CLASSES)), 1.5), id="out-of-range"),
    pytest.param(lambda n: np.full((n, len(CLASSES)), 0.1), id="rows-not-normalized"),
    pytest.param(lambda n: [[0.0] * len(CLASSES) for _ in range(n - 1)] + [[0.0] * (len(CLASSES) - 1)], id="ragged"),
    pytest.param(lambda n: "not-a-matrix", id="not-numeric"),
    pytest.param(lambda n: None, id="none"),
]


@pytest.mark.parametrize("malformed", MALFORMED)
def test_malformed_probability_matrix_maps_to_model_uncertain(inputs, malformed):
    scorer = RecordingScorer(lambda requests: malformed(len(requests)))
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert out["related_ids"] == [] and out["evidence_refs"] == []
    assert "矩阵" in out["reason"]


@pytest.mark.parametrize("updates,code", [
    ({"enable_llm": False}, "disabled"),
    ({"mock_llm": True}, "mock_not_used"),
    ({"provider": "trial"}, "trial_not_used"),
    ({"provider": "mock"}, "mock_not_used"),
])
def test_config_boundaries_never_call_scorer(inputs, updates, code):
    for name, value in updates.items():
        setattr(inputs[3], name, value)
    scorer = constant("complement")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == code
    assert scorer.calls == []


def test_empty_existing_uses_empty_sentinel_and_scorer_judgement(inputs):
    inputs[1].clear()
    for relation in ("complement", "temporary"):
        scorer = constant(relation)
        out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
        assert out["relation"] == relation and out["related_ids"] == []
        assert out["evidence_refs"] == ["e1"]
        assert len(scorer.calls) == 1 and len(scorer.calls[0]) == 1
        assert scorer.calls[0][0]["existing"] == [], "空哨兵必须带着空 existing 交给判断"
    scorer = constant("uncertain")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert len(scorer.calls) == 1


@pytest.mark.parametrize("relation", ["duplicate", "contextual"])
def test_empty_existing_reference_requiring_class_falls_to_gate(inputs, relation):
    inputs[1].clear()
    out = run(inputs, adapter.make_predictor(constant(relation), threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "missing_related_ids"


def test_low_confidence_pair_maps_to_model_uncertain(inputs):
    scorer = capped({"r1": ("complement", 0.4)})
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.5))
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert out["related_ids"] == [] and out["evidence_refs"] == []
    decided = run(inputs, adapter.make_predictor(scorer, threshold=0.3))
    assert decided["relation"] == "complement" and decided["related_ids"] == ["r1"]


def test_conflict_survives_complement_noise_and_undecided_pairs(inputs):
    inputs[1][:] = rules_for(inputs, 3)
    scorer = by_rule(r1="conflict")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "conflict" and out["related_ids"] == ["r1"]
    assert out["error_code"] is None and out["evidence_refs"] == ["e1"]
    scorer = capped({"r0": ("complement", 0.5), "r1": ("conflict", 0.9), "r2": ("complement", 0.5)})
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.8))
    assert out["relation"] == "conflict" and out["related_ids"] == ["r1"]


def test_mixed_condition_time_replacement_is_ambiguous(inputs):
    inputs[1][:] = rules_for(inputs, 3)
    out = run(inputs, adapter.make_predictor(by_rule(r0="contextual", r1="temporary"), threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "ambiguous_relation"
    assert out["related_ids"] == [] and out["evidence_refs"] == []
    out = run(inputs, adapter.make_predictor(by_rule(r0="contextual"), threshold=0.0))
    assert out["relation"] == "contextual" and out["related_ids"] == ["r0"]


def test_undecided_pair_blocks_allow_verdict(inputs):
    inputs[1][:] = rules_for(inputs, 2)
    scorer = capped({"r0": ("complement", 0.9), "r1": ("complement", 0.3)})
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.5))
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert out["related_ids"] == [] and out["evidence_refs"] == []
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.2))
    assert out["relation"] == "complement" and out["related_ids"] == ["r0", "r1"]


def test_implicit_candidate_supersede_left_to_gate(inputs):
    inputs[0]["tier"] = "implicit"
    scorer = constant("supersede")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "replacement_requires_explicit"
    assert len(scorer.calls) == 1, "适配器不预判，仍逐对评分后交 M._decision"


def test_inherited_global_supersede_left_to_gate(inputs):
    inputs[1][0]["project_id"] = ""
    out = run(inputs, adapter.make_predictor(constant("supersede"), threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "scope_replacement_forbidden"


@pytest.mark.parametrize("count,batches", [(1, [1]), (8, [8]), (64, [64]), (65, [64, 1]), (129, [64, 64, 1])])
def test_full_pairwise_scoring_without_top_k(inputs, count, batches):
    inputs[1][:] = rules_for(inputs, count)
    scorer = constant("complement")
    predictor = adapter.make_predictor(scorer, threshold=0.0)
    out = run(inputs, predictor)
    assert out["relation"] == "complement" and out["error_code"] is None
    assert [len(call) for call in scorer.calls] == batches
    assert all(len(request["existing"]) <= 1 for call in scorer.calls for request in call)
    scored = {request["existing"][0]["id"] for call in scorer.calls for request in call}
    assert scored == {f"r{index}" for index in range(count)}
    assert predictor.stats["pairs"] == count
    assert out["related_ids"] == [f"r{index}" for index in range(count)]


def test_long_rule_text_still_scores_every_pair(inputs):
    inputs[1][:] = [{**inputs[1][0], "id": f"r{index}", "text": "句" * 1900} for index in range(20)]
    scorer = constant("complement")
    predictor = adapter.make_predictor(scorer, threshold=0.0)
    out = run(inputs, predictor)
    assert out["relation"] == "complement"
    assert len(scorer.calls) >= 2
    scored = {request["existing"][0]["id"] for call in scorer.calls for request in call}
    assert scored == {f"r{index}" for index in range(20)}
    assert predictor.stats["pairs"] == 20


def test_cross_batch_uncertain_overrides_conflict(inputs):
    """跨批规则如实记录：任一批次 uncertain 覆盖另一批次的 conflict（与批内 conflict 优先不等价）。"""
    inputs[1][:] = rules_for(inputs, 65)
    scorer = capped({**{f"r{index}": ("complement", 0.5) for index in range(64)},
                     "r64": ("conflict", 0.9)})
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.8))
    assert [len(call) for call in scorer.calls] == [64, 1]
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert out["related_ids"] == [], "跨批 uncertain 时不得保留 conflict 判定"


def test_cross_batch_union_pollutes_related_as_reported(inputs):
    """跨批并集如实记录：related_ids 含兼容批次的引用（不擅自剪裁、不改 M）。"""
    inputs[1][:] = rules_for(inputs, 65)
    scorer = by_rule(r64="conflict")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert out["relation"] == "conflict"
    assert out["related_ids"] == [f"r{index}" for index in range(65)], "跨批并集是生产合并的如实结果"


def test_cross_batch_mixing_is_not_detected_within_batch(inputs):
    """批内混杂诚实检查不跨批生效：批间 contextual+temporary 按生产严重度序合并。"""
    inputs[1][:] = rules_for(inputs, 65)
    scorer = by_rule(r0="contextual", r1="contextual", r2="contextual",
                     r64="temporary")
    out = run(inputs, adapter.make_predictor(scorer, threshold=0.0))
    assert [len(call) for call in scorer.calls] == [64, 1]
    assert out["relation"] == "contextual" and out["error_code"] is None


def test_references_are_limited_to_real_input_ids(inputs):
    inputs[1][:] = rules_for(inputs, 2)
    scorer = by_rule(r1="conflict")
    predictor = adapter.make_predictor(scorer, threshold=0.0)
    out = run(inputs, predictor)
    assert out["related_ids"] == ["r1"]
    assert set(out["evidence_refs"]) <= {"e1"}
    request = scorer.calls[0][0]
    direct = predictor(copy.deepcopy(request))
    assert set(direct) == FIELDS
    assert set(direct["related_ids"]) <= {"r0", "r1"}
    assert set(direct["evidence_refs"]) <= {"e1"}


def test_make_predictor_validates_scorer_threshold_and_class_order(inputs):
    class NoProba:
        classes = list(CLASSES)

    with pytest.raises(ValueError):
        adapter.make_predictor(NoProba(), threshold=0.0)
    with pytest.raises(ValueError):
        adapter.make_predictor(object())
    permuted = CLASSES[3:] + CLASSES[:3]
    with pytest.raises(ValueError):
        adapter.make_predictor(RecordingScorer(lambda requests: one_hot("complement", len(requests)),
                                               classes=list(permuted)), threshold=0.0)
    for bad in (1.5, -0.1, True, "0.5", None):
        with pytest.raises(ValueError):
            adapter.make_predictor(constant("complement"), threshold=bad)
    with pytest.raises(ValueError, match="未校准"):
        adapter.make_predictor(constant("complement"))                           


def test_make_predictor_requires_pair_verdict_single_source(monkeypatch):
    class NoVerdict:
        CLASSES = CLASSES

    monkeypatch.setattr(adapter, "_metrics_module", lambda: NoVerdict)
    with pytest.raises(ValueError):
        adapter.make_predictor(constant("complement"), threshold=0.0)


def test_unknown_verdict_class_fails_closed(inputs, monkeypatch):
    class Bogus:
        CLASSES = CLASSES

        @staticmethod
        def pair_verdict(probs, request, threshold):
            return "activate"

    monkeypatch.setattr(adapter, "_metrics_module", lambda: Bogus)
    out = run(inputs, adapter.make_predictor(constant("complement"), threshold=0.0))
    assert out["relation"] == "uncertain" and out["error_code"] == "model_uncertain"
    assert "未知类别" in out["reason"]


def test_offline_deterministic_and_mutation_free(inputs):
    scorer = constant("conflict")
    predictor = adapter.make_predictor(scorer, threshold=0.0)
    assert not hasattr(adapter, "call_llm") and not hasattr(evaluate, "call_llm")
    before = copy.deepcopy(inputs[:3])
    first = run(inputs, predictor)
    request = copy.deepcopy(scorer.calls[0][0])
    snapshot = copy.deepcopy(request)
    assert predictor(copy.deepcopy(request)) == predictor(copy.deepcopy(request)) == first
    assert request == snapshot, "predictor 不改写输入请求"
    assert inputs[:3] == before
    assert run(inputs, predictor) == first
    assert predictor.stats["calls"] == 4 and predictor.stats["pairs"] == 4


                                                                                

def prepare_stub(inputs, rules=None):
    candidate, existing, evidence, cfg = inputs
    selected = existing if rules is None else rules

    def prepare(row, classes):
        label = row.get("label", "complement")
        meta = {
            "label": label,
            "group_id": row.get("group_id", "g1"),
            "label_origin": row.get("label_origin", "agent_review"),
            "acceptable": row.get("acceptable", [label]),
            "expected_refs": list(row.get("expected", [])),
        }
        return candidate, selected, evidence, cfg, meta

    return prepare


def test_evaluate_sets_reports_predictions_calls_and_latency(inputs, monkeypatch):
    rows = [{"id": f"row-{index}", "label": "complement", "group_id": "g1",
             "expected": ["r0", "r1", "e1"]} for index in range(3)]
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs, rules_for(inputs, 2)))
    predictor = adapter.make_predictor(constant("complement"), threshold=0.0)
    report = evaluate.evaluate_sets(predictor, rows)
    assert set(report) == {"rows", "summary", "semantics"}
    dumped = json.dumps(report, ensure_ascii=False)
    assert "非等价" in dumped and "不构成对每条关系的实证" in report["semantics"]
    summary = report["summary"]
    assert summary["row_count"] == 3 and summary["scored_rows"] == 3
    assert summary["valid_rows"] == 3 and summary["gate_error_rows"] == 0
    assert summary["row_error_rows"] == 0
    assert summary["predictor_calls"] == 3 and summary["scored_pairs"] == 6
    assert summary["relations"] == {"complement": 3} and summary["errors"] == {}
    assert summary["agreement"]["primary"] == 1.0
    assert summary["agreement"]["acceptable"] == 1.0
    assert summary["agreement"]["n"] == 3
    assert abs(summary["macro_f1"] - 1.0 / 7) < 1e-12
    assert summary["confusion"]["matrix"][0][0] == 3
    assert summary["operation"]["dangerous_allow_count"] == 0
    assert summary["operation"]["dangerous_allow_rate"] == \
        {"numerator": 0, "denominator": 0, "value": None}, "0 分母必须 null"
    assert summary["operation"]["conditional_risk"] == \
        {"numerator": 0, "denominator": 3, "value": 0.0}
    assert summary["operation"]["false_block_rate"] == \
        {"numerator": 0, "denominator": 3, "value": 0.0}
    assert summary["refs"]["precision"] == 1.0 and summary["refs"]["recall"] == 1.0
    assert summary["refs"]["unknown_total"] == 0 and summary["refs"]["known_non_expected_total"] == 0
    assert summary["refs"]["missed_expected_total"] == 0
    assert summary["bootstrap"]["group_clusters"] == 1
    assert summary["bootstrap"]["macro_f1"] is None, "单簇不给 CI"
    assert [item[1] for item in summary["latency_ms"]["trace"]] == [1, 1, 1]
    first = report["rows"][0]
    assert first["row_id"] == "row-0" and first["rules"] == 2 and first["calls"] == 1
    assert first["scored_pairs"] == 2 and first["error"] is None
    assert first["related"] == ["r0", "r1"]
    assert first["predictions"] and set(first["predictions"][0]) == FIELDS
    assert all(value >= 0 for value in first["latency_ms"])


def test_evaluate_sets_row_errors_fail_zero_and_never_fake_uncertain(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_row_error_type", lambda: RowErrorDouble)
    prepare = prepare_stub(inputs)

    def guarded(row, classes):
        if row.get("id") == "bad":
            raise RowErrorDouble("invalid_request")
        if row.get("id") == "worse":
            raise evaluate.RowContractError("invalid_label")
        return prepare(row, classes)

    monkeypatch.setattr(evaluate, "_prepare", guarded)
    predictor = adapter.make_predictor(constant("complement"), threshold=0.0)
    report = evaluate.evaluate_sets(predictor, [
        {"id": "good", "label": "complement", "group_id": "g1", "expected": ["r0", "r1", "e1"]},
        {"id": "bad", "label": "complement", "group_id": "g1"},
        {"id": "worse", "label": "complement", "group_id": "g1"},
    ])
    bad, worse = report["rows"][1], report["rows"][2]
    for record, code in ((bad, "invalid_request"), (worse, "invalid_label")):
        assert record["row_error"] == code
        assert record["decision"] is None and record["calls"] == 0
        assert record["predictions"] == [] and record["scored_pairs"] == 0
    assert report["summary"]["row_errors"] == {"invalid_label": 1, "invalid_request": 1}
    assert report["summary"]["scored_rows"] == 1
    assert report["summary"]["relations"] == {"complement": 1}
    assert report["summary"]["agreement"]["n"] == 1


def test_evaluate_sets_agreement_risk_refs_and_cluster_ci(inputs, monkeypatch):
    rows = [
        {"id": "a", "label": "complement", "group_id": "g1", "acceptable": ["complement"],
         "expected": ["r0", "r1", "e1"]},
        {"id": "b", "label": "duplicate", "group_id": "g1", "acceptable": ["duplicate", "complement"],
         "expected": ["r0"]},
        {"id": "c", "label": "conflict", "group_id": "g2", "acceptable": ["conflict"],
         "expected": ["r1"]},
        {"id": "d", "label": "uncertain", "group_id": "g2", "acceptable": ["uncertain"],
         "expected": []},
    ]
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs, rules_for(inputs, 2)))
    predictor = adapter.make_predictor(constant("complement"), threshold=0.0)
    report = evaluate.evaluate_sets(predictor, rows)
    summary = report["summary"]
    assert summary["agreement"]["primary"] == 0.25
    assert summary["agreement"]["acceptable"] == 0.5
    assert summary["agreement"]["n"] == 4
    assert summary["operation"]["dangerous_allow_count"] == 2
    assert summary["operation"]["gold_blocking_count"] == 2
    assert summary["operation"]["dangerous_allow_rate"] == \
        {"numerator": 2, "denominator": 2, "value": pytest.approx(1.0)}, "metrics 口径：/ 真实阻塞"
    assert summary["operation"]["conditional_risk"] == \
        {"numerator": 2, "denominator": 4, "value": pytest.approx(0.5)}, "放行决策上的条件风险"
    assert summary["operation"]["false_block_rate"] == \
        {"numerator": 0, "denominator": 2, "value": 0.0}, "metrics 口径：/ 真实允许"
    assert summary["operation"]["false_block_conditional_precision"] == \
        {"numerator": 0, "denominator": 0, "value": None}, "无阻塞决策时 0 分母 null"
    refs = summary["refs"]
    assert refs["predicted_total"] == 12 and refs["expected_total"] == 5 and refs["hits"] == 5
    assert abs(refs["precision"] - 5 / 12) < 1e-12
    assert refs["recall"] == 1.0
    assert refs["unknown_total"] == 0 and refs["unknown_rate"] == 0.0
    assert refs["known_non_expected_total"] == 7
    assert refs["missed_expected_total"] == 0
    assert summary["bootstrap"]["group_clusters"] == 2
    macro_ci = summary["bootstrap"]["macro_f1"]
    assert macro_ci is not None and macro_ci["lo"] <= macro_ci["hi"]
    boot = summary["bootstrap"]
    assert boot["dangerous_allow_rate"]["numerator"] == 2
    assert boot["dangerous_allow_rate"]["denominator"] == 2
    assert boot["conditional_risk"]["denominator"] == 4
    assert boot["false_block_rate"]["denominator"] == 2
    assert boot["false_block_conditional_precision"] is None, "分母恒 0 时无点/CI"
    assert abs(summary["macro_f1"] - 0.4 / 7) < 1e-12
    assert report["rows"][3]["refs"]["recall"] is None, "期望引用为空时 0 分母 null"


def test_prepare_acceptable_labels_paths(inputs, monkeypatch):
    candidate, existing, evidence, cfg = inputs
    monkeypatch.setattr(evaluate, "_materialize", lambda request: (candidate, existing, evidence, cfg))
    base = {"id": "x", "label": "duplicate", "group_id": "g", "label_origin": "agent_review",
            "related_ids": ["r1"], "evidence_refs": ["e1"]}
    assert evaluate._prepare(base, CLASSES)[4]["acceptable"] == ["duplicate"]
    meta = evaluate._prepare({**base, "acceptable_labels": ["duplicate", "complement"]}, CLASSES)[4]
    assert meta["acceptable"] == ["duplicate", "complement"]
    meta = evaluate._prepare({**base, "phenomena": {"acceptable_labels": ["conflict", "duplicate"]}},
                             CLASSES)[4]
    assert meta["acceptable"] == ["conflict", "duplicate"]
    for broken in ({**base, "acceptable_labels": []},
                   {**base, "acceptable_labels": ["unknown"]},
                   {**base, "label": "nope"},
                   {**base, "group_id": ""},
                   {**base, "related_ids": "r1"},
                   {**base, "evidence_refs": None}):
        with pytest.raises(evaluate.RowContractError):
            evaluate._prepare(broken, CLASSES)


def test_evaluate_sets_without_stats_counts_calls_only(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))

    def echo(request):
        return {"relation": "complement", "related_ids": [],
                "reason": "stub judgement", "evidence_refs": [request["evidence"][0]["id"]],
                "error_code": None}

    report = evaluate.evaluate_sets(echo, [{"id": "row-1", "label": "complement", "group_id": "g1"}])
    assert report["rows"][0]["calls"] == 1
    assert report["rows"][0]["scored_pairs"] is None
    assert report["summary"]["scored_pairs"] is None
    assert report["summary"]["relations"] == {"complement": 1}


def _pseudo_record(index, label, relation, group_id):
    """_summarize 单测用的最小已判定记录（字段与真实行记录同形）。"""
    return {"index": index, "row_id": f"row-{index}", "group_id": group_id, "label": label,
            "label_origin": "agent_review", "acceptable": [label], "row_error": None,
            "rules": 1, "predictions": [],
            "decision": {"relation": relation, "related_ids": [], "evidence_refs": [],
                         "reason": "pseudo", "error_code": None},
            "error": None, "related": [],
            "refs": {"predicted": [], "expected": [], "hits": 0, "precision": None, "recall": None,
                     "unknown_ids": [], "known_non_expected": [], "missed_expected": []},
            "primary_agree": False, "acceptable_agree": False,
            "calls": 0, "latency_ms": [], "scored_pairs": 0}


def test_operation_rates_use_metrics_denominators_with_known_counts():
    import time

    from research.memory_relations import metrics as metrics_mod

    blocking_labels = ("conflict", "supersede", "temporary", "uncertain")
    allowing_labels = ("complement", "duplicate", "contextual")
    records = []
    for _ in range(21):                                 
        records.append(_pseudo_record(len(records), blocking_labels[len(records) % 4],
                                      "complement", f"g{len(records) % 5}"))
    for _ in range(8):                       
        records.append(_pseudo_record(len(records), allowing_labels[len(records) % 3],
                                      "complement", f"g{len(records) % 5}"))
    for _ in range(5):           
        records.append(_pseudo_record(len(records), blocking_labels[len(records) % 4],
                                      "conflict", f"g{len(records) % 5}"))
    for _ in range(3):                         
        records.append(_pseudo_record(len(records), allowing_labels[len(records) % 3],
                                      "conflict", f"g{len(records) % 5}"))
    summary = evaluate._summarize(records, {}, metrics_mod, CLASSES, frozenset(ALLOWING),
                                  frozenset(BLOCKING), time.perf_counter())
    operation = summary["operation"]
    assert operation["n"] == 37
    assert operation["dangerous_allow_count"] == 21
    assert operation["allowing_pred_count"] == 29 and operation["blocking_pred_count"] == 8
    assert operation["gold_blocking_count"] == 26 and operation["gold_allowing_count"] == 11
    assert operation["dangerous_allow_rate"] == \
        {"numerator": 21, "denominator": 26, "value": pytest.approx(21 / 26)}
    assert operation["conditional_risk"] == \
        {"numerator": 21, "denominator": 29, "value": pytest.approx(21 / 29)}
    assert operation["coverage"] == \
        {"numerator": 29, "denominator": 37, "value": pytest.approx(29 / 37)}
    assert operation["false_block_rate"] == \
        {"numerator": 3, "denominator": 11, "value": pytest.approx(3 / 11)}
    assert operation["false_block_conditional_precision"] == \
        {"numerator": 3, "denominator": 8, "value": pytest.approx(3 / 8)}
    boot = summary["bootstrap"]
    assert boot["dangerous_allow_rate"]["numerator"] == 21
    assert boot["dangerous_allow_rate"]["denominator"] == 26
    assert boot["conditional_risk"]["denominator"] == 29
    assert boot["false_block_rate"]["denominator"] == 11
    assert boot["false_block_conditional_precision"]["denominator"] == 8
    assert boot["macro_f1"] is not None


def test_zero_risk_gets_rare_upper_bound_and_zero_denominator_nulls():
    import time

    from research.memory_relations import metrics as metrics_mod

    safe = [_pseudo_record(index, ("complement", "duplicate", "contextual")[index % 3],
                           "complement", f"g{index % 4}") for index in range(8)]
    summary = evaluate._summarize(safe, {}, metrics_mod, CLASSES, frozenset(ALLOWING),
                                  frozenset(BLOCKING), time.perf_counter())
    operation = summary["operation"]
    assert operation["dangerous_allow_rate"] == {"numerator": 0, "denominator": 0, "value": None}
    assert operation["conditional_risk"] == {"numerator": 0, "denominator": 8, "value": 0.0}
    boot = summary["bootstrap"]
    assert boot["dangerous_allow_rate"] is None, "0 分母不产生点/CI"
    record = boot["conditional_risk"]
    assert record["zero_errors"] is True and record["rare_upper_bound"] is not None
    assert "簇相关" in record["rare_bound_note"]


                                                                      

def fake_modules(monkeypatch, tmp_path, inputs, *, meta=None):
    candidate, existing, evidence, cfg = inputs
    fake_schema = types.ModuleType("research.memory_relations.schema")
    fake_schema.RowError = RowErrorDouble
    fake_schema.materialize_request = lambda request: (candidate, existing, evidence, cfg)
    fake_schema.load_rows = lambda path: [{
        "id": "row-1", "label": "complement", "group_id": "g1", "label_origin": "agent_review",
        "related_ids": ["r1"], "evidence_refs": ["e1"], "request": {},
    }]
    fake_features = types.ModuleType("research.memory_relations.features")
    fake_features.load_scorer = lambda model_dir, device="cpu": constant("complement")
    monkeypatch.setitem(sys.modules, "research.memory_relations.schema", fake_schema)
    monkeypatch.setitem(sys.modules, "research.memory_relations.features", fake_features)
                                                   
    import research.memory_relations as package
    monkeypatch.setattr(package, "schema", fake_schema, raising=False)
    monkeypatch.setattr(package, "features", fake_features, raising=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if meta is not None:
        (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return model_dir


def test_cli_writes_report_and_predictions_with_meta_threshold(tmp_path, inputs, monkeypatch):
    model_dir = fake_modules(monkeypatch, tmp_path, inputs,
                             meta={"threshold": 0.4, "threshold_feasible": True})
    out = tmp_path / "nested" / "report.json"
    code = evaluate.main(["--model-dir", str(model_dir), "--data", str(tmp_path / "data.jsonl"),
                          "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["operation"] == {
        "requested_threshold": None, "meta_threshold": 0.4,
        "meta_path": str(model_dir / "meta.json"), "meta_threshold_feasible": True,
        "threshold": 0.4, "threshold_source": "meta", "calibrated": True, "diagnostic": False,
    }
    assert report["rows"][0]["decision"]["relation"] == "complement"
    predictions = Path(report["predictions_file"])
    assert predictions.is_file() and predictions.name == "report.predictions.jsonl"
    row = json.loads(predictions.read_text(encoding="utf-8").strip())
    assert row["relation"] == "complement" and row["primary_agree"] is True


def test_cli_threshold_discipline(tmp_path, inputs, monkeypatch):
    model_dir = fake_modules(monkeypatch, tmp_path, inputs,
                             meta={"threshold": None, "threshold_feasible": False})
    out = tmp_path / "report.json"
    args = ["--model-dir", str(model_dir), "--data", str(tmp_path / "data.jsonl"), "--out", str(out)]
    assert evaluate.main(args + ["--threshold", "0.0"]) == 2, "显式阈值必须配 --diagnostic"
    assert evaluate.main(args) == 2, "meta threshold=null 时拒绝已校准操作"
    assert not out.exists()
    assert evaluate.main(args + ["--diagnostic", "--threshold", "0.0"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["operation"]["calibrated"] is False
    assert report["operation"]["diagnostic"] is True
    assert report["operation"]["threshold"] == 0.0
    assert report["operation"]["threshold_source"] == "cli"
    assert report["operation"]["requested_threshold"] == 0.0 and report["operation"]["meta_threshold"] is None
    assert evaluate.main(args + ["--diagnostic"]) == 2, "无显式阈值且 meta 无数值时必须拒绝"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["operation"]["threshold"] == 0.0, "拒绝路径不得改写已有报告"
    assert evaluate.main(args) == 2


                                                                                        

@pytest.fixture
def knowledge(tmp_path):
    conversations = ConversationStore(tmp_path)
    workspace = WorkspaceStore(tmp_path)
    conversations.workspace = workspace
    service = KnowledgeService(conversations, workspace)
    cfg = LLMConfig(enable_llm=True, api_key="synthetic-knowledge-key",
                    base_url="https://model.example/v1", model="relation-model")
    endpoint = hashlib.sha256(f"{cfg.base_url}|{cfg.model}".encode()).hexdigest()[:12]
    return service, cfg, endpoint


def propose_knowledge(service, endpoint, *, request_id="request-1"):
    text = "以后汇报先给结论，再给依据。"
    return service.propose("user-a", endpoint, "", request_id=request_id,
                           operations=[{"op": "create", "after": {"kind": "memory", "title": "汇报偏好",
                                                                  "text": text, "trigger": "", "payload": {}}}],
                           source_refs=[{"type": "user", "text": text}])


def assess_knowledge(service, cfg, endpoint, receipt, predictor):
    return service.assess("user-a", endpoint, change_id=receipt["change_id"],
                          expected_change_revision=receipt["change_revision"], cfg=cfg, predictor=predictor)


def commit_knowledge(service, endpoint, receipt, *, human_confirmed=True):
    return service.commit("user-a", endpoint, change_id=receipt["change_id"],
                          expected_change_revision=receipt["change_revision"],
                          assessment_id=receipt["assessment_id"],
                          expected_catalog_generation=receipt["catalog_generation"],
                          human_confirmed=human_confirmed)


def test_knowledge_replay_complement_still_requires_human_commit(knowledge):
    service, cfg, endpoint = knowledge
    receipt = propose_knowledge(service, endpoint)
    scorer = constant("complement")
    assessed = assess_knowledge(service, cfg, endpoint, receipt,
                                adapter.make_predictor(scorer, threshold=0.0))
    assert assessed["ok"] and assessed["stage"] == "pending"
    assert assessed["assessment"]["committable"] is True
    pairs = scorer.calls[0]
    assert pairs[0]["context"]["complete_impact_set"] is True
    assert any(rule["scope"] == "readonly_system" for pair in pairs for rule in pair["existing"])
    denied = commit_knowledge(service, endpoint, assessed, human_confirmed=False)
    assert denied["error_code"] == "human_confirmation_required"
    assert service.entries("user-a", endpoint)["items"] == []
    assert commit_knowledge(service, endpoint, assessed)["ok"]
    assert len(service.entries("user-a", endpoint)["items"]) == 1


def test_knowledge_replay_uncertain_is_not_committable(knowledge):
    service, cfg, endpoint = knowledge
    receipt = propose_knowledge(service, endpoint)
    assessed = assess_knowledge(service, cfg, endpoint, receipt,
                                adapter.make_predictor(constant("uncertain"), threshold=0.0))
    assert not assessed["ok"]
    assert assessed["error_code"] == "model_uncertain"
    assert assessed["stage"] == "draft"
    assert assessed["assessment"]["committable"] is False
    assert commit_knowledge(service, endpoint, assessed)["error_code"] == "knowledge_not_committable"
    assert service.entries("user-a", endpoint)["items"] == []


                                                                                    

def test_real_e2e_tiny_classifier_schema_and_cli(tmp_path):
    """真实小分类器保存/重载 + 真实 schema + build_data 语料 + evaluate.main（不用 stub）。

    仅当 features/schema/train/metrics.pair_verdict/build_data 全部就绪时运行；
    未就绪自动 skip（不静默降级，也不写主树）。
    """
    features = pytest.importorskip("research.memory_relations.features")
    schema = pytest.importorskip("research.memory_relations.schema")              
    train = pytest.importorskip("research.memory_relations.train")
    metrics_mod = pytest.importorskip("research.memory_relations.metrics")
    if not callable(getattr(metrics_mod, "pair_verdict", None)):
        pytest.skip("metrics.pair_verdict 尚未合入：真实端到端评测等待 metrics 团队")
    build = pytest.importorskip("research.memory_relations.build_data")

    corpus = build.build_corpus(20261011, build.PLAN_QUICK)
    rows = corpus["rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]

    texts = [features.request_text(row["request"]) for row in train_rows]
    labels = np.asarray([features.CLASSES.index(row["label"]) for row in train_rows], dtype=int)
    model, class_indices = train.fit_model("char-logreg", texts, labels, 13)
    scorer = features.RelationScorer(kind="char-logreg", model=model, class_indices=class_indices)
    model_dir = tmp_path / "scorer"
    train.save_scorer(scorer, model_dir, threads=1, batch_size=8)
    parity = train.check_reload_parity(scorer, model_dir,
                                       [row["request"] for row in heldout_rows[:8]], device="cpu")
    assert parity["ok"], parity
    assert features.load_scorer(model_dir, device="cpu").classes == features.CLASSES

    bad_missing = json.loads(json.dumps(heldout_rows[0]))
    bad_missing["id"] = "bad-missing-context"
    bad_missing["request"] = {"candidate": {"text": "x", "tier": "explicit"}, "existing": [],
                              "evidence": []}
    bad_context = json.loads(json.dumps(heldout_rows[1]))
    bad_context["id"] = "bad-unknown-context"
    bad_context["request"] = {
        "candidate": {"principal": "user-a", "endpoint_fp": "0" * 12, "project_id": "p",
                      "text": "提交前先格式化。", "trigger": "", "tier": "explicit", "evidence": []},
        "existing": [], "evidence": [],
        "context": {"scope": "project", "expected": []}}
    data_path = tmp_path / "heldout.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in heldout_rows + [bad_missing, bad_context]), encoding="utf-8")

    (model_dir / "meta.json").write_text(json.dumps(
        {"threshold": 0.4, "threshold_feasible": True}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert evaluate.main(["--model-dir", str(model_dir), "--data", str(data_path),
                          "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["operation"]["calibrated"] is True and report["operation"]["threshold"] == 0.4
    summary = report["summary"]
    assert summary["row_count"] == len(heldout_rows) + 2
    assert summary["scored_rows"] == len(heldout_rows)
    assert summary["row_errors"] == {"invalid_request": 1, "invalid_context": 1}
    assert summary["agreement"]["primary"] is not None
    assert summary["agreement"]["acceptable"] is not None
    assert summary["macro_f1"] is not None
    assert summary["confusion"]["classes"] == list(features.CLASSES)
    assert summary["valid_rows"] == summary["scored_rows"] - summary["gate_error_rows"]
    assert summary["refs"]["hits"] <= summary["refs"]["predicted_total"]
    assert summary["refs"]["unknown_total"] == 0, "引用只能来自输入（unknown id 为 0）"
    assert summary["refs"]["missed_expected_total"] >= 0
    assert set(summary["refs"]) >= {"precision", "recall", "unknown_total", "unknown_rate",
                                    "known_non_expected_total", "missed_expected_total"}
    assert summary["bootstrap"]["group_clusters"] >= 2
    assert Path(report["predictions_file"]).is_file()
    for record in report["rows"][-2:]:
        assert record["row_error"] and record["decision"] is None and record["calls"] == 0

    assert evaluate.main(["--model-dir", str(model_dir), "--data", str(data_path),
                          "--out", str(out), "--threshold", "0.0"]) == 2
    (model_dir / "meta.json").write_text(json.dumps(
        {"threshold": None, "threshold_feasible": False}), encoding="utf-8")
    assert evaluate.main(["--model-dir", str(model_dir), "--data", str(data_path),
                          "--out", str(out)]) == 2
    assert evaluate.main(["--model-dir", str(model_dir), "--data", str(data_path),
                          "--out", str(out), "--diagnostic", "--threshold", "0.0"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["operation"]["calibrated"] is False


                                                                                         

def _tiny_joint_artifacts(tmp_path):
    """真实 tiny BERT（2 层 hidden 32）+ 真实 producer 工件（非 mock 替身）；禁止下载。

    两臂（frozen-joint/head_only、finetuned-joint/full）各经 finetune.save_scorer 持久化，
    meta 顶层携带 calibration 选点 threshold=0.8；另存 threshold=null 的未校准工件。
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    finetune = pytest.importorskip("research.memory_relations.finetune")
    from transformers import BertConfig, BertModel, BertTokenizer

    corpus = ("汇报先给结论，再给依据。汇报先说结论，再解释原因。"
              "以后汇报先给结论，再给依据。汇总写一行。做校正，然后汇总。")
    alphabet = sorted(set(corpus + "[]_abcdefghijklmnopqrstuvwxyz0123456789"))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet
    backbone_dir = tmp_path / "tiny-backbone"
    backbone_dir.mkdir()
    (backbone_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(backbone_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(str(backbone_dir))
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(str(backbone_dir))
    fingerprint = FEAT.model_fingerprint(backbone_dir)

    def persist(name, kind, threshold, feasible, seed):
        model, _ = finetune.build_joint_model(backbone_dir)
        scorer = finetune.CrossEncoderScorer(kind=kind, model=model, tokenizer=tokenizer,
                                             max_length=48, device="cpu", temperature=1.0,
                                             threshold=threshold, threshold_feasible=feasible)
        artifact = tmp_path / name
        finetune.save_scorer(scorer, artifact, weights=finetune.EXPECTED_WEIGHTS[kind],
                             backbone_dir=backbone_dir, backbone_fingerprint=fingerprint,
                             train_meta={"seed": seed, "smoke": True})
        return artifact

    return {"frozen": persist("artifact-frozen", "frozen-joint", 0.8, True, 13),
            "full": persist("artifact-full", "finetuned-joint", 0.8, True, 17),
            "null": persist("artifact-null", "frozen-joint", None, False, 19)}


def _evaluate_rows():
    quote = "以后汇报先给结论，再给依据。"
    scope = {"principal": "user-a", "endpoint_fp": "c27e1bc81a50", "project_id": "proj-alpha"}
    request = {
        "candidate": {**scope, "text": "汇报先给结论，再给依据。", "trigger": "汇报时",
                      "tier": "explicit", "evidence": [quote]},
        "existing": [{**scope, "id": "r1", "text": "汇报先说结论，再解释原因。",
                      "trigger": "汇报时", "tier": "explicit", "evidence": []}],
        "evidence": [{"id": "e1", "text": quote}],
        "context": {"scope": "project", "replaceable_ids": ["r1"], "inherited_global_ids": []},
    }
    return [{"id": "e2e-1", "group_id": "g1", "split": "ood", "request": request,
             "label": "duplicate", "label_origin": "agent_review", "related_ids": ["r1"],
             "evidence_refs": ["e1"],
             "phenomena": {"domain": "writing", "relational_cases": ["duplicate"],
                           "rationale": "e2e"}}]


def test_public_loader_finetune_tiny_cpu_through_adapter_and_cli(tmp_path, inputs):
    """公共 loader→adapter→evaluate 的真实 tiny CPU 端到端（meta 由真实 producer 写，无 mock）。

    两臂工件（frozen-joint/head_only、finetuned-joint/full）meta 顶层 threshold=0.8 可行：
    默认即校准运行（threshold_source=meta）；null 工件（threshold=null）默认清晰拒绝（退出 2，
    不产出报告），显式 --diagnostic --threshold 0.0 成功且 calibrated=False。fixture 不改 meta。
    """
    artifacts = _tiny_joint_artifacts(tmp_path)
    rows = _evaluate_rows()
    data_path = tmp_path / "rows.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    out_path = tmp_path / "report.json"

    for name, kind in (("frozen", "frozen-joint"), ("full", "finetuned-joint")):
        meta = json.loads((artifacts[name] / "meta.json").read_text(encoding="utf-8"))
        assert meta["threshold"] == 0.8 and meta["threshold_feasible"] is True
        loaded = FEAT.load_scorer(artifacts[name], device="cpu")                       
        assert type(loaded).__name__ == "CrossEncoderScorer"
        assert tuple(loaded.classes) == FEAT.CLASSES and loaded.kind == kind
        assert loaded.threshold == 0.8 and loaded.threshold_feasible is True
        outcome = run(inputs, adapter.make_predictor(loaded, threshold=0.0))
        assert set(outcome) == FIELDS and outcome["relation"] in FEAT.CLASSES
        args = ["--model-dir", str(artifacts[name]), "--data", str(data_path),
                "--out", str(out_path)]
        assert evaluate.main(args) == 0
        report = json.loads(out_path.read_text(encoding="utf-8"))
        operation = report["operation"]
        assert operation["calibrated"] is True and operation["threshold"] == 0.8
        assert operation["threshold_source"] == "meta" and operation["diagnostic"] is False
        assert operation["requested_threshold"] is None and operation["meta_threshold"] == 0.8
        summary = report["summary"]
        assert summary["row_count"] == len(rows) and summary["valid_rows"] <= summary["scored_rows"]
        assert set(summary["refs"]) >= {"unknown_total", "known_non_expected_total",
                                        "missed_expected_total"}

    null_meta = json.loads((artifacts["null"] / "meta.json").read_text(encoding="utf-8"))
    assert null_meta["threshold"] is None and null_meta["threshold_feasible"] is False
    loaded = FEAT.load_scorer(artifacts["null"], device="cpu")
    assert loaded.threshold is None and loaded.threshold_feasible is False
    outcome = run(inputs, adapter.make_predictor(loaded, threshold=0.0))
    assert set(outcome) == FIELDS and outcome["relation"] in FEAT.CLASSES
    null_out = tmp_path / "null-report.json"
    args = ["--model-dir", str(artifacts["null"]), "--data", str(data_path), "--out", str(null_out)]
    assert evaluate.main(args + ["--threshold", "0.0"]) == 2, "显式阈值必须配 --diagnostic"
    assert evaluate.main(args) == 2, "meta threshold=null：默认必须清晰拒绝"
    assert not null_out.exists()
    assert evaluate.main(args + ["--diagnostic", "--threshold", "0.0"]) == 0
    report = json.loads(null_out.read_text(encoding="utf-8"))
    assert report["operation"]["diagnostic"] is True and report["operation"]["calibrated"] is False
    assert report["operation"]["threshold"] == 0.0
    assert report["operation"]["threshold_source"] == "cli"
    summary = report["summary"]
    assert summary["row_count"] == len(rows) and summary["valid_rows"] <= summary["scored_rows"]
    assert set(summary["refs"]) >= {"unknown_total", "known_non_expected_total",
                                    "missed_expected_total"}


def test_all_attempt_denominator_keeps_row_errors_and_semantic_uncertain(inputs, monkeypatch):
    prepare = prepare_stub(inputs)

    def guarded(row, classes):
        if row["id"] == "bad":
            raise evaluate.RowContractError("invalid_row")
        return prepare(row, classes)

    monkeypatch.setattr(evaluate, "_prepare", guarded)
    rows = [{"id": "good", "label": "uncertain", "group_id": "g1"},
            {"id": "bad", "label": "uncertain", "group_id": "g2"}]
    report = evaluate.evaluate_sets(adapter.make_predictor(constant("uncertain"), threshold=0.), rows)
    summary = report["summary"]
    assert summary["operation"]["n"] == 2
    assert summary["all_attempt"]["completion"] == {"numerator": 1., "denominator": 2., "value": .5}
    assert summary["all_attempt"]["correct_completed"] == {"numerator": 1., "denominator": 2., "value": .5}
    assert summary["semantic_uncertain_rows"] == 1 and summary["system_failure_rows"] == 1
    assert summary["valid_rows"] == 0 and summary["agreement"]["primary"] is None
    assert "secondary" in summary["classification_scope"]
    assert summary["gated_set"]["confusion"]["matrix"][-1][-1] == 1
    assert summary["gated_set"]["confusion"]["matrix"][-1][-2] == 1


def test_predictor_exception_attempt_and_latency_are_recorded_in_finally(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))

    def fail(request):
        raise TimeoutError("not recorded private exception body")

    report = evaluate.evaluate_sets(fail, [{"id": "fail", "label": "complement", "group_id": "g"}])
    row, summary = report["rows"][0], report["summary"]
    assert row["decision"]["error_code"] == "predictor_timeout" and row["completed"] is False
    assert row["calls"] == 1 and row["call_failures"] == ["TimeoutError"]
    assert len(row["latency_ms"]) == 1 and row["latency_ms"][0] >= 0
    assert summary["predictor_calls"] == summary["failed_predictor_calls"] == 1
    assert summary["all_attempt"]["completion"]["value"] == 0.
    assert summary["operation"]["false_block_count"] == 0
    assert summary["operation"]["failed_gold_allowing_count"] == 1
    assert "not recorded" not in json.dumps(report)


def test_raw_classifier_pair_gate_and_production_gate_are_separate(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))
    inputs[0]["tier"] = "implicit"
    rows = [{"id": "denied", "label": "supersede", "group_id": "g"}]
    predictor = evaluate.make_evaluation_predictor(constant("supersede"), 0.)
    report = evaluate.evaluate_sets(predictor, rows)
    summary = report["summary"]
    index = CLASSES.index("supersede")
    assert summary["classifier"]["classifier_relation"]["confusion"]["matrix"][index][index] == 1
    assert summary["classifier"]["pair_relation"]["confusion"]["matrix"][index][index] == 1
    assert summary["raw_set"]["confusion"]["matrix"][index][index] == 1
    assert summary["gated_set"]["confusion"]["matrix"][index][-1] == 1
    assert report["rows"][0]["error"] == "replacement_requires_explicit"
    assert summary["all_attempt"]["correct_completed"]["value"] == 0.


def test_low_confidence_pair_is_semantic_abstention_not_system_failure(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))
    scorer = RecordingScorer(lambda requests: dist("complement", .4, len(requests)))
    predictor = evaluate.make_evaluation_predictor(scorer, .8)
    report = evaluate.evaluate_sets(predictor, [{"label": "complement", "group_id": "g"}])
    summary = report["summary"]
    assert summary["classifier"]["classifier_relation"]["confusion"]["matrix"][0][0] == 1
    assert summary["classifier"]["pair_relation"]["confusion"]["matrix"][0][-1] == 1
    assert summary["semantic_uncertain_rows"] == 1 and summary["system_failure_rows"] == 0
    assert summary["all_attempt"]["completion"]["value"] == 1.
    assert summary["all_attempt"]["correct_completed"]["value"] == 0.


def test_malformed_classifier_matrix_is_system_failure_not_semantic_uncertain(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))
    scorer = RecordingScorer(lambda requests: np.full((len(requests), len(CLASSES)), np.nan))
    report = evaluate.evaluate_sets(evaluate.make_evaluation_predictor(scorer, 0.),
                                   [{"label": "uncertain", "group_id": "g"}])
    row, summary = report["rows"][0], report["summary"]
    assert row["error"] == "model_uncertain"
    assert row["classifier_errors"] == ["invalid_probability_matrix"]
    assert summary["classifier"]["observed"] is True
    assert summary["classifier"]["pair_attempts"] == summary["classifier"]["missing_outputs"] == 1
    assert summary["semantic_uncertain_rows"] == 0 and summary["system_failure_rows"] == 1
    assert summary["all_attempt"]["correct_completed"]["value"] == 0.


def test_role_swapped_references_do_not_get_union_credit(inputs, monkeypatch):
    monkeypatch.setattr(evaluate, "_materialize", lambda request: inputs)
    row = {"id": "swap", "label": "complement", "group_id": "g", "label_origin": "agent_review",
           "related_ids": ["r1"], "evidence_refs": ["e1"], "request": {}}

    def swap(request):
        return {"relation": "complement", "related_ids": ["e1"], "evidence_refs": ["r1"],
                "reason": "synthetic role swap", "error_code": None}

    report = evaluate.evaluate_sets(swap, [row])
    roles = report["summary"]["refs_by_role"]
    assert report["rows"][0]["raw_decision"]["relation"] == "complement"
    assert report["rows"][0]["completed"] is False
    for role in ("related_ids", "evidence_refs"):
        assert roles["raw"][role]["precision"]["value"] == 0.
        assert roles["raw"][role]["recall"]["value"] == 0.
        assert roles["raw"][role]["unknown_total"] == 1
        assert roles["gated"][role]["recall"]["value"] == 0.


def test_role_recall_keeps_expected_refs_from_bad_request(inputs, monkeypatch):
    def bad(request):
        raise RowErrorDouble("invalid_request")

    monkeypatch.setattr(evaluate, "_row_error_type", lambda: RowErrorDouble)
    monkeypatch.setattr(evaluate, "_materialize", bad)
    row = {"label": "complement", "group_id": "g", "related_ids": ["r1"], "evidence_refs": ["e1"]}
    report = evaluate.evaluate_sets(constant("complement"), [row])
    refs = report["summary"]["refs_by_role"]["gated"]
    assert refs["related_ids"]["recall"] == {"numerator": 0, "denominator": 1, "value": 0.}
    assert refs["evidence_refs"]["recall"] == {"numerator": 0, "denominator": 1, "value": 0.}
    assert report["summary"]["all_attempt"]["n"] == 1


def test_group_equal_and_primary_operating_do_not_treat_variants_as_independent(inputs, monkeypatch):
    prepare = prepare_stub(inputs)

    def prepared(row, classes):
        candidate, existing, evidence, cfg, meta = prepare(row, classes)
        meta["operating_primary"] = row["phenomena"]["operating_primary"]
        return candidate, existing, evidence, cfg, meta

    monkeypatch.setattr(evaluate, "_prepare", prepared)
    rows = [{"label": "complement", "group_id": "a", "phenomena": {"operating_primary": i == 0}}
            for i in range(20)]
    rows.append({"label": "conflict", "group_id": "b", "phenomena": {"operating_primary": True}})
    report = evaluate.evaluate_sets(adapter.make_predictor(constant("complement"), threshold=0.), rows)
    summary = report["summary"]
    assert summary["operation"]["conditional_risk"]["value"] == pytest.approx(1 / 21)
    assert summary["group_equal"]["operation"]["conditional_risk"]["value"] == pytest.approx(.5)
    primary = summary["primary_operating"]
    assert primary["available"] is True and primary["n"] == 2
    assert primary["operation"]["conditional_risk"]["denominator"] == 2
    assert primary["joint_bounds"]["conditional_risk_upper"] > .5
    record = summary["group_equal"]["bootstrap"]["conditional_risk"]
    assert record["raw_numerator"] == 1 and record["raw_denominator"] == 21
    assert record["numerator"] == 1 and record["denominator"] == pytest.approx(2)


def test_oracle_probe_reuses_real_adapter_and_exposes_cross_batch_union(inputs, monkeypatch):
    rules = rules_for(inputs, 65)
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs, rules))
    pairs = [{"rule_id": rule["id"], "relation": "conflict" if i == 64 else "complement"}
             for i, rule in enumerate(rules)]
    row = {"label": "conflict", "group_id": "g", "expected": ["r64", "e1"],
           "phenomena": {"pair_labels": pairs}}
    predictor = adapter.make_predictor(by_rule(r64="conflict"), threshold=0.)
    report = evaluate.evaluate_sets(predictor, [row], oracle_aggregation=True)
    oracle = report["rows"][0]["oracle"]
    assert oracle["available"] is True and oracle["pairs"] == 65
    assert oracle["primary_agree"] is True
    assert oracle["decision"]["related_ids"] == [f"r{i}" for i in range(65)]
    assert report["summary"]["refs_by_role"]["gated"]["related_ids"]["known_non_expected_total"] == 64
    row["phenomena"]["pair_labels"].pop()
    report = evaluate.evaluate_sets(predictor, [row], oracle_aggregation=True)
    assert report["rows"][0]["oracle"]["available"] is False


@pytest.mark.parametrize("feasible", [None, False, 1, "true", [], {}])
def test_numeric_meta_threshold_requires_strict_true_feasibility(tmp_path, feasible):
    model = tmp_path / "model"
    model.mkdir()
    (model / "meta.json").write_text(json.dumps({"threshold": .4, "threshold_feasible": feasible}),
                                    encoding="utf-8")
    with pytest.raises(ValueError, match="threshold_feasible"):
        evaluate._operating_threshold(None, False, model)
    assert evaluate._operating_threshold(None, True, model)["calibrated"] is False


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1., 1.1, True, "0.4"])
def test_meta_invalid_threshold_never_runs_even_diagnostic(tmp_path, threshold):
    model = tmp_path / "model"
    model.mkdir()
    (model / "meta.json").write_text(json.dumps({"threshold": threshold, "threshold_feasible": True}),
                                    encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate._operating_threshold(None, True, model)


def test_all_attempt_handles_empty_and_non_dict_rows():
    def unused(request):
        raise AssertionError("invalid input cannot call predictor")

    for rows in ([], [None, ["bad"], {"label": ["complement"]}]):
        report = evaluate.evaluate_sets(unused, rows)
        assert report["summary"]["all_attempt"]["n"] == len(rows)
        assert report["summary"]["predictor_calls"] == 0
        assert report["summary"]["all_attempt"]["correct_completed"]["value"] == (0. if rows else None)


def test_oracle_separates_perfect_pair_labels_from_set_aggregation_failure(inputs, monkeypatch):
    rules = rules_for(inputs, 2)
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs, rules))
    row = {"label": "contextual", "group_id": "g", "expected": ["r0", "e1"],
           "phenomena": {"pair_labels": [{"rule_id": "r0", "relation": "contextual"},
                                          {"rule_id": "r1", "relation": "temporary"}]}}
    scorer = by_rule(r0="contextual", r1="temporary")
    report = evaluate.evaluate_sets(evaluate.make_evaluation_predictor(scorer, 0.), [row],
                                   oracle_aggregation=True)
    summary = report["summary"]
    assert summary["classifier"]["classifier_relation"]["errors"] == 0
    assert summary["classifier"]["pair_relation"]["errors"] == 0
    assert summary["oracle_aggregation"]["set_relation_errors"] == 1
    assert report["rows"][0]["oracle"]["decision"]["error_code"] == "ambiguous_relation"
    assert summary["all_attempt"]["completion"]["value"] == 1.
    assert summary["all_attempt"]["correct_completed"]["value"] == 0.


def test_observation_wrapper_preserves_real_adapter_results_and_empty_oracle(inputs, monkeypatch):
    with pytest.raises(ValueError):
        evaluate.make_evaluation_predictor(types.SimpleNamespace(classes=list(CLASSES)), 0.)
    with pytest.raises(ValueError):
        evaluate.make_evaluation_predictor(object(), 0.)
    scorer = constant("complement")
    expected = run(inputs, adapter.make_predictor(scorer, threshold=0.))
    observed = evaluate.make_evaluation_predictor(scorer, 0.)
    assert run(inputs, observed) == expected
    assert observed.stats == {"calls": 1, "pairs": 1}
    inputs[1].clear()
    monkeypatch.setattr(evaluate, "_prepare", prepare_stub(inputs))
    row = {"label": "complement", "group_id": "g", "expected": ["e1"],
           "phenomena": {"pair_labels": []}}
    report = evaluate.evaluate_sets(observed, [row], oracle_aggregation=True)
    assert report["rows"][0]["oracle"]["primary_agree"] is True
    assert report["rows"][0]["oracle"]["pairs"] == 1
    assert report["summary"]["primary_operating"]["available"] is False
    assert "joint_bounds" not in report["summary"]["primary_operating"]
