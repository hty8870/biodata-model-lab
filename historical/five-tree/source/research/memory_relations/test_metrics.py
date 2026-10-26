"""分区校准与组级统计的已知反例；合成数值仅验证数学与契约，不是安全认证。"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta

from research.memory_relations import metrics as M


def partition_case(*, audit_groups=500, audit_allow=400, audit_errors=0, variants=1):
    labels, predictions, groups, roles, primary = [], [], [], [], []
    for role, count, allowed in (("prob", 8, 4), ("policy", 100, 80),
                                ("audit", audit_groups, audit_allow)):
        for index in range(count):
            label = "complement" if index < allowed else "conflict"
            prediction = label
            if role == "audit" and index < audit_errors:
                label = "conflict"
            for variant in range(variants if role == "audit" else 1):
                labels.append(M.CLASSES.index(label))
                predictions.append(M.CLASSES.index(prediction))
                groups.append(f"{role}-{index}")
                roles.append(role)
                primary.append(variant == 0)
    logits = np.full((len(labels), len(M.CLASSES)), -10.0)
    logits[np.arange(len(labels)), predictions] = 10.0
    return (logits, np.asarray(labels), np.ones(len(labels), dtype=bool), groups, roles,
            np.asarray(primary))


def test_partitioned_accepts_only_jointly_audited_point():
    case = partition_case()
    result = M.calibrate_partitioned(*case)
    assert result["threshold"] == 0.0 and result["threshold_feasible"] is True
    assert result["selection"]["point_feasible"] is True
    audit = result["audit"]
    assert audit["primary_count"] == audit["independent_groups"] == 500
    assert audit["allow_count"] == 400 and audit["error_count"] == 0
    assert audit["conditional_risk_upper"] == pytest.approx(beta.ppf(.975, 1, 400))
    assert audit["coverage_lower"] == pytest.approx(beta.ppf(.025, 400, 101))
    assert audit["risk_alpha"] == pytest.approx(.025)
    assert audit["coverage_alpha"] == pytest.approx(.025)
    assert audit["minimum_zero_error_allow_groups"] == 368
    assert audit["sufficient"] is True and "human gold" in audit["note"]


@pytest.mark.parametrize("groups,allowed,errors", [(50, 40, 0), (500, 400, 1), (600, 300, 0)])
def test_point_estimates_do_not_certify_sparse_risky_or_low_coverage_audit(groups, allowed, errors):
    result = M.calibrate_partitioned(*partition_case(audit_groups=groups, audit_allow=allowed,
                                                     audit_errors=errors))
    assert result["selection"]["candidate_threshold"] == 0.0
    assert result["selection"]["point_feasible"] is True
    assert result["audit"]["threshold"] == 0.0 and result["audit"]["locked"] is True
    assert result["threshold"] is None and result["threshold_feasible"] is False
    assert result["audit"]["sufficient"] is False


def test_variants_never_inflate_primary_audit_evidence():
    small = M.calibrate_partitioned(*partition_case(audit_groups=50, audit_allow=40))
    repeated = M.calibrate_partitioned(*partition_case(audit_groups=50, audit_allow=40, variants=20))
    assert repeated["audit"]["sample_count"] == 1000
    for name in ("independent_groups", "primary_count", "allow_count", "error_count",
                 "conditional_risk_upper", "coverage_lower"):
        assert repeated["audit"][name] == small["audit"][name]
    assert repeated["threshold"] is None


def test_audit_does_not_refit_temperature_or_reselect_threshold():
    case = partition_case()
    original = M.calibrate_partitioned(*case)
    changed = case[0].copy()
    audit = np.asarray(case[4]) == "audit"
    changed[audit] *= -1
    result = M.calibrate_partitioned(changed, *case[1:])
    assert result["temperature"] == original["temperature"]
    assert result["selection"] == original["selection"]
    assert result["threshold"] is None


def test_no_candidate_leaves_audit_unmeasured_not_fake_tau0():
    z, y, triggers, groups, roles, primary = partition_case()
    policy = np.asarray(roles) == "policy"
    z[policy] = -10
    z[policy, M.CLASSES.index("conflict")] = 10
    result = M.calibrate_partitioned(z, y, triggers, groups, roles, primary)
    assert result["selection"]["candidate_threshold"] is None
    assert result["selection"]["point"] is None
    assert result["selection"]["point_feasible"] is False
    assert result["audit"]["locked"] is False
    for name in ("threshold", "allow_count", "error_count", "conditional_risk_upper", "coverage_lower"):
        assert result["audit"][name] is None


def test_missing_prob_partition_is_explicitly_unfitted_and_not_calibrated():
    z, y, triggers, groups, roles, primary = partition_case()
    mask = np.asarray(roles) != "prob"
    result = M.calibrate_partitioned(z[mask], y[mask], triggers[mask],
                                    np.asarray(groups)[mask], np.asarray(roles)[mask], primary[mask])
    assert result["temperature"] == 1.0
    assert result["selection"]["temperature_fitted"] is False
    assert result["threshold"] is None


@pytest.mark.parametrize("invalid", ["role_leak", "unknown_role", "two_primary", "no_primary",
                                     "bool_label", "float_label", "nan", "int_primary", "shape"])
def test_partition_validation_fails_closed(invalid):
    z, y, triggers, groups, roles, primary = partition_case(audit_groups=4, audit_allow=3, variants=2)
    if invalid == "role_leak":
        groups[-1] = groups[0]
    elif invalid == "unknown_role":
        roles[-1] = "heldout"
    elif invalid == "two_primary":
        primary[-1] = True
    elif invalid == "no_primary":
        primary[0] = False
    elif invalid == "bool_label":
        y = y.astype(bool)
    elif invalid == "float_label":
        y = y.astype(float)
    elif invalid == "nan":
        z[0, 0] = np.nan
    elif invalid == "int_primary":
        primary = primary.astype(int)
    elif invalid == "shape":
        z = z[:, :-1]
    with pytest.raises(ValueError):
        M.calibrate_partitioned(z, y, triggers, groups, roles, primary)


def test_group_prior_mathematical_limit_is_model_independent():
    labels = np.tile(np.arange(len(M.CLASSES)), 10)
    prior = M.group_prior(labels, [f"g{i}" for i in range(len(labels))])
    assert prior["allowing_prior"] == pytest.approx(3 / 7)
    assert prior["max_coverage_at_target_risk"] == pytest.approx((3 / 7) / .99)
    assert prior["min_coverage_possible"] is False
    labels = np.asarray([M.CLASSES.index("complement")] * 99 + [M.CLASSES.index("conflict")])
    prior = M.group_prior(labels, ["safe"] * 99 + ["unsafe"])
    assert prior["allowing_prior"] == pytest.approx(.5)
    assert prior["sample_count"] == 100 and prior["independent_groups"] == 2


def test_group_equal_temperature_is_invariant_to_duplicating_one_root():
    z = np.asarray([[3, 1, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0]], dtype=float)
    y = np.asarray([0, 1])
    original = M.fit_temperature(z, y, sample_weight=M._group_weights(["a", "b"]))
    duplicate_z = np.concatenate([np.repeat(z[:1], 50, axis=0), z[1:]])
    duplicate_y = np.asarray([0] * 50 + [1])
    repeated = M.fit_temperature(duplicate_z, duplicate_y,
                                 sample_weight=M._group_weights(["a"] * 50 + ["b"]))
    assert repeated == pytest.approx(original)


def test_group_equal_bootstrap_reuses_engine_and_retains_raw_counts():
    labels = np.asarray([0] * 99 + [1])
    probs = np.zeros((100, len(M.CLASSES)))
    probs[:, 0] = 1
    groups = ["safe"] * 99 + ["unsafe"]
    row = M.bootstrap_ci(probs, labels, np.ones(100, dtype=bool), groups, 0., samples=40)
    equal = M.bootstrap_ci(probs, labels, np.ones(100, dtype=bool), groups, 0., samples=40,
                           weighting="group_equal")
    assert row["conditional_risk"]["point"] == pytest.approx(.01)
    record = equal["conditional_risk"]
    assert record["point"] == pytest.approx(.5)
    assert record["numerator"] == pytest.approx(1.) and record["denominator"] == pytest.approx(2.)
    assert record["raw_numerator"] == 1 and record["raw_denominator"] == 100
    assert record["rare_upper_bound"] is None
    assert equal["raw_operating"]["conditional_risk"] == .01


def test_empty_and_single_root_bootstrap_do_not_fabricate_intervals():
    for groups in ([], ["a"]):
        result = M.cluster_bootstrap(groups, lambda index, weights: None if not len(index) else 0.,
                                     samples=20)
        assert result["ci"] is None and result["invalid"] == 20


def test_pair_verdict_prior_gate_remains_unchanged():
    probs = np.zeros(len(M.CLASSES))
    probs[M.CLASSES.index("complement")] = .5
    probs[M.CLASSES.index("conflict")] = .5
    request = {"candidate": {"text": "x", "trigger": ""}, "existing": [], "evidence": [], "context": {}}
    assert M.pair_verdict(probs, request, 0.) == "uncertain"
    probs[:] = 0
    probs[M.CLASSES.index("contextual")] = 1
    assert M.pair_verdict(probs, request, 0.) == "uncertain"
    assert M.TARGET_DANGEROUS_ALLOW == .01 and M.MIN_COVERAGE == .5


@pytest.mark.parametrize("n_classes", [3, 5, 7])
def test_failure_aware_core_preserves_legacy_macro_and_weighted_sklearn(n_classes):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(44)
    gold = rng.integers(n_classes, size=80)
    predicted = rng.integers(n_classes, size=80)
    weights = rng.uniform(0., 3., size=80)
    classes = tuple(str(i) for i in range(n_classes))
    for sample_weight in (None, weights):
        expected = f1_score(gold, predicted, labels=list(range(n_classes)), average="macro",
                            zero_division=0, sample_weight=sample_weight)
        assert M.macro_f1(gold, predicted, n_classes=n_classes, sample_weight=sample_weight) == pytest.approx(expected)
        summary = M.classification_summary(gold, predicted, classes=classes, sample_weight=sample_weight)
        assert summary["macro_f1"] == pytest.approx(expected)
        assert summary["failure_weight"] == 0
    assert M.macro_f1(gold % 3, predicted % 3) == M.macro_f1(gold % 3, predicted % 3, n_classes=7)


def test_failure_predictions_add_only_gold_false_negatives_with_no_extra_class():
    summary = M.classification_summary([0, 0, 1, 1], [0, -1, 0, -1], classes=("a", "b"),
                                       sample_weight=[1., 3., 2., 4.])
    assert summary["n"] == 4 and summary["weight"] == 10
    assert summary["accuracy"] == .1
    assert summary["hard_success_weight"] == 3 and summary["failure_weight"] == 7
    assert summary["confusion_matrix"] == {"classes": ["a", "b"], "matrix": [[1., 0.], [2., 0.]],
                                           "failure_by_gold": [3., 4.]}
    assert summary["per_class"]["a"]["fn"] == 3
    assert summary["per_class"]["a"]["fp"] == 2
    assert summary["per_class"]["b"]["fn"] == 6
    assert summary["macro_f1"] == pytest.approx((2 / 7) / 2)
    assert M.macro_f1([0, 1], [-1, -1], n_classes=2) == 0
    with pytest.raises(ValueError):
        M.classification_summary([0], [2], classes=("a", "b"))


def test_empty_summary_is_null_but_legacy_math_keeps_zero():
    summary = M.classification_summary([], [], classes=("a", "b"))
    assert summary["accuracy"] is summary["macro_f1"] is None
    assert M.macro_f1([], []) == 0
    summary = M.probability_summary([], [], n_classes=2)
    assert summary["coverage"] is summary["brier"] is summary["ece"] is None
    assert summary["nll"] == {"value": None, "status": "not_available"}
    assert M.ece(np.empty((0, 2)), []) == M.brier(np.empty((0, 2)), []) == 0


def test_probability_missing_invalid_and_zero_gold_are_not_filled_or_epsilon_clipped():
    summary = M.probability_summary([[1., 0.], None, [float("nan"), 0.], [.2, .8]], [1, 0, 1, 1],
                                    n_classes=2)
    assert summary["n"] == 4 and summary["available_count"] == 2
    assert summary["missing_count"] == summary["invalid_count"] == 1
    assert summary["coverage"] == .5 and summary["zero_gold_count"] == 1
    assert summary["nll"] == {"value": None, "status": "positive_infinity"}
    assert summary["brier"] == pytest.approx((2. + .08) / 2)
    assert M.probability_summary([None], [0], n_classes=2)["nll"]["status"] == "not_available"
    import json
    json.dumps(summary, allow_nan=False)


def test_probability_weights_reuse_brier_ece_and_ignore_zero_weight_infinite_loss():
    p = np.asarray([[1., 0.], [.2, .8]])
    y, weights = [1, 1], [0., 2.]
    summary = M.probability_summary(p, y, n_classes=2, sample_weight=weights)
    assert summary["coverage"] == 1 and summary["zero_gold_count"] == 0
    assert summary["nll"]["value"] == pytest.approx(-np.log(.8))
    assert summary["brier"] == M.brier(p[1:], [1]) == pytest.approx(.08)
    assert summary["ece"] == M.ece(p[1:], [1]) == pytest.approx(.2)
    duplicated = np.repeat(p, [3, 2], axis=0)
    assert M.brier(p, y, sample_weight=[3, 2]) == pytest.approx(M.brier(duplicated, [1] * 5))
    assert M.ece(p, y, sample_weight=[3, 2]) == pytest.approx(M.ece(duplicated, [1] * 5))


@pytest.mark.parametrize("value,status", [(None, "missing"), ([.4], "invalid_shape"),
                                           ([.4, .4], "invalid_simplex"),
                                           ([1.1, -.1], "invalid_simplex"),
                                           ([float("inf"), 0], "nonfinite")])
def test_probability_status_does_not_fabricate_simplex(value, status):
    assert M.probability_row(value, 2) == (status, None)
