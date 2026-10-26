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
