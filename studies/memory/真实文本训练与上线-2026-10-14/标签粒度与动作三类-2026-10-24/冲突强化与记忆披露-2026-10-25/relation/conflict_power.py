"""只依赖锁定规格和显式假设的 prospective 检出限；不读取预测或拟合模型。"""
from __future__ import annotations

import math
from collections import Counter

import conflict_common as K
from conflict_diagnostics import semantic_clusters


def prospective_mdes(specs) -> dict:
    rows = [{"id": item["pair_id"], "family_id": spec["family_id"],
             "coarse_label": item["expected_coarse"]}
            for spec in specs if spec["split"] == "heldout"
            for item in spec["items"] if item["pool"] in (K.POOL_SHARED, K.POOL_CONTROL)]
    conflict = [index for index, row in enumerate(rows) if row["coarse_label"] == "conflict"]
    n = len(conflict)
    if not n:
        raise ValueError("prospective recall MDES requires conflict rows")
    dependencies = semantic_clusters(rows, specs)
    groupings = {
        "family": [row["family_id"] for row in rows],
        "family_plus_semantic_components": dependencies["ids"],
    }
    analysis = {}
    for grouping, cluster_ids in groupings.items():
        sizes = Counter(cluster_ids[index] for index in conflict)
        size_weighted = sum(size * size for size in sizes.values()) / n
        scenarios = []
        for discordance in (0.1, 0.25, 0.5, 1.0):
            for icc in (0.0, 0.1, 0.3, 0.5, 1.0):
                design_effect = 1 + (size_weighted - 1) * icc
                se_null = math.sqrt(discordance * design_effect / n)
                mdes = 2.802 * se_null * 100
                scenarios.append({"paired_discordance_q": discordance, "paired_difference_icc": icc,
                                  "design_effect": design_effect, "effective_n_approx": n / design_effect,
                                  "mdes_pp_normal_approx": mdes,
                                  "ten_pp_exceeds_approx_mdes": 10 >= mdes})
        analysis[grouping] = {"clusters_with_conflict": len(sizes),
                              "conflict_cluster_sizes": sorted(sizes.values()),
                              "scenarios": scenarios}
    return {
        "schema": "conflict-prospective-mdes-v2.2", "prospective": True,
        "reads_predictions": False, "natural_rows": len(rows), "conflict_n": n,
        "family_n": len({row["family_id"] for row in rows}),
        "spec_sha256": K.sha256_json(specs), "target_pp": 10,
        "formula": "100*(1.96+0.842)*sqrt(q*(1+(sum(m_j^2)/N-1)*ICC)/N)",
        "alpha_two_sided": 0.05, "power_target": 0.8,
        "assumptions": ["q is the prospective probability of paired correctness disagreement, not an estimated baseline accuracy",
                        "ICC is correlation of paired correctness differences within a declared cluster",
                        "normal approximation ignores finite-cluster corrections and boundary/ceiling effects; not a power guarantee",
                        "24 families and repeated semantics are not 96 independent experimental units",
                        "the approved equivalent-control stratum has shared semantics and is not independent semantic novelty; collection readiness requires the current protocol audit"],
        "semantic_dependencies": {key: value for key, value in dependencies.items() if key != "ids"},
        "analyses": analysis,
        "interpretation": "10pp is not guaranteed detectable; a failure to reach 10pp cannot be interpreted as evidence of no effect",
    }
