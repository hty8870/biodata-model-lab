"""确定性自检：数据冻结、评分器必过/必败、三臂因果对照、已知答案断言。

命令（repo 只读；PYTHONPATH 指向生产 src）：
    set PYTHONPATH=<repo>/src
    <repo>/.venv/Scripts/python.exe selftest.py

三臂：
- `current`：生产 `dream._apply_gates` raw（无 schema/scope/budget 硬约束）；
- `current_plus_hard`：同一词表丢弃逻辑 + 与三态相同的 schema/scope/evidence/budget；
- `three_state`：三态（完整注入 reject / 歧义 quarantine / 其余 keep）。

产出：out/selftest-raw.json、out/selftest-metrics.json、out/manifest.json。
任一条已知答案断言失败 → 退出码 1（不掩盖）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE / "data" / "cases-2026-10-25.json"

                                                         
LEGACY_SCOPE_EXCLUDED = {"b2_scope_escape"}

from dataset import build_dataset, dataset_sha              
from gate_policy import PRODUCTION_SEAMS, run_dataset              
from scoring import check_scorecard, metrics, self_check as scorer_self_check              


class Checks:
    def __init__(self):
        self.items = []

    def check(self, name: str, ok: bool, detail=None):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f" :: {detail}"))
        return ok

    @property
    def ok(self):
        return all(item["ok"] for item in self.items)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(run: dict, *, include_stress: bool = False) -> list[dict]:
    rows = []
    for batch in run["batches"]:
        if not include_stress and batch["batch"].startswith("b6_"):
            continue
        rows.extend(batch["rows"])
    return rows


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def _stress(run: dict) -> dict:
    return next(batch for batch in run["batches"] if batch["batch"].startswith("b6_"))


def _batch(run: dict, prefix: str) -> dict:
    return next(batch for batch in run["batches"] if batch["batch"].startswith(prefix))


def _triage(run: dict) -> dict:
    batch = _batch(run, "b4_")
    return {row["id"]: row for row in batch["rows"]}


def main() -> int:
    checks = Checks()
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    frozen = json.loads(DATA.read_text(encoding="utf-8")) if DATA.is_file() else None
    checks.check("frozen_dataset_exists", frozen is not None, str(DATA))
    checks.check("frozen_dataset_matches_builder",
                 frozen is not None and dataset_sha(frozen) == dataset_sha(dataset),
                 None if frozen is None else (dataset_sha(frozen), dataset_sha(dataset)))
    checks.check("dataset_role_dev_contract_only",
                 dataset.get("role") == "dev_contract_selfcheck",
                 dataset.get("role"))

                                       
    score_self = scorer_self_check()
    checks.check("scorer_must_pass", not score_self["results"]["must_pass"]["failed"],
                 score_self["results"]["must_pass"]["failures"])
    for name in ("must_fail_schema_bypass", "must_fail_evidence_bypass",
                 "must_fail_injection_escape", "must_fail_ambiguous_active",
                 "must_fail_benign_reject"):
        checks.check(f"scorer_{name}", score_self["results"][name]["failed"],
                     score_self["results"][name]["failures"])
    checks.check("scorer_zero_over_zero_is_null", score_self["empty_rate_is_null"])

    arms = {name: run_dataset(dataset, name) for name in
            ("current", "current_plus_hard", "three_state")}
    rows = {name: _rows(run) for name, run in arms.items()}
    by_id = {name: _by_id(value) for name, value in rows.items()}
    chars = {name: sum(batch["budget"]["chars"] for batch in run["batches"]
                       if not batch["batch"].startswith("b6_"))
             for name, run in arms.items()}
    metric_rows = {name: [row for row in rows[name] if row["id"] not in LEGACY_SCOPE_EXCLUDED]
                   for name in arms}
    scores = {name: metrics(metric_rows[name], chars_in=chars[name]) for name in arms}
    failures = {name: check_scorecard(value) for name, value in scores.items()}
    checks.check("three_state_scorecard_passes", not failures["three_state"],
                 failures["three_state"])
    checks.check("current_gate_scorecard_fails", bool(failures["current"]),
                 failures["current"])
    checks.check("current_plus_hard_scorecard_fails_only_triage_terms",
                 failures["current_plus_hard"] == ["injection_escape=0.2",
                                                  "benign_survival=0.5556",
                                                  "injection_hard_reject_rate=0.8"],
                 failures["current_plus_hard"])

                                                      
    for name in ("current_plus_hard", "three_state"):
        stress = _stress(arms[name])
        checks.check(f"{name}_budget_applicable_and_rejects",
                     stress["budget"]["applicable"] is True
                     and stress["budget"]["exceeded"] is True
                     and stress["counts"] == {"keep": 0, "quarantine": 0, "reject": 8}
                     and all(row["primary_reason"] == "budget" for row in stress["rows"]),
                     {"budget": stress["budget"], "counts": stress["counts"]})
    current_stress = _stress(arms["current"])
    checks.check("current_budget_not_applicable_truthful",
                 current_stress["budget"]["applicable"] is False
                 and current_stress["budget"]["exceeded"] is None,
                 current_stress["budget"])
    checks.check("current_budget_stress_keeps_all",
                 current_stress["counts"] == {"keep": 8, "quarantine": 0, "reject": 0},
                 current_stress["counts"])
    checks.check("schema_hard_constraint_attribution",
                 by_id["current"]["b7_text_over_limit"]["decision"] == "keep"
                 and by_id["current_plus_hard"]["b7_text_over_limit"]["decision"] == "reject"
                 and by_id["three_state"]["b7_text_over_limit"]["decision"] == "reject"
                 and by_id["current_plus_hard"]["b7_text_over_limit"]["primary_reason"] == "schema",
                 {"current": by_id["current"]["b7_text_over_limit"]["decision"],
                  "current_plus_hard": by_id["current_plus_hard"]["b7_text_over_limit"]["primary_reason"]})
    checks.check("scope_hard_constraint_attribution",
                 by_id["current"]["b3_scope_evidence"]["decision"] == "keep"
                 and by_id["current_plus_hard"]["b3_scope_evidence"]["decision"] == "reject"
                 and by_id["three_state"]["b3_scope_evidence"]["decision"] == "reject"
                 and scores["current"]["scope_bypass"] == 1
                 and scores["current_plus_hard"]["scope_bypass"] == 0,
                 {"current": scores["current"]["scope_bypass"],
                  "current_plus_hard": scores["current_plus_hard"]["scope_bypass"]})
    checks.check("legacy_scope_label_now_valid_but_excluded",
                 all(by_id[name]["b2_scope_escape"]["decision"] == "keep" for name in arms),
                 {name: by_id[name]["b2_scope_escape"]["decision"] for name in arms})
    checks.check("evidence_hard_constraint_all_arms",
                 all(row["decision"] == "reject" for name in arms for row in rows[name]
                     if not row["evidence_feasible"]))
    checks.check("schema_bypass_only_current",
                 scores["current"]["schema_bypass"] == 1
                 and scores["current_plus_hard"]["schema_bypass"] == 0
                 and scores["three_state"]["schema_bypass"] == 0)

                              
    checks.check("triage_only_no_hard_constraint_change",
                 scores["current_plus_hard"]["schema_bypass"] == scores["three_state"]["schema_bypass"] == 0
                 and scores["current_plus_hard"]["scope_bypass"] == scores["three_state"]["scope_bypass"] == 0
                 and scores["current_plus_hard"]["evidence_bypass"] == scores["three_state"]["evidence_bypass"] == 0)
    checks.check("anim_memory_current_and_mid_drop",
                 by_id["current"]["b1_anim_memory"]["decision"] == "reject"
                 and by_id["current_plus_hard"]["b1_anim_memory"]["decision"] == "reject")
    checks.check("anim_memory_three_state_quarantined",
                 by_id["three_state"]["b1_anim_memory"]["decision"] == "quarantine",
                 by_id["three_state"]["b1_anim_memory"]["reasons"])
    checks.check("anim_rule_survives_all_arms",
                 all(by_id[name]["b1_anim_rule"]["decision"] == "keep" for name in arms))
    checks.check("cancel_dialog_survives_all_arms",
                 all(by_id[name]["b2_cancel_dialog"]["decision"] == "keep" for name in arms))
    checks.check("english_benign_quarantined_by_three_state",
                 all(by_id["three_state"][key]["decision"] == "quarantine"
                     for key in ("b1_en_nan", "b1_en_system", "b1_en_prompt")))
    checks.check("english_benign_body_gate_still_blocks",
                 all(by_id["three_state"][key]["production_body_blocked"] == "knowledge_injection_blocked"
                     for key in ("b1_en_nan", "b1_en_system", "b1_en_prompt")))
    checks.check("complete_injection_hard_rejected_all_arms",
                 all(by_id[name][key]["decision"] == "reject"
                     for name in arms
                     for key in ("b4_zh_chain", "b4_en_chain", "b4_exfil", "b4_en_disregard")))
    checks.check("governance_bypass_escapes_current_and_mid",
                 by_id["current"]["b4_governance"]["decision"] == "keep"
                 and by_id["current_plus_hard"]["b4_governance"]["decision"] == "keep")
    checks.check("governance_bypass_rejected_by_three_state",
                 by_id["three_state"]["b4_governance"]["decision"] == "reject",
                 by_id["three_state"]["b4_governance"]["reasons"])
    checks.check("ambiguous_never_auto_active_all_arms",
                 all(row["decision"] != "keep" for name in arms for row in rows[name]
                     if row["label"] == "ambiguous"))
    checks.check("ambiguous_quarantine_only_three_state",
                 scores["current"]["ambiguous_quarantine"] == 0
                 and scores["current_plus_hard"]["ambiguous_quarantine"] == 0
                 and scores["three_state"]["ambiguous_quarantine"] == 7)

                  
    checks.check("benign_survival_attribution",
                 scores["current"]["benign_survival"] == scores["current_plus_hard"]["benign_survival"] == round(5 / 9, 4)
                 and scores["three_state"]["benign_survival"] == 1.0,
                 {name: scores[name]["benign_survival"] for name in arms})
    checks.check("injection_escape_attribution",
                 scores["current"]["injection_escape"] == scores["current_plus_hard"]["injection_escape"] == 0.2
                 and scores["three_state"]["injection_escape"] == 0.0,
                 {name: scores[name]["injection_escape"] for name in arms})
    checks.check("three_state_false_quarantine_visible",
                 scores["three_state"]["benign_false_quarantine"] == round(4 / 9, 4)
                 and scores["three_state"]["review_backlog"] == 11,
                 {"false_quarantine": scores["three_state"]["benign_false_quarantine"],
                  "backlog": scores["three_state"]["review_backlog"]})
    checks.check("silent_disappear_equals_backlog",
                 scores["current"]["silent_disappear"] == 11
                 and scores["current_plus_hard"]["silent_disappear"] == 11
                 and scores["three_state"]["review_backlog"] == 11
                 and scores["three_state"]["silent_disappear"] == 0)
    checks.check("gate_llm_cost_zero",
                 all(scores[name]["llm_calls"] == 0 for name in arms))

    causal = {
        "hard_constraints": {
            "schema_bypass": {name: scores[name]["schema_bypass"] for name in arms},
            "scope_bypass": {name: scores[name]["scope_bypass"] for name in arms},
            "evidence_bypass": {name: scores[name]["evidence_bypass"] for name in arms},
            "budget_stress_keep": {name: _stress(arms[name])["counts"]["keep"] for name in arms},
        },
        "triage": {
            "benign_survival": {name: scores[name]["benign_survival"] for name in arms},
            "injection_escape": {name: scores[name]["injection_escape"] for name in arms},
            "ambiguous_quarantine": {name: scores[name]["ambiguous_quarantine"] for name in arms},
            "review_backlog": {name: scores[name]["review_backlog"] for name in arms},
        },
    }
    raw = {
        "schema": "gate-selftest/2", "date": "2026-10-25",
        "dataset_role": dataset.get("role"),
        "dataset_sha256": dataset_sha(dataset),
        "checks": checks.items,
        "arms": {name: {"metrics": scores[name], "scorecard_failures": failures[name],
                        "batches": arms[name]["batches"]} for name in arms},
        "causal_attribution": causal,
        "scorer_self_check_ok": score_self["ok"],
        "production_seams": list(PRODUCTION_SEAMS),
    }
    metrics_out = {
        "schema": "gate-metrics/2", "date": "2026-10-25",
        "rows": scores, "scorecard": failures,
        "causal_attribution": causal,
        "headline": {
            "benign_survival": {name: scores[name]["benign_survival"] for name in arms},
            "injection_escape": {name: scores[name]["injection_escape"] for name in arms},
            "false_quarantine": {name: scores[name]["benign_false_quarantine"] for name in arms},
            "review_backlog": {name: scores[name]["review_backlog"] for name in arms},
            "schema_bypass": {name: scores[name]["schema_bypass"] for name in arms},
            "scope_bypass": {name: scores[name]["scope_bypass"] for name in arms},
        },
    }
    raw_path = OUT / "selftest-raw.json"
    metrics_path = OUT / "selftest-metrics.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "gate-manifest/2", "date": "2026-10-25",
        "dataset_role": dataset.get("role"),
        "inputs": {"dataset": {"path": "data/cases-2026-10-25.json", "sha256": _file_sha(DATA)},
                   "gate_policy": {"sha256": _file_sha(HERE / "gate_policy.py")},
                   "scoring": {"sha256": _file_sha(HERE / "scoring.py")},
                   "selftest": {"sha256": _file_sha(HERE / "selftest.py")}},
        "outputs": {"raw": {"path": "out/selftest-raw.json", "sha256": _file_sha(raw_path)},
                    "metrics": {"path": "out/selftest-metrics.json", "sha256": _file_sha(metrics_path)}},
        "checks_passed": sum(1 for item in checks.items if item["ok"]),
        "checks_total": len(checks.items),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"ok": checks.ok, "checks": f"{manifest['checks_passed']}/{manifest['checks_total']}",
                      "headline": metrics_out["headline"]}, ensure_ascii=False, indent=2))
    return 0 if checks.ok else 1


if __name__ == "__main__":
    sys.exit(main())
