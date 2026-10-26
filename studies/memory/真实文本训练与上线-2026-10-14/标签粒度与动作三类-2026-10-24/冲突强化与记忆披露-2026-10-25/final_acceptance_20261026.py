from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E / "relation"))
import conflict_common as K
import score_conflict as S
from run_six_fits_20261026 import verify_run

os.environ["PYTHONPATH"] = str(K.REPO / "src")

R = E.parents[3]
FILES = {}


def track(path):
    path = Path(path).resolve()
    assert path.is_relative_to(R.resolve()) and path.is_file(), path
    FILES[path.relative_to(R).as_posix()] = {"sha256": K.sha256_file(path), "bytes": path.stat().st_size}
    return path


def load(path):
    return K.read_json(track(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    fits = {(a, s): verify_run(a, s) for s in K.SEEDS for a in K.TRAIN_ROUTES}
    for seed in K.SEEDS:
        a, b = fits["r2", seed], fits["r3", seed]
        assert a["init"] == b["init"] and a["data"]["train"] == b["data"]["train"]
        assert a["plan"]["sha256"] == b["plan"]["sha256"] and a["budget"] == b["budget"]
    counts = {"new-heldout": 432, "new-heldout-natural": 288, "new-heldout-counterfactual": 144, "old-heldout": 150, "composite": 48}
    aliases = []
    for seed in K.SEEDS:
        for arm in K.SCORE_ROUTES:
            alias = f"{arm}-seed{seed}"
            entry = S.resolve_run(None, str(K.PARENT_EXP / "runs" / f"direct3-seed{seed}")) if arm == "r0" else S.resolve_run(alias, None)
            lock_path = track(K.RUNS_DIR / f"final-lock-{alias}-terminal.json")
            lock = K.load_lock(lock_path)
            assert lock["run_lock_sha256"] == K.sha256_file(track(entry["directory"] / "run-lock.json"))
            for checkpoint in ("best-dev", "terminal"):
                for path in (entry["directory"] / "checkpoints" / checkpoint).rglob("*"):
                    if path.is_file(): track(path)
            actual = K.checkpoint_file_shas(entry["directory"] / "checkpoints/terminal")
            for route, n in counts.items():
                assert lock["checkpoint_files_sha256"][route + ":terminal"] == actual
                K.check_lock(lock, route=route, run_dir=entry["directory"], checkpoint="terminal")
                score = load(K.HERE / "evaluations" / alias / f"{route}.json")
                assert score["alias"] == alias and score["n"] == n
                assert len(score["predictions"]) == len({p["id"] for p in score["predictions"]}) == n
                assert score["reload"]["meta"]["step"] == 300 and score["checkpoint"] == "terminal-300"
                assert score["route_fingerprint"] == K.route_fingerprint_any(route)[0]
                assert score["final_lock_sha256"] == K.sha256_file(lock_path)
            fresh = load(K.HERE / "evaluations" / alias / "independent-reload.json")
            assert fresh["ok"] and fresh["n"] == 432 and fresh["argmax_mismatches"] == fresh["max_probability_delta"] == 0
            path = E / "baseline-regression" / f"direct3-seed{seed}-terminal-ocnli-diagnostic.json" if arm == "r0" else K.RUNS_DIR / alias / "scores/ocnli-diagnostic-terminal.json"
            public = load(path)
            accounting = public["accounting"]
            assert accounting["scored_with_gold"] == len(public["predictions"]) == 1502
            assert accounting["clusters_with_gold"] == 1099 and accounting["gold_counts"]["contradiction"] == 452
            assert accounting["sealed_exposed_to_model"] == 0 and public["not_three_class_NLI_score"]
            assert not public["summary"]["threshold_fitted"]
            aliases.append(alias)
    summary = load(K.HERE / "evaluations/summary-20261026.json")
    assert len(summary["inputs"]) == 59
    for relative, expected in summary["inputs"].items():
        path = track(K.HERE / relative)
        assert K.sha256_file(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
    comparisons = load(K.HERE / "evaluations/fixed-comparisons-20261026.json")
    assert len(comparisons["comparisons"]) == 20 and not any(comparisons["both_seeds_meet_target"].values())
    selected = next(c for c in comparisons["comparisons"] if (c["seed"], c["route"], c["left_arm"], c["right_arm"]) == (42, "new-heldout-natural", "r0", "r1"))
    assert [round(x, 4) for x in selected["bootstrap_semantic_cluster"]["metrics"]["macro_f1"]["delta_ci95"]] == [.0177, .0814]
    assert summary["target_feasibility"]["42"]["target_feasible"] is False
    control_dir = E / "disclosure/pilot_v4/missing_controls_20261026"
    controls = load(control_dir / "summary.json")
    assert controls["status"] == "complete" and controls["collection"]["execute"]["succeeded"] == controls["ledger"]["finished"] == 24
    assert not controls["ledger"]["pending_unknown"] and controls["usage"]["all"]["usage_unknown"] == 0
    assert controls["usage"]["all"]["provider_total_tokens"] == 35700
    for kind, numerator in (("metadata_only", 2), ("no_citation", 11)):
        row = controls["analysis"]["controls"][kind]
        assert (row["n_calls"], row["n_evaluable"], row["action_value_match"], row["layered_success"]) == (12, 11, numerator, 2)
        assert row["citation_empty_all_calls"] == 12 and not row["violating_citation_tasks"]
    assert len(list(control_dir.glob("*.outcome.json"))) == 24
    for path in control_dir.rglob("*"):
        if path.is_file(): track(path)
    command = subprocess.run([sys.executable, "-B", "analyze_20261025.py", "verify"], cwd=E / "disclosure/pilot_v4/diagnostics_20261025", capture_output=True, text=True, encoding="utf-8", check=True)
    memory = json.loads(command.stdout)
    assert memory["ok"] and memory["files"] == 168 and memory["main"]["input_files"] == 179 and memory["main"]["result_files"] == 450
    for relative in ("out/frozen_manifest.json", "out/final_results_manifest_v1.json", "diagnostics_20261025/diagnostics_manifest.json"):
        track(E / "disclosure/pilot_v4" / relative)
    tests = ("build_conflict_data.py", "train_conflict_arms.py", "score_conflict.py", "compare_fixed_results_20261026.py", "extract_fixed_results_20261026.py", "relation_call_costs_20261026.py", "score_public_regression_20261026.py")
    for name in tests:
        track(K.HERE / name)
        assert track(K.AUDITS_DIR / f"final-selftest-{name}-20261026.log").stat().st_size > 0
    for path in (Path(__file__), K.HERE / "close_semantic_review_20261026.py", E / "disclosure/missing_controls_20261026.py", K.AUDITS_DIR / "independent-point-metric-check-20261026.json", K.AUDITS_DIR / "semantic-review-final-20261026.json", K.AUDITS_DIR / "all-fits-and-final-locks-20261026.json", K.AUDITS_DIR / "call-costs-20261026.json"):
        track(path)
    table = K.HERE / "evaluations/结果表-2026-10-26.md"
    assert "family bootstrap 95% CI" in table.read_text(encoding="utf-8")
    review_dir = R / "00-汇报与面谈"
    reviews = [p for p in review_dir.glob("*2026-10-26.md") if "冲突强化与记忆披露" in p.name]
    assert len(reviews) == 2
    for path in reviews: track(path)
    report = E / "报告-2026-10-26.md"
    content = report.read_text(encoding="utf-8", errors="strict")
    assert "本轮研究已完成" in content and "[.0177,.0814]" in content and "除R1-seed13外" in content
    docs = (report, E / "运行记录-2026-10-25.md", E / "设计-2026-10-25.md", E / "disclosure/阶段结果-记忆主对照-2026-10-25.md", table, R / "README.md")
    links = 0
    for path in docs:
        track(path)
        text = path.read_text(encoding="utf-8", errors="strict")
        if path.name == "README.md":
            lines = [line for line in text.splitlines() if E.name in line]
            assert len(lines) == 1
            text = lines[0]
        if path.stat().st_size > 20480 and path.name != "README.md": assert "导航" in text[:1800]
        for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if link.startswith(("http://", "https://", "#")): continue
            assert (path.parent / link.split("#", 1)[0]).resolve().exists(), (path, link)
            links += 1
    status = subprocess.run(["git", "-C", str(K.REPO), "status", "--porcelain"], capture_output=True, text=True, check=True)
    head = subprocess.run(["git", "-C", str(K.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert not status.stdout.strip() and head == "86a982ff"
    result = {"schema": "research-goal-acceptance/1", "verified_at": datetime.now().astimezone().isoformat(), "status": "verified", "cuda_fits": 6, "terminal_models": aliases, "scores_per_model": counts, "independent_reload_max_delta": 0, "public_diagnostic_rows": 1502, "public_conflicts": 452, "public_clusters": 1099, "sealed_model_exposure": 0, "paired_comparisons": 20, "source_inputs_sha_verified": 59, "memory_frozen": memory, "real_control_calls": {"empty": 12, "metadata_only": 12, "no_citation": 12}, "memory_receipts": {"calls": 364, "tokens": 465894}, "relation_receipt_lower_bound": {"calls": 189, "tokens": 726598}, "research_selftest_clis_passed": len(tests), "additional_controls_selftest_passed": 17, "review": {"requested_model": "gpt-6-astra", "effort": "high", "session": "01a174ec-fdf1-78f3-bba9-0e259a3de1f6", "backend_attested": False, "high_medium_findings": 0, "low_document_findings_fixed": 2, "coverage_excludes_extra_independent_gate_and_raw_relation_cost_recalculation": True}, "markdown_links_checked": links, "production_repository": {"head": head, "status_clean": True, "deployment_performed": False}, "not_run": [{"item": "production full pytest / Web smoke / frozen retrieval gate", "reason": "No production deliverable code, corpus or deployment changed; research runtime/selftests/actual inference/recomputation were validated instead."}], "limitations": ["AI labels are not human gold", "two seeds and clustered toy tasks do not prove universal transfer or non-inferiority", "real controls are post-hoc, not new preregistered blind data", "projection does not fully check condition/exception", "stale rejection is not auto-refresh", "receipt costs exclude unknown development/coordination/review tokens"], "files": FILES}
    final = E / "验收-2026-10-26.json"
    if args.seal:
        assert not final.exists(), "Existing acceptance must be preserved; run without --seal to reverify"
        K.write_json_atomic(final, result)
    elif final.exists():
        frozen = K.read_json(final)
        assert frozen["files"] == FILES, "Sealed file footprint changed"
    print(json.dumps({"status": "verified", "cuda_fits": 6, "terminal_models": len(aliases), "paired_comparisons": 20, "source_inputs": 59, "files_hashed": len(FILES), "links_checked": links, "sealed": final.exists(), "acceptance_sha256": K.sha256_file(final) if final.exists() else None}))


if __name__ == "__main__":
    main()
