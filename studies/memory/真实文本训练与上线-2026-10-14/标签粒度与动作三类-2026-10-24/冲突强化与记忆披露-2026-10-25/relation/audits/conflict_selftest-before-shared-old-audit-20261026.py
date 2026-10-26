"""v2.2 离线反例：测试通过不代表被测数据的严格审计通过。"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch


def run_data_selftest(B) -> dict:
    K, S, D = B.K, B.S, B.DIAG
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    previous_temp = tempfile.tempdir
    tempfile.tempdir = str(K.SELFTEST_DIR)
    checks = []

    def record(name, ok, **extra):
        checks.append({"check": name, "ok": bool(ok), **extra})
        if not ok:
            raise AssertionError(f"{name}: {extra}")

    try:
        record("label_plan", True, detail=S.label_plan_check())
        record("mechanism_mix", len(S.mechanism_label_check()) == 7)
        specs = B.build_specs()
        record("split_families_v22", dict(Counter(s["split"] for s in specs)) ==
               {"train": 24, "dev": 12, "heldout": 24})
        twins = D.semantic_twins(specs)
        record("cross_factor_pair_and_any_side_zero", twins["cross_ok"], cross=twins["cross"])
        record("within_factor_repeats_reported", all(
            block["rows"] == block["unique"] + block["excess_rows"]
            and block["excess_rows"] > 0
            for split in K.SPLITS for block in [twins["within"][split]["pair"]]),
            counts={split: twins["within"][split]["pair"]["excess_rows"] for split in K.SPLITS})
        meaning_twins = D.semantic_twins(specs, meaning=True)
        record("cross_classifier_meaning_pair_and_any_side_zero", meaning_twins["cross_ok"],
               cross=meaning_twins["cross"])
        base_factor = S.factors(S.SETTINGS[0], items_scope=S.SETTINGS[0]["items"][:2],
                                measure=S._measure("at_most", high=3))
        equivalent_surface = dict(base_factor, items_scope=list(reversed(base_factor["items_scope"])),
                                  measure=S._measure("within", 0, 3))
        record("dedup_key_canonicalizes_item_order_and_interval_alias", D.factor_key(
            base_factor, S.SETTINGS[0]) == D.factor_key(equivalent_surface, S.SETTINGS[0])
            and K.sha256_json(base_factor) != K.sha256_json(equivalent_surface))
        leaked = copy.deepcopy(specs)
        first = next(s for s in leaked if s["split"] == "train")
        last = next(s for s in leaked if s["split"] == "heldout")
        last["setting"] = first["setting"]
        last["items"][0]["old"] = first["items"][0]["old"]
        last["items"][0]["new"] = first["items"][0]["new"]
        record("semantic_twin_must_fail_without_text_leak", not D.semantic_twins(leaked)["cross_ok"])
        record("twelve_settings_per_split", all(len(twins["strata"][split]["setting"]) == 12
                                                  for split in K.SPLITS))
        with tempfile.TemporaryDirectory(prefix="data-v22-") as directory:
            root = Path(directory)
            baseline = B._run_pipeline(root / "baseline")
            report = baseline["report"]
            failed_names = [name for name, value in report["checks"].items() if not value["ok"]]
            record("fixture_all_29_checks_pass", not failed_names and len(report["checks"]) == 29,
                   failed=failed_names)
            contrast = report["checks"]["control_matching"]
            negative_equivalent_cases = []
            for mutation in ("both_not_equivalent", "nonzero_leaf_difference"):
                bad_specs = copy.deepcopy(specs)
                bad_spec = bad_specs[0]
                bad = next(i for i in bad_spec["items"] if i["kind"] == "control"
                           and i["expected_coarse"] == "equivalent")
                shadow = next(i for i in bad_spec["items"] if i["pair_id"] == B.dyad_side_id(
                    bad["matched_window_id"], bad["matched_side"]))
                if mutation == "both_not_equivalent":
                    bad["new"]["action"] = "prohibit"
                    shadow["new"]["action"] = "prohibit"
                else:
                    bad["new"]["window"]["year"] = S.SETTINGS_BY_ID[bad_spec["setting"]]["years"][0]
                negative_equivalent_cases.append(any(f["id"] == bad["pair_id"]
                                                    for f in B.control_spec_contrasts(bad_specs)))
            record("explicit_equivalent_contract_positive_and_negative",
                   contrast["equivalent_layer"]["rows"] == contrast["equivalent_layer"]["accepted"] == 120
                   and contrast["non_equivalent_layer"]["rows"] == contrast["non_equivalent_layer"]["accepted"] == 240
                   and contrast["contrast_failure_count"] == 0
                   and all(e["control_label"] == e["shadow_label"] == "equivalent"
                           and e["same_matched_semantic_pair"] and e["leaf_differences"] == 0
                           for e in contrast["equivalent_layer"]["members"])
                   and all(negative_equivalent_cases), negative_cases=negative_equivalent_cases)
            record("routes_v22", all(report["route_shapes"][route] == 1289 for route in K.TRAIN_ROUTES)
                   and report["route_shapes"]["dev"] == 479
                   and report["route_shapes"]["dev-natural"] == 407
                   and report["route_shapes"]["dev-counterfactual"] == 407)
            record("heldout_denominators_v22", report["heldout_denominators"] == {
                "natural": {"rows": 288, "conflict": 96},
                "counterfactual": {"rows": 144, "conflict": 48}})
            rows = baseline["rows"]
            all_rows = [r for split in K.SPLITS for r in rows[split]]
            by_id = {r["id"]: r for r in all_rows}
            controls = [r for r in all_rows if K.row_pool(r) == K.POOL_CONTROL]
            record("all_controls_share_accepted_literal_old", len(controls) == 360 and all(
                K.pair_sides(r["pair"])[0] == K.pair_sides(by_id[r["phenomena"]["matched_pair_id"]]["pair"])[0]
                for r in controls))
            record("shared_old_four_node_declared_groups_pass",
                   report["checks"]["shared_old_only_in_dyads"]["ok"])
            eq_spec = specs[0]
            eq_item = next(i for i in eq_spec["items"] if i["kind"] == "control" and i["recipe"] == "E2")
            eq_record = B._fixture_record(eq_spec)
            eq_entries = {entry["pair_id"]: entry for entry in eq_record["items"]}
            eq_shadow = B.dyad_side_id(eq_item["matched_window_id"], eq_item["matched_side"])
            eq_entries[eq_item["pair_id"]]["new"] = eq_entries[eq_shadow]["new"]
            eq_rows, eq_problems = B.rows_for_family(eq_spec, eq_record)
            eq_by_id = {row["id"]: row for row in eq_rows}
            record("equivalent_wording_overlap_is_separate_from_semantic_contract", not eq_problems
                   and K.content_digest(eq_by_id[eq_item["pair_id"]]) == K.content_digest(eq_by_id[eq_shadow])
                   and contrast["literal_novelty"]["real_author_evidence"] is False
                   and contrast["literal_novelty"]["equivalent"]["rows"] == 120)
            e2p = [r for r in all_rows if r["dyad"] and r["phenomena"]["recipe"] == "D7"]
            record("E_P_never_assigned_conflict", len(e2p) == 120 and all(
                r["dyad"]["role"] == "nonconflict" for r in e2p))
            for tamper, required in (("label", "spec_alignment"),
                                     ("cross_split", "cross_split_content_digest"),
                                     ("dyad", "dyad_integrity"), ("author", "author_isolation"),
                                     ("fixture_flag", "fixture_contract"), ("control", "control_matching")):
                changed = B._run_pipeline(root / tamper, tamper=tamper)["report"]
                detected = not changed["checks"][required]["ok"]
                if tamper == "control":
                    detected = detected and not changed["checks"][required]["operational_ok"]
                record("must_fail_" + tamper, detected, check=required)
            missing = False
            try:
                B._run_pipeline(root / "missing", tamper="missing_author")
            except B.AuthorError:
                missing = True
            record("missing_author_rejected", missing)
            out = baseline["out"]
            clean = copy.deepcopy(rows)

            def audit_mutation(name, mutate, check_name, operational=False):
                changed = copy.deepcopy(clean)
                mutate(changed)
                for split in K.SPLITS:
                    K.write_jsonl_atomic(out / f"{split}.jsonl", changed[split])
                result = B.audit_dataset(processed_dir=out, manifest_path=out / "manifest.json",
                                         specs=specs, mode="fixture")
                entry = result["checks"][check_name]
                record(name, not entry["operational_ok" if operational else "ok"])

            def control_old(changed):
                r = next(r for r in changed["train"] if K.row_pool(r) == K.POOL_CONTROL)
                r["pair"]["old"]["text"] += " "
                r["visible_text"] = K.SR.semantic_pair_text(r["pair"])
            audit_mutation("literal_old_whitespace_mismatch_rejected", control_old,
                           "control_matching", True)

            def unrelated_old(changed):
                shared = [r for r in changed["train"] if K.row_pool(r) == K.POOL_SHARED]
                shared[1]["pair"]["old"]["text"] = shared[0]["pair"]["old"]["text"]
                shared[1]["visible_text"] = K.SR.semantic_pair_text(shared[1]["pair"])
            audit_mutation("unrelated_shared_old_rejected", unrelated_old, "shared_old_only_in_dyads")

            def cf_twin(changed):
                r = next(r for r in changed["train"] if K.row_pool(r) == K.POOL_CONTROL)
                shadow = next(s for s in changed["train"] if s["id"] == r["phenomena"]["matched_pair_id"])
                r["pair"] = copy.deepcopy(shadow["pair"])
                r["pair"]["frame"]["background"] += "另一个中性场景。"
                r["visible_text"] = K.SR.semantic_pair_text(r["pair"])
            audit_mutation("CF_content_twin_not_hidden_by_background", cf_twin, "control_matching", True)
            sample = copy.deepcopy(all_rows[0])
            digest = K.content_digest(sample)
            sample["pair"]["frame"]["background"] = "完全不同的背景。"
            record("content_digest_excludes_background", digest == K.content_digest(sample))
            near_rows = copy.deepcopy(all_rows[:3])
            before = B.near_duplicates(near_rows)
            for r in near_rows:
                r["pair"]["frame"]["background"] = "相同且很长的背景" * 100
                r["visible_text"] = "相同包装" * 100
            record("near_duplicate_uses_old_new_only", before == B.near_duplicates(near_rows))
            spec = next(s for s in specs if s["split"] == "train")
            author_dir = root / "authors"
            payload = B.fixture_author(spec)
            path = author_dir / f"{spec['family_id']}.json"
            valid = B.write_author_record(author_dir, spec, payload, model=K.AUTHOR_PLAN["train"][0],
                                          prompt=B.build_prompt(spec), usage={"calls": 0}, source="author")
            record("resume_verified_record", B.verified_record(path, spec, K.AUTHOR_PLAN["train"]))
            for name, change in (("spec_sha", {"spec_sha256": "old"}),
                                 ("prompt_sha", {"prompt_sha256": "old"}),
                                 ("duplicate_item", {"items": valid["items"] + valid["items"][:1]})):
                K.write_json_atomic(path, dict(valid, **change))
                record("resume_rejects_" + name, not B.verified_record(path, spec, K.AUTHOR_PLAN["train"]))
            K.write_json_atomic(path, dict(valid, control_policy_version="control-matching-v2.2"))
            old_policy_rejected = not B.verified_record(path, spec, K.AUTHOR_PLAN["train"])
            K.write_json_atomic(path, valid)
            modified = copy.deepcopy(spec)
            modified["index"] += 1000
            record("resume_rejects_spec_or_policy_change_even_with_same_prompt",
                   old_policy_rejected and B.build_prompt(modified) == B.build_prompt(spec)
                   and not B.verified_record(path, modified, K.AUTHOR_PLAN["train"]))
            B.write_author_record(author_dir, spec, payload, model=K.AUTHOR_PLAN["train"][0],
                                  prompt=B.build_prompt(spec), usage={}, source="author")
            record("previous_accepted_author_retained", len(list((author_dir / "previous").glob("*.json"))) == 1)
            outcome = {"text": "failed raw", "succeeded": False, "thinking": False, "max_tokens": 8000}
            raw1 = B.write_raw_attempt(author_dir, spec, B.build_prompt(spec), outcome, 1)
            raw2 = B.write_raw_attempt(author_dir, spec, B.build_prompt(spec), dict(outcome, text="second"), 1)
            record("raw_same_process_counter_restart_no_overwrite", raw1 != raw2
                   and K.read_json(raw1)["response_text"] == "failed raw")
            spec_path = root / "one-spec.json"
            K.write_json_atomic(spec_path, spec)
            code = ("import sys; from pathlib import Path; import build_conflict_data as B; "
                    "s=B.K.read_json(sys.argv[1]); B.write_raw_attempt(Path(sys.argv[2]),s,B.build_prompt(s),"
                    "{'text':'child raw','succeeded':False},1)")
            child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
            child_results = [subprocess.run([sys.executable, "-B", "-c", code, str(spec_path), str(author_dir)],
                                            cwd=K.HERE, env=child_env, capture_output=True, timeout=60)
                             for _ in range(2)]
            record("raw_cross_process_restart_no_overwrite", all(r.returncode == 0 for r in child_results)
                   and len(list((author_dir / "raw").glob("*.json"))) == 4)
            with patch.object(B, "ChatChannel", side_effect=AssertionError("must not initialize an LLM")):
                B.require_feasible_controls(specs)
            record("approved_specs_pass_author_preflight_without_call", not B.control_spec_contrasts(specs)
                   and B.POLICY_IDENTITY["control_policy_version"] == K.CONTROL_POLICY_VERSION
                   and B.AUTHOR_PROMPT_VERSION in B.build_prompt(spec))
            changed_specs = copy.deepcopy(specs)
            candidate_spec = next(s for s in changed_specs if s["split"] == "train")
            candidate_item = next(i for i in candidate_spec["items"] if i["kind"] == "control"
                                  and i["expected_coarse"] != "equivalent")
            shadow = next(i for i in candidate_spec["items"] if i["pair_id"] ==
                          B.dyad_side_id(candidate_item["matched_window_id"], candidate_item["matched_side"]))
            candidate_item["new"] = copy.deepcopy(shadow["new"])
            failures = B.control_spec_contrasts(changed_specs)
            record("non_equivalent_leaf_and_CF_meaning_twin_rejected", any(
                failure["id"] == candidate_item["pair_id"] and failure["leaf_differences"] == 0
                and failure["factor_pair_twin"] for failure in failures))
            original = K.PROCESSED_DIR
            K.PROCESSED_DIR = out
            try:
                refused = 0
                for purpose in ("train", "audit"):
                    try:
                        K.route_rows("r2", purpose=purpose, manifest=baseline["manifest"])
                    except PermissionError:
                        refused += 1
                record("fixture_refused_for_real_train_and_score_routes", refused == 2)
            finally:
                K.PROCESSED_DIR = original
        setting = S.SETTINGS[0]
        f = S.factors(setting, items_scope=setting["items"][:2])
        subset = dict(f, items_scope=setting["items"][:1])
        record("items_scope_subset_direction", S.classify_with(f, subset, setting) == "old_entails_new_strict"
               and S.classify_with(subset, f, setting) == "new_entails_old_strict")
        record("empty_items_scope_not_all", S.classify_with(f, dict(f, items_scope=[]), setting)
               == "overlap_incomparable")
        excepted = dict(f, exception={"kind": "except", "item": setting["items"][0]})
        record("exception_removes_item", S.classify_with(f, excepted, setting) == "old_entails_new_strict")
        singleton = S.factors(setting, items_scope=setting["items"][:1])
        empty = dict(singleton, exception={"kind": "except", "item": setting["items"][0]})
        inert = dict(singleton, scope=setting["scopes"][1],
                     exception={"kind": "except", "item": setting["items"][1]})
        global_exception = S.factors(setting, exception={"kind": "except", "item": setting["items"][0]})
        record("empty_applicability_rejected_by_domain_and_text", bool(S.factor_problems(empty, setting))
               and bool(S.check_text_gates(S.render_fixture(empty, setting), empty, setting)))
        record("inert_exception_rejected_by_domain_and_text", bool(S.factor_problems(inert, setting))
               and bool(S.check_text_gates(S.render_fixture(inert, setting), inert, setting)))
        record("active_global_and_subset_exceptions_accepted", not S.factor_problems(global_exception, setting)
               and not S.factor_problems(excepted, setting))
        record("effective_changes_ignore_inert_token_and_interval_alias",
               len(S.effective_factor_differences(singleton, inert, setting)) == 1
               and not S.effective_factor_differences(base_factor, equivalent_surface, setting))
        record("all_spec_domains_and_control_effects_valid", all(
            not S.factor_problems(item[side], S.SETTINGS_BY_ID[spec["setting"]])
            for spec in specs for item in spec["items"] for side in ("old", "new"))
            and all(len(entry["effective_factor_changes"]) >= 2 for entry in
                    B.control_spec_contrasts(specs, include_accepted=True) if entry["coarse"] != "equivalent"))
        payload = B.fixture_author(specs[0])
        partition = S.reference_partition_text(S.SETTINGS_BY_ID[specs[0]["setting"]])
        payload["background"] = payload["background"].replace(partition, "")
        record("missing_reference_partition_rejected", bool(B.validate_author_payload(specs[0], payload)))
        record("all_settings_and_D5_authors_see_disjoint_references", all(
            S.reference_partition_text(S.SETTINGS_BY_ID[spec["setting"]]) in spec["background_hint"]
            and spec["background_hint"] in B.build_prompt(spec) for spec in specs))
        good = S.render_fixture(f, setting)
        record("valid_text_gate", not S.check_text_gates(good, f, setting))
        record("negated_require_rejected", bool(S.check_text_gates(good.replace("必须", "不需要"), f, setting)))
        prohibited = dict(f, action="prohibit")
        bad = S.render_fixture(prohibited, setting).replace("不得", "不得不")
        record("double_negation_rejected", bool(S.check_text_gates(bad, prohibited, setting)))
        e2 = next(i for s in specs for i in s["items"] if i["recipe"] == "E2")
        es = next(s for s in specs if e2 in s["items"])
        esetting = S.SETTINGS_BY_ID[es["setting"]]
        ranged = S.render_fixture(e2["new"], esetting, style="new", surface={"form": "range"})
        bounded = S.render_fixture(e2["new"], esetting, style="new", surface={"form": "bounds"})
        record("E2_actual_range_and_bounds_surfaces", not S.check_text_gates(
            bounded, e2["new"], esetting, surface={"kind": "numeric", "form": "bounds"})
            and bool(S.check_text_gates(ranged, e2["new"], esetting,
                                       surface={"kind": "numeric", "form": "bounds"})))
        B._chat_selftest(record)
    finally:
        tempfile.tempdir = previous_temp
    result = {"selftest": True, "ok": True, "checks": checks,
              "dataset_audit_ok": report["ok"], "control_policy_version": K.CONTROL_POLICY_VERSION}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_score_checks(M, record):
    import numpy as np
    import build_conflict_data as B
    from conflict_diagnostics import semantic_clusters
    from conflict_power import prospective_mdes
    K = M.K
    rows = [
        {"id": "ce-c", "family_id": "a", "coarse_label": "conflict", "label": "disjoint",
         "dyad": {"dyad_id": "ce", "role": "conflict"}},
        {"id": "ce-e", "family_id": "a", "coarse_label": "equivalent", "label": "equivalent",
         "dyad": {"dyad_id": "ce", "role": "nonconflict"}},
        {"id": "ep-e", "family_id": "b", "coarse_label": "equivalent", "label": "equivalent",
         "dyad": {"dyad_id": "ep", "role": "nonconflict"}},
        {"id": "ep-p", "family_id": "b", "coarse_label": "compatible", "label": "old_entails_new_strict",
         "dyad": {"dyad_id": "ep", "role": "nonconflict"}},
    ]
    gold = np.asarray([K.COARSE.index(r["coarse_label"]) for r in rows])
    pred = np.asarray([K.COARSE.index(v) for v in ("conflict", "compatible", "equivalent", "compatible")])
    table = M.dyad_table(rows, gold, pred, np.eye(3)[pred])
    record("exact_dyad_not_binary_conflict_detection", table["both_labels_correct"] == 1
           and table["conflict_dyads_both_labels_correct"] == 0
           and table["conflict_detected_and_other_not_conflict"] == 1
           and table["e_to_p_both_labels_correct"] == 1
           and table["both_labels_correct_rate"] == 0.5
           and table["denominators"]["e_to_p_both_labels_correct"] == 1)
    empty = M.dyad_table([], np.asarray([], dtype=int), np.asarray([], dtype=int), np.zeros((0, 3)))
    record("zero_dyad_rates_null", empty["both_labels_correct_rate"] is None
           and empty["e_to_p_both_labels_correct_rate"] is None)
    partial_refused = False
    try:
        M.dyad_table(rows[:1], gold[:1], pred[:1], np.eye(3)[pred[:1]])
    except ValueError:
        partial_refused = True
    record("incomplete_dyad_refused", partial_refused)
    left = {"predictions": [{"id": r["id"], "pred_coarse": K.COARSE[int(p)]} for r, p in zip(rows, pred)]}
    shuffled = {"predictions": list(reversed(left["predictions"]))}
    same = M.compare_runs(left=left, right=shuffled, rows=rows, rounds=30, specs=[])
    record("paired_predictions_aligned_by_id", same["delta"]["accuracy"] == 0)
    duplicate_refused = False
    try:
        M.compare_runs(left=left, right={"predictions": [left["predictions"][0]] * 4}, rows=rows, rounds=20)
    except ValueError:
        duplicate_refused = True
    record("duplicate_prediction_ids_refused", duplicate_refused)
    absent = M._paired_bootstrap(gold[2:], pred[2:], pred[2:], rows[2:], rounds=20)
    record("absent_class_bootstrap_null_not_nan", absent["metrics"]["conflict_recall"]["delta_ci95"] is None
           and absent["metrics"]["conflict_recall"]["valid_rounds"] == 0)
    duplicate_specs = [
        {"setting": "s", "family_id": "a", "items": [{"pair_id": "ce-c", "old": {"v": 1}, "new": {"v": 2}},
                                                       {"pair_id": "ce-e", "old": {"v": 3}, "new": {"v": 4}}]},
        {"setting": "s", "family_id": "b", "items": [{"pair_id": "ep-e", "old": {"v": 2}, "new": {"v": 5}},
                                                       {"pair_id": "ep-p", "old": {"v": 6}, "new": {"v": 7}}]},
    ]
    linked = semantic_clusters(rows, duplicate_specs)
    record("semantic_side_link_merges_families", linked["clusters"] == 1
           and linked["cross_family_clusters"] == 1 and linked["spec_mapped_rows"] == 4)
    paired = M.compare_runs(left=left, right=left, rows=rows, rounds=30, specs=duplicate_specs)
    record("family_and_semantic_bootstrap_both_exposed", paired["bootstrap_family"]["clusters"] == 2
           and paired["bootstrap_semantic_cluster"]["clusters"] == 1
           and not paired["bootstrap_semantic_cluster"]["independent_rows_assumed"])
    power = prospective_mdes(B.build_specs())
    scenarios = power["analyses"]["family"]["scenarios"]
    independent = next(s for s in scenarios if s["paired_discordance_q"] == 0.5
                       and s["paired_difference_icc"] == 0.0)
    correlated = next(s for s in scenarios if s["paired_discordance_q"] == 0.5
                      and s["paired_difference_icc"] == 1.0)
    record("prospective_MDES_96_conflicts_24_families", power["conflict_n"] == 96
           and power["family_n"] == 24 and not power["reads_predictions"]
           and abs(independent["mdes_pp_normal_approx"] - 2.802 * np.sqrt(0.5 / 96) * 100) < 1e-10
           and correlated["effective_n_approx"] == 24
           and independent["mdes_pp_normal_approx"] > 10)
