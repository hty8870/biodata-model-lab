from pathlib import Path
import json
from research.memory_relations import head_design as H

base = Path(__file__).resolve().parent
study = base / "formal-study"
context = H.load_prepare_context(study / "prepare")
protocol = context["protocol"]
payload = H._read_json(study / "train-dev" / "runs.json")
manifest = H._read_json(study / "train-dev" / "manifest.json")
freeze = H._read_json(study / "freeze" / "freeze.json")
H.verify_source_metadata(payload)
budget = H.validate_training_budget(payload, manifest, protocol)
H.verify_selections(payload["runs"], protocol)
rule = H.evaluate_architecture_rule(payload["runs"], protocol)
assert H._canonical(rule) == H._canonical(freeze["architecture_rule"])
views = []
for seed in protocol["training"]["seeds"]:
    for arm in H.TRAINABLE_ARMS:
        run = payload["runs"][f"{arm}-{seed}"]
        for view in ("coarse", "fine"):
            selection = run.get(view)
            if selection is None:
                continue
            row = H._history_at(run["history"], selection["step"])
            views.append({"seed": seed, "arm": arm, "view": view, "step": row["step"],
                          "dev_coarse_nll": row["dev_coarse_nll"], "dev_fine_nll": row["dev_fine_nll"],
                          "coarse_macro_f1": row["coarse_macro_f1"], "fine_macro_f1": row["fine_macro_f1"],
                          "coarse_accuracy": row["coarse_accuracy"], "fine_accuracy": row["fine_accuracy"],
                          "coarse_recall": dict(zip(protocol["semantics"]["parents"], row["coarse_recall"])),
                          "fine_recall": None if row["fine_recall"] is None else dict(zip(protocol["semantics"]["leaves"], row["fine_recall"])),
                          "coarse_support": dict(zip(protocol["semantics"]["parents"], row["coarse_support"])),
                          "fine_support": dict(zip(protocol["semantics"]["leaves"], row["fine_support"])),
                          "models": run["models"][view], "fine_diagnostic_only": arm == "B"})
access_id = H.final_access_identity(context)
ledger = H._read_attempts(H.study_ledger_directory())
attempts = [row for row in ledger if row.get("record") == "attempt" and row.get("final_access_id") == access_id]
assert not attempts
assert not (study / "finalize").exists()
actuals = {key: {"base": {field: run["base"][field] for field in ("updates","executed_updates","eval_calls","rng","initial_q","wall_seconds","cpu_seconds")},
                 "extension": {field: run["extended"][field] for field in ("start_step","updates","executed_updates","eval_calls","rng","wall_seconds","cpu_seconds")},
                 "terminal": run["terminal"], "history_grid": [row["step"] for row in run["history"]]} for key, run in payload["runs"].items()}
results = {"status": "dev-only-freeze", "source_bundle": H.source_bundle_info(), "protocol_sha256": H.load_protocol()[1],
           "source_test_sha256": H.sha256_file(Path(H.__file__).with_name("test_head_design.py")),
           "freeze_sha256": H.sha256_file(study / "freeze" / "freeze.json"),
           "freeze_manifest_sha256": H.sha256_file(study / "freeze" / "manifest.json"),
           "train_manifest_sha256": H.sha256_file(study / "train-dev" / "manifest.json"),
           "runs_sha256": H.sha256_file(study / "train-dev" / "runs.json"),
           "budget": budget, "extension": payload["extension"], "architecture_rule": rule, "selected_views": views,
           "actual_runs": actuals, "checkpoint_validation": freeze["checkpoint_validation"],
           "formal_pool_id": access_id, "formal_attempt_count": len(attempts), "formal_final_scored": False,
           "pool_stats": context["manifest"]["pools"], "denominators": context["manifest"]["denominators"],
           "novelty": context["manifest"]["novelty"],
           "cost": {"train_wall_seconds": manifest["duration_seconds"], "train_cpu_seconds": manifest["cpu_seconds"],
                    "freeze_wall_seconds": H._read_json(study / "freeze" / "manifest.json")["duration_seconds"],
                    "external_inference_calls": 0, "device": "cpu", "threads": 1, "hardware_energy_cost": None}}
H._write_json(base / "formal-dev-results.json", results)
print("seed arm view step coarseNLL fineNLL coarseF1 fineF1 Erec Crec eqrec oldrec newrec")
for row in views:
    fine = row["fine_recall"] or {}
    values = [row["dev_coarse_nll"], row["dev_fine_nll"], row["coarse_macro_f1"], row["fine_macro_f1"],
              row["coarse_recall"]["E"], row["coarse_recall"]["C"], fine.get("equivalent"), fine.get("old_entails_new_strict"), fine.get("new_entails_old_strict")]
    print(row["seed"], row["arm"], row["view"], row["step"], " ".join("-" if value is None else f"{value:.9f}" for value in values))
print(json.dumps({"rule": rule, "budget": budget, "cost": results["cost"], "dev_stats": results["pool_stats"]["dev"]["stats"], "formal_attempt_count": 0}, indent=2))
