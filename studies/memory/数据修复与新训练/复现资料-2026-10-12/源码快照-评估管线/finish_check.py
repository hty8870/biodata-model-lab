"""收尾只读验证与输出清单，不加载模型或重新预测。"""
import ast
import json
from pathlib import Path
import subprocess

import binding as B

binding = B.verify()
reference_modules = B.read_json(B.REF / "ref-ready-manifest.json")["version_binding"]["modules"]
for path, key in (("research/memory_relations/features.py", "features_sha256"),
                  ("research/memory_relations/schema.py", "schema_sha256"),
                  ("src/dataset_recommender/agent/memory_relations.py", "production_sha256")):
    assert B.sha(B.ROOT / path) == reference_modules[key], key
started = B.read_json(B.BASE / "execution-started.json")
assert started["runner_sha256"] == B.sha(B.BASE / "formal_evaluation.py")
assert started["binding_script_sha256"] == B.sha(B.BASE / "binding.py")
for path in B.BASE.glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
prior_hashes = B.read_json(B.BASE / "reports/output-hashes.json")
for name, expected in prior_hashes.items():
    assert B.sha(B.BASE / name) == expected, name
execution = B.read_json(B.BASE / "reports/execution-result.json")
assert execution["status"] == "completed"
assert execution["physical_predictions"]["terminal_original_requests"]["completed_predictions"] == 872
assert sum(v["completed_predictions"] for v in execution["physical_predictions"].values()) == 2322
summary = B.read_json(B.BASE / "reports/verified-summary.json")
assert summary["terminal"]["all"]["views"]["classifier"]["known_reference_n"] == 871
assert summary["terminal"]["all"]["views"]["classifier"]["confirmed_correct"] == 667
assert summary["terminal"]["all"]["correct_completed_overall"]["denominator"] == 872
assert summary["terminal"]["all"]["completion"]["numerator"] == 871
assert summary["terminal"]["all"]["system_failure"] == 1
assert summary["excluded_no_truth"]["truth"] is None
report = B.BASE / "reports/REPORT.md"
text = report.read_text(encoding="utf-8")
if report.stat().st_size > 20000:
    assert "阅读导航" in text
assert "不是未见成绩" in text and "871/872" in text and "model_uncertain" in text
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=B.ROOT, text=True).strip()
assert head == binding["git_head"]
result = {"at": B.now(), "status": "validated", "formal_inference_rerun": False,
          "protected_files_unchanged": len(binding["protected_sha256"]),
          "runner_and_binding_scripts_unchanged_since_execution": True,
          "prior_prediction_and_aggregate_output_hashes_unchanged": len(prior_hashes),
          "new_python_ast_files": len(list(B.BASE.glob("*.py"))), "report_bytes": report.stat().st_size,
          "git_head_unchanged": True, "terminal_expected": 872, "terminal_reference": 871,
          "terminal_uncertain_reference": 98, "physical_terminal_predictions": 872,
          "physical_all_predictions": 2322, "rows_are_not_independent_roots": True,
          "proxy_not_human_gold_or_production_safe": True}
B.save_json(B.BASE / "reports/final-validation.json", result)
files = sorted(p for p in B.BASE.rglob("*") if p.is_file() and not any(
    part.startswith(("selftest-tmp", "research-qa-tmp")) for part in p.relative_to(B.BASE).parts))
B.save_json(B.BASE / "reports/final-output-manifest.json", {
    "at": B.now(), "scope": str(B.BASE), "source_binding_sha256": B.sha(B.BINDING),
    "files": {p.relative_to(B.BASE).as_posix(): {"sha256": B.sha(p), "bytes": p.stat().st_size} for p in files},
    "excluded_from_inventory": "scoped disposable pytest temporary fixture directories only",
    "no_production_source_edits": True})
print(json.dumps(result, ensure_ascii=False))
