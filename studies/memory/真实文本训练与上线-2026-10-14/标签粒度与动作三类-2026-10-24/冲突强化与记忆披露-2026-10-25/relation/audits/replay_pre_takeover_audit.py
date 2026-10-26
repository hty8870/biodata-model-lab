"""从接手时保全的源码/fixture 重现旧红灯；不是原审计文件的字节恢复。"""
import importlib.util
import json
import sys
from pathlib import Path

relation = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(relation))
import conflict_common as K

archive = relation / "_归档" / "v2.2-takeover-20261025"
source = archive / "source" / "build_conflict_data.py"
module_spec = importlib.util.spec_from_file_location("pre_takeover_builder", source)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
processed = archive / "selftest-pre-v2.2" / "processed"
specs = module.build_specs()
report = module.audit_dataset(processed_dir=processed, manifest_path=processed / "manifest.json",
                              specs=specs, mode="fixture")
report["reproduction"] = {
    "original_report_bytes_recovered": False,
    "note": "The first takeover fixture-bundle accidentally reused the v2.1 report filename. That new report was moved to audit-first-v2.2-misnamed-as-v2.1.json; this file is a fresh reproduction from preserved pre-takeover source and fixture, not the overwritten original report.",
    "builder_sha256": K.sha256_file(source),
    "fixture_manifest_sha256": K.sha256_file(processed / "manifest.json"),
}
K.write_json_atomic(archive / "pre-takeover-audit-reproduced.json", report)
summary = {"checks": len(report["checks"]), "passed": sum(v["ok"] for v in report["checks"].values()),
           "failed": [k for k, v in report["checks"].items() if not v["ok"]],
           "original_report_bytes_recovered": False}
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["checks"] == 25 and summary["passed"] == 23 else 1)
