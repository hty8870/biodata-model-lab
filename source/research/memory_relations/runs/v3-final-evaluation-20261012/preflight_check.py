"""仅验证冻结注册表和缓存投影键，不进行模型预测。"""
from collections import Counter
import json

import binding as B
import formal_evaluation as R

binding = B.verify()
manifest = B.read_json(B.REF / "ref-ready-manifest.json")
references = R.S.load_rows(B.REF / "final-heldout.jsonl") + R.S.load_rows(B.REF / "final-ood.jsonl")
originals = []
for info in manifest["version_binding"]["sealed"].values():
    originals.extend(B.read_jsonl(info["path"]))
registry = R.build_registry(B.read_jsonl(B.REF / "expected-cases.jsonl"), references,
                            B.read_jsonl(B.REF / "reference_status.jsonl"), originals)
projection_errors = Counter()
for entry in registry.values():
    try:
        candidate, existing, evidence, cfg = R.S.materialize_request(entry["request"])
        projected, _ = R.M._request(candidate, existing, evidence, cfg.api_key)
        if R.FT.joint_text(entry["request"]) != R.FT.joint_text(projected):
            projection_errors["joint_input_different_after_projection"] += 1
    except Exception as exc:
        projection_errors[type(exc).__name__ + ":" + str(getattr(exc, "code", "not_recorded"))] += 1
old_rows = R.S.load_rows(B.OLD)
result = {"at": B.now(), "model_predictions": 0, "expected": len(registry), "references": len(references),
          "reference_counts": dict(Counter(r["split"] for r in references)),
          "roots_by_split": {split: len({r["group_id"] for r in registry.values() if r["split"] == split})
                             for split in ("heldout", "ood")},
          "unlabelled": [r["id"] for r in registry.values() if "reference" not in r],
          "projection_errors": dict(projection_errors),
          "old_set": {"rows": len(old_rows), "groups": len({r["group_id"] for r in old_rows}),
                      "rule_counts": dict(Counter(len(r["request"]["existing"]) for r in old_rows)),
                      "label_counts": dict(Counter(r["label"] for r in old_rows))},
          "source_hashes_verified": True}
assert result["roots_by_split"] == {"heldout": 40, "ood": 40}
B.save_json(B.BASE / "reports/preflight-check.json", result)
print(json.dumps(result, ensure_ascii=False))
