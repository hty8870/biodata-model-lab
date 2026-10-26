from pathlib import Path
import json

import build_conflict_data as B

ROOT = Path(__file__).resolve().parent
specs = {s["family_id"]: s for s in B.K.read_jsonl(B.K.SPECS_PATH)}
checks = []
for family, text in (
    ("cf-039", "2025 年这一次，所有场景禁止排期发布内容。"),
    ("cf-051", "2024年所有工单禁止分流。"),
):
    spec = specs[family]
    item = next(i for i in spec["items"] if i["pair_id"] == family + "-c2")
    factor = item["new"]
    setting = B.S.SETTINGS_BY_ID[spec["setting"]]
    year = str(factor["window"]["year"])
    assert int(year) in B.S.expected_tokens(factor, setting)["numbers"]
    good = B.S.check_text_gates(text, factor, setting)
    missing = B.S.check_text_gates(text.replace(year, ""), factor, setting)
    polarity = B.S.check_text_gates(text.replace("禁止", "必须"), factor, setting)
    assert not good, good
    assert f"missing number {year}" in missing, missing
    assert "missing polarity token" in polarity, polarity
    checks.append({"family_id": family, "valid_text": text, "valid_errors": good,
                   "missing_year_errors": missing, "reversed_polarity_errors": polarity})
result = {"passed": 6, "checks": checks, "gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
          "note": "Year and polarity remain required; exclusive-window wording is not a rejection criterion. Semantic review is separate."}
B.K.write_json_atomic(ROOT / "audits/window-gate-regression-20261026.json", result)
print(json.dumps(result, ensure_ascii=False))
