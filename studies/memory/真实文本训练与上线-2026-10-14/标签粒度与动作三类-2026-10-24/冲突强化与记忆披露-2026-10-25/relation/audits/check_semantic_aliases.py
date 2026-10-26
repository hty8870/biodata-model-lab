import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_conflict_data as B

K, S = B.K, B.S
specs = K.read_jsonl(K.SPECS_PATH)


def key(spec, factor):
    f = S.with_setting(factor, S.SETTINGS_BY_ID[spec["setting"]])
    return K.sha256_json([f["subject"], f["slot"], f["object"], f.get("referent") or "",
                          sorted(S._applicability(f)), sorted(S._allowed(f))])


entries = []
for spec in specs:
    for item in spec["items"]:
        entries.append((spec, item, key(spec, item["old"]), key(spec, item["new"])))
report = {"cross": {}, "within_pair_excess": {}, "control_cf_meaning_twins": []}
for left, right in (("train", "dev"), ("train", "heldout"), ("dev", "heldout")):
    pairs = {s: {(o, n) for sp, _, o, n in entries if sp["split"] == s} for s in (left, right)}
    sides = {s: {v for sp, _, o, n in entries if sp["split"] == s for v in (o, n)} for s in (left, right)}
    report["cross"][f"{left}|{right}"] = {"pair": len(pairs[left] & pairs[right]),
                                            "any_side": len(sides[left] & sides[right])}
for split in K.SPLITS:
    pairs = [(o, n) for sp, _, o, n in entries if sp["split"] == split]
    report["within_pair_excess"][split] = len(pairs) - len(set(pairs))
cf = {(o, n) for _, item, o, n in entries if item["dyad"]}
for spec, item, o, n in entries:
    if item["kind"] == "control" and (o, n) in cf:
        report["control_cf_meaning_twins"].append({"id": item["pair_id"], "recipe": item["recipe"],
                                                   "coarse": item["expected_coarse"]})
report["control_twins_by_recipe"] = dict(Counter(item["recipe"] for item in report["control_cf_meaning_twins"]))
K.write_json_atomic(K.AUDITS_DIR / "semantic-aliases-v2.2.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
