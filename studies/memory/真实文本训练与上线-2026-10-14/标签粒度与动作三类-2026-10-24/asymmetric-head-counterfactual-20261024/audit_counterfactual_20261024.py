from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(directory):
    return {row["id"]: row for row in map(json.loads, (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines())}


def review(path):
    text = path.read_text(encoding="utf-8")
    value, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    assert value["n"] == len(value["answers"]) == len({row["id"] for row in value["answers"]})
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    v1, v2, final = (read_rows(ROOT / directory) for directory in ("data", "data-v2", "data-v3"))
    assert v1.keys() == v2.keys() == final.keys()
    first = review(ROOT / "ai-blind-review-part1.md")
    second = review(ROOT / "ai-blind-review-part2.md")
    revised = review(ROOT / "ai-blind-review-revision.md")
    answers = first["answers"] + second["answers"]
    assert len(answers) == len({row["id"] for row in answers}) == 120
    original_answers = {row["id"]: row["action"] for row in answers}
    original_mismatches = [key for key in v1 if original_answers[key] != v1[key]["action"]]
    assert not original_mismatches
    changed = [key for key in v1 if v1[key]["pair"] != v2[key]["pair"]]
    assert len(changed) == 6 and {v2[key]["family"] for key in changed} == {"export_format"}
    revision_answers = {row["id"]: row["action"] for row in revised["answers"]}
    assert set(changed) == set(revision_answers)
    assert not revised["ambiguities"] and not revised["definition_issues"]
    combined = dict(original_answers, **revision_answers)
    for key in final:
        old, new = v2[key], final[key]
        assert old["action_index"] == new["action_index"] == v1[key]["action_index"]
        assert old["pair"]["old"] == new["pair"]["old"] and old["pair"]["new"] == new["pair"]["new"]
        assert new["pair"]["frame"]["background"] == "两条都是同一项目的长期记忆。每项设置独立且只能取一个值，未提及项不作限定。"
        assert combined[key] == new["action"]
        assert new["usage"] == "diagnostic_only_never_train_or_select"
    manifest = json.loads((ROOT / "data-v3/manifest.json").read_text(encoding="utf-8"))
    assert manifest["n"] == manifest["input_complete"] == 120 and manifest["token_max"] <= 384
    assert manifest["information_preservation_denominator"] == manifest["conflict_denominator"] == 80
    assert all(sha(ROOT / "data-v3" / name) == expected for name, expected in manifest["sha256"].items())
    result = {"status": "passed", "initial_blind_ai_agreement": {"correct": 120, "n": 120, "human_gold": False},
              "initial_ambiguity_flags": first["ambiguities"] + second["ambiguities"],
              "initial_definition_notes": first["definition_issues"] + second["definition_issues"],
              "revised_export_family_blind_agreement": {"correct": 6, "n": 6, "ambiguities": revised["ambiguities"]},
              "unchanged_rows_after_content_revision": 114, "final_reference_actions": combined,
              "neutral_frame_revision": "v3 removes role-dependent policy from shared frame; old/new text and gold unchanged from v2; task priority remains fixed externally",
              "final_labels_ai_checked": 120, "semantic_families": 20, "triplets": 40,
              "final_source_sha256": {name: sha(ROOT / "data-v3" / name) for name in ("rows.jsonl", "inputs.jsonl", "manifest.json")},
              "reference_scope": "program oracle plus one independent AI judgment per row; no human gold or semantic independence certification"}
    (ROOT / "data-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "final_reference_actions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
