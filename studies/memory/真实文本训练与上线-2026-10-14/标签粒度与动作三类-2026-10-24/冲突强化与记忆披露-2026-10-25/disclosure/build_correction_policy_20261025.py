from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "parent_analysis_20261025"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, text: str) -> None:
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, path
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.codex_output.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == "{")
    raw = "".join(lines[start:])
    reference, end = json.JSONDecoder().raw_decode(raw)
    source_path = OUT / "source-only-reference-input-v1.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    atoms = {atom["id"]: atom for atom in source["atoms"]}
    old = {task["task_id"]: task for task in (json.loads(line) for line in
        (HERE / "pilot_v4/data/tasks.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(reference["tasks"]) == len({t["task_id"] for t in reference["tasks"]}) == 12
    assert {t["task_id"] for t in reference["tasks"]} == set(old)
    raw_path = OUT / "source-only-reference-v1.raw.json"
    publish(raw_path, raw[:end] + "\n")
    aliases_path = OUT / "source_value_aliases_v1.json"
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    policy = {"schema": "memory-measurement-correction-policy/1",
        "reference_kind": "source_only_AI_proxy_with_parent_canonical_mapping_not_human_gold",
        "status": "posthoc_diagnostic_not_preregistered_confirmatory_score",
        "source_input_sha256": sha(source_path), "raw_reference_sha256": sha(raw_path),
        "reference_session": "01a1736f-25d7-7d03-9a58-e92714bd04c4",
        "cli_output_sha256": sha(args.codex_output), "source_aliases_sha256": sha(aliases_path),
        "builder_sha256": sha(Path(__file__)), "tasks": []}
    for task in reference["tasks"]:
        assert task["status"] in ("valid", "ambiguous")
        if task["status"] == "ambiguous":
            assert task["decision"] is None and task["value"] is None and task["minimal_atom_sets"] == []
            policy["tasks"].append({**task, "value_variants": [],
                "canonical_mapping": "not scored; original result retained separately"})
            continue
        expected = old[task["task_id"]]["expected"]
        assert task["decision"] == expected["decision"]
        assert all(aid in atoms for group in task["minimal_atom_sets"] for aid in group)
        variants = [expected["value"]]
        if task["decision"] in ("allow", "deny"):
            if task["value"] not in variants:
                variants.append(task["value"])
            if len(task["minimal_atom_sets"]) == 1 and len(task["minimal_atom_sets"][0]) == 1 \
                    and expected["value_mode"] == "exact":
                literal = atoms[task["minimal_atom_sets"][0][0]]["text"]
                if literal not in variants:
                    variants.append(literal)
        for value in aliases.get(task["task_id"], []):
            if value not in variants:
                variants.append(value)
        policy["tasks"].append({**task, "reference_value_description": task["value"],
            "value": expected["value"], "value_mode": expected["value_mode"], "value_variants": variants,
            "canonical_mapping": "source-only answer and original short code checked semantically equivalent; ask/no_memory encoded as null"})
    policy["metric_limits"] = [
        "Action/value uses finite source-derived aliases, not unrestricted human semantic judgment",
        "Original strict field contract remains a separate unchanged metric",
        "Minimal evidence alternatives are source-only and identical across arms",
        "Ambiguous tasks remain visible but are excluded from new diagnostic denominators"]
    output = OUT / "correction-policy-v1.json"
    publish(output, json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"reference_tasks": 12, "valid": sum(t["status"] == "valid" for t in policy["tasks"]),
        "ambiguous": [t["task_id"] for t in policy["tasks"] if t["status"] != "valid"],
        "policy_sha256": sha(output), "new_experiment_service_calls": 0,
        "codex_reference_requests": 1, "codex_usage_tokens": None}))


if __name__ == "__main__":
    main()
