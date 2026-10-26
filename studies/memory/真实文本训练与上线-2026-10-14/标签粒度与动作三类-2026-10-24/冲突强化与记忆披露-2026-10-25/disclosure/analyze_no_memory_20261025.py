from __future__ import annotations

import hashlib
import json
from pathlib import Path

import measurement_correction_20261025 as M
import no_memory_controls_20261025 as N

HERE = Path(__file__).resolve().parent


def main() -> None:
    ctx = M.load_context()
    requests, _, _ = N.build_requests(N.load_tasks(), N.load_identity())
    empty = {"disclosure": N.empty_disclosure(), "item_ids": [], "coverage": {}}
    controls = []
    for request in requests:
        receipt = N.channels.response(N.REAL_DIR, request)
        task_id = request["task"]["task_id"]
        result = M.evaluate(receipt["text"], empty, ctx["policy_by_task"][task_id], ctx["public_facts"])
        controls.append({"task_id": task_id, "answer_text": receipt["text"], "new": result,
            "usage": receipt["usage"], "receipt_sha256": N.D.file_sha256(N.REAL_DIR / f"{request['request_id']}.outcome.json")})
    assert len(controls) == len({r["task_id"] for r in controls}) == 12
    by_task = {r["task_id"]: r for r in controls}
    observed = M.build_observations(ctx)
    paired = {}
    for arm in ("P-full", "A-full"):
        selected = [r for r in observed if r["source"] == "main_ds" and r["arm"] == arm and r["scale"] == 12]
        assert len(selected) == 12
        counts = {"both_match": 0, "memory_only": 0, "control_only": 0, "neither": 0, "unknown": 0}
        rows = []
        for row in selected:
            control = by_task[row["task_id"]]
            if row["new"]["status"] == "unknown":
                counts["unknown"] += 1
                state = "unknown"
            else:
                memory_match = row["new"]["action_alias_value_match"]
                control_match = control["new"]["action_alias_value_match"]
                state = "both_match" if memory_match and control_match else (
                    "memory_only" if memory_match else "control_only" if control_match else "neither")
                counts[state] += 1
            rows.append({"task_id": row["task_id"], "memory_match": row["new"]["action_alias_value_match"],
                         "control_match": control["new"]["action_alias_value_match"], "state": state})
        counts["scored_n"] = 12 - counts["unknown"]
        counts["memory_matches"] = counts["both_match"] + counts["memory_only"]
        counts["control_matches"] = counts["both_match"] + counts["control_only"]
        paired[arm] = {"counts": counts, "pairs": rows}
    result = {"schema": "matched-empty-memory-analysis/1", "new_model_calls": 0,
        "control_calls_already_collected": 12, "control_tokens": sum(r["usage"]["total_tokens"] for r in controls),
        "policy_sha256": N.D.file_sha256(M.POLICY),
        "control_run_manifest_sha256": N.D.file_sha256(N.REAL_DIR / "run_manifest.json"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls": controls, "paired": paired,
        "limits": ["Source-only synthetic AI reference; not human gold or real-user outcomes",
            "Posthoc matched S12 controls, not a randomized confirmatory trial",
            "One control per task is reused for both full-memory arms; not 24 independent controls",
            "Compare action/value, not the tautological absence of citations in empty memory",
            "Same requested provider/model/config, hidden backend weights are not attested",
            "Finite aliases and schema matching do not constitute complete natural-language judgment"]}
    digest = N.D.sha256(result)
    output = HERE / "parent_analysis_20261025" / f"matched-no-memory-{digest[:16]}.json"
    M.publish(output, result)
    print(json.dumps({"output": str(output), "control_tokens": result["control_tokens"],
                      "paired_counts": {k: v["counts"] for k, v in paired.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
