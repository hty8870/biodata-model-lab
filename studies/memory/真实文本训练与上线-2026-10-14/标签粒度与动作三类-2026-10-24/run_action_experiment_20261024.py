from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from research.memory_relations import finetune as FT

from action_common_20261024 import ROOT, dump


def run(script, repo, arguments, name):
    log = ROOT / (name + ".log")
    command = [sys.executable, "-B", str(ROOT / script), "--repo-root", str(repo), *arguments]
    print("START " + name, flush=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=repo, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit {code}; preserved log; no implicit retry")
    print("DONE " + name, flush=True)


def result(arm, seed):
    return json.loads((ROOT / "runs" / f"{arm}-seed{seed}" / "result.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    assert not (ROOT / "design-lock.json").exists(), "already started; resume from explicit completed-run state"
    probe = json.loads((ROOT / "probe-fit-36/result.json").read_text(encoding="utf-8"))
    review = json.loads((ROOT / "pilot-review-comparison.json").read_text(encoding="utf-8"))
    assert probe["passed"] and review["agreement"] == review["n"] == 36
    paths = [f"{directory}/{split}.jsonl" for directory in ("data-v1", "inputs-v1") for split in ("train", "dev", "heldout")]
    paths += ["action_common_20261024.py", "train_action_arm_20261024.py", "实验设计-动作三类与嵌套细分-2026-10-24.md"]
    lock = {"status": "go", "research_only": True, "runs": [["a3", 13], ["f6", 13], ["a3", 20260919], ["f6", 20260919], ["r6", 13]],
            "updates": 300, "inputs_sha256": {name: FT.file_sha256(ROOT / name) for name in paths},
            "initial_runs": [["a3", 13], ["f6", 13]], "second_seed_rule": "abs(dev F6-A3 macroF1)>=.02, before heldout",
            "random_control_rule": "dev F6-A3 macroF1>=.02, before heldout",
            "no_new_206_usage": True, "probe_correct": probe["final"]["correct"],
            "codex_review_session": "01a16ab3-63a3-7fa3-a1e7-7df5ab193434",
            "known_scope": "finite normalized binary constraints; synthetic text; no human gold or deployment"}
    dump(ROOT / "design-lock.json", lock)
    completed = []
    for arm in ("a3", "f6"):
        run("train_action_arm_20261024.py", args.repo_root, ["--arm", arm, "--seed", "13"], arm + "-seed13")
        completed.append({"arm": arm, "seed": 13})
        dump(ROOT / "progress.json", {"completed": completed, "heldout_opened": False})
    baseline, fine = result("a3", 13), result("f6", 13)
    delta = fine["selected_dev"]["macro_f1"] - baseline["selected_dev"]["macro_f1"]
    decision = {"seed13_dev_macro_delta_f6_minus_a3": delta, "second_seed": abs(delta) >= .02,
                "random_control": delta >= .02, "heldout_opened_at_decision": False}
    dump(ROOT / "development-decision.json", decision)
    if decision["second_seed"]:
        for arm in ("a3", "f6"):
            run("train_action_arm_20261024.py", args.repo_root, ["--arm", arm, "--seed", "20260919"], arm + "-seed20260919")
            completed.append({"arm": arm, "seed": 20260919})
            dump(ROOT / "progress.json", {"completed": completed, "heldout_opened": False})
    if decision["random_control"]:
        run("train_action_arm_20261024.py", args.repo_root, ["--arm", "r6", "--seed", "13"], "r6-seed13")
        completed.append({"arm": "r6", "seed": 13})
        dump(ROOT / "progress.json", {"completed": completed, "heldout_opened": False})
    for item in completed:
        current = result(item["arm"], item["seed"])
        item["checkpoint_sha256"] = current["checkpoint_sha256"]
    dump(ROOT / "evaluation-lock.json", {"status": "ready-for-one-heldout-pass", "runs": completed,
                                        "development_decision_sha256": FT.file_sha256(ROOT / "development-decision.json"),
                                        "design_lock_sha256": FT.file_sha256(ROOT / "design-lock.json")})
    run("evaluate_action_runs_20261024.py", args.repo_root, [], "independent-reload-and-heldout")
    dump(ROOT / "progress.json", {"completed": completed, "heldout_opened": True, "evaluation_completed": True})
    print("MAIN EXPERIMENT COMPLETE; report and independent metric verification remain", flush=True)


if __name__ == "__main__":
    main()
