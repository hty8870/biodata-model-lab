from pathlib import Path
import json
import subprocess
import sys
import time
from research.memory_relations import head_design as H

root = Path(__file__).resolve().parents[4]
base = Path(__file__).resolve().parent
role = sys.argv[1]
study = base / (role + "-study")
rel = lambda path: path.relative_to(root).as_posix()
steps = [
    ("formal", ["formal", "--study", role, "--out", rel(study / "formal")], 0),
    ("prepare", ["prepare", "--study-role", role, "--out", rel(study / "prepare")], 0),
]
if role == "smoke":
    steps += [
        ("train-dev", ["train-dev", "--prepare", rel(study / "prepare"), "--out", rel(study / "train-dev"), "--updates", "100"], 0),
        ("freeze", ["freeze", "--prepare", rel(study / "prepare"), "--train-dev", rel(study / "train-dev"), "--out", rel(study / "freeze")], 0),
    ]
    common = ["finalize", "--prepare", rel(study / "prepare"), "--train-dev", rel(study / "train-dev"), "--freeze", rel(study / "freeze" / "freeze.json")]
    steps += [
        ("score", [*common, "--out", rel(study / "finalize")], 0),
        ("repeat-refused", [*common, "--out", rel(study / "repeat-refused")], 5),
        ("reproduce", [*common, "--reproduce", "--out", rel(study / "reproduce")], 0),
        ("verify-only", [*common, "--verify-only", "--out", rel(study / "verify-only")], 0),
    ]
steps.append(("status", ["status", "--run", rel(study)], 0))
record = {"role": role, "source_bundle": H.source_bundle_info(), "protocol_sha256": H.load_protocol()[1], "steps": [], "pilot": role == "smoke", "formal_scored": False}
for name, args, expected in steps:
    cmd = [sys.executable, "-B", "-m", "research.memory_relations.head_design", *args]
    start = time.time()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, encoding="utf-8", errors="strict")
    log = base / f"{role}-{name}.log"
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    row = {"stage": name, "command": cmd, "exit_code": proc.returncode, "expected_exit_code": expected, "wall_seconds": time.time() - start, "log": rel(log)}
    record["steps"].append(row)
    H._write_json(base / f"{role}-cli-validation.json", record)
    print(f"{role}/{name}: exit={proc.returncode} expected={expected} wall={row['wall_seconds']:.3f}s", flush=True)
    if proc.returncode != expected:
        raise SystemExit(proc.returncode or 1)
print(json.dumps({"source_bundle_sha256": record["source_bundle"]["sha256"], "protocol_sha256": record["protocol_sha256"], "all_expected": True, "formal_scored": False}), flush=True)
