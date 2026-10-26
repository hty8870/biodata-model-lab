"""固定模型与源版本绑定；不读取audit，不更改原工件或参考。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
MODEL = ROOT / "research/memory_relations/runs/v3-data-ablation-20261012/new-adjudicated/scorers/finetuned-joint-size3290-seed13"
LOCK = ROOT / "research/memory_relations/runs/v3-study-lock-20261012/lock.json"
REF = ROOT / ".git/validation-worktrees/mr3-terminal-20261012/research/memory_relations/runs/terminal-reference/freeze/20261012-093519"
OLD = ROOT / "research/memory_relations/data/set_challenge.jsonl"
ATTEMPT = BASE / "formal-evaluation-attempt.json"
BINDING = BASE / "source-binding.json"
EXPECTED_HASHES = {
    MODEL / "model.pt": "3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498",
    MODEL / "meta.json": "7ab44014d1b84b6e50be45e84ddb33685661b00bc3017544e33b39e4b544f948",
    LOCK: "48f6a42cba47ef089769ced67308c38db1d297e7d91babd7f214230d16afae24",
    REF / "final-heldout.jsonl": "04fc924610e9295f3ed361c5eea36e510d9e9af357cf98261bdd63b9a1cde6be",
    REF / "final-ood.jsonl": "07bff8f1bdc37f00ec1466ca2102fdf1a5bf6000851db63e7033bd7a205a1be6",
    REF / "expected-cases.jsonl": "748e71eed4e51ead2ad4dc895288522aa0982c912a8efe047ac63cb848f5816d",
    REF / "reference_status.jsonl": "33f2149c369d4aa773f0cf307ea7d55e06b4dd74cb1421eaa66831ce7d97d919",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def check_hashes(hashes):
    for path, expected in hashes.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"hash_mismatch:{Path(path).name}")
    return True


def source_hashes():
    files = sorted((ROOT / "research/memory_relations").glob("*.py"))
    files += [ROOT / "src/dataset_recommender/agent/memory_relations.py",
              ROOT / "src/dataset_recommender/llm/dream.py",
              ROOT / "src/dataset_recommender/agent/agent_exec.py",
              ROOT / "research/memory_relations/spec.json",
              ROOT / "research/memory_relations/data/source_protocol_v3.json",
              ROOT / "research/memory_relations/data/teacher_protocol_v3.json"]
    return {str(path): sha(path) for path in files}


def bind():
    if ATTEMPT.exists() or BINDING.exists() or (BASE / "execution-started.json").exists():
        raise FileExistsError("formal_attempt_already_exists_no_fresh_retry")
    check_hashes(EXPECTED_HASHES)
    manifest, lock, meta = read_json(REF / "ref-ready-manifest.json"), read_json(LOCK), read_json(MODEL / "meta.json")
    if not (manifest["expected"] == 872 and manifest["usable"] == 871
            and manifest["human_gold"] is False and manifest["status"].startswith("ready-")):
        raise ValueError("reference_not_ready")
    if Path(lock["model_dir"]).resolve() != MODEL.resolve():
        raise ValueError("locked_model_path_mismatch")
    if not (meta["temperature"] == lock["calibration"]["temperature"] == 0.980590113228623
            and meta["threshold"] is None and meta["threshold_feasible"] is False
            and lock["calibration"]["selection"]["point"] is None):
        raise ValueError("locked_operating_metadata_mismatch")
    hashes = {str(path): value for path, value in EXPECTED_HASHES.items()}
    hashes[str(REF / "ref-ready-manifest.json")] = sha(REF / "ref-ready-manifest.json")
    hashes[str(OLD)] = sha(OLD)
    hashes.update(source_hashes())
    for info in manifest["version_binding"]["sealed"].values():
        hashes[info["path"]] = info["sha256"]
    check_hashes(hashes)
    result = {"schema": "locked-formal-evaluation-binding-v1", "created_at": now(),
              "root": str(ROOT), "git_head": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "model_dir": str(MODEL), "reference_dir": str(REF), "classes": lock["classes"],
              "temperature": meta["temperature"], "policy_threshold": None,
              "diagnostic_threshold": 0.0, "diagnostic_only": True, "risk_certificate": False,
              "human_gold": False, "expected_counts": {"heldout": 445, "ood": 427},
              "reference_counts": {"heldout": 444, "ood": 427},
              "protected_sha256": hashes, "python": sys.executable}
    marker = {"schema": "formal-evaluation-attempt-v1", "created_at": now(), "status": "bound",
              "single_attempt": True, "initial_formal_prediction_count": 0,
              "binding_sha256": canonical_hash(result), "model_weights_sha256": hashes[str(MODEL / "model.pt")],
              "model_meta_sha256": hashes[str(MODEL / "meta.json")], "lock_sha256": hashes[str(LOCK)],
              "note": "Exclusive x-create; failure retains marker and scores. No retrain/reselect/audit."}
    save_json(ATTEMPT, marker)
    save_json(BINDING, result)
    print(json.dumps({"status": "bound", "attempt": str(ATTEMPT), "protected_files": len(hashes)}))


def verify():
    binding, marker = read_json(BINDING), read_json(ATTEMPT)
    if canonical_hash(binding) != marker["binding_sha256"]:
        raise ValueError("binding_integrity_mismatch")
    check_hashes(binding["protected_sha256"])
    return binding


def describe():
    binding = verify()
    result = {}
    for name in ("expected-cases.jsonl", "final-heldout.jsonl", "final-ood.jsonl", "reference_status.jsonl"):
        rows = read_jsonl(REF / name)
        result[name] = {"rows": len(rows), "keys": sorted(rows[0]) if rows else [],
                        "request_counts": sum("request" in row for row in rows),
                        "source_id_counts": sum("source_id" in row for row in rows),
                        "split_counts": dict(Counter(row.get("split") for row in rows)),
                        "rule_counts": dict(Counter(len(row.get("request", {}).get("existing", [])) for row in rows))}
        if rows and isinstance(rows[0].get("request"), dict):
            result[name]["request_keys"] = sorted(rows[0]["request"])
        if rows and isinstance(rows[0].get("phenomena"), dict):
            result[name]["phenomena_keys"] = sorted(rows[0]["phenomena"])
    result["source_bound"] = True
    save_json(BASE / "reports/input-shape.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bind", "describe", "verify"))
    args = parser.parse_args()
    if args.mode == "bind":
        bind()
    elif args.mode == "describe":
        describe()
    else:
        binding = verify()
        print(json.dumps({"verified": True, "protected_files": len(binding["protected_sha256"])}))
