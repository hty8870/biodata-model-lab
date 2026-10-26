from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, path
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    log = args.codex_output.read_text(encoding="utf-8")
    lines = log.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("{"))
    text = "".join(lines[start:]).lstrip()
    result, end = json.JSONDecoder().raw_decode(text)
    source = {r["case_id"]: r for r in (json.loads(s) for s in args.cases.read_text(encoding="utf-8").splitlines())}
    refs = {r["case_id"]: r for r in (json.loads(s) for s in args.references.read_text(encoding="utf-8").splitlines())}
    answers = result["cases"]
    assert len(answers) == len({r["case_id"] for r in answers}) == len(source)
    assert {r["case_id"] for r in answers} == source.keys() == refs.keys()
    joined = []
    for answer in answers:
        assert type(answer["ambiguous"]) is bool and isinstance(answer["reason"], str) and answer["reason"].strip()
        assert answer["relation"] in ("equivalent", "compatible", "conflict", None)
        if answer["relation"] is None:
            assert answer["ambiguous"]
        reference = refs[answer["case_id"]]
        state = "ambiguous" if answer["ambiguous"] else (
            "agrees_spec" if answer["relation"] == reference["expected_coarse"] else "differs_spec")
        joined.append({**source[answer["case_id"]], "adjudication": answer, "reference": reference, "state": state})
    session = re.search(r"Thread ready \(([^)]+)\)", log)
    manifest = {"schema": "source-only-relation-adjudication/1", "case_n": len(joined),
        "counts": dict(Counter(r["state"] for r in joined)), "not_human_gold": True,
        "selection": "disagreement or malformed prior review; not an unbiased accuracy sample",
        "request_model": "gpt-6-astra", "effort": "high", "backend_attested": False,
        "session_id": session.group(1) if session else None,
        "input_cases_sha256": sha(args.cases), "references_sha256": sha(args.references),
        "cli_output_sha256": sha(args.codex_output), "recorder_sha256": sha(Path(__file__)), "joined": joined}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    folder = ROOT / "audits" / f"source-only-adjudication-{digest[:16]}"
    publish(folder / "reference.raw.json", text[:end] + "\n")
    publish(folder / "joined.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    unresolved = [r for r in joined if r["state"] != "agrees_spec"]
    print(json.dumps({"output": str(folder), "counts": manifest["counts"], "session": manifest["session_id"],
        "unresolved": [{"id": r["reference"]["source_id"], "oracle": r["reference"]["expected_coarse"],
                        "adjudication": r["adjudication"], "old": r["old"], "new": r["new"], "background": r["background"]}
                       for r in unresolved]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
