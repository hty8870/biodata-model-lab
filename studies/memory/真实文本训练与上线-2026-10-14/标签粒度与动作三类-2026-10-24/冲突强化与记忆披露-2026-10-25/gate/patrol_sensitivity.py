"""构造去 summary 的 patrol 敏感性输入；ID、证据与 scoring_gold 不变。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    samples = source["samples"] if isinstance(source, dict) else source
    transformed = []
    for row in samples:
        item = dict(row["item"], summary="")
        updated = dict(row, item=item, sensitivity="summary_removed")
        assert updated["scoring_gold"] == row["scoring_gold"]
        transformed.append(updated)
    payload = {
        "schema": "patrol-samples-sensitivity/1",
        "date": "2026-10-25",
        "kind": source.get("kind") if isinstance(source, dict) else None,
        "source_file": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "transformation": "item.summary=''；其它字段、sample_id、evidence与scoring_gold不变",
        "samples": transformed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(transformed), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "out": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
