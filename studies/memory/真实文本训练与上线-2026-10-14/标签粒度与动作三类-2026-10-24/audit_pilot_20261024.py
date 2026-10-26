from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    if args.response is None:
        rows = [json.loads(line) for line in (ROOT / "data-v1/pilot-36.jsonl").read_text(encoding="utf-8").splitlines()]
        random.Random(20261024).shuffle(rows)
        blind, mapping = [], []
        for index, row in enumerate(rows, 1):
            identifier = f"p{index:02d}"
            blind.append({"id": identifier, "pair": row["pair"]})
            mapping.append({"id": identifier, "source_id": row["id"], "reference": row["action"], "fine": row["fine"]})
        (ROOT / "pilot-blind-36.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
        (ROOT / "pilot-private-map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Prepared 36 blind pairs; no labels or formal constraints in reviewer input")
    else:
        reference = {row["id"]: row for row in json.loads((ROOT / "pilot-private-map.json").read_text(encoding="utf-8"))}
        response = json.loads(args.response.read_text(encoding="utf-8"))
        assert len(response) == len(reference) and {row["id"] for row in response} == set(reference)
        joined = [dict(reference[row["id"]], **{key: value for key, value in row.items() if key != "id"}) for row in response]
        errors = [row for row in joined if row["reference"] != row["prediction"]]
        result = {"n": len(joined), "agreement": len(joined) - len(errors), "differences": errors,
                  "status": "AI blind review against program oracle, not human gold", "joined": joined}
        (ROOT / "pilot-review-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("n", "agreement", "differences", "status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
