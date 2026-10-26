from __future__ import annotations

import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / "data-unexposed-v1" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
backgrounds = sorted({row["pair"]["frame"]["background"] for row in rows})
assert len(backgrounds) == 2
strata = list(itertools.product(range(3), range(2), range(2)))
groups = {key: [] for key in strata}
for row in rows:
    key = (row["action_index"], row["orientation"], backgrounds.index(row["pair"]["frame"]["background"]))
    groups[key].append(row)
assert all(groups.values())
selected = []
for key in strata:
    row = min(groups[key], key=lambda item: (item["family"], item["id"]))
    selected.append({
        "stratum": {"action_index": key[0], "orientation": key[1], "background_index": key[2]},
        "id": row["id"], "family": row["family"], "domain": row["domain"], "action": row["action"],
        "old": row["pair"]["old"]["text"], "new": row["pair"]["new"]["text"],
        "background": row["pair"]["frame"]["background"], "formal": row["formal"],
        "uncovered_old_count": row["uncovered_old_count"], "retained_old_text": row["retained_old_text"],
    })
print(json.dumps({"selection": "lexicographically first row in each action/orientation/background stratum",
                  "n": len(selected), "rows": selected}, ensure_ascii=False, indent=2))
