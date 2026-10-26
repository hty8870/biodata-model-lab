from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR

ROOT = Path(__file__).resolve().parent
MOTHER = "artifacts/worktrees/semantic-formal-train-20261015/research/memory_relations/runs/v8-formal-preparation-20261015/raw-mother3-seed-13"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "inputs-v1"
    output.mkdir(exist_ok=False)
    mother = args.repo_root / MOTHER
    assert FT.file_sha256(mother / "nli_model.pt") == "99e9b75f826533289bf7b43eb5336916c7929c6bf3f08b65750e431ebedcf26c"
    tokenizer = AutoTokenizer.from_pretrained(mother / "tokenizer", local_files_only=True)
    manifest = {"recipe": SR.SEMANTIC_INPUT_RECIPE, "max_length": 384,
                "tokenizer_encoding_sha256": NR.tokenizer_encoding_sha256(tokenizer), "splits": {},
                "source_sha256": {"renderer": FT.file_sha256(Path(SR.__file__)), "script": FT.file_sha256(__file__)}}
    visible_encodings = {}
    for split in ("train", "dev", "heldout"):
        path = ROOT / "data-v1" / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        results, incomplete = [], []
        for row in rows:
            prepared = SR.prepare_semantic_input(row["pair"], tokenizer, max_length=384,
                                                 role_independent_frame=True, limits=SR.SEMANTIC_LIMITS_RESEARCH)
            forward = prepared["directions"]["forward"]
            if not forward["complete"]:
                incomplete.append(row["id"])
            encoding = forward["encoding"]
            key = digest(encoding)
            if key in visible_encodings:
                previous = visible_encodings[key]
                assert previous["split"] == split, "identical encoded input crosses splits"
                assert previous["fine_index"] == row["fine_index"], "identical encoded input has different labels"
            visible_encodings[key] = row
            results.append({"id": row["id"], "cluster_id": row["cluster_id"], "domain": row["domain"],
                            "action_index": row["action_index"], "fine_index": row["fine_index"],
                            "encoding": encoding, "visible_text": forward["text"], "view_sha256": forward["view_sha256"],
                            "encoding_sha256": key, "complete": forward["complete"],
                            "preserve_count": len(row["formal"]["preserve_keys"]),
                            "conflict_count": len(row["formal"]["conflict_keys"]),
                            "old_clause_count": len(row["formal"]["old"]), "new_clause_count": len(row["formal"]["new"])})
        target = output / (split + ".jsonl")
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
        lengths = [len(row["encoding"]["input_ids"]) for row in results]
        manifest["splits"][split] = {"n": len(rows), "scenario_groups": len({row["cluster_id"] for row in rows}),
                                     "outer_document_topics": len({row["domain"] for row in rows}),
                                     "actions": dict(Counter(row["action_index"] for row in rows)),
                                     "fine": dict(Counter(row["fine_index"] for row in rows)),
                                     "complete": len(rows) - len(incomplete), "incomplete_ids": incomplete,
                                     "token_min": min(lengths), "token_max": max(lengths), "token_sum": sum(lengths),
                                     "source_sha256": FT.file_sha256(path), "inputs_sha256": FT.file_sha256(target)}
        print(json.dumps({"split": split, **manifest["splits"][split]}, ensure_ascii=False), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert all(item["complete"] == item["n"] for item in manifest["splits"].values()), "incomplete input: revise data before training"


if __name__ == "__main__":
    main()
