from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import action_common_20261024 as AC
import build_action_data_20261024 as DATA
from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT

BACKGROUND = "两条都是同一项目的长期记忆。每项设置独立且只能取一个值，未提及项不作限定。新条目在相同设置的冲突处优先，旧条目未被覆盖且仍有效的内容需要保留。"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-name", default="data")
    args = parser.parse_args()
    output = ROOT / args.output_name
    output.mkdir(exist_ok=False)
    checks = DATA.selftest()
    specs_path = ROOT / "case-specs-20261024.json"
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    assert len(specs) == len({spec["family"] for spec in specs}) == 20
    tokenizer = AutoTokenizer.from_pretrained(args.repo_root / AC.MOTHER / "tokenizer", local_files_only=True)
    previous_encodings = {digest(json.loads(line)["encoding"]) for split in ("train", "dev")
                          for line in (ROOT.parent / "inputs-v1" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()}
    rows, inputs = [], []
    for family_index, spec in enumerate(specs):
        for orientation in range(2):
            a, b = orientation, orientation ^ (family_index % 2)
            old = {"0": a, "1": b}
            mode = "both_changed" if (family_index + orientation) % 2 == 0 else "carry_valid_old"
            whole_b = 1 - b if mode == "both_changed" else b
            cases = [(0, {"0": a, "2": 1}), (1, {"0": 1 - a, "1": whole_b}), (2, {"0": 1 - a, "2": 1})]
            old_clauses = [spec["a"][a], spec["b"][b]]
            if orientation:
                old_clauses.reverse()
            old_text = "；".join(old_clauses) + "。"
            group = spec["family"] + "-" + str(orientation)
            for expected, new in cases:
                reference = DATA.oracle(old, new)
                assert reference["action"] == expected
                conflict = [key for key in old.keys() & new.keys() if old[key] != new[key]]
                uncovered = old.keys() - new.keys()
                direct_action = 0 if not conflict else (2 if uncovered else 1)
                assert direct_action == reference["action"]
                assert direct_action != (reference["action"] + 1) % 3
                new_clauses = [spec["a"][new["0"]], spec["b"][new["1"]] if "1" in new else spec["c"]]
                if (family_index + orientation) % 2:
                    new_clauses.reverse()
                pair = {"old": {"kind": "memory", "text": old_text},
                        "new": {"kind": "memory", "text": "；".join(new_clauses) + "。"},
                        "frame": {"mode": "same_domain", "background": BACKGROUND}}
                item_id = "cf-" + digest([spec["family"], orientation, expected])[:14]
                row = {"id": item_id, "family": spec["family"], "domain": spec["domain"], "cluster_id": group,
                       "orientation": orientation, "whole_mode": mode if expected == 1 else None,
                       "pair": pair, "action_index": expected, "fine_index": reference["fine"],
                       "action": AC.ACTIONS[expected], "formal": {"old": old, "new": new, **reference},
                       "uncovered_old_count": len(uncovered), "preserve_count": len(reference["preserve_keys"]),
                       "conflict_count": len(reference["conflict_keys"]), "retained_old_text": spec["b"][b] if uncovered else "",
                       "label_origin": "finite-slot-oracle plus independent AI text review; not human gold",
                       "usage": "diagnostic_only_never_train_or_select"}
                prepared = SR.prepare_semantic_input(pair, tokenizer, max_length=384, role_independent_frame=True,
                                                     limits=SR.SEMANTIC_LIMITS_RESEARCH)
                forward = prepared["directions"]["forward"]
                assert forward["complete"], item_id
                key = digest(forward["encoding"])
                assert key not in previous_encodings
                rows.append(row)
                inputs.append({**{key: value for key, value in row.items() if key not in ("formal", "pair")},
                               "encoding": forward["encoding"], "visible_text": forward["text"],
                               "view_sha256": forward["view_sha256"], "encoding_sha256": key, "complete": True})
    random.Random(20261024 + 31).shuffle(rows)
    order = {row["id"]: index for index, row in enumerate(rows)}
    inputs.sort(key=lambda row: order[row["id"]])
    assert len(rows) == len({row["id"] for row in rows}) == len({digest(row["pair"]) for row in rows}) == 120
    assert Counter(row["action_index"] for row in rows) == {0: 40, 1: 40, 2: 40}
    assert Counter(row["whole_mode"] for row in rows if row["action_index"] == 1) == {"both_changed": 20, "carry_valid_old": 20}
    assert len({row["pair"]["old"]["text"] for row in rows}) == len({row["cluster_id"] for row in rows}) == 40
    assert len({row["encoding_sha256"] for row in inputs}) == 120
    for group in {row["cluster_id"] for row in rows}:
        selected = [row for row in rows if row["cluster_id"] == group]
        assert len(selected) == 3 and {row["action_index"] for row in selected} == {0, 1, 2}
        assert all(row["pair"]["old"]["text"].count("；") == row["pair"]["new"]["text"].count("；") == 1 for row in selected)
    write_rows(output / "rows.jsonl", rows)
    write_rows(output / "inputs.jsonl", inputs)
    families = sorted({row["family"] for row in rows})
    for part in range(2):
        selected = set(families[part::2])
        blind = [{"id": row["id"], "pair": row["pair"]} for row in rows if row["family"] in selected]
        assert len(blind) == 60
        write_rows(output / f"blind-part-{part + 1}.jsonl", blind)
    lengths = [len(row["encoding"]["input_ids"]) for row in inputs]
    manifest = {"n": len(rows), "semantic_families": 20, "triplets": 40, "actions": {AC.ACTIONS[k]: v for k, v in Counter(row["action_index"] for row in rows).items()},
                "information_preservation_denominator": sum(row["uncovered_old_count"] > 0 for row in rows),
                "conflict_denominator": sum(row["conflict_count"] > 0 for row in rows), "input_complete": 120,
                "token_min": min(lengths), "token_max": max(lengths), "new_clause_count_every_row": 2,
                "training_usage": False, "selection_usage": False, "human_gold": False,
                "old_train_dev_encoding_intersection": 0, "oracle_selftests": checks,
                "sha256": {name: FT.file_sha256(output / name) for name in ("rows.jsonl", "inputs.jsonl", "blind-part-1.jsonl", "blind-part-2.jsonl")},
                "spec_sha256": FT.file_sha256(specs_path), "script_sha256": FT.file_sha256(__file__),
                "renderer_sha256": FT.file_sha256(SR.__file__), "oracle_source_sha256": FT.file_sha256(DATA.__file__)}
    dump(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
