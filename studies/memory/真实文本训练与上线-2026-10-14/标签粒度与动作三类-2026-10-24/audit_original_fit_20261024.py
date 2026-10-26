from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from research.memory_relations import metrics as EM
from research.memory_relations import semantic_artifacts as A
from research.memory_relations import semantic_data as D
from research.memory_relations import semantic_model as SM

ROOT = Path(__file__).resolve().parent
CLASSES = SR.SEMANTIC_RELATIONS
SOURCE = "artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v7-fine-development-20261014/teacher-completion-20261015-c71/delivery"
RUNTIME = "artifacts/relation-confidence-20261023/bundle-v2/runtime"
PINS = {"qualified-source.json": "f6160efdb9a5ae3586ffa22275543b3ca3c7ad352b0cc1a6a518022e414ecfaf",
        "allocation.json": "3d380ac6d280f171b84cfe37584ae6df460a8cb6bb6433158320fa7220f59a7c"}
META_PIN = "f600b2d46fa41f0fc18d697b7b1f139a3cabb457340f89ff30547bb89c39d742"


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summary(rows, direction="forward"):
    labels = np.asarray([CLASSES.index(row["gold" if direction == "forward" else "reverse_gold"]) for row in rows])
    probabilities = np.asarray([row[direction]["probabilities"] for row in rows], dtype=np.float64)
    assert probabilities.shape == (len(rows), 5) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    predictions = probabilities.argmax(axis=1)
    result = EM.classification_summary(labels, predictions, classes=CLASSES)
    result.update(correct=int((labels == predictions).sum()), ce=float(-np.log(probabilities[np.arange(len(rows)), labels]).mean()),
                  above_080=int((probabilities.max(axis=1) >= 0.8).sum()))
    return result


def selftest():
    rows = [{"gold": name, "forward": {"probabilities": [float(i == j) * .975 + .005 for j in range(5)]}}
            for i, name in enumerate(CLASSES)]
    assert summary(rows)["correct"] == 5
    for row in rows:
        row["forward"]["probabilities"] = np.roll(row["forward"]["probabilities"], 1).tolist()
    assert summary(rows)["correct"] == 0
    return {"perfect_5_of_5": True, "wrong_0_of_5": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", default="original-fit-v1")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    checks = selftest()
    source = args.repo_root / SOURCE
    protected = [source / name for name in ("qualified-source.json", "allocation.json", "source_rows.jsonl", "qualifications.jsonl")]
    protected.extend(args.repo_root / RUNTIME / name for name in ("semantic_meta.json", "semantic_model.pt"))
    old_path = args.repo_root / "artifacts/v50-v3-seed13-dev-20261020/dev113/predictions.jsonl"
    protected.append(old_path)
    before = {path.relative_to(args.repo_root).as_posix(): FT.file_sha256(path) for path in protected}
    for name, pin in PINS.items():
        assert FT.file_sha256(source / name) == pin, name
    print("pins checked; loading selected100 runtime on " + str(device), flush=True)
    scorer, meta, payload = A.load_runtime_artifact(args.repo_root / RUNTIME, expected_meta_sha256=META_PIN, device=str(device))
    del payload
    assert meta["train"]["origin"]["selected_state_sha256"] == "6dfaf5f122383ed358c1ea77617c743913047fa5a76262aed2c8cff391a3868b"
    bundle = D.load_qualified_source(source / "qualified-source.json")
    rows, lineage = SM.qualified_stage_rows(source / "qualified-source.json", source / "allocation.json", bundle=bundle)
    assert Counter(row["split"] for row in rows) == {"train": 401, "dev": 113}
    assert not ({row["cluster_id"] for row in rows if row["split"] == "train"} & {row["cluster_id"] for row in rows if row["split"] == "dev"})
    print("qualified 514 rows; preparing original v3 views", flush=True)
    orients = SM.oriented_rows(rows, scorer.tokenizer, 384)
    assert len(orients) == 1028
    model = scorer.model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(orients), 16):
            batch = FT.encode_prepared_views(scorer.tokenizer, [x["encoding"] for x in orients[start:start + 16]], device)
            probabilities.extend(model(**batch).float().softmax(dim=-1).cpu().tolist())
    old_predictions = {row["id"]: row for row in (json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines())}
    outputs, differences = [], []
    for index, row in enumerate(rows):
        left, right = orients[index * 2:index * 2 + 2]
        record = {key: row[key] for key in ("id", "root_id", "cluster_id", "split")}
        record.update(gold=left["label"], reverse_gold=right["label"], complete=left["prepared"]["complete"],
                      language=row.get("language", "unknown"), family=row.get("family", "unknown"),
                      pair=row["pair"], label_origin=row["label_origin"])
        for offset, (direction, item) in enumerate((("forward", left), ("reverse", right))):
            values = probabilities[index * 2 + offset]
            view = item["input_view"]
            encoding = item["encoding"]
            record[direction] = {"probabilities": values, "prediction": CLASSES[int(np.argmax(values))],
                                 "visible_text": view["text"], "view_sha256": view["view_sha256"],
                                 "encoding_sha256": D.digest(encoding), "token_length": len(encoding["input_ids"]),
                                 "complete": view["complete"]}
            if row["split"] == "dev":
                old = old_predictions[row["id"]]
                assert old["gold"] == record["gold"] and old["reverse_gold"] == record["reverse_gold"]
                assert old["input_audit"]["directions"][direction]["view_sha256"] == view["view_sha256"]
                differences.append(float(np.abs(np.asarray(values) - np.asarray(old[direction]["probabilities"])).max()))
                assert old[direction]["prediction"] == record[direction]["prediction"]
        outputs.append(record)
    assert max(differences) <= 1e-6, max(differences)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    result = {"model": "v3-c5-seed13-selected100", "new_training": False, "device": str(device),
              "analysis_selftest": checks, "dev_reference_max_probability_difference": max(differences),
              "source_sha256": before, "groups": {}, "seconds": time.perf_counter() - started,
              "input_recipe": meta["input_recipe"], "runtime_meta_pin": META_PIN,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    for split in ("train", "dev"):
        selected = [row for row in outputs if row["split"] == split]
        result["groups"][split] = {"n": len(selected), "clusters": len({row["cluster_id"] for row in selected}),
                                   "forward": summary(selected), "reverse": summary(selected, "reverse"),
                                   "complete": {}, "swap_consistent": sum(row["forward"]["prediction"] == SR.swap_relation(row["reverse"]["prediction"]) for row in selected)}
        for complete in (True, False):
            subset = [row for row in selected if row["complete"] == complete]
            result["groups"][split]["complete"][str(complete).lower()] = {"n": len(subset), "forward": summary(subset) if subset else None}
        print(json.dumps({"split": split, "correct": result["groups"][split]["forward"]["correct"], "n": len(selected), "ce": result["groups"][split]["forward"]["ce"]}), flush=True)
    assert all(FT.file_sha256(args.repo_root / name) == value for name, value in before.items())
    result["source_unchanged"] = True
    result["seconds"] = time.perf_counter() - started
    dump(output / "summary.json", result)
    print(json.dumps({"completed": True, "seconds": result["seconds"], "dev_parity_max": max(differences)}), flush=True)


if __name__ == "__main__":
    main()
