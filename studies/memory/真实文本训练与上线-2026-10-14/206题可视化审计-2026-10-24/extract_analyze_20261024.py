from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import metrics as M
from research.memory_relations import semantic_data as D

OUT = Path(__file__).resolve().parent
CLASSES = SR.SEMANTIC_RELATIONS
ZH = dict(zip(CLASSES, ("等价", "旧严格蕴含新", "新严格蕴含旧", "互斥", "相交但互不包含")))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def classification(rows):
    y = [CLASSES.index(row["gold"]) for row in rows]
    predicted = [CLASSES.index(row["prediction"]) if row["prediction"] in CLASSES else -1 for row in rows]
    return M.classification_summary(y, predicted, classes=CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = args.repo_root / "artifacts/v55-final-preflight-20261021"
    parent_path = base / "fine-final-parent-verification.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    model = next(item for item in parent["models"] if item["model_id"] == "v3-c5-seed13")
    sources = []

    def record_source(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None:
            assert actual == expected, (path.name, "source hash mismatch")
        logical = path.relative_to(args.repo_root).as_posix()
        sources.append({"logical_path": logical, "bytes": path.stat().st_size, "sha256": actual})
        return path

    record_source(parent_path)
    prediction_path = record_source(model["files"]["predictions.jsonl"]["path"], model["files"]["predictions.jsonl"]["sha256"])
    for name in ("metadata.json", "report.json"):
        record_source(model["files"][name]["path"], model["files"][name]["sha256"])
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == len({row["id"] for row in predictions}) == len({row["root_id"] for row in predictions}) == 206
    assert len({row["cluster_id"] for row in predictions}) == 43
    cohort_path = record_source(base / "fine-final-cohort.json", "a0558b250aea7a64c2f78dab05c930ca62fbcdfa455c4da85c9609b41c3c7656")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    raw_by_source = {}
    manifest_hashes = {}
    for source in cohort["sources"]:
        manifest_path = record_source(source["source_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset = manifest["assets"]["source_rows"]
        raw_path = record_source(asset["path"], asset["sha256"])
        raw_by_source[source["source_id"]] = {row["id"]: row for row in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines())}
        manifest_hashes[source["source_id"]] = sha(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_root / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
    merged = []
    for ordinal, pred in enumerate(predictions, 1):
        raw = raw_by_source[pred["source_id"]][pred["id"]]
        assert D.digest(raw) == pred["evidence"]["source_row_sha256"]
        assert pred["source_manifest_sha256"] == manifest_hashes[pred["source_id"]]
        for key in ("id", "root_id", "cluster_id", "split"):
            assert raw[key] == pred[key]
        assert raw["label"] == pred["gold"]
        pair = D.student_pair(raw)
        prepared = SR.prepare_semantic_input(pair, tokenizer, max_length=384,
                                             role_independent_frame=pred["evidence"]["frame_role_independent"], limits=SR.SEMANTIC_LIMITS_RESEARCH)
        audit = pred["input_audit"]
        assert prepared["raw_pair_sha256"] == audit["raw_pair_sha256"]
        assert prepared["complete"] == audit["complete"]
        for direction in ("forward", "reverse"):
            view = prepared["directions"][direction]
            assert view["view_sha256"] == audit["directions"][direction]["view_sha256"]
            assert D.digest(view["encoding"]) == audit["directions"][direction]["encoding_sha256"]
        probabilities = pred["forward"]["probabilities"]
        SR.validate_probabilities(probabilities)
        assert CLASSES[int(np.argmax(probabilities))] == pred["forward"]["prediction"]
        reverse_probabilities = pred["reverse"]["probabilities"]
        SR.validate_probabilities(reverse_probabilities)
        old, new = raw["pair"]["old"], raw["pair"]["new"]
        merged.append({
            "number": ordinal, "id": pred["id"], "root_id": pred["root_id"], "cluster_id": pred["cluster_id"],
            "split": pred["split"], "family": pred["family"], "language": pred["language"], "kind": pred["kind"],
            "mode": raw["phenomena"]["mode"], "scenario": raw["phenomena"].get("scenario", ""),
            "gold": pred["gold"], "gold_zh": ZH[pred["gold"]], "prediction": pred["forward"]["prediction"],
            "prediction_zh": ZH[pred["forward"]["prediction"]], "correct": pred["gold"] == pred["forward"]["prediction"],
            "probabilities": probabilities, "confidence": max(probabilities),
            "margin": float(np.sort(probabilities)[-1] - np.sort(probabilities)[-2]),
            "reverse_gold": pred["reverse_gold"], "reverse_prediction": pred["reverse"]["prediction"],
            "reverse_probabilities": reverse_probabilities,
            "swap_consistent": pred["reverse"]["prediction"] == SR.swap_relation(pred["forward"]["prediction"]),
            "complete": bool(audit["complete"]), "full_tokens": max(audit["full_visible_token_lengths"]),
            "visible_tokens": audit["directions"]["forward"]["token_length"],
            "old_text": old["text"], "new_text": new["text"], "background": raw["pair"].get("frame", {}).get("background", ""),
            "pair": raw["pair"], "model_visible_forward": prepared["directions"]["forward"]["text"],
            "model_visible_reverse": prepared["directions"]["reverse"]["text"],
            "source_id": pred["source_id"], "source_row_sha256": pred["evidence"]["source_row_sha256"],
            "view_sha256": audit["directions"]["forward"]["view_sha256"],
            "label_origin": raw["label_origin"], "nl_fidelity_origin": raw["NL_fidelity_origin"],
            "adjudication_origin": raw["adjudication_origin"], "teacher_evidence": raw["teacher_evidence"],
            "evidence": pred["evidence"], "input_audit": audit,
        })
    metrics = classification(merged)
    assert sum(row["correct"] for row in merged) == model["correct"] == 46
    assert abs(metrics["accuracy"] - model["accuracy"]) < 1e-12
    assert abs(metrics["macro_f1"] - model["macro_f1"]) < 1e-12
    assert Counter(row["gold"] for row in merged) == model["gold_counts"]
    assert sum(row["complete"] for row in merged) == 68
    grouped = {}
    for field in ("split", "complete", "kind", "family", "language", "mode", "label_origin"):
        grouped[field] = []
        for value in sorted({row[field] for row in merged}, key=str):
            members = [row for row in merged if row[field] == value]
            grouped[field].append({"value": value, "n": len(members), "correct": sum(row["correct"] for row in members),
                                   "classification": classification(members)})
    majority = max(CLASSES, key=lambda name: sum(row["gold"] == name for row in merged))
    majority_rows = [dict(row, prediction=majority) for row in merged]
    selftest_good = M.classification_summary(np.arange(5), np.arange(5), classes=CLASSES)
    selftest_bad = M.classification_summary(np.arange(5), (np.arange(5) + 1) % 5, classes=CLASSES)
    assert selftest_good["accuracy"] == selftest_good["macro_f1"] == 1
    assert selftest_bad["accuracy"] == selftest_bad["macro_f1"] == 0
    summary = {
        "model_id": model["model_id"], "n": len(merged), "clusters": 43, "classification": metrics,
        "majority_baseline": {"class": majority, "classification": classification(majority_rows), "posthoc_descriptive_only": True},
        "groups": grouped, "prediction_counts": dict(Counter(row["prediction"] for row in merged)),
        "confidence": {"min": min(row["confidence"] for row in merged), "max": max(row["confidence"] for row in merged),
                       "mean": float(np.mean([row["confidence"] for row in merged])),
                       "above_080": sum(row["confidence"] >= 0.8 for row in merged)},
        "swap_consistent": sum(row["swap_consistent"] for row in merged),
        "source_reconstruction": {"rows_joined_by_id_and_hash": 206, "view_and_encoding_sha_equal": 206,
                                  "model_or_api_calls": 0, "labels_modified": 0},
        "raw_lengths": {field: {"max": max(len(row[field]) for row in merged), "median": float(np.median([len(row[field]) for row in merged]))}
                        for field in ("old_text", "new_text", "background", "model_visible_forward")},
        "archived_models": [{key: value for key, value in item.items() if key != "files"} for item in parent["models"]],
        "analysis_selftest": {"perfect": selftest_good["accuracy"], "intentionally_wrong": selftest_bad["accuracy"]},
        "seconds": time.perf_counter() - started,
    }
    dump(OUT / "analysis-summary-2026-10-24.json", summary)
    with (OUT / "joined-206-2026-10-24.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    dump(OUT / "source-manifest-2026-10-24.json", {"files": sources, "source_bytes_unchanged": all(sha(args.repo_root / item["logical_path"]) == item["sha256"] for item in sources)})
    print(json.dumps({key: value for key, value in summary.items() if key not in ("groups", "archived_models")}, ensure_ascii=False, indent=2))
    print(json.dumps({"groups": {field: [{"value": row["value"], "n": row["n"], "correct": row["correct"]} for row in values] for field, values in grouped.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
