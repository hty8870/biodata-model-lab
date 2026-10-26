from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import build_conflict_data as B
import conflict_common as K
import prepare_relation_reviews_20261025 as G
import project_row_reviews_20261025 as P

ROOT = Path(__file__).resolve().parent


def signature(background, old, new):
    return K.sha256_json({"background": background, "old": old, "new": new})


def signature_row(row):
    pair = row["pair"]
    return signature(pair["frame"]["background"], pair["old"]["text"], pair["new"]["text"])


def collect_reviews(root, prefixes, evidence):
    original = (P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ)
    P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ = root, root / "data/review_inputs", root / "data/reviews", root / "data/review_adjudication"
    try:
        for prefix in prefixes:
            phase = P.load_phase(prefix)
            for unit in phase["units"]:
                binding = P.check_binding(phase, unit, phase["prompts"][unit["id"]])
                assert not binding["problems"], binding["problems"]
                parsed = json.loads(binding["outcome"]["text"])["pairs"]
                rows, _ = P.project_rows(parsed, {p["id"] for p in unit["pairs"]})
                for review in rows:
                    reference = phase["refs_by_id"].get(review["id"])
                    if review["status"] != "valid_classified" or review["ambiguous"] or not reference:
                        continue
                    if review["relation"] != reference["expected_coarse"]:
                        continue
                    pair = phase["pairs_by_id"][review["id"]]
                    evidence[(reference["source_id"], signature(pair["background"], pair["old"], pair["new"]), review["relation"])] = {
                        "kind": "blinded_cross_model_review", "phase": prefix,
                        "source_root": root.relative_to(ROOT).as_posix(), "source": binding["source"],
                        "review_id": review["id"], "reason": review["reason"]}
    finally:
        P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ = original


def add_adjudication(path, evidence):
    document = K.read_json(path)
    for row in document["joined"]:
        answer, ref = row["adjudication"], row["reference"]
        if row["state"] != "agrees_spec" or answer["ambiguous"]:
            continue
        assert answer["relation"] == ref["expected_coarse"]
        evidence[(ref["source_id"], signature(row["background"], row["old"], row["new"]), answer["relation"])] = {
            "kind": document.get("review_kind", "independent_source_only_adjudication"), "path": path.relative_to(ROOT).as_posix(),
            "sha256": K.sha256_file(path), "case_id": row["case_id"], "reason": answer["reason"]}


def selftest():
    a = signature("共同范围", "索取文件", "索取文件")
    assert a == signature("共同范围", "索取文件", "索取文件")
    assert a != signature("共同范围", "索取文件", "提供文件")
    assert a != signature("不同范围", "索取文件", "索取文件")
    proof = {("x", a, "equivalent"): "evidence"}
    assert ("x", a, "equivalent") in proof and ("x", a, "compatible") not in proof
    print(json.dumps({"selftest_checks": 4, "passed": 4}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authors", type=Path)
    parser.add_argument("--stage", type=Path, action="append", default=[])
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.authors:
        parser.error("--authors is required")
    evidence = {}
    collect_reviews(ROOT, ("train-dev-v2", "composite-v2", "heldout-v2"), evidence)
    for path in (
        ROOT / "audits/source-only-adjudication-311601a29505d132/joined.json",
        ROOT / "audits/source-only-adjudication-4efa709e8e406e6a/joined.json",
        ROOT / "audits/parent-semantic-adjudication-20261026.json",
    ):
        add_adjudication(path, evidence)
    for stage in args.stage:
        stage = stage.resolve()
        prefixes = [p.name.removesuffix(".manifest.json") for p in sorted((stage / "data/review_inputs").glob("*.manifest.json"))]
        assert prefixes, stage
        collect_reviews(stage, prefixes, evidence)
    rows = [r for split in K.SPLITS for r in K.read_jsonl(K.PROCESSED_DIR / f"{split}.jsonl")]
    assert len(rows) == 1080
    composite = K.read_jsonl(K.composite_rows_path())
    assert len(composite) == 48
    rows += composite
    missing, reviewed = [], []
    for row in rows:
        key = (row["id"], signature_row(row), row["coarse_label"])
        proof = evidence.get(key)
        if proof is None:
            missing.append({"id": row["id"], "text_sha256": key[1], "coarse_label": key[2]})
        else:
            reviewed.append({"id": row["id"], "text_sha256": key[1], "coarse_label": key[2], "proof": proof})
    specs = K.read_jsonl(K.SPECS_PATH)
    assert len(specs) == 60
    actual = {row["id"]: row for row in rows}
    assert len(actual) == len(rows)
    for spec in specs:
        author_path = args.authors / f"{spec['family_id']}.json"
        assert B.verified_record(author_path, spec, K.AUTHOR_PLAN[spec["split"]])
        generated, errors = B.rows_for_family(spec, K.read_json(author_path))
        assert not errors, errors
        for row in generated:
            assert actual[row["id"]]["pair"] == row["pair"]
            assert actual[row["id"]]["coarse_label"] == row["coarse_label"]
    markers = ("逐字使用", "完整列出", "最高优先级", "required_phrases")
    artifacts = [{"id": row["id"], "side": side, "marker": marker}
                 for row in rows for side in ("old", "new") for marker in markers
                 if marker in row["pair"][side]["text"]]
    unresolved = bool(missing or artifacts)
    record = {"schema": "semantic-review-closure/1", "status": "unresolved" if unresolved else "accepted_for_fixed_training",
              "not_human_gold": True, "rows": len(rows), "reviewed": len(reviewed), "missing": missing,
              "generation_instruction_artifacts": artifacts,
              "by_split": dict(Counter(r["split"] for r in rows)),
              "proof_kinds": dict(Counter(r["proof"]["kind"] for r in reviewed)),
              "processed_sha256": {s: K.sha256_file(K.PROCESSED_DIR / f"{s}.jsonl") for s in K.SPLITS},
              "composite_sha256": K.sha256_file(K.composite_rows_path()),
              "authors": args.authors.resolve().relative_to(ROOT).as_posix(),
              "specs_sha256": K.sha256_file(K.SPECS_PATH), "runner_sha256": K.sha256_file(Path(__file__)),
              "reviewed_rows": reviewed,
              "limitation": "AI source review and controlled-spec checks are proxy evidence, not proof of perfect labels; shared blind spots remain possible."}
    digest = K.sha256_json(record)
    K.write_json_atomic(ROOT / f"audits/semantic-review-closure-{digest[:16]}.json", record)
    if not unresolved:
        path = ROOT / "audits/semantic-review-final-20261026.json"
        if path.exists():
            assert K.read_json(path) == record, "Existing closure differs; preserve it and investigate"
        else:
            K.write_json_atomic(path, record)
    print(json.dumps({k: v for k, v in record.items() if k != "reviewed_rows"}, ensure_ascii=False))
    if unresolved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
