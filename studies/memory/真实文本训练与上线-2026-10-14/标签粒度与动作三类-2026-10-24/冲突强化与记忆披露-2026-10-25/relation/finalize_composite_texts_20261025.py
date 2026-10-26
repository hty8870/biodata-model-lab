from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import composite_cases as C
import prepare_relation_reviews_20261025 as G

B, K, S = C.B, C.K, C.S
ROOT = Path(__file__).resolve().parent
AUTHORS = C.COMPOSITE_DIR / "authors-real-r2p1"
OUT = C.COMPOSITE_DIR / "final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    specs = K.read_jsonl(C.COMPOSITE_SPECS)
    by_id = {s["family_id"]: s for s in specs}
    raw_paths = sorted((AUTHORS / "raw").glob("*.json"))
    assert len(raw_paths) == 12
    raw_by_id, style = {}, []
    for path in raw_paths:
        raw = K.read_json(path)
        spec = by_id[raw["family_id"]]
        assert raw["succeeded"] and raw["finish_reason"] == "stop"
        assert raw["spec_sha256"] == K.sha256_json(spec)
        assert raw["prompt_sha256"] == K.sha256_text(B.build_prompt(spec))
        payload = B.parse_author_response(raw["response_text"], spec)
        assert not B.validate_author_payload(spec, payload)
        for item in spec["items"]:
            setting = S.SETTINGS_BY_ID[spec["setting"]]
            for side in ("old", "new"):
                text = payload["items"][item["pair_id"]][side]
                surface = (item.get("surface") or {}).get(side)
                warnings = S.surface_warnings(text, item[side], surface)
                if warnings:
                    style.append({"pair_id": item["pair_id"], "side": side, "warnings": warnings})
        raw_by_id[raw["family_id"]] = (raw, path, payload)
    spec = by_id["cfx-01"]
    raw, path, payload = raw_by_id["cfx-01"]
    item = next(i for i in spec["items"] if i["pair_id"] == "cfx-01-p4")
    text = payload["items"][item["pair_id"]]["old"]
    assert "preferred_paraphrase_token_missing" in S.surface_warnings(text, item["old"], item["surface"]["old"])
    missing = text
    for token in sorted(S.POLARITY_TOKENS["require"], key=len, reverse=True):
        missing = missing.replace(token, "")
    assert "missing polarity token" in S.check_text_gates(missing, item["old"], S.SETTINGS_BY_ID[spec["setting"]], surface=item["surface"]["old"])
    if args.check_only:
        print(json.dumps({"real_raw_valid": 12, "style_preference_and_polarity_checks": 2, "new_model_calls": 0}))
        return
    newly_accepted = 0
    for family, (raw, path, payload) in raw_by_id.items():
        spec = by_id[family]
        target = AUTHORS / f"{family}.json"
        if not B.verified_record(target, spec, K.AUTHOR_PLAN["composite"]):
            identity = {k: raw.get(k) for k in ("provider", "model", "base_url_sha256", "thinking", "max_tokens")}
            identity.update({"channel": "chat", "replayed_without_model_call": True,
                "source_raw": path.relative_to(ROOT).as_posix(), "source_raw_sha256": K.sha256_file(path),
                "text_gate_sha256": K.sha256_file(Path(S.__file__))})
            B.write_author_record(AUTHORS, spec, payload, model=raw["model"], prompt=B.build_prompt(spec),
                                  usage=raw["usage"], source="author", identity=identity)
            newly_accepted += 1
    records = B.load_author_records(AUTHORS)
    for spec in specs:
        assert B.verified_record(AUTHORS / f"{spec['family_id']}.json", spec, K.AUTHOR_PLAN["composite"])
    if not (OUT / "rows.jsonl").exists():
        C.assemble(specs, records, fixture=False, out_dir=OUT)
    else:
        K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=K.SELFTEST_DIR) as temporary:
            comparison = Path(temporary)
            C.assemble(specs, records, fixture=False, out_dir=comparison)
            assert (comparison / "rows.jsonl").read_bytes() == (OUT / "rows.jsonl").read_bytes()
    audit = C.audit(out_dir=OUT, mode="final")
    assert audit["ok"], audit
    audit.pop("generated")
    rows = K.read_jsonl(OUT / "rows.jsonl")
    tokenizer = B._load_tokenizer()
    lengths = [len(tokenizer.encode(r["visible_text"], add_special_tokens=True)) for r in rows]
    assert max(lengths) <= 768
    main_specs = K.read_jsonl(K.SPECS_PATH)
    main_keys = {split: {B.DIAG.item_keys(s, i, meaning=True)["pair"] for s in main_specs if s["split"] == split for i in s["items"]}
                 for split in ("train", "dev", "heldout")}
    overlap = {split: [i["pair_id"] for s in specs for i in s["items"] if B.DIAG.item_keys(s, i, meaning=True)["pair"] in keys]
               for split, keys in main_keys.items()}
    result = {"schema": "real-composite-materialization/1", "new_model_calls": 0, "groups": 12, "pairs": 48,
        "all_groups_contain_conflict": True, "always_escalate_group_action_baseline": 1.0,
        "purpose": "four-pair exact-set diagnostic only; cannot establish conflict-free bypass or router utility",
        "semantic_pair_overlap_with_new60": overlap, "literal_audit": audit,
        "style_warnings": style, "input_tokens": {"max": max(lengths), "limit": 768, "truncated": 0},
        "original_calls": 12, "original_tokens": sum(r["usage"]["total_tokens"] for r, _, _ in raw_by_id.values()),
        "raw_sha256": {family: K.sha256_file(path) for family, (_, path, _) in raw_by_id.items()},
        "author_record_sha256": {s["family_id"]: K.sha256_file(AUTHORS / f"{s['family_id']}.json") for s in specs},
        "rows_sha256": K.sha256_file(OUT / "rows.jsonl"), "gate_sha256": K.sha256_file(Path(S.__file__)),
        "runner_sha256": K.sha256_file(Path(__file__)), "status": "materialized_not_student_evaluated_or_independently_reviewed"}
    G.publish(OUT / "materialization-audit.json", json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"groups": 12, "pairs": 48, "newly_accepted_without_model_call": newly_accepted,
        "original_tokens": result["original_tokens"], "input_tokens": result["input_tokens"],
        "semantic_overlap_counts": {k: len(v) for k, v in overlap.items()}, "audit_checks": len(audit["checks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
