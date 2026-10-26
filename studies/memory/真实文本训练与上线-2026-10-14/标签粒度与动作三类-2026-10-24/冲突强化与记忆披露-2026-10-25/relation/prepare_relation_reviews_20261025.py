from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import build_conflict_data as B
import composite_cases as C

ROOT = Path(__file__).resolve().parent
AUTHORS = ROOT / "data/authors/confirmed-r2p1"
OUT = ROOT / "data/review_inputs"
PHASES = {
    "train-dev": {"specs": B.K.SPECS_PATH, "authors": AUTHORS, "splits": ("train", "dev"),
                  "families": 36, "per_family": 18, "rows": 648,
                  "review_model": "batiai/qwen3.8-27b:iq3"},
    "heldout": {"specs": B.K.SPECS_PATH, "authors": AUTHORS, "splits": ("heldout",),
                "families": 24, "per_family": 18, "rows": 432,
                "review_model": "deepseek-v4-flash"},
    "composite": {"specs": C.COMPOSITE_SPECS, "authors": C.COMPOSITE_DIR / "authors-real-r2p1",
                  "splits": ("composite",), "families": 12, "per_family": 4, "rows": 48,
                  "review_model": "batiai/qwen3.8-27b:iq3"},
}
BLIND_FIELDS = ["expected labels", "recipe", "split", "author", "source family/id"]
STATUS = "inputs_prepared_not_reviewed_or_trained"


def opaque(value: str) -> str:
    return "u_" + hashlib.sha256(("relation-review-v1/" + value).encode("utf-8")).hexdigest()[:16]


PROMPT_TEMPLATES = {
    "v1": ("独立审核中文记忆句对的逻辑关系。只依据背景和实际文字，不猜测语料作者的意图。"
           "equivalent=两句互相蕴含、实际要求相同；compatible=可以共同遵守，包括单向严格包含或互不相关；"
           "conflict=至少存在一个共同适用情境，两个义务不能同时满足。"
           "不同对象、不相交作用域或不相交时间不能仅因动词相反就判冲突。"
           "语气、词序、措辞不同不必然改变义务；若文字有真实歧义，ambiguous=true并说明。"
           "不要决定新增、替换、临时等操作。每个id必须判断一次。只输出JSON，不要Markdown。"
           "格式：{\"pairs\":[{\"id\":\"...\",\"relation\":\"equivalent或compatible或conflict\","
           "\"ambiguous\":false,\"reason\":\"简短且具体的文字依据\"}]}\n"),
    "v2": ("独立审核中文记忆句对的逻辑关系。只依据背景和实际文字，不猜测语料作者的意图。"
           "判断两条规则能否共同遵守：外部适用条件只包括主体、对象、作用域、事项、时间等，不包括待选择的数量或是否执行某个动作本身。"
           "先在每个共同适用域内，分别列出两条义务可接受的执行值或行为，再求交集；交集非空（包括严格包含与部分重叠）就是 compatible，只有交集为空才是 conflict。"
           "不能因为某个选值满足一方却违反另一方就判 conflict；这只说明该值不在交集内，不说明交集为空。"
           "若没有共同适用域，两条规则可同时遵守，判 compatible；不同对象、不相交作用域或不相交时间不能仅因动词相反就判 conflict。"
           "equivalent 要求两句互相蕴含，不能仅凭文字相似。不要臆造清单的成员包含或排除关系；确实无法确定时 ambiguous=true 留给后续审核，不要强行等价或强行冲突。"
           "例：[2,4] 与 [3,5] 可共存；[2,4] 与 [5,6] 冲突。"
           "语气、词序、措辞不同不必然改变义务；若文字有真实歧义，ambiguous=true并说明。"
           "不要决定新增、替换、临时等操作。每个id必须判断一次。先写不超过40个汉字的具体依据（reason），再填最终 relation；relation 必须与依据一致，不要长篇自我辩论。只输出JSON，不要Markdown。"
           "格式：{\"pairs\":[{\"id\":\"...\",\"reason\":\"...\",\"relation\":\"equivalent或compatible或conflict\",\"ambiguous\":false}]}\n"),
}
VERSIONS = tuple(PROMPT_TEMPLATES)


def prompt_for(unit: dict, version: str = "v1") -> str:
    if set(unit) != {"id", "background", "pairs"}:
        raise ValueError("Review unit must not expose source metadata or labels")
    if any(set(pair) != {"id", "old", "new"} for pair in unit["pairs"]):
        raise ValueError("Review pair contains non-text metadata")
    if version not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown review prompt version: {version}")
    return PROMPT_TEMPLATES[version] + json.dumps(unit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def publish(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Frozen review input differs: {path.name}"
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def selftest() -> None:
    unit = {"id": "x", "background": "同一对象同时刻。", "pairs": [{"id": "p", "old": "至少七个。", "new": "至多五个。"}]}
    assert "至少七个" in prompt_for(unit)
    try:
        prompt_for(dict(unit, expected="conflict"))
    except ValueError:
        pass
    else:
        raise AssertionError("Reference leakage accepted")
    try:
        prompt_for(dict(unit, pairs=[dict(unit["pairs"][0], recipe="C1")]))
    except ValueError:
        pass
    else:
        raise AssertionError("Pair metadata leakage accepted")
    print(json.dumps({"selftest_checks": 3, "passed": 3}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--phase", choices=PHASES, default="train-dev")
    parser.add_argument("--version", choices=VERSIONS, default="v1")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    source = PHASES[args.phase]
    prefix = f"{args.phase}-{args.version}"
    specs = B.K.read_jsonl(source["specs"])
    selected = [s for s in specs if s["split"] in source["splits"]]
    if len(selected) != source["families"]:
        raise SystemExit(f"refuse: {args.phase} expects {source['families']} families in "
                         f"{source['specs'].name}, found {len(selected)}")
    records = B.load_author_records(source["authors"])
    missing = [s["family_id"] for s in selected if not B.verified_record(
        source["authors"] / f"{s['family_id']}.json", s, B.K.AUTHOR_PLAN[s["split"]])]
    if missing:
        raise SystemExit(f"refuse: {args.phase} inputs incomplete: {len(missing)}/{len(selected)} "
                         f"families lack verified author records; expected {source['rows']} pairs; "
                         f"no files written")
    units, references, rows = [], [], []
    for spec in selected:
        record = records[spec["family_id"]]
        materialized, errors = B.rows_for_family(spec, record)
        assert not errors and len(materialized) == source["per_family"]
        rows.extend(materialized)
        unit = {"id": opaque(spec["family_id"]), "background": record["background"], "pairs": []}
        for row in materialized:
            unit["pairs"].append({"id": opaque(row["id"]), "old": row["pair"]["old"]["text"], "new": row["pair"]["new"]["text"]})
            references.append({"review_id": opaque(row["id"]), "unit_id": unit["id"], "source_id": row["id"],
                "family_id": spec["family_id"], "split": spec["split"], "expected_coarse": row["coarse_label"],
                "expected_fine": row["label"], "label_origin": row["label_origin"]})
        prompt_for(unit, args.version)
        units.append(unit)
    assert len(rows) == source["rows"]
    if args.phase == "composite":
        final_dir = C.COMPOSITE_DIR / "final"
        final_manifest = B.K.read_json(final_dir / "manifest.json")
        assert final_manifest["fixture"] is False and final_manifest["rows"] == len(rows)
        assert final_manifest["sha256"] == B.K.sha256_file(final_dir / "rows.jsonl")
        final_rows = {row["id"]: row for row in B.K.read_jsonl(final_dir / "rows.jsonl")}
        assert set(final_rows) == {row["id"] for row in rows}
        assert all(final_rows[row["id"]]["pair"] == row["pair"] for row in rows)
        assert all(final_rows[row["id"]]["visible_text"] == row["visible_text"] for row in rows)
    tokenizer = B._load_tokenizer()
    lengths = [len(tokenizer.encode(row["visible_text"], add_special_tokens=True)) for row in rows]
    assert len(lengths) == source["rows"] and max(lengths) <= 768
    units_path = OUT / f"{prefix}.units.jsonl"
    refs_path = OUT / f"{prefix}.references.jsonl"
    manifest_path = OUT / f"{prefix}.manifest.json"
    publish(units_path, "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in units))
    publish(refs_path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in references))
    expected = {"schema": "blinded-relation-review-inputs/1", "families": len(units), "rows": len(rows),
        "student_input_tokens": {"max": max(lengths), "mean": sum(lengths) / len(lengths), "limit": 768, "truncated": 0},
        "split_counts": dict(Counter(r["split"] for r in rows)), "coarse_counts": dict(Counter(r["coarse_label"] for r in rows)),
        "new_model_calls": 0, "units_sha256": B.K.sha256_file(units_path), "references_sha256": B.K.sha256_file(refs_path),
        "prompt_sha256_by_unit": {u["id"]: B.K.sha256_text(prompt_for(u, args.version)) for u in units},
        "author_sha256_by_family": {s["family_id"]: B.K.sha256_file(source["authors"] / f"{s['family_id']}.json") for s in selected},
        "specs_sha256": B.K.sha256_file(source["specs"]), "review_model": source["review_model"],
        "status": STATUS, "blind_fields": BLIND_FIELDS}
    if manifest_path.exists():
        existing = B.K.read_json(manifest_path)
        mismatched = sorted(key for key, value in expected.items() if existing.get(key) != value)
        if mismatched:
            raise SystemExit(f"refuse: frozen {manifest_path.name} differs on {mismatched}; "
                             "historical manifest is not rewritten")
        print(json.dumps({"phase": args.phase, "manifest": "verified_existing",
            "manifest_sha256": B.K.sha256_file(manifest_path), "families": len(units), "rows": len(rows),
            "units_sha256": expected["units_sha256"], "references_sha256": expected["references_sha256"],
            "prompt_units_verified": len(expected["prompt_sha256_by_unit"]),
            "historical_generator_sha256": existing.get("generator_sha256"),
            "historical_gate_sha256": existing.get("gate_sha256"),
            "current_generator_sha256": B.K.sha256_file(Path(__file__)), "new_model_calls": 0}, ensure_ascii=False))
        return
    manifest = {**expected, "phase": args.phase, "gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
                "generator_sha256": B.K.sha256_file(Path(__file__))}
    publish(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prompt_sha256_by_unit", "author_sha256_by_family")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
