"""复合集合（整组路由）的规格、作者接口、装配与审计。

复合案例 = 一个场景里的 4 条记忆对照；路由决策是“整组是否需要人工/LLM 复核”，
因此评分必须按组全对（all-correct）而不是只算单对。该集合**不参与训练或选点**，
作者与 split 也独立于 60 family 训练集，只作为锁后的策略路由证据。

CLI
---
    python composite_cases.py                     # selftest（fixture）
    python composite_cases.py specs
    python composite_cases.py author --model <name> --channel ollama|fixture [--limit N]
    python composite_cases.py assemble [--fixture | --authors DIR]
    python composite_cases.py audit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import conflict_common as K
import conflict_semantics as S
import build_conflict_data as B

COMPOSITE_DIR = K.HERE / "data" / "composite"
COMPOSITE_SPECS = COMPOSITE_DIR / "specs.jsonl"
COMPOSITE_ROWS = COMPOSITE_DIR / "rows.jsonl"
COMPOSITE_MANIFEST = COMPOSITE_DIR / "manifest.json"
GROUP_COUNT = 12
PAIRS_PER_GROUP = 4
                                           
GROUP_RECIPE_PLANS = (
    ("C1", "C3", "P1", "E1"),
    ("C2", "C5", "P3", "E2"),
    ("C3", "C1", "P4", "E1"),
    ("C4", "C2", "P6", "E2"),
    ("C5", "C3", "P7", "E1"),
    ("C1", "C4", "P2", "E2"),
)


def build_specs() -> list[dict]:
    specs = []
    for group in range(GROUP_COUNT):
        setting = S.SETTINGS[(group * 7) % len(S.SETTINGS)]
        recipes = GROUP_RECIPE_PLANS[group % len(GROUP_RECIPE_PLANS)]
        group_id = f"cfx-{group + 1:02d}"
        items = []
        for position, recipe in enumerate(recipes):
            s = S.slot(setting, 100 + group, position)
            old, new = S.build_natural(recipe, setting, s)
            label = S.classify_with(old, new, setting)
            if label != S.NATURAL_RECIPES[recipe]["label"]:
                raise AssertionError(f"{group_id}:{recipe} label drift")
            items.append(B._item(group_id, pair_id=f"{group_id}-p{position + 1}", kind="natural",
                                 recipe=recipe, label=label, old=old, new=new, setting=setting,
                                 position=position))
        counts = {name: 0 for name in K.COARSE}
        for item in items:
            counts[item["expected_coarse"]] += 1
        if counts != {"conflict": 2, "compatible": 1, "equivalent": 1}:
            raise AssertionError(f"{group_id} mix {counts}")
        specs.append({
            "schema": "conflict-composite-spec-v1", "family_id": group_id, "index": group,
            "split": "composite", "mechanism": "composite", "mechanism_name": "复合案例整组路由",
            "setting": setting["id"], "setting_name": setting["name"], "author_role": "composite",
            "background_hint": (f"{setting['name']}复合场景：{setting['scopes'][0]}与"
                                f"{setting['scopes'][1]}在同一批记忆里并存，四条要求分别约束"
                                f"不同事项，需要一起判断。{S.reference_partition_text(setting)}"),
            "items": items,
        })
    return specs


def fixture_record(spec: dict) -> dict:
    return {"schema": B.AUTHOR_SCHEMA, "family_id": spec["family_id"], "split": "composite",
            "author_role": "composite", "model": K.FIXTURE_AUTHOR, "source": "fixture",
            "created": K.utc_now(), "background": spec["background_hint"],
            "items": [{"pair_id": key, "old": value["old"], "new": value["new"]}
                      for key, value in sorted(B.fixture_author(spec)["items"].items())]}


def assemble(specs, records: dict, *, fixture: bool, out_dir: Path = COMPOSITE_DIR) -> dict:
    problems = []
    rows = []
    for spec in specs:
        record = records.get(spec["family_id"])
        if record is None:
            problems.append(f"{spec['family_id']}: missing author record")
            continue
        if record.get("source") != ("fixture" if fixture else "author"):
            problems.append(f"{spec['family_id']}: record source {record.get('source')!r}")
        payload_problems = B.validate_author_payload(
            spec, {"background": record["background"],
                   "items": {entry["pair_id"]: entry for entry in record["items"]}})
        problems.extend(f"{spec['family_id']}: {problem}" for problem in payload_problems)
        family_rows, row_problems = B.rows_for_family(spec, record)
        problems.extend(f"{spec['family_id']}: {problem}" for problem in row_problems)
        for row in family_rows:
            row["phenomena"]["composite_group"] = spec["family_id"]
            row["root_id"] = f"{spec['family_id']}-{row['id'].rsplit('-', 1)[-1]}"
        rows.extend(family_rows)
    if problems:
        raise B.AuthorError("composite assembly problems:\n" + "\n".join(problems[:30]))
    rows.sort(key=lambda row: row["id"])
    K.write_jsonl_atomic(out_dir / "rows.jsonl", rows)
    manifest = {"schema": "conflict-composite-manifest-v1", "created": K.utc_now(),
                "fixture": bool(fixture), "groups": len(specs), "rows": len(rows),
                "sha256": K.sha256_file(out_dir / "rows.jsonl"),
                "coarse": K.family_balance(rows),
                "authors": sorted({row["author"] for row in rows}),
                "spec_sha256": K.sha256_json(specs),
                "policy": "只用于锁后整组路由评分；不参与训练或选点"}
    K.write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def audit(out_dir: Path = COMPOSITE_DIR, *, mode: str = "final") -> dict:
    rows = K.read_jsonl(out_dir / "rows.jsonl")
    manifest = K.read_json(out_dir / "manifest.json") if (out_dir / "manifest.json").is_file() else {}
    checks = {}

    def check(name, ok, **extra):
        checks[name] = {"ok": bool(ok), **extra}

    groups = {}
    for row in rows:
        groups.setdefault(row["family_id"], []).append(row)
    group_problems = []
    for group_id, members in groups.items():
        if len(members) != PAIRS_PER_GROUP:
            group_problems.append(f"{group_id}: {len(members)} pairs")
        counts = K.family_balance(members)
        if counts != {"conflict": 2, "compatible": 1, "equivalent": 1}:
            group_problems.append(f"{group_id}: mix {counts}")
        if {row["phenomena"].get("composite_group") for row in members} != {group_id}:
            group_problems.append(f"{group_id}: composite_group mismatch")
    check("group_shape", not group_problems, problems=group_problems[:10])

    ids = [row["id"] for row in rows]
    check("id_unique", len(ids) == len(set(ids)))
    contract = []
    for row in rows:
        contract.extend(K.validate_new_row(row, "composite"))
    check("row_contract", not contract, problems=contract[:10])

    serialize = []
    for row in rows:
        visible = K.SR.semantic_pair_text(K.SR.validate_semantic_pair(row["pair"]))
        if visible != row["visible_text"]:
            serialize.append(row["id"])
    check("serialization", not serialize, ids=serialize[:5])

    overlap = []
    corpus = []
    for split in ("train", "dev", "heldout"):
        path = K.PROCESSED_DIR / f"{split}.jsonl"
        if path.is_file():
            corpus.extend((row, split) for row in K.read_jsonl(path))
    for split in ("train", "dev", "calibration", "heldout"):
        path = K.PARENT_PROCESSED / f"{split}.jsonl"
        if path.is_file():
            corpus.extend((row, f"old-{split}") for row in K.read_jsonl(path))
    if corpus:
        digests = {K.pair_digest(row) for row, _ in corpus}
        visible = {K.normalize_text(row["visible_text"]) for row, _ in corpus}
        for row in rows:
            if K.pair_digest(row) in digests or K.normalize_text(row["visible_text"]) in visible:
                overlap.append(row["id"])
    check("no_overlap_with_main_and_old", not overlap, ids=overlap[:5])
    fixture_rows = sum(1 for row in rows if row.get("source") == K.FIXTURE_SOURCE)
    expected_fixture = mode == "fixture"
    check("fixture_contract",
          (manifest.get("fixture") is True) == expected_fixture
          and (fixture_rows == len(rows)) == expected_fixture,
          manifest_fixture=manifest.get("fixture"), fixture_rows=fixture_rows,
          total=len(rows), mode=mode)
    report = {"schema": "conflict-composite-audit-v1", "generated": K.utc_now(), "mode": mode,
              "ok": all(entry["ok"] for entry in checks.values()), "groups": len(groups),
              "rows": len(rows), "checks": checks}
    return report


def cmd_specs(args) -> int:
    specs = build_specs()
    K.write_jsonl_atomic(COMPOSITE_SPECS, specs)
    print(json.dumps({"command": "composite-specs", "sha256": K.sha256_file(COMPOSITE_SPECS),
                      "groups": len(specs),
                      "rows": sum(len(spec["items"]) for spec in specs)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_author(args) -> int:
    specs = K.read_jsonl(COMPOSITE_SPECS)
    selected = specs[: args.limit or len(specs)]
    route = B.author_route_guard("composite", args.channel, args.model, args.budget_calls)
    expected_models = K.AUTHOR_PLAN.get("composite", ())
    out_dir = Path(args.out) if args.out else COMPOSITE_DIR / "authors" / B._slug(args.model or "fixture")
    if args.channel == "fixture":
        out_dir = B.guard_fixture_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger) if args.ledger else out_dir / "calls.jsonl"
    chat = None
    if args.channel == "chat":
        chat = B.ChatChannel(args.model)
        chat.prepare()
    budget = int(args.budget_calls or 0)
    written = calls = skipped = 0
    failures: list[dict] = []
    for spec in selected:
        target = out_dir / f"{spec['family_id']}.json"
        if args.resume and B.verified_record(target, spec, expected_models):
            skipped += 1
            continue
        if budget and calls >= budget and args.channel in ("chat", "ollama"):
            break
        prompt = B.build_prompt(spec)
        identity = None
        if args.channel == "fixture":
            payload = B.fixture_author(spec)
            B.write_author_record(out_dir, spec, payload, model=K.FIXTURE_AUTHOR,
                                  prompt=prompt, usage={"calls": 0}, source="fixture")
            written += 1
            continue
        channel = chat if chat is not None else B.OllamaChannel(args.model)
        outcome = channel.complete(prompt)
        calls += 1
        raw_path = B.write_raw_attempt(out_dir, spec, prompt, outcome, calls)
        status = "ok" if outcome["succeeded"] else "incomplete"
        B.append_call_ledger(ledger, {
            "raw_attempt": raw_path.relative_to(out_dir).as_posix(),
            "spec_sha256": K.sha256_json(spec),
            "max_tokens": outcome.get("max_tokens"), "thinking": outcome.get("thinking"),
            "family_id": spec["family_id"], "split": "composite", "channel": args.channel,
            "model": outcome.get("model") or args.model, "provider": outcome.get("provider"),
            "status": status, "finish_reason": outcome.get("finish_reason"),
            "error_kind": B.error_kind(outcome.get("error")),
            "usage": outcome.get("usage"), "latency_s": outcome.get("latency_s"),
            "prompt_sha256": K.sha256_text(prompt),
            "response_sha256": (K.sha256_text(outcome["text"])
                                if isinstance(outcome.get("text"), str) else None),
            "base_url_sha256": outcome.get("base_url_sha256"), "created": K.utc_now(),
        })
        if not outcome["succeeded"]:
            failures.append({"family_id": spec["family_id"], "status": status,
                             "finish_reason": outcome.get("finish_reason"),
                             "error_kind": B.error_kind(outcome.get("error"))})
            continue
        try:
            payload = B.parse_author_response(outcome["text"], spec)
        except B.AuthorError as error:
            failures.append({"family_id": spec["family_id"], "status": "unparsed",
                             "error_kind": str(error)[:80]})
            continue
        problems = B.validate_author_payload(spec, payload)
        if problems:
            failures.append({"family_id": spec["family_id"], "status": "rejected",
                             "error_kind": problems[0][:80]})
            continue
        identity = {"provider": outcome["provider"], "model": outcome["model"],
                    "base_url_sha256": outcome["base_url_sha256"],
                    "finish_reason": outcome["finish_reason"],
                    "latency_s": outcome["latency_s"], "channel": args.channel}
        B.write_author_record(out_dir, spec, payload, model=args.model, prompt=prompt,
                              usage=outcome["usage"], source="author", identity=identity)
        written += 1
    print(json.dumps({"command": "composite-author", "channel": args.channel, "model": args.model,
                      "out": str(out_dir), "written": written, "calls": calls,
                      "skipped_verified": skipped, "failures": len(failures),
                      "failed": failures[:10], "budget": budget or None, "route": route,
                      "ledger": str(ledger)}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def cmd_assemble(args) -> int:
    specs = K.read_jsonl(COMPOSITE_SPECS)
    if args.fixture:
        records = {spec["family_id"]: fixture_record(spec) for spec in specs}
    else:
        if not args.authors:
            raise SystemExit("需要 --authors DIR 或 --fixture")
        records = B.load_author_records(Path(args.authors))
        missing = [spec["family_id"] for spec in specs if spec["family_id"] not in records]
        if missing:
            raise SystemExit(f"缺少 {len(missing)} 个复合组：{missing[:5]}")
    out_dir = Path(args.out) if args.out else COMPOSITE_DIR
    if args.fixture:
        out_dir = B.guard_fixture_output(out_dir)
    manifest = assemble(specs, records, fixture=bool(args.fixture), out_dir=out_dir)
    print(json.dumps({"command": "composite-assemble", **manifest}, ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    report = audit(mode=args.mode)
    K.write_json_atomic(K.AUDITS_DIR / f"composite-{args.mode}.json", report)
    print(json.dumps({"command": "composite-audit", "ok": report["ok"],
                      "failed": [name for name, entry in report["checks"].items() if not entry["ok"]],
                      "groups": report["groups"], "rows": report["rows"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def run_selftest() -> dict:
    import tempfile
    checks = []

    def record(name, ok, **extra):
        checks.append({"check": name, "ok": bool(ok), **extra})
        if not ok:
            raise AssertionError(f"{name}: {extra}")

    specs = build_specs()
    record("spec_shape", len(specs) == GROUP_COUNT
           and all(len(spec["items"]) == PAIRS_PER_GROUP for spec in specs))
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conflict-composite-selftest-", dir=K.SELFTEST_DIR) as temporary:
        out = Path(temporary) / "composite"
        records = {spec["family_id"]: fixture_record(spec) for spec in specs}
        manifest = assemble(specs, records, fixture=True, out_dir=out)
        passed = audit(out_dir=out, mode="fixture")
        record("fixture_pipeline", passed["ok"],
               failed=[name for name, entry in passed["checks"].items() if not entry["ok"]])
                     
        rows = K.read_jsonl(out / "rows.jsonl")
        rows[0]["label"] = "equivalent"
        rows[0]["coarse_label"] = "equivalent"
        K.write_jsonl_atomic(out / "rows.jsonl", rows)
        failed = audit(out_dir=out, mode="fixture")
        record("must_fail_group_mix", not failed["ok"],
               failed=[name for name, entry in failed["checks"].items() if not entry["ok"]])
                           
        dropped = dict(records)
        dropped.pop(specs[0]["family_id"])
        rejected = False
        try:
            assemble(specs, dropped, fixture=True, out_dir=out)
        except Exception as error:
            rejected = "missing author record" in str(error)
        record("must_fail_missing_author", rejected)
                                                    
        manifest = assemble(specs, records, fixture=True, out_dir=out)
        manifest["fixture"] = False
        K.write_json_atomic(out / "manifest.json", manifest)
        failed = audit(out_dir=out, mode="final")
        record("must_fail_fixture_disguise",
               [name for name, entry in failed["checks"].items() if not entry["ok"]] == ["fixture_contract"],
               failed=[name for name, entry in failed["checks"].items() if not entry["ok"]])
    print(json.dumps({"selftest": True, "ok": True, "checks": checks}, ensure_ascii=False, indent=2))
    return {"ok": True, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(func=lambda args: (run_selftest(), 0)[1])
    sub.add_parser("specs").set_defaults(func=cmd_specs)
    author = sub.add_parser("author")
    author.add_argument("--model", default=None)
    author.add_argument("--channel", default="chat", choices=("chat", "fixture"),
                        help="chat=复用生产 llm_client（composite 预注册；必须 --model 与 --budget-calls）；"
                             "fixture=仅 selftest；composite 不走本机 ollama/file")
    author.add_argument("--out", default=None)
    author.add_argument("--limit", type=int, default=0)
    author.add_argument("--budget-calls", type=int, default=0,
                        help="chat/ollama 的最大调用次数；chat 必填（单次失败不自动重试）")
    author.add_argument("--resume", action="store_true", help="只跳过已验收文件")
    author.add_argument("--ledger", default=None)
    author.set_defaults(func=cmd_author)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--authors", default=None)
    assemble_parser.add_argument("--fixture", action="store_true")
    assemble_parser.add_argument("--out", default=None)
    assemble_parser.set_defaults(func=cmd_assemble)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--mode", default="final", choices=("final", "fixture"))
    audit_parser.set_defaults(func=cmd_audit)
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        return (run_selftest(), 0)[1]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
