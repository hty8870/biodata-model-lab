                      
"""生成/合并 C 路匿名忠实性审核；不读取模型预测，不调用 LLM。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SCHEMA = "ag-ablation-fidelity-v1"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_id(case_id: str, side: str) -> str:
    return hashlib.sha256(f"fidelity-v1|{case_id}|{side}".encode()).hexdigest()[:16]


def build() -> dict:
    raw = read_jsonl(DATA / "c" / "raw.jsonl")
    cases = read_jsonl(DATA / "processed" / "cases.jsonl")
    expected = {(case["case_id"], side) for case in cases for side in ("old", "new")}
    latest = {}
    for row in raw:
        key = (row.get("case_id"), row.get("side"))
        if key in expected and (row.get("call") or {}).get("succeeded") and row.get("raw_text"):
            if key not in latest or int(row["attempt"]) >= int(latest[key]["attempt"]):
                latest[key] = row
    missing = sorted(expected - set(latest))
    if missing:
        raise SystemExit(f"正式采集尚未完整：缺 {len(missing)} side，例如 {missing[:4]}；不生成正式盲审包")
    case_map = {case["case_id"]: case for case in cases}
    key_rows = []
    blind = []
    for case_id, side in sorted(latest):
        row = latest[(case_id, side)]
        aid = audit_id(case_id, side)
        key_rows.append({"audit_id": aid, "case_id": case_id, "side": side})
        blind.append({
            "audit_id": aid,
            "source": case_map[case_id][side]["text"],
            "status": row["status"],
            "kept": (row.get("gate") or {}).get("kept") or [],
            "note": "只评 text 是否忠实且为可独立更新的完整义务；没有 gold/模型分数。",
        })
    random.Random(20261025).shuffle(blind)
    for index in range(4):
        write_jsonl(DATA / "review" / f"blind-{index:02d}.jsonl", blind[index * 75:(index + 1) * 75])
    write_json(DATA / "review" / "key.json", key_rows)
    return {"sides": len(blind), "batches": 4, "per_batch": 75}


def merge(review_dir: Path, out: Path, final: bool) -> dict:
    keys = json.loads((DATA / "review" / "key.json").read_text(encoding="utf-8"))
    annotations = []
    for index in range(4):
        annotations.extend(read_jsonl(review_dir / f"review-{index:02d}.jsonl"))
    by_id = {}
    for row in annotations:
        aid = row.get("audit_id")
        if aid in by_id:
            raise SystemExit(f"重复 audit_id: {aid}")
        if not isinstance(row.get("faithful"), bool) or not isinstance(row.get("atomic"), bool):
            raise SystemExit(f"faithful/atomic 必须为 bool: {aid}")
        by_id[aid] = row
    expected_ids = {row["audit_id"] for row in keys}
    if set(by_id) != expected_ids:
        raise SystemExit(f"审核 ID 集合不符：missing={sorted(expected_ids-set(by_id))[:4]} extra={sorted(set(by_id)-expected_ids)[:4]}")
    grouped = {}
    for key in keys:
        aid = key["audit_id"]
        row = by_id[aid]
        grouped.setdefault(key["case_id"], []).append({**key, **row})
    cases = {}
    details = {}
    for case_id, rows in sorted(grouped.items()):
        passed = all(row["faithful"] and row["atomic"] for row in rows)
        cases[case_id] = {"A": True, "B": True, "C": passed}
        details[case_id] = rows
    payload = {"schema": SCHEMA,
               "source": "blind AI side review + main-agent adjudication; not human gold",
               "cases": cases}
    write_json(out, payload)
    write_json(DATA / "review" / "merged-details.json", details)
    return {"cases": len(cases), "sides": len(annotations),
            "c_true": sum(row["C"] for row in cases.values()),
            "c_false": sum(not row["C"] for row in cases.values()),
            "out": str(out), "final": final}


def selftest() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "review").mkdir()
        write_json(root / "review" / "key.json", [
            {"audit_id": "a", "case_id": "x", "side": "old"},
            {"audit_id": "b", "case_id": "x", "side": "new"},
        ])
        for index in range(4):
            rows = [] if index else [
                {"audit_id": "a", "faithful": True, "atomic": True},
                {"audit_id": "b", "faithful": True, "atomic": False},
            ]
            write_jsonl(root / "review" / f"review-{index:02d}.jsonl", rows)
        global DATA
        old = DATA
        try:
            DATA = root
            result = merge(root / "review", root / "candidate.json", False)
            assert result["c_false"] == 1 and result["c_true"] == 0
            bad = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
            assert bad["cases"]["x"] == {"A": True, "B": True, "C": False}
            (root / "review" / "review-00.jsonl").write_text(
                '{"audit_id":"unknown","faithful":true,"atomic":true}\n', encoding="utf-8")
            try:
                merge(root / "review", root / "bad.json", False)
            except SystemExit as exc:
                assert "审核 ID 集合不符" in str(exc)
            else:
                raise AssertionError("未知 audit_id 未被拒绝")
        finally:
            DATA = old
    return {"ok": True, "positive": "false merge", "negative": "unknown id rejected"}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--review-dir", default=str(DATA / "review"))
    merge_parser.add_argument("--out", default=str(DATA / "review" / "fidelity-candidate.json"))
    merge_parser.add_argument("--final", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "build":
        result = build()
    elif args.command == "merge":
        out = Path(args.out)
        result = merge(Path(args.review_dir), out, args.final)
        if args.final and out != DATA / "processed" / "fidelity.json":
            raise SystemExit("--final 只允许写 data/processed/fidelity.json；请先审 candidate")
    else:
        result = selftest()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
