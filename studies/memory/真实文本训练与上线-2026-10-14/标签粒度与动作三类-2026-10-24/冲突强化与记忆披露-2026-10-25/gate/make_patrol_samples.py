"""从独立 final 的 non-reject 项生成冻结 patrol samples；不调用 LLM。

- `quarantine`：three_state 决策为 quarantine 的条目；
- `active`：three_state 决策为 keep 的条目（active/eligible 抽样，不一定是已 active）；
- gold 只放在 `scoring_gold`，`patrol.build_prompt` 只投影公开字段，LLM 看不到标签/臂名；
- 输出两个独立文件 + manifest + SHA，供主 Agent 后续真实调用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from heldout_score import validate_dataset
from patrol import build_prompt

PUBLIC_FIELDS = ("id", "text", "summary", "trigger", "evidence", "tier", "scope")


def _sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_value(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _public(item: dict) -> dict:
    return {key: item[key] for key in PUBLIC_FIELDS if key in item}


def build_samples(dataset: dict, final: dict, *, source_paths: dict) -> tuple[list[dict], list[dict]]:
    validate_dataset(dataset)
    items = {entry["item"]["id"]: entry["item"]
             for batch in dataset["batches"] for entry in batch["cases"]}
    rows = [row for batch in final["arms"]["three_state"]["batches"] for row in batch["rows"]]
    quarantine, active = [], []
    for row in rows:
        if row["decision"] not in {"keep", "quarantine"}:
            continue
        item = items.get(row["id"])
        if item is None:
            raise ValueError(f"final row {row['id']} 不在独立数据中")
        sample = {
            "sample_id": ("final-q-" if row["decision"] == "quarantine" else "final-a-") + row["id"],
            "kind": "quarantine" if row["decision"] == "quarantine" else "active",
            "item": _public(item),
            "scoring_gold": {
                "label": item.get("label"),
                "evidence_feasible": item.get("evidence_feasible"),
                "scope_valid": item.get("scope_valid"),
                "schema_valid": item.get("schema_valid", True),
                "decision": row["decision"],
                "primary_reason": row["primary_reason"],
                "injection_form": row["injection_form"],
            },
            "source": {"dataset": source_paths["dataset"], "final": source_paths["final"],
                       "arm": "three_state", "instrument_note": final.get("instrument_note")},
        }
        (quarantine if sample["kind"] == "quarantine" else active).append(sample)
                                           
    for sample in quarantine + active:
        prompt = build_prompt(sample)
        payload = json.loads(prompt.split("条目 JSON：\n", 1)[1])
        if set(payload[0]) - set(PUBLIC_FIELDS):
            raise AssertionError(f"{sample['sample_id']}: prompt 暴露非公开字段")
        for forbidden in ("scoring_gold", "three_state", "injection_form", "evidence_feasible"):
            if forbidden in prompt:
                raise AssertionError(f"{sample['sample_id']}: prompt 含 {forbidden}")
    return quarantine, active


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成冻结 patrol samples（不调用 LLM）")
    parser.add_argument("--data", type=Path,
                        default=Path("independent/data/cases-2026-10-25.json"))
    parser.add_argument("--final", type=Path, default=Path("out/heldout-final.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args(argv)
    dataset = json.loads(args.data.read_text(encoding="utf-8"))
    final = json.loads(args.final.read_text(encoding="utf-8"))
    if final.get("role") != "final" or not final.get("instrument_note"):
        raise SystemExit("final 必须 role=final 且带 instrument_note")
    quarantine, active = build_samples(
        dataset, final, source_paths={"dataset": str(args.data), "final": str(args.final)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, kind, samples in (
            ("patrol-final-quarantine.json", "quarantine", quarantine),
            ("patrol-final-active.json", "active", active)):
        path = args.out_dir / name
        payload = {"schema": "patrol-samples/1", "date": "2026-10-25", "kind": kind,
                   "note": ("quarantine 复核样本；gold 仅在 scoring_gold，prompt 不含"
                            if kind == "quarantine" else
                            "active/eligible 抽样审计；只给建议，不自动改库；gold 仅在 scoring_gold"),
                   "dataset_sha256": _sha_bytes(args.data),
                   "final_sha256": _sha_bytes(args.final),
                   "instrument_note": final.get("instrument_note"),
                   "samples": samples}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files[name] = {"path": str(path), "sha256": _sha_bytes(path), "count": len(samples)}
    manifest = {"schema": "patrol-samples-manifest/1", "date": "2026-10-25",
                "dataset_sha256": _sha_bytes(args.data),
                "final_sha256": _sha_bytes(args.final),
                "instrument_note": final.get("instrument_note"),
                "counts": {"quarantine": len(quarantine), "active": len(active)},
                "files": files,
                "prompt_leak_check": True,
                "note": "gold 只在 scoring_gold；patrol.build_prompt 只投影公开字段；本步未调用 LLM"}
    manifest_path = args.out_dir / "patrol-final-samples-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"ok": True, "counts": manifest["counts"], "files": files,
                      "manifest": {"path": str(manifest_path),
                                   "sha256": _sha_bytes(manifest_path)}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
