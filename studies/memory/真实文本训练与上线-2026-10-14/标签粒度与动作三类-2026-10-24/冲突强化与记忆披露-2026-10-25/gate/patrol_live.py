"""用生产 LLM 通道执行冻结 patrol 样本；只写研究 JSONL，不连接知识库。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parent
sys.path.insert(0, str(GATE))

from patrol import ClientResult, collect, load_records, load_samples, redact_model_identity, score
from dataset_recommender.llm.llm_client import (
    call_llm, is_complete_finish_reason, load_llm_config, should_use_llm,
)


def numeric_usage(raw_response) -> dict:
    if not isinstance(raw_response, dict):
        return {}
    usage = raw_response.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {key: value for key, value in usage.items() if type(value) in (int, float)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    samples = [sample for path in args.samples for sample in load_samples(path)]
    cfg = None
    identity = {"provider": "dry-run"}
    client = None
    if not args.dry_run:
        os.environ["LLM_MODEL"] = args.model
        cfg = load_llm_config()
        ready, reason = should_use_llm(cfg)
        if not ready:
            raise SystemExit(f"LLM 配置不可用：{reason}")
        identity = redact_model_identity(cfg.provider, cfg.model, cfg.base_url)

        def client(prompt):
            result = call_llm(prompt, cfg)
            complete = (result.attempted and result.succeeded
                        and is_complete_finish_reason(result.finish_reason))
            return ClientResult(
                raw=result.text if complete else None,
                usage=numeric_usage(result.raw_response),
                model_identity=redact_model_identity(
                    result.provider or cfg.provider, result.model or cfg.model, cfg.base_url),
            )

    collected = collect(samples, client=client, path=args.records,
                        max_calls=args.max_calls, model_identity=identity,
                        dry_run=args.dry_run)
    records, errors = load_records(args.records)
    result = score(records, samples, active_before=[], active_after=[])
    result.update(
        line_errors=errors,
        collection={key: collected[key] for key in ("calls", "skipped", "stopped", "record_count")},
        model=args.model,
        dry_run=args.dry_run,
        sample_files=[str(path) for path in args.samples],
        records_file=str(args.records),
        note="采集器未实例化 KnowledgeService；active/eligible 不是实际 active 库存。",
        active_unchanged=None,
        active_snapshot_verified=False,
        collector_db_mutations=0,
    )
    result["warnings"] = list(dict.fromkeys([
        *result.get("warnings", []),
        "未读取真实 active 库存快照；active_unchanged=null。collector_db_mutations=0 只表示采集器未写库。",
    ]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n": result["n"], "calls": collected["calls"],
        "completeness": result["completeness"], "quality_mean": result["quality_mean"],
        "unsafe_keep": result["unsafe_keep"], "usage_total": result["usage_total"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
