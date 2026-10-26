                       
"""分流三改生产路径评测（2026-09-19）：全部经生产函数实测，不是离线重放。

PART A（L0 级联，零 LLM）：生产 `_l0_route_verdict`（env 开关真开）在 2393 全语料上
  跑 binary/three_class 两模式——覆盖率/一致率 vs 金标（binary: binary_labels.jsonl；
  three_class: route_labels_v2.jsonl），以及 L0 单次评分时延分布。
PART B（二元 LLM 共识生产路径，真实 DeepSeek，抽样）：生产 `_run_route_consensus`
  mode="binary"（prompt 构造与打标同口径：`_route_context_zh` 空现场 + 原话尾段），
  抽分层样本实测与金标一致率 + 票级时延（票实录的 ms 字段）。
PART C（检索闸，零 LLM）：生产 `turn._prelim_gate_skip` 在 prelim_labels 全量上
  ——retrieve 召回（误放率，要求 ≥99%）/ skip 正确跳过率 / flight 起跑率；
  并实测 flight 本体 `rule_match_summary` 与闸评分的时延对照（闸收益证据）。

输出  research/route_eval/production_router_eval.json + 控制台摘要。

Usage:
  .venv/Scripts/python.exe research/route_eval/eval_production_router.py           # A+C（零 LLM）
  .venv/Scripts/python.exe research/route_eval/eval_production_router.py --llm 120 # 含 B（抽样 120）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent import agent_exec as ae              
from dataset_recommender.agent import l0_router as l0              
from dataset_recommender.agent import turn as turn_mod              

EVAL_DIR = AGENT_ROOT / "research" / "route_eval"
OUT = EVAL_DIR / "production_router_eval.json"


def _load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def part_a(binary_gold: list[dict], three_gold: list[dict]) -> dict:
    os.environ["BIODATA_ROUTE_L0"] = "1"
    out: dict = {}
    for mode, rows, gold_key, suite_map in (
            ("binary", binary_gold, "gate", {"write": "general", "read": "read"}),
            ("three_class", three_gold, "route_llm", None)):
        covered = agree = 0
        lat: list[float] = []
        for r in rows:
            utt = r["utterance"]
            t0 = time.perf_counter()
            hit = ae._l0_route_verdict(utt, mode)
            lat.append((time.perf_counter() - t0) * 1e6)
            if hit is None:
                continue
            covered += 1
            suite, _info = hit
            want = (suite_map[r[gold_key]] if suite_map else r[gold_key])
            agree += int(suite == want)
        lat.sort()
        n = len(rows)
        out[mode] = {
            "n": n, "covered": covered,
            "coverage": covered / n,
            "agreement_covered": agree / max(covered, 1),
            "l0_latency_us_p50": lat[n // 2], "l0_latency_us_p99": lat[int(n * 0.99)],
        }
    return out


def part_b(binary_gold: list[dict], sample_n: int) -> dict:
    from dataset_recommender.llm.llm_client import LLMConfig, load_env_candidates
    load_env_candidates(AGENT_ROOT)
    if not os.environ.get("OPENAI_API_KEY"):
        return {"skipped": "OPENAI_API_KEY missing"}
    model = ae._build_chat_model(LLMConfig(
        enable_llm=True, mock_llm=False, provider="trial",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash", timeout=60.0,
        temperature=0.2, max_tokens=8000, thinking=False))
                            
    by_gate: dict[str, list[dict]] = {}
    for r in binary_gold:
        by_gate.setdefault(r["gate"], []).append(r)
    per = max(1, sample_n // 2)
    sample = by_gate.get("write", [])[:per] + by_gate.get("read", [])[:per]
    os.environ["BIODATA_ROUTE_MODE"] = "binary"
    suite_map = {"write": "general", "read": "read"}
    agree = 0
    vote_ms: list[int] = []
    t0 = time.monotonic()
    for i, r in enumerate(sample):
        utt = r["utterance"]
        context = ae._route_context_zh(utt, has_results=False, result_total=0,
                                       retrieval=None, current_query="", current_filters=None)
        prompt = context + "\n\n----- 用户原话 -----\n" + utt
        route, votes = ae._run_route_consensus(model, prompt, mode="binary")
        vote_ms.extend(int(v.get("ms") or 0) for v in votes if v.get("ok"))
        agree += int(route == suite_map[r["gate"]])
        if (i + 1) % 20 == 0:
            print(f"  B: {i+1}/{len(sample)} agree={agree}", flush=True)
    vote_ms.sort()
    n = len(sample)
    return {
        "n": n, "agreement": agree / n,
        "wall_s": round(time.monotonic() - t0, 1),
        "vote_ms_p50": vote_ms[len(vote_ms) // 2] if vote_ms else None,
        "vote_ms_p99": vote_ms[int(len(vote_ms) * 0.99)] if vote_ms else None,
        "n_votes": len(vote_ms),
    }


def part_c(prelim_rows: list[dict]) -> dict:
    os.environ["BIODATA_PRELIM_GATE"] = "1"
    tp = fn = tn = fp = 0
    gate_lat: list[float] = []
    for r in prelim_rows:
        utt = r["utterance"]
        t0 = time.perf_counter()
        skip = turn_mod._prelim_gate_skip(utt)
        gate_lat.append((time.perf_counter() - t0) * 1e6)
        want_retrieve = r["gate"] == "retrieve"
        if want_retrieve and not skip:
            tp += 1
        elif want_retrieve and skip:
            fn += 1
        elif not want_retrieve and skip:
            tn += 1
        else:
            fp += 1
    gate_lat.sort()
    n = len(prelim_rows)
                                                           
    flight_ms: dict[str, list[float]] = {}
    try:
        from dataset_recommender.agent.turn import rule_match_summary
        probes = [r["utterance"] for r in prelim_rows[:30]]
        for label, sp in (("auto_recall_off", {"strategy": "auto", "recall": "off"}),
                          ("fixed_recall_off", {"strategy": "fixed", "recall": "off"}),
                          ("auto_recall_vector", {"strategy": "auto", "recall": "vector"})):
            lat = []
            rule_match_summary("预热检索单细胞数据", search_params=dict(sp))      
            for utt in probes:
                t0 = time.perf_counter()
                rule_match_summary(utt, search_params=dict(sp))
                lat.append((time.perf_counter() - t0) * 1e3)
            lat.sort()
            flight_ms[label] = {"p50": round(lat[len(lat) // 2], 1),
                                "p99": round(lat[int(len(lat) * 0.99)], 1)}
    except Exception as exc:
        flight_ms["error"] = type(exc).__name__                            
    return {
        "n": n,
        "retrieve_recall": tp / max(tp + fn, 1),
        "false_skip": fn,
        "skip_correct_rate": tn / max(tn + fp, 1),
        "flight_start_rate": (tp + fp) / n,
        "gate_latency_us_p50": gate_lat[n // 2],
        "gate_latency_us_p99": gate_lat[int(n * 0.99)],
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "flight_body_ms": flight_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", type=int, default=0, help="PART B 抽样量（0=跳过）")
    args = parser.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    binary_gold = _load_jsonl("binary_labels.jsonl")
    three_gold = _load_jsonl("route_labels_v2.jsonl")
    prelim_rows = _load_jsonl("prelim_labels.jsonl")
    report: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("PART A: L0 级联（生产 _l0_route_verdict）", flush=True)
    report["part_a_l0_cascade"] = part_a(binary_gold, three_gold)
    print(json.dumps(report["part_a_l0_cascade"], ensure_ascii=False, indent=2), flush=True)
    if args.llm:
        print(f"PART B: 二元 LLM 共识生产路径（抽样 {args.llm}）", flush=True)
        report["part_b_binary_llm"] = part_b(binary_gold, args.llm)
        print(json.dumps(report["part_b_binary_llm"], ensure_ascii=False, indent=2), flush=True)
    print("PART C: 检索执行闸（生产 turn._prelim_gate_skip）", flush=True)
    report["part_c_prelim_gate"] = part_c(prelim_rows)
    print(json.dumps(report["part_c_prelim_gate"], ensure_ascii=False, indent=2), flush=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
