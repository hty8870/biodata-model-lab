                       
"""route_consensus 生产口径打标（System-One 分流器 MVP，2026-09-19）。

逐字复刻生产投票协议——**不抄逻辑，直接调生产真源函数**：
  system prompt  = agent_exec._route_consensus_prompt()（prompts/route_consensus.md 文件即真源）
  human prompt   = agent_exec._route_context_zh 的中性空现场（无结果/无当前查询/无检索概览）
                   + "\\n\\n----- 用户原话 -----\\n" + utterance
  投票协议       = agent_exec._run_route_consensus 本身（并行 0.0/0.8 → 分歧加投 0.5
                   → 多数决（唯一最高且 ≥2 有效票）→ 机械兜底 general）
  模型通道       = 生产 trial 同款：api.deepseek.com/v1 · deepseek-v4-flash · thinking=disabled
                   （经生产构造器 _build_chat_model + LLMConfig，2026-09-19 冒烟验证：
                   不带 thinking=disabled 时该模型 content 为空）

中性空现场是**冷启动第一句**的分布（has_results=False / 无 current_query / retrieval=None）。
生产上共识还会看到关键词快速计数段的检索概览行——本批刻意不含，作为已知分布缺口，
后续批次补「带概览行」对照切片（见研究/Jev分流器可行性-2026-09-19.md 遗留问题）。

输入  research/route_eval/route_corpus.jsonl
输出  research/route_eval/route_labels.jsonl（断点续跑，按 qid 跳过）：
  {qid, utterance, route_gold, route_llm, decided_by, votes: [...], model, ts}
  decided_by: agree2（首发两票一致）| majority3（加投后多数决）| fallback_general（机械兜底）

Usage:
  .venv/Scripts/python.exe research/route_eval/label_route_consensus.py
  .venv/Scripts/python.exe research/route_eval/label_route_consensus.py --limit 6  # 冒烟
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent.agent_exec import (              
    _build_chat_model, _route_consensus_prompt, _route_context_zh, _run_route_consensus)
from dataset_recommender.llm.llm_client import LLMConfig, load_env_candidates              

CORPUS = AGENT_ROOT / "research" / "route_eval" / "route_corpus.jsonl"
OUT = AGENT_ROOT / "research" / "route_eval" / "route_labels.jsonl"
TZ = timezone(timedelta(hours=8))


def build_model():
    """生产 trial 通道同款配置（thinking=disabled 是 trial 默认形态）。"""
    return _build_chat_model(LLMConfig(
        enable_llm=True, mock_llm=False, provider="trial",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash", timeout=60.0,
        temperature=0.2, max_tokens=8000, thinking=False))


def label_one(row: dict, model, with_retrieval: bool = False) -> dict:
    utt = row["utterance"]
    retrieval = None
    if with_retrieval:
                                                                    
                                           
        from dataset_recommender.agent.turn import rule_match_summary
        retrieval = rule_match_summary(utt)
    context = _route_context_zh(utt, has_results=False, result_total=0,
                                retrieval=retrieval, current_query="", current_filters=None)
    prompt = context + "\n\n----- 用户原话 -----\n" + utt
    votes: list[dict] = []
    route = ""
    for _ in range(3):                     
        route, votes = _run_route_consensus(model, prompt)
        if any(v.get("ok") for v in votes):
            break
        time.sleep(2)
    valid = [v for v in votes if v.get("ok")]
    if len(votes) == 2 and len(valid) == 2 and valid[0]["route"] == valid[1]["route"]:
        decided_by = "agree2"
    elif valid:
        counts: dict[str, int] = {}
        for v in valid:
            counts[v["route"]] = counts.get(v["route"], 0) + 1
        top = max(counts.values())
        winners = [r for r, c in counts.items() if c == top]
        decided_by = "majority3" if (len(winners) == 1 and top >= 2) else "fallback_general"
    else:
        decided_by = "fallback_general"
    return {
        "qid": row["qid"], "utterance": utt, "route_gold": row.get("route_gold"),
        "route_llm": route, "decided_by": decided_by, "votes": votes,
        "model": "deepseek-v4-flash", "ts": datetime.now(TZ).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--limit", type=int, default=0, help="只打前 N 条（冒烟用）")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retrieval-overview", action="store_true",
                        help="对照切片：带生产规则匹配概览行打标（默认输出 route_labels_retrieval.jsonl）")
    args = parser.parse_args()
    if args.retrieval_overview and args.out == str(OUT):
        args.out = str(OUT.with_name("route_labels_retrieval.jsonl"))
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    load_env_candidates(AGENT_ROOT)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing (checked .env)", file=sys.stderr)
        return 2
                                                        
    _ = _route_consensus_prompt()

    rows = [json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])
    todo = [r for r in rows if r["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"corpus={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return 0

    model = build_model()
    written = 0
    t0 = time.monotonic()
    with out_path.open("a", encoding="utf-8", newline="\n") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(label_one, row, model, args.retrieval_overview): row
                    for row in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % 50 == 0:
                    rate = written / max(1e-9, time.monotonic() - t0)
                    eta = (len(todo) - written) / max(rate, 1e-9) / 60
                    print(f"  {written}/{len(todo)} ({rate:.1f}/s, eta {eta:.1f}min)", flush=True)
    print(f"labeled={written} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
