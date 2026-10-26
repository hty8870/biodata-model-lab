                       
"""二元写盘闸金标打标（二元写闸设计评估，2026-09-19）。

评估「分流改成二元：加载写盘工具与否」的金标构造。投票协议复刻生产 route_consensus
的结构（并行 0.0/0.8 → 分歧加投 0.5 → 多数决），但判定定义是研究草稿（本脚本内嵌，
不进生产 prompts/）——生产尚无此函数，故此处是实验镜像而非复用。

判定定义（写盘 = 需要会产生本地下载或数据写入后果的工具）：
  write：打包下载、导出引文、生成投稿材料、外部库入库/删除/恢复/同步更新
  read ：找数据/改条件/查看重排/对比/兼容/FAIR/查库容/检查更新(只看不同步)/
         联网搜库(只搜不入库)/概念问答/闲聊；拿不准 → read（只读是安全默认）

系统提示词真源 = 生产 `prompts/route_write_gate.md`（2026-09-19 分流三改批转正，
binary_labels.jsonl 打标时该文件与下列逐字一致——脚本内置 SHA256 漂移校验：
prompt 一改即报错，必须先按新口径重打标再评估，绝不用旧金标评新 prompt）。

输入  research/route_eval/route_corpus.jsonl
输出  research/route_eval/binary_labels.jsonl（断点续跑）：
  {qid, utterance, gate, decided_by, votes, model, ts}

Usage:
  .venv/Scripts/python.exe research/route_eval/label_binary_gate.py --limit 6  # 冒烟
  .venv/Scripts/python.exe research/route_eval/label_binary_gate.py            # 全量
"""
from __future__ import annotations

import argparse
import hashlib
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
    _build_chat_model, _route_context_zh, _route_write_gate_prompt)
from dataset_recommender.llm.llm_client import LLMConfig, load_env_candidates              

CORPUS = AGENT_ROOT / "research" / "route_eval" / "route_corpus.jsonl"
OUT = AGENT_ROOT / "research" / "route_eval" / "binary_labels.jsonl"
TZ = timezone(timedelta(hours=8))

                                                                       
_LABELING_PROMPT_SHA256 = "01ff952fbac282a0bb897e76c2611ab1a8742d04d505f7391574efcb61c3648f"


def _system_prompt() -> str:
    prompt = _route_write_gate_prompt().strip()
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if digest != _LABELING_PROMPT_SHA256:
        raise SystemExit(
            "route_write_gate.md 已漂移（sha256 不符）——金标是按旧口径打的，"
            "必须先重打标再评估；确认重打后更新脚本内 _LABELING_PROMPT_SHA256。")
    return prompt

_VALID = ("write", "read")


def build_model():
    return _build_chat_model(LLMConfig(
        enable_llm=True, mock_llm=False, provider="trial",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash", timeout=60.0,
        temperature=0.2, max_tokens=8000, thinking=False))


def _one_vote(model, prompt: str, temperature: float) -> dict:
    model.temperature = temperature
    try:
        resp = model.invoke([
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt}])
        text = (resp.content or "").strip() if hasattr(resp, "content") else str(resp)
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
        gate = str(obj.get("gate", "")).strip().lower()
        if gate in _VALID:
            return {"ok": True, "gate": gate, "reason": str(obj.get("reason", ""))[:60],
                    "temperature": temperature}
        return {"ok": False, "gate": "", "raw": text[:120], "temperature": temperature}
    except Exception as exc:
        return {"ok": False, "gate": "", "error": str(exc)[:120], "temperature": temperature}


def _run_binary_consensus(model, prompt: str) -> tuple[str, list[dict]]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(_one_vote, model, prompt, 0.0)
        f1 = pool.submit(_one_vote, model, prompt, 0.8)
        votes = [f0.result(), f1.result()]
    valid = [v for v in votes if v["ok"]]
    if len(valid) == 2 and valid[0]["gate"] != valid[1]["gate"]:
        votes.append(_one_vote(model, prompt, 0.5))
        valid = [v for v in votes if v["ok"]]
    if not valid:
        return "read", votes
    counts: dict[str, int] = {}
    for v in valid:
        counts[v["gate"]] = counts.get(v["gate"], 0) + 1
    top = max(counts.values())
    winners = [g for g, c in counts.items() if c == top]
    if len(winners) == 1 and top >= 2:
        return winners[0], votes
    if len(valid) == 2 and valid[0]["gate"] == valid[1]["gate"]:
        return valid[0]["gate"], votes
    return "read", votes                        


def label_one(row: dict, model) -> dict:
    utt = row["utterance"]
    context = _route_context_zh(utt, has_results=False, result_total=0,
                                retrieval=None, current_query="", current_filters=None)
    prompt = context + "\n\n----- 用户原话 -----\n" + utt
    gate, votes = "", []
    for _ in range(3):
        gate, votes = _run_binary_consensus(model, prompt)
        if any(v.get("ok") for v in votes):
            break
        time.sleep(2)
    valid = [v for v in votes if v.get("ok")]
    if len(votes) == 2 and len(valid) == 2 and valid[0]["gate"] == valid[1]["gate"]:
        decided_by = "agree2"
    elif valid:
        counts: dict[str, int] = {}
        for v in valid:
            counts[v["gate"]] = counts.get(v["gate"], 0) + 1
        top = max(counts.values())
        winners = [g for g, c in counts.items() if c == top]
        decided_by = "majority3" if (len(winners) == 1 and top >= 2) else "fallback_read"
    else:
        decided_by = "fallback_read"
    return {"qid": row["qid"], "utterance": utt, "gate": gate,
            "decided_by": decided_by, "votes": votes,
            "model": "deepseek-v4-flash", "ts": datetime.now(TZ).isoformat(timespec="seconds")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    load_env_candidates(AGENT_ROOT)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing (checked .env)", file=sys.stderr)
        return 2
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
            futs = {pool.submit(label_one, row, model): row for row in todo}
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
