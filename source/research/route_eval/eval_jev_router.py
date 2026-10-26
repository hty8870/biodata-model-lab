                       
"""Jev（TypeSafe systemone）真实 API 分流评测（2026-09-19，用户授权 key 后启动）。

对照组 = 生产 LLM 共识标签（label_route_consensus.py 产出）。输入构造与打标脚本逐字一致：
  state = agent_exec._route_context_zh 的中性空现场 + "\\n\\n----- 用户原话 -----\\n" + utterance
路由定义（criteria）逐字拷贝自 prompts/route_consensus.md（Jev 无 system 通道，
prompt 的路由定义只能经 instructions/criteria 传入）；脚本启动时做漂移校验——
若 prompt 文件已不含本脚本内的定义文本则报错退出，强制同步更新。

模式：
  默认      三分类 choice（search/action/general）→ jev_labels.jsonl
  --binary  二元 noul（是否需要写盘/下载类工具）→ jev_binary_labels.jsonl
评分：
  --score --gold route_labels_v2.jsonl [--gold-field route_llm]
  指标：一致率、逐类混淆、ECE（max-prob 10 桶）、级联曲线（覆盖率-准确率）、
  延迟分位、token/成本（$0.042/M input，output 免费）。

Usage:
  .venv/Scripts/python.exe research/route_eval/eval_jev_router.py --limit 5   # 冒烟
  .venv/Scripts/python.exe research/route_eval/eval_jev_router.py             # 全量
  .venv/Scripts/python.exe research/route_eval/eval_jev_router.py --score --gold research/route_eval/route_labels_v2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent.agent_exec import _route_context_zh              
from dataset_recommender.llm.llm_client import load_env_candidates              

CORPUS = AGENT_ROOT / "research" / "route_eval" / "route_corpus.jsonl"
OUT_3WAY = AGENT_ROOT / "research" / "route_eval" / "jev_labels.jsonl"
OUT_BIN = AGENT_ROOT / "research" / "route_eval" / "jev_binary_labels.jsonl"
PROMPT_MD = AGENT_ROOT / "prompts" / "route_consensus.md"
API_URL = "https://api.typesafe.ai/v1/systemone"
TZ = timezone(timedelta(hours=8))

                                                              
_INSTR_3WAY = (
    "你是单细胞公开数据集检索工具的分流器：只负责判断这句话该走哪条处理路线，"
    "不执行任何动作。给你用户原话、会话现场（当前查询/生效条件/初步检索命中数），"
    "判断它该走 search、action 还是 general。")
_CRITERIA_3WAY = {
    "search": "用户在找数据 / 改检索条件 / 贴编号直链（GSE…、E-MTAB…、DOI）/ "
              "问「有没有…数据」——目标是最优化检索结果。"
              "例：「找找人类肺癌的单细胞数据」「换成小鼠的再查一遍」「GSE123456 这个数据集有吗」",
    "action": "用户要动作——打包下载、引文、投稿材料、联网搜库、检查更新、入库、"
              "管护自己上传的外部库（导入/删除/恢复/清点）、问库容/系统状态"
              "（数据库现在多少条、库里有多少数据集）。"
              "例：「检查 10x 有没有更新，有新增的人类肺数据就搜来入库」「把前 5 条打包下载」"
              "「数据库现在多少条了，报个数」",
    "general": "闲聊、概念问答、检索+动作混合诉求、或拿不准——兜底路线（全集工具，安全地板）。"
               "例：「什么是单细胞测序」「找肺癌数据然后打包前 3 条」",
}
_INSTR_BIN = (
    "单细胞公开数据集检索工具的网关：判断处理这句话是否需要用到会产生本地下载或数据写入"
    "后果的工具（打包下载、引文导出、投稿材料生成、外部库导入/删除/恢复/同步更新）。"
    "只需要检索、重排、查看结果、对比、兼容/FAIR 检查、查库容、检查更新、联网搜索，"
    "以及纯问答/闲聊，都算不需要。")
_CRITERIA_BIN = {"true": "需要写盘/下载/入库类工具", "false": "只读工具或纯问答即可"}


def _drift_guard() -> str:
    text = PROMPT_MD.read_text(encoding="utf-8")
    norm = lambda s: "".join(s.split()).replace("**", "")
    body = norm(text)
    for frag in ("用户在找数据/改检索条件/贴编号直链", "问库容/系统状态",
                 "闲聊、概念问答、检索+动作混合诉求"):
        if norm(frag) not in body:
            raise SystemExit(f"prompt 漂移校验失败：{PROMPT_MD.name} 已不含「{frag}」，"
                             "请同步更新本脚本的 criteria 后再跑")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_state(utt: str) -> str:
    context = _route_context_zh(utt, has_results=False, result_total=0,
                                retrieval=None, current_query="", current_filters=None)
    return context + "\n\n----- 用户原话 -----\n" + utt


def jev_call(session, state: str, binary: bool) -> dict:
    if binary:
        question = {"type": "noul", "instructions": _INSTR_BIN, "criteria": _CRITERIA_BIN}
        key = "needs_write"
    else:
        question = {"type": "choice", "instructions": _INSTR_3WAY, "criteria": _CRITERIA_3WAY}
        key = "route"
    body = {"state": state, "model": "jev-latest", "questions": {key: question}}
    last_err = ""
    for attempt in range(6):
        try:
            resp = session.post(API_URL, json=body, timeout=30)
            if resp.status_code in (429, 529) or resp.status_code >= 500:
                last_err = f"http {resp.status_code}"
                time.sleep((2 ** attempt) + random.random())
                continue
            resp.raise_for_status()
            data = resp.json()
            ans = data["answers"][key]
            usage = data.get("usage") or {}
            return {"answer": ans, "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens")}
        except Exception as exc:                    
            last_err = str(exc)[:200]
            time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"jev 调用连续失败：{last_err}")


def label_one(row: dict, session, binary: bool, prompt_sha: str) -> dict:
    utt = row["utterance"]
    t0 = time.monotonic()
    out = jev_call(session, build_state(utt), binary)
    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    ans = out["answer"]
    rec = {"qid": row["qid"], "utterance": utt,
           "latency_ms": latency_ms, "input_tokens": out["input_tokens"],
           "output_tokens": out["output_tokens"], "model": "jev-latest",
           "prompt_sha256": prompt_sha, "ts": datetime.now(TZ).isoformat(timespec="seconds")}
    if binary:
        rec["jev_write_p"] = ans["noul"]
    else:
        rec["jev_route"] = ans["choice"]
        rec["probabilities"] = ans["probabilities"]
        rec["confidence"] = ans.get("confidence")
    return rec


def run(args) -> int:
    import requests
    prompt_sha = _drift_guard()
    rows = [json.loads(line) for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out_path = Path(args.out) if args.out else (OUT_BIN if args.binary else OUT_3WAY)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])
    todo = [r for r in rows if r["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"corpus={len(rows)} done={len(done)} todo={len(todo)} mode={'binary' if args.binary else '3way'}",
          flush=True)
    if not todo:
        return 0
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                            "Content-Type": "application/json"})
    written = 0
    t0 = time.monotonic()
    with out_path.open("a", encoding="utf-8", newline="\n") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(label_one, row, session, args.binary, prompt_sha): row
                    for row in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % 50 == 0 or written == len(todo):
                    rate = written / max(1e-9, time.monotonic() - t0)
                    eta = (len(todo) - written) / max(rate, 1e-9) / 60
                    print(f"  {written}/{len(todo)} ({rate:.1f}/s, eta {eta:.1f}min)", flush=True)
    print(f"labeled={written} -> {out_path}", flush=True)
    return 0


def score(args) -> int:
    pred_path = Path(args.pred) if args.pred else (OUT_BIN if args.binary else OUT_3WAY)
    preds = {json.loads(l)["qid"]: json.loads(l)
             for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    golds = {json.loads(l)["qid"]: json.loads(l)
             for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()}
    field = args.gold_field
    pairs = []
    for qid, p in preds.items():
        g = golds.get(qid)
        if not g or g.get(field) in (None, ""):
            continue
        if args.binary:
            conf = max(p["jev_write_p"], 1 - p["jev_write_p"])
            pred = bool(p["jev_write_p"] >= 0.5)
            gold = g[field] in (True, 1, "true", "True", "write")
        else:
            conf = max(p["probabilities"].values())
            pred = p["jev_route"]
            gold = g[field]
        pairs.append({"qid": qid, "conf": conf, "pred": pred, "gold": gold,
                      "correct": pred == gold, "latency_ms": p["latency_ms"],
                      "input_tokens": p.get("input_tokens") or 0})
    n = len(pairs)
    if not n:
        print("no overlap between pred and gold", file=sys.stderr)
        return 2
    acc = sum(r["correct"] for r in pairs) / n
          
    per_class: dict[str, dict[str, int]] = {}
    for r in pairs:
        c = per_class.setdefault(str(r["gold"]), {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(r["correct"])
                                   
    bins = [[] for _ in range(10)]
    for r in pairs:
        bins[min(9, int(r["conf"] * 10))].append(r)
    ece = 0.0
    reliability = []
    for b in bins:
        if not b:
            continue
        bc = sum(x["conf"] for x in b) / len(b)
        ba = sum(x["correct"] for x in b) / len(b)
        ece += len(b) / n * abs(bc - ba)
        reliability.append({"conf": round(bc, 4), "acc": round(ba, 4), "n": len(b)})
               
    cascade = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995]:
        covered = [r for r in pairs if r["conf"] >= tau]
        if not covered:
            continue
        cascade.append({"tau": tau, "coverage": round(len(covered) / n, 4),
                        "acc": round(sum(r["correct"] for r in covered) / len(covered), 4)})
    lat = sorted(r["latency_ms"] for r in pairs)
    pct = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)
    tokens_in = sum(r["input_tokens"] for r in pairs)
    report = {
        "pred": str(pred_path), "gold": args.gold, "gold_field": field, "n": n,
        "agreement": round(acc, 4),
        "per_class": {k: {"n": v["n"], "acc": round(v["correct"] / v["n"], 4)}
                      for k, v in sorted(per_class.items())},
        "ece": round(ece, 4), "reliability": reliability, "cascade": cascade,
        "latency_ms": {"p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "max": lat[-1]},
        "input_tokens_total": tokens_in,
        "cost_usd": round(tokens_in * 0.042 / 1e6, 4),
    }
    report_path = pred_path.with_name(pred_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"-> {report_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--out", default="")
    parser.add_argument("--pred", default="", help="评分模式下预测文件（默认按 mode 取）")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--binary", action="store_true", help="二元 noul：是否需要写盘工具")
    parser.add_argument("--score", action="store_true", help="评分模式（不调用 API）")
    parser.add_argument("--gold", default="", help="评分用金标文件")
    parser.add_argument("--gold-field", default="route_llm")
    args = parser.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    load_env_candidates(AGENT_ROOT)
    if args.score:
        if not args.gold:
            print("--score 需要 --gold", file=sys.stderr)
            return 2
        return score(args)
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY missing (checked .env)", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
