                       
"""RLCD 校准蒸馏软标签生成（真·小 Jev 批，2026-09-19）。

真 Jev 的第二层（第一层=单前向受限 softmax 形态，已由 train_qwen_decider.py 复刻）：
让教师模型对**学生将见到的同一个 prompt** 做 n 次温度采样，把经验频率分布当作
软标签（KL 蒸馏目标）——学生学到的不只是「答案是什么」，还有「教师有多确定」，
即 RLCD 的校准内生思想（校准不再是事后温度膏药）。

- 教师：deepseek-v4-flash（thinking disabled），temperature=1.0，每 (行,任务) n=20 票
- 任务：binary 写闸 + prelim 检索闸（route3 维持 L0+aug 兜底通道，不蒸）
- 行集：combined_all.jsonl 的 train 切（与 train_qwen_decider.py 同一 SEED/分层，
  cal/test 只留金标不蒸）+ OOD 108 条（split=ood，只作诊断，永不入训练）
- 断点续跑：追加写 JSONL，重启跳过已完成的 (qid, task)
- prompt 单源：直接 import train_qwen_decider.TASKS/SYSTEM（改 prompt 即自动换分布）

输出 research/route_eval/rlcd_soft_labels.jsonl：
  {qid, split, task, utterance, n, votes:{opt:k}, invalid, gold, teacher_majority, ts}

Usage:
  .venv/Scripts/python.exe research/route_eval/gen_rlcd_soft_labels.py --limit 2 --n 4  # 冒烟
  .venv/Scripts/python.exe research/route_eval/gen_rlcd_soft_labels.py                  # 全量
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "research" / "route_eval"))

from train_qwen_decider import EVAL_DIR, SEED, SYSTEM, TASKS              

OUT = EVAL_DIR / "rlcd_soft_labels.jsonl"
TZ = timezone(timedelta(hours=8))
GEN_TASKS = ("binary", "prelim")
GOLD_KEY = {"binary": "gate", "prelim": "prelim"}

_lock = threading.Lock()


def parse_vote(text: str, options: list[str]) -> str | None:
    t = (text or "").strip().lower()
    for o in options:
        if o in t:
            return o
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="每 (行,任务) 采样票数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 行（冒烟）")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    from dataset_recommender.llm.llm_client import load_env_candidates
    load_env_candidates(AGENT_ROOT)
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                           base_url="https://api.deepseek.com/v1", timeout=60)

    rows = [json.loads(line) for line in (EVAL_DIR / "combined_all.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    y_split = np.array([TASKS["route3"]["options"].index(r["route"]) for r in rows])
    idx_tr, _ = train_test_split(np.arange(len(rows)), test_size=0.4,
                                 random_state=SEED, stratify=y_split)
    train_rows = []
    for i in sorted(idx_tr.tolist()):
        r = dict(rows[i])
        r["qid"] = f"row{i:05d}"                                     
        train_rows.append((r, "train"))
    ood_rows = [json.loads(line) for line in (EVAL_DIR / "ood_route_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    work_rows = train_rows + [(r, "ood") for r in ood_rows]
    if args.limit:
        work_rows = work_rows[: args.limit]

    done: set[tuple[str, str]] = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["task"]))

    jobs = []
    for r, split in work_rows:
        for task in GEN_TASKS:
            if (r["qid"], task) not in done:
                jobs.append((r, split, task))
    total_votes = len(jobs) * args.n
    print(f"rows={len(work_rows)} jobs={len(jobs)} votes={total_votes} "
          f"(已完成 {len(done)} 组跳过)", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return 0

    def run_job(row: dict, split: str, task: str) -> dict:
        spec = TASKS[task]
        user = spec["prompt"].format(utt=row["utterance"])
        votes: Counter = Counter()
        invalid = 0
        for _ in range(args.n):
            for attempt in (0, 1):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-v4-flash", temperature=1.0, max_tokens=8,
                        extra_body={"thinking": {"type": "disabled"}},
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user}])
                    v = parse_vote(resp.choices[0].message.content, spec["options"])
                    if v is None:
                        invalid += 1
                    else:
                        votes[v] += 1
                    break
                except Exception:
                    if attempt == 1:
                        invalid += 1
                    else:
                        time.sleep(1.5)
        gold = row[GOLD_KEY[task]]
        majority = votes.most_common(1)[0][0] if votes else ""
        return {"qid": row["qid"], "split": split, "task": task,
                "utterance": row["utterance"], "n": args.n,
                "votes": dict(votes), "invalid": invalid, "gold": gold,
                "teacher_majority": majority,
                "ts": datetime.now(TZ).isoformat(timespec="seconds")}

    t0 = time.monotonic()
    finished = 0
    agree = 0
    with OUT.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_job, r, split, task) for r, split, task in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            with _lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            finished += 1
            if rec["teacher_majority"] == rec["gold"]:
                agree += 1
            if finished % 50 == 0 or finished == len(jobs):
                rate = finished / max(time.monotonic() - t0, 1e-9)
                eta = (len(jobs) - finished) / max(rate, 1e-9)
                print(f"  {finished}/{len(jobs)} 组 教师多数与金标一致率(至今)="
                      f"{agree / finished:.4f} ETA {eta / 60:.1f} min", flush=True)

    print(f"done in {(time.monotonic() - t0) / 60:.1f} min -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
