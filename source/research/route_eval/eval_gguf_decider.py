                       
"""GGUF 量化决策器评测（真·小 Jev 批，2026-09-19）。

用 llama-server（本地子进程，/completion 裸 prompt 端点，n_probs=20）对量化后的
GGUF 工件跑与训练侧同一评测：ID test（acc/ECE/级联/prelim recall-floor）+ OOD 108
+ 决策时延分布。prompt 用脚本内手工拼的 chat 文本（与训练时
enable_thinking=False 渲染结果**逐字节一致**），绕开 server 模板 kwarg 差异。

选项概率 = 首生成 token 的 top-20 logprob 里捞各选项 token，受限 softmax；
选项缺席 top-20 按最低可见 logprob-5 兜底（量化把正确选项挤出 top-20 本身就是
要暴露的信号，如实计入）。

Usage:
  .venv/Scripts/python.exe research/route_eval/eval_gguf_decider.py \
      --gguf research/route_eval/gguf/qwen3_decider_b-Q4_K_M.gguf [--threads 2] [--limit 0]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "research" / "route_eval"))
from train_qwen_decider import (EVAL_DIR, SEED, SYSTEM, TASKS,              
                                cascade_curve, ece, fit_temp_on_logits)

LLAMA_DIR = EVAL_DIR / "_llamacpp" / "cpu-x64"
SERVER = LLAMA_DIR / "llama-server.exe"

                                                           
                                                                
PROMPT_FMT = ("<|im_start|>system\n{system}<|im_end|>\n"
              "<|im_start|>user\n{user}<|im_end|>\n"
              "<|im_start|>assistant\n{think_block}")


def build_prompt(utt: str, task: str, think_block: str) -> str:
    user = TASKS[task]["prompt"].format(utt=utt)
    return PROMPT_FMT.format(system=SYSTEM, user=user, think_block=think_block)


def wait_server(port: int, proc: subprocess.Popen, timeout: float = 180.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("llama-server health timeout")


def top_logprobs(port: int, prompt: str, n_probs: int = 20) -> list[tuple[str, float]]:
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                       "top_k": 1, "n_probs": n_probs, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read().decode())
    cps = obj.get("completion_probabilities") or []
    if not cps:
        return []
    cp = cps[0]
                                                                            
    if isinstance(cp.get("probs"), list):
        return [(str(p.get("tok_str")), float(np.log(max(p.get("prob"), 1e-45))))
                for p in cp["probs"]]
    if isinstance(cp.get("top_logprobs"), list):
        return [(str(p.get("token")), float(p.get("logprob"))) for p in cp["top_logprobs"]]
    return []


def option_prob_from_top(top: list[tuple[str, float]], options: list[str]) -> np.ndarray:
    lps = {}
    floor = min((lp for _, lp in top), default=-20.0) - 5.0
    for o in options:
        hit = [lp for tok, lp in top if tok.strip().lower() == o]
        lps[o] = max(hit) if hit else floor
    z = np.array([lps[o] for o in options])
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def eval_set(model_port: int, exs: list[dict], think_block: str,
             lat: list[float] | None = None) -> dict:
    per_task: dict[str, dict] = {}
    for ex in exs:
        per_task.setdefault(ex["task"], {"probs": [], "gold": []})
    for i, ex in enumerate(exs):
        prompt = build_prompt(ex["utterance"], ex["task"], think_block)
        t0 = time.perf_counter()
        top = top_logprobs(model_port, prompt)
        ms = (time.perf_counter() - t0) * 1000
        if lat is not None:
            lat.append(ms)
        spec = TASKS[ex["task"]]
        p = option_prob_from_top(top, spec["options"])
        per_task[ex["task"]]["probs"].append(p)
        per_task[ex["task"]]["gold"].append(spec["options"].index(ex["label"]))
        if (i + 1) % 100 == 0:
            print(f"    eval {i + 1}/{len(exs)}", flush=True)
    return {t: (np.array(d["probs"]), np.array(d["gold"])) for t, d in per_task.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--threads", type=int, default=2, help="模拟服务器 2 vCPU")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-think-block", action="store_true",
                    help="prompt 不拼空 think 块（Qwen2.5 系）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--from-report", default="",
                    help="给训练臂 train_report.json：跳过 cal 评测，温度/τ 直取报告"
                         "（量化档与 torch 侧共用同一工作点，省一半评测时间）")
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    think_block = "" if args.no_think_block else "<think>\n\n</think>\n\n"

    rows = [json.loads(line) for line in (EVAL_DIR / "combined_all.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    y_split = np.array([TASKS["route3"]["options"].index(r["route"]) for r in rows])
    idx = np.arange(len(rows))
    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        idx, y_split, test_size=0.4, random_state=SEED, stratify=y_split)
    idx_cal, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    tasks = ("binary", "prelim")

    def mk_exs(idxs):
        return [{"rid": int(i), "task": t, "utterance": rows[i]["utterance"],
                 "label": rows[i][TASKS[t]["label_key"]]}
                for i in sorted(idxs.tolist()) for t in tasks]

    ex_cal, ex_te = mk_exs(idx_cal), mk_exs(idx_te)
    ood_rows = [json.loads(line) for line in (EVAL_DIR / "ood_route_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    ex_ood = [{"rid": i, "task": t, "utterance": r["utterance"],
               "label": r[{"binary": "gate", "prelim": "prelim"}[t]]}
              for i, r in enumerate(ood_rows) for t in tasks]
    if args.limit:
        ex_cal, ex_te, ex_ood = ex_cal[: args.limit], ex_te[: args.limit], ex_ood[: args.limit]

    proc = subprocess.Popen(
        [str(SERVER), "-m", args.gguf, "--port", str(args.port),
         "-c", "2048", "-t", str(args.threads), "-ngl", "0", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_server(args.port, proc)
        print(f"server up (threads={args.threads})", flush=True)
        lat: list[float] = []
        if args.from_report:
            print("== cal skipped (--from-report) ==", flush=True)
            cal = None
        else:
            print("== cal ==", flush=True)
            cal = eval_set(args.port, ex_cal, think_block)
        print("== test ==", flush=True)
        te = eval_set(args.port, ex_te, think_block, lat)
        print("== ood ==", flush=True)
        ood = eval_set(args.port, ex_ood, think_block, lat)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    report: dict = {"gguf": args.gguf, "threads": args.threads, "tag": args.tag,
                    "from_report": args.from_report or None}
    ref: dict = {}
    if args.from_report:
        ref = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
    temps: dict[str, float] = {}
    for task in tasks:
        if args.from_report:
            temp = float(ref[task]["temp"])
        else:
            pc, yc = cal[task]
            temp = fit_temp_on_logits(np.log(np.clip(pc, 1e-12, 1.0)), yc)
        temps[task] = temp
        pt, yt = te[task]
        pt_cal = np.exp(np.log(np.clip(pt, 1e-12, 1.0)) / temp)
        pt_cal = pt_cal / pt_cal.sum(axis=1, keepdims=True)
        acc = float((pt_cal.argmax(axis=1) == yt).mean())
        entry = {"temp": temp, "test_acc": acc,
                 "ece_raw": ece(pt, yt), "ece_cal": ece(pt_cal, yt),
                 "cascade": cascade_curve(pt_cal, yt)}
        if task == "prelim":
            pos = TASKS["prelim"]["options"].index("retrieve")
            if args.from_report:
                tau = float(ref["prelim"]["tau_retrieve"])
            else:
                pc_cal = np.exp(np.log(np.clip(pc, 1e-12, 1.0)) / temp)
                pc_cal = pc_cal / pc_cal.sum(axis=1, keepdims=True)
                pos_scores = np.sort(pc_cal[yc == pos, pos])
                tau = float(pos_scores[min(int(np.floor(0.005 * len(pos_scores))),
                                           len(pos_scores) - 1)])
            start = pt_cal[:, pos] >= tau
            entry["tau_retrieve"] = tau
            entry["test_recall"] = float(start[yt == pos].mean())
            entry["test_skip_correct"] = float((~start[yt != pos]).mean())
        report[task] = entry
        print(f"[{task}] acc={acc:.4f} ece {entry['ece_raw']:.4f}->"
              f"{entry['ece_cal']:.4f} temp={temp:.2f}", flush=True)
    for task in tasks:
        probs, gold = ood[task]
        p_cal = np.exp(np.log(np.clip(probs, 1e-12, 1.0)) / temps[task])
        p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)
        entry = {"ood_acc": float((p_cal.argmax(axis=1) == gold).mean())}
        if task == "prelim" and "tau_retrieve" in report["prelim"]:
            pos = TASKS["prelim"]["options"].index("retrieve")
            start = p_cal[:, pos] >= report["prelim"]["tau_retrieve"]
            entry["ood_recall"] = float(start[gold == pos].mean())
            entry["ood_skip_correct"] = float((~start[gold != pos]).mean())
        report.setdefault("ood", {})[task] = entry
        print(f"[OOD {task}] {entry}", flush=True)
    lat.sort()
    if lat:
        report["latency_ms"] = {"p50": round(lat[len(lat) // 2], 1),
                                "p99": round(lat[int(len(lat) * 0.99)], 1),
                                "n": len(lat)}
        print(f"latency p50={report['latency_ms']['p50']}ms "
              f"p99={report['latency_ms']['p99']}ms (threads={args.threads})", flush=True)
    out = EVAL_DIR / f"gguf_eval_{args.tag or Path(args.gguf).stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
