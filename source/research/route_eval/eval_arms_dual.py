                       
"""双臂对比 + 双判官模拟评测（真·小 Jev 批，2026-09-19）。

对各训练臂（qwen3_decider_a/b/...）在**同一金标**上回答三个问题：
1. 单官 τ 闸：ID test / OOD 的 acc、ECE、级联曲线、prelim recall-floor（直读
   各臂 train_report.json，不重算）。
2. 双判官一致闸（binary）：新 decider × 生产 L0（l0_route_binary）在
   binary_labels.jsonl（2393 全语料金标）+ OOD 108 上的覆盖率/一致率——
   口径与现行生产（_l0_route_verdict 双判官段）一致：两官标签一致→采，分歧→弃。
3. prelim 双闸模拟：retrieve 召回 vs skip 正确率，四种规则对比——
   L0 单官（现状）、Jev 单官、双官一致才 skip（保守=现状增强）、双官一致闸
   （一致即采，分歧→retrieve 放行）。

输出 research/route_eval/arms_dual_eval.json + 控制台摘要。

Usage:
  .venv/Scripts/python.exe research/route_eval/eval_arms_dual.py --arms qwen3_decider_a,qwen3_decider_b
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "research" / "route_eval"))

from train_qwen_decider import TASKS, SYSTEM, chat_ids              

EVAL_DIR = AGENT_ROOT / "research" / "route_eval"


def load_arm(arm: str):
    """加载训练臂（adapter+spec），返回 score(task, utt) -> {label, confidence, probs}。"""
    import torch
    from modelscope import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    arm_dir = EVAL_DIR / arm
    spec = json.loads((arm_dir / "decider_spec.json").read_text(encoding="utf-8"))
    base_path = snapshot_download(spec["base_model"])
    tok = AutoTokenizer.from_pretrained(base_path)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, arm_dir / "adapter").to("cuda")
    model.eval()

    @torch.no_grad()
    def score(task: str, utt: str) -> dict:
        tspec = spec["tasks"][task]
        msgs = [{"role": "system", "content": spec["system"]},
                {"role": "user", "content": tspec["prompt"].format(utt=utt)}]
        ids = torch.tensor([chat_ids(tok, msgs)]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits[0, -1, :].float()
        options = list(tspec["options"])
        ids_opt = [int(tspec["option_token_ids"][o]) for o in options]
        temp = max(float(tspec["temperature"]), 1e-3)
        probs = torch.softmax(logits[ids_opt] / temp, dim=-1).cpu().tolist()
        best = max(range(len(probs)), key=lambda k: probs[k])
        return {"label": options[best], "confidence": probs[best],
                "probs": {o: probs[k] for k, o in enumerate(options)}}
    return score, spec


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def dual_stats(pairs: list[tuple[str, str, str]]) -> dict:
    """pairs = (decider_label, l0_label, gold)。一致即采。"""
    cov = [(d, g) for d, l, g in pairs if d == l]
    return {"n": len(pairs), "coverage": len(cov) / max(len(pairs), 1),
            "agree_on_covered": float(np.mean([d == g for d, g in cov])) if cov else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    from dataset_recommender.agent import l0_router as l0
    binary_gold = load_jsonl("binary_labels.jsonl")
    prelim_gold = load_jsonl("prelim_labels.jsonl")
    ood = load_jsonl("ood_route_queries.jsonl")
    print(f"gold: binary={len(binary_gold)} prelim={len(prelim_gold)} ood={len(ood)}",
          flush=True)

                      
    l0_bin = {r["utterance"]: l0.score(l0.ROUTE_BINARY, r["utterance"]) for r in binary_gold}
    l0_bin_ood = {r["utterance"]: l0.score(l0.ROUTE_BINARY, r["utterance"]) for r in ood}
    l0_pre = {r["utterance"]: l0.score(l0.PRELIM_GATE, r["utterance"]) for r in prelim_gold}
    l0_pre_ood = {r["utterance"]: l0.score(l0.PRELIM_GATE, r["utterance"]) for r in ood}
    prelim_tau_l0 = l0.operating_tau(l0.PRELIM_GATE, "", "tau_retrieve", 0.5)

    report: dict = {"arms": {}}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"\n===== {arm} =====", flush=True)
        t0 = time.monotonic()
        score, spec = load_arm(arm)
        entry: dict = {"train_report": json.loads(
            (EVAL_DIR / arm / "train_report.json").read_text(encoding="utf-8"))}

                                         
        pairs, pairs_ood = [], []
        lat = []
        for r in binary_gold:
            utt = r["utterance"]
            t1 = time.perf_counter()
            d = score("binary", utt)
            lat.append((time.perf_counter() - t1) * 1000)
            l = l0_bin[utt]
            if d and l:
                pairs.append((d["label"], l["label"], r["gate"]))
        for r in ood:
            utt = r["utterance"]
            d = score("binary", utt)
            l = l0_bin_ood[utt]
            if d and l:
                pairs_ood.append((d["label"], l["label"], r["gate"]))
        entry["binary_dual_corpus"] = dual_stats(pairs)
        entry["binary_dual_ood"] = dual_stats(pairs_ood)
        lat.sort()
        entry["binary_dual_corpus"]["decider_latency_ms_p50"] = round(lat[len(lat) // 2], 1)
        print(f"binary dual corpus: {entry['binary_dual_corpus']}", flush=True)
        print(f"binary dual OOD   : {entry['binary_dual_ood']}", flush=True)

                                              
        raw_scores: dict = {"corpus": [], "ood": []}

        def prelim_rule_stats(rows, gold_key, l0map, bucket: str):
            rules = {"l0_only": [0, 0, 0], "jev_only": [0, 0, 0],
                     "dual_skip": [0, 0, 0], "dual_agree": [0, 0, 0]}
                                                               
            agg = {k: {"start": 0, "pos": 0, "rec_hit": 0, "neg": 0, "skip_hit": 0}
                   for k in rules}
            jev_tau = float(spec.get("operating_points", {}).get(
                "prelim", {}).get("tau_retrieve", 0.5))
            for r in rows:
                utt = r["utterance"]
                gold = r[gold_key]
                d = score("prelim", utt)
                l = l0map[utt]
                if d is None or l is None:
                    continue
                p_ret_j = float(d["probs"]["retrieve"])
                p_ret_l = float(l["probs"]["retrieve"])
                raw_scores[bucket].append({"utt": utt, "gold": gold,
                                           "l0_p": round(p_ret_l, 6),
                                           "jev_p": round(p_ret_j, 6)})
                dec = {
                    "l0_only": p_ret_l >= prelim_tau_l0,
                    "jev_only": p_ret_j >= jev_tau,
                    "dual_skip": (p_ret_l >= prelim_tau_l0) or (p_ret_j >= jev_tau),
                    "dual_agree": (d["label"] == l["label"] and d["label"] == "skip")
                                  is False,
                }
                                                       
                                                        
                for k, start in dec.items():
                    a = agg[k]
                    a["start"] += int(start)
                    if gold == "retrieve":
                        a["pos"] += 1
                        a["rec_hit"] += int(start)
                    else:
                        a["neg"] += 1
                        a["skip_hit"] += int(not start)
            return {k: {"start_rate": a["start"] / max(a["pos"] + a["neg"], 1),
                        "recall": a["rec_hit"] / max(a["pos"], 1),
                        "skip_correct": a["skip_hit"] / max(a["neg"], 1)}
                    for k, a in agg.items()}

        entry["prelim_rules_corpus"] = prelim_rule_stats(prelim_gold, "gate", l0_pre,
                                                         "corpus")
        entry["prelim_rules_ood"] = prelim_rule_stats(ood, "prelim", l0_pre_ood, "ood")
        entry["prelim_raw_scores"] = raw_scores                   
        print(f"prelim corpus: {json.dumps(entry['prelim_rules_corpus'], ensure_ascii=False)}",
              flush=True)
        print(f"prelim OOD   : {json.dumps(entry['prelim_rules_ood'], ensure_ascii=False)}",
              flush=True)
        entry["eval_minutes"] = round((time.monotonic() - t0) / 60, 1)
        report["arms"][arm] = entry

                                                   
    soft_path = EVAL_DIR / "rlcd_soft_labels.jsonl"
    if soft_path.exists():
        floors: dict = {}
        for line in soft_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("split") != "ood":
                continue
            f = floors.setdefault(d["task"], {"n": 0, "maj_agree": 0, "mean_votes_max": []})
            f["n"] += 1
            f["maj_agree"] += int(d["teacher_majority"] == d["gold"])
            if d["votes"]:
                f["mean_votes_max"].append(max(d["votes"].values()) / max(d["n"], 1))
        report["teacher_ood_floor"] = {
            t: {"n": f["n"], "majority_vs_gold": f["maj_agree"] / max(f["n"], 1),
                "mean_top_share": float(np.mean(f["mean_votes_max"])) if f["mean_votes_max"] else None}
            for t, f in floors.items()}
        print(f"teacher OOD floor: {json.dumps(report['teacher_ood_floor'], ensure_ascii=False)}",
              flush=True)

    out = EVAL_DIR / "arms_dual_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
