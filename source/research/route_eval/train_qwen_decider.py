                       
"""Jev 开源复刻：Qwen 蒸馏决策器（单前向并行选项打分，2026-09-19）。

形状 = System One 的贫农复刻：chat prompt 一次前向 → 答案位 logits →
只对**选项 token 集合**做 softmax → 类型化决策 + 概率分布 + 置信度。

训练两种模式（--kl-weight/--ce-weight 组合）：
- 硬 CE（kl=0,ce=1，旧默认）：在自家 LLM 金标上做答案位单 token 交叉熵——
  只复刻形态（第一层）。
- 真 RLCD 校准蒸馏（kl=1,ce=0）：以 gen_rlcd_soft_labels.py 采的教师频率
  分布为软标签，答案位受限 log-softmax 上最小化 KL(教师‖学生)——
  学生连「教师有多确定」一起学，校准内生（第二层）。无软标签的样本自动
  回退硬 CE 并在报告里记覆盖率。

LoRA(q,v, r=8) ——适配器 ~MB 级可入库，基础模型走 fetch 模式（与 embedding
模型同纪律）。任务集合可用 --tasks 裁剪（默认三任务合一：route3/binary/prelim；
真 RLCD 臂只用 binary+prelim），行级共享切分（60/20/20，seed 20260919，
按 route3 分层——全任务 train/cal/test 行互斥，杜绝跨任务泄漏）。

基座可 --base 换（默认 Qwen/Qwen2.5-0.5B-Instruct；真 RLCD 臂 Qwen/Qwen3-1.7B）。
Qwen3 混合思考模板自动 enable_thinking=False，老模板回退。

校准：cal 上对受限 softmax 的 NLL 拟合每任务温度（与 L0 同一方法论；
RLCD 臂预期温度≈1，即「不需要膏药」本身就是验收点之一）。
评测：ID test（acc/ECE/级联曲线/prelim recall-floor τ）+ OOD 108 条 + GPU 单例时延。

Usage:
  .venv/Scripts/python.exe research/route_eval/train_qwen_decider.py [--epochs 2] [--smoke]
  .venv/Scripts/python.exe research/route_eval/train_qwen_decider.py \
      --base Qwen/Qwen3-1.7B --tasks binary,prelim \
      --soft-labels research/route_eval/rlcd_soft_labels.jsonl \
      --kl-weight 1 --ce-weight 0 --out-name qwen3_decider_b
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

AGENT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = AGENT_ROOT / "research" / "route_eval"
OUT_DIR = EVAL_DIR / "qwen_decider"                        

SEED = 20260919
BASE = "Qwen/Qwen2.5-0.5B-Instruct"                             

SYSTEM = "你是决策器，只回答给定选项中的一个词。"
TASKS: dict[str, dict] = {
    "binary": {
        "options": ["write", "read"],
        "prompt": ("句子：{utt}\n任务：判断处理这句话是否需要用到会产生本地下载或数据写入"
                   "后果的工具。\nwrite=打包下载数据集、导出引文、生成投稿材料、把数据入库/"
                   "删除/恢复/同步进用户的外部库（含联网搜来入库、检查更新并同步）；\n"
                   "read=其余一切：找数据、改检索条件、查看/重排/对比/兼容/FAIR、查库容、"
                   "检查更新（只看不同步）、联网搜（只搜不入库）、清点外部库、取消/否定先前"
                   "动作、概念问答、闲聊。拿不准判 read。\n只答 write 或 read"),
        "label_key": "gate",
    },
    "route3": {
        "options": ["search", "action", "general"],
        "prompt": ("句子：{utt}\n任务：判断这句话走哪条路线。\n"
                   "search=找数据/改检索条件/贴编号直链(GSE…)/问有没有某数据（含口语、拼音、"
                   "英文、错字、指代改条件如「换成小鼠的」）；\n"
                   "action=要动作：打包下载、引文、投稿材料、联网搜库、检查更新、入库/删除/"
                   "恢复/同步、管护外部库、问库容（库里多少条）；\n"
                   "general=闲聊、概念问答、检索+动作混合诉求、或拿不准。\n"
                   "只答 search、action 或 general"),
        "label_key": "route",
    },
    "binary3": {
        "options": ["write", "read", "escalate"],
        "prompt": ("句子：{utt}\n任务：判断处理这句话是否需要用到会产生本地下载或数据写入"
                   "后果的工具。\nwrite=打包下载数据集、导出引文、生成投稿材料、把数据入库/"
                   "删除/恢复/同步进用户的外部库（含联网搜来入库、检查更新并同步）；\n"
                   "read=其余一切：找数据、改检索条件、查看/重排/对比/兼容/FAIR、查库容、"
                   "检查更新（只看不同步）、联网搜（只搜不入库）、清点外部库、取消/否定先前"
                   "动作、概念问答、闲聊；\nescalate=拿不准——两可、意图不明、判了也没把握"
                   "（交更强的外援判断，不要乱猜）。\n只答 write、read 或 escalate"),
        "label_key": "gate",
    },
    "prelim": {
        "options": ["retrieve", "skip"],
        "prompt": ("句子：{utt}\n任务：判断处理这句话是否需要先跑一遍本地数据集检索。\n"
                   "retrieve=找数据/问有没有某数据/改检索条件（含口语、拼音、英文、错字、"
                   "「换成人的再查一遍」这类指代改条件）；\n"
                   "skip=管护动作（下载/引文/检查更新/入库/同步/清点）、库容问答、概念问答、"
                   "闲聊、取消先前动作。\n只答 retrieve 或 skip"),
        "label_key": "prelim",
    },
}


def chat_ids(tok, msgs: list[dict]) -> list[int]:
    """Qwen3 混合思考模板需 enable_thinking=False；老模板不支持该 kwarg 则回退。
    transformers 5.x 返回 Encoding（需取 input_ids 展平），4.x 返回 list[int]——归一。"""
    try:
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if isinstance(enc, list):
        return enc
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_examples(rows: list[dict], tasks: tuple[str, ...],
                   soft: dict[tuple[int, str], dict] | None = None) -> list[dict]:
    exs = []
    for i, r in enumerate(rows):
        for task in tasks:
            spec = TASKS[task]
            dist = None
            if soft:
                votes = soft.get((i, task))
                if votes:
                    total = sum(votes.get(o, 0) for o in spec["options"])
                    if total:
                        dist = [votes.get(o, 0) / total for o in spec["options"]]
            exs.append({"rid": i, "task": task, "utterance": r["utterance"],
                        "label": r[spec["label_key"]], "dist": dist})
    return exs


class DeciderDataset(Dataset):
    """每条样本 = prompt_ids + [answer_id] + [eos]，另带答案位/选项 id/软分布。

    pos = 答案 token 的前一位（其 logits 预测答案）；dist 为教师频率分布
    （无软标签时全零 + has_dist=False，KL 项自动归零）。
    """

    def __init__(self, exs: list[dict], tok, option_ids: dict[str, dict[str, int]]):
        self.items = []
        for ex in exs:
            spec = TASKS[ex["task"]]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": spec["prompt"].format(utt=ex["utterance"])}]
            prefix = chat_ids(tok, msgs)
            ans = option_ids[ex["task"]][ex["label"]]
            ids = prefix + [ans, tok.eos_token_id]
            labels = [-100] * len(prefix) + [ans, tok.eos_token_id]
            opt_ids = [option_ids[ex["task"]][o] for o in spec["options"]]
            has_dist = ex["dist"] is not None
            dist = ex["dist"] if has_dist else [0.0] * len(opt_ids)
            self.items.append((ids, labels, len(prefix) - 1, opt_ids, dist, has_dist))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(batch, pad_id: int):
    """左填充训练批（2026-09-20 C 臂 OOM 修复）：左填后答案/EOS 恒为最后两位，
    配合 forward(logits_to_keep=3) 只算末三位 lm_head——全序列 logits 不再
    物化（Qwen3.5 的 26 万词表下 (B,L,V) 曾单批 ~4.2GB 直接 OOM）。
    返回 pos = 预测答案位的输入索引（width-3）；labels 末两位恒为 [ans, eos]。"""
    width = max(len(ids) for ids, *_ in batch)
    kmax = max(len(o) for _, _, _, o, _, _ in batch)
    input_ids, labels, attn, poss, opt_ids, dists, has = [], [], [], [], [], [], []
    for ids, lab, pos, o, d, h in batch:
        pad = width - len(ids)
        input_ids.append([pad_id] * pad + ids)
        labels.append([-100] * pad + lab)
        attn.append([0] * pad + [1] * len(ids))
        poss.append(width - 3)
        opt_ids.append(o + [-1] * (kmax - len(o)))
        dists.append(d + [0.0] * (kmax - len(d)))
        has.append(h)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn),
            torch.tensor(poss), torch.tensor(opt_ids), torch.tensor(dists),
            torch.tensor(has, dtype=torch.bool))


@torch.no_grad()
def option_probs(model, tok, exs: list[dict], option_ids, device, batch_size=16):
    """单前向并行选项打分：返回 {task: (probs[N,K], gold[N])}。"""
    model.eval()
    task_names = sorted({e["task"] for e in exs})
    per_task: dict[str, dict] = {t: {"probs": [], "gold": []} for t in task_names}
    for lo in range(0, len(exs), batch_size):
        chunk = exs[lo:lo + batch_size]
        prefixes = []
        for ex in chunk:
            spec = TASKS[ex["task"]]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": spec["prompt"].format(utt=ex["utterance"])}]
            prefixes.append(chat_ids(tok, msgs))
        width = max(len(p) for p in prefixes)
                                      
        input_ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p
                                  for p in prefixes]).to(device)
        attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p)
                             for p in prefixes]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            try:
                logits = model(input_ids=input_ids, attention_mask=attn,
                               logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(input_ids=input_ids,
                               attention_mask=attn).logits[:, -1, :].float()
        for j, ex in enumerate(chunk):
            ids = [option_ids[ex["task"]][o] for o in TASKS[ex["task"]]["options"]]
            p = torch.softmax(logits[j, ids], dim=-1).cpu().numpy()
            per_task[ex["task"]]["probs"].append(p)
            per_task[ex["task"]]["gold"].append(
                TASKS[ex["task"]]["options"].index(ex["label"]))
    return {t: (np.array(d["probs"]), np.array(d["gold"])) for t, d in per_task.items()}


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo_, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo_) & (conf <= hi) if lo_ > 0 else (conf >= lo_) & (conf <= hi)
        if m.sum():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def fit_temp_on_logits(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t: float) -> float:
        z = logits / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())
    grid = np.linspace(0.3, 6.0, 40)
    best = min(grid, key=nll)
    lo, hi = max(0.1, best - 0.4), best + 0.4
    gr = (5 ** 0.5 - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(40):
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    return float((lo + hi) / 2)


def cascade_curve(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    order = np.argsort(-conf)
    correct = (pred[order] == y[order]).astype(float)
    n = len(order)
    out = []
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        k = max(1, int(round(cov * n)))
        out.append({"coverage": cov, "accuracy": float(correct[:k].mean()),
                    "tau_at_edge": float(conf[order[k - 1]])})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="只跑 40 步验证管线")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--out-name", default="qwen_decider")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--tasks", default="binary,route3,prelim",
                        help="逗号分隔任务子集（默认全三任务）")
    parser.add_argument("--soft-labels", default="",
                        help="gen_rlcd_soft_labels.py 产物路径（split=train 行生效）")
    parser.add_argument("--kl-weight", type=float, default=0.0,
                        help="软标签 KL 权重（真 RLCD 臂 =1）")
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="硬 CE 权重（真 RLCD 臂 =0）")
    args = parser.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = args.base
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    for t in tasks:
        assert t in TASKS, t

                                 
    try:
        from modelscope import snapshot_download
        base_path = snapshot_download(base)
    except Exception:
        base_path = base
    tok = AutoTokenizer.from_pretrained(base_path)
    tok.pad_token = tok.eos_token

                                                    
    option_ids: dict[str, dict[str, int]] = {}
    for task in tasks:
        spec = TASKS[task]
        ids = {}
        for o in spec["options"]:
            enc = tok.encode(o, add_special_tokens=False)
            ids[o] = enc[0]
        assert len(set(ids.values())) == len(ids), (task, ids)
        option_ids[task] = ids
        print(f"options[{task}]: {ids}", flush=True)

                                           
    soft: dict[tuple[int, str], dict] | None = None
    if args.soft_labels:
        soft = {}
        for line in Path(args.soft_labels).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("split") != "train":
                continue
            soft[(int(d["qid"][3:]), d["task"])] = d["votes"]
        print(f"soft labels: {len(soft)} (行,任务) 组", flush=True)

    rows = [json.loads(line) for line in (EVAL_DIR / "combined_all.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
                                                
    y_split = np.array([TASKS["route3"]["options"].index(r["route"]) for r in rows])
    idx = np.arange(len(rows))
    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        idx, y_split, test_size=0.4, random_state=SEED, stratify=y_split)
    idx_cal, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    all_exs = build_examples(rows, tasks, soft)
    ex_tr = [e for e in all_exs if e["rid"] in set(idx_tr.tolist())]
    ex_cal = [e for e in all_exs if e["rid"] in set(idx_cal.tolist())]
    ex_te = [e for e in all_exs if e["rid"] in set(idx_te.tolist())]
    if args.smoke:
        ex_tr = ex_tr[: args.batch * 40]
    n_soft = sum(1 for e in ex_tr if e["dist"] is not None)
    print(f"rows={len(rows)} train/cal/test={len(idx_tr)}/{len(idx_cal)}/{len(idx_te)} "
          f"exs={len(ex_tr)}/{len(ex_cal)}/{len(ex_te)} "
          f"soft覆盖率={n_soft}/{len(ex_tr)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16).to(device)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds = DeciderDataset(ex_tr, tok, option_ids)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = len(dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    model.train()
    t0 = time.monotonic()
    step = 0
    for ep in range(args.epochs):
        for input_ids, labels, attn, poss, opt_ids, dists, has in dl:
            input_ids, labels, attn = (input_ids.to(device), labels.to(device),
                                       attn.to(device))
            poss, opt_ids, dists, has = (poss.to(device), opt_ids.to(device),
                                         dists.to(device), has.to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                try:
                    logits = model(input_ids=input_ids, attention_mask=attn,
                                   logits_to_keep=3).logits
                except TypeError:
                    logits = model(input_ids=input_ids,
                                   attention_mask=attn).logits[:, -3:, :]
                                                               
            loss = 0.0
            if args.ce_weight:
                ce_logits = logits[:, :2, :].reshape(-1, logits.size(-1)).float()
                ce_targets = labels[:, -2:].reshape(-1)
                loss = loss + args.ce_weight * torch.nn.functional.cross_entropy(
                    ce_logits, ce_targets)
            if args.kl_weight:
                                                             
                ans_logits = logits[:, 0, :].float()
                tok_logits = ans_logits.gather(1, opt_ids.clamp(min=0))
                tok_logits = tok_logits.masked_fill(opt_ids < 0, float("-inf"))
                logp = torch.log_softmax(tok_logits, dim=-1)
                kl = -(dists * logp).sum(dim=-1)
                denom = has.sum().clamp(min=1)
                loss = loss + args.kl_weight * (kl * has).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                rate = step / max(time.monotonic() - t0, 1e-9)
                print(f"  ep{ep} step {step}/{steps} loss={loss.item():.4f} "
                      f"({rate:.1f} it/s)", flush=True)
    print(f"train done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)

                     
    mode = ("rlcd" if args.kl_weight and not args.ce_weight else
            "mix" if args.kl_weight else "hard")
    report: dict = {"base": base, "epochs": args.epochs, "lr": args.lr,
                    "seed": SEED, "split": "shared-row 60/20/20 by route3",
                    "tasks": list(tasks), "training_mode": mode,
                    "kl_weight": args.kl_weight, "ce_weight": args.ce_weight,
                    "soft_coverage": n_soft / max(len(ex_tr), 1)}
    cal_probs = option_probs(model, tok, ex_cal, option_ids, device)
    te_probs = option_probs(model, tok, ex_te, option_ids, device)
    temps: dict[str, float] = {}
    for task in tasks:
        pc, yc = cal_probs[task]
        logits_c = np.log(np.clip(pc, 1e-12, 1.0))
        temp = fit_temp_on_logits(logits_c, yc)
        temps[task] = temp
        pt, yt = te_probs[task]
        pt_cal = torch.softmax(torch.tensor(np.log(np.clip(pt, 1e-12, 1.0)) / temp),
                               dim=-1).numpy()
        acc = float((pt_cal.argmax(axis=1) == yt).mean())
        report[task] = {
            "temp": temp, "test_acc": acc,
            "ece_raw": ece(pt, yt), "ece_cal": ece(pt_cal, yt),
            "cascade": cascade_curve(pt_cal, yt),
        }
        print(f"[{task}] acc={acc:.4f} ece {report[task]['ece_raw']:.4f}->"
              f"{report[task]['ece_cal']:.4f} temp={temp:.2f}", flush=True)
                                                    
        if task == "prelim":
            pos = TASKS["prelim"]["options"].index("retrieve")
            pc_cal = torch.softmax(torch.tensor(
                np.log(np.clip(pc, 1e-12, 1.0)) / temp), dim=-1).numpy()
            pos_scores = np.sort(pc_cal[yc == pos, pos])
            tau = float(pos_scores[min(int(np.floor(0.005 * len(pos_scores))),
                                       len(pos_scores) - 1)])
            start = pt_cal[:, pos] >= tau
            is_pos = yt == pos
            report["prelim"]["tau_retrieve"] = tau
            report["prelim"]["test_recall"] = float(start[is_pos].mean())
            report["prelim"]["test_skip_correct"] = float((~start[~is_pos]).mean())
            print(f"  prelim τ_retrieve={tau:.4f} test_recall="
                  f"{report['prelim']['test_recall']:.4f}", flush=True)

                       
    ood_rows = [json.loads(line) for line in (EVAL_DIR / "ood_route_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    ood_exs = []
    key_map_all = {"gate": "binary", "route": "route3", "prelim": "prelim"}
    key_map = {f: t for f, t in key_map_all.items() if t in tasks}
    for i, r in enumerate(ood_rows):
        for field, task in key_map.items():
            ood_exs.append({"rid": i, "task": task, "utterance": r["utterance"],
                            "label": r[field]})
    ood_probs = option_probs(model, tok, ood_exs, option_ids, device)
    ood_report: dict = {}
    for field, task in key_map.items():
        probs, gold = ood_probs[task]
        p_cal = torch.softmax(torch.tensor(
            np.log(np.clip(probs, 1e-12, 1.0)) / temps[task]), dim=-1).numpy()
        acc = float((p_cal.argmax(axis=1) == gold).mean())
        entry: dict = {"ood_acc": acc}
        if task == "prelim" and "tau_retrieve" in report["prelim"]:
            pos = TASKS["prelim"]["options"].index("retrieve")
            start = p_cal[:, pos] >= report["prelim"]["tau_retrieve"]
            is_pos = gold == pos
            entry["ood_recall"] = float(start[is_pos].mean())
            entry["ood_skip_correct"] = float((~start[~is_pos]).mean())
        ood_report[task] = entry
        print(f"[OOD {task}] acc={acc:.4f} {entry}", flush=True)
    report["ood"] = ood_report

                           
    probe = [{"rid": 0, "task": tasks[0], "utterance": "找找人类肺癌的单细胞数据",
              "label": rows[0][TASKS[tasks[0]]["label_key"]]}]
    option_probs(model, tok, probe * 4, option_ids, device)      
    t0 = time.perf_counter()
    n_probe = 30
    option_probs(model, tok, probe * n_probe, option_ids, device, batch_size=1)
    ms = (time.perf_counter() - t0) * 1000 / n_probe
    report["latency_ms_per_decision_gpu"] = round(ms, 1)
    print(f"latency ~{ms:.1f} ms/decision (gpu, batch=1)", flush=True)

                  
    out_dir = EVAL_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    export = {
        "format": "decider-router-v1",
        "base_model": base,
        "adapter": "adapter",
        "system": SYSTEM,
        "training_mode": mode,
        "tasks": {t: {"options": TASKS[t]["options"], "prompt": TASKS[t]["prompt"],
                      "option_token_ids": option_ids[t], "temperature": temps[t]}
                  for t in tasks},
        "operating_points": {
            t: ({"kind": "recall_floor",
                 "tau_retrieve": report["prelim"].get("tau_retrieve", 0.5)}
                if t == "prelim" else
                {"kind": "cascade", "tau": next(
                    (c["tau_at_edge"] for c in report[t]["cascade"]
                     if c["coverage"] >= 0.9), 0.99)})
            for t in tasks
        },
        "metrics": report,
    }
    (out_dir / "decider_spec.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_dir / 'adapter'} + decider_spec.json + train_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
