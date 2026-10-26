                       
"""Qwen3-Reranker-0.6B 补测：官方 yes/no CausalLM 路径。

embedding_bench 的通用 CrossEncoder 路径对它不适用——该仓库 config 是
Qwen3ForCausalLM、无序列分类头（score.weight 随机初始化，分数无意义）。
官方用法：im_start 模板 + Instruct/Query/Document 包装，取末位 logits 的
yes/no 二项 softmax 概率作为相关分。本脚本复用 embedding_bench 的金标装载
与 ndcg，结果合并写回 _embedding_bench.json。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))

import embedding_bench as eb              

MODEL = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCT = ("Given a Chinese natural-language query, retrieve relevant public "
            "single-cell omics dataset metadata records")
MAX_LEN = 2048
BATCH = 8

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
          "based on the Query and the Instruct provided. Note that the answer can "
          "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    gold = eb.load_gold()
    need = {u for g in gold for u in g["labels"]}
    doc_texts = eb.load_doc_texts(need)
    gold = [{"query_id": g["query_id"], "query": g["query"],
             "labels": {u: s for u, s in g["labels"].items() if u in doc_texts}}
            for g in gold]
    gold = [g for g in gold if len(g["labels"]) >= 2]
    print(f"gold: {len(gold)} queries, {sum(len(g['labels']) for g in gold)} pairs",
          flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    tid_y = tok.convert_tokens_to_ids("yes")
    tid_n = tok.convert_tokens_to_ids("no")
    pre = tok(PREFIX, return_tensors=None)["input_ids"]
    suf = tok(SUFFIX, return_tensors=None)["input_ids"]

    pairs, idx = [], []
    for gi, g in enumerate(gold):
        q = f"<Instruct>: {INSTRUCT}\n<Query>: {g['query']}\n"
        for uid in g["labels"]:
            pairs.append(q + f"<Document>: {doc_texts[uid]}")
            idx.append((gi, uid))

    t0 = time.time()
    per_q = {}
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        enc = tok(chunk, padding=False, truncation="longest_first",
                  return_attention_mask=False,
                  max_length=MAX_LEN - len(pre) - len(suf))
        ids = [pre + e + suf for e in enc["input_ids"]]
        pad = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**pad).logits[:, -1, :]
        s = torch.nn.functional.log_softmax(logits[:, [tid_n, tid_y]], dim=1)[:, 1]
        for (gi, uid), v in zip(idx[i:i + BATCH], s.tolist()):
            per_q.setdefault(gi, {})[uid] = v
        done = i + len(chunk)
        if done % (BATCH * 10) < BATCH or done == len(pairs):
            print(f"  {done}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)

    n3, n10, p3 = [], [], []
    for gi, g in enumerate(gold):
        order = sorted(g["labels"], key=lambda u: -per_q[gi][u])
        v3, v10 = eb.ndcg(order, g["labels"], 3), eb.ndcg(order, g["labels"], 10)
        if v3 is not None:
            n3.append(v3)
        if v10 is not None:
            n10.append(v10)
        p3.append(sum(1 for u in order[:3] if g["labels"].get(u, 0) >= 2) / 3)
    res = {"ndcg3": round(sum(n3) / len(n3), 4),
           "ndcg10": round(sum(n10) / len(n10), 4),
           "p3_rel2": round(sum(p3) / len(p3), 4), "n_queries": len(n3),
           "path": "official_yesno_causallm", "max_len": MAX_LEN,
           "instruction": INSTRUCT,
           "elapsed_s": round(time.time() - t0)}
    rp = DATA / "_embedding_bench.json"
    report = json.loads(rp.read_text(encoding="utf-8"))
    report["models"][f"ce:{MODEL}"] = res
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ce:{MODEL}:", res, flush=True)


if __name__ == "__main__":
    main()
