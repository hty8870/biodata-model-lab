                       
"""route_consensus 分流器训练语料合成（System-One 分流器 MVP，2026-09-19）。

三类路由语料：
  search  —— 收割既有合成集（synth_heldout 151 + ood 605），全是检索意图
  action  —— 管护动作：真实种子（curate_examples/candidates）+ 模板扩写 + LLM 改写
  general —— 概念问答/闲聊/越界/模糊：模板 + LLM 改写

机械诚实过滤（与生产投票面分布对齐——被机械预闸命中的句子在生产上直定 general、
LLM 投票永远看不到，故不进入训练分布）：
  · 任何类：`hybrid_intent_gate` 命中 → 丢弃（并计数留痕）
  · search：`action_hit` 丢弃（ contested 句不进检索类）
  · action：必须 `action_hit`（构造保证）；同子句内夹带检索信号不丢（如
    「联网搜一下10x有没有肺癌的数据」是合法动作句，金标由 LLM 共识票裁决检验）
  · general：`action_hit`/`search_hit` 均 False（纯净兜底类）

输出 research/route_eval/route_corpus.jsonl：
  {qid, utterance, route_gold, source, signals:{action_hit, search_hit}}

追加安全：按 utterance 归一化去重；--qid-prefix 区分增补批次（如 route2）。

Usage:
  .venv/Scripts/python.exe research/route_eval/synth_route_corpus.py
  .venv/Scripts/python.exe research/route_eval/synth_route_corpus.py --no-llm   # 只跑零 LLM 部分
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent.hybrid_policy import (              
    action_hit, hybrid_intent_gate, search_hit)
from dataset_recommender.llm.llm_client import load_env_candidates              

OUT_DIR = AGENT_ROOT / "research" / "route_eval"
HELDOUT = AGENT_ROOT / "research" / "ltr_data" / "synth_heldout_xscore.jsonl"
OOD = AGENT_ROOT / "research" / "ood_eval" / "ood_queries.jsonl"
CURATE_SEEDS = (AGENT_ROOT / ".userdata" / "curate_examples.jsonl",
                AGENT_ROOT / ".userdata" / "curate_example_candidates.jsonl")

GEN_MODEL = "deepseek-v4-flash"                                 
GEN_URL = "https://api.deepseek.com/v1/chat/completions"
TARGET_ACTION = 800
TARGET_GENERAL = 800

SOURCES = ("10x", "ArrayExpress", "GEO", "CELLxGENE", "HCA", "ENA")

                                                                             
                                   
                                                                             

ACTION_TEMPLATES = [
          
    "检查{s}有没有更新", "看看{s}是否有更新", "{s}有更新吗", "帮我检查下{s}的更新情况",
    "{s}最近有没有更新", "检查一下{s}有没有更新", "帮我看看{s}有没有更新", "{s}是不是有更新",
           
    "把{s}上新的数据同步入库", "把{s}新增的数据集收进库里", "更新一下{s}的数据",
    "{s}的新数据帮我入库", "把{s}最新的数据集纳入库中", "同步一下{s}的最新数据",
                                             
    "联网搜一下{s}", "在线搜{s}上的新数据集", "上网搜{s}看看有什么新东西",
    "联网搜一下{s}有没有新公开的数据",
          
    "数据库现在状态怎么样", "库里一共有多少条数据", "汇报一下库容", "清点一下数据库",
    "库里现有多少条数据", "现在库里有多少条", "汇报数据库的当前状态",
                                  
    "把前{n}条打包下载", "下载第{n}条数据集", "把前{n}条结果打个包", "导出前{n}条的引文",
    "把选中的几条打包", "前{n}条的参考文献帮我导出", "把这批结果整理成包",
               
    "上传我自己的数据集", "把我上次上传的删掉", "删除我上传的外部库", "从回收站找回那条数据",
    "把我传错的那条撤销删除", "回收站里那条恢复一下", "清空回收站",
]

GENERAL_TEMPLATES = [
          
    "什么是{t}", "{t}的原理是什么", "{t}和{t2}有什么区别", "为什么要用{t}",
    "{t}适合做什么研究", "{t}的优缺点是什么", "能不能讲讲{t}", "{t}大概是怎么工作的",
            
    "你好", "在吗", "谢谢", "你是谁", "你能做什么", "这个工具怎么用",
    "你的数据是从哪来的", "这个库多久更新一次", "你都能查哪些来源", "好的我明白了",
                   
    "帮我写一篇关于{d}的综述", "把这段英文翻译成中文", "帮我分析我自己测的数据",
    "给我画一个火山图", "帮我写实验方案", "帮我设计引物", "帮我改论文", "帮我做差异分析",
            
    "嗯", "好的", "这样吗", "OK", "行", "再看看", "有点意思", "原来如此",
]

TECHS = ("单细胞测序", "空间转录组", "scRNA-seq", "snRNA-seq", "VDJ 测序", "ATAC-seq",
         "UMI", "FFPE 样本", "Visium", "Xenium", "10x 平台", "批次效应", "细胞注释",
         "降维聚类", "marker 基因", "单细胞转录组")
DISEASES_ZH = ("肺癌", "乳腺癌", "阿尔茨海默病", "肺纤维化", "肝癌", "结肠炎",
               "糖尿病", "白血病", "胶质瘤", "心肌梗死")

ACTION_LLM_PROMPT = """你是数据构造助手。为「单细胞公开数据集检索工具」构造用户原话语料，类别=管护动作（action）：用户要系统**执行一个操作**（不是检索数据本身）。
动作家族与例子：
- 检查更新：检查10x有没有更新 / 看看ArrayExpress是否有更新
- 入库同步：把10x新增的数据集同步入库 / 更新一下GEO的数据
- 联网搜库：联网搜一下CELLxGENE / 在线搜GEO上的新数据
- 清点库容：数据库现在有多少条 / 汇报一下库的状态
- 下载打包：把前5条打包下载 / 下载第2个数据集
- 引文导出：导出这几条的引文 / 把前3条的参考文献打包
- 上传删除恢复：上传我自己的数据集 / 把我上次上传的删掉 / 从回收站找回那条

再造 {n} 条互不相同的动作请求。要求：真实用户口吻（口语、简短、可带语气词与错别字）；
每条只含动作意图，**不要**混入「找/推荐/有没有…数据」这类检索诉求；
不要编号，只输出 JSON 字符串数组。"""

GENERAL_LLM_PROMPT = """你是数据构造助手。为「单细胞公开数据集检索工具」构造用户原话语料，类别=general（非检索非动作）：
闲聊 / 概念问答（单细胞、空间转录组、测序技术原理）/ 越界请求（写综述、翻译、分析用户自己的数据、画图、改论文）/ 模糊表达（嗯、好的、再看看）。
例子：什么是单细胞测序 / 你好 / 帮我写一篇文献综述 / 嗯嗯好的 / 空间转录组和单细胞测序有什么区别 / 你们的数据来源有哪些

再造 {n} 条互不相同的句子。要求：真实用户口吻；**不要**出现「找/搜/检索/有没有…数据/
下载/打包/入库/更新数据/删除/上传」这类检索或动作诉求；不要编号，只输出 JSON 字符串数组。"""


def norm(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def call_llm(prompt: str, api_key: str, max_tokens: int = 4096,
             temperature: float = 0.9, retries: int = 3) -> str:
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    req = urllib.request.Request(GEN_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            return str(data["choices"][0]["message"].get("content") or "")
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def parse_str_array(text: str) -> list[str]:
    start = text.find("[")
    if start < 0:
        return []
    try:
        arr, _ = json.JSONDecoder().raw_decode(text, start)
    except Exception:
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]


def signals(utt: str) -> dict:
    return {"action_hit": action_hit(utt), "search_hit": search_hit(utt)}


def keep(utt: str, route: str, stats: dict) -> bool:
    """机械诚实过滤：唯一真源。返回是否保留，并把丢弃原因计进 stats。"""
    def drop(reason: str) -> bool:
        stats[reason] = stats.get(reason, 0) + 1
        return False
    if not (1 <= len(utt) <= 160):
        return drop("len")
    if hybrid_intent_gate(utt):
        return drop("gate")
    sig = signals(utt)
    if route == "search" and sig["action_hit"]:
        return drop("search_actionhit")
    if route == "action" and not sig["action_hit"]:
        return drop("action_nohit")
    if route == "general" and (sig["action_hit"] or sig["search_hit"]):
        return drop("general_signal")
    return True


def harvest_search(seen: set[str], stats: dict) -> list[dict]:
    rows: list[dict] = []
    for path, tag in ((HELDOUT, "heldout"), (OOD, "ood")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            utt = str(obj.get("query") or "").strip()
            src = tag if tag == "heldout" else f"ood:{obj.get('ood_class')}"
            if norm(utt) in seen or not keep(utt, "search", stats):
                stats["search_drop"] = stats.get("search_drop", 0) + 1
                continue
            seen.add(norm(utt))
            rows.append({"utterance": utt, "route_gold": "search", "source": src})
    return rows


def action_candidates(rng: random.Random) -> list[tuple[str, str]]:
    """(utterance, source) 候选流：真实种子 → 模板。"""
    out: list[tuple[str, str]] = []
    seen_local: set[str] = set()
    for path in CURATE_SEEDS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt = str(json.loads(line).get("utterance") or "").strip()
            if utt and norm(utt) not in seen_local:
                seen_local.add(norm(utt))
                out.append((utt, "curate_seed"))
    for tpl in ACTION_TEMPLATES:
        variants = [tpl]
        if "{s}" in tpl:
            variants = [tpl.replace("{s}", s) for s in SOURCES]
        elif "{n}" in tpl:
            variants = [tpl.replace("{n}", str(n)) for n in (2, 3, 5, 10)]
        for utt in variants:
            if norm(utt) not in seen_local:
                seen_local.add(norm(utt))
                out.append((utt, "template"))
    rng.shuffle(out)
    return out


def general_candidates(rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen_local: set[str] = set()
    for tpl in GENERAL_TEMPLATES:
        variants = [tpl]
        if "{t2}" in tpl:
            pairs = [(t1, t2) for t1 in TECHS for t2 in TECHS if t1 != t2]
            rng.shuffle(pairs)
            variants = [tpl.replace("{t2}", t2).replace("{t}", t1)
                        for t1, t2 in pairs[:12]]
        elif "{t}" in tpl:
            variants = [tpl.replace("{t}", t) for t in TECHS]
        elif "{d}" in tpl:
            variants = [tpl.replace("{d}", d) for d in DISEASES_ZH]
        for utt in variants:
            if norm(utt) not in seen_local:
                seen_local.add(norm(utt))
                out.append((utt, "template"))
    rng.shuffle(out)
    return out


def llm_expand(prompt: str, n_total: int, batch: int, api_key: str,
               concurrency: int) -> list[str]:
    calls = max(1, (n_total + batch - 1) // batch)
    got: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(call_llm, prompt.format(n=batch), api_key)
                for _ in range(calls)]
        for fut in as_completed(futs):
            got.extend(parse_str_array(fut.result()))
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--qid-prefix", default="route")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--target-action", type=int, default=TARGET_ACTION)
    parser.add_argument("--target-general", type=int, default=TARGET_GENERAL)
    parser.add_argument("--no-llm", action="store_true", help="只跑收割+模板，不做 LLM 扩写")
    parser.add_argument("--out", default=str(OUT_DIR / "route_corpus.jsonl"))
    args = parser.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    load_env_candidates(AGENT_ROOT)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.no_llm and not api_key:
        print("OPENAI_API_KEY missing (checked .env)", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    counters: dict[str, int] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                seen.add(norm(obj["utterance"]))
                counters[obj["route_gold"]] = counters.get(obj["route_gold"], 0) + 1
    print(f"existing: {counters} ({len(seen)} unique)", flush=True)

    stats: dict[str, int] = {}
    rows: list[dict] = []

                            
    rows.extend(harvest_search(seen, stats))
    n_search = len(rows)
    print(f"[search] harvested={n_search} drops={stats.get('search_drop', 0)}", flush=True)

                                     
    n_action = 0
    for utt, src in action_candidates(rng):
        if n_action >= args.target_action:
            break
        if norm(utt) in seen or not keep(utt, "action", stats):
            continue
        seen.add(norm(utt))
        rows.append({"utterance": utt, "route_gold": "action", "source": src})
        n_action += 1
    print(f"[action] seeds+templates kept={n_action} stats={stats}", flush=True)
    if not args.no_llm and n_action < args.target_action:
        need = args.target_action - n_action
        for utt in llm_expand(ACTION_LLM_PROMPT, int(need * 1.6), 10, api_key, args.concurrency):
            if n_action >= args.target_action:
                break
            if norm(utt) in seen or not keep(utt, "action", stats):
                continue
            seen.add(norm(utt))
            rows.append({"utterance": utt, "route_gold": "action", "source": "llm_expand"})
            n_action += 1
        print(f"[action] after llm_expand={n_action}", flush=True)

                                   
    n_general = 0
    for utt, src in general_candidates(rng):
        if n_general >= args.target_general:
            break
        if norm(utt) in seen or not keep(utt, "general", stats):
            continue
        seen.add(norm(utt))
        rows.append({"utterance": utt, "route_gold": "general", "source": src})
        n_general += 1
    print(f"[general] templates kept={n_general}", flush=True)
    if not args.no_llm and n_general < args.target_general:
        need = args.target_general - n_general
        for utt in llm_expand(GENERAL_LLM_PROMPT, int(need * 1.6), 10, api_key, args.concurrency):
            if n_general >= args.target_general:
                break
            if norm(utt) in seen or not keep(utt, "general", stats):
                continue
            seen.add(norm(utt))
            rows.append({"utterance": utt, "route_gold": "general", "source": "llm_expand"})
            n_general += 1
        print(f"[general] after llm_expand={n_general}", flush=True)

    idx: dict[str, int] = {}
    with out_path.open("a", encoding="utf-8", newline="\n") as out:
        for row in rows:
            route = row["route_gold"]
            qid = f"{args.qid_prefix}-{route}-{counters.get(route, 0) + idx.get(route, 0):04d}"
            idx[route] = idx.get(route, 0) + 1
            row["qid"] = qid
            row["signals"] = signals(row["utterance"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"written={len(rows)} (search={n_search} action={n_action} general={n_general})",
          flush=True)
    print(f"filter stats={stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
