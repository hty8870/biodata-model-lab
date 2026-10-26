                       
"""OOD 导向数据增强集（2026-09-19，零 LLM，标签由构造继承/确定）。

目的：让 L0 与 Qwen 复刻在**完全相同的增广数据**上训练，公平回答
「小 transformer 是否比 char-ngram 线性模型泛化更强」。OOD 108 条评测集
（ood_route_queries.jsonl）**永不入训练**。

三类增广（全部确定性）：
A. 拼音转写：从既有 2393 语料抽样，pypinyin 整句转写（ASCII 原样保留），
   三标签（route/gate/prelim）按 utterance 从三份金标表继承。
B. 口语化变换：检索句填充（前缀/后缀/替换口语动词），标签同 A 继承。
C. 模板构造（与评测集句式刻意不同）：英文检索/动作、边界（联网搜不入库/
   只检查/取消）、否定、库容变体——标签由构造确定。

输出 research/route_eval/aug_route_queries.jsonl（{qid, utterance, gate,
prelim, route, hybrid, src}）
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pypinyin import Style, lazy_pinyin

EVAL_DIR = Path(__file__).resolve().parent
OUT = EVAL_DIR / "aug_route_queries.jsonl"
rng = random.Random(20260919)

TOPICS_ZH = ["人类肺癌", "乳腺癌", "小鼠脑", "斑马鱼", "人类肾", "肝癌", "结直肠癌",
             "小鼠心脏", "人类胰腺", "皮肤", "血液", "骨髓", "人类卵巢", "前列腺癌"]
TOPICS_EN = ["human lung cancer", "breast cancer", "mouse brain", "zebrafish",
             "human kidney", "liver cancer", "colon cancer", "mouse heart",
             "human pancreas", "skin", "blood", "bone marrow", "human ovary",
             "prostate cancer"]
SOURCES = ["10x", "ENCODE", "GEO"]


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def inherit_maps() -> dict[str, dict]:
    """utterance → {route, gate, prelim} 三表合并（以 utterance 为键）。"""
    routes = {r["utterance"]: r["route_llm"] for r in load_jsonl("route_labels_v2.jsonl")}
    gates = {r["utterance"]: r["gate"] for r in load_jsonl("binary_labels.jsonl")}
    prelims = {r["utterance"]: r["gate"] for r in load_jsonl("prelim_labels.jsonl")}
    out = {}
    for utt, route in routes.items():
        if utt in gates and utt in prelims:
            out[utt] = {"route": route, "gate": gates[utt], "prelim": prelims[utt]}
    return out


def to_pinyin(text: str) -> str:
    """整句转拼音（ASCII 段原样保留、中文转连写拼音词，词间空格）。"""
    tokens = lazy_pinyin(text, style=Style.NORMAL, errors=lambda s: list(s))
    return " ".join(t for t in tokens if t.strip())


COLLOQUIAL_PREFIX = ["那个，", "诶，", "哎，", "麻烦，", "你好，"]
COLLOQUIAL_SUFFIX = ["呗", "啊", "呀", "哈", "哇", ""]
COLLOQUIAL_REWRITE = [("帮我找", "帮我瞅瞅有没有"), ("找找", "看看有没有"),
                      ("搜一下", "搜搜看"), ("帮我搜", "帮我看看有没有")]

EN_SEARCH_PATTERNS = [
    "find {t} scRNA-seq data", "any single cell data on {t}?",
    "do you have single cell RNA-seq for {t}", "looking for {t} single cell datasets",
    "search {t} scRNA data please", "is there {t} single cell data in the library",
]
EN_ACTION_PATTERNS = [
    ("download top {n} datasets for me", "write", "skip", "action"),
    ("export the citations of these results", "write", "skip", "action"),
    ("sync {s} updates into my library", "write", "skip", "action"),
    ("check {s} for new releases", "read", "skip", "action"),
    ("search online for {t} and import into library", "write", "skip", "action"),
    ("search online for {t}, just look, do not import", "read", "skip", "action"),
    ("how many datasets do we have now", "read", "skip", "action"),
]
ZH_BOUNDARY_PATTERNS = [
    ("联网搜{t}，先别入库", "read", "skip", "action"),
    ("联网搜{t}数据帮我入库", "write", "skip", "action"),
    ("检查{s}有没有更新，只看不同步", "read", "skip", "action"),
    ("检查{s}更新，有新增就同步进来", "write", "skip", "action"),
    ("刚才说的入库先取消了", "read", "skip", "action"),
    ("不用同步了，先这样吧", "read", "skip", "action"),
    ("把回收站里那条{t}的恢复一下", "write", "skip", "action"),
    ("帮我清点一下我上传的外部库", "read", "skip", "action"),
    ("不用打包了，直接给我看结果就行", "read", "retrieve", "search"),
    ("别下载，就查查有没有{t}的", "read", "retrieve", "search"),
    ("现在库里数据有多少了", "read", "skip", "action"),
    ("最近库里有没有新增数据", "read", "skip", "action"),
]


def main() -> int:
    inherit = inherit_maps()
    rows: list[dict] = []

                                               
    by_route: dict[str, list[str]] = {}
    for utt, lab in inherit.items():
        by_route.setdefault(lab["route"], []).append(utt)
    picked = (rng.sample(by_route["search"], min(260, len(by_route["search"])))
              + rng.sample(by_route["action"], min(60, len(by_route["action"])))
              + rng.sample(by_route["general"], min(60, len(by_route["general"]))))
    for utt in picked:
        lab = inherit[utt]
        rows.append({"utterance": to_pinyin(utt), "gate": lab["gate"],
                     "prelim": lab["prelim"], "route": lab["route"],
                     "hybrid": False, "src": "pinyin"})

                                
    for utt in rng.sample(by_route["search"], min(220, len(by_route["search"]))):
        lab = inherit[utt]
        v = utt
        for old, new in COLLOQUIAL_REWRITE:
            if old in v:
                v = v.replace(old, new, 1)
                break
        if rng.random() < 0.6:
            v = rng.choice(COLLOQUIAL_PREFIX) + v
        suf = rng.choice(COLLOQUIAL_SUFFIX)
        if suf and not v.endswith(("吗", "没", "呢", "吧")):
            v = v + suf
        if v != utt:
            rows.append({"utterance": v, "gate": lab["gate"], "prelim": lab["prelim"],
                         "route": lab["route"], "hybrid": False, "src": "colloquial"})

             
    for t_en in TOPICS_EN:
        for pat in rng.sample(EN_SEARCH_PATTERNS, 3):
            rows.append({"utterance": pat.format(t=t_en), "gate": "read",
                         "prelim": "retrieve", "route": "search",
                         "hybrid": False, "src": "en_search"})
        for pat, gate, prelim, route in rng.sample(EN_ACTION_PATTERNS, 2):
            rows.append({"utterance": pat.format(t=t_en, s=rng.choice(SOURCES),
                                                 n=rng.randint(2, 8)),
                         "gate": gate, "prelim": prelim, "route": route,
                         "hybrid": False, "src": "en_action"})
    for t_zh in TOPICS_ZH:
        for pat, gate, prelim, route in rng.sample(ZH_BOUNDARY_PATTERNS, 3):
            rows.append({"utterance": pat.format(t=t_zh, s=rng.choice(SOURCES)),
                         "gate": gate, "prelim": prelim, "route": route,
                         "hybrid": False, "src": "zh_boundary"})

    for i, r in enumerate(rows):
        r["qid"] = f"aug-{r['src']}-{i:04d}"
        r["note"] = ""
                                 
    ood_utts = {r["utterance"] for r in load_jsonl("ood_route_queries.jsonl")}
    before = len(rows)
    rows = [r for r in rows if r["utterance"] not in ood_utts]
           
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        if r["utterance"] not in seen:
            seen.add(r["utterance"])
            uniq.append(r)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"n={len(uniq)} (ood-overlap dropped {before - len(rows)})")
    print("src:", dict(Counter(r["src"] for r in uniq)))
    print("gate×prelim:", dict(Counter((r["gate"], r["prelim"]) for r in uniq)))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
