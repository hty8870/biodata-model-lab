                       
"""E7 金标准落地（2026-09-28，e7_设计-2026-09-28.md §2）。
仲裁者：主线 agent 逐条读 utterance+tools+expect+note 全上下文，对正则预测盲视。
口径：
  B = 观测门控写——utterance 含「check 类观测内容决定写类动词做不做」的子句
      （如「若有新增就搜/同步/下载/入库」）；写动机本身是条件性的。
  A = 绝对禁令/无写动机但 expect 带写类 forbid——写无论观测如何都不该做，
      或用户只让 check、未请求任何写。
  N = 其余（无写请求且无 forbid，或写请求无条件）。
borderline = 存在混合结构（条件读门控/条件+显式被禁请求/门控在搜索结果而非 check）。
适配披露（设计 §2 之后）：阴性抽审发现 forbid∩WRITE 候选规则漏掉 hot 世界与
expect 空世界的 B（15 抽 6 中），故仲裁从「候选+抽样」扩为全量 167。此适配发生在
任何训练/评估之前，预注册判定（E7-1~E7-5）不受影响。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

B = """b01a b01b b03a b03b b10a b10b j10a k01_k06x k01a l08a l08b mix03
b14a i09b l01_l02x k11b k02 b04a
a07a a07b b02a b02b b04b b06b b09a b09b b12 b12a b12b b14b g04 i04a i04b i09a
j03 j03a j03b j07a j07b k02a k02b k05 k05a k05b k10_k07x k10a k11 k11a
l01_l03x l02_l01x l02d l03_l01x l03d l07 l07a l07b""".split()

A = """a01a a01b a03a a03b a04a a04b b05a b05b b11a b11b c02a c04a c05a c05b
i01a i01b i02a i02b i03a i03b i07a i07b j01a j08a j08b j09a j09b
j11a j11b j12a j13a k03a k06a k09 k09a k09b
b07a b07b""".split()

BORDERLINE = {
    "j02a": "Codex v0.8 B2 更正：请求的写(search)被允许且要求，被禁 sync 未注册工具表——非观测门控亦非「请求被禁的写」，改 N",
    "j02b": "同 j02a",
    "k04a": "Codex v0.8 B2 更正：expect 要求首步 sync（无条件写需求），被禁的是未请求的额外 search——非「无写动机」，改 N",
    "k04b": "同 k04a",
    "b07a": "写绝对取消+条件读门控（混合）",
    "b07b": "同 b07a",
    "k06a": "条件门控落在读（没的话→再查两家），无写动机",
    "mix03": "条件写+显式写请求同被 forbid（冷世界陷阱混合）",
    "l09_l04x": "门控在搜索结果（找到就入库）而非 check 观测",
}

RATIONALE = {
    "B": "含观测门控写子句（若/有…就…→搜/同步/下载/入库）",
    "A": "无写动机或写被绝对禁止（取消/只查不问后续）",
    "N": "无 forbid 且写请求无条件（或直接搜索/顺序链/只读）",
}


def main():
    rows = json.load(open(Path(__file__).with_name("e7_candidates_0928.json"),
                          encoding="utf-8"))["rows"]
    ids = {r["id"] for r in rows}
    gold = {}
    for r in rows:
        cid = r["id"]
        if cid in B:
            lab = "B"
        elif cid in A:
            lab = "A"
        else:
            lab = "N"
        gold[cid] = {"label": lab, "rationale": RATIONALE[lab]}
        if cid in BORDERLINE:
            gold[cid]["borderline"] = True
            gold[cid]["rationale"] += "；" + BORDERLINE[cid]
    assert set(B) <= ids and set(A) <= ids, "标签表里有未知 id"
    assert not (set(B) & set(A)), "B/A 表重叠"
    covered = set(B) | set(A)
    nN = len(ids - covered)
            
    cnt = Counter(g["label"] for g in gold.values())
                                          
    fam = defaultdict(set)
    idfam = {r["id"]: r["family"] for r in rows}
    for cid, g in gold.items():
        fam[idfam[cid]].add(g["label"])
    mixed = {f: sorted(v) for f, v in fam.items() if len(v) > 1}
    out = {
        "_readme": __doc__,
        "counts": dict(cnt),
        "mixed_label_families": mixed,
        "gold": gold,
    }
    Path(__file__).with_name("e7_gold_0928.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("counts:", dict(cnt), "(N 自动=剩余", nN, ")")
    print("families with mixed labels:", mixed if mixed else "无")


if __name__ == "__main__":
    main()
