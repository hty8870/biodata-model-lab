                       
"""OOD 路由泛化评测集构造（2026-09-19，零 LLM，人工金标）。

动机：L0（TF-IDF char n-gram + LogReg）的 98-99% 一致率是在与训练集同分布的
合成语料上测的——用户质疑其真实泛化力与理解力。本集专门覆盖**远离合成模板
分布**的输入模式，每项人工定金标（gate=write/read、prelim=retrieve/skip、
route=search/action/general、hybrid=是否混合诉求），判据与
prompts/route_write_gate.md / label_prelim_gate.py 的口径逐字一致。

类别（qid 前缀）：
  piy  拼音（无中文）        eng  全英文              col  口语碎句
  typo 错字/同音字           mix  中英混杂            long 长句科研风
  bnd  write/read 边界      neg  否定取消            cap  库容/元问题变体
  bare 裸指代改条件          gen  概念问答/闲聊变体   adv  杂项对抗
输出 research/route_eval/ood_route_queries.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "ood_route_queries.jsonl"

                                                     
ROWS: list[tuple[str, str, str, str, str, bool, str]] = [
                      
    ("piy", "fei ai de dan xi bao shu ju you ma", "read", "retrieve", "search", False, "拼音检索"),
    ("piy", "bang wo zhao ren lei fei xian ai dan xi bao shu ju ji", "read", "retrieve", "search", False, ""),
    ("piy", "xiao shu nao de scRNA shu ju you meiyou", "read", "retrieve", "search", False, ""),
    ("piy", "you meiyou gan ai de dan xi bao zhuan lu zu shu ju", "read", "retrieve", "search", False, ""),
    ("piy", "zhao yi xia ru xian ai de shu ju", "read", "retrieve", "search", False, ""),
    ("piy", "ban ma yu de dan xi bao shu ju you ma", "read", "retrieve", "search", False, ""),
    ("piy", "ren yuan wei chang dao de dan xi bao shu ju you meiyou", "read", "retrieve", "search", False, ""),
    ("piy", "bang wo ba qian wu tiao da bao xia zai", "write", "skip", "action", False, "拼音动作-下载"),
    ("piy", "dao chu zhe ji tiao de yin wen", "write", "skip", "action", False, "拼音动作-引文"),
    ("piy", "jian cha 10x you meiyou geng xin", "read", "skip", "action", False, "拼音动作-只读检查"),
    ("piy", "tong bu yi xia 10x de shu ju", "write", "skip", "action", False, "拼音动作-同步入库"),
    ("piy", "xian zai shu ju ku li you duo shao tiao", "read", "skip", "action", False, "拼音库容"),
                   
    ("eng", "any single-cell RNA-seq datasets on human lung cancer?", "read", "retrieve", "search", False, ""),
    ("eng", "find me mouse brain scRNA data please", "read", "retrieve", "search", False, ""),
    ("eng", "do you have spatial transcriptomics data for breast cancer", "read", "retrieve", "search", False, ""),
    ("eng", "looking for single nucleus RNA sequencing of human kidney", "read", "retrieve", "search", False, ""),
    ("eng", "is there any zebrafish single cell atlas in your database", "read", "retrieve", "search", False, ""),
    ("eng", "search for colon cancer scRNA-seq with treatment response", "read", "retrieve", "search", False, ""),
    ("eng", "download the top 5 results as a package", "write", "skip", "action", False, ""),
    ("eng", "export citations for these datasets", "write", "skip", "action", False, ""),
    ("eng", "sync the latest 10x datasets into my library", "write", "skip", "action", False, ""),
    ("eng", "search online for human lung data and import it", "write", "skip", "action", False, "联网搜+入库=write"),
    ("eng", "check whether 10x has new releases", "read", "skip", "action", False, "只查不同步=read"),
    ("eng", "how many datasets are in the library now", "read", "skip", "action", False, ""),
    ("eng", "what is single cell sequencing anyway", "read", "skip", "general", False, "概念问答"),
    ("eng", "GSE123456 do you have this one", "read", "retrieve", "search", False, "编号直链"),
                    
    ("col", "那个……肺的，单细胞的，有吗", "read", "retrieve", "search", False, ""),
    ("col", "肺癌的有没有啊", "read", "retrieve", "search", False, ""),
    ("col", "就想看看小鼠脑子的数据有没有", "read", "retrieve", "search", False, ""),
    ("col", "诶你这儿有乳腺癌的单细胞不", "read", "retrieve", "search", False, ""),
    ("col", "肝癌的，人呢，有数据吗", "read", "retrieve", "search", False, ""),
    ("col", "帮我瞅瞅有没有斑马鱼的", "read", "retrieve", "search", False, ""),
    ("col", "空间转录组的有没有哇", "read", "retrieve", "search", False, ""),
    ("col", "这个GSE号能查到不", "read", "retrieve", "search", False, ""),
    ("col", "前五条给我打包一下呗", "write", "skip", "action", False, ""),
    ("col", "把引文导出来给我哈", "write", "skip", "action", False, ""),
    ("col", "帮我看看库里现在多少条了呀", "read", "skip", "action", False, ""),
    ("col", "在吗在吗", "read", "skip", "general", False, "闲聊"),
    ("col", "哦哦那算了不用了", "read", "skip", "general", False, "闲聊收尾"),
                      
    ("typo", "肺艾单细胞数据有吗", "read", "retrieve", "search", False, "癌→艾"),
    ("typo", "帮我找干癌细胞的数据", "read", "retrieve", "search", False, "肝→干 歧义（干细胞/肝癌）——检索向判 search"),
    ("typo", "闲细泡测序是啥", "read", "skip", "general", False, "细胞→细泡+概念问答"),
    ("typo", "乳线癌的数据有么", "read", "retrieve", "search", False, "腺→线"),
    ("typo", "小鼠恼的数据", "read", "retrieve", "search", False, "脑→恼"),
    ("typo", "申报遗传的有没有", "read", "retrieve", "search", False, "肾→申 错字串"),
    ("typo", "单细跑转录组", "read", "retrieve", "search", False, "胞→跑"),
    ("typo", "帮我jian查一下更新", "read", "skip", "action", False, "检→jian 中英混"),
                    
    ("mix", "帮我找human lung的scRNA数据集", "read", "retrieve", "search", False, ""),
    ("mix", "有没有mouse brain的single cell", "read", "retrieve", "search", False, ""),
    ("mix", "breast cancer的单细胞数据有吗", "read", "retrieve", "search", False, ""),
    ("mix", "check一下10x有没有更新", "read", "skip", "action", False, ""),
    ("mix", "把top3打包download给我", "write", "skip", "action", False, ""),
    ("mix", "scRNA-seq的肺癌，人的，有没", "read", "retrieve", "search", False, ""),
    ("mix", "帮我sync一下最新数据", "write", "skip", "action", False, ""),
    ("mix", "spatial转录组有没有人的肺", "read", "retrieve", "search", False, ""),
                     
    ("long", "老师让找的，人源、肺腺癌、最好带治疗前后对照的单细胞转录组数据，有合适的吗", "read", "retrieve", "search", False, ""),
    ("long", "我们课题要做肿瘤微环境，想找人的结直肠癌单细胞数据，最好是公开能直接下载的那种", "read", "retrieve", "search", False, ""),
    ("long", "麻烦帮我看看库里有没有小鼠心脏发育过程的单细胞测序数据，时间序列的最好", "read", "retrieve", "search", False, ""),
    ("long", "想对比一下人和小鼠肺部的单细胞图谱差异，先帮我找两边各自有什么数据", "read", "retrieve", "search", False, ""),
    ("long", "投稿要补材料，把刚才这几条数据集的引文按标准格式导出来，顺便看下库容", "write", "skip", "general", True, "混合：引文+库容"),
    ("long", "先把10x那边的更新同步进库里，然后帮我找找人类肺的单细胞数据集", "write", "skip", "general", True, "混合：同步+检索"),
    ("long", "师兄说10x最近有更新，你帮我查一下，有新增的人肺数据就直接搜来入库", "write", "skip", "action", False, "条件动作链=action"),
    ("long", "检查下ENCODE有没有更新，顺便帮我找乳腺癌的单细胞数据集", "read", "retrieve", "general", True, "混合：检查(只读)+检索"),
    ("long", "我想知道目前这个库覆盖了多少种癌症类型的单细胞数据，能大概说一下吗", "read", "skip", "action", False, "库容变体"),
    ("long", "下周组会要汇报，帮我看看库里现在有多少数据集、最近有没有新增", "read", "skip", "action", False, "库容+近期变动=db_status"),
                             
    ("bnd", "联网搜人类肺数据入库", "write", "skip", "action", False, "上批 L0 判错样本"),
    ("bnd", "联网搜一下人类肺的单细胞数据，只看看先不入库", "read", "skip", "action", False, "联网搜但不入库=read（spec 明文）"),
    ("bnd", "检查10x更新，有新增就同步进来", "write", "skip", "action", False, "检查+同步=sync=write"),
    ("bnd", "检查10x更新就行，别入库", "read", "skip", "action", False, "只检查=read"),
    ("bnd", "把这几条恢复一下，刚才误删了", "write", "skip", "action", False, "恢复=write"),
    ("bnd", "回收站里那条删掉吧不要了", "write", "skip", "action", False, "删除=write"),
    ("bnd", "我上传的那些数据帮我清点一下", "read", "skip", "action", False, "清点=read"),
    ("bnd", "前两条对比下，第三条导出引文", "write", "skip", "general", True, "混合：对比+引文"),
    ("bnd", "把第一个和第三个的引文都给我", "write", "skip", "action", False, ""),
    ("bnd", "这些数据FAIR合规吗帮我查下", "read", "skip", "action", False, "FAIR 检查=read"),
                    
    ("neg", "别下了，先不用管10x那个事了", "read", "skip", "action", False, "取消→极性门 cancelled；无写动作"),
    ("neg", "不用打包了谢谢", "read", "skip", "general", False, "取消下载=无动作"),
    ("neg", "刚才说同步的事先取消", "read", "skip", "action", False, "取消同步"),
    ("neg", "算了不用入库了，就看看", "read", "retrieve", "search", False, "否定入库+检索意图→search"),
    ("neg", "别给我推荐小鼠的，只要人的", "read", "retrieve", "search", False, "排除条件检索"),
    ("neg", "不是让你下载，是让你查有没有", "read", "retrieve", "search", False, "澄清=检索"),
                        
    ("cap", "现在数据量多少了", "read", "skip", "action", False, ""),
    ("cap", "库里现在有啥呀", "read", "skip", "action", False, "库容口语"),
    ("cap", "你这数据库大不大啊", "read", "skip", "action", False, ""),
    ("cap", "最近库里有新增数据集吗", "read", "skip", "action", False, "近期变动=db_status"),
    ("cap", "你这个库都覆盖哪些来源", "read", "skip", "action", False, "来源清点=db_status 口径"),
    ("cap", "库是不是该更新了", "read", "skip", "action", False, "含糊管护→action"),
                      
    ("bare", "那老鼠的呢", "read", "retrieve", "search", False, "指代改条件（小鼠→老鼠）"),
    ("bare", "换成人的再查一遍", "read", "retrieve", "search", False, ""),
    ("bare", "不要斑马鱼了，看果蝇的", "read", "retrieve", "search", False, ""),
    ("bare", "上一批结果里第二个是啥", "read", "skip", "general", False, "指代旧结果=元问题"),
    ("bare", "刚才那些都不要，重新来", "read", "retrieve", "search", False, "重置检索"),
    ("bare", "就按刚才那个条件再查一次", "read", "retrieve", "search", False, ""),
                         
    ("gen", "单细胞测序和普通转录组有啥区别", "read", "skip", "general", False, ""),
    ("gen", "你能干啥呀", "read", "skip", "general", False, "能力问答"),
    ("gen", "谢谢啦", "read", "skip", "general", False, ""),
    ("gen", "这个系统是谁做的", "read", "skip", "general", False, ""),
    ("gen", "什么是空间转录组，跟单细胞啥关系", "read", "skip", "general", False, ""),
    ("gen", "FASTQ文件是啥格式", "read", "skip", "general", False, ""),
    ("gen", "今天天气真好", "read", "skip", "general", False, "完全无关闲聊"),
                    
    ("adv", "肺肺肺肺肺", "read", "retrieve", "search", False, "重复碎词——检索意图可辨"),
    ("adv", "？？？", "read", "skip", "general", False, "无语义——拿不准判 read/general"),
    ("adv", "GSE", "read", "retrieve", "search", False, "残缺编号——检索意图"),
    ("adv", "帮我写个爬虫爬GEO", "read", "skip", "general", False, "越界请求=婉拒向，非检索非写盘"),
    ("adv", "把所有数据都下载下来我本地用", "write", "skip", "action", False, "全量下载=write"),
    ("adv", "你能直接访问GEO吗", "read", "skip", "general", False, "能力问答"),
    ("adv", "肿瘤", "read", "retrieve", "search", False, "单词检索"),
    ("adv", "single cell", "read", "retrieve", "search", False, "英文单词检索"),
]


def main() -> int:
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for i, (cat, utt, gate, prelim, route, hybrid, note) in enumerate(ROWS):
            f.write(json.dumps({
                "qid": f"ood-{cat}-{i:03d}", "utterance": utt, "cat": cat,
                "gate": gate, "prelim": prelim, "route": route,
                "hybrid": hybrid, "note": note,
            }, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = Counter((r[2], r[3]) for r in ROWS)
    cats = Counter(r[0] for r in ROWS)
    print(f"n={len(ROWS)} cats={dict(cats)}")
    print(f"gate×prelim={dict(dist)}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
