from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
ACTIONS = ("coexist", "replace_all", "merge_preserve")
FINE = ("coexist_equivalent", "coexist_old_strict", "coexist_new_strict", "coexist_overlap", "replace_conflict", "merge_conflict")
PARENTS = (0, 0, 0, 0, 1, 2)
FEATURES = [
    ("语种", "只用中文", "只用英文"), ("引文", "附上出处", "不附出处"),
    ("排版", "采用单栏", "采用双栏"), ("时间", "使用24小时制", "使用12小时制"),
    ("数值", "小数保留两位", "小数保留三位"), ("列表", "列表带编号", "列表不编号"),
    ("代码", "使用Python代码", "使用R代码"), ("图像", "图像保存为PNG", "图像保存为SVG"),
    ("标题", "保留章节标题", "省去章节标题"), ("字体", "使用简体字", "使用繁体字"),
    ("表格", "表格采用CSV", "表格采用Markdown"), ("主题", "采用浅色主题", "采用深色主题"),
    ("页面", "使用纵向页面", "使用横向页面"), ("纸张", "纸张设为A4", "纸张设为Letter"),
    ("文件", "文件合成一个压缩包", "文件分别保存"), ("顺序", "记录按时间正序排列", "记录按时间倒序排列"),
]
DOMAINS = ["会议纪要", "文献摘要", "实验记录", "项目简报", "课程讲义", "测试报告", "访谈整理", "统计周报", "研究提纲", "数据说明", "图表注释", "阅读笔记", "问卷汇总", "需求分析", "评审意见", "归档目录", "培训资料", "进度说明", "结果清单", "展示脚本"]
CONTEXTS = [("草稿", "终稿"), ("初审版", "复审版"), ("内部版", "发布版"), ("屏幕版", "打印版"), ("工作日版", "周末版")]
WORLD = np.asarray(list(itertools.product((0, 1), repeat=8)), dtype=np.int8)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def masks(constraints):
    return np.all(np.asarray([WORLD[:, int(key)] == value for key, value in constraints.items()]), axis=0)


def oracle(old, new):
    left, right = masks(old), masks(new)
    assert left.any() and right.any()
    conflict = sorted(int(key) for key in set(old) & set(new) if old[key] != new[key])
    residual = sorted(int(key) for key in set(old) - set(new))
    if not np.any(left & right):
        assert conflict
        fine = 5 if residual else 4
    elif np.array_equal(left, right):
        fine = 0
    elif np.all(~left | right):
        fine = 1
    elif np.all(~right | left):
        fine = 2
    else:
        fine = 3
    assert (fine >= 4) == bool(conflict)
    return {"fine": fine, "action": PARENTS[fine], "conflict_keys": conflict,
            "preserve_keys": residual if conflict else [], "old_worlds": int(left.sum()), "new_worlds": int(right.sum())}


def sample_constraints(rng, target):
    while True:
        keys = rng.sample(range(8), rng.randint(2, 5))
        old = {str(k): rng.randrange(2) for k in keys}
        if target == 0:
            new = dict(old)
        elif target == 1:
            new = {k: old[k] for k in rng.sample(list(old), rng.randint(1, len(old) - 1))}
        elif target == 2:
            new = dict(old)
            for k in rng.sample([str(k) for k in range(8) if str(k) not in old], rng.randint(1, min(2, 8 - len(old)))):
                new[k] = rng.randrange(2)
        elif target == 3:
            kept = rng.sample(list(old), rng.randint(1, len(old) - 1))
            new = {k: old[k] for k in kept}
            for k in rng.sample([str(k) for k in range(8) if str(k) not in old], rng.randint(1, min(3, 8 - len(old)))):
                new[k] = rng.randrange(2)
        elif target == 4:
            new = dict(old)
            for k in rng.sample(list(old), rng.randint(1, len(old))):
                new[k] = 1 - old[k]
            for k in rng.sample([str(k) for k in range(8) if str(k) not in old], rng.randint(0, min(2, 8 - len(old)))):
                new[k] = rng.randrange(2)
        else:
            kept = rng.sample(list(old), rng.randint(1, len(old) - 1))
            new = {k: old[k] for k in kept}
            for k in rng.sample(kept, rng.randint(1, len(kept))):
                new[k] = 1 - old[k]
            for k in rng.sample([str(k) for k in range(8) if str(k) not in old], rng.randint(0, min(3, 8 - len(old)))):
                new[k] = rng.randrange(2)
        result = oracle(old, new)
        if result["fine"] == target:
            return old, new, result


def render(constraints, feature_ids, scopes, style, rng):
    entries = []
    used = set()
    order = list(range(4))
    rng.shuffle(order)
    for slot in order:
        keys = [str(slot), str(slot + 4)]
        if all(key in constraints for key in keys) and constraints[keys[0]] == constraints[keys[1]] and rng.random() < .6:
            predicate = FEATURES[feature_ids[slot]][constraints[keys[0]] + 1]
            entries.append(f"{scopes[0]}与{scopes[1]}均{predicate}")
            used.update(keys)
    pairs = [(key, value) for key, value in constraints.items() if key not in used]
    rng.shuffle(pairs)
    for key, value in pairs:
        scope, slot = divmod(int(key), 4)
        predicate = FEATURES[feature_ids[slot]][value + 1]
        if style == 0:
            entries.append(f"{scopes[scope]}：{predicate}")
        elif style == 1:
            entries.append(f"在{scopes[scope]}中，{predicate}")
        elif style == 2:
            entries.append(f"{predicate}（仅指{scopes[scope]}）")
        elif style == 3:
            entries.append(f"{scopes[scope]}需要{predicate}")
        elif style == 4:
            entries.append(f"对于{scopes[scope]}，要求{predicate}")
        elif style == 5:
            entries.append(f"{scopes[scope]}的约定是{predicate}")
        elif style == 6:
            entries.append(f"若为{scopes[scope]}则{predicate}")
        else:
            entries.append(f"{predicate}，范围限于{scopes[scope]}")
    rng.shuffle(entries)
    return "；".join(entries) + "。"


def selftest():
    cases = [({"0": 0}, {"0": 0}, 0), ({"0": 0, "1": 1}, {"0": 0}, 1),
             ({"0": 0}, {"0": 0, "1": 1}, 2), ({"0": 0}, {"1": 1}, 3),
             ({"0": 0}, {"0": 1}, 4), ({"0": 0, "1": 1}, {"0": 1}, 5),
             ({"0": 0, "1": 1}, {"0": 1, "1": 1}, 4),
             ({"0": 0, "4": 0}, {"0": 1}, 5),
             ({"0": 0, "4": 0}, {"0": 1, "4": 0}, 4)]
    for old, new, expected in cases:
        assert oracle(old, new)["fine"] == expected
    correct = [oracle(old, new)["action"] for old, new, _ in cases]
    assert accuracy_score(correct, correct) == 1
    assert accuracy_score(correct, [(x + 1) % 3 for x in correct]) == 0
    return {"known_semantics_cases": len(cases), "preserve_counterexample": True, "covered_residual_counterexample": True,
            "perfect_metric": True, "deliberately_wrong_metric": True}


def main():
    output = ROOT / "data-v1"
    output.mkdir(exist_ok=False)
    checks = selftest()
    groups = {"train": [], "dev": [], "heldout": []}
    all_keys = set()
    for domain, title in enumerate(DOMAINS):
        split = "train" if domain < 12 else "dev" if domain < 15 else "heldout"
        for sub in range(5):
            cluster = f"scene-{domain:02d}-{sub}"
            rng = random.Random(20261024 + domain * 37 + sub * 1009)
            feature_ids = rng.sample(range(len(FEATURES)), 4)
            scopes = CONTEXTS[sub]
            styles = (0, 1, 2, 3) if split == "train" else (4, 5) if split == "dev" else (6, 7)
            targets = [0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5]
            rng.shuffle(targets)
            for number, target in enumerate(targets):
                while True:
                    old, new, result = sample_constraints(rng, target)
                    pair = {"old": {"kind": "memory", "text": render(old, feature_ids, scopes, rng.choice(styles), rng)},
                            "new": {"kind": "memory", "text": render(new, feature_ids, scopes, rng.choice(styles), rng)},
                            "frame": {"mode": "same_domain", "background": f"同一份{title}有{scopes[0]}和{scopes[1]}两种情形。每种情形各项设置独立，且每项只能选择一个值。未提及的设置不作限定。"}}
                    key = json.dumps(pair, ensure_ascii=False, sort_keys=True)
                    if key not in all_keys:
                        all_keys.add(key)
                        break
                record = {"id": f"{cluster}-pair-{number:02d}", "cluster_id": cluster, "split": split,
                          "pair": pair, "action": ACTIONS[result["action"]], "fine": FINE[result["fine"]],
                          "action_index": result["action"], "fine_index": result["fine"],
                          "formal": {"old": old, "new": new, "feature_ids": feature_ids, "scopes": scopes, **result},
                          "label_origin": "finite-independent-slot-oracle-not-human-gold", "domain": title}
                groups[split].append(record)
    for split, rows in groups.items():
        assert len(rows) == {"train": 720, "dev": 180, "heldout": 300}[split]
        assert len({row["cluster_id"] for row in rows}) == {"train": 60, "dev": 15, "heldout": 25}[split]
        assert Counter(row["action"] for row in rows) == {name: len(rows) // 3 for name in ACTIONS}
        (output / (split + ".jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    train, dev = groups["train"], groups["dev"]
    text = lambda row: row["pair"]["old"]["text"] + " [NEW] " + row["pair"]["new"]["text"]
    classifier = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=20000),
                               LogisticRegression(C=1, max_iter=1000, random_state=13))
    classifier.fit([text(row) for row in train], [row["action_index"] for row in train])
    guesses = classifier.predict([text(row) for row in dev])
    length_features = lambda rows: np.asarray([[len(row["pair"]["old"]["text"]), len(row["pair"]["new"]["text"]),
                                             row["pair"]["old"]["text"].count("；") + 1, row["pair"]["new"]["text"].count("；") + 1] for row in rows])
    length_model = LogisticRegression(max_iter=1000, random_state=13).fit(length_features(train), [row["action_index"] for row in train])
    length_guess = length_model.predict(length_features(dev))
    result = {"schema": "action-granularity-data-pilot-v1", "selftest": checks,
              "split_unit": "domain groups; all five context variants and each group twelve semantic pairs stay together",
              "different_expression_styles_by_split": True, "not_human_gold": True, "heldout_scores_not_computed": True,
              "split_counts": {split: {"n": len(rows), "clusters": len({row["cluster_id"] for row in rows}),
                                        "actions": dict(Counter(row["action"] for row in rows)), "fine": dict(Counter(row["fine"] for row in rows))} for split, rows in groups.items()},
              "dev_baselines": {"constant_accuracy": 1 / 3,
                                "char_tfidf": {"accuracy": float(accuracy_score([row["action_index"] for row in dev], guesses)), "macro_f1": float(f1_score([row["action_index"] for row in dev], guesses, average="macro"))},
                                "length_and_clause_count": {"accuracy": float(accuracy_score([row["action_index"] for row in dev], length_guess)), "macro_f1": float(f1_score([row["action_index"] for row in dev], length_guess, average="macro"))}},
              "sha256": {split: hashlib.sha256((output / (split + ".jsonl")).read_bytes()).hexdigest() for split in groups}}
    pilot = []
    for target in range(6):
        pilot.extend([row for row in train if row["fine_index"] == target][:6])
    (output / "pilot-36.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot), encoding="utf-8")
    dump(output / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
