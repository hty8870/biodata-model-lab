"""补充核查：中文train逐类、官方根代表逐类、精确overlap、角色。"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import semantic_data as D
from research.memory_relations.public_benchmarks import evaluate as E

CANON = Path("artifacts/worktrees/semantic-formal-v12-20261016/research/memory_relations/runs/v34-v3-joint-preparation-20261019/prepared")
FINE = Path("artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v7-fine-development-20261014/teacher-completion-20261015-c71/delivery/source_rows.jsonl")
COHORT = Path("artifacts/v55-final-preflight-20261021/fine-final-cohort.json")
DATA_RUN = Path("artifacts/worktrees/semantic-assets-20261014/research/memory_relations/runs/v6-public-assets-20261014/v4-public-data-close-20261013")


def stream(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pair_signature(pair):
    old, new = pair["old"], pair["new"]
    return json.dumps([old.get("text", ""), new.get("text", "")], ensure_ascii=False)


def canonical_ocnli(name):
    rows = [row for row in stream(CANON / name)
            if row.get("label_space") == "nli3" and row.get("lineage", {}).get("dataset") == "ocnli"]
    print(f"== canonical {name} ocnli rows {len(rows)}")
    print("  official_splits", dict(Counter(row["lineage"]["official_split"] for row in rows)))
    print("  protected_roles", dict(Counter(row["lineage"].get("protected_role") for row in rows)))
    print("  label_origin", dict(Counter(row.get("label_origin") for row in rows)))
    print("  gold_status", dict(Counter(row["lineage"].get("gold_status") for row in rows)))
    print("  label_method", dict(Counter(row["lineage"].get("label_method") for row in rows)))
    by_root = defaultdict(list)
    for row in rows:
        by_root[row["root_id"]].append(row)
    print("  roots", len(by_root), "max rows per root", max(len(v) for v in by_root.values()))
    rep = {root: min(items, key=lambda r: r["id"]) for root, items in by_root.items()}
    print("  representative labels", dict(Counter(row["label"] for row in rep.values())))
    return rows, rep


def fine_stats():
    rows = list(stream(FINE))
    train = [row for row in rows if row["split"] == "train"]
    zh = [row for row in train if row.get("phenomena", {}).get("language") == "zh"]
    for name, subset in (("train", train), ("zh-train", zh)):
        classes = Counter(SR.SEMANTIC_TO_NLI[row["label"]] for row in subset)
        leaves = Counter(row["label"] for row in subset)
        clusters = {row["cluster_id"] for row in subset}
        roles = Counter(row.get("phenomena", {}).get("author", {}).get("role") for row in subset)
        modes = Counter(row.get("pair", {}).get("frame", {}).get("mode") for row in subset)
        backgrounds = Counter(bool(row.get("pair", {}).get("frame", {}).get("background")) for row in subset)
        swapped = Counter(row.get("teacher_evidence", {}).get("swapped") for row in subset)
        print(f"== fine {name}: roots {len(subset)} clusters {len(clusters)}")
        print("  classes", dict(classes), "leaves", dict(leaves))
        print("  roles", dict(roles), "modes", dict(modes), "bg_nonempty", dict(backgrounds), "swapped", dict(swapped))
    return rows, train, zh


def cohort_rows():
    doc = json.loads(COHORT.read_text(encoding="utf-8"))
    rows = []
    for entry in doc["sources"]:
        bundle = D.load_qualified_source(Path(entry["source_manifest"]),
                                         construction_scope=Path(entry["construction_scope"]),
                                         run_id=entry["run_id"])
        rows.extend(bundle["rows"])
    return rows


def official_dev():
    loaded = E.load_data_run(DATA_RUN)
    return [row for row in loaded["rows"] if row["dataset"] == "ocnli" and row["official_split"] == "dev"]


if __name__ == "__main__":
    train_rows, train_rep = canonical_ocnli("train.jsonl")
    dev_rows, dev_rep = canonical_ocnli("dev.jsonl")
    fine_rows, fine_train, fine_zh = fine_stats()
    cohort = cohort_rows()
    official = official_dev()
    print("== exact overlaps")
    train_roots = {row["root_id"] for row in train_rows}
    dev_roots = {row["root_id"] for row in dev_rows}
    print("  canonical ocnli train/dev root overlap", len(train_roots & dev_roots))
    train_pairs = {pair_signature(row["pair"]) for row in train_rows}
    dev_pairs = {pair_signature(row["pair"]) for row in dev_rows}
    print("  canonical ocnli train/dev exact pair overlap", len(train_pairs & dev_pairs))
    official_pairs = {json.dumps([row["premise"], row["hypothesis"]], ensure_ascii=False) for row in official}
    print("  canonical ocnli train vs official dev exact pair overlap", len(train_pairs & official_pairs))
    fine_train_pairs = {pair_signature(row["pair"]) for row in fine_train}
    cohort_pairs = {pair_signature(row["pair"]) for row in cohort}
    print("  fine train vs final cohort exact pair overlap", len(fine_train_pairs & cohort_pairs))
    fine_zh_pairs = {pair_signature(row["pair"]) for row in fine_zh}
    official_pairs_canonical = {pair_signature(row["pair"]) for row in train_rows}
    print("  fine zh train vs canonical ocnli train exact pair overlap", len(fine_zh_pairs & official_pairs_canonical))
    print("== cohort splits/roles")
    print("  splits", dict(Counter(row["split"] for row in cohort)))
    print("  roles", dict(Counter(row.get("phenomena", {}).get("author", {}).get("role") for row in cohort)))
    print("  modes", dict(Counter(row.get("pair", {}).get("frame", {}).get("mode") for row in cohort)))
    print("  languages", dict(Counter(row.get("phenomena", {}).get("language") for row in cohort)))
    print("  labels", dict(Counter(row["label"] for row in cohort)))
