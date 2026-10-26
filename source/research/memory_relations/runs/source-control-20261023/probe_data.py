"""只读数据核查：canonical train/dev、fine source_rows、fine final cohort、OCNLI 原生 dev。"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CANON = Path("artifacts/worktrees/semantic-formal-v12-20261016/research/memory_relations/runs/v34-v3-joint-preparation-20261019/prepared")
FINE = Path("artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v7-fine-development-20261014/teacher-completion-20261015-c71/delivery/source_rows.jsonl")
COHORT = Path("artifacts/v55-final-preflight-20261021/fine-final-cohort.json")
DATA_RUN = Path("artifacts/worktrees/semantic-assets-20261014/research/memory_relations/runs/v6-public-assets-20261014/v4-public-data-close-20261013")


def stream(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize_canonical(name):
    totals = Counter()
    labels = defaultdict(Counter)
    roots = defaultdict(set)
    clusters = defaultdict(set)
    datasets = Counter()
    fine_sample = None
    for row in stream(CANON / name):
        origin = "fine" if row.get("label_space") == "semantic5" else "public"
        dataset = row.get("lineage", {}).get("dataset") or row.get("source_binding", {}).get("dataset") or "?"
        split = row.get("split")
        totals[(origin, dataset, split)] += 1
        labels[(origin, dataset, split)][row.get("label")] += 1
        roots[(origin, dataset, split)].add(row.get("root_id"))
        clusters[(origin, dataset, split)].add(row.get("cluster_id"))
        datasets[(dataset, split)] += 1
        if origin == "fine" and fine_sample is None:
            fine_sample = row
    print(f"== canonical {name}: {sum(totals.values())} rows")
    for key in sorted(totals):
        print(" ", key, "rows", totals[key], "roots", len(roots[key]), "clusters", len(clusters[key]), "labels", dict(labels[key]))
    print("  fine sample keys:", sorted(fine_sample or {}))
    if fine_sample:
        print("  fine sample lineage:", json.dumps(fine_sample.get("lineage"), ensure_ascii=False)[:800])
        print("  fine sample source_binding:", json.dumps(fine_sample.get("source_binding"), ensure_ascii=False)[:800])
        print("  fine sample pair-frame:", json.dumps(fine_sample.get("pair", {}).get("frame"), ensure_ascii=False)[:600])


def summarize_fine():
    rows = list(stream(FINE))
    print(f"== fine source_rows: {len(rows)}")
    by_split = Counter(row["split"] for row in rows)
    by_label = defaultdict(Counter)
    roots = defaultdict(set)
    clusters = defaultdict(set)
    languages = defaultdict(Counter)
    for row in rows:
        split = row["split"]
        by_label[split][row["label"]] += 1
        roots[split].add(row["root_id"])
        clusters[split].add(row["cluster_id"])
        languages[split][row.get("phenomena", {}).get("language")] += 1
    print("  splits:", dict(by_split))
    for split in sorted(by_split):
        print(" ", split, "roots", len(roots[split]), "clusters", len(clusters[split]),
              "labels", dict(by_label[split]), "langs", dict(languages[split]))
    dup = Counter(row["root_id"] for row in rows)
    print("  duplicate roots:", {key: value for key, value in dup.items() if value > 1})
    return rows


def summarize_cohort():
    from dataset_recommender.agent import semantic_relations as SR
    from research.memory_relations import semantic_data as D
    doc = json.loads(COHORT.read_text(encoding="utf-8"))
    all_rows = []
    for entry in doc["sources"]:
        manifest = Path(entry["source_manifest"])
        scope = Path(entry["construction_scope"])
        bundle = D.load_qualified_source(manifest, construction_scope=scope, run_id=entry["run_id"])
        rows = bundle["rows"]
        print(f"== cohort {entry['source_id']}: rows {len(rows)}")
        print("  splits", dict(Counter(row["split"] for row in rows)),
              "labels", dict(Counter(row["label"] for row in rows)),
              "roots", len({row["root_id"] for row in rows}),
              "clusters", len({row["cluster_id"] for row in rows}))
        print("  languages", dict(Counter(row.get("phenomena", {}).get("language") for row in rows)))
        print("  projected", dict(Counter(SR.SEMANTIC_TO_NLI[row["label"]] for row in rows)))
        all_rows.extend(rows)
    print("  total", len(all_rows), "unique roots", len({row["root_id"] for row in all_rows}),
          "unique clusters", len({row["cluster_id"] for row in all_rows}))
    return all_rows


def summarize_official():
    from research.memory_relations.public_benchmarks import evaluate as E
    loaded = E.load_data_run(DATA_RUN)
    rows = [row for row in loaded["rows"] if row["dataset"] == "ocnli" and row["official_split"] == "dev"]
    print(f"== OCNLI dev: {len(rows)} rows")
    print("  labels", dict(Counter(row["gold_label"] for row in rows)))
    print("  missing", sum(1 for row in rows if row["gold_label"] is None))
    print("  groups", len({row["group_id"] for row in rows}))
    print("  manifest sha", loaded["manifest_sha256"])
    print("  files", [item for item in loaded["files"] if item["dataset"] == "ocnli"])
    return rows


if __name__ == "__main__":
    summarize_canonical("train.jsonl")
    summarize_canonical("dev.jsonl")
    fine = summarize_fine()
    cohort = summarize_cohort()
    official = summarize_official()
    fine_train = {row["root_id"] for row in fine if row["split"] == "train"}
    fine_all = {row["root_id"] for row in fine}
    cohort_roots = {row["root_id"] for row in cohort}
    print("== overlaps")
    print("  fine-train roots ∩ cohort roots:", len(fine_train & cohort_roots))
    print("  fine-all roots ∩ cohort roots:", len(fine_all & cohort_roots))
    print("  fine-train roots:", len(fine_train))
