import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e7_train2_0928 as previous

SEED = 20261005


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lineage_groups(rows, old_groups):
    parent = {}
    edges = []

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b, kind):
        ra, rb = find(a), find(b)
        if ra != rb:
            low, high = sorted((ra, rb))
            parent[high] = low
        edges.append({"child": a, "parent": b, "kind": kind})

    for row, old in zip(rows, old_groups, strict=True):
        union(row["family"], old, "previous_group")
    for row in rows:
        family, note = row["family"], row.get("note", "")
        for m in re.finditer(r"aug:mix:([A-Za-z0-9_+]+)", note):
            for ancestor in m.group(1).split("+"):
                union(family, ancestor, "explicit_mix")
        for m in re.finditer(r"aug:(?:degrade_of|noise_of):([A-Za-z0-9_]+)", note):
            union(family, m.group(1), "explicit_augmentation")
        for m in re.finditer(r"([a-z]+\d+)\s*同句", note):
            union(family, m.group(1), "explicit_same_sentence")
    groups = [find(r["family"]) for r in rows]
    for e in edges:
        e["child_component"] = find(e["child"])
        e["parent_component"] = find(e["parent"])
        if e["child_component"] != e["parent_component"]:
            raise AssertionError("Lineage edge crosses a component")
    return groups, edges


def verify_edges(rows, groups, edges):
    assignment = {r["family"]: g for r, g in zip(rows, groups, strict=True)}
    for e in edges:
        child, par = e["child"], e["parent"]
        if child in assignment and par in assignment and assignment[child] != assignment[par]:
            raise ValueError("Explicit ancestry leaks across folds")
    return True


def counts(y, selected):
    y, selected = np.asarray(y, dtype=bool), np.asarray(selected, dtype=bool)
    if y.shape != selected.shape:
        raise ValueError("Prediction shape mismatch")
    tp = int(np.sum(y & selected))
    fp = int(np.sum(~y & selected))
    fn = int(np.sum(y & ~selected))
    tn = int(np.sum(~y & ~selected))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall_B": tp / (tp + fn) if tp + fn else None,
            "precision_B": tp / (tp + fp) if tp + fp else None,
            "non_B_false_positive_rate": fp / (fp + tn) if fp + tn else None}


def matched_point(y, score, recall_min):
    feasible = []
    for threshold in np.unique(score):
        point = counts(y, score >= threshold)
        if point["recall_B"] >= recall_min - 1e-12:
            feasible.append({"threshold": float(threshold), **point})
    if not feasible:
        raise ValueError("No feasible descriptive operating point")
    best = max(feasible, key=lambda r: (r["precision_B"], r["threshold"]))
    return {"interpretation": "descriptive test-label-selected curve point, NOT deployment validation", **best}


def paired_bootstrap(y, a, b, groups, draws):
    rng = np.random.RandomState(SEED)
    values = []
    gs = sorted(set(groups))
    idxs = {g: np.flatnonzero(np.asarray(groups) == g) for g in gs}
    degenerate = 0
    for _ in range(draws):
        chosen = rng.choice(gs, len(gs), replace=True)
        idx = np.concatenate([idxs[g] for g in chosen])
        if np.unique(y[idx]).size != 2:
            degenerate += 1
            continue
        values.append(float(average_precision_score(y[idx], a[idx]) -
                            average_precision_score(y[idx], b[idx])))
    if not values:
        raise ValueError("No valid bootstrap replicates")
    values = np.asarray(values)
    return {"draws_requested": draws, "draws_valid": len(values), "degenerate": degenerate,
            "delta_AP": float(average_precision_score(y, a) - average_precision_score(y, b)),
            "percentile_95": np.percentile(values, [2.5, 97.5]).tolist(),
            "percentile_90": np.percentile(values, [5, 95]).tolist(),
            "median": float(np.median(values)),
            "resampling_fraction_le_0": float(np.mean(values <= 0)),
            "resampling_fraction_le_0_05": float(np.mean(values <= 0.05)),
            "scope": "fixed OOF predictions; cluster resampling only; fractions are NOT p-values"}


def normalized_units(units):
    out = {}
    for key, value in units.items():
        case, verb = key.split("|", 1)
        key = case + "|" + verb.removeprefix("curate.")
        if key in out:
            raise ValueError("Duplicate unit after name normalization")
        out[key] = value
    return out


def agreement(a, b, keys):
    aa, bb = [a[k]["label"] for k in keys], [b[k]["label"] for k in keys]
    n = len(keys)
    return {"n": n, "agree": sum(x == y for x, y in zip(aa, bb, strict=True)),
            "kappa": float(cohen_kappa_score(aa, bb)),
            "disagreement_ids": [k for k, x, y in zip(keys, aa, bb, strict=True) if x != y]}


def audit_annotations():
    a = normalized_units(load("e7r_gold_ai1_0928.json")["units"])
    b = normalized_units(load("e7r_gold_ai2_0928.json")["units"])
    consensus = normalized_units(load("e7r_consensus_0928.json")["units"])
    raw_units = load("e7r_units_0928.json")["units"]
    units = {u["case"] + "|" + u["verb"].removeprefix("curate."): u for u in raw_units}
    if len(units) != len(raw_units) or not (set(a) == set(b) == set(consensus) == set(units)):
        raise ValueError("Missing, duplicate or misaligned annotations")
    fixed = {k: dict(v) for k, v in a.items()}
    changes = []
    mechanical = set()
    for key, u in units.items():
        if u["in_forbid"] and not u["in_tools"]:
            mechanical.add(key)
            if fixed[key]["label"] != "X":
                changes.append({"id": key, "before": fixed[key]["label"], "after": "X",
                                "basis": "forbidden AND not registered; deterministic guideline rule"})
                fixed[key]["label"] = "X"
    keys = sorted(a)
    labelsets = defaultdict(list)
    for key, v in consensus.items():
        labelsets[key.split("|", 1)[0]].append(v["label"])
    task_labels = load("e7r_tasklabels_0928.json")
    for case, labels in labelsets.items():
        derived = "B" if "G" in labels else ("A" if set(labels) & {"F", "X"} else "N")
        if task_labels[case] != derived:
            raise ValueError("Consensus-derived task label mismatch")
    return {
        "raw": agreement(a, b, keys),
        "after_X_rule_correction": agreement(fixed, b, keys),
        "after_correction_without_mechanical_X": agreement(fixed, b, sorted(set(keys) - mechanical)),
        "mechanically_decidable_n": len(mechanical), "transformations": changes,
        "adjudications": [{"id": k, "AI1_corrected": fixed[k]["label"],
                           "AI2": b[k]["label"], "final": consensus[k]["label"]}
                          for k in keys if fixed[k]["label"] != b[k]["label"]],
        "borderline_consensus_units": sum(bool(x.get("borderline")) for x in consensus.values()),
        "task_label_counts": dict(Counter(task_labels.values())),
        "interpretation": "post-hoc rule correction is not raw independent annotation agreement",
    }


def selftest():
    n = 0
    rows = [{"family": "child1", "note": "aug:mix:missing+parent"},
            {"family": "child2", "note": "aug:noise_of:missing"},
            {"family": "parent", "note": ""}, {"family": "other", "note": ""}]
    groups, edges = lineage_groups(rows, [r["family"] for r in rows])
    assert groups[0] == groups[1] == groups[2] != groups[3]
    n += 1
    verify_edges(rows, groups, edges)
    n += 1
    try:
        verify_edges(rows, [r["family"] for r in rows], edges)
    except ValueError:
        n += 1
    else:
        raise AssertionError("Leaking ancestry was not detected")
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    cp = counts(y, score >= 0.7)
    assert (cp["TP"], cp["FP"], cp["FN"], cp["TN"]) == (2, 1, 0, 1)
    n += 1
    mp = matched_point(y, score, 1.0)
    assert mp["threshold"] == 0.7 and mp["precision_B"] == 2 / 3
    n += 1
    assert average_precision_score([1, 1, 0, 0], [1, .9, .2, .1]) == 1.0
    n += 1
    assert average_precision_score([1, 1, 0, 0], [.1, .2, .9, 1]) < .5
    n += 1
    bs = paired_bootstrap(y, score, score, ["a", "b", "a", "b"], 30)
    assert bs["percentile_95"] == [0.0, 0.0]
    n += 1
    try:
        normalized_units({"a|curate.search_online": {}, "a|search_online": {}})
    except ValueError:
        n += 1
    else:
        raise AssertionError("Duplicate normalized units accepted")
    return {"passed": n, "failed": 0}


def main(out, draws):
    if out.exists():
        raise FileExistsError("Refusing to replace old research output")
    st = selftest()
    rows = previous.load_rows_labels()
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training rows")
    tasklabels = load("e7r_tasklabels_0928.json")
    if set(tasklabels) != {r["id"] for r in rows}:
        raise ValueError("Missing task labels")
    for row in rows:
        if (row["label"] == "B") != (tasklabels[row["id"]] == "B"):
            raise ValueError("B target changed")
    y = np.array([int(r["label"] == "B") for r in rows])
    old = previous.merged_groups(rows)
    new, edges = lineage_groups(rows, old)
    verify_edges(rows, new, edges)
    source_names = ["e7_candidates_0928.json", "e7_gold_0928.json", "e7_train2_0928.py",
                    "e7r_units_0928.json", "e7r_gold_ai1_0928.json", "e7r_gold_ai2_0928.json",
                    "e7r_consensus_0928.json", "e7r_tasklabels_0928.json",
                    "e7_reaudit_设计-2026-10-05.md"]
    inputs = {name: sha(HERE / name) for name in source_names}
    p_r1 = np.array([previous.r1_score(r["utterance"]) for r in rows])
    p_r2 = np.array([float(r["has_check"] and bool(r["write_avail"])) for r in rows])
    action_rule = p_r1 * np.array([bool(re.search(r"入库|同步|下载|搜|检索", r["utterance"])) for r in rows])
    out.mkdir(parents=True)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save("manifest.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                           "seed": SEED, "input_sha256": inputs, "driver_sha256": sha(__file__),
                           "status": "running", "model_calls": 0,
                           "target": "task structural B label, not action emission or value of waiting"})
    save("selftest.json", st)
    save("annotation_audit.json", audit_annotations())
    save("groups.json", {"n_previous": len(set(old)), "n_lineage": len(set(new)), "edges": edges,
                         "rows": [{"id": r["id"], "family": r["family"], "old": a, "new": b}
                                  for r, a, b in zip(rows, old, new, strict=True)]})
    summary = {"n": len(rows), "B": int(y.sum()), "groups_old": len(set(old)), "groups_new": len(set(new)),
               "R1_AP": float(average_precision_score(y, p_r1)),
               "R2_AP": float(average_precision_score(y, p_r2)),
               "exploratory_action_rule_AP": float(average_precision_score(y, action_rule)),
               "R1_operating_point": counts(y, p_r1 > 0), "models": {}}
    for tag, groups, signature in [("old_full", old, True), ("lineage_full", new, True),
                                    ("lineage_text", new, False)]:
        print(f"training {tag}: {len(set(groups))} groups", flush=True)
        p, thresholds = previous.lofocv_predict(rows, y, groups, use_sig=signature, return_threshold=True)
        if not np.isfinite(p).all() or not np.isfinite(thresholds).all():
            raise ValueError("Missing OOF prediction or threshold")
        ap = float(average_precision_score(y, p))
        if tag == "old_full" and abs(ap - load("e7_results2_0928.json")["LTR_full"]["AP"]) > 1e-6:
            raise RuntimeError("Original protocol did not replicate; stop before new protocol")
        records = [{"id": r["id"], "group": g, "B": int(t), "score": float(s),
                    "train_only_threshold": float(th)}
                   for r, g, t, s, th in zip(rows, groups, y, p, thresholds, strict=True)]
        save(tag + "_oof.json", records)
        result = {"AP": ap, "train_only_threshold_point": counts(y, p >= thresholds),
                  "descriptive_matched_recall": matched_point(y, p, summary["R1_operating_point"]["recall_B"]),
                  "paired_delta_R1": paired_bootstrap(y, p, p_r1, groups, draws),
                  "paired_delta_R2": paired_bootstrap(y, p, p_r2, groups, draws),
                  "paired_delta_exploratory_action_rule": paired_bootstrap(y, p, action_rule, groups, draws)}
        summary["models"][tag] = result
        save("summary.json", summary)
        print(f"{tag} AP={ap:.6f}", flush=True)
    if any(sha(HERE / name) != value for name, value in inputs.items()):
        raise RuntimeError("Input drift during computation; batch invalid")
    save("completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                             "status": "completed", "input_hashes_unchanged": True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", type=Path, default=HERE / "e7_reaudit_1005")
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), ensure_ascii=False))
    else:
        main(a.out, a.bootstrap)
