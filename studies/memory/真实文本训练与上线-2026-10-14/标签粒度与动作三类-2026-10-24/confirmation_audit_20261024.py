from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT

from action_common_20261024 import ACTIONS, FINE, ROOT, dump, evaluate, initialize
from analyze_supplemental_results_20261024 import length_features, records
from build_action_data_20261024 import CONTEXTS, render, sample_constraints, selftest
from evaluate_action_runs_20261024 import paired_interval
from mean_pool_experiment_20261024 import mean_forward
from verify_action_experiment_20261024 import manual


TOPICS = ("需求变更单", "实验附注", "接口说明", "文稿校对", "分析备忘")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    formal = row["formal"]
    sides = [tuple(sorted((formal["scopes"][int(k) // 4], formal["feature_ids"][int(k) % 4], value)
                          for k, value in formal[side].items())) for side in ("old", "new")]
    return tuple(sorted(sides))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "confirmation-v2"
    output.mkdir(exist_ok=False)
    models = []
    for item in read(ROOT / "evaluation-lock.json")["runs"]:
        name = f"{item['arm']}-seed{item['seed']}"
        models.append({"name": name, "arm": item["arm"], "seed": item["seed"], "pool": "cls", "directory": f"runs/{name}"})
    models += [{"name": "mean-pool-seed13", "arm": "a3", "seed": 13, "pool": "mean", "directory": "pooling-mean-seed13"},
               {"name": "posthoc-r6-seed13", "arm": "r6", "seed": 13, "pool": "cls", "directory": "runs/r6-seed13"}]
    for item in models:
        directory = ROOT / item["directory"]
        item["checkpoint_sha256"] = read(directory / "result.json")["checkpoint_sha256"]
        assert FT.file_sha256(directory / "selected.pt") == item["checkpoint_sha256"]
    lock = {"fixed_generator_seed": 20261024 + 100003, "models": models, "no_retraining_or_reselection": True,
            "protocol_sha256": FT.file_sha256(ROOT / "确认集修补协议-2026-10-24.md"), "script_sha256": FT.file_sha256(__file__),
            "old_source_sha256": {split: FT.file_sha256(ROOT / "data-v1" / (split + ".jsonl")) for split in ("train", "dev", "heldout")}}
    dump(output / "lock.json", lock)
    prior = {}
    cross = []
    train = []
    for split in ("train", "dev", "heldout"):
        rows = [json.loads(line) for line in (ROOT / "data-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
        if split == "train":
            train = rows
        for row in rows:
            identity = key(row)
            if identity in prior and prior[identity]["split"] != split:
                cross.append([prior[identity]["id"], row["id"]])
            prior[identity] = row
    assert len(prior) == 1199 and len(cross) == 1
    seen = set(prior)
    confirmations = []
    for topic_index, title in enumerate(TOPICS):
        for sub in range(5):
            rng = random.Random(lock["fixed_generator_seed"] + topic_index * 37 + sub * 1009)
            feature_ids = rng.sample(range(16), 4)
            scopes = CONTEXTS[sub]
            targets = [0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5]
            rng.shuffle(targets)
            for index, target in enumerate(targets):
                while True:
                    old, new, reference = sample_constraints(rng, target)
                    row = {"id": f"confirmation-{topic_index}-{sub}-{index}", "cluster_id": f"confirmation-{topic_index}-{sub}", "domain": title,
                           "action_index": reference["action"], "fine_index": reference["fine"], "action": ACTIONS[reference["action"]],
                           "fine": FINE[reference["fine"]], "formal": {"old": old, "new": new, "feature_ids": feature_ids, "scopes": scopes, **reference}}
                    identity = key(row)
                    if identity not in seen:
                        seen.add(identity)
                        break
                row["pair"] = {"old": {"kind": "memory", "text": render(old, feature_ids, scopes, rng.choice((6, 7)), rng)},
                               "new": {"kind": "memory", "text": render(new, feature_ids, scopes, rng.choice((6, 7)), rng)},
                               "frame": {"mode": "same_domain", "background": f"同一份{title}有{scopes[0]}和{scopes[1]}两种情形。每种情形各项设置独立，且每项只能选择一个值。未提及的设置不作限定。"}}
                confirmations.append(row)
    assert len(confirmations) == 300 and len({key(row) for row in confirmations}) == 300
    assert not (set(prior) & {key(row) for row in confirmations})
    (output / "rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in confirmations), encoding="utf-8")
    torch.set_num_threads(4)
    device = FT.resolve_device("cuda:0")
    torch.cuda.set_device(device)
    data_inputs = None
    evaluated = {}
    for item in models:
        scorer = initialize(args.repo_root, item["arm"], item["seed"])
        if item["pool"] == "mean":
            scorer.model.forward = MethodType(mean_forward, scorer.model)
        directory = ROOT / item["directory"]
        assert FT.file_sha256(directory / "selected.pt") == item["checkpoint_sha256"]
        scorer.model.load_state_dict(torch.load(directory / "selected.pt", map_location="cpu", weights_only=True), strict=True)
        scorer.model.to(device)
        if data_inputs is None:
            data_inputs = []
            for row in confirmations:
                prepared = SR.prepare_semantic_input(row["pair"], scorer.tokenizer, max_length=384, role_independent_frame=True, limits=SR.SEMANTIC_LIMITS_RESEARCH)
                view = prepared["directions"]["forward"]
                assert view["complete"]
                data_inputs.append({"id": row["id"], "cluster_id": row["cluster_id"], "domain": row["domain"],
                                    "action_index": row["action_index"], "fine_index": row["fine_index"], "encoding": view["encoding"],
                                    "preserve_count": len(row["formal"]["preserve_keys"]), "uncovered_old_count": len(set(row["formal"]["old"]) - set(row["formal"]["new"])),
                                    "conflict_count": len(row["formal"]["conflict_keys"])})
        summary, predictions = evaluate(scorer.model, scorer.tokenizer, data_inputs, item["arm"], device)
        computed = manual(confirmations, predictions)
        assert all(np.isclose(summary[name], value, atol=1e-10, rtol=0) for name, value in computed.items() if name != "matrix")
        evaluated[item["name"]] = {"summary": summary, "predictions": predictions, "manual": computed}
        dump(output / (item["name"] + ".json"), evaluated[item["name"]])
        print(json.dumps({"name": item["name"], **computed}, ensure_ascii=False), flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    text = lambda row: row["pair"]["old"]["text"] + " [NEW] " + row["pair"]["new"]["text"]
    char = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=20000), LogisticRegression(C=1, max_iter=1000, random_state=13))
    char.fit([text(row) for row in train], [row["action_index"] for row in train])
    length = LogisticRegression(max_iter=1000, random_state=13).fit(length_features(train), [row["action_index"] for row in train])
    baselines = {"char_tfidf": manual(confirmations, records(confirmations, char.predict_proba([text(row) for row in confirmations]).tolist())),
                 "length_clause_count": manual(confirmations, records(confirmations, length.predict_proba(length_features(confirmations)).tolist()))}
    final = {"scope": "fresh fixed-sample same-generator confirmation; no canonical unordered semantic pair repeats old1200 or within new300",
             "n": 300, "old_cross_split_duplicates": cross, "old_unique": len(prior), "new_unique": 300,
             "oracle_selftest": selftest(), "raw_data_sha256": FT.file_sha256(output / "rows.jsonl"),
             "models": {name: {key: value for key, value in record.items() if key != "predictions"} for name, record in evaluated.items()},
             "baselines": baselines, "paired_interval_fixed_two_seeds": paired_interval(evaluated, data_inputs, [13, 20260919]), "no_retraining": True}
    dump(output / "summary.json", final)
    assert all(FT.file_sha256(ROOT / "data-v1" / (split + ".jsonl")) == value for split, value in lock["old_source_sha256"].items())
    print("CONFIRMATION COMPLETE; old results and all weights preserved", flush=True)


if __name__ == "__main__":
    main()
