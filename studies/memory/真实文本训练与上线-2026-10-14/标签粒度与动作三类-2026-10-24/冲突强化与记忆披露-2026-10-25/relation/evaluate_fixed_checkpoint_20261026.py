from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
import conflict_common as K
import score_conflict as S

ROOT = Path(__file__).resolve().parent
ROUTES = ("new-heldout", "new-heldout-natural", "new-heldout-counterfactual", "old-heldout", "composite")
ALIASES = tuple(f"{route}-seed{seed}" for seed in K.SEEDS for route in K.SCORE_ROUTES)


def publish(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", required=True, choices=ALIASES)
    args = parser.parse_args()
    alias = args.alias
    route, seed_text = alias.split("-seed")
    seed = int(seed_text)
    entry = (S.resolve_run(None, str(K.PARENT_EXP / "runs" / f"direct3-seed{seed}"))
             if route == "r0" else S.resolve_run(alias, None))
    lock_path = K.RUNS_DIR / f"final-lock-{alias}-terminal.json"
    lock = K.load_lock(lock_path)
    assert lock["run"] == entry["run"]
    assert lock["run_lock_sha256"] == K.sha256_file(entry["directory"] / "run-lock.json")
    actual_files = K.checkpoint_file_shas(entry["directory"] / "checkpoints/terminal")
    rows_by_route = {}
    for name in ROUTES:
        assert lock["checkpoint_files_sha256"][f"{name}:terminal"] == actual_files
        K.check_lock(lock, route=name, run_dir=entry["directory"], checkpoint="terminal")
        rows_by_route[name] = S.load_rows_for_scoring(name, lock=lock)
    output_dir = ROOT / "evaluations" / alias
    if output_dir.exists():
        raise FileExistsError(f"Preserve existing evaluation: {output_dir}")
    device = S._cuda("cuda:0")
    start = time.perf_counter()
    model, tokenizer, reload_info = S.load_model(entry, checkpoint="terminal", device=device)
    scores = {}
    for name, rows in rows_by_route.items():
        result = S.score_rows(rows=rows, model=model, tokenizer=tokenizer, device=device, split_label=name)
        result.update(alias=alias, run=entry["run"], checkpoint="terminal-300", reload=reload_info,
                      final_lock_sha256=K.sha256_file(lock_path),
                      route_fingerprint=K.route_fingerprint_any(name)[0])
        publish(output_dir / f"{name}.json", result)
        scores[name] = result
        print(json.dumps({"alias": alias, "route": name, "n": result["n"], "coarse": result["coarse"]}, ensure_ascii=False), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer, second_reload = S.load_model(entry, checkpoint="terminal", device=device)
    fresh = S.score_rows(rows=rows_by_route["new-heldout"], model=model, tokenizer=tokenizer,
                         device=device, split_label="new-heldout")
    maximum, mismatches = K.TRH.compare_predictions(
        {r["id"]: r for r in scores["new-heldout"]["predictions"]}, fresh["predictions"])
    verification = {"alias": alias, "checkpoint": "terminal-300", "n": len(fresh["predictions"]),
                    "max_probability_delta": maximum, "argmax_mismatches": mismatches,
                    "ok": mismatches == 0 and maximum <= 1e-6, "reload": second_reload,
                    "gpu": torch.cuda.get_device_name(device), "seconds": time.perf_counter() - start,
                    "runner_sha256": K.sha256_file(Path(__file__))}
    publish(output_dir / "independent-reload.json", verification)
    assert verification["ok"], verification
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
