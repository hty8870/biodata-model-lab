                       
"""LTR retraining experiment grid for cross-model consensus labels.

Three staging configurations (via the vec-layer copy trick, no code change to
train_ltr.py — a file named ``synth_<split>_vec.jsonl`` is picked up as the
richest layer automatically):

- ``exp_xscore``:  train=new consensus labels, eval=new consensus labels
- ``exp_cross``:   train=old single labels, eval=new consensus labels
- (reference)      main dir: train=old, eval=old — existing _ltr_metrics_*.json

Each config runs ``--label sem`` and ``--label mix`` with the default seed.
Internal research tool; not part of the public delivery surface.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "ltr_data"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SEED = "20260910"

CONFIGS = {
    "exp_xscore": {"train": "xscore", "heldout": "xscore"},
    "exp_cross": {"train": "vec", "heldout": "xscore"},
}


def stage(name: str, train_layer: str, held_layer: str) -> Path:
    d = DATA / f"_{name}"
    d.mkdir(exist_ok=True)
    for split, layer in (("train", train_layer), ("heldout", held_layer)):
        src = DATA / f"synth_{split}_{layer}.jsonl"
        dst = d / f"synth_{split}_vec.jsonl"
        shutil.copyfile(src, dst)
    return d


def run_one(in_dir: Path, label: str, out: Path) -> dict:
    cmd = [str(PY), "-X", "utf8", str(ROOT / "scripts" / "train_ltr.py"),
           "--in-dir", str(in_dir), "--label", label, "--seed", SEED, "--out", str(out)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"train_ltr failed for {in_dir.name}/{label}")
    print(f"  [{in_dir.name}/{label}] done in {dt:.0f}s -> {out.name}")
    return json.loads(out.read_text(encoding="utf-8"))


def key_metrics(payload: dict) -> dict:
    res = payload["results"]
    return {
        "lex@5": res["baselines"]["lex order"]["nDCG@5"],
        "linear@5": res["baselines"]["linear a=0.5"]["nDCG@5"],
        "rrf@5": res["baselines"]["rrf k=60"]["nDCG@5"],
        "gbdt_full@5": res["models"]["hist_gbdt"]["full"]["nDCG@5"],
        "ranknet_full@5": res["models"]["ranknet_mlp"]["full"]["nDCG@5"],
        "ranknet_full@10": res["models"]["ranknet_mlp"]["full"]["nDCG@10"],
        "ranknet_full_MRR": res["models"]["ranknet_mlp"]["full"]["MRR"],
    }


def main() -> None:
    table = {}
    for name, layers in CONFIGS.items():
        d = stage(name, layers["train"], layers["heldout"])
        for label in ("sem", "mix"):
            out = DATA / f"_ltr_metrics_{label}_{name}.json"
            table[f"{label}/{name}"] = key_metrics(run_one(d, label, out))
    for label in ("sem", "mix"):
        ref = json.loads((DATA / f"_ltr_metrics_{label}.json").read_text(encoding="utf-8"))
        table[f"{label}/ref_old_labels"] = key_metrics(ref)

    cols = ["lex@5", "linear@5", "rrf@5", "gbdt_full@5", "ranknet_full@5", "ranknet_full@10", "ranknet_full_MRR"]
    print(f"\n{'config':<26}" + "".join(f"{c:>16}" for c in cols))
    for cfg, m in table.items():
        print(f"{cfg:<26}" + "".join(f"{(m[c] if m[c] is not None else float('nan')):>16.4f}" for c in cols))
    (DATA / "_xscore_experiment_table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote _xscore_experiment_table.json")


if __name__ == "__main__":
    main()
