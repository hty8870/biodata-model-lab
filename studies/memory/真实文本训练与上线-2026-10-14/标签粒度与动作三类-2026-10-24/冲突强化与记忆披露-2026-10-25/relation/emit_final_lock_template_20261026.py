from __future__ import annotations

import argparse
import json
from pathlib import Path

import conflict_common as K
import score_public_regression_20261026 as O


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    arm, seed_text = args.alias.split("-seed")
    seed = int(seed_text)
    assert arm in K.SCORE_ROUTES and seed in K.SEEDS
    physical = f"direct3-seed{seed}" if arm == "r0" else args.alias
    original_runs = K.RUNS_DIR
    K.RUNS_DIR = K.PARENT_EXP / "runs" if arm == "r0" else K.RUNS_DIR
    try:
        template = K.lock_requirements(run=physical, checkpoint="terminal")
        directory = K.RUNS_DIR / physical
    finally:
        K.RUNS_DIR = original_runs
    files = K.checkpoint_file_shas(directory / "checkpoints/terminal")
    assert files
    for route in template["locked_routes"]:
        template["checkpoint_files_sha256"][f"{route}:terminal"] = files
    _, accounting = O.O.diagnostic_rows()
    template.update(created=K.utc_now(), alias=args.alias,
                    checkpoint="terminal-300", checkpoint_directory=str(directory / "checkpoints/terminal"),
                    semantic_review_sha256=K.sha256_file(K.HERE / "audits/semantic-review-final-20261026.json"),
                    ocnli_input_sha256=accounting["selected_input_sha256"],
                    ocnli_checkpoint_files_sha256=files,
                    template_emitter_sha256=K.sha256_file(Path(__file__)),
                    note="TEMPLATE ONLY: the main coordinator must inspect and publish the final lock; no final lock is written by this script.")
    print(json.dumps(template, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
