from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import conflict_common as K
import score_conflict as S

sys.path.insert(0, str(K.ROOT / "baseline-regression"))
import ocnli_conflict_20261025 as O


def check_public_lock(lock, entry, accounting, files):
    assert lock["run"] == entry["run"]
    assert lock["run_lock_sha256"] == K.sha256_file(entry["directory"] / "run-lock.json")
    assert lock["ocnli_input_sha256"] == accounting["selected_input_sha256"]
    assert lock["ocnli_checkpoint_files_sha256"] == files
    assert entry["lock"]["protocol"]["updates"] == 300
    assert K.read_json(entry["directory"] / "result.json")["selection"]["terminal_step"] == 300


def selftest():
    O.selftest()
    rows, accounting = O.diagnostic_rows()
    assert len(rows) == 1502 and accounting["sealed_exposed_to_model"] == 0
    assert accounting["gold_counts"]["contradiction"] == 452
    try:
        O.summarize(rows, np.zeros((len(rows), 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid probabilities accepted")
    import tempfile
    K.SELFTEST_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=K.SELFTEST_DIR) as temporary:
        directory = Path(temporary)
        K.write_json_atomic(directory / "run-lock.json", {"fixture_only": True})
        K.write_json_atomic(directory / "result.json", {"selection": {"terminal_step": 300}})
        entry = {"run": "fixture-only", "directory": directory, "lock": {"protocol": {"updates": 300}}}
        files = {"adapter/fixture": "expected"}
        lock = {"run": entry["run"], "run_lock_sha256": K.sha256_file(directory / "run-lock.json"),
                "ocnli_input_sha256": accounting["selected_input_sha256"],
                "ocnli_checkpoint_files_sha256": files}
        check_public_lock(lock, entry, accounting, files)
        for changed in (dict(lock, ocnli_input_sha256="wrong"),
                        dict(lock, ocnli_checkpoint_files_sha256={"adapter/fixture": "wrong"})):
            try:
                check_public_lock(changed, entry, accounting, files)
            except AssertionError:
                pass
            else:
                raise AssertionError("Mismatched public lock accepted")
    print(json.dumps({"public_adapter_checks": 6, "passed": 6,
                      "selected_input_sha256": accounting["selected_input_sha256"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--unlock-final", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run or not args.unlock_final:
        parser.error("--run and --unlock-final are required")
    entry = S.resolve_run(args.run, None)
    rows, accounting = O.diagnostic_rows()
    lock = K.load_lock(args.unlock_final)
    files = K.checkpoint_file_shas(entry["directory"] / "checkpoints/terminal")
    check_public_lock(lock, entry, accounting, files)
    output_path = entry["directory"] / "scores/ocnli-diagnostic-terminal.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    device = S._cuda("cuda:0")
    start = time.perf_counter()
    model, tokenizer, reload_info = S.load_model(entry, checkpoint="terminal", device=device)
    length = O.T.C.assert_fits(tokenizer, [r["visible_text"] for r in rows], K.MAX_LENGTH, "OCNLI diagnostic")
    probabilities = O.T.forward_probabilities(model, tokenizer, rows, "direct3", device, K.MAX_LENGTH,
                                              True, O.T.default_encode(tokenizer, device, K.MAX_LENGTH), batch_size=16)
    torch.cuda.synchronize(device)
    summary = O.summarize(rows, probabilities)
    gold = np.asarray([r["gold_nli"] == "contradiction" for r in rows])
    pred = probabilities.argmax(axis=1) == O.T.COARSE_CONFLICT
    summary["accuracy_cluster_ci"] = O.EM.cluster_bootstrap(
        [r["cluster_id"] for r in rows],
        lambda indices, weights: float(np.average((gold == pred)[indices], weights=weights)),
        samples=500, confidence=.95, seed=20260919)
    result = {"schema": "ocnli-conflict-transfer-diagnostic/1", "run": entry["run"],
              "checkpoint": "terminal-300", "seed": entry["seed"],
              "not_three_class_NLI_score": True, "not_new_blind_final": True,
              "accounting": accounting, "checkpoint_files_sha256": files,
              "runner_sha256": K.sha256_file(Path(__file__)), "reload": reload_info,
              "device": torch.cuda.get_device_name(device), "length_profile": length,
              "elapsed_seconds": time.perf_counter() - start, "summary": summary,
              "predictions": [{"id": row["id"], "cluster_id": row["cluster_id"],
                               "gold_nli": row["gold_nli"], "coarse_probabilities": probabilities[i].tolist(),
                               "predicted_coarse": K.COARSE[int(probabilities[i].argmax())]}
                              for i, row in enumerate(rows)]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
