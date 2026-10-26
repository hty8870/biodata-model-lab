                       
"""实验 A/B 的臂驱动器：按预注册网格逐臂调 `scripts/train_ltr.py`，账落在实验目录里。

网格（全部 `--deterministic`，评测集固定 = heldout（5 评委语义金标）+ ood_evalonly（OOD 面））：

- **实验 A（特征来源臂）**：`feature_set ∈ {f0,f1,f2} × data ∈ {d0,d1} × seed ∈ SEEDS`（30 臂），
  训练标签固定在现役口径 `mix`，**评测标签固定在 `sem`**。
- **实验 B（标签臂）**：`train_label ∈ {det,mix,sem} × data ∈ {d0,d1} × seed ∈ SEEDS`（30 臂），
  特征固定在 F1，**评测标签同样固定 `sem`**。

数据面靠 `--in-dir` / `--mix-in` 显式给（不许 MIX_IN_PATTERNS 隐式推）：
f0 臂用现役目录 `research/ltr_data` + 显式 mix-in；f1/f2 臂用构建器产出的已合并变体目录。

每个臂一个子目录：`metrics.json`（train_ltr 的原始账）与 `per_query.jsonl`（逐查询 nDCG@10 +
top-10 uid，配对分析用）。**臂产物落在未跟踪区**（`.`gitignore` 的 `research/ltr_data/_exp_*/`）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
EXP_DIR = HERE.parent
OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260929_ab"

PYTHON = AGENT_ROOT.parent / "agent" / ".venv" / "Scripts" / "python.exe"
TRAINER = AGENT_ROOT / "scripts" / "train_ltr.py"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
OOD_TRAINABLE = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_trainable_xscore.jsonl"
WAVE2 = LTR_DATA / "synth_train_wave2_20260929_xscore.jsonl"
OOD_EVALONLY_RAW = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_evalonly_xscore.jsonl"
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"

                                                   
SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)

FEATURE_SETS_A = ("f0", "f1", "f2")
LABELS_B = ("det", "mix", "sem")
DATA = ("d0", "d1")

                                            
TRAIN_LABEL_A = "mix"
EVAL_LABEL = "sem"


def arm_dir(arm_id: str) -> Path:
    return OUT_ROOT / "arms" / arm_id


def build_command(feature_set: str, data: str, seed: int, train_label: str, arm_id: str,
                  *, create_dirs: bool = True) -> "list[str]":
    """一臂的完整命令行（数据面显式给出，评测集固定两份）。`create_dirs=False` 供 --dry-run。"""
    cmd = [str(PYTHON), str(TRAINER), "--deterministic", "--feature-set", feature_set,
           "--train-label", train_label, "--eval-label", EVAL_LABEL, "--seed", str(seed)]
    if feature_set == "f0":
        cmd += ["--in-dir", str(LTR_DATA)]
    else:
        cmd += ["--in-dir", str(OUT_ROOT / "data" / f"{data}_{feature_set}")]
    if data == "d0":
        if feature_set == "f0":
            cmd += ["--mix-in", str(OOD_TRAINABLE)]
    elif feature_set == "f0":
        cmd += ["--mix-in", str(OOD_TRAINABLE), "--mix-in", str(WAVE2)]
    ood_eval = (OOD_EVALONLY_RAW if feature_set == "f0"
                else OUT_ROOT / "data" / "eval" / f"oodonly_{feature_set}.jsonl")
    heldout_gold = (HELDOUT_GOLD if feature_set == "f0"
                    else OUT_ROOT / "data" / "eval" / f"heldout5_{feature_set}.jsonl")
    for path in (ood_eval, heldout_gold):
        if not path.is_file():
            raise SystemExit(f"! 评测集缺失（先跑 build_f1_features.py）：{path}")
    out = arm_dir(arm_id)
    if create_dirs:
        out.mkdir(parents=True, exist_ok=True)
    cmd += ["--eval-set", f"heldout5={heldout_gold}",
            "--eval-set", f"oodonly={ood_eval}",
            "--per-query-out", str(out / "per_query.jsonl"),
            "--out", str(out / "metrics.json")]
    return cmd


def arm_ids() -> "list[tuple[str, dict]]":
    """预注册的 60 臂清单（A 30 + B 30）。"""
    arms: "list[tuple[str, dict]]" = []
    for feature_set in FEATURE_SETS_A:
        for data in DATA:
            for seed in SEEDS:
                arms.append((f"A_{feature_set}_{data}_{seed}",
                             {"experiment": "A", "feature_set": feature_set, "data": data, "seed": seed,
                              "train_label": TRAIN_LABEL_A, "eval_label": EVAL_LABEL}))
    for label in LABELS_B:
        for data in DATA:
            for seed in SEEDS:
                arms.append((f"B_{label}_{data}_{seed}",
                             {"experiment": "B", "feature_set": "f1", "data": data, "seed": seed,
                              "train_label": label, "eval_label": EVAL_LABEL}))
    return arms


def run(arm_id: str, meta: dict, log_path: Path) -> dict:
    cmd = build_command(meta["feature_set"], meta["data"], meta["seed"], meta["train_label"], arm_id)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(AGENT_ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    log_path.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
                        encoding="utf-8")
    entry = {**meta, "arm_id": arm_id, "returncode": proc.returncode,
             "elapsed_s": round(time.time() - started, 2), "command": cmd}
    if proc.returncode != 0:
        entry["stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
    return entry


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("A", "B"), default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="只跑这些 seed（缺省 = 预注册的 5 个）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    wanted_seeds = set(args.seeds) if args.seeds else set(SEEDS)
    plan = [(arm_id, meta) for arm_id, meta in arm_ids()
            if (args.only is None or meta["experiment"] == args.only) and meta["seed"] in wanted_seeds]
    print(f"{len(plan)} 个臂待跑（预注册 60 臂减去过滤）")
    if args.dry_run:
        for arm_id, meta in plan:
            print(" ", arm_id, " ".join(build_command(meta["feature_set"], meta["data"], meta["seed"],
                                                      meta["train_label"], arm_id, create_dirs=False)))
        return 0

    index_path = OUT_ROOT / "arms_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    t0 = time.time()
    failed = []
    for i, (arm_id, meta) in enumerate(plan, 1):
        out = arm_dir(arm_id)
        if (out / "metrics.json").is_file() and (out / "per_query.jsonl").is_file() and arm_id in index \
                and index[arm_id].get("returncode") == 0:
            print(f"  [{i}/{len(plan)}] {arm_id} 已存在，跳过")
            continue
        entry = run(arm_id, meta, out / "train.log")
        index[arm_id] = entry
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status = "ok" if entry["returncode"] == 0 else f"FAIL rc={entry['returncode']}"
        print(f"  [{i}/{len(plan)}] {arm_id} {status} ({entry['elapsed_s']}s, 累计 {time.time()-t0:.0f}s)")
        if entry["returncode"] != 0:
            failed.append(arm_id)

    print(f"\n完成：{len(plan) - len(failed)}/{len(plan)} 成功，{len(failed)} 失败"
          f"（总 {time.time()-t0:.0f}s）")
    for arm_id in failed:
        print("  FAIL", arm_id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
