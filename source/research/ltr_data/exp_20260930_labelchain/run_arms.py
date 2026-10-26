                       
"""实验 S 的臂驱动器：2 臂 × 5 seed，全部 --deterministic，复用 W23 的运行模式与账本格式。

- `S_SQ_<seed>`：现役生产链（in-dir = research/ltr_data，vec 层 = 单评委）；
- `S_CONS_<seed>`：staging 目录（唯一差异 = 基础层换 3 评委共识标签，先跑 stage_data.py）；
- 两臂的并入层（OOD-trainable + wave2）、评测三面（heldout5/oodonly/heldsingle）、
  train-label mix、eval-label sem、feature-set f0 完全相同。

臂账与 W23 同构（arms_index.json + arms/<id>/{metrics.json,per_query.jsonl,train.log}），
产物落未跟踪区 `research/ltr_data/_exp_20260930_labelchain/`。已完成的臂自动跳过（幂等重入）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_labelchain"

PYTHON = AGENT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = AGENT_ROOT / "scripts" / "train_ltr.py"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
CONS_DIR = OUT_ROOT / "data" / "cons"
OOD_TRAINABLE = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_trainable_xscore.jsonl"
WAVE2 = LTR_DATA / "synth_train_wave2_20260929_xscore.jsonl"
HELDOUT_GOLD = LTR_DATA / "synth_heldout_xscore.jsonl"
OOD_EVALONLY = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_evalonly_xscore.jsonl"
HELDOUT_SINGLE = LTR_DATA / "synth_heldout_vec.jsonl"

SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)
ARMS = ("SQ", "CONS")


def build_command(arm: str, seed: int, arm_id: str, *, create_dirs: bool = True) -> list[str]:
    in_dir = LTR_DATA if arm == "SQ" else CONS_DIR
    out = OUT_ROOT / "arms" / arm_id
    if create_dirs:
        out.mkdir(parents=True, exist_ok=True)
    return [str(PYTHON), str(TRAINER), "--deterministic",
            "--feature-set", "f0", "--train-label", "mix", "--eval-label", "sem",
            "--seed", str(seed),
            "--in-dir", str(in_dir),
            "--mix-in", str(OOD_TRAINABLE), "--mix-in", str(WAVE2),
            "--eval-set", f"heldout5={HELDOUT_GOLD}",
            "--eval-set", f"oodonly={OOD_EVALONLY}",
            "--eval-set", f"heldsingle={HELDOUT_SINGLE}",
            "--per-query-out", str(out / "per_query.jsonl"),
            "--out", str(out / "metrics.json")]


def arm_ids() -> list[tuple[str, str, int]]:
    return [(f"S_{arm}_{seed}", arm, seed) for arm in ARMS for seed in SEEDS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not (CONS_DIR / "synth_train_vec.jsonl").is_file():
        raise SystemExit("! staging 不存在，先跑 stage_data.py（含奇偶校验）")

    plan = arm_ids()
    if args.dry_run:
        for arm_id, arm, seed in plan:
            print(" ", arm_id, " ".join(build_command(arm, seed, arm_id, create_dirs=False)))
        return 0

    index_path = OUT_ROOT / "arms_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    t0 = time.time()
    failed = []
    for i, (arm_id, arm, seed) in enumerate(plan, 1):
        out = OUT_ROOT / "arms" / arm_id
        if (out / "metrics.json").is_file() and (out / "per_query.jsonl").is_file() \
                and index.get(arm_id, {}).get("returncode") == 0:
            print(f"  [{i}/{len(plan)}] {arm_id} 已存在，跳过")
            continue
        cmd = build_command(arm, seed, arm_id)
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(AGENT_ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        (out / "train.log").write_text(
            proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8")
        entry = {"arm_id": arm_id, "arm": arm, "seed": seed, "returncode": proc.returncode,
                 "elapsed_s": round(time.time() - started, 2), "command": cmd}
        if proc.returncode != 0:
            entry["stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
            failed.append(arm_id)
        index[arm_id] = entry
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        print(f"  [{i}/{len(plan)}] {arm_id} "
              f"{'ok' if proc.returncode == 0 else 'FAIL'} ({entry['elapsed_s']}s, "
              f"累计 {time.time() - t0:.0f}s)")

    print(f"\n完成：{len(plan) - len(failed)}/{len(plan)} 成功（总 {time.time() - t0:.0f}s）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
