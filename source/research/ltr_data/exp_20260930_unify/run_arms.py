                       
"""R3 的臂驱动器：f0 vs f3 训练配方 × 5 seed，全部 --deterministic，预注册见
`研究/10-自我递归与自动训练/LTR特征单源化R3-预注册-2026-09-30.md`。

- `R_F0_<seed>`：现役配方（in-dir=staging：标准 train 层 + f3 化 heldout 层；
  并入 OOD-trainable + wave2），训练特征 f0、**评测特征 f3**（生产服役语义）；
- `R_F3_<seed>`：单源配方（in-dir=`_exp_20260930_f3/data/d1_f3`，预合并变体，无 mix-in），
  训练/评测特征均 f3；
- 两臂评测集完全相同：heldout5f3 + oodonlyf3（生产池形状）。

staging 的必要性：train_ltr 的 in-dir heldout 永远会被评测；评测特征为 f3 时，
标准 heldout 行没有 f3_feats 会按契约抛错（fail-closed），故 F0 臂的 heldout 层
staging 成 f3 化文件。train 层逐字节 = 现役（训练数据不许变）。

臂账与实验 S/W23 同构，产物落 gitignored `research/ltr_data/_exp_20260930_unify/`。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
OUT_ROOT = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_unify"

PYTHON = AGENT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = AGENT_ROOT / "scripts" / "train_ltr.py"
LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
F3_DATA = AGENT_ROOT / "research" / "ltr_data" / "_exp_20260930_f3" / "data"
OOD_TRAINABLE = AGENT_ROOT / "research" / "ood_eval" / "synth_ood_trainable_xscore.jsonl"
WAVE2 = LTR_DATA / "synth_train_wave2_20260929_xscore.jsonl"
HELDOUT5_F3 = F3_DATA / "eval" / "heldout5_f3.jsonl"
OODONLY_F3 = F3_DATA / "eval" / "oodonly_f3.jsonl"
F0_STAGE = OUT_ROOT / "data" / "f0train"

SEEDS = (20260910, 20260917, 20260924, 20261001, 20261008)


def stage_f0_dir() -> None:
    """F0 臂的 in-dir：train 层逐字节 = 现役；heldout 层 = heldout5_f3（带 f3_feats）。"""
    F0_STAGE.mkdir(parents=True, exist_ok=True)
    for name in ("synth_train_vec.jsonl", "synth_train_labeled.jsonl",
                 "synth_train_nl.jsonl", "synth_train.jsonl"):
        shutil.copyfile(LTR_DATA / name, F0_STAGE / name)
    shutil.copyfile(HELDOUT5_F3, F0_STAGE / "synth_heldout_vec.jsonl")


def build_command(arm: str, seed: int, arm_id: str, *, create_dirs: bool = True) -> list[str]:
    out = OUT_ROOT / "arms" / arm_id
    if create_dirs:
        out.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(TRAINER), "--deterministic",
           "--train-label", "mix", "--eval-label", "sem", "--seed", str(seed)]
    if arm == "F0":
        cmd += ["--feature-set", "f0", "--eval-feature-set", "f3",
                "--in-dir", str(F0_STAGE),
                "--mix-in", str(OOD_TRAINABLE), "--mix-in", str(WAVE2)]
    elif arm == "F0P":
                                                      
                                                         
                                   
        cmd += ["--feature-set", "f0", "--eval-feature-set", "f3",
                "--in-dir", str(F3_DATA / "d1_f3")]
    else:
        cmd += ["--feature-set", "f3", "--eval-feature-set", "f3",
                "--in-dir", str(F3_DATA / "d1_f3")]
    cmd += ["--eval-set", f"heldout5f3={HELDOUT5_F3}",
            "--eval-set", f"oodonlyf3={OODONLY_F3}",
            "--per-query-out", str(out / "per_query.jsonl"),
            "--out", str(out / "metrics.json")]
    return cmd


def arm_ids() -> list[tuple[str, str, int]]:
    return [(f"R_{arm}_{seed}", arm, seed) for arm in ("F0", "F3", "F0P") for seed in SEEDS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for path in (HELDOUT5_F3, OODONLY_F3, F3_DATA / "d1_f3" / "synth_train.jsonl"):
        if not path.is_file():
            raise SystemExit(f"! f3 数据缺失（先回迁/重建 _exp_20260930_f3）：{path}")
    stage_f0_dir()

    if args.dry_run:
        for arm_id, arm, seed in arm_ids():
            print(" ", arm_id, " ".join(build_command(arm, seed, arm_id, create_dirs=False)))
        return 0

    index_path = OUT_ROOT / "arms_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    t0 = time.time()
    failed = []
    plan = arm_ids()
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
