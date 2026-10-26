                       
"""实验 S 的 staging：构造 CONS 臂的数据目录，并跑奇偶校验（已知答案，不过则炸）。

staging 技巧（与 Sep 11 run_xscore_experiments.py 同款、不改 train_ltr.py）：在独立目录里放
一份名为 ``synth_train_vec.jsonl`` 的文件，内容 = ``synth_train_xscore.jsonl``——
``LAYER_SUFFIXES``  richest-first 会把它当作最富层读入。heldout 层放现役单评委文件原样拷贝，
保证两臂的 in-dir 缺省 heldout 也完全一致（分析只走 --eval-set 的具名面，in-dir heldout 不进判据）。

奇偶校验（预注册 §6.1）：
1. 两臂 load_split("train")/("heldout") 的查询集相同；
2. 逐查询 f0 特征矩阵（train_ltr.candidate_features）逐位相等——特征不该感知标签来源；
3. 候选级字段差异只允许落在语义标签族（llm_semantic / llm_semantic_single / xscore* /
   llm_reason / label_model）；出现其他字段差异即 fail；
4. heldout 行两臂完全一致（同一来源文件拷贝）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]
sys.path.insert(0, str(AGENT_ROOT / "scripts"))
import train_ltr              

LTR_DATA = AGENT_ROOT / "research" / "ltr_data"
OUT_ROOT = LTR_DATA / "_exp_20260930_labelchain"
CONS_DIR = OUT_ROOT / "data" / "cons"

LABEL_FIELDS = {"llm_semantic", "llm_semantic_single", "xscore", "xscore_n", "xscore_spread",
                "llm_reason", "label_model"}


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage() -> None:
    """写 CONS 数据目录。train 层必须按**基础文件行序**重排：`load_split` 合并后的行序 =

    最 plain 层（synth_train.jsonl）的行序，而行序会经 build_training_matrix 影响训练动态——
    不重排会让「行序」混进自变量。候选内序已验证与 vec 层特征逐位一致，保持 xscore 原序即可。
    """
    CONS_DIR.mkdir(parents=True, exist_ok=True)
    base_order = [row["query_id"] for row in _load_rows(LTR_DATA / "synth_train.jsonl")]
    xs_by_qid = {row["query_id"]: row for row in _load_rows(LTR_DATA / "synth_train_xscore.jsonl")}
    missing = [q for q in base_order if q not in xs_by_qid]
    if missing:
        raise SystemExit(f"! xscore 层缺 {len(missing)} 个查询，无法保序 staging：{missing[:5]}")
    with (CONS_DIR / "synth_train_vec.jsonl").open("w", encoding="utf-8") as fh:
        for qid in base_order:
            fh.write(json.dumps(xs_by_qid[qid], ensure_ascii=False) + "\n")
    shutil.copyfile(LTR_DATA / "synth_heldout_vec.jsonl", CONS_DIR / "synth_heldout_vec.jsonl")
    print(f"staged: {CONS_DIR}")
    print(f"  synth_train_vec.jsonl   <- synth_train_xscore.jsonl（按 synth_train.jsonl 行序重排）")
    print(f"  synth_heldout_vec.jsonl <- synth_heldout_vec.jsonl    （与现役逐字节相同）")


def parity() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    sq_train, sq_info = train_ltr.load_split(LTR_DATA, "train")
    cons_train, cons_info = train_ltr.load_split(CONS_DIR, "train")
    sq_held, _ = train_ltr.load_split(LTR_DATA, "heldout")
    cons_held, _ = train_ltr.load_split(CONS_DIR, "heldout")

    sq_qids = [r["query_id"] for r in sq_train]
    cons_qids = [r["query_id"] for r in cons_train]
    check("查询集一致且行序相同（train）", sq_qids == cons_qids,
          f"SQ={len(sq_qids)} CONS={len(cons_qids)} 顺序相同={sq_qids == cons_qids}")
    check("SQ 层账符合现役（vec 层 1103 查询）",
          sq_info["layers"].get("synth_train_vec.jsonl") == 1103 and len(sq_train) == 1103,
          f"layers={sq_info['layers']}")

    cons_by_qid = {r["query_id"]: r for r in cons_train}
    n_feat_mismatch = 0
    bad_fields: dict[str, int] = {}
    n_label_diff = 0
    for row in sq_train:
        other = cons_by_qid[row["query_id"]]
        f_sq, _, _ = train_ltr.candidate_features(row, "f0")
        f_cons, _, _ = train_ltr.candidate_features(other, "f0")
        a = np.asarray(f_sq, dtype=np.float64)
        b = np.asarray(f_cons, dtype=np.float64)
        if a.shape != b.shape or not np.array_equal(a, b):
            n_feat_mismatch += 1
        c_sq = {c["uid"]: c for c in row.get("candidates") or []}
        c_cons = {c["uid"]: c for c in other.get("candidates") or []}
        if set(c_sq) != set(c_cons):
            bad_fields["<uid 集合不同>"] = bad_fields.get("<uid 集合不同>", 0) + 1
            continue
        for uid in c_sq:
            keys = set(c_sq[uid]) | set(c_cons[uid])
            for key in keys - LABEL_FIELDS:
                if c_sq[uid].get(key) != c_cons[uid].get(key):
                    bad_fields[key] = bad_fields.get(key, 0) + 1
            if c_sq[uid].get("llm_semantic") != c_cons[uid].get("llm_semantic"):
                n_label_diff += 1
    check("f0 特征矩阵逐位相等（全 1103 查询）", n_feat_mismatch == 0,
          f"mismatch_queries={n_feat_mismatch}")
    check("非标签族字段零差异", not bad_fields, f"bad_fields={bad_fields or '{}'}")
    check("语义标签确实不同（自变量生效）", n_label_diff > 5000,
          f"label_diff_cands={n_label_diff}/11311")

    def _rows_by_qid(rows: list[dict]) -> dict[str, dict]:
        return {r["query_id"]: r for r in rows}

    def _canon(rows_by_qid: dict[str, dict]) -> str:
                                                         
                                         
        return json.dumps({q: rows_by_qid[q] for q in sorted(rows_by_qid)},
                          sort_keys=True, ensure_ascii=False)

    same_held = _canon(_rows_by_qid(sq_held)) == _canon(_rows_by_qid(cons_held))
    check("heldout 行两臂一致（行序无关）", same_held, f"queries={len(sq_held)}")

    print(f"parity: {'全部通过' if not failures else '失败 ' + ', '.join(failures)}")
    return 1 if failures else 0


def main() -> int:
    stage()
    return parity()


if __name__ == "__main__":
    raise SystemExit(main())
