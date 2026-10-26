                       
"""Build the union labelling input for OOD candidates and merge labels back.

``synth_ood_label_input_vec.jsonl``: same rows as ``synth_ood_vec.jsonl`` but
with ``candidates`` = union of track-A and track-B candidates (deduped by uid,
track-A fields preferred) so llm_cross_score can label every candidate in one
pass (file name follows its ``synth_{split}_vec.jsonl`` convention for
``--splits ood_label_input``). ``merge_back`` then writes each pair's
llm_semantic/xscore into both tracks of ``synth_ood_vec.jsonl`` in place
(backed up once as .prelabel).
Internal research tool.

Usage:
  python prep_label_input.py                       # union -> the OOD labelling input
  python prep_label_input.py --src <vec.jsonl> --dst <union.jsonl>
  python prep_label_input.py merge                 # labels back into synth_ood_vec.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OOD = Path(__file__).resolve().parent


def prep(src: Path, dst: Path) -> None:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    n_union = 0
    for r in rows:
        by_uid: dict[str, dict] = {}
        for c in r["candidates"] + r.get("candidates_rescue", []):
            uid = c.get("uid")
            if uid and uid not in by_uid:
                by_uid[uid] = c
        n_union += len(by_uid)
        row = dict(r)
        row["candidates"] = list(by_uid.values())
        out.append(row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"union pairs: {n_union} across {len(out)} queries -> {dst}")


def merge_back() -> None:
    labeled: dict[tuple[str, str], dict] = {}
    src = OOD / "synth_ood_label_input_xscore.jsonl"
    for l in src.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        for c in r["candidates"]:
            labeled[(r["query_id"], c["uid"])] = {
                k: c.get(k) for k in ("llm_semantic", "llm_semantic_single", "xscore",
                                      "xscore_n", "xscore_spread", "label_model")
            }
    main = OOD / "synth_ood_vec.jsonl"
    bak = OOD / "synth_ood_vec.jsonl.prelabel"
    if not bak.exists():
        shutil.copyfile(main, bak)
    rows = [json.loads(l) for l in main.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    for r in rows:
        for track in ("candidates", "candidates_rescue"):
            for c in r[track]:
                lab = labeled.get((r["query_id"], c.get("uid") or ""))
                if lab:
                    c.update(lab)
                    n += 1
    main.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    print(f"merged labels into {n} candidate slots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=OOD / "synth_ood_vec.jsonl",
                        help="harvested pairs to union（默认 OOD 收割文件）")
    parser.add_argument("--dst", type=Path, default=OOD / "synth_ood_label_input_vec.jsonl",
                        help="union labelling input to write")
    parser.add_argument("command", nargs="?", choices=("prep", "merge"), default="prep")
    args = parser.parse_args()
    if args.command == "merge":
        merge_back()
    else:
        prep(args.src, args.dst)
