                       
"""把 OOD 交叉打分集拆成「可训练层」与「只评测层」，让门 3 的评测集不再进训练。

为什么（治本）：`scripts/ltr_promote.py` 的第三门在 `research/ood_eval/` 的交叉打分集上
做冻结对照，而**同一个文件**又被 `MIX_IN_PATTERNS` 并进训练（`candidate` 的缺省 mix-in）——
冻结两个模型不会让训练数据变成测试数据，门 3 此前比的其实是"候选在自己见过的那批行上
是否不输现役"，评测不独立。本脚本把源文件按 **query_id 归一化哈希**一分为二：

- `synth_ood_trainable_xscore.jsonl`：可训练层，留在 `MIX_IN_PATTERNS` 里；
- `synth_ood_evalonly_xscore.jsonl`：只评测层，门 3 的缺省评测集。

名义比例 70/30（`--ratio`）；分桶是逐查询独立哈希，故实测比例在二项噪声内浮动
（当前切分：345/520 查询 = 0.6635，352/533 行 = 0.6604）。

源文件 `synth_ood_label_input_xscore.jsonl` 原样留档（不改、不删、不再被任何一侧引用），
它是这次拆分的输入凭据。

口径与不变量：
- **分桶只看 query_id**：同一 query_id 的**全部行**（A/B 复测的双行）必落同一侧，两侧
  query_id 集合不相交——行级随机会漏成两侧共享同一个查询。
- **哈希 + 固定 seed**（`--seed`，默认 20260929）而非 `random.shuffle`：拆分可复现，
  重跑逐字节一致，任何人任何机器上都是同一刀。
- **逐行搬运、不重新序列化**：输出行与源行**逐字节相同**（键序、浮点写法、空白都不动），
  换行统一为 LF（写盘显式钉住，不跟随平台翻译），避免"拆一趟顺手改了格式"这类无痕漂移。
- 归一化 = 去掉首尾与内部重复空白 + casefold；`ood-1` 与 `OOD-1 ` 分到同一侧。

用法：
    python scripts/split_ood_evalonly.py                     # 就地生成两份（幂等，重跑同字节）
    python scripts/split_ood_evalonly.py --check             # 只核对既有拆分，不写文件
    python scripts/split_ood_evalonly.py --ratio 0.6         # 换切分比例（须重跑门 3 基准）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "research" / "ood_eval" / "synth_ood_label_input_xscore.jsonl"
DEFAULT_OUT_DIR = ROOT / "research" / "ood_eval"

                                                  
TRAINABLE_NAME = "synth_ood_trainable_xscore.jsonl"
EVALONLY_NAME = "synth_ood_evalonly_xscore.jsonl"

                                              
                            
DEFAULT_SEED = 20260929

                                          
DEFAULT_TRAIN_RATIO = 0.7


def normalize_query_id(query_id: str) -> str:
    """归一化 query_id：空白折叠 + casefold（同一查询的书写差异不该分到两侧）。"""
    return " ".join(str(query_id).split()).casefold()


def bucket_of(query_id: str, seed: int = DEFAULT_SEED) -> float:
    """query_id → [0, 1) 的稳定分桶值（sha256 前 8 字节 / 2**64）。

    用哈希而不是 Python `hash()`：`hash()` 受 `PYTHONHASHSEED` 影响，跨进程不稳定，
    用它做拆分等于每次重跑换一份评测集。
    """
    digest = hashlib.sha256(f"{seed}:{normalize_query_id(query_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_lines(lines: "list[str]", *, seed: int = DEFAULT_SEED,
                train_ratio: float = DEFAULT_TRAIN_RATIO) -> "tuple[list[str], list[str]]":
    """把源行拆成 (可训练, 只评测) 两组，各自保持源顺序，行文本原样搬运。

    空行丢弃；解析不出 `query_id` 的行当场抛 ValueError（撕裂的尾行必须炸，
    不能静默丢行——丢了就是评测集悄悄缩水）。
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 必须在 (0,1) 内，收到 {train_ratio}")
    trainable: list[str] = []
    evalonly: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {lineno} 行不是合法 JSON（源文件被截断？）：{exc}") from exc
        query_id = row.get("query_id")
        if not query_id:
            raise ValueError(f"第 {lineno} 行缺 query_id——分桶无从进行")
        target = trainable if bucket_of(query_id, seed) < train_ratio else evalonly
        target.append(line)
    return trainable, evalonly


def query_ids_of(lines: "list[str]") -> "set[str]":
    return {normalize_query_id(json.loads(line)["query_id"]) for line in lines if line.strip()}


def _read_lines(path: Path) -> "list[str]":
    return path.read_text(encoding="utf-8").splitlines()


def render(lines: "list[str]") -> bytes:
    """两份派生文件的字节口径：每行一个 JSON + LF 结尾（**不跟随平台换行**）。

    `--check` 与写盘共用它，于是"重跑逐位一致"是字节级的：换行被翻译成 CRLF 也算漂移。
    """
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _write_lines(path: Path, lines: "list[str]") -> None:
    path.write_bytes(render(lines))


def _split_report(source: Path, trainable: "list[str]", evalonly: "list[str]") -> dict:
    train_ids, eval_ids = query_ids_of(trainable), query_ids_of(evalonly)
    total_rows = len(trainable) + len(evalonly)
    return {
        "source": str(source),
        "seed": DEFAULT_SEED,
        "rows": {"trainable": len(trainable), "evalonly": len(evalonly), "total": total_rows},
        "queries": {"trainable": len(train_ids), "evalonly": len(eval_ids),
                    "overlap": len(train_ids & eval_ids)},
        "train_ratio_rows": round(len(trainable) / total_rows, 4) if total_rows else None,
        "train_ratio_queries": round(len(train_ids) / (len(train_ids) + len(eval_ids)), 4)
        if (train_ids or eval_ids) else None,
    }


def _print_report(report: dict, paths: "tuple[Path, Path] | None" = None) -> None:
    rows, queries = report["rows"], report["queries"]
    print(f"源文件：{report['source']}（seed={report['seed']}）")
    print(f"  可训练层：{rows['trainable']} 行 / {queries['trainable']} 查询"
          f"（行占比 {report['train_ratio_rows']}，查询占比 {report['train_ratio_queries']}）")
    print(f"  只评测层：{rows['evalonly']} 行 / {queries['evalonly']} 查询")
    print(f"  两侧 query_id 交集：{queries['overlap']}（必须为 0）")
    if paths is not None:
        print(f"wrote {paths[0]}\nwrote {paths[1]}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                        help=f"源 OOD 交叉打分集（默认 {DEFAULT_IN.relative_to(ROOT).as_posix()}）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="可训练层占比（默认 0.7）")
    parser.add_argument("--check", action="store_true",
                        help="只按源文件重算并核对既有两份派生文件，不写盘")
    args = parser.parse_args(argv)

    source: Path = args.in_path
    if not source.is_file():
        print(f"! 源文件缺失：{source}", file=sys.stderr)
        return 1
    trainable, evalonly = split_lines(_read_lines(source), seed=args.seed, train_ratio=args.ratio)
    report = _split_report(source, trainable, evalonly)
    report["seed"] = args.seed
    out_dir: Path = args.out_dir
    train_path, eval_path = out_dir / TRAINABLE_NAME, out_dir / EVALONLY_NAME

    if args.check:
        drift = []
        for path, expected in ((train_path, trainable), (eval_path, evalonly)):
            if not path.is_file():
                drift.append(f"{path} 缺失")
            elif path.read_bytes() != render(expected):
                drift.append(f"{path} 与源文件重算结果不一致（内容或换行口径漂移）")
        _print_report(report)
        if drift:
            for item in drift:
                print(f"! {item}", file=sys.stderr)
            return 1
        print("check: 两份派生文件与源文件重算结果逐字节一致")
        return 0

    if report["queries"]["overlap"]:
        print("! 两侧 query_id 有交集——拆分逻辑不自洽，拒绝落盘", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(train_path, trainable)
    _write_lines(eval_path, evalonly)
    _print_report(report, (train_path, eval_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
