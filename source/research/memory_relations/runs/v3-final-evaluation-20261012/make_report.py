"""从一次评价持久工件生成可核对的报告，不进行推理或调参。"""
from collections import Counter
import json

import binding as B

P = B.BASE / "reports"


def pct(number):
    return "null" if number is None else f"{number * 100:.2f}%"


def ci(interval):
    return "null" if interval is None else f"[{pct(interval[0])}, {pct(interval[1])}]"


def ratio(record):
    return f"{record['numerator']:g}/{record['denominator']:g} ({pct(record['value'])})"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]


def main():
    B.verify()
    summary = B.read_json(P / "verified-summary.json")
    binding = B.read_json(B.BINDING)
    execution = B.read_json(P / "execution-result.json")
    old_full = B.read_json(P / "old-exposed-set-result.json")
    old_rows = B.read_jsonl(P / "old-exposed-set.predictions.jsonl")
    gated_rows = B.read_jsonl(P / "heldout.predictions.jsonl") + B.read_jsonl(P / "ood.predictions.jsonl")
    structural = B.read_json(P / "structural-boundary-result.json")
    failure_meta = [{"id": r["row_id"], "root": r["group_id"], "reference": r["label"],
                     "rules": r["rules"], "raw_relation": r["raw_decision"]["relation"] if r["raw_decision"] else None,
                     "gated_relation": r["decision"]["relation"] if r["decision"] else None,
                     "code": r["error"], "completed": r["completed"]} for r in gated_rows if not r["completed"]]
    cross_uncertain = [{"id": r["row_id"], "rules": r["rules"],
                        "batch_relations": [pred["relation"] for pred in r["predictions"]],
                        "merged_relation": r["decision"]["relation"], "error": r["error"],
                        "reference": r["label"], "completed": r["completed"]} for r in old_rows
                       if len(r["predictions"]) > 1 and any(pred["relation"] == "conflict" for pred in r["predictions"])
                       and any(pred["relation"] == "uncertain" for pred in r["predictions"])]
    oracle_mismatches = [{"id": r["row_id"], "reference": r["label"], "oracle_decision": r["oracle"]["decision"]}
                         for r in old_rows if r["oracle"]["available"] and not r["oracle"]["primary_agree"]]
    diagnostics = {"actual_terminal_failure_metadata": failure_meta, "cross_batch_uncertain_over_conflict": cross_uncertain,
                   "old_oracle_mismatches": oracle_mismatches,
                   "old_oracle_missing_reason_counts": dict(Counter(r["oracle"].get("reason") for r in old_rows if not r["oracle"]["available"])),
                   "prediction_calls_in_report_generation": 0}
    B.save_json(P / "causes-and-aggregation-diagnostics.json", diagnostics)
    lines = ["# 锁定记忆关系模型：正式单次评价与旧set诊断", "",
             "## 1. 结论与边界", "",
             "已完成一次真实CUDA评价，不是准备阶段。所有分数都是冻结AI代理参考上的 **proxy agreement**，不是真人accuracy、生产安全或风险证书。模型、数据、类序、温度、阈值和聚合均未调整。",
             "872个原request全部至少真实预测一次；871个有参考（heldout444/OOD427），1个excluded无truth仍进入872注册表并真实预测。accepted uncertain保留98行（47/51），不剔除。",
             "独立统计单位为40+40个语义根；根等权为主，行等权为次。95%区间复用metrics.cluster_bootstrap、500次根重采样，不把变体行当独立根；不衡量真人核验误差或真实用户分布。", "",
             "## 2. 已锁定的输入和实际入口", "",
             f"- Model: `{B.MODEL.relative_to(B.ROOT).as_posix()}`",
             f"- Weights SHA256: `{summary['model_weights_sha256']}`",
             f"- Meta SHA256: `{summary['model_meta_sha256']}`",
             f"- Lock SHA256: `{summary['lock_sha256']}`",
             f"- 温度 `{summary['temperature']}`；policy threshold=null/feasible=false；candidate point=null。τ=0只用于明确未校准诊断，不是选中的运行策略。",
             "- 唯一加载器 `features.load_scorer(M, device='cuda:0')` → 原 `finetune.CrossEncoderScorer.predict_proba`；不调温度、不换模型。",
             "- 原概率经内容+模型+源码版本绑定缓存，`evaluate.make_evaluation_predictor` → 原 `adapter.make_predictor` → 真 `M.assess_relations`/`evaluate.evaluate_sets`。set raw已经过pair gate，不是classifier原始argmax。",
             "- 原M meta直接用作来源，旧set未改用single-audit导出meta；weights仍为同一唯一锁定工件。",
             "- 入口实际为 `run_once.ps1` → `formal_evaluation.py --run --device cuda:0 --diagnostic --threshold 0 --oracle-aggregation`；完整argv/退出码见 `formal-cli-result.json`。",
             "- exclusive `formal-evaluation-attempt.json` 与 `execution-started.json` 保留；绑定时间先于预测，所有失败/分数保全。初始预测0，无重复正式尝试或另建study/audit。",
             "- 36个原源码/模型/lock/输入文件前后hash不变；完整路径与SHA见 `../source-binding.json`，输出SHA见 `output-hashes.json`。", "",
             "| Frozen reference file | SHA256 |", "| --- | --- |"]
    for name in ("final-heldout.jsonl", "final-ood.jsonl", "expected-cases.jsonl", "reference_status.jsonl"):
        lines.append(f"| {name} | `{binding['protected_sha256'][str(B.REF / name)]}` |")
    lines += ["", "## 3. Classifier原始argmax（闸前）", "",
              "这里的raw指机械闸前argmax；概率已按锁定正温度校准，正温度不改变argmax。准确性分母仅有参考的871行（444/427），不是872，也不是legacy valid-only。", ""]
    rows = []
    for split in ("heldout", "ood", "all"):
        view = summary["terminal"][split]["views"]["classifier"]
        rows.append([split, f"{view['known_reference_n']} / {view['n_planned']}", view["known_reference_roots"],
                     pct(view["root_agreement"]), ci(view["root_agreement_ci"]), pct(view["root_macro_f1"]),
                     ci(view["root_macro_f1_ci"]), f"{view['confirmed_correct']}/{view['known_reference_n']} ({pct(view['row_agreement'])})", pct(view["row_macro_f1"])])
    lines += table(["split", "reference/planned", "roots", "root agreement", "95%CI", "root macroF1", "95%CI", "row agreement", "row macroF1"], rows)
    lines += ["", "### 七类support与classifier混淆（行次口径）", "",
              "每行是真实参考类，每列是预测类；uncertain是第七类参考，不是缺标。根加权完整混淆和逐类P/R/F1保存在verified-summary.json。", ""]
    lines += table(["class", "heldout support", "OOD support", "all support"],
                   [[name, int(summary["terminal"]["heldout"]["views"]["classifier"]["support"][name]),
                     int(summary["terminal"]["ood"]["views"]["classifier"]["support"][name]),
                     int(summary["terminal"]["all"]["views"]["classifier"]["support"][name])] for name in summary["classes"]])
    for split in ("heldout", "ood"):
        lines += ["", f"**{split} classifier confusion**", ""]
        confusion = summary["terminal"][split]["views"]["classifier"]["confusion"]
        lines += table(["reference / prediction", *summary["classes"]],
                       [[name, *[int(value) for value in confusion[i][:-1]]] for i, name in enumerate(summary["classes"])])
    lines += ["", "## 4. Pair gate、set raw、M闸后与全N分母", "",
              "classifier和pair gate在本次872个单规则/空规则请求上argmax完全相同；set raw与pair gate也无关系类改写。M引用机械闸另产生1个真实失败，不得当成功uncertain。", ""]
    for source in ("pair_gate", "adapter_raw", "production_gated"):
        lines += [f"**{source}**", ""]
        lines += table(["split", "root agreement", "root macroF1", "row macroF1", "known correct", "completed / planned", "confirmed correct / planned"],
                       [[split, pct(v["root_agreement"]), pct(v["root_macro_f1"]), pct(v["row_macro_f1"]),
                         f"{v['confirmed_correct']}/{v['known_reference_n']}", ratio(v["completion"]), ratio(v["confirmed_correct_overall"])]
                        for split in ("heldout", "ood", "all") for v in [summary["terminal"][split]["views"][source]]])
        lines.append("")
    lines += table(["split", "M完成", "M correct_completed / Nplanned", "semantic uncertain", "system failure"],
                   [[split, ratio(report["completion"]), ratio(report["correct_completed_overall"]), report["semantic_uncertain"], report["system_failure"]]
                    for split, report in summary["terminal"].items()])
    lines += ["", "总体classifier confirmedcorrect=667/872，known-reference agreement=667/871，reference coverage=871/872；这些分母分别回答不同问题。",
              "M completion也恰好为871/872，但不是同一子集：excluded18014002完成了，另一个有参考case机械失败。未知reference的1行不当错误、不当semantic uncertain、不拿原作者label当gold。", ""]
    for failure in failure_meta:
        lines.append(f"- 真实失效：id `{failure['id']}`、root `{failure['root']}`、reference=`{failure['reference']}`、existing={failure['rules']}；raw=`{failure['raw_relation']}`，M回`{failure['gated_relation']}`/`{failure['code']}`，completed=false。空existing却预测需要related_ids的关系，引用闸按原合同拒绝。该行仍留在444已知分母和445/872全计划分母，绝不把失败转成语义uncertain得分。")
    excluded = summary["excluded_no_truth"]
    lines += [f"- 无参考行18014002：raw/pair/gated=`{excluded['gated_relation']}`，completed={excluded['gated_completed']}，reference_status=excluded，truth/correctness=null。", "",
              "### 诊断操作统计（不代表可行policy）", ""]
    lines += table(["split", "row allowing / planned", "row dangerous_allow / allowing", "root coverage", "root conditional_risk", "root false_block_rate"],
                   [[split, ratio(r["row_operation"]["coverage"]), ratio(r["row_operation"]["conditional_risk"]),
                     pct(r["root_operation"]["coverage"]["value"]), pct(r["root_operation"]["conditional_risk"]["value"]),
                     pct(r["root_operation"]["false_block_rate"]["value"])] for split, r in summary["terminal"].items()])
    lines += ["", "缺reference的放行在风险统计中只能标注unknown reference mass；conditional_risk是可确认错误的份额，不据此验证1%目标。", "",
              "## 5. 引用角色质量：raw/gated分开", "",
              "引用范围合法≠引用语义正确。adapter把输入evidence集合当宽泛引用；unknown id=0并不证明每条引用支持预测关系。只有871有参考行进入期望引用分母；excluded没有被人为补期望引用。", ""]
    lines += table(["split", "source", "role", "row precision", "row recall", "known nonexpected", "missed expected", "unknown IDs", "root precision [CI]", "root recall [CI]"],
                   [[split, source, role, ratio(stats["precision"]), ratio(stats["recall"]), stats["known_non_expected_total"],
                     stats["missed_expected_total"], stats["unknown_total"],
                     f"{pct(weighted['precision']['value'])} {ci(weighted['precision']['ci'])}",
                     f"{pct(weighted['recall']['value'])} {ci(weighted['recall']['ci'])}"]
                    for split, report in summary["terminal"].items()
                    for source in ("raw", "gated") for role, stats in report["refs_by_role_row"][source].items()
                    for weighted in [report["refs_by_role_root"][source][role]]])
    lines += ["", "root引用P/R的加权分子分母、每类support、失败列、所有95%CI与失效cause均保存在verified-summary.json；原始逐case引用列表保存在各split predictions.jsonl。", "",
              "## 6. 旧曝光set：全输入诊断，不是未见成绩", "",
              "旧set_challenge.jsonl的37输入/35根全部经同锁模型与原adapter/M执行，包含0/1/8/64/65/129条规则；无删题、修改label或聚合。真实入口evaluate_sets(..., oracle_aggregation=True)，显式diagnostic τ=0。", ""]
    old = summary["old_set"]
    lines += [f"- completion={ratio(old['completion'])}；correct_completed/Nplanned={ratio(old['correct_completed_overall'])}；semantic uncertain={old['semantic_uncertain']}，system failure={old['system_failure']}。",
              f"- 根等权set agreement={pct(old['gated_views']['root_agreement'])} {ci(old['gated_views']['root_agreement_ci'])}，macroF1={pct(old['gated_views']['root_macro_f1'])} {ci(old['gated_views']['root_macro_f1_ci'])}；行次macroF1={pct(old['gated_views']['row_macro_f1'])}。",
              f"- pair逻辑观察{old['scored_pairs']}对、{old['predictor_calls']}个真实M predictor批次；{old['pairs_without_reference']}对缺完整pair参考，不能把set label复制为pair gold。classifier/pair gate仅有{old['pair_views']['classifier']['known_reference_n']}对可对照；其七类support/混淆和根口径在verified-summary.json保留。",
              f"- 缺参考pairs={old['pairs_without_reference']}/{old['observed_pair_outputs']}不作系统失败；旧set的每行set参考仍在37分母里。", ""]
    lines += table(["rules", "planned", "completion", "correct_completed/N", "semantic uncertain", "observed predictor calls", "pair attempts", "oracle available / correct"],
                   [[count, row["planned"], f"{row['completed']}/{row['planned']}", f"{row['correct_completed']}/{row['planned']}", row["semantic_uncertain"],
                     json.dumps(row["predictor_call_counts"]), row["pair_attempts"], f"{row['oracle_available']} / {row['oracle_correct']}"]
                    for count, row in old["by_rule_count"].items()])
    oracle = old["oracle_aggregation"]
    lines += ["", f"**Oracle聚合诊断覆盖只有{oracle['rows']}/37**：10个空/单规则case有完整pair proxy标签，经同pair gate→adapter→M后{ratio(oracle['agreement'])}，1个关系差异。其余27个多规则case缺完整pair_labels，保留unavailable，不伪造、不拿这9/10当37/37或人工oracle。", ""]
    for row in oracle_mismatches:
        lines.append(f"- Oracle差异id `{row['id']}`：参考`{row['reference']}`，reference pair经公共gate/adapter/M得`{row['oracle_decision']['relation']}`，code=`{row['oracle_decision']['error_code']}`。该探针包含机械pair gate/生产gate，不是纯聚合或独立真人裁判。")
    lines += ["", "跨批uncertain覆盖conflict与引用并集沿用真实生产逻辑（不改成批内conflict优先）：", ""]
    for row in cross_uncertain:
        lines.append(f"- `{row['id']}` / {row['rules']} rules：批次`{row['batch_relations']}` → `{row['merged_relation']}`/`{row['error']}`；reference=`{row['reference']}`，completed={row['completed']}。")
    if not cross_uncertain:
        lines.append("- 本轮旧模型输出没有出现conflict+uncertain跨批组合；原机制由相关专项fixture验证，不把合成fixture当真实模型效果。")
    lines += ["", "旧set引用：", ""]
    lines += table(["source", "role", "precision", "recall", "known nonexpected", "missed", "unknown IDs"],
                   [[source, role, ratio(r["precision"]), ratio(r["recall"]), r["known_non_expected_total"], r["missed_expected_total"], r["unknown_total"]]
                    for source in ("raw", "gated") for role, r in old["refs_by_role_row"][source].items()])
    lines += ["", "oracle引用P/R分角色单列在old-exposed-set-result.json；批次间related_ids/evidence_refs按原M并集，不擅自按最高关系裁剪。范围合法仍可能known-nonexpected或漏标，均不冒充语义证据。", "",
              "## 7. 公开结构fixture与byte/context边界", "",
              "只用公开合成fixture，不标人工gold/新set成绩。实际锁模型预测，重复可见pair输入复用缓存；每个规则id都经过原M分批和引用闸，不top-k。", ""]
    lines += table(["rule count", "long rules", "actual batch sizes", "observed pairs", "final relation/code"],
                   [[r["rules"], r["long_rules"], [b["rules"] for b in r["batches"]], r["observed_pairs"],
                     f"{r['decision']['relation']} / {r['decision']['error_code']}"] for r in structural["rows"]])
    lines += ["", "当前M._MAX_REQUEST=32000实际比较ensure_ascii=False JSON的字符数，而不是UTF8字节数；两者均记录于structural-boundary-result.json，未修改门。长64规则递归二分为16×4；非聚合单字段/证据面超限不被切批治愈。",
              "context.batch由真实M生成index/count；存储schema只接受scope/权限context键，不能把batch context作为原存储request伪造。缓存键仅包含真joint输入与固定版本，id/权限context不进入分类特征，权限仍交原M重新验证。",
              "0规则fixture真实模型预测需要related的contextual后被M引用闸拒绝，不是成功semantic uncertain。input shape合法、pair有限、计数完整不代表最终关系可用。", "",
              "## 8. 物理预测、复用、runtime与不变性", ""]
    lines += table(["phase", "physical request/pair predictions", "calls", "forward batches", "failures"],
                   [[name, counts["completed_predictions"], counts["calls"], counts["forward_batches"], counts["failures"]]
                    for name, counts in execution["physical_predictions"].items()])
    lines += ["", "正式原request872次物理预测，后续classifier/pair/set/refs各视图均复用该次概率，adapter/M回放新增物理预测0；excluded同样被真预测。旧set1444逻辑pairs只新算1443（1缓存命中）；公开结构fixture只新算3个不同特征输入；公共重载探针2×2额外预测不属于终测。合计2322个物理预测，其中CPU2/CUDA2320。",
              f"CUDA:0 / RTX5070Ti Laptop / torch2.11.0+cu128，hidden384/max_length384/7类，参数{summary['runtime']['parameter_count']}；eval/no_grad，不创建optimizer或训练。全部872×7输出有限、[0,1]内、行和有效。",
              f"真实执行{execution['started_at']} → {execution['ended_at']}，Python内{execution['elapsed_seconds']:.3f}s；loader {summary['runtime']['loader_seconds']:.3f}s。峰allocated {summary['runtime']['peak_allocated_mb']:.3f}MB / reserved {summary['runtime']['peak_reserved_mb']:.3f}MB。",
              "开始GPU利用率0%、占用933MiB，既有Python PID28996等context明确记录且未结束/重启；CPU probe actor不占GPU。本阶段只此评价使用CUDA，系统context快照与本进程PID见runtime-final.json，不凭context存在断言有在训任务。",
              "公共F.load_scorer CPU重载对2个公开合成结构probe与CUDA argmax完全同，最大概率差1.4991371588557456e-08 <1e-6。weights/meta/lock和36个绑定输入/源码hash全程不变。",
              "Provider计费tokens=null/unknown，不从本地tokenizer或AI代理label行数虚构计费用量；本评价没有发起外部LLM调用。", "",
              "## 9. 验证与保留的失败", "",
              "- 小fixture自测：初轮12pass/10fail（本脚本合成fixture错误使用permanence=stable，schema只接受persistent/temporary/unknown）；改本fixture为合法persistent，22pass。原schema/闸/模型/参考未改，初始失败log保留。",
              "- 初轮PowerShell检查wrapper把Write-Output状态字符串混进bool返回，导致自测非零后仍继续专项；已改Write-Host并返回严格bool。真实exit_code=1和全部日志保留，没有把初轮脚本exit0当通过。",
              "- 预测完成后的新报告生成器两次SyntaxError（列表括号缺失）已只改该新脚本修正；失败记录report-generation-initial-failure.json保留，报告成功生成，未重跑任何模型预测。",
              "- 研究专项test_adapter/test_metrics/test_integration_v3/test_audit_flow：131pass，107个既有joblib/NumPy deprecation warning，无skip/xfail；相关scope验证不重复full。"
              "- 真CLI `python -B -m research.memory_relations.evaluate --model-dir M --data old --out unavailable-policy-cli-output.json`（无diagnostic）真实exit2，拒绝null/false policy，零输出工件；精确argv/stderr保存policy-cli-rejection.json。不是模型坏或正式run失败。",
              "- 原request/registry/root/count/finite/cache绑定/输出覆盖/源码前后hash只读核验通过；汇总与报告生成新增模型预测0。", "",
              "## 10. 失败解释、局限与未运行项", "",
              "最多10个错例存在bounded-case-explanations.json（按注册顺序，不是随机样本，不用来调模型）。已读的小范围显示duplicate↔contextual、complement↔duplicate/contextual与conflict→contextual边界混淆；这些是与proxy参考的不一致，不宣称真人裁决。完整case概率与闸后cause都已持久化，但不向LLM倾倒全部题目。",
              "未进行：full质量门重复、再audit、调参/重训/重选/阈值挑选、新人工gold/new-unseen set评价、生产接线/上线、commit/push、安装/下载、用户会话操作。没有riskcertificate，不能把old exposed set或成功引用机械闸当生产安全结论。",
              "模型从此固定；成绩差也保全，任何改模型只能开另一实验，不能把同872参考再称fresh。",
              "报告主JSON：terminal-result.json（全部分视图/区间）、verified-summary.json（核验+补全根引用分母）、old-exposed-set-result.json；逐case见heldout/ood/unlabelled/old-exposed-set.predictions.jsonl；真实概率terminal-probabilities.jsonl与cache-entries.jsonl；资源、CLI、失败证据均同目录。", ""]
    report = P / "REPORT.md"
    if len("\n".join(lines).encode("utf-8")) > 20000:
        sections = [(index, line[3:]) for index, line in enumerate(lines) if line.startswith("## ")]
        navigation = ["## 阅读导航", "", "按下表按需阅读，详表和每case工件以本次JSON真源核对。", "",
                      "| 节 | 起始行 | 何时读 |", "| --- | --- | --- |"]
        inserted = len(navigation) + len(sections) + 1
        for index, title in sections:
            navigation.append(f"| {title} | REPORT.md:{index + 1 + inserted} | 核对该项结论或证据时 |")
        navigation.append("")
        lines[2:2] = navigation
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"report": str(report), "terminal_failures": failure_meta,
                      "old_cross_batch_uncertain_over_conflict": cross_uncertain,
                      "old_oracle_mismatches": oracle_mismatches,
                      "reference_views": {split: {"classifier": summary["terminal"][split]["views"]["classifier"]["root_agreement_ci"],
                                                  "macro_f1": summary["terminal"][split]["views"]["classifier"]["root_macro_f1_ci"]} for split in ("heldout", "ood")},
                      "old_set_root": {k: old["gated_views"][k] for k in ("root_agreement", "root_macro_f1", "row_macro_f1")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
