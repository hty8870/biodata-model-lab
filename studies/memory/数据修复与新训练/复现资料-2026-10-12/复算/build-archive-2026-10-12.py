                       
"""复现资料归档构建（2026-10-12，标准库-only）。

用途：按显式白名单从 ROOT（交付代码仓）与其 .git/validation-worktrees 复制冻结数据、
终评参考、冻结协议、研究源码快照与已生成概率/统计缓存到本归档目录，逐件记录
source SHA256 与 archive SHA256；不覆盖任何已存在同名文件；不读 .env/私钥/用户文件。

纪律：
- 只复制白名单；weights / QA 运行时 / 大日志 / sealed-private / calibration+audit 分区不复制；
- 复制前对源文件做高精度秘密标记扫描，命中即停止该件并记录（不擅自脱敏冒充原件）；
- 共享 dev/prob/policy 三臂同 SHA，只存 1 份并写映射；
- 研究树为 untracked，版本绑定以逐件 SHA 为准（git HEAD 仅作上下文）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"<archive-root>")
ARCHIVE = Path(__file__).resolve().parents[1]

WA = ".git/validation-worktrees/mr3-data-20261011/research/memory_relations/runs/v3-adjudication-20261012/frozen"
WT = ".git/validation-worktrees/mr3-terminal-20261012/research/memory_relations/runs/terminal-reference/freeze/20261012-093519"
EV = "research/memory_relations/runs/v3-final-evaluation-20261012"
AB = "research/memory_relations/runs/v3-data-ablation-20261012"
GK = "research/memory_relations/runs/v3-governance-fix-20261012"
RV = "research/memory_relations/runs/v3-final-research-validation-20261012"

SECRET_HARD_MARKERS = [
    b"BEGIN RSA PRIVATE KEY", b"BEGIN OPENSSH PRIVATE KEY", b"BEGIN EC PRIVATE KEY",
    b"kimi.pem", b"id_rsa",
]
                                               
SECRET_SOFT_MARKERS = [
    b"BEGIN PRIVATE KEY", b"AKIA", b"ghp_", b"password", b"passwd",
    b"api_key", b"apikey", b"secret_key", b"sk-zen",
]
SECRET_REVIEW = {
    "research/memory_relations/schema.py": {
        "markers": ["api_key"], "verdict": "benign",
        "evidence": "LOCAL_PROVIDER 配置默认参数 api_key=None（字段名，非值）"},
    "research/memory_relations/teachers.py": {
        "markers": ["password"], "verdict": "benign",
        "evidence": "urlsplit(base_url).password 属性安全检查（字段名）"},
    "research/memory_relations/test_adapter.py": {
        "markers": ["api_key"], "verdict": "benign",
        "evidence": "测试夹具合成键 api_key=\"test-memory-relations-key\"/\"synthetic-knowledge-key\""},
    "research/memory_relations/test_teachers.py": {
        "markers": ["api_key", "sk-zen"], "verdict": "benign",
        "evidence": "离线测试夹具 KEY=\"sk-zen-test-SECRET-4f21\"（自身命名含 test/SECRET，仅用于崩溃转发回归；无网络 fixture 拒绝真实连接）"},
    "research/memory_relations/test_training.py": {
        "markers": ["api_key"], "verdict": "benign", "evidence": "测试夹具 api_key=None（字段名）"},
    "src/dataset_recommender/agent/memory_relations.py": {
        "markers": ["password", "api_key"], "verdict": "benign",
        "evidence": "秘密检测正则（password/secret 词表）与 cfg.api_key 类型检查（字段名）"},
    "research/memory_relations/runs/v3-final-evaluation-20261012/preflight_check.py": {
        "markers": ["api_key"], "verdict": "benign", "evidence": "cfg.api_key 字段读取（正式评估前置检查）"},
    "research/memory_relations/runs/v3-final-evaluation-20261012/test_formal_evaluation.py": {
        "markers": ["api_key"], "verdict": "benign", "evidence": "cfg.api_key 字段读取（批次切分测试）"},
    "research/memory_relations/runs/v3-final-research-validation-20261012/research-tests-final.xml": {
        "markers": ["BEGIN PRIVATE KEY", "AKIA", "ghp_", "password", "api_key"], "verdict": "benign",
        "evidence": "pytest 用例 ID 参数，显式为合成夹具（synthetic-pem-material / EEEEEEEEEEEEEEEE / ghp_-AAAA… / synthetic-json-password / api_key-None|replace_me|8）"},
    "research/memory_relations/runs/v3-final-research-validation-20261012/research-tests.xml": {
        "markers": ["BEGIN PRIVATE KEY", "AKIA", "ghp_", "password", "api_key"], "verdict": "benign",
        "evidence": "同上（初轮 XML，同类合成夹具用例 ID）"},
}

                                                                
WHITELIST = [
                                                    
    (f"{WA}/legacy-original/train.jsonl", "数据-冻结三臂/legacy-original.train.jsonl",
     "旧数据原标签臂训练集（2522 行/16 根）", "memory-relations-source-v3 行契约（9 字段）"),
    (f"{WA}/legacy-relabelled/train.jsonl", "数据-冻结三臂/legacy-relabelled.train.jsonl",
     "旧数据显式 relabel 臂训练集（2522 行/16 根；与 legacy-original 同文本/同成员/同顺序，96 label + 51 evidence 差）", "同上"),
    (f"{WA}/new-adjudicated/train.jsonl", "数据-冻结三臂/new-adjudicated.train.jsonl",
     "新裁决臂训练集（3290 行/235 根；被选入正式锁）", "同上"),
    (f"{WA}/new-adjudicated/dev.jsonl", "数据-冻结三臂/shared.dev.jsonl",
     "三臂共享 dev（405 行/40 根；best-dev 选点用）", "同上"),
    (f"{WA}/new-adjudicated/prob.jsonl", "数据-冻结三臂/shared.prob.jsonl",
     "三臂共享 prob 校准分区（116 行/12 组；温度拟合用）", "同上"),
    (f"{WA}/new-adjudicated/policy.jsonl", "数据-冻结三臂/shared.policy.jsonl",
     "三臂共享 policy 校准分区（120 行/13 组；候选阈值选点用）", "同上"),
    (f"{WA}/legacy-original/manifest.json", "数据-冻结三臂/legacy-original.frozen-manifest.json",
     "legacy-original 冻结清单（分区计数/类别分布）", "memory-relations frozen arm manifest"),
    (f"{WA}/legacy-relabelled/manifest.json", "数据-冻结三臂/legacy-relabelled.frozen-manifest.json",
     "legacy-relabelled 冻结清单", "同上"),
    (f"{WA}/new-adjudicated/manifest.json", "数据-冻结三臂/new-adjudicated.frozen-manifest.json",
     "new-adjudicated 冻结清单", "同上"),
                      
    (f"{WT}/expected-cases.jsonl", "数据-终评参考/expected-cases-872.jsonl",
     "872 个原请求注册表（id/packet/split/case_sha256；无 truth 字段）", "terminal-reference expected cases"),
    (f"{WT}/final-heldout.jsonl", "数据-终评参考/final-heldout.jsonl",
     "heldout 终评参考（444 行；接受 uncertain 98 行含 heldout 47）", "memory-relations source 行契约 + label/label_origin"),
    (f"{WT}/final-ood.jsonl", "数据-终评参考/final-ood.jsonl",
     "OOD 终评参考（427 行；含 uncertain 51）", "同上"),
    (f"{WT}/reference_status.jsonl", "数据-终评参考/reference_status.jsonl",
     "参考状态 sidecar（仅 excluded/unresolved；1 行 excluded，无 truth、无补标）", "terminal-reference status sidecar"),
    (f"{WT}/manifest.json", "数据-终评参考/freeze-manifest.json",
     "终评参考冻结清单（872/871/98/1 计数与逐件 SHA）", "memory-relations-terminal-reference-freeze-v1"),
    (f"{WT}/ref-ready-manifest.json", "数据-终评参考/ref-ready-manifest.json",
     "参考就绪清单（含学生完成度 not-run 说明与 sidecar 声明）", "同上"),
             
    ("research/memory_relations/runs/v3-coordination/source-protocol-frozen.json", "协议-冻结/source-protocol-frozen.json",
     "写作前冻结的源协议真源（400 根预分配/作者政策/标签手册绑定）", "memory-relations-source-v3（targets-not-generated 时点）"),
    ("research/memory_relations/data/label_manual_v3.md", "协议-冻结/label_manual_v3.md",
     "七类标签手册（写作与裁决的标签规范）", "label manual v3"),
    ("research/memory_relations/data/source_protocol_v3.json", "协议-冻结/source_protocol_v3.json",
     "数据目录内协议副本（与 runs 侧冻结件对照）", "memory-relations-source-v3"),
    ("research/memory_relations/data/teacher_protocol_v3.json", "协议-冻结/teacher_protocol_v3.json",
     "教师回执协议（DS/primary 审查口径）", "teacher protocol v3"),
    (f"{EV}/source-binding.json", "协议-冻结/final-evaluation-source-binding.json",
     "终评锁定绑定（模型/参考/源码/sealed 逐件 SHA）", "locked-formal-evaluation-binding-v1"),
    (f"{EV}/formal-evaluation-attempt.json", "协议-冻结/formal-evaluation-attempt.json",
     "一次性正式终评 attempt 标记（0 次重复预测的纪律凭证）", "formal evaluation attempt marker"),
    (f"{AB}/legacy-original/planned_manifest.json", "协议-冻结/training-input-contract-legacy-original.json",
     "legacy-original fit 前输入契约（train/dev/prob/policy 分区 SHA）", "memory-relations-finetune-v1-planned"),
    (f"{AB}/legacy-relabelled/planned_manifest.json", "协议-冻结/training-input-contract-legacy-relabelled.json",
     "legacy-relabelled fit 前输入契约", "同上"),
    (f"{AB}/new-adjudicated/planned_manifest.json", "协议-冻结/training-input-contract-new-adjudicated.json",
     "new-adjudicated fit 前输入契约（含被选 run 预算与采样规则）", "同上"),
    (f"{WA}/exclusions.jsonl", "协议-冻结/frozen-exclusions.jsonl",
     "冻结期排除台账（audit/未决/cal 隔离 214 行口径）", "frozen exclusions ledger"),
    (f"{WA}/legacy-change-provenance.jsonl", "协议-冻结/legacy-change-provenance.jsonl",
     "两旧臂 96 label + 51 evidence 差异的逐行来源台账", "legacy change provenance"),
                                                
    ("research/memory_relations/adapter.py", "源码快照-research-memory-relations/adapter.py", "研究适配层（evaluate/assess 接缝）", "python source"),
    ("research/memory_relations/build_data.py", "源码快照-research-memory-relations/build_data.py", "数据构建/装配入口", "python source"),
    ("research/memory_relations/evaluate.py", "源码快照-research-memory-relations/evaluate.py", "集合/行评价实现", "python source"),
    ("research/memory_relations/features.py", "源码快照-research-memory-relations/features.py", "特征/输入配方（joint input v1）", "python source"),
    ("research/memory_relations/finetune.py", "源码快照-research-memory-relations/finetune.py", "微调训练器（400 updates 预算）", "python source"),
    ("research/memory_relations/metrics.py", "源码快照-research-memory-relations/metrics.py", "校准/审计/统计口径实现（point_feasible/368 下界所在）", "python source"),
    ("research/memory_relations/schema.py", "源码快照-research-memory-relations/schema.py", "行契约与规范化", "python source"),
    ("research/memory_relations/teachers.py", "源码快照-research-memory-relations/teachers.py", "教师回执采集协议", "python source"),
    ("research/memory_relations/train.py", "源码快照-research-memory-relations/train.py", "分组采样/训练/锁定入口", "python source"),
    ("research/memory_relations/test_adapter.py", "源码快照-research-memory-relations/test_adapter.py", "研究测试（adapter）", "python test"),
    ("research/memory_relations/test_audit_flow.py", "源码快照-research-memory-relations/test_audit_flow.py", "研究测试（审计流）", "python test"),
    ("research/memory_relations/test_challenge.py", "源码快照-research-memory-relations/test_challenge.py", "研究测试（challenge）", "python test"),
    ("research/memory_relations/test_data.py", "源码快照-research-memory-relations/test_data.py", "研究测试（数据）", "python test"),
    ("research/memory_relations/test_finetune.py", "源码快照-research-memory-relations/test_finetune.py", "研究测试（微调）", "python test"),
    ("research/memory_relations/test_integration_v3.py", "源码快照-research-memory-relations/test_integration_v3.py", "研究测试（v3 集成）", "python test"),
    ("research/memory_relations/test_metrics.py", "源码快照-research-memory-relations/test_metrics.py", "研究测试（统计口径）", "python test"),
    ("research/memory_relations/test_source_v3.py", "源码快照-research-memory-relations/test_source_v3.py", "研究测试（源契约）", "python test"),
    ("research/memory_relations/test_teachers.py", "源码快照-research-memory-relations/test_teachers.py", "研究测试（教师）", "python test"),
    ("research/memory_relations/test_training.py", "源码快照-research-memory-relations/test_training.py", "研究测试（训练）", "python test"),
    ("research/memory_relations/spec.json", "源码快照-research-memory-relations/spec.json", "实验规格（类别/预算/接缝声明）", "json spec"),
    ("research/memory_relations/README.md", "源码快照-research-memory-relations/README.md", "研究模块 README", "markdown"),
    ("src/dataset_recommender/agent/memory_relations.py", "源码快照-生产侧/memory_relations.py",
     "生产侧标签来源 prompt（source_of_labels 所指）", "python source（tracked 交付仓）"),
    (f"{EV}/binding.py", "源码快照-评估管线/binding.py", "终评绑定/哈希校验脚本", "python source"),
    (f"{EV}/formal_evaluation.py", "源码快照-评估管线/formal_evaluation.py", "一次性正式终评驱动", "python source"),
    (f"{EV}/preflight_check.py", "源码快照-评估管线/preflight_check.py", "终评前置检查", "python source"),
    (f"{EV}/run_once.ps1", "源码快照-评估管线/run_once.ps1", "一次性运行包装", "powershell"),
    (f"{EV}/verify_and_summarize.py", "源码快照-评估管线/verify_and_summarize.py", "终评结果复算/汇总", "python source"),
    (f"{EV}/make_report.py", "源码快照-评估管线/make_report.py", "REPORT.md 生成器", "python source"),
    (f"{EV}/finish_check.py", "源码快照-评估管线/finish_check.py", "收尾核销检查", "python source"),
    (f"{EV}/test_formal_evaluation.py", "源码快照-评估管线/test_formal_evaluation.py", "终评驱动测试", "python test"),
                   
    (f"{EV}/reports/terminal-probabilities.jsonl", "缓存-概率与统计/terminal-probabilities.jsonl",
     "872 个原请求的模型概率（classifier/pair 关系 + 七类概率向量；含 reference 列以便离线复算，均为已曝光合成代理标签）", "terminal probabilities（每行 id/split/probabilities/reference_label 等）"),
    (f"{EV}/reports/cache-entries.jsonl", "缓存-概率与统计/cache-entries.jsonl",
     "底层物理预测缓存（2318 条 model_identity/cache_key/概率向量）", "model cache entries"),
    (f"{EV}/reports/REPORT.md", "缓存-概率与统计/terminal-eval-REPORT.md",
     "终评正式报告（数字、口径与失败保全）", "markdown report"),
    (f"{EV}/reports/verified-summary.json", "缓存-概率与统计/terminal-eval-verified-summary.json",
     "终评复算汇总（根/行 F1、CI、风险侧、旧 set 诊断）", "verified summary json"),
    (f"{EV}/reports/compact-summary.json", "缓存-概率与统计/terminal-eval-compact-summary.json",
     "终评紧凑摘要", "compact summary json"),
    (f"{EV}/reports/output-hashes.json", "缓存-概率与统计/terminal-eval-output-hashes.json",
     "终评产物哈希清单", "output hashes json"),
    (f"{EV}/reports/final-validation.json", "缓存-概率与统计/terminal-eval-final-validation.json",
     "终评最终校验回执", "final validation json"),
    (f"{EV}/reports/old-exposed-set-result.json", "缓存-概率与统计/old-exposed-set-result.json",
     "旧曝光 set 回放诊断（23/37；related P 24/861=2.79%；仅诊断非未见）", "old exposed set result json"),
    (f"{AB}/training-verification-final.json", "缓存-概率与统计/training-verification-final.json",
     "九次正式 CUDA run 与一次 dev 锁定的核验汇总（含 PID 绑定修复证据）", "training verification json"),
    ("research/memory_relations/runs/v3-root-curves-20261012/curve-verification-final.json", "缓存-概率与统计/curve-verification-final.json",
     "根规模曲线核验（60/120 新训 6 run；235 复用主臂）", "curve verification json"),
    ("research/memory_relations/runs/v3-single-audit-20261012/audit.json", "缓存-概率与统计/single-audit.json",
     "唯一锁一次审计结果（no_candidate；threshold/界全 null）", "audit json"),
    ("research/memory_relations/runs/v3-study-lock-20261012/lock.json", "缓存-概率与统计/study-lock.json",
     "研究锁（候选温度/point_feasible=false；audit deferred）", "memory-relations-audit-lock-v1"),
    ("research/memory_relations/runs/v3-study-lock-20261012/commitment.json", "缓存-概率与统计/study-commitment.json",
     "锁定承诺（模型/数据绑定）", "commitment json"),
    ("research/memory_relations/runs/v3-study-lock-20261012/audit-attempt.json", "缓存-概率与统计/study-audit-attempt.json",
     "审计 attempt 记录（一次且不重试）", "audit attempt json"),
    (f"{AB}/nine-run-gpu-records.json", "缓存-概率与统计/nine-run-gpu-records.json",
     "九 run GPU 记录聚合", "nine run gpu records json"),
    (f"{AB}/final-study-result.json", "缓存-概率与统计/final-study-result.json",
     "研究完成回执（选中 new-adjudicated seed13）", "final study result json"),
    (f"{AB}/dev-only-selection.json", "缓存-概率与统计/dev-only-selection.json",
     "dev-only 选臂记录（中位数最近 seed 规则）", "selection json"),
    (f"{AB}/process-identity-legacy-original.json", "缓存-概率与统计/process-identity-legacy-original.json",
     "首轮 PID 误拦修复的存活 CIM 回执（真实解释器 PID 31188）", "process identity json"),
    (f"{AB}/new-adjudicated/scorers/finetuned-joint-size3290-seed13/meta.json", "缓存-概率与统计/selected-scorer-meta.json",
     "被选 scorer 元数据（temperature/threshold=null/weights_sha256）", "memory-relations-finetune-scorer-v1"),
    (f"{AB}/new-adjudicated/training/finetuned-joint-size3290-seed13.jsonl", "缓存-概率与统计/selected-training-log.jsonl",
     "被选 run 的逐更新训练/评估日志", "finetune training jsonl"),
                
    ("artifacts/quality-memory-relations-final-research-strong-scope-20261012.json", "验收-门与修复-2026-10-12/quality-full-passed-strong-scope.json",
     "最终有效 full 门 artifact（14/14 passed；6670+21skip；governance 2.844s）", "quality gate report json"),
    ("artifacts/quality-memory-relations-final-research-20261012.json", "验收-门与修复-2026-10-12/quality-full-failed-original.json",
     "原 full artifact（failed：agent-governance 855.063s 超 600s 预算，stdout 含 PASSED）", "quality gate report json"),
    (f"{GK}/full-attempt-1-failed.json", "验收-门与修复-2026-10-12/quality-full-failed-fixed.json",
     "修复后 attempt1（failed：public-mirror-source 'mirror source is not tracked'）", "quality gate report json"),
    (f"{GK}/full-strong-attempt1-failed.json", "验收-门与修复-2026-10-12/quality-full-failed-strong-attempt1.json",
     "strong attempt1（failed：'public source references private material'）", "quality gate report json"),
    (f"{GK}/terminal-summary-validated.json", "验收-门与修复-2026-10-12/terminal-summary-validated.json",
     "治理修复最终核验汇总（698 未重跑、1207 保护件、12 critical 不变、0 模型调用）", "terminal summary json"),
    (f"{RV}/research-tests-final.xml", "验收-门与修复-2026-10-12/research-tests-final.xml",
     "研究 QA 终版（698 passed/0 failed/0 skipped）", "junit xml"),
    (f"{RV}/research-tests.xml", "验收-门与修复-2026-10-12/research-tests-initial.xml",
     "研究 QA 初轮（5 failed；wrapper CUDA mask 事故，保留失败）", "junit xml"),
    (f"{RV}/environment-diagnosis.json", "验收-门与修复-2026-10-12/environment-diagnosis.json",
     "初轮失败环境诊断（设备可见性）", "diagnosis json"),
    (f"{RV}/governance-failure-diagnosis.json", "验收-门与修复-2026-10-12/governance-failure-diagnosis.json",
     "治理脚本超时诊断（595 reparse 条目、0 跟随）", "diagnosis json"),
    (f"{RV}/readiness-summary-final.json", "验收-门与修复-2026-10-12/readiness-summary-final.json",
     "研究 QA 就绪摘要", "readiness json"),
    (f"{RV}/readiness-closeout.json", "验收-门与修复-2026-10-12/readiness-closeout.json",
     "就绪收口（BLOCKED_NOT_RELEASE_READY；修复前时点）", "closeout json"),
]

NOT_COPIED = [
    (f"{AB}/new-adjudicated/scorers/finetuned-joint-size3290-seed13/model.pt",
     "被选权重 449MB（sha256 3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498）：任何需要模型前向的复现实验都必须取回该文件；仅凭归档中的概率缓存不能重做新推理，只能做可计算复算"),
    ("research/memory_relations/runs/v3-final-evaluation-20261012/reports/heldout.predictions.jsonl", "逐请求原始预测输出；终评概率缓存已覆盖 872 请求口径"),
    ("research/memory_relations/runs/v3-final-evaluation-20261012/reports/ood.predictions.jsonl", "同上"),
    ("research/memory_relations/runs/v3-final-evaluation-20261012/reports/old-exposed-set.predictions.jsonl", "旧 set 原始预测；聚合诊断已归档"),
    ("research/memory_relations/runs/v3-final-evaluation-20261012/reports/physical-prediction-calls.jsonl", "调用级日志；聚合计数已归档"),
    (f"{AB}/legacy-original/metrics.json", "逐更新训练指标（约 593KB）；曲线/训练核验聚合已归档"),
    (f"{AB}/legacy-relabelled/metrics.json", "同上"),
    (f"{AB}/new-adjudicated/metrics.json", "同上"),
    (f"{AB}/legacy-original/curve.jsonl", "逐点曲线原始件；curve-verification-final.json 已聚合"),
    (f"{AB}/legacy-relabelled/curve.jsonl", "同上"),
    (f"{AB}/new-adjudicated/curve.jsonl", "同上"),
    (f"{AB}/legacy-original/predictions.jsonl", "dev 预测原始件；终评概率缓存已覆盖终评口径"),
    (f"{AB}/legacy-relabelled/predictions.jsonl", "同上"),
    (f"{AB}/new-adjudicated/predictions.jsonl", "同上"),
    (f"{AB}/legacy-original.gpu-telemetry.jsonl", "GPU 遥测原始流；nine-run-gpu-records.json 已聚合"),
    (f"{AB}/legacy-relabelled.gpu-telemetry.jsonl", "同上"),
    (f"{AB}/new-adjudicated.gpu-telemetry.jsonl", "同上"),
    (f"{WA}/provenance-row-ledger.jsonl", "全量行级 provenance（18MB）；差异台账（exclusions/legacy-change）已归档"),
    (f"{WA}/calibration-preallocation-review.jsonl", "cal 预分配复核台账；cal/audit 分区按纪律不进入归档"),
    ("research/memory_relations/runs/v3-final-research-validation-20261012/validation.log", "QA 运行日志（71KB）；关键摘要与 XML 已归档"),
]

EXCLUDED_BY_DESIGN = [
    ("frozen/.../calibration.jsonl 与 audit 相关分区", "三臂共享 cal 236 行/25 根与唯一审计数据按根隔离纪律不进入归档（白名单仅 dev/prob/policy）"),
    (".git/validation-worktrees/mr3-data-20261011/.../v3-blind-import/sealed-private 与 v3-blind-prepare/sealed-private", "写作侧私有台账/映射（不得外带）"),
    (".git/validation-worktrees/*/QA 目录、pytest scratch、qa-runtime*", "QA 运行时目录（含 reparse 链接），不递归复制"),
    ("research/memory_relations/runs/**/*.log、events、telemetry 原始大日志", "大日志不进入归档；聚合统计已覆盖报告引用"),
    ("models/embeddings/paraphrase-multilingual-MiniLM-L12-v2（骨干与 tokenizer）", "base weights 不复制：重训/重推必须另行获取该骨干（fingerprint 1416f5fa5282f9b69713c63450fccdef）；归档不是完全独立便携重训包"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def secret_scan(path: Path, src_rel: str) -> list:
    """硬标记 fail-closed；软标记必须在复核表内（benign）否则 fail-closed。"""
    hits = []
    data = path.read_bytes()
    for marker in SECRET_HARD_MARKERS:
        if marker in data:
            hits.append({"marker": marker.decode("ascii"), "severity": "hard"})
    soft_hits = []
    for marker in SECRET_SOFT_MARKERS:
        if marker in data:
            soft_hits.append(marker.decode("ascii"))
    if soft_hits:
        review = SECRET_REVIEW.get(src_rel)
        covered = sorted(set(soft_hits) - set(review["markers"])) if review else soft_hits
        if review and review.get("verdict") == "benign" and not covered:
            hits.append({"marker": ",".join(soft_hits), "severity": "soft-reviewed-benign",
                         "evidence": review["evidence"]})
        else:
            hits.append({"marker": ",".join(soft_hits), "severity": "soft-unreviewed", "covered": covered})
    return hits


def read_git_head() -> str:
    head = (ROOT / ".git" / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref = head[5:].strip()
        p = ROOT / ".git" / ref
        if p.is_file():
            return p.read_text(encoding="ascii").strip()
        packed = ROOT / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ")[0]
    return head


def main() -> int:
    created = datetime.now(timezone.utc).isoformat()
    entries = []
    blocked = []
    total_bytes = 0
    scan_review_log = []
    for src_rel, arc_rel, purpose, schema in WHITELIST:
        src = ROOT / src_rel.replace("/", os.sep)
        dst = ARCHIVE / arc_rel.replace("/", os.sep)
        if not src.is_file():
            blocked.append({"source_repo_relative": src_rel, "reason": "missing_source"})
            print(f"[MISS] {src_rel}")
            continue
        hits = secret_scan(src, src_rel)
        hard = [h for h in hits if h["severity"] in ("hard", "soft-unreviewed")]
        if hard:
            blocked.append({"source_repo_relative": src_rel, "reason": "secret_marker_scan_hit", "markers": hits})
            print(f"[SECRET-STOP] {src_rel} markers={hard}")
            continue
        reviewed = [h for h in hits if h["severity"] == "soft-reviewed-benign"]
        if reviewed:
            scan_review_log.append({"source_repo_relative": src_rel, "markers": reviewed[0]["marker"],
                                    "verdict": "benign", "evidence": reviewed[0]["evidence"]})
        src_sha = sha256_of(src)
        adopted = False
        if dst.exists():
            dst_sha = sha256_of(dst)
            if dst_sha == src_sha:
                adopted = True                       
            else:
                blocked.append({"source_repo_relative": src_rel, "archive_path": arc_rel, "reason": "target_exists_different_no_overwrite"})
                print(f"[EXISTS-DIFF] {arc_rel}")
                continue
        if not adopted:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            dst_sha = sha256_of(dst)
            if dst_sha != src_sha:
                blocked.append({"source_repo_relative": src_rel, "reason": "copy_sha_mismatch"})
                print(f"[SHA-FAIL] {src_rel}")
                continue
        size = dst.stat().st_size
        total_bytes += size
        row = {
            "archive_path": arc_rel,
            "source_repo_relative": src_rel,
            "sha256": dst_sha,
            "bytes": size,
            "purpose": purpose,
            "schema": schema,
            "fresh": False,
        }
        if arc_rel.endswith(".jsonl"):
            row["rows"] = count_rows(dst)
        entries.append(row)
        print(f"[{'ADOPT' if adopted else 'OK'}] {arc_rel} {dst_sha[:12]} {size}")

                                                                
    shared_parts = ["dev", "prob", "policy"]
    shared_check = {}
    for part in shared_parts:
        shas = {}
        for arm in ["legacy-original", "legacy-relabelled", "new-adjudicated"]:
            p = ROOT / f"{WA}/{arm}/{part}.jsonl".replace("/", os.sep)
            shas[arm] = sha256_of(p)
        same = len(set(shas.values())) == 1
        shared_check[part] = {"same": same, "sha256": shas["new-adjudicated"] if same else None, "by_arm": shas}
        assert same, f"shared partition {part} differs across arms"
    print("[OK] cross-arm shared equality verified for dev/prob/policy")

                                            
    def iter_rows(rel):
        with open(ROOT / rel.replace("/", os.sep), "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
    label_diff = evidence_diff = text_diff = 0
    n = 0
    for a, b in zip(iter_rows(f"{WA}/legacy-original/train.jsonl"), iter_rows(f"{WA}/legacy-relabelled/train.jsonl")):
        n += 1
        if a["request"] != b["request"]:
            text_diff += 1
        if a["label"] != b["label"]:
            label_diff += 1
        if a["evidence_refs"] != b["evidence_refs"] or a["related_ids"] != b["related_ids"]:
            evidence_diff += 1
    assert (n, text_diff, label_diff, evidence_diff) == (2522, 0, 96, 51), (n, text_diff, label_diff, evidence_diff)
    print(f"[OK] legacy pair diff = text 0 / label 96 / evidence 51 over {n} rows")

          
    shared_mapping = {
        "shared.dev.jsonl": {"used_by_arms": ["legacy-original", "legacy-relabelled", "new-adjudicated"], "identical_sha_across_arms": shared_check["dev"]["same"], "arms_not_duplicated": True},
        "shared.prob.jsonl": {"used_by_arms": ["legacy-original", "legacy-relabelled", "new-adjudicated"], "identical_sha_across_arms": shared_check["prob"]["same"], "arms_not_duplicated": True},
        "shared.policy.jsonl": {"used_by_arms": ["legacy-original", "legacy-relabelled", "new-adjudicated"], "identical_sha_across_arms": shared_check["policy"]["same"], "arms_not_duplicated": True},
    }
                                   
    by_arc = {e["archive_path"]: e for e in entries}
    for name in list(shared_mapping):
        arc = f"数据-冻结三臂/{name}"
        if arc in by_arc:
            shared_mapping[name]["archived_sha256"] = by_arc[arc]["sha256"]

    not_copied = []
    for rel, reason in NOT_COPIED:
        if "已归档" in reason and rel.startswith("research"):
            continue
        p = ROOT / rel.replace("/", os.sep)
        item = {"source_repo_relative": rel, "reason": reason}
        if p.is_file():
            item["sha256"] = sha256_of(p)
            item["bytes"] = p.stat().st_size
        not_copied.append(item)

    manifest = {
        "schema": "memory-relations-repro-archive-manifest-v1",
        "created_at": created,
        "archive_root_relative": "复现资料-2026-10-12（位于 数据修复与新训练/ 内）",
        "root_git_head_context": read_git_head(),
        "root_git_head_note": "研究树 research/ 为 untracked：git HEAD 仅作上下文，绑定以逐件 SHA256 为准（d64 一个 commit 不足以绑定研究源码）",
        "freshness": {
            "status": "already-exposed-synthetic-material",
            "human_gold": False,
            "source": "RA 自造合成数据 + AI 代理标签",
            "statement": "全部内容已在本轮研究与报告中曝光过，不得再称为 fresh/未见；不包含真实用户资料",
        },
        "files": entries,
        "shared_partition_mapping": shared_mapping,
        "not_copied": not_copied,
        "excluded_by_design": [{"item": a, "reason": b} for a, b in EXCLUDED_BY_DESIGN],
        "blocked_items": blocked,
        "secret_scan_review": {
            "hard_markers": [m.decode() for m in SECRET_HARD_MARKERS],
            "note": "硬标记 fail-closed；软标记命中需逐件上下文复核为合成夹具/字段名/正则",
            "reviewed_benign": scan_review_log,
        },
        "model_requiring_vs_computable_note": (
            "可计算复算（需归档内文件即可）：分区计数/行数、两旧臂 96+51 差异、"
            "872 参考覆盖与 667/871、667/872 口径、风险分子分母、旧 set 23/37 与 24/861 等；"
            "需要模型的复现实验（必须另取 model.pt 449MB 与 backbone）：任何新前向/重训/换阈值评估、"
            "重算物理预测缓存。"
        ),
    }
    out = ARCHIVE / "manifest-复现档案-2026-10-12.json"
    assert not out.exists(), "manifest already exists; refusing to overwrite"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[DONE] files={len(entries)} bytes={total_bytes} blocked={len(blocked)} manifest={out.name}")
    return 0 if not blocked else 1


if __name__ == "__main__":
    sys.exit(main())
