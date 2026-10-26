"""NLI 三类参考臂（research-only）：同骨干同配方的目标匹配参考工具，独立于七类工件。

定位与边界（对齐 study-v4.json 的 native_nli 视图 arm_spec：「同骨干同配方 3 类 NLI 参考臂；
只组装既有原语」）：

- 输入仍是共同 memory-shaped request，唯一序列化复用 finetune.joint_text（不新增第二份配方）；
  主向槽位 existing=[premise]、candidate=hypothesis（交换向由 harness 另列），方向与槽位映射
  写进工件 meta 供消费者对齐；本模块不推断方向，也不替代 harness 的方向编排。
- 只组装既有原语：finetune 的 joint_text / build_joint_model(num_labels=) /
  build_joint_optimizer / encode_batch / cpu_state / training_windows / logits_for /
  truncation_profile / resolve_device / resolve_bf16 / autocast_for / file_sha256 /
  validate_training_options，指标复用 metrics.macro_f1(n_classes=3) 与 metrics._group_weights
  （后者是项目既有的组等权唯一实现，evaluate.py 同样直接消费）；既不调用也不弱化
  train_model 的七类硬闸与七类 save_scorer/load_scorer 工件 schema。
- 本模块只是**参考工具**，不是改进成功的声明：NLI 三类与七类任务不等价，禁任何硬映射
  （neutral≠uncertain、entailment≠duplicate、non-entailment≠conflict），不得据此产出七类预测
  或风险/放行/认证表述；标准参考与原七类模型的训练数据不同，分数差异不能直接当因果结论。
- 标签空间固定 NLI_CLASSES=(entailment, neutral, contradiction)（对齐 study
  label_spaces.native.ocnli 的类序）；模型硬闸（恰 3 类头 + 头宽 == 实际骨干
  config.hidden_size）在 fit 前、scorer 初始化、预测输出、保存与加载处统一执行，非 3 类头、
  结构不一致或输出形状不符一律拒绝；模型被冻结骨干（含部分冻结）也一律拒绝训练/保存/导出。
- 工件按「实际对象」绑定而不是按声明目录：方向只从 scorer 派生（显式不符即拒），config 用实际
  配置内容摘要绑定（不靠 shape-only），tokenizer 把 scorer 实际对象序列化进工件自有 tokenizer/
  目录并与重建结果逐部件互验（vocab ID 映射、fast backend 全量序列化、padding/truncation/
  model_max_length 等运行配置、探针编码），加载只从该自有状态重建并复核文件指纹 + 语义状态摘要，
  不可重建对象写前拒绝、绝不静默换 tokenizer；权重用 sha256 + 键集合绑定；backbone_path 只是出处标签。
- 训练入口只接受 train 行（split 必填且为 train，带 sealed 标记一律拒绝），dev 行只用于
  best-dev 选择；模块不提供任何验收数据入口，独立验收数据不得进入 fit 或 checkpoint 选择。
- NLI 参考臂按声明是端到端微调：优化器必须覆盖全部可训练参数（含骨干），首个 backward 必须
  看到骨干梯度，不得把只更新 head 的运行称为 full。
- backbone 导出只写全新 own 路径：目标必须不存在、路径组件不得是符号链接/junction/reparse
  （按未解析路径逐级检查，含 .. 组件拒），源骨干按真实身份（含经 junction/alias 加载）解析并与
  其别名子树共同围栏（目标字面拼写与真实位置两侧都比对，不做目标解析优先）；导出后复核源目录
  指纹不变；manifest 记录实际 config 摘要与 tokenizer 绑定，导出物不含 NLI head。
- 只用本地权重（local_files_only=True），绝不下载模型；请求 GPU 不可用一律报错，绝不静默回退。

典型用法（预算与数据由父侧正式冻结后调用，本模块不自带 CLI 与任何默认实验）：

    rows = [{"split": "train", "label": "entailment", "request": {...}}, ...]
    model, _ = finetune.build_joint_model(embedding_dir, num_labels=len(NLI_CLASSES))
    training = train_nli(model, tokenizer, rows, seed=13, epochs=3, batch_size=8, grad_accum=2,
                         max_length=384, device=torch.device("cpu"), bf16=False)
    artifact = save_nli_artifact(NLIScorer(model=model, tokenizer=tokenizer, max_length=384),
                                 out_dir, backbone_dir=embedding_dir,
                                 train_meta={"budget": training["budget"], "selection": training["selection"]})
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import stat
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from dataset_recommender.agent.semantic_relations import NLI_CLASSES
from research.memory_relations import features as FEAT
from research.memory_relations import finetune as FT
from research.memory_relations import metrics as EM

NLI_LABEL_SPACE = "nli3"
NLI_SCHEMA = "memory-relations-nli-reference-scorer-v1"
NLI_KIND = "nli-reference-joint"
NLI_WEIGHTS_FILE = "nli_model.pt"
NLI_META_FILE = "nli_meta.json"
NLI_BACKBONE_EXPORT_SCHEMA = "memory-relations-nli-backbone-export-v1"
                                             
NLI_TOKENIZER_DIR = "tokenizer"
                                                         
                        
_TOKENIZER_PROBE_TEXTS = ("Hello world!", "  spaced\tand\nnew  ", "MIXED Case 123",
                          "一个男人正在外面走路。", "café ☕", "a,b.c/d;e:f", "")
                                                                                    
NLI_PRIMARY_DIRECTION = "hypothesis_as_candidate"
NLI_DIRECTIONS = ("hypothesis_as_candidate", "premise_as_candidate")
NLI_JOINT_ROLES = {
    "hypothesis_as_candidate": {"premise": "existing", "hypothesis": "candidate"},
    "premise_as_candidate": {"premise": "candidate", "hypothesis": "existing"},
}
                                                                                   
FIT_SPLITS = ("train",)
DEV_SPLITS = ("train", "dev", "calibration")


def label_indexes(labels) -> np.ndarray:
    """NLI 标签 → 0..2 索引（类序唯一真源 NLI_CLASSES）；越界值与七类名一律拒绝，不做硬映射。"""
    items = list(labels) if isinstance(labels, (list, tuple, np.ndarray)) else [labels]
    indexes: list[int] = []
    for item in items:
        if isinstance(item, (bool, np.bool_)) or item is None:
            raise ValueError(f"NLI 标签 {item!r} 无效：标签空间只有 {NLI_CLASSES}")
        if isinstance(item, str):
            if item not in NLI_CLASSES:
                raise ValueError(f"NLI 标签 {item!r} 不在 {NLI_CLASSES} 内：禁止七类硬映射")
            index = NLI_CLASSES.index(item)
        elif isinstance(item, (int, np.integer)):
            index = int(item)
            if not 0 <= index < len(NLI_CLASSES):
                raise ValueError(f"NLI 标签索引 {index} 越界（只有 0..{len(NLI_CLASSES) - 1}）")
        else:
            raise ValueError(f"NLI 标签 {item!r} 类型无效（只接受类名或 0..2 整数）")
        indexes.append(index)
    return np.asarray(indexes, dtype=np.int64)


def row_labels(rows: Sequence[dict]) -> np.ndarray:
    """行序列 → 0..2 索引（类序唯一真源 NLI_CLASSES）。"""
    return label_indexes([row.get("label") for row in rows])


def require_three_classes(labels) -> None:
    """NLI 训练标签存在性硬闸：三类缺一即拒绝训练（与七类硬闸同精神，但只在本模块生效）。"""
    present = set(label_indexes(labels).tolist())
    missing = [name for index, name in enumerate(NLI_CLASSES) if index not in present]
    if missing:
        raise ValueError(f"NLI 训练样本未覆盖全部三类，缺 {missing}：拒绝训练")


def _checked_rows(rows: Sequence[dict], *, stage: str) -> list[dict]:
    """train-only/封存守卫：split 必填且受控、sealed 标记一律拒绝、request 必须是对象（fail-closed）。"""
    if not rows:
        raise ValueError(f"{stage} 行不能为空")
    allowed = FIT_SPLITS if stage == "fit" else DEV_SPLITS
    checked = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{stage} 第 {index} 行必须是字典")
        if row.get("sealed"):
            raise ValueError(f"{stage} 第 {index} 行带 sealed 标记：封存/终测数据不得进入训练或选择")
        split = row.get("split")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"{stage} 第 {index} 行缺少 split：无法证明 train-only"
                             "（与 train.load_rows 的 split 口径一致，缺字段 fail-closed）")
        if split not in allowed:
            raise ValueError(f"{stage} 第 {index} 行 split={split!r} 不允许（只接受 {allowed}）："
                             "train-only/封存拒绝")
        if not isinstance(row.get("request"), dict):
            raise ValueError(f"{stage} 第 {index} 行缺少 request 对象")
        checked.append(row)
    return checked


def _row_groups(rows: Sequence[dict]) -> "list[str] | None":
    """dev 行分组口径（fail-loud）：全行带合法非空 group_id → root-equal；全行都不带该字段 →
    row_native；字段存在却为空/None/非字符串，或合法与缺失混用 → 拒绝，不静默降级权重口径。"""
    supplied = ["group_id" in row for row in rows]
    valid = [isinstance(row.get("group_id"), str) and bool(row["group_id"].strip()) for row in rows]
    if all(valid):
        return [row["group_id"] for row in rows]
    if any(supplied):
        bad = [row.get("group_id") for row, ok in zip(rows, valid) if not ok]
        raise ValueError(f"dev 行 group_id 必须全行带合法非空字符串或全行不带；收到畸形值 {bad[:3]!r}："
                         "拒绝把畸形分组静默当作 row_native")
    return None


def _check_direction(direction: str) -> dict:
    if direction not in NLI_DIRECTIONS:
        raise ValueError(f"NLI 方向只支持 {NLI_DIRECTIONS}，实际 {direction!r}")
    return dict(NLI_JOINT_ROLES[direction])


def _require_nli_model(model, *, where: str) -> int:
    """NLI 模型硬闸（训练/推理/工件共用）：恰 3 类头，且头宽 == 实际骨干 config.hidden_size。"""
    classifier = getattr(model, "classifier", None)
    out_features = getattr(classifier, "out_features", None)
    if not isinstance(out_features, int) or int(out_features) != len(NLI_CLASSES):
        raise ValueError(f"{where} 只接受 {len(NLI_CLASSES)} 类头（{NLI_CLASSES}），实际 "
                         f"{out_features!r}：默认七类/其它头模型必须走各自路径，"
                         "不得在 NLI 参考臂上截列或扩列")
    config = getattr(getattr(model, "backbone", None), "config", None)
    hidden = getattr(config, "hidden_size", None)
    in_features = getattr(classifier, "in_features", None)
    if not isinstance(hidden, int) or not isinstance(in_features, int) or int(in_features) != int(hidden):
        raise ValueError(f"{where} 头宽 {in_features!r} 与实际骨干 config.hidden_size {hidden!r} 不一致："
                         "拒绝结构不一致的模型")
    return int(hidden)


def _require_trainable_backbone(model, *, where: str) -> None:
    """NLI 参考臂按声明端到端微调：任一骨干参数被冻结（含部分冻结）即拒绝，禁止虚称已训练骨干。"""
    backbone = getattr(model, "backbone", None)
    names = list(backbone.named_parameters()) if backbone is not None else []
    if not names:
        raise ValueError(f"{where} 找不到骨干参数")
    frozen = [name for name, param in names if not param.requires_grad]
    if frozen:
        raise ValueError(f"{where} 检测到冻结骨干参数（{frozen[:3]}…）：NLI 参考臂按声明必须端到端"
                         "微调；如需只训 head 请另立 kind，且不得把它保存/导出为 NLI-trained 骨干")


def _config_digest(config) -> str:
    """实际 config 内容摘要（剔除 _name_or_path 出处标签；不按声明目录推断，不靠 shape-only）。"""
    payload = dict(config.to_dict())
    payload.pop("_name_or_path", None)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_summary(config) -> dict:
    summary = {}
    for key in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "max_position_embeddings", "vocab_size", "layer_norm_eps",
                "hidden_act", "hidden_dropout_prob", "attention_probs_dropout_prob"):
        value = getattr(config, key, None)
        if isinstance(value, (int, float, str, bool)):
            summary[key] = value
    return summary


def _tokenizer_source_dir(tokenizer, *, where: str) -> Path:
    """实际 tokenizer 声明的本地来源目录（只作审计出处；绑定不靠该目录，靠工件自有的序列化状态）。"""
    source = getattr(tokenizer, "name_or_path", "") or ""
    path = Path(source)
    if not source or not path.is_dir():
        raise ValueError(f"{where} tokenizer 没有可复核的本地来源目录（name_or_path={source!r}）："
                         "NLI 工件必须声明训练时实际使用 tokenizer 的本地出处，禁止无名内存态 tokenizer")
    return path


def _reconstruct_tokenizer(path, *, use_fast: bool, where: str):
    """从序列化状态重建 tokenizer（保存/加载/导出共用同一机制）：按 fast/slow 能力对齐，
    不信任目录自述、不静默换类（slow 对象不会被转换成别类而漂移）。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=bool(use_fast))


def _tokenizer_state(tokenizer, *, where: str) -> dict:
    """实际 tokenizer 的编码语义状态快照（保存侧实际对象与重建对象逐部件互验的唯一口径）。

    覆盖：vocab ID 映射（同长度改序也会被捕获）、fast backend 全量序列化（vocab 顺序/merges/
    normalizer/pre_tokenizer/post_processor/decoder）、padding/truncation/model_max_length/
    is_fast 运行配置、special tokens 映射与固定探针编码。取不到必需能力（get_vocab/编码）即拒绝：
    不可复核的 tokenizer 不得进入工件。"""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError(f"{where} tokenizer 不支持 get_vocab：无法绑定实际 ID 映射，拒绝不可复核对象")
    state = {"vocab": get_vocab()}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        state["backend"] = to_str()
    runtime = {}
    for name in ("padding_side", "truncation_side", "model_max_length", "is_fast"):
        try:
            runtime[name] = json.loads(json.dumps(getattr(tokenizer, name, None),
                                                  ensure_ascii=False))
        except (TypeError, ValueError):
            runtime[name] = str(getattr(tokenizer, name, None))
    state["runtime"] = runtime
    special = getattr(tokenizer, "special_tokens_map", None)
    if special is not None:
        state["special_tokens"] = json.loads(json.dumps(special, ensure_ascii=False, default=str))
    probes = []
    if callable(to_str):
                                                                               
                                                                              
                                   
        probes = [list(backend.encode(text).ids) for text in _TOKENIZER_PROBE_TEXTS]
    else:
        for text in _TOKENIZER_PROBE_TEXTS:
            encoded = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)
            probes.append(list(encoded["input_ids"]))
    state["probes"] = probes
    return state


def _tokenizer_state_sha256(state: dict) -> str:
    """语义状态摘要（工件/载荷/导出记录的都是这一个口径）。"""
    text = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_encoding_sha256(tokenizer) -> str:
    """绑定 continuation 编码语义；batch 编码每次显式覆盖 padding/truncation 的暂态。"""
    import copy

    clone = copy.deepcopy(tokenizer)
    clone("", add_special_tokens=True, padding=False, truncation=False)
    return _tokenizer_state_sha256(_tokenizer_state(clone, where="continuation"))


def _tokenizer_state_differences(actual: dict, rebuilt: dict) -> list:
    """两份状态快照的差异部件名（用于 fail-loud 报错，不输出整份词表）。"""
    return sorted(key for key in set(actual) | set(rebuilt) if actual.get(key) != rebuilt.get(key))


def _tokenizer_runtime_summary(tokenizer) -> dict:
    """工件 meta 可读的运行配置摘要（加载时按同一口径复核；强绑定仍是语义状态摘要）。"""
    summary = {"vocab_size": int(len(tokenizer)),
               "is_fast": bool(getattr(tokenizer, "is_fast", False))}
    for name in ("padding_side", "truncation_side", "model_max_length"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, (str, int, float)):
            summary[name] = value
    return summary


def _serialize_tokenizer_state(tokenizer, target_dir: Path, *, where: str) -> dict:
    """把实际 tokenizer 的完整状态写入工件自有目录并用同一机制复核（保存入口唯一实现）。

    重建结果必须与保存侧实际对象逐部件一致（vocab ID 映射/backend 序列化/运行配置/探针编码），
    不一致即拒绝——不可重建的实际状态绝不静默换成另一个 tokenizer。既不删除也不信任目标目录中
    本调用之外的文件：外来旧状态会破坏重建互验而 fail-loud。"""
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        raise ValueError(f"{where} tokenizer 不支持 save_pretrained：无法把实际对象状态写入自有目录，"
                         "拒绝不绑定实际对象的 NLI 工件")
    if _is_reparse(target_dir) or (target_dir.exists() and not target_dir.is_dir()):
        raise ValueError(f"{where} tokenizer 状态目录被占用或重定向：{target_dir}")
    if target_dir.is_dir():
        redirected = [child for child in target_dir.iterdir() if _is_reparse(child)]
        if redirected:
            raise ValueError(f"{where} tokenizer 状态目录内含链接/junction 组件："
                             f"{[str(path) for path in redirected[:3]]}：拒绝可能被重定向的写入")
    target_dir.mkdir(parents=True, exist_ok=True)
    actual_state = _tokenizer_state(tokenizer, where=where)
    save_pretrained(str(target_dir))
    use_fast = bool(getattr(tokenizer, "is_fast", True))
    rebuilt = _reconstruct_tokenizer(target_dir, use_fast=use_fast, where=where)
    rebuilt_state = _tokenizer_state(rebuilt, where=where)
    differences = _tokenizer_state_differences(actual_state, rebuilt_state)
    if differences:
        raise ValueError(f"{where} tokenizer 无法按序列化状态精确重建（差异部件 {differences}）："
                         "拒绝写入不可复现的工件，绝不静默换 tokenizer")
    return {"state_sha256": _tokenizer_state_sha256(actual_state),
            "files_sha256": FEAT.model_fingerprint(target_dir),
            "runtime": _tokenizer_runtime_summary(tokenizer)}


def _is_reparse(path: Path) -> bool:
    """符号链接 / junction / reparse point（按未解析路径判定；无法判定时 fail-closed 视为可疑）。"""
    if os.path.islink(path):
        return True
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _reparse_components(path: Path) -> list:
    """逐级检查未解析路径上的链接/reparse 组件（不做 realpath，避免先把重定向解析掉）。"""
    found, current = [], path
    while True:
        if _is_reparse(current):
            found.append(current)
        parent = current.parent
        if parent == current:
            return found
        current = parent


def _normalized(path) -> str:
    """字面绝对路径（不 resolve）；Windows 下按 normcase 大小写归一。"""
    return os.path.normcase(os.path.abspath(str(path)))


def _within(child, parent) -> bool:
    """child == parent 或位于 parent 之下（字面路径判定，不解析链接）。"""
    child_key, parent_key = _normalized(child), _normalized(parent)
    return child_key == parent_key or child_key.startswith(parent_key.rstrip("\\/") + os.sep)


def eval_nli_metrics(model, tokenizer, rows: Sequence[dict], *, max_length: int,
                     device: torch.device, batch_size: int = FT.INFER_BATCH) -> dict:
    """NLI eval 唯一通道（dev 选择与诊断共用）：row-native，或全行带 group_id 时 root-equal。

    只做前向指标：不改模型、不选权重、不接受验收数据语义，fit 之外的任何评估数据都不得
    反向进入 checkpoint 选择（选择逻辑在 train_nli 内，只吃 dev_rows）。
    """
    _require_nli_model(model, where="eval_nli_metrics")
    rows = list(rows)
    groups = _row_groups(rows) if rows else None
    weighting = "root_equal" if groups is not None else "row_native"
    if not rows:
        return {"n": 0, "n_groups": None, "loss": None, "row_loss": None, "accuracy": None,
                "macro_f1": None, "weighting": weighting, "mode": "eval(no dropout)",
                "prediction_counts": {name: 0 for name in NLI_CLASSES}}
    labels = row_labels(rows)
    logits = FT.logits_for(model, tokenizer, [row["request"] for row in rows], max_length=max_length,
                           device=str(device), batch_size=batch_size)
    losses = nn.functional.cross_entropy(torch.from_numpy(logits).float(), torch.from_numpy(labels),
                                         reduction="none").numpy()
    weights = EM._group_weights(groups) if groups is not None else np.ones(len(rows))
    predictions = logits.argmax(axis=1)
    return {
        "n": len(rows), "n_groups": len(set(groups)) if groups is not None else None,
        "weighting": weighting,
        "loss": float(np.average(losses, weights=weights)),
        "row_loss": float(losses.mean()),
        "accuracy": float(np.average(predictions == labels, weights=weights)),
        "macro_f1": EM.macro_f1(labels, predictions, n_classes=len(NLI_CLASSES), sample_weight=weights),
        "mode": "eval(no dropout)",
        "prediction_counts": {name: int((predictions == index).sum())
                              for index, name in enumerate(NLI_CLASSES)},
    }


def train_nli(
    model,
    tokenizer,
    rows: Sequence[dict],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    device: torch.device,
    bf16: bool,
    lr_backbone: float = FT.LR_BACKBONE,
    lr_head: float = FT.LR_HEAD,
    weight_decay: float = FT.WEIGHT_DECAY,
    checkpoint_selection: str = "final",
    selection_metric: str = "dev-loss",
    patience: int = 0,
    min_delta: float = 0.0,
    max_steps: int = 0,
    eval_every_steps: int = 0,
    eval_batch_size: int = FT.INFER_BATCH,
    dev_rows: "Sequence[dict] | None" = None,
    direction: str = NLI_PRIMARY_DIRECTION,
    event_callback=None,
) -> dict:
    """固定实例预算的 NLI 三类微调：唯一 joint_text / 优化器 / 窗口 / 编码原语，dev-only 选权重。

    - 行合同：{"request": memory-shaped dict, "label": 类名或 0..2, "split": "train"[, "group_id"]}；
      fit 只接受 split=train，带 sealed 标记或其它 split 一律拒绝；缺 split fail-closed。
    - 验收边界：本函数没有验收数据入口；dev_rows 只用于 best-dev 选择，独立验收不得进入 fit
      或 checkpoint 选择（验收评估在 train_nli 之外调用 eval_nli_metrics）。
    - 预算语义与七类臂一致：fixed-updates 每 update 恰 batch×accum 个实例、可跨 epoch 填满；
      epochs 模式跑满；token 计数含 special token、padding 单列、不含 dev/eval（复用同一窗口机制）。
    - 结果携带 budget/history/updates/selection/truncation/device_log，供父侧冻结正式预算与记账。
    """
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求 GPU（{device}）但 CUDA 不可用：如实报错，不回退 CPU")
    FT.validate_training_options(epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
                                 max_length=max_length, checkpoint_selection=checkpoint_selection,
                                 selection_metric=selection_metric, patience=patience,
                                 min_delta=min_delta, max_steps=max_steps,
                                 eval_every_steps=eval_every_steps, eval_batch_size=eval_batch_size)
    if not 0 <= int(seed) < 2 ** 32:
        raise ValueError("seed 必须在 [0, 2**32) 内（可复现要求）")
    max_positions = getattr(getattr(getattr(model, "backbone", None), "config", None),
                            "max_position_embeddings", None)
    if isinstance(max_positions, int) and int(max_length) > max_positions:
        raise ValueError(f"max_length {max_length} 超过骨干 max_position_embeddings "
                         f"{max_positions}：窗口必须落在骨干可表示范围内")
    hidden_size = _require_nli_model(model, where="train_nli")
    _require_trainable_backbone(model, where="train_nli")
    roles = _check_direction(direction)
    fit_rows = _checked_rows(rows, stage="fit")
    dev = _checked_rows(list(dev_rows), stage="dev") if dev_rows is not None else []
    labels = row_labels(fit_rows)
    require_three_classes(labels)
    if dev:
        row_labels(dev)
    elif checkpoint_selection == "best-dev":
        raise ValueError("best-dev 必须提供独立 dev 行（split=dev/calibration）")
    dev_groups = _row_groups(dev) if dev else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    optimizer = FT.build_joint_optimizer(model, train_backbone=True, lr_backbone=lr_backbone,
                                         lr_head=lr_head, weight_decay=weight_decay)
    covered = {id(param) for group in optimizer.param_groups for param in group["params"]}
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    backbone_ids = {id(param) for param in model.backbone.parameters()}
    if covered != trainable or not backbone_ids <= covered:
        raise RuntimeError("NLI 优化器未覆盖全部可训练参数（含骨干）：拒绝不完整覆盖的 full 训练")
    texts = [FT.joint_text(row["request"]) for row in fit_rows]
    planned_steps = max_steps or epochs * ((len(texts) + batch_size * grad_accum - 1) // (batch_size * grad_accum))
    planned_instances = max_steps * batch_size * grad_accum if max_steps else epochs * len(texts)

    def loss_fn(logits, chunk, window):
        if logits.ndim != 2 or logits.shape != (len(chunk), len(NLI_CLASSES)):
            raise RuntimeError(f"NLI 训练输出形状不符：{tuple(logits.shape)}")
        targets = torch.as_tensor(labels[chunk], dtype=torch.long, device=device)
        return nn.functional.cross_entropy(logits, targets, reduction="sum"), int(
            (logits.detach().argmax(dim=1) == targets).sum())

    def evaluate():
        return eval_nli_metrics(model, tokenizer, dev, max_length=max_length,
                                device=device, batch_size=eval_batch_size)

    windows = FT.training_windows(len(texts), epochs=epochs, batch_size=batch_size,
                                   grad_accum=grad_accum, max_steps=max_steps, rng=np.random.default_rng(seed))
    result = FT.fit_joint(model, tokenizer, texts, optimizer=optimizer, windows=windows,
                          planned_steps=planned_steps, planned_instances=planned_instances,
                          max_length=max_length, device=device, bf16=bf16, loss_fn=loss_fn,
                          evaluate=evaluate if dev else None, checkpoint_selection=checkpoint_selection,
                          selection_metric=selection_metric, patience=patience, min_delta=min_delta,
                          eval_every_steps=eval_every_steps, require_backbone_grad=True,
                          event_callback=event_callback)
    result.pop("continuation")
    backbone_grad_seen = result.pop("backbone_grad_seen")
    result["budget"]["mode"] = "fixed-updates" if max_steps else "epochs"
    result["selection"].update({"weighting": "root_equal" if dev_groups is not None else "row_native",
                                "dev_rows": len(dev),
                                "dev_groups": len(set(dev_groups)) if dev_groups is not None else None,
                                "secondary_metric": "dev_row_loss" if dev else None})
    result.update({
        "data": {"n_train": len(fit_rows), "n_dev": len(dev), "label_space": NLI_LABEL_SPACE,
                 "direction": direction, "joint_roles": roles,
                 "train_counts": {name: int((labels == index).sum()) for index, name in enumerate(NLI_CLASSES)},
                 "train_splits": dict(Counter(row["split"] for row in fit_rows)),
                 "dev_splits": dict(Counter(row["split"] for row in dev))},
        "truncation": {"train": FT.truncation_profile(tokenizer, [row["request"] for row in fit_rows], max_length)},
        "device_log": {"resolved": str(device), "cuda_available": bool(torch.cuda.is_available()),
                       "bf16": bool(bf16), "torch_threads": int(torch.get_num_threads())},
        "guard": {"head_classes": len(NLI_CLASSES), "hidden_size": hidden_size,
                  "backbone_trainable": True, "full_optimizer_coverage": True,
                  "optimizer_parameters": len(trainable), "backbone_parameters": len(backbone_ids),
                  "backbone_grad_seen": bool(backbone_grad_seen)},
    })
    if dev:
        result["truncation"]["dev"] = FT.truncation_profile(tokenizer, [row["request"] for row in dev], max_length)
    if event_callback is not None:
        event_callback({"event": "selection_restored", **result["selection"], "budget": result["budget"]})
    return result


class NLIScorer:
    """NLI 三类参考臂推理封装：memory-shaped request → 同一 joint_text → n×3 概率。

    classes 恒为 NLI_CLASSES（概率列序即该类序）；不做温度/阈值校准，不产出七类 verdict、
    风险/放行/认证表述。设备经唯一解析器解析：请求 GPU 不可用即报错，不回退 CPU。
    """

    classes = NLI_CLASSES

    def __init__(self, *, model, tokenizer, max_length: int, device: str = "cpu",
                 batch_size: int = FT.INFER_BATCH,
                 direction: str = NLI_PRIMARY_DIRECTION) -> None:
        _check_direction(direction)
        _require_nli_model(model, where="NLIScorer")
        self.direction = direction
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = str(FT.resolve_device(device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _items(requests) -> list:
        return [requests] if isinstance(requests, dict) else list(requests)

    def logits(self, requests) -> np.ndarray:
        """requests：请求字典序列（也接受单个请求）→ (n, len(classes)) logits。"""
        items = self._items(requests)
        if not items:
            return np.zeros((0, len(NLI_CLASSES)), dtype=np.float64)
        logits = FT.logits_for(self.model, self.tokenizer, items, max_length=self.max_length,
                               device=self.device, batch_size=self.batch_size)
        if logits.ndim != 2 or logits.shape[1] != len(NLI_CLASSES):
            raise RuntimeError(f"NLI 推理输出形状不符：{tuple(logits.shape)}"
                               f"（必须 (n, {len(NLI_CLASSES)})）：拒绝用其它输出空间冒充三类")
        return logits

    def predict_proba(self, requests) -> np.ndarray:
        """校准前（softmax）概率；列序 = NLI_CLASSES，消费者必须按该类序映射标签名。"""
        items = self._items(requests)
        if not items:
            return np.zeros((0, len(NLI_CLASSES)), dtype=np.float64)
        return FEAT.softmax(self.logits(items))

    def predict(self, requests) -> "list[str]":
        """argmax → 标签名（不是七类名，也不做任何七类归约）。"""
        items = self._items(requests)
        if not items:
            return []
        return [NLI_CLASSES[index] for index in self.predict_proba(items).argmax(axis=1)]


def nli_artifact_spec() -> dict:
    return {"schema": NLI_SCHEMA, "classes": NLI_CLASSES, "label_space": NLI_LABEL_SPACE,
            "kind": NLI_KIND, "recipe": FT.INPUT_RECIPE, "weights_file": NLI_WEIGHTS_FILE,
            "meta_file": NLI_META_FILE, "roles": NLI_JOINT_ROLES,
            "model_guard": _require_nli_model, "scorer_type": NLIScorer}


def save_nli_artifact(scorer: NLIScorer, directory: "str | Path", *, weights: str = "full",
                      backbone_dir: "str | Path", backbone_fingerprint: "str | None" = None,
                      train_meta: dict, direction: "str | None" = None) -> dict:
    """旧三类工件硬闸；保存、实际 config/tokenizer 绑定委托唯一可配置机制。"""
    return save_joint_artifact(scorer, directory, spec=nli_artifact_spec(), weights=weights,
                               backbone_dir=backbone_dir, backbone_fingerprint=backbone_fingerprint,
                               train_meta=train_meta, direction=direction)


def save_joint_artifact(scorer, directory, *, spec, weights="full", backbone_dir,
                        backbone_fingerprint=None, train_meta, direction=None,
                        bindings=None, extra_payload=None) -> dict:
    """Fresh full 工件唯一保存机制；类序/配方/方向是输入，实际对象与重建状态双向互验。"""
    NLI_SCHEMA, NLI_CLASSES = spec["schema"], tuple(spec["classes"])
    NLI_LABEL_SPACE, NLI_KIND = spec["label_space"], spec["kind"]
    NLI_WEIGHTS_FILE, NLI_META_FILE = spec["weights_file"], spec["meta_file"]
    recipe = spec["recipe"]
    _require_nli_model = spec["model_guard"]
    def _check_direction(value):
        if value not in spec["roles"]:
            raise ValueError("工件方向无效")
        return dict(spec["roles"][value])

    if weights != "full":
        raise ValueError("NLI 工件只支持 weights=full（参考臂端到端权重）")
    actual_direction = getattr(scorer, "direction", None)
    if direction is None:
        direction = actual_direction
    elif direction != actual_direction:
        raise ValueError(f"显式 direction={direction!r} 与 scorer 实际方向 {actual_direction!r} 不一致："
                         "方向只能从 scorer 派生，拒绝把逆向臂写成主向")
    roles = _check_direction(direction)
    if tuple(getattr(scorer, "classes", ()) or ()) != NLI_CLASSES:
        raise ValueError(f"NLI 工件只接受 classes={NLI_CLASSES} 的 scorer，实际 "
                         f"{tuple(getattr(scorer, 'classes', ()) or ())!r}")
    hidden_size = _require_nli_model(scorer.model, where="save_nli_artifact")
    _require_trainable_backbone(scorer.model, where="save_nli_artifact")
    dropout = float(scorer.model.dropout.p)
    state = FT.cpu_state(scorer.model)
    if not any(name.startswith("backbone.") for name in state) \
            or not any(name.startswith("classifier.") for name in state):
        raise ValueError("NLI 全量载荷必须同时含 backbone.* 与 classifier.* 权重")
    backbone_dir = Path(backbone_dir)
    if not backbone_dir.is_dir():
        raise FileNotFoundError(f"backbone_dir 不存在：{backbone_dir}")
    with torch.random.fork_rng(devices=[]):
        source_model, effective_config = FT.build_joint_model(backbone_dir, num_labels=len(NLI_CLASSES))
    del source_model
    actual_config = getattr(getattr(scorer.model, "backbone", None), "config", None)
    if actual_config is None:
        raise ValueError("模型没有可绑定的 backbone.config：拒绝无配置绑定的工件")
    config_digest = _config_digest(actual_config)
    if _config_digest(effective_config) != config_digest:
        raise ValueError("backbone_dir 的有效 config 与实际模型 config 不一致：拒绝按声明目录绑定工件")
    input_params = {"recipe": recipe, "max_length": int(scorer.max_length)}
    if "input_limits" in spec:
        input_params["input_limits"] = dict(spec["input_limits"])
    if input_params["max_length"] < 1:
        raise ValueError("scorer max_length must be positive")
    validator = spec.get("validate_training")
    if validator is not None:
        validator(train_meta, extra_payload, input_params=input_params)
    measured_fingerprint = FEAT.model_fingerprint(backbone_dir)
    if backbone_fingerprint is not None and str(backbone_fingerprint) != measured_fingerprint:
        raise ValueError("调用方 backbone_fingerprint 与目录实测指纹不符：拒绝绑定未复核的声明值")
    tokenizer_path = _tokenizer_source_dir(scorer.tokenizer, where="save_nli_artifact")
    directory = Path(directory)
    if directory.exists() or os.path.islink(directory) or _is_reparse(directory):
        raise ValueError(f"NLI 工件只允许写入全新目录：{directory} 已存在或被重定向"
                         "（禁止覆盖、复用空目录或写入链接；绝不触碰既有工件）")
    if ".." in directory.parts or _reparse_components(directory):
        raise ValueError("artifact destination has redirected or parent components")
    if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("non-finite artifact weights")
    directory.mkdir(parents=True)
    try:
        actual_config.save_pretrained(str(directory / "config"))
        owned = {"effective_config_dir": "config",
                 "effective_config_files_sha256": FEAT.model_fingerprint(directory / "config"),
                 "train_sha256": _tokenizer_state_sha256(train_meta)}
        if "input_limits" in input_params:
            owned["input_limits"] = input_params["input_limits"]
        tokenizer_state = _serialize_tokenizer_state(scorer.tokenizer,
                                                     directory / NLI_TOKENIZER_DIR,
                                                     where="save_nli_artifact")
        payload = {"schema": NLI_SCHEMA, "state_dict": state, "classes": list(NLI_CLASSES),
                   "recipe": recipe, "max_length": input_params["max_length"], "label_space": NLI_LABEL_SPACE, "kind": NLI_KIND,
                   "weights": weights, "direction": direction, "joint_roles": roles,
                   "backbone_config_sha256": config_digest, "hidden_size": hidden_size,
                   "dropout": dropout, "tokenizer_state_sha256": tokenizer_state["state_sha256"]}
        if bindings is not None:
            payload["bindings"] = bindings
        if extra_payload is not None:
            payload["continuation"] = extra_payload
        payload.update(owned)
        torch.save(payload, directory / NLI_WEIGHTS_FILE)
        meta = {
            "schema": NLI_SCHEMA, "kind": NLI_KIND, "classes": list(NLI_CLASSES),
            "label_space": NLI_LABEL_SPACE, "input_recipe": recipe,
            "direction": direction, "joint_roles": roles,
            "max_length": int(scorer.max_length), "batch_size": int(scorer.batch_size),
            "weights": weights, "weights_file": NLI_WEIGHTS_FILE,
            "weights_sha256": FT.file_sha256(directory / NLI_WEIGHTS_FILE),
            "backbone_path": str(backbone_dir), "backbone_fingerprint": measured_fingerprint,
            "backbone_config_sha256": config_digest, "backbone_config": _config_summary(actual_config),
            "tokenizer_dir": NLI_TOKENIZER_DIR, "tokenizer_state_sha256": tokenizer_state["state_sha256"],
            "tokenizer_files_sha256": tokenizer_state["files_sha256"],
            "tokenizer_runtime": tokenizer_state["runtime"], "tokenizer_path": str(tokenizer_path),
            "hidden_size": hidden_size, "dropout": dropout,
            "binding_note": "actual config and serialized tokenizer state; full weights; source paths are provenance",
            "created_at": datetime.now().isoformat(timespec="seconds"), "train": train_meta,
        }
        if bindings is not None:
            meta["bindings"] = bindings
        meta.update(owned)
        with (directory / NLI_META_FILE).open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
    except BaseException:
                                                    
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return meta


def is_nli_artifact(model_dir: "str | Path") -> bool:
    """NLI 工件识别（nli_meta.json 且 schema 匹配）；七类 is_finetune_artifact 对 NLI 目录恒为 False。"""
    meta_path = Path(model_dir) / NLI_META_FILE
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(meta, dict) and meta.get("schema") == NLI_SCHEMA


def load_nli_scorer(model_dir: "str | Path", device: str = "cpu") -> NLIScorer:
    """只接受旧三类 schema/类序/方向；共享机制不放宽旧 loader。"""
    return load_joint_artifact(model_dir, device=device, spec=nli_artifact_spec())[0]


def load_joint_artifact(model_dir, *, spec, device="cpu", bindings=None):
    directory = Path(model_dir)
    meta_path, weights_path = directory / spec["meta_file"], directory / spec["weights_file"]
    if not meta_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"NLI 工件不完整：{directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = tuple(spec["classes"])
    for key, value in (("schema", spec["schema"]), ("classes", list(classes)),
                       ("label_space", spec["label_space"]), ("input_recipe", spec["recipe"]),
                       ("kind", spec["kind"]), ("weights", "full"), ("weights_file", spec["weights_file"])):
        if meta.get(key) != value:
            name = {"classes": "类别序", "input_recipe": "输入配方", "label_space": "标签空间"}.get(key, key)
            raise ValueError(f"NLI 工件 {name} 不符")
    direction = meta.get("direction")
    if direction not in spec["roles"] or meta.get("joint_roles") != spec["roles"][direction]:
        raise ValueError("NLI 工件方向与槽位映射（joint_roles）不一致")
    for key in ("backbone_path", "backbone_config_sha256", "tokenizer_dir",
                "tokenizer_state_sha256", "tokenizer_files_sha256", "tokenizer_runtime", "hidden_size", "dropout"):
        if key not in meta:
            raise ValueError(f"NLI 工件 meta 缺少 {key}")
    if FT.file_sha256(weights_path) != meta.get("weights_sha256"):
        raise ValueError("NLI 权重文件哈希不符（损坏或被替换）")
    if bindings is not None and meta.get("bindings") != bindings:
        raise ValueError("artifact bindings/source drift")
    if meta["tokenizer_dir"] != NLI_TOKENIZER_DIR:
        raise ValueError("NLI 工件 tokenizer_dir 不符")
    tokenizer_dir = directory / NLI_TOKENIZER_DIR
    if not tokenizer_dir.is_dir() or _is_reparse(tokenizer_dir):
        raise ValueError("NLI 工件 tokenizer 状态目录缺失或被重定向")
    if FEAT.model_fingerprint(tokenizer_dir) != meta["tokenizer_files_sha256"]:
        raise ValueError("NLI 工件 tokenizer 状态文件指纹不符（被替换或损坏）")
    runtime = meta.get("tokenizer_runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("is_fast"), bool):
        raise ValueError("NLI 工件 tokenizer_runtime 无效")
    tokenizer = _reconstruct_tokenizer(tokenizer_dir, use_fast=runtime["is_fast"], where="load_joint_artifact")
    if _tokenizer_state_sha256(_tokenizer_state(tokenizer, where="load_joint_artifact")) != meta["tokenizer_state_sha256"]:
        raise ValueError("NLI 工件 tokenizer 语义状态摘要不符")
    summary = _tokenizer_runtime_summary(tokenizer)
    for key, value in runtime.items():
        if summary.get(key) != value:
            raise ValueError(f"NLI 工件 tokenizer 运行配置 {key} 与记录不符")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("NLI 权重载荷结构不符")
    for key in ("schema", "classes", "label_space", "kind", "weights", "direction", "joint_roles",
                "backbone_config_sha256", "tokenizer_state_sha256", "dropout"):
        if payload.get(key) != meta[key]:
            name = {"tokenizer_state_sha256": "tokenizer 状态摘要", "backbone_config_sha256": "backbone config 摘要"}.get(key, key)
            raise ValueError(f"NLI 载荷 {name} 与 meta 不一致")
    if payload.get("recipe") != spec["recipe"]:
        raise ValueError("NLI 载荷配方不符")
    if bindings is not None and payload.get("bindings") != bindings:
        raise ValueError("payload bindings/source drift")
    input_params = None
    if "max_length" in payload or spec.get("require_input_parameters"):
        length = payload.get("max_length")
        if type(length) is not int or length < 1 or type(meta.get("max_length")) is not int or length != meta["max_length"]:
            raise ValueError("scorer input max_length meta/payload mismatch")
        input_params = {"recipe": payload["recipe"], "max_length": length}
        if "input_limits" in spec:
            if meta.get("input_limits") != payload.get("input_limits") or meta.get("input_limits") != spec["input_limits"]:
                raise ValueError("scorer input limits meta/payload mismatch")
            input_params["input_limits"] = payload["input_limits"]
    validator = spec.get("validate_training")
    if "train_sha256" in meta or validator is not None:
        if meta.get("train_sha256") != payload.get("train_sha256") \
                or meta.get("train_sha256") != _tokenizer_state_sha256(meta.get("train")):
            raise ValueError("training ledger hash mismatch")
    if validator is not None:
        try:
            validator(meta["train"], payload.get("continuation"), state_dict=payload["state_dict"], input_params=input_params)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("training ledger malformed") from error
    if "effective_config_dir" in meta:
        from transformers import AutoConfig

        config_dir = directory / "config"
        if meta["effective_config_dir"] != "config" or payload.get("effective_config_dir") != "config" \
                or FEAT.model_fingerprint(config_dir) != meta["effective_config_files_sha256"] \
                or payload.get("effective_config_files_sha256") != meta["effective_config_files_sha256"]:
            raise ValueError("effective config files drift")
        owned_config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
        if _config_digest(owned_config) != meta["backbone_config_sha256"]:
            raise ValueError("effective config digest drift")
    model, config = FT.build_joint_model(meta["backbone_path"], dropout=float(meta["dropout"]), num_labels=len(classes))
    if _config_digest(config) != meta["backbone_config_sha256"]:
        raise ValueError("NLI 工件绑定的骨干 config 与实际 config 不一致")
    actual_hidden = spec["model_guard"](model, where="load_joint_artifact")
    if actual_hidden != int(meta["hidden_size"]) or int(payload.get("hidden_size", -1)) != actual_hidden:
        raise ValueError("NLI 工件 hidden_size 与实际 config 不一致")
    if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in payload["state_dict"].values()):
        raise ValueError("non-finite artifact weights")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"NLI 权重含多余键：{sorted(unexpected)[:5]}")
    if missing:
        raise ValueError(f"NLI 全量权重缺键：{sorted(missing)[:5]}")
    scorer = spec["scorer_type"](model=model, tokenizer=tokenizer, max_length=int(meta["max_length"]),
                                  device=device, batch_size=int(meta.get("batch_size", FT.INFER_BATCH)),
                                  direction=direction)
    return scorer, meta, payload


def check_nli_reload_parity(scorer: NLIScorer, directory: "str | Path",
                            probe_requests: Sequence[dict], *, device: str = "cpu",
                            atol: float = 1e-6) -> dict:
    """state_dict 重载 parity：概率/argmax/类序/方向/窗口必须一致，否则 fail-loud。"""
    reloaded = load_nli_scorer(directory, device=device)
    before = scorer.predict_proba(list(probe_requests))
    after = reloaded.predict_proba(list(probe_requests))
    difference = float(np.max(np.abs(before - after))) if len(before) else 0.0
    argmax_match = bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1)))
    ok = (argmax_match and difference <= atol and tuple(reloaded.classes) == NLI_CLASSES
          and reloaded.direction == scorer.direction and reloaded.max_length == scorer.max_length)
    if not ok:
        raise RuntimeError(f"NLI 重载 parity 失败：max_abs_diff={difference}, argmax_match={argmax_match}")
    return {"max_abs_diff": difference, "argmax_match": argmax_match, "ok": True}


def export_backbone(model, tokenizer, directory: "str | Path",
                    *, direction: "str | None" = None) -> dict:
    """把 NLI 训练后的 backbone + tokenizer 导出到**全新目录**，供未来七类 --embedding-model warm-start。

    写前 fail-closed：目标必须不存在、按未解析路径逐级拒绝链接/junction/reparse 组件与 .. 组件
    （不做目标解析优先，避免先把重定向解析掉）；源骨干按**真实身份**解析（经 junction/alias 加载
    也解析到实际目录），目标字面拼写与其真实位置两侧都不得落入源或源别名子树；源不可解析即拒。
    冻结骨干的模型拒绝导出（不得虚称 NLI-trained 骨干）。导出把实际 tokenizer 序列化进目标并与
    重建逐部件互验、复核导出 config 摘要与源目录指纹不变；manifest 记录实际 config 摘要与 tokenizer
    绑定；导出物不含 NLI head，原骨干目录与既有工件不被触碰。
    """
    if direction is not None:
        _check_direction(direction)
    _require_nli_model(model, where="export_backbone")
    _require_trainable_backbone(model, where="export_backbone")
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "save_pretrained"):
        raise ValueError("模型没有可导出的 backbone（需要 save_pretrained）")
    if not hasattr(tokenizer, "save_pretrained"):
        raise ValueError("tokenizer 不支持 save_pretrained")
    source_config = getattr(backbone, "config", None)
    if source_config is None:
        raise ValueError("骨干没有 config：无法证明导出目标不在源内，fail-closed")
    source = getattr(source_config, "_name_or_path", "") or ""
    if not source or not Path(source).is_dir():
        raise ValueError(f"无法解析源骨干目录（config._name_or_path={source!r}）：导出必须能证明"
                         "目标不在源内，provenance 不明即 fail-closed")
    source_dir = Path(source)
    try:
        resolved_source = Path(os.path.realpath(source_dir, strict=True))
    except OSError as error:
        raise ValueError(f"源骨干目录无法解析真实身份（{source_dir}）：{error}：fail-closed，"
                         "拒绝可能在重定向路径下写入的导出") from None
    if not resolved_source.is_dir():
        raise ValueError(f"源骨干真实身份不是目录（{resolved_source}）：fail-closed")
    target = Path(directory)
    if ".." in target.parts:
        raise ValueError(f"导出目标含 .. 组件：{target}：拒绝可能被词法折叠绕过的路径")
    if target.exists() or os.path.islink(target) or _is_reparse(target):
        raise FileExistsError(f"导出目标已存在或被重定向：{target}；只允许全新目录"
                              "（禁止覆盖、复用空目录或写入链接）")
    redirects = _reparse_components(target)
    if redirects:
        raise ValueError(f"导出目标路径含符号链接/junction/reparse 组件："
                         f"{[str(path) for path in redirects[:3]]}：拒绝可能被重定向的写入")
    if _within(target, source_dir) or _within(target, resolved_source):
        raise ValueError(f"导出目标位于或等于源骨干目录（按真实身份比对）：{target}"
                         f"（源 {source_dir} → {resolved_source}）：只允许独立全新目录")
    resolved_target = os.path.realpath(target, strict=False)
    if _within(resolved_target, resolved_source):
        raise ValueError(f"导出目标真实位置位于源骨干目录内：{resolved_target}"
                         f"（源 {resolved_source}）：拒绝写入源或源别名子树")
    tokenizer_path = _tokenizer_source_dir(tokenizer, where="export_backbone")
    source_fingerprint = FEAT.model_fingerprint(source_dir)
    config_digest = _config_digest(source_config)
    actual_tokenizer_state = _tokenizer_state(tokenizer, where="export_backbone")
    target.mkdir(parents=True)
    try:
        backbone.save_pretrained(str(target))
        tokenizer.save_pretrained(str(target))
        from transformers import AutoConfig

        exported_config = AutoConfig.from_pretrained(str(target), local_files_only=True)
        if _config_digest(exported_config) != config_digest:
            raise ValueError("导出骨架的 config 与源骨干 config 不一致：拒绝交付不可复现的 warm-start 目录")
        exported_tokenizer = _reconstruct_tokenizer(
            target, use_fast=bool(getattr(tokenizer, "is_fast", True)), where="export_backbone")
        differences = _tokenizer_state_differences(
            actual_tokenizer_state,
            _tokenizer_state(exported_tokenizer, where="export_backbone"))
        if differences:
            raise ValueError(f"导出 tokenizer 无法按导出目录精确重建（差异部件 {differences}）："
                             "拒绝交付无法复现的 warm-start 目录")
        if FEAT.model_fingerprint(source_dir) != source_fingerprint:
            raise RuntimeError("导出改动了源骨干目录：违反只向目标写入的边界")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)                       
        raise
    files = sorted(path.name for path in target.rglob("*") if path.is_file())
    return {"schema": NLI_BACKBONE_EXPORT_SCHEMA, "directory": str(target), "files": files,
            "fingerprint": FEAT.model_fingerprint(target),
            "config_sha256": config_digest, "config": _config_summary(exported_config),
            "source_backbone_path": str(source_dir),
            "source_backbone_resolved": str(resolved_source),
            "direction": direction,
            "tokenizer_state_sha256": _tokenizer_state_sha256(actual_tokenizer_state),
            "tokenizer_runtime": _tokenizer_runtime_summary(tokenizer),
            "tokenizer_source": {"path": str(tokenizer_path), "vocab_size": int(len(tokenizer))},
            "created_at": datetime.now().isoformat(timespec="seconds")}
