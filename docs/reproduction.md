# 取得工件、运行模型与复算实验

## 1. 先确定复现的是哪一层

本仓区分三种工作：**重算原预测的指标**、**重载已训权重进行新输入推理**、**重新训练一个实验臂**。前两种不需要重新付费生成标签或跑GPU训练。第三种需要对应数据版本、选点规则和环境；不能用今天的输入或基模HEAD替代缺失的历史记录，再称逐位复现。

代码在Git中可浏览；体积大的数据与权重在[v0.1.0 Release](https://github.com/hty8870/biodata-model-lab/releases/tag/v0.1.0)。[assets.json](../assets.json)登记每包大小、SHA-256和下载URL；[catalog.json](../models/catalog.json)把工件ID对应到任务、类序、基模和检查点。146个工件记录包含不同seed、格式、初始化与检查点，不是146种独立模型。

```sh
git clone https://github.com/hty8870/biodata-model-lab.git
cd biodata-model-lab
python tools/fetch.py --list
```

下载器只取得显式指定的一项，校验SHA后解压；不会默认下载全部模型，也不覆盖内容不同的现有文件。神经权重使用safetensors；历史joblib只应从可信且已验哈希的来源加载。

## 2. 环境

基础推理栈以Python 3.12、PyTorch 2.11、transformers 4.57.6、PEFT 0.21记录。安装示例：

```sh
python -m venv .venv
# 激活该环境后：
python -m pip install -r requirements.txt
```

CUDA版本和驱动取决于机器。`requirements/qwen35.txt`是Qwen3.5架构的独立环境配方，不与4.57栈混装；本次公开迁移没有重新验收该栈。依赖列表是已记录版本，不是覆盖所有平台的哈希锁或位级数值保证。

历史训练用过sentence-transformers 3.4.1，存在[GHSA-jhr6-gm9c-rqjv](https://github.com/advisories/GHSA-jhr6-gm9c-rqjv)告警，**不放入默认推理安装**。重现历史特征流水线时应在隔离、无密钥且只读可信本地工件的环境中审查依赖；升级后另做数值验收。公开保留旧方法不表示推荐旧依赖用于联网生产服务。

仅下载、纯Python L0和指标复算不需要torch。以下推理均默认CPU，不联网下载基模；如需GPU显式加`--device cuda`。使用GPU或不同批量可能产生浮点差异，应报告实际环境与容差。

## 3. 最小运行：纯Python工具分流

```sh
python tools/l0_predict.py --model models/l0/l0_route_binary.json --text "帮我下载这些数据"
```

输出类别概率、温度及记录工作点。预测类别不等于授权执行，置信度低于工作点时应进入原升级路径。其它两个JSON分别对应检索预取和历史三分类。

## 4. LTR：先用冻结九维特征检查权重

```sh
python tools/fetch.py --model ltr-active --output .cache/ltr
python tools/predict.py --model-dir .cache/ltr --input examples/ltr-features.jsonl
```

输入九维向量严格按随包`ltr-spec.json`的顺序，使用同一均值与标准差归一。输出是排序分数，不是相关性概率。示例只验证模型计算；端到端检索还需要产品的语料、硬过滤与同源特征构造，不能用两条示例证明实际检索收益。

## 5. 记忆关系：使用正确的输入配方

### 最后有部署验收证据的五叶版本

```sh
python tools/fetch.py --model memory-five-last-deployed --output .cache/c5
python tools/fetch.py --base sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --output .cache/minilm
python tools/prepare_pair.py --base .cache/minilm --input examples/memory-pairs.jsonl --output .cache/c5-pairs.jsonl --max-length 384
python tools/predict.py --model-dir .cache/c5 --base .cache/minilm --input .cache/c5-pairs.jsonl
```

`prepare_pair.py`复用公开源的V3树表示，保留old/new方向、预算完整性与fallback提示；它不把ID、gold或教师票数写入模型文本。旧七类与早期表示使用各自冻结formatter或已归档编码，**不能统一套V3再声称重算原成绩**。

### 后期R2原子关系研究候选

```sh
python tools/fetch.py --model memory-r2-seed13-terminal --output .cache/r2
python tools/fetch.py --base Qwen/Qwen3-0.6B --output .cache/qwen3
python tools/prepare_pair.py --base .cache/qwen3 --input examples/memory-pairs.jsonl --output .cache/r2-pairs.jsonl --max-length 768 --role-independent-frame
python tools/predict.py --model-dir .cache/r2 --base .cache/qwen3 --input .cache/r2-pairs.jsonl
```

R2的输出序为`equivalent / compatible / conflict`，不是五叶，也不是`coexist / replace_all / merge_preserve`动作三类。示例用于接口检查；正式指标使用冻结题集的原visible_text／encoding和方向。未经单独校准的softmax不构成风险概率或写入许可。

## 6. Qwen工具路由与其它格式

路由adapter需要其对应基模，不能套用记忆模型的0.6B底座。例如`routing-qwen3-rlcd`对应Qwen3-1.7B。未附本地快照的基模，下载器要求显式`--upstream`，并使用[base-models.json](../provenance/base-models.json)记录的revision：

```sh
python tools/fetch.py --model routing-qwen3-rlcd --output .cache/router
python tools/fetch.py --base Qwen/Qwen3-1.7B --upstream --output .cache/qwen17
# 输入JSONL每行形如 {"id":"example","text":"查找人类肺癌数据"}
python tools/predict.py --model-dir .cache/router --base .cache/qwen17 --input request.jsonl --task binary
```

只有上游观察revision而无历史commit时，明确标为历史未记录；今日能下载不证明与旧训练时完全相同。路由读取任务prompt、候选token和温度，输出选项概率，不生成任意文本。上述1.7B示例未在本次迁移重新下载并运行。

有限集合模型使用随附模拟引擎和位图语义；历史sklearn模型使用对应特征流水线，不伪装成统一自然文本接口。模型目录登记未找到的原始GBDT对象、CV拟合和未执行的完整训练，不补造权重。

## 7. 下载数据并重算逐题表

```sh
python tools/fetch.py --source-data
python tools/recompute.py --selftest
```

数据包解压后保留`source/`、`studies/`、`historical/`和`datasets/`层级。实际联合训练输入位于`datasets/semantic-text/{early,tree}/{train,dev}.jsonl`：每版118186训练行、13202开发行；公开与原件的pair、label、label_space逐行一致，变更在来源元数据中登记。第三方正文保留其原许可，尤其ANLI／OCNLI的非商业条件，见[THIRD_PARTY.md](../THIRD_PARTY.md)。

R2与LLM的最终CSV位于：

`studies/memory/真实文本训练与上线-2026-10-14/标签粒度与动作三类-2026-10-24/R2与LLM配对对照-2026-10-26/results/最终逐题对照-2026-10-26.csv`

```sh
python tools/recompute.py --csv "studies/memory/真实文本训练与上线-2026-10-14/标签粒度与动作三类-2026-10-24/R2与LLM配对对照-2026-10-26/results/最终逐题对照-2026-10-26.csv" --domain memory --prediction r2_seed13
```

其余列为`r2_seed42`、`llm_single`、`llm_batch8_clean`；432题正确数应分别为410、418、408、423。该工具重算行等权点指标与混淆矩阵；根等权、连通簇bootstrap和配对区间须使用对应归档分析脚本，不能混用口径。

## 8. 重新训练：从实验配方而非一条万能命令出发

| 阶段 | 训练／评测入口 | 必须一起固定的内容 |
|---|---|---|
| LTR | `source/scripts/train_ltr.py`，`source/research/ltr_data/` | 数据层合并、train/eval标签、f0/f3特征、mix-in范围、seed、偏好对采样；现役早于正式注册，部分指纹未记 |
| 工具分流 | `source/research/`中的路由训练、蒸馏和导出脚本；随包`routing-spec.json` | 共享split、增广成员、任务prompt、候选token、CE/KL目标、温度、工作点和目标量化栈 |
| 历史七类 | `historical/seven/source/research/memory_relations/`及`studies/memory/数据修复与新训练/` | 原七类类序、语义根划分、作者、裁决、训练规模、根等权选点与风险认证失败 |
| 文本五叶／dual3 | `historical/five-early/source/`、`historical/five-tree/source/`与`source/research/memory_relations/` | 母模型、canonical输入、方向、来源混合、父NLL、250更新与selected检查点；源码指纹不足处不宣称完全同源 |
| 动作／基座对照 | `studies/memory/真实文本训练与上线-2026-10-14/标签粒度与动作三类-2026-10-24/`及相邻`固定数据拟合与基座对照-2026-10-24/` | 嵌套细类映射、相同动作任务、初始化聚合概率、完整输入、曝光量、选点与二次确认 |
| 原子关系／R1–R3 | 上述目录下`原子关系跨模型复验与Qwen训练-2026-10-25/train_relation_heads.py`、`冲突强化与记忆披露-2026-10-25/relation/train_conflict_arms.py` | direct3/fine5任务、family与作者曝光、LoRA配置、同terminal预算、反事实配对、margin对照与两seed |

历史锁可能绑定旧目录或原始`.pt`容器。公开safetensors保留选中张量，但不是原恢复容器的字节替身；脱敏源码的公开SHA也不同于原SHA。应建立**新的明确复现锁**并引用原锁，而不是删掉断言绕过。原结果、配方复跑和修改后的新实验必须分开登记。

本次公开迁移没有重训全部模型、没有重新调用付费教师或重跑全部终测。实际验收范围与数值见[validation.json](../provenance/validation.json)。
