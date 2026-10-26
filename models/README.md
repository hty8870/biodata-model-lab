# 模型与工件目录

这里按研究用途组织。下表数量是保存/发布的版本、种子、检查点或表示格式，不是独立架构数量；基模初始化、程序模拟、研究候选和产品状态分别标记。

| 研究组 | 工件数 |
|---|---:|
| 检索融合RankNet | 2 |
| Qwen工具路由 | 7 |
| 官方／自建来源三类对照 | 2 |
| 历史七类关系 | 60 |
| 固定数据基座与拟合对照 | 8 |
| 动作三类／六叶监督 | 10 |
| 自然／反事实冲突增强 | 12 |
| 原子关系三类／五叶对照 | 8 |
| 最后验收部署的五叶版本 | 1 |
| 文本五叶／双向三类（早期表示） | 6 |
| 文本五叶／双向三类（树表示） | 6 |
| 有限集合头设计模拟 | 12 |
| NLI参照／原始初始化 | 2 |
| 字符TF-IDF线性路由 | 10 |

完整结构见[catalog.json](catalog.json)，包括源工件SHA、公开包SHA、类序、基模、状态、种子/检查点、保存张量及验证方式。

## 常用入口

| 别名 | 工件ID |
|---|---|
| `ltr-active` | `ltr-ranknet-active-49d305ac` |
| `routing-qwen3-rlcd` | `routing-qwen3_decider_b-371fe986` |
| `memory-r2-seed13-terminal` | `memory-conflict-augmentation-2026-10-25-relation-runs-r2-seed13-checkpoints-terminal-8249593a` |
| `memory-atomic-direct3-seed13-dev-best` | `memory-atomic-qwen--2026-10-25-runs-direct3-seed13-checkpoints-best-dev-184ececc` |
| `memory-five-last-deployed` | `memory-five-active-v3-c5-seed13-active-a74fd57c` |

`memory-r2-seed13-terminal`是复现示例，不是从本轮分数挑出的上线赢家。模型只返回分数/关系，不批准编辑。

## 直接可运行的轻量模型

不下载基座即可运行：

```sh
python tools/l0_predict.py --model models/l0/l0_route_binary.json --text "帮我下载这些数据"
```

它输出概率与记录工作点；低置信应按原策略升级，不等于看到label就执行写入。

## 权重与基模

较大工件放在Releases；公开包剥离优化器/恢复状态，神经张量逐项核对，adapter和已序列化估计器另按原字节核对。joblib/pickle只应从可信、验哈希的来源加载。

三个本地留存基模（MiniLM、Qwen3-0.6B、MacBERT）附可下载快照；其余基模提供官方地址、发布时观察到的revision及历史未记录说明。不得把今日HEAD倒填为旧训练版本。

历史GBDT比较器与任务结构CV没有找到单一原始终态权重；配方、数据、原结果照实披露。未执行的对称C6完整训练也不会伪造一个下载项。

不同格式的相同权重与不同seed不能重复算成额外独立实验。研究高分、代码接入和真实部署是不同事实。
