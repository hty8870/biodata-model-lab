# 来源、许可与引用

## 本项目贡献

本仓自有代码按根目录MIT许可公开。研究文稿、程序构造的例子、代理标签、实验结果与训练参数用于可审计研究披露；**根目录代码许可不重新授权第三方基模、基准正文、目录原文或模型提供商的材料**。混合数据必须按记录来源核对上游条件，不应将所有输入统一标成“本项目原创”。

模型资产的`model.json`记录基模和任务；其中Apache-2.0说明上游基模的许可与需保留的义务，不构成对训练数据及所有用途的统一商用权利保证。涉及非商业NLI数据的研究应遵守其NC条件；公开研究权重不等于生产安全认证或任意用途授权。

## 基模

发布时通过官方模型元数据核对，以下六个基模均报告Apache-2.0。完整校验时间、观察到的upstream revision和历史记录状态见[base-models.json](provenance/base-models.json)。标准许可文本保留在[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)。

| 基模 | 用途 | 历史revision情况 |
|---|---|---|
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 向量特征、早期关系编码器、NLI参照 | 历史commit未完整记录；本地文件指纹另留，不将今日HEAD倒填成训练版本 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 前代工具路由LoRA | 同上 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 工具分流硬/软目标及弃权研究 | 同上 |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 基座×训练目标对照 | 同上；需要相应transformers架构支持 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 动作／原子关系／冲突增强LoRA | `c1899de289a04d12100db370d81485cdf75e47ca`，在原实验中钉住 |
| [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base) | 固定数据基座对照 | `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`，在原实验中钉住 |

微调与adapter属于本项目实际训练，不能称为上述团队官方发布的下游成品。原模型名称和来源用于归属与复现，不表示认可。本地教师量化标签不在“已认证基模版本”清单中。

## 公开基准

| 基准 | 冻结来源revision | 上游许可 |
|---|---|---|
| [OCNLI](https://github.com/CLUEbenchmark/OCNLI) | `b53efdee17257a5c33993cf6fcf8ffff0497ea0e` | [CC BY-NC 2.0](https://github.com/CLUEbenchmark/OCNLI#license) |
| [ANLI](https://github.com/facebookresearch/anli) | `b5ce27af54a53bd495af7260fdc485000eed15f8` | [CC BY-NC 4.0](https://github.com/facebookresearch/anli/blob/main/LICENSE) |
| [HANS](https://github.com/tommccoy1/hans) | `7299f6f657089ce06a0f98e7e81f8d0f5b7741ce` | [MIT](https://github.com/tommccoy1/hans/blob/master/LICENSE.md) |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9` | [CC BY 4.0](https://stanfordnlp.github.io/contract-nli/LICENSE) |

请引用各基准原论文。冻结联合训练输入保留了实际使用的基准正文，位于`source-data.zip`中的`datasets/semantic-text/`；其它诊断按来源配置从官方重取。公开副本保留上表归属与许可链接，ANLI和OCNLI的NC条件不因混合或重排消失。结果中的ID、官方标签、模型预测和聚合指标不把任务改成“真人记忆金标”。不能将官方无gold的test行当作错误，也不能把参考生成方式不同的HANS与人工标注无区别合并。

## 方法来源与改动声明

- RankNet使用成对偏好学习；本仓提供实际特征、采样、损失、导出和评测配方，不声称发明学习排序。
- System-One式类型化选项打分和采样频率蒸馏是方法借鉴，不是官方Jev模型的权重或训练复现。
- 标签粒度研究参考Chen等的[Understanding the Impact of Label Granularity on CNN-based Image Classification](https://arxiv.org/abs/1901.07012)（v1）、Pirovano等的[The Advantage of Fine-Grained Training](https://arxiv.org/html/2509.05130v2)（v2），以及[GEORGE: No Subclass Left Behind](https://proceedings.neurips.cc/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)。借鉴分别是同粗任务比较、边界对齐与弱子群切片，未把图像研究结果当成文本任务定律。
- 记忆存储／阅读粒度参考[LongMemEval](https://arxiv.org/html/2410.10813v1)、[A-MEM](https://arxiv.org/abs/2502.12110)与[Molecular Facts](https://arxiv.org/abs/2406.20079)。它们用于提出表示、语境与披露路径的可检验假设，不替代本项目的实证结果。
- 公开副本移除了凭据、用户特定路径、运维信息与不必要过程注释；转换清单同时保留原件与公开件SHA。训练数据未因结果好坏重标；公开迁移也不产生新的训练成绩。

外部模型API的请求别名、回显、时间和参数只能证明所记录调用，不能独立认证服务端隐藏权重。费用是相应实验的usage与公开单价估算，不是整个项目账单或未来价格承诺。
