"""biodata 领域包：公开单细胞数据集目录（本项目的默认领域与骨架能力测试集）。

- `vocabulary.py`：受控词表 CATALOG + 平台/assay/modality 派生规则（自 `retrieval/vocabulary.py` 迁入）。
- `dimensions.py`：6 个检索维度的描述符（逐项复刻旧 retriever 写死 if 链的行为）。
- `pack.py`：`get_pack()` 装配 DomainPack（提示词目录指向 repo 资源树 `prompts/domains/biodata/`）。
"""
