                       
"""出处事实核 —— 「我们**凭什么**这么说」的单一真源。确定性、离线、只读。

## 为什么需要这个模块

`fair.py` 生成的文本**会被用户粘进论文**。同一个字段值，在检索器里和在稿件里，可承受的错误方向
是**相反的**：

- 检索过滤器：`has_raw_data=False` 宁可漏标不可错报 —— 代价只是少召回一条，保住「FASTQ 违规 0」。
- 稿件陈述句：同一个 `False` 一旦变成 `"Raw sequencing data are not available for this dataset."`，
  就是一句**审稿人一查即穿的假话**。

 验证：`ingest_arrayexpress.py:214` 的注释原文是「保守：不逐条核实 FASTQ，宁可漏不可错报」，
即 1786 条 `False` 是**猜的**；而 DAS 把它们连同另外 2844 条一起印成事实（全库 4630 条 = 81.7%），
`fair.assess_fair` 的 R2 证据栏更写着「来源明确标注无原始 FASTQ」—— **来源从未标注过**。
这与 修掉的「298 条 disease='unknown' 被当已核验」是同一个 bug 家族：
**把「我们不知道」编码成「它没有」**，只不过这次的产物进的是论文。

## 三条不变量（有 pytest 钉死，不是叮嘱）

1. **不存在无作用域的否定**。任何 `state != "listed"` 必须携带非空 `scope`——说清是「在哪份清单里
   没有」，而不是「它不存在」。`raw_data_provenance` 对空 scope 直接 raise，不给静默降级的口子。
2. **`public_accession` 返回的字符串必须是该记录 `url` 的子串**。实测：4900 条 external 剥掉内部
   前缀后 4900/4900 命中自身 url。这条断言把「拿内部主键冒充公开编号」这类错误钉死在结构上。
3. **`as_of` 只能来自真实抓取戳**（`inspection.snapshot_info()['snapshot_date']`）。取不到就是 `None`，
   **绝不拿 `published_date` 冒充**——那是数据集的发表日，不是我们的核验日。`as_of=None` 时消费方
   禁止渲染任何日期。

## 边界

本模块**只被** `fair.py` / `reuse_pack.py` / `webapp.py` / `mcp_server.py` 消费；
`retriever` / `workflow` / `evaluate_recommendation` **从不 import 它**（结构性隔离，`fair.py` 是范式）
→ 冻结 767 基准逐位不受影响。只读元数据，不联网、不写盘、不调 LLM。
"""
from __future__ import annotations

from typing import Any

from . import downloads, inspection

                                                                       

SOURCE_10X = "10x Genomics"
SOURCE_CELLXGENE = "CELLxGENE Discover"
SOURCE_ARRAYEXPRESS = "ArrayExpress"
SOURCE_HCA = "Human Cell Atlas"
SOURCE_SCEA = "EBI Single Cell Expression Atlas"
SOURCE_ENCODE = "ENCODE"
                                                              
SOURCE_HUBMAP = "HuBMAP"
SOURCE_SCP = "Broad Single Cell Portal"
SOURCE_GEO = "NCBI GEO"
                                                  
                                                                          
SOURCE_ZENODO = "Zenodo"
SOURCE_REFINEBIO = "refine.bio"

                                                                            

                               
LISTED = "listed"
                                                    
                                              
NOT_LISTED = "not_listed"
                                         
NOT_CHECKED = "not_checked"

                                              
BASIS_LEDGER_PROBE = "ledger_probe"                         
BASIS_SOURCE_MANIFEST = "source_manifest"                   
BASIS_REPOSITORY_POLICY = "repository_policy"                             
BASIS_UNVERIFIED = "unverified"                

                                                 
  
                                                                         
                                                                 
                                                                                 
                                                                               
                                                           
_PREFIX_IS_PUBLIC_ACCESSION = {
    "ae": True,
    "ebi": True,
    "cxg": False,
    "hca": False,
    "encode": True,                                                  
                                                     
                                                          
                                                 
                                                    
    "geo": True,                                                    
    "zenodo": True,                                                        
    "refinebio": True,                                                             
    "scp": True,                                                              
    "hubmap": False,                                                              
}

                                                                           
 
                                             
                                           
 
                                                                               
                                                                                    
                                                                  
                                                                                                
                                                                             
                                                              
                                                                             

                                                 
_PREFIX_IS_PLATFORM_ID = {
    "cxg": True,
    "hca": True,
    "ae": False,                                                   
    "ebi": False,
    "encode": False,
                                                     
                                                     
                                                                      
    "hubmap": True,
    "geo": False,
    "zenodo": False,
    "refinebio": False,
    "scp": False,
}


def _text(value: object) -> str:
    return str(value or "").strip()


def english_source_name(source: object) -> str:
    """英文语境里可以安全使用的来源名。

    五个公开来源的名字本来就是 ASCII（`10x Genomics` / `CELLxGENE Discover` …），
    但**用户上传的记录来源名可能是中文**（如「用户上传」）。把它直接拼进英文声明会产出
    中英混排的句子，而这句话是要被粘进稿件的 → 非 ASCII 一律退回中性说法。

    注意**只**约束来源名这类由我们拼接的成分。`dataset_name` 不走这里：数据集真叫中文名时，
    英文声明里出现它是**对的**——那是它的名字，不是我们漏进去的样板文案。
    """
    text = _text(source)
    return text if (text and text.isascii()) else "the source repository"


def _finalize(verdict: dict[str, Any]) -> dict[str, Any]:
    """不变量 1 的**结构性**闸门：所有出处判定都必须从这里出去。

    「不存在无作用域的否定」如果只靠 pytest 守，那它守的是**今天写的那几个分支**；
    以后谁加一个新来源、复制粘贴一个 return，测试照样绿。放在唯一出口上 raise，
    才是「不给静默降级的口子」——新分支忘了写 scope，第一次调用就炸，而不是
    悄悄印一句无作用域的否定进别人的稿件。

    **双语是硬要求，不是锦上添花**：`scope` 进中文界面，`scope_en` 进**英文稿件**。
     的测试当场抓到过一句混血产物——
    `"Raw sequencing data (FASTQ) are listed in the 10x Genomics 官方下载页的文件清单."`
    ——它会被原样粘进论文。所以两个都必须非空，缺一个就炸。

    这里不区分肯定/否定：LISTED 的措辞（"listed in the {scope}"）同样要靠作用域才成立，
    统一要求比「否定才需要」少一个口子。
    """
    for key in ("scope", "scope_en"):
        if not _text(verdict.get(key)):
            raise ValueError(
                f"出处判定 state={verdict.get('state')!r} 缺少作用域（{key}）。"
                "断言必须说清「在哪份清单里」，否则「没列出」会被读成「它不存在」；"
                "scope_en 缺失则会把中文漏进英文稿件。"
            )
    if not _text(verdict["scope_en"]).isascii():
        raise ValueError(
            f"scope_en={verdict['scope_en']!r} 含非 ASCII 字符。它会被拼进**英文稿件**，"
            "必须是纯英文样板文案；来源名等外来成分请先过 english_source_name()。"
        )
    return verdict


def snapshot_as_of() -> str | None:
    """本工具最近一次真实抓取/核验的时间戳；取不到 → None（**绝不用别的日期冒充**）。"""
    try:
        info = inspection.snapshot_info()
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    return _text(info.get("snapshot_date")) or None


def public_accession(dataset_uid: object, source: object = "") -> str | None:
    """该数据集**可以写进稿件**的公开编号；没有就返回 None（**不编**）。

    验证（全库 5667）：external 4900 条 uid 一律带内部前缀
    `cxg:`(2198) / `ae:`(1786) / `hca:`(532) / `ebi:`(384)，base 767 条无前缀但是 **URL slug**
    （如 `multiome-gemx-5k-mouse-kidney`），也不是 accession。

    修复前 `fair.py` 无条件写 `under accession "{uid}"`，于是 `/api/fair` 真实吐出过：
      - `under accession "cxg:24921392-22ed-479a-9144-7d40adf148ae"`  ← 内部 UUID，根本不是编号
      - `under accession "ae:E-MTAB-11814"`                            ← 真编号被内部前缀污染
      - `under accession "multiome-gemx-5k-mouse-kidney"`              ← URL slug
    **5667/5667 = 100% 的记录都被印上了一个编造的 accession。**

    返回非 None 时保证是 `record.url` 的子串（不变量 2，有测试钉死）。
    """
    uid = _text(dataset_uid)
    if not uid or ":" not in uid:
                                                              
                                 
        return None
    prefix, _, bare = uid.partition(":")
    if not bare:
        return None
    if not _PREFIX_IS_PUBLIC_ACCESSION.get(prefix.lower(), False):
        return None
    return bare


def platform_id(dataset_uid: object, source: object = "") -> str | None:
    """平台内部但**稳定公开**的定位符（CELLxGENE / HCA 的 UUID）；没有 → None。

    它可以写进稿件（"in CELLxGENE Discover as dataset <UUID>"），但**不能**写成
    "under accession <UUID>" —— 上游不用 accession 这套编号法（见 `_PREFIX_IS_PUBLIC_ACCESSION`）。
    与 `public_accession` 互斥：一条记录不会两者都有。
    """
    uid = _text(dataset_uid)
    if not uid or ":" not in uid:
        return None
    prefix, _, bare = uid.partition(":")
    if not bare:
        return None
    return bare if _PREFIX_IS_PLATFORM_ID.get(prefix.lower(), False) else None


def collection_doi(item: dict[str, Any]) -> str | None:
    """产出该数据集的那篇**论文**的 DOI；没有 → None。

    **粒度警告（决定了它的措辞）**：本库的 `collection_doi` 是 **collection 级**的，指向
    产出这批数据的**论文**，不是 dataset 级的数据引用。实测 2155 条有值、100% 来自 CELLxGENE
    （2155/2198 = 98%），其余四源全 0。

    所以它只能被表述为「关联发表物 / associated with collection DOI」，
    **绝不能**当成「引用本数据集就用这个 DOI」—— 那是引错文献粒度：一个 collection 里可能有
    几十个 dataset，而读者会以为这个 DOI 唯一指向你用的那一个。

    此前该字段是**零消费点的死字段**（`src/` 与 `web/` 全无引用）。这里给它接上消费者，
    但只用它承担它真正能承担的语义。

    Zenodo 适配器把**数据记录自己的 DOI**（conceptdoi/doi/pids）写进该原始字段
    （供编号直查/标识符匹配用）——它不是论文 DOI，本访问器对 Zenodo 来源一律回 None，
    绝不把数据 DOI 谎称为「关联论文 DOI」；数据 DOI 经 `dataset_doi()` 按正确对象类型读取。
    """
    if _text(item.get("source")) == SOURCE_ZENODO:
        return None
    doi = _text(item.get("collection_doi"))
    if not doi or doi.lower() in {"unknown", "n/a", "na", "none", "null", "-"}:
        return None
    return doi


def dataset_doi(item: dict[str, Any]) -> str | None:
    """**数据记录自己的 DOI**（dataset/collection-deposit 级）；不是论文 DOI。

    数据 DOI 的类型化出口：Zenodo 的 conceptdoi/doi 指向数据记录（conceptdoi = 全部版本的
    最新版，版本粒度更稳），存在原始记录的 `collection_doi` 字段与新增 `dataset_doi` 字段里。
    消费侧要展示/导出时用本访问器，并如实标注为「数据集 DOI」；论文 DOI 走
    `collection_doi()`——两者不允许再混用同一句话。"""
    if _text(item.get("source")) != SOURCE_ZENODO:
        return None
    doi = _text(item.get("dataset_doi")) or _text(item.get("collection_doi"))
    if not doi or doi.lower() in {"unknown", "n/a", "na", "none", "null", "-"}:
        return None
    return doi


def raw_data_provenance(item: dict[str, Any]) -> dict[str, Any]:
    """这条记录的**原始测序数据可及性**，我们到底知道什么、凭什么知道。

    入参是 `webapp._web_item_from_record` 构造的 item（`fair.py` 拿到的就是它），
    需要 `source` / `dataset_uid` / `has_raw_data`。

    返回 `{state, scope, basis, evidence, as_of, elsewhere_hint}`：
      - `state`：LISTED / NOT_LISTED / NOT_CHECKED
      - `scope`：**否定断言的作用域**，非空（不变量 1）。「在 X 里没列出」≠「不存在」。
      - `basis`：证据等级，见 BASIS_* 常量
      - `evidence`：中文人读说明
      - `as_of`：核验时间戳；只有 ledger_probe 档才有，其余 None
      - `elsewhere_hint`：来源自己声明的「原始数据在别处」线索（目前只有 SCEA 有）

    逐源依据（全部来自读 ingest 脚本原文，非推测）：
      - **10x base**：`normalizer.py:243-249` 用活台账实测值 `downloads.has_fastq` 覆盖索引值
        （注释：「索引对 99 条过度声称 has_raw_data，用更正信号覆盖」）→ 台账真的逐条 Range-GET 验过 →
        `ledger_probe`。注意 NOT_LISTED 仍只是「10x 下载页没挂 FASTQ」，普通 Chromium scRNA 的
        raw reads 大概率存在于 GEO/SRA —— 所以**永远不写「不存在」**。
      - **HCA**：`ingest_hca.py:122` `has_raw_data=_has_fastq(hit)`，读的是 DCP 的 `fileTypeSummaries`
        （逐条读过来源清单）→ `source_manifest`。
      - **CELLxGENE**：`ingest_cellxgene.py:11` 「据实标 False：CELLxGENE 资产是处理后的 H5AD/RDS 矩阵」
        → 这是**库级事实**、真，但非逐条核验 → `repository_policy`。
      - **EBI SCEA**：`ingest_ebi_scea.py:28` 「据实标 False；原始 FASTQ 在 ENA/ArrayExpress（accession 可溯源）」
        → 同为库级事实，**且来源自己说了原始数据在哪** → `repository_policy` + `elsewhere_hint`。
      - **ArrayExpress**：`ingest_arrayexpress.py:19` 「**保守置 False**：不逐条列文件确认 FASTQ 存在，
        宁可漏标也不错标」→ 这 1786 条 False **是猜的** → `NOT_CHECKED`。**这是本模块存在的首要理由。**

    逐记录证据优先：在逐源默认之前，`_per_record_raw_verdict` 先检查记录自带的
    `metadata_provenance`。只认四种严格形状——ArrayExpress（complete + fields.files /
    fields.has_raw_data 均 complete）、SCEA（cross_source_enrichment.file_evidence.complete）、
    HCA（complete + raw_data.complete）、ENCODE（complete + files.complete）——形状不认或
    complete 缺失一律返回 None、落回来源级默认，语义不变。证据是本工具逐条/逐实验核验的
    wire 抓取，故 basis 一律 `source_manifest`，as_of 只取记录内真实 retrieved_at，
    取不到就留 None（ENCODE 快照无抓取戳，恒为 None）。
    """
    return _finalize(_raw_provenance_impl(item))


def _nonempty_str(value: object) -> str | None:
    """非空 str 原样返回，其余一律 None（as_of 只认真实抓取戳，不拿别的值冒充）。"""
    return value if isinstance(value, str) and value.strip() else None


def _per_record_raw_verdict(item: dict[str, Any]) -> dict[str, Any] | None:
    """逐记录证据优先入口。返回 None = 无可用逐记录证据，落回来源级默认。

    严格按形状判定：`metadata_provenance` 非 dict、complete 缺失/非 True、或不是下面四种
    形状之一，都返回 None —— 宁可落回保守的来源级默认，也不拿形状不明的字段当证据。
    """
    source = _text(item.get("source"))
    mp = item.get("metadata_provenance")
    if not isinstance(mp, dict):
        return None
    has_raw = item.get("has_raw_data")

                                                    
    if source == SOURCE_ARRAYEXPRESS:
        if mp.get("complete") is not True:
            return None
        fields = mp.get("fields")
        if not isinstance(fields, dict):
            return None
        if not (isinstance(fields.get("files"), dict) and fields["files"].get("complete") is True):
            return None
        if not (isinstance(fields.get("has_raw_data"), dict) and fields["has_raw_data"].get("complete") is True):
            return None
        scope = "ArrayExpress/BioStudies 记录页文件清单（本工具逐条核验）"
        scope_en = "the ArrayExpress/BioStudies record file listing (verified record-by-record by this tool)"
        as_of = _nonempty_str(mp.get("retrieved_at"))
        if has_raw is True:
            return {
                "state": LISTED,
                "scope": scope,
                "scope_en": scope_en,
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "本工具逐条读取该记录的 BioStudies 文件清单，其中列出原始测序文件。",
                "as_of": as_of,
                "elsewhere_hint": None,
            }
        return {
            "state": NOT_LISTED,
            "scope": scope,
            "scope_en": scope_en,
            "basis": BASIS_SOURCE_MANIFEST,
            "evidence": "本工具逐条读取该记录的 BioStudies 文件清单，其中未列出原始测序文件。这不代表原始数据不存在。",
            "as_of": as_of,
            "elsewhere_hint": "ArrayExpress 的 NGS 原始序列通常存放于 ENA 并由该记录提供链接，请到来源页面核实。",
        }

                                                       
    if source == SOURCE_SCEA:
        enrichment = mp.get("cross_source_enrichment")
        if not isinstance(enrichment, dict):
            return None
        file_evidence = enrichment.get("file_evidence")
        if not isinstance(file_evidence, dict) or file_evidence.get("complete") is not True:
            return None
        doc = enrichment.get("biostudies_document")
        as_of = _nonempty_str(doc.get("retrieved_at")) if isinstance(doc, dict) else None
        scope = "BioStudies 研究文档与关联 ENA 文件报告（本工具逐条核验）"
        scope_en = "the BioStudies study document and linked ENA file reports (verified record-by-record by this tool)"
        accessions = item.get("ena_study_accessions")
        ena = accessions[0] if isinstance(accessions, list) and accessions else None
        if has_raw is True:
            return {
                "state": LISTED,
                "scope": scope,
                "scope_en": scope_en,
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "本工具逐条读取该记录的 BioStudies 研究文档与关联 ENA 文件报告，ENA 文件报告中列出原始测序文件。",
                "as_of": as_of,
                "elsewhere_hint": None,
            }
        return {
            "state": NOT_LISTED,
            "scope": scope,
            "scope_en": scope_en,
            "basis": BASIS_SOURCE_MANIFEST,
            "evidence": "本工具逐条读取该记录的 BioStudies 研究文档与关联 ENA 文件报告，ENA 文件报告中未列出原始测序文件。这不代表原始数据不存在。",
            "as_of": as_of,
            "elsewhere_hint": (f"该记录关联 ENA study {ena}，可经该编号到 ENA 核实原始数据。" if ena else None),
        }

                                 
    if source == SOURCE_HCA:
        if mp.get("complete") is not True:
            return None
        raw_data = mp.get("raw_data")
        if not isinstance(raw_data, dict) or raw_data.get("complete") is not True:
            return None
        value = raw_data.get("value")
        if not (value is True and has_raw is True) and value is not False:
                                                                 
            return None
        scope = "Human Cell Atlas Data Coordination Platform 的文件清单（本工具逐条核验）"
        scope_en = "the Human Cell Atlas Data Coordination Platform file manifest (verified record-by-record by this tool)"
        as_of = _nonempty_str(mp.get("retrieved_at"))
        if value is True and has_raw is True:
            return {
                "state": LISTED,
                "scope": scope,
                "scope_en": scope_en,
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "本工具逐条读取该 project 的 HCA DCP 文件清单，其中实测到 fastq/fastq.gz 文件元数据。",
                "as_of": as_of,
                "elsewhere_hint": None,
            }
        return {
            "state": NOT_LISTED,
            "scope": scope,
            "scope_en": scope_en,
            "basis": BASIS_SOURCE_MANIFEST,
            "evidence": "本工具逐条读取该 project 的 HCA DCP 文件清单，其中未实测到 fastq 文件元数据。这不代表原始数据不存在。",
            "as_of": as_of,
            "elsewhere_hint": "原始测序数据（若已公开）可能存于其它归档库，需自行核实。",
        }

                                     
    if source == SOURCE_ENCODE:
        if mp.get("complete") is not True:
            return None
        files = mp.get("files")
        if not isinstance(files, dict) or files.get("complete") is not True:
            return None
        if has_raw is None:
            return None
        scope = "ENCODE 门户实验文件清单（本工具逐实验核验）"
        scope_en = "the ENCODE portal experiment file listing (verified per experiment by this tool)"
        if has_raw is True:
            return {
                "state": LISTED,
                "scope": scope,
                "scope_en": scope_en,
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "本工具逐实验核验 ENCODE 门户文件清单，其中列出 FASTQ 原始测序文件。",
                "as_of": None,                      
                "elsewhere_hint": None,
            }
        return {
            "state": NOT_LISTED,
            "scope": scope,
            "scope_en": scope_en,
            "basis": BASIS_SOURCE_MANIFEST,
            "evidence": "本工具逐实验核验 ENCODE 门户文件清单，其中未列出 FASTQ。这不代表原始数据不存在。",
            "as_of": None,
            "elsewhere_hint": None,
        }

    return None


def _raw_provenance_impl(item: dict[str, Any]) -> dict[str, Any]:
                                                            
                                        
    verdict = _per_record_raw_verdict(item)
    if verdict is not None:
        return verdict

    source = _text(item.get("source")) or SOURCE_10X
    uid = _text(item.get("dataset_uid"))
    has_raw = item.get("has_raw_data")

                                                        
    if source == SOURCE_ARRAYEXPRESS:
        return {
            "state": NOT_CHECKED,
            "scope": "ArrayExpress（本工具抓取时未逐条核实文件清单）",
            "scope_en": "ArrayExpress record (file listing not individually verified by this tool)",
            "basis": BASIS_UNVERIFIED,
            "evidence": "本工具未核验该数据集的原始数据可及性；索引里的「无 FASTQ」是抓取时的保守占位，不是来源的声明。",
            "as_of": None,
                                                       
                                                              
                                                           
            "elsewhere_hint": "ArrayExpress 的 NGS 原始序列通常存放于 ENA 并由该记录提供链接，请到来源页面核实。",
        }

                                                   
    if source == SOURCE_CELLXGENE:
        return {
            "state": NOT_LISTED,
            "scope": "CELLxGENE Discover（该库按设计只分发处理后表达矩阵）",
            "scope_en": "CELLxGENE Discover, which by design distributes processed expression matrices only",
            "basis": BASIS_REPOSITORY_POLICY,
            "evidence": "CELLxGENE Discover 分发的是处理后的 H5AD/RDS 矩阵，不托管原始 reads。这是库级事实，非逐条核验。",
            "as_of": None,
            "elsewhere_hint": "原始测序数据（若已公开）通常存于 GEO / ENA / SRA 等归档库，需自行核实。",
        }
    if source == SOURCE_SCEA:
        return {
            "state": NOT_LISTED,
            "scope": "EBI Single Cell Expression Atlas（该库只托管处理后表达矩阵）",
            "scope_en": "the EBI Single Cell Expression Atlas, which hosts processed expression matrices only",
            "basis": BASIS_REPOSITORY_POLICY,
            "evidence": "EBI SCEA 托管的是处理后表达矩阵，不托管原始 reads。这是库级事实，非逐条核验。",
            "as_of": None,
                                                          
            "elsewhere_hint": "该来源声明原始 FASTQ 存于 ENA / ArrayExpress，可经 accession 溯源。",
        }

                                               
    if source == SOURCE_HCA:
        if has_raw is True:
            return {
                "state": LISTED,
                "scope": "Human Cell Atlas Data Coordination Platform 的文件类型清单",
                "scope_en": "the Human Cell Atlas Data Coordination Platform file type summary",
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "HCA DCP 的 fileTypeSummaries 声明该 project 含 fastq/fastq.gz 文件。",
                "as_of": None,
                "elsewhere_hint": None,
            }
        if has_raw is False:
            return {
                "state": NOT_LISTED,
                "scope": "Human Cell Atlas Data Coordination Platform 的文件类型清单",
                "scope_en": "the Human Cell Atlas Data Coordination Platform file type summary",
                "basis": BASIS_SOURCE_MANIFEST,
                "evidence": "HCA DCP 的 fileTypeSummaries 未声明 fastq 文件。",
                "as_of": None,
                "elsewhere_hint": "原始测序数据（若已公开）可能存于其它归档库，需自行核实。",
            }
        return _unchecked(source)

                                            
    ledger = downloads.has_fastq(uid) if uid else None
    if ledger is True:
        return {
            "state": LISTED,
            "scope": "10x Genomics 官方下载页的文件清单（本工具逐文件实测）",
            "scope_en": "the 10x Genomics dataset download page file listing",
            "basis": BASIS_LEDGER_PROBE,
            "evidence": "本工具的文件级台账在该数据集下实测到 FASTQ 原始 reads 直链。",
            "as_of": snapshot_as_of(),
            "elsewhere_hint": None,
        }
    if ledger is False:
        return {
            "state": NOT_LISTED,
            "scope": "10x Genomics 官方下载页的文件清单（本工具逐文件实测）",
            "scope_en": "the 10x Genomics dataset download page file listing",
            "basis": BASIS_LEDGER_PROBE,
                                                      
                                                           
            "evidence": "本工具实测该数据集的 10x 下载页未挂出 FASTQ 直链。这不代表原始数据不存在。",
            "as_of": snapshot_as_of(),
            "elsewhere_hint": "部分 10x 官方数据集的原始 reads 另存于 SRA/GEO；成像类平台（如 Xenium）则天然无 FASTQ。",
        }
    return _unchecked(source)


def _unchecked(source: str) -> dict[str, Any]:
    return {
        "state": NOT_CHECKED,
        "scope": f"{source}（本工具未核验）",
                                                         
                                        
        "scope_en": f"{english_source_name(source)} (not verified by this tool)",
        "basis": BASIS_UNVERIFIED,
        "evidence": "本工具未核验该数据集的原始数据可及性。",
        "as_of": None,
        "elsewhere_hint": None,
    }


def size_bytes_or_none(item: dict[str, Any]) -> int | None:
    """来源声明的数据体积（字节）；未知 → None。

    实测：`filesize` 键在 4900 条 external 上存在，但只有 CELLxGENE 2198 条是真值，
    另外 2702 条（ArrayExpress 1786 + HCA 532 + SCEA 384）**硬编码 0** —— 把「未知」
    编码成「0 字节」。任何直接渲染 filesize 的产物都在这 2702 条上撒谎，
    所以这里把 `<=0` 一律收敛成 None（「不知道」），由消费方显式显示为未知。
    """
    raw = item.get("filesize")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None
