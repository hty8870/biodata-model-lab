                       
"""条件板：把「再说一句话改条件」变成一次**可预览、可撤销、绝不静默乱改**的条件编辑。

## 这是什么

搜出结果之后，用户想接着说「换成小鼠」「再加一条：要有 FASTQ 的」「去掉组织限制」。
本模块把这句话规划成**一次具体的改动**，回答三件事：改什么、改完的句子长什么样、有没有连带改到别的地方。
它**只规划、不执行**：不检索、不装载数据集目录、不落盘、不写日志、不联网、不用大模型。
真正的检索仍由调用方拿着 `next_request` 去走既有 `/api/recommend`。

## 三条撑起整个设计的判断

1. **「现在按什么在筛」的唯一真源是上一次检索的返回**（`current_filters` = 那次响应的 `active_filters`）。
   本模块**绝不**用自己解析出来的结果去回答这个问题。原因是四条链路会让两者错位：
   来源专名在 `parse_query` **之前**就被 `resolve_search_request` 整段剥掉；网页上的年份下拉框根本不在句子里；
   部署方通过 `KEYWORD_MAPPING_PATH` 注入的说法本模块的调用方不一定给；开了关键词审核时产生当前这批结果的
   是系统改写后的句子而不是输入框里那句。自己重新解析的结果会凭空多出一条「物种：Human」
   （「人类细胞图谱」里的「人类」），然后在改写时把这个来源专名打碎、静默换掉检索范围。

2. **来源范围是一个事后校验看不见的东西**。`QueryIntent` 里没有来源这一项，所以「检索范围被换掉」
   在结构上不可能被条件比对抓到。三道防线：来源专名整段设为保护区并**等长遮蔽**后再定位；
   任何编辑碰到保护区 → 拒绝；候选句重算一次来源专名，与原句不一致 → 拒绝。

3. **只动开关的编辑才允许直接执行**（去掉某条 / 放宽某条，都一键可逆）；
   **凡是改到句子文字的，一律先给预览、等用户点确认**。事后校验只能证明条件结构一致，
   证明不了这就是用户的意思。

## 刻意不做的事

- 不把「换成 / 去掉 / 放宽」这些说话方式词加进 `vocabulary` 的填充词表：那张表在冻结评测的热路径上
  （参与未识别实义词的判定）。它们是本模块的私有常量，在句子进 `parse_query` **之前**就被剥掉——
  尤其「去掉 / 去除」本身就是可执行否定前缀，不先剥掉，「去掉组织限制」会被当成「排除组织」。
- 不开「直接注入约束」的入参口子：正向约束的唯一载体是查询文字，绕过 `parse_query` 就绕过了
  它全部的 fail-closed 门（否定白名单、或/也许弃权、未收录实义词残差门、极性冲突、非法日历日）。
- 不用大模型。规则法覆盖不了所有口语，这是本设计承认的代价；代价由「每一行右边都有按钮」补上。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from ..retrieval.query_parser import (
    PS_EXECUTABLE,
    QueryIntent,
                                            
                                               
    _alias_occurrences,
    _merged_catalog,
    active_filters,
    annotate_query_spans,
                                          
                                                                       
    detect_action_markers as _detect_action_markers,
    detect_action_verbs as _detect_action_verbs,
    parse_query,
)
from ..retrieval.search_request import mask_source_spans, source_alias_spans
from ..retrieval.query_lexicon import (
    NEG_EXISTENTIAL_PHRASES,
    NEG_MORPHEMES_CN,
)
from ..domain.registry import domain_dimensions, domain_labels_zh, get_domain
from ..domain.vocab_tools import suggestable_terms
from .workflow import (
    apply_explicit_date_range,
    apply_suppressed_constraints,
    sanitize_facet_filters,
    sanitize_lenient_dims,
    sanitize_suppressed,
)
                                                    
                                                           
from ..agent.action_plan import MAX_UTTERANCE_CHARS
from .request_validation import MAX_QUERY_CHARS

BOARD_SCHEMA = "biodata-board/v1"

                                                                 
ST_AUTO = "auto_apply"
ST_CONFIRM = "needs_confirm"
ST_CHOICE = "needs_choice"
ST_SUGGEST = "suggest"
ST_UNKNOWN = "not_understood"
ST_REJECTED = "rejected"

OPS = ("add", "replace", "widen", "remove", "lenient", "restart", "suggest")


class BoardError(Exception):
    """入参非法。带 `code` 供 HTTP / MCP 层映射成稳定错误码。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


                                                                               
 
                                                        

_LEAD_HEADS = ("把", "将", "对", "请", "帮我", "麻烦", "能不能", "可以", "我要", "我想")
_LEAD_PUNCT = "：:，,。.、；;！!？?　 \t"

_OP_REPLACE = ("换成", "改成", "改为", "换为", "替换成", "替换为", "改用", "换到", "换做", "换")
_OP_ADD = ("再加一条", "再加上", "再加", "加一条", "加上", "另外", "还要", "再要", "同时", "并且", "而且", "补充一条")
_OP_REMOVE = ("去掉", "去除", "取消", "撤掉", "删掉", "解除", "不限制", "别管", "不按", "不筛", "不要按")
                                                
                                                     
_OP_LENIENT = ("放宽", "宽松", "没标注的也算", "未标注的也算", "缺这项的也算", "没填的也算", "不限")
_OP_RESTART = ("重新开始", "重新搜", "重新检索", "重来", "重新找", "换个问题")

_OP_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("replace", _OP_REPLACE),
    ("add", _OP_ADD),
    ("remove", _OP_REMOVE),
    ("lenient", _OP_LENIENT),
    ("restart", _OP_RESTART),
)
_FAMILY_NAME_CN = {
    "replace": "换成", "add": "再加", "remove": "去掉",
    "lenient": "放宽", "restart": "重新开始",
}

                                                           
_QUANT_HEAD = ("只要", "只需要", "只需", "仅要", "仅", "只", "必须", "一定要", "要求", "限定", "限", "要")

                                                   
                                                        
                                             
def _dim_nouns() -> tuple[tuple[str, str], ...]:
    pack = get_domain()
    if pack.dim_nouns:
        return pack.dim_nouns
    derived = tuple((d.label_zh, d.name) for d in pack.dimensions if d.label_zh)
    return derived + (
        ("原始数据", "has_raw_data"), ("发表时间", "date"), ("发表年份", "date"),
        ("时间", "date"), ("年份", "date"),
    )


_LABEL_RAW = "原始数据"
_LABEL_DATE = "发表时间"

                                                                                    
_NEG_MORPHEMES = NEG_MORPHEMES_CN
_EN_NEG_WORDS = re.compile(
    r"(?<![a-z0-9_])(no|not|non|without|exclude|excluding|except|avoid|omit|skip|neither|nor)(?![a-z])",
    re.IGNORECASE,
)


                                                                            
 
                                                     
                         

COPY: dict[str, str] = {
    "multi_value_note": "同一行里的几个值，满足其中一个就算符合。",
                                                                        
                                                               
                                              
                                                                             
    "volatile_note": "刷新页面会清空「回到上一步」的记步，条件本身不受影响。",
    "button_hint": "说不明白就直接点上面每条右边的按钮，功能是一样的。",
    "suggest_title": "我认识这些说法（能不能搜到，要搜了才知道）",

    "empty_utterance": "没有收到要改什么，条件一个字都没有改。",
    "not_understood": "这句话我没有听懂，条件一个字都没有改。",
    "conflict_ops": "这句话里同时出现了「{a}」和「{b}」，我不确定你要哪一种。",
    "no_payload": "这句话里没有说要改成什么，条件一个字都没有改。",
    "no_such_filter": "你现在没有按「{label}」筛，所以没有可以去掉的「{label}」条件。",
    "nothing_to_replace": (
        "你现在没有按「{label}」筛，所以没有可以替换的内容。"
        "如果是想加上这一条，可以说「再加一条：{after}」。"
    ),
    "override_applied": "这一步会按你改过的这句话检索。",
    "override_changed": "跟着变的条件：{changes}",
    "unknown_term": "「{term}」这个词我不认识，条件没有改动。",
    "unknown_term_no_mapping": (
        "我在本地的词表里没找到「{term}」，这不代表系统不支持——可以直接在搜索框里改写整句试试。"
    ),
    "cross_dim": "「{term}」里既有{a}也有{b}，你是想改哪一个？",
    "lenient_unsupported": "「{label}」这一项不支持「放宽」。如果想完全不按它筛，可以说「去掉{label}限制」。",
    "lenient_no_gap": "「{label}」这一项现在没有没填的数据集，放宽不会多出结果，所以这一步没有执行。",
                                         
                                                  
                     
    "lenient_coverage_unknown": (
        "已经把没标注「{label}」的数据集也算进来了。我这边没有拿到这一项有多少条没标注，"
        "所以说不出这一步会多出多少——以重新检索后的结果为准。"
    ),
    "source_in_utterance": "数据来源请在搜索框下面的「数据来源」里改，这句话没有改动任何条件。",
    "source_selfcheck_failed": "这句话里的数据来源我没能对上，为安全起见这一步没有执行。",
    "non_ascii_case_shift": "这句话里有我没法准确定位的字符，为安全起见这一步没有执行。",

    "applied_remove": "已经不按「{label}」筛了。这个条件还在，只是这次没用它，随时可以恢复。",
    "applied_lenient": "没标注「{label}」的数据集现在也算符合；标得明确又和你要的不一样的仍然不要。",
    "preview_replace": "这一步会把「{label}」从 {before} 换成 {after}。",
    "preview_add_new": "这一步会新增一条「{label}：{after}」。",
    "preview_widen": "这一步会让「{label}」变成 {before} 或 {after}，满足其中一个就算符合。",
    "preview_restart": "这一步会丢掉现在的全部条件（包括已忽略和已放宽的），只按「{payload}」重新检索。",
    "preview_removed_text": "会被去掉的原文：",
    "preview_final": "实际会拿去检索的句子（可以直接改）",
    "order_may_change": "筛选条件已经按你说的改了；结果的先后顺序可能和刚才略有不同。",
    "facet_cleared": "「在已筛出的结果里再缩小」的选择会一起清掉，因为要重新从全部数据集里挑。",
    "rewritten_basis": "当前这批结果来自系统改写后的句子，这次修改是在改写后的句子上做的。",

    "choice_prompt": "你现在的「{label}」是 {before}。「{after}」要怎么处理？",
    "choice_replace": "换成 {after}",
    "choice_widen": "{before}、{after} 都算（满足其一即可）",
    "choice_restart": "当成新的检索重新开始（会丢掉现在的全部条件）",

    "rejected": "按这句话改完之后，除了你说的那一处，还有别的地方也跟着变了，所以没有执行。条件保持原样。",
    "rejected_source": "这句话里点到了数据来源，改动会让检索范围跟着变，所以没有执行。",
    "rejected_unexecutable": "加上这句以后我没把握，所以这一步没做，上一步的条件和结果都保持原样。",
    "rejected_no_effect": "这句话没有真的改到任何条件，所以没有执行。",
}

_MISMATCH_ZH = {
    "unexpected_or": "这一条上多出了你没说过的取值",
    "collateral_change": "另一条条件也跟着变了",
    "no_effect": "改完之后条件没有任何变化",
    "lost_filter": "本该保留的条件不见了",
    "became_unexecutable": "改完之后这句话我没法执行",
    "source_scope_changed": "数据来源的范围跟着变了",
    "protected_region_touched": "改动碰到了数据来源专名",
}


                                                                      

def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_lead(text: str) -> str:
    """反复剥掉前导标点与话头（「把疾病放宽」的「把」、「请帮我换成小鼠」的「请帮我」）。"""
    out = str(text or "")
    changed = True
    while changed:
        changed = False
        stripped = out.lstrip(_LEAD_PUNCT)
        if stripped != out:
            out = stripped
            changed = True
        for head in sorted(_LEAD_HEADS, key=len, reverse=True):
            if out.startswith(head):
                out = out[len(head):]
                changed = True
                break
    return out.strip()


def _strip_quant(text: str) -> str:
    out = _strip_lead(text)
    changed = True
    while changed:
        changed = False
        for head in sorted(_QUANT_HEAD, key=len, reverse=True):
            if out.startswith(head):
                out = _strip_lead(out[len(head):])
                changed = True
                break
    return out.strip()


def _has_negation(text: str) -> bool:
    low = str(text or "").lower()
    return any(m in low for m in _NEG_MORPHEMES) or bool(_EN_NEG_WORDS.search(low))


def _dim_label(dim: str, polarity: str = "include") -> str:
    if dim == "has_raw_data":
        return _LABEL_RAW
    if dim == "date":
        return _LABEL_DATE
    base = domain_labels_zh().get(dim, dim)
    return ("排除·" + base) if polarity == "exclude" else base


def _label_for_filter_id(filter_id: str) -> str:
    """从筛选项编号还原中文条件名——与 `active_filters` 的标签规则同构，不依赖前端传旧快照。"""
    fid = str(filter_id or "")
                                                                      
                                                   
                                                          
    if fid.startswith("prefer:"):
        pdim = fid[len("prefer:"):]
        if pdim == "raw":
            return "优先·" + _LABEL_RAW
        if pdim == "source":
            return "优先·数据来源"
        return "优先·" + domain_labels_zh().get(pdim, _LABEL_DATE if pdim == "date" else pdim)
    if fid.startswith("raw:"):
        return _LABEL_RAW
    if fid.startswith("date:"):
        return _LABEL_DATE
    polarity, _, dim = fid.partition(":")
    return _dim_label(dim, "exclude" if polarity == "exclude" else "include")


def _context_window(text: str, start: int, end: int, pad: int = 4) -> str:
    """被中和的原文片段，前后各留几个字。

    单字别名（人 / 鼠）被中和时，只显示「被去掉的原文：人」用户完全判断不出发生了什么，
    必须带上下文才看得出自己的词被拆了。
    """
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


                                                                         

def parse_command(utterance: str, *, implicit_restrict: bool = False) -> dict:
    """把用户这句话拆成「要做哪种改动 + 改动的内容」。纯函数。

    两段式，顺序不能反：
      第一段（不锚定）先在整句上扫一遍五族说话方式词，**命中两族及以上直接判冲突**。
        长词优先消费，避免「不限制」被「不限」重复命中而假冲突。
      第二段才从句子里抹掉那个词、剥掉前导话头与量词头，剩下的就是改动内容。
        刻意不做「必须在句首」的限制：「把疾病放宽」「组织去掉」这类把说话方式词放在后面的说法非常常见。

    `implicit_restrict`（量词约束档）只在**对话上下文**（屏上已有结果）
    由调用方显式打开：「只要人类的」「只要含 FASTQ 的」是在当前条件上**收窄**的说法，
    不是一句新检索——统一路由此前把它当新检索短路易地，整批条件（组织/疾病…）被静默冲掉，
    正是「继续对话根本不可用」的根因。剥掉量词头后按 add 交给既有规划链：
    维度未设 → 新增一条；已设同值 → 如实「没有真的改到任何条件」；已设异值 → 三选一。
    两道保险：payload 里必须真有检索信号（否则「帮我处理一下数据库」剥掉话头也会被当成改动），
    且首屏检索框不传这个旗标（「只要 FASTQ 的人类数据」当首句仍是新检索）。
    """
    text = str(utterance or "").strip()
    if not text:
        return {"op": "", "payload": "", "reason": "empty", "detail": COPY["empty_utterance"]}

    tokens: list[tuple[str, str]] = []
    for family, words in _OP_FAMILIES:
        for word in words:
            tokens.append((word, family))
    tokens.sort(key=lambda t: len(t[0]), reverse=True)

    working = text
    hit_families: list[str] = []
    hit_token: dict[str, str] = {}
    for word, family in tokens:
        idx = working.find(word)
        if idx < 0:
            continue
        if family not in hit_families:
            hit_families.append(family)
            hit_token[family] = word
        working = working[:idx] + " " * len(word) + working[idx + len(word):]

    if not hit_families:
        if implicit_restrict:
                                                               
                                                                
            stripped = _strip_quant(text)
            if stripped and stripped != text:
                try:
                    has_content = bool(active_filters(parse_query(stripped)))
                except Exception:
                    has_content = False                              
                if has_content:
                    return {"op": "add", "payload": stripped, "reason": "", "detail": "", "operator": "只"}
        return {"op": "", "payload": "", "reason": "no_operator", "detail": COPY["not_understood"]}
    if len(hit_families) >= 2:
        first, second = hit_families[0], hit_families[1]
        return {
            "op": "", "payload": "", "reason": "conflicting_operators",
            "detail": COPY["conflict_ops"].format(
                a=_FAMILY_NAME_CN.get(first, first), b=_FAMILY_NAME_CN.get(second, second)
            ),
        }

    family = hit_families[0]
    payload = _strip_quant(_norm_ws(working))
    return {"op": family, "payload": payload, "reason": "", "detail": "", "operator": hit_token[family]}


def classify_utterance(utterance: str) -> str:
    """这句话**不是**改条件的话时，它到底想干什么。纯函数、只读、不联网。

    动机：「继续对话」此前只认「改条件」这一种说法，其余一律回
    「没听懂 + 一排改条件的出路」。可用户在第一个搜索框里能说的远不止改条件——
    能说「帮我打包前20条」，能直接贴一个 GSE 编号，也能重打一整句新需求。
    于是同一个人在同一个页面上，换个输入框说同样的话就不work，这不是「功能少一点」，
    是**两个入口不是同一个产品**。

    路由三档（互斥，按优先级）：
      · ``identifier`` —— 话里点名了某个数据集编号/直链（`identifiers.classify` 是 `search`、
                          编号出现在句中任意位置都算）。复用与首个搜索框标识符直链反查同一套判据。
      · ``action``     —— 含执行**动作**说法（打包/下载脚本/导出…）。复用
                          `query_parser.detect_action_verbs`，与检索侧同一份词表，
                          不另抄一份（本项目在「两份手抄必漂移」上栽过多次）。
                          **只看动作词、不看对象词**：「清单/批量/脚本/引文」是产物的名字，
                          出现在句子里不代表用户在要求做这件事。
      · ``new_query``  —— 其余有实义内容的整句：当作一次全新的检索。

    **编号必须排在执行词前面**（修）：`identifiers.classify` 用的是 `search`，
    「把 E-MTAB-1234 打包」两档都命中。旧顺序判成 action → 在**当前屏上那批毫不相干的结果**上
    打开打包面板，用户点名的那一条根本不在里面。先按编号把那一条查出来，才谈得上打包它。
    这句话里的执行诉求由调用方据 ``action_markers`` 显式告知用户（不静默吞掉）。

    返回 ``""`` 表示「确实什么也不是」（空串/纯标点），此时保持原来的「没听懂」文案。
    **本函数不执行任何动作**，只回答「这句话该交给谁」；真正做事的仍是既有的那条管线。
    """
    from ..content.identifiers import classify as _classify_identifier

    text = str(utterance or "").strip()
    if not text:
        return ""
    if _classify_identifier(text):
        return "identifier"
                                                          
                                          
    if _detect_action_verbs(text):
        return "action"
                                            
    if any(ch.isalnum() or "一" <= ch <= "鿿" for ch in text):
        return "new_query"
    return ""


                                                                              

def _existential_blanked(text: str) -> str:
    """把存在性问句短语（「有没有 / 有无 / 是否有…」）等长抹掉后的文本。

    「库里有没有斑马鱼的」里的「没有」是**提问**的一部分、不是排除操作符——
    `NEG_EXISTENTIAL_PHRASES` 是否定语法的既有真源（解析热路径同一份），这里只读、不另抄。
    等长替换是本仓库的既有纪律：不让左右残字意外相邻拼出原本不存在的词。
    """
    low = str(text or "").lower()
    for phrase in sorted(NEG_EXISTENTIAL_PHRASES, key=len, reverse=True):
        low = low.replace(phrase, " " * len(phrase))
    return low


def _has_retrieval_signal(text: str, keyword_mapping: dict | None = None) -> bool:
    """这句话里有没有**检索信号**：活动领域包受控词表任一 alias 命中，或点名了数据集编号/直链。

    词表与匹配语义全部复用既有单一真源——活动领域 catalog（`query_parser._merged_catalog`
    同径合并部署方注入的说法）+ `query_parser._alias_occurrences`（与解析热路径同一份
    ASCII 词边界判定）+ `identifiers.classify`（与首个搜索框标识符直链反查同一套判据）。
    **不另抄第二份词表。**
    """
    from ..content.identifiers import classify as _classify_identifier

    text = str(text or "").strip()
    if not text:
        return False
    if _classify_identifier(text):
        return True
    catalog = _merged_catalog(keyword_mapping)
    low = text.lower()
    for dim in domain_dimensions():
        for entry in catalog.get(dim, []):
            for alias in entry.get("aliases", []):
                a = str(alias).lower().strip()
                if a and _alias_occurrences(low, a):
                    return True
    return False


def search_shaped(text: str, keyword_mapping: dict | None = None) -> bool:
    """「这句话长不长一张检索的脸」——无改动操作、无否定（存在性问句不算）、有检索信号。

    纯函数、只读、不联网。它是 `/api/utterance` 在 **LLM 缺席/失败**时的规则兜底判据
    （架构定稿：关键词匹配 → LLM 判断是唯一管线，此前还有一个
    「明确检索句绕过 LLM」的短路设计 `is_clear_search`，与用户定稿相悖，已清除）：
    兜底 none 时，长检索脸的句子照旧按检索落地（弃权诚实卡如实摆出未收录词，
    比一句「没听懂」信息多）；零信号歧义句与真否定句不许 fail-open 成检索——
    「没有小鼠的数据」是真否定，必须留在护栏回音（否定豁免是硬线）。
    """
    text = str(text or "").strip()
    if not text:
        return False
    command = parse_command(text)
    if command["op"] or command["reason"] == "conflicting_operators":
        return False
    if _has_negation(_existential_blanked(text)):
        return False
    return _has_retrieval_signal(text, keyword_mapping)


                                                                              

def board_view(
    current_filters: Sequence[dict],
    facet_filters: Sequence[dict],
    lenient_dims: Sequence[str],
    suppressed_constraints: Sequence[str],
    coverage_dims: Sequence[str],
) -> dict:
    """把「上一次检索返回的条件」+「三个开关」摊成四个分区的行。**只是再投影，不重新判断任何事。**"""
    coverage = {str(d) for d in (coverage_dims or [])}
    rows: list[dict] = []

    for item in current_filters or []:
        dim = str(item.get("dim") or "")
        rows.append({
                                                           
                                                                    
            "zone": "prefer" if str(item.get("polarity") or "") == "prefer" else "query",
            "dim": dim,
            "label": str(item.get("label") or _dim_label(dim, str(item.get("polarity") or "include"))),
            "values": [str(v) for v in (item.get("values") or [])],
            "filter_id": str(item.get("filter_id") or ""),
            "polarity": str(item.get("polarity") or "include"),
            "removable": bool(item.get("filter_id")),
            "editable": dim in domain_dimensions() and str(item.get("polarity") or "include") == "include",
            "lenientable": dim in domain_dimensions() and dim in coverage,
            "note": "",
        })

    for item in facet_filters or []:
        dim = str(item.get("dim") or "")
        rows.append({
            "zone": "facet",
            "dim": dim,
            "label": str(item.get("label") or _dim_label(dim)),
            "values": [str(item.get("display") or item.get("value") or "")],
            "filter_id": "",
            "polarity": "include",
            "removable": True,
            "editable": False,
            "lenientable": False,
            "note": "",
        })

    for dim in lenient_dims or []:
        rows.append({
            "zone": "lenient",
            "dim": str(dim),
            "label": _dim_label(str(dim)),
            "values": [],
            "filter_id": "",
            "polarity": "include",
            "removable": True,
            "editable": False,
            "lenientable": False,
            "note": "",
        })

                                             
                                              
    for filter_id in suppressed_constraints or []:
        rows.append({
                                                 
            "zone": "prefer_off" if str(filter_id).startswith("prefer:") else "suppressed",
            "dim": str(filter_id).partition(":")[2] or str(filter_id),
            "label": _label_for_filter_id(str(filter_id)),
            "values": [],
            "filter_id": str(filter_id),
            "polarity": ("exclude" if str(filter_id).startswith("exclude:")
                         else "prefer" if str(filter_id).startswith("prefer:") else "include"),
            "removable": True,
            "editable": False,
            "lenientable": False,
            "note": "",
        })

                                                      
    notes = [{"zone": zone} for zone in ("query", "facet", "lenient", "suppressed")]
    return {"rows": rows, "notes": notes}


                                                                     

def _snapshot(intent: QueryIntent) -> dict:
    return {
        "include": {d: sorted(set(v)) for d, v in intent.constraints.items() if v},
        "exclude": {d: sorted(set(v)) for d, v in intent.excluded_constraints.items() if v},
        "raw": intent.has_raw_data_required,
        "date": (intent.date_from or "", intent.date_to or ""),
    }


def _values(snapshot: dict, polarity: str, dim: str) -> list[str]:
    return list(snapshot.get(polarity, {}).get(dim, []))


def _mismatch(code: str, dim: str, before: Any, after: Any, detail: str = "") -> dict:
    return {
        "code": code, "dim": dim,
        "before": before, "after": after,
        "detail": detail or _MISMATCH_ZH.get(code, code),
    }


def verify_edit(
    op: str,
    dim: str,
    before_intent: QueryIntent,
    after_intent: QueryIntent,
    payload_intent: "QueryIntent | None",
    before_query: str,
    after_query: str,
    active_before: Sequence[dict] = (),
    active_after: Sequence[dict] = (),
    removed_filter_ids: Sequence[str] = (),
) -> dict:
    """改完之后，除了用户说的那一处，别的地方有没有跟着变。

    判据**按改动类型分派**——用一套通用判据会把「都算」这种合法的放宽当成异常判死，
    那样冲突三选一里唯一诚实的那个选项就永远点不动。
    """
    mismatches: list[dict] = []
    before = _snapshot(before_intent)
    after = _snapshot(after_intent)
    payload = _snapshot(payload_intent) if payload_intent is not None else {"include": {}, "exclude": {}}

    if after_intent.parse_status != PS_EXECUTABLE:
        mismatches.append(_mismatch("became_unexecutable", dim, before_intent.parse_status,
                                    after_intent.parse_status,
                                    after_intent.abstain_detail or after_intent.clarification_detail
                                    or _MISMATCH_ZH["became_unexecutable"]))
        return {"ok": False, "mismatches": mismatches}

    if op in ("remove", "lenient"):
        if _norm_ws(before_query) != _norm_ws(after_query):
            mismatches.append(_mismatch("collateral_change", dim, _norm_ws(before_query), _norm_ws(after_query),
                                        "这一步不该改到句子里的字，但句子变了"))
        ids_before = {str(f.get("filter_id")) for f in active_before}
        ids_after = {str(f.get("filter_id")) for f in active_after}
        if op == "remove":
            still_there = [fid for fid in removed_filter_ids if fid in ids_after]
            if still_there:
                mismatches.append(_mismatch("no_effect", dim, sorted(ids_before), sorted(ids_after),
                                            COPY["rejected_no_effect"]))
            lost = (ids_before - ids_after) - set(removed_filter_ids)
            if lost:
                mismatches.append(_mismatch("lost_filter", dim, sorted(ids_before), sorted(ids_after)))
        else:
            if ids_before != ids_after:
                mismatches.append(_mismatch("collateral_change", dim, sorted(ids_before), sorted(ids_after),
                                            "放宽不该改变任何条件，但条件变了"))
        return {"ok": not mismatches, "mismatches": mismatches}

    if op == "restart":
                                                 
        return {"ok": True, "mismatches": []}

                                                   
    target_expect_include = _values(payload, "include", dim)
    target_expect_exclude = _values(payload, "exclude", dim)

    if op == "replace":
        got = _values(after, "include", dim)
        if sorted(got) != sorted(target_expect_include):
            mismatches.append(_mismatch("unexpected_or", dim, _values(before, "include", dim), got))
    elif op == "widen":
        want = sorted(set(_values(before, "include", dim)) | set(target_expect_include))
        got = sorted(_values(after, "include", dim))
        if got != want:
            mismatches.append(_mismatch("unexpected_or", dim, want, got))
    elif op == "add":
        got_inc = sorted(_values(after, "include", dim))
        got_exc = sorted(_values(after, "exclude", dim))
        want_inc = sorted(set(_values(before, "include", dim)) | set(target_expect_include))
        want_exc = sorted(set(_values(before, "exclude", dim)) | set(target_expect_exclude))
        if got_inc != want_inc or got_exc != want_exc:
            mismatches.append(_mismatch("unexpected_or", dim,
                                        {"include": want_inc, "exclude": want_exc},
                                        {"include": got_inc, "exclude": got_exc}))

                                                          
                                                  
    if dim == "has_raw_data" and payload_intent is not None:
        want_raw = payload_intent.has_raw_data_required
        if after["raw"] != want_raw:
            mismatches.append(_mismatch("unexpected_or", dim, before["raw"], after["raw"]))
    if dim == "date" and payload_intent is not None:
        want_date = (payload_intent.date_from or "", payload_intent.date_to or "")
        if want_date != ("", "") and after["date"] != want_date:
            mismatches.append(_mismatch("unexpected_or", dim, before["date"], after["date"]))

                                  
    for other in domain_dimensions():
        if other == dim:
            continue
        for polarity in ("include", "exclude"):
            was, now = _values(before, polarity, other), _values(after, polarity, other)
            if sorted(was) != sorted(now):
                code = "lost_filter" if (was and not now) else "collateral_change"
                mismatches.append(_mismatch(code, other, was, now))
    if dim != "has_raw_data" and before["raw"] != after["raw"]:
        mismatches.append(_mismatch("collateral_change", "has_raw_data", before["raw"], after["raw"]))
    if dim != "date" and before["date"] != after["date"]:
        mismatches.append(_mismatch("collateral_change", "date", before["date"], after["date"]))

    return {"ok": not mismatches, "mismatches": mismatches}


                                                                        

def _describe_changes(before_intent: QueryIntent, after_intent: QueryIntent) -> str:
    """用户手改整句之后，到底有哪几条条件跟着变了——照实列出来给他看，而不是替他判对错。"""
    before = {(f["filter_id"], tuple(f["values"])) for f in active_filters(before_intent)}
    after = {(f["filter_id"], tuple(f["values"])) for f in active_filters(after_intent)}
    parts: list[str] = []
    for filter_id, values in sorted(after - before):
        parts.append(f"{_label_for_filter_id(filter_id)} 变成 {'、'.join(values) or '不限'}")
    for filter_id, _values in sorted(before - after):
        if filter_id not in {fid for fid, _v in after}:
            parts.append(f"不再按{_label_for_filter_id(filter_id)}筛")
    return "；".join(parts)


def _render(query: str, protect: set, edits: list, suffix: str, masked: bool) -> str:
    """按「原句 + 若干段替换 + 末尾追加」渲染候选句。

    保护区（数据来源专名）永远不被 edits 覆盖，所以同一份 edits 渲染两次——一次保留专名（真正拿去检索的句子），
    一次把专名换成空格（拿去比对条件的句子）——两者结构逐位对应。
    被中和的片段一律换成**一个空格**而不是删掉：删字符会让左右残字贴到一起，拼出一个原本不存在的别名。
    """
    def seg(lo: int, hi: int) -> str:
        if not masked:
            return query[lo:hi]
        return "".join(" " if i in protect else query[i] for i in range(lo, hi))

    out: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda e: e[0]):
        if start < cursor:
            continue
        out.append(seg(cursor, start))
        out.append(replacement)
        cursor = end
    out.append(seg(cursor, len(query)))
    return _norm_ws("".join(out) + suffix)


                                                                      

def _clean_current_filters(raw: object) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw[:32]:
        if not isinstance(item, dict):
            continue
        filter_id = str(item.get("filter_id") or "").strip()
        dim = str(item.get("dim") or "").strip()
        if not filter_id or not dim:
            continue
        out.append({
            "filter_id": filter_id,
            "dim": dim,
            "polarity": str(item.get("polarity") or "include").strip() or "include",
            "label": str(item.get("label") or _dim_label(dim, str(item.get("polarity") or "include"))),
            "values": [str(v) for v in (item.get("values") or [])][:12],
        })
    return out


def _normalize_inputs(
    query: str, utterance: str, forced_op: str, dim: str, candidate_override: str,
    current_filters: object, suppressed_constraints: object, lenient_dims: object,
    facet_filters: object, coverage_dims: object, date_from: str, date_to: str,
) -> dict:
    query = str(query or "")
    utterance = str(utterance or "").strip()
    candidate_override = str(candidate_override or "")
    if len(query) > MAX_QUERY_CHARS:
        raise BoardError("too_large", f"查询太长（上限 {MAX_QUERY_CHARS} 个字符）。")
    if len(utterance) > MAX_UTTERANCE_CHARS:
        raise BoardError("too_large", f"这句话太长（上限 {MAX_UTTERANCE_CHARS} 个字符）。")
    if len(candidate_override) > MAX_QUERY_CHARS:
        raise BoardError("too_large", f"改后的句子太长（上限 {MAX_QUERY_CHARS} 个字符）。")
    forced_op = str(forced_op or "").strip()
    if forced_op and forced_op not in OPS:
        raise BoardError("bad_param", f"forced_op 只能是 {'/'.join(OPS)} 之一。")
    dim = str(dim or "").strip()
    if not utterance and not forced_op:
        raise BoardError("empty_input", "请说一句要怎么改，或者点条件右边的按钮。")
    return {
        "query": query,
        "utterance": utterance,
        "forced_op": forced_op,
        "dim": dim,
        "candidate_override": candidate_override,
        "current_filters": _clean_current_filters(current_filters),
        "suppressed_constraints": sanitize_suppressed(suppressed_constraints),
        "lenient_dims": sorted(sanitize_lenient_dims(lenient_dims)),
        "facet_filters": sanitize_facet_filters(facet_filters),
        "coverage_dims": sorted({str(d).strip() for d in (coverage_dims or []) if str(d).strip()}),
                                                 
                                                      
                                                  
        "coverage_known": coverage_dims is not None,
        "date_from": str(date_from or "").strip(),
        "date_to": str(date_to or "").strip(),
    }


def _blank(
    norm: dict, status: str, *, op: str = "", dim: str = "", message: str = "", detail: str = "",
    suggestions: "list[dict] | None" = None, choices: "list[dict] | None" = None,
    verify: "dict | None" = None, removed_text: "list[dict] | None" = None,
    dropped_terms: "list[str] | None" = None,
) -> dict:
    """一个「什么都没改」的回执。所有键在任何状态下都在场，缺省是 None / '' / []。"""
    return {
        "schema": BOARD_SCHEMA,
        "status": status,
        "op": op,
        "dim": dim,
        "message": message,
        "detail": detail,
                                                  
                                                     
                                                         
        "route": classify_utterance(norm["utterance"]),
                                                      
                                                              
                                                   
        "action_markers": _detect_action_markers(norm["utterance"]),
        "next_request": None,
        "removed_text": removed_text or [],
        "dropped_terms": dropped_terms or [],
        "verify": verify or {"ok": True, "mismatches": []},
        "choices": choices or [],
        "suggestions": suggestions or [],
        "board_view": board_view(norm["current_filters"], norm["facet_filters"],
                                 norm["lenient_dims"], norm["suppressed_constraints"],
                                 norm["coverage_dims"]),
        "echoed": {
            "query": norm["query"],
            "utterance": norm["utterance"],
            "suppressed_constraints": list(norm["suppressed_constraints"]),
            "lenient_dims": list(norm["lenient_dims"]),
            "facet_filters": [dict(f) for f in norm["facet_filters"]],
            "coverage_dims": list(norm["coverage_dims"]),
            "date_from": norm["date_from"],
            "date_to": norm["date_to"],
        },
    }


def _locate_dim_in_payload(payload: str, keyword_mapping) -> dict:
    """这段话说的是哪一项。返回 {dim, spans, dims} —— dims 有两个及以上就是跨项歧义。"""
    spans = annotate_query_spans(payload, keyword_mapping)
    dims: list[str] = []
    for span in spans:
        if span["dim"] not in dims:
            dims.append(span["dim"])
    return {"dims": dims, "spans": spans, "dim": dims[0] if len(dims) == 1 else ""}


def _payload_text(intent: "QueryIntent | None", dim: str, fallback: str) -> str:
    """这段话在那一项上到底说了什么，用于预览与三选一的按钮文字。

    结构化项走 display 名（Human / Mouse）；原始数据与发表时间没有 display，
    走 `active_filters` 生成的那句人话（需要 FASTQ / ≥），
    否则预览会写成「原始数据：含 FASTQ 的」这种把用户原话当取值的句子。
    """
    if intent is None:
        return fallback
    if dim in domain_dimensions():
        values = list(intent.display_map.get(dim, [])) or list(intent.excluded_display.get(dim, []))
        if values:
            return "、".join(values)
    for item in active_filters(intent):
        if item.get("dim") == dim:
            return "、".join(str(v) for v in item.get("values", []))
    return fallback


def _dim_from_noun(text: str) -> str:
    low = str(text or "").lower()
    for noun, dim in sorted(_dim_nouns(), key=lambda t: len(t[0]), reverse=True):
        if noun in low:
            return dim
    return ""


def plan_edit(
    query: str = "",
    utterance: str = "",
    *,
    forced_op: str = "",
    dim: str = "",
    candidate_override: str = "",
    current_filters: object = None,
    resolution: object = None,
    suppressed_constraints: object = None,
    lenient_dims: object = None,
    facet_filters: object = None,
    coverage_dims: object = None,
    date_from: str = "",
    date_to: str = "",
    keyword_mapping: object = None,
    implicit_restrict: bool = False,
) -> dict:
    """规划一次条件改动。纯函数：不检索、不装载数据集目录、不落盘、不联网。

    `query` 必须是**产生当前这批结果的那句话**（开了关键词审核时是改写后那句），
    `current_filters` 必须是**上一次检索返回的 active_filters 原样**——这两个入参就是本模块的全部记忆。

    `candidate_override` 是用户在预览框里手改后的整句。它**沿用同一次规划**：
    只有当这句话本身先规划成功（因而界面上才出现那个可编辑的预览框）时才会走到这里。
    直接调接口的一方若给了改后的句子却给不出一句能规划成功的原话，会得到「没听懂」而不是执行——
    这是有意的 fail-closed，不是遗漏。

    `implicit_restrict`：透传给 `parse_command` 的量词约束档开关（「只要…」按 add 收窄），
    只由**对话上下文**的调用方（`/api/utterance` 且屏上有结果）打开；首屏检索与 MCP 默认关。
    """
    norm = _normalize_inputs(query, utterance, forced_op, dim, candidate_override,
                             current_filters, suppressed_constraints, lenient_dims,
                             facet_filters, coverage_dims, date_from, date_to)
    raw_query = norm["query"]

                                        
    src_spans = source_alias_spans(raw_query)
    protect = {i for s in src_spans for i in range(s["start"], s["end"])}
    masked_query = mask_source_spans(raw_query, src_spans)

    if len(raw_query.lower()) != len(raw_query):
        return _blank(norm, ST_REJECTED, message=COPY["non_ascii_case_shift"],
                      verify={"ok": False, "mismatches": [_mismatch("protected_region_touched", "", "", "",
                                                                    COPY["non_ascii_case_shift"])]})

                                                  
    source_selfcheck_ok = True
    if isinstance(resolution, dict) and resolution.get("parsed_query") is not None:
        source_selfcheck_ok = _norm_ws(masked_query) == _norm_ws(str(resolution.get("parsed_query")))

                             
    if norm["forced_op"]:
        op = norm["forced_op"]
        payload = _strip_quant(norm["utterance"]) if norm["utterance"] else ""
        target_dim = norm["dim"]
    else:
        command = parse_command(norm["utterance"], implicit_restrict=implicit_restrict)
        op = command["op"]
        payload = command["payload"]
        target_dim = ""
        if not op:
            return _blank(norm, ST_UNKNOWN, message=COPY["not_understood"], detail=command["detail"],
                          suggestions=[])

    if op == "suggest":
        target_dim = norm["dim"] or _dim_from_noun(payload)
        return _blank(norm, ST_SUGGEST, op="suggest", dim=target_dim,
                      message=COPY["suggest_title"],
                      suggestions=suggestable_terms(target_dim) if target_dim in domain_dimensions() else [])

                                        
    if payload and source_alias_spans(payload):
        return _blank(norm, ST_UNKNOWN, op=op, message=COPY["source_in_utterance"])

    by_dim: dict[str, list[dict]] = {}
    for item in norm["current_filters"]:
        by_dim.setdefault(item["dim"], []).append(item)

                           
    payload_intent: "QueryIntent | None" = None
    removed_filter_ids: list[str] = []

    if op in ("remove", "lenient"):
        target_dim = target_dim or _dim_from_noun(payload)
        if not target_dim:
            located = _locate_dim_in_payload(payload, keyword_mapping)
            if len(located["dims"]) == 1:
                target_dim = located["dims"][0]
            elif len(located["dims"]) >= 2:
                return _blank(norm, ST_UNKNOWN, op=op,
                              message=COPY["cross_dim"].format(
                                  term=payload, a=_dim_label(located["dims"][0]),
                                  b=_dim_label(located["dims"][1])))
        if not target_dim:
            return _blank(norm, ST_UNKNOWN, op=op, message=COPY["not_understood"],
                          detail=COPY["no_payload"])
        rows = by_dim.get(target_dim, [])
        if op == "remove":
            if not rows:
                return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                              message=COPY["no_such_filter"].format(label=_dim_label(target_dim)))
                                            
                                                              
                                                   
            removed_filter_ids = [r["filter_id"] for r in rows]
        else:
            if target_dim not in domain_dimensions():
                return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                              message=COPY["lenient_unsupported"].format(label=_dim_label(target_dim)))
                                                 
                                              
            if norm["coverage_known"] and target_dim not in norm["coverage_dims"]:
                return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                              message=COPY["lenient_no_gap"].format(label=_dim_label(target_dim)))
    elif op in ("replace", "widen", "add"):
        if not payload:
            return _blank(norm, ST_UNKNOWN, op=op, message=COPY["no_payload"])
        payload_intent = parse_query(payload, keyword_mapping)
        located = _locate_dim_in_payload(payload, keyword_mapping)
        if not located["dims"]:
            suggestions = suggestable_terms(target_dim) if target_dim in domain_dimensions() else []
            message = (COPY["unknown_term"] if keyword_mapping else COPY["unknown_term_no_mapping"]).format(term=payload)
            return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim, message=message, suggestions=suggestions)
        if len(located["dims"]) >= 2:
            return _blank(norm, ST_UNKNOWN, op=op,
                          message=COPY["cross_dim"].format(
                              term=payload, a=_dim_label(located["dims"][0]),
                              b=_dim_label(located["dims"][1])))
        located_dim = located["dims"][0]
        if target_dim and target_dim != located_dim:
            return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                          message=COPY["cross_dim"].format(term=payload, a=_dim_label(target_dim),
                                                           b=_dim_label(located_dim)))
        target_dim = located_dim
        if payload_intent.parse_status != PS_EXECUTABLE:
            return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim, message=COPY["not_understood"],
                          detail=payload_intent.abstain_detail or payload_intent.clarification_detail)
    elif op == "restart":
        if not payload:
            return _blank(norm, ST_UNKNOWN, op=op, message=COPY["no_payload"])
    else:
        return _blank(norm, ST_UNKNOWN, message=COPY["not_understood"])

                                               
    if op in ("replace", "widen", "add") and not source_selfcheck_ok:
        return _blank(norm, ST_REJECTED, op=op, dim=target_dim, message=COPY["source_selfcheck_failed"],
                      verify={"ok": False, "mismatches": [_mismatch("source_scope_changed", target_dim,
                                                                    _norm_ws(masked_query), "")]})

                                
                                       
                                                
                                 
    positive_payload = bool(payload_intent and payload_intent.constraints.get(target_dim))
    if op == "add" and positive_payload and by_dim.get(target_dim):
        existing = by_dim[target_dim][0]
                                                      
                                                   
        new_vals = {str(v).lower() for v in payload_intent.constraints.get(target_dim, [])}
        cur_vals = {str(v).lower() for v in (existing.get("values") or [])}
        if new_vals and new_vals <= cur_vals:
            return _blank(norm, ST_REJECTED, op=op, dim=target_dim, message=COPY["rejected_no_effect"])
        before_text = "、".join(existing["values"]) or _dim_label(target_dim)
        after_text = _payload_text(payload_intent, target_dim, payload)
        choices = [
            {"id": "replace", "op": "replace", "dim": target_dim, "payload": payload,
             "label": COPY["choice_replace"].format(after=after_text)},
            {"id": "widen", "op": "widen", "dim": target_dim, "payload": payload,
             "label": COPY["choice_widen"].format(before=before_text, after=after_text)},
            {"id": "restart", "op": "restart", "dim": "", "payload": payload,
             "label": COPY["choice_restart"]},
        ]
        return _blank(norm, ST_CHOICE, op=op, dim=target_dim,
                      message=COPY["choice_prompt"].format(label=_dim_label(target_dim),
                                                           before=before_text, after=after_text),
                      choices=choices)

                        
    next_suppressed = list(norm["suppressed_constraints"])
    next_lenient = list(norm["lenient_dims"])
    next_facets = [dict(f) for f in norm["facet_filters"]]
    next_from, next_to = norm["date_from"], norm["date_to"]
    edits: list[tuple[int, int, str]] = []
    suffix = ""
    removed_text: list[dict] = []
    dropped_terms: list[str] = []

    if op == "remove":
        for fid in removed_filter_ids:
            if fid not in next_suppressed:
                next_suppressed.append(fid)
        next_suppressed = sanitize_suppressed(next_suppressed)
    elif op == "lenient":
        if target_dim not in next_lenient:
            next_lenient.append(target_dim)
        next_lenient = sorted(sanitize_lenient_dims(next_lenient))
    elif op == "restart":
        next_suppressed, next_lenient, next_facets = [], [], []
    else:
        text_spans = [s for s in annotate_query_spans(masked_query, keyword_mapping)
                      if s["dim"] == target_dim]
        if op == "replace":
            if not text_spans:
                return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                              message=COPY["nothing_to_replace"].format(label=_dim_label(target_dim),
                                                                        after=payload))
            for span in text_spans:
                edits.append((span["start"], span["end"], " "))
                removed_text.append({"text": span["text"],
                                     "context": _context_window(raw_query, span["start"], span["end"])})
            suffix = " " + payload
                                                           
            next_suppressed = [f for f in next_suppressed if f != f"include:{target_dim}"]
        elif op == "widen":
            if not text_spans:
                return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                              message=COPY["nothing_to_replace"].format(label=_dim_label(target_dim),
                                                                        after=payload))
            first = text_spans[0]
                                                  
                                     
            edits.append((first["start"], first["end"], f"{first['text']}、{payload}"))
        elif op == "add":
                                            
                                      
            if _has_negation(payload):
                suffix = " " + payload
            else:
                keep = [s for s in _locate_dim_in_payload(payload, keyword_mapping)["spans"]]
                covered = "".join(payload[s["start"]:s["end"]] for s in keep)
                leftover = payload
                for span in sorted(keep, key=lambda s: s["start"], reverse=True):
                    leftover = leftover[:span["start"]] + " " + leftover[span["end"]:]
                leftover_intent = parse_query(_norm_ws(leftover), keyword_mapping) if _norm_ws(leftover) else None
                if leftover_intent is not None and leftover_intent.parse_status != PS_EXECUTABLE:
                                                    
                    return _blank(norm, ST_UNKNOWN, op=op, dim=target_dim,
                                  message=COPY["unknown_term"].format(term=_norm_ws(leftover)),
                                  suggestions=suggestable_terms(target_dim) if target_dim in domain_dimensions() else [])
                                                         
                                                           
                if leftover_intent is not None:
                    dropped_terms = list(leftover_intent.unused_query_terms)
                suffix = " " + (covered if covered else payload)

                                     
    override_mode = bool(norm["candidate_override"].strip())
    if override_mode:
        candidate_full = _norm_ws(norm["candidate_override"])
        candidate_masked = _norm_ws(mask_source_spans(candidate_full))
        removed_text = []
    elif op in ("remove", "lenient"):
        candidate_full = _norm_ws(raw_query)
        candidate_masked = _norm_ws(masked_query)
    elif op == "restart":
        candidate_full = _norm_ws(payload)
        candidate_masked = _norm_ws(mask_source_spans(candidate_full))
    else:
        candidate_full = _render(raw_query, protect, edits, suffix, masked=False)
        candidate_masked = _render(raw_query, protect, edits, suffix, masked=True)

                              
    before_intent = parse_query(_norm_ws(masked_query), keyword_mapping)
    apply_explicit_date_range(before_intent, norm["date_from"], norm["date_to"])
    apply_suppressed_constraints(before_intent, norm["suppressed_constraints"])
                                               
                                           
    active_before = active_filters(before_intent)

    after_intent = parse_query(candidate_masked, keyword_mapping)
    apply_explicit_date_range(after_intent, next_from, next_to)
    apply_suppressed_constraints(after_intent, next_suppressed)
    active_after = active_filters(after_intent)

    if override_mode:
                                            
                                        
        verify = {"ok": after_intent.parse_status == PS_EXECUTABLE, "mismatches": []}
        if not verify["ok"]:
            verify["mismatches"] = [_mismatch("became_unexecutable", target_dim,
                                              before_intent.parse_status, after_intent.parse_status,
                                              after_intent.abstain_detail
                                              or after_intent.clarification_detail
                                              or _MISMATCH_ZH["became_unexecutable"])]
    else:
        verify = verify_edit(op, target_dim, before_intent, after_intent, payload_intent,
                             _norm_ws(masked_query), candidate_masked,
                             active_before=active_before, active_after=active_after,
                             removed_filter_ids=removed_filter_ids)

                                    
                                    
    if op != "restart":
        before_src = [(s["source"], raw_query[s["start"]:s["end"]]) for s in src_spans]
        after_src = [(s["source"], candidate_full[s["start"]:s["end"]])
                     for s in source_alias_spans(candidate_full)]
        if sorted(before_src) != sorted(after_src):
            verify = {"ok": False,
                      "mismatches": list(verify["mismatches"]) + [
                          _mismatch("source_scope_changed", "source",
                                    [t for _s, t in before_src], [t for _s, t in after_src])]}

    if not verify["ok"]:
        first = verify["mismatches"][0]["code"]
        message = COPY["rejected"]
        if first in ("source_scope_changed", "protected_region_touched"):
            message = COPY["rejected_source"]
        elif first == "became_unexecutable":
            message = COPY["rejected_unexecutable"]
        elif first == "no_effect":
            message = COPY["rejected_no_effect"]
        return _blank(norm, ST_REJECTED, op=op, dim=target_dim, message=message,
                      detail="；".join(m["detail"] for m in verify["mismatches"]),
                      verify=verify, removed_text=removed_text, dropped_terms=dropped_terms)

                        
                                               
                                   
    facet_note = ""
    if op in ("add", "replace", "widen", "restart") and next_facets:
        next_facets = []
        facet_note = COPY["facet_cleared"]

                             
    next_request = {
        "query": candidate_full,
        "suppressed_constraints": list(next_suppressed),
        "lenient_dims": list(next_lenient),
        "facet_filters": [dict(f) for f in next_facets],
        "date_from": next_from,
        "date_to": next_to,
    }

                                           
                                                 
    negative_payload = bool(
        payload_intent
        and payload_intent.excluded_constraints.get(target_dim)
        and not payload_intent.constraints.get(target_dim)
    )
    label = _dim_label(target_dim, "exclude" if negative_payload else "include")
    extra_detail = ""
    if override_mode:
        status = ST_CONFIRM
        message = COPY["override_applied"]
        changes = _describe_changes(before_intent, after_intent)
        extra_detail = COPY["override_changed"].format(changes=changes) if changes else ""
    elif op == "remove":
        status = ST_AUTO
        labels = "、".join(dict.fromkeys(_label_for_filter_id(f) for f in removed_filter_ids))
        message = COPY["applied_remove"].format(label=labels or label)
    elif op == "lenient":
        status = ST_AUTO
        message = (COPY["applied_lenient"] if norm["coverage_known"]
                   else COPY["lenient_coverage_unknown"]).format(label=label)
    else:
        status = ST_CONFIRM
        before_text = _payload_text(before_intent, target_dim, "")
        after_text = _payload_text(payload_intent, target_dim, payload)
        if op == "replace":
            message = (COPY["preview_replace"].format(label=label, before=before_text, after=after_text)
                       if before_text else COPY["preview_add_new"].format(label=label, after=after_text))
        elif op == "widen":
            message = COPY["preview_widen"].format(label=label, before=before_text, after=after_text)
        elif op == "restart":
            message = COPY["preview_restart"].format(payload=payload)
        else:
            message = COPY["preview_add_new"].format(label=label, after=after_text)

    result = _blank(norm, status, op=op, dim=target_dim, message=message,
                    verify=verify, removed_text=removed_text, dropped_terms=dropped_terms)
    result["next_request"] = next_request
    result["detail"] = "；".join(x for x in (
        extra_detail, facet_note, COPY["order_may_change"] if status == ST_CONFIRM else "") if x)
    return result
