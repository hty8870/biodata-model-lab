                       
"""dream 记忆：基于历史对话整理出用户记忆（手动「整理记忆」按钮触发，LLM 护栏输出）。

定位与诚实语义（设计决定：手动触发 + 可以发给已配置的 LLM）：
- 素材是**前端从本机历史快照组织**的对话记录（query + chat），服务端零存储、无后台任务。
- LLM 配置随请求带来（与 /api/utterance 同契约：请求级 key 优先、绝不持久化）；
  无 key → `no_key`，前端如实提示去设置配置，**不静默退化成编造**。
- **AI 开关（enable_llm）照常生效**：手动触发不等于绕过用户明确关闭的 AI——关闭时一个
  调用都不发（不烧 token），返回 `disabled` 让人话说明去哪开（2026-10-04 裁决；不物化
  「强制启用副本」以免绕过用户意图与调用方配额前提）。
- 封闭输出护栏（对齐 action_plan 哲学）：只接受 JSON 数组 `[{"text","summary","evidence"}]`；
  解析失败 / 结构不符 / 超界 → 该条丢弃，全丢 = 空清单（前端如实说「这次没整理出」），
  绝不允许模型随口一段散文变成「记忆」。
- 产出**一律是 AI 生成物**：返回 `generated: true`，前端徽标「AI 整理」，预览勾选后才写入，
  绝不冒充用户手写偏好。

2026-08-08 B6 调研六候选批——两道新机械闸（epub「持久记忆是第四致命要素」+「升级需两条
非失败轨迹支持」+ Mem0 对账思想；审核 修订：词法过滤只当纵深防御，不当安全门）：
1. **出处核验（证据门槛，真正的安全门）**：每条候选必须附 1~2 段逐字摘自对话原文的
   evidence；服务端归一化核验——每段 span 真实落在**单条消息**里（跨消息拼接的伪
   evidence 不算，E-03），且全部 span 合计覆盖 **≥2 段不同对话**（对话按内容 hash
   去重，防同一段重复提交凑数；k="sys" 系统消息不算证据面）。核验不过 → 丢弃记
   `dropped.evidence`。
2. **注入审查（纵深防御）**：候选 text 命中指令形态（「忽略/你现在是/系统提示/指令/
   prompt/system/ignore/不要返回」等元词与行为封禁词）→ 丢弃记 `dropped.injection`。
   它只是兜底——真正的防线是出处核验 + 封闭输出闸 + 人工预览勾选。
安全门语义不变：任何一道不过都**丢弃**（宁缺毋滥），不降级、不编造。

2026-09-29 协作规则卡（记忆层 RSI 落地，additive）：抽取扩「协作规则」类目
（用户对助手行为的持久要求），条目在 `{text, summary, evidence}` 上加性多两个字段
`{trigger?, tier?}`——**tier 在场即规则条目**（其余类目逐位不变，不加任何字段）。
2026-09-29 融合批：原第 3 类「明确说过的约束」与「协作规则」语义重复（都是「不要 X /
必须 Y」的持久要求），**退役并轨**进后者——抽取 prompt 由四类目收敛为三类目，类目定义与
示例只留一处；咨询层（只存本机、不进上下文）只保留「研究偏好 / 工作习惯」两类。输出侧
判据一个字不改（仍看 `tier` 在不在）：并轨落在**数据**（prompt 的类目定义）上，不新增
任何「把某条记忆改判成规则」的代码分支——按关键词猜类目会在「不要小鼠」这类合法记忆偏好
上误判，且那等于同一件事有两条判定通道。
两道闸按**类目分档**（`_GATE_PROFILES` 表驱动，不写特例分支）：
- 注入审查换词表：规则类目**保留元词封禁**（忽略/你现在是/系统提示/指令/prompt/system/
  ignore/override 等），但**放行行为否定词**「不要/不许(返回|显示|回答)」——规则本体
  常就是「不要自动执行」这类行为约束，原词表会连规则一起吃掉；
- 出处核验分档：`tier=explicit`（用户明说这是一条规则）**1 段逐字出处**即成立——
  用户逐条点「确认」这个动作本身替代了第二段证据；`tier=implicit`（从多次行为推断）
  仍要 ≥2 段不同对话。tier 缺失/非法一律按 `implicit` 计（fail-safe：宁可要求更多证据）。
规则条目**不进 `memories`**，走独立 `rules` 字段 + 服务端账本（agent/collab_rules.py）。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Iterable, NamedTuple

from .llm_client import (
    LLMConfig,
    _sanitize_provider_error,
    call_llm,
    load_llm_config,
    should_use_llm as _should_use_llm,
)

DREAM_MAX_ITEMS = 8                                 
DREAM_TEXT_MAX = 120                        
DREAM_SUMMARY_MAX = 240                     
DREAM_EVIDENCE_MAX = 120                          
DREAM_EVIDENCE_SPANS = 2                 
DREAM_MIN_SPAN_CHARS = 4                                      
DREAM_TRIGGER_MAX = 80                           
DREAM_MAX_CONVERSATIONS = 12                
DREAM_CHAT_PER_CONV_MAX = 40                                     
DREAM_CONV_CHAR_MAX = 1500                          

                                                     
                                           
DREAM_TIER_EXPLICIT = "explicit"
DREAM_TIER_IMPLICIT = "implicit"
DREAM_TIERS = (DREAM_TIER_EXPLICIT, DREAM_TIER_IMPLICIT)
                                                         
_TIER_ALIASES = {
    "explicit": DREAM_TIER_EXPLICIT, "明说": DREAM_TIER_EXPLICIT, "明确": DREAM_TIER_EXPLICIT,
    "implicit": DREAM_TIER_IMPLICIT, "推断": DREAM_TIER_IMPLICIT, "隐含": DREAM_TIER_IMPLICIT,
}
                                                  
                                   
_RULE_KIND_HINTS = frozenset({"rule", "rules", "规则", "协作规则", "collab_rule"})

_PROMPT_HEAD = """你是一个「研究记忆整理员」。下面是一位生物医学数据检索用户的历史对话记录
（每段：当时的检索需求 + 对话过程）。请从中提炼**值得长期记住**的信息，写成条目。

只提炼这三类：
1. 研究偏好：反复出现或被明确强调的方向（物种/组织/疾病/平台/数据类型偏好）；
2. 工作习惯：反复出现的操作意图（例如总要 FASTQ 原始数据、总要打包下载、关注某类来源）；
3. 协作规则：用户**提出过的持久要求**——既包括对**助手行为**的要求（例如「不要自动替我
   执行写操作」「汇报里不要用夸张措辞」「先给结论再给依据」），也包括用户明确说过要长期
   遵守的约束（例如「只要人类数据」「排除小鼠」「只看近三年」）。第 3 类要另外填两个字段：
   - tier="explicit"：用户**明确说过这是一条规则/要求**（例如「以后都……」「记住……」）；
   - tier="implicit"：你从用户多次行为里**推断**出来的规则（用户自己没有明说）；
   - trigger：这条规则**什么时候适用**（例如「涉及写操作时」「汇报结果时」），没有就不填。
   前 2 类**不要**填 trigger / tier。

规则：
- 只写对话里**确实出现**的信息，不许推测、不许编造没有说过的偏好；
- 一次性、随口的话不写（只出现一次且无强调的不写）；
- 每条 text 是可直接回填检索框的一句需求或偏好陈述（不超过 60 字）；
  第 3 类的 text 写的是**对助手的要求或长期约束**（「不要…」「先…再…」「只要…」），
  不是一次性的检索需求；
- 每条 summary 用一句话写明依据（哪段对话、出现了几次）；
- 每条必须附 evidence：1~2 段**逐字摘自对话原文**的小片段（每段不超过 60 字）。
  系统会逐字核验：这些片段必须真的出现在对话里、且加起来至少覆盖**两段不同对话**；
  例外：tier="explicit" 的第 3 类条目——用户明确说过一次即可，1 段就够；
  核验不过的条目会被直接丢弃——所以**照抄原文、不要改写、不要翻译**；
- 最多 8 条；没有值得写的就返回空数组。

输出**只有** JSON 数组，不要任何其它文字：
[{"text": "…", "summary": "…", "evidence": ["原文片段", "…"]}, …]
第 3 类（协作规则）多两个字段：
[{"text": "…", "summary": "…", "evidence": ["原文片段"], "tier": "explicit", "trigger": "…"}, …]

历史对话：
"""


class DreamError(Exception):
    """机器码 + 人读提示；接口层翻成 4xx/503。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _clip_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """服务端入参收口（纵深防御：前端截断过一遍，这里只认白名单字段再截一遍）。"""
    out: list[dict[str, Any]] = []
    for conv in (conversations or [])[:DREAM_MAX_CONVERSATIONS]:
        if not isinstance(conv, dict):
            continue
        query = str(conv.get("query") or "").strip()[:300]
        chat: list[dict[str, str]] = []
        for msg in (conv.get("chat") or [])[:DREAM_CHAT_PER_CONV_MAX]:
            if not isinstance(msg, dict):
                continue
            text = str(msg.get("t") or "").strip()[:300]
            note = str(msg.get("n") or "").strip()[:300]
            kind = str(msg.get("k") or "")
            if text:
                chat.append({"k": kind, "t": text, "n": note})
        if query or chat:
            out.append({"query": query, "chat": chat})
    return out


def build_dream_prompt(conversations: list[dict[str, Any]]) -> str:
    """程序生成 prompt（action_plan 模式：模型只见到整理好的素材，不见原始自由文本指令位）。"""
    parts = [_PROMPT_HEAD]
    for i, conv in enumerate(_clip_conversations(conversations), 1):
        lines = [f"—— 第 {i} 段 ——"]
        if conv["query"]:
            lines.append(f"检索需求：{conv['query']}")
        budget = DREAM_CONV_CHAR_MAX - sum(len(x) for x in lines)
        for msg in conv["chat"]:
            who = {"say": "用户", "refine": "细化", "action": "用户(执行)", "sys": "系统"}.get(msg["k"], "用户")
            line = f"{who}：{msg['t']}" + (f"（{msg['n']}）" if msg["n"] else "")
            if budget - len(line) < 0:
                lines.append("……（本段后文从略）")
                break
            lines.append(line)
            budget -= len(line)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_tier(value: Any) -> str:
    """tier 归一（单一真源）：认不出的一律回落 `implicit`——fail-safe 方向是
    「多要一段证据」，绝不让一个写错的值把非法条目放宽成 explicit。"""
    return _TIER_ALIASES.get(_norm(value), DREAM_TIER_IMPLICIT)


def _is_rule_item(item: dict[str, Any]) -> bool:
    """规则条目判据：**带 tier 字段**（或显式写了 kind=rule 之类的线索）。
    其余类目两个字段都不填 → 判定为 False，走原路径，产物逐位不变。"""
    if _norm(item.get("tier")):
        return True
    return _norm(item.get("kind") or item.get("type") or "") in _RULE_KIND_HINTS


def normalize_rule_candidate(item: Any) -> dict[str, Any] | None:
    """协作规则候选收口（**单一真源**：dream 抽取端与入库端点都用它）。

    做三件事：字段白名单 + 长度截断（`DREAM_*` 上限）+ 注入审查（规则档词表）。
    不合格（非 dict / 无 text / 命中注入词表）返回 None——调用方记 `rejected` 计数，
    绝不降级、绝不猜。返回形状与 `parse_dream_output` 的规则条目逐位同形，故入库端
    与抽取端不会长出两套契约。"""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()[:DREAM_TEXT_MAX]
    if not text:
        return None
    trigger = str(item.get("trigger") or "").strip()[:DREAM_TRIGGER_MAX]
    summary = str(item.get("summary") or "").strip()[:DREAM_SUMMARY_MAX]
    if _injection_hit(_GATE_PROFILES["rule"], (text, summary, trigger)):
        return None
    raw_ev = item.get("evidence")
    spans_in = raw_ev if isinstance(raw_ev, list) else ([raw_ev] if raw_ev else [])
    spans: list[str] = []
    for span in spans_in:
        span_text = str(span or "").strip()[:DREAM_EVIDENCE_MAX]
        if span_text and span_text not in spans:
            spans.append(span_text)
        if len(spans) >= DREAM_EVIDENCE_SPANS:
            break
    out: dict[str, Any] = {"text": text, "summary": summary, "evidence": spans,
                           "tier": normalize_tier(item.get("tier"))}
    if trigger:
        out["trigger"] = trigger
    return out


def parse_dream_output(raw: str | None) -> list[dict[str, Any]]:
    """封闭输出解析：只认 JSON 数组 [{text, summary, evidence}]；坏条丢弃、全坏为空（绝不编造）。
    evidence 容忍模型给单字符串（收进列表）；每段截 DREAM_EVIDENCE_MAX、每条至多
    DREAM_EVIDENCE_SPANS 段、空段丢弃——**没有 evidence 的条目在这里不丢**（解析≠核验），
    由 `_evidence_checks` 统一裁决（那样丢弃原因才可计数上报）。
    第 3 类（协作规则）的条目加性多 `tier`（必在）与 `trigger`（非空才在）：**tier 在场即
    规则条目**；其余类目的产物因此与历史**逐位同形**（不多键、不改值）。"""
    if not raw:
        return []
    text = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        t = str(item.get("text") or "").strip()[:DREAM_TEXT_MAX]
        s = str(item.get("summary") or "").strip()[:DREAM_SUMMARY_MAX]
        raw_ev = item.get("evidence")
        spans_in = raw_ev if isinstance(raw_ev, list) else ([raw_ev] if raw_ev else [])
        spans: list[str] = []
        for span in spans_in:
            span_text = str(span or "").strip()[:DREAM_EVIDENCE_MAX]
            if span_text and span_text not in spans:
                spans.append(span_text)
            if len(spans) >= DREAM_EVIDENCE_SPANS:
                break
        key = _norm(t)
        if not t or key in seen:
            continue
        seen.add(key)
        row: dict[str, Any] = {"text": t, "summary": s, "evidence": spans}
        if _is_rule_item(item):
            row["tier"] = normalize_tier(item.get("tier"))
            trigger = str(item.get("trigger") or "").strip()[:DREAM_TRIGGER_MAX]
            if trigger:
                row["trigger"] = trigger
        out.append(row)
        if len(out) >= DREAM_MAX_ITEMS:
            break
    return out


def _default_llm_call(prompt: str, config: LLMConfig) -> str | None:
    result = call_llm(prompt, config)
    return result.text if result.succeeded else None


                                                                       

                                             
                                                  
                          
_INJECTION_RE = re.compile(
    r"(忽略|你现在是|你是(一个|名)?(AI|助手|模型)|系统提示|提示词|指令|jailbreak|override"
    r"|prompt|system|ignore|不要(返回|显示|回答)|不许(返回|显示|回答))",
    re.IGNORECASE)

                                            
                                        
                                                   
                 
_RULE_INJECTION_RE = re.compile(
    r"(忽略(之前|上面|以上|前面)|你现在是|你是(一个|名)?(AI|助手|模型)|系统提示|提示词"
    r"|jailbreak|override|prompt|system|ignore)",
    re.IGNORECASE)


class _GateProfile(NamedTuple):
    """一个类目的两道闸参数（**表驱动**：新类目只加一行数据，不再加 if 分支）。"""

    injection_re: re.Pattern[str]
                                                    
    coverage_by_tier: dict[str, int]
    default_tier: str


                
                                            
                                                               
_GATE_PROFILES: dict[str, _GateProfile] = {
    "memory": _GateProfile(_INJECTION_RE, {"": 2}, ""),
    "rule": _GateProfile(_RULE_INJECTION_RE,
                         {DREAM_TIER_EXPLICIT: 1, DREAM_TIER_IMPLICIT: 2},
                         DREAM_TIER_IMPLICIT),
}


def _injection_hit(profile: _GateProfile, fields: Iterable[str]) -> bool:
    return any(profile.injection_re.search(str(f or "")) for f in fields)


def _gate_profile_of(item: dict[str, Any]) -> _GateProfile:
    """条目 → 闸参数：规则条目走规则档，其余走记忆档（判据单一真源在 `_is_rule_item`）。"""
    return _GATE_PROFILES["rule" if _is_rule_item(item) else "memory"]


def _required_coverage(profile: _GateProfile, item: dict[str, Any]) -> int:
    return profile.coverage_by_tier.get(
        str(item.get("tier") or profile.default_tier),
        profile.coverage_by_tier[profile.default_tier])


def _conversation_evidence_pools(conversations: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """可引用原文面：**逐条消息一个 pool 单元**（query 一条、每条消息的 t/n 各一条，归一化
    文本），返回 (对话序号, 单条消息文本) 对——证据必须逐字摘自**单条消息**，「消息 A 尾 +
    消息 B 头」跨消息拼接的伪 evidence 不得通过（E-03：此前整段 join 成一个串，拼接可绕过）。
    对话仍按**内容 hash 去重**——同一段对话重复提交只算一段（防「复制同一段凑两段」绕过
    升级门槛）；k="sys" 的系统消息不算证据面。"""
    pools: list[tuple[int, str]] = []
    seen: set[str] = set()
    for conv_index, conv in enumerate(conversations):
        parts = [str(conv.get("query") or "")]
        for msg in conv.get("chat") or []:
            if not isinstance(msg, dict) or str(msg.get("k") or "") == "sys":
                continue
            parts.append(str(msg.get("t") or ""))
            parts.append(str(msg.get("n") or ""))
        units = [n for n in (_norm(p) for p in parts) if n]
        full = " ".join(units)
        if not full:
            continue
        digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        pools.extend((conv_index, unit) for unit in units)
    return pools


def _evidence_checks(spans: list[str], pools: list[tuple[int, str]]) -> tuple[bool, int]:
    """逐段核验 + 覆盖计数，返回 `(全部 span 合格?, 覆盖的不同对话段数)`。

    - **逐段合格**：每个非空 span 归一后必须**逐字**落在某条消息里——任一段造假即整条
      不合格（旧口径只累计覆盖数：真＋假两段可因真段使 coverage 达标而夹带假段放行）。
    - **覆盖计数**：短于 DREAM_MIN_SPAN_CHARS 的片段照核验但不计覆盖（防「数据」这类
      随处可命中的伪证据凑对话数）。"""
    covered: set[int] = set()
    for span in spans:
        needle = _norm(span)
        if not needle:
            continue
        hit = False
        for conv_index, unit in pools:
            if needle in unit:
                hit = True
                if len(needle) >= DREAM_MIN_SPAN_CHARS:
                    covered.add(conv_index)
        if not hit:
            return False, len(covered)
    return True, len(covered)


def _apply_gates(memories: list[dict[str, Any]],
                 conversations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """两道闸（注入审查 → 出处核验），被拦条目计数上报；任何一道不过都丢弃，不降级。
    2026-09-29 起按**条目类目**取闸参数（`_GATE_PROFILES`）：记忆类目逐位不变，
    规则类目换词表 + 按 tier 要证据（explicit 1 段 / implicit 2 段）。"""
    pools = _conversation_evidence_pools(conversations)
    kept: list[dict[str, Any]] = []
    dropped = {"injection": 0, "evidence": 0}
    for mem in memories:
        profile = _gate_profile_of(mem)
        if _injection_hit(profile, (mem.get("text"), mem.get("summary"), mem.get("trigger"))):
            dropped["injection"] += 1
            continue
        ev_ok, ev_coverage = _evidence_checks(mem.get("evidence") or [], pools)
        if not ev_ok or ev_coverage < _required_coverage(profile, mem):
            dropped["evidence"] += 1
            continue
        kept.append(mem)
    return kept, dropped


def _partition_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """按类目分流：带 tier 的进规则清单，其余进记忆清单（两条生命周期不同：
    记忆写浏览器本机、规则进服务端账本等用户逐条确认）。原始顺序各自保持。"""
    memories = [c for c in candidates if "tier" not in c]
    rules = [c for c in candidates if "tier" in c]
    return memories, rules


def dream_from_conversations(
    conversations: list[dict[str, Any]],
    *,
    config: LLMConfig | None = None,
    llm_call: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """对话快照 → 记忆候选清单 + 协作规则候选清单。`llm_call` 可注入（签名
    `(prompt) -> str | None`），确定性测试据此完全避开网络（与 action_plan 同契约）。
    `count` 恒等于 `memories` 长度（旧语义不动）；规则条数另给 `rules_count`——
    规则**绝不混进 `memories`**（前端记忆 kind 白名单只认 note/search/dream，
    混进去会被静默折算成「已存需求」，正是本批要避开的静默降级）。"""
    convs = _clip_conversations(conversations)
    if not convs:
        raise DreamError("empty_input", "还没有可以整理的历史对话。")
    if llm_call is None:
        try:
            cfg = config or load_llm_config()
        except Exception as exc:                                             
                                                                
            detail = _sanitize_provider_error(exc)[:80]
            raise DreamError("config_error", f"AI 配置读取失败（{detail}）——请检查环境变量配置后重试。") from exc
                                                                              
                                                             
                                                  
                                                           
                           
        ok, reason = _should_use_llm(cfg, require_enabled=False)
        if not ok:
            if reason == "mock_not_used":
                raise DreamError("mock_not_used", "当前是 mock 演示模式，不会真的调用 AI——请关闭 MOCK_LLM 并配好密钥后再整理记忆。")
            raise DreamError("no_key", "还没有配置 AI——先到「设置 → AI / API 配置」填好密钥，再来整理记忆。")
        if not cfg.enable_llm:
            raise DreamError("disabled", "AI 已在设置里关闭——要整理记忆，请先在「设置 → AI / API 配置」开启 AI（或使用网站试用额度）。")
        llm_call = lambda prompt: _default_llm_call(prompt, cfg)              
    raw = llm_call(build_dream_prompt(convs))
    if raw is None:
        raise DreamError("llm_failed", "这次整理没能连上 AI，请稍后再试。")
    candidates = parse_dream_output(raw)
    kept, dropped = _apply_gates(candidates, convs)
    memories, rules = _partition_candidates(kept)
    return {"ok": True, "generated": True, "memories": memories, "rules": rules,
            "count": len(memories), "rules_count": len(rules), "dropped": dropped}
