"""查询词法（中文/英文自然语言理解的语用语言学词表，骨架默认）。

原属 `retrieval/vocabulary.py`；2026-09-23 骨肉分离批拆出：领域数据（CATALOG/派生规则）
迁入 `domains/biodata/`，本文件保留语言级词法——否定/或/填充词/raw 说法/动作词等，
对任意领域包生效（换领域不换中文）。块内常量与旧 vocabulary.py 逐字一致。
"""
from __future__ import annotations

RAW_REQUIRED_ALIASES = [
    "fastq", "原始数据", "有原始数据", "raw data", "from fastq",
    "可重新跑", "重新跑流程", "重新比对", "重新分析", "可复现", "重跑",
                                                          
                                                                 
                                                
    "原始文件", "原始序列", "原始测序数据", "原始 fastq", "原始fastq",
    "raw file", "raw files", "raw fastq", "raw sequencing",
]
RAW_NOT_REQUIRED_ALIASES = ["无需fastq", "不需要fastq", "不需要原始数据", "不要原始数据", "无需原始数据"]

                                               
                                                         
 
                                                        
                                                                                     
                                                    
 
                                                                 
                                                     
                                                         
                         
                                                               
                                                                  
                                                                   
                                                                  
                             
                                             
                                            
                                             
                         
 
             
                                              
                                                
                                                                          
                                                       
 
                                              
                                                          
                                                          
                                                         
                            
                                                  
                           
                                                 
ACTION_VERBS = (
    "打包", "打个包", "生成包", "做成包", "整理成",
    "下载脚本", "下载链接", "批量下载", "下载", "拉取", "抓取",
    "导出", "download script", "export",
)

ACTION_NOUNS = (
    "引文", "参考文献", "引用格式", "清单", "列表文件", "文件清单",
    "脚本", "批量", "manifest", "citation",
)

                                               
                                                         
ACTION_MARKERS = tuple(dict.fromkeys(ACTION_VERBS + ACTION_NOUNS))

                                    
                                                                 
                                                
                                            
                                         
CURATE_OP_MARKERS = (
    "导入", "上传",
    "删除", "删掉", "回收站", "撤销删除", "找回",
    "联网搜", "在线搜", "上网搜", "网上搜",
    "检查更新", "有没有更新", "是否有更新", "有更新吗",
    "清点", "汇报",
)

FILLER_GRAMMAR = [
    "推荐", "帮我", "给我", "我想", "想要", "需要", "找", "查", "搜索", "搜", "检索",
    "一些", "一点", "点", "些", "的", "了", "吗", "呢", "吧", "呀", "啊", "和", "与",
    "数据", "数据集", "数据库", "样本", "样品", "相关", "关于", "看看", "看", "有关",
    "请", "麻烦", "谢谢", "以及", "还有", "这", "那", "个", "条", "份",
    "方面", "类", "type", "dataset", "datasets", "data", "please", "show", "me",
    "the", "a", "an", "of", "for", "with", "give", "find", "want", "some", "about", "recommend",
                                   
    "组织", "平台", "可以", "测序", "转录组", "图谱", "细胞", "基因", "表达",
    "项目", "实验", "分析", "结果", "文件", "序列", "谱", "库", "技术", "方案",
                                                        
                                                                    
                                           
    "是", "有", "的话", "这个", "那个", "我", "你",
                                       
    "经典", "典型", "常见", "最新", "高质量", "优质", "标准", "好", "好的", "优秀",
    "一份", "几个", "若干", "各种", "各类", "所有", "全部", "更多", "经典的",
    "帮忙", "看一下", "了解", "关注", "研究", "做", "用", "适合", "合适", "常用",
                                                                     
                                                         
                                                    
                                                    
                                                            
                                                   
                                                    
                                                                      
    "来自", "出自", "源自", "源于", "来源于", "来源", "产自", "取自", "选自",
    "收录", "发布", "官方", "提供", "上传",
    "针对", "面向", "基于", "根据", "按照", "依据", "涉及", "通过", "筛选", "限定",
    "跑出来", "跑出", "生成", "产出", "得到", "做出", "测得", "处理", "处理过", "整理",
    "一批", "一套", "一系列", "大概", "大约", "差不多", "尽可能", "越多越好", "多一些",
                                                               
                                         
                                               
                                                          
                           
    "比较", "想比较", "都想", "出来", "公开", "公开可", "开放", "状态", "什么", "随便什么",
    "之前", "编号", "登录号", "哪里", "资源", "一起", "搞到", "链接", "现成", "部位",
    "召回", "匹配", "变化", "模型", "and", "metadata", "shuju", "shujv", "tcga",
                                                         
                                                 
                                                          
            
    "老板", "师兄", "师姐", "导师", "老师", "让", "组会", "毕设", "开题", "课题", "课题组",
    "救命", "微信", "朋友", "同学", "下午", "明天", "一会儿", "咱们", "群里", "信息",
    "化学", "试剂", "版本", "流程", "管线", "组学",
                                                             
                                                 
    "from", "via", "using", "used", "based", "generated", "produced", "made",
    "obtained", "derived", "hosted", "published", "available", "related",
    "source", "sources", "sequencing", "library", "libraries", "sample", "samples",
    "study", "studies", "atlas", "project", "projects", "platform", "technology",
    "assay", "experiment", "experiments", "cell", "cells", "any", "all", "more",
    "looking", "need", "list", "get", "top",
                                              
                                                           
                                                         
    "integrated", "curated", "annotated", "processed", "aggregated", "combined",
    "merged", "harmonized", "reference", "public", "collection", "resource", "database",
                                                          
                                                          
                                        
    "发表", "已发表", "出版", "汇总", "覆盖",
    "要", "原始", "联合", "结合", "配套", "同时", "另外", "此外", "其中", "包括", "包含",
    "published in", "reported", "deposited", "submitted", "released",
                                                               
                                     
                                                      
    "rna", "dna", "mrna", "cdna", "seq", "omics",
                                                                 
                                               
                                             
                                                     
                                                
                                                       
                                                                     
                   
                                                            
                                                             
                                                     
                                         
                                                                        
                                                     
                                                               
    "一下", "感染", "整合", "整合分析", "genomics",
                                                             
                                                                              
                             
                                                       
                                          
                                                             
                                                   
                                                 
                                      
    "或", "或者", "最好", "尽量", "如果可以", "可以的话", "如果有", "希望", "如果",
    "preferably", "ideally", "possible",
                                                                        
                                           
                                                              
                                         
                                                            
                                                               
                                                                   
                                                    
                                                 
                                                    
                                                      
                                                                        
                                                                      
                                                             
                                                           
                                                  
    "都行", "就是", "下到", "得", "里头", "能", "检测",
]
                              
                                                 
                                                   
                                                          
FILLER_GRAMMAR += [m for m in ACTION_MARKERS if m not in set(FILLER_GRAMMAR)]
                                                                   
                                                           
FILLER_DOMAIN = [
    "患者", "病人", "受试者", "供体", "捐赠者", "免疫",
    "成人", "成年", "儿童", "婴儿", "胎儿", "新生儿", "青少年", "老年", "年轻",
    "男性", "女性", "雄性", "雌性", "男", "女",
                                                           
                                                        
                                                   
                                                    
                                                        
                                                     
                           
    "神经元", "巨噬", "巨噬细胞", "上皮", "内皮", "成纤维", "间质", "基质",
                                                       
                         
    "tme", "ctc", "ctdna",
    "髓系", "淋巴", "树突", "小胶质", "星形胶质", "浆细胞", "祖细胞",
                                   
    "对照", "对照组", "野生型", "野生", "敲除", "过表达", "治疗前", "治疗后", "治疗",
    "用药", "化疗", "放疗", "复发", "转移", "转移性", "原发", "分期",
    "早期", "中期", "晚期", "进展期", "急性", "慢性",
                                         
    "高深度", "深度", "万级", "千级", "十万", "大规模", "小规模", "高分辨率", "低分辨率",
                                 
    "新鲜", "冷冻", "冻存", "石蜡", "ffpe", "活检", "穿刺", "尸检", "手术", "术后",
                                                      
    "表达矩阵", "计数矩阵", "矩阵", "注释", "counts", "loom", "barcode", "matrix",
    "ranger", "cellranger", "seurat", "scanpy",
                                                     
                                           
                                       
    "微环境", "肿瘤微环境", "免疫微环境", "浸润", "免疫浸润",
    "发育", "发育时序", "时序", "拟时序", "轨迹", "分化", "再生", "衰老", "稳态",
    "谱系", "亚型", "异质性", "多样性", "可塑性", "极化", "激活",
    "类器官", "器官", "原代", "细胞系", "共培养", "培养",
                                                      
                                                
    "毛细胞", "单核细胞", "干细胞", "免疫细胞", "肿瘤细胞", "基质细胞", "间充质",
    "成体", "胚系",
                                                            
                                                 
                                                                
                                                           
                                            
    "特发性", "原发性", "继发性", "获得性", "遗传性", "家族性",
]

                                              
                                                           
                                                                     
                                                         
                                                         
 
                                                                       
ALIAS_PROTECTED_COMPOUNDS = (
    "单核细胞",                                             
    "胸腺嘧啶",                            
    "血管紧张素",                                           
    "胰岛素",                               
                                                        
                                     
    "肾上腺素",                                
    "去甲肾上腺素",                              
    "肝素",                          
    "胸腺肽",                          
    "甲状腺素",                          
                                                             
                                                   
                                          
    "咽炎", "喉炎",
)
                                                       
                                                 
                                                                                        
FILLER_TOKENS = FILLER_GRAMMAR + FILLER_DOMAIN

                                              
                                                                       
                                                                           
                                                 
                                                                          
                                                                
                                                   
                                                             
                                                             
                                                   
                              
OR_MARKERS = ["或者", "或", " or "]

                                               
                                                       
                                       
 
              
                               
                                                         
                                             
                                                                     
           
 
                                              
                            
                                                
                                    
HEDGE_PREFER_PREFIX_CN = ("最好", "尽量", "如果可以", "可以的话", "如果有",
                          "preferably", "ideally", "if possible")

                                  
                                                      
                                                    
                                                        
                                                
                                   
SOFT_PREFER_PREFIX_CN = ("优先考虑", "优先选择", "优先", "偏向", "倾向于", "倾向")

                                                    
PREFER_CONNECTOR_CHARS = "是用选取要有带的"
                                                     
                                           
                                              
HEDGE_CONNECTOR_CHARS = "是用选取要有带"
                                                               
                                              
PREFER_PREFIXES_ALL = tuple(SOFT_PREFER_PREFIX_CN) + tuple(HEDGE_PREFER_PREFIX_CN)

                                                                              
                                               
                                                                              
                                                     
                                                 
                                                                              

                                                       
RAW_TERMS = ("fastq", "原始数据", "raw data")

                            
EXEC_NEG_PREFIX_CN = (
    "不要", "没有", "没", "无",
    "不带", "不含", "不包含", "不包括",
    "排除", "剔除", "去除", "去掉", "排掉",
    "拒绝",
                                                         
                                                     
    "别带",
)
                                   
                                                                
                                                            
                                                     
                                                                      
                                                                     
EXEC_NEG_PREFIX_EN = ("no", "not", "without", "exclude", "excluding", "free of")
                                           
EXEC_NEG_CIRCUMFIX_CN = (("除了", "以外"), ("除了", "之外"), ("除", "以外"), ("除", "之外"))
             
                                                      
                                                     
                                                   
EXEC_NEG_SUFFIX_CN = ("除外", "的不要", "的就不用给了")
                                         
                                                   
                                      
                                            
                           
LIST_CONNECTORS = ("和", "与", "及", "以及", "、", "或者", "或",
                                                                            
                                                              
                   " or ", " and ")

                                     
NEGATION_GUARDS_CN = (
    "不是", "并非", "并不是", "不排除", "未排除", "不能排除",
    "不用", "不必", "无须", "无需", "不需要", "没必要", "没有必要",
    "不存在", "并无", "并没有", "不涉及",
    "禁止", "避免", "避开", "绕开",
    "删掉", "移除", "过滤掉", "拒收",
                                                 
    "不", "没", "无", "非", "未", "勿", "莫", "别", "否", "免",
)
NEGATION_GUARDS_EN = (
    "not", "never", "except", "excepting", "other than",
    "avoid", "omit", "omitting", "skip", "reject", "remove",
    "free of", "lacking", "minus", "neither", "nor",
    "don't", "do not", "needn't", "need not", "not required", "non",
)

                                                
                                                        
                                                                     
                                                                        
NEG_MORPHEMES_CN = tuple(sorted(set(EXEC_NEG_PREFIX_CN) | set(NEGATION_GUARDS_CN), key=len, reverse=True))
NEG_MORPHEMES_EN = tuple(sorted(set(EXEC_NEG_PREFIX_EN) | set(NEGATION_GUARDS_EN), key=len, reverse=True))
                                                                 
                                                                        
                                                              
                                                                     
                                                                     
NEGATION_EXEMPT_COMPOUNDS = (
    "非编码rna", "长非编码rna", "非编码",
    "non-coding rna", "non coding rna", "non-coding", "non coding",
)
                                              
NEG_EXISTENTIAL_PHRASES = ("有没有", "有无", "是否有", "是否包含", "是否含", "是否有的")
                      
NEG_CONDITIONAL_MARKERS = ("如果", "假如", "要是", "除非", "若")
                  
NEG_INTERROGATIVE_MARKERS = ("是否排除", "要不要排除", "是否要排除", "该不该排除")

import re as _re

                                                           
RAW_FORBIDDEN_PATTERNS = tuple(_re.compile(p) for p in (
    r"(?:不要|没有|没|无|不带|不含|不包含|不包括|排除|剔除|去除|去掉|排掉|拒绝)\s*(?:fastq|原始数据)",
    r"(?:fastq|原始数据)\s*除外",
    r"(?:除了|除)\s*(?:fastq|原始数据)\s*(?:以外|之外)",
    r"(?<![a-z0-9_])no(?:\s+|-)(?:fastq|raw(?:\s+|-)data)(?![a-z0-9_])",
                                                                     
                                                                     
                                                         
                                                                             
    r"(?<![a-z0-9_])(?:without|exclude|excluding|not|free\s+of)\s+"
    r"(?:fastq|raw(?:\s+|-)data)(?![a-z0-9_])",
))
                                                   
RAW_OPTIONAL_PATTERNS = tuple((_re.compile(p), action) for p, action in (
    (r"(?:不需要|无需|无须|不用|不必|没必要)\s*(?:fastq|原始数据)", "clarify"),
    (r"(?:fastq|原始数据)\s*(?:不需要|不是必需|非必需)", "clarify"),
    (r"(?<![a-z0-9_])(?:don't need|do not need|need not)\s+(?:fastq|raw data)", "clarify"),
    (r"(?:fastq|raw data)\s+(?:not required|not necessary)", "clarify"),
    (r"(?:fastq|原始数据)\s*可有可无", "drop_constraint"),
    (r"(?:对\s*)?(?:fastq|原始数据)\s*(?:没有要求|不作要求)", "drop_constraint"),
    (r"with or without\s+(?:fastq|raw data)", "drop_constraint"),
    (r"(?:fastq|raw data)\s+optional", "drop_constraint"),
    (r"no requirement for\s+(?:fastq|raw data)", "drop_constraint"),
))
