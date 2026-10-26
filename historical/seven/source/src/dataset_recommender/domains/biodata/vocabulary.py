"""biodata 领域词表（catalog）+ 平台/assay/modality 归一规则。

2026-09-23 骨肉分离批自 `retrieval/vocabulary.py` 迁入本领域包：查询词法（语言级）
留在骨架 `retrieval/query_lexicon.py`；catalog 驱动的通用机制（suggestable_terms /
拼音索引）在 `domain/vocab_tools.py`。本模块只含 biodata 的领域数据与派生规则。

设计要点（与 Codex 辩论收敛结论一致）：
- hard_filter 的可靠性 = 约束抽取的召回率。词表尽量覆盖语料真实词汇 + 常见别名，
  并**收录库中不存在的已知实体**（如 elephant/大象），好让解析器"看见"约束→正确返回无结果。
- platform_family（Visium/Xenium/Chromium/Atera）与 assay（ATAC/Multiome/Flex/CNV/GEX…）
  是两个不同维度，分开建模，不再混用凌乱的 chemistry 串。
"""
from __future__ import annotations

                                                                 
                         
PLATFORM_FAMILY_RULES: list[tuple[str, str]] = [
    ("visium", "visium"),
    ("xenium", "xenium"),
    ("atera", "atera"),
    ("chromium", "chromium"),
]
                                         
ASSAY_RULES: list[tuple[str, str]] = [
    ("multiome", "multiome"),
    ("atac", "atac"),
    ("flex", "flex"),
    ("cnv", "cnv"),
    ("in situ", "in_situ"),
    ("spatial", "spatial_gex"),
    ("immune profiling", "immune"),
    ("de novo", "de_novo"),
    ("genome & exome", "genome_exome"),
    ("5'", "gex_5p"),
    ("3'", "gex_3p"),
    ("gene expression", "gex"),
]


def derive_platform_family(platform_raw: str) -> str:
    text = (platform_raw or "").lower()
    for key, canonical in PLATFORM_FAMILY_RULES:
        if key in text:
            return canonical
    return ""


def derive_assay(chemistry: str) -> str:
    text = (chemistry or "").lower()
    for key, canonical in ASSAY_RULES:
        if key in text:
            return canonical
    return ""


                                                         
                                                             
                                                                                
                                                                                              
                                                                  
                                                                            
                                                                                                
                                                                          
_MODALITY_SPATIAL_MARKERS = (
    "visium", "xenium", "atera", "slide-seq", "slide-seqv2", "stereo-seq", "merfish",
    "seqfish", "starmap", "cosmx", "geomx", "dbit-seq", "hdst", "osmfish", "curio",
    "spatial", "in situ", "in-situ",
)
_MODALITY_SINGLECELL_MARKERS = (
    "single cell", "single-cell", "single cells", "single nucleus", "single-nucleus",
    "snrna", "sn-rna", "snmc", "scrna", "sc-rna", "scatac", "sc-atac",
    "smart-seq", "smartseq", "smart-like", "drop-seq", "dropseq",
    "sci-rna-seq", "sci-rna", "seq-well", "seqwell", "cel-seq", "indrop", "in-drop",
    "mars-seq", "cite-seq", "citeseq", "fluidigm c1", "bd rhapsody",
    "gexscope", "strt-seq", "microwell", "split-seq", "10x", "chromium",
)
_MODALITY_BULK_MARKERS = (
    "rna-seq of coding rna", "rna-seq of non coding rna", "rna-seq of non-coding rna",
    "rna-seq of total rna", "non coding rna", "non-coding rna",
    "transcription profiling by array", "microarray", "chip-seq", "chip seq",
    "dna-seq", "methylation", "bisulfite", "microrna", "rip-seq", "ribo-seq",
    "proteomic", "wgs", "wes", "high-throughput sequencing",
)


def derive_modality(platform_family: str, chem_text: str) -> str:
    """数据模态：'single-cell' / 'spatial' / ''（未知，fail-closed）。单一真源，normalizer 存字段、
    retriever 读存字段、评测裁判从 raw 独立重算三方共用。base 全走 platform_family（确定），外部才扫 chemistry。"""
    pf = (platform_family or "").strip().lower()
    if pf in ("visium", "xenium", "atera"):
        return "spatial"
    if pf == "chromium":
        return "single-cell"
                                                         
    t = (chem_text or "").strip().lower()
    if not t:
        return ""
    if any(m in t for m in _MODALITY_SPATIAL_MARKERS):
        return "spatial"
    if any(m in t for m in _MODALITY_SINGLECELL_MARKERS):
        return "single-cell"
    if any(m in t for m in _MODALITY_BULK_MARKERS):
        return ""                                           
    return ""                                   


                                                             
                                                                       
                                                                                 
                                                                                         
                                                                  
                                                                                              
                                                                                
                                                                                             
                                                                                       
                                                                                          
                                                                                                                      
CATALOG: dict[str, list[dict[str, object]]] = {
    "species": [
        {"aliases": ["人类", "人", "人体", "human", "homo sapiens", "sapiens"], "targets": ["human"], "display": "Human"},
        {"aliases": ["小鼠", "鼠", "老鼠", "mouse", "mice", "murine", "mus musculus", "musculus"], "targets": ["mouse"], "display": "Mouse"},
        {"aliases": ["斑马鱼", "zebrafish", "danio rerio", "danio"], "targets": ["zebrafish"], "display": "Zebrafish"},
        {"aliases": ["大鼠", "rat", "norvegicus", "wistar", "rattus"], "targets": ["rat", "norvegicus"], "display": "Rat"},
        {"aliases": ["恒河猴", "猕猴", "猴", "macaque", "rhesus", "macaca", "mulatta"], "targets": ["macaque", "rhesus"], "display": "Macaque"},
        {"aliases": ["狗", "犬", "dog", "canine", "canis", "canis familiaris"], "targets": ["dog"], "display": "Dog"},
        {"aliases": ["果蝇", "drosophila", "fruit fly", "melanogaster"], "targets": ["drosophila"], "display": "Drosophila"},
                                                                                  
                                                              
        {"aliases": ["拟南芥", "arabidopsis", "thaliana"], "targets": ["arabidopsis", "thaliana"], "display": "Arabidopsis"},
                                                                            
                                                                              
        {"aliases": ["玉米", "maize", "zea mays", "zea"], "targets": ["maize", "zea mays"], "display": "Maize"},
        {"aliases": ["水稻", "oryza sativa", "oryza"], "targets": ["oryza"], "display": "Rice"},
        {"aliases": ["线虫", "秀丽隐杆线虫", "caenorhabditis", "c. elegans", "c.elegans", "elegans", "nematode"], "targets": ["elegans", "caenorhabditis"], "display": "C. elegans"},
        {"aliases": ["酵母", "酿酒酵母", "yeast", "saccharomyces", "cerevisiae"], "targets": ["saccharomyces"], "display": "Yeast"},
        {"aliases": ["狨", "狨猴", "marmoset", "callithrix"], "targets": ["marmoset", "callithrix"], "display": "Marmoset"},
        {"aliases": ["鼠狐猴", "mouse lemur", "microcebus"], "targets": ["microcebus"], "display": "Mouse lemur"},
        {"aliases": ["黑猩猩", "chimpanzee", "troglodytes"], "targets": ["chimpanzee", "troglodytes"], "display": "Chimpanzee"},
                                                                     
                                                                                                      
                                                                                       
                                                                           
                                                                                 
        {"aliases": ["非人灵长类", "非人类灵长类", "非人灵长", "非人猿类",
                     "non-human primate", "non-human primates", "non human primate",
                     "nonhuman primate", "nhp"],
         "targets": ["macaque", "rhesus", "mulatta", "marmoset", "callithrix",
                     "microcebus", "chimpanzee", "troglodytes", "cynomolgus", "fascicularis", "primate"],
         "display": "Non-human Primate"},
        {"aliases": ["爪蟾", "非洲爪蟾", "xenopus"], "targets": ["xenopus"], "display": "Xenopus"},
                                                      
        {"aliases": ["大象", "象", "elephant"], "targets": ["elephant"], "display": "Elephant", "absent": True},
        {"aliases": ["猪", "pig", "porcine", "swine"], "targets": ["pig"], "display": "Pig", "absent": True},
        {"aliases": ["牛", "cattle", "bovine"], "targets": ["cattle", "bovine"], "display": "Cattle", "absent": True},
        {"aliases": ["鸡", "chicken", "avian"], "targets": ["chicken"], "display": "Chicken", "absent": True},
        {"aliases": ["兔", "rabbit"], "targets": ["rabbit"], "display": "Rabbit", "absent": True},
    ],
    "tissue": [
        {"aliases": ["乳腺", "乳房", "breast", "mammary", "mammary gland"], "targets": ["breast"], "display": "Breast"},
        {"aliases": ["肺", "肺部", "lung", "pulmonary"], "targets": ["lung"], "display": "Lung"},
        {"aliases": ["脑", "大脑", "脑组织", "brain", "cortex", "cerebral", "cerebrum"], "targets": ["brain"], "display": "Brain"},
        {"aliases": ["肝", "肝脏", "liver", "hepatic"], "targets": ["liver"], "display": "Liver"},
        {"aliases": ["肾", "肾脏", "kidney", "renal"], "targets": ["kidney"], "display": "Kidney"},
        {"aliases": ["心", "心脏", "heart", "cardiac"], "targets": ["heart"], "display": "Heart"},
        {"aliases": ["血", "血液", "外周血", "blood", "pbmc", "peripheral blood"], "targets": ["blood", "pbmc"], "display": "Blood"},
        {"aliases": ["皮肤", "表皮", "skin", "cutaneous", "dermal", "epidermal", "epidermis", "dermis"], "targets": ["skin"], "display": "Skin"},
        {"aliases": ["骨髓", "bone marrow", "marrow"], "targets": ["bone marrow"], "display": "Bone Marrow"},
        {"aliases": ["卵巢", "ovary", "ovarian"], "targets": ["ovary", "ovarian"], "display": "Ovary"},
        {"aliases": ["淋巴结", "lymph node"], "targets": ["lymph node"], "display": "Lymph Node"},
        {"aliases": ["脾", "脾脏", "spleen", "splenic"], "targets": ["spleen"], "display": "Spleen"},
        {"aliases": ["胰", "胰腺", "pancreas", "pancreatic"], "targets": ["pancreas"], "display": "Pancreas"},
        {"aliases": ["胃", "stomach", "gastric"], "targets": ["stomach", "gastric"], "display": "Stomach"},
        {"aliases": ["前列腺", "prostate", "prostatic"], "targets": ["prostate"], "display": "Prostate"},
        {"aliases": ["结肠", "肠", "colon", "colorectal", "intestine"], "targets": ["colon", "intestin"], "display": "Colon"},
        {"aliases": ["宫颈", "子宫颈", "cervix", "cervical"], "targets": ["cervix", "cervical"], "display": "Cervix"},
        {"aliases": ["胚胎", "胚", "embryo", "embryonic"], "targets": ["embryo"], "display": "Embryo"},
        {"aliases": ["扁桃体", "tonsil", "tonsillar"], "targets": ["tonsil"], "display": "Tonsil"},
        {"aliases": ["肌肉", "骨骼肌", "muscle", "skeletal muscle"], "targets": ["muscle"], "display": "Muscle"},
                                                             
        {"aliases": ["胸腺", "thymus", "thymic"], "targets": ["thymus"], "display": "Thymus"},
        {"aliases": ["小脑组织", "小脑", "cerebellum", "cerebellar"], "targets": ["cerebell"], "display": "Cerebellum"},                                  
        {"aliases": ["海马", "海马体", "hippocampus", "hippocampal"], "targets": ["hippocamp"], "display": "Hippocampus"},
        {"aliases": ["视网膜", "retina", "retinal"], "targets": ["retina"], "display": "Retina"},
        {"aliases": ["眼睛", "眼部"], "targets": ["eye"], "display": "Eye"},                                                                                  
        {"aliases": ["脊髓", "spinal cord", "spinal"], "targets": ["spinal"], "display": "Spinal Cord"},
        {"aliases": ["睾丸", "testis", "testicular"], "targets": ["testis", "testicular"], "display": "Testis"},
        {"aliases": ["回肠", "ileum", "ileal"], "targets": ["ileum"], "display": "Ileum"},
        {"aliases": ["十二指肠", "duodenum", "duodenal"], "targets": ["duodenum"], "display": "Duodenum"},
        {"aliases": ["直肠", "rectum", "rectal"], "targets": ["rectum", "rectal"], "display": "Rectum"},
        {"aliases": ["气管", "trachea", "tracheal"], "targets": ["trachea"], "display": "Trachea"},
        {"aliases": ["支气管", "bronchus", "bronchi", "bronchial"], "targets": ["bronch"], "display": "Bronchus"},
        {"aliases": ["脂肪", "脂肪组织", "adipose", "adipose tissue"], "targets": ["adipose"], "display": "Adipose"},
        {"aliases": ["主动脉", "aorta", "aortic"], "targets": ["aort"], "display": "Aorta"},
        {"aliases": ["子宫", "uterus", "uterine"], "targets": ["uterus", "uterine"], "display": "Uterus"},
        {"aliases": ["甲状腺", "thyroid"], "targets": ["thyroid"], "display": "Thyroid"},
        {"aliases": ["食管", "食道", "esophagus", "esophageal", "oesophagus"], "targets": ["esophag", "oesophag"], "display": "Esophagus"},
        {"aliases": ["胎盘", "placenta", "placental"], "targets": ["placenta"], "display": "Placenta"},
                                                                        
                                                                
                                                         
                                                                        
                                               
                                                                                                
                                                                                                
                                                                           
                                                        
                                                       
        {"aliases": ["膀胱", "bladder", "urinary bladder"], "targets": ["bladder"], "display": "Bladder"},
        {"aliases": ["耳蜗", "内耳", "cochlea", "cochlear"], "targets": ["cochlea"], "display": "Cochlea"},
        {"aliases": ["肾上腺", "adrenal", "adrenal gland"], "targets": ["adrenal"], "display": "Adrenal Gland"},
        {"aliases": ["下丘脑", "hypothalamus", "hypothalamic"], "targets": ["hypothalam"], "display": "Hypothalamus"},
        {"aliases": ["杏仁核", "amygdala"], "targets": ["amygdala"], "display": "Amygdala"},
        {"aliases": ["纹状体", "striatum", "caudate nucleus", "caudate-putamen", "putamen"],
         "targets": ["striatum", "caudate nucleus", "caudate-putamen", "putamen"], "display": "Striatum"},
        {"aliases": ["垂体", "脑垂体", "pituitary"], "targets": ["pituitary"], "display": "Pituitary"},
        {"aliases": ["嗅球", "嗅觉", "olfactory"], "targets": ["olfactory"], "display": "Olfactory"},
        {"aliases": ["角膜", "cornea", "corneal"], "targets": ["cornea"], "display": "Cornea"},
        {"aliases": ["牙龈", "gingiva", "gingival"], "targets": ["gingiva"], "display": "Gingiva"},
        {"aliases": ["口腔", "口腔黏膜", "颊黏膜", "oral cavity", "buccal"],
         "targets": ["oral cavity", "buccal", "mouth"], "display": "Oral Cavity"},
        {"aliases": ["舌头", "舌", "tongue"], "targets": ["tongue"], "display": "Tongue"},
        {"aliases": ["膈肌", "横膈", "diaphragm"], "targets": ["diaphragm"], "display": "Diaphragm"},
        {"aliases": ["腹膜", "腹腔", "peritoneum", "peritoneal"], "targets": ["peritone"], "display": "Peritoneum"},
        {"aliases": ["胸膜", "胸水", "胸腔积液", "pleura", "pleural"], "targets": ["pleura"], "display": "Pleura"},
        {"aliases": ["脐带", "脐带血", "umbilical", "umbilical cord"], "targets": ["umbilical"], "display": "Umbilical Cord"},
        {"aliases": ["输卵管", "fallopian", "fallopian tube", "oviduct"], "targets": ["fallopian", "oviduct"], "display": "Fallopian Tube"},
                                                                     
                                                                   
                                             
        {"aliases": ["子宫内膜", "endometrium"], "targets": ["endometrium"], "display": "Endometrium"},
        {"aliases": ["牙齿", "牙髓", "牙", "tooth", "dental pulp"], "targets": ["tooth", "dental"], "display": "Tooth"},
        {"aliases": ["软骨", "cartilage"], "targets": ["cartilage"], "display": "Cartilage"},
        {"aliases": ["关节", "滑膜", "synovium", "synovial"], "targets": ["synovi", "joint"], "display": "Joint/Synovium"},
        {"aliases": ["血管", "动脉", "静脉", "blood vessel", "vasculature", "artery"],
         "targets": ["blood vessel", "vasculature", "artery", "vein"], "display": "Blood Vessel"},
        {"aliases": ["空肠", "jejunum", "jejunal"], "targets": ["jejunum"], "display": "Jejunum"},
        {"aliases": ["盲肠", "caecum", "cecum"], "targets": ["caecum", "cecum"], "display": "Caecum"},
        {"aliases": ["阑尾", "appendix"], "targets": ["appendix"], "display": "Appendix"},
                                                           
                                    
        {"aliases": ["大脑皮层", "大脑皮质", "皮层", "cortex", "cortical"], "targets": ["cortex"], "display": "Cortex"},
        {"aliases": ["唾液腺", "涎腺", "salivary", "salivary gland"], "targets": ["salivary"], "display": "Salivary Gland"},
        {"aliases": ["胆管", "胆道", "bile duct", "cholangiocyte"], "targets": ["bile duct", "cholangi"], "display": "Bile Duct"},
        {"aliases": ["中脑", "midbrain"], "targets": ["midbrain"], "display": "Midbrain"},
        {"aliases": ["脑干", "后脑", "hindbrain", "brainstem", "brain stem"], "targets": ["hindbrain", "brainstem"], "display": "Brainstem"},
        {"aliases": ["鼻腔", "鼻", "nasal", "nose"], "targets": ["nasal", "nose"], "display": "Nose"},
        {"aliases": ["咽部", "咽", "pharynx", "pharyngeal"], "targets": ["pharyn"], "display": "Pharynx"},
        {"aliases": ["喉部", "喉", "larynx", "laryngeal"], "targets": ["laryn"], "display": "Larynx"},
        {"aliases": ["附睾", "epididymis"], "targets": ["epididymis"], "display": "Epididymis"},
    ],
    "disease": [
        {"aliases": ["乳腺癌", "breast cancer", "idc", "invasive ductal carcinoma", "ductal carcinoma", "tnbc"],
         "targets": ["breast cancer", "invasive ductal carcinoma", "ductal carcinoma"], "display": "Breast Cancer"},
        {"aliases": ["肺癌", "lung cancer", "nsclc", "non-small cell lung", "非小细胞肺癌", "luad", "肺腺癌"],
         "targets": ["lung cancer", "non-small cell lung"], "display": "Lung Cancer"},
        {"aliases": ["肝癌", "liver cancer", "hcc", "hepatocellular"],
         "targets": ["hepatocellular", "liver cancer"], "display": "Liver Cancer"},
                                                        
                                                                                
                                                                          
                                                                                    
                                                           
                                                                                        
        {"aliases": ["淋巴瘤", "lymphoma", "非霍奇金淋巴瘤", "非霍奇金",
                     "non-hodgkin lymphoma", "non-hodgkin"],
         "targets": ["lymphoma"], "display": "Lymphoma"},
                                                                              
                                                                          
        {"aliases": ["霍奇金淋巴瘤", "霍奇金", "hodgkin lymphoma", "hodgkin's lymphoma", "hodgkin"],
         "targets": ["hodgkin"], "display": "Hodgkin's Lymphoma"},
                                                                       
        {"aliases": ["小淋巴细胞淋巴瘤", "small lymphocytic lymphoma"],
         "targets": ["small lymphocytic"], "display": "Small Lymphocytic Lymphoma"},
        {"aliases": ["黑色素瘤", "melanoma"], "targets": ["melanoma"], "display": "Melanoma"},
        {"aliases": ["胶质母细胞瘤", "胶质瘤", "glioblastoma", "glioma", "gbm", "交质母细胞瘤", "交质瘤"], "targets": ["glioblastoma", "glioma"], "display": "Glioblastoma"},
        {"aliases": ["胃癌", "gastric cancer", "gastric adenocarcinoma"], "targets": ["gastric"], "display": "Gastric Cancer"},
                                                                          
                                                                             
        {"aliases": ["白血病", "leukemia", "leukaemia", "cll"], "targets": ["leukemia"], "display": "Leukemia"},
                                                                       
        {"aliases": ["急性髓系白血病", "急性髓细胞白血病", "急性粒细胞白血病", "急髓",
                     "acute myeloid leukemia", "acute myeloid", "aml"],
         "targets": ["acute myeloid"], "display": "Acute Myeloid Leukemia"},
                                                                      
                                                         
        {"aliases": ["急性淋巴细胞白血病", "急性淋巴母细胞白血病", "急淋",
                     "acute lymphoblastic leukemia", "acute lymphoblastic",
                     "acute lymphoid leukemia", "acute lymphoid"],
         "targets": ["acute lymphoblastic", "acute lymphoid"], "display": "Acute Lymphoblastic/Lymphoid Leukemia"},
        {"aliases": ["卵巢癌", "ovarian cancer", "ovarian carcinoma"], "targets": ["ovarian"], "display": "Ovarian Cancer"},
        {"aliases": ["宫颈癌", "cervical cancer"], "targets": ["cervical"], "display": "Cervical Cancer"},
        {"aliases": ["结直肠癌", "结肠癌", "colorectal cancer", "crc"], "targets": ["colorectal", "crc"], "display": "Colorectal Cancer"},
                                          
                                        
        {"aliases": ["子宫内膜癌", "endometrial cancer", "endometrial carcinoma", "endometrial adenocarcinoma"],
         "targets": ["endometrial"], "display": "Endometrial Cancer"},
        {"aliases": ["腺癌", "adenocarcinoma"], "targets": ["adenocarcinoma"], "display": "Adenocarcinoma"},
        {"aliases": ["鳞癌", "鳞状细胞癌", "squamous"], "targets": ["squamous"], "display": "Squamous Carcinoma"},
        {"aliases": ["健康", "正常", "healthy", "normal"], "targets": ["healthy", "normal"], "display": "Healthy/Normal"},
                                                             
        {"aliases": ["新冠", "新冠肺炎", "新型冠状病毒", "covid", "covid-19", "sars-cov-2"], "targets": ["covid"], "display": "COVID-19"},
        {"aliases": ["阿尔茨海默", "阿尔兹海默", "老年痴呆", "alzheimer", "alzheimer's"], "targets": ["alzheimer"], "display": "Alzheimer's Disease"},
        {"aliases": ["痴呆", "失智", "dementia"], "targets": ["dementia"], "display": "Dementia"},
        {"aliases": ["帕金森", "parkinson", "parkinson's"], "targets": ["parkinson"], "display": "Parkinson's Disease"},
        {"aliases": ["多发性硬化", "multiple sclerosis"], "targets": ["multiple sclerosis"], "display": "Multiple Sclerosis"},
        {"aliases": ["肌萎缩侧索硬化", "渐冻症", "amyotrophic lateral sclerosis", "amyotrophic"], "targets": ["amyotrophic"], "display": "ALS"},
        {"aliases": ["癫痫", "epilepsy", "epileptic"], "targets": ["epilepsy"], "display": "Epilepsy"},
        {"aliases": ["糖尿病", "diabetes", "diabetic"], "targets": ["diabet"], "display": "Diabetes"},
        {"aliases": ["心肌梗死", "心梗", "myocardial infarction"], "targets": ["myocardial infarction", "infarction"], "display": "Myocardial Infarction"},
        {"aliases": ["心力衰竭", "心衰", "heart failure"], "targets": ["heart failure"], "display": "Heart Failure"},
        {"aliases": ["动脉粥样硬化", "atherosclerosis", "atherosclerotic"], "targets": ["atherosclero"], "display": "Atherosclerosis"},
        {"aliases": ["克罗恩", "克罗恩病", "crohn", "crohn's"], "targets": ["crohn"], "display": "Crohn's Disease"},
        {"aliases": ["溃疡性结肠炎", "ulcerative colitis", "溃结", "溃样性结肠炎"], "targets": ["ulcerative colitis"], "display": "Ulcerative Colitis"},
        {"aliases": ["炎症性肠病", "inflammatory bowel disease", "ibd"], "targets": ["inflammatory bowel"], "display": "IBD"},
        {"aliases": ["慢阻肺", "慢性阻塞性肺", "chronic obstructive pulmonary", "copd"], "targets": ["chronic obstructive"], "display": "COPD"},
        {"aliases": ["哮喘", "asthma", "asthmatic"], "targets": ["asthma"], "display": "Asthma"},
        {"aliases": ["肺纤维化", "肺间质纤维化", "间质性肺", "pulmonary fibrosis", "interstitial lung"], "targets": ["pulmonary fibrosis", "interstitial lung"], "display": "Pulmonary Fibrosis"},
        {"aliases": ["囊性纤维化", "cystic fibrosis"], "targets": ["cystic fibrosis"], "display": "Cystic Fibrosis"},
        {"aliases": ["银屑病", "牛皮癣", "psoriasis", "psoriatic"], "targets": ["psoriasis", "psoriatic"], "display": "Psoriasis"},
        {"aliases": ["神经母细胞瘤", "neuroblastoma"], "targets": ["neuroblastoma"], "display": "Neuroblastoma"},
                                                                        
                                                       
                                                                     
                                                  
                                         
                                                     
                                                         
        {"aliases": ["纤维化", "fibrosis", "fibrotic"], "targets": ["fibrosis"], "display": "Fibrosis"},
        {"aliases": ["多囊肾病", "多囊肾", "polycystic kidney"], "targets": ["polycystic kidney"], "display": "Polycystic Kidney Disease"},
        {"aliases": ["扩张型心肌病", "肥厚型心肌病", "心肌病", "cardiomyopathy"], "targets": ["cardiomyopathy"], "display": "Cardiomyopathy"},
        {"aliases": ["心肌炎", "myocarditis"], "targets": ["myocarditis"], "display": "Myocarditis"},
        {"aliases": ["高血压", "hypertension", "hypertensive"], "targets": ["hypertens"], "display": "Hypertension"},
        {"aliases": ["肥胖", "obesity", "obese"], "targets": ["obes"], "display": "Obesity"},
        {"aliases": ["非酒精性脂肪肝", "脂肪肝", "fatty liver", "nafld", "steatohepatitis"],
         "targets": ["fatty liver", "steatohepatitis"], "display": "Fatty Liver Disease"},
        {"aliases": ["肝硬化", "cirrhosis", "cirrhotic"], "targets": ["cirrhosis"], "display": "Cirrhosis"},
        {"aliases": ["肝炎", "hepatitis"], "targets": ["hepatitis"], "display": "Hepatitis"},
        {"aliases": ["胰腺癌", "pancreatic cancer", "pancreatic ductal adenocarcinoma", "pdac"],
         "targets": ["pancreatic cancer", "pancreatic ductal"], "display": "Pancreatic Cancer"},
        {"aliases": ["肾细胞癌", "肾癌", "renal cell carcinoma", "kidney cancer", "rcc"],
         "targets": ["renal cell", "kidney cancer"], "display": "Renal Cell Carcinoma"},
        {"aliases": ["膀胱癌", "尿路上皮癌", "bladder cancer", "urothelial carcinoma"],
         "targets": ["bladder cancer", "urothelial"], "display": "Bladder Cancer"},
        {"aliases": ["甲状腺癌", "thyroid cancer", "thyroid carcinoma"],
         "targets": ["thyroid cancer", "thyroid carcinoma"], "display": "Thyroid Cancer"},
        {"aliases": ["多发性骨髓瘤", "骨髓瘤", "multiple myeloma", "myeloma"], "targets": ["myeloma"], "display": "Myeloma"},
        {"aliases": ["骨肉瘤", "肉瘤", "osteosarcoma", "sarcoma"], "targets": ["sarcoma"], "display": "Sarcoma"},
        {"aliases": ["髓母细胞瘤", "medulloblastoma", "髓母"], "targets": ["medulloblastoma"], "display": "Medulloblastoma"},
        {"aliases": ["视网膜母细胞瘤", "retinoblastoma"], "targets": ["retinoblastoma"], "display": "Retinoblastoma"},
        {"aliases": ["间皮瘤", "mesothelioma"], "targets": ["mesothelioma"], "display": "Mesothelioma"},
        {"aliases": ["类风湿关节炎", "类风湿", "rheumatoid"], "targets": ["rheumatoid"], "display": "Rheumatoid Arthritis"},
        {"aliases": ["骨关节炎", "osteoarthritis"], "targets": ["osteoarthritis"], "display": "Osteoarthritis"},
        {"aliases": ["关节炎", "arthritis"], "targets": ["arthritis"], "display": "Arthritis"},
        {"aliases": ["系统性红斑狼疮", "红斑狼疮", "狼疮", "lupus"], "targets": ["lupus"], "display": "Lupus"},
        {"aliases": ["干燥综合征", "舍格伦综合征", "sjogren"], "targets": ["sjogren"], "display": "Sjogren Syndrome"},
        {"aliases": ["肺结核", "结核", "tuberculosis"], "targets": ["tuberculosis"], "display": "Tuberculosis"},
        {"aliases": ["艾滋病", "hiv"], "targets": ["hiv"], "display": "HIV"},
        {"aliases": ["流行性感冒", "流感", "influenza"], "targets": ["influenza"], "display": "Influenza"},
        {"aliases": ["脓毒症", "败血症", "sepsis", "septic"], "targets": ["sepsis"], "display": "Sepsis"},
        {"aliases": ["唐氏综合征", "down syndrome"], "targets": ["down syndrome"], "display": "Down Syndrome"},
                                                              
                                                       
        {"aliases": ["杜氏肌营养不良", "肌营养不良", "muscular dystrophy"], "targets": ["dystrophy"], "display": "Muscular Dystrophy"},
        {"aliases": ["青光眼", "glaucoma"], "targets": ["glaucoma"], "display": "Glaucoma"},
        {"aliases": ["老年黄斑变性", "黄斑变性", "macular degeneration"], "targets": ["macular degeneration"], "display": "Macular Degeneration"},
        {"aliases": ["白内障", "cataract"], "targets": ["cataract"], "display": "Cataract"},
        {"aliases": ["过敏", "变态反应", "allergy", "allergic", "hypersensitivity"],
         "targets": ["allerg", "hypersensitiv"], "display": "Allergy"},
        {"aliases": ["特应性皮炎", "湿疹", "皮炎", "dermatitis", "eczema"],
         "targets": ["dermatitis", "eczema"], "display": "Dermatitis/Eczema"},
        {"aliases": ["脊髓损伤", "spinal cord injury"], "targets": ["spinal cord injury"], "display": "Spinal Cord Injury"},
        {"aliases": ["脑卒中", "脑梗死", "中风", "卒中", "脑梗", "stroke"], "targets": ["stroke"], "display": "Stroke"},
        {"aliases": ["先天性心脏病", "先心病", "congenital heart"], "targets": ["congenital heart"], "display": "Congenital Heart Disease"},
        {"aliases": ["胃炎", "gastritis"], "targets": ["gastritis"], "display": "Gastritis"},
        {"aliases": ["巨细胞病毒", "cytomegalovirus", "cmv"], "targets": ["cytomegalovirus"], "display": "Cytomegalovirus Infection"},
        {"aliases": ["急性肾损伤", "acute kidney injury"], "targets": ["acute kidney injury"], "display": "Acute Kidney Injury"},
        {"aliases": ["慢性肾脏病", "慢性肾病", "chronic kidney disease", "慢姓肾病", "慢肾"], "targets": ["chronic kidney disease"], "display": "Chronic Kidney Disease"},
        {"aliases": ["肾小球肾炎", "肾病", "肾炎", "nephropathy", "nephritis", "glomerulonephritis"],
         "targets": ["nephropathy", "nephritis"], "display": "Nephropathy/Nephritis"},
        {"aliases": ["肺炎", "pneumonia"], "targets": ["pneumonia"], "display": "Pneumonia"},
                                                                      
                                                             
                                                                
                                    
        {"aliases": ["血管瘤", "肝血管瘤", "hemangioma"], "targets": ["hemangioma"], "display": "Hemangioma"},
                                                        
                                                               
                                                         
                                                                  
        {"aliases": ["非整倍体", "aneuploidy", "aneuploid"], "targets": ["aneuploid"], "display": "Aneuploidy"},
        {"aliases": ["巴雷特食管", "barrett"], "targets": ["barrett"], "display": "Barrett Esophagus"},
        {"aliases": ["头颈部鳞癌", "头颈鳞癌", "head and neck"], "targets": ["head and neck"], "display": "Head and Neck Carcinoma"},
        {"aliases": ["周围神经病", "神经病变", "neuropathy"], "targets": ["neuropathy"], "display": "Neuropathy"},
        {"aliases": ["主动脉狭窄", "aortic stenosis"], "targets": ["aortic stenosis"], "display": "Aortic Stenosis"},
        {"aliases": ["心房颤动", "房颤", "atrial fibrillation"], "targets": ["atrial fibrillation"], "display": "Atrial Fibrillation"},
        {"aliases": ["贫血", "anemia", "anaemia"], "targets": ["anemia", "anaemia"], "display": "Anemia"},
        {"aliases": ["深静脉血栓", "血栓", "thrombosis"], "targets": ["thrombosis"], "display": "Thrombosis"},
        {"aliases": ["肺气肿", "emphysema"], "targets": ["emphysema"], "display": "Emphysema"},
        {"aliases": ["脑出血", "出血", "hemorrhage", "haemorrhage"], "targets": ["hemorrhage", "haemorrhage"], "display": "Hemorrhage"},
        {"aliases": ["癌", "肿瘤", "cancer", "tumor", "tumour", "carcinoma", "neoplasm"],
         "targets": ["cancer", "carcinoma"], "display": "Cancer (generic)", "generic": True},
    ],
                                    
    "platform": [
        {"aliases": ["空间转录组", "空间", "visium", "spatial transcriptomics", "spatial"], "targets": ["visium"], "display": "Visium"},
        {"aliases": ["xenium", "原位", "in situ"], "targets": ["xenium"], "display": "Xenium"},
        {"aliases": ["chromium"], "targets": ["chromium"], "display": "Chromium (platform)"},
        {"aliases": ["atera"], "targets": ["atera"], "display": "Atera", "absent_ok": True},
    ],
                                                            
                                                                                 
                                                                        
    "modality": [
                                                           
                                                        
                                                       
        {"aliases": ["单细胞", "单细胞测序", "单细胞转录组", "单核", "单核测序",
                     "single cell", "single-cell", "single nucleus", "single-nucleus",
                     "snrna", "snrna-seq", "scrna", "scrna-seq", "sc-rna-seq",
                     "scrnaseq", "snrnaseq",
                                                              
                     "单细包", "单系胞", "danxibao"],
         "targets": ["single-cell"], "display": "Single-cell"},
    ],
                          
    "assay": [
                                                               
                                                                    
                                              
        {"aliases": ["atac", "scatac", "sc-atac", "snatac", "sn-atac", "atac-seq",
                     "染色质", "可及性", "chromatin accessibility"], "targets": ["atac"], "display": "ATAC"},
        {"aliases": ["multiome", "多组学"], "targets": ["multiome"], "display": "Multiome"},
        {"aliases": ["flex", "固定rna", "fixed rna"], "targets": ["flex"], "display": "Flex"},
        {"aliases": ["cnv", "拷贝数"], "targets": ["cnv"], "display": "CNV"},
                                                                             
                                                        
                                                                                                
                                                                             
                                                                                
        {"aliases": ["免疫组库", "免疫受体库", "免疫受体", "immune profiling", "immune repertoire",
                     "vdj", "v(d)j", "v（d）j", "tcr", "bcr"],
         "targets": ["immune", "v(d)j"], "display": "Immune Profiling"},
                                                                  
                                                               
        {"aliases": ["smart-seq", "smart-seq2", "smartseq", "smartseq2", "smart-like"], "targets": ["smart"], "display": "Smart-seq"},
        {"aliases": ["drop-seq", "dropseq"], "targets": ["drop-seq"], "display": "Drop-seq"},
        {"aliases": ["slide-seq", "slide-seqv2", "slideseq"], "targets": ["slide-seq"], "display": "Slide-seq"},
        {"aliases": ["sci-rna-seq", "sci-rna-seq3"], "targets": ["sci-rna-seq"], "display": "sci-RNA-seq"},
        {"aliases": ["merfish"], "targets": ["merfish"], "display": "MERFISH"},
        {"aliases": ["seq-well", "seqwell"], "targets": ["seq-well"], "display": "Seq-Well"},
        {"aliases": ["cite-seq", "citeseq"], "targets": ["cite-seq"], "display": "CITE-seq"},
                                                           
                                                             
                                                   
                      
        {"aliases": ["indrop", "in-drop"], "targets": ["indrop"], "display": "inDrop", "absent_ok": True},
        {"aliases": ["bd rhapsody", "rhapsody"], "targets": ["rhapsody"], "display": "BD Rhapsody", "absent_ok": True},
        {"aliases": ["evercode", "parse biosciences"], "targets": ["evercode"], "display": "Parse Evercode", "absent_ok": True},
        {"aliases": ["stereo-seq", "stereoseq"], "targets": ["stereo-seq"], "display": "Stereo-seq", "absent_ok": True},
        {"aliases": ["cosmx"], "targets": ["cosmx"], "display": "CosMx", "absent_ok": True},
        {"aliases": ["geomx"], "targets": ["geomx"], "display": "GeoMx", "absent_ok": True},
        {"aliases": ["fluidigm c1", "fluidigm"], "targets": ["fluidigm"], "display": "Fluidigm C1", "absent_ok": True},
        {"aliases": ["mars-seq", "marsseq"], "targets": ["mars-seq"], "display": "MARS-seq", "absent_ok": True},
        {"aliases": ["strt-seq"], "targets": ["strt-seq"], "display": "STRT-seq", "absent_ok": True},
        {"aliases": ["split-seq", "splitseq"], "targets": ["split-seq"], "display": "Split-seq", "absent_ok": True},
        {"aliases": ["cellplex"], "targets": ["cellplex"], "display": "CellPlex", "absent_ok": True},
                                                                   
                                                                            
                                                                  
        {"aliases": ["codex"], "targets": ["codex"], "display": "CODEX", "absent_ok": True},
        {"aliases": ["mibi"], "targets": ["mibi"], "display": "MIBI", "absent_ok": True},
        {"aliases": ["phenocycler"], "targets": ["phenocycler"], "display": "PhenoCycler", "absent_ok": True},
        {"aliases": ["seqfish"], "targets": ["seqfish"], "display": "seqFISH", "absent_ok": True},
        {"aliases": ["snare-seq", "snare-seq2", "snareseq2"], "targets": ["snare-seq"], "display": "SNARE-seq2", "absent_ok": True},
        {"aliases": ["cell dive"], "targets": ["cell dive"], "display": "Cell DIVE", "absent_ok": True},
        {"aliases": ["dbit"], "targets": ["dbit"], "display": "DBiT", "absent_ok": True},
        {"aliases": ["hifi-slide", "hifislide"], "targets": ["hifi-slide"], "display": "HiFi-Slide", "absent_ok": True},
        {"aliases": ["pixel-seq", "pixel-seqv2", "pixelseq"], "targets": ["pixel-seq"], "display": "Pixel-seqV2", "absent_ok": True},
        {"aliases": ["molecular cartography"], "targets": ["molecular cartography"], "display": "Molecular Cartography", "absent_ok": True},
                                                                     
                                                                         
                                                                       
                                                     
                                                                
                                                                           
                                                           
                                  
        {"aliases": ["perturb-seq", "perturbseq"], "targets": ["perturb-seq"], "display": "Perturb-seq"},
        {"aliases": ["snuc-seq", "snucseq"], "targets": ["snuc-seq"], "display": "sNuc-Seq"},
        {"aliases": ["div-seq"], "targets": ["div-seq"], "display": "Div-Seq"},
        {"aliases": ["dronc-seq"], "targets": ["dronc-seq"], "display": "DRonC-seq"},
        {"aliases": ["pick-seq"], "targets": ["pick-seq"], "display": "Pick-seq"},
    ],
}
