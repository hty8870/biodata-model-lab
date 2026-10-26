"""Build the Chinese report from immutable metrics; no model calls."""
import argparse,json,re
from pathlib import Path
import numpy as np

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def f(x):return '—' if x is None else f'{x:.9f}'
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+'\n'.join('|'+ '|'.join(str(v) for v in row)+'|' for row in rows)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root
    if a.out.exists():raise SystemExit('report exists')
    R=root/'research/memory_relations';base=R/'runs/head-design-v5-20261013-r4'; data=load(base/'formal-analysis-final/final-analysis.json');dual=load(base/'formal-analysis-final/A-dual-hard.json');dev=load(base/'formal-dev-results.json');maths=load(base/'mathematical-checks-final.json');qc=load(root/'artifacts/head-design-full-r4.json');summary=load(base/'formal-study/finalize/summary.json')
    assert qc['status']=='passed' and len(qc['gates'])==14 and all(g['status']=='passed' for g in qc['gates'])
    commands=[c for g in qc['gates'] for c in g['commands']];assert len(commands)==73 and all(c['exit_code']==0 for c in commands)
    leaves=['equivalent','old_entails_new_strict','new_entails_old_strict','disjoint','overlap_incomparable']; parts=[]
    def add(x):parts.append(x.strip()+'\n')
    def section(n,title,text):add(f'<a id="s{n}"></a>\n\n## {n}. {title}\n\n{text}')
    add('''# 语义关系与记忆编辑解耦：轻量分类头的结构分析与有限集合对照

本报告交付受控有限集合研究的完整方法、反例、试错记录、冻结dev与一次正式final、质量证据和复现资料。它不是论文投稿稿、不是人类金标实验、不是自然语言头训练结果、不是生产部署或真实用户收益证明。整个研究已受N8 pilot与独立评审启发；准确表述是“修订冻结设计在此前未评分的N10 pool上评价”。

**阅读导航**：设计/为何否决混轴七类看[1–4](#s1)；最小概率表示与候选看[5–6](#s5)；数据/预算/迭代看[7–9](#s7)；实际dev/final与A强基线看[10–11](#s10)；分布偏移/交换/校准看[12–13](#s12)；独立数学容量证据看[14](#s14)；两轮评审及质量看[15–16](#s15)；未来NL/复用与交付边界看[17–18](#s17)。数值表只读冻结产物，不反馈训练。
''')
    section(1,'问题背景与研究问题','''记忆维护需要处理新增、补充、重复、冲突、替代、失效、作用域迁移与引用保存。但“语义联系是什么”“是否有权改”“怎样改”“结果是否成功”不是同一个问题。一个小分类头不能单独裁决整套维护事务，更不应从最终类别直接触发不可逆动作。

本研究只检验两个可审计问题：第一，在非空可满足、解释域一致的对象对上，五叶联合概率表示是否消除了单向或双向粗边缘的信息缺口；第二，在固定原始bitmap输入、共同骨干和实际优化预算下，A/B/C/H的粗/细监督与参数化表现怎样。前者有代数与反例证据，后者只有本次条件分区的学习成绩，两类证据不能互相替代。

BioData Agent的生产用途仍是公开单细胞数据集元数据目录。本研究不接生产，不处理序列或实验方案，没有改动冻结784语料、推荐器、网页或真实记忆提交链。旧公开NLI目标继续暂停，也没有读取旧封存题或七类锁定权重。''')
    section(2,'为什么推翻七类作为单一flat语义标签','''原七类 duplicate/conflict/complement/contextual/supersede/temporary/uncertain 混合至少四根轴。duplicate和部分conflict是相对内容关系；contextual与temporary带适用域/生命周期；supersede是维护意图与替代事务；uncertain是知识状态。一个条目可以同时临时、重复、拟替代，并且需要未知解析状态。把这些压成互斥标签会丢信息，也会让业务类别随权限/时间变化而漂移。

因此否决的是“七类充当唯一、排他的语义真相并直接映动作”，不是否认七类业务词汇有用。它们可以保留在显示、授权、生命周期或事件面，前提是逻辑真源分离。模型不再猜已知project/version/TTL/权限等可信元数据，不把补充和替代当成集合关系标签。旧七类模型、校准温度或1%风险证据也不因此获得向新任务迁移的资格。

五叶不是通过改名扩成另一套七类，而是明确限定数学问题。自然语言真实任务可能有软偏好、模态、量词、程序、示例或不明确绑定；这些不是五叶bitmap成功即可覆盖的内容。需要另行定义输入解释契约、真人标注与实际编辑验证。''')
    section(3,'职责分层：语义、绑定、授权、执行分开','''建议保留四层。①确定性可信层处理对象ID/版本、项目作用域、时间、来源、引用与可改字段，能读取的元数据不交小模型猜。②解释/定位层把内容分成实际绑定子句，保留type、modal、quantifier、frame、time、conditions、payload和evidence，并标明覆盖与歧义。③语义层仅对同一绑定、共同解释域的一对对象输出五叶或单列unresolved，概率表示只承诺这一对。④维护层依据用户授权、完整变更集、依赖/引用保存、并发版本与读写契约生成/审核/执行CRUD，再独立验证结果。

一个overlap标签不告诉系统公共子句在哪里。P=A∧B,H=A∧C与P=A∧B,H=C∧D都可为o，只有前者有显式可保留A。对齐和定位要由上游绑定证据提供。一个new-entails-old也不证明有权删除原文，更不能当操作成功回执。

多对象知识库一致性不能从逐对标签多样性推出。相对P={1,2,3}的H1={1,2,3},H2={1,2},H3={1,4}有等价/严格包含/相交三类，但联合仍含1；H1={1,2},H2={2,3},H3={1,3}对同参照均同类，联合却为空。r2的标签混类⇒库冲突论证被r3否决；r4完整AND/core才检查抽象联合满足性。''')
    section(4,'CRUD相对完备性与编辑oracle的准确含义','''在有限、已定位、可版本化对象集合的假设下，任何目标库快照相对原库的差异都可分解为创建新增对象、更新保留对象、删除/撤回消失对象；作用域move是受检查的字段更新，替代是有出处的create+retract原子组合。这里的“相对完备”是状态表达能力，不是证明这些动作都有授权、解释正确、原子成功或覆盖现实用户意图。

研究编辑oracle不复制生产提交状态机。对象提供mask、guard、cohort、frame、binding、version和依赖/证据。断言真集为¬guard∨claim；完整cohort先AND，随后枚举最小不可满足core。有限域最小core至多Nworld成员，默认搜索min(n,Nworld)，研究资源上限每cohort16成员/世界，超过或人为压低cap时显式unsupported/incomplete/review，而不输出正常。

四元反例U4上的14/13/11/7：任意三者AND非空，四者AND为空，原cap3必漏。r4已检出。core身份绑定真实语义/guard/frame/cohort而非只ID/version，同版本改mask或跨cohort不能假retained；changed单列review，无新增冲突证书始终不颁。frame15→31迁移在当前合同中拒绝。模板/例子不自动合取。

关系回执必须与实际对象内容及上下文绑定，content/full_conditional/overlap三个模式分别求oracle，假payload、过期/同版本换mask、cross-frame拒绝。完整条件句与适用域内容不等价：P=(S⇒A),H=(S⇒¬A)在全域可能相交，在S内内容不相容。没有清楚模式就不能把两种gold混为同一标签。''')
    section(5,'语义域、五叶、M与双向不可识别','''给定共同有限解释域U，P,H为非空可满足集合：e=P=H；f=P严格包含于H（old entails new）；r=H严格包含于P；d=P∩H为空；o有交集且两者互不包含。五者在该域互斥完备。未知、空满足集、背景不一致、条件重叠未知/为空都是单列unresolved，不是neutral；生命周期不混入叶层。

按e/f/r/d/o顺序，父E/N/C映射为 E=e+f，N=r+o，C=d。M=[[1,1,0,0,0],[0,0,1,0,1],[0,0,0,1,0]]，单向rank3/kernel_dim2。交换T保留e/d/o并对换f/r；双向堆叠[M;MT]rank4/kernel_dim1，nullvector(1,-1,-1,0,1)。硬方向EE/EN/NE/CC/NN唯一恢复e/f/r/d/o，但双向软边缘还缺一个耦合自由度。

p1=(.5,0,0,0,.5)与p2=(0,.5,.5,0,0)给出相同(.5,.5,0,.5,.5,0)双边缘。若两个可区分观察条件等概率出现而decoder只能看该共同边缘，最佳共同posterior是(.25,.25,.25,0,.25)，期望NLL为ln4；能用真实joint的decoder为ln2，差ln2 nats。这是解析条件下的表示信息损失，不是观测ML收益；decoder如能再读原观察或额外耦合信息，下界不成立。

最小非冗余概率坐标是五概率单纯形的4自由度，不是五个动作行为线性独立，也不是只能用5个flat logits。另一个重要区分：合法p=(.20,.25,.13,.40,.02)的双边缘可行，但两个parent MAP是EC。概率可实现、硬组合可映射、相对gold是否错误是三个不同账目。''')
    section(6,'候选头、函数类、损失和参数量','''共同骨干为20→64→64 ReLU，无dropout，5504参数；A尾3×(64+1)=195，总5699；C/H尾5×(64+1)=325，总5829，B与C同flat5形状。H的3父logits+E/N各一binary logit有4概率自由度，两个条件分支独立，不强迫alpha=beta。D只是五叶图的coarse消费面，不另训练、不重复预算。

A：3粗CE。B：5flat但只有grouped coarse NLL，父内比例不可识别，其fine仅诊断。C：B+λ条件fine NLL。H：独立父分布与两个条件概率组合五叶，用同样粗/细损失。全部细标签且λ=1时，grouped coarse+conditional fine严格等于5CE；禁止再叠完整fine CE造成粗损失双计。

数值全log空间：flat log_softmax/logsumexp，H log_softmax(parent)+logsigmoid(±branch)。±1000值/梯度对独立CE、H饱和错分支非零有限梯度、H Jacobian rank4和独立参数/gradient等已专项验证。初始bias只用train粗先验，父内均匀；B/C同flat权重，A/H同父权重，全臂同骨干/批序，E/N独立RNG，initial q存理论与实前向。

C/H相同自由度和参数量不代表相同函数类。固定表示标量x上，C的z_e=x,z_f=-x,z_r=z_d=z_o=0给log(qE/qC)=log(exp x+exp−x)，不是H父仿射函数；H根均1/3、alpha=sigmoid x、beta=1/2给log(pe/pd)=log sigmoid x，不是flat叶仿射log-odds。骨干可改变编码但把问题转向容量/数据/优化，不能从单次实验推出结构支配。''')
    section(7,'数据与目标分布','''N10有1023非空实体、1046529有序对；五叶计数1023/57002/57002/57002/874500，父58025/931502/57002。互补族511，minside1..5分层；train/dev/cal/final family319/65/63/64，实体639/130/64prob+62policy/128final，pair408321/16900/4096/3844/16384，排除跨split589048和cal跨pool7936。FULL只入train。实体及互补family不跨role。

输入只有P/H原始10位bitmap拼接，没有label、ID、集合运算捷径或relational features；gold由独立满足集oracle产生。目标是每隔离池自然有序表的conditional partition distribution，不是1046529全域uniform风险，更不是自然语言泛化。eq占比是1/n，互补比例也被完整family提高，三个seed不是三个独立split。

N8的原finaltoy900与cal已在pilot中评分，整个研究从中得到方法反馈。它们退休为development；r3旧宏F1与旧oracle错误不构成有效选型证据。N10此前未正式评分但生成器公开，封存是流程承诺非信息论秘密。稳定pool key不含目录/时间/source修订，换prepare不产生新未曝光资格。''')
    section(8,'逐项研究决策与r1–r4试错','''r1建立8世界穷举与初版编辑/学习harness，未完成正式学习和完整验证。r2加入log损失、两个estimands、明确H替换阈值、先判再延长、真实freeze/attempt链，但Tiny已曝光N8。r3否决“关系混类=库冲突”，引入真实mask/guard/core并拆为语义、学习、CLI三模块；迁移N10新域与pilot诚实化。阶段3的55 passed只证明已测断言，没有证明正确性。

r4独立复审指出真实缺陷并在正式训练前amend：四元core漏检、core签名漏语义/frame、receipt仅非空、frame可变15→31、macroF1将0TP类记NaN剔除、续段checkpoint丢弃与1500+3000预算双计、finalize只信recommended/formal flag、换prepare洗曝光、逆向NaN从分母消失、geometry位宽及source_files/逐题输出漂移。r4逐项修复后103专项通过，200→300真续训与一次300同state，并恢复basebest/extensionbest/H自由度/initialq/非法marginal等反例。

旧失败留档：三类gold[0,1,2]全pred1旧macro .5而正确1/6；二类gold[0,1]全pred0旧2/3而正确1/3；全错旧NaN而正确0。14/13/11/7旧core空、15→31旧codes空、未归一化marginal true、fake receipt accepted、逆向失败coverage虚高都在defect-probe。r4全部固定注册类zero_division=0，不为绿删红。

本终测阶段另有分析/环境失败：研究首次708 passed/2环境缺失/1skip，纯预分配metadata字节恢复并用现成base MiniLM junction后710 passed/1skip；无源/test基线变更。后处理把float32置换GEMM日志一致性容差设得过严，改为报告实际残差，不改预测或重评分。数学输出曾重复dual键覆盖rank字段，断言本身正确；新out修正输出字段并保留原。辅助取skip原因时basetemp误放research保护区及超Windows路径，21失败为该非标准探针配置错误；标准full早已完整绿色，保护逻辑未放松。''')
    section(9,'优化预算、两个estimands与冻结规则','''CPU1线程、Adam .001、batch128，三个paired seeds13/20260919/424242。1500百步dev，1100→1500任一独立arm/coarse或C/H/fine meanseed增益≥.01才全臂续3000；先落decision再1501。base终态+Adam+下一批索引增量继续，保留1500证据，合并30点全程best，同分取最早。

实际36000 updates、360 training dev calls、4608000样本呈现，360checkpoint/24 terminal Adam，训练wall214.222s/CPU148.703s。freeze又重载360checkpoint实值重算dev和terminal/Adam/批序，wall20.185s；不把验证调用重复算训练。

粗问题：A/B/C/H各用coarsepoint。细结构：只C/H各用finepoint，并在该finepoint核粗退化。默认C，H须meanfine改善≥max(.01,.05*Cmean)、三个pairedseed全正、meancoarse H−C≤.01。dev改善[+.001791056,−.000997514,+.000017174]，mean .000270238<.01且一seed负，故保留C。不能把C推荐写成F1或NLL经验支配；final禁止再挑种子、point、head或重新训练。''')
    section(10,'实际dev与正式final：全部冻结视图','''评分前pre-final-analysis-plan已经登记Astra第二轮条件分布/交换/校准意见。fresh verify-only通过且不消耗ledger，随后正常score一次，18views294912行；相同state的coarse/fine视图是同一预测，不是额外独立样本。以下NLL单位nats，cal只指coarse温度描述性重组合，B fine是unidentified诊断，A概率fine不存在。''')
    rows=[]
    for name,v in data['views'].items():
        s=v['summary']; rows.append([name,v['selection']['step'],f(s['coarse_nll']),f(s['coarse_nll_calibrated']),f(s['fine_nll']),f(s['fine_nll_calibrated']),f(s['coarse_macro_f1']),f(s['calibrated_metrics']['coarse']['macro_f1']),f(s['fine_macro_f1']),f(s['calibrated_metrics']['fine']['macro_f1'])])
    add(table(['candidate','step','coarse raw','coarse cal','fine raw','fine cal','coarseF1 raw','coarseF1 cal','fineF1 raw','fineF1 cal'],rows))
    add('Dev全部18视图与state路径见formal-dev-results.json和dev-only-draft.md。正式final统计由294912 JSONL/CSV逐行回读重算，无新模型调用，完整每类支持/混淆/失败原因、每stratum和swap在final-analysis.json与selected-metrics.csv。A summary里的fine空/占位0不代表A双向恢复表现，下节明确展示强Adual。')
    section(11,'Adual强基线与目标tradeoff（不隐藏）','''A同一个3类模型分别计算P/H与H/P，proto组合恢复5硬标签。非法/失败进入reject列，计作相应gold的FN，全16384分母。reject是诊断列而非第六语义类；宏F1只平均固定五个真实类。没有唯一软joint，因此没有五叶概率NLL。softjoint不兼容、hard reject、gold错误仍分账。''')
    add(table(['seed','Adual accuracy','Adual macroF1','reject','legal wrong','total wrong','reverse failed','softjoint incompatible'],[[s,f(v['accuracy']),f(v['macro_f1']),v['reject'],v['legal_wrong'],v['total_wrong'],v['reverse_failed'],v['soft_joint_incompatible']] for s,v in dual['A_dual'].items()]))
    add(f"Adual mean acc {f(dual['mean_accuracy'])}、mean macroF1 {f(dual['mean_macro_f1'])}，eq recall三个seed均1；总错204=reject128+合法错76。此zero-model-call复算与协调者独立统计一致。Adual每类完整recall与5×6混淆见A-dual-hard.json。")
    means=[]
    for arm in ('A','B','C','H'):
        vs=[v['summary'] for n,v in data['views'].items() if v['selection']['arm']==arm and v['selection']['view']=='coarse']
        means.append([arm,f(np.mean([s['coarse_nll'] for s in vs])),f(np.mean([s['coarse_macro_f1'] for s in vs])),f(np.mean([s['fine_accuracy'] for s in vs])) if arm!='A' else '不定义概率fine',f(np.mean([s['fine_macro_f1'] for s in vs])) if arm!='A' else 'Adual另列'])
    add(table(['arm(coarsepoint)','mean coarseNLL','mean coarseF1','mean fine acc raw','mean fineF1 raw'],means))
    add('C mean fine acc .996032715 > Adual .995849609 > H .995646159；但Adual fine宏F1在对C/H时各2赢1输，并非全面支配。C fine宏F1mean .992196756、NLLmean .019789558；H宏F1mean .991770410、NLLmean .019717792，H仅seed13好、两seed劣，不支持支配。coarseA宏F1mean .993406112/NLL .009822286，相对C .989320263/.015965117与H .988306090/.015709432体现细监督的粗目标tradeoff。C推荐理由是单调用完整joint5概率接口和既定dev规则，不是accuracy/F1支配。Adual需要两个backbone前向，参数共享不自动省第二次计算。')
    add(table(['seed','dev C−H fine','final raw C−H fine','final cal C−H fine','raw保持dev方向'],[[r['seed'],f(r['dev_fine_improvement_C_minus_H']),f(r['final_fine_improvement_raw_C_minus_H']),f(r['final_fine_improvement_cal_C_minus_H']),r['same_sign_raw_as_dev']] for r in data['paired_final_descriptive']]))
    add('seed424242没有保持dev的微小H优势，必须如实记录；不据此改原推荐或换种子。三个优化seed只代表这一次固定split的训练随机性。')
    section(12,'>255新颖性导致的四prefix结构偏移','''两互补成员都>255等价256..767，最高两位只能01/10，每类64entities；四个有序组合各4096。sameprefix共有新增世界，不可能d；differentprefix双方各有对方不含世界，只能d/o。gold逐行验证这些支持约束成立。新颖性并非一般新实体域，对角线/互补比、基数和类先验也与train/cal不同，不能说全域generic泛化。下面全部视图都报告，不据strata再次选择。''')
    rows=[]
    for name,v in data['views'].items():
        for key,s in v['strata'].items():
            if key.startswith('prefix:'):rows.append([name,key,s['n'],json.dumps(s['gold_support'],ensure_ascii=False),f(s['fine_nll_raw']),f(s['fine_nll_cal']),json.dumps(s['fine_errors_raw'],ensure_ascii=False),json.dumps(s['fine_errors_cal'],ensure_ascii=False)])
    add(table(['candidate','prefix','support','gold counts','fineNLL raw','fineNLL cal','errors raw(gold→pred)','errors cal'],rows))
    add('该表A的概率fine为空，hard诊断另列；不能将空值解释为没有错误。各candidate的diagonal/complement true/false、P/H基数1..9及联合基数strata原始支持/NLL/error分布完整见final-analysis.json，不为表简短而取消分层。final对角线128/16384，互补有序对128/16384。prob/policy对角线率1/64、1/62而final1/128，coarse校准也面临先验偏移。')
    section(13,'交换一致性、校准、失败与错误案例','''单pair五叶归一化给出联合表示，不保证独立p(P,H)=T p(H,P)，更不保证多对象知识库合取一致。现有全有序final行按对象交换配对，零新增推理；下面概率残差和MAP一致性与gold准确率分账。正向与置换后反向float32 GEMM可能产生微小log差，不能错误承诺bit-level。''')
    rows=[]
    for name,v in data['views'].items():
        sw=v['swap'];tw=v['A_two_way_hard'];rows.append([name,f(sw['raw']['max_probability_linf']),f(sw['raw']['mean_probability_linf']),f(sw['raw']['hard_fine_swap_consistency_all_attempt']),f(sw['calibrated']['max_probability_linf']),f(sw['calibrated']['hard_fine_swap_consistency_all_attempt']),f(tw['hard_swap_consistency_all_attempt']) if tw else '—'])
    add(table(['candidate','raw最大概率差','raw平均概率差','raw fine hard一致','cal最大概率差','cal fine hard一致','Adual hard一致'],rows))
    add('coarse T在cal-prob4096对上、[.25,4]的61点拟合；alpha/beta条件保持不变，再重组leaf概率，raw/cal均报告。policy3844对取90%经验coverage只描述，不是风险证书，未独立校准五叶后验。某些seed原始到cal不改善也照录。所有source/model SHA、有限性、simplex、失败reason及原CSV/JSONL已经回读；B细层不识别不会因温度而变成掌握五叶。')
    err=load(base/'formal-analysis-final/error-examples.json')
    add('错误案例按预先固定candidate/index和每gold→pred最前3个产生，不是挑最有利故事；以下只列前12，完整列表与全量error分布在独立JSON：')
    add(table(['candidate','index','P','H','gold','raw fine','cal fine','reason'],[[r['candidate'],r['index'],r['p'],r['h'],r['gold_leaf'],r['fine_map'],r['fine_map_calibrated'],r['failure_reason']] for r in err[:12]]))
    section(14,'独立数学、构造容量与参考gauge（与学习成绩分开）',f'''NumPy实际rank(M)={maths['ranks']['parent_rank']}、rank(dual)={maths['ranks']['dual_rank']}，kernel维数2/1与nullvector均zero-residual验收；解析ln2/ln4差{maths['nonidentifiability']['gap_nats']}nats，不是ML收益。

本writer只在内存实际执行一个坐标置换、1046529对、2093058正/反body sample-passes；flat5/H/Aforward/Areverse/合法方向5恢复harderrors均0，不保存oracleweights。这不是学习/holdout/NLP，不给性能排名。公式first30 u=relu(p−h),v=relu(h−p),w=relu(p+h−1)；second4 S=relu(1−Σu),R=relu(1−Σv),D=relu(1−Σw),E=relu(1−Σu−Σv)；flat κ(E,S−E,R−E,D,1−S−R−D+E)，A κ(S,1−S−D,D)，H bE=κ(2E−S),bN=κ(2R−1)，κ10，其余权重/bias全0。代码和json提供直接复现。

协调者另提供外部助手的10置换、10465290对、20930580正反body sample-passes、全harderror0，约35.9s计算/41.6s启动总。它是外部观察，不写成自己的十置换；combined-allheads6154是共享构造测试容器，不是每部署candidate。实际A5699、C/H5829。

flat5减o参考为4logodds+固定0；H父减N参考为2root坐标+2conditionals。自己的float64随机10000×64验证max概率差flat={maths['gauge']['flat_max_probability_difference']},H={maths['gauge']['H_max_probability_difference']}；协调者独立复算值6.661338147750939e−16/5.551115123125783e−16另有出处。尾参数325→260只省65，SGD/Adam不在该变换下保持训练轨迹，不导第二压缩artifact/loader，保留成熟单通道5输出。4维是概率非冗余坐标，不是动作独立证明；实际dtype、阈值/MAP边界不承诺bit-level，更没有真实NLP收益。''')
    section(15,'两轮Astra原文与采纳/不采纳','''两轮最终正文独立存astra-review.md和astra-review-r2.md，未复制进度/思维摘要/邮箱。请求模型分别gpt-6-astra/xhigh与gpt-6-astra/high、同线程回执；requested不是backend attestation，不称真人gold。

第一轮采纳：target分布明确、两个estimands、共同延长的逐arm/metric触发、H配对阈值、log数值、粗校准与父内比例、粒度与动作解耦、函数类互不支配。第一条方向表误写来自协调者题面，协议合法集合一直正确，记录为提示词修订而非虚报实现bug。

第二轮采纳：01/10四prefix分层及diagonal/complement/cardinality；H粗门在实际finepoint；单pair coherent不等跨call一致，现有题行交换诊断；coarse T不校准alpha/beta、不发风险证书；N8已曝光而整个研究受pilot启发。补了秩/核/解析NLL、单置换容量与gauge附录。未采纳新增学习模拟/扩大语义标签/再训NLI来证明接口信息优势；也不采用参考gauge导第二模型通道。没有从final回改既定规则。''')
    section(16,'质量验证、依赖与真实失败记录','''必要研究回归明确忽略research/memory_relations/runs，避免搜到旧source-snapshot过时tests。真实710 passed/1skip(v0旧产物不在WT)，118.41s；tiny-NLI/公开数据集机制均只fixture/本地base，不运行旧公开基准真实预测或正式NLI训练。首次2失败属缺asset，原日志保留；协议/审计fixture纯metadata确认后byte复制，SHA d2394abb73c985a7bd2878b327d208040be773a5a5d14646a75b6151cbda5cc6与a19b45510be32bb920e0b80a4cc0e21d8831f1515162885e48262344665e1cef。测试显式WT模型路径不听env，因此junction只指向Root现成base MiniLM，不安装下载、不读取旧七类锁权重，不打包该base模型。

standard full profile真实14/14门、73commands全部exit0，pytest仅tests+devcontext/tests，compile不含research，故两套证据分开。冻结784条，Top1/Top5各97.7%，硬/FASTQ违规0、NoResult10/10。full/研究绿只证明契约与集成，不证明ML收益；没有远端CI/生产部署声称。完整JSON在quality/head-design-full-r4.json，原命令与日志保留。''')
    add(table(['gate','status','commands','seconds'],[[g['id'],g['status'],len(g['commands']),g['duration_seconds']] for g in qc['gates']]))
    skip=load(base/'standard-pytest-skip-reasons.json');add('已实际取得skip理由：'+ '; '.join(skip['skip_lines'])+'。另installer两项缺构建runtime、public-only在private树跳过等都是既定门声明，不为本研究fake skip。该取skip补充命令因把basetemp放research与过长Windows路径导致21临时环境失败，不是standard full命令；20被保护目录拒绝、1Git Filename too long。原日志保留，未修改源码/保护基线来遮红；成功full标准临时目录已经覆盖相同tests集合。')
    section(17,'真实NL/人类编辑证据还缺什么、可复用什么','''下一阶段需要独立NL标注与human edit闭环：①保留type/modal/quantifier/frame/time/conditions/payload/evidence，不混硬事实、义务、软偏好与例子；②已知身份、版本、scope、TTL由可信元数据直接给，不再让小模型猜；③把逐句绑定、覆盖/歧义、全条件语句与交叠内容gold区分；④标注真实用户授权与完整changeset，并核对提交后实际语义/依赖/引用。自然语言解释准确性、真人偏好、操作可靠性和风险校准需要各自独立单位/采样，不能从bitmap题数虚增。

soft preference、program/procedure、example的通用性是待验证假设。比如建议而非义务、时间变动许可、程序产出等，不能直接拿集合蕴含硬头当最终决策。未知解析/空域/背景不一致必须单列，不用neutral吞掉。五叶不承担权限、定位或执行成功。

可复用：协议几何/投影、log损失、固定类指标、完整分区与曝光身份、选点/预算/manifest守卫、数学附录、受控编辑反例与复現代码；生产提交/绑定解析需独立接口接入。旧七类词汇/数据记录经验可作设计线索，但旧锁权重、校准、风险证书与教师代理标签不继承为新gold。新头/新数据/新监督要分开比较，不能把结构、额外细监督和分布修订混成单一因果提升。''')
    section(18,'成本、交付与使用边界','''训练无外部API，CPU1、wall214.222s/CPU148.703s；freeze验证360点wall20.185s。唯一正式score18view294912行，但只有12独立artifact，各三pool+3次A逆向共39 forward batches/341040 forward rows，wall86.13s（含链路验证），不是重复18模型计算。后处理和Adual复算零模型调用；硬件电耗/折旧与助手会话费用未折算。

source三模块bundle f79d954566121315a6fbac367542dba6cae5111d404c3d39a407251136f88294、protocol03a88da1546335b60a3e0c5531dc7b6f9d747b6c71e902dbc3bad4eefa29ed41、freeze120d960141f97d36cd0da9bacfc5eaf108e8e26016942581075198295f440934不变。formal pool9c251f95942bd1a2f41c90d9faf02ffed8c7380a8952f49023d39017a583baaf只有一次mode=score attempt及success，不是reproduce。后续不得final挑点/重训/洗曝光。

外部package包含本报告、两轮评审正文、预注册/amendments、按namespace布置的冻结source/tests、对应完整正式与pilot学习产物、source快照、逐题/metrics、全部比较器weights/360checkpoint/24terminal、质量证据、校验索引与路径map。原bytes不改，路径移动用map说明；Root运行依赖与包中的code/data回读核验、独立smoke(不再score正式final)证明的范围明确。省略的pytest tmp仅临时探针、完整仍在原WT，列表如实记录，不能称全部历史自包含。

一次流程异常也必须保留：阶段4开工local-only devcontext自动迁移16条历史usage到共享git cache/per-WT projection，local-only不等read-only；没有清改共享账本。此阶段没有再调用resolve/build/status或广读历史/cache。没有commit/push/merge/deploy，资料不包含.env/.userdata/秘值/preflight邮箱/旧goal资产。''')
    add('''## 文件导航与核验方式

- `reports/astra-review.md` / `reports/astra-review-r2.md`：两轮最终正文与归因。
- `reports/pre-final-analysis-plan.md`：任何正式评分前的计划。
- `results/final-analysis.json`、`results/A-dual-hard.json`、`results/selected-metrics.csv`、`results/error-examples.json`：全文统计和诊断。
- `evidence/head-design-v5-20261013-r4/formal-study/finalize/`：294912原JSONL/CSV、summary及manifest。
- `source/research/memory_relations/`：原始三模块、protocol、tests；未引入分析脚本到sourcebundle。
- `reports/analysis/`：仅数学/报告/回读程序；`复现说明.md`给实际命令，`path-map.json`说明移动与原路径，`file-index.json`逐文件sha/来源。

报告中的错误分布与统计只做描述。C固定推荐基于single-call完整joint接口与已冻结规则；Adual是一条强硬标签基线，不能用A摘要概率fine空值把它藏起来。全部部署或真人/NLP推广仍需要独立授权和证据。
''')
    text='\n'.join(parts);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text,encoding='utf-8');print(json.dumps({'report':str(a.out),'bytes':len(text.encode('utf-8')),'gates':14,'commands':73,'model_calls':0}))
if __name__=='__main__':main()
