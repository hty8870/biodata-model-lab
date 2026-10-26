"""Full-denominator A hard decoder: reject is an error column, not a sixth semantic leaf."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source-root',type=Path);a=p.parse_args()
    if a.out.exists():raise SystemExit('out exists')
    sys.path.insert(0,str(a.source_root or Path(__file__).resolve().parents[5]))
    from research.memory_relations import head_design as H
    proto=H._read_json(a.study/'prepare/protocol-frozen.json')['content']; leaves=proto['semantics']['leaves'];idx={v:i for i,v in enumerate(leaves)}
    cm={s:np.zeros((6,6),dtype=np.int64) for s in proto['training']['seeds']};counts={s:Counter() for s in cm};rows=0
    with (a.study/'finalize/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);rows+=1
            if r['arm']!='A':continue
            s=r['seed'];e=r['two_way']; pred=None if not e or e.get('status')=='failed' else H.combination_leaf(e['hard_a'],e['hard_b'],proto)
            cm[s][idx[r['gold_leaf']],5 if pred is None else idx[pred]]+=1
            counts[s]['reverse_failed']+=int(not e or e.get('status')=='failed')
            counts[s]['soft_joint_incompatible']+=int(bool(e) and e.get('joint_feasible') is False)
    assert rows==294912
    result={}
    for s,m in cm.items():
        stats=H.confusion_metrics(m); n=int(m.sum()); tp=int(np.trace(m[:5,:5]));reject=int(m[:5,5].sum())
        result[str(s)]={'n':n,'accuracy':tp/n,'macro_f1':float(np.mean(stats['f1'][:5])),'per_class_recall':dict(zip(leaves,stats['recall'][:5])),
                        'support':dict(zip(leaves,stats['support'][:5])),'reject':reject,'total_wrong':n-tp,'legal_wrong':n-tp-reject,**dict(counts[s]),
                        'confusion_5_gold_by_5_predict_plus_reject':m[:5].tolist(),'fine_probability_nll':None}
    out={'total_question_rows_streamed':rows,'model_calls':0,'A_dual':result,'reject_is_full_FN_not_semantic_class':True,
         'mean_accuracy':float(np.mean([v['accuracy'] for v in result.values()])),'mean_macro_f1':float(np.mean([v['macro_f1'] for v in result.values()]))}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='A_dual'},indent=2))
if __name__=='__main__':main()
