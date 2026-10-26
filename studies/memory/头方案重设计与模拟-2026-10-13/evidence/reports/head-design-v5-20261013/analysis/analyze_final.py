"""Read-only frozen-question analysis; never runs a model or changes selection."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--source-root',type=Path)
    a=p.parse_args()
    source_root=a.source_root or Path(__file__).resolve().parents[5]
    sys.path.insert(0,str(source_root))
    from research.memory_relations import head_design as H
    from research.memory_relations.head_design_learning import _csv_cell
    if a.out.exists(): raise SystemExit('analysis out already exists')
    started=time.time(); cpu=time.process_time()
    frozen=H._read_json(a.study/'freeze/freeze.json'); proto=H._read_json(a.study/'prepare/protocol-frozen.json')['content']; maps=H.index_maps(proto)
    summary=H._read_json(a.study/'finalize/summary.json'); manifest=H._read_json(a.study/'finalize/manifest.json')
    assert H.source_bundle_sha256()==frozen['source_bundle_sha256']=='f79d954566121315a6fbac367542dba6cae5111d404c3d39a407251136f88294'
    for rel,sha in manifest['artifacts'].items(): assert H.sha256_file(a.study/'finalize'/rel)==sha,rel
    candidate={v['candidate']:v for v in frozen['candidate_views']}; counts={}; views={}; pairs_reference=None; total=0; errors=[]
    def counts_for(rows,pred_key):
        return dict(sorted(Counter(f"{r['gold_leaf']}->{r.get(pred_key)}" for r in rows if r.get(pred_key)!=r['gold_leaf']).items()))
    def block(rows):
        s=H.summarize_rows(rows,proto,maps)
        return {'n':len(rows),'gold_support':dict(sorted(Counter(r['gold_leaf'] for r in rows).items())),
                'coarse_nll_raw':s['coarse_nll'],'coarse_nll_cal':s['coarse_nll_calibrated'],
                'fine_nll_raw':s['fine_nll'],'fine_nll_cal':s['fine_nll_calibrated'],
                'coarse_accuracy_raw':s['coarse_accuracy'],'fine_accuracy_raw':s['fine_accuracy'],
                'coarse_accuracy_cal':s['calibrated_metrics']['coarse']['accuracy'],'fine_accuracy_cal':s['calibrated_metrics']['fine']['accuracy'],
                'fine_errors_raw':counts_for(rows,'fine_map') if rows[0]['arm']!='A' else None,
                'fine_errors_cal':counts_for(rows,'fine_map_calibrated') if rows[0]['arm']!='A' else None}
    def process(name,rows):
        nonlocal pairs_reference
        assert name in candidate and len(rows)==16384
        assert [r['index'] for r in rows]==list(range(16384))
        pairs=[(r['p'],r['h']) for r in rows]
        if pairs_reference is None: pairs_reference=pairs
        else: assert pairs==pairs_reference
        recomputed=H.summarize_rows(rows,proto,maps)
        expected={k:v for k,v in summary['views'][name].items() if k not in ('temperature','policy')}
        assert H._canonical(recomputed)==H._canonical(expected),name
        by_pair={pair:r for pair,r in zip(pairs,rows)}; assert len(by_pair)==16384
        strata=defaultdict(list)
        for r in rows:
            ph,hh=r['p']>>8,r['h']>>8
            assert ph in (1,2) and hh in (1,2)
            if ph==hh: assert r['gold_leaf']!='disjoint'
            else: assert r['gold_leaf'] in ('disjoint','overlap_incomparable')
            strata[f'prefix:{ph:02b}/{hh:02b}'].append(r)
            strata[f"diagonal:{r['p']==r['h']}"].append(r)
            strata[f"complement:{(r['p']^r['h'])==1023}"].append(r)
            pc,hc=r['p'].bit_count(),r['h'].bit_count()
            strata[f'cardinality-p:{pc}'].append(r); strata[f'cardinality-h:{hc}'].append(r); strata[f'cardinality-pair:{pc}/{hc}'].append(r)
        for key in ('prefix:01/01','prefix:01/10','prefix:10/01','prefix:10/10'): assert len(strata[key])==4096
        swap={}; swap_index=[maps['leaf_index'][proto['semantics']['swap_map'][leaf]] for leaf in maps['leaves']]
        for suffix in ('','_calibrated'):
            residual=[]; hard=0; parentfirst=0; available=0
            for r in rows:
                reverse=by_pair[(r['h'],r['p'])]
                if r.get('p5'+suffix) is not None and reverse.get('p5'+suffix) is not None:
                    residual.append(float(np.max(np.abs(np.asarray(r['p5'+suffix])-np.asarray(reverse['p5'+suffix])[swap_index]))))
                    available+=1; hard+=proto['semantics']['swap_map'][r['fine_map'+suffix]]==reverse['fine_map'+suffix]
                    parentfirst+=proto['semantics']['swap_map'][r['parent_first'+suffix]]==reverse['parent_first'+suffix]
            swap['raw' if not suffix else 'calibrated']={'attempted':len(rows),'available':available,'failed':len(rows)-available,
                'max_probability_linf':max(residual) if residual else None,'mean_probability_linf':float(np.mean(residual)) if residual else None,
                'hard_fine_swap_consistency_all_attempt':hard/len(rows) if available else None,'parent_first_swap_consistency_all_attempt':parentfirst/len(rows) if available else None}
        two=None
        if rows[0]['arm']=='A':
            legal=correct=hard_swap=failed=0; combo_counts=Counter(); fine_conf=Counter()
            for r in rows:
                e=r['two_way']; reverse=by_pair[(r['h'],r['p'])]; label=None
                if e and e.get('status')!='failed':
                    combo_counts[e['combo']]+=1; label=H.combination_leaf(e['hard_a'],e['hard_b'],proto)
                    if e.get('logp3_reverse') is not None:
                        e['stored_vs_ordered_reverse_log_linf']=float(np.max(np.abs(np.asarray(e['logp3_reverse'])-reverse['logp3'])))
                        e['stored_vs_ordered_reverse_probability_linf']=float(np.max(np.abs(np.exp(e['logp3_reverse'])-np.exp(reverse['logp3']))))
                else: failed+=1
                legal+=label is not None; correct+=label==r['gold_leaf']; fine_conf[f"{r['gold_leaf']}->{label}"]+=1
                er=reverse.get('two_way'); revlabel=None if not er or er.get('status')=='failed' else H.combination_leaf(er['hard_a'],er['hard_b'],proto)
                if label is not None: hard_swap+=proto['semantics']['swap_map'][label]==revlabel
            two={'n':len(rows),'failed':failed,'legal':legal,'illegal':len(rows)-legal,'coverage':legal/len(rows),'fine_hard_correct_all_attempt':correct/len(rows),
                 'fine_hard_accuracy_legal_only':correct/legal if legal else None,'hard_swap_consistency_all_attempt':hard_swap/len(rows),
                 'combo_counts':dict(sorted(combo_counts.items())),'confusion_all_attempt':dict(sorted(fine_conf.items())),'probability_fine_nll':None,
                 'stored_vs_ordered_reverse_log_linf_max':max((r['two_way'].get('stored_vs_ordered_reverse_log_linf',0) for r in rows),default=0),
                 'stored_vs_ordered_reverse_probability_linf_max':max((r['two_way'].get('stored_vs_ordered_reverse_probability_linf',0) for r in rows),default=0),
                 'note':'two backbone calls; no unique joint posterior inferred; marginal feasibility/illegal hard combo are not gold error; float32 ordered/permuted GEMM may differ numerically, residuals reported not bit-parity promised'}
        samples=Counter()
        for r in rows:
            for pred in ('parent_map','fine_map','fine_map_calibrated'):
                gold='gold_parent' if pred=='parent_map' else 'gold_leaf'
                if r.get(pred) is None or r[pred]==r[gold]: continue
                key=(pred,r[gold],r[pred])
                if samples[key]<3:
                    errors.append({k:r.get(k) for k in ('candidate','arm','seed','view','step','index','p','h','gold_leaf','gold_parent','parent_map','fine_map','fine_map_calibrated','coarse_nll','fine_nll','fine_nll_calibrated','failure_reason')}); samples[key]+=1
        views[name]={'selection':candidate[name],'summary':recomputed,'temperature':summary['views'][name]['temperature'],'policy':summary['views'][name]['policy'],
                     'strata':{key:block(value) for key,value in sorted(strata.items())},'swap':swap,'A_two_way_hard':two,'fine_diagnostic_only':rows[0]['arm']=='B'}
        counts[name]=len(rows)
    with (a.study/'finalize/questions.csv').open(encoding='utf-8',newline='') as cf, (a.study/'finalize/questions.jsonl').open(encoding='utf-8') as jf:
        reader=csv.DictReader(cf); current=None; rows=[]
        for line,flat in zip(jf,reader,strict=True):
            r=json.loads(line); assert flat=={key:_csv_cell(value) for key,value in r.items()}
            assert set(r)==set(proto['evaluation']['per_question_fields']); assert r['model_sha256']==candidate[r['candidate']]['model_sha256']
            if current is not None and current!=r['candidate']: process(current,rows); rows=[]
            current=r['candidate']; rows.append(r); total+=1
        if rows: process(current,rows)
    assert total==294912 and set(views)==set(candidate)
    entities={p for p,h in pairs_reference}; assert len(entities)==128
    prefix_entities=dict(Counter(f'{p>>8:02b}' for p in entities)); assert prefix_entities=={'01':64,'10':64}
    paired=[]
    for seed in proto['training']['seeds']:
        c,h=views[f'C-{seed}-fine']['summary'],views[f'H-{seed}-fine']['summary']
        dev=next(row for row in frozen['architecture_rule']['paired_differences'] if row['seed']==seed)
        paired.append({'seed':seed,'dev_fine_improvement_C_minus_H':dev['fine_improvement'],'final_fine_improvement_raw_C_minus_H':c['fine_nll']-h['fine_nll'],
                       'final_fine_improvement_cal_C_minus_H':c['fine_nll_calibrated']-h['fine_nll_calibrated'],
                       'final_coarse_delta_raw_H_minus_C':h['coarse_nll']-c['coarse_nll'],'final_coarse_delta_cal_H_minus_C':h['coarse_nll_calibrated']-c['coarse_nll_calibrated'],
                       'same_sign_raw_as_dev':np.sign(dev['fine_improvement'])==np.sign(c['fine_nll']-h['fine_nll'])})
        paired[-1]['same_sign_raw_as_dev']=bool(paired[-1]['same_sign_raw_as_dev'])
    out={'source_bundle_sha256':H.source_bundle_sha256(),'protocol_sha256':summary['protocol_sha256'],'freeze_sha256':H.sha256_file(a.study/'freeze/freeze.json'),
         'summary_sha256':H.sha256_file(a.study/'finalize/summary.json'),'csv_sha256':H.sha256_file(a.study/'finalize/questions.csv'),'questions_sha256':H.sha256_file(a.study/'finalize/questions.jsonl'),
         'rows':total,'prefix_entity_support':prefix_entities,'views':views,'paired_final_descriptive':paired,'fixed_recommendation':'C','rule_changed':False,
         'new_model_calls':0,'duplicate_views_are_not_independent_samples':True,'score_inference_calls':summary['inference_calls'],'wall_seconds':time.time()-started,'cpu_seconds':time.process_time()-cpu}
    a.out.mkdir(parents=True); H._write_json(a.out/'final-analysis.json',out); H._write_json(a.out/'error-examples.json',errors)
    with (a.out/'selected-metrics.csv').open('w',encoding='utf-8',newline='') as f:
        cols=['candidate','arm','seed','view','step','coarse_nll_raw','coarse_nll_cal','fine_nll_raw','fine_nll_cal','coarse_macro_f1_raw','fine_macro_f1_raw','coarse_macro_f1_cal','fine_macro_f1_cal','coarse_recall_raw','fine_recall_raw','coarse_recall_cal','fine_recall_cal']
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for name,v in views.items():
            s=v['summary']; r={key:v['selection'].get(key) for key in cols[:5]}
            for kind in ('coarse','fine'):
                r[f'{kind}_nll_raw']=s[f'{kind}_nll']; r[f'{kind}_nll_cal']=s[f'{kind}_nll_calibrated']; r[f'{kind}_macro_f1_raw']=s[f'{kind}_macro_f1']; r[f'{kind}_macro_f1_cal']=s['calibrated_metrics'][kind]['macro_f1']
                r[f'{kind}_recall_raw']=json.dumps(s[f'{kind}_per_class_recall']); r[f'{kind}_recall_cal']=json.dumps(s['calibrated_metrics'][kind]['per_class_recall'])
            w.writerow(r)
    print(json.dumps({'rows':total,'views':len(views),'prefix_entities':prefix_entities,'paired':paired,'score_calls':summary['inference_calls'],'wall_seconds':out['wall_seconds']},indent=2))

if __name__=='__main__': main()
