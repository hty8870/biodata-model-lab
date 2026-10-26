import sys,os,json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
denied=[]
def guard(e,a):
 if e=='open' and ((isinstance(a[1],str) and any(c in a[1] for c in 'wax+')) or (a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))): denied.append(e); raise PermissionError('read-only')
 if e in {'os.mkdir','os.remove','os.rename','os.rmdir','subprocess.Popen','socket.connect','socket.bind'}: denied.append(e); raise PermissionError('read-only:'+e)
sys.addaudithook(guard)
R=Path.cwd(); W=R/'artifacts/worktrees/semantic-input-cache-20261015'; N=R/'artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v8-prob-policy-development-20261015'; P=N/'delivery'
T=R/'artifacts/worktrees/semantic-formal-train-20261015/research/memory_relations/runs/v8-formal-preparation-20261015/raw-mother3-seed-13/tokenizer'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(t,x): print(t+' '+json.dumps(x,sort_keys=True),flush=True)
sys.path[:0]=[str(W),str(W/'src')]
from research.memory_relations import semantic_data as D
from dataset_recommender.agent import semantic_relations as S
files=[W/'research/memory_relations/semantic_model.py',Path(D.__file__),Path(S.__file__),P/'source_rows.jsonl',P/'qualifications.jsonl',P/'qualified_rows.jsonl']
before=[sha(p) for p in files]
assert before[0].startswith('52dbfccd') and before[3]=='56f671aa0c8c3a8d88a68fbd2984986d71b2edecf11fe883c0b687d91b0568c4' and before[4]=='51a697fbcf3094ce1c10db36604c1fda5b7b5a25265d5bcf4feff1fc98de9d1f'
rows=[D.validate_row(r) for r in D.load_rows(P/'source_rows.jsonl')]; qb={q['id']:q for q in D.load_rows(P/'qualifications.jsonl')}
sup={r['id']:D.supervision(r,qualification=qb.get(r['id'])) for r in rows}
from transformers import AutoTokenizer
import transformers
tok=AutoTokenizer.from_pretrained(str(T),local_files_only=True,trust_remote_code=False)
assert tok.is_fast and not any(x in sys.modules for x in ('torch','tensorflow','flax'))
emit('TOKENIZER',{'class':type(tok).__name__,'version':transformers.__version__,'is_fast':tok.is_fast,'model_input_names':tok.model_input_names,'files':{p.name:sha(p) for p in sorted(T.iterdir()) if p.is_file() and p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','spiece.model','added_tokens.json')},'backend_sha256':hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),'recipe':S.SEMANTIC_INPUT_RECIPE,'limits':dict(S.SEMANTIC_LIMITS_RESEARCH),'max_length':384})
start=time.perf_counter(); good={}; bad={}; fstats={}
for r in rows:
 k=r['id']; pair=D.student_pair(r); dec=k in qb and sup[k]['loss_mask']==[1,1]
 try: item=S.prepare_semantic_input(pair,tok,max_length=384,role_independent_frame=dec,limits=S.SEMANTIC_LIMITS_RESEARCH)
 except S.SemanticRelationError as e: bad[k]=e.code; continue
 assert item['raw_text']==S.semantic_pair_text(pair,limits=S.SEMANTIC_LIMITS_RESEARCH)
 f,b=item['directions']['forward'],item['directions']['reverse']
 assert f['visible_information_sha256']==b['visible_information_sha256']==item['visible_information_sha256'] and f['complete']==b['complete']==item['complete']
 for v in (f,b):
  assert v['role_body_covered']=={'old':True,'new':True} and v['token_length']==len(v['encoding']['input_ids'])<=384
  assert tok(v['text'],add_special_tokens=True,padding=False,truncation=False)['input_ids']==v['encoding']['input_ids']
 good[k]=item
 pieces=S.semantic_pair_pieces(pair,limits=S.SEMANTIC_LIMITS_RESEARCH)['fields']; leaves={tuple(p['path']) for p in pieces if p['leaf']}; audit=f['fields']; fs={}
 for p in audit:
  if len(p['path'])!=2: continue
  leaf=[x for x in audit if tuple(x['path']) in leaves and x['path'][:2]==p['path']]
  original=sum(x['original_range'][1]-x['original_range'][0] for x in leaf)
  visible=sum(b-a for x in leaf for a,b in x['visible_source_ranges'])
  assert 0<=visible<=original
  fs['/'.join(p['path'])]=[1,int(p['lost']),int(p['omitted']),original,original-visible]
 fstats[k]=fs
assert len(good)+len(bad)==120 and set(good)==set(qb)
def q(v):
 v=sorted(v)
 if not v:return None
 def at(p):
  x=(len(v)-1)*p; a=int(x); b=min(a+1,len(v)-1)
  return round(v[a]+(v[b]-v[a])*(x-a),2)
 return [v[0],at(.5),at(.9),at(.95),v[-1]]
def field(rr):
 z={}
 for r in rr:
  for k,v in fstats.get(r['id'],{}).items(): z[k]=[a+b for a,b in zip(z.get(k,[0]*5),v)]
 return z
emit('METRIC_KEYS',{'quantiles':['min','p50','p90','p95','max'],'field':['present','lost_rows','omitted_rows','leaf_JSON_literal_chars','lost_leaf_JSON_literal_chars'],'support':['all','qualified','prepared','complete','full_forward_p50','full_forward_max','body_literal_lost_fraction']})
for part in ('probability','policy'):
 rr=[r for r in rows if r['split']==part]; ok=[r for r in rr if r['id'] in good]
 result={'all120_subset':{'attempted':len(rr),'prepared_pairs':len(ok),'failed_pairs':len(rr)-len(ok),'errors':dict(Counter(bad[r['id']] for r in rr if r['id'] in bad))},'qualified99_subset':{'attempted':len(ok),'prepared_pairs':len(ok),'failed_pairs':0},'directions_successful_sets_identical':{},'fields_original_paths_same_both_directions':field(rr),'same_visible_info_hash_pairs':len(ok)}
 for i,d in enumerate(('forward','reverse')):
  result['directions_successful_sets_identical'][d]={'prepared':len(ok),'complete':sum(good[r['id']]['complete'] for r in ok),'truncated':sum(not good[r['id']]['complete'] for r in ok),'both_role_bodies_covered':len(ok),'encoded_tokens':q([good[r['id']]['directions'][d]['token_length'] for r in ok]),'full_visible_tokens':q([good[r['id']]['full_visible_token_lengths'][i] for r in ok])}
 emit('WINDOW_'+part,result)
 support={}
 for axis in ('kind','family'):
  groups=defaultdict(list)
  for r in rr:
   key=r['pair']['old']['kind']+'+'+r['pair']['new']['kind'] if axis=='kind' else r['phenomena']['family']; groups[key].append(r)
  out={}
  for key,group in sorted(groups.items()):
   oks=[r for r in group if r['id'] in good]; fs=field(group); total=sum(v[3] for k,v in fs.items() if k in ('old/text','new/text')); lost=sum(v[4] for k,v in fs.items() if k in ('old/text','new/text')); full=[good[r['id']]['full_visible_token_lengths'][0] for r in oks]
   out[key]=[len(group),sum(r['id'] in qb for r in group),len(oks),sum(good[r['id']]['complete'] for r in oks),None if not full else q(full)[1],None if not full else max(full),None if not total else round(lost/total,4)]
  support[axis]=out
 emit('SUPPORT_'+part,support)
emit('FAILURES',{'counts':dict(Counter(r['split']+' '+str(sup[r['id']]['loss_mask'])+' '+bad[r['id']] for r in rows if r['id'] in bad)),'ids':[{'id':r['id'],'source_line':i,'stage':r['split']} for i,r in enumerate(rows,1) if r['id'] in bad]})
assert before==[sha(p) for p in files] and not any(x in sys.modules for x in ('torch','tensorflow','flax'))
emit('DONE',{'good':len(good),'bad':len(bad),'actual_encoded_orientations':len(good)*2,'seconds':round(time.perf_counter()-start,2),'hashes_unchanged':True,'no_NN_imported':True,'denied_write_operations':len(denied),'filelock_probe_conservative_false':not sys.modules['filelock._strict']._LINK_HONORS_FOLLOW_SYMLINKS})
