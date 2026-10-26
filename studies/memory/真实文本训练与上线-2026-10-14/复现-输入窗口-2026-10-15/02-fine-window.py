import sys,os,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
blocked=Counter()
def guard(e,a):
 if e=='open' and ((isinstance(a[1],str) and any(c in a[1] for c in 'wax+')) or (a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))): blocked[e]+=1; raise PermissionError('read-only')
 if e in {'os.mkdir','os.remove','os.rename','os.rmdir','subprocess.Popen','socket.connect','socket.bind'}: blocked[e]+=1; raise PermissionError('read-only:'+e)
sys.addaudithook(guard)
R=Path.cwd(); W=R/'artifacts/worktrees/semantic-input-cache-20261015'; P=R/'artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v7-fine-development-20261014/teacher-completion-20261015-c71/delivery'; T=R/'artifacts/worktrees/semantic-formal-train-20261015/research/memory_relations/runs/v8-formal-preparation-20261015/raw-mother3-seed-13/tokenizer'
sys.path[:0]=[str(W),str(W/'src')]
from research.memory_relations import semantic_data as D
from research.memory_relations import semantic_model as SM
from dataset_recommender.agent import semantic_relations as S
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(m.__file__) for m in (SM,D,S)]+[P/n for n in ('qualified-source.json','source_rows.jsonl','qualifications.jsonl','allocation.json')]+[T/'tokenizer.json']; before=[sha(p) for p in paths]
assert [s[:8] for s in before[:3]]==['52dbfccd','9d744164','6d9dfe29'] and before[4]=='98b910bbef07fbbdcffbf13555a7ace2521e3f54d3fd210c4ee2a13556017dca'
bundle=D.load_qualified_source(P/'qualified-source.json'); rows,_=SM.qualified_stage_rows(P/'qualified-source.json',P/'allocation.json',bundle=bundle)
assert len(rows)==514 and len({r['root_id'] for r in rows})==514
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(str(T),local_files_only=True,trust_remote_code=False)
assert tok.is_fast and before[-1]=='cad551d5600a84242d0973327029452a1e3672ba6313c2a3c3d69c4310e12719'
def nt(s):return len(tok(s,add_special_tokens=True,padding=False,truncation=False)['input_ids'])
def q(v):
 v=sorted(x for x in v if x is not None)
 if not v:return None
 def at(p):
  x=(len(v)-1)*p; a=int(x); b=min(a+1,len(v)-1); return round(v[a]+(v[b]-v[a])*(x-a),2)
 return [v[0],at(.5),at(.9),v[-1]]
records=[]; extremes=[]
for r in rows:
 pair=r['pair']; src=S.semantic_pair_pieces(pair,limits=S.SEMANTIC_LIMITS_RESEARCH); leaves=[f for f in src['fields'] if f['leaf']]; body='\n'.join([pair['old']['text'],pair['new']['text'],pair['frame']['background']])
 vals='\n'.join(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,separators=(',',':')) for role in ('frame','old','new') for v in pair[role].values())
 x={'id':r['id'],'split':r['split'],'root':r['root_id'],'cluster':r['cluster_id'],'kind':pair['old']['kind']+'+'+pair['new']['kind'],'family':r['phenomena']['family'],'top':sum(len(pair[s]) for s in ('frame','old','new')),'leaf':len(leaves),'pieces':len(src['fields'])-1,'chars':sum(len(pair[s][k]) for s,k in [('old','text'),('new','text'),('frame','background')]),'raw':nt(src['raw_text']),'values':nt(vals),'body':nt(body),'ok':False,'c':False,'fields':{}}
 try:item=S.prepare_semantic_input(pair,tok,max_length=384,role_independent_frame=r['source_binding']['evidence']['frame_role_independent'],limits=S.SEMANTIC_LIMITS_RESEARCH)
 except S.SemanticRelationError as e:x['error']=e.code; records.append(x); continue
 f,b=item['directions']['forward'],item['directions']['reverse']
 assert f['visible_information_sha256']==b['visible_information_sha256']==item['visible_information_sha256'] and f['complete']==b['complete']==item['complete']
 for v in (f,b):
  assert v['role_body_covered']=={'old':True,'new':True} and v['token_length']==nt(v['text'])<=384
 x.update(ok=True,c=item['complete'],full=item['full_visible_token_lengths'],enc=[f['token_length'],b['token_length']])
 audit=f['fields']; leafpaths={tuple(z['path']) for z in leaves}
 for z in audit:
  if len(z['path'])!=2:continue
  ls=[a for a in audit if tuple(a['path']) in leafpaths and a['path'][:2]==z['path']]; original=sum(a['original_range'][1]-a['original_range'][0] for a in ls); visible=sum(b-a for z in ls for a,b in z['visible_source_ranges']); assert 0<=visible<=original
  path='/'.join(z['path']); x['fields'][path]=[int(z['lost']),int(z['omitted']),original,original-visible]
  if path in ('old/text','new/text','frame/background') and z['lost']:extremes.append([original-visible,r['id'],path,original,visible,x['chars'],x['values'],x['full'][0]])
 records.append(x)
def fields(rr):
 out={}
 for r in rr:
  for k,v in r['fields'].items():out[k]=[a+b for a,b in zip(out.get(k,[0]*4),v)]
 return out
def loss(rr):
 f=fields(rr); z=[]
 for key in ('old/text','new/text','frame/background'):
  a=f.get(key,[0]*4);z.extend([a[3],a[2]])
 pp=[v for k,v in f.items() if k.endswith('/payload')];z.extend([sum(v[3] for v in pp),sum(v[2] for v in pp)])
 z.extend([sum(v[0] for k,v in f.items() if k.endswith('/kind')),sum(v[0] for k,v in f.items() if k.endswith('/permanence'))]);return z
def stats(rr):
 ok=[r for r in rr if r['ok']];c=sum(r['c'] for r in ok)
 return {'N':len(rr),'ok':len(ok),'fail':dict(Counter(r['error'] for r in rr if not r['ok'])),'roots':len({r['root'] for r in rr}),'clusters':len({r['cluster'] for r in rr}),'C':c,'T':len(ok)-c,'root_trunc_pct':round((len(ok)-c)/len(rr)*100,4),'raw':q([r['raw'] for r in rr]),'fullF':q([r['full'][0] for r in ok]),'fullR':q([r['full'][1] for r in ok]),'encF':q([r['enc'][0] for r in ok]),'encR':q([r['enc'][1] for r in ok]),'values':q([r['values'] for r in rr]),'bodyframe':q([r['body'] for r in rr]),'fields':fields(rr)}
def group(rr):
 ok=[r for r in rr if r['ok']]
 return [len(rr),sum(r['c'] for r in ok),len(tr:=[r for r in rr if r['split']=='train']),sum(r['c'] for r in tr),len(de:=[r for r in rr if r['split']=='dev']),sum(r['c'] for r in de),q([r['values'] for r in rr]),q([r['full'][0] for r in ok]),loss(rr)]
def emit(t,x):print(t+' '+json.dumps(x,separators=(',',':'),sort_keys=True),flush=True)
for s in ('train','dev','all'):emit('STAGE_'+s,stats(records if s=='all' else [r for r in records if r['split']==s]))
for axis in ('kind','family','top'):
 g=defaultdict(list)
 for r in records:g[str(r[axis])].append(r)
 emit('GROUP_'+axis,{k:group(v) for k,v in sorted(g.items())})
for axis,bounds in [('leaf',[14,24,40]),('chars',[128,256,512,1024])]:
 g=defaultdict(list)
 for r in records:g[next((str(b) for b in bounds if r[axis]<=b),'over')].append(r)
 emit('STRATA_'+axis,{k:group(v) for k,v in g.items()})
emit('BUDGET_DIAG',{'values_le384':group([r for r in records if r['values']<=384]),'values_gt384':group([r for r in records if r['values']>384]),'bodyframe_le384':group([r for r in records if r['body']<=384]),'bodyframe_gt384':group([r for r in records if r['body']>384]),'piece_count':q([r['pieces'] for r in records]),'raw_body_frame_chars':q([r['chars'] for r in records])})
emit('EXTREMES',sorted(extremes,reverse=True)[:5])
assert before==[sha(p) for p in paths] and not sys.modules['torch'].cuda.is_initialized()
emit('DONE',{'N':len(records),'ok':sum(r['ok'] for r in records),'samevisible_and_both_bodies':sum(r['ok'] for r in records),'hashes_unchanged':True,'blocked':dict(blocked),'cuda_initialized':False})
