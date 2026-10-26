                       
"""一次性诊断：量化档 prelim 选项 token 在 top-20 的缺席率 + 与 torch 的 logit 差。"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, 'research/route_eval')
from train_qwen_decider import EVAL_DIR, SEED, TASKS
from eval_gguf_decider import build_prompt, top_logprobs, wait_server

rows=[json.loads(l) for l in open(EVAL_DIR/'combined_all.jsonl',encoding='utf-8') if l.strip()]
y=np.array([TASKS['route3']['options'].index(r['route']) for r in rows])
idx=np.arange(len(rows))
_,itmp,_,yt=train_test_split(idx,y,test_size=0.4,random_state=SEED,stratify=y)
_,ite,_,_=train_test_split(itmp,yt,test_size=0.5,random_state=SEED,stratify=yt)
exs=[{'task':'prelim','utterance':rows[i]['utterance'],'label':rows[i]['prelim']} for i in sorted(ite.tolist())[:40]]

port=8125
proc=subprocess.Popen([str(EVAL_DIR/'_llamacpp/cpu-x64/llama-server.exe'),'-m',
    str(EVAL_DIR/'gguf/qwen3_decider_a-Q4_K_M.gguf'),'--port',str(port),'-c','2048','-t','8','-ngl','0','--no-warmup'],
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    wait_server(port,proc)
    miss=0; ranks=[]; cases=[]
    for ex in exs:
        top=top_logprobs(port,build_prompt(ex['utterance'],'prelim','<think>\n\n</think>\n\n'))
        toks=[t.strip().lower() for t,_ in top]
        lp={t.strip().lower():l for t,l in top}
        r_m = 'retrieve' not in toks; s_m = 'skip' not in toks
        if r_m or s_m:
            miss+=1
            cases.append((ex['utterance'][:30],ex['label'],r_m,s_m,[(t,round(l,2)) for t,l in top[:6]]))
        else:
            ranks.append(lp['retrieve']-lp['skip'])
    print(f'n={len(exs)} option-miss={miss} ({miss/len(exs):.2%})')
    if ranks: 
        ranks=np.array(ranks); print(f'logit差(retrieve-skip) 在位样本: mean={ranks.mean():.2f} std={ranks.std():.2f} 近零带±1内={np.mean(np.abs(ranks)<1):.2%}')
    for c in cases[:6]: print('MISS例:',c)
finally:
    proc.terminate()
