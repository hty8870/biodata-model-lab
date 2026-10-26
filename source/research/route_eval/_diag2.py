                       
import json, subprocess, sys
sys.path.insert(0, 'research/route_eval')
from train_qwen_decider import EVAL_DIR, TASKS
from eval_gguf_decider import build_prompt, top_logprobs, wait_server
port=8126
proc=subprocess.Popen([str(EVAL_DIR/'_llamacpp/cpu-x64/llama-server.exe'),'-m',
    str(EVAL_DIR/'gguf/qwen3_decider_a-Q3_K_M.gguf'),'--port',str(port),'-c','2048','-t','8','-ngl','0','--no-warmup'],
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    wait_server(port,proc)
    for utt,task in [('把刚才那几个数据集打包下载给我','binary'),('有没有小鼠脑的单细胞数据','prelim'),('找找人类肺癌的单细胞数据','binary')]:
        top=top_logprobs(port,build_prompt(utt,task,'<think>\n\n</think>\n\n'))
        opts=TASKS[task]['options']
        print(f'[{task}] {utt[:20]}')
        print('  top8:', [(t,round(l,2)) for t,l in top[:8]])
        lp={t.strip().lower():l for t,l in top}
        print('  选项logit:', {o:round(lp.get(o,-99),3) for o in opts}, '差:', round(lp.get(opts[0],-99)-lp.get(opts[1],-99),3))
finally:
    proc.terminate()
