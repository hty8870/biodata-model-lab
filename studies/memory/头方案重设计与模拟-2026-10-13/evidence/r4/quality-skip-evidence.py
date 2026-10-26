from pathlib import Path
import sys,json,subprocess,time
from scripts.quality_gate import _sanitized_environment
base=Path(__file__).resolve().parent; root=base.parents[3]
runtime=base/'quality-skip-runtime'; runtime.mkdir(exist_ok=False)
env,sanitized,cleared=_sanitized_environment(runtime)
cmd=[sys.executable,'-B','-m','pytest','tests','devcontext/tests','-q','-rs','-p','no:cacheprovider','--basetemp',str(base/'quality-skip-tmp')]
start=time.time(); proc=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
(base/'standard-pytest-skip-reasons.log').write_text(proc.stdout,encoding='utf-8')
record={'command':cmd,'exit_code':proc.returncode,'wall_seconds':time.time()-start,'same_standard_pytest_selection_with_rs_only':True,'standard_full_already_passed':True,'secret_values_sanitized_count':sanitized,'skip_lines':[line for line in proc.stdout.splitlines() if line.startswith('SKIPPED ')]}
(base/'standard-pytest-skip-reasons.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2));raise SystemExit(proc.returncode)
