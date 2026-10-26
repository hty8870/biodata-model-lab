from pathlib import Path
import json,hashlib,shutil
root=Path('<archive-root>')
wt=Path.cwd()
rel=Path('research/memory_relations/runs/v3-coordination')
records=[]
for name in ('source-protocol-frozen.json','preallocation-audit.json'):
    payload=(root/rel/name).read_bytes(); data=json.loads(payload)
    if name.startswith('source'):
        assert data['status']=='targets-not-generated' and data['human_gold_available'] is False
        for row in data['root_assignments']:
            assert set(row)<= {'root_id','author','split','calibration_role','primary_variant','operating_allow'}
        assert hashlib.sha256(payload).hexdigest()=='d2394abb73c985a7bd2878b327d208040be773a5a5d14646a75b6151cbda5cc6'
    else:
        assert data['rows_are_not_generated_yet'] is True and data['human_gold'] is False
    target=wt/rel/name
    if target.exists(): raise SystemExit('fixture target already exists')
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/rel/name,target)
    assert target.read_bytes()==payload
    records.append({'path':str(rel/name),'sha256':hashlib.sha256(payload).hexdigest(),'pure_metadata':True,'copied_after_actual_missing_asset_failure':True})
(wt/'research/memory_relations/runs/head-design-v5-20261013-r4/test-metadata-restoration.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
