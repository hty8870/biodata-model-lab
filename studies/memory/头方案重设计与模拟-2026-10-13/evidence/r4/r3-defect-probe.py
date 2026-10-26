import json
import numpy as np
from dataclasses import replace
from research.memory_relations import head_design as H
p = H.load_protocol()[0]
checks = {}
for gold, pred in [([0,1,2],[1,1,1]),([0,1],[0,0]),([0,1],[1,0])]:
    checks[str((gold,pred))] = H._macro_f1_and_recall(np.array(pred),np.array(gold),len(set(gold)))[0]
objs = [H._object(str(i),'quad',m) for i,m in enumerate([14,13,11,7])]
b = H.EditLibrary(7,15,tuple(objs[:3])); a = H.EditLibrary(7,15,tuple(objs))
checks['four_core'] = H.audit_edit(b,a,H._intent(('3',),allowed_ops=frozenset({'create'})))
checks['frame_migration'] = H.audit_edit(H._library(H._base_objects()),replace(H._library(H._base_objects()),full_mask=31),H._intent(()))
checks['unnormalized_joint'] = H.joint_marginals_feasible([1,0,1],[1,0,0],p)
checks['fake_receipt'] = H.verify_relation_receipt(H.RelationReceipt('obj_alpha','obj_beta',1,1,'fake','fake','new_entails_old_strict'), H._base_objects(),protocol=p)
g = H.study_geometry(p,'smoke'); maps=H.index_maps(p); lp=np.log([[.6,.3,.1],[.6,.3,.1]]); rev=lp.copy();rev[0]=np.nan
entries=H._two_way_entries([(1,1),(2,2)],lp,rev,p)
rows=H._question_rows(dict(candidate='A-13-coarse',arm='A',seed=13,view='coarse',step=100,model_sha256='0'*64),[(1,1),(2,2)],lp,None,lp,None,p,maps,g,two_way=entries)
checks['reverse_failure_rows']=rows
checks['reverse_failure_summary']=H.summarize_rows(rows,p,maps)
print(json.dumps(checks,indent=2,allow_nan=True))
raise SystemExit(1)
