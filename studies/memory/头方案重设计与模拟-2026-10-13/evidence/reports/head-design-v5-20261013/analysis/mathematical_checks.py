"""Analytic identifiability, reference gauge and one in-memory capacity census; not learning."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): raise SystemExit('out already exists')
    started=time.time(); cpu=time.process_time(); torch.set_num_threads(1)
    M=np.array([[1,1,0,0,0],[0,0,1,0,1],[0,0,0,1,0]],dtype=np.float64)
    T=np.eye(5)[[0,2,1,3,4]]; dual=np.concatenate([M,M@T]); null=np.array([1,-1,-1,0,1.],dtype=np.float64)
    p1=np.array([.5,0,0,0,.5]); p2=np.array([0,.5,.5,0,0]); pooled=(p1+p2)/2
    assert np.linalg.matrix_rank(M)==3 and np.linalg.matrix_rank(dual)==4 and np.all(dual@null==0)
    assert np.array_equal(dual@p1,dual@p2)
    def softmax(z):
        z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
    rng=np.random.default_rng(20261013); h=rng.normal(size=(10000,64)); W=rng.normal(scale=.03,size=(5,64)); b=rng.normal(scale=.03,size=5)
    z=h@W.T+b; wr=W[:4]-W[4]; br=b[:4]-b[4]; ref=np.column_stack([h@wr.T+br,np.zeros(10000)])
    flat_diff=float(np.max(np.abs(softmax(z)-softmax(ref))))
    rootW=rng.normal(scale=.03,size=(3,64)); rootb=rng.normal(scale=.03,size=3); condW=rng.normal(scale=.03,size=(2,64)); condb=rng.normal(scale=.03,size=2)
    roots=h@rootW.T+rootb; cond=h@condW.T+condb
    rr=np.column_stack([h@(rootW[0]-rootW[1])+(rootb[0]-rootb[1]),np.zeros(10000),h@(rootW[2]-rootW[1])+(rootb[2]-rootb[1])])
    def compose(q,c):
        alpha=1/(1+np.exp(-c[:,0])); beta=1/(1+np.exp(-c[:,1])); return np.column_stack([q[:,0]*alpha,q[:,0]*(1-alpha),q[:,1]*beta,q[:,2],q[:,1]*(1-beta)])
    h_diff=float(np.max(np.abs(compose(softmax(roots),cond)-compose(softmax(rr),cond))))
    assert flat_diff<1e-12 and h_diff<1e-12
    first=nn.Linear(20,64); second=nn.Linear(64,64); flat=nn.Linear(64,5); parent=nn.Linear(64,3); branch=nn.Linear(64,2); k=10.
    with torch.no_grad():
        for module in (first,second,flat,parent,branch): module.weight.zero_(); module.bias.zero_()
        for i in range(10):
            first.weight[i,i]=1; first.weight[i,i+10]=-1
            first.weight[i+10,i]=-1; first.weight[i+10,i+10]=1
            first.weight[i+20,i]=1; first.weight[i+20,i+10]=1; first.bias[i+20]=-1
        second.weight[0,:10]=-1; second.bias[0]=1
        second.weight[1,10:20]=-1; second.bias[1]=1
        second.weight[2,20:30]=-1; second.bias[2]=1
        second.weight[3,:20]=-1; second.bias[3]=1
        flat.weight[0,3]=k; flat.weight[1,0]=k; flat.weight[1,3]=-k; flat.weight[2,1]=k; flat.weight[2,3]=-k; flat.weight[3,2]=k
        flat.weight[4,:4]=torch.tensor([-k,-k,-k,k]); flat.bias[4]=k
        parent.weight[0,0]=k; parent.weight[1,0]=-k; parent.weight[1,2]=-k; parent.bias[1]=k; parent.weight[2,2]=k
        branch.weight[0,3]=2*k; branch.weight[0,0]=-k; branch.weight[1,1]=2*k; branch.bias[1]=-k
    params=lambda m:sum(v.numel() for v in m.parameters())
    body_count=params(first)+params(second); assert body_count==5504 and params(flat)==325 and params(parent)==195 and params(parent)+params(branch)==325
    masks=np.arange(1,1024,dtype=np.int64); prem=np.repeat(masks,1023); hyp=np.tile(masks,1023); overlap=prem&hyp
    gold=np.full(len(prem),4); gold[overlap==0]=3; gold[(overlap==hyp)&(prem!=hyp)]=2; gold[(overlap==prem)&(prem!=hyp)]=1; gold[prem==hyp]=0
    parent_of=np.array([0,0,1,2,1]); reverse_leaf=np.array([0,2,1,3,4]); decoder=np.full((3,3),-1); decoder[0,0]=0;decoder[0,1]=1;decoder[1,0]=2;decoder[2,2]=3;decoder[1,1]=4
    perm=rng.permutation(10); errors={name:0 for name in ('flat5','H5','A_forward','A_reverse','decoded_legal_directions5','illegal_direction_combo')}
    calc_start=time.time()
    with torch.no_grad():
        for start in range(0,len(prem),16384):
            ps,hs=prem[start:start+16384],hyp[start:start+16384]
            pb=((ps[:,None]>>np.arange(10))&1).astype(np.float32)[:,perm]; hb=((hs[:,None]>>np.arange(10))&1).astype(np.float32)[:,perm]
            x=torch.from_numpy(np.concatenate([pb,hb],1)); xr=torch.from_numpy(np.concatenate([hb,pb],1))
            feat=F.relu(second(F.relu(first(x)))); revfeat=F.relu(second(F.relu(first(xr))))
            pred=flat(feat).argmax(1).numpy(); ar=parent(feat).argmax(1).numpy(); brp=parent(revfeat).argmax(1).numpy()
            q=F.log_softmax(parent(feat),1); c=branch(feat)
            log5=torch.stack([q[:,0]+F.logsigmoid(c[:,0]),q[:,0]+F.logsigmoid(-c[:,0]),q[:,1]+F.logsigmoid(c[:,1]),q[:,2],q[:,1]+F.logsigmoid(-c[:,1])],1)
            g=gold[start:start+len(ps)]
            errors['flat5']+=int((pred!=g).sum());errors['H5']+=int((log5.argmax(1).numpy()!=g).sum())
            errors['A_forward']+=int((ar!=parent_of[g]).sum());errors['A_reverse']+=int((brp!=parent_of[reverse_leaf[g]]).sum())
            decoded=decoder[ar,brp];errors['decoded_legal_directions5']+=int((decoded!=g).sum()); errors['illegal_direction_combo']+=int((decoded<0).sum())
    assert not any(errors.values())
    result={'ranks':{'parent_rank':int(np.linalg.matrix_rank(M)),'dual_rank':int(np.linalg.matrix_rank(dual)),'parent_kernel_dim':2,'dual_kernel_dim':1,'M':M.tolist(),'dual_map':dual.tolist(),'dual_nullvector':null.tolist()},
            'nonidentifiability':{'p1':p1.tolist(),'p2':p2.tolist(),'common_dual_marginal':(dual@p1).tolist(),'pooled':pooled.tolist(),'perfect_joint_nll':math.log(2),'marginal_only_optimal_nll':math.log(4),'gap_nats':math.log(2),'analytic_not_ML_gain':True,'requires_decoder_observes_only_marginals':True},
            'gauge':{'own_seed':20261013,'samples':10000,'hidden':64,'dtype':'float64','flat_max_probability_difference':flat_diff,'H_max_probability_difference':h_diff,'head_params_original':325,'head_params_reference':260,'saved':65,'new_artifact_exported':False,'optimizer_trajectory_invariant':False,
                     'coordinator_external_review_values':{'flat':6.661338147750939e-16,'H':5.551115123125783e-16,'attribution':'provided independent architecture review; not claimed as own observed values'}},
            'capacity':{'permutations_actually_run':1,'coordinate_permutation':perm.tolist(),'pairs':len(prem),'body_sample_passes':2*len(prem),'hard_errors':errors,'candidate_parameters':{'A':body_count+params(parent),'C':body_count+params(flat),'H':body_count+params(parent)+params(branch)},'kappa':k,'compute_wall_seconds':time.time()-calc_start,'weights_saved':False,'not_learning_holdout_NLP_or_architecture_ranking':True,
                        'coordinator_external_capacity_result':{'permutations':10,'pairs':10465290,'body_sample_passes':20930580,'all_hard_errors':0,'compute_seconds_approx':35.9,'startup_total_seconds_approx':41.6,'combined_allheads_parameters':6154,'attribution':'provided external assistant observation, separate from our one-permutation run'}},
            'runtime':{'python':__import__('platform').python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':torch.get_num_threads()},'wall_seconds':time.time()-started,'cpu_seconds':time.process_time()-cpu}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
