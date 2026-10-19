import sys,time,json; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(2)
from countworld.scene import SceneCfg
from countworld.train import *
from countworld.evaluate import full_report
mode=sys.argv[1]; pre=sys.argv[2]=='pre'; steps=int(sys.argv[3]); usage=float(sys.argv[4]); score=sys.argv[5]
torch.manual_seed(0)
ag=Agent(V=10,Lmax=1)
cfg=SceneCfg(max_count=9,extra=6)
bf=lambda f:1.5*f
if pre: train_pair(ag,cfg,steps=3000,mode='rl',lr=1e-3,eval_every=250,stop_acc=0.97,beta_fn=lambda f:0.0)
if mode=='sup': train_sup(ag,cfg,steps=steps,lr=1e-3,min_len=1,smode='st',usage=usage,beta_fn=bf)
else: train_rl(ag,cfg,steps=steps,lr=1e-3,min_len=1,usage=usage,beta_fn=bf,score=score,eval_every=250,ent_coef=0.01)
r=full_report(ag,cfg,np.random.default_rng(1),min_len=1)
print('REPORT',json.dumps(r,default=str)[:1800])
