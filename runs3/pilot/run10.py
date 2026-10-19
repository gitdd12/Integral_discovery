import sys,json; sys.path.insert(0,'.')
import torch, numpy as np
torch.set_num_threads(2)
from countworld.scene import SceneCfg
from countworld.train import *
from countworld.evaluate import full_report
lr=float(sys.argv[1]); B=int(sys.argv[2]); bmax=float(sys.argv[3]); ent=float(sys.argv[4]); score=sys.argv[5]; steps=int(sys.argv[6])
torch.manual_seed(1)
ag=Agent(V=10,Lmax=1)
st=torch.load('/tmp/pre10.pt'); ag.bb.load_state_dict(st['bb']); ag.judge.load_state_dict(st['judge']); ag.speaker.load_state_dict(st['speaker'])
cfg=SceneCfg(max_count=9,extra=6)
train_rl(ag,cfg,steps=steps,B=B,lr=lr,min_len=1,usage=0.5,beta_fn=lambda f:bmax*f,score=score,eval_every=250,ent_coef=ent)
r=full_report(ag,cfg,np.random.default_rng(1),min_len=1)
print('REPORT',json.dumps(r,default=str)[:1500])
