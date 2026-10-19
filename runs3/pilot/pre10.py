import sys; sys.path.insert(0,'.')
import torch
from countworld.scene import SceneCfg
from countworld.train import *
torch.manual_seed(0)
ag=Agent(V=10,Lmax=1)
train_pair(ag,SceneCfg(max_count=9,extra=6),steps=4000,mode='rl',lr=5e-4,eval_every=250,stop_acc=0.985,beta_fn=lambda f:0.0)
torch.save({'bb':ag.bb.state_dict(),'judge':ag.judge.state_dict(),'speaker':ag.speaker.state_dict()},'/tmp/pre10.pt')
