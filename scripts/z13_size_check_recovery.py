"""Read-only diagnosis of the size invariant using identical backbone features."""
from pathlib import Path
import sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_candidates_night as n

class Capture:
    def __init__(self, backbone): self.backbone=backbone; self.features=[]
    def init(self,x): self.features=[]; return self.backbone.init(x)
    def step(self,state,last,current):
        state,features=self.backbone.step(state,last,current)
        self.features.append(features.detach().clone())
        return state,features

class Replay:
    def __init__(self,features): self.features=features; self.pos=0
    def init(self,x): self.pos=0; return None
    def step(self,state,last,current):
        feature=self.features[self.pos]; self.pos+=1
        return None,feature

def diagnose():
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.z13_candidate_modules import run_candidate
    from tracking.z13_readout import run_sequence
    from tracking.z13_spatial import make_spatial_readout
    torch.set_num_threads(2)
    p=n.read(n.OUT/'protocol.json')
    for name,digest in p['source_sha256'].items(): assert n.sha(ROOT/name)==digest,name
    folder=n.OUT/'C/size'; prior=n.read(folder/'prediction_progress.json')
    data=n.contextual_data(p['evaluation'])
    by={}
    for i,l in enumerate(data.labels):
        if not l[8]: by.setdefault(l[0],[]).append(i)
    eps=sorted(by); start=prior['next']; observations=[]
    with GpuLock('flyvis_fovea_gpu'):
        est,bank,geom,bb,ro,dev,base=n.build(torch,'C','size')
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert ck['step']==500
        ro.load_state_dict(ck['readout']); bb.net.load_state_dict(ck['backbone']); ro.eval()
        assert all(torch.equal(v.cpu(),base['readout'][k].cpu()) for k,v in ro.state_dict().items() if k in base['readout'])
        parent=make_spatial_readout(torch)(34,geom.points,bank.eye_size,geom.neighbors).to(dev)
        parent.load_state_dict(base['readout']); parent.eval()
        end=time.monotonic()+240
        with torch.no_grad():
            for left in range(start,min(start+100,len(eps)),4):
                assert n.now().isoformat()<n.SOFT and time.monotonic()<end
                es=eps[left:left+4]
                hx,mask,crops=n.batch(data,torch,dev,es,[max(data.labels[i][1] for i in by[e]) for e in es])
                cap=Capture(bb)
                got,_=run_candidate(torch,cap,ro,hx,mask,crops,geom,bank.eye_size)
                same=run_sequence(torch,Replay(cap.features),parent,hx,mask)
                other=Capture(bb)
                repeat=run_sequence(torch,other,parent,hx,mask)
                row=dict(left=left,shared_max=float((got-same).abs().max()),shared_exact=bool(torch.equal(got,same)),
                    repeated_max=float((got-repeat).abs().max()),repeated_close=bool(torch.allclose(got,repeat,atol=1e-5,rtol=1e-5)),
                    features_max=max(float((a-b).abs().max()) for a,b in zip(cap.features,other.features)))
                observations.append(row)
                n.write(n.OUT/'size_recovery_diagnostic.json',dict(finished=False,observations=observations,checkpoint_sha256=n.sha(folder/'checkpoint.pt')))
                assert row['shared_exact'],row
                if not row['repeated_close']: break
    n.write(n.OUT/'size_recovery_diagnostic.json',dict(finished=True,shared_exact=all(r['shared_exact'] for r in observations),
        separate_pass_mismatch=any(not r['repeated_close'] for r in observations),observations=observations,
        checkpoint_sha256=n.sha(folder/'checkpoint.pt'),script_sha256=n.sha(Path(__file__)),completed=n.now().isoformat()))

if __name__=='__main__': diagnose()
