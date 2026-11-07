"""Служебное продолжение: тот же стимул, исходный срок, кэш оценки по весам."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import b1_residual_train_20261027 as worker
from tracking.b1_fast_stimulus import fast_init
from tracking.z13_readout import FlyBackbone
import torch
import numpy as np


def main():
    p=worker.read(worker.RUN/'protocol.json')
    service=worker.read(worker.RUN/'service_recovery1.json')
    for name,h in service['sha256'].items():assert worker.sha(ROOT/name)==h,name
    FlyBackbone.init=fast_init
    build=worker.build;one=worker.one;context={}
    def build_bound(torch,arm):
        context.clear();context['arm']=arm
        return build(torch,arm)
    def one_bound(torch,bb,head,row,dataset,arm):
        if row['video'] not in p['fit20'] or torch.is_grad_enabled():return one(torch,bb,head,row,dataset,arm)
        folder=worker.RUN/arm;path=folder/'evaluation_cache.json'
        if 'cache' not in context:
            digest=worker.sha(folder/'checkpoint.pt')
            cache=worker.read(path) if path.exists() else dict(checkpoint_sha256=digest,rows={})
            assert cache['checkpoint_sha256']==digest
            context.update(cache=cache,count=0)
        cache=context['cache'];key=str(row['uid']);i=dataset.index[row['uid']]
        if key in cache['rows']:
            updated=torch.as_tensor([cache['rows'][key]],device=bb.dev,dtype=torch.float64)
            return updated,None,None,i
        answer=one(torch,bb,head,row,dataset,arm)
        cache['rows'][key]=answer[0][0].detach().cpu().tolist();context['count']+=1
        if context['count']%25==0:
            worker.write(path,cache)
            worker.status('evaluating frozen residual with saved progress',arm=arm,evaluated_cached=len(cache['rows']),uid=row['uid'])
        return answer
    worker.build=build_bound;worker.one=one_bound
    worker.run()


if __name__=='__main__':main()
