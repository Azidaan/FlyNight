"""Read-only generic cache readiness; no neural or comparison evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASES=('runs/domain_matched_native_20261109','runs/full_template_native_20261109','runs/identity_assignment_20261108/full_precision')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
def relative(path):return str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\','/')
def locate(video,sequence,slot):
    found=[];k=key(video,sequence)
    for base in BASES:
        path=ROOT/base/'features'/video/(k+'.'+slot+'.npy')
        if not path.exists():continue
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        item=dict(path=relative(path),sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype));meta=path.with_suffix('.json')
        if meta.exists():
            m=read(meta);assert m['video']==video and m['sequence']==sequence and m['sha256']==item['sha256'];item.update(metadata=relative(meta),metadata_sha256=sha(meta),retinal_sources=m.get('retinal_sources',{}))
        found.append(item)
    return found
def readiness(video,frame,proposal):
    history=[[int(h['frame']),h['crop']] for h in proposal['history']];current=history+proposal['search'];assert history and current and current[-1][0]==frame and history[-1][0]<frame
    endpoints=[]
    for sequence in [history,current]:
        maps=locate(video,sequence,'C34');assert maps,('full_C34_cache_missing',video,key(video,sequence));assert all(m['shape']==[721,34] for m in maps)
        sources={}
        for m in maps:sources.update(m.get('retinal_sources',{}))
        views=[]
        for f,crop in sequence:
            k=hashlib.sha256(json.dumps([int(f),crop],separators=(',',':')).encode()).hexdigest()[:32];candidates=[n for n in sources if Path(n).stem==k and (ROOT/n).exists()]
            if not candidates:
                candidates=[relative(p) for base in BASES for p in [ROOT/base/'retinal'/video/(k+'.npy')] if p.exists()]
            checked=[]
            for n in sorted(set(candidates)):
                path=ROOT/n;h=sha(path);assert n not in sources or sources[n]==h;arr=np.load(path,mmap_mode='r',allow_pickle=False);assert arr.shape==(721,);checked.append(dict(path=n,sha256=h,shape=list(arr.shape),dtype=str(arr.dtype)))
            views.append(dict(frame=f,crop=crop,ready=bool(checked),files=checked))
        endpoints.append(dict(sequence=sequence,key=key(video,sequence),maps=maps,views=views,retinal_ready=all(x['ready'] for x in views)))
    clock=ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json';anchors=read(clock)['anchors'] if clock.exists() else {}
    return dict(video=video,frame=frame,old_id=proposal['old_id'],raw_id=proposal['raw_id'],proposal_geometry={k:proposal[k] for k in ['reference_frame','reference_box','reference_crop','current_box','current_crop','geometry','winner','full_cells']},endpoints=endpoints,needs_reference_core_parity_pass=True,needs_reference_terminal_hold_pass=True,core_proxy_allowed=False,clock_path=relative(clock) if clock.exists() else None,reference_annotation_anchor=anchors.get(str(history[-1][0])),current_annotation_anchor=anchors.get(str(frame)),evaluation_ready=bool(str(history[-1][0]) in anchors and str(frame) in anchors),scope='Metadata and hashes only; no map/channel/head/Census/appearance outcome computed.')
