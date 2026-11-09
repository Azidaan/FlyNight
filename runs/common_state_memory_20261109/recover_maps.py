"""Recover absent snapshots only; prefixes propagated once for each causal path."""
import study as s
from study import ROOT,OUT,read,sha,write,verify,status,np,sys,datetime,MSK,STOP,OTHER
def ready_array(k,name,p):
    item=p['requests'][k]['ready'].get(name)
    if item:
        with np.load(ROOT/item['path'],allow_pickle=False) as z:return z[item['array']].copy()
    return None
def load(k,p):
    arrays={n:ready_array(k,n,p) for n in ['C8','C34','C_OTHER','R1']}
    path=OUT/'maps'/(k+'.npz');marker=path.with_suffix('.json')
    if marker.exists():
        assert sha(path)==read(marker)['sha256']
        with np.load(path,allow_pickle=False) as z:
            for n in z.files:assert arrays[n] is None;arrays[n]=z[n].copy()
    if arrays['C34'] is None and arrays['C8'] is not None and arrays['C_OTHER'] is not None:
        full=np.empty((721,34),np.float32);full[:,4:12]=arrays['C8'];full[:,OTHER]=arrays['C_OTHER'];arrays['C34']=full
    return arrays
def register():
    p=read(OUT/'protocol.json');verify(p);r=read(OUT/'retinal_complete.json');verify(r)
    sources=[OUT/'protocol.json',OUT/'retinal_complete.json',s.Path(__file__)]+list((ROOT/'tools/flyvis-runtime/flyvis/network').glob('*.py'))+list((ROOT/'tools/flyvis-runtime/flyvis/datasets/rendering').glob('*.py'))
    for k,r in p['requests'].items():
        assert s.key(r['video'],r['sequence'])==k
        assert all(r['sequence'][i][0]<=r['sequence'][i+1][0] for i in range(len(r['sequence'])-1))
        assert all(len(c)==4 and 0<=c[0]<c[2]<=1280 and 0<=c[1]<c[3]<=720 for f,c in r['sequence'])
    write('map_protocol.json',dict(registered=s.stamp(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},method='Exact original A path and common B path. Prefixes shared, earlier/latest/current snapshots saved from one state path. Already complete34-channel or8+26 maps never re-inferred; propagation through a ready prefix is only needed for a new suffix. Missing26 outputs checked against any ready8channels<=1e-4. R134 computed only when absent and compared to ready R1 as a witness when inference is needed for FlyVis. Clock, fade-in, weights and normalization unchanged.'))
def run():
    mp=read(OUT/'map_protocol.json');verify(mp);p=read(OUT/'protocol.json');verify(p);ret=read(OUT/'retinal_complete.json');verify(ret)
    if (OUT/'maps_complete.json').exists():verify(read(OUT/'maps_complete.json'));return
    import torch,psutil
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm,receptor_features_rich
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    stats={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']}
    cm,cs=[np.array(stats['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(stats['R1'][n])[:34] for n in ['mean','sd']]
    bb=None;weights=None;done={};parity=[];new=0;retina={}
    (OUT/'maps').mkdir(exist_ok=True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for c in p['cases']:
            if len(c['commonplan']['cells'])<3:continue
            video=c['video']
            if video not in retina:
                with np.load(OUT/'retinal'/(video+'.npz'),allow_pickle=False) as z:retina[video]={(int(f),tuple(map(int,cr))):h.copy() for f,cr,h in zip(z['frames'],z['crops'],z['hex'])}
            for arm,seq,snap in [('A',c['original'],['A_last','A_current']),('B',c['common'],['B_early','B_last','B_current'])]:
                keys={c['snapshots'][name] for name in snap};missing={k for k in keys if load(k,p)['C34'] is None or load(k,p)['R1'] is None}
                if not missing:continue
                if datetime.now(MSK)>=STOP:raise RuntimeError('Scientific save/stop deadline')
                hx=np.stack([retina[video][(f,tuple(cr))] for f,cr in seq]).astype(np.float32);rich=receptor_features_rich(hx[None].astype(float));ends={len(p['requests'][k]['sequence'])-1:k for k in missing}
                need_core=any(load(k,p)['C34'] is None for k in missing)
                if need_core:
                    if bb is None:
                        activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training;assert bb.steps==2 and abs(bb.dt-1/60)<1e-12 and bb.fade_in_seconds==1.;weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
                    x=torch.from_numpy(hx[None]).to(bb.dev);state=bb.init(x[:,:1])
                for t in range(1,max(ends)+1):
                    if need_core:state,f=bb.step(state,x[:,t-1],x[:,t])
                    if t not in ends:continue
                    k=ends[t];a=load(k,p);values={};witness={}
                    r=((rich[t][0].T-rm)/rs).astype(np.float32)
                    if a['R1'] is None:values['R1']=r
                    else:witness['R1']=float(np.max(abs(r-a['R1'])));assert witness['R1']<=1e-4
                    if a['C34'] is None:
                        full=((f[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                        if a['C8'] is not None:
                            witness['C8']=float(np.max(abs(full[:,4:12]-a['C8'])));assert witness['C8']<=1e-4
                            values['C_OTHER']=full[:,OTHER]
                        else:values['C34']=full
                    if values:
                        dest=OUT/'maps'/(k+'.npz');assert not dest.exists()
                        with dest.open('xb') as out:np.savez_compressed(out,**values)
                        write(s.Path('maps')/(k+'.json'),dict(sha256=sha(dest),witness=witness,source_arm=arm,sequence_key=k,protocol_sha256=sha(OUT/'map_protocol.json')));new+=1
                    parity.append(dict(key=k,errors=witness));status('maps',new=new,keys_ready=sum(load(k,p)['C34'] is not None and load(k,p)['R1'] is not None for k in p['requests']),total=len(p['requests']));print('MAP',arm,k,witness,flush=True)
                if need_core:del x,state,f
        if bb is not None:assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    sources={}
    for k in p['requests']:
        a=load(k,p);assert a['C34'] is not None and a['R1'] is not None
        for item in p['requests'][k]['ready'].values():sources[item['path']]=sha(ROOT/item['path'])
        path=OUT/'maps'/(k+'.npz')
        if path.exists():sources[path.relative_to(ROOT).as_posix()]=sha(path)
    verify(mp);verify(p);verify(ret);write('maps_complete.json',dict(passed=True,sources=sources,new_snapshot_files=new,total_snapshots=len(p['requests']),weights_unchanged=True,parity=parity,shared_state_path=True));status('maps_complete',snapshots=len(p['requests']));print('MAPS COMPLETE',flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
