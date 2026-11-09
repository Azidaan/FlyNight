"""Exact accounting at ALREADY chosen winners, no repeated search or tuning."""
import study_v2 as s
from study_v2 import *
from tracking.flyvis_activity import OUTPUT_CELL_TYPES
from tracking.channel_balanced_correspondence import centered_contributions
from tracking.hex_linear_interpolation import sample_linear
def register_attribution():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'));files=[Path(__file__),OUT/'results.json',OUT/'summary.json',OUT/'protocol.json',OUT/'maps_complete.json',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/flyvis_activity.py']
    write('attribution_protocol.json',dict(registered=stamp(),sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},scope='Supplementary accounting for the frozen twelve-query negative result. Read only saved C34-selected winners; do not rerun19 shifts, pick channels, fit weights or alter any decision.',representations=['C34','R134_temporal','R134_spatial'],identity='S=sum_pc [(a_pc-mean(a_c))*(b_pc-mean(b_c)) / (||a_c||*||b_c||*number_of_reference_active_channels)]. Exact constant channels contribute zero. Sum of cell/channel terms must equal the already saved score at the already saved winner.',quantities='Per channel and cell signed contributions for own and foreign. For C34 explicitly report sum from fixed T4/T5 indices4:12 and remaining26, not a new ranking policy. C8 score at C34 winning shift compared descriptively with ready C8 maximum to distinguish channel averaging from winning-shift change.',spatial_caveat='Convert existing winner query to physical pixels, count points inside own GT rectangle only, foreign only, both or neither. Ground truth used only after scoring. Rectangles are not pixel masks; point inclusion does not establish ownership of nonlocal neural activity.',stop='Identity and current maps unchanged, accounting error<=1e-10. No deployment or independent-validation claim. Failed identities remain failures; attribution is algebra, not causal intervention.'))
def attribute():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'attribution_protocol.json'));verify(read(OUT/'maps_complete.json'));r=read(OUT/'results.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;records=[];maximum_error=0.
    for c,row in zip(p['cases'],r['records']):
        a0,a1=[arrays(k,p) for k in c['keys']]
        def features(a):
            inp=a['R1'].astype(float)
            return dict(C34=a['C34'].astype(float),R134_temporal=inp,R134_spatial=neighbor_aug(inp[:,:17].T[None],nb)[0].T)
        ref,cur=features(a0),features(a1);gt={k['kind']:np.array(k['hits'][0]['box']) for k in c['candidates']};ct=np.array(c['sequence'][-1][1]);scale=(ct[2:]-ct[:2])/g['eye_size'];rec=dict(uid=c['uid'],video=c['video'],variants={})
        for name in ['C34','R134_temporal','R134_spatial']:
            vals=[]
            for candidate in c['candidates']:
                saved=next(k for k in row['variants'][name] if k['kind']==candidate['kind']);ix=np.array(candidate['plan']['cells']);query=np.array(candidate['plan']['query'])[:,saved['winner']];patch=sample_linear(reader.mesh,cur[name],query);ac=centered_contributions(ref[name][ix],patch);assert ac is not None;error=abs(ac['balanced_score']-saved['maximum']);maximum_error=max(maximum_error,error);assert error<1e-10
                physical=query*scale+ct[:2];inside={k:((physical>=b[:2])&(physical<=b[2:])).all(1) for k,b in gt.items()};masks=dict(own_only=inside['own']&~inside['foreign'],foreign_only=inside['foreign']&~inside['own'],both=inside['own']&inside['foreign'],neither=~inside['own']&~inside['foreign']);membership={k:dict(count=int(mask.sum()),signed_score_contribution=float(ac['balanced_cell'][mask].sum())) for k,mask in masks.items()};assert sum(v['count'] for v in membership.values())==len(ix)
                item=dict(kind=candidate['kind'],winner=saved['winner'],saved_score=saved['maximum'],channel_contributions=ac['balanced_channel'],cell_contributions=ac['balanced_cell'],physical_query=physical,rectangle_membership=membership,reference_active_channels=int(ac['reference_active'].sum()))
                if name=='C34':
                    item['channel_names']=OUTPUT_CELL_TYPES;item['T4T5_sum']=float(ac['balanced_channel'][4:12].sum());item['other26_sum']=float(np.r_[ac['balanced_channel'][:4],ac['balanced_channel'][12:]].sum());item['ready_C8_maximum']=next(k for k in row['variants']['FlyVis'] if k['kind']==candidate['kind'])['maximum']
                vals.append(item)
            bykind={v['kind']:v for v in vals};delta=np.array(bykind['own']['channel_contributions'])-bykind['foreign']['channel_contributions'];assert abs(delta.sum()-(bykind['own']['saved_score']-bykind['foreign']['saved_score']))<1e-10
            rec['variants'][name]=dict(candidates=vals,own_minus_foreign_by_channel=delta,own_minus_foreign=float(delta.sum()))
        records.append(rec)
    verify(p);verify(read(OUT/'attribution_protocol.json'));write('attribution.json',dict(passed=True,records=records,maximum_accounting_error=maximum_error,zero_new_shift_searches=True,zero_new_model_maps=True,algebra_not_neural_causality=True));print('Attributed already chosen winners',maximum_error,flush=True)
if __name__=='__main__':{'register':register_attribution,'run':attribute}[sys.argv[1]]()
