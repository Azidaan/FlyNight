from data import *
p=read(OUT/'protocol.json');records={r['id']:load_record(r) for r in p['records']};cs=np.load(OUT/'color_scores.npz');cw=np.load(OUT/'color_readers.npz');oldw=np.load(OUT/'capacity_readers.npz');oldnorm=np.load(OUT/'readers.npz');oldscore=np.load(OUT/'capacity_scores.npz');ts=np.load(OUT/'tensor_scores.npz');ps=np.load(OUT/'position_scores.npz');pw=np.load(OUT/'position_readers.npz');maximum=0.;checks=[]
def forward(w,prefix,x):
    x=x.astype('float32');values=[]
    for seed in [42,43,44]:
        pref=prefix+str(seed)+'__';h=np.maximum(0,x@w[pref+'0.weight'].T+w[pref+'0.bias']);s=h@w[pref+'2.weight'].T+w[pref+'2.bias'];s-=s.max(1,keepdims=True);e=np.exp(s);values.append(e/e.sum(1,keepdims=True))
    return np.mean(values,0)
def independent_metric(pred,d):
    truth=d['truth'];known=(truth>=0)&SUPPORT;tp=int((pred&known&(truth==1)).sum());fp=int((pred&known&(truth!=1)).sum());fn=int((~pred&known&(truth==1)).sum());negative=known&d['inside']&(truth!=1);n=int(negative.sum());fi=int((negative&pred).sum());return dict(tp=tp,fp=fp,fn=fn,iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),inside_negative_pixels=n,inside_false_positive=fi,inside_fpr=fi/n if n else None)
def check_metric(row,pred,d):
    exact=independent_metric(pred,d)
    for k,v in exact.items():
        if v is None:assert row[k] is None
        else:assert abs(row[k]-v)<1e-12,(k,row[k],v)
def color_variants(d):
    rgb=d['features']['RGB+G'];gray=rgb.copy();gray[:,:3]=(rgb[:,:3]@np.array([.114,.587,.299]))[:,None];gray[:,3:6]=(rgb[:,3:6]@np.array([.114,.587,.299]))[:,None];f=d['features']['FlyVis'];z=np.zeros_like(f);return dict(RGB8=rgb,Gray8=gray,Gray8frozen=gray,FlyRGB76=np.c_[f,rgb],ZeroFlyRGB76=np.c_[z,rgb],FlyGray76=np.c_[f,gray],ZeroFlyGray76=np.c_[z,gray])
cr=read(OUT/'color_result.json');tr=read(OUT/'tensor_result.json');pr=read(OUT/'position_result.json')
for case in [c for c in p['cases'] if c['primary']]:
    cid=case['id'];rd=records[case['reference']];qd=records[case['query']]
    for part,d in [('reference',rd),('query',qd)]:
        vv=color_variants(d)
        for arm,raw in vv.items():
            key=cid+'__'+arm+'__'+part;given=cs[key]
            if arm=='RGB8':expected=oldscore[cid+'__RGB+G__mlp__'+part]
            elif arm=='Gray8frozen':expected=forward(oldw,cid+'__RGB+G__mlp__',(raw-oldnorm[cid+'__RGB+G__mu'])/oldnorm[cid+'__RGB+G__sd'])
            else:expected=forward(cw,cid+'__'+arm+'__',(raw-cw[cid+'__'+arm+'__mu'])/cw[cid+'__'+arm+'__sd'])
            error=float(abs(expected-given).max());maximum=max(maximum,error);assert error<1e-5;row=next(r for r in cr['records'] if r['case']==cid and r['arm']==arm and r['part']==part);check_metric(row,given.argmax(1)[NEAREST]==1,d);checks.append(dict(kind='color',key=key,error=error))
    prefix=cid+'__FlyRGB76';base=(color_variants(qd)['FlyRGB76']-cw[prefix+'__mu'])/cw[prefix+'__sd'];reference=(color_variants(rd)['FlyRGB76']-cw[prefix+'__mu'])/cw[prefix+'__sd']
    for mode in ['baseline','zero_c34','query_recenter','query_rescale','spatial_shuffle','reference_c34']:
        x=base.copy()
        if mode=='zero_c34':x[:,:68]=0
        elif mode=='query_recenter':x[:,:68]=base[:,:68]-base[:,:68].mean(0)
        elif mode=='query_rescale':x[:,:68]=(base[:,:68]-base[:,:68].mean(0))/np.maximum(base[:,:68].std(0),1e-8)
        elif mode=='spatial_shuffle':x[:,:68]=base[np.random.default_rng(42).permutation(721),:68]
        elif mode=='reference_c34':x[:,:68]=reference[:,:68]
        assert np.array_equal(x[:,68:],base[:,68:]);given=ts[cid+'__'+mode];expected=forward(cw,prefix+'__',x);error=float(abs(expected-given).max());maximum=max(maximum,error);assert error<1e-5;row=next(r for r in tr['records'] if r['case']==cid and r['variant']==mode);check_metric(row,given.argmax(1)[NEAREST]==1,qd);checks.append(dict(kind='tensor',key=cid+'__'+mode,error=error))
    f=np.concatenate([fun(2*np.pi*(PTS[:,axis,None]/391)*np.geomspace(.5,8,17)) for axis in [0,1] for fun in [np.sin,np.cos]],1)
    for part,d in [('reference',rd),('query',qd)]:
        x=(np.c_[f,d['features']['RGB+G']]-pw[cid+'__mu'])/pw[cid+'__sd'];given=ps[cid+'__'+part];expected=forward(pw,cid+'__',x);error=float(abs(expected-given).max());maximum=max(maximum,error);assert error<1e-5;checks.append(dict(kind='position',key=cid+'__'+part,error=error))
    position_preds={'FourierRGB76':ps[cid+'__query'].argmax(1)[NEAREST]==1,'copy_FlyRGB_reference':cs[cid+'__FlyRGB76__reference'].argmax(1)[NEAREST]==1,'copy_ZeroFlyRGB_reference':cs[cid+'__ZeroFlyRGB76__reference'].argmax(1)[NEAREST]==1,'copy_manual_reference':rd['truth']==1}
    for mode,pred in position_preds.items():check_metric(next(r for r in pr['records'] if r['case']==cid and r['variant']==mode),pred,qd)
for protocol in ['color_protocol.json','tensor_protocol.json','position_protocol.json']:
    for n,h in read(OUT/protocol)['inputs'].items():assert sha(ROOT/n)==h,n
alignment=read(ROOT/'research_flyvis_multiscene_masks_20261114/alignment_audit.json');assert alignment['passed'];source=ROOT/'tracking/foveated_eyes.py';assert sha(source)==read(ROOT/'research_input_precision_probe_20261112/protocol.json')['inputs']['tracking/foveated_eyes.py']
save('diagnostics_audit.json',dict(passed=True,checks=checks,max_score_error=maximum,source_unchanged=True,current_RGB_and_coordinates_unchanged_in_tensor_controls=True,grayscale_provenance=dict(foveated_eyes_sha256=sha(source),gray_conversion_line=117,verified_retinal_alignment_records=len(alignment['records']),renderer='research_flyvis_multiscene_masks_20261114/prepare_data_v3.py:36',scope='Actual archived retina preparation converts BGR to grayscale before Lamina. Color diagnostic itself uses linear luminance without uint8 rounding and is not an exact replica of the full Lamina path.'),artifact_hashes={n:sha(OUT/n) for n in ['color_result.json','color_scores.npz','color_readers.npz','tensor_result.json','tensor_scores.npz','position_result.json','position_scores.npz','position_readers.npz']}))
print('DIAGNOSTICS_AUDITED',maximum,len(checks))
