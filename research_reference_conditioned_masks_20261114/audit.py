from data import *
p=read(OUT/'protocol.json');r=read(OUT/'result.json');scores=np.load(OUT/'scores.npz');models=np.load(OUT/'readers.npz');arrays=np.load(OUT/'evaluation_arrays.npz');loaded={v['id']:load_record(v) for v in p['records']};checks=[];support=[];maxerr=0.;undefined=[]
for key,score in scores.items():
    cid,arm,control,reader,part=key.split('__');case=next(c for c in p['cases'] if c['id']==cid);d=loaded[case[part]];prefix='__'.join([cid,arm,control,reader]);x=(d['features'][arm]-models[cid+'__'+arm+'__mu'])/models[cid+'__'+arm+'__sd']
    if reader=='ridge':expected=np.column_stack((x,np.ones(len(x))))@models[prefix+'__coef']
    else:expected=(x/np.maximum(np.sqrt((x*x).sum(1,keepdims=True)),1e-8))@models[prefix+'__prototypes'].T
    err=float(np.max(np.abs(expected-score)));maxerr=max(maxerr,err);assert err<1e-10
    labels=score.argmax(1);raster=arrays[key];assert np.array_equal(raster,labels[NEAREST]);truth=arrays['truth__'+case[part]];known=(truth>=0)&arrays['support'];cm=np.bincount(truth[known]*3+raster[known],minlength=9).reshape(3,3);tp=int(cm[1,1]);fp=int(cm[:,1].sum()-tp);fn=int(cm[1,:].sum()-tp);negative=known&arrays['inside__'+case[part]]&(truth!=1);n=int(negative.sum());fi=int(((raster==1)&negative).sum());row=next(a for a in r['records'] if all(a[k]==v for k,v in dict(case=cid,arm=arm,control=control,reader=reader,part=part).items()))
    for k,v in dict(tp=tp,fp=fp,fn=fn,inside_negative_pixels=n,inside_false_positive=fi).items():assert row[k]==v
    assert abs(row['iou']-tp/max(1,tp+fp+fn))<1e-12;assert abs(row['recall']-tp/max(1,tp+fn))<1e-12
    valid_fpr=fi/n if n else None
    if n:assert abs(row['inside_fpr']-valid_fpr)<1e-12
    else:undefined.append(key)
    checks.append(dict(key=key,confusion=cm.tolist(),inside_fpr_valid=valid_fpr))
    if part=='reference' and control=='correct':
        y=d['cell_labels'];valid=y>=0;cc=np.bincount(y[valid]*3+labels[valid],minlength=9).reshape(3,3);support.append(dict(case=cid,arm=arm,reader=reader,confusion=cc.tolist(),balanced_accuracy=float(np.mean(np.diag(cc)/cc.sum(1)))))
primary=[x for x in r['records'] if x['control']=='correct' and x['reader']=='ridge' and x['part']=='query'];means={arm:float(np.mean([x['iou'] for x in primary if x['arm']==arm])) for arm in ['FlyVis+G','Retina+G','Geometry','RGB+G','FlyVis','Retina']};wins=sum(next(x['iou'] for x in primary if x['case']==row['case'] and x['arm']=='FlyVis+G')>row['iou'] for row in primary if row['arm']=='Retina+G');control_means={name:float(np.mean([x['iou'] for x in r['records'] if x['control']==name and x['reader']=='ridge' and x['part']=='query' and x['arm']=='FlyVis+G'])) for name in ['correct','wrong_target','shuffled_labels']}
save('summary.json',dict(scientific_passed=False,quality_passed=False,increment_passed=means['FlyVis+G']>means['Retina+G'] and wins>=2,means=means,wins_over_retina=wins,reference_control_means=control_means,reference_control_passed=control_means['correct']>max(control_means['wrong_target'],control_means['shuffled_labels']),cannot_evaluate_inside_fpr_cases=['day_shop','new_night'],explicit_failure='old2179 has substantial known inside false positives; other2query ROIs contain0annotated negativepixels, so their0FPR was an undefined denominator, not a passed separation test.',source='result.json preserved unchanged; undefined sentinel corrected only in this audit/summary.'))
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('audit.json',dict(passed=True,records=checks,reference_cell_confusions=support,undefined_inside_fpr=undefined,source_unchanged=True,max_score_reconstruction_error=maxerr,important_correction='Original measurement used max(1,denominator) and output0when no negatives existed; explicitly invalid for a passed gate. Preserve original results, require positive annotated denominator in subsequent reporting.',artifact_hashes={n:sha(OUT/n) for n in ['result.json','scores.npz','readers.npz','evaluation_arrays.npz','summary.json']}))
print('AUDIT_COMPLETE',means,'undefined',len(undefined))
