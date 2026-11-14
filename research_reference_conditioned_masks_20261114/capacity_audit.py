from data import *
p=read(OUT/'protocol.json');r=read(OUT/'capacity_result.json');scores=np.load(OUT/'capacity_scores.npz');w=np.load(OUT/'capacity_readers.npz');normal=np.load(OUT/'readers.npz');data={v['id']:load_record(v) for v in p['records']};maximum=0.;verified=[]
for key,given in scores.items():
    cid,arm,method,part=key.split('__');case=next(c for c in p['cases'] if c['id']==cid);d=data[case[part]];prefix=cid+'__'+arm;x=(d['features'][arm]-normal[prefix+'__mu'])/normal[prefix+'__sd']
    if method=='unregularized':expected=np.c_[x,np.ones(len(x))]@w[prefix+'__unregularized']
    else:
        ss=[];x=x.astype('float32')
        for seed in [42,43,44]:
            pref=prefix+'__mlp__'+str(seed)+'__';h=np.maximum(0,x@w[pref+'0.weight'].T+w[pref+'0.bias']);s=h@w[pref+'2.weight'].T+w[pref+'2.bias'];s-=s.max(1,keepdims=True);e=np.exp(s);ss.append(e/e.sum(1,keepdims=True))
        expected=np.mean(ss,0)
    err=float(np.max(np.abs(expected-given)));maximum=max(maximum,err);assert err<1e-5
    labels=given.argmax(1)[NEAREST];truth=d['truth'];known=SUPPORT&(truth>=0);cm=np.bincount(truth[known]*3+labels[known],minlength=9).reshape(3,3);tp=int(cm[1,1]);fp=int(cm[:,1].sum()-tp);fn=int(cm[1,:].sum()-tp);neg=known&d['inside']&(truth!=1);n=int(neg.sum());fpi=int(((labels==1)&neg).sum());row=next(v for v in r['records'] if all(v[k]==vv for k,vv in dict(case=cid,arm=arm,method=method,part=part).items()))
    for k,v in dict(tp=tp,fp=fp,fn=fn,inside_negative_pixels=n,inside_false_positive=fpi).items():assert row[k]==v
    assert abs(row['iou']-tp/max(1,tp+fp+fn))<1e-12;assert abs(row['recall']-tp/max(1,tp+fn))<1e-12
    if n:assert abs(row['inside_fpr']-fpi/n)<1e-12
    else:assert row['inside_fpr'] is None
    verified.append(dict(key=key,confusion=cm.tolist(),score_error=err))
    if part=='reference':
        y=d['cell_labels'];ok=y>=0;cc=np.bincount(y[ok]*3+given.argmax(1)[ok],minlength=9).reshape(3,3);reference=next(a for a in r['reference_confusions'] if a['case']==cid and a['arm']==arm and a['method']==method);assert reference['confusion']==cc.tolist()
certificates=[]
for o in r['oracle']:
    case=next(c for c in p['cases'] if c['id']==o['case']);d=data[case[o['part']]];truth=d['truth'];known=SUPPORT&(truth>=0);t=np.bincount(NEAREST[known&(truth==1)],minlength=721);f=np.bincount(NEAREST[known&(truth!=1)],minlength=721);assert t.tolist()==o['target_pixels_per_cell'] and f.tolist()==o['negative_pixels_per_cell'];s=np.array(o['selected_cells']);j=t[s].sum()/(t.sum()+f[s].sum());assert abs(j-o['max_known_target_iou'])<1e-12;bound=np.maximum(t-j*f,0).sum()-j*t.sum();assert abs(bound)<1e-6;certificates.append(dict(case=o['case'],part=o['part'],max_known_target_iou=float(j),optimality_residual=float(bound)))
for n,h in read(OUT/'capacity_protocol.json')['inputs'].items():assert sha(ROOT/n)==h,n
save('capacity_audit.json',dict(passed=True,records=verified,oracle_certificates=certificates,max_score_error=maximum,source_unchanged=True,artifact_hashes={n:sha(OUT/n) for n in ['capacity_result.json','capacity_scores.npz','capacity_readers.npz']}))
print('CAPACITY_AUDIT_PASSED',maximum,len(verified))
