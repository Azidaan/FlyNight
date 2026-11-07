"""Дополнительный описательный контроль знака только во время движения."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery2_20261104'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json')
 mask=np.load(OUT/'geometry.npz')['mask'];rows=[]
 for c in p['cases']:
  if c['split']!='evaluation':continue
  arms={}
  for arm in ('stock','fast'):
   f=OUT/arm/'outputs'/(c['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256'];q=np.load(f)['flow'][0,1][:,mask].mean(1)
   for mode in ('raw','subtract'):
    v=q.copy()
    if mode=='subtract':v-=q[11:21].mean(0)
    v*=r['gains'][arm+'_'+mode];u=v[31:55,c['axis']]
    arms[arm+'_'+mode+'_calibrated']=dict(positive_fraction=float(np.mean(u>0)),mean_signed_output=float(u.mean()),
      mean_abs_error=float(np.abs(u-c['speed']).mean()))
  rows.append(dict(case=c,arms=arms))
 groups={}
 for start in ('still','moving'):
  for ct in (1.,.25):
   cohort=[x for x in rows if x['case']['start']==start and x['case']['contrast']==ct];assert len(cohort)==16
   groups[f'{start}_{ct}']={}
   for arm in cohort[0]['arms']:
    groups[f'{start}_{ct}'][arm]={key:float(np.mean([x['arms'][arm][key] for x in cohort])) for key in ('positive_fraction','mean_signed_output','mean_abs_error')}
 data=dict(complete=True,descriptive_post_hoc=True,cases=len(rows),frames='31–54: движение до разворота для обоих начал',groups=groups,rows=rows,
  original_report_sha256=sha(OUT/'report.json'),source_sha256=sha(Path(__file__)),
  note='Дополнение без изменения заранее зарегистрированного критерия. В исходном поле correct_before_fraction интервал21–54 включает неподвижные кадры в условии still; его нельзя называть долей правильного направления.')
 (OUT/'predirection_check.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(groups,ensure_ascii=False))
if __name__=='__main__':main()
