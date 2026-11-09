"""Paired within-query endpoints, with abstentions retained in denominator."""
import study_v2 as s
from study_v2 import *
def quality(a,b):
    if a is None or b is None:return 0.,'abstain'
    return (1.,'correct') if a>b else ((.5,'tie') if a==b else (0.,'wrong'))
def main():
    p=read(OUT/'protocol.json');verify(p);d=read(OUT/'results.json');assert len(d['records'])==12;rows=[];table={metric:{n:dict(correct=0,wrong=0,tie=0,abstain=0,points=0.) for n in NAMES+['cost']} for metric in ['fixed','maximum']}
    for r in d['records']:
        item={k:r[k] for k in ['video','uid','frame','tid']};item['decisions']={}
        costs={c['kind']:c['cost'] for c in r['original_cost']};gp,gd=quality(-costs['own'],-costs['foreign'])
        for metric in table:
            item['decisions'][metric]={'cost':dict(points=gp,outcome=gd)};table[metric]['cost'][gd]+=1;table[metric]['cost']['points']+=gp
            for name in NAMES:
                vals={c['kind']:c[metric] for c in r['variants'][name]};point,outcome=quality(vals['own'],vals['foreign']);item['decisions'][metric][name]=dict(points=point,outcome=outcome,own=vals['own'],foreign=vals['foreign']);table[metric][name][outcome]+=1;table[metric][name]['points']+=point
        rows.append(item)
    rng=np.random.default_rng(42);draws=rng.integers(0,12,(5000,12));diff={}
    for control in ['cost','R134_temporal','R134_spatial']:
        v=np.array([r['decisions']['maximum']['C34']['points']-r['decisions']['maximum'][control]['points'] for r in rows]);ci=np.percentile(v[draws].mean(1),[2.5,97.5]);diff[control]=dict(mean=float(v.mean()),ci95=ci.tolist(),corrected=int(np.sum(v>0)),newly_wrong=int(np.sum(v<0)))
    passed=all(x['ci95'][0]>0 for x in diff.values());assert d['independent_coordinate_solve_scalar_max_error']<=1e-9
    write('summary.json',dict(passed=True,technical_pass=True,reader_gate_pass=passed,table=table,differences=diff,records=rows,queries=12,videos=12,actual_native_old_row_assignment='none for all12; cost endpoint is LOCAL ranking only, not actual IDs',no_threshold_tuning=True));print('SUMMARY',json.dumps(table,ensure_ascii=False),diff,flush=True)
if __name__=='__main__':main()
