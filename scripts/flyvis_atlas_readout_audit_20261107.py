"""Re-rank stored margins independently; cluster by video and retain every screen result."""
import csv,gzip,json,sys,math
from pathlib import Path
import numpy as np
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flyvis_atlas_readout_20261107 import R,BASE,ID,CON,WEB,io

def audit():
    inputs=io.read(R/'input_checks.json');p=io.read(R/'protocol.json');assert io.sha(R/'raw.npz')==inputs['raw_sha256']
    assert io.sha(BASE/'choices.csv.gz')==p['sources'][(BASE/'choices.csv.gz').as_posix()] and io.sha(R/'protocol.json')==inputs['protocol_sha256']
    with np.load(R/'raw.npz',allow_pickle=False) as f:us,m=f['uid'],f['margins']
    with gzip.open(BASE/'choices.csv.gz','rt') as f:
        labs={int(z['uid']):z for z in csv.DictReader(f) if z['split']=='evaluate26' and z['exactly2']=='1' and z['label_reason']=='ok' and z['exclusive_choice']}
    assert len(labs)==len(set(us))==len(us)==172 and set(labs)==set(us)
    vids=io.read(BASE/'protocol.json')['videos']['evaluate26']
    vi=np.array([vids.index(labs[u]['video']) for u in us]); y=np.array([int(labs[u]['target_side']) for u in us])
    ranks=np.where(abs(m)<=1e-10,0,np.sign(m)).astype(int); assert np.array_equal(ranks[:,1],[int(labs[u]['whole_choice']) for u in us])
    s=(1+ranks*y[:,None])*.5;assert s[:,1].sum()==104
    size=np.bincount(vi,minlength=len(vids)); rng=np.random.default_rng(42)
    draws=rng.integers(len(vids),size=(20000,len(vids)))
    mult=np.stack([(draws==v).sum(axis=1) for v in range(len(vids))],1)
    den=mult@size;mult=mult[den>0];den=den[den>0]
    def signprob(up,down):
        ng=up+down
        if ng==0:return 1.
        return min(1.,2.*sum(math.comb(ng,k) for k in range(min(up,down)+1))/(2**ng))
    def table(cols,base):
        diff=s[:,cols]-s[:,[base]]; bv=np.stack([diff[vi==v].sum(0) for v in range(len(vids))])
        stats=[]
        for o in range(0,len(cols),16):
            boot=(mult@bv[:,o:o+16])*100./den[:,None];ci=np.percentile(boot,[2.5,97.5],axis=0)
            for step,j in enumerate(cols[o:o+16]):
                k=o+step; changes=ranks[:,j]!=ranks[:,base]
                stats.append(dict(credit=float(s[:,j].sum()),percent=float(100*s[:,j].mean()),delta=float(100*diff[:,k].mean()),
                   changes=int(changes.sum()),correctToWrong=int(((s[:,j]==0)&(s[:,base]==1)).sum()),
                   wrongToCorrect=int(((s[:,j]==1)&(s[:,base]==0)).sum()),ties=int((s[:,j]==.5).sum()),
                   ci=ci[:,step].tolist(),signP=signprob(int((bv[:,k]>0).sum()),int((bv[:,k]<0).sum()))))
        return stats
    cf=table([0],1)[0]; rows=table(list(range(2,757)),0)
    pv=[r['signP'] for r in rows];adj=0
    for rank,i in enumerate(np.argsort(pv)):
        adj=max(adj,min(1.,(len(rows)-rank)*pv[i]));rows[i]['holmP']=adj
    for i,r in enumerate(rows):
        r.update(name=p['types'][i] if i<34 else str(i-34),index=i if i<34 else i-34,
                 strongerCredit=float(s[:,757+i].sum()),strongerChanges=int((ranks[:,757+i]!=ranks[:,0]).sum()))
    pr=dict(n=len(us),videos=int((size>0).sum()),types=p['types'],baseline=dict(credit=float(s[:,0].sum()),percent=float(100*s[:,0].mean())),
            t4t5=dict(credit=float(s[:,1].sum()),percent=float(100*s[:,1].mean())),comparison=cf,conditions=rows[:34],positions=rows[34:],
            familySignificant=sum(r['holmP']<.05 for r in rows),raw_sha256=inputs['raw_sha256'])
    io.dump(R/'report.json',pr,True);WEB.mkdir(parents=True,exist_ok=True)
    def export(n,key,data):
        (WEB/n).write_text('window.'+key+'='+json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';\n',encoding='utf-8')
    export('study.js','FLY_STUDY',pr)
    c=io.read(CON); graph=dict(types=[n['name'] for n in c['nodes']],inputs=c['input_units'],outputs=c['output_units'],
        edges=[[e['src'],e['tar'],sum(x[1] for x in e['offsets'])] for e in c['edges']],
        edgeUnit='Сумма контактов в шаблоне пространственных смещений; не число связей всей развёрнутой сети.')
    export('connectome.js','FLY_CONNECTOME',graph)
    result=f"Целые рамки: 34 прямых канала — {pr['baseline']['credit']:g} из 172, T4/T5 — 104; разность {cf['delta']:+.2f} п.п., 95% [{cf['ci'][0]:.2f}; {cf['ci'][1]:.2f}]. Семейная поправка для 755 ослаблений: {pr['familySignificant']} прошли порог 0,05."
    print(result)
    cat=io.read(io.CAT);card=next(x for x in cat['experiments'] if x['id']==ID);assert card['state']=='ready'
    st='completed_positive' if cf['ci'][0]>0 else 'completed_negative' if cf['ci'][1]<0 else 'inconclusive'
    card.update(state=st,done=datetime.now().date().isoformat(),obtained=result+' Источник: '+(R/'report.json').as_posix()+'.',
        interpretation='Прямое сравнение только в фиксированном считывателе; изменение победителя не означает улучшение полного трекера.',
        cannot_claim='Переиспользованы исторические видео; ослаблены нормированные выходы. Не эффект изменения ядра или отдельного биологического нейрона.',
        decision='Сохранены все варианты и исходные баллы; не выбирать новый канал по тем же 172 эпизодам.')
    for item,met,where in zip(card['acceptance'],[True,st=='completed_positive'],['input_checks.json','report.json']):
        item.update(met=met,evidence=(R/where).as_posix())
    card['evidence'].append(dict(kind='file',ref=(R/'report.json').as_posix(),claim='Все 34 типа и 721 позиция, сравнения и неопределённость'))
    io.dump(io.CAT,cat);io.journal('выходы FlyVis, `'+ID+'`. '+result+' Сохранены необработанные баллы и проверка выбора. '+(R/'report.json').as_posix()+'.')
    with Path('FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:f.write('\n## Проверка полного прямого считывания, 07.11.2026\n\n'+result+' Источник: '+(R/'report.json').as_posix()+'. Изменялись только выходы перед считыванием.\n')

if __name__=='__main__':audit()
