"""Display already audited contrasts; does not select groups or recompute metrics."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_state_subgroups_20261107'
def main():
    source=OUT/'independent_audit.json';a=json.loads(source.read_text(encoding='utf-8'));assert a['passed']
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));groups=list(p['groups']);labels=[p['labels'][g] for g in groups]
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True)
    for col,ct in enumerate(['1.0','0.25']):
        for row,(scenario,metric) in enumerate([('reverse','capped_latency'),('continue','early_epe')]):
            items=[a['paired'][f'{ct}:{scenario}:erase_{g}:{metric}'] for g in groups]
            ci_key='ci_family8' if row==0 and ct=='1.0' else 'ci95'
            y=np.array([q['mean'] for q in items]);interval=np.array([q['ci95' if g=='parent' else ci_key] for g,q in zip(groups,items)]);ax=axes[row,col]
            ax.axhline(0,color='#444444',linewidth=.8)
            ax.bar(np.arange(len(groups)),y,color=['#3f9277' if v<0 else '#d97757' for v in y],width=.65)
            ax.errorbar(np.arange(len(groups)),y,yerr=np.maximum(0,np.stack([y-interval[:,0],interval[:,1]-y])),fmt='none',ecolor='#2f2a24',capsize=3,linewidth=1)
            ax.set_xticks(np.arange(len(groups)),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.set_title(('Обычный контраст' if ct=='1.0' else 'Слабый контраст')+' · '+('разворот' if row==0 else 'движение продолжается'))
            ax.set_ylabel('Изменение задержки, кадры\n(с ограничением 24)' if row==0 else 'Изменение ошибки первых 6 кадров,\nноминальные пиксели/кадр')
    fig.suptitle('Влияние подмены состояния на разворот и продолжение движения',fontsize=14)
    fig.text(.07,.015,'Разница: после подмены минус исходная модель. Ниже нуля — лучше. Восемь новых для серии сцен. Родительская группа — контроль повторяемости.\nИнтервалы: 99,375% для обычного контраста при развороте; остальные и родительская группа 95%, описательные. Донор искусственный.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));dest=OUT/'state_groups_tradeoff.png';fig.savefig(dest,dpi=180);plt.close(fig)
    (OUT/'figure_provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),figure_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
