"""Display already audited contrasts; does not select groups or recompute metrics."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_partial_state_swap_20261106'
def main():
    source=OUT/'independent_audit.json';a=json.loads(source.read_text(encoding='utf-8'));assert a['passed']
    groups=['retina','intermediate','t4','t5','other_readout','t4t5'];labels=['R1–R8','Промежуточные','T4','T5','Прочие\nсчитываемые','T4 + T5']
    fig,axes=plt.subplots(2,2,figsize=(12,8.2),sharex=True)
    for col,ct in enumerate(['1.0','0.25']):
        for row,(scenario,metric) in enumerate([('reverse','capped_latency'),('continue','early_epe')]):
            items=[a['paired'][f'{ct}:{scenario}:erase_{g}:{metric}'] for g in groups]
            ci_key='ci_family6' if row==0 and ct=='1.0' else 'ci95'
            y=np.array([q['mean'] for q in items]);interval=np.array([q[ci_key] for q in items]);ax=axes[row,col]
            ax.axhline(0,color='#444444',linewidth=.8)
            ax.bar(np.arange(6),y,color=['#3f9277' if v<0 else '#d97757' for v in y],width=.65)
            ax.errorbar(np.arange(6),y,yerr=np.maximum(0,np.stack([y-interval[:,0],interval[:,1]-y])),fmt='none',ecolor='#2f2a24',capsize=3,linewidth=1)
            ax.set_xticks(np.arange(6),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.set_title(('Обычный контраст' if ct=='1.0' else 'Слабый контраст')+' · '+('разворот' if row==0 else 'движение продолжается'))
            ax.set_ylabel('Изменение задержки, кадры\n(с ограничением 24)' if row==0 else 'Изменение ошибки первых 6 кадров,\nноминальные пиксели/кадр')
    fig.suptitle('Влияние подмены состояния на разворот и продолжение движения',fontsize=14)
    fig.text(.07,.025,'Разница: после подмены минус исходная модель. Ниже нуля — лучше. Четыре ранее изученные сцены.\nИнтервалы: 99,1667% для обычного контраста при развороте; остальные 95%, описательные. Донор искусственный.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));dest=OUT/'state_groups_tradeoff.png';fig.savefig(dest,dpi=180);plt.close(fig)
    (OUT/'figure_provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),figure_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
