"""Просмотр из реальных проверенных рёбер Markdown, не снимок интерфейса Obsidian."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/b123_20261027/graph_fundamental_final'
def main():
    data=json.loads((RUN/'audit.json').read_text(encoding='utf-8'));assert data['passed']
    nodes=data['hypothesis_nodes'];parents=sorted(n for n in nodes if n.startswith('09 Запланировано/'))
    leaves=sorted(n for n in nodes if n.startswith('Гипотезы — подходы/'))
    positions={'Гипотезы':(.5,.08)}
    for x,parent in zip((.17,.5,.83),parents):positions[parent]=(x,.43)
    for x,leaf in zip((.075,.245,.415,.585,.755,.925),leaves):positions[leaf]=(x,.85)
    fig,ax=plt.subplots(figsize=(20,10));fig.patch.set_facecolor('white')
    for a,b in data['hypothesis_edges']:
        xa,ya=positions[a];xb,yb=positions[b];ax.plot([xa,xb],[ya,yb],color='#65738a',linewidth=2.5,zorder=1)
    for name,(x,y) in positions.items():
        ax.scatter([x],[y],s=450 if name=='Гипотезы' else 290,color='#4685cf' if name=='Гипотезы' else '#ed3c45',zorder=3)
        label=Path(name).name
        if name=='Гипотезы':lines=label
        else:lines='\n'.join(textwrap.wrap(label,width=28 if name in leaves else 38,break_long_words=False))
        ax.text(x,y-.035,lines,ha='center',va='top',fontsize=12 if name not in leaves else 10,
            color='#17273a',bbox=dict(facecolor='white',edgecolor='none',pad=3),zorder=2)
    ax.text(.5,.99,'Гипотезы → Б1–Б3 → подходы к проверке',ha='center',va='top',fontsize=20,color='#17273a')
    ax.text(.5,.945,'9 видимых связей внутри дерева. Переходы к статьям и результатам — стелс-связи.',ha='center',fontsize=12,color='#52647a')
    ax.set_xlim(0,1);ax.set_ylim(-.02,1.02);ax.axis('off');fig.tight_layout()
    fig.savefig(RUN/'Гипотезы — проверенная структура.png',dpi=130,bbox_inches='tight')
    fig.savefig(RUN/'Гипотезы — проверенная структура.svg',bbox_inches='tight')
    print('Preview rendered from audited Markdown edges.')
if __name__=='__main__':main()
