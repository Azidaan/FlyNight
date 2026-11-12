import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'factorial_result.json').read_text(encoding='utf8'));assert json.loads((OUT/'factorial_audit.json').read_text(encoding='utf8'))['passed']
def find(var,pair):return next(r for r in d['records'] if r['old_id']==832 and r['representation']=='C34' and r['query']=='C34' and r['variant']==var and r['pair']==pair and r['head']=='native')
variants=['detector_affine_saved_offset','gt_center_saved_offset'];pairs=[['index','index'],['terminal_hold','index']];z=np.array([[find(v,p)['score'] for p in pairs] for v in variants]);rho=np.array([[find(v,p)['mean_similarity'] for p in pairs] for v in variants]);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
fig=plt.figure(figsize=(10.6,6.8),facecolor='white');ax=fig.add_axes([.22,.30,.65,.49]);im=ax.imshow(z,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-.75,vcenter=0,vmax=.75),aspect='auto');ax.set_xticks([0,1],['Исходная история','Удержание только\nисходного изображения']);ax.xaxis.tick_top();ax.set_yticks([0,1],['Рамки детектора','Центры GT-рамок']);ax.tick_params(length=0,pad=12);ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which='minor',color='white',linewidth=4);ax.tick_params(which='minor',bottom=False,left=False)
for i in range(2):
 for j in range(2):
  val=z[i,j];label=('Принятие' if val>0 else 'Отказ');ax.text(j,i-.10,f'{val:+.6f}'.replace('.',','),ha='center',va='center',fontsize=23,fontweight='bold',color='#17252b');ax.text(j,i+.17,f'{label}; ρ = {rho[i,j]:.3f}'.replace('.',','),ha='center',va='center',fontsize=12,color='#17252b')
for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.90,.30,.022,.49]);cb=fig.colorbar(im,cax=cax,ticks=[-.75,0,.75]);cb.ax.set_yticklabels(['−0,75','0','+0,75']);cb.set_label('Выход замороженной головы',fontsize=10)
fig.text(.04,.94,'832: геометрия и история действуют совместно',fontsize=19,fontweight='bold',ha='left');fig.text(.04,.875,'C34, исходный C34-сдвиг; порог принятия во всех ячейках один: > 0',fontsize=12,ha='left')
fig.text(.04,.19,'Во всех четырёх условиях: те же 149 ячеек, веса, текущая карта и 7 геометрических признаков головы.\nУдержание: ранее сохранённая синтетическая история конечного исходного изображения.\nGT — эталонная рамка; это диагностическое вмешательство с известной разметкой.',fontsize=11,linespacing=1.6,ha='left')
fig.text(.04,.065,'Около 12 пикселей — поправка между двумя правилами координат, а не измеренное движение детали.\nСоответствие физических частей, полная траектория и HOTA здесь не проверены.',fontsize=10,color='#4b5563',linespacing=1.6,ha='left')
for ext in ['png','svg']:fig.savefig(OUT/('832_geometry_reference_state.'+ext),dpi=180)
manifest=dict(source='factorial_result.json',filters=dict(old_id=832,representation='C34',query='C34',head='native',support='common'),variants=variants,pairs=pairs,values=z.tolist(),mean_pearson=rho.tolist(),files={ext:hashlib.sha256((OUT/('832_geometry_reference_state.'+ext)).read_bytes()).hexdigest() for ext in ['png','svg']})
with (OUT/'figure_manifest.json').open('x',encoding='utf8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
print('PLOT SAVED')
