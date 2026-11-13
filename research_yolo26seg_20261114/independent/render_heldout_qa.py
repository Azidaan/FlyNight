"""Plot original query pixels and stored masks for all three held-out cases."""
from pathlib import Path
import os,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
read=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=read(OUT/'holdout/manifest.json')
bank=np.load(OUT/'stage_heldout/masks.npz')
audit=read(HERE/'heldout_audit_v2.json')
green=np.array([52,218,119])/255;orange=np.array([255,162,50])/255;magenta=np.array([239,90,229])/255
fig,axes=plt.subplots(3,3,figsize=(15,10.6),dpi=150)
metadata=[]
for row,event in enumerate(manifest['events']):
    eid=event['id'];role=next(r for r in event['roles'] if r['role']=='query')
    image=cv2.cvtColor(cv2.imread(str(ROOT/role['image'])),cv2.COLOR_BGR2RGB)
    target=bank[eid+'_query_manual'];other=bank[eid+'_query_other'];auto=bank[eid+'_query_automatic']
    ys,xs=np.where(target|other)
    bx1,by1,bx2,by2=role['box']
    x1=max(0,int(min(xs.min(),bx1))-30);x2=min(image.shape[1],int(max(xs.max()+1,bx2))+30)
    y1=max(0,int(min(ys.min(),by1))-30);y2=min(image.shape[0],int(max(ys.max()+1,by2))+30)
    # Display region depends only on fixed ROI/manual regions, never similarity.
    for col in range(3):
        ax=axes[row,col];ax.imshow(image[y1:y2,x1:x2],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        if col==1:
            layer=np.zeros((*target.shape,4));layer[target]=[*green,.48];layer[other]=[*orange,.48]
            ax.imshow(layer[y1:y2,x1:x2],interpolation='nearest')
        if col==2:
            layer=np.zeros((*auto.shape,4));layer[auto]=[*magenta,.43]
            ax.imshow(layer[y1:y2,x1:x2],interpolation='nearest')
            ax.contour(target[y1:y2,x1:x2].astype(int),levels=[.5],colors=[green],linewidths=1)
            ax.contour(other[y1:y2,x1:x2].astype(int),levels=[.5],colors=[orange],linewidths=1)
        ax.set_title(['Исходный фрагмент','Ручные видимые ядра','Выбранная автоматическая маска'][col]+'\nКадр '+str(role['frame']),fontsize=11)
    title={'B_day_shop':'День: у магазинов','B_night_rain':'Ночь и дождь: цель пропущена','B_dusk_queue':'Сумерки: очередь машин'}[eid]
    axes[row,0].set_ylabel(title,fontsize=11)
    metadata.append(dict(event=eid,image=role['image'],source_sha256=hashlib.sha256((ROOT/role['image']).read_bytes()).hexdigest(),display_crop_xyxy=[x1,y1,x2,y2]))
fig.suptitle('Три новых случая: те же настройки, реальные пиксели и маски',fontsize=15,y=.995)
fig.legend(handles=[Patch(color=green,label='Ядро цели'),Patch(color=orange,label='Ядро соседа'),Patch(color=magenta,label='Автомаска (цвет не ID)')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),fontsize=10)
fig.text(.5,.009,'Яркость исходников не менялась. Ручные ядра частичные. В ночном случае маска истории цели отсутствует, текущая автомаска принадлежит соседу.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.97))
destination=HERE/'heldout_original_manual_auto_QA.png';fig.savefig(destination);plt.close(fig)
(HERE/'heldout_QA_receipt.json').write_text(json.dumps(dict(file=destination.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),images=metadata,source_masks_sha256=hashlib.sha256((OUT/'stage_heldout/masks.npz').read_bytes()).hexdigest(),photometric_enhancement=False,scope='Measured-mask overlays on unchanged source pixels; colors do not imply persistent IDs.'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(str(destination))
