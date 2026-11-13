"""Plot measured masks over unchanged source pixels; no image enhancement."""
from pathlib import Path
import os,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
manifest=json.loads((HERE.parent/'case2179.json').read_text(encoding='utf8'))['events'][0]
masks=np.load(HERE.parent/'stage_2179/masks.npz')
original_masks=np.load(ROOT/manifest['masks'])
receipt=json.loads((HERE/'2179_audit_v2.json').read_text(encoding='utf8'))
crop=(890,225,1280,500)
green=np.array([52,218,119])/255
orange=np.array([255,162,50])/255
magenta=np.array([239,90,229])/255
fig,axes=plt.subplots(3,3,figsize=(15,10.7),dpi=150)
for row,role in enumerate(manifest['roles']):
    frame=role['frame'];image=cv2.cvtColor(cv2.imread(str(ROOT/role['image'])),cv2.COLOR_BGR2RGB)
    red=original_masks[str(frame)+'_target'];black=original_masks[str(frame)+'_neighbor']
    auto=masks['2179_'+role['role']+'_automatic']
    x1,y1,x2,y2=crop
    for col in range(3):
        ax=axes[row,col];ax.imshow(image[y1:y2,x1:x2],interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        if col==1:
            rgba=np.zeros((*red.shape,4),float)
            rgba[red]=[*green,.48];rgba[black]=[*orange,.48]
            ax.imshow(rgba[y1:y2,x1:x2],interpolation='nearest')
        if col==2:
            rgba=np.zeros((*auto.shape,4),float);rgba[auto]=[*magenta,.43]
            ax.imshow(rgba[y1:y2,x1:x2],interpolation='nearest')
            ax.contour(red[y1:y2,x1:x2].astype(int),levels=[.5],colors=[green],linewidths=.9)
            ax.contour(black[y1:y2,x1:x2].astype(int),levels=[.5],colors=[orange],linewidths=.9)
        prefix=['Исходный фрагмент','Ручные видимые ядра','Выбранная автоматическая маска'][col]
        ax.set_title(prefix+'\nКадр '+str(frame),fontsize=11)
    label={'target_ref':'История дальней машины','competitor_ref':'История ближней машины','query':'Текущая смешанная рамка'}[role['role']]
    axes[row,0].set_ylabel(label,fontsize=11)
fig.suptitle('2179: реальные пиксели и сохранённые маски YOLO26n-seg',fontsize=15,y=.995)
fig.legend(handles=[Patch(color=green,label='Ядро дальней машины'),Patch(color=orange,label='Ядро ближней машины'),Patch(color=magenta,label='Выбранная автомаска (цвет не ID)')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),fontsize=10)
fig.text(.5,.008,'Одинаковый фрагмент x=890…1279, y=225…499; исходная яркость сохранена. Ручные ядра — частичные области, не полные контуры.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.97))
destination=HERE/'2179_original_manual_auto_QA.png'
fig.savefig(destination);plt.close(fig)
info=dict(file=destination.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),source_images={role['image']:hashlib.sha256((ROOT/role['image']).read_bytes()).hexdigest() for role in manifest['roles']},mask_sha256=hashlib.sha256((HERE.parent/'stage_2179/masks.npz').read_bytes()).hexdigest(),display_crop_xyxy=list(crop),photometric_enhancement=False,display='Nearest-neighbor source display plus measured-mask alpha overlays/contours. Colors indicate mask layers, not persistent IDs.',quantitative_source='2179_audit_v2.json')
(HERE/'2179_QA_receipt.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(str(destination))
