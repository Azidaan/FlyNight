from common import *
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
f=next(r for r in read(OUT/'correspondence.json')['records'] if r['old_id']==574)
m=next(r for r in read(OUT/'box_membership.json')['records'] if r['old_id']==574)
img=np.array(Image.open(ROOT/'research_failure_localization_20261112/frames/574_937_full.png'))
q=np.array(f['originals']['C34']);box=np.array(m['current_candidate_GT_box']);inside=((q>=box[:2])&(q<=box[2:])).all(1)
assert int((~inside).sum())==50
fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
det=np.array(next(c['case']['current_box'] for c in read(OUT/'protocol.json')['cases'] if c['case']['old_id']==574))
for ax in axes:
    ax.imshow(img);ax.set_xlim(460,520);ax.set_ylim(389,333);ax.set_xlabel('x, пиксели исходного кадра');ax.set_ylabel('y, пиксели')
    for bb,color,label in [(m['current_old_GT_box'],'lime','Прежний объект 00115019'),(m['current_candidate_GT_box'],'orange','Разметка кандидата 00115011'),(det,'red','Рамка детектора')]:
        x1,y1,x2,y2=bb;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,lw=1.6,label=label))
axes[0].set_title('Разметка двух объектов и детекция');axes[0].legend(loc='upper left',fontsize=8)
axes[1].scatter(q[inside,0],q[inside,1],s=9,c='cyan',label='70 точек внутри разметки кандидата')
axes[1].scatter(q[~inside,0],q[~inside,1],s=13,c='magenta',marker='x',label='50 точек вне разметки кандидата')
axes[1].set_title('Где исходный считыватель берёт значения');axes[1].legend(loc='upper left',fontsize=8)
fig.suptitle('Случай 574: сходство считается по области вне разметки кандидата\nРамки показывают геометрию разметки, а не принадлежность каждого пикселя',fontsize=11)
fig.savefig(OUT/'support_574_v2.png',dpi=150);plt.close(fig)
save('plot_manifest_v2.json',dict(passed=True,image='support_574_v2.png',sha256=sha(OUT/'support_574_v2.png'),sources={n:sha(OUT/n) for n in ['correspondence.json','box_membership.json','plot_support_v2.py']}))
