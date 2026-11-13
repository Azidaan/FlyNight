"""Show actual saved predictions; no model calls or prediction edits."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path('E:/Dissertation');OUT=Path(__file__).resolve().parent
for var in ['TEMP','TMP','TMPDIR']:os.environ[var]=str(OUT/'tmp')
import hashlib,json
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw,ImageFont

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf'),size)
p=read(OUT/'protocol.json');res=read(OUT/'result.json')
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');pts=np.asarray(g['points'],int)
hull=ConvexHull(pts);support=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),pts[hull.vertices].astype(np.int32),1).astype(bool)
pred=np.load(OUT/'predictions.npz',allow_pickle=False)
manual=np.load(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz',allow_pickle=False)
colors={1:np.array([255,89,105.]),2:np.array([26,216,238.])}
def overlay(rgb,labels,unknown=None):
    out=rgb.astype(float).copy()
    for cls,color in colors.items():out[labels==cls]=out[labels==cls]*.48+color*.52
    if unknown is not None:out[unknown]=out[unknown]*.78+np.array([244,192,60])*.22
    out[~support]*=.35
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)).resize((720,720),Image.Resampling.NEAREST)

for frame in [1164,1165]:
    rec=next(x for x in p['records'] if x['frame']==frame);x1,y1,x2,y2=rec['crop']
    original=cv2.cvtColor(cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{frame}.png')),cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(original[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_LINEAR)
    labels=np.zeros((391,391),np.int8)
    for cls,name in [(1,'target'),(2,'neighbor')]:
        m=manual[f'{frame}_{name}'][y1:y2,x1:x2].astype(np.uint8)
        labels[cv2.resize(m,(391,391),interpolation=cv2.INTER_NEAREST).astype(bool)]=cls
    uncertainty=cv2.resize(manual[f'{frame}_uncertain'][y1:y2,x1:x2].astype(np.uint8),(391,391),interpolation=cv2.INTER_NEAREST).astype(bool)
    canvas=Image.new('RGB',(1536,1970),(19,25,36));d=ImageDraw.Draw(canvas)
    thumb=Image.fromarray(original).resize((384,216),Image.Resampling.LANCZOS)
    td=ImageDraw.Draw(thumb);td.rectangle((x1*.3,y1*.3,x2*.3,y2*.3),outline=(255,213,65),width=3)
    canvas.paste(thumb,(30,24))
    d.text((440,28),'Реальная проба маски FlyVis',font=font(41,True),fill='white')
    d.text((440,88),f'Проверочный кадр {frame}; обучающий кадр — 1159',font=font(28),fill=(205,220,239))
    d.text((440,134),'Ядро заморожено. Обучен только линейный считыватель.',font=font(25),fill=(205,220,239))
    d.text((440,176),'Это те же машины той же сцены, не независимый тест.',font=font(25),fill=(255,203,107))
    d.text((30,245),f'Жёлтая рамка: окно глаза {x2-x1} × {y2-y1} пикселей. В нём 721 ячейка; затемнение — вне их покрытия.',font=font(25),fill=(201,216,236))
    entries=[('Исходное изображение',overlay(rgb,np.zeros((391,391),np.int8)), 'Окно глаза увеличено; цвета исходного кадра.'),
        ('Ручные видимые части',overlay(rgb,labels,uncertainty),'Только для оценки; жёлтым — неопределённая область.'),
        ('Предсказание: FlyVis + считыватель',overlay(rgb,pred[f'FlyVis34_{frame}_raster']),'Готовые нейронные карты, 34 признака на ячейку.'),
        ('Контроль: вход глаза + тот же считыватель',overlay(rgb,pred[f'Retina_history34_{frame}_raster']),'Та же история входа, 34 признака, без ядра FlyVis.')]
    for i,(title,im,caption) in enumerate(entries):
        col=i%2;row=i//2;x=30+col*756;y=302+row*810
        d.text((x,y),title,font=font(27,True),fill='white')
        canvas.paste(im,(x,y+43))
        d.text((x,y+771),caption,font=font(19),fill=(198,212,231))
    d.rectangle((30,1930,54,1952),fill=tuple(colors[1].astype(int)));d.text((64,1926),'Дальняя машина',font=font(23),fill='white')
    d.rectangle((310,1930,334,1952),fill=tuple(colors[2].astype(int)));d.text((344,1926),'Ближняя машина',font=font(23),fill='white')
    fly=next(r for r in res['scores'] if r['frame']==frame and r['arm']=='FlyVis34')
    ctl=next(r for r in res['scores'] if r['frame']==frame and r['arm']=='Retina_history34')
    d.text((660,1926),f'Средний IoU частей: FlyVis {100*fly["car_mean_iou"]:.2f}% / контроль {100*ctl["car_mean_iou"]:.2f}%',font=font(23),fill=(255,203,107))
    canvas.save(OUT/f'flyvis_mask_actual_{frame}.png',compress_level=6)
print('ACTUAL_VISUALS_SAVED')
