from common import *
import numpy as np,cv2
from scipy.stats import mannwhitneyu
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');inference=read(OUT/'inference_result.json');assert inference['source_unchanged'];maps=np.load(OUT/'holdout_maps.npz');rows=[];arrays={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
def rank(a,b):return float(mannwhitneyu(b,a,method='asymptotic').statistic)/(len(a)*len(b))
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for record in p['records']:
    x1,y1,x2,y2=record['crop'];h,w=y2-y1,x2-x1
    masks={}
    for key in ['target','neighbor']:
        square=cv2.fillPoly(np.zeros((391,391),np.uint8),[np.array(record['polygons'][key],np.int32)],1);masks[key]=cv2.resize(square,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool);assert masks[key].sum()>=10,(record['id'],key)
    assert not (masks['target']&masks['neighbor']).any()
    yy=np.broadcast_to(np.arange(y1,y2)[:,None],(h,w));y_rank=rank(yy[masks['target']],yy[masks['neighbor']])
    sheet=Image.new('RGB',(1173,445),'#20232a');draw=ImageDraw.Draw(sheet);original=cv2.imread(str(ROOT/record['image']))[y1:y2,x1:x2];sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(original,(391,391)),cv2.COLOR_BGR2RGB)),(0,36));draw.text((5,8),'Исходный кадр',font=font,fill='white')
    for j,size in enumerate([518,1036],1):
        d=maps[record['id']+'__'+str(size)][y1:y2,x1:x2];a=d[masks['target']];b=d[masks['neighbor']];probability=rank(a,b);meda=float(np.median(a));medb=float(np.median(b));gate=meda<medb and probability>.75
        rows.append(dict(id=record['id'],video=record['video'],frame=record['frame'],input_size=size,native_crop_size=[w,h],target_pixels=len(a),neighbor_pixels=len(b),target_median=meda,neighbor_median=medb,target_q10_q90=np.quantile(a,[.1,.9]).tolist(),neighbor_q10_q90=np.quantile(b,[.1,.9]).tolist(),pair_rank_probability=probability,vertical_y_rank_probability=y_rank,median_order_correct=meda<medb,gate=gate,depth_exceeds_y=probability>y_rank+1e-12))
        arrays[record['id']+'__'+str(size)+'__target']=a;arrays[record['id']+'__'+str(size)+'__neighbor']=b
        lo,hi=np.quantile(d,[.02,.98]);color=cv2.applyColorMap(np.uint8(np.clip((d-lo)/max(1e-8,hi-lo),0,1)*255),cv2.COLORMAP_TURBO);color=cv2.resize(color,(391,391))
        for key,col in [('target',(0,255,0)),('neighbor',(255,180,20))]:cv2.polylines(color,[np.array(record['polygons'][key],np.int32)],True,col,1)
        sheet.paste(Image.fromarray(cv2.cvtColor(color,cv2.COLOR_BGR2RGB)),(391*j,36));draw.text((391*j+5,8),'Вход '+str(size)+': '+('прошёл' if gate else 'не прошёл'),font=font,fill='white')
    draw.text((5,427),'Контуры ручные; тёплое ближе. Каждая карта имеет свою шкалу.',font=font,fill='white');sheet.save(OUT/(record['id']+'_depth.png'))
with (OUT/'evaluation_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
primary=[r for r in rows if r['input_size']==1036];quality=all(r['gate'] for r in primary);wins=sum(r['depth_exceeds_y'] for r in primary);device=read(OUT/'benchmark_result.json')['device_gate_passed'];inputs_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items());assert inputs_unchanged
save('result.json',dict(records=rows,quality_passed=quality,added_information_passed=wins>=2,scientific_passed=quality and wins>=2 and device,depth_wins_over_y=wins,source_unchanged=True,model_trained=False,primary_input_size=1036,new_videos=3,manual_label_scope='Conservative visible parts and ordinal scene interpretation, not complete segmentation or 3D ground truth.'))
print(json.dumps(rows),flush=True)
