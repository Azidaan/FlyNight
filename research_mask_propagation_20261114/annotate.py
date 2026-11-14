from common import *
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
# Conservative visible cores drawn on previously inspected 3x details.
# Coordinates are in detail pixels: fullframe=(900+x/3,200+y/3).
shared=dict(target=[[[590,187],[778,185],[844,205],[868,225],[828,239],[653,239],[611,225]],[[711,281],[845,271],[891,286],[891,309],[845,309],[831,298],[764,299],[730,319],[717,358],[704,353]]],wheel=[800,376,27],occluder=[[[171,351],[420,358],[407,489],[248,518],[192,480]],[[574,295],[610,323],[642,394],[643,513],[620,513],[600,404]]],neighbor=[[[978,287],[1125,276],[1125,355],[968,374],[934,344]]],background=[[[710,474],[867,465],[888,564],[721,573]]])
labels={1171:dict(target=[[[350,252],[444,207],[479,199],[467,245],[408,268]],[[204,294],[388,283],[428,286],[417,300],[234,306]]],occluder=[[[175,350],[431,360],[413,494],[247,531],[193,480]],[[711,279],[772,259],[822,307],[828,454],[727,467],[691,421]]],neighbor=[[[891,293],[1125,295],[1125,333],[941,333],[901,345]]],background=[[[880,494],[1100,485],[1100,580],[888,583]]]),1183:shared,1195:dict(shared,occluder=[[[138,274],[255,241],[286,339],[271,491],[159,510],[106,430]],[[576,291],[608,320],[642,394],[643,508],[620,508],[600,404]]]),1207:dict(shared,target=[[[642,188],[777,187],[838,205],[866,226],[826,239],[651,239]],[[711,281],[841,274],[888,286],[885,309],[846,309],[829,298],[764,300],[730,319],[716,358],[703,352]]],occluder=[[[407,245],[521,220],[548,256],[557,398],[520,502],[413,501],[389,359]],[[129,352],[299,352],[293,486],[184,517],[142,467]]])}
arrays={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
for frame,groups in labels.items():
 truth=np.full((720,1280),-1,np.int8)
 for name,cls in [('background',0),('target',1),('neighbor',2),('occluder',3)]:
  for poly in groups[name]:
   points=np.rint(np.array(poly)/3+np.array([900,200])).astype('int32');m=cv2.fillPoly(np.zeros(truth.shape,'uint8'),[points],1).astype(bool);assert (truth[m]<0).all(),(frame,name);truth[m]=cls
 if 'wheel' in groups:
  x,y,r=groups['wheel'];m=cv2.circle(np.zeros(truth.shape,'uint8'),(round(900+x/3),round(200+y/3)),round(r/3),1,-1).astype(bool);assert (truth[m]<0).all();truth[m]=1
 arrays[str(frame)]=truth;im=cv2.imread(str(OUT/f'frame_{frame}.png'))
 for cls,col in [(0,[180,180,180]),(1,[40,240,30]),(2,[255,90,40]),(3,[30,30,245])]:
  m=truth==cls;im[m]=(im[m]*.4+np.array(col)*.6).astype('uint8')
 cv2.imwrite(str(OUT/f'labels_{frame}.png'),cv2.resize(im[200:450,900:1280],(1140,750)))
with (OUT/'manual_query_cores.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('manual_query_cores.json',dict(created=now(),polygons_in_3x_detail=labels,coordinate_rule='fullframe=(900+x/3,200+y/3), round nearest; conservative visiblecores, unknown outside; wheel circles in detailpixels',classes={'0':'background','1':'target SUV','2':'other vehicle','3':'person or refuse cart'},counts={k:{str(c):int((v==c).sum()) for c in [0,1,2,3]} for k,v in arrays.items()},prediction_access=False,scope='Manual visiblecores only, not fullcontours; newframes samevideo. Annotatedbeforemodeloutputs.'))
print('QUERY_CORES_DRAFTED',flush=True)
