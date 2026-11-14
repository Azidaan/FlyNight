from common import *
import cv2,numpy as np
d=read(OUT/'design.json');views=read(OUT/'annotation_views.json')['records'];registry={}
def group(t,n,b):return dict(target=t,neighbor=n,background=b)
registry[771]=group([[202,295],[232,260],[274,250],[322,256],[315,280],[281,299],[207,308]],[[318,331],[369,304],[408,303],[414,435],[303,440],[296,382]],[[477,263],[550,267],[548,388],[479,386]])
registry[777]=group([[209,295],[237,264],[279,252],[331,255],[323,278],[282,300],[213,308]],[[334,326],[393,297],[442,300],[444,431],[320,436],[312,374]],[[494,263],[562,265],[561,386],[498,387]])
registry[789]=group([[206,308],[246,278],[285,266],[345,268],[336,294],[293,311],[213,319]],[[381,313],[432,287],[509,285],[519,443],[365,446],[347,365]],[[210,100],[265,104],[265,168],[213,166]])
registry[1099]=group([[235,298],[265,287],[330,294],[352,310],[340,327],[257,329],[231,319]],[[195,430],[300,410],[453,432],[470,533],[203,530]],[[45,215],[85,214],[84,332],[46,338]])
registry[1111]=group([[238,237],[270,224],[332,235],[353,250],[338,268],[252,270],[233,258]],[[194,363],[316,343],[443,365],[462,462],[191,461]],[[43,207],[80,208],[80,310],[43,312]])
registry[329]=group([],[[204,183],[386,179],[483,194],[489,238],[445,268],[249,246],[183,226]],[[23,374],[98,371],[108,453],[27,455]])
registry[335]=group([],[[492,313],[540,300],[587,314],[589,491],[491,480]],[[122,368],[304,367],[318,474],[128,481]])
registry[347]=group([],[[417,248],[491,221],[588,221],[588,351],[409,352]],[[101,429],[288,414],[315,484],[109,491]])
src=read(ROOT/'research_reference_conditioned_masks_20261114/protocol.json');previous=next(r for r in src['records'] if r['id']=='b23a79d1-43dfeecd_1093');truths={};items=[]
for v in views:
 fr=v['frame'];vid=v['video'];truth=np.full((720,1280),-1,np.int8);x,y,rr,bb=v['annotation_only_crop'];side=rr-x
 if fr==1093:
  a,b,c,e=previous['crop']
  for key,cl in [('background',0),('target',1),('neighbor',2)]:
   polygon=np.array(previous['polygons'][key],'int32');m=cv2.fillPoly(np.zeros((391,391),'uint8'),[polygon],1);local=cv2.resize(m,(c-a,e-b),interpolation=cv2.INTER_NEAREST).astype(bool);patch=truth[b:e,a:c];assert (patch[local]<0).all();patch[local]=cl
  origin='Previously frozen1093 visiblecores, resized nearest from original391eye to nativeimage.'
 else:
  for key,cl in [('background',0),('target',1),('neighbor',2)]:
   poly=registry[fr][key]
   if not poly:continue
   coords=np.rint((np.array(poly)+.5)*side/600-.5+[x,y]).astype('int32');m=cv2.fillPoly(np.zeros(truth.shape,'uint8'),[coords],1).astype(bool);assert (truth[m]<0).all(),(vid,fr,key);truth[m]=cl
  origin='Conservative visiblecores drawn in600-pixel annotation-only view; target unavailable on thirdvideo queries.'
 key=vid+'__'+str(fr);truths[key]=truth;im=cv2.imread(str(OUT/f'{vid}_{fr}.png'))
 for cls,col in [(0,[180,180,180]),(1,[40,240,30]),(2,[255,90,40])]:
  mask=truth==cls;im[mask]=(im[mask]*.4+np.array(col)*.6).astype('uint8')
 cv2.imwrite(str(OUT/f'label_review_{vid}_{fr}.png'),cv2.resize(im[y:bb,x:rr],(600,600)));items.append(dict(case=v['case'],video=vid,frame=fr,counts={str(c):int((truth==c).sum()) for c in [0,1,2]},origin=origin,visible_target=bool((truth==1).any())))
with (OUT/'query_cores.npz').open('xb') as f:np.savez_compressed(f,**truths)
save('query_labels.json',dict(created=now(),records=items,polygons_in_600_view=registry,scope='Manually drawn and visually reviewed by assistant, partialcores only. Thirdvideo has no confidently visible target in requestedframes; counts/FPR on known negatives and total predictedarea reported without an absenttargetIoUwin.'))
print('QUERY_CORES_READY_FOR_VISUAL_REVIEW')
