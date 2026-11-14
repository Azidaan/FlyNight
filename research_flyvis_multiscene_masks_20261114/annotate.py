from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
assert not (OUT/'predictions.npz').exists(),'Annotations must precede reader predictions'
p=read(OUT/'protocol.json');test=[r for r in p['records'] if r['split']=='test']
annotations=[
 dict(positive=[[[192,149],[259,142],[277,161],[274,207],[249,231],[177,236],[174,213],[190,191]]],outer=[[102,129],[277,126],[296,163],[290,250],[267,270],[100,271]],negative=[[[134,167],[153,166],[162,202],[157,239],[147,288],[133,289],[134,242],[119,220]]],note='Black vehicle behind cyclist; cyclist certain non-target; uncertain occluded edges excluded.'),
 dict(positive=[[[108,179],[145,175],[160,186],[159,231],[134,244],[106,240]]],outer=[[87,142],[196,142],[205,191],[180,261],[86,268]],negative=[],note='Small dark vehicle left of foreground SUV; foreground occlusion boundary unknown.'),
 dict(positive=[[[109,202],[151,193],[174,176],[225,173],[260,184],[275,204],[273,218],[109,219]]],outer=[[95,204],[143,181],[171,163],[230,163],[271,182],[290,200],[287,229],[93,229]],negative=[],note='Only visible car above divider; divider itself negative.'),
 dict(positive=[[[153,207],[187,190],[221,179],[277,177],[312,187],[328,207],[326,222],[153,222]]],outer=[[135,207],[180,179],[219,161],[282,161],[322,178],[341,205],[340,237],[134,237]],negative=[],note='Dark right-hand car above divider; left car negative.'),
 dict(positive=[[[137,162],[158,161],[169,175],[166,187],[139,187]],[[140,197],[168,196],[173,239],[155,263],[141,246]]],outer=[[110,104],[190,101],[213,145],[205,271],[180,291],[118,287]],negative=[],note='Difficult night parked middle car. Restrict positives to visible rear strip and light; most dark/occluded extent unknown. Neighboring BDD box anchors confirm car category, not pixel contour.'),
 dict(positive=[[[218,151],[239,154],[259,178],[263,224],[245,230],[219,213]],[[193,141],[207,140],[215,165],[198,175]]],outer=[[129,105],[249,104],[288,138],[296,285],[154,295],[128,250]],negative=[],note='Night middle parked car mostly occluded by left car; only narrow visible upper/right cores positive. Neighboring BDD box anchors are not mask ground truth.'),
 dict(positive=[[[117,182],[164,172],[170,163],[231,164],[253,178],[276,191],[277,218],[111,218]]],outer=[[100,163],[160,160],[169,153],[244,154],[272,172],[294,205],[290,237],[102,237]],negative=[],note='Yellow taxi body behind tree; uncertain low body edge omitted.'),
 dict(positive=[[[159,169],[194,162],[264,163],[279,181],[279,203],[155,203]]],outer=[[136,153],[269,151],[293,169],[302,222],[136,229]],negative=[],note='Dark van side; walking person and other vehicles negative outside uncertain outer polygon.')]
arrays={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);sheet=Image.new('RGB',(1564,850),'#202020')
for i,(r,a) in enumerate(zip(test,annotations)):
 def mask(poly):return cv2.fillPoly(np.zeros((391,391),np.uint8),[np.array(poly,np.int32)],1).astype(bool)
 pos=np.zeros((391,391),bool)
 for poly in a['positive']:pos|=mask(poly)
 outer=mask(a['outer'])|pos;neg=~outer
 for poly in a['negative']:neg|=mask(poly)
 assert not (pos&neg).any();arrays[r['id']+'__positive']=pos;arrays[r['id']+'__negative']=neg;a['id']=r['id'];a['positive_pixels']=int(pos.sum());a['negative_pixels']=int(neg.sum())
 x1,y1,x2,y2=r['crop'];im=cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT/r['image']))[y1:y2,x1:x2],(391,391)),cv2.COLOR_BGR2RGB);im[pos]=(im[pos]*.4+[153,96,0]).astype(np.uint8);unknown=~(pos|neg);im[unknown]=(im[unknown]*.5+80).astype(np.uint8);tile=Image.new('RGB',(391,425),'#202020');tile.paste(Image.fromarray(im),(0,34));ImageDraw.Draw(tile).text((3,3),f"{i} {r['video'][:8]}:{r['frame']} orange=core",font=font,fill='white');sheet.paste(tile,((i%4)*391,(i//4)*425))
with (OUT/'manual_test_masks.npz').open('xb') as f:np.savez_compressed(f,**arrays)
sheet.save(OUT/'qa/manual_test_cores.png')
save('manual_test_annotations.json',dict(created=now(),coordinates='391x391 crop of sequential raw frame, before any reader predictions',annotations=annotations,source_sha256=sha(__file__),mask_sha256=sha(OUT/'manual_test_masks.npz'),method='Agent-reviewed conservative visible cores and definite non-target pixels; uncertain contour band omitted. Not official BDD segmentation labels and not an independent human annotator.',extra_box_anchor='data/bdd100k/labels/box_track_20/val/b230132b-ff8f2719.json frames107,108,137; used only to check that ambiguous night targets are annotated cars, not to fabricate contours. These box-label indices are adjacent context, not asserted exact decoded-frame pixel alignment.',no_reader_outcomes_seen=True))
save('teacher_review.json',dict(created=now(),source='qa/teacher_accepted_v2_0.png and qa/teacher_accepted_v2_1.png',exclude_ids={'b1e3e9f5-92377424_266':'Small blurred vehicle, individual silhouette cannot be verified visually.','b20b69d2-ca16c907_538':'Very small partly occluded vehicle adjacent to pole and foreground vehicle; cannot verify teacher instance boundary.'},review='All17 automatically accepted teacher overlays viewed before fitting; remaining are still approximate pseudo-labels, not manually corrected dense contours.',teacher_result='teacher_combined_result.json',teacher_masks='teacher_combined_masks.npz',no_reader_outcomes_seen=True))
print('ANNOTATIONS_FROZEN',len(test))
