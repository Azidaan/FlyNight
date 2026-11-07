"""Проверить источник уже измеренной прибавки без новых признаков, обучения и HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def main():
    import numpy as np,joblib
    from scipy.optimize import linear_sum_assignment
    source=ROOT/'runs/z13_val20_matched_20261024';r=read(source/'native_masa_recovery2/report.json');out=RUN/'masa_saved_explanation';out.mkdir(exist_ok=True)
    head=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib');names=['gap','dist','log_size_ratio','same_class','a_len','b_len','thumb_cos','hist_l1','hexin_cos','fly_cos','fly_T123','fly_T4','fly_T5','fly_Tm','fly_TmY','dino_cos','masa_cos']
    columns=[names[i] for i in head['cols']];assert columns==['gap','dist','log_size_ratio','same_class','a_len','b_len','thumb_cos','hist_l1','hexin_cos','masa_cos']
    per={};start=time.monotonic();totals=collections.Counter()
    for v in r['videos']:
        rawpath=source/'xywh/evaluated_raw'/(v+'.json');masapath=source/'xywh/evaluated_masa'/(v+'.json');gtpath=ROOT/'data/bdd100k/labels/box_track_20/val'/(v+'.json')
        raw=read(rawpath);masa=read(masapath);gt=read(gtpath);assert len(raw)==len(masa)==len(gt)
        mapping=collections.defaultdict(set);support=collections.defaultdict(collections.Counter);boxes=0;changed=0
        for a,b,g in zip(raw,masa,gt):
            assert {k:x for k,x in a.items() if k!='labels'}=={k:x for k,x in b.items() if k!='labels'}
            assert len(a['labels'])==len(b['labels']);assert len({str(l['id']) for l in b['labels']})==len(b['labels'])
            for x,y in zip(a['labels'],b['labels']):
                assert {k:z for k,z in x.items() if k!='id'}=={k:z for k,z in y.items() if k!='id'}
                old,new=str(x['id']),str(y['id']);mapping[old].add(new);boxes+=1;changed+=old!=new
            for cls in {l['category'] for l in a['labels']}:
                aa=[l for l in a['labels'] if l['category']==cls]
                bb=[l for l in g['labels'] if l['category']==cls and not (l.get('attributes') or {}).get('crowd',False)]
                if not aa or not bb:continue
                x=np.array([[l['box2d'][k] for k in ['x1','y1','x2','y2']] for l in aa]);y=np.array([[l['box2d'][k] for k in ['x1','y1','x2','y2']] for l in bb])
                lo=np.maximum(x[:,None,:2],y[None,:,:2]);hi=np.minimum(x[:,None,2:],y[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2)
                area=np.maximum(x[:,2:]-x[:,:2],0).prod(1);other=np.maximum(y[:,2:]-y[:,:2],0).prod(1);iou=inter/np.maximum(area[:,None]+other[None,:]-inter,1e-9)
                ii,jj=linear_sum_assignment(np.where(iou>=.5,1-iou,1e6))
                for i,j in zip(ii,jj):
                    if iou[i,j]>=.5:support[str(aa[i]['id'])][(cls,str(bb[j]['id']))]+=1
        assert all(len(x)==1 for x in mapping.values());groups=collections.defaultdict(list)
        for old,ns in mapping.items():groups[next(iter(ns))].append(old)
        counts=collections.Counter();examples={}
        for new,olds in groups.items():
            if len(olds)<2:continue
            labels=[];known=True
            for old in olds:
                c=support[old];n=sum(c.values())
                if n<2 or c.most_common(1)[0][1]/n<.9:known=False;break
                labels.append(c.most_common(1)[0][0])
            kind='unknown_or_mixed' if not known else ('consistent' if len(set(labels))==1 else 'conflicting')
            counts[kind]+=1
            if kind not in examples:examples[kind]=dict(output_id=new,raw_ids=olds,dominant_gt=labels)
        item=dict(frame_count=len(raw),box_count=boxes,changed_id_rows=changed,raw_ids=len(mapping),stitched_ids=len(groups),visible_merged_groups=sum(counts.values()),
            supported_groups=dict(counts),examples=examples,only_ids_changed=True,each_raw_id_has_one_output_id=True,no_duplicate_output_ids_per_frame=True,
            source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [rawpath,masapath,gtpath]})
        per[v]=item;totals.update({k:item[k] for k in ['frame_count','box_count','changed_id_rows','raw_ids','stitched_ids','visible_merged_groups']});totals.update(counts)
        write(out/'status.json',dict(state='running',video=v,complete_videos=len(per),updated=now()),required=False)
    a=r['pooled']['xywh_raw'];b=r['pooled']['xywh_masa'];metrics={k:dict(raw=a[k],masa=b[k],delta=b[k]-a[k]) for k in ['HOTA','DetA','AssA','LocA','IDF1','IDSw','CLR_FP','CLR_FN']}
    assert metrics['CLR_FP']['delta']==metrics['CLR_FN']['delta']==0 and metrics['AssA']['delta']>0
    report=dict(complete=True,source='runs/z13_val20_matched_20261024/native_masa_recovery2/report.json',source_sha256=sha(source/'native_masa_recovery2/report.json'),
        measured_metrics=metrics,paired=r['paired']['xywh_masa-xywh_raw'],totals=dict(totals),per_video=per,
        actual_method=dict(scorer='Frozen logistic regression + scaler; greedy offline stitching',features=columns,threshold=.6,max_gap_raw_frames=150,max_distance_box_sizes=5,
            standardized_coefficients=dict(zip(columns,head['model'][-1].coef_[0].tolist())),model_sha256=sha(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib')),
        measured_conclusion='На всех20 сохранённых выходах менялись толькоID;тежекадры/рамки/классы/баллы/число. Измеренная прибавка относится к объединению обрывков и улучшению ассоциации,а не новым обнаружениям или рамкам.',
        unresolved='Отдельный вклад MASA-признака не изолирован:голова использует геометрию,миниатюру,гистограмму,входглаза и MASA. Нужен отдельный контроль той же сшивки без MASA-признака;он сейчас не запускался. Нельзя приписывать весь прирост одной нейросети.',
        label_caveat='Согласованность групп—условная диагностика:однозначное попарное IoU>=.5 того же класса,неcrowd;длякаждогоисходногоID>=2совпадения и>=90%доминирующейGTличности. Непокрытые/смешанные группы неизвестны,неошибки. Это не точная глобальная дельта HOTA каждой связи,не тест обучаемости и не маски/GTвхода трекера.',
        ground_truth_only_in_diagnostic=True,new_training=False,new_gpu_inference=False,new_hota_evaluation=False,seconds=time.monotonic()-start,updated=now())
    write(out/'report.json',report);write(out/'status.json',dict(state='completed',updated=now()));print('Saved MASA mechanism audit:',dict(totals))

if __name__=='__main__':main()
