"""Парный профиль FP32/BF16: цена и изменения масок, без HOTA."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_precision_profile';IDENT='sam2_mask_precision_profile_v1';MSK=timezone(timedelta(hours=3))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists()
    old=ROOT/'runs/b123_20261027/sam2_output_pilot';end=read(old/'controller_result.json');price=read(old/'price.json')
    assert end['all_owned_ended'] and not price['proceed']
    completed=read(old/'completed_videos.json');assert len(completed['videos'])==1
    original=read(ROOT/'runs/b123_20261027/sam2_runtime_profile/protocol.json')
    sources=list(original['sha256'])+['scripts/b123_sam2_precision_profile_20261027.py',
        'tools/sam2-runtime-20261027/README.md','runs/b123_20261027/sam2_output_pilot/price.json']
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),device='cuda',budget_seconds=300,author_stop='2026-10-27T07:20:00+03:00',
        video=original['video'],frames=30,arms=['FP32','BF16'],
        action='На одних30реальныхкадрах,одинаковыхвесах и назначениях сравнить только численную точность: FP32 против torch.autocast(cuda,bfloat16), как в официальных примерах SAM2. В одном процессе поочерёдно, состояние сбрасывается. Сохранить пиксельные маски и полные выдачи обоих вариантов.',
        reason='FP32остановлен своим заранее записанным ценовым критерием. Это новая реализация вычислений,не продление старого7200с и не повтор полного завершённого научного прогона. Парный повтор30кадров нужен для нового измерения изменений пиксельных масок, которых прежний профиль не сохранял.',
        acceptance=['Все выходы и маски конечны, входы причинны и идентичны, одинаковые ключи официальных весов',
                    'Сохранить изменения масок: IoU для каждой общей пары кадр/ID, отличающиеся составы и рамки; не предполагать эквивалентность BF16',
                    'Измерить оба времени и память. Если BF16 не быстрее, не запускать полный вариант ради стоимости. Допуск всей серии в любом случае по новому первому полному видео и прежней формуле'],
        limitation='Не HOTA, не доказательство равенства качества. Изменение точности может изменить маски и отказы; полный BF16пилот будет отдельным вариантом с новым протоколом.',
        sha256={name:sha(ROOT/name) for name in sorted(set(sources))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    previous=next(c for c in cat['experiments'] if c['id']=='sam2_mask_output_recovery_pilot_v1')
    fields=dict(summary='Полный FP32пилот остановлен по заранее заданной стоимости после первого видео; исход20неизвестен.',
        obtained='runs/b123_20261027/sam2_output_pilot/price.json:first_video_seconds=283.687,conservative_remaining_seconds=10900.106,available_seconds=6910.297,proceed=false. '
            'На первом видео baseline.HOTA=52.98064687819384 и sam2_mask.HOTA=51.87025698574441; источники baseline/per_video.json и sam2_mask/per_video.json в том же каталоге. '
            'completed_videos.json:только1видео,baseline_exact=true,all_image_requests_causal=true. Это не полный20видеорезультат.',
        interpretation='Стоимость FP32не позволила допустить оставшиеся19видео по зафиксированному правилу. Отдельное первое видео не устанавливает пользу или вред всей масочной ветви.',
        decision='Сохранить первый видеопроход и ценовой отказ. Не продлевать бюджет и не переносить знак на20видео. Проверить отдельно BF16как рекомендованный авторами режим вычислений,с явным контролем численных изменений.',
        cannot_claim=previous['cannot_claim']+' Нет HOTA20, парного интервала или отрицательного результата всей масочной ветви; остановка была по цене.')
    for field,value in fields.items():
        if previous.get(field) and previous[field]!=value:previous.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=field,old=previous[field],reason='Ценовой отказ после первого полного видео; прежняя постановка сохранена.'))
        previous[field]=value
    previous.update(state='inconclusive',done='2026-10-27')
    for item,met in zip(previous['acceptance'],[False,True,False]):item.update(met=met,evidence='runs/b123_20261027/sam2_output_pilot/completed_videos.json')
    previous['evidence'].append(dict(kind='file',ref='runs/b123_20261027/sam2_output_pilot/price.json',claim='Заранее заданная остановка по измеренной стоимости; не научный провал метода'))
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Восстановление по маске — стоимость и численные изменения при BF16',area='flyvis',state='ready',started='2026-10-27',
        question='Даст ли рекомендованная авторами пониженная точность достаточно дешёвый путь, и какие маски при этом изменятся?',
        hypothesis='BF16может ускорить вычисления, но равенство масок и HOTAне предполагается.',expected=p['action'],summary=p['reason'],
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],files=sources[-3:-2],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Одна смена точности,критерии и бюджет до измерения')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'масочный FP32не прошёл ценовой допуск; поставлена парная проверка BF16',
        '*Получено FP32.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n'
        '*Новая постановка.* '+p['action']+' '+p['reason']+'\n\n'
        '*Граница и решение.* '+p['limitation']+' Пиксельные совпадения не используются для настройки порога. '
        '300снаодномGPU,официальныйрежимвtools/sam2-runtime-20261027/README.md:90,111. '
        'Старые код1,ошибкаassert и все выходы сохраняются,конец дерева проверен. '
        'Источник:runs/b123_20261027/sam2_precision_profile/protocol.json. Процент HOTA20кбазе неизвестен.')
    print('Recorded FP32 cost stop and prepared paired precision profile')
def run():
    import torch,numpy as np,cv2
    torch.set_num_threads(2)
    from tracking.sam2_mask_recovery import load_predictor,MaskOutputRecovery
    p=read(RUN/'protocol.json')
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    video=p['video'];cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{video}.mov'));images=[]
    for _ in range(p['frames']):
        ok,image=cap.read();assert ok;images.append(image)
    cap.release()
    with np.load(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz') as z:raw=z['outputs']
    first=raw[raw[:,0]==0][0];ident=int(first[5]);other=next(row for row in raw[raw[:,0]==3] if int(row[5])!=ident)
    tracks=[SimpleNamespace(track_id=int(row[5]),score=float(row[6]),cls=float(row[7])) for row in [first,other]]
    arms={};all_masks={};outputs={}
    for arm in p['arms']:
        write(RUN/'status.json',dict(state='running',stage='Парный профиль точности',arm=arm,updated=datetime.now(MSK).isoformat()))
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        model=load_predictor('cuda');branch=MaskOutputRecovery(model,p['frames']);masks={};frames_out={}
        original=model.propagate_in_video
        def capture(state,*a,**kw):
            for frame,ids,logits in original(state,*a,**kw):
                assert torch.isfinite(logits).all()
                for k,obj in enumerate(ids):masks[(frame,int(obj))]=(logits[k,0]>0).cpu().numpy().copy()
                yield frame,ids,logits
        model.propagate_in_video=capture
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=arm=='BF16'):
            for frame,image in enumerate(images):
                source=raw[raw[:,0]==frame];base=np.column_stack([source[:,1:],np.zeros(len(source),np.float32)]).astype(np.float32)
                lost=[]
                for track,start_frame,end_frame in [(tracks[0],1,9),(tracks[1],4,15)]:
                    if start_frame<=frame<=end_frame:
                        base=base[base[:,4]!=track.track_id];lost.append(track)
                out=branch.step(frame,image,base,lost);assert np.isfinite(out).all();frames_out[str(frame)]=out.tolist()
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        all_masks[arm]=masks;outputs[arm]=frames_out
        np.savez_compressed(RUN/(arm+'_masks.npz'),**{f'{f}_{i}':np.packbits(mask) for (f,i),mask in masks.items()})
        arms[arm]=dict(seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            initialized=branch.initialized,propagated=branch.propagated,rejected=branch.rejected,
            all_image_requests_causal=all(requested<=seen for seen,requested in branch.frames.requests),mask_count=len(masks))
        write(RUN/(arm+'_outputs.json'),frames_out)
        del branch,model;torch.cuda.empty_cache()
    a,b=all_masks['FP32'],all_masks['BF16'];common=sorted(set(a)&set(b));ious=[]
    for key in common:
        union=np.logical_or(a[key],b[key]).sum();intersection=np.logical_and(a[key],b[key]).sum()
        ious.append(dict(frame=key[0],id=key[1],iou=float(intersection/union) if union else 1.))
    result=dict(complete=True,arms=arms,speedup=arms['FP32']['seconds']/arms['BF16']['seconds'],
        pixel_mask_comparison=ious,fp32_only_keys=[list(k) for k in sorted(set(a)-set(b))],bf16_only_keys=[list(k) for k in sorted(set(b)-set(a))],
        output_rows_equal=outputs['FP32']==outputs['BF16'],
        all_outputs_finite=True,all_image_requests_causal=all(x['all_image_requests_causal'] for x in arms.values()),
        bf16_faster=arms['BF16']['seconds']<arms['FP32']['seconds'],hota_measured=False,
        source='Официальный README SAM2 использует autocast(cuda,bfloat16); точность масок проверена отдельно,эквивалентностьнепредполагается.',
        limitation=p['limitation'])
    write(RUN/'report.json',result);write(RUN/'status.json',dict(state='completed',stage='Цена и различия точности сохранены',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
