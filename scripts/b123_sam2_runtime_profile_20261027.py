"""Реальный ограниченный профиль собственной масочной ветви; не HOTA."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_runtime_profile';IDENT='sam2_mask_runtime_profile_v1';MSK=timezone(timedelta(hours=3))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists()
    weights=read(ROOT/'runs/b123_20261027/sam2_weights_download/report.json');assert weights['complete']
    end=read(ROOT/'runs/b123_20261027/sam2_weights_download/controller_result.json');assert end['all_owned_ended'] and end['exit_code']==0
    video=read(ROOT/'runs/z13_val20_matched_20261024/raw/protocol.json')['videos'][0]
    sources=['scripts/b123_sam2_runtime_profile_20261027.py','tracking/sam2_mask_recovery.py',
        weights['target'],'tools/sam2-deps-20261027/provenance.json','tools/sam2-runtime-20261027/provenance.json',
        f'data/bdd100k/videos/val/{video}.mov',f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz']
    sources += [x['path'] for x in read(ROOT/'tools/sam2-runtime-20261027/provenance.json')['files']]
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),budget_seconds=300,author_stop='2026-10-27T07:20:00+03:00',
        device='cuda',video=video,frames=30,checkpoint=weights['target'],checkpoint_sha256=weights['sha256'],
        action='Исполнить масочный перенос по30реальнымкадрам на одной траектории с условно отключёнными текущими назначениями; добавить вторую траекторию после начала, проверить удаление вернувшегося объекта. Модель получает только прошлую реальную рамку и уже декодированные кадры. Это стоимость и проверка реализации, не точность восстановления.',
        adaptation='Собственная SAM2.1tiny вместо точной LTR. Без27масок, обученного селектора и DeAOT. Штатный object_score>0 и непустая маска; при IoU>0.9 с текущим выходом отказ. Без C++заполнения дыр и компиляции. FP32, память состояния наCPU; нормировка/размер1024как в официальном коде. Источники авторов неизменны, заменён только причинный ввод кадров.',
        acceptance=['Все ключи официального checkpoint загружены строго, конечные выходы',
                    'Ни одного обращения к будущему; добавление объекта после начала и снятие вернувшегося объекта работают',
                    'Реальная стоимость30кадров и пикGPUпамяти сохранены до решения о полном пилоте'],
        limitation='Не научная проверка LTR/SGAR-MOT и не HOTA. Искусственное отключение назначений здесь только для гарантированного исполнения ветви, не оценка качества.',
        sha256={name:sha(ROOT/name) for name in sorted(set(sources))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';raw=path.read_bytes();data=json.loads(raw)
    assert not any(x['id']==IDENT for x in data['experiments'])
    data['experiments'].append(dict(id=IDENT,title='Восстановление по маске — причинный ввод и стоимость SAM2',area='flyvis',state='ready',started='2026-10-27',
        question='Можно ли причинно исполнять компактную масочную ветвь в нашем бюджете без чтения будущих изображений?',
        hypothesis='Потоковый ввод с ограниченным прошлым позволит выполнять масочное продолжение в текущем окружении.',
        expected=p['action'],summary=p['adaptation'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],files=sources[:2],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Условия,границавывода,стоимостьикритериидорасчёта')]))
    assert path.read_bytes()==raw;write(path,data)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлен короткий профиль собственной масочной ветви',
        '*Постановка.* '+p['action']+'\n\n*Отличия.* '+p['adaptation']+'\n\n'
        '*Условия.* ОдинGPU,300с,30реальныхкадров,безразметки. '+p['limitation']+' '
        'Источник:runs/b123_20261027/sam2_runtime_profile/protocol.json. '
        'Официальные веса156008466байт получены за исходные300с, SHA7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69. '
        'Hydra/OmegaConf/ANTLR изолированы вtools,основное окружение не менялось. Первоначальныйpipзавершился отсутствиемbdist_wheel; '
        'вместо изменения окружения пакеты изPyPIраспакованы с проверкойSHA,setupнеисполнялся.')
    print('Prepared SAM2 causal runtime profile; validate before publication')
def run():
    p=read(RUN/'protocol.json');start=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    def status(stage,**kw):write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    import torch,numpy as np,cv2
    torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    from tracking.sam2_mask_recovery import load_predictor,MaskOutputRecovery
    status('Строгая загрузка SAM2.1tiny');predictor=load_predictor('cuda')
    state=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=True)['model']
    assert set(state)==set(predictor.state_dict())
    coverage=dict(strict=True,state_keys=len(state),checkpoint_sha256=p['checkpoint_sha256'])
    branch=MaskOutputRecovery(predictor,p['frames']);torch.cuda.reset_peak_memory_stats()
    with np.load(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{p["video"]}.npz') as z:raw=z['outputs']
    first=raw[raw[:,0]==0];assert len(first)>=1;ident=int(first[0,5]);track=SimpleNamespace(track_id=ident,score=float(first[0,6]),cls=float(first[0,7]))
    cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{p["video"]}.mov'));assert cap.isOpened()
    times=[];counts=[];second=None;removed=False
    with torch.inference_mode():
        for frame in range(p['frames']):
            status('Причинный перенос масок',done=frame,total=p['frames'])
            ok,image=cap.read();assert ok
            source=raw[raw[:,0]==frame];base=np.column_stack([source[:,1:],np.zeros(len(source),np.float32)]).astype(np.float32)
            lost=[]
            if 1<=frame<=9:
                base=base[base[:,4]!=ident];lost.append(track)
            if frame==4:
                candidates=[r for r in raw[raw[:,0]==3] if int(r[5])!=ident]
                if candidates:
                    row=candidates[0];second=SimpleNamespace(track_id=int(row[5]),score=float(row[6]),cls=float(row[7]))
            if second is not None and 4<=frame<=15:
                base=base[base[:,4]!=second.track_id];lost.append(second)
            torch.cuda.synchronize();t=time.monotonic();extra=branch.step(frame,image,base,lost);torch.cuda.synchronize()
            times.append(time.monotonic()-t);counts.append(len(extra))
            if frame==10:removed=ident not in branch.state['obj_ids']
    cap.release()
    assert all(requested<=seen for seen,requested in branch.frames.requests) and removed
    assert second is not None,'Need a real second track for late-addition check'
    report=dict(complete=True,coverage=coverage,frames=p['frames'],frame_seconds=times,mean_frame_seconds=float(np.mean(times)),
        max_frame_seconds=float(np.max(times)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        added_masks=counts,initialized=branch.initialized,propagated=branch.propagated,rejected=branch.rejected,
        all_image_requests_causal=True,late_second_object_added=True,returned_object_removed=removed,
        seconds=time.monotonic()-start,hota_measured=False,ltr_reproduced=False,sgar_reproduced=False,limitation=p['limitation'])
    assert sum(counts)>0,'No nonempty mask observed in technical profile'
    write(RUN/'report.json',report);write(RUN/'status.json',dict(state='completed',stage='Причинная масочная ветвь исполнилась;HOTAещёнет',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
