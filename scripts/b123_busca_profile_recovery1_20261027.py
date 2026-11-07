"""Ограниченный профиль авторской BUSCA на реальных областях; не HOTA."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/busca_runtime_profile/recovery1'
MSK=timezone(timedelta(hours=3))
IDENT='busca_runtime_profile_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.next')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def prepare():
    assert not RUN.exists()
    video=read(ROOT/'runs/z13_val20_matched_20261024/raw/protocol.json')['videos'][0]
    sources=['tracking/busca_runtime.py','scripts/b123_busca_runtime_profile_20261027.py',
        'weights/busca_official_20261027/model_busca.pth','tools/busca-deps-20261027/provenance.json',
        'tools/busca-deps-20261027/positional_encodings/torch_encodings.py',
        'runs/b123_20261027/recovery_source_snapshot/manifest.json',
        f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz',
        f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json']
    manifest=read(ROOT/sources[5])
    sources += [x['path'] for x in manifest['files'] if x['repository']=='lorenzovaquero/BUSCA']
    sources += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/f'data/bdd100k/images/track/val/{video}').glob('*.jpg'))[:60]]
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),author_stop='2026-10-27T07:20:00+03:00',budget_seconds=300,
        video=video,device='cuda',checkpoint='weights/busca_official_20261027/model_busca.pth',
        selection='Первое из фиксированных20val. В первых60готовых изображениях найти первыйID штатного трекера с11прошлыми наблюдениями. Первые11областей — память,пять ближайших областей последнего кадра — кандидаты. Разметку не читать. Это профиль,не оценка качества.',
        batches=[1,2,4],repeats=3,seq_len=11,num_candidates=5,
        service_changes='Полный checkpoint загружается строго,включая всеReIDвеса. Пропустить избыточные ImageNet/featureзагрузки только после сверки всехключей. Авторская таблица PE заменена хранением независимых осей с побайтным сравнениемFP16всех125позиций уменьшенной таблицы иполногоforward. Самиавторскиеисходникинеправятся. Прочиеусловияпрямогопроходасохранены.',
        acceptance=['Все379ключей checkpoint точно соответствуют архитектуре,никаких случайнооставленныхвесов',
                    'Экономное PE побайтно равно авторской реализации на полной малойобласти иполномпозиционномforward',
                    'Прямойпроход на реальныхобластях конечен,вероятностинормированы;стоимостьиGPUпамятьсохранены'],
        limitation='Техническаясовместимостьистоимость,не BUSCAнаполномтрекере,неHOTA,необучениеинеутверждениепользы.',
        sha256={name:sha(ROOT/name) for name in sorted(set(sources))})
    write(RUN/'protocol.json',p)
    catalog=ROOT/'experiments/project_history.json';before=catalog.read_bytes();data=json.loads(before)
    assert not any(c['id']==IDENT for c in data['experiments'])
    data['experiments'].append(dict(id=IDENT,title='Восстановление по истории объекта — совместимость и стоимость авторской BUSCA',area='flyvis',state='ready',started='2026-10-27',
        question='Можно ли исполнить авторскую BUSCA с полностью проверенными весами в нашем окружении и бюджете?',
        hypothesis='Полный checkpoint содержит извлекатель внешности; независимые оси PE позволяют сохранить вычисление без огромной таблицы.',
        expected=p['service_changes'],summary=p['limitation'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],files=sources[:2],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Условия,лимитикритериидорасчёта')]))
    assert before==catalog.read_bytes();write(catalog,data)
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT=IDENT;J.journal('поставлен ограниченный профиль авторской BUSCA',
        '*Постановка.* Проверитьстрогуюполнотуполученныхвесов,эквивалентноеэкономноеPEистоимостьреальногопрямогопрохода. '
        'Триразмерапакета,по3повтора;первыефиксированныереальныеобластиизval,безметок. '
        '300снаодномGPU,контроллеринезависимыйсрок07:20. '+p['limitation']+' '
        'Источник: `runs/b123_20261027/busca_runtime_profile/protocol.json`. '
        'ПолнаяавторскаятаблицаPEтребуетмиллиардыэлементов;заменяетсятолькоспособеёхранениясобязательнойточнойпроверкой. '
        'Зависимостьpositional-encodings6.0.3полученаизPyPIвtools/busca-deps-20261027,7519байт,проверенSHA256;основноеокружениенеменялось.')
    print('Prepared '+IDENT+'; validate before queue publication')
def run():
    start=time.monotonic();p=read(RUN/'protocol.json')
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    def status(stage,**kw):write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    status('Проверка эквивалентности позиционного кодирования')
    import torch,numpy as np,cv2
    torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    from tracking.busca_runtime import verify_lazy_encoding,load_model
    pe=verify_lazy_encoding();write(RUN/'positional_equivalence.json',pe)
    status('Загрузка всех авторских весов')
    model,coverage=load_model('cuda');write(RUN/'weight_coverage.json',coverage)
    from busca.tracking import get_bbox_crop
    video=p['video'];align=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json')['matches']
    images=sorted((ROOT/f'data/bdd100k/images/track/val/{video}').glob('*.jpg'))[:60]
    with np.load(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz') as z:raw=z['outputs']
    observed={}
    for index,(image,match) in enumerate(zip(images,align)):
        f=int(match['video_frame_index']);rows=raw[raw[:,0]==f]
        for row in rows:observed.setdefault(int(row[5]),[]).append((index,row))
    ident=next(ident for ident in sorted(observed) if len(observed[ident])>=11)
    history=observed[ident][:11];crops=[];boxes=[]
    for index,row in history:
        im=cv2.imread(str(images[index]));box=row[1:5]
        crops.append(get_bbox_crop(im,box,normalize=False));boxes.append(box)
    last_index,last_row=history[-1];f=int(align[last_index]['video_frame_index']);rows=raw[raw[:,0]==f]
    centers=(rows[:,1:3]+rows[:,3:5])/2;ref=(last_row[1:3]+last_row[3:5])/2
    order=np.argsort(np.sum((centers-ref)**2,axis=1))[:5]
    im=cv2.imread(str(images[last_index]));candidate_boxes=rows[order,1:5]
    candidates=[get_bbox_crop(im,box,normalize=False) for box in candidate_boxes]
    from busca.tracking import missing_candidate_bbox
    while len(candidates)<5:
        candidates.append(np.zeros((384,128,3),dtype=np.uint8))
        candidate_boxes=np.concatenate([candidate_boxes,missing_candidate_bbox(flavour="ltrb")[None]],axis=0)
    write(RUN/'real_inputs.json',dict(video=video,track_id=ident,memory_indices=[i for i,_ in history],
        candidate_frame=f,memory_boxes=np.array(boxes).tolist(),candidate_boxes=candidate_boxes.tolist(),labels_used=False))
    stats=[]
    for batch in p['batches']:
        status('Прямой проход на реальных областях',batch=batch)
        def tensor(value):
            arr=np.repeat(np.array(value)[None],batch,axis=0)
            arr=model._normalize_embeddings_batch(arr)
            return torch.from_numpy(arr[..., [2,1,0]]).permute(0,1,4,2,3).contiguous().to('cuda')
        mem=tensor(crops);can=tensor(candidates)
        mb=torch.tensor(np.repeat(np.array(boxes)[None],batch,axis=0),dtype=torch.float32,device='cuda')
        cb=torch.tensor(np.repeat(candidate_boxes[None],batch,axis=0),dtype=torch.float32,device='cuda')
        elapsed=[];torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for repeat in range(p['repeats']+1):
                torch.cuda.synchronize();t=time.monotonic()
                logits=model(mem,can,mb,cb);probs=model.softmax(logits)
                torch.cuda.synchronize();duration=time.monotonic()-t
                assert torch.isfinite(logits).all() and torch.isfinite(probs).all()
                assert float((probs.sum(-1)-1).abs().max())<1e-6
                if repeat:elapsed.append(duration)
        stats.append(dict(batch=batch,seconds=elapsed,seconds_per_track=float(np.mean(elapsed))/batch,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),output_shape=list(logits.shape),
            example_probs=probs[0].cpu().tolist()))
        write(RUN/'timings.json',stats)
    write(RUN/'report.json',dict(complete=True,weight_coverage=coverage,positional_equivalence=pe,timings=stats,
        seconds=time.monotonic()-start,hota_measured=False,full_tracker_executed=False,training_executed=False,
        next_action='Подключить решение BUSCA к неизменному XYWH с точным выключенным контролем; измерить полное первое видео и стоимость до запуска20.',limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',stage='Авторская сеть исполнилась;полныйтрекерещёнет',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
