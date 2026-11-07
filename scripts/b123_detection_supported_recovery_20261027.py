"""Новый причинный повтор сохранённых кандидатов с поддержкой текущим детектором."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/b123_20261027';RUN=BASE/'detection_supported_recovery';IDENT='detection_supported_recovery_v1'
MSK=timezone(timedelta(hours=3))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.next');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def prepare():
    assert not RUN.exists()
    sam=BASE/'sam2_query_output_pilot';assert read(sam/'independent_audit.json')['passed']
    for folder in [sam,BASE/'sam2_query_audit']:
        end=read(folder/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended']
    old=read(sam/'protocol.json')
    names=['scripts/b123_detection_supported_recovery_20261027.py','tracking/recovery_detection_support.py',
        'scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py','scripts/z13_continue.py',
        'runs/b123_20261027/sam2_query_output_pilot/report.json','runs/b123_20261027/recovery_motion_control/report.json',
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
    for v in old['videos']:
        names += [f'runs/b123_20261027/sam2_query_output_pilot/baseline/{v}.json',f'runs/b123_20261027/sam2_query_output_pilot/sam2_mask/{v}.json',
            f'runs/b123_20261027/recovery_motion_control/emit_lost_kalman/{v}.json',
            f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.json',f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz',
            f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json',f'data/bdd100k/labels/box_track_20/val/{v}.json']
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),device='cpu',budget_seconds=900,author_stop='2026-10-27T07:20:00+03:00',
        videos=old['videos'],arms=['baseline','kalman_support','mask_support'],iou_threshold=.5,confidence_low=old['configs']['xywh']['track_low_thresh'],
        action='Одно новое правило выдачи: рамка потерянного объекта сохраняется только при поддержке текущего реального обнаружения того же класса. '
            'Обнаружение должно иметь conf>штатного track_low_thresh=0.1 и IoU>=0.5 с кандидатом, но IoU<0.5 с каждой текущей рамкой базы. '
            'Взаимно однозначное максимальное сопоставление по IoU, без объединения с confidence. После поддержки сохраняется исходная рамка кандидата, не рамка детектора. '
            'Тот же фильтр для причинного масочного кандидата и прогноза Калмана, исходная выдача базы сохраняется побайтно в JSON.',
        sources_and_rationale='ByteTrack(ECCV2022),https://arxiv.org/abs/2110.06864: реальные слабые обнаружения могут восстанавливать объекты и отсеивать фон. '
            'Текущий первичный byte_tracker.py:_second_association использует порог стоимости0.5; новая версия фиксирует геометрический IoU0.5 без подбора. '
            'Собственное правило объединения SAM2 и детектора не является опубликованным алгоритмом ByteTrack или LTR. '
            'LTR:https://mucientes.pages.citius.gal/pubs/Bendana25_ibpria.pdf использует другой выбор маски и DeAOT, здесь они не воспроизводятся.',
        counterargument='Если YOLO полностью пропустил объект, такое правило его не восстановит. Поддержка чужим объектом и ложным обнаружением остаётся возможной. '
            'Внутренние назначения трекера неизвестны: свободно означает только геометрически не занято текущей выдачей. '
            'Сравнение маски с Калманом включает собственный отказ SAM2, а не изолирует только геометрию маски.',
        baseline='Те же сохранённые20видео, времена запроса и текущие детекции. Кандидаты уже рассчитаны причинно; их повторная обработка не меняет память модели или исходного трекера. '
            'Выход базового JSON совпадает с исходным rawXYWH, HOTA должна точно повториться. Разметка доступна только оценщику.',
        cost_gate='Первое полное видео и все3оценки:2*его стоимость*19+60 должны помещаться в оставшиеся900с идо07:20. Иначе сохранить остаток без продления.',
        acceptance=['Все20видео, точные исходные baselineJSON и HOTA, только добавленные уникальныеID',
            'Одинаковый фильтр для двух источников кандидатов; только текущие реальные детекции, без подбора порогов или разметки во входе',
            'HOTA/AssA/IDF1/FP/FN/IDSw и парные95%интервалы. Польза всей ветви:нижняя граница mask_support−baseline>0; вклад масочного пути:mask_support−kalman_support>0'],
        limitation='Новый исследовательский вариант на повторно использованных20val,не независимая проверка. Правило выбрано после отрицательного итога старой реализации; '
            'численные пороги не оптимизируются по исходам. Это повтор сохранённых кандидатов при фиксированном их причинном состоянии, не новая обратная связь в трекер. '
            'Нельзя называть точнойLTR,SGAR или вкладом ядра FlyVis. Нет нового инференса SAM2 или YOLO, данные/окружение не меняются.',
        sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Восстановление потерянного объекта — поддержка текущим обнаружением',area='flyvis',state='ready',started='2026-10-27',
        question='Даст ли поддержка текущим детектором пользу выдаче потерянного объекта и добавляет ли масочный путь пользу сверх прогноза Калмана?',
        hypothesis='Реальное текущее обнаружение может отвергнуть ложное продолжение; знак HOTA и преимущество маски над Калманом неизвестны.',
        expected=p['action'],summary=p['baseline'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],files=names[:2],
        evidence=[dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/protocol.json',claim='Действие, источники, возражение, одинаковые контроли и критерии до расчёта')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлена выдача с поддержкой текущим обнаружением',
        '*Постановка.* '+p['action']+'\n\n*Основание.* '+p['sources_and_rationale']+'\n\n*Сильное возражение.* '+p['counterargument']+'\n\n'
        '*Различающая проверка.* '+p['acceptance'][-1]+'\n\n*Бюджет и решение.* CPU900с,первое видео определяет цену,без новогоGPU. '
        'Польза потребует положительной нижней границы,иначе не принимать. '+p['limitation']+' Источник:runs/b123_20261027/detection_supported_recovery/protocol.json.')
    print('Prepared detector-supported replay; catalogue checks required before publication')
def run():
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,CLASS_NAMES
    from tracking.recovery_detection_support import supported_candidates
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');start=time.monotonic();deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    def status(stage,**kw):
        if time.time()>deadline-3 or (RUN/'stop.request').exists():raise TimeoutError('Original stage budget or author stop')
        write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    per={a:{} for a in p['arms']};counters={};timing={}
    for index,v in enumerate(p['videos']):
        began=time.monotonic();status('Причинный повтор с текущими обнаружениями',video=v,index=index,total=20)
        base=read(BASE/'sam2_query_output_pilot/baseline'/(v+'.json'));original=read(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.json');assert base==original
        proposals=dict(mask_support=read(BASE/'sam2_query_output_pilot/sam2_mask'/(v+'.json')),kalman_support=read(BASE/'recovery_motion_control/emit_lost_kalman'/(v+'.json')))
        matches=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')['matches'];assert len(matches)==len(base)
        dets=load_detection_cache(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz')
        rows={a:[] for a in p['arms']};counts={a:dict(candidates=0,accepted=0) for a in proposals}
        for i,(current,match) in enumerate(zip(base,matches)):
            xyxy,conf,classes=detections_for_frame(dets,int(match['video_frame_index']))
            detections=[dict(category=CLASS_NAMES[int(cls)],box2d=dict(zip(('x1','y1','x2','y2'),map(float,bb)))) for bb,score,cls in zip(xyxy,conf,classes) if score>p['confidence_low']]
            rows['baseline'].append(copy.deepcopy(current))
            for arm,source in proposals.items():
                prev=source[i];assert prev['name']==current['name'] and prev['labels'][:len(current['labels'])]==current['labels']
                candidates=prev['labels'][len(current['labels']):]
                accepted,stats=supported_candidates(current['labels'],candidates,detections,p['iou_threshold'])
                output=copy.deepcopy(current);output['labels']+=copy.deepcopy(accepted)
                assert len({x['id'] for x in output['labels']})==len(output['labels'])
                rows[arm].append(output)
                for key in counts[arm]:counts[arm][key]+=stats[key]
        for arm in p['arms']:
            write(RUN/arm/(v+'.json'),rows[arm]);status('Оценка выхода с поддержкой детектора',video=v,arm=arm)
            per[arm][v]=evaluate_hota(RUN/arm,[v],split='val');write(RUN/arm/'per_video.json',per[arm])
        control=read(BASE/'recovery_motion_control/report.json');assert per['baseline'][v]==control['per_video']['baseline'][v]
        counters[v]=counts;timing[v]=time.monotonic()-began;write(RUN/'completed_videos.json',dict(videos=list(timing),timing=timing,counters=counters))
        if index==0:
            cost=2*timing[v]*19+60;available=deadline-time.time();write(RUN/'price.json',dict(first_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    pooled={}
    for arm in p['arms']:status('Общая оценка20видео',arm=arm);pooled[arm]=evaluate_hota(RUN/arm,p['videos'],split='val')
    assert pooled['baseline']==control['pooled']['baseline']
    contrasts={name:paired({v:per[a][v]['HOTA'] for v in p['videos']},{v:per[b][v]['HOTA'] for v in p['videos']}) for name,a,b in [('mask_vs_baseline','mask_support','baseline'),('kalman_vs_baseline','kalman_support','baseline'),('mask_vs_kalman','mask_support','kalman_support')]}
    write(RUN/'report.json',dict(complete=True,videos=p['videos'],pooled=pooled,per_video=per,paired=contrasts,counters=counters,timing=timing,seconds=time.monotonic()-start,limitation=p['limitation'],tracker_state_changed=False,new_inference=False))
    write(RUN/'status.json',dict(state='completed',stage='Полный итог сохранён',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
