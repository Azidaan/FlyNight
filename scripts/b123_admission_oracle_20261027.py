"""Проверка пространства для отбора, не реальная прибавка и не максимум HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';PARENT=BASE/'supported_transfer/recovery2';RUN=BASE/'admission_oracle';IDENT='recovery_admission_oracle_v1'
from scripts.b123_supported_transfer_recovery1_20261027 import write
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    assert not RUN.exists();r=read(PARENT/'report.json');assert r['complete'] and read(PARENT/'independent_audit.json')['passed']
    names=['scripts/b123_admission_oracle_20261027.py','tracking/recovery_admission_oracle.py','tracking/recovery_detection_support.py','evaluation/hota_eval.py','scripts/z13_continue.py',
        'runs/b123_20261027/supported_transfer/recovery2/report.json']
    for v in r['videos']:names += [f'runs/b123_20261027/supported_transfer/recovery2/baseline/{v}.json',f'runs/b123_20261027/supported_transfer/recovery2/supported/{v}.json',f'data/bdd100k/labels/box_track_20/val/{v}.json']
    p=dict(id=IDENT,budget_seconds=350,author_stop='2026-10-27T07:20:00+03:00',device='cpu',videos=r['videos'],iou_threshold=.5,
        question='Есть ли полезный запас у уже подготовленных кандидатов, если доступ к выдаче проверяет разметка текущего объекта и прошлого наблюдения?',
        action='Те же3738сохранённых кандидатов новой20части,без новых рамок/ID/координат. Прошлую истинную личность кандидата устанавливает последнее совпадение базовогоIDсGTприIoU>=0.5. '
            'В текущем кадре допуск только к тому же GTобъекту приIoU>=0.5,еслион не покрыт текущей базой;не болееодногодобавлениянаGT. История обновляется только базой,после решения о допуске.',
        counterargument='Контроль использует недоступную модели разметку. Даже большой выигрыш не доказывает обучаемость такого выбора или пользу FlyVis. '
            'Это одна допустимая политика выбора,не оптимизация глобальнойHOTA и НЕ верхняя граница или максимальныйпотолок. Есливыигрышанет,отсутствиеполезныхдействийне доказано.',
        decision_value='Положительная нижняя граница oracle−baseline покажет существование полезного сочетания этихдобавлений и обоснует подготовку обучаемого отказа на отдельных train/dev. '
            'Иначе этот контроль не обосновывает дорогую тренировку селектора;потребуется другаягеометрия/память/набордействий. Ниодин исход не разрешаетподбиратьпорогинаэтихval.',
        grounding='Существующая постановка строгого action gate проекта использует разметку для диагностики доступныхдействий:runs/z13_box_supervision_followup_20261025/strict_action_gate.json. '
            'Здесь другой наборкандидатов/выборка,старыечислаи потолокнепереносятся.',
        acceptance=['Все20видео,точныебазовыеJSON,толькоподмножествоисходныхдобавленийбезизменениярамокилиID',
            'Толькоисторияисходнойбазывпрошлом;явныйGT-вход,неприписыватьприбавкуреальномумодулю',
            'ПолнаяHOTAи95%парныйинтервалдиагностическоговыбора;знакнижнейграницыопределяетрешениеподготовкиданных,неприёмкумодели'],
        limitation='Диагностика с истиннойразметкой на ужеиспользованной20частиval. Не реальныйтрекер,не научнаяприбавкаобученноймодели,неотдельныйвкладмухи,немаксимумHOTA. '
            'Истиннаяличностьпоследнегонаблюдениялокальная;правилоотборане оптимизируетдальнейшуюHOTA. Результатыстарогоbaseline/реальногоправилане пересчитываются,новыйTrackEvalтолькодлядиагностики.',
        sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Доступные продолжения — диагностический допуск с разметкой',area='flyvis',state='ready',started='2026-10-27',question=p['question'],
        hypothesis='В текущем наборе продолжений может быть полезное сочетание,которое реальное правило пока не отбирает;это проверка существования,не обучаемости.',expected=p['action'],
        summary=p['counterargument'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],files=names[:3],
        evidence=[dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/protocol.json',claim='Условия GT-контроля,границывыводаирешение до расчёта')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлен диагностический допуск с разметкой,не потолокHOTA',
        '*Постановка.* '+p['question']+' '+p['action']+'\n\n*Сильное возражение.* '+p['counterargument']+'\n\n*Что изменит решение.* '+p['decision_value']+'\n\n'
        '*Объём.* CPU350с,теже20видео/кандидаты,безGPU/новыхданных/обучения. Первыйполныйпроходдолжендопустить2*t*19+60востаток;иначеостановкабезпродления. '
        'Толькооднадиагностикапередследующимдействиемобучаемогоотказа. '+p['limitation']+' Источник:runs/b123_20261027/admission_oracle/protocol.json.')
    print('Prepared GT-only admission diagnostic; not real model improvement or HOTA maximum')
def run():
    from tracking.recovery_admission_oracle import PastIdentityAdmissionOracle
    from evaluation.hota_eval import evaluate_hota,VALID
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');start=time.monotonic();deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    parent=read(PARENT/'report.json');per={};timing={};counts={}
    def status(**kw):
        if time.time()>deadline-3 or (RUN/'stop.request').exists():raise TimeoutError('Original diagnostic budget/author stop')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw))
    for i,v in enumerate(p['videos']):
        began=time.monotonic();status(video=v,index=i,total=20);base=read(PARENT/'baseline'/(v+'.json'));proposals=read(PARENT/'supported'/(v+'.json'));truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json')
        assert len(base)==len(proposals)==len(truth);oracle=PastIdentityAdmissionOracle(p['iou_threshold']);rows=[]
        for a,b,g in zip(base,proposals,truth):
            assert b['labels'][:len(a['labels'])]==a['labels'];candidates=b['labels'][len(a['labels']):]
            accepted=oracle.step(a['labels'],candidates,[x for x in g['labels'] if x['category'] in VALID]);out=copy.deepcopy(a);out['labels']+=copy.deepcopy(accepted);assert len({x['id'] for x in out['labels']})==len(out['labels']);rows.append(out)
        write(RUN/'oracle'/(v+'.json'),rows);per[v]=evaluate_hota(RUN/'oracle',[v],split='val');write(RUN/'oracle/per_video.json',per);counts[v]=dict(oracle.counts);timing[v]=time.monotonic()-began
        if i==0:
            cost=2*timing[v]*19+60;remaining=deadline-time.time();write(RUN/'price.json',dict(first_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=remaining,proceed=cost<remaining));assert cost<remaining
    status(stage='Общая оценка GT-диагностики,не реальной модели');pooled=evaluate_hota(RUN/'oracle',p['videos'],split='val')
    contrasts={name:paired({v:x['HOTA'] for v,x in per.items()},{v:x['HOTA'] for v,x in control.items()}) for name,control in [('oracle_vs_baseline',parent['per_video']['baseline']),('oracle_vs_real_rule',parent['per_video']['supported'])]}
    write(RUN/'report.json',dict(complete=True,oracle_diagnostic_only=True,real_tracker_improvement=False,maximum_or_upper_bound=False,videos=p['videos'],
        pooled=dict(baseline=parent['pooled']['baseline'],real_rule=parent['pooled']['supported'],oracle=pooled),per_video=dict(baseline=parent['per_video']['baseline'],real_rule=parent['per_video']['supported'],oracle=per),paired=contrasts,counts=counts,seconds=time.monotonic()-start,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat(),stage='Диагностический допуск с разметкой завершён'))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
