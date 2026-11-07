"""Новый алгоритмический вариант: масочная выдача по запросу потребителя."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_query_output_pilot';IDENT='sam2_query_mask_output_pilot_v1'
ORIGINAL=ROOT/'scripts/b123_sam2_output_pilot_20261027.py'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepare():
    assert not RUN.exists()
    profile=ROOT/'runs/b123_20261027/sam2_precision_profile';r=read(profile/'report.json');end=read(profile/'controller_result.json')
    assert r['complete'] and end['exit_code']==0 and end['all_owned_ended']
    old=read(ROOT/'runs/b123_20261027/sam2_output_pilot/protocol.json')
    p=dict(old);p.update(id=IDENT,created=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        original_worker_sha256=sha(ORIGINAL),budget_seconds=7200,
        action=old['action']+' Единственная алгоритмическая замена — нейронный перенос маски только при запросе выдачи. Все исходные кадры по-прежнему причинно наблюдаются, реальные назначения и изображения сохраняются.',
        adaptation=old['adaptation']+' Численная точность остаётся FP32. В этой версии шаги масочного состояния пропускаются между запросами выдачи: история сети меняется, эквивалентность плотному переносу не предполагается.',
        output_schedule='Потребитель запрашивает выход на тех же заранее фиксированных кадрах оценки. Используются только их времена/индексы из старой alignment, не рамки, ID, качество или исход ошибки. Это режим выдачи по запросу, а не доказательство качества непрерывного30Гцмасочного выхода. Тот же запрос может быть подан в любой текущий момент; будущие изображения запрещены.',
        reason='Плотный FP32не прошёл цену. BF16ускорил короткий парный профиль только1.0905раза и меняет маски; этого по оценке недостаточно для исходного7200с. Поэтому не повторять дорогое первое полное видео с заведомо слабым выигрышем. Изменить частоту обращения к модулю, не численную точность или пороги.',
        limitation=old['limitation']+' Проверяется только причинная выдача на фиксированных запросах потребителя. Времена запросов доступны модулю, разметка объектов нет. Нельзя приписывать итог плотной FP32или BF16реализации.')
    p['sha256']=dict(old['sha256'])
    for name in ['scripts/b123_sam2_query_output_pilot_20261027.py','tracking/sam2_query_recovery.py',
                 'runs/b123_20261027/sam2_precision_profile/report.json']:
        p['sha256'][name]=sha(ROOT/name)
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    card=next(c for c in cat['experiments'] if c['id']=='sam2_mask_precision_profile_v1');assert card['state']=='ready'
    card.update(state='infrastructure',done='2026-10-27',
        obtained='runs/b123_20261027/sam2_precision_profile/report.json:FP32.seconds=3.578,BF16.seconds=3.281,speedup=1.0905211825658088;21общаямаска,IoU0.975798..0.998087,output_rows_equal=false. Все выходы конечны, входы причинны.',
        interpretation='Пониженная точность немного ускорила этот короткий профиль, но изменила пиксели и рамки. Значительного выигрыша стоимости и равенства HOTA не установлено.',
        decision='Не тратить полное видео на плотныйBF16при недостаточном ожидаемом выигрыше. Перейти к отдельно зарегистрированному алгоритмическому варианту масочной выдачи по запросу.')
    for item in card['acceptance']:item.update(met=True,evidence='runs/b123_20261027/sam2_precision_profile/report.json')
    card['evidence'].append(dict(kind='file',ref='runs/b123_20261027/sam2_precision_profile/report.json',claim='Парная стоимость и пиксельные изменения,неHOTA'))
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Восстановление потерянного объекта — маска по запросу выдачи',area='flyvis',state='ready',started='2026-10-27',
        question='Даст ли обновление масочного состояния только по запросам потребителя полезную HOTA при доступной стоимости?',
        hypothesis='Пропуск нейронных шагов между запросами уменьшит стоимость, но может изменить качество памяти; знак HOTA неизвестен.',
        expected=p['action'],summary=p['output_schedule'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],
        files=['scripts/b123_sam2_query_output_pilot_20261027.py','tracking/sam2_query_recovery.py'],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Одна замена каденса,времена запросов и ограничения зафиксированы до прогона')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'BF16дал малое ускорение; поставлена масочная выдача по запросу',
        '*Парный профиль.* '+card['obtained']+' '+card['interpretation']+'\n\n'
        '*Следующее действие.* '+p['action']+'\n\n*Времена запросов.* '+p['output_schedule']+'\n\n'
        '*Почему так.* '+p['reason']+' Отбор реализации сделан по цене, качество первого видео не использовалось для настройки. '
        'ПлотныйFP32непродолжается,готовыйрезультатсохранён. ПлотныйBF16не запускался на полном видео. '
        'НоваясобственнаяреализациянеявляетсявоспроизведениемLTR. '
        'ОдинGPU,новый7200сбюджетнепозже07:20;первое полное видео снова решает2*time*19+120. '
        'Источник:runs/b123_20261027/sam2_query_output_pilot/protocol.json. HOTAнеизвестна.')
    print('Prepared query-only mask pilot; no dense BF16 full video repeated')
def run():
    p=read(RUN/'protocol.json');assert sha(ORIGINAL)==p['original_worker_sha256']
    source=ORIGINAL.read_text(encoding='utf-8')
    old='from tracking.sam2_mask_recovery import load_predictor,MaskOutputRecovery'
    assert source.count(old)==1
    source=source.replace(old,'from tracking.sam2_mask_recovery import load_predictor\n    from tracking.sam2_query_recovery import QueryMaskOutputRecovery as MaskOutputRecovery')
    old='extra=branch.step(frame,image,base,lost)';assert source.count(old)==1
    source=source.replace(old,'extra=branch.step(frame,image,base,lost,request_output=(frame in anchors))')
    write(RUN/'source_transform.json',dict(original_source_sha256=p['original_worker_sha256'],
        transformed_source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
        changes=['Replace recovery class with query-aware class','Request neural output only at fixed consumer timestamps'],
        precision_changed=False,original_file_changed=False))
    namespace={'__file__':str(ORIGINAL),'__name__':'frozen_worker_import'}
    exec(compile(source,str(ORIGINAL),'exec'),namespace);namespace['RUN']=RUN;namespace['run']()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
