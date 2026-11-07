"""Техническая проверка последовательного декодирования внутри старого окна обучения."""
from pathlib import Path
from datetime import datetime
import sys,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    diagnostic=read(RUN/'pixel_box_decode_diagnostic.json');assert diagnostic['stored_matches_sequential'] and not diagnostic['stored_matches_seek']
    for name in ['pixel_box_audit','pixel_box_learning','pixel_box_schedule']:
        old=read(RUN/name/'controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not any(matching(k,v) for k,v in old['owned'].items())
    # Старый аудит085 уже истёк и НЕ возобновляется. Исправленная проверка становится
    # обязательной CPU-подготовкой в ещё действующем абсолютном окне090.
    parent=read(RUN/'pixel_box_learning/execution_deadline.json')['deadline'];end=datetime.fromisoformat(parent).timestamp()
    remaining=math.floor(end-time.time());audit_budget=600;schedule_budget=120;train_budget=remaining-audit_budget-schedule_budget-60
    assert train_budget>900
    old_source=ROOT/'scripts/audit_pixel_box_dataset_20261027.py';new_source=ROOT/'scripts/audit_pixel_box_dataset_recovery1_20261027.py';assert not new_source.exists()
    source=old_source.read_text(encoding='utf-8')
    source=source.replace("RUN=BASE/'pixel_box_audit'","RUN=BASE/'pixel_box_audit/recovery1'")
    before="""            for i in sorted({0,len(records)-1}):
                label,previous=by[i];r=records[i]
                for side,fr,lab in [(1,r['frame'],label),(0,r['frame'] if previous is None else previous[0],label if previous is None else previous[1])]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,image=cap.read();assert ok
                    expected=region(image,coordinates(lab),p['size'],p['context']);assert np.array_equal(a[i,side],expected),(v,i,side)
                    pixel_checks+=1
"""
    after="""            wanted={}
            for i in sorted({0,len(records)-1}):
                label,previous=by[i];r=records[i]
                for side,fr,lab in [(1,r['frame'],label),(0,r['frame'] if previous is None else previous[0],label if previous is None else previous[1])]:
                    wanted.setdefault(fr,[]).append((i,side,lab))
            for fr in range(max(wanted)+1):
                ok,image=cap.read();assert ok
                for i,side,lab in wanted.get(fr,[]):
                    expected=region(image,coordinates(lab),p['size'],p['context']);assert np.array_equal(a[i,side],expected),(v,i,side)
                    pixel_checks+=1
"""
    assert source.count(before)==1;source=source.replace(before,after)
    source=source.replace("write(data/'audit.json',dict(passed=True,","write(data/'audit.json',dict(passed=True,sequential_decode=True,original_failed_audit_preserved=True,")
    new_source.write_text(source,encoding='utf-8')
    old_schedule=ROOT/'scripts/pixel_box_evaluation_schedule_20261027.py';new_schedule=ROOT/'scripts/pixel_box_evaluation_schedule_recovery1_20261027.py';assert not new_schedule.exists()
    source=old_schedule.read_text(encoding='utf-8')
    assert source.count("RUN/'pixel_box_learning/controller_result.json'")==1
    source=source.replace("RUN/'pixel_box_learning/controller_result.json'","RUN/'pixel_box_learning/recovery1/controller_result.json'")
    assert source.count("depends='pixel_box_learning'")==1
    source=source.replace("depends='pixel_box_learning'","depends='pixel_box_learning/recovery1'")
    new_schedule.write_text(source,encoding='utf-8')
    specifications=[('096_pixel_box_decode_audit_recovery1','pixel_box_audit/recovery1',new_source.relative_to(ROOT).as_posix(),audit_budget,{}),
        ('097_pixel_box_learning_recovery1','pixel_box_learning/recovery1','scripts/train_pixel_box_refinement_recovery1_20261027.py',train_budget,read(RUN/'pixel_box_learning/protocol.json')),
        ('098_pixel_box_schedule_recovery1','pixel_box_schedule/recovery1',new_schedule.relative_to(ROOT).as_posix(),schedule_budget,{})]
    jobs=[]
    for name,folder,script,budget,original in specifications:
        assert not (RUN/folder).exists();p=dict(original);p.update(parent_experiment='pixel_box_refinement_v1',created=now(),budget_seconds=budget,
            parent_absolute_deadline=parent,author_stop=read(RUN/'authorization.json')['save_at'],
            correction='Неизменные сохранённые данные проверяются последовательным чтением. Старые085/090/095ошибки сохранены. Дополнительного GPU-обучения прежде не было. Все3технические ступени помещаются в прежнее абсолютное окно090;085непродлевается.',
            sha256={s:sha(ROOT/s) for s in [script,'tracking/pixel_box_refinement.py','scripts/train_pixel_box_refinement_20261027.py','runs/recovery_research_20261027/pixel_box_dataset/protocol.json','runs/recovery_research_20261027/pixel_box_decode_diagnostic.json']})
        write(RUN/folder/'protocol.json',p)
        jobs.append((name,dict(name=name,folder=(RUN/folder).relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/folder/'protocol.json'))))
    write(RUN/'pixel_box_decode_recovery1.json',dict(created=now(),parent_absolute_deadline=parent,remaining_at_preparation=remaining,audit_budget=audit_budget,training_budget=train_budget,schedule_budget=schedule_budget,
        no_old_budget_extended=True,old_audit_closed=True,old_sources_and_failures_preserved=True,source_sha256=sha(__file__)))
    journal('pixel_box_refinement_v1','выявлен неверный произвольный переход декодера; восстановление в прежнем окне обучения',
        '*Сбой.* 085остановиласьна3видео при сравнении пикселей;090и095неполучилидопуск и завершилисьдообучения. Старые источники,stderr,controller_result(code1/all_owned_ended) сохранены. '
        '*Проверено.* pixel_box_decode_diagnostic.json:сохранённаяобластьпобайтносовпадаетспоследовательнымчтением,нонеспрямымпереходомCAP_PROP_POS_FRAMES;накадре251максимальноерасхождение110. '
        '*Исправление.* Аудиторчитаетпоследовательно,какнеизменныеданные/вывод. GT/метки/пиксели/режиммодели/критериинеменяются. Старый085(600с)истёкиНЕпродлевается. '
        'Проверкаиобучениевосстанавливаютсявещёдействующемабсолютномокне090до'+parent+',сразделениемоставшегосявремени:CPUаудит600с,обучение'+str(train_budget)+'с,подготовка120с,резерв60с. '
        'Источник pixel_box_decode_recovery1.json;это уменьшениеоставшегосябюджетаобучения,не новый7200с. ЗаключениеHOTAотсутствует;сбойаудитанеотрицательныйнаучныйрезультат. '
        'Задания096/097/098втойжеочереди;готовые080данныенеповторять.')
    for name,job in jobs:write(RUN/'ready_jobs'/(name+'.json'),job)
    print('Published bounded technical recovery inside old090 absolute deadline: '+parent)

if __name__=='__main__':main()
