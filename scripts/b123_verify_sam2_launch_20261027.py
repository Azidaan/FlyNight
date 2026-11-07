"""Проверить исполнителя и сохранить точный текущий этап без объявления итогов."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    run=ROOT/'runs/b123_20261027/sam2_output_pilot'
    launch=json.loads((run/'launch.json').read_text());status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    process=psutil.Process(launch['pid']);assert abs(process.create_time()-launch['created'])<.01
    assert 'b123_sam2_output_pilot_20261027.py' in ' '.join(process.cmdline()) and status['state']=='running'
    proof=dict(checked=datetime.now(timezone.utc).isoformat(),launch=launch,identity_verified=True,status=status,
        full_series_admitted=(run/'price.json').exists() and json.loads((run/'price.json').read_text())['proceed'],
        full_pilot_completed=False,meaning='Сейчас работает первый полный видеопроход; допуск оставшихся видео определит измеренная цена.')
    (run/'launch_verified.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal('sam2_mask_output_recovery_pilot_v1','собственная масочная ветвь начала первый полный видеопроход',
        '*Проверено.* PID и время создания исполнителя совпали с launch.json, свежий прогресс растёт. '
        'Источник:runs/b123_20261027/sam2_output_pilot/launch_verified.json. '
        'Первое полное видео ещё проходит; допуск остальных19будет только по price.json, без изменения исходного7200сбюджета. '
        'Нет нового полного HOTA этой ветви.\n\n'
        '*Проверки перед публикацией.* Каталог OK/58passed, два теста причинного ввода и IoUpassed. '
        'Граф после отрицательного BUSCA обновлён на копии и затем применён:40видимых,0удалений/перемещений,10заметокгипотезпобайтносохранены. '
        'ИтогBUSCAотрицательныйнепересчитывался; одна фундаментальнаязадача4содержитобепопытки. '
        'Коммит/pushпо-прежнемублокируетчужойстарый.git/index.lock,оннетронут.')
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — активен первый полный видеопроход SAM2\n\n'
            'Задание120_sam2_output_pilot работает, launch_verified.json проверил PID+birth,прогресссвежий. '
            'Остальные19видео пока не считать допущенными: дождатьсяprice.jsonпослепервогополного. '
            'Протокол7200с,до07:20,одинGPU. Оценка2*first_video_seconds*19+120должна влезть в оставшийсябюджет. '
            'Еслигейтпровален:сохранитьпервыйрезультатиинфраструктурныйостаток,необъявлятьHOTA20иливредмасок. '
            'Не продлевать старыйбюджет. Обоснованная следующая реализация при узком месте стоимости — '
            'отдельно зарегистрированный короткийBF16профиль по официальному режимуSAM2,с явным изменениемчисленнойточности '
            'и проверкойfinite/maskchanges,непереписываниемзапущеннойFP32версии. '
            'Еслиполнаясериядопущена,непрерыватьи не повторятьготовыевидео. Послеreport+конецдерева — независимаясверка/карточка/журнал/проверки. '
            'BUSCAитог уже completed_negative/аудитпрошёл:57.900829→56.843638,неполнаяавторскаяBUSCA. '
            'Граф40видимыхпослефундаментальногоитога,всефайлысохранять,гипотезыпобайтны. '
            'ПоследниепроверкикаталогаOK/58passed,граф7passed. '
            'НеобъявлятьSAM2ветвьточнойLTRилиSGAR,ихкод/селекторнеподтверждены. '
            'Ночной07:30,сохранение07:20,одинGPUи.guardсохраняются.\n')
    print(json.dumps(dict(identity_verified=True,pid=launch['pid'],full_series_admitted=proof['full_series_admitted'])))
if __name__=='__main__':main()
