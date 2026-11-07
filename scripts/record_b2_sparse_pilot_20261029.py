"""Обучение Б2 и остановка по цене ещё не научная проверка HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_sparse_training_record';source=RUN/'b2_sparse_training_pilot'
    c=service.read(source/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=service.read(source/'report.json')
    h=service.read(ROOT/'experiments/project_history.json')
    card=next(x for x in h['experiments'] if x['id']=='b2_sparse_training_pilot_v1')
    card.update(state='infrastructure' if r['complete'] else 'inconclusive',done='2026-10-29',
        obtained='runs/continuation_20261028/b2_sparse_training_pilot/report.json: complete='+str(r['complete'])+
                 ', фактические optimizer_steps каждого из6вариантов,параметры,цена и источники завершённых видео в report.json/completed; '
                 'разреженная выборка по10целевых запросов на каждом из104историческихtrain/dev,одна эпоха seed26.',
        interpretation='Обучающий пилот авторской системы; полная научная проверка Б2 не выполнена. '
                       'Детекторная точность наtrain не оценивалась; GTтолькоцели последнего кадра. '+
                       ('Все6ветвей обучены в заданном коротком режиме.' if r['complete'] else 'Сохранён неполный режим и его причина; это не отрицательный научный результат.'),
        decision='После независимого аудита обученных весов нужен полный сопоставимый детектор и трекер на историческихval, '
                 'native/zero/last/mean/C/D/R1/B,полныеHOTA/AssA/IDF1/FP/FN/IDSwи парныеинтервалы;допускпоотдельнойфактическойцене. '
                 'При новом прямом выборе автора продолжение матрицы истории имеет согласованный приоритет.',
        cannot_claim='Однаэпоха/одинseed/10запросовнаtrainвидео — ограниченный пилот,неполное обучение на20441клипе. '
                     'Нет HOTA/точностиval/подтвержденияядра734параметра. Фиксированная P3позиция не совмещает движение объектов; '
                     'last/mean не чистый контроль порядка. Номинальные параметры C/D/R1/Bразличаютсяи явно опубликованы.')
    for a in card['acceptance']:a.update(met=True if r['complete'] else None,evidence='runs/continuation_20261028/b2_sparse_training_pilot/report.json')
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Завершён или сохранён по цене ограниченный обучающий пилот Б2',card['obtained']+' '+card['interpretation']+' '+card['cannot_claim'])
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_b2_causal_detector.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0
    service.write(folder/'report.json',dict(passed=True,source_sha256=service.sha(source/'report.json'),scientific_verdict=None,updated=service.now()))


if __name__=='__main__':main()
