"""Записать уже завершённый масочный пилот; модели и оценщик не запускаются."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_query_output_pilot';SOURCE=RUN.relative_to(ROOT).as_posix()+'/report.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    r=read(RUN/'report.json');audit=read(RUN/'independent_audit.json');p=read(RUN/'protocol.json')
    assert r['complete'] and audit['passed'] and audit['full_pilot_completed']
    for folder in [RUN,ROOT/'runs/b123_20261027/sam2_query_audit']:
        end=read(folder/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended'] and not end['timed_out']
        for pid,birth in end['owned'].items():
            if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    b=r['pooled']['baseline'];n=r['pooled']['sam2_mask'];assert r['paired']['vs_baseline']['ci95'][1]<0
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    c=next(c for c in cat['experiments'] if c['id']=='sam2_query_mask_output_pilot_v1');assert c['state']=='ready'
    fields=dict(summary='Масочная выдача по запросу восстановила часть пропусков, но ухудшила HOTA исходного трекера. В стенд не принимается.',
        obtained=f'{SOURCE}: pooled.baseline.HOTA={b["HOTA"]}; pooled.sam2_mask.HOTA={n["HOTA"]}; paired={r["paired"]}. '
            f'CLR_FP {b["CLR_FP"]}→{n["CLR_FP"]}, CLR_FN {b["CLR_FN"]}→{n["CLR_FN"]}, IDSw {b["IDSw"]}→{n["IDSw"]}, '
            f'AssA {b["AssA"]}→{n["AssA"]}, IDF1 {b["IDF1"]}→{n["IDF1"]}; источник pooled. '
            f'Стоимость {r["seconds"]}с; источник seconds. Независимая сверка: runs/b123_20261027/sam2_query_output_pilot/independent_audit.json.',
        interpretation='Испытанное правило выдачи масочной рамки с нативной оценкой присутствия и отказом по перекрытию не обеспечило пользу. '
            'Измерены рост ложных рамок и смен идентификатора при сокращении пропусков. Какая доля вреда вызвана неверной личностью, геометрией или повторным выходом, отдельно не измерено. '
            'Нельзя объявлять конкретную внутреннюю причину установленной. Это результат выдачи по запросу, а не плотного переноса на каждом кадре.',
        decision='Не принимать эту реализацию в стенд. Следующая проверяемая замена — требовать для выдачи поддержку текущего реального обнаружения. '
            'Сравнить одинаковое правило сопоставления при масочном кандидате и при прогнозе Калмана; это отделяет вклад масочной истории от самого ограничения по детектору. '
            'Другая обоснованная замена — точный обученный выбор маски LTR, но его код и веса пока не подтверждены. Пороги на использованных val не подбирать.',
        cannot_claim=p['limitation']+' Нельзя считать эту адаптацию точной LTR, SGAR-MOT, проверкой научных Б1–Б3 или вкладом ядра FlyVis. '
            'В поле adaptation первичного отчёта унаследована фраза о каждом исходном кадре, затем уточнён пропуск нейронных шагов. '
            'Точное исполнение: изображения и реальные назначения наблюдались каждый кадр; нейронная маска обновлялась только по фиксированным запросам. Исторический отчёт сохранён без изменения.')
    for key,value in fields.items():
        old=c.get(key,'')
        if old and old!=value:c.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=old,reason='Записан полный измеренный итог; прежний текст сохранён.'))
        c[key]=value
    c.update(state='completed_negative',done='2026-10-27')
    for item,met in zip(c['acceptance'],[True,True,False]):item.update(met=met,evidence=SOURCE)
    c['metrics']=[dict(key='HOTA',value=n['HOTA'],baseline=b['HOTA'],subset='same20val rawXYWH; query-only FP32 SAM2 output; noMASA',source=SOURCE,locator='pooled.sam2_mask.HOTA')]
    c['evidence'] += [dict(kind='file',ref=SOURCE,claim='Полный отрицательный итог на20видео'),dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/independent_audit.json',claim='Сверка сохранённых выходов, метрик и трёх парных сравнений')]
    assert path.read_bytes()==before;path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    relative=100*(n['HOTA']/b['HOTA']-1)
    from scripts.b123_safe_journal_20261027 import journal
    journal(c['id'],'масочная выдача по запросу: пропусков меньше, HOTA ниже',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n'
        '*Поправка описания исполнения.* '+fields['cannot_claim']+'\n\n'
        f'*Сводка к базе.* Масочная выдача по запросу → HOTA → {relative:.6f}% к базе; доля измеренного потолка неизвестна. '
        'Оба контроллера код0/all_owned_ended, PID/создание проверены. Выходы и TrackEval не пересчитывались. '
        'Независимая сверка воспроизвела все три парных сравнения и их интервалы. Не принимать в стенд; точная LTR не опровергнута.')
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — масочная выдача завершена отрицательно\n\n'
            '140/150 завершены код0/all_owned_ended, независимый аудит passed. HOTA57.900829→56.277186, paired−1.677418[−2.451217;−0.976960],2/20лучше. '
            'FP3331→10218,FN13525→10517,IDSw1001→1131. Лучше простого продолжения20/20,хуже BUSCA по парному интервалу. '
            'Карточка sam2_query_mask_output_pilot_v1 completed_negative с revisions/HOTA/source/locator. '
            'Нужны каталог/58тестов и обновление одной фундаментальной задачи4,безтехническихузлов и изменения гипотез. '
            'Следующая замена: выдавать только рамки текущего реального детектора,которые поддержаны масочным кандидатом; '
            'контрольтотжепоКалману,всёостальноеодинаково. Толькоодинзаранеефиксированныйпорогизштатнойассоциации,никакогоподборапоval. '
            'Возможен CPU-повтор сохранённых причинных кандидатов без новогоGPU; отдельная карточка/протокол до измерения. '
            'ТочнаяLTRнеподменяется,Б1–Б3HOTAпо-прежнемунепроверены. Старыеэтапынеповторять,непродлевать. '
            'Guard recovery1проверятьпоPID+birth,07:20сохранение/07:29:30стоп/07:30приостановитьавтоматизацию,.git/index.lockнетронут.\n')
    print(json.dumps(dict(recorded=True,HOTA=n['HOTA'],relative_percent=relative,paired=r['paired']['vs_baseline']),ensure_ascii=False))
if __name__=='__main__':main()
