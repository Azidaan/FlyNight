"""Сохранить частный положительный результат и непроверенную добавку маски."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'detection_supported_recovery';SOURCE=RUN.relative_to(ROOT).as_posix()+'/report.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    r=read(RUN/'report.json');a=read(RUN/'independent_audit.json');p=read(RUN/'protocol.json');assert r['complete'] and a['passed']
    for folder in [RUN,BASE/'detection_supported_audit']:
        e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended']
        for pid,birth in e['owned'].items():
            if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    b=r['pooled']['baseline'];m=r['pooled']['mask_support'];k=r['pooled']['kalman_support']
    assert r['paired']['mask_vs_baseline']['ci95'][0]>0 and r['paired']['kalman_vs_baseline']['ci95'][0]>0
    assert r['paired']['mask_vs_kalman']['ci95'][0]<0<r['paired']['mask_vs_kalman']['ci95'][1]
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);c=next(c for c in cat['experiments'] if c['id']=='detection_supported_recovery_v1');assert c['state']=='ready'
    fields=dict(summary='На этих20видео поддержка текущим обнаружением улучшила HOTA и для Калмана, и для масочного кандидата. Дополнительная польза масочного пути не подтверждена.',
        obtained=f'{SOURCE}: pooled.baseline.HOTA={b["HOTA"]}; pooled.kalman_support.HOTA={k["HOTA"]}; pooled.mask_support.HOTA={m["HOTA"]}; paired={r["paired"]}. '
            f'Калман:FP={k["CLR_FP"]},FN={k["CLR_FN"]},IDSw={k["IDSw"]},AssA={k["AssA"]},IDF1={k["IDF1"]}. '
            f'Маска:FP={m["CLR_FP"]},FN={m["CLR_FN"]},IDSw={m["IDSw"]},AssA={m["AssA"]},IDF1={m["IDF1"]};источникpooled. '
            f'Стоимость={r["seconds"]}с;источникseconds. Независимаясверка:runs/b123_20261027/detection_supported_recovery/independent_audit.json.',
        interpretation='Предварительная польза нового правила выдачи на использованных видео подтверждена парными интервалами. '
            'Дешёвый контроль без масок получил почти ту же среднюю прибавку; добавочный эффект масочного пути не установлен. '
            'Интервал с нулём не доказывает эквивалентности. FP и IDSw выросли в обоих вариантах, поэтому выигрыш HOTA не означает улучшения всех метрик. '
            'Это результат всего проекта; ядро FlyVis и нейронный модуль FlyVis здесь не использовались.',
        decision='Сохранить частные положительные сравнения и общий неопределённый вывод о добавке маски. '
            'Подготовить выключенный по умолчанию исполняемый вариант Калман+поддержка текущим обнаружением и проверить точное совпадение с этим сохранённым выходом. '
            'До независимого испытания не заменять основной baseline. Порогов не подбирать, GPU-маски ради той же прибавки не пересчитывать.',
        cannot_claim=p['limitation']+' Нельзя считать установленным преимущество масочной ветви над сопоставимым контролем, эквивалентность вариантов или пользу на новых видео. '
            'Два частных положительных сравнения не снимают непройденного критерия дополнительной пользы маски.')
    for key,value in fields.items():
        old=c.get(key,'')
        if old and old!=value:c.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=old,reason='Сохранён полный измеренный итог с частной пользой и непройденным критерием добавки маски.'))
        c[key]=value
    c.update(state='inconclusive',done='2026-10-27')
    for item,met in zip(c['acceptance'],[True,True,False]):item.update(met=met,evidence=SOURCE)
    c['metrics']=[dict(key='HOTA',value=val['HOTA'],baseline=b['HOTA'],subset='same20val;current detection support;'+arm,source=SOURCE,locator='pooled.'+arm+'.HOTA') for arm,val in [('kalman_support',k),('mask_support',m)]]
    h=hashlib.sha256((RUN/'report.json').read_bytes()).hexdigest()
    c['confirmed_results']=[dict(id='current_detection_support_'+arm,title=title,category='tracker',metric='HOTA',unit='points',
        comparison=title+' против исходного XYWHByteTrack',subset='те же20val,выдача по фиксированным запросам,безMASA',
        conditions='Одни текущие обнаружения и один геометрический фильтр; основная выдача и состояние трекера сохранены.',
        limitations='Исследовательская проверка на повторно использованных видео; добавка маски не подтверждена, FP/IDSw выросли; не вклад FlyVis.',
        evidence=dict(source=SOURCE,sha256=h,locator=['paired',contrast]),baseline=dict(source=SOURCE,sha256=h,locator=['pooled','baseline','HOTA']))
        for arm,title,contrast in [('kalman','Калман с поддержкой текущим обнаружением','kalman_vs_baseline'),('mask','Маска с поддержкой текущим обнаружением','mask_vs_baseline')]]
    c['evidence'] += [dict(kind='file',ref=SOURCE,claim='Три полных парных сравнения'),dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/independent_audit.json',claim='Выходы, источники и все интервалы проверены независимо')]
    assert path.read_bytes()==before;path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal(c['id'],'поддержка текущим обнаружением дала прибавку, добавка маски не подтверждена',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n'
        f'*Сводка к базе.* Калман+поддержка → HOTA → {100*(k["HOTA"]/b["HOTA"]-1):.6f}% к базе; маска+поддержка → HOTA → {100*(m["HOTA"]/b["HOTA"]-1):.6f}% к базе. '
        'Доля измеренного потолка неизвестна. Непройденный критерий добавки маски сохранён; карточкаinconclusive,два частных сравнения вconfirmed_results. '
        'Это не разрешение подбирать настройки по этимval или автоматически менять основной стенд.')
    with (BASE/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — частная польза текущего обнаружения подтверждена\n\n'
            '160/170код0/all_owned_ended,independent_auditpassed. HOTAбаза57.900829,Калман+поддержка58.289066,маска+поддержка58.316904. '
            'pairedКалман−база+0.388059[0.095371;0.703891],маска−база+0.398720[0.058419;0.743423]. '
            'Маска−Калман+0.010661[−0.078896;0.111054]:нетдоказаннойдобавкимаски,нетдоказаннойэквивалентности. '
            'КалманFP4436,FN12280,IDSw1100;маскаFP4772,FN11706,IDSw1266. '
            'Карточкаinconclusiveсдвумяconfirmed_results,третийкритерийmet=falseсохранён. Каталог/58тестовещёнужны. '
            'Следующийшаг—выключенныйпоумолчаниюонлайнКалман+детектор:входтолькотекущиедетекции/исходныйlostKalman,точныйповторсохранённых20выходов,безповторногоTrackEval. '
            'Основнойbaselineнеменять;эточастнаяпредварительнаяпользапроекта,неFlyVis,нетнезависимогоитоговоготеста. '
            'Послеинтеграциизафиксироватьостатокнезависимойпроверки,неподбиратьпороги. Ночнойсроксохраняется.\n')
    print(json.dumps(dict(recorded=True,overall_state=c['state'],paired=r['paired']),ensure_ascii=False))
if __name__=='__main__':main()
