"""Записать полный результат переноса без выбора порогов и повторного расчёта."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,hashlib
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'supported_transfer/recovery2';SOURCE=RUN.relative_to(ROOT).as_posix()+'/report.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    r=read(RUN/'report.json');a=read(RUN/'independent_audit.json');p=read(RUN/'protocol.json');assert r['complete'] and a['passed'] and a['completed_ten_outputs_preserved_exactly']
    for folder in [RUN,BASE/'supported_transfer/recovery2_audit']:
        e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['timed_out']
        for pid,birth in e['owned'].items():
            if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    b=r['pooled']['baseline'];n=r['pooled']['supported'];contrast=r['paired'];low,high=contrast['ci95']
    state='completed_positive' if low>0 else 'completed_negative' if high<0 else 'inconclusive'
    finding='Положительная прибавка HOTA подтверждена на другой части прежнего val.' if low>0 else 'На другой части прежнего val подтверждён вред HOTA.' if high<0 else 'На другой части прежнего val польза HOTA не подтверждена: парный интервал включает ноль.'
    relative=100*(n['HOTA']/b['HOTA']-1)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);c=next(c for c in cat['experiments'] if c['id']=='detection_support_transfer_v1');assert c['state']=='ready'
    fields=dict(summary=finding,
        obtained=f'{SOURCE}: pooled.baseline.HOTA={b["HOTA"]};pooled.supported.HOTA={n["HOTA"]};paired={contrast}. '
            f'CLR_FP {b["CLR_FP"]}→{n["CLR_FP"]}, CLR_FN {b["CLR_FN"]}→{n["CLR_FN"]}, IDSw {b["IDSw"]}→{n["IDSw"]}, '
            f'AssA {b["AssA"]}→{n["AssA"]},IDF1 {b["IDF1"]}→{n["IDF1"]};источникpooled. '
            'Сверка:runs/b123_20261027/supported_transfer/recovery2/independent_audit.json.',
        interpretation=finding+' Выборка не пересекается с20разработки правила,но принадлежит прежнемуval с историей использования. '
            'Это измерение параллельного выходного модуля проекта. Ядро и полный модуль FlyVis, SAM2 и точнаяLTR здесь не использовались. '
            'Причины изменения ошибок личности, геометрии и числа ложных рамок этим сравнением по отдельности не установлены.',
        decision=('Сохранить фиксированный опциональный модуль и положительный результат переноса. Основнойbaseline без прямого решения автора не заменять; '
            'независимый итоговый тест и проверка эксплуатационной стоимости остаются отдельными работами.' if low>0 else
            'Не принимать модуль в основнойbaseline по результату переноса. Сохранить положительный частный итог на20разработки и этот результат отдельно. '
            'Не подбирать пороги на новой20части и не пересчитывать готовые видео. Следующую замену проверять на заранее разделённых данных разработки и проверки.'),
        cannot_claim=p['limitation']+' Подготовка времени одного видео уточнена по исходным пикселям после конфликта уменьшенных изображений; окно поиска не расширено, '
            'аннотации объектов и частные HOTA для решения об исправлении не использовались. Десять готовых видео сохранены по хешам и не пересчитывались. '
            'Положительный результат первых20 не следует автоматически из результата этих20 и наоборот. Интервал с нулём не доказывает отсутствия эффекта или эквивалентности.')
    for key,value in fields.items():
        old=c.get(key,'')
        if old and old!=value:c.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=old,reason='Записан полный результат переноса, прежняя формулировка сохранена.'))
        c[key]=value
    c.update(state=state,done='2026-10-27')
    for item,met in zip(c['acceptance'],[True,low>0,True]):item.update(met=met,evidence=SOURCE)
    c['metrics']=[dict(key='HOTA',value=n['HOTA'],baseline=b['HOTA'],subset='other20prior-val;disjoint-from-rule-development;fixedKalman+current-detection-support',source=SOURCE,locator='pooled.supported.HOTA')]
    c['evidence'] += [dict(kind='file',ref=SOURCE,claim='Полное сравнение на другой20части прежнегоval'),dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/independent_audit.json',claim='Входы,сырые массивы,разделениевыборок,интервалы,сохранность готовых10проверены'),dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/service_recovery.json',claim='Проверенное исправление временнойпривязки,старый срок и повторный расчёт готового исключён')]
    if low>0:
        h=hashlib.sha256((RUN/'report.json').read_bytes()).hexdigest()
        c['confirmed_results']=[dict(id='current_detection_support_transfer',title='Фиксированная поддержка обнаружением на другой части val',category='tracker',metric='HOTA',unit='points',
            comparison='Неизменный XYWH с выходным правилом против того же XYWH без дополнений',subset='20других прежнихvalвидео,безпересеченияс20разработки',
            conditions='Одни текущие YOLO26обнаружения,фиксированное правило,основное состояние трекера не меняется.',limitations=p['limitation'],
            evidence=dict(source=SOURCE,sha256=h,locator=['paired']),baseline=dict(source=SOURCE,sha256=h,locator=['pooled','baseline','HOTA']))]
    assert path.read_bytes()==before;path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal(c['id'],'фиксированный модуль проверен на другой20частиval',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n'
        f'*Сводка к своей базе.* Правило текущей поддержки → HOTA → {relative:.6f}% к baseline именно этой20частиval;доля измеренного потолка неизвестна. '
        'С базой57.900829первых20это число напрямую не сравнивается. Два контроллера завершеныкод0/all_owned_ended,независимыйаудитpassed. '
        'Прежниеошибкизаписи/выравниваниясохраненыотдельно,готовые10видеонеповторены. '
        'Остаток:точнаяLTRиосновныеБ1–Б3непровереныHOTA;нетновогоитоговогонезависимогонабора. Основнойbaselineосталсянеизменным.')
    with (BASE/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — полный результат переноса сохранён\n\n'
            f'{SOURCE};state={state};baseline.HOTA={b["HOTA"]};supported.HOTA={n["HOTA"]};paired={contrast}. '
            '240/250код0/all_owned_ended,independent_auditpassed.10готовыхвидеопохешамсохранены,новые10доведены,исправлениепривязкипоисходнымпикселямпроверено. '
            'Карточкаdetection_support_transfer_v1записанасrevisions/HOTA/source/locator;нужныкаталог/58тестов и обновлениеоднойфундаментальнойзадачи4. '
            'Неподбиратьпороги/неповторятьготовое,основнойbaselineнеменять. Необъявлятьэтовалвпервыеиспользованнымпроектомтестом. '
            'ОсновныеБ1–Б3непровереныHOTA,точнаяLTRнереализована,собственныеадаптацииразличать. '
            'Ночнойостатоксохранить;новыедлинныесериипередсрокомнезапускать. Сохранение07:20,стопдеревьев07:29:30,07:30автоматизациюприостановить.\n')
    print(json.dumps(dict(recorded=True,state=state,baseline_HOTA=b['HOTA'],HOTA=n['HOTA'],paired=contrast,relative_percent=relative),ensure_ascii=False))
if __name__=='__main__':main()
