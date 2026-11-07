"""Записать, что малая HOTA не отделена от калибровки без изображений."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/pixel_box_numeric_control/report.json';r=read(ROOT/source);a=read(RUN/'pixel_box_numeric_control/independent_audit.json')
    assert r['complete'] and a['passed']
    for name in ['pixel_box_numeric_control','pixel_box_numeric_audit']:
        c=read(RUN/name/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    pairs=r['paired']['all40'];assert pairs['numeric_vs_baseline']['ci95'][0]>0
    assert all(pairs[f'{arm}_seed{s}_a0.5_vs_numeric']['ci95'][0]<=0 for arm in ['current','temporal'] for s in [42,43,44])
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='pixel_box_numeric_control_v1');assert not card['obtained']
    card.update(state='inconclusive',done='2026-10-27',obtained=source+':frozen='+json.dumps(r['frozen'],ensure_ascii=False)+';pooled='+json.dumps(r['pooled'],ensure_ascii=False)+';paired.all40='+json.dumps(pairs,ensure_ascii=False)+
        '. runs/recovery_research_20261027/pixel_box_numeric_control/independent_audit.json:passed=true; все40,смещения/JSON/выборdev/независимыеCIпроверены.',
        interpretation='Калибровка без изображений дала малую подтверждённую прибавку к исходному трекеру. Ни один из шести пиксельных вариантов не подтвердил превосходства над ней. '
            'Поэтому малую прибавку пиксельной реализации нельзя считать доказательством пользы изображения; это не эквивалентность и не доказательство бесполезности изображений.',
        decision='Не продолжать настройку геометрических голов по этим val. Сохранить частные положительные сравнения и перейти к изменению реальных назначений внутри основного трекера. Основной baseline не менять.')
    for acc in card['acceptance'][:2]:acc.update(met=True,evidence='runs/recovery_research_20261027/pixel_box_numeric_control/independent_audit.json:passed=true')
    card['acceptance'][2].update(met=False,evidence=source+':paired.all40:все6пиксельныхконтрастовкчисловомуконтролюci95включают0')
    b=r['pooled']['baseline']['HOTA'];value=r['pooled']['numeric_control']['HOTA'];card['metrics']=[dict(key='HOTA',value=value,baseline=b,subset='40 historical val',source=source,locator=['pooled','numeric_control','HOTA'])]
    card['confirmed_results']=[dict(id='numeric_box_calibration_vs_baseline40',title='Дешёвая калибровка рамок — малая прибавка без изображений',category='tracker',metric='HOTA',unit='points',comparison='Фиксированная наdevлинейнаякалибровкабезизображенийпротивисходноготрекера',subset='40 исторических val',conditions='Коэффициенты толькоtrain;метод/сила.5выбранатолько35devдоегоval;тежеисходныерамки/ID/классы/баллы/число',limitations='Последующаяпоисковаяпроверкапослепиксельногоval,не независимыйновыйитоговыйтест. ЯдроFlyVisнеиспользовалось,критерийотдельнойпользыпикселейнепройден.',evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','all40','numeric_vs_baseline']))]
    card['evidence'] += [dict(kind='file',ref=source,claim='ПолнаяHOTAдешёвогоконтроляисравненияспрежними6CNN'),dict(kind='file',ref='runs/recovery_research_20261027/pixel_box_numeric_control/independent_audit.json',claim='Независимаяпроверкавсехвыходовиинтервалов')]
    pixels=next(c for c in cat['experiments'] if c['id']=='pixel_box_refinement_v1')
    for field,extra in [('interpretation',' Последующая проверка завершена: линейная калибровка без изображений дала сопоставимую малую прибавку, а превосходство всех6CNN над ней не подтверждено; источник '+source+':paired.all40. Эквивалентность не проверялась.'),('decision',' Дешёвый контроль выполнен, полная научная попытка геометрии закончена. Следующее действие — основное сопоставление; не повторять готовые CNN/пороговые сетки.')]:
        old=pixels[field];pixels.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=field,old=old,reason='Получен заранее поставленный дополнительный контроль без изображений.'));pixels[field]=old+extra
    pixels['evidence'].append(dict(kind='file',ref=source,claim='Дополнительная польза пикселей над числовой калибровкой не подтверждена'))
    assert path.read_bytes()==before;write(path,cat)
    journal(card['id'],'малая прибавка рамок не требует доказанного вклада изображений',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты к базе.* HOTA '+str(value)+' против '+str(b)+',относительнаяприбавка '+str(100*(value/b-1))+'%. Долюпотолкасчитатьнельзя:максимумнеизвестен. '
        'Каталог требует обязательных проверок. Старые выводы пиксельной карточки сохранены в revisions, основной базис не изменён, .git/index.lock не тронут.')
    print('Recorded numeric calibration and unconfirmed added pixel benefit')

if __name__=='__main__':main()
