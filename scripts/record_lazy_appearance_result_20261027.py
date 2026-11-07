"""Записать полный результат назначения по внешнему виду после независимого повторения."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/lazy_appearance_validation/report.json'
    r=read(ROOT/source);a=read(RUN/'lazy_appearance_validation/independent_audit.json')
    assert r['complete'] and a['passed'] and a['report_sha256']==sha(ROOT/source)
    for folder in ['lazy_appearance_validation','lazy_appearance_validation_audit']:
        c=read(RUN/folder/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    pairs=r['paired']['all40'];positive=all(v['ci95'][0]>0 for v in pairs.values()) and r['frozen']['weight']>0
    assert positive and a['prespecified_decision']['all40_positive_vs_baseline_and_all3_nulls']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    card=next(c for c in cat['experiments'] if c['id']=='lazy_appearance_association_v1');assert not card['obtained']
    card.update(state='completed_positive',done='2026-10-27',
        obtained=source+':frozen='+json.dumps(r['frozen'],ensure_ascii=False)+';pooled='+json.dumps(r['pooled'],ensure_ascii=False)+';paired='+json.dumps(r['paired'],ensure_ascii=False)+'. runs/recovery_research_20261027/lazy_appearance_validation/independent_audit.json:passed=true;полное повторение всех назначений и выходов,хеши/прошлое/независимыеCI проверены.',
        interpretation='На совокупных40 исторических val эта фиксированная поправка настоящих назначений дала малую положительную парную HOTA к исходной базе и всем трём бессмысленным контролям. '
            'На каждой отдельной половине20 интервал к базе включает0. Нижний предел на40 близок к нулю, большой прирост и переносимость на новые данные не установлены. '
            'Преимущество над случайным штрафом отделяет полезность признаков в этой реализации; оно не устанавливает превосходства над сильной MASA или всеми способами сопоставления.',
        decision='Сохранить положительный ограниченный итог. Проверить фиксированную композицию с ранее выбранной числовой калибровкой и сопоставимый сильный MASA-контроль на общей первой20части. '
            'Веса/силы/пороги по val не перенастраивать; основной baseline не менять.')
    for acc in card['acceptance']:acc.update(met=True,evidence='runs/recovery_research_20261027/lazy_appearance_validation/independent_audit.json:passed=true;'+source+':paired.all40')
    b=r['pooled']['baseline']['HOTA'];v=r['pooled']['appearance']['HOTA']
    card['metrics']=[dict(key='HOTA',value=v,baseline=b,subset='40 historical val',source=source,locator=['pooled','appearance','HOTA'])]
    card['confirmed_results']=[dict(id='online_appearance_vs_native40',title='Внешний вид в спорном назначении — малая прибавка HOTA',category='tracker',metric='HOTA',unit='points',
        comparison='Реальное первое назначение с фиксированным внешним видом против исходного XYWHByteTrack',subset='40 исторических val',
        conditions='Теже детекции/YOLO/буфер/порог;вес0,1 выбрантолько35dev,сетьзаморожена. Положительные нижние пределы также ко всем3бессмысленнымконтролям.',
        limitations='Малая прибавка,нижний предел к базе близок0;каждаяполовина20неопределённа. Историческийval,не новыйнезависимыйитоговыйтест. НетядраFlyVis,нетдоказанного превосходстванадMASA.',
        evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','all40','appearance_vs_baseline']))]
    card['evidence'] += [dict(kind='file',ref=source,claim='Полныйнаучныйитогсорокавидео'),dict(kind='file',ref='runs/recovery_research_20261027/lazy_appearance_validation/independent_audit.json',claim='Полноеповторение назначенийи независимыеинтервалы')]
    assert path.read_bytes()==before;write(path,cat)
    journal(card['id'],'внешний вид реальных назначений дал малую прибавку на совокупных40 видео',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты к базе.* СовокупнаяHOTA '+str(b)+'→'+str(v)+';относительнаяприбавка '+str(100*(v/b-1))+'%. Доля максимума неизвестна:GTдиагностика не является потолком.'
        ' Исходный135отказ и исправленный137аудит сохранены отдельно. Каталог требует обязательныхпроверок;lockнетронут,коммит/pushнебыли.')
    print('Recorded positive but small online association result')

if __name__=='__main__':main()
