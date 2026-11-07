"""Записать фиксированную композицию и сильный контроль без смешения выборок."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/fixed_composition/report.json';r=read(ROOT/source);a=read(RUN/'fixed_composition/independent_audit.json')
    assert r['complete'] and a['passed'] and a['report_sha256']==sha(ROOT/source)
    for sub in ['fixed_composition','fixed_composition_audit']:
        c=read(RUN/sub/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    pairs=r['paired']['all40'];positive=all(x['ci95'][0]>0 for x in pairs.values());assert positive==a['prespecified_composition_better_than_both_parents']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='fixed_association_geometry_composition_v1');assert not card['obtained']
    card.update(state='completed_positive' if positive else 'inconclusive',done='2026-10-27',
        obtained=source+':pooled='+json.dumps(r['pooled'],ensure_ascii=False)+';paired='+json.dumps(r['paired'],ensure_ascii=False)+'. runs/recovery_research_20261027/fixed_composition/independent_audit.json:passed=true;40полныхвыходов/20MASA,нулевойпуть/прежняякалибровка/прошлое/хеши/независимыеCIпроверены.',
        interpretation=('Фиксированная композиция дала положительный парный интервал ко всем трём вариантам: исходной базе и обеим отдельным поправкам на совокупных40 видео.' if positive else
            'Превосходство фиксированной композиции над базой и обоими отдельными родителями одновременно не подтверждено. Частные положительные сравнения сохраняются отдельно.')+
            ' Сильный старый MASA сравнивается только на общей первой20части,с отдельными парными интервалами в paired.development20. '
            'Он использует офлайн-сшивку,новая ветвь—текущие назначения;это сравнение полных систем при разных временных режимах,не изоляция архитектуры. '
            'Эффекты не складывались заранее. Ни историческая выборка,ни малая прибавка не доказывают большого запаса,обобщения или новизны.',
        decision='Не перенастраивать готовые ветви по этим val. Сохранить полный итог и сильный сопоставимый контроль. '
            'Для следующего изменения назначений рассмотреть замену извлекателя,обученного на людях,на уже имеющийся классонезависимый MASA,с тем же действием/нулевыми контролями и выбором толькоdev. '
            'Наличие опубликованного метода и готовых весов не гарантирует прибавку;новая реализация требует отдельного протокола и короткой фактической цены. Основнойbaselineне менять.')
    card['acceptance'][0].update(met=True,evidence='runs/recovery_research_20261027/fixed_composition/independent_audit.json:passed=true')
    card['acceptance'][1].update(met=positive,evidence=source+':paired.all40;сильныеMASAсравнениявpaired.development20сохраняютсяотдельно')
    card['acceptance'][2].update(met=True,evidence='runs/recovery_research_20261027/fixed_composition/independent_audit.json:passed=true')
    b=r['pooled']['baseline']['HOTA'];v=r['pooled']['appearance_numeric']['HOTA']
    card['metrics']=[dict(key='HOTA',value=v,baseline=b,subset='40 historical val',source=source,locator=['pooled','appearance_numeric','HOTA'])]
    if pairs['appearance_numeric_vs_baseline']['ci95'][0]>0:
        card['confirmed_results']=[dict(id='fixed_composition_vs_baseline40',title='Фиксированные назначения и калибровка — общая малая прибавка',category='tracker',metric='HOTA',unit='points',
            comparison='Композиция прежнего сопоставления по внешнему виду и числовой калибровки против исходного трекера',subset='40 исторических val',
            conditions='Обе части зафиксированы наtrain/devранее;послеихvalнормировки/силы/веса/порогине менялись. Новых весов/изображений/GPUнет.',
            limitations='Последующая поисковая композиция на уже просмотренныхисторическихval,не новый независимыйитоговыйтест. Отдельный сильный MASAконтроль толькообщие20,инойвременнойрежим. НетядраFlyVis.',
            evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','all40','appearance_numeric_vs_baseline']))]
    if r['paired']['development20']['masa_numeric_vs_masa']['ci95'][0]>0:
        card.setdefault('confirmed_results',[]).append(dict(id='fixed_numeric_over_old_masa20',title='Готовая MASA с фиксированной калибровкой — малая прибавка на общих двадцати видео',category='tracker',metric='HOTA',unit='points',
            comparison='Одна прежняя числовая калибровка поверх готовой MASA против той же MASA без калибровки',subset='20 общих исторических val',
            conditions='Теже готовые офлайнMASAID/рамки/баллы/число;калибровка обученаtrain,выбранаdevранее и зафиксирована. Новойсшивкинебыло.',
            limitations='Последующий поисковый контроль,не новый независимыйтест. Только20;не сравниватьабсолютнуюHOTAсбазой40. Положительнаямаладобавкакоординат,не новое преимуществоонлайнсопоставленияилиFlyVis.',
            evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','development20','masa_numeric_vs_masa'])))
    card['evidence'] += [dict(kind='file',ref=source,claim='Полные исходы композиции и общего20MASAконтроля'),dict(kind='file',ref='runs/recovery_research_20261027/fixed_composition/independent_audit.json',claim='Независимое восстановление всех координат и интервалов')]
    assert path.read_bytes()==before;write(path,cat)
    journal(card['id'],'фиксированная композиция и сильный общий контроль полностью проверены',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты к базе.* СовокупнаяHOTA40 '+str(b)+'→'+str(v)+';относительнаяразность '+str(100*(v/b-1))+'%. Максимумнеизвестен,доляпотолканеопределима. '
        'Сильныйконтрольна20не сравниваетсясбазой40. Каталогтребуетобязательныхпроверок,основнойbaseline/старыйlockне менялись.')
    print('Recorded fixed composition; primary positive:',positive)

if __name__=='__main__':main()
