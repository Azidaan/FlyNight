"""Записать полный результат допуска, сохранив непройденный критерий правила."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/learned_validation/report.json'
    report=read(ROOT/source);audit=read(RUN/'learned_validation/independent_audit.json')
    done=read(RUN/'learned_validation/controller_result.json')
    assert report['complete'] and audit['passed'] and done['exit_code']==0 and done['all_owned_ended']
    pairs=report['paired']['all40'];base=report['pooled']['baseline']['HOTA']
    assert all(pairs[f'appearance_seed{s}_vs_baseline']['ci95'][0]>0 for s in [42,43,44])
    assert all(pairs[f'appearance_seed{s}_vs_rule']['ci95'][0]<0 for s in [42,43,44])
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    card=next(c for c in cat['experiments'] if c['id']=='recovery_admission_learning_v1');assert not card['obtained']
    card.update(state='inconclusive',done='2026-10-27',obtained=(
        source+': pooled.HOTA: baseline='+str(base)+', rule='+str(report['pooled']['rule']['HOTA'])+
        '; appearance seeds42/43/44='+str([report['pooled'][f'appearance_seed{s}']['HOTA'] for s in [42,43,44]])+
        '. Все парные сравнения: '+source+':paired.all40='+json.dumps(pairs,ensure_ascii=False)+
        '. FP/FN/IDSw/AssA/IDF1 находятся в том же report.pooled и per_video. '
        'runs/recovery_research_20261027/learned_validation/independent_audit.json: passed=true; все40видео,точныйбазис/правило,фиксированныевеса/пороги,строгоепрошлое.'),
        interpretation='На совокупных40историческихval модель с областями немного лучше исходного трекера во всех трёх сидах. '
            'Преимущество над дешёвым фиксированным правилом не подтверждено ни в одном сиде. Дополнительный вклад областей к числовому контролю '
            'подтверждён для42/43, но не44, поэтому устойчивый вклад не установлен. Это не доказательство эквивалентности. '
            'Общий критерий не выполнен; частный положительный результат относительно исходного трекера сохраняется отдельно.',
        decision='Эту полную попытку завершить. Не подбирать пороги или архитектуру по этим val и не менять основной baseline. '
            'Перейти к отдельной поправке геометрии по изображению, для которой уже измерен диагностический резерв; '
            'предварительно отличить её от прежнего сглаживания и выбора детекторной рамки.')
    card['acceptance'][1].update(met=True,evidence='runs/recovery_research_20261027/frozen_for_validation.json; learned_validation/independent_audit.json:fixed_weights_and_thresholds_exact=true')
    card['acceptance'][2].update(met=False,evidence=source+':paired.all40: все appearance_vs_rule ci95 включают0; appearance_vs_numeric_seed44 ci95 включает0. Уточнение всех3сидов записано в протоколе до val.')
    old=card['cannot_claim'];card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field='cannot_claim',old=old,reason='Полный val завершён; прежнее ограничение до проверки сохранено.'))
    card['cannot_claim']='Нельзя заявлять большую прибавку, преимущество над правилом, устойчивую добавку внешнего вида или эквивалентность контролям. '
    card['cannot_claim']+='Все40val имеют историю проекта: это не новый независимый итоговый тест. Не проверены основные Б1–Б3; ядро и полный модуль FlyVis здесь не используются. '
    card['cannot_claim']+='Диагностический резерв с GT не является потолком или достижимой прибавкой. Основные назначения и память исходного трекера не изменялись.'
    card['metrics']=[dict(key='HOTA',value=report['pooled']['appearance_seed42']['HOTA'],baseline=base,subset='40 historical val, fixed seed42',source=source,locator=['pooled','appearance_seed42','HOTA'])]
    card['confirmed_results']=[dict(id='appearance_admission_vs_original_40',title='Обучаемый допуск с областями — небольшая польза относительно исходного трекера',category='tracker',metric='HOTA',unit='points',
        comparison='Фиксированная модель с областями, первый заранее заданный сид42, против исходного трекера',subset='40 прежних val:20разработки+20переноса',
        conditions='Веса и пороги выбраны толькоtrain/dev до val. Все3сида имеют положительный интервал к исходной базе на совокупных40; здесь показан первый, а не лучший сид.',
        limitations='Общий исход inconclusive: преимущество над правилом не подтверждено; добавка областей не устойчива во всех3сидах. Исторический val, не новый независимый тест. FlyVis не использовался.',
        evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','all40','appearance_seed42_vs_baseline']))]
    card['evidence'] += [dict(kind='file',ref=source,claim='Полная HOTA шести фиксированных моделей и оба контроля'),dict(kind='file',ref='runs/recovery_research_20261027/learned_validation/independent_audit.json',claim='Независимая сверка всех выходов, весов, прошлого и интервалов')]
    assert path.read_bytes()==before;write(path,cat)
    relative=100*(report['pooled']['appearance_seed42']['HOTA']/base-1)
    journal(card['id'],'полная HOTA обучаемого допуска: малая польза к базе, преимущество над правилом не подтверждено',
        '*Постановка.* Одинаковые кандидаты, две головы с равным числом параметров, три сида, веса и пороги толькоtrain/dev. '
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты к базе.* Первый фиксированный сид42: HOTA '+str(report['pooled']['appearance_seed42']['HOTA'])+' против '+str(base)+
        ', относительное изменение '+str(relative)+'%. Долю потолка считать нельзя: максимальная достижимая HOTA не установлена. '
        'Контроллер код0/all_owned_ended; каталог требует обязательных проверок. Старый .git/index.lock не тронут, коммит/push не выполнены.')
    print('Recorded inconclusive full experiment and qualified partial benefit; baseline unchanged')

if __name__=='__main__':main()
