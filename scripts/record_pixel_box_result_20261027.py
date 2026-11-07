"""Полный измеренный итог геометрии; отдельный вклад прошлого не подтверждён."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/pixel_box_validation/report.json';r=read(ROOT/source);a=read(RUN/'pixel_box_validation/independent_audit.json')
    assert r['complete'] and a['passed']
    for folder in ['pixel_box_audit/recovery1','pixel_box_learning/recovery1','pixel_box_schedule/recovery1','pixel_box_dev','pixel_box_dev_audit','pixel_box_validation','pixel_box_validation_audit']:
        c=read(RUN/folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['timed_out']
    pairs=r['paired']['all40'];assert all(pairs[f'{arm}_seed{s}_a0.5_vs_baseline']['ci95'][0]>0 for arm in ['current','temporal'] for s in [42,43,44])
    assert not all(pairs['temporal_vs_current_seed'+str(s)]['ci95'][0]>0 for s in [42,43,44])
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='pixel_box_refinement_v1');assert not card['obtained']
    card.update(state='inconclusive',done='2026-10-27',obtained=source+': pooled='+json.dumps(r['pooled'],ensure_ascii=False)+'; paired.all40='+json.dumps(pairs,ensure_ascii=False)+
        '. Источник runs/recovery_research_20261027/pixel_box_dataset/audit.json:165видео,29208областей,25207известныхметок,660проверокпикселей. '
        'pixel_box_learning/recovery1/report.json:6моделей,336812параметровкаждая,15эпох,20060train/5147devизвестныхпримеров. '
        'pixel_box_frozen_for_validation.json:всеalpha=.5выбранытольконаdev. pixel_box_validation/independent_audit.json:passed=true.',
        interpretation='Обе версии дали небольшую подтверждённую прибавку к исходному трекеру на совокупных40историческихval во всех трёх сидах. '
            'Дополнительная польза прошлого изображения подтверждена только для первого сида, а не для всех трёх: общий временной критерий не выполнен. '
            'Геометрические изменения также меняют результаты сопоставления оценщика, хотя ID выхода не меняются; IDSw немного выросли. '
            'Польза пиксельной информации относительно более дешёвой систематической калибровки ещё не отделена.',
        decision='Сохранить частную пользу уточнения и непройденный временной критерий. Не менять основной baseline и не настраивать модель по val. '
            'Одна ограниченная следующая проверка — сравнение с калибровкой по классам и числовым признакам без изображений; затем перейти к основному сопоставлению, не тратить окно на тюнинг рамок.')
    card['acceptance'][0].update(met=True,evidence='runs/recovery_research_20261027/pixel_box_dataset/audit.json:passed=true; all_labels_recomputed/strict_past/train_dev_disjoint/exact_training_frames_excluded')
    card['acceptance'][1].update(met=True,evidence='runs/recovery_research_20261027/pixel_box_dev/independent_audit.json и pixel_box_validation/independent_audit.json:checkpoint_hashes_verified/null_exact/strict_past=true; все336812параметров')
    card['acceptance'][2].update(met=False,evidence=source+':paired.all40: оба варианта выше базы во всех3сидах, но temporal_vs_current_seed43/44 ci95 включают0; критерий добавки времени не выполнен')
    old=card['cannot_claim'];card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field='cannot_claim',old=old,reason='Полный измеренный итог получен; ограничения до расчёта сохраняются.'))
    card['cannot_claim']=old+' Нельзя заявлять большую прибавку, устойчивый вклад прошлого изображения, отдельную пользу пикселей против числовой калибровки или достижение GT-резерва. '
    card['cannot_claim']+='Эти40видео не являются новым независимым итоговым тестом. Результат не приписывается ядру или полному модулю FlyVis; основные Б1–Б3 этим не проверены.'
    base=r['pooled']['baseline']['HOTA'];card['metrics']=[dict(key='HOTA',value=r['pooled']['current_seed42_a0.5']['HOTA'],baseline=base,subset='40 historical val, first fixed current seed42',source=source,locator=['pooled','current_seed42_a0.5','HOTA'])]
    card['confirmed_results']=[dict(id='current_pixels_box_correction_vs_original40',title='Уточнение рамок по текущему изображению — малая прибавка полного трекера',category='tracker',metric='HOTA',unit='points',
        comparison='Первый заранее заданный сид42,current,сила.5 выбранатолькодев,против исходного трекера',subset='40 историческихval:20разработки+20переноса',
        conditions='Все3сидаcurrentимеют положительную нижнюю границу к базе; показан первый,не лучший. Рамки уточняются при тех же ID/классах/баллах/числе выходов.',
        limitations='Общий исход inconclusive: добавка времени не устойчива; вклад пикселей относительно дешёвой калибровки не отделён. Историческийval,не независимый итоговыйтест. FlyVisнеиспользовался.',
        evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','all40','current_seed42_a0.5_vs_baseline']))]
    card['evidence'] += [dict(kind='file',ref=source,claim='Полная HOTA всех шести фиксированных моделей и исходного варианта'),dict(kind='file',ref='runs/recovery_research_20261027/pixel_box_validation/independent_audit.json',claim='Все поправки и выходы восстановлены, CPU-повтор и независимые интервалы')]
    assert path.read_bytes()==before;write(path,cat)
    journal(card['id'],'уточнение рамок дало небольшую прибавку, устойчивый вклад прошлого не подтверждён',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты к базе.* current42,HOTA '+str(r['pooled']['current_seed42_a0.5']['HOTA'])+' против '+str(base)+',относительное изменение '+str(100*(r['pooled']['current_seed42_a0.5']['HOTA']/base-1))+
        '%. Долю потолка считать нельзя: максимум не установлен. Все собственные деревья этапов закончены, прежний абсолютный предел обучения соблюдён;старыеотказысохранены. '
        'Каталогтребуетпроверок,основнойbaselineне менялся,.git/index.lockнетронут.')
    print('Recorded qualified geometry benefit and failed temporal criterion')

if __name__=='__main__':main()
