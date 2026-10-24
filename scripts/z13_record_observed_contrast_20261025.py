"""Verify the fixed observed-contrast prototype and record its bounded outcome."""
from pathlib import Path
from collections import Counter
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_observed_contrast_20261025 import RUN,CAUSE,IDENT,read,write,sha,now,journal
W=ROOT/'runs/z13_weekend_20261024'


def main():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not(RUN/'finish_recorded.json').exists()
    p,r,e=read(RUN/'protocol.json'),read(RUN/'report.json'),read(RUN/'controller_result.json')
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'] and r['complete']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    verify_sources(p)
    bound=read(CAUSE/'rows.json')
    for arm,modes in r['results'].items():
        source={x['label']:x for x in bound if x['arm']==arm}
        for mode,result in modes.items():
            chosen=set(r['chosen_labels'][arm][mode]);assert len(chosen)<=84
            counts=Counter();pv={v:dict(known=0,beneficial=0,harmful=0) for v in result['per_video']}
            for idx,row in source.items():
                if row['kind']>=0:pv[row['video']]['known']+=1
                if idx not in chosen:continue
                counts['selected']+=1
                if row['kind']<0:counts['unknown']+=1
                else:
                    counts['known']+=1;counts[row['outcome']]+=1
                    pv[row['video']]['beneficial']+=row['outcome']=='beneficial';pv[row['video']]['harmful']+=row['outcome']=='harmful'
            assert dict(counts)==result['counts'] and pv==result['per_video']
    for name,value in r['paired'].items():
        left,right=name.split('-');a,mode=left.split('_',1);b,other=right.split('_',1)
        assert paired(r['results'][a][mode]['net_frequency'],r['results'][b][other]['net_frequency'])==value
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='Фиксированное сравнение с наблюдавшимися фоновыми пиками не прошло критерий применения.',
        obtained='runs/z13_observed_contrast_20261025/report.json → results: '+json.dumps({a:{m:x['counts'] for m,x in modes.items()} for a,modes in r['results'].items()},ensure_ascii=False)+
            ';paired: '+json.dumps(r['paired'],ensure_ascii=False)+'. Dнаблюдаемыхэпизодов2188,сфоновымиобразцами1149(prototype_statistics.D).',
        interpretation='Новая конкретная реализация сравнения цели с отдельными наблюдавшимися фоновыми пиками дала1полезноеиз84выбранныхD,21неизвестныйисход. Толькоодновидеосположительнымизменением;нижняяграницаДИравна0. Это не подтверждённое улучшение отказа,непределпредставленияиливсейлинииFlyVis.',
        decision='Не переносить эту версию в HOTA. Сохранить прототип и отрицательный критерий. Простые фиксированные скалярные декодеры пока не решают отказ. Дальнейшая замена должна иметь иной механизм и собственный дешёвый критерий;не продолжать подбор косинусных весов на этих26видео.',
        cannot_claim=p['limits']+' Принцип цель−фон уже был вz13_candidate_foreground_v1:среднийконтекствнерасширеннойрамки,обучаемыйgain,722выхода. Здесьдругаяреализация:отдельныенаблюдаемыефоновыепики,L2косинус,фиксированныйкоэффициент1,замороженныепрототипы,сохранённыйцентр. Это не первыйвпроектеконтрастсфоном и не новаяэмпирическаяпричинапровалапрежнегоforeground.')
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Полный проверенный итог фиксированного прототипа.'))
        card[key]=value
    card.update(state='completed_positive' if r['primary_pass'] else 'inconclusive',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True,evidence='report.json;sources_unchanged;all8326rows')
    card['acceptance'][1].update(met=r['primary_pass'],evidence='runs/z13_observed_contrast_20261025/report.json → paired')
    card['evidence'].append(dict(kind='file',ref='runs/z13_observed_contrast_20261025/report.json',claim='Всеисходы,контрольбезмухи,парныеинтервалы'))
    write(cp,cat)
    journal('итог фиксированного узла цель−наблюдавшийся фон',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+
        '\n\nУточнениеистории: '+fields['cannot_claim']+
        '\n\nHOTAнеизмерялась,проценткстендуидоляизмеренногопотолканеопределены. '
        'Архитектурнаявеха:ксуществующемуотказудобавленпроверенныйвнутреннийвходфоновыхпрототипов;жёлтаяветвьснепройденнымкритерием. Однолистовуюсхемуобновить,научныеисточник инепересчитывать.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(RUN/'report.json'),all_owned_ended=True,independent_counts_and_intervals=True,primary_pass=r['primary_pass']))
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'Новыйвнутреннийпрототип z13_observed_contrast_v1 завершёнexit0/деревоокончено. '
        'Источникruns/z13_observed_contrast_20261025/report.json. Цельминуссреднеесходствособразцаминаблюдавшихсяфоновыхпиков,коэффициент1,NULL0,безобучения,тотжеtop1центр. '
        'D:1полезный/84,21неизвестный;C:2вредных/84;R1:тольконейтральные72известных/84. '
        'ТриосновныхнижнихДИ=0,primary_pass=False;непродвигатькHOTAинеподбиратьвеса. '
        'Это родственникпрежнегоforeground,новаяконкретнаяреализациясотдельнымипикамивместообщегоконтекста;невыдаватьпринципзаневиданнуюновизну. '
        'Полныйитогпроверенпоисходнымстрокам,нужныкаталог/тесты/граф/публикацияиоднолистоваяархитектурапоэтойвехе. '
        'Всеисследовательскиепроцессыэтойстадиикончились. Разрешение26.10 07:40сохраняется;готовогоновогоисполнителяещёнет. '
        'Продолжатьможнотольконовымобоснованныммеханизмомсдешёвымгейтом,безповторапроваленныхверсий.\n\nНиже прежние передачи сохранены.\n\n')
    path=W/'HANDOFF.md';path.write_text(header+path.read_text(encoding='utf-8'),encoding='utf-8')


if __name__=='__main__':main()
