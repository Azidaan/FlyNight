"""Record a verified geometry diagnostic; do not register a tracker improvement."""
import csv
from datetime import datetime
import gzip
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import flyvis_pair_support_overlap_20261107 as core
RUN=core.RUN/'recovery1'

def main():
    r=core.read(RUN/'report.json');audit=core.read(RUN/'audit.json');ctl=core.read(RUN/'controller.json')
    assert audit['passed'] and audit['report_sha256']==core.sha(RUN/'report.json') and ctl['reason'] is None and ctl['returncode']==0
    s=r['totals']['evaluate26'];q=s['queries_boxes'];a=s['pairs']['exactly_two'];all_=r['totals']['all']['queries_boxes']
    n=sum(a.values());assert n==q['exactly_two']
    with gzip.open(RUN/'pairs.csv.gz','rt',encoding='ascii') as f:
        ratios=[int(z['shared'])/(int(z['na'])+int(z['nb'])-int(z['shared'])) for z in csv.DictReader(f)
            if z['split']=='evaluate26' and z['exactly_two_admitted']=='1' and z['category'].startswith('partial')]
    assert len(ratios)==a['partial_contained']+a['partial_both_unique']
    primary=(f"На 26 видео при ровно двух реальных кандидатах ({n} запросов): раздельные наборы — {a['disjoint']}, общие ячейки и свои у каждой рамки — {a['partial_both_unique']}, один набор строго внутри другого — {a['partial_contained']}, полное совпадение — {a['identical']}.")
    empty=(f"Перед требованием непустой рамки пустых {q['pre_empty']} из {q['pre_boxes']} предъявлений с допустимым центром; если включить все центры внутри квадрата до отсечения по близости к сетке — {q['square_empty']} из {q['square_boxes']}. Это не уникальные объекты.")
    interp="Полное совпадение редко; большинство пар имеет собственные ячейки у обеих рамок. Геометрия не доказывает различие активности и не идентифицирует объект."
    decision="Сохранить результаты; допуск прежней четырёхзонной схемы не пересматривать. Обучение и проверка полного трекера не запускались."
    scope="Локальное окно глаза вокруг прогноза, одна сетка и момент на пару; только реальные детекции одного класса. Отказ и отсутствие назначения не являются рамками. Повторно использованные видео TRAIN; наборы колонок не измеряют HOTA или пользу FlyVis."
    times=f"Вычисление с проверками: {r['seconds']:.2f} с, пик физической памяти рабочего процесса {r['process_peak_working_set_bytes']/2**20:.1f} МиБ. GPU не использовался."
    ratios_text=f"В {len(ratios)} частично пересекающихся парах медианная доля общих ячеек от их объединения — {100*statistics.median(ratios):.2f}%. Это дополнительное описание, не критерий допуска."
    ev=s['pairs']['beneficial_vs_incorrect']; total_ev=sum(ev.values())
    diag=f"Среди {total_ev} пар с ранее известным полезным действием и неверным конкурентом: {ev['disjoint']} без общих ячеек, {ev['partial_both_unique']+ev['partial_contained']} с частичным пересечением, {ev['identical']} одинаковых. Это пары, не уникальные события смены ID."
    record=RUN/'РЕЗУЛЬТАТ.md'
    report=f"# Какие ячейки FlyVis видят две конкурирующие рамки\n\n{primary}\n\n{ratios_text}\n\n{empty}\n\n{diag}\n\n{scope}\n\n{interp} {decision}\n\nПо трём сохранённым выборкам проверены {all_['queries']} запросов на 130 видео; результаты по выборкам и каждому видео раздельны в report.json и by_video.json. В audit.json — сверка {audit['input_hashes_verified']} исходных файлов и {audit['unique_pair_rows']} строк пар. {times}\n\nАрхив включает protocol.json, report.json, audit.json, by_video.json, input_sha256.json и pairs.csv.gz. Первая попытка остановлена до завершения первого видео: порядок строк в журналах отличался. Исправлено сопоставлением по UID; все проверки исходных кандидатов и геометрии сохранены.\n"
    with record.open('x',encoding='utf-8') as f:f.write(report)
    cat=core.read(core.CAT);c=next(c for c in cat['experiments'] if c['id']==core.IDENT);assert c['state']=='ready'
    c.update(state='infrastructure',done=datetime.now().strftime('%Y-%m-%d'),summary=primary,obtained=primary+' '+empty+' Источник: '+(RUN/'report.json').as_posix()+' (totals.evaluate26).',interpretation=interp,decision=decision,cannot_claim=scope)
    c['acceptance'][0].update(met=True,evidence=(RUN/'audit.json').as_posix()+'; '+(RUN/'report.json').as_posix())
    for name,claim in [('protocol.json','Исполнение с соединением по UID'),('report.json','Полные номера выборок и счётчики'),('audit.json','Сверка источников и независимая классификация всех строк'),('РЕЗУЛЬТАТ.md','Вывод и ограничения')]:
        c['evidence'].append(dict(kind='file',ref=(RUN/name).as_posix(),claim=claim))
    core.dump(core.CAT,cat)
    core.journal('результат геометрии конкурирующих рамок (`'+core.IDENT+'`). *Получено.* '+primary+' '+empty+' '+times+' *Вывод.* '+interp+' *Решение.* '+decision+' Служебный отказ порядка строк сохранён; исправление по UID. Источник: '+record.as_posix()+'.')
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:
        f.write('\n## 07.11.2026 — общие колонки глаза у двух конкурирующих рамок\n\n'+primary+' '+empty+' '+interp+' '+decision+' Источник: '+record.as_posix()+'\n')

if __name__=='__main__':main()
