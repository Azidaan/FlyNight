"""Сводка измеренных прибавок с раздельными выборками и без диагностических потолков."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def main():
    from scripts.z13_continue import paired
    common='runs/z13_val20_matched_20261024/native_masa_recovery2/report.json';r=read(ROOT/common)
    specs=[('TrackTrack вместо XYWHByteTrack','tracktrack_raw','xywh_raw','Замена всего исходного трекера;безMASA'),
        ('TrackTrack при общей сшивке с MASA','tracktrack_masa','xywh_masa','ОдинаковаясшивкасMASAуобоих;не добавлять к предыдущейстроке'),
        ('Сшиватель с признаками MASA поверх XYWHByteTrack','xywh_masa','xywh_raw','Польза всей сшивки;отдельныйпризнакMASAпока неизолирован'),
        ('BoT-SORT вместо XYWHByteTrack','botsort_raw','xywh_raw','Замена исходного трекера;безMASA')]
    rows=[]
    for title,a,b,note in specs:
        value=r['paired'][a+'-'+b];base=r['pooled'][b]['HOTA'];actual=r['pooled'][a]['HOTA']
        rows.append(dict(title=title,n=20,delta=value['mean_per_video'],ci95=value['ci95'],baseline_hota=base,hota=actual,relative_pooled_percent=100*(actual/base-1),limitation=note,source=common,locator=['paired',a+'-'+b]))
    dev=read(ROOT/'runs/motion_scheme_search/dev.json');val=read(ROOT/'runs/motion_scheme_search/val7.json');ds=dev['_summary']['r5_all_dis']
    val_pair=paired({v:x['HOTA'] for v,x in val['r5_all_dis'].items()},{v:x['HOTA'] for v,x in val['r5_all_none'].items()})
    base=read(ROOT/'runs/video_rate/combined/summary.json')['metrics']['bytetrack']['HOTA'];xywh=read(ROOT/'runs/video_rate/xywh_validation7.json')['results']['h0p25_l0p1_b30_m0p8']['metrics']['HOTA']
    historical=[dict(title='XYWHByteTrack вместо ByteTrack',n=7,baseline_hota=base,hota=xywh,pooled_delta=xywh-base,ci95=None,
        limitation='Историческое семивидеосравнение;в исходной сводке нет парного интервала,надёжность не равна строкам20',sources=['runs/video_rate/combined/summary.json','runs/video_rate/xywh_validation7.json']),
        dict(title='Классическое движение DIS при5Гц',n=35,delta=ds['minus_none'],ci95=ds['ci95'],transfer_val7=val_pair,
        limitation='Результат наdev,который видел детектор. На7valинтервал включает0;нет подтверждения переносимости',sources=['runs/motion_scheme_search/dev.json','runs/motion_scheme_search/val7.json'])]
    small=[]
    for file,path,title in [('runs/b123_20261027/detection_supported_recovery/report.json',None,'Калман+поддержка обнаружением'),
        ('runs/recovery_research_20261027/fixed_composition/report.json',['paired','all40','appearance_numeric_vs_baseline'],'Фиксированные назначения+калибровка рамок')]:
        data=read(ROOT/file)
        if path is None:
            # В старом отчёте названия контрастов читаются из первичного JSON.
            candidates=data.get('paired',{});small.append(dict(title=title,source=file,paired=candidates));continue
        value=data
        for key in path:value=value[key]
        small.append(dict(title=title,source=file,paired=value))
    report=dict(created=now(),comparable20=rows,historical_or_development=historical,smaller_effects=small,
        scope='ФактическаяHOTA. БезGTпотолков/вспомогательныхметрик/разницысвреднымконтролем. Всеисторическииспользованныевыборки,не новыйнезависимыйтест.',
        source_sha256={p:sha(ROOT/p) for p in [common,*historical[0]['sources'],*historical[1]['sources']]})
    write(RUN/'hota_gains_evidence_table.json',report)
    f=lambda x:f'{x:.3f}'.replace('.',',')
    text='# Что давало измеренную прибавку HOTA\n\nСводка по сохранённым первичным файлам. «Серьёзный» не был критерием опытов: ниже показаны величина,интервал и ограничения. Все группы исторически использованы в проекте,не независимый итоговыйтест.\n\n'
    text+='## 1. Более заметные подтверждённые сравнения на одних20 видео\n\nПрибавка — средняя парная разность HOTA по видео;она может отличаться от разницы совокупных HOTA. Строки не складываются.\n\n'
    text+='| Изменение | Прибавка,п.п. | 95%интервал | Что это означает |\n|---|---:|---|---|\n'
    for x in rows:text+='| '+x['title']+' | +'+f(x['delta'])+' | ['+f(x['ci95'][0])+'; '+f(x['ci95'][1])+'] | '+x['limitation']+' |\n'
    text+='\n[Первичный общий отчёт]('+(ROOT/common).as_uri()+'). Выборка,детекции,оценщик одинаковы;алгоритмы не оптимально настроены,сшиватель обучался на парахXYWH и может переноситься неодинаково.\n\n'
    text+='## 2. Более крупные исторические сдвиги с существенными ограничениями\n\n'
    text+='| Изменение | Измеренный сдвиг | Ограничение |\n|---|---|---|\n'
    text+='| XYWHByteTrack вместоByteTrack | '+f(base)+' → '+f(xywh)+'; +'+f(xywh-base)+'п.п.совокупнойHOTA | Только7видео;парногоинтервала в этойсводкенет |\n'
    text+='| Классическое движениеDIS,5Гц | +'+f(ds['minus_none'])+'п.п.на35dev,ДИ['+f(ds['ci95'][0])+';'+f(ds['ci95'][1])+'] | Детекторвиделdev;на7valпереноснеподтверждён:ДИ['+f(val_pair['ci95'][0])+';'+f(val_pair['ci95'][1])+'] |\n'
    for p in [*historical[0]['sources'],*historical[1]['sources']]:text+='\n['+p+']('+(ROOT/p).as_uri()+').\n'
    text+='\n## 3. Меньшие последние эффекты\n\nКалман с поддержкой текущим обнаружением дал+0,388п.п.[0,095;0,704]на20разработки,но перенос на другие20 не подтверждён. Фиксированная композиция назначений и калибровки дала+0,260[0,172;0,355]на40. Это положительные ограниченные результаты,не большие скачки. Подробные источники—hota_gains_evidence_table.json.\n\n'
    text+='## 4. Следствие для текущего приоритета\n\nMASA—одна из заметных прибавок при сохранении XYWHByteTrack,но не единственная полезнаязамена. Разбираем вклад её признака в готовомсшивателе,затемБ1→Б2→Б3. НовыхпокадровыхMASAнепродолжаем. ПроцентысовокупнойHOTAксвоейбазе—relative_pooled_percent в JSON;доля максимума неизвестна.\n'
    target=ROOT/'dissertation/Проверенные прибавки HOTA — сравнения и ограничения.md';assert not target.exists();target.write_text(text,encoding='utf-8');print(target)

if __name__=='__main__':main()
