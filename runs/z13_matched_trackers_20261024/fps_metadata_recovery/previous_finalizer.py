"""Accept only the complete same-input benchmark, then record all comparisons."""
from pathlib import Path
import sys,json,csv,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_trackers_20261024 import RUN,VIDEOS,ARMS,read,write,sha,now,journal,IDENT
from scripts.z13_matched_pipeline_20261024 import STAGE,SYSTEMS
from scripts.z13_hota_deadline_guard import WindowsProcesses
ORIGINAL_STAGE=STAGE
RECOVERY=RUN/'gpu_recovery1'
if (RECOVERY/'protocol.json').exists():STAGE=RECOVERY


def ended(folder):
    if folder==ORIGINAL_STAGE and (RECOVERY/'protocol.json').exists():
        old=read(folder/'controller_result.json');p=read(RECOVERY/'protocol.json')
        assert old['exit_code']==1 and old['all_owned_ended'] and not old['remaining']
        assert sha(folder/'controller_result.json')==p['service_recovery']['old_result_sha256']
        folder=RECOVERY
    e=read(folder/'controller_result.json');c=read(folder/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['worker_ended'] and e['all_owned_ended'] and not e['hard_timeout'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'];assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())


def main():
    import numpy as np,torch
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not (RUN/'finish_recorded.json').exists();ended(RUN);ended(STAGE)
    protocol=read(RUN/'protocol.json');verify_sources(protocol);verify_sources(read(STAGE/'protocol.json'))
    raw=read(RUN/'raw_report.json');assert len(raw['xywh_exact_replay'])==7 and all(raw['xywh_exact_replay'].values())
    r=read(RUN/'report.json');assert r['all_conditions_common'] and r['videos']==list(VIDEOS) and len(r['pooled'])==12
    for n,h in r['output_sha256'].items():assert sha(ROOT/n)==h
    jobs=list((STAGE/'jobs').glob('*.json'));assert len(jobs)==49
    for path in jobs:
        m=read(path)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
    for arm in SYSTEMS:
        for v in VIDEOS:
            with np.load(RUN/arm/'signatures'/v/'signatures.npz') as z:
                assert int(z['frame_count'])==protocol['frames'][v] and np.isfinite(z['outputs']).all()
                if arm in ARMS:
                    with np.load(RUN/arm/'raw'/(v+'.npz')) as old:assert np.array_equal(old['outputs'],z['outputs'])
            with np.load(RUN/arm/'masa'/(v+'.npz')) as z:assert all(np.isfinite(z[k]).all() for k in ('first','last'))
        for kind in ('raw','masa'):assert sorted(r['per_video'][arm+'_'+kind])==sorted(VIDEOS)
    pv={k:{v:m['HOTA'] for v,m in values.items()} for k,values in r['per_video'].items()}
    for key,value in r['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    # The two deployed D versions have the same saved backbone, not two selected cores.
    legacy=torch.load(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/D/model.pt',map_location='cpu',weights_only=False)
    trained=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    assert sorted(legacy['backbone'])==sorted(trained['backbone']) and all(torch.equal(legacy['backbone'][k],trained['backbone'][k]) for k in legacy['backbone'])
    names={'bytetrack':'ByteTrack','botsort':'BoT-SORT','tracktrack':'TrackTrack','xywh':'Наш XYWHByteTrack','legacyD':'Наш XYWHByteTrack + D (З13 v2)','centerD':'Наш XYWHByteTrack + D (центрированный считыватель)'}
    order=('bytetrack','botsort','tracktrack','xywh','legacyD','centerD');summary={}
    for arm in order:
        value=r['pooled'][arm+'_masa']['HOTA'];base=r['pooled']['xywh_masa']['HOTA'];contrast=None if arm=='xywh' else r['paired'][arm+'_masa-xywh_masa']
        summary[arm]=dict(raw_hota=r['pooled'][arm+'_raw']['HOTA'],same_masa_hota=value,pooled_delta_from_xywh_masa=value-base,relative_percent_from_xywh_masa=100*(value-base)/base,paired=contrast)
    write(RUN/'verified_summary.json',dict(summary=summary,frame_count=sum(protocol['frames'].values()),videos=7,all49jobs_verified=True,raw_and_signature_replay=True,identical_D_backbone=True,all_owned_ended=True,report_sha256=sha(RUN/'report.json')))
    def number(x):return f'{x:.4f}'.replace('.',',')
    text='Все строки ниже измерены на одних семи val-видео BDD100K: каждый исходный кадр (около30кадров/с),одинаковые кэши YOLO26s960/conf0,01,один оценщик TrackEval и все размеченные кадры. Эти видео отсутствуют в обучающем списке детектора; val-набор использовался при его контроле и выборе весов. Это общий поисковый стенд,а не независимый итоговый тест.\n\n'
    text+='| Система | HOTA без сшивки | HOTA с одинаковой MASA |\n|---|---:|---:|\n'
    for arm in order:text+='| '+names[arm]+' | '+number(summary[arm]['raw_hota'])+' | '+number(summary[arm]['same_masa_hota'])+' |\n'
    text+='\nMASA одинакова во всех строках: зафиксированная голова,обученная ранее на130train,окно150кадров,расстояние5размероврамки,порог0,6. Она не перенастраивается под отдельный трекер. Нативные пороги уверенности0,25/0,1/0,25 и буфер30общие; внутренние стоимости сопоставления и компенсация движения камеры остаются штатными для каждого метода. Это не сравнение оптимально настроенных вариантов каждого алгоритма.\n\n'
    for arm in ('legacyD','centerD'):
        d=summary[arm]['paired'];text+=f"{names[arm]} относительно того же XYWHByteTrack с MASA: разность общих HOTA {summary[arm]['pooled_delta_from_xywh_masa']:+.6f} пункта ({summary[arm]['relative_percent_from_xywh_masa']:+.6f}% к базе); средняя разность по7видео {d['mean_per_video']:+.6f},95%интервал [{d['ci95'][0]:+.6f};{d['ci95'][1]:+.6f}],лучше на{d['better']}/7. "+('Положительная добавка в этом фиксированном сравнении подтверждена.' if d['ci95'][0]>0 else 'Положительная добавка в этом фиксированном сравнении не подтверждена.')+'\n\n'
    text+='Семь видео неоднократно использовались в исследовании; их мало для общего рейтинга и итогового вывода о переносе. Одна сохранённая D и один проход применения не доказывают независимость от любого сида. Перенос общей MASA-головы,обученной на жизнях XYWH,может различаться между трекерами; поэтому показаны обе колонки. Старые значения на200val и35dev в эту таблицу не включены. Сырые выходы каждого нативного метода воспроизведены независимым проходом при извлечении признаков. Полный источник: runs/z13_matched_trackers_20261024/report.json; проверки: verified_summary.json. Доля измеренного потолка не определена.\n'
    (RUN/'results.md').write_text(text,encoding='utf-8')
    with (RUN/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Система','HOTA без сшивки','HOTA одинаковая MASA','Сдвиг общей HOTA к XYWH MASA','Процент к XYWH MASA'])
        for a in order:writer.writerow([names[a],summary[a]['raw_hota'],summary[a]['same_masa_hota'],summary[a]['pooled_delta_from_xywh_masa'],summary[a]['relative_percent_from_xywh_masa']])
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source='runs/z13_matched_trackers_20261024/report.json'
    interpretation='Общее сравнение выполнено; прежний смешанный рейтинг заменён измерениями на одинаковых входах. Положительное состояние этой карточки означает полноту и воспроизведение сравнения,а не автоматический успех FlyVis. Добавка каждой D определяется только её парным интервалом против того же XYWH.'
    fields=dict(summary=interpretation,obtained=source+' → pooled/per_video/paired. Полная таблица и проценты: runs/z13_matched_trackers_20261024/verified_summary.json → summary: '+json.dumps(summary,ensure_ascii=False)+'.49GPU-заданий,28исходныхнативныхпрогонов,обеобщиепроцедуры на7видео завершены;сырые выходы и общий MASA-базис воспроизведены.',interpretation=interpretation,decision='Использовать только эту общую таблицу с явным n=7 и ограничениями. Не выбирать параметры по этим результатам и не открывать75отложенных; расширение набора оформлять отдельно. Эта оценка не меняет веса и не принимает новую архитектуру FlyVis.')
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=k,old=card[k],reason='Полное проверенное общее сравнение.'))
        card[k]=v
    card.update(state='completed_positive',done=f'{now():%Y-%m-%d}')
    for a in card['acceptance']:a.update(met=True,evidence=source+'; verified_summary.json; controller_result.json; gpu_stage/controller_result.json')
    card['metrics']=[dict(key='HOTA',value=r['pooled'][a+'_masa']['HOTA'],baseline=r['pooled']['xywh_masa']['HOTA'],subset='7 common val '+a+' sameMASA',source=source,locator='pooled.'+a+'_masa.HOTA') for a in order]
    card['confirmed_results']=[]
    for a in ('legacyD','centerD'):
        key=a+'_masa-xywh_masa'
        if r['paired'][key]['ci95'][0]>0:card['confirmed_results'].append(dict(id=IDENT+'_'+a,title=names[a]+' против одинакового стенда',category='tracker',metric='HOTA',unit='points',comparison=key,subset='7ранее использованных val,не входят в gradient-train детектора',conditions='Одни видео/исходныекадры/детекции/TrackEval/общая MASA; одна сохранённая D,безнастройкипо7val;парныйбутстрэп5000сид1.',limitations=protocol['limitations'],evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'] += [dict(kind='file',ref='runs/z13_matched_trackers_20261024/'+n,claim=c) for n,c in [('verified_summary.json','Полная проверка одинаковых условий и49заданий'),('results.md','Одна общая таблица и ограничения'),('raw_report.json','Нативные трекеры без сшивки')]];write(catpath,cat)
    journal('полный итог сравнения на одинаковых условиях','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+fields['decision']+'\n\nСводка к базе и проценты для каждого варианта: verified_summary.json/summary. Общие и средниеповидео сдвиги различаются; доля потолка неизвестна. Предыдущие числа35dev/200val сохранены как исторические,в новой таблице не смешиваются.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete_comparison=True,scientific_effect_not_implied_by_completion=True,report_sha256=sha(RUN/'report.json'),all_owned_ended=True))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
