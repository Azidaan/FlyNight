"""Record direct authorization and freeze one development gate before CUDA."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'memory_write_preparation';folder=RUN/'memory_write_dev'
    auth=service.read(RUN/'memory_write_authorization.json')
    assert auth['human_message']=='Да, проверить правило записи памяти'
    assert service.read(RUN/'masa_agent_review/verified.json')['passed']
    old=service.read(RUN/'masa_query_fp32_recovery1/protocol.json')
    training=service.read(ROOT/'runs/continuation_20261028/b2_sparse_training_pilot/protocol.json')
    entry=next(x for x in training['entries'] if x['split']=='fit20')
    manifest=service.read(ROOT/'runs/continuation_20261028/b2_causal_manifest/videos'/(entry['video']+'.json'))
    video=entry['video'];source=dict(video=video,video_path=manifest['video_path'],alignment=manifest['alignment'],
        detections=f'runs/train_cache/{video}/detections/raw_video.npz')
    files=['tracking/query_memory_write.py','tracking/query_appearance_comparison.py','tracking/lazy_appearance_association.py',
        'tracking/frame_masa_appearance.py','scripts/memory_write_dev_20261029.py','scripts/audit_memory_write_dev_20261029.py',
        'tracking/masa_detic_prepared.py','tracking/masa_detic_bdd_input.py','tracking/masa_r50.py',
        'weights/masa/masa_r50.pth','weights/masa_detic_author_20261029/detic_masa.pth',
        'runs/continuation_20261029/memory_write_authorization.json',*list(source.values())[1:]]
    p=dict(id='query_memory_write_v1',created=service.now(),budget_seconds=86400,source=source,
        config=old['config'],appearance_weight=old['appearance_weight'],beta=.9,
        policies={'last':'Replace by last actually assigned query observation.',
            'mean':'Fixed exponential moving average: normalize(.9*previous+.1*current).',
            'cautious':'Same average with alpha=.9+.1*(1-clip((score-track_high_thresh)/(1-track_high_thresh),0,1)).'},
        official_mechanism_source='https://raw.githubusercontent.com/GerardMaggiolino/Deep-OC-SORT/main/trackers/integrated_ocsort_embedding/ocsort.py',
        scope='Only write policy changes; separate matched strata R501333 and Detic1333. Component adaptation, not complete Deep OC-SORT or FlyVis core.',
        strongest_counterargument='A confidently detected neighbour can still overwrite the wrong identity. Slower update can preserve an initial mistake or stale view. Confidence is not identity truth.',
        gate='One complete first fit20 development video selected by existing fixed order, all8trackers; independent exact CPU raw/JSON/cost/memory/metric replay and zero-state parity. No quality-based tuning/variant selection. This gate is implementation readiness, not evidence of benefit.',
        subsequent_val='Before val freeze a separate full tracker protocol. Reuse the already completed20 last/native outputs with SHA; reuse exact existing normalized vectors where present and compute only missing dynamic query ROIs for new policies. Never feed new vectors to old trained heads. All new outputs audited, feature checks1e-5. No tuning on historical val.',
        prospective_primary='Four memory comparisons: cautious-last and cautious-mean separately for R501333 and Detic1333. Publish ordinary95%paired20k bootstrap seed1 and Bonferroni98.75% intervals; main positive only if all four corrected lower bounds>0. Native reference, all HOTA/AssA/IDF1/FP/FN/IDSw. Historical20val not new independent test.',
        time='Stage time gates cancelled; overall save30.10 07:30/ownedstop07:39:30/hardstop07:40 unchanged. Save first full video cost, no guaranteed chain fit.',
        no_training=True,gt_in_predictor=False,detector_has_seen_development_train=True,
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    service.write(RUN/'memory_write_dev_audit/protocol.json',dict(budget_seconds=86400,parent_sha=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');assert not any(x['id']==p['id'] for x in h['experiments'])
    original=next(x for x in h['experiments'] if x['id']=='reid_masa_detic_v1')
    h['experiments'].append(dict(id=p['id'],title='Запись памяти объекта — последнее наблюдение, усреднение и обновление по уверенности',area=original['area'],state='ready',started='2026-10-29',
        question='Достаточно ли изменить запись памяти при прежних признаках и сопоставлении, чтобы улучшить полный трекер?',
        hypothesis='Более осторожное обновление может уменьшить вред неудачных записей; причина загрязнения памяти пока не установлена.',
        expected='Причинная реализация и независимая проверка dev; затем отдельно замороженная полная проверка четырёх сравнений памяти.',
        acceptance=[dict(criterion='Полное видео dev, нулевой контроль и независимый численный повтор всех выходов/памяти/метрик',met=None),dict(criterion=p['prospective_primary'],met=None)],
        evidence=[dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Условия и контроли до нового dev/val; прямое разрешение автора')],
        cannot_claim='Подготовка и dev не доказывают пользу трекеру. Уверенность детектора не метка личности. Нет нового обучения и изменения ядра FlyVis. Новая научная серия val ещё не запущена.'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Прямое разрешение на отдельную проверку записи памяти; протокол до dev',
        'Автор: «Да, проверить правило записи памяти». Последняя запись / экспоненциальное усреднение beta0,9 / та же формула с доверием к score из официального DeepOC-SORT; это адаптация компонента, не весь метод. Одинаковые признаки двух извлекателей при размере1333, прежние YOLO/query/геометрия/вес0,1. Уверенный сосед остаётся сильным возражением. Первый dev в фиксированном fit20 порядке '+video+' целиком,8вариантов,безGTвпредикторе;полныйотдельныйCPUаудит. Детектор видел train,непроверкаегообобщения. Источник runs/continuation_20261029/memory_write_dev/protocol.json. Следующий val требует отдельной заморозки; готовые last/native/векторы поSHAнеповторяются. Общий срок30.10 07:40МСК.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_memory_write.py','tests/test_query_appearance_comparison.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(passed=True,ready=True,updated=service.now()))
    for index,name,script in [('090','memory_write_dev','scripts/memory_write_dev_20261029.py'),('095','memory_write_dev_audit','scripts/audit_memory_write_dev_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=str((RUN/name).relative_to(ROOT)),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
