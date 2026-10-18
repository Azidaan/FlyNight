"""Freeze dev validation conditions and audit all local inputs before launch."""
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_target_center_dev import RUN,TRAIN,read,sha,write


def main():
    if RUN.exists(): raise RuntimeError('Dev run already prepared; inspect instead of overwriting')
    previous=read(TRAIN/'protocol.json');result=read(TRAIN/'report.json');assert result['primary_pass']
    selected=read(ROOT/'runs/train_cache/selection.json')['videos'];videos=selected['dev']
    assert len(videos)==35 and len(set(videos))==35
    assert not set(videos)&set(previous['fit']+previous['evaluate'])
    model='runs/z13_spatial_v2_20261013/D/full/model.pt'
    sources=set(previous['sha256'])|{'scripts/z13_target_center_dev.py','scripts/z13_target_center_dev_controller.py',
        'scripts/z13_prepare_target_center_dev.py','scripts/z13_check_t1.py','scripts/z13_data.py',
        'scripts/z13_spatial_train.py','tracking/z13_spatial.py','tracking/z13_checkpoint.py',
        'runs/train_cache/selection.json','runs/z13_target_center_20261018/report.json',
        'tracking/retina.py','tracking/foveated_eyes.py','scripts/fovea_target_30hz.py',
        'scripts/freeze_bench.py','scripts/run_bdd_video_rate_experiment.py',
        'runs/flyvis/results/flow/0000/000/best_chkpt'}
    for v in videos:
        sources.update([f'runs/fly_signatures/{v}/signatures.npz',f'runs/train_cache/{v}/alignment.json',
            f'runs/train_cache/{v}/detections/raw_video.npz',f'runs/train_cache/{v}/motion/flyvis_tiled.json',
            f'data/bdd100k/labels/box_track_20/train/{v}.json',f'data/bdd100k/videos/train/{v}.mov'])
    missing=[name for name in sources if not (ROOT/name).is_file()]
    assert not missing,missing
    for name,digest in previous['sha256'].items(): assert sha(ROOT/name)==digest,name
    now=datetime.now(timezone(timedelta(hours=3)))
    protocol=dict(id='z13_target_center_dev_v1',created_at=now.isoformat(),
        authorization='Author replied Давай to the proposed fixed dev35 validation; no new core training.',
        videos=videos,fit=previous['fit'],technical_train_video=previous['evaluate'][0],model=model,
        choices=result['choices'],device='cpu',budget_seconds=14400,
        budget_note='Conservative execution ceiling of four CPU hours including extraction, not a GPU authorization; preserve partial work on deadline.',
        extraction='Existing z13_data.extract unchanged: same baseline detections/tracker/retina/crops/labels; dedicated new data directory. Full T1 label/geometry validation on all35. No downloads.',
        fitting='Only per-episode observed exemplar regression; frozen lambdas from train20; normalization from same train20. No dev label enters prediction or parameter selection.',
        primary=['D_center-D_prior','D_center-D_binary'],
        comparisons=[['D_center','D_prior'],['D_center','D_binary'],['C_center','C_binary'],['R1_center','R1_binary'],['D_center','R1_center'],['C_center','R1_center'],['D_center','C_center']],
        acceptance='Both primary paired-video 95% CI lower bounds >0; FlyVis contribution separately D_center-R1_center lower>0. Same bootstrap5000 seed1. No dev retuning after outcome.',
        counterargument='Search result may not transfer beyond reused train26; response-peak confidence already ranks interventions poorly.',
        decision_value='Confirm or reject transfer of frozen localization before any new reliability model or tracker intervention.',
        cannot_claim='dev35 has been used earlier in this project; not blind heldout. No HOTA, detector changes, GPU, core training, heldout75 or downloads.',
        sha256={name:sha(ROOT/name) for name in sorted(sources)})
    RUN.mkdir();write(RUN/'protocol.json',protocol)
    write(RUN/'readiness.json',dict(local_inputs_ready=True,dev_videos=35,existing_dev_episodes=0,source_hashes=len(sources),train_dev_disjoint=True))
    write(RUN/'status.json',dict(state='ready',stage='Fixed dev35 protocol; awaiting controller',done=0,total=35))
    catpath=ROOT/'experiments/project_history.json';catalog=read(catpath)
    assert not any(c['id']==protocol['id'] for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=protocol['id'],title='З13: перенос центрированного считывателя на dev35',area='flyvis',state='ready',started=now.date().isoformat(),
        question='Сохраняется ли выигрыш замороженной центрированной цели считывателя на dev35?',
        hypothesis='Одна и та же цель и параметры, выбранные на train20, улучшат локализацию D и на dev35; это проверка переносимости, не слепое подтверждение.',
        expected='D prior против D binary/center; C binary/center и R1 binary/center на одинаковых35видео. Нормировка только на прежнем train20, λ из завершённого опыта. Данные извлекаются прежним кодом из локальных видео и кэшей. Бюджет исполнения ограничен четырьмя часами CPU; по сроку сохраняются полные видео. Протокол runs/z13_target_center_dev_20261018/protocol.json.',
        acceptance=[dict(criterion='Полнота35видео, исходные метки/геометрия и воспроизведение сохранённого train-контроля',met=None),
            dict(criterion='D_center-D_prior и D_center-D_binary: обе нижние границы парного повидео95%ДИ выше нуля',met=None),
            dict(criterion='D_center-R1_center: нижняя граница95%ДИ выше нуля',met=None)],
        tags=['Z13','fly-problem-P8'],runs=['runs/z13_target_center_dev_20261018'],files=['scripts/z13_target_center_dev.py','scripts/z13_target_center_dev_controller.py'],
        evidence=[dict(kind='json',ref='runs/z13_target_center_dev_20261018/protocol.json',claim='Условия до просмотра dev-результатов')]))
    catpath.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');marker='## Журнал изменений\n';assert marker in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — постановка `z13_target_center_dev_v1`\n\n'
    entry+='*Постановка.* Автор согласовал предложенную проверку замороженной версии на dev35. Готовых эпизодов dev нет; локальные35видео, детекции, сигнатуры, исходная разметка и выравнивание найдены. Подготовлено выделенное извлечение прежним кодом и полная проверка меток/координат. Протокол runs/z13_target_center_dev_20261018/protocol.json: λ прежние, нормировка только train20; D prior/binary/center, C binary/center, R1 binary/center. Два основных повидеоДИ должны быть выше нуля, контроль без мухи оценивается отдельно. Это повторно используемый dev, не слепой тест. Служебный предел — четыре часа CPU плюс30с завершения; новых GPU, обучения ядра, HOTA, отложенных75 и загрузок нет. Автоматизация чата остаётся выключенной. Полного dev-результата пока нет.\n'
    path.write_text(plan.replace(marker,marker+entry,1),encoding='utf-8')
    print(json.dumps(dict(run=str(RUN),sources=len(sources),videos=35,budget_seconds=14400)))


if __name__=='__main__':main()
