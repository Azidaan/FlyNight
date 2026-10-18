"""Freeze paired aligned-core protocol, authorization and gates before GPU work."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_aligned_core_20261018'


def main():
    if RUN.exists():
        raise RuntimeError('Already prepared')
    local=timezone(timedelta(hours=3)); now=datetime.now(local)
    assert now.date().isoformat()=='2026-10-18'
    gate=read(ROOT/'runs/z13_parallel_next_20261018/differentiable/report.json')
    end=read(ROOT/'runs/z13_parallel_next_20261018/differentiable/controller_result.json')
    assert gate['primary_pass'] and gate['weights_unchanged'] and end['exit_code']==0 and end['worker_ended']
    source=read(ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']
    dev=read(ROOT/'runs/z13_target_center_dev_20261018/protocol.json')
    # Reuse the exact cached dev list rather than selecting new videos.
    dev_videos=dev.get('dev_videos',dev.get('videos'))
    assert isinstance(dev_videos,list) and len(dev_videos)==35
    assert not set(source['fit'])&set(source['predict'])
    assert not set(source['fit']+source['predict'])&set(dev_videos)
    parent=read(ROOT/'runs/z13_target_center_20261018/protocol.json')
    hashes=dict(parent['sha256'])
    paths=['scripts/z13_aligned_core.py','scripts/z13_aligned_controller.py','scripts/z13_prepare_aligned.py',
           'scripts/z13_record_next.py','tracking/z13_target_center_torch.py','tracking/z13_aligned_loss.py',
           'tests/test_z13_aligned_loss.py','tracking/gpu_lock.py',
           'runs/z13_parallel_next_20261018/differentiable/report.json',
           'runs/z13_parallel_next_20261018/differentiable/controller_result.json']
    paths += [f'runs/z13_data/{v}.npz' for v in source['fit']+source['predict']]
    paths += [f'runs/z13_target_center_20261018/{a}/center_rows.json' for a in ('D','C','R1')]
    paths += [f'runs/z13_target_center_dev_20261018/aggregate/{a}_center_rows.json' for a in ('D','C','R1')]
    paths += [f'runs/z13_target_center_dev_20261018/data/{v}.npz' for v in dev_videos]
    hashes.update({p:sha(ROOT/p) for p in paths})
    protocol=dict(id='z13_aligned_core_v1',created_at=now.isoformat(),sha256=hashes,
       authorization='Author explicitly authorized justified runs, including long runs, during 18.10.2026 while away.',
       save_stop_at=now.replace(hour=23,minute=50,second=0,microsecond=0).isoformat(),
       hard_stop_at=now.replace(hour=23,minute=59,second=30,microsecond=0).isoformat(),
       initial_model='runs/z13_spatial_v2_20261013/D/full/model.pt',fit=source['fit'],train26=source['predict'],dev35=dev_videos,
       action='Adapt only734 D core parameters through observed-exemplar ridge. No global readout parameters; all references fixed.',
       lambda_value=.1,feature_normalization='Same initial D stats from200 exemplar episodes of original fit20; hold fixed after adaptation.',
       training='Adam original core lrs and anchor coefficient1 around initial D; seed0,16 examples/update, microbatch1; known kinds0/1, all original fit104, no new data.',
       target='Exemplar target unchanged observed-center Gaussian. Query loss uses true query center for training only and Gaussian width from observed exemplar; absent query allzero.',
       loss='DiMP robust squared query loss: scores-target where target>0.05, relu(scores) otherwise. No annotation enters inference readout.',
       milestones=[250,1000],pilot_continuation='After fixed250 updates, continue only if mean pairedDadapted-Dinitial>0 and its upper95%CI>=0.02. Failure stops this run, not the research line.',
       primary='At1000steps: lower95%video CI Dadapted-Dinitial>0, mean>=2pp, and lower95%CI Dadapted-Ccenter>0. Other comparisons descriptive. No checkpoint selection by dev.',
       dev_gate='Only if full1000 train26 primary passes, evaluate the exact same frozen1000 checkpoint on cacheddev35 with no tuning. Reused dev is exploratory.',
       reproduction='Before updates: all initialD train26 rows paired, max center diff<0.5px, score diff<0.001; lambda unchanged. Do not weaken tolerances.',
       resources='One GPU under flyvis_fovea_gpu mutex, microbatch1, Idle, checkpoint every10updates and on stop, no downloads/videos/new caches.',
       hota=False,heldout75=False,chat_automation=False,
       limitations='Tests one complete new training regime, not a causal isolation of alignment alone. One seed, reused train26/dev. No HOTA. Utility gate failed, so tracker intervention is not authorized by these metrics alone.',
       literature=['https://arxiv.org/html/1904.07220v2','https://www.nature.com/articles/s41586-024-07939-3'])
    RUN.mkdir();write(RUN/'protocol.json',protocol);write(RUN/'status.json',dict(state='ready',stage='protocol recorded before GPU',done=0,total=1000))
    path=ROOT/'experiments/project_history.json';catalog=read(path)
    catalog['experiments'].append(dict(id=protocol['id'],title='З13: обучение D через модель цели с центрированной регрессией',
       area='flyvis',state='ready',started=now.date().isoformat(),question='Улучшает ли новый режим обучения D её центры относительно начальной D и штатной C?',
       hypothesis='Градиент через применяемую модель цели может сделать признаки D полезнее; причинность прежнего отставания не установлена.',
       expected='250 шагов с фиксированным условием продолжения до1000; только ядро734. НачальнаяD воспроизводится, считывание и нормировка фиксированы. Dev35 только после полного train-критерия; HOTA/75 не запускаются.',
       acceptance=[dict(criterion=c,met=None) for c in (protocol['reproduction'],protocol['primary'])],
       tags=['Z13','fly-problem-P2','fly-problem-P8'],runs=['runs/z13_aligned_core_20261018'],
       files=['scripts/z13_aligned_core.py','tracking/z13_target_center_torch.py','tracking/z13_aligned_loss.py'],
       evidence=[dict(kind='json',ref='runs/z13_aligned_core_20261018/protocol.json',claim='Постановка, бюджет и критерии до GPU'),
                 dict(kind='json',ref='runs/z13_parallel_next_20261018/differentiable/report.json',claim='Пройденная техническая проверка градиента без обновления весов')]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n'
    entry=f'''\n**{now:%d.%m.%Y, %H:%M} MSK** — постановка обучения D через нынешний считыватель

Автор разрешил обоснованные длительные расчёты до конца18.10.2026. z13_aligned_core_v1, runs/z13_aligned_core_20261018/protocol.json: исходнаяD с тем же центрированным гребневым считыванием λ0,1 и фиксированной исходной нормировкой; меняются только734параметра ядра. Обучение на прежних104видео: запрос с гауссовой целью по истинному центру только в функции обучения, ширина от наблюдаемого образца, квадратичная потеря с односторонним фоновым штрафом по DiMP§3.6; исходные скорости обучения и притяжение, пакет16 микропакет1. Это проверка всего нового режима, а не чистая причинная абляция согласования. До обновления — воспроизведение всех исходных D-строк с прежними допусками. После250шагов продолжать до1000, только если средняя разность к начальнойD положительна и верхнийДИ допускает2п.п. Полный критерий: нижниеДИ к начальнойD и C_center выше0, средняя кD не ниже2п.п. Только тогда фиксированная проверка сохранённой1000-версии на прежнемdev35. При неуспехе не подбирать иную контрольную точку поdev. ОдинGPU, независимое сохранение/остановка23:50, принудительное завершение своего дерева23:59:30MSK. HOTA и75отложенных не запускать: критерий ограниченного оценщика полезности не пройден. Новых загрузок и обработки видео нет.
'''
    assert anchor in plan;path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    print('Aligned-core protocol frozen; source hashes:',len(hashes))


if __name__=='__main__':
    main()
