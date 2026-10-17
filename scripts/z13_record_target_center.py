"""Protocol and project records for one authorized target-readout correction."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_target_center_20261018'
ID='z13_target_center_v1'

def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--finish',action='store_true');a=ap.parse_args()
    catpath=ROOT/'experiments/project_history.json';cat=json.loads(catpath.read_text(encoding='utf-8'))
    if not a.finish:
        if any(c['id']==ID for c in cat['experiments']):raise RuntimeError('Already prepared')
        RUN.mkdir(exist_ok=False)
        job=json.loads((ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json').read_text())['config']['job']
        paths=['scripts/z13_target_center_probe.py','scripts/z13_target_center_controller.py','tracking/z13_target_center.py','scripts/z13_target_model.py','scripts/z13_train.py','tracking/z13_readout.py',
               'runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_spatial_v2_20261013/D/full/identity.json','runs/z13_continue_20261017/baseline/rows.json']
        for arm in ('D','C','R1'):
            base='runs/z13_target_model_r1_20261017' if arm=='R1' else 'runs/z13_target_model_20261017'
            paths.extend([f'{base}/{arm}/rows.json',f'{base}/{arm}/choice.json'])
        paths.extend(f'runs/z13_data/{v}.npz' for v in job['fit'][:20]+job['predict'])
        protocol=dict(id=ID,date='2026-10-18',arms=['D','C','R1'],device='cpu',
          intervention='Only regression target: uniform binary observed mask -> Gaussian at observed-mask bounding center, sigma=max((occupied extent+spacing)/4,spacing/2). No search GT or Kalman forecast in target.',
          source='https://arxiv.org/html/1904.07220 sections 3.1 and 3.6; inspired target shape, not full DiMP reproduction',
          fit=job['fit'][:20],evaluate=job['predict'],lambdas=[.001,.01,.1,1.,10.],
          selection='Each arm/mode independently selects lambda on the same fixed fit20; no selection on train26; same feature normalization and features within paired modes.',
          baseline='Recompute binary control on exactly same features and check saved prior outputs before interpreting center target.',
          primary='D_center vs prior readout D and vs binary online D: both video-paired 95% CI lower bounds >0. No arbitrary 2pp detection limit. Exploratory advancement only, not a deployment criterion.',
          secondary='C_center vs C_binary, R1_center vs R1_binary; D/C_center vs R1_center; known nonuseful and unknown separately at fixed 1/2/5% rank coverage. Peak z-score is not a calibrated probability; HOTA not measured.',
          counterargument='Per-column features may not encode relative object center; forcing a center peak may discard useful object appearance. This test cannot prove cause.',
          budget='CPU timing 180s + 30s hard grace; full probe only if estimate fits 20min CPU, otherwise request budget. Full hard grace 30s. No GPU/core training/dev/heldout75.',
          sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        dump(RUN/'protocol.json',protocol)
        cat['experiments'].append(dict(id=ID,title='З13: пик центра вместо сплошной рамки в модели цели',area='flyvis',state='planned',started='2026-10-18',
          question='Улучшит ли цель регрессии с максимумом в центре наблюдаемой рамки считывание на той же D?',
          hypothesis='Проверяемая версия: сплошная маска не различает центр и края; центрированная цель улучшит локализацию. Причина прежних ошибок не установлена.',
          expected='Одна замена цели в гребневой регрессии. Парные binary/center на одинаковых признаках D, C и входа глаза R1. Одна сетка λ, выбор только на прежних 20 train104; оценка train26. Основные сравнения с исходным считывателем D и binary D, нижняя граница повидео ДИ > 0; это поисковый критерий продвижения. Порог 2 п.п. не считается разрешающей способностью. Протокол runs/z13_target_center_20261018/protocol.json. Контраргумент: линейный признак колонки может не кодировать положение относительно центра цели.',
          acceptance=[dict(criterion='Воспроизведение binary контроля и равенство строк',met=None),dict(criterion='D_center лучше прежнего D и binary D: оба повидео ДИ выше нуля',met=None)],
          tags=['Z13','fly-problem-P8'],runs=['runs/z13_target_center_20261018'],files=['tracking/z13_target_center.py','scripts/z13_target_center_probe.py'],
          evidence=[dict(kind='json',ref='runs/z13_target_center_20261018/protocol.json',claim='Зафиксированные условия до прогона')]))
        entry='*Постановка.* Согласованное действие — доработка модели цели. После диагностики расхождений выбрана одна замена: максимум регрессионной цели у центра последней наблюдаемой рамки вместо одинаковой единицы внутри неё (DiMP, https://arxiv.org/html/1904.07220, §3.1, §3.6). Это гипотеза, не установленная причина. Код tracking/z13_target_center.py принимает только признаки и наблюдаемую маску. Парные варианты на D/C/R1; λ выбирается на прежних 20 видео train104, оценка train26; исходный binary воспроизводится. Протокол runs/z13_target_center_20261018/protocol.json. Сначала ограниченный CPU-замер; полный CPU-пилот только в пределах 20 минут. Нет GPU, нового обучения ядра, dev, HOTA или отложенных 75.'
    else:
        if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
        card=next(c for c in cat['experiments'] if c['id']==ID)
        if (RUN/'report.json').exists():
            rep=json.loads((RUN/'report.json').read_text());d=rep['paired']['D_center-D_prior'];b=rep['paired']['D_center-D_binary']
            result=f"runs/z13_target_center_20261018/report.json: D_center−прежний D {d['mean_per_video']*100:.3f} п.п., ДИ {[round(x*100,3) for x in d['ci95']]}; D_center−binary D {b['mean_per_video']*100:.3f} п.п., ДИ {[round(x*100,3) for x in b['ci95']]} (paired)."
            card.update(state='completed_positive' if rep['primary_pass'] else 'completed_negative',done='2026-10-18',obtained=result,
              interpretation='Критерий продвижения '+('выполнен' if rep['primary_pass'] else 'не выполнен')+'. Поисковый результат на повторно использованном train26.',
              decision=rep['decision'],cannot_claim='Нет HOTA, независимой валидации, доказательства причины ошибки или роли анатомии. Балл пика не калиброван как вероятность присутствия.')
            for i,x in enumerate(card['acceptance']):x.update(met=True if i==0 else rep['primary_pass'],evidence='runs/z13_target_center_20261018/report.json; <arm>/reproduction.json')
            card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_20261018/report.json',claim='Результат парного изменения цели'))
            entry='*Получено (`'+ID+'`).* '+result+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'. HOTA не измерялась.'
        else:
            benchmark=json.loads((RUN/'benchmark.json').read_text()) if (RUN/'benchmark.json').exists() else None
            measured=(f"На 16 эпизодах две цели на CPU: {benchmark['timing']['seconds']:.3f} с, {benchmark['timing']['frames']} кадров интегрирования; оценка трёх вариантов {benchmark['rough_total_seconds']/60:.1f} мин (runs/z13_target_center_20261018/benchmark.json). " if benchmark else '')
            card.update(state='ready',obtained=('Подготовлены код и CPU-тесты; полный результат отсутствует. '+measured+'Первый запуск: ошибка CPU/CUDA, код 1; повторный служебный запуск с явным CPU: код 0, worker_ended=true (benchmark_controller_result.json, recovery_benchmark_result.json в runs/z13_target_center_20261018).') if benchmark else 'Подготовлены код и CPU-тесты; замер не завершён, см. журналы контроллера.',
                        decision='Полный прогон требует согласования бюджета; не выдавать подготовку за измеренный эффект.')
            card['evidence'].append(dict(kind='file',ref='reports/z13_readout_change_20261018.md',claim='Механизм, ограничения и исправление CPU-запуска'))
            if benchmark:card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_20261018/benchmark.json',claim='Измеренная стоимость двух пакетов; оценка полной проверки предварительная'))
            card['files']+=['scripts/z13_target_center_cpu_recovery.py','scripts/z13_target_center_recovery_controller.py']
            entry='*Подготовлено (`'+ID+'`).* '+card['obtained']+'\n\n*Решение.* Полного сравнения нет: оценка превышает выделенные 20 минут. Требуется согласование CPU-бюджета, предлагается до 3 часов с независимой остановкой. Измерение времени не является эффектом правки. HOTA не измерялась.'
        dump(RUN/'finish_recorded.json',{'recorded':True})
    dump(catpath,cat)
    planpath=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=planpath.read_text(encoding='utf-8');marker='## Журнал изменений\n';assert marker in plan
    stamp=datetime.now().strftime('%H:%M')
    planpath.write_text(plan.replace(marker,marker+f'\n**18.10.2026, {stamp} MSK** — {ID}\n\n'+entry+'\n',1),encoding='utf-8')

if __name__=='__main__':main()
