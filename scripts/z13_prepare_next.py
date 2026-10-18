"""Register bounded gates before they execute."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import RUN, read, write, sha


def main():
    if RUN.exists():
        raise RuntimeError('Already prepared')
    source = read(ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']
    assert not set(source['fit']) & set(source['predict'])
    paths = ['tracking/z13_target_center.py', 'tracking/z13_target_center_torch.py',
             'tracking/z13_utility.py', 'tracking/z13_readout.py', 'tracking/z13_checkpoint.py',
             'scripts/z13_parallel_next.py', 'scripts/z13_next_controller.py', 'scripts/z13_prepare_next.py',
             'scripts/z13_target_model.py', 'scripts/z13_train.py', 'scripts/z13_continue.py',
             'scripts/z13_score_swap.py', 'tests/test_z13_next_gates.py',
             'runs/z13_spatial_v2_20261013/D/full/identity.json', 'runs/z13_spatial_v2_20261013/D/full/model.pt']
    paths += [f'runs/z13_target_center_20261018/{a}/center_rows.json' for a in ('D','C','R1')]
    paths += [f'runs/z13_data/{v}.npz' for v in source['predict'] + source['fit'][:1]]
    now = datetime.now(timezone(timedelta(hours=3)))
    protocol = dict(created_at=now.isoformat(), sha256={p: sha(ROOT/p) for p in paths},
        authorization='User: launch needed runs in parallel; while away, justified long runs during 18.10.2026. These two gates: 600 CPU seconds each +10s termination.',
        deadline_scope='Longer follow-up must end within 18.10.2026 MSK; these gates stop independently after610s.',
        budget_seconds=600, arms=['D','C','R1'], device='cpu', dev=False, hota=False, heldout75=False,
        utility=dict(action='Learn a utility output on fixed saved centers; no response maps in this limited implementation.',
          estimator='StandardScaler + LogisticRegression C1 lbfgs max_iter500; no tuning. 26 leave-one-video-out folds; training excludes unknown outcomes.',
          inputs='Whitelisted center/pred/crop/pos plus current center peak z. GT error, GT scale, kind and utility are never features.',
          primary='At global top1%, both paired95%video-bootstrap lower bounds of D utility minus original and D utility minus geometry exceed0. Secondary2/5% cannot rescue failure.',
          matched_control='Same fitting rule and observable schema for D/C/R1; per-arm geometry-only control. No calibrated risk or stand-benefit verdict.',
          counterargument='A linear model on limited observables may fail despite useful information in full response maps; crossfit is exploratory on reused train26.'),
        differentiable=dict(action='Torch ridge implementation, no optimizer steps.',
          primary='NumPy parity<1e-10, numerical gradcheck passes, finite nonzero gradient to actual D parameters; weights and source files unchanged.',
          test='Random seed19 for numerical check; first natural positive example in first training video for real D gradient. No search annotation enters the readout; squared-output probe loss only.',
          limitation='A technical gradient is not improved localization, successful training or HOTA.'),
        literature=['https://arxiv.org/html/1904.07220v2','https://arxiv.org/html/1811.07628',
                    'https://arxiv.org/html/2006.04388','https://proceedings.mlr.press/v97/geifman19a.html'])
    RUN.mkdir(); write(RUN/'protocol.json', protocol)
    for kind in ('utility','differentiable'):
        (RUN/kind).mkdir(); write(RUN/kind/'status.json', dict(state='ready',stage='registered before run',done=0,total=0))
    catalog_path=ROOT/'experiments/project_history.json'; catalog=read(catalog_path)
    for kind, title, question, criteria in (
        ('utility','З13: оценка полезности по наблюдаемым признакам',
         'Поможет ли обучение точному событию полезности при фиксированных центрах D?',
         ['Разделение по видео и исключение истинных ошибок/рамок из входов проверены', protocol['utility']['primary']]),
        ('differentiable','З13: дифференцируемое считывание модели цели',
         'Можно ли передать градиент через нынешнее считывание до ядра D без изменения выхода?',
         [protocol['differentiable']['primary']])):
        cid='z13_next_'+kind+'_v1'
        catalog['experiments'].append(dict(id=cid,title=title,area='flyvis',state='ready',started=now.date().isoformat(),
          question=question,hypothesis=protocol[kind]['action'],expected=protocol[kind]['primary'],
          acceptance=[dict(criterion=c,met=None) for c in criteria],tags=['Z13','fly-problem-P8'],
          runs=[f'runs/z13_parallel_next_20261018/{kind}'],files=['scripts/z13_parallel_next.py'],
          evidence=[dict(kind='json',ref='runs/z13_parallel_next_20261018/protocol.json',claim='Условия до запуска')]))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; plan=path.read_text(encoding='utf-8'); anchor='## Журнал изменений\n'
    entry=f'''\n**{now:%d.%m.%Y, %H:%M} MSK** — постановка параллельных проверок FlyVis D

Автор разрешил необходимые параллельные прогоны и обоснованные длительные расчёты в течение 18.10.2026. До вычислений записан runs/z13_parallel_next_20261018/protocol.json. z13_next_utility_v1: фиксированные центры D/C/R1, логистическая оценка точного события полезности по наблюдаемой геометрии и высоте текущего пика; исключение целого видео при оценке, отдельный контроль без балла. Основной отбор1%, дополнительные2%/5%; два нижних парных ДИ против исходного балла и геометрии должны быть выше0. Это ограниченный вариант без полных карт ответа, не калибровка полного риска и не HOTA. z13_next_differentiable_v1: численное воспроизведение гребневого считывания и градиент до D без шага оптимизатора. Предел каждой CPU-проверки600с плюс10с завершения; независимый контроллер, отдельные консоли, без вызовов чата по расписанию. Последующее обучение требует готовой технической проверки и отдельного записанного протокола, но дополнительное разрешение автора на обоснованный расчёт в пределах дня уже получено. Отложенные75 не затрагиваются.
'''
    assert anchor in plan; path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    print('Registered two gates; sources frozen:',len(paths))


if __name__=='__main__':
    main()
