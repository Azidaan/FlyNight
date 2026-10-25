"""Fresh bounded S0/S1a: frozen models, fit20 cross-fitted policy comparison."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys
import subprocess
from datetime import datetime, timedelta
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_continue import paired
from scripts.z13_localization_labels_20261023 import verify_sources
from tracking.z13_margin_rule import crossfit, MARGIN_GRID, PROBABILITY_GRID
RUN = ROOT/'runs/z13_margin_rule_audit_20261025'
OLD = ROOT/'runs/z13_box_supervision_followup_20261025'
IDENT = 'z13_margin_rule_audit_v1'
PARENT = 'z13_default_class_readout_v1'


def journal(title, body):
    from scripts import z13_candidate_reserve_20261021 as m
    previous = m.IDENT
    try:
        m.IDENT = IDENT; m.journal(title, body)
    finally:
        m.IDENT = previous


def revise(card, key, value, reason):
    if card.get(key) == value: return
    card.setdefault('revisions', []).append(dict(date=now().strftime('%Y-%m-%d'),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        field=key, old=card.get(key, ''), reason=reason))
    card[key] = value


def prepare():
    assert not RUN.exists()
    old = read(OLD/'protocol.json'); verify_sources(old)
    result = read(OLD/'controller_result.json'); assert result['all_owned_ended'] and result['exit_code'] == 0
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps = WindowsProcesses(); assert all(ps.creation(int(pid)) != b for pid,b in result['owned'].items())
    names = ['tracking/z13_margin_rule.py', 'scripts/z13_margin_rule_audit_20261025.py',
             'tests/test_z13_margin_rule.py', 'scripts/z13_bounded_recovery_controller_20261022.py']
    names += [f.relative_to(ROOT).as_posix() for f in (OLD/'training').glob('*/seed*/*')
              if f.name in ('model.pt','report.json','rows.json','done.json')]
    names += [(OLD/'cache/report.json').relative_to(ROOT).as_posix()]
    start = now(); stop = min(start+timedelta(seconds=1200), datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    p = dict(id=IDENT, parent=PARENT, created=start.isoformat(), device='cpu', timezone='Europe/Moscow',
        budget_seconds=1200, save_at=stop.isoformat(), hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(), fit20=old['fit20'], evaluate26=old['evaluate26'],
        folds={v:i%5 for i,v in enumerate(sorted(old['fit20']))}, seeds=[0,1,2],
        probability_grid=PROBABILITY_GRID, margin_grid=MARGIN_GRID, epsilon=0,
        action='Same frozen score vectors and full-box action. Only probability threshold versus direct positive advantage threshold changes.',
        s0='Exact saved predictions/rows, weights and labels of all24 models; train26 only reproduction, no new policy evaluation or selection there.',
        s1='Models trained on84; five fixed video folds within separate fit20. Calibrate each policy on16videos, evaluate4. Same seeds/arms/context. No best-seed selection. Both threshold grids11values fixed before calculation. No AUC gate.',
        criterion='Margin-only advancement requires mean D_context net>=.02 and paired-video95%lower>0 against zero, old_probability D, C, R1, geometry_context. Otherwise do not promote margin-only to26/HOTA. Failure does not test the new loss or prove absent information.',
        limitation='Repeated development fit20; overlapping calibration folds, bootstrap is descriptive conditional on this partition, not an independent certification. New fit104 learning requires its own preregistered budget. Unknown outcomes excluded from net but counts retained.',
        sha256={**old['sha256'], **{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p); write(RUN/'status.json',dict(state='ready',updated=now().isoformat(),stage='С0 затем С1а; без обучения'))
    cat=read(ROOT/'experiments/project_history.json'); parent=next(x for x in cat['experiments'] if x['id']==PARENT)
    write(RUN/'parent_before.json',parent)
    reason='До запуска; автор после разбора разрешил пробовать дальнейший план. AUC0,93 условная бинормальная оценка, не универсальная граница; зонд не доказывает потолок информации.'
    revise(parent,'hypothesis','Рабочая гипотеза: обучение явному решению по умолчанию и последствиям альтернатив может улучшить использование сохранённых признаков. MSE к good−harm и нулевой балл исходного решения сами по себе допустимы; нормированный порог зависит от других кандидатов. Причина прежнего провала и польза замены не доказаны.',reason)
    revise(parent,'summary','Подготовка ограниченного опыта считывания. С0: точное воспроизведение24моделей. С1а: только правило выбора на fit20 с разбиением по видео. Затем отдельный протокол новой функции ошибки на fit104; AUC диагностическая, не жёсткий порог. train26 только одна итоговая новая оценка после допуска.',reason)
    revise(parent,'expected','С0/С1а по runs/z13_margin_rule_audit_20261025/protocol.json. 1200с CPU+20с,5фиксированных частей fit20,epsilon0 для замороженных оценок. Следующая С1б: проверка извлекаемого полезного сигнала новой функцией ошибки на fit104 с раздельным обучением/калибровкой/оценкой по видео. До запуска С1б отдельный бюджет,epsilon и протокол; это не измерение потолка информации. С2: неизменный основной критерий +2п.п. на повторно использованных train26 с положительными ДИ против исходного решения,прежнейD,C,R1,безмухи. dev/75/HOTA не используются.',reason)
    revised=[dict(x) for x in parent['acceptance']]
    revised[1]=dict(criterion='С1: измерить чистую пользу фактически выбранных действий при раздельном по видео подборе и проверке порога; AUC описательная,0,93 не универсальный стоп. С1а не является проверкой новой функции ошибки.',met=None)
    revised[2]=dict(criterion='Допуск новой политики к С2: положительные нижние парные95%границы по видео D против нулевого вмешательства и контроля без мухи на внутренних проверочных предсказаниях fit104. При неопределённости не объявлять отсутствия информации; не продвигать к26 до допуска.',met=None)
    revise(parent,'acceptance',revised,reason)
    revise(parent,'cannot_claim','Нет результата новой функции ошибки. Нельзя утверждать доказанную ошибочность MSE/нулевого балла, отсутствие информации по провалу зонда или универсальную необходимость AUC0,93. С1а проверяет только правило выбора. Повторные fit20/train26; проценты HOTA и доля потолка не определены.',reason)
    cat['experiments'].append(dict(id=IDENT,title='Прямой порог пользы при прежних24считывателях FlyVis',area='flyvis',state='running',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],question='Объясняет ли правило выбора часть провала при неизменных оценках?',hypothesis='Сравнение с нулём и прямой порог могут устранить влияние числа кандидатов; польза не гарантирована.',expected=p['s0']+' '+p['s1'],acceptance=[dict(criterion=p['s0'],met=None),dict(criterion=p['criterion'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Критерии до расчёта')] ))
    write(ROOT/'experiments/project_history.json',cat)
    journal('согласованное уточнение С1 и запуск проверки правила','*Постановка.* '+p['action']+' '+p['s1']+'\n\n*Критерии.* '+p['criterion']+'\n\n*Поправка.* '+reason+' Прежние поля сохранены в revisions и parent_before.json. С1б пока не запускалась.\n\n*Бюджет.*1200с CPU+20с, срок '+p['hard_stop']+'.')


def status(stage, **kw):
    p=read(RUN/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists(): raise TimeoutError('Original stage budget')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from tracking.z13_region_assignment_context import make_context_ranker
    p=read(RUN/'protocol.json');verify_sources(p);ns=namespace();entries={}
    for v,h in read(OLD/'cache/report.json')['manifest'].items():
        f=OLD/'cache/videos'/(v+'.json');assert sha(f)==h
        entries.update({int(k):x for k,x in read(f)['entries'].items()})
    records={};models=0;bindings={}
    for arm in ('D','C','R1','geometry'):
        for split in ('train26','fit20'):
            data=ns['load_split'](torch,'D' if arm=='geometry' else arm,split)
            uids=data['uid'].tolist()
            if split in bindings: assert uids==bindings[split]
            else: bindings[split]=uids
            extra=torch.zeros((*data['mask'].shape,6),dtype=torch.float32)
            for i,uid in enumerate(uids):
                e=entries[uid];n=int(data['mask'][i].sum());assert n==len(e['choices'])
                assert int(data['kind'][i])==e['kind']
                data['good'][i,:n]=torch.tensor(e['good'],dtype=torch.bool)
                data['harm'][i,:n]=torch.tensor(e['harm'],dtype=torch.bool)
                extra[i,:n]=torch.tensor(e['context'])
            for mode in ('context','blank'):
                d=dict(data,geo=torch.cat([data['geo'],extra if mode=='context' else torch.zeros_like(extra)],-1))
                variant=arm+'_'+mode
                for seed in p['seeds']:
                    status('Воспроизведение и проверка правила',variant=variant,seed=seed,split=split,done=models,total=24)
                    dest=OLD/'training'/variant/f'seed{seed}';done=read(dest/'done.json');rec=read(dest/'report.json')
                    assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256']
                    assert sha(dest/'rows.json')==rec['rows_sha256']
                    model=make_context_ranker(torch,activity=arm!='geometry')
                    model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
                    logits=ns['predict'](torch,model,d)
                    if split=='train26':
                        rows,per,counts=ns['policy_rows'](d,logits,rec['tau'],p['evaluate26'])
                        assert rows==read(dest/'rows.json') and per==rec['per_video'] and counts==rec['counts'];models+=1
                        write(RUN/'reproduction'/f'{variant}_{seed}.json',dict(exact=True,source_model_sha256=done['model_sha256'],source_rows_sha256=rec['rows_sha256']))
                    else:
                        for rule in ('probability','margin'):
                            r,chosen=crossfit(logits,d['good'].numpy(),d['harm'].numpy(),d['kind'].numpy(),d['video'].numpy(),p['fit20'],p['folds'],rule)
                            rows=[dict(uid=uid,video=p['fit20'][int(d['video'][i])],kind=int(d['kind'][i]),action=int(chosen[i])) for i,uid in enumerate(uids)]
                            folder=RUN/'policies'/variant/f'seed{seed}'/rule
                            write(folder/'rows.json',rows);r['rows_sha256']=sha(folder/'rows.json');write(folder/'report.json',r)
                            records[variant,seed,rule]=r
    assert models==24 and not torch.cuda.is_initialized()
    write(RUN/'reproduction.json',dict(models=24,all_saved_predictions_exact=True,training_models=0,new_train26_policy_evaluations=0))
    means={rule:{a:{v:float(np.mean([records[a,s,rule]['per_video'][v] for s in p['seeds']])) for v in p['fit20']} for a in ('D_context','D_blank','C_context','C_blank','R1_context','R1_blank','geometry_context','geometry_blank')} for rule in ('probability','margin')}
    d=means['margin']['D_context'];zero={v:0. for v in d}
    comp={'Dmargin-baseline':paired(d,zero),'Dmargin-Dprobability':paired(d,means['probability']['D_context'])}
    for a in ('C_context','R1_context','geometry_context','D_blank'):comp['Dmargin-'+a]=paired(d,means['margin'][a])
    primary=['Dmargin-baseline','Dmargin-Dprobability','Dmargin-C_context','Dmargin-R1_context','Dmargin-geometry_context']
    passed=comp['Dmargin-baseline']['mean_per_video']>=.02 and all(comp[k]['ci95'][0]>0 for k in primary)
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,models_reproduced=models,new_models=0,margin_only_pass=passed,paired=comp,means=means,fit_videos=20,folds=5,new_train26_policy_evaluations=0,hota_measured=False,cuda_initialized=False))
    write(RUN/'status.json',dict(state='completed',stage='С0 и С1а завершены',updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv: prepare()
    elif '--worker' in sys.argv: worker()
    else: raise RuntimeError('Explicit mode required')
