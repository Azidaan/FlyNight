"""Matched signed-utility readout: only the regional training objective changes."""
from pathlib import Path
from datetime import timedelta,datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import ast
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_balanced_20261025 as balanced
from scripts import z13_region_assignment_recovery2_20261025 as original
from tracking.z13_region_value_loss import signed_action_loss
RUN=ROOT/'runs/z13_region_value_20261025'
IDENT='z13_region_value_v1'


def segment(source,name):
    tree=ast.parse(source)
    return ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


def training_source():
    code=segment(original.source(),'train_worker')
    changes=[("loss=marginal_choice_loss(torch,scores,d['good'][idx],d['mask'][idx],1.)",
              "loss=signed_action_loss(torch,scores,d['good'][idx],d['harm'][idx],d['mask'][idx])"),
             ("train_known=len(known),cuda_initialized=","train_known=len(known),loss_target='beneficial-harmful',cuda_initialized="),
             ("passed=comp['D-baseline']['mean_per_video']>=.02",
              "comp['D-value-D-uniform']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])\n    comp['D-value-D-balanced']=paired(means['D'],read(BALANCED/'training/report.json')['mean_seed_per_video']['D'])\n    passed=comp['D-baseline']['mean_per_video']>=.02")]
    for old,new in changes:assert code.count(old)==1;code=code.replace(old,new)
    return code


def namespace():
    ns={**vars(balanced),'RUN':RUN,'IDENT':IDENT,'BASE':original.RUN,'BALANCED':balanced.RUN,'signed_action_loss':signed_action_loss}
    text=(ROOT/'scripts/z13_region_balanced_20261025.py').read_text(encoding='utf-8')
    for name in ('journal','status','finish','publish','monitor'):
        code=segment(text,name)
        if name=='finish':
            code=code.replace("m['positive_weight']==p['positive_weight']", "m['loss_target']=='beneficial-harmful'")
            code=code.replace("comp['D-balanced-D-uniform']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])",
                "comp['D-value-D-uniform']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])\n    comp['D-value-D-balanced']=paired(means['D'],read(BALANCED/'training/report.json')['mean_seed_per_video']['D'])")
            code=code.replace('Равный вес запросов:','Обучение знаковой полезности:').replace('Изменён только вес классов запроса.','Изменена только цель ошибки: польза +1, нейтральный исход0, вред −1.').replace('Балансировка классов регионального узла закончена.','Знаковая полезность регионального узла закончена.').replace('итог равного веса запросов вмешательства','итог знаковой полезности назначения').replace('Проверен полный пилот взвешивания классов.','Проверен полный пилот знаковой полезности.')
        if name=='publish':
            code=code.replace('tracking/z13_region_balanced_loss.py','tracking/z13_region_value_loss.py').replace('tests/test_z13_region_balanced_loss.py','tests/test_z13_region_value_loss.py').replace('scripts/z13_region_balanced_console.ps1','scripts/z13_region_value_console.ps1')
            code=code.replace("Path(__file__).relative_to(ROOT).as_posix()", "'scripts/z13_region_value_20261025.py'")
            code=code.replace('Evaluate balanced action supervision of regional FlyVis readout','Evaluate signed action utility in the regional FlyVis readout')
        exec(compile(code,str(Path(__file__)),'exec'),ns)
    return ns


def worker():
    ns=namespace();ns['load_split']=original.namespace()['load_split']
    p=read(RUN/'protocol.json');assert hashlib.sha256(training_source().encode()).hexdigest()==p['executed_training_sha256']
    exec(compile(training_source(),str(Path(__file__)),'exec'),ns);ns['train_worker']()
    write(RUN/'status.json',dict(state='completed',stage='All12 signed-utility models saved',updated=now().isoformat()))


def prepare():
    assert not RUN.exists()
    previous=read(balanced.RUN/'publication.json');assert previous['push_completed']
    r=read(balanced.RUN/'training/report.json');assert r['complete'] and not r['primary_pass']
    base=read(original.RUN/'protocol.json');start=now();stop=min(start+timedelta(seconds=1800),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_value_loss.py','tests/test_z13_region_value_loss.py',
           'scripts/z13_region_value_console.ps1','scripts/z13_region_balanced_20261025.py',
           'runs/z13_region_balanced_20261025/training/report.json','runs/z13_region_balanced_20261025/independent_audit.json',
           'runs/z13_region_assignment_20261025/recovery2/features/report.json','runs/z13_region_assignment_20261025/recovery2/training/report.json']
    p=dict(base,at=start.isoformat(),budget_seconds=1800,device='cpu',timezone='Europe/Moscow',save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),
       stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
       action='Only training loss changes from marginal cross entropy to per-query mean masked squared error against good-harm. Same head, features, initialization, UID minibatches,1150steps/128,three seeds, policy decoder and fit20 threshold grid.',
       primary='D-baseline mean>=2pp; all six lower95% paired-video bounds>0: baseline,C,R1,geometry,uniform-D,balanced-D.',
       counterargument='The same observed region can have different global assignment outcomes; missing assignment context remains. Signed regression may merely collapse near zero. It must beat the identical no-FlyVis control, not just produce more interventions.',
       distinction='Previous regional losses supervise beneficial versus unchanged and never consume cached harm flags. This retains zero for neutral actions and -1 for harmful actions; not the old next_utility center/IoU target.',
       literature='Mozannar and Sontag, ICML2020, Consistent Estimators for Learning to Defer to an Expert, sections3-4: https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf . Supports cost-sensitive joint-system objective; our squared signed regression is a separate adaptation, not their loss or tracking evidence.',
       next='Pass -> separate gated closed-loop runtime parity/price/HOTA. Fail -> no HOTA; stop scalar-loss variants and test missing observable assignment context as a separate node.',
       executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'training/protocol.json',p)
    cat=read(ROOT/'experiments/project_history.json');parent=next(c for c in cat['experiments'] if c['id']=='z13_region_assignment_v1')
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Считыватель знаковой полезности сопоставления',area=parent['area'],state='ready',
        question='Поможет ли различение полезного, нейтрального и вредного действия в цели обучения регионального считывателя?',
        hypothesis='Рабочая гипотеза: прежняя цель одинаково штрафует нейтральные и вредные ответы, хотя критерий считает пользу минус вред. Изменение цели может улучшить допуск; причинность ещё не доказана.',
        expected='D улучшает назначения при равных C/R1/геометрии и превосходит прежние региональные потери.',
        acceptance=[dict(criterion='Все12 моделей и первичные строки, неизменные признаки/пакеты/пороговый протокол подтверждены',met=None),dict(criterion=p['primary'],met=None)],
        started=start.strftime('%Y-%m-%d'),evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Критерии и отличие от прежних потерь до обучения')],
        tags=parent['tags'],obtained='',interpretation='',decision='',cannot_claim='Однокадровые назначения на повторно использованных train26, не HOTA; аналогия cost-sensitive learning не доказывает эффективность здесь.'))
    write(ROOT/'experiments/project_history.json',cat)
    namespace()['journal']('постановка знаковой полезности назначения',
       '*Постановка.* Меняется только ошибка того же считывателя: средний по действиям квадрат ошибки относительно полезно +1, нейтрально0, вредно −1. '
       'Прежняя marginal_choice_loss использует good, но не harm; нейтральные и вредные действия оба противопоставлены неизменному ответу. '
       'Это подтверждено исходниками, но причина неудач пока гипотеза. Прежние признаки, архитектура,1150шагов, пакеты и порог fit20. '
       '12 CPU-моделей,1800с, исходные сохранённые признаки без GPU. '+p['primary']+' '+p['counterargument']+' '+p['literature']+' Источник: '+(RUN/'protocol.json').relative_to(ROOT).as_posix()+'.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--monitor' in sys.argv:namespace()['monitor']()
    else:raise RuntimeError('Explicit mode required')
