"""Service-only remaining-video price correction before stock GPU execution."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_stock_controls_20261024 as S


def transformed():
    text=inspect.getsource(S.worker);old='estimate=2*measured*7+600';assert text.count(old)==1
    return text.replace(old,'estimate=2*measured*6+600')


def verify():
    p=S.read(S.STAGE/'execution_v2.json');assert p['service_sha256']==S.sha(Path(__file__)) and p['science_protocol_sha256']==S.sha(S.STAGE/'protocol.json')


if __name__=='__main__':
    if '--prepare' in sys.argv:
        assert not (S.STAGE/'controller.json').exists() and not (S.STAGE/'execution_v2.json').exists()
        S.write(S.STAGE/'execution_v2.json',dict(at=S.now().isoformat(),service_sha256=S.sha(Path(__file__)),science_protocol_sha256=S.sha(S.STAGE/'protocol.json'),change='Correct only the first-complete-video remaining-scope price:2xmeasured*6remaining+600,not7including completed one. Safety factor,reserve,10800seconds,science/weights/data/thresholds unchanged. Same documented accounting correction as parent; registered before any stock outcomes.'))
        S.journal('поправлен учёт оставшихся роликов до запуска контролей','Тот же служебный дефект ценового учёта обнаружен в ещё не начавшей вычисления очередиC/B/R1: после первого готового ролика остаются6,а не7. Отдельная версия scripts/z13_matched_stock_execution2_20261024.py меняет только учёт остатка при прежнихмножителе2/резерве600с/бюджете10800с. Исходный stock_controls/protocol.json и научный код неизменны,метаданные execution_v2.json. РезультатовC/B/R1 ещё нет,порог по эффективности не меняется.')
    elif '--worker' in sys.argv:
        verify();S.__file__=str(Path(__file__));ns=vars(S).copy();exec(compile(transformed(),str(Path(__file__)),'exec'),ns)
        try:ns['worker']()
        except Exception as e:S.write(S.STAGE/'status.json',dict(state='failed',stage=str(e),updated=S.now().isoformat()));raise
    elif '--job' in sys.argv:
        verify();mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in S.VIDEOS and arm in S.SYSTEMS+('all',)
        result=S.build_signature(arm,video) if mode=='signature' else S.build_masa(video)
        paths=[S.RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [S.RUN/a/'masa'/(video+'.npz') for a in S.SYSTEMS]+[S.STAGE/'baseline_masa_replay'/(video+'.npz')]
        S.write(S.STAGE/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():S.sha(q) for q in paths},at=S.now().isoformat()))
    else:raise RuntimeError('Explicit service mode required')
