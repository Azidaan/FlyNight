"""Прямое решение автора: общий срок сохраняется, меньшие бюджеты отменены.

Отдельная версия исполнения; старые результаты и замороженные исходники неизменны.
"""
from pathlib import Path
from datetime import datetime
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuation_service_20261028 import RUN, service

POLICY = RUN / 'stage_budget_policy_20261029' / 'authorization.json'


def policy():
    a = service.read(POLICY)
    assert a['human_message'] == 'Только бюджеты этапов; общую остановку в 07:40 оставить'
    assert a['stage_time_limits_cancelled'] is True
    auth = service.read(RUN / 'authorization.json')
    assert a['hard_stop'] == auth['hard_stop']
    return a


def ended(folder):
    c = service.read(folder / 'controller_result.json')
    assert c['all_owned_ended']
    assert not any(service.matching(k, v) for k, v in c['owned'].items())
    return c


def transformed(kind, source_folder, audit_folder):
    a = policy()
    names = {'gpu': 'run_history_full_tracker_pilot_20261028.py',
             'audit': 'audit_history_full_tracker_20261028.py',
             'record': 'record_history_full_tracker_20261028.py'}
    original = ROOT / 'scripts' / names[kind]
    p = service.read(source_folder / 'protocol.json')
    assert service.sha(original) == p['sha256'][original.relative_to(ROOT).as_posix()]
    code = original.read_text(encoding='utf-8')
    code = code.replace('history_tracker_pilot', source_folder.name)
    code = code.replace('history_tracker_audit', audit_folder.name)
    code = code.replace('history_tracker_record', 'history_tracker_author_record1')
    if kind in ['gpu', 'audit']:
        old = "datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()"
        assert code.count(old) == 1
        code = code.replace(old, repr(datetime.fromisoformat(a['save_at']).timestamp()))
        # Цена остаётся в отчёте как прогноз; прямое решение отменило этот стоп.
        assert code.count("if not price['proceed']:") == 1
        code = code.replace("if not price['proceed']:", "if False:  # Author cancelled stage time gate")
    if kind == 'audit':
        old = "expected_cost=matrix.dtype.type(old['original']+self.alpha*(.5-1/(1+np.exp(-np.clip(old['score'],-80,80))))) if self.alpha else matrix.dtype.type(old['original'])"
        assert code.count(old) == 1
        code = code.replace(old, "expected_cost=replay_cost(old['original'],old['score'],self.alpha,matrix.dtype)")
    return code, original


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert service.sha(src) == service.sha(dst)
    else:
        os.link(src, dst)
        assert service.sha(src) == service.sha(dst)


def execute(kind):
    policy()
    initial = RUN / 'history_tracker_author_continuation1'
    audit = RUN / 'history_tracker_author_audit1'
    proof = POLICY.parent / 'selected_gpu_result.json'
    if kind == 'gpu':
        c = ended(initial)
        r = service.read(initial / 'report.json') if (initial / 'report.json').exists() else {}
        if c['exit_code'] == 0 and r.get('complete'):
            service.write(proof, dict(folder=initial.relative_to(ROOT).as_posix(),
                original_controller_sha256=service.sha(initial/'controller_result.json'),
                gpu_repeated=False, complete=True, updated=service.now()))
            return
        # Автоматическое продолжение только установленного старого временного стопа.
        err = (initial / 'stderr.log').read_text(encoding='utf-8', errors='replace')
        assert c['timed_out'] or 'Original history full tracker budget' in err or r.get('stopped_on_cost'), err[-1500:]
        target = RUN / 'history_tracker_stage_time_continuation2'
        target.mkdir(exist_ok=True)
        service.write(target/'protocol.json', service.read(initial/'protocol.json'))
        link(initial/'frozen.json', target/'frozen.json')
        reused = []
        for s in service.read(initial/'protocol.json')['sources']:
            v = s['video']; src = initial/'videos'/v
            if not (src/'report.json').exists():
                # Сохранённый GPU-проход без метрик нельзя повторять: нужен CPU-recovery.
                assert not (src/'tracking_completed.json').exists(), ('Recover CPU metrics only', v)
                continue
            e = service.read(src/'report.json')
            assert all(service.sha(ROOT/f) == h for f,h in e['sha256'].items())
            for f in src.rglob('*'):
                if f.is_file(): link(f, target/'videos'/v/f.relative_to(src))
            for n in e['metrics']: link(initial/n/(v+'.json'), target/n/(v+'.json'))
            reused.append(v)
        service.write(target/'resume_proof.json',dict(reused=reused,
            original_controller_sha256=service.sha(initial/'controller_result.json'),
            policy_sha256=service.sha(POLICY), updated=service.now()))
        source = target
        service.write(proof,dict(folder=target.relative_to(ROOT).as_posix(),
            complete=False, reused=reused, updated=service.now()))
    else:
        source = ROOT / service.read(proof)['folder']
        if source != initial:
            followup = POLICY.parent / 'gpu_followup'
            c = ended(followup)
            assert c['exit_code'] == 0 and not c['timed_out']
            if not (source/'controller_result.json').exists():
                service.write(source/'controller_result.json', c)
                service.write(source/'controller_source.json', dict(
                    source=(followup/'controller_result.json').relative_to(ROOT).as_posix(),
                    sha256=service.sha(followup/'controller_result.json')))
    code, original = transformed(kind, source, audit)
    ns = {'__name__': 'history_stage_time_'+kind, '__file__': str(original)}
    if kind == 'audit':
        from tracking.history_cost_replay import replay_cost
        ns['replay_cost'] = replay_cost
    exec(compile(code, str(original), 'exec'), ns)
    ns['main']()


if __name__ == '__main__':
    kind = Path(sys.argv[0]).stem.rsplit('_',1)[-1]
    raise RuntimeError('Use dedicated gpu/audit/record entry point')
