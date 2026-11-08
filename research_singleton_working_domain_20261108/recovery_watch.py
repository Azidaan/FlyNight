import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_singleton_working_domain_20261108.common import *
RUN=RUN/'recovery1'


def same(pid, birth):
    import psutil
    try:
        p = psutil.Process(int(pid))
        return p if abs(p.create_time()-birth) < .01 else None
    except psutil.NoSuchProcess:
        return None


def main():
    import psutil
    root = read(RUN/'controller.started.json')
    window = read(RUN/'execution_window.json'); owned = {}
    while True:
        p = same(root['pid'], root['created'])
        if p is not None:
            assert any('research_singleton_working_domain_20261108/recovery_controller.py' in x.replace('\\','/') for x in p.cmdline())
            for q in [p, *p.children(recursive=True)]:
                try: owned[q.pid] = q.create_time()
                except psutil.NoSuchProcess: pass
        else:
            live = [pid for pid, birth in owned.items() if same(pid, birth) is not None]
            if not live:
                write(RUN/'deadline_result.json', dict(passed=(RUN/'queue_result.json').exists(),
                    all_owned_ended=True, stopped_for=None, ended=datetime.now(MSK).isoformat()), new=True)
                return
        now = datetime.now(MSK)
        reason = 'fifteen_minute_deadline' if now >= datetime.fromisoformat(window['deadline']) else None
        if not (RUN/'first_zero_parity.json').exists() and now >= datetime.fromisoformat(window['first_video_deadline']):
            reason = reason or 'first_video_price_cap'
        if reason:
            killed = []
            for pid, birth in reversed(list(owned.items())):
                q = same(pid, birth)
                if q is not None:
                    q.kill(); killed.append(pid)
            remaining = []
            for pid, birth in owned.items():
                q = same(pid, birth)
                if q is not None:
                    try: q.wait(timeout=5)
                    except psutil.TimeoutExpired: remaining.append(pid)
            write(RUN/'deadline_result.json', dict(passed=False, stopped_for=reason,
                killed=killed, remaining=remaining, all_owned_ended=not remaining,
                ended=datetime.now(MSK).isoformat()), new=True)
            return
        time.sleep(.1)


if __name__ == '__main__':
    main()
