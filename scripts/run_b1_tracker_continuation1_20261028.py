"""Разрешённое автором продолжение19видео; первое не вычисляется повторно."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def continuation_source():
    source = (ROOT / 'scripts/run_b1_tracker_20261028.py').read_text(encoding='utf-8')
    old = "        proof, timing, hashes = {}, {}, {}\n"
    new = old + """        seed = read(FOLDER / 'resume_seed.json')
        per = seed['per_video']
        proof, timing, hashes = seed['proof'], seed['timing'], seed['output_sha256']
        for path, digest in hashes.items():
            assert sha(ROOT / path) == digest, ('Reused first video changed', path)
        write(FOLDER / 'resume_verified.json', dict(passed=True, videos=seed['completed_videos'],
            inference_repeated=False, old_audit_sha256=seed['old_audit_sha256'], updated=now()))
"""
    assert source.count(old) == 1; source = source.replace(old, new)
    old = "            assert not (FOLDER / 'completed' / (v + '.json')).exists(), 'Never repeat completed video'"
    new = """            if v in seed['completed_videos']:
                status(stage='Первое готовое видео сохранено без повторного расчёта', video=v, index=index, total=len(p['sources']))
                continue
""" + old
    assert source.count(old) == 1; source = source.replace(old, new)
    return source


def main():
    from scripts import run_b1_tracker_20261028 as original
    import tracking.b1_tracker_runtime as runtime
    import tracking.foveated_eyes as eyes
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    from tracking.b1_eye_layout import CanonicalEyeBank
    folder = RUN / 'b1_tracker/continuation1'; p = read(folder / 'protocol.json')
    assert read(folder / 'author_override.json')['cost_stop_override_authorized']
    for name in ['b1_tracker/recovery2', 'b1_tracker_audit/recovery2']:
        c = read(RUN / name / 'controller_result.json')
        assert c['exit_code'] == 0 and c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    assert read(RUN / 'b1_tracker/recovery2/independent_audit.json')['passed']
    runtime.ResidualEngine = DeterministicResidualEngine
    eyes.EyeBank = CanonicalEyeBank
    def reuse_regression(engines, target):
        path = RUN / 'b1_tracker/recovery1/incremental_regression.json'
        proof = read(path); assert proof['passed']
        write(target / 'incremental_regression.json', dict(proof, reused_completed_proof=True,
            source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path)))
    code = continuation_source()
    assert hashlib.sha256(code.encode()).hexdigest() == p['executed_worker_sha256']
    ns = {**vars(original), '__name__': 'b1_continuation_worker', 'FOLDER': folder, 'regression': reuse_regression}
    exec(compile(code, str(ROOT / 'scripts/run_b1_tracker_20261028.py'), 'exec'), ns)
    # exec исходника задаёт исходный FOLDER/regression; адреса меняются явно после него.
    ns['FOLDER'] = folder; ns['regression'] = reuse_regression
    ns['run']()


if __name__ == '__main__': main()
