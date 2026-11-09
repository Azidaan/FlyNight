"""Record the numerical question before its fixed saved-matrix calculation."""
from common import *
start('cpu',timeout=600)
assert not (OUT/'boundary_protocol.json').exists()
p=read(OUT/'protocol.json')
files=[OUT/n for n in ['register_boundary.py','boundary_audit.py','comparison_report.json','id_fate.json',
    'execution_protocol.json','evaluation_protocol.json']]
files += [ROOT/'tracking/spatial_identity_assignment.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py']
files += [p for p in (ROOT/'.venv/Lib/site-packages/lap').glob('*') if p.is_file() and (p.suffix=='.py' or p.suffix=='.pyd')]
for arm in p['arms']:
    files += list((OUT/'rollouts'/arm).glob('*.receipt.json'))
    files += list((OUT/'rollouts'/arm).glob('*.events.json.gz'))
write('boundary_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    question='Can the recorded verifier-positive native edge still be above cost_limit because of float32 arithmetic, and does the full original matching reproduce?',
    hypotheses=[dict(rank=1,status='arithmetic risk, solver effect not yet measured',mechanism='Float32(1)-.2 is0.800000011920929 >Pythonfloat.8; finite-limit assignment can prefer leaving row/column unmatched.'),
        dict(rank=2,status='known architecture, case effect not yet measured',mechanism='Even a below-limit edge may lose to a cheaper competing track in the global objective.'),
        dict(rank=3,status='known architecture, case effect measured separately',mechanism='Native duplicate removal after matching may remove the old public ID.')],
    fixed_tests='Three1x1 values:float32 subtraction1-.2, float64 subtraction1-.2, one adjacent float64 below. Same fixed cost_limit.8. Then ALL saved frames containing accepted proposals: reproduce stored matching exactly; reconstruct only same-alpha subtraction infloat64 and compare assignments. No feature/cell/model/threshold variant, no rollout.',
    source_limit='Actual imported lap module and its compiled file must be identified. Distribution metadata alone does not identify solver because lap andlapx share import name. Current hashes are recorded now, not retroactively asserted as earlier-run dependency hashes.',
    primary_code='Installed lap/_lapjv_wp.py describes finite-limit rejection at half-limit per unmatched side. Read actual installed source; matching.py calls lap.lapjv. Public original https://github.com/rathaROG/lapx , earlier parent implementation https://raw.githubusercontent.com/gatagat/lap/v0.5.13/_lapjv_cpp/_lapjv.pyx ; guessed lapx tag raw paths returned404 and are NOT treated as verified0.9.4kernel source.',
    decision='Only after observed full pilot and this fixed check decide whether a minimal numerical implementation correction is needed. No in-place change of frozen current pilot. One-frame assignment changes never counted as preserved future IDs.',
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files}))
print('Numerical boundary audit registered',flush=True)
