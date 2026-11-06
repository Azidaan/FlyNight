"""Freeze exact owned source/evidence paths, without staging or modifying sources."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = json.loads((OUT / 'before_sha256.json').read_text(encoding='utf-8'))
owned = [dict(path=r['path'], sha256=sha(ROOT / r['path']), snapshot=r['snapshot'], previous_sha256=r['sha256']) for r in before]
new = 'tests/test_motion_calibration_binding.py'
owned.append(dict(path=new, sha256=sha(ROOT / new), snapshot=None, previous_sha256=None))
artifacts = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
             and p.name != 'manifest.json' and '__pycache__' not in p.parts
             and not any(part.startswith('pytest_tmp') for part in p.parts)}
report = dict(owned_files=owned, artifacts=artifacts,
              decision='legacy_v1 ordinary defaults; both new geometries explicit opt-in after two failed promotion gates',
              final_checks='tests_final.log: 274 passed; final_default_report.json exact native points/vectors; original_native_v2_report.json exact full-estimator CPU parity with cuda_initialized=false',
              next_package_excluded='tests/test_flyvis_observation_transaction.py and its future production fix belong to root; not included here')
(OUT / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(owned=len(owned), artifacts=len(artifacts), motion_sha256=sha(ROOT / 'tracking/flyvis_motion.py'),
                      k1_sha256=sha(ROOT / 'scripts/fly_finetune_k1.py'))))
