"""Новая реализация экспорта: однократное чтение NPZ и пакетные сравнения."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'runs/b123_20261027/b1_base_export_vectorized'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(prepare):
    if prepare:
        OUT.mkdir(exist_ok=False)
        sources = ['scripts/b1_base_export_vectorized_20261027.py', 'scripts/b1_export_base_20261027.py',
                   'scripts/b1_npz_recovery_20261026.py', 'tracking/b1_appearance.py', 'tracking/b1_appearance_fast.py',
                   'tests/test_b1_appearance_fast.py', 'runs/b1_real_inputs_20261026/base_inputs.npz',
                   'runs/b1_real_inputs_20261026/base_predictions.npz']
        p = dict(id='b1_base_export_vectorized_v1', created=datetime.now(timezone.utc).isoformat(),
            budget_seconds=300, author_stop='2026-10-27T07:20:00+03:00',
            question='Can cached NPZ loading and vectorized exact relative comparison reconstruct live base features within bounded CPU cost?',
            change='New implementation; prior export attempts and expired deadlines preserved. No training or new predictions fitted.',
            acceptance=['Maximum finite feature difference <1e-5; identical missing values',
                        'Exact saved identity and gate predictions on fixed saved rows',
                        'Same existing PCA recipe, no labels in features, no GPU'],
            sha256={s:sha(ROOT/s) for s in sources})
        (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:
        p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        for name,digest in p['sha256'].items():
            assert sha(ROOT/name)==digest,name
        from scripts.b1_npz_recovery_20261026 import install
        install()
        from tracking import b1_appearance
        from tracking.b1_appearance_fast import relative_features
        from scripts import b1_export_base_20261027 as worker
        b1_appearance.relative_features=relative_features
        # Original main() insists on creating a new directory. Keep protocol separate
        # from its untouched outputs and compare the same fixed arrays and models.
        worker.OUT=OUT/'outputs'
        worker.main()


if __name__=='__main__':
    main('--prepare' in sys.argv)
