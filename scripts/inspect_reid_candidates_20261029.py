"""Read-only upstream inspection for the three author-requested candidates.

Downloads source text/metadata only. Never imports remote code or loads weights.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/continuation_20261028/reid_candidates_preparation'


def main():
    OUT.mkdir(exist_ok=True)
    evidence, errors = [], []

    def fetch(url, name):
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Dissertation-readiness/1.0'})
            with urllib.request.urlopen(request, timeout=25) as response:
                content = response.read(8 * 1024 * 1024)
            if target.exists():
                assert target.read_bytes() == content, 'Upstream changed; preserve original snapshot'
            else:
                target.write_bytes(content)
            evidence.append(dict(url=url, path=target.relative_to(ROOT).as_posix(),
                                 sha256=hashlib.sha256(content).hexdigest()))
            return content
        except Exception as error:
            errors.append(dict(url=url, error=repr(error)))
            return None

    trees = {}
    for repo, prefix in [('siyuanliii/masa', 'masa'), ('whucsy/OG-ReID', 'mgor')]:
        raw = fetch(f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1', prefix+'/tree.json')
        if raw:
            tree = json.loads(raw)
            trees[prefix] = tree
            paths = [entry['path'] for entry in tree['tree'] if entry['type'] == 'blob']
            wanted = [p for p in paths if p == 'README.md' or p == 'requirements.txt']
            if prefix == 'masa':
                wanted += [p for p in paths if p in (
                    'docs/model_zoo.md', 'configs/masa-detic/bdd_test/masa_detic_bdd_mot_test.py',
                    'projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-masa.py',
                    'masa/models/detectors/detic_masa.py')]
            else:
                wanted += [p for p in paths if p.startswith('configs/object/') or p == 'train_meta.py']
            for path in wanted:
                fetch(f'https://raw.githubusercontent.com/{repo}/{tree["sha"]}/{path}', prefix+'/'+path)

    fetch('https://huggingface.co/api/models/dereksiyuanli/masa/tree/main?recursive=true&expand=false', 'masa/weights_index.json')
    fetch('https://api2.openreview.net/notes?forum=INnZIRw2XE&limit=1000', 'scaletrack/openreview.json')
    fetch('https://api.github.com/users/siyuanliii/repos?per_page=100', 'scaletrack/author_repositories.json')
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ('torch', 'torchvision', 'timm', 'mmdet', 'mmcv', 'mmengine', 'detectron2')}
    report = dict(created=datetime.now(timezone.utc).isoformat(),
        scope='Source and dependency inspection only; no inference, training or HOTA',
        scientific_verdict=None, inference_executed=False, weights_downloaded=False,
        dependencies=dependencies, source_revisions={k:v['sha'] for k,v in trees.items()},
        evidence=evidence, errors=errors)
    (OUT/'availability.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(evidence=len(evidence), errors=errors, dependencies=dependencies), ensure_ascii=False))


if __name__ == '__main__':
    main()
