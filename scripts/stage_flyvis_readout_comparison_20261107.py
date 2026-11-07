"""Explicit safe publication scope, excluding data, labels, models and predictions."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,read,write
assert read(OUT/'checks_summary.json')['all_owned_ended']
paths=['.gitattributes','FLYVIS_TRACKER_RESEARCH_ROADMAP.md','dissertation/ЕДИНЫЙ_ПЛАН.md','experiments/project_history.json','experiments/system_problems.json',
    'scripts/obsidian_fundamental_overview.py','tracking/flyvis_cached_readouts.py','tests/test_flyvis_cached_readouts.py','research_sigma_pi_20261107/ПРИМЕНЕНИЕ.md','scripts/describe_flyvis_readout_fit_20261107.py']
paths += ['ObsidianFlyVis/'+p for p in read(OUT/'graph_live_verified.json')['changed']]
paths += [p.as_posix() for p in Path('scripts').glob('*flyvis_readout_comparison*20261107.py')]
paths += [p.as_posix() for p in OUT.iterdir() if p.is_file() and (p.suffix in {'.json','.md','.log','.err'} or p.name=='.gitignore')]
assert all(not p.endswith(('.npz','.pt','.png','.jpg','.pdf')) for p in paths)
assert all(Path(p).stat().st_size<3_000_000 for p in paths if p.startswith(('scripts/','runs/flyvis_readout_comparison_20261107/','tracking/','tests/')))
write(OUT/'publication_scope.json',dict(files=paths,private_arrays_and_models_included=False))
paths.append((OUT/'publication_scope.json').as_posix())
assert not subprocess.check_output(['git','diff','--cached','--name-only'])
subprocess.run(['git','add','-f','--',*paths],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
print('Staged',len(paths),'explicit files')
