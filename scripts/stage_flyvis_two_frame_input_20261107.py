"""Stage a strict source/receipt whitelist; never include retinal inputs or maps."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,read,write
assert read(OUT/'checks_summary.json')['all_owned_ended']
paths=['.gitattributes','FLYVIS_TRACKER_RESEARCH_ROADMAP.md','dissertation/ЕДИНЫЙ_ПЛАН.md','experiments/project_history.json','scripts/obsidian_fundamental_overview.py','research_sigma_pi_20261107/ОЦЕНКА.md']
paths+=['ObsidianFlyVis/'+p for p in read(OUT/'graph_live_verified.json')['changed']]
paths += [p.as_posix() for p in Path('scripts').glob('*flyvis_two_frame*20261107.py')]
paths += [p.as_posix() for p in OUT.iterdir() if p.is_file() and (p.suffix in {'.json','.md','.log','.err'} or p.name=='.gitignore')]
assert all(not p.endswith(('.npz','.pdf','.png','.jpg','.txt')) for p in paths)
assert all(Path(p).stat().st_size<2_000_000 for p in paths if p.startswith(('scripts/','runs/flyvis_two_frame_input_20261107/','research_sigma_pi_20261107/')))
write(OUT/'publication_scope.json',dict(files=paths,retinal_inputs_and_maps_included=False))
paths.append((OUT/'publication_scope.json').as_posix())
assert not subprocess.check_output(['git','diff','--cached','--name-only'])
subprocess.run(['git','add','-f','--',*paths],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
print('Staged',len(paths),'explicit files')
