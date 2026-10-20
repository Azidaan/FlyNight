"""Publish only the requested results section, its sources and the updated agent rules."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.obsidian_confirmed_results import collect,GROUPS,HUB
from scripts.build_obsidian_vault import safe
from scripts.z13_parallel_next import read,write


def main():
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    run=ROOT/'runs/confirmed_results_graph_20261021'
    entries=collect(ROOT,read(ROOT/'experiments/project_history.json'),read(ROOT/'experiments/metrics_glossary.json'))
    assert len(entries)==9
    graph=read(run/'vault_preservation.json')
    protected={p.replace(chr(92),'/') for p in graph['protected_existing_edits']}
    paths=['AGENTS.md','experiments/CONFIRMED_RESULTS.md','experiments/project_history.json',
        'experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/build_obsidian_vault.py','scripts/obsidian_confirmed_results.py',
        'scripts/seed_confirmed_results_20261021.py','scripts/validate_project_history.py',
        'scripts/publish_confirmed_graph_20261021.py','tests/test_obsidian_confirmed_results.py',
        'ObsidianFlyVis/00 Главная.md','ObsidianFlyVis/'+HUB+'.md']
    paths += ['ObsidianFlyVis/11 Лучшие результаты/'+title+'.md' for title in GROUPS.values()]
    paths += ['ObsidianFlyVis/11 Лучшие результаты/Сравнения/'+safe(x['id'])+'.md' for x in entries]
    paths += ['ObsidianFlyVis/04 Термины и метрики/Доля полезных центров, среднее по видео.md',
              'ObsidianFlyVis/04 Термины и метрики/Термины и метрики.md']
    paths=sorted(set(paths))
    assert not (set(paths)&protected),'Publication would include preexisting graph edits'
    assert all((ROOT/p).is_file() for p in paths)
    write(run/'commit_scope.json',dict(paths=paths,source_checked_results=len(entries),
          excluded='Preexisting graph edits, run logs, caches, model binaries and unrelated files'))
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert not (ROOT/'.git/index.lock').exists()
    staged={p for p in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0') if p}
    assert staged<=set(paths),'Unrelated staged files'
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    staged={p for p in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0') if p}
    assert staged<=set(paths)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Add confirmed-results graph and Moscow calculation windows',
                    '-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True,timeout=90)
    write(run/'publish_result.json',dict(commit=head,published=result.returncode==0,
          exit_code=result.returncode,message=(result.stdout+result.stderr)[-3000:]))
    print(json.dumps(dict(commit=head,published=result.returncode==0)))
    assert result.returncode==0


if __name__=='__main__': main()
