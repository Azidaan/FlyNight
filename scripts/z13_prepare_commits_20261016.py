"""Prepare explicit commit path lists without modifying Git's index."""
from pathlib import Path
import ast,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/z13_commit_preparation_20261016'

def git(*args): return subprocess.check_output(['git','-c','core.quotepath=false',*args],cwd=ROOT,text=True,encoding='utf-8').splitlines()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    untracked=git('ls-files','--others','--exclude-standard','--','scripts','tracking','tests')
    code=[p for p in untracked if Path(p).name.startswith(('z13_','record_z13_','test_z13_'))]
    code+=['AGENTS.md','scripts/z13_build_vault_preserving_edits.py']
    for name in code:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name)
    research=[]
    for folder in ('research_flyvis_china_2026-10-15','research_flyvis_decision_trees_2026-10-15'):
        research += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts]
    names={'protocol.json','report.json','result.json','summary.json','final_verification.json','inference_identity.json',
        'reuse_proof.json','verification.json','recovery_verified.json','size_recovery_diagnostic.json','recovery2_protocol.json',
        'recovery2_controller_result.json','completion_verified.json','technical.json','completed.json','size_center_check.json','shared_activity_check.json'}
    artifacts=[]
    for run in (ROOT/'runs').glob('z13_*'):
        if run.name in ('z13_data',): continue
        for p in run.rglob('*.json'):
            if p.name in names and p.stat().st_size<2_000_000:
                artifacts.append(str(p.relative_to(ROOT)).replace('\\','/'))
    records=['dissertation/ЕДИНЫЙ_ПЛАН.md','experiments/project_history.json']+artifacts
    protected=json.loads((ROOT/'runs/z13_spatial_hota_v1_20261014/vault_preservation.json').read_text(encoding='utf-8'))['protected_existing_edits']
    protected={p.replace('\\','/') for p in protected}
    graph=[]
    candidates=git('ls-files','--others','--exclude-standard','--','ObsidianFlyVis')
    for p in candidates:
        if p in protected: continue
        if '/01 Муха/Эксперименты/' in '/'+p and Path(p).stem.startswith(('z13_','fly_spatial_hota_policy')): graph.append(p)
        elif '/09 Запланировано/' in '/'+p and Path(p).stem.startswith('z13_'): graph.append(p)
        elif '/06 Журнал/' in '/'+p and Path(p).stem.startswith(('2026-10-15','2026-10-16')): graph.append(p)
    groups={'code':sorted(set(code)),'research_and_results':sorted(set(research+records+graph))}
    assert not any(p in protected for paths in groups.values() for p in paths)
    for key,paths in groups.items():
        (OUT/(key+'.paths')).write_bytes(('\0'.join(paths)+'\0').encode('utf-8'))
    manifest=dict(groups={k:[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths] for k,paths in groups.items()},
        protected_graph_paths=sorted(protected),index_changed=False,lock_removed=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:dict(files=len(v),bytes=sum((ROOT/p).stat().st_size for p in v)) for k,v in groups.items()}))

if __name__=='__main__': main()
