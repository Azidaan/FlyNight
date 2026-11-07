"""Freeze the task's own evidence without staging other worktree changes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_state_subgroups_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert '71 passed' in (OUT/'record_tests.log').read_text(encoding='utf-8-sig')
    assert 'OK: каталог согласован с источниками' in (OUT/'catalog_check.log').read_text(encoding='utf-8-sig')
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    sources=dict(p['frozen_sources'])
    if (OUT/'recovery_manifest.json').exists():
        recovery=json.loads((OUT/'recovery_manifest.json').read_text(encoding='utf-8'))
        sources.update(recovery['new_sources'])
        assert json.loads((OUT/'batch_equivalence.json').read_text())['passed']
        for case,reference in recovery['reused_cases'].items():
            assert sha(OUT/'cases'/(case+'.npz'))==reference['sha256']
    for f,h in sources.items():
        assert sha(ROOT/f)==h,f
        if f.endswith('.py'):
            target=OUT/'frozen_sources'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
    # A binary archive preserves original line endings even across Git checkouts.
    archive=OUT/'frozen_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in (OUT/'frozen_sources').rglob('*'):
            if f.is_file():z.write(f,f.relative_to(OUT/'frozen_sources').as_posix())
        z.write(OUT/'protocol.json','protocol.json')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            expected=sha(OUT/'protocol.json') if name=='protocol.json' else sources[name]
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
    c=next(c for c in json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))['experiments'] if c['id']==p['id'])
    problem=next(x for x in json.loads((ROOT/'experiments/system_problems.json').read_text(encoding='utf-8'))['problems'] if x['id']=='С7')
    case=problem['cases'][-1];assert OUT.name in case['fact']
    snapshot=dict(card=c,problem_id=problem['id'],appended_problem_case=case,problem_evidence=[e for e in problem['evidence'] if OUT.name in e['source']],note='Exact task-specific entries; shared catalog, chronology, roadmap and vault retain their other uncommitted work and are not staged wholesale.')
    (OUT/'record_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    artifacts={f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.suffix in ('.json','.md','.log') and f.name!='completion_verified.json'}
    result=dict(complete=True,checked_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),artifacts=artifacts,cases={c['id']:sha(OUT/'cases'/(c['id']+'.npz')) for c in p['cases']},catalog_passed=True,tests_passed=71,source_hashes_still_exact=True,source_archive_sha256=sha(archive),scope='Completed bounded partial-state history diagnostic with continuing-motion control and documentation; no further scientific run scheduled.')
    (OUT/'completion_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Completion verified; 64 cases, 71 tests.')
if __name__=='__main__':main()
