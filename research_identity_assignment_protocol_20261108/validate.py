"""Validate prepared documents and bindings; never run a scientific comparison."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=read(OUT/'protocol.json');q=read(OUT/'queries.json')['rows'];e=read(OUT/'evaluation_strata.json')['rows']
    assert p['status']=='prepared_not_executed_not_launch_ready' and not p['readiness']['launch_ready']
    assert len(q)==len(e)==1991
    assert len({x['uid'] for x in q})==len(q) and {x['uid'] for x in q}=={x['uid'] for x in e}
    allowed={'uid','video','split','candidate_high_ids','source','frame','tid','episode'}
    assert all(set(x)==allowed for x in q)
    assert all(len(x['candidate_high_ids'])==2 and len(set(x['candidate_high_ids']))==2 for x in q)
    for split,total,known,positive,negative in [('fit20',623,323,90,233),('evaluate26',1368,641,178,463)]:
        subset=[x for x in e if x['split']==split]
        assert len(subset)==total and sum(x['primary_known'] for x in subset)==known
        assert sum(x['reason']=='ok' for x in subset)==positive
        assert sum(x['reason']=='neither_target' for x in subset)==negative
        assert all(x['primary_known']==(x['reason'] in ['ok','neither_target']) for x in subset)
    assert len(set(p['arms']))==8
    comparisons=p['inference']['primary_comparisons']
    assert len(comparisons)==p['inference']['family_size']==7
    assert len({tuple(x) for x in comparisons})==7 and all(a in p['arms'] and b in p['arms'] for a,b in comparisons)
    assert abs(p['inference']['ci_levels_percent'][1]-99.28571428571429)<1e-10
    assert p['calibration']['alpha_grid']==p['calibration']['beta_grid']==[0,.1,.2,.4,.8]
    dev={x['video'] for x in q if x['split']=='fit20'}
    pilot=set(p['cohorts']['future_val20']['videos'])
    assert len(dev)==len(pilot)==20 and not dev&pilot
    assert not p['cohorts']['future_val20']['researcher_unseen']
    for name,h in p['source_sha256'].items():
        path=(ROOT/name).resolve();path.relative_to(ROOT.resolve());assert sha(path)==h,name
    checks=read(OUT/'preparation_checks.json')
    for name,h in checks['protected_sha256'].items():assert sha(ROOT/name)==h,name
    assert sha(ROOT/'dissertation/План целей по Трекеру.md')==checks['tracker_plan_sha256']
    assert sha(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md')==checks['working_plan_sha256_after']
    for name,h in read(OUT/'checksums.json')['files'].items():assert sha(OUT/name)==h,name
    changed=[x.decode('utf-8') for x in subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).split(b'\0') if x]
    assert set(changed)=={'dissertation/ЕДИНЫЙ_ПЛАН.md','dissertation/План целей по Трекеру.md'},changed
    tree=ast.parse((OUT/'prepare.py').read_text(encoding='utf-8'))
    imported={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert not imported&{'torch','numpy','cv2','tracking','ultralytics'}
    text=(OUT/'ПРОТОКОЛ.md').read_text(encoding='utf-8')
    assert all('`'+arm+'`' in text for arm in p['arms'])
    assert not any(line.startswith(('- ','* ')) for line in text.splitlines())
    value=dict(passed=True,scientific_runs=0,model_imports=0,query_bindings=1991,
        label_free_runtime_manifest=True,known_development_queries=323,known_historical_evaluation_queries=641,
        arms=8,primary_comparisons=7,pilot_development_video_overlap=0,
        source_files_verified=len(p['source_sha256']),catalog_glossary_and_problem_records_unchanged=True,
        runtime_and_graph_unchanged=True,tracker_plan_unchanged=True,
        scope='Static protocol, provenance and manifest checks only. No model, tracker, scoring, HOTA or hypothesis execution.',
        protocol_sha256=sha(OUT/'protocol.json'),document_sha256=sha(OUT/'ПРОТОКОЛ.md'),validator_sha256=sha(__file__))
    with (OUT/'validation.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
