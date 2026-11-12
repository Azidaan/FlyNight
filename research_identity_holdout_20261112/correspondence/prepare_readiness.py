"""Metadata-only preparation for fixed, independently selected A events."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,gzip,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from metadata import readiness,read,sha,relative,ROOT
OUT=Path(__file__).resolve().parent
selected=ROOT/'research_identity_holdout_20261112/review/selected_events_A.json';selection=read(selected);native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json');archives={};cases=[];inputs={relative(selected):sha(selected)};failures=[]
for eventmeta in selection['events']:
    archive=ROOT/eventmeta['event_archive']
    if str(archive) not in archives:
        with gzip.open(archive,'rt',encoding='utf8') as f:archives[str(archive)]=json.load(f)['events']
        inputs[relative(archive)]=sha(archive)
    event=next(e for e in archives[str(archive)] if e['frame']==eventmeta['frame'])
    for proposalmeta in eventmeta['proposal_keys']:
        matching=[p for p in event['proposals'] if all(p[k]==proposalmeta[k] for k in ['old_id','raw_id','row','column','reference_frame'])];assert len(matching)==1
        proposal=matching[0]
        try:
            ready=readiness(eventmeta['video'],eventmeta['frame'],proposal)
            ready['case_key']=eventmeta['event_key']+'|'+str(proposal['old_id'])+'|'+str(proposal['raw_id']);ready['event_key']=eventmeta['event_key'];ready['event_archive']=relative(archive);ready['target_row']=proposal['row'];ready['target_column']=proposal['column'];ready['video_path']=native['videos'][eventmeta['video']]['video_path'];cases.append(ready)
            for ep in ready['endpoints']:
                for m in ep['maps']:
                    inputs[m['path']]=m['sha256']
                    if 'metadata' in m:inputs[m['metadata']]=m['metadata_sha256']
                for v in ep['views']:
                    for item in v['files']:inputs[item['path']]=item['sha256']
            if ready['clock_path']:inputs[ready['clock_path']]=sha(ROOT/ready['clock_path'])
        except Exception as ex:
            failures.append(dict(event_key=eventmeta['event_key'],proposal_key=proposalmeta,error=repr(ex)))
out=dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),selection_path=relative(selected),selection_sha256=sha(selected),cases=cases,failures=failures,inputs=inputs,summary=dict(events=len(selection['events']),selected_proposals=sum(len(e['proposal_keys']) for e in selection['events']),ready_cases=len(cases),retinal_ready_cases=sum(all(e['retinal_ready'] for e in c['endpoints']) for c in cases),aligned_evaluation_cases=sum(c['evaluation_ready'] for c in cases),unique_reference_contexts=len({c['endpoints'][0]['key'] for c in cases}),necessary_core_passes=2*len({c['endpoints'][0]['key'] for c in cases})),scope='Metadata and hashes only, no Census costs/head outcomes/appearance ranks. Reserved B not read. Selection scope is repair-choice-video-disjoint, not globally unseen historical cohort.')
with (OUT/'readiness.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print(json.dumps(out['summary'],ensure_ascii=False));print('FAILURES',len(failures))
