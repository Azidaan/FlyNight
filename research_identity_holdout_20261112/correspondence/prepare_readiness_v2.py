"""Strict video-unexposed A selection, metadata only; B stays unopened."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,gzip
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from metadata import readiness,read,sha,relative,ROOT
OUT=Path(__file__).resolve().parent
selected=ROOT/'research_identity_holdout_20261112/review/selected_events_A_v2.json';selection=read(selected);native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json');cases=[];inputs={relative(selected):sha(selected)};failures=[]
for archive_name in sorted({e['event_archive'] for e in selection['events']}):
    archive=ROOT/archive_name
    with gzip.open(archive,'rt',encoding='utf8') as f:events=json.load(f)['events']
    inputs[relative(archive)]=sha(archive);byframe={e['frame']:e for e in events}
    for eventmeta in [e for e in selection['events'] if e['event_archive']==archive_name]:
        event=byframe[eventmeta['frame']]
        for proposalmeta in eventmeta['proposal_keys']:
            matching=[p for p in event['proposals'] if all(p[k]==proposalmeta[k] for k in ['old_id','raw_id','row','column','reference_frame'])];assert len(matching)==1;proposal=matching[0]
            try:
                ready=readiness(eventmeta['video'],eventmeta['frame'],proposal);ready['case_key']=eventmeta['event_key']+'|'+str(proposal['old_id'])+'|'+str(proposal['raw_id']);ready['event_key']=eventmeta['event_key'];ready['event_archive']=relative(archive);ready['target_row']=proposal['row'];ready['target_column']=proposal['column'];ready['video_path']=native['videos'][eventmeta['video']]['video_path'];cases.append(ready)
                for ep in ready['endpoints']:
                    for m in ep['maps']:
                        inputs[m['path']]=m['sha256']
                        if 'metadata' in m:inputs[m['metadata']]=m['metadata_sha256']
                    for view in ep['views']:
                        for item in view['files']:inputs[item['path']]=item['sha256']
                if ready['clock_path']:inputs[ready['clock_path']]=sha(ROOT/ready['clock_path'])
            except Exception as ex:failures.append(dict(event_key=eventmeta['event_key'],proposal_key=proposalmeta,error=repr(ex)))
    del byframe,events
out=dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),selection_path=relative(selected),selection_sha256=sha(selected),cases=cases,failures=failures,inputs=inputs,summary=dict(events=len(selection['events']),selected_proposals=sum(len(e['proposal_keys']) for e in selection['events']),ready_cases=len(cases),retinal_ready_cases=sum(all(e['retinal_ready'] for e in c['endpoints']) for c in cases),both_endpoint_anchors=sum(c['evaluation_ready'] for c in cases),current_anchor=sum(c['current_annotation_anchor'] is not None for c in cases),reference_anchor=sum(c['reference_annotation_anchor'] is not None for c in cases),unique_reference_contexts=len({c['endpoints'][0]['key'] for c in cases}),necessary_core_passes=2*len({c['endpoints'][0]['key'] for c in cases})),scope='Strict A v2 metadata and hashes only. No outcomes, B not read. Initial85 readiness is retained as rejected draft exposure audit. Evaluation semantics decided independently by review agent.')
with (OUT/'readiness_v2.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print(json.dumps(out['summary'],ensure_ascii=False));print('FAILURES',len(failures))
