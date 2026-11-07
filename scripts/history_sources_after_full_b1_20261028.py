"""Готовность реальных источников новой матрицы после полного аудитаБ1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    f=RUN/'history_sources';p=read(f/'protocol.json')
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    audit=RUN/'b1_tracker_audit';c=read(audit/'controller_result.json');a=read(audit/'report.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert a['passed'] and a['full_series'] and a['videos']==20
    old=read(ROOT/'runs/b1_real_inputs_20261026/protocol.json');rows=[]
    for split in ['fit84','fit20']:
        for v in old[split]:
            paths=dict(video=f'data/bdd100k/videos/train/{v}.mov',detections=f'runs/train_cache/{v}/detections/raw_video.npz',
                observations=f'runs/b1_real_inputs_20261026/inputs/{v}.json',appearance=f'runs/z13_masa_regions_20261026/recovery1/videos/{v}.npz',retinal=f'runs/z13_data/{v}.npz')
            rows.append(dict(video=v,split=split,paths=paths,exists={k:(ROOT/path).is_file() for k,path in paths.items()}))
    write(f/'report.json',dict(preparation_completed=True,inventory=rows,available_video_counts={k:sum(r['exists'][k] for r in rows) for k in rows[0]['exists']},
        ready_for_training=False,models_trained=0,hota_measured=False,full_b1_audit_sha256=sha(audit/'report.json'),updated=now()))

if __name__=='__main__':main()
