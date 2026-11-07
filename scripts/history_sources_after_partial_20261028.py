"""Подготовка новой программы после завершения расчёта и проверки19готовых видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now

def main():
    folder=RUN/'history_sources_after_partial';p=read(folder/'protocol.json')
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    previous=RUN/'b1_partial_audit';c=read(previous/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    a=read(previous/'report.json');assert a['passed'] and not a['full_series'] and a['scientific_verdict'] is None
    old=read(ROOT/'runs/b1_real_inputs_20261026/protocol.json');rows=[]
    for split in ['fit84','fit20']:
        for video in old[split]:
            paths=dict(video=f'data/bdd100k/videos/train/{video}.mov',detections=f'runs/train_cache/{video}/detections/raw_video.npz',
                retinal=f'runs/z13_data/{video}.npz',observations=f'runs/b1_real_inputs_20261026/inputs/{video}.json',
                appearance=f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz')
            rows.append(dict(video=video,split=split,paths=paths,exists={k:(ROOT/v).is_file() for k,v in paths.items()}))
    write(folder/'report.json',dict(preparation_completed=True,b1_complete=False,b1_partial_audit_sha256=sha(previous/'report.json'),
        inventory=rows,available_video_counts={k:sum(x['exists'][k] for x in rows) for k in rows[0]['exists']},
        ready_for_training=False,models_trained=0,hota_measured=False,scientific_verdict=None,
        next_action='Реальные причинные входы и общий обучаемый интерфейс, затем отдельная цена. Инвентаризация не измеряет пользу памяти.',updated=now()))

if __name__=='__main__':main()
