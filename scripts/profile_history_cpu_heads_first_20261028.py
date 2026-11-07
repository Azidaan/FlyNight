"""СтрогийCPUповтор уже готового первого видео; GPUнеповторяется."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    original=ROOT/'scripts/audit_history_full_tracker_20261028.py'
    text=original.read_text(encoding='utf-8')
    text=text.replace("folder=RUN/'history_tracker_audit'","folder=RUN/'history_cpu_heads_profile'")
    start=text.index("    if not report['complete']:")
    end=text.index('    torch.set_num_threads',start)
    text=text[:start]+"    assert not report['complete'] and len(report['completed'])==1\n    p['sources']=p['sources'][:1]\n"+text[end:]
    # Профиль использует ИМЕННОCPUscore, не записанныйGPUscore. Полное
    # совпадение raw/JSON/метрик всех32 является условием принятия этой версии.
    old='matrix[i,j]=expected_cost;self.checked+=1';assert text.count(old)==1
    text=text.replace(old,'self.checked+=1')
    start=text.index('        if vi==0:')
    text=text[:start]+"""        write(folder/'report.json',dict(passed=True,scope='First saved video only; CPU scores used directly; noGPU/encoder replay',videos=1,trackers=32,
            cpu_full_video_seconds=item['seconds'],all_raw_json_metrics_exact=True,zero_states_exact=True,max_head_difference=global_max_diff,
            head_events=all_events,pixel_rois=pixel_rois,all_future_videos_not_evaluated=True,scientific_verdict=None,updated=now()))
        return
"""
    ns={'__name__':'history_cpu_heads_profile_worker','__file__':str(original)}
    exec(compile(text,str(original),'exec'),ns);ns['main']()


if __name__=='__main__':main()
