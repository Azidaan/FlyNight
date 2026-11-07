"""Независимая сверка ценового отказа без повторения HOTA или модели."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import psutil
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027/sam2_output_pilot'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    output=RUN/'cost_stop_audit.json';assert not output.exists()
    p=read(RUN/'protocol.json');price=read(RUN/'price.json');end=read(RUN/'controller_result.json');completed=read(RUN/'completed_videos.json')
    assert end['exit_code']==1 and end['all_owned_ended'] and not end['timed_out']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    assert price['proceed'] is False and abs(price['conservative_remaining_seconds']-(2*price['first_video_seconds']*19+120))<1e-8
    assert price['conservative_remaining_seconds']>price['available_seconds']
    video=completed['videos'][0];assert len(completed['videos'])==1
    assert read(RUN/'baseline'/(video+'.json'))==read(ROOT/'runs/z13_val20_matched_20261024/raw/xywh/raw'/(video+'.json'))
    assert read(RUN/'baseline/per_video.json')[video]==read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json')['per_video']['baseline'][video]
    for name,h in p['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    result=dict(passed=True,at=datetime.now(timezone.utc).isoformat(),stopped_by_predeclared_cost=True,
        first_video_baseline_exact=True,old_sources_and_inputs_unchanged=True,process_tree_ended=True,
        full20_completed=False,method_rejected_for_accuracy=False,price=price,
        retained_first_video=video,interpretation='Ограничение вычислительной стоимости FP32. Нельзя переносить знак одного видео на всю масочную ветвь.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Cost stop independently verified; first video retained')
if __name__=='__main__':main()
