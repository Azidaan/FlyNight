"""Причинные пятикадровые клипыБ2; ниGPU,ниповторастаройградиентнойпробы."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    folder=RUN/'b2_causal_manifest';sources=read(RUN/'history_observed_export/report.json')['completed']
    entries=[]
    for source in sources:
        v=source['video'];alignment=ROOT/f'runs/train_cache/{v}/alignment.json';gt=ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json'
        matches=read(alignment)['matches'];truth=read(gt)
        assert len(matches)==len(truth)
        frames=[int(m['video_frame_index']) for m in matches]
        assert all(b>a for a,b in zip(frames,frames[1:]))
        assert all(truth[int(m['reference_index'])]['name']==m['reference_name'] for m in matches)
        clips=[dict(input_frames=frames[i-4:i+1],target_reference_index=int(matches[i]['reference_index']),target_reference_name=matches[i]['reference_name']) for i in range(4,len(matches))]
        assert all(len(c['input_frames'])==5 and c['input_frames'][-1]==max(c['input_frames']) for c in clips)
        path=folder/'videos'/(v+'.json');write(path,dict(video=v,split=source['split'],video_path=f'data/bdd100k/videos/train/{v}.mov',alignment=alignment.relative_to(ROOT).as_posix(),gt_only_detection_targets=gt.relative_to(ROOT).as_posix(),clips=clips,alignment_sha256=sha(alignment),gt_sha256=sha(gt)))
        entries.append(dict(video=v,split=source['split'],clips=len(clips),manifest=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    assert len(entries)==104 and sum(e['split']=='fit84' for e in entries)==84 and sum(e['split']=='fit20' for e in entries)==20
    report=dict(passed=True,complete=True,entries=entries,clips=sum(e['clips'] for e in entries),train_videos=84,dev_videos=20,video_count=104,
        image_or_feature_exported=False,training_started=False,gpu_used=False,hota_measured=False,scientific_hypothesis_tested=False,
        scope='Manifest only; real input frames chosen only by saved query schedule,GT only last-frame loss targets;no validation video/model selection',updated=now())
    write(folder/'report.json',report)
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b2_causal_manifest_v1')
    card.update(state='infrastructure',done='2026-10-28',obtained='runs/continuation_20261028/b2_causal_manifest/report.json:104историческихtrain/dev,клипов='+str(report['clips'])+',причинныепятикадровыеуказатели;безGPU/экспортапризнаков/обучения/HOTA.',
        interpretation='Подготовлена единица данных Б2. Это не повтор старых трёх шагов градиента и не научная проверка гипотезы.',
        decision='ДоGPUзафиксироватьсопоставимыйреальныйпутьYOLO/ветви/контролей,разрешение/частоту/метки/обучение/стоимость/полнуюHOTA;припродолженииматрицыисторииБ2остаётсяожидатьGPU.',
        cannot_claim='НетобученныхвесовБ2/экспортаP3/детекторнойточности/HOTA. TrainвиделYOLO;точностьдетекторатольконаval. Б1/Б2/Б3/параллельнуюматрицуисториине смешивать.')
    for a in card['acceptance']:a.update(met=True,evidence='runs/continuation_20261028/b2_causal_manifest/report.json')
    write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal('b2_causal_manifest_v1','Подготовлены реальные причинные указатели пятикадровых клипов Б2',
        card['obtained']+' '+card['interpretation']+' Старые3градиентныхшагаБ2неповторялись. GPUдляматрицыисториине занимался;новыйдолгийрасчётБ2этимшагомне разрешается.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300).returncode
        assert code==0


if __name__=='__main__':main()
