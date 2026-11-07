"""Все индексы/хеши/нормы и последовательный повтор реальных пикселей трёх видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    import numpy as np,cv2
    cv2.setNumThreads(1)
    from tracking.causal_motion_history import CausalMotionHistory
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    source=RUN/'history_visual_inputs';folder=RUN/'history_visual_audit';p=read(source/'protocol.json')
    c=read(source/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=read(source/'report.json')
    if not r['complete']:
        write(folder/'report.json',dict(passed=False,full_dataset=False,reason='Неполный кэш по стоимости, не научный отрицательный результат',hota_measured=False,updated=now()));return
    pixel_videos={0,len(r['completed'])//2,len(r['completed'])-1};rois=0;pixel_checked=0
    for vi,entry in enumerate(r['completed']):
        v=entry['video'];f=source/'videos'/v
        assert all(sha(f/k)==digest for k,digest in entry['sha256'].items())
        embeddings=np.load(f/'embeddings.npy',mmap_mode='r');images=np.load(f/'images.npy',mmap_mode='r')
        assert embeddings.shape==(entry['rois'],256) and images.shape==(entry['rois'],3,p['image_size'],p['image_size']) and images.dtype==np.uint8
        assert np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
        pair=RUN/'history_pair_dataset/videos'/v/'inputs.npz';assert sha(pair)==entry['input_sha256']
        with np.load(f/'index.npz') as z,np.load(pair) as x:
            current=z['current_index'];history=z['history_index'];mask=x['history_valid']
            assert np.array_equal(z['roi_frame'][current],x['frame']) and np.array_equal(z['roi_det'][current],x['det'])
            assert np.array_equal(z['roi_frame'][history[mask]],x['history_frame'][mask]) and np.array_equal(z['roi_det'][history[mask]],x['history_det'][mask])
            assert np.all(z['roi_frame'][history[mask]]<np.broadcast_to(x['frame'][:,None],mask.shape)[mask])
            keys=list(zip(z['roi_frame'].tolist(),z['roi_det'].tolist()))
        if vi in pixel_videos:
            video_path=ROOT/f'data/bdd100k/videos/train/{v}.mov';assert sha(video_path)==entry['video_sha256']
            capture=cv2.VideoCapture(str(video_path));assert capture.isOpened()
            cache=load_detection_cache(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz')
            schedule={int(x['video_frame_index']) for x in read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches']}
            at={}
            for i,(frame,det) in enumerate(keys):at.setdefault(frame,[]).append((i,det))
            motion=CausalMotionHistory(**p['motion'])
            for frame in range(max(schedule)+1):
                ok,image=capture.read();assert ok
                if frame not in schedule:continue
                motion.update(image,frame)
                if frame in at:
                    b,_,_=detections_for_frame(cache,frame)
                    for i,det in at[frame]:assert np.array_equal(motion.crop(b[det],image.shape,p['image_size'],p['context']),images[i]);pixel_checked+=1
            capture.release()
        rois+=entry['rois'];write(folder/'status.json',dict(state='running',done=vi+1,total=len(r['completed']),rois=rois,pixel_rois_checked=pixel_checked,updated=now()),required=False)
    write(folder/'report.json',dict(passed=True,full_dataset=True,videos=len(r['completed']),rois=rois,pixel_rois_checked=pixel_checked,weights_frozen=True,training_ready=False,hota_measured=False,updated=now()))
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='history_visual_inputs_v1')
    card.update(state='infrastructure',done='2026-10-28',obtained='runs/continuation_20261028/history_visual_inputs/report.json и history_visual_audit/report.json:полные104видео, rois='+str(rois)+', pixel_rois_checked='+str(pixel_checked)+'. Реальные изображения/причинный след/фиксированные256признаков,безGT/обучения/HOTA.',
        interpretation='Подготовлены общие входы будущих вариантов1/3/4. Изображение давности изменений и замороженный извлекатель сами по себе не доказывают пользу истории.',
        decision='Реализовать одинаково обучаемый интерфейс всех сочетаний и простых контролей; фиксировать потери/разбиения/параметры/стоимость до обучения, затем полный трекер.',
        cannot_claim='Это не обучение/полнаяHOTA1/3/4 и не новая проверкаMASA. Компенсация только переноса камеры, не вращения/перспективы; при ненадёжности сброс/маска. Историческиеtrainвиделдетектор. ВкладядраFlyVisэтимне измерен.')
    for a in card['acceptance']:a.update(met=True,evidence='runs/continuation_20261028/history_visual_audit/report.json')
    write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_visual_inputs_completed','Изображения истории движения и банк фиксированных признаков подготовлены',
        'Источник runs/continuation_20261028/history_visual_audit/report.json:104видео,rois='+str(rois)+',pixel_rois_checked='+str(pixel_checked)+'. Совпали всеиндексы/прошлое/хеши/нормы,пиксели трёхвидеоповторены последовательно. Обучение/HOTAнепроводились. Дальшеобщийобучаемыйинтерфейс/контроли/цена.')
    import subprocess
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0

if __name__=='__main__':main()
