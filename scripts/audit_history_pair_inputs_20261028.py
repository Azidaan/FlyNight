"""Проверить все причинные указатели/типы/маски, не считать это памятью или HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    import numpy as np
    source=RUN/'history_pair_dataset';folder=RUN/'history_pair_audit';p=read(folder/'protocol.json')
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    c=read(source/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=read(source/'report.json')
    if not r['complete']:
        write(folder/'report.json',dict(passed=False,full_dataset=False,reason='Подготовка остановлена по стоимости; это не научный отрицательный результат',hota_measured=False,updated=now()));return
    allowed={'frame','track','det','box','geometry','history_frame','history_det','history_box','history_quality','history_valid'}
    splits={};total=known=positive=0
    for index,entry in enumerate(r['completed']):
        v=entry['video'];f=source/'videos'/v
        assert sha(f/'inputs.npz')==entry['inputs_sha256'] and sha(f/'targets.npz')==entry['targets_sha256']
        with np.load(f/'inputs.npz') as x,np.load(f/'targets.npz') as y:
            assert set(x.files)==allowed and set(y.files)=={'identity','known'}
            n=len(x['frame']);assert n==entry['pairs'] and x['geometry'].shape==(n,14)
            assert np.isfinite(x['geometry']).all() and np.isfinite(x['box']).all()
            mask=x['history_valid'];assert np.all(x['history_frame'][mask]<np.broadcast_to(x['frame'][:,None],mask.shape)[mask])
            assert np.all(x['history_det'][mask]>=0)
            assert np.all(y['identity'][~y['known']]==-1) and np.all(np.isin(y['identity'][y['known']],[0,1]))
            total+=n;known+=int(y['known'].sum());positive+=int((y['identity']==1).sum());splits[v]=entry['split']
        write(folder/'status.json',dict(state='running',done=index+1,total=len(r['completed']),pairs=total,updated=now()),required=False)
    assert len(splits)==104 and known>0 and positive>0
    report=dict(passed=True,full_dataset=True,videos=104,pairs=total,known=known,positive=positive,unknown=total-known,
        inputs_have_no_gt_fields=True,unknown_not_negative=True,all_history_strict_past=True,baseline_replay_exact=r['baseline_replay_exact'],
        training_ready=False,hota_measured=False,remaining=['Совмещённые изображения движения и текущих областей','Временной банк признаков внешности','Общий обучаемый интерфейс, протокол обучения и стоимость','Полная парная HOTA всех сочетаний'],updated=now())
    write(folder/'report.json',report)
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='history_pair_inputs_v1')
    card.update(state='infrastructure',done='2026-10-28',obtained='runs/continuation_20261028/history_pair_dataset/report.json и history_pair_audit/report.json:полные104историческихtrain/dev, pairs='+str(total)+', known='+str(known)+', positive='+str(positive)+', unknown='+str(total-known)+'. Все временные указатели причинные, GTполятолькоtargets, исходныйCPUтрекерточенкраннемуэкспорту. Обучения/HOTAнет.',
        interpretation='Подготовлены реальные задачи сопоставления с прошлым наблюдением. Это готовность числовых входов и меток, не доказательство пользы истории или ядра.',
        decision='Продолжить реальный экспорт изображений/признаков и общий обучаемый интерфейс1/3/4, затем фактическая цена до обучения. Не переигрывать завершённыйБ1.',cannot_claim='Нет обученной модели/полнойHOTA1/3/4. РабочийIDне истиннаяличность;unknownне negative. Детектор виделэтиtrainвидео;это подготовка и разработка,не независимая оценка детектора.')
    for item in card['acceptance']:item.update(met=True,evidence='runs/continuation_20261028/history_pair_audit/report.json')
    write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_pair_inputs_completed','Реальные пары и причинная история для1/3/4подготовлены',
        'Источник runs/continuation_20261028/history_pair_audit/report.json:104видео,pairs='+str(total)+',known='+str(known)+',positive='+str(positive)+',unknown='+str(total-known)+'. Это входы/метки,не обучение/HOTA/памятьFlyVis. Всеисториистрогопрошлые;разметкавtargets,не входах. Дальшеизображения/внешность/обучаемыйинтерфейс/цена.')
    import subprocess
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0,args

if __name__=='__main__':main()
