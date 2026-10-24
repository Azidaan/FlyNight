"""Independently replay all 36 checkpoints and quota metrics before recording."""
from pathlib import Path
import os,sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from scripts import z13_competing_candidates_20261024 as S
from scripts.z13_record_filter_descriptor_20261024 import ended


def main():
    import numpy as np,torch
    torch.cuda.is_available=lambda:False;torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_candidate_ranker import make_ranker
    from tracking.z13_competing_candidates import choice_rows,quota_precision
    from scripts.z13_candidate_ranker_20261021 import pack,predict
    from scripts.z13_candidate_identity_20261021 import load
    from scripts.z13_continue import paired,per_video_useful
    assert not (S.RUN/'finish_recorded.json').exists();ended(S.RUN);p=S.read(S.RUN/'protocol.json');S.P.verify_sources(p);r=S.read(S.RUN/'report.json')
    assert all(S.sha(ROOT/n)==h for n,h in S.read(S.RUN/'cache_manifest.json').items())
    assert len(S.read(S.RUN/'reproduction.json'))==6 and S.read(S.RUN/'timing.json')['old32_loss_max_abs']<=1e-6
    ev=S.read(S.CACHE/'evaluate/labels.json');models={};rows={};precision={};centers={}
    for arm in S.ARMS:
        seq,_=load('evaluate',arm);ea=pack(ev,seq)
        for mode,activity in (('activity',True),('geometry',False)):
            for kind,folder in (('joint',S.RUN),('bce_common',S.OLD)):
                predictions=[]
                for seed in (0,1,2):
                    sd=folder/arm/mode/f'seed{seed}';meta=S.read(sd/'training.json');assert meta['steps']==1150 and meta['presented_rows']==145950
                    assert S.sha(sd/'model.pt')==meta['model_sha256'];models[sd.relative_to(ROOT).as_posix()]=meta['model_sha256']
                    model=make_ranker(torch,activity);model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True));predictions.append(predict(torch,model,ea))
                key=arm+'_'+kind+'_'+mode;rr=choice_rows(ev,ea,np.mean(predictions,0))
                if kind=='joint':assert rr==S.read(S.RUN/arm/mode/'ensemble_rows.json')
                rows[key]=rr
            rows[arm+'_bce_native_'+mode]=S.read(S.OLD/arm/mode/'ensemble_rows.json')
    for key,rr in rows.items():
        precision[key],detail=quota_precision(rr);centers[key]=per_video_useful(rr)
        assert precision[key]==r['precision_per_video'][key] and detail==r['quota_details'][key] and centers[key]==r['centers_per_video'][key]
    for target,field in ((precision,'paired_precision'),(centers,'paired_centers')):
        for key,value in r[field].items():
            a,b=key.split('-');assert paired(target[a],target[b])==value
    d=r['paired_precision']['D_joint_activity-D_bce_common_activity'];controlkeys=['D_joint_activity-D_joint_geometry','D_joint_activity-R1_joint_activity']
    criteria=[True,d['n']==26 and d['mean_per_video']>=.05 and d['ci95'][0]>0,all(r['paired_precision'][k]['n']==26 and r['paired_precision'][k]['ci95'][0]>0 for k in controlkeys) and r['paired_centers']['D_joint_activity-D_bce_common_activity']['mean_per_video']>=0]
    assert all(criteria)==r['primary_pass'];state='completed_positive' if all(criteria) else 'completed_negative'
    cp=ROOT/'experiments/project_history.json';cat=S.read(cp);card=next(c for c in cat['experiments'] if c['id']==S.IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source=(S.RUN/'report.json').relative_to(ROOT).as_posix()
    summary=('Совместная функция ошибки прошла заданные поисковые критерии;перенос ещё не измерен.' if all(criteria) else 'Эта совместная функция ошибки не прошла заданные поисковые критерии;действующий считыватель сохраняется.')
    fields=dict(summary=summary,obtained=source+' → paired_precision/paired_centers/summary/quota_details: '+json.dumps(dict(precision=r['paired_precision'],centers=r['paired_centers']),ensure_ascii=False)+'.18новыхмоделей и18старыхконтролей независимо воспроизведены из весов;все первичные строки и квоты проверены.',interpretation=summary+' Причина прежних неудач этим опытом не устанавливается; совместная нормировка доступна обоим сравниваемым вариантам.',decision='Сохранить все исходы и3сида;не выбирать порог/сид поtrain26. '+('Следующая ступень требует отдельного протокола причинного применения.' if all(criteria) else 'Не переносить эту версию в HOTA;выбрать следующий узел по измерениям и первоисточникам.'))
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=f'{S.now():%Y-%m-%d}',commit=head,field=key,old=card[key],reason='Полный независимо воспроизведённый исход функции ошибки.'))
        card[key]=value
    card.update(state=state,done=f'{S.now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence=source+'; reproduction.json; timing.json; controller_result.json; finish_recorded.json')
    card['metrics']=[dict(key='known_useful_selected_fraction',value=r['summary']['D_joint_activity']['known_precision_mean_video'],baseline=r['summary']['D_bce_common_activity']['known_precision_mean_video'],subset='Повторные26train,верхние5%по каждому видео;известные выбранные исходы',source=source,locator='summary.D_joint_activity.known_precision_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_joint_activity-D_bce_common_activity','D_joint_activity-D_joint_geometry','D_joint_activity-R1_joint_activity'):
        if r['paired_precision'][key]['n']==26 and r['paired_precision'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=S.IDENT+'_'+key.replace('-','_vs_'),title='Совместная ошибка: '+key,category='representation' if 'R1' in key else 'readout',metric='known_useful_selected_fraction',unit='fraction',comparison=key,subset='Повторные26train,верхние5%каждоговидео,известныеисходы',conditions='Один ранжировщик ификсированныевходы;3сида,1150шагов145950предъявлений;общая нормировкакандидатов и нулевогоотказа. Неизвестные исходы отдельно.',limitations=p['counterargument']+' Частное сравнение не отменяет непройденные основные критерии. НетHOTA.',evidence=dict(source=source,sha256=S.sha(S.RUN/'report.json'),locator=['paired_precision',key])))
    card['evidence'].append(dict(kind='file',ref=source,claim='Полный исход с независимо воспроизведёнными весами/строками/квотами'));S.write(cp,cat)
    baseline=r['summary']['D_bce_common_activity']['known_precision_mean_video'];relative=100*d['mean_per_video']/baseline if baseline else None
    S.journal('полный итог совместной функции ошибки','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nСводка: доляполезныхсредиизвестныхпри5%отборе,DотносительноBCEсобщейнормировкой '+str(relative)+'%;доляпотолкаHOTAнеизмерена.')
    S.write(S.RUN/'finish_recorded.json',dict(at=S.now().isoformat(),state=state,primary_pass=all(criteria),report_sha256=S.sha(S.RUN/'report.json'),models=models,all36_checkpoints_replayed=True,all_quota_rows_verified=True,all_paired_recomputed=True,relative_percent=relative))


if __name__=='__main__':main()
