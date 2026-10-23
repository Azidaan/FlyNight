from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_effect_path_audit_20261023 import RUN,IDENT,read,write,journal,prior

if __name__=='__main__':
    assert not (RUN/'finish_recorded.json').exists();prior.verify_sources(read(RUN/'protocol.json'))
    r=read(RUN/'report.json');s=r['summary'];unchanged=sum(v['events']['interventions'] for v in r['per_video'].values() if not v['raw']['different_frames'])
    assert s['videos']==35 and s['events']['interventions']==1162 and (s['hota_better'],s['hota_equal'],s['hota_worse'])==(4,31,0)
    analysis=dict(interventions_in_videos_with_identical_raw_outputs=unchanged,raw_changed_frames=sum(len(v['raw']['different_frames']) for v in r['per_video'].values()),raw_geometry_changed_frames=sum(len(v['raw']['geometry_different_frames']) for v in r['per_video'].values()),note='Количество различающихся кадров не равно количеству причинных вмешательств; изменение одного ID может размножиться на дальнейших кадрах.')
    write(RUN/'analysis.json',analysis);cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(x for x in cat['experiments'] if x['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    obtained='runs/z13_effect_path_audit_20261023/report.json → summary: '+json.dumps(s,ensure_ascii=False)+'. analysis.json: '+json.dumps(analysis,ensure_ascii=False)
    interpretation='Большая часть видео не меняет даже видимый сырой выход, несмотря на события поправки. Там, где выход меняется, часть различий относится только к переименованию ID или не отражается в HOTA. Это локализует ограничение эффекта на уровне сохранённых выходов, но не устанавливает причину каждого вмешательства.'
    decision='Не запускать поиск ухудшившегося D-видео против стенда: таких видео здесь нет. Продолжать проверку считывания на train; новую политику или порог по dev не подбирать. Для отдельного причинного разбора нужен журнал одного вмешательства и сопоставлений, а не повторный общий HOTA.'
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if c.get(field):c.setdefault('revisions',[]).append(dict(date=f'{prior.now():%Y-%m-%d}',commit=head,field=field,old=c[field],reason='Итог аудита существующих источников.'))
        c[field]=value
    c.update(state='completed_positive',done=f'{prior.now():%Y-%m-%d}')
    c['acceptance'][0].update(met=True,evidence='runs/z13_effect_path_audit_20261023/report.json; analysis.json')
    c['evidence'].append(dict(kind='file',ref='runs/z13_effect_path_audit_20261023/report.json',claim='Аудит35пар, не новая эффективность модели'));write(cp,cat)
    journal('итог аудита пути эффекта D','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка: нового процента улучшения нет; прежняя HOTA D61,65836372 против61,65477894 (+0,005814% относительно стенда), практический критерий не пройден. Доля потолка HOTA неизвестна. Положительное состояние карточки означает выполненный диагностический критерий, не улучшение модели.')
    write(RUN/'finish_recorded.json',dict(at=prior.now().isoformat(),state='completed_positive',new_model_or_hota_run=False));print(json.dumps(analysis,ensure_ascii=False))
