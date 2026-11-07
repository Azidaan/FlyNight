"""Выбрать пороги только по HOTA разработочной части после фиксации шести моделей."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
RUN=BASE/'dev_threshold'

def prepare():
    assert not RUN.exists()
    learning=BASE/'admission_learning';result=read(learning/'report.json');assert result['complete']
    controller=read(learning/'controller_result.json');assert controller['exit_code']==0 and controller['all_owned_ended']
    sources=['scripts/select_recovery_admission_threshold_20261027.py',str((learning/'report.json').relative_to(ROOT))]
    for m in result['models']:
        for name in [m['checkpoint'],m['normalization'],Path(m['checkpoint']).stem+'_dev_probabilities.npy']:sources.append((learning/name).relative_to(ROOT).as_posix())
        assert sha(learning/m['checkpoint'])==m['checkpoint_sha256'] and sha(learning/m['normalization'])==m['normalization_sha256']
    for item in result['video_rows']['dev']:
        folder=BASE/'admission_dataset/dev'/item['video']
        sources += [(folder/n).relative_to(ROOT).as_posix() for n in ['records.json','baseline.json','supported.json']]
    p=dict(parent_experiment='recovery_admission_learning_v1',created=datetime.now(timezone.utc).isoformat(),budget_seconds=1800,
        author_stop=read(BASE/'authorization.json')['save_at'],device='cpu',thresholds=[.5,.7,.9],
        action='Выбрать порог каждогофиксированногосида/вариантатольконаполнойdevHOTAизранееуказаннойсетки[0.5,0.7,0.9]. '
               'Цель—максимальнаясредняяHOTAповидео,приравенствебольшийпорог. Никакоговыборапофинальномуval. '
               'Всеисходныерамкисохраняются,фильтруютсятолькокандидатыпрежнегоправила. Неизвестныеметкиврешениинепередаются.',
        acceptance=['Все35dev,3сидадвухвариантов,тожесохранённоебазовоеисправило,равныевходы',
                    'Пороги/весафиксируютсядоval,исходныйвыходпобайтновсоставеlabelsпрефикса,уникальныеID',
                    'Данныеdevвиделдетектор;этовыборполитикиобучения,неоценкаобобщениядетектораилиитоговаяпольза'],
        cost_gate='Первое полноеdev-видеосовсемивариантами:2*стоимость*34+120с<остаток1800с,иначестопбезпродления.',
        sha256={n:sha(ROOT/n) for n in set(sources)})
    write(RUN/'protocol.json',p)
    job=dict(name='060_dev_threshold',folder=RUN.relative_to(ROOT).as_posix(),script='scripts/select_recovery_admission_threshold_20261027.py',
        script_sha256=sha(__file__),protocol_sha256=sha(RUN/'protocol.json'))
    write(BASE/'ready_jobs/060_dev_threshold.json',job)
    from scripts.b123_safe_journal_20261027 import journal
    journal('recovery_admission_learning_v1','порог допуска выбирается на полной разработочной HOTA до val',
        '*Сделано.* Все6моделей(3сида×2варианта)обучены;весафиксируютсяпохешамв dev_threshold/protocol.json. '
        'Этоещёнезавершённаяпроверкапользы. '+p['action']+' '+p['cost_gate']+' '
        'Ступеньвходитвзаранеезарегистрированную recovery_admission_learning_v1,не новаягипотеза. '
        'Никакойподгонкипоval,невоспроизведениеполнойBUSCA,ядроFlyVisнеиспользуется.')
    print('Prepared fixed-grid development HOTA threshold selection; no val used')

def run():
    import numpy as np
    from evaluation.hota_eval import evaluate_hota
    p=read(RUN/'protocol.json');learning=read(BASE/'admission_learning/report.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic()
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    variants={};probabilities={}
    for model in learning['models']:
        name=Path(model['checkpoint']).stem
        prob=np.load(BASE/'admission_learning'/(name+'_dev_probabilities.npy'));assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        probabilities[name]=prob
        for threshold in p['thresholds']:variants[name+'_t'+str(threshold)]=dict(model=name,threshold=threshold)
    per={name:{} for name in ['baseline','rule',*variants]};timing={}
    def status(**kw):
        if time.time()>end-3 or (RUN/'stop.request').exists():raise TimeoutError('Development threshold budget')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    for index,item in enumerate(learning['video_rows']['dev']):
        began=time.monotonic();video=item['video'];folder=BASE/'admission_dataset/dev'/video
        base=read(folder/'baseline.json');rule=read(folder/'supported.json');records=read(folder/'records.json')
        assert len(records)==item['count'];by_model={}
        for name,prob in probabilities.items():
            scores=prob[item['start']:item['start']+item['count']];assert len(scores)==len(records);lookup={}
            for record,value in zip(records,scores):lookup.setdefault(record['reference'],{})[record['id']]=float(value)
            by_model[name]=lookup
        for name in per:
            status(video=video,index=index,total=35,variant=name)
            if name=='baseline':rows=base
            elif name=='rule':rows=rule
            else:
                spec=variants[name];lookup=by_model[spec['model']];rows=[]
                for ref,(b,s) in enumerate(zip(base,rule)):
                    assert s['labels'][:len(b['labels'])]==b['labels'];candidates=s['labels'][len(b['labels']):]
                    assert len(lookup.get(ref,{}))==len(candidates)
                    extra=[c for c in candidates if lookup[ref][c['id']]>=spec['threshold']]
                    out=copy.deepcopy(b);out['labels']+=copy.deepcopy(extra)
                    assert out['labels'][:len(b['labels'])]==b['labels'] and len({q['id'] for q in out['labels']})==len(out['labels'])
                    rows.append(out)
            write(RUN/name/(video+'.json'),rows);per[name][video]=evaluate_hota(RUN/name,[video],split='train')
        timing[video]=time.monotonic()-began;write(RUN/'per_video.json',per)
        if index==0:
            cost=2*timing[video]*34+120;available=end-time.time()
            write(RUN/'price.json',dict(first_full_video_seconds=timing[video],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    chosen=[];scores={name:float(np.mean([m['HOTA'] for m in values.values()])) for name,values in per.items()}
    for model in learning['models']:
        name=Path(model['checkpoint']).stem
        best=max(p['thresholds'],key=lambda threshold:(scores[name+'_t'+str(threshold)],threshold))
        chosen.append(dict(**model,threshold=best,dev_mean_hota=scores[name+'_t'+str(best)]))
    result=dict(complete=True,chosen=chosen,mean_per_video_hota=scores,per_video=per,timing=timing,
        val_used_for_selection=False,weights_frozen_before_val=True,selection_is_not_final_benefit=True,seconds=time.monotonic()-started)
    write(RUN/'report.json',result);write(BASE/'frozen_for_validation.json',dict(models=chosen,source_report_sha256=sha(RUN/'report.json'),changed_on_val=False))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
