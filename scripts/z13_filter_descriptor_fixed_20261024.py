"""Version2: fixed original candidate data, observed-only coefficients, CPU."""
from pathlib import Path
from datetime import datetime
import os,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts import z13_filter_descriptor_cache_20261023 as ORIGINAL
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_filter_descriptor_fixed_20261024/cache';TRAIN=RUN.parent/'train';IDENT='z13_filter_descriptor_fixed_v2'


def journal(title,text):
    old=ORIGINAL.prior.IDENT
    try:ORIGINAL.prior.IDENT=IDENT;ORIGINAL.prior.journal(title,text)
    finally:ORIGINAL.prior.IDENT=old


def transformed():
    source=(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py').read_text(encoding='utf-8')
    pairs=[("os.environ['CUDA_VISIBLE_DEVICES']=''","os.environ['CUDA_VISIBLE_DEVICES']='-1'"),
      ("('seed_features','centers','scores','count')","('seed_features','centers','scores','count','features')"),
      ("r=candidates(score,x,geom);n=len(r['indices']);ref=refs[arm][e]","ref=refs[arm][e];n=int(ref['count']);r=fixed_replay(score,x,geom,ref['centers'][:n],ref['scores'][:n],data[e][2],bank.eye_size);fresh=candidates(score,x,geom);selection_same=bool(len(fresh['indices'])==n and np.array_equal(fresh['indices'],r['indices']))"),
      ("centers=np.asarray([to_frame(c,data[e][2],bank.eye_size) for c in r['centers']])","centers=r['centers']"),
      ("check=dict(episode=e,projection_max_abs=exact,seed_max_abs=seed_error,center_max_px=center_error,score_max_abs=score_error)","feature_error=float(np.abs(r['features']-ref['features'][:n]).max());assert feature_error<=.001;check=dict(episode=e,projection_max_abs=exact,seed_max_abs=seed_error,center_max_px=center_error,score_max_abs=score_error,fixed_candidate_features_max_abs=feature_error,anchor_indices=r['indices'].tolist(),regenerated_selection_equal=selection_same)"),
      ("if not (RUN/'timing.json').exists():","if not (RUN/'timing.json').exists() and not record.get('reused_original',False):"),
      ("estimate=record['seconds']*groups_total*2+180","estimate=record['seconds']*(groups_total-done+1)*2+180")]
    for old,new in pairs:assert source.count(old)==1,old;source=source.replace(old,new)
    return source


def prepare():
    assert not RUN.exists();base=read(ORIGINAL.RUN/'protocol.json');ORIGINAL.prior.verify_sources(base);auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    a=ORIGINAL.RUN/'parity_audit_cpu_20261024';b=ORIGINAL.RUN/'peak_audit_cpu_20261024'
    assert read(a/'controller_result.json')['exit_code']==read(b/'controller_result.json')['exit_code']==0
    assert read(b/'report.json')['all_original_peak_values_within_tolerance']
    names=['scripts/z13_filter_descriptor_fixed_20261024.py','tracking/z13_fixed_candidate_replay.py','tests/test_z13_fixed_candidate_replay.py','scripts/z13_filter_descriptor_fixed_console.ps1',(a/'report.json').relative_to(ROOT).as_posix(),(b/'report.json').relative_to(ROOT).as_posix()]
    p={**base,'id':IDENT,'created':now().isoformat(),'device':'cpu','budget_seconds':18000,'save_at':auth['save_at'],'hard_stop':auth['hard_stop'],'stop_request_path':(RUN/'stop.request').relative_to(ROOT).as_posix(),'authorization_file':'runs/z13_weekend_20261024/authorization.json','original_protocol_sha256':sha(ORIGINAL.RUN/'protocol.json'),'executed_source_sha256':hashlib.sha256(transformed().encode()).hexdigest(),'action':base['action'],
      'parity':'Version2 before outcomes: all immutable cached query candidates/centers/features/scores and labels used by both matched arms remain byte-for-byte unchanged (source hashes). Observed descriptors: projection<=1e-10,seed maxabs<=.001; reproduce original fixed observed anchors count,centers<.5px,scores<.001; additional candidate-feature maxabs<=.001 for new chunks. Do not regenerate candidate sets in the scientific comparison. Original version failed regenerated peak-set replay and remains inconclusive; this version does not assert that predicate is reproduced. Log all regenerated selection differences. Previously successful chunks retain their stricter original regeneration checks and hashes,not recomputed.',
      'cause_evidence':'CPU audit: identical original normalizer,2D/3D and layout; saved old450/195/245 values within2.1e-6px/6e-7score. New418 vs419 has only7.0943e-8 response margin; diagnostic perturbation2e-7 produces exact old candidate indices. Discrete local maxima can change under tiny forward numerical differences. Original GPU map not saved,so its bitwise values and exact historical rounding cannot be proved.',
      'timing':'CPU only, at most18000s under a new author-approved weekend-stage budget. Keep587completed original chunks. First newly computed group:2xremaining_groups_cost+180must fit remaining stage budget. No tolerance weakening, no best-seed or data changes. Training after full verification in separate CPU process,at most1800s.',
      'sha256':{**base['sha256'],**{n:sha(ROOT/n) for n in names}}}
    retained={}
    for path in sorted(ORIGINAL.RUN.glob('*/*/chunk_*.json')):
        r=read(path);assert all(x['projection_max_abs']<=1e-10 and x['seed_max_abs']<=.001 and x['center_max_px']<.5 and x['score_max_abs']<.001 for x in r['checks']);retained[path.relative_to(ROOT).as_posix()]=sha(path)
        copy={**r,'reused_original':True,'original_chunk':path.relative_to(ROOT).as_posix(),'original_sha256':sha(path)};write(RUN/path.relative_to(ORIGINAL.RUN),copy)
    assert len(retained)==587;p['retained_original_chunks']=retained;write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: коэффициенты цели на неизменных кандидатах,версия2',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','DiMP','engineering-future'],question='Помогают ли наблюдаемые коэффициенты различающего фильтра сверх среднего образца при строго неизменных кандидатах,архитектуре и бюджете обучения?',hypothesis=base['counterargument'],expected='Исходные104train/26оценки,те же36CPUмоделей/3сида/1150шагов145950предъявлений. CPUизвлечение18000с,обучение1800с,до26.10 07:40МСК. Версия1 сохраняется с невоспроизведённым дискретным вторичным максимумом.',acceptance=[dict(criterion=p['parity']+' Все9наборов/18старыхмоделей/32шага проверены.',met=None),dict(criterion='D ridge-D mean >=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='D ridge лучше D old activity,D old geometry,R1 ridge,D ridge geometry; все нижние95%ДИ>0.',met=None)],summary='Новая версия технического переноса зарегистрирована до исходов; научная проверка коэффициентов/равного контроля прежняя.',obtained='',interpretation='',decision='',cannot_claim='Не доказано побитовое воспроизведение прежнего GPUпрохода или прежнего выбора вторичных максимумов. Кандидаты научного опыта берутся только из исходного неизменного кэша. Карта уже использует фильтр; нет гарантии новой информации,причинного диагноза всейFlyVis,безопасногоотказа илиHOTA. Повторно используемыеtrain26.',metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Фиксированные данные и критерии до исходов версии2')]))
    write(cp,cat);journal('версия2 описания цели: исходные кандидаты неизменны','*Постановка.* '+p['action']+'\n\n'+p['parity']+'\n\n'+p['cause_evidence']+'\n\nПрежний опыт z13_filter_descriptor_v1 остаётсяinconclusive,старый неуспех не подменяется. Новый заранее заданный служебный контроль проверяет фактически используемые неизменные кандидаты,а не повторное принятие дискретного решения на другой численно полученной карте. Научные критерии2п.п./парныйДИ и сильныеконтроли прежние. Сохранены587исходныхготовыхблоков по хешам,не повторяются. CPU18000с,после полного извлечения отдельныйCPU1800с;безCUDA/загрузок/75отложенных. Протокол runs/z13_filter_descriptor_fixed_20261024/cache/protocol.json.')


def worker():
    import torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_fixed_candidate_replay import replay
    p=read(RUN/'protocol.json');ORIGINAL.prior.verify_sources(p);assert sha(ORIGINAL.RUN/'protocol.json')==p['original_protocol_sha256']
    for n,h in p['retained_original_chunks'].items():assert sha(ROOT/n)==h
    source=transformed();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    ns={'__name__':'fixed_candidate_descriptor_v2','__file__':str(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py'),'fixed_replay':replay};exec(compile(source,ns['__file__'],'exec'),ns);ns.update(RUN=RUN,IDENT=IDENT,journal=journal,T0=time.monotonic())
    ns['worker']();r=read(RUN/'report.json');r.update(verification_policy='Original fixed observed candidate anchors; no claim of identical regenerated secondary maxima.',original_candidate_sources_unchanged=True,original_failed_version_preserved=True);write(RUN/'report.json',r)


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
