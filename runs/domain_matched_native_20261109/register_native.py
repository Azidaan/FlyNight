"""Preregister the actual-ID pilot before any new trajectory or metric."""
import os,sys,json,hashlib,subprocess,time,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'runs/full_template_native_20261109';TRAIN=ROOT/'research_domain_matched_reader_20261109'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(900,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.domain_native.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
assert not (OUT/'protocol.json').exists()
assert read(TRAIN/'final_checks.json')['passed'] and read(TRAIN/'summary.json')['gate'] and read(TRAIN/'audit.json')['passed']
delivery=read(OUT/'delivery_report.json');assert delivery['passed'] and delivery['summary']['accepted_above_float32_limit']==delivery['summary']['global_assignments_changed']==0
old=read(OLD/'protocol.json');oldphase=read(OLD/'execution_protocol.json');trained=read(TRAIN/'execution_protocol.json')
shared=['tracking/observed_spatial_reader.py','tracking/full_template_correspondence.py','tracking/channel_identity_head.py','tracking/hex_linear_interpolation.py',
    'tracking/channel_balanced_correspondence.py','tracking/z13_readout.py','tracking/foveated_eyes.py','tracking/offline_association.py',
    'scripts/z13_target_model.py','runs/flyvis/results/flow/0000/000/best_chkpt','runs/z13_target_center_dev_20261018/C/normalization.json','runs/z13_target_center_dev_20261018/R1/normalization.json']
for name in shared:
    digest=sha(ROOT/name);assert oldphase['sources'][name]==digest,name
    if name in trained['sources']:assert trained['sources'][name]==digest,name
videos=old['videos'];assert len(videos)==20 and not set(videos)&set(read(TRAIN/'protocol_v2.json')['videos'])
assets={};manifest=read(OLD/'artifacts_sha256.json')['files']
for folder in ['features','retinal']:
    for path in (OLD/folder).rglob('*.npy'):
        name=path.relative_to(ROOT).as_posix();digest=sha(path);assert manifest[name]==digest,name
        item=dict(source=name,sha256=digest,target=path.relative_to(OLD).as_posix(),kind=folder)
        if folder=='features':
            meta=path.with_suffix('.json');m=read(meta);assert m['sha256']==digest and m['execution_sha256']==sha(OLD/'execution_protocol.json')
            key=hashlib.sha256(json.dumps([m['video'],m['sequence']],separators=(',',':')).encode()).hexdigest()[:32]
            assert path.name.startswith(key+'.');item.update(metadata=meta.relative_to(ROOT).as_posix(),metadata_sha256=sha(meta))
        assets[name]=item
save('cache_registry.json',dict(passed=True,assets=assets,old_execution_sha256=sha(OLD/'execution_protocol.json'),
    recipe='Exact closed native C34/R1 and eye arrays, same inputs,checkpoint,normalization,old readout source. Copied once into new cache, preserving origin metadata. No network or retina rerun for these assets.'))
destination=OUT/'deployment.json';assert not destination.exists();shutil.copyfile(TRAIN/'deployment.json',destination);assert sha(destination)==sha(TRAIN/'deployment.json')
files=[Path(__file__),OUT/'runtime_adapter.py',OUT/'native_run.py',OUT/'prepare_cache.py',OUT/'native_queue.py',OUT/'check_records.py',OUT/'cache_registry.json',destination,
    OUT/'delivery_report.json',TRAIN/'protocol_v2.json',TRAIN/'summary.json',TRAIN/'audit.json',TRAIN/'final_checks.json',TRAIN/'deployment.json',
    OLD/'native_run.py',OLD/'native_runtime.py',OLD/'common.py',OLD/'sparse_views.py',OLD/'sequential_views.py',OLD/'zero_complete.json',OLD/'input_witness_recovery.json',
    OLD/'comparison_report.json',OLD/'id_fate.json',OLD/'artifacts_sha256.json',ROOT/'datasets/bdd100k_loader.py']
files += [ROOT/name for name in shared]
files += [ROOT/name for name in oldphase['sources'] if name.endswith('.py') and not name.startswith('runs/full_template_native_20261109/')]
for video,entry in videos.items():
    for name,digest in entry['sources'].items():assert sha(ROOT/name)==digest,name;files.append(ROOT/name)
    files.append(ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json')
    for arm in ['G_R1','G_FLYVIS','G_PERM']:
        path=ROOT/f'runs/identity_assignment_20261108/full_precision/rollouts/{arm}/{video}.receipt.json'
        if path.exists():files.append(path)
sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}
protocol=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
    authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),hard_stop='2026-11-09T23:59:00+03:00',sources=sources,
    question='Does domain-matched training of the unchanged full-map reader preserve real oldIDs and reject foreign continuations in the actual global native tracker?',
    action='Only the four deployed model coefficients/geometry normalization change from the previous91pair heads to the fixed26pair working-domain heads. Keep old reader,19shifts,core,detector,alpha0.2,beta0,match0.8,float32cost,domain5or6 and full native lifecycle.',
    matched_raw_geometry='Add G_DOMAIN: SAME observable eligibility/domain, but constant positive verifier, no learned head/image/core. It isolates merely applying the geometry cost correction. It differs from the older cached legacy_G, whose eligibility was broader; report that old result secondarily.',
    training='26directlyconfirmedeligibleedges,20own/6foreign,20trainingvideos disjoint from all20nativevideos. These are development sets. No threshold or fit changes from protocol_v2. Core frozen.',
    arms=['G7','G_DOMAIN','R134_spatial','R134_temporal','C34'],videos=videos,native_config=old['native_config'],alpha=.2,beta=0,
    zero_witness_videos=[next(iter(videos)),list(videos)[-1]],
    zero_gate='Reuse the40readyfull20disabled/empty controls with identical source/input passports. Supplement with two predefined COMPLETE-video active-empty witnesses for this metadata/model-location adapter. Exact fullrawarrays and annotatedJSON. Association functions and archived rollout AST are literally inherited; new metadata does not enter costs. Do not repeat ready core,baseline metrics or full zero suite.',
    input_gate='Old21exactretinalwitnesses/20videos and oldcachedarrays retained. Closed map/retinal assets reused under new explicit cache receipts. Restore only genuinely missing causal sequence maps; never rebuild deleted full processed videos.',
    provenance='Fix first-confirmation logger origins using existing actual-current native pointers or unique exact frame/confidence/class. Ambiguity is rawID-1and historyrawNone, never nearest-box or invented label. Adds metadata only; zero gate checks unchanged tracking. No GT/ID/owner is a head feature.',
    statistics=dict(unit='video',bootstrap_draws=20000,seed=42,primary=['C34_vs_native_B0','C34_vs_G7','C34_vs_R134_spatial','C34_vs_R134_temporal','C34_vs_G_DOMAIN'],
        intervals='Same five-comparison family as the previous pilot, two-sided99%percentilevideo-clusterintervals; official pooled TrackEval components, not averagingvideoHOTApercentages. Original oldC34/G7/legacyG reported as secondary95%comparisons for provenance and training attribution.',
        comparability='The raw geometry primary control now has EXACT same domain and cost as the learned arms; older legacy_G has a broader domain and remains a disclosed secondary reference.'),
    acceptance=[dict(criterion='Exact two new full wrapper witnesses plus40readyall20zero controls and unchanged shared-source/recipe fingerprints; core weights unchanged; no ready metric or core work repeated.',met=None),
        dict(criterion='AllfiveprimaryC34lowerHOTAbounds>0; nonnegative pointAssA/IDF1 differences and no totalIDSw/FP/FNincrease against native and matched controls. Otherwise no confirmed addedFlyVis or accepted final tracker. OldC34improvement alone insufficient.',met=None),
        dict(criterion='Persist complete proposals/refusals,fullglobalmatrix/assignment,publicoldID,duplicates and actual futureIDfate for allfivearms; direct labels only at actualannotationframes with unique confirmed sources. Retain unknown and separately report earlier return versus genuinely absent native return within observed video.',met=None)],
    stop='No independent final test before the actual native criteria pass. Do not repair threshold,alpha,model,reader or dtype after viewing these results. Unexpected newcost1delivery cases are evidence for a separate matched intervention, not an in-place change.',
    future_ID_limit='Known reference/current endpoints do not prove every intermediate identity. A missing observed return is not proof of permanent loss. Numeric oldID values across diverged runs require raw-source correspondence, not equality by number.')
save('protocol.json',protocol)
save('execution_protocol.json',dict(registered=protocol['registered'],head=protocol['head'],sources=dict(sources,**{(OUT/'protocol.json').relative_to(ROOT).as_posix():sha(OUT/'protocol.json')}),
    controls=protocol['arms'],acceptance=protocol['acceptance'],cache='cache_registry.json',save_stop=STOP.isoformat()))
identity='domain_matched_native_identity_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before);assert all(c['id']!=identity for c in catalog['experiments'])
catalog['experiments'].append(dict(id=identity,title='Рабочее обучение FlyVis — нативные назначения и сохранение ID',area='flyvis',state='planned',started='2026-11-09',
    question=protocol['question'],hypothesis='Новый причинный рабочий пул может уменьшить чужие разрешения без потери своих, но результат может исчезнуть при конкуренции и удалении дубликатов. Сильнейшая альтернатива — достаточно геометрической поправки или обычного34канальноговхода.',
    expected='Пять фиксированных вариантов на20прежнихvalвидео, общий решатель, прежняяfloat32доставка и прямой аудит судьбыID. Независимый финальный тест ещё не запускается.',acceptance=protocol['acceptance'],obtained='',interpretation='',decision='',cannot_claim='Предварительная точность17/20своих и2/6чужих не является сохранённымиID. Пилот разработческий, не новый независимый тест. Дефект ядра не установлен.',
    evidence=[dict(kind='file',ref='runs/domain_matched_native_20261109/protocol.json',claim='Условия и критерии до новых траекторий'),dict(kind='file',ref='research_domain_matched_reader_20261109/summary.json',claim='Заранее пройденный допуск на рабочих обучающих связях')],tags=['fly-problem-P1']))
atomic(path,(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before);topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Проверка реальных назначений'];atomic(path,(json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Зарегистрирован нативный пилот рабочего обучения\n\n*Постановка (`'+identity+'`).* Пять20видеовариантов: новыеG7,C34,дваобычных34канальныхвхода и чистаягеометриявТОМЖЕдопускебезголовыG_DOMAIN. Меняются только обучающиеданные/окончательныекоэффициенты; прежниефункциисчитывания,globalassignment,float32,alpha0,2иmatch0,8. Новаяголована213готовыхконтекстахразрешила136рёбер,но0свыходомзапорогfloat32и0измененийназначенияприfloat64; доставкуне менять, `runs/domain_matched_native_20261109/delivery_report.json`. Двафиксированныхполныхсвидетелянулевогоадаптера дополняют40готовых20видеоконтролей, анеповторяютих. Готовые карты иB0метрикипереиспользуются. КритерииHOTA/защиты и прямыхIDдо новыхвыходов: `runs/domain_matched_native_20261109/protocol.json`. ПрежнийGсболее широкимдопускомоставленвторичнымсравнением; основноесравнениепоправкистоимоститеперьполностьюсогласовано. Независимыйфинальныйтестещёнеразрешёнкзапускукритерием.\n\n'
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
print('REGISTERED_NATIVE_WORKING_DOMAIN_PILOT',len(assets),'reused assets',flush=True)
