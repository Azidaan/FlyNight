"""Register the actual-ID experiment before input recovery or new rollouts."""
from common import *
start('cpu')
import gzip

proposal_path=ROOT/'research_full_template_native_pilot_20261109/proposal_v2.json'
proposal=read(proposal_path);legacy=read(LEGACY/'protocol.json')
assert read(TRAIN/'original_geometry_audit.json')['passed']
assert read(TRAIN/'original_geometry_results.json')['selected_development_arm']=='C34_full7_channels'
first=next(iter(legacy['videos']));event_path=LEGACY/'rollouts/G_R1'/(first+'.events.json.gz')
with gzip.open(event_path,'rt',encoding='utf8') as f:events=json.load(f)
from sparse_views import SparseViews
available={}
for event in events:
    for row in event['rows']:
        sequence=[[h['frame'],h['crop']] for h in row['history']]+row['search']
        for frame,crop in sequence:
            key=SparseViews.key(frame,crop);path=LEGACY/'retinal'/first/(key+'.npy')
            if path.exists():available[(int(frame),tuple(crop))]=path
candidates=sorted(available);one=next(c for c in candidates if c[0]>=100);two=next(c for c in candidates if c[0]>=200 and c[0]>one[0])
witnesses=[dict(video=first,frame=frame,crop=list(crop),cached=available[(frame,crop)].relative_to(ROOT).as_posix()) for frame,crop in [one,two]]
files=[Path(__file__),OUT/'common.py',OUT/'sparse_views.py',OUT/'witness.py',proposal_path,
    TRAIN/'original_geometry_protocol.json',TRAIN/'original_geometry_results.json',TRAIN/'original_geometry_audit.json',TRAIN/'features.json',
    ROOT/'tracking/retina.py',ROOT/'tracking/full_template_correspondence.py',ROOT/'tracking/channel_identity_head.py',ROOT/'tracking/xywh_bytetrack.py',ROOT/'tracking/spatial_identity_assignment.py',
    ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
    ROOT/'research_identity_assignment_run_20261108/full_planner.py',ROOT/'research_identity_assignment_run_20261108/full_run.py',ROOT/'research_identity_assignment_run_20261108/full_retina.py',ROOT/'research_identity_assignment_run_20261108/full_precision_io.py',
    LEGACY/'protocol.json',event_path]+[ROOT/w['cached'] for w in witnesses]
p=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() if False else '',
    authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),hard_stop='2026-11-09T23:59:00+03:00',
    question='Does the admitted full-template/original-geometry verifier preserve real old IDs and reject foreign continuations after native global assignment and duplicate removal?',
    primary_candidate='C34_full7_channels',training='Exactly91 strict-last pairs from32 historical videos, as frozen. Fit final G7,C34full7,R134spatialfull7,R134temporalfull7 models once,lambda1,threshold0. No val outcome used.',
    videos=legacy['videos'],native_config=legacy['native_config'],alpha=.2,beta=0,car_class=2,
    policy='Filter the existing geometric proposals before GLOBAL native first association. Class2 only, at least2 distinct reference observations and old support>=3. Same reference history,crops,affine19-shift plans for all arms. No support/history -> no additional correction. Undefined visual patch -> same G7 fallback. No alias or output-ID rewrite. Native re_activate,births and duplicate removal.',
    arms=['G7','R134_spatial','R134_temporal','C34'],reused_controls=['native_B0','legacy_unfiltered_G'],
    zero_gate='Exact full raw arrays AND annotated JSON for ALL20 videos, twice: alpha0 disabled and active alpha0.2 with empty accepted votes. Reuse baseline HOTA; source and dtypes pinned.',
    input_gate='Two deterministic cached retinal views from the first fixed video, at/after100 and200, restored both from nearest past checkpoint and from frame0. Require exact array equality and native per-frame dtype. No huge processed-video rebuild. Cache states must not be later than requested input frame.',
    witnesses=witnesses,source_video_sha256=legacy['videos'][first]['sources'][legacy['videos'][first]['video_path']],
    amendment_before_rollout='Дописано до прогона: готовый старый G без проверяющей головы добавлен как пятый первичный контроль общего сдвига стоимости. Прежние четыре сравнения сохранены. Причина — отделить сигнал от уже существующего геометрического выигрыша; исходов нового полного метода ещё нет.',
    primary_comparisons=[['C34','native_B0'],['C34','legacy_unfiltered_G'],['C34','G7'],['C34','R134_spatial'],['C34','R134_temporal']],
    statistics='Exactly same TrackEval setup;20000 paired video draws,seed42;five contrasts use99% family intervals, also descriptive95%. All20 videos fixed; no outcome-based subset or threshold selection.',
    acceptance='Allfive adjusted lower HOTA bounds>0; mean AssA/IDF1 changes nonnegative and no totalIDSw,FP,FN increase relative to native and matched controls. Otherwise no confirmed added FlyVis benefit. Report ordinary-input/geometry behavior separately without crediting core.',
    ID_fate='Record every proposal/refusal,original/corrected matrix,global result,public old IDs,duplicate removal,and subsequent ID trajectory. Direct own/foreign and false-refusal labels only when existing annotations support them; retain unknown cases explicitly.',
    stage='Input restoration now. Actual runtime, deployment and execution-phase source passports will be registered before their use; no rollout or newHOTA yet.',
    sources={f.relative_to(ROOT).as_posix():sha(f) for f in files})
import subprocess
p['head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
write('protocol.json',p)
path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments'])
cat['experiments'].append(dict(id=OUT.name,title='Полный шаблон FlyVis — реальные назначения и судьба ID',area='flyvis',state='running',started='2026-11-09',
    question=p['question'],hypothesis='Сохранённая пространственная опора и замороженная проверка принадлежности могут улучшить реальные продолжения. Альтернатива — весь выигрыш объясняется геометрической поправкой, обычным входом или теряется при глобальном назначении и удалении дубликатов.',
    expected='Точные входы и нулевое вмешательство, затем фиксированный полный пилот20 val-видео с согласованными контролями и прямой судьбойID.',
    acceptance=[dict(criterion=p['input_gate'],met=None),dict(criterion=p['zero_gate'],met=None),dict(criterion=p['acceptance'],met=None),dict(criterion=p['ID_fate'],met=None)],
    obtained='',interpretation='',decision='',cannot_claim='',evidence=[dict(kind='file',ref='runs/'+OUT.name+'/protocol.json',claim='Критерии и добавленный геометрический контроль до новых исходов.')],tags=['fly-problem-P1']))
temporary=path.with_suffix('.native_template.pending');temporary.write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode());temporary.replace(path)
journal('Зафиксирован реальный пилот полного шаблона FlyVis',f'*Постановка (`{OUT.name}`).* Все20 val-видео прежнего фиксированного состава, без пересечения с32 обучающими. Перед реальными назначениями — точность восстановления входов и два полных нулевых контроля. Затем C34 и такая же геометрическая/пространственная/временная проверка, прежниеalpha0,2 и match0,8,штраф1 и порог0; нативный общий решатель и жизненный цикл. Дописан до прогонов готовый старыйG как пятый первичный контроль общего сдвига стоимости; семейные интервалы99%. Прямую судьбуID не достраивать при отсутствии разметки. Большиекадры не восстанавливать, только недостающие небольшие виды с проверкойпо кэшу. Источник `runs/{OUT.name}/protocol.json`. Реальных исходов нового метода ещё нет.')
status('registered',videos=20,witnesses=witnesses)
print('Native-ID protocol registered; first input witnesses',[(w['frame'],w['crop']) for w in witnesses],flush=True)
