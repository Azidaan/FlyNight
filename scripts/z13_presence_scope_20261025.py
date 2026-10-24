"""One bounded read-only prerequisite for a target-visibility readout repair."""
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import os
import sys
import time
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_localization_labels_20261023 import verify_sources
RUN = ROOT/'runs/z13_presence_scope_20261025'
SOURCE = ROOT/'runs/z13_refusal_causes_20261025/rows.json'
IDENT = 'z13_presence_scope_v1'
ARMS = ('D', 'C', 'R1')
T0 = time.monotonic()


def journal(title, text):
    from scripts import z13_localization_labels_20261023 as m
    old = m.IDENT
    try:
        m.IDENT = IDENT
        m.journal(title, text)
    finally:
        m.IDENT = old


def prepare():
    assert not RUN.exists(), 'No duplicate or overwritten result'
    old = read(ROOT/'runs/z13_candidate_data104_20261021/protocol.json')
    fit, evaluate = old['train_full104'], old['evaluation26']
    assert len(fit) == 104 and len(evaluate) == 26 and not set(fit) & set(evaluate)
    files = ['tracking/z13_presence_scope.py', 'tests/test_z13_presence_scope.py',
        'scripts/z13_presence_scope_20261025.py', 'scripts/z13_presence_scope_console.ps1',
        'scripts/z13_data.py', 'scripts/z13_spatial_train.py', 'scripts/z13_train.py',
        'tracking/z13_ridge_localizer.py', 'tracking/z13_spatial.py',
        SOURCE.relative_to(ROOT).as_posix(), 'runs/z13_candidate_data104_20261021/protocol.json',
        'scripts/z13_bounded_recovery_controller_20261022.py']
    files += [f'runs/z13_data/{v}.npz' for v in fit+evaluate]
    stop = now()+timedelta(seconds=600)
    assert stop < datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at'])
    p = dict(id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu',
        budget_seconds=600, save_at=stop.isoformat(), hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(), fit104=fit, evaluate26=evaluate,
        action='Candidate repair: replace the current map-height refusal by learned target visibility, preserving its center and original within-present ranking. Before fitting, audit which absence labels are established and test a privileged perfect-binary-visibility priority with that SAME ranking. No learner or weight change in this prerequisite.',
        model='Observed identity -> target visibility -> response peak -> proposed center -> tracker assignment. A correct presence decision need not imply a useful center or useful assignment.',
        rules=['Read actual labels/code before proposing a new cause.', 'Unknown observed identity is not a false initial detection.', 'Fixed inputs, paired video intervals; oracle is not inference or HOTA.'],
        method='Read only metadata arrays for all130 training videos; partition every label into present, annotated target outside square, annotated target outside hex field, target not annotated in query, unknown observed identity/stale, or unknown query annotation gap. Report augmented and natural data separately. No raw hex frames, pixels, dev or heldout loaded. No annotation correction.',
        oracle='On identical known5822 natural queries forD/C/R1 select ceil(1%*5822)=59. Base sorts original score; visibility-priority sorts kind1 first, then original score, then label. Unknown2504 stay excluded from BOTH arms and separately counted, never relabeled absent. This is an oracle diagnostic restricted to the known cohort, not a deployable policy or an upper bound on all possible continuous visibility scorers. Separately, suppress kind0 only in the unchanged actual top1% selection without refilling to isolate pure rejection from reranking.',
        acceptance='Dvisibility vs baseline0, Dheight, Cvisibility, R1visibility: all four lower95%paired-videoCI>0. Only this specific presence-priority repair is eligible for a separate learned version if all pass; otherwise stop presence-only correction preserving this ranking. Do not conclude absence learning in general cannot help.',
        counterargument='Even perfect target presence may retain badly localized or unnecessary corrections. Existing old heads already learned kind0; this is not a claim no refusal training existed. Current ridge confidence omits that head. Missing query annotation is not proof of physical absence. Unknown initial identity can come from sparse supervision. Annotation-based oracle is optimistic and cannot prove a learned model can recover it.',
        distinction='Not localization_labels (candidate proximity), semantic_sampling (same-class row frequency), map_utility (center-gain prediction from map statistics), or the failed foreign-score formula. This prerequisite asks whether perfect binary presence, without changed localization/within-present ordering, is sufficient for the proposed next repair.',
        hypotheses=[dict(version='Map peak is not target visibility', status='code fact; efficacy unproven'), dict(version='Certain negative labels differ from missing annotation', status='code fact; counts to measure'), dict(version='Wrong initial targets cause rejection errors', status='hypothesis; unknownID alone cannot establish it')],
        literature='https://arxiv.org/html/2105.11595v1 sections3.4,5.2: distinguish a valid template with absent target from an incorrect initial region. Its gains are not evidence for our diagnostic or transferable HOTA gains.',
        next='Pass: separately register target-visibility learner with fit-only calibration and same centers/controls. Fail: do not train this presence-only version; next action must change localization or target-conditioned within-present ranking, not another scalar threshold. At most this one prerequisite for the proposed action.',
        sha256={name:sha(ROOT/name) for name in files})
    write(RUN/'protocol.json', p)
    write(RUN/'status.json', dict(state='ready', stage='Аудит меток и точного знания присутствия', done=0, total=130, updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json'; cat=read(cp)
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT, title='З13: достаточно ли точного знания присутствия для отказа', area='flyvis', state='ready', started=now().strftime('%Y-%m-%d'), tags=['fly-problem-P8'],
        question='Исправит ли только знание присутствия отказ, если центр и порядок по высоте среди присутствующих целей прежние?',
        hypothesis='Недостающий обучаемый выход отсутствия — кандидат, но полезность этого действия должна пройти проверку до обучения.',
        expected=p['method']+' '+p['oracle'], acceptance=[dict(criterion='Все130видео/все8326строк каждого варианта точно связаны; разбиения исчерпывающие; неизвестные не переименованы в отрицательные.',met=None),dict(criterion=p['acceptance'],met=None)],
        summary='Одна диагностическая проверка перед возможной заменой отказа; не внедрение.', obtained='', interpretation='', decision='', cannot_claim=p['counterargument'], metrics=[], commits=[], files=files[:4], evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие, альтернатива и критерии до расчёта')]))
    write(cp,cat)
    journal('постановка проверки знания присутствия', '*Действие.* '+p['action']+'\n\n*Проверка.* '+p['method']+' '+p['oracle']+'\n\n*Критерий.* '+p['acceptance']+'\n\n*Сильнейшее возражение.* '+p['counterargument']+'\n\nCPU600с+20с; независимый контроллер. Источник мотива: '+p['literature']+'\n\nТри применяемых правила: '+ '; '.join(p['rules']))


def worker():
    import numpy as np
    from tracking.z13_presence_scope import label_reasons, select_known, counts
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json'); verify_sources(p)
    rows=[]; totals={}; video_rows=[]
    for group, videos in [('fit104',p['fit104']),('evaluate26',p['evaluate26'])]:
        counters={k:Counter() for k in ('all_augmented_and_natural','natural_only')}
        for v in videos:
            if time.monotonic()-T0>p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():
                raise TimeoutError('Original budget')
            with np.load(ROOT/f'runs/z13_data/{v}.npz') as d:
                ep=d['lab_ep']; kind=d['lab_kind']; center=d['lab_center']; crop=d['lab_crop']
                identity=d['ep_identity'][ep]; dist=d['ep_id_dist'][ep]; natural=~d['ep_pert'][ep]
                reasons=label_reasons(kind,center,crop,identity,dist)
                vc={}
                for scope, mask in [('all_augmented_and_natural',np.ones(len(ep),bool)),('natural_only',natural)]:
                    cc=Counter(r for r,ok in zip(reasons,mask) if ok); counters[scope].update(cc); vc[scope]=dict(cc)
                video_rows.append(dict(group=group,video=v,counts=vc))
            write(RUN/'status.json',dict(state='running',stage='Происхождение меток, без изменения данных',done=len(video_rows),total=130,updated=now().isoformat()))
        totals[group]={k:dict(c) for k,c in counters.items()}
    old=read(SOURCE); outcome_rows=[r for r in old if r['candidate']==0]
    pv={}; tally={}; frozen={}
    for a in ARMS:
        rr=[r for r in outcome_rows if r['arm']==a]; assert len(rr)==8326
        assert len({r['label'] for r in rr})==8326
        assert sum(r['kind']>=0 for r in rr)==5822
        rows.extend(rr)
        orig=[r for r in rr if r['selected']]
        frozen[a]=dict(original=counts(orig),suppress_absent_no_refill=counts([r for r in orig if r['kind']!=0]),unknown_not_decided=sum(r['kind']<0 for r in orig))
        for mode in ('height','visibility'):
            chosen=select_known(rr,mode=='visibility'); assert len(chosen)==59
            key=a+'_'+mode; tally[key]=counts(chosen)|{'labels':[r['label'] for r in chosen]}
            pv[key]={v:sum(int(r['outcome']=='beneficial')-int(r['outcome']=='harmful') for r in chosen if r['video']==v)/sum(r['video']==v and r['kind']>=0 for r in rr) for v in p['evaluate26']}
    comp={'Dvisibility-baseline':paired(pv['D_visibility'],{v:0. for v in p['evaluate26']})}
    for key in ('D_height','C_visibility','R1_visibility'):
        comp['Dvisibility-'+key]=paired(pv['D_visibility'],pv[key])
    verify_sources(p)
    write(RUN/'video_label_counts.json',video_rows); write(RUN/'rows.json',rows)
    write(RUN/'report.json',dict(complete=True,label_counts=totals,counts=tally,frozen_policy=frozen,per_video=pv,paired=comp,primary_pass=all(c['ci95'][0]>0 for c in comp.values()), rows_sha256=sha(RUN/'rows.json'), counts_sha256=sha(RUN/'video_label_counts.json'),hota_measured=False,weights_changed=False, limits=p['counterargument']))
    write(RUN/'status.json',dict(state='completed',stage='Метки и диагностическое знание присутствия проверены',done=130,total=130,updated=now().isoformat()))


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    while not (RUN/'controller_result.json').exists():
        if now()>datetime.fromisoformat(read(RUN/'protocol.json')['hard_stop'])+timedelta(seconds=60): raise TimeoutError('Missing controller')
        time.sleep(2)
    e=read(RUN/'controller_result.json'); ps=WindowsProcesses()
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    p=read(RUN/'protocol.json'); verify_sources(p); r=read(RUN/'report.json'); assert r['complete']
    assert sha(RUN/'rows.json')==r['rows_sha256'] and sha(RUN/'video_label_counts.json')==r['counts_sha256']
    from scripts.z13_continue import paired
    # Independent re-ranking and intervals from raw canonical query rows.
    raw=read(RUN/'rows.json'); pv={}
    for a in ARMS:
        rr=[x for x in raw if x['arm']==a and x['kind']>=0]
        for mode in ('height','visibility'):
            order=sorted(rr,key=lambda x: (-int(x['kind']==1),-x['score'],x['label']) if mode=='visibility' else (-x['score'],x['label']))[:59]
            c={k:sum(x['outcome']==k for x in order) for k in ('beneficial','harmful','neutral')}; c.update(unknown=0,selected=59,labels=[x['label'] for x in order]);assert c==r['counts'][a+'_'+mode]
            pv[a+'_'+mode]={v:sum((x['outcome']=='beneficial')-(x['outcome']=='harmful') for x in order if x['video']==v)/sum(x['video']==v for x in rr) for v in p['evaluate26']}
    assert pv==r['per_video']
    comp={'Dvisibility-baseline':paired(pv['D_visibility'],{v:0. for v in p['evaluate26']})}
    for key in ('D_height','C_visibility','R1_visibility'):comp['Dvisibility-'+key]=paired(pv['D_visibility'],pv[key])
    assert comp==r['paired']; passed=all(x['ci95'][0]>0 for x in comp.values()); assert passed==r['primary_pass']
    vc=read(RUN/'video_label_counts.json'); assert len(vc)==130
    for group in ('fit104','evaluate26'):
        for scope in ('natural_only','all_augmented_and_natural'):
            cc=Counter()
            for v in vc:
                if v['group']==group: cc.update(v['counts'][scope])
            assert dict(cc)==r['label_counts'][group][scope]
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),row_counts_selection_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    source=(RUN/'report.json').relative_to(ROOT).as_posix()
    fields=dict(summary='Проверены метки и идеальное бинарное знание присутствия при прежнем порядке пиков; критерий '+('пройден.' if passed else 'не пройден.'),
        obtained=source+' → label_counts: '+str(r['label_counts'])+'; counts: '+str({k:{z:v for z,v in c.items() if z!='labels'} for k,c in r['counts'].items()})+'; paired: '+str(comp)+'. Отдельно сохранён unchanged-frozen-policy отказ без пополнения.',
        interpretation='Это привилегированная проверка достаточности одного бинарного признака для конкретного порядка выбора; не предел всех обучаемых оценок, не доказательство причин и не новая версия FlyVis.',
        decision=p['next']+' Итог основного критерия: '+str(passed)+'.',cannot_claim=p['counterargument'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Проверены исходы диагностического знания присутствия.'))
        card[k]=v
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True,evidence=(RUN/'independent_audit.json').relative_to(ROOT).as_posix());card['acceptance'][1].update(met=passed,evidence=source)
    card['evidence'].append(dict(kind='file',ref=source,claim='Разбиение меток и все парные сравнения'))
    write(cp,cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed,report_sha256=sha(RUN/'report.json')))
    journal('итог знания присутствия перед заменой отказа','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA не измерялась; процент к стенду и доля потолка не определены. Диагностика не добавляет нового архитектурного узла. Новых подтверждённых улучшений для витрины нет.')
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен. '+source+' содержит исходы. '+fields['decision']+' Старые деревья окончены, новые веса не обучались. Видимость цели не равна полезности для сопоставления; неизвестные не стали отрицательными. Публикация runs/z13_presence_scope_20261025/publication.json. Следующий узел выбирать по действующему плану, не повторять этот oracle и не ослаблять критерий. Авторизация26.10 07:40МСК остаётся.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf-8'),encoding='utf-8')
    publish()


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_presence_scope.py','-q']]):
        with (RUN/f'checks_{i}.log').open('w',encoding='utf-8') as f:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph_before.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    foreign_unchanged()
    names=['tracking/z13_presence_scope.py','tests/test_z13_presence_scope.py','scripts/z13_presence_scope_20261025.py','scripts/z13_presence_scope_console.ps1','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [f.relative_to(ROOT).as_posix() for f in RUN.glob('*.json') if f.name!='publication.json']
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names)); assert '.gitignore' not in names and not set(names)&foreign
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Audit target-presence supervision before refusal training','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    elif '--finish' in sys.argv:
        try:finish()
        except Exception as exc:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
    else:raise RuntimeError('Explicit mode required')
