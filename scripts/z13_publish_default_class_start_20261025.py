"""Record completed rule-only audit and publish exact owned preparation files."""
from pathlib import Path
import os, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_margin_rule_audit_20261025 import revise,journal
from scripts.z13_continue import paired
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
AUDIT=ROOT/'runs/z13_margin_rule_audit_20261025';RUN=ROOT/'runs/z13_default_class_trial_20261025'


def record():
    assert not (AUDIT/'finish_recorded.json').exists()
    p=read(AUDIT/'protocol.json');verify_sources(p);r=read(AUDIT/'report.json');e=read(AUDIT/'controller_result.json')
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    entries={}
    for v in p['fit20']:
        entries.update({int(k):x for k,x in read(ROOT/f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries'].items()})
    paths=list((AUDIT/'policies').glob('*/seed*/*/report.json'));assert len(paths)==48
    for path in paths:
        rec=read(path);rows=read(path.with_name('rows.json'));assert sha(path.with_name('rows.json'))==rec['rows_sha256']
        per={};counts=dict(rows=len(rows),known=0,selected=0,unknown_selected=0,beneficial=0,harmful=0)
        for v in p['fit20']:
            rr=[x for x in rows if x['video']==v];net=known=0
            for x in rr:
                en=entries[x['uid']];assert x['kind']==en['kind'];a=x['action'];k=x['kind']>=0
                known+=k;counts['known']+=k;counts['selected']+=a>=0;counts['unknown_selected']+=a>=0 and not k
                good=bool(k and a>=0 and en['good'][a]);harm=bool(k and a>=0 and en['harm'][a])
                counts['beneficial']+=good;counts['harmful']+=harm;net+=int(good)-int(harm)
            per[v]=net/known
        assert per==rec['per_video'] and counts==rec['counts']
        for f in rec['folds']:
            assert not set(f['calibration_videos'])&set(f['validation_videos'])
    means={rule:{arm:{v:float(np.mean([read(AUDIT/'policies'/arm/f'seed{s}'/rule/'report.json')['per_video'][v] for s in p['seeds']])) for v in p['fit20']} for arm in r['means'][rule]} for rule in r['means']}
    assert means==r['means'];d=means['margin']['D_context']
    comp={'Dmargin-baseline':paired(d,{v:0. for v in d}),'Dmargin-Dprobability':paired(d,means['probability']['D_context'])}
    for a in ('C_context','R1_context','geometry_context','D_blank'):comp['Dmargin-'+a]=paired(d,means['margin'][a])
    assert comp==r['paired']
    write(AUDIT/'independent_audit.json',dict(at=now().isoformat(),all48_policy_rows_labels_counts_means_and_intervals_verified=True,report_sha256=sha(AUDIT/'report.json'),all_owned_ended=True))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']=='z13_margin_rule_audit_v1')
    reason='Завершённая проверка по неизменному протоколу; первичные строки и интервалы независимо сверены.'
    for key,val in dict(summary='Все24модели воспроизведены. Замена одного порога не прошла допуск на20обучающих видео.',obtained='runs/z13_margin_rule_audit_20261025/report.json → paired: '+str(comp),interpretation='Изолировано только правило выбора, оценки и действие прежние. Прямой порог не дал достаточной пользы. Интервалы условны на повторныхfit20 и фиксированном разбиении; новая функция ошибки не проверялась.',decision='Не продвигать только замену порога на26/HOTA. Перейти к отдельно зарегистрированной проверке нового обучения отказу наfit104; провал правила не доказывает отсутствие информации.').items():revise(card,key,val,reason)
    card['state']='completed_positive' if r['margin_only_pass'] else 'completed_negative';card['done']=now().strftime('%Y-%m-%d')
    card['acceptance'][0].update(met=True,evidence='runs/z13_margin_rule_audit_20261025/reproduction.json')
    card['acceptance'][1].update(met=r['margin_only_pass'],evidence='runs/z13_margin_rule_audit_20261025/report.json')
    parent=next(x for x in cat['experiments'] if x['id']=='z13_default_class_readout_v1')
    parent['acceptance'][0].update(met=True,evidence='runs/z13_margin_rule_audit_20261025/reproduction.json')
    parent['acceptance'][1].update(met=True,evidence='runs/z13_margin_rule_audit_20261025/independent_audit.json; измерение выполнено, научный допуск только замены порога не пройден')
    write(ROOT/'experiments/project_history.json',cat)
    write(AUDIT/'finish_recorded.json',dict(at=now().isoformat(),complete=True,margin_only_pass=r['margin_only_pass']))
    write(AUDIT/'confirmed_review.json',dict(eligible_new_confirmed_results=0,reason='No positive primary D comparison lower bound. Negative D-R1 is reported, not hidden.'))
    journal('итог С0/С1а и переход к обучению отказа','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+' НоваяС1б: runs/z13_default_class_trial_20261025/protocol.json,120внутренних моделей,7200сCPU+20с, условный переходк26встроен. Архитектурнуюпересборку объединитьситогомновогоузлаотказа; этаступеньменяеттолькопорог.\n\nHOTA/проценткстенду/доляпотолканеизмерены.')


def checks():
    assert not (AUDIT/'checks.json').exists();foreign_unchanged()
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_margin_rule.py','tests/test_z13_default_class.py','-q'])]:
        proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True)
        (AUDIT/(label+'.stdout.log')).write_bytes(proc.stdout);(AUDIT/(label+'.stderr.log')).write_bytes(proc.stderr)
        results[label]=dict(exit_code=proc.returncode,tail=proc.stdout.decode('utf8',errors='replace')[-1300:]);assert proc.returncode==0,results[label]
        print(label+' passed',flush=True)
    assert before==fingerprint(ROOT)
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(AUDIT/'vault_final.json')],cwd=ROOT,capture_output=True)
    (AUDIT/'vault.stdout.log').write_bytes(proc.stdout);(AUDIT/'vault.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0
    foreign_unchanged();write(AUDIT/'checks.json',dict(at=now().isoformat(),results=results,inputs_unchanged_during_tests=True))


def publish():
    assert not (AUDIT/'publication.json').exists();assert read(AUDIT/'checks.json')['inputs_unchanged_during_tests']
    verify_sources(read(AUDIT/'protocol.json'));verify_sources(read(RUN/'protocol.json'));foreign=foreign_unchanged()
    protected={n.replace(chr(92),'/') for n in read(AUDIT/'vault_final.json')['protected_existing_edits']}
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf8').strip()
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    names=['tracking/z13_margin_rule.py','tracking/z13_default_class.py','tests/test_z13_margin_rule.py','tests/test_z13_default_class.py','scripts/z13_margin_rule_audit_20261025.py','scripts/z13_default_class_trial_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [f.relative_to(ROOT).as_posix() for f in AUDIT.rglob('*.json') if '.owned_backup' not in f.as_posix() and f.name!='publication.json']
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','controller.json','price.json') if (RUN/n).is_file()]
    names += [n for _,n in graph_rows() if n not in protected and (ROOT/n).is_file()]
    names=sorted(set(names));assert not(set(names)&foreign) and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Audit frozen-score decisions and start matched explicit-default readouts','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(AUDIT/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit)


if __name__=='__main__':
    if '--record' in sys.argv:record()
    elif '--checks' in sys.argv:checks()
    elif '--publish' in sys.argv:publish()
