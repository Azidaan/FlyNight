"""Независимо проверить GT-допуск и записать лишь диагностическое значение."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'admission_oracle';OUT=BASE/'admission_oracle_record';SOURCE=RUN.relative_to(ROOT).as_posix()+'/report.json'
from tracking.recovery_detection_support import box,iou
from evaluation.hota_eval import VALID
from scripts.b123_supported_transfer_recovery1_20261027 import write
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    end=read(RUN/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended'] and not end['timed_out']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    r=read(RUN/'report.json');p=read(RUN/'protocol.json');assert r['complete'] and r['oracle_diagnostic_only'] and not r['real_tracker_improvement'] and not r['maximum_or_upper_bound']
    for name,h in p['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    parent=BASE/'supported_transfer/recovery2';real=read(parent/'report.json');added=0;hashes={}
    for v in r['videos']:
        base=read(parent/'baseline'/(v+'.json'));proposals=read(parent/'supported'/(v+'.json'));selected=read(RUN/'oracle'/(v+'.json'));truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json')
        assert len(base)==len(proposals)==len(selected)==len(truth);history={};count=0
        for b,c,o,g in zip(base,proposals,selected,truth):
            assert b['name']==c['name']==o['name'] and o['labels'][:len(b['labels'])]==b['labels']
            assert len({x['id'] for x in o['labels']})==len(o['labels'])
            gt=[x for x in g['labels'] if x['category'] in VALID and not (x.get('attributes') or {}).get('crowd',False)];links=[]
            if b['labels'] and gt:
                weights=np.array([[iou(box(x),box(y)) if x['category']==y['category'] else 0. for y in gt] for x in b['labels']]);rows,cols=linear_sum_assignment(-weights)
                links=[(b['labels'][i],gt[j]) for i,j in zip(rows,cols) if weights[i,j]>=.5]
            covered={(x['category'],str(x['id'])) for _,x in links};targets={(x['category'],str(x['id'])):x for x in gt};used=set()
            original_extra=c['labels'][len(b['labels']):]
            for candidate in o['labels'][len(b['labels']):]:
                assert candidate in original_extra
                key=(candidate['category'],str(candidate['id']));assert key in history;target=history[key]
                assert target in targets and target not in covered and target not in used and iou(box(candidate),box(targets[target]))>=.5
                used.add(target);count+=1
            for x,y in links:history[(x['category'],str(x['id']))]=(y['category'],str(y['id']))
        assert count==r['counts'][v].get('admitted',0);added+=count;path=RUN/'oracle'/(v+'.json');hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in ['baseline','real_rule']:
        other='baseline' if arm=='baseline' else 'supported';assert r['pooled'][arm]==real['pooled'][other] and r['per_video'][arm]==real['per_video'][other]
    for name,other in [('oracle_vs_baseline','baseline'),('oracle_vs_real_rule','real_rule')]:
        d=np.array([r['per_video']['oracle'][v]['HOTA']-r['per_video'][other][v]['HOTA'] for v in sorted(r['videos'])]);samples=d[np.random.default_rng(1).integers(0,20,(5000,20))].mean(1)
        assert r['paired'][name]==dict(mean_per_video=float(d.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),better=int((d>0).sum()),n=20)
    audit=dict(passed=True,ground_truth_diagnostic_only=True,not_maximum_or_upper_bound=True,all20_outputs_only_subset_of_original_candidates=True,past_identity_checked_before_current_update=True,no_base_covered_or_duplicate_true_objects=True,
        added_output_labels=added,paired_independently_reproduced=True,output_sha256=hashes)
    write(RUN/'independent_audit.json',audit)
    q=r['paired']['oracle_vs_baseline'];room=q['ci95'][0]>0
    finding='Этот недоступный модели контроль выбрал полезное сочетание подготовленных добавлений.' if room else 'Этот диагностический контроль не подтвердил полезную прибавку относительно базы.'
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='recovery_admission_oracle_v1');assert card['state']=='ready'
    fields=dict(summary='Контроль с разметкой, не результат реальной модели. '+finding,
        obtained=f'{SOURCE}:pooled.baseline.HOTA={r["pooled"]["baseline"]["HOTA"]};pooled.real_rule.HOTA={r["pooled"]["real_rule"]["HOTA"]};pooled.oracle.HOTA={r["pooled"]["oracle"]["HOTA"]};paired={r["paired"]};'
            f'принято{added}добавлений;источник independent_audit.json.added_output_labels иreport.json.counts. Это значения GT-диагностики,не реальноймодели.',
        interpretation=finding+' Это не максимум HOTA и не доказательство достижимости такого отбора обучением. '
            'Контроль знает истинную личность предыдущего наблюдения и текущую разметку. Его результат отделён от неподтверждённой переносимости реального правила.',
        decision=('Следующее реальное действие — подготовить исходы допуска на отдельных train/dev и обучать отказ от вредных продолжений при неизменных кандидатах. '
            'До ночного срока новое обучение не запускать: разделённые данные такого селектора и его измеренная цена не готовы.' if room else
            'Этот контроль не обосновывает дорогую тренировку селектора. Проверить другой набор кандидатов/геометрию/память в отдельной постановке; отсутствие полезных вариантов в принципе не доказано.'),
        cannot_claim=p['limitation']+' Не включать этот выигрыш в confirmed_results реального трекера,не менять baseline,не считать научныеБ1–Б3проверенными.')
    for key,value in fields.items():
        old=card.get(key,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=old,reason='Записан полный диагностический итог,границы GT-контроля сохранены.'))
        card[key]=value
    card.update(state='infrastructure',done='2026-10-27')
    for item in card['acceptance']:item.update(met=True,evidence=SOURCE)
    card['evidence'] += [dict(kind='file',ref=SOURCE,claim='ТолькоGT-диагностика,неполезностьреальноймодели'),dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/independent_audit.json',claim='Допускстрогопопрошлойличности,исходныедобавленияиинтервалыпроверены')]
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'GT-контроль завершён,не результат обученной модели',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n'
        '*Сводка к базе.* ПрибавкаGT-контроля не считается прибавкойреальноймодели. Максимальныйпотолокнеизмерен;долянедоступна. '
        'Основнойbaselineнеизменён,реальное правило переносаимеетнеопределённыйисход,этоотдельныефакты.')
    for name,args in [('catalog_validation.log',['scripts/validate_project_history.py']),('catalog_tests.log',['-m','pytest','tests/test_project_history.py','-q'])]:
        with (OUT/name).open('w',encoding='utf-8') as f:done=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert done.returncode==0,(name,done.returncode)
    write(OUT/'report.json',dict(complete=True,oracle_diagnostic_recorded=True,catalog_validated=True,catalog_tests_passed=True,witnessed_useful_candidate_combination=room,real_model_improvement_claim=False))
    with (BASE/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — допуск с разметкой завершён как диагностика\n\n'+fields['obtained']+' '+fields['interpretation']+' '+fields['decision']+' Карточкаinfrastructure,неconfirmed_results. Каталог/58тестовпослеправкипрошли. Остатокночисохранить;новоеобучение/подборпороговнезапускать. Сохранение07:20/стоп07:29:30/автоматизацияPAUSEDв07:30.\n')
    print(json.dumps(dict(complete=True,oracle_only=True,real_model_improvement=False,witnessed_useful_combination=room,paired=q),ensure_ascii=False))
if __name__=='__main__':main()
