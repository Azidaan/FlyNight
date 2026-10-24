"""Record the third diagnosis and publish only this investigation's own files."""
from pathlib import Path
from collections import Counter
import inspect
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_refusal_causes_20261025 import RUN,IDENT,read,write,sha,now,journal
W=ROOT/'runs/z13_weekend_20261024'


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    stage=RUN/'normalization';assert not(stage/'finish_recorded.json').exists()
    p,e,r=read(stage/'protocol.json'),read(stage/'controller_result.json'),read(stage/'report.json')
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'] and r['complete']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    verify_sources(p)
    from scripts.z13_continue import paired
    actual=read(RUN/'rows.json')
    for arm,modes in r['results'].items():
        source={x['label']:x for x in actual if x['arm']==arm}
        for mode,result in modes.items():
            selected=set(r['chosen_labels'][arm][mode]);assert len(selected)==84
            c=Counter();pv={v:dict(known=0,beneficial=0,harmful=0) for v in result['per_video']}
            for label,row in source.items():
                if row['kind']>=0:pv[row['video']]['known']+=1
                if label not in selected:continue
                c['selected']+=1;c['unknown' if row['kind']<0 else 'known']+=1
                if row['kind']>=0:
                    c[row['outcome']]+=1
                    pv[row['video']]['beneficial']+=row['outcome']=='beneficial'
                    pv[row['video']]['harmful']+=row['outcome']=='harmful'
            assert dict(c)==result['counts'] and pv==result['per_video']
    for name,value in r['paired'].items():
        left,right=name.split('-');a,mode=left.split('_',1);b,other=right.split('_',1)
        assert paired(r['results'][a][mode]['net_frequency'],r['results'][b][other]['net_frequency'])==value
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    old=card['obtained'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field='obtained',old=old,reason='Добавлен заранее зарегистрированный контроль нормировки; прежний текст сохранён.'))
    source='runs/z13_refusal_causes_20261025/normalization/report.json'
    contrast=r['paired']['D_centered_peak-D_z']
    card['obtained']=old+' '+source+' → Dcentered_peak: '+json.dumps(r['results']['D']['centered_peak']['counts'])+';Dcentered_peak-Dz: '+json.dumps(contrast)+'. primary_pass='+str(r['primary_pass'])+'.'
    old=card['decision'];card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field='decision',old=old,reason='Завершены три диагностические ступени; простой скалярный ремонт не принят.'))
    card['decision']='Не внедрять замену нормировки: заранее записанный критерий не пройден. Не продолжать подбор скалярного порога на этих26видео и не повторять прежние отрицательные проверки абсолютного отклика/метки локализации/семантической выборки. Текущий разбор завершён: неподходящая цель отказа и ограниченность предложений разделены, но конкретная новая модель ещё не принята. Следующий ремонт должен менять внутренний узел отказа или предлагаемые муху состояния, иметь новый проверяемый механизм и контроль без мухи; прежний непройденный гейт обучения по меткам назначения сохраняется.'
    card['evidence'].append(dict(kind='file',ref=source,claim='Контроль роли деления на разброс карты, отрицательный критерий перехода'))
    write(cp,cat)
    write(stage/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(stage/'report.json'),all_owned_ended=True,independent_counts_and_intervals=True,primary_pass=r['primary_pass']))
    journal('итог контроля нормировки:одного удаления деления недостаточно',
        '*Получено.* '+source+' → Dcentered_peak: '+json.dumps(r['results']['D']['centered_peak']['counts'])+
        ';Dcentered_peak-Dz: '+json.dumps(contrast)+'. ДИ включает0, primary_pass='+str(r['primary_pass'])+'. '
        'МедианаSDкартыDдляkind0='+str(r['statistics']['D']['0']['sd'][1])+',дляkind1='+str(r['statistics']['D']['1']['sd'][1])+
        ';это описательная статистика,не доказательство одинаковости распределений. '
        '*Вывод.* Высокаястандартизованнаявершина не являетсяполезностью;нооднанормировка не объясняет весьпровалотказа. '
        '*Решение.* '+card['decision']+'\n\nТриступени завершены,новойHOTAнет,проценткbaselineидоляизмеренногопотолканеопределены. '
        'Счётчики62из84ужеизвестны22.10,новоеточное назначениеибиндингпервоисточниковнеподаютсякакновоенаблюдениеэтогосчётчика.')
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'Разбор отказа z13_refusal_causes_v1 завершён:3короткихCPUступени/exit0/все свои деревья окончены. '
        'Источник runs/z13_refusal_causes_20261025:report.json,trace/report.json,diagnosis.json,explanation.json,normalization/report.json. '
        'D фиксированнаяполитика:84выбранных,64известных нейтральных/20неизвестных,48полезныхtop1отвергнуты. '
        '62из64известных — цельнеаннотированавкадре;это не доказательство физическогоотсутствия. '
        '895из1001ошибокнеисправимыэтимитремякандидатами,58исправимытольконеtop1. '
        'У17старыхполезныхповцентрувредныхкандидатов15нетправильнойhighдетекции,2правильнаястоимостьпересекает0,8. '
        'Контрпример UID34278:ошибкацентра10,24→7,65px,ноcost0,7668→0,8139;порогегоНЕприменял. '
        'УдалениеSDизбалла дало1полезныйиз84при36неизвестных,ДИприроставключает0;версиянепринята,непродвигатькHOTA. '
        'Не повторять простую абсолютнуюуверенность,локализационнуюметку,семантическуювыборку,обучениепарикондиционированиепоиском:этиверсииужеотрицательны. '
        'Готового проверенного нового исполнителя модернизациипоканет;невыдаватьзавершённыйразборзаидущийпрогон. '
        'Следующийузелнужновыбратьсновымпроверяемыммеханизмом. Остановлена конкретная скалярнаяправкапослетрёхдиагностик,не линияFlyVis. '
        'Авторизация26.10 07:40остаётся,меньшиебюджетынепродлеваются. Чужие57графправок/.gitignoreсохранить.\n\nНиже прежние передачи сохранены.\n\n')
    path=W/'HANDOFF.md';path.write_text(header+path.read_text(encoding='utf-8'),encoding='utf-8')


def publish():
    from scripts import z13_publish_assignment_effect_20261024 as P
    P.RUN=RUN
    extra=[Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/z13_refusal_causes_20261025.py','scripts/z13_refusal_trace_20261025.py',
        'scripts/z13_record_refusal_causes_20261025.py','scripts/z13_refusal_explain_20261025.py',
        'scripts/z13_refusal_normalization_20261025.py','scripts/z13_refusal_causes_console.ps1',
        'scripts/z13_refusal_trace_console.ps1','tests/test_z13_refusal_causes.py','tests/test_z13_refusal_trace.py']
    for stage in (RUN,RUN/'trace',RUN/'normalization'):
        for name in ('protocol.json','controller.json','controller_result.json','launch.json','report.json','finish_recorded.json',
            'rows.json','diagnosis.json','explanation.json','box_counterexample.png','box_counterexample.svg',
            'executed_replay_source.txt','D_map_statistics.json','C_map_statistics.json','R1_map_statistics.json'):
            if(stage/name).is_file():extra.append((stage/name).relative_to(ROOT).as_posix())
    extra += [x.relative_to(ROOT).as_posix() for x in (RUN/'trace/videos').glob('*.json')]
    source=inspect.getsource(P.publish)
    assert source.count("names=['experiments/project_history.json'")==1
    source=source.replace("names=['experiments/project_history.json'","names=EXTRA+['experiments/project_history.json'")
    source=source.replace("'tests/test_z13_assignment_effect.py','-q'","'tests/test_z13_assignment_effect.py','tests/test_z13_refusal_causes.py','tests/test_z13_refusal_trace.py','-q'")
    source=source.replace("prefix='assignment_effect_'+mode","prefix='refusal_causes_final'")
    source=source.replace("title='Correct invalid crop-sampling premise and start exact assignment-effect supervision' if mode=='start' else 'Record complete assignment-effect supervision with exact baseline replay'", "title='Trace FlyVis refusal failures and reject scalar normalization repair'")
    namespace={**vars(P),'EXTRA':extra}
    exec(compile(source,str(Path(__file__)), 'exec'),namespace)
    namespace['publish']('refusal_causes_final')


def resume_publication():
    """Reuse passed checks after SVG whitespace-only export repair."""
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
    from evaluation.project_history.index import fingerprint
    checks=read(W/'refusal_causes_final_checks.json')
    assert checks['catalog_exit']==checks['tests_exit']==0 and checks['vault_completed']
    assert fingerprint(ROOT)==checks['fingerprint_after']
    protected=foreign_unchanged()
    names=[x for x in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)) if x]
    assert not set(names)&protected and '.gitignore' not in names
    exact={'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/z13_publish_refusal_causes_20261025.py','scripts/z13_record_refusal_causes_20261025.py',
        'scripts/z13_refusal_causes_20261025.py','scripts/z13_refusal_trace_20261025.py',
        'scripts/z13_refusal_explain_20261025.py','scripts/z13_refusal_normalization_20261025.py',
        'scripts/z13_refusal_causes_console.ps1','scripts/z13_refusal_trace_console.ps1',
        'tests/test_z13_refusal_causes.py','tests/test_z13_refusal_trace.py',
        'runs/z13_weekend_20261024/HANDOFF.md','runs/z13_weekend_20261024/refusal_causes_final_checks.json',
        'runs/z13_weekend_20261024/refusal_causes_final_owned_graph.json','runs/z13_weekend_20261024/refusal_causes_final_vault.json'}
    for name in names:
        path=(ROOT/name).resolve();path.relative_to(ROOT.resolve())
        if name in exact:continue
        if path.is_relative_to(RUN.resolve()):
            assert path.suffix in ('.json','.png','.svg','.txt')
        else:
            path.relative_to((ROOT/'ObsidianFlyVis').resolve());assert path.suffix=='.md'
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Trace FlyVis refusal failures and reject scalar normalization repair','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged()
    write(W/'refusal_causes_final_publication.json',dict(at=now().isoformat(),commit=head,push_completed=True,paths=names,
        service_repair='Matplotlib SVG trailing spaces stripped; scientific JSON unchanged; passed catalog/73tests/vault checks retained with matching fingerprint.'))
    print(head)


if __name__=='__main__':
    if '--finish' in sys.argv:finish()
    elif '--publish' in sys.argv:publish()
    elif '--resume-publication' in sys.argv:resume_publication()
    else:raise ValueError('Choose --finish or --publish')
