"""Local completion queue and independently bounded feature audit, no model polling."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
FEATURES=ROOT/'runs/z13_temporal_region_features_20261025'
OUT=FEATURES/'postprocess'


def queue():
    assert not (OUT/'protocol.json').exists()
    ps=WindowsProcesses();write(FEATURES/'completion_queue.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),script=Path(__file__).relative_to(ROOT).as_posix()))
    p=read(FEATURES/'protocol.json');hard=datetime.fromisoformat(p['hard_stop'])
    while not (FEATURES/'controller_result.json').exists():
        if now()>hard+timedelta(seconds=30):raise TimeoutError('Feature controller did not finish in original window')
        time.sleep(3)
    e=read(FEATURES/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert read(FEATURES/'report.json')['complete']
    stop=min(now()+timedelta(seconds=600),datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    write(OUT/'protocol.json',dict(device='cpu',budget_seconds=600,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(OUT/'stop.request').relative_to(ROOT).as_posix(),source_sha256=sha(__file__),feature_report_sha256=sha(FEATURES/'report.json')))
    result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_bounded_recovery_controller_20261022.py','--run',str(OUT),'--script',str(Path(__file__))],cwd=ROOT,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    write(FEATURES/'completion_queue_result.json',dict(ended=now().isoformat(),postprocess_exit=result.returncode))


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_margin_rule_audit_20261025 import revise
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    p=read(FEATURES/'protocol.json');verify_sources(p);r=read(FEATURES/'report.json');q=read(OUT/'protocol.json')
    assert q['source_sha256']==sha(__file__) and q['feature_report_sha256']==sha(FEATURES/'report.json')
    refs={}
    for name,h in read(ROOT/'runs/z13_temporal_observation_gate_20261025/report.json')['manifest'].items():
        assert sha(ROOT/name)==h;refs.update({x['uid']:x for x in read(ROOT/name)['rows'] if x['eligible']})
    reference={};seen={a:set() for a in ('D','C','R1')};first_frames={a:0 for a in seen}
    for name,h in r['manifest'].items():
        if now()>=datetime.fromisoformat(q['save_at']) or (OUT/'stop.request').exists():raise TimeoutError('Independent audit budget')
        assert sha(ROOT/name)==h;marker=read(ROOT/name);path=(ROOT/name).with_suffix('.npz');assert sha(path)==marker['sha256']
        assert marker['max_center_px']<.5 and marker['max_score']<.001 and marker['previous_save_reload_exact']
        arm=Path(name).parts[-3]
        with np.load(path) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for i,u in enumerate(z['uid']):
                u=int(u);assert u not in seen[arm] and u in refs;seen[arm].add(u)
                n=int(z['previous_mask'][i].sum());assert n>=2 and z['previous_mask'][i,:n].all() and not z['previous_mask'][i,n:].any()
                ref=refs[u];assert int(z['previous_frame'][i])==ref['previous']['frame']
                assert int(z['gap_frames'][i])==ref['current_frame']-ref['previous']['frame']>0
                binding=(n,z['previous_geo'][i,:n].tobytes(),int(z['previous_frame'][i]),int(z['gap_frames'][i]),int(z['relative_step'][i]))
                if u not in reference:reference[u]=binding
                else:assert reference[u]==binding
                first_frames[arm]+=int(z['relative_step'][i])==0
    assert all(u==set(refs) for u in seen.values()) and r['rows_each_arm']=={a:22090 for a in seen}
    write(OUT/'independent_audit.json',dict(at=now().isoformat(),all_feature_hashes_previous_geometry_uid_time_bindings_and_old_peak_tolerances_verified=True,rows_each_arm={a:len(s) for a,s in seen.items()},first_fadein_frame_rows=first_frames,feature_report_sha256=sha(FEATURES/'report.json'),model_trained=False,hota_measured=False))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']=='z13_temporal_region_features_v1')
    fields=dict(obtained='runs/z13_temporal_region_features_20261025/report.json: '+str(r['rows_each_arm'])+';groups='+str(r['groups'])+'. postprocess/independent_audit.json проверилхеши,одинаковыесвязкиUID/время/геометрию,D/C/R1 ипрежниедопуски. Обученныхмоделей0.',
        interpretation='Подготовлены равные признаки прошлых реальных областей для22090запросов; прежние текущие ответы сохранены в допусках. Это успешная подготовка данных, не подтверждение различения личности или полезного вмешательства.',
        decision='Данные готовы к отдельной постановке временного считывателя: заранее задать отличие от старого выбора3пиков, равные контроли безистории/безмухи/C/R1, прежние научные критерии и измеренныйGPUбюджет. Обучение этой очередью не запускалось. Пространственный модуль временно отложен.',
        cannot_claim='Нельзя заявлять выученную память личности, улучшениеFlyVis/HOTA или готовый эффективный узел. Ссылки на прошлое не гарантируют истинную личность; обучение и независимая оценка эффективности ещё отсутствуют.')
    for k,v in fields.items():revise(card,k,v,'Полное извлечение и независимый аудит равных временных признаков, без обучения.')
    card.update(state='completed_positive',done=now().strftime('%Y-%m-%d'),summary='Равные временные признаки подготовлены и проверены; эффективность модели ещё не измерена.')
    for a in card['acceptance']:a.update(met=True,evidence='runs/z13_temporal_region_features_20261025/postprocess/independent_audit.json;price.json')
    card['evidence'].append(dict(kind='file',ref=(OUT/'independent_audit.json').relative_to(ROOT).as_posix(),claim='Независимая сверка подготовки признаков,0моделей'))
    write(ROOT/'experiments/project_history.json',cat)
    from scripts.z13_temporal_observation_gate_20261025 import journal
    journal('равные временные признаки D/C/R1 подготовлены',fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA/проценткстенду/доляпотолканеизмерены.')
    from scripts.z13_spatial_set_readout_20261025 import handoff
    handoff('GPUподготовка z13_temporal_region_features_v1 завершенаexit0/allownedended. ВсеD/C/R1по22090UID, postprocess/independent_audit.json проверилсвязки/геометрию/время/хеши/прежниепики. Это0обученныхмоделей,неHOTA. Следующее действие—отдельныйвременнойсчитыватель,егоисполнитель/протоколещёнеготовы;проверитьотличиеотпрежних3пиков/памяти. Очередьавтоматическизаписываетитог/проверки/граф,нонетренирует. publication_pending. Пространственныймодульвременноотложен. Два листаdrawio;одинGPU/скрытыеокна/57чужихправок/.gitignoreсохранить. До26Oct07:40поавторскомуокну,старые бюджеты непродлевались.')
    foreign_unchanged()
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','-q'])]:
        proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf8'},capture_output=True)
        (OUT/(label+'.stdout.log')).write_bytes(proc.stdout);(OUT/(label+'.stderr.log')).write_bytes(proc.stderr);assert proc.returncode==0
    prev=ROOT/'runs/z13_temporal_observation_gate_20261025/publication_checks/vault.json';protected={n.replace(chr(92),'/') for n in read(prev)['protected_existing_edits']}
    own={n:sha(ROOT/n) for _,n in graph_rows() if n not in protected and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'};write(OUT/'owned_graph.json',own)
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(OUT/'vault.json'),'--refresh-owned',str(OUT/'owned_graph.json')],cwd=ROOT,capture_output=True)
    (OUT/'vault.stdout.log').write_bytes(proc.stdout);(OUT/'vault.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0
    foreign_unchanged();write(OUT/'finish_recorded.json',dict(at=now().isoformat(),complete=True,publication_pending=True,model_trained=False,hota_measured=False))


if __name__=='__main__':
    try:
        if '--worker' in sys.argv:worker()
        else:queue()
    except Exception as exc:
        write(FEATURES/'completion_queue_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
