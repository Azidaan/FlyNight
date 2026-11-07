"""Freeze the directly approved cost intervention before any new trial."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'cost_use_preparation';folder=RUN/'cost_use_dev'
    auth=service.read(RUN/'cost_use_authorization.json')
    assert auth['accepted'] and auth['human_message']=='Да, проверить применение признаков'
    assert service.sha(ROOT/auth['proposal'])==auth['proposal_sha256']
    c=service.read(RUN/'bounded_write_agent_record/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    old=service.read(RUN/'memory_write_dev/protocol.json')
    inputs=service.read(RUN/'memory_write_dev/tracking_completed.json')['sha256']
    assert all(service.sha(ROOT/f)==h for f,h in inputs.items())
    files=['tracking/query_cost_use.py','tests/test_query_cost_use.py','tracking/query_appearance_comparison.py',
        'tracking/xywh_bytetrack.py','tracking/frame_masa_appearance.py','tracking/offline_association.py',
        'scripts/cost_use_dev_20261029.py','scripts/audit_cost_use_dev_20261029.py',
        'scripts/assignment_write_dev_20261029.py','scripts/run_bdd_video_rate_experiment.py',
        'scripts/cost_use_val_20261029.py','scripts/audit_cost_use_val_20261029.py','scripts/record_cost_use_val_20261029.py',
        'scripts/prepare_cost_use_dev_20261029.py','scripts/prepare_cost_use_val_20261029.py',
        'runs/continuation_20261029/cost_use_authorization.json',auth['proposal'],
        'runs/continuation_20261029/memory_write_dev/protocol.json',old['source']['detections'],old['source']['alignment'],
        'data/bdd100k/labels/box_track_20/train/'+old['source']['video']+'.json',*inputs]
    p=dict(id='query_cost_use_v1',created=service.now(),budget_seconds=86400,source=old['source'],config=old['config'],appearance_weight=old['appearance_weight'],
        new_variants=['r50_centered','detic_centered','r50_shift'],neutral_distance=.5,
        action='Keep last actually assigned query observation. Change only cost use on original geometrically eligible ambiguous first-association edges: old c+float32(.1)*d; centered c+float32(.1)*(d-float32(.5)); common shift c-float32(.05). d=(1-cos)/2. Float32 multiplication before addition, no new eligible edges. Shift computes features for a real shadow-like path but discards their content.',
        strongest_counterargument='Centering may merely soften the solver threshold; common signal-free shift is therefore compulsory. Appearance can still reward a wrong identity. The previous nonnegative penalty crossing a threshold was not ground-truth error evidence.',
        dev_gate='One complete fixed first fit20 TRAIN video0001542f-5ce3cf52,1213frames;3new variants plus native-shadow and real centered zero-shadow. All raw/JSON/cost/write/metrics independently replayed on CPU with exact old vectors. Negative-cost solver, old parity, geometry and signal-free independence unit checks. No quality selection/tuning. Detector has seen TRAIN; this gate is technical only.',
        prospective_primary='Four centered-old and centered-shift comparisons across R501333/Detic1333,paired20000seed1,ordinary95%andBonferroni98.75%;main iff all4correctedlower>0. Full20historicalval separately frozen after dev. Native3 secondary, not main.',
        partial_confirmation_rule='Added before dev: positive confirmed_results require ci_global7 lower>0, Bonferroni family of4primary+3preplannednative; primary main unchanged. Extractor comparison descriptive95% only, not eligible for confirmation.',
        no_training=True,no_encoder_inference=True,gt_in_predictor=False,detector_has_seen_development_train=True,
        deadline='Save30.10 07:30/ownedstop07:39:30/hardstop07:40. Smaller stage caps cancelled. Disk reserve5GB remains binding. No promised full-chain completion time.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    service.write(RUN/'cost_use_dev_audit/protocol.json',dict(budget_seconds=86400,parent_sha256=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');assert not any(x['id']==p['id'] for x in h['experiments'])
    h['experiments'].append(dict(id=p['id'],title='Применение признаков — центрированная поправка и общий сдвиг стоимости',area='reid',state='ready',started=datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
        question='Даёт ли содержание признаков пользу при центрировании стоимости сверх общего сдвига без внешности?',
        hypothesis='Непрерывно положительная добавка может мешать сопоставлению у порога; центрирование может помочь, но тот же эффект способен дать простой сдвиг. Причина прежних научных неудач не установлена.',
        expected='Технический dev с независимым повтором, затем полный опыт с отдельным контролем сдвига без признаков. Прирост не обещан.',
        acceptance=[dict(criterion=p['dev_gate'],met=None),dict(criterion=p['prospective_primary'],met=None)],
        cannot_claim='Подготовка и dev не подтверждают пользу. Память последнего наблюдения/YOLO/признаки/query/геометрия неизменны; изменяется только cost. Это авторская адаптация, не полный Deep SORT/BoT-SORT и не ядро FlyVis. Исторические20val после последовательных исследований не новый независимый тест. Четыре главных сравнения сохраняются; дополнительное правило семьи7 для частных подтверждений записано до dev.',
        evidence=[dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Прямое разрешение автора и условия до нового dev/val')]))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Прямое разрешение на применение признаков; отдельный протокол до dev',
        'Автор прямо ответил «Да, проверить применение признаков»; точный ответ/ссылка на вопрос/SHA предложения в cost_use_authorization.json. Три проверки записи памяти завершены; новая постановка меняет cost, память last сохраняется. Прежнее c+0,1d / центрированное c+0,1(d−0,5) / общий сдвиг c−0,05 без внешности. Геометрически запрещённые рёбра не открываются, YOLO/query/признаки/.1 не меняются; середина диапазона0,5 задана до пробы. Сильная альтернатива — одно смягчение порога. Протокол cost_use_dev/protocol.json до новых выходов, первый фиксированный TRAIN dev целиком/3новых/zero/полный CPUповтор; старые точные векторы без encoder. 3unitpassed3,43с: отрицательные стоимости/геометрия/oldexact/zeroexact/сдвиг независимо от содержания. Основной val4primary/20kseed1/95%/Bonf98,75%; до dev дополнено более строгое правило семьи7 для частных подтверждений, не изменяющее главный критерий. Общий срок30.10 07:40 неизменен, готовые20/обучения не повторяются.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_cost_use.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,n,s in [('310','cost_use_dev','scripts/cost_use_dev_20261029.py'),('320','cost_use_dev_audit','scripts/audit_cost_use_dev_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{n}.json',dict(name=n,folder=str((RUN/n).relative_to(ROOT)),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(RUN/n/'protocol.json')))

if __name__=='__main__':main()
