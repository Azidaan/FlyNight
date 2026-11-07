"""Freeze matched comparison before any new val inference, then publish chain."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'masa_query_preparation';folder=RUN/'masa_query_pilot'
    c=service.read(RUN/'masa_output_interface/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert service.read(RUN/'masa_output_interface/report.json')['passed']
    assert not folder.exists();folder.mkdir()
    old=service.read(ROOT/'runs/continuation_20261028/b1_tracker/protocol.json')
    sources=old['sources'];assert len(sources)==20
    frozen=service.read(ROOT/'runs/recovery_research_20261027/lazy_appearance_frozen.json')
    files=['scripts/run_masa_detic_query_pilot_20261029.py','scripts/audit_masa_detic_query_pilot_20261029.py',
           'scripts/record_masa_detic_query_pilot_20261029.py','tracking/query_appearance_comparison.py',
           'tracking/masa_detic_prepared.py','tracking/masa_detic_bdd_input.py','tracking/masa_r50.py',
           'tracking/frame_masa_appearance.py','tracking/lazy_appearance_association.py',
           'runs/recovery_research_20261027/lazy_appearance_frozen.json',
           'runs/continuation_20261029/masa_output_interface/report.json',
           'weights/masa/masa_r50.pth','weights/masa_detic_author_20261029/detic_masa.pth']
    for s in sources:files.extend(s[k] for k in ['detections','alignment','baseline','raw_array','video_path'])
    p=dict(id='reid_masa_detic_v1',budget_seconds=86400,created=service.now(),sources=sources,config=old['config'],
        variants=['native','zero','r50_1024','r50_1333','detic_1333'],appearance_weight=frozen['weight'],
        intervention='Only original alignment query times: same old ambiguity rule/margin.15 and same additive appearance cost weight.1; no new admissible geometry edges; own last actually assigned query observation per track. Other frames original native association. Full30Hztracker output and future states really change.',
        comparison='Primary Detic1333 vs R501333 isolates extractor at same input resolution, tracker action and memory. Native is system reference; R501024 separates input-size effect. Zero computes Detic association then discards it; exact raw/JSON/mean/cov every frame. Both pretrained encoders frozen.',
        feature_cpu_gpu_normalized_tolerance=1e-5,feature_audit_video_indices=[0,10,19],
        audit='Every frame/assignment/event/raw/JSON/metric replayed on CPU from hash-bound recorded normalized vectors. Before use, fixed first query frame in videos0/10/19 independently reencoded on CPU for all3feature variants within1e-5. This is sampled encoder verification, not exhaustive pixel replay.',
        primary_criteria='Both lower95%paired-video-bootstrap20k seed1 HOTA bounds Detic-native and Detic-R501333 positive. Report all per-video/full pooled HOTA/AssA/IDF1/FP/FN/IDSw, unfavourable results too. No val tuning.',
        price='First complete five-tracker video including metrics/save yields measured conservative remaining estimate; stage time gates cancelled, absolute save30.10 07:30 and stop07:40 remain. Disk gate remains binding.',
        limitations='Historical20val already used in B1, not new independent test. Query-only association is our adapter, not the complete official MASA tracker. Alpha transferred unchanged from earlierdev, no Detic-specific HPO. Original raw-feature FP32 reference gate remains failed; separately preregistered normalized interface passed on two TRAIN frames; shared FPN/fusion/head is not full independent MMDetection.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    service.write(folder/'protocol.json',p)
    for name in ['masa_query_audit','masa_query_record']:
        service.write(RUN/name/'protocol.json',dict(budget_seconds=86400,parent=p['id'],created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']==p['id'])
    assert card['state']=='planned';card['state']='ready'
    card.setdefault('evidence',[]).append(dict(kind='file',ref='runs/continuation_20261029/masa_query_pilot/protocol.json',claim='До нового val заморожены20видео/5контролей/запросы/веса/статистика/аудит'))
    card['expected']='Полныйсогласованныйпилот20историческихval/5вариантов,нулевойконтрольточен,всеназначенияиполныеHOTAнезависимопроверены;двапервичныхсравненияDetic-nativeиDetic-R501333.'
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Новое окно и согласованный полный пилот MASA-Detic до val',
        'Прямое «Продолжаем, и когда будет ближайший результат?» открыло новоеокно до30.10 07:40МСК; save07:30/guard07:39:30, одинGPU. '
        'runs/continuation_20261029/masa_output_interface/report.json: заранее заданныйнормированныйвыход1e-5пройденна2trainкадрах,максимумвектора3.949e-7/косинуса4.619e-7. СтарыйсыройFP32gate1e-4НЕизменён/НЕпройден. '
        'Новыйпротоколmasa_query_pilot/protocol.json:20старыхval/5вариантов/native/zero/R501024/R501333/Detic1333; одинаковоеquery-сопоставление,реальныераздельныеистории,фиксированныйстарыйdevвес.1. '
        'Меняетсяреальноназначениеполного30Гцтрекера,новыхдетекций/данных/обучениянет. CPUвседискретныевыходы/метрикииотдельныевыборочныепризнаки; этоадаптациясистемы,неофициальныйтрекерMASAинеядроFlyVis. Результатпокачествупоканеизмерен.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_appearance_comparison.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert result==0,(i,result)
    service.write(checks/'report.json',dict(passed=True,ready=True,updated=service.now()))
    for index,name,script in [('010','masa_query_pilot','scripts/run_masa_detic_query_pilot_20261029.py'),
                              ('020','masa_query_audit','scripts/audit_masa_detic_query_pilot_20261029.py'),
                              ('025','masa_query_record','scripts/record_masa_detic_query_pilot_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=(RUN/name).relative_to(ROOT).as_posix(),script=script,
            script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
