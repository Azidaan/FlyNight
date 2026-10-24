"""Verify completed input/raw milestones and save reviewable matched raw table."""
from pathlib import Path
import sys,json,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_common_20261024 as S
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024'


def main():
    from scripts.z13_continue import paired
    from scripts.z13_localization_labels_20261023 import verify_sources
    for stage,n in ((S.RUN/'inputs_recovery1',21),(S.RUN/'raw',84)):
        S.ended(stage);p=read(stage/'protocol.json');verify_sources(p);assert read(stage/'finish_recorded.json')['report_sha256']==sha(stage/'report.json');markers=list((stage/'jobs').glob('*.json'));assert len(markers)==n
        for f in markers:
            for name,h in read(f)['outputs'].items():assert sha(ROOT/name)==h
    r=read(S.RUN/'raw/report.json');pv={a:{v:m['HOTA'] for v,m in rows.items()} for a,rows in r['per_video'].items()}
    for key,value in r['paired'].items():a,b=key.split('-');assert paired(pv[a],pv[b])==value and value['n']==20
    base=r['pooled']['xywh']['HOTA'];relative={a:100*(x['HOTA']-base)/base for a,x in r['pooled'].items()}
    write(S.RUN/'raw/verified_milestone.json',dict(at=now().isoformat(),report_sha256=sha(S.RUN/'raw/report.json'),inputs21_raw84_hashes_verified=True,paired_recomputed=True,relative_percent_from_xywh=relative,raw_only=True,no_flyvis_or_masa=True))
    names=dict(xywh='XYWHByteTrack',bytetrack='ByteTrack',botsort='BoT-SORT',tracktrack='TrackTrack');fmt=lambda x:f'{x:.4f}'.replace('.',',');text='Фиксированные20val BDD100K,исходные кадры,общие детекции YOLO26s960/conf0,01 и TrackEval. Только исходные трекеры,без FlyVis и сшивки MASA.\n\n| Система | HOTA без сшивки |\n|---|---:|\n'
    for a in S.ARMS:text+='| '+names[a]+' | '+fmt(r['pooled'][a]['HOTA'])+' |\n'
    text+='\nСредние парные разности HOTA по20видео относительно XYWHByteTrack:\n\n'
    for a in S.ARMS[1:]:d=r['paired'][a+'-xywh'];text+=names[a]+': '+fmt(d['mean_per_video'])+' ['+fmt(d['ci95'][0])+'; '+fmt(d['ci95'][1])+'], лучше на '+str(d['better'])+'/20.\n\n'
    text+='Общие HOTA и средние разности по видео имеют разный способ агрегирования. Ранее использовавшиеся val,не независимый итоговый тест. Эти числа нельзя сравнивать с таблицами после MASA или выдавать за результат FlyVis. Первичный источник: raw/report.json;хеши и проценты к базе: raw/verified_milestone.json. Доля потолка не определена.\n';(S.RUN/'raw/results.md').write_text(text,encoding='utf-8')
    S.journal('проверены полные входы и четыре исходных трекера20val','*Получено.* runs/z13_val20_matched_20261024/raw/report.json → pooled/paired: '+json.dumps(dict(HOTA={a:x['HOTA'] for a,x in r['pooled'].items()},paired=r['paired']),ensure_ascii=False)+';inputs_recovery1/report.json:21готовое задание.\n\n*Вывод.* Входы21 и исходные84задания проверены по хешам;exit0 и конец каждого дерева подтверждены. BoT-SORT и TrackTrack выше XYWH в исходной процедуре;это не прирост FlyVis и не рейтинг после MASA.\n\n*Решение.* Продолжается заранее зафиксированное common/10систем с общей MASA и контролями C/B/R1;проваленные новые варианты не продвигаются. Проценты к базе:raw/verified_milestone.json;доля потолка неизвестна.')
    for name,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_benchmark_adapter.py','tests/test_z13_ridge_live_shape.py','-q'])]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True);(W/('val20_milestone_'+name+'.log')).write_bytes(p.stdout);(W/('val20_milestone_'+name+'.stderr.log')).write_bytes(p.stderr);assert p.returncode==0,(name,p.returncode,p.stderr[-2000:]);print(name,0,flush=True)
    write(W/'val20_milestone_checks.json',dict(at=now().isoformat(),catalog_exit=0,tests_exit=0))


if __name__=='__main__':main()
