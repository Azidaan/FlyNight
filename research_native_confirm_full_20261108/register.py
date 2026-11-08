"""Register the full pilot only after the independently audited new gate passes."""
import sys,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_20261108.common import *


def main():
    assert read(RUN/'queue_result.json')['passed'];conditional=read(RUN/'comparison_report.json');assert read(RUN/'independent_audit.json')['comparison_sha256']==sha(RUN/'comparison_report.json')
    if not conditional['full_gate']['proceed']:
        write(RUN/'full_skipped.json',dict(registered=datetime.now(MSK).isoformat(),reason='Registered conditional gate failed; no full native recovery or model simulation started.',source_sha256=sha(RUN/'comparison_report.json')),new=True);return False
    selection=read(RUN/'selection.json');old=read(PARENT/'protocol.json');assert read(PARENT/'independent_audit.json')['passed'];source={};ready={};metrics={};motion={};videos=old['videos'];base_precision=BASE/'full_precision'
    for video,entry in videos.items():
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n;source[n]=h
        ready[video]={};metrics[video]=[];seal=read(PARENT/'completed'/(video+'.json'));source[(PARENT/'completed'/(video+'.json')).relative_to(ROOT).as_posix()]=sha(PARENT/'completed'/(video+'.json'))
        for arm in old['parameters']:
            path=PARENT/'metrics'/arm/(video+'.json');n=path.relative_to(ROOT).as_posix();assert sha(path)==seal['artifacts'][n];record=read(path);details=path.with_suffix('.details.json.gz');dn=details.relative_to(ROOT).as_posix();assert sha(details)==seal['artifacts'][dn]==record['details_sha256'];source[n]=sha(path);source[dn]=sha(details);metrics[video].append(dict(path=n,sha256=sha(path),output_sha256=record['output_sha256']))
        for folder in [PARENT,base_precision]:
            for path in (folder/'rollouts').glob('*/'+video+'.receipt.json'):
                receipt=read(path);source[path.relative_to(ROOT).as_posix()]=sha(path)
                for n,h in receipt.get('feature_files',{}).items():
                    path=ROOT/n
                    if path.name.split('.')[-2] not in ['C','R1','IMAGE','DIS']:continue
                    assert sha(path)==h,n;source[n]=h;ready[video].setdefault(path.name,dict(path=n,sha256=h))
            marker=folder/'cache_reuse'/(video+'.json')
            if marker.exists():
                source[marker.relative_to(ROOT).as_posix()]=sha(marker)
                for n,r in read(marker).get('features_reused',{}).items():
                    path=ROOT/n
                    if path.name.split('.')[-2] not in ['C','R1','IMAGE','DIS']:continue
                    assert sha(path)==r['sha256'];source[n]=r['sha256'];ready[video].setdefault(path.name,dict(path=n,sha256=r['sha256']))
        path=PARENT/'ordinary_motion'/(video+'.npz');marker=path.with_suffix('.json');m=read(marker);assert sha(path)==m['sha256'] and m['source_sha256']==entry['sources'][entry['video_path']];motion[video]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path));source[path.relative_to(ROOT).as_posix()]=sha(path);source[marker.relative_to(ROOT).as_posix()]=sha(marker)
        clock=PARENT/'runtime_inputs'/(video+'.json');dest=FULL/'runtime_inputs'/clock.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(clock,dest);source[dest.relative_to(ROOT).as_posix()]=sha(dest)
        base_return=PARENT/'additional/B0'/(video+'.json');source[base_return.relative_to(ROOT).as_posix()]=sha(base_return)
        # Bind the already prepared native-precision input container by its
        # immutable original manifest. Receptors verifies it before and after use.
        processed=base_precision/'processed'/(video+'.json');source[processed.relative_to(ROOT).as_posix()]=sha(processed)
    chosen=selection['selected_system'];arms=list(selection['variants']);pairs=[(chosen,'B0'),('AGE_FIT','B0'),('AGE6','B0'),('GEOMETRY','B0'),('IMAGE','B0')]+[('FLYVIS',a) for a in arms if a!='FLYVIS'];assert len(set(pairs))<=16
    p=dict(id='native_confirm_empirical_risk_full_v1_20261108',registered=datetime.now(MSK).isoformat(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_sources_uncommitted=True,active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',videos=videos,native_config=old['native_config'],parameters={a:dict(alpha=.2,threshold=cfg.get('threshold'),cap=cfg['cap']) for a,cfg in selection['variants'].items()},variants=selection['variants'],thresholds={a:cfg.get('threshold') for a,cfg in selection['variants'].items()},recency_cap=selection['shared_cap'],alpha=.2,matching_threshold=.8,selected_system=chosen,eligible_systems=list(dict.fromkeys([chosen,'AGE_FIT','AGE6','GEOMETRY','IMAGE','FLYVIS'])),primary_comparisons=pairs,inference=dict(family=16,repetitions=20000,seed=42,unit='paired_video'),ready_feature_sources=ready,ready_metric_sources=metrics,motion_sources=motion,
        selection_sha256=sha(RUN/'selection.json'),conditional_sha256=sha(RUN/'comparison_report.json'),conditional_audit_sha256=sha(RUN/'independent_audit.json'),
        acceptance=read(RUN/'protocol.json')['full_acceptance'],secondary_acceptance='The fixed strong no-FlyVis controls and FLYVIS may be reported as secondary predeclared systems if they meet the identical full guards under the same conservative family16 correction. Do not reselect thresholds on20val; selected-system and secondary outcomes remain separate.',
        scope='Historical20val, not an independent final test. No image/detector/tracker-output recomputation except one necessary native metadata recovery; all ready model maps, motion, metrics and sparse return counts reused when identical. Keep every native physical box and change only canonical public IDs after actual first native confirmation.')
    write(FULL/'protocol.json',p,new=True)
    inherited=read(ROOT/'runs/singleton_verify_20261108/full_phase.json')['sources'].copy()
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    source.update(inherited);source.update(read(RUN/'phase.json')['sources'])
    for path in Path(__file__).parent.glob('*.py'):source[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [RUN/'selection.json',RUN/'comparison_report.json',RUN/'independent_audit.json',FULL/'protocol.json']:source[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=source),new=True)
    trace_entries={v:dict(raw=e['detections'],baseline=e['raw_array'],frames=e['frames'],sources=e['sources'],export=None) for v,e in videos.items()};write(FULL/'native_trace_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources=source,videos=trace_entries,native_config=p['native_config'],require_complete_cached_B0_equality=True),new=True)
    catalog=read(ROOT/'experiments/project_history.json');e=next(e for e in catalog['experiments'] if e['id']=='native_confirm_empirical_risk_v1_20261108');e['state']='running';e['obtained']='После подбора только наfit20 выбран FlyVis с ограничением 6 кадров после последнего наблюдения. На26 он дал 7 правильных/0 наблюдаемых чужих связей; геометрия с тем же ограничением — 9/0. Дополнительная польза FlyVis не установлена. Условие полного пилота пройдено; все карты уже готовы, новых вызовов ядра для этого сравнения не было (runs/native_confirm_risk_20261108/selection.json; comparison_report.json; independent_audit.json).';e['interpretation']='Эмпирическое ограничение изменило выбор системы, но ноль ошибок на малой выборке не является гарантией. Полная HOTA и устойчивость личности ещё не проверены.';e['decision']='Выполнить зарегистрированный полный пилот с точным сохранением штатных рамок и заранее заданными ограничениями вреда. Проблемы 2–4 отложены.';e['evidence'] += [dict(kind='file',ref='runs/native_confirm_risk_20261108/full/protocol.json',claim='Полный протокол до новых20valисходов'),dict(kind='file',ref='runs/native_confirm_risk_20261108/comparison_report.json',claim='Условный гейт пройден, не итог полного трекера')];write(ROOT/'experiments/project_history.json',catalog)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK');entry=f'\n\n**{stamp}** — Первая проблема: допуск по наблюдаемым ошибкам прошёл предварительное условие.\n\n*Получено (`native_confirm_empirical_risk_v1_20261108`).* Наfit20 выбран FlyVis с давностью не более6кадров после наблюдения, то есть максимум5промежуточных пропусков; 2известных правильных/0чужих связей. На26 — 7/0, геометрия при том же ограничении — 9/0; добавка FlyVis не подтверждена. Независимо проверены460журналов,542520кадров причинной выдачи и13интервалов. Переиспользовано1394готовых исхода; вычислены66уникальных новых вариантов допуска; ядро, детектор и кадры повторно не запускались (`runs/native_confirm_risk_20261108/comparison_report.json`; `independent_audit.json`).\n\n*Решение.* Зарегистрирован полный20valпилот с16сравнениями, неизменными физическими рамками и прежними критериями вреда. Выбранная система и заранее заданные сильные контроли оцениваются раздельно; пороги по20val не меняются (`runs/native_confirm_risk_20261108/full/protocol.json`). Проблема1ещё открыта,2–4отложены.\n';marker='## Журнал изменений';assert text.count(marker)==1;plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8',newline='\n');status('full_pilot_registered',videos=len(videos),selected_system=chosen,deadline=p['deadline']);return True


if __name__=='__main__':main()
