"""Fit-only metadata gate for recalibrating a readout with actual foreign examples."""
import os,sys,ast
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from collections import Counter
from research_identity_assignment_run_20261108.common import *
RUN=ROOT/'runs/bbox_fit_readout_readiness_20261108'


def register():
    files=['research_bbox_fit_readout_readiness_20261108.py','tracking/bbox_singleton_gate.py','tracking/box_normalized_correspondence.py','tracking/own_template_verification.py','runs/bbox_normalized_signals_20261108/full/comparison_report.json','runs/bbox_normalized_signals_20261108/full/independent_audit.json'];assert read(ROOT/'runs/bbox_normalized_signals_20261108/full/queue_result.json')['passed']
    write(RUN/'protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in files},question='Do exact ready fit20maps contain at least3known positive and3known foreign ordinary-singleton pairs for a meaningful readout-threshold calibration?',scope='Metadata and availability only. Use fit20, ordinaryIoU0.2,age<=6andthe same observed-box-relative alignment. Actual known anchors select training examples only, never runtime candidates. No26/20valthreshold fitting, no new neural/image/flow/tracker/evaluator run. Source thresholds and completed experiments unchanged.',strongest_alternative='The fixed readout thresholds may be miscalibrated in the new alignment context. The narrow fit gate had4known positive and0foreign cases; FLYVIS permitted only1full20edge, so lack of a tracking gain does not establish absence of identity information.',gate='Only if>=3positive and>=3foreign fit edges have both exact ready causal sequence maps and shared3cell support may a separately registered CPU-only calibration be considered. Otherwise no calibration or neural restoration is authorized by this metadata step.'),new=True)
    cat=read(ROOT/'experiments/project_history.json');cat['experiments'].append(dict(id='bbox_fit_readout_readiness_v1_20261108',title='Одна свободная рамка — достаточно ли готовых положительных и чужих примеров подбора',area='flyvis',state='planned',started=datetime.now(MSK).date().isoformat(),question='Есть ли наfit20готовые пространственные карты известных правильных и чужих одиночных продолжений для калибровки порога?',hypothesis='Один фиксированный порог в изменённом совмещении может почти отключить сигнал; прежде нового вывода нужны положительные и отрицательные примеры только на подборе.',expected='Измерить готовность без новых нейронных расчётов и не подбирать по26или20val.',acceptance=[dict(criterion='Не менее3известныхправильныхи3чужихfit-парсдвумяготовымикаузальнымикартамиитремяобщимиреференснымиячейками.',met=None)],obtained='',interpretation='',decision='',cannot_claim='Это готовность данных и доступность обучения, не польза порога, FlyVis или полного трекера. Только первая проблема;2–4отложены.',evidence=[dict(kind='file',ref='runs/bbox_fit_readout_readiness_20261108/protocol.json',claim='Условия до проверки доступности')]))
    write(ROOT/'experiments/project_history.json',cat)


def main():
    import numpy as np
    from tracking.bbox_singleton_gate import proposals
    from tracking.box_normalized_correspondence import box_transform
    from tracking.own_template_verification import correspondence
    from tracking.spatial_identity_assignment import frame_points
    from research_identity_assignment_run_20261108.full_planner import dis_domain
    from research_singleton_verify_20261108.common import sequence_key
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');entries=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['videos'];ready={};sources={};counts=Counter();rows=[];verified=set()
    for name in ['bbox_normalized_signals_20261108','singleton_verify_20261108','native_confirm_policy_20261108','identity_assignment_20261108']:
        folder=ROOT/'runs'/name;manifest=folder/'features_complete.json';sources[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
        for key,h in read(manifest)['files'].items():ready.setdefault(key,dict(path=(folder/'features'/(key+'.npz')).relative_to(ROOT).as_posix(),sha256=h))
    for video,entry in entries.items():
        if entry['split']!='fit20':continue
        meta=ROOT/'runs/bbox_confirmation_inventory_20261108/native'/(video+'.json');labels=ROOT/'runs/bbox_singleton_pilot_20261108/labels'/(video+'.json');sources[meta.relative_to(ROOT).as_posix()]=sha(meta);sources[labels.relative_to(ROOT).as_posix()]=sha(labels);lookup=read(labels)['rows']
        for d in read(meta)['events']:
            for e in proposals(d,cap=6,threshold=.2,bidirectional=False):
                label=lookup[f"{d['frame']}|{e['old_tid']}|{e['new_tid']}"]
                if not label['known']:counts['unknown_training_pairs_excluded']+=1;continue
                kind='positive' if label['positive'] else 'foreign';counts['known_'+kind]+=1;row=e['row_data'];c=next(c for r in d['rows'] if r['old_tid']==e['old_tid'] for c in r['candidates'] if c['new_tid']==e['new_tid']);transform=box_transform(row['history'][-1]['box'],c['box']);cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pairs=correspondence(transform,row['history'],row['current_crop'],c['box'],g,cv,rv);ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];keys=[sequence_key(video,seq) for seq in [ref,cur]];available=len(ref)>=2 and pairs is not None and all(k in ready for k in keys)
                if available:
                    for key in keys:
                        if key not in verified:
                            r=ready[key];assert sha(ROOT/r['path'])==r['sha256'];verified.add(key)
                            with np.load(ROOT/r['path'],allow_pickle=False) as z:assert z['C'].shape==(721,8) and all(n in z.files for n in ['R1','IMAGE','DIS','DIS_VALID'])
                    counts['ready_'+kind]+=1
                else:counts['missing_or_unsupported_'+kind]+=1
                rows.append(dict(video=video,frame=d['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],kind=kind,available=bool(available),reference_key=keys[0],current_key=keys[1],row_data=row,box=c['box'],transform=transform,column=e['column'],age=e['age'],cost=e['cost']))
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    proceed=counts['ready_positive']>=3 and counts['ready_foreign']>=3;write(RUN/'readiness.json',dict(passed=True,counts=dict(counts),rows=rows,ready_sequences={k:ready[k] for k in verified},source_checksums=sources,calibration_gate=bool(proceed),new_neural_image_tracker_evaluator_calls=0),new=True)
    cat=read(ROOT/'experiments/project_history.json');card=next(e for e in cat['experiments'] if e['id']=='bbox_fit_readout_readiness_v1_20261108');card['state']='infrastructure';card['done']=datetime.now(MSK).date().isoformat();card['obtained']=f'Наfit20: известныеправильные{counts["known_positive"]},чужие{counts["known_foreign"]};готовыподдержанныесдвумяточнымикартами{counts["ready_positive"]}и{counts["ready_foreign"]}. Новыхрасчётовядра,детектора,кадровитрекеранет(runs/bbox_fit_readout_readiness_20261108/readiness.json).';card['interpretation']='Готовность примеров не доказывает, что калибровка улучшит считыватель. Предыдущее ограничение давности2оставило только4положительныхбезизвестныхчужихпримеров.';card['decision']='CPU-калибровку можно только отдельно зарегистрировать до новых оценок;20valи26по порогу не подбирать.' if proceed else 'Не запускать калибровку по неполноценному готовому подмножеству. Подготовить отдельный протокол недостающих fit-представлений; без нового нейронного запуска этой проверкой.';card['acceptance'][0].update(met=bool(proceed),evidence='runs/bbox_fit_readout_readiness_20261108/readiness.json');write(ROOT/'experiments/project_history.json',cat)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');entry=f'\n\n**{datetime.now(MSK).strftime("%d.%m.%Y, %H:%M MSK")}** — Первая проблема: готовность калибровки считывателя на настоящих чужих fit-примерах.\n\n*Получено (`bbox_fit_readout_readiness_v1_20261108`).* {card["obtained"]}\n\n*Граница.* Калибровка и новые нейронные расчёты этим шагом не запускаются. Порог0,25на новом совмещении разрешил всего1вмешательствоFlyVisна20val; это допускает объяснение через избыточный отказ, а не отсутствие пространственной информации. {card["decision"]} Активна только первая проблема;2–4отложены.\n';marker='## Журнал изменений';i=text.index(marker)+len(marker);plan.write_text(text[:i]+entry+text[i:],encoding='utf-8');print(dict(counts),proceed)


if __name__=='__main__':
    if '--register' in sys.argv:register()
    else:main()
