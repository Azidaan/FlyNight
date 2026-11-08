"""Register the conditional-qualified secondary and matched controls before20val."""
import sys,subprocess,shutil,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def image_rate_cutoff(name):
    values=[];target=0;entries=read(INVENTORY/'protocol.json')['videos']
    for video,e in entries.items():
        if e['split']!='fit20':continue
        path=RUN/'scores'/(video+'.json')
        if not path.exists():continue
        for row in read(path)['edges'].values():
            image=row['scores']['IMAGE'];target+=image is not None and image>=.5
            if row['scores'][name] is not None:values.append(float(row['scores'][name]))
    values.sort(reverse=True);cutoff='off' if not target or not values else values[-1] if target>=len(values) else (values[target-1]+values[target])/2 if values[target-1]>values[target] else values[target-1]
    return cutoff,dict(target=target,defined=len(values),actual=0 if cutoff=='off' else sum(v>=cutoff for v in values),fit_labels_used=False)


def main():
    import copy
    assert read(RUN/'queue_result.json')['passed'];conditional=read(RUN/'comparison_report.json');assert conditional['full_gate']['proceed'] and read(RUN/'independent_audit.json')['comparison_sha256']==sha(RUN/'comparison_report.json');selection=read(RUN/'selection.json');old=read(PARENT/'protocol.json');sources={};ready={};metrics={};motion=old['motion_sources'];videos=old['videos'];variants=copy.deepcopy(selection['configs']);rates={}
    for arm,name in [('RANDOM_IMAGE','RANDOM'),('PERM_IMAGE','PERM')]:
        cutoff,rates[arm]=image_rate_cutoff(name);variants[arm]=dict(enabled=True,score_name=name,threshold=cutoff)
    for cfg in variants.values():cfg['cap']=2
    for video,entry in videos.items():
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n;sources[n]=h
        ready[video]=dict(old['ready_feature_sources'].get(video,{}));receipt=PARENT/'feature_receipts'/(video+'.json');r=read(receipt);assert sha(receipt)==read(PARENT/'features_complete.json')['videos'][video]
        for n,h in r['artifacts'].items():
            if Path(n).suffix=='.npy' and Path(n).name.split('.')[-2] in ['C','R1','IMAGE','DIS']:assert sha(ROOT/n)==h;ready[video][Path(n).name]=dict(path=n,sha256=h)
        for value in ready[video].values():assert sha(ROOT/value['path'])==value['sha256'];sources[value['path']]=value['sha256']
        metrics[video]=[]
        for parent in [PARENT,ROOT/'runs/singleton_verify_20261108/full']:
            parent_protocol=read(parent/'protocol.json')
            for arm in parent_protocol['parameters']:
                path=parent/'metrics'/arm/(video+'.json');record=read(path);details=path.with_suffix('.details.json.gz');output=parent/'outputs'/arm/(video+'.json');assert sha(details)==record['details_sha256'] and sha(output)==record['output_sha256']
                for file in [path,details,output]:sources[file.relative_to(ROOT).as_posix()]=sha(file)
                metrics[video].append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),output_sha256=record['output_sha256']))
        for n in [motion[video]['path'],f'runs/bbox_confirmation_inventory_20261108/native/{video}.json',f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz']:sources[n]=sha(ROOT/n)
        assert sources[motion[video]['path']]==motion[video]['sha256']
        for folder in ['runtime_inputs','additional/B0']:
            oldpath=PARENT/folder/(video+'.json');sources[oldpath.relative_to(ROOT).as_posix()]=sha(oldpath)
        clock=PARENT/'runtime_inputs'/(video+'.json');dest=FULL/'runtime_inputs'/clock.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(clock,dest);sources[dest.relative_to(ROOT).as_posix()]=sha(dest)
    arms=list(variants);pairs=[(a,'B0') for a in ['SUPPORT','IMAGE','R1','DIS','FLYVIS']]+[('FLYVIS',a) for a in arms if a!='FLYVIS']+[('IMAGE',a) for a in ['RANDOM_IMAGE','PERM_IMAGE']];assert len(set(pairs))<=20
    eligible=conditional['full_gate']['qualified_preregistered_systems'];assert eligible==['IMAGE']
    p=dict(id='bbox_singleton_normalized_full_v1_20261108',registered=datetime.now(MSK).isoformat(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_sources_uncommitted=True,active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',videos=videos,native_config=old['native_config'],variants=variants,parameters={a:dict(alpha=.025,cap=2,threshold=v.get('threshold')) for a,v in variants.items()},thresholds={a:v.get('threshold') for a,v in variants.items()},alpha=.025,matching_threshold=.8,selected_system=selection['selected'],eligible_systems=eligible,primary_comparisons=pairs,ready_feature_sources=ready,ready_metric_sources=metrics,motion_sources=motion,image_frequency_controls=rates,
        selected_primary_status='SUPPORT remains the fit-selected primary and failed the conditional foreign-link guard; it is retained as a control, not promoted by full20. IMAGE is a preregistered secondary qualified by the fixed screen. No26score thresholds or20valoutcomes are used to choose new parameters.',
        scope='Historical20val, not independent final data. Exactly the same native physical outputs and original internal assignments. Short-gap pilot: age2means one intermediate missing frame; passing cannot establish resolution of all long occlusions or problems2–4.',
        acceptance=read(RUN/'protocol.json')['full_acceptance'],image_attribution='An IMAGE system benefit overB0does not alone prove spatial identity information. Added image-rate random/permutation controls use only fit score-pass counts and are frozen before20val. Preserve their comparisons even if unfavorable.',inference=dict(family=20,seed=42,repetitions=20000,unit='paired_video'))
    write(FULL/'protocol.json',p,new=True)
    sources.update(phase()['sources']);sources.update(read(RUN/'comparison_phase.json')['sources']);inherited=read(ROOT/'runs/native_confirm_risk_20261108/full_phase.json')['sources']
    for n,h in inherited.items():assert sha(ROOT/n)==h,n;sources[n]=h
    for path in [RUN/'selection.json',RUN/'comparison_report.json',RUN/'independent_audit.json',FULL/'protocol.json']:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    cat=read(ROOT/'experiments/project_history.json');card=next(e for e in cat['experiments'] if e['id']=='bbox_singleton_normalized_maps_v1_20261108');card['state']='running';card['obtained']='Восстановлены132отсутствующие последовательностиC,20готовыхпереиспользованы;322выреза=184новых+138готовых. Независимо проверены532пространственныхоценки,414журналов,379764кадра,194назначения,12интервалов. На26геометрия13правильных/1чужая,IMAGE7/0,FlyVis3/0. Условие полного испытания прошёл только заранее заданныйIMAGE; выбраннаянаfitгеометрияSUPPORTнепрошла. FLYVISнепревзошёлсогласованныеконтроли (runs/bbox_normalized_signals_20261108/comparison_report.json; independent_audit.json).';card['interpretation']='Относительное совмещение вернуло возможность сравнения, но само не устранило чужую связь. Обычное изображение отсекло её и сохранило7известныхправильных; пространственныйFlyVisсохранил3. Полнаяпользасистемынеизмерена. ОсновнойвыбранныйSUPPORTнепрошёлпредварительныйкритерий; IMAGEпроверяетсякак заранее заданнаяальтернатива.';card['decision']='Зарегистрирован полный20valпилот IMAGEсо всеми согласованными сигналами и дополнительными контролями частоты изображения. Не менять пороги; не повторять готовые Nativeтраектории,карты,движение,метрики. Первую проблему не закрывать по коротким разрывам; проблемы2–4отложены.';card['acceptance'][0].update(met=True,evidence='runs/bbox_normalized_signals_20261108/inventory.json: supported fit positives4; input_complete.json; features_complete.json');card['acceptance'][1].update(met=True,evidence='runs/bbox_normalized_signals_20261108/comparison_report.json: qualified_preregistered_systems=[IMAGE]; independent_audit.json');card['evidence'] += [dict(kind='file',ref='runs/bbox_normalized_signals_20261108/comparison_report.json',claim='Условный результат и неуспех выбранной геометрии'),dict(kind='file',ref='runs/bbox_normalized_signals_20261108/independent_audit.json',claim='Независимая проверка'),dict(kind='file',ref='runs/bbox_normalized_signals_20261108/full/protocol.json',claim='Полное испытание до20valисходов')];write(ROOT/'experiments/project_history.json',cat)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK');entry=f'\n\n**{stamp}** — Первая проблема: пространственное сравнение доступно; IMAGEпрошёл условие полного пилота, FlyVisне прошёл.\n\n*Получено (`bbox_singleton_normalized_maps_v1_20261108`).* Восстановлены132последовательности,20готовыхкартпереиспользованы;322выреза:184новых,138готовых. На26геометрияSUPPORT/BBOXдала13правильных/1чужую,IMAGE7/0,FlyVis3/0. У IMAGE J=7,95%ДИна100известных[2,632;12,069],поправленный20[0,909;15,000]; уFlyVisJ=3,нижняя95%граница0. ДобавкаFlyVisнеподтверждена. Проверены532оценки,414журналов,379764кадра,194назначения,12интервалов (`runs/bbox_normalized_signals_20261108/comparison_report.json`; `independent_audit.json`). У3из4правильныхfitотказов старойгеометрии нижнийдопустимыйполныйразмерпревышалверхний;четвёртыйотказпо положениюкрая,неячейкам (`runs/identity_evidence_20261108/alignment_diagnostic.json`).\n\n*Решение.* ВыбранныйнаfitSUPPORTосталсянепрошедшимусловие;заранеезаданнаяальтернативаIMAGEдопущена к20valпо прежнемукритерию. До20valзафиксированы дополнительные случайныйиперестановочныйконтролисчастотойIMAGEпоfitбезметок. Всего11вариантов,поправкана20сравнений;_NATIVEтраекториине пересчитываются, восстанавливаютсятолькоотсутствующиекарты. Полный критерий и ограничения вреда прежние (`runs/bbox_normalized_signals_20261108/full/protocol.json`). Давность2—одинпромежуточныйпропущенныйкадр; это не решение длительныхперекрытийиливсейпервойпроблемы. Проблемы2–4отложены.\n';marker='## Журнал изменений';i=text.index(marker)+len(marker);plan.write_text(text[:i]+entry+text[i:],encoding='utf-8')


if __name__=='__main__':main()
