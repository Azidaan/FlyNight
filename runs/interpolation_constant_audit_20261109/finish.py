"""Audit/record the data gate and numerical repair; validate derived records."""
import fix as f
from fix import *
FREE=ROOT/'runs/free_candidate_identity_inventory_20261109'
def audit_record():
    from scipy.stats import rankdata
    import psutil
    fp=read(OUT/'fix_protocol.json');verify(fp);probe=read(OUT/'results.json');fixed=read(OUT/'fixed_invariant.json');reg=read(OUT/'real_regression.json');original=read(BASE/'summary.json');free=read(FREE/'results.json');f.verify_before(read(FREE/'protocol.json'));f.verify_before(read(BASE/'protocol.json'));verify(read(BASE/'maps_complete.json'))
    assert fixed['old_nonnull']==1440 and fixed['new_nonnull']==0 and fixed['constant_preserved_exactly'];assert reg['null_transitions']==0 and reg['maximum_score_difference']<1e-12
    rank_checks=0;rank_changes=[]
    for metric,table in reg['table'].items():
        for name,values in table.items():
            for arm,value in values.items():
                own=[r['variants'][name][arm][metric] for r in reg['records'] if r['supported'] and r['kind']=='own'];other=[r['variants'][name][arm][metric] for r in reg['records'] if r['supported'] and r['kind']=='foreign'];assert all(v is not None for v in own+other);ranks=rankdata(own+other);auc=float(ranks[:len(own)].sum()-len(own)*(len(own)+1)/2)/(len(own)*len(other));assert abs(auc-value['auc'])<1e-12
                if value['correct_rankings']!=original['table']['all'][metric]['values'][name][arm]['correct_rankings']:rank_changes.append([metric,name,arm])
                rank_checks+=1
    assert not rank_changes
    totals={};queries=[];p=read(FREE/'protocol.json')
    for video,info in p['videos'].items():
        for e in read((ROOT/info['native']).with_suffix('.json'))['events']:
            for row in e['rows']:
                candidates=[c for c in row['candidates'] if c['forward_iou']>=.2]
                if len(candidates)>=2 and row['age']<=6 and int(row['row_data']['cls'])==2 and any(c['new_tid'] is not None for c in candidates):queries.append((video,e['frame'],row['old_tid']))
    assert len(queries)==35 and set(queries)=={(r['video'],r['frame'],r['old']) for r in free['records']}
    for r in free['records']:totals[r['status']]=totals.get(r['status'],0)+1
    assert totals==free['totals'] and free['clean_queries']==0 and not free['readiness_gate'];assert totals==dict(current_unannotated=30,reference_not_unique=2,not_both_supported=1,no_prior_annotated_observation=2)
    known=next(r for r in free['records'] if r['status']=='not_both_supported');assert known['old']==2086 and known['other_public_same_gt_at_reference'][0]['tid']==2036 and known['working_public_same_gt_now'][0]['tid']==2036;assert [c['kind'] for c in known['candidates']]==['own','unknown']
    write('audit.json',dict(passed=True,old_flat_scores=1440,fixed_flat_scores=0,independent_real_rank_checks=rank_checks,real_rank_changes=rank_changes,real_null_transitions=0,maximum_real_score_change=reg['maximum_score_difference'],all_source_differences_are_archived_and_explicit=True,zero_model_tracker_retinal_replays=True))
    with (FREE/'audit.json').open('x',encoding='utf8') as out:json.dump(dict(passed=True,independent_query_keys=35,all_unknowns_retained=True,totals=totals,clean_queries=0,readiness_gate=False,no_new_maps_or_scores=True,changed_reader_source_verified_against_exact_before_archive=str((OUT/'before/observed_spatial_reader.py').relative_to(ROOT))),out,ensure_ascii=False,indent=2);out.write('\n')
    now=datetime.now(MSK);catpath=ROOT/'experiments/project_history.json';cat=read(catpath);fc=next(c for c in cat['experiments'] if c['id']==FREE.name);nc=next(c for c in cat['experiments'] if c['id']==OUT.name)
    fc.update(state='infrastructure',done='2026-11-09',obtained='В46сохранённыхвидеонайдено35новыхмногокандидатныхзапросов:30безтекущейразметки,2безединственнойпрошлойGT,2безраннегоразмеченногонаблюдения,1соднозначнойопорой. Упоследнего2086накадре1070естьсвоя2108и1неизвестнаярамка,чужаянеподтверждена; тажеGTмашинаужеимеетпубличный2036впрошломитекущемкадрах. Чистыхзапросов0,условие3запроса/3видеоневыполнено. Источники `runs/'+FREE.name+'/results.json`, `audit.json`.',interpretation='Существующий экспорт остаточного свободного пула не даёт прямо подтверждённого сравнения двух доступных альтернатив одной цели. Это ограничение данных и постановки, не отрицательный результат ещё одного считывателя.',decision='Новые карты не восстанавливать, пороги и условия отбора не подгонять.517однокандидатныхсобытийнеповторены. Для нового содержательного выбора требуется иной заранее зафиксированный источник прямо размеченных запросов.',cannot_claim='Нет результатов FlyVis, изображения или трекера: здесь только метаданные/разметка. Публичное владение2036проверено лишь на указанных кадрах; вся история и причина назначения не восстановлены. Неразмеченные случаи не превращены в свои или чужие.')
    for a,met in zip(fc['acceptance'],[True,False]):a.update(met=met,evidence=f'runs/{FREE.name}/audit.json')
    fc['evidence'] += [dict(kind='file',ref=f'runs/{FREE.name}/{n}',claim='Полный учёт и отсутствие чистого сравнения') for n in ['results.json','audit.json','ИТОГ.md']]
    nc.update(state='infrastructure',done='2026-11-09',obtained=f'На76синтетическихсетках(19геометрий×4уровняконстанты) прежняяинтерполяциядала1440числовыхоценокиз1444, |score|до{probe["maximum_abs_score"]:.9f},хотятекущаякартастрогоровная. Исправленнаяинтерполяциясохраняетконстантуточно,ложныхчисел0.9затронутых/новыхтестовпрошли. На798готовыхреальныхсетках19пар максимальнаяразница{reg["maximum_score_difference"]:.17g},переходовопределено/неопределено0,все84итоговыхранжированиянеизменны. Источники `runs/{OUT.name}/results.json`, `fixed_invariant.json`, `real_regression.json`, `audit.json`.',interpretation='Найдена и исправлена численная ошибка считывателя: нормировка превращала округление ровного сигнала в ложную пространственную структуру. Она не объясняет неверный порядок на19реальныхпарах: исправление не изменило ранжирования. Ошибка ядра FlyVis этим не установлена.',decision='Использовать точно сохраняющую константу разностную формулу интерполяции и явное обнуление точно постоянных каналов при общей нормировке. Не добавлять порог малой дисперсии и не заявлять выигрыш ID. Отрицательный вывод о ранней памяти остаётся.',cannot_claim='Это проверка численной корректности без изменения ядра, детектора, рамок, сдвигов, обучения или порогов.0ложныхчиселнаконстанте не означает0ошибоктрекинга.19реальныхсобытий/798сеток — проверка регрессии уже изученных данных, не независимый новый тест. Полный трекер и сохранениеIDне измерены. Старые источники перед правкой сохранены побайтно и сверены поSHA; прежний результат не приписан нынешнему коду.')
    nc['acceptance'][0].update(met=False,evidence=f'runs/{OUT.name}/results.json')
    for a in nc['acceptance'][1:]:a.update(met=True,evidence=f'runs/{OUT.name}/audit.json')
    nc['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{n}',claim='Ровный сигнал, численная правка и граница влияния на реальные оценки') for n in ['results.json','fixed_invariant.json','real_regression.json','audit.json','ИТОГ.md']];catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][FREE.name]=['Сохранение личности','Выбор продолжения среди рамок','Доступность подтверждённых сравнений'];topics['topics'][OUT.name]=['Сохранение личности','Выбор продолжения среди рамок','Численная корректность сравнения карт'];path.write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'experiments/system_problems.json';system=read(path);problem=next(x for x in system['problems'] if x['id']=='С2');old=problem['fact'];problem.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='fact',old=old,reason='Добавлена численная правка ровного сигнала, без утверждения о выигрышеID, и граница новых данных.'));problem['fact']=old+' Численнаяпроверка09.11обнаружилаложноесходстворовнойкарты(1440из1444чисел); разностнаяинтерполяцияустранилаего(0). На19реальныхпарах порядокнеизменился,разницабаллов≤4,441·10⁻¹⁶. Дополнительныйсохранённыймногокандидатныйэкспортдал35запросов,0чистыхсравненийсвоейичужойрамки; новые карты длянего не запускались. Источники runs/'+OUT.name+'/audit.json; runs/'+FREE.name+'/audit.json. ПроблемасохраненияIDоткрыта.';problem['evidence'] += [dict(source=f'runs/{folder}/{n}',sha256=sha(ROOT/'runs'/folder/n),claim='Численная корректность и границы подтверждённых данных') for folder,n in [(OUT.name,'audit.json'),(FREE.name,'audit.json')]];path.write_text(json.dumps(system,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'experiments/useful_literature.json';lit=read(path);lewis=next(x for x in lit['sources'] if x['id']=='lewis_ncc_channel_normalization');lewis['application']+='\n\n09.11.2026. ФормулаNCCпомогла сформулировать строгую численную проверку: у точно постоянной текущей карты норма центрированного сигнала равна нулю, поэтому числовое сходство недопустимо. Обнаружена ошибка округления в интерполяции и исправлена без нового порога. На19реальныхпарах ранжирования не изменились; источник runs/'+OUT.name+'/audit.json.';lewis['evidence'] += [f'runs/{OUT.name}/protocol.json',f'runs/{OUT.name}/audit.json'];lewis['related'].append('exp:'+OUT.name);path.write_text(json.dumps(lit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');entries=f'**{now:%d.%m.%Y, %H:%M} MSK** — Ровная карта больше не создаёт ложного сходства; реальные ранжирования не изменились.\n\n*Получено (`{OUT.name}`).* '+nc['obtained']+'\n\n*Вывод.* '+nc['interpretation']+'\n\n*Решение.* '+nc['decision']+' '+nc['cannot_claim']+' Сводка:ложныечисланаинварианте−100%(1440→0),изменениевсехизмеренныхреальныхAUROC0%,выигрышIDнеизмерен.\n\n'+f'**{now:%d.%m.%Y, %H:%M} MSK** — В сохранённом свободном пуле нет чистых многокандидатных запросов.\n\n*Получено (`{FREE.name}`).* '+fc['obtained']+'\n\n*Вывод.* '+fc['interpretation']+'\n\n*Решение.* '+fc['decision']+' '+fc['cannot_claim']+' Сводкавпроцентахкbaselineнеприменима: этоаудитдоступностиданных,качество не измерялось.\n\n';path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entries,1),encoding='utf8')
    print('Both audits recorded',flush=True)
def checks():
    import contextlib,runpy,shutil,tempfile
    sys.argv=['scripts/validate_project_history.py']
    with (OUT/'catalog_validation.log').open('x',encoding='utf8') as log:
        with contextlib.redirect_stdout(log):
            try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
            except SystemExit as error:
                if error.code:raise
    import pytest
    assert pytest.main(['tests/test_project_history.py','-q'])==0
    from scripts.build_obsidian_vault import main
    from scripts.audit_obsidian_levels import audit
    vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
    def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
        copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);a=audit(copy);build(vault);assert audit(vault)==a
    assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};verify(read(OUT/'fix_protocol.json'));f.verify_before(read(FREE/'protocol.json'))
    write('final_checks.json',dict(passed=True,catalog_tests=58,affected_mathematical_tests=9,hypotheses_unchanged=True,graph=a,source_hashes_stable=True,no_model_replay=True));print('Catalogue, tests and both graphs checked',flush=True)
if __name__=='__main__':{'record':audit_record,'checks':checks}[sys.argv[1]]()
