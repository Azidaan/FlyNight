"""Independent accounting, human reports and derived-record checks."""
import audit as a
from audit import *
F=ROOT/'runs/candidate_relative_alignment_20261109'
FOLDERS=[DATA,BASE,F,OUT]
def text(path,value):Path(path).write_text(value.strip()+'\n',encoding='utf8')
def record():
    p=read(OUT/'protocol.json');s.verify(p);r=read(OUT/'results.json');bp=read(BASE/'protocol.json');q=read(BASE/'summary.json');fr=read(F/'readiness.json');s.verify(read(F/'protocol.json'))
    assert r['totals']==dict(queries=12,all_edges_rejected=6,some_locally_admitted=6,own_admitted=4,foreign_admitted=2,admitted_with_cheaper_competitor=6,admitted_without_cheaper_competitor=0)
    # Check independent scalar accounting, original state flags and public times.
    source=read(DATA/'protocol.json');n=0
    for case,row in zip(bp['cases'],r['records']):
        ctx=read(ROOT/source['videos'][case['video']]['context'])['entries'][str(case['uid'])]
        with np.load(ROOT/source['videos'][case['video']]['native'],allow_pickle=False) as z:outputs=z['outputs']
        for k,candidate in zip(row['candidates'],case['candidates']):
            vector=ctx['context'][ctx['choices'].index(candidate['column'])];assert k['cost']==vector[0] and k['lowest_competing_cost']==vector[1] and vector[4]==0
            assert k['local_admitted']==(vector[0]<=.8);assert k['cheaper_competing_edge']==(vector[1]<vector[0])
            for owner in k['public_owners']:
                frames=[int(x[0]) for x in outputs if int(x[5])==owner['tid']];assert owner['first_public']==min(frames)
                before=[f for f in frames if f<case['frame']];assert owner['last_public_before']==(max(before) if before else None)
            n+=1
    assert n==24 and fr['supported_queries']==5 and not fr['passed'] and not (F/'results.json').exists()
    assert sum(x['supported'] for x in fr['plans'])==5
    # Independent algebraic witness for each retained affine cell coordinate.
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');pts=np.array(geometry['points']);maxerr=0.
    for case,row in zip(bp['cases'],fr['plans']):
        rc=np.array(case['sequence'][case['reference_relative']][1]);ct=np.array(case['sequence'][-1][1]);b0=np.array(case['reference_raw']['state_box'])
        for candidate,k in zip(case['candidates'],row['candidates']):
            ix=np.array(k['cells'],int)
            if not len(ix):continue
            x=pts[ix]*(rc[2:]-rc[:2])/geometry['eye_size']+rc[:2];bb=np.array(candidate['box']);uv=(x-b0[:2])/(b0[2:]-b0[:2]);physical=bb[:2]+uv*(bb[2:]-bb[:2]);expected=(physical[:,None]+np.array(candidate['plan']['offsets'])-ct[:2])/((ct[2:]-ct[:2])/geometry['eye_size']);err=float(np.max(abs(expected-np.array(k['relative']['query']))));maxerr=max(maxerr,err)
            assert k['prediction']['cells']==k['relative']['cells']==k['cells']
    assert maxerr<1e-10
    write('independent_audit.json',dict(passed=True,edges_verified=24,public_first_and_previous_frames_verified=True,unknown_solver_stage_retained=True,threshold_source='historical_freeze_bench.py exact historical SHA-256; Git bd01d1026ae9456a6762ca62cd086cf4f45fe2fe, original CRLF recovered',initial_registration_stopped_before_protocol='Current freeze_bench source did not match old SHA; historical source recovered and verified before new protocol.',no_activity_or_solver_replay=True))
    with (F/'audit.json').open('x',encoding='utf8') as out:json.dump(dict(passed=True,math=read(F/'math.json'),supported_queries=5,required=8,no_new_scores=True,common_support_verified=True,independent_coordinate_max_error=maxerr),out,ensure_ascii=False,indent=2);out.write('\n')
    path=ROOT/'experiments/project_history.json';cat=read(path);cards={x['id']:x for x in cat['experiments']};now=s.stamp();head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    def update(card,**fields):
        for field,value in fields.items():
            if field in ['summary','obtained','interpretation','decision','cannot_claim'] and card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date=now,commit=head,field=field,old=card[field],reason='Уточнена читаемость новой записи и границы вывода; первичный результат и критерии не изменены.'))
            card[field]=value
    d=cards[DATA.name];e=cards[BASE.name];f=cards[F.name];g=cards[OUT.name]
    update(d,obtained=f'На 84 исторических обучающих видео рассмотрены 3156 сохранённых запросов. После прямой проверки прошлой и обеих текущих сырых рамок по разметке осталось 18 запросов на 15 видео. По заранее записанному правилу выбраны первые 12 запросов разных видео. Все входы глаза уже сохранены; 36 сопоставлений с разметкой независимо проверены. Источники: `runs/{DATA.name}/results.json`, `audit.json`.',interpretation='Пригодные сравнения нашлись до общего назначения рамок. Свободный остаток после назначения и исходный пул кандидатов — разные наблюдаемые части системы.',decision='Использован только зафиксированный набор из 12 запросов. Номера объектов не менялись.',cannot_claim='Это исторические обучающие видео, использованные прежними обучаемыми методами; нового официального разбиения данных нет. Личность подтверждена в двух конкретных моментах, а не на всей промежуточной истории.')
    update(e,obtained=f'На 12 запросах разных видео C34 правильно выбирает свою рамку до и после поиска в 7/12 случаях. T4/T5: 4/12 → 8/12. Обычные временные признаки из 34 каналов: 5/12 → 5/12; пространственные: 6/12 → 6/12. Исходная локальная стоимость: 8/12. C34 исправил 3 выбора стоимости и испортил 4; разность −1/12, парный 95% интервал [−0,5; 0,333333]. Против временного контроля +2/12, нижняя граница интервала равна 0. Восстановлены только 24 отсутствовавших снимка из готового входа глаза. Независимая ошибка проверки интерполяции и сходства ≤2,221·10⁻¹⁶. Источники: `runs/{BASE.name}/summary.json`, `audit.json`.',interpretation='Заранее выбранный основной C34 не превзошёл геометрическую стоимость. Благоприятный результат T4/T5 нельзя назначить основным после просмотра ответов. По сравнению с простыми признаками уверенная добавка FlyVis не установлена.',decision='Не внедрять изменение назначения и не подбирать пороги на этих запросах. Допустима ограниченная диагностика причины ошибок, отмеченная как исследование уже просмотренных случаев.',cannot_claim='Все 12 старых рядов остались без назначения в исходном первом решателе. Локальный результат 8/12 не означает восемь сохранённых ID. Не измерены дальнейшая судьба номера, HOTA или число смен ID; ядро и детектор не изменены.')
    update(f,state='infrastructure',done='2026-11-09',obtained=f'Математические проверки пройдены. На неизменном общем наборе ячеек оба кандидата поддержаны лишь в 5 из 12 запросов, при заранее установленном минимуме 8. Новых сходств не считали, готовые оценки и карты не повторяли. Независимое восстановление координат отличается не более чем на {maxerr:.3g}. Источники: `runs/{F.name}/readiness.json`, `audit.json`.',interpretation='Смена выравнивания часто меняет сам допустимый участок шаблона. На этих данных нельзя честно приписать разницу оценок одному преобразованию координат при достаточном числе сопоставимых запросов.',decision='Остановлено на геометрическом допуске. Не уменьшать число ячеек или набор сдвигов ради прохождения проверки.',cannot_claim='Ни полезность, ни бесполезность относительного выравнивания здесь не измерены. Относительные координаты уже были в прежнем методе; новой является только непроверенная комбинация с нынешним считывателем. Эти 12 случаев уже просмотрены и не являются независимой проверкой.')
    update(g,state='infrastructure',done='2026-11-09',obtained=f'В 6/12 запросах оба ребра дороже прежнего порога 0,8. В остальных 6/12 допустимы 6 рёбер: 4 к своей и 2 к чужой рамке. У каждого из шести есть более дешёвое ребро другого ряда к той же детекции. Своя рамка публично принадлежит прежнему другому ID в 6 запросах, впервые публичному в этом кадре — в 1, не имеет публичного владельца — в 5. Для чужой: 10/1/1. Проверены 24 исходных контекста. Источники: `runs/{OUT.name}/results.json`, `independent_audit.json`.',interpretation='Отказ нельзя считать одной ошибкой считывателя: отдельно видны отсечение стоимостью и конкуренция исходной матрицы. Нынешний ID рамки не доказывает, что именно он выиграл первое назначение.',decision='Сохранить эти классы отказов отдельно. Любое последующее исправление проверять через общий решатель и судьбу ID, сохраняя контроль без FlyVis.',cannot_claim='Не восстановлена полная матрица или индекс ряда с самым дешёвым конкурирующим ребром. Первое публичное появление не равно внутреннему рождению трека. Этот аудит не исправил ни одного ID и не измерил качество нового трекера.')
    for c in [f,g]:
        for i,item in enumerate(c['acceptance']):item.update(met=(i==0),evidence=f'runs/{F.name}/audit.json' if c is f else f'runs/{OUT.name}/independent_audit.json')
        c['evidence'] += [dict(kind='file',ref=f'runs/{c["id"]}/{name}',claim='Проверенный итог и границы интерпретации') for name in (['readiness.json','audit.json','ИТОГ.md'] if c is f else ['results.json','independent_audit.json','ИТОГ.md'])]
    text(path,json.dumps(cat,ensure_ascii=False,indent=2))
    topics=read(ROOT/'experiments/obsidian_experiment_topics.json');topics['topics'][F.name]=['Сохранение личности','Выбор продолжения среди рамок','Соответствие участков карты'];topics['topics'][OUT.name]=['Сохранение личности','Выбор продолжения среди рамок','Причины отказа общего назначения'];text(ROOT/'experiments/obsidian_experiment_topics.json',json.dumps(topics,ensure_ascii=False,indent=2))
    for folder,card in [(DATA,d),(BASE,e),(F,f),(OUT,g)]:
        text(folder/'ИТОГ.md',f'# {card["title"]}\n\n1. Получено\n\n{card["obtained"]}\n\n2. Что это означает\n\n{card["interpretation"]}\n\n3. Решение\n\n{card["decision"]}\n\n4. Граница вывода\n\n{card["cannot_claim"]}')
    systempath=ROOT/'experiments/system_problems.json';system=read(systempath);problem=next(x for x in system['problems'] if x['id']=='С2');old=problem['fact'];problem.setdefault('revisions',[]).append(dict(date=now,field='fact',old=old,reason='Добавлено прямое сравнение внутри запроса и разделение причин отказа.'));problem['fact']=old+' На 12 прямо подтверждённых запросах из 12 других исторических видео C34 выбрал свою рамку в 7 случаях, исходная локальная стоимость — в 8; критерий преимущества не выполнен. Из 12 исходных отказов 6 имеют только рёбра выше порога, остальные 6 имеют допустимое ребро с более дешёвой конкуренцией. Смена выравнивания не прошла допуск сопоставимой поддержки: 5/12 при минимуме 8. Источники: runs/'+BASE.name+'/summary.json; runs/'+OUT.name+'/independent_audit.json; runs/'+F.name+'/audit.json. Сохранение личности остаётся открытым.'
    for folder,name in [(BASE,'summary.json'),(OUT,'independent_audit.json'),(F,'audit.json')]:problem['evidence'].append(dict(source=f'runs/{folder.name}/{name}',sha256=sha(folder/name),claim='Прямой выбор, причины отказа и граница геометрического сравнения'))
    text(systempath,json.dumps(system,ensure_ascii=False,indent=2))
    litpath=ROOT/'experiments/useful_literature.json';lit=read(litpath);lewis=next(x for x in lit['sources'] if x['id']=='lewis_ncc_channel_normalization');lewis['application']+='\n\n09.11.2026. Ограничение NCC при изменении масштаба и перспективы использовано для постановки проверки относительного преобразования рамок. Обратимость преобразования математически проверена; также показано, что корреляция двух линейных профилей может быть 1 без совпадения личности. Геометрический допуск общей поддержки не пройден (5/12 при минимуме 8), поэтому новых сходств и заявления о пользе нет. Источник runs/'+F.name+'/audit.json.';lewis['related'].append('exp:'+F.name);lewis['evidence'] += [f'runs/{F.name}/math.json',f'runs/{F.name}/audit.json'];text(litpath,json.dumps(lit,ensure_ascii=False,indent=2))
    for card in [f,g]:s.journal(card['title'],f'*Получено (`{card["id"]}`).* {card["obtained"]}\n\n*Вывод.* {card["interpretation"]}\n\n*Решение.* {card["decision"]} {card["cannot_claim"]}')
    s.journal('Уточнены записи прямого сравнения: выигрыш считывателя и сохранение ID различаются',f'*Поправка оформления.* Прежние слитно написанные формулировки новых карточек сохранены в revisions, новые тексты разделены словами. Научные числа и критерии не менялись. {e["obtained"]}\n\n*Граница.* {e["cannot_claim"]} Первая регистрация аудита отказов остановилась до протокола из-за несовпадения нынешнего freeze_bench.py с историческим SHA. Из Git восстановлен прежний исходник с точным историческим SHA; порог подтверждён по нему, а не по нынешней копии.')
    print('All four outcomes recorded with readable reports',flush=True)
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
        copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
    assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};s.verify(read(OUT/'protocol.json'));s.verify(read(F/'protocol.json'));s.verify(read(BASE/'protocol.json'));s.verify(read(BASE/'maps_complete.json'))
    write('final_checks.json',dict(passed=True,catalog_tests=58,hypotheses_unchanged=True,graph=result,source_hashes_stable=True,no_repeated_model_or_tracker_runs=True));print('Catalogue, tests and graph passed',flush=True)
if __name__=='__main__':{'record':record,'checks':checks}[sys.argv[1]]()
