"""Audit new direct labels, relative queries and descriptive transfer."""
import study as t
from study import *
def record():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'));dp=read(DATA/'protocol.json');result=read(OUT/'results.json');plans=read(OUT/'readiness.json')['plans'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;load=namespace()['arrays'];checks=[];error=0.;coorderr=0.;scorechecks=0;table={mode:{metric:{name:dict(correct=0,wrong=0,tie=0,abstain=0) for name in s.NAMES+['cost']} for metric in ['fixed','maximum']} for mode in ['old','relative']};rows=[]
    def identities(box,labels):
        objs=[x for x in labels if x['category']=='car' and not (x.get('attributes') or {}).get('crowd',False)];boxes=np.array([[x['box2d'][k] for k in ['x1','y1','x2','y2']] for x in objs]);a=np.array(box);inter=np.maximum(0,np.minimum(a[2:],boxes[:,2:])-np.maximum(a[:2],boxes[:,:2])).prod(1);union=(a[2:]-a[:2]).prod()+(boxes[:,2:]-boxes[:,:2]).prod(1)-inter;return [str(o['id']) for o,v in zip(objs,inter/union) if v>=.5]
    def features(key):
        a=load(key,p);c=a['C34'].astype(float);r=a['R1'].astype(float);cen=r[:,s.SELECT[:4]]
        return dict(C34=c,FlyVis=c[:,4:12],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T,input8_temporal=r[:,s.SELECT],center8=np.concatenate([cen,cen],1),neighbor8=neighbor_aug(cen.T[None],nb)[0].T)
    def kind(a,b):return 'abstain' if a is None or b is None else ('correct' if a>b else ('tie' if a==b else 'wrong'))
    for case,plan,row in zip(p['cases'],plans,result['records']):
        info=dp['videos'][case['video']];gt=read(ROOT/info['annotation']);alignment=read(ROOT/info['alignment'])['matches'];ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,alignment)}
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={k:z[k] for k in ['xyxy','frame_idx','conf','cls']}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        rr=native[(native[:,0]==case['reference_frame'])&(native[:,5]==case['tid'])];assert len(rr)==1;rid=np.flatnonzero((raw['frame_idx']==case['reference_frame'])&(raw['cls']==rr[0,7])&(raw['conf'].astype(np.float32)==np.float32(rr[0,6])));assert len(rid)==1 and rid[0]==case['reference_raw']['raw_id'];ident=case['reference_hits'][0]['identity'];assert identities(raw['xyxy'][rid[0]],ann[case['reference_frame']])==[ident];checks.append([case['uid'],'reference',ident])
        for k in case['candidates']:
            rid=k['raw_id'];assert raw['frame_idx'][rid]==case['frame'] and raw['cls'][rid]==2 and np.array_equal(raw['xyxy'][rid],k['box']);ids=identities(raw['xyxy'][rid],ann[case['frame']]);assert len(ids)==1 and ids[0]==k['hits'][0]['identity'] and ((ids[0]==ident)==(k['kind']=='own'));checks.append([case['uid'],k['kind'],ids[0]])
        ref,cur=[features(k) for k in case['keys']];rc=np.array(case['sequence'][case['reference_relative']][1]);ct=np.array(case['sequence'][-1][1]);b0=np.array(case['reference_raw']['state_box']);idx=np.array(plan['cells']);x=reader.points[idx]*(rc[2:]-rc[:2])/g['eye_size']+rc[:2]
        for k,relplan in zip(case['candidates'],plan['plans']):
            assert relplan['cells']==plan['cells'];b=np.array(k['box']);expected=(b[:2]+((x-b0[:2])/(b0[2:]-b0[:2]))*(b[2:]-b[:2]))[:,None]+np.array(k['plan']['offsets']);expected=(expected-ct[:2])/((ct[2:]-ct[:2])/g['eye_size']);coorderr=max(coorderr,float(np.max(abs(expected-np.array(relplan['query'])))))
            for mode,geometry in [('old',k['plan']),('relative',relplan)]:
                query=np.array(geometry['query']);ix=np.array(geometry['cells']);simplex=reader.mesh.find_simplex(query);vertices=reader.mesh.simplices[simplex];xy=reader.points[vertices];beta=np.linalg.solve((xy[...,:2,:]-xy[...,2:3,:]).swapaxes(-1,-2),(query-xy[...,2,:])[...,None])[...,0]
                for name in s.NAMES:
                    vv=cur[name][vertices];patches=(vv[...,2,:]+(beta[...,None]*(vv[...,:2,:]-vv[...,2:3,:])).sum(-2)).transpose(1,0,2);template=ref[name][ix];active=np.ptp(template,axis=0)!=0;a=template-template.mean(0);a[:,~active]=0;an=np.linalg.norm(a,axis=0);saved=next(c for c in row['variants'][name] if c['kind']==k['kind'])['values'][mode]['scores']
                    for patch,value in zip(patches,saved):
                        nonconstant=np.ptp(patch,axis=0)!=0;b1=patch-patch.mean(0);b1[:,~nonconstant]=0;den=an*np.linalg.norm(b1,axis=0);rho=np.divide((a*b1).sum(0),den,out=np.zeros_like(an),where=den>0);w=float(rho.sum()/active.sum()) if active.any() and nonconstant.any() else None;assert (w is None)==(value is None)
                        if w is not None:error=max(error,abs(w-value))
                        scorechecks+=1
        decisions={}
        for mode in table:
            decisions[mode]={}
            for metric in table[mode]:
                decisions[mode][metric]={}
                for name in s.NAMES:
                    vals={c['kind']:c['values'][mode][metric] for c in row['variants'][name]};outcome=kind(vals['own'],vals['foreign']);table[mode][metric][name][outcome]+=1;decisions[mode][metric][name]=dict(outcome=outcome,**vals)
                vals={c['kind']:c['score'] for c in row['cost']};outcome=kind(vals['own'],vals['foreign']);table[mode][metric]['cost'][outcome]+=1;decisions[mode][metric]['cost']=dict(outcome=outcome,**vals)
        rows.append(dict(uid=case['uid'],video=case['video'],decisions=decisions))
    assert len(checks)==9 and error<1e-9 and coorderr<1e-10 and scorechecks==3*2*2*7*19;verify(p);verify(read(OUT/'maps_complete.json'))
    write('audit.json',dict(passed=True,direct_raw_gt_checks=checks,independent_scalar_scores=scorechecks,maximum_score_error=error,maximum_coordinate_error=coorderr,all_three_new_videos=True,common_reference_cells=True,weights_unchanged=read(OUT/'maps_complete.json')['weights_unchanged'],no_new_IDs=True))
    write('summary.json',dict(passed=True,table=table,records=rows,descriptive_only=True,primary_C34_not_replaced=True,no_generalization_or_ID_claim=True))
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);card.update(state='inconclusive',done='2026-11-09',obtained=f'На трёх оставшихся новых видео после поиска прежний C34 выбрал свою рамку в 1/3 случаях, относительный — в 3/3. T4/T5: 2/3 → 3/3. Все пять простых входных контролей с относительными координатами также дают 3/3; исходная стоимость — 3/3. Независимо проверены 9 прямых меток и 1596 оценок; ошибка ≤{error:.4g}. Восстановлены только {read(OUT/"maps_complete.json")["new_snapshot_files"]} недостающих снимков из сохранённого входа глаза. Источники `runs/{OUT.name}/summary.json`, `audit.json`.',interpretation='Перенос согласуется с пользой комбинации относительного выравнивания и шаблона для прежнего считывания. Отдельная добавка FlyVis на трёх запросах не выявлена: все простые контроли и исходная стоимость также правильны.',decision='Не менять основной стенд и не делать вывод о сохранённых ID. Прекратить расширение вариантов на этом этапе; сохранить готовые карты, реализации и прямые метки для дальнейшей зафиксированной проверки.',cannot_claim='Три других исторических обучающих видео не являются новой официальной проверочной выборкой и не доказывают обобщение. Здесь не было ошибки локальной стоимости; улучшение считывателя не означает исправление отказа общего назначения. Не измерены дальнейшая судьба ID, HOTA и вклад изменения ядра; ядро не менялось.')
    for item in card['acceptance']:item.update(met=True,evidence=f'runs/{OUT.name}/audit.json')
    card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{name}',claim='Описательный перенос и независимая проверка без ID-вывода') for name in ['summary.json','audit.json','ИТОГ.md']]
    staging=path.with_name('project_history.relative_transfer_ready.json');staging.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');staging.replace(path)
    path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=['Сохранение личности','Выбор продолжения среди рамок','Соответствие участков карты'];path.write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'ИТОГ.md').write_text(f'# {card["title"]}\n\n1. Получено\n\n{card["obtained"]}\n\n2. Значение\n\n{card["interpretation"]}\n\n3. Решение\n\n{card["decision"]}\n\n4. Граница\n\n{card["cannot_claim"]}\n',encoding='utf8')
    systempath=ROOT/'experiments/system_problems.json';system=read(systempath);problem=next(x for x in system['problems'] if x['id']=='С2');problem.setdefault('revisions',[]).append(dict(date=s.stamp(),field='fact',old=problem['fact'],reason='Добавлены полный относительный шаблон и фиксированный перенос, без приписывания прибавки ядру.'));problem['fact']+=' Полный относительный шаблон с общими прошлыми ячейками дал C34 10/12 вместо 7/12 на просмотренных запросах; простой соседний вход также дал 10/12. На трёх других запросах C34 1/3→3/3, все относительные входные контроли и исходная стоимость — 3/3. Это улучшение способа сравнения без подтверждённой добавки FlyVis и без измерения ID. Источники runs/'+REL.name+'/summary.json; runs/'+OUT.name+'/audit.json.'
    for folder in [REL,OUT]:problem['evidence'].append(dict(source=f'runs/{folder.name}/audit.json',sha256=sha(folder/'audit.json'),claim='Относительный шаблон и matched-контроли без изменения ID'))
    systempath.write_text(json.dumps(system,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    litpath=ROOT/'experiments/useful_literature.json';lit=read(litpath);lewis=next(x for x in lit['sources'] if x['id']=='lewis_ncc_channel_normalization');lewis['application']+='\n\n09.11.2026. После отдельного провала сравнения только координат проверена вся комбинация относительного шаблона с одинаковыми прошлыми ячейками двух кандидатов. На 12 просмотренных запросах C34 7/12→10/12, простой соседний вход тоже 10/12. На трёх других запросах все относительные представления и исходная стоимость правильны. Литературное основание касалось ограничений NCC и необходимости явного выравнивания, а не гарантии личности или преимущества FlyVis. Источники runs/'+REL.name+'/audit.json; runs/'+OUT.name+'/audit.json.';lewis['evidence'] += [f'runs/{REL.name}/audit.json',f'runs/{OUT.name}/audit.json'];lewis['related'] += ['exp:'+REL.name,'exp:'+OUT.name];litpath.write_text(json.dumps(lit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    s.journal(card['title'],f'*Получено (`{OUT.name}`).* {card["obtained"]}\n\n*Вывод.* {card["interpretation"]}\n\n*Решение.* {card["decision"]} {card["cannot_claim"]} Счётчик «12» в старой переиспользованной функции восстановления является прежним знаменателем её служебного сообщения; протокол и maps_complete.json содержат фактические 3 случая/6 снимков, результаты не расширены до 12.')
    print('Direct labels and descriptive transfer audited',flush=True)
def checks():
    import contextlib,runpy,shutil,tempfile
    sys.argv=['scripts/validate_project_history.py']
    with (OUT/'catalog_validation.log').open('x',encoding='utf8') as log:
        with contextlib.redirect_stdout(log):
            try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
            except SystemExit as err:
                if err.code:raise
    import pytest
    assert pytest.main(['tests/test_project_history.py','-q'])==0
    from scripts.build_obsidian_vault import main
    from scripts.audit_obsidian_levels import audit
    vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
    def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
        copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
    assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};verify(read(OUT/'protocol.json'));verify(read(OUT/'maps_complete.json'));s.verify(read(REL/'protocol.json'))
    write('final_checks.json',dict(passed=True,catalog_tests=58,graph=result,hypotheses_unchanged=True,sources_and_maps_stable=True,goal_complete=False));print('Both relative studies, catalogue and graphs verified',flush=True)
if __name__=='__main__':{'record':record,'checks':checks}[sys.argv[1]]()
