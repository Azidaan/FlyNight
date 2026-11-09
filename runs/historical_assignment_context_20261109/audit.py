"""Metadata-only explanation of native refusals; no solver replay or inference."""
import sys,json,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/historical_two_candidate_comparison_20261109';DATA=ROOT/'runs/historical_direct_query_inventory_20261109'
sys.path.insert(0,str(BASE));import study_v2 as s
import numpy as np
read=s.read;sha=s.sha
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(s.cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def register():
    p=read(BASE/'protocol.json');s.verify(p);source=read(DATA/'protocol.json');s.verify(source);historic=read(ROOT/'runs/z13_spatial_bins_gate_20261025/recovery1/protocol.json');arch=OUT/'historical_freeze_bench.py';commit='bd01d1026ae9456a6762ca62cd086cf4f45fe2fe'
    # The first registration attempt correctly stopped because today's file
    # differs. Recover the historical Git blob with its original CRLF bytes;
    # the saved historical SHA, not this reconstruction guess, is the test.
    old=subprocess.check_output(['git','show',commit+':scripts/freeze_bench.py'],cwd=ROOT).replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    if not arch.exists():arch.write_bytes(old)
    assert sha(arch)==historic['sha256']['scripts/freeze_bench.py'];config=next(ast.literal_eval(n.value) for n in ast.parse(old.decode('utf8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='XYWH_CONFIG' for t in n.targets));assert config['match_thresh']==.8
    files=[Path(__file__),arch,BASE/'protocol.json',BASE/'results.json',DATA/'protocol.json',ROOT/'tracking/z13_region_assignment_context.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
    # The old saved protocol must attest these actual sources, not merely a
    # currently plausible source file bearing the same name.
    for name in ['.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']:
        assert sha(ROOT/name)==historic['sha256'][name],name
    for c in p['cases']:
        for k in ['native','context']:files.append(ROOT/source['videos'][c['video']][k])
    write('protocol.json',dict(registered=s.stamp(),sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},cases=[c['uid'] for c in p['cases']],threshold=.8,scope='Twelve already evaluated queries only, metadata/context, no score/model/solver recalculation. Examine all24 actual edges, retain missing ownership, no policy tuning.',definitions='Context vector: original edge, lowest other-track edge to same detection, lowest other-detection edge, cost-minus-competitor, old-row assigned flag, action equals original. Local admitted means cost<=historical match_thresh .8, not actual assignment. Public owner is recovered by exact raw confidence/class/frame from prior audited inventory. First PUBLIC occurrence is not internal birth/activation and does not identify assignment stage.',limitations='A cheaper competing edge proves competition in original matrix, but neither full solver alternative nor identity of that competing row is saved here. Current owner can arise later in the pipeline. Do not equate owner with winner of first association. No claim of ID fate.',acceptance=[dict(criterion='All24 contexts verified against original costs, all current public owner histories and local threshold admitted counts retained; unknown stage/rival identity not invented.',met=None)]))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Почему прежний ряд остался без рамки — порог, конкуренция и публичные владельцы',area='flyvis',state='running',started='2026-11-09',question='Что непосредственно видно в исходной матрице 12 отказов, и существовали ли нынешние владельцы раньше?',hypothesis='Часть отказов объясняется локальным порогом, часть — более дешёвыми конкурирующими рёбрами. По текущему публичному ID нельзя автоматически определить этап назначения.',expected='Аудит 24 готовых рёбер и публичных историй без нового трекера, активности или оценок сходства.',acceptance=read(OUT/'protocol.json')['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Правила до полного разбора контекста')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');s.journal('Зафиксирован разбор исходных отказов без перезапуска трекера',f'*Постановка (`{OUT.name}`).* Для тех же 12 запросов проверяются 24 сохранённых ребра: прежний порог 0,8, минимальная стоимость другого ряда, исходный признак неназначения и первая публичная дата нынешнего владельца. Источники алгоритма сверены с историческим протоколом. Первое публичное появление не называется рождением трека; нынешний владелец не называется победителем первого назначения без доказательства. Источник `runs/{OUT.name}/protocol.json`.')
def run():
    p=read(OUT/'protocol.json');s.verify(p);bp=read(BASE/'protocol.json');source=read(DATA/'protocol.json');rows=[];counts={};totals={'queries':12,'all_edges_rejected':0,'some_locally_admitted':0,'own_admitted':0,'foreign_admitted':0,'admitted_with_cheaper_competitor':0,'admitted_without_cheaper_competitor':0};bykind={k:{'public_before':0,'first_public_at_query':0,'not_public':0} for k in ['own','foreign']}
    for c in bp['cases']:
        info=source['videos'][c['video']];entry=read(ROOT/info['context'])['entries'][str(c['uid'])]
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        assert entry['context'][0][4]==0;cs=[]
        for k in c['candidates']:
            pos=entry['choices'].index(k['column']);ctx=np.array(entry['context'][pos]);assert ctx[0]==k['original_cost'] and ctx[4]==0 and len(ctx)==6
            admitted=bool(ctx[0]<=p['threshold']);cheaper=bool(ctx[1]<ctx[0]);owners=[]
            for tid in k['current_native_owners']:
                history=native[native[:,5]==tid];assert len(history)>0;first=int(history[:,0].min());previous=history[history[:,0]<c['frame']];owners.append(dict(tid=tid,first_public=first,last_public_before=int(previous[:,0].max()) if len(previous) else None,category='public_before' if first<c['frame'] else 'first_public_at_query'))
            category='not_public' if not owners else ('public_before' if any(o['category']=='public_before' for o in owners) else 'first_public_at_query');bykind[k['kind']][category]+=1
            if admitted:totals[k['kind']+'_admitted']+=1;totals['admitted_with_cheaper_competitor' if cheaper else 'admitted_without_cheaper_competitor']+=1
            cs.append(dict(kind=k['kind'],raw_id=k['raw_id'],cost=float(ctx[0]),lowest_competing_cost=float(ctx[1]),alternative_cost=float(ctx[2]),local_admitted=admitted,cheaper_competing_edge=cheaper,public_owners=owners,owner_category=category))
        n=sum(k['local_admitted'] for k in cs);totals['all_edges_rejected' if n==0 else 'some_locally_admitted']+=1;counts[n]=counts.get(n,0)+1;rows.append(dict(uid=c['uid'],video=c['video'],frame=c['frame'],old_tid=c['tid'],original_row_assigned=False,candidates=cs))
    s.verify(p);write('results.json',dict(passed=True,totals=totals,admitted_edges_per_query=counts,public_ownership=bykind,records=rows,no_solver_replay=True,no_map_or_score_calculation=True));print(totals,bykind,flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
