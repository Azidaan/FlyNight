"""Create a separate matched worker from the verified previous service template."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'scripts/z13_filter_descriptor_20261023.py';assert not path.exists()
    source=(ROOT/'scripts/z13_search_conditioned_20261023.py').read_text(encoding='utf-8')
    start=source.index('def prepare():');end=source.index('\ndef worker():',start)
    prepare='''def prepare():
    assert not RUN.exists();from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN
    cache=read(CRUN/'protocol.json');prior.verify_sources(cache)
    end=read(CRUN/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended'] and end['worker_ended']
    assert read(CRUN/'report.json')['parity_pass']
    names=['scripts/z13_filter_descriptor_20261023.py','tracking/z13_filter_descriptor.py','tests/test_z13_filter_descriptor.py','scripts/z13_filter_descriptor_console.ps1','runs/z13_filter_descriptor_cache_20261023/report.json']
    names += [p.relative_to(ROOT).as_posix() for p in CRUN.glob('*/*/descriptors.json')]
    base=read(prior.RUN/'protocol.json');p={**cache,'created':now().isoformat(),'budget_seconds':1800,'stop_request_path':(RUN/'stop.request').relative_to(ROOT).as_posix(),'train':base['train'],'evaluate':base['evaluate'],'seeds':[0,1,2],'arms':list(ARMS),'sha256':{**cache['sha256'],**{n:sha(ROOT/n) for n in names}}}
    write(RUN/'protocol.json',p);journal('переход к обучению описания цели','Воспроизведение наблюдаемых входов и все коэффициенты сохранены: runs/z13_filter_descriptor_cache_20261023/report.json. Отдельный CPU-процесс,36моделей и1800с, основные критерии прежние. Протокол runs/z13_filter_descriptor_20261023/protocol.json.')

'''
    source=source[:start]+prepare+source[end:]
    source=source.replace('z13_search_conditioned','z13_filter_descriptor').replace('make_predictor','make_dual')
    source=source.replace('from tracking.z13_filter_descriptor import make_dual','from tracking.z13_filter_descriptor import make_ranker as make_dual')
    source=source.replace("access=='search'",'True')
    source=source.replace('make_dual(torch,True,True)','make_dual(torch,True)').replace('make_dual(torch,activity,True)','make_dual(torch,activity)')
    source=source.replace("('search','template')","('ridge','mean')").replace('_search_','_ridge_').replace('_template_','_mean_')
    source=source.replace("        ta,ea=data[arm];t=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]\n        for access in ('ridge','mean'):\n", """        ta0,ea0=data[arm]
        from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN
        # Episode numbers restart for each split. Keep the two training stores separate.
        d20=read(CRUN/arm/'fit20/descriptors.json');d84=read(CRUN/arm/'fit84/descriptors.json');de=read(CRUN/arm/'train26/descriptors.json')
        for access in ('ridge','mean'):
            ta=list(ta0);ea=list(ea0)
            if access=='ridge':
                ta[1]=np.asarray([d20[str(l['episode'])] for l in l20]+[d84[str(l['episode'])] for l in extra],dtype=np.float32)
                ea[1]=np.asarray([de[str(l['episode'])] for l in ev],dtype=np.float32)
            t=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]
""")
    assert 'ta0,ea0' in source
    source=source.replace("model=make_dual(torch,True);start", "model=make_dual(torch,True);start")
    source=source.replace('Обучение предсказателя весов цели','Обучение выборщика: фильтр против среднего')
    source=source.replace('No HOTA/calibrated refusal; exploratory reused26train.','No HOTA/calibrated refusal; coefficients only from observed frames; exploratory reused26train.')
    source=source.replace('5400s','1800s')
    source=source.replace("'search_geometry'","'ridge_geometry'")
    source=source.replace('Matched target-weight prediction with/without current-search access.','Matched rankers: observed ridge coefficients versus foreground mean.')
    path.write_text(source,encoding='utf-8')
    console=(ROOT/'scripts/z13_search_conditioned_console.ps1').read_text(encoding='utf-8-sig').replace('z13_search_conditioned_20261023','z13_filter_descriptor_cache_20261023').replace('FlyVis — модель цели и текущий поиск','FlyVis — коэффициенты модели цели').replace('FlyVis: предсказание весов модели цели; CPU, 36 моделей','FlyVis: наблюдаемые признаки и коэффициенты; затем 36 CPU моделей').replace('Готово моделей: {0}/36.','Готово моделей: {0}/36 (после извлечения).')
    (ROOT/'scripts/z13_filter_descriptor_console.ps1').write_text(console,encoding='utf-8-sig')


if __name__=='__main__':main()
