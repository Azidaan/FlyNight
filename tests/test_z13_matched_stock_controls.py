from pathlib import Path
from scripts import z13_matched_stock_controls_20261024 as S


def test_extension_does_not_mutate_running_benchmark():
    assert S.STAGE!=S.P.STAGE
    assert not set(S.SYSTEMS)&set(S.P.SYSTEMS)
    assert len(S.SYSTEMS)==4 and len(S.VIDEOS)==7
    for arm,path in S.MODELS.items():
        p=(S.ROOT/path).resolve();p.relative_to(S.ROOT.resolve());assert p.is_file()
        if arm.startswith('center'):
            cfg=S.read(p);assert cfg['arm']==arm.removeprefix('center')
            assert cfg['tau']>0 and cfg['lambda_value']>0


def test_signature_builder_uses_dynamic_localizer():
    import inspect
    from scripts import build_fly_signatures as B
    source=inspect.getsource(B.build_video)
    assert 'from tracking.z13_readout import Localizer' in source
    assert '--skip-fly' in Path(S.__file__).read_text(encoding='utf-8')
    assert S.MODELS['legacyC'].endswith('/C/model.pt')
    assert S.MODELS['legacyB'].endswith('/B/model.pt')
