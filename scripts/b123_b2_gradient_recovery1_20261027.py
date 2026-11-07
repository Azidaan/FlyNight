"""Служебное заполнение отсутствующих стандартных коэффициентов функции потерь."""
from pathlib import Path
import hashlib,json,sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/b2_gradient/recovery1'

def main():
    service=json.loads((RUN/'service.json').read_text(encoding='utf-8'))
    for n,h in service['sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    from scripts import b2_gradient_profile_20261027 as worker
    source=inspect.getsource(worker.run)
    old='detector.args=get_cfg(detector.args)';new='detector.args=get_cfg(overrides=detector.args)'
    assert source.count(old)==1
    worker.RUN=RUN
    from tracking.z13_readout import FlyBackbone
    from tracking.b1_fast_stimulus import fast_init
    from tracking.b1_index_copy_stimulus import index_copy_step
    FlyBackbone.init=fast_init;FlyBackbone.step=index_copy_step
    namespace=vars(worker).copy()
    exec(compile(source.replace(old,new),str(Path(__file__).resolve()),'exec'),namespace)
    namespace['run']()

if __name__=='__main__':main()
