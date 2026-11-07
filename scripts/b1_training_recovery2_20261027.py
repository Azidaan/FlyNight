"""Продолжение с того же checkpoint: иной эквивалентный способ записи входа."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    service=json.loads((ROOT/'runs/b123_20261027/b1_training/service_recovery2.json').read_text(encoding='utf-8'))
    for name,digest in service['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    from tracking.z13_readout import FlyBackbone
    from tracking.b1_index_copy_stimulus import index_copy_step
    from scripts.b1_training_recovery1_20261027 import main as resume
    FlyBackbone.step=index_copy_step
    resume()


if __name__=='__main__':main()
