"""Reuse original scientific acceptance, preserving prior run receipts/revisions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import RUN,read,sha


def main():
    auth=read(RUN/'resume1_authorization.json');assert read(RUN/'resume1_preflight.json')['authorization_sha256']==sha(RUN/'resume1_authorization.json')
    source=(ROOT/'scripts/z13_record_center_hota_20261021.py').read_text(encoding='utf-8')
    old="marker=RUN/(x.phase+'_recorded.json')";assert old in source;source=source.replace(old,"marker=RUN/('resume1_'+x.phase+'_recorded.json')")
    for old,new in [('full_controller_result.json','resume1_controller_result.json'),('full_launch.json','resume1_launch.json'),('night_finished.json','resume1_night_finished.json')]:source=source.replace(old,new)
    source=source.replace("['git','rev-parse','HEAD']","['git','--no-optional-locks','rev-parse','HEAD']")
    scope=dict(__name__='resumed_center_hota_records',__file__=str(ROOT/'scripts/z13_record_center_hota_20261021.py'));exec(compile(source,'resumed_center_hota_records','exec'),scope);scope['main']()


if __name__=='__main__':main()
