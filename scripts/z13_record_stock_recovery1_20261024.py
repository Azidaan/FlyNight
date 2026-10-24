"""Accept full resumed stock controls only after validating the old failure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_record_matched_stock_20261024 as F
from scripts.z13_record_matched_recovery3_20261024 import original_ended
OLD=F.STAGE;REC=F.RUN/'stock_recovery1'


def main():
    p=F.read(REC/'protocol.json')['service_recovery'];e=F.read(OLD/'controller_result.json')
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert F.sha(OLD/'controller_result.json')==p['previous_result_sha256'] and F.sha(OLD/'protocol.json')==p['science_protocol_sha256']
    original_ended(REC);F.STAGE=REC;F.main()


if __name__=='__main__':main()
