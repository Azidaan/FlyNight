"""Verify preserved recovery chain before accepting fresh-interpreter result."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_record_matched_trackers_20261024 as F
REC=F.RUN/'gpu_recovery3'
original_ended=F.ended


def ended(folder):
    if folder in (F.ORIGINAL_STAGE,F.RECOVERY,F.RECOVERY2,REC):
        for before,after,key,rkey,pkey in ((F.RECOVERY,F.RECOVERY2,'service_recovery2','previous_result_sha256','previous_protocol_sha256'),(F.RECOVERY2,REC,'service_recovery3','previous_result_sha256','previous_protocol_sha256')):
            e=F.read(before/'controller_result.json');p=F.read(after/'protocol.json')[key]
            assert e['exit_code']!=0 and e['all_owned_ended'] and not e['remaining']
            assert F.sha(before/'controller_result.json')==p[rkey] and F.sha(before/'protocol.json')==p[pkey]
        folder=REC
    return original_ended(folder)


def install():F.STAGE=REC;F.ended=ended


if __name__=='__main__':install();F.main()
