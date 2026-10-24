"""Preserved queue3 logic redirected to independently validated recovery3."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    old=ROOT/'scripts/z13_matched_stock_queue3_20261024.py'
    source=old.read_text(encoding='utf-8')
    changes={
        'z13_matched_pipeline_recovery2_20261024':'z13_matched_pipeline_recovery3_20261024',
        'z13_record_matched_trackers_20261024':'z13_record_matched_recovery3_20261024',
        'z13_matched_stock_execution3_20261024':'z13_matched_stock_execution4_20261024',
        'queue3':'queue4','execution_v3':'execution_v4',
        "old=read(STAGE/'queue2.json')":"old=read(STAGE/'queue3.json')",
        "(STAGE/'queue2_failure.json')":"(STAGE/'queue3_failure.json')",
        'record_matched_recovery2':'record_matched_recovery3',
        'matched_recovery2':'matched_recovery3',
    }
    for a,b in changes.items():
        assert a in source,a
        source=source.replace(a,b)
    exec(compile(source,str(Path(__file__)),'exec'),globals())
