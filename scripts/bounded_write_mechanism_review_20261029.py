"""Manipulation check of the preregistered age intervention; no tracker rerun."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import assignment_write_mechanism_review_20261029 as review

def main():
    source=inspect.getsource(review.main).replace('assignment_write_mechanism_review','bounded_write_mechanism_review')
    old="for policy,base,name in [('last',RUN/'masa_query_fp32_recovery1',key),('guarded',RUN/'assignment_write_val',encoder+'_guarded')]:"
    assert source.count(old)==1
    source=source.replace(old,"for policy,base,name in [('last',RUN/'masa_query_fp32_recovery1',key),('guarded',RUN/'assignment_write_val',encoder+'_guarded'),('bounded',RUN/'bounded_write_val',encoder+'_bounded'),('periodic',RUN/'bounded_write_val',encoder+'_periodic')]:")
    source=source.replace("for policy in ['last','guarded']:","for policy in ['last','guarded','bounded','periodic']:")
    ns=dict(vars(review));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
