"""Save numerical predicate proof; perturbation is diagnostic only, never deployed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def main():
    import numpy as np
    from tracking.z13_readout import HexGeometry
    from tracking.z13_candidates import candidates
    path=ROOT/'runs/z13_filter_descriptor_cache_20261023/parity_audit_cpu_20261024/observed_response_maps.npz'
    with np.load(path) as z:s=z['contiguous4_stored_2d']
    points=np.asarray([[13*v+195,int(13*(u+v/2))+195] for u in range(-15,16) for v in range(max(-15,-15-u),min(15,15-u)+1)],np.float32);g=HexGeometry(points,391)
    shifted=s.copy();shifted[418]-=2e-7;x=np.zeros((721,1))
    a=candidates(s,x,g)['indices'].tolist();b=candidates(shifted,x,g)['indices'].tolist();assert a==[450,418,195] and b==[450,195,245]
    out=ROOT/'runs/z13_filter_descriptor_cache_20261023/peak_audit_cpu_20261024/numeric_predicate_proof.json'
    assert not out.exists()
    write(out,dict(at=now().isoformat(),source_map_sha256=sha(path),source_sha256=sha(Path(__file__)),margin_418_minus_419=float(s[418]-s[419]),unmodified_selected=a,diagnostic_perturbation=dict(index=418,delta=-2e-7,selected=b),old_cached_selected=[450,195,245],used_in_scientific_run=False,exact_old_gpu_rounding_unknown=True,interpretation='A tiny response perturbation changes eligibility of418,therefore secondary candidate set. The fixed scientific candidate cache is never modified; this proof is not a corrected prediction or efficacy result.'))
    print(str(out))


if __name__=='__main__':main()
