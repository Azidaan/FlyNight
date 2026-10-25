"""Recover own interrupted rendering; require byte-identical regenerated draw.io."""
from pathlib import Path
import sys,shutil,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OUT=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/deferred_20261025'


def main():
    from scripts import build_flyvis_architecture as gen
    target=gen.OUT;backup=OUT/'interrupted_architecture';render=OUT/'recovered_architecture'
    assert not backup.exists() and not render.exists()
    text=(target/'FlyVis_architecture.drawio').read_text(encoding='utf8')
    assert 'ОТЛОЖЕНО 25.10.2026' in text and 'z13_spatial_set_readout_v1' in text
    backup.mkdir();hashes={}
    for path in target.iterdir():
        if path.is_file():
            hashes[path.name]=sha(path);shutil.copy2(path,backup/path.name)
            assert sha(backup/path.name)==hashes[path.name]
    write(OUT/'architecture_interruption.json',dict(at=now().isoformat(),error='Own render wrote drawio/svg/png then OSError22 writing existing preview; rerun guard refused mismatched stale receipt.',preserved_partial_hashes=hashes,unchanged_generation_source_sha256=sha(ROOT/'scripts/build_flyvis_architecture.py')))
    gen.OUT=render;gen.main()
    # Only our exact generated partial diagram may be replaced. A manual edit
    # would not be byte-identical to this independently generated fresh output.
    assert sha(render/'FlyVis_architecture.drawio')==hashes['FlyVis_architecture.drawio']
    assert all(sha(target/name)==digest for name,digest in hashes.items())
    for path in render.iterdir():
        if path.is_file():
            temporary=target/(path.name+'.deferred.next');shutil.copy2(path,temporary);os.replace(temporary,target/path.name)
    receipt=read(target/'validation.json')
    assert receipt['diagram_sheets']==1 and all(sha(target/n)==h for n,h in receipt['outputs'].items())
    write(OUT/'architecture_recovery.json',dict(at=now().isoformat(),own_partial_drawio_matches_fresh_generation_exact=True,all_outputs_promoted_and_verified=True,source_sha256=sha(ROOT/'scripts/build_flyvis_architecture.py')))


if __name__=='__main__':main()
