from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    old=ROOT/'scripts/check_masa_detic_backbone_reference_recovery2_20261029.py'
    script='scripts/check_masa_detic_backbone_reference_float64_20261029.py'
    text=old.read_text(encoding='utf-8').replace('masa_detic_backbone_reference_recovery2','masa_detic_backbone_reference_float64')
    text=text.replace('    reference=swin_b(weights=None).features.eval()', '    model.backbone.double()\n    reference=swin_b(weights=None).features.eval().double()')
    text=text.replace('torch.linspace(-1,1,3*height*width)', 'torch.linspace(-1,1,3*height*width,dtype=torch.float64)')
    text=text.replace('full_embedding_parity=False,scientific_verdict=None', "comparison_precision='float64',float32_tolerance_passed=False,full_embedding_parity=False,scientific_verdict=None")
    compile(text,script,'exec');assert not (ROOT/script).exists();(ROOT/script).write_text(text,encoding='utf-8')
    folder=RUN/'masa_detic_backbone_reference_float64';assert not folder.exists();folder.mkdir()
    p=service.read(RUN/'masa_detic_backbone_reference_recovery2/protocol.json')
    p['sha256']={f:service.sha(ROOT/f) for f in [script,'tracking/masa_detic_prepared.py']}
    p['scope']='Last structural discriminator in float64; same1e-4 tolerance; FP32 mismatch remains unaccepted; not full embedding/HOTA'
    service.write(folder/'protocol.json',p)
    service.write(RUN/'ready_jobs/151_masa_detic_backbone_reference_float64.json',dict(name='151_masa_detic_backbone_reference_float64',folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))

if __name__=='__main__':main()
