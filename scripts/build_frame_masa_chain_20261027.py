"""Создать отдельную версию прежней цепочки; замороженные источники не меняются."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def save(name,text):
    p=ROOT/'scripts'/name;assert not p.exists();p.write_text(text,encoding='utf-8')

def main():
    old=(ROOT/'scripts/run_lazy_appearance_association_20261027.py').read_text(encoding='utf-8-sig')
    replacements={
        'from tracking.busca_runtime import load_model':'from tracking.masa_r50 import build',
        'from tracking.lazy_appearance_association import AppearanceServer,LazyAppearanceXYWH':'from tracking.lazy_appearance_association import LazyAppearanceXYWH\n    from tracking.frame_masa_appearance import FrameMasaServer as AppearanceServer',
        "model,coverage=load_model('cuda');encoder=model.reid_encoder;encoder.use_domain_adaptation=False;encoder.model.eval()":"model=build('cuda');encoder=model;coverage=dict(checkpoint_sha256=sha(ROOT/'weights/masa/masa_r50.pth'),regime='existing_frame_masa_fp32_eval')",
        'encoder.model.named_buffers()':'encoder.named_buffers()',
        "lazy_appearance_frozen.json":"frame_masa_frozen.json",
        "            cap.release()":"                server.finish_frame()\n            cap.release()",
        "            pooled={}\n":"            previous_reid=read(BASE/'lazy_appearance_validation/report.json')['per_video']['appearance']\n            for group,vs in groups.items():\n                comparisons[group]['appearance_vs_previous_reid']=paired({v:per['appearance'][v]['HOTA'] for v in vs},{v:previous_reid[v]['HOTA'] for v in vs})\n            pooled={}\n",
        "batchnorm_buffers_unchanged=True,ground_truth_used_by_predictor=False":"batchnorm_buffers_unchanged=True,previous_reid=previous_reid,ground_truth_used_by_predictor=False",
    }
    for a,b in replacements.items():
        assert a in old,a;old=old.replace(a,b)
    save('run_frame_masa_association_20261027.py',old)
    audit=(ROOT/'scripts/audit_lazy_appearance_association_recovery1_20261027.py').read_text(encoding='utf-8-sig').replace('lazy_appearance_frozen.json','frame_masa_frozen.json')
    exact="d=np.array([r['per_video']['appearance'][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in vs])"
    assert exact in audit
    audit=audit.replace(exact,"reference=r['previous_reid'] if b=='previous_reid' else r['per_video'][b]\n                d=np.array([r['per_video']['appearance'][v]['HOTA']-reference[v]['HOTA'] for v in vs])")
    save('audit_frame_masa_association_20261027.py',audit)
    for phase in ['dev','validation']:
        save('frame_masa_'+phase+'_20261027.py',"from scripts.run_frame_masa_association_20261027 import run\nif __name__=='__main__':run('frame_masa_"+phase+"')\n")
        after="\n    from scripts.prepare_frame_masa_chain_20261027 import prepare_validation\n    prepare_validation()" if phase=='dev' else ''
        save('frame_masa_'+phase+'_audit_20261027.py',"from scripts.audit_frame_masa_association_20261027 import audit\nif __name__=='__main__':\n    audit('frame_masa_"+phase+"')"+after+'\n')
    print('Created isolated MASA runner and assignment audits; previous sources untouched')

if __name__=='__main__':main()
