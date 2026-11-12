from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
text=(ROOT/'research_fixed_crop_probe_20261112/publish.py').read_text(encoding='utf8')
text=text.replace("['common.py','register.py','render.py','worker.py','audit.py','finish.py','publish.py']","['common.py','setup_probe.py','setup_publication.py','register.py','worker.py','audit.py','finish.py','publish.py']")
text=text.replace("'render_supervisor.json','render_manifest.json','retinal_baseline_gate.json','render_process.log',","'coordinate_audit.json','state_audit.json',")
(OUT/'publish.py').write_text(text,encoding='utf8')
print('PUBLICATION_HELPER_READY')
