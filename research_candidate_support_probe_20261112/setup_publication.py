from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
text=(ROOT/'research_terminal_response_probe_20261112/publish.py').read_text(encoding='utf8')
text=text.replace("['common.py','setup_probe.py','setup_publication.py','register.py','worker.py','audit.py','finish.py','publish.py']","['common.py','setup.py','setup_publication.py','register.py','worker.py','audit.py','finish.py','publish.py']")
text=text.replace("'coordinate_audit.json','state_audit.json','loaded_model.json','baseline_gate.json','replay.json'","'execution_start.json','baseline_gate.json','effects.json'")
(OUT/'publish.py').write_text(text,encoding='utf8')
print('PUBLICATION_HELPER_READY')
