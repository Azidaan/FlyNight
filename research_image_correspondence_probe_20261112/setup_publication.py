from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
text=(ROOT/'research_terminal_response_probe_20261112/publish.py').read_text(encoding='utf8')
text=text.replace("['common.py','setup_probe.py','setup_publication.py','register.py','worker.py','audit.py','finish.py','publish.py']","['common.py','setup.py','setup_publication.py','register.py','worker.py','audit.py','box_membership.py','detector_box_audit.py','clarify_support.py','plot_support.py','plot_support_v2.py','finish.py','publish.py']")
text=text.replace("'coordinate_audit.json','state_audit.json','loaded_model.json','baseline_gate.json','replay.json'","'correspondence.json','membership_protocol.json','box_membership.json','detector_box_audit.json','execution_start.json','baseline_gate.json','plot_manifest.json','plot_manifest_v2.json','support_574.png','support_574_v2.png'")
text=text.replace("input=path.read_bytes().replace(b'\\r\\n',b'\\n')", "input=(path.read_bytes() if path.suffix.lower()=='.png' else path.read_bytes().replace(b'\\r\\n',b'\\n'))")
(OUT/'publish.py').write_text(text,encoding='utf8')
print('PUBLICATION_HELPER_READY')
