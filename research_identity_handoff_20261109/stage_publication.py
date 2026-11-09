"""Stage only the explicit handoff manifest and independently audit graph links."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'runs/domain_matched_native_20261109/stage_publication.py';code=source.read_text(encoding='utf8')
old="ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'tmp/domain_native_publication'";assert code.count(old)==1;code=code.replace(old,"ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_handoff_publication'")
code=code.replace('len(staged)>100','len(staged)>0');exec(compile(code,str(source),'exec'))
