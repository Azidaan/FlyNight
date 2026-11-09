"""Reuse the full physical-index audit with this publication's explicit whitelist."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/domain_matched_native_20261109/stage_publication.py';text=source.read_text(encoding='utf8')
old="ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'tmp/domain_native_publication'"
assert text.count(old)==1;text=text.replace(old,"ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_reader_review_publication'")
assert text.count('len(staged)>100')==1;text=text.replace('len(staged)>100','len(staged)>0')
exec(compile(text,str(source),'exec'))
