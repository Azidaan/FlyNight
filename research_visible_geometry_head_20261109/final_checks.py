"""Reuse the frozen QA procedure with new receipt names after result recording."""
import hashlib
from pathlib import Path
source=Path(__file__).with_name('preflight.py')
text=source.read_text(encoding='utf8')
for old,new in [('catalog_validation.log','final_catalog_validation.log'),('catalog_tests.log','final_catalog_tests.log'),('catalog_tests_receipt.json','final_catalog_tests_receipt.json'),('preflight.json','final_checks.json')]:
    assert old in text
    text=text.replace(old,new)
text=text.replace("'goal_achieved':False","'goal_achieved':False")
print('FINAL_QA_SOURCE',hashlib.sha256(source.read_bytes()).hexdigest(),flush=True)
exec(compile(text,str(source),'exec'))
