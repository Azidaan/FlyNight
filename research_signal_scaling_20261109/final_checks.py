"""Required QA after recording the scientific outcome; distinct receipts."""
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'research_visible_geometry_head_20261109/preflight.py'
text=source.read_text(encoding='utf8')
for old,new in [('catalog_validation.log','final_catalog_validation.log'),('catalog_tests.log','final_catalog_tests.log'),('catalog_tests_receipt.json','final_catalog_tests_receipt.json'),('preflight.json','final_checks.json')]:
    assert old in text;text=text.replace(old,new)
exec(compile(text,str(source),'exec'))
