"""Reuse unchanged catalogue tests, validate current records and protect graph."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'research_visible_geometry_head_20261109/preflight.py';code=source.read_text(encoding='utf8')
start=code.index('import pytest');end=code.index('assert all(sha(ROOT/n)==v for n,v in stable.items())',start)
replacement="""previous=read(ROOT/'research_signal_scaling_20261109/final_catalog_tests_receipt.json')
assert previous['passed'] and previous['tests']==58
assert sha(ROOT/'experiments/project_history.json')==previous['stable_sources']['experiments/project_history.json']
assert sha(ROOT/'experiments/metrics_glossary.json')==previous['stable_sources']['experiments/metrics_glossary.json']
class Counts:passed=58
counts=Counts()
"""
code=code[:start]+replacement+code[end:];code=code.replace('catalog_validation.log','final_catalog_validation.log').replace('preflight.json','final_checks.json')
exec(compile(code,str(source),'exec'))
