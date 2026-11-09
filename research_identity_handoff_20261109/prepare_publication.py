"""Publish the report and handoff while excluding foreign drafts and old archives."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;NATIVE=ROOT/'runs/domain_matched_native_20261109';sys.path.insert(0,str(NATIVE))
import evaluation_common
evaluation_common.OUT=OUT
source=NATIVE/'prepare_publication.py';text=source.read_text(encoding='utf8')
def change(a,b):
    global text
    assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
change("TEMP=ROOT/'tmp/domain_native_publication'","TEMP=ROOT/'tmp/identity_handoff_publication'")
change("checks['catalog_tests']==58","checks['tests']==58")
change("=='42ba6c1b337f21bb3c2dcc479130426d094ab971'","=='6e46b01c281b99cb0a7c05352952951dcfb0307d'")
change('assert len(headers)>=4','assert len(headers)>=1')
needle="assert not any('/Гипотезы/' in p or p=='AGENTS.md' or p.startswith('scripts/') or p.endswith('.obsidian/graph.json') for p in selected)"
extra="""
for name in ['dissertation/reports/Сохранение личности объектов — результаты проверок и продолжение.pdf','research_identity_handoff_20261109/ЦЕЛЬ_НОВОГО_ЧАТА.txt','research_identity_handoff_20261109/ПРОМПТ_НОВОГО_ЧАТА.txt','tracking/channel_agreement_correspondence.py']:
    selected[name]=str(ROOT/name)
for path in (ROOT/'research_agreement_readout_20261109').iterdir():
    if path.is_file() and path.suffix in ('.py','.json','.md'):selected[path.relative_to(ROOT).as_posix()]=str(path)
"""
change(needle,extra+'\n'+needle)
change("scope='Completed domain-matched native development comparison with matched geometry and image controls. Publish compact primary records/code; arrays,fulloutputs,compressedrawlogs and metricdetails remain local. Foreign calendar/plugin and two foreign journal entries excluded. Goal stays open until independent actual-ID success.'",
       "scope='Requested retrospective PDF report, verified source links and new-chat goal/prompt. Scientific identity problem remains unresolved. New agreement-reader candidate is explicitly deferred and has no real-data protocol/run. No unverified DOCX or draft PDFs published. Foreign drafts, hypothesis tree, arrays and browser content excluded.'")
exec(compile(text,str(source),'exec'))
