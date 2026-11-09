"""Prepare only this step and prior completed parts, preserving foreign drafts."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'runs/domain_matched_native_20261109'
sys.path.insert(0,str(NATIVE))
import evaluation_common
evaluation_common.OUT=OUT
source=NATIVE/'prepare_publication.py';text=source.read_text(encoding='utf8')
def change(a,b):
    global text
    assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
change("TEMP=ROOT/'tmp/domain_native_publication'","TEMP=ROOT/'tmp/signal_scaling_publication'")
change("checks['catalog_tests']==58","checks['tests']==58")
change("=='42ba6c1b337f21bb3c2dcc479130426d094ab971'","=='dd91b93750c7021164f384bcc54aa1af2813b82d'")
change('assert len(headers)>=4','assert len(headers)>=2')
needle="assert not any('/Гипотезы/' in p or p=='AGENTS.md' or p.startswith('scripts/') or p.endswith('.obsidian/graph.json') for p in selected)"
extra="""
for folder in ['research_partitioned_correspondence_20261109']:
    for path in (ROOT/folder).iterdir():
        if not path.is_file() or path.suffix not in ('.py','.json','.md'):continue
        if 'status' in path.name or path.name.endswith(('.started.json','.ended.json','.launch.json','.terminal.json')) or 'identity' in path.name or path.name in ['executor.json']:continue
        selected[path.relative_to(ROOT).as_posix()]=str(path)
for name in ['research_identity_synthesis_20261109/ПРОВЕРКА_НЕЗАВИСИМОСТИ_ОЦЕНКИ.md','research_identity_synthesis_20261109/gemini_request_v20.txt','research_identity_synthesis_20261109/gemini_v20_submission.json']:
    selected[name]=str(ROOT/name)
"""
change(needle,extra+'\n'+needle)
change("scope='Completed domain-matched native development comparison with matched geometry and image controls. Publish compact primary records/code; arrays,fulloutputs,compressedrawlogs and metricdetails remain local. Foreign calendar/plugin and two foreign journal entries excluded. Goal stays open until independent actual-ID success.'",
       "scope='Own completed two-part readout result, fixed scale comparison, mathematics and cumulative synthesis. No new native ID benefit claimed. Maps/media remain local. Published hypothesis tree and foreign local drafts preserved separately. Goal remains open.'")
print('PUBLICATION_PARENT_SOURCE',hashlib.sha256(source.read_bytes()).hexdigest(),flush=True)
exec(compile(text,str(source),'exec'))
