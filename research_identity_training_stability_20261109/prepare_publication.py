"""Prepare compact own results while preserving published and local foreign drafts."""
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
change("TEMP=ROOT/'tmp/domain_native_publication'","TEMP=ROOT/'tmp/identity_reader_review_publication'")
change("checks['catalog_tests']==58","checks['tests']==58")
change("=='42ba6c1b337f21bb3c2dcc479130426d094ab971'","=='10f93b74494bda0587ec58676805316b026ec2f5'")
change("assert len(headers)>=4", "assert len(headers)>=4")
needle="assert not any('/Гипотезы/' in p or p=='AGENTS.md' or p.startswith('scripts/') or p.endswith('.obsidian/graph.json') for p in selected)"
extra="""
for folder in ['research_shift_contrast_20261109','research_visible_geometry_head_20261109','research_identity_training_stability_20261109']:
    for path in (ROOT/folder).iterdir():
        if not path.is_file() or path.suffix not in ('.py','.json','.md','.txt'):continue
        if 'status' in path.name or path.name.endswith(('.started.json','.ended.json','.launch.json','.terminal.json')) or 'identity' in path.name or path.name in ['executor.json','launch_chain.json']:continue
        selected[path.relative_to(ROOT).as_posix()]=str(path)
for name in ['tracking/shift_profile_contrast.py','research_identity_synthesis_20261109/ПРОВЕРКА_GEMINI_v18.md','research_identity_synthesis_20261109/gemini_request_v18.txt','research_identity_synthesis_20261109/gemini_v18_displayed.txt','research_identity_synthesis_20261109/gemini_v18_submitted.txt','research_identity_synthesis_20261109/gemini_v18_submission.json','research_identity_synthesis_20261109/gemini_answer_v18_accessibility.txt','research_identity_synthesis_20261109/gemini_v18_capture.json']:
    selected[name]=str(ROOT/name)
"""
change(needle,extra+'\n'+needle)
change("scope='Completed domain-matched native development comparison with matched geometry and image controls. Publish compact primary records/code; arrays,fulloutputs,compressedrawlogs and metricdetails remain local. Foreign calendar/plugin and two foreign journal entries excluded. Goal stays open until independent actual-ID success.'",
       "scope='Own negative contrast and visible-support results, verified Gemini synthesis, and ready-model training-sensitivity comparison. No new native benefit claimed. Arrays remain local. Published hypothesis tree and local foreign drafts are preserved separately. Goal remains open until independent actual-ID success.'")
print('PUBLICATION_PARENT_SOURCE',hashlib.sha256(source.read_bytes()).hexdigest(),flush=True)
exec(compile(text,str(source),'exec'))
