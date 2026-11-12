"""Prepare a bounded terminal-response check before any modified model replay."""
from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
prior=ROOT/'research_fixed_crop_probe_20261112'
text=(prior/'common.py').read_text(encoding='utf8').replace('fixed_crop_probe_20261112','terminal_response_probe_20261112')
(OUT/'common.py').write_text(text,encoding='utf8')
# Reuse the independently implemented score evaluator, replacing only input schedule validation.
text=(prior/'audit.py').read_text(encoding='utf8')
start=text.index("render=read(OUT/'render_manifest.json')")
end=text.index("pred=read(",start)
text=text[:start]+text[end:]
start=text.index("                length=ep['length'];")
end=text.index("                for arm in ['C34','R1']:",start)
replacement='''                length=ep['length'];hx=expanded[f'{mode}_{ci}_{ei}'];prefix=original[:length]
                if mode=='terminal_hold':prefix=np.tile(prefix[-1:],(length,1))
                target=np.vstack([prefix,np.tile(prefix[-1:],(30,1))]) if mode.endswith('_hold') else prefix
                assert np.array_equal(hx,target) and np.array_equal(hx[-1],original[length-1])
                rec=next(r for r in rep['records'] if r['case']==ci and r['endpoint']==ei and r['mode']==mode)
                assert rec['update_calls']==len(target)-1 and rec['terminal_input_unchanged']
'''
text=text[:start]+replacement+text[end:]
text=text.replace("['last_pair_fixed','all_fixed']","['history_hold','terminal_hold']")
text=text.replace('Independent frame/crop schedule reconstruction and view lookup;','Independent constant-input schedule reconstruction;')
(OUT/'audit.py').write_text(text,encoding='utf8')
(OUT/'.gitignore').write_text('*.npz\nbefore/\ngraph_*/\npublication*.json\npublication_plan.md\nprepare*.log\n',encoding='utf8')
print('PREPARED')
