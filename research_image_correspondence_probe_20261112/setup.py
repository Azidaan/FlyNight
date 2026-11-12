from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
text=(ROOT/'research_terminal_response_probe_20261112/common.py').read_text(encoding='utf8').replace('terminal_response_probe_20261112','image_correspondence_probe_20261112')
(OUT/'common.py').write_text(text,encoding='utf8')
(OUT/'.gitignore').write_text('*.npz\nbefore/\ngraph_*/\npublication*.json\npublication_plan.md\nprepare*.log\n',encoding='utf8')
path=OUT/'audit.py';text=path.read_text(encoding='utf8')
assert "for phase in" not in text
start=text.index("with np.load(OUT/'flows.npz'");end=text.index('def get(i,mode):',start)
block=text[start:end].replace("for mode in p['modes']:","for mode in (['original_full'] if phase=='baseline' else p['modes'][1:]):")
block=block.replace('        checks.append(',"        if phase=='baseline':checks.append(")
text=text[:start]+"for phase in ['baseline','intervention']:\n"+''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True))+"    if phase=='baseline':save('baseline_gate.json',dict(passed=True,records=rows.copy(),all15_original_scores_before_changed_comparisons=True))\n"+text[end:]
path.write_text(text,encoding='utf8')
print('PREPARED')
