from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
text=(ROOT/'research_image_correspondence_probe_20261112/common.py').read_text(encoding='utf8').replace('image_correspondence_probe_20261112','candidate_support_probe_20261112')
(OUT/'common.py').write_text(text,encoding='utf8')
(OUT/'.gitignore').write_text('*.npz\nbefore/\ngraph_*/\npublication*.json\npublication_plan.md\nprepare*.log\n',encoding='utf8')
print('PREPARED')
