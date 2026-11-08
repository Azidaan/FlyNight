from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parent.parent/'verify.py'
sys.path.insert(0,str(SOURCE.parent))
code=SOURCE.read_text(encoding='utf-8')
for filename in ('events.json','summary.json','audit.json'):
    code=code.replace("P/'"+filename+"'","P/'recovery1'/ '"+filename+"'")
exec(compile(code,str(SOURCE),'exec'),{'__file__':str(SOURCE),'__name__':'__main__'})
