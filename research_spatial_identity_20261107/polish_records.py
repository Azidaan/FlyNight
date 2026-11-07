"""Typography only, retaining prior catalog text in revisions."""
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,write,datetime,MSK,verified_protocol


def spacing(s):
    s=re.sub(r'([А-Яа-яЁё])([0-9])',r'\1 \2',s)
    s=re.sub(r'([0-9])([А-Яа-яЁё])',r'\1 \2',s)
    s=re.sub(r'([А-Яа-яЁё])ID',r'\1 ID',s)
    s=re.sub(r'ID([0-9])',r'ID \1',s)
    s=re.sub(r'ID([А-Яа-яЁё])',r'ID \1',s)
    s=re.sub(r'([0-9])GT',r'\1 GT',s)
    s=re.sub(r'GT([А-Яа-яЁё])',r'GT \1',s)
    s=re.sub(r'—(?=[0-9\[])','— ',s)
    s=re.sub(r':(?=[0-9])',': ',s)
    s=s.replace('%и','% и').replace('%интер','% интер').replace('%границ','% границ')
    return s


def main():
    report=RUN/'РЕЗУЛЬТАТ.md';report.write_text(spacing(report.read_text(encoding='utf-8')),encoding='utf-8')
    path=ROOT/'experiments/project_history.json';p=read(path)
    e=next(x for x in p['experiments'] if x['id']=='spatial_identity_maps_v1_20261107')
    for field in ['obtained','interpretation','decision','cannot_claim']:
        old=e[field];new=spacing(old)
        if old!=new:
            e['revisions'].append(dict(date=datetime.now(MSK).date().isoformat(),commit=verified_protocol()['git'],field=field,old=old,reason='Только пробелы для читаемости; числа, смысл и состояние сохранены.'))
            e[field]=new
    write(path,p)
    path=ROOT/'experiments/system_problems.json';p=read(path)
    for problem in p['problems']:
        if problem['id'] in ['С1','С3']:
            case=problem['cases'][-1]
            for k,v in case.items():
                if isinstance(v,str):case[k]=spacing(v)
    write(path,p)


if __name__=='__main__':main()
