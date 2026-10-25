"""Correct prepared catalogue evidence schema; frozen scientific code is unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now,sha


def main():
    path=ROOT/'experiments/project_history.json';cat=read(path)
    card=next(c for c in cat['experiments'] if c['id']=='z13_region_balanced_v1')
    old=card['evidence'];assert len(old)==2 and all(isinstance(x,str) for x in old)
    card['evidence']=[dict(kind='file',ref=old[0],claim='Предварительные критерии, равные контроли и бюджеты'),
                      dict(kind='file',ref=old[1],claim='Разбор отказа на обучающих и калибровочных данных')]
    write(path,cat)
    write(ROOT/'runs/z13_region_balanced_20261025/metadata_repair.json',dict(at=now().isoformat(),
       error='Evidence strings lack mandatory ref fields',old=old,new=card['evidence'],
       scientific_source_unchanged=True,source_sha256=sha(Path(__file__))))
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT='z13_region_balanced_v1'
    notes.journal('исправлена служебная схема ссылок карточки до запуска',
        'Предварительный валидатор обнаружил две строки evidence без обязательного ref. '
        'Они заменены на объекты kind/ref/claim; исходный красный журнал сохранён. '
        'Научные источники, скомпилированная функция обучения, данные, веса классов и первоначальный бюджет неизменны. '
        'Источник: runs/z13_region_balanced_20261025/metadata_repair.json.')


if __name__=='__main__':main()
