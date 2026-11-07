"""Record the distinct admitted-write experiment, preserving the EMA experiment."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import record_memory_write_val_20261029 as record

def main():
    source=inspect.getsource(record.main)
    source=source.replace('memory_write_val','assignment_write_val').replace('query_memory_write_v1','query_assignment_write_v1')
    source=source.replace('Осторожное обновление памяти','Запись после однозначного назначения')
    source=source.replace('осторожного обновления','записи после однозначного назначения')
    source=source.replace('Осторожная запись памяти','Допуск записи памяти')
    source=source.replace('Deep OC-SORT','KeepTrack')
    old='Уверенность детектора не является меткой личности; медленное обновление может закреплять ошибку. Beta0,9 задана до dev, качество dev не использовалось для выбора варианта.'
    assert source.count(old)==1
    source=source.replace(old,'Геометрически однозначная пара не доказывает личность; запрет записи может сохранять ошибку или устаревшую внешность. Новых численных порогов нет, качество dev не использовалось для выбора варианта.')
    source=source.replace('Те же признаки, детекции и назначения; beta0,9;','Те же признаки, детекции и назначения; взаимно однозначная пара;')
    ns=dict(vars(record));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
