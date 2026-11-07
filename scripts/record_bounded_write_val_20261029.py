from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import record_memory_write_val_20261029 as record

def main():
    source=inspect.getsource(record.main).replace('memory_write_val','bounded_write_val').replace('query_memory_write_v1','query_bounded_write_v1')
    source=source.replace('Осторожное обновление памяти','Ограниченный допуск записи').replace('осторожного обновления','ограниченного допуска записи')
    source=source.replace('обычным усреднением','периодическим обновлением').replace('обычное усреднение','периодическое обновление')
    source=source.replace('Осторожная запись памяти','Ограниченный допуск записи').replace('Deep OC-SORT','KeepTrack')
    old='Уверенность детектора не является меткой личности; медленное обновление может закреплять ошибку. Beta0,9 задана до dev, качество dev не использовалось для выбора варианта.'
    assert source.count(old)==1
    source=source.replace(old,'Принудительное обновление может записать чужую личность. Срок взят из исходного max_frames_lost=30, не подбирался; качество dev не использовалось для выбора варианта. Срок проверяется после реального назначения, не создаёт наблюдений во время пропуска.')
    source=source.replace('Те же признаки, детекции и назначения; beta0,9;','Те же признаки, детекции и назначения; исходный срок max_frames_lost;')
    ns=dict(vars(record));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
