"""Close the accepted write-only comparison honestly; retain all prior text."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import clarify_memory_write_result_20261029 as record

def main():
    source=inspect.getsource(record.main).replace('memory_write','bounded_write')
    start=source.index('    text=');end=source.index('\n    for key,value',start)
    replacement="""    diag=service.read(RUN/'bounded_write_mechanism_review/report.json')['totals']
    one=a['primary_comparisons']['r50_last'];two=a['primary_comparisons']['detic_last']
    text=(f'Ограниченный допуск записи хуже последнего наблюдения с обоими извлекателями. Парные разницы bounded−last: R50 {one["mean_per_video"]:.6f}, интервал Бонферрони98,75% {one["ci_bonferroni_9875"]}; Detic {two["mean_per_video"]:.6f}, интервал {two["ci_bonferroni_9875"]}. Оба скорректированных интервала целиком отрицательны. Оба сравнения с периодическим обновлением и обе вторичные разницы с неограниченным допуском имеют интервалы через ноль. Главный критерий не пройден, подтверждённых положительных сравнений нет. Источник: '+src+'. '+f'Средняя давность чтения при bounded составляет {diag["r50_bounded"]["mean_memory_age_frames"]:.3f}/{diag["detic_bounded"]["mean_memory_age_frames"]:.3f} кадров против {diag["r50_guarded"]["mean_memory_age_frames"]:.3f}/{diag["detic_guarded"]["mean_memory_age_frames"]:.3f} при неограниченном допуске, максимум74 вместо613. Источник runs/continuation_20261029/bounded_write_mechanism_review/report.json. Исправление операционной задержки измерено, но само по себе не дало подтверждённой пользы HOTA; наборы событий различаются, единая корневая причина не доказана.')
    decision='Простые изменения записи памяти при текущем способе сопоставления не продвигать как улучшение. В следующем опыте сохранять последнее наблюдение как контроль; сроков и beta по этим val не перебирать. Линия памяти и проект FlyVis этим не закрываются. Следующий обоснованный кандидат — отделить общий сдвиг стоимости от содержания внешности, с самостоятельным контролем без признаков. Это изменение сопоставления выходит за согласованную ветку только записи и требует отдельного прямого решения автора; предложение подготовлено, научное задание не опубликовано.'"""
    source=source[:start]+replacement+source[end:]
    source=source.replace('cautious−last','bounded−last').replace('осторожная запись хуже последнего наблюдения','ограниченный допуск хуже последнего наблюдения')
    old="    service.write(ROOT/'experiments/system_problems.json',problems)";assert source.count(old)==1
    source=source.replace(old,"    problem['evidence'].append(dict(source='runs/continuation_20261029/bounded_write_mechanism_review/report.json',sha256=service.sha(RUN/'bounded_write_mechanism_review/report.json'),claim='Проверка фактического сокращения давности, не доказательство причинности качества'))\n"+old)
    ns=dict(vars(record));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
