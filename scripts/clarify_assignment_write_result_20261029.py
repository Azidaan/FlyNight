"""Preserve the original admitted-write record and explain negative contrasts."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import clarify_memory_write_result_20261029 as record

def main():
    source=inspect.getsource(record.main)
    source=source.replace('memory_write','assignment_write').replace('query_assignment_write_v1','query_assignment_write_v1')
    start=source.index('    text=');end=source.index('\n    for key,value',start)
    replacement="""    diag=service.read(RUN/'assignment_write_mechanism_review/report.json')['totals']
    first=a['primary_comparisons']['r50_last'];second=a['primary_comparisons']['detic_last']
    text=(f'Запись только после взаимно однозначного назначения ухудшила HOTA относительно последнего наблюдения с обоими извлекателями. Средние парные разницы: R50 {first["mean_per_video"]:.6f}, интервал Бонферрони98,75% {first["ci_bonferroni_9875"]}; Detic {second["mean_per_video"]:.6f}, интервал {second["ci_bonferroni_9875"]}. Оба скорректированных интервала целиком отрицательны. Относительно обычного усреднения оба скорректированных интервала включают ноль. Общий критерий пользы не пройден; положительных подтверждённых сравнений нет. Источник: '+src+'. '+f'В спорных сопоставлениях средняя давность читаемого наблюдения выросла с {diag["r50_last"]["mean_memory_age_frames"]:.3f}/{diag["detic_last"]["mean_memory_age_frames"]:.3f} до {diag["r50_guarded"]["mean_memory_age_frames"]:.3f}/{diag["detic_guarded"]["mean_memory_age_frames"]:.3f} кадров, максимум с66 до613. Источник: runs/continuation_20261029/assignment_write_mechanism_review/report.json. Наборы событий различаются; это измеренное старение памяти, а не доказанная причинность ухудшения или истинная личность.')
    decision='Не продвигать неограниченный запрет записи как улучшение. Сохранить общий зарегистрированный inconclusive и отрицательные частные сравнения. Следующее одно действие в разрешённой ветке — принудительно обновлять при следующем реальном назначении после прежнего max_frames_lost, без нового подбираемого числа; сравнить с периодическим обновлением того же срока и с last. Сначала отдельный dev-протокол. Первичная ошибка max_time_lost до регистрации сохранена отдельно; установленное поле max_frames_lost проверено по коду. Причина загрязнения личности пока не установлена.'"""
    source=source[:start]+replacement+source[end:]
    source=source.replace('cautious−last','guarded−last')
    source=source.replace('осторожная запись хуже последнего наблюдения','допуск записи без срока хуже последнего наблюдения')
    old="    service.write(ROOT/'experiments/system_problems.json',problems)"
    assert source.count(old)==1
    source=source.replace(old,"    problem['evidence'].append(dict(source='runs/continuation_20261029/assignment_write_mechanism_review/report.json',sha256=service.sha(RUN/'assignment_write_mechanism_review/report.json'),claim='Давность памяти и пересечения порога, не доказательство причинности'))\n"+old)
    ns=dict(vars(record));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
