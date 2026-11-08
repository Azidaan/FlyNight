"""Self-contained review context; exact code, paired outcomes and known uncertainties."""
import sys,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def function_source(path,name):
    text=path.read_text(encoding='utf-8-sig');tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])


def main():
    paired=read(RUN/'gemini_paired_review.json');report=read(RUN/'comparison_report.json');mechanism=read(RUN/'mechanism_report.json')
    source=ROOT/'tracking/spatial_identity_assignment.py';native=ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py'
    context=dict(
      purpose='Критическая проверка проекта YOLO26 + XYWHByteTrack + полный зрительный модуль FlyVis. Доступа к проекту у рецензента нет; достаточные факты и код приведены здесь.',
      responsibility='Прежний запрос составлял Codex. Неполное описание формулы или реализации — недостаток переданного контекста, а не доказательство некомпетентности Gemini.',
      status='Условная однокадровая проверка завершена. Полный последовательный пилот20val выполняется. Его HOTA ещё не является готовым результатом. Ниже только проверенные сохранённые факты.',
      system=dict(detector='YOLO26s, сохранённые неизменные детекции; текущие вмешательства не меняют детектор.',
        tracker='XYWHByteTrack: один общий первый решатель, затем штатные второй каскад, подтверждение новых треков, рождение и очистка дубликатов.',
        flyvis_levels='Ядро FlyVis — рекуррентная сеть. Полный модуль — подготовка входа, ядро, состояние и считывание. Авторская система включает детектор, трекер, память и параллельные ветви. FlyVis сама по себе не выдаёт ID объектов.',
        features='Проверяемый сигнал: восемь пространственных нормированных активностей T4a–d/T5a–d, сохраняемых по колонкам. Это НЕ предложенные ранее полярность P=(ON−OFF)/(ON+OFF) и веса W.',
        reference='Каждый конкурирующий трек имеет собственную историю последних четырёх реальных наблюдений; при короткой истории первое повторяется. Текущий причинный путь — четыре наблюдения выбранного трека плюс его поисковые кадры до текущего. Текущий глаз общий для соперников одного столбца.',
        alignment='Одно совмещение определяется только геометрией и ковариацией Калмана, минимумом нормированной невязки на фиксированной сетке; сигнал и GT не выбирают положение. НЕТ поиска максимума корреляции по сдвигам и НЕТ требования выигрыша при всех совмещениях.',
        score='По общим допустимым пространственным соответствиям: центрирование каждого канала и нормированное скалярное произведение всех каналов/колонок. Общие ячейки рамок не удаляются; повторные отображения на одну прошлую колонку усредняются.',
        geometry_bonus='alpha применяется только к Lost-строке с ровно двумя поисковыми реальными рамками, из которых ровно одна допустима по фиксированной геометрии. Это предложение общему решателю, не принудительное назначение и не доказательство личности.',
        correction='C_new[i,j] = C_base[i,j] − alpha*g[i,j] − beta*(S[i,j] − mean(S[k,j] for ALL k in R_j INCLUDING i)). Рецензент прежде использовал среднее БЕЗ i; это другая величина.',
        undefined='Недостаточная общая поддержка или неопределённое сходство хотя бы одного соперника отключает зрительную поправку всего столбца; геометрическая поправка сохраняется.',
        identity='Возврат вызывает re_activate(new_id=False). Конечный исход определяется после штатного обновления и удаления дубликатов, не по первой победе.'),
      actual_code=dict(corrected_cost=function_source(source,'corrected_cost'),centered_similarity=function_source(source,'centered_similarity'),
        remove_duplicate_stracks=function_source(native,'remove_duplicate_stracks'),
        tracklet_len='В фактическом STrack.re_activate tracklet_len сбрасывается в0, update увеличивает его на1. Это НЕ накопленное число подтверждений за всю жизнь. Для такого показателя нужен отдельный причинно обновляемый счётчик.'),
      frozen_parameters=report['parameters'],
      conditional_results=dict(unit='641 запрос из исторических26видео; одно вмешательство из исходного состояния, не полный трекер.',
        totals={a:report['summaries']['evaluate26'][a]['totals'] for a in ['B0','G','G_R1','G_FLYVIS','G_PERM']},
        paired_video_comparisons=report['evaluate26_comparisons'],
        paired_geometry_to_flyvis=paired['transitions'],changed_queries=paired['changes'],
        correct_interpretation='Против геометрии FlyVis отменил7назначений(4правильных,3чужих), добавил2(1правильное,1чужое). Из маргинальных22/11и19/9 этого вывести нельзя. Отсутствие значимого преимущества не доказывает равенство случайному прореживанию.',
        limitations=report['limitations']),
      original_two_cases=mechanism['requested_cases'],
      cleanup_facts=dict(G=report['summaries']['evaluate26']['G']['totals'].get('confirmed_duplicate_removal_after_first_assignment'),
        G_FLYVIS=report['summaries']['evaluate26']['G_FLYVIS']['totals'].get('confirmed_duplicate_removal_after_first_assignment'),
        interpretation='Удаление выигравшего ID действительно наблюдается. Сам факт конкуренции за детекцию не гарантирует IoU>.85 между состояниями треков. Обновлённая рамка Калмана не обязана точно равняться детекции.'),
      coverage=dict(source='coverage_followup_recovery1.json',
        distinct_questions='Физическое перекрытие машины, усечение детекторной рамки, выход за квадрат входного глаза, выпуклая оболочка центров721колонки и поддержка13×13 входного фильтра — разные вещи.',
        implementation='Retina обрабатывает весь исходный кадр причинно; затем вырезается предсказанное квадратное окно, cv2.INTER_AREA до391×391, 13×13 BoxEye и721колонка. Расширение окна с повторным уменьшением меняет плотность сетки на объекте.',
        measured='Из178положительных запросов: истинный кандидат полностью в квадрате139раз, в оболочке центров102раза, в объединённой входной поддержке112раз. Геометрия не доказывает, что обрезание вызвало ошибку узнавания.',
        current_scope='Причинное сравнение расширенного входа ещё НЕ выполнено; полный текущий пилот не меняет окно.'),
      literature_checks=[
        dict(source='https://arxiv.org/html/2203.14360',finding='ORU исправляет фильтр после восстановления соответствия; статья не доказывает тождество двух треков по интерполяции между ними. Официальный CVF PDF при проверке вернул403; авторская версия arXiv доступна.'),
        dict(source='https://arxiv.org/html/2308.00783',finding='Hybrid-SORT использует направления четырёх углов, высотное пересечение и предсказанную уверенность детектора. TCM не является правилом владения ID по числу прошлых подтверждений. AAAI PDF был закрыт автоматической защитой; сверена авторская версия.'),
        dict(source='https://raw.githubusercontent.com/vukasin-stanojevic/BoostTrack/master/tracker/boost_track.py',finding='Код увеличивает уверенность подходящих слабых детекций; несопоставленные детекции выше порога рождают новые треки. Общего жёсткого запрета рождения возле существующего трека здесь нет.'),
        dict(source='https://raw.githubusercontent.com/dyhBUPT/StrongSORT/master/AFLink/model.py',finding='AFLink — обучаемый сверточный временной модуль с классификатором; forward оставляет первые3компоненты(t,x,y), а не все(t,x,y,w,h).'),
        dict(source='https://raw.githubusercontent.com/dyhBUPT/StrongSORT/master/AFLink/AppFreeLink.py',finding='Обрабатывает готовый файл траекторий и переписываетID. Реализация допускает временной порог(-10,30), поэтому абсолютное утверждение «только непересекающиеся» неточно.'),
        dict(source='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09351.pdf',finding='BUSCA продолжает несопоставленные траектории после основного сопоставления. Если нужная детекция уже занята новым ID той же машины, отдельный каскад сам по себе старый ID не вернёт.')],
      requested_review='Ранжируй не более3минимальных изменений по ценности проверки. Для каждого: какую проблему решает, точная операция в коде, сильнейшая альтернатива, один различающий контроль без FlyVis, критерий пользы полного трекера и риск. Отделяй установленное от гипотезы. Не называй необходимым/достаточным критерием тождества неиспытанную эвристику. Не объединяй перенос состояния, слияние ID, очистку дубликатов, новое центрирование и новый порог в один неделимый патч. Для покрытия раздели изменение входа и маски считывания при одинаковых физических координатах, разрешении, временной истории и выравнивании. Не обещай чистую аддитивную причинную декомпозицию при взаимодействующих факторах. Новые опыты по этому тексту автоматически не запускаются.',
      provenance=dict(commit='7f702f5364bb2133b15c3a6c0ddf28dd7c716aeb',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source,native,RUN/'comparison_report.json',RUN/'gemini_paired_review.json',RUN/'mechanism_report.json']}))
    write(RUN/'gemini_context_corrected.json',context,new=True)


if __name__=='__main__':main()
