"""Проспективная регистрация только первой разрешённой проверки Я1."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.flyvis_y1_latency_20261104 import OUT, sha, write, status


def main():
    OUT.mkdir(exist_ok=False)
    now = datetime.now(ZoneInfo('Europe/Moscow'))
    assert now.date().isoformat() == '2026-11-04'
    names = ['experiments/project_history.json', 'experiments/metrics_glossary.json',
             'dissertation/ЕДИНЫЙ_ПЛАН.md', 'FLYVIS_TRACKER_RESEARCH_ROADMAP.md',
             'experiments/flyvis_core_dynamics_hypothesis.json']
    for name in names:
        target = OUT / 'before' / name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write('authorization.json', dict(author_message='Давай, продолжаем', authorized_msk=now.isoformat(),
          context='Ответ на предложение подготовить протокол и выполнить только первую проверку штатной FlyVis без обучения',
          scope='Первый диагностический гейт Я1: известное движение, остановка, разворот; подготовка, исполнение, аудит и запись результата',
          adaptive_core_training_authorized=False, other_hypotheses_authorized=False,
          previous_graph_only_authorization='runs/flyvis_dynamics_hypothesis_20261102/before.json'))
    videos = json.loads((ROOT / 'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos']['dev'][:4]
    cases = []
    for scene in range(4):
        for contrast in [1., .25]:
            for speed in [1, 4]:
                for axis in [0, 1]:
                    for event in ['stop', 'reverse']:
                        cases.append(dict(id=f's{scene}_c{contrast}_v{speed}_a{axis}_{event}', scene=scene,
                                          contrast=contrast, speed=speed, axis=axis, event=event))
    frozen = ['scripts/flyvis_y1_latency_20261104.py', 'scripts/audit_flyvis_y1_latency_20261104.py',
              'scripts/control_flyvis_y1_latency_20261104.py', 'tracking/flyvis_latency_metrics.py',
              'tests/test_flyvis_latency_metrics.py', 'tracking/flyvis_motion.py', 'tracking/flyvis_adapter.py',
              'tools/flyvis-runtime/flyvis/network/network.py', 'tools/flyvis-runtime/flyvis/network/dynamics.py',
              'tools/flyvis-runtime/flyvis/datasets/rendering/eye.py', 'runs/train_cache/selection.json',
              'runs/flyvis/results/flow/0000/000/best_chkpt', 'runs/flyvis/results/flow/0000/000/chkpts/chkpt_00000',
              'runs/flyvis/results/flow/0000/000/_meta.yaml']
    p = dict(id='flyvis_y1_latency_gate_v1', registered_msk=now.isoformat(), videos=videos, cases=cases,
             source_image='first decoded grayscale frame of each of the first four fixed dev videos; original TRAIN split',
             fps=30., dt=1 / 60, internal_steps=2, frames=79, fade_in_seconds=1.,
             initial_static_frames=[0, 30], motion_frames=[31, 54], event_frames=[55, 78],
             geometry='512x512 central crop, integer source-image translations, resized once to 391x391; no reflected/padded scene content',
             central_mask='receptor-center radius <= 0.5*195; fixed before inference',
             modes=['hold', 'linear'], aggregations=['mean', 'last'], model='flow/0000/000',
             training=False, parameter_intervention=False, feature_tf32=False,
             output_units='Existing estimator convention gain=1: decoder*(512/169), y inverted. Absolute amplitude is nominal and not independently calibrated.',
             normalization='Subtract static baseline frames21..30; divide by mean signed pre-event plateau frames45..54. Diagnostic temporal normalization only, no fitted network/decoder.',
             eligibility='positive pre-event plateau > max(3*static_std,3e-5); floor is three times registered full-response CPU/GPU tolerance',
             reversal_delay='first post-event sample followed by two more negative signed responses; 0 means correct direction on the first changed interval',
             stop_delay='first post-event sample followed by two more |response/pre-event plateau|<=0.2; null if not reached by end',
             primary='16 high-contrast reversal cases: hold + last internal step versus DIS; all 16 pairs must be eligible and uncensored',
             proceed='Necessary gate only: mean excess delay>=1 video frame and lower95% paired bootstrap over four source images>0; otherwise no automatic adaptive-dynamics development',
             bootstrap=dict(seed=42, draws=10000, cluster='source image/video, four clusters; exploratory diagnostic, not population proof'),
             secondary='stop residual/settling, low contrast, hold vs linear, mean vs last, nominal amplitude/EPE, R1/T4/T5 traces; no search for a favorable subset',
             audit='Independently reconstruct all 5056 frames, integral-image BoxEye reference, all DIS and saved-response metrics; full CPU network replay of the first fixed hold case only',
             cpu_gpu_abs_tolerance=1e-5, disk_reserve_bytes=5_000_000_000,
             first_complete_case_price='Time is forecast only. Disk gate 2*first_saved_case*remaining+5GB<free.',
             hard_stop='2026-11-05T07:39:30+03:00', save_by='2026-11-05T07:30:00+03:00', watchdog_seconds=600,
             limitation='Global image translation on four TRAIN-scene backgrounds; not object localization, detector generalization, HOTA, proof of defective core, or validation of adaptive time constants.',
             frozen_sources={n: sha(ROOT / n) for n in frozen})
    write('protocol.json', p)
    write('unit_checks.json', dict(passed=4, seconds=10.06, command='pytest tests/test_flyvis_latency_metrics.py -q',
          scope='alignment, scale invariance, censored/weak response, exact source pixels, causal common prefix'))
    history_path = ROOT / 'experiments/project_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    assert not any(c['id'] == p['id'] for c in history['experiments'])
    history['experiments'].append(dict(id=p['id'], title='Я1 — Проверить задержку штатной FlyVis при остановке и развороте',
        area='flyvis', state='planned', started='2026-11-04', tags=['Y1', 'motion-diagnostic', 'core-dynamics'],
        question='Есть ли воспроизводимая лишняя задержка смены направления на выходе штатной FlyVis после исключения смешивания кадров и усреднения внутренних шагов?',
        hypothesis='Постоянные временные константы ядра могут ограничивать переходный отклик. Дефект ядра и причина задержки пока не установлены; подготовка входа и считыватель остаются альтернативами.',
        expected='Отделить ошибку величины движения от временной задержки и решить, есть ли основание проектировать адаптивную динамику. Заранее положительного исхода не предполагается.',
        acceptance=[dict(criterion='Полная серия64последовательностей/2способа подачи, замороженные веса, корректные входы/времена/хеши и независимый аудит всех метрик; одна полная CPU-сверка<=1e-5.', met=None),
                    dict(criterion='Все16пар высококонтрастного разворота hold+last противDIS измеримы без цензурирования; средняя лишняя задержка>=1кадра и нижняя95%граница по4исходнымизображениям>0.', met=None)],
        done='', obtained='', interpretation='', decision='', cannot_claim='', metrics=[],
        evidence=[dict(kind='json', ref='runs/flyvis_y1_latency_20261104/authorization.json', claim='Новое разрешение на первую проверку без обучения'),
                  dict(kind='json', ref='runs/flyvis_y1_latency_20261104/protocol.json', claim='Условия до новых выходов'),
                  dict(kind='json', ref='runs/fly_k1_v2/summary.json', claim='Прежний результат дообучения не повторяется и не отменяется')]))
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    glossary_path = ROOT / 'experiments/metrics_glossary.json'
    glossary = json.loads(glossary_path.read_text(encoding='utf-8'))
    glossary['metrics']['flyvis_transition_delay_frames'] = dict(title='Задержка переходного отклика FlyVis',
        ru='Число видеокадров от первой пары изображений с новым движением до трёх последовательных выходов с новым знаком; для остановки — до трёх выходов в пределах20% прежнего плато.',
        better='lower', unit='frames', caveat='Ноль означает реакцию на первом изменённом интервале; null — слабый отклик либо правое цензурирование. Операционная метрика, не доказательство причины в ядре.',
        source='runs/flyvis_y1_latency_20261104/protocol.json')
    glossary['metrics']['flyvis_nominal_motion_epe_px'] = dict(title='Ошибка номинального перемещения FlyVis',
        ru='Средняя евклидова ошибка между известным искусственным перемещением и выходом штатного преобразования декодера в координаты исходного фрагмента при усилении1.',
        better='lower', unit='px/frame', caveat='Масштаб выходного декодера независимо не откалиброван; это диагностический выход существующего преобразования, не точность локализации объектов.',
        source='runs/flyvis_y1_latency_20261104/protocol.json')
    glossary_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — Автор разрешил первую проверку Я1 без обучения

*Постановка.* Точный ответ автора: «Давай, продолжаем» на предложение выполнить первую проверку штатной FlyVis без обучения. Разрешение `runs/flyvis_y1_latency_20261104/authorization.json` заменяет прежнее ограничение только записи в граф в части этого первого диагностического этапа. Обучение адаптивного ядра и остальные гипотезы не разрешены. Карточка `flyvis_y1_latency_gate_v1`, протокол `runs/flyvis_y1_latency_20261104/protocol.json` зафиксированы до новых выходов. Четыре первых dev-фона существующего TRAIN-разбиения, первый кадр каждого; известные целочисленные сдвиги, скорости1/4pxзакадр, двеоси, контраст1/0,25, остановка/разворот:64последовательности. Это проверка зрительного движения, не обобщения детектора или трекинга.

*Получено.* Исполнение подготовлено отдельно;64последовательности ещё не рассчитаны. Проверки входов и метрик:4passed10,06с (`runs/flyvis_y1_latency_20261104/unit_checks.json`). Фиксированы штатное ядро/декодер, время30Гц с двумя шагами1/60с, удержание текущего кадра и линейное смешивание, выход последнего шага и средний; отдельныйDIS. Временная задержка нормируется по собственному плато до события; абсолютная амплитуда номинальна по прежнему преобразованию, не обучается. В словаре до измерения добавлены задержка перехода и номинальная ошибка перемещения. Пороги до измерения и исходники сSHA — в протоколе. Прежние К1/Б1/Б2 и остальные готовые серии не повторяются.

*Вывод.* Гипотеза задержки и её причины остаётся непроверенной. Даже прохождение необходимого гейта не доказывает причину в постоянных константах; альтернатива входа и считывателя сохраняется. Независимая CPU-проверка всех входов и метрик, одна полная CPU-сверка ответа сети<=1e-5 обязательны. Четыре исходных фона — четыре единицы сводки; кадры не независимые наблюдения.

*Решение.* После проверки каталога выполнить одну скрытую цепочку подготовки/GPU/CPU-сверки/аудита, одинGPU/мьютекс и контрольPID+birth. Сохранение05.11в07:30МСК, остановкадеревьев07:39:30МСК; время этапа — прогноз, не новыйcap;watchdog600с. Далее сохранить измеренный исход и решить, оправдана ли подготовка адаптивной динамики. Никакого автоматического обучения.

'''
    raw = plan.read_bytes();marker = '## Журнал изменений'.encode()
    ending = b'\r\n' if marker + b'\r\n' in raw else b'\n'
    plan.write_bytes(raw.replace(marker + ending, marker + ending + b'\n' + entry.encode('utf-8'), 1))
    status(stage='registered', done=0, total=64)
    print('Registered prospective64case diagnostic; catalog checks required before launch.')


if __name__ == '__main__':main()
