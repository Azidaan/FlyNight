"""Register authorized execution before any parity or scientific output."""
import subprocess
from research_identity_assignment_run_20261108.common import *


def main():
    parent=ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json'
    p=read(parent);assert not (RUN/'execution_protocol.json').exists()
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    assert base=='7f702f5364bb2133b15c3a6c0ddf28dd7c716aeb'
    sources=['tracking/spatial_identity_assignment.py','tracking/xywh_bytetrack.py','tracking/z13_region_assignment.py',
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
        '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
        '.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py',
        'research_identity_assignment_run_20261108/common.py','research_identity_assignment_run_20261108/zero_parity.py',
        'tests/test_spatial_identity_assignment.py']
    write(RUN/'execution_protocol.json',dict(id='identity_assignment_v2_run_20261108',registered=datetime.now(MSK).isoformat(),
        parent_protocol=str(parent.relative_to(ROOT)),parent_sha256=sha(parent),base_commit=base,
        authorization='User requested implementation, missing-data recovery and matched comparison. First zero-intervention parity; no repeated completed calculations. Emphasize correct/wrong reactivations and harm. Follow all prespecified gates.',
        deadline='2026-11-09T07:40:00+03:00',arms=p['arms'],acceptance=p['inference'],
        sources={n:sha(ROOT/n) for n in sources}),new=True)
    rows=read(ROOT/'research_spatial_identity_20261107/readiness_rows.json')
    first=min((r for r in rows if r['split']=='fit84'),key=lambda r:(r['video'],r['episode'],r['uid']))
    phase={n:sha(ROOT/n) for n in sources}
    raw=f"runs/train_cache/{first['video']}/detections/raw_video.npz";phase[raw]=sha(ROOT/raw)
    write(RUN/'zero_phase.json',dict(video=first['video'],first_uid=first['uid'],sources=phase,
        scope='First complete source-native vs zero-intervention rollout and states. No prior complete parity receipt exists.'),new=True)
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    assert not any(e['id']=='identity_assignment_v2_run_20261108' for e in cat['experiments'])
    cat['experiments'].append(dict(id='identity_assignment_v2_run_20261108',title='Единственная рамка и конкурирующие ID — геометрия против пространственного сигнала FlyVis',area='flyvis',
        state='running',started=datetime.now(MSK).date().isoformat(),
        question='Исправляет ли общий решатель возврат прежнего ID без лишних чужих назначений и добавляет ли FlyVis пользу сверх общей геометрии и контролей?',
        hypothesis='Геометрическое предложение и относительное сходство историй могут изменить правильное назначение; польза не установлена.',
        expected='Правильные восстановления превышают неправильные; вклад сигнала отделён от геометрии и общего сдвига.',
        acceptance=[dict(criterion='Нулевые поправки точно воспроизводят исходные назначения, состояния и полный выход.',met=None),
          dict(criterion='Полный экспорт причинных запросов, конкурентов и положительных/отрицательных исходов без GT во входе.',met=None),
          dict(criterion='Параметры выбраны только на fit20; оценка evaluate26 без подбора, с ценой ошибок остальных треков.',met=None),
          dict(criterion='Переход к полному пилоту только по заранее заданному J-гейту; принятие только по HOTA и ограничениям вреда протокола.',met=None)],
        obtained='',interpretation='',decision='',cannot_claim='Исторические разработческие запросы и val; неизвестные метки не отрицательные. Условные назначения не равны HOTA. Критерии исходного протокола не меняются.',
        evidence=[dict(kind='file',ref='runs/identity_assignment_20261108/execution_protocol.json',claim='Прямое разрешение и условия до исполнения'),
                  dict(kind='file',ref='research_identity_assignment_protocol_20261108/protocol_v2.json',claim='Постановка, контроли и критерии до опыта')]))
    write(cp,cat)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — По прямому поручению автора начато исполнение `identity_assignment_v2_run_20261108`: реализация поправки, восстановление недостающих данных и сравнение с FlyVis/без него. Сначала полное совпадение нулевого вмешательства с исходным трекером. Готовые карты и расчёты переиспользуются по хешам. Приоритет проверки — правильные и чужие назначения, вред конкурирующим ID; условный выбор не подменяет HOTA. Критерии из `research_identity_assignment_protocol_20261108/protocol_v2.json` сохранены. Источники и срок — `runs/identity_assignment_20261108/execution_protocol.json`. Результатов пока нет.\n\n"
    write(RUN/'registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    status('registered_zero_parity_pending',scientific_result=None)


if __name__=='__main__':main()
