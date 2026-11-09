"""Preserve the source's practical role and update the now-completed coordinate check."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'experiments/useful_literature.json'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();before=path.read_bytes();p=json.loads(before)
adam=next(s for s in p['sources'] if s['id']=='adam2006_fragment_geometry')
old=adam['limitation'];adam.setdefault('revisions',[]).append(dict(date='2026-11-09',commit=head,field='limitation',old=old,reason='Координатный контроль и фиксированная замена с прежними весами теперь завершены.'))
adam['limitation']='Статья не устанавливает привязку к открытому углу и не подтверждает пользуFlyVis. Наш координатный контроль выполнен: различие существенно, но фиксированная замена с прежними весами не исправила два своих отказа. Исправленный полезный трекер не получен.'
adam['evidence'] += ['research_geometry_consistency_20261109/coordinate_report.json','research_coordinate_reader_20261109/comparison.json']
identity='sugiyama2007_covariate_shift_caution';assert all(s['id']!=identity for s in p['sources'])
p['sources'].append(dict(id=identity,title='Рабочая область обучения и условия перевзвешивания — Sugiyama и соавторы',
    problem_group='Сохранение личности',method_group='Обучение в рабочей области',
    publication='Masashi Sugiyama, Matthias Krauledat, Klaus-Robert Müller. Covariate Shift Adaptation by Importance Weighted Cross Validation. JMLR8(35),2007,985–1005.',
    url='https://www.jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf',
    sections='Проверены раздел2.1,формула3, и раздел3 с условиями изменения распределения и доказательством взвешенной проверки.',
    established='Авторы рассматривают изменение распределения входов при неизменном условном распределении выходов. Веса и свойства проверки выводятся через отношение плотностей и эти предпосылки.',
    problem='Прежняя голова обучалась на чужих связях вне необходимого геометрического допуска, а применялась внутри него.',
    application='Источник помог не переносить обычную межвидеовую точность автоматически на другой условный пул и отделить сбор реально допустимых чужих от необоснованного обещания исправить всё перевзвешиванием.',
    limitation='Это не готовый метод для нашей задачи: сохранение условного распределения не установлено. Ноль чужих в конечной выборке не означает нулевую вероятность. Вывод о недостатке конкретных примеров наш, по локальному аудиту; статья не доказывает пользуFlyVis или достаточность26связей.',
    evidence=['research_geometry_consistency_20261109/training_states_complete.json','research_working_domain_identity_20261109/results.json','research_domain_matched_reader_20261109/protocol_v2.json'],related=['sys:С2']))
temp=path.with_suffix('.domain_literature.pending');assert not temp.exists();temp.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for _ in range(30):
    assert path.read_bytes()==before
    try:temp.replace(path);break
    except PermissionError:time.sleep(.1)
else:raise PermissionError(str(path))
print('USED_LITERATURE_RECORDED_WITH_LIMITS',flush=True)
