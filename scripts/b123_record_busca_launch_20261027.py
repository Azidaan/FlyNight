"""Сохранить измеренный допуск полного пилота и точный остаток LTR."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027';PILOT=RUN/'busca_output_pilot'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def main():
    price=read(PILOT/'price.json');assert price['proceed']
    launch=read(PILOT/'launch.json');process=psutil.Process(launch['pid'])
    assert abs(process.create_time()-launch['created'])<.01
    assert 'b123_busca_output_pilot_20261027.py' in ' '.join(process.cmdline())
    status=read(PILOT/'status.json');assert status['state']=='running'
    completed=read(PILOT/'completed_videos.json');assert all(completed['exact'].values())
    proof=dict(checked=datetime.now(timezone.utc).isoformat(),launch=launch,price=price,
        first_completed_video=next(iter(completed['exact'])),baseline_arrays_exact=True,
        baseline_hota_exact=True,full_pilot_completed=False,source_report='completed_videos.json',
        limitations='Проверен запуск и первый полный видеопроход, не итог20. Собственная выходная адаптация, не воспроизведение полного BUSCA.')
    (PILOT/'launch_verified.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_output_recovery_pilot_v1'
    J.journal('выходная адаптация BUSCA допущена к полному пилоту по измеренной цене',
        '*Получено до продолжения.* Первое полное видео заняло35,359с, консервативная оценка остатка1463,642с '
        'при доступных5360,006с; источник `runs/b123_20261027/busca_output_pilot/price.json`. '
        'Исходные массивы XYWH и HOTA первого видео воспроизведены точно, '
        'источник completed_videos.json и per_video.json в том же каталоге. '
        'ПровереныPID/время создания активного исполнителя и свежий прогресс; launch_verified.json.\n\n'
        '*Решение.* Продолжить исходную серию20val без изменения порогов и без повторения готового видео. '
        'Итога20 и парного интервала ещё нет. Это собственная адаптация только дополнительной выдачи, '
        'а не полный трекер статьи BUSCA; вклад ядра FlyVis не проверяется. '
        'Каталог до публикации задания: OK/58passed; два теста строгого прошлого и неизменного трекера passed.\n\n'
        '*Остаток LTR.* Адресно проверен публичный список репозиториев автора: '
        '`runs/b123_20261027/ltr_exact_source_lookup/report.json`, сохранены исходный ответ и SHA. '
        'В ответе только личная страница и SGAR-MOT; точные код и веса селектора LTR не подтверждены. '
        'Это ограничение воспроизведения, не доказательство отсутствия реализации вообще и не отрицательный результат метода. '
        'SGAR-MOT нельзя молча выдать за LTR. После BUSCA зафиксировать итог и готовить масочную адаптацию '
        'только под собственным именем с явными отличиями. Срок07:30МСК сохраняется.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — полный пилот выходной BUSCA работает\n\n'
            'Задание100_busca_output_pilot запущено существующей queue_recovery2. '
            'launch_verified.json проверяетPID+birth. Первое видео35.359с,price.proceed=true: '
            '2*35.359*19+120=1463.642с <5360.006с. Первый rawXYWH и HOTA точно совпали. '
            'Это выходная адаптация,НЕполнаяBUSCA: трекер не меняется, память11сильныхреальныхназначений, '
            'безCMC/reliabilityгейта,пакеты4. Источники/критериидоизмерения вbusca_output_pilot/protocol.json. '
            'Не прерывать готовый пилот ради интерфейса и не повторять готовые видео. '
            'После report.json и конца дерева: независимо проверить все20baseline и обаpaired-сравнения, '
            'обновить карточку (HOTA с source/locator и ограничениями), журнал, обязательные проверки. '
            'Технический busca_runtime_profile уже записан какinfrastructure,егонепереобучать. '
            'Каталог до запускаOK/58passed. .git/index.lock нетронут. '
            'LTRexactlookup: уавтора два публичныхрепозитория — страница иSGAR-MOT; '
            'точный LTRкод/селекторвеса не подтверждены. Это не доказательство невозможности. '
            'SGAR-MOT другой метод, не подменять LTR. Масочную адаптацию при подготовке так и назвать; '
            'нужны реальные веса и предзаписанная цена, одинGPU после BUSCA, срок07:30.\n')
    print(json.dumps(dict(active_pid=launch['pid'],identity_verified=True,price_proceed=True,baseline_exact=True)))
if __name__=='__main__':main()
