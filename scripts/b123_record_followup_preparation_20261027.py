"""Записать факты подготовки и доказать, что готовая очередь увидит новые этапы."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    registry=read(RUN/'process_registry.json')['processes']
    queue=next(r for r in registry if r['pid']==34712)
    process=psutil.Process(queue['pid'])
    assert abs(process.create_time()-queue['created'])<.01
    assert any('b123_ready_queue_recovery2_20261027.py' in s for s in process.cmdline())
    jobs={}
    for name in ('40_busca_readiness.json','50_ltr_readiness.json','60_recovery_motion_control.json'):
        j=read(RUN/'ready_jobs'/name);folder=ROOT/j['folder']
        assert sha(ROOT/j['script'])==j['script_sha256']
        assert sha(folder/'protocol.json')==j['protocol_sha256']
        jobs[name]=j
    guard=read(RUN/'deadline_status.json')
    freshness=(datetime.now(timezone.utc)-datetime.fromisoformat(guard['updated'])).total_seconds()
    assert freshness<10
    value=dict(at=datetime.now(timezone(timedelta(hours=3))).isoformat(),queue_identity=queue,
        jobs=jobs,deadline_guard_fresh_seconds=freshness,original_authorization_unchanged=True,
        source_manifest=read(RUN/'recovery_source_snapshot/manifest.json'),
        source_preparation_failures_preserved=[
            'Первый сетевой запрос в ограниченной среде:WinError10013;публичное чтение прошло через разрешённый сетевой доступ.',
            'Системное хранилище TLS дало CERTIFICATE_VERIFY_FAILED;использован certifi с сохранением проверки сертификата.',
            'Архив BUSCA превысил установленный предел16МБ;архив не сохранён,получены только выбранные текстовые файлы через дерево GitHub.'],
        weights_link_checks=[dict(file_id='15LB6SPHtDc-4_fLQtRIzU1YWOTF6vNf',
            urls=['https://drive.usercontent.google.com/download?id=15LB6SPHtDc-4_fLQtRIzU1YWOTF6vNf&export=download&confirm=t',
                  'https://drive.google.com/uc?export=download&id=15LB6SPHtDc-4_fLQtRIzU1YWOTF6vNf'],
            observed_http_status=[404,404],weights_downloaded=False,
            interpretation='Ограничение доступности проверенных адресов в этой сессии. Доступность папки/другого официального адреса ещё не установлена;это не доказательство удаления весов.')],
        no_method_gpu_run_yet=True,no_new_dataset=True,hypothesis_graph_unchanged=True)
    target=RUN/'followup_preparation_verified.json';assert not target.exists()
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — контроль60 и публичные исходники\n\n'
            'Готово задание60_recovery_motion_control после40/50:900сCPU,тот же20val,строгий повтор rawXYWH,дополнительная выдача текущего lostKalman до30исходныхкадров. '
            'Это общий простой контроль,не BUSCA/LTR,не потолок. Условия до расчёта в recovery_motion_control/protocol.json. '
            'Производитель очереди recovery2 подтверждён поPID+созданию;guard свежий. '
            'Снимки23текстовыхисходников/точныекоммиты в recovery_source_snapshot/manifest.json. '
            'GoogleDriveпервогоBUSCAвеса:дваофициальныхdownloadадресаHTTP404;папка/другиеофициальныемаршрутынепроверены. '
            'Не объявлять веса удалёнными или метод отрицательным. Подробности followup_preparation_verified.json. '
            'УSGAR-MOTестьдокументацияиMIRвеса,ноэтодругаяработа,неподменятьLTR. '
            'Продолжатьреальнуюреализациюпослеочереди,неждатьсообщенияавтора.\n')
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_ltr_followup_preparation'
    J.journal('проверена ночная очередь продолжения и сохранены публичные исходники',
        '*Сделано.* `runs/b123_20261027/followup_preparation_verified.json`:PID/время создания очереди подтверждены,хеши заданий40/50/60сверены,независимый срок свежий. '
        'Текстовые исходники23файла,199591байт:точныекоммиты и хеши в `recovery_source_snapshot/manifest.json`. '
        'BUSCA88a9ed7e400ddd2bda5028c36c254cbc5726ce98;SGAR-MOT3bb920998723e89da89590f71f41f5a0a002e2cd. '
        'Сторонние модели не исполнялись,веса и наборы не загружались.\n\n'
        '*Ограничение.* Два проверенных downloadадреса первогоBUSCAвеса далиHTTP404;это не доказывает отсутствие веса по всем адресам. '
        'SGAR-MOT—другаяработа,неLTR. Исходные условия текущей серии не менялись. '
        'Автоматизация claude-flyvis обновлена на продолжениеBUSCA→LTRдо07:30,без дубликата. '
        'Обычный граф пересобран02:05:47,1244файла,четыреузлагипотез/настройки/чужаязаметкапобайтносохранены:graph_sync_recovery_followup/report.json. '
        'Коммит/pushпо-прежнемублокируетсуществующий.git/index.lock;оннетронут. HOTAновыхметодовнеизмерена.')
    print(json.dumps(dict(verified=True,jobs=list(jobs),queue_pid=process.pid,guard_fresh_seconds=freshness)))

if __name__=='__main__':main()
