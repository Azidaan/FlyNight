"""Record the author's approved three-hour CPU budget without changing the protocol."""
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_target_center_20261018'


def main():
    for name in ('full_authorization.json', 'recovery_full.json', 'recovery_full_result.json', 'report.json', 'stop.request'):
        if (RUN / name).exists():
            raise RuntimeError('Inspect existing artifact before any launch: ' + name)
    workers = {'z13_target_center_probe.py', 'z13_target_center_cpu_recovery.py', 'z13_target_center_recovery_controller.py'}
    for proc in psutil.process_iter(['cmdline']):
        if any(Path(arg).name in workers for arg in (proc.info['cmdline'] or [])):
            raise RuntimeError(f'Existing worker/controller {proc.pid}')
    proto = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    for rel, digest in proto['sha256'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    now = datetime.now(timezone(timedelta(hours=3)))
    auth = dict(approved=True, authorization='Author replied Да to CPU probe up to 3 hours plus 30 seconds grace.',
                recorded_at=now.isoformat(), budget_seconds=10800, hard_grace_seconds=30,
                scope='Frozen D/C/R1 binary/center comparison; CPU only; no core training, GPU, dev, heldout75 or HOTA.',
                protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest(),
                frozen_sources_verified=len(proto['sha256']))
    with (RUN/'full_authorization.json').open('x', encoding='utf-8') as handle:
        json.dump(auth, handle, ensure_ascii=False, indent=2)
    catpath=ROOT/'experiments/project_history.json'
    cat=json.loads(catpath.read_text(encoding='utf-8'))
    card=next(c for c in cat['experiments'] if c['id']=='z13_target_center_v1')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old=card.get('decision','')
    card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=head,field='decision',old=old,
        reason='Автор согласовал полный CPU-прогон до 3 часов и 30 секунд завершения.'))
    card['state']='running'
    card['decision']='Автор разрешил полный парный CPU-прогон D/C/R1: до 10800 с + 30 с завершения, без обучения ядра. Запуск и окончание подтверждаются full_launch.json и recovery_full_result.json, не прежним замером времени.'
    card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_20261018/full_authorization.json',claim='Разрешение автора и неизменность научного протокола'))
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    planpath=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=planpath.read_text(encoding='utf-8')
    marker='## Журнал изменений\n';assert marker in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — разрешён полный CPU-прогон `z13_target_center_v1`\n\n'
    entry+='*Постановка.* Автор согласовал до 3 часов на парное сравнение целей считывателя D/C/R1 и 30 секунд на завершение. Научный протокол не меняется; разрешение runs/z13_target_center_20261018/full_authorization.json. Хеши исходных файлов проверены перед запуском. CPU, без обучения ядра, GPU, dev, отложенных 75 и HOTA. Независимый контроллер ограничивает собственное дерево процессов; отдельная консоль только отображает ход. Измеренного результата полного прогона пока нет.\n'
    planpath.write_text(plan.replace(marker,marker+entry,1),encoding='utf-8')
    (RUN/'status.json').write_text(json.dumps(dict(state='starting',stage='Authorized full CPU run; starting controller',done=0,total=0)),encoding='utf-8')
    print(json.dumps(auth,ensure_ascii=False))


if __name__ == '__main__':
    main()
