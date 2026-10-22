from datetime import datetime
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_localization_labels_20261023 import read,write,sha,now,RUN,journal
folder=RUN/'recovery1';assert not folder.exists();folder.mkdir()
p=read(RUN/'protocol.json');c=read(RUN/'controller.json')
p.update(budget_seconds=max(1,int((datetime.fromisoformat(c['save_at'])-now()).total_seconds())),save_at=c['save_at'],hard_stop=c['hard_stop'],original_protocol_sha256=sha(RUN/'protocol.json'),service_change='Enable faulthandler only; first attempt crashed0xC0000005 before new training. Original scientific source unchanged. Same original deadline.')
p['sha256']['scripts/z13_localization_labels_recovery_20261023.py']=sha(ROOT/'scripts/z13_localization_labels_recovery_20261023.py')
write(folder/'protocol.json',p)
for n in ['status.json','label_audit.json','reproduction.json','controller_result.json']:
    shutil.copy2(RUN/n,folder/('original_'+n))
write(RUN/'execution.json',dict(controller_root=folder.relative_to(ROOT).as_posix(),original_deadline=c['hard_stop']))
journal('служебная ошибка CPU до обучения нового варианта','Прежние18моделей воспроизведены; мишень D2298кандидатов/1740строк (label_audit.json). На пробе32старых шагов процесс завершился0xC0000005 без Python-исключения, новые модели не обучены. Исходные журналы сохранены. recovery1/protocol.json включает только трассировку аварии, прежний срок '+c['hard_stop']+'. Код и настройки опыта не меняются. Отдельная консоль исправлена копией UTF8BOM, исходный файл не перезаписан. Автор дополнительно разрешил автономные необходимые прогоны/исправления/коммиты до07:40МСК23.10.2026.')
