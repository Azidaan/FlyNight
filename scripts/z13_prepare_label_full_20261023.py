from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_localization_labels_20261023 import read,write,sha,now,RUN,journal,IDENT
folder=RUN/'full';assert not folder.exists();folder.mkdir()
end=read(RUN/'recovery2/controller_result.json');assert end['all_owned_ended']
assert read(RUN/'timing.json')['old_loss_max_abs']==0
p=read(RUN/'protocol.json')
p.update(budget_seconds=3600,original_protocol_sha256=sha(RUN/'protocol.json'),authorization='Author explicitly authorized necessary nonstop runs, decisions and repairs until07:40MSK23.10.2026. Preparation completed; measured conservative estimate2323 seconds; separate full CPU phase<=3600seconds plus20seconds termination. No prior phase extended.',service_change='Empty CUDA mask; identical frozen scientific worker and18models/settings. Only resource phase differs from preparation; source protocol remains preserved.')
p['sha256']['scripts/z13_localization_labels_full_20261023.py']=sha(ROOT/'scripts/z13_localization_labels_full_20261023.py')
p['sha256']['runs/z13_localization_labels_20261023/protocol.json']=sha(RUN/'protocol.json')
write(folder/'protocol.json',p)
for name in ['timing.json','label_audit.json','reproduction.json','execution.json']:
    shutil.copy2(RUN/name,folder/('preparation_'+name))
write(RUN/'execution.json',dict(controller_root=folder.relative_to(ROOT).as_posix(),phase='separate_full_after_preparation',author_deadline=p['hard_stop']))
catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(v for v in cat['experiments'] if v['id']==IDENT)
card['state']='running'
card['evidence'].append(dict(kind='file',ref='runs/z13_localization_labels_20261023/full/protocol.json',claim='Отдельный полный CPU этап после пробы, разрешён новой автономной инструкцией автора; научные условия прежние'))
write(catpath,cat)
journal('подготовка завершена, отдельный полный этап новой метки','Воспроизведены все18моделей и32старых шага (расхождение потерь0); технический сбой устранён пустой маской CUDA при CPU. Консервативная стоимость2323с по timing.json превысила первоначальный900с предел подготовки; подготовка завершилась без новых обученных моделей. По новому прямому разрешению автора на необходимые автономные прогоны до07:40 оформлен отдельный полный CPU этап3600с+20с завершения (full/protocol.json). Исходный протокол и старые аварии сохранены;18моделей,1150шагов,веса классов,сид,данные,метрика и критерии не меняются. Новых научных результатов ещё нет.')
