from datetime import datetime
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_localization_labels_20261023 import read,write,sha,now,RUN,journal
folder=RUN/'recovery2';assert not folder.exists();folder.mkdir()
p=read(RUN/'protocol.json');c=read(RUN/'controller.json')
p.update(budget_seconds=max(1,int((datetime.fromisoformat(c['save_at'])-now()).total_seconds())),save_at=c['save_at'],hard_stop=c['hard_stop'],original_protocol_sha256=sha(RUN/'protocol.json'),service_change='Only CUDA_VISIBLE_DEVICES empty string instead of -1. Synthetic32step CPU autograd crashes with -1 and passes with empty string on torch2.10.0+cu130. Same CPU tensors,2threads,optimizer,loss,seeds. Original900second deadline retained.')
p['sha256']['scripts/z13_localization_labels_recovery2_20261023.py']=sha(ROOT/'scripts/z13_localization_labels_recovery2_20261023.py')
write(folder/'protocol.json',p)
shutil.copy2(RUN/'execution.json',folder/'previous_execution.json')
write(RUN/'execution.json',dict(controller_root=folder.relative_to(ROOT).as_posix(),original_deadline=c['hard_stop']))
journal('устранён воспроизводимый сбой CPU autograd','На синтетическом прежнем узле32шага аварийно завершались при CUDA_VISIBLE_DEVICES=-1; при пустой строке все32шага прошли. Успешный прежний CPU data104 также использовал пустую строку. recovery1 сохранил трассу _engine_run_backward. В recovery2 меняется только служебная маска устройства; научный worker неизменен,2потока и алгоритм прежние, контроль32старых шагов обязателен. Срок остаётся '+c['hard_stop']+'. Это служебная диагностика, не научный результат.')
