"""Прочитать состав полученного checkpoint на CPU; модели не исполняются."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027/busca_checkpoint_inventory'
def write(name,value):
    path=RUN/name;temp=path.with_suffix('.next');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def main():
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    path=ROOT/p['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==p['checkpoint_sha256']
    write('status.json',dict(state='running',stage='Состав авторского checkpoint на CPU',updated=datetime.now(timezone.utc).isoformat()))
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True)
    state=value['model_state_dict'] if 'model_state_dict' in value else value
    tensors={k:v for k,v in state.items() if isinstance(v,torch.Tensor)}
    reid={k:v for k,v in tensors.items() if k.startswith('reid_encoder.model.')}
    result=dict(complete=True,top_keys=list(value)[:20],state_keys=len(state),tensor_keys=len(tensors),
        reid_tensor_keys=len(reid),contains_reid_tensors=bool(reid),
        shapes={k:list(v.shape) for k,v in tensors.items()},
        total_tensor_elements=sum(v.numel() for v in tensors.values()),cuda_initialized=torch.cuda.is_initialized(),
        models_executed=False,all_required_weights_verified=False,
        next_action='Если полученный checkpoint содержит все используемые веса извлекателя, сверить ключи с исходной архитектурой и не загружать повторно избыточный model_feats. Совместимость и короткий профиль — отдельная ступень до полной проверки трекера.')
    assert not result['cuda_initialized']
    write('report.json',result);write('status.json',dict(state='completed',stage='Состав сохранён;полнота ещё проверяется',updated=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
