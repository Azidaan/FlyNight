"""Service-only Windows CPU isolation repair; frozen scientific source unchanged."""
from pathlib import Path
import sys
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
original=ROOT/'scripts/z13_balanced_ridge_20261022.py'
recovery=ROOT/'runs/z13_balanced_ridge_20261022/recovery_cpu'
p=json.loads((recovery/'protocol.json').read_text(encoding='utf-8'))
assert hashlib.sha256(original.read_bytes()).hexdigest()==p['original_script_sha256']
source=original.read_text(encoding='utf-8')
old='import torch\n    torch.set_num_threads(2);torch.set_default_device(\'cpu\')'
assert source.count(old)==1
source=source.replace(old,"import torch\n    torch.cuda.is_available=lambda:False\n    torch.cuda.device_count=lambda:0\n    torch.set_num_threads(2);torch.set_default_device('cpu')\n    assert str(torch.ones(1,device='cpu').device)=='cpu'")
assert hashlib.sha256(source.encode('utf-8')).hexdigest()==p['executed_source_sha256']
exec(compile(source,str(original),'exec'),dict(__name__='__main__',__file__=str(original)))
