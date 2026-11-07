"""Тот же независимый аудит для отдельного служебного восстановления."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];ORIGINAL=ROOT/'scripts/b123_audit_supported_transfer_20261027.py';RUN=ROOT/'runs/b123_20261027/supported_transfer/recovery1';OUT=ROOT/'runs/b123_20261027/supported_transfer/recovery1_audit'
def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==protocol['original_audit_sha256']
    ns={'__name__':'frozen_audit_import','__file__':str(ORIGINAL)};exec(compile(ORIGINAL.read_text(encoding='utf-8'),str(ORIGINAL),'exec'),ns);ns['RUN']=RUN;ns['OUT']=OUT;ns['main']()
if __name__=='__main__':main()
