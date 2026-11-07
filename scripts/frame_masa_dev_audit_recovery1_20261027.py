from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_frame_masa_association_20261027 as worker
from scripts import prepare_frame_masa_chain_20261027 as preparation
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
original=worker.read
def recover(path):
    if Path(path)==RUN/'frame_masa_dev/controller_result.json':return read(RUN/'frame_masa_dev/recovery1/controller_result.json')
    return original(path)
worker.read=recover
preparation.read=recover
# Конец исправленногоdevстрогопроверенфункциейauditдоэтогоперехода.
base_publish=preparation.publish
def publish(name,folder,script):
    changed={'scripts/frame_masa_validation_20261027.py':'scripts/frame_masa_validation_recovery1_20261027.py','scripts/frame_masa_validation_audit_20261027.py':'scripts/frame_masa_validation_audit_recovery1_20261027.py'}
    return base_publish(name,folder,changed.get(script,script))
preparation.publish=publish
if __name__=='__main__':
    worker.audit('frame_masa_dev')
    preparation.prepare_validation()
