"""Подготовленная GPUпроба допускается только после полного аудита данных."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read
if __name__=='__main__':
    audit=read(RUN/'admission_dataset/audit.json');assert audit['passed']
    controller=read(RUN/'admission_dataset_audit/controller_result.json')
    assert controller['exit_code']==0 and controller['all_owned_ended']
    from scripts.recovery_appearance_gpu_profile_20261027 import run
    run()
