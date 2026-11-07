from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_frame_masa_association_20261027 as worker
from scripts.recovery_research_service_20261027 import RUN,read
original=worker.read
def recover(path):
    if Path(path)==RUN/'frame_masa_dev/execution_deadline.json':return read(RUN/'frame_masa_dev/recovery1/execution_deadline.json')
    return original(path)
worker.read=recover
if __name__=='__main__':worker.run('frame_masa_dev')
