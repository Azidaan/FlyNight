from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_frame_masa_fp16_association_20261027 import audit
if __name__=='__main__':
    audit('frame_masa_fp16_validation')
