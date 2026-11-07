from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_frame_masa_association_20261027 import audit
if __name__=='__main__':audit('frame_masa_validation')
