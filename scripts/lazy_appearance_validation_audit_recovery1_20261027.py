from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_lazy_appearance_association_recovery1_20261027 import audit
if __name__=='__main__':audit('lazy_appearance_validation')
