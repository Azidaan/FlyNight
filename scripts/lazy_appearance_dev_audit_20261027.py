from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_lazy_appearance_association_20261027 import audit
from scripts.prepare_lazy_appearance_validation_20261027 import prepare
if __name__=='__main__':
    audit('lazy_appearance_dev')
    prepare()
