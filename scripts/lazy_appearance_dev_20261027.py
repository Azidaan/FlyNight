from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_lazy_appearance_association_20261027 import run
if __name__=='__main__':run('lazy_appearance_dev')
