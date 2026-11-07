import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.assignment_write_dev_20261029 import main
if __name__=='__main__':
    sys.argv.append('--audit');main()
