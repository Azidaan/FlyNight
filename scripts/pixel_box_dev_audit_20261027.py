from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_pixel_box_evaluation_20261027 import audit
from scripts.pixel_box_evaluation_schedule_20261027 import prepare_val
if __name__=='__main__':
    audit('pixel_box_dev')
    prepare_val()
