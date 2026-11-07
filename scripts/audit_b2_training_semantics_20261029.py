"""New version: exact predictor and D projection flags from original dev pass."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from scripts import audit_b2_training_gpu_predictor_20261029 as audit
    from tracking.b2_training_semantics_loader import load_branches_training_semantics
    original_read=Path.read_text
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    def read_text(path,*a,**kw):
        code=original_read(path,*a,**kw)
        if path==source:
            code=code.replace("folder=RUN/'b2_sparse_training_audit'", "folder=RUN/'b2_training_semantics_audit'")
            # Keep the exact function loaded into the compiled namespace.
            code=code.replace("    from tracking.b2_pilot_runtime import load_branches,modified_features", "    from tracking.b2_pilot_runtime import modified_features\n    from tracking.b2_training_semantics_loader import load_branches_training_semantics as load_branches")
        return code
    old_read=audit.service.read
    def service_read(path):
        if path==audit.RUN/'b2_training_gpu_predictor_audit/protocol.json':path=audit.RUN/'b2_training_semantics_audit/protocol.json'
        return old_read(path)
    audit.service.read=service_read
    Path.read_text=read_text
    try:audit.main()
    finally:Path.read_text=original_read

if __name__=='__main__':main()
