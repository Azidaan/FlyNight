"""Adapt the proved native trace exporter to missing val metadata, without running it."""
import sys,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import write,sha,datetime,MSK


def main():
    original=ROOT/'research_singleton_native_confirm_20261108/native_trace.py';text=original.read_text(encoding='utf-8');source_sha=sha(original)
    replacements={
        "RUN=ROOT/'runs/singleton_native_confirm_20261108';status=for_run(RUN)":"from research_native_confirm_full_20261108.common import FULL\nRUN=FULL;status=for_run(RUN)",
        "entry=protocol['videos'][video];p=ROOT/entry['raw'];old=read(ROOT/entry['export']);oldmat=np.load((ROOT/entry['export']).with_suffix('.npz'),allow_pickle=False)":"entry=protocol['videos'][video];p=ROOT/entry['raw'];old=read(ROOT/entry['export']) if entry.get('export') else None;oldmat=np.load((ROOT/entry['export']).with_suffix('.npz'),allow_pickle=False) if old else None",
        "total=int(frames.max())+1;planner=Planner":"total=entry.get('frames',int(frames.max())+1);planner=Planner",
        "context['unconfirmed'] or str(planner.frame) in old['snapshots']":"context['unconfirmed'] or old is not None and str(planner.frame) in old['snapshots']",
        "if str(planner.frame) in old['snapshots']:":"if old is not None and str(planner.frame) in old['snapshots']:",
        "if str(frame) in old['snapshots']:":"if old is not None and str(frame) in old['snapshots']:",
        "oldmat.close();dest.parent.mkdir":"if oldmat is not None:oldmat.close()\n    dest.parent.mkdir",
        "native_fit20_metadata_recovery":"native_val20_missing_metadata_recovery",
        "native_fit20_trace_complete":"native_val20_trace_complete",
    }
    for old,new in replacements.items():
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    ast.parse(text);dest=Path(__file__).parent/'native_trace.py';assert not dest.exists();dest.write_text(text,encoding='utf-8',newline='\n')
    write(ROOT/'runs/native_confirm_policy_20261108/full_trace_source_preparation.json',dict(prepared=datetime.now(MSK).isoformat(),original_source_sha256=source_sha,adapted_source_sha256=sha(dest),changes=list(replacements),data_processed=False,scope='Allow val frames with no old query export; still require complete equality to ready B0. No scientific thresholds or native update changed.'),new=True)


if __name__=='__main__':main()
