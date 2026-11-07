"""Тотжестрогийпрофиль,исправленаадресациятипаNumPy2приаудитесуммы."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    original=ROOT/'scripts/profile_history_cpu_heads_first_20261028.py'
    code=original.read_text(encoding='utf-8')
    old="expected_cost=matrix.dtype.type(old['original']+self.alpha*(.5-1/(1+np.exp(-np.clip(old['score'],-80,80))))) if self.alpha else matrix.dtype.type(old['original'])"
    new="expected_cost=matrix.dtype.type(matrix.dtype.type(old['original'])+float(self.alpha*(.5-1/(1+np.exp(-np.clip(float(old['score']),-80,80)))))) if self.alpha else matrix.dtype.type(old['original'])"
    target="text=original.read_text(encoding='utf-8')"
    assert code.count(target)==1
    code=code.replace(target,target+'.replace('+repr(old)+','+repr(new)+')')
    code=code.replace("folder=RUN/'history_cpu_heads_profile'","folder=RUN/'history_cpu_heads_profile_recovery1'")
    ns={'__name__':'history_cpu_profile_recovery1','__file__':str(original)}
    exec(compile(code,str(original),'exec'),ns);ns['main']()


if __name__=='__main__':main()
