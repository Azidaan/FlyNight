"""Строгий прежний аудит; третья версия явно согласует все CPU-ссылки."""
from pathlib import Path
import sys,os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    source=ROOT/'scripts/audit_b2_sparse_training_recovery2_20261029.py'
    code=source.read_text(encoding='utf-8').replace('b2_sparse_training_audit_recovery2','b2_sparse_training_audit_recovery3').replace('b2_cpu_loader_recovery2','b2_cpu_loader_recovery3')
    ns={'__name__':'b2_audit_cpu_recovery3','__file__':str(source)}
    exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
