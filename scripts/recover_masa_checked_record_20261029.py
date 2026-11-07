"""Reuse completed CPU summary; repair registry history serialization only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    old=RUN/'masa_final_checked_record'
    controller=service.read(old/'controller_result.json')
    assert controller['exit_code']==1 and controller['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in controller['owned'].items())
    saved=service.read(old/'scientific_summary.json')
    assert saved['passed'] and saved['complete'] and len(saved['pooled'])==5
    assert saved['audit_sha256']==service.sha(RUN/'masa_audit_fp32_recovery1/report.json')
    source=ROOT/'scripts/finalize_masa_checked_result_20261029.py'
    p=service.read(RUN/'masa_final_checked_record_recovery1/protocol.json')
    assert service.sha(source)==p['source_sha256']
    code=source.read_text(encoding='utf-8').replace('masa_final_checked_record','masa_final_checked_record_recovery1')
    start=code.index('    pooled={}\n')
    end=code.index("    assert pooled['native']==pooled['zero']",start)
    code=code[:start]+"    pooled=saved_summary['pooled']  # completed CPU calculation reused, no metric rerun\n"+code[end:]
    anchor="previous=method.copy()"
    assert code.count(anchor)==1
    code=code.replace(anchor,"previous={k:v for k,v in method.items() if k!='history'}")
    code=code.replace("if card.get(key):card.setdefault", "if card.get(key) and card[key]!=value:card.setdefault")
    ns={'__name__':'masa_record_recovery1','__file__':str(source),'saved_summary':saved}
    exec(compile(code,str(source),'exec'),ns);ns['main']()

if __name__=='__main__':main()
