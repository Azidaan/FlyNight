"""Run specified tests in a fresh process and record whether CUDA was initialized."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', required=True, type=Path)
parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
sys.path.insert(0, str(Path.cwd()))

import pytest
import torch

assert not torch.cuda.is_initialized(), 'Fresh verification process unexpectedly initialized CUDA'
reports = []


class Reports:
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.outcome != 'passed':
            reports.append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                                detail=str(report.longrepr) if report.longrepr else None))


code = int(pytest.main(args.pytest_args, plugins=[Reports()]))
cuda_initialized = torch.cuda.is_initialized()
flyvis = sys.modules.get('flyvis')
result = dict(exit_code=code, cwd=str(Path.cwd()), pytest_args=args.pytest_args,
              cuda_initialized_before=False, cuda_initialized_after=cuda_initialized,
              flyvis_device=str(flyvis.device) if flyvis else None,
              default_torch_device=str(torch.get_default_device()), cases=reports)
args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
assert not cuda_initialized, 'Verification tests unexpectedly initialized CUDA'
raise SystemExit(code)
