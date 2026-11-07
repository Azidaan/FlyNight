import os,subprocess,sys
import pytest
from tracking.windows_memory_job import MemoryJob

def test_reject_invalid_limit_without_assigning_current_test_process():
    with pytest.raises(ValueError):MemoryJob(0)

@pytest.mark.skipif(os.name!="nt",reason="Windows job limits")
def test_os_refuses_allocation_above_job_limit_in_isolated_child():
    code=("from tracking.windows_memory_job import MemoryJob\n"
          "job=MemoryJob(48*1024**2)\n"
          "assert job.read()['limit_active']\n"
          "try:\n x=bytearray(64*1024**2)\n"
          "except MemoryError:\n print('ALLOCATION_BLOCKED')\n"
          "else:\n raise AssertionError('OS memory cap ineffective')\n")
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert "ALLOCATION_BLOCKED" in result.stdout
