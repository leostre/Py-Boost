"""Run unit tests without CUDA acceptance checks or scientific GPU training.

Usage: python tools/hasboost_cpu_tests.py tests -q
Real GPU validation remains explicit: use the original documented GPU protocol.
"""
import os
from pathlib import Path
import sys


class CPUOnly:
    def pytest_configure(self,config):
        import pytest
        self.original=pytest.importorskip
        def importorskip(name,*args,**kwargs):
            if name.split('.')[0]=='cupy':
                pytest.skip('CPU publication verification; CUDA acceptance is a separate protocol')
            return self.original(name,*args,**kwargs)
        pytest.importorskip=importorskip

    def pytest_unconfigure(self,config):
        import pytest
        pytest.importorskip=self.original


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['HASBOOST_GPU_ACCEPTANCE']='0'
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root));sys.path.insert(0,str(Path.cwd()))
    import pytest
    return pytest.main(sys.argv[1:] or ['tests','-q'],plugins=[CPUOnly()])


if __name__=='__main__':raise SystemExit(main())
