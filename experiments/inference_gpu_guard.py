"""Read-only A100 exclusivity gate; never terminates another process."""
from dataclasses import asdict, dataclass
import subprocess
import threading
import time


@dataclass(frozen=True)
class GpuSample:
    time_ns: int
    name: str
    memory_mib: int
    utilization: int
    pids: tuple[int, ...]
    total_mib: int = 0


def sample_gpu():
    info = subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,memory.used,utilization.gpu,memory.total', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(info) != 1:
        raise RuntimeError('exactly one GPU required')
    name, memory, utilization, total = (p.strip() for p in info[0].split(','))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    return GpuSample(time.time_ns(), name, int(memory), int(utilization),
                     tuple(sorted(int(p.strip()) for p in apps.splitlines() if p.strip())), int(total))


def idle_reason(sample):
    if 'A100' not in sample.name:
        return 'not A100'
    if sample.pids:
        return 'foreign compute context present (idle contexts also count as occupied)'
    if sample.memory_mib > 256 or sample.utilization > 5:
        return 'device not idle'
    return None


def idle_probe():
    samples = []
    for i in range(3):
        sample = sample_gpu()
        samples.append(asdict(sample))
        reason = idle_reason(sample)
        if reason:
            return False, reason, samples
        if i < 2:
            time.sleep(1)
    return True, None, samples


def capacity_reason(sample, minimum_free_mib):
    if 'A100' not in sample.name:
        return 'not A100'
    if sample.total_mib-sample.memory_mib < minimum_free_mib:
        return 'insufficient free GPU memory'
    return None


def gpu_probe(policy, minimum_free_mib=2048):
    if policy == 'exclusive':
        return idle_probe()
    if policy != 'shared':
        raise ValueError('unknown GPU sharing policy')
    sample = sample_gpu()
    reason = capacity_reason(sample, minimum_free_mib)
    return reason is None, reason, [asdict(sample)]


class ExclusiveGpuGuard:
    """Initialize only after a clean probe and a single own CUDA context.

    nvidia-smi reports host PIDs, not necessarily container PIDs. The single new
    PID after an empty-device probe is the worker; two contexts are rejected.
    Poll throughout the block; any extra context invalidates the entire block.
    """
    def __init__(self):
        sample = sample_gpu()
        if 'A100' not in sample.name or len(sample.pids) != 1:
            raise RuntimeError('cannot establish exclusive single-worker GPU context')
        self.allowed = sample.pids
        self.samples = [asdict(sample)]
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def check(self):
        sample = sample_gpu()
        self.samples.append(asdict(sample))
        if sample.pids != self.allowed:
            self.error = 'foreign GPU context appeared; discard block'
        if self.error:
            raise RuntimeError(self.error)

    def _monitor(self):
        while not self.stop.wait(.3):
            try:
                self.check()
            except Exception as error:
                self.error = str(error)
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        self.check()


class SharedGpuObserver(ExclusiveGpuGuard):
    """User-authorized shared GPU: observe foreign activity, do not reject it."""
    def __init__(self):
        sample = sample_gpu()
        if 'A100' not in sample.name:
            raise RuntimeError('not A100')
        self.samples = [asdict(sample)]
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def check(self):
        sample = sample_gpu()
        self.samples.append(asdict(sample))
        reason = capacity_reason(sample, 256)
        if reason:
            self.error = reason
        if self.error:
            raise RuntimeError(self.error)
