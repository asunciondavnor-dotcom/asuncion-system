"""Low-volume resource diagnostics for the Render backend."""
from contextlib import contextmanager
from functools import cache
import json
import os
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, process_time, sleep
from typing import NamedTuple


MIB = 1024 * 1024
SAMPLE_SECONDS = 300
_peak_lock = Lock()
_observed_peak = 0


class Snapshot(NamedTuple):
    clock: float
    cpu_seconds: float
    ram_mib: float | None
    peak_ram_mib: float | None
    scope: str
    peak_source: str


def _enabled():
    return os.environ.get('PSMS_RENDER_OCR_BUDGET') == '1'


@cache
def _cgroup_directory():
    try:
        group = next(row[3:] for row in Path('/proc/self/cgroup').read_text().splitlines()
                     if row.startswith('0::'))
    except (OSError, StopIteration):
        return None
    return Path('/sys/fs/cgroup') / group.lstrip('/')


def _snapshot():
    group = _cgroup_directory()
    if group is not None:
        try:
            current = int((group / 'memory.current').read_text())
            cpu = dict(row.split() for row in (group / 'cpu.stat').read_text().splitlines())
            seconds = int(cpu['usage_usec']) / 1_000_000
        except (OSError, ValueError, KeyError):
            pass
        else:
            try:
                peak = int((group / 'memory.peak').read_text())
                source = 'cgroup'
            except (OSError, ValueError):
                global _observed_peak
                with _peak_lock:
                    _observed_peak = max(_observed_peak, current)
                    peak = _observed_peak
                source = 'sampled'
            return Snapshot(monotonic(), seconds, current / MIB, max(peak, current) / MIB, 'cgroup', source)

    try:
        resident = int(Path('/proc/self/statm').read_text().split()[1])
        current_mib = resident * os.sysconf('SC_PAGE_SIZE') / MIB
    except (OSError, ValueError, IndexError):
        current_mib = None
    try:
        import resource
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, OSError):
        peak_mib = None
    return Snapshot(monotonic(), process_time(), current_mib, peak_mib, 'process', 'process')


def _log(operation, start, end, status):
    elapsed = max(end.clock - start.clock, 0)
    cpu = max(end.cpu_seconds - start.cpu_seconds, 0) if end.scope == start.scope else None
    message = {
        'event': 'psms_resource',
        'operation': operation,
        'status': status,
        'elapsed_ms': round(elapsed * 1000),
        'ram_mib': round(end.ram_mib, 1) if end.ram_mib is not None else None,
        'peak_ram_mib': round(end.peak_ram_mib, 1) if end.peak_ram_mib is not None else None,
        'cpu_seconds': round(cpu, 4) if cpu is not None else None,
        'cpu_pct_one_core': round(cpu / elapsed * 100, 1) if cpu is not None and elapsed else None,
        'scope': end.scope,
        'peak_source': end.peak_source,
    }
    try:
        print(json.dumps(message, separators=(',', ':')), flush=True)
    except (OSError, ValueError):
        pass


@contextmanager
def measure(operation):
    if not _enabled():
        yield
        return
    start = _snapshot()
    status = 'error'
    try:
        yield
        status = 'ok'
    finally:
        _log(operation, start, _snapshot(), status)


def _sample():
    previous = _snapshot()
    while True:
        sleep(SAMPLE_SECONDS)
        current = _snapshot()
        _log('interval', previous, current, 'ok')
        previous = current


def start_sampler():
    if _enabled():
        Thread(target=_sample, name='psms-resource-sampler', daemon=True).start()
