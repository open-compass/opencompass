import threading
from datetime import datetime
from pathlib import Path

from filelock import SoftFileLock


class HeartBeatManager:

    def __init__(self, work_dir: str | Path, fname: str = 'infer_heartbeat'):
        self.hb_file = Path(work_dir) / fname
        self._lock = SoftFileLock(Path(work_dir) / '.locks' / fname)

    def start_heartbeat(self, write_interval: float = 5.):
        Path(self.hb_file).parent.mkdir(exist_ok=True)
        stop_event = threading.Event()

        def _run():
            while not stop_event.is_set():
                try:
                    with self._lock:
                        self.hb_file.write_text(datetime.now().isoformat())
                except Exception:
                    pass
                stop_event.wait(write_interval)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return stop_event, thread

    def last_heartbeat(self) -> float:
        '''Return the seconds after the last heartbeat.'''
        if not self.hb_file.exists():
            return float('inf')
        try:
            with self._lock:
                content = self.hb_file.read_text()
            ts = datetime.fromisoformat(content)
        except Exception:
            ts = datetime.fromtimestamp(self.hb_file.lstat().st_mtime)

        interval = datetime.now() - ts

        return interval.total_seconds()
