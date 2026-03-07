import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TypedDict

from filelock import SoftFileLock

from .abbr import get_infer_output_path


class InferStatus(TypedDict):
    status: str
    total: int | None
    completed: int | None


def safe_read(file: Path, work_dir: Path):
    sig = '--'.join(file.resolve().relative_to(work_dir.resolve()).parts)
    with SoftFileLock(work_dir / '.locks' / sig):
        content = file.read_text()
    return content


def safe_write(file: Path, content: str, work_dir: Path):
    sig = '--'.join(file.resolve().relative_to(work_dir.resolve()).parts)
    with SoftFileLock(work_dir / '.locks' / sig):
        file.write_text(content)


class InferStatusManager:

    def __init__(
        self,
        work_dir: str | Path,
        model_cfg: dict,
        dataset_cfg: dict,
    ) -> None:
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.work_dir = Path(work_dir)
        self.infer_status = dict(status='pending', completed=0, total=None)
        self.status_path = Path(
            get_infer_output_path(
                model_cfg,
                dataset_cfg,
                str(Path(work_dir) / 'infer_status'),
                'json',
            ))
        self.status_path.parent.mkdir(exist_ok=True, parents=True)
        self.safe_read = partial(safe_read, work_dir=self.work_dir)
        self.safe_write = partial(safe_write, work_dir=self.work_dir)

    def update(
        self,
        status: str | None = None,
        total: int | None = None,
        completed: int | None = None,
    ):
        new_status = self.infer_status.copy()
        if status is not None:
            new_status['status'] = status
        if total is not None:
            new_status['total'] = total
        if completed is not None:
            new_status['completed'] = completed
        self._maybe_write(new_status)

    def _maybe_write(self, entry: dict) -> None:
        if self.infer_status != entry:
            self.infer_status = entry
            self.write_task_status()

    def write_task_status(self):
        payload = self.infer_status.copy()
        payload['updated_at'] = datetime.now().isoformat()
        self.safe_write(self.status_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def get_task_status(self) -> dict[str, InferStatus]:
        stem = self.status_path.stem
        suffix = self.status_path.suffix
        if self.status_path.exists():
            return {stem: json.loads(self.safe_read(self.status_path))}
        # Child format: xxxx_0.json, xxxx_1.json
        children = [
            child for child in self.status_path.parent.glob(f'{stem}*{suffix}')
            if child.stem[len(stem) + 1:].isdecimal()
        ]
        return {
            child.stem: json.loads(self.safe_read(child))
            for child in children
        }
