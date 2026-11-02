"""Reuse a read-only HF dataset cache with local locks and intermediates."""

import argparse
import os
import shutil
from pathlib import Path


def prepare_cache(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if not source.is_dir():
        raise ValueError(f'Not a dataset cache directory: {source}')
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Cache destination must be new: {destination}')
    if source == destination or source in destination.resolve().parents:
        raise ValueError('Writable cache must be outside the shared cache')

    destination.mkdir(parents=True)
    count = 0
    directory_times = []
    metadata = {'dataset_info.json', 'dataset_infos.json', 'state.json'}
    for root, dirs, files in os.walk(source, followlinks=True):
        dirs[:] = [name for name in dirs if not name.endswith('.incomplete')]
        target = destination / Path(root).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(('.lock', '.incomplete')):
                continue
            original = Path(root) / name
            if not original.is_file():
                continue
            if name in metadata:
                shutil.copyfile(original, target / name)
            else:
                (target / name).symlink_to(original)
            count += 1
        stat = os.stat(root)
        directory_times.append((target, stat.st_atime_ns, stat.st_mtime_ns))
    for target, atime, mtime in reversed(directory_times):
        os.utime(target, ns=(atime, mtime))
    print(f'Prepared writable HF cache: {destination} ({count} cached files)',
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    prepare_cache(args.source, args.destination)
