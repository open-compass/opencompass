"""The CI cache must allow locks/writes without changing shared datasets."""

import importlib.util
import os
from pathlib import Path

import pytest
from filelock import FileLock

_script = (Path(__file__).resolve().parents[2] / '.github' / 'scripts' /
           'prepare_hf_cache.py')
_spec = importlib.util.spec_from_file_location('prepare_hf_cache', _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_writable_cache_preserves_source(tmp_path):
    source = tmp_path / 'shared'
    cached = source / 'dataset' / 'default' / '1.0' / 'hash'
    cached.mkdir(parents=True)
    (cached / 'data.arrow').write_bytes(b'cached payload')
    (cached / 'dataset_info.json').write_text('{}')
    (source / 'old.lock').write_text('shared lock')
    incomplete = source / 'pending.incomplete'
    incomplete.mkdir()
    (incomplete / 'partial.arrow').write_bytes(b'incomplete')
    os.utime(cached, (1000, 1000))
    os.utime(cached.parent, (2000, 2000))

    target = tmp_path / 'local'
    _module.prepare_cache(source, target)
    local = target / cached.relative_to(source)
    assert (local / 'data.arrow').is_symlink()
    assert (local / 'data.arrow').read_bytes() == b'cached payload'
    assert not (target / 'old.lock').exists()
    assert not (target / 'pending.incomplete').exists()
    assert local.stat().st_mtime == 1000
    assert local.parent.stat().st_mtime == 2000
    with FileLock(str(local) + '_builder.lock'):
        (local / 'dataset_info.json').write_text('{"updated": true}')
        (local / 'cache-new.arrow').write_bytes(b'new map output')
    assert (cached / 'dataset_info.json').read_text() == '{}'
    assert not (cached / 'cache-new.arrow').exists()
    assert (source / 'old.lock').read_text() == 'shared lock'


def test_cache_rejects_existing_or_nested_destination(tmp_path):
    source = tmp_path / 'shared'
    source.mkdir()
    with pytest.raises(ValueError, match='must be new'):
        _module.prepare_cache(source, source)
    with pytest.raises(ValueError, match='outside'):
        _module.prepare_cache(source, source / 'local')
