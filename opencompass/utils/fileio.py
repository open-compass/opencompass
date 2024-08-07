import gzip
import hashlib
import io
import os
import os.path
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager

import mmengine.fileio as fileio
from mmengine.fileio import LocalBackend, get_file_backend


def patch_func(module, fn_name_to_wrap):
    backup = getattr(patch_func, '_backup', [])
    fn_to_wrap = getattr(module, fn_name_to_wrap)

    def wrap(fn_new):
        setattr(module, fn_name_to_wrap, fn_new)
        backup.append((module, fn_name_to_wrap, fn_to_wrap))
        setattr(fn_new, '_fallback', fn_to_wrap)
        setattr(patch_func, '_backup', backup)
        return fn_new

    return wrap


@contextmanager
def patch_fileio(global_vars=None):
    if getattr(patch_fileio, '_patched', False):
        # Only patch once, avoid error caused by patch nestly.
        yield
        return
    import builtins

    @patch_func(builtins, 'open')
    def open(file, mode='r', *args, **kwargs):
        backend = get_file_backend(file)
        if isinstance(backend, LocalBackend):
            return open._fallback(file, mode, *args, **kwargs)
        if 'b' in mode:
            return io.BytesIO(backend.get(file, *args, **kwargs))
        else:
            return io.StringIO(backend.get_text(file, *args, **kwargs))

    if global_vars is not None and 'open' in global_vars:
        bak_open = global_vars['open']
        global_vars['open'] = builtins.open

    import os

    @patch_func(os.path, 'join')
    def join(a, *paths):
        backend = get_file_backend(a)
        if isinstance(backend, LocalBackend):
            return join._fallback(a, *paths)
        paths = [item for item in paths if len(item) > 0]
        return backend.join_path(a, *paths)

    @patch_func(os.path, 'isdir')
    def isdir(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return isdir._fallback(path)
        return backend.isdir(path)

    @patch_func(os.path, 'isfile')
    def isfile(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return isfile._fallback(path)
        return backend.isfile(path)

    @patch_func(os.path, 'exists')
    def exists(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return exists._fallback(path)
        return backend.exists(path)

    @patch_func(os, 'listdir')
    def listdir(path):
        backend = get_file_backend(path)
        if isinstance(backend, LocalBackend):
            return listdir._fallback(path)
        return backend.list_dir_or_file(path)

    import filecmp

    @patch_func(filecmp, 'cmp')
    def cmp(f1, f2, *args, **kwargs):
        with fileio.get_local_path(f1) as f1, fileio.get_local_path(f2) as f2:
            return cmp._fallback(f1, f2, *args, **kwargs)

    import shutil

    @patch_func(shutil, 'copy')
    def copy(src, dst, **kwargs):
        backend = get_file_backend(src)
        if isinstance(backend, LocalBackend):
            return copy._fallback(src, dst, **kwargs)
        return backend.copyfile_to_local(str(src), str(dst))

    import torch

    @patch_func(torch, 'load')
    def load(f, *args, **kwargs):
        if isinstance(f, str):
            f = io.BytesIO(fileio.get(f))
        return load._fallback(f, *args, **kwargs)

    try:
        setattr(patch_fileio, '_patched', True)
        yield
    finally:
        for patched_fn in patch_func._backup:
            (module, fn_name_to_wrap, fn_to_wrap) = patched_fn
            setattr(module, fn_name_to_wrap, fn_to_wrap)
        if global_vars is not None and 'open' in global_vars:
            global_vars['open'] = bak_open
        setattr(patch_fileio, '_patched', False)


def patch_hf_auto_model(cache_dir=None):
    if hasattr('patch_hf_auto_model', '_patched'):
        return

    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    ori_model_pt = PreTrainedModel.from_pretrained

    @classmethod
    def model_pt(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs['cache_dir'] = cache_dir
        if not isinstance(get_file_backend(pretrained_model_name_or_path),
                          LocalBackend):
            kwargs['local_files_only'] = True
        if cache_dir is not None and not isinstance(
                get_file_backend(cache_dir), LocalBackend):
            kwargs['local_files_only'] = True

        with patch_fileio():
            res = ori_model_pt.__func__(cls, pretrained_model_name_or_path,
                                        *args, **kwargs)
        return res

    PreTrainedModel.from_pretrained = model_pt

    # transformers copied the `from_pretrained` to all subclasses,
    # so we have to modify all classes
    for auto_class in [
            _BaseAutoModelClass, *_BaseAutoModelClass.__subclasses__()
    ]:
        ori_auto_pt = auto_class.from_pretrained

        @classmethod
        def auto_pt(cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs['cache_dir'] = cache_dir
            if not isinstance(get_file_backend(pretrained_model_name_or_path),
                              LocalBackend):
                kwargs['local_files_only'] = True
            if cache_dir is not None and not isinstance(
                    get_file_backend(cache_dir), LocalBackend):
                kwargs['local_files_only'] = True

            with patch_fileio():
                res = ori_auto_pt.__func__(cls, pretrained_model_name_or_path,
                                           *args, **kwargs)
            return res

        auto_class.from_pretrained = auto_pt

    patch_hf_auto_model._patched = True


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024):
    md5 = hashlib.md5()
    backend = get_file_backend(fpath, enable_singleton=True)
    if isinstance(backend, LocalBackend):
        # Enable chunk update for local file.
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
    else:
        md5.update(backend.get(fpath))
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path.

    Modified from
    https://pytorch.org/docs/stable/hub.html#torch.hub.download_url_to_file

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved,
            e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded
            file should start with ``hash_prefix``. Defaults to None.
        progress (bool): whether or not to display a progress bar to stderr.
            Defaults to True
    """
    file_size = None
    req = urllib.request.Request(url)
    u = urllib.request.urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders('Content-Length')
    else:
        content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after download is
    # complete. This prevents a local file being overridden by a broken
    # download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    import rich.progress
    columns = [
        rich.progress.DownloadColumn(),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.TimeRemainingColumn(),
    ]
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with rich.progress.Progress(*columns) as pbar:
            task = pbar.add_task('download', total=file_size, visible=progress)
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(task, advance=len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from.
        root (str): Directory to place downloaded file in.
        filename (str | None): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f'Using downloaded and verified file: {fpath}')
    else:
        try:
            print(f'Downloading {url} to {fpath}')
            download_url_to_file(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      f' Downloading {url} to {fpath}')
                download_url_to_file(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')


def _is_tarxz(filename):
    return filename.endswith('.tar.xz')


def _is_tar(filename):
    return filename.endswith('.tar')


def _is_targz(filename):
    return filename.endswith('.tar.gz')


def _is_tgz(filename):
    return filename.endswith('.tgz')


def _is_gzip(filename):
    return filename.endswith('.gz') and not filename.endswith('.tar.gz')


def _is_zip(filename):
    return filename.endswith('.zip')


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url,
                                 download_root,
                                 extract_root=None,
                                 filename=None,
                                 md5=None,
                                 remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f'Extracting {archive} to {extract_root}')
    extract_archive(archive, extract_root, remove_finished)
