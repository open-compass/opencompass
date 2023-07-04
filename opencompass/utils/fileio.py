import io
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
