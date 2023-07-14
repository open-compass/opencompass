import asyncio
import concurrent.futures
import hashlib
import io
import os
import pickle
import socket
import threading
from asyncio import InvalidStateError
from asyncio.tasks import ALL_COMPLETED
from datetime import datetime
from typing import Awaitable, Callable, Dict, List

import torch
from internlm.utils.common import SingletonMeta

MB = 1024**2


def compute_file_md5_by_chunk(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_inmemory_md5_by_chunk(obj_name):
    hash_md5 = hashlib.md5()
    with io.BytesIO() as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class ThreadLock:

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, a, b, c):
        self.lock.release()


class StorageClient:

    def __init__(self) -> None:
        """Abstract base class for storage clients, which can be 'boto3'
        clients, 'petrel' clients, or file system clients."""
        pass

    def load(self, fp):
        raise NotImplementedError

    def async_upload_fileobj(self, local_nvme_path, fp):
        raise NotImplementedError

    def gen_and_upload_md5(self, local_nvme_path, fp):
        raise NotImplementedError

    def sync_upload_fileobj(self, fp, states):
        raise NotImplementedError

    def check_folder(self, folder):
        raise NotImplementedError

    def get_fns(self, folder):
        raise NotImplementedError


class LocalClient(StorageClient):

    def __init__(self) -> None:
        self.client = None

    def async_upload_fileobj(self, local_nvme_path, fp):
        self.gen_and_upload_md5(local_nvme_path, fp)
        os.system(f'cp {local_nvme_path} {fp}')

    def gen_and_upload_md5(self, local_nvme_path, fp):
        md5 = compute_file_md5_by_chunk(local_nvme_path)
        buket_name, file_name = os.path.split(fp)
        md5_folder = os.path.join(buket_name, 'md5')
        md5_fp = os.path.join(md5_folder, file_name + '.md5')
        try:
            torch.save(md5, md5_fp)
        except Exception:
            os.makedirs(md5_folder, exist_ok=True)
            torch.save(md5, md5_fp)

    def sync_upload_fileobj(self, fp, states):
        torch.save(states, fp)

    def load(self, fp):
        assert os.path.exists(
            fp), f'{fp} is not found in {socket.gethostname()}'
        with open(fp, 'rb') as f:
            states = torch.load(f)
        return states

    def check_folder(self, folder):
        assert os.path.exists(folder), folder

    def get_fns(self, folder):
        assert os.path.exists(folder)
        fns = os.listdir(folder)
        return fns


class PrintLogger:

    def info(*args, **kwargs):
        print(*args, **kwargs)

    def error(*args, **kwargs):
        print(*args, **kwargs)


def get_mount_point_free_size(path: str):
    if os.path.exists(path):
        st = os.statvfs(path)
        # f_bavail: Number of free blocks for unprivileged users.
        # f_bsize: Filesystem block size.
        # return unit is TB.
        return st.f_bavail * st.f_bsize / (1024**4)
    else:
        raise FileNotFoundError(path)


def check_tmp_folder_accessibility(tmp_local_folder: str):
    ret = True
    ret &= os.access(tmp_local_folder, os.W_OK)
    ret &= os.access(tmp_local_folder, os.R_OK)
    if ret is False:
        error_str = f"tmp_local_folder:\"{tmp_local_folder}\", dose not have read and write permissions in {socket.gethostname()}"
        raise RuntimeError(error_str)


class StorageManager(metaclass=SingletonMeta):

    def __init__(self,
                 logger=None,
                 tmp_folder=None,
                 async_mode=True,
                 multipart=False,
                 n_async_workers=16) -> None:
        """StorageManager.

        Args:
            async_mode (bool, optional): _description_. Defaults to True.
            multipart (bool, optional): _description_. Defaults to False.
            n_async_workers (int, optional): _description_. Defaults to 16.

        Raises:
            RuntimeError: _description_
        """
        self.async_mode = async_mode
        self._async_stack = []
        self._async_loop = asyncio.new_event_loop() if async_mode else None
        self.n_async_workers = n_async_workers
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=n_async_workers) if async_mode else None
        self._step_counter = 0
        self._exception_list = []
        self.tmp_local_folder = tmp_folder
        self._to_be_del_files = []
        self._multipart = multipart

        self._lock_task_peeding = ThreadLock(
        )  # Used to synchronize the monitoring thread and the main thread.
        self._task_peeding = False  # Mark whether there is a peeding asynchronous task.

        self.logger = logger if logger is not None else PrintLogger()

        if self.async_mode:
            if self.tmp_local_folder:
                check_tmp_folder_accessibility(
                    os.path.dirname(self.tmp_local_folder))

                # Try to create tmp folder
                try:
                    os.makedirs(self.tmp_local_folder, exist_ok=True)
                except FileExistsError:
                    pass

                # In case it is a directory created by other users, we check the permissions again.
                check_tmp_folder_accessibility(self.tmp_local_folder)

                # Try to clean tmp folder's empty folder.
                self.try_delete_tmpfile(self.tmp_local_folder)

                # Available storage space check.
                free_size = get_mount_point_free_size(self.tmp_local_folder)
                if free_size < 0.5:
                    self.logger.error(
                        f"tmp_local_folder only have \"{free_size}\" TB free space, less then 1TB!"
                    )
                    raise RuntimeError(
                        f'Insufficient temporary storage space on {socket.gethostname()}'
                    )
            else:
                self.logger.info("tmp_local_folder didn't set! only load ")

        self.net_cli = {'boto3': None, 'Petrel': None}
        self._wait_lock = threading.Lock()

    def set_logger(self, logger):
        self.logger = logger

    def try_delete_tmpfile(self, path: str):
        for f in os.listdir(path):
            if f.split('.')[-1] == 'tmpfile':
                try:
                    os.remove(f)
                except BaseException:
                    # There is a race on multi-process delete, we don't throw any exceptions.
                    pass
                else:
                    self.logger.info(f'Delete tmpfile: {f}, in folder:{path}')

    def _get_local_cli(self):
        return LocalClient()

    def _get_client(self, fp: str):
        return self._get_local_cli()

    def check_folder(self, folder) -> None:
        self._get_client(folder).check_folder(folder)

    def get_fns(self, folder) -> List[str]:
        return self._get_client(folder).get_fns(folder)

    def _del_tmp_folder(self):
        for fp in self._to_be_del_files:
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass
            except Exception as e:
                self.logger.error(
                    f"delete file: {fp}, failed for reason:\"{e}\"")
            else:
                self.logger.info(f'del {fp}')

    def _get_tmp_file_name(self, fp: str):
        base_path = os.path.join(self.tmp_local_folder, fp.split('/')[-1])
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        pid = os.getpid()
        step = self._step_counter
        return '-'.join([base_path, current_time,
                         str(pid),
                         str(step)]) + '.tmpfile'

    def save(self, step: int, fp: str, states):
        if not self.tmp_local_folder:
            e_str = 'StorageManager is not set tmp_local_folder, so save cannot be performed.'
            self.logger.error(e_str)
            raise RuntimeError(e_str)

        if self.async_mode:
            if step != -1:  # 如果 step 为 -1 表示我们在进行模拟上传
                if self._step_counter != step:
                    self.wait()
                    self._exception_list.clear()
                    self._step_counter = step

            tmp_step_file = self._get_tmp_file_name(fp)
            self.logger.info(f'Create tmpfile: {tmp_step_file}, fp is {fp}')
            self._to_be_del_files.append(tmp_step_file)

            with open(tmp_step_file, 'wb') as f:
                torch.save(states, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            self.async_executor(
                self._get_client(fp).async_upload_fileobj, tmp_step_file, fp)
        else:
            self._get_client(fp).sync_upload_fileobj(fp, states)

    def load(self, fp: str) -> Dict:
        if self.wait():
            return self._get_client(fp).load(fp)
        else:
            raise RuntimeError(
                "Async checkpoint saving is failed! can't load checkpoint")

    async def _sync_tasks(self) -> Awaitable[None]:
        if self._async_stack:
            await asyncio.wait(self._async_stack, return_when=ALL_COMPLETED)
            count = 0
            while self._async_stack:
                t = self._async_stack.pop(0)
                try:
                    e = t.exception()
                    if e:
                        self._exception_list.append((e, count))
                        self.logger.error(
                            f'File:{self._to_be_del_files[count]}, upload failed for {e}'
                        )
                        raise e
                    count += 1
                except InvalidStateError:
                    # Not finished. https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.exception
                    pass

    def async_executor(self, fn: Callable, *args, **kwargs) -> None:
        """
        Overview:
            Execute task in background, then append the future instance in _async_stack.
        Arguments:
            - fn (:obj:`Callable`): Synchronization function.
        """
        if not self._async_loop:
            raise Exception(
                'Event loop was not initialized, please call this function in async or parallel mode'
            )
        t = self._async_loop.run_in_executor(self._thread_pool, fn, *args,
                                             **kwargs)
        self._async_stack.append(t)

    def wait(self) -> bool:
        # If there is no task that is peeding, return directly
        if self._task_peeding is False:
            return True

        with self._lock_task_peeding:
            # Double-check, if someone completes the wait, it will return directly.
            if self._task_peeding is False:
                return True

            if self._async_loop:
                self._async_loop.run_until_complete(self._sync_tasks())

            self._task_peeding = False

            if len(self._exception_list) != 0:
                for e in self._exception_list:
                    file_id, error_msg = e[1], e[0]
                    self.logger.error(
                        "Node:{}, Error: Checkpoint save to OSS failed on the {}th step for file\"{}\", error:\"{}\""
                        .format(socket.gethostname(), self._step_counter,
                                self._to_be_del_files[file_id], error_msg))

                    # self._get_client(file_name).sync_upload_fileobj()
                    raise RuntimeError(
                        f"{socket.gethostname()}, Step:{self._step_counter} upload file:\"{self._to_be_del_files[file_id]}\" failed: \"{error_msg}\""
                    )

            self._del_tmp_folder()  # clean all tmp file.
            self._exception_list = []
            self._to_be_del_files = []
            self.logger.info(
                f'Step:{self._step_counter} all asynchronous uploads succeeded!'
            )
        return True

    def __del__(self):
        if self.async_mode:
            self.wait()
            self._del_tmp_folder()
            # try_clean_empty_folder(self.tmp_local_folder)

    def set_finish_flag(self, step: int):
        """Setting _task_peeding to True means that all asynchronous upload
        operations in the current step are queued.

        Args:
            step (int): identifier, but don't use it for now
        """
        self._task_peeding = True


def init_storage_manager(*args, **kwargs):
    # We don't need to worry about the reference count dropping to 0,
    # since we'll be adding it to a static dict.
    StorageManager(*args, **kwargs)


def get_storage_manager() -> StorageManager:
    """only for StorageManager which has been inited."""
    return StorageManager()
