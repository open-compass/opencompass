import os
import re

import timm.models.hub as timm_hub
import torch
import torch.distributed as dist
from mmengine.dist import is_distributed, is_main_process
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def download_cached_file(url, check_hash=True, progress=False):
    """Download a file from a URL and cache it locally.

    If the file already exists, it is not downloaded again. If distributed,
    only the main process downloads the file, and the other processes wait for
    the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_distributed():
        dist.barrier()

    return get_cached_file_path()


def is_url(input_url):
    """Check if an input string is a url.

    look for http(s):// and ignoring the case
    """
    is_url = re.match(r'^(?:http)s?://', input_url, re.IGNORECASE) is not None
    return is_url
