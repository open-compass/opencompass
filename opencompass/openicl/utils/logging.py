import logging

import torch.distributed as dist

LOG_LEVEL = logging.INFO
SUBPROCESS_LOG_LEVEL = logging.ERROR
LOG_FORMATTER = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'


def get_logger(name, level=LOG_LEVEL, log_file=None, file_mode='w'):
    formatter = logging.Formatter(LOG_FORMATTER)

    logger = logging.getLogger(name)

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    if rank == 0:
        logger.setLevel(level)
    else:
        logger.setLevel(SUBPROCESS_LOG_LEVEL)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    return logger
