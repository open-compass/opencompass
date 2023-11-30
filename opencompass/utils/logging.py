import logging
import os

from mmengine.logging import MMLogger

_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}


def get_logger(log_level='INFO', filter_duplicate_level=None) -> MMLogger:
    """Get the logger for OpenCompass.

    Args:
        log_level (str): The log level. Default: 'INFO'. Choices are 'DEBUG',
            'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    if not MMLogger.check_instance_created('OpenCompass'):
        logger = MMLogger.get_instance('OpenCompass',
                                       logger_name='OpenCompass',
                                       log_level=log_level)
    else:
        logger = MMLogger.get_instance('OpenCompass')

    if filter_duplicate_level is None:
        # export OPENCOMPASS_FILTER_DUPLICATE_LEVEL=error
        # export OPENCOMPASS_FILTER_DUPLICATE_LEVEL=error,warning
        filter_duplicate_level = os.getenv(
            'OPENCOMPASS_FILTER_DUPLICATE_LEVEL', None)

    if filter_duplicate_level:
        logger.addFilter(
            FilterDuplicateMessage('OpenCompass', filter_duplicate_level))

    return logger


class FilterDuplicateMessage(logging.Filter):
    """Filter the repeated message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name, filter_duplicate_level):
        super().__init__(name)
        self.seen: set = set()

        if isinstance(filter_duplicate_level, str):
            filter_duplicate_level = filter_duplicate_level.split(',')

        self.filter_duplicate_level = []
        for level in filter_duplicate_level:
            _level = level.strip().upper()
            if _level not in _nameToLevel:
                raise ValueError(f'Invalid log level: {_level}')
            self.filter_duplicate_level.append(_nameToLevel[_level])

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the repeated error message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno not in self.filter_duplicate_level:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False
