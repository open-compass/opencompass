from mmengine.logging import MMLogger


def get_logger(log_level='INFO') -> MMLogger:
    """Get the logger for OpenCompass.

    Args:
        log_level (str): The log level. Default: 'INFO'. Choices are 'DEBUG',
            'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    return MMLogger.get_instance('OpenCompass',
                                 logger_name='OpenCompass',
                                 log_level=log_level)
