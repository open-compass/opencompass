def is_supa_available() -> bool:
    try:
        import torch
        import torch_supa
    except ImportError:
        return False

    supa: Any = getattr(torch, 'supa', None)
    if supa is None:
        return False

    is_available = getattr(supa, 'is_available', None)
    if callable(is_available):
        try:
            return bool(is_available())
        except (RuntimeError, OSError):
            return False

    device_count = getattr(supa, 'device_count', None)
    if callable(device_count):
        try:
            return device_count() > 0
        except (RuntimeError, OSError):
            return False

    return False
