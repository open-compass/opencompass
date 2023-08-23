import re

from importlib_metadata import PackageNotFoundError, distribution
from mmengine.utils import digit_version


def satisfy_requirement(dep):
    pat = '(' + '|'.join(['>=', '==', '>']) + ')'
    parts = re.split(pat, dep, maxsplit=1)
    parts = [p.strip() for p in parts]
    package = parts[0]
    if len(parts) > 1:
        op, version = parts[1:]
        op = {
            '>=': '__ge__',
            '==': '__eq__',
            '>': '__gt__',
            '<': '__lt__',
            '<=': '__le__'
        }[op]
    else:
        op, version = None, None

    try:
        dist = distribution(package)
        if op is None or getattr(digit_version(dist.version), op)(
                digit_version(version)):
            return True
    except PackageNotFoundError:
        pass

    return False
