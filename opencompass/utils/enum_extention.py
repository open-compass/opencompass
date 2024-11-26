# Copied from the source code of `enum.StrEnum` to support Python<3.11
from enum import Enum


class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings."""

    def __new__(cls, *values):
        """values must already be of type `str`"""
        if len(values) > 3:
            raise TypeError('too many arguments for str(): %r' % (values, ))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError('%r is not a string' % (values[0], ))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError('encoding must be a string, not %r' %
                                (values[1], ))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError('errors must be a string, not %r' %
                                (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    def _generate_next_value_(name, start, count, last_values):
        """Return the lower-cased version of the member name."""
        return name.lower()
