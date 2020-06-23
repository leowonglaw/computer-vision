from typing import Any
import functools


def rgetattr(obj: Any, attr: str, default=None):
    ''' Enhanced getattr, allows to get an attribute recurively/nested
        Example: rgetattr(obj, "attr.attr_in_attr")
    '''
    _getattr = lambda obj, attr: getattr(obj, attr, default)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj: Any, attr: str, value: Any):
    ''' Enhanced setattr, allows to set an attribute in any level
        Example: setattr(obj, "attr.attr_in_attr", "setting value")
    '''
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, value)
