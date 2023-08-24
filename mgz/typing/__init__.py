from .shape_typing import *
from .data_utils import *
from abc import abstractmethod, ABCMeta

def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider