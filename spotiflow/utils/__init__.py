from __future__ import absolute_import, print_function

from .utils import *
from .matching import *
from .peaks import *
from .parallel import tile_iterator

class NotRegisteredError(Exception):
    """Custom exception to be raised when a model or dataset is not registered.
    """
    pass
