from __future__ import absolute_import, print_function

from .fitting import estimate_params
from .matching import *
from .parallel import tile_iterator
from .peaks import *
from .utils import *


class NotRegisteredError(Exception):
    """Custom exception to be raised when a model or dataset is not registered.
    """
    pass
