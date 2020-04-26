# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

from .preprocess import reader

__all__ = ['reader']
