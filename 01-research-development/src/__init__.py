"""
Price Matrix Research & Development Package

This package contains all the research and development components
for the Price Matrix financial pricing system.
"""

__version__ = "1.0.0"
__author__ = "Price Matrix Development Team"

# Import main modules for easy access
from . import data
from . import models
from . import pricing
from . import utils

__all__ = [
    'data',
    'models',
    'pricing',
    'utils'
]