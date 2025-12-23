"""
pyito: Fast, memory-efficient SDE solver with JIT compilation.

A high-performance Python library for simulating stochastic differential equations
using Numba's JIT compilation. Designed to outperform sdeint with parallel execution,
online memory reduction, and strict reproducibility.
"""

__version__ = "0.1.0"
__author__ = "RR"

from .core import SDE, integrate
from .types import Method, OutputPolicy, NoiseType

__all__ = [
    'SDE',
    'integrate',
    'Method',
    'OutputPolicy',
    'NoiseType',
]
