"""Type definitions and enums for numba-sde."""
from enum import Enum
from typing import Callable, Tuple, Union
import numpy as np

# Type aliases
Array = np.ndarray
Scalar = Union[float, int]
DriftFunc = Callable[[float, Union[float, Array], tuple], Union[float, Array]]
DiffusionFunc = Callable[[float, Union[float, Array], tuple], Union[float, Array]]


class Method(Enum):
    """Solver methods."""
    EULER_MARUYAMA = 'euler_maruyama'
    MILSTEIN = 'milstein'
    DF_MILSTEIN = 'df_milstein'  # Derivative-Free Milstein


class OutputPolicy(Enum):
    """Memory management strategies."""
    SAVE_ALL = 'all'        # (Steps, Paths, Dims)
    SAVE_FINAL = 'final'    # (Paths, Dims)
    # SAVE_CHECKPOINTS = 'checkpoints'  # Future implementation


class NoiseType(Enum):
    """Noise correlation structure."""
    SCALAR = 'scalar'
    DIAGONAL = 'diagonal'
    CORRELATED = 'correlated'