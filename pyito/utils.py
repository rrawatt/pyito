"""Utility functions for numba-sde."""
import numpy as np
from typing import Union, Tuple
from .types import Array, NoiseType


def validate_y0(y0: Union[float, Array]) -> Tuple[Array, int]:
    """
    Validate and normalize initial condition.
    
    Returns:
        (y0_normalized, dimensions)
    """
    if np.isscalar(y0):
        return np.array([y0], dtype=np.float64), 1
    
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim != 1:
        raise ValueError(f"y0 must be 1D array or scalar, got shape {y0.shape}")
    
    return y0, len(y0)


def validate_tspan(tspan: Tuple[float, float]) -> Tuple[float, float]:
    """Validate time span."""
    if len(tspan) != 2:
        raise ValueError(f"tspan must be (t0, t1), got {tspan}")
    
    t0, t1 = float(tspan[0]), float(tspan[1])
    
    if t1 <= t0:
        raise ValueError(f"t1 must be > t0, got t0={t0}, t1={t1}")
    
    return t0, t1


def validate_dt(dt: float, t0: float, t1: float) -> Tuple[float, int]:
    """
    Validate time step and compute number of steps.
    
    Returns:
        (dt, n_steps)
    """
    dt = float(dt)
    
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    if dt > (t1 - t0):
        raise ValueError(f"dt={dt} is too large for interval [{t0}, {t1}]")
    
    n_steps = int(np.ceil((t1 - t0) / dt))
    
    return dt, n_steps


def infer_noise_type(diffusion_output: Union[float, Array], dims: int) -> NoiseType:
    """
    Infer noise structure from diffusion output.
    
    Args:
        diffusion_output: Sample output from diffusion function
        dims: System dimensions
    """
    if np.isscalar(diffusion_output):
        return NoiseType.SCALAR
    
    output_arr = np.asarray(diffusion_output)
    
    if output_arr.ndim == 1 and len(output_arr) == dims:
        return NoiseType.DIAGONAL
    
    if output_arr.ndim == 2 and output_arr.shape == (dims, dims):
        return NoiseType.CORRELATED
    
    raise ValueError(
        f"Unexpected diffusion output shape {output_arr.shape} for {dims}D system. "
        f"Expected: scalar, ({dims},), or ({dims}, {dims})"
    )
