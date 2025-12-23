"""Noise generation utilities for numba-sde."""
import numpy as np
from numba import njit


@njit(fastmath=True)
def generate_scalar_noise(dt: float) -> float:
    """Generate scalar Wiener increment dW ~ N(0, dt)."""
    return np.random.normal(0.0, np.sqrt(dt))


@njit(fastmath=True)
def generate_diagonal_noise(dt: float, dims: int, dW: np.ndarray) -> None:
    """
    Generate diagonal (independent) Wiener increments.
    
    Args:
        dt: Time step
        dims: Number of dimensions
        dW: Output array (modified in-place)
    """
    sqrt_dt = np.sqrt(dt)
    for i in range(dims):
        dW[i] = np.random.normal(0.0, sqrt_dt)


@njit(fastmath=True)
def generate_correlated_noise(dt: float, dims: int, L: np.ndarray, dW: np.ndarray) -> None:
    """
    Generate correlated Wiener increments using Cholesky decomposition.
    
    Args:
        dt: Time step
        dims: Number of dimensions
        L: Cholesky factor of covariance matrix (L @ L.T = Sigma)
        dW: Output array (modified in-place)
    """
    sqrt_dt = np.sqrt(dt)
    
    # Generate independent standard normal vector Z
    Z = np.empty(dims)
    for i in range(dims):
        Z[i] = np.random.normal(0.0, 1.0)
    
    # Apply Cholesky: dW = sqrt(dt) * L @ Z
    for i in range(dims):
        dW[i] = 0.0
        for j in range(dims):
            dW[i] += L[i, j] * Z[j]
        dW[i] *= sqrt_dt
