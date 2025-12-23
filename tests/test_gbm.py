import numpy as np
import time
import pytest
from numba import njit
from pyito import SDE, integrate, Method

MU = 0.05
SIGMA = 0.2
X0 = 100.0
T = 1.0
DT = 0.001
PATHS = 50_000  

@njit(fastmath=True)
def drift(t, x, args):
    return args[0] * x

@njit(fastmath=True)
def diffusion(t, x, args):
    return args[1] * x

@njit(fastmath=True)
def diffusion_deriv(t, x, args):
    return np.array([args[1]])

gbm_sde = SDE(
    drift, diffusion, diffusion_deriv, 
    args=(MU, SIGMA)
)

def check_stats(results, tolerance_mean=0.5, tolerance_std=0.5):
    """Compare simulation stats vs Analytical Truth"""
    expected_mean = X0 * np.exp(MU * T)
    expected_std = np.sqrt(X0**2 * np.exp(2*MU*T) * (np.exp(SIGMA**2 * T) - 1))
    
    sim_mean = np.mean(results)
    sim_std = np.std(results)
    
    print(f"  > Expected Mean: {expected_mean:.4f} | Sim Mean: {sim_mean:.4f}")
    print(f"  > Expected Std:  {expected_std:.4f} | Sim Std:  {sim_std:.4f}")
    
    assert np.abs(sim_mean - expected_mean) < tolerance_mean, "Mean diverged!"
    assert np.abs(sim_std - expected_std) < tolerance_std, "Volatility diverged!"
    print("  [PASS] Statistics match theory.\n")

def test_euler_maruyama():
    print("\n--- Testing Euler-Maruyama ---")
    res = integrate(gbm_sde, X0, (0, T), DT, method='euler_maruyama', n_paths=PATHS, seed=42)
    check_stats(res)

def test_milstein():
    print("--- Testing Milstein (Analytical Deriv) ---")
    res = integrate(gbm_sde, X0, (0, T), DT, method='milstein', n_paths=PATHS, seed=42)
    check_stats(res)

def test_df_milstein():
    print("--- Testing Derivative-Free Milstein ---")
    res = integrate(gbm_sde, X0, (0, T), DT, method='df_milstein', n_paths=PATHS, seed=42)
    check_stats(res)

if __name__ == "__main__":
    test_euler_maruyama()
    test_milstein()
    test_df_milstein()