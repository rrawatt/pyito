import numpy as np
from numba import njit
from pyito import SDE, integrate

CORR = 0.8
COV = np.array([[1.0, CORR], [CORR, 1.0]])

@njit
def drift_zero(t, y, args):
    return np.zeros_like(y)

@njit
def diffusion_matrix(t, y, args):
    return np.array([[1.0, 0.0], [0.8, 0.6]])

def test_correlation_structure():
    print("\n--- Testing Correlated Noise Structure ---")
    
    sde = SDE(drift_zero, diffusion_matrix, args=())
    
    dt = 1.0
    n_paths = 100_000
    
    results = integrate(
        sde, y0=[0.0, 0.0], tspan=(0, dt), dt=dt, 
        method='euler_maruyama', n_paths=n_paths, seed=123
    )
    
    sim_corr_matrix = np.corrcoef(results[:, 0], results[:, 1])
    sim_rho = sim_corr_matrix[0, 1]
    
    print(f"  > Target Rho: {CORR}")
    print(f"  > Sim Rho:    {sim_rho:.4f}")
    
    assert np.abs(sim_rho - CORR) < 0.01, "Correlation structure failed!"
    print("  [PASS] Correlation matrix is correct.")

if __name__ == "__main__":
    test_correlation_structure()