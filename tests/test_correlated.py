import numpy as np
from numba import njit
from pyito import SDE, integrate

# --- Define Correlated Brownian Motion ---
# Drift = 0, Diffusion = Cholesky(Sigma)
# This is basically just testing the noise generator
CORR = 0.8
COV = np.array([[1.0, CORR], [CORR, 1.0]])

@njit
def drift_zero(t, y, args):
    return np.zeros_like(y)

@njit
def diffusion_matrix(t, y, args):
    # We must return the 'B' matrix such that B @ B.T = Covariance
    # For testing, we pre-calculate Cholesky here or pass it in.
    # Let's emulate a "constant correlation" model.
    # Lower Cholesky of [[1, 0.8], [0.8, 1]] is [[1, 0], [0.8, 0.6]]
    return np.array([[1.0, 0.0], [0.8, 0.6]])

def test_correlation_structure():
    print("\n--- Testing Correlated Noise Structure ---")
    
    sde = SDE(drift_zero, diffusion_matrix, args=())
    
    # Run simulation
    # We simulate 1 step to check the noise correlation directly
    dt = 1.0
    n_paths = 100_000
    
    results = integrate(
        sde, y0=[0.0, 0.0], tspan=(0, dt), dt=dt, 
        method='euler_maruyama', n_paths=n_paths, seed=123
    )
    
    # Calculate correlation of the results
    # Since drift is 0 and dt=1, Result ~ N(0, Sigma)
    sim_corr_matrix = np.corrcoef(results[:, 0], results[:, 1])
    sim_rho = sim_corr_matrix[0, 1]
    
    print(f"  > Target Rho: {CORR}")
    print(f"  > Sim Rho:    {sim_rho:.4f}")
    
    assert np.abs(sim_rho - CORR) < 0.01, "Correlation structure failed!"
    print("  [PASS] Correlation matrix is correct.")

if __name__ == "__main__":
    test_correlation_structure()