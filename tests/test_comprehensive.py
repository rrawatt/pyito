import numpy as np
import pytest
from numba import njit
from pyito import SDE, integrate, Method, OutputPolicy

# ============================================================================
# 1. TEST SETUP: PHYSICS DEFINITIONS
# ============================================================================

# --- CONSTANTS ---
MU = 0.05
SIGMA = 0.2
X0 = 100.0
T = 1.0
DT = 0.005  
PATHS = 50_000 
SEED = 42

# --- SCALAR GBM FUNCTIONS ---
@njit(fastmath=True)
def scalar_drift(t, x, args):
    return args[0] * x

@njit(fastmath=True)
def scalar_diff(t, x, args):
    return args[1] * x

@njit(fastmath=True)
def scalar_diff_deriv(t, x, args):
    return np.full_like(x, args[1])

# --- DIAGONAL GBM FUNCTIONS (2D Independent) ---
@njit(fastmath=True)
def diag_drift(t, x, args):
    return args[0] * x

@njit(fastmath=True)
def diag_diff(t, x, args):
    return args[1] * x

@njit(fastmath=True)
def diag_diff_deriv(t, x, args):
    return np.full_like(x, args[1])

# --- CORRELATED GBM FUNCTIONS (2D) ---
CORR = 0.8
L_MAT = np.array([[1.0, 0.0], [0.8, 0.6]])

@njit(fastmath=True)
def corr_drift(t, x, args):
    return args[0] * x

@njit(fastmath=True)
def corr_diff(t, x, args):
    sigma = args[1]
    res = np.empty((2, 2))
    res[0, 0] = sigma * x[0] * L_MAT[0, 0]
    res[0, 1] = sigma * x[0] * L_MAT[0, 1]
    res[1, 0] = sigma * x[1] * L_MAT[1, 0]
    res[1, 1] = sigma * x[1] * L_MAT[1, 1]
    return res

# ============================================================================
# 2. HELPER: STATISTICAL VALIDATION
# ============================================================================

def validate_gbm_stats(results, dims=1, label="Test"):
    """Checks if simulation matches analytical GBM moments."""
    # Analytical Expectations
    exp_mean = X0 * np.exp(MU * T)
    exp_std = np.sqrt(X0**2 * np.exp(2*MU*T) * (np.exp(SIGMA**2 * T) - 1))
    
    if results.ndim == 3: 
        final_vals = results[-1]
    else:
        final_vals = results

    # Check Mean
    sim_mean = np.mean(final_vals)
    err_mean = np.abs(sim_mean - exp_mean)
    
    # Check Std Dev
    sim_std = np.std(final_vals)
    err_std = np.abs(sim_std - exp_std)

    print(f"[{label}] Mean Err: {err_mean:.4f}, Std Err: {err_std:.4f}")
    
    # Tolerances (Monte Carlo error is roughly 1/sqrt(N))
    assert err_mean < 1.0, f"{label} Mean diverged significantly!"
    assert err_std < 1.0, f"{label} Volatility diverged significantly!"


# ============================================================================
# 3. TEST CASES
# ============================================================================

def test_scalar_kernels():
    """Test all methods for Scalar noise."""
    print("\n--- SCALAR TESTS ---")
    sde = SDE(scalar_drift, scalar_diff, scalar_diff_deriv, args=(MU, SIGMA))
    
    # 1. Euler
    res = integrate(sde, X0, (0, T), DT, method='euler_maruyama', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, label="Scalar Euler")
    
    # 2. Milstein
    res = integrate(sde, X0, (0, T), DT, method='milstein', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, label="Scalar Milstein")
    
    # 3. DF Milstein
    res = integrate(sde, X0, (0, T), DT, method='df_milstein', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, label="Scalar DF-Milstein")
    
    # 4. Check OutputPolicy.SAVE_ALL
    res_all = integrate(sde, X0, (0, T), DT, output='all', n_paths=100)
    assert res_all.shape[0] > 1, "SAVE_ALL failed to return time steps"
    assert res_all.shape[1] == 100, "SAVE_ALL path count mismatch"

def test_diagonal_kernels():
    """Test all methods for Diagonal (Vector) noise."""
    print("\n--- DIAGONAL TESTS ---")
    sde = SDE(diag_drift, diag_diff, diag_diff_deriv, args=(MU, SIGMA))
    y0_vec = np.array([X0, X0])
    
    # 1. Euler
    res = integrate(sde, y0_vec, (0, T), DT, method='euler_maruyama', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, dims=2, label="Diagonal Euler")
    
    # 2. Milstein
    res = integrate(sde, y0_vec, (0, T), DT, method='milstein', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, dims=2, label="Diagonal Milstein")

    # 3. DF Milstein (The newly fixed kernel!)
    res = integrate(sde, y0_vec, (0, T), DT, method='df_milstein', n_paths=PATHS, seed=SEED)
    validate_gbm_stats(res, dims=2, label="Diagonal DF-Milstein")

def test_correlated_kernels():
    """Test Euler for Correlated noise and verify correlation."""
    print("\n--- CORRELATED TESTS ---")
    sde = SDE(corr_drift, corr_diff, args=(MU, SIGMA))
    y0_vec = np.array([X0, X0])
    
    # 1. Euler 
    res = integrate(sde, y0_vec, (0, T), DT, method='euler_maruyama', n_paths=PATHS, seed=SEED)
    
    # Validate Moments
    validate_gbm_stats(res, dims=2, label="Correlated Euler")
    
    # Validate Correlation
    log_returns = np.log(res / X0)
    sim_corr = np.corrcoef(log_returns[:, 0], log_returns[:, 1])[0, 1]
    
    print(f"[Correlated Check] Target Rho: {CORR}, Sim Rho: {sim_corr:.4f}")
    assert np.abs(sim_corr - CORR) < 0.05, "Correlation structure failed!"

def test_safety_guards():
    """Ensure invalid combinations raise errors."""
    print("\n--- SAFETY TESTS ---")
    
    # 1. Milstein + Correlated (Should Fail)
    sde_corr = SDE(corr_drift, corr_diff, args=(MU, SIGMA))
    y0 = np.array([100., 100.])
    
    with pytest.raises(NotImplementedError) as exc:
        integrate(sde_corr, y0, (0, 1), 0.1, method='milstein')
    assert "Correlated" in str(exc.value)
    
    # 2. DF Milstein + Correlated (Should Fail)
    with pytest.raises(NotImplementedError) as exc:
        integrate(sde_corr, y0, (0, 1), 0.1, method='df_milstein')
    assert "Correlated" in str(exc.value)

    print("[PASS] Safety guards active.")

def test_reproducibility():
    """Ensure seed guarantees identical results."""
    print("\n--- REPRODUCIBILITY TESTS ---")
    sde = SDE(scalar_drift, scalar_diff, args=(MU, SIGMA))
    
    # Run 1
    res1 = integrate(sde, X0, (0, 1), 0.1, n_paths=100, seed=999)
    # Run 2
    res2 = integrate(sde, X0, (0, 1), 0.1, n_paths=100, seed=999)
    # Run 3 (Different Seed)
    res3 = integrate(sde, X0, (0, 1), 0.1, n_paths=100, seed=1000)
    
    assert np.allclose(res1, res2), "Seeding failed! Results differ for same seed."
    assert not np.allclose(res1, res3), "Seeding failed! Results identical for diff seed."
    print("[PASS] Results are reproducible.")

if __name__ == "__main__":
    test_scalar_kernels()
    test_diagonal_kernels()
    test_correlated_kernels()
    test_safety_guards()
    test_reproducibility()
    print("\n\n>>> ALL COMPREHENSIVE TESTS PASSED <<<")