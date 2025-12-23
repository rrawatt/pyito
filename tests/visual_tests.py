import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pyito import SDE, integrate

# ==============================================================================
# TEST 1: Ornstein-Uhlenbeck (Mean Reverting)
# dX = theta * (mu - X) * dt + sigma * dW
# ==============================================================================
print("--- Simulating Ornstein-Uhlenbeck Process ---")

# Parameters
THETA = 0.7
MU = 1.5
SIGMA = 0.3
X0 = 0.0  # Start away from mean to show reversion
T_SPAN = (0.0, 5.0)
DT = 0.01
PATHS = 100

@njit(fastmath=True)
def ou_drift(t, x, args):
    theta, mu, sigma = args
    return theta * (mu - x)

@njit(fastmath=True)
def ou_diffusion(t, x, args):
    theta, mu, sigma = args
    # Scalar noise: returns float or array of same shape as x
    return np.full_like(x, sigma) 

# Setup & Run
ou_sde = SDE(ou_drift, ou_diffusion, args=(THETA, MU, SIGMA))
# Use 'all' output policy to get full paths for plotting
ou_results = integrate(ou_sde, X0, T_SPAN, DT, n_paths=PATHS, output='all', seed=42)

# Plot
t_grid = np.linspace(T_SPAN[0], T_SPAN[1], ou_results.shape[0])
plt.figure(figsize=(10, 4))
plt.plot(t_grid, ou_results[:, :10, 0], alpha=0.3, lw=1) # Plot first 10 paths
plt.plot(t_grid, np.mean(ou_results[:, :, 0], axis=1), 'k--', lw=2, label="Simulated Mean")
plt.axhline(MU, color='r', linestyle=':', lw=2, label="Long-Term Mean (mu)")
plt.title("Test 1: Ornstein-Uhlenbeck (Mean Reversion)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ==============================================================================
# TEST 2: Cox-Ingersoll-Ross (Square Root Diffusion)
# dX = a * (b - X) * dt + sigma * sqrt(X) * dW
# ==============================================================================
print("--- Simulating Cox-Ingersoll-Ross Process ---")

A = 3.0
B = 0.05
SIGMA_CIR = 0.1
R0 = 0.05

@njit(fastmath=True)
def cir_drift(t, x, args):
    a, b, sigma = args
    return a * (b - x)

@njit(fastmath=True)
def cir_diffusion(t, x, args):
    a, b, sigma = args
    # Safe sqrt: np.maximum ensures no NaN if x dips slightly negative due to discretization
    return sigma * np.sqrt(np.maximum(x, 0.0))

cir_sde = SDE(cir_drift, cir_diffusion, args=(A, B, SIGMA_CIR))
cir_results = integrate(cir_sde, R0, T_SPAN, DT, n_paths=PATHS, output='all', seed=42)

plt.figure(figsize=(10, 4))
plt.plot(t_grid, cir_results[:, :10, 0], alpha=0.3, lw=1)
plt.plot(t_grid, np.mean(cir_results[:, :, 0], axis=1), 'b-', lw=2, label="Simulated Mean")
plt.axhline(B, color='r', linestyle=':', label="Long-Term Mean (b)")
plt.title("Test 2: Cox-Ingersoll-Ross (Interest Rate Model)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ==============================================================================
# TEST 3: Heston Stochastic Volatility (Correlated Noise)
# dS = mu * S * dt + sqrt(v) * S * dW_s
# dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
# Corr(dW_s, dW_v) = rho
# ==============================================================================
print("--- Simulating Heston Model ---")

MU_H = 0.05       # Drift of Asset
KAPPA = 2.0       # Mean reversion speed of vol
THETA_H = 0.04    # Long term variance
XI = 0.1          # Volatility of volatility
RHO = -0.7        # Correlation (Leverage effect)

# Initial State: [Price=100, Variance=0.04]
Y0 = np.array([100.0, 0.04]) 

# Cholesky Decomposition for Correlation Matrix [[1, rho], [rho, 1]]
# L = [[1, 0], [rho, sqrt(1-rho^2)]]
L_row1 = np.array([RHO, np.sqrt(1 - RHO**2)])

@njit(fastmath=True)
def heston_drift(t, y, args):
    # y[0] = Price (S), y[1] = Variance (v)
    mu, kappa, theta, xi = args
    
    ds = mu * y[0]
    dv = kappa * (theta - y[1])
    return np.array([ds, dv])

@njit(fastmath=True)
def heston_diffusion(t, y, args):
    # Must return Matrix B such that diffusion term = B @ dW_uncorrelated
    mu, kappa, theta, xi = args
    
    S = y[0]
    v = np.maximum(y[1], 0.0) # Ensure variance is non-negative
    sqrt_v = np.sqrt(v)
    
    # We construct the correlated diffusion matrix manually
    # Row 1 (Price): sqrt(v) * S * [1, 0]  <-- dW_s is pure noise 1
    # Row 2 (Vol):   xi * sqrt(v) * [rho, sqrt(1-rho^2)] <-- dW_v is mixed
    
    B = np.zeros((2, 2))
    
    # dS term
    B[0, 0] = sqrt_v * S
    B[0, 1] = 0.0
    
    # dv term (Correlated)
    B[1, 0] = xi * sqrt_v * L_row1[0] # xi*sqrt(v)*rho
    B[1, 1] = xi * sqrt_v * L_row1[1] # xi*sqrt(v)*sqrt(1-rho^2)
    
    return B

heston_sde = SDE(heston_drift, heston_diffusion, args=(MU_H, KAPPA, THETA_H, XI))
heston_results = integrate(heston_sde, Y0, (0.0, 1.0), DT, n_paths=5, output='all', seed=42)

# Plot Heston
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Price Paths
ax1.plot(np.linspace(0, 1, heston_results.shape[0]), heston_results[:, :, 0])
ax1.set_title("Test 3a: Heston Asset Prices (Spot)")
ax1.set_ylabel("Price ($S_t$)")
ax1.grid(True, alpha=0.3)

# Volatility Paths
ax2.plot(np.linspace(0, 1, heston_results.shape[0]), heston_results[:, :, 1], color='orange')
ax2.set_title("Test 3b: Heston Variance Processes (Vol)")
ax2.set_ylabel("Variance ($v_t$)")
ax2.set_xlabel("Time (Years)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()