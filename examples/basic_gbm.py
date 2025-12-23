"""
Example: Geometric Brownian Motion (Black-Scholes Model)

Simulates stock prices using:
    dS = mu * S * dt + sigma * S * dW

This is the foundation of option pricing in finance.
"""
import numpy as np
from pyito import SDE, integrate
import time


def drift(t, y, args):
    """Drift term: mu * S"""
    mu = args[0]
    return mu * y


def diffusion(t, y, args):
    """Diffusion term: sigma * S"""
    sigma = args[1]
    return sigma * y[0]


def main():
    print("=" * 70)
    print("Geometric Brownian Motion - Stock Price Simulation")
    print("=" * 70)
    
    # Parameters
    S0 = 100.0      # Initial stock price
    mu = 0.05       # Drift (5% annual return)
    sigma = 0.2     # Volatility (20% annual)
    T = 1.0         # Time horizon (1 year)
    dt = 0.001      # Time step
    n_paths = 100000  # Number of simulation paths
    
    print(f"\nParameters:")
    print(f"  Initial Price (S0): ${S0:.2f}")
    print(f"  Drift (μ): {mu*100:.1f}%")
    print(f"  Volatility (σ): {sigma*100:.1f}%")
    print(f"  Time Horizon: {T} year")
    print(f"  Time Step: {dt}")
    print(f"  Number of Paths: {n_paths:,}")
    
    # Define SDE
    sde = SDE(drift, diffusion, args=(mu, sigma))
    
    # Run simulation
    print(f"\nRunning simulation...")
    start = time.time()
    
    result = integrate(
        sde,
        y0=S0,
        tspan=(0, T),
        dt=dt,
        method='euler_maruyama',
        n_paths=n_paths,
        output='final',
        seed=42
    )
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Speed: {n_paths/elapsed:,.0f} paths/second")
    
    # Extract final prices
    final_prices = result[:, 0]
    
    # Theoretical values
    theoretical_mean = S0 * np.exp(mu * T)
    theoretical_std = S0 * np.exp(mu * T) * np.sqrt(np.exp(sigma**2 * T) - 1)
    
    # Empirical values
    empirical_mean = final_prices.mean()
    empirical_std = final_prices.std()
    
    print(f"\n{'Metric':<30} {'Theoretical':<15} {'Empirical':<15} {'Error %':<10}")
    print("-" * 70)
    print(f"{'Mean Final Price':<30} ${theoretical_mean:<14.2f} ${empirical_mean:<14.2f} {abs(empirical_mean-theoretical_mean)/theoretical_mean*100:<9.2f}%")
    print(f"{'Std Dev Final Price':<30} ${theoretical_std:<14.2f} ${empirical_std:<14.2f} {abs(empirical_std-theoretical_std)/theoretical_std*100:<9.2f}%")
    
    # Risk metrics
    percentiles = [5, 25, 50, 75, 95]
    print(f"\nPrice Distribution (Percentiles):")
    for p in percentiles:
        value = np.percentile(final_prices, p)
        print(f"  {p:2d}th percentile: ${value:7.2f}")
    
    # Value at Risk (VaR)
    var_95 = S0 - np.percentile(final_prices, 5)
    print(f"\nRisk Metrics:")
    print(f"  95% VaR (Value at Risk): ${var_95:.2f}")
    print(f"  Probability of loss: {(final_prices < S0).mean()*100:.2f}%")
    print(f"  Maximum drawdown: ${S0 - final_prices.min():.2f}")
    print(f"  Maximum gain: ${final_prices.max() - S0:.2f}")
    
    # Demonstrate reproducibility
    print(f"\n" + "="*70)
    print("Testing Reproducibility")
    print("="*70)
    
    result2 = integrate(
        sde, y0=S0, tspan=(0, T), dt=dt,
        method='euler_maruyama', n_paths=10,
        output='final', seed=999
    )
    
    result3 = integrate(
        sde, y0=S0, tspan=(0, T), dt=dt,
        method='euler_maruyama', n_paths=10,
        output='final', seed=999
    )
    
    if np.allclose(result2, result3):
        print("✓ Same seed produces identical results")
    else:
        print("✗ Reproducibility test failed")
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
