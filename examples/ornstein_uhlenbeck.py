"""
Example: Ornstein-Uhlenbeck Process

Mean-reverting process used in interest rate models and pairs trading:
    dX = theta * (mu - X) * dt + sigma * dW

Properties:
- Mean reversion: X tends toward mu
- Speed: theta controls how fast
- Volatility: sigma controls randomness
"""
import numpy as np
from pyito import SDE, integrate
import time


def drift(t, y, args):
    """Mean reversion: theta * (mu - X)"""
    theta, mu, sigma = args
    return theta * (mu - y)


def diffusion(t, y, args):
    """Constant volatility"""
    theta, mu, sigma = args
    return sigma


def main():
    print("=" * 70)
    print("Ornstein-Uhlenbeck Process - Mean Reversion")
    print("=" * 70)
    
    # Parameters
    X0 = 5.0        # Initial value
    theta = 2.0     # Mean reversion speed
    mu = 0.0        # Long-term mean
    sigma = 1.0     # Volatility
    T = 5.0         # Time horizon
    dt = 0.01       # Time step
    n_paths = 10000
    
    print(f"\nParameters:")
    print(f"  Initial Value (X0): {X0:.2f}")
    print(f"  Reversion Speed (θ): {theta:.2f}")
    print(f"  Long-term Mean (μ): {mu:.2f}")
    print(f"  Volatility (σ): {sigma:.2f}")
    print(f"  Time Horizon: {T} years")
    print(f"  Number of Paths: {n_paths:,}")
    
    # Define SDE
    sde = SDE(drift, diffusion, args=(theta, mu, sigma))
    
    # Simulate with full path storage
    print(f"\nSimulating full paths...")
    start = time.time()
    
    result = integrate(
        sde,
        y0=X0,
        tspan=(0, T),
        dt=dt,
        method='euler_maruyama',
        n_paths=n_paths,
        output='all',  # Store all timesteps
        seed=42
    )
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f} seconds")
    
    # Analyze convergence to mean
    times = np.linspace(0, T, result.shape[0])
    mean_path = result[:, :, 0].mean(axis=1)  # Average across paths
    
    print(f"\nMean Reversion Analysis:")
    print(f"{'Time':<10} {'Mean X(t)':<15} {'Std Dev':<15} {'Distance to μ':<15}")
    print("-" * 60)
    
    for i, t in enumerate([0, 1, 2, 3, 4, 5]):
        idx = int(t / dt)
        if idx < len(mean_path):
            mean_val = mean_path[idx]
            std_val = result[idx, :, 0].std()
            distance = abs(mean_val - mu)
            print(f"{t:<10.1f} {mean_val:<15.4f} {std_val:<15.4f} {distance:<15.4f}")
    
    # Theoretical stationary distribution
    # X(∞) ~ N(mu, sigma^2 / (2*theta))
    stationary_var = sigma**2 / (2 * theta)
    stationary_std = np.sqrt(stationary_var)
    
    final_vals = result[-1, :, 0]
    
    print(f"\nStationary Distribution:")
    print(f"  Theoretical Mean: {mu:.4f}")
    print(f"  Empirical Mean: {final_vals.mean():.4f}")
    print(f"  Theoretical Std: {stationary_std:.4f}")
    print(f"  Empirical Std: {final_vals.std():.4f}")
    
    # Half-life calculation
    # Time for |X - mu| to reduce by 50%
    half_life = np.log(2) / theta
    print(f"\nDynamics:")
    print(f"  Half-life: {half_life:.3f} years")
    print(f"  (Time to halve distance from mean)")
    
    # Demonstrate memory efficiency
    print(f"\n" + "="*70)
    print("Memory Efficiency Demonstration")
    print("="*70)
    
    # Large simulation with final-only storage
    n_paths_large = 1_000_000
    print(f"\nSimulating {n_paths_large:,} paths (final values only)...")
    
    start = time.time()
    result_final = integrate(
        sde,
        y0=X0,
        tspan=(0, T),
        dt=dt,
        method='euler_maruyama',
        n_paths=n_paths_large,
        output='final',  # Memory efficient!
        seed=42
    )
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Speed: {n_paths_large/elapsed:,.0f} paths/second")
    print(f"Memory: Final result is only {result_final.nbytes/1024/1024:.1f} MB")
    print(f"  (vs ~{result.nbytes * n_paths_large / n_paths / 1024 / 1024 / 1024:.1f} GB for full paths)")
    
    final_mean = result_final[:, 0].mean()
    final_std = result_final[:, 0].std()
    
    print(f"\nFinal Statistics (1M paths):")
    print(f"  Mean: {final_mean:.4f} (theoretical: {mu:.4f})")
    print(f"  Std: {final_std:.4f} (theoretical: {stationary_std:.4f})")
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
