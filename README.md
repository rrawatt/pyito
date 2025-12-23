# PyIto

**High-performance SDE solver with JIT compilation and parallel execution.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyIto** is a Python library for simulating Stochastic Differential Equations (SDEs). Built on the **Numba** compiler, it translates Python model definitions into optimized machine code (LLVM) and automatically parallelizes Monte Carlo simulations across all available CPU cores.

It is designed for quantitative finance and research applications where **execution speed** and **memory efficiency** are critical.

---

## ✨ Key Features

* **🚀 JIT Compilation:** Compiles drift and diffusion functions to machine code for C-like performance.
* **🔥 Auto-Parallelism:** Automatically distributes paths across CPU cores using `numba.prange`.
* **📉 Zero-RAM Noise Generation:** Generates Brownian motion on-the-fly, allowing for millions of paths without memory overhead.
* **🧠 Derivative-Free Milstein:** Achieves Strong Order 1.0 accuracy without requiring manual calculation of diffusion derivatives.
* **📊 Flexible Memory Management:** Choose to store full path histories or only terminal states to optimize RAM usage.

---

## 🛠️ Installation

```bash
pip install pyito

```

Requirements: `numpy`, `numba*`

---

## ⚡ Quick Start

Here is a minimal example simulating a mean-reverting **Ornstein-Uhlenbeck** process:
$$ dX_t = \theta(\mu - X_t)dt + \sigma dW_t $$

```python
import numpy as np
from numba import njit
from pyito import SDE, integrate

# 1. Define Physics (Must be JIT-compatible)
@njit
def drift(t, x, args):
    theta, mu, sigma = args
    return theta * (mu - x)

@njit
def diffusion(t, x, args):
    theta, mu, sigma = args
    return sigma  # Constant volatility

# 2. Setup Model
# Params: theta=0.7, mu=1.5, sigma=0.3
sde = SDE(drift, diffusion, args=(0.7, 1.5, 0.3))

# 3. Run Simulation
# Simulates 100,000 paths in parallel
# Returns only the final state (at t=1.0) to save memory
results = integrate(sde, y0=0.0, tspan=(0, 1.0), dt=0.01, n_paths=100_000)

print(f"Mean: {np.mean(results):.4f}")
print(f"Std:  {np.std(results):.4f}")

```

---

## 🧮 Supported Algorithms

PyIto supports the following solvers via the `method` argument:

| Method | Strong Order | Description |
| --- | --- | --- |
| `euler_maruyama` | 0.5 | The standard, efficient default solver. |
| `milstein` | 1.0 | Higher accuracy for multiplicative noise. Requires providing `diffusion_deriv`. |
| `df_milstein` | 1.0 | **Derivative-Free Milstein.** Approximates derivatives numerically. Recommended for high accuracy without the math overhead. |

---

## 📚 Advanced Usage

### 1. Derivative-Free Milstein

To get higher accuracy without calculating  manually, use the derivative-free solver. It uses a finite-difference approximation inside the kernel.

```python
results = integrate(sde, ..., method='df_milstein')

```

### 2. Correlated Noise (Multi-Dimensional)

PyIto natively handles correlated noise for systems like the Heston model. If your diffusion function returns a **matrix**, the solver infers a correlated structure.

* **Scalar Output:** Single-factor noise.
* **Vector Output:** Independent diagonal noise.
* **Matrix Output:** Correlated noise (Diffusion Matrix  such that ).

### 3. Memory Optimization

For large-scale Monte Carlo (e.g., 10 million paths), storing the full history is impossible. PyIto allows you to control output verbosity:

```python
# Returns array of shape (Steps, Paths, Dims) - RAM Heavy
history = integrate(..., output='all')

# Returns array of shape (Paths, Dims) - RAM Efficient
terminal_values = integrate(..., output='final')

```

---

## ⚠️ Performance Note

PyIto is optimized to saturate your CPU.

* **Laptop Users:** Ensure your OS power plan is set to **"Best Performance"**. In "Power Saver" mode, operating systems often park high-performance cores, significantly throttling the parallel speedup.
* **Warm-up:** The first call to `integrate` will take an extra 1-2 seconds while Numba compiles your functions. Subsequent runs will be instant.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create your feature branch.
3. Install dev dependencies: `pip install -e .[dev]`
4. Run tests: `pytest tests/`

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

