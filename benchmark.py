import os

os.environ["NUMBA_NUM_THREADS"] = "12"

import time
import numpy as np
from numba import njit, config
from pyito import SDE, integrate

print(f"Numba is using {config.NUMBA_NUM_THREADS} threads.")


# --- Setup ---
PATHS = 1_000_000
DT = 0.001
T_END = 1
STEPS = int(T_END / DT) 

@njit
def f(t, x, args): return -x
@njit
def g(t, x, args): return 1.0

sde = SDE(f, g)

print(f"--- BENCHMARK: {PATHS:,} Paths, {STEPS} Steps (dt={DT}) ---")

# 1. Warmup
print("Warmup (JIT Compilation)...")
integrate(sde, 1.0, (0, 1), 0.1, n_paths=100)

# 2. PyIto Implementation
print(f"Running PyIto...")
start = time.time()
integrate(sde, 1.0, (0, T_END), DT, n_paths=PATHS)
numba_time = time.time() - start
print(f"PyIto Time:   {numba_time:.4f}s")

# 3. Numpy Implementation
print(f"Running NumPy...")
start = time.time()
x = np.ones(PATHS)
sq_dt = np.sqrt(DT)

for _ in range(STEPS):
    x += -x * DT + 1.0 * np.random.normal(0, sq_dt, PATHS)
numpy_time = time.time() - start
print(f"NumPy Time:   {numpy_time:.4f}s")

# 4. Result
print(f"\n>>> Speedup Factor: {numpy_time / numba_time:.1f}x FASTER <<<")