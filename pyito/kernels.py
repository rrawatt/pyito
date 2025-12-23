"""JIT-compiled solver kernels for numba-sde."""
import numpy as np
from numba import njit, prange

# ============================================================================
# EULER-MARUYAMA KERNELS
# ============================================================================

@njit(parallel=True, fastmath=True)
def euler_maruyama_scalar_final(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args) # Scalar output
            dW = np.random.normal(0.0, sqrt_dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def euler_maruyama_scalar_all(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            dW = np.random.normal(0.0, sqrt_dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW
            t += dt
            result[step + 1, path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def euler_maruyama_diagonal_final(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args) # Vector output (dims,)
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                y[i] += a[i] * dt + b[i] * dW
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def euler_maruyama_diagonal_all(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                y[i] += a[i] * dt + b[i] * dW
            t += dt
            result[step + 1, path_idx] = y
    return result


@njit(parallel=True, fastmath=True)
def euler_maruyama_correlated_final(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        # Pre-allocate noise vector to avoid allocation inside loop
        dW_vec = np.empty(dims) 
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            B = diffusion_func(t, y, args) # Matrix output (dims, dims)
            
            # Generate independent Brownian motions
            for i in range(dims):
                dW_vec[i] = np.random.normal(0.0, sqrt_dt)
                
            # y = y + a*dt + B @ dW
            # Manual matrix multiplication for performance/numba clarity
            for i in range(dims):
                diffusion_term = 0.0
                for j in range(dims):
                    diffusion_term += B[i, j] * dW_vec[j]
                y[i] += a[i] * dt + diffusion_term
                
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def euler_maruyama_correlated_all(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        result[0, path_idx] = y
        dW_vec = np.empty(dims)
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            B = diffusion_func(t, y, args)
            
            for i in range(dims):
                dW_vec[i] = np.random.normal(0.0, sqrt_dt)
                
            for i in range(dims):
                diffusion_term = 0.0
                for j in range(dims):
                    diffusion_term += B[i, j] * dW_vec[j]
                y[i] += a[i] * dt + diffusion_term
                
            t += dt
            result[step + 1, path_idx] = y
    return result

# ============================================================================
# MILSTEIN KERNELS 
# ============================================================================

@njit(parallel=True, fastmath=True)
def milstein_scalar_final(drift_func, diffusion_func, diffusion_deriv_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            b_prime = diffusion_deriv_func(t, y, args)
            
            dW = np.random.normal(0.0, sqrt_dt)
            # FIX: correction is a scalar here, no [i] indexing needed
            correction = 0.5 * b * b_prime * (dW**2 - dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW + correction
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def milstein_scalar_all(drift_func, diffusion_func, diffusion_deriv_func, args, y0, t0, dt, n_steps, n_paths, seed):

    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            b_prime = diffusion_deriv_func(t, y, args)
            
            dW = np.random.normal(0.0, sqrt_dt)
            correction = 0.5 * b * b_prime * (dW**2 - dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW + correction
            t += dt
            result[step + 1, path_idx] = y
    return result


@njit(parallel=True, fastmath=True)
def milstein_diagonal_final(drift_func, diffusion_func, diffusion_deriv_func, args, y0, t0, dt, n_steps, n_paths, seed):
    """
    Milstein method for Diagonal noise (independent dW per dimension).
    """
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)       # Vector (dims,)
            bx = diffusion_deriv_func(t, y, args) # Vector (dims,) - Jacobian diagonal
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                # Element-wise Milstein Correction
                # correction = 0.5 * b_i * (db_i/dx_i) * (dW_i^2 - dt)
                correction = 0.5 * b[i] * bx[i] * (dW**2 - dt)
                y[i] += a[i] * dt + b[i] * dW + correction
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def milstein_diagonal_all(drift_func, diffusion_func, diffusion_deriv_func, args, y0, t0, dt, n_steps, n_paths, seed):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            bx = diffusion_deriv_func(t, y, args)
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                correction = 0.5 * b[i] * bx[i] * (dW**2 - dt)
                y[i] += a[i] * dt + b[i] * dW + correction
            t += dt
            result[step + 1, path_idx] = y
    return result


# ============================================================================
# DF MILSTEIN KERNELS
# ============================================================================

@njit(parallel=True, fastmath=True)
def df_milstein_scalar_final(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed, epsilon=1e-5):
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    inv_2eps = 1.0 / (2.0 * epsilon)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        y_plus = np.empty(dims)
        y_minus = np.empty(dims)
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            
            # Scalar Approximation: Perturb all dims because b(y) is scalar
            for i in range(dims):
                y_plus[i] = y[i] + epsilon
                y_minus[i] = y[i] - epsilon
                
            b_plus = diffusion_func(t, y_plus, args)
            b_minus = diffusion_func(t, y_minus, args)
            b_prime = (b_plus - b_minus) * inv_2eps
            
            dW = np.random.normal(0.0, sqrt_dt)
            correction = 0.5 * b * b_prime * (dW**2 - dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW + correction
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def df_milstein_scalar_all(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed, epsilon=1e-5):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    inv_2eps = 1.0 / (2.0 * epsilon)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        y_plus = np.empty(dims)
        y_minus = np.empty(dims)
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            
            for i in range(dims):
                y_plus[i] = y[i] + epsilon
                y_minus[i] = y[i] - epsilon
                
            b_plus = diffusion_func(t, y_plus, args)
            b_minus = diffusion_func(t, y_minus, args)
            b_prime = (b_plus - b_minus) * inv_2eps
            
            dW = np.random.normal(0.0, sqrt_dt)
            correction = 0.5 * b * b_prime * (dW**2 - dt)
            
            for i in range(dims):
                y[i] += a[i] * dt + b * dW + correction
            t += dt
            result[step + 1, path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def df_milstein_diagonal_final(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed, epsilon=1e-5):
    """
    Derivative-Free Milstein for Diagonal noise.
    """
    dims = len(y0)
    result = np.empty((n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    inv_2eps = 1.0 / (2.0 * epsilon)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        y_plus = np.empty(dims)
        y_minus = np.empty(dims)
        t = t0
        
        for _ in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args) # Vector
            
            # Approximate diagonal derivative
            for i in range(dims):
                y_plus[i] = y[i] + epsilon
                y_minus[i] = y[i] - epsilon
                
            b_plus = diffusion_func(t, y_plus, args)
            b_minus = diffusion_func(t, y_minus, args)
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                
                # Finite diff approx for db_i/dx_i (diagonal element)
                b_prime_i = (b_plus[i] - b_minus[i]) * inv_2eps
                
                # Milstein correction
                correction = 0.5 * b[i] * b_prime_i * (dW**2 - dt)
                
                y[i] += a[i] * dt + b[i] * dW + correction
            
            t += dt
            
        result[path_idx] = y
    return result

@njit(parallel=True, fastmath=True)
def df_milstein_diagonal_all(drift_func, diffusion_func, args, y0, t0, dt, n_steps, n_paths, seed, epsilon=1e-5):
    dims = len(y0)
    result = np.empty((n_steps + 1, n_paths, dims))
    sqrt_dt = np.sqrt(dt)
    inv_2eps = 1.0 / (2.0 * epsilon)
    
    for path_idx in prange(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        y_plus = np.empty(dims)
        y_minus = np.empty(dims)
        t = t0
        result[0, path_idx] = y
        
        for step in range(n_steps):
            a = drift_func(t, y, args)
            b = diffusion_func(t, y, args)
            
            for i in range(dims):
                y_plus[i] = y[i] + epsilon
                y_minus[i] = y[i] - epsilon
                
            b_plus = diffusion_func(t, y_plus, args)
            b_minus = diffusion_func(t, y_minus, args)
            
            for i in range(dims):
                dW = np.random.normal(0.0, sqrt_dt)
                b_prime_i = (b_plus[i] - b_minus[i]) * inv_2eps
                correction = 0.5 * b[i] * b_prime_i * (dW**2 - dt)
                y[i] += a[i] * dt + b[i] * dW + correction
                
            t += dt
            result[step + 1, path_idx] = y
    return result