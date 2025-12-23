"""Core API for numba-sde."""
import numpy as np
from typing import Union, Tuple, Optional, Callable
from .types import Array, Method, OutputPolicy, NoiseType, DriftFunc, DiffusionFunc
from .utils import (
    validate_y0, validate_tspan, validate_dt, 
    infer_noise_type
)
from .compiler import JITFactory, dispatch_kernel


class SDE:
    def __init__(self, drift: DriftFunc, diffusion: DiffusionFunc, diffusion_deriv: Optional[Callable] = None, args: tuple = ()):
        JITFactory.validate_signature(drift, "drift")
        JITFactory.validate_signature(diffusion, "diffusion")
        if diffusion_deriv is not None:
            JITFactory.validate_signature(diffusion_deriv, "diffusion_deriv")
        
        self.drift = drift
        self.diffusion = diffusion
        self.diffusion_deriv = diffusion_deriv
        self.args = args
        
        self._compiled_drift = None
        self._compiled_diffusion = None
        self._compiled_diffusion_deriv = None
    
    def compile(self):
        if self._compiled_drift is None:
            self._compiled_drift = JITFactory.compile_drift(self.drift)
        if self._compiled_diffusion is None:
            self._compiled_diffusion = JITFactory.compile_diffusion(self.diffusion)
        if self.diffusion_deriv is not None and self._compiled_diffusion_deriv is None:
            self._compiled_diffusion_deriv = JITFactory.compile_diffusion_derivative(self.diffusion_deriv)
    
    def get_compiled_functions(self):
        self.compile()
        return (self._compiled_drift, self._compiled_diffusion, self._compiled_diffusion_deriv)


def integrate(
    sde: SDE, y0: Union[float, Array], tspan: Tuple[float, float], dt: float,
    method: str = 'euler_maruyama', n_paths: int = 1000, output: str = 'final',
    seed: Optional[int] = None, debug: bool = False, **kwargs
) -> Array:
    
    y0_arr, dims = validate_y0(y0)
    t0, t1 = validate_tspan(tspan)
    dt_validated, n_steps = validate_dt(dt, t0, t1)
    
    try:
        method_enum = Method(method)
    except ValueError:
        raise ValueError(f"Invalid method '{method}'. Choose from {[m.value for m in Method]}")
    
    try:
        output_enum = OutputPolicy(output)
    except ValueError:
        raise ValueError(f"Invalid output '{output}'. Choose from {[o.value for o in OutputPolicy]}")
    
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    
    # Infer noise type
    test_output = sde.diffusion(t0, y0_arr, sde.args)
    noise_type = infer_noise_type(test_output, dims)
    
    # --- DEBUG MODE ---
    if debug:
        return _integrate_debug(
            sde, y0_arr, t0, dt_validated, n_steps, n_paths,
            method_enum, output_enum, noise_type, seed
        )
    
    # --- COMPILATION & DISPATCH ---
    drift_compiled, diffusion_compiled, diffusion_deriv_compiled = sde.get_compiled_functions()
    
    # Safety Guards
    if method_enum == Method.MILSTEIN and noise_type == NoiseType.CORRELATED:
        raise NotImplementedError("Milstein method with Correlated noise is not yet supported.")

    # --- THE FIX IS HERE ---
    # We changed != SCALAR to == CORRELATED to allow DIAGONAL noise.
    if method_enum == Method.DF_MILSTEIN and noise_type == NoiseType.CORRELATED:
         raise NotImplementedError("Derivative-Free Milstein does not support Correlated noise yet.")
    
    if method_enum == Method.MILSTEIN and diffusion_deriv_compiled is None:
        raise ValueError("Milstein method requires diffusion_deriv. Use 'df_milstein' otherwise.")
    
    # Dispatch
    kernel = dispatch_kernel(method_enum.value, noise_type.value, output_enum.value)
    
    # Execute
    if method_enum == Method.MILSTEIN:
        result = kernel(
            drift_compiled, diffusion_compiled, diffusion_deriv_compiled, sde.args,
            y0_arr, t0, dt_validated, n_steps, n_paths, seed
        )
    elif method_enum == Method.DF_MILSTEIN:
        epsilon = kwargs.get('epsilon', 1e-5)
        result = kernel(
            drift_compiled, diffusion_compiled, sde.args,
            y0_arr, t0, dt_validated, n_steps, n_paths, seed, epsilon
        )
    else:  # Euler-Maruyama
        result = kernel(
            drift_compiled, diffusion_compiled, sde.args,
            y0_arr, t0, dt_validated, n_steps, n_paths, seed
        )
    
    return result


def _integrate_debug(sde, y0, t0, dt, n_steps, n_paths, method, output, noise_type, seed):
    dims = len(y0)
    if output == OutputPolicy.SAVE_ALL:
        result = np.zeros((n_steps + 1, n_paths, dims))
    else:
        result = np.zeros((n_paths, dims))
    
    for path_idx in range(n_paths):
        np.random.seed(seed + path_idx)
        y = y0.copy()
        t = t0
        
        if output == OutputPolicy.SAVE_ALL:
            result[0, path_idx, :] = y
        
        for step in range(n_steps):
            a = np.asarray(sde.drift(t, y, sde.args))
            b_val = sde.diffusion(t, y, sde.args)
            
            if noise_type == NoiseType.SCALAR:
                dW = np.random.normal(0.0, np.sqrt(dt))
                diffusion_term = b_val * dW
            elif noise_type == NoiseType.DIAGONAL:
                dW = np.random.normal(0.0, np.sqrt(dt), size=dims)
                diffusion_term = np.asarray(b_val) * dW
            elif noise_type == NoiseType.CORRELATED:
                B = np.asarray(b_val)
                dW_vec = np.random.normal(0.0, np.sqrt(dt), size=dims)
                diffusion_term = B @ dW_vec
            else:
                 raise ValueError(f"Unknown noise type: {noise_type}")
            
            y = y + a * dt + diffusion_term
            t += dt
            
            if output == OutputPolicy.SAVE_ALL:
                result[step + 1, path_idx, :] = y
        
        if output == OutputPolicy.SAVE_FINAL:
            result[path_idx, :] = y
            
    return result