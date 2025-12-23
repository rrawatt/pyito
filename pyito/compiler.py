"""JIT compilation middleware for numba-sde."""
from numba import njit
from typing import Callable, Optional
import inspect


class JITFactory:
    """
    The JIT Factory - Inspects and compiles user functions.
    """
    
    @staticmethod
    def is_numba_compiled(func: Callable) -> bool:
        """Check if function is already JIT-compiled."""
        return hasattr(func, 'py_func') or hasattr(func, '_numba_type_')
    
    @staticmethod
    def compile_drift(drift_func: Callable) -> Callable:
        """
        Compile drift function if it's pure Python.
        
        Args:
            drift_func: User's drift function a(t, y, args)
        
        Returns:
            JIT-compiled version
        """
        if JITFactory.is_numba_compiled(drift_func):
            return drift_func
        
        # Compile with nopython=True and fastmath for performance
        return njit(fastmath=True)(drift_func)
    
    @staticmethod
    def compile_diffusion(diffusion_func: Callable) -> Callable:
        """
        Compile diffusion function if it's pure Python.
        
        Args:
            diffusion_func: User's diffusion function b(t, y, args)
        
        Returns:
            JIT-compiled version
        """
        if JITFactory.is_numba_compiled(diffusion_func):
            return diffusion_func
        
        return njit(fastmath=True)(diffusion_func)
    
    @staticmethod
    def compile_diffusion_derivative(
        diffusion_deriv_func: Optional[Callable]
    ) -> Optional[Callable]:
        """
        Compile diffusion derivative function if provided.
        
        Args:
            diffusion_deriv_func: User's derivative function b'(t, y, args)
        
        Returns:
            JIT-compiled version or None
        """
        if diffusion_deriv_func is None:
            return None
        
        if JITFactory.is_numba_compiled(diffusion_deriv_func):
            return diffusion_deriv_func
        
        return njit(fastmath=True)(diffusion_deriv_func)
    
    @staticmethod
    def validate_signature(func: Callable, name: str) -> None:
        """
        Validate that function has correct signature: (t, y, args) -> ...
        
        Args:
            func: Function to validate
            name: Function name for error messages
        """
        try:
            # Try to get signature (may fail for compiled functions)
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            if len(params) != 3:
                raise ValueError(
                    f"{name} must have signature (t, y, args), got {len(params)} parameters"
                )
        except (ValueError, TypeError):
            # If we can't inspect (e.g., already compiled), trust the user
            pass


def dispatch_kernel(method: str, noise_type: str, output_policy: str):
    """
    Dispatch to appropriate kernel based on method, noise type, and output.
    
    Args:
        method: 'euler_maruyama', 'milstein', or 'df_milstein'
        noise_type: 'scalar', 'diagonal', or 'correlated'
        output_policy: 'final' or 'all'
    
    Returns:
        Appropriate kernel function
    """
    from . import kernels
    
    kernel_name = f"{method}_{noise_type}_{output_policy}"
    
    try:
        kernel = getattr(kernels, kernel_name)
        return kernel
    except AttributeError:
        raise NotImplementedError(
            f"Kernel '{kernel_name}' not implemented. "
            f"Method: {method}, Noise: {noise_type}, Output: {output_policy}"
        )
