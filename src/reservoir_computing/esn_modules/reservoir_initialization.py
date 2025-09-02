"""
🏗️ Reservoir Initialization Mixin - Advanced ESN Weight Setup
==============================================================

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Herbert Jaeger (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"

This module provides advanced reservoir initialization strategies for Echo State Networks.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class ReservoirInitializationMixin:
    """
    Mixin class for advanced reservoir initialization strategies.
    
    This class provides methods for initializing reservoir weights using various
    strategies optimized for different tasks and data characteristics.
    """
    
    def initialize_reservoir(
        self,
        n_reservoir: int,
        spectral_radius: float = 0.95,
        density: float = 0.1,
        random_seed: Optional[int] = None,
        initialization_type: str = 'uniform'
    ) -> np.ndarray:
        """
        Initialize reservoir weight matrix.
        
        Args:
            n_reservoir: Number of reservoir units
            spectral_radius: Desired spectral radius
            density: Connection density (0-1)
            random_seed: Random seed for reproducibility
            initialization_type: Type of initialization ('uniform', 'normal', 'sparse')
            
        Returns:
            Initialized reservoir weight matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if initialization_type == 'uniform':
            W = np.random.uniform(-1, 1, (n_reservoir, n_reservoir))
        elif initialization_type == 'normal':
            W = np.random.normal(0, 1, (n_reservoir, n_reservoir))
        elif initialization_type == 'sparse':
            W = np.random.normal(0, 1, (n_reservoir, n_reservoir))
            # Make it sparse
            mask = np.random.random((n_reservoir, n_reservoir)) < density
            W = W * mask
        else:
            raise ValueError(f"Unknown initialization type: {initialization_type}")
        
        # Apply sparsity
        if initialization_type != 'sparse':
            mask = np.random.random((n_reservoir, n_reservoir)) < density
            W = W * mask
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        
        if current_spectral_radius > 0:
            W = W * (spectral_radius / current_spectral_radius)
        
        return W
    
    def initialize_input_weights(
        self,
        n_reservoir: int,
        n_inputs: int,
        input_scaling: float = 1.0,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Initialize input weight matrix.
        
        Args:
            n_reservoir: Number of reservoir units
            n_inputs: Number of input features
            input_scaling: Scaling factor for input weights
            random_seed: Random seed for reproducibility
            
        Returns:
            Initialized input weight matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        W_in = np.random.uniform(-input_scaling, input_scaling, (n_reservoir, n_inputs))
        return W_in
    
    def initialize_bias_weights(
        self,
        n_reservoir: int,
        bias_scaling: float = 0.1,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Initialize bias weights.
        
        Args:
            n_reservoir: Number of reservoir units
            bias_scaling: Scaling factor for bias weights
            random_seed: Random seed for reproducibility
            
        Returns:
            Initialized bias weight vector
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        bias = np.random.uniform(-bias_scaling, bias_scaling, n_reservoir)
        return bias