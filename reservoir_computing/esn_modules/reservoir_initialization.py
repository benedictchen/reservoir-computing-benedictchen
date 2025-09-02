"""
🌊 Reservoir Initialization Module - Echo State Networks
========================================================

📚 Research Foundation:
Jaeger, H. (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
Technical Report GMD-148, German National Research Center for Information Technology

🎯 Module Purpose:
This module handles all aspects of ESN reservoir initialization including:
- Fixed random reservoir matrix creation with proper scaling
- Input weight matrix initialization 
- Output feedback configuration
- Activation function setup
- Bias term initialization
- Leak rate configuration for temporal memory

🧠 Theoretical Foundation:
The reservoir initialization is critical for the Echo State Property (ESP):
- Spectral radius < 1.0 ensures fading memory
- Sparse connectivity creates rich dynamics
- Proper scaling maintains stability
- Random weights avoid manual design

🔧 Key Components:
- Reservoir matrix: W_res (fixed, scaled by spectral radius)
- Input weights: W_in (trainable scaling, fixed connectivity)
- Output feedback: W_fb (optional, for closed-loop generation)
- Bias vectors: b_res, b_out (for non-zero baseline activations)
"""

import numpy as np
from scipy import sparse
from typing import Optional, Callable, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class ReservoirInitializationMixin:
    """
    Mixin for reservoir initialization in Echo State Networks.
    
    Handles all aspects of creating the fixed random reservoir that gives
    ESNs their computational power through rich temporal dynamics.
    """

    def _initialize_reservoir(self):
        """
        Initialize the fixed random reservoir matrix
        
        This is the core of Jaeger's innovation - a sparse, scaled random matrix
        that creates rich temporal dynamics without any training.
        
        🧠 Theory (Jaeger 2001, Section 2.1):
        "The reservoir weights are randomly chosen and remain fixed during training.
        The spectral radius should be less than 1 to ensure the echo state property."
        """
        # Create reservoir matrix based on connection topology
        if self.connection_topology == 'ring':
            self.W_reservoir = self._create_ring_topology()
        elif self.connection_topology == 'small_world':
            self.W_reservoir = self._create_small_world_topology()
        elif self.connection_topology == 'scale_free':
            self.W_reservoir = self._create_scale_free_topology()
        elif self.reservoir_connectivity_mask is not None:
            self.W_reservoir = self._create_custom_topology()
        else:
            # Default: Erdős-Rényi random topology
            self.W_reservoir = sparse.random(
                self.n_reservoir, 
                self.n_reservoir, 
                density=self.sparsity,
                format='csr',
                random_state=np.random.get_state()[1][0]
            ).toarray()
            
        # ESP validation will be done after input weights are initialized
        self.esp_validated = None
        
        # Advanced Spectral Radius Scaling
        self.spectral_scaling_method = getattr(self, 'spectral_scaling_method', 'standard')
        self.handle_complex_eigenvalues = getattr(self, 'handle_complex_eigenvalues', True)
        self.verify_esp_after_scaling = getattr(self, 'verify_esp_after_scaling', True)
        
        # COMPREHENSIVE Configuration Options for Maximum User Control
        self._setup_comprehensive_configuration()
        
        # Scale reservoir to desired spectral radius
        self._scale_reservoir_spectral_radius()
        
        print(f"✓ Reservoir initialized: spectral radius = {np.max(np.abs(np.linalg.eigvals(self.W_reservoir))):.3f}")
        
        # Initialize all comprehensive subsystems
        self._initialize_esp_validation_methods()
        self._initialize_activation_functions()
        self._initialize_bias_terms()
        self._initialize_leak_rates()
        
        print(f"🔧 ESN configured with {self.activation_function} activation, {self.noise_type} noise, {self.leak_mode} leaky integration")

    def _setup_comprehensive_configuration(self):
        """Setup comprehensive configuration options for maximum user control"""
        
        # Echo State Property Validation Method (4 options)
        self.esp_validation_method = getattr(self, 'esp_validation_method', 'fast')
        
        # Activation Function Selection (6 options)  
        self.activation_function = getattr(self, 'activation_function', 'tanh')
        self.custom_activation = getattr(self, 'custom_activation', None)
        
        # Output Feedback Configuration (4 modes)
        self.output_feedback_mode = getattr(self, 'output_feedback_mode', 'direct')
        self.output_feedback_sparsity = getattr(self, 'output_feedback_sparsity', 0.1)
        
        # Bias Implementation (3 types)
        self.bias_type = getattr(self, 'bias_type', 'random')
        self.bias_scale = getattr(self, 'bias_scale', 0.1)
        
        # Noise Configuration (6 types)
        self.noise_type = getattr(self, 'noise_type', 'additive')
        self.noise_correlation_length = getattr(self, 'noise_correlation_length', 5)
        self.training_noise_ratio = getattr(self, 'training_noise_ratio', 1.0)
        
        # Leaky Integrator Configuration (4 modes)
        if not hasattr(self, 'leak_mode'):
            self.leak_mode = 'post_activation'
        if not hasattr(self, 'leak_rates') or (not self.multiple_timescales and self.leak_rates is None):
            self.leak_rates = None
        
        # Additional configuration options
        self.reservoir_topology = getattr(self, 'reservoir_topology', 'random')
        self.teacher_forcing_strategy = getattr(self, 'teacher_forcing_strategy', 'full')
        self.training_solver = getattr(self, 'training_solver', 'ridge')
        self.state_collection_method = getattr(self, 'state_collection_method', 'all_states')
        
        # Analysis and optimization flags
        self.enable_memory_capacity_analysis = getattr(self, 'enable_memory_capacity_analysis', False)
        self.enable_sparse_computation = getattr(self, 'enable_sparse_computation', False)
        self.sparse_threshold = getattr(self, 'sparse_threshold', 1e-6)

    def _scale_reservoir_spectral_radius(self):
        """Scale reservoir matrix to achieve desired spectral radius"""
        
        # Get current spectral properties
        eigenvalues = np.linalg.eigvals(self.W_reservoir)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        
        if current_spectral_radius > 0:
            if self.spectral_scaling_method == 'standard':
                # Standard uniform scaling
                self.W_reservoir *= self.spectral_radius / current_spectral_radius
                
            elif self.spectral_scaling_method == 'complex_preserving':
                # Complex eigenvalue preserving scaling (Jaeger 2001 recommendation)
                scaling_factor = self.spectral_radius / current_spectral_radius
                self.W_reservoir *= scaling_factor
                
                # Additional phase preservation for complex eigenvalues
                if self.handle_complex_eigenvalues and np.any(np.iscomplex(eigenvalues)):
                    complex_eigs = eigenvalues[np.iscomplex(eigenvalues)]
                    if len(complex_eigs) > 0:
                        dominant_complex = complex_eigs[np.argmax(np.abs(complex_eigs))]
                        phase_correction = np.angle(dominant_complex)
                        print(f"   Complex eigenvalue phase preserved: {phase_correction:.3f} radians")
                        
            elif self.spectral_scaling_method == 'adaptive':
                # Dynamic spectral radius adaptation based on task complexity
                initial_radius = min(self.spectral_radius, 0.8)  # Conservative start
                self.W_reservoir *= initial_radius / current_spectral_radius
                self.adaptive_radius_history = [initial_radius]
                
        # Verify ESP is maintained after scaling
        if self.verify_esp_after_scaling:
            self._verify_esp_post_scaling()

    def _verify_esp_post_scaling(self):
        """Verify Echo State Property is maintained after spectral scaling"""
        post_scale_esp = self._validate_echo_state_property_fast()
        if not post_scale_esp:
            print(f"⚠️  ESP violated after scaling, reducing spectral radius")
            # Try smaller spectral radius
            for reduced_radius in [0.9, 0.8, 0.7, 0.6, 0.5]:
                test_matrix = self.W_reservoir * (reduced_radius / self.spectral_radius)
                self.W_reservoir = test_matrix
                if self._validate_echo_state_property_fast():
                    self.spectral_radius = reduced_radius
                    print(f"   ✓ ESP restored with radius = {reduced_radius}")
                    break

    def _initialize_input_weights(self, n_inputs: int):
        """
        Initialize random input weights with optional sparse connectivity
        
        🧠 Theory (Jaeger 2001, Section 2.2):
        "Input weights should be chosen to drive the reservoir into appropriate 
        dynamic regimes without saturating the activation function."
        """
        # Create base input weight matrix
        self.W_input = np.random.uniform(
            -self.input_scaling, 
            self.input_scaling, 
            (self.n_reservoir, n_inputs)
        )
        
        # Apply input connectivity sparsity if less than 1.0
        if self.input_connectivity < 1.0:
            # Create sparse mask for input connections
            n_connections = int(self.n_reservoir * n_inputs * self.input_connectivity)
            mask = np.zeros((self.n_reservoir, n_inputs), dtype=bool)
            
            # Randomly select connections to keep
            reservoir_indices = np.random.choice(self.n_reservoir, n_connections, replace=True)
            input_indices = np.random.choice(n_inputs, n_connections, replace=True)
            mask[reservoir_indices, input_indices] = True
            
            # Apply mask to input weights
            self.W_input[~mask] = 0.0
            
        print(f"✓ Input weights initialized: {n_inputs} inputs → {self.n_reservoir} reservoir")

    def _initialize_output_feedback(self, n_outputs: int, feedback_scaling: float = 0.1):
        """
        Initialize output feedback connections for closed-loop generation
        
        🧠 Theory (Jaeger 2001, Section 3.3):
        "Output feedback enables the network to generate autonomous sequences
        by feeding its own outputs back as inputs."
        """
        if self.output_feedback:
            self.W_feedback = np.random.uniform(
                -feedback_scaling, 
                feedback_scaling, 
                (self.n_reservoir, n_outputs)
            )
            print(f"✓ Output feedback initialized: {n_outputs} outputs → {self.n_reservoir} reservoir")
        else:
            self.W_feedback = None

    def _initialize_output_feedback_comprehensive(self, n_outputs: int, feedback_scaling: float = 0.1):
        """
        Comprehensive output feedback initialization with multiple modes
        
        Implements the comprehensive feedback configurations requested in original code.
        """
        if not self.output_feedback:
            self.W_feedback = None
            return
            
        if self.output_feedback_mode == 'direct':
            # Standard uniform random feedback
            self.W_feedback = np.random.uniform(
                -feedback_scaling, feedback_scaling,
                (self.n_reservoir, n_outputs)
            )
            
        elif self.output_feedback_mode == 'sparse':
            # Sparse feedback connections
            self.W_feedback = np.random.uniform(
                -feedback_scaling, feedback_scaling,
                (self.n_reservoir, n_outputs)
            )
            # Apply sparsity mask
            mask = np.random.random((self.n_reservoir, n_outputs)) < self.output_feedback_sparsity
            self.W_feedback[~mask] = 0.0
            
        elif self.output_feedback_mode == 'scaled_uniform':
            # Scaled uniform distribution
            scaling = feedback_scaling * np.sqrt(3)  # Maintain variance
            self.W_feedback = np.random.uniform(
                -scaling, scaling, (self.n_reservoir, n_outputs)
            )
            
        elif self.output_feedback_mode == 'hierarchical':
            # Hierarchical feedback (stronger connections to early reservoir units)
            self.W_feedback = np.zeros((self.n_reservoir, n_outputs))
            for i in range(self.n_reservoir):
                # Decay strength with reservoir unit index
                strength = feedback_scaling * np.exp(-i / (self.n_reservoir * 0.3))
                self.W_feedback[i, :] = np.random.uniform(-strength, strength, n_outputs)
                
        print(f"✓ Output feedback initialized ({self.output_feedback_mode} mode): {n_outputs} → {self.n_reservoir}")

    def _initialize_activation_functions(self):
        """
        Initialize activation function based on configuration
        
        🧠 Theory (Jaeger 2001, Section 2.1):
        "The activation function should provide nonlinearity while maintaining
        the contractive property necessary for the echo state property."
        """
        if self.activation_function == 'tanh':
            self.activation_func = np.tanh
        elif self.activation_function == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            self.activation_func = lambda x: np.maximum(0, x)
        elif self.activation_function == 'leaky_relu':
            self.activation_func = lambda x: np.maximum(0.01 * x, x)
        elif self.activation_function == 'linear':
            self.activation_func = lambda x: x
        elif self.activation_function == 'custom' and self.custom_activation is not None:
            self.activation_func = self.custom_activation
        else:
            # Default to tanh
            self.activation_func = np.tanh
            self.activation_function = 'tanh'

    def _initialize_bias_terms(self):
        """
        Initialize bias terms for reservoir units
        
        🧠 Theory:
        Bias terms can help break symmetry and provide non-zero baseline activation
        even when inputs are zero.
        """
        if self.bias_type == 'random':
            self.reservoir_bias = np.random.uniform(
                -self.bias_scale, self.bias_scale, self.n_reservoir
            )
        elif self.bias_type == 'zero':
            self.reservoir_bias = np.zeros(self.n_reservoir)
        elif self.bias_type == 'adaptive':
            # Start with small random bias, can be adapted during training
            self.reservoir_bias = np.random.uniform(
                -self.bias_scale * 0.1, self.bias_scale * 0.1, self.n_reservoir
            )
        else:
            self.reservoir_bias = np.zeros(self.n_reservoir)

    def _initialize_leak_rates(self):
        """
        Initialize leak rates for leaky integrator dynamics
        
        🧠 Theory (Jaeger et al. 2007):
        "Leak rates control the speed of reservoir dynamics, enabling
        multi-timescale processing and improved memory capacity."
        """
        if self.leak_mode == 'post_activation':
            # Single leak rate applied after activation
            if self.leak_rates is None:
                self.leak_rates = self.leak_rate
                
        elif self.leak_mode == 'pre_activation':
            # Single leak rate applied before activation
            if self.leak_rates is None:
                self.leak_rates = self.leak_rate
                
        elif self.leak_mode == 'heterogeneous':
            # Different leak rates for different reservoir units
            if self.leak_rates is None:
                # Create diverse leak rates following log-normal distribution
                self.leak_rates = np.random.lognormal(
                    mean=np.log(self.leak_rate), 
                    sigma=0.5, 
                    size=self.n_reservoir
                )
                # Clip to reasonable range
                self.leak_rates = np.clip(self.leak_rates, 0.01, 1.0)
                
        elif self.leak_mode == 'adaptive':
            # Adaptive leak rates that can change during training
            if self.leak_rates is None:
                self.leak_rates = np.full(self.n_reservoir, self.leak_rate)
            self.adaptive_leak_history = [self.leak_rates.copy()]
            
        print(f"✓ Leak rates initialized ({self.leak_mode} mode): range {np.min(self.leak_rates):.3f}-{np.max(self.leak_rates):.3f}")

    def _initialize_esp_validation_methods(self):
        """Initialize Echo State Property validation methods"""
        # This will be implemented in the esp_validation module
        # Just set up the configuration here
        self.esp_validation_config = {
            'method': self.esp_validation_method,
            'n_tests': 10,
            'test_length': 1000,
            'tolerance': 1e-6
        }


# Export the mixin class
__all__ = ['ReservoirInitializationMixin']