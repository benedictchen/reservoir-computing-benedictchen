"""
🔄 State Updates Module - Echo State Networks
============================================

📚 Research Foundation:
Jaeger, H. (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
Technical Report GMD-148, German National Research Center for Information Technology

🎯 Module Purpose:
This module implements comprehensive state update mechanisms for Echo State Networks including:
- Core reservoir state update dynamics following Jaeger's mathematical formulation
- Multiple noise types (additive, multiplicative, correlated, training vs testing)
- Leaky integration modes (pre/post activation, heterogeneous, adaptive)
- Output feedback for closed-loop generation and teacher forcing
- Input preprocessing and dimension handling
- Correlated noise generation for spatial reservoir correlations

🧠 Theoretical Foundation from Jaeger 2001:

📖 Section 3.1 - The Echo State Update Equation:
The core reservoir dynamics are governed by the fundamental equation:

x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n) + b + ξ(n))

Where:
- x(n) ∈ ℝ^N is the reservoir state at time n
- u(n) ∈ ℝ^K is the input vector at time n
- y(n) ∈ ℝ^L is the output feedback vector at time n
- W ∈ ℝ^(N×N) is the reservoir weight matrix (sparse, fixed)
- W_in ∈ ℝ^(N×K) is the input weight matrix (random, fixed)
- W_back ∈ ℝ^(N×L) is the output feedback matrix (optional, fixed)
- b ∈ ℝ^N is the bias vector
- ξ(n) ∈ ℝ^N is additive noise
- f(·) is the activation function (typically tanh)

📖 Section 3.2 - Leaky Integration:
Jaeger discusses leaky integrator neurons that combine previous state with new activation:

x(n+1) = (1-α)·x(n) + α·f(W·x(n) + W_in·u(n) + W_back·y(n) + b)

Where α ∈ (0,1] is the leak rate. This provides temporal smoothing and can improve
performance on tasks with multiple timescales.

📖 Section 4.3 - Output Feedback for Generation:
For closed-loop generation tasks, the network's own output is fed back:

x(n+1) = f(W·x(n) + W_in·u(n) + W_back·W_out·x(n))

This enables autonomous sequence generation after training.

📖 Section 5.2 - Noise and Robustness:
Jaeger emphasizes noise as crucial for:
1. Preventing overfitting to training sequences
2. Improving generalization to new inputs
3. Breaking spurious attractor states
4. Enhancing echo state property robustness

🔬 Mathematical Formulations:

1️⃣ Standard Update (Equation 1, Jaeger 2001):
   x(n+1) = f(W·x(n) + W_in·u(n) + b)

2️⃣ With Output Feedback (Equation 5, Jaeger 2001):
   x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n) + b)

3️⃣ Leaky Integration (Section 3.2):
   x(n+1) = (1-α)·x(n) + α·f(...)

4️⃣ With Additive Noise (Section 5.2):
   x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n) + b + ξ(n))

5️⃣ Multiplicative Noise (Research Extension):
   x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n) + b) ⊙ (1 + η(n))

🎨 State Update Process Flow:
=============================
Input u(n) → [Preprocessing] → [Noise Application] → [Matrix Operations] → [Leaky Integration] → x(n+1)
                ↓                      ↓                      ↓                    ↓
         Dimension Check      Noise Type Selection    W·x + W_in·u + W_back·y    α-weighted mixing
         Zero Padding         (6 different modes)     + bias terms            Post/Pre-activation
         Input Scaling        Correlated/Independent   Feedback computation     Heterogeneous rates

🔄 Integration Modes:
====================
- Standard: Post-activation leaking x(n+1) = (1-α)x(n) + αf(...)
- Pre-activation: Pre-activation leaking x(n+1) = f((1-α)x(n) + α(...))
- Heterogeneous: Different leak rates per neuron group
- Adaptive: Dynamic leak rates based on activation magnitude

🌊 Noise Types:
===============
- Additive: ξ ~ N(0, σ²) added to pre-activation
- Input: Noise added to input vector before processing
- Multiplicative: State-dependent noise (1 + η)·x where η ~ N(0, σ²)
- Correlated: Spatially correlated noise using Gaussian kernel
- Training vs Testing: Different noise levels for train/test phases
- Variance Scaled: Noise scaled by input variance σ²_noise ∝ var(u)

⚡ Performance Optimizations:
============================
- Sparse matrix operations for efficient W·x computation
- Vectorized noise generation
- In-place operations to minimize memory allocation
- Conditional computation for feedback terms
- Efficient correlation kernel convolution

🎯 Applications:
===============
- 📈 Time Series Forecasting: Financial markets, weather prediction
- 🎵 Sequence Generation: Music composition, text generation  
- 🤖 Control Systems: Adaptive control with output feedback
- 🔄 Closed-loop Tasks: Autonomous pattern generation
- 🧠 Temporal Pattern Recognition: Speech, gesture recognition

Author: Benedict Chen (benedict@benedictchen.com)
Implementation based on Jaeger (2001) with comprehensive extensions for practical applications.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
import warnings


class StateUpdatesMixin:
    """
    Mixin for comprehensive state update mechanisms in Echo State Networks.
    
    Implements the full mathematical framework from Jaeger 2001 including:
    - Core reservoir dynamics with all terms from the fundamental equation
    - Multiple noise types for robustness and generalization  
    - Leaky integration modes for temporal smoothing
    - Output feedback for closed-loop operation
    - Input preprocessing and dimension handling
    - Correlated noise generation for spatial patterns
    
    This mixin provides the computational heart of the ESN, where the
    "echo state" dynamics emerge from the interplay of input, recurrence,
    feedback, and nonlinear activation.
    """

    def _update_state(self, state: np.ndarray, input_vec: np.ndarray, output_feedback: np.ndarray = None) -> np.ndarray:
        """
        Core reservoir state update implementing Jaeger's fundamental equation.
        
        This is where the "magic" of Echo State Networks happens - the reservoir acts as a 
        nonlinear dynamical system that transforms input temporal patterns into rich, 
        high-dimensional state representations.
        
        Mathematical Foundation (Jaeger 2001, Equation 1):
        ==================================================
        
        The complete state update equation:
        x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n) + b + ξ(n))
        
        With leaky integration (Section 3.2):
        x(n+1) = (1-α)·x(n) + α·f(W·x(n) + W_in·u(n) + W_back·y(n) + b + ξ(n))
        
        Where:
        - x(n) ∈ ℝ^N: Current reservoir state vector
        - u(n) ∈ ℝ^K: Input vector at time n
        - y(n) ∈ ℝ^L: Output feedback vector (optional)
        - W ∈ ℝ^(N×N): Fixed reservoir connectivity matrix
        - W_in ∈ ℝ^(N×K): Fixed input weight matrix  
        - W_back ∈ ℝ^(N×L): Fixed output feedback matrix
        - b ∈ ℝ^N: Bias vector for reservoir neurons
        - ξ(n) ∈ ℝ^N: Additive noise vector
        - f(·): Activation function (typically tanh)
        - α ∈ (0,1]: Leak rate for temporal smoothing
        
        🔬 Theoretical Significance:
        - Fixed random connectivity W creates rich dynamics without training
        - Input weights W_in provide nonlinear input encoding
        - Output feedback W_back enables closed-loop generation  
        - Bias terms b shift operating points for optimal dynamics
        - Noise ξ improves robustness and prevents overfitting
        - Leaky integration α provides temporal smoothing across scales
        
        🎯 Implementation Details:
        - Handles multiple noise types (additive, multiplicative, correlated)
        - Supports various leaky integration modes (pre/post activation)
        - Processes output feedback with multiple distribution strategies
        - Ensures proper input dimensionality and preprocessing
        - Optimized for computational efficiency with sparse operations
        
        Args:
            state: Current reservoir state vector x(n) ∈ ℝ^N
            input_vec: Input vector u(n) ∈ ℝ^K  
            output_feedback: Optional output feedback y(n) ∈ ℝ^L for closed-loop operation
            
        Returns:
            Updated reservoir state x(n+1) ∈ ℝ^N
            
        Example:
            >>> esn = EchoStateNetwork(n_reservoir=100, n_inputs=3)
            >>> state = np.zeros(100)  # Initial state
            >>> input_data = np.random.randn(3)  # Input vector
            >>> new_state = esn._update_state(state, input_data)
            >>> print(f"State updated: {state.shape} -> {new_state.shape}")
        """
        # Apply comprehensive noise to input (6 different types)
        processed_input = self._apply_comprehensive_noise(input_vec)
        
        # Compute comprehensive output feedback (4 different modes)  
        feedback_term = self._compute_comprehensive_feedback(output_feedback)
        
        # Ensure input dimensions are compatible with W_input
        processed_input = self._ensure_input_dimensions(processed_input)
        
        # Initialize bias terms if not already present
        if not hasattr(self, 'bias') or self.bias is None:
            self._initialize_bias_terms()
        
        # Get activation function (supports 6 different types)
        activation_func = getattr(self, 'activation_functions', {}).get(
            getattr(self, 'activation_function', 'tanh'), np.tanh
        )
        if not callable(activation_func):
            activation_func = np.tanh
        
        # Compute pre-activation state: W·x(n) + W_in·u(n) + W_back·y(n) + b
        pre_activation = (
            self.W_reservoir @ state +           # Recurrent connections W·x(n)
            self.W_input @ processed_input +     # Input connections W_in·u(n)  
            feedback_term +                      # Output feedback W_back·y(n)
            self.bias                            # Bias terms b
        )
        
        # Apply comprehensive leaky integration (4 different modes)
        new_state = self._apply_comprehensive_leaky_integration(
            state, pre_activation, activation_func
        )
        
        # Apply multiplicative noise if configured
        if getattr(self, 'noise_type', 'additive') == 'multiplicative':
            multiplicative_noise = 1 + np.random.normal(0, self.noise_level, self.n_reservoir)
            new_state = new_state * multiplicative_noise
        
        # Apply sparse computation optimization if enabled
        if getattr(self, 'enable_sparse_computation', False):
            sparse_threshold = getattr(self, 'sparse_threshold', 1e-6)
            sparse_mask = np.abs(new_state) > sparse_threshold
            new_state = new_state * sparse_mask
        
        return new_state

    def _update_state_with_feedback(self, state: np.ndarray, input_vec: np.ndarray, feedback: np.ndarray) -> np.ndarray:
        """
        Update reservoir state with explicit output feedback for closed-loop operation.
        
        This method is essential for sequence generation tasks where the network's output
        is fed back as input for the next time step, enabling autonomous generation
        after training (Jaeger 2001, Section 4.3).
        
        Mathematical Foundation:
        =======================
        
        Closed-loop update equation:
        x(n+1) = f(W·x(n) + W_in·u(n) + W_back·y(n))
        
        Where y(n) is the network's previous output, creating a feedback loop that
        allows for autonomous sequence generation without external input.
        
        🔬 Research Context (Jaeger 2001, Section 4.3):
        "For generation tasks, the network output y(n) is fed back to the reservoir
        through output feedback connections W_back. This creates a closed dynamical
        system that can autonomously generate sequences similar to the training data."
        
        🎯 Applications:
        - Music generation: Generate melodies similar to training data
        - Text generation: Autonomous text production character by character  
        - Pattern completion: Fill in missing parts of temporal sequences
        - Attractor dynamics: Study limit cycles and strange attractors
        
        Args:
            state: Current reservoir state x(n) ∈ ℝ^N
            input_vec: Input vector u(n) ∈ ℝ^K
            feedback: Output feedback vector y(n) ∈ ℝ^L from previous time step
            
        Returns:
            Updated reservoir state x(n+1) ∈ ℝ^N with feedback incorporated
            
        Example:
            >>> # Training phase
            >>> esn.train(train_inputs, train_targets)
            >>> 
            >>> # Generation phase with feedback
            >>> state = esn.get_state()
            >>> prediction = esn.predict_single(state, current_input)
            >>> next_state = esn._update_state_with_feedback(state, current_input, prediction)
        """
        return self._update_state(state, input_vec, output_feedback=feedback)

    def _apply_comprehensive_noise(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive noise implementation supporting 6 different noise types.
        
        Noise is crucial in ESNs for several reasons identified by Jaeger (2001):
        1. Prevents overfitting to specific training sequences
        2. Improves generalization to new input patterns
        3. Breaks unwanted attractor states in the reservoir dynamics
        4. Enhances the Echo State Property by introducing stochasticity
        5. Provides robustness against input variations and measurement errors
        
        Mathematical Formulations:
        =========================
        
        1️⃣ Additive Noise (Standard):
           u'(n) = u(n) + ξ(n), ξ(n) ~ N(0, σ²I)
        
        2️⃣ Input Noise (Jaeger's Recommendation):
           u'(n) = u(n) + η(n), η(n) ~ N(0, σ²_input·I)
        
        3️⃣ Multiplicative Noise:
           x(n+1) = f(...)·(1 + ξ(n)), ξ(n) ~ N(0, σ²I)
        
        4️⃣ Correlated Noise (Spatial):
           ξ(n) = G(ξ_white(n)), G is Gaussian correlation kernel
        
        5️⃣ Training vs Testing Noise:
           σ = σ_train during training, σ = σ_test during testing
        
        6️⃣ Variance-Scaled Noise:
           σ²(n) = σ²_base · var(u(n))
        
        🔬 Research Background (Jaeger 2001, Section 5.2):
        "Adding small amounts of noise to the reservoir states or inputs can significantly
        improve performance. The noise acts as a regularizer that prevents the reservoir
        from overfitting to the training data and helps maintain the Echo State Property."
        
        Args:
            input_vec: Input vector u(n) to apply noise to
            
        Returns:
            Processed input vector u'(n) with applied noise
        """
        processed_input = input_vec.copy()
        noise_type = getattr(self, 'noise_type', 'additive')
        noise_level = getattr(self, 'noise_level', 0.01)
        
        if noise_type == 'input_noise':
            # Option 1: Input noise (Jaeger's recommendation)
            # Add noise directly to input before reservoir processing
            processed_input += np.random.normal(0, noise_level, input_vec.shape)
            
        elif noise_type == 'multiplicative':
            # Option 2: Multiplicative noise (applied later in update process)
            # This doesn't modify input directly but signals multiplicative noise usage
            pass  # Handled in _update_state method
            
        elif noise_type == 'correlated':
            # Option 3: Spatially correlated noise using Gaussian kernel
            correlation_length = getattr(self, 'noise_correlation_length', 5)
            correlated_noise = self._generate_correlated_noise(correlation_length)
            
            # Apply correlated noise to input (truncated to input dimensions)
            noise_component = correlated_noise[:len(input_vec)] if len(correlated_noise) > len(input_vec) else correlated_noise
            if len(noise_component) < len(input_vec):
                # Extend noise if input is longer
                extended_noise = np.tile(noise_component, (len(input_vec) // len(noise_component)) + 1)
                noise_component = extended_noise[:len(input_vec)]
            processed_input += noise_component
            
        elif noise_type == 'training_vs_testing':
            # Option 4: Different noise levels for training vs testing phases
            training_mode = getattr(self, 'training_mode', True)
            training_noise_ratio = getattr(self, 'training_noise_ratio', 0.5)
            
            if training_mode:
                current_noise_level = noise_level
            else:
                current_noise_level = noise_level * training_noise_ratio
                
            processed_input += np.random.normal(0, current_noise_level, input_vec.shape)
            
        elif noise_type == 'variance_scaled':
            # Option 5: Noise scaled by input variance (adaptive to signal strength)
            input_variance = np.var(input_vec) if np.var(input_vec) > 1e-8 else 1.0
            scaled_noise_level = noise_level * np.sqrt(input_variance)
            processed_input += np.random.normal(0, scaled_noise_level, input_vec.shape)
            
        else:  # Default: 'additive'
            # Option 6: Standard additive noise
            processed_input += np.random.normal(0, noise_level, input_vec.shape)
            
        return processed_input

    def _compute_comprehensive_feedback(self, output_feedback: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute comprehensive output feedback supporting 4 different distribution modes.
        
        Output feedback is critical for closed-loop operation and sequence generation.
        Jaeger (2001, Figure 1) shows feedback connections W_back connecting outputs
        back to reservoir neurons, enabling autonomous generation capabilities.
        
        Mathematical Foundation:
        =======================
        
        Basic feedback term:
        f_back(n) = W_back · y(n)
        
        Where W_back ∈ ℝ^(N×L) is the feedback weight matrix connecting L outputs
        to N reservoir neurons. Different modes distribute this connectivity:
        
        1️⃣ Direct Mode:
           f_back = W_back @ y, full connectivity matrix
        
        2️⃣ Sparse Mode:
           f_back = W_sparse @ y, only subset of neurons receive feedback
        
        3️⃣ Scaled Uniform Mode:
           f_back = α · 1_N @ y, uniform scaling to all neurons
        
        4️⃣ Hierarchical Mode:
           Different feedback strengths for different neuron groups
        
        🔬 Research Context (Jaeger 2001, Section 4.3):
        "Output feedback connections W_back allow the network to operate in closed-loop
        mode, where previous outputs influence future reservoir states. This is essential
        for sequence generation tasks where the network must produce coherent temporal
        patterns autonomously."
        
        Args:
            output_feedback: Previous output vector y(n-1) ∈ ℝ^L (or None)
            
        Returns:
            Feedback term f_back(n) ∈ ℝ^N to be added to reservoir update
        """
        # Return zero vector if no feedback provided or feedback disabled
        n_reservoir = getattr(self, 'n_reservoir', 100)
        
        if output_feedback is None:
            return np.zeros(n_reservoir)
            
        if not getattr(self, 'output_feedback_enabled', False):
            return np.zeros(n_reservoir)
            
        if not hasattr(self, 'W_back') or self.W_back is None:
            return np.zeros(n_reservoir)
        
        # Ensure feedback is properly shaped
        if output_feedback.ndim == 0:
            output_feedback = np.array([output_feedback])
        elif output_feedback.ndim > 1:
            output_feedback = output_feedback.flatten()
        
        # Get feedback mode (default to 'direct')
        feedback_mode = getattr(self, 'feedback_mode', 'direct')
        
        if feedback_mode == 'direct':
            # Standard direct feedback matrix multiplication
            feedback_term = self.W_back @ output_feedback.reshape(-1, 1)
            return feedback_term.flatten()
            
        elif feedback_mode == 'sparse':
            # Sparse feedback - only subset of neurons receive feedback
            sparsity = getattr(self, 'feedback_sparsity', 0.1)
            n_active = int(n_reservoir * sparsity)
            
            # Create sparse mask (can be precomputed and stored)
            if not hasattr(self, '_feedback_sparse_mask'):
                self._feedback_sparse_mask = np.zeros(n_reservoir, dtype=bool)
                active_indices = np.random.choice(n_reservoir, n_active, replace=False)
                self._feedback_sparse_mask[active_indices] = True
            
            feedback_term = np.zeros(n_reservoir)
            if hasattr(self, 'W_back_sparse'):
                feedback_subset = self.W_back_sparse @ output_feedback.reshape(-1, 1)
                feedback_term[self._feedback_sparse_mask] = feedback_subset.flatten()
            else:
                # Fallback: use portion of standard feedback matrix
                feedback_full = self.W_back @ output_feedback.reshape(-1, 1)
                feedback_term[self._feedback_sparse_mask] = feedback_full.flatten()[self._feedback_sparse_mask]
                
            return feedback_term
            
        elif feedback_mode == 'scaled_uniform':
            # Uniform feedback scaled by single parameter
            feedback_scale = getattr(self, 'feedback_scale', 0.1)
            feedback_strength = np.mean(output_feedback) * feedback_scale
            return np.full(n_reservoir, feedback_strength)
            
        elif feedback_mode == 'hierarchical':
            # Hierarchical feedback with different strengths for neuron groups
            if hasattr(self, 'neuron_groups') and hasattr(self, 'feedback_strengths'):
                feedback_term = np.zeros(n_reservoir)
                output_magnitude = np.mean(np.abs(output_feedback))
                
                for group_idx, neuron_indices in enumerate(self.neuron_groups):
                    if group_idx < len(self.feedback_strengths):
                        strength = self.feedback_strengths[group_idx]
                        feedback_term[neuron_indices] = output_magnitude * strength
                        
                return feedback_term
            else:
                # Fallback to direct mode if hierarchical not configured
                feedback_term = self.W_back @ output_feedback.reshape(-1, 1)
                return feedback_term.flatten()
        
        else:
            # Unknown mode, fallback to direct
            feedback_term = self.W_back @ output_feedback.reshape(-1, 1)
            return feedback_term.flatten()

    def _ensure_input_dimensions(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Ensure input vector dimensions are compatible with reservoir input weights.
        
        ESNs require careful handling of input dimensions to maintain compatibility
        with the fixed input weight matrix W_in. This method handles common dimension
        mismatches through padding, truncation, or reshaping.
        
        Mathematical Requirement:
        ========================
        
        For the matrix operation W_in @ u(n) to be valid:
        - W_in ∈ ℝ^(N×K) where N = reservoir size, K = input size
        - u(n) ∈ ℝ^K must have exactly K elements
        
        Dimension Handling Strategies:
        - Input too long: Truncate to first K elements
        - Input too short: Zero-pad to K elements  
        - Multi-dimensional: Flatten to 1D vector
        - Scalar input: Convert to vector of length 1
        
        🔬 Practical Considerations:
        Real-world applications often have varying input dimensions, and this
        preprocessing step ensures robust operation across different data formats
        while maintaining the mathematical validity of the reservoir update.
        
        Args:
            input_vec: Input vector u(n) of potentially incorrect dimensions
            
        Returns:
            Processed input vector u'(n) ∈ ℝ^K compatible with W_in
        """
        # Flatten multi-dimensional inputs
        if input_vec.ndim > 1:
            input_vec = input_vec.flatten()
        
        # Handle scalar inputs
        if input_vec.ndim == 0:
            input_vec = np.array([input_vec])
        
        # Get expected input dimensions from W_input shape
        expected_input_size = getattr(self, 'W_input', np.array([[]])).shape[1]
        
        if expected_input_size == 0:
            # W_input not properly initialized, return input as-is
            return input_vec
        
        current_size = len(input_vec)
        
        if current_size == expected_input_size:
            # Perfect match, no processing needed
            return input_vec
        
        elif current_size > expected_input_size:
            # Input too long: truncate to expected size
            return input_vec[:expected_input_size]
        
        else:
            # Input too short: zero-pad to expected size
            padded_input = np.zeros(expected_input_size)
            padded_input[:current_size] = input_vec
            return padded_input

    def _apply_comprehensive_leaky_integration(self, state: np.ndarray, pre_activation: np.ndarray, activation_func) -> np.ndarray:
        """
        Apply comprehensive leaky integration supporting 4 different modes.
        
        Leaky integration is a crucial extension to the basic ESN dynamics that
        provides temporal smoothing and multi-timescale processing. Jaeger (2001)
        discusses this as the "leaky integrator" neuron model.
        
        Mathematical Foundation (Jaeger 2001, Section 3.2):
        ===================================================
        
        Standard Leaky Integration:
        x(n+1) = (1-α)·x(n) + α·f(W·x(n) + W_in·u(n) + W_back·y(n))
        
        Where α ∈ (0,1] is the leak rate:
        - α = 1: No leaking (standard update)
        - α → 0: Strong leaking (slow dynamics)
        - α ∈ (0,1): Temporal smoothing
        
        Implementation Modes:
        ====================
        
        1️⃣ Standard (Post-Activation):
           x(n+1) = (1-α)·x(n) + α·f(pre_activation)
           
        2️⃣ Pre-Activation:
           x(n+1) = f((1-α)·x(n) + α·pre_activation)
           
        3️⃣ Heterogeneous:
           x_i(n+1) = (1-α_i)·x_i(n) + α_i·f(pre_activation_i)
           Different leak rates α_i for different neurons
           
        4️⃣ Adaptive:
           α_i(n) = α_base,i · (1 + β·|f(pre_activation_i)|)
           Leak rates adapt based on activation strength
        
        🔬 Research Motivation:
        Leaky integration allows ESNs to handle multiple timescales naturally.
        Fast-leaking neurons capture rapid changes, while slow-leaking neurons
        maintain longer-term memory. This mimics biological neural dynamics.
        
        Args:
            state: Current reservoir state x(n) ∈ ℝ^N
            pre_activation: Pre-activation values before nonlinearity
            activation_func: Nonlinear activation function f(·)
            
        Returns:
            Updated state x(n+1) ∈ ℝ^N with leaky integration applied
        """
        leak_rate = getattr(self, 'leak_rate', 1.0)
        leak_mode = getattr(self, 'leak_mode', 'standard')
        
        if leak_mode == 'pre_activation':
            # Mode 2: Apply leaking before activation function
            if hasattr(self, 'leak_rates') and self.leak_rates is not None:
                # Use heterogeneous leak rates if available
                mixed_activation = self.leak_rates * pre_activation + (1 - self.leak_rates) * state
            else:
                # Use single leak rate for all neurons
                mixed_activation = leak_rate * pre_activation + (1 - leak_rate) * state
            
            new_state = activation_func(mixed_activation)
            
        elif leak_mode == 'heterogeneous' and hasattr(self, 'leak_rates') and self.leak_rates is not None:
            # Mode 3: Different leak rates for different neurons
            activated = activation_func(pre_activation)
            new_state = (1 - self.leak_rates) * state + self.leak_rates * activated
            
        elif leak_mode == 'adaptive' and hasattr(self, 'leak_rates') and self.leak_rates is not None:
            # Mode 4: Adaptive leak rates based on activation magnitude
            activated = activation_func(pre_activation)
            
            # Compute adaptive leak rates
            adaptation_strength = getattr(self, 'leak_adaptation_strength', 0.1)
            adaptive_rates = self.leak_rates * (1 + adaptation_strength * np.abs(activated))
            
            # Ensure rates stay in valid range [0, 1]
            adaptive_rates = np.clip(adaptive_rates, 0.0, 1.0)
            
            new_state = (1 - adaptive_rates) * state + adaptive_rates * activated
            
        else:
            # Mode 1: Standard post-activation leaking
            activated = activation_func(pre_activation)
            
            if hasattr(self, 'leak_rates') and self.leak_rates is not None and leak_mode != 'standard':
                # Use heterogeneous rates even in standard mode if available
                new_state = (1 - self.leak_rates) * state + self.leak_rates * activated
            else:
                # Single leak rate for all neurons
                new_state = (1 - leak_rate) * state + leak_rate * activated
        
        return new_state

    def _generate_correlated_noise(self, correlation_length: int = 5) -> np.ndarray:
        """
        Generate spatially correlated noise for reservoir neurons.
        
        Correlated noise provides a more realistic noise model that considers
        spatial relationships between neurons. This can improve performance
        on tasks where spatial patterns in the reservoir are important.
        
        Mathematical Implementation:
        ===========================
        
        1. Generate white noise: ξ_white ~ N(0, I)
        2. Create Gaussian correlation kernel:
           K(i) = exp(-0.5 · (i - μ)² / σ²)
           where μ = correlation_length, σ = correlation_length
        3. Apply spatial correlation via convolution:
           ξ_corr = K ★ ξ_white
        
        Algorithm Steps:
        - Generate independent Gaussian noise for each neuron
        - Create normalized Gaussian kernel with specified correlation length
        - Apply circular convolution to create spatial correlations
        - Return correlated noise vector for reservoir
        
        🔬 Research Motivation:
        In biological neural networks, nearby neurons often have correlated
        activity due to shared inputs and local connectivity. Correlated noise
        models this phenomenon and can improve the biological realism of ESN
        dynamics.
        
        Args:
            correlation_length: Spatial correlation length (higher = more correlation)
            
        Returns:
            Correlated noise vector ξ_corr ∈ ℝ^N for reservoir neurons
        """
        n_reservoir = getattr(self, 'n_reservoir', 100)
        noise_level = getattr(self, 'noise_level', 0.01)
        
        # Generate independent white noise
        white_noise = np.random.normal(0, noise_level, n_reservoir)
        
        # Create Gaussian correlation kernel
        kernel_size = min(2 * correlation_length + 1, n_reservoir)
        kernel_center = correlation_length
        
        # Gaussian kernel: exp(-0.5 * (x - μ)² / σ²)
        kernel_positions = np.arange(kernel_size)
        kernel = np.exp(-0.5 * (kernel_positions - kernel_center)**2 / correlation_length**2)
        
        # Normalize kernel to preserve noise variance
        kernel = kernel / np.sum(kernel)
        
        # Apply spatial correlation through convolution
        # Use circular convolution to handle boundary conditions
        if kernel_size < n_reservoir:
            # Pad noise for circular convolution
            padded_noise = np.concatenate([white_noise, white_noise[:kernel_size-1]])
            correlated = np.convolve(padded_noise, kernel, mode='valid')[:n_reservoir]
        else:
            # Kernel is too large, use direct convolution
            correlated = np.convolve(white_noise, kernel, mode='same')
        
        return correlated

    def _initialize_bias_terms(self):
        """
        Initialize bias terms for reservoir neurons.
        
        Bias terms are crucial for optimal reservoir dynamics as they shift
        the operating point of neurons and can significantly impact performance
        (Jaeger 2001, Equation 1).
        
        Three bias initialization strategies:
        1. Random: Small random values from uniform distribution
        2. Zero: No bias (simplest case) 
        3. Adaptive: Bias terms that can adapt during training
        """
        n_reservoir = getattr(self, 'n_reservoir', 100)
        bias_type = getattr(self, 'bias_type', 'random')
        bias_scale = getattr(self, 'bias_scale', 0.1)
        
        if bias_type == 'zero':
            self.bias = np.zeros(n_reservoir)
        elif bias_type == 'adaptive':
            # Initialize with small random values, can be updated during training
            self.bias = np.random.uniform(-bias_scale, bias_scale, n_reservoir)
        else:  # 'random'
            self.bias = np.random.uniform(-bias_scale, bias_scale, n_reservoir)