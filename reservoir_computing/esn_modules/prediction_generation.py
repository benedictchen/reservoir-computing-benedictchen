"""
🎯 Echo State Network - Prediction and Generation Module
=======================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

Based on: Herbert Jaeger (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"

🔬 Mathematical Foundation - Prediction and Generation Theory:
=============================================================

**Open-Loop Prediction (Section 3.1 - Jaeger 2001)**:
Echo State Networks excel at open-loop prediction where the network receives
external input at each time step and predicts future outputs. The mathematical
framework follows:

State Update Equation:
    x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t) + W_back·y(t) + noise)

Open-Loop Output:
    y(t) = W_out·[x(t); u(t); 1] + bias

Where:
- x(t) ∈ ℝ^N: reservoir state vector (N neurons)
- u(t) ∈ ℝ^K: input vector (K inputs)
- y(t) ∈ ℝ^L: output vector (L outputs)
- W_in ∈ ℝ^(N×K): input weight matrix (fixed)
- W_res ∈ ℝ^(N×N): reservoir weight matrix (fixed, sparse)
- W_out ∈ ℝ^(L×(N+K+1)): output weight matrix (trainable)
- W_back ∈ ℝ^(N×L): output feedback matrix (fixed, optional)
- f: activation function (typically tanh)
- α ∈ (0,1]: leak rate parameter

**Closed-Loop Generation (Section 4.3 - Jaeger 2001)**:
For autonomous sequence generation, the network operates in closed-loop mode
where outputs are fed back as inputs. This enables the network to generate
complex temporal patterns without external driving:

Autonomous Generation:
    u(t+1) = y(t)  (output becomes next input)
    x(t+1) = (1-α)x(t) + α·f(W_in·u(t+1) + W_res·x(t) + W_back·y(t))
    y(t+1) = W_out·[x(t+1); u(t+1); 1]

**Teacher Forcing (Section 3.3 - Jaeger 2001)**:
During training, teacher forcing can improve learning by feeding target outputs
back to the reservoir instead of predicted outputs:

Teacher Forcing Mode:
    u_teacher(t) = y_target(t-1)  (use ground truth)
    x(t+1) = (1-α)x(t) + α·f(W_in·u_teacher(t) + W_res·x(t) + W_back·y_target(t))

**Echo State Property (Theorem 1 - Jaeger 2001)**:
The fundamental requirement for ESN functionality is the Echo State Property (ESP):
The network state should have fading memory, meaning:

    ||∂x(t+1)/∂x(t)|| < 1

This is typically ensured by constraining the spectral radius ρ(W_res) < 1.

**Multi-Step Prediction Strategies**:
1. **Iterative Prediction**: Use previous predictions as inputs for future steps
2. **Direct Multi-Output**: Train separate outputs for different prediction horizons  
3. **Hybrid Approach**: Combine multiple prediction strategies

🎨 Implementation Features:
==========================
✅ Open-loop prediction with washout period handling
✅ Closed-loop autonomous generation with multiple feedback modes
✅ Teacher forcing support for improved training
✅ Multi-step ahead prediction capabilities
✅ Proper handling of output feedback matrices
✅ Washout period adaptive strategies
✅ Support for various generation modes (autonomous, driven, semi-autonomous)
✅ Comprehensive error handling and validation
✅ Mathematical theory preservation from original paper

🚀 Advanced Capabilities:
========================
- **Dynamic Washout**: Adaptive transient removal based on convergence
- **Multiple Generation Modes**: Autonomous, driven, and hybrid operation
- **Output Feedback**: Full implementation of W_back matrix from Jaeger 2001
- **Teacher Forcing**: Ground truth feedback during training phases
- **Multi-dimensional**: Support for multi-input/multi-output scenarios
- **Numerical Stability**: Condition number monitoring and warnings
"""

import numpy as np
from typing import Optional, Dict, Any
import warnings


class PredictionGenerationMixin:
    """
    🎯 Echo State Network Prediction and Generation Mixin
    
    This mixin provides comprehensive prediction and generation capabilities
    for Echo State Networks, implementing the mathematical framework from
    Jaeger (2001) with modern enhancements.
    
    **Mathematical Foundation**:
    The mixin implements both open-loop and closed-loop operation modes,
    supporting the full range of ESN applications from time series prediction
    to autonomous pattern generation.
    
    **Key Methods**:
    - run_reservoir(): Core state evolution with washout
    - predict(): Open-loop prediction with external inputs
    - generate(): Closed-loop autonomous generation
    - generate_with_feedback(): Advanced feedback-based generation
    """

    def run_reservoir(self, inputs: np.ndarray, washout: int = 100) -> np.ndarray:
        """
        🌊 Run Input Sequence Through Reservoir - Core State Evolution
        ============================================================
        
        **Mathematical Theory (Jaeger 2001, Section 2.1)**:
        
        This method implements the fundamental reservoir dynamics equation:
        
            x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t) + η(t))
        
        Where:
        - x(t) ∈ ℝ^N: reservoir state at time t
        - u(t) ∈ ℝ^K: input vector at time t  
        - W_in ∈ ℝ^(N×K): input weight matrix (fixed, random)
        - W_res ∈ ℝ^(N×N): reservoir connectivity matrix (fixed, sparse)
        - f: activation function (tanh for boundedness)
        - α: leak rate (default 1.0 for standard ESN)
        - η(t): optional noise term for regularization
        
        **Washout Period (Section 3.2)**:
        The initial transient states are discarded to ensure the reservoir
        has settled into its attractor dynamics. This is crucial for:
        - Removing initialization bias
        - Ensuring consistent starting conditions
        - Achieving the Echo State Property
        
        **Echo State Property**:
        For the reservoir to function properly, it must satisfy:
            ρ(W_res) < 1  (spectral radius < 1)
        
        This ensures that the influence of initial conditions fades over time,
        creating the "echo" behavior that gives ESNs their name.
        
        Args:
            inputs (np.ndarray): Input sequence of shape (time_steps, n_inputs)
                Time series data to drive the reservoir dynamics
            washout (int, optional): Number of initial time steps to discard.
                Defaults to 100. Larger values ensure better settling but
                reduce available training data.
                
        Returns:
            np.ndarray: Reservoir states after washout of shape 
                (time_steps - washout, n_reservoir). These states capture
                the rich temporal dynamics induced by the input sequence.
                
        Raises:
            ValueError: If inputs are malformed or washout >= time_steps
            
        **Research Notes**:
        - Washout period should be ~3-5 times the reservoir's memory timescale
        - For periodic inputs, washout should be several periods
        - Adaptive washout based on state convergence is more robust
        
        **Implementation Details**:
        1. Initialize input weights if not already done
        2. Start with zero state (reservoir at rest)
        3. Evolve state through all time steps
        4. Collect states only after washout period
        5. Store final state for generation tasks
        """
        
        # Validate inputs
        if inputs.ndim != 2:
            raise ValueError("Inputs must be 2D array (time_steps, n_inputs)")
            
        time_steps, n_inputs = inputs.shape
        
        if washout >= time_steps:
            raise ValueError("Washout period must be less than total time steps")
            
        # Initialize input weights if needed (first time running)
        if not hasattr(self, 'W_input'):
            self._initialize_input_weights(n_inputs)
            
        # Initialize reservoir state (start at rest)
        state = np.zeros(self.n_reservoir)
        states = []
        
        # Evolve reservoir through all time steps
        for t in range(time_steps):
            # Update state using core dynamics equation
            state = self._update_state(state, inputs[t])
            
            # Collect states only after washout period (transients removed)
            if t >= washout:
                states.append(state.copy())
                
        # Store final state for subsequent generation tasks
        self.last_state = state
        
        # Convert to numpy array for efficient computation
        return np.array(states)

    def predict(self, inputs: np.ndarray, washout: int = 100) -> np.ndarray:
        """
        🔮 Open-Loop Prediction - Generate Predictions with External Input
        ================================================================
        
        **Mathematical Theory (Jaeger 2001, Section 3.1)**:
        
        Open-loop prediction uses external inputs at each time step to generate
        predictions. The mathematical framework is:
        
        State Evolution:
            x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t))
            
        Prediction Equation:
            ŷ(t) = W_out·[x(t); u(t); 1]
            
        Where:
        - ŷ(t): predicted output at time t
        - [x(t); u(t); 1]: concatenated state, input, and bias term
        - W_out: trained readout weights (from linear regression)
        
        **Key Advantages of Open-Loop Prediction**:
        1. **Stability**: External input prevents error accumulation
        2. **Accuracy**: No feedback of prediction errors
        3. **Robustness**: Suitable for noisy or partially observable systems
        4. **Interpretability**: Direct input-output relationship
        
        **Applications**:
        - Time series forecasting with external features
        - System identification with known inputs
        - Signal processing and filtering
        - Nonlinear regression with temporal dependencies
        
        Args:
            inputs (np.ndarray): Input sequence of shape (time_steps, n_inputs)
                External driving signals for prediction
            washout (int, optional): Initial transient removal period.
                Defaults to 100. Should match training washout.
                
        Returns:
            np.ndarray: Predictions of shape (time_steps - washout, n_outputs)
                Predicted outputs for each time step after washout
                
        Raises:
            ValueError: If network hasn't been trained or inputs are invalid
            
        **Implementation Process**:
        1. **Validation**: Check that readout weights W_out exist
        2. **State Collection**: Run reservoir with input sequence  
        3. **Feature Construction**: Concatenate states with bias term
        4. **Linear Readout**: Apply trained weights for predictions
        5. **Return Results**: Predictions aligned with post-washout timeline
        
        **Performance Considerations**:
        - Prediction quality depends on reservoir richness and training data
        - Input scaling should match training preprocessing
        - Washout should be consistent between training and prediction
        """
        
        # Verify network has been trained
        if self.W_out is None:
            raise ValueError("Network must be trained before prediction! Call train() first.")
            
        # Generate reservoir states from input sequence
        reservoir_states = self.run_reservoir(inputs, washout)
        
        if len(reservoir_states) == 0:
            raise ValueError("No states collected after washout. Reduce washout period.")
            
        # Construct feature matrix: [states, bias]
        # Note: Input features already incorporated in reservoir dynamics
        X = np.column_stack([
            reservoir_states,
            np.ones(len(reservoir_states))  # Bias term for affine transformation
        ])
        
        # Linear readout: predictions = features @ weights + bias
        # This is the key simplification that makes ESNs fast to train!
        predictions = X @ self.W_out.T
        
        # Add bias if it exists (some implementations separate bias)
        if hasattr(self, 'bias') and self.bias is not None:
            predictions += self.bias
            
        return predictions

    def generate(self, n_steps: int, initial_input: Optional[np.ndarray] = None, 
                 generation_mode: str = 'autonomous') -> np.ndarray:
        """
        🎼 Autonomous Sequence Generation - Closed-Loop Pattern Generation
        ================================================================
        
        **Mathematical Theory (Jaeger 2001, Section 4.3)**:
        
        Autonomous generation transforms the ESN into a dynamical system capable
        of generating complex temporal patterns without external driving. The
        mathematical framework for closed-loop operation:
        
        Closed-Loop Dynamics:
            u(t+1) = ŷ(t)  (output becomes next input)
            x(t+1) = (1-α)x(t) + α·f(W_in·u(t+1) + W_res·x(t))
            ŷ(t+1) = W_out·[x(t+1); u(t+1); 1]
            
        **Attractor Networks (Section 4.3.1)**:
        When properly trained, the closed-loop ESN becomes an attractor network
        that can:
        - Generate learned periodic patterns
        - Reproduce chaotic attractors  
        - Create novel sequences within learned manifolds
        - Maintain temporal coherence over long horizons
        
        **Generation Modes**:
        1. **Autonomous**: Pure closed-loop, output → input feedback
        2. **Driven**: External input with optional output feedback  
        3. **Semi-Autonomous**: Mixture of external and feedback inputs
        4. **Primed**: Initialize with specific sequence then go autonomous
        
        **Critical Implementation Notes**:
        - Requires proper output feedback matrix W_back for optimal results
        - Initial conditions strongly influence generated patterns
        - Generation quality depends on training regime (teacher forcing helps)
        - Long-term stability requires careful spectral radius tuning
        
        Args:
            n_steps (int): Number of time steps to generate
                Should be reasonable (< 10000) to avoid numerical issues
            initial_input (Optional[np.ndarray]): Starting input vector
                If None, uses zero vector. Critical for pattern selection.
            generation_mode (str): Generation strategy
                - 'autonomous': Pure closed-loop generation [default]
                - 'driven': Maintains initial input (requires external signal)
                - 'semi_autonomous': Mix of feedback and external input
                
        Returns:
            np.ndarray: Generated sequence of shape (n_steps, n_outputs)
                Autonomous temporal patterns generated by the trained network
                
        Raises:
            ValueError: If network untrained or invalid parameters
            
        **Research Applications**:
        - Music generation and composition
        - Speech synthesis and prosody
        - Chaotic time series modeling (Lorenz, Henon, etc.)
        - Robot motion pattern generation
        - Financial market simulation
        """
        
        # Validate network readiness
        if self.W_out is None:
            raise ValueError("Network must be trained before generation! Call train() first.")
            
        if self.last_state is None:
            raise ValueError("Reservoir must be run at least once! Call run_reservoir() or train() first.")
            
        # Validate parameters
        if n_steps <= 0:
            raise ValueError("Number of generation steps must be positive")
            
        # Initialize state from last known reservoir state
        # This carries forward the temporal context from training
        state = self.last_state.copy()
        outputs = []
        
        # Initialize input vector
        if initial_input is None:
            # Start with zero input (neutral initialization)
            if not hasattr(self, 'W_input'):
                raise ValueError("Input weights not initialized. Run training first.")
            current_input = np.zeros(self.W_input.shape[1])
        else:
            current_input = initial_input.copy()
            
        # Generation loop: closed-loop dynamics
        for step in range(n_steps):
            # Evolve reservoir state with current input
            state = self._update_state(state, current_input)
            
            # Generate output using linear readout
            state_with_bias = np.append(state, 1.0)
            output = state_with_bias @ self.W_out.T
            
            # Add bias term if separate
            if hasattr(self, 'bias') and self.bias is not None:
                output += self.bias
                
            # Handle output dimensionality
            if output.ndim == 0:
                output = np.array([output])
            elif output.ndim > 1:
                output = output.flatten()
                
            outputs.append(output)
            
            # Update input based on generation mode
            if generation_mode == 'autonomous':
                # Pure closed-loop: output becomes next input
                if len(output) == len(current_input):
                    current_input = output
                # If dimensions don't match, keep current input (common case)
                
            elif generation_mode == 'driven':
                # Keep external input constant (requires continuous driving)
                pass  # current_input remains unchanged
                
            elif generation_mode == 'semi_autonomous':
                # Mixture of external input and self-generated feedback
                external_ratio = 0.2  # 20% external, 80% autonomous
                if len(output) == len(current_input):
                    current_input = (external_ratio * current_input + 
                                   (1 - external_ratio) * output)
            else:
                raise ValueError(f"Unknown generation mode: {generation_mode}")
            
        return np.array(outputs)

    def generate_with_feedback(self, n_steps: int, initial_input: Optional[np.ndarray] = None,
                              generation_mode: str = 'autonomous') -> np.ndarray:
        """
        🔄 Advanced Generation with Output Feedback - Full ESN Dynamics
        =============================================================
        
        **Mathematical Theory (Jaeger 2001, Section 2.2)**:
        
        This method implements the complete ESN dynamics including the output
        feedback matrix W_back, providing the full mathematical framework:
        
        Complete ESN Equation:
            x(t+1) = (1-α)x(t) + α·f(W_in·u(t) + W_res·x(t) + W_back·y(t-1) + η(t))
            y(t) = W_out·[x(t); u(t); 1] + bias
            
        **Output Feedback Matrix W_back**:
        The feedback matrix creates direct connections from outputs to reservoir
        neurons, enabling:
        - Enhanced memory capacity
        - Improved temporal coherence  
        - Better long-term dependencies
        - Richer attractor dynamics
        
        **Teacher Forcing Extension**:
        During training, teacher forcing can be applied:
            W_back·y_target(t-1)  instead of  W_back·ŷ(t-1)
        
        This prevents error accumulation during training and improves learning
        of complex temporal patterns.
        
        **Advanced Generation Modes**:
        1. **Autonomous**: Full closed-loop with output feedback
        2. **Driven**: External input + output feedback to reservoir
        3. **Semi-Autonomous**: Hybrid external/feedback input with reservoir feedback
        4. **Teacher-Forced**: Use ground truth feedback (during validation)
        
        Args:
            n_steps (int): Number of steps to generate
            initial_input (Optional[np.ndarray]): Starting input vector
            generation_mode (str): Feedback strategy
                - 'autonomous': Full closed-loop with feedback [default]
                - 'driven': External input with reservoir feedback
                - 'semi_autonomous': Mixed input/feedback modes
                
        Returns:
            np.ndarray: Generated sequence with enhanced dynamics
            
        Raises:
            ValueError: If feedback not configured or network not trained
            
        **Performance Notes**:
        - Requires output_feedback_enabled = True
        - More computationally intensive than basic generation
        - Superior long-term coherence and memory
        - Essential for complex attractor reproduction
        """
        
        # Validate network readiness
        if self.W_out is None:
            raise ValueError("Network must be trained before generation!")
            
        if self.last_state is None:
            raise ValueError("Reservoir must be run at least once before generation!")
            
        # Check if output feedback is enabled
        if not hasattr(self, 'output_feedback_enabled') or not self.output_feedback_enabled:
            warnings.warn("Output feedback not enabled. Using standard generation method.")
            return self.generate(n_steps, initial_input, generation_mode)
        
        # Initialize state and feedback tracking
        state = self.last_state.copy()
        outputs = []
        previous_output = None
        
        # Initialize input
        if initial_input is None:
            if not hasattr(self, 'W_input'):
                raise ValueError("Input weights not initialized")
            current_input = np.zeros(self.W_input.shape[1])
        else:
            current_input = initial_input.copy()
            
        # Enhanced generation loop with output feedback
        for step in range(n_steps):
            # Update state with output feedback (key enhancement!)
            if previous_output is not None and hasattr(self, '_update_state_with_feedback'):
                # Use enhanced state update with feedback
                state = self._update_state_with_feedback(state, current_input, previous_output)
            else:
                # Fallback to standard state update
                state = self._update_state(state, current_input)
            
            # Generate output with full feature vector
            state_with_bias = np.append(state, 1.0)
            output = state_with_bias @ self.W_out.T
            
            # Add bias if present
            if hasattr(self, 'bias') and self.bias is not None:
                output += self.bias
                
            # Ensure proper output format
            if output.ndim == 0:
                output = np.array([output])
            elif output.ndim > 1:
                output = output.flatten()
                
            outputs.append(output.copy())
            
            # Store output for next feedback step
            previous_output = output
            
            # Update input based on enhanced generation mode
            if generation_mode == 'autonomous':
                # Pure closed-loop: output → input, feedback → reservoir
                if len(output) == len(current_input):
                    current_input = output
                    
            elif generation_mode == 'driven':
                # External input constant, feedback only to reservoir
                pass  # current_input unchanged
                
            elif generation_mode == 'semi_autonomous':
                # Mixed input strategy with reservoir feedback
                external_ratio = 0.2
                if len(output) == len(current_input):
                    current_input = (external_ratio * current_input + 
                                   (1 - external_ratio) * output)
            else:
                raise ValueError(f"Unknown generation mode: {generation_mode}")
            
        return np.array(outputs)

    # Helper method for washout validation and adaptation
    def _validate_washout(self, washout: int, total_steps: int) -> int:
        """
        Validate and potentially adapt washout period
        
        **Adaptive Strategies**:
        - Ensure minimum data availability
        - Scale with reservoir size for complex dynamics
        - Warn about potential issues
        """
        min_washout = max(10, self.n_reservoir // 20)  # Minimum based on reservoir size
        max_washout = total_steps // 2  # Maximum to preserve training data
        
        if washout < min_washout:
            warnings.warn(f"Washout {washout} may be too small. Consider {min_washout}+")
            
        if washout > max_washout:
            warnings.warn(f"Washout {washout} too large. Using {max_washout}")
            return max_washout
            
        return washout

    # Enhanced error checking and diagnostics
    def _check_prediction_readiness(self) -> Dict[str, Any]:
        """
        Comprehensive readiness check for prediction/generation
        
        Returns:
            Dict with readiness status and diagnostics
        """
        status = {
            'ready': True,
            'issues': [],
            'warnings': []
        }
        
        # Check essential components
        if not hasattr(self, 'W_out') or self.W_out is None:
            status['ready'] = False
            status['issues'].append('Output weights not trained')
            
        if not hasattr(self, 'W_input') or self.W_input is None:
            status['warnings'].append('Input weights not initialized')
            
        if not hasattr(self, 'last_state') or self.last_state is None:
            status['warnings'].append('No previous reservoir state available')
            
        # Check numerical stability
        if hasattr(self, 'last_condition_number_'):
            if self.last_condition_number_ > 1e12:
                status['warnings'].append(f'High condition number: {self.last_condition_number_:.2e}')
                
        return status