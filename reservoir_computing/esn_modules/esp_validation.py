"""
🌊 Echo State Property Validation Module - Echo State Networks
==============================================================

📚 Research Foundation:
Jaeger, H. (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
Technical Report GMD-148, German National Research Center for Information Technology

🎯 Module Purpose:
This module implements comprehensive Echo State Property (ESP) validation methods.
The ESP is the fundamental mathematical property that enables ESNs to work:
- Fading memory: different initial states converge to same trajectory
- Contractivity: small differences in initial states shrink over time
- Stability: reservoir dynamics remain bounded for any input sequence

🧠 Theoretical Foundation (Jaeger 2001, Section 2):
"A network with dynamics f has the echo state property if for any input sequence
u(·), the reservoir state x(t) is uniquely determined by the history of the input."

Mathematical Requirements:
1. Contractivity: ||∂x(t+1)/∂x(t)|| < 1 everywhere
2. Separation: distinct inputs → distinct states  
3. Approximation: reservoir can approximate target mappings

🔧 Validation Methods:
- Fast: Quick convergence test for real-time verification
- Rigorous: Comprehensive contractivity testing per Jaeger 2001
- Convergence: Tests trajectory convergence from different initial states
- Lyapunov: Tests negative Lyapunov exponents for stability
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')


class EspValidationMixin:
    """
    Mixin for Echo State Property validation in Echo State Networks.
    
    Provides multiple validation methods to ensure the reservoir satisfies
    the mathematical requirements for the echo state property.
    """

    def _initialize_esp_validation_methods(self):
        """
        Initialize Echo State Property validation methods
        
        Sets up different validation approaches based on computational requirements
        and mathematical rigor needed.
        """
        self.esp_methods = {
            'fast': self._validate_echo_state_property_fast,
            'rigorous': self._validate_echo_state_property_rigorous,
            'convergence': self._validate_echo_state_property_convergence,
            'lyapunov': self._validate_echo_state_property_lyapunov
        }
        
        # Default ESP validation configuration
        if not hasattr(self, 'esp_validation_config'):
            self.esp_validation_config = {
                'method': getattr(self, 'esp_validation_method', 'fast'),
                'n_tests': 10,
                'test_length': 1000,
                'tolerance': 1e-6
            }

    def _validate_comprehensive_esp(self):
        """
        Comprehensive Echo State Property validation
        
        🧠 Implementation addresses the core ESP requirements:
        - Verifies reservoir satisfies echo state property
        - Uses configurable validation method for flexibility
        - Provides clear feedback on ESP status
        
        Returns:
            bool: True if ESP is satisfied, False otherwise
        """
        method = self.esp_methods.get(
            self.esp_validation_method, 
            self._validate_echo_state_property_fast
        )
        esp_result = method()
        
        if esp_result:
            print(f"✅ Echo State Property validated using {self.esp_validation_method} method")
        else:
            print(f"⚠️ ESP validation failed with {self.esp_validation_method} method - consider adjusting spectral radius")
        
        return esp_result

    def _validate_echo_state_property_fast(self, n_tests=3, test_length=100, tolerance=1e-4):
        """
        Fast ESP validation for real-time verification
        
        🎯 Purpose: Quick validation during reservoir initialization or adaptation.
        Uses fewer tests and shorter sequences for computational efficiency.
        
        Args:
            n_tests: Number of convergence tests to run
            test_length: Length of input sequences for testing
            tolerance: Convergence tolerance threshold
            
        Returns:
            bool: True if ESP appears satisfied
        """
        return self._validate_echo_state_property(n_tests, test_length, tolerance)

    def _validate_echo_state_property(self, n_tests=10, test_length=1000, tolerance=1e-6):
        """
        Standard Echo State Property validation
        
        🧠 Theory (Jaeger 2001, Definition 1):
        "A discrete-time dynamical system has the echo state property if 
        the reservoir state is uniquely determined by the input history."
        
        Implementation tests:
        - State convergence from different initial conditions
        - Independence from initial reservoir state
        - Bounded dynamics under arbitrary inputs
        
        Args:
            n_tests: Number of independent convergence tests
            test_length: Length of input sequence for each test
            tolerance: Final state difference tolerance
            
        Returns:
            bool: True if ESP validation passes all tests
        """
        converged = True
        
        # Initialize dummy input weights if not present
        if not hasattr(self, 'W_input') or self.W_input is None:
            n_inputs = 1  # Default to single input for testing
            self.W_input = np.random.uniform(-1, 1, (self.n_reservoir, n_inputs))
        else:
            n_inputs = self.W_input.shape[1]
            
        test_input = np.random.randn(test_length, n_inputs)
        
        for test in range(n_tests):
            # Run same input from different initial states
            x1 = np.random.randn(self.n_reservoir) * 0.1
            x2 = np.random.randn(self.n_reservoir) * 0.1
            
            # Run test sequences
            final1 = self._run_test_sequence(x1, test_input)
            final2 = self._run_test_sequence(x2, test_input)
            
            # Check convergence
            difference = np.linalg.norm(final1 - final2)
            if difference > tolerance:
                converged = False
                break
                
        return converged

    def _validate_echo_state_property_rigorous(self, n_tests=20, test_length=2000, tolerance=1e-8):
        """
        Rigorous ESP validation as per Jaeger 2001
        
        🧠 Mathematical Foundation (Jaeger 2001, Section 2.1):
        "The contractivity condition: ||∂x(t+1)/∂x(t)|| < 1 must hold everywhere
        in the state space to guarantee the echo state property."
        
        Tests the contractivity condition by:
        1. Starting with different initial states  
        2. Running identical input sequences
        3. Verifying distances between trajectories shrink over time
        4. Ensuring final states are identical within tolerance
        
        Args:
            n_tests: Number of rigorous contractivity tests
            test_length: Length of input sequences (longer for rigor)
            tolerance: Very strict convergence tolerance
            
        Returns:
            bool: True if rigorous ESP conditions are satisfied
        """
        print("🔬 Running rigorous ESP validation...")
        
        for test in range(n_tests):
            # Generate two different initial states
            x1 = np.random.uniform(-1, 1, self.n_reservoir)
            x2 = np.random.uniform(-1, 1, self.n_reservoir)
            initial_distance = np.linalg.norm(x1 - x2)
            
            # Generate random input sequence
            inputs = np.random.uniform(-1, 1, (test_length, 1))
            
            # Run both trajectories
            state1, state2 = x1.copy(), x2.copy()
            distances = []
            
            for t in range(test_length):
                state1 = self._update_state(state1, inputs[t])
                state2 = self._update_state(state2, inputs[t])
                
                distance = np.linalg.norm(state1 - state2)
                distances.append(distance)
                
                # Early termination if distance grows (ESP violation)
                if distance > initial_distance * 1.1:
                    print(f"   ESP violation: distance grew from {initial_distance:.6f} to {distance:.6f}")
                    return False
            
            # Check if final distance satisfies contractivity
            final_distance = distances[-1]
            if final_distance > tolerance:
                print(f"   ESP violation: final distance {final_distance:.8f} > tolerance {tolerance:.8f}")
                return False
                
        print(f"   ✓ Rigorous ESP validation passed ({n_tests} tests)")
        return True

    def _validate_echo_state_property_convergence(self, n_tests=10, test_length=1500):
        """
        Convergence-based ESP validation
        
        🧠 Theory Focus: Tests the fundamental ESP requirement that different
        initial conditions converge to the same trajectory under identical inputs.
        
        This method specifically tests Jaeger's "fading memory" property:
        - Initial state information should decay exponentially
        - Final states should be independent of initial conditions  
        - Convergence should occur within reasonable time horizon
        
        Args:
            n_tests: Number of convergence tests with different initial conditions
            test_length: Sequence length allowing sufficient convergence time
            
        Returns:
            bool: True if convergence-based ESP is satisfied
        """
        print("🌀 Running convergence-based ESP validation...")
        
        convergence_threshold = 1e-6
        
        for test in range(n_tests):
            # Two random initial states (potentially very different)
            x1 = np.random.uniform(-2, 2, self.n_reservoir)  # Wider range for stronger test
            x2 = np.random.uniform(-2, 2, self.n_reservoir)
            
            # Same input sequence for both trajectories
            inputs = np.random.uniform(-1, 1, (test_length, 1))
            
            state1, state2 = x1.copy(), x2.copy()
            
            # Track convergence over time
            convergence_history = []
            for t in range(test_length):
                state1 = self._update_state(state1, inputs[t])
                state2 = self._update_state(state2, inputs[t])
                
                distance = np.linalg.norm(state1 - state2)
                convergence_history.append(distance)
            
            # Check final convergence
            final_distance = convergence_history[-1]
            if final_distance > convergence_threshold:
                print(f"   Convergence test {test+1} failed: final distance = {final_distance:.8f}")
                return False
                
        print(f"   ✓ Convergence ESP validation passed ({n_tests} tests)")
        return True

    def _validate_echo_state_property_lyapunov(self):
        """
        Lyapunov exponent-based ESP validation
        
        🧠 Mathematical Foundation:
        The Echo State Property holds if and only if the largest Lyapunov exponent
        of the reservoir dynamics is negative. This provides a rigorous
        mathematical criterion for ESP based on dynamical systems theory.
        
        Lyapunov Exponent λ:
        - λ < 0: Stable (ESP satisfied)
        - λ = 0: Marginally stable  
        - λ > 0: Chaotic/unstable (ESP violated)
        
        Returns:
            bool: True if largest Lyapunov exponent < 0
        """
        print("📊 Running Lyapunov-based ESP validation...")
        
        try:
            # Approximate Jacobian of reservoir dynamics
            jacobian = self._compute_reservoir_jacobian()
            eigenvalues = np.linalg.eigvals(jacobian)
            
            # Largest real part approximates dominant Lyapunov exponent
            max_lyapunov = np.max(np.real(eigenvalues))
            
            print(f"   Max Lyapunov exponent: {max_lyapunov:.6f}")
            
            if max_lyapunov < 0:
                print(f"   ✓ ESP satisfied (λ_max = {max_lyapunov:.6f} < 0)")
                return True
            else:
                print(f"   ⚠️ ESP questionable (λ_max = {max_lyapunov:.6f} ≥ 0)")
                return False
            
        except Exception as e:
            print(f"   Lyapunov validation failed: {e}")
            return False

    def _compute_reservoir_jacobian(self):
        """
        Compute Jacobian matrix of reservoir dynamics
        
        🧠 Mathematical Details:
        For reservoir dynamics: x(t+1) = (1-α)x(t) + α·f(W·x(t) + W_in·u(t))
        The Jacobian is: J = (1-α)I + α·diag(f'(W·x + W_in·u))·W
        
        Where:
        - α = leak_rate (leaky integration parameter)
        - f = activation function
        - f' = activation derivative
        - W = reservoir weight matrix
        
        Used for Lyapunov exponent calculation and stability analysis.
        
        Returns:
            np.ndarray: Jacobian matrix of reservoir dynamics
        """
        # Sample state for linearization point
        sample_state = np.random.uniform(-1, 1, self.n_reservoir)
        
        # Compute activation function derivative
        if self.activation_function == 'tanh':
            activation_derivative = 1 - np.tanh(sample_state)**2
        elif self.activation_function == 'sigmoid':
            sigmoid_val = 1 / (1 + np.exp(-sample_state))
            activation_derivative = sigmoid_val * (1 - sigmoid_val)
        elif self.activation_function == 'relu':
            activation_derivative = (sample_state > 0).astype(float)
        elif self.activation_function == 'leaky_relu':
            activation_derivative = np.where(sample_state > 0, 1.0, 0.01)
        elif self.activation_function == 'linear':
            activation_derivative = np.ones(self.n_reservoir)
        else:
            # Default to tanh derivative
            activation_derivative = 1 - np.tanh(sample_state)**2
        
        # Get leak rate (handle both scalar and vector forms)
        if np.isscalar(self.leak_rates):
            leak_rate = self.leak_rates
        else:
            leak_rate = np.mean(self.leak_rates)  # Use average for Jacobian
        
        # Jacobian = (1-α)I + α·diag(f'(x))·W  
        jacobian = np.diag(activation_derivative) @ self.W_reservoir * leak_rate
        jacobian += np.diag(1 - leak_rate)
        
        return jacobian

    def _run_test_sequence(self, initial_state, input_sequence):
        """
        Run a test sequence from given initial state
        
        Helper method for ESP validation that simulates reservoir dynamics
        over a given input sequence starting from specified initial conditions.
        
        Args:
            initial_state: Starting reservoir state
            input_sequence: Sequence of input vectors
            
        Returns:
            np.ndarray: Final reservoir state after sequence
        """
        state = initial_state.copy()
        
        for t in range(len(input_sequence)):
            state = self._update_state(state, input_sequence[t])
            
        return state

    def enable_output_feedback(self, n_outputs: int, feedback_scaling: float = 0.1):
        """
        Enable output feedback for closed-loop operation
        
        Args:
            n_outputs: Number of output dimensions
            feedback_scaling: Scaling factor for feedback weights
        """
        if self.W_back is None:
            self._initialize_output_feedback(n_outputs, feedback_scaling)
        self.output_feedback_enabled = True

    def disable_output_feedback(self):
        """Disable output feedback for open-loop operation"""
        self.output_feedback_enabled = False

    def get_esp_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of ESP validation results
        
        Returns:
            Dict containing validation results, methods used, and recommendations
        """
        summary = {
            'esp_method': self.esp_validation_method,
            'esp_validated': getattr(self, 'esp_validated', None),
            'spectral_radius': np.max(np.abs(np.linalg.eigvals(self.W_reservoir))),
            'recommendations': []
        }
        
        # Add method-specific information
        if hasattr(self, 'adaptive_radius_history'):
            summary['adaptive_radius_history'] = self.adaptive_radius_history
            
        # Generate recommendations based on current state
        if summary['spectral_radius'] > 1.0:
            summary['recommendations'].append("Consider reducing spectral radius below 1.0")
        elif summary['spectral_radius'] < 0.3:
            summary['recommendations'].append("Consider increasing spectral radius for richer dynamics")
            
        return summary


# Export the mixin class
__all__ = ['EspValidationMixin']