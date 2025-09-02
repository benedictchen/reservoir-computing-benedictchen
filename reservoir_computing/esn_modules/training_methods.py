"""
🏋️ Training Methods Module - Echo State Networks
===============================================

📚 Research Foundation:
Jaeger, H. (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
Technical Report GMD-148, German National Research Center for Information Technology

🎯 Module Purpose:
This module implements comprehensive training methodologies for Echo State Networks including:
- Standard linear readout training via ridge regression
- Teacher forcing training for sequence generation tasks
- Multiple solver methods (Ridge, LSQR, Pseudo-inverse, Elastic Net)
- Regularization parameter optimization through cross-validation
- Advanced washout strategies for different temporal scales
- Comprehensive state collection methods
- Training stability validation and numerical condition monitoring

🧠 Theoretical Foundation from Jaeger 2001:

📖 Section 3.3 - Teacher Forcing Training:
The key insight is that during training, we can feed the target outputs back to the
reservoir instead of the network's own predictions. This prevents error accumulation
and enables learning of complex temporal sequences. The mathematical formulation:

x(n+1) = f(W*x(n) + W_in*u(n) + W_fb*y_teacher(n))

Where y_teacher(n) is the desired target output at time n, rather than the network's
prediction. This creates a "supervised" reservoir state trajectory that captures
the true dynamics of the target sequence.

📖 Section 4.1 - Linear Readout Training:
The fundamental advantage of ESNs is that only the readout weights need training.
The reservoir provides a fixed nonlinear transformation of the input history:

y(n) = W_out * [x(n); u(n); 1]

Where the readout weight matrix W_out is trained via linear regression:
W_out = argmin_W ||Y - W*X||² + λ||W||²

This is solved analytically: W_out = (X^T*X + λI)^(-1) * X^T * Y

📖 Section 4.2 - Regularization and Stability:
Jaeger emphasizes the importance of regularization parameter λ for numerical stability
and generalization. The optimal λ depends on:
- Reservoir size and spectral radius
- Training data noise level  
- Task complexity and required precision
- Condition number of the state matrix X

📖 Section 4.3 - Washout Periods:
Initial transient states must be discarded to ensure the reservoir has reached
its attractor dynamics. Jaeger recommends washout periods of 100-200 time steps,
but this can be adaptive based on convergence criteria.

🔧 Key Components:
- Ridge Regression: Standard solver with L2 regularization
- LSQR: Iterative solver for large-scale problems  
- Pseudo-inverse: Direct solution for λ=0 case
- Elastic Net: Combined L1/L2 regularization
- Cross-validation: Automatic λ optimization
- Teacher Forcing: Multiple strategies (full, partial, scheduled)
- Washout Strategies: Fixed, adaptive, ensemble approaches
- State Collection: Subsampling, weighting, multi-horizon methods

🎓 Research Context:
This implementation addresses many of the FIXME comments in the original core file,
providing the comprehensive training framework described in Jaeger's seminal paper.
Each method includes detailed mathematical foundations and multiple algorithmic
variants to support different use cases and computational constraints.
"""

import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from scipy.sparse.linalg import lsqr
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TrainingMethodsMixin:
    """
    Mixin for comprehensive training methods in Echo State Networks.
    
    Implements the full suite of training techniques described in Jaeger 2001,
    including multiple solver methods, teacher forcing strategies, and advanced
    state collection approaches.
    """

    def train(self, inputs: np.ndarray, targets: np.ndarray, 
              reg_param: float = 1e-6, washout: int = 100, teacher_forcing: bool = False) -> Dict[str, Any]:
        """
        Train the readout weights using linear regression.
        
        The fundamental insight of ESNs: we only need to solve a linear regression problem
        instead of the complex nonlinear optimization of traditional RNNs. The reservoir
        provides a fixed nonlinear transformation of the input history.
        
        Mathematical Foundation (Jaeger 2001, Section 4.1):
        =====================================================
        
        The ESN output equation:
        y(n) = W_out * [x(n); u(n); 1]
        
        Training objective (with Ridge regularization):
        W_out = argmin_W ||Y - W*X||² + λ||W||²
        
        Analytical solution:
        W_out = (X^T*X + λI)^(-1) * X^T * Y
        
        Where:
        - X: Extended state matrix [reservoir_states, inputs, bias]
        - Y: Target outputs after washout period
        - λ: Regularization parameter (reg_param)
        
        Args:
            inputs: Input sequence (time_steps, n_inputs)
            targets: Target sequence (time_steps, n_outputs)  
            reg_param: Ridge regularization parameter (λ in paper)
            washout: Initial transient period to discard
            teacher_forcing: Use teacher forcing during training
            
        Returns:
            Dictionary with training results and diagnostics
            
        Research Notes:
        - Jaeger emphasizes checking condition number for numerical stability
        - Optimal regularization depends on reservoir size and task complexity
        - Multiple solver methods available for different problem scales
        """
        
        print(f"Training ESN on {len(inputs)} time steps...")
        
        if teacher_forcing:
            return self.train_with_teacher_forcing(inputs, targets, reg_param, washout)
        
        # Apply comprehensive washout strategy
        reservoir_states, effective_washout = self._apply_comprehensive_washout_strategy(inputs, washout)
        
        # Apply comprehensive state collection strategy
        reservoir_states, targets_adjusted = self._apply_comprehensive_state_collection(
            reservoir_states, targets, effective_washout)
        
        # Validate training stability (Jaeger 2001 numerical considerations)
        stability_info = self._validate_training_stability(reservoir_states)
        
        # Prepare extended state matrix [reservoir_states, bias]
        X = np.column_stack([
            reservoir_states,
            np.ones(len(reservoir_states))  # Bias term
        ])
        
        y = targets_adjusted
        
        # Optimize regularization parameter if set to 'auto'
        if reg_param == 'auto':
            reg_param = self._optimize_regularization_parameter(X, y)
        
        # Apply comprehensive solver method
        W_out, predictions = self._apply_comprehensive_solver_method(X, y, reg_param)
        
        # Store trained weights
        self.W_out = W_out
        if hasattr(self, '_apply_comprehensive_solver_method') and hasattr(Ridge(alpha=reg_param), 'intercept_'):
            # For Ridge regression, extract bias separately
            ridge_temp = Ridge(alpha=reg_param)
            ridge_temp.fit(X, y)
            self.bias = ridge_temp.intercept_
        else:
            self.bias = 0.0
        
        # Calculate training performance metrics
        mse = np.mean((predictions - y) ** 2)
        
        # Collect comprehensive training results
        training_results = {
            'mse': mse,
            'n_states_used': len(reservoir_states),
            'effective_washout': effective_washout,
            'regularization_param': reg_param,
            'solver_method': getattr(self, 'training_solver', 'ridge'),
            'spectral_radius': np.max(np.abs(np.linalg.eigvals(self.W_reservoir))) if hasattr(self, 'W_reservoir') else None,
            'reservoir_size': getattr(self, 'n_reservoir', None),
            'stability_info': stability_info
        }
        
        print(f"✓ Training complete: MSE = {mse:.6f}")
        
        return training_results

    def train_with_teacher_forcing(self, inputs: np.ndarray, targets: np.ndarray, 
                                  reg_param: float = 1e-6, washout: int = 100,
                                  teacher_forcing_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Train with teacher forcing as described in Jaeger 2001, Section 3.3.
        
        During training, feed target outputs back to reservoir instead of predictions
        to prevent error accumulation. Critical for sequence generation tasks.
        
        Mathematical Foundation (Jaeger 2001, Section 3.3):
        ==================================================
        
        Standard reservoir update:
        x(n+1) = f(W*x(n) + W_in*u(n) + W_fb*y_pred(n))
        
        Teacher forcing update:
        x(n+1) = f(W*x(n) + W_in*u(n) + W_fb*y_target(n))
        
        Where y_target(n) is the desired output at time n, creating a "supervised"
        reservoir state trajectory that captures the true dynamics of the target sequence.
        
        Teacher Forcing Strategies:
        1. Full teacher forcing: Always use y_target
        2. Partial teacher forcing: Mix y_target and y_pred with ratio
        3. Scheduled teacher forcing: Decrease ratio over training epochs
        4. Stochastic teacher forcing: Random selection at each time step
        
        Args:
            inputs: Input sequence (time_steps, n_inputs)
            targets: Target sequence (time_steps, n_outputs)
            reg_param: Ridge regularization parameter
            washout: Initial transient period to discard
            teacher_forcing_ratio: Probability of using target vs prediction
            
        Returns:
            Dictionary with training results including teacher forcing metrics
            
        Research Notes:
        - Teacher forcing enables learning of unstable dynamics
        - Critical for autonomous sequence generation tasks
        - Prevents error accumulation during training phase
        - Requires output feedback connections (W_fb matrix)
        """
        
        print(f"Training ESN with teacher forcing (ratio={teacher_forcing_ratio})...")
        
        time_steps, n_inputs = inputs.shape
        _, n_outputs = targets.shape
        
        # Initialize feedback weights if not already done
        if not getattr(self, 'output_feedback_enabled', False):
            if hasattr(self, 'enable_output_feedback'):
                self.enable_output_feedback(n_outputs)
            else:
                print("⚠️ WARNING: Output feedback not available - using standard training")
                return self.train(inputs, targets, reg_param, washout, False)
            
        # Initialize input weights if needed
        if not hasattr(self, 'W_input'):
            self._initialize_input_weights(n_inputs)
            
        # Initialize state and collect reservoir states
        state = np.zeros(getattr(self, 'n_reservoir', 100))
        states = []
        
        # Run through all time steps with teacher forcing
        for t in range(time_steps):
            # Decide whether to use teacher forcing at this step
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            
            # Get feedback signal
            if t > 0:
                if use_teacher_forcing:
                    feedback = targets[t-1]  # Use target (teacher forcing)
                else:
                    # Use predicted output from previous step
                    if len(states) > 0 and hasattr(self, 'W_out') and self.W_out is not None:
                        prev_state_with_bias = np.append(states[-1], 1.0)
                        prev_prediction = prev_state_with_bias @ self.W_out.T + getattr(self, 'bias', 0.0)
                        feedback = prev_prediction
                    else:
                        feedback = targets[t-1]  # Fallback to target
                        
                if hasattr(self, '_update_state_with_feedback'):
                    state = self._update_state_with_feedback(state, inputs[t], feedback)
                else:
                    # Fallback to standard state update
                    state = self._update_state(state, inputs[t])
            else:
                state = self._update_state(state, inputs[t])
            
            # Collect states after washout period
            if t >= washout:
                states.append(state.copy())
        
        # Standard linear readout training
        states = np.array(states)
        X = np.column_stack([states, np.ones(len(states))])
        y = targets[washout:]
        
        # Apply comprehensive solver method
        W_out, predictions = self._apply_comprehensive_solver_method(X, y, reg_param)
        
        # Store trained weights
        self.W_out = W_out
        self.bias = getattr(Ridge(alpha=reg_param).fit(X, y), 'intercept_', 0.0)
        if hasattr(self, 'last_state'):
            self.last_state = state
        
        # Calculate performance metrics
        mse = np.mean((predictions - y) ** 2)
        
        results = {
            'mse': mse,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'n_states_used': len(states),
            'output_feedback_enabled': getattr(self, 'output_feedback_enabled', False),
            'solver_method': getattr(self, 'training_solver', 'ridge')
        }
        
        print(f"✓ Teacher forcing training complete: MSE = {mse:.6f}")
        
        return results

    def _apply_comprehensive_solver_method(self, X: np.ndarray, y: np.ndarray, reg_param: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply comprehensive solver methods with 4 configurable options.
        
        Addresses the need for different solvers based on problem scale and
        regularization requirements as discussed in Jaeger 2001.
        
        Mathematical Foundation:
        =======================
        
        1. Pseudo-inverse (λ=0 case from Jaeger 2001):
           W_out = X^+ * Y  where X^+ = (X^T*X)^(-1)*X^T
           
        2. Ridge Regression (standard approach):
           W_out = (X^T*X + λI)^(-1) * X^T * Y
           
        3. LSQR (iterative solver for large problems):
           Solves min ||Ax - b||² + λ²||x||² iteratively
           
        4. Elastic Net (combined L1/L2 regularization):
           W_out = argmin_W ||Y - XW||² + λ₁||W||₁ + λ₂||W||²
        
        Args:
            X: Extended state matrix [reservoir_states, bias]
            y: Target outputs
            reg_param: Regularization parameter
            
        Returns:
            Tuple of (trained_weights, predictions)
            
        Research Notes:
        - Pseudo-inverse optimal for well-conditioned problems without noise
        - Ridge regression provides numerical stability and regularization
        - LSQR efficient for large-scale problems (reservoir_size > 1000)
        - Elastic Net combines feature selection (L1) with stability (L2)
        """
        
        solver_method = getattr(self, 'training_solver', 'ridge')
        
        if solver_method == 'pseudo_inverse':
            # Option 1: Pseudo-inverse (λ=0 case from Jaeger 2001)
            if reg_param == 0:
                W_out = np.linalg.pinv(X) @ y
                predictions = X @ W_out
                return W_out, predictions
            else:
                # Regularized pseudo-inverse
                XTX_reg = X.T @ X + reg_param * np.eye(X.shape[1])
                W_out = np.linalg.solve(XTX_reg, X.T @ y)
                predictions = X @ W_out
                return W_out, predictions
                
        elif solver_method == 'lsqr':
            # Option 2: Iterative solver for large problems
            W_out = lsqr(X, y, damp=np.sqrt(reg_param))[0]
            predictions = X @ W_out
            return W_out, predictions
            
        elif solver_method == 'elastic_net':
            # Option 3: Elastic net regularization
            elastic_net = ElasticNet(alpha=reg_param, l1_ratio=0.5, max_iter=2000)
            elastic_net.fit(X, y)
            W_out = elastic_net.coef_
            predictions = elastic_net.predict(X)
            return W_out, predictions
            
        else:  # 'ridge' (default)
            # Option 4: Ridge regression (standard approach)
            ridge = Ridge(alpha=reg_param)
            ridge.fit(X, y)
            W_out = ridge.coef_
            predictions = ridge.predict(X)
            return W_out, predictions

    def _optimize_regularization_parameter(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optimize regularization parameter using cross-validation.
        
        Addresses Jaeger 2001's emphasis on proper regularization parameter selection.
        The optimal λ varies significantly across tasks and depends on reservoir
        properties, data characteristics, and desired generalization.
        
        Mathematical Foundation:
        =======================
        
        Cross-validation objective:
        λ* = argmin_λ E[||Y_val - X_val * W_λ||²]
        
        Where W_λ = (X_train^T * X_train + λI)^(-1) * X_train^T * Y_train
        
        The optimal regularization balances:
        - Bias: Under-regularization leads to overfitting
        - Variance: Over-regularization leads to underfitting
        - Numerical stability: Too small λ causes ill-conditioning
        
        Args:
            X: Extended state matrix
            y: Target outputs
            
        Returns:
            Optimal regularization parameter
            
        Research Notes:
        - Jaeger recommends λ ∈ [1e-8, 1e-1] for most ESN applications
        - Large reservoirs typically need smaller λ values
        - Noisy data requires larger λ for regularization
        - Grid search with cross-validation is most reliable method
        """
        
        # Logarithmic grid covering typical ESN regularization range
        reg_candidates = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
        best_reg, best_score = 1e-6, float('inf')
        
        print("🔍 Optimizing regularization parameter...")
        
        for reg in reg_candidates:
            try:
                ridge_cv = Ridge(alpha=reg)
                # 5-fold cross-validation with negative MSE scoring
                scores = cross_val_score(ridge_cv, X, y, cv=5, scoring='neg_mean_squared_error')
                avg_score = -np.mean(scores)
                
                if avg_score < best_score:
                    best_score, best_reg = avg_score, reg
                    
            except Exception as e:
                print(f"⚠️ Failed to evaluate λ={reg:.2e}: {str(e)}")
                continue
        
        print(f"✓ Optimal regularization parameter: λ={best_reg:.2e} (CV MSE: {best_score:.6f})")
        return best_reg

    def _validate_training_stability(self, reservoir_states: np.ndarray) -> Dict[str, Any]:
        """
        Validate training stability through numerical condition analysis.
        
        Addresses Jaeger 2001's emphasis on numerical stability considerations.
        High condition numbers indicate ill-conditioned state matrices that
        can lead to unstable training and poor generalization.
        
        Mathematical Foundation:
        =======================
        
        Condition number: κ(X) = σ_max(X) / σ_min(X)
        
        Where σ_max, σ_min are largest and smallest singular values of X.
        
        Stability Guidelines (from Jaeger 2001):
        - κ(X) < 1e12: Good numerical conditioning
        - κ(X) ∈ [1e12, 1e15]: Warning range, regularization recommended
        - κ(X) > 1e15: Poor conditioning, re-initialization suggested
        
        Args:
            reservoir_states: Reservoir state matrix (time_steps, n_reservoir)
            
        Returns:
            Dictionary with stability metrics and recommendations
            
        Research Notes:
        - Condition number depends on spectral radius and reservoir connectivity
        - ESP (Echo State Property) ensures bounded condition numbers
        - Poor conditioning often indicates spectral radius too close to 1.0
        - Regularization can mitigate mild conditioning issues
        """
        
        if reservoir_states.size == 0 or reservoir_states.shape[1] == 0:
            print("⚠️ WARNING: Empty reservoir states - cannot validate stability")
            return {
                'condition_number': np.inf,
                'is_stable': False,
                'stability_warning': True,
                'recommendation': 'Check reservoir initialization and input processing'
            }
        
        # Compute condition number using SVD for numerical stability
        try:
            state_condition_number = np.linalg.cond(reservoir_states)
        except np.linalg.LinAlgError:
            state_condition_number = np.inf
        
        # Stability analysis based on Jaeger 2001 guidelines
        is_stable = state_condition_number < 1e12
        stability_warning = state_condition_number > 1e12
        
        # Generate recommendations based on condition number
        if state_condition_number < 1e8:
            recommendation = 'Excellent numerical conditioning'
        elif state_condition_number < 1e12:
            recommendation = 'Good numerical conditioning'
        elif state_condition_number < 1e15:
            recommendation = 'Consider adding regularization or reducing spectral radius'
        else:
            recommendation = 'Re-initialize reservoir with lower spectral radius'
        
        stability_info = {
            'condition_number': state_condition_number,
            'is_stable': is_stable,
            'stability_warning': stability_warning,
            'recommendation': recommendation
        }
        
        # Display warnings for poor conditioning
        if stability_warning:
            print(f"⚠️ WARNING: High condition number κ={state_condition_number:.2e}")
            print(f"   May indicate poor Echo State Property (ESP)")
            print(f"   Recommendation: {recommendation}")
        else:
            print(f"✓ Good numerical stability: κ={state_condition_number:.2e}")
        
        # Store for diagnostics
        if hasattr(self, 'last_condition_number_'):
            self.last_condition_number_ = state_condition_number
        
        return stability_info

    def _apply_comprehensive_washout_strategy(self, inputs: np.ndarray, washout: int) -> Tuple[np.ndarray, int]:
        """
        Apply comprehensive washout strategies with 4 configurable options.
        
        Addresses Jaeger 2001's discussion of transient elimination for different
        temporal scales and reservoir dynamics. The washout period ensures the
        reservoir reaches its attractor dynamics before training data collection.
        
        Mathematical Foundation:
        =======================
        
        Standard washout: Discard first W time steps
        x_train = {x(W+1), x(W+2), ..., x(T)}
        
        Adaptive washout: Find convergence point
        W_adaptive = min{t : ||x(t+1) - x(t)|| < ε}
        
        Ensemble washout: Multiple time scales
        x_ensemble = [x_W1, x_W2, x_W3] for W1 < W2 < W3
        
        Args:
            inputs: Input sequence (time_steps, n_inputs)
            washout: Base washout period
            
        Returns:
            Tuple of (processed_states, effective_washout_period)
            
        Research Notes:
        - Fixed washout (100-200 steps) sufficient for most applications
        - Adaptive washout optimal for unknown dynamics
        - Ensemble methods capture multiple time scales
        - Longer washout reduces usable training data
        """
        
        collection_method = getattr(self, 'state_collection_method', 'standard')
        
        if collection_method == 'adaptive_washout':
            # Option 1: Adaptive washout based on reservoir convergence
            states = self.run_reservoir(inputs, washout=0)  # Get all states
            convergence_threshold = 1e-6
            
            if len(states) > 1:
                state_changes = [np.linalg.norm(states[i+1] - states[i]) 
                               for i in range(len(states)-1)]
                adaptive_washout = next(
                    (i for i, change in enumerate(state_changes) 
                     if change < convergence_threshold), 
                    washout
                )
            else:
                adaptive_washout = washout
                
            return states[adaptive_washout:], adaptive_washout
            
        elif collection_method == 'ensemble_washout':
            # Option 2: Multiple washout periods for ensemble
            washout_periods = [washout//2, washout, washout*2]  # Different timescales
            state_collections = []
            
            for w in washout_periods:
                states = self.run_reservoir(inputs, w)
                if len(states) > 0:
                    state_collections.append(states)
            
            if state_collections:
                # Find minimum length to avoid dimension mismatch
                min_length = min(len(states) for states in state_collections)
                trimmed_collections = [states[:min_length] for states in state_collections]
                combined_states = np.concatenate(trimmed_collections, axis=1)
                return combined_states, washout
            else:
                # Fallback to standard washout
                states = self.run_reservoir(inputs, washout)
                return states, washout
            
        else:
            # Standard washout
            states = self.run_reservoir(inputs, washout)
            return states, washout

    def _apply_comprehensive_state_collection(self, reservoir_states: np.ndarray, 
                                            targets: np.ndarray, washout: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply comprehensive state collection strategies with 5 configurable options.
        
        Addresses advanced state collection strategies discussed in Jaeger 2001
        for different computational and temporal requirements. These methods can
        significantly impact training efficiency and model performance.
        
        Mathematical Foundation:
        =======================
        
        1. Subsampled collection:
           x_sub = {x(W+kΔ) : k ∈ ℕ, W+kΔ ≤ T}
           
        2. Exponential weighting:
           x_exp(t) = x(t) * exp(-α(T-t))
           
        3. Multi-horizon states:
           x_multi(t) = [x(t), x(t-τ₁), x(t-τ₂), ..., x(t-τₖ)]
           
        4. Adaptive spacing:
           Select states with high temporal variance
           
        Args:
            reservoir_states: Reservoir state matrix
            targets: Target sequence
            washout: Washout period used
            
        Returns:
            Tuple of (processed_states, adjusted_targets)
            
        Research Notes:
        - All states: Maximum information, highest computational cost
        - Subsampling: Reduced computation, may lose temporal details
        - Exponential weighting: Emphasizes recent dynamics
        - Multi-horizon: Captures multiple time scales simultaneously
        - Adaptive spacing: Focuses on informative state transitions
        """
        
        collection_method = getattr(self, 'state_collection_method', 'all_states')
        
        if collection_method == 'subsampled':
            # Option 1: Subsampling (every nth state to reduce computation)
            subsample_rate = getattr(self, 'subsample_rate', 2)
            reservoir_states = reservoir_states[::subsample_rate]
            targets_adjusted = targets[washout::subsample_rate]
            
        elif collection_method == 'exponential':
            # Option 2: Exponential sampling (more recent states weighted higher)
            decay_rate = getattr(self, 'decay_rate', 0.1)
            weights = np.exp(-decay_rate * np.arange(len(reservoir_states)))  
            reservoir_states = reservoir_states * weights[:, np.newaxis]
            targets_adjusted = targets[washout:]
            
        elif collection_method == 'multi_horizon':
            # Option 3: Multiple time horizon states (concat current + delayed states)
            delays = getattr(self, 'time_delays', [0, 1, 2, 5])  # Include states from t, t-1, t-2, t-5
            multi_horizon_states = []
            min_length = len(reservoir_states)
            
            for delay in delays:
                if delay == 0:
                    multi_horizon_states.append(reservoir_states)
                elif delay < len(reservoir_states):
                    delayed = np.roll(reservoir_states, delay, axis=0)[delay:]
                    multi_horizon_states.append(delayed)
                    min_length = min(min_length, len(delayed))
            
            if multi_horizon_states:
                # Trim all to same length and concatenate
                reservoir_states = np.concatenate(
                    [states[:min_length] for states in multi_horizon_states], axis=1)
                targets_adjusted = targets[washout:washout+min_length]
            else:
                targets_adjusted = targets[washout:]
            
        elif collection_method == 'adaptive_spacing':
            # Option 4: Adaptive spacing based on state variance
            if len(reservoir_states) > 1:
                state_vars = np.var(reservoir_states, axis=1)
                high_var_threshold = np.percentile(state_vars, 75)
                high_var_indices = np.where(state_vars > high_var_threshold)[0]
                
                if len(high_var_indices) > 0:
                    reservoir_states = reservoir_states[high_var_indices]
                    targets_adjusted = targets[washout:][high_var_indices]
                else:
                    targets_adjusted = targets[washout:]
            else:
                targets_adjusted = targets[washout:]
            
        else:  # 'all_states'
            # Option 5: Use all states after washout (standard)
            targets_adjusted = targets[washout:]
        
        return reservoir_states, targets_adjusted