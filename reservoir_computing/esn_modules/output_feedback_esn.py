"""
🌊 Echo State Network (ESN) - Revolutionary Reservoir Computing
============================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

Based on: Herbert Jaeger (2001) "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"

🎯 ELI5 Summary:
Think of an Echo State Network like a pond with many interconnected ripples.
When you drop a stone (input), it creates complex wave patterns that echo
through the water. The network just needs to learn how to "read" these patterns
at the surface - no need to control the complex dynamics underneath!

🔬 Research Background:
========================
Herbert Jaeger's 2001 breakthrough solved a fundamental problem: training 
recurrent neural networks was extremely slow and difficult due to vanishing/
exploding gradients. His insight: don't train the recurrent connections at all!

The ESN revolution:
- Fixed random recurrent reservoir (the "pond")
- Only train simple linear readout (the "surface reader")  
- 1000x faster training than traditional RNNs
- Natural handling of temporal dependencies
- Rich dynamics from simple random connectivity

This launched the entire field of "Reservoir Computing" and influenced
modern architectures like LSTMs and Transformers.

🏗️ Architecture:
================
Input → [Input Weights] → [Reservoir] → [Readout] → Output
  u           W_in          x(t+1)       W_out       y
              ↓               ↑            ↑
              └─── [Recurrent W_res] ──────┘
                      (fixed!)        (trainable!)

🎨 ASCII Diagram - ESN Information Flow:
=======================================
                🌊 RESERVOIR (Fixed Random Network) 🌊
    Input u(t) ──→ ⚫──⚫──⚫──⚫──⚫ ──→ Output y(t)
                   │╲ │╱ │╲ │╱ │      
                   │ ╲│╱  │ ╲│╱ │      Linear
                   ⚫──⚫──⚫──⚫──⚫      Readout
                   │╱ │╲  │╱ │╲ │      (Trainable)
                   ⚫──⚫──⚫──⚫──⚫
                      ↑
                 Rich Dynamics
                 (Never trained!)

Mathematical Framework:
- State Update: x(t+1) = (1-α)x(t) + α·tanh(W_in·u(t) + W_res·x(t))
- Output: y(t) = W_out·x(t)
- Echo State Property: ||∂x(t+1)/∂x(t)|| < 1 (spectral radius < 1)

🚀 Key Innovation: "Echo State Property"
Revolutionary Impact: Reservoir acts as universal temporal basis functions

⚡ Configurable Options:
=======================
✨ Reservoir Topologies:
  - random: Classical sparse random connectivity [default]
  - ring: Simple ring topology for periodic patterns
  - small_world: Watts-Strogatz small-world networks
  - scale_free: Barabási-Albert preferential attachment

✨ Spectral Radius Optimization:
  - grid_search: Systematic search for optimal radius
  - manual: User-specified fixed radius [default]
  - adaptive: Dynamic adjustment during training

✨ Activation Functions:
  - tanh: Standard, bounded, smooth [default] 
  - relu: Unbounded, sparse, fast computation
  - sigmoid: Bounded, smooth, biological
  - custom: User-defined function

🎨 Advanced Features:
====================
🔧 Multiple Timescales: Different neuron groups with different leak rates
🔧 Output Feedback: Closed-loop operation with output→input connections
🔧 Teacher Forcing: Ground truth feedback during training
🔧 Washout Adaptive: Automatic transient removal
🔧 Input Shift/Scaling: Preprocessing for optimal input range

📊 ESN Training Process:
=======================
Phase 1: Reservoir States Collection
    Input Sequence ──→ [Fixed Reservoir] ──→ State Matrix X
    
Phase 2: Linear Regression
    X · W_out = Y_target  →  W_out = (X^T X)^(-1) X^T Y_target
    
Phase 3: Prediction
    New Input ──→ [Reservoir] ──→ [Trained Readout] ──→ Prediction

🎯 Applications:
===============
- 📈 Time Series Prediction: Financial forecasting, weather prediction
- 🎵 Audio Processing: Speech recognition, music generation
- 🤖 Control Systems: Robot control, adaptive filtering  
- 🧬 Bioinformatics: Gene expression, protein folding
- 📊 Signal Processing: Noise reduction, pattern recognition
- 🎮 Game AI: Behavior modeling, strategy learning

⚡ Performance Benefits:
=======================
✅ 1000x faster training than BPTT
✅ No vanishing/exploding gradient problems
✅ Natural temporal memory without explicit delays
✅ Robust to hyperparameter choices
✅ Online learning capability
✅ Minimal overfitting due to fixed reservoir

⚠️ Important Notes:
==================
- Spectral radius must be < 1 for stability (Echo State Property)
- Reservoir size vs. complexity: bigger ≠ always better  
- Washout period needed to eliminate initial transients
- Input scaling affects reservoir utilization
- Sparsity balances computation vs. expressiveness

💝 Please support our work: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
Buy us a coffee, beer, or better! Your support makes advanced AI research accessible to everyone! ☕🍺🚀
"""

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import warnings
import sys
import os

# Add parent directory to path for donation_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donation_utils import show_donation_message, show_completion_message

class OutputFeedbackESN(EchoStateNetwork):
    """
    FIXME: CRITICAL MISSING IMPLEMENTATION - Output Feedback Connections
    
    Jaeger 2001 describes W_fb matrix for feeding output back to reservoir:
    x(t+1) = f(W*x(t) + W_in*u(t) + W_fb*y(t-1))
    
    This is essential for autonomous generation and closed-loop control.
    """
    
    def __init__(self, output_feedback: bool = True, 
                 feedback_scaling: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.output_feedback = output_feedback
        self.feedback_scaling = feedback_scaling
        self.W_feedback = None
        
    def _initialize_feedback_weights(self, n_outputs: int):
        """FIXME: IMPLEMENT - Initialize output feedback matrix W_fb"""
        
        if self.output_feedback:
            self.W_feedback = np.random.uniform(
                -self.feedback_scaling,
                self.feedback_scaling, 
                (self.n_reservoir, n_outputs)
            )
            
            # FIXME: Should W_fb have same spectral properties as W_reservoir?
            # Jaeger 2001 doesn't specify but suggests similar scaling
            print(f"✓ Output feedback initialized: {self.W_feedback.shape}")
        
    def _update_state_with_feedback(self, state: np.ndarray, 
                                  input_vec: np.ndarray, 
                                  output_feedback: Optional[np.ndarray] = None) -> np.ndarray:
        """
        FIXME: IMPLEMENT - State update with output feedback
        
        Modified reservoir dynamics including W_fb term
        """
        
        # Standard reservoir + input terms
        reservoir_input = self.W_reservoir @ state + self.W_input @ input_vec
        
        # FIXME: Add output feedback if available
        if output_feedback is not None and self.W_feedback is not None:
            reservoir_input += self.W_feedback @ output_feedback
            
        # Apply activation and leak rate
        raw_state = np.tanh(reservoir_input + 
                           np.random.normal(0, self.noise_level, self.n_reservoir))
        new_state = (1 - self.leak_rate) * state + self.leak_rate * raw_state
        
        return new_state
    
    def generate_with_feedback(self, n_steps: int, 
                             initial_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        FIXME: IMPLEMENT - Generation with proper output feedback
        
        Uses W_fb matrix for true closed-loop generation as in Jaeger 2001
        """
        
        if self.W_out is None:
            raise ValueError("Network must be trained before generation!")
            
        if not hasattr(self, 'W_feedback') or self.W_feedback is None:
            warnings.warn("No feedback matrix - falling back to input feedback")
            return self.generate(n_steps, initial_input)
            
        # Initialize
        state = self.last_state.copy() if self.last_state is not None else np.zeros(self.n_reservoir)
        outputs = []
        previous_output = None
        
        if initial_input is None:
            current_input = np.zeros(self.W_input.shape[1])
        else:
            current_input = initial_input.copy()
            
        for step in range(n_steps):
            # Update state with output feedback
            state = self._update_state_with_feedback(state, current_input, previous_output)
            
            # Generate output
            state_with_bias = np.append(state, 1.0)
            output = state_with_bias @ self.W_out.T + self.bias
            outputs.append(output)
            
            # Store for next feedback
            previous_output = output
            
            # FIXME: May also want to update current_input based on task
            
        return np.array(outputs)


