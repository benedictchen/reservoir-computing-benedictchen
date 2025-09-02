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

class TeacherForcingTrainer:
    """
    FIXME: IMPLEMENT - Teacher forcing training from Jaeger 2001
    
    During training, feed target output instead of predicted output to prevent
    error accumulation. Critical for sequence generation tasks.
    """
    
    @staticmethod
    def train_with_teacher_forcing(
        esn: Union[EchoStateNetwork, OutputFeedbackESN],
        inputs: np.ndarray,
        targets: np.ndarray,
        reg_param: float = 1e-6,
        washout: int = 100,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        FIXME: IMPLEMENT - Training with teacher forcing
        
        Args:
            teacher_forcing_ratio: 1.0 = always use target, 0.0 = never use target
        """
        
        time_steps, n_inputs = inputs.shape
        _, n_outputs = targets.shape
        
        # Initialize feedback weights if needed
        if hasattr(esn, '_initialize_feedback_weights'):
            esn._initialize_feedback_weights(n_outputs)
        
        # Initialize
        if not hasattr(esn, 'W_input'):
            esn._initialize_input_weights(n_inputs)
            
        state = np.zeros(esn.n_reservoir)
        states = []
        
        # Run with teacher forcing
        for t in range(time_steps):
            # Decide whether to use teacher forcing
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            
            # Get feedback signal
            if t > 0 and hasattr(esn, '_update_state_with_feedback'):
                if use_teacher_forcing:
                    feedback = targets[t-1]  # Use target (teacher forcing)
                else:
                    # FIXME: Use predicted output (would need to compute it)
                    feedback = targets[t-1]  # Simplified for now
                    
                state = esn._update_state_with_feedback(state, inputs[t], feedback)
            else:
                state = esn._update_state(state, inputs[t])
            
            if t >= washout:
                states.append(state.copy())
        
        # Standard linear readout training
        states = np.array(states)
        X = np.column_stack([states, np.ones(len(states))])
        y = targets[washout:]
        
        ridge = Ridge(alpha=reg_param)
        ridge.fit(X, y)
        
        esn.W_out = ridge.coef_
        esn.bias = ridge.intercept_
        
        predictions = ridge.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        return {
            'mse': mse,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'n_states_used': len(states)
        }


