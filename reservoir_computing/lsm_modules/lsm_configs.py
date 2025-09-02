"""
🧠 Liquid State Machine - Neural Reservoir Computing
===================================================

📚 Research Paper:
Maass, W., Natschläger, T., & Markram, H. (2002)
"Real-Time Computing Without Stable States: A New Framework for Neural Computation 
Based on Perturbations"
Neural Computation, 14(11), 2531-2560

🎯 ELI5 Summary:
Imagine your brain as a fish tank full of tiny robots (neurons) that communicate by sending
electrical sparks to each other. When you drop a pebble (input) into the tank, it creates
ripples that bounce around between the robots. Each robot remembers what happened and 
changes how it behaves. This "liquid" of activity patterns can solve complex problems
without needing to find a stable solution - the constantly changing patterns ARE the solution!

🧪 Research Background:
Traditional neural networks require stable states and convergence to fixed points.
Maass et al. revolutionized this by showing that:
- Temporal dynamics in recurrent networks can perform universal computation
- No equilibrium states needed - perturbations drive computation
- Biological neural microcircuits naturally implement this principle
- Short-term synaptic plasticity enables rich temporal processing

🔬 Mathematical Framework:
The LSM separates into two components:
- Liquid (L): Dynamic reservoir of spiking neurons with recurrent connectivity
- Readout (R): Maps liquid states to desired outputs

Liquid dynamics: dV/dt = -V/τ + RI(t) + noise
State extraction: x(t) = {spike patterns over time window}
Readout: y(t) = f(x(t)) where f is typically linear

🎨 ASCII Diagram - LSM Architecture:
=====================================

    Input Stream u(t)
         │
         ▼
    ┌─────────────┐
    │   LIQUID    │  ← Recurrent spiking neural network
    │  ┌─────────┐ │     - 100-1000 LIF neurons
    │  │ ○───○─○ │ │     - ~15% connectivity  
    │  │ │ ╲ ╱ │ │ │     - Dynamic synapses
    │  │ ○─○─○─○ │ │     - Temporal dynamics
    │  │  ╱ │ ╲  │ │
    │  │ ○───○─○ │ │
    │  └─────────┘ │
    └─────────────┘
         │
         ▼ x(t) - Liquid States
    ┌─────────────┐
    │   READOUT   │  ← Linear readout function
    │  ╔═══════╗  │     - Maps states→outputs
    │  ║ W×x(t) ║  │     - Trainable weights W
    │  ╚═══════╝  │     - No recurrence
    └─────────────┘
         │
         ▼
    Output y(t)

🏗️ Implementation Features:
✅ Multiple neuron models (LIF, Izhikevich, biological)
✅ Dynamic synapses with short-term plasticity  
✅ Configurable network topologies
✅ Paper-accurate Maass 2002 parameters
✅ Multiple readout mechanisms
✅ Temporal pattern classification
✅ Real-time processing capabilities

🎛️ Configuration Options:
- Neuron types: Simple LIF, Biological LIF, Izhikevich, Hodgkin-Huxley
- Synapse models: Static, Markram Dynamic, Tsodyks-Markram STP
- Connectivity: Random, Distance-dependent, Small-world, Scale-free
- State extraction: Spike counts, PSP decay, Membrane potentials
- Readout: Linear regression, Population neurons, P-delta learning

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

🔗 Related Work: Echo State Networks, Neural Reservoir Computing, Biological Neural Networks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Literal
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import sys
import os

# Add parent directory to path for donation_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donation_utils import show_donation_message, show_completion_message


# Configuration options for different components
class LIFNeuronConfig:
    """
    Configurable LIF Neuron parameters with multiple preset options
    
    Now supports paper-accurate parameters from Maass 2002
    """
    # Model type determines parameter defaults
    model_type: NeuronModelType = NeuronModelType.MAASS_2002_LIF
    
    # Core LIF parameters (will be set based on model_type if None)
    tau_m: Optional[float] = None  # Membrane time constant (ms)
    tau_ref: Optional[float] = None  # Refractory period (ms)
    v_reset: Optional[float] = None  # Reset potential (mV)
    v_thresh: Optional[float] = None  # Spike threshold (mV)
    v_rest: Optional[float] = None  # Resting potential (mV)
    
    # Biological parameters (Maass 2002 accurate)
    input_resistance: Optional[float] = None  # Input resistance (MΩ)
    background_current: Optional[float] = None  # Background current (nA)
    
    # Synaptic parameters
    tau_syn_exc: Optional[float] = None  # Excitatory synaptic time constant (ms)
    tau_syn_inh: Optional[float] = None  # Inhibitory synaptic time constant (ms)
    
    # Noise parameters
    membrane_noise_std: float = 0.0  # Membrane noise standard deviation
    current_noise_std: float = 0.0  # Current noise standard deviation
    
    def __post_init__(self):
        """Set default parameters based on model type"""
        if self.model_type == NeuronModelType.SIMPLE_LIF:
            # Current implementation defaults
            self.tau_m = self.tau_m or 20.0
            self.tau_ref = self.tau_ref or 2.0
            self.v_reset = self.v_reset or -70.0
            self.v_thresh = self.v_thresh or -54.0
            self.v_rest = self.v_rest or -70.0
            self.input_resistance = self.input_resistance or 1.0
            self.background_current = self.background_current or 0.0
            self.tau_syn_exc = self.tau_syn_exc or 3.0
            self.tau_syn_inh = self.tau_syn_inh or 6.0
            
        elif self.model_type == NeuronModelType.MAASS_2002_LIF:
            # Paper-accurate parameters from Maass et al. 2002
            self.tau_m = self.tau_m or 30.0  # 30ms for excitatory
            self.tau_ref = self.tau_ref or 3.0  # 3ms excitatory, 2ms inhibitory
            self.v_reset = self.v_reset or 13.5  # Reset to 13.5mV
            self.v_thresh = self.v_thresh or 15.0  # Threshold 15mV
            self.v_rest = self.v_rest or 0.0  # Resting potential 0mV
            self.input_resistance = self.input_resistance or 1.0  # 1 MΩ
            self.background_current = self.background_current or 13.5  # 13.5 nA
            self.tau_syn_exc = self.tau_syn_exc or 3.0  # 3ms excitatory PSCs
            self.tau_syn_inh = self.tau_syn_inh or 6.0  # 6ms inhibitory PSCs
            
        elif self.model_type == NeuronModelType.BIOLOGICAL_LIF:
            # Enhanced biological realism
            self.tau_m = self.tau_m or 20.0
            self.tau_ref = self.tau_ref or 2.0
            self.v_reset = self.v_reset or -65.0
            self.v_thresh = self.v_thresh or -50.0
            self.v_rest = self.v_rest or -70.0
            self.input_resistance = self.input_resistance or 1.0
            self.background_current = self.background_current or 10.0
            self.tau_syn_exc = self.tau_syn_exc or 2.0
            self.tau_syn_inh = self.tau_syn_inh or 10.0
            self.membrane_noise_std = self.membrane_noise_std or 1.0
            self.current_noise_std = self.current_noise_std or 0.5
            
        elif self.model_type == NeuronModelType.ADAPTIVE_LIF:
            # Adaptive LIF with adaptation currents
            self.tau_m = self.tau_m or 20.0
            self.tau_ref = self.tau_ref or 2.0
            self.v_reset = self.v_reset or -70.0
            self.v_thresh = self.v_thresh or -50.0
            self.v_rest = self.v_rest or -70.0
            self.input_resistance = self.input_resistance or 1.0
            self.background_current = self.background_current or 5.0
            self.tau_syn_exc = self.tau_syn_exc or 3.0
            self.tau_syn_inh = self.tau_syn_inh or 6.0


@dataclass
class LSMConfig:
    """
    Complete configuration for Liquid State Machine
    
    Provides multiple implementation options for each component
    allowing users to choose between approaches
    """
    # Liquid structure
    n_liquid: int = 135  # Paper default: 135 neurons (15×3×3)
    excitatory_ratio: float = 0.8
    dt: float = 0.1  # Integration time step (ms)
    
    # Input/output dimensions
    input_dim: int = 10  # Number of input channels
    output_dim: int = 2  # Number of output classes
    
    # Neuron configuration
    neuron_config: Optional[LIFNeuronConfig] = None
    neuron_type: NeuronModelType = NeuronModelType.MAASS_2002_LIF  # Backward compatibility
    
    # Connectivity configuration
    connectivity_type: ConnectivityType = ConnectivityType.DISTANCE_DEPENDENT
    connectivity_prob: float = 0.1
    connectivity_params: Optional[Dict] = None  # Additional connectivity parameters
    lambda_param: float = 2.0  # Spatial decay constant
    spatial_organization: bool = True
    
    # Synapse configuration
    synapse_type: SynapseModelType = SynapseModelType.MARKRAM_DYNAMIC
    
    # Liquid state extraction
    state_type: LiquidStateType = LiquidStateType.PSP_DECAY
    state_tau_decay: float = 30.0  # PSP decay time constant (ms)
    state_window_size: float = 50.0  # For spike count/firing rate methods
    state_tau_scales: Optional[List[float]] = None  # For multi-timescale
    
    # Readout configuration
    readout_type: ReadoutType = ReadoutType.LINEAR_REGRESSION
    
    def __post_init__(self):
        """Set defaults for optional fields"""
        if self.neuron_config is None:
            self.neuron_config = LIFNeuronConfig(model_type=NeuronModelType.MAASS_2002_LIF)
        
        if self.state_tau_scales is None and self.state_type == LiquidStateType.MULTI_TIMESCALE:
            self.state_tau_scales = [3.0, 10.0, 30.0, 100.0]

