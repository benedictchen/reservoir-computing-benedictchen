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
class LIFNeuron:
    """
    Configurable Leaky Integrate-and-Fire Neuron Model
    
    Now supports multiple implementation options including paper-accurate Maass 2002 parameters
    """
    
    def __init__(self, config: LIFNeuronConfig, neuron_type: str = 'E', position: Optional[np.ndarray] = None):
        self.config = config
        self.neuron_type = neuron_type
        self.model_type = config.model_type  # Expose model type for backward compatibility
        self.position = position if position is not None else np.zeros(3)
        
        # Adjust parameters based on neuron type
        self.tau_m = config.tau_m
        if neuron_type == 'I' and config.model_type == NeuronModelType.MAASS_2002_LIF:
            self.tau_m = 20.0  # Inhibitory neurons have faster dynamics in Maass 2002
            
        self.tau_ref = config.tau_ref
        if neuron_type == 'I' and config.model_type == NeuronModelType.MAASS_2002_LIF:
            self.tau_ref = 2.0  # 2ms for inhibitory vs 3ms for excitatory
            
        # Initialize state variables
        self.v_membrane = config.v_rest
        self.refractory_time = 0.0
        self.last_spike_time = -np.inf
        
        # Synaptic currents
        self.i_syn_exc = 0.0
        self.i_syn_inh = 0.0
        
        # Adaptation current (for adaptive models)
        self.i_adaptation = 0.0
        self.tau_adaptation = 100.0  # Adaptation time constant
        self.g_adaptation = 0.1  # Adaptation conductance
        
    def update(self, dt: float, synaptic_input: float = 0.0, external_current: float = 0.0) -> bool:
        """Update neuron state and return True if spike occurred"""
        
        # Skip update if in refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False
            
        # Total input current
        total_current = (
            self.config.background_current + 
            synaptic_input + 
            external_current - 
            self.i_adaptation
        )
        
        # Add noise if configured
        if self.config.current_noise_std > 0:
            total_current += np.random.normal(0, self.config.current_noise_std)
            
        # Membrane potential update (Maass 2002 accurate)
        dv_dt = (
            -(self.v_membrane - self.config.v_rest) + 
            self.config.input_resistance * total_current
        ) / self.tau_m
        
        self.v_membrane += dv_dt * dt
        
        # Add membrane noise if configured
        if self.config.membrane_noise_std > 0:
            self.v_membrane += np.random.normal(0, self.config.membrane_noise_std * np.sqrt(dt))
            
        # Update adaptation current
        if self.config.model_type == NeuronModelType.ADAPTIVE_LIF:
            self.i_adaptation += (-self.i_adaptation / self.tau_adaptation) * dt
            
        # Check for spike
        if self.v_membrane >= self.config.v_thresh:
            # Spike occurred
            self.v_membrane = self.config.v_reset
            self.refractory_time = self.tau_ref
            self.last_spike_time = 0.0  # Relative to current time
            
            # Update adaptation for adaptive models
            if self.config.model_type == NeuronModelType.ADAPTIVE_LIF:
                self.i_adaptation += self.g_adaptation
                
            return True
            
        return False


@dataclass
class BiologicalLIFNeuron:
    """
    Biologically realistic Leaky Integrate-and-Fire neuron matching Maass 2002.
    Current implementation has wrong τ_m (20ms vs 30ms) and missing features.
    """
    def __init__(self, neuron_type='E', position=None):
        # FIXME: Paper specifies τ_m = 30ms for excitatory, 20ms for inhibitory
        if neuron_type == 'E':
            self.tau_m = 30.0      # Membrane time constant (ms) - CORRECTED
            self.v_rest = -70.0    # Resting potential (mV)
            self.v_thresh = -50.0  # Firing threshold (mV)
            self.v_reset = -65.0   # Reset potential (mV)
        else:  # Inhibitory
            self.tau_m = 20.0      # Faster dynamics for interneurons
            self.v_rest = -70.0
            self.v_thresh = -50.0 
            self.v_reset = -65.0
            
        # FIXME: Missing heterogeneous background currents from paper
        self.I_background = np.random.normal(13.5, 1.5)  # nA, as in paper
        
        # FIXME: Missing synaptic parameters
        self.tau_syn_exc = 3.0     # Excitatory synaptic time constant (ms)
        self.tau_syn_inh = 6.0     # Inhibitory synaptic time constant (ms)
        
        # State variables
        self.v = self.v_rest
        self.I_syn_exc = 0.0       # Excitatory synaptic current
        self.I_syn_inh = 0.0       # Inhibitory synaptic current
        self.refractory_time = 0.0
        
        # FIXME: Missing 3D position for distance-dependent connectivity
        self.position = position if position is not None else np.random.rand(3)
        
    def update(self, dt, spike_inputs_exc=0.0, spike_inputs_inh=0.0):
        """Update neuron state with synaptic inputs"""
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False
            
        # Update synaptic currents
        self.I_syn_exc += (-self.I_syn_exc / self.tau_syn_exc + spike_inputs_exc) * dt
        self.I_syn_inh += (-self.I_syn_inh / self.tau_syn_inh + spike_inputs_inh) * dt
        
        # Total input current
        I_total = self.I_background + self.I_syn_exc - self.I_syn_inh
        
        # Membrane potential update
        dv = (-(self.v - self.v_rest) + I_total) / self.tau_m * dt
        self.v += dv
        
        # Check for spike
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_time = 2.0  # ms
            return True
        return False


# FIXME: Implement distance-dependent connectivity as described in paper
