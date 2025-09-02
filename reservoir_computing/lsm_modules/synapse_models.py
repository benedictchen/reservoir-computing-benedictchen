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
class DynamicSynapseConfig:
    """
    Configuration for dynamic synapses with multiple model options
    
    Implements the Markram et al. 1997/1998 model as used in Maass 2002
    
    Paper-accurate parameter ranges (from Maass 2002):
    - U: 0.03-0.6 (release probability parameter)
    - D: 0.1-3.0 seconds (depression time constant, 100-3000ms range)
    - F: 0.02-1.0 seconds (facilitation time constant, 20-1000ms range)
    """
    synapse_type: SynapseModelType = SynapseModelType.MARKRAM_DYNAMIC
    
    # Markram model parameters (U, D, F) - defaults are connection-type specific
    U: float = 0.5  # Release probability parameter (range: 0.03-0.6)
    D: float = 1.1  # Depression time constant in seconds (range: 0.1-3.0s)
    F: float = 0.05  # Facilitation time constant in seconds (range: 0.02-1.0s)
    
    # Connection type specific parameters (from Maass 2002)
    connection_type: str = "EE"  # EE, EI, IE, II
    
    # Scaling amplitude
    amplitude: float = 30.0  # nA - scaling factor
    
    # Synaptic delay (important for realistic neural dynamics)
    delay: float = 1.0  # ms - synaptic transmission delay
    
    def __post_init__(self):
        """Set parameters based on connection type (Maass 2002 Table)"""
        if self.connection_type == "EE":
            self.U = 0.5 if self.U == 0.5 else self.U  # Default or user-specified
            self.D = 1.1 if self.D == 1.1 else self.D
            self.F = 0.05 if self.F == 0.05 else self.F
            self.amplitude = 30.0 if self.amplitude == 30.0 else self.amplitude
        elif self.connection_type == "EI":
            self.U = 0.05 if self.U == 0.5 else self.U  # Override default
            self.D = 0.125 if self.D == 1.1 else self.D
            self.F = 1.2 if self.F == 0.05 else self.F
            self.amplitude = 60.0 if self.amplitude == 30.0 else self.amplitude
        elif self.connection_type == "IE":
            self.U = 0.25 if self.U == 0.5 else self.U
            self.D = 0.7 if self.D == 1.1 else self.D
            self.F = 0.02 if self.F == 0.05 else self.F
            self.amplitude = -19.0 if self.amplitude == 30.0 else self.amplitude  # Inhibitory
        elif self.connection_type == "II":
            self.U = 0.32 if self.U == 0.5 else self.U
            self.D = 0.144 if self.D == 1.1 else self.D
            self.F = 0.06 if self.F == 0.05 else self.F
            self.amplitude = -19.0 if self.amplitude == 30.0 else self.amplitude  # Inhibitory


class DynamicSynapse:
    """
    Dynamic synapse implementation supporting multiple models
    
    Implements the Markram et al. (1997) model as specified in Maass 2002
    """
    
    def __init__(self, config: DynamicSynapseConfig, pre_neuron_type: str = 'E', post_neuron_type: str = 'E'):
        # Set connection type based on neuron types
        config.connection_type = f"{pre_neuron_type}{post_neuron_type}"
        config.__post_init__()  # Recalculate parameters
        
        self.config = config
        self.model_type = config.synapse_type  # Expose model type for backward compatibility
        self.U = config.U  # Baseline release probability
        self.D = config.D  # Depression time constant (seconds)
        self.F = config.F  # Facilitation time constant (seconds)
        self.amplitude = config.amplitude
        
        # State variables
        self.u = self.U  # Current release probability
        self.R = 1.0     # Available resources (0-1)
        self.last_spike_time = -np.inf
        
        # Static synapse fallback
        self.static_weight = config.amplitude
        
    def process_spike(self, current_time: float) -> float:
        """
        Process incoming spike and return synaptic efficacy
        
        Returns:
            Synaptic efficacy (scaled amplitude)
        """
        if self.config.synapse_type == SynapseModelType.STATIC:
            return self.static_weight
            
        # Time since last spike
        dt = current_time - self.last_spike_time
        
        if dt > 0 and self.last_spike_time > -np.inf:
            # Recovery from depression and facilitation decay
            self.R = 1.0 - (1.0 - self.R) * np.exp(-dt / self.D)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.F)
            
        # Spike-triggered changes
        synaptic_efficacy = self.u * self.R * self.amplitude
        
        # Update state for next spike
        self.u += self.U * (1.0 - self.u)  # Facilitation
        self.R -= self.u * self.R           # Depression
        self.last_spike_time = current_time
        
        return synaptic_efficacy
        
    def get_current_weight(self) -> float:
        """Get current synaptic weight without spike"""
        if self.config.synapse_type == SynapseModelType.STATIC:
            return self.static_weight
        return self.u * self.R * self.amplitude


class AlternativeDynamicSynapse:
    """
    Alternative Dynamic synapse implementation using Markram et al. (1997) model.
    Shows paper-accurate parameter ranges for enhanced heterogeneous implementations.
    
    From Maass 2002: "The model for dynamic synapses is based on the 
    phenomenological model of Markram et al. (1997)"
    """
    def __init__(self, U=0.5, D=1100, F=50):
        # Paper defaults with heterogeneous distributions needed
        self.U = U      # Release probability parameter (0.03-0.6 range)
        self.D = D      # Depression time constant (ms) (100-3000 range) 
        self.F = F      # Facilitation time constant (ms) (20-1000 range)
        self.u = U      # Current release probability
        self.R = 1.0    # Current available resources
        self.last_spike_time = -np.inf
        
    def process_spike(self, spike_time):
        """Update synaptic efficacy for incoming spike"""
        dt = spike_time - self.last_spike_time
        if dt > 0:
            # Recovery from depression and facilitation decay
            self.R = 1.0 - (1.0 - self.R) * np.exp(-dt / self.D)
            self.u = self.U + (self.u - self.U) * np.exp(-dt / self.F)
        
        # Spike-triggered changes
        amplitude = self.u * self.R  # Synaptic efficacy
        self.u += self.U * (1.0 - self.u)  # Facilitation
        self.R -= self.u * self.R           # Depression
        self.last_spike_time = spike_time
        
        return amplitude

# FIXME: Implement biologically realistic LIF with correct paper parameters
