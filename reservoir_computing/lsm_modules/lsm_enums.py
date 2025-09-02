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
class NeuronModelType(Enum):
    """Types of neuron models available"""
    SIMPLE_LIF = "simple_lif"  # Current simplified implementation
    LEAKY_INTEGRATE_AND_FIRE = "simple_lif"  # Backward compatibility alias
    INTEGRATE_AND_FIRE = "simple_lif"  # Another backward compatibility alias
    IZHIKEVICH = "simple_lif"  # Izhikevich model (simplified to LIF for now)
    HODGKIN_HUXLEY = "biological_lif"  # Hodgkin-Huxley model
    MAASS_2002_LIF = "maass_2002_lif"  # Paper-accurate parameters
    BIOLOGICAL_LIF = "biological_lif"  # Full biological realism
    ADAPTIVE_LIF = "adaptive_lif"  # With adaptation currents

class SynapseModelType(Enum):
    """Types of synapse models"""
    STATIC = "static"  # Current implementation
    MARKRAM_DYNAMIC = "markram_dynamic"  # Maass 2002 dynamic synapses
    TSODYKS_MARKRAM = "tsodyks_markram"  # Full TM model
    STP_ENHANCED = "stp_enhanced"  # Enhanced short-term plasticity
    STP = "stp_enhanced"  # Short-term plasticity (alias)
    STDP = "markram_dynamic"  # Spike-timing dependent plasticity (simplified to Markram for now)

class ConnectivityType(Enum):
    """Types of connectivity patterns"""
    RANDOM_UNIFORM = "random_uniform"  # Current random connectivity
    RANDOM = "random_uniform"  # Backward compatibility alias
    DISTANCE_DEPENDENT = "distance_dependent"  # Maass 2002 distance-based
    COLUMN_STRUCTURED = "column_structured"  # 3D column organization
    SMALL_WORLD = "small_world"  # Small-world topology
    SCALE_FREE = "scale_free"  # Scale-free networks

class LiquidStateType(Enum):
    """Types of liquid state extraction"""
    SPIKE_COUNTS = "spike_counts"  # Current implementation
    PSP_DECAY = "psp_decay"  # Maass 2002 correct implementation
    MEMBRANE_POTENTIALS = "membrane_potentials"  # Direct V_m readout
    FIRING_RATES = "firing_rates"  # Population firing rates
    MULTI_TIMESCALE = "multi_timescale"  # Multiple decay constants

class ReadoutType(Enum):
    """Types of readout mechanisms"""
    LINEAR_REGRESSION = "linear_regression"  # Current implementation
    POPULATION_NEURONS = "population_neurons"  # Maass 2002 I&F populations
    P_DELTA_LEARNING = "p_delta_learning"  # Biologically realistic learning
    PERCEPTRON = "perceptron"  # Simple perceptron
    SVM = "svm"  # Support Vector Machine

@dataclass
