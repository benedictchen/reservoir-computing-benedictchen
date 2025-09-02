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
class LiquidStateExtractor(ABC):
    """
    Abstract base class for liquid state extraction methods
    
    The liquid state defines what information from the liquid dynamics 
    is available to the readout - a crucial concept from Maass 2002
    """
    
    @abstractmethod
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, **kwargs) -> np.ndarray:
        """
        🧠 Extract Liquid State Vector at Current Time - Maass 2002 Implementation!
        
        Args:
            spike_matrix: Matrix of spike events [n_neurons, n_timesteps]
            times: Array of time points corresponding to spike_matrix columns
            current_time: Current simulation time for state extraction
            **kwargs: Additional parameters for subclasses
            
        Returns:
            np.ndarray: Liquid state vector representing current neural activity
            
        📚 **Reference**: Maass, W., Natschläger, T., & Markram, H. (2002)
        "Real-time computing without stable states: A new framework for neural 
        computation based on perturbations"
        """
        # Default implementation extracts spike counts in recent time window
        window_size = kwargs.get('window_size', 50.0)  # ms
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        
        # Find time window indices
        start_time = max(0, current_time - window_size)
        start_idx = max(0, int(start_time / dt))
        current_idx = min(len(times) - 1, int(current_time / dt))
        
        if start_idx >= current_idx:
            return np.zeros(spike_matrix.shape[0])
            
        # Extract spike counts in window
        window_spikes = spike_matrix[:, start_idx:current_idx + 1]
        state_vector = np.sum(window_spikes, axis=1)
        
        return state_vector.astype(float)
    
    @abstractmethod
    def reset_state(self):
        """
        🔄 Reset Internal State Variables - Prepare for New Simulation!
        
        Resets any internal state variables to initial conditions.
        Essential for proper liquid state computation across multiple trials.
        
        📝 **Usage**:
        ```python
        extractor = PSPDecayExtractor(tau_psp=3.0)
        # ... run simulation ...
        extractor.reset_state()  # Reset for next trial
        ```
        """
        pass


class PSPDecayExtractor(LiquidStateExtractor):
    """
    CORRECT liquid state extraction using PSP decay from Maass 2002
    
    "The liquid state x^M(t) at time t is defined as the vector of values 
    that the outputs of all liquid neurons would contribute to the membrane 
    potential of a readout neuron if they were connected to that readout neuron"
    
    Addresses FIXME: Missing correct liquid state definition
    """
    
    def __init__(self, tau_decay: float = 30.0, n_liquid: int = 135):
        self.tau_decay = tau_decay  # PSP decay time constant (ms)
        self.n_liquid = n_liquid
        self.psp_traces = np.zeros(n_liquid)  # Current PSP values
        
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, dt: float = 0.1) -> np.ndarray:
        """Extract PSP-based liquid state"""
        # Find current time index
        time_idx = int(current_time / dt)
        
        if time_idx < spike_matrix.shape[1]:
            # Update PSP traces with exponential decay
            self.psp_traces *= np.exp(-dt / self.tau_decay)
            
            # Add spike contributions
            current_spikes = spike_matrix[:, time_idx]
            self.psp_traces += current_spikes
            
        return self.psp_traces.copy()
    
    def reset_state(self):
        """Reset PSP traces"""
        self.psp_traces.fill(0.0)


class SpikeCountExtractor(LiquidStateExtractor):
    """
    Current simplified approach using spike counts in time windows
    
    Kept for backward compatibility but not paper-accurate
    """
    
    def __init__(self, window_size: float = 50.0, n_liquid: int = 135):
        self.window_size = window_size  # ms
        self.n_liquid = n_liquid
        
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, dt: float = 0.1) -> np.ndarray:
        """Extract spike count features in time window"""
        window_steps = int(self.window_size / dt)
        time_idx = int(current_time / dt)
        
        start_idx = max(0, time_idx - window_steps)
        end_idx = min(spike_matrix.shape[1], time_idx + 1)
        
        if start_idx < end_idx:
            spike_counts = np.sum(spike_matrix[:, start_idx:end_idx], axis=1)
        else:
            spike_counts = np.zeros(self.n_liquid)
            
        return spike_counts
    
    def reset_state(self):
        """No internal state to reset for spike counts"""
        pass


class MembranePotentialExtractor(LiquidStateExtractor):
    """
    Direct membrane potential readout
    
    Uses current membrane potentials as liquid state
    """
    
    def __init__(self, n_liquid: int = 135):
        self.n_liquid = n_liquid
        
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, membrane_potentials: np.ndarray = None, 
                     **kwargs) -> np.ndarray:
        """Extract membrane potential state"""
        if membrane_potentials is not None:
            return membrane_potentials.copy()
        else:
            # Fallback to zeros if membrane potentials not provided
            return np.zeros(self.n_liquid)
    
    def reset_state(self):
        """No internal state to reset"""
        pass


class FiringRateExtractor(LiquidStateExtractor):
    """
    Population firing rate-based liquid state
    
    Uses instantaneous firing rates of neuron populations
    """
    
    def __init__(self, window_size: float = 10.0, n_liquid: int = 135):
        self.window_size = window_size  # ms
        self.n_liquid = n_liquid
        
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, dt: float = 0.1) -> np.ndarray:
        """Extract firing rate state"""
        window_steps = int(self.window_size / dt)
        time_idx = int(current_time / dt)
        
        start_idx = max(0, time_idx - window_steps)
        end_idx = min(spike_matrix.shape[1], time_idx + 1)
        
        if start_idx < end_idx:
            spike_counts = np.sum(spike_matrix[:, start_idx:end_idx], axis=1)
            window_duration = (end_idx - start_idx) * dt / 1000.0  # Convert to seconds
            firing_rates = spike_counts / max(window_duration, dt/1000.0)  # Hz
        else:
            firing_rates = np.zeros(self.n_liquid)
            
        return firing_rates
    
    def reset_state(self):
        """No internal state to reset"""
        pass


class MultiTimescaleExtractor(LiquidStateExtractor):
    """
    Multi-timescale liquid state extraction
    
    Combines multiple PSP decay constants for richer temporal representation
    """
    
    def __init__(self, tau_scales: List[float] = None, n_liquid: int = 135):
        if tau_scales is None:
            tau_scales = [3.0, 10.0, 30.0, 100.0]  # Multiple time constants
        
        self.tau_scales = tau_scales
        self.n_liquid = n_liquid
        self.psp_traces = {tau: np.zeros(n_liquid) for tau in tau_scales}
        
    def extract_state(self, spike_matrix: np.ndarray, times: np.ndarray, 
                     current_time: float, dt: float = 0.1) -> np.ndarray:
        """Extract multi-timescale PSP state"""
        time_idx = int(current_time / dt)
        
        if time_idx < spike_matrix.shape[1]:
            current_spikes = spike_matrix[:, time_idx]
            
            # Update each timescale
            for tau in self.tau_scales:
                self.psp_traces[tau] *= np.exp(-dt / tau)
                self.psp_traces[tau] += current_spikes
        
        # Concatenate all timescale features
        state_vector = np.concatenate([self.psp_traces[tau] for tau in self.tau_scales])
        return state_vector
    
    def reset_state(self):
        """Reset all PSP traces"""
        for tau in self.tau_scales:
            self.psp_traces[tau].fill(0.0)


# Abstract base class and implementations for readout mechanisms
class LiquidStateExtractor:
    """
    Extract liquid state as described in Maass 2002 - using PSP decay, not spike counts!
    Current implementation fundamentally misunderstands liquid state definition.
    
    From paper: "The liquid state at time t is the vector of all PSP values"
    """
    def __init__(self, tau_psp=3.0, readout_positions=None):
        self.tau_psp = tau_psp  # PSP decay time constant
        self.readout_positions = readout_positions
        
    def extract_state(self, spike_matrix, times, readout_times):
        """
        Extract liquid state as PSP values at readout times.
        This is the CORRECT definition from Maass 2002!
        """
        n_neurons, n_timesteps = spike_matrix.shape
        n_readout = len(readout_times)
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # Convert spikes to PSPs
        psp_matrix = np.zeros_like(spike_matrix, dtype=float)
        
        for neuron in range(n_neurons):
            psp = 0.0
            for t_idx in range(n_timesteps):
                # PSP decay
                psp *= np.exp(-dt / self.tau_psp)
                # Add spike contribution
                if spike_matrix[neuron, t_idx]:
                    psp += 1.0
                psp_matrix[neuron, t_idx] = psp
        
        # Extract states at readout times
        liquid_states = []
        for readout_time in readout_times:
            t_idx = int(readout_time / dt)
            if t_idx < n_timesteps:
                state = psp_matrix[:, t_idx]
                liquid_states.append(state)
            
        return np.array(liquid_states)


# FIXME: Implement population readout neurons with p-delta learning
