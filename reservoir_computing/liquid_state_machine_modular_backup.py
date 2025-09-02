"""
🌊 Liquid State Machine (LSM) - Refactored Modular Implementation
============================================================

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Maass, Natschläger & Markram (2002) "Real-Time Computing Without Stable States"

Refactored modular implementation with clear separation of concerns for better maintainability.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .lsm_config import LSMConfig, NeuronModelType, LiquidStateType, ReadoutType
from .lif_neuron import LIFNeuron, LIFNeuronPopulation
from .synaptic_models import ConnectivityMatrix, create_connectivity_matrix
from .liquid_state_extractors import create_liquid_state_extractor
from .readout_mechanisms import create_readout_mechanism

# Configuration options for different components
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

@dataclass
class LIFNeuronConfig:
    """
    Configurable LIF Neuron parameters with multiple preset options
    """
    model_type: NeuronModelType = NeuronModelType.SIMPLE_LIF
    
    # Core LIF parameters
    threshold: float = -50.0    # Firing threshold (mV)
    reset_potential: float = -65.0  # Reset potential after spike (mV)
    rest_potential: float = -65.0   # Resting potential (mV)
    membrane_time_constant: float = 20.0  # tau_m (ms)
    resistance: float = 1.0  # Membrane resistance (MOhm)
    
    # Refractory period
    refractory_period: float = 2.0  # ms
    
    # Noise parameters
    noise_std: float = 0.0  # Standard deviation of noise current
    
    def __post_init__(self):
        """Set parameters based on model type"""
        if self.model_type == NeuronModelType.MAASS_2002_LIF:
            # Paper-accurate parameters from Maass et al. 2002
            self.threshold = -54.0  # mV
            self.reset_potential = -60.0  # mV
            self.rest_potential = -70.0  # mV
            self.membrane_time_constant = 30.0  # ms
            self.refractory_period = 3.0  # ms
            self.noise_std = 0.5  # nA
        elif self.model_type == NeuronModelType.BIOLOGICAL_LIF:
            # More biologically realistic parameters
            self.threshold = -55.0  # mV
            self.reset_potential = -70.0  # mV
            self.rest_potential = -70.0  # mV
            self.membrane_time_constant = 10.0  # ms (faster)
            self.refractory_period = 2.0  # ms
            self.noise_std = 1.0  # nA (more noise)

@dataclass
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


@dataclass
class LSMState:
    """Current state of the LSM"""
    liquid_spikes: np.ndarray
    liquid_membrane_potentials: np.ndarray
    liquid_state_vector: np.ndarray
    time_step: int
    current_time: float


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


class LiquidStateMachine:
    """
    🌊 Liquid State Machine - Refactored Modular Implementation
    
    Refactored modular implementation of Maass et al. (2002) LSM with clear 
    separation of concerns for better maintainability.
    
    Key Innovation: Biological spiking neurons in a "liquid" that processes temporal information
    Revolutionary because: Shows computation doesn't require stable states - dynamics ARE the computation
    """
    
    def __init__(self, config: Optional[LSMConfig] = None, **kwargs):
        """Initialize Liquid State Machine"""
        
        # Handle backward compatibility for parameter names
        neuron_model = None
        synapse_model = None
        
        if 'n_neurons' in kwargs and 'n_liquid' not in kwargs:
            kwargs['n_liquid'] = kwargs.pop('n_neurons')
        if 'input_dim' in kwargs and 'n_input' not in kwargs:
            kwargs['n_input'] = kwargs.pop('input_dim')
        if 'output_dim' in kwargs and 'n_output' not in kwargs:
            kwargs['n_output'] = kwargs.pop('output_dim')
        if 'neuron_model' in kwargs:
            neuron_model = kwargs.pop('neuron_model')
        if 'synapse_model' in kwargs:
            synapse_model = kwargs.pop('synapse_model')  # Handle by connectivity layer
        if 'connectivity_params' in kwargs:
            connectivity_params = kwargs.pop('connectivity_params')
            # Map connectivity params to config fields
            if 'connection_probability' in connectivity_params:
                kwargs['p_connect'] = connectivity_params['connection_probability']
        if 'random_seed' in kwargs:
            kwargs.pop('random_seed')  # Handle separately
        
        # Use provided config or create default
        if config is None:
            config = LSMConfig(**kwargs)
        
        # Handle neuron model override if specified
        if neuron_model is not None:
            if config.neuron_config is None:
                config.neuron_config = LIFNeuronConfig(model_type=neuron_model)
            else:
                config.neuron_config.model_type = neuron_model
                config.neuron_config.__post_init__()  # Recalculate parameters
        
        self.config = config
        
        # Initialize liquid (reservoir) population
        self.liquid_population = LIFNeuronPopulation(
            n_neurons=config.n_liquid,
            config=config.neuron_config,
            exc_ratio=0.8,  # 80% excitatory as in Maass 2002
            random_positions=True
        )
        
        # Initialize connectivity
        positions = self.liquid_population.get_positions()
        self.connectivity = create_connectivity_matrix(
            n_neurons=config.n_liquid,
            connectivity_type=config.connectivity_type.value,
            p_connect=config.p_connect,
            positions=positions
        )
        
        # Initialize liquid state extractor
        self.state_extractor = create_liquid_state_extractor(
            extractor_type=config.liquid_state_type.value,
            n_liquid=config.n_liquid
        )
        
        # Initialize readout mechanism
        self.readout_mechanism = create_readout_mechanism(
            readout_type=config.readout_type.value,
            n_outputs=config.n_output
        )
        
        # Simulation state
        self.current_time = 0.0
        self.dt = config.dt
        self.spike_history = []  # (time, neuron_id) pairs
        self.spike_matrix = None  # Full spike matrix for analysis
        self.membrane_potential_history = []
        
        # Input/output handling
        self.input_weights = self._initialize_input_weights()
        self.trained = False
        
        print(f"✓ Liquid State Machine initialized")
        print(f"   Liquid neurons: {config.n_liquid} ({config.neuron_config.model_type.value})")
        print(f"   Connectivity: {config.connectivity_type.value} (p={config.p_connect})")
        print(f"   State extraction: {config.liquid_state_type.value}")
        print(f"   Readout: {config.readout_type.value}")
        print(f"   Time step: {config.dt} ms")
    
    # Backward compatibility properties
    @property
    def n_neurons(self) -> int:
        """Backward compatibility: alias for n_liquid"""
        return self.config.n_liquid
    
    @property
    def input_dim(self) -> int:
        """Backward compatibility: alias for n_input"""
        return self.config.n_input
    
    @property
    def output_dim(self) -> int:
        """Backward compatibility: alias for n_output"""
        return self.config.n_output
    
    @property
    def connectivity_type(self):
        """Backward compatibility: alias for connectivity_type"""
        return self.config.connectivity_type
    
    @property  
    def neuron_model(self):
        """Backward compatibility: alias for neuron model type"""
        return self.config.neuron_config.model_type
    
    @property
    def liquid_state_type(self):
        """Backward compatibility: alias for liquid_state_type"""
        return self.config.liquid_state_type
    
    @property
    def synapse_model(self):
        """Backward compatibility: return static synapse model for now"""
        return SynapseModelType.STATIC
    
    @property
    def neurons(self):
        """Backward compatibility: access to neuron population"""
        return self.liquid_population.neurons
    
    def _initialize_input_weights(self) -> np.ndarray:
        """Initialize input->liquid connections"""
        # Random input connections (Maass 2002 style)
        input_weights = np.random.normal(
            0, 1.0, (self.config.n_liquid, self.config.n_input)
        )
        
        # Only connect to ~30% of liquid neurons (sparse input)
        mask = np.random.random((self.config.n_liquid, self.config.n_input)) > 0.7
        input_weights[mask] = 0
        
        return input_weights
    
    def run_liquid(self, input_sequence: np.ndarray, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Run liquid dynamics with input sequence
        
        Args:
            input_sequence: Input data [time_steps, n_inputs]
            duration: Duration in ms (if None, inferred from sequence length)
            
        Returns:
            Dictionary with simulation results
        """
        if duration is None:
            duration = len(input_sequence) * self.dt
        
        n_steps = len(input_sequence)
        
        # Initialize storage
        self.spike_matrix = np.zeros((self.config.n_liquid, n_steps), dtype=bool)
        membrane_potentials = np.zeros((self.config.n_liquid, n_steps))
        liquid_states = np.zeros((n_steps, self.state_extractor.n_liquid))
        
        # Reset components
        self.liquid_population.reset_all()
        self.connectivity.reset_all_synapses()
        self.state_extractor.reset_state()
        self.current_time = 0.0
        
        # Simulation loop
        for step in range(n_steps):
            # Get current input
            current_input = input_sequence[step] if step < len(input_sequence) else np.zeros(self.config.n_input)
            
            # Convert input to currents for liquid neurons
            input_currents = self.input_weights @ current_input
            
            # Get synaptic currents from liquid connectivity
            spike_dict = {i: bool(self.spike_matrix[i, max(0, step-1)]) for i in range(self.config.n_liquid)}
            synaptic_currents = np.zeros(self.config.n_liquid)
            
            for neuron_idx in range(self.config.n_liquid):
                synaptic_currents[neuron_idx] = self.connectivity.get_synaptic_current(
                    neuron_idx, self.dt, spike_dict
                )
            
            # Update liquid neurons
            total_currents = input_currents + synaptic_currents
            spikes = self.liquid_population.update_all(
                dt=self.dt,
                synaptic_inputs=synaptic_currents,
                external_currents=input_currents
            )
            
            # Store results
            self.spike_matrix[:, step] = spikes
            membrane_potentials[:, step] = self.liquid_population.get_membrane_potentials()
            
            # Extract liquid state
            times = np.arange(step + 1) * self.dt
            # Pass membrane potentials only for extractors that need them
            if hasattr(self.state_extractor, '__class__') and 'MembranePotential' in self.state_extractor.__class__.__name__:
                liquid_state = self.state_extractor.extract_state(
                    self.spike_matrix[:, :step+1],
                    times,
                    self.current_time,
                    dt=self.dt,
                    membrane_potentials=membrane_potentials[:, step]
                )
            else:
                liquid_state = self.state_extractor.extract_state(
                    self.spike_matrix[:, :step+1],
                    times,
                    self.current_time,
                    dt=self.dt
                )
            liquid_states[step] = liquid_state[:len(liquid_states[step])]  # Handle dimension mismatch
            
            # Update time
            self.current_time += self.dt
        
        return {
            'spike_matrix': self.spike_matrix,
            'membrane_potentials': membrane_potentials,
            'liquid_states': liquid_states,
            'times': np.arange(n_steps) * self.dt,
            'n_spikes_total': np.sum(self.spike_matrix),
            'mean_firing_rate': np.mean(np.sum(self.spike_matrix, axis=1)) / (duration / 1000)  # Hz
        }
    
    def train_readout(self, input_sequences: List[np.ndarray], 
                     target_sequences: List[np.ndarray]) -> Dict[str, Any]:
        """
        Train readout mechanism on liquid states
        
        Args:
            input_sequences: List of input sequences
            target_sequences: List of target sequences
            
        Returns:
            Training results
        """
        # Collect liquid states for all sequences
        all_liquid_states = []
        all_targets = []
        
        for input_seq, target_seq in zip(input_sequences, target_sequences):
            # Run liquid
            result = self.run_liquid(input_seq)
            liquid_states = result['liquid_states']
            
            # Align with targets (handle different lengths)
            min_len = min(len(liquid_states), len(target_seq))
            all_liquid_states.append(liquid_states[:min_len])
            all_targets.append(target_seq[:min_len])
        
        # Concatenate all data
        features = np.vstack(all_liquid_states)
        targets = np.vstack(all_targets)
        
        # Train readout
        training_results = self.readout_mechanism.train(features, targets)
        self.trained = True
        
        return training_results
    
    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained LSM
        
        Args:
            input_sequence: Input sequence [time_steps, n_inputs]
            
        Returns:
            Predictions [time_steps, n_outputs]
        """
        if not self.trained:
            raise ValueError("LSM must be trained before prediction")
        
        # Run liquid
        result = self.run_liquid(input_sequence)
        liquid_states = result['liquid_states']
        
        # Generate predictions
        predictions = self.readout_mechanism.predict(liquid_states)
        
        return predictions
    
    def compute_memory_capacity(self, max_delay: int = 10, n_trials: int = 20) -> Dict[str, Any]:
        """
        Compute memory capacity of the liquid
        
        Tests ability to remember past inputs at various delays
        """
        from sklearn.linear_model import Ridge
        
        capacities = []
        
        for delay in range(1, max_delay + 1):
            trial_scores = []
            
            for trial in range(n_trials):
                # Generate random input sequence
                sequence_length = 200 + delay
                input_seq = np.random.randn(sequence_length, self.config.n_input)
                
                # Target is delayed input
                target = input_seq[:-delay, 0]  # First input dimension
                
                # Run liquid
                result = self.run_liquid(input_seq)
                liquid_states = result['liquid_states'][delay:]  # Remove first 'delay' states
                
                # Fit linear model
                if len(liquid_states) == len(target):
                    reg = Ridge(alpha=0.01)
                    reg.fit(liquid_states, target)
                    score = reg.score(liquid_states, target)
                    trial_scores.append(max(0, score))  # Clamp to positive
            
            if trial_scores:
                capacities.append(np.mean(trial_scores))
            else:
                capacities.append(0)
        
        total_capacity = sum(capacities)
        
        return {
            'capacities_by_delay': capacities,
            'total_memory_capacity': total_capacity,
            'max_delay_tested': max_delay,
            'n_trials_per_delay': n_trials
        }
    
    def compute_separation_property(self, n_test_pairs: int = 50) -> Dict[str, Any]:
        """
        Test separation property of the liquid
        
        Different inputs should produce different liquid states
        """
        separation_scores = []
        
        for _ in range(n_test_pairs):
            # Generate two different random inputs
            input1 = np.random.randn(100, self.config.n_input)
            input2 = np.random.randn(100, self.config.n_input)
            
            # Run liquid on both
            result1 = self.run_liquid(input1)
            result2 = self.run_liquid(input2)
            
            # Compare final liquid states
            state1 = result1['liquid_states'][-1]
            state2 = result2['liquid_states'][-1]
            
            # Compute separation (normalized distance)
            separation = np.linalg.norm(state1 - state2) / np.sqrt(len(state1))
            separation_scores.append(separation)
        
        return {
            'mean_separation': np.mean(separation_scores),
            'std_separation': np.std(separation_scores),
            'separation_scores': separation_scores
        }
    
    def analyze_liquid_dynamics(self, input_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive analysis of liquid dynamics
        
        Returns various metrics about the liquid's computational properties
        """
        # Run liquid
        result = self.run_liquid(input_sequence)
        
        # Connectivity analysis
        conn_stats = self.connectivity.get_connectivity_stats()
        
        # Activity statistics
        spike_matrix = result['spike_matrix']
        membrane_potentials = result['membrane_potentials']
        
        firing_rates = np.sum(spike_matrix, axis=1) / (len(input_sequence) * self.dt / 1000)  # Hz
        mean_membrane_potential = np.mean(membrane_potentials, axis=1)
        
        # Synchronization analysis (simplified)
        population_activity = np.sum(spike_matrix, axis=0)
        synchrony_index = np.std(population_activity) / np.mean(population_activity) if np.mean(population_activity) > 0 else 0
        
        # Liquid state diversity
        liquid_states = result['liquid_states']
        state_diversity = np.mean(np.std(liquid_states, axis=0))
        
        return {
            'connectivity_stats': conn_stats,
            'mean_firing_rate': float(np.mean(firing_rates)),
            'std_firing_rate': float(np.std(firing_rates)),
            'mean_membrane_potential': float(np.mean(mean_membrane_potential)),
            'synchrony_index': float(synchrony_index),
            'state_diversity': float(state_diversity),
            'total_spikes': int(result['n_spikes_total']),
            'liquid_states_shape': liquid_states.shape,
            'simulation_duration_ms': float(len(input_sequence) * self.dt)
        }
    
    def visualize_activity(self, result: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize liquid activity
        
        Creates plots showing spike raster, membrane potentials, and liquid states
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        spike_matrix = result['spike_matrix']
        membrane_potentials = result['membrane_potentials']
        liquid_states = result['liquid_states']
        times = result['times']
        
        # Spike raster plot
        spike_times, spike_neurons = np.where(spike_matrix)
        spike_times = spike_times * self.dt  # Convert to time
        axes[0, 0].scatter(spike_times, spike_neurons, s=0.5, alpha=0.6)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Neuron Index')
        axes[0, 0].set_title('Spike Raster Plot')
        
        # Population activity
        pop_activity = np.sum(spike_matrix, axis=0)
        axes[0, 1].plot(times, pop_activity)
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Population Spike Count')
        axes[0, 1].set_title('Population Activity')
        
        # Sample membrane potentials
        n_show = min(10, self.config.n_liquid)
        for i in range(n_show):
            axes[1, 0].plot(times, membrane_potentials[i], alpha=0.7, linewidth=0.5)
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Membrane Potential (mV)')
        axes[1, 0].set_title(f'Sample Membrane Potentials ({n_show} neurons)')
        
        # Liquid state evolution
        if liquid_states.shape[1] > 1:
            # Show first few dimensions
            n_dims = min(5, liquid_states.shape[1])
            for i in range(n_dims):
                axes[1, 1].plot(times, liquid_states[:, i], alpha=0.8, label=f'Dim {i}')
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(times, liquid_states[:, 0])
        
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Liquid State Value')
        axes[1, 1].set_title('Liquid State Evolution')
        
        plt.tight_layout()
        return fig
    
    def save_state(self, filepath: str):
        """Save LSM state to file"""
        state_dict = {
            'config': self.config,
            'input_weights': self.input_weights,
            'trained': self.trained,
            'current_time': self.current_time
        }
        np.savez(filepath, **state_dict)
    
    def load_state(self, filepath: str):
        """Load LSM state from file"""
        state = np.load(filepath, allow_pickle=True)
        self.config = state['config'].item()
        self.input_weights = state['input_weights']
        self.trained = bool(state['trained'])
        self.current_time = float(state['current_time'])
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about LSM configuration"""
        return {
            'config': {
                'n_liquid': self.config.n_liquid,
                'n_input': self.config.n_input,
                'n_output': self.config.n_output,
                'neuron_model': self.config.neuron_config.model_type.value,
                'connectivity_type': self.config.connectivity_type.value,
                'liquid_state_type': self.config.liquid_state_type.value,
                'readout_type': self.config.readout_type.value,
                'dt': self.config.dt
            },
            'connectivity_stats': self.connectivity.get_connectivity_stats(),
            'trained': self.trained,
            'current_time': self.current_time
        }
    
    def __repr__(self):
        return (f"LiquidStateMachine(n_liquid={self.config.n_liquid}, "
                f"connectivity={self.config.connectivity_type.value}, "
                f"state_extraction={self.config.liquid_state_type.value}, "
                f"readout={self.config.readout_type.value})")


def demonstrate_lsm_features():
    """Demonstrate key LSM features"""
    print("🌊 Liquid State Machine Demonstration")
    print("=" * 50)
    
    # Create LSM with Maass 2002 configuration
    config = LSMConfig(
        n_liquid=100,
        n_input=2,
        n_output=1,
        neuron_config=None,  # Use default MAASS_2002_LIF
        dt=0.1  # 0.1ms timestep
    )
    
    lsm = LiquidStateMachine(config)
    
    # Generate sample input (sine waves)
    t = np.arange(0, 200, config.dt)  # 200ms
    input1 = np.sin(2 * np.pi * t / 50)  # 20Hz sine wave
    input2 = np.cos(2 * np.pi * t / 30)  # ~33Hz cosine wave
    input_sequence = np.column_stack([input1, input2])
    
    # Run liquid
    result = lsm.run_liquid(input_sequence)
    print(f"✓ Liquid simulation complete: {result['n_spikes_total']} total spikes")
    print(f"  Mean firing rate: {result['mean_firing_rate']:.1f} Hz")
    
    # Analyze dynamics
    analysis = lsm.analyze_liquid_dynamics(input_sequence)
    print(f"✓ Liquid analysis:")
    print(f"  Connectivity density: {analysis['connectivity_stats']['density']:.3f}")
    print(f"  State diversity: {analysis['state_diversity']:.3f}")
    print(f"  Synchrony index: {analysis['synchrony_index']:.3f}")
    
    # Test memory capacity
    memory_result = lsm.compute_memory_capacity(max_delay=5, n_trials=5)
    print(f"✓ Memory capacity: {memory_result['total_memory_capacity']:.2f}")
    
    print("\n🎉 LSM demonstration complete!")
    
    return lsm