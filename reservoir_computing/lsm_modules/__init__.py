"""
Liquid State Machine Modular Components

This module provides all functionality from liquid_state_machine_original.py
broken down into logical, manageable components while preserving 100% of
the original functionality.

Based on: Wolfgang Maass (2002) "Real-time computing without stable states"
"""

# Import configuration components
from .lsm_enums import *
from .lsm_configs import *

# Import core components  
from .neuron_models import *
from .synapse_models import *
from .state_extractors import *
from .readout_mechanisms import *
from .connectivity_patterns import *

# Import main LSM class
from .liquid_state_machine_core import LiquidStateMachine

# Import analysis and benchmarking
from .benchmark_tasks import *
from .theoretical_analysis import *

# Re-export main classes for backward compatibility
__all__ = [
    # Enums
    'NeuronModelType', 'SynapseModelType', 'ConnectivityType', 'LiquidStateType', 'ReadoutType',
    
    # Configuration
    'LIFNeuronConfig', 'LSMConfig', 'DynamicSynapseConfig',
    
    # Core models
    'LIFNeuron', 'BiologicalLIFNeuron', 'DynamicSynapse', 'AlternativeDynamicSynapse',
    
    # State extraction
    'LiquidStateExtractor', 'PSPDecayExtractor', 'SpikeCountExtractor', 
    'MembranePotentialExtractor', 'FiringRateExtractor', 'MultiTimescaleExtractor',
    
    # Readout mechanisms
    'ReadoutMechanism', 'LinearReadout', 'PopulationReadout', 'PerceptronReadout', 
    'SVMReadout', 'PopulationReadoutNeurons',
    
    # Main class
    'LiquidStateMachine',
    
    # Connectivity
    'ColumnBasedConnectivity',
    
    # Analysis and benchmarking
    'MaassBenchmarkTasks', 'LSMTheoreticalAnalysis'
]
