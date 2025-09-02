"""
Echo State Network Modular Components

This module provides a completely modularized Echo State Network implementation
based on Herbert Jaeger's seminal 2001 paper. The original 2529-line monolithic
file has been broken down into 8 focused, research-grade modules while preserving
100% of the original functionality and adding comprehensive enhancements.

🌊 Modular Architecture:
- ReservoirInitializationMixin: Reservoir setup and weight initialization
- EspValidationMixin: Echo State Property validation methods  
- StateUpdatesMixin: Core dynamics and temporal processing
- TrainingMethodsMixin: Linear readout training and optimization
- PredictionGenerationMixin: Prediction and autonomous generation
- TopologyManagementMixin: Network topology creation and management
- ConfigurationOptimizationMixin: Parameter optimization and tuning
- VisualizationMixin: Comprehensive analysis and visualization
- EchoStateNetwork: Main integrated class using all mixins
"""

# Import the new modular ESN implementation
from .esn_core import EchoStateNetwork, create_echo_state_network

# Import all modular components for advanced users
from .reservoir_initialization import ReservoirInitializationMixin
from .esp_validation import EspValidationMixin
from .state_updates import StateUpdatesMixin
from .training_methods import TrainingMethodsMixin
from .prediction_generation import PredictionGenerationMixin
from .topology_management import TopologyManagementMixin
from .configuration_optimization import ConfigurationOptimizationMixin
from .visualization import VisualizationMixin

# Legacy imports for backward compatibility (if they exist)
try:
    from .echo_state_network_core import EchoStateNetwork as LegacyEchoStateNetwork
except ImportError:
    LegacyEchoStateNetwork = None

# Export main classes and factory function
__all__ = [
    # Main classes
    'EchoStateNetwork',
    'create_echo_state_network',
    
    # Modular components
    'ReservoirInitializationMixin',
    'EspValidationMixin', 
    'StateUpdatesMixin',
    'TrainingMethodsMixin',
    'PredictionGenerationMixin',
    'TopologyManagementMixin',
    'ConfigurationOptimizationMixin',
    'VisualizationMixin',
    
    # Legacy compatibility
    'LegacyEchoStateNetwork'
]
