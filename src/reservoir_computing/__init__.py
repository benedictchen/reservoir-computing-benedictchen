"""
Reservoir Computing Library
Based on: Jaeger (2001) Echo State Networks & Maass (2002) Liquid State Machines

This library implements the revolutionary concept of fixed random reservoirs
with trainable readout layers, enabling efficient temporal pattern processing.
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\n🌊 Reservoir Computing Library - Made possible by Benedict Chen")
        print("   \033]8;;mailto:benedict@benedictchen.com\033\\benedict@benedictchen.com\033]8;;\033\\")
        print("   Support his work: \033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\033\\🍺 Buy him a beer\033]8;;\033\\")
    except:
        print("\n🌊 Reservoir Computing Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("   Support: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")

# Import REAL implementations directly - no stubs or fallbacks!

# Core Echo State Network implementation - research-grade quality
from .echo_state_network import (
    EchoStateNetwork,
    EchoStatePropertyValidator,
    StructuredReservoirTopologies, 
    JaegerBenchmarkTasks,
    OutputFeedbackESN,
    TeacherForcingTrainer,
    OnlineLearningESN,
    optimize_spectral_radius,
    validate_esp,
    run_benchmark_suite
)

# Alternative import path for core ESN with factory function
from .esn_modules.esn_core import create_echo_state_network

ECHO_STATE_AVAILABLE = True

# Try to import Liquid State Machine (may not exist)
try:
    from .liquid_state_machine import (
        LiquidStateMachine,
        LSMConfig,
        NeuronModelType,
        SynapseModelType,
        ConnectivityType,
        LiquidStateType,
        ReadoutType,
        LIFNeuron,
        DynamicSynapse,
        LiquidStateExtractor,
        PSPDecayExtractor,
        SpikeCountExtractor,
        MembranePotentialExtractor,
        FiringRateExtractor,
        MultiTimescaleExtractor,
        ReadoutMechanism,
        LinearReadout,
        PopulationReadout,
        LSMTheoreticalAnalysis,
        MaassBenchmarkTasks,
        create_lsm_with_presets,
        run_lsm_benchmark_suite
    )
    LSM_AVAILABLE = True
except ImportError:
    LSM_AVAILABLE = False
    # Provide placeholder classes
    LiquidStateMachine = None
    LSMConfig = None

# Try to import other components
try:
    from .hierarchical_reservoir import HierarchicalReservoir
except ImportError:
    HierarchicalReservoir = None
    
try:
    from .reservoir_optimizer import ReservoirOptimizer
except ImportError:
    ReservoirOptimizer = None
    
try:
    from .neuromorphic_interface import NeuromorphicReservoir  
except ImportError:
    NeuromorphicReservoir = None

# Show attribution on library import
_print_attribution()

__version__ = "1.0.0"
__authors__ = ["Based on Jaeger (2001)", "Maass et al. (2002)"]

# Build __all__ based on what's actually available
__all__ = []

# Always include the core EchoStateNetwork (either real or fallback)
__all__.append("EchoStateNetwork")

# Add ESN features if available
if ECHO_STATE_AVAILABLE:
    __all__.extend([
        "EchoStatePropertyValidator",
        "StructuredReservoirTopologies",
        "JaegerBenchmarkTasks",
        "OutputFeedbackESN",
        "TeacherForcingTrainer", 
        "OnlineLearningESN",
        "optimize_spectral_radius",
        "validate_esp",
        "run_benchmark_suite"
    ])

# Add LSM features if available
if LSM_AVAILABLE:
    __all__.extend([
        "LiquidStateMachine",
        "LSMConfig",
        "NeuronModelType",
        "SynapseModelType", 
        "ConnectivityType",
        "LiquidStateType",
        "ReadoutType",
        "LIFNeuron",
        "DynamicSynapse",
        "LiquidStateExtractor",
        "PSPDecayExtractor",
        "SpikeCountExtractor", 
        "MembranePotentialExtractor",
        "FiringRateExtractor",
        "MultiTimescaleExtractor",
        "ReadoutMechanism",
        "LinearReadout",
        "PopulationReadout",
        "LSMTheoreticalAnalysis",
        "MaassBenchmarkTasks",
        "create_lsm_with_presets",
        "run_lsm_benchmark_suite"
    ])

# Add optional components if they exist
if HierarchicalReservoir is not None:
    __all__.append("HierarchicalReservoir")
if ReservoirOptimizer is not None:
    __all__.append("ReservoirOptimizer")
if NeuromorphicReservoir is not None:
    __all__.append("NeuromorphicReservoir")