# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/reservoir-computing/workflows/CI/badge.svg)](https://github.com/benedictchen/reservoir-computing/actions)
[![PyPI version](https://badge.fury.io/py/reservoir-computing-benedictchen.svg)](https://badge.fury.io/py/reservoir-computing-benedictchen)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Reservoir Computing

🧠 Echo State Networks and Liquid State Machines for temporal pattern recognition and neuromorphic computing

**Jaeger, H. (2001)** - "The Echo State Approach to Analysing and Training Recurrent Neural Networks"  
**Maass, W., Natschläger, T., & Markram, H. (2002)** - "Real-time computing without stable states"

## 📦 Installation

```bash
pip install reservoir-computing-benedictchen
```

## 🚀 Quick Start

### Echo State Network Example
```python
from reservoir_computing import EchoStateNetwork
import numpy as np

# Create ESN for time series prediction
esn = EchoStateNetwork(
    input_size=1,
    reservoir_size=100, 
    output_size=1,
    spectral_radius=0.95,
    input_scaling=1.0,
    leak_rate=0.3
)

# Generate sample data (sine wave)
X = np.sin(np.linspace(0, 20*np.pi, 1000)).reshape(-1, 1)
y = np.sin(np.linspace(0.1, 20*np.pi + 0.1, 1000)).reshape(-1, 1)

# Train the network
esn.fit(X[:800], y[:800])

# Make predictions
predictions = esn.predict(X[800:])
```

### Liquid State Machine Example  
```python
from reservoir_computing import LiquidStateMachine
import numpy as np

# Create LSM with spiking neurons
lsm = LiquidStateMachine(
    input_size=10,
    liquid_size=200,
    output_size=3,
    neuron_model='lif',  # Leaky Integrate-and-Fire
    connection_probability=0.1
)

# Spike train input
spike_data = np.random.poisson(0.1, (100, 10))
targets = np.random.randint(0, 3, 100)

# Train the readout
lsm.fit(spike_data[:80], targets[:80])

# Classify spike patterns
predictions = lsm.predict(spike_data[80:])
```

## 🧬 Advanced Features

### Modular Architecture
```python
from reservoir_computing.esn_modules import (
    EchoStateNetworkCore,
    PropertyValidator, 
    TopologyManagement,
    TrainingMethods
)

from reservoir_computing.lsm_modules import (
    LiquidStateMachineCore,
    NeuronModels,
    ConnectivityPatterns,
    StateExtractors
)
```

### Hierarchical Reservoirs
```python
from reservoir_computing import HierarchicalReservoir

# Multi-layer reservoir computing
hierarchical = HierarchicalReservoir(
    layers=[
        {'size': 50, 'spectral_radius': 0.9},
        {'size': 100, 'spectral_radius': 0.95},
        {'size': 50, 'spectral_radius': 0.8}
    ],
    inter_layer_scaling=0.1
)
```

## 🔬 Research Foundation

This implementation provides research-accurate implementations of:

- **Echo State Networks (ESN)**: Jaeger's pioneering recurrent neural network approach with the echo state property
- **Liquid State Machines (LSM)**: Maass's biologically-inspired computing paradigm using spiking neural networks  
- **Neuromorphic Computing**: Brain-inspired computing architectures for temporal processing
- **Reservoir Computing Theory**: Mathematical foundations of recurrent neural network dynamics

### Key Algorithmic Features
- **Spectral Radius Optimization**: Ensuring echo state property maintenance
- **Leaky Integrator Neurons**: Biologically-plausible neuron dynamics
- **Sparse Connectivity**: Efficient reservoir topologies
- **Spike-Based Processing**: Event-driven neuromorphic computation
- **Temporal Memory**: Fading memory properties for sequence processing

## 📊 Implementation Highlights

- **Research Accuracy**: Faithful implementation of original papers
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Performance Optimized**: NumPy/SciPy backend for computational efficiency  
- **Educational Value**: Clear code structure for learning reservoir computing
- **Extensible**: Easy to modify and extend for research applications

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**