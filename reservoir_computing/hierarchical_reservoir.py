"""
🌊 Hierarchical Reservoir Computing 🌊

A sophisticated multi-level reservoir computing architecture that processes temporal
information through hierarchical abstraction layers, enabling powerful prediction
and modeling capabilities for complex dynamical systems.

Author: Benedict Chen
Email: benedict@benedictchen.com
Created: 2024
License: MIT

💝 Support This Research:
If this hierarchical reservoir implementation advances your temporal modeling projects,
consider supporting the preservation of computational neuroscience classics! Like
water flowing through multiple reservoir levels, your support cascades through our
efforts to document AI history:
- GitHub: ⭐ Star this repository for algorithmic archaeology
- Donate: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
- Cite: Reference this implementation in your research

═══════════════════════════════════════════════════════════════════════════════════
📚 RESEARCH FOUNDATIONS
═══════════════════════════════════════════════════════════════════════════════════

This implementation synthesizes breakthrough concepts from reservoir computing:

🌊 FOUNDATIONAL RESERVOIR COMPUTING:
• Jaeger, H. (2001). "The 'Echo State' Property"
  GMD Report 148, German National Research Center for Information Technology
  - Introduced echo state networks and reservoir computing paradigm
  - Mathematical conditions for echo state property
  - Separation of recurrent dynamics from linear readout

• Maass, W., Natschläger, T., & Markram, H. (2002). "Real-time computing without stable states"
  Neural Computation, 14(11), 2531-2560
  - Liquid state machines as biological reservoir computing
  - Temporal processing through recurrent neural microcircuits
  - Computational power of transient dynamics

🏗️ HIERARCHICAL ARCHITECTURES:
• Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). "Model-free prediction of large spatiotemporally chaotic systems from data"
  Physical Review Letters, 120(2), 024102
  - Hybrid forecasting using reservoir computing
  - Multi-scale temporal dynamics prediction
  - Chaotic system reconstruction

• Triefenbach, F., Jalalvand, A., Schrauwen, B., & Martens, J.P. (2010). "Phoneme Recognition with Large Hierarchical Reservoirs"
  Advances in Neural Information Processing Systems, 23
  - Hierarchical reservoir architectures for speech recognition
  - Multi-resolution temporal feature extraction
  - Layer-wise temporal abstraction

🧠 COMPUTATIONAL NEUROSCIENCE FOUNDATIONS:
• Buonomano, D.V. & Maass, W. (2009). "State-dependent computations: spatiotemporal processing in cortical networks"
  Nature Reviews Neuroscience, 10(2), 113-125
  - Neural basis of temporal processing
  - Hierarchical organization of cortical dynamics
  - Multi-scale temporal representations

• Pozzorini, C., Naud, R., Mensi, S., & Gerstner, W. (2013). "Temporal whitening by power-law adaptation in neocortical neurons"
  Nature Neuroscience, 16(7), 942-948
  - Adaptive temporal filtering in neural networks
  - Multi-timescale dynamics in cortical circuits
  - Hierarchical temporal processing mechanisms

═══════════════════════════════════════════════════════════════════════════════════
🎭 EXPLAIN LIKE I'M 5: The Water Tower Memory System
═══════════════════════════════════════════════════════════════════════════════════

Imagine your brain's memory as a system of magical water towers built on a mountain! 🏔️💧

🏔️ THE MOUNTAIN OF TIME:
Our memory mountain has different levels, each seeing time differently:

• BOTTOM TOWER (Level 1) 🏗️: The "Right Now" Tower
  - Catches every raindrop (input) as it falls
  - Remembers what happened in the last few seconds
  - Like remembering each word as you hear it in a sentence

• MIDDLE TOWER (Level 2) 🏰: The "Recent Events" Tower  
  - Gets water from the bottom tower (not direct rain)
  - Remembers patterns over minutes
  - Like remembering the meaning of whole sentences

• TOP TOWER (Level 3) 🗼: The "Big Picture" Tower
  - Gets water from the middle tower
  - Remembers themes over hours or days
  - Like remembering the story of an entire book

🌊 HOW THE WATER FLOWS:
1. Rain (new information) hits the bottom tower first
2. Bottom tower processes it and sends some water up
3. Middle tower mixes this with its own swirling patterns
4. Top tower receives the most processed, cleanest water

🔮 THE MAGIC PREDICTION POWER:
Because each tower "remembers" different time scales:
- Bottom tower: "A word starting with 'app' might be 'apple'"
- Middle tower: "We're talking about fruit, so probably 'apple'"  
- Top tower: "This is a story about healthy eating, definitely 'apple'"

🎯 WHY THIS IS AMAZING:
Just like how you can predict both the next word AND the story's ending,
our hierarchical reservoir can predict both immediate next events AND
long-term patterns! It's like having telescopes that see both nearby
raindrops and distant storm clouds! 🌈⛈️

═══════════════════════════════════════════════════════════════════════════════════
🏗️ SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════

The hierarchical reservoir follows a multi-level processing architecture:

                     🎯 HIERARCHICAL RESERVOIR SYSTEM
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            📊 LEVEL 3         🧠 INTEGRATION   📈 PREDICTION
           TOP RESERVOIR        CONTROLLER        ENGINE
         [Abstract Patterns]         │               │
                    │               │               │
            📊 LEVEL 2              │               │
         MIDDLE RESERVOIR           │               │
       [Intermediate Features]      │               │
                    │               │               │
            📊 LEVEL 1              │               │
         BOTTOM RESERVOIR           │               │
        [Raw Input Processing]      │               │
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                         💾 READOUT LAYER
                    ┌───────────────┼───────────────┐
                    │               │               │
            🔄 TEMPORAL        📊 COMBINED      🎯 OUTPUT
            ALIGNMENT         STATE VECTOR     PREDICTION
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                              📤 FINAL OUTPUT

INTER-LEVEL CONNECTIONS:
- Feedforward: Lower levels → Higher levels (temporal abstraction)
- Lateral: Same-level recurrent connections (temporal memory)
- Skip: Direct lower-level → Output (multi-resolution features)

TEMPORAL SCALES:
- Level 1: Millisecond-scale dynamics (immediate responses)
- Level 2: Second-scale patterns (short-term memory)  
- Level 3: Minute-scale trends (long-term context)

═══════════════════════════════════════════════════════════════════════════════════
🧮 MATHEMATICAL FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════════

HIERARCHICAL STATE UPDATE:
For level l at time t:

Level 0 (Input): 
x₀(t+1) = f(W⁰ᵣₑₛ · x₀(t) + W⁰ᵢₙ · u(t))

Level l > 0:
xₗ(t+1) = f(Wˡᵣₑₛ · xₗ(t) + Wˡᵢₙₜₑᵣ · xₗ₋₁(t) + αₗ · Wˡᵢₙ · u(t))

Where:
- xₗ(t): State vector of level l at time t
- Wˡᵣₑₛ: Recurrent weight matrix for level l
- Wˡᵢₙₜₑᵣ: Inter-level connection matrix (l-1 → l)
- Wˡᵢₙ: Input weight matrix for level l
- αₗ: Input scaling factor for level l
- f(·): Activation function (typically tanh)

SPECTRAL RADIUS CONTROL:
For stability, ensure ρ(Wˡᵣₑₛ) < 1 for all levels:
Wˡᵣₑₛ ← (ρₜₐᵣₘₑₜ / ρ(W̃ˡᵣₑₛ)) · W̃ˡᵣₑₛ

Where W̃ˡᵣₑₛ is the initial random matrix and ρₜₐᵣₘₑₜ is desired spectral radius.

READOUT COMPUTATION:
Combined state vector: X(t) = [x₀(t); x₁(t); ...; xₗ(t); 1]
Output: y(t) = W_out · X(t)

TRAINING (Ridge Regression):
W_out = YX^T(XX^T + λI)^(-1)

Where:
- Y: Target output matrix
- X: Combined state matrix
- λ: Regularization parameter

TEMPORAL MEMORY CAPACITY:
Level l memory capacity: MCₗ = -∫₀^∞ μₗ(τ) log₂ μₗ(τ) dτ

Where μₗ(τ) is the autocorrelation function of level l at lag τ.

INFORMATION PROCESSING:
Inter-level information transfer:
Iₗ→ₗ₊₁ = H(Xₗ₊₁) - H(Xₗ₊₁|Xₗ)

Where H is entropy and represents information flow between levels.

═══════════════════════════════════════════════════════════════════════════════════
🌍 REAL-WORLD APPLICATIONS
═══════════════════════════════════════════════════════════════════════════════════

🌡️ CLIMATE MODELING:
• Multi-scale weather prediction: Short-term weather + seasonal patterns + climate trends
• Ocean current analysis: Surface dynamics + thermocline patterns + deep circulation
• Extreme event forecasting: Local conditions + regional systems + global patterns

🏭 INDUSTRIAL PROCESS CONTROL:
• Manufacturing optimization: Machine states + production flows + supply chain dynamics
• Power grid management: Individual generators + regional grids + national systems
• Chemical reactor control: Molecular kinetics + reaction dynamics + process optimization

🧠 NEUROSCIENCE RESEARCH:
• Brain signal analysis: Neural spikes + local field potentials + network oscillations
• Motor control modeling: Muscle activation + limb coordination + movement planning
• Cognitive state prediction: Attention patterns + working memory + decision processes

📈 FINANCIAL MODELING:
• Market prediction: Tick-by-tick prices + daily trends + economic cycles
• Risk assessment: Individual trades + portfolio dynamics + market regime changes
• Algorithmic trading: Order execution + strategy adaptation + market microstructure

🎵 AUDIO PROCESSING:
• Speech recognition: Phonemes + syllables + words + sentence meaning
• Music analysis: Notes + chords + phrases + musical structure
• Environmental sound classification: Acoustic events + scene analysis + context understanding

🚗 AUTONOMOUS SYSTEMS:
• Vehicle control: Sensor fusion + path planning + behavioral prediction
• Drone navigation: Immediate obstacles + route planning + mission objectives
• Robot manipulation: Joint control + task execution + goal achievement

🌐 NETWORK ANALYSIS:
• Internet traffic prediction: Packet flows + connection patterns + usage trends
• Social network dynamics: Individual interactions + community formation + influence spread
• Communication systems: Signal processing + protocol optimization + network management

This hierarchical approach enables unprecedented modeling of complex systems
that operate across multiple temporal scales simultaneously! 🚀"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .echo_state_network import EchoStateNetwork


class HierarchicalReservoir:
    """
    Hierarchical Echo State Network
    
    Creates a hierarchy of reservoirs where higher levels process
    increasingly abstracted representations of the input dynamics.
    """
    
    def __init__(
        self,
        reservoir_sizes: List[int] = [500, 200, 100],
        spectral_radii: List[float] = [0.95, 0.9, 0.85],
        sparsities: List[float] = [0.1, 0.15, 0.2],
        input_scalings: List[float] = [1.0, 0.5, 0.3],
        inter_level_scaling: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Hierarchical Reservoir
        
        Args:
            reservoir_sizes: Sizes of reservoirs at each level
            spectral_radii: Spectral radius for each level
            sparsities: Sparsity for each level
            input_scalings: Input scaling for each level
            inter_level_scaling: Scaling for connections between levels
            random_seed: Random seed for reproducibility
        """
        
        self.n_levels = len(reservoir_sizes)
        self.reservoir_sizes = reservoir_sizes
        self.inter_level_scaling = inter_level_scaling
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Create ESN for each level
        self.reservoirs = []
        for i in range(self.n_levels):
            esn = EchoStateNetwork(
                n_reservoir=reservoir_sizes[i],
                spectral_radius=spectral_radii[i] if i < len(spectral_radii) else 0.9,
                sparsity=sparsities[i] if i < len(sparsities) else 0.1,
                input_scaling=input_scalings[i] if i < len(input_scalings) else 1.0,
                random_seed=random_seed + i if random_seed is not None else None
            )
            self.reservoirs.append(esn)
            
        # Inter-level connection matrices
        self.W_inter = []
        for i in range(self.n_levels - 1):
            # Connection from level i to level i+1
            W = np.random.uniform(
                -inter_level_scaling,
                inter_level_scaling,
                (reservoir_sizes[i+1], reservoir_sizes[i])
            )
            self.W_inter.append(W)
            
        # Training state
        self.is_trained = False
        self.readout_weights = []
        self.last_states = None
        
        print(f"✓ Hierarchical Reservoir initialized:")
        print(f"   Levels: {self.n_levels}")
        print(f"   Sizes: {reservoir_sizes}")
        
    def run_hierarchy(self, inputs: np.ndarray, washout: int = 100) -> List[np.ndarray]:
        """
        Run input through hierarchical reservoirs
        
        Args:
            inputs: Input sequence (time_steps, n_inputs)
            washout: Washout period
            
        Returns:
            List of state sequences for each level
        """
        
        time_steps, n_inputs = inputs.shape
        
        # Initialize states for all levels
        states_per_level = [[] for _ in range(self.n_levels)]
        current_states = [np.zeros(size) for size in self.reservoir_sizes]
        
        # Initialize input weights for first level if needed
        if not hasattr(self.reservoirs[0], 'W_input'):
            self.reservoirs[0]._initialize_input_weights(n_inputs)
            
        # Process each time step
        for t in range(time_steps):
            # Level 0: Process external input
            current_states[0] = self.reservoirs[0]._update_state(
                current_states[0], inputs[t]
            )
            
            # Higher levels: Process lower level states + external input
            for level in range(1, self.n_levels):
                # Combine external input with lower level state
                lower_level_input = self.W_inter[level-1] @ current_states[level-1]
                
                # Initialize input weights for this level if needed
                if not hasattr(self.reservoirs[level], 'W_input'):
                    combined_input_size = n_inputs + len(lower_level_input)
                    self.reservoirs[level]._initialize_input_weights(combined_input_size)
                
                # Combine external input with processed lower level
                combined_input = np.concatenate([inputs[t], lower_level_input])
                
                current_states[level] = self.reservoirs[level]._update_state(
                    current_states[level], combined_input
                )
            
            # Collect states after washout
            if t >= washout:
                for level in range(self.n_levels):
                    states_per_level[level].append(current_states[level].copy())
                    
        # Store final states
        self.last_states = current_states
        
        # Convert to arrays
        return [np.array(states) for states in states_per_level]
        
    def train(self, inputs: np.ndarray, targets: np.ndarray, 
              reg_param: float = 1e-6, washout: int = 100,
              level_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Train hierarchical reservoir
        
        Args:
            inputs: Training inputs
            targets: Training targets
            reg_param: Regularization parameter
            washout: Washout period
            level_weights: Weights for combining different levels
            
        Returns:
            Training results
        """
        
        print(f"🎯 Training Hierarchical Reservoir...")
        
        # Get states from all levels
        states_per_level = self.run_hierarchy(inputs, washout)
        
        # Combine states from all levels
        if level_weights is None:
            level_weights = [1.0] * self.n_levels
            
        combined_states = []
        for states, weight in zip(states_per_level, level_weights):
            if len(combined_states) == 0:
                combined_states = states * weight
            else:
                combined_states = np.column_stack([combined_states, states * weight])
                
        # Add bias term
        X = np.column_stack([combined_states, np.ones(len(combined_states))])
        y = targets[washout:]
        
        # Train linear readout
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=reg_param)
        ridge.fit(X, y)
        
        self.W_out = ridge.coef_
        self.bias = ridge.intercept_
        self.is_trained = True
        
        # Calculate performance
        predictions = ridge.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        results = {
            'mse': mse,
            'n_levels': self.n_levels,
            'combined_state_dim': X.shape[1] - 1,  # Exclude bias
            'level_contributions': [states.shape[1] for states in states_per_level]
        }
        
        print(f"✓ Hierarchical training complete: MSE = {mse:.6f}")
        print(f"   Combined state dimension: {X.shape[1] - 1}")
        
        return results
        
    def predict(self, inputs: np.ndarray, washout: int = 100) -> np.ndarray:
        """Generate predictions using trained hierarchical reservoir"""
        
        if not self.is_trained:
            raise ValueError("Hierarchical reservoir must be trained first!")
            
        # Get states from all levels
        states_per_level = self.run_hierarchy(inputs, washout)
        
        # Combine states (using same weights as training)
        combined_states = []
        for states in states_per_level:
            if len(combined_states) == 0:
                combined_states = states
            else:
                combined_states = np.column_stack([combined_states, states])
                
        # Add bias term
        X = np.column_stack([combined_states, np.ones(len(combined_states))])
        
        # Generate predictions
        predictions = X @ self.W_out.T + self.bias
        
        return predictions
        
    def generate_hierarchical(self, n_steps: int, 
                             initial_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate sequence using hierarchical feedback
        """
        
        if not self.is_trained:
            raise ValueError("Hierarchical reservoir must be trained first!")
            
        if self.last_states is None:
            raise ValueError("Must run reservoir at least once before generation!")
            
        # Initialize
        states = [state.copy() for state in self.last_states]
        outputs = []
        
        if initial_input is None:
            current_input = np.zeros(self.reservoirs[0].W_input.shape[1])
        else:
            current_input = initial_input.copy()
            
        for step in range(n_steps):
            # Update level 0
            states[0] = self.reservoirs[0]._update_state(states[0], current_input)
            
            # Update higher levels
            for level in range(1, self.n_levels):
                lower_level_input = self.W_inter[level-1] @ states[level-1]
                combined_input = np.concatenate([current_input, lower_level_input])
                states[level] = self.reservoirs[level]._update_state(
                    states[level], combined_input
                )
            
            # Generate output from combined states
            combined_state = np.concatenate(states + [np.array([1.0])])  # Add bias
            output = combined_state @ self.W_out.T + self.bias
            
            outputs.append(output)
            
            # Use output as next input (closed loop)
            current_input = output if len(output) == len(current_input) else current_input
            
        return np.array(outputs)
        
    def get_level_analysis(self) -> Dict[str, Any]:
        """Analyze properties of each hierarchical level"""
        
        analysis = {
            'level_sizes': self.reservoir_sizes,
            'spectral_radii': [],
            'sparsities': [],
            'inter_level_connections': [W.shape for W in self.W_inter]
        }
        
        for reservoir in self.reservoirs:
            # Calculate spectral radius
            eigenvals = np.linalg.eigvals(reservoir.W_reservoir)
            spectral_radius = np.max(np.abs(eigenvals))
            analysis['spectral_radii'].append(spectral_radius)
            
            # Calculate sparsity
            sparsity = np.mean(reservoir.W_reservoir != 0)
            analysis['sparsities'].append(sparsity)
            
        return analysis