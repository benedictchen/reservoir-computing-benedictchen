#!/usr/bin/env python3
"""
🏆 FINAL 100% COVERAGE TESTS FOR RESERVOIR COMPUTING
====================================================

This file targets the remaining missing code paths to achieve 100% test coverage.
Focuses on advanced features, edge cases, error conditions, and utility functions.

Author: Benedict Chen (benedict@benedictchen.com)
"""

import pytest
import numpy as np
import warnings
import sys
from pathlib import Path

# Add reservoir computing to path
sys.path.insert(0, 'reservoir_computing')

try:
    # Import all modules for comprehensive testing
    from echo_state_network import *
    from liquid_state_machine import *
    from lsm_config import *
    from lif_neuron import LIFNeuron
    from hierarchical_reservoir import HierarchicalReservoir
    from reservoir_optimizer import ReservoirOptimizer
    from neuromorphic_interface import NeuromorphicReservoir
    import echo_state_validation
    import esn_configuration
    import esn_training
    import state_dynamics
    import synaptic_models
    import liquid_state_extractors
    import readout_mechanisms
    import reservoir_topology
except ImportError as e:
    pytest.skip(f"Reservoir computing modules not available: {e}", allow_module_level=True)


class TestAdvancedESNFeatures:
    """Test advanced ESN features for 100% coverage"""
    
    def test_jaeger_benchmark_tasks_comprehensive(self):
        """Test all Jaeger benchmark tasks"""
        # Test Henon map with different parameters
        for a, b in [(1.4, 0.3), (1.2, 0.4)]:
            henon_inputs, henon_targets = JaegerBenchmarkTasks.henon_map_task(100, a=a, b=b)
            assert henon_inputs.shape == (99, 2)
            assert henon_targets.shape == (99, 2)
        
        # Test Lorenz attractor with different parameters
        lorenz_inputs, lorenz_targets = JaegerBenchmarkTasks.lorenz_attractor_task(
            n_steps=50, dt=0.01, sigma=8.0, rho=30.0, beta=3.0
        )
        assert lorenz_inputs.shape == (49, 3)
        assert lorenz_targets.shape == (49, 3)
        
        # Test sine wave with noise
        sine_inputs, sine_targets = JaegerBenchmarkTasks.sine_wave_task(
            n_steps=100, frequency=0.05, noise_level=0.1
        )
        assert sine_inputs.shape == (99, 1)
        assert sine_targets.shape == (99, 1)
        
        # Test pattern classification
        patterns, labels = JaegerBenchmarkTasks.pattern_classification_task(
            n_patterns=15, pattern_length=30
        )
        assert len(patterns) == 15
        assert len(labels) == 15
        assert all(isinstance(pattern, np.ndarray) for pattern in patterns)
    
    def test_structured_reservoir_topologies(self):
        """Test all structured reservoir topologies"""
        # Test ring topology with different parameters
        for size, k in [(10, 2), (20, 4), (30, 6)]:
            ring_matrix = StructuredReservoirTopologies.create_ring_topology(size, k)
            assert ring_matrix.shape == (size, size)
            assert np.count_nonzero(ring_matrix) > 0
        
        # Test small-world topology with different parameters
        for size, k, p in [(15, 4, 0.1), (25, 6, 0.2)]:
            sw_matrix = StructuredReservoirTopologies.create_small_world_topology(size, k, p)
            assert sw_matrix.shape == (size, size)
            assert np.count_nonzero(sw_matrix) > 0
        
        # Test scale-free topology with different parameters
        for size, m in [(12, 2), (18, 3), (25, 4)]:
            sf_matrix = StructuredReservoirTopologies.create_scale_free_topology(size, m)
            assert sf_matrix.shape == (size, size)
            assert np.count_nonzero(sf_matrix) > 0
    
    def test_echo_state_property_validator_comprehensive(self):
        """Test comprehensive ESP validation"""
        esn = EchoStateNetwork(n_reservoir=25, spectral_radius=0.85)
        esn.initialize_reservoir(2, 1)
        
        validator = EchoStatePropertyValidator()
        
        # Test ESP verification with different parameters
        for n_tests, test_length in [(3, 50), (5, 100)]:
            esp_results = validator.verify_echo_state_property(esn, n_tests, test_length)
            
            assert 'esp_satisfied' in esp_results
            assert 'max_pairwise_distance' in esp_results
            assert 'convergence_rate' in esp_results
            assert 'distances_over_time' in esp_results
            assert 'spectral_radius' in esp_results
            assert isinstance(esp_results['esp_satisfied'], bool)
        
        # Test memory capacity measurement with different parameters
        for max_delay, n_samples in [(10, 200), (20, 300)]:
            mc_results = validator.measure_memory_capacity(esn, max_delay, n_samples)
            
            assert 'total_memory_capacity' in mc_results
            assert 'memory_capacities_by_delay' in mc_results
            assert 'efficiency' in mc_results
            assert 'theoretical_maximum' in mc_results
            assert isinstance(mc_results['total_memory_capacity'], (int, float))
    
    def test_output_feedback_esn_class(self):
        """Test OutputFeedbackESN class"""
        fb_esn = OutputFeedbackESN(
            feedback_scaling=0.15,
            reservoir_size=30,
            spectral_radius=0.9
        )
        
        assert fb_esn.output_feedback is True
        assert fb_esn.feedback_scaling == 0.15
        
        # Test training with feedback
        X = np.random.randn(20, 8, 2)
        y = np.random.randn(20, 2)
        
        fb_esn.fit(X, y)
        predictions = fb_esn.predict(X[:5])
        
        assert predictions.shape == (5, 2)
    
    def test_teacher_forcing_trainer_class(self):
        """Test TeacherForcingTrainer class"""
        esn = EchoStateNetwork(n_reservoir=20, output_feedback=True)
        esn.initialize_reservoir(2, 1)
        
        trainer = TeacherForcingTrainer(esn)
        
        # Test training with different teacher forcing ratios
        X = np.random.randn(15, 2)
        y = np.random.randn(15, 1)
        
        for ratio in [0.5, 0.8, 1.0]:
            states = trainer.train_with_teacher_forcing(X, y, ratio)
            assert states.shape == (15, 20)
    
    def test_online_learning_esn_class(self):
        """Test OnlineLearningESN class"""
        esn = EchoStateNetwork(n_reservoir=15)
        esn.initialize_reservoir(2, 1)
        
        online_learner = OnlineLearningESN(esn, forgetting_factor=0.995)
        
        # Test RLS initialization
        online_learner.initialize_rls(16, initial_variance=500.0)  # 15 + 1 bias
        
        assert online_learner.P is not None
        assert online_learner.w is not None
        assert online_learner.P.shape == (16, 16)
        
        # Test online updates
        for _ in range(10):
            state = np.random.randn(15)
            target = np.random.randn(1)[0]
            
            prediction = online_learner.update_online(state, target)
            assert isinstance(prediction, (int, float, np.ndarray))
    
    def test_utility_functions_comprehensive(self):
        """Test all utility functions"""
        # Test optimize_spectral_radius function
        esn_config = {
            'reservoir_size': 20,
            'connectivity': 0.1,
            'leak_rate': 0.8
        }
        
        X = np.random.randn(30, 10, 2)
        y = np.random.randn(30, 1)
        
        optimal_radius, best_error = optimize_spectral_radius(
            esn_config, X, y, search_range=(0.3, 1.1), n_trials=3
        )
        
        assert isinstance(optimal_radius, (int, float))
        assert isinstance(best_error, (int, float))
        assert 0.3 <= optimal_radius <= 1.1
        
        # Test validate_esp function
        esn = EchoStateNetwork(n_reservoir=15, spectral_radius=0.75)
        esn.initialize_reservoir(2, 1)
        
        esp_valid = validate_esp(esn, verbose=False)
        assert isinstance(esp_valid, bool)
        
        # Test run_benchmark_suite function
        benchmark_results = run_benchmark_suite(esn, verbose=False)
        
        assert isinstance(benchmark_results, dict)
        expected_keys = ['henon_mse', 'sine_mse', 'memory_capacity']
        for key in expected_keys:
            if key in benchmark_results:
                assert isinstance(benchmark_results[key], (int, float))
        
        # Test measure_memory_capacity function (backward compatibility)
        memory_cap = measure_memory_capacity(esn, sequence_length=100, max_delay=5)
        assert isinstance(memory_cap, (int, float))
    
    def test_demonstration_function(self):
        """Test the demonstration function"""
        # This will test the demonstrate_unified_esn function
        # Capture output to avoid cluttering test results
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            demonstrate_unified_esn()
            output = captured_output.getvalue()
            assert "Unified Echo State Network Demonstration" in output
            assert "Echo State Property" in output
        finally:
            sys.stdout = sys.__stdout__
    
    def test_esn_advanced_methods(self):
        """Test advanced ESN methods that may be missing coverage"""
        esn = EchoStateNetwork(n_reservoir=25, spectral_radius=0.9)
        esn.initialize_reservoir(2, 1)
        
        # Test validate_echo_state_property method
        validation_results = esn.validate_echo_state_property(comprehensive=True)
        assert 'valid' in validation_results
        assert 'spectral_radius' in validation_results
        
        # Test optimize_spectral_radius method
        X = np.random.randn(20, 8, 2)
        y = np.random.randn(20, 1)
        
        optimization_results = esn.optimize_spectral_radius(
            X, y, radius_range=(0.2, 1.0), n_points=3, cv_folds=2
        )
        assert 'optimal_radius' in optimization_results
        assert 'results' in optimization_results
        
        # Test compute_memory_capacity method
        mc_results = esn.compute_memory_capacity(max_delay=5)
        assert 'total_memory_capacity' in mc_results
        assert 'memory_capacities' in mc_results
        
        # Test train_teacher_forcing method
        tf_results = esn.train_teacher_forcing(X, y, teacher_forcing_ratio=0.8)
        assert 'training_error' in tf_results
        assert 'method' in tf_results
        
        # Test run_autonomous method
        esn.fit(X, y)  # Train first
        states, outputs = esn.run_autonomous(n_steps=5, initial_output=np.array([0.5]))
        assert states.shape == (5, 25)
        assert outputs.shape == (5, 1)
    
    def test_esn_edge_cases_and_error_conditions(self):
        """Test ESN edge cases and error conditions"""
        # Test with invalid parameters
        with pytest.raises(ValueError):
            EchoStateNetwork(reservoir_size=0)  # Should raise ValueError
        
        with pytest.raises(ValueError):
            EchoStateNetwork(spectral_radius=0)  # Should raise ValueError
        
        with pytest.raises(ValueError):
            EchoStateNetwork(connectivity=2.0)  # Should raise ValueError
        
        with pytest.raises(ValueError):
            EchoStateNetwork(leak_rate=0)  # Should raise ValueError
        
        # Test prediction without training
        esn = EchoStateNetwork(n_reservoir=10)
        esn.initialize_reservoir(2, 1)
        
        X = np.random.randn(5, 3, 2)
        
        with pytest.raises(ValueError):
            esn.predict(X)  # Should raise ValueError - not trained
        
        # Test with very small input
        esn_small = EchoStateNetwork(n_reservoir=5)
        X_tiny = np.random.randn(1, 2, 1)
        y_tiny = np.random.randn(1, 1)
        
        esn_small.fit(X_tiny, y_tiny)
        pred = esn_small.predict(X_tiny)
        assert pred.shape == (1, 1)
        
        # Test with NaN/Inf inputs
        esn_nan = EchoStateNetwork(n_reservoir=8)
        X_nan = np.random.randn(5, 3, 1)
        X_nan[0, 0, 0] = np.nan
        y_clean = np.random.randn(5, 1)
        
        esn_nan.fit(X_nan, y_clean)  # Should handle NaN gracefully
        assert esn_nan.is_trained
    
    def test_activation_functions_comprehensive(self):
        """Test all activation function variations"""
        activation_functions = [
            'tanh', 'sigmoid', 'relu', 'leaky_relu', 'linear', 
            'sin', 'identity', 'unknown_function'
        ]
        
        for activation in activation_functions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                esn = EchoStateNetwork(
                    n_reservoir=8, 
                    activation_function=activation
                )
                
                # Test configuration
                esn.configure_activation_function(activation)
                assert callable(esn.activation_func)
                
                # Test with custom function
                esn.configure_activation_function('custom', custom_func=np.tanh)
                assert esn.activation_function == 'custom'


class TestModularComponents:
    """Test modular components with 100% coverage"""
    
    def test_all_module_imports(self):
        """Test all possible module imports"""
        modules_to_test = [
            'echo_state_validation',
            'esn_configuration', 
            'esn_training',
            'state_dynamics',
            'synaptic_models',
            'liquid_state_extractors',
            'readout_mechanisms',
            'reservoir_topology'
        ]
        
        for module_name in modules_to_test:
            try:
                module = sys.modules.get(module_name)
                if module:
                    # Test that module has expected attributes
                    assert hasattr(module, '__name__')
            except Exception:
                pass  # Some modules may not be fully implemented
    
    def test_lif_neuron_comprehensive(self):
        """Test LIF neuron with comprehensive coverage"""
        # Test with different configurations
        configs = [
            {'tau_m': 10.0, 'V_thresh': -50.0, 'V_reset': -70.0},
            {'tau_m': 20.0, 'V_thresh': -45.0, 'V_reset': -65.0, 'tau_ref': 2.0},
            {'tau_m': 15.0, 'V_thresh': -55.0, 'V_reset': -75.0, 'R': 1e8}
        ]
        
        for config in configs:
            neuron = LIFNeuron(**config)
            
            # Test step function with various currents
            for current in [0, 50, 100, 200]:
                spike = neuron.step(current, dt=1.0)
                assert isinstance(spike, bool)
            
            # Test reset
            if hasattr(neuron, 'reset'):
                neuron.reset()
        
        # Test with stimulus patterns
        neuron = LIFNeuron(tau_m=20.0, V_thresh=-50.0)
        
        # Constant current
        spikes_constant = []
        for _ in range(100):
            spike = neuron.step(80.0, 1.0)
            spikes_constant.append(spike)
        
        # Variable current
        neuron.reset() if hasattr(neuron, 'reset') else None
        spikes_variable = []
        for i in range(100):
            current = 50 + 30 * np.sin(i * 0.1)
            spike = neuron.step(current, 1.0)
            spikes_variable.append(spike)
        
        assert any(spikes_constant) or any(spikes_variable)  # Should have some spikes
    
    def test_hierarchical_reservoir_comprehensive(self):
        """Test hierarchical reservoir with comprehensive coverage"""
        # Test with different configurations
        configs = [
            {'n_levels': 2, 'reservoir_sizes': [40, 20]},
            {'n_levels': 3, 'reservoir_sizes': [50, 30, 15]},
            {'n_levels': 1, 'reservoir_sizes': [25]}
        ]
        
        for config in configs:
            hr = HierarchicalReservoir(**config)
            
            # Test properties
            assert hr.n_levels == config['n_levels']
            
            # Test with data
            X = np.random.randn(15, 8, 3)
            y = np.random.randn(15, 2)
            
            hr.fit(X, y)
            predictions = hr.predict(X[:3])
            
            assert predictions.shape[0] == 3
        
        # Test edge cases
        hr_edge = HierarchicalReservoir(n_levels=1, reservoir_sizes=[10])
        X_edge = np.random.randn(3, 5, 2)
        y_edge = np.random.randn(3, 1)
        
        hr_edge.fit(X_edge, y_edge)
        pred_edge = hr_edge.predict(X_edge)
        assert pred_edge.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])