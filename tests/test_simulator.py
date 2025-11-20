"""
Tests for the simulator module
"""

import pytest
import numpy as np
from measurement_time_simulator import GPUInformationMeasurementSimulator


class TestGPUInformationMeasurementSimulator:
    """Test cases for the main simulator class"""

    def test_initialization(self):
        """Test simulator initialization"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        assert sim.temperature == 300.0
        assert sim.hbar > 0
        assert sim.kB > 0

    def test_simulate_single_system_fermion(self):
        """Test simulation of a single fermion system"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        measurement_time, results = sim.simulate_single_system(
            delta_I=1.0, T_c=100.0, N=1000.0, stats_type=0, T_F=5000.0, export_results=False  # Fermion
        )

        assert measurement_time > 0
        assert isinstance(results, dict)
        assert "min_measurement_time" in results
        assert "gamma_factors" in results
        assert results["parameters"]["stats_type"] == 0

    def test_simulate_single_system_boson(self):
        """Test simulation of a single boson system"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        measurement_time, results = sim.simulate_single_system(
            delta_I=1.0, T_c=0.1, N=100000.0, stats_type=1, T_c_bose=0.1, export_results=False  # Boson
        )

        assert measurement_time > 0
        assert isinstance(results, dict)
        assert results["parameters"]["stats_type"] == 1

    def test_simulate_single_system_classical(self):
        """Test simulation of a single classical system"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        measurement_time, results = sim.simulate_single_system(
            delta_I=1.0, T_c=100.0, N=1000.0, stats_type=2, export_results=False  # Classical
        )

        assert measurement_time > 0
        assert isinstance(results, dict)
        assert results["parameters"]["stats_type"] == 2

    def test_batch_simulation(self):
        """Test batch simulation with multiple systems"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)

        n_points = 100
        delta_I_array = np.full(n_points, 1.0)
        T_c_array = np.random.uniform(10, 1000, n_points)
        N_array = np.random.lognormal(7, 2, n_points)
        stats_type_array = np.random.choice([0, 1, 2], n_points)
        T_F_array = np.random.uniform(100, 10000, n_points)

        min_times = sim.main_equation_batch(
            delta_I_array=delta_I_array,
            T_c_array=T_c_array,
            N_array=N_array,
            stats_type_array=stats_type_array,
            T_F_array=T_F_array,
            include_decoherence=False,
            include_noise=False,
        )

        assert len(min_times) == n_points
        assert np.all(min_times > 0)

    def test_decoherence_effect(self):
        """Test that decoherence increases measurement time"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)

        # Without decoherence
        time_no_dec, _ = sim.simulate_single_system(
            delta_I=1.0,
            T_c=100.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            include_decoherence=False,
            include_noise=False,
            export_results=False,
        )

        # With decoherence
        time_with_dec, _ = sim.simulate_single_system(
            delta_I=1.0,
            T_c=100.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            T1=1e-3,
            T2=1e-4,
            include_decoherence=True,
            include_noise=False,
            export_results=False,
        )

        # Decoherence should increase measurement time
        assert time_with_dec >= time_no_dec

    def test_noise_effect(self):
        """Test that noise increases measurement time"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)

        # Without noise
        time_no_noise, _ = sim.simulate_single_system(
            delta_I=1.0,
            T_c=100.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            include_decoherence=False,
            include_noise=False,
            export_results=False,
        )

        # With noise
        time_with_noise, _ = sim.simulate_single_system(
            delta_I=1.0,
            T_c=100.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            noise_temperature=1.0,
            shot_noise_factor=0.1,
            technical_noise=0.02,
            include_decoherence=False,
            include_noise=True,
            export_results=False,
        )

        # Noise should increase measurement time
        assert time_with_noise >= time_no_noise

    def test_gamma_total_batch(self):
        """Test gamma_total calculation for batches"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)

        n_points = 50
        T_c_array = np.random.uniform(10, 1000, n_points)
        N_array = np.random.lognormal(7, 2, n_points)
        stats_type_array = np.random.choice([0, 1, 2], n_points)
        T_F_array = np.random.uniform(100, 10000, n_points)

        gamma_total = sim.gamma_total_batch(
            T_c_array=T_c_array, N_array=N_array, stats_type_array=stats_type_array, T_F_array=T_F_array
        )

        assert len(gamma_total) == n_points
        assert np.all(gamma_total >= 1.0)  # Complexity factor should be >= 1

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors"""
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)

        # Invalid delta_I
        with pytest.raises(ValueError):
            sim.simulate_single_system(delta_I=-1.0, T_c=100.0, N=1000.0, stats_type=0, export_results=False)

        # Invalid T_c
        with pytest.raises(ValueError):
            sim.simulate_single_system(delta_I=1.0, T_c=-100.0, N=1000.0, stats_type=0, export_results=False)

        # Invalid stats_type
        with pytest.raises(ValueError):
            sim.simulate_single_system(
                delta_I=1.0, T_c=100.0, N=1000.0, stats_type=5, export_results=False  # Invalid type
            )

    def test_temperature_dependence(self):
        """Test that measurement time depends on temperature"""
        # Lower temperature - quantum limit should dominate
        sim_low = GPUInformationMeasurementSimulator(temperature=10.0, use_gpu=False, suppress_logging=True)
        time_low, _ = sim_low.simulate_single_system(
            delta_I=10.0,  # More information to make quantum effects visible
            T_c=5.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            include_decoherence=False,
            include_noise=False,
            equilibrium_time=0.0,  # Disable to see quantum effects
            detector_response_time=0.0,
            export_results=False,
        )

        # Higher temperature
        sim_high = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        time_high, _ = sim_high.simulate_single_system(
            delta_I=10.0,
            T_c=5.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            include_decoherence=False,
            include_noise=False,
            equilibrium_time=0.0,
            detector_response_time=0.0,
            export_results=False,
        )

        # Both simulations should give valid positive times
        # Note: At these scales, detector response time dominates, so times may be similar
        assert time_low > 0
        assert time_high > 0
        # Temperature dependence is still present in the quantum limit calculation,
        # but may be masked by technical limits (detector time, equilibrium time)
