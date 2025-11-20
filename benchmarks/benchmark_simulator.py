#!/usr/bin/env python3
"""
Performance benchmarks for the measurement time simulator

This script benchmarks various components of the simulator to identify
performance bottlenecks and track performance improvements over time.
"""

import time
import numpy as np
from datetime import datetime
from measurement_time_simulator import GPUInformationMeasurementSimulator
import os


class BenchmarkRunner:
    """Runs performance benchmarks for the simulator"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.results = []
        
    def benchmark(self, name, func, iterations=10):
        """
        Run a benchmark and record results
        
        Parameters:
        - name: Name of the benchmark
        - func: Function to benchmark
        - iterations: Number of iterations to run
        """
        print(f"\n🔬 Running benchmark: {name}")
        print(f"   Iterations: {iterations}")
        
        times = []
        for i in range(iterations):
            start = time.time()
            result = func()
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            
            if i == 0:
                # Print first iteration for validation
                print(f"   First iteration: {elapsed:.4f}s")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"   Mean time: {mean_time:.4f}s (±{std_time:.4f}s)")
        print(f"   Min/Max: {min_time:.4f}s / {max_time:.4f}s")
        
        self.results.append({
            'name': name,
            'mean': mean_time,
            'std': std_time,
            'min': min_time,
            'max': max_time,
            'iterations': iterations,
            'use_gpu': self.use_gpu
        })
        
        return mean_time
    
    def save_results(self, filename=None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "benchmarks/results"
            os.makedirs(results_dir, exist_ok=True)
            filename = f"{results_dir}/benchmark_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PERFORMANCE BENCHMARK RESULTS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU Enabled: {self.use_gpu}\n")
            f.write("=" * 70 + "\n\n")
            
            for result in self.results:
                f.write(f"Benchmark: {result['name']}\n")
                f.write(f"  Mean time: {result['mean']:.4f}s (±{result['std']:.4f}s)\n")
                f.write(f"  Min/Max: {result['min']:.4f}s / {result['max']:.4f}s\n")
                f.write(f"  Iterations: {result['iterations']}\n")
                f.write("\n")
        
        print(f"\n📊 Results saved to: {filename}")
        return filename


def run_benchmarks():
    """Run all benchmarks"""
    print("=" * 70)
    print("MEASUREMENT TIME SIMULATOR - PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    # Try with CPU first
    runner = BenchmarkRunner(use_gpu=False)
    
    # Benchmark 1: Single system simulation (small)
    def bench_single_small():
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        measurement_time, _ = sim.simulate_single_system(
            delta_I=1.0,
            T_c=100.0,
            N=1000.0,
            stats_type=0,
            T_F=5000.0,
            include_decoherence=True,
            include_noise=True,
            export_results=False
        )
        return measurement_time
    
    runner.benchmark("Single System (Small, N=1000)", bench_single_small, iterations=10)
    
    # Benchmark 2: Single system simulation (large)
    def bench_single_large():
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        measurement_time, _ = sim.simulate_single_system(
            delta_I=8.0,
            T_c=100.0,
            N=100000.0,
            stats_type=0,
            T_F=10000.0,
            U=0.5,
            W=1.0,
            include_decoherence=True,
            include_noise=True,
            export_results=False
        )
        return measurement_time
    
    runner.benchmark("Single System (Large, N=100000)", bench_single_large, iterations=10)
    
    # Benchmark 3: Batch simulation (small batch)
    def bench_batch_small():
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
            include_decoherence=True,
            include_noise=True
        )
        return min_times
    
    runner.benchmark("Batch Simulation (100 systems)", bench_batch_small, iterations=5)
    
    # Benchmark 4: Batch simulation (medium batch)
    def bench_batch_medium():
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        n_points = 1000
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
            include_decoherence=True,
            include_noise=True
        )
        return min_times
    
    runner.benchmark("Batch Simulation (1000 systems)", bench_batch_medium, iterations=3)
    
    # Benchmark 5: Batch simulation (large batch)
    def bench_batch_large():
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        n_points = 10000
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
            include_decoherence=True,
            include_noise=True
        )
        return min_times
    
    runner.benchmark("Batch Simulation (10000 systems)", bench_batch_large, iterations=2)
    
    # Benchmark 6: Gamma total calculation
    def bench_gamma_total():
        sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
        n_points = 10000
        T_c_array = np.random.uniform(10, 1000, n_points)
        N_array = np.random.lognormal(7, 2, n_points)
        stats_type_array = np.random.choice([0, 1, 2], n_points)
        T_F_array = np.random.uniform(100, 10000, n_points)
        
        gamma_total = sim.gamma_total_batch(
            T_c_array=T_c_array,
            N_array=N_array,
            stats_type_array=stats_type_array,
            T_F_array=T_F_array
        )
        return gamma_total
    
    runner.benchmark("Gamma Total Calculation (10000 systems)", bench_gamma_total, iterations=5)
    
    # Save results
    runner.save_results()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    for result in runner.results:
        throughput = result['iterations'] / (result['mean'] * result['iterations'])
        print(f"{result['name']:<50} {result['mean']:>8.4f}s")
    
    print("\n💡 Performance Notes:")
    print("   - For batch operations with >10000 systems, GPU acceleration is recommended")
    print("   - Single system simulations are fast enough on CPU")
    print("   - Results may vary based on system specifications")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
