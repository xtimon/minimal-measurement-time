"""
Tests for the exporter module
"""

import pytest
import os
import tempfile
from measurement_time_simulator import ResultExporter


class TestResultExporter:
    """Test cases for the ResultExporter class"""
    
    def test_export_to_text_detailed(self):
        """Test exporting results to text file with detailed format"""
        results = {
            'min_measurement_time': 1.23e-9,
            'min_measurement_time_ideal': 1.0e-9,
            'parameters': {
                'temperature': 300.0,
                'delta_I': 1.0,
                'T_c': 100.0,
                'N': 1000,
                'stats_type': 0,
                'T_F': 5000.0
            },
            'gamma_factors': {
                'total': 5.5,
                'base': 2.0,
                'statistics': 1.5,
                'correlations': 1.0,
                'quasiparticles': 1.0
            },
            'decoherence': {
                'T1': 1e-3,
                'T2': 1e-4,
                'factor': 1.5,
                'enabled': True
            },
            'noise': {
                'temperature': 1.0,
                'shot_noise_factor': 0.1,
                'technical_noise': 0.02,
                'factor': 2.0,
                'enabled': True
            },
            'limits': {
                'fundamental': 2.5e-12,
                'technical': 1e-9,
                'landauer': 1.3e-14,
                'quantum': 1e-14
            }
        }
        
        exporter = ResultExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily change results directory
            original_dir = ResultExporter.RESULTS_DIR
            ResultExporter.RESULTS_DIR = tmpdir
            
            filename = exporter.export_to_text(results, format_type="detailed")
            
            # Restore original directory
            ResultExporter.RESULTS_DIR = original_dir
            
            assert os.path.exists(filename)
            with open(filename, 'r') as f:
                content = f.read()
                assert 'РЕЗУЛЬТАТЫ СИМУЛЯЦИИ' in content
                assert '1.230e-09' in content or '1.23e-09' in content
    
    def test_export_to_text_summary(self):
        """Test exporting results to text file with summary format"""
        results = {
            'min_measurement_time': 1.23e-9,
            'parameters': {
                'temperature': 300.0,
                'stats_type': 0,
                'N': 1000
            },
            'gamma_factors': {
                'total': 5.5
            }
        }
        
        exporter = ResultExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = ResultExporter.RESULTS_DIR
            ResultExporter.RESULTS_DIR = tmpdir
            
            filename = exporter.export_to_text(results, format_type="summary")
            
            ResultExporter.RESULTS_DIR = original_dir
            
            assert os.path.exists(filename)
            with open(filename, 'r') as f:
                content = f.read()
                assert 'КРАТКИЙ ОТЧЕТ' in content
    
    def test_export_comparison(self):
        """Test exporting comparison of multiple systems"""
        systems_data = [
            {
                'name': 'System 1',
                'measurement_time': 1e-9,
                'complexity': 5.0,
                'type': 'Fermion'
            },
            {
                'name': 'System 2',
                'measurement_time': 2e-9,
                'complexity': 6.0,
                'type': 'Boson'
            }
        ]
        
        exporter = ResultExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = ResultExporter.RESULTS_DIR
            ResultExporter.RESULTS_DIR = tmpdir
            
            filename = exporter.export_comparison(systems_data)
            
            ResultExporter.RESULTS_DIR = original_dir
            
            assert os.path.exists(filename)
            with open(filename, 'r') as f:
                content = f.read()
                assert 'СРАВНЕНИЕ ФИЗИЧЕСКИХ СИСТЕМ' in content
                assert 'System 1' in content
                assert 'System 2' in content
