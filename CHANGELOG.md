# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CI/CD Pipeline**: Complete GitHub Actions workflows for testing, linting, and deployment
  - Automated testing on Python 3.8, 3.9, 3.10, 3.11
  - Code quality checks (black, flake8, mypy)
  - Automatic PyPI publishing on release
  - Performance benchmarking workflow
  
- **Performance Optimizations**: Major performance improvements
  - JIT compilation with Numba for 2-5x speedup on critical functions
  - Caching and memoization for repeated calculations
  - Optimized array operations with better memory layout
  - Chunked processing for large datasets
  - Automatic CPU/GPU mode selection
  
- **Testing Infrastructure**: Comprehensive test suite
  - Unit tests for simulator, exporter, and optimizations
  - Integration tests for end-to-end workflows
  - Test coverage tracking with pytest-cov
  - Configuration files: pytest.ini, .flake8, pyproject.toml
  
- **Performance Benchmarks**: Benchmarking suite
  - Automated benchmarks for different operation sizes
  - CPU vs GPU performance comparison
  - Results tracking over time
  - Located in `benchmarks/` directory
  
- **Documentation**: Enhanced documentation
  - OPTIMIZATIONS.md: Complete guide to performance features
  - CONTRIBUTING.md: Contribution guidelines
  - .github/README.md: CI/CD documentation
  - Updated main README with performance info
  
- **Optimization Module**: New `optimizations.py` module
  - `PhysicsCache`: Intelligent caching for physics calculations
  - `OptimizedCalculator`: Cached calculator with automatic optimization
  - JIT-compiled functions for critical operations
  - `batch_compute_with_chunking`: Memory-efficient batch processing
  - `optimize_array_operations`: Array optimization utilities

### Changed
- Updated `__init__.py` to export optimization utilities
- README now includes performance benchmarks and CI/CD information
- setup.py includes development dependencies

### Fixed
- None in this release

## [0.1.0] - 2024-11-19

### Added
- Initial release
- Basic simulator implementation
- Support for fermions, bosons, and classical particles
- Decoherence modeling (T1, T2)
- Noise modeling (thermal, shot, technical, 1/f, quantum, environment)
- GPU acceleration with CuPy
- Result export functionality
- Validation scripts
- Example simulations

[Unreleased]: https://github.com/xtimon/minimal-measurement-time/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xtimon/minimal-measurement-time/releases/tag/v0.1.0
