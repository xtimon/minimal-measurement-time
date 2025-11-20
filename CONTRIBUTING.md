# Contributing to Measurement Time Simulator

Thank you for your interest in contributing to the Measurement Time Simulator! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/measurement-time-simulator.git
   cd measurement-time-simulator
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,gpu]"
   ```

4. **Run Tests**
   ```bash
   pytest -v
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation changes
- `perf/` for performance improvements

### 2. Make Changes

- Write clear, concise code
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed

### 3. Run Tests and Linting

```bash
# Run tests
pytest -v --cov=measurement_time_simulator

# Format code
black measurement_time_simulator/

# Lint code
flake8 measurement_time_simulator/

# Type check
mypy measurement_time_simulator/
```

### 4. Commit Changes

Write clear commit messages:

```bash
git commit -m "feat: add new optimization for gamma calculation"
git commit -m "fix: correct decoherence factor calculation"
git commit -m "docs: update README with GPU instructions"
```

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `perf:` for performance improvements
- `test:` for test changes
- `refactor:` for code refactoring

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use `black` for code formatting (line length: 120)
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def calculate_something(param1: float, param2: int) -> float:
    """
    Calculate something based on parameters.
    
    Parameters:
    - param1: Description of param1
    - param2: Description of param2
    
    Returns:
    - Calculated result
    
    Raises:
    - ValueError: If param1 is negative
    
    Example:
    ```python
    result = calculate_something(1.0, 5)
    ```
    """
    pass
```

## Testing

### Writing Tests

- Write tests for all new features
- Aim for >90% code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

Example test:

```python
def test_new_feature():
    """Test that new feature works correctly"""
    sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=False, suppress_logging=True)
    result = sim.new_feature(param=1.0)
    assert result > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_simulator.py

# Run with coverage
pytest --cov=measurement_time_simulator --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## Performance Optimization

When adding performance optimizations:

1. **Benchmark First**: Measure before optimizing
   ```bash
   python benchmarks/benchmark_simulator.py
   ```

2. **Document Changes**: Update `OPTIMIZATIONS.md`

3. **Add Benchmarks**: Add benchmark tests for new optimizations

4. **Test Both Paths**: Ensure CPU and GPU paths work correctly

## Documentation

### Updating Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update OPTIMIZATIONS.md for performance changes

### Building Documentation (if applicable)

```bash
cd docs
make html
```

## Continuous Integration

All pull requests must pass CI checks:

- ✅ Tests pass on Python 3.8, 3.9, 3.10, 3.11
- ✅ Code is formatted with black
- ✅ Linting passes (flake8)
- ✅ Type checking passes (mypy, optional)
- ✅ Validation scripts run successfully

## Versioning

We use [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for new features (backwards-compatible)
- PATCH version for bug fixes

## Pull Request Process

1. **Describe Your Changes**
   - What does this PR do?
   - Why is this change needed?
   - What tests were added?

2. **Link Issues**
   - Reference related issues: `Fixes #123`

3. **Update Documentation**
   - Update README if needed
   - Update CHANGELOG

4. **Request Review**
   - Tag relevant reviewers
   - Respond to feedback

5. **Merge**
   - Squash and merge when approved
   - Delete branch after merge

## Code Review Guidelines

When reviewing pull requests:

- ✅ Code is clear and well-documented
- ✅ Tests are comprehensive
- ✅ No performance regressions
- ✅ Documentation is updated
- ✅ Follows project style

## Reporting Bugs

### Bug Report Template

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - Python version
   - Package version
   - OS
   - GPU info (if applicable)

Example:

```markdown
## Bug: Incorrect result for large N

**Steps to Reproduce:**
```python
sim = GPUInformationMeasurementSimulator(temperature=300.0)
result = sim.simulate_single_system(N=1e10, ...)
```

**Expected:** Result should be finite
**Actual:** Result is NaN

**Environment:**
- Python 3.10
- measurement-time-simulator 0.1.0
- Ubuntu 22.04
```

## Feature Requests

We welcome feature requests! Please include:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Impact**: Who benefits from this feature?

## Questions?

- 💬 Open a GitHub Discussion
- 📧 Contact maintainers
- 📖 Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Thank you for contributing to make this project better! 🎉
