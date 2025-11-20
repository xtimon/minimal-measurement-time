# GitHub Actions CI/CD Pipeline

This repository uses GitHub Actions for continuous integration and deployment.

## Workflows

### CI Workflow (`.github/workflows/ci.yml`)

The CI workflow runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **Test**: Runs tests on Python 3.8, 3.9, 3.10, and 3.11
  - Installs dependencies
  - Runs pytest with coverage
  - Uploads coverage to Codecov

- **Lint**: Checks code quality
  - Runs `black` for code formatting
  - Runs `flake8` for linting
  - Runs `mypy` for type checking

- **Validate**: Runs validation and example scripts
  - Executes `validate_results.py`
  - Executes `example_simulation.py`

### Publish Workflow (`.github/workflows/publish.yml`)

The publish workflow runs when a release is published or manually triggered.

**Jobs:**
- **Build**: Builds the distribution packages
  - Creates source distribution and wheel
  - Validates packages with `twine check`

- **Publish to TestPyPI**: Publishes to Test PyPI (manual trigger only)
- **Publish to PyPI**: Publishes to PyPI (on release)

### Performance Workflow (`.github/workflows/performance.yml`)

The performance workflow runs benchmarks to track performance over time.

**Jobs:**
- **Benchmark**: Runs performance benchmarks
  - Executes `benchmarks/benchmark_simulator.py`
  - Uploads results as artifacts

## Setup

### Required Secrets

For the publish workflow to work, you need to set up the following:

1. **PyPI Publishing** (using Trusted Publishers):
   - Go to PyPI → Your Account → Publishing
   - Add GitHub as a trusted publisher
   - No tokens needed!

2. **TestPyPI Publishing** (optional):
   - Same as above, but on test.pypi.org

### Code Coverage

The CI workflow automatically uploads coverage reports to Codecov. To view coverage:

1. Sign up at https://codecov.io/
2. Connect your GitHub repository
3. Coverage reports will be available after each CI run

## Local Development

To run the same checks locally before pushing:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v --cov=measurement_time_simulator

# Run linting
black measurement_time_simulator/
flake8 measurement_time_simulator/
mypy measurement_time_simulator/

# Run benchmarks
python benchmarks/benchmark_simulator.py

# Run validation
python validate_results.py
```

