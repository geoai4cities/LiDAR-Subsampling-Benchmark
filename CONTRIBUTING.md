# Contributing to LiDAR Subsampling Benchmark

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Describe the issue clearly
- Include steps to reproduce
- Provide system information

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8
- Use Black for formatting: `black src/`
- Use type hints where possible
- Add docstrings to functions and classes

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Maintain >80% code coverage

### Documentation
- Update README.md if needed
- Add docstrings to new functions
- Update API documentation

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/LiDAR-Subsampling-Benchmark.git
cd LiDAR-Subsampling-Benchmark

# Create development environment
conda env create -f environment.yml
conda activate lidar-benchmark

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

## Questions?

Feel free to open an issue or contact the maintainers.
