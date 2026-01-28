# Contributing to Structure-Constrained SINDy

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/structure-constrained-sindy.git
   cd structure-constrained-sindy
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[all]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   make test
   ```
4. Run linting:
   ```bash
   make lint
   ```
5. Commit your changes with a descriptive message
6. Push and create a pull request

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [Ruff](https://docs.astral.sh/ruff/) for linting
- Maximum line length is 100 characters
- Use type hints for function signatures
- Write NumPy-style docstrings

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a PR
- Aim for high test coverage

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure CI passes
- Request review from maintainers

## Reporting Issues

- Use the issue templates provided
- Include minimal reproducible examples
- Provide system information (Python version, OS, etc.)

## Questions?

Feel free to open an issue for questions or discussions.
