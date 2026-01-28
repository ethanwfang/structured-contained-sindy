.PHONY: install install-dev test test-all lint format docs clean help

# Default target
help:
	@echo "Structure-Constrained SINDy - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package in development mode"
	@echo "  make install-dev    Install with all development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run unit tests"
	@echo "  make test-all       Run all tests (unit + integration)"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (ruff, black --check, mypy)"
	@echo "  make format         Auto-format code with black and ruff"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and caches"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

# Testing
test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

test-cov:
	pytest tests/unit/ -v --cov=sc_sindy --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# Code quality
lint:
	ruff check src/
	black --check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/

# Documentation
docs:
	cd docs && make html
	@echo "Documentation: docs/_build/html/index.html"

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
