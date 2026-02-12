.PHONY: help install dev test lint format clean build

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the project with uv
	uv sync

dev:  ## Install the project with dev dependencies
	uv sync --all-extras

test:  ## Run tests
	uv run pytest

test-coverage:  ## Run tests with coverage
	uv run pytest --cov=ship_model_lib --cov-report=html --cov-report=term

lint:  ## Run ruff linter
	uv run ruff check .

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check --fix .

format:  ## Format code with ruff
	uv run ruff format .

format-check:  ## Check code formatting
	uv run ruff format --check .

check:  ## Run all checks (lint + format)
	uv run ruff check --fix .
	uv run ruff format .

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv build

jupyter:  ## Start Jupyter Lab
	uv run jupyter lab

update:  ## Update all dependencies
	uv lock --upgrade

lock:  ## Update lock file
	uv lock
