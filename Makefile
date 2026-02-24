.PHONY: install dev test lint format type-check check clean

# Install all dependencies (including dev tools)
install:
	uv sync --all-extras

# Run the FastAPI dev server with hot reload
dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run the test suite
test:
	uv run pytest -v

# Lint the code (check only, no fixes)
lint:
	uv run ruff check .

# Auto-format and fix lint issues
format:
	uv run ruff format .
	uv run ruff check --fix .

# Run type checking
type-check:
	uv run mypy app/

# Run all checks (what CI will run)
check: lint type-check test

# Remove cached files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
