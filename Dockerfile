# Start from official Python 3.11 slim image
# "slim" = smaller image, no extra tools we don't need
FROM python:3.11-slim

# Set working directory inside the container
# All commands after this run from /app
WORKDIR /app

# Install uv (our package manager)
RUN pip install uv

# Copy dependency files first
# Why first? Docker caches layers. If we copy code first,
# every code change would reinstall all packages. Slow.
# This way packages are only reinstalled when pyproject.toml changes.
COPY pyproject.toml uv.lock ./

# Install only production dependencies (no dev tools like pytest/ruff)
RUN uv sync --no-dev

# Copy the application code
COPY app/ ./app/

# Expose the port the app runs on
EXPOSE 9000

# Command to start the server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
