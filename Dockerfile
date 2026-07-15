FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# git is required while hip-cargo is a git dependency; drop this layer when
# the dependency is re-pinned to a tagged release.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package with full dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["mbeams", "--help"]
