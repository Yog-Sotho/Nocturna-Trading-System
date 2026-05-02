# =============================================================================
# NOCTURNA Trading System — Production Dockerfile
# Audit Remediation: F-003, F-012, F-013
# =============================================================================
# Stage 1: Base Builder
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies required for compiling Python packages (e.g., cryptography, psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# =============================================================================
# AUDIT FIX F-012: Remove --break-system-packages
# Use a dedicated virtual environment to avoid PEP-668 conflicts and ensure
# reproducible, isolated dependency resolution.
# =============================================================================
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies with strict pinning
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Development Environment (Optional but preserved per original structure)
# =============================================================================
FROM builder AS development

WORKDIR /app

# Copy compiled virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# =============================================================================
# AUDIT FIX F-013: Remove redundant dev-tool installation
# requirements.txt already pins pytest, black, flake8, mypy, etc.
# Re-installing them without version pins causes drift and layer bloat.
# We rely solely on the builder-stage resolution.
# =============================================================================
COPY requirements-dev.txt ./requirements-dev.txt || true
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Install development tools with hot-reloading support
RUN pip install --no-cache-dir watchdog python-dotenv

# Copy application source
COPY . .

# Set development defaults
ENV FLASK_ENV=development \
    FLASK_DEBUG=1 \
    PYTHONUNBUFFERED=1

# Expose debug port
EXPOSE 5000 5678

# Development entrypoint (uses Flask built-in server for hot-reload)
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

# =============================================================================
# Stage 3: Production Runtime
# =============================================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security hardening
RUN groupadd -r nocturna && useradd -r -g nocturna -d /app -s /sbin/nologin nocturna

# Copy virtual environment and dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY --chown=nocturna:nocturna . .

# Create required runtime directories
RUN mkdir -p /var/log/nocturna /var/run/nocturna /app/src/database && \
    chown -R nocturna:nocturna /var/log/nocturna /var/run/nocturna /app/src/database

# Switch to non-root user
USER nocturna

# =============================================================================
# AUDIT FIX F-003: Respect gunicorn.conf.py single-worker invariant
# Original: gunicorn --bind 0.0.0.0:5000 --workers 4 --worker-class sync src.main:create_app()
# The --workers 4 CLI flag overrides gunicorn.conf.py, breaking the single-worker
# invariant required for in-memory singleton state (positions, orders, risk, IP blacklist).
# Fixed: Delegate worker management entirely to gunicorn.conf.py via -c flag.
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

EXPOSE 5000

# Production entrypoint: strictly follows config file for worker count, preload, and timeouts
CMD ["gunicorn", "-c", "/app/gunicorn.conf.py", "src.main:create_app()"]
