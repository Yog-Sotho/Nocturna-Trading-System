# =============================================================================
# NOCTURNA Trading System - Production Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# Base Stage - Python Environment
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Set environment variables for security
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ENV=production

# Production settings
ENV FLASK_ENV=production \
    FLASK_DEBUG=false \
    SECURE_COOKIES=true \
    SQLALCHEMY_ECHO=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 nocturna && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home nocturna

WORKDIR /app

# -----------------------------------------------------------------------------
# Dependencies Stage - Install Python packages
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir --break-system-packages && \
    pip check

# -----------------------------------------------------------------------------
# Production Stage - Final application image
# -----------------------------------------------------------------------------
FROM base AS production

# Copy installed dependencies
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=1000:1000 . /app/

# Create directories with correct permissions
RUN mkdir -p /app/logs /app/database /app/static && \
    chown -R 1000:1000 /app

WORKDIR /app

# Switch to non-root user
USER nocturna

# Expose application port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "sync", "--worker-tmp-dir", "/dev/shm", "--timeout", "120", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "src.main:create_app()"]

# -----------------------------------------------------------------------------
# Development Stage
# -----------------------------------------------------------------------------
FROM base AS development

COPY . /app/
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --break-system-packages && \
    pip install pytest pytest-asyncio pytest-mock black flake8 mypy

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "src.main:app", "run", "--host", "0.0.0.0", "--port", "5000", "--debug"]

# -----------------------------------------------------------------------------
# Testing Stage
# -----------------------------------------------------------------------------
FROM base AS testing

COPY . /app/
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --break-system-packages && \
    pip install pytest pytest-asyncio pytest-mock pytest-cov coverage

CMD ["pytest", "-v", "--cov=src", "--cov-report=term-missing", "tests/"]
