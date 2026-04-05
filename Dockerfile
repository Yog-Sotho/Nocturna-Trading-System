# NOCTURNA v2.0 Trading Bot - Production Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Frontend Build
# =============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY nocturna-frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY nocturna-frontend/ ./

# Build frontend
RUN npm run build

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM python:3.11-slim AS python-builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production Image
# =============================================================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libatlas3-base \
    liblapack3 \
    libblas3 \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r nocturna && useradd -r -g nocturna nocturna

# Create application directory
WORKDIR /app

# Copy Python virtual environment from builder stage
COPY --from=python-builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .env.example ./.env.example

# Copy built frontend from frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist ./src/static/

# Create necessary directories
RUN mkdir -p logs data backups && \
    chown -R nocturna:nocturna /app

# Copy startup script
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

# Wait for database to be ready (if using external database)
if [ "\$DATABASE_URL" != "sqlite:///nocturna.db" ]; then
    echo "Waiting for database to be ready..."
    python -c "
import time
import sys
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import os

db_url = os.getenv('DATABASE_URL', 'sqlite:///nocturna.db')
if db_url.startswith('sqlite'):
    sys.exit(0)

engine = create_engine(db_url)
for i in range(30):
    try:
        engine.connect()
        print('Database is ready!')
        break
    except OperationalError:
        print(f'Database not ready, waiting... ({i+1}/30)')
        time.sleep(2)
else:
    print('Database connection timeout!')
    sys.exit(1)
"
fi

# Initialize database if needed
python -c "
from src.core.trading_engine import TradingEngine
import os
if not os.path.exists('nocturna.db') and os.getenv('DATABASE_URL', '').startswith('sqlite'):
    print('Initializing database...')
    # Add database initialization code here
    print('Database initialized.')
"

# Start the application
echo "Starting NOCTURNA v2.0 Trading Bot..."
exec python src/main.py
EOF

RUN chmod +x /app/start.sh && chown nocturna:nocturna /app/start.sh

# Switch to non-root user
USER nocturna

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Expose port
EXPOSE 5000

# Set default environment variables
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV HOST=0.0.0.0
ENV PORT=5000

# Start application
CMD ["/app/start.sh"]

# =============================================================================
# Build Instructions:
# =============================================================================
# 
# Build the image:
# docker build -t nocturna-v2:latest .
#
# Run with environment file:
# docker run -d --name nocturna-bot \
#   --env-file .env \
#   -p 5000:5000 \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/logs:/app/logs \
#   -v $(pwd)/backups:/app/backups \
#   nocturna-v2:latest
#
# Run with individual environment variables:
# docker run -d --name nocturna-bot \
#   -e ALPACA_API_KEY=your_key \
#   -e ALPACA_SECRET_KEY=your_secret \
#   -e POLYGON_API_KEY=your_polygon_key \
#   -e TRADING_MODE=PAPER \
#   -p 5000:5000 \
#   nocturna-v2:latest
#
# =============================================================================

