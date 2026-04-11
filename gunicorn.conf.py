# =============================================================================
# NOCTURNA Trading System - Gunicorn Configuration
# =============================================================================

import multiprocessing
import os

# =============================================================================
# SERVER SOCKET
# =============================================================================

# Bind to the specified address and port
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:5000')

# Use a file descriptor to inherit the socket from systemd
# bind = 'fd://'

# =============================================================================
# WORKER PROCESSES
# =============================================================================

# Number of worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))

# Number of threads per worker
threads = int(os.environ.get('GUNICORN_THREADS', 4))

# Worker class
worker_class = 'sync'  # Use 'gevent' or 'eventlet' for async

# Worker timeout in seconds
timeout = 120

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Keep-alive timeout
keepalive = 5

# Maximum number of requests a worker will handle before respawning
max_requests = 1000
max_requests_jitter = 50

# =============================================================================
# PROCESS NAMING
# =============================================================================

# Format string for worker process names
proc_name = 'nocturna-trading'

# =============================================================================
# SERVER MECHANICS
# =============================================================================

# Daemonize (run in background) - DON'T USE WITH DOCKER
daemon = False

# PID file for daemonized master
pidfile = '/var/run/nocturna/gunicorn.pid'

# User/Group for worker processes
# user = 'nocturna'
# group = 'nocturna'

# Directory to chroot to before spawning
# chroot = '/var/www/nocturna'

# Working directory
working_directory = '/app'

# =============================================================================
# LOGGING
# =============================================================================

# Access log file (- for stdout)
accesslog = '-'

# Error log file (- for stderr)
errorlog = '-'

# Log level
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')

# Custom access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Log format for error logs
log_config_format = '%(asctime)s [%(process)d] [%(levelname)s] %(message)s'

# =============================================================================
# SECURITY
# =============================================================================

# Preload application code before forking workers
# This allows sharing application memory between workers
preload_app = True

# Disable redirecting stdout/stderr to files
capture_output = False

# Enable Python stack traces
traceback = True

# =============================================================================
# SERVER HOOKS
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting NOCTURNA Trading System...")
    server.log.info(f"Workers: {server.app.cfg.workers}")
    server.log.info(f"Threads: {server.app.cfg.threads}")


def on_reload(server):
    """Called to recycle workers during a configuration reload."""
    server.log.info("Reloading NOCTURNA Trading System...")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("NOCTURNA Trading System is ready to accept connections")
    server.log.info(f"Listening on: {server.app.cfg.bind}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("Shutting down NOCTURNA Trading System...")


def child_exit(server, worker):
    """Called just after a worker has been exited, in the master process."""
    server.log.info(f"Worker {worker.pid} exited")


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received SIGINT/SIGQUIT")


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    worker.log.warning(f"Worker {worker.pid} received SIGABRT")


# =============================================================================
# SERVER MECHANICS
# =============================================================================

# Use temporary directory for worker temp files
worker_tmp_dir = '/dev/shm'

# =============================================================================
# SSL (if needed)
# =============================================================================

# Keyfile = '/path/to/keyfile'
# Certfile = '/path/to/certfile'
# SSL Version and Ciphers (optional)
# ssl_version = 'TLSv1_2'
# ciphers = 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256'
