# Multi-stage production container for Agent Kernel
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim AS runner

# Create a non-root user and initialize app directory
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appgroup /app

WORKDIR /app

# Copy installed dependencies, ensuring correct ownership
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy source files atomically with non-root ownership
COPY --chown=appuser:appgroup . .

# Ensure data directory and application workspace are writable by appuser
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

# Drop privileges
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "gunicorn api.app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}"]
