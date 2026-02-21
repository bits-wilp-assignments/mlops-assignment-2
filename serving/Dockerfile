# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY serving/requirements.txt /app/serving/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/serving/requirements.txt

# Copy application code
COPY common/ /app/common/
COPY serving/ /app/serving/

# Note: Models are NOT copied into the image
# Models should be loaded from MLflow registry at runtime
# For local development, mount models directory as a volume if needed

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Models are loaded from MLflow registry by default
# Set MODEL_PATH only for local development with volume mounts

# Run the application
CMD ["uvicorn", "serving.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
