# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment vars to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required by MLflow, sklearn, transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application code
COPY src/serving .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Expose Flask port
EXPOSE 5000

# Set default environment variables (override in docker run or compose)
ENV MLFLOW_ALLOWED_HOSTS="*" \
    MLFLOW_TRACKING_URI="http://localhost:5050"

# Start the Flask server
CMD ["python", "app.py"]
