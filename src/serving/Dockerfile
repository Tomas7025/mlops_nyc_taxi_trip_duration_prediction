# Use a lightweight Python base image
FROM python:3.13-slim

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

COPY src/training/requirements.txt .
# ---- Install dependencies (same as CI) ----
# RUN uv sync --frozen
RUN pip install --upgrade pip && \
   pip install -r requirements.txt

# Copy application code
COPY src/serving .

# ---- Copy the transformers.py file and __init__.py ----
COPY src/transformers.py .


# # Expose Flask port
# EXPOSE 5000

# # Set default environment variables (override in docker run or compose)
# ENV MLFLOW_ALLOWED_HOSTS="*" \
#     MLFLOW_TRACKING_URI="http://localhost:5000"

# Start the Flask server
CMD ["python", "server.py"]
