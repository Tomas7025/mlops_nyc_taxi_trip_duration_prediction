# ---- Base Python image ----
FROM python:3.13-slim AS base

# Avoid pyc & flush output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Install OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN pip install uv

# ---- Set work directory ----
WORKDIR /app

# ---- Copy dependency files ----
COPY src/training/ .

# ---- Copy the transformers.py ----
COPY src/transformers.py .

# ---- Get the data/ volume ----
# VOLUME ["/app/data"]

# ---- Install dependencies (same as CI) ----
# RUN uv sync --frozen
RUN pip install --upgrade pip && \
   pip install -r requirements.txt

# ---- By default, run the CI training ----
# ENTRYPOINT ["uv", "run", "python", "train_model.py"]
ENTRYPOINT ["python", "train_model.py"]
# ENTRYPOINT ["uv", "run", "-m", "src.training.train"]
 