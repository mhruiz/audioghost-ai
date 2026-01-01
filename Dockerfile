FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies
# First copy requirements to utilize cache
COPY backend/requirements.txt /app/backend/requirements.txt

# Install dependencies using uv with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system git+https://github.com/facebookresearch/sam-audio.git && \
    uv pip install --system -r /app/backend/requirements.txt

# Copy the entire project
COPY . /app

# Expose the API port
EXPOSE 9722

# Default working directory for the application
WORKDIR /app/backend

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9722"]
