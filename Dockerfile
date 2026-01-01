FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

# Install system dependencies
# ffmpeg is required for audio processing
# git is required to install sam-audio from github
# curl is good to have
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# First copy requirements to utilize cache
COPY backend/requirements.txt /app/backend/requirements.txt

# Install SAM Audio from GitHub
RUN pip install git+https://github.com/facebookresearch/sam-audio.git

# Install backend dependencies
RUN pip install -r /app/backend/requirements.txt

# Copy the entire project
COPY . /app

# Expose the API port
EXPOSE 8000

# Default working directory for the application
WORKDIR /app/backend

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
