# AudioGhost AI - Docker Guide

This guide explains how to run the AudioGhost AI backend using Docker.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **NVIDIA GPU** with drivers installed.
- **NVIDIA Container Toolkit** installed (for GPU support in Docker).

### HuggingFace Token
To download the models, you need a HuggingFace token with access to `facebook/sam-audio-large`.
There are two ways to provide it:
- **Environment File (Recommended)**: Create a file named `.env` in the root directory (you can copy `.env.example`) and set your token in `HF_TOKEN`.
- **Token File**: Create a file named `.hf_token` inside the `backend/` directory and paste your token there.

## Setup

1. **Verify Directories**
   Ensure the following directories exist (created automatically by the setup, but good to check):
   - `data/redis` (Redis persistence)
   - `data/uploads` (Audio uploads)
   - `data/outputs` (Processed audio)
   - `data/checkpoints` (Model weights - **Explicitly configured via `HF_HOME`**)

2. **Build and Start**
   Run the following command to build the images and start the services:

   ```bash
   docker compose up -d --build
   ```

   > 💡 **Faster Builds**: The Dockerfile now uses `uv` for ultra-fast dependency installation and caches packages during the build process.

   This will start:
   - **api**: The FastAPI backend (Port 9722)
   - **worker**: The Celery worker for processing audio
   - **redis_docker**: The Redis message broker (Port 6379)

## Usage

### Check Status
Check if services are running:
```bash
docker compose ps
```

View logs:
```bash
docker compose logs -f
```

### Accessing the API
The API is available at: http://localhost:9722
- **Docs**: http://localhost:9722/docs
- **Health Check**: http://localhost:9722/health

### Processing Audio
You can interact with the API via the Swagger UI (`/docs`) or using `curl`.

**Example request:**
```bash
curl -X 'POST' \
  'http://localhost:9722/api/separate/' \
  -F 'file=@/path/to/your/audio.mp3' \
  -F 'description=vocals' \
  -F 'mode=extract'
```

### Stopping
To stop the services:
```bash
docker compose down
```

## Data Persistence & Cache Management

### Saved Data
All important data is stored in the `data/` directory in the project root:
- **Redis data**: `data/redis`
- **Files**: `data/uploads` and `data/outputs`
- **Models**: `data/checkpoints` (Configured via `HF_HOME=/app/checkpoints`)

### Build Cache (Advanced)
We use `uv` with Docker's build cache (`--mount=type=cache`) to speed up re-builds. 
- **Where is it?**: This cache is managed by Docker internally (usually in `/var/lib/docker/buildkit`). It is NOT in your project directory.
- **How to clean it?**: If you want to free up space used by these cached packages (e.g., if you are running low on disk space), run:
  
  ```bash
  docker builder prune
  ```
  
  This command will ask for confirmation and delete **all** build caches (not just for this project). It is safe to run; the next build will simply download dependencies again.
