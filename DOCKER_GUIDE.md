# AudioGhost AI - Docker Guide

This guide explains how to run the AudioGhost AI backend using Docker.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **NVIDIA GPU** with drivers installed.
- **NVIDIA Container Toolkit** installed (for GPU support in Docker).

## Setup

1. **Verify Directories**
   Ensure the following directories exist (created automatically by the setup, but good to check):
   - `data/redis` (Redis persistence)
   - `data/uploads` (Audio uploads)
   - `data/outputs` (Processed audio)
   - `data/checkpoints` (Model weights)

2. **Build and Start**
   Run the following command to build the images and start the services:

   ```bash
   docker-compose up -d --build
   ```

   This will start:
   - **api**: The FastAPI backend (Port 8000)
   - **worker**: The Celery worker for processing audio
   - **redis_docker**: The Redis message broker (Port 6379)

## Usage

### Check Status
Check if services are running:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs -f
```

### Accessing the API
The API is available at: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Processing Audio
You can interact with the API via the Swagger UI (`/docs`) or using `curl`.

**Example request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/separate/' \
  -F 'file=@/path/to/your/audio.mp3' \
  -F 'description=vocals' \
  -F 'mode=extract'
```

### Stopping
To stop the services:
```bash
docker-compose down
```

## Data Persistence
All data is stored in the `data/` directory in the project root.
- **Redis data**: `data/redis`
- **Files**: `data/uploads` and `data/outputs`
- **Models**: `data/checkpoints`

Use `docker-compose down -v` if you want to remove volumes (note: since we use bind mounts, the data in `data/` will actually PERSIST on your disk even if you remove volumes).
