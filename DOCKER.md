# Docker Setup — Sentinel

Run the full stack (frontend + backend + AI models) with Docker on any platform.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2+
- Model files in place:
  - `checkpoints/llava-fastvithd_0.5b_stage3/` (FastVLM weights)
  - `checkpoints/stgcn_ntu60_joint.pth` (ST-GCN action model)
  - `yolo11n.onnx`, `yolo11n-pose.onnx` (YOLO detection models)
- A `.env` file at the project root (copy from `.env.example` or create with your `OPENAI_API_KEY`)

## Quick Start (CPU — works on Mac, Windows, Linux)

```bash
docker compose up --build
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health check**: http://localhost:8000/api/health

## GPU Mode (NVIDIA CUDA — Linux / Windows with WSL2)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Stopping

```bash
docker compose down
```

## Architecture

```
┌─────────────┐       ┌──────────────┐
│  Frontend    │──────▶│   Backend    │
│  Next.js     │ :3000 │   FastAPI    │ :8000
│  (static)    │       │  + AI models │
└─────────────┘       └──────────────┘
                            │
                     ┌──────┴──────┐
                     │  Volumes    │
                     │ checkpoints │
                     │ yolo*.onnx  │
                     │ logs/       │
                     └─────────────┘
```

## Model files

Model weights are **not** baked into the Docker image — they're mounted as read-only volumes from your host. This keeps images small (~3 GB for CPU, ~8 GB for GPU) and lets you swap models without rebuilding.

Download models before the first run:

```bash
bash get_models.sh          # YOLO + FastVLM checkpoints
bash get_action_models.sh   # ST-GCN action recognition
```

## Environment Variables

| Variable              | Default                     | Description                                     |
| --------------------- | --------------------------- | ----------------------------------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`     | Backend URL (baked into frontend at build time) |
| `CORS_ORIGINS`        | `http://localhost:3000,...` | Comma-separated allowed origins                 |
| `OPENAI_API_KEY`      | —                           | For the AI Chat tab (optional)                  |

### Deploying to a remote server

If the backend runs on a different host/port, rebuild the frontend with the correct URL:

```bash
NEXT_PUBLIC_API_URL=http://your-server:8000 docker compose up --build
```

## Rebuilding a single service

```bash
docker compose build backend   # rebuild only the backend
docker compose build frontend  # rebuild only the frontend
```

## Logs

Backend logs are persisted to `./logs/` on the host via a volume mount.

## Platform Notes

| Platform                        | Mode       | Notes                                                 |
| ------------------------------- | ---------- | ----------------------------------------------------- |
| **macOS** (Intel/Apple Silicon) | CPU        | MPS is not available inside Docker; models run on CPU |
| **Windows** (WSL2)              | CPU or GPU | GPU requires NVIDIA Container Toolkit in WSL2         |
| **Linux**                       | CPU or GPU | Native GPU support with NVIDIA Container Toolkit      |
