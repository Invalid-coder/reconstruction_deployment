# LAM Avatar Orchestrator - Docker Deployment

This repository provides a fully portable, containerized deployment setup for the **Large Avatar Model (LAM)** and its accompanying **Orchestrator Service**. 

The setup utilizes Docker Compose to spin up two interconnected microservices from a single, shared PyTorch base image:
1. **LAM Service**: Clones the official `aigc3d/LAM` repository, runs headless Blender 4.0.2, and handles the heavy lifting of 3D avatar reconstruction using your custom `lam_service.py`.
2. **Orchestrator Service**: Acts as an API gateway and asset registry, proxying requests to the internal LAM service and managing user face assets and zip files.

---

## 📋 Prerequisites

Before deploying, ensure your host machine has the following installed:
* **Linux OS** (Ubuntu 20.04/22.04 recommended)
* **NVIDIA GPU** (Required for LAM processing)
* **NVIDIA Drivers** installed and functioning (`nvidia-smi` should work)
* **Docker** and **Docker Compose** (v2.x recommended)
* **NVIDIA Container Toolkit** (Required to pass the GPU into the Docker containers)

---

## 📁 Directory Structure

Ensure your project directory is organized exactly like this before building:

```text
my_deployment/
├── lam_service.py           # Your custom LAM backend script
├── orchestrator_service.py  # The stripped-down orchestrator API script
├── requirements_orch.txt    # Python dependencies for the orchestrator
├── Dockerfile               # Unified Dockerfile for both services
├── docker-compose.yml       # Compose file defining the multi-container setup
└── .env                     # Environment variables configuration
```

# ⚙️ Configuration (.env)
The .env file controls the network mapping, storage paths, and security keys. Create a .env file in the root directory with the following content:

## LAM Service Config
Internal hostname (docker-compose automatically maps this to the container name)
```text
LAM_HOST=lam-service
LAM_PORT=8000
```

## Orchestrator Config
The public port exposed to your host machine
```text
ORCHESTRATOR_PORT=9000
```

### Security key required in the header (x-api-key) for Orchestrator endpoints
```text
API_KEY=dev-key-change-in-production
```

### Persistent storage directory mapped to the Docker volume
```text
STORAGE_ROOT=/app/orchestrator_storage
```

Note: Change dev-key-change-in-production to a secure string before deploying to a public server.

# 🚀 Deployment Instructions
1. Build and Start the Containers
To build the images (which will install system dependencies, download Blender, and clone the LAM repository) and start the services in detached mode, run:

```text
docker-compose up --build -d
```

Note: The initial build will take several minutes as it downloads PyTorch, Blender, and compiles dependencies.

2. Verify the Services are Running
Check the status of your containers:

```text
docker-compose ps
```

You should see both lam-service and lam-orchestrator with a status of Up.

3. Monitor the Logs
If you need to troubleshoot or watch the reconstruction progress in real-time, view the container logs:

```text
# View Orchestrator logs
docker-compose logs -f orchestrator

# View LAM Backend logs
docker-compose logs -f lam-service
```

Or another option:

```text
docker-compose build --no-cache
docker-compose up -d
```

# 📡 API Usage & Endpoints
You will interact primarily with the Orchestrator Service running on port 9000. All requests must include the x-api-key header matching your .env file.

1. Upload a Face Image for Reconstruction
Uploads an image, proxies it to the LAM service for 3D generation, and registers the resulting asset.

Request:

```text
curl -X POST "http://localhost:9000/assets/face" \
  -H "x-api-key: dev-key-change-in-production" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/photo.jpg"
```

Response:
```text
{
  "asset_id": "123e4567-e89b-12d3-a456-426614174000",
  "zip_url": "/assets/face/123e4567-e89b-12d3-a456-426614174000/download",
  "created_at": "2026-03-01T12:00:00.000000",
  "original_filename": "photo.jpg"
}
```

2. Download the Reconstructed Avatar
Downloads the .zip file containing the 3D meshes, blendshapes, and textures.

Request:

```text
curl -X GET "http://localhost:9000/assets/user" \
  -H "x-api-key: dev-key-change-in-production"
```

3. List User Assets
Retrieve all registered assets associated with your API key.

Request:

```text
curl -X GET "http://localhost:9000/assets/user" \
  -H "x-api-key: dev-key-change-in-production"
```

# 🧹 Maintenance & Cleanup
To stop the services:

```text
docker-compose down
```

To completely wipe the database/storage and start fresh:

```text
# This will delete the Docker volume containing your generated avatars!
docker-compose down -v
```

To rebuild after changing code (like updating lam_service.py):

```text
# Use --no-cache to force git clone to fetch the latest repo changes if needed
docker-compose build --no-cache
docker-compose up -d
```