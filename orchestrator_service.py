import os
import uuid
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import aiofiles
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
class ServiceConfig:
    """External service endpoints"""
    LAM_HOST = os.getenv("LAM_HOST", "lam-service")
    LAM_PORT = int(os.getenv("LAM_PORT", "8000"))

    # Storage paths
    STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "./orchestrator_storage"))
    FACES_DIR = STORAGE_ROOT / "faces"
    REGISTRY_FILE = STORAGE_ROOT / "registry.json"

    # Auth
    API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")


# ============================================================================
# Asset Registry
# ============================================================================
class AssetRegistry:
    """Simple file-based registry for faces"""

    def __init__(self, registry_file: Path):
        self.registry_file = registry_file
        self.data = self._load()
        ServiceConfig.FACES_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"faces": {}, "users": {}}

    def _save(self):
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def register_face(self, user_id: str, asset_id: str, zip_path: str, original_filename: str) -> Dict[str, Any]:
        entry = {
            "asset_id": asset_id,
            "user_id": user_id,
            "zip_path": zip_path,
            "original_filename": original_filename,
            "created_at": datetime.utcnow().isoformat()
        }
        self.data["faces"][asset_id] = entry
        if user_id not in self.data["users"]:
            self.data["users"][user_id] = {"faces": []}
        self.data["users"][user_id]["faces"].append(asset_id)
        self._save()
        return entry

    def get_face(self, asset_id: str) -> Optional[Dict[str, Any]]:
        return self.data["faces"].get(asset_id)

    def get_user_assets(self, user_id: str) -> Dict[str, Any]:
        return self.data["users"].get(user_id, {"faces": []})


# ============================================================================
# Service Clients
# ============================================================================
class LAMClient:
    """Client for LAM Avatar Reconstruction Service"""

    def __init__(self):
        self.base_url = f"http://{ServiceConfig.LAM_HOST}:{ServiceConfig.LAM_PORT}"

    async def reconstruct_avatar(self, image_path: str, output_path: str) -> str:
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/png')}
                logger.info(f"Sending reconstruction request to {self.base_url}/reconstruct-avatar")
                response = requests.post(
                    f"{self.base_url}/reconstruct-avatar",
                    files=files,
                    timeout=300
                )
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Avatar reconstructed successfully: {output_path}")
                return output_path
            else:
                error_msg = response.json().get('detail', response.text)
                raise HTTPException(status_code=response.status_code, detail=f"LAM reconstruction failed: {error_msg}")
        except requests.exceptions.RequestException as e:
            logger.error(f"LAM service error: {e}")
            raise HTTPException(status_code=503, detail=f"LAM service unavailable: {str(e)}")


# ============================================================================
# API Models & Auth
# ============================================================================
class FaceAssetResponse(BaseModel):
    asset_id: str
    zip_url: str
    created_at: str
    original_filename: str


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != ServiceConfig.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ============================================================================
# Application Setup
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Avatar Orchestrator Service...")
    app.state.registry = AssetRegistry(ServiceConfig.REGISTRY_FILE)
    app.state.lam_client = LAMClient()
    logger.info("Service initialized successfully")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="LAM Orchestrator", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    return {"status": "healthy", "service": "LAM Orchestrator", "endpoints": {"face_upload": "/assets/face"}}


@app.post("/assets/face", response_model=FaceAssetResponse)
async def upload_face_asset(file: UploadFile = File(...), user_id: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Face upload request from user {user_id}")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    asset_id = str(uuid.uuid4())
    image_path = ServiceConfig.FACES_DIR / f"{asset_id}_{file.filename}"

    async with aiofiles.open(image_path, 'wb') as f:
        await f.write(await file.read())

    logger.info(f"[{request_id}] Image saved: {image_path}")

    zip_path = ServiceConfig.FACES_DIR / f"{asset_id}_avatar.zip"
    try:
        await app.state.lam_client.reconstruct_avatar(str(image_path), str(zip_path))
    except Exception as e:
        if image_path.exists(): image_path.unlink()
        raise

    entry = app.state.registry.register_face(user_id, asset_id, str(zip_path), file.filename)

    logger.info(f"[{request_id}] Face asset registered: {asset_id}")

    return FaceAssetResponse(asset_id=asset_id, zip_url=f"/assets/face/{asset_id}/download",
                             created_at=entry["created_at"], original_filename=file.filename)


@app.get("/assets/user")
async def get_user_assets(user_id: str = Depends(verify_api_key)):
    assets = app.state.registry.get_user_assets(user_id)
    faces = []
    for asset_id in assets.get("faces", []):
        face = app.state.registry.get_face(asset_id)
        if face:
            faces.append({"asset_id": face["asset_id"], "zip_url": f"/assets/face/{asset_id}/download",
                          "created_at": face["created_at"], "original_filename": face["original_filename"]})
    return {"user_id": user_id, "faces": faces}


@app.get("/assets/face/{asset_id}/download")
async def download_face_asset(asset_id: str, user_id: str = Depends(verify_api_key)):
    face = app.state.registry.get_face(asset_id)
    if not face or face["user_id"] != user_id:
        raise HTTPException(status_code=404 if not face else 403,
                            detail="Asset not found" if not face else "Access denied")

    zip_path = Path(face["zip_path"])
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found")

    return StreamingResponse(open(zip_path, "rb"), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={zip_path.name}"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uvicorn.run("orchestrator_service:app", host=args.host, port=args.port)