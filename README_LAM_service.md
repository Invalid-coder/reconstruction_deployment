# LAM Avatar Reconstruction Service

## Running the Server

### Start the server with Blender path:

```bash
python lam_service.py --blender_path ~/software/blender-4.0.2-linux-x64/blender
```

### Optional: Custom host and port

```bash
python lam_service.py \
  --blender_path ~/software/blender-4.0.2-linux-x64/blender \
  --host 0.0.0.0 \
  --port 8000
```

### Using environment variable:

```bash
export BLENDER_PATH=~/software/blender-4.0.2-linux-x64/blender
python lam_service.py
```

---

## Using the Client

### Basic usage:

```python
from lam_client import LAMClient

# Connect to server
client = LAMClient(host="localhost", port=8000)

# Check if server is ready
client.health_check()

# Reconstruct avatar from image
client.reconstruct("path/to/image.jpg")
```

### Remote server:

```python
client = LAMClient(host="184.105.87.177", port=8000)
client.reconstruct("my_photo.jpg", output_zip="my_avatar.zip")
```

### Batch processing:

```python
images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
results = client.reconstruct_batch(images, output_dir="avatars_output")
```

### Command line:

```bash
python lam_client.py
```

Edit the `__main__` section in `lam_client.py` to customize the image path and server address.