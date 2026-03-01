# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, Blender requirements, and Git
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1-mesa-glx \
    libxi6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Blender 4.0.2 (Headless)
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz \
    && tar -xf blender-4.0.2-linux-x64.tar.xz -C /opt/ \
    && rm blender-4.0.2-linux-x64.tar.xz

ENV BLENDER_PATH="/opt/blender-4.0.2-linux-x64/blender"

WORKDIR /app

# Clone the official LAM repository
RUN git clone https://github.com/aigc3d/LAM.git /app/LAM

# Copy your custom LAM service script into the cloned repository
COPY lam_service.py /app/LAM/lam_service.py

# Install LAM dependencies
RUN pip install --no-cache-dir -r /app/LAM/requirements.txt

# Install Orchestrator dependencies
COPY requirements_orch.txt /app/requirements_orch.txt
RUN pip install --no-cache-dir -r /app/requirements_orch.txt

# Copy the orchestrator service script
COPY orchestrator_service.py /app/orchestrator_service.py

# Set the default entry point to bash (Docker Compose will override this)
CMD ["bash"]