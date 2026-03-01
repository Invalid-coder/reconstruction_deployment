# 1. Use the exact PyTorch version requested by install_cu121.sh (2.3.0) with DEVEL to get nvcc
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Explicitly set CUDA architectures to prevent "IndexError: list index out of range"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV MAX_JOBS=4

# Install system dependencies, Blender requirements, Git, and C++ build tools
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1-mesa-glx \
    libxi6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Download and install Blender 4.0.2 (Headless)
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz \
    && tar -xf blender-4.0.2-linux-x64.tar.xz -C /opt/ \
    && rm blender-4.0.2-linux-x64.tar.xz

ENV BLENDER_PATH="/opt/blender-4.0.2-linux-x64/blender"

WORKDIR /app

# Clone the official LAM repository
RUN git clone https://github.com/aigc3d/LAM.git /app/LAM

# Pre-install fundamental build dependencies needed for Cython/C++ compilation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel Cython numpy==1.23.0 ninja

# Install PyTorch and xformers exactly as specified in install_cu121.sh
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the LAM requirements
RUN pip install --no-cache-dir -r /app/LAM/requirements.txt

# Compile the FaceBoxesV2 extension (crucial step from install_cu121.sh)
RUN cd /app/LAM/external/landmark_detection/FaceBoxesV2/utils/ && sh make.sh

# Copy your custom LAM service script into the cloned repository
COPY lam_service.py /app/LAM/lam_service.py

# Install Orchestrator dependencies
COPY requirements_orch.txt /app/requirements_orch.txt
RUN pip install --no-cache-dir -r /app/requirements_orch.txt

# Copy the orchestrator service script
COPY orchestrator_service.py /app/orchestrator_service.py

# Set the default entry point to bash (Docker Compose will override this)
CMD ["bash"]