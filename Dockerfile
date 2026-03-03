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
# ADDED libxkbcommon-x11-0, libxkbcommon0, and libsm6 for Blender 4.0
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
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libsm6 \
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

# Install the rest of the LAM requirements WITH --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation -r /app/LAM/requirements.txt

# Compile the FaceBoxesV2 extension (crucial step from install_cu121.sh)
RUN cd /app/LAM/external/landmark_detection/FaceBoxesV2/utils/ && sh make.sh

WORKDIR /app/LAM

# ==========================================================
# DOWNLOAD LAM MODELS AND ASSETS (from LAM README)
# ==========================================================
# Download assets (including FLAME models) and extract them
RUN huggingface-cli download 3DAIGC/LAM-assets --local-dir ./tmp \
    && tar -xf ./tmp/LAM_assets.tar \
    && rm ./tmp/LAM_assets.tar \
    && tar -xf ./tmp/thirdparty_models.tar \
    && rm -rf ./tmp/ \
    && huggingface-cli download 3DAIGC/LAM-20K --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
# ==========================================================

# ==========================================================
# FIX FOR FBX MODULE AND AVATAR EXPORT (from LAM documentation)
# ==========================================================
# Download and install the pre-built FBX SDK Python wheel provided by LAM authors
RUN wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl \
    && pip install fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl \
    && rm fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl

# Install other requirements needed for Avatar Export
RUN pip install --no-cache-dir pathlib patool

# Download and extract the chatting avatar template files
RUN wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar \
    && mkdir -p assets/ \
    && tar -xf sample_oac.tar -C assets/ \
    && rm sample_oac.tar
# ==========================================================

# Copy your custom LAM service script into the cloned repository
COPY lam_service.py /app/LAM/lam_service.py

# Back to main app directory for orchestrator
WORKDIR /app

# Install Orchestrator dependencies
COPY requirements_orch.txt /app/requirements_orch.txt
RUN pip install --no-cache-dir -r /app/requirements_orch.txt

# Copy the orchestrator service script
COPY orchestrator_service.py /app/orchestrator_service.py

# Set the default entry point to bash
CMD ["bash"]