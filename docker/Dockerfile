# =============================================================================
# ORB_SLAM3_Remastered — Development Environment
# Modern C++17 refactoring for arm64 industrial deployment
# Base: arm64 Ubuntu 22.04
# =============================================================================
# Strategy:
#   - System deps (this Dockerfile): Eigen3, OpenCV, build tools
#   - Project submodules (built via scripts/build_third_party.sh):
#     Pangolin, DLib, DBoW2, g2o, OpenGV
# =============================================================================

FROM ubuntu:22.04

LABEL maintainer="ORB_SLAM3_Remastered"
LABEL description="ORB-SLAM3 Remastered Dev Environment (arm64)"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# =============================================================================
# System packages & build tools
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    # Eigen3 3.4.0 (header-only, stable API)
    libeigen3-dev \
    # OpenCV 4.5.4 (system-level, no modification needed)
    # On Jetson Orin, replaced by JetPack SDK's CUDA-accelerated OpenCV
    libopencv-dev \
    # Pangolin build dependencies (X11 + OpenGL)
    libglew-dev \
    libgl1-mesa-dev \
    libwayland-dev \
    wayland-protocols \
    libxkbcommon-dev \
    libepoxy-dev \
    # Boost serialization (required by ORB_SLAM3 map save/load)
    libboost-serialization-dev \
    # OpenSSL crypto
    libssl-dev \
    # Python (evaluation scripts)
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Update shared library cache
RUN ldconfig

# =============================================================================
# Workspace
# =============================================================================
WORKDIR /workspace

# Verify system installations
RUN echo "=== System Dependencies ===" && \
    echo "Eigen3: $(pkg-config --modversion eigen3)" && \
    echo "OpenCV: $(pkg-config --modversion opencv4)" && \
    echo "CMake:  $(cmake --version | head -1)" && \
    echo "GCC:    $(gcc --version | head -1)"

CMD ["/bin/bash"]
