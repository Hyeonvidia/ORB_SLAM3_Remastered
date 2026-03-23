# =============================================================================
# ORB_SLAM3_Remastered — Development Environment
# Modern C++17 refactoring for arm64 industrial deployment
# Base: arm64 Ubuntu 22.04
# =============================================================================
# Strategy:
#   - Stage 1: System deps + third-party builds (cached in image)
#   - Runtime: Only main project needs building (fast iteration)
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
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    libeigen3-dev \
    libopencv-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libwayland-dev \
    wayland-protocols \
    libxkbcommon-dev \
    libepoxy-dev \
    libboost-serialization-dev \
    libgtk-3-dev \
    libssl-dev \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig

# =============================================================================
# Copy third-party sources and build script
# =============================================================================
WORKDIR /workspace

# Copy only third-party sources and build script (for Docker cache efficiency)
COPY third_party/ /workspace/third_party/
COPY scripts/build_third_party.sh /workspace/scripts/build_third_party.sh

# Build and install third-party libraries into /usr/local
RUN bash /workspace/scripts/build_third_party.sh && ldconfig

# Clean up build temps (sources stay for headers)
RUN rm -rf /tmp/third_party_build

# Verify
RUN echo "=== System Dependencies ===" && \
    echo "Eigen3: $(pkg-config --modversion eigen3)" && \
    echo "OpenCV: $(pkg-config --modversion opencv4)" && \
    echo "CMake:  $(cmake --version | head -1)" && \
    echo "GCC:    $(gcc --version | head -1)" && \
    echo "Pangolin: $(ls /usr/local/lib/libpango* | head -1)" && \
    echo "DBoW2: $(ls /usr/local/lib/libDBoW2*)" && \
    echo "DLib: $(ls /usr/local/lib/libDLib*)"

CMD ["/bin/bash"]
