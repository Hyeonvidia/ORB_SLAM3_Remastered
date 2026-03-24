# =============================================================================
# ORB_SLAM3_Remastered — Development Environment
# Modern C++17 refactoring for arm64 industrial deployment
# Base: arm64 Ubuntu 22.04
# =============================================================================
# Strategy:
#   - System deps + third-party builds cached in image
#   - ccache for fast incremental rebuilds
#   - Runtime: only main project needs building
# =============================================================================

FROM ubuntu:22.04

LABEL maintainer="ORB_SLAM3_Remastered"
LABEL description="ORB-SLAM3 Remastered Dev Environment (arm64)"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# =============================================================================
# System packages & build tools + ccache
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ccache \
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
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-date-time-dev \
    libboost-regex-dev \
    libboost-timer-dev \
    libboost-chrono-dev \
    libboost-program-options-dev \
    libtbb-dev \
    libgtk-3-dev \
    libssl-dev \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig

# =============================================================================
# ccache configuration
# =============================================================================
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=2G
ENV PATH="/usr/lib/ccache:${PATH}"

# Symlink ccache as compiler (cmake will pick up /usr/lib/ccache/g++ etc.)
RUN ln -sf /usr/bin/ccache /usr/local/bin/gcc && \
    ln -sf /usr/bin/ccache /usr/local/bin/g++ && \
    ln -sf /usr/bin/ccache /usr/local/bin/cc && \
    ln -sf /usr/bin/ccache /usr/local/bin/c++

# =============================================================================
# Third-party libraries (cached in image)
# =============================================================================
WORKDIR /workspace

COPY third_party/ /workspace/third_party/
COPY scripts/build_third_party.sh /workspace/scripts/build_third_party.sh

RUN bash /workspace/scripts/build_third_party.sh && ldconfig
RUN rm -rf /tmp/third_party_build

# Verify
RUN echo "=== System Dependencies ===" && \
    echo "Eigen3: $(pkg-config --modversion eigen3)" && \
    echo "OpenCV: $(pkg-config --modversion opencv4)" && \
    echo "CMake:  $(cmake --version | head -1)" && \
    echo "GCC:    $(gcc --version | head -1)" && \
    echo "ccache: $(ccache --version | head -1)" && \
    echo "Pangolin: $(ls /usr/local/lib/libpango* | head -1)" && \
    echo "DBoW2: $(ls /usr/local/lib/libDBoW2*)" && \
    echo "DLib: $(ls /usr/local/lib/libDLib*)"

CMD ["/bin/bash"]
