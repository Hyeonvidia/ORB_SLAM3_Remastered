# ORB_SLAM3_Remastered

Modern C++17 refactoring of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for arm64 industrial deployment.

## Key Changes from Original

- **C++17** (original: C++11)
- **No `-march=native`** — portable across arm64 variants (Jetson Orin, etc.)
- **Third-party as git submodules** with wrappers for modifications
- **Docker-first** build environment (Ubuntu 22.04 arm64)

## Third-party Dependencies

| Library | Management | Purpose |
|---------|-----------|---------|
| g2o | Fork submodule (`Hyeonvidia/g2o-orbslam3`) | Graph optimization |
| DBoW2 | Upstream submodule + wrapper | Place recognition |
| DLib | Upstream submodule | DBoW2 utilities |
| Sophus | Upstream submodule | Lie groups (header-only) |
| Pangolin | Upstream submodule | Visualization |
| OpenGV | Upstream submodule + wrapper | PnP solvers |

## Quick Start

```bash
# Build Docker image
docker build -t orb-slam3-remastered:dev -f docker/Dockerfile .

# Test display forwarding
docker compose run --rm display-test

# Build everything
docker compose run --rm build
```

## License

GPLv3 (same as original ORB-SLAM3)
