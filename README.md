# ORB_SLAM3_Remastered

Modern C++17 refactoring of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for arm64 industrial deployment.

Based on the paper: *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM* (IEEE TRO 2021)

---

## Key Problems Solved / 주요 해결 사항

### 1. Build System Modernization / 빌드 시스템 현대화

| Problem | Solution |
|---------|----------|
| Original uses C++11, `-march=native` (x86 only) | **C++17**, no `-march=native` — portable across arm64 (Jetson Orin) |
| Third-party code copied and modified in `Thirdparty/` | **6 git submodules** + wrapper pattern, source directories stay clean after build |
| No reproducible build environment | **Docker-first** (Ubuntu 22.04 arm64), third-party pre-built in image |
| Slow rebuild on every code change | **ccache** — clean rebuild 3s (vs 1m40s without) |

- 원본은 C++11에 `-march=native`(x86 전용)를 사용하여 arm64에서 빌드 불가
- Third-party 코드를 직접 복사/수정하던 방식을 **git submodule + wrapper**로 분리
- Docker 이미지에 third-party를 사전 빌드하고 **ccache**로 재빌드 3초 달성

### 2. Bug Fixes from Original / 원본 버그 수정

| Bug | Impact | Fix |
|-----|--------|-----|
| `Settings.hpp`: `GeometricCamera*` uninitialized pointers | **Segfault** on KITTI stereo rectified mode | Initialize to `nullptr`, fallback to Camera1 |
| `LoopClosing.hpp`: `bool mnFullBAIdx` with `++` operator | **Compile error** in C++17 (bool increment forbidden) | Changed to `int` |
| `monotonic_clock` usage in Examples | **Deprecated** in C++17, removed in some compilers | Replaced with `steady_clock` |
| `using namespace std` leaked from DBoW2 `TemplatedVocabulary.h` | Hidden dependency, fragile builds | Removed all `using namespace std`, added explicit `std::` prefix (40 files, ~1940 lines) |
| DBoW2 `loadFromTextFile` not in upstream | Cannot load ORBvoc.txt vocabulary | Wrapper class with text file loading |
| DBoW2 `BowVector`/`FeatureVector` missing `serialize` | Boost serialization fails | Non-intrusive serialization in wrapper |

- 원본 Settings.hpp의 초기화되지 않은 포인터로 KITTI Stereo Rectified에서 **segfault** 발생 → `nullptr` 초기화로 해결
- C++17에서 `bool++` 금지 → `int`로 변경
- DBoW2 upstream에 없는 `loadFromTextFile`, `serialize`를 **wrapper**로 해결

### 3. Architecture Refactoring / 아키텍처 리팩토링

**Modular directory structure aligned with ORB-SLAM3 paper (Fig. 1):**

```
include/
├── core/           Frame, KeyFrame, MapPoint, Map, ImuTypes, Converter
├── tracking/       Tracking, ORBextractor, FeatureExtractor, ORBmatcher
├── mapping/        LocalMapping, Optimizer, G2oTypes
├── recognition/    PlaceRecognition, KeyFrameDatabase, ORBVocabulary
├── closing/        LoopClosing, Sim3Solver
├── atlas/          Atlas
├── visualization/  Viewer, FrameDrawer, MapDrawer
├── camera/         GeometricCamera, Pinhole, KannalaBrandt8, CameraFactory
├── io/             Settings, SerializationUtils, TrajectoryWriter
└── System.hpp      Thin orchestrator (834 lines, down from 1553)
```

- 논문 Fig. 1의 모듈 구조에 맞춰 9개 디렉토리로 재배치
- **PlaceRecognition** 모듈을 LoopClosing에서 독립 추출 (~650줄)
- **TrajectoryWriter**를 System.cpp에서 추출 (System.cpp 1553줄 → 834줄, -46%)

### 4. Design Patterns Applied / 적용된 디자인 패턴

| Pattern | Application |
|---------|------------|
| **Factory** | `CameraFactory` — Pinhole/KannalaBrandt8 creation from config string |
| **Observer** | `TrackingObserver` — FrameDrawer/MapDrawer decoupled from Tracking |
| **RAII** | `unique_ptr` for Camera pointers (Settings) and ORBextractor (Tracking) |
| **Strategy** | `FeatureExtractor` interface + `KeypointDistributor` (OctTree/Grid) |
| **Template Method** | `SlamRunner` + `DatasetLoader` — Examples reduced from 100-300 lines to ~30 lines |

- **Factory**: 카메라 모델 생성을 문자열 기반으로 통합
- **Observer**: Tracking과 시각화 모듈 간 결합도 감소
- **RAII**: raw pointer 메모리 누수 제거 (`unique_ptr`)
- **Strategy**: 특징점 추출/분포 알고리즘 교체 가능 (OctTree ↔ Grid)
- **Template Method**: 6개 Example 파일의 중복 코드를 공통 base로 추출

### 5. Modern C++ Style / 모던 C++ 스타일

| Change | Scope |
|--------|-------|
| `#pragma once` | All 31 headers |
| `nullptr` | 115 occurrences across 13 files |
| `override` | 105+ virtual function overrides in 4 files |
| `std::` prefix | All STL types explicitly qualified (no `using namespace std`) |
| `.hpp` / `.cpp` | All headers and sources renamed |

### 6. Testing / 테스트

- **8 GTest suites, 71 tests** — all passing
- Modules tested: Converter, ImuTypes, ORBextractor, GeometricTools, ORBmatcher, DBoW2Wrapper, MapPoint, Map

### 7. Dataset Validation / 데이터셋 검증

| Dataset | Mode | Result | Performance |
|---------|------|--------|-------------|
| EuRoC MH01 | Mono | 367 KFs | 11.2 ms/frame |
| EuRoC MH01 | Stereo | 152 KFs | — |
| KITTI 00 | Mono | 15 KFs | 13.5 ms/frame |
| KITTI 00 | Stereo | Loop ×4 detected | 23.4 ms/frame |

---

## Third-party Dependencies

| Library | Management | Purpose |
|---------|-----------|---------|
| g2o | Fork submodule ([Hyeonvidia/g2o-orbslam3](https://github.com/Hyeonvidia/g2o-orbslam3)) | Graph optimization |
| DBoW2 | Upstream submodule + wrapper | Place recognition |
| DLib | Upstream submodule | DBoW2 utilities |
| Sophus | Upstream submodule | Lie groups (header-only) |
| Pangolin | Upstream submodule | Visualization |
| OpenGV | Upstream submodule + wrapper | PnP solvers |

Wrappers in `third_party/wrappers/`: `slam3_dbow2_wrapper`, `slam3_opengv_wrapper`, `slam3_g2o_wrapper`

---

## Quick Start

```bash
# 1. Build Docker image (includes all third-party, one-time)
docker build -t orb-slam3-remastered:dev -f docker/Dockerfile .

# 2. Start persistent dev container
docker compose up dev -d

# 3. Enter container and build (3 seconds with ccache)
docker compose exec dev bash
bash scripts/build.sh

# 4. Run SLAM
./bin/mono_euroc Vocabulary/ORBvoc.txt \
    Examples/Monocular/EuRoC.yaml \
    /datasets/EuRoc/MH01/ \
    Examples/Monocular/EuRoC_TimeStamps/MH01.txt

# 5. Run tests
for t in bin/test_*; do $t --gtest_brief=1; done
```

### With Pangolin Viewer (macOS + XQuartz)

```bash
xhost +localhost
docker compose exec dev bash -c "
    DISPLAY=host.docker.internal:0 \
    XDG_RUNTIME_DIR=/tmp \
    ./bin/stereo_kitti Vocabulary/ORBvoc.txt \
        Examples/Stereo/KITTI00-02.yaml \
        /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/00"
```

---

## Project Structure

```
ORB_SLAM3_Remastered/
├── docker/                 Dockerfile + display test
├── scripts/                build_third_party.sh, build.sh
├── third_party/
│   ├── g2o/                submodule (fork)
│   ├── DBoW2/              submodule (upstream)
│   ├── DLib/               submodule (upstream)
│   ├── Sophus/             submodule (upstream)
│   ├── Pangolin/           submodule (upstream)
│   ├── OpenGV/             submodule (upstream)
│   └── wrappers/           dbow2_wrapper, opengv_wrapper, g2o_wrapper
├── include/                Modular headers (9 subdirectories)
├── src/                    Modular sources (9 subdirectories)
├── Examples/               Refactored with SlamRunner/DatasetLoader
├── tests/                  8 GTest suites
└── Vocabulary/             ORBvoc.txt.tar.gz
```

---

## License

GPLv3 (same as original ORB-SLAM3)
