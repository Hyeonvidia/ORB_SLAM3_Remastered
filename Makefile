.PHONY: help up down restart attach build euroc_mono euroc_mono_inertial euroc_stereo euroc_stereo_inertial kitti_mono kitti_stereo build_gtsam build_abtest kitti_stereo_gtsam kitti_stereo_abtest

DOCKER_EXEC = docker compose exec dev bash -c
DOCKER_RUN = docker compose run --rm dev bash -c
LD_LIB = LD_LIBRARY_PATH=/workspace/lib:/usr/local/lib

help:
	@echo "ORB-SLAM3 Remastered Makefile"
	@echo "--------------------------------------------------------"
	@echo "Container Management:"
	@echo "  make up       - Start docker container"
	@echo "  make down     - Stop and remove docker container"
	@echo "  make restart  - Restart docker container cleanly"
	@echo "  make attach   - Access bash shell inside container"
	@echo "  make build    - Compile ORB-SLAM3 codebase inside container"
	@echo ""
	@echo "EuRoC Dataset (MH01):"
	@echo "  make euroc_mono"
	@echo "  make euroc_mono_inertial"
	@echo "  make euroc_stereo"
	@echo "  make euroc_stereo_inertial"
	@echo ""
	@echo "KITTI Dataset (Sequence 00):"
	@echo "  make kitti_mono"
	@echo "  make kitti_stereo"
	@echo ""
	@echo "GTSAM-specific:"
	@echo "  make build_gtsam          - Build with GTSAM-only backend"
	@echo "  make build_abtest         - Build with A/B test (g2o+GTSAM)"
	@echo "  make kitti_stereo_gtsam   - Run KITTI stereo with GTSAM"
	@echo "  make kitti_stereo_abtest  - Run KITTI stereo with A/B test"
	@echo "--------------------------------------------------------"

up:
	xhost +local:docker
	docker compose up -d

down:
	docker compose down

restart:
	docker compose down
	xhost +local:docker
	docker compose up -d

attach:
	docker compose exec dev bash

build:
	$(DOCKER_EXEC) "mkdir -p /workspace/build && cd /workspace/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8"

euroc_mono:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/mono_euroc Vocabulary/ORBvoc.txt examples/Monocular/EuRoC.yaml /datasets/EuRoc/MH01 examples/Monocular/EuRoC_TimeStamps/MH01.txt"

euroc_mono_inertial:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/mono_inertial_euroc Vocabulary/ORBvoc.txt examples/Monocular-Inertial/EuRoC.yaml /datasets/EuRoc/MH01 examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt"

euroc_stereo:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/stereo_euroc Vocabulary/ORBvoc.txt examples/Stereo/EuRoC.yaml /datasets/EuRoc/MH01 examples/Stereo/EuRoC_TimeStamps/MH01.txt"

euroc_stereo_inertial:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/stereo_inertial_euroc Vocabulary/ORBvoc.txt examples/Stereo-Inertial/EuRoC.yaml /datasets/EuRoc/MH01 examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt"

kitti_mono:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/mono_kitti Vocabulary/ORBvoc.txt examples/Monocular/KITTI00-02.yaml /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/00"

kitti_stereo:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/stereo_kitti Vocabulary/ORBvoc.txt examples/Stereo/KITTI00-02.yaml /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/00"

# GTSAM-specific targets
build_gtsam:
	$(DOCKER_EXEC) "cd /workspace && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GTSAM=ON && make -j8"

build_abtest:
	$(DOCKER_EXEC) "cd /workspace && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GTSAM=ON -DABTEST_MODE=ON && make -j8"

kitti_stereo_gtsam:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/stereo_kitti Vocabulary/ORBvoc.txt examples/Stereo/KITTI00-02.yaml /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/00 2>&1 | tee /workspace/kitti_gtsam.log"

kitti_stereo_abtest:
	$(DOCKER_RUN) "cd /workspace && $(LD_LIB) ./bin/stereo_kitti Vocabulary/ORBvoc.txt examples/Stereo/KITTI00-02.yaml /datasets/kitti_dataset/data_odometry_gray/dataset/sequences/00 2>&1 | tee /workspace/kitti_abtest.log"
