#!/usr/bin/env bash
# Runs one EuRoC sequence in one sensor configuration inside the dev container
# and writes its trajectory into results/.
#
#   ./tools/run_euroc.sh mono MH01
#   ./tools/run_euroc.sh stereo-inertial MH01
#
# Configurations: mono, mono-inertial, stereo, stereo-inertial
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:?usage: $0 <mono|mono-inertial|stereo|stereo-inertial> <sequence>}"
SEQ="${2:?usage: $0 <config> <sequence, e.g. MH01>}"

case "$CONFIG" in
  mono)            BIN=Monocular/mono_euroc;                        YAML=Monocular ;;
  mono-inertial)   BIN=Monocular-Inertial/mono_inertial_euroc;      YAML=Monocular-Inertial ;;
  stereo)          BIN=Stereo/stereo_euroc;                         YAML=Stereo ;;
  stereo-inertial) BIN=Stereo-Inertial/stereo_inertial_euroc;       YAML=Stereo-Inertial ;;
  *) echo "unknown configuration: $CONFIG" >&2; exit 2 ;;
esac

TAG="${SEQ}_${CONFIG//-/_}"
LOG="results/${TAG}.log"
mkdir -p results

echo "== ${CONFIG} on ${SEQ} -> results/f_${TAG}.txt (log: ${LOG})"
./docker/run.sh -- bash -c "
  cd /workspace/results &&
  /workspace/build/bin/${BIN} \
    /workspace/Vocabulary/ORBvoc.txt \
    /workspace/Examples/${YAML}/EuRoC.yaml \
    /datasets/EuRoC/${SEQ} \
    /workspace/Examples/${YAML}/EuRoC_TimeStamps/${SEQ}.txt \
    ${TAG} > /workspace/${LOG} 2>&1"

echo "   exit=$?  poses=$(wc -l < "results/f_${TAG}.txt" 2>/dev/null || echo 0)"
