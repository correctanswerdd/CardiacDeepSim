#!/usr/bin/env bash
# Every step of the paper in Docker (see README.md). Run from any directory; outputs go to ./outputs/<step>/.
#   ./run.sh simulate [--steps N] [--check]          step 01: cbcbeat simulation -> outputs/01_simulate/output.pickle
#   ./run.sh make-data [--source simulation|published]  step 02: data set -> outputs/02_make_data/dataset.npy
#   ./run.sh train lstm|node [--seed 0]              step 03: retrain a model (GPU) -> outputs/03_train/
#   ./run.sh evaluate table1 [--checkpoints DIR]     step 04: Table 1 -> outputs/04_evaluate/table1.{json,md}
#   ./run.sh figures fig1_traces|fig2_vm_comparison|fig3_long_term|fig4_propagation
#                                                    step 05: one figure -> outputs/05_figures/
# Images cds-sim:2026-11 (Dockerfile.sim) and cds-torch:2026-11 (Dockerfile) are built the first time.
# The paper's data is read from ./published/ (PUBLISHED_DIR=<path> to use another folder).
# Training uses the GPU (docker --gpus all); GPUS=none trains on the CPU. Steps 02, 04, 05 always run on the CPU.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Windows bash (MSYS): keep /work paths as they are

HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
SIM_IMAGE="${SIM_IMAGE:-cds-sim:2026-11}"
TORCH_IMAGE="${TORCH_IMAGE:-cds-torch:2026-11}"
GPUS="${GPUS:-all}"

image() {  # build an image the first time it is needed
    docker image inspect "$1" >/dev/null 2>&1 || docker build -f "$HERE/$2" -t "$1" "$HERE"
}
mounts() {
    MOUNTS=(-v "$HERE:/work" -w /work -e PYTHONDONTWRITEBYTECODE=1)
    if [ -n "${PUBLISHED_DIR:-}" ]; then
        MOUNTS+=(-v "$(cd "$PUBLISHED_DIR" && (pwd -W 2>/dev/null || pwd)):/published:ro" -e PUBLISHED_DIR_IN_CONTAINER=/published)
    fi
}
sim() { image "$SIM_IMAGE" Dockerfile.sim; mounts; docker run --rm "${MOUNTS[@]}" "$SIM_IMAGE" "$@"; }
torch_cpu() { image "$TORCH_IMAGE" Dockerfile; mounts; docker run --rm "${MOUNTS[@]}" "$TORCH_IMAGE" "$@"; }
torch_gpu() {
    image "$TORCH_IMAGE" Dockerfile; mounts
    local gpu=(); [ "$GPUS" = none ] || gpu=(--gpus "$GPUS")
    docker run --rm "${gpu[@]}" --shm-size 8g "${MOUNTS[@]}" "$TORCH_IMAGE" "$@"
}

mode="${1:-}"; shift || true
case "$mode" in
    simulate)  sim python3 01_simulate/simulate.py "$@" ;;
    make-data) torch_cpu python 02_make_data/make_data.py "$@" ;;
    train)     which="${1:?train needs lstm or node}"; shift
               case "$which" in lstm|node) torch_gpu python "03_train/train_$which.py" "$@" ;;
                   *) echo "train lstm|node" >&2; exit 2 ;; esac ;;
    evaluate)  name="${1:?evaluate needs a script name, e.g. table1}"; shift; torch_cpu python "04_evaluate/$name.py" "$@" ;;
    figures)   name="${1:?figures needs a script name, e.g. fig2_vm_comparison}"; shift; torch_cpu python "05_figures/$name.py" "$@" ;;
    *)         sed -n '2,/^set /p' "$0" | grep '^#'; exit 2 ;;
esac
