#!/bin/bash
# Hyperparameter screening of the classical-ANN WANN on Acrobot, discrete Mountain Car and Car:
# Phase 1 (screening_reduce.py: Optuna TPE + search-space reduction) then Phases 2+3 (screening_full.py:
# full-budget search in the reduced space, top-K re-run with several seeds), one task after the other.
#
# Same protocol and search ranges as wann-cpp's run_all.sh for the SNN (same order, same --rounds/--n/--top/
# --seeds). Only the hyperparameters that exist in both are searched (alg_probMoo, prob_addConn/addNode/
# enable/mutAct/initEnable, select_cullRatio/eliteRatio/tournSize); the SNN-only ones are not.
#
# THIS TRAINS (hours to days). Environment variables:
#   TASKS         space-separated subset / order     (default: "acrobot disc_mc car")
#                 also available: bipedal bottleneck
#   JOBS_REDUCE   parallel trials in Phase 1          (default 8)
#   OMP_REDUCE    OpenMP threads per trial, Phase 1   (default 64)
#   JOBS_FULL     parallel runs in Phases 2+3         (default 3)
#   OMP_FULL      OpenMP threads per run, Phases 2+3  (default 190)
#   ROUNDS        Phase 1 reduction rounds            (default 4)
#   N_REDUCE      Phase 1 trials per round            (default 30)
#   N_FULL        Phase 2 trials                      (default 20)
#   TOP           Phase 3 configs re-run              (default 3)
#   SEEDS         Phase 3 seeds per config            (default 11)
#   PYTHON        interpreter with optuna/pandas/scipy (default: .venv/bin/python if it exists, else python3)
#   EXE           use this trainer instead of ./build/wann_<task> (e.g. a fake one for testing)
#   DRY_RUN=1     only print the commands
#
# Usage:
#   bash run_all.sh
#   JOBS_REDUCE=4 OMP_REDUCE=128 bash run_all.sh
#   TASKS="car" bash run_all.sh
#   DRY_RUN=1 bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
TASKS=${TASKS:-"acrobot disc_mc car"}
JOBS_REDUCE=${JOBS_REDUCE:-8}
OMP_REDUCE=${OMP_REDUCE:-64}
JOBS_FULL=${JOBS_FULL:-3}
OMP_FULL=${OMP_FULL:-190}
ROUNDS=${ROUNDS:-4}
N_REDUCE=${N_REDUCE:-30}
N_FULL=${N_FULL:-20}
TOP=${TOP:-3}
SEEDS=${SEEDS:-11}

run() {
    echo "+ $*"
    if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}

for task in $TASKS; do
    case "$task" in
        acrobot|disc_mc|car|bipedal|bottleneck) ;;
        *) echo "Unknown task '$task' (acrobot disc_mc car bipedal bottleneck)" >&2; exit 1 ;;
    esac
    if [ -z "${DRY_RUN:-}" ] && [ -z "${EXE:-}" ] && [ ! -x "./build/wann_${task}" ]; then
        echo "ERROR: ./build/wann_${task} not found. Build first: ./build.sh" >&2
        exit 1
    fi
    exe_args=()
    if [ -n "${EXE:-}" ]; then exe_args=(--exe "$EXE"); fi

    echo "=== $task ==="
    run "$PYTHON" screening_reduce.py --task "$task" ${exe_args[@]+"${exe_args[@]}"} \
        --rounds "$ROUNDS" --n "$N_REDUCE" \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"
    run "$PYTHON" screening_full.py --task "$task" ${exe_args[@]+"${exe_args[@]}"} \
        --mode both --n "$N_FULL" --top "$TOP" --seeds "$SEEDS" \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
done
