#!/bin/bash
# Repite las Fases 2 y 3 (screening_full.py) de acrobot, disc_mc y car reutilizando la Fase 1
# ya hecha (screening_reduce/<task>/space_log.jsonl), sin sobreescribir los resultados anteriores.
#
# Salidas nuevas (con TAG=v2):  screening_full/<task>_v2/   log/full_p2_<task>_v2/   log/full_p3_<task>_v2/
# Las de la corrida anterior (screening_full/<task>/, log/full_p*_<task>/) no se tocan.
#
# USO:   bash run_full_again.sh [TAG]      (TAG por defecto: v2)
# OJO:   ENTRENA durante horas o días. Requiere ./build.sh y la Fase 1 de cada tarea.

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
TASKS="car"
JOBS_FULL=2          # corridas de Fase 2+3 en paralelo
OMP_FULL=265         # hebras OpenMP por corrida             -> 3 x 190 = 570
N_FULL=20            # Fase 2: ensayos
TOP=3                # Fase 3: mejores configuraciones a validar
SEEDS=11             # Fase 3: semillas por configuración
PATIENCE=0           # early stopping (generaciones sin récord); 0 = desactivado también en car
SEED=0               # semilla base de la búsqueda (cámbiala para que no repita los mismos ensayos)
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-v2}"
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi

# Comprobaciones previas, antes de empezar a entrenar
for task in $TASKS; do
    if [ ! -x "./build/wann_${task}" ]; then
        echo "ERROR: falta ./build/wann_${task}. Compila primero con ./build.sh" >&2
        exit 1
    fi
    if [ ! -s "screening_reduce/${task}/space_log.jsonl" ]; then
        echo "ERROR: falta la Fase 1 de ${task} (screening_reduce/${task}/space_log.jsonl)" >&2
        exit 1
    fi
    if [ -e "screening_full/${task}_${TAG}" ]; then
        echo "ERROR: screening_full/${task}_${TAG} ya existe; usa otro TAG" >&2
        exit 1
    fi
done

for task in $TASKS; do
    echo "=== $task (tag: $TAG, Fase 1: screening_reduce/$task) ==="
    "$PYTHON" screening_full.py --task "$task" \
        --mode both --n "$N_FULL" --top "$TOP" --seeds "$SEEDS" \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL" --seed "$SEED" \
        --early-stop-patience "$PATIENCE" \
        --tag "$TAG" --reduce-from "$task"
done
