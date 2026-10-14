#!/bin/bash
# Screening de hiperparámetros del WANN con ANN clásica en Acrobot, Mountain Car discreto y Car.
# Para cada tarea, en este orden:
#   Fase 1   screening_reduce.py  (Optuna TPE + reducción del espacio de búsqueda)
#   Fase 2+3 screening_full.py    (búsqueda a presupuesto completo en el espacio reducido, y
#                                  re-ejecución de las mejores configuraciones con varias semillas)
#
# Mismo protocolo y mismos rangos que el run_all.sh de wann-cpp (SNN). Solo se buscan los
# hiperparámetros que existen en ambos (alg_probMoo, prob_addConn/addNode/enable/mutAct/initEnable,
# select_cullRatio/eliteRatio/tournSize).
#
# USO:   bash run_all.sh        (sin argumentos; los valores se cambian aquí abajo)
# OJO:   ENTRENA durante horas o días. Requiere haber compilado antes (./build.sh).

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
# Paralelismo (hebras totales usadas ≈ JOBS × OMP; la máquina tiene 576)
JOBS_REDUCE=8        # ensayos de Fase 1 en paralelo
OMP_REDUCE=64        # hebras OpenMP por ensayo en Fase 1      -> 8 x 64  = 512
JOBS_FULL=3          # corridas de Fase 2+3 en paralelo
OMP_FULL=190         # hebras OpenMP por corrida en Fase 2+3   -> 3 x 190 = 570

# Presupuesto
ROUNDS=4             # Fase 1: rondas de reducción
N_REDUCE=30          # Fase 1: ensayos por ronda
N_FULL=20            # Fase 2: ensayos
TOP=3                # Fase 3: mejores configuraciones a validar
SEEDS=11             # Fase 3: semillas por configuración
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi

for task in acrobot disc_mc car; do
    if [ ! -x "./build/wann_${task}" ]; then
        echo "ERROR: falta ./build/wann_${task}. Compila primero con ./build.sh" >&2
        exit 1
    fi
done

for task in acrobot disc_mc car; do
    echo "=== $task ==="
    "$PYTHON" screening_reduce.py --task "$task" \
        --rounds "$ROUNDS" --n "$N_REDUCE" \
        --jobs "$JOBS_REDUCE" --omp "$OMP_REDUCE"
    "$PYTHON" screening_full.py --task "$task" \
        --mode both --n "$N_FULL" --top "$TOP" --seeds "$SEEDS" \
        --jobs "$JOBS_FULL" --omp "$OMP_FULL"
done
