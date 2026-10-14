#!/usr/bin/env bash
# Checks that this repo simulates the same environments as the SNN code (wann-cpp).
#
#   tests/env/check_against_snn.sh [path/to/wann-cpp] [build-dir]
#
# For each task: build a fixed random network, let wann-cpp's prebuilt wann_<task>_eval run one episode
# with it (spiking controller, a given episode seed) and record the trajectory; replay the SNN's actions
# open-loop in this repo's environment with the same seed (env_replay_<task>); compare observations and
# rewards step by step. Evaluation only — nothing is trained and wann-cpp is not modified (only read/run).
set -u
SNN=${1:-$(dirname "$0")/../../../wann-cpp}
BUILD=${2:-$(dirname "$0")/../../build}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
fail=0
# task : in out hidden : net-seed : snn weight : config
SPECS="car:9 2 5:9:5.0:car_snn.json
acrobot:6 3 4:5:3.0:acrobot_snn.json
disc_mc:2 3 3:9:3.0:disc_mc_snn.json
bipedal:24 4 6:5:3.0:bipedal_snn.json
bottleneck:12 2 4:22:60.0:bottleneck_snn.json"
while IFS=: read -r task dims nseed w cfg; do
  [ -z "$task" ] && continue
  [ -x "$BUILD/env_replay_$task" ] || { echo "== $task: $BUILD/env_replay_$task not built, skipping"; continue; }
  [ -x "$SNN/build/wann_${task}_eval" ] || { echo "== $task: $SNN/build/wann_${task}_eval not found, skipping"; continue; }
  # shellcheck disable=SC2086
  python3 "$HERE/mknet.py" "$OUT/$task.out" $dims $nseed
  for eseed in 7 21 99; do
    echo "== $task  (episode seed $eseed, shared weight $w)"
    ( cd "$SNN" && ./build/wann_${task}_eval -f "$OUT/$task.out" -d "p/$cfg" -w "$w" -n 1 -s "$eseed" -S 0 \
        -o "$OUT/snn_${task}_$eseed.csv" >/dev/null ) || { echo "SNN run failed"; fail=1; continue; }
    "$BUILD/env_replay_$task" -d "$SNN/p/$cfg" -c "$OUT/snn_${task}_$eseed.csv" -s "$eseed" \
        -o "$OUT/ann_${task}_$eseed.csv" || { fail=1; continue; }
    python3 "$HERE/compare_env.py" "$OUT/snn_${task}_$eseed.csv" "$OUT/ann_${task}_$eseed.csv" ${TOL:+--tol $TOL} || fail=1
  done
done <<< "$SPECS"
echo "(files in $OUT)"
exit $fail
