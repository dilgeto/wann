#!/usr/bin/env bash
# End-to-end test of screening_reduce.py / screening_full.py with a fake trainer (no training, no C++ run).
#   tests/screening/run_screening_test.sh [python]
# Works in a scratch directory (the scripts use paths relative to the cwd).
set -euo pipefail
PY=${1:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$(mktemp -d); cd "$W"
cp -r "$ROOT/p" .; cp "$ROOT/screening_reduce.py" "$ROOT/screening_full.py" .
FAKE="$ROOT/tests/screening/fake_trainer.py"; chmod +x "$FAKE"
export FAKE_TRAINER_DELAY=0.03
echo "scratch dir: $W"

echo "== search ranges equal wann-cpp's for the shared hyperparameters"
$PY "$ROOT/tests/screening/check_ranges_vs_snn.py" | tail -2

echo "== run_all.sh (acrobot, disc_mc, car; tiny sizes, fake trainer)"
cp "$ROOT/run_all.sh" .
EXE="$FAKE" PYTHON="$PY" ROUNDS=1 N_REDUCE=6 N_FULL=4 TOP=2 SEEDS=2 JOBS_REDUCE=3 OMP_REDUCE=1 JOBS_FULL=2 OMP_FULL=1 \
    bash run_all.sh > run_all.txt
for t in acrobot disc_mc car; do
  test -s screening_reduce/$t/results.csv && test -s screening_full/$t/p3_summary.csv && echo "  $t: reduce + full outputs present"
done
rm -rf screening_reduce screening_full log

echo "== Phase 1: screening_reduce (2 rounds x 16 trials; slow fake trainer so the pruner gets to poll)"
FAKE_TRAINER_DELAY=0.4 $PY screening_reduce.py --task acrobot --exe "$FAKE" --rounds 2 --n 16 --jobs 4 --omp 1 --seed 3 > phase1.txt
grep -E "Round [0-9]+: [0-9]+ complete, [0-9]+ pruned|eliminated|Total trials" phase1.txt || true
test -s screening_reduce/acrobot/results.csv
test -s screening_reduce/acrobot/space_log.jsonl
test -s screening_reduce/acrobot/analysis.txt

echo "== the pruner must discard the WORSE trials (it polls fitPeak; nodeMed in the fake stats is anti-correlated with fitness)"
$PY - "$FAKE" <<'PYEOF'
import json, glob, re, sys, importlib.util, pandas as pd
spec = importlib.util.spec_from_file_location("fake_trainer", sys.argv[1]); ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
loss = lambda f: (lambda c: sum(((c[k] - ft.OPT[k]) / ft.SCALE[k]) ** 2 for k in ft.OPT))(json.load(open(f)))
done = {(int(r), int(t)) for r, t in pd.read_csv("screening_reduce/acrobot/results.csv")[["round", "trial"]].values}
good, bad = [], []
for f in glob.glob("log/reduce_acrobot/r*_t*_merged.json"):
    r, t = map(int, re.search(r"r(\d+)_t(\d+)", f).groups())
    (good if (r, t) in done else bad).append(loss(f))
print(f"completed trials: {len(good)} (mean loss {sum(good)/len(good):.3f}); pruned: {len(bad)} (mean loss {sum(bad)/max(1,len(bad)):.3f})")
assert bad, "the pruner never pruned anything: the stats polling loop is not exercised"
assert sum(bad)/len(bad) > sum(good)/len(good), "pruned trials are not worse than completed ones: the pruner reads the wrong column?"
m = json.load(open("log/reduce_acrobot/r00_t0000_merged.json"))      # fidelity actually sent to the trainer
assert m["alg_nVals"] == 6 and m["maxGen"] == 128 and m["popSize"] == 96 and m["alg_nReps"] == 2, m
PYEOF

echo "== resume: a second invocation reloads results and the narrowed space"
$PY screening_reduce.py --task acrobot --exe "$FAKE" --rounds 3 --n 8 --jobs 4 --omp 1 --seed 3 | grep -E "Resuming|Total trials"

echo "== Phase 2+3: screening_full"
$PY screening_full.py --task acrobot --exe "$FAKE" --mode both --n 10 --top 2 --seeds 2 --jobs 4 --omp 1 --seed 5 | tail -22
test -s screening_full/acrobot/p2_results.csv
test -s screening_full/acrobot/p3_summary.csv
test -s screening_full/acrobot/p3_validation.csv
test -s screening_full/acrobot/analysis.txt

echo "== analyse-only modes"
$PY screening_reduce.py --task acrobot --mode analyse | tail -3
$PY screening_full.py   --task acrobot --mode analyse | tail -3

echo "== every task: configs the scripts send are accepted by the C++ Hyperparams (fake trainer rejects unknown keys)"
for t in car acrobot disc_mc bipedal bottleneck; do
  $PY screening_reduce.py --task $t --exe "$FAKE" --rounds 1 --n 4 --jobs 2 --omp 1 --tag chk | grep -E "Round 0: [0-9]+ complete|Traceback|ERROR" || true
done

$PY - <<'PYEOF'
import json, glob, pandas as pd
m3 = json.load(open("log/full_p3_acrobot/rank00_seed00_merged.json"))
assert m3["save_mod"] == 10 and m3["early_stop_patience"] == 0, m3             # acrobot: early stop stays off
ov = json.load(open("screening_full/acrobot/p3_configs/rank00_seed00.json"))
assert "early_stop_patience" not in ov, ov                                      # ... because the script does not set it
assert set(ov) == {"alg_probMoo","prob_addConn","prob_addNode","prob_enable","prob_mutAct","prob_initEnable",
                   "select_cullRatio","select_eliteRatio","select_tournSize","save_mod"}, sorted(ov)
assert isinstance(ov["select_tournSize"], int), ov["select_tournSize"]          # ints stay ints in the JSON the trainer reads
car = json.load(open(glob.glob("log/full_p3_car/*_merged.json")[0])) if glob.glob("log/full_p3_car/*_merged.json") else None
p2 = pd.read_csv("screening_full/acrobot/p2_results.csv")
print("phase-2 best:", p2.nlargest(1, "peak_fitness").iloc[0].round(3).to_dict())
sp = [json.loads(l) for l in open("screening_reduce/acrobot/space_log.jsonl")]
print("space after last round:", {k: v[1:] for k, v in sp[-1]["space"].items()})
PYEOF
echo "OK ($W)"
