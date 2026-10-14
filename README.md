# wann

Weight Agnostic Neural Networks (NEAT-style topology search with a single shared
weight) for **classical feed-forward ANNs**, in C++17 + OpenMP, on
[rl-tools](https://github.com/rl-tools/rl-tools) environments.

Derived from `../wann-cpp` (the SNN version): the evolutionary core (`Wann`, `Ind`,
`NsgaSort`, `DataGatherer`) is copied from there; everything SNN-specific is gone, and the
fitness function evaluates a plain ANN forward pass instead of an SNN simulation.

## Build

```bash
./build.sh            # Release -> build/
ctest --test-dir build
```

Needs nlohmann_json (system package, or fetched), OpenMP, and an rl-tools checkout. By default
`../snn-simulator/external/rl-tools/include`; override with `-DRL_TOOLS_INCLUDE_DIR=...`.

## Tasks

One executable per environment (rl-tools has non-inline symbols, so two environments cannot share a
binary). Each has a trainer `wann_<task>` and an evaluator `wann_<task>_eval`; hyperparameters in `p/<task>_ann.json`.

| Task | Binary | In → out | Action | Fitness (what evolves) |
|---|---|---|---|---|
| Racing car (`CarTrack`) | `wann_car` | 9 → 2 | throttle, steering ∈ [-1,1] | Σ vx |
| Acrobot (torque ±1) | `wann_acrobot` | 6 → 3 | argmax → torque −1/0/+1 (or 6 → 1: continuous) | env return + `reward_shaping_scale` · Δ(tip height) |
| Mountain Car, discrete | `wann_disc_mc` | 2 → 3 | argmax → push left / none / right | −1/step + `reward_shaping_scale` · Δsin(3x) |
| BipedalWalker (Hardcore by default) | `wann_bipedal` | 24 → 4 | 4 torques ∈ [-1,1] | env return |
| Bottleneck (5 agents) | `wann_bottleneck` | 12 → 2 | linear / angular acceleration ∈ [-1,1] | env return + `reward_shaping_scale` · Δφ − `bottleneck_reverse_penalty_scale` · reverse |

```bash
OMP_NUM_THREADS=576 ./build/wann_acrobot -d p/acrobot_ann.json -o acro -s 42    # train
./build/wann_acrobot_eval -f log/acro_best.out -d p/acrobot_ann.json -n 20 -S best   # evaluate / replay CSV
./build/wann_acrobot_eval -f log/acro_best.out -d p/acrobot_ann.json -a              # reward per shared weight
```
Use the same `-d` config for training and evaluation (it defines the task geometry).

Parallelism: one OpenMP task per (individual, shared weight value) pair.

Notes per task:
- **Inputs.** Each observation channel is mapped to [-1, 1] over the same range the SNN code uses for its
  encoders (there mapped to [0, 1]) and clamped; the car uses value/bound. `ann_nInput` may be smaller than
  the full observation (the first channels are used).
- **Discrete actions** use `argmax` with the first maximum winning ties (as `np.argmax`). A network whose
  outputs are all equal (e.g. no connections) therefore always picks action 0 — unlike the SNN code, where
  "no spikes" means the neutral action.
- **Shaping.** `reward_shaping_scale` etc. are the same terms as wann-cpp and are on in the provided configs
  (acrobot 10, mountain car 10, bottleneck 2 + reverse penalty 0.02); set them to 0 for the plain return.
  `wann_*_eval` prints both the shaped fitness and the environment return when they differ.
- **Bottleneck.** All five agents share one network (parameter sharing); a feed-forward ANN has no state, so the
  same compiled net is just evaluated once per agent.
- **BipedalWalker** is not in rl-tools: it is the Box2D-free port in `../snn-simulator/include`
  (`-DBIPEDAL_INCLUDE_DIR=...` to change). `episode_steps` can only shorten the environment's own limit.
- Shared weights: `{-2, -1, -0.5, 0.5, 1, 2}` (`alg_wDist: "standard"`, `alg_nVals: 6`), otherwise
  `linspace(-ann_absWCap, ann_absWCap, alg_nVals)`. Negative values are meaningful (the shared weight keeps
  its sign), unlike in the SNN code where it is a magnitude.

## Hyperparameter screening (Optuna)

Port of wann-cpp's `screening_reduce.py` / `screening_full.py` (same algorithm, same CLI and output layout), driving
the `wann_<task>` trainers of this repo. Needs `pip install optuna numpy pandas scipy scikit-learn`.

```bash
python screening_reduce.py --task car --rounds 4 --n 30 --jobs 8 --omp 64          # Phase 1: narrow the search space
python screening_full.py   --task car --mode both --n 20 --top 3 --seeds 11 --jobs 3 --omp 190   # Phase 2 + 3
bash run_all.sh                      # acrobot, disc_mc and car in sequence; no arguments (budget / parallelism are set at the top of the file)
```
Tasks: `car acrobot disc_mc bipedal bottleneck`. Results land in `screening_reduce/<task>/` (`results.csv`,
`space_log.jsonl`, `analysis.txt`) and `screening_full/<task>/` (`p2_results.csv`, `p3_summary.csv`,
`p3_configs/rank<NN>_seed<NN>.json`, ...); the trainers' logs in `log/reduce_<task>/`, `log/full_p2_<task>/`,
`log/full_p3_<task>/`. `--mode analyse` re-reads them without running anything; `--tag X` keeps a run apart.

- **Phase 1** runs TPE trials of a short training (`TASK_DEFAULTS[..]["fidelity"]`), prunes with `MedianPruner` on `fitPeak`
  (polled from the stats file while it trains) and, after each round, drops the bins of every hyperparameter that are
  significantly worse (one-sided Mann-Whitney, α = 0.05), until nothing is eliminated.
- **Phase 2** repeats TPE with the full budget of `p/<task>_ann.json` inside the reduced space; **Phase 3** re-runs the top-K
  configurations with several seeds.
- **Searched** (9): `alg_probMoo prob_addConn prob_addNode prob_enable prob_mutAct prob_initEnable select_cullRatio
  select_eliteRatio select_tournSize`, with exactly the SNN screening's ranges (checked by the test against
  `../wann-cpp/screening_reduce.py`). Note `alg_probMoo` stops at 0.70, so the original WANN's 0.80 is outside the search.
  Edit `INITIAL_SPACE` (comment a line out to freeze a parameter) or pass `--space-override file.json`.
- **Left out vs. the SNN screening**: `prob_toggleExcitatory` and the SNN microparameters (`snn_window_ms`, `snn_tau_*`,
  `snn_ttfs_threshold`) do not exist here; the encoder/decoder combinations are gone (one ANN per task, so `run_key` is just
  the task); and there is **no automatic evaluation + bootstrap step** after Phase 3 — that one compares the SNN against ANN/PPO
  baselines, and this repo is the ANN side. Phase-3 models (`log/full_p3_<task>/rank<NN>_seed<NN>_best.out`, `.wi`) are in the
  format `wann_<task>_eval` reads.
- **Same protocol as the SNN screening**: `alg_nVals` is 6 in every fidelity (the SNN tasks ignore `alg_nVals`, they always use
  6 shared weights); `--early-stop-patience` defaults to 100 for `car` and 0 for the rest (in wann-cpp it only exists in the car
  trainer; here every trainer supports it, so pass a value to use it elsewhere). `bottleneck` has no SNN counterpart in the screening;
  its budget is chosen like acrobot's.
- Trial seeds are `seed + round*100000 + trial` as in the original; note that re-running the script (resume) restarts the round
  counter, so trials of the resumed run reuse seeds and file names `r00_t…` of the first one (same as wann-cpp).

Tested without training: `tests/screening/run_screening_test.sh [python]` runs both scripts end to end against
`tests/screening/fake_trainer.py`, a stand-in for the C++ trainer with the same CLI that rejects any hyperparameter the real
`Hyperparams` does not know and writes a synthetic `_stats.out` peaking at a known optimum (the tuned values move towards it;
resume, phase 2/3, analysis and all five tasks are exercised).

## Same environments as the SNN code (wann-cpp)

The point of this repo is to compare ANN-WANN with SNN-WANN, so the five environments must be the same.
That is checked, not assumed:

```bash
tests/env/check_against_snn.sh [../wann-cpp] [build]
```
For each task it builds a fixed random network, lets wann-cpp's prebuilt `wann_<task>_eval` run an episode with
it (spiking controller, a given episode seed) and record the trajectory, replays the SNN's recorded actions
open-loop in this repo's environment with the same seed (`build/env_replay_<task>`), and compares observations,
rewards and episode length step by step (`tests/env/compare_env.py`). Evaluation only; wann-cpp is just run.

Result: car, acrobot, mountain car, bottleneck and bipedal give **bit-identical** trajectories (difference exactly 0,
same number of steps) on 3 episode seeds each, with controllers that actually vary their actions (e.g. 24 distinct
throttle/steering values over a 161-step car episode, all three actions in acrobot and mountain car, 4 independent
torques in bipedal). This covers initial-state / terrain sampling, dynamics, observations, rewards (incl. shaping
terms), termination and step limits.

What makes it hold, and what to keep aligned:
- **Compile flags.** `-march=native` lets the compiler fuse multiply-adds, which changes results in the last bits;
  BipedalWalker's contact physics amplifies that (episodes diverged after ~50-90 steps). So `-march=native` is OFF by
  default here, as in wann-cpp (plain `-O3`). `-DWANN_MARCH_NATIVE=ON` is faster but breaks bit-identity.
- **Seeds.** Episode seeds follow wann-cpp's scheme (`gen seed * 10000 + individual`, then `* 10000 + weightIdx * 100 + rep`),
  so with the same `-s` both repos train on the same sequence of initial conditions / terrains.
- **CSV conventions** match wann-cpp's trajectory files (for acrobot and mountain car the `reward` column is the shaped step
  reward and mountain car's `action` is the winning index), so the existing replay scripts work on both.
- **Evaluation protocol** (`alg_nReps`, `popSize`, `maxGen`, shaping scales, step limits) is the same as in `p/*_snn.json`.
  What still differs between the ANN and SNN configs is the search itself: operator probabilities (`prob_addConn/addNode`),
  `alg_probMoo`, `ann_actRange` (1-10 classical activations vs 1-6 neuron types) and `save_mod` — the ANN ones follow the Python
  WANN. The shared weights also differ by design (signed `{-2..2}` here, positive magnitudes `{0.5..8}` there).
- Input *encoding* is the one intended difference: the SNN spike/current encoders vs ANN values in [-1, 1].

## Fidelity to the original Python WANN

The evolutionary core is a port of `WANNRelease/WANN/wann_src` (`wann.py`, `_variation.py`,
`_speciate.py`, `nsga_sort.py`, `ind.py`), checked against it by running the real Python code:

```bash
python tests/gen_python_fixtures.py     # regenerates tests/fixtures/python_equiv.json (needs numpy)
ctest --test-dir build                  # python_equivalence, operators, forward_equivalence
```

`python_equivalence` compares, on 528 genomes produced by the Python algorithm and 300 NSGA cases:
genome expression (topological order, weight matrix incl. NaN, activations, `nConn`), the exact set of
connections `mutAddConn` may create, activation functions, the forward pass, and NSGA-II ranks.
`operators` checks what depends on the RNG (roulette frequencies, `alg_probMoo`, gene order of the
initial population, invariants of the mutations).

Operators and probabilities are the same as Python: roulette over `[addConn, addNode, enable, mutAct]`
(`mutAct` only on hidden nodes, new activation chosen as `listXor`), one topological change per child,
`probMoo` alternating `[mean, 1/nConn]` / `[mean, max]`, cull, elitism (counted after the cull), tournament
selection, weight-only crossover. Compared with wann-cpp this removes: per-connection excitatory/inhibitory
polarity, random immigrants, the output-majority guard, and the lineage bookkeeping.

Things that are *not* identical, and cannot be:
- **RNG.** numpy's generator vs `std::mt19937`: the same seed does not give the same run. Draw counts per
  operation differ too, so only distributions are comparable (tested), not trajectories.
- **Tie order in NSGA.** The extreme points of a front all have infinite crowding distance and tie, so the
  order of `np.argsort` on ties decides the best ranks. It is implementation-defined; numpy >= 2 uses SIMD
  sorts on some CPUs (different order). The port reproduces numpy's scalar introsort (what the original
  ran on); fixtures are generated with the SIMD sorts disabled. Python on a modern CPU will break ties differently.
- **Floating point.** `np.dot` sums in BLAS order, the port sums sequentially (agreement ~1e-9 in tests).
- `Wann.gen` is never incremented in the Python (so `birth` is always 0); here it counts generations.
  It only labels innovations and individuals.
- `wann_car` does not re-evaluate new best individuals with many seeds (`checkBest` in `wann_train.py`),
  which only affects which individual is reported as best, not the search.

Deliberate differences from the SNN code base (wann-cpp): no SNN dependency; `setWeights` keeps the sign of
the shared weight; forward pass is `CompiledNet` (CSR, no allocation per step). The tests only exercise the
algorithm core and the car task; the other four environments were checked by hand (hand-built networks, e.g. a "push along the velocity" policy solves Mountain Car), not against Python,
since the original has no such tasks.

## Adding another environment

Derive `Ann<X>Task` from `AnnTask` (implement `runEpisode` and `trajectoryHeader`, see `AnnDiscMCTask` for the
smallest one), add `main_<x>.cpp` / `main_eval_<x>.cpp` calling `trainMain<>` / `evalMain<>` from `Driver.h`, and one
`add_ann_task(...)` line in `CMakeLists.txt`.
