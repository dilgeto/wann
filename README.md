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

## Car task (`wann_car`)

rl-tools `CarTrack`: 9 inputs (x, y, heading, vx, vy, yaw rate, 3 lidar distances) scaled to
[-1, 1] / [0, 1]; 2 outputs (throttle, steering) clamped to [-1, 1]; reward = vx per step.

```bash
OMP_NUM_THREADS=576 ./build/wann_car -d p/car_ann.json -o car -s 42   # train
./build/wann_car_eval -f log/car_best.out -n 20 -S best                # evaluate / replay CSV
./build/wann_car_eval -f log/car_best.out -a                           # reward per shared weight
```

Parallelism: one OpenMP task per (individual, shared weight value) pair, so a population of
480 with 6 weights gives 2880 tasks per generation.

Shared weights: `{-2, -1, -0.5, 0.5, 1, 2}` (`alg_wDist: "standard"`, `alg_nVals: 6`), otherwise
`linspace(-ann_absWCap, ann_absWCap, alg_nVals)`. Negative values are meaningful (the shared
weight keeps its sign), unlike in the SNN code where it is a magnitude.

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
the shared weight; forward pass is `AnnCarTask::CompiledNet` (CSR, no allocation per step).

## Adding another environment

Write `Ann<X>Task` next to `AnnCarTask` (same `compile` / `evaluateWeight` pattern) and its
`main_<x>.cpp`. rl-tools has non-inline symbols, so keep one environment per executable.
