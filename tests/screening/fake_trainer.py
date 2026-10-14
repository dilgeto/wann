#!/usr/bin/env python3
"""Stand-in for ./build/wann_<task> used to test screening_reduce.py / screening_full.py WITHOUT training.

Same CLI as the real trainer (-d base.json [-p overrides.json] -o prefix -s seed). It
  * merges the JSON files and rejects any key the real C++ Hyperparams does not know (read from
    include/wann/Hyperparams.h), so a misspelled hyperparameter in the screening scripts fails here;
  * writes log/<prefix>_stats.out progressively, in the real 8-column format, with a synthetic
    fitness that peaks at a known point of the search space (OPT below) plus seed-dependent noise;
  * dumps the merged configuration to log/<prefix>_merged.json for the test to inspect.
If screening works, the tuned values should drift towards OPT.
"""
import json, math, os, random, re, sys, time
from pathlib import Path

OPT = {"alg_probMoo": 0.50, "prob_addConn": 0.30, "prob_addNode": 0.20, "prob_enable": 0.10,
       "prob_mutAct": 0.40, "prob_initEnable": 0.50, "select_cullRatio": 0.20,
       "select_eliteRatio": 0.15, "select_tournSize": 6}
SCALE = {"alg_probMoo": 0.9, "prob_addConn": 0.45, "prob_addNode": 0.35, "prob_enable": 0.24,
         "prob_mutAct": 0.6, "prob_initEnable": 0.6, "select_cullRatio": 0.45,
         "select_eliteRatio": 0.35, "select_tournSize": 14}

def known_keys(hyp_header: Path) -> set:
    return set(re.findall(r'get\(p\.\w+,\s*"(\w+)"\)', hyp_header.read_text()))

def main():
    a = sys.argv[1:]
    base = override = prefix = None; seed = 0
    i = 0
    while i < len(a):
        if   a[i] == "-d": base = a[i+1]
        elif a[i] == "-p": override = a[i+1]
        elif a[i] == "-o": prefix = a[i+1]
        elif a[i] == "-s": seed = int(a[i+1])
        else: sys.exit(f"fake_trainer: bad arg {a[i]}")
        i += 2
    cfg = json.load(open(base))
    if override:
        cfg.update(json.loads(override) if override.lstrip().startswith("{") else json.load(open(override)))
    keys = known_keys(Path(__file__).resolve().parents[2] / "include/wann/Hyperparams.h")
    unknown = sorted(set(cfg) - keys)
    if unknown:
        sys.exit(f"fake_trainer: unknown hyperparameter(s) for the C++ trainer: {unknown}")

    rnd = random.Random(seed)
    loss = sum(((cfg[k] - OPT[k]) / SCALE[k]) ** 2 for k in OPT if k in cfg)
    final = 300.0 - 400.0 * loss + rnd.gauss(0, 3.0)

    out = Path("log") / (prefix + "_stats.out"); out.parent.mkdir(parents=True, exist_ok=True)
    Path("log", prefix + "_merged.json").write_text(json.dumps(cfg, indent=1))
    maxGen, save_mod = int(cfg["maxGen"]), int(cfg["save_mod"])
    n = max(1, maxGen // save_mod)
    delay = float(os.environ.get("FAKE_TRAINER_DELAY", "0.05"))
    rows = []
    for k in range(1, n + 1):
        frac = 1 - math.exp(-3.0 * k / n)
        peak = final * frac + rnd.gauss(0, 1.0)
        top = peak - 5.0
        # nodeMed is deliberately ANTI-correlated with fitness: if a screening script ever reads the
        # wrong column (the stats file has 8 columns; a 7-name read shifts them by one) its pruner
        # would then keep the worst trials, which test_screening checks against.
        node_med = 100.0 - final / 3.0
        rows.append([k * save_mod * cfg["popSize"], top - 20, peak, top, peak, node_med, 9, top - 2])
        out.write_text("\n".join(",".join(f"{v:.6f}" for v in r) for r in rows) + "\n")
        time.sleep(delay)
    Path("log", prefix + "_best.out").write_text("0\n"); Path("log", prefix + "_best.wi").write_text("0\n")

if __name__ == "__main__":
    main()
