#!/usr/bin/env python3
"""Compare a trajectory CSV recorded by the SNN code (wann-cpp) with the one this repo's environment
produces when driven with the same actions and the same episode seed (env_replay_<task>).
Both files have the same column layout; header names are ignored for bottleneck (they differ).

usage: compare_env.py snn.csv ann.csv [--tol 1e-4]
Exit status 0 if same number of steps and every value agrees within tol.
"""
import csv, sys

def read(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        return header, [[float(x) for x in row] for row in r if row]

def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    tol = 1e-4
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1]); a = [x for x in a if x != str(tol) and x != sys.argv[sys.argv.index("--tol") + 1]]
    hs, rs = read(a[0]); hn, rn = read(a[1])
    ok = True
    print(f"steps: snn={len(rs)} ann={len(rn)}")
    if len(rs) != len(rn): ok = False
    if len(hs) != len(hn): print(f"column count differs: {len(hs)} vs {len(hn)}"); return 1
    n = min(len(rs), len(rn))
    worst = {}
    first_bad = None
    for t in range(n):
        for c in range(len(hs)):
            d = abs(rs[t][c] - rn[t][c])
            if d > worst.get(c, (0, 0))[0]: worst[c] = (d, t)
            if d > tol and first_bad is None: first_bad = (t, hs[c], rs[t][c], rn[t][c])
    bad = [(hs[c], d, t) for c, (d, t) in worst.items() if d > tol]
    top = sorted(worst.items(), key=lambda kv: -kv[1][0])[:3]
    print("max |diff| over all columns/steps: " + ", ".join(f"{hs[c]}={d:.2e}@step{t}" for c, (d, t) in top))
    # how much did the recorded controller actually do? (an environment test with a constant action proves little)
    act_cols = [i for i, h in enumerate(hs) if h.startswith(("a", "throttle", "steering", "action")) and "obs" not in h and h != "step"]
    distinct = {hs[i]: len({round(r[i], 4) for r in rs}) for i in act_cols[:6]}
    print("distinct values per action column:", distinct)
    if first_bad: print("first disagreement (step, column, snn, ann):", first_bad)
    if bad: ok = False
    print("SAME ENVIRONMENT" if ok else "DIFFERENT")
    return 0 if ok else 1

sys.exit(main())
