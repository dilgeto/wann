"""Generates tests/fixtures/python_equiv.json by running the ORIGINAL Python WANN
(WANNRelease/WANN/wann_src) so that tests/test_python_equiv.cpp can check the C++
port against it.

    python tests/gen_python_fixtures.py [path/to/WANNRelease/WANN]

Needs only numpy (gym is stubbed: the algorithm itself does not use it).
Covers the deterministic parts: genome expression (getNodeOrder), the candidate
set of mutAddConn (obtained by driving the real mutAddConn with patched numpy
RNG calls), forward pass, activation functions and NSGA-II ranking.
"""
import json, math, os, sys, types, warnings

# np.argsort's order for tied keys is implementation-defined, and numpy >= 2 swaps in
# SIMD sorts on some CPUs. The original code ran on numpy's scalar introsort, which
# the C++ port reproduces, so force it here (feature names differ across numpy versions).
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES",
                      "X86_V3 X86_V4 AVX2 AVX512F AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR")
import numpy as np
_probe = np.array([47434.97, 53443.61, np.inf, np.inf, 52152.47])
if list(np.argsort(-_probe)) != [2, 3, 1, 4, 0]:
    sys.exit("numpy is using a SIMD argsort (unstable for ties); could not force the scalar one")

warnings.filterwarnings("ignore")
WANN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/investigación/brain-tokyo-workshop/WANNRelease/WANN")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "python_equiv.json")

dom = types.ModuleType("domain"); dom.games = {}
mk = types.ModuleType("domain.make_env"); mk.make_env = lambda *a, **k: None
sys.modules["domain"] = dom; sys.modules["domain.make_env"] = mk
sys.path.insert(0, WANN_DIR)
from wann_src.wann import Wann
from wann_src.ind import getNodeOrder, act, applyAct
from wann_src.nsga_sort import nsga_sort


def num(x):
    x = float(x)
    return "nan" if math.isnan(x) else x


def py_candidates(w, nodeG, connG):
    """Candidate destinations of mutAddConn for every source, using the real
    Python mutAddConn with permutation/shuffle forced."""
    order, _ = getNodeOrder(nodeG, connG)
    ids = [int(i) for i in nodeG[0, order]]
    res = {}
    real_perm, real_shuf = np.random.permutation, np.random.shuffle
    try:
        for k, srcId in enumerate(ids):
            cands, j = [], 0
            while True:
                seen = {"L": 0}
                def perm(n, k=k):
                    return np.array([k] + [x for x in range(n) if x != k])
                def shuf(arr, j=j, seen=seen):
                    seen["L"] = len(arr)
                    if j < len(arr):
                        t = arr[0].copy(); arr[0] = arr[j]; arr[j] = t
                np.random.permutation, np.random.shuffle = perm, shuf
                out, _ = w.mutAddConn(connG.copy(), nodeG.copy(), w.innov.copy(), 0)
                if out.shape[1] == connG.shape[1] or int(out[1, -1]) != srcId:
                    break                      # this source has no free destination
                cands.append(int(out[2, -1]))
                j += 1
                if j >= seen["L"]:
                    break
            res[srcId] = sorted(cands)
    finally:
        np.random.permutation, np.random.shuffle = real_perm, real_shuf
    return [[s, res[s]] for s in ids]


def dump_ind(w, ind, rng):
    N = ind.node.shape[1]
    wMat = ind.wMat
    nz = [[i, j, num(wMat[i, j])] for i in range(N) for j in range(N)
          if wMat[i, j] != 0 or math.isnan(wMat[i, j])]
    nIn, nOut = int(ind.nInput), int(ind.nOutput)
    acts = []
    for _ in range(2):
        x = rng.uniform(-1, 1, nIn)
        row = []
        for wv in (-2, -1, -0.5, 0.5, 1, 2):
            cMat = np.copy(ind.wVec).reshape(N, N)           # task.setWeights
            cMat[np.isnan(cMat)] = 0
            cMat[cMat != 0] = 1.0
            out = act(cMat * wv, ind.aVec, nIn, nOut, x)
            row.append([float(v) for v in np.ravel(out)])
        acts.append({"in": x.tolist(), "out": row})
    return {
        "nodes": ind.node.T.tolist(),
        "conns": [[int(c[0]), int(c[1]), int(c[2]), float(c[3]), int(c[4])] for c in ind.conn.T],
        "N": N, "nIn": nIn, "nOut": nOut,
        "aVec": [int(a) for a in ind.aVec],
        "wMat": nz, "nConn": int(ind.nConn),
        "addConn": py_candidates(w, ind.node, ind.conn),
        "act": acts,
    }


def run_config(seed, hyp, gens, snap_every):
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1)
    w = Wann(hyp)
    genomes = []
    for g in range(gens):
        pop = w.ask()
        if g % snap_every == 0 or g == gens - 1:
            genomes += [dump_ind(w, ind, rng) for ind in pop]
        reward = rng.normal(size=(len(pop), 6)) * 10
        w.tell(reward)
    return genomes


base = dict(popSize=24, ann_nInput=9, ann_nOutput=2, ann_initAct=1, ann_actRange=list(range(1, 11)),
            prob_initEnable=0.5, prob_addConn=0.25, prob_addNode=0.25, prob_enable=0.1, prob_mutAct=0.4,
            prob_crossover=0.0, select_cullRatio=0.2, select_eliteRatio=0.2, select_tournSize=4,
            alg_probMoo=0.3)
genomes = []
genomes += run_config(1, dict(base), 40, 4)
genomes += run_config(2, dict(base, ann_nInput=3, ann_nOutput=1, ann_actRange=[1, 2, 3, 4, 5, 6],
                              prob_initEnable=0.3, prob_addNode=0.4, prob_addConn=0.3), 40, 4)

# --- NSGA-II ------------------------------------------------------------
rs = np.random.RandomState(7)
nsga = []
for case in range(300):
    n = int(rs.randint(2, 16)) if case < 150 else int(rs.randint(17, 90))
    ties = case < 100 or 150 <= case < 175          # ties only up to n <= 16 (stable in numpy) + a few large ones
    scale = rs.choice([1.0, 100.0, 0.01])
    obj = rs.normal(size=(n, 2)) * scale
    if case % 5 == 0:
        obj[:, 1] = 1.0 / rs.randint(1, 12, size=n)  # the 1/nConn objective
    if ties:
        obj = np.round(obj, 1 if scale >= 1 else 3)
    if case % 7 == 0:
        obj[rs.randint(0, n), 0] = 0.0
    nsga.append({"obj": obj.tolist(), "rank": [int(r) for r in nsga_sort(obj)],
                 "ties": bool(ties), "largeTies": bool(ties and n > 16)})

# --- activation functions ----------------------------------------------
xs = [-3.0, -1.5, -1.0, -0.5, -1e-9, 0.0, 1e-9, 0.25, 0.5, 1.0, 2.0, 4.0]
actfn = [{"id": a, "x": xs, "y": [float(applyAct(a, np.float64(x))) for x in xs]} for a in range(1, 13)]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"genomes": genomes, "nsga": nsga, "actfn": actfn}, f, separators=(",", ":"))
print("wrote", OUT, len(genomes), "genomes,", len(nsga), "nsga cases")
