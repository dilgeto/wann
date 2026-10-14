import random, sys
# dense-ish random topology: bias+inputs -> hidden -> outputs and direct input->output edges; sign = polarity
def make(path, nIn, nOut, nHid, seed, pdirect=0.6, phid=0.5):
    random.seed(seed)
    N = 1 + nIn + nOut + nHid
    # node order: bias, inputs, hidden, outputs  (topological, as expressed by wann)
    ins = list(range(0, 1 + nIn)); hid = list(range(1 + nIn, 1 + nIn + nHid)); outs = list(range(1 + nIn + nHid, N))
    rows = [[0.0] * N + [1.0] for _ in range(N)]
    sgn = lambda: 1.0 if random.random() < 0.7 else -1.0
    for o in outs:
        rows[0][o] = 1.0   # bias drives every output
        for i in ins[1:]:
            if random.random() < pdirect: rows[i][o] = sgn()
        for h in hid:
            if random.random() < phid: rows[h][o] = sgn()
    for h in hid:
        for i in ins:
            if random.random() < phid: rows[i][h] = sgn()
    for r in range(N): rows[r][N] = float(random.choice([1, 2, 3, 4, 5, 6]))
    open(path, "w").write("\n".join(",".join(f"{v:e}" for v in r) for r in rows) + "\n")
if __name__ == "__main__":
    path, nIn, nOut, nHid, seed = sys.argv[1], *map(int, sys.argv[2:])
    make(path, nIn, nOut, nHid, seed)
