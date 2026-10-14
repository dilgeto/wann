// Port of WANNRelease/WANN/wann_src/{wann.py,_variation.py,_speciate.py}.
// Random draws come from std::mt19937 instead of numpy's generator, so a given
// seed does not give the same run as Python; the operators and their
// probabilities are the same.
#include "../include/wann/Wann.h"
#include "../include/wann/NsgaSort.h"
#include "../include/wann/Random.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace wann {

namespace {

// Python _variation.py::listXor(b, c): elements that b and c do not share.
// Used with b = [current activation], c = ann_actRange.
std::vector<int> listXor(const std::vector<int>& b, const std::vector<int>& c) {
    std::vector<int> out;
    auto contains = [](const std::vector<int>& v, int x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };
    for (int a : b) if (!contains(b, a) || !contains(c, a)) out.push_back(a);
    for (int a : c) if (!contains(b, a) || !contains(c, a)) out.push_back(a);
    return out;
}

struct NodeKey { int id; double layer; };

// nodeKey of Python mutAddConn: node ids in topological order with their layer.
// Empty on a cyclic genome.
std::vector<NodeKey> layerNodes(const std::vector<NodeGene>& nodes,
                                const std::vector<ConnGene>& conns)
{
    const int nNodes = static_cast<int>(nodes.size());
    int nIns = 0, nOuts = 0;
    for (const auto& n : nodes) {
        if (n.type == 1 || n.type == 4) ++nIns;
        if (n.type == 2)                ++nOuts;
    }

    auto [order, wMat] = getNodeOrder(nodes, conns);
    if (order.empty()) return {};

    // hMat = wMat[nIns:-nOuts, nIns:-nOuts]
    const int nHidden = nNodes - nIns - nOuts;
    std::vector<double> hMat(nHidden * nHidden, 0.0);
    for (int i = 0; i < nHidden; ++i)
        for (int j = 0; j < nHidden; ++j)
            hMat[i * nHidden + j] = wMat[(nIns + i) * nNodes + (nIns + j)];

    // Python: hLay = getLayer(hMat) + 1; lastLayer = max(hLay) + 1 (or 1 if no hidden).
    std::vector<double> hLay = getLayer(hMat, nHidden);
    for (double& l : hLay) l += 1.0;
    double lastLayer = 1.0;
    if (!hLay.empty()) lastLayer = *std::max_element(hLay.begin(), hLay.end()) + 1.0;

    std::vector<NodeKey> key(nNodes);
    for (int i = 0; i < nIns; ++i)     key[i]                 = {nodes[order[i]].id, 0.0};
    for (int i = 0; i < nHidden; ++i)  key[nIns + i]          = {nodes[order[nIns + i]].id, hLay[i]};
    for (int i = 0; i < nOuts; ++i)    key[nIns + nHidden + i] = {nodes[order[nIns + nHidden + i]].id, lastLayer};
    return key;
}

// Indices into `key` that node `srcIdx` may still connect to.
std::vector<int> candidatesFor(const std::vector<NodeKey>& key,
                               const std::vector<ConnGene>& conns, int srcIdx)
{
    const int srcId = key[srcIdx].id;
    std::unordered_set<int> existing;  // ids of destinations that already have a gene from src
    for (const auto& c : conns)
        if (c.src == srcId) existing.insert(c.dst);

    std::vector<int> dest;
    for (int k = 0; k < static_cast<int>(key.size()); ++k)
        if (key[k].layer > key[srcIdx].layer && !existing.count(key[k].id))
            dest.push_back(k);
    return dest;
}

} // namespace

std::vector<std::pair<int, std::vector<int>>>
addConnCandidates(const std::vector<NodeGene>& nodes, const std::vector<ConnGene>& conns)
{
    std::vector<std::pair<int, std::vector<int>>> out;
    auto key = layerNodes(nodes, conns);
    for (int s = 0; s < static_cast<int>(key.size()); ++s) {
        std::vector<int> ids;
        for (int k : candidatesFor(key, conns, s)) ids.push_back(key[k].id);
        out.push_back({key[s].id, std::move(ids)});
    }
    return out;
}

// =========================================================================
// Construction / ask / tell
// =========================================================================

Wann::Wann(const Hyperparams& hyp) : p(hyp) {}

std::vector<Ind>& Wann::ask() {
    mutStats_ = MutStats{};
    if (pop.empty()) {
        initPop();
    } else {
        probMoo();
        speciate();
        evolvePop();
    }
    ++gen;
    return pop;
}

void Wann::tell(const std::vector<std::vector<double>>& reward) {
    for (int i = 0; i < static_cast<int>(reward.size()); ++i) {
        double sum = 0.0, best = -std::numeric_limits<double>::infinity();
        for (double r : reward[i]) {
            sum += r;
            best = std::max(best, r);
        }
        pop[i].fitness = sum / static_cast<double>(reward[i].size());
        pop[i].fitMax  = best;
        // nConn is set by express() and not overwritten here.
    }
}

// =========================================================================
// initPop (Python Wann.initPop)
// =========================================================================
void Wann::initPop() {
    const int nIn  = p.ann_nInput;
    const int nOut = p.ann_nOutput;

    // Nodes: id 0 = bias (type 4), 1..nIn = inputs (type 1), then outputs (type 2).
    std::vector<NodeGene> nodes;
    nodes.push_back({0, 4, p.ann_initAct});
    for (int i = 1; i <= nIn;          ++i) nodes.push_back({i, 1, p.ann_initAct});
    for (int i = nIn + 1; i <= nIn + nOut; ++i) nodes.push_back({i, 2, p.ann_initAct});

    // Connections: every (bias + input) -> every output. Same gene order as
    // Python: ins = [0..nIn] tiled once per output, outs repeated per input,
    // i.e. destination in the outer loop and source in the inner one.
    std::vector<ConnGene> base;
    int innovIdx = 0;
    for (int dst = nIn + 1; dst <= nIn + nOut; ++dst)
        for (int src = 0; src <= nIn; ++src)
            base.push_back({innovIdx++, src, dst, 1.0, true});

    // Population: weight value does not matter (shared weight); each connection
    // is enabled independently with probability prob_initEnable.
    pop.clear();
    pop.reserve(p.popSize);
    for (int i = 0; i < p.popSize; ++i) {
        std::vector<ConnGene> conns = base;
        for (auto& c : conns) c.enabled = randDouble() < p.prob_initEnable;
        Ind ind(conns, nodes);
        ind.express();
        ind.birth = 0;
        pop.push_back(std::move(ind));
    }

    // Innovation record: [innov, src, dst, newNode = -1, gen = 0].
    innov.clear();
    for (const auto& c : base) innov.push_back({c.innov, c.src, c.dst, -1, 0});
}

// =========================================================================
// probMoo – rank population via Pareto dominance (NSGA-II).
// With probability alg_probMoo the objectives are [meanFit, 1/nConn],
// otherwise [meanFit, maxFit].
// =========================================================================
void Wann::probMoo() {
    const int n = static_cast<int>(pop.size());

    // Python: if p['alg_probMoo'] < rand(): [mean, max] else [mean, 1/nConn]
    const bool useMooConn = !(p.alg_probMoo < randDouble());
    mutStats_.mooConn = useMooConn ? 1 : 0;

    std::vector<std::vector<double>> objVals(n, std::vector<double>(2));
    for (int i = 0; i < n; ++i) {
        int nc = std::max(pop[i].nConn, 1);  // no conns is always pareto optimal (but boring)
        objVals[i][0] = pop[i].fitness;
        objVals[i][1] = useMooConn ? 1.0 / static_cast<double>(nc) : pop[i].fitMax;
    }

    auto rank = nsga_sort(objVals);
    for (int i = 0; i < n; ++i) pop[i].rank = rank[i];
}

// =========================================================================
// speciate – WANN does not use species: the whole population is one species.
// =========================================================================
void Wann::speciate() {
    species.resize(1);
    species[0].seedIdx    = 0;
    species[0].nOffspring = p.popSize;
    species[0].memberIdx.resize(pop.size());
    std::iota(species[0].memberIdx.begin(), species[0].memberIdx.end(), 0);
    for (auto& ind : pop) ind.species = 0;
}

void Wann::evolvePop() {
    std::vector<Ind> newPop;
    newPop.reserve(p.popSize);
    for (const auto& sp : species) {
        auto children = recombine(sp);
        for (auto& c : children) newPop.push_back(std::move(c));
    }
    pop = std::move(newPop);
}

// =========================================================================
// recombine (Python _variation.py::recombine)
//   sort by rank -> cull the worst -> keep elites unchanged ->
//   tournament selection -> crossover or copy -> one topological mutation
// =========================================================================
std::vector<Ind> Wann::recombine(const Species& sp) {
    std::vector<Ind*> members;
    members.reserve(sp.memberIdx.size());
    for (int idx : sp.memberIdx) members.push_back(&pop[idx]);

    std::stable_sort(members.begin(), members.end(),
                     [](const Ind* a, const Ind* b) { return a->rank < b->rank; });

    int nOffspring = sp.nOffspring;

    // Cull: eliminate worst individuals from the breeding pool.
    int toCull = static_cast<int>(std::floor(p.select_cullRatio
                                             * static_cast<double>(members.size())));
    if (toCull > 0) members.resize(members.size() - toCull);
    const int poolSize = static_cast<int>(members.size());

    // Elitism: the best individuals pass unchanged (count taken after the cull).
    std::vector<Ind> children;
    children.reserve(nOffspring);
    int nElites = static_cast<int>(std::floor(static_cast<double>(poolSize)
                                              * p.select_eliteRatio));
    for (int i = 0; i < nElites; ++i) {
        children.push_back(*members[i]);
        --nOffspring;
    }

    // Tournament: the winner is the smallest index among select_tournSize
    // random picks (members are sorted by rank).
    auto tournament = [&]() -> int {
        int best = randInt(0, poolSize - 1);
        for (int t = 1; t < p.select_tournSize; ++t)
            best = std::min(best, randInt(0, poolSize - 1));
        return best;
    };

    for (int i = 0; i < nOffspring; ++i) {
        int pa = tournament();
        int pb = tournament();
        if (pa > pb) std::swap(pa, pb);  // pa = more fit (lower index)

        Ind child;
        if (!(randDouble() <= p.prob_crossover)) {
            // Mutation only: take only the fitter parent.
            child = Ind(members[pa]->conns, members[pa]->nodes);
        } else {
            child = crossover(*members[pa], *members[pb]);
        }

        topoMutate(child);
        child.express();
        children.push_back(std::move(child));
    }
    return children;
}

// =========================================================================
// crossover (Python _variation.py::crossover)
// Nodes and connections come from the fitter parent; for each connection
// gene both parents share (same innovation number) the weight is taken from
// parent B with probability 0.5.
// =========================================================================
Ind Wann::crossover(const Ind& parentA, const Ind& parentB) {
    Ind child(parentA.conns, parentA.nodes);
    for (auto& ca : child.conns) {
        for (const auto& cb : parentB.conns) {
            if (cb.innov == ca.innov) {
                if (randDouble() < 0.5) ca.weight = cb.weight;
                break;
            }
        }
    }
    return child;
}

// =========================================================================
// mutAddConn (Python _variation.py::mutAddConn)
// =========================================================================
bool Wann::mutAddConn(std::vector<ConnGene>& conns,
                      const std::vector<NodeGene>& nodes)
{
    auto key = layerNodes(nodes, conns);
    if (key.empty()) return false;

    // Sources in random order; the first with a free destination wins.
    std::vector<int> sources(key.size());
    std::iota(sources.begin(), sources.end(), 0);
    wann::shuffle(sources);

    for (int srcIdx : sources) {
        std::vector<int> dest = candidatesFor(key, conns, srcIdx);
        if (dest.empty()) continue;

        wann::shuffle(dest);
        const int srcId = key[srcIdx].id;
        const int dstId = key[dest[0]].id;

        const int newInnov = innov.back().innov + 1;
        conns.push_back({newInnov, srcId, dstId, 1.0, true});
        innov.push_back({newInnov, srcId, dstId, -1, gen});
        return true;
    }
    return false;
}

// =========================================================================
// mutAddNode (Python _variation.py::mutAddNode)
// =========================================================================
bool Wann::mutAddNode(std::vector<ConnGene>& conns,
                      std::vector<NodeGene>& nodes)
{
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) active.push_back(i);
    if (active.empty()) return false;

    const int nextInnov = innov.back().innov + 1;
    const int connSplit = active[randInt(0, static_cast<int>(active.size()) - 1)];

    const auto& actRange = p.ann_actRange;
    const int newActivation = actRange[randInt(0, static_cast<int>(actRange.size()) - 1)];

    // New node id = max destination id in the innovation record + 1.
    int newNodeId = 0;
    for (const auto& ir : innov) newNodeId = std::max(newNodeId, ir.dst);
    ++newNodeId;

    // To the new node the weight is 1; from it, the original weight (minimises
    // the disruption of adding the node).
    ConnGene connTo = conns[connSplit];
    connTo.innov  = nextInnov;
    connTo.dst    = newNodeId;
    connTo.weight = 1.0;

    ConnGene connFrom = conns[connSplit];
    connFrom.innov = nextInnov + 1;
    connFrom.src   = newNodeId;

    conns[connSplit].enabled = false;

    innov.push_back({nextInnov,     connTo.src,   connTo.dst,   newNodeId, gen});
    innov.push_back({nextInnov + 1, connFrom.src, connFrom.dst, -1,        gen});

    nodes.push_back({newNodeId, 3, newActivation});
    conns.push_back(connTo);
    conns.push_back(connFrom);
    return true;
}

// =========================================================================
// topoMutate (Python _variation.py::topoMutate): exactly one topological change.
// Roulette over [addConn, addNode, enable, mutAct].
// =========================================================================
std::pair<int,bool> Wann::topoMutate(Ind& child) {
    auto& conns = child.conns;
    auto& nodes = child.nodes;

    const double roulette[4] = {p.prob_addConn, p.prob_addNode, p.prob_enable, p.prob_mutAct};
    const double total = roulette[0] + roulette[1] + roulette[2] + roulette[3];
    const double spin  = randDouble() * total;

    int choice = 4;
    double slot = roulette[0];
    for (int i = 1; i < 4; ++i) {
        if (spin < slot) { choice = i; break; }
        slot += roulette[i];
    }

    bool applied = false;
    switch (choice) {
        case 1:  // Add connection
            applied = mutAddConn(conns, nodes);
            break;

        case 2:  // Add node
            applied = mutAddNode(conns, nodes);
            break;

        case 3: {  // Enable a disabled connection
            std::vector<int> disabled;
            for (int i = 0; i < static_cast<int>(conns.size()); ++i)
                if (!conns[i].enabled) disabled.push_back(i);
            if (!disabled.empty()) {
                conns[disabled[randInt(0, static_cast<int>(disabled.size()) - 1)]].enabled = true;
                applied = true;
            }
            break;
        }

        case 4: {  // Mutate the activation of a hidden node
            // Hidden nodes are the ones after bias, inputs and outputs.
            const int start = 1 + child.nInput + child.nOutput;
            const int end   = static_cast<int>(nodes.size());
            if (start != end) {
                const int mutNode = randInt(start, end - 1);
                std::vector<int> pool = listXor({nodes[mutNode].activation}, p.ann_actRange);
                if (!pool.empty()) {
                    nodes[mutNode].activation = pool[randInt(0, static_cast<int>(pool.size()) - 1)];
                    applied = true;
                }
            }
            break;
        }

        default: break;
    }

    ++mutStats_.chosen[choice - 1];
    if (applied) ++mutStats_.applied[choice - 1];

    child.birth = gen;
    return {choice - 1, applied};
}

} // namespace wann
