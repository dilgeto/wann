#pragma once
#include "Hyperparams.h"
#include "Ind.h"
#include <utility>
#include <vector>

namespace wann {

// Per-generation counters of the variation operators, filled by ask().
// Index order for the arrays: [addConn, addNode, enable, mutAct].
// `chosen` = times the roulette selected the operator; `applied` = times it
// actually changed the genome (a chosen mutation can be a no-op, e.g. addConn
// with no free (src,dst) pair, enable with no disabled connection).
// `mooConn` is 1 if the generation's selection used [meanFit, 1/nConn]
// (prob alg_probMoo), 0 if it used [meanFit, maxFit], -1 at gen 0 (no selection).
// Pure bookkeeping: it does not influence the search.
struct MutStats {
    int chosen[4]  = {0, 0, 0, 0};
    int applied[4] = {0, 0, 0, 0};
    int mooConn    = -1;
};

// Candidate destinations of mutAddConn, per source node (exposed for tests).
// Mirrors the layering of Python _variation.py::mutAddConn: nodes are sorted
// topologically and assigned to layers (inputs/bias 0, hidden 1.., outputs one
// past the deepest hidden layer); a connection may go from a node to a node of
// a strictly higher layer, unless a gene src->dst already exists (enabled or not).
// Returns, for every node in topological order, {source node id, destination
// node ids (ascending topological position)}. Empty on a cyclic genome.
std::vector<std::pair<int, std::vector<int>>>
addConnCandidates(const std::vector<NodeGene>& nodes,
                  const std::vector<ConnGene>& conns);

// -------------------------------------------------------------------------
// Wann – main evolutionary algorithm (ask / tell interface).
// Port of WANNRelease/WANN/wann_src (wann.py, _variation.py, _speciate.py):
// a single species, NSGA-II ranking, elitism + tournament selection, and one
// topological mutation per child.
//
// Usage:
//   Wann wann(hyp);
//   for (int gen = 0; gen < hyp.maxGen; ++gen) {
//       auto& pop = wann.ask();        // get current population
//       // evaluate each ind: reward[i][j] = fitness of pop[i] with weight j
//       wann.tell(reward);             // assign fitness
//   }
// -------------------------------------------------------------------------
class Wann {
public:
    explicit Wann(const Hyperparams& hyp);

    // Returns reference to the current population (after ask()).
    std::vector<Ind>& ask();

    // Assigns fitness from a 2-D reward matrix.
    // reward[i][j] = fitness of pop[i] evaluated with the j-th weight value.
    void tell(const std::vector<std::vector<double>>& reward);

    // Direct read access to population and generation counter.
    const std::vector<Ind>& population() const { return pop; }
    int generation() const { return gen; }

    // Operator counters for the population returned by the last ask().
    const MutStats& lastMutStats() const { return mutStats_; }

    // Current population (public for DataGatherer access).
    std::vector<Ind> pop;

private:
    Hyperparams p;
    std::vector<InnovRecord> innov;
    int gen = 0;
    MutStats mutStats_;

    // Minimal species container (WANN uses a single species).
    struct Species {
        int         seedIdx    = 0;    // index into pop
        std::vector<int> memberIdx;    // indices into pop
        int         nOffspring = 0;
    };
    std::vector<Species> species;

    void initPop();
    void probMoo();
    void speciate();
    void evolvePop();

    // Returns children produced from one species; updates innov.
    std::vector<Ind> recombine(const Species& sp);

    Ind    crossover   (const Ind& parentA, const Ind& parentB);
    // Returns {operator index 0..3 (MutStats order), whether it changed the genome}.
    std::pair<int,bool> topoMutate(Ind& child);
    // The mutation operators return true iff they changed the genome.
    bool   mutAddConn(std::vector<ConnGene>& conns,
                      const std::vector<NodeGene>& nodes);
    bool   mutAddNode(std::vector<ConnGene>& conns,
                      std::vector<NodeGene>& nodes);
};

} // namespace wann
