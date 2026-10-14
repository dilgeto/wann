// Statistical / structural checks of the variation operators (the RNG-driven part that
// test_python_equiv cannot compare draw by draw). No environment is involved: fitness is random.
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/Wann.h"

#include <cmath>
#include <iostream>
#include <set>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++failures; std::cerr << "FAIL: " << msg << '\n'; } } while (0)

static bool sameGenome(const wann::Ind& a, const wann::Ind& b) {
    if (a.nodes.size() != b.nodes.size() || a.conns.size() != b.conns.size()) return false;
    for (size_t i = 0; i < a.nodes.size(); ++i)
        if (a.nodes[i].id != b.nodes[i].id || a.nodes[i].type != b.nodes[i].type
            || a.nodes[i].activation != b.nodes[i].activation) return false;
    for (size_t i = 0; i < a.conns.size(); ++i)
        if (a.conns[i].innov != b.conns[i].innov || a.conns[i].src != b.conns[i].src
            || a.conns[i].dst != b.conns[i].dst || a.conns[i].enabled != b.conns[i].enabled
            || a.conns[i].weight != b.conns[i].weight) return false;
    return true;
}

int main() {
    wann::Hyperparams hyp;
    hyp.popSize = 200; hyp.ann_nInput = 9; hyp.ann_nOutput = 2; hyp.ann_initAct = 1;
    hyp.ann_actRange = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    hyp.prob_addConn = 0.25; hyp.prob_addNode = 0.15; hyp.prob_enable = 0.10; hyp.prob_mutAct = 0.50;
    hyp.prob_initEnable = 0.3;
    hyp.select_cullRatio = 0.2; hyp.select_eliteRatio = 0.2; hyp.select_tournSize = 8;
    wann::seedRng(123);
    wann::Wann alg(hyp);

    // ---- generation 0 ----
    auto& pop0 = alg.ask();
    CHECK(static_cast<int>(pop0.size()) == hyp.popSize, "gen 0 population size");
    long enabled = 0, total = 0;
    for (const auto& ind : pop0) {
        CHECK(ind.nodes.size() == 1 + 9 + 2, "gen 0 node count");
        for (size_t k = 0; k < ind.conns.size(); ++k) {
            const auto& c = ind.conns[k];
            // Python order: destination outer, source (bias, inputs) inner
            CHECK(c.innov == int(k) && c.src == int(k % 10) && c.dst == 10 + int(k / 10), "gen 0 gene order, gene " << k);
            CHECK(c.weight == 1.0, "gen 0 weight");
            enabled += c.enabled; ++total;
        }
    }
    double fe = double(enabled) / total;
    CHECK(std::abs(fe - 0.3) < 4 * std::sqrt(0.3 * 0.7 / total), "gen 0 prob_initEnable: " << fe);

    // ---- evolve with random fitness ----
    long chosen[4] = {0, 0, 0, 0}, nChosen = 0;
    long mooConn = 0, mooGens = 0;
    int genomesWithHidden = 0;
    const int nElites = int(std::floor(std::floor(hyp.popSize * (1 - hyp.select_cullRatio)) * hyp.select_eliteRatio));
    for (int g = 1; g <= 60; ++g) {
        std::vector<wann::Ind> prev = alg.population();
        std::vector<std::vector<double>> reward(prev.size(), std::vector<double>(6));
        for (auto& r : reward) for (auto& x : r) x = wann::randDouble(-50, 50);
        alg.tell(reward);

        auto& pop = alg.ask();
        CHECK(static_cast<int>(pop.size()) == hyp.popSize, "population size at gen " << g);

        // exactly nElites children are unchanged copies of parents (the elites), the rest are
        // mutated: count how many children equal some previous genome.
        int identical = 0;
        for (const auto& c : pop) for (const auto& q : prev) if (sameGenome(c, q)) { ++identical; break; }
        CHECK(identical >= nElites, "gen " << g << ": fewer unchanged copies (" << identical << ") than elites (" << nElites << ")");

        const auto& ms = alg.lastMutStats();
        for (int k = 0; k < 4; ++k) { chosen[k] += ms.chosen[k]; nChosen += ms.chosen[k]; }
        CHECK(ms.mooConn == 0 || ms.mooConn == 1, "mooConn flag");
        mooConn += ms.mooConn; ++mooGens;

        for (const auto& ind : pop) {
            CHECK(!ind.wMat.empty(), "individual not expressed");
            // mutAct only touches hidden nodes: bias, inputs and outputs keep ann_initAct
            std::set<int> ids;
            for (const auto& n : ind.nodes) {
                CHECK(ids.insert(n.id).second, "duplicate node id");
                if (n.type != 3) CHECK(n.activation == hyp.ann_initAct, "non-hidden node activation changed");
                else CHECK(n.activation >= 1 && n.activation <= 10, "hidden activation out of range");
            }
            // no duplicated (src,dst) gene, no connection into bias/inputs, innovation ids unique
            std::set<std::pair<int,int>> pairs; std::set<int> innovs;
            for (const auto& c : ind.conns) {
                CHECK(pairs.insert({c.src, c.dst}).second, "duplicate connection gene");
                CHECK(innovs.insert(c.innov).second, "duplicate innovation id");
                for (const auto& n : ind.nodes) if (n.id == c.dst) CHECK(n.type == 2 || n.type == 3, "connection into input/bias");
            }
            if (ind.nodes.size() > 12) ++genomesWithHidden;
        }
    }

    // roulette frequencies: chosen[k] / total vs prob_k / sum
    const double pr[4] = {hyp.prob_addConn, hyp.prob_addNode, hyp.prob_enable, hyp.prob_mutAct};
    const double sum = pr[0] + pr[1] + pr[2] + pr[3];
    for (int k = 0; k < 4; ++k) {
        double e = pr[k] / sum, f = double(chosen[k]) / nChosen;
        double tol = 4 * std::sqrt(e * (1 - e) / nChosen);
        CHECK(std::abs(f - e) < tol, "operator " << k << " frequency " << f << " vs expected " << e);
        std::cout << "operator " << k << ": " << f << " (expected " << e << ")\n";
    }
    // alg_probMoo = 0.8 default: fraction of generations ranked on [mean, 1/nConn]
    double fm = double(mooConn) / mooGens;
    std::cout << "probMoo conn objective: " << fm << " of generations (alg_probMoo=" << hyp.alg_probMoo << ")\n";
    CHECK(std::abs(fm - hyp.alg_probMoo) < 4 * std::sqrt(hyp.alg_probMoo * (1 - hyp.alg_probMoo) / mooGens) + 0.01,
          "probMoo frequency " << fm);
    CHECK(genomesWithHidden > 0, "no genome ever grew a hidden node");

    std::cout << (failures ? "FAILED\n" : "OK\n");
    return failures ? 1 : 0;
}
