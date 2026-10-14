// Checks, without any training, that
//   1. AnnCarTask::CompiledNet::forward() is bit-identical to wann::act(setWeights(...)),
//      on topologies produced by genetic operators alone (random fitness, no environment);
//   2. evaluating a network on the car task is deterministic for a fixed seed and finite.
#include "../include/wann/AnnCarTask.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/Wann.h"

#include <cmath>
#include <iostream>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++failures; std::cerr << "FAIL: " << msg << '\n'; } } while (0)

int main() {
    wann::Hyperparams hyp;
    hyp.popSize = 64; hyp.alg_nReps = 2; hyp.car_episode_steps = 200;
    hyp.ann_nInput = 9; hyp.ann_nOutput = 2;
    wann::seedRng(1);

    wann::Wann alg(hyp);
    wann::AnnCarTask task(hyp);

    int nets = 0, withHidden = 0;
    for (int gen = 0; gen < 30; ++gen) {
        auto& pop = alg.ask();
        std::vector<std::vector<double>> reward(pop.size(), std::vector<double>(hyp.alg_nVals));
        for (auto& r : reward) for (auto& x : r) x = wann::randDouble(0.0, 100.0);  // no env involved

        for (const auto& ind : pop) {
            if (ind.wVec.empty()) continue;
            ++nets;
            if (ind.nNodes > 1 + hyp.ann_nInput + hyp.ann_nOutput) ++withHidden;
            auto net = wann::AnnCarTask::compile(ind.wVec, ind.aVec, hyp.ann_nInput, hyp.ann_nOutput);
            std::vector<double> node(net.nNodes), out(hyp.ann_nOutput);
            for (double w : task.weightVals()) {
                auto wMat = wann::setWeights(ind.wVec, w);
                for (int trial = 0; trial < 3; ++trial) {
                    std::vector<double> in(hyp.ann_nInput);
                    for (auto& v : in) v = wann::randDouble(-1.0, 1.0);
                    auto ref = wann::act(wMat, ind.aVec, hyp.ann_nInput, hyp.ann_nOutput, in);
                    net.forward(w, in.data(), node.data(), out.data());
                    for (int k = 0; k < hyp.ann_nOutput; ++k)
                        CHECK(ref[k] == out[k] || (std::isnan(ref[k]) && std::isnan(out[k])),
                              "gen " << gen << " w=" << w << " out[" << k << "]: " << ref[k] << " vs " << out[k]);
                }
            }
        }
        alg.tell(reward);
    }
    std::cout << "forward equivalence: " << nets << " networks (" << withHidden << " with hidden nodes)\n";
    CHECK(withHidden > 0, "no network with hidden nodes was generated; test is not exercising topologies");

    auto& pop = alg.ask();
    for (int i = 0; i < 3; ++i) {
        auto a = task.getDistFitness(pop[i].wVec, pop[i].aVec, 7);
        auto b = task.getDistFitness(pop[i].wVec, pop[i].aVec, 7);
        for (size_t k = 0; k < a.size(); ++k) {
            CHECK(a[k] == b[k], "non-deterministic evaluation");
            CHECK(std::isfinite(a[k]), "non-finite reward");
        }
    }

    std::cout << (failures ? "FAILED\n" : "OK\n");
    return failures ? 1 : 0;
}
