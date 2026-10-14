// wann_car – WANN evolution of classical ANNs + rl-tools CarTrack.
//
// Usage:
//   ./wann_car [-d p/car_ann.json] [-p overrides.json] [-o prefix] [-s seed]
//
// Parallelism: OpenMP over (individual, weight-value) pairs. Set OMP_NUM_THREADS
// to control the number of threads.

#include "../include/wann/AnnCarTask.h"
#include "../include/wann/DataGatherer.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"
#include "../include/wann/Random.h"
#include "../include/wann/Wann.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int REPLAY_INTERVAL = 256;

static std::vector<std::vector<double>>
evalPop(const std::vector<wann::Ind>& pop, const wann::AnnCarTask& task,
        int nInput, int nOutput, int seed)
{
    const int n  = static_cast<int>(pop.size());
    const int nW = task.numWeightVals();

    // Phase A: compile each network once (independent of the weight value).
    std::vector<wann::AnnCarTask::CompiledNet> nets(n);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
        nets[i] = wann::AnnCarTask::compile(pop[i].wVec, pop[i].aVec, nInput, nOutput);

    // Phase B: one task per (individual, weight value) so idle threads keep
    // finding work near the end of a generation.
    std::vector<std::vector<double>> reward(n, std::vector<double>(nW, 0.0));
    const int total = n * nW;
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < total; ++idx) {
        const int i  = idx / nW;
        const int wi = idx % nW;
        reward[i][wi] = task.evaluateWeight(nets[i], wi, seed * 10000 + i);
    }
    return reward;
}

int main(int argc, char* argv[]) {
    std::string defaultHyp = "p/car_ann.json";
    std::string overrideHyp;
    std::string outPrefix  = "ann_car";
    uint32_t    seed       = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-d" && i+1 < argc) { defaultHyp  = argv[++i]; }
        else if (arg == "-p" && i+1 < argc) { overrideHyp = argv[++i]; }
        else if (arg == "-o" && i+1 < argc) { outPrefix   = argv[++i]; }
        else if (arg == "-s" && i+1 < argc) { seed = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else {
            std::cerr << "Usage: wann_car [-d default.json] [-p overrides.json]"
                         " [-o prefix] [-s seed]\n";
            return 1;
        }
    }

    wann::Hyperparams hyp;
    try {
        hyp = wann::loadHyp(defaultHyp);
        if (!overrideHyp.empty()) wann::updateHyp(hyp, overrideHyp);
    } catch (const std::exception& e) {
        std::cerr << "Error loading hyperparameters: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Task: ANN Car (CarTrack)"
              << "  nInput="  << hyp.ann_nInput
              << "  nOutput=" << hyp.ann_nOutput
              << "  popSize=" << hyp.popSize
              << "  maxGen="  << hyp.maxGen << '\n';

    wann::seedRng(seed);
    fs::create_directories("log");
    const std::string replayDir = "log/" + outPrefix + "_replay";
    fs::create_directories(replayDir);

    std::unique_ptr<wann::AnnCarTask> taskPtr;
    try { taskPtr = std::make_unique<wann::AnnCarTask>(hyp); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    const wann::AnnCarTask& task = *taskPtr;

    wann::Wann         alg(hyp);
    wann::DataGatherer data(outPrefix, hyp);

    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    double bestFitnessSoFar = -std::numeric_limits<double>::infinity();
    int    gensSinceImprove = 0;
    int    lastGen          = hyp.maxGen - 1;
    bool   stoppedEarly     = false;

    for (int gen = 0; gen < hyp.maxGen; ++gen) {
        auto& pop    = alg.ask();
        const int evalSeed = static_cast<int>(seed) + gen;
        auto  reward = evalPop(pop, task, hyp.ann_nInput, hyp.ann_nOutput, evalSeed);
        alg.tell(reward);

        int eliteIdx = static_cast<int>(
            std::max_element(pop.begin(), pop.end(),
                [](const wann::Ind& a, const wann::Ind& b){ return a.fitness < b.fitness; })
            - pop.begin());
        const auto& rw = reward[eliteIdx];
        int bestWi = static_cast<int>(std::max_element(rw.begin(), rw.end()) - rw.begin());

        data.setBestWi(bestWi);
        data.gatherData(pop);
        data.gatherMutStats(alg.lastMutStats());
        std::cout << gen << "\t - \t" << data.display() << '\n';

        bool earlyStop = false;
        if (hyp.early_stop_patience > 0) {
            double eliteFitness = pop[eliteIdx].fitness;
            if (eliteFitness > bestFitnessSoFar) { bestFitnessSoFar = eliteFitness; gensSinceImprove = 0; }
            else                                 { ++gensSinceImprove; }
            if (gensSinceImprove >= hyp.early_stop_patience) {
                earlyStop = stoppedEarly = true;
                lastGen = gen;
                std::cout << "Early stopping: no new fitTop in " << hyp.early_stop_patience
                          << " generations (gen " << gen << "/" << hyp.maxGen - 1 << ")\n";
            }
        }

        if (gen % hyp.save_mod == 0 || earlyStop) {
            data.save(gen);
            data.savePareto(pop, gen);
        }

        // Elite trajectory for replay. Seed matches what evalPop() gave this individual.
        if ((gen % REPLAY_INTERVAL == 0 || gen == hyp.maxGen - 1 || earlyStop)
                && !pop[eliteIdx].wVec.empty()) {
            std::ostringstream fname;
            fname << replayDir << "/gen_" << std::setw(4) << std::setfill('0') << gen << ".csv";
            try {
                long long ep = wann::AnnCarTask::episodeSeed(evalSeed * 10000 + eliteIdx, bestWi, 0);
                task.exportTrajectory(pop[eliteIdx].wVec, pop[eliteIdx].aVec,
                                      task.weightVals()[bestWi], ep, fname.str());
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not save replay gen " << gen << ": " << e.what() << '\n';
            }
        }

        if (earlyStop) break;
    }

    int    gensRun = lastGen + 1;
    double total_s = std::chrono::duration<double>(Clock::now() - t_start).count();

    std::ofstream tlog("log/" + outPrefix + "_time.log");
    tlog << std::fixed << std::setprecision(3)
         << "total_s           " << total_s            << '\n'
         << "per_gen_s         " << total_s / gensRun  << '\n'
         << "maxGen            " << hyp.maxGen         << '\n'
         << "gensRun           " << gensRun            << '\n'
         << "popSize           " << hyp.popSize        << '\n'
         << "early_stopped     " << (stoppedEarly ? "true" : "false") << '\n';
    std::cout << "Time: " << total_s << " s  (" << total_s / gensRun << " s/gen)\n";

    data.save();
    std::cout << "Done. Results written to log/" << outPrefix << "_*\n";
    return 0;
}
