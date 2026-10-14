#pragma once
// Command-line drivers shared by every environment: trainMain<Task> (evolution loop with
// OpenMP over (individual, weight value) pairs) and evalMain<Task> (evaluate a saved net).
// Header-only templates, so each environment's executable instantiates them with its own Task.
// rl-tools has non-inline symbols: link exactly one environment per executable.

#include "AnnTask.h"
#include "DataGatherer.h"
#include "Hyperparams.h"
#include "Ind.h"
#include "Random.h"
#include "Wann.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace wann {

struct TaskCli {
    const char* binary;         // e.g. "wann_car"
    const char* title;          // e.g. "ANN Car (CarTrack)"
    const char* defaultHyp;     // e.g. "p/car_ann.json"
    const char* defaultPrefix;  // e.g. "ann_car"
};

namespace detail {

constexpr int REPLAY_INTERVAL = 256;

// reward[i][wi] for the whole population. One OpenMP task per (individual, weight value).
inline std::vector<std::vector<double>>
evalPop(const std::vector<Ind>& pop, const AnnTask& task, int seed) {
    const int n  = static_cast<int>(pop.size());
    const int nW = task.numWeightVals();

    // Phase A: compile each network once (independent of the weight value).
    std::vector<CompiledNet> nets(n);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) nets[i] = task.compile(pop[i].wVec, pop[i].aVec);

    // Phase B: one task per (individual, weight value) so idle threads keep
    // finding work near the end of a generation.
    std::vector<std::vector<double>> reward(n, std::vector<double>(nW, 0.0));
    const int total = n * nW;
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < total; ++idx) {
        const int i = idx / nW, wi = idx % nW;
        reward[i][wi] = task.evaluateWeight(nets[i], wi, seed * 10000 + i);
    }
    return reward;
}

} // namespace detail

// ------------------------------------------------------------------------------------
// wann_<task> [-d default.json] [-p overrides.json] [-o prefix] [-s seed]
// ------------------------------------------------------------------------------------
template <typename Task>
int trainMain(int argc, char* argv[], const TaskCli& cli) {
    namespace fs = std::filesystem;
    std::string defaultHyp = cli.defaultHyp, overrideHyp, outPrefix = cli.defaultPrefix;
    uint32_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-d" && i+1 < argc) defaultHyp  = argv[++i];
        else if (arg == "-p" && i+1 < argc) overrideHyp = argv[++i];
        else if (arg == "-o" && i+1 < argc) outPrefix   = argv[++i];
        else if (arg == "-s" && i+1 < argc) seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        else {
            std::cerr << "Usage: " << cli.binary << " [-d default.json] [-p overrides.json]"
                         " [-o prefix] [-s seed]\n";
            return 1;
        }
    }

    Hyperparams hyp;
    try {
        hyp = loadHyp(defaultHyp);
        if (!overrideHyp.empty()) updateHyp(hyp, overrideHyp);
    } catch (const std::exception& e) {
        std::cerr << "Error loading hyperparameters: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Task: " << cli.title
              << "  nInput="  << hyp.ann_nInput << "  nOutput=" << hyp.ann_nOutput
              << "  popSize=" << hyp.popSize     << "  maxGen="  << hyp.maxGen << '\n';

    seedRng(seed);
    fs::create_directories("log");
    const std::string replayDir = "log/" + outPrefix + "_replay";
    fs::create_directories(replayDir);

    std::unique_ptr<Task> taskPtr;
    try { taskPtr = std::make_unique<Task>(hyp); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    const Task& task = *taskPtr;

    Wann         alg(hyp);
    DataGatherer data(outPrefix, hyp);

    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();

    double bestFitnessSoFar = -std::numeric_limits<double>::infinity();
    int    gensSinceImprove = 0;
    int    lastGen          = hyp.maxGen - 1;
    bool   stoppedEarly     = false;

    for (int gen = 0; gen < hyp.maxGen; ++gen) {
        auto& pop = alg.ask();
        const int evalSeed = static_cast<int>(seed) + gen;
        auto reward = detail::evalPop(pop, task, evalSeed);
        alg.tell(reward);

        const int eliteIdx = static_cast<int>(
            std::max_element(pop.begin(), pop.end(),
                [](const Ind& a, const Ind& b){ return a.fitness < b.fitness; }) - pop.begin());
        const auto& rw = reward[eliteIdx];
        const int bestWi = static_cast<int>(std::max_element(rw.begin(), rw.end()) - rw.begin());

        data.setBestWi(bestWi);
        data.gatherData(pop);
        data.gatherMutStats(alg.lastMutStats());
        std::cout << gen << "\t - \t" << data.display() << '\n';

        bool earlyStop = false;
        if (hyp.early_stop_patience > 0) {
            if (pop[eliteIdx].fitness > bestFitnessSoFar) { bestFitnessSoFar = pop[eliteIdx].fitness; gensSinceImprove = 0; }
            else                                          { ++gensSinceImprove; }
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
        if ((gen % detail::REPLAY_INTERVAL == 0 || gen == hyp.maxGen - 1 || earlyStop)
                && !pop[eliteIdx].wVec.empty()) {
            std::ostringstream fname;
            fname << replayDir << "/gen_" << std::setw(4) << std::setfill('0') << gen << ".csv";
            try {
                const long long ep = AnnTask::episodeSeed(evalSeed * 10000 + eliteIdx, bestWi, 0);
                task.exportTrajectory(pop[eliteIdx].wVec, pop[eliteIdx].aVec,
                                      task.weightVals()[bestWi], ep, fname.str());
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not save replay gen " << gen << ": " << e.what() << '\n';
            }
        }

        if (earlyStop) break;
    }

    const int    gensRun = lastGen + 1;
    const double total_s = std::chrono::duration<double>(Clock::now() - t_start).count();

    std::ofstream tlog("log/" + outPrefix + "_time.log");
    tlog << std::fixed << std::setprecision(3)
         << "total_s           " << total_s           << '\n'
         << "per_gen_s         " << total_s / gensRun << '\n'
         << "maxGen            " << hyp.maxGen        << '\n'
         << "gensRun           " << gensRun           << '\n'
         << "popSize           " << hyp.popSize       << '\n'
         << "early_stopped     " << (stoppedEarly ? "true" : "false") << '\n';
    std::cout << "Time: " << total_s << " s  (" << total_s / gensRun << " s/gen)\n";

    data.save();
    std::cout << "Done. Results written to log/" << outPrefix << "_*\n";
    return 0;
}

// ------------------------------------------------------------------------------------
// wann_<task>_eval [-f net.out] [-d config.json] [-p overrides.json] [-w weight|best]
//                  [-n episodes] [-s seed] [-t steps] [-S best|N] [-o out.csv] [-a]
// ------------------------------------------------------------------------------------
template <typename Task>
int evalMain(int argc, char* argv[], const TaskCli& cli) {
    std::string netFile = std::string("log/") + cli.defaultPrefix + "_best.out";
    std::string configFile = cli.defaultHyp, overrides, weightArg = "best", saveArg, outFile;
    int  nEpisodes = 10, seed = 0, maxSteps = 0;
    bool allWeights = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-f" && i+1 < argc) netFile    = argv[++i];
        else if (arg == "-d" && i+1 < argc) configFile = argv[++i];
        else if (arg == "-p" && i+1 < argc) overrides  = argv[++i];
        else if (arg == "-w" && i+1 < argc) weightArg  = argv[++i];
        else if (arg == "-n" && i+1 < argc) nEpisodes  = std::atoi(argv[++i]);
        else if (arg == "-s" && i+1 < argc) seed       = std::atoi(argv[++i]);
        else if (arg == "-t" && i+1 < argc) maxSteps   = std::atoi(argv[++i]);
        else if (arg == "-S" && i+1 < argc) saveArg    = argv[++i];
        else if (arg == "-o" && i+1 < argc) outFile    = argv[++i];
        else if (arg == "-a")               allWeights = true;
        else {
            std::cerr << "Usage: " << cli.binary << "_eval [-f net.out] [-d config.json] [-p overrides.json]"
                         " [-w weight|best] [-n episodes] [-s seed] [-t steps]"
                         " [-S best|N] [-o out.csv] [-a]\n";
            return 1;
        }
    }

    Hyperparams hyp;
    std::unique_ptr<Task> taskPtr;
    std::vector<double> wVec;
    std::vector<int>    aVec;
    try {
        hyp = loadHyp(configFile);
        if (!overrides.empty()) updateHyp(hyp, overrides);
        taskPtr = std::make_unique<Task>(hyp);
        auto [w, a, wKey] = importNet(netFile);
        wVec = std::move(w); aVec = std::move(a);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    Task& task = *taskPtr;
    if (maxSteps > 0) task.setEpisodeSteps(maxSteps);
    const auto& W = task.weightVals();

    if (allWeights) {
        auto r = task.getDistFitness(wVec, aVec, seed);
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < W.size(); ++i)
            std::cout << "  w=" << std::setw(5) << W[i] << "  reward=" << r[i] << '\n';
        return 0;
    }

    int    wi = 0;
    double weight = W[0];
    if (weightArg == "best") {
        std::string wiFile = netFile.substr(0, netFile.rfind('.')) + ".wi";
        std::ifstream wif(wiFile);
        if (wif && (wif >> wi) && wi >= 0 && wi < static_cast<int>(W.size())) {
            std::cout << "Weight read from " << wiFile << ": " << W[wi] << "  (wi=" << wi << ")\n";
        } else {
            auto r = task.getDistFitness(wVec, aVec, seed);
            wi = static_cast<int>(std::max_element(r.begin(), r.end()) - r.begin());
            std::cout << "Best weight: " << W[wi] << "  (reward=" << r[wi] << ")\n";
        }
        weight = W[wi];
    } else {
        weight = std::stod(weightArg);
        std::cout << "Fixed weight: " << weight << '\n';
    }

    std::cout << "Evaluating " << nEpisodes << " episodes (base seed=" << seed << ")...\n";
    auto [shaped, original] = task.evalEpisodes(wVec, aVec, weight, nEpisodes, seed);

    auto summary = [&](const char* label, const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / nEpisodes;
        double sq = 0; for (double r : v) sq += (r - mean) * (r - mean);
        std::cout << "  [" << label << "] Mean=" << mean << "  StdDev=" << std::sqrt(sq / nEpisodes)
                  << "  Min=" << *std::min_element(v.begin(), v.end())
                  << "  Max=" << *std::max_element(v.begin(), v.end()) << '\n';
    };
    const bool hasShaping = (shaped != original);
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < nEpisodes; ++i) {
        std::cout << "  ep " << std::setw(3) << i << ": " << shaped[i];
        if (hasShaping) std::cout << "  (environment return " << original[i] << ")";
        std::cout << '\n';
    }
    std::cout << "\nSummary (" << nEpisodes << " episodes):\n";
    summary(hasShaping ? "fitness (shaped)" : "return", shaped);
    if (hasShaping) summary("environment return", original);

    if (!saveArg.empty()) {
        int ep = saveArg == "best"
            ? static_cast<int>(std::max_element(shaped.begin(), shaped.end()) - shaped.begin())
            : std::atoi(saveArg.c_str());
        if (outFile.empty())
            outFile = netFile.substr(0, netFile.rfind('.')) + "_ep" + std::to_string(ep) + "_replay.csv";
        task.exportTrajectory(wVec, aVec, weight, seed + ep, outFile);
        std::cout << "Replay saved to: " << outFile << '\n';
    }
    return 0;
}

} // namespace wann
