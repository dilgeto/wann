// wann_car_eval – loads a trained network and evaluates N episodes.
//
//   ./wann_car_eval                                   (log/ann_car_best.out, best weight)
//   ./wann_car_eval -f log/ann_car_best.out -d p/car_ann.json -n 20 -s 0
//   ./wann_car_eval -w 1.0 -n 50                      (fixed shared weight)
//   ./wann_car_eval -n 20 -S best -o replay.csv       (save the best episode's trajectory)
//   ./wann_car_eval -a                                (table over all shared weights)

#include "../include/wann/AnnCarTask.h"
#include "../include/wann/Hyperparams.h"
#include "../include/wann/Ind.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string netFile    = "log/ann_car_best.out";
    std::string configFile = "p/car_ann.json";
    std::string overrides;
    std::string weightArg  = "best";
    std::string saveArg;
    std::string outFile;
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
            std::cerr << "Usage: wann_car_eval [-f net.out] [-d config.json] [-p overrides.json]"
                         " [-w weight|best] [-n episodes] [-s seed] [-t steps]"
                         " [-S best|N] [-o out.csv] [-a]\n";
            return 1;
        }
    }

    wann::Hyperparams hyp;
    std::unique_ptr<wann::AnnCarTask> taskPtr;
    std::vector<double> wVec;
    std::vector<int>    aVec;
    try {
        hyp = wann::loadHyp(configFile);
        if (!overrides.empty()) wann::updateHyp(hyp, overrides);
        taskPtr = std::make_unique<wann::AnnCarTask>(hyp);
        auto [w, a, wKey] = wann::importNet(netFile);
        wVec = std::move(w); aVec = std::move(a);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    wann::AnnCarTask& task = *taskPtr;
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
    auto rewards = task.evalEpisodes(wVec, aVec, weight, nEpisodes, seed);

    double mean = std::accumulate(rewards.begin(), rewards.end(), 0.0) / nEpisodes;
    double sq = 0; for (double r : rewards) sq += (r - mean) * (r - mean);
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < nEpisodes; ++i)
        std::cout << "  ep " << std::setw(3) << i << ": " << rewards[i] << '\n';
    std::cout << "\nMean=" << mean << "  StdDev=" << std::sqrt(sq / nEpisodes)
              << "  Min=" << *std::min_element(rewards.begin(), rewards.end())
              << "  Max=" << *std::max_element(rewards.begin(), rewards.end()) << '\n';

    if (!saveArg.empty()) {
        int ep = saveArg == "best"
            ? static_cast<int>(std::max_element(rewards.begin(), rewards.end()) - rewards.begin())
            : std::atoi(saveArg.c_str());
        if (outFile.empty())
            outFile = netFile.substr(0, netFile.rfind('.')) + "_ep" + std::to_string(ep) + "_replay.csv";
        task.exportTrajectory(wVec, aVec, weight, seed + ep, outFile);
        std::cout << "Replay saved to: " << outFile << '\n';
    }
    return 0;
}
