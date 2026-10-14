#pragma once
// env_replay_<task>: drives the ANN task's environment open-loop with the actions recorded in a
// trajectory CSV written by the SNN code (wann-cpp's wann_<task>_eval -S), using the same episode
// seed, and writes this repo's trajectory CSV. tests/env/compare_env.py then checks that
// observations and rewards agree step by step — i.e. that both repos simulate the same environment.
//
//   env_replay_<task> -d config.json -c snn_trajectory.csv -s episodeSeed -o ann_trajectory.csv
#include "../../include/wann/AnnTask.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace wann::replay {

struct Csv {
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows;
};

inline Csv readCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Csv c;
    std::string line;
    std::getline(f, line);
    { std::stringstream ss(line); std::string t; while (std::getline(ss, t, ',')) c.header.push_back(t); }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> r; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, ',')) r.push_back(std::stod(t));
        c.rows.push_back(std::move(r));
    }
    return c;
}

inline int column(const Csv& c, const std::string& name) {
    for (size_t i = 0; i < c.header.size(); ++i) if (c.header[i] == name) return static_cast<int>(i);
    throw std::runtime_error("column not found: " + name);
}

// One-hot of the discrete action a ∈ {-1, 0, 1, ...} (winner = a + 1), as an argmax network would output.
inline std::vector<double> oneHot(double action, int n) {
    std::vector<double> v(n, 0.0);
    v[static_cast<int>(std::lround(action)) + 1] = 1.0;
    return v;
}

// `tapeOf(csv)` returns the rows of network outputs that reproduce the recorded actions.
template <typename Task, typename TapeFn>
int run(int argc, char** argv, TapeFn tapeOf) {
    std::string cfg, in, out; long long seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-d" && i+1 < argc) cfg  = argv[++i];
        else if (a == "-c" && i+1 < argc) in   = argv[++i];
        else if (a == "-o" && i+1 < argc) out  = argv[++i];
        else if (a == "-s" && i+1 < argc) seed = std::atoll(argv[++i]);
    }
    if (cfg.empty() || in.empty() || out.empty()) {
        std::cerr << "usage: env_replay -d config.json -c snn_trajectory.csv -s episodeSeed -o out.csv\n";
        return 1;
    }
    try {
        Hyperparams hyp = loadHyp(cfg);
        Task task(hyp);
        std::vector<std::vector<double>> tape = tapeOf(readCsv(in));

        CompiledNet net;   // no connections: the tape provides the outputs
        net.nInput = hyp.ann_nInput; net.nOutput = hyp.ann_nOutput;
        net.nNodes = 1 + net.nInput + net.nOutput;
        net.rowStart.assign(net.nNodes + 1, 0);
        net.act.assign(net.nNodes, 1);
        net.tape = &tape;
        task.exportTrajectory(net, 1.0, seed, out);
        if (net.tapePos != tape.size())
            std::cerr << "note: environment consumed " << net.tapePos << " actions, the recording has " << tape.size() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}

} // namespace wann::replay
