// Differential test against the original Python WANN (WANNRelease/WANN/wann_src).
// tests/fixtures/python_equiv.json is produced by tests/gen_python_fixtures.py from the
// real Python code; here the C++ port must reproduce it:
//   * genome expression (getNodeOrder: topological order, weight matrix incl. NaN, aVec, nConn)
//   * the set of (src, dst) pairs mutAddConn can create
//   * ANN forward pass for shared weights
//   * activation functions
//   * NSGA-II ranking
// Tie order in NSGA follows numpy's scalar introsort (fixtures are generated with SIMD
// sorts disabled). RNG-driven choices cannot be compared draw by draw (numpy vs mt19937);
// those are covered by test_operators.cpp (frequencies / invariants).
#include "../include/wann/AnnCarTask.h"
#include "../include/wann/CompiledNet.h"
#include "../include/wann/Ind.h"
#include "../include/wann/NsgaSort.h"
#include "../include/wann/Wann.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

using json = nlohmann::json;

static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; if (failures <= 25) std::cerr << "FAIL: " << msg << '\n'; } } while (0)

static double num(const json& j) { return j.is_string() ? std::nan("") : j.get<double>(); }
static bool close(double a, double b, double tol = 1e-9) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/fixtures/python_equiv.json";
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << '\n'; return 2; }
    json fx; f >> fx;

    // ---- activation functions ----
    for (const auto& a : fx["actfn"]) {
        int id = a["id"];
        for (size_t i = 0; i < a["x"].size(); ++i)
            CHECK(close(wann::applyAct(id, a["x"][i]), a["y"][i], 1e-12),
                  "applyAct(" << id << ", " << a["x"][i] << ") = " << wann::applyAct(id, a["x"][i])
                              << " vs python " << a["y"][i]);
    }

    // ---- genomes ----
    int gi = 0, withHidden = 0, withCand = 0;
    for (const auto& g : fx["genomes"]) {
        ++gi;
        std::vector<wann::NodeGene> nodes;
        for (const auto& n : g["nodes"]) nodes.push_back({int(n[0].get<double>()), int(n[1].get<double>()), int(n[2].get<double>())});
        std::vector<wann::ConnGene> conns;
        for (const auto& c : g["conns"]) conns.push_back({c[0], c[1], c[2], c[3], c[4].get<int>() != 0});

        wann::Ind ind(conns, nodes);
        const bool ok = ind.express();
        CHECK(ok, "genome " << gi << ": express() failed");
        if (!ok) continue;
        const int N = g["N"];
        if (N > 1 + int(g["nIn"]) + int(g["nOut"])) ++withHidden;

        CHECK(ind.nNodes == N, "genome " << gi << ": nNodes");
        CHECK(ind.nConn == g["nConn"].get<int>(), "genome " << gi << ": nConn " << ind.nConn << " vs " << g["nConn"]);
        for (int i = 0; i < N; ++i)
            CHECK(ind.aVec[i] == g["aVec"][i].get<int>(), "genome " << gi << ": aVec[" << i << "]");

        std::vector<double> expW(N * N, 0.0);
        for (const auto& e : g["wMat"]) expW[e[0].get<int>() * N + e[1].get<int>()] = num(e[2]);
        for (int i = 0; i < N * N; ++i)
            CHECK(close(ind.wMat[i], expW[i], 0.0), "genome " << gi << ": wMat[" << i / N << "," << i % N << "] "
                                                    << ind.wMat[i] << " vs " << expW[i]);

        // mutAddConn candidates
        std::map<int, std::vector<int>> exp, got;
        for (const auto& e : g["addConn"]) exp[e[0]] = e[1].get<std::vector<int>>();
        for (auto& [s, d] : wann::addConnCandidates(nodes, conns)) { std::sort(d.begin(), d.end()); got[s] = d; }
        CHECK(exp == got, "genome " << gi << ": mutAddConn candidates differ");
        for (auto& [s, d] : exp) if (!d.empty()) { ++withCand; break; }

        // forward pass: python task.setWeights + act
        const double Wv[6] = {-2, -1, -0.5, 0.5, 1, 2};
        const int nIn = g["nIn"], nOut = g["nOut"];
        auto net = wann::CompiledNet::compile(ind.wVec, ind.aVec, nIn, nOut);
        std::vector<double> node(N), out(nOut);
        for (const auto& a : g["act"]) {
            std::vector<double> in = a["in"].get<std::vector<double>>();
            for (int k = 0; k < 6; ++k) {
                auto ref = wann::act(wann::setWeights(ind.wVec, Wv[k]), ind.aVec, nIn, nOut, in);
                net.forward(Wv[k], in.data(), node.data(), out.data());
                for (int o = 0; o < nOut; ++o) {
                    double py = a["out"][k][o];
                    CHECK(close(ref[o], py, 1e-9), "genome " << gi << ": act w=" << Wv[k] << " out " << ref[o] << " vs python " << py);
                    CHECK(close(out[o], py, 1e-9), "genome " << gi << ": compiled net w=" << Wv[k] << " out " << out[o] << " vs python " << py);
                }
            }
        }
    }
    std::cout << "genomes: " << gi << " (" << withHidden << " with hidden nodes, " << withCand << " with addConn candidates)\n";
    CHECK(withHidden > 100, "fixtures contain too few genomes with hidden nodes");

    // ---- NSGA-II ----
    int nsgaCases = 0, largeTies = 0;
    for (const auto& c : fx["nsga"]) {
        std::vector<std::vector<double>> obj = c["obj"].get<std::vector<std::vector<double>>>();
        std::vector<int> exp = c["rank"].get<std::vector<int>>();
        ++nsgaCases;
        if (c["largeTies"].get<bool>()) ++largeTies;   // n > 16 with tied keys: needs numpy's exact introsort
        CHECK(wann::nsga_sort(obj) == exp, "nsga case with n=" << obj.size() << " (ties=" << c["ties"] << ") differs");
    }
    std::cout << "nsga: " << nsgaCases << " cases compared exactly (" << largeTies << " with ties and n > 16)\n";

    std::cout << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
