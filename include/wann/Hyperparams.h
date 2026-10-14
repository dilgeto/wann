#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace wann {

struct Hyperparams {
    // --- algorithm ---
    std::string task          = "ann_car";
    std::string alg_wDist     = "standard";   // "standard" | "linspace"
    int    alg_nVals          = 6;
    int    alg_nReps          = 4;
    double alg_probMoo        = 0.80;
    int    maxGen             = 2048;
    int    popSize            = 128;

    // --- mutation probabilities ---
    double prob_crossover          = 0.0;
    double prob_mutAct             = 0.50;
    double prob_addNode            = 0.25;
    double prob_addConn            = 0.20;
    double prob_enable             = 0.05;
    double prob_initEnable         = 0.5;

    // --- selection ---
    double select_cullRatio   = 0.2;
    double select_eliteRatio  = 0.2;
    int    select_tournSize   = 8;

    // --- I/O ---
    int    save_mod           = 8;
    int    bestReps           = 20;

    // --- early stopping ---
    // Generations without a new fitTop record (running-best elite fitness)
    // before stopping early. 0 = disabled (run the full maxGen).
    int    early_stop_patience = 0;

    // --- network / task ---
    int    ann_nInput         = 9;
    int    ann_nOutput        = 2;
    int    ann_initAct        = 1;
    std::vector<int> ann_actRange = {1,2,3,4,5,6,7,8,9,10};
    // alg_wDist == "linspace": shared weights are linspace(-absWCap, absWCap, alg_nVals).
    double ann_absWCap        = 2.0;

    // --- car task ---
    int    car_episode_steps  = 1000;   // max env steps per episode
};

namespace detail {

inline void applyJson(Hyperparams& p, const nlohmann::json& j) {
    auto get = [&](auto& field, const char* key) {
        if (j.contains(key))
            field = j.at(key).get<std::remove_reference_t<decltype(field)>>();
    };
    get(p.task,                   "task");
    get(p.alg_wDist,              "alg_wDist");
    get(p.alg_nVals,              "alg_nVals");
    get(p.alg_nReps,              "alg_nReps");
    get(p.alg_probMoo,            "alg_probMoo");
    get(p.maxGen,                 "maxGen");
    get(p.popSize,                "popSize");
    get(p.prob_crossover,         "prob_crossover");
    get(p.prob_mutAct,            "prob_mutAct");
    get(p.prob_addNode,           "prob_addNode");
    get(p.prob_addConn,           "prob_addConn");
    get(p.prob_enable,            "prob_enable");
    get(p.prob_initEnable,        "prob_initEnable");
    get(p.select_cullRatio,       "select_cullRatio");
    get(p.select_eliteRatio,      "select_eliteRatio");
    get(p.select_tournSize,       "select_tournSize");
    get(p.save_mod,               "save_mod");
    get(p.bestReps,               "bestReps");
    get(p.early_stop_patience,    "early_stop_patience");
    get(p.ann_nInput,             "ann_nInput");
    get(p.ann_nOutput,            "ann_nOutput");
    get(p.ann_initAct,            "ann_initAct");
    get(p.ann_actRange,           "ann_actRange");
    get(p.ann_absWCap,            "ann_absWCap");
    get(p.car_episode_steps,      "car_episode_steps");
}

// Parse a string that is either a file path or an inline JSON object.
inline nlohmann::json parseFileOrInline(const std::string& s) {
    if (!s.empty() && s.front() == '{')
        return nlohmann::json::parse(s);
    std::ifstream f(s);
    if (!f) throw std::runtime_error("Cannot open hyperparameter file: " + s);
    nlohmann::json j;
    f >> j;
    return j;
}

} // namespace detail

// Load base hyperparameters. Accepts a file path or an inline JSON string.
inline Hyperparams loadHyp(const std::string& src) {
    Hyperparams p;
    detail::applyJson(p, detail::parseFileOrInline(src));
    return p;
}

// Merge overrides into an existing Hyperparams.
// Accepts a file path or an inline JSON string (e.g. '{"maxGen":1}').
inline void updateHyp(Hyperparams& p, const std::string& src) {
    detail::applyJson(p, detail::parseFileOrInline(src));
}

} // namespace wann
