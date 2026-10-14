#pragma once

#include "CompiledNet.h"
#include "Hyperparams.h"
#include "Task.h"

#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace wann {

// Linear map of v from [lo, hi] to [-1, 1], clamped. ANN inputs use the same ranges the
// SNN code uses for its encoders (there mapped to [0, 1]).
inline double scaleSym(double v, double lo, double hi) {
    return std::clamp(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0);
}

// Index of the largest of out[0..n-1] (first one on ties, like np.argmax; NaN counts as lowest).
inline int argmaxFirst(const double* out, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (std::isnan(out[best]) ? !std::isnan(out[i]) : out[i] > out[best]) best = i;
    return best;
}

// Replace NaN (a diverged network, inf - inf) by 0 and clamp to [-1, 1].
inline double actionClamp(double v) { return std::isnan(v) ? 0.0 : std::clamp(v, -1.0, 1.0); }

// -------------------------------------------------------------------------
// AnnTask – common part of the environment tasks: a classical feed-forward ANN
// evaluated with a set of shared weight values, averaged over alg_nReps episodes.
// A subclass supplies one thing: runEpisode().
//
// Shared weights: alg_wDist == "standard" and alg_nVals == 6 -> {-2, -1, -0.5, 0.5, 1, 2}
// (as in the original WANN), otherwise linspace(-ann_absWCap, ann_absWCap, alg_nVals).
// -------------------------------------------------------------------------
class AnnTask : public ITask {
public:
    struct EpisodeResult {
        double shaped;    // what the evolution maximises (== original when there is no shaping)
        double original;  // the environment's own return
    };

    AnnTask(const Hyperparams& hyp, int defaultEpisodeSteps);
    ~AnnTask() override = default;

    void setEpisodeSteps(int n) { episodeSteps_ = n; }
    int  episodeSteps() const { return episodeSteps_; }
    const std::vector<double>& weightVals() const { return weightVals_; }
    int  numWeightVals() const override { return static_cast<int>(weightVals_.size()); }

    CompiledNet compile(const std::vector<double>& wVec, const std::vector<int>& aVec) const {
        return CompiledNet::compile(wVec, aVec, nInput_, nOutput_);
    }

    // Mean (shaped) reward over alg_nReps episodes for one (network, weight-value) pair.
    // Thread-safe: this is the unit the training loop dispatches on.
    double evaluateWeight(const CompiledNet& net, int wi, int seed = -1) const;

    std::vector<double> getDistFitness(const std::vector<double>& wVec,
                                       const std::vector<int>&    aVec,
                                       int seed = -1) override;

    // nEpisodes episodes (seeds baseSeed .. baseSeed+nEpisodes-1) with a fixed weight.
    std::pair<std::vector<double>, std::vector<double>>   // {shaped, original}
    evalEpisodes(const std::vector<double>& wVec, const std::vector<int>& aVec,
                 double weight, int nEpisodes, int baseSeed) const;

    // Run one episode and write its trajectory CSV (header from trajectoryHeader()).
    // episodeSeed is the exact seed of the episode.
    void exportTrajectory(const std::vector<double>& wVec, const std::vector<int>& aVec,
                          double weight, long long episodeSeed, const std::string& outFile) const;

    // Same, for an already compiled network.
    void exportTrajectory(const CompiledNet& net, double weight, long long episodeSeed,
                          const std::string& outFile) const;

    // Seed used by evaluateWeight() for repetition `rep` of weight index `wi`
    // in an evaluation with base seed `seed`.
    static long long episodeSeed(int seed, int wi, int rep) {
        return static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
    }

protected:
    // Run one episode with shared weight w; if traj != nullptr append one CSV row per step.
    virtual EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                                     std::ostream* traj) const = 0;
    virtual const char* trajectoryHeader() const = 0;

    int nInput_, nOutput_, nReps_;
    int episodeSteps_;
    std::vector<double> weightVals_;
};

} // namespace wann
