#pragma once

#include "Hyperparams.h"
#include "Task.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace wann {

// ITask implementation: WANN (classical feed-forward ANN) + rl-tools Car (CarTrack variant).
//
// Observation (9 dims, raw from rl-tools):
//   x, y           – car position in track frame (m)
//   mu             – heading angle (rad)
//   vx, vy         – longitudinal / lateral velocity (m/s)
//   omega          – yaw rate (rad/s)
//   lidar_L, lidar_C, lidar_R – normalised distance to track boundary [0, 1]
// The network receives the first ann_nInput of them, scaled to [-1, 1]
// (x, y, mu, vx, vy, omega) or left in [0, 1] (lidar).
//
// Action (2 dims): throttle/brake ∈ [-1, 1], steering ∈ [-1, 1]
// Reward: vx per step. Terminal: car leaves the track or EPISODE steps reached.
//
// Shared weights: every connection carries the same value w, with
// alg_wDist == "standard" and alg_nVals == 6 -> {-2, -1, -0.5, 0.5, 1, 2}
// (as in the original WANN), otherwise linspace(-ann_absWCap, ann_absWCap, alg_nVals).
class AnnCarTask : public ITask {
public:
    static constexpr double VX_MAX    = 3.0;    // m/s – normalisation bounds
    static constexpr double VY_MAX    = 2.0;
    static constexpr double OMEGA_MAX = 10.0;   // rad/s

    // Feed-forward network compiled from an expressed individual: incoming
    // edges per node in CSR form (only src < dst, the topological order),
    // so evaluation does no allocation and visits only existing connections.
    // Equivalent to wann::act(setWeights(wVec, w), ...) — see tests/.
    struct CompiledNet {
        int nNodes = 0, nInput = 0, nOutput = 0;
        std::vector<int>    rowStart;  // [nNodes+1] into src/sign, indexed by dst
        std::vector<int>    src;
        std::vector<double> sign;      // +1 / -1 (polarity of the genome weight)
        std::vector<int>    act;       // activation id per node

        // node: scratch buffer of nNodes doubles. Writes nOutput values to out.
        void forward(double w, const double* in, double* node, double* out) const;
    };

    explicit AnnCarTask(const Hyperparams& hyp);

    void setEpisodeSteps(int n) { episodeSteps_ = n; }

    const std::vector<double>& weightVals() const { return weightVals_; }

    static CompiledNet compile(const std::vector<double>& wVec,
                               const std::vector<int>&    aVec,
                               int nInput, int nOutput);

    // Mean reward over alg_nReps episodes for one (network, weight-value) pair.
    // Thread-safe: this is the unit evalPop() dispatches on.
    double evaluateWeight(const CompiledNet& net, int wi, int seed = -1) const;

    std::vector<double> getDistFitness(const std::vector<double>& wVec,
                                       const std::vector<int>&    aVec,
                                       int seed = -1) override;

    int numWeightVals() const override { return static_cast<int>(weightVals_.size()); }

    // Reward of nEpisodes episodes (seeds baseSeed .. baseSeed+nEpisodes-1) with a fixed weight.
    std::vector<double> evalEpisodes(const std::vector<double>& wVec,
                                     const std::vector<int>&    aVec,
                                     double weight, int nEpisodes, int baseSeed) const;

    // Run one episode and write its trajectory CSV. episodeSeed is the exact seed of the episode.
    // Columns: step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,throttle,steering,reward
    void exportTrajectory(const std::vector<double>& wVec,
                          const std::vector<int>&    aVec,
                          double weight, long long episodeSeed,
                          const std::string& outFile) const;

    // Seed used by evaluateWeight() for repetition `rep` of weight index `wi`
    // in an evaluation with base seed `seed`.
    static long long episodeSeed(int seed, int wi, int rep) {
        return static_cast<long long>(seed < 0 ? 0 : seed) * 10000 + wi * 100 + rep;
    }

private:
    int nInput_, nOutput_, nReps_;
    int episodeSteps_;
    std::vector<double> weightVals_;

    // Runs one episode; if traj != nullptr appends a CSV row per step.
    double runEpisode(const CompiledNet& net, double w, long long seed, std::ostream* traj) const;
};

} // namespace wann
