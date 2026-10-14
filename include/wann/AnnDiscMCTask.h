#pragma once

#include "AnnTask.h"

namespace wann {

// WANN (classical ANN) + rl-tools MountainCar with discrete actions (Gymnasium MountainCar-v0:
// power 0.001, goal at 0.5).
//
// Observation (2 dims): position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]; the network gets them
// mapped to [-1, 1].
// Action (3 discrete): argmax over ann_nOutput = 3 outputs; force = winner - 1
//   (0 = push left, 1 = no push, 2 = push right).
// Reward: -1 per step; with reward_shaping_scale != 0 the training fitness adds
//   scale * (sin(3 x') - sin(3 x)) (potential-based). The environment return is -(steps taken).
// Episode: 200 steps, or earlier when position >= 0.5.
class AnnDiscMCTask : public AnnTask {
public:
    static constexpr int N_ACTIONS = 3;
    explicit AnnDiscMCTask(const Hyperparams& hyp);

protected:
    EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                             std::ostream* traj) const override;
    // action = winning output index (0 left, 1 none, 2 right); reward = shaped step reward
    const char* trajectoryHeader() const override { return "step,position,velocity,action,reward"; }

private:
    double shapingScale_;
};

} // namespace wann
