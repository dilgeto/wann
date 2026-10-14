#pragma once

#include "AnnTask.h"

namespace wann {

// WANN (classical ANN) + BipedalWalker(Hardcore)-v3: the Box2D-free C++ port living in
// snn-simulator (include/rl/environments/bipedal_walker/). Hardcore by default;
// bipedal_hardcore = false gives the plain BipedalWalker.
//
// Observation (24 dims, Gymnasium order):
//   0..3   hull angle, angular velocity, vx, vy (already scaled by the env)
//   4..8   leg 1: hip angle, hip speed, knee angle+1, knee speed, contact
//   9..13  leg 2: same
//   14..23 lidar, 10 rays, in [0,1]
// Each channel is mapped to [-1, 1] over a fixed range (see OBS_LIMITS in the .cpp) and clamped.
// The network gets the first ann_nInput of them (24 for the full observation).
//
// Action (4 dims, ann_nOutput = 4): hip/knee torques of each leg = clamp(output, -1, 1).
//
// Reward: Gymnasium's (130*dx/SCALE - 5*|Δhull angle| - torque cost; -100 on a fall). The terrain
// is resampled every episode from the episode seed, so fitness measures generalisation across
// terrains, not memorising one course. Step limit: the environment's (2000 hardcore / 1600 normal)
// unless episode_steps is set (which can only shorten it).
class AnnBipedalTask : public AnnTask {
public:
    static constexpr int N_OBS     = 24;
    static constexpr int N_ACTIONS = 4;
    explicit AnnBipedalTask(const Hyperparams& hyp);

protected:
    EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                             std::ostream* traj) const override;
    const char* trajectoryHeader() const override { return header_.c_str(); }

private:
    template <typename ENV>
    EpisodeResult runEpisodeImpl(const CompiledNet& net, double w, long long seed,
                                 std::ostream* traj) const;  // defined (and instantiated) in the .cpp
    bool hardcore_;
    int  maxSteps_;      // 0 = the environment's own limit
    std::string header_;
};

} // namespace wann
