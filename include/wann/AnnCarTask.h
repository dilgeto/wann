#pragma once

#include "AnnTask.h"

namespace wann {

// WANN (classical feed-forward ANN) + rl-tools Car (CarTrack variant).
//
// Observation (9 dims, raw from rl-tools):
//   x, y           – car position in track frame (m)
//   mu             – heading angle (rad)
//   vx, vy         – longitudinal / lateral velocity (m/s)
//   omega          – yaw rate (rad/s)
//   lidar_L, lidar_C, lidar_R – normalised distance to track boundary [0, 1]
// The network receives the first ann_nInput of them: x, y, mu, vx, vy, omega divided by their
// bounds (so roughly in [-1, 1]); lidar left in [0, 1].
//
// Action (2 dims): throttle/brake ∈ [-1, 1], steering ∈ [-1, 1]  (ann_nOutput = 2)
// Reward: vx per step. Terminal: car leaves the track, or the step limit (default 1000).
class AnnCarTask : public AnnTask {
public:
    static constexpr double VX_MAX    = 3.0;    // m/s – normalisation bounds
    static constexpr double VY_MAX    = 2.0;
    static constexpr double OMEGA_MAX = 10.0;   // rad/s

    explicit AnnCarTask(const Hyperparams& hyp);

protected:
    EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                             std::ostream* traj) const override;
    const char* trajectoryHeader() const override {
        return "step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,throttle,steering,reward";
    }
};

} // namespace wann
