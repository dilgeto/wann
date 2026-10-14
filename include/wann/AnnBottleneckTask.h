#pragma once

#include "AnnTask.h"

namespace wann {

// WANN (classical ANN) + rl-tools multi-agent Bottleneck, with the Parameters of rl-tools' own
// benchmark config for this environment (N_AGENTS = 5, LIDAR_FOV = 90°, BOTTLENECK_WIDTH = 5,
// AGENT_MAX_SPEED = 4, accelerations 20; 20x20 arena, 200 steps).
//
// One network with shared weights controls all agents (parameter sharing, decentralised
// execution — the scheme rl-tools' PPO uses for this environment). A feed-forward ANN has no
// internal state, so the same network is simply evaluated once per agent per step.
//
// Per-agent observation (12 dims): position.x, position.y, orientation, velocity.x, velocity.y,
//   angular_velocity, alive ∈ {-1, 1}, lidar[0..4]; each mapped to [-1, 1] over a fixed range.
// Per-agent action (2 dims, ann_nOutput = 2): linear acceleration, angular acceleration,
//   clamp(output, -1, 1).
//
// rlt::reward() returns one scalar summed over agents (with a one-time death penalty and a
// cross-the-bottleneck bonus). Training fitness adds, if enabled:
//   reward_shaping_scale * (phi(s') - phi(s)),  phi = (sum of agents' x position) / arena width
//   - bottleneck_reverse_penalty_scale * sum over living agents of max(0, -velocity·heading)
// (the second one is not potential-based: it changes what is optimal). The environment's own
// return is reported as "original". Episode ends at step 200 or when every agent is dead.
class AnnBottleneckTask : public AnnTask {
public:
    static constexpr int N_AGENTS             = 5;
    static constexpr int PER_AGENT_OBS_DIM    = 12;   // 7 + LIDAR_RESOLUTION(5)
    static constexpr int PER_AGENT_ACTION_DIM = 2;
    explicit AnnBottleneckTask(const Hyperparams& hyp);

protected:
    EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                             std::ostream* traj) const override;
    const char* trajectoryHeader() const override { return header_.c_str(); }

private:
    double shapingScale_, reversePenaltyScale_;
    std::string header_;
};

} // namespace wann
