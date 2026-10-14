#pragma once

#include "AnnTask.h"

namespace wann {

// WANN (classical ANN) + rl-tools Acrobot, torque range ±1 N·m (Gymnasium Acrobot-v1 scale).
//
// Observation (6 dims): cos θ1, sin θ1, cos θ2, sin θ2, θ1_dot, θ2_dot. The network gets the
// first ann_nInput of them; the velocities are divided by their bounds (4π, 9π).
//
// Action:
//   ann_nOutput = 3  ->  argmax over the outputs: torque = (winner - 1) ∈ {-1, 0, +1}
//   ann_nOutput = 1  ->  continuous torque = clamp(output, -1, 1)
//
// Reward: the environment's reward each step plus, if reward_shaping_scale != 0, the potential-based
// term scale * (h(s') - h(s)) with h = tip height = -cos θ1 - cos(θ1+θ2). Both are reported by
// evalEpisodes() ("shaped" is what evolution maximises). Episode ends at the environment's step
// limit (500) or when the tip is above the pivot.
class AnnAcrobotTask : public AnnTask {
public:
    static constexpr double MAX_TORQUE = 1.0;
    static constexpr double MAX_VEL_1  = 4.0 * 3.14159265358979323846;
    static constexpr double MAX_VEL_2  = 9.0 * 3.14159265358979323846;

    explicit AnnAcrobotTask(const Hyperparams& hyp);

protected:
    EpisodeResult runEpisode(const CompiledNet& net, double w, long long seed,
                             std::ostream* traj) const override;
    const char* trajectoryHeader() const override {
        return "step,cos_th1,sin_th1,cos_th2,sin_th2,dth1,dth2,action,reward";
    }

private:
    double shapingScale_;
};

} // namespace wann
