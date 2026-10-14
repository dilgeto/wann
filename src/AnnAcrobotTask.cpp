#include "../include/wann/AnnAcrobotTask.h"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;

// Torque range ±1 N·m to match Gymnasium Acrobot-v1 (default rl-tools uses ±5 N·m)
struct GymAcrobotParams : rlt::rl::environments::acrobot::DefaultParameters<T> {
    static constexpr T MIN_TORQUE = -1.0;
    static constexpr T MAX_TORQUE = +1.0;
};
using AcrobotSpec = rlt::rl::environments::acrobot::Specification<T, TI, GymAcrobotParams>;
using Env         = rlt::rl::environments::Acrobot<AcrobotSpec>;
using RNG         = typename rlt::devices::random::CPU::ENGINE<>;
using ObsMatrix   = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 6, false>>;
using ActMatrix   = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;

namespace wann {

AnnAcrobotTask::AnnAcrobotTask(const Hyperparams& hyp)
    : AnnTask(hyp, static_cast<int>(Env::EPISODE_STEP_LIMIT)), shapingScale_(hyp.reward_shaping_scale)
{
    if (nInput_ < 1 || nInput_ > 6)
        throw std::runtime_error("ann_nInput must be in [1, 6] for acrobot; got " + std::to_string(nInput_));
    if (nOutput_ != 1 && nOutput_ != 3)
        throw std::runtime_error("ann_nOutput must be 1 (continuous torque) or 3 (argmax) for acrobot; got "
                                 + std::to_string(nOutput_));
    if (episodeSteps_ > static_cast<int>(Env::EPISODE_STEP_LIMIT))
        episodeSteps_ = static_cast<int>(Env::EPISODE_STEP_LIMIT);
}

AnnTask::EpisodeResult AnnAcrobotTask::runEpisode(const CompiledNet& net, double w, long long seed,
                                                  std::ostream* traj) const {
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::acrobot::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;
    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    std::vector<double> node(net.nNodes, 0.0), out(nOutput_);
    double in[6];
    double shaped = 0.0, original = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        double o[6];
        for (int i = 0; i < 6; ++i) o[i] = rlt::get(obs_mat, 0, i);
        in[0] = o[0]; in[1] = o[1]; in[2] = o[2]; in[3] = o[3];            // cos/sin: already in [-1, 1]
        in[4] = std::clamp(o[4] / MAX_VEL_1, -1.0, 1.0);
        in[5] = std::clamp(o[5] / MAX_VEL_2, -1.0, 1.0);

        net.forward(w, in, node.data(), out.data());
        const double torque = (nOutput_ == 3) ? static_cast<double>(argmaxFirst(out.data(), 3) - 1)
                                              : actionClamp(out[0]);
        rlt::set(action_mat, 0, 0, torque);  // already in [-1, 1] (MAX_TORQUE = 1)

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double r = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        double rs = r;
        if (shapingScale_ != 0.0)
            rs += shapingScale_ * (
                -std::cos(next_state.theta_1) - std::cos(next_state.theta_1 + next_state.theta_2)
                + std::cos(state.theta_1)     + std::cos(state.theta_1 + state.theta_2));
        original += r;
        shaped   += rs;

        // The reward column holds the shaped step reward, as in the SNN code's trajectory CSV.
        if (traj)
            *traj << step << ',' << o[0] << ',' << o[1] << ',' << o[2] << ',' << o[3] << ','
                  << o[4] << ',' << o[5] << ',' << torque << ',' << rs << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return {shaped, original};
}

} // namespace wann
