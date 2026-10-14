#include "../include/wann/AnnDiscMCTask.h"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/mountain_car/operations_cpu.h>

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG    = typename rlt::devices::random::CPU::ENGINE<>;

// Parameters matching Gymnasium MountainCar-v0 (discrete).
struct DiscMCParams {
    static constexpr T min_action         = -1.0;
    static constexpr T max_action         =  1.0;
    static constexpr T min_position       = -1.2;
    static constexpr T max_position       =  0.6;
    static constexpr T max_speed          =  0.07;
    static constexpr T power              =  0.001;  // Gymnasium discrete power
    static constexpr T goal_position      =  0.5;    // Gymnasium discrete goal
    static constexpr T goal_velocity      =  0.0;
    static constexpr T action_cost        =  0.0;    // no action penalty
    static constexpr T goal_reward        =  0.0;    // reward handled here
    static constexpr T initial_position_min = -0.6;
    static constexpr T initial_position_max = -0.4;
};

using MCSpec    = rlt::rl::environments::mountain_car::Specification<T, TI, DiscMCParams>;
using Env       = rlt::rl::environments::MountainCarContinuous<MCSpec>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>>;

namespace {
constexpr double POS_MIN = -1.2, POS_MAX = 0.6;
constexpr double VEL_MAX = 0.07;
} // namespace

namespace wann {

AnnDiscMCTask::AnnDiscMCTask(const Hyperparams& hyp) : AnnTask(hyp, 200), shapingScale_(hyp.reward_shaping_scale)
{
    if (nInput_ < 1 || nInput_ > 2)
        throw std::runtime_error("ann_nInput must be 1 or 2 for mountain car; got " + std::to_string(nInput_));
    if (nOutput_ != N_ACTIONS)
        throw std::runtime_error("ann_nOutput must be 3 (left / none / right) for discrete mountain car; got "
                                 + std::to_string(nOutput_));
}

AnnTask::EpisodeResult AnnDiscMCTask::runEpisode(const CompiledNet& net, double w, long long seed,
                                                 std::ostream* traj) const {
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::mountain_car::Observation<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;
    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    std::vector<double> node(net.nNodes, 0.0), out(nOutput_);
    double in[2];
    double shaped = 0.0, original = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        const double pos = rlt::get(obs_mat, 0, 0);
        const double vel = rlt::get(obs_mat, 0, 1);
        in[0] = scaleSym(pos, POS_MIN, POS_MAX);
        in[1] = scaleSym(vel, -VEL_MAX, VEL_MAX);

        net.forward(w, in, node.data(), out.data());
        const int winner = argmaxFirst(out.data(), N_ACTIONS);
        const double force = static_cast<double>(winner - 1);
        rlt::set(action_mat, 0, 0, force);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double r  = -1.0;
        const double rs = r + shapingScale_ * (std::sin(3.0 * next_state.position) - std::sin(3.0 * pos));
        original += r;
        shaped   += rs;

        // As in the SNN code's trajectory CSV: the action column is the winning index (0, 1, 2),
        // the reward column the shaped step reward.
        if (traj) *traj << step << ',' << pos << ',' << vel << ',' << winner << ',' << rs << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return {shaped, original};
}

} // namespace wann
