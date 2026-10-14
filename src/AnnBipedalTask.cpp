#include "../include/wann/AnnBipedalTask.h"

#include <rl_tools/operations/cpu.h>
// Box2D-free BipedalWalker port, header-only, from snn-simulator/include/.
#include <rl/environments/bipedal_walker/operations_generic.h>

#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>

namespace rlt  = rl_tools;
namespace phys = ::bipedal_walker::physics;

using EnvT   = float;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG    = typename rlt::devices::random::CPU::ENGINE<>;

using EnvSpec     = rlt::rl::environments::bipedal_walker::Specification<EnvT, TI>;
using NormalEnv   = rlt::rl::environments::BipedalWalker<EnvSpec>;
using HardcoreEnv = rlt::rl::environments::BipedalWalkerHardcore<EnvSpec>;

using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<EnvT, TI, 1, 24, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<EnvT, TI, 1, 4, false>>;

static_assert(NormalEnv::Observation::DIM == wann::AnnBipedalTask::N_OBS, "BipedalWalker observation must be 24-dim");
static_assert(NormalEnv::ACTION_DIM == wann::AnnBipedalTask::N_ACTIONS, "BipedalWalker action must be 4-dim");

namespace {
// Range per observation channel (same ranges as the SNN code's encoders). The env scales most
// values but gives no hard bounds (Gymnasium's observation space is ±inf):
//   hull angle / angular velocity / vx / vy: ±1 covers normal walking; a hull tilted past ~1 rad
//     is about to touch the ground (episode over).
//   joint angles: the joint limits (hip [-0.8, 1.1], knee+1 [-0.6, 0.9]) plus a small margin.
//   joint speeds (already divided by the motor's nominal speed): ±1.5.
//   contacts {0,1}, lidar [0,1]: already normalised.
// Values outside the range are clamped.
constexpr std::array<std::pair<double, double>, 24> OBS_LIMITS = {{
    {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0},       // hull
    {-0.9, 1.2}, {-1.5, 1.5}, {-0.7, 1.0}, {-1.5, 1.5}, {0.0, 1.0},  // leg 1
    {-0.9, 1.2}, {-1.5, 1.5}, {-0.7, 1.0}, {-1.5, 1.5}, {0.0, 1.0},  // leg 2
    {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},      // lidar
    {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},
}};

std::string makeHeader() {
    std::string h = "step,hull_x";
    for (int i = 0; i < 24; ++i) h += ",obs" + std::to_string(i);
    for (int k = 0; k < 4; ++k)  h += ",a" + std::to_string(k);
    return h + ",reward";
}
} // namespace

namespace wann {

AnnBipedalTask::AnnBipedalTask(const Hyperparams& hyp)
    : AnnTask(hyp, 0), hardcore_(hyp.bipedal_hardcore), maxSteps_(hyp.episode_steps), header_(makeHeader())
{
    if (nInput_ < 1 || nInput_ > N_OBS)
        throw std::runtime_error("ann_nInput must be in [1, 24] for bipedal walker; got " + std::to_string(nInput_));
    if (nOutput_ != N_ACTIONS)
        throw std::runtime_error("ann_nOutput must be 4 (hip/knee torques) for bipedal walker; got "
                                 + std::to_string(nOutput_));
}

AnnTask::EpisodeResult AnnBipedalTask::runEpisode(const CompiledNet& net, double w, long long seed,
                                                  std::ostream* traj) const {
    return hardcore_ ? runEpisodeImpl<HardcoreEnv>(net, w, seed, traj)
                     : runEpisodeImpl<NormalEnv>(net, w, seed, traj);
}

template <typename ENV>
AnnTask::EpisodeResult AnnBipedalTask::runEpisodeImpl(const CompiledNet& net, double w, long long seed,
                                                      std::ostream* traj) const {
    DEVICE device;
    ENV env;
    typename ENV::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));

    // Terrain (normal or hardcore, picked by overload on ENV) and the initial push on the hull
    // are both drawn from the episode seed.
    rlt::sample_initial_parameters(device, env, params, rng);

    typename ENV::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;
    typename ENV::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    const int stepLimit = (maxSteps_ > 0)
        ? std::min<int>(maxSteps_, static_cast<int>(ENV::EPISODE_STEP_LIMIT))
        : static_cast<int>(ENV::EPISODE_STEP_LIMIT);

    std::vector<double> node(net.nNodes, 0.0);
    double obs[N_OBS], in[N_OBS], out[N_ACTIONS], action[N_ACTIONS];
    double total = 0.0;

    for (int step = 0; step < stepLimit; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        for (int i = 0; i < N_OBS; ++i) {
            obs[i] = static_cast<double>(rlt::get(obs_mat, 0, i));
            in[i]  = scaleSym(obs[i], OBS_LIMITS[i].first, OBS_LIMITS[i].second);
        }

        net.forward(w, in, node.data(), out);
        for (int k = 0; k < N_ACTIONS; ++k) {
            action[k] = actionClamp(out[k]);
            rlt::set(action_mat, 0, k, static_cast<EnvT>(action[k]));
        }

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double reward = static_cast<double>(
            rlt::reward(device, env, params, state, action_mat, next_state, rng));
        total += reward;

        if (traj) {
            *traj << step << ',' << phys::body_origin(params.geometry, state.bodies, 0).x;
            for (int i = 0; i < N_OBS; ++i) *traj << ',' << obs[i];
            for (int k = 0; k < N_ACTIONS; ++k) *traj << ',' << action[k];
            *traj << ',' << reward << '\n';
        }

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return {total, total};
}

template AnnTask::EpisodeResult AnnBipedalTask::runEpisodeImpl<NormalEnv>(const CompiledNet&, double, long long, std::ostream*) const;
template AnnTask::EpisodeResult AnnBipedalTask::runEpisodeImpl<HardcoreEnv>(const CompiledNet&, double, long long, std::ostream*) const;

} // namespace wann
