#include "../include/wann/AnnBottleneckTask.h"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>

namespace rlt = rl_tools;

using T          = double;
using TI         = size_t;
using DEVICE     = rlt::devices::DefaultCPU;
using RNG        = typename rlt::devices::random::CPU::ENGINE<>;
using TypePolicy = rlt::numeric_types::Policy<T>;

// Same Parameters as rl-tools' own benchmark for this environment
// (rl::zoo::bottleneck_v0::ENVIRONMENT_FACTORY::ENVIRONMENT_PARAMETERS in
// src/rl/zoo/bottleneck-v0/environment.h). Fields not listed here (LIDAR_RESOLUTION, arena size,
// EPISODE_STEP_LIMIT, ...) inherit DefaultParameters unchanged.
template <typename TT, typename TTI>
struct BenchmarkParameters : rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<TT, TTI> {
    static constexpr TTI N_AGENTS                     = 5;
    static constexpr TT   LIDAR_FOV                   = rlt::math::PI<TT> * 90 / 180;
    static constexpr TT   BOTTLENECK_WIDTH             = 5;
    static constexpr TT   AGENT_MAX_SPEED              = 4;
    static constexpr TT   AGENT_MAX_ANGULAR_VELOCITY   = 4;
    static constexpr TT   AGENT_MAX_ACCELERATION       = 20;
    static constexpr TT   AGENT_MAX_ANGULAR_ACCELERATION = 20;
};

using EnvSpec = rlt::rl::environments::multi_agent::bottleneck::Specification<
    TypePolicy, TI, BenchmarkParameters<T, TI>>;
using Env     = rlt::rl::environments::multi_agent::Bottleneck<EnvSpec>;

using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::Observation::DIM, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, Env::ACTION_DIM, false>>;

static_assert(Env::PARAMETERS::N_AGENTS == wann::AnnBottleneckTask::N_AGENTS,
              "BenchmarkParameters::N_AGENTS must match AnnBottleneckTask::N_AGENTS");
static_assert(Env::Observation::DIM ==
                  wann::AnnBottleneckTask::N_AGENTS * wann::AnnBottleneckTask::PER_AGENT_OBS_DIM,
              "Bottleneck per-agent observation must be 12-dim (7 + LIDAR_RESOLUTION)");
static_assert(Env::ACTION_DIM ==
                  wann::AnnBottleneckTask::N_AGENTS * wann::AnnBottleneckTask::PER_AGENT_ACTION_DIM,
              "Bottleneck per-agent action must be 2-dim");

namespace {
constexpr int N_AGENTS = wann::AnnBottleneckTask::N_AGENTS;
constexpr int OBS      = wann::AnnBottleneckTask::PER_AGENT_OBS_DIM;
constexpr int ACT      = wann::AnnBottleneckTask::PER_AGENT_ACTION_DIM;

// Range per per-agent observation channel (ARENA 20x20, AGENT_MAX_SPEED = AGENT_MAX_ANGULAR_VELOCITY = 4).
// Order matches observe() in bottleneck/operations_generic.h.
//
// Lidar: update_lidar() declares LIDAR_RANGE but never uses it to clamp the ray or the returned
// intersection — the arena's bounding rectangle always intersects, so the real upper bound on a
// reported distance is the arena's diagonal (sqrt(20^2+20^2) ≈ 28.28), not the nominal 3.
constexpr double LIDAR_MAX_DIST = 28.284271247461902;
constexpr std::array<std::pair<double, double>, 12> OBS_LIMITS = {{
    {0.0, 20.0}, {0.0, 20.0},                 // position.x, position.y
    {-3.14159265358979323846, 3.14159265358979323846},  // orientation
    {-4.0, 4.0}, {-4.0, 4.0},                 // velocity.x, velocity.y
    {-4.0, 4.0},                              // angular_velocity
    {-1.0, 1.0},                              // alive flag
    {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST},
    {0.0, LIDAR_MAX_DIST}, {0.0, LIDAR_MAX_DIST},  // lidar x5
}};

// Potential-based shaping: phi(s) = (sum of agents' x position) / ARENA_WIDTH. Dead agents freeze
// in place, so phi is continuous across a death transition.
double phi(const Env::State& s) {
    double sum = 0.0;
    for (int i = 0; i < N_AGENTS; ++i) sum += s.agent_states[i].position[0];
    return sum / Env::PARAMETERS::ARENA_WIDTH;
}

// Per-step penalty for driving opposite to one's own heading: velocity·heading is positive when
// moving the way you face; only the negative part is penalised, scaled by how fast. Gated on the
// agent being alive before the step (a dead agent's velocity stays frozen in next_state).
double reversePenalty(const Env::State& state, const Env::State& next_state) {
    double total = 0.0;
    for (int i = 0; i < N_AGENTS; ++i) {
        if (state.agent_states[i].dead) continue;
        const auto& a = next_state.agent_states[i];
        const double forward = a.velocity[0] * std::cos(a.orientation)
                              + a.velocity[1] * std::sin(a.orientation);
        total += std::max(0.0, -forward);
    }
    return total;
}

std::string makeHeader() {
    std::string h = "step";
    for (int a = 0; a < N_AGENTS; ++a) {
        for (int i = 0; i < OBS; ++i) h += ",agent" + std::to_string(a) + "_obs" + std::to_string(i);
        for (int k = 0; k < ACT; ++k) h += ",agent" + std::to_string(a) + "_a" + std::to_string(k);
    }
    return h + ",reward";
}
} // namespace

namespace wann {

AnnBottleneckTask::AnnBottleneckTask(const Hyperparams& hyp)
    : AnnTask(hyp, static_cast<int>(Env::EPISODE_STEP_LIMIT))
    , shapingScale_(hyp.reward_shaping_scale)
    , reversePenaltyScale_(hyp.bottleneck_reverse_penalty_scale)
    , header_(makeHeader())
{
    if (nInput_ < 1 || nInput_ > PER_AGENT_OBS_DIM)
        throw std::runtime_error("ann_nInput must be in [1, 12] for bottleneck; got " + std::to_string(nInput_));
    if (nOutput_ != PER_AGENT_ACTION_DIM)
        throw std::runtime_error("ann_nOutput must be 2 (linear, angular acceleration) for bottleneck; got "
                                 + std::to_string(nOutput_));
    if (episodeSteps_ > static_cast<int>(Env::EPISODE_STEP_LIMIT))
        episodeSteps_ = static_cast<int>(Env::EPISODE_STEP_LIMIT);
}

AnnTask::EpisodeResult AnnBottleneckTask::runEpisode(const CompiledNet& net, double w, long long seed,
                                                     std::ostream* traj) const {
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
    rlt::initial_parameters(device, env, params);

    typename Env::Observation obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;
    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    std::vector<double> node(net.nNodes, 0.0);
    double obs[OBS], in[OBS], out[ACT];
    double shaped = 0.0, original = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        if (traj) *traj << step;

        for (int agent = 0; agent < N_AGENTS; ++agent) {
            for (int i = 0; i < OBS; ++i) {
                obs[i] = static_cast<double>(rlt::get(obs_mat, 0, agent * OBS + i));
                in[i]  = scaleSym(obs[i], OBS_LIMITS[i].first, OBS_LIMITS[i].second);
            }
            net.forward(w, in, node.data(), out);
            for (int k = 0; k < ACT; ++k) {
                const double a = actionClamp(out[k]);
                rlt::set(action_mat, 0, agent * ACT + k, static_cast<T>(a));
                out[k] = a;
            }
            if (traj) {
                for (int i = 0; i < OBS; ++i) *traj << ',' << obs[i];
                for (int k = 0; k < ACT; ++k) *traj << ',' << out[k];
            }
        }

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double reward = static_cast<double>(
            rlt::reward(device, env, params, state, action_mat, next_state, rng));
        original += reward;
        shaped   += reward;
        if (shapingScale_ != 0.0)        shaped += shapingScale_ * (phi(next_state) - phi(state));
        if (reversePenaltyScale_ != 0.0) shaped -= reversePenaltyScale_ * reversePenalty(state, next_state);

        if (traj) *traj << ',' << reward << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return {shaped, original};
}

} // namespace wann
