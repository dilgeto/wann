#include "../include/wann/AnnCarTask.h"

#include <rl_tools/operations/cpu.h>
// operations_generic (not operations_cpu): init() fills the track from the
// hardcoded default_track instead of loading a BMP from disk.
#include <rl_tools/rl/environments/car/operations_generic.h>

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace rlt = rl_tools;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;

// Track: 100×100 cells, 50 mm per cell -> 5 m × 5 m arena (same as wann-cpp).
using CarSpec   = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 50>;
using Env       = rlt::rl::environments::CarTrack<CarSpec>;
using RNG       = typename rlt::devices::random::CPU::ENGINE<>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 9, false>>;
using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>>;

namespace wann {

namespace {
constexpr double BOUND = CarSpec::TRACK_SCALE * 100 / 2.0;  // ±2.5 m
constexpr int    OBS_DIM = 9;
} // namespace

AnnCarTask::AnnCarTask(const Hyperparams& hyp) : AnnTask(hyp, 1000)
{
    if (nInput_ < 1 || nInput_ > OBS_DIM)
        throw std::runtime_error("ann_nInput must be in [1, 9] for the car task; got "
                                 + std::to_string(nInput_));
    if (nOutput_ != 2)
        throw std::runtime_error("ann_nOutput must be 2 (throttle, steering) for the car task; got "
                                 + std::to_string(nOutput_));
}

AnnTask::EpisodeResult AnnCarTask::runEpisode(const CompiledNet& net, double w, long long seed,
                                              std::ostream* traj) const {
    DEVICE device;
    Env env;
    Env::Parameters params;
    RNG rng;
    rlt::init(device, rng, static_cast<typename DEVICE::index_t>(seed));
    rlt::init(device, env);
    rlt::initial_parameters(device, env, params);

    rlt::rl::environments::car::ObservationCarTrack<TI> obs_type;
    ObsMatrix obs_mat;
    ActMatrix action_mat;
    Env::State state, next_state;
    rlt::sample_initial_state(device, env, params, state, rng);

    std::vector<double> node(net.nNodes, 0.0);
    double in[OBS_DIM], out[2];
    double total = 0.0;

    for (int step = 0; step < episodeSteps_; ++step) {
        rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
        double o[OBS_DIM];
        for (int i = 0; i < OBS_DIM; ++i) o[i] = rlt::get(obs_mat, 0, i);

        in[0] = o[0] / BOUND;      in[1] = o[1] / BOUND;      in[2] = o[2] / M_PI;
        in[3] = o[3] / VX_MAX;     in[4] = o[4] / VY_MAX;     in[5] = o[5] / OMEGA_MAX;
        in[6] = o[6];              in[7] = o[7];              in[8] = o[8];

        net.forward(w, in, node.data(), out);
        const double throttle = actionClamp(out[0]);
        const double steering = actionClamp(out[1]);
        rlt::set(action_mat, 0, 0, throttle);
        rlt::set(action_mat, 0, 1, steering);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        const double reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        total += reward;

        if (traj)
            *traj << step << ',' << o[0] << ',' << o[1] << ',' << o[2] << ',' << o[3] << ','
                  << o[4] << ',' << o[5] << ',' << o[6] << ',' << o[7] << ',' << o[8] << ','
                  << throttle << ',' << steering << ',' << reward << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return {total, total};
}

} // namespace wann
