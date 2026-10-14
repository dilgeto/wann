#include "../include/wann/AnnCarTask.h"
#include "../include/wann/Ind.h"

#include <rl_tools/operations/cpu.h>
// operations_generic (not operations_cpu): init() fills the track from the
// hardcoded default_track instead of loading a BMP from disk.
#include <rl_tools/rl/environments/car/operations_generic.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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

std::vector<double> makeWeightVals(const Hyperparams& hyp) {
    if (hyp.alg_wDist == "standard" && hyp.alg_nVals == 6)
        return {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0};
    std::vector<double> v(hyp.alg_nVals);
    for (int i = 0; i < hyp.alg_nVals; ++i)
        v[i] = hyp.alg_nVals == 1 ? 0.0
             : -hyp.ann_absWCap + 2.0 * hyp.ann_absWCap * i / (hyp.alg_nVals - 1);
    return v;
}
} // namespace

AnnCarTask::AnnCarTask(const Hyperparams& hyp)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , episodeSteps_(hyp.car_episode_steps), weightVals_(makeWeightVals(hyp))
{
    if (nInput_ < 1 || nInput_ > OBS_DIM)
        throw std::runtime_error("ann_nInput must be in [1, 9] for the car task; got "
                                 + std::to_string(nInput_));
    if (nOutput_ != 2)
        throw std::runtime_error("ann_nOutput must be 2 (throttle, steering) for the car task; got "
                                 + std::to_string(nOutput_));
}

// -------------------------------------------------------------------------
// CompiledNet
// -------------------------------------------------------------------------
AnnCarTask::CompiledNet AnnCarTask::compile(const std::vector<double>& wVec,
                                            const std::vector<int>&    aVec,
                                            int nInput, int nOutput)
{
    CompiledNet net;
    const int N = static_cast<int>(std::lround(std::sqrt(static_cast<double>(wVec.size()))));
    net.nNodes = N; net.nInput = nInput; net.nOutput = nOutput;
    net.act = aVec;
    net.rowStart.assign(N + 1, 0);
    // Same visiting order as act(): dst ascending, src ascending, so the
    // floating-point sums are bit-identical.
    for (int dst = 0; dst < N; ++dst) {
        net.rowStart[dst] = static_cast<int>(net.src.size());
        for (int s = 0; s < dst; ++s) {
            double w = wVec[s * N + dst];
            if (w != 0.0 && !std::isnan(w)) {
                net.src.push_back(s);
                net.sign.push_back(w > 0.0 ? 1.0 : -1.0);
            }
        }
    }
    net.rowStart[N] = static_cast<int>(net.src.size());
    return net;
}

void AnnCarTask::CompiledNet::forward(double w, const double* in, double* node, double* out) const {
    node[0] = 1.0;  // bias
    for (int i = 0; i < nInput; ++i) node[1 + i] = in[i];
    for (int dst = nInput + 1; dst < nNodes; ++dst) {
        double raw = 0.0;
        for (int e = rowStart[dst]; e < rowStart[dst + 1]; ++e)
            raw += node[src[e]] * (sign[e] * w);
        node[dst] = applyAct(act[dst], raw);
    }
    for (int i = 0; i < nOutput; ++i) out[i] = node[nNodes - nOutput + i];
}

// -------------------------------------------------------------------------
// Episode
// -------------------------------------------------------------------------
double AnnCarTask::runEpisode(const CompiledNet& net, double w, long long seed, std::ostream* traj) const {
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
        // A diverged network (inf - inf) must not poison the physics.
        double throttle = std::isnan(out[0]) ? 0.0 : std::clamp(out[0], -1.0, 1.0);
        double steering = std::isnan(out[1]) ? 0.0 : std::clamp(out[1], -1.0, 1.0);
        rlt::set(action_mat, 0, 0, throttle);
        rlt::set(action_mat, 0, 1, steering);

        rlt::step(device, env, params, state, action_mat, next_state, rng);
        double reward = rlt::reward(device, env, params, state, action_mat, next_state, rng);
        total += reward;

        if (traj)
            *traj << step << ',' << o[0] << ',' << o[1] << ',' << o[2] << ',' << o[3] << ','
                  << o[4] << ',' << o[5] << ',' << o[6] << ',' << o[7] << ',' << o[8] << ','
                  << throttle << ',' << steering << ',' << reward << '\n';

        state = next_state;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    return total;
}

double AnnCarTask::evaluateWeight(const CompiledNet& net, int wi, int seed) const {
    double total = 0.0;
    for (int rep = 0; rep < nReps_; ++rep)
        total += runEpisode(net, weightVals_[wi], episodeSeed(seed, wi, rep), nullptr);
    return total / static_cast<double>(nReps_);
}

std::vector<double> AnnCarTask::getDistFitness(const std::vector<double>& wVec,
                                               const std::vector<int>&    aVec,
                                               int seed)
{
    CompiledNet net = compile(wVec, aVec, nInput_, nOutput_);
    std::vector<double> rewards(weightVals_.size());
    for (int wi = 0; wi < numWeightVals(); ++wi)
        rewards[wi] = evaluateWeight(net, wi, seed);
    return rewards;
}

std::vector<double> AnnCarTask::evalEpisodes(const std::vector<double>& wVec,
                                             const std::vector<int>&    aVec,
                                             double weight, int nEpisodes, int baseSeed) const
{
    CompiledNet net = compile(wVec, aVec, nInput_, nOutput_);
    std::vector<double> rewards(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i)
        rewards[i] = runEpisode(net, weight, baseSeed + i, nullptr);
    return rewards;
}

void AnnCarTask::exportTrajectory(const std::vector<double>& wVec,
                                  const std::vector<int>&    aVec,
                                  double weight, long long episodeSeed,
                                  const std::string& outFile) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6);
    csv << "step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,throttle,steering,reward\n";
    CompiledNet net = compile(wVec, aVec, nInput_, nOutput_);
    runEpisode(net, weight, episodeSeed, &csv);
}

} // namespace wann
