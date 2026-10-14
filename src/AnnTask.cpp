#include "../include/wann/AnnTask.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace wann {

namespace {
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

AnnTask::AnnTask(const Hyperparams& hyp, int defaultEpisodeSteps)
    : nInput_(hyp.ann_nInput), nOutput_(hyp.ann_nOutput), nReps_(hyp.alg_nReps)
    , episodeSteps_(hyp.episode_steps > 0 ? hyp.episode_steps : defaultEpisodeSteps)
    , weightVals_(makeWeightVals(hyp))
{}

double AnnTask::evaluateWeight(const CompiledNet& net, int wi, int seed) const {
    double total = 0.0;
    for (int rep = 0; rep < nReps_; ++rep)
        total += runEpisode(net, weightVals_[wi], episodeSeed(seed, wi, rep), nullptr).shaped;
    return total / static_cast<double>(nReps_);
}

std::vector<double> AnnTask::getDistFitness(const std::vector<double>& wVec,
                                            const std::vector<int>&    aVec, int seed)
{
    CompiledNet net = compile(wVec, aVec);
    std::vector<double> rewards(weightVals_.size());
    for (int wi = 0; wi < numWeightVals(); ++wi) rewards[wi] = evaluateWeight(net, wi, seed);
    return rewards;
}

std::pair<std::vector<double>, std::vector<double>>
AnnTask::evalEpisodes(const std::vector<double>& wVec, const std::vector<int>& aVec,
                      double weight, int nEpisodes, int baseSeed) const
{
    CompiledNet net = compile(wVec, aVec);
    std::vector<double> shaped(nEpisodes), original(nEpisodes);
    for (int i = 0; i < nEpisodes; ++i) {
        EpisodeResult r = runEpisode(net, weight, baseSeed + i, nullptr);
        shaped[i] = r.shaped; original[i] = r.original;
    }
    return {shaped, original};
}

void AnnTask::exportTrajectory(const std::vector<double>& wVec, const std::vector<int>& aVec,
                               double weight, long long episodeSeed, const std::string& outFile) const
{
    exportTrajectory(compile(wVec, aVec), weight, episodeSeed, outFile);
}

void AnnTask::exportTrajectory(const CompiledNet& net, double weight, long long episodeSeed,
                               const std::string& outFile) const
{
    std::ofstream csv(outFile);
    if (!csv) throw std::runtime_error("Cannot write: " + outFile);
    csv << std::fixed << std::setprecision(6) << trajectoryHeader() << '\n';
    runEpisode(net, weight, episodeSeed, &csv);
}

} // namespace wann
