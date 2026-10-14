#pragma once
#include <cstddef>
#include <vector>

namespace wann {

// Feed-forward network compiled from an expressed individual (wVec/aVec): incoming
// edges per node in CSR form (only src < dst, the topological order), so evaluation
// does no allocation and visits only existing connections. Equivalent to
// wann::act(setWeights(wVec, w), ...) — see tests/.
//
// Node layout (as produced by Ind::express): [bias | inputs | hidden... | outputs].
struct CompiledNet {
    int nNodes = 0, nInput = 0, nOutput = 0;
    std::vector<int>    rowStart;  // [nNodes+1] into src/sign, indexed by dst
    std::vector<int>    src;
    std::vector<double> sign;      // +1 / -1 (polarity of the genome weight)
    std::vector<int>    act;       // activation id per node

    // Test hook (tests/env_replay.h): when set, forward() ignores the network and replays
    // recorded outputs, one row per call (zeros once exhausted). Used to drive an environment
    // with the actions another controller took, to compare environments.
    const std::vector<std::vector<double>>* tape = nullptr;
    mutable std::size_t tapePos = 0;

    static CompiledNet compile(const std::vector<double>& wVec,
                               const std::vector<int>&    aVec,
                               int nInput, int nOutput);

    // node: scratch buffer of nNodes doubles. Writes nOutput values to out.
    // The shared weight w multiplies every connection.
    void forward(double w, const double* in, double* node, double* out) const;
};

} // namespace wann
