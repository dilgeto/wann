#include "../include/wann/CompiledNet.h"
#include "../include/wann/Ind.h"

#include <cmath>

namespace wann {

CompiledNet CompiledNet::compile(const std::vector<double>& wVec,
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

void CompiledNet::forward(double w, const double* in, double* node, double* out) const {
    if (tape) {
        const std::vector<double>* row = tapePos < tape->size() ? &(*tape)[tapePos] : nullptr;
        ++tapePos;
        for (int i = 0; i < nOutput; ++i) out[i] = row ? (*row)[i] : 0.0;
        return;
    }
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

} // namespace wann
